"""Load challenge TSVs into Polars frames with an enforced schema.

Layer: data
See AGENTS.md §8.2 (Schema and nulls), §8.3 (ID integrity)

This is the domain layer. Byte-level file access lives in
:mod:`team_diamond.data.storage`; locating the dataset lives in
:mod:`team_diamond.data.paths`. Neither of those knows what a "ground truth"
is, and this module knows nothing about S3.

Two decisions worth stating up front
------------------------------------
**IDs are strings, always.** ``entity_id`` and ``source1_entity_id`` are cast to
``pl.Utf8`` on read. They are opaque identifiers: never cast to an integer,
never strip leading zeros, never assume S2 and S3 share a namespace
(AGENTS.md §8.3).

**Ground truth is read with the empty-match branch.** ``matched_entity_ids`` is
``""`` for singletons, and a CSV reader will happily hand back ``null`` for an
empty field depending on version. Those two cases mean the same thing and both
mean *zero* matches. An earlier EDA script omitted this branch, counted ``""``
as a one-element list, and produced a contradictory ``{0: 5594}`` bucket as a
result. That bug is why this loader exists and why it is tested.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import polars as pl

from team_diamond.data.paths import DatasetPaths
from team_diamond.data.storage import ParquetCache, is_s3_path

__all__ = [
    "Split",
    "load_entities",
    "load_ground_truth",
    "load_split",
    "ground_truth_frame",
    "match_cardinality",
    "cache_for_path",
]

Split = Literal["train", "test"]

_GT_COLUMNS: tuple[str, ...] = ("source1_entity_id", "matched_entity_ids")


def _split_of(filename: str) -> str:
    """Infer the split from a dataset file name."""
    head = filename.split("_", 1)[0]
    if head not in ("train", "test"):
        raise ValueError(
            f"cannot infer a split from {filename!r}; expected a 'train_' or "
            f"'test_' prefix"
        )
    return head


def cache_for_path(path: str | os.PathLike[str]) -> ParquetCache:
    """Build a Parquet cache that can serve the dataset file at ``path``.

    The cache is keyed on the file *name*, so this works identically for a local
    path and an S3 URI, and a run started from the laptop and a run started from
    SageMaker share the same cache rather than duplicating it.

    Args:
        path: Full location of a dataset file, local or ``s3://``.

    Returns:
        A cache whose ``root`` is the parent of the file's directory.
    """
    raw = str(path)
    if is_s3_path(raw):
        head, _, tail = raw.rpartition("/")
        parent, _, _ = head.rpartition("/")
        return ParquetCache.for_root(f"s3://{parent}")
    return ParquetCache.for_root(str(Path(raw).expanduser().parent.parent))


def load_entities(
    path: str | os.PathLike[str],
    *,
    source: int,
    cache: ParquetCache | None = None,
    columns: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Load one entity file with the schema enforced.

    Args:
        path: Location of a ``*_sourceN.tsv`` file, local or ``s3://``.
        source: 1, 2, or 3. Recorded as a column so slices by source stay
            possible without re-deriving provenance from a filename.
        cache: Cache to use. Defaults to one derived from ``path``.
        columns: Subset of columns to read. Pruning keeps peak memory
            proportional to what is needed (AGENTS.md §14).

    Returns:
        A frame with ``entity_id``, ``business_name``, ``business_address``,
        ``country``, ``source``. Text nulls are filled with ``""``.

    Raises:
        FileNotFoundError: If the file does not exist locally or in S3.
        ValueError: If required columns are missing or ``source`` is invalid.
    """
    if source not in (1, 2, 3):
        raise ValueError(f"source must be 1, 2, or 3, got {source!r}")

    filename = str(path).rsplit("/", 1)[-1]
    store = cache if cache is not None else cache_for_path(path)
    frame = store.get_entities(
        _split_of(filename), source, columns=list(columns) if columns else None
    )
    return frame


def load_ground_truth(
    path: str | os.PathLike[str], *, cache: ParquetCache | None = None
) -> pl.DataFrame:
    """Load the training ground truth with the singleton branch handled.

    Args:
        path: Location of ``train_ground_truth.tsv``, local or ``s3://``.
        cache: Cache to use. Defaults to one derived from ``path``.

    Returns:
        Frame with ``source1_entity_id``, ``matched_entity_ids`` (raw string),
        and ``match_ids`` (``List[Utf8]``), where a singleton is an **empty
        list**, never a list containing ``""``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the expected columns are missing.
    """
    filename = str(path).rsplit("/", 1)[-1]
    store = cache if cache is not None else cache_for_path(path)
    frame = store.get_frame(
        _split_of(filename), filename, columns=list(_GT_COLUMNS)
    )
    missing = [c for c in _GT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{filename}: missing expected columns {missing}")
    return ground_truth_frame(frame)


def ground_truth_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a parsed ``match_ids`` list column to a raw ground-truth frame.

    Split out so the empty-string rule is testable without touching disk.
    """
    if "matched_entity_ids" not in frame.columns:
        raise ValueError("frame has no 'matched_entity_ids' column")

    return frame.with_columns(
        pl.when(
            pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == "")
        )
        .then(pl.lit([]).cast(pl.List(pl.Utf8)))
        .otherwise(pl.col("matched_entity_ids").str.split(","))
        .alias("match_ids")
    )


def load_split(
    paths: DatasetPaths, split: Split, *, cache: ParquetCache | None = None
) -> dict[str, pl.DataFrame]:
    """Load all three sources of a split, plus ground truth for train only.

    Args:
        paths: Verified dataset locations.
        split: ``"train"`` or ``"test"``.
        cache: Cache to use. Defaults to one rooted at ``paths``. Passing an
            explicit cache lets a caller materialise once and reuse across
            splits, which is what the pipeline does.

    Returns:
        ``{"source1": ..., "source2": ..., "source3": ...}`` and, for train,
        ``"ground_truth"``.

    Raises:
        ValueError: If ``split`` is not train or test.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    store = cache if cache is not None else paths.cache()
    out: dict[str, pl.DataFrame] = {}
    for source in (1, 2, 3):
        location = paths.file(split, f"{split}_source{source}.tsv")
        out[f"source{source}"] = load_entities(
            location, source=source, cache=store
        )
    if split == "train":
        out["ground_truth"] = load_ground_truth(
            paths.train_ground_truth, cache=store
        )
    return out


def match_cardinality(gt: pl.DataFrame) -> pl.DataFrame:
    """Per-entity true match count, with the full distribution.

    Returns:
        Frame with ``m`` (match count) and ``n`` (entities with that count),
        sorted by ``m``.

    Raises:
        ValueError: If the buckets do not sum to the row count, which catches
            empty-string miscounting immediately.
    """
    if "match_ids" not in gt.columns:
        gt = ground_truth_frame(gt)

    hist = (
        gt.select(pl.col("match_ids").list.len().alias("m"))
        .group_by("m")
        .len()
        .sort("m")
        .rename({"len": "n"})
    )
    total_entities = int(hist["n"].sum())
    if total_entities != gt.height:
        raise ValueError(
            f"match-cardinality buckets sum to {total_entities:,} but the "
            f"ground truth has {gt.height:,} rows"
        )
    return hist
