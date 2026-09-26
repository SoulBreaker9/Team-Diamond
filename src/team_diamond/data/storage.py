"""Read the challenge dataset from local disk or S3, and cache it as Parquet.

Layer: data
See AGENTS.md §5 ("The dataset is not in the repository"), §14 (Memory discipline)

Why this module exists
----------------------
The dataset is ~2.4 GiB of tab-separated text across 7 files. Two facts make a
naive ``pl.read_csv`` the wrong tool on the target machine:

1. **TSV parsing dominates the runtime.** Normalisation has to run over 26M
   records, and it will be re-run many times while features are iterated on.
   Re-parsing text every time makes the feedback loop minutes-per-experiment.
2. **The execution environment changed.** The laptop has 15 GiB of RAM; the
   SageMaker training instance (``ml.r5.xlarge``) has 32 GiB. Code written and
   profiled on the laptop can still be OOM-killed on a real run, so the
   Parquet cache is also what keeps peak memory bounded: columnar, compressed,
   and selectable via ``columns=`` so a 500 MB file never needs to be fully
   resident to read two columns out of it.

So: **materialise once to Parquet, then always read Parquet.** The cache is
content-addressed by file name and is safe to delete at any time.

S3 access goes through boto3, not through Polars' S3 reader
------------------------------------------------------------
Polars can read ``s3://`` directly, but it resolves credentials through its own
chain, which does not reliably pick up the SageMaker instance role from the
IMDS credential endpoint. boto3's chain does. Rather than depend on which one a
given notebook happens to configure, this module resolves a boto3 session
itself and streams bytes, then hands a *local* file to Polars. Credentials
therefore come from exactly one place, and the failure mode when they are wrong
is a clear boto3 error rather than an empty frame.

The cache directory is local to the instance on purpose: the S3 objects are the
source of truth, and a local cache is disposable. Nothing here ever writes back
to S3.

Compliance note
---------------
This is challenge data plumbing, not matching. No external data source is
consulted; see AGENTS.md §9.2.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

__all__ = [
    "CacheMiss",
    "ParquetCache",
    "S3Location",
    "is_s3_path",
    "parse_s3_path",
    "read_csv_bytes",
    "s3_session",
]

#: Environment variable overriding the Parquet cache location.
CACHE_ROOT_ENV = "TEAM_DIAMOND_CACHE_ROOT"

#: Default cache location, used when the environment variable is unset.
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "team_diamond"

#: Entity files are read with these columns. Reading all four for a 500 MB file
#: is what the raw TSV does anyway, but naming them keeps the intent explicit and
#: makes ``columns=`` pruning available to every caller.
ENTITY_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)


class CacheMiss(FileNotFoundError):
    """Raised when a requested cache entry does not exist."""


@dataclass(frozen=True, slots=True)
class S3Location:
    """A parsed ``s3://bucket/key`` location."""

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def is_s3_path(path: str | os.PathLike[str]) -> bool:
    """True if ``path`` is an S3 URI rather than a filesystem path."""
    return str(path).startswith("s3://")


def parse_s3_path(path: str | os.PathLike[str]) -> S3Location:
    """Split an ``s3://bucket/prefix`` URI into its parts.

    Args:
        path: An S3 URI.

    Returns:
        The parsed location, with any trailing slash removed from the key.

    Raises:
        ValueError: If ``path`` is not an S3 URI, or names no bucket.
    """
    raw = str(path)
    if not raw.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {raw!r}")
    remainder = raw[len("s3://") :]
    bucket, _, key = remainder.partition("/")
    if not bucket:
        raise ValueError(f"S3 URI names no bucket: {raw!r}")
    return S3Location(bucket=bucket, key=key.strip("/"))


def s3_session() -> "object":  # noqa: ANN401 - avoids importing boto3 for typing
    """Return a boto3 session using the ambient credential chain.

    Deliberately not cached at module level: a session caches credentials, and
    in a notebook the role or profile can change between cells. Creating one per
    call is cheap relative to the data transfer it guards.

    Returns:
        A ``boto3.session.Session``.
    """
    import boto3  # imported lazily so local-only runs need no boto3

    profile = os.environ.get("AWS_PROFILE")
    if profile:
        return boto3.Session(profile_name=profile)
    return boto3.Session()


def read_csv_bytes(path: str | os.PathLike[str], *, timeout: int = 3600) -> Path:
    """Materialise ``path`` as a local file and return the local path.

    For a filesystem path this is a no-op. For an S3 URI the object is streamed
    to a temporary file, which is removed when the caller is done with it; the
    caller owns the returned path and is responsible for cleanup.

    Args:
        path: Local path or ``s3://`` URI.
        timeout: Socket timeout in seconds for the S3 transfer.

    Returns:
        A local :class:`~pathlib.Path` holding the file's bytes.

    Raises:
        FileNotFoundError: If the object does not exist.
    """
    if not is_s3_path(path):
        local = Path(path).expanduser()
        if not local.is_file():
            raise FileNotFoundError(f"no such file: {local}")
        return local

    location = parse_s3_path(path)
    client = s3_session().client("s3", config=None)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed via the client
        suffix=".tsv", delete=False
    )
    try:
        client.download_fileobj(
            location.bucket,
            location.key,
            handle,
            Config=_transfer_config(timeout),
        )
    except Exception as error:  # noqa: BLE001 - re-raised with context
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise FileNotFoundError(
            f"could not read s3://{location.bucket}/{location.key}: {error}\n"
            f"Check that the object exists and that this instance's role has "
            f"s3:GetObject on it."
        ) from error
    handle.close()
    return Path(handle.name)


def _transfer_config(timeout: int):  # noqa: ANN202 - boto3 type is dynamic
    from botocore.config import Config

    return Config(
        retries={"max_attempts": 5, "mode": "adaptive"},
        connect_timeout=60,
        read_timeout=timeout,
        max_pool_connections=16,
    )


class ParquetCache:
    """Local Parquet mirror of the challenge TSVs.

    Construct with :meth:`for_root`. Every ``get_*`` method returns a
    :class:`polars.DataFrame`; the first call for a given file materialises it,
    and later calls read the cached Parquet.

    The cache is keyed by the *logical* file name (``train_source1.tsv``), not by
    the remote path, so moving the dataset between the laptop and S3 reuses the
    same cache instead of duplicating 2.4 GiB.
    """

    __slots__ = ("_root", "_remote_root", "_session")

    def __init__(self, remote_root: str | os.PathLike[str], cache_root: Path) -> None:
        self._remote_root = str(remote_root).rstrip("/")
        self._session = None
        self._root = cache_root
        self._root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_root(
        cls,
        remote_root: str | os.PathLike[str],
        cache_root: str | os.PathLike[str] | None = None,
    ) -> ParquetCache:
        """Build a cache for ``remote_root``.

        Args:
            remote_root: Local directory or ``s3://`` prefix holding ``train/``
                and ``test/``.
            cache_root: Where to keep Parquet. Defaults to
                ``$TEAM_DIAMOND_CACHE_ROOT`` or ``~/.cache/team_diamond``.

        Returns:
            A ready cache. Note this does **not** verify the layout; call
            :meth:`has` or a ``get_*`` method to find out.
        """
        if cache_root is None:
            cache_root = os.environ.get(CACHE_ROOT_ENV) or DEFAULT_CACHE_ROOT
        return cls(remote_root, Path(cache_root).expanduser())

    @property
    def root(self) -> Path:
        """The cache directory."""
        return self._root

    @property
    def remote_root(self) -> str:
        """The dataset root this cache mirrors."""
        return self._remote_root

    def _remote(self, split: str, filename: str) -> str:
        return f"{self._remote_root}/{split}/{filename}"

    def cache_path(self, split: str, filename: str) -> Path:
        """Local Parquet path for one remote file."""
        return self._root / f"{split}__{filename.replace('.tsv', '.parquet')}"

    def has(self, split: str, filename: str) -> bool:
        """True if the Parquet cache holds this file."""
        return self.cache_path(split, filename).is_file()

    def _materialise(
        self, split: str, filename: str, *, has_header: bool
    ) -> Path:
        target = self.cache_path(split, filename)
        if target.is_file():
            return target

        source = self._remote(split, filename)
        staging = target.with_suffix(".partial")
        # `downloaded` distinguishes a temporary S3 copy from a real dataset
        # file. Only the former may be deleted: unlinking a local source would
        # destroy the user's dataset (AGENTS.md §8.1, data integrity).
        downloaded = is_s3_path(source)
        local = read_csv_bytes(source)
        try:
            frame = _read_tsv(local, has_header=has_header)
            frame.write_parquet(staging, compression="zstd", statistics=True)
            # Rename only on success, so an interrupted run never leaves a
            # truncated file that a later run would treat as a valid cache hit.
            staging.replace(target)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        finally:
            if downloaded:
                local.unlink(missing_ok=True)
        return target

    def get_frame(
        self,
        split: str,
        filename: str,
        *,
        columns: Sequence[str] | None = None,
        has_header: bool = True,
    ) -> pl.DataFrame:
        """Read one dataset file, materialising the Parquet cache if needed.

        Args:
            split: ``"train"`` or ``"test"``.
            filename: File name inside the split directory.
            columns: Subset of columns to read. Pruning is the main reason this
                method exists: it keeps peak memory proportional to the columns
                actually needed.
            has_header: Whether the TSV has a header row.

        Returns:
            The requested columns as a frame.

        Raises:
            FileNotFoundError: If the remote file is absent.
        """
        path = self._materialise(split, filename, has_header=has_header)
        return pl.read_parquet(path, columns=list(columns) if columns else None)

    def get_entities(
        self, split: str, source: int, *, columns: Sequence[str] | None = None
    ) -> pl.DataFrame:
        """Read ``{split}_source{N}.tsv`` with the entity schema enforced.

        Args:
            split: ``"train"`` or ``"test"``.
            source: 1, 2, or 3.
            columns: Subset to read; defaults to all four entity columns.

        Returns:
            A frame with the entity columns, nulls in the text columns filled
            with ``""`` (AGENTS.md §8.2), and a ``source`` column.
        """
        wanted = list(columns) if columns else list(ENTITY_COLUMNS)
        frame = self.get_frame(split, f"{split}_source{source}.tsv", columns=wanted)
        return _enforce_entity_schema(frame, source=source)

    def get_ground_truth(self, split: str = "train") -> pl.DataFrame:
        """Read ``{split}_ground_truth.tsv`` if it exists, else raise.

        The test split has no ground truth by construction, so this is expected
        to fail there; the error message says so rather than looking like a
        missing upload.
        """
        if split != "train":
            raise FileNotFoundError(
                f"the {split} split has no ground truth file; test labels are "
                f"held out by the organisers (AGENTS.md §9.1)"
            )
        from team_diamond.data.loading import ground_truth_frame

        frame = self.get_frame(
            split, "train_ground_truth.tsv", columns=["source1_entity_id", "matched_entity_ids"]
        )
        return ground_truth_frame(frame)

    def clear(self) -> None:
        """Delete every cached Parquet file. Safe; the TSVs remain the truth."""
        for path in self._root.glob("*.parquet"):
            path.unlink(missing_ok=True)
        for path in self._root.glob("*.partial"):
            path.unlink(missing_ok=True)


def _enforce_entity_schema(frame: pl.DataFrame, *, source: int) -> pl.DataFrame:
    """Cast ids to Utf8, fill text nulls, and tag the source number."""
    missing = [c for c in ENTITY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"source{source}: missing expected columns {missing}; "
            f"found {frame.columns}"
        )
    return frame.with_columns(
        [
            pl.col("entity_id").cast(pl.Utf8),
            *[pl.col(c).fill_null("").cast(pl.Utf8) for c in ENTITY_COLUMNS[1:]],
            pl.lit(source).alias("source"),
        ]
    )


def _read_tsv(path: Path, *, has_header: bool) -> pl.DataFrame:
    """Parse a challenge TSV with the settings this corpus needs.

    ``quote_char=None`` and an explicit schema matter: the files contain
    business names with commas, quotes, and tabs-in-prose, and letting Polars
    sniff quoting shifts field boundaries and silently corrupts names.
    """
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        has_header=has_header,
        infer_schema_length=0,
        null_values=[],
    )


def copy_cache_to_disk(cache: ParquetCache, destination: Path) -> Path:
    """Copy the whole Parquet cache to ``destination``.

    Useful when the run should be reproducible from an artefact rather than
    from S3. Returns the destination path.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for path in cache.root.glob("*.parquet"):
        shutil.copy2(path, destination / path.name)
    return destination
