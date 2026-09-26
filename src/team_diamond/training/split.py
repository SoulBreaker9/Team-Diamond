"""Entity-aware train/validation splitting.

Layer: training
See AGENTS.md §12 (Validation rules), §8.3 (ID integrity)

Why naive random splitting is banned here
-----------------------------------------
AGENTS.md §12 forbids random *pair* splitting as the primary validation
strategy, and the reason is concrete rather than ritualistic. Splitting pairs at
random puts two pairs describing the same underlying business on opposite sides
of the split, often with a near-duplicate name and address. The model then
scores well on validation for a reason that will not reproduce on test, and the
resulting $F_{0.5}$ is fiction.

So the unit of splitting is the **S1 entity**: every pair belonging to one S1
entity lands entirely on one side.

Why that is sufficient, and where it stops being sufficient
-----------------------------------------------------------
There is a second leakage route specific to entity resolution: the same
*vendor* appearing in both folds. A vendor matched by a training S1 entity and
also retrieved as a candidate for a validation S1 entity would let the model
memorise vendor-level appearance.

This project does not have that problem, and the reason is a **measured** fact
rather than an assumption (experiment E003): the training ground truth is a
strict partial matching — 7,638,365 pairs map to 7,638,365 distinct vendor ids,
with zero shared. Every vendor is the true match of at most one S1 entity, so no
vendor's label can appear in two folds.

That measurement is load-bearing, so
:func:`assert_partial_matching` **re-verifies it on the data at hand** instead of
trusting the note. If a future data version breaks the property, the split
fails loudly instead of producing a flattering score. Vendors still appear in
both folds' *candidate sets*, which is correct and not a leak: they are
unlabelled there, and the inference-time vendor pool is shared in exactly the
same way.

Stratification
--------------
Folds are balanced on the three things that most affect the score and least
affect the split's validity: whether the entity is a singleton, its country, and
a coarse bucket of its match count. Balancing the singleton rate matters more
than it might appear — singletons score 1.0 for emitting nothing and 0.0 for
emitting anything, so an unbalanced validation fold changes the achievable $F_{0.5}$
through no change in model quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from team_diamond.training.labels import positive_pairs

__all__ = [
    "Fold",
    "EntitySplit",
    "entity_aware_split",
    "assert_partial_matching",
    "stratify_key",
    "DEFAULT_SEED",
]

#: Seed for every stochastic step in this module (AGENTS.md §16).
DEFAULT_SEED: Final[int] = 20260926

#: Match-count buckets. Coarse on purpose: the tail beyond 5 is 0.4% of
#: entities, and stratifying finely on rare strata produces folds that differ
#: in ways that show up as metric noise rather than as signal.
_CARDINALITY_EDGES: Final[tuple[int, ...]] = (1, 2, 3, 5, 8)


@dataclass(frozen=True, slots=True)
class Fold:
    """One side of a split, as a set of S1 entity ids."""

    name: str
    entity_ids: frozenset[str]

    def __len__(self) -> int:
        return len(self.entity_ids)

    def contains_frame(self, frame: pl.DataFrame, column: str = "s1_id") -> pl.DataFrame:
        """Filter a pair frame down to this fold's entities.

        Args:
            frame: Any frame carrying the S1 id column.
            column: S1 id column name.

        Returns:
            The filtered frame. Entity ids are compared as opaque strings; no
            casting, casing, or trimming is applied (AGENTS.md §8.3).
        """
        return frame.filter(pl.col(column).is_in(list(self.entity_ids)))


@dataclass(frozen=True, slots=True)
class EntitySplit:
    """A named train/validation partition of S1 entities."""

    train: Fold
    validation: Fold
    strata: pl.DataFrame
    seed: int
    partial_matching_verified: bool

    def describe(self) -> str:
        """Human-readable summary, including the realised strata balance."""
        lines = [
            "EntitySplit",
            f"  seed                   : {self.seed}",
            f"  strict partial matching: verified={self.partial_matching_verified}",
            f"  train entities        : {len(self.train):,}",
            f"  validation entities   : {len(self.validation):,}",
            "",
            "  strata balance (share of each fold)",
        ]
        for row in self.strata.iter_rows(named=True):
            lines.append(
                f"    {row['stratum']:<28} "
                f"train={row['train_share']:.4f}  val={row['validation_share']:.4f}"
            )
        return "\n".join(lines)


def assert_partial_matching(ground_truth: pl.DataFrame) -> bool:
    """Verify that no vendor is the true match of two S1 entities.

    This is the property that makes S1-level splitting sufficient. It was
    MEASURED on the current training data (E003) and is re-checked here because
    a silent change to it would turn every validation number into an
    over-estimate with no visible symptom.

    Args:
        ground_truth: Raw ground truth frame.

    Returns:
        ``True`` if the property holds.

    Raises:
        ValueError: If any vendor id appears in more than one S1 entity's match
            set. The message states how many are affected and points at the
            remedy, because the correct response is to change the split unit
            (cluster on vendor-connected components), not to suppress the error.
    """
    pairs = positive_pairs(ground_truth)
    if pairs.is_empty():
        return True

    offenders = (
        pairs.group_by("vendor_id")
        .agg(pl.col("s1_id").n_unique().alias("n_entities"))
        .filter(pl.col("n_entities") > 1)
    )
    n_offenders = offenders.height
    if n_offenders:
        raise ValueError(
            f"ground truth is not a strict partial matching: {n_offenders:,} "
            f"vendor id(s) are the true match of more than one S1 entity "
            f"(e.g. {offenders['vendor_id'].head(3).to_list()}).\n"
            f"S1-level splitting would then leak a vendor across folds and any "
            f"validation score from it would be an over-estimate.\n"
            f"Remedy: cluster S1 entities into vendor-connected components and "
            f"split over components, not over entities."
        )
    return True


def _cardinality_bucket(count: int) -> str:
    """Map a match count to a coarse stratum label."""
    for edge in _CARDINALITY_EDGES:
        if count <= edge:
            return f"m<={edge}"
    return f"m>{_CARDINALITY_EDGES[-1]}"


def stratify_key(
    s1: pl.DataFrame,
    ground_truth: pl.DataFrame,
    *,
    s1_id_column: str = "entity_id",
    country_column: str = "country",
) -> pl.DataFrame:
    """Build the stratification frame for S1 entities.

    Args:
        s1: Normalised S1 records, with an id and a country.
        ground_truth: Raw ground truth, for the match count.
        s1_id_column: S1 id column in ``s1``.
        country_column: Country column in ``s1``.

    Returns:
        Frame with one row per S1 entity and a ``stratum`` column combining
        country with a match-count bucket. Entities with no ground-truth row are
        treated as singletons rather than dropped, so the frame covers exactly
        the entities that will be scored.
    """
    counts = (
        positive_pairs(ground_truth)
        .group_by("s1_id")
        .len()
        .rename({"s1_id": s1_id_column, "len": "match_count"})
    )
    frame = s1.select(s1_id_column, country_column).join(
        counts, on=s1_id_column, how="left"
    ).with_columns(pl.col("match_count").fill_null(0).cast(pl.Int64))
    return frame.with_columns(
        pl.col("match_count")
        .map_elements(_cardinality_bucket, return_dtype=pl.Utf8)
        .alias("cardinality_bucket")
    ).with_columns(
        (pl.col("country").fill_null("").cast(pl.Utf8) + "|" + pl.col("cardinality_bucket"))
        .alias("stratum")
    )


def _assign_within_strata(
    strata: pl.DataFrame,
    *,
    fraction: float,
    seed: int,
) -> pl.Series:
    """Assign a boolean validation flag, balanced within each stratum.

    Balancing is done by shuffling entity ids inside each stratum and taking a
    prefix, rather than by relying on a global random split. A global split would
    leave rare strata (a country with few singleton entities, say) landing
    entirely on one side, which is exactly the kind of imbalance that makes a
    validation $F_{0.5}$ uninterpretable.
    """
    rng = np.random.default_rng(seed)
    flags: list[bool] = []
    ordered = strata.sort("s1_id")
    for _, group in ordered.group_by("stratum", maintain_order=True):
        ids = group["s1_id"].to_list()
        n = len(ids)
        n_validation = int(round(n * fraction))
        # A stratum of size 1 would otherwise always be all-train or all-val by
        # coin flip; forcing at least one member to validation keeps every
        # stratum represented when the fold is large enough to warrant it.
        if 0 < n < 2 * max(1, n_validation):
            n_validation = 1
        n_validation = min(n_validation, n)
        permutation = rng.permutation(n)
        chosen = set(permutation[:n_validation].tolist())
        flags.extend(index in chosen for index in range(n))
    return pl.Series(flags, dtype=pl.Boolean)


def entity_aware_split(
    s1: pl.DataFrame,
    ground_truth: pl.DataFrame,
    *,
    validation_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
    verify_partial_matching: bool = True,
    s1_id_column: str = "entity_id",
    country_column: str = "country",
) -> EntitySplit:
    """Split S1 entities into a training and a validation fold.

    Args:
        s1: Normalised S1 records, with an id and a country.
        ground_truth: Raw ground truth for the split being partitioned.
        validation_fraction: Share of entities placed in validation. This is a
            *configuration* value, not a tuned one; record whatever is chosen in
            the experiment row.
        seed: Seed for the within-stratum shuffle. The split is fully
            reproducible from it.
        verify_partial_matching: Re-verify the strict-partial-matching property
            before splitting. Leave enabled except when profiling a data
            version known to violate it.
        s1_id_column: S1 id column in ``s1``.
        country_column: Country column in ``s1``.

    Returns:
        The split, with the realised strata balance attached.

    Raises:
        ValueError: If ``validation_fraction`` is outside ``(0, 1)``, if S1 ids
            are not unique, or if the ground truth is not a strict partial
            matching and verification is enabled.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be in (0, 1), got {validation_fraction!r}"
        )

    s1_ids = s1[s1_id_column]
    if s1_ids.n_unique() != s1.height:
        n_dupes = s1.height - s1_ids.n_unique()
        raise ValueError(
            f"S1 has {n_dupes:,} duplicate id(s). A duplicate S1 id would place "
            f"two records for one entity in the same fold by accident and make "
            f"the split unreproducible; find and resolve the source first."
        )

    verified = (
        assert_partial_matching(ground_truth) if verify_partial_matching else False
    )

    strata = stratify_key(
        s1, ground_truth, s1_id_column=s1_id_column, country_column=country_column
    )
    strata = strata.with_columns(
        pl.col(s1_id_column).cast(pl.Utf8).alias("s1_id")
    ).sort("s1_id")

    is_validation = _assign_within_strata(
        strata, fraction=validation_fraction, seed=seed
    )
    strata = strata.with_columns(is_validation.alias("is_validation"))

    train_ids = frozenset(
        strata.filter(~pl.col("is_validation"))["s1_id"].to_list()
    )
    validation_ids = frozenset(
        strata.filter(pl.col("is_validation"))["s1_id"].to_list()
    )

    n_train = len(train_ids) + len(validation_ids)
    summary = (
        strata.group_by("stratum")
        .agg(
            pl.len().alias("n"),
            (pl.col("is_validation").sum() / pl.len()).alias("validation_share"),
        )
        .with_columns(
            (1.0 - pl.col("validation_share")).alias("train_share")
        )
        .select("stratum", "n", pl.col("train_share").round(6), pl.col("validation_share").round(6))
        .sort("stratum")
    )

    if n_train == 0 or not train_ids or not validation_ids:
        raise ValueError(
            "degenerate split: one side is empty. Check validation_fraction and "
            "the stratum sizes."
        )

    return EntitySplit(
        train=Fold("train", train_ids),
        validation=Fold("validation", validation_ids),
        strata=summary,
        seed=seed,
        partial_matching_verified=verified,
    )
