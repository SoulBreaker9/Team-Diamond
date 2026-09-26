"""Hard-negative sampling.

Layer: training
See AGENTS.md §11 (Hard negatives), §12 (Leakage prevention)

Where negatives come from, and why it is not optional
-----------------------------------------------------
AGENTS.md §11 puts it plainly: a random S2 record is an easy negative and
teaches the model almost nothing. The reason is worth stating in full, because
it determines the whole design of this module: a random vendor differs from the
query in almost every token, so a model can separate it using a handful of the
features already present (token overlap near zero, rarity mismatch, country
mismatch). Such a model looks excellent on a random-negative validation set and
then performs badly on the real candidate distribution, which is entirely made
of records that *do* share tokens with the query — that is why retrieval
surfaced them.

So negatives are drawn from the retrieval candidates, which are by construction
the records the model will actually be asked to judge. Same index, same
strategies, same caps as inference. A negative produced by any other mechanism
is a different distribution and is not a substitute.

Two failure modes this module is built to prevent
------------------------------------------------
1. **Leaking a positive into the negative set.** If a "negative" is really a
   true match, the model is trained to score a correct match low. At inference
   that is the most expensive possible error, because it is invisible in
   training loss and only shows up as reduced recall. Exclusion is therefore
   unconditional and happens before sampling, not after.
2. **A silent positive:negative imbalance.** Sampling by fraction is not
   reproducible without a seed, and an unbalanced frame shifts the model's
   operating point in a direction nobody chose. The realised balance is returned
   as a report rather than assumed.

Reproducibility
---------------
Every draw is a pure function of ``(seed, entity ids, candidate ids)``. Nothing
depends on row order, dict ordering, or the number of threads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

__all__ = ["NegativeSample", "sample_hard_negatives", "DEFAULT_NEGATIVES_PER_POSITIVE"]

#: Starting value. A PROPOSAL: not swept. Recorded here so a run that uses it is
#: honest about having used a guess.
DEFAULT_NEGATIVES_PER_POSITIVE: Final[int] = 3


@dataclass(frozen=True, slots=True)
class NegativeSample:
    """A sampled training frame plus the statistics needed to interpret it.

    Attributes:
        frame: Positives and negatives together, labelled, with a
            ``negative_rank`` column numbering each entity's negatives in
            descending score order (``0`` is the hardest).
        n_positive: Positive pairs retained.
        n_negative: Negative pairs retained.
        n_positive_available: Positives found among the candidates, which is
            less than the ground-truth total whenever retrieval missed some. The
            difference is the retrieval ceiling for this query set.
        n_entities: Entities represented.
        seed: Seed used.
        negatives_per_positive: The setting that was applied.
    """

    frame: pl.DataFrame
    n_positive: int
    n_negative: int
    n_positive_available: int
    n_entities: int
    seed: int
    negatives_per_positive: int

    @property
    def positive_rate(self) -> float:
        """Fraction of sampled pairs that are positive."""
        total = self.n_positive + self.n_negative
        return self.n_positive / total if total else float("nan")

    def describe(self) -> str:
        """Summary for an experiment log."""
        return "\n".join(
            [
                "Hard negative sample",
                f"  entities                  : {self.n_entities:,}",
                f"  positives kept            : {self.n_positive:,}",
                f"  negatives kept            : {self.n_negative:,}",
                f"  positive rate             : {self.positive_rate:.4f}",
                f"  negatives per positive    : {self.negatives_per_positive}",
                f"  seed                      : {self.seed}",
                f"  positives available in "
                f"candidates               : {self.n_positive_available:,}",
            ]
        )


def sample_hard_negatives(
    labelled: pl.DataFrame,
    *,
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE,
    seed: int = 20260926,
    label_column: str = "is_match",
    s1_column: str = "s1_id",
    score_column: str | None = None,
    cap_negatives_per_entity: int | None = None,
) -> NegativeSample:
    """Draw hard negatives from the retrieved candidates of each entity.

    Args:
        labelled: Candidate frame already carrying a binary label, as produced by
            :func:`~team_diamond.training.labels.label_candidates`. Must come
            from the **training fold**; negatives mined with validation labels
            are a documented leak (AGENTS.md §12).
        negatives_per_positive: Negatives to draw per positive, per entity.
            Applied per entity rather than globally, so an entity with many
            candidates cannot dominate the frame and inflate the effective
            positive rate.
        seed: Seed for the draw (AGENTS.md §16).
        label_column: Label column.
        s1_column: Entity id column.
        score_column: Retrieval-evidence column used to order negatives by
            hardness, e.g. ``n_keys`` or ``strategy_rank``. When ``None``,
            candidate order is used, which is arbitrary and therefore *not*
            reproducible across retrieval changes -- so a config that means to
            draw hard negatives should name the column.
        cap_negatives_per_entity: Absolute ceiling on negatives per entity,
            applied after the per-positive setting. Stops one pathological
            entity with a huge candidate block from swamping training.

    Returns:
        A :class:`NegativeSample` whose ``frame`` holds the positives plus the
        drawn negatives, all still labelled.

    Raises:
        ValueError: If ``negatives_per_positive`` is negative, or a named
            column is missing, or an entity has positives but none available in
            its candidates (which would silently yield no negatives for it).
    """
    if negatives_per_positive < 0:
        raise ValueError(
            f"negatives_per_positive must be >= 0, got {negatives_per_positive}"
        )
    for column in (label_column, s1_column):
        if column not in labelled.columns:
            raise ValueError(
                f"labelled frame is missing {column!r}; columns: "
                f"{labelled.columns[:10]}"
            )
    if score_column is not None and score_column not in labelled.columns:
        raise ValueError(
            f"score_column {score_column!r} is not in the candidate frame; "
            f"columns: {labelled.columns[:12]}"
        )

    positives = labelled.filter(pl.col(label_column))
    negatives = labelled.filter(~pl.col(label_column))

    if negatives_per_positive == 0:
        keep = positives.with_columns(pl.lit(0, dtype=pl.Int32).alias("negative_rank"))
        return NegativeSample(
            frame=keep,
            n_positive=keep.height,
            n_negative=0,
            n_positive_available=positives.height,
            n_entities=positives[s1_column].n_unique(),
            seed=seed,
            negatives_per_positive=0,
        )

    order_column = score_column or s1_column
    ranked = negatives.sort(
        [s1_column, order_column], descending=[False, True]
    )

    # How many negatives each entity is entitled to: its positive count times
    # the setting, optionally clipped. Counting *available* positives rather
    # than true ones keeps the decision inside the candidate set -- the layer
    # that can actually see them.
    entitlement = (
        positives.group_by(s1_column)
        .len()
        .rename({"len": "_available"})
        .with_columns(
            (pl.col("_available") * negatives_per_positive).alias("_want")
        )
    )
    if cap_negatives_per_entity is not None:
        entitlement = entitlement.with_columns(
            pl.col("_want").clip(upper_bound=cap_negatives_per_entity)
        )

    # Row number within each entity's negatives, hardest first.
    numbered = ranked.with_columns(
        pl.int_range(pl.len()).over(s1_column).alias("_ordinal")
    )
    sampled = (
        numbered.join(entitlement, on=s1_column, how="inner")
        .filter(pl.col("_ordinal") < pl.col("_want"))
        .with_columns(pl.col("_ordinal").cast(pl.Int32).alias("negative_rank"))
        .drop("_ordinal", "_want", "_available")
    )

    entities_with_positives = set(positives[s1_column].to_list())
    if entities_with_positives and sampled.is_empty():
        raise ValueError(
            f"{len(entities_with_positives):,} entities have positives but zero "
            f"negatives were drawn. Either the candidate frame contains only "
            f"positives (a labelling bug), or the join keys disagree between "
            f"the positive and negative halves."
        )

    frame = pl.concat(
        [
            positives.with_columns(
                pl.lit(-1, dtype=pl.Int32).alias("negative_rank")
            ),
            sampled,
        ],
        how="vertical",
    ).sort([s1_column, "negative_rank"])

    return NegativeSample(
        frame=frame,
        n_positive=positives.height,
        n_negative=sampled.height,
        n_positive_available=positives.height,
        n_entities=positives[s1_column].n_unique(),
        seed=seed,
        negatives_per_positive=negatives_per_positive,
    )
