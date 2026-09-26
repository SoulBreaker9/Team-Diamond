"""Ground-truth attachment — the single leakage boundary in the project.

Layer: training
See AGENTS.md §12 (Validation rules), §11 (Hard negatives)

Why this is its own module
--------------------------
Every other module in the project may be called freely at inference time. This
one cannot: it reads the labels, and anything downstream of it is training-only
code by construction. Isolating it means a leak has to be an explicit ``import``
rather than a parameter that happens to be passed.

What is and is not a leak
-------------------------
The concern in AGENTS.md §12 is real but easy to state too broadly, and an
over-broad rule causes its own damage — it leads to "safe" workarounds that
quietly weaken the model. Being precise:

**Not a leak.** The vendor pool is shared across folds. S2 and S3 records carry
no labels, and the inference-time vendor pool is likewise fully available, so a
fold-scoped retrieval index over the same vendors is a faithful simulation of
inference rather than a leak. Building a *smaller* validation index would make
validation optimistic for a reason that has nothing to do with model quality.

**A leak.** Attaching a label to a pair whose S1 entity belongs to the
validation fold, then training on it. Or mining hard negatives with validation
labels, which is subtler and more common.

**Ambiguous, treated as a leak until measured.** Any statistic fitted on the
full labelled corpus and then used to score validation rows — a global IDF over
train *and* validation S1 records, a document frequency that counts validation
vendors. The safe pattern is to build such tables from the vendor pool only,
which is available at inference, and to pass the pool explicitly so the choice
is visible in the call rather than implicit in a default.

The fingerprint
---------------
:func:`label_candidates` leaves an auditable trace: the labelled frame carries
``label_source`` and ``label_fold`` columns naming where each label came from.
An inference frame cannot acquire those columns without someone deliberately
calling this function, so a leak shows up as a column that should not be in a
submission path.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = [
    "LABEL_COLUMN",
    "FOLD_COLUMN",
    "LABEL_SOURCE_COLUMN",
    "label_candidates",
    "positive_pairs",
    "label_histogram",
]

LABEL_COLUMN: Final[str] = "is_match"
FOLD_COLUMN: Final[str] = "fold"
LABEL_SOURCE_COLUMN: Final[str] = "label_source"


def positive_pairs(ground_truth: pl.DataFrame) -> pl.DataFrame:
    """Explode ground truth into a long ``(s1_id, vendor_id)`` pair frame.

    Args:
        ground_truth: Frame with ``source1_entity_id`` and ``matched_entity_ids``
            as a comma-separated string, as read from the challenge file.

    Returns:
        Frame with ``s1_id`` and ``vendor_id``, de-duplicated, one row per true
        match. Singletons contribute no rows — that absence is the label, and
        dropping them here is why :func:`label_candidates` must attach labels to
        a *join*, never by filtering the positive set.
    """
    return (
        ground_truth.select(
            pl.col("source1_entity_id").cast(pl.Utf8).alias("s1_id"),
            pl.col("matched_entity_ids").cast(pl.Utf8).alias("raw"),
        )
        .with_columns(
            pl.when(pl.col("raw").str.strip_chars().is_null()
                    | (pl.col("raw").str.strip_chars() == ""))
            .then(pl.lit([]))
            .otherwise(
                pl.col("raw").str.split(",").list.eval(
                    pl.element().str.strip_chars()
                )
            )
            .alias("ids")
        )
        .explode("ids")
        .filter(pl.col("ids").is_not_null())
        .select("s1_id", pl.col("ids").alias("vendor_id"))
        .unique()
    )


def label_candidates(
    candidates: pl.DataFrame,
    ground_truth: pl.DataFrame,
    *,
    fold: str,
    source: str = "ground_truth",
    s1_column: str = "s1_id",
    vendor_column: str = "vendor_id",
) -> pl.DataFrame:
    """Attach a binary match label to a candidate frame.

    The join is a left semi/anti pattern rather than a filter, so a candidate
    that is *not* a true match becomes an explicit ``False``. This matters for
    singletons: an S1 entity with no true matches must still contribute
    negatives, and any implementation that starts from the positive set drops
    them — quietly biasing the model toward always finding a match, which is the
    single most damaging error available to this layer.

    Args:
        candidates: Candidate pairs, as produced by
            :func:`~team_diamond.retrieval.candidates.generate_candidates`.
        ground_truth: Raw ground truth for the split.
        fold: Name of the fold these labels belong to, recorded in the output
            so a mixed-fold training frame is detectable by inspection.
        source: Provenance string recorded in the output.
        s1_column: Entity id column in ``candidates``.
        vendor_column: Vendor id column in ``candidates``.

    Returns:
        ``candidates`` with :data:`LABEL_COLUMN` (``Boolean``),
        :data:`FOLD_COLUMN` and :data:`LABEL_SOURCE_COLUMN` added. Row order and
        row count are preserved exactly, so a labelled frame and its input are
        positionally comparable.

    Raises:
        ValueError: If a required column is missing, or if any vendor id in the
            ground truth is not a string.
    """
    for column in (s1_column, vendor_column):
        if column not in candidates.columns:
            raise ValueError(
                f"candidate frame is missing {column!r}; columns: "
                f"{candidates.columns[:10]}"
            )

    positives = positive_pairs(ground_truth).select(
        pl.col("s1_id").alias(s1_column),
        pl.col("vendor_id").alias(vendor_column),
        pl.lit(True).alias(LABEL_COLUMN),
    )

    return (
        candidates.join(positives, on=[s1_column, vendor_column], how="left")
        .with_columns(pl.col(LABEL_COLUMN).fill_null(False))
        .with_columns(
            pl.lit(fold).alias(FOLD_COLUMN),
            pl.lit(source).alias(LABEL_SOURCE_COLUMN),
        )
    )


def label_histogram(
    labelled: pl.DataFrame, ground_truth: pl.DataFrame
) -> pl.DataFrame:
    """Summarise a labelled frame, for an experiment log.

    The headline row is ``pair_recall_ceiling``. It is the fraction of true
    matches that appear anywhere in the candidate set, and it is the arithmetic
    bound on entity-level $F_{0.5}$: no matcher can recover a pair retrieval
    never produced (AGENTS.md §10). Reporting it here means a retrieval change
    and a model change can never be confused for one another -- if the ceiling
    moves, the cause is upstream of this layer entirely.

    Per-entity statistics are reported alongside the pooled rate because the two
    answer different questions. A low pooled recall concentrated in a few
    hopeless entities is a different problem from a uniform shortfall, and the
    per-entity view is what tells them apart.

    Args:
        labelled: Output of :func:`label_candidates`.
        ground_truth: The same ground truth used for labelling.

    Returns:
        Long frame of ``(metric, value)`` rows.
    """
    per_entity = labelled.group_by("s1_id").agg(
        pl.col(LABEL_COLUMN).sum().alias("found")
    )
    true_pairs = positive_pairs(ground_truth)
    true_per_entity = (
        true_pairs.group_by("s1_id").len().rename({"len": "expected"})
    )
    joined = per_entity.join(
        true_per_entity, on="s1_id", how="full", coalesce=True
    ).with_columns(
        pl.col("found").fill_null(0), pl.col("expected").fill_null(0)
    )

    total_found = int(joined["found"].sum())
    total_expected = int(joined["expected"].sum())
    ceiling = total_found / total_expected if total_expected else float("nan")

    return pl.DataFrame(
        {
            "metric": [
                "pairs",
                "positive_pairs",
                "pair_positive_rate",
                "entities_scored",
                "entities_with_no_candidate",
                "entities_with_no_true_match",
                "entities_missed_entirely",
                "entities_fully_recalled",
                "true_pairs_total",
                "pair_recall_ceiling",
            ],
            "value": [
                str(labelled.height),
                str(int(labelled[LABEL_COLUMN].sum())),
                (
                    f"{float(labelled[LABEL_COLUMN].mean()):.6f}"
                    if labelled.height
                    else "nan"
                ),
                str(joined.height),
                str(int((joined["found"] == 0).sum())),
                str(int((joined["expected"] == 0).sum())),
                str(
                    int(
                        (
                            (joined["expected"] > 0)
                            & (joined["found"] == 0)
                        ).sum()
                    )
                ),
                str(
                    int(
                        (
                            (joined["expected"] > 0)
                            & (joined["found"] >= joined["expected"])
                        ).sum()
                    )
                ),
                str(total_expected),
                f"{ceiling:.6f}" if total_expected else "nan",
            ],
        }
    )
