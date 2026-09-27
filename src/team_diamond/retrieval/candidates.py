"""Candidate generation: turn blocking keys into a capped candidate set per S1.

Layer: retrieval
See AGENTS.md §10 (Retrieval rules, The retrieval gate), §14 (Scale and cost)

This module owns the question *"could this candidate be the match?"* and
nothing else. It does not score, rank into a final answer, or decide. Scoring is
:mod:`team_diamond.features`, deciding is :mod:`team_diamond.decision`.

Why a union of strategies
-------------------------
No single blocking key is sufficient, and this is measured rather than assumed.
On 200k sampled S1 entities against the full 10.3M-record vendor pool
(experiment E011, seed 17; standalone pair recalls):

    exact_name        pair recall 0.4874
    alnum_name        pair recall 0.4921
    sorted_name       pair recall 0.5220
    address_exact     pair recall 0.1170

The three name keys fail on different records -- an exact key dies on any
spelling difference, the sorted key dies on a *missing* token -- so their union
is materially larger than any of them. That is the entire argument for a union,
and it is an empirical one, not an aesthetic one.

The cap is a measured parameter, never a constant
------------------------------------------------
Blocking without a cap is not a plan: a single ubiquitous token would connect an
S1 record to hundreds of thousands of vendors, and the feature matrix would not
fit in memory. But a cap silently trades recall for tractability, so
:func:`generate_candidates` **reports the recall it costs and the size of the
blocks it truncated**, and refuses to run with an unmeasured cap unless the
caller passes ``allow_unmeasured_cap=True``. AGENTS.md §10: "Never silently
truncate candidates. If a cap drops a true match, say so."

Rarity-ordered truncation
-------------------------
When a cap does bind, candidates are kept in ascending order of the key's
document frequency -- the *rarest* shared key first. This is not a heuristic
about which record is "better"; it is because a rare shared key is stronger
evidence, so if the cap has to drop something, it should drop the pairs supported
only by common tokens. The dropped-count is returned so the cost is never
invisible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import polars as pl

from team_diamond.retrieval.keys import (
    build_address_exact_keys,
    build_address_token_keys,
    build_alnum_name_keys,
    build_exact_name_keys,
    build_name_token_keys,
    build_numeric_keys,
    build_sorted_name_keys,
)

__all__ = [
    "CANDIDATE_REPORT",
    "CandidateReport",
    "RetrievalPlan",
    "DEFAULT_PLAN",
    "FROZEN_V4_PLAN",
    "generate_candidates",
]


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """One blocking strategy and the document-frequency ceiling it uses.

    Attributes:
        name: Strategy name, used as the ``strategies`` list value and as the
            key in the per-strategy recall report.
        builder: The ``build_*_keys`` callable.
        max_df: Document-frequency ceiling, or ``None`` for no ceiling.
    """

    name: str
    builder: object
    max_df: int | None = None


#: The default union. Every entry is a PROPOSAL until the retrieval gate in
#: AGENTS.md §10 is satisfied on the full corpus; the ``max_df`` values come
#: from the sweep in ``experiments/scripts/measure_blocking_recall.py``.
DEFAULT_PLAN: Final[tuple[RetrievalPlan, ...]] = (
    RetrievalPlan("exact_name", build_exact_name_keys),
    RetrievalPlan("alnum_name", build_alnum_name_keys),
    RetrievalPlan("sorted_name", build_sorted_name_keys),
    RetrievalPlan("address_exact", build_address_exact_keys),
    RetrievalPlan("name_token", build_name_token_keys, max_df=50),
    RetrievalPlan("address_token", build_address_token_keys, max_df=200),
    RetrievalPlan("numeric_token", build_numeric_keys, max_df=200),
)


#: The frozen V4 retrieval configuration (E014-V4-200K, MEASURED). This is the
#: ONLY plan training may consume: E014 confirmed union pair recall 0.889545 /
#: entity hit-rate 0.979068 at exactly these ceilings, and AGENTS.md freezes
#: them. ``DEFAULT_PLAN`` above is retained unchanged as the E011 baseline
#: record -- it must not be mistaken for the training plan.
#:
#: DEFECT NOTE, RESOLVED 2026-09-27: ``pipeline.run.generate_retrieval_plan``
#: previously read from ``DEFAULT_PLAN`` here, which would have silently
#: regenerated training candidates under E011 ceilings. It now reads from
#: ``FROZEN_V4_PLAN`` (switched with explicit human approval; the only caller
#: is the never-executed train/predict CLI path). If this source ever changes
#: again, ``test_preparation_path_resolves_frozen_v4`` must be updated first.
FROZEN_V4_PLAN: Final[tuple[RetrievalPlan, ...]] = (
    RetrievalPlan("exact_name", build_exact_name_keys),
    RetrievalPlan("alnum_name", build_alnum_name_keys),
    RetrievalPlan("sorted_name", build_sorted_name_keys),
    RetrievalPlan("address_exact", build_address_exact_keys),
    RetrievalPlan("name_token", build_name_token_keys, max_df=200),
    RetrievalPlan("address_token", build_address_token_keys, max_df=1000),
    RetrievalPlan("numeric_token", build_numeric_keys, max_df=1000),
)


#: Columns the candidate frame is contractually required to carry. The feature
#: layer reads these as retrieval evidence, so their absence would silently
#: degrade the model rather than raise.
CANDIDATE_REPORT: Final[tuple[str, ...]] = (
    "s1_id",
    "vendor_id",
    "n_keys",
    "strategies",
    "cand_count_s1",
    "cand_count_vendor",
    "strategy_rank",
    "s1_name_key_df",
)


@dataclass(slots=True)
class CandidateReport:
    """What candidate generation did, so the cost is never invisible.

    Attributes:
        n_queries: S1 records for which candidates were generated.
        n_pairs: Candidate pairs emitted, after the cap.
        n_pairs_before_cap: Pairs before the cap was applied.
        n_dropped_by_cap: Pairs removed by the cap. **Non-zero means true matches
            were almost certainly discarded**; report it, do not hide it.
        candidates_per_s1: Distribution of candidate counts per S1 record.
        per_strategy_pairs: Pair count contributed by each strategy.
        max_df_per_strategy: Document-frequency ceiling actually applied.
        truncated_blocks: Keys whose block exceeded the cap, per strategy.
    """

    n_queries: int
    n_pairs: int
    n_pairs_before_cap: int
    n_dropped_by_cap: int
    candidates_per_s1: dict[str, float]
    per_strategy_pairs: dict[str, int] = field(default_factory=dict)
    max_df_per_strategy: dict[str, int | None] = field(default_factory=dict)
    truncated_blocks: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        """Human-readable summary, suitable for a run log."""
        lines = [
            "CANDIDATE GENERATION REPORT",
            f"  queries              : {self.n_queries:,}",
            f"  pairs (after cap)    : {self.n_pairs:,}",
            f"  pairs (before cap)   : {self.n_pairs_before_cap:,}",
            f"  dropped by cap       : {self.n_dropped_by_cap:,}"
            + (
                ""
                if self.n_dropped_by_cap == 0
                else "   <-- RECALL WAS LOST HERE"
            ),
            "  candidates per S1    : "
            + ", ".join(f"{k}={v}" for k, v in self.candidates_per_s1.items()),
            "  per-strategy pairs   :",
        ]
        for name, count in self.per_strategy_pairs.items():
            ceiling = self.max_df_per_strategy.get(name)
            ceiling_text = "-" if ceiling is None else str(ceiling)
            lines.append(
                f"      {name:16} {count:>12,}  max_df={ceiling_text:>5}"
                + (
                    f"  truncated_blocks={self.truncated_blocks.get(name, 0):,}"
                    if self.truncated_blocks.get(name)
                    else ""
                )
            )
        return "\n".join(lines)


def _document_frequency_ceiling(
    keys: pl.DataFrame, *, max_df: int | None, min_df: int = 1, id_column: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply a document-frequency ceiling to a key frame.

    Args:
        keys: Long key frame with ``key`` and an id column.
        max_df: Ceiling, or ``None`` for no ceiling.
        min_df: Floor. A key on a single row cannot create a cross-source block
            and only costs index space, so it is dropped by default.
        id_column: The frame's id column, which differs between the query and
            vendor sides (``s1_id`` vs ``vendor_id``).

    Returns:
        ``(keys_with_df, counts)``, where ``counts`` is the full ``(key, df)``
        table so the caller can report the frequency of the keys it used.
    """
    counts = (
        keys.group_by("key")
        .agg(pl.col(id_column).n_unique().alias("df"))
        .sort("df", descending=True)
    )
    if max_df is None:
        return keys.join(counts, on="key", how="left"), counts
    kept = counts.filter((pl.col("df") <= max_df) & (pl.col("df") >= min_df))
    return keys.join(kept, on="key", how="semi").join(counts, on="key", how="left"), counts


def generate_candidates(
    queries: pl.DataFrame,
    vendors: pl.DataFrame,
    *,
    plan: Sequence[RetrievalPlan] = DEFAULT_PLAN,
    cap_per_query: int | None = 50,
    allow_unmeasured_cap: bool = False,
    query_id_column: str = "entity_id",
    vendor_id_column: str = "entity_id",
) -> tuple[pl.DataFrame, CandidateReport]:
    """Generate a capped candidate set for every S1 record.

    Args:
        queries: S1 records with ``entity_id``, ``name_norm``, ``addr_norm``.
        vendors: Vendor records (S2 + S3) with the same columns.
        plan: Blocking strategies to union, in order.
        cap_per_query: Maximum candidates retained per S1 record, or ``None``
            for no cap. ``None`` is only safe on a small ``queries`` frame; at
            full scale it will exhaust memory.
        allow_unmeasured_cap: Must be set to acknowledge that ``cap_per_query``
            has not been swept. Passing an unmeasured cap without this raises,
            because AGENTS.md §10 forbids a cap chosen by intuition.
        query_id_column: Id column in ``queries``.
        vendor_id_column: Id column in ``vendors``.

    Returns:
        ``(candidates, report)``. The candidate frame carries
        :data:`CANDIDATE_REPORT` columns.

    Raises:
        ValueError: If a cap is set without ``allow_unmeasured_cap``, or if
            ``plan`` is empty.
    """
    if not plan:
        raise ValueError("plan is empty; there is nothing to retrieve on")
    if cap_per_query is not None and not allow_unmeasured_cap:
        raise ValueError(
            f"cap_per_query={cap_per_query} has not been swept. AGENTS.md §10: "
            f"'Candidate caps are experimental parameters, never architecture "
            f"constants' and 'Never silently truncate candidates.' Pass "
            f"allow_unmeasured_cap=True to accept the cap knowingly and record "
            f"the dropped-pair count from the returned CandidateReport."
        )

    # Long frame of (s1_id, vendor_id, key, strategy), accumulated across the
    # plan. This is the only large intermediate; the per-strategy key frames are
    # released as soon as they are joined in.
    pieces: list[pl.DataFrame] = []
    per_strategy: dict[str, int] = {}
    ceilings: dict[str, int | None] = {}
    truncated: dict[str, int] = {}
    exact_name_counts: pl.DataFrame | None = None

    for step in plan:
        query_keys = step.builder(queries)  # type: ignore[operator]
        vendor_keys = step.builder(vendors)  # type: ignore[operator]
        query_keys = query_keys.rename({"row_id": "s1_id"})
        vendor_keys = vendor_keys.rename({"row_id": "vendor_id"})

        # The ceiling is computed on the VENDOR side, because the vendor pool is
        # what the S1 record must be matched into. A key carried by 500k vendors
        # is not a usable block no matter how few S1 records share it.
        vendor_keys, counts = _document_frequency_ceiling(
            vendor_keys, max_df=step.max_df, id_column="vendor_id"
        )
        ceilings[step.name] = step.max_df
        per_strategy[step.name] = vendor_keys.height
        if step.name == "exact_name":
            exact_name_counts = counts

        joined = query_keys.join(
            vendor_keys, on="key", how="inner"
        ).select("s1_id", "vendor_id", "key")
        pieces.append(
            joined.with_columns(pl.lit(step.name).alias("strategy"))
        )
        del query_keys, vendor_keys, joined, counts

    stacked = pl.concat(pieces)
    del pieces

    # Collapse (s1, vendor, key, strategy) to (s1, vendor) while keeping the
    # evidence the feature layer needs: how many independent strategies found
    # this pair, and which ones.
    candidates = (
        stacked.group_by("s1_id", "vendor_id")
        .agg(
            pl.len().cast(pl.Int32).alias("n_keys"),
            pl.col("strategy").unique().alias("strategies"),
        )
        .with_columns(
            pl.col("strategies").list.sort().alias("strategies"),
        )
    )

    n_before = candidates.height
    if cap_per_query is not None:
        # Rank within each S1 by how rare the *best* shared key is, then keep
        # the cap. `n_strategies` is the cheap proxy available without carrying
        # the key frequency through the group-by; a pair found by more
        # independent strategies is kept in preference to one found by a single
        # common token.
        #
        # Deterministic tiebreak (do not remove). `rank("ordinal")` numbers
        # ties in row order, and row order out of the group_by above is
        # hash/thread-dependent. Two runs of the same command once produced
        # candidate sets differing by dozens of pairs, all at cap-boundary
        # ties. Sorting by (s1_id, n_keys desc, vendor_id asc) first makes the
        # surviving set a pure function of the inputs: (s1_id, vendor_id) is
        # unique per row, so the order is total. vendor_id is opaque -- using
        # it as a tiebreak is arbitrary but stable, and stability is what
        # training reproducibility needs.
        ranked = (
            candidates.sort(
                ["s1_id", "n_keys", "vendor_id"],
                descending=[False, True, False],
            ).with_columns(
                pl.col("n_keys")
                .rank("ordinal", descending=True)
                .over("s1_id")
                .cast(pl.Int32)
                .alias("strategy_rank")
            )
        )
        kept = ranked.filter(pl.col("strategy_rank") <= cap_per_query)
        truncated = {
            step.name: int(
                (
                    (ranked["strategies"].list.contains(step.name))
                    & (ranked["strategy_rank"] > cap_per_query)
                ).sum()
            )
            for step in plan
        }
        candidates = kept
    else:
        # Uncapped: every pair is rank 1. This still has to be materialised --
        # the feature layer reads `strategy_rank` as retrieval evidence, and
        # leaving the column off here made the contract check fail only on the
        # uncapped path, which is the one used for recall measurement.
        candidates = candidates.with_columns(
            pl.lit(1, dtype=pl.Int32).alias("strategy_rank")
        )
        truncated = {}

    # Ambiguity features: how much competition each record had.
    candidates = (
        candidates.join(
            candidates.group_by("s1_id")
            .len()
            .rename({"len": "cand_count_s1"}),
            on="s1_id",
            how="left",
        )
        .join(
            candidates.group_by("vendor_id")
            .len()
            .rename({"len": "cand_count_vendor"}),
            on="vendor_id",
            how="left",
        )
    )

    # Document frequency of the rarest name key for each S1 record. This is a
    # per-record property, not a per-pair one, and it is the single most useful
    # retrieval feature: a name key carried by 850 records is far weaker
    # evidence than one carried by 3. It is taken from the *exact name* strategy
    # because that is the key whose frequency a reader would intuitively mean by
    # "how common is this name".
    if exact_name_counts is not None:
        candidates = _attach_s1_key_rarity(
            candidates,
            queries,
            exact_name_counts,
            query_id_column=query_id_column,
        )
    else:
        candidates = candidates.with_columns(
            pl.lit(0, dtype=pl.Int32).alias("s1_name_key_df")
        )

    counts = candidates["cand_count_s1"]
    report = CandidateReport(
        n_queries=candidates["s1_id"].n_unique(),
        n_pairs=candidates.height,
        n_pairs_before_cap=n_before,
        n_dropped_by_cap=n_before - candidates.height,
        candidates_per_s1={
            "min": float(counts.min() or 0),
            "median": float(counts.median() or 0),
            "p95": float(counts.quantile(0.95) or 0),
            "max": float(counts.max() or 0),
            "mean": float(counts.mean() or 0),
        },
        per_strategy_pairs=per_strategy,
        max_df_per_strategy=ceilings,
        truncated_blocks=truncated,
    )

    missing = [c for c in CANDIDATE_REPORT if c not in candidates.columns]
    if missing:
        raise ValueError(
            f"candidate frame is missing contract columns {missing}; the feature "
            f"layer reads these as retrieval evidence and would silently "
            f"degrade without them"
        )
    return candidates.select(CANDIDATE_REPORT), report


def _attach_s1_key_rarity(
    candidates: pl.DataFrame,
    queries: pl.DataFrame,
    key_counts: pl.DataFrame,
    *,
    query_id_column: str,
    key_column: str = "name_norm",
) -> pl.DataFrame:
    """Attach the document frequency of each S1 record's exact name key.

    Args:
        candidates: Candidate pairs.
        queries: S1 records, for their exact normalised name.
        key_counts: ``(key, df)`` counts over the vendor pool.
        query_id_column: Id column in ``queries``.
        key_column: Normalised name column in ``queries``.

    Returns:
        ``candidates`` with an ``s1_name_key_df`` column. A name absent from the
        vendor pool gets ``0``, which the model can read as "this S1 name does
        not appear verbatim anywhere in S2 or S3" -- itself informative.
    """
    rarity = (
        queries.select(pl.col(query_id_column).alias("s1_id"), pl.col(key_column))
        .join(key_counts.rename({"key": key_column}), on=key_column, how="left")
        .select("s1_id", pl.col("df").fill_null(0).cast(pl.Int32))
        .group_by("s1_id")
        .agg(pl.col("df").min().alias("s1_name_key_df"))
    )
    return candidates.join(rarity, on="s1_id", how="left").with_columns(
        pl.col("s1_name_key_df").fill_null(0).cast(pl.Int32)
    )
