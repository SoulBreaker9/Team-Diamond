"""Recall statistics derived from a per-pair strategy bitmask.

Layer: retrieval
See AGENTS.md §10 (The retrieval gate)

Why this module exists
----------------------
Measuring a union of blocking strategies only ever needs one small artifact: a
``(row_id, vendor_id, bit)`` table with one row per *true pair*, where bit *i* is
set when strategy *i* retrieved that pair. Everything else -- per-strategy
recall, union recall, coverage, reachability -- is arithmetic on that table.

The reason this arithmetic lives in ``src/`` rather than in an experiment
script is that it was previously written twice, once per driver, and the two
copies drifted into three different bugs:

1. **Cumulative reported as standalone.** ``E010``/``E011`` stored the running
   union recall in a column named ``pair_recall``, which made ``address_exact``
   look like it retrieved 59.0% of true pairs when it retrieved 11.7%.
2. **"Exactly one" meaning "one to six".** ``bit > 0 & bit < all_bits`` counts
   every pair using one to six strategies. In E012 this printed 1318 next to
   the correct 347 in its own coverage histogram -- the artifact contradicted
   itself.
3. **Metadata from module constants instead of CLI arguments.** E012 recorded
   ``query_rows = 200000`` for a run that used ``--query-rows 500``.

None of these were modelling mistakes; they were three ways of writing the same
summary differently. So there is now exactly one implementation, both drivers
import it, and it is tested directly.

The three quantities, and why they are not interchangeable
---------------------------------------------------------
``standalone_*``
    What this strategy achieves **on its own**. This is the strategy's quality.
``cumulative_*``
    What this strategy or any **earlier** one achieves. This is a property of a
    prefix of the plan, not of the strategy. Quoting it as a strategy's recall
    is how the E011 numbers became misleading.
``marginal_pair_recall``
    ``cumulative[i] - cumulative[i-1]``: the pairs this strategy finds that
    nothing before it found. This is what justifies keeping the strategy in the
    plan, and it is the column to read before keeping or dropping one.

Why a bitmask is sufficient
---------------------------
A bit is set only by the strategy that owns it, so a prefix mask
``(1 << (i + 1)) - 1`` selects exactly the pairs retrieved by the first ``i + 1``
strategies. No key table is needed, so a saved bitmask reproduces every number
in the report exactly, and the corpus does not have to be re-read.

Consequently a bitmask must contain **exactly one row per true pair** -- hence
the explicit row-count check below rather than a silent division.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

__all__ = [
    "BITMASK_SCHEMA",
    "ENTITY_RECALL_DEFINITION",
    "RECALL_REQUIRED_FOR_F05_95",
    "coverage_histogram",
    "popcount",
    "recall_report",
    "render_recall_table",
    "strategy_recall_table",
    "union_recall",
    "validate_bitmask",
]

#: Columns every bitmask must have, in this order.
BITMASK_SCHEMA = ("row_id", "vendor_id", "bit")

#: F_0.5 = 1.25R / (0.25 + R) reaches 0.95 at this pair recall.
RECALL_REQUIRED_FOR_F05_95 = 0.7917


def popcount(bit: int) -> int:
    """Number of strategies that retrieved a pair with this bitmask."""
    return bin(bit).count("1")


def validate_bitmask(found: pl.DataFrame, n_pairs: int | None = None) -> None:
    """Fail loudly if ``found`` cannot support a recall measurement.

    A bitmask missing a column, carrying a non-integer bit, or holding a
    different number of rows than there are true pairs produces a number that
    looks like a recall but is not one. All three are silent failures, so all
    three are refused here (AGENTS.md §5: never defer a bad input to a confusing
    downstream error).

    Args:
        found: The bitmask frame.
        n_pairs: Expected row count, i.e. the number of true pairs.

    Raises:
        ValueError: If a required column is absent, ``bit`` is not an integer
            dtype, or the row count disagrees with ``n_pairs``.
    """
    missing = [c for c in BITMASK_SCHEMA if c not in found.columns]
    if missing:
        raise ValueError(
            f"bitmask is missing required column(s) {missing}; "
            f"expected {list(BITMASK_SCHEMA)}, got {found.columns}"
        )
    if found.schema["bit"] != pl.Int64 and not found.schema["bit"].is_integer():
        raise ValueError(
            f"bitmask column 'bit' must be an integer dtype, got {found.schema['bit']}"
        )
    if n_pairs is not None and found.height != n_pairs:
        raise ValueError(
            f"bitmask has {found.height:,} rows but there are {n_pairs:,} true pairs. "
            f"A bitmask must contain exactly one row per true pair, otherwise the "
            f"recalls computed from it are not recalls."
        )


def strategy_recall_table(
    found: pl.DataFrame,
    plan: Sequence[str],
    *,
    n_pairs: int,
    n_entities: int,
) -> list[dict]:
    """Per-strategy recall, split into standalone / cumulative / marginal.

    Args:
        found: ``(row_id, vendor_id, bit)``, one row per true pair. Bit *i*
            belongs to ``plan[i]``.
        plan: Strategy names in the order their bits were assigned.
        n_pairs: Total true pairs. Must equal ``found.height``.
        n_entities: Distinct S1 entities having at least one true pair. For a
            complete bitmask this is ``found['row_id'].n_unique()``.

    Returns:
        One dict per strategy, in plan order, with keys ``strategy``,
        ``standalone_pair_recall``, ``standalone_entity_recall``,
        ``cumulative_pair_recall``, ``cumulative_entity_recall``,
        ``marginal_pair_recall`` and ``standalone_hits``.

    Raises:
        ValueError: See :func:`validate_bitmask`.
    """
    validate_bitmask(found, n_pairs)
    bits = found["bit"]
    previous_cumulative = 0.0
    table: list[dict] = []
    for index, name in enumerate(plan):
        bit = 1 << index
        prefix = (1 << (index + 1)) - 1

        standalone_mask = (bits & bit) > 0
        cumulative_mask = (bits & prefix) > 0

        standalone_hits = int(standalone_mask.sum())
        standalone_pair = standalone_hits / n_pairs
        cumulative_pair = int(cumulative_mask.sum()) / n_pairs
        table.append(
            {
                "strategy": name,
                "bit": bit,
                "standalone_pair_recall": round(standalone_pair, 6),
                "standalone_entity_recall": round(
                    found.filter(standalone_mask)["row_id"].n_unique() / n_entities, 6
                ),
                "cumulative_pair_recall": round(cumulative_pair, 6),
                "cumulative_entity_recall": round(
                    found.filter(cumulative_mask)["row_id"].n_unique() / n_entities, 6
                ),
                "marginal_pair_recall": round(cumulative_pair - previous_cumulative, 6),
                "standalone_hits": standalone_hits,
            }
        )
        previous_cumulative = cumulative_pair
    return table


def coverage_histogram(found: pl.DataFrame, n_strategies: int) -> dict[str, int]:
    """Count of true pairs retrieved by exactly *k* strategies, keyed by ``str(k)``.

    Every key from ``"0"`` to ``str(n_strategies)`` is present, including zeros.
    Omitting empty buckets made the shape of the result depend on the data, so a
    reader could not tell "no pair used three strategies" from "the key is
    missing". Key ``"0"`` is the miss count.

    Args:
        found: The bitmask.
        n_strategies: Number of strategies in the plan.

    Returns:
        ``{"0": missed, "1": ..., ..., str(n_strategies): ...}``.
    """
    bits = found["bit"]
    # A single set bit means exactly one strategy retrieved the pair. Testing
    # `bit < all_bits` instead -- which an earlier revision did -- counts every
    # pair using one to six strategies and silently merges six buckets into one.
    histogram = {str(k): 0 for k in range(n_strategies + 1)}
    for k, n in (
        found.select(
            pl.col("bit")
            .map_elements(popcount, return_dtype=pl.Int32)
            .alias("n_strategies")
        )
        .group_by("n_strategies")
        .len()
        .iter_rows()
    ):
        histogram[str(int(k))] = int(n)
    return histogram


def union_recall(
    found: pl.DataFrame, *, n_pairs: int, n_entities: int
) -> tuple[float, float, int]:
    """``(pair_recall, entity_hit_rate, hits)`` for the union of all strategies.

    ``entity_hit_rate`` is the fraction of S1 entities with **at least one** true
    pair present in the candidate set. It is deliberately *not* a per-entity
    recall: an entity whose four true matches were all retrieved except one
    counts as a hit here, because the question being answered is "did retrieval
    surface this business at all", not "did it surface every record of it".

    That is the right question here. The competition metric is entity-level macro
    $F_{0.5}$ over a predicted id *list*, so a partially retrieved entity is not
    automatically lost -- but the partial case is real and is exactly what
    :func:`strategy_recall_table`'s ``cumulative_entity_recall`` and the
    ``coverage_histogram`` expose. Reading the two together is the only honest
    way to know how much of the union is one lucky hit versus broad coverage.

    The name is unchanged from E011/E012 (``union.entity_recall``) so the frozen
    artifacts stay comparable; the definition is spelled out in the report so no
    reader has to guess.

    Args:
        found: The bitmask.
        n_pairs: Total true pairs.
        n_entities: Entities with at least one true pair.

    Returns:
        ``(pair_recall, entity_hit_rate, retrieved_pair_count)``.
    """
    mask = found["bit"] > 0
    hits = int(mask.sum())
    return (
        hits / n_pairs,
        found.filter(mask)["row_id"].n_unique() / n_entities,
        hits,
    )


ENTITY_RECALL_DEFINITION = (
    "union.entity_recall = fraction of S1 entities with AT LEAST ONE true pair in "
    "the candidate set. It is an entity hit rate, not a per-entity recall: an "
    "entity with 3 of its 4 true pairs retrieved counts as a hit. Use "
    "per_strategy[].cumulative_entity_recall and coverage_histogram to see how "
    "much of the union rests on partial coverage."
)


def recall_report(
    found: pl.DataFrame,
    plan: Sequence[str],
    *,
    n_pairs: int,
    n_entities: int,
    run_config: dict | None = None,
    counts: dict | None = None,
) -> dict:
    """Assemble the full recall report from one bitmask.

    Every number here comes from :func:`strategy_recall_table`,
    :func:`coverage_histogram` and :func:`union_recall`, so the printed table,
    the JSON artifact and any later re-analysis of the same bitmask are derived
    from one source of truth and cannot disagree.

    Args:
        found: The bitmask.
        plan: Strategy names in bit order.
        n_pairs: Total true pairs.
        n_entities: Entities with at least one true pair.
        run_config: Provenance -- seed, query rows, ceilings, git commit. Recorded
            verbatim; nothing here is inferred from module constants.
        counts: Extra counts to merge in (vendor pool size, query count, ...).

    Returns:
        A JSON-serialisable dict.
    """
    table = strategy_recall_table(found, plan, n_pairs=n_pairs, n_entities=n_entities)
    union_pair, union_entity, union_hits = union_recall(
        found, n_pairs=n_pairs, n_entities=n_entities
    )
    histogram = coverage_histogram(found, len(plan))
    all_bits = (1 << len(plan)) - 1
    exactly_one = histogram["1"]
    missed = n_pairs - union_hits

    # Two self-consistency checks, asserted rather than assumed. E012's published
    # artifact violated both -- its `retrieved_by_exactly_one` said 1318 while the
    # coverage histogram three lines above said 347 -- and nothing caught it
    # because nothing checked.
    if sum(histogram.values()) != n_pairs:
        raise ValueError(
            f"coverage histogram sums to {sum(histogram.values()):,} but there are "
            f"{n_pairs:,} true pairs"
        )
    if histogram["0"] != missed:
        raise ValueError(
            f"coverage histogram reports {histogram['0']:,} missed pairs but the union "
            f"mask leaves {missed:,}; these are computed differently and must agree"
        )
    best_f05 = 1.25 * union_pair / (0.25 + union_pair)
    return {
        "config": dict(run_config or {}),
        "counts": {
            "true_pairs": n_pairs,
            "entities_with_matches": n_entities,
            **dict(counts or {}),
        },
        "per_strategy": [
            {k: v for k, v in row.items() if k != "bit"} | {"max_df": None}
            for row in table
        ],
        "union": {
            "pair_recall": round(union_pair, 6),
            "entity_recall": round(union_entity, 6),
            "entity_recall_definition": ENTITY_RECALL_DEFINITION,
            "retrieved_pairs": union_hits,
        },
        "missed_pairs": missed,
        "missed_percentage": round(100 * missed / n_pairs, 2),
        "retrieved_by_exactly_one": exactly_one,
        "retrieved_by_all_strategies": int((found["bit"] == all_bits).sum()),
        "coverage_histogram": histogram,
        "coverage_histogram_note": (
            "histogram[str(k)] counts true pairs retrieved by exactly k strategies. "
            "histogram['0'] == missed_pairs. The buckets are mutually exclusive and "
            "exhaustive, so they sum to counts.true_pairs."
        ),
        "reachability": {
            "measured_pair_recall": round(union_pair, 6),
            "max_f05_at_perfect_precision": round(best_f05, 6),
            "f05_95_requires_recall": RECALL_REQUIRED_FOR_F05_95,
        },
    }


def render_recall_table(table: Sequence[dict], union_pair: float, union_entity: float) -> str:
    """Format the strategy table with unambiguous column names.

    Column headers say which is which because the whole point of the fix is that
    a reader must not have to guess whether a number is standalone or cumulative.
    """
    header = (
        f"{'strategy':16} {'standalone':>11} {'cumulative':>11} {'marginal':>10} "
        f"{'st. entity':>11} {'cu. entity':>11} {'hits':>10}"
    )
    lines = [
        header,
        "-" * len(header),
    ]
    for row in table:
        lines.append(
            f"{row['strategy']:16} {row['standalone_pair_recall']:>11.4f} "
            f"{row['cumulative_pair_recall']:>11.4f} {row['marginal_pair_recall']:>+10.4f} "
            f"{row['standalone_entity_recall']:>11.4f} {row['cumulative_entity_recall']:>11.4f} "
            f"{row['standalone_hits']:>10,}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'UNION':16} {'-':>11} {union_pair:>11.4f} {'-':>10} "
        f"{'-':>11} {union_entity:>11.4f}"
    )
    lines.append("")
    lines.append("standalone = this strategy alone.  cumulative = this strategy or an earlier one.")
    lines.append("marginal   = what this strategy adds that nothing earlier found.")
    return "\n".join(lines)
