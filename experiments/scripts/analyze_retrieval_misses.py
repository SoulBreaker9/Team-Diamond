#!/usr/bin/env python
"""E012: Retrieval Miss Analysis. Which true pairs does the plan never find, and why?

Layer: retrieval (experiment driver)
See AGENTS.md §10 (The retrieval gate), §14 (Scale and cost)

What this answers
-----------------
E011 measured the *aggregate* recall of a seven-strategy plan. It did not say
what the missed pairs look like. This script produces the per-pair strategy
bitmask and then profiles the pairs no strategy retrieved, against the pairs
that were retrieved, so a retrieval change can be aimed at an observed failure
rather than a guess.

Sample size
-----------
The sample is whatever ``--query-rows`` says, and the output records that number
verbatim. An earlier revision of this script wrote the module default
(200 000) into the artifact regardless of the flag actually used, so a
500-query diagnostic run was published as a 200 000-query result and a
within-sample percentage was read as a population figure. That is fixed: the
report's ``config`` is the only source of the run's parameters, and
``sample_scope`` states in words what the numbers may and may not be generalised
to.

Every percentage in this report is a percentage **of the sampled pairs**. None
of them is a population estimate. To make a claim about the full corpus, raise
``--query-rows`` and re-run.

Why the bitmask is saved
------------------------
It is the only artifact that keeps per-strategy attribution, and it is small
(one row per true pair). With it, the entire analysis can be redone -- including
after a fix to this script -- without re-running retrieval. Pass it back with
``--bitmask-in``.

No new retrieval strategies are introduced. No model is trained. Ground truth is
used only to decide which pairs are true, never to build any key or index.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import polars as pl

from team_diamond.data import DatasetPaths, load_ground_truth, load_split
from team_diamond.preprocessing.normalize_fast import normalize_columns
from team_diamond.retrieval.keys import (
    build_address_exact_keys,
    build_address_token_keys,
    build_alnum_name_keys,
    build_exact_name_keys,
    build_name_token_keys,
    build_numeric_keys,
    build_sorted_name_keys,
)
from team_diamond.retrieval.recall_stats import (
    coverage_histogram,
    strategy_recall_table,
    union_recall,
    validate_bitmask,
)

# Defaults for the CLI only. These are *never* written into a report: the report
# records the arguments actually used, so a small diagnostic run cannot be
# published under the default's name.
DEFAULT_QUERY_ROWS = 200_000
DEFAULT_SEED = 17

NAME_TOKEN_MAX_DF = 50
ADDRESS_TOKEN_MAX_DF = 200
NUMERIC_MAX_DF = 200

PLAN: list[tuple[str, object, int | None]] = [
    ("exact_name", build_exact_name_keys, None),
    ("alnum_name", build_alnum_name_keys, None),
    ("sorted_name", build_sorted_name_keys, None),
    ("address_exact", build_address_exact_keys, None),
    ("name_token", build_name_token_keys, NAME_TOKEN_MAX_DF),
    ("address_token", build_address_token_keys, ADDRESS_TOKEN_MAX_DF),
    ("numeric_token", build_numeric_keys, NUMERIC_MAX_DF),
]

STRATEGY_BITS = {name: 1 << i for i, (name, _, _) in enumerate(PLAN)}
STRATEGY_MAX_DF = {name: max_df for name, _, max_df in PLAN}
ALL_STRATEGIES_BIT = (1 << len(PLAN)) - 1


def rss_gb() -> float:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("nan")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():5.2f}GiB  {message}", flush=True)


def hits_for_strategy(
    strategy: str,
    builder,
    max_df: int | None,
    queries: pl.DataFrame,
    vendors: pl.LazyFrame,
    true_pairs: pl.DataFrame,
    relevant_vendors: pl.Series,
) -> pl.DataFrame:
    """True pairs retrieved by one strategy. Returns ``(row_id, vendor_id)``."""
    vendor_keys_lazy = builder(vendors)

    kept: pl.DataFrame | None = None
    if max_df is not None:
        df_counts = (
            vendor_keys_lazy.group_by("key")
            .agg(pl.col("row_id").n_unique().alias("df"))
            .collect(engine="streaming")
        )
        kept = (
            df_counts.filter((pl.col("df") <= max_df) & (pl.col("df") >= 1))
            .select("key")
            .lazy()
        )
        del df_counts
        gc.collect()

    vendor_side = (
        vendor_keys_lazy.filter(pl.col("row_id").is_in(relevant_vendors.implode()))
        .select(pl.col("row_id").alias("vendor_id"), "key")
        .unique()
    )
    if kept is not None:
        vendor_side = vendor_side.join(kept, on="key", how="semi")
    vendor_keys = vendor_side.collect()
    del vendor_side, vendor_keys_lazy
    gc.collect()

    query_keys = builder(queries.lazy()).collect()
    if kept is not None:
        query_keys = query_keys.join(kept.collect(), on="key", how="semi")
    query_keys = query_keys.select("row_id", "key").unique()

    hits = (
        true_pairs.join(query_keys, on="row_id", how="inner")
        .rename({"key": "k"})
        .join(vendor_keys, left_on=["vendor_id", "k"], right_on=["vendor_id", "key"], how="semi")
        .select("row_id", "vendor_id")
        .unique()
    )
    del vendor_keys, query_keys
    gc.collect()
    log(f"  {strategy:<14} hits={hits.height:>9,}")
    return hits


def compute_retrieval_bitmask(
    queries: pl.DataFrame,
    vendors: pl.DataFrame,
    true_pairs: pl.DataFrame,
    relevant_vendors: pl.Series,
) -> pl.DataFrame:
    """Run all strategies and return a frame with (row_id, vendor_id, bitmask)."""
    found = true_pairs.with_columns(pl.lit(0).alias("bit"))

    for index, (name, builder, max_df) in enumerate(PLAN):
        hits = hits_for_strategy(name, builder, max_df, queries, vendors.lazy(), true_pairs, relevant_vendors)
        bit = 1 << index

        found = (
            found.join(
                hits.with_columns(pl.lit(bit).alias("b")),
                on=["row_id", "vendor_id"],
                how="left",
            )
            .with_columns((pl.col("bit") | pl.col("b").fill_null(0)).alias("bit"))
            .select("row_id", "vendor_id", "bit")
        )
        del hits
        gc.collect()

    return found


def analyze_misses(
    found: pl.DataFrame,
    true_pairs: pl.DataFrame,
    vendors_norm: pl.DataFrame,
    vendors_raw: pl.DataFrame,
    s1_norm: pl.DataFrame,
    s1_raw: pl.DataFrame,
    run_config: dict,
) -> dict:
    """Profile the true pairs no strategy retrieved, against the ones that were.

    Pure with respect to the filesystem: every input is passed in, so this can be
    exercised on synthetic frames and re-run from a saved bitmask.

    Args:
        found: ``(row_id, vendor_id, bit)``, one row per true pair.
        true_pairs: ``(row_id, vendor_id)``, the ground-truth pairs.
        vendors_norm: Normalised vendor pool; needs ``entity_id``, ``name_norm``,
            ``addr_norm``, ``country``, ``source``, ``has_address``, ``has_name``.
        vendors_raw: Raw vendor pool; needs ``entity_id``, ``business_name``,
            ``business_address``, ``source``.
        s1_norm: Normalised S1; needs ``entity_id``, ``business_name``,
            ``business_address``, ``country``, ``name_norm``, ``addr_norm``,
            ``has_address``, ``has_name``.
        s1_raw: Raw S1; needs ``entity_id``, ``business_name``,
            ``business_address``.
        run_config: The parameters this run actually used. Recorded verbatim.
            Nothing in the output may come from a module-level default.

    Returns:
        A JSON-serialisable dict.

    Raises:
        ValueError: If ``found`` is not a bitmask over exactly the true pairs.
    """
    validate_bitmask(found, true_pairs.height)
    n_pairs = true_pairs.height
    n_entities = true_pairs["row_id"].n_unique()

    u_pair, u_ent, union_hits = union_recall(
        found, n_pairs=n_pairs, n_entities=n_entities
    )
    missed = found.filter(pl.col("bit") == 0)
    missed_count = missed.height

    # Per-strategy recall comes from the shared module, so E012 and E011/E010
    # cannot disagree about what "pair_recall" means.
    per_strategy = [
        {k: v for k, v in row.items() if k != "bit"}
        | {"max_df": dict(STRATEGY_MAX_DF).get(row["strategy"])}
        for row in strategy_recall_table(
            found,
            [name for name, _, _ in PLAN],
            n_pairs=n_pairs,
            n_entities=n_entities,
        )
    ]

    # Every number in `strategy_coverage` is now read from the shared
    # histogram, so it cannot contradict the histogram any more. E012's
    # published artifact said `retrieved_by_exactly_one: 1318` while the very
    # same block reported a distribution whose "1" bucket was 347 -- the first
    # counted every pair using one to six strategies.
    histogram = coverage_histogram(found, len(PLAN))
    strategy_coverage = {
        "histogram": histogram,
        "note": (
            "histogram[str(k)] counts true pairs retrieved by exactly k "
            "strategies. Buckets are mutually exclusive and exhaustive, so they "
            "sum to recall.true_pairs. histogram['0'] equals missed_pairs."
        ),
        "missed_pairs_all_zero": histogram["0"],
        "retrieved_by_exactly_one": histogram["1"],
        "retrieved_by_two_or_more": union_hits - histogram["1"],
        "retrieved_by_all_strategies": int(
            (found["bit"] == ALL_STRATEGIES_BIT).sum()
        ),
        "histogram_sums_to_true_pairs": sum(histogram.values()) == n_pairs,
    }
    if missed_count != histogram["0"]:
        raise ValueError(
            f"miss count {missed_count} disagrees with coverage histogram {histogram['0']}"
        )

    # --- attach the source records, once, for both groups -------------------
    # Earlier revisions built the vendor raw join twice inline and left a third
    # unused copy behind. The column names also drifted: `extract_hard_examples`
    # read `business_name_vendor` while the frame carried
    # `business_name_vendor_raw`, so every published hard example had a null
    # vendor name and address.
    s1_side = s1_norm.select(
        pl.col("entity_id").alias("row_id"),
        pl.col("business_name").alias("business_name_s1"),
        pl.col("business_address").alias("business_address_s1"),
        pl.col("country").alias("country_s1"),
        pl.col("name_norm").alias("name_norm_s1"),
        pl.col("addr_norm").alias("addr_norm_s1"),
        pl.col("has_address").alias("has_address_s1"),
        pl.col("has_name").alias("has_name_s1"),
    )
    vendor_side = vendors_norm.select(
        pl.col("entity_id").alias("vendor_id"),
        pl.col("name_norm").alias("name_norm_vendor"),
        pl.col("addr_norm").alias("addr_norm_vendor"),
        pl.col("country").alias("country_vendor"),
        pl.col("source").alias("source_vendor"),
        pl.col("has_address").alias("has_address_vendor"),
        pl.col("has_name").alias("has_name_vendor"),
    )
    s1_raw_side = s1_raw.select(
        pl.col("entity_id").alias("row_id"),
        pl.col("business_name").alias("business_name_s1_raw"),
        pl.col("business_address").alias("business_address_s1_raw"),
    )
    vendor_raw_side = vendors_raw.select(
        pl.col("entity_id").alias("vendor_id"),
        pl.col("business_name").alias("business_name_vendor_raw"),
        pl.col("business_address").alias("business_address_vendor_raw"),
        pl.col("source").alias("source_vendor_raw"),
    )

    def attach(pairs: pl.DataFrame) -> pl.DataFrame:
        return (
            pairs.join(s1_side, on="row_id", how="left")
            .join(vendor_side, on="vendor_id", how="left")
            .join(s1_raw_side, on="row_id", how="left")
            .join(vendor_raw_side, on="vendor_id", how="left")
        )

    # `found` already carries (row_id, vendor_id) for every true pair, so no
    # re-join with true_pairs is needed and no rows can be duplicated.
    missed_with_data = attach(missed)
    retrieved_with_data = attach(found.filter(pl.col("bit") > 0))

    # A. Country breakdown
    country_breakdown = analyze_country_breakdown(missed_with_data, retrieved_with_data)

    # B. Source breakdown
    source_breakdown = analyze_source_breakdown(missed_with_data, retrieved_with_data)

    # C. Missing-field breakdown
    missing_field_breakdown = analyze_missing_fields(missed_with_data, retrieved_with_data)

    # D. Name similarity diagnostics
    name_diagnostics = analyze_name_similarity(missed_with_data, retrieved_with_data)

    # E. Address similarity diagnostics
    addr_diagnostics = analyze_address_similarity(missed_with_data, retrieved_with_data)

    # F. Failure-mode categorization
    failure_modes = categorize_failure_modes(missed_with_data, retrieved_with_data)

    # G. Hard examples
    hard_examples = extract_hard_examples(missed_with_data, max_examples=100)

    return {
        "experiment_id": "E012",
        "config": dict(run_config),
        "sample_scope": _sample_scope(run_config, n_pairs, missed_count),
        "recall": {
            "true_pairs": n_pairs,
            "entities_with_matches": n_entities,
            "union_pair_recall": round(u_pair, 6),
            "union_entity_recall": round(u_ent, 6),
            "retrieved_pairs": union_hits,
            "missed_pairs": missed_count,
            "missed_percentage": round(100 * missed_count / n_pairs, 2),
            "max_f05_at_perfect_precision": round(1.25 * u_pair / (0.25 + u_pair), 6),
        },
        "per_strategy": per_strategy,
        "strategy_coverage": strategy_coverage,
        "country_breakdown": country_breakdown,
        "source_breakdown": source_breakdown,
        "missing_field_breakdown": missing_field_breakdown,
        "name_similarity_diagnostics": name_diagnostics,
        "address_similarity_diagnostics": addr_diagnostics,
        "failure_modes": failure_modes,
        "hard_examples": hard_examples,
        "recall_by_country": country_breakdown.get("recall_by_country", {}),
        "recall_by_source": source_breakdown.get("recall_by_source", {}),
    }


def _sample_scope(run_config: dict, n_pairs: int, missed_count: int) -> dict:
    """State, in the artifact itself, what the percentages may be generalised to.

    A diagnostic sample is not a population. Recording the caveat next to the
    numbers is the only reliable defence against a later reader -- or a later
    agent -- treating 386/420 as 386/154,640.
    """
    query_rows = run_config.get("query_rows")
    return {
        "queries_sampled": query_rows,
        "true_pairs_in_sample": n_pairs,
        "missed_pairs_in_sample": missed_count,
        "population": (
            "S1 training entities, sampled uniformly at random with the recorded "
            "seed, then restricted to those with at least one ground-truth match."
        ),
        "generalisation": (
            f"Every percentage in this report is a percentage of the {n_pairs:,} "
            f"true pairs in this {query_rows:,}-query sample. These figures "
            "describe the sample only. They are NOT estimates of the same "
            "quantities over the full corpus, and must not be quoted as such. "
            "Re-run with a larger --query-rows before making a population claim."
        ),
    }


def analyze_country_breakdown(missed: pl.DataFrame, retrieved: pl.DataFrame) -> dict:
    """Analyze missed vs retrieved by country (S1 country)."""
    def count_by_country(df: pl.DataFrame) -> dict[str, int]:
        vc = df["country_s1"].value_counts()
        return {
            str(row[0]): int(row[1]) for row in vc.iter_rows()
        }

    missed_by_country = count_by_country(missed)
    retrieved_by_country = count_by_country(retrieved)

    all_countries = set(missed_by_country.keys()) | set(retrieved_by_country.keys())
    recall_by_country = {}
    for c in all_countries:
        m = missed_by_country.get(c, 0)
        r = retrieved_by_country.get(c, 0)
        total = m + r
        recall_by_country[c] = {
            "missed": m,
            "retrieved": r,
            "total": total,
            "recall": round(r / total, 6) if total > 0 else 0.0,
        }

    return {
        "missed_by_country": missed_by_country,
        "retrieved_by_country": retrieved_by_country,
        "recall_by_country": recall_by_country,
    }


def analyze_source_breakdown(missed: pl.DataFrame, retrieved: pl.DataFrame) -> dict:
    """Analyze missed vs retrieved by vendor source (2 or 3)."""
    def count_by_source(df: pl.DataFrame) -> dict[str, int]:
        vc = df["source_vendor"].value_counts()
        return {
            str(row[0]): int(row[1]) for row in vc.iter_rows()
        }

    missed_by_source = count_by_source(missed)
    retrieved_by_source = count_by_source(retrieved)

    all_sources = set(missed_by_source.keys()) | set(retrieved_by_source.keys())
    recall_by_source = {}
    for s in all_sources:
        m = missed_by_source.get(s, 0)
        r = retrieved_by_source.get(s, 0)
        total = m + r
        recall_by_source[s] = {
            "missed": m,
            "retrieved": r,
            "total": total,
            "recall": round(r / total, 6) if total > 0 else 0.0,
        }

    return {
        "missed_by_source": missed_by_source,
        "retrieved_by_source": retrieved_by_source,
        "recall_by_source": recall_by_source,
    }


def analyze_missing_fields(missed: pl.DataFrame, retrieved: pl.DataFrame) -> dict:
    """Analyze missing name/address fields for missed vs retrieved."""
    def field_stats(df: pl.DataFrame, prefix: str) -> dict:
        n = df.height
        if n == 0:
            return {"total": 0}
        name_missing = df[f"has_name_{prefix}"].not_().sum() if f"has_name_{prefix}" in df.columns else 0
        addr_missing = df[f"has_address_{prefix}"].not_().sum() if f"has_address_{prefix}" in df.columns else 0
        both_missing = ((df[f"has_name_{prefix}"].not_()) & (df[f"has_address_{prefix}"].not_())).sum() if f"has_name_{prefix}" in df.columns else 0
        neither_missing = (df[f"has_name_{prefix}"] & df[f"has_address_{prefix}"]).sum() if f"has_name_{prefix}" in df.columns else 0
        return {
            "total": n,
            "name_missing": int(name_missing),
            "address_missing": int(addr_missing),
            "both_missing": int(both_missing),
            "neither_missing": int(neither_missing),
        }

    return {
        "missed": field_stats(missed, "s1"),
        "retrieved": field_stats(retrieved, "s1"),
        "missed_vendor": field_stats(missed, "vendor"),
        "retrieved_vendor": field_stats(retrieved, "vendor"),
    }


def token_count(text: pl.Expr) -> pl.Expr:
    """Number of non-empty space-separated tokens in a normalised string.

    ``str.split(" ").list.len()`` looks equivalent and is not: it reports **1**
    for the empty string, and N+1 for any doubled internal space. An empty
    string here means "this field is missing" -- S2 and S3 both contain records
    with no address, and they are exactly the records the missing-address
    failure mode is about. The old expression therefore counted a missing
    address as one token, inflating the statistic meant to characterise it.

    Note the obvious-looking fix is also wrong: ``list.eval(predicate)``
    applies the predicate element-wise and keeps the list length, so
    ``list.eval(...).list.len()`` still returns 1 for ``""``. The boolean list
    has to be *summed*.

    Args:
        text: Expression yielding the normalised string.

    Returns:
        Expression yielding a ``u32`` token count: 0 for an empty or all-space
        string, and unaffected by doubled spaces.
    """
    return (
        text.str.split(" ")
        .list.eval(pl.element().str.len_chars() > 0)
        .list.sum()
        .cast(pl.UInt32)
    )


def analyze_name_similarity(missed: pl.DataFrame, retrieved: pl.DataFrame) -> dict:
    """Compare name similarity distributions for missed vs retrieved."""
    def compute_similarities(df: pl.DataFrame) -> dict:
        if df.height == 0:
            return {}
        name_s1 = df["name_norm_s1"] if "name_norm_s1" in df.columns else pl.lit("")
        name_v = df["name_norm_vendor"] if "name_norm_vendor" in df.columns else pl.lit("")

        similarities = df.select(
            exact=(name_s1 == name_v).cast(pl.Float32),
            len_ratio=(name_s1.str.len_chars().cast(pl.Float32) /
                       name_v.str.len_chars().clip(lower_bound=1).cast(pl.Float32)),
            s1_len=name_s1.str.len_chars(),
            v_len=name_v.str.len_chars(),
            s1_tokens=token_count(name_s1),
            v_tokens=token_count(name_v),
        )

        return {
            "exact_equality": {
                "mean": float(similarities["exact"].mean()),
                "median": float(similarities["exact"].median()),
                "p95": float(similarities["exact"].quantile(0.95)),
            },
            "len_ratio": {
                "mean": float(similarities["len_ratio"].mean()),
                "median": float(similarities["len_ratio"].median()),
            },
            "s1_token_count": {
                "mean": float(similarities["s1_tokens"].mean()),
                "median": float(similarities["s1_tokens"].median()),
            },
            "vendor_token_count": {
                "mean": float(similarities["v_tokens"].mean()),
                "median": float(similarities["v_tokens"].median()),
            },
        }

    return {
        "missed": compute_similarities(missed),
        "retrieved": compute_similarities(retrieved),
    }


def analyze_address_similarity(missed: pl.DataFrame, retrieved: pl.DataFrame) -> dict:
    """Compare address similarity distributions for missed vs retrieved."""
    def compute_similarities(df: pl.DataFrame) -> dict:
        if df.height == 0:
            return {}
        addr_s1 = df["addr_norm_s1"] if "addr_norm_s1" in df.columns else pl.lit("")
        addr_v = df["addr_norm_vendor"] if "addr_norm_vendor" in df.columns else pl.lit("")

        similarities = df.select(
            exact=(addr_s1 == addr_v).cast(pl.Float32),
            len_ratio=(addr_s1.str.len_chars().cast(pl.Float32) /
                       addr_v.str.len_chars().clip(lower_bound=1).cast(pl.Float32)),
            s1_len=addr_s1.str.len_chars(),
            v_len=addr_v.str.len_chars(),
            s1_tokens=token_count(addr_s1),
            v_tokens=token_count(addr_v),
        )

        return {
            "exact_equality": {
                "mean": float(similarities["exact"].mean()),
                "median": float(similarities["exact"].median()),
                "p95": float(similarities["exact"].quantile(0.95)),
            },
            "len_ratio": {
                "mean": float(similarities["len_ratio"].mean()),
                "median": float(similarities["len_ratio"].median()),
            },
            "s1_token_count": {
                "mean": float(similarities["s1_tokens"].mean()),
                "median": float(similarities["s1_tokens"].median()),
            },
            "vendor_token_count": {
                "mean": float(similarities["v_tokens"].mean()),
                "median": float(similarities["v_tokens"].median()),
            },
        }

    return {
        "missed": compute_similarities(missed),
        "retrieved": compute_similarities(retrieved),
    }


def categorize_failure_modes(missed: pl.DataFrame, retrieved: pl.DataFrame) -> dict:
    """Partition the missed pairs into mutually exclusive failure modes.

    Why a partition and not a set of indicators
    -------------------------------------------
    An earlier revision emitted six overlapping boolean counts under a note
    saying "categories are not mutually exclusive". That is the worst of both
    worlds: the numbers do not sum to the total, so a reader cannot tell what
    fraction of misses each mode accounts for, and the note invites exactly the
    mistake of reading one count as a share. On the 500-query sample the counts
    happened to be disjoint and sum to 420, which made the misleading note look
    harmless.

    So the primary output is now a **strict partition**: a ``when/then``
    chain, evaluated in order, first match wins. Every missed pair lands in
    exactly one bucket, ``unclassified`` is reported explicitly, and the counts
    are asserted to sum to the total. If a future data shape ever falls through
    every branch, it shows up as a non-zero ``unclassified`` rather than
    silently vanishing.

    Two branches are *expected* to be empty, and that expectation is itself
    informative. A missed pair cannot have exactly equal normalised names:
    ``exact_name`` has no document-frequency ceiling, so equal names always
    produce a hit. Likewise an exactly equal address is always caught by
    ``address_exact``. A non-zero count in either branch would mean retrieval
    was skipped, not that the pair is hard -- so they are reported separately
    from the genuine modes.

    Args:
        missed: Missed pairs with the source columns attached.
        retrieved: Retrieved pairs with the source columns attached. Not used
            for the partition itself; retained for signature symmetry and
            future comparison.

    Returns:
        Dict with ``partition`` (disjoint, sums to total), ``indicators``
        (overlapping raw booleans, kept for continuity with E011 reporting) and
        an ``integrity`` block.
    """
    if missed.height == 0:
        return {
            "total": 0,
            "partition": {},
            "indicators": {},
            "integrity": {"partition_sums_to_total": True, "unclassified": 0},
        }

    df = missed.with_columns(
        [
            (pl.col("name_norm_s1") == pl.col("name_norm_vendor")).alias("name_exact"),
            (pl.col("addr_norm_s1") == pl.col("addr_norm_vendor")).alias("addr_exact"),
            (
                pl.col("name_norm_s1").str.len_chars().cast(pl.Float32)
                / pl.col("name_norm_vendor")
                .str.len_chars()
                .clip(lower_bound=1)
                .cast(pl.Float32)
            ).alias("name_len_ratio"),
            (
                pl.col("addr_norm_s1").str.len_chars().cast(pl.Float32)
                / pl.col("addr_norm_vendor")
                .str.len_chars()
                .clip(lower_bound=1)
                .cast(pl.Float32)
            ).alias("addr_len_ratio"),
            pl.col("has_name_s1").alias("s1_has_name"),
            pl.col("has_address_s1").alias("s1_has_addr"),
            pl.col("has_name_vendor").alias("v_has_name"),
            pl.col("has_address_vendor").alias("v_has_addr"),
        ]
    )

    name_missing = ~pl.col("s1_has_name") | ~pl.col("v_has_name")
    addr_missing = ~pl.col("s1_has_addr") | ~pl.col("v_has_addr")
    s1_blank = ~pl.col("s1_has_name") & ~pl.col("s1_has_addr")
    vendor_blank = ~pl.col("v_has_name") & ~pl.col("v_has_addr")

    # First match wins, so the buckets cannot overlap by construction.
    partition_expr = (
        pl.when(s1_blank)
        .then(pl.lit("s1_record_blank"))
        .when(vendor_blank)
        .then(pl.lit("vendor_record_blank"))
        .when(name_missing)
        .then(pl.lit("name_missing_either_side"))
        .when(addr_missing)
        .then(pl.lit("address_missing_either_side"))
        .when(pl.col("name_exact") & ~pl.col("addr_exact"))
        .then(pl.lit("name_identical_address_differs"))
        .when(pl.col("addr_exact") & ~pl.col("name_exact"))
        .then(pl.lit("address_identical_name_differs"))
        .when(
            ~pl.col("name_exact")
            & ~pl.col("addr_exact")
            & pl.col("s1_has_name")
            & pl.col("v_has_name")
            & pl.col("s1_has_addr")
            & pl.col("v_has_addr")
        )
        .then(pl.lit("all_present_nothing_identical"))
        .otherwise(pl.lit("unclassified"))
    )

    labelled = df.with_columns(partition_expr.alias("failure_mode"))
    counts = (
        labelled.group_by("failure_mode")
        .len()
        .sort("failure_mode")
    )
    partition = {
        row["failure_mode"]: {
            "count": int(row["len"]),
            "percentage": round(100 * int(row["len"]) / missed.height, 2),
        }
        for row in counts.iter_rows(named=True)
    }

    # Overlapping raw indicators, retained because they answer questions the
    # partition cannot (e.g. how many misses are cross-script *and* name-only).
    indicator_expr = {
        "s1_record_blank": s1_blank,
        "vendor_record_blank": vendor_blank,
        "name_missing_either_side": name_missing,
        "address_missing_either_side": addr_missing,
        "name_identical": pl.col("name_exact"),
        "address_identical": pl.col("addr_exact"),
        "all_four_fields_present": (
            pl.col("s1_has_name")
            & pl.col("v_has_name")
            & pl.col("s1_has_addr")
            & pl.col("v_has_addr")
        ),
    }
    indicators = {
        name: {"count": int(df.select(expr.sum()).item())}
        for name, expr in indicator_expr.items()
    }

    partitioned_total = sum(v["count"] for v in partition.values())
    unclassified = partition.get("unclassified", {}).get("count", 0)
    # These two must be zero: an exact name or address is always retrieved,
    # because the strategies that use them have no document-frequency ceiling.
    unreachable = {
        "name_identical_address_differs": partition.get(
            "name_identical_address_differs", {}
        ).get("count", 0),
        "address_identical_name_differs": partition.get(
            "address_identical_name_differs", {}
        ).get("count", 0),
    }

    result = {
        "total": missed.height,
        "partition": partition,
        "partition_note": (
            "Strict partition, first match wins. Counts sum to `total` exactly; "
            "`unclassified` must be 0. Percentages are of this sample's missed "
            "pairs only."
        ),
        "indicators": indicators,
        "indicators_note": (
            "Overlapping boolean counts. A pair may appear under several keys, so "
            "these do NOT sum to `total` and must not be read as shares."
        ),
        "integrity": {
            "partition_sums_to_total": partitioned_total == missed.height,
            "partitioned_total": partitioned_total,
            "unclassified": unclassified,
            "expected_empty_buckets": unreachable,
            "expected_empty_note": (
                "Both must be 0. A non-zero count means an exactly equal name or "
                "address was not retrieved, which contradicts an unceiled exact "
                "strategy and indicates a bug rather than a hard pair."
            ),
        },
    }
    if partitioned_total != missed.height:
        raise ValueError(
            f"failure-mode partition sums to {partitioned_total} but there are "
            f"{missed.height} missed pairs; the partition is not exhaustive"
        )
    return result


def extract_hard_examples(missed: pl.DataFrame, max_examples: int = 100) -> list:
    """Extract a representative sample of difficult missed pairs."""
    if missed.height == 0:
        return []

    sample = missed.sample(n=min(max_examples, missed.height), seed=42)
    examples = []
    for row in sample.iter_rows(named=True):
        examples.append({
            "s1_entity_id": row.get("row_id"),
            "s1_business_name": row.get("business_name_s1"),
            "s1_business_address": row.get("business_address_s1"),
            "s1_country": row.get("country_s1"),
            "s1_name_norm": row.get("name_norm_s1"),
            "s1_addr_norm": row.get("addr_norm_s1"),
            "true_vendor_id": row.get("vendor_id"),
            # The raw vendor columns carry a `_raw` suffix -- the frame joins two
            # sources (normalised and raw) and the suffix is what keeps them
            # apart. An earlier revision read `business_name_vendor`, a column
            # that does not exist, so every published hard example carried a null
            # vendor name and address while the normalised ones were populated.
            "vendor_business_name": row.get("business_name_vendor_raw"),
            "vendor_business_address": row.get("business_address_vendor_raw"),
            "vendor_country": row.get("country_vendor"),
            "vendor_name_norm": row.get("name_norm_vendor"),
            "vendor_addr_norm": row.get("addr_norm_vendor"),
            "vendor_source": row.get("source_vendor"),
        })

    # A hard example exists to let a human look at the two records side by side.
    # If the vendor half is empty there is nothing to look at, and publishing it
    # anyway hides a broken join behind a well-formatted artifact. That is
    # precisely what happened: an earlier revision read
    # `business_name_vendor` while the frame carried
    # `business_name_vendor_raw`, so all 100 published examples had a null raw
    # vendor name -- and because the *normalised* fields were read correctly,
    # the examples still looked populated, so nothing looked wrong.
    #
    # The check is therefore on the raw name alone. Checking "raw AND normalised
    # are both missing" would have passed on exactly the broken artifact this is
    # meant to catch.
    blank_vendor_name = [
        e for e in examples if not (e["vendor_business_name"] or "").strip()
    ]
    if blank_vendor_name:
        raise ValueError(
            f"{len(blank_vendor_name)} of {len(examples)} hard examples have an empty "
            f"raw vendor name (first: vendor_id="
            f"{blank_vendor_name[0]['true_vendor_id']!r}, "
            f"source={blank_vendor_name[0]['vendor_source']!r}). Either the vendor "
            f"join is broken or these vendors genuinely have no name. The first is a "
            f"bug; the second is a data finding that belongs in failure_modes, not in "
            f"a list of examples that are supposed to be readable. Nothing is written."
        )
    return examples


def build_run_config(args: argparse.Namespace) -> dict:
    """Assemble the provenance block for a report, from the arguments given.

    This exists as a separate function for one reason. An earlier revision built
    this dict inline inside ``main`` and filled ``query_rows`` and ``seed`` from
    the *module defaults* rather than from ``args``. The run used
    ``--query-rows 500``, the artifact recorded ``200000``, and every
    within-sample percentage in the report was subsequently read as a
    population figure -- a diagnostic run published at full scale.

    The failure was invisible because the wrong number was a perfectly plausible
    number. Extracting the construction makes it directly testable: feed it a
    namespace with unusual values and check they come through unchanged.

    Args:
        args: The parsed CLI namespace.

    Returns:
        The provenance dict, recorded verbatim in the report as ``config``.
    """
    return {
        "experiment_id": args.experiment_id,
        "query_rows": args.query_rows,
        "seed": args.seed,
        "strategies": [name for name, _, _ in PLAN],
        "max_df": {name: max_df for name, _, max_df in PLAN if max_df is not None},
        "vendor_pool": "train source2 + source3 (full)",
        "bitmask_input": args.bitmask_input,
        "bitmask_output": args.bitmask_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-rows", type=int, default=DEFAULT_QUERY_ROWS,
                        help="S1 entities to sample. Recorded verbatim in the "
                             "report; a small value makes every percentage a "
                             "statement about the sample, not the corpus.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bitmask-output", type=str,
                        help="Save the per-pair strategy bitmask to parquet. This "
                             "is the only artifact that keeps per-strategy "
                             "attribution, and it lets the whole analysis be redone "
                             "via --bitmask-input without re-running retrieval.")
    parser.add_argument("--bitmask-input", type=str,
                        help="Analyse a saved bitmask instead of recomputing it. "
                             "Retrieval is skipped; the corpus is still read to "
                             "attach the source text for the examples.")
    parser.add_argument("--experiment-id", type=str, default="E012")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The run's parameters, assembled once from the arguments actually supplied
    # and recorded verbatim.
    run_config = build_run_config(args)

    t0 = time.time()
    paths = DatasetPaths.discover()

    # Load and normalize data (same as E011)
    split = load_split(paths, "train")
    s1_norm = normalize_columns(split["source1"])
    s1_raw = split["source1"]
    vendors_norm = pl.concat(
        [normalize_columns(split["source2"]), normalize_columns(split["source3"])],
        how="vertical",
    ).select("entity_id", "name_norm", "addr_norm", "country", "source", "has_address", "has_name")
    vendors_raw = pl.concat(
        [split["source2"], split["source3"]], how="vertical"
    ).select("entity_id", "business_name", "business_address", "country", "source")

    log(f"normalised S1 {s1_norm.height:,} / vendor pool {vendors_norm.height:,} in {time.time() - t0:.1f}s")

    queries = (
        s1_norm.sample(n=min(args.query_rows, s1_norm.height), seed=args.seed)
        .with_columns(pl.col("entity_id").alias("row_id"))
    )

    gt = load_ground_truth(paths.train_ground_truth).select("source1_entity_id", "match_ids")
    true_pairs = (
        queries.select("row_id")
        .join(gt, left_on="row_id", right_on="source1_entity_id")
        .explode("match_ids", empty_as_null=True)
        .filter(pl.col("match_ids").is_not_null())
        .select("row_id", pl.col("match_ids").alias("vendor_id"))
        .unique()
    )
    relevant_vendors = true_pairs["vendor_id"].unique()

    log(f"{queries.height:,} queries / {true_pairs.height:,} true pairs / "
        f"{queries['row_id'].n_unique():,} entities / {relevant_vendors.len():,} vendors")

    if args.bitmask_input:
        found = pl.read_parquet(args.bitmask_input)
        validate_bitmask(found, true_pairs.height)
        log(f"Loaded bitmask from {args.bitmask_input} ({found.height:,} rows); "
            f"retrieval not re-run")
    else:
        log("Computing retrieval bitmask...")
        found = compute_retrieval_bitmask(queries, vendors_norm, true_pairs, relevant_vendors)

        if args.bitmask_output:
            Path(args.bitmask_output).parent.mkdir(parents=True, exist_ok=True)
            found.write_parquet(args.bitmask_output)
            log(f"Bitmask saved to {args.bitmask_output}")

    log("Analyzing retrieval misses...")
    results = analyze_misses(
        found,
        true_pairs,
        vendors_norm,
        vendors_raw,
        s1_norm,
        s1_raw,
        run_config,
    )

    results["runtime_seconds"] = round(time.time() - t0, 1)
    results["peak_rss_gib"] = round(rss_gb(), 2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    log(f"Results written to {output_path}")
    log(f"Total runtime {results['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()