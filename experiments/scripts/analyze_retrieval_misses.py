#!/usr/bin/env python
"""E012: Retrieval Miss Analysis for the frozen E011 evaluation sample.

Layer: retrieval (experiment driver)
See AGENTS.md §10 (The retrieval gate), §14 (Scale and cost)

This script analyzes the true pairs that were NOT retrieved by ANY of the
seven E011 strategies. It re-runs the retrieval on the exact frozen E011
sample (200K queries, seed=17, full 10.3M vendor corpus) to produce the
per-pair bitmask, then performs detailed failure-mode analysis.

The E011 JSON artifacts contain only aggregate statistics. This script
re-runs the retrieval (necessary to obtain per-pair bitmasks) and saves
them, then performs the requested analyses.

No new retrieval strategies are introduced. No model is trained.
Ground truth is used ONLY for evaluation and analysis.
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
    queries: pl.DataFrame,
    vendors_norm: pl.DataFrame,
    vendors_raw: pl.DataFrame,
    s1_norm: pl.DataFrame,
    s1_raw: pl.DataFrame,
    gt: pl.DataFrame,
) -> dict:
    """Perform all requested analyses on missed vs retrieved pairs."""
    n_pairs = true_pairs.height
    n_entities = true_pairs["row_id"].n_unique()

    # Union bitmask
    union = found.filter(pl.col("bit") > 0)
    u_pair = union.height / n_pairs
    u_ent = union["row_id"].n_unique() / n_entities

    # Missed pairs
    missed = found.filter(pl.col("bit") == 0)
    missed_count = missed.height

    # Per-strategy stats
    per_strategy = []
    for name, bit in STRATEGY_BITS.items():
        got = (found["bit"] & bit) > 0
        n_got = int(got.sum())
        per_strategy.append({
            "strategy": name,
            "pair_recall": round(n_got / n_pairs, 6),
            "hits": n_got,
        })

    # Distribution of strategies per retrieved pair
    strategy_counts = found.with_columns(
        pl.col("bit").map_elements(lambda x: bin(x).count("1"), return_dtype=pl.Int32).alias("n_strategies")
    )
    strategy_dist = strategy_counts["n_strategies"].value_counts().sort("n_strategies")
    strategy_distribution = {
        str(row["n_strategies"]): row["count"] for row in strategy_dist.iter_rows(named=True)
    }

    # Retrieve original data for missed pairs
    missed_pairs = missed.join(true_pairs, on=["row_id", "vendor_id"], how="inner")

    # Rename columns to avoid conflicts and ensure proper suffixes
    s1_norm_renamed = s1_norm.select(
        pl.col("entity_id").alias("row_id"),
        pl.col("business_name").alias("business_name_s1"),
        pl.col("business_address").alias("business_address_s1"),
        pl.col("country").alias("country_s1"),
        pl.col("name_norm").alias("name_norm_s1"),
        pl.col("addr_norm").alias("addr_norm_s1"),
        pl.col("has_address").alias("has_address_s1"),
        pl.col("has_name").alias("has_name_s1"),
    )
    vendors_norm_renamed = vendors_norm.select(
        pl.col("entity_id").alias("vendor_id"),
        pl.col("name_norm").alias("name_norm_vendor"),
        pl.col("addr_norm").alias("addr_norm_vendor"),
        pl.col("country").alias("country_vendor"),
        pl.col("source").alias("source_vendor"),
        pl.col("has_address").alias("has_address_vendor"),
        pl.col("has_name").alias("has_name_vendor"),
    )
    s1_raw_renamed = s1_raw.select(
        pl.col("entity_id").alias("row_id"),
        pl.col("business_name").alias("business_name_s1_raw"),
        pl.col("business_address").alias("business_address_s1_raw"),
    )
    vendors_raw_renamed = vendors_raw.select(
        pl.col("entity_id").alias("vendor_id"),
        pl.col("business_name").alias("business_name_vendor_raw"),
        pl.col("business_address").alias("business_address_vendor_raw"),
        pl.col("source").alias("source_vendor_raw"),
    )

    missed_with_data = missed_pairs.join(
        s1_norm_renamed, on="row_id", how="left"
    ).join(
        vendors_norm_renamed, on="vendor_id", how="left"
    ).join(
        s1_raw_renamed, on="row_id", how="left"
    ).join(
        vendors_raw.select(
            pl.col("entity_id").alias("vendor_id"),
            pl.col("business_name").alias("business_name_vendor_raw"),
            pl.col("business_address").alias("business_address_vendor_raw"),
            pl.col("source").alias("source_vendor_raw"),
        ),
        on="vendor_id", how="left"
    )

    # Also get retrieved pairs for comparison
    retrieved_pairs = union.join(true_pairs, on=["row_id", "vendor_id"], how="inner")
    retrieved_with_data = retrieved_pairs.join(
        s1_norm_renamed, on="row_id", how="left"
    ).join(
        vendors_norm_renamed, on="vendor_id", how="left"
    ).join(
        s1_raw_renamed, on="row_id", how="left"
    ).join(
        vendors_raw.select(
            pl.col("entity_id").alias("vendor_id"),
            pl.col("business_name").alias("business_name_vendor_raw"),
            pl.col("business_address").alias("business_address_vendor_raw"),
            pl.col("source").alias("source_vendor_raw"),
        ),
        on="vendor_id", how="left"
    )

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

    # F. Strategy coverage
    strategy_coverage = {
        "distribution": strategy_distribution,
        "missed_pairs_all_zero": missed_count,
        "retrieved_by_exactly_one": int(((found["bit"] > 0) & (found["bit"] < ALL_STRATEGIES_BIT)).sum()),
    }

    # G. Failure-mode categorization
    failure_modes = categorize_failure_modes(missed_with_data, retrieved_with_data)

    # H. Hard examples
    hard_examples = extract_hard_examples(missed_with_data, max_examples=100)

    # I. Per-country and per-source recall
    recall_by_country = country_breakdown.get("recall_by_country", {})
    recall_by_source = source_breakdown.get("recall_by_source", {})

    return {
        "experiment_id": "E012",
        "e011_baseline": {
            "query_rows": DEFAULT_QUERY_ROWS,
            "seed": DEFAULT_SEED,
            "true_pairs": n_pairs,
            "entities_with_matches": n_entities,
            "union_pair_recall": round(u_pair, 6),
            "union_entity_recall": round(u_ent, 6),
            "missed_pairs": missed_count,
            "missed_percentage": round(100 * missed_count / n_pairs, 2),
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
            s1_tokens=name_s1.str.split(" ").list.len(),
            v_tokens=name_v.str.split(" ").list.len(),
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
            s1_tokens=addr_s1.str.split(" ").list.len(),
            v_tokens=addr_v.str.split(" ").list.len(),
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
    """Categorize missed pairs into failure modes based on observable data."""
    if missed.height == 0:
        return {}

    # Compute basic indicators
    df = missed.with_columns([
        (pl.col("name_norm_s1") == pl.col("name_norm_vendor")).alias("name_exact"),
        (pl.col("addr_norm_s1") == pl.col("addr_norm_vendor")).alias("addr_exact"),
        (pl.col("name_norm_s1").str.len_chars().cast(pl.Float32) /
         pl.col("name_norm_vendor").str.len_chars().clip(lower_bound=1).cast(pl.Float32)).alias("name_len_ratio"),
        (pl.col("addr_norm_s1").str.len_chars().cast(pl.Float32) /
         pl.col("addr_norm_vendor").str.len_chars().clip(lower_bound=1).cast(pl.Float32)).alias("addr_len_ratio"),
        pl.col("has_name_s1").alias("s1_has_name"),
        pl.col("has_address_s1").alias("s1_has_addr"),
        pl.col("has_name_vendor").alias("v_has_name"),
        pl.col("has_address_vendor").alias("v_has_addr"),
    ])

    # Evaluate categories on the DataFrame
    total = missed.height
    
    categories_expr = {
        "both_missing": ((~pl.col("s1_has_name")) & (~pl.col("s1_has_addr")) |
                            (~pl.col("v_has_name")) & (~pl.col("v_has_addr"))),
        "name_strong_addr_weak": ((pl.col("name_exact")) & (~pl.col("addr_exact")) &
                                       pl.col("s1_has_addr") & pl.col("v_has_addr")),
        "addr_strong_name_weak": ((pl.col("addr_exact")) & (~pl.col("name_exact")) &
                                       pl.col("s1_has_name") & pl.col("v_has_name")),
        "both_weak_both_present": ((~pl.col("name_exact")) & (~pl.col("addr_exact")) &
                                        pl.col("s1_has_name") & pl.col("v_has_name") &
                                        pl.col("s1_has_addr") & pl.col("v_has_addr")),
        "missing_name_either_side": ((~pl.col("s1_has_name")) | (~pl.col("v_has_name"))),
        "missing_address_either_side": ((~pl.col("s1_has_addr")) | (~pl.col("v_has_addr"))),
    }

    # Evaluate on the DataFrame
    categories = {}
    for name, expr in categories_expr.items():
        val = df.select(expr.sum()).item()
        categories[name] = int(val)

    total = missed.height
    return {
        "total": total,
        "categories": {k: {"count": v, "percentage": round(100 * v / total, 2)} for k, v in categories.items()},
        "note": "Categories are not mutually exclusive; a pair may match multiple categories."
    }


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
            "vendor_business_name": row.get("business_name_vendor"),
            "vendor_business_address": row.get("business_address_vendor"),
            "vendor_country": row.get("country_vendor"),
            "vendor_name_norm": row.get("name_norm_vendor"),
            "vendor_addr_norm": row.get("addr_norm_vendor"),
            "vendor_source": row.get("source_vendor"),
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-rows", type=int, default=DEFAULT_QUERY_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bitmask-output", type=str, help="Optional: save bitmask to parquet")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Compute retrieval bitmask
    log("Computing retrieval bitmask...")
    found = compute_retrieval_bitmask(queries, vendors_norm, true_pairs, relevant_vendors)

    # Save bitmask if requested
    if args.bitmask_output:
        Path(args.bitmask_output).parent.mkdir(parents=True, exist_ok=True)
        found.write_parquet(args.bitmask_output)
        log(f"Bitmask saved to {args.bitmask_output}")

    # Analyze
    log("Analyzing retrieval misses...")
    results = analyze_misses(
        found, true_pairs, queries,
        vendors_norm,  # normalized vendor pool
        pl.concat([split["source2"], split["source3"]], how="vertical")
            .select("entity_id", "business_name", "business_address", "country", "source"),  # raw
        normalize_columns(split["source1"]),  # normalized S1
        split["source1"],  # raw S1
        gt
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