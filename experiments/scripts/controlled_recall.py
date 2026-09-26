"""Controlled retrieval correctness experiment.

Layer: retrieval (experiment driver)
See AGENTS.md §10 (The retrieval gate)

This experiment uses a vendor pool containing ALL true vendors for the queried
S1 entities plus random distractors. It is for debugging strategy behaviour,
NOT for measuring real recall ceiling.

The recall numbers here are upper bounds (true vendors guaranteed present);
the real ceiling is measured by Experiment B on the full corpus.
"""

from __future__ import annotations

import argparse
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

DEFAULT_QUERY_ROWS = 2_000
DEFAULT_DISTRACTORS = 50_000
DEFAULT_SEED = 17

# Document-frequency ceilings (from measure_blocking_recall.py sweep)
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


def build_vendor_pool(
    paths: DatasetPaths,
    true_vendor_ids: pl.Series,
    n_distractors: int,
    seed: int,
) -> pl.DataFrame:
    """Build vendor pool = all true vendors + random distractors.

    Args:
        paths: Dataset paths
        true_vendor_ids: Vendor IDs that are true matches for queried S1
        n_distractors: Number of random S2/S3 rows to add
        seed: Random seed

    Returns:
        Normalised vendor pool with guaranteed true vendors
    """
    # Load full S2 and S3
    s2 = pl.read_csv(paths.file("train", "train_source2.tsv"), separator="\t")
    s3 = pl.read_csv(paths.file("train", "train_source3.tsv"), separator="\t")

    # Select true vendors explicitly
    true_s2 = s2.filter(pl.col("entity_id").is_in(true_vendor_ids.implode()))
    true_s3 = s3.filter(pl.col("entity_id").is_in(true_vendor_ids.implode()))

    # Sample distractors from the remainder
    s2_remainder = s2.filter(~pl.col("entity_id").is_in(true_vendor_ids.implode()))
    s3_remainder = s3.filter(~pl.col("entity_id").is_in(true_vendor_ids.implode()))

    distractor_s2 = s2_remainder.sample(n=n_distractors // 2, seed=seed)
    distractor_s3 = s3_remainder.sample(n=n_distractors - n_distractors // 2, seed=seed + 1)

    # Concatenate and normalise
    pool = pl.concat([
        true_s2, true_s3, distractor_s2, distractor_s3
    ], how="vertical")

    log(f"pool collected: {pool.height:,} rows ({true_s2.height + true_s3.height:,} true + {n_distractors:,} distractors)")

    pool = normalize_columns(pool)
    # Add source column if missing (normalize_columns doesn't preserve it)
    if "source" not in pool.columns:
        pool = pool.with_columns(pl.lit(0).alias("source"))
    return pool.select("entity_id", "name_norm", "addr_norm", "country", "source")


def hits_for_strategy(
    strategy: str,
    builder,
    max_df: int | None,
    queries: pl.DataFrame,
    vendors: pl.DataFrame,
    true_pairs: pl.DataFrame,
) -> pl.DataFrame:
    """True pairs retrieved by one strategy. Returns ``(row_id, vendor_id)``."""
    vendor_keys = builder(vendors.lazy()).collect()
    if max_df is not None:
        df_counts = vendor_keys.group_by("key").agg(
            pl.col("row_id").n_unique().alias("df")
        )
        kept_keys = df_counts.filter(
            (pl.col("df") <= max_df) & (pl.col("df") >= 1)
        ).select("key")
        vendor_keys = vendor_keys.join(kept_keys, on="key", how="semi")

    vendor_side = vendor_keys.select(
        pl.col("row_id").alias("vendor_id"), "key"
    ).unique()

    query_keys = builder(queries.lazy()).collect()
    if max_df is not None:
        query_keys = query_keys.join(kept_keys, on="key", how="semi")
    query_keys = query_keys.select("row_id", "key").unique()

    hits = (
        true_pairs.join(query_keys, on="row_id", how="inner")
        .rename({"key": "k"})
        .join(vendor_side, left_on=["vendor_id", "k"], right_on=["vendor_id", "key"], how="semi")
        .select("row_id", "vendor_id")
        .unique()
    )
    log(f"  {strategy:<14} hits={hits.height:>6,}  vendor_keys={vendor_side.height:,}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-rows", type=int, default=DEFAULT_QUERY_ROWS)
    parser.add_argument("--distractors", type=int, default=DEFAULT_DISTRACTORS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    paths = DatasetPaths.discover()

    # Load and normalise S1
    split = load_split(paths, "train")
    s1 = normalize_columns(split["source1"])

    # Ground truth pairs
    gt = load_ground_truth(paths.train_ground_truth).select(
        "source1_entity_id", "match_ids"
    )
    gt_pairs = gt.explode("match_ids", empty_as_null=True).filter(
        pl.col("match_ids").is_not_null()
    ).select(
        pl.col("source1_entity_id").alias("row_id"),
        pl.col("match_ids").alias("vendor_id")
    ).unique()

    # Sample queries that have at least one true match
    s1_with_matches = s1.filter(pl.col("entity_id").is_in(gt_pairs["row_id"].unique()))
    queries = s1_with_matches.sample(
        n=min(args.query_rows, s1_with_matches.height), seed=args.seed
    ).with_columns(pl.col("entity_id").alias("row_id"))

    # True pairs for these queries
    true_pairs = queries.select("row_id").join(gt_pairs, on="row_id", how="inner")
    true_vendor_ids = true_pairs["vendor_id"].unique()

    log(
        f"{queries.height:,} queries / {true_pairs.height:,} true pairs / "
        f"{queries['row_id'].n_unique():,} entities / "
        f"{true_vendor_ids.len():,} distinct target vendors"
    )

    # Build vendor pool with guaranteed true vendors + distractors
    vendors = build_vendor_pool(paths, true_vendor_ids, args.distractors, args.seed)

    # Run each strategy
    found = true_pairs.with_columns(pl.lit(0).alias("bit"))
    per_strategy: list[dict] = []
    previous_union = 0.0

    print(f"\n{'strategy':16} {'max_df':>8} {'pair recall':>12} {'entity recall':>14} {'union gain':>11}")
    print("-" * 70)

    for index, (name, builder, max_df) in enumerate(PLAN):
        hits = hits_for_strategy(name, builder, max_df, queries, vendors, true_pairs)
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

        pair_r = int((found["bit"] > 0).sum()) / true_pairs.height
        entity_r = found.filter(pl.col("bit") > 0)["row_id"].n_unique() / queries.height
        union_gain = pair_r - previous_union
        previous_union = pair_r

        per_strategy.append({
            "strategy": name,
            "max_df": max_df,
            "pair_recall": round(pair_r, 6),
            "entity_recall": round(entity_r, 6),
            "union_gain": round(union_gain, 6),
            "hits": hits.height,
        })
        print(
            f"{name:16} {str(max_df) if max_df else '-':>8} {pair_r:>11.4f}  "
            f"{entity_r:>13.4f}  {union_gain:>+10.4f}",
            flush=True,
        )

    union = found.filter(pl.col("bit") > 0)
    u_pair = union.height / true_pairs.height
    u_ent = union["row_id"].n_unique() / queries.height
    print("-" * 70)
    print(f"{'UNION':16} {'-':>8} {u_pair:>11.4f}  {u_ent:>13.4f}")

    # Candidate count distribution (using the same plan format as generate_candidates expects)
    from team_diamond.retrieval.candidates import generate_candidates, RetrievalPlan
    plan_objs = [RetrievalPlan(name=p[0], builder=p[1], max_df=p[2]) for p in PLAN]
    candidates, report = generate_candidates(
        queries, vendors, plan=plan_objs, cap_per_query=None, allow_unmeasured_cap=False
    )
    cand_counts = candidates.group_by("s1_id").len()
    cand_stats = {
        "median": int(cand_counts["len"].median()),
        "p95": int(cand_counts["len"].quantile(0.95)),
        "max": int(cand_counts["len"].max()),
        "mean": float(cand_counts["len"].mean()),
    }

    # Candidate count by strategy
    cand_by_strat = candidates["strategies"].explode().value_counts().sort("count", descending=True)
    strat_counts = {
        row["strategies"]: row["count"]
        for row in cand_by_strat.iter_rows(named=True)
    }

    # Reachability
    print("\nREACHABILITY OF F0.5 (assumes PERFECT classifier on these candidates)")
    print(f"{'pair recall R':>16} {'best possible F0.5':>21}")
    for r in (u_pair, 0.60, 0.70, 0.75, 0.7917, 0.85, 0.90, 0.95):
        best = 1.25 * r / (0.25 + r)
        if abs(r - u_pair) < 1e-9:
            marker = "  <-- measured ceiling"
        elif abs(r - 0.7917) < 1e-4:
            marker = "  <-- R required for F0.5 = 0.95"
        else:
            marker = ""
        print(f"{r:>16.4f} {best:>21.4f}{marker}")

    # Results object
    results = {
        "experiment_id": "E010",
        "date": time.strftime("%Y-%m-%d"),
        "config": {
            "query_rows": args.query_rows,
            "distractors": args.distractors,
            "seed": args.seed,
            "strategies": [p[0] for p in PLAN],
        },
        "counts": {
            "queries": queries.height,
            "true_pairs": true_pairs.height,
            "entities": queries["row_id"].n_unique(),
            "distinct_true_vendors": true_vendor_ids.len(),
            "vendor_pool_size": vendors.height,
        },
        "per_strategy": per_strategy,
        "union": {
            "pair_recall": round(u_pair, 6),
            "entity_recall": round(u_ent, 6),
        },
        "candidate_stats": cand_stats,
        "candidate_by_strategy": strat_counts,
        "reachability": {
            "measured_pair_recall": round(u_pair, 6),
            "max_f05_at_perfect_precision": round(1.25 * u_pair / (0.25 + u_pair), 6),
            "f05_95_requires_recall": 0.7917,
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "peak_rss_gib": round(rss_gb(), 2),
    }

    output_path.write_text(json.dumps(results, indent=2))
    log(f"Results written to {output_path}")
    log(f"total runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()