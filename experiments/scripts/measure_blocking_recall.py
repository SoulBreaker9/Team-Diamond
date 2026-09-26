"""Measure candidate recall per blocking strategy. The retrieval gate.

Layer: retrieval (experiment driver)
See AGENTS.md §10 (The retrieval gate), §16 (Experiment discipline)

The question
------------
Which blocking keys connect an S1 entity to its true matches, and at what
candidate cost? Nothing downstream can exceed this recall, so it is measured
before any matcher is built.

The method, and why it is shaped this way
-----------------------------------------
Recall is computed on a **sampled set of S1 queries against the full vendor
pool**. The vendor pool is never sampled: sampling it would inflate recall by
removing the distractors that make the problem hard.

Recall uses a **semi-join**, not a materialised candidate set::

    hit = true_pairs.join(s1_keys, on="row_id")
                  .join(vendor_keys.rename(...), on=["vendor_id","key"], how="semi")

A semi-join answers "does this pair share a key?" without expanding to every
candidate. That matters: the full candidate set for a token strategy is orders
of magnitude larger than the true pairs, and materialising it only to count it
would cost memory for no extra information. Candidate *counts* are therefore
measured separately, on a smaller query subsample, where the full set is
affordable.

Sample sizes are printed with every result. Recall on a sample is an estimate,
and reporting it without the sample size would present a PRELIMINARY number as
if it were MEASURED (AGENTS.md §0).

Usage
-----
    uv run python experiments/scripts/measure_blocking_recall.py
    uv run python experiments/scripts/measure_blocking_recall.py --query-rows 400000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
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
    document_frequency,
    filter_by_document_frequency,
)

# Sample size for the recall estimate. Larger is better; this is bounded by the
# local machine, not by principle. On a 15 GB / 20-core box 200k queries is
# comfortable. The real execution environment has more, and this should be
# re-run there before any number is treated as final.
DEFAULT_QUERY_ROWS = 200_000

# Separate, smaller sample for candidate-count distribution, where the full
# candidate set IS materialised.
DEFAULT_COUNT_ROWS = 20_000

# Document-frequency ceilings swept for the token strategies. A token carried by
# more than max_df rows cannot produce a usable block. Sweeping rather than
# fixing, because the right ceiling trades recall against a cost that has not
# been measured yet (AGENTS.md §10: caps are experimental parameters).
DF_SWEEP = (10, 50, 200, 1000, 5000)


@dataclass(frozen=True)
class StrategyResult:
    """Outcome of one blocking strategy at one document-frequency ceiling."""

    strategy: str
    max_df: int | None
    pair_recall: float
    entity_recall: float
    recall_by_country: dict[str, float]
    key_rows: int
    distinct_keys: int


def _entity_recall(hits: pl.DataFrame, total_pairs: int, entities: int) -> float:
    """Fraction of S1 entities with at least one true match retrieved.

    Reported alongside pair recall because the competition metric is
    entity-level. An entity whose match count is 1 behaves very differently from
    one with 4, and pair recall alone hides that.
    """
    entities_with_hit = hits.select("row_id").unique().height
    _ = total_pairs
    return entities_with_hit / entities


def _recall_by_country(
    s1: pl.DataFrame, hits: pl.DataFrame, entities_by_country: dict[str, int]
) -> dict[str, float]:
    """Entity-level recall per country, using the entity denominator.

    Note the limitation: train contains only US and India, so **France recall is
    not measurable here at all**. France is 14.98% of test S1 and absent from
    train. Any strategy tuned on this number is tuned on 85% of the test
    distribution. That gap is a property of the challenge, not something this
    script can fix, and it is why AGENTS.md §8.5 forbids country branches.
    """
    hit_countries = (
        s1.join(hits.select("row_id").unique(), on="row_id", how="semi")
        .group_by("country")
        .agg(pl.col("row_id").n_unique().alias("hit"))
    )
    out = {}
    for row in hit_countries.iter_rows(named=True):
        denom = entities_by_country.get(row["country"], 0)
        out[row["country"]] = row["hit"] / denom if denom else float("nan")
    return out


def _recall_for_strategy(
    name: str,
    s1: pl.DataFrame,
    s1_keys: pl.DataFrame,
    vendor_keys: pl.DataFrame,
    true_pairs: pl.DataFrame,
    max_df: int | None,
) -> StrategyResult:
    """Pair-level and entity-level recall for one strategy."""
    # (row_id, vendor_id, key) for every key carried by the S1 side of a true
    # pair, then a semi-join asking "does the vendor carry that same key?".
    # The semi-join is on (vendor_id, key) because the vendor frame has no
    # row_id -- and a semi-join is what keeps the intermediate at
    # |true pairs| x |keys per entity| instead of the full candidate set.
    s1_side = true_pairs.join(s1_keys, on="row_id", how="inner").rename(
        {"key": "s1_key"}
    )
    vendor_side = vendor_keys.rename({"key": "s1_key"})
    hits = s1_side.join(
        vendor_side, on=["vendor_id", "s1_key"], how="semi"
    ).select("row_id", "vendor_id").unique()

    entities = s1.height
    by_country = {
        row["country"]: row["n"]
        for row in s1.group_by("country")
        .agg(pl.col("row_id").n_unique().alias("n"))
        .iter_rows(named=True)
    }
    return StrategyResult(
        strategy=name,
        max_df=max_df,
        pair_recall=hits.height / true_pairs.height,
        entity_recall=_entity_recall(hits, true_pairs.height, entities),
        recall_by_country=_recall_by_country(s1, hits, by_country),
        key_rows=s1_keys.height + vendor_keys.height,
        distinct_keys=vendor_keys["key"].n_unique(),
    )


def _candidate_counts(
    s1: pl.DataFrame,
    s1_keys: pl.DataFrame,
    vendor_keys: pl.DataFrame,
    max_df: int | None,
) -> dict[str, float]:
    """Candidate-count distribution for a small query sample.

    Materialises the full candidate set, so this is why the query sample is an
    order of magnitude smaller than the recall sample. The reported numbers
    describe the S1 subsample, not the whole split, and the script says so.
    """
    candidates = s1_keys.join(vendor_keys, on="key", how="inner", suffix="_v")
    per_entity = (
        candidates.group_by("row_id")
        .agg(pl.col("vendor_id_v").n_unique().alias("n_cand"))
        .select("n_cand")
    )
    if per_entity.height == 0:
        return {"median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    stats = per_entity.select(
        pl.col("n_cand").median().alias("median"),
        pl.col("n_cand").quantile(0.95).alias("p95"),
        pl.col("n_cand").max().alias("max"),
        pl.col("n_cand").mean().alias("mean"),
    ).row(0, named=True)
    _ = max_df
    return {k: float(v) for k, v in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-rows", type=int, default=DEFAULT_QUERY_ROWS)
    parser.add_argument("--count-rows", type=int, default=DEFAULT_COUNT_ROWS)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    paths = DatasetPaths.discover()
    print(f"dataset: {paths.root}")
    print(
        f"recall sample: {args.query_rows:,} S1 entities (seed {args.seed}); "
        f"count sample: {args.count_rows:,} S1 entities. Vendor pool: ALL rows.\n"
    )

    t0 = time.time()
    split = load_split(paths, "train")
    print(f"loaded train sources in {time.time() - t0:.1f}s")

    s1 = normalize_columns(split["source1"])
    s2 = normalize_columns(split["source2"])
    s3 = normalize_columns(split["source3"])
    vendors = pl.concat([s2, s3], how="vertical").with_columns(
        pl.lit("v").alias("side")
    )
    print(
        f"normalised: S1 {s1.height:,} / S2 {s2.height:,} / S3 {s3.height:,} "
        f"-> vendor pool {vendors.height:,}\n"
    )

    # --- queries and ground truth, restricted to the sample -------------------
    # `row_id` is the identifier the key builders emit, so the query frame needs
    # it under that name too. entity_id is preserved alongside it -- IDs are
    # opaque and must never be renamed away in the output (AGENTS.md §8.3).
    queries = s1.sample(n=min(args.query_rows, s1.height), seed=args.seed).with_columns(
        pl.col("entity_id").alias("row_id")
    )
    gt = load_ground_truth(paths.train_ground_truth).select(
        "source1_entity_id", "match_ids"
    )
    true_pairs = (
        queries.select("row_id")
        .join(gt, left_on="row_id", right_on="source1_entity_id", how="inner")
        .explode("match_ids")
        .filter(pl.col("match_ids").is_not_null() & (pl.col("match_ids") != ""))
        .select("row_id", pl.col("match_ids").alias("vendor_id"))
        .unique()
    )
    print(
        f"sampled {queries.height:,} S1 queries covering {true_pairs.height:,} "
        f"true pairs ({true_pairs['row_id'].n_unique():,} entities, "
        f"{true_pairs.height / max(queries.height, 1):.2f} matches/entity)\n"
    )

    # --- strategies ----------------------------------------------------------
    # Token strategies are swept over document-frequency ceilings. Whole-string
    # strategies have no df parameter and are run once.
    whole_string = {
        "exact_name": build_exact_name_keys,
        "alnum_name": build_alnum_name_keys,
        "sorted_name": build_sorted_name_keys,
        "address_exact": build_address_exact_keys,
    }
    token_strategy = {
        "name_token": build_name_token_keys,
        "address_token": build_address_token_keys,
        "numeric_token": build_numeric_keys,
    }

    vendor_key_cache: dict[tuple[str, int | None], pl.DataFrame] = {}

    def vendor_keys_for(name: str, max_df: int | None) -> pl.DataFrame:
        cache_key = (name, max_df)
        if cache_key not in vendor_key_cache:
            keys = token_strategy[name](vendors)
            if max_df is not None:
                keys = filter_by_document_frequency(keys, max_df=max_df)
            vendor_key_cache[cache_key] = keys.select("row_id", "key").rename(
                {"row_id": "vendor_id"}
            )
        return vendor_key_cache[cache_key]

    results: list[StrategyResult] = []

    print("=" * 108)
    print(
        f"{'strategy':16} {'max_df':>8} {'pair recall':>12} {'entity recall':>14} "
        f"{'vendor keys':>13} {'key rows':>12}  per-country entity recall"
    )
    print("=" * 108)

    for name, builder in whole_string.items():
        q_keys = builder(queries)
        v_keys = builder(vendors).rename({"row_id": "vendor_id"})
        result = _recall_for_strategy(name, queries, q_keys, v_keys, true_pairs, None)
        results.append(result)
        detail = "  ".join(
            f"{c} {r:.3f}" for c, r in sorted(result.recall_by_country.items())
        )
        print(
            f"{name:16} {'-':>8} {result.pair_recall:>11.4f}  "
            f"{result.entity_recall:>13.4f}  {result.distinct_keys:>13,} "
            f"{result.key_rows:>12,}  {detail}"
        )

    for name, builder in token_strategy.items():
        raw = builder(vendors)
        for max_df in DF_SWEEP:
            q_raw = builder(queries)
            if max_df is None:
                v_keys = raw.rename({"row_id": "vendor_id"}).select("vendor_id", "key")
                q_keys = q_raw
            else:
                v_keys = vendor_keys_for(name, max_df)
                q_keys = filter_by_document_frequency(q_raw, max_df=max_df)
            result = _recall_for_strategy(
                name, queries, q_keys, v_keys, true_pairs, max_df
            )
            results.append(result)
            detail = "  ".join(
                f"{c} {r:.3f}" for c, r in sorted(result.recall_by_country.items())
            )
            print(
                f"{name:16} {max_df:>8,} {result.pair_recall:>11.4f}  "
                f"{result.entity_recall:>13.4f}  {result.distinct_keys:>13,} "
                f"{result.key_rows:>12,}  {detail}"
            )

    # --- union ---------------------------------------------------------------
    print("=" * 108)
    print("\nUNION of the recall-maximising configurations")
    best = {}
    for name in token_strategy:
        candidates = [r for r in results if r.strategy == name]
        best[name] = max(candidates, key=lambda r: r.pair_recall)

    union_s1 = None
    union_vendor = None
    for name, builder in whole_string.items():
        q = builder(queries)
        v = builder(vendors).rename({"row_id": "vendor_id"}).select("vendor_id", "key")
        union_s1 = q if union_s1 is None else pl.concat([union_s1, q])
        union_vendor = v if union_vendor is None else pl.concat([union_vendor, v])
    for name, result in best.items():
        q = filter_by_document_frequency(
            token_strategy[name](queries), max_df=result.max_df
        )
        v = vendor_keys_for(name, result.max_df)
        union_s1 = pl.concat([union_s1, q])
        union_vendor = pl.concat([union_vendor, v])

    union_s1 = union_s1.unique()
    union_vendor = union_vendor.unique()
    union_result = _recall_for_strategy(
        "UNION", queries, union_s1, union_vendor, true_pairs, None
    )
    detail = "  ".join(f"{c} {r:.3f}" for c, r in sorted(union_result.recall_by_country.items()))
    print(
        f"{'UNION':16} {'-':>8} {union_result.pair_recall:>11.4f}  "
        f"{union_result.entity_recall:>13.4f}  {union_result.distinct_keys:>13,} "
        f"{union_result.key_rows:>12,}  {detail}"
    )

    # --- candidate cost ------------------------------------------------------
    print(f"\nCANDIDATE COUNTS (materialised on {args.count_rows:,} S1 queries)")
    small = queries.sample(n=min(args.count_rows, queries.height), seed=args.seed)
    print(f"{'strategy':16} {'max_df':>8} {'median':>9} {'p95':>10} {'max':>10} {'mean':>10}")
    print("-" * 68)
    for name, builder in whole_string.items():
        counts = _candidate_counts(
            small, builder(small), builder(vendors).rename({"row_id": "vendor_id"}), None
        )
        print(
            f"{name:16} {'-':>8} {counts['median']:>9.0f} {counts['p95']:>10.0f} "
            f"{counts['max']:>10,.0f} {counts['mean']:>10.1f}"
        )
    for name, result in best.items():
        counts = _candidate_counts(
            small,
            filter_by_document_frequency(token_strategy[name](small), max_df=result.max_df),
            vendor_keys_for(name, result.max_df),
            result.max_df,
        )
        print(
            f"{name:16} {result.max_df:>8,} {counts['median']:>9.0f} "
            f"{counts['p95']:>10.0f} {counts['max']:>10,.0f} {counts['mean']:>10.1f}"
        )
    counts = _candidate_counts(small, union_s1, union_vendor, None)
    print(
        f"{'UNION':16} {'-':>8} {counts['median']:>9.0f} {counts['p95']:>10.0f} "
        f"{counts['max']:>10,.0f} {counts['mean']:>10.1f}"
    )

    # --- what the df ceiling costs -------------------------------------------
    print("\nDF-CEILING COST for the token strategies (recall lost by capping)")
    for name in token_strategy:
        rows = [r for r in results if r.strategy == name]
        uncapped = rows[-1].max_df
        print(f"  {name}:")
        for r in rows:
            if r.max_df == uncapped:
                continue
            print(
                f"    max_df={r.max_df:>6,}  pair recall {r.pair_recall:.4f}  "
                f"keys {r.distinct_keys:>10,}  key rows {r.key_rows:>12,}"
            )

    print(
        f"\ntotal runtime {time.time() - t0:.1f}s\n"
        "NOT MEASURED YET: France recall. Train has no French rows, so this "
        "experiment cannot see the 14.98% of test S1 that is French."
    )


if __name__ == "__main__":
    main()
