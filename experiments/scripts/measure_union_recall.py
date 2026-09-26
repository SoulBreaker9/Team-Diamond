"""Measure candidate recall of a UNION of blocking strategies. The ceiling.

Layer: retrieval (experiment driver)
See AGENTS.md §10 (The retrieval gate), §14 (Scale and cost)

Why this script exists separately
---------------------------------
``measure_blocking_recall.py`` sweeps every strategy against every
document-frequency ceiling, which is the right way to *choose* strategies. This
script answers one narrower question: **if I union the strategies, what recall
ceiling does the matcher face?** That number decides what $F_{0.5}$ is reachable
at all, so it is needed before spending any effort on the model.

It is worth stating the arithmetic up front, because it governs everything
downstream. With perfect precision and pair recall R, the best achievable
$F_{0.5}$ is::

    1.25 * R / (0.25 + R)

Solving for 0.95 gives **R >= 0.7917**. No classifier, however good, can exceed
the recall of the candidate set it scores. So a 0.95 target is a *retrieval*
target before it is a modelling target.

Memory design — this script is deliberately built to not take the machine down
-----------------------------------------------------------------------------
The first version of this script was OOM-killed on a 15 GiB box, and took the
desktop session with it. Three changes fix that, and the reasoning generalises
to every retrieval experiment we will run at this scale:

1. **Never accumulate key tables.** Measuring a union does not require holding
   the union. All that is needed is, for each true pair, *which* strategies
   retrieved it. That is a small ``(row_id, vendor_id, strategy_bitmask)``
   table of ~690k rows, independent of corpus size. The old script instead did
   ``union = concat(union, keys)`` seven times over 10.3M vendors, which is the
   entire corpus multiplied by seven, held live, and is what killed it.
2. **Compute document frequency on the full pool, but only *materialise* keys
   for the vendors that matter.** DF is a corpus statistic: computing it on a
   subsample would make the filter meaningless. But only ~690k of the 10.3M
   vendors are true matches of the sampled queries, so the *keys* are only
   needed for those. This is a 15x reduction and it does not weaken the
   measurement, because DF still sees every vendor.
3. **Stream the group-by, and free eagerly.** DF is computed with
   ``collect(engine="streaming")`` so the exploded token frame is never fully
   resident, and each strategy's frames are dropped before the next begins.

RSS is printed at every stage so a regression in peak memory is visible in the
log rather than inferred from a crash.

Usage
-----
    uv run python experiments/scripts/measure_union_recall.py --query-rows 200000
"""

from __future__ import annotations

import argparse
import gc
import time

import polars as pl

from team_diamond.data import DatasetPaths, load_ground_truth, load_split
from team_diamond.preprocessing.normalize_fast import normalize_columns
from team_diamond.retrieval.candidates import generate_candidates, RetrievalPlan
from team_diamond.retrieval.keys import (
    build_address_exact_keys,
    build_address_token_keys,
    build_alnum_name_keys,
    build_exact_name_keys,
    build_name_token_keys,
    build_numeric_keys,
    build_sorted_name_keys,
    document_frequency,
)

DEFAULT_QUERY_ROWS = 200_000
DEFAULT_CAP_SWEEP_SAMPLE = 10_000  # Vendor pool size for local cap sweep

# Document-frequency ceilings for the token strategies. Chosen from the sweep in
# measure_blocking_recall.py; recorded here as explicit values so this script is
# reproducible on its own. These remain PROPOSALS until the retrieval gate in
# AGENTS.md §10 is satisfied on the full corpus.
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

# Cap sweep values for Experiment B. These are PROPOSALS until measured.
CAP_SWEEP: list[int] = [100, 150, 200, 250, 300, 400]

def _plan_objects() -> list:
    """Get PLAN as RetrievalPlan objects for generate_candidates."""
    return [RetrievalPlan(name=p[0], builder=p[1], max_df=p[2]) for p in PLAN]


def rss_gb() -> float:
    """Resident set size of this process in GiB, for the memory log."""
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
    """True pairs retrieved by one strategy. Returns ``(row_id, vendor_id)``.

    The vendor side is materialised **only** for ``relevant_vendors``. That is
    sound because a hit requires the true vendor to be present, so restricting
    to the vendors that appear in ``true_pairs`` cannot remove a hit. Document
    frequency, which decides which keys survive, is still computed over the
    entire pool.
    """
    vendor_keys_lazy = builder(vendors)  # lazy: never fully materialised

    kept: pl.DataFrame | None = None
    if max_df is not None:
        # Full-pool DF, streamed. Only the (key, df) aggregate is materialised,
        # which is orders of magnitude smaller than the exploded key frame.
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
    log(f"  {strategy:<14} hits={hits.height:>9,}  vendor keys materialised")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-rows", type=int, default=DEFAULT_QUERY_ROWS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cap-sweep", action="store_true", default=False,
                        help="Run cap sweep after uncapped measurement. "
                             "Defaults to small vendor sample unless --full-corpus is set.")
    parser.add_argument("--full-corpus", action="store_true", default=False,
                        help="Run cap sweep on full 10.3M vendor corpus. "
                             "REQUIRES SAGEMAKER WITH 32GB RAM. Do not use locally.")
    args = parser.parse_args()

    paths = DatasetPaths.discover()
    t0 = time.time()
    split = load_split(paths, "train")
    s1 = normalize_columns(split["source1"])
    vendors = pl.concat(
        [
            normalize_columns(split["source2"]),
            normalize_columns(split["source3"]),
        ],
        how="vertical",
    ).select("entity_id", "name_norm", "addr_norm", "source")
    log(
        f"normalised S1 {s1.height:,} / vendor pool {vendors.height:,} "
        f"in {time.time() - t0:.1f}s"
    )
    vendors_lazy = vendors.lazy()

    queries = (
        s1.sample(n=min(args.query_rows, s1.height), seed=args.seed)
        .with_columns(pl.col("entity_id").alias("row_id"))
    )
    gt = load_ground_truth(paths.train_ground_truth).select(
        "source1_entity_id", "match_ids"
    )
    true_pairs = (
        queries.select("row_id")
        .join(gt, left_on="row_id", right_on="source1_entity_id")
        .explode("match_ids", empty_as_null=True)
        .filter(pl.col("match_ids").is_not_null())
        .select("row_id", pl.col("match_ids").alias("vendor_id"))
        .unique()
    )
    n_entities = true_pairs["row_id"].n_unique()
    n_pairs = true_pairs.height
    relevant_vendors = true_pairs["vendor_id"].unique()
    log(
        f"{queries.height:,} queries / {n_pairs:,} true pairs / "
        f"{n_entities:,} entities with >=1 match / "
        f"{relevant_vendors.len():,} distinct target vendors"
    )

    # strategy_bit: one bit per strategy, so the union is a bitwise OR and the
    # per-strategy attribution survives without keeping any key table around.
    found = true_pairs.with_columns(pl.lit(0).alias("bit"))
    per_strategy: list[tuple[str, int, float, float]] = []
    previous_union = 0.0

    print(f"\n{'strategy':16} {'max_df':>8} {'pair recall':>12} {'entity recall':>14} {'union gain':>11}")
    print("-" * 70)

    for index, (name, builder, max_df) in enumerate(PLAN):
        hits = hits_for_strategy(
            name, builder, max_df, queries, vendors_lazy, true_pairs, relevant_vendors
        )
        bit = 1 << index
        # OR the strategy's bit into every pair it retrieved. A left join plus
        # fill_null(0) leaves untouched pairs at their existing bitmask, so the
        # accumulated `found` frame is the union so far -- in ~690k rows rather
        # than in seven full-corpus key tables.
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

        pair_r = int((found["bit"] > 0).sum()) / n_pairs
        entity_r = (
            found.filter(pl.col("bit") > 0)["row_id"].n_unique() / n_entities
        )
        union_gain = pair_r - previous_union
        previous_union = pair_r
        per_strategy.append((name, bit, pair_r, entity_r))
        print(
            f"{name:16} {str(max_df) if max_df else '-':>8} {pair_r:>11.4f}  "
            f"{entity_r:>13.4f}  {union_gain:>+10.4f}",
            flush=True,
        )

    union = found.filter(pl.col("bit") > 0)
    u_pair = union.height / n_pairs
    u_ent = union["row_id"].n_unique() / n_entities
    print("-" * 70)
    print(f"{'UNION':16} {'-':>8} {u_pair:>11.4f}  {u_ent:>13.4f}")

    # --- the ceiling ---------------------------------------------------------
    print("\nREACHABILITY OF F0.5 (assumes a PERFECT classifier on these candidates)")
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

    # --- where recall is lost ------------------------------------------------
    print("\nWHERE THE REMAINING RECALL IS LOST")
    missed = found.filter(pl.col("bit") == 0)
    print(
        f"  missed true pairs: {missed.height:,} of {n_pairs:,} "
        f"({100 * missed.height / n_pairs:.2f}%)"
    )
    print("  strategy that came closest, for pairs that strategy did not retrieve:")
    for index, (name, bit, _, _) in enumerate(per_strategy):
        got = (found["bit"] & bit) > 0
        n_got = int(got.sum())
        print(
            f"    {name:16} retrieved {n_got:>9,}  "
            f"({100 * n_got / n_pairs:6.2f}% of all true pairs)"
        )

    # How much of the loss is addressable: a pair retrieved by exactly one
    # strategy was nearly reachable, a pair retrieved by none was not.
    total_bits = (1 << len(PLAN)) - 1
    print(
        f"  pairs no strategy retrieved:      "
        f"{int((found['bit'] == 0).sum()):>9,}"
    )
    print(
        f"  pairs retrieved by exactly one:   "
        f"{int(((found['bit'] > 0) & (found['bit'] < total_bits)).sum()):>9,}"
    )
    # --- cap sweep -----------------------------------------------------------
    # The uncapped measurement above is the reference ceiling.
    # Now measure recall at each cap value to quantify the recall cost of capping.
    if not args.cap_sweep:
        log("Cap sweep skipped (use --cap-sweep to enable)")
    else:
        log("Starting cap sweep...")

        # Determine vendor pool for cap sweep
        if args.full_corpus:
            # Full corpus - requires 32GB+ RAM, only safe on SageMaker
            log("WARNING: Running cap sweep on FULL 10.3M vendor corpus. "
                "This requires 32GB+ RAM. Ensure you are on SageMaker ml.r5.xlarge.")
            vendors_for_gen = vendors
            cap_sweep_label = "FULL CORPUS"
        else:
            # Local safe mode: use a small vendor sample
            log(f"Local cap sweep mode: using {DEFAULT_CAP_SWEEP_SAMPLE:,} vendor sample "
                f"(not full corpus). Use --full-corpus on SageMaker for real results.")
            vendors_for_gen = vendors.sample(n=min(DEFAULT_CAP_SWEEP_SAMPLE, vendors.height), seed=args.seed + 100)
            cap_sweep_label = f"SAMPLE ({DEFAULT_CAP_SWEEP_SAMPLE:,} vendors)"

        cap_results: list[dict] = []

        # queries already has entity_id, name_norm, addr_norm, country
        queries_for_gen = queries.select("entity_id", "name_norm", "addr_norm", "country")

        plan_objs = _plan_objects()

        for cap in CAP_SWEEP:
            log(f"  Cap sweep ({cap_sweep_label}): cap_per_query={cap}")
            candidates, report = generate_candidates(
                queries_for_gen,
                vendors_for_gen,
                plan=plan_objs,
                cap_per_query=cap,
                allow_unmeasured_cap=True,  # This is the measurement sweep itself
                query_id_column="entity_id",
                vendor_id_column="entity_id",
            )

            # Join candidates with true_pairs to measure recall at this cap
            # true_pairs uses row_id (entity_id), candidates uses s1_id (entity_id)
            hits = true_pairs.join(
                candidates.select("s1_id", "vendor_id"),
                left_on=["row_id", "vendor_id"],
                right_on=["s1_id", "vendor_id"],
                how="semi"
            )
            pair_r = hits.height / n_pairs
            entity_r = hits["row_id"].n_unique() / n_entities

            cap_results.append({
                "cap": cap,
                "pair_recall": round(pair_r, 6),
                "entity_recall": round(entity_r, 6),
                "candidates": candidates.height,
                "candidates_per_query_median": int(candidates.group_by("s1_id").len()["len"].median()),
                "candidates_per_query_p95": int(candidates.group_by("s1_id").len()["len"].quantile(0.95)),
                "candidates_per_query_max": int(candidates.group_by("s1_id").len()["len"].max()),
                "dropped_by_cap": report.n_dropped_by_cap,
            })
            log(f"    cap={cap:>4}  pair_recall={pair_r:.4f}  entity_recall={entity_r:.4f}  candidates={candidates.height:,}")

        # Print cap sweep summary
        print(f"\nCAP SWEEP RESULTS ({cap_sweep_label} - uncapped reference ceiling above)")
        print(f"{'cap':>6} {'pair recall':>12} {'entity recall':>14} {'candidates':>12} {'median/q':>10} {'p95/q':>10} {'max/q':>8} {'dropped':>10}")
        print("-" * 90)
        for cr in cap_results:
            print(
                f"{cr['cap']:>6} {cr['pair_recall']:>11.4f}  {cr['entity_recall']:>13.4f}  "
                f"{cr['candidates']:>12,}  {cr['candidates_per_query_median']:>10,}  "
                f"{cr['candidates_per_query_p95']:>10,}  {cr['candidates_per_query_max']:>8,}  "
                f"{cr['dropped_by_cap']:>10,}"
            )

    # --- the ceiling (uncapped) ---------------------------------------------
    print("\nREACHABILITY OF F0.5 (assumes a PERFECT classifier on these candidates)")
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

    # --- where recall is lost ------------------------------------------------
    print("\nWHERE THE REMAINING RECALL IS LOST")
    missed = found.filter(pl.col("bit") == 0)
    print(
        f"  missed true pairs: {missed.height:,} of {n_pairs:,} "
        f"({100 * missed.height / n_pairs:.2f}%)"
    )
    print("  strategy that came closest, for pairs that strategy did not retrieve:")
    for index, (name, bit, _, _) in enumerate(per_strategy):
        got = (found["bit"] & bit) > 0
        n_got = int(got.sum())
        print(
            f"    {name:16} retrieved {n_got:>9,}  "
            f"({100 * n_got / n_pairs:6.2f}% of all true pairs)"
        )

    # How much of the loss is addressable: a pair retrieved by exactly one
    # strategy was nearly reachable, a pair retrieved by none was not.
    total_bits = (1 << len(PLAN)) - 1
    print(
        f"  pairs no strategy retrieved:      "
        f"{int((found['bit'] == 0).sum()):>9,}"
    )
    print(
        f"  pairs retrieved by exactly one:   "
        f"{int(((found['bit'] > 0) & (found['bit'] < total_bits)).sum()):>9,}"
    )
    del found, union, missed, vendors, s1, queries, true_pairs
    gc.collect()
    log(f"total runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
