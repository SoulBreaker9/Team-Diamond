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
import json
import time
from pathlib import Path
from typing import Final

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
from team_diamond.retrieval.recall_stats import (
    RECALL_REQUIRED_FOR_F05_95,
    recall_report as build_recall_report,
    render_recall_table,
    validate_bitmask,
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

#: Strategies that carry a document-frequency ceiling. Only these accept
#: `--ceiling` overrides. The four exact strategies have no ceiling by design:
#: an exact key either matches or it does not, so there is no frequency
#: tradeoff to sweep. (If a future experiment wants a ceiling on an exact
#: strategy, extend this tuple deliberately — never by accident.)
CEILING_STRATEGIES: Final = ("name_token", "address_token", "numeric_token")


def parse_ceiling_overrides(values: list[str] | None) -> dict[str, int]:
    """Parse `--ceiling NAME=INT` overrides into ``{strategy: max_df}``.

    This is the E014 mechanism: each variant is the baseline PLAN with one or
    more ceilings replaced. No overrides means the baseline exactly.

    Args:
        values: Raw ``--ceiling`` strings, e.g. ``["name_token=200"]``.

    Returns:
        Mapping of strategy name to replacement ceiling. Empty when no
        overrides were given.

    Raises:
        ValueError: On unknown strategies, malformed values, or non-positive
            ceilings. A sweep variant must say what it changes; a typo that
            silently ran the baseline would publish the baseline as a variant.
    """
    overrides: dict[str, int] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(
                f"Bad --ceiling {raw!r}: expected NAME=INT, e.g. name_token=200"
            )
        name, _, value = raw.partition("=")
        name, value = name.strip(), value.strip()
        if name not in CEILING_STRATEGIES:
            raise ValueError(
                f"Bad --ceiling {raw!r}: {name!r} carries no ceiling. "
                f"Overridable strategies: {', '.join(CEILING_STRATEGIES)}"
            )
        try:
            ceiling = int(value)
        except ValueError:
            raise ValueError(f"Bad --ceiling {raw!r}: {value!r} is not an integer")
        if ceiling <= 0:
            raise ValueError(f"Bad --ceiling {raw!r}: ceiling must be positive")
        overrides[name] = ceiling
    return overrides


def effective_plan(overrides: dict[str, int] | None) -> list:
    """The PLAN with ceiling overrides applied, as ``(name, builder, max_df)``.

    With no overrides this is identical to PLAN — the baseline run is the
    default invocation, not a special case, so the baseline cannot drift from
    what E011 measured. The module-global PLAN is never mutated.
    """
    overrides = overrides or {}
    return [
        (name, builder, overrides.get(name, max_df)) for name, builder, max_df in PLAN
    ]


def candidate_volume_for_strategy(
    strategy: str,
    builder,
    max_df: int | None,
    queries: pl.DataFrame,
    vendors: pl.LazyFrame,
) -> dict:
    """Candidate-pair volume a strategy would generate, from aggregates only.

    E014 must weigh recalled pairs against generated candidates, but the
    bitmask loop deliberately never materialises candidate sets. This fills
    that gap without materialising them: for each surviving key, the block it
    creates holds ``n_query_rows_with_key × df`` pairs, and the sum over keys
    is the strategy's pair volume.

    This is a pre-dedup, pre-cap upper bound: pairs sharing several keys are
    counted once per key, and no cap is applied. As a *cost denominator* that
    is the right direction — it can only overstate cost, never hide it. It is
    NOT a candidate count and must not be quoted as one.

    Args:
        strategy: Strategy name, for the report row.
        builder: The ``build_*_keys`` callable.
        max_df: Ceiling applied, or ``None``.
        queries: Sampled S1 frame (needs ``row_id``).
        vendors: Full vendor pool, lazy.

    Returns:
        ``{strategy, max_df, kept_keys, dropped_keys, candidate_pairs}`` where
        ``dropped_keys`` counts keys removed by the ceiling and
        ``candidate_pairs`` is the block-size sum over kept keys.
    """
    df_counts = (
        builder(vendors)
        .group_by("key")
        .agg(pl.col("row_id").n_unique().alias("df"))
        .collect(engine="streaming")
    )
    total_keys = df_counts.height
    if max_df is None:
        kept = df_counts
    else:
        kept = df_counts.filter((pl.col("df") <= max_df) & (pl.col("df") >= 1))
    dropped = total_keys - kept.height
    per_key_queries = (
        builder(queries.lazy()).collect().group_by("key").len().rename({"len": "nq"})
    )
    blocks = per_key_queries.join(kept.select("key", "df"), on="key", how="inner")
    pairs = int((blocks["nq"] * blocks["df"]).sum()) if blocks.height else 0
    return {
        "strategy": strategy,
        "max_df": max_df,
        "kept_keys": kept.height,
        "dropped_keys": dropped,
        "candidate_pairs": pairs,
    }


def recall_by_country(
    found: pl.DataFrame, query_country: pl.DataFrame
) -> dict[str, dict[str, float | int]]:
    """Per-country true-pair recall from a bitmask plus a row_id→country map.

    Args:
        found: The bitmask, one row per true pair.
        query_country: ``(row_id, country)`` frame, one row per queried S1.

    Returns:
        ``{country: {true_pairs, retrieved_pairs, pair_recall}}``. Countries
        with no retrieved pairs report 0.0, never null — absence of recall is
        a measurement, not missing data.
    """
    joined = found.join(query_country, on="row_id", how="left")
    if joined["country"].null_count():
        missing = joined.filter(pl.col("country").is_null())["row_id"].unique()
        raise ValueError(
            f"{missing.len()} true pairs have no country mapping "
            f"(e.g. {missing.head(3).to_list()}). The query sample and the "
            f"country map disagree; stop, do not publish a partial breakdown."
        )
    out: dict[str, dict[str, float | int]] = {}
    # Iterating a group_by yields ((key,), frame): the key arrives as a tuple.
    for (country,), group in joined.group_by("country", maintain_order=True):
        total = group.height
        retrieved = int((group["bit"] > 0).sum())
        out[country] = {
            "true_pairs": total,
            "retrieved_pairs": retrieved,
            "pair_recall": round(retrieved / total, 6),
        }
    return out

def _plan_objects(plan: list | None = None) -> list:
    """Get a plan as RetrievalPlan objects for generate_candidates.

    Args:
        plan: ``(name, builder, max_df)`` rows. Defaults to the module PLAN,
            i.e. the baseline.
    """
    rows = plan if plan is not None else PLAN
    return [RetrievalPlan(name=p[0], builder=p[1], max_df=p[2]) for p in rows]


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


def print_report(report: dict) -> None:
    """Print a report produced by ``team_diamond.retrieval.recall_stats``.

    The arithmetic lives in ``src/`` so that E010/E011, E012 and any later
    experiment cannot each define "standalone recall" slightly differently. See
    that module for why that mattered.
    """
    plan_rows = report["per_strategy"]
    print(
        render_recall_table(
            plan_rows,
            report["union"]["pair_recall"],
            report["union"]["entity_recall"],
        )
    )
    ceilings = report["config"].get("max_df", {})
    if ceilings:
        print("\ndocument-frequency ceilings in force:")
        for name, value in ceilings.items():
            print(f"  {name:16} max_df={value}")

    print("\nREACHABILITY OF F0.5 (assumes a PERFECT classifier on these candidates)")
    print(f"{'pair recall R':>16} {'best possible F0.5':>21}")
    measured = report["union"]["pair_recall"]
    for r in (measured, 0.60, 0.70, 0.75, RECALL_REQUIRED_FOR_F05_95, 0.85, 0.90, 0.95):
        best = 1.25 * r / (0.25 + r)
        if abs(r - measured) < 1e-9:
            marker = "  <-- measured ceiling"
        elif abs(r - RECALL_REQUIRED_FOR_F05_95) < 1e-4:
            marker = "  <-- R required for F0.5 = 0.95"
        else:
            marker = ""
        print(f"{r:>16.4f} {best:>21.4f}{marker}")

    n_pairs = report["counts"]["true_pairs"]
    exactly_one = report["retrieved_by_exactly_one"]
    print("\nWHERE THE REMAINING RECALL IS LOST")
    print(f"  total true pairs:                 {n_pairs:>9,}")
    print(f"  retrieved by no strategy:         {report['missed_pairs']:>9,}"
          f"  ({report['missed_percentage']:6.2f}%)")
    print(f"  retrieved by exactly one:         {exactly_one:>9,}"
          f"  ({100 * exactly_one / n_pairs:6.2f}%)")
    retrieved = n_pairs - report["missed_pairs"]
    print(f"  retrieved by two or more:         {retrieved - exactly_one:>9,}"
          f"  ({100 * (retrieved - exactly_one) / n_pairs:6.2f}%)")
    print("  strategies per true pair:")
    for k, count in report["coverage_histogram"].items():
        print(f"    {k:>2}  {count:>9,}  ({100 * count / n_pairs:6.2f}%)")
    print("  read a strategy's own quality from the standalone column and whether it")
    print("  earns its place from the marginal column.")


def build_report(
    found: pl.DataFrame,
    *,
    n_pairs: int,
    n_entities: int,
    run_config: dict,
    counts: dict | None = None,
    plan: list | None = None,
    country_breakdown: dict | None = None,
) -> dict:
    """Assemble the recall report, attaching each strategy's max_df.

    ``team_diamond.retrieval.recall_stats.recall_report`` deals in strategy names
    because it must work from a bitmask alone; the driver knows the ceilings, so
    it merges them back in here.

    Args:
        plan: The effective ``(name, builder, max_df)`` plan the run used.
            Defaults to the module PLAN. A ceiling-variant run MUST pass its
            effective plan here — recording baseline ceilings for a variant run
            is the same class of bug as E011's cumulative/standalone confusion.
        country_breakdown: Optional per-country recall from
            :func:`recall_by_country`. ``None`` (bitmask-only path, no corpus)
            records an explicit null with its reason rather than a silent gap.
    """
    rows = plan if plan is not None else PLAN
    report = build_recall_report(
        found,
        [name for name, _, _ in rows],
        n_pairs=n_pairs,
        n_entities=n_entities,
        run_config=run_config,
        counts=counts,
    )
    ceilings = {name: max_df for name, _, max_df in rows if max_df is not None}
    for row in report["per_strategy"]:
        row["max_df"] = ceilings.get(row["strategy"])
    report["recall_by_country"] = country_breakdown
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-rows", type=int, default=DEFAULT_QUERY_ROWS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--bitmask-out",
        type=str,
        default=None,
        help="Save the per-pair strategy bitmask to parquet. The bitmask is the "
        "only artifact that preserves per-strategy attribution, so a run whose "
        "bitmask is not saved cannot have its standalone recalls re-derived later.",
    )
    parser.add_argument(
        "--bitmask-in",
        type=str,
        default=None,
        help="Recompute every recall statistic from a saved bitmask instead of "
        "running retrieval. Reproduces the report exactly and needs neither the "
        "corpus nor the document frequencies.",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Write the report as JSON."
    )
    parser.add_argument("--cap-sweep", action="store_true", default=False,
                        help="Run cap sweep after uncapped measurement. "
                             "Defaults to small vendor sample unless --full-corpus is set.")
    parser.add_argument("--full-corpus", action="store_true", default=False,
                        help="Run cap sweep on full 10.3M vendor corpus. "
                             "REQUIRES SAGEMAKER WITH 32GB RAM. Do not use locally.")
    parser.add_argument(
        "--ceiling",
        action="append",
        default=[],
        metavar="NAME=INT",
        help="Override a document-frequency ceiling for this run, e.g. "
             "--ceiling name_token=200. Repeatable. Only name_token, "
             "address_token and numeric_token accept overrides. With no "
             "--ceiling flags the run is the E011 baseline exactly.",
    )
    args = parser.parse_args()

    # The effective plan is fixed once, up front, and every downstream consumer
    # — the retrieval loop, the report, the cap sweep — reads this object.
    # There is exactly one place where "which ceilings ran" is decided, so the
    # report cannot describe different ceilings than the retrieval used.
    ceiling_overrides = parse_ceiling_overrides(args.ceiling)
    plan = effective_plan(ceiling_overrides)
    if ceiling_overrides:
        log(
            "Ceiling variant: "
            + ", ".join(f"{k}={v}" for k, v in sorted(ceiling_overrides.items()))
            + " (all other strategies at baseline)"
        )

    t0 = time.time()

    # --- fast path: re-derive the whole report from a frozen bitmask ---------
    # A bitmask holds exactly one row per true pair, so n_pairs and the number of
    # entities with >=1 match both follow from it. That is what makes a saved
    # bitmask sufficient evidence on its own.
    if args.bitmask_in:
        log(f"Loading bitmask from {args.bitmask_in} (retrieval not re-run)")
        found = pl.read_parquet(args.bitmask_in)
        validate_bitmask(found)
        report = build_report(
            found,
            n_pairs=found.height,
            n_entities=found["row_id"].n_unique(),
            plan=plan,
            country_breakdown=None,
            run_config={
                "source": "bitmask",
                "bitmask_path": args.bitmask_in,
                "query_rows": args.query_rows,
                "seed": args.seed,
                "strategies": [name for name, _, _ in plan],
                "max_df": {
                    name: max_df for name, _, max_df in plan if max_df is not None
                },
                "ceiling_overrides": ceiling_overrides,
                "country_note": (
                    "recall_by_country is null on the bitmask-only path: "
                    "country lives on the corpus, which this path does not read. "
                    "Use analyze_retrieval_misses.py --bitmask-input for the "
                    "country breakdown."
                ),
            },
            counts={"bitmask_rows": found.height},
        )
        print_report(report)
        report["runtime_seconds"] = round(time.time() - t0, 1)
        report["peak_rss_gib"] = round(rss_gb(), 2)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2))
            log(f"Report written to {out}")
        log(f"total runtime {report['runtime_seconds']:.1f}s")
        return

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
    # No recall is computed inside this loop: every metric is derived afterwards
    # from the finished bitmask by `recall_report`, so the loop cannot and the
    # report cannot disagree.
    found = true_pairs.with_columns(pl.lit(0).alias("bit"))

    candidate_volumes: list[dict] = []
    for index, (name, builder, max_df) in enumerate(plan):
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
        # Cost denominator for the same strategy, measured from aggregates only
        # (never materialised pairs). Kept inside the loop so volume and recall
        # always describe the same ceilings.
        candidate_volumes.append(
            candidate_volume_for_strategy(name, builder, max_df, queries, vendors_lazy)
        )
        gc.collect()

    # The bitmask is the only artifact that keeps per-strategy attribution.
    # Written before anything else so it survives even if reporting fails.
    if args.bitmask_out:
        out = Path(args.bitmask_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        found.write_parquet(out)
        log(f"Bitmask saved to {out}")

    report = build_report(
        found,
        n_pairs=n_pairs,
        n_entities=n_entities,
        plan=plan,
        country_breakdown=recall_by_country(
            found, queries.select("row_id", "country").unique()
        ),
        run_config={
            "source": "live",
            "query_rows": queries.height,
            "seed": args.seed,
            "requested_query_rows": args.query_rows,
            "strategies": [name for name, _, _ in plan],
            "max_df": {name: max_df for name, _, max_df in plan if max_df is not None},
            "ceiling_overrides": ceiling_overrides,
        },
        counts={
            "vendor_pool": vendors.height,
            "queries": queries.height,
            "distinct_true_vendors": relevant_vendors.len(),
        },
    )
    report["candidate_volume"] = candidate_volumes
    report["candidate_volume_note"] = (
        "Per-strategy candidate-pair volume from key-block aggregates "
        "(sum over kept keys of n_query_rows * df). Pre-dedup, pre-cap upper "
        "bound: a cost denominator for recall-per-additional-candidate, NOT a "
        "candidate count. Do not quote it as one."
    )
    print_report(report)

    # --- the ceiling ---------------------------------------------------------
    # (printed inside `recall_report` above -- the ceiling and the miss
    # breakdown are reported from the same bitmask as the strategy table, so
    # they cannot drift apart)
    # --- cap sweep -----------------------------------------------------------
    # The uncapped measurement above is the reference ceiling.
    # Now measure recall at each cap value to quantify the recall cost of capping.
    cap_results: list[dict] = []
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

        # queries already has entity_id, name_norm, addr_norm, country
        queries_for_gen = queries.select("entity_id", "name_norm", "addr_norm", "country")

        plan_objs = _plan_objects(plan)

        for cap in CAP_SWEEP:
            log(f"  Cap sweep ({cap_sweep_label}): cap_per_query={cap}")
            candidates, gen_report = generate_candidates(
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

            per_query = candidates.group_by("s1_id").len()["len"]
            log(f"    cap={cap:>4}  pair_recall={pair_r:.4f}  "
                f"entity_recall={entity_r:.4f}  candidates={candidates.height:,}")
            cap_results.append({
                "cap": cap,
                "pair_recall": round(pair_r, 6),
                "entity_recall": round(entity_r, 6),
                "candidates": candidates.height,
                "candidates_per_query_median": int(per_query.median()),
                "candidates_per_query_p95": int(per_query.quantile(0.95)),
                "candidates_per_query_max": int(per_query.max()),
                "dropped_by_cap": gen_report.n_dropped_by_cap,
            })
            del candidates, hits, per_query
            gc.collect()

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

    del found, vendors, s1, queries, true_pairs
    gc.collect()

    report["config"]["cap_sweep"] = {
        "enabled": bool(args.cap_sweep),
        "full_corpus": bool(args.full_corpus),
        "results": cap_results if args.cap_sweep else [],
    }
    report["runtime_seconds"] = round(time.time() - t0, 1)
    report["peak_rss_gib"] = round(rss_gb(), 2)
    log(f"total runtime {report['runtime_seconds']:.1f}s")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        log(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
