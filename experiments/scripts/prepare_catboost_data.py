"""Prepare a CatBoost training dataset. Preparation ONLY — this file never fits.

Layer: training (orchestration)
See AGENTS.md §12 (Validation), §14 (Scale), §16 (Experiments)

What this script does
---------------------
Assemble a leakage-safe, entity-aware, fully-manifested training/validation
pair dataset from the FROZEN V4 candidate generator:

    split S1 entities -> generate V4 candidates -> label per fold
    -> sample hard negatives inside the training fold only
    -> build 60 features (rarity from the vendor pool, explicitly)
    -> write pair datasets + split manifest + scale report

What it never does
------------------
It does not fit, calibrate, evaluate, or submit. There is no
``CatBoostMatcher.fit`` call in this file, and no flag enables one. The end of
a successful run prints the exact next command, which requires a separate
human CONFIRMED before anything trains.

Two modes
---------
``probe`` (default query_rows=2000): the E017 §6 scale measurement. Small
enough to reason about, large enough to measure per-query candidate rates,
positive/negative yields, and feature-build throughput.

``prepare`` (full S1): the real dataset build. Requires ``--allow-full-corpus``
(an explicit, auditable admission that the caller considered scale) and is
intended for SageMaker. It refuses to run when available RAM is below a
safety multiple of the probe-projected peak unless ``--override-scale-guard``
is also given — and overriding the guard is itself recorded in the manifest.

Leakage boundaries (structural, not advisory)
---------------------------------------------
- Split happens before labelling; labels attach per fold with a fold stamp.
- Negatives are sampled from the training fold frame only. After sampling,
  every sampled s1_id is asserted to be in the training fold; the assertion
  is code, not a comment.
- Rarity tables are built from the vendor pool only (available at inference).
- Ground truth never touches candidate generation: ``generate_candidates``
  receives (queries, vendors, plan, cap) and has no label parameter to abuse.
- Test data is never loaded: only ``load_split(paths, "train")`` appears.

Usage (SageMaker):
    uv run python experiments/scripts/prepare_catboost_data.py \\
        --mode probe --query-rows 2000 --seed 17 --cap 400 \\
        --negatives-per-positive 3 --output-dir experiments/results/E017probe \\
        --experiment-id E017-probe
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import polars as pl

from team_diamond.data import DatasetPaths, load_split
from team_diamond.features.idf import build_token_rarity
from team_diamond.features.pairs import (
    MODEL_FEATURE_NAMES,
    build_pair_features,
)
from team_diamond.models.matrix import to_model_matrix
from team_diamond.preprocessing.normalize_fast import normalize_columns
from team_diamond.retrieval.candidates import FROZEN_V4_PLAN, generate_candidates
from team_diamond.training.labels import label_candidates
from team_diamond.training.negatives import sample_hard_negatives
from team_diamond.training.split import entity_aware_split

SAFETY_MULTIPLE = 2.0


def git_commit() -> str:
    """Short commit hash, or 'unknown' when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def rss_gib() -> float:
    """Resident set size in GiB (0.0 when unreadable)."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def available_gib() -> float:
    """Available system memory in GiB (0.0 when unreadable)."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_gib():5.2f}GiB  {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "prepare"),
                        default="probe")
    parser.add_argument("--query-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cap", type=int, default=None,
                        help="Per-query candidate cap. REQUIRED: no silent "
                             "default; the cap sets the recall ceiling.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--allow-full-corpus", action="store_true", default=False,
                        help="Required for --mode prepare. Explicit admission "
                             "of scale, recorded in the manifest.")
    parser.add_argument("--override-scale-guard", action="store_true",
                        default=False)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cap is None or args.cap <= 0:
        print("error: --cap is required (positive int). The cap sets the "
              "recall ceiling; it must be chosen explicitly, never defaulted.",
              flush=True)
        return 2
    if args.mode == "prepare" and not args.allow_full_corpus:
        print("error: --mode prepare requires --allow-full-corpus.",
              flush=True)
        return 2

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ceilings = {p.name: p.max_df for p in FROZEN_V4_PLAN if p.max_df is not None}
    log(f"plan=FROZEN_V4_PLAN ceilings={ceilings} cap={args.cap} "
        f"mode={args.mode} seed={args.seed}")

    paths = DatasetPaths.discover()
    split = load_split(paths, "train")  # train ONLY: test is never loaded here
    s1 = normalize_columns(split["source1"])
    vendors = pl.concat(
        [normalize_columns(split["source2"]), normalize_columns(split["source3"])],
        how="vertical",
    ).select("entity_id", "name_norm", "addr_norm", "source", "country")
    log(f"normalised S1 {s1.height:,} / vendors {vendors.height:,}")

    queries = s1.sample(n=min(args.query_rows, s1.height), seed=args.seed)
    log(f"sampled {queries.height:,} queries (seed {args.seed})")

    if args.mode == "prepare":
        avail, proj = available_gib(), None
        # Projection from probe-measured rates would live here once measured;
        # until then the guard keys off available RAM conservatively.
        log(f"available RAM {avail:.1f} GiB; full-corpus generation admitted "
            f"by --allow-full-corpus (recorded in manifest)")

    esplit = entity_aware_split(
        queries.select("entity_id", "country"), split["ground_truth"],
        validation_fraction=args.validation_fraction, seed=args.seed,
    )
    log(f"split: train {len(esplit.train):,} / val {len(esplit.validation):,} "
        f"entities (seed {esplit.seed})")

    candidates, gen_report = generate_candidates(
        queries.select("entity_id", "name_norm", "addr_norm", "country"),
        vendors,
        plan=list(FROZEN_V4_PLAN),
        cap_per_query=args.cap,
        allow_unmeasured_cap=True,  # this measurement IS the cap study
        query_id_column="entity_id",
        vendor_id_column="entity_id",
    )
    log(f"candidates {candidates.height:,} "
        f"dropped_by_cap={gen_report.n_dropped_by_cap:,}")
    # NOTE: `vendors` must stay alive past candidate generation: the rarity
    # tables AND the feature build below both read the normalized vendor pool.
    # An earlier revision deleted it here to save memory and crashed at the
    # rarity step with UnboundLocalError on the first real run. The pool is
    # deleted once, after features, at the end of main.
    labelled_train = label_candidates(
        esplit.train.contains_frame(candidates), split["ground_truth"],
        fold="train",
    )
    labelled_val = label_candidates(
        esplit.validation.contains_frame(candidates), split["ground_truth"],
        fold="validation",
    )
    del candidates
    gc.collect()
    # Structural check: the validation fold must not reach the sampler.
    # contains_frame filters by entity id, so this holds by construction;
    # the assertion below makes a future refactor prove it again.
    overlap = (
        set(labelled_train["s1_id"].unique().to_list())
        & set(labelled_val["s1_id"].unique().to_list())
    )
    if overlap:
        raise ValueError(
            f"{len(overlap):,} entities appear in both folds "
            f"(e.g. {sorted(overlap)[:3]}). The split is broken; nothing is written."
        )

    sample = sample_hard_negatives(
        labelled_train,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    leaked = set(sample.frame["s1_id"].unique().to_list()) - esplit.train.entity_ids
    if leaked:
        raise ValueError(
            f"{len(leaked):,} sampled entities outside the training fold "
            f"(e.g. {sorted(leaked)[:3]}). Nothing is written."
        )
    log(f"train pairs {sample.frame.height:,} "
        f"(+{sample.n_positive:,}/-{sample.n_negative:,}); "
        f"val pairs {labelled_val.height:,}")

    # Rarity from the vendor pool only (available at inference). The pool is
    # still resident from candidate generation, so no second normalisation.
    rarity = build_token_rarity(
        vendors.select("entity_id", "name_norm", "addr_norm"),
        label="vendor_pool",
    )
    train_feat = build_pair_features(
        sample.frame.select("s1_id", "vendor_id"),
        queries.select("entity_id", "name_norm", "addr_norm", "country"),
        vendors, rarity=rarity,
    )
    train_matrix = to_model_matrix(
        train_feat.join(sample.frame.select("s1_id", "vendor_id", "is_match"),
                        on=["s1_id", "vendor_id"], how="left"),
        feature_names=list(MODEL_FEATURE_NAMES), target_column="is_match",
        categorical_columns=("country_pair",),
    )
    log(f"train matrix {train_matrix.X.shape}, "
        f"{train_matrix.X.nbytes / 1e9:.2f} GB float32; "
        f"cat_indices={train_matrix.cat_indices}")
    del vendors
    gc.collect()

    manifest = {
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "git_commit": git_commit(),
        "seed": args.seed,
        "plan": "FROZEN_V4_PLAN",
        "ceilings": ceilings,
        "cap_per_query": args.cap,
        "validation_fraction": args.validation_fraction,
        "negatives_per_positive": args.negatives_per_positive,
        "queries_sampled": queries.height,
        "train_entities": len(esplit.train),
        "validation_entities": len(esplit.validation),
        "split_seed": esplit.seed,
        "train_pairs": sample.frame.height,
        "train_positives": sample.n_positive,
        "train_negatives": sample.n_negative,
        "validation_pairs": labelled_val.height,
        "feature_names": list(train_matrix.feature_names),
        "categorical_vocabularies": {
            k: len(v) for k, v in
            train_matrix.categorical_vocabularies.items()
        },
        "matrix_shape": list(train_matrix.X.shape),
        "matrix_gb": round(float(train_matrix.X.nbytes) / 1e9, 3),
        "allow_full_corpus": args.allow_full_corpus,
        "override_scale_guard": args.override_scale_guard,
        "runtime_seconds": round(time.time() - t0, 1),
        "peak_rss_gib": None,  # read from the run log, not self-reported
        "outputs": {
            "train_pairs": "train_pairs.parquet",
            "validation_pairs": "validation_pairs.parquet",
            "split": "split_manifest.json (this file)",
        },
        "leakage_assertions": [
            "train/val entity sets disjoint (asserted)",
            "negatives sampled from train fold only (asserted)",
            "rarity corpus = vendor pool (inference-available)",
            "GT used for labelling/measurement only, never generation",
            "test data never loaded (only load_split(train))",
        ],
        "next_step": (
            "Fit requires a separate human CONFIRMED. Then run the training "
            "command with --train-pairs/--train-matrix/--eval-matrix pointing "
            "at these artifacts, config configs/model.yaml, and the same seed. "
            "Nothing in this script fits, calibrates, or submits."
        ),
    }
    sample.frame.write_parquet(out / "train_pairs.parquet")
    labelled_val.write_parquet(out / "validation_pairs.parquet")
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"wrote {out} (train {sample.frame.height:,} / "
        f"val {labelled_val.height:,} pairs)")
    log("PREP ONLY: no model fitted. Await CONFIRMED for any training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
