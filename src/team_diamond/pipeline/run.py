"""End-to-end orchestration: split, label, featurise, train, decide, submit.

Layer: pipeline
See AGENTS.md §7 (Architecture invariants), §14 (Scale and cost rules),
§16 (Experiment discipline), §17 (Submission contract)

Layer order, and the direction it may not run in
------------------------------------------------
.. code-block:: text

    data -> normalization -> retrieval -> features -> matching -> decision
         -> evaluation / submission

The dependency direction is fixed (AGENTS.md §7). This module is the only place
all of them appear together, and it is therefore the only place a layer-inversion
bug would be invisible. Two consequences shape the code:

- Each stage is a separate function with an explicit frame in and frame out, so
  a stage can be re-run, inspected, or replaced without touching the others. The
  alternative -- one ``main()`` that does all six steps -- is what makes a
  pipeline impossible to debug at 26M records.
- No stage is invoked implicitly. :func:`train_and_evaluate` does not fetch
  candidates; it receives them. That is what keeps the decision layer from
  silently reaching back into retrieval.

Nothing here runs at import time
--------------------------------
Every function in this module touches the full corpus if called, and calling one
is expensive. There is no module-level work, no cached singleton, and no
"convenience" entry point that runs on import -- an import must be free. Use the
CLI (:func:`main`) or call the stages explicitly.

The scale guard
---------------
:func:`guard_full_corpus_run` is called before the first stage that would
materialise a candidate set. See :mod:`team_diamond.pipeline.scale` for why that
guard is in code rather than in a checklist.

Fold safety
-----------
The split happens before labelling, and labels are attached per fold. The vendor
pool is shared between folds, which is **not** a leak: S2 and S3 are unlabelled
and the inference-time pool is equally shared. IDF is built from that pool
explicitly, never from a corpus that includes labelled rows from the fold being
scored.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final

import polars as pl

from team_diamond.config import Config, load_config
from team_diamond.data import DatasetPaths, load_split
from team_diamond.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    build_pair_features,
    build_token_rarity,
)
from team_diamond.decision import (
    MarginalValuePolicy,
    MarginalValuePolicyParams,
    apply_policy,
    compare_policies,
)
from team_diamond.models import CatBoostMatcher, to_model_matrix
from team_diamond.pipeline.scale import (
    MemoryTrace,
    ScaleEstimate,
    guard_full_corpus_run,
)
from team_diamond.pipeline.submission import (
    build_submission,
    validate_submission,
    write_submission,
)
from team_diamond.preprocessing.normalize_fast import normalize_columns
from team_diamond.retrieval.candidates import RetrievalPlan, generate_candidates
from team_diamond.training import (
    LABEL_COLUMN,
    EntitySplit,
    entity_aware_split,
    label_candidates,
    label_histogram,
    sample_hard_negatives,
)
from team_diamond.training.negatives import NegativeSample

__all__ = [
    "VENDOR_COLUMNS",
    "RunReport",
    "git_commit",
    "prepare_vendor_pool",
    "generate_retrieval_plan",
    "featurise",
    "train_and_evaluate",
    "predict_and_submit",
    "main",
]

#: Columns the vendor pool must carry, in the order the feature layer selects.
VENDOR_COLUMNS: Final[tuple[str, ...]] = (
    "entity_id",
    "name_norm",
    "addr_norm",
    "country",
    "source",
)


def git_commit(repo: Path | None = None) -> str:
    """Current commit hash, for the experiment record.

    Returns:
        The short hash, or ``"unknown"`` if git is unavailable — which is
        itself worth knowing, because a run without a commit cannot be
        reproduced (AGENTS.md §16).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo or Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def prepare_vendor_pool(
    split: dict[str, pl.DataFrame], *, trace: MemoryTrace | None = None
) -> pl.DataFrame:
    """Normalise and concatenate S2 and S3 into one searchable pool.

    The two sources are concatenated rather than kept separate because a single
    entity may match in S2, S3, or both, and retrieval must not prefer one
    source. The originating source is retained as the ``source`` column so the
    feature layer can use it, and so a submission's ids remain traceable to the
    file they came from (AGENTS.md §8.3).

    Args:
        split: Output of :func:`~team_diamond.data.loading.load_split`.
        trace: Optional memory trace to record the stage against.

    Returns:
        Frame with :data:`VENDOR_COLUMNS`, deduplicated on ``entity_id``.

    Raises:
        ValueError: If S2 or S3 is empty, or a vendor id appears in both.
    """
    with (trace.stage("prepare_vendor_pool") if trace else _null_stage()):
        frames = []
        for source in (2, 3):
            raw = split[f"source{source}"]
            if raw.is_empty():
                raise ValueError(
                    f"source{source} is empty; the dataset layout has changed"
                )
            frames.append(
                normalize_columns(raw).select(*VENDOR_COLUMNS)
            )
        vendors = pl.concat(frames, how="vertical")

        duplicated = (
            vendors.group_by("entity_id").len().filter(pl.col("len") > 1)
        )
        if duplicated.height:
            raise ValueError(
                f"{duplicated.height:,} vendor id(s) appear in both S2 and S3 "
                f"(e.g. {duplicated['entity_id'].head(3).to_list()}). S2 and S3 "
                f"ids are not assumed to share a namespace (AGENTS.md §8.3); "
                f"resolve the overlap before proceeding, because a duplicate "
                f"vendor would double its weight in retrieval."
            )
        return vendors


def generate_retrieval_plan(config: Config) -> tuple[RetrievalPlan, ...]:
    """Read the enabled retrieval strategies from config.

    Args:
        config: Merged configuration.

    Returns:
        The enabled plans, in the config's order.
    """
    from team_diamond.retrieval.candidates import DEFAULT_PLAN

    by_name = {plan.name: plan for plan in DEFAULT_PLAN}
    requested = config.get("retrieval.strategies")
    if not requested:
        return DEFAULT_PLAN
    plans: list[RetrievalPlan] = []
    for name in requested:
        if name not in by_name:
            raise ValueError(
                f"retrieval.strategies names unknown strategy {name!r}; "
                f"available: {sorted(by_name)}"
            )
        plans.append(by_name[name])
    return tuple(plans)


def featurise(
    candidates: pl.DataFrame,
    s1: pl.DataFrame,
    vendors: pl.DataFrame,
    *,
    trace: MemoryTrace | None = None,
) -> pl.DataFrame:
    """Build the feature matrix rows for a candidate set.

    IDF is built from the **vendor pool only**. That is the corpus available at
    inference, so a model validated with it faces no advantage it would not have
    on test. Building it over a corpus that also contained labelled rows would
    be a leak of exactly the kind AGENTS.md §12 lists, and the explicit argument
    here makes the choice visible rather than default.

    Args:
        candidates: Candidate pairs, including retrieval-evidence columns.
        s1: Normalised S1 records.
        vendors: Normalised vendor pool.
        trace: Optional memory trace.

    Returns:
        Feature frame with :data:`~team_diamond.features.pairs.FEATURE_COLUMNS`
        plus ``country_pair`` and the id columns.
    """
    with (trace.stage("build_token_rarity") if trace else _null_stage()):
        rarity = build_token_rarity(vendors, label="vendor_pool")

    with (trace.stage("build_pair_features") if trace else _null_stage()):
        return build_pair_features(candidates, s1, vendors, rarity=rarity)


@dataclass(slots=True)
class RunReport:
    """Everything needed to write one row of ``experiments/registry.csv``.

    Attributes:
        experiment_id: Identifier assigned by the caller.
        date: Run date.
        git_commit: Commit the run was executed at.
        seed: Seed used for splitting, negative sampling, and the model.
        config_snapshot: Merged config, so the run is reproducible from the log.
        n_s1, n_vendors: Dataset sizes.
        candidate_recall_ceiling: Fraction of true matches present in the
            candidate set. **The arithmetic bound on achievable F0.5**; nothing
            downstream can exceed it.
        label_summary: Output of
            :func:`~team_diamond.training.labels.label_histogram`.
        negative_summary: Output of
            :func:`~team_diamond.training.negatives.sample_hard_negatives`.
        split_summary: Output of :meth:`EntitySplit.describe`.
        validation_f05: Entity-level macro F0.5 on the validation fold. The
            number that matters.
        pooled_f05: Count-pooled variant, tracked for risk R1 only.
        policy_table: Policy comparison from :func:`compare_policies`.
        chosen_policy: Name of the policy selected for the submission.
        feature_importance: Top features by gain.
        memory: Peak RSS across the run.
        notes: Free text, including anything unmeasured.
    """

    experiment_id: str
    date: str
    git_commit: str
    seed: int
    config_snapshot: str
    n_s1: int
    n_vendors: int
    candidate_recall_ceiling: float
    label_summary: pl.DataFrame
    negative_summary: NegativeSample
    split_summary: str
    validation_f05: float
    pooled_f05: float
    policy_table: pl.DataFrame
    chosen_policy: str
    feature_importance: pl.DataFrame
    memory_peak_gib: float
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_registry_row(self) -> dict[str, Any]:
        """Flatten into a registry row.

        Only measured values are written. Anything unmeasured goes in ``notes``
        as a word, not a number, so the registry can never be read as claiming a
        result that does not exist.
        """
        labels = dict(
            zip(
                self.label_summary["metric"].to_list(),
                self.label_summary["value"].to_list(),
                strict=True,
            )
        )
        return {
            "experiment_id": self.experiment_id,
            "date": self.date,
            "component": "full_pipeline",
            "change": self.notes or "first end-to-end run",
            "dataset_split": "train/entity_aware_80_20",
            "git_commit": self.git_commit,
            "seed": self.seed,
            "candidate_recall": round(self.candidate_recall_ceiling, 6),
            "pair_precision": self.policy_table["pair_precision"][0]
            if self.policy_table.height
            else None,
            "pair_recall": self.policy_table["pair_recall"][0]
            if self.policy_table.height
            else None,
            "entity_f05": round(self.validation_f05, 6),
            "runtime": self.extra.get("runtime_minutes"),
            "memory": f"{self.memory_peak_gib:.2f}GiB",
            "notes": (
                f"policy={self.chosen_policy}; pooled_f05={self.pooled_f05:.6f}; "
                f"n_s1={self.n_s1}; n_vendors={self.n_vendors}; "
                f"negatives_per_positive="
                f"{self.negative_summary.negatives_per_positive}; "
                f"positives_available="
                f"{self.negative_summary.n_positive_available}"
            ),
        }

    def describe(self) -> str:
        """Full human-readable report."""
        return "\n\n".join(
            [
                f"RUN REPORT {self.experiment_id} ({self.date}) "
                f"@ {self.git_commit}",
                self.split_summary,
                "\n".join(
                    f"  {m:<24} {v}"
                    for m, v in zip(
                        self.label_summary["metric"].to_list(),
                        self.label_summary["value"].to_list(),
                        strict=True,
                    )
                ),
                self.negative_summary.describe(),
                f"ENTITY F0.5 (validation)  : {self.validation_f05:.6f}"
                f"   <-- the number that decides",
                f"pooled F0.5 (risk R1 only) : {self.pooled_f05:.6f}",
                f"candidate recall ceiling   : "
                f"{self.candidate_recall_ceiling:.6f}",
                f"peak memory                : {self.memory_peak_gib:.2f} GiB",
                "policy comparison:",
                str(self.policy_table),
                "top features:",
                str(self.feature_importance.head(15)),
            ]
        )


def train_and_evaluate(
    s1: pl.DataFrame,
    vendors: pl.DataFrame,
    ground_truth: pl.DataFrame,
    candidates: pl.DataFrame,
    config: Config,
    *,
    experiment_id: str,
    seed: int = 20260926,
    trace: MemoryTrace | None = None,
) -> tuple[CatBoostMatcher, RunReport]:
    """Fit the matcher on a training fold and score it on a held-out fold.

    The fold discipline in three lines: split S1 entities, label only the
    training side, and never let the validation side reach the sampler. The
    vendor pool and its IDF are shared between folds by design, because they are
    unlabelled and equally available at inference.

    Args:
        s1: Normalised S1 records.
        vendors: Normalised vendor pool.
        ground_truth: Training ground truth.
        candidates: Candidates for the queried S1 subset. Must already have
            passed the scale guard, since by this point the expensive
            intermediate has been materialised.
        config: Merged configuration.
        experiment_id: Identifier for the registry row.
        seed: Seed for splitting, negative sampling, and the model.
        trace: Optional memory trace.

    Returns:
        The fitted matcher and the run report.

    Raises:
        ValueError: Propagated from the split, labelling, or sampling stages.
    """
    trace = trace or MemoryTrace(experiment_id)

    with trace.stage("entity_aware_split"):
        split = entity_aware_split(
            s1, ground_truth, validation_fraction=0.2, seed=seed
        )

    with trace.stage("label_train_fold"):
        train_candidates = split.train.contains_frame(candidates)
        train_labelled = label_candidates(
            train_candidates, ground_truth, fold="train"
        )
        validation_candidates = split.validation.contains_frame(candidates)
        validation_labelled = label_candidates(
            validation_candidates, ground_truth, fold="validation"
        )

    with trace.stage("label_histogram"):
        validation_summary = label_histogram(validation_labelled, ground_truth)
        ceiling = float(
            dict(
                zip(
                    validation_summary["metric"].to_list(),
                    validation_summary["value"].to_list(),
                    strict=True,
                )
            )["pair_recall_ceiling"]
        )

    negatives_per_positive = int(
        config.get("training.negatives_per_positive", 3) or 3
    )
    with trace.stage("sample_hard_negatives"):
        sample = sample_hard_negatives(
            train_labelled,
            negatives_per_positive=negatives_per_positive,
            seed=seed,
            score_column="n_keys",
        )

    with trace.stage("featurise_train"):
        train_features = featurise(sample.frame, s1, vendors, trace=trace)
    with trace.stage("featurise_validation"):
        validation_features = featurise(
            validation_candidates, s1, vendors, trace=trace
        )

    with trace.stage("to_model_matrix"):
        train_matrix = to_model_matrix(
            train_features,
            feature_names=FEATURE_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
            target_column=LABEL_COLUMN,
        )
        validation_matrix = to_model_matrix(
            validation_features,
            feature_names=FEATURE_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
            target_column=LABEL_COLUMN,
            categorical_vocabularies=train_matrix.categorical_vocabularies,
        )

    with trace.stage("fit_catboost"):
        matcher = CatBoostMatcher.from_config(config)
        matcher.fit(train_matrix, eval_matrix=validation_matrix)

    with trace.stage("score_validation"):
        probabilities = matcher.predict_proba(validation_matrix)
        scored = validation_candidates.with_columns(
            pl.Series("match_probability", probabilities, dtype=pl.Float64)
        )

    truth_frame = _truth_as_lists(ground_truth, split.validation.entity_ids)
    with trace.stage("compare_policies"):
        table = compare_policies(scored, truth_frame)

    report = RunReport(
        experiment_id=experiment_id,
        date=date.today().isoformat(),
        git_commit=git_commit(),
        seed=seed,
        config_snapshot=config.describe(),
        n_s1=s1.height,
        n_vendors=vendors.height,
        candidate_recall_ceiling=ceiling,
        label_summary=validation_summary,
        negative_summary=sample,
        split_summary=split.describe(),
        validation_f05=float(table["entity_f05"][0]),
        pooled_f05=float(table["pooled_f05"][0]),
        policy_table=table,
        chosen_policy=str(table["policy"][0]),
        feature_importance=matcher.feature_importance(),
        memory_peak_gib=trace.peak_bytes() / (1024**3),
        notes=config.get("experiments.notes", "") or "",
    )
    print(report.describe())
    print(trace.describe())
    return matcher, report


def _truth_as_lists(
    ground_truth: pl.DataFrame, entity_ids: frozenset[str]
) -> pl.DataFrame:
    """Ground truth restricted to a set of S1 entities, in list form.

    Singletons are retained with an empty list, because ``score_entities`` needs
    them: an entity absent from the truth mapping cannot be scored, and dropping
    singletons would delete exactly the entities where over-prediction is most
    expensive.
    """
    from team_diamond.training.labels import positive_pairs

    pairs = positive_pairs(ground_truth).filter(
        pl.col("s1_id").is_in(list(entity_ids))
    )
    return (
        pairs.group_by("s1_id")
        .agg(pl.col("vendor_id"))
        .select(pl.col("s1_id").alias("s1_entity_id"), pl.col("vendor_id").alias("matched_entity_ids"))
    )


def predict_and_submit(
    s1_test: pl.DataFrame,
    vendors: pl.DataFrame,
    candidates: pl.DataFrame,
    matcher: CatBoostMatcher,
    policy: MarginalValuePolicy | None,
    *,
    output_path: Path,
    trace: MemoryTrace | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score test candidates and write a validated submission.

    Args:
        s1_test: Normalised test S1 records; supplies the row list and order.
        vendors: Normalised vendor pool, for IDF and for the valid-id set.
        candidates: Candidates for the test S1 set.
        matcher: A fitted model.
        policy: Decision policy. ``None`` uses the configured marginal-value
            policy.
        output_path: Destination for ``matching_results.tsv``.
        trace: Optional memory trace.

    Returns:
        The submission frame and its validation report.

    Raises:
        ValueError: If the submission fails validation; the write is refused
            rather than producing a file that must not be submitted
            (AGENTS.md §17).
    """
    trace = trace or MemoryTrace("predict")

    with trace.stage("featurise_test"):
        features = featurise(candidates, s1_test, vendors, trace=trace)

    with trace.stage("score_test"):
        matrix = to_model_matrix(
            features,
            feature_names=FEATURE_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
            categorical_vocabularies=matcher.categorical_vocabularies(),
        )
        probabilities = matcher.predict_proba(matrix)
        scored = candidates.with_columns(
            pl.Series("match_probability", probabilities, dtype=pl.Float64)
        )

    with trace.stage("apply_policy"):
        chosen = policy or MarginalValuePolicy(
            MarginalValuePolicyParams(cardinality_prior=3.46)
        )
        emissions = apply_policy(scored, chosen)

    with trace.stage("build_submission"):
        submission = build_submission(emissions, s1_test)
        valid_ids = vendors["entity_id"].cast(pl.Utf8).to_list()
        report = validate_submission(submission, s1_test, valid_ids)
        print(report.describe())
        if not report.ok:
            raise ValueError(
                "submission failed validation and was NOT written:\n"
                + report.describe()
            )
        write_submission(submission, output_path)

    print(trace.describe())
    return submission, report


class _null_stage:
    """No-op stand-in for :meth:`MemoryTrace.stage` when no trace is given."""

    def __enter__(self) -> "_null_stage":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """Command-line entry point.

    Stages are separate subcommands rather than one ``all`` flag, because on a
    32 GiB instance the stages have genuinely different resource profiles and
    running them together means the cheap ones cannot be tried without paying
    for the expensive one.

    Usage:
        ``train`` -- split, featurise, fit, score the validation fold.
        ``predict`` -- score test candidates and write a submission.
        ``plan`` -- print the cost estimate required for a full-corpus run.

    Returns:
        Process exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    command = argv[0]
    if command == "plan":
        print(
            ScaleEstimate(
                n_s1=2_206_821,
                n_vendors=10_306_516,
                candidates_per_query=50,
                bytes_per_pair=512,
                peak_multiplier=2.5,
                runtime_minutes=180.0,
                rationale="UNMEASURED - fill this in before running",
                measured_from="UNMEASURED",
            ).describe()
        )
        return 0

    config = load_config("model", "retrieval", "features", "submission")
    print("unmeasured config parameters:")
    for parameter in config.unmeasured():
        print(f"  - {parameter}")
    print()

    paths = DatasetPaths.discover()
    trace = MemoryTrace(command)
    split_name = "train" if command == "train" else "test"

    with trace.stage(f"load_{split_name}"):
        split = load_split(paths, split_name)

    s1 = normalize_columns(split["source1"])
    vendors = prepare_vendor_pool(split, trace=trace)

    # The guard sits here, before the first stage that would materialise a
    # candidate set. Placing it after generation would be pointless: the memory
    # is already spent.
    guard_full_corpus_run(vendors.height, None)

    plan = generate_retrieval_plan(config)
    print(
        "retrieval strategies: "
        + ", ".join(p.name for p in plan)
        + f"\ncandidates per query cap: {config.get('retrieval.cap_per_query', 'none (uncapped)')}"
    )

    if command == "train":
        if "ground_truth" not in split:
            raise SystemExit("no ground truth available; cannot train on test")
        with trace.stage("generate_candidates"):
            candidates, report = generate_candidates(
                s1,
                vendors,
                plan=plan,
                cap_per_query=config.get("retrieval.cap_per_query"),
                allow_unmeasured_cap=bool(
                    config.get("retrieval.allow_unmeasured_cap", False)
                ),
            )
        print(report.describe())
        matcher, _ = train_and_evaluate(
            s1, vendors, split["ground_truth"], candidates, config,
            experiment_id=f"cli-{split_name}",
        )
        out = Path("artifacts") / f"matcher-{split_name}"
        matcher.save(out)
        print(f"saved matcher to {out}")
        return 0

    if command == "predict":
        model_dir = Path("artifacts") / "matcher-train"
        if not model_dir.is_dir():
            raise SystemExit(f"no trained model at {model_dir}; run `train` first")
        matcher = CatBoostMatcher.load(model_dir)
        with trace.stage("generate_candidates"):
            candidates, report = generate_candidates(
                s1,
                vendors,
                plan=plan,
                cap_per_query=config.get("retrieval.cap_per_query"),
                allow_unmeasured_cap=bool(
                    config.get("retrieval.allow_unmeasured_cap", False)
                ),
            )
        print(report.describe())
        predict_and_submit(
            s1, vendors, candidates, matcher, None,
            output_path=Path("matching_results.tsv"), trace=trace,
        )
        return 0

    print(f"unknown command {command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
