"""CatBoost pairwise match model.

Layer: models
See AGENTS.md §11 (Model preference), §12 (Validation rules)

CatBoost, and why
----------------
AGENTS.md §11 names LightGBM as the default. **CatBoost is used here by
explicit team decision**, and the deviation is recorded rather than quietly
taken -- the reasons are that native categorical handling covers
``country_pair`` without hand-rolled encoding, and symmetric trees are a
reasonable prior for name/address comparison. Both libraries are locked in
``pyproject.toml`` so the choice is reversible by configuration rather than by
reinstalling.

``configs/model.yaml`` still says ``type: lightgbm``. That is a known,
recorded inconsistency, flagged rather than silently reconciled (AGENTS.md §3:
do not edit one side of a documented contradiction to make it disappear).

Two environment facts are encoded rather than left to documentation:

- **No GPU.** ``ml.r5.xlarge`` is a CPU-only instance family. A stray
  ``task_type="GPU"`` fails at fit time with an opaque error, so the value is
  pinned to ``CPU`` and any request for GPU is refused with an explanation.
- **Determinism.** ``random_seed``, ``thread_count`` and ``allow_writing_files``
  are all pinned. The last one matters more than it looks: by default CatBoost
  writes a ``catboost_info/`` directory next to the working directory on every
  fit, which is both a repository-pollution problem and a source of files that
  look like results but are not.

Probability is not confidence
-----------------------------
Raw CatBoost output is a ranking score, not a calibrated probability, and the
decision layer needs something it can threshold and compare across entities.
:func:`calibrate_isotonic` fits an isotonic mapping on a held-out fold, kept as
a separate step so the calibration surface is never trained on the same rows it
is evaluated on. Whether calibration is applied at all is a PROPOSAL until an
experiment shows it moves entity-level $F_{0.5}$; pair-level AUC cannot answer
that question.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from team_diamond.models.matrix import ModelMatrix

__all__ = ["CatBoostMatcher", "MatcherParams", "FittedCalibrator", "DEFAULT_PARAMS"]

#: Sensible starting parameters. Every one of these is a PROPOSAL until swept;
#: the config file is the place to record what was actually chosen, and
#: :meth:`CatBoostMatcher.from_config` prefers the config over these.
DEFAULT_PARAMS: Final[dict[str, Any]] = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 2_000,
    "learning_rate": 0.08,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "rsm": 0.8,
    "random_seed": 20260926,
    "task_type": "CPU",
    "allow_writing_files": False,
    "verbose": False,
}


@dataclass(frozen=True, slots=True)
class MatcherParams:
    """Validated CatBoost parameters.

    Attributes:
        iterations: Tree count. Also the early-stopping ceiling, so it should be
            set generously and cut short by ``early_stopping_rounds``.
        learning_rate: Shrinkage. Trades iterations against overfitting.
        depth: Tree depth. Deeper trees capture feature interactions between
            name, address, and rarity evidence, which is where most of the
            signal is; they also overfit small positive counts, and positives
            are a small fraction of pairs.
        l2_leaf_reg: Leaf L2 penalty.
        random_strength: Random score noise for split selection.
        rsm: Feature subsampling rate, per tree.
        random_seed: Explicit seed, for reproducibility (AGENTS.md §16).
        thread_count: Cores to use. Defaults to the real core count, never a
            hardcoded assumption about the machine.
        early_stopping_rounds: Stop after this many evaluations without
            improvement on the eval set.
        class_weights: Optional ``"scale_pos_weight"`` or ``"balanced"``.
            Left ``None`` by default: weighting shifts the probabilities, which
            then interacts with calibration, and the cleaner way to move the
            operating point is the decision layer's threshold.
    """

    iterations: int = 2_000
    learning_rate: float = 0.08
    depth: int = 8
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.0
    rsm: float = 0.8
    random_seed: int = 20260926
    thread_count: int | None = None
    early_stopping_rounds: int | None = 200
    class_weights: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_threads(self) -> int:
        """The thread count actually used."""
        if self.thread_count is not None:
            return self.thread_count
        return max(1, os.cpu_count() or 1)

    def to_catboost(self) -> dict[str, Any]:
        """Render the keyword arguments for ``CatBoostClassifier``."""
        params: dict[str, Any] = {
            "loss_function": DEFAULT_PARAMS["loss_function"],
            "eval_metric": DEFAULT_PARAMS["eval_metric"],
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "random_strength": self.random_strength,
            "rsm": self.rsm,
            "random_seed": self.random_seed,
            "task_type": "CPU",
            "allow_writing_files": False,
            "verbose": False,
            "thread_count": self.resolved_threads(),
            **self.extra,
        }
        if self.class_weights == "balanced":
            params["auto_class_weights"] = "Balanced"
        elif self.class_weights == "scale_pos_weight":
            params["auto_class_weights"] = "SqrtBalanced"
        return params


@dataclass(slots=True)
class FittedCalibrator:
    """Isotonic calibration, kept separate from the model.

    Attributes:
        mapping: Fitted ``sklearn.isotonic.IsotonicRegression``, or ``None``
            when calibration was not applied.
        fitted: Whether a mapping is present.
    """

    mapping: Any | None = None

    @property
    def fitted(self) -> bool:
        return self.mapping is not None

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities.

        Args:
            scores: Raw model scores in [0, 1].

        Returns:
            Calibrated probabilities, or ``scores`` unchanged when no mapping
            has been fitted.
        """
        if self.mapping is None:
            return scores
        return np.asarray(self.mapping.predict(scores), dtype=np.float64)

    def save(self, path: Path) -> None:
        """Persist the mapping alongside the model."""
        import pickle

        with path.open("wb") as handle:
            pickle.dump(self.mapping, handle)

    @classmethod
    def load(cls, path: Path) -> FittedCalibrator:
        """Restore a mapping, tolerating its absence."""
        if not path.is_file():
            return cls(mapping=None)
        import pickle

        with path.open("rb") as handle:
            return cls(mapping=pickle.load(handle))


class CatBoostMatcher:
    """Trainable, saveable pairwise match classifier."""

    def __init__(self, params: MatcherParams | None = None) -> None:
        self.params = params or MatcherParams()
        self._model: Any | None = None
        self._feature_names: tuple[str, ...] = ()
        self._cat_indices: tuple[int, ...] = ()
        self._vocabularies: dict[str, dict[str, int]] = {}
        self.calibrator = FittedCalibrator()
        self._best_iteration: int | None = None

    # -- construction --------------------------------------------------------
    @classmethod
    def from_config(cls, config: Any) -> CatBoostMatcher:
        """Build from a :class:`~team_diamond.config.Config`.

        Only values present and non-null in the config override the defaults,
        so a config full of placeholders still produces a runnable model -- and
        :meth:`~team_diamond.config.Config.unmeasured` is what records that it
        is running on defaults.

        Args:
            config: Merged configuration, typically ``load_config("model")``.

        Returns:
            An unfitted matcher.

        Raises:
            ValueError: If ``model.task_type`` asks for a GPU.
        """
        task_type = config.get("model.params.task_type", "CPU")
        if str(task_type).upper() != "CPU":
            raise ValueError(
                f"model.params.task_type={task_type!r} is not supported: the "
                f"target instance (ml.r5.xlarge) is a CPU-only family and "
                f"CatBoost will fail at fit time with an opaque error. Use CPU, "
                f"or move to a GPU family and revisit."
            )

        params = MatcherParams(
            iterations=int(config.get("model.params.iterations", DEFAULT_PARAMS["iterations"])),
            learning_rate=float(config.get("model.params.learning_rate", DEFAULT_PARAMS["learning_rate"])),
            depth=int(config.get("model.params.depth", DEFAULT_PARAMS["depth"])),
            l2_leaf_reg=float(config.get("model.params.l2_leaf_reg", DEFAULT_PARAMS["l2_leaf_reg"])),
            random_strength=float(config.get("model.params.random_strength", DEFAULT_PARAMS["random_strength"])),
            rsm=float(config.get("model.params.rsm", DEFAULT_PARAMS["rsm"])),
            random_seed=int(config.get("model.params.random_seed", DEFAULT_PARAMS["random_seed"])),
            thread_count=config.get("model.params.thread_count"),
            early_stopping_rounds=config.get("model.params.early_stopping_rounds", 200),
            class_weights=config.get("model.params.class_weights"),
        )
        return cls(params)

    # -- fitting -------------------------------------------------------------
    def fit(
        self,
        train: ModelMatrix,
        *,
        eval_matrix: ModelMatrix | None = None,
    ) -> CatBoostMatcher:
        """Fit the classifier.

        Args:
            train: Training matrix with labels.
            eval_matrix: Held-out matrix for early stopping. When omitted, the
                full ``iterations`` are used and no early stopping happens --
                which is fine for a final refit on all data and wrong for a
                run whose quality you intend to read off a validation number.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If ``train`` has no labels, or the eval matrix's
                feature names or categorical positions disagree with training.
                A mismatch there is the classic silent-corruption route, so it
                is refused explicitly.
        """
        from catboost import CatBoostClassifier, Pool

        if train.y is None:
            raise ValueError("train matrix has no labels; nothing to fit on")
        if train.X.shape[0] == 0:
            raise ValueError("train matrix has no rows")

        self._feature_names = train.feature_names
        self._cat_indices = train.cat_indices
        self._vocabularies = dict(train.categorical_vocabularies)

        self._model = CatBoostClassifier(**self.params.to_catboost())

        train_pool = Pool(
            train.X, label=train.y, cat_features=list(self._cat_indices) or None
        )
        fit_kwargs: dict[str, Any] = {}
        if eval_matrix is not None:
            self._check_compatible(eval_matrix)
            eval_pool = Pool(
                eval_matrix.X,
                label=eval_matrix.y,
                cat_features=list(self._cat_indices) or None,
            )
            fit_kwargs["eval_set"] = [eval_pool]
            if self.params.early_stopping_rounds:
                fit_kwargs["early_stopping_rounds"] = self.params.early_stopping_rounds
            fit_kwargs["use_best_model"] = True

        self._model.fit(train_pool, **fit_kwargs)
        self._best_iteration = getattr(self._model, "best_iteration_", None)
        return self

    def _check_compatible(self, other: ModelMatrix) -> None:
        """Refuse a matrix that would be interpreted wrongly."""
        if self._feature_names and other.feature_names != self._feature_names:
            raise ValueError(
                f"eval matrix feature order differs from training. The model "
                f"is positional, so a reordered matrix would be scored against "
                f"the wrong features and report a confident, wrong number.\n"
                f"  train: {self._feature_names[:6]}...\n"
                f"  eval : {other.feature_names[:6]}..."
            )
        if self._cat_indices and other.cat_indices != self._cat_indices:
            raise ValueError(
                f"eval matrix categorical positions {other.cat_indices} differ "
                f"from training {self._cat_indices}"
            )

    # -- prediction ----------------------------------------------------------
    def predict_proba(self, matrix: ModelMatrix) -> np.ndarray:
        """Score pairs.

        Args:
            matrix: Feature matrix, with or without labels.

        Returns:
            Calibrated match probability per row, shape ``(n,)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the matrix layout disagrees with training.
        """
        if self._model is None:
            raise RuntimeError("predict_proba called before fit")
        self._check_compatible(matrix)
        raw = self._model.predict_proba(matrix.X)[:, 1]
        return self.calibrator.apply(raw)

    def score(self, matrix: ModelMatrix) -> np.ndarray:
        """Alias for :meth:`predict_proba`, for readability at call sites."""
        return self.predict_proba(matrix)

    # -- calibration ---------------------------------------------------------
    def fit_calibrator(
        self, matrix: ModelMatrix, *, target: ModelMatrix | None = None
    ) -> FittedCalibrator:
        """Fit isotonic calibration on held-out scores.

        Isotonic is monotone piecewise-constant, so it cannot invert the
        model's ranking -- it can only fix the mapping from score to
        probability. That is the property wanted here: the decision layer
        relies on both, and a calibrator that reorders candidates would silently
        change which pair wins.

        Args:
            matrix: Scored matrix **with** labels, and ideally a fold the model
                was not fitted on. Fitting calibration on the training fold
                would produce an optimistic mapping and an optimistic decision
                threshold.
            target: Optional separate matrix to fit on, when ``matrix`` is used
                for something else.

        Returns:
            The fitted calibrator.
        """
        from sklearn.isotonic import IsotonicRegression

        source = target if target is not None else matrix
        if source.y is None:
            raise ValueError(
                "calibration needs labels; pass a matrix with a target column"
            )
        raw = self._model.predict_proba(source.X)[:, 1]
        mapping = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        mapping.fit(raw, source.y)
        self.calibrator = FittedCalibrator(mapping=mapping)
        return self.calibrator

    # -- interpretation ------------------------------------------------------
    def feature_importance(self, *, top: int | None = None) -> pl.DataFrame:
        """Per-feature importance, most important first.

        Returns:
            Frame with ``feature``, ``importance``, and ``rank``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._model is None:
            raise RuntimeError("feature_importance called before fit")
        values = self._model.get_feature_importance()
        frame = pl.DataFrame(
            {
                "feature": list(self._feature_names),
                "importance": [float(v) for v in values],
            }
        ).sort("importance", descending=True)
        if top is not None:
            frame = frame.head(top)
        return frame.with_columns(
            pl.col("importance").rank("ordinal", descending=False).cast(pl.Int32).alias("rank")
        )

    def best_iteration(self) -> int | None:
        """Iteration early stopping selected, or ``None`` if it did not run."""
        return self._best_iteration

    def categorical_vocabularies(self) -> dict[str, dict[str, int]]:
        """The frozen categorical encodings this model was fitted on.

        Pass this to :func:`~team_diamond.models.matrix.to_model_matrix` when
        scoring new data. Reusing it is what guarantees a categorical column is
        encoded with the codes the model learned; refitting a vocabulary against
        test data would silently permute the codes, and the model would still
        emit confident predictions from the permuted inputs.

        Returns:
            A copy of the ``label -> code`` mapping per categorical column. Empty
            for an unfitted model.
        """
        return {name: dict(vocab) for name, vocab in self._vocabularies.items()}

    # -- persistence ---------------------------------------------------------
    def save(self, directory: Path) -> Path:
        """Write the model, its metadata, and its calibrator to a directory.

        The feature order and categorical vocabularies are written alongside the
        model because without them the artefact cannot be loaded safely -- see
        :mod:`team_diamond.models.matrix` for why that is not optional.

        Args:
            directory: Target directory, created if absent.

        Returns:
            The directory written.
        """
        if self._model is None:
            raise RuntimeError("cannot save an unfitted model")
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(directory / "matcher.cbm"))
        meta = {
            "feature_names": list(self._feature_names),
            "cat_indices": list(self._cat_indices),
            "categorical_vocabularies": self._vocabularies,
            "params": {
                k: v for k, v in self.params.to_catboost().items() if k != "verbose"
            },
            "best_iteration": self._best_iteration,
            "calibrated": self.calibrator.fitted,
        }
        (directory / "matcher_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        self.calibrator.save(directory / "calibrator.pkl")
        return directory

    @classmethod
    def load(cls, directory: Path) -> CatBoostMatcher:
        """Restore a model saved by :meth:`save`.

        Args:
            directory: Directory written by :meth:`save`.

        Returns:
            A ready matcher, with its calibrator if one was fitted.

        Raises:
            FileNotFoundError: If the model or metadata file is missing.
        """
        from catboost import CatBoostClassifier

        model_path = directory / "matcher.cbm"
        meta_path = directory / "matcher_meta.json"
        if not model_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"{directory} does not contain a saved matcher "
                f"(expected {model_path.name} and {meta_path.name})"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        restored = cls()
        restored._model = CatBoostClassifier()
        restored._model.load_model(str(model_path))
        restored._feature_names = tuple(meta["feature_names"])
        restored._cat_indices = tuple(meta["cat_indices"])
        restored._vocabularies = meta["categorical_vocabularies"]
        restored._best_iteration = meta.get("best_iteration")
        restored.calibrator = FittedCalibrator.load(directory / "calibrator.pkl")
        return restored

    def describe(self) -> str:
        """Summary of the fitted state, for an experiment log."""
        lines = [
            "CatBoostMatcher",
            f"  fitted          : {self._model is not None}",
            f"  features        : {len(self._feature_names)}",
            f"  categorical idx : {list(self._cat_indices)}",
            f"  best iteration  : {self._best_iteration}",
            f"  calibrated      : {self.calibrator.fitted}",
            f"  threads         : {self.params.resolved_threads()}",
            f"  seed            : {self.params.random_seed}",
        ]
        return "\n".join(lines)
