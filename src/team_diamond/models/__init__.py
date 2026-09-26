"""models — The pairwise match classifier.

STATUS: implemented, untrained. No model has been fitted and no F0.5 has been
measured, so every parameter in ``configs/model.yaml`` is still a PROPOSAL.

Modules
-------
``matrix``
    Feature frame to positional ``numpy`` matrix, with a strict column-order
    contract and a frozen categorical vocabulary. This is where a reordered
    column would otherwise be silently misread by a positionally-fitted model.
``catboost_matcher``
    The CatBoost classifier, its parameter set, isotonic calibration, and
    save/load that carries the feature order and vocabulary with it.

Why CatBoost
------------
AGENTS.md §11 names LightGBM as the default first model. **CatBoost is used by
explicit team decision.** The reasons: native categorical handling for
``country_pair``, and ordered target statistics that suit a problem where
records repeat with noise. ``lightgbm`` remains in ``pyproject.toml`` so the
choice is reversible by configuration rather than by reinstalling.

The deviation is recorded rather than quietly taken. ``configs/model.yaml``
still says ``type: lightgbm``; that contradiction is flagged, not silently
reconciled (AGENTS.md §3).

Layer contract (AGENTS.md §7)
-----------------------------
emit a calibrated per-pair probability; emit no id list and no decision

Depends on
----------
features (the FEATURE_COLUMNS contract)

Must not
--------
threshold a probability, cap emissions, or decide which ids to emit — that is
the decision layer's job
"""

from __future__ import annotations

from team_diamond.models.catboost_matcher import (
    DEFAULT_PARAMS,
    CatBoostMatcher,
    FittedCalibrator,
    MatcherParams,
)
from team_diamond.models.matrix import ModelMatrix, encode_categorical, to_model_matrix

__all__ = [
    "DEFAULT_PARAMS",
    "CatBoostMatcher",
    "FittedCalibrator",
    "MatcherParams",
    "ModelMatrix",
    "encode_categorical",
    "to_model_matrix",
]
