"""training — Assembling a training set without leaking.

STATUS: implemented, unrun. No split, negative sample, or training frame has
been produced from the real data yet.

Modules
-------
``labels``
    :func:`~team_diamond.training.labels.label_candidates`, the one place in the
    project that reads ground truth. Isolated so that reaching labels requires a
    deliberate import, and so a labelled frame carries ``label_fold`` and
    ``label_source`` columns that can never appear in an inference frame by
    accident.
``split``
    Entity-aware train/validation partitioning, stratified by country and match
    count, with the strict-partial-matching property re-verified rather than
    assumed.
``negatives``
    Hard negatives drawn from the same retrieval index used at inference.

What is and is not a leak
-------------------------
The vendor pool is shared between folds and that is **not** a leak: S2 and S3
carry no labels, and the inference-time vendor pool is equally shared, so a
fold-scoped index over the same vendors faithfully simulates inference. Sizing a
validation index down instead would flatter the result for a reason unrelated to
model quality.

Attaching labels to a validation-fold entity, or mining negatives with
validation labels, **is** a leak. Both are blocked structurally: the split
happens before labelling, and :func:`~team_diamond.training.labels.label_candidates`
records which fold each label belongs to.

Layer contract (AGENTS.md §7)
-----------------------------
produce training and validation frames; never transform, score, or decide

Depends on
----------
data, retrieval, preprocessing

Must not
--------
reach the test split, or build any statistic over labels from more than the
training fold
"""

from __future__ import annotations

from team_diamond.training.labels import (
    FOLD_COLUMN,
    LABEL_COLUMN,
    LABEL_SOURCE_COLUMN,
    label_candidates,
    label_histogram,
    positive_pairs,
)
from team_diamond.training.negatives import (
    DEFAULT_NEGATIVES_PER_POSITIVE,
    NegativeSample,
    sample_hard_negatives,
)
from team_diamond.training.split import (
    DEFAULT_SEED,
    EntitySplit,
    Fold,
    assert_partial_matching,
    entity_aware_split,
    stratify_key,
)

__all__ = [
    "DEFAULT_NEGATIVES_PER_POSITIVE",
    "DEFAULT_SEED",
    "FOLD_COLUMN",
    "LABEL_COLUMN",
    "LABEL_SOURCE_COLUMN",
    "EntitySplit",
    "Fold",
    "NegativeSample",
    "assert_partial_matching",
    "entity_aware_split",
    "label_candidates",
    "label_histogram",
    "positive_pairs",
    "sample_hard_negatives",
    "stratify_key",
]
