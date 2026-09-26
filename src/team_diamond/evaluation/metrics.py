"""Entity-level macro F0.5 — the competition metric.

Layer: evaluation
See AGENTS.md §Metrics, §Metric Priority

This is the only number that decides whether a change was an improvement.
Pair-level precision/recall are diagnostic; see the note at the bottom.

Provenance
----------
Re-derived from first principles in ``suggestion.md`` §2 and cross-checked
against the expected-F0.5 curve in §3.5. The closed form:

    F0.5 = (1 + beta^2) * P * R / (beta^2 * P + R),  beta = 0.5
         = 1.25 * t / (0.25 * m + k)

where, per S1 entity, ``m`` = |true matches|, ``k`` = |emitted matches|,
``t`` = |true AND emitted|.

The metric is MACRO: the per-entity score is averaged over entities. It is not
a pooled micro count. That distinction is the single largest correctness risk in
this file, so :func:`macro_f0_5` and :func:`pooled_f0_5` both exist and are
tested against each other on data where they provably differ.

Edge cases are explicit rather than delegated to a library
----------------------------------------------------------
Three cases need a rule, and all three are easy to get wrong:

1. ``m == 0`` (a true singleton) and ``k == 0`` (we emitted nothing) -> **1.0**.
   Correctly predicting "no match" is a *success*, not an abstention.
2. ``m == 0`` and ``k > 0`` -> **0.0**. We invented matches for a singleton.
3. ``k == 0`` or ``t == 0`` with ``m > 0`` -> **0.0**. Precision or recall is
   zero, so the harmonic-style expression collapses to zero.

Cases 1 and 2 are the reason ``sklearn.fbeta_score`` must not be used here: it
has no representation for "truth is an empty set" and will either raise or
silently score it as if the entity did not exist.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "EntityScore",
    "macro_f0_5",
    "pooled_f0_5",
    "score_entities",
    "expected_f0_5",
]


@dataclass(frozen=True, slots=True)
class EntityScore:
    """Per-entity breakdown, for error analysis rather than for the headline.

    Attributes:
        s1_id: The S1 entity id.
        m: Number of true matches.
        k: Number of emitted matches.
        t: Number of correct matches.
        f0_5: This entity's contribution to the macro average, in [0, 1].
    """

    s1_id: str
    m: int
    k: int
    t: int
    f0_5: float


def _entity_f0_5(m: int, k: int, t: int) -> float:
    """Score a single entity from its three counts."""
    if m == 0:
        # Truth is a singleton. Emitting nothing is correct; emitting anything is not.
        return 1.0 if k == 0 else 0.0
    if k == 0 or t == 0:
        return 0.0
    return 1.25 * t / (0.25 * m + k)


def score_entities(
    truth: Mapping[str, Iterable[str]],
    pred: Mapping[str, Sequence[str]],
) -> list[EntityScore]:
    """Score every entity in ``truth`` and return the per-entity breakdown.

    Args:
        truth: ``{s1_id: set_of_true_match_ids}``. An empty iterable means a
            true singleton. **Every** S1 entity must be a key, including
            singletons — an entity missing from this mapping is a data bug, and
            silently skipping it would inflate the score.
        pred: ``{s1_id: emitted_match_ids}``. An entity absent here is treated
            as having emitted nothing, which is the correct interpretation of
            "no row for this entity" in a submission.

    Returns:
        One :class:`EntityScore` per key in ``truth``, in iteration order.

    Raises:
        ValueError: If an emitted id is not a string, or the counts contradict
            each other (t must not exceed m or k).
    """
    scores: list[EntityScore] = []
    for s1_id, true_ids in truth.items():
        true_set = set(true_ids)
        emitted = list(pred.get(s1_id, ()))
        if not all(isinstance(x, str) for x in emitted):
            raise ValueError(f"{s1_id}: emitted ids must be strings, got {emitted!r}")

        # A duplicate emitted id cannot earn credit twice; treat the prediction
        # as the *set* it semantically is.
        emitted_set = set(emitted)
        m, k = len(true_set), len(emitted_set)
        t = len(true_set & emitted_set)
        if t > m or t > k:
            raise ValueError(f"{s1_id}: inconsistent counts m={m} k={k} t={t}")
        scores.append(EntityScore(s1_id, m, k, t, _entity_f0_5(m, k, t)))
    return scores


def macro_f0_5(
    truth: Mapping[str, Iterable[str]],
    pred: Mapping[str, Sequence[str]],
) -> float:
    """Entity-level macro F0.5: the mean of the per-entity scores.

    This is the competition metric as we understand it (VERIFIED by
    re-derivation; see ``suggestion.md`` §2 and Appendix A).

    The official scorer is **not** in this repository, and one open risk (R1) is
    that it pools counts instead of averaging per entity. If that turns out to
    be true, this number is the wrong target. Use :func:`pooled_f0_5` to see
    both, and treat the gap as the size of that risk.

    Args:
        truth: ``{s1_id: true_match_ids}``; empty iterable means singleton.
        pred: ``{s1_id: emitted_match_ids}``; missing key means nothing emitted.

    Returns:
        The macro average in [0, 1]. Returns ``nan`` for empty ``truth`` rather
        than 0.0, because "no entities" is a broken input, not a zero score.
    """
    scores = score_entities(truth, pred)
    if not scores:
        return float("nan")
    return float(np.mean([s.f0_5 for s in scores]))


def pooled_f0_5(
    truth: Mapping[str, Iterable[str]],
    pred: Mapping[str, Sequence[str]],
) -> float:
    """Count-pooled F0.5 — the alternative semantics we are NOT optimising for.

    Aggregates every pair into one global confusion matrix, then computes
    F0.5 once. Included **only** to quantify risk R1: if this is what the
    official scorer does, optimising :func:`macro_f0_5` is optimising the wrong
    objective.

    Singletons contribute no pairs here, so they cannot be scored at all — which
    is itself an argument that pooling is unlikely to be the official semantics.
    """
    total_m = total_k = total_t = 0
    for s1_id, true_ids in truth.items():
        true_set = set(true_ids)
        emitted_set = set(pred.get(s1_id, ()))
        total_m += len(true_set)
        total_k += len(emitted_set)
        total_t += len(true_set & emitted_set)
    if total_k == 0 or total_t == 0:
        return 0.0
    return 1.25 * total_t / (0.25 * total_m + total_k)


# Measured prior over the number of true matches per S1 entity.
#
# EVIDENCE: MEASURED on the full training ground truth, 2,206,821 entities.
# Reproduce with the script in ``suggestion.md`` Appendix C; recorded as
# experiment E002 in ``experiments/registry.csv``. Re-verified 2026-09-26:
# mean m = 3.46125, singletons = 123,247.
#
# These are TRAIN counts. Test is a different country mix (France appears only
# in test), so the true test prior is UNMEASURED. Using the train prior for
# France is an explicit assumption, not a verified fact.
MATCH_COUNT_PRIOR: dict[int, float] = {
    0: 0.055847,
    1: 0.053994,
    2: 0.170020,
    3: 0.240551,
    4: 0.219374,
    5: 0.145887,
    6: 0.074713,
    7: 0.028987,
    8: 0.008464,
    9: 0.000238,
    10: 0.000016,
}
# Tail m >= 11 is 0.0004% of entities; lumped into m=10 to keep the prior
# finite. The resulting bias in expected F0.5 is far below measurement noise.


def _poisson_binomial(probs: np.ndarray) -> np.ndarray:
    """Distribution of the number of successes in independent Bernoulli trials.

    Args:
        probs: Independent success probabilities, shape ``(n,)``.

    Returns:
        Array of length ``n + 1``; index ``t`` is P(exactly t successes).
        Sums to 1.0.
    """
    dist = np.zeros(probs.size + 1, dtype=np.float64)
    dist[0] = 1.0
    for i, p in enumerate(probs):
        new = np.zeros(i + 2, dtype=np.float64)
        new[:-1] += dist[: i + 1] * (1.0 - p)   # trial fails: count unchanged
        new[1:] += dist[: i + 1] * p            # trial succeeds: count +1
        dist = new
    return dist


def expected_f0_5(p_correct: np.ndarray, n_candidates: int | None = None) -> np.ndarray:
    """Expected entity F0.5 for each choice of how many candidates to emit.

    The core of the decision layer (``suggestion.md`` §3). For each ``k``,
    marginalise the metric over the **measured** prior on the entity's true
    match count, rather than plugging in the mean. This distinction is the
    whole point of the function:

    - Marginalising gives ``E[F0.5 | emit top-k]`` and correctly makes ``k=0``
      worth only ``P(m=0) = 5.58%``. Emitting nothing is almost always wrong.
    - Plugging in ``E[m]`` and setting ``k=0 -> 1.0`` would make "emit nothing"
      optimal for **every** entity, i.e. an all-empty submission. That bug is
      exactly what an earlier draft of this file contained.

    Args:
        p_correct: Candidate probabilities, sorted **descending**, shape ``(K,)``.
        n_candidates: Defaults to ``len(p_correct)``.

    Returns:
        Array of shape ``(K + 1,)``. Index ``k`` is the expected F0.5 of
        emitting the top ``k`` candidates. ``argmax`` is the recommended ``k``.

    Notes:
        **Assumptions, stated so they can be attacked:**

        1. Each true match is independently captured by the corresponding
           top-ranked candidate with probability ``p_i``. Independence is an
           **approximation**.
        2. Because the ground truth is a **strict partial matching** (VERIFIED —
           7,638,365 pairs map to 7,638,365 distinct vendor ids, zero shared,
           experiment E003), true matches can be paired 1:1 with the top-``m``
           candidates without collision. This is what makes assumption 1
           coherent rather than merely convenient.
        3. The train prior applies to test. Unverified for France.
    """
    p = np.asarray(p_correct, dtype=np.float64)
    k_max = p.size if n_candidates is None else n_candidates
    if p.size != k_max:
        raise ValueError(f"p_correct has {p.size} entries, expected {k_max}")
    if k_max and np.any(np.diff(p) > 0):
        raise ValueError("p_correct must be sorted descending")

    out = np.empty(k_max + 1, dtype=np.float64)
    for k in range(k_max + 1):
        if k == 0:
            # Emitting nothing is correct only for a true singleton.
            out[k] = MATCH_COUNT_PRIOR[0]
            continue
        total = 0.0
        for m, p_m in MATCH_COUNT_PRIOR.items():
            if m == 0:
                # Singleton: any emission at all scores 0.
                continue
            n_trials = min(m, k)
            t_dist = _poisson_binomial(p[:n_trials])
            for t in range(n_trials + 1):
                total += p_m * t_dist[t] * _entity_f0_5(m, k, t)
        out[k] = total
    return out


# ---------------------------------------------------------------------------
# Why sklearn is not used here
# ---------------------------------------------------------------------------
# `sklearn.metrics.fbeta_score(beta=0.5)` computes the same harmonic expression,
# but only over pairs that exist. It has no concept of m == 0, so true
# singletons either raise or are dropped. Dropping them is the dangerous case:
# it deletes exactly the entities where the precision/recall tradeoff is most
# costly, and it would flatter every model that over-predicts.
