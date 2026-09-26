"""Entity-level match selection.

Layer: decision
See AGENTS.md §13 (Decision rules), §17 (Submission contract)

Three owners, three questions
-----------------------------
AGENTS.md §7 is explicit that "should we emit this candidate?" belongs to the
decision layer alone, and :mod:`team_diamond.retrieval.candidates` must not be
reached into from here. This module therefore takes scored pairs and returns
id lists, and does nothing else.

Why a single threshold is the wrong shape
-----------------------------------------
AGENTS.md §13 forbids ``prediction = probability > 0.5`` and gives the reason:
``A=0.97, B=0.31`` and ``A=0.54, B=0.51`` both contain a candidate above 0.5 and
are entirely different situations. The first is a confident singleton; the
second is an ambiguous entity where emitting the top choice may well be wrong.

The marginal-value rule
-----------------------
Rather than invent a heuristic, :class:`MarginalValuePolicy` derives its
threshold from the metric. For one entity with true match set size $m$,
$k$ correct predictions out of $n$ emitted:

.. math::

    F_\\beta = \\frac{(1+\\beta)k}{\\beta n + m}

Consider emitting one more candidate whose probability of being a true match
is $p$. Averaging over the two outcomes and requiring the expected score to
improve gives

.. math::

    p > \\frac{\\beta k}{m + \\beta n}

Two consequences fall out, and they are the whole point:

- **The threshold is not constant.** With nothing emitted ($k=n=0$) the bound is
  $p>0$: emitting a first candidate is *always* worth it, however unlikely.
  After one correct emission ($k=n=1$, $m=1$) the bound is $\beta/(1+\\beta)$,
  which for $\\beta=0.5$ is 0.20 -- lower than the first, because a second
  guess on a one-match entity is cheap optionality.
- **Thresholds fall as $k$ grows and rise as $n$ grows**, so the rule
  automatically becomes stricter the more speculative the entity gets, without
  anyone tuning a schedule.

$m$ is the true cardinality and is **unobserved at inference time**, so it is
supplied as a prior (``cardinality_prior``) that must be swept, not assumed.
$k$ is unobserved too, and is plugged in with its expected value: the sum of
the probabilities already emitted. That substitution is the one approximation
in the derivation and is called out in the class docstring.

No transitive closure
---------------------
Nothing here merges entities, unions neighbourhoods, or propagates a match
through a chain. AGENTS.md §13 is explicit that $A\\!\\to\\!B$, $B\\!\\to\\!C$
does not imply $A\\!\\to\\!C$; cross-source relationships are evidence, and
evidence belongs in the feature layer where it can be weighed rather than
asserted.

Choosing a policy
-----------------
Every policy here is a **PROPOSAL** until it wins a controlled comparison on
entity-level $F_{0.5}$. :func:`compare_policies` exists to run that comparison
against the measured metric rather than against intuition, and
:func:`simplest_within_tolerance` implements AGENTS.md §13's "ship the simplest
policy that is statistically indistinguishable from the best".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import numpy as np
import polars as pl

__all__ = [
    "DecisionPolicy",
    "EmitNothingPolicy",
    "FixedThresholdPolicy",
    "TopKPolicy",
    "MarginalValuePolicy",
    "MarginalValuePolicyParams",
    "PolicyOutcome",
    "apply_policy",
    "compare_policies",
    "simplest_within_tolerance",
]

#: Beta for the competition metric. Entity-level macro F0.5 (AGENTS.md §12).
BETA: Final[float] = 0.5

#: Score column the decision layer reads. Fixed, so a policy can never be
#: pointed at a raw feature by accident and start thresholding a similarity.
SCORE_COLUMN: Final[str] = "match_probability"


class DecisionPolicy(Protocol):
    """Select which scored candidates to emit for one entity."""

    name: str

    def select(self, scores: Sequence[float]) -> list[int]:
        """Return the indices of ``scores`` to emit, best first.

        Args:
            scores: Candidate probabilities for one S1 entity, **in descending
                order**. Implementations may assume the ordering and must not
                rely on the original pair order.

        Returns:
            Indices into ``scores``. **May be empty** -- a singleton or a
            no-match entity is a legitimate outcome and must never be forced
            into a guess (AGENTS.md §13).
        """
        ...


@dataclass(frozen=True, slots=True)
class EmitNothingPolicy:
    """Never emit a match.

    Not a strawman. With 5.58% of training entities being singletons, "predict
    nothing anywhere" has a real $F_{0.5}$, and any policy must beat it. It is
    the floor that keeps a confident-looking policy honest.
    """

    name: Final[str] = "emit_nothing"

    def select(self, scores: Sequence[float]) -> list[int]:
        """Return nothing, always."""
        return []


@dataclass(frozen=True, slots=True)
class FixedThresholdPolicy:
    """Emit every candidate at or above one global threshold, capped.

    This is the ``probability > 0.5`` shape that AGENTS.md §13 rules out, kept
    as the baseline it is meant to be beaten by.
    """

    threshold: float = 0.5
    max_per_entity: int = 5
    name: Final[str] = "fixed_threshold"

    def select(self, scores: Sequence[float]) -> list[int]:
        """Indices of candidates at or above ``threshold``, up to the cap."""
        out: list[int] = []
        for index, score in enumerate(scores):
            if score < self.threshold:
                break
            out.append(index)
            if len(out) >= self.max_per_entity:
                break
        return out


@dataclass(frozen=True, slots=True)
class TopKPolicy:
    """Emit the top ``k`` candidates unconditionally.

    Included because it is the natural rival to any confidence-based rule: if
    the model's *ranking* is good and its *calibration* is poor, ``top-k`` wins,
    and finding out which of the two is broken is worth an experiment.
    """

    k: int = 1
    min_score: float = 0.0
    name: Final[str] = "top_k"

    def select(self, scores: Sequence[float]) -> list[int]:
        """The first ``k`` indices whose score clears ``min_score``."""
        out: list[int] = []
        for index, score in enumerate(scores):
            if score < self.min_score:
                break
            out.append(index)
            if len(out) >= self.k:
                break
        return out


@dataclass(frozen=True, slots=True)
class MarginalValuePolicyParams:
    """Parameters for :class:`MarginalValuePolicy`.

    Attributes:
        beta: The $F_\\beta$ the decision optimises. Fixed at 0.5 for this
            competition; exposed only so a sweep can confirm the metric, not a
            tuned number, is what drives the rule.
        cardinality_prior: Stand-in for the unobserved $m$. **Must be swept.**
            A value of 1.0 is the natural starting point for "most entities
            have one match", but the measured training mean is 3.46, so 1.0 is
            not a safe default -- it is a starting point.
        max_per_entity: Hard ceiling on emissions. A safety valve, not a policy:
            measured cardinality has a long tail and a model bug should not be
            able to emit 400 ids for one entity.
        min_score: Absolute floor below which nothing is emitted regardless of
            the marginal rule. Guards the case where a badly calibrated model
            makes every marginal test pass.
    """

    beta: float = BETA
    cardinality_prior: float = 1.0
    max_per_entity: int = 5
    min_score: float = 0.0


@dataclass(frozen=True, slots=True)
class MarginalValuePolicy:
    """Emit while the metric says another guess pays for itself.

    Implements $p_i > \\beta\\hat{k} / (m_\\text{prior} + \\beta n)$, walking
    candidates in descending score order, with $\\hat{k}$ the expected number
    of correct emissions so far.

    **PROPOSAL.** This is a derived rule, not a measured one. It must be
    compared against :class:`FixedThresholdPolicy` and :class:`TopKPolicy` on
    entity-level $F_{0.5}$ before it is adopted, and ``cardinality_prior`` must
    be swept rather than set from a measured mean and forgotten.
    """

    params: MarginalValuePolicyParams = field(
        default_factory=MarginalValuePolicyParams
    )
    name: Final[str] = "marginal_value"

    def threshold(self, *, expected_correct: float, emitted: int) -> float:
        """The score bound for the next candidate.

        Args:
            expected_correct: $\\hat{k}$, the expected number of correct
                emissions so far.
            emitted: $n$, how many have been emitted.

        Returns:
            The minimum score worth emitting.
        """
        p = self.params
        denominator = p.cardinality_prior + p.beta * emitted
        if denominator <= 0:
            return float("inf")
        return (p.beta * expected_correct) / denominator

    def select(self, scores: Sequence[float]) -> list[int]:
        """Indices whose score clears the running marginal bound."""
        p = self.params
        out: list[int] = []
        expected_correct = 0.0
        for index, score in enumerate(scores):
            if index >= p.max_per_entity:
                break
            if score < p.min_score:
                break
            if score <= self.threshold(
                expected_correct=expected_correct, emitted=index
            ):
                # Sorted descending, so the first failure is final: every later
                # candidate scores lower and faces a higher bound.
                break
            out.append(index)
            expected_correct += score
        return out


#: Policies in the order a reader should consider them: simplest first.
DEFAULT_POLICIES: Final[tuple[DecisionPolicy, ...]] = (
    EmitNothingPolicy(),
    TopKPolicy(k=1),
    FixedThresholdPolicy(),
    MarginalValuePolicy(),
)


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    """What one policy did, and what it scored.

    Attributes:
        name: Policy name.
        f05: Entity-level macro $F_{0.5}$, the metric that decides.
        n_entities: Entities evaluated.
        n_emitted: Total ids emitted across all entities.
        n_true_positive: Correct emissions.
        mean_emitted_per_entity: Emissions per entity; 0.0 for a policy that
            never emits.
        n_empty: Entities with an empty prediction, which is a legitimate
            outcome and not an error.
        entities_without_candidates: S1 entities that were never scored at all.
            Non-zero means retrieval never produced a candidate for them, so
            they score 0 by construction and the ceiling is lower than the model.
        pooled_f0_5: Count-pooled variant, tracked only to quantify risk R1 (the
            official scorer might pool rather than macro-average). Not the target.
        pair_precision: Diagnostic only (AGENTS.md §12).
        pair_recall: Diagnostic only.
    """

    name: str
    f05: float
    n_entities: int
    n_emitted: int
    n_true_positive: int
    mean_emitted_per_entity: float
    n_empty: int
    pair_precision: float
    pair_recall: float
    entities_without_candidates: int = 0
    pooled_f0_5: float = 0.0

    def row(self) -> dict[str, Any]:
        """Flat mapping, for a results table."""
        return {
            "policy": self.name,
            "entity_f05": round(self.f05, 6),
            "entities": self.n_entities,
            "emitted": self.n_emitted,
            "true_positive": self.n_true_positive,
            "mean_emitted": round(self.mean_emitted_per_entity, 4),
            "n_empty": self.n_empty,
            "no_candidates": self.entities_without_candidates,
            "pair_precision": round(self.pair_precision, 6),
            "pair_recall": round(self.pair_recall, 6),
            "pooled_f05": round(self.pooled_f0_5, 6),
        }


def apply_policy(
    scored: pl.DataFrame,
    policy: DecisionPolicy,
    *,
    score_column: str = SCORE_COLUMN,
    s1_column: str = "s1_id",
    vendor_column: str = "vendor_id",
) -> pl.DataFrame:
    """Run a policy over scored candidates, producing per-entity emissions.

    Args:
        scored: Candidate pairs with a score column. Row order within an
            entity is irrelevant; candidates are re-sorted here.
        policy: The policy to apply.
        score_column: Column holding match probability.
        s1_column: Entity id column.
        vendor_column: Vendor id column.

    Returns:
        Frame with ``s1_entity_id`` and ``matched_entity_ids`` (``List[Utf8]``,
        possibly empty) for **every** entity present in ``scored``, sorted by
        ``s1_entity_id`` so the output is deterministic.

    Raises:
        ValueError: If a required column is missing.
    """
    for column in (score_column, s1_column, vendor_column):
        if column not in scored.columns:
            raise ValueError(
                f"scored frame is missing {column!r}; columns: {scored.columns[:10]}"
            )

    ordered = scored.sort([s1_column, score_column], descending=[False, True])

    emitted: list[str] = []
    lists: list[list[str]] = []
    current: str | None = None
    buffer: list[float] = []
    vendors: list[str] = []

    def flush() -> None:
        if current is None:
            return
        chosen = policy.select(buffer)
        emitted.append(current)
        lists.append([vendors[i] for i in chosen])

    for s1_id, score, vendor_id in ordered.select(
        s1_column, score_column, vendor_column
    ).iter_rows():
        if s1_id != current:
            flush()
            current = s1_id
            buffer = []
            vendors = []
        buffer.append(float(score))
        vendors.append(str(vendor_id))
    flush()

    return pl.DataFrame(
        {
            "s1_entity_id": pl.Series(emitted, dtype=pl.Utf8),
            "matched_entity_ids": pl.Series(lists, dtype=pl.List(pl.Utf8)),
        }
    ).sort("s1_entity_id")


def compare_policies(
    scored: pl.DataFrame,
    truth: pl.DataFrame,
    policies: Iterable[DecisionPolicy] = DEFAULT_POLICIES,
    *,
    score_column: str = SCORE_COLUMN,
    s1_column: str = "s1_id",
    vendor_column: str = "vendor_id",
    truth_s1_column: str = "s1_entity_id",
    truth_list_column: str = "matched_entity_ids",
) -> pl.DataFrame:
    """Score several policies on the same labelled data, ranked by $F_{0.5}$.

    This is the controlled comparison AGENTS.md §13 requires before any policy
    is adopted. Everything is evaluated on one fixed candidate set, so a policy
    is not accidentally credited with a different retrieval result.

    Args:
        scored: Labelled-or-unlabelled candidate pairs with scores.
        truth: Frame with ``truth_s1_column`` and ``truth_list_column``.
        policies: Policies to compare.
        score_column: Score column in ``scored``.
        s1_column: Entity id column in ``scored``.
        vendor_column: Vendor id column in ``scored``.
        truth_s1_column: Entity id column in ``truth``.
        truth_list_column: List-of-ids column in ``truth``.

    Returns:
        One :class:`PolicyOutcome` row per policy, best $F_{0.5}$ first.
    """
    truth_map = _truth_to_mapping(truth, truth_s1_column, truth_list_column)
    scored_s1 = scored[s1_column].unique()

    outcomes: list[PolicyOutcome] = []
    for policy in policies:
        predictions = apply_policy(
            scored, policy, score_column=score_column,
            s1_column=s1_column, vendor_column=vendor_column,
        )
        pred_map = _predictions_to_mapping(predictions)
        outcomes.append(
            _score_policy(policy.name, truth_map, pred_map, scored_s1)
        )
    return pl.DataFrame([o.row() for o in outcomes]).sort("entity_f05", descending=True)


def _truth_to_mapping(
    truth: pl.DataFrame, s1_column: str, list_column: str
) -> dict[str, set[str]]:
    """Ground-truth frame to the ``{s1_id: {match ids}}`` mapping the metric takes.

    Raises:
        ValueError: If a column is missing, or a row's list contains a non-string.
    """
    for column in (s1_column, list_column):
        if column not in truth.columns:
            raise ValueError(
                f"truth frame is missing {column!r}; columns: {truth.columns[:10]}"
            )
    out: dict[str, set[str]] = {}
    for s1_id, ids in truth.select(s1_column, list_column).iter_rows():
        if not isinstance(ids, (list, tuple)):
            raise ValueError(
                f"truth {s1_column}={s1_id!r}: expected a list of ids, got "
                f"{type(ids).__name__}. Explode a comma-separated string first."
            )
        bad = [x for x in ids if not isinstance(x, str)]
        if bad:
            raise ValueError(
                f"truth {s1_id!r}: {len(bad)} non-string id(s), e.g. {bad[:3]}. "
                f"Source ids are opaque (AGENTS.md §8.3) and must not be cast."
            )
        out[str(s1_id)] = set(ids)
    return out


def _predictions_to_mapping(predictions: pl.DataFrame) -> dict[str, list[str]]:
    """Emissions frame to ``{s1_id: [ids]}``."""
    return {
        str(s1_id): list(ids)
        for s1_id, ids in predictions.select(
            "s1_entity_id", "matched_entity_ids"
        ).iter_rows()
    }


def _score_policy(
    name: str,
    truth: dict[str, set[str]],
    pred: dict[str, list[str]],
    scored_s1: pl.Series,
) -> PolicyOutcome:
    """Score one policy's emissions, aggregating the per-entity breakdown.

    Entity coverage is reconciled explicitly against the candidate set, because
    two different bugs produce the same understated entity count:

    - an S1 entity whose true matches were never retrieved, and
    - an S1 entity that retrieval never queried at all.

    Both must read as zero, and neither may quietly shrink the denominator, so
    the count of missing entities is reported rather than absorbed.
    """
    from team_diamond.evaluation.metrics import pooled_f0_5, score_entities

    scores = score_entities(truth, pred)
    f05 = float(np.mean([s.f0_5 for s in scores])) if scores else float("nan")

    n_entities = len(scores)
    n_emitted = sum(s.k for s in scores)
    n_true_positive = sum(s.t for s in scores)
    n_empty = sum(1 for s in scores if s.k == 0)
    n_possible = sum(s.m for s in scores)

    pair_precision = n_true_positive / n_emitted if n_emitted else 0.0
    pair_recall = n_true_positive / n_possible if n_possible else 0.0

    return PolicyOutcome(
        name=name,
        f05=f05,
        n_entities=n_entities,
        n_emitted=n_emitted,
        n_true_positive=n_true_positive,
        mean_emitted_per_entity=(n_emitted / n_entities if n_entities else 0.0),
        n_empty=n_empty,
        pair_precision=pair_precision,
        pair_recall=pair_recall,
        entities_without_candidates=int(scored_s1.len() - len(pred)),
        pooled_f0_5=pooled_f0_5(truth, pred),
    )


def simplest_within_tolerance(
    table: pl.DataFrame,
    *,
    tolerance: float,
    complexity_column: str = "complexity",
) -> str:
    """Pick the simplest policy not meaningfully worse than the best.

    AGENTS.md §13: "Ship the simplest policy that is statistically
    indistinguishable from the best." Complexity is taken as an explicit column
    rather than inferred from policy names, because the ranking of two
    mechanisms by their Python type is a matter of taste, not fact.

    Args:
        table: Output of :func:`compare_policies` plus a complexity column.
        tolerance: Absolute $F_{0.5}$ margin treated as indistinguishable.
            Should be at least the run-to-run noise of the metric; measuring
            that noise is a prerequisite, and this function cannot guess it.
        complexity_column: Integer column where lower means simpler.

    Returns:
        The name of the chosen policy.

    Raises:
        ValueError: If ``table`` is empty or the complexity column is absent.
    """
    if table.is_empty():
        raise ValueError("no policy outcomes to choose from")
    if complexity_column not in table.columns:
        raise ValueError(
            f"table has no {complexity_column!r} column; add one so simplicity "
            f"is an explicit input to the choice rather than a matter of taste"
        )

    best = float(table["entity_f05"].max())
    eligible = table.filter(pl.col("entity_f05") >= best - tolerance)
    return str(eligible.sort(complexity_column).row(0, named=True)["policy"])
