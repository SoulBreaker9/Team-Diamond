"""decision — *Should* we emit this candidate?

STATUS: implemented, unvalidated. Every policy here is a **PROPOSAL** until it
wins a controlled comparison on entity-level $F_{0.5}$ (AGENTS.md §13).

Modules
-------
``policy``
    The policies, the metric-derived marginal-value rule, and the comparison
    harness that decides between them.

The marginal-value rule
-----------------------
:class:`~team_diamond.decision.policy.MarginalValuePolicy` does not use a
guessed threshold. Requiring that one more emission raise the expected value of
$F_\\beta = (1+\\beta)k / (\\beta n + m)$ gives

.. math::

    p > \\frac{\\beta k}{m + \\beta n}

so the bound falls as correct emissions accumulate, rises as guesses accumulate,
and is zero when nothing has been emitted. It is derived, not tuned — but
derivation is not measurement, and it still has to beat
:class:`~team_diamond.decision.policy.FixedThresholdPolicy` on the real metric
before it ships.

Two quantities in that rule are unobservable at inference: the true cardinality
$m$ and the realised $k$. They are supplied as ``cardinality_prior`` and as the
sum of emitted probabilities respectively. **Both are approximations**, and the
cardinality prior in particular must be swept rather than set from the measured
training mean and forgotten.

Layer contract (AGENTS.md §7)
-----------------------------
consume scored candidates; emit an id list, possibly empty

Depends on
----------
models (for probabilities), evaluation (for scoring a comparison)

Must not
--------
retrieve candidates, alter the candidate set, or read ground truth. A policy
that needed labels would not be a policy.
"""

from __future__ import annotations

from team_diamond.decision.policy import (
    BETA,
    SCORE_COLUMN,
    DEFAULT_POLICIES,
    DecisionPolicy,
    EmitNothingPolicy,
    FixedThresholdPolicy,
    MarginalValuePolicy,
    MarginalValuePolicyParams,
    PolicyOutcome,
    TopKPolicy,
    apply_policy,
    compare_policies,
    simplest_within_tolerance,
)

__all__ = [
    "BETA",
    "DEFAULT_POLICIES",
    "SCORE_COLUMN",
    "DecisionPolicy",
    "EmitNothingPolicy",
    "FixedThresholdPolicy",
    "MarginalValuePolicy",
    "MarginalValuePolicyParams",
    "PolicyOutcome",
    "TopKPolicy",
    "apply_policy",
    "compare_policies",
    "simplest_within_tolerance",
]
