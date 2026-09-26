"""Tests for the entity-level macro F0.5 metric.

These are the most important tests in the repository. Every later claim that a
change "improved F0.5" is measured by this function, so its edge cases are
pinned here rather than trusted.

Run: ``uv run pytest tests/ -v``
"""

from __future__ import annotations

import numpy as np
import pytest

from team_diamond.evaluation.metrics import (
    expected_f0_5,
    macro_f0_5,
    pooled_f0_5,
    score_entities,
)


class TestSingletonRule:
    """m == 0: predicting "no match" correctly scores 1.0, not 0.0."""

    def test_correct_singleton_scores_one(self) -> None:
        truth = {"A": []}
        pred = {"A": []}
        assert macro_f0_5(truth, pred) == 1.0

    def test_singleton_with_false_positive_scores_zero(self) -> None:
        truth = {"A": []}
        pred = {"A": ["X"]}
        assert macro_f0_5(truth, pred) == 0.0

    def test_missing_prediction_key_counts_as_singleton_correct(self) -> None:
        """No row in a submission means nothing was emitted — that is correct."""
        truth = {"A": []}
        assert macro_f0_5(truth, {}) == 1.0

    def test_several_ids_on_a_singleton_all_count_as_false_positives(self) -> None:
        truth = {"A": []}
        pred = {"A": ["X", "Y", "Z"]}
        assert macro_f0_5(truth, pred) == 0.0


class TestClosedForm:
    """The 1.25t / (0.25m + k) expression, checked against hand-computed values."""

    def test_perfect_prediction(self) -> None:
        truth = {"A": ["X", "Y"]}
        assert macro_f0_5(truth, {"A": ["X", "Y"]}) == pytest.approx(1.0)

    def test_half_recalled_perfect_precision(self) -> None:
        # m=2, k=1, t=1 -> 1.25*1 / (0.25*2 + 1) = 1.25/1.5
        truth = {"A": ["X", "Y"]}
        assert macro_f0_5(truth, {"A": ["X"]}) == pytest.approx(1.25 / 1.5)

    def test_full_recall_with_one_false_positive(self) -> None:
        # m=2, k=3, t=2 -> 1.25*2 / (0.25*2 + 3) = 2.5/3.5
        truth = {"A": ["X", "Y"]}
        assert macro_f0_5(truth, {"A": ["X", "Y", "Z"]}) == pytest.approx(2.5 / 3.5)

    @pytest.mark.parametrize(
        ("m", "k", "t", "expected"),
        [
            (4, 3, 3, 1.25 * 3 / (0.25 * 4 + 3)),   # under-emission
            (4, 5, 4, 1.25 * 4 / (0.25 * 4 + 5)),   # over-emission
            (1, 1, 1, 1.0),
            (6, 6, 5, 1.25 * 5 / (0.25 * 6 + 6)),
        ],
    )
    def test_closed_form_matches_arithmetic(self, m: int, k: int, t: int, expected: float) -> None:
        truth_ids = [f"TRUE{i}" for i in range(t)] + [f"MISS{i}" for i in range(m - t)]
        pred_ids = [f"TRUE{i}" for i in range(t)] + [f"FP{i}" for i in range(k - t)]
        assert macro_f0_5({"A": truth_ids}, {"A": pred_ids}) == pytest.approx(expected)

    def test_over_emission_costs_more_than_under_emission(self) -> None:
        """The asymmetry that motivates the whole decision layer.

        Dropping one true match is cheaper than adding one wrong one, because
        F0.5 weights precision (k in the denominator) more than recall.
        """
        truth = {"A": ["1", "2", "3", "4"]}
        under = macro_f0_5(truth, {"A": ["1", "2", "3"]})        # k=3, t=3
        over = macro_f0_5(truth, {"A": ["1", "2", "3", "4", "9"]})  # k=5, t=4
        assert under > over


class TestZeroCases:
    def test_no_emission_when_matches_exist_scores_zero(self) -> None:
        assert macro_f0_5({"A": ["X"]}, {"A": []}) == 0.0

    def test_wrong_id_scores_zero(self) -> None:
        assert macro_f0_5({"A": ["X"]}, {"A": ["Y"]}) == 0.0


class TestMacroVsPooled:
    """Macro averages entities; pooling aggregates pairs. They are not equal."""

    def test_macro_and_pooled_differ_on_mixed_cardinality(self) -> None:
        truth = {
            "big": [f"b{i}" for i in range(10)],   # 10 true matches
            "small": ["s0"],                        # 1 true match
        }
        pred = {"big": [f"b{i}" for i in range(10)], "small": ["WRONG"]}
        macro = macro_f0_5(truth, pred)
        pooled = pooled_f0_5(truth, pred)
        assert macro != pytest.approx(pooled)
        # macro is dragged down by the small entity failing entirely
        assert macro == pytest.approx(0.5)
        # pooled sums counts first: m=11, k=11 (10 correct + 1 wrong), t=10
        assert pooled == pytest.approx(1.25 * 10 / (0.25 * 11 + 11))

    def test_singletons_are_underpunished_by_pooling(self) -> None:
        """Why pooling is probably not the official semantics.

        A false positive on a singleton still costs under pooling — it adds to
        the denominator — but the singleton contributes no true pairs, so the
        damage is diluted rather than landing on that entity alone.
        """
        truth = {"A": [], "B": ["X"]}
        assert pooled_f0_5(truth, {"A": [], "B": ["X"]}) == pytest.approx(1.0)

        pred_bad = {"A": ["FP"], "B": ["X"]}
        # macro charges the failure to entity A alone: 0.5
        assert macro_f0_5(truth, pred_bad) == pytest.approx(0.5)
        # pooled blends it with B's clean pair: m=1, k=2, t=1
        assert pooled_f0_5(truth, pred_bad) == pytest.approx(1.25 / 2.25)
        assert pooled_f0_5(truth, pred_bad) > macro_f0_5(truth, pred_bad)


class TestRobustness:
    def test_duplicate_emitted_ids_do_not_double_count(self) -> None:
        truth = {"A": ["X"]}
        assert macro_f0_5(truth, {"A": ["X", "X", "X"]}) == pytest.approx(1.0)

    def test_empty_truth_returns_nan_not_zero(self) -> None:
        """Zero would read as "scored 0%"; nan correctly signals a broken input."""
        assert np.isnan(macro_f0_5({}, {}))

    def test_count_guard_is_defensive_only(self) -> None:
        """t <= min(m, k) always holds for set inputs, so the guard is unreachable
        through the public API. It exists for callers that pass raw counts."""
        assert score_entities({"A": ["X"]}, {"A": ["X", "Y"]})[0].t == 1

    def test_non_string_emitted_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must be strings"):
            macro_f0_5({"A": ["X"]}, {"A": [123]})  # type: ignore[dict-item]

    def test_score_entities_reports_breakdown(self) -> None:
        scores = score_entities({"A": ["X", "Y"]}, {"A": ["X"]})
        assert len(scores) == 1
        assert (scores[0].m, scores[0].k, scores[0].t) == (2, 1, 1)


class TestExpectedF05:
    """Drives the decision layer: how many candidates should we emit?"""

    def test_returns_one_entry_per_candidate_count(self) -> None:
        p = np.array([0.9, 0.5, 0.2])
        out = expected_f0_5(p, 3)
        assert out.shape == (4,)

    def test_emitting_nothing_is_worth_only_the_singleton_rate(self) -> None:
        """Regression guard for a real bug.

        An earlier version hardcoded ``k=0 -> 1.0``, which made "emit nothing"
        optimal for every entity and would have produced an all-empty
        submission. Emitting nothing is correct only for the 5.58% of entities
        that are true singletons.
        """
        out = expected_f0_5(np.array([0.9, 0.5, 0.2]), 3)
        assert out[0] == pytest.approx(0.055847, abs=1e-6)
        assert out[0] < out[1]

    def test_no_confidence_means_emit_nothing(self) -> None:
        """With no informative candidate, the singleton prior is the best guess."""
        out = expected_f0_5(np.array([0.01, 0.01, 0.01]), 3)
        assert int(np.argmax(out)) == 0

    def test_high_confidence_emits_several(self) -> None:
        """A confident entity should emit more than one id.

        Guards against a degenerate policy that always emits exactly one, which
        is the failure mode a naive `top-1` baseline would have.
        """
        out = expected_f0_5(np.array([0.99, 0.97, 0.95, 0.90, 0.10]), 5)
        assert int(np.argmax(out)) > 1

    def test_one_strong_candidate_beats_two_weak_ones(self) -> None:
        strong = expected_f0_5(np.array([0.99, 0.01]), 2)
        weak = expected_f0_5(np.array([0.55, 0.54]), 2)
        assert int(np.argmax(strong)) == 1
        assert int(np.argmax(weak)) == 2

    def test_curve_is_not_monotonic(self) -> None:
        """Emitting more must be able to hurt, or the rule would just emit all."""
        out = expected_f0_5(np.array([0.99, 0.99, 0.02, 0.02]), 4)
        assert out[2] > out[3]

    def test_rejects_unsorted_input(self) -> None:
        with pytest.raises(ValueError, match="sorted descending"):
            expected_f0_5(np.array([0.1, 0.9]), 2)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            expected_f0_5(np.array([0.9, 0.5]), 3)

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_expected_value_never_exceeds_one(self, n: int) -> None:
        """Regression guard for a real bug.

        An early version of ``_poisson_binomial`` did not conserve probability
        mass, so the "expectation" reached 1.58 — impossible for a quantity
        bounded by 1. Any bug in the marginalisation shows up here first.
        """
        for high in (0.1, 0.5, 0.9, 0.99):
            out = expected_f0_5(np.full(n, high), n)
            assert np.all(out >= -1e-12), f"negative expectation at p={high}"
            assert np.all(out <= 1.0 + 1e-12), f"expectation > 1 at p={high}: {out}"

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_poisson_binomial_conserves_mass(self, n: int) -> None:
        from team_diamond.evaluation.metrics import _poisson_binomial

        for high in (0.1, 0.5, 0.9, 0.99):
            dist = _poisson_binomial(np.full(n, high))
            assert dist.shape == (n + 1,)
            assert dist.sum() == pytest.approx(1.0)
            assert np.all(dist >= 0.0)
