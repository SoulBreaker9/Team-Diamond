"""Recall statistics from a bitmask. The standalone/cumulative/marginal split.

Run:

    uv run pytest tests/test_recall_stats.py

Why these tests exist
--------------------
Three separate reporting defects shipped in E010, E011 and E012, all of them in
this arithmetic, all of them invisible in the output because the numbers were
plausible:

1. E010/E011 stored the *cumulative* union recall in a column named
   ``pair_recall``, making ``address_exact`` look like 59.0% standalone when it
   was 11.7%.
2. E012 computed ``retrieved_by_exactly_one`` as ``bit > 0 AND bit < all_bits``,
   which counts every pair using one to six strategies -- and contradicted the
   coverage histogram printed three lines above it.
3. The same three quantities were implemented separately in each driver, so a
   fix in one place did not reach the others.

These are unit tests on the arithmetic, plus one regression anchor against a
frozen real bitmask, because a unit test that only checks a synthetic case will
happily agree with a fresh bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from team_diamond.retrieval.recall_stats import (
    RECALL_REQUIRED_FOR_F05_95,
    coverage_histogram,
    popcount,
    recall_report,
    strategy_recall_table,
    union_recall,
    validate_bitmask,
)

#: Four strategies. Small enough that every expected value is checkable by hand.
PLAN = ["alpha", "beta", "gamma", "delta"]

BITMASK_PATH = Path("experiments/results/E012_bitmask.parquet")
E012_PATH = Path("experiments/results/E012_retrieval_miss_analysis.json")
E011_PATH = Path("experiments/results/E011_full_recall.json")


def _bitmask(bits: list[int], row_ids: list[str] | None = None) -> pl.DataFrame:
    """A bitmask frame from a list of bit values, one row per true pair."""
    n = len(bits)
    return pl.DataFrame(
        {
            "row_id": row_ids if row_ids is not None else [f"S1-{i}" for i in range(n)],
            "vendor_id": [f"V-{i}" for i in range(n)],
            "bit": pl.Series(bits, dtype=pl.Int64),
        }
    )


# --------------------------------------------------------------------------
# The bug this module was written to prevent
# --------------------------------------------------------------------------


def test_standalone_is_not_cumulative():
    """Standalone and cumulative must be different numbers, and both correct.

    Four pairs over four strategies, each retrieved by exactly one strategy:
    bits 1, 2, 4, 8. Then strategy ``gamma`` retrieves nothing on its own, and
    ``delta`` retrieves everything -- so the two columns have to diverge.
    """
    found = _bitmask([0b0001, 0b0010, 0b0100, 0b1000])
    table = strategy_recall_table(found, PLAN, n_pairs=4, n_entities=4)
    by_name = {row["strategy"]: row for row in table}

    # Each strategy found its own single pair, alone.
    for name in PLAN:
        assert by_name[name]["standalone_pair_recall"] == 0.25
        assert by_name[name]["standalone_hits"] == 1
        assert by_name[name]["standalone_entity_recall"] == 0.25

    # Cumulative climbs 0.25 -> 0.50 -> 0.75 -> 1.00, and each marginal is 0.25.
    assert [r["cumulative_pair_recall"] for r in table] == [0.25, 0.5, 0.75, 1.0]
    assert [r["marginal_pair_recall"] for r in table] == [0.25, 0.25, 0.25, 0.25]

    # A strategy that retrieves nothing standalone but overlaps earlier ones
    # would report cumulative > standalone. That asymmetry is the whole point.
    mixed = _bitmask([0b0011, 0b0011, 0b0011, 0b0000])
    rows = {r["strategy"]: r for r in strategy_recall_table(mixed, PLAN, n_pairs=4, n_entities=4)}
    assert rows["alpha"]["standalone_pair_recall"] == 0.75
    assert rows["alpha"]["cumulative_pair_recall"] == 0.75
    assert rows["beta"]["standalone_pair_recall"] == 0.75
    assert rows["beta"]["cumulative_pair_recall"] == 0.75
    assert rows["gamma"]["standalone_pair_recall"] == 0.0
    assert rows["gamma"]["cumulative_pair_recall"] == 0.75
    assert rows["gamma"]["marginal_pair_recall"] == 0.0


def test_standalone_never_exceeds_cumulative():
    """Structural invariant: the plan only ever adds strategies."""
    found = _bitmask([0b0001, 0b0110, 0b1100, 0b1001, 0b0000, 0b1111, 0b1010])
    for row in strategy_recall_table(found, PLAN, n_pairs=7, n_entities=6):
        assert row["standalone_pair_recall"] <= row["cumulative_pair_recall"]
        assert row["standalone_entity_recall"] <= row["cumulative_entity_recall"]


# --------------------------------------------------------------------------
# "Exactly one" must mean exactly one
# --------------------------------------------------------------------------


def test_retrieved_by_exactly_one_counts_single_bit_masks_only():
    """Regression for E012: `bit < all_bits` is not "exactly one".

    Six pairs: one missed, three found by a single strategy, one found by three
    strategies, one found by all four. The old predicate
    ``bit > 0 AND bit < 0b1111`` returns 5 here; the answer is 3.
    """
    found = _bitmask([0b0000, 0b0001, 0b0010, 0b0100, 0b0111, 0b1111])
    histogram = coverage_histogram(found, len(PLAN))

    # Empty buckets are present with an explicit zero, so "no pair used two
    # strategies" is distinguishable from "the key is missing".
    assert histogram == {"0": 1, "1": 3, "2": 0, "3": 1, "4": 1}
    assert sum(histogram.values()) == 6
    assert histogram["1"] == 3

    report = recall_report(found, PLAN, n_pairs=6, n_entities=6)
    assert report["retrieved_by_exactly_one"] == 3
    assert report["retrieved_by_all_strategies"] == 1
    assert report["missed_pairs"] == 1
    # The two must agree, or the artifact contradicts itself as E012's did.
    assert report["retrieved_by_exactly_one"] == report["coverage_histogram"]["1"]


def test_coverage_histogram_sums_to_total_and_key_zero_is_the_miss_count():
    found = _bitmask([0, 1, 3, 7, 15, 0, 2, 5])
    histogram = coverage_histogram(found, len(PLAN))
    assert sum(histogram.values()) == found.height
    assert histogram["0"] == 2
    pair_recall, _, hits = union_recall(found, n_pairs=8, n_entities=8)
    assert hits == 6
    assert pair_recall == pytest.approx(0.75)
    assert histogram["0"] == found.height - hits
    # Every key from 0 to n_strategies is present, zero-valued if necessary.
    assert sorted(histogram) == ["0", "1", "2", "3", "4"]


def test_popcount():
    assert [popcount(b) for b in (0, 1, 2, 3, 127)] == [0, 1, 1, 2, 7]


# --------------------------------------------------------------------------
# Entity recall is a different quantity from pair recall
# --------------------------------------------------------------------------


def test_entity_recall_is_a_hit_rate_not_a_per_entity_recall():
    """Pins the definition of `union.entity_recall` precisely.

    An entity with one of its four true pairs retrieved still counts as a hit,
    because the question is "did retrieval surface this business at all". The
    number is therefore always >= a per-entity recall would be, and the artifact
    has to say so, or a reader will assume the stricter definition.
    """
    found = _bitmask(
        [0b0001, 0b0000, 0b0000, 0b0000, 0b0001],
        row_ids=["S1-a", "S1-a", "S1-a", "S1-a", "S1-b"],
    )
    report = recall_report(found, PLAN, n_pairs=5, n_entities=2)
    # 2 of 5 pairs retrieved, but both entities appear at least once.
    assert report["union"]["pair_recall"] == pytest.approx(0.4)
    assert report["union"]["entity_recall"] == pytest.approx(1.0)
    assert "AT LEAST ONE" in report["union"]["entity_recall_definition"]
    assert "not a per-entity recall" in report["union"]["entity_recall_definition"]

    # The partial coverage is still visible where it belongs: in the coverage
    # histogram, which is how you tell a broad union from a lucky-hit union.
    assert report["coverage_histogram"] == {"0": 3, "1": 2, "2": 0, "3": 0, "4": 0}


def test_a_missed_entity_does_not_count_towards_entity_recall():
    found = _bitmask(
        [0b0001, 0b0001, 0b0000, 0b0000],
        row_ids=["S1-a", "S1-a", "S1-b", "S1-b"],
    )
    report = recall_report(found, PLAN, n_pairs=4, n_entities=2)
    assert report["union"]["pair_recall"] == pytest.approx(0.5)
    assert report["union"]["entity_recall"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# A bitmask that cannot support a recall is refused, not divided by
# --------------------------------------------------------------------------


def test_validate_bitmask_rejects_wrong_row_count():
    found = _bitmask([1, 2, 3])
    with pytest.raises(ValueError, match="one row per true pair"):
        validate_bitmask(found, 4)


def test_validate_bitmask_rejects_missing_column():
    found = pl.DataFrame({"row_id": ["a"], "bit": pl.Series([1], dtype=pl.Int64)})
    with pytest.raises(ValueError, match="missing required column"):
        validate_bitmask(found)


def test_strategy_recall_table_propagates_the_row_count_check():
    with pytest.raises(ValueError, match="one row per true pair"):
        strategy_recall_table(_bitmask([1, 2]), PLAN, n_pairs=3, n_entities=2)


# --------------------------------------------------------------------------
# Reachability arithmetic
# --------------------------------------------------------------------------


def test_reachability_follows_from_pair_recall():
    """F_0.5 = 1.25R / (0.25 + R); 0.95 needs R >= 0.7917.

    This is the number that decides whether retrieval or the model is the
    binding constraint, so it is pinned rather than left to a print statement.
    The bitmasks are built to hit each recall exactly.
    """
    for hits, total in ((0, 10), (5, 10), (7, 10), (8, 10), (10, 10)):
        found = _bitmask([0b0001] * hits + [0] * (total - hits))
        recall = hits / total
        report = recall_report(found, PLAN, n_pairs=total, n_entities=total)
        assert report["union"]["pair_recall"] == pytest.approx(recall)
        assert report["reachability"]["measured_pair_recall"] == pytest.approx(
            report["union"]["pair_recall"]
        )
        assert report["reachability"]["max_f05_at_perfect_precision"] == pytest.approx(
            1.25 * recall / (0.25 + recall)
        )
        assert report["reachability"]["f05_95_requires_recall"] == RECALL_REQUIRED_FOR_F05_95

    # 0.7917 is the documented crossing point, checked directly.
    r = RECALL_REQUIRED_FOR_F05_95
    assert 1.25 * r / (0.25 + r) == pytest.approx(0.95, abs=1e-4)
    # And the measured E011 recall is below it, which is why retrieval is the
    # binding constraint right now.
    assert 0.7766 < RECALL_REQUIRED_FOR_F05_95


# --------------------------------------------------------------------------
# Regression anchors against frozen real artifacts
# --------------------------------------------------------------------------


@pytest.mark.skipif(not BITMASK_PATH.exists(), reason="E012 bitmask artifact not present")
def test_frozen_e012_bitmask_reproduces_stored_standalone_recalls():
    """The corrected code must agree with E012's already-correct numbers.

    E012 computed standalone recall by bitmask intersection and got it right, so
    its artifact is an independent check on this module. If a change here moves
    any of these seven values, the change is wrong, not the artifact.
    """
    found = pl.read_parquet(BITMASK_PATH)
    stored = json.loads(E012_PATH.read_text())
    by_name = {row["strategy"]: row for row in stored["per_strategy"]}

    n_pairs = found.height
    n_entities = found["row_id"].n_unique()
    assert (n_pairs, n_entities) == (1740, 469)

    table = strategy_recall_table(
        found, [row["strategy"] for row in stored["per_strategy"]],
        n_pairs=n_pairs, n_entities=n_entities,
    )
    for row in table:
        expected = by_name[row["strategy"]]["standalone_pair_recall"]
        assert row["standalone_pair_recall"] == pytest.approx(expected, abs=1e-6)
        assert row["standalone_hits"] == by_name[row["strategy"]]["standalone_hits"]

    report = recall_report(
        found, [row["strategy"] for row in stored["per_strategy"]],
        n_pairs=n_pairs, n_entities=n_entities,
    )
    assert report["union"]["pair_recall"] == pytest.approx(
        stored["recall"]["union_pair_recall"], abs=1e-6
    )
    assert report["union"]["entity_recall"] == pytest.approx(
        stored["recall"]["union_entity_recall"], abs=1e-6
    )
    assert report["missed_pairs"] == stored["recall"]["missed_pairs"] == 420
    # E012's coverage histogram was always right; the summary field was not.
    assert report["coverage_histogram"] == stored["strategy_coverage"]["histogram"]


@pytest.mark.skipif(not E011_PATH.exists(), reason="E011 artifact not present")
def test_corrected_e011_standalone_matches_the_audit_values():
    """E011's corrected standalone recalls are the audited ground truth.

    These seven numbers were re-derived by hand from the ``hits`` column of the
    original artifact and independently confirmed. They are the values every
    strategy-quality discussion must now use.
    """
    stored = json.loads(E011_PATH.read_text())
    expected = {
        "exact_name": 0.4874,
        "alnum_name": 0.4921,
        "sorted_name": 0.5220,
        "address_exact": 0.1170,
        "name_token": 0.1121,
        "address_token": 0.4061,
        "numeric_token": 0.0855,
    }
    n_pairs = stored["counts"]["true_pairs"]
    assert n_pairs == 692_176

    for row in stored["per_strategy"]:
        name = row["strategy"]
        assert name in expected
        # Standalone must be recoverable from the hit count alone.
        assert row["standalone_hits"] / n_pairs == pytest.approx(
            row["standalone_pair_recall"], abs=1e-6
        )
        assert round(row["standalone_pair_recall"], 4) == expected[name]

    # And the decomposition must be internally consistent: the marginals
    # telescope to the union. E011 stored these columns rounded to four decimal
    # places, so the telescoping sum can drift by a few units in the last place
    # and the tolerance has to allow for that rounding.
    running = 0.0
    for row in stored["per_strategy"]:
        running += row["marginal_pair_recall"]
        assert row["cumulative_pair_recall"] == pytest.approx(running, abs=1e-3)
    assert round(running, 4) == pytest.approx(stored["union"]["pair_recall"], abs=1e-4)
    assert round(stored["union"]["pair_recall"], 4) == 0.7766
    assert round(1 - stored["missed_pairs"] / n_pairs, 4) == 0.7766


@pytest.mark.skipif(not E011_PATH.exists(), reason="E011 artifact not present")
def test_e011_standalone_is_far_below_cumulative_for_the_weak_strategies():
    """Pins the exact magnitude of the bug that was reported.

    ``address_exact`` and ``name_token`` were the two that read as far stronger
    than they are. If anyone re-derives these from the wrong column, this fails.
    """
    stored = json.loads(E011_PATH.read_text())
    by_name = {row["strategy"]: row for row in stored["per_strategy"]}
    assert by_name["address_exact"]["standalone_pair_recall"] == 0.117011
    assert by_name["address_exact"]["cumulative_pair_recall"] == 0.5903
    assert by_name["name_token"]["standalone_pair_recall"] == 0.112117
    assert by_name["name_token"]["cumulative_pair_recall"] == 0.6242
    # The entity column was cumulative too.
    for row in stored["per_strategy"]:
        assert row["standalone_entity_recall"] is None, (
            "standalone entity recall was not recoverable from E011; if a "
            "bitmask ever made it available it should be filled in, not guessed"
        )


@pytest.mark.skipif(not E011_PATH.exists(), reason="E011 artifact not present")
def test_e011_counts_are_integers_and_satisfy_the_coverage_identity():
    """A count reconstructed from a rounded recall is not a count.

    ``retrieved_by_all_strategies`` was briefly published as 651.88, derived as
    ``true_pairs * union_recall - fewer_than_all``. The recall is stored rounded
    to four places, so the product carried ~7.9 pairs of rounding error and the
    result was fractional -- which is the only reason anyone noticed.

    The exact route is integer-only, and it is the only one allowed: the count
    of retrieved pairs is ``true_pairs - missed_pairs``, with no division.
    """
    stored = json.loads(E011_PATH.read_text())
    n_pairs = stored["counts"]["true_pairs"]
    missed = stored["missed_pairs"]
    fewer = stored["retrieved_by_fewer_than_all_strategies"]
    all_strategies = stored["retrieved_by_all_strategies"]

    assert isinstance(all_strategies, int), "a pair count must be an integer"
    assert all_strategies == (n_pairs - missed) - fewer == 644

    # The stored recall is the exact one correctly rounded to four places, but it
    # is rounded *up* here (0.776589 -> 0.7766), so multiplying it by 692,176
    # inflates the count by 7.88. That is the whole mechanism: a correct-looking
    # rounded rate is still not the integer behind it.
    exact = (n_pairs - missed) / n_pairs
    assert exact == pytest.approx(stored["union"]["pair_recall"], abs=5e-5)
    assert exact < stored["union"]["pair_recall"]
    assert n_pairs * stored["union"]["pair_recall"] - fewer != exact * n_pairs - fewer
    assert (n_pairs * stored["union"]["pair_recall"] - fewer) % 1 != 0.0


@pytest.mark.skipif(not E011_PATH.exists(), reason="E011 artifact not present")
def test_e011_correction_block_names_the_unrecoverable_fields():
    """The artifact must state what it could not fix, not just what it did."""
    stored = json.loads(E011_PATH.read_text())
    corrections = stored["_corrections"]
    assert corrections["buggy_commit"] == "1335ab7"
    unrecoverable = corrections["unrecoverable_fields"]
    assert "standalone_entity_recall" in unrecoverable
    assert "NOT RECOVERABLE" in unrecoverable["standalone_entity_recall"]
    # Every corrected count is an integer, and the old meanings are documented.
    for row in stored["per_strategy"]:
        assert isinstance(row["standalone_hits"], int)
    assert "MISNAMED" in corrections["old_field_meaning"]["pair_recall"]
    assert "CORRECT" in corrections["old_field_meaning"]["hits"]
