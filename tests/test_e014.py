"""E014 infrastructure: ceiling overrides, candidate volume, country recall.

Run:

    uv run pytest tests/test_e014.py

E014 asks whether raising the document-frequency ceilings recovers enough true
pairs to justify the extra candidate-generation cost. The measurement needs
three things the baseline driver did not have:

1. A way to run a ceiling variant that cannot be confused with the baseline
   (``--ceiling NAME=INT`` + an effective plan threaded everywhere).
2. A cost denominator measured without materialising candidate sets
   (:func:`candidate_volume_for_strategy`).
3. A per-country breakdown of the same bitmask (:func:`recall_by_country`).

Every test here is synthetic and runs in milliseconds. The tests pin the exact
properties E014's validity rests on: variants say what they changed, the
baseline is the default invocation, volume arithmetic is exact, and a missing
country mapping fails loudly instead of publishing a partial breakdown.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

SCRIPT_PATH = Path("experiments/scripts/measure_union_recall.py")
E014_CONFIG = Path("configs/e014.yaml")


def _load_driver():
    spec = importlib.util.spec_from_file_location("measure_union_recall", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mur = _load_driver()

NAMES = [name for name, _, _ in mur.PLAN]
BASELINE_CEILINGS = {name: md for name, _, md in mur.PLAN if md is not None}


# --------------------------------------------------------------------------
# --ceiling parsing
# --------------------------------------------------------------------------


def test_no_overrides_parses_to_empty():
    assert mur.parse_ceiling_overrides([]) == {}
    assert mur.parse_ceiling_overrides(None) == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["name_token=200"], {"name_token": 200}),
        (
            ["name_token=200", "address_token=1000", "numeric_token=1000"],
            {"name_token": 200, "address_token": 1000, "numeric_token": 1000},
        ),
        (["  address_token = 500 "], {"address_token": 500}),
    ],
)
def test_valid_overrides_parse(raw, expected):
    assert mur.parse_ceiling_overrides(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        ["exact_name=5"],  # exact strategies carry no ceiling
        ["name_token"],  # missing =INT
        ["name_token=abc"],  # not an integer
        ["name_token=0"],  # non-positive
        ["name_token=-10"],  # non-positive
        ["name_tokne=200"],  # typo must not silently run the baseline
    ],
)
def test_invalid_overrides_raise(raw):
    with pytest.raises(ValueError):
        mur.parse_ceiling_overrides(raw)


# --------------------------------------------------------------------------
# Effective plan: the baseline is the default, not a special case
# --------------------------------------------------------------------------


def test_no_overrides_gives_the_baseline_plan_exactly():
    plan = mur.effective_plan({})
    assert [(n, m) for n, _, m in plan] == [(n, m) for n, _, m in mur.PLAN]
    assert [b for _, b, _ in plan] == [b for _, b, _ in mur.PLAN]


def test_module_plan_is_never_mutated():
    before = [(n, m) for n, _, m in mur.PLAN]
    mur.effective_plan({"name_token": 200, "address_token": 1000})
    assert [(n, m) for n, _, m in mur.PLAN] == before


def test_override_changes_only_the_named_strategy():
    plan = mur.effective_plan({"address_token": 1000})
    by_name = {n: m for n, _, m in plan}
    assert by_name == {"exact_name": None, "alnum_name": None, "sorted_name": None,
                       "address_exact": None, "name_token": 50,
                       "address_token": 1000, "numeric_token": 200}


def test_baseline_ceilings_are_the_e011_values():
    assert BASELINE_CEILINGS == {"name_token": 50, "address_token": 200,
                                 "numeric_token": 200}


# --------------------------------------------------------------------------
# Candidate volume: exact block arithmetic on synthetic frames
# --------------------------------------------------------------------------


def _volume_frames():
    """Three vendors, two queries, hand-countable token blocks.

    Vendor tokens: alpha x2, beta x1, gamma x1, delta x1.
    Query tokens:  alpha x1, beta x1, zeta x1.
    """
    from team_diamond.retrieval.keys import build_name_token_keys

    vendors = pl.DataFrame(
        {"entity_id": ["v1", "v2", "v3"],
         "name_norm": ["alpha beta", "alpha gamma", "delta"]}
    ).lazy()
    queries = pl.DataFrame(
        {"entity_id": ["q1", "q2"], "row_id": ["q1", "q2"],
         "name_norm": ["alpha beta", "zeta"]}
    )
    return build_name_token_keys, vendors, queries


def test_volume_with_ceiling_counts_only_kept_blocks():
    builder, vendors, queries = _volume_frames()
    vol = mur.candidate_volume_for_strategy(
        "name_token", builder, 1, queries, vendors
    )
    # Kept (df<=1): beta, gamma, delta. Dropped: alpha.
    assert vol["kept_keys"] == 3
    assert vol["dropped_keys"] == 1
    # Only beta is shared with a query: 1 query row x df 1 = 1 pair.
    assert vol["candidate_pairs"] == 1
    assert vol["max_df"] == 1


def test_volume_without_ceiling_counts_everything():
    builder, vendors, queries = _volume_frames()
    vol = mur.candidate_volume_for_strategy(
        "name_token", builder, None, queries, vendors
    )
    assert vol["kept_keys"] == 4
    assert vol["dropped_keys"] == 0
    # alpha: 1x2, beta: 1x1. gamma/delta/zeta touch no query.
    assert vol["candidate_pairs"] == 3


def test_raising_the_ceiling_monotonically_grows_volume():
    builder, vendors, queries = _volume_frames()
    volumes = [
        mur.candidate_volume_for_strategy("name_token", builder, c, queries, vendors)[
            "candidate_pairs"
        ]
        for c in (1, 2, None)
    ]
    assert volumes == sorted(volumes), "a higher ceiling cannot remove pairs"
    assert volumes[0] == 1 and volumes[-1] == 3


def test_volume_on_empty_queries_is_zero_not_null():
    builder, vendors, _ = _volume_frames()
    empty = pl.DataFrame(
        {"entity_id": pl.Series([], dtype=pl.Utf8),
         "row_id": pl.Series([], dtype=pl.Utf8),
         "name_norm": pl.Series([], dtype=pl.Utf8)}
    )
    vol = mur.candidate_volume_for_strategy(
        "name_token", builder, 50, empty, vendors
    )
    assert vol["candidate_pairs"] == 0


# --------------------------------------------------------------------------
# Country recall from a bitmask
# --------------------------------------------------------------------------


def test_recall_by_country_reports_zero_not_null():
    found = pl.DataFrame(
        {"row_id": ["a", "a", "b"],
         "vendor_id": ["v1", "v2", "v3"],
         "bit": pl.Series([1, 0, 0], dtype=pl.Int64)}
    )
    mapping = pl.DataFrame({"row_id": ["a", "b"], "country": ["IN", "US"]})
    out = mur.recall_by_country(found, mapping)
    assert out["IN"] == {"true_pairs": 2, "retrieved_pairs": 1, "pair_recall": 0.5}
    # No retrieved pairs is a measurement (0.0), not missing data.
    assert out["US"] == {"true_pairs": 1, "retrieved_pairs": 0, "pair_recall": 0.0}


def test_recall_by_country_refuses_a_partial_mapping():
    found = pl.DataFrame(
        {"row_id": ["a", "b"], "vendor_id": ["v1", "v2"],
         "bit": pl.Series([1, 1], dtype=pl.Int64)}
    )
    mapping = pl.DataFrame({"row_id": ["a"], "country": ["IN"]})
    with pytest.raises(ValueError, match="no country mapping"):
        mur.recall_by_country(found, mapping)


# --------------------------------------------------------------------------
# build_report records the plan that ran, not the module default
# --------------------------------------------------------------------------


def _bitmask():
    return pl.DataFrame(
        {"row_id": ["x", "x", "y"], "vendor_id": ["v1", "v2", "v3"],
         "bit": pl.Series([1, 2, 0], dtype=pl.Int64)}
    )


def test_build_report_records_variant_ceilings():
    plan = mur.effective_plan({"name_token": 200})
    report = mur.build_report(
        _bitmask(), n_pairs=3, n_entities=2, plan=plan,
        run_config={"seed": 17}, counts={},
    )
    by_name = {r["strategy"]: r for r in report["per_strategy"]}
    assert by_name["name_token"]["max_df"] == 200
    assert by_name["address_token"]["max_df"] == 200
    assert by_name["exact_name"]["max_df"] is None


def test_build_report_country_null_means_bitmask_path():
    report = mur.build_report(
        _bitmask(), n_pairs=3, n_entities=2, run_config={}, counts={},
    )
    assert report["recall_by_country"] is None


def test_build_report_carries_a_supplied_country_breakdown():
    breakdown = {"IN": {"true_pairs": 2, "retrieved_pairs": 1, "pair_recall": 0.5}}
    report = mur.build_report(
        _bitmask(), n_pairs=3, n_entities=2, run_config={}, counts={},
        country_breakdown=breakdown,
    )
    assert report["recall_by_country"] == breakdown


# --------------------------------------------------------------------------
# E014 config: the matrix isolates one ceiling at a time
# --------------------------------------------------------------------------


def test_e014_config_declares_five_variants_with_a_fixed_sample():
    cfg = yaml.safe_load(E014_CONFIG.read_text())["e014"]
    assert cfg["seed"] == 17
    assert cfg["screen_query_rows"] == 20000
    variants = cfg["variants"]
    assert set(variants) == {"V0_baseline", "V1_name_token_200",
                             "V2_address_token_1000", "V3_numeric_token_1000",
                             "V4_combined_200_1000_1000"}
    assert variants["V0_baseline"] == []
    # V1-V3 each move exactly one ceiling.
    for variant in ("V1_name_token_200", "V2_address_token_1000",
                    "V3_numeric_token_1000"):
        assert len(variants[variant]) == 1
        parsed = mur.parse_ceiling_overrides(variants[variant])
        assert len(parsed) == 1
    # The combined variant is the only multi-ceiling run.
    assert len(mur.parse_ceiling_overrides(variants["V4_combined_200_1000_1000"])) == 3
    # Every flag in the matrix parses against the real parser.
    for flags in variants.values():
        mur.parse_ceiling_overrides(flags)
    # Baseline ceilings in the config match the code baseline.
    assert cfg["baseline_ceilings"] == {"name_token": 50, "address_token": 200,
                                        "numeric_token": 200}
    assert cfg["baseline_ceilings"] == BASELINE_CEILINGS


# --------------------------------------------------------------------------
# Cap-report helpers: the scale probe must record distribution, not just totals
# --------------------------------------------------------------------------


def _cap_candidates():
    """Synthetic candidate frame: q1x3, q2x1, q3x0 of 3 queries."""
    return pl.DataFrame(
        {"s1_id": ["q1", "q1", "q1", "q2"], "vendor_id": ["v1", "v2", "v3", "v1"]}
    )


class _FakeReport:
    n_dropped_by_cap = 2
    n_pairs_before_cap = 6
    per_strategy_pairs = {"name_token": 4, "address_token": 2}


def test_parse_caps_absent_means_full_grid():
    assert mur.parse_caps(None) == list(mur.CAP_SWEEP)


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "200,x", "", " "])
def test_parse_caps_rejects_non_positive_integers(bad):
    with pytest.raises(ValueError):
        mur.parse_caps(bad)


def test_parse_caps_selects_subset():
    assert mur.parse_caps("200,400") == [200, 400]
    assert mur.parse_caps(" 400 ") == [400]


def test_summarize_cap_candidates_counts_zeros_and_bytes():
    out = mur.summarize_cap_candidates(_cap_candidates(), _FakeReport(), 3)
    assert out["candidates"] == 4
    assert out["queries"] == 3
    assert out["queries_with_candidates"] == 2
    assert out["queries_zero_candidates"] == 1
    assert out["candidates_per_query_mean_all"] == pytest.approx(4 / 3, abs=1e-3)
    assert out["candidates_per_query_median_nonzero"] == 2
    assert out["candidates_per_query_max"] == 3
    assert out["candidates_per_query_p99_nonzero"] >= out["candidates_per_query_p95_nonzero"]
    assert out["dropped_by_cap"] == 2
    assert out["pairs_before_cap"] == 6
    assert out["candidate_bytes"] > 0
    assert out["per_strategy_pairs"] == {"name_token": 4, "address_token": 2}


def test_summarize_cap_candidates_json_serialisable():
    import json

    out = mur.summarize_cap_candidates(_cap_candidates(), _FakeReport(), 3)
    json.dumps(out)
