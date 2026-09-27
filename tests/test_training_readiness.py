"""Training-pipeline readiness: split, labels, negatives, matrix, frozen plan.

Run: uv run pytest tests/test_training_readiness.py -q

Every test here is synthetic and runs in milliseconds. They pin the exact
properties E017's verdict rests on: the split never shares entities, labels
are correct for singletons/duplicates/multi-matches, negatives come from the
given (train-fold) frame with per-entity entitlement, the matrix is strict
about order, and the frozen V4 plan cannot drift from the measured ceilings.
"""

from __future__ import annotations

import polars as pl
import pytest

from team_diamond.models.matrix import to_model_matrix
from team_diamond.retrieval.candidates import DEFAULT_PLAN, FROZEN_V4_PLAN
from team_diamond.training.labels import label_candidates, positive_pairs
from team_diamond.training.negatives import sample_hard_negatives
from team_diamond.training.split import assert_partial_matching, entity_aware_split


def _gt() -> pl.DataFrame:
    """Two multi-match entities, one single-match, one singleton, one dupe row."""
    return pl.DataFrame(
        {
            "source1_entity_id": ["a", "a", "b", "c", "d", "d"],
            "matched_entity_ids": [
                "v1,v2", "v1,v2",  # duplicate GT row for a (must dedup)
                "v3",
                "",  # singleton c
                "v4,v5,v6",  # multi-match d
                "v4,v5,v6",  # hmm, duplicate again? no — keep single
            ][:6],
        }
    )


def _s1() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": ["a", "b", "c", "d", "e"],
            "country": ["IN", "US", "IN", "US", "IN"],
        }
    )


# --------------------------------------------------------------------------
# Frozen plan
# --------------------------------------------------------------------------


def test_frozen_v4_plan_carries_exactly_the_measured_ceilings():
    got = {p.name: p.max_df for p in FROZEN_V4_PLAN}
    assert got == {
        "exact_name": None, "alnum_name": None, "sorted_name": None,
        "address_exact": None, "name_token": 200, "address_token": 1000,
        "numeric_token": 1000,
    }


def test_frozen_plan_matches_baseline_strategies_and_builders():
    assert [p.name for p in FROZEN_V4_PLAN] == [p.name for p in DEFAULT_PLAN]
    assert [p.builder for p in FROZEN_V4_PLAN] == [p.builder for p in DEFAULT_PLAN]
    # ... differing ONLY in the three raised ceilings.
    diffs = [
        p.name for p, q in zip(FROZEN_V4_PLAN, DEFAULT_PLAN) if p.max_df != q.max_df
    ]
    assert diffs == ["name_token", "address_token", "numeric_token"]


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_positive_pairs_dedup_and_singleton_absence():
    gt = pl.DataFrame(
        {
            "source1_entity_id": ["a", "a", "b", "c"],
            "matched_entity_ids": ["v1,v2", "v1,v2", "v3", ""],
        }
    )
    out = positive_pairs(gt).sort("s1_id", "vendor_id")
    assert out.to_dicts() == [
        {"s1_id": "a", "vendor_id": "v1"},
        {"s1_id": "a", "vendor_id": "v2"},
        {"s1_id": "b", "vendor_id": "v3"},
    ]


def test_label_candidates_marks_matches_and_nonmatches():
    cands = pl.DataFrame(
        {"s1_id": ["a", "a", "b"], "vendor_id": ["v1", "v9", "v3"]}
    )
    gt = pl.DataFrame(
        {
            "source1_entity_id": ["a", "b"],
            "matched_entity_ids": ["v1,v2", "v3"],
        }
    )
    out = label_candidates(cands, gt, fold="train").sort("s1_id", "vendor_id")
    assert out["is_match"].to_list() == [True, False, True]
    assert out["is_match"].dtype == pl.Boolean
    assert set(out["fold"].to_list()) == {"train"}


def test_singleton_entities_get_no_positive_rows_but_stay_addressable():
    # Singletons contribute no positive pairs; the split (not the labeller)
    # is what keeps them. The labeller must simply not invent rows for them.
    gt = pl.DataFrame(
        {"source1_entity_id": ["c"], "matched_entity_ids": [""]}
    )
    assert positive_pairs(gt).height == 0
    cands = pl.DataFrame({"s1_id": ["c"], "vendor_id": ["v9"]})
    out = label_candidates(cands, gt, fold="validation")
    assert out["is_match"].to_list() == [False]


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def test_split_has_no_entity_overlap_and_covers_everyone():
    gt = pl.DataFrame(
        {
            "source1_entity_id": [f"e{i}" for i in range(40)],
            "matched_entity_ids": ["v" + str(i) if i % 3 else "" for i in range(40)],
        }
    )
    s1 = pl.DataFrame(
        {
            "entity_id": [f"e{i}" for i in range(40)],
            "country": ["IN" if i % 2 else "US" for i in range(40)],
        }
    )
    assert_partial_matching(gt)
    split = entity_aware_split(s1, gt, validation_fraction=0.2, seed=17)
    assert not (split.train.entity_ids & split.validation.entity_ids)
    assert split.train.entity_ids | split.validation.entity_ids == set(
        s1["entity_id"].to_list()
    )
    # Deterministic.
    again = entity_aware_split(s1, gt, validation_fraction=0.2, seed=17)
    assert again.train.entity_ids == split.train.entity_ids


def test_split_rejects_bad_fractions_and_duplicate_ids():
    gt = pl.DataFrame(
        {"source1_entity_id": ["e0"], "matched_entity_ids": ["v0"]}
    )
    s1 = pl.DataFrame({"entity_id": ["e0"], "country": ["IN"]})
    with pytest.raises(ValueError):
        entity_aware_split(s1, gt, validation_fraction=0.0, seed=17)
    dup = pl.DataFrame({"entity_id": ["e0", "e0"], "country": ["IN", "IN"]})
    with pytest.raises(ValueError, match="duplicate"):
        entity_aware_split(dup, gt, validation_fraction=0.2, seed=17)


def test_split_rejects_shared_vendor_ids():
    gt = pl.DataFrame(
        {
            "source1_entity_id": ["e0", "e1"],
            "matched_entity_ids": ["v0", "v0"],  # v0 claimed twice: not allowed
        }
    )
    s1 = pl.DataFrame({"entity_id": ["e0", "e1"], "country": ["IN", "US"]})
    with pytest.raises(ValueError):
        entity_aware_split(s1, gt, validation_fraction=0.5, seed=17)


# --------------------------------------------------------------------------
# Negatives
# --------------------------------------------------------------------------


def _labelled() -> pl.DataFrame:
    # a: 2 positives, 5 negatives. b: 1 positive, 1 negative.
    return pl.DataFrame(
        {
            "s1_id": ["a"] * 7 + ["b"] * 2,
            "vendor_id": [f"v{i}" for i in range(9)],
            "is_match": [True, True, False, False, False, False, False, True, False],
            "n_keys": [9, 8, 7, 6, 5, 4, 3, 9, 2],
        }
    )


def test_negatives_per_entity_entitlement_and_determinism():
    first = sample_hard_negatives(
        _labelled(), negatives_per_positive=2, seed=17, score_column="n_keys"
    )
    second = sample_hard_negatives(
        _labelled(), negatives_per_positive=2, seed=17, score_column="n_keys"
    )
    assert first.frame.sort("s1_id", "vendor_id").equals(
        second.frame.sort("s1_id", "vendor_id")
    )
    # a entitled to 2*2=4 (hardest first: n_keys 7,6,5,4), b to 2*1=1.
    assert first.n_positive == 3
    assert first.n_negative == 5
    kept = first.frame.filter(~pl.col("is_match")).sort("s1_id", "n_keys")
    assert kept.filter(pl.col("s1_id") == "a")["n_keys"].to_list() == [4, 5, 6, 7]


def test_negatives_raise_when_an_entity_has_positives_but_no_negatives():
    only_pos = pl.DataFrame(
        {"s1_id": ["a", "a"], "vendor_id": ["v1", "v2"], "is_match": [True, True]}
    )
    with pytest.raises(ValueError, match="zero negatives"):
        sample_hard_negatives(only_pos, negatives_per_positive=3, seed=17)


def test_zero_negatives_setting_keeps_positives_only():
    out = sample_hard_negatives(_labelled(), negatives_per_positive=0, seed=17)
    assert out.n_negative == 0
    assert out.n_positive == 3


# --------------------------------------------------------------------------
# Matrix
# --------------------------------------------------------------------------


def test_matrix_strict_order_and_missing_raise():
    frame = pl.DataFrame({"f1": [1.0, 2.0], "f2": ["x", "y"]})
    with pytest.raises(ValueError, match="missing declared"):
        to_model_matrix(frame, feature_names=["f1", "f2", "f3"])
    m = to_model_matrix(
        frame, feature_names=["f2", "f1"], categorical_columns=["f2"]
    )
    assert m.feature_names == ("f2", "f1")
    assert m.cat_indices == (0,)
    assert m.X.shape == (2, 2)
    assert m.y is None


def test_matrix_rejects_null_categoricals_and_accepts_nan():
    import numpy as np

    frame = pl.DataFrame({"f1": [1.0, float("nan")], "f2": ["x", None]})
    with pytest.raises(ValueError, match="null"):
        to_model_matrix(frame, feature_names=["f1", "f2"],
                        categorical_columns=["f2"])
    m = to_model_matrix(frame.drop("f2"), feature_names=["f1"])
    assert bool(np.isnan(m.X[1, 0]))


# --------------------------------------------------------------------------
# model.yaml records the implementable configuration
# --------------------------------------------------------------------------


def test_model_yaml_matches_the_catboost_implementation():
    import yaml

    cfg = yaml.safe_load(
        __import__("pathlib").Path("configs/model.yaml").read_text()
    )["model"]
    assert cfg["type"] == "catboost"
    assert cfg["params"]["task_type"] == "CPU"
    assert cfg["params"]["random_seed"] == 20260926
    assert cfg["params"]["allow_writing_files"] is False
    # Tunables stay null placeholders: an unset value must fall back to code
    # defaults, never to an invented number.
    for key in ("iterations", "learning_rate", "depth", "thread_count",
                "early_stopping_rounds", "class_weights"):
        assert cfg["params"][key] is None, key


# --------------------------------------------------------------------------
# Preparation path resolves the frozen plan (E017 blocking defect)
# --------------------------------------------------------------------------


def test_preparation_path_resolves_frozen_v4():
    """The training-preparation call path must yield V4 ceilings.

    This exercises the REAL resolution function
    (``pipeline.run.generate_retrieval_plan``), not the constants: if someone
    points that function back at DEFAULT_PLAN, the ceilings come out
    50/200/200 and this fails. A test that only compared the two constants
    would pass straight through that regression.
    """
    from pathlib import Path

    from team_diamond.config import Config
    from team_diamond.pipeline.run import generate_retrieval_plan

    empty = Config(data={}, sources=())
    plan = generate_retrieval_plan(empty)
    assert [p.name for p in plan] == [p.name for p in FROZEN_V4_PLAN]
    assert {p.name: p.max_df for p in plan} == {
        "exact_name": None, "alnum_name": None, "sorted_name": None,
        "address_exact": None, "name_token": 200, "address_token": 1000,
        "numeric_token": 1000,
    }

    subset = Config(
        data={"retrieval": {"strategies": ["name_token", "address_token"]}},
        sources=(Path("test"),),
    )
    sub = generate_retrieval_plan(subset)
    assert {p.name: p.max_df for p in sub} == {
        "name_token": 200, "address_token": 1000,
    }

    with pytest.raises(ValueError, match="unknown strategy"):
        generate_retrieval_plan(
            Config(data={"retrieval": {"strategies": ["nope"]}}, sources=())
        )
