"""E012 miss analysis: metadata provenance, partition integrity, hard examples.

Run:

    uv run pytest tests/test_miss_analysis.py

Why these tests exist
---------------------
E012's published artifact was wrong in four ways, none of which raised an error:

1. It recorded ``query_rows = 200000`` for a run that used ``--query-rows 500``,
   because ``main`` filled the metadata block from module defaults instead of
   from ``args``. A 500-query diagnostic was published as a full-scale result.
2. ``retrieved_by_exactly_one`` was 1318 while the coverage histogram in the
   same block said 347 -- the artifact contradicted itself.
3. ``failure_modes`` was six overlapping boolean counts under a note claiming
   they were not mutually exclusive. On this sample they happened to be
   disjoint and sum to the total, so the note looked harmless and nobody
   checked. A reader summing them got 420 with no way to know whether that was
   guaranteed.
4. All 100 hard examples had a null vendor name and address, because the
   extractor read ``business_name_vendor`` while the frame carried
   ``business_name_vendor_raw``. The normalised fields were read correctly, so
   the examples *looked* populated.

Every number that can be checked is checked here. The tests are synthetic and
run in milliseconds; the real-artifact assertions at the bottom are skipped if
the artifacts are absent.

Import note
-----------
The experiment driver is a script, not a library module, so it is loaded by path.
That is deliberate: it keeps ``src/`` free of anything that only an experiment
needs, while still letting the analysis functions be tested directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

SCRIPT_PATH = Path("experiments/scripts/analyze_retrieval_misses.py")
ARTIFACT_PATH = Path("experiments/results/E012_retrieval_miss_analysis.json")
BITMASK_PATH = Path("experiments/results/E012_bitmask.parquet")


def _load_driver():
    """Import the E012 driver from its path."""
    spec = importlib.util.spec_from_file_location("analyze_retrieval_misses", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


arm = _load_driver()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _sources(
    rows: list[tuple[str, str, str, str, str, str, bool, bool]],
    *,
    prefix: str,
) -> pl.DataFrame:
    """Build a normalised or raw side-table from (id, name, addr, country, source, *_raw name).

    Args:
        rows: Tuples of ``(id, business_name, business_address, country,
            source, role)`` where role is ``"s1"`` or ``"vendor"``.
        prefix: ``"s1"`` or ``"vendor"``, used to build the column names the
            driver expects.

    Returns:
        A frame with both normalised and raw text columns, matching the shape
        ``analyze_misses`` receives.
    """
    is_s1 = prefix == "s1"
    return pl.DataFrame(
        {
            "entity_id": [r[0] for r in rows],
            "business_name": [r[1] for r in rows],
            "business_address": [r[2] for r in rows],
            "country": [r[3] for r in rows],
            "source": [r[4] for r in rows],
            "name_norm": [r[1].lower() for r in rows],
            "addr_norm": [r[2].lower() for r in rows],
            "has_name": [bool(r[1]) for r in rows],
            "has_address": [bool(r[2]) for r in rows],
        }
    )


def _sample_case():
    """A six-pair sample: two missed for different reasons, four retrieved.

    Missed pairs:
      - ``e1``/``v1``: vendor has no address at all -> address missing
      - ``e2``/``v2``: both sides fully populated, nothing identical -> the
        genuine "all present, nothing identical" mode

    Retrieved pairs cover the multi-match entity ``e3``, so entity and pair
    recall differ and the entity hit-rate definition is exercised.

    Returns:
        ``(found, true_pairs, s1_norm, s1_raw, vendors_norm, vendors_raw)``
    """
    s1_rows = [
        ("e1", "Alpha Traders", "12 High Street", "GB", "1"),
        ("e2", "Beta Foods", "3 Mill Road", "GB", "1"),
        ("e3", "Gamma Holdings", "9 Park Lane", "GB", "1"),
        ("e4", "Delta Works", "1 Dock Road", "GB", "1"),
    ]
    vendor_rows = [
        ("v1", "Alpha Traders", "", "GB", "2"),  # address missing
        ("v2", "Beta Fds", "3 Mil Road", "GB", "2"),  # both present, nothing identical
        ("v3", "Gamma Holdings Ltd", "9 Park Ln", "GB", "2"),
        ("v4", "Delta Works", "1 Dock Road", "GB", "3"),
        ("v5", "Gamma Holdings", "9 Park Lane", "GB", "3"),
    ]
    bits = [0b0000, 0b0000, 0b0001, 0b0110, 0b0001, 0b0010]
    row_ids = ["e1", "e2", "e3", "e3", "e4", "e4"]
    vendor_ids = ["v1", "v2", "v3", "v4", "v5", "v5"]
    found = pl.DataFrame(
        {
            "row_id": row_ids,
            "vendor_id": vendor_ids,
            "bit": pl.Series(bits, dtype=pl.Int64),
        }
    )
    true_pairs = found.select("row_id", "vendor_id")
    return (
        found,
        true_pairs,
        _sources(s1_rows, prefix="s1"),
        _sources(s1_rows, prefix="s1"),
        _sources(vendor_rows, prefix="vendor"),
        _sources(vendor_rows, prefix="vendor"),
    )


def _run_config(**overrides) -> dict:
    args = argparse.Namespace(
        experiment_id="E012",
        query_rows=7,
        seed=99,
        bitmask_input=None,
        bitmask_output=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return arm.build_run_config(args)


def _analyze(**overrides) -> dict:
    found, true_pairs, s1_norm, s1_raw, v_norm, v_raw = _sample_case()
    return arm.analyze_misses(
        found, true_pairs, v_norm, v_raw, s1_norm, s1_raw, _run_config(**overrides)
    )


# --------------------------------------------------------------------------
# Bug B: the report must record the arguments actually used
# --------------------------------------------------------------------------


def test_run_config_records_the_arguments_not_the_module_defaults():
    """Regression for the 500-queries-published-as-200,000 bug.

    The values are deliberately odd (7 and 99) so that if anything falls back
    to ``DEFAULT_QUERY_ROWS`` / ``DEFAULT_SEED`` the test fails loudly instead
    of accidentally agreeing.
    """
    config = _run_config()
    assert config["query_rows"] == 7
    assert config["seed"] == 99
    assert arm.DEFAULT_QUERY_ROWS == 200_000
    assert arm.DEFAULT_SEED == 17
    assert config["query_rows"] != arm.DEFAULT_QUERY_ROWS
    assert config["seed"] != arm.DEFAULT_SEED


def test_run_config_records_the_ceiling_in_force():
    """The df ceilings are the suspected cause of the misses, so they are provenance."""
    config = _run_config()
    assert config["max_df"] == {"name_token": 50, "address_token": 200, "numeric_token": 200}
    assert config["strategies"] == [name for name, _, _ in arm.PLAN]


def test_report_carries_the_config_and_the_sample_caveat():
    report = _analyze()
    assert report["config"]["query_rows"] == 7
    assert report["config"]["seed"] == 99
    scope = report["sample_scope"]
    assert scope["queries_sampled"] == 7
    assert scope["true_pairs_in_sample"] == 6
    assert scope["missed_pairs_in_sample"] == 2
    # The caveat must be present in the artifact, not only in the source, because
    # the artifact is what gets read.
    assert "NOT estimates" in scope["generalisation"]


def test_report_has_no_module_default_sentinels():
    """No value anywhere in the report may equal the module defaults by accident."""
    report = _analyze()
    flat = json.dumps(report)
    assert str(arm.DEFAULT_QUERY_ROWS) not in flat
    assert "DEFAULT_QUERY_ROWS" not in flat
    assert "DEFAULT_SEED" not in flat


# --------------------------------------------------------------------------
# Bug D: the coverage histogram must not contradict itself
# --------------------------------------------------------------------------


def test_coverage_fields_are_all_read_from_one_histogram():
    report = _analyze()
    coverage = report["strategy_coverage"]
    histogram = coverage["histogram"]
    assert coverage["retrieved_by_exactly_one"] == histogram["1"]
    assert coverage["missed_pairs_all_zero"] == histogram["0"] == report["recall"]["missed_pairs"]
    assert coverage["histogram_sums_to_true_pairs"]
    assert sum(histogram.values()) == report["recall"]["true_pairs"]
    assert coverage["retrieved_by_two_or_more"] == (
        report["recall"]["retrieved_pairs"] - histogram["1"]
    )


def test_coverage_histogram_uses_a_true_power_of_two_test():
    """`bit < all_bits` is not "exactly one"; it is "one to six".

    The four retrieved pairs here use bits 1, 6, 1 and 2. The old predicate
    ``bit > 0 AND bit < 127`` counts all four; the correct answer is the three
    whose mask holds a single bit. The 0b0110 pair is the one that separates
    them, which is why the fixture contains one.
    """
    report = _analyze()
    histogram = report["strategy_coverage"]["histogram"]
    assert report["strategy_coverage"]["retrieved_by_exactly_one"] == 3
    assert histogram["1"] == 3
    assert histogram["2"] == 1, "the 0b0110 pair, which the old predicate miscounted"
    # What the buggy predicate would have reported:
    assert report["recall"]["retrieved_pairs"] == 4
    assert report["strategy_coverage"]["retrieved_by_exactly_one"] < (
        report["recall"]["retrieved_pairs"]
    )


# --------------------------------------------------------------------------
# Bug E: the failure modes must be a partition, not a set of indicators
# --------------------------------------------------------------------------


def test_failure_modes_partition_is_disjoint_and_exhaustive():
    report = _analyze()
    modes = report["failure_modes"]
    assert modes["total"] == 2
    counts = {k: v["count"] for k, v in modes["partition"].items()}
    assert sum(counts.values()) == modes["total"]
    assert modes["integrity"]["partition_sums_to_total"]
    assert modes["integrity"]["unclassified"] == 0
    # The two misses land in different buckets, which is the point of a
    # partition: a reader can attribute 100% of the misses.
    assert counts["address_missing_either_side"] == 1
    assert counts["all_present_nothing_identical"] == 1


def test_failure_modes_percentages_sum_to_one_hundred():
    report = _analyze()
    partition = report["failure_modes"]["partition"]
    assert sum(v["percentage"] for v in partition.values()) == pytest.approx(100.0)


def test_failure_modes_indicators_are_labelled_as_overlapping():
    """The raw booleans are kept, but must not look like shares."""
    report = _analyze()
    modes = report["failure_modes"]
    assert "do NOT sum" in modes["indicators_note"]
    assert set(modes["indicators"]) == {
        "s1_record_blank",
        "vendor_record_blank",
        "name_missing_either_side",
        "address_missing_either_side",
        "name_identical",
        "address_identical",
        "all_four_fields_present",
    }
    # The indicators are genuinely overlapping here: the address-missing pair is
    # also counted under all_four_fields_present == False, and the other pair is
    # counted under all_four_fields_present == True. Their sum is not the total.
    indicator_total = sum(v["count"] for v in modes["indicators"].values())
    assert indicator_total != modes["total"]


def test_exactly_identical_name_or_address_lands_in_its_own_impossible_bucket():
    """A missed pair cannot have identical names: exact_name has no ceiling.

    Such a pair should be impossible in real output, so the partition names the
    case and reports it as zero rather than hiding it. On this sample both are 0.
    """
    report = _analyze()
    empty = report["failure_modes"]["integrity"]["expected_empty_buckets"]
    assert empty == {
        "name_identical_address_differs": 0,
        "address_identical_name_differs": 0,
    }


def test_failure_modes_on_empty_input():
    empty = pl.DataFrame(
        schema={
            "row_id": pl.Utf8,
            "vendor_id": pl.Utf8,
            "name_norm_s1": pl.Utf8,
            "name_norm_vendor": pl.Utf8,
            "addr_norm_s1": pl.Utf8,
            "addr_norm_vendor": pl.Utf8,
            "has_name_s1": pl.Boolean,
            "has_address_s1": pl.Boolean,
            "has_name_vendor": pl.Boolean,
            "has_address_vendor": pl.Boolean,
        }
    )
    result = arm.categorize_failure_modes(empty, empty)
    assert result["total"] == 0
    assert result["partition"] == {}
    assert result["integrity"]["unclassified"] == 0


# --------------------------------------------------------------------------
# Bug C: hard examples must contain the vendor text they are supposed to show
# --------------------------------------------------------------------------


def test_hard_examples_carry_the_raw_vendor_text():
    report = _analyze()
    examples = report["hard_examples"]
    assert len(examples) == 2
    for example in examples:
        assert example["vendor_business_name"], "vendor name must not be null"
        assert example["vendor_name_norm"]
        assert example["s1_business_name"]
        assert example["s1_name_norm"]


def test_hard_examples_contain_real_text_not_placeholders():
    """The actual strings, so a null-producing regression cannot pass silently."""
    report = _analyze()
    by_vendor = {e["true_vendor_id"]: e for e in report["hard_examples"]}
    assert by_vendor["v1"]["vendor_business_name"] == "Alpha Traders"
    assert by_vendor["v2"]["vendor_business_name"] == "Beta Fds"
    assert by_vendor["v2"]["s1_business_name"] == "Beta Foods"


def test_hard_examples_refuse_a_broken_vendor_join():
    """If the raw vendor name is missing, fail instead of publishing nulls.

    This is the assertion that catches the original defect. The frame carries
    ``business_name_vendor_raw``; the extractor read ``business_name_vendor``;
    every published example came out with a null raw name while the normalised
    fields stayed populated, so the artifact looked fine. A guard on "raw *and*
    normalised both missing" would have passed on it -- the guard is on the raw
    name alone, which is the only thing this check is about.
    """
    _, _, s1_norm, _, v_norm, _ = _sample_case()
    # The exact state the original artifact was published in: raw text gone,
    # normalised text intact.
    v_raw_without_text = v_norm.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("business_name"),
        pl.lit(None, dtype=pl.Utf8).alias("business_address"),
    )
    with pytest.raises(ValueError, match="empty raw vendor name"):
        arm.extract_hard_examples(
            _attach_missing(v_raw_without_text, s1_norm, v_norm),
            max_examples=10,
        )


def test_hard_examples_refuse_a_whitespace_only_vendor_name():
    """Blank and whitespace-only are both unusable in an example listing."""
    _, _, s1_norm, _, v_norm, _ = _sample_case()
    v_raw_blank = v_norm.with_columns(pl.lit("   ", dtype=pl.Utf8).alias("business_name"))
    with pytest.raises(ValueError, match="empty raw vendor name"):
        arm.extract_hard_examples(
            _attach_missing(v_raw_blank, s1_norm, v_norm), max_examples=10
        )


def test_extract_hard_examples_accepts_a_healthy_frame():
    """The same call with intact text must succeed, so the guard is not vacuous."""
    _, _, s1_norm, _, v_norm, v_raw = _sample_case()
    examples = arm.extract_hard_examples(
        _attach_missing(v_raw, s1_norm, v_norm), max_examples=10
    )
    assert len(examples) == 2
    assert all(e["vendor_business_name"] for e in examples)
    # v1 has no address, and that is allowed: an empty address is real data, and
    # the example exists precisely to show it.
    by_vendor = {e["true_vendor_id"]: e for e in examples}
    assert by_vendor["v1"]["vendor_business_address"] == ""


def test_extract_hard_examples_on_no_misses():
    assert arm.extract_hard_examples(pl.DataFrame(), max_examples=10) == []


def _attach_missing(
    v_raw: pl.DataFrame, s1_norm: pl.DataFrame, v_norm: pl.DataFrame
) -> pl.DataFrame:
    """Build the joined missed-pairs frame the extractor is given.

    Mirrors the joins inside ``analyze_misses`` so the test exercises the
    extractor's contract without duplicating the whole analysis.
    """
    found, _, _, _, _, _ = _sample_case()
    # The two pairs the fixture marks as missed (bit == 0).
    missed = found.filter(pl.col("bit") == 0)
    s1_side = s1_norm.select(
        pl.col("entity_id").alias("row_id"),
        pl.col("business_name").alias("business_name_s1"),
        pl.col("business_address").alias("business_address_s1"),
        pl.col("country").alias("country_s1"),
        pl.col("name_norm").alias("name_norm_s1"),
        pl.col("addr_norm").alias("addr_norm_s1"),
        pl.col("has_address").alias("has_address_s1"),
        pl.col("has_name").alias("has_name_s1"),
    )
    vendor_side = v_norm.select(
        pl.col("entity_id").alias("vendor_id"),
        pl.col("name_norm").alias("name_norm_vendor"),
        pl.col("addr_norm").alias("addr_norm_vendor"),
        pl.col("country").alias("country_vendor"),
        pl.col("source").alias("source_vendor"),
        pl.col("has_address").alias("has_address_vendor"),
        pl.col("has_name").alias("has_name_vendor"),
    )
    vendor_raw_side = v_raw.select(
        pl.col("entity_id").alias("vendor_id"),
        pl.col("business_name").alias("business_name_vendor_raw"),
        pl.col("business_address").alias("business_address_vendor_raw"),
        pl.col("source").alias("source_vendor_raw"),
    )
    return (
        missed.join(s1_side, on="row_id", how="left")
        .join(vendor_side, on="vendor_id", how="left")
        .join(vendor_raw_side, on="vendor_id", how="left")
    )


# --------------------------------------------------------------------------
# Bug F: an empty string is zero tokens, not one
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        (" ", 0),
        ("   ", 0),
        ("one", 1),
        ("one two", 2),
        ("one  two", 2),  # doubled space must not add a token
        ("  leading", 1),
        ("trailing  ", 1),
        (None, None),  # a null address stays null, never becomes 0 or 1
    ],
)
def test_token_count_ignores_empty_tokens(text, expected):
    # Built from a column rather than `pl.lit` so the dtype is Utf8, which is
    # what a normalised address column actually carries.
    result = pl.DataFrame({"x": [text]}, schema={"x": pl.Utf8}).select(
        arm.token_count(pl.col("x"))
    )
    assert result.item() == expected


def test_token_count_differs_from_the_naive_split_length():
    """The old expression reported 1 for "", which inflated the missing-address stats."""
    values = ["", "a b", "a  b"]
    frame = pl.DataFrame({"x": values})
    naive = frame.select(pl.col("x").str.split(" ").list.len()).to_series().to_list()
    fixed = frame.select(arm.token_count(pl.col("x"))).to_series().to_list()
    assert naive == [1, 2, 3]
    assert fixed == [0, 2, 2]
    assert naive != fixed


def test_token_count_agrees_with_a_regex_token_count():
    """Independent cross-check of the count."""
    frame = pl.DataFrame({"x": ["", "one", "one two", "one  two three", " ", None]})
    result = frame.select(
        arm.token_count(pl.col("x")).alias("fixed"),
        pl.col("x").str.count_matches(r"\S+").alias("regex"),
    )
    non_null = result.filter(pl.col("fixed").is_not_null())
    assert non_null.select((pl.col("fixed") == pl.col("regex")).all()).item()


def test_address_diagnostics_report_zero_tokens_for_a_missing_address():
    """The end-to-end effect of the token-count fix.

    The two missed pairs are: v1 with no address at all, and v2 with
    ``"3 Mil Road"`` (three tokens). The fixed count averages ``(0 + 3) / 2``;
    the old expression averaged ``(1 + 3) / 2``, inventing a token for the
    record that has no address.
    """
    report = _analyze()
    missed = report["address_similarity_diagnostics"]["missed"]
    assert missed["vendor_token_count"]["mean"] == pytest.approx(1.5)
    # The S1 side has no missing addresses in this fixture, so it is unchanged.
    assert missed["s1_token_count"]["mean"] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# The per-strategy numbers come from the shared module, not a second copy
# --------------------------------------------------------------------------


def test_per_strategy_uses_the_shared_standalone_cumulative_marginal_split():
    report = _analyze()
    rows = {r["strategy"]: r for r in report["per_strategy"]}
    assert set(rows) == {name for name, _, _ in arm.PLAN}
    for row in rows.values():
        assert "standalone_pair_recall" in row
        assert "cumulative_pair_recall" in row
        assert "marginal_pair_recall" in row
        # The ambiguous bare name must be gone -- that was the E011 bug.
        assert "pair_recall" not in row
    # Six true pairs. exact_name and alnum_name each found two; sorted_name one;
    # the remaining four found none, because the synthetic bitmask only sets the
    # first three bits.
    assert rows["exact_name"]["standalone_hits"] == 2
    # The recalls are rounded to six places in the artifact, so the tolerance has
    # to allow for that: 1/6 rounds to 0.166667.
    assert rows["exact_name"]["standalone_pair_recall"] == pytest.approx(2 / 6, abs=1e-6)
    assert rows["alnum_name"]["standalone_pair_recall"] == pytest.approx(2 / 6, abs=1e-6)
    assert rows["sorted_name"]["standalone_pair_recall"] == pytest.approx(1 / 6, abs=1e-6)
    for name in ("address_exact", "name_token", "address_token", "numeric_token"):
        assert rows[name]["standalone_pair_recall"] == 0.0
    # Cumulative is a property of the plan prefix, and by the last strategy it
    # equals the union: no strategy after sorted_name added anything.
    assert rows["sorted_name"]["cumulative_pair_recall"] == pytest.approx(4 / 6, abs=1e-6)
    assert rows["numeric_token"]["cumulative_pair_recall"] == pytest.approx(
        report["recall"]["union_pair_recall"]
    )


# --------------------------------------------------------------------------
# Regression anchors against the real artifact
# --------------------------------------------------------------------------


@pytest.mark.skipif(not ARTIFACT_PATH.exists(), reason="E012 artifact not present")
def test_published_artifact_records_the_sample_it_actually_ran_on():
    artifact = json.loads(ARTIFACT_PATH.read_text())
    assert artifact["config"]["query_rows"] == 500, (
        "E012 ran at --query-rows 500. If this artifact says 200000, the metadata "
        "bug is back and every percentage in it is being misread as a population "
        "figure."
    )
    assert artifact["config"]["seed"] == 17
    assert artifact["sample_scope"]["queries_sampled"] == 500
    assert "500-query sample" in artifact["sample_scope"]["generalisation"]


@pytest.mark.skipif(not ARTIFACT_PATH.exists(), reason="E012 artifact not present")
def test_published_artifact_is_internally_consistent():
    artifact = json.loads(ARTIFACT_PATH.read_text())
    recall = artifact["recall"]
    assert recall["true_pairs"] == 1740
    assert recall["missed_pairs"] == 420
    assert recall["retrieved_pairs"] == 1320
    assert recall["retrieved_pairs"] + recall["missed_pairs"] == recall["true_pairs"]
    assert round(recall["union_pair_recall"], 6) == pytest.approx(1320 / 1740, abs=1e-6)
    assert round(recall["union_entity_recall"], 6) == pytest.approx(0.946695, abs=1e-6)

    coverage = artifact["strategy_coverage"]
    histogram = coverage["histogram"]
    assert sum(histogram.values()) == recall["true_pairs"]
    assert histogram["0"] == recall["missed_pairs"]
    assert coverage["retrieved_by_exactly_one"] == histogram["1"] == 347
    assert coverage["retrieved_by_all_strategies"] == histogram["7"] == 2

    modes = artifact["failure_modes"]
    assert sum(v["count"] for v in modes["partition"].values()) == modes["total"] == 420
    assert modes["integrity"]["partition_sums_to_total"]
    assert modes["integrity"]["unclassified"] == 0
    assert modes["partition"]["all_present_nothing_identical"]["count"] == 386
    assert modes["partition"]["address_missing_either_side"]["count"] == 34


@pytest.mark.skipif(not ARTIFACT_PATH.exists(), reason="E012 artifact not present")
def test_published_artifact_hard_examples_have_vendor_text():
    artifact = json.loads(ARTIFACT_PATH.read_text())
    examples = artifact["hard_examples"]
    assert len(examples) == 100
    nulls = sum(1 for e in examples if not e["vendor_business_name"])
    assert nulls == 0, (
        "all 100 hard examples had a null vendor name in the original artifact; "
        "the extractor read the wrong column"
    )


@pytest.mark.skipif(not BITMASK_PATH.exists(), reason="E012 bitmask not present")
def test_bitmask_is_one_row_per_true_pair_over_the_seven_strategies():
    from team_diamond.retrieval.recall_stats import validate_bitmask

    found = pl.read_parquet(BITMASK_PATH)
    validate_bitmask(found, 1740)
    assert found.columns == ["row_id", "vendor_id", "bit"]
    assert int(found["bit"].min()) == 0
    assert int(found["bit"].max()) == (1 << len(arm.PLAN)) - 1
    assert found.select(pl.struct(["row_id", "vendor_id"]).n_unique()).item() == found.height
    assert int((found["bit"] == 0).sum()) == 420
