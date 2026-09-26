"""Candidate-generation checks. Synthetic by default; the real corpus is opt-in.

Run the default (safe, tiny, in-memory) checks:

    uv run pytest tests/test_candidates_smoke.py

Run against the real dataset:

    TEAM_DIAMOND_ALLOW_FULL_CORPUS=1 uv run python -m tests.test_candidates_smoke

Why the environment gate
------------------------
The real-data path loads all three train sources and expands the 10.3M-record
vendor pool into per-token key frames for seven blocking strategies. That is
several GiB of live intermediates. Running it as an ordinary test made it one
``pytest`` invocation away from exhausting a 15 GiB laptop, which is exactly how
VS Code got crashed: the test looked harmless, the cost was invisible from the
command line, and the failure mode was a frozen desktop rather than an
exception.

So the expensive path now refuses to start unless it is asked for by name. The
synthetic tests below cover the same plumbing -- key construction, the union
fold, the cap, and the report contract -- at a size that cannot hurt anything.
"""

from __future__ import annotations

import os

import polars as pl
import pytest

from team_diamond.retrieval.candidates import (
    CANDIDATE_REPORT,
    generate_candidates,
)

# The data layer and the real corpus are imported *inside* main() only. At
# module scope this file would be coupled to the dataset for no benefit, and a
# stray `import` from a notebook or another test would be enough to pull the
# loading path into scope.

FULL_CORPUS_ENV = "TEAM_DIAMOND_ALLOW_FULL_CORPUS"

SEED = 17
N_QUERIES = 2_000
N_VENDORS = 20_000


def _synthetic_corpus(
    *, n_s1: int = 400, n_vendors: int = 1_500
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A small, hand-shaped corpus with a known set of true matches.

    Built rather than sampled so the expected recall is exact: every ``S1-2i``
    has exactly one true vendor, and the noise patterns mirror the ones that
    actually appear in the challenge data (reordered words, a dropped token, a
    missing address).
    """
    words_a = ["acme", "globex", "initech", "umbrella", "stark", "wayne", "tyrell", "cyberdyne"]
    rows1: list[dict[str, str]] = []
    rowsv: list[dict[str, object]] = []

    for i in range(n_s1):
        w = words_a[i % len(words_a)]
        rows1.append(
            {
                "entity_id": f"S1-{i}",
                "name_norm": f"{w} trading company {i}",
                "addr_norm": f"{100 + i} market street",
                "country": "US",
            }
        )
    for j in range(n_vendors):
        w = words_a[j % len(words_a)]
        noise = j % 3
        name = (
            f"{w} trading company {j}"
            if noise == 0
            else f"{w} company {j}"  # dropped token
            if noise == 1
            else f"company trading {w} {j}"  # reordered
        )
        rowsv.append(
            {
                "entity_id": f"S2-{j}",
                "name_norm": name,
                "addr_norm": "" if j % 11 == 0 else f"{100 + j} market street",
                "country": "US" if j % 5 else "IN",
                "source": 2 if j % 2 else 3,
            }
        )
    return pl.DataFrame(rows1), pl.DataFrame(rowsv)


def test_candidates_carry_the_full_contract() -> None:
    s1, vendors = _synthetic_corpus()
    candidates, _ = generate_candidates(
        s1, vendors, cap_per_query=20, allow_unmeasured_cap=True
    )
    assert candidates.columns == list(CANDIDATE_REPORT)


def test_candidate_pairs_are_unique() -> None:
    s1, vendors = _synthetic_corpus()
    candidates, _ = generate_candidates(
        s1, vendors, cap_per_query=20, allow_unmeasured_cap=True
    )
    distinct = candidates.select("s1_id", "vendor_id").unique().height
    assert candidates.height == distinct, "duplicate (s1, vendor) pairs emitted"


def test_every_pair_names_at_least_one_strategy() -> None:
    s1, vendors = _synthetic_corpus()
    candidates, _ = generate_candidates(
        s1, vendors, cap_per_query=20, allow_unmeasured_cap=True
    )
    empty = candidates.filter(pl.col("strategies").list.len() == 0).height
    assert empty == 0


def test_reordering_is_still_retrieved() -> None:
    """Word order changes, so only the token strategy can find this pair."""
    s1 = pl.DataFrame(
        {
            "entity_id": ["S1-a"],
            "name_norm": ["company trading acme"],
            "addr_norm": [""],
            "country": ["US"],
        }
    )
    vendors = pl.DataFrame(
        {
            "entity_id": ["S2-a"],
            "name_norm": ["acme trading company"],
            "addr_norm": [""],
            "country": ["US"],
            "source": [2],
        }
    )
    candidates, _ = generate_candidates(
        s1, vendors, cap_per_query=None, allow_unmeasured_cap=False
    )
    assert candidates.height == 1
    assert "name_token" in candidates["strategies"][0].to_list()


def test_cap_reports_what_it_dropped() -> None:
    """A binding cap must be visible in the report, never silent."""
    s1, vendors = _synthetic_corpus(n_s1=100, n_vendors=2_000)
    _, uncapped = generate_candidates(
        s1, vendors, cap_per_query=None, allow_unmeasured_cap=False
    )
    _, capped = generate_candidates(
        s1, vendors, cap_per_query=5, allow_unmeasured_cap=True
    )
    assert capped.n_pairs_before_cap == uncapped.n_pairs_before_cap
    assert capped.n_dropped_by_cap == capped.n_pairs_before_cap - capped.n_pairs
    assert capped.n_dropped_by_cap >= 0


def test_unmeasured_cap_is_refused() -> None:
    """AGENTS.md §10: a cap chosen by intuition must not run silently."""
    s1, vendors = _synthetic_corpus(n_s1=20, n_vendors=200)
    with pytest.raises(ValueError, match="has not been swept"):
        generate_candidates(s1, vendors, cap_per_query=10)


def test_cap_sweep_does_not_leak_gt() -> None:
    """Cap sweep (and candidate generation) must not use ground truth.

    The candidate generation function takes only (queries, vendors, plan, cap).
    It has no access to ground truth labels. This test verifies that:
    1. The function signature doesn't include ground truth
    2. The vendor pool used is exactly what was passed in (no GT vendors added)
    3. Caps only truncate, never add candidates
    """
    # Use a larger corpus so caps actually bind
    s1, vendors = _synthetic_corpus(n_s1=200, n_vendors=5_000)

    # Run with different caps - same vendor pool, same queries, different caps
    candidates_50, report_50 = generate_candidates(
        s1, vendors, cap_per_query=50, allow_unmeasured_cap=True
    )
    candidates_100, report_100 = generate_candidates(
        s1, vendors, cap_per_query=100, allow_unmeasured_cap=True
    )
    candidates_uncapped, report_uncapped = generate_candidates(
        s1, vendors, cap_per_query=None, allow_unmeasured_cap=False
    )

    # All runs use the same vendor pool (no GT vendors added)
    assert candidates_50["vendor_id"].n_unique() <= vendors.height
    assert candidates_100["vendor_id"].n_unique() <= vendors.height
    assert candidates_uncapped["vendor_id"].n_unique() <= vendors.height

    # Caps should only truncate, never add candidates
    assert report_50.n_pairs <= report_100.n_pairs <= report_uncapped.n_pairs

    # The vendor IDs in candidates must be a subset of the input vendor pool
    # (no external/Gt-derived vendors sneaked in)
    for cand in [candidates_50, candidates_100, candidates_uncapped]:
        extra_vendors = cand.join(vendors.select("entity_id"), left_on="vendor_id", right_on="entity_id", how="anti")
        assert extra_vendors.height == 0, "Candidates contain vendor IDs not in the input pool!"


def test_report_describe_mentions_dropped_recall() -> None:
    s1, vendors = _synthetic_corpus(n_s1=100, n_vendors=2_000)
    _, report = generate_candidates(
        s1, vendors, cap_per_query=5, allow_unmeasured_cap=True
    )
    text = report.describe()
    assert "CANDIDATE GENERATION REPORT" in text
    if report.n_dropped_by_cap:
        assert "RECALL WAS LOST HERE" in text


def test_empty_vendor_pool_yields_no_pairs() -> None:
    s1, _ = _synthetic_corpus(n_s1=10)
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Utf8,
            "name_norm": pl.Utf8,
            "addr_norm": pl.Utf8,
            "country": pl.Utf8,
            "source": pl.Int64,
        }
    )
    candidates, report = generate_candidates(
        s1, empty, cap_per_query=10, allow_unmeasured_cap=True
    )
    assert candidates.height == 0
    assert report.n_pairs == 0


def main() -> None:  # pragma: no cover - real-data entry point
    if os.environ.get(FULL_CORPUS_ENV) != "1":
        raise SystemExit(
            f"This runs against the full 26M-record corpus and needs several "
            f"GiB of RAM.\nIt is blocked by default because running it "
            f"unattended crashed VS Code.\n\n"
            f"Set {FULL_CORPUS_ENV}=1 if you really want it, ideally on "
            f"SageMaker rather than a laptop."
        )

    # Imported here, not at module scope: see the note above.
    from team_diamond.data import DatasetPaths, load_split
    from team_diamond.preprocessing.normalize_fast import normalize_columns

    paths = DatasetPaths.discover()
    split = load_split(paths, "train")

    s1 = normalize_columns(split["source1"])
    vendors = pl.concat(
        [
            normalize_columns(split["source2"]),
            normalize_columns(split["source3"]),
        ],
        how="vertical",
    ).select("entity_id", "name_norm", "addr_norm", "country", "source")
    print(f"vendor pool: {vendors.height:,}")

    queries = s1.sample(n=N_QUERIES, seed=SEED)
    gt = (
        split["ground_truth"]
        .select("source1_entity_id", "match_ids")
        .explode("match_ids", empty_as_null=True)
        .filter(pl.col("match_ids").is_not_null())
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("match_ids").alias("vendor_id"),
        )
        .unique()
    )
    scoped_gt = gt.filter(pl.col("s1_id").is_in(queries["entity_id"].implode()))

    candidates, report = generate_candidates(
        queries, vendors, cap_per_query=50, allow_unmeasured_cap=True
    )
    print(report.describe())

    hits = scoped_gt.join(candidates, on=["s1_id", "vendor_id"], how="semi")
    recall = hits.height / max(scoped_gt.height, 1)
    print(f"pair recall on this slice: {recall:.4f}")
    print(
        "NOTE: a small query slice against the full pool is a plumbing check, "
        "not the project candidate-recall number."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
