"""Hand-checked end-to-end test of the feature builder.

Run directly (not via pytest) as a smoke check while iterating:

    uv run python -m tests.test_features_smoke
"""

from __future__ import annotations

import polars as pl

from team_diamond.features import (
    FEATURE_COLUMNS,
    build_pair_features,
    build_token_rarity,
)

S1 = pl.DataFrame(
    {
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "name_norm": ["prime money forex", "orelees barbershop", "khan silks"],
        "addr_norm": ["17560 ellis road tahlequah ok", "1795 westchester drive", ""],
        "country": ["US", "US", "IN"],
    }
)

VENDORS = pl.DataFrame(
    {
        "entity_id": ["S2-1", "S2-2", "S3-1", "S2-3"],
        "name_norm": [
            "forex prime money",
            "orelees barber shop",
            "khan silk store",
            "completely different",
        ],
        "addr_norm": [
            "17560 ellis rd tahlequah ok 74155",
            "1795 westchester dr",
            "",
            "1 unrelated street",
        ],
        "country": ["US", "US", "IN", "FR"],
        "source": [2, 2, 3, 2],
    }
)

PAIRS = pl.DataFrame(
    {
        "s1_id": ["S1-1", "S1-2", "S1-3", "S1-1"],
        "vendor_id": ["S2-1", "S2-2", "S3-1", "S2-3"],
        "n_keys": [2, 1, 3, 1],
        "strategies": [
            ["exact_name", "name_token"],
            ["exact_name"],
            ["name_token", "address_token", "numeric_token"],
            ["name_token"],
        ],
        "cand_count_s1": [4, 2, 1, 4],
        "cand_count_vendor": [1, 1, 2, 7],
        "strategy_rank": [1, 1, 1, 4],
        "s1_name_key_df": [3, 1, 1, 850],
    }
)


def main() -> None:
    # The rarity corpus is the vendor pool only -- deliberately *not* the S1
    # rows, because at inference time the S1 rows do not exist yet and including
    # them would make the feature distribution differ between train and test.
    rarity = build_token_rarity(
        VENDORS.select("entity_id", "name_norm", "addr_norm"),
        label="smoke:train-vendors",
    )
    out = build_pair_features(PAIRS, S1, VENDORS, rarity=rarity)

    print(f"pairs in : {PAIRS.height}")
    print(f"pairs out: {out.height}")
    print(f"features : {len(FEATURE_COLUMNS)} declared, "
          f"{len(out.columns) - 3} produced (+ 2 id columns)\n")

    # Column-order contract: CatBoost is fed positionally.
    expected = ["s1_id", "vendor_id", "country_pair", *FEATURE_COLUMNS]
    assert out.columns == expected, (
        f"column order drift!\n  got     : {out.columns[:8]}...\n"
        f"  expected: {expected[:8]}..."
    )
    print("column order matches FEATURE_COLUMNS contract: OK")

    nulls = {
        c: out[c].null_count()
        for c in FEATURE_COLUMNS
        if out[c].null_count() > 0
    }
    print(f"null feature cells: {nulls if nulls else 'none'}\n")

    show = [
        "s1_id", "vendor_id",
        "name_exact", "name_sorted_exact", "name_jaccard", "name_containment",
        "fz_name_token_set_ratio", "name_prefix_overlap",
        "addr_jaccard", "addr_containment", "num_sig_jaccard",
        "has_v_addr", "addr_conflict", "name_in_addr",
        "name_idf_min_shared", "name_idf_sum_shared", "name_idf_weighted_jaccard",
        "same_country", "n_strategies", "cand_count_vendor", "is_top1_by_key_count",
    ]
    with pl.Config(tbl_cols=len(show), tbl_width_chars=250, fmt_str_lengths=12):
        print(out.select(show))

    print("\n--- expected behaviour ---")
    checks = [
        ("word-reordered names are not name_exact",
         out.row(0, named=True)["name_exact"] == 0.0),
        ("...but sorted_name catches them (name_sorted_exact)",
         out.row(0, named=True)["name_sorted_exact"] == 1.0),
        ("token_set_ratio is high for the reorder",
         out.row(0, named=True)["fz_name_token_set_ratio"] > 0.9),
        # S1 has "17560"; S2 has "17560 74155". The house number agrees and the
        # postcode is absent on one side, so Jaccard is 1/2 -- the feature
        # correctly rewards partial numeric agreement without claiming a full
        # match on a number the S1 record simply does not carry.
        ("house number agrees, postcode missing on S1 -> 0.5",
         out.row(0, named=True)["num_sig_jaccard"] == 0.5),
        ("address token overlap is partial, not perfect",
         0.0 < out.row(0, named=True)["addr_jaccard"] < 1.0),
        ("unrelated pair has near-zero name similarity",
         out.row(3, named=True)["name_jaccard"] == 0.0),
        ("address-less vendor yields has_v_addr == 0",
         out.row(2, named=True)["has_v_addr"] == 0.0),
        ("address-less vendor does NOT fake perfect address overlap",
         out.row(2, named=True)["addr_containment"] == 0.0),
        ("country mismatch is visible",
         out.row(3, named=True)["same_country"] == 0.0),
        ("n_strategies counts retrieval evidence",
         out.row(2, named=True)["n_strategies"] == 3.0),
        ("a very common name key has high df (S1-1's 850-row key)",
         out.row(3, named=True)["s1_name_key_df"] == 850.0),
        ("no nulls anywhere in the feature matrix",
         sum(out[c].null_count() for c in FEATURE_COLUMNS) == 0),
        ("no NaN/inf leaked into the feature matrix",
         all(
             out[c].is_finite().all()
             for c in FEATURE_COLUMNS
             if out[c].dtype.is_float()
         )),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
