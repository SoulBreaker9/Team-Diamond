"""Per-candidate-pair feature matrix.

Layer: features
See AGENTS.md §11 (Features), AGENTS.md §7 (layer separation)

What this module does and does not do
-------------------------------------
It turns a set of candidate pairs into a numeric matrix. It does **not** rank,
cap, threshold, or decide anything: given the same pairs and the same rarity
tables, it produces the same matrix. The decision of which pairs exist belongs
to :mod:`team_diamond.retrieval.candidates`; the decision of which pairs to
emit belongs to :mod:`team_diamond.decision`.

The four feature families
------------------------
1. **Name similarity** -- token overlap, four rapidfuzz ratios, prefix overlap.
2. **Address similarity** -- the same shape, plus a numeric-signature feature.
3. **Cross-field** -- does the name appear inside the address, and vice versa.
   These fire on a large fraction of genuine matches where the name is spelled
   correctly but formatted completely differently, and they are the main reason
   a high-address-noise pair can still be matched.
4. **Rarity and retrieval evidence** -- how *rare* the shared tokens are, how
   many independent strategies retrieved the pair, and how much competition the
   candidate had. This is the family that distinguishes "similar" from
   "identifiable", and per AGENTS.md §11 it is the one that matters most.

Why numeric signatures rather than a house-number column
-------------------------------------------------------
House numbers and postcodes survive address reformatting far better than the
surrounding prose, but they are only discriminative *within* a locality: "12"
alone is nearly meaningless. Rather than a bespoke parser with a country-shaped
heuristic attached (forbidden by AGENTS.md §8.5), each record's digit runs are
reduced to a sorted, de-duplicated signature string such as ``"12 48014"``, and
ordinary token-overlap statistics are computed over that. The comparison stays
language-agnostic, and a shared *pair* of numbers still scores higher than a
shared single number, which is the behaviour we want.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import polars as pl

from team_diamond.features.idf import UNSEEN_IDF, TokenRarity
from team_diamond.features.similarity import (
    FUZZY_SCORERS,
    batch_fuzzy_ratios,
    clean_token_list,
    prefix_overlap,
    token_overlap_features,
)

__all__ = [
    "FEATURE_COLUMNS",
    "PAIR_KEY_COLUMNS",
    "build_pair_features",
    "numeric_signature",
]

#: Identifying columns of the pair frame, never fed to the model.
PAIR_KEY_COLUMNS: Final[tuple[str, ...]] = ("s1_id", "vendor_id")

#: The full ordered feature list. Order is part of the contract: the model is
#: fitted positionally by CatBoost from a numpy array, so this tuple, the column
#: order actually produced, and the training matrix must agree exactly. The
#: builder asserts that they do rather than trusting a caller.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # -- name ---------------------------------------------------------------
    "name_exact",
    "name_alnum_exact",
    "name_sorted_exact",
    "name_jaccard",
    "name_dice",
    "name_containment",
    "name_coverage",
    "name_overlap",
    "name_left_only",
    "name_right_only",
    "name_prefix_overlap",
    "name_len_abs_diff",
    "name_len_ratio",
    "fz_name_ratio",
    "fz_name_partial_ratio",
    "fz_name_token_set_ratio",
    "fz_name_token_sort_ratio",
    # -- address ------------------------------------------------------------
    "addr_exact",
    "addr_jaccard",
    "addr_dice",
    "addr_containment",
    "addr_coverage",
    "addr_overlap",
    "addr_prefix_overlap",
    "addr_len_abs_diff",
    "fz_addr_ratio",
    "fz_addr_partial_ratio",
    "fz_addr_token_set_ratio",
    "fz_addr_token_sort_ratio",
    "num_sig_exact",
    "num_sig_jaccard",
    "num_sig_overlap",
    "has_s1_addr",
    "has_v_addr",
    "both_addr_present",
    "addr_conflict",
    # -- cross-field --------------------------------------------------------
    "name_in_addr",
    "addr_in_name",
    "name_token_addr_coverage",
    # -- rarity -------------------------------------------------------------
    "name_idf_min_shared",
    "name_idf_sum_shared",
    "name_idf_weighted_jaccard",
    "name_idf_max",
    "addr_idf_min_shared",
    "addr_idf_sum_shared",
    # -- retrieval evidence -------------------------------------------------
    "same_country",
    "source_is_s3",
    "n_strategies",
    "n_keys",
    "cand_count_s1",
    "cand_count_vendor",
    "strategy_rank",
    "is_top1_by_key_count",
    "s1_name_key_df",
    # -- record shape -------------------------------------------------------
    "s1_name_len",
    "v_name_len",
    "s1_addr_len",
    "v_addr_len",
    "s1_n_tokens",
    "v_n_tokens",
)

#: Categorical columns, passed to CatBoost as such rather than one-hot encoded
#: by hand. Country is categorical for the same reason it must not be branched
#: on in code (AGENTS.md §8.5): the test split contains a country absent from
#: train, and a tree over country *values* degrades to an unseen-value bucket,
#: whereas a tree over country *properties* does not.
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = ("country_pair",)


def numeric_signature(addr: pl.Expr) -> pl.Expr:
    """Reduce an address to its sorted, de-duplicated digit runs.

    Args:
        addr: Expression yielding a normalised address string.

    Returns:
        Expression yielding a space-separated signature such as ``"12 48014"``,
        or ``""`` when the address contains no digits.
    """
    digits = (
        addr.fill_null("")
        .str.replace_all(r"\D+", " ", literal=False)
        .str.replace_all(r"\s+", " ", literal=False)
        .str.strip_chars()
    )
    return (
        pl.when(digits == "")
        .then(pl.lit(""))
        .otherwise(
            digits.str.split(" ")
            .list.unique()
            .list.sort()
            .list.join(" ")
        )
    )


def _sorted_name_expr(name: pl.Expr) -> pl.Expr:
    """Token-sorted, de-duplicated name, as a comparable string."""
    return (
        name.fill_null("")
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.unique()
        .list.sort()
        .list.join(" ")
    )


def _alnum_name_expr(name: pl.Expr) -> pl.Expr:
    """Name with all non-alphanumerics removed."""
    return name.fill_null("").str.replace_all(r"[^\w]+", "", literal=False)


def _attach_record_columns(
    pairs: pl.DataFrame,
    records: pl.DataFrame,
    *,
    id_column: str,
    prefix: str,
    passthrough: Sequence[str] = (),
) -> pl.DataFrame:
    """Join normalised record fields onto the pair frame under a prefix.

    Only the columns the feature layer needs are carried across, and list
    columns are avoided entirely: a ``List`` column multiplied across 35M pairs
    costs memory proportional to the token count, whereas the normalised string
    is one pointer per pair and is split lazily at feature time.

    Args:
        pairs: The pair frame, already renamed so the id column is ``id_column``.
        records: Record frame keyed on ``id_column``.
        id_column: Key column present in both frames.
        prefix: ``"s1"`` or ``"v"``; prefixed onto every attached column.
        passthrough: Extra record columns to carry verbatim under the prefix,
            for fields that are features in their own right (``source``).

    Returns:
        ``pairs`` with the record columns attached.
    """
    wanted = records.select(
        pl.col(id_column).alias("_rid"),
        pl.col("name_norm").alias(f"{prefix}_name"),
        pl.col("addr_norm").alias(f"{prefix}_addr"),
        pl.col("country").alias(f"{prefix}_country"),
        numeric_signature(pl.col("addr_norm")).alias(f"{prefix}_num"),
        *[pl.col(c).alias(f"{prefix}_{c}") for c in passthrough],
    )
    return pairs.join(wanted, left_on=id_column, right_on="_rid", how="left")


def _overlap_block(
    frame: pl.DataFrame, *, left: str, right: str, prefix: str, id_column: str
) -> pl.DataFrame:
    """Token-overlap struct for one field, unnested into prefixed columns.

    ``id_column`` is carried through so the caller can join the block back onto
    the pair frame by key rather than relying on row order.
    """
    struct = token_overlap_features(
        clean_token_list(left), clean_token_list(right)
    ).alias("_ov")
    block = frame.with_columns(struct).select(
        pl.col(id_column),
        pl.col("_ov").struct.field("jaccard").alias(f"{prefix}_jaccard"),
        pl.col("_ov").struct.field("dice").alias(f"{prefix}_dice"),
        pl.col("_ov").struct.field("containment").alias(f"{prefix}_containment"),
        pl.col("_ov").struct.field("coverage").alias(f"{prefix}_coverage"),
        pl.col("_ov").struct.field("overlap").cast(pl.Int32).alias(f"{prefix}_overlap"),
        pl.col("_ov").struct.field("n_left").cast(pl.Int32).alias(f"{prefix}_n_left"),
        pl.col("_ov").struct.field("n_right").cast(pl.Int32).alias(f"{prefix}_n_right"),
    )
    if prefix == "name":
        block = block.with_columns(
            (pl.col("name_n_left") - pl.col("name_overlap"))
            .cast(pl.Int32)
            .alias("name_left_only"),
            (pl.col("name_n_right") - pl.col("name_overlap"))
            .cast(pl.Int32)
            .alias("name_right_only"),
        )
    return block


def _idf_overlap_block(
    frame: pl.DataFrame,
    rarity: TokenRarity,
    *,
    left: str,
    right: str,
    id_column: str,
    prefix: str,
) -> pl.DataFrame:
    """Rarity-weighted overlap for one field, via explode and self-join.

    A per-pair subquery would be the obvious way to do this and is unusable at
    35M pairs. Instead the pair frame is exploded into ``(pair, token)`` rows
    for each side, the two sides are self-joined on ``(pair, token)`` to isolate
    shared tokens, and the IDF table is joined on ``token``. Two grouped
    aggregates then give everything needed. The frame is ~4x the pair count
    rather than a correlated lookup per pair.

    Args:
        frame: Pair frame, already carrying the two normalised text columns.
        rarity: Token rarity tables. **Must be scoped to the corpus the model is
            being evaluated on** -- see :mod:`team_diamond.features.idf`.
        left: Normalised text column for the S1 side.
        right: Normalised text column for the vendor side.
        id_column: Unique per-pair id used to group the aggregates.
        prefix: ``"name"`` or ``"addr"``.

    Returns:
        Frame of ``(id_column, {prefix}_idf_min_shared, {prefix}_idf_sum_shared,
        {prefix}_idf_weighted_jaccard, {prefix}_idf_max)``.
    """
    table = (rarity.name.frame if prefix == "name" else rarity.address.frame).rename(
        {"token": "tok"}
    )
    if table.is_empty():
        return frame.select(
            pl.col(id_column),
            pl.lit(0.0).alias(f"{prefix}_idf_min_shared"),
            pl.lit(0.0).alias(f"{prefix}_idf_sum_shared"),
            pl.lit(0.0).alias(f"{prefix}_idf_weighted_jaccard"),
            pl.lit(0.0).alias(f"{prefix}_idf_max"),
        )

    long = (
        frame.select(pl.col(id_column), pl.col(left).alias("tok"), pl.lit(0).alias("side"))
        .with_columns(clean_token_list("tok").alias("tok"))
        .explode("tok")
        .filter(pl.col("tok") != "")
    )
    long_right = (
        frame.select(pl.col(id_column), pl.col(right).alias("tok"), pl.lit(1).alias("side"))
        .with_columns(clean_token_list("tok").alias("tok"))
        .explode("tok")
        .filter(pl.col("tok") != "")
    )
    both = pl.concat([long, long_right])

    # Isolating shared tokens: a pair's token appearing with both side values.
    shared = (
        both.group_by(id_column, "tok")
        .agg(pl.col("side").n_unique().alias("_sides"))
        .filter(pl.col("_sides") == 2)
        .select(pl.col(id_column), "tok")
        .join(table, on="tok", how="left")
        .with_columns(pl.col("idf").fill_null(UNSEEN_IDF))
    )
    # Anchor the per-pair aggregate on the FULL id set, not on `both`. When
    # both sides of a pair are empty -- an address-less S1 record against an
    # address-less vendor, which is not rare here -- `both` contributes no rows
    # for that pair, and an aggregate built from it would omit the pair
    # entirely. The caller's outer left join would then fill all four columns
    # with null, including the ones this function is meant to guarantee as 0.0.
    per_pair = (
        frame.select(pl.col(id_column))
        .unique()
        .join(
            both.join(table, on="tok", how="left")
            .with_columns(pl.col("idf").fill_null(UNSEEN_IDF))
            .group_by(id_column)
            .agg(pl.col("idf").sum().alias("_total_idf")),
            on=id_column,
            how="left",
        )
        .with_columns(pl.col("_total_idf").fill_null(0.0))
    )
    shared_agg = shared.group_by(id_column).agg(
        pl.col("idf").min().alias(f"{prefix}_idf_min_shared"),
        pl.col("idf").sum().alias(f"{prefix}_idf_sum_shared"),
        pl.col("idf").max().alias(f"{prefix}_idf_max"),
    )

    # Two separate with_columns calls, deliberately. Polars evaluates every
    # expression in one with_columns against the *input* frame, so filling the
    # nulls and then dividing by them in the same call divides null by null and
    # yields a null ratio -- silently, and for exactly the pairs with no shared
    # token, which are the pairs a null would distort most.
    out = per_pair.join(shared_agg, on=id_column, how="left").with_columns(
        pl.col(f"{prefix}_idf_min_shared").fill_null(0.0),
        pl.col(f"{prefix}_idf_sum_shared").fill_null(0.0),
        pl.col(f"{prefix}_idf_max").fill_null(0.0),
    )
    out = out.with_columns(
        (
            pl.col(f"{prefix}_idf_sum_shared")
            / pl.when(pl.col("_total_idf") > 0)
            .then(pl.col("_total_idf"))
            .otherwise(1.0)
        ).alias(f"{prefix}_idf_weighted_jaccard"),
    )
    return out.select(
        pl.col(id_column),
        pl.col(f"{prefix}_idf_min_shared"),
        pl.col(f"{prefix}_idf_sum_shared"),
        pl.col(f"{prefix}_idf_weighted_jaccard"),
        pl.col(f"{prefix}_idf_max"),
    )


def build_pair_features(
    pairs: pl.DataFrame,
    s1: pl.DataFrame,
    vendors: pl.DataFrame,
    *,
    rarity: TokenRarity,
    s1_id_column: str = "s1_id",
    vendor_id_column: str = "vendor_id",
    s1_name_column: str = "name_norm",
    s1_addr_column: str = "addr_norm",
    vendor_name_column: str = "name_norm",
    vendor_addr_column: str = "addr_norm",
    fuzzy_scorers: Sequence[str] = FUZZY_SCORERS,
    fuzzy_workers: int = -1,
) -> pl.DataFrame:
    """Build the numeric feature matrix for a set of candidate pairs.

    Args:
        pairs: Long frame with at least ``s1_id`` and ``vendor_id``. May also
            carry retrieval evidence: ``n_keys``, ``strategies`` (list of
            strategy names), ``cand_count_s1``, ``cand_count_vendor``,
            ``strategy_rank``, ``s1_name_key_df``. Missing columns are filled
            with neutral defaults so a minimal pair frame still works.
        s1: S1 records with ``entity_id``, ``name_norm``, ``addr_norm``,
            ``country``.
        vendors: Vendor records (S2 + S3) with the same columns plus ``source``.
        rarity: Name and address token rarity. **Fold-scoped**; see
            :mod:`team_diamond.features.idf`.
        s1_id_column: Pair column holding the S1 entity id.
        vendor_id_column: Pair column holding the vendor entity id.
        s1_name_column: Name column in ``s1``.
        s1_addr_column: Address column in ``s1``.
        vendor_name_column: Name column in ``vendors``.
        vendor_addr_column: Address column in ``vendors``.
        fuzzy_scorers: Subset of rapidfuzz scorers to compute.
        fuzzy_workers: rapidfuzz worker count; ``-1`` uses every core.

    Returns:
        Frame with ``s1_id``, ``vendor_id``, and every column in
        :data:`FEATURE_COLUMNS` plus ``country_pair``.

    Raises:
        ValueError: If the required id columns are absent from ``pairs``, or if
            a produced column is missing from :data:`FEATURE_COLUMNS`. The
            second check exists because CatBoost is fed a positional numpy
            array, and a silent column-order drift would corrupt every
            prediction without raising anything.
    """
    missing = [c for c in (s1_id_column, vendor_id_column) if c not in pairs.columns]
    if missing:
        raise ValueError(
            f"pairs frame is missing required id column(s) {missing}; "
            f"present: {pairs.columns}"
        )

    frame = pairs
    # Fill optional retrieval-evidence columns so the arithmetic below is total.
    defaults: dict[str, pl.Expr] = {
        "n_keys": pl.lit(1, dtype=pl.Int32),
        "cand_count_s1": pl.lit(1, dtype=pl.Int32),
        "cand_count_vendor": pl.lit(1, dtype=pl.Int32),
        "strategy_rank": pl.lit(1, dtype=pl.Int32),
        "s1_name_key_df": pl.lit(0, dtype=pl.Int32),
        "strategies": pl.lit([]).cast(pl.List(pl.Utf8)),
    }
    for name, expr in defaults.items():
        if name not in frame.columns:
            frame = frame.with_columns(expr.alias(name))

    # A stable per-pair id, so the two IDF aggregations can be joined back
    # without relying on row order surviving every operation above.
    frame = frame.with_row_index("_pair_ix")

    # Join on the pair frame's own id column names, so the identifying columns
    # survive into the output rather than being renamed away and dropped.
    frame = _attach_record_columns(
        frame,
        s1.select(
            pl.col("entity_id").alias(s1_id_column),
            pl.col(s1_name_column).fill_null("").alias("name_norm"),
            pl.col(s1_addr_column).fill_null("").alias("addr_norm"),
            pl.col("country").fill_null("").alias("country"),
        ),
        id_column=s1_id_column,
        prefix="s1",
    )

    frame = _attach_record_columns(
        frame,
        vendors.select(
            pl.col("entity_id").alias(vendor_id_column),
            pl.col(vendor_name_column).fill_null("").alias("name_norm"),
            pl.col(vendor_addr_column).fill_null("").alias("addr_norm"),
            pl.col("country").fill_null("").alias("country"),
            pl.col("source").fill_null(0).alias("source"),
        ),
        id_column=vendor_id_column,
        prefix="v",
        passthrough=("source",),
    )

    frame = frame.with_columns(
        (pl.col("s1_country") + "|" + pl.col("v_country")).alias("country_pair"),
        pl.col("s1_num").alias("_s1_num"),
        pl.col("v_num").alias("_v_num"),
    )

    # -- name block ----------------------------------------------------------
    frame = frame.with_columns(
        (pl.col("s1_name") == pl.col("v_name"))
        .cast(pl.Float32)
        .alias("name_exact"),
        (_alnum_name_expr(pl.col("s1_name")) == _alnum_name_expr(pl.col("v_name")))
        .cast(pl.Float32)
        .alias("name_alnum_exact"),
        (_sorted_name_expr(pl.col("s1_name")) == _sorted_name_expr(pl.col("v_name")))
        .cast(pl.Float32)
        .alias("name_sorted_exact"),
        prefix_overlap(pl.col("s1_name"), pl.col("v_name")).cast(pl.Float32)
        .alias("name_prefix_overlap"),
        (pl.col("s1_name").str.len_chars() - pl.col("v_name").str.len_chars())
        .abs()
        .cast(pl.Float32)
        .alias("name_len_abs_diff"),
        (
            pl.col("s1_name").str.len_chars().cast(pl.Float32)
            / pl.col("v_name").str.len_chars().clip(lower_bound=1).cast(pl.Float32)
        ).alias("name_len_ratio"),
    )
    frame = frame.join(
        _overlap_block(
            frame, left="s1_name", right="v_name", prefix="name",
            id_column="_pair_ix",
        ),
        on="_pair_ix",
        how="left",
    )

    # -- address block -------------------------------------------------------
    frame = frame.with_columns(
        (pl.col("s1_addr") == pl.col("v_addr")).cast(pl.Float32).alias("addr_exact"),
        prefix_overlap(pl.col("s1_addr"), pl.col("v_addr")).cast(pl.Float32)
        .alias("addr_prefix_overlap"),
        (pl.col("s1_addr").str.len_chars() - pl.col("v_addr").str.len_chars())
        .abs()
        .cast(pl.Float32)
        .alias("addr_len_abs_diff"),
        (pl.col("s1_addr").str.len_chars() > 0).cast(pl.Float32).alias("has_s1_addr"),
        (pl.col("v_addr").str.len_chars() > 0).cast(pl.Float32).alias("has_v_addr"),
    )
    frame = frame.with_columns(
        (pl.min_horizontal(pl.col("has_s1_addr"), pl.col("has_v_addr")))
        .alias("both_addr_present"),
        (
            # One side has an address and the other does not. Not a contradiction
            # on its own -- the vendor pool is 3.3% address-less -- but the
            # model should be able to weigh it explicitly rather than infer it
            # from two separate zero features.
            (pl.max_horizontal(pl.col("has_s1_addr"), pl.col("has_v_addr"))
             - pl.min_horizontal(pl.col("has_s1_addr"), pl.col("has_v_addr")))
        ).alias("addr_conflict"),
    )
    frame = frame.join(
        _overlap_block(
            frame, left="s1_addr", right="v_addr", prefix="addr",
            id_column="_pair_ix",
        ),
        on="_pair_ix",
        how="left",
    )

    # -- numeric signature ---------------------------------------------------
    num_block = frame.select(
        pl.col("_pair_ix"),
        (pl.col("_s1_num") == pl.col("_v_num"))
        .cast(pl.Float32)
        .alias("num_sig_exact"),
    ).join(
        frame.select(
            pl.col("_pair_ix"),
            token_overlap_features(
                clean_token_list("_s1_num"), clean_token_list("_v_num")
            ).alias("_numov"),
        ),
        on="_pair_ix",
        how="left",
    ).select(
        pl.col("_pair_ix"),
        "num_sig_exact",
        pl.col("_numov").struct.field("jaccard").cast(pl.Float32)
        .alias("num_sig_jaccard"),
        pl.col("_numov").struct.field("overlap").cast(pl.Float32)
        .alias("num_sig_overlap"),
    )
    frame = frame.join(num_block, on="_pair_ix", how="left")

    # -- cross-field ---------------------------------------------------------
    frame = frame.with_columns(
        (
            pl.when(pl.col("v_addr").str.len_chars() > 0)
            .then(pl.col("s1_name").str.contains(pl.col("v_addr"), literal=True))
            .otherwise(False)
        )
        .cast(pl.Float32)
        .alias("name_in_addr"),
        (
            pl.when(pl.col("s1_addr").str.len_chars() > 0)
            .then(pl.col("v_addr").str.contains(pl.col("s1_addr"), literal=True))
            .otherwise(False)
        )
        .cast(pl.Float32)
        .alias("addr_in_name"),
    )
    cross = (
        frame.select(pl.col("_pair_ix"), pl.col("s1_name"), pl.col("v_addr"))
        .with_columns(
            clean_token_list("s1_name").alias("_name_toks"),
            clean_token_list("v_addr").alias("_addr_toks"),
        )
        .select(
            pl.col("_pair_ix"),
            token_overlap_features(
                pl.col("_name_toks"), pl.col("_addr_toks")
            ).alias("_x"),
        )
        .select(
            pl.col("_pair_ix"),
            pl.col("_x")
            .struct.field("coverage")
            .cast(pl.Float32)
            .alias("name_token_addr_coverage"),
        )
    )
    frame = frame.join(cross, on="_pair_ix", how="left")

    # -- retrieval evidence --------------------------------------------------
    frame = frame.with_columns(
        (pl.col("s1_country") == pl.col("v_country"))
        .cast(pl.Float32)
        .alias("same_country"),
        (pl.col("v_source") == 3).cast(pl.Float32).alias("source_is_s3"),
        pl.col("strategies").list.len().cast(pl.Float32).alias("n_strategies"),
        pl.col("n_keys").cast(pl.Float32).alias("n_keys"),
        pl.col("cand_count_s1").cast(pl.Float32).alias("cand_count_s1"),
        pl.col("cand_count_vendor").cast(pl.Float32).alias("cand_count_vendor"),
        pl.col("strategy_rank").cast(pl.Float32).alias("strategy_rank"),
        (
            pl.min_horizontal(pl.col("strategy_rank"), pl.lit(1, dtype=pl.Int32))
            == 1
        )
        .cast(pl.Float32)
        .alias("is_top1_by_key_count"),
        pl.col("s1_name_key_df").cast(pl.Float32).alias("s1_name_key_df"),
    )

    # -- record shape --------------------------------------------------------
    frame = frame.with_columns(
        pl.col("s1_name").str.len_chars().cast(pl.Float32).alias("s1_name_len"),
        pl.col("v_name").str.len_chars().cast(pl.Float32).alias("v_name_len"),
        pl.col("s1_addr").str.len_chars().cast(pl.Float32).alias("s1_addr_len"),
        pl.col("v_addr").str.len_chars().cast(pl.Float32).alias("v_addr_len"),
        pl.col("name_n_left").cast(pl.Float32).alias("s1_n_tokens"),
        pl.col("name_n_right").cast(pl.Float32).alias("v_n_tokens"),
    )

    # -- rarity blocks (explode + self-join; see _idf_overlap_block) ----------
    for prefix, left, right in (
        ("name", "s1_name", "v_name"),
        ("addr", "s1_addr", "v_addr"),
    ):
        frame = frame.join(
            _idf_overlap_block(
                frame, rarity, left=left, right=right,
                id_column="_pair_ix", prefix=prefix,
            ),
            on="_pair_ix",
            how="left",
        )

    # -- fuzzy ratios (rapidfuzz, C++, all cores) ----------------------------
    fuzzy: dict[str, np.ndarray] = {}
    for field, left, right in (
        ("name", "s1_name", "v_name"),
        ("addr", "s1_addr", "v_addr"),
    ):
        fuzzy.update(
            batch_fuzzy_ratios(
                frame[left].to_numpy(),
                frame[right].to_numpy(),
                scorers=fuzzy_scorers,
                workers=fuzzy_workers,
                prefix=f"fz_{field}_",
            )
        )
    frame = frame.with_columns(
        [pl.Series(name, values) for name, values in fuzzy.items()]
    )

    # -- integrity check -----------------------------------------------------
    absent = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if absent:
        raise ValueError(
            f"build_pair_features did not produce declared feature column(s) "
            f"{absent}. FEATURE_COLUMNS is the contract the model is fitted "
            f"against; update both together or the matrix will be misaligned."
        )

    keep = [*PAIR_KEY_COLUMNS, *CATEGORICAL_COLUMNS, *FEATURE_COLUMNS]
    return frame.select(keep)
