"""Blocking keys: the mechanism that makes candidate generation tractable.

Layer: retrieval
See AGENTS.md §10 (Retrieval rules), §14 (Scale and cost)

Why blocking is mandatory
-------------------------
Exhaustive comparison is 2.2M x 10.3M = 2.3e13 pairs on train. A blocking key
reduces that to a join over shared keys. The cost of a key is that it can miss
a true match, so **candidate recall is the ceiling on entity-level $F_{0.5}$**
and must be measured per key, not assumed.

This module only *produces keys*. It does not rank, score, cap, or decide. The
decision of which keys to union, and what to cap them at, belongs to
:mod:`team_diamond.retrieval.candidates` and to experiment (AGENTS.md §10:
"caps are experimental parameters, never constants").

The contract for every strategy
-------------------------------
``build_*_keys`` returns a **long** frame with exactly two columns:

``row_id``
    The row's identifier, passed straight through from the input. Never cast,
    normalised, or regenerated (AGENTS.md §8.3).
``key``
    A non-empty string. Empty and null keys are dropped, because an empty key
    would match every other empty key and produce a quadratic block.

Long format is deliberate. It makes the union of strategies a single
``concat`` + ``unique``, it makes a document-frequency filter a single
``group_by``, and it keeps every strategy composable without any of them
knowing about the others.

Language-agnostic by construction
---------------------------------
No strategy branches on country (AGENTS.md §8.5). The test split contains a
jurisdiction absent from train, so a country-conditional key would be an
untested branch on 14.98% of the test set. Everything here is either a Unicode
property or a structural property of the normalised text.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = [
    "KEY_COLUMNS",
    "build_exact_name_keys",
    "build_alnum_name_keys",
    "build_sorted_name_keys",
    "build_name_token_keys",
    "build_address_exact_keys",
    "build_address_token_keys",
    "build_numeric_keys",
    "document_frequency",
    "filter_by_document_frequency",
    "tokenise",
]

#: The exact column contract every key builder returns.
KEY_COLUMNS: Final[tuple[str, str]] = ("row_id", "key")

# Longest-first alternation of numeric runs. Kept as a regex rather than a
# token filter so that "12b" yields "12" and "f-7" yields "7": abbreviations and
# unit suffixes are glued to the number in this corpus often enough that a
# whole-token check would discard real house numbers and postcodes.
# A key must be at least this many characters to be worth indexing. Below it the
# block degenerates: measured on train, the single most frequent 2-character
# name token occurs in a large enough share of S2 that indexing it produces a
# block too large to be useful. The value is a PROPOSAL pending the df sweep in
# experiments/scripts/measure_blocking_recall.py.
MIN_KEY_LENGTH: Final[int] = 3


def tokenise(normalised: pl.Expr) -> pl.Expr:
    """Turn a space-separated normalised string into a list of tokens.

    The result is list-valued; the caller explodes it at frame level (see
    :func:`_long` for why it cannot be done inside the expression).

    Args:
        normalised: Expression yielding a single-space-separated string.

    Returns:
        A list-of-string expression, one element per token.
    """
    return normalised.str.split(" ")


def _long(
    frame: pl.DataFrame, id_column: str, keys: pl.Expr, *, explode: bool = False
) -> pl.DataFrame:
    """Assemble the long key frame, dropping empty and null keys.

    The explode happens at **frame** level, not inside the expression. Polars
    cannot ``explode`` a column that an expression is still computing, so
    ``pl.col(c).str.split(" ").explode()`` inside a ``select`` fails with
    "unable to find column". Materialising the list column first and then
    exploding by name is the only form that works.

    Dropping empty keys is not cosmetic. Two records with no usable name would
    both carry the empty key and would therefore be declared candidates for each
    other. With millions of address-less records that single empty key becomes
    the largest block in the index, and it connects records that share no
    evidence at all.

    Args:
        frame: Source rows.
        id_column: Identifier column, passed through unchanged.
        keys: Expression producing the key column, list-valued when ``explode``.
        explode: Whether ``keys`` yields a list per row.

    Returns:
        Long frame of ``(row_id, key)``, de-duplicated.
    """
    out = frame.with_columns(keys.alias("key")).select(
        pl.col(id_column).cast(pl.Utf8).alias("row_id"), "key"
    )
    if explode:
        # `empty_as_null=True` is spelled out rather than left to the default:
        # Polars 2.0 flips it to False. Either way the row is dropped by the
        # length filter below (a null yields null, and `filter` keeps only
        # True), so the *outcome* is identical -- but relying on a default that
        # is scheduled to change means a future upgrade silently alters key
        # construction, and a retrieval recall number is not something to
        # rediscover from scratch.
        out = out.explode("key", empty_as_null=True)
    return out.filter(pl.col("key").str.len_chars() > 0).unique()


def build_exact_name_keys(
    frame: pl.DataFrame, *, id_column: str = "entity_id"
) -> pl.DataFrame:
    """One key per record: the whole normalised name.

    Cheapest strategy and the highest-precision one. Recall is limited by name
    noise: any spelling difference defeats it entirely, so it is a component of
    a union and never the whole strategy.

    Args:
        frame: Frame with a ``name_norm`` column.
        id_column: Identifier column, passed through unchanged.

    Returns:
        Long frame of ``(row_id, key)``.
    """
    return _long(frame, id_column, pl.col("name_norm"))


def build_alnum_name_keys(
    frame: pl.DataFrame, *, id_column: str = "entity_id"
) -> pl.DataFrame:
    """One key per record: the name with every non-alphanumeric removed.

    Absorbs punctuation and spacing differences that survive normalisation
    ("A-B-C" vs "A B C" vs "ABC"). Weaker than :func:`build_exact_name_keys` as
    a signal, stronger as a recall net.

    Args:
        frame: Frame with a ``name_norm`` column.
        id_column: Identifier column, passed through unchanged.

    Returns:
        Long frame of ``(row_id, key)``.
    """
    return _long(
        frame,
        id_column,
        pl.col("name_norm").str.replace_all(r"[^\w]+", "", literal=False),
    )


def build_sorted_name_keys(
    frame: pl.DataFrame, *, id_column: str = "entity_id"
) -> pl.DataFrame:
    """One key per record: the name's tokens, de-duplicated and sorted.

    Absorbs word-order differences, which normalisation cannot fix. "Prime
    Money Forex" and "Forex Prime Money" share this key and not
    ``name_norm``. Distinct businesses with the same token multiset also share
    it, so this is a recall net, not evidence.

    Args:
        frame: Frame with a ``name_norm`` column.
        id_column: Identifier column, passed through unchanged.

    Returns:
        Long frame of ``(row_id, key)``.
    """
    return _long(
        frame,
        id_column,
        pl.col("name_norm")
        .str.split(" ")
        .list.unique()
        .list.sort()
        .list.join(" "),
    )


def build_name_token_keys(
    frame: pl.DataFrame, *, id_column: str = "entity_id"
) -> pl.DataFrame:
    """One key per name token.

    The main recall net for token-level noise ("Bakers Delight" vs "Bakers
    Delight Cafe"). Produces very high-cardinality blocks on common tokens, so
    it **must** be paired with
    :func:`~team_diamond.retrieval.keys.filter_by_document_frequency` before it
    is used at scale. The threshold is an experimental parameter, not a
    constant.

    Args:
        frame: Frame with a ``name_norm`` column.
        id_column: Identifier column, passed through unchanged.

    Returns:
        Long frame of ``(row_id, key)``.
    """
    return _long(frame, id_column, tokenise(pl.col("name_norm")), explode=True)


def build_address_exact_keys(
    frame: pl.DataFrame, *, id_column: str = "entity_id"
) -> pl.DataFrame:
    """One key per record: the whole normalised address.

    Args:
        frame: Frame with an ``addr_norm`` column.
        id_column: Identifier column, passed through unchanged.

    Returns:
        Long frame of ``(row_id, key)``.
    """
    return _long(frame, id_column, pl.col("addr_norm"))


def build_address_token_keys(
    frame: pl.DataFrame, *, id_column: str = "entity_id"
) -> pl.DataFrame:
    """One key per address token.

    Street names and city names are the most reusable address components, so
    this recovers matches where the full address string differs in unit, floor,
    or punctuation. Requires a document-frequency filter, as with
    :func:`build_name_token_keys`.

    Args:
        frame: Frame with an ``addr_norm`` column.
        id_column: Identifier column, passed through unchanged.

    Returns:
        Long frame of ``(row_id, key)``.
    """
    return _long(frame, id_column, tokenise(pl.col("addr_norm")), explode=True)


def build_numeric_keys(
    frame: pl.DataFrame, *, id_column: str = "entity_id"
) -> pl.DataFrame:
    """One key per maximal digit run in the address.

    House numbers, unit numbers, and postcodes survive address reformatting far
    better than the surrounding prose. Kept separate from
    :func:`build_address_token_keys` because numeric runs need a different
    document-frequency treatment: a house number is low-cardinality but highly
    discriminative *within* a city, whereas a bare number is meaningless
    without locality.

    Args:
        frame: Frame with an ``addr_norm`` column.
        id_column: Identifier column, passed through unchanged.

    Returns:
        Long frame of ``(row_id, key)``.
    """
    return _long(
        frame,
        id_column,
        pl.col("addr_norm")
        # Blank out every non-digit, which isolates each numeric run into its own
        # whitespace-delimited token. Done on the whole string rather than per
        # token so that "12b" yields "12" and "f-7" yields "7" -- unit suffixes
        # are glued to numbers in this corpus often enough that a whole-token
        # check would discard real house numbers.
        .str.replace_all(r"\D+", " ", literal=False)
        .str.replace_all(r"\s+", " ", literal=False)
        .str.strip_chars()
        .str.split(" "),
        explode=True,
    )


def document_frequency(keys: pl.DataFrame) -> pl.DataFrame:
    """Count how many distinct rows carry each key.

    Args:
        keys: Long key frame from any ``build_*_keys`` function.

    Returns:
        Frame of ``(key, df)``, one row per key, sorted by descending
        frequency so the top offenders are the first rows.
    """
    return (
        keys.group_by("key")
        .agg(pl.col("row_id").n_unique().alias("df"))
        .sort("df", descending=True)
    )


def filter_by_document_frequency(
    keys: pl.DataFrame, *, max_df: int, min_df: int = 1
) -> pl.DataFrame:
    """Keep only keys carried by between ``min_df`` and ``max_df`` rows.

    ``max_df`` removes ubiquitous tokens that would create an unusable block.
    Reporting the recall this costs is the caller's job, not this function's:
    this is a filter and it is silent about what it dropped, so a caller that
    does not measure the cost is violating AGENTS.md §10.

    Args:
        keys: Long key frame.
        max_df: Drop keys carried by more than this many rows.
        min_df: Drop keys carried by fewer than this many rows. A key on a single
            row cannot create a cross-source block and only costs index space.

    Returns:
        The filtered long key frame, with a ``df`` column added.
    """
    counts = document_frequency(keys)
    kept = counts.filter((pl.col("df") <= max_df) & (pl.col("df") >= min_df))
    return keys.join(kept.select("key", "df"), on="key", how="semi")
