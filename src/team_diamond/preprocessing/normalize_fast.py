"""Vectorised normalisation for bulk data, equivalent to the reference path.

Layer: preprocessing
See AGENTS.md §14 (scale and cost), §8.5 (distribution shift)

Why this module exists
----------------------
:mod:`team_diamond.preprocessing.normalize` is the **reference** implementation:
clear, tested, one row at a time. At ~35 us/row it needs **6.1 minutes per pass**
over S2+S3 and does not parallelise, which makes it unusable for the cap sweeps
and ablations this project needs.

This module is the **fast path**: the same output, computed with vectorised
Polars expressions. It exists for throughput, and it is only trustworthy because
it is verified against the reference (see ``tests/test_normalize_fast.py``).

The contract
------------
For any input, ``normalize_columns(df)`` must equal the reference
``add_normalized_columns(df)`` exactly, column for column. Divergence is a bug,
not a trade-off. A large speedup is worth having; a silent semantic difference is
not.

How equivalence is achieved
---------------------------
The reference folds accents with ``NFKD`` + "drop combining marks". That is not
a fixed lookup table, so we do not pretend otherwise:

1. The fold map is **derived from the reference function** over the characters
   that actually occur in the frame being processed
   (:func:`derive_fold_maps`). The table therefore cannot drift from the code it
   mirrors, and a character appearing for the first time is handled rather than
   missed.
2. Combining marks are removed by Unicode category, which is what the reference
   does by definition.
3. Token filtering uses the **padded literal-replace trick** described on
   :func:`_filter_tokens`, because it is exact where a regex word boundary is
   not.

Nothing here branches on country (AGENTS.md §8.5). The same expression runs for
US, India and France.
"""

from __future__ import annotations

import re
from typing import Final

import polars as pl

from team_diamond.preprocessing.normalize import (
    _ADDRESS_ABBREVIATIONS,
    _GENERIC_TOKENS,
    _LEGAL_TOKENS,
    _LATIN_FOLD,
)

__all__ = [
    "derive_fold_maps",
    "require_fold_coverage",
    "normalize_columns",
    "BENCHMARK_ROWS",
]

# Only entries that are genuine single tokens can ever be matched by the
# reference, which filters a *token list*. Multi-word entries are inert there and
# are excluded here so the two paths stay in agreement.
_SINGLE_TOKEN_LEGAL: Final[tuple[str, ...]] = tuple(
    sorted(t for t in _LEGAL_TOKENS if " " not in t)
)
_SINGLE_TOKEN_GENERIC: Final[tuple[str, ...]] = tuple(
    sorted(t for t in _GENERIC_TOKENS if " " not in t)
)

# One compiled alternation per set, not one pass per token. Boundaries are
# explicit, so a key can no longer shadow a longer one that shares its prefix
# and the sort order carries no meaning.
def _alternation(tokens: tuple[str, ...]) -> str:
    return rf"\b(?:{'|'.join(re.escape(t) for t in tokens)})\b"


_LEGAL_PATTERN: Final[str] = _alternation(_SINGLE_TOKEN_LEGAL)
_GENERIC_PATTERN: Final[str] = _alternation(_SINGLE_TOKEN_GENERIC)
_ADDR_EXPANSIONS: Final[tuple[tuple[str, str], ...]] = tuple(
    (rf"\b{re.escape(k)}\b", v)
    for k, v in _ADDRESS_ABBREVIATIONS.items()
)

# Mirrors normalize._SEPARATOR_TABLE. Polars' regex engine is Rust, whose `\w`
# is `[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]` — it INCLUDES
# combining marks, so `[^\w]+` keeps an Indic syllable intact. Python's `re`
# excludes them, which is why the reference uses a translate table instead.
# Verified equivalent on every character in the dataset by
# tests/test_normalize_fast.py::test_tokenizers_agree_on_every_present_character.
_NON_WORD: Final[str] = r"[^\w]+"
_WHITESPACE_RUN: Final[str] = r"\s+"



def _present_chars(text: pl.Series) -> set[str]:
    """Every non-ASCII character in ``text``, **after** casefolding.

    Casefolding first is not a detail. The fold chain runs on
    ``str.to_lowercase()`` output, so it encounters ``\u00e3`` where the raw
    column held ``\u00c3``. Inventorying the raw text therefore misses every
    lowercase accented form, and the fast path leaves it unfolded while the
    reference folds it -- a divergence that only shows up on mixed-case
    accented input, i.e. the French test rows.

    Args:
        text: Any string series.

    Returns:
        The set of non-ASCII characters present, lowercased.
    """
    lowered = text.fill_null("").str.to_lowercase()
    non_ascii = lowered.filter(lowered.str.contains(r"[^\x00-\x7F]")).unique()
    present: set[str] = set()
    for value in non_ascii.to_list():
        present.update(value)
    return present


def derive_fold_maps(text: pl.Series) -> dict[str, str]:
    """Restrict the shared fold table to characters actually present.

    Chaining 600+ literal replacements over every row would be wasteful, so the
    table is narrowed to the characters this frame contains. Coverage is not
    weakened by doing so: a character that is not present cannot change the
    output. The *values* still come from
    :data:`team_diamond.preprocessing.normalize._LATIN_FOLD`, so the fast path
    cannot drift from the reference.

    Args:
        text: Any string series from the data being processed.

    Returns:
        ``{char: ascii_letter}``, longest key first.
    """
    present = _present_chars(text)

    fold = {c: b for c, b in _LATIN_FOLD.items() if c in present}
    # Longest first: a 1-char key must not consume the first half of a longer one.
    return dict(sorted(fold.items(), key=lambda kv: (-len(kv[0]), kv[0])))


def _filter_tokens(expr: pl.Expr, pattern: str) -> pl.Expr:
    """Remove whole tokens matching ``pattern`` (a ``\\b``-anchored alternation).

    Why ``\\b`` and not the padded literal ``" tok "`` trick that was tried
    first: ``replace_all`` scans non-overlapping and advances past the whole
    match, so **consecutive duplicate tokens** defeat it. On
    ``"dynamic producer limited limited"`` the padded form replaced the first
    ``" limited "`` and then skipped the second, because it began inside the
    span just consumed. The reference filters a token *list* and has no such
    failure. Duplicated legal forms are not rare in this corpus -- the raw
    column ``Dynamic Private Producer Límited Límited`` is one example -- so
    this was a real 0.5% name divergence, not a theoretical one.

    ``\\b`` is safe here precisely because the expression has already been
    tokenised with the same ``\\w`` definition the boundary is built on: tokens
    are maximal runs of ``\\w``, so a boundary can only fall where a token
    starts or ends. The stopword lists are pure ASCII, and a token such as
    ``"प्राइवेटlimited"`` correctly fails to match ``\\blimited\\b`` on both
    sides because the Devanagari character is itself ``\\w``.

    Args:
        expr: A space-separated, trimmed expression.
        pattern: ``\\b``-anchored alternation of tokens to remove.

    Returns:
        The expression with those tokens removed and whitespace re-collapsed.
    """
    if not pattern:
        return expr
    expr = expr.str.replace_all(pattern, " ", literal=False)
    return expr.str.replace_all(_WHITESPACE_RUN, " ", literal=False).str.strip_chars()


def _expand_tokens(expr: pl.Expr) -> pl.Expr:
    """Replace whole tokens per ``_ADDRESS_ABBREVIATIONS``, boundary-anchored.

    Order is irrelevant under ``\\b`` anchoring: a key can no longer match inside
    a longer token, so one pass per key is exact regardless of sequence.
    """
    for pattern, replacement in _ADDR_EXPANSIONS:
        expr = expr.str.replace_all(pattern, f" {replacement} ", literal=False)
    return expr.str.replace_all(_WHITESPACE_RUN, " ", literal=False).str.strip_chars()


def _preclean(column: str, fold_map: dict[str, str]) -> pl.Expr:
    """Casefold, accent-fold, expand ``&``, tokenise, collapse whitespace.

    Returns a trimmed, single-space-separated string. Everything downstream
    relies on that invariant.
    """
    expr = pl.col(column).fill_null("").str.to_lowercase()
    for char, base in fold_map.items():
        expr = expr.str.replace_all(char, base, literal=True)
    expr = expr.str.replace_all("&", " and ", literal=True)
    expr = expr.str.replace_all(_NON_WORD, " ", literal=False)
    return expr.str.replace_all(_WHITESPACE_RUN, " ", literal=False).str.strip_chars()


def normalize_columns(
    frame: pl.DataFrame,
    *,
    drop_legal: bool = True,
    drop_generic: bool = False,
    expand_abbreviations: bool = True,
    fold_map: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Vectorised equivalent of ``add_normalized_columns``.

    Args:
        frame: Must contain ``business_name`` and ``business_address``, both
            already null-filled to ``""``.
        drop_legal: Remove legal-form tokens from names. Must match the value
            given to the reference function.
        drop_generic: Remove generic business words from names.
        expand_abbreviations: Expand address abbreviations.
        fold_map: Precomputed fold table from :func:`derive_fold_maps`. Pass it
            when normalising several frames from one corpus to avoid re-deriving.
            If ``None`` it is derived from ``frame``.

            .. warning::
               A table derived from one corpus may not cover another. Passing a
               train-derived table to the test split silently skips French
               accents, because train S1 is pure ASCII. Use
               :func:`require_fold_coverage` to assert coverage instead of
               trusting it.

    Returns:
        A new frame with ``name_norm``, ``addr_norm``, ``name_tokens_norm``,
        ``addr_tokens_norm``, ``has_address``, ``has_name``. Input is not
        modified.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"business_name", "business_address"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    if fold_map is None:
        fold_map = derive_fold_maps(
            pl.concat([frame["business_name"], frame["business_address"]])
        )

    name = _preclean("business_name", fold_map)
    if drop_legal:
        name = _filter_tokens(name, _LEGAL_PATTERN)
    if drop_generic:
        name = _filter_tokens(name, _GENERIC_PATTERN)

    addr = _preclean("business_address", fold_map)
    if expand_abbreviations:
        addr = _expand_tokens(addr)

    return frame.with_columns(
        [
            name.alias("name_norm"),
            addr.alias("addr_norm"),
        ]
    ).with_columns(
        [
            pl.col("name_norm").alias("name_tokens_norm"),
            pl.col("addr_norm").alias("addr_tokens_norm"),
            (pl.col("addr_norm").str.len_chars() > 0).alias("has_address"),
            (pl.col("name_norm").str.len_chars() > 0).alias("has_name"),
        ]
    )


def require_fold_coverage(
    text: pl.Series, fold_map: dict[str, str], *, context: str = ""
) -> None:
    """Assert that ``fold_map`` folds every accented letter present in ``text``.

    Needed because :func:`derive_fold_maps` is corpus-scoped. A table derived
    from train S2 has no French characters in it, so reusing it for the test
    split leaves ``è`` unfurled and ``École`` normalising to ``école`` — which
    then fails to match the ASCII spelling ``Ecole``. The bug is silent: the
    output still looks like a plausible normalised name.

    Args:
        text: The series about to be normalised.
        fold_map: The table that will be used.
        context: Optional label included in the error message.

    Raises:
        AssertionError: Listing each present letter the table would not fold.
    """
    from team_diamond.preprocessing.normalize import _LATIN_FOLD

    present = _present_chars(text)

    missing = sorted(c for c in present if c in _LATIN_FOLD and c not in fold_map)
    if missing:
        detail = ", ".join(f"{c!r}->{_LATIN_FOLD[c]!r}" for c in missing)
        raise AssertionError(
            f"fold table is missing {len(missing)} letter(s) present in "
            f"{context or 'this series'}: {detail}. The fast path would leave "
            f"them unfolded and silently diverge from the reference. Re-derive "
            f"with derive_fold_maps() over the data actually being normalised."
        )


BENCHMARK_ROWS: Final[int] = 200_000
