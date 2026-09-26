"""Vectorised string-similarity primitives for candidate pairs.

Layer: features
See AGENTS.md §11 (Features), DOCS/feature_dictionary.md

Why this module exists
----------------------
Every pairwise feature in :mod:`team_diamond.features.pairs` bottoms out in one
of a handful of string comparisons, applied to tens of millions of pairs.
Two decisions follow from that, and both are load-bearing:

**Several ratios, not one blended score.** Jaccard on token sets ignores
multiplicity, so "Acme Acme Ltd" and "Acme Ltd" score 1.0. Containment ignores
length entirely, so a short name inside a long one also scores 1.0. These
disagree in exactly the informative cases, so :func:`token_overlap_features`
reports Jaccard, Dice, and containment side by side and lets the classifier
decide. Collapsing them into one number here would destroy that choice before
the model ever saw it.

**Fuzzy ratios come from rapidfuzz, in C++, over numpy arrays.** An earlier
draft of this file hand-rolled character n-grams in Python. That was the wrong
call twice over: it was far slower than ``rapidfuzz.process.cdist``, and it
reimplemented a solved problem worse. The hand-rolled code was deleted rather
than kept "for reference" — see :func:`batch_fuzzy_ratios`, which does the same
job across all cores with a dtype-controlled output buffer.

**Similarity is not confidence** (AGENTS.md §11). Everything here measures
*similarity*. Discriminative power comes from weighting by rarity, which is
:mod:`team_diamond.features.idf`'s job, and fold-safety is enforced there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import polars as pl

__all__ = [
    "FUZZY_SCORERS",
    "batch_fuzzy_ratios",
    "prefix_overlap",
    "token_overlap_features",
]

#: The fuzzy scorers computed for every pair, and the feature-name prefix each
#: one produces. Chosen because they fail differently:
#:
#: ``ratio``
#:     Whole-string similarity. Sensitive to insertions and reorderings.
#: ``partial_ratio``
#:     Best alignment of the shorter string inside the longer. Recovers
#:     truncation and "the extra word" noise, which is common here.
#: ``token_set_ratio``
#:     Ignores everything outside the shared token set. Recovers added legal
#:     suffixes ("Ltd", "Inc", "Pty Ltd") that carry no identifying information.
#: ``token_sort_ratio``
#:     Order-insensitive whole-string comparison. Recovers word reordering.
FUZZY_SCORERS: Final[tuple[str, ...]] = (
    "ratio",
    "partial_ratio",
    "token_set_ratio",
    "token_sort_ratio",
)

#: Rows per chunk in :func:`batch_fuzzy_ratios`. Sized so the temporary uint8
#: output buffer stays around 100 MB: large enough to amortise call overhead,
#: small enough that peak memory does not scale with the pair count.
_CHUNK_ROWS: Final[int] = 1_000_000


def _empty_list() -> pl.Expr:
    """An empty ``List(Utf8)`` literal, for null-safe fill."""
    return pl.lit([]).cast(pl.List(pl.Utf8))


def clean_token_list(column: str) -> pl.Expr:
    """Split a normalised column into tokens, dropping empties.

    The emptiness handling here is not cosmetic. ``"".str.split(" ")`` returns
    ``[""]`` -- a one-element list, not an empty list -- and ``"a  b"`` returns
    ``["a", "", "b"]``. Feeding either straight into a set-overlap ratio
    produces a *perfect* score for two records that share no evidence at all::

        union([""], [""]) == [""],  intersection == [""],  so Jaccard == 1.0

    Since 3.3-3.4% of S2 and S3 records have no address (measured, E004), that
    single line would hand a maximal-similarity feature to roughly one pair in
    thirty, none of it informative. Entity-level $F_{0.5}$ weights precision
    above recall, so a feature that manufactures perfect agreement between two
    empty fields is one of the most expensive mistakes available here.

    Filtering to non-empty tokens makes ``""`` produce a genuinely empty list,
    which the ratio guards then score as 0.0.
    """
    text = pl.col(column).fill_null("")
    return (
        pl.when(text == "")
        .then(_empty_list())
        .otherwise(
            text.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        )
    )


def token_split_expr(column: str) -> pl.Expr:
    """Split a normalised, space-separated column into a list of tokens.

    Args:
        column: Name of a column holding space-separated normalised text.

    Returns:
        A list-of-string expression, one element per non-empty token.
    """
    return clean_token_list(column)


def _drop_empty_tokens(tokens: pl.Expr) -> pl.Expr:
    """Strip null and empty-string entries from a list-of-tokens expression.

    Applied inside :func:`token_overlap_features` so a caller that passes a raw
    ``str.split(" ")`` still gets correct numbers. The empty-token bug described
    in :func:`clean_token_list` is silent, produces entirely plausible-looking
    output, and is much cheaper to make impossible here than to notice later in
    a feature-importance plot.
    """
    return tokens.fill_null(_empty_list()).list.eval(
        pl.element().filter(pl.element().is_not_null() & (pl.element() != ""))
    )


def token_overlap_features(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    """Token-set overlap statistics between two list-of-token expressions.

    Args:
        left: List-of-string expression for the first record.
        right: List-of-string expression for the second record.

    Returns:
        A struct expression with fields ``jaccard``, ``dice``, ``containment``,
        ``overlap``, ``left_only``, ``right_only``, ``n_left``, ``n_right``.

    Notes:
        Ratios are guarded against a zero denominator and return ``0.0``. That
        is the right default rather than ``1.0``: two records with no usable
        tokens share *no* evidence, and scoring that as a perfect match is
        precisely the false positive that entity-level $F_{0.5}$ punishes.
    """
    lhs = _drop_empty_tokens(left)
    rhs = _drop_empty_tokens(right)

    intersection = lhs.list.set_intersection(rhs).list.len()
    union = lhs.list.set_union(rhs).list.len()
    n_left = lhs.list.len()
    n_right = rhs.list.len()
    smaller = pl.min_horizontal(n_left, n_right)
    bigger = pl.max_horizontal(n_left, n_right)

    return pl.struct(
        [
            pl.when(union > 0)
            .then(intersection.cast(pl.Float64) / union)
            .otherwise(0.0)
            .alias("jaccard"),
            pl.when((n_left + n_right) > 0)
            .then(2.0 * intersection / (n_left + n_right))
            .otherwise(0.0)
            .alias("dice"),
            pl.when(smaller > 0)
            .then(intersection.cast(pl.Float64) / smaller)
            .otherwise(0.0)
            .alias("containment"),
            pl.when(bigger > 0)
            .then(intersection.cast(pl.Float64) / bigger)
            .otherwise(0.0)
            .alias("coverage"),
            intersection.alias("overlap"),
            (n_left - intersection).alias("left_only"),
            (n_right - intersection).alias("right_only"),
            n_left.alias("n_left"),
            n_right.alias("n_right"),
        ]
    )


def prefix_overlap(left: pl.Expr, right: pl.Expr, *, depth: int = 4) -> pl.Expr:
    """Longest-common-prefix length between two strings, normalised by ``depth``.

    Catches a class of noise that token overlap alone misses: truncation at the
    end of a field ("123 Main St" vs "123 Main St Ste 400"). A *leading* match
    is separately meaningful because businesses differ far more in their
    suffixes than in their beginnings.

    Args:
        left: String expression for the first record.
        right: String expression for the second record.
        depth: Maximum prefix length considered, and the denominator.

    Returns:
        A float in ``[0, 1]``; ``0.0`` when either side is empty.

    Notes:
        The denominator is ``depth``, **not** the shorter string's length.
        Normalising by the shorter string makes the feature useless on short
        names: "abc" against "abcdef" shares a three-character prefix, which is
        almost no evidence, but would score ``3/3 = 1.0`` -- a perfect match.
        Dividing by the fixed depth scores it ``3/4``, which is what it is
        worth. The trade-off is that two genuinely identical 2-character names
        also cap at ``2/4``, so ``depth`` should be chosen with the shortest
        meaningful business name in mind rather than the longest.
    """
    lhs = left.fill_null("")
    rhs = right.fill_null("")
    non_empty = pl.min_horizontal(lhs.str.len_chars(), rhs.str.len_chars())

    # Longest matching prefix, found by testing progressively longer slices.
    # `depth` is a small constant, so this is a fixed number of vectorised
    # comparisons rather than a loop over rows.
    best = pl.lit(0, dtype=pl.Int32)
    for width in range(1, depth + 1):
        best = (
            pl.when(lhs.str.slice(0, width) == rhs.str.slice(0, width))
            .then(pl.lit(width, dtype=pl.Int32))
            .otherwise(best)
        )

    return (
        pl.when(non_empty > 0)
        .then(best.cast(pl.Float64) / pl.lit(float(depth)))
        .otherwise(0.0)
    )


def batch_fuzzy_ratios(
    left: Sequence[str] | np.ndarray,
    right: Sequence[str] | np.ndarray,
    *,
    scorers: Sequence[str] = FUZZY_SCORERS,
    workers: int = -1,
    prefix: str = "fz_",
) -> dict[str, np.ndarray]:
    """Compute several rapidfuzz ratios for aligned string arrays.

    Args:
        left: Array of first strings, length ``n``.
        right: Array of second strings, same length ``n``.
        scorers: Subset of :data:`FUZZY_SCORERS`.
        workers: rapidfuzz worker count; ``-1`` uses every core.
        prefix: Feature-name prefix, so the caller can namespace the result.

    Returns:
        ``{f"{prefix}{scorer}": float32 array of shape (n,)}`` in [0, 1].
        Values are divided by 100 so every similarity feature in the matrix is
        on the same [0, 1] scale, which matters because a gradient-boosted tree
        splits on absolute values and a 0-100 column would dominate the
        thresholds learned for 0-1 columns.

    Raises:
        ValueError: If the arrays differ in length, or a scorer is unknown.
    """
    from rapidfuzz import fuzz, process

    unknown = [s for s in scorers if s not in FUZZY_SCORERS]
    if unknown:
        raise ValueError(f"unknown scorers {unknown}; expected a subset of {FUZZY_SCORERS}")

    left_arr = np.asarray(left, dtype=object)
    right_arr = np.asarray(right, dtype=object)
    if left_arr.shape[0] != right_arr.shape[0]:
        raise ValueError(
            f"left has {left_arr.shape[0]} entries but right has {right_arr.shape[0]}; "
            f"these arrays are index-aligned and must be the same length"
        )

    out: dict[str, np.ndarray] = {}
    for scorer in scorers:
        fn = {
            "ratio": fuzz.ratio,
            "partial_ratio": fuzz.partial_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
            "token_sort_ratio": fuzz.token_sort_ratio,
        }[scorer]
        # Chunked so the uint8 result buffer is bounded regardless of n. A
        # 35M-pair run would otherwise allocate 35 MB per scorer, which is
        # survivable, but the intermediate score matrix inside cdist is larger
        # and is not bounded by us.
        pieces: list[np.ndarray] = []
        for start in range(0, left_arr.shape[0], _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, left_arr.shape[0])
            block = process.cdist(
                left_arr[start:stop],
                right_arr[start:stop],
                scorer=fn,
                dtype=np.uint8,
                workers=workers,
            )
            pieces.append(np.diagonal(block).astype(np.float32) / np.float32(100.0))
        out[f"{prefix}{scorer}"] = (
            np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        )
    return out
