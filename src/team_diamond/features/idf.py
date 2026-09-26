"""Token rarity statistics, and the leakage rules that govern them.

Layer: features
See AGENTS.md §11 ("Similarity is not confidence"), §12 (Leakage prevention)

The idea
--------
Two businesses called "Prime Money" and two called "Khan Silks" can both score
a perfect 1.0 Jaccard on their names. Only the second pair is evidence. Token
rarity is what separates them, and it is the single most valuable feature family
in this task -- more so than any string-similarity score.

The leakage hazard
------------------
Document frequency is a **corpus statistic**, and AGENTS.md §12 is explicit that
a global IDF computed over all entities leaks:

    "Any statistic used by the matcher must be built using only information
    available within the corresponding training fold. ... A global IDF computed
    over all entities leaks."

Why it leaks here is worth being concrete about, because it is not obvious.
The *vendor* corpus (S2 + S3) is available unlabelled at inference time, so
computing DF over it is legitimate for inference. But if DF is computed over
S2 + S3 **plus the validation S1 rows**, then the validation rows' own
occurrences inflate or deflate the DF of every token they contain, and a
validation entity whose name token happens to be rare gets an IDF that was
tuned using its own existence. The model then sees a feature distribution at
validation time that it will not see at test time, and validation F0.5 becomes
optimistic by an amount nobody can estimate after the fact.

The rule this module enforces
-----------------------------
:class:`TokenRarity` is always built from an explicit, named corpus:

- **Inference** builds it from the vendor pool (S2 + S3) only. S1 rows never
  contribute, in train or in test. This is available at inference and is
  therefore not leakage.
- **Validation** builds it from the vendor pool of the *training fold only*,
  with the validation S1 rows and their S2/S3 neighbours excluded.

The second is stricter than strictly necessary (the vendor pool is unlabelled,
so including it would not leak labels), and it is deliberately so: it makes the
validation-time feature distribution match the test-time one, which is the
property we actually need. A distribution that differs between validation and
test makes the validation number uninterpretable regardless of whether any
individual label leaked.

Every builder takes the corpus as an argument rather than reaching for a
global. There is no module-level singleton, so "did someone remember to scope
this to the fold?" is answered by the type signature instead of by review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import polars as pl

__all__ = ["IdfTable", "TokenRarity", "build_token_rarity", "DEFAULT_SMOOTHING"]

#: Additive smoothing for IDF. Prevents a token seen in every record from
#: producing a log(0), and keeps the value finite for unseen tokens.
DEFAULT_SMOOTHING: Final[float] = 1.0


#: IDF assigned to a token absent from the corpus. Deliberately larger than any
#: attainable in-corpus value: "this token was never seen" is maximal evidence
#: of rarity, and it must not collapse into the same number as a merely-uncommon
#: token, nor become a null that a tree has to learn to route around.
UNSEEN_IDF: Final[float] = 30.0


@dataclass(frozen=True, slots=True)
class IdfTable:
    """An ``(token, idf)`` lookup, plus the corpus size it was built from.

    Attributes:
        frame: Frame with columns ``token`` (Utf8) and ``idf`` (Float64).
        n_documents: Number of documents the frequencies were counted over.
            Recorded so a feature matrix can be checked for corpus mismatch,
            which is the symptom of a fold-scoping mistake.
    """

    frame: pl.DataFrame
    n_documents: int

    def joined_to(self, frame: pl.DataFrame, *, on: str = "token") -> pl.DataFrame:
        """Attach IDF values to a long frame of tokens.

        Args:
            frame: Long frame with a ``token`` column.
            on: Token column name.

        Returns:
            ``frame`` with an ``idf`` column. Tokens absent from the table get
            :attr:`UNSEEN_IDF` rather than null, because "this token was never
            seen in the corpus" is itself a strong rarity signal and must not
            become a missing value.
        """
        return frame.join(self.frame, on=on, how="left").with_columns(
            pl.col("idf").fill_null(UNSEEN_IDF)
        )

    @property
    def max_idf(self) -> float:
        """Largest IDF in the table, i.e. the rarest token's value."""
        if self.frame.is_empty():
            return 0.0
        return float(self.frame["idf"].max() or 0.0)


#: IDF assigned to a token absent from the corpus. Set to the theoretical
#: maximum of ``log((N+1)/1)`` for a token appearing exactly once, which is the
#: correct limit, rounded up so unseen is always at least as rare as seen-once.
UNSEEN_IDF: Final[float] = 30.0


@dataclass(frozen=True, slots=True)
class TokenRarity:
    """Name-token and address-token rarity, built from one explicit corpus.

    Attributes:
        name: IDF over name tokens.
        address: IDF over address tokens.
        n_documents: Documents the statistics were computed over.
        corpus_label: Human-readable description of the corpus, recorded in the
            experiment registry so a fold-scoping decision is auditable after
            the fact rather than reconstructed from memory.
    """

    name: IdfTable
    address: IdfTable
    n_documents: int
    corpus_label: str

    @classmethod
    def empty(cls, *, label: str = "empty") -> TokenRarity:
        """A rarity object with no data, so feature code needs no null branch."""
        blank = pl.DataFrame(
            schema={"token": pl.Utf8, "idf": pl.Float64}
        )
        return cls(
            name=IdfTable(blank, 0),
            address=IdfTable(blank, 0),
            n_documents=0,
            corpus_label=label,
        )


def _idf_from_counts(
    counts: pl.DataFrame, *, n_documents: int, smoothing: float
) -> IdfTable:
    """Turn a ``(token, df)`` frame into an ``(token, idf)`` table.

    Uses the smoothed ratio ``log((N + s) / (df + s))``. It is finite for every
    ``df`` in ``[1, N]`` and strictly decreasing in ``df``, so the most frequent
    token always has the lowest IDF and no special-casing of the extremes is
    needed.
    """
    if n_documents <= 0 or counts.is_empty():
        return IdfTable(pl.DataFrame(schema={"token": pl.Utf8, "idf": pl.Float64}), 0)
    numerator = math.log(n_documents + smoothing)
    return IdfTable(
        counts.select(
            "token",
            (
                pl.lit(numerator, dtype=pl.Float64)
                - (pl.col("df") + smoothing).log()
            )
            .cast(pl.Float64)
            .alias("idf"),
        ),
        n_documents,
    )


def _document_frequency(
    frame: pl.DataFrame, *, tokens_column: str, id_column: str
) -> pl.DataFrame:
    """Count how many distinct records contain each token.

    Counting **documents** rather than token occurrences matters because of the
    duplicate-token artefact this corpus has: "Prime Prime Money" contains
    "prime" twice. Counting occurrences would give that record a document
    frequency contribution of two, inflating the apparent commonness of every
    token it repeats and biasing IDF downwards exactly for the noisiest names.

    Args:
        frame: Rows with a list-of-tokens column and a unique id column.
        tokens_column: List-of-string column of tokens.
        id_column: Column uniquely identifying a record.

    Returns:
        ``(token, df)`` sorted by descending frequency.
    """
    return (
        frame.select(pl.col(id_column), pl.col(tokens_column))
        .explode(tokens_column)
        .filter(pl.col(tokens_column).is_not_null() & (pl.col(tokens_column) != ""))
        .group_by(tokens_column)
        .agg(pl.col(id_column).n_unique().alias("df"))
        .rename({tokens_column: "token"})
        .sort("df", descending=True)
    )


def build_token_rarity(
    records: pl.DataFrame,
    *,
    id_column: str = "entity_id",
    name_column: str = "name_norm",
    address_column: str = "addr_norm",
    smoothing: float = DEFAULT_SMOOTHING,
    label: str = "corpus",
) -> TokenRarity:
    """Build name and address IDF tables from an explicit set of records.

    Args:
        records: The corpus to count over. **This is the leakage boundary.**
            Pass the vendor pool for inference; pass the training fold's vendor
            pool for validation. Never pass a set that includes the rows whose
            labels the model is being scored on.
        id_column: Unique id, used to count documents rather than token
            occurrences, so a record repeating a token counts once.
        name_column: Normalised name. Must already be tokenised or
            space-separated; it is split on spaces here.
        address_column: Normalised address, split on spaces here.
        smoothing: Additive smoothing for the IDF ratio.
        label: Description recorded in the returned object.

    Returns:
        The two IDF tables and the corpus size they were built from.
    """
    n_documents = records.height
    if n_documents == 0:
        return TokenRarity.empty(label=label)

    missing = [
        c
        for c in (id_column, name_column, address_column)
        if c not in records.columns
    ]
    if missing:
        raise ValueError(
            f"build_token_rarity needs columns {missing}; present: {records.columns}"
        )

    prepared = records.select(
        pl.col(id_column),
        pl.col(name_column).fill_null("").alias("name_tokens"),
        pl.col(address_column).fill_null("").alias("addr_tokens"),
    ).with_columns(
        [
            pl.when(pl.col("name_tokens") == "")
            .then(pl.lit([]).cast(pl.List(pl.Utf8)))
            .otherwise(pl.col("name_tokens").str.split(" "))
            .alias("name_tokens"),
            pl.when(pl.col("addr_tokens") == "")
            .then(pl.lit([]).cast(pl.List(pl.Utf8)))
            .otherwise(pl.col("addr_tokens").str.split(" "))
            .alias("addr_tokens"),
        ]
    )

    name_counts = _document_frequency(
        prepared, tokens_column="name_tokens", id_column=id_column
    )
    addr_counts = _document_frequency(
        prepared, tokens_column="addr_tokens", id_column=id_column
    )

    return TokenRarity(
        name=_idf_from_counts(name_counts, n_documents=n_documents, smoothing=smoothing),
        address=_idf_from_counts(addr_counts, n_documents=n_documents, smoothing=smoothing),
        n_documents=n_documents,
        corpus_label=label,
    )
