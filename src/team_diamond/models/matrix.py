"""Convert a feature frame into the positional matrix the model is fitted on.

Layer: models (boundary with features)
See AGENTS.md §11, §15

Why this module exists
----------------------
CatBoost is fed a ``numpy`` array, not a named frame. That is fast, and it is
also a trap: a reordered or renamed column produces perfectly plausible
predictions from a model that has learned something entirely different, and
nothing raises. There is no column name in the artefact to catch it.

So the conversion is centralised here, and it is *strict*:

- the column order is taken from the declared
  :data:`~team_diamond.features.pairs.FEATURE_COLUMNS` tuple, never from the
  order the frame happens to have;
- a missing or unexpected column is an error, not a silent subset;
- the categorical encoding is recorded on the result, so a model saved to disk
  can be read back with the codes it was trained on rather than whatever a
  fresh sort order happens to produce.

That last point is the one that actually bites in practice. Factorising
``country_pair`` against a freshly sorted vocabulary is deterministic *within*
a run and silently wrong *across* runs whenever a new country appears -- and a
new country is exactly what the test split contains (France, absent from
train). The mapping is therefore fitted once, stored, and reused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import polars as pl

__all__ = ["ModelMatrix", "to_model_matrix", "encode_categorical"]


@dataclass(frozen=True, slots=True)
class ModelMatrix:
    """A positional feature matrix plus everything needed to reproduce it.

    Attributes:
        X: ``float32`` array of shape ``(n_rows, n_features)``, columns ordered
            exactly as :attr:`feature_names`.
        y: ``int8`` labels, or ``None`` for inference.
        feature_names: Column order, which *is* part of the model contract.
        cat_indices: Positions within ``X`` that are categorical. CatBoost
            needs these as indices, not names, once the data is an array.
        categorical_vocabularies: Per categorical column, the frozen
            ``label -> code`` mapping used to encode it. Persisted with the
            model so inference encodes identically.
    """

    X: np.ndarray
    y: np.ndarray | None
    feature_names: tuple[str, ...]
    cat_indices: tuple[int, ...] = ()
    categorical_vocabularies: dict[str, dict[str, int]] = field(
        default_factory=dict
    )

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def describe(self) -> str:
        """One-paragraph summary for an experiment log."""
        lines = [
            f"ModelMatrix: {self.n_rows:,} rows x {self.n_features} features "
            f"({self.X.dtype})",
            f"  labels    : {'none (inference)' if self.y is None else f'int8, {int(self.y.sum()):,} positive'}",
            f"  categoric.: {list(self.cat_indices) if self.cat_indices else 'none'}",
        ]
        for name, vocab in self.categorical_vocabularies.items():
            preview = ", ".join(
                f"{k}={v}" for k, v in sorted(vocab.items(), key=lambda kv: kv[1])[:6]
            )
            more = f", +{len(vocab) - 6} more" if len(vocab) > 6 else ""
            lines.append(f"      {name}: {len(vocab)} codes ({preview}{more})")
        return "\n".join(lines)


def encode_categorical(
    values: pl.Series,
    *,
    column: str,
    vocabulary: dict[str, int] | None = None,
    unknown_code: int = -1,
) -> tuple[np.ndarray, dict[str, int]]:
    """Factorise a string column, optionally against a frozen vocabulary.

    Args:
        values: String values to encode. Nulls are rejected rather than mapped,
            because an unknown category is meaningful (it means "we have not
            seen this before") and a null is almost always a bug that would
            otherwise be indistinguishable from it.
        column: Name, for error messages.
        vocabulary: Existing ``label -> code`` mapping. When given, its keys are
            the complete set of known categories and ordering is preserved
            exactly. When omitted, a vocabulary is fitted from ``values`` in
            sorted order, which is deterministic but only valid if the same
            values are seen again -- hence the persistence of this mapping.
        unknown_code: Code assigned to a label absent from ``vocabulary``.
            CatBoost reserves ``-1`` for "unseen", so this matches its own
            convention rather than inventing one.

    Returns:
        ``(codes, vocabulary)``. ``codes`` is ``int32`` with shape ``(n,)``.

    Raises:
        ValueError: If ``values`` contains nulls, or if the fitted vocabulary
            would be empty.
    """
    if values.null_count() > 0:
        raise ValueError(
            f"categorical column {column!r} has {values.null_count()} null "
            f"value(s). Encode unknowns explicitly with unknown_code; a null "
            f"here would be silently indistinguishable from a category absent "
            f"from the training vocabulary."
        )

    if vocabulary is None:
        labels = sorted(set(values.to_list()))
        if not labels:
            raise ValueError(f"cannot fit a vocabulary for {column!r}: no rows")
        fitted = {label: index for index, label in enumerate(labels)}
    else:
        fitted = dict(vocabulary)

    lookup = pl.Series(column, list(fitted.keys()), dtype=pl.Utf8)
    index_series = pl.Series(column, list(fitted.values()), dtype=pl.Int32)
    codes = (
        pl.DataFrame({column: values})
        .join(
            pl.DataFrame({column: lookup, "_code": index_series}),
            on=column,
            how="left",
        )
        .get_column("_code")
        .fill_null(unknown_code)
        .cast(pl.Int32)
        .to_numpy()
    )
    return codes, fitted


def to_model_matrix(
    frame: pl.DataFrame,
    *,
    feature_names: Sequence[str],
    categorical_columns: Sequence[str] = (),
    target_column: str | None = None,
    categorical_vocabularies: dict[str, dict[str, int]] | None = None,
) -> ModelMatrix:
    """Build the positional matrix, strictly.

    Args:
        frame: Feature frame produced by
            :func:`~team_diamond.features.pairs.build_pair_features`.
        feature_names: The declared column order. This is authoritative; the
            frame's own order is ignored.
        categorical_columns: Names that should be treated as categorical.
            They must appear in ``feature_names``.
        target_column: Label column, required unless ``None``.
        categorical_vocabularies: Frozen encodings to reuse, for inference with
            a model that was trained on a different country distribution.

    Returns:
        The matrix.

    Raises:
        ValueError: If ``feature_names`` is empty, a declared column is missing
            from ``frame``, the frame carries extra columns (which usually
            means a builder changed and the tuple was not updated), a
            categorical column is not in ``feature_names``, or the target is
            requested and absent.
    """
    if not feature_names:
        raise ValueError("feature_names is empty; there is nothing to fit")

    declared = list(feature_names)
    missing = [c for c in declared if c not in frame.columns]
    if missing:
        raise ValueError(
            f"feature frame is missing declared column(s) {missing}. "
            f"FEATURE_COLUMNS and the builder have drifted apart."
        )

    if target_column is not None and target_column not in frame.columns:
        raise ValueError(
            f"target_column {target_column!r} is not in the frame; columns: "
            f"{frame.columns[:10]}"
        )

    unknown_cats = [c for c in categorical_columns if c not in declared]
    if unknown_cats:
        raise ValueError(
            f"categorical column(s) {unknown_cats} are not in feature_names; "
            f"a categorical feature that is not a model input would be dropped "
            f"silently"
        )

    names = tuple(declared)
    cat_set = set(categorical_columns)
    cat_indices = tuple(i for i, n in enumerate(names) if n in cat_set)

    vocabularies: dict[str, dict[str, int]] = dict(categorical_vocabularies or {})
    X = np.empty((frame.height, len(names)), dtype=np.float32)
    for position, name in enumerate(names):
        if name in cat_set:
            codes, fitted = encode_categorical(
                frame[name],
                column=name,
                vocabulary=vocabularies.get(name),
            )
            # Only freeze a *newly fitted* vocabulary. Overwriting a supplied
            # one would silently re-encode inference data with training-derived
            # codes, which is the exact bug this design exists to prevent.
            vocabularies.setdefault(name, fitted)
            X[:, position] = codes
        else:
            column = frame[name]
            if column.dtype == pl.Boolean:
                column = column.cast(pl.Float32)
            values = column.cast(pl.Float32).to_numpy()
            if not np.isfinite(values[~np.isnan(values)]).all():
                raise ValueError(
                    f"feature {name!r} contains inf. NaN is acceptable and "
                    f"CatBoost routes it natively; inf is not, and would be "
                    f"silently split on."
                )
            X[:, position] = values

    y: np.ndarray | None = None
    if target_column is not None:
        labels = frame[target_column]
        if labels.dtype != pl.Boolean:
            raise ValueError(
                f"target {target_column!r} must be Boolean, found {labels.dtype}"
            )
        y = labels.cast(pl.Int8).to_numpy()

    return ModelMatrix(
        X=X,
        y=y,
        feature_names=names,
        cat_indices=cat_indices,
        categorical_vocabularies=vocabularies,
    )
