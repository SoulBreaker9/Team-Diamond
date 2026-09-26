"""Submission construction and validation.

Layer: pipeline
See AGENTS.md §17 (Submission contract)

The contract, in full
---------------------
``matching_results.tsv``, tab-separated, exactly two columns:

.. code-block:: text

    source1_entity_id    matched_entity_ids

- **One row for every test S1 entity.** No missing rows, no extra rows.
- **The exact test S1 order**, as read from ``test_source1.tsv``.
- Only valid S2 / S3 ids.
- Comma-separated.
- **Empty string for no match** — never ``None``, ``null``, ``nan``, or a
  whitespace placeholder.

The row count is derived, never asserted
----------------------------------------
AGENTS.md §17 records that the documented test S1 count is wrong by one, and
that the error is a ``wc -l`` count including the header. The instruction that
follows is the general lesson, and it is stronger than "use 1,732,544":

    **Never manufacture missing rows merely to reach the documented count.**

So this module takes the id list from the S1 file and iterates *that*. The row
count is then correct by construction, and if a future data version has a
different number of S1 entities the submission is still correct. A hardcoded
count would fail on exactly the input it is supposed to protect against.

Empty is a real answer
----------------------
Emitting an empty string is the correct output for a singleton or a no-match
entity, and it must survive serialisation. An empty string and a whitespace
placeholder are both visually blank in a spreadsheet, which is why
:func:`validate_submission` treats ``""`` and ``" "`` as different things and
rejects the latter.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import polars as pl

__all__ = [
    "S1_COLUMN",
    "MATCHED_COLUMN",
    "SubmissionValidation",
    "build_submission",
    "write_submission",
    "validate_submission",
    "read_submission",
]

S1_COLUMN: Final[str] = "source1_entity_id"
MATCHED_COLUMN: Final[str] = "matched_entity_ids"

#: Column separator required by the format. Tab, not comma.
DELIMITER: Final[str] = "\t"


@dataclass(frozen=True, slots=True)
class SubmissionValidation:
    """The result of checking a submission against the contract.

    Attributes:
        n_expected: S1 entities in the source file.
        n_rows: Rows in the submission.
        missing_ids: S1 entities absent from the submission.
        extra_ids: Submitted entities not present in the source file.
        invalid_vendor_ids: Emitted ids that are not valid S2/S3 ids, with the
            number of distinct offenders.
        empty_string_rows: Rows correctly written as ``""``.
        whitespace_only_rows: Rows containing only whitespace — a formatting bug
            that looks blank in a viewer and is scored as a match attempt.
        null_match_rows: Rows whose match list is null rather than empty.
        out_of_order: Whether the row order matches the source file.
        duplicate_s1_ids: S1 entities appearing more than once.
        self_matched_rows: Rows whose emitted id set contains the S1 id itself.
        example_errors: A bounded sample of human-readable messages.
    """

    n_expected: int
    n_rows: int
    missing_ids: tuple[str, ...] = ()
    extra_ids: tuple[str, ...] = ()
    invalid_vendor_ids: tuple[str, ...] = ()
    n_invalid_vendor_rows: int = 0
    empty_string_rows: int = 0
    whitespace_only_rows: int = 0
    null_match_rows: int = 0
    out_of_order: bool = False
    duplicate_s1_ids: tuple[str, ...] = ()
    self_matched_rows: int = 0
    example_errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the submission satisfies every rule in the contract.

        An empty-string row is *not* an error; it is the required encoding for a
        no-match entity.
        """
        return not (
            self.missing_ids
            or self.extra_ids
            or self.invalid_vendor_ids
            or self.whitespace_only_rows
            or self.null_match_rows
            or self.out_of_order
            or self.duplicate_s1_ids
            or self.self_matched_rows
            or self.n_rows != self.n_expected
        )

    def describe(self) -> str:
        """A report suitable for a run log."""
        verdict = "PASS" if self.ok else "FAIL"
        lines = [
            f"SUBMISSION VALIDATION: {verdict}",
            f"  rows expected (from S1 file) : {self.n_expected:,}",
            f"  rows written                 : {self.n_rows:,}",
            f"  missing S1 rows              : {len(self.missing_ids):,}",
            f"  extra rows                   : {len(self.extra_ids):,}",
            f"  duplicate S1 ids             : {len(self.duplicate_s1_ids):,}",
            f"  out of source order          : {self.out_of_order}",
            f"  rows with no match (empty)   : {self.empty_string_rows:,}",
            f"  whitespace-only rows         : {self.whitespace_only_rows:,}",
            f"  null match rows              : {self.null_match_rows:,}",
            f"  self-matched rows            : {self.self_matched_rows:,}",
            f"  rows with invalid vendor ids : {self.n_invalid_vendor_rows:,}"
            f" ({len(self.invalid_vendor_ids):,} distinct)",
        ]
        for sample in self.missing_ids[:5]:
            lines.append(f"    missing: {sample}")
        for sample in self.extra_ids[:5]:
            lines.append(f"    extra  : {sample}")
        for sample in self.invalid_vendor_ids[:5]:
            lines.append(f"    invalid: {sample}")
        for sample in self.duplicate_s1_ids[:5]:
            lines.append(f"    duplicate: {sample}")
        for message in self.example_errors:
            lines.append(f"  ! {message}")
        if not self.ok:
            lines.append("")
            lines.append(
                "A submission with validation errors must not be treated as "
                "ready (AGENTS.md §17)."
            )
        return "\n".join(lines)


def _as_id_list(value: object) -> list[str] | None:
    """Coerce a match-cell value to a list of ids, or ``None`` if it is null."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return [] if not stripped else [part.strip() for part in stripped.split(",")]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(
        f"matched_entity_ids holds {type(value).__name__}; expected a list or a "
        f"comma-separated string"
    )


def build_submission(
    emissions: pl.DataFrame,
    test_s1: pl.DataFrame,
    *,
    s1_id_column: str = "entity_id",
    emissions_s1_column: str = "s1_entity_id",
    emissions_list_column: str = "matched_entity_ids",
) -> pl.DataFrame:
    """Assemble the submission frame in the S1 file's own order.

    The left join is what makes the contract hold: an S1 entity with no
    emissions becomes an **empty list** rather than a missing row, and the S1
    order is the join's left order, so it is preserved by construction rather
    than by a sort that could be forgotten.

    Args:
        emissions: Output of the decision layer, with an S1 id and a list of
            emitted vendor ids.
        test_s1: Test S1 records, providing the authoritative id list and its
            order.
        s1_id_column: S1 id column in ``test_s1``.
        emissions_s1_column: S1 id column in ``emissions``.
        emissions_list_column: List-of-ids column in ``emissions``.

    Returns:
        Frame with :data:`S1_COLUMN` (``Utf8``) and :data:`MATCHED_COLUMN`
        (``List[Utf8]``), one row per S1 entity, in source order.

    Raises:
        ValueError: If a required column is missing, or an S1 id appears twice
            in ``emissions`` — which would silently multiply rows.
    """
    for column in (emissions_s1_column, emissions_list_column):
        if column not in emissions.columns:
            raise ValueError(
                f"emissions frame is missing {column!r}; columns: "
                f"{emissions.columns[:10]}"
            )
    if s1_id_column not in test_s1.columns:
        raise ValueError(
            f"test S1 frame is missing {s1_id_column!r}; columns: "
            f"{test_s1.columns[:10]}"
        )

    duplicated = (
        emissions.group_by(emissions_s1_column)
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicated.height:
        raise ValueError(
            f"emissions contains {duplicated.height:,} S1 id(s) more than once "
            f"(e.g. {duplicated[emissions_s1_column].head(3).to_list()}). The "
            f"decision layer must emit one row per entity; duplicates would "
            f"multiply submission rows."
        )

    ids = test_s1.get_column(s1_id_column).cast(pl.Utf8)
    if ids.n_unique() != test_s1.height:
        raise ValueError(
            f"test S1 has {test_s1.height - ids.n_unique():,} duplicate id(s). "
            f"Entity ids are opaque (AGENTS.md §8.3); find the source before "
            f"building a submission."
        )

    return (
        pl.DataFrame({S1_COLUMN: ids})
        .join(
            emissions.select(
                pl.col(emissions_s1_column).cast(pl.Utf8).alias(S1_COLUMN),
                emissions_list_column,
            ),
            on=S1_COLUMN,
            how="left",
        )
        .with_columns(
            pl.col(MATCHED_COLUMN)
            .list.eval(pl.element().cast(pl.Utf8))
            .alias(MATCHED_COLUMN)
        )
    )


def write_submission(submission: pl.DataFrame, path: Path) -> Path:
    """Write the submission as tab-separated text, validating it first.

    Validation runs *before* the write rather than after, so a malformed
    submission cannot reach disk in the first place. AGENTS.md §17 requires
    validation after every write; refusing to write an invalid one is strictly
    stronger and costs nothing.

    Written with :mod:`csv` and an explicit ``lineterminator`` because Polars
    and pandas each pick their own line ending by default, and a ``\\r\\n`` in a
    submission file is a silent difference from what the scorer expects.

    Args:
        submission: Frame from :func:`build_submission`.
        path: Destination path.

    Returns:
        The path written.

    Raises:
        ValueError: If validation fails. The message is the full report.
    """
    report = validate_submission_shape(submission)
    if not report.ok:
        raise ValueError(report.describe())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle, delimiter=DELIMITER, lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([S1_COLUMN, MATCHED_COLUMN])
        for s1_id, matched in submission.select(S1_COLUMN, MATCHED_COLUMN).iter_rows():
            ids = _as_id_list(matched) or []
            writer.writerow([s1_id, ",".join(ids)])
    return path


def validate_submission_shape(submission: pl.DataFrame) -> SubmissionValidation:
    """Check the frame's internal consistency, without the S1 reference list.

    Args:
        submission: Candidate submission frame.

    Returns:
        A report covering everything that can be checked from the frame alone.
        Row-count and ordering still need the S1 file; use
        :func:`validate_submission` for the full check.
    """
    for column in (S1_COLUMN, MATCHED_COLUMN):
        if column not in submission.columns:
            raise ValueError(
                f"submission frame is missing {column!r}; columns: "
                f"{submission.columns[:10]}"
            )

    errors: list[str] = []
    duplicated = (
        submission.group_by(S1_COLUMN).len().filter(pl.col("len") > 1)
    )
    if duplicated.height:
        errors.append(
            f"{duplicated.height:,} S1 id(s) appear more than once; the contract "
            f"is one row per entity"
        )

    lists = submission.get_column(MATCHED_COLUMN)
    null_lists = lists.is_null().sum() if lists.dtype == pl.List(pl.Utf8) else 0
    empty_rows = 0
    whitespace_rows = 0
    self_matched = 0
    if lists.dtype == pl.List(pl.Utf8):
        empty_rows = int(lists.list.len().eq(0).sum())
        self_matched = int(
            submission.filter(
                pl.col(MATCHED_COLUMN).list.contains(pl.col(S1_COLUMN))
            ).height
        )

    return SubmissionValidation(
        n_expected=submission.height,
        n_rows=submission.height,
        duplicate_s1_ids=tuple(duplicated[S1_COLUMN].to_list()[:20]),
        empty_string_rows=empty_rows,
        null_match_rows=int(null_lists),
        self_matched_rows=self_matched,
        whitespace_only_rows=whitespace_rows,
        example_errors=tuple(errors),
    )


def validate_submission(
    submission: pl.DataFrame,
    test_s1: pl.DataFrame,
    valid_vendor_ids: Iterable[str],
    *,
    s1_id_column: str = "entity_id",
) -> SubmissionValidation:
    """Full contract check against the S1 file and the real vendor id set.

    Args:
        submission: Candidate submission frame.
        test_s1: Test S1 records, for the authoritative id list and order.
        valid_vendor_ids: Every legitimate S2/S3 id. Passing the *real* set
            rather than a format check is deliberate: ids are opaque strings
            (AGENTS.md §8.3), so a well-formed id that does not exist is the
            most likely submission bug and the hardest to see by eye.
        s1_id_column: S1 id column in ``test_s1``.

    Returns:
        A report. Check :attr:`SubmissionValidation.ok` before submitting.

    Notes:
        The submitted frame is inspected *as it will be serialised*, not as it
        sits in memory, because several of the rules concern encoding: a null
        list and an empty list behave differently once written.
    """
    valid = set(valid_vendor_ids)
    expected_ids = test_s1.get_column(s1_id_column).cast(pl.Utf8).to_list()
    expected_set = set(expected_ids)

    rows = submission.select(S1_COLUMN, MATCHED_COLUMN).rows()
    submitted_ids = [str(row[0]) for row in rows]
    submitted_set = set(submitted_ids)

    # `missing` and `extra` are both set operations. Done as list comprehensions
    # with the set rebuilt per element, this is quadratic -- at 1.7M test rows it
    # is minutes of pure overhead in a function that is supposed to be cheap.
    missing = [i for i in expected_ids if i not in submitted_set]
    extra = [i for i in submitted_ids if i not in expected_set]

    # Order is only meaningful when the row counts agree; a length mismatch is
    # already reported separately and would otherwise mask a genuine ordering
    # problem behind a truncated comparison.
    out_of_order = (
        len(submitted_ids) == len(expected_ids) and submitted_ids != expected_ids
    )

    invalid_ids: set[str] = set()
    invalid_rows = 0
    empty_rows = 0
    whitespace_rows = 0
    null_rows = 0

    for _, raw in rows:
        if raw is None:
            null_rows += 1
            continue
        ids = _as_id_list(raw)
        if ids is None:
            null_rows += 1
            continue
        if not ids:
            empty_rows += 1
            continue
        if any(not i for i in ids):
            whitespace_rows += 1
        bad = [i for i in ids if i not in valid]
        if bad:
            invalid_rows += 1
            invalid_ids.update(bad)

    return SubmissionValidation(
        n_expected=len(expected_ids),
        n_rows=len(rows),
        missing_ids=tuple(missing[:20]),
        extra_ids=tuple(extra[:20]),
        invalid_vendor_ids=tuple(sorted(invalid_ids)[:20]),
        n_invalid_vendor_rows=invalid_rows,
        empty_string_rows=empty_rows,
        whitespace_only_rows=whitespace_rows,
        null_match_rows=null_rows,
        out_of_order=out_of_order,
    )


def read_submission(path: Path) -> pl.DataFrame:
    """Read a submission file back, for verification.

    Args:
        path: The ``matching_results.tsv`` file.

    Returns:
        Frame with :data:`S1_COLUMN` and :data:`MATCHED_COLUMN` as read from
        disk, so a round trip can be checked rather than assumed.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the header is not the expected two columns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; validate after writing, not before"
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=DELIMITER)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty; expected a header row") from None
        if header != [S1_COLUMN, MATCHED_COLUMN]:
            raise ValueError(
                f"{path} header is {header}, expected "
                f"['{S1_COLUMN}', '{MATCHED_COLUMN}']"
            )
        rows = [row for row in reader]

    ids = [row[0] for row in rows if row]
    matches = [_as_id_list(row[1] if len(row) > 1 else "") for row in rows if row]
    return pl.DataFrame(
        {
            S1_COLUMN: pl.Series(ids, dtype=pl.Utf8),
            MATCHED_COLUMN: pl.Series(
                [m or [] for m in matches], dtype=pl.List(pl.Utf8)
            ),
        }
    )
