#!/usr/bin/env python
"""Validate ``matching_results.tsv`` against the submission contract.

Usage
-----
.. code-block:: bash

    python utils/validate_submission.py matching_results.tsv

Exits 0 if the submission satisfies every rule, 1 otherwise. The check is
performed against the *real* S1 and vendor id sets from the dataset rather than
against the shape of the file, because the most damaging submission bug is a
well-formed id that does not exist, and that is invisible to a format check.

See AGENTS.md §17. A submission with validation errors must not be treated as
ready, so run this after every write.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import polars as pl  # noqa: E402

from team_diamond.data import DatasetPaths, load_split  # noqa: E402
from team_diamond.pipeline.submission import (  # noqa: E402
    read_submission,
    validate_submission,
)


def main(argv: list[str]) -> int:
    """Run the validator.

    Args:
        argv: Command-line arguments, excluding the program name.

    Returns:
        Process exit code: 0 on a valid submission, 1 on an invalid one or a
        usage/data error.
    """
    if len(argv) != 1:
        print(
            "usage: python utils/validate_submission.py <matching_results.tsv>",
            file=sys.stderr,
        )
        return 1

    target = Path(argv[0])
    if not target.is_file():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 1

    try:
        submission = read_submission(target)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: could not read submission: {exc}", file=sys.stderr)
        return 1

    print(f"read {target}: {submission.height:,} rows")

    try:
        paths = DatasetPaths.discover()
        split = load_split(paths, "test")
    except Exception as exc:  # noqa: BLE001
        print(
            f"error: could not load the test split to validate against: {exc}\n"
            f"       The submission cannot be validated without the dataset, so "
            f"it is NOT considered ready.",
            file=sys.stderr,
        )
        return 1

    vendors = pl.concat(
        [split["source2"], split["source3"]], how="vertical"
    ).get_column("entity_id")
    valid_ids = vendors.cast(pl.Utf8).unique().to_list()

    report = validate_submission(submission, split["source1"], valid_ids)
    print(report.describe())

    if report.ok:
        print("\nSubmission is valid.")
        return 0

    print(
        "\nSubmission is NOT valid. Fix the problems above before submitting.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
