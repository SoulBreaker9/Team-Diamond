"""pipeline — Wire the layers together for a reproducible end-to-end run.

STATUS: implemented. Every stage is a separate function; nothing runs at import
time.

Layer contract (AGENTS.md §5. Source of truth and layout)
---------------------------------------------------------
keep every stage separately measurable; never collapse layers for speed

Depends on
----------
all

Must not
--------
contain business logic — that belongs in the layer that owns it
"""

from __future__ import annotations

from team_diamond.pipeline.run import (
    VENDOR_COLUMNS,
    RunReport,
    git_commit,
    prepare_vendor_pool,
    generate_retrieval_plan,
    featurise,
    train_and_evaluate,
    predict_and_submit,
    main,
)
from team_diamond.pipeline.scale import (
    ScaleEstimate,
    InsufficientMemory,
    StageRecord,
    MemoryTrace,
    rss_bytes,
    available_bytes,
    total_bytes,
    GiB,
    MIN_FREE_BYTES_FOR_FULL_RUN,
    FULL_RUN_VENDOR_THRESHOLD,
    guard_full_corpus_run,
)
from team_diamond.pipeline.submission import (
    S1_COLUMN,
    MATCHED_COLUMN,
    SubmissionValidation,
    build_submission,
    write_submission,
    validate_submission,
    read_submission,
)

__all__ = [
    "VENDOR_COLUMNS",
    "RunReport",
    "git_commit",
    "prepare_vendor_pool",
    "generate_retrieval_plan",
    "featurise",
    "train_and_evaluate",
    "predict_and_submit",
    "main",
    "ScaleEstimate",
    "InsufficientMemory",
    "StageRecord",
    "MemoryTrace",
    "rss_bytes",
    "available_bytes",
    "total_bytes",
    "GiB",
    "MIN_FREE_BYTES_FOR_FULL_RUN",
    "FULL_RUN_VENDOR_THRESHOLD",
    "guard_full_corpus_run",
    "S1_COLUMN",
    "MATCHED_COLUMN",
    "SubmissionValidation",
    "build_submission",
    "write_submission",
    "validate_submission",
    "read_submission",
]