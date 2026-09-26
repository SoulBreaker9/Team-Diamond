"""Data layer — locating, caching, and loading the challenge dataset.

Layer: data
See AGENTS.md §5 (the dataset is not in the repository), §8.2 (schema and nulls)

Split into three concerns on purpose, because they fail in different ways and
the failure message should say which one broke:

- :mod:`team_diamond.data.paths` answers *where the data is*, for local disk and
  for S3, and never returns an unverified root.
- :mod:`team_diamond.data.storage` answers *how the bytes are read*, and mirrors
  the TSVs into a local Parquet cache so repeated runs are not dominated by
  text parsing.
- :mod:`team_diamond.data.loading` answers *what the records mean*, and enforces
  the schema so downstream layers cannot silently receive nulls where they
  expect strings.
"""

from __future__ import annotations

from team_diamond.data.loading import (
    cache_for_path,
    ground_truth_frame,
    load_entities,
    load_ground_truth,
    load_split,
    match_cardinality,
)
from team_diamond.data.paths import (
    EXPECTED_ROW_COUNTS,
    SPLIT_FILES,
    DatasetNotFoundError,
    DatasetPaths,
    resolve_data_root,
    verify_layout,
)
from team_diamond.data.storage import (
    CacheMiss,
    ParquetCache,
    S3Location,
    is_s3_path,
    parse_s3_path,
    s3_session,
)

__all__ = [
    "EXPECTED_ROW_COUNTS",
    "SPLIT_FILES",
    "CacheMiss",
    "DatasetNotFoundError",
    "DatasetPaths",
    "ParquetCache",
    "S3Location",
    "cache_for_path",
    "ground_truth_frame",
    "is_s3_path",
    "load_entities",
    "load_ground_truth",
    "load_split",
    "match_cardinality",
    "parse_s3_path",
    "resolve_data_root",
    "s3_session",
    "verify_layout",
]
