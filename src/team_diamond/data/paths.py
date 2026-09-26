"""Locate the challenge dataset on local disk or in S3, and fail loudly.

Layer: data
See AGENTS.md §5 ("The dataset is not in the repository"), §8.4 (verify, don't assume)

Why this module was rewritten
-----------------------------
The first version of this file only understood local ``Path`` objects and
hardcoded one laptop directory. That is wrong for the target environment: the
run happens in a SageMaker Jupyter Space where the dataset is in S3 at
``s3://<bucket>/student_resource/...``. Three requirements follow, and each one
is a rule from AGENTS.md rather than a preference:

1. **A dataset root may be an S3 URI.** Path arithmetic has to work for both
   schemes, so it goes through string joining rather than ``Path.__truediv__``.
2. **The layout must be verified, never assumed.** The local layout is
   ``<root>/dataset/{train,test}/*.tsv``; whether S3 carries the ``dataset/``
   level is not something to guess. This module *probes* candidate layouts and,
   if none match, raises an error listing every location tried and the exact
   keys it looked for. This is the §5 requirement that a path helper must never
   return a plausible-looking nonexistent path.
3. **Credentials must resolve in the execution environment.** Probing S3 goes
   through boto3 so the SageMaker instance role is picked up. If boto3 or
   credentials are missing, the error says which, rather than surfacing later
   as an empty frame.

Resolution order
----------------
1. Explicit ``override`` argument.
2. ``TEAM_DIAMOND_DATA_ROOT`` environment variable.
3. ``TEAM_DIAMOND_S3_PREFIX`` environment variable (bucket + prefix).
4. Built-in candidates: the S3 bucket this project uses, then the known local
   directories, then ``./data``.

Every candidate is verified to contain both splits before being accepted, so
step 4 cannot silently pick a wrong directory.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

from team_diamond.data.storage import is_s3_path, parse_s3_path, s3_session

__all__ = [
    "DatasetNotFoundError",
    "DatasetPaths",
    "EXPECTED_ROW_COUNTS",
    "SPLIT_FILES",
    "resolve_data_root",
    "verify_layout",
]

# Measured record counts. Recorded so a mismatch is *detectable*, but NEVER
# used to build a path or to size an output. AGENTS.md §17: the submission row
# count is read from the file at run time.
#
# The challenge documentation states 1,732,545 test S1 rows. That is wrong: the
# real count is 1,732,544, and the documented figure is a `wc -l` line count
# that includes the header. Verified via the official validator's own
# ``read_ids()``. See experiments/registry.csv E001.
EXPECTED_ROW_COUNTS: dict[str, int] = {
    "train_source1.tsv": 2_206_821,
    "train_source2.tsv": 5_034_616,
    "train_source3.tsv": 5_285_603,
    "train_ground_truth.tsv": 2_206_821,
    "test_source1.tsv": 1_732_544,
    "test_source2.tsv": 4_887_273,
    "test_source3.tsv": 5_082_316,
}

#: The S3 bucket this project's dataset lives in, as used in the SageMaker
#: Jupyter Space. This is the *dataset* bucket provisioned for the challenge,
#: not the default SageMaker bucket, and it is a constant only because the
#: bucket was provisioned to the team; it is still verified like every other
#: candidate and can be overridden by ``TEAM_DIAMOND_S3_PREFIX``.
DEFAULT_S3_BUCKET = "sagemaker-ap-south-1-115775806821"

#: Environment variable holding a full dataset root, local or S3.
DATA_ROOT_ENV = "TEAM_DIAMOND_DATA_ROOT"

#: Environment variable holding an S3 prefix to probe.
S3_PREFIX_ENV = "TEAM_DIAMOND_S3_PREFIX"

#: File names expected in each split.
SPLIT_FILES: dict[str, tuple[str, ...]] = {
    "train": (
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    ),
    "test": ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"),
}

#: Sub-paths probed *below* a candidate root, in order. The local copy nests the
#: splits under ``dataset/``; whether S3 does is unknown until probed, so both
#: are tried. A root that satisfies either is accepted.
_SUBPATH_CANDIDATES: tuple[str, ...] = ("", "dataset", "data")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when no dataset root satisfies the expected layout."""


def _join(root: str, *parts: str) -> str:
    """Join path segments for either a local path or an S3 URI.

    ``pathlib`` cannot be used because the root may be an S3 URI, and S3 keys
    are always forward-slashed regardless of the local platform.
    """
    cleaned = root.rstrip("/")
    for part in parts:
        cleaned = f"{cleaned}/{part.strip('/')}"
    return cleaned


def _local_join(root: str, *parts: str) -> str:
    """Join segments and convert to a local path, for local roots only."""
    import pathlib

    return str(pathlib.Path(root).joinpath(*parts))


def _s3_keys(prefix: str) -> set[str]:
    """List every object key under an S3 prefix, recursively.

    Pagination is implemented so a larger prefix cannot silently produce
    a partial view and a false "not found".
    """
    location = parse_s3_path(prefix)
    client = s3_session().client("s3")
    keys: set[str] = set()
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {
            "Bucket": location.bucket,
            "Prefix": f"{location.key}/" if location.key else "",
        }
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)  # type: ignore[arg-type]
        for item in response.get("Contents", ()):
            keys.add(str(item["Key"]))
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not token:
            break
    return keys


def verify_layout(root: str) -> str | None:
    """Return the sub-path under ``root`` that holds the splits, or ``None``.

    Args:
        root: A local directory or ``s3://`` URI.

    Returns:
        The sub-path (``""``, ``"dataset"``, ...) that contains both
        ``train/train_source1.tsv`` and ``test/test_source1.tsv``, or ``None``
        if no candidate sub-path satisfies that.
    """
    if is_s3_path(root):
        try:
            available = _s3_keys(root)
        except Exception as error:  # noqa: BLE001 - reported as "not found"
            raise DatasetNotFoundError(
                f"could not list S3 prefix {root!r}: {error}\n"
                f"Check the bucket name, the prefix, and that this instance's "
                f"role has s3:ListBucket on it."
            ) from error

        # For S3, `available` contains keys relative to the bucket (e.g., "prefix/train/file.tsv").
        # Extract the relative prefix from the root for comparison.
        parsed = parse_s3_path(root)
        root_relative = parsed.key.rstrip("/") if parsed.key else ""
        if root_relative:
            root_relative += "/"

        for sub in _SUBPATH_CANDIDATES:
            if sub:
                probe_relative = f"{root_relative}{sub.strip('/')}/"
            else:
                probe_relative = root_relative
            wanted = {
                f"{probe_relative}{split}/{name}"
                for split, names in SPLIT_FILES.items()
                for name in names
            }
            if wanted <= available:
                return sub
        return None

    import pathlib

    base = pathlib.Path(root).expanduser()
    if not base.is_dir():
        return None
    for sub in _SUBPATH_CANDIDATES:
        candidate = base / sub if sub else base
        if not candidate.is_dir():
            continue
        if all(
            (candidate / split / name).is_file()
            for split, names in SPLIT_FILES.items()
            for name in names
        ):
            return sub
    return None


def _candidate_roots() -> list[str]:
    """Every root to try, most specific first."""
    candidates: list[str] = []
    env_root = os.environ.get(DATA_ROOT_ENV)
    if env_root:
        candidates.append(env_root)
    env_s3 = os.environ.get(S3_PREFIX_ENV)
    if env_s3:
        candidates.append(env_s3)
    candidates.extend(
        [
            f"s3://{DEFAULT_S3_BUCKET}/student_resource",
            f"s3://{DEFAULT_S3_BUCKET}/student_resource/dataset",
            "/home/hetm/Desktop/Hackathon/6ab10eb3b23ba_student_resource"
            "/student_resource/dataset",
            "data",
            "../data",
            "dataset",
        ]
    )
    return candidates


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    """Verified locations of every challenge file.

    Attributes:
        root: The dataset root, including the discovered sub-path, so that
            ``root`` alone is sufficient to locate every file.
        base: The candidate root that was verified, before the sub-path.
        sub_path: Which sub-path under ``base`` was found to hold the splits.
    """

    root: str
    base: str
    sub_path: str

    # -- construction --------------------------------------------------------
    @classmethod
    def discover(cls, override: str | os.PathLike[str] | None = None) -> DatasetPaths:
        """Locate and verify the dataset.

        Args:
            override: Explicit root. If given it must be valid; a bad override
                is an error, never a reason to fall through to discovery.

        Returns:
            Verified paths.

        Raises:
            DatasetNotFoundError: If no candidate satisfies the layout.
        """
        if override is not None:
            root = str(override)
            sub = verify_layout(root)
            if sub is None:
                raise DatasetNotFoundError(_layout_error([root], root=root))
            return cls(root=_join(root, sub) if sub else root.rstrip("/"), base=root, sub_path=sub)

        tried: list[str] = []
        problems: list[str] = []
        for candidate in _candidate_roots():
            tried.append(candidate)
            try:
                sub = verify_layout(candidate)
            except DatasetNotFoundError as error:
                # An unreachable candidate is recorded, not fatal, so that a
                # developer on a laptop without AWS credentials can still fall
                # back to a local copy. If nothing else matches, the reason is
                # reported verbatim below -- so a genuine credentials or
                # permissions failure is still loud.
                problems.append(f"  - {candidate}\n      {first_line(str(error))}")
                continue
            if sub is not None:
                return cls(
                    root=_join(candidate, sub) if sub else candidate.rstrip("/"),
                    base=candidate,
                    sub_path=sub,
                )
        raise DatasetNotFoundError(_layout_error(tried, problems=problems))

    # -- path construction ---------------------------------------------------
    def file(self, split: str, filename: str) -> str:
        """Full location of one dataset file, as a local path or S3 URI."""
        if split not in SPLIT_FILES:
            raise ValueError(f"unknown split {split!r}; expected one of {list(SPLIT_FILES)}")
        if filename not in SPLIT_FILES[split]:
            raise ValueError(f"{filename!r} is not a file of the {split!r} split")
        return _join(self.root, split, filename)

    def _p(self, split: str, filename: str) -> str:
        location = self.file(split, filename)
        if is_s3_path(location):
            return location  # existence is proven by verify_layout
        import pathlib

        if not pathlib.Path(location).is_file():
            raise DatasetNotFoundError(f"expected file is missing: {location}")
        return location

    @property
    def train_source1(self) -> str:
        return self._p("train", "train_source1.tsv")

    @property
    def train_source2(self) -> str:
        return self._p("train", "train_source2.tsv")

    @property
    def train_source3(self) -> str:
        return self._p("train", "train_source3.tsv")

    @property
    def train_ground_truth(self) -> str:
        return self._p("train", "train_ground_truth.tsv")

    @property
    def test_source1(self) -> str:
        return self._p("test", "test_source1.tsv")

    @property
    def test_source2(self) -> str:
        return self._p("test", "test_source2.tsv")

    @property
    def test_source3(self) -> str:
        return self._p("test", "test_source3.tsv")

    # -- helpers -------------------------------------------------------------
    @property
    def is_remote(self) -> bool:
        """True when the dataset lives in S3 rather than on local disk."""
        return is_s3_path(self.root)

    def cache(self, cache_root: str | os.PathLike[str] | None = None):
        """Build a :class:`~team_diamond.data.storage.ParquetCache` for this root.

        Args:
            cache_root: Parquet cache location; see
                :meth:`~team_diamond.data.storage.ParquetCache.for_root`.

        Returns:
            A cache rooted at this dataset's verified location.
        """
        from team_diamond.data.storage import ParquetCache

        return ParquetCache.for_root(self.root, cache_root)

    def expected_row_count(self, filename: str) -> int | None:
        """Measured row count for ``filename``, or ``None`` if unknown.

        For *test* files the published counts are ``wc -l`` line counts that
        include the header, so the value here is the corrected record count.
        Never use this to build an output; read the ids from the file instead
        (AGENTS.md §17).
        """
        return EXPECTED_ROW_COUNTS.get(filename)

    def __str__(self) -> str:
        scheme = "s3" if self.is_remote else "local"
        return f"DatasetPaths({scheme}: {self.root})"


def resolve_data_root(override: str | os.PathLike[str] | None = None) -> str:
    """Return a verified dataset root as a string.

    A thin wrapper over :meth:`DatasetPaths.discover` kept for callers that only
    want the root. Prefer :meth:`DatasetPaths.discover`, which also hands back
    the sub-path that was discovered.

    Args:
        override: Explicit root, local or ``s3://``. Must be valid.

    Returns:
        The verified root, with the discovered sub-path already appended.

    Raises:
        DatasetNotFoundError: If no candidate satisfies the layout.
    """
    return DatasetPaths.discover(override).root


def first_line(text: str) -> str:
    """First non-empty line of a message, for compact error reporting."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()


def _layout_error(
    tried: Iterable[str],
    root: str | None = None,
    problems: Iterable[str] | None = None,
) -> str:
    """Build an error message that makes the fix obvious from the text alone."""
    lines = [
        "Could not locate the challenge dataset.",
        "",
        "Looked for, under each candidate root:",
    ]
    for split, names in SPLIT_FILES.items():
        lines.append(f"  <root>/{split}/{{ {' , '.join(names) }}}")
    lines += [
        "",
        "Sub-paths probed below each root: "
        + ", ".join(repr(s) for s in _SUBPATH_CANDIDATES),
        "",
        "Tried:",
    ]
    lines += [f"  - {c}" for c in tried]
    problem_list = list(problems or ())
    if problem_list:
        lines += ["", "Candidates that could not be inspected:", *problem_list]
    lines += [
        "",
        "Fix by setting one of:",
        f"  export {DATA_ROOT_ENV}=s3://<bucket>/student_resource/dataset",
        f"  export {S3_PREFIX_ENV}=s3://<bucket>/student_resource",
        f"  export {DATA_ROOT_ENV}=/local/path/to/dataset",
    ]
    if root:
        lines += ["", f"(configured root was: {root})"]
    return "\n".join(lines)
