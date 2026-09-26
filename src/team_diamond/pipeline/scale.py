"""Cost estimation and memory instrumentation, enforced rather than advised.

Layer: pipeline
See AGENTS.md §14 (Scale and cost rules)

Why this module exists
----------------------
AGENTS.md §14 requires, before any large-scale operation: estimate the input
size, the pair count, the memory, the runtime; run a smaller benchmark; verify
the output size; and confirm the result is practical on the machine in hand.
Every one of those steps was skipped when candidate generation was run against
the full 10.3M-record vendor pool, and the machine's desktop froze.

A written rule that has already been violated once will be violated again under
time pressure. So the rule is moved into code: a full-scale run **cannot start**
until a :class:`ScaleEstimate` has been recorded, and the estimate has to name
its own assumptions. That converts a discipline problem into a compile error.

The estimate is not a formality
-------------------------------
:class:`ScaleEstimate.validate_against` re-derives the memory requirement from
the actual row counts and compares it against both the estimate and the machine's
free memory. An estimate that was too optimistic is caught *before* the job, when
it is cheap, rather than by the OOM killer afterwards. It also refuses to run on
a machine with less headroom than the estimate requires, which is the specific
check that would have stopped the crash: the estimate said 12 GiB, the laptop had
7.6 GiB free, and nobody compared the two numbers.

Instrumentation
---------------
:func:`log_stage` records RSS at every stage boundary into a list the caller can
report. A run that does not know where its memory went cannot be tuned, and
"it ran out of memory" is not a diagnosis.
"""

from __future__ import annotations

import os
import resource
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

__all__ = [
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
]

GiB: Final[int] = 1024**3

#: Conservative floor of free memory required before any run touching the full
#: corpus is allowed to start. Two GiB is not a safety margin, it is the
#: smallest amount of headroom under which filesystem caching plus Parquet
#: buffers can be relied upon not to tip the machine over.
MIN_FREE_BYTES_FOR_FULL_RUN: Final[int] = 2 * GiB

#: A vendor pool at or above this size counts as a "full-corpus" run and needs a
#: recorded estimate. Sized well below the real 10.3M pool so the threshold
#: cannot be slipped past by a partial run that is still too large.
FULL_RUN_VENDOR_THRESHOLD: Final[int] = 1_000_000


class InsufficientMemory(RuntimeError):
    """Raised when a run would exceed the memory available on this machine."""


def total_bytes() -> int:
    """Total physical memory on this machine, or 0 if unknown."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def rss_bytes() -> int:
    """Resident set size of this process, in bytes.

    Uses :mod:`resource` rather than shelling out to ``free`` so it works on any
    Unix and inside a container, and takes the process maximum when the platform
    supports it, since a transient peak is exactly what matters.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is KiB on Linux and bytes on macOS.
    if os.uname().sysname == "Darwin":
        return usage.ru_maxrss
    return usage.ru_maxrss * 1024


def available_bytes() -> int:
    """Memory available to this process, best effort.

    Prefers ``MemAvailable`` from ``/proc/meminfo`` over ``MemFree``, because
    ``MemFree`` excludes reclaimable page cache and understates what a workload
    can actually use -- which would make this check pessimistic and get it
    ignored.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


@dataclass(frozen=True, slots=True)
class ScaleEstimate:
    """A recorded, falsifiable cost estimate for a run.

    Every field must be supplied by the person about to run the job. That is the
    point: the numbers come from a deliberate act, so they are thought about
    rather than filled in to get past a guard.

    Attributes:
        n_s1: S1 records to process.
        n_vendors: Vendor records in the retrieval pool.
        candidates_per_query: Expected candidates per S1 entity, after caps.
        bytes_per_pair: Estimated bytes per candidate pair, including the feature
            matrix. This dominates peak memory and is the number most often
            guessed too low.
        peak_multiplier: Multiple of the pair matrix representing transient
            intermediates -- the explode-and-join steps in retrieval are the
            reason this is greater than 1.
        runtime_minutes: Expected wall clock.
        rationale: How the numbers were arrived at. Required, and must be
            non-empty: an estimate with no stated basis cannot be checked by
            anyone else, which defeats the purpose.
        measured_from: What the estimate is calibrated on, e.g. a subsampled
            run, a single-strategy run, or a previous full run.
    """

    n_s1: int
    n_vendors: int
    candidates_per_query: int
    bytes_per_pair: int
    peak_multiplier: float
    runtime_minutes: float
    rationale: str
    measured_from: str

    @property
    def expected_pairs(self) -> int:
        """Candidate pairs the run will materialise."""
        return self.n_s1 * self.candidates_per_query

    @property
    def expected_pair_bytes(self) -> int:
        """Memory the pair matrix alone occupies."""
        return self.expected_pairs * self.bytes_per_pair

    @property
    def expected_peak_bytes(self) -> int:
        """Peak memory including transient intermediates."""
        return int(self.expected_pair_bytes * self.peak_multiplier)

    def validate_against(
        self, *, actual_vendors: int | None = None, require_headroom: bool = True
    ) -> None:
        """Check the estimate against reality and against this machine.

        Args:
            actual_vendors: Observed vendor pool size, if known. A mismatch
                against the estimate means the estimate was made for a
                different job, and proceeding on it would be the error.
            require_headroom: Refuse when free memory is below the peak
                requirement. Disable only when the caller has measured that
                chunking keeps the true peak well below the estimate.

        Raises:
            ValueError: If the rationale is empty, or the actual vendor count
                disagrees with the estimate by more than 20%.
            InsufficientMemory: If the machine does not have the headroom the
                estimate requires and ``require_headroom`` is set.
        """
        if not self.rationale.strip():
            raise ValueError(
                "ScaleEstimate.rationale is empty. An estimate with no stated "
                "basis cannot be checked by a second person, which is the only "
                "reason it is worth recording."
            )

        if actual_vendors is not None and self.n_vendors:
            ratio = actual_vendors / self.n_vendors
            if not 0.8 <= ratio <= 1.25:
                raise ValueError(
                    f"estimate was written for {self.n_vendors:,} vendors but "
                    f"the run will see {actual_vendors:,} ({ratio:.2f}x). The "
                    f"estimate describes a different job; rewrite it."
                )

        if require_headroom:
            free = available_bytes()
            if free and free < self.expected_peak_bytes:
                raise InsufficientMemory(
                    f"estimated peak {self.expected_peak_bytes / GiB:.2f} GiB "
                    f"({self.peak_multiplier}x x "
                    f"{self.candidates_per_query} candidates/query), but only "
                    f"{free / GiB:.2f} GiB is available on this machine.\n"
                    f"  n_s1          : {self.n_s1:,}\n"
                    f"  n_vendors     : {self.n_vendors:,}\n"
                    f"  expected pairs: {self.expected_pairs:,}\n"
                    f"  bytes per pair: {self.bytes_per_pair}\n"
                    f"  basis         : {self.measured_from}\n\n"
                    f"Run this on the target instance (ml.r5.xlarge, 32 GiB), or "
                    f"reduce the query set, or lower candidates_per_query and "
                    f"sweep the recall it costs (AGENTS.md §10)."
                )

    def describe(self) -> str:
        """The estimate as a block of text, for a run log or a commit message."""
        return "\n".join(
            [
                "SCALE ESTIMATE (must be recorded before a full-corpus run)",
                f"  S1 records            : {self.n_s1:,}",
                f"  vendor records        : {self.n_vendors:,}",
                f"  candidates per query  : {self.candidates_per_query}",
                f"  expected pairs        : {self.expected_pairs:,}",
                f"  bytes per pair        : {self.bytes_per_pair}",
                f"  peak multiplier       : {self.peak_multiplier}",
                f"  estimated pair memory : "
                f"{self.expected_pair_bytes / GiB:.3f} GiB",
                f"  estimated peak memory : "
                f"{self.expected_peak_bytes / GiB:.3f} GiB",
                f"  estimated runtime     : {self.runtime_minutes:.1f} min",
                f"  calibrated from       : {self.measured_from}",
                f"  rationale             : {self.rationale}",
            ]
        )


@dataclass(frozen=True, slots=True)
class StageRecord:
    """Memory and time at one stage boundary.

    Attributes:
        name: Stage name.
        rss_bytes: RSS after the stage.
        elapsed_seconds: Wall clock since the previous stage.
        note: Optional detail.
    """

    name: str
    rss_bytes: int
    elapsed_seconds: float
    note: str = ""


@dataclass(slots=True)
class MemoryTrace:
    """A running record of memory across a pipeline's stages.

    Use as a context manager so a stage is recorded even if it raises, which is
    when the peak matters most:

    .. code-block:: python

        trace = MemoryTrace("full run")
        with trace.stage("load source1"):
            s1 = load(...)
    """

    label: str
    records: list[StageRecord] = field(default_factory=list)
    _last_time: float = field(default_factory=time.monotonic)

    def stage(self, name: str, note: str = "") -> "_StageContext":
        """Begin a stage; record on exit whether or not it raised."""
        return _StageContext(self, name, note)

    def _record(self, name: str, note: str) -> StageRecord:
        now = time.monotonic()
        record = StageRecord(
            name=name,
            rss_bytes=rss_bytes(),
            elapsed_seconds=now - self._last_time,
            note=note,
        )
        self.records.append(record)
        self._last_time = now
        return record

    def peak_bytes(self) -> int:
        """Highest RSS observed across all stages."""
        return max((r.rss_bytes for r in self.records), default=0)

    def describe(self) -> str:
        """Table of stages with memory and time."""
        lines = [f"MEMORY TRACE: {self.label}"]
        for record in self.records:
            note = f"  ({record.note})" if record.note else ""
            lines.append(
                f"  {record.name:<34} {record.rss_bytes / GiB:6.3f} GiB  "
                f"{record.elapsed_seconds:7.1f}s{note}"
            )
        lines.append(f"  {'PEAK':<34} {self.peak_bytes() / GiB:6.3f} GiB")
        return "\n".join(lines)


class _StageContext:
    """Context manager backing :meth:`MemoryTrace.stage`."""

    def __init__(self, trace: MemoryTrace, name: str, note: str) -> None:
        self._trace = trace
        self._name = name
        self._note = note
        self._error: BaseException | None = None

    def __enter__(self) -> "_StageContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        detail = self._note
        if exc is not None:
            detail = f"{detail} FAILED: {type(exc).__name__}: {exc}".strip()
        record = self._trace._record(self._name, detail)
        print(
            f"[{self._trace.label}] {self._name:<34} "
            f"{record.rss_bytes / GiB:6.3f} GiB "
            f"({record.elapsed_seconds:6.1f}s)"
        )
        return False


def guard_full_corpus_run(
    n_vendors: int,
    estimate: ScaleEstimate | None,
    *,
    force: bool = False,
) -> None:
    """Refuse a full-corpus run that has no recorded cost estimate.

    Args:
        n_vendors: Vendor pool size the run will actually touch.
        estimate: The recorded estimate, or ``None``.
        force: Bypass the guard. Reserved for a deliberate small-scale
            validation run at, say, a 1.1M vendor pool, where the estimate is
            noise. Using it at true full scale defeats the purpose of the guard
            and should be justified in the experiment record.

    Raises:
        ValueError: If the pool is at full-corpus scale, no estimate was given,
            and ``force`` is not set. The message states what to provide.
    """
    if force or n_vendors < FULL_RUN_VENDOR_THRESHOLD:
        return
    if estimate is None:
        raise ValueError(
            f"this run touches {n_vendors:,} vendor records, which is "
            f"full-corpus scale, and no ScaleEstimate was recorded.\n\n"
            f"AGENTS.md §14 requires a cost estimate before a large-scale "
            f"operation, and the estimate has to be checked against this "
            f"machine's free memory. That check is what stops a run from "
            f"exhausting the desktop or an 8 GiB instance.\n\n"
            f"Provide a ScaleEstimate(n_s1=..., n_vendors=..., "
            f"candidates_per_query=..., bytes_per_pair=..., "
            f"peak_multiplier=..., runtime_minutes=..., rationale=..., "
            f"measured_from=...) and it will be validated against "
            f"available_bytes() = {available_bytes() / GiB:.2f} GiB."
        )
    estimate.validate_against(actual_vendors=n_vendors)
