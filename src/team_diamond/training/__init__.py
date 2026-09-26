"""training — Fit the pairwise matcher and its calibration.

STATUS: partially implemented. See the modules listed in ``__all__``.

Layer contract (AGENTS.md §16. Experiment discipline)
---------------------------------
record seed, commit, and config for every run

Depends on
----------
features

Must not
--------
touch the test split under any circumstances
"""

from __future__ import annotations

__all__: list[str] = []
