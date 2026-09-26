"""features — Turn a candidate pair into a feature vector.

STATUS: partially implemented. See the modules listed in ``__all__``.

Layer contract (AGENTS.md §11. Matching rules)
---------------------------------
be deterministic and side-effect free; build every statistic from the training fold only

Depends on
----------
preprocessing

Must not
--------
read labels; access anything unavailable at inference time
"""

from __future__ import annotations

__all__: list[str] = []
