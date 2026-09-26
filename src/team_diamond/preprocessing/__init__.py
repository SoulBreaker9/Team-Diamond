"""preprocessing — Raw records -> canonical records (normalised, null-safe).

STATUS: partially implemented. See the modules listed in ``__all__``.

Layer contract (AGENTS.md §8.2 Schema and nulls)
---------------------------------
normalise names and addresses; convert nulls to a safe sentinel BEFORE any string processing; never invent values

Depends on
----------
data

Must not
--------
decide match likelihood; touch the label column
"""

from __future__ import annotations

__all__: list[str] = []
