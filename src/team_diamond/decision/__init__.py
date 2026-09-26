"""decision — Decide the emitted id list: *should* we emit this candidate?

STATUS: partially implemented. See the modules listed in ``__all__``.

Layer contract (AGENTS.md §13. Decision rules)
---------------------------------
be able to emit an empty list; account for margin and ambiguity, not just probability > 0.5

Depends on
----------
models

Must not
--------
perform hidden retrieval; change candidate generation
"""

from __future__ import annotations

__all__: list[str] = []
