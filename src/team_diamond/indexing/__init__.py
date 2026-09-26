"""indexing — Build and persist the multi-view indexes retrieval queries against.

STATUS: partially implemented. See the modules listed in ``__all__``.

Layer contract (AGENTS.md §10. Retrieval rules)
---------------------------------
precompute structures once; never scan the corpus per query

Depends on
----------
preprocessing

Must not
--------
perform per-query work; the matcher must not do hidden retrieval
"""

from __future__ import annotations

__all__: list[str] = []
