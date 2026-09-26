"""retrieval — Produce the candidate set: *could* this be the match?

STATUS: partially implemented. See the modules listed in ``__all__``.

Layer contract (AGENTS.md §10. Retrieval rules)
---------------------------------
return candidates and retrieval evidence; emit no match probability

Depends on
----------
indexing

Must not
--------
filter candidates by a learned threshold — that is the decision layer
"""

from __future__ import annotations

__all__: list[str] = []
