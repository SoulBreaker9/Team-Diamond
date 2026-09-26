"""retrieval — Produce the candidate set: *could* this be the match?

STATUS: partially implemented. See the modules listed in ``__all__``.

Implemented
-----------
``keys``
    Blocking-key builders. These are the mechanism that makes candidate
    generation tractable; see that module for the key contract and for why
    every strategy returns a long ``(row_id, key)`` frame.

Not yet implemented
-------------------
Candidate assembly, ranking, and per-strategy caps. Nothing here ranks,
scores, or decides; that is the decision layer's job (AGENTS.md §7).

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

__all__: list[str] = ["keys"]
