"""pipeline — Wire the layers together for a reproducible end-to-end run.

STATUS: partially implemented. See the modules listed in ``__all__``.

Layer contract (AGENTS.md §5. Source of truth and layout)
---------------------------------
keep every stage separately measurable; never collapse layers for speed

Depends on
----------
all

Must not
--------
contain business logic — that belongs in the layer that owns it
"""

from __future__ import annotations

__all__: list[str] = []
