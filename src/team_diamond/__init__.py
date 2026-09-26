"""Team Diamond — Amazon ML Challenge 2026.

Multi-source business entity resolution, scored by entity-level macro F0.5.

Layer contract
--------------
The package is a strict pipeline. Each subpackage owns one layer and must not
reach into another's responsibilities:

    data -> preprocessing -> indexing -> retrieval -> features
         -> training -> models -> decision -> evaluation

    data -> preprocessing -> retrieval -> features -> matching -> decision
         -> evaluation/submission

Three questions, three different owners (AGENTS.md §7):

* *Could* this candidate be the match?      -> retrieval
* *How likely* is this candidate to match?  -> models
* *Should* we emit this candidate?          -> decision

Status
------
This is a scaffold. Implemented and tested: the evaluation metric, dataset
loading, and text normalisation. Declared boundaries without implementation:
indexing, retrieval, features, training, models, decision, pipeline.

Nothing here has produced a competition score. See ``suggestion.md`` for
proposals and ``experiments/registry.csv`` for measured evidence.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
