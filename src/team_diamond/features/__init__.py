"""Feature layer — turn candidate pairs into a numeric matrix.

Layer: features
See AGENTS.md §7 (layer separation), §11 (Features)

Three modules, split by *when* they run rather than by what they compute:

- :mod:`team_diamond.features.idf` builds corpus statistics. It runs once per
  corpus, before any pair exists, and it is the only module with a leakage
  surface.
- :mod:`team_diamond.features.similarity` holds the string primitives. Pure
  functions of two values, no state, no corpus.
- :mod:`team_diamond.features.pairs` assembles the matrix for a given pair set.

The split exists so the leakage boundary in :mod:`~team_diamond.features.idf` is
a single importable object with an explicit corpus argument, rather than a
convention spread across whichever feature function happens to need a
frequency.
"""

from __future__ import annotations

from team_diamond.features.idf import (
    UNSEEN_IDF,
    IdfTable,
    TokenRarity,
    build_token_rarity,
)
from team_diamond.features.pairs import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    PAIR_KEY_COLUMNS,
    build_pair_features,
    numeric_signature,
)
from team_diamond.features.similarity import (
    FUZZY_SCORERS,
    batch_fuzzy_ratios,
    clean_token_list,
    prefix_overlap,
    token_overlap_features,
)

__all__ = [
    "CATEGORICAL_COLUMNS",
    "FEATURE_COLUMNS",
    "FUZZY_SCORERS",
    "PAIR_KEY_COLUMNS",
    "UNSEEN_IDF",
    "IdfTable",
    "TokenRarity",
    "batch_fuzzy_ratios",
    "build_pair_features",
    "build_token_rarity",
    "clean_token_list",
    "numeric_signature",
    "prefix_overlap",
    "token_overlap_features",
]
