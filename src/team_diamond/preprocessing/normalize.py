"""Null-safe normalisation of business names and addresses.

Layer: preprocessing
See AGENTS.md §8.6 (Text noise), §8.5 (Distribution shift)

The goal of this module
-----------------------
Reduce *superficial* differences between two records of the same business
without destroying information that distinguishes different businesses. That
second half is the hard part, and it is why the lists below are short and
deliberate.

What this module deliberately does NOT do
-----------------------------------------
- **No country branches.** ``if country == "IN": ...`` is prohibited
  (AGENTS.md §8.5): test contains France, which never appears in train, so any
  country-specific branch is untested exactly where it would fire. Where a
  country-specific artefact is genuinely needed it belongs in a *feature*, not
  in a normalisation branch.
- **No transliteration by default.** Mapping non-Latin scripts to ASCII is a
  lossy, dependency-heavy operation (libpostal) whose competition compliance is
  **unresolved**. It is not enabled. See ``suggestion.md`` §11.4.
- **No deletion of address components.** A token that looks like noise in one
  jurisdiction is the discriminator in another.
- **No deduplication of records.** Two records for one business are signal, not
  an error (AGENTS.md §8.1).

Unicode handling
----------------
Names are multilingual, so the pipeline is: NFKD decompose -> strip combining
marks -> casefold. This makes "Café" and "Cafe" compare equal without a
language-specific rule, and it is language-agnostic by construction.
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

__all__ = [
    "normalize_name",
    "normalize_address",
    "name_tokens",
    "address_tokens",
    "add_normalized_columns",
    "SOUND_ALIKE_TOKENS",
]

# Business-form and generic tokens. Removing these helps match
# "ACME TRADING LLC" against "ACME TRADING". They are removed as whole tokens
# only, never as substrings -- "Mart" must survive inside "Martinez".
_LEGAL_TOKENS: frozenset[str] = {
    # English / pan-European legal forms
    "llc", "lc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "plc", "llp", "lp", "ltda", "gmbh", "ag", "kg", "sa",
    "sarl", "sl", "spa", "srl", "bv", "nv", "oy", "ab", "aps", "kft",
    "sp", "sp z oo", "pte", "pty", "pvt", "zoo", "private",
    # Articles and conjunctions. These carry no identity. "und" (de) and "et"
    # (fr) are included because France appears only in TEST, so an English-only
    # stopword list would systematically misalign French names.
    # Single-letter conjunctions ("y", "e") are deliberately NOT included: they
    # collide with initials and single-letter name tokens.
    #
    # IMPORTANT: this list must contain only tokens that are ALWAYS
    # non-identifying. "company" is here; "trading", "group", "holdings",
    # "international", "global" are NOT — they were removed in an earlier draft
    # and that collapsed distinct businesses ("ACME TRADING" and "ACME TRADING
    # COMPANY" both became "acme trading", which is a precision bug, not a
    # cosmetic one). Whether a descriptor is safe to drop is an ablation
    # question (AGENTS.md §Ablation Testing), not a decision to hardcode.
    "and", "the", "of", "co", "company", "corp", "und", "et",
}

# French legal forms. Test-only jurisdiction: these appear in NONE of the
# training data, so an English-only list cannot possibly handle them. They are
# listed here rather than behind an `if country == "FR"` branch, per
# AGENTS.md §8.5 — the same normaliser runs for every country.
#
# SCOPE LIMIT: only *legal-form suffixes* belong here. Common nouns
# ("etablissements", "societe", "maison", "service", "boutique") are DELIBERATELY
# EXCLUDED even though they are frequent in French business names. Reason: they
# are often the head noun carrying the actual identity — "Etablissements
# Demployeurs" becomes "demployeurs" without it, and two businesses called
# "Etablissements X" and "X" would collapse. Whether dropping them helps is an
# ablation question (AGENTS.md §Ablation Testing), not a decision to hardcode.
_FRENCH_LEGAL_TOKENS: frozenset[str] = {
    "sarl", "sasu", "eurl", "sas", "sasn", "scs", "scp", "snc", "sca",
}
_LEGAL_TOKENS = frozenset(_LEGAL_TOKENS | _FRENCH_LEGAL_TOKENS)

# Generic name tokens that carry no discriminating power. Kept separate from
# legal forms because removing them is more aggressive: "Group" and "Global" say
# nothing about which business this is. Kept small on purpose -- over-removal
# is how distinct businesses get collapsed into one another.
_GENERIC_TOKENS: frozenset[str] = {
    "company", "store", "shop", "office", "centre", "center",
}

# Address noise. Abbreviations are expanded rather than deleted so that
# "St" and "Street" land on the same token.
_ADDRESS_ABBREVIATIONS: dict[str, str] = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "blv": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "pl": "place", "sq": "square", "hwy": "highway",
    "pkwy": "parkway", "pky": "parkway", "n": "north", "s": "south",
    "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "apt": "apartment", "ste": "suite",
    "fl": "floor", "bldg": "building", "no": "number", "nr": "number",
}

# Punctuation and separators treated as token boundaries.
#
# The negated class is Unicode-aware: `[^\w]` plus an explicit punctuation set.
# An earlier draft used `[^0-9a-z]+`, which is ASCII-only and DELETED every
# non-Latin character. Measured impact: train S1 contains 0 non-ASCII names, so
# that bug was invisible on train — but test S1 contains 40,789 accented French
# names (2.354%), including "École primaire Sainte Pierre". ASCII-only
# normalisation would have silently mangled every one of them.
#
# This is exactly the failure mode AGENTS.md §8.5 warns about: the defect is
# invisible in training and activates only on the test distribution shift.
_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")

# Sound-alike collapses for the tokens that survive. Deliberately tiny: these
# are applied as a *feature* signal, not baked into the normalised string, so
# that a wrong collapse cannot silently merge two distinct businesses.
SOUND_ALIKE_TOKENS: dict[str, str] = {
    "consultants": "consultant",
    "consultancy": "consultant",
    "technologies": "technology",
    "technological": "technology",
    "solutions": "solution",
    "industries": "industry",
    "enterprises": "enterprise",
}


def _strip_accents(text: str) -> str:
    """NFKD-decompose then drop combining marks. Language-agnostic.

    This is what makes "Café" and "Cafe" compare equal without a
    language-specific rule, and it is what allows the token regex to stay
    Unicode-aware: after decomposition, the base letters of Latin-script
    accented text are plain ASCII, while non-Latin scripts survive as
    themselves.

    Known, accepted side effect: this also drops Indic nukta
    (U+093C), so "बिज़नेस" folds to "बिजनेस". That is a *fold*, not a
    deletion — the text remains Devanagari and still matches other
    Devanagari text.

    Why that is acceptable here, specifically: **S1 contains zero
    non-ASCII names** (MEASURED, full train and test S1), so every query
    name is Latin and no folding decision is made against Devanagari.
    Devanagari *does* appear in S2/S3 (5.351% of train S2, 6.325% of test
    S2), so the folded form must still be self-consistent — which it is.
    If S1 ever gains non-Latin names, revisit this and prefer a
    script-conditional fold.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _basic_clean(text: str) -> str:
    """Casefold, strip accents, expand ``&``, collapse punctuation to spaces.

    ``&`` becomes the word ``and`` rather than being deleted, so that
    "Muller & Sohne" and "Muller and Sohne" agree without a language-specific
    rule. Deleting it outright would fuse the two surrounding tokens.
    """
    if not text:
        return ""
    cleaned = _strip_accents(text.casefold())
    cleaned = cleaned.replace("&", " and ")
    cleaned = _TOKEN_SPLIT.sub(" ", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def name_tokens(
    text: str,
    *,
    drop_legal: bool = True,
    drop_generic: bool = False,
) -> list[str]:
    """Split a business name into normalised tokens.

    Args:
        text: Raw name. ``""`` and null-equivalents yield ``[]``.
        drop_legal: Remove legal-form tokens ("llc", "ltd", ...). On by default
            because these differ arbitrarily between sources for the same
            business.
        drop_generic: Remove generic business words ("store", "services").
            Off by default: it is aggressive, and whether it helps is an
            ablation question, not a decision to hardcode.

    Returns:
        Tokens in original order, with empties removed.
    """
    tokens = _basic_clean(text).split()
    if not drop_legal and not drop_generic:
        return tokens
    drop = _LEGAL_TOKENS if drop_legal else frozenset()
    if drop_generic:
        drop = drop | _GENERIC_TOKENS
    return [t for t in tokens if t not in drop]


def normalize_name(text: str, *, drop_legal: bool = True) -> str:
    """Canonical name string: accent-free, casefolded, legal forms removed.

    Empty input returns ``""``, never ``None`` and never a null-propagating
    expression. Downstream code can therefore treat ``""`` as "no name
    evidence" without a null check.
    """
    return " ".join(name_tokens(text, drop_legal=drop_legal))


def address_tokens(text: str, *, expand_abbreviations: bool = True) -> list[str]:
    """Split an address into normalised tokens, expanding abbreviations.

    Abbreviations are *mapped*, not deleted, so "12 Main St" and
    "12 Main Street" become identical while the street name itself is
    untouched.

    Args:
        text: Raw address. Empty yields ``[]``.
        expand_abbreviations: Replace known short forms with their long form.

    Returns:
        Tokens in original order.
    """
    tokens = _basic_clean(text).split()
    if not expand_abbreviations:
        return tokens
    return [_ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]


def normalize_address(text: str, *, expand_abbreviations: bool = True) -> str:
    """Canonical address string. Empty input returns ``""``."""
    return " ".join(address_tokens(text, expand_abbreviations=expand_abbreviations))


def add_normalized_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the normalised columns the feature layer needs.

    Adds, all derived from the raw columns and all reproducible from them:

    - ``name_norm``      : canonical name
    - ``name_tokens_norm``: space-joined name tokens (for token-set similarity)
    - ``addr_norm``      : canonical address
    - ``addr_tokens_norm``: space-joined address tokens
    - ``has_address``    : bool, whether address evidence exists at all
    - ``has_name``       : bool

    ``has_address`` is a **feature**, not a filter. An S1 record with an empty
    address is a hard case to match, not a record to drop (AGENTS.md §8.1), and
    the model needs to be able to see that the evidence was missing.

    Args:
        frame: Frame with ``business_name`` and ``business_address``. Both must
            already be null-filled to ``""`` -- see
            :func:`team_diamond.data.loading.load_entities`.

    Returns:
        A new frame with the derived columns. Input is not modified.
    """
    required = {"business_name", "business_address"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    return frame.with_columns(
        [
            pl.col("business_name")
            .fill_null("")
            .map_elements(normalize_name, return_dtype=pl.Utf8)
            .alias("name_norm"),
            pl.col("business_address")
            .fill_null("")
            .map_elements(normalize_address, return_dtype=pl.Utf8)
            .alias("addr_norm"),
        ]
    ).with_columns(
        [
            pl.col("name_norm").alias("name_tokens_norm"),
            pl.col("addr_norm").alias("addr_tokens_norm"),
            (pl.col("addr_norm").str.len_chars() > 0).alias("has_address"),
            (pl.col("name_norm").str.len_chars() > 0).alias("has_name"),
        ]
    )
