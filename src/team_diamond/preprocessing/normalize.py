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

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
# Tokenisation is done by TRANSLATION, not by a negated regex character class,
# and that choice is load-bearing.
#
# Python's `re` defines `\w` as `[a-zA-Z0-9_]` plus Unicode *alphanumerics*,
# which EXCLUDES combining marks. Rust's regex — which backs Polars — defines
# `\w` as `[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]`, which INCLUDES
# them. The two engines therefore disagree, and the Python side is the one that
# destroys data.
#
# Concretely, a negated class shreds Devanagari into individual letters:
#
#     normalize_name("प्राइवेट")   # was -> 'पर इव ट'
#
# because the matras (U+093E, U+093F, ...) are category Mc, are not `\w` in
# Python, and are not removed by the accent stripper (see `_fold_latin`). The
# word is destroyed. 5.4% of train S2 and 6.3% of test S2 carry Devanagari, so
# this silently deletes a large slice of the vendor-side name signal.
#
# The fix is to define "word character" the way the fast path does — by Unicode
# category — and to use one definition on both sides. `_SEPARATOR_TABLE` is that
# definition: anything not in L*, M*, N*, Pc, or whitespace becomes a space.
#
# Kept categories and why:
#   L*  letters            M*  combining marks (must stay attached to the base)
#   N*  digits             Pc  connector punctuation, i.e. "_"
# A mark is never a separator, so an Indic syllable survives as one token.
def _is_word_char(char: str) -> bool:
    """True for characters that must never act as a token boundary.

    Defined to match **Rust's** ``\\w``, not Python's, because the fast path runs
    on Rust and the two must agree exactly. Rust's definition is::

        \\w == [\\p{Alphabetic}\\p{M}\\p{Nd}\\p{Pc}\\p{Join_Control}]

    which expands to the categories below. The subtlety is the numbers: Rust
    keeps ``Nd`` (decimal digits) but **not** ``No`` or ``Nf``. Python's
    ``str.isalnum`` keeps all of them, so a naive port keeps ``\\u00bd`` (``1/2``)
    as a token while the fast path discards it as a separator. That is how
    "1/2 Price Store" became "1/2 price" on the reference side and "price" on the
    fast side. The symbol reading is the better one anyway, so the reference is
    corrected to match Rust.

    Known residual gap, accepted and tested rather than papered over: Rust's
    ``\\p{Alphabetic}`` also contains ``Other_Alphabetic`` marks, which Python
    exposes no flag for. Keeping all of ``M*`` is a superset of Rust there, so
    the reference is marginally more conservative (it glues slightly more
    together). ``tests/test_normalize_fast.py`` checks agreement on every
    character actually present in the corpus rather than on theory.

    Args:
        char: A single character.

    Returns:
        Whether the character belongs inside a token.
    """
    if char.isspace():
        return True
    category = unicodedata.category(char)
    if category in ("Nd", "Pc", "Nl", "Cf"):
        # Nd decimal digits, Pc connector punctuation ("_"), Nl letter numbers,
        # Cf format controls (Rust includes Join_Control, U+200C/U+200D).
        return True
    return category[0] in ("L", "M")


_SEPARATOR_TABLE: dict[int, str] = {
    code_point: " "
    for code_point in range(0x0000, 0x10000)
    if not _is_word_char(chr(code_point))
}
_WHITESPACE = re.compile(r"\s+")


def _build_latin_fold() -> dict[str, str]:
    """Letters whose NFKD decomposition is exactly one ASCII letter -> that letter.

    Scanned once at import. Kept as a table rather than a per-character NFKD
    call because :mod:`team_diamond.preprocessing.normalize_fast` needs the
    identical table, and two independently derived tables would eventually
    disagree.
    """
    table: dict[str, str] = {}
    for code_point in range(0x0000, 0x10000):
        char = chr(code_point)
        # ASCII letters decompose to themselves. Including them would make the
        # fast path perform ~50 no-op string passes per column, and would make
        # `require_fold_coverage` report ASCII characters as "missing folds",
        # which is noise rather than a real coverage gap.
        if char.isascii() or not unicodedata.category(char).startswith("L"):
            continue
        stripped = "".join(
            ch
            for ch in unicodedata.normalize("NFKD", char)
            if not unicodedata.combining(ch)
        )
        if len(stripped) == 1 and stripped.isascii() and stripped.isalpha():
            table[char] = stripped
    return table


_LATIN_FOLD: dict[str, str] = _build_latin_fold()


def _fold_latin(text: str) -> str:
    """Fold accented **letters** to ASCII. Language-agnostic by construction.

    The rule is a Unicode property, not a per-language list: *if a character is
    a letter whose canonical decomposition is a single ASCII letter, replace it
    with that letter.* Everything else is left untouched.

    Leaving everything else alone is the important half. An earlier version used
    ``NFKD`` + "drop anything with ``unicodedata.combining(ch) != 0``", which is
    wrong twice over:

    - It is not the same as "drop combining marks". ``unicodedata.combining``
      returns the canonical combining *class*, which is 0 for many real marks
      (U+0947 DEVANAGARI VOWEL SIGN E is category Mn but returns 0), so marks
      slipped through and were then split as separators.
    - Even a correct mark-strip is destructive outside Latin. Devanagari matras
      carry the vowel of the syllable; dropping them merges distinct words.

    Consequences of the narrower rule, all intentional:

    - ``Café`` -> ``cafe`` and ``Müller`` -> ``muller`` still work.
    - ``प्राइवेट`` is preserved exactly, matras included.
    - ``Œ`` (U+0152, 169 occurrences) has no ASCII decomposition, so it stays
      ``Œ`` rather than becoming ``oe``. Whether to add an explicit ligature map
      is an open ablation item, not a silent default.
    - ``°`` and ``½`` are symbols rather than letters, so they are not folded
      and instead act as token separators.
    """
    return "".join(_LATIN_FOLD.get(char, char) for char in text)

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
    """Casefold, fold Latin accents, expand ``&``, tokenise, collapse whitespace.

    ``&`` becomes the word ``and`` rather than being deleted, so that
    "Muller & Sohne" and "Muller and Sohne" agree without a language-specific
    rule. Deleting it outright would fuse the two surrounding tokens.

    Tokenisation translates every non-word character to a space (see
    ``_SEPARATOR_TABLE``) rather than applying a negated regex class, because
    Python and Rust disagree on what ``\\w`` includes and the disagreement
    destroys Indic text.
    """
    if not text:
        return ""
    cleaned = _fold_latin(text.casefold())
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.translate(_SEPARATOR_TABLE)
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


def normalize_name(
    text: str, *, drop_legal: bool = True, drop_generic: bool = False
) -> str:
    """Canonical name string: accent-free, casefolded, legal forms removed.

    Empty input returns ``""``, never ``None`` and never a null-propagating
    expression. Downstream code can therefore treat ``""`` as "no name
    evidence" without a null check.

    ``drop_generic`` exists so this function has the same flag surface as
    :func:`team_diamond.preprocessing.normalize_fast.normalize_columns`. An
    asymmetry there is not cosmetic: it means the equivalence test cannot cover
    a flag combination, and a divergence in that combination would be invisible.
    """
    return " ".join(
        name_tokens(text, drop_legal=drop_legal, drop_generic=drop_generic)
    )


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


def add_normalized_columns(
    frame: pl.DataFrame,
    *,
    drop_legal: bool = True,
    drop_generic: bool = False,
    expand_abbreviations: bool = True,
) -> pl.DataFrame:
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
            .map_elements(
                lambda t: normalize_name(
                    t, drop_legal=drop_legal, drop_generic=drop_generic
                ),
                return_dtype=pl.Utf8,
            )
            .alias("name_norm"),
            pl.col("business_address")
            .fill_null("")
            .map_elements(
                lambda t: normalize_address(t, expand_abbreviations=expand_abbreviations),
                return_dtype=pl.Utf8,
            )
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
