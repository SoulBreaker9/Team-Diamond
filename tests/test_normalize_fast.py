"""The fast path must be byte-identical to the reference implementation.

Layer: preprocessing
See AGENTS.md §16 (Ablation — a second code path needs its own proof)

Why this file exists
--------------------
``normalize_fast`` is a **reimplementation** of ``normalize`` for throughput. A
reimplementation that quietly disagrees with its original is worse than no fast
path at all, because every downstream measurement inherits the divergence while
still looking plausible.

This suite was written after the divergence was found, not before. Running it
caught four real bugs, of which one was serious:

- **The reference was shredding Indic text.** ``normalize_name("प्राइवेट")``
  returned ``'पर इव ट'`` because Python's ``re`` excludes combining marks from
  ``\\w`` while Rust's (Polars) includes them. The two engines tokenise
  differently, and the reference was the destructive one. 5.4% of train S2 and
  6.3% of test S2 carry Devanagari.
- Legal forms at position 0 were not removed by the fast path, because the
  padded-literal-replace trick only padded the right-hand side.
- A fold table derived from one corpus silently under-covered another, leaving
  French accents unfolded.
- Marks were being deleted on the fast path after the tokenizer had been fixed,
  which is no longer correct behaviour.

Run: ``uv run pytest tests/ -v``
"""

from __future__ import annotations

import unicodedata

import polars as pl
import pytest

from team_diamond.preprocessing.normalize import (
    _LATIN_FOLD,
    _SEPARATOR_TABLE,
    _basic_clean,
    add_normalized_columns,
    normalize_address,
    normalize_name,
)
from team_diamond.preprocessing.normalize_fast import (
    derive_fold_maps,
    normalize_columns,
    require_fold_coverage,
)

DATASET = "/home/hetm/Desktop/Hackathon/6ab10eb3b23ba_student_resource/student_resource/dataset"
NORM_COLUMNS = ("name_norm", "addr_norm", "has_address", "has_name")

# Cases chosen to hit each rule rather than to be realistic.
EDGE_CASES = [
    # Latin accents must fold
    "Café Français", "Müller & Söhne GmbH", "Établissements Demployeurs SARL",
    "Lège-Cap-Ferret Club SASU", "Société Générale", "SMITH & CO.",
    # Indic must survive INTACT, matras included
    "प्राइवेट लिमिटेड", "सुप्रीम अल्फा फूड्स", "राम मार्केटिंग", "கடை ஒன்று",
    "ಕರ್ನಾಟಕ ಅಂಗಡಿ", "తెలుగు వ్యాపారం", "ਪੰਜਾਬੀ ਵਪਾਰ", "ગુજરાતી વ્યાપાર",
    "বাংলা ব্যবসা", "ଓଡ଼ିଆ ବ୍ୟବସା",
    # Legal form at position 0 (the padding bug)
    "LLC Dental Partners", "SRL Rossi", "Private Holdings Ltd",
    # Symbols that must act as separators, not fold
    "12°C Shop", "½ Price Store", "Aº Outlet",
    # Ligature with no ASCII decomposition: must NOT become "oe"
    "Œuvre Atelier", "Cœur Café",
    # Structural
    "", "   ", "---", "Orelee's Barbershop", "B+ Retail Inc", "A", "12",
]
EDGE_ADDRESSES = [
    "12 Main St", "12 Main Street", "FLAT 402, PUNE, Maharashtra", "",
    "9 Boulevard de la République, Roubaix", "Door No.254, 5Th Main",
    "DOOR NO 113 , NEW NO.73", "NO 49 CO SUMIT KUMAR SO MUNNA LAL",
    "प्राइवेट लिमिटेड रोड", "Box #42, Ã\x80\x98Câ\x80\x99 Layout",
]


def _edge_frame() -> pl.DataFrame:
    n = max(len(EDGE_CASES), len(EDGE_ADDRESSES))
    names = (EDGE_CASES + EDGE_CASES)[-n:] if n > len(EDGE_CASES) else EDGE_CASES
    addrs = (EDGE_ADDRESSES * 4)[:n]
    return pl.DataFrame(
        {"business_name": names, "business_address": addrs, "entity_id": [str(i) for i in range(n)]}
    )


class TestFastPathMatchesReference:
    """The contract: identical output, column for column."""

    @pytest.mark.parametrize("drop_legal", [True, False])
    @pytest.mark.parametrize("drop_generic", [True, False])
    @pytest.mark.parametrize("expand_abbreviations", [True, False])
    def test_edge_cases_all_flag_combinations(
        self, drop_legal: bool, drop_generic: bool, expand_abbreviations: bool
    ) -> None:
        frame = _edge_frame()
        # The reference must get the same flags. Comparing a drop_legal=False
        # fast path against a drop_legal=True reference is a test bug, not a bug
        # in either implementation.
        ref = add_normalized_columns(
            frame, drop_legal=drop_legal, drop_generic=drop_generic,
            expand_abbreviations=expand_abbreviations,
        )
        fast = normalize_columns(
            frame,
            drop_legal=drop_legal,
            drop_generic=drop_generic,
            expand_abbreviations=expand_abbreviations,
        )
        for col in NORM_COLUMNS:
            expected, got = ref[col].to_list(), fast[col].to_list()
            bad = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
            assert not bad, f"{col}: {len(bad)} diff(s), first: {bad[0]}"

    def test_edge_case_names(self) -> None:
        """Spot-check the values themselves, not just their equality."""
        frame = _edge_frame()
        out = dict(zip(frame["business_name"].to_list(), normalize_columns(frame)["name_norm"].to_list()))
        assert out["Café Français"] == "cafe francais"
        assert out["Établissements Demployeurs SARL"] == "etablissements demployeurs"
        # Indic preserved exactly, matras and all
        # Devanagari legal forms are NOT stripped: _LEGAL_TOKENS is an ASCII
        # list, and adding Hindi/Tamil legal-form lists would be exactly the
        # per-language branching AGENTS.md §8.5 rules out. Left as-is.
        assert out["प्राइवेट लिमिटेड"] == "प्राइवेट लिमिटेड"
        assert out["सुप्रीम अल्फा फूड्स"] == "सुप्रीम अल्फा फूड्स"
        # legal form removed even at position 0
        assert out["LLC Dental Partners"] == "dental partners"
        assert out["SRL Rossi"] == "rossi"
        # ligature NOT expanded
        assert out["Œuvre Atelier"] == "œuvre atelier"
        # symbol becomes a separator
        assert out["12°C Shop"] == "12 c shop"


class TestIndicIsNotDestroyed:
    """The bug that motivated rewriting the tokenizer."""

    def test_reference_preserves_devanagari_words(self) -> None:
        out = normalize_name("प्राइवेट")
        assert out == "प्राइवेट", f"Devanagari was shredded into {out!r}"

    def test_reference_preserves_other_indic_scripts(self) -> None:
        for text in ("கடை ஒன்று", "ಕರ್ನಾಟಕ", "ਪੰਜਾਬੀ", "ગુજરાતી", "বাংলা", "ଓଡ଼ିଆ"):
            out = normalize_name(text)
            assert out == text, f"{text!r} -> {out!r}"

    def test_word_count_is_preserved(self) -> None:
        """A word must not gain internal spaces."""
        raw = "सुप्रीम अल्फा फूड्स प्राइवेट लिमिटेड"
        out = normalize_name(raw)
        assert len(out.split()) == len(raw.split()) == 5

    def test_matra_categories_are_word_chars(self) -> None:
        """U+093E (Mc) must not be a separator. It is not for Rust, and must not
        be for us either."""
        for cp in (0x093E, 0x093F, 0x0940, 0x0947, 0x094D, 0x0902, 0x093C):
            char = chr(cp)
            assert ord(char) not in _SEPARATOR_TABLE, f"U+{cp:04X} treated as a separator"


class TestTokenizersAgree:
    """Python's `re` and Rust's regex must define "word character" identically."""

    def test_every_ascii_char_agrees(self) -> None:
        for cp in range(0x00, 0x80):
            char = chr(cp)
            # Rust: \w == [\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}].
            # Whitespace is excluded from the comparison: Rust's [^\w]+ turns it
            # into a space, and we keep it and collapse runs afterwards. Both
            # routes end in the same single-space-separated string, so treating
            # whitespace as agreeing-by-construction is correct, whereas
            # claiming it agrees definitionally is not.
            ours = (ord(char) in _SEPARATOR_TABLE) and not char.isspace()
            rust = (
                char.isascii()
                and not (char.isalnum() or char == "_")
                and not char.isspace()
            )
            assert ours == rust, f"U+{cp:04X} {char!r}"

    def test_every_present_non_ascii_char_agrees(self) -> None:
        """The real corpus, not a synthetic one."""
        try:
            frame = pl.read_csv(
                f"{DATASET}/train/train_source2.tsv",
                separator="\t", quote_char=None, empty_string_is_null=False,
                n_rows=400_000,
            )
        except FileNotFoundError:
            pytest.skip("dataset not available")
        present: set[str] = set()
        for series in ("business_name", "business_address"):
            for value in frame[series].unique().to_list():
                present.update(value or "")
        marks = [c for c in present if unicodedata.category(c).startswith("M")]
        assert marks, "expected combining marks in the corpus"
        for char in marks:
            assert ord(char) not in _SEPARATOR_TABLE, f"{char!r} (U+{ord(char):04X}) is a separator"


class TestFoldTableCoverage:
    def test_derived_table_folds_what_is_present(self) -> None:
        frame = _edge_frame()
        table = derive_fold_maps(
            pl.concat([frame["business_name"], frame["business_address"]])
        )
        for char, base in _LATIN_FOLD.items():
            # The table is keyed on the POST-lowercase inventory, because that is
            # what the fold chain sees. "Café" arrives as "cafe" + U+00E3, so the
            # table must hold U+00E3 and must NOT bother with U+00C3.
            lowered = char.lower()
            if lowered not in _LATIN_FOLD:
                continue
            present = (
                frame["business_name"].str.contains(lowered, literal=True).any()
                or frame["business_address"].str.contains(lowered, literal=True).any()
            )
            if present:
                assert (
                    table.get(lowered) == _LATIN_FOLD[lowered]
                ), f"{lowered!r} present but not folded"

    def test_derived_table_excludes_ascii_self_maps(self) -> None:
        """A table full of "a"->"a" entries would be ~50 wasted passes per column."""
        table = derive_fold_maps(pl.Series("x", ["Café"]))
        assert table == {"é": "e"}
        assert not any(k.isascii() for k in table)

    def test_require_coverage_rejects_a_train_derived_table_on_french(self) -> None:
        """The silent-divergence guard must actually fire."""
        french = pl.Series("x", ["Lège-Cap-Ferret", "École Primaire"])
        train_only: dict[str, str] = {}
        with pytest.raises(AssertionError, match="missing"):
            require_fold_coverage(french, train_only, context="french test")

    def test_require_coverage_passes_when_derived(self) -> None:
        frame = _edge_frame()
        table = derive_fold_maps(pl.concat([frame["business_name"], frame["business_address"]]))
        require_fold_coverage(frame["business_name"], table, context="edge frame")

    def test_fold_only_applies_to_letters(self) -> None:
        """Symbols must never be in the table, or they would fold to letters."""
        for char in ("°", "½", "©", "€"):
            assert char not in _LATIN_FOLD


class TestIdempotenceBothPaths:
    @pytest.mark.parametrize(
        "raw", ["Café Français", "प्राइवेट लिमिटेड", "LLC Dental", "Œuvre", "12°C Shop"]
    )
    def test_reference_is_idempotent(self, raw: str) -> None:
        once = normalize_name(raw)
        assert normalize_name(once) == once
        addr = normalize_address(raw)
        assert normalize_address(addr) == addr

    def test_fast_path_is_idempotent(self) -> None:
        frame = _edge_frame()
        once = normalize_columns(frame)
        # Drop the raw columns first, otherwise renaming onto them collides.
        round_tripped = once.drop("business_name", "business_address").rename(
            {"name_norm": "business_name", "addr_norm": "business_address"}
        )
        twice = normalize_columns(round_tripped)
        for col in NORM_COLUMNS:
            assert once[col].to_list() == twice[col].to_list(), col


class TestBasicCleanContract:
    def test_never_returns_none_or_null(self) -> None:
        for value in ("", "  ", "---", "\x00", "\u00a0"):
            assert isinstance(_basic_clean(value), str)

    def test_collapses_whitespace(self) -> None:
        assert _basic_clean("  a   b  ") == "a b"
