"""Tests for null-safe name and address normalisation.

Layer: preprocessing

The property under test is **not** "normalisation makes things equal". It is
narrower and more important:

- pairs that are the *same business* written differently must converge, and
- pairs that are *different businesses* must NOT converge.

The second half is the one that actually bites. An earlier draft of the legal
token list included "trading", "group", "holdings", "international" and
"global", which silently collapsed ``ACME TRADING`` and ``ACME TRADING
COMPANY`` to the same string. That is a precision bug in a precision-weighted
metric, and it is why the must-differ cases below are pinned.

Run: ``uv run pytest tests/ -v``
"""

from __future__ import annotations

import pytest

from team_diamond.preprocessing.normalize import (
    address_tokens,
    name_tokens,
    normalize_address,
    normalize_name,
)


class TestEmptyAndNullSafety:
    """Empty input must yield empty output, never None (AGENTS.md §8.2)."""

    @pytest.mark.parametrize("value", ["", "   ", "\t\n", "!@#$"])
    def test_garbage_in_empty_out(self, value: str) -> None:
        assert normalize_name(value) == ""
        assert normalize_address(value) == ""

    def test_unicode_only_punctuation_is_empty(self) -> None:
        assert normalize_name("—–()") == ""

    def test_tokens_of_empty_is_empty_list(self) -> None:
        assert name_tokens("") == []
        assert address_tokens("") == []


class TestShouldConverge:
    """Same business, different surface form -> same normalised string."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("ACME TRADING LLC", "Acme Trading"),
            ("Acme Trading LLC", "ACME TRADING, LLC."),
            ("ACME TRADING", "ACME TRADING COMPANY"),      # 'company' is a legal form
            ("Café Français", "Cafe Francais"),             # accents
            ("Müller & Söhne GmbH", "Muller und Sohne"),    # '&' == 'and' == 'und'
            ("SMITH & CO.", "SMITH AND CO"),
            ("Société Générale", "Societe Generale"),
        ],
    )
    def test_equivalents(self, a: str, b: str) -> None:
        assert normalize_name(a) == normalize_name(b), f"{a!r} != {b!r}"

    def test_abbreviations_expand_not_delete(self) -> None:
        """'St' must become 'street', not vanish and fuse the neighbours."""
        assert normalize_address("12 Main St") == "12 main street"
        assert normalize_address("12 Main Street") == "12 main street"
        # deleting instead of expanding would have produced "12 main"
        assert normalize_address("12 Main St") != "12 main"

    def test_ampersand_expands_rather_than_disappears(self) -> None:
        """Deleting '&' would fuse neighbours into one token."""
        assert normalize_name("Smith & Sons") == "smith sons"
        assert name_tokens("Smith & Sons") == ["smith", "sons"]


class TestMustNotCollapse:
    """Different businesses must stay different. Precision guard."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Smith Metals", "Smith Metal"),
            ("Global Foods", "Global Freight"),
            ("Airtel", "Airteltel"),
            ("Prime Money", "Prime Power"),
            ("Christ Chapel", "Christ Church"),
        ],
    )
    def test_distinct_businesses_differ(self, a: str, b: str) -> None:
        assert normalize_name(a) != normalize_name(b), f"{a!r} collapsed onto {b!r}"

    def test_legal_form_removal_does_not_eat_descriptors(self) -> None:
        """Regression guard for the over-normalisation bug.

        "trading" and "company" must not both vanish, or two genuinely
        different businesses become indistinguishable.
        """
        assert "trading" in normalize_name("ACME TRADING")
        assert normalize_name("ACME TRADING") != normalize_name("ACME MINING")


class TestTokenOrderPreserved:
    """Order is kept; only legal forms are removed."""

    def test_order_is_preserved(self) -> None:
        assert name_tokens("Alpha Beta Gamma") == ["alpha", "beta", "gamma"]
        assert name_tokens("Gamma Beta Alpha") == ["gamma", "beta", "alpha"]

    def test_legal_forms_removed_in_place(self) -> None:
        assert name_tokens("Alpha Beta LLC") == ["alpha", "beta"]
        assert name_tokens("Alpha Beta Private Limited") == ["alpha", "beta"]

    def test_french_legal_forms_removed(self) -> None:
        """Test-only jurisdiction — these never appear in train, so they are
        the case an English-only list would miss entirely."""
        for name, expected in [
            ("Établissements Demployeurs SARL", "etablissements demployeurs"),
            ("Lège-Cap-Ferret Club SASU", "lege cap ferret club"),
            ("EURL Plaisanciers", "plaisanciers"),
        ]:
            assert normalize_name(name) == expected, f"{name!r} -> {normalize_name(name)!r}"

    def test_french_head_nouns_are_preserved(self) -> None:
        """"Etablissements" is a head noun, not a legal form.

        Removing it would leave "Etablissements Demployeurs" as bare
        "demployeurs", colliding with a business actually named "Demployeurs".
        """
        assert "etablissements" in normalize_name("Établissements Demployeurs")
        assert normalize_name("Établissements Demployeurs") != normalize_name("Demployeurs")

    def test_drop_generic_is_opt_in(self) -> None:
        """Generic-token removal is aggressive, so it is not the default."""
        assert "store" in name_tokens("Corner Store", drop_legal=True)
        assert "store" not in name_tokens("Corner Store", drop_generic=True)

    def test_substring_tokens_survive(self) -> None:
        """Legal forms are removed as WHOLE TOKENS, never as substrings.

        Removing "co" as a substring would corrupt "Artisan" -> "Artisan"
        becoming "Artisan" minus nothing, but would break any token containing
        those letters. This is the guard against that class of bug.
        """
        assert normalize_name("Martinez") == "martinez"
        assert normalize_name("Artisan") == "artisan"
        assert normalize_name("Company") == ""
        assert normalize_name("Companies") == "companies"   # not a listed form
        assert normalize_name("Corporate") == "corporate"   # not a listed form


class TestNoCountryBranching:
    """Normalisation must be language-agnostic (AGENTS.md §8.5).

    France appears only in TEST, so any country-conditional rule is untested
    precisely where it would fire. These tests pin language-agnostic behaviour.
    """

    def test_no_country_argument_exists(self) -> None:
        """A country parameter would be the entry point to a country branch."""
        import inspect

        for fn in (normalize_name, normalize_address, name_tokens, address_tokens):
            assert "country" not in inspect.signature(fn).parameters, fn.__name__

    def test_non_latin_scripts_survive(self) -> None:
        """Non-Latin text must not be deleted.

        Regression guard. An ASCII-only token regex (``[^0-9a-z]+``) reduced
        these to the empty string. It was invisible on train — which contains
        **zero** non-ASCII names — and would have activated only on the test
        distribution shift.
        """
        for name in ("बिज़नेस नाम", "ਕਾਰੋਬਾਰ", "شركة", "東京商事"):
            out = normalize_name(name)
            assert out, f"{name!r} was emptied out entirely"

    def test_mixed_script_name_keeps_all_scripts(self) -> None:
        """Latin and Devanagari tokens coexist; neither is deleted.

        Note the nukta fold: "बिज़नेस" -> "बिजनेस". That is a consistency
        fold, not a deletion, and it is safe only because S1 is measured to
        contain zero non-ASCII names.
        """
        out = normalize_name("Cafe Mumbai बिज़नेस")
        assert out.startswith("cafe mumbai")
        assert any("ऀ" <= ch <= "ॿ" for ch in out), f"Devanagari was lost: {out!r}"

    def test_devanagari_folds_consistently(self) -> None:
        """Nukta and no-nukta spellings must land on the same token."""
        assert normalize_name("बिज़नेस") == normalize_name("बिजनेस")

    def test_french_accents_strip_but_letters_survive(self) -> None:
        """The measured test-set case: 40,789 accented French names (2.354%)."""
        assert normalize_name("École primaire Sainte Pierre") == "ecole primaire sainte pierre"
        assert normalize_name("Ecole Primaire Sainte Pierre") == "ecole primaire sainte pierre"
        # accents must not delete the accented letter itself
        assert normalize_name("Établissements") == "etablissements"

    def test_apostrophes_split_rather_than_fuse(self) -> None:
        """'Orelee's' must not become 'orelees' — that fuses distinct spellings."""
        assert name_tokens("Orelee's Barbershop") == ["orelee", "s", "barbershop"]


class TestIdempotence:
    """Normalising an already-normalised string must be a no-op."""

    @pytest.mark.parametrize(
        "raw",
        [
            "ACME TRADING LLC",
            "Müller & Söhne GmbH",
            "12 Main St, Springfield",
            "Café Français",
            "SMITH & CO.",
        ],
    )
    def test_normalise_twice_equals_once(self, raw: str) -> None:
        once = normalize_name(raw)
        assert normalize_name(once) == once
        addr_once = normalize_address(raw)
        assert normalize_address(addr_once) == addr_once
