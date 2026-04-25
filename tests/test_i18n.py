"""Tests for the bilingual UI helper :mod:`inqtrix.i18n`."""

from __future__ import annotations

import re
import string

import pytest

from inqtrix.i18n import MESSAGES, detect_ui_language, t


_PLACEHOLDER_RE = re.compile(r"\{(\w+)(?::[^}]*)?\}")


def _placeholders(template: str) -> set[str]:
    """Return the set of named placeholders used in ``template``."""
    return set(_PLACEHOLDER_RE.findall(template))


class TestDetectUILanguage:

    def test_empty_input_falls_back_to_english(self):
        assert detect_ui_language("") == "en"
        assert detect_ui_language("   ") == "en"

    def test_umlaut_is_strong_de_signal(self):
        assert detect_ui_language("Hallo Welt mit ä") == "de"
        assert detect_ui_language("Größenordnung") == "de"
        assert detect_ui_language("Übersicht") == "de"

    def test_eszett_is_strong_de_signal(self):
        assert detect_ui_language("Straße der Fragen") == "de"

    def test_de_stopwords_without_umlauts_detected(self):
        assert detect_ui_language("Wie ist der Stand der Forschung zu Quanten?") == "de"
        assert detect_ui_language("Was ist passiert?") == "de"

    def test_pure_english_returns_en(self):
        assert detect_ui_language("How does the system work?") == "en"
        assert detect_ui_language("What is the current state of research?") == "en"

    def test_unknown_language_falls_back_to_english(self):
        # French — no DE markers, no DE stopwords, so EN fallback.
        assert detect_ui_language("Quel est l'état actuel?") == "en"

    def test_proper_noun_only_returns_en(self):
        # No stopwords either way — fall back to EN.
        assert detect_ui_language("Bitcoin price 2026") == "en"


class TestTranslator:

    def test_de_state_returns_de_string(self):
        assert t({"language": "de"}, "classify_start") == "Analysiere Frage..."

    def test_en_state_returns_en_string(self):
        assert t({"language": "en"}, "classify_start") == "Analyzing question..."

    def test_unknown_language_falls_back_to_en(self):
        assert t({"language": "fr"}, "classify_start") == "Analyzing question..."

    def test_missing_language_falls_back_to_en(self):
        assert t({}, "classify_start") == "Analyzing question..."

    def test_empty_language_falls_back_to_en(self):
        assert t({"language": ""}, "classify_start") == "Analyzing question..."

    def test_none_state_falls_back_to_en(self):
        assert t(None, "classify_start") == "Analyzing question..."  # type: ignore[arg-type]

    def test_format_kwargs_substituted(self):
        msg = t({"language": "de"}, "plan_start", round=2, max_rounds=4)
        assert "Runde 2/4" in msg

        msg_en = t({"language": "en"}, "plan_start", round=2, max_rounds=4)
        assert "round 2/4" in msg_en

    def test_missing_kwargs_returns_template_unchanged(self):
        # Defensive: missing placeholder must not crash the run.
        msg = t({"language": "de"}, "plan_start")
        assert "{round}" in msg or "Runde" in msg

    def test_unknown_key_returns_key_itself(self):
        assert t({"language": "de"}, "this_key_does_not_exist") == "this_key_does_not_exist"

    def test_de_only_when_lowercased_matches(self):
        # Defensive: case-insensitive de detection.
        assert t({"language": "DE"}, "classify_start") == "Analysiere Frage..."
        assert t({"language": "De"}, "classify_start") == "Analysiere Frage..."

    def test_search_quality_summary_uses_format_specs(self):
        # Confirms ``{quality:.2f}`` style format specs work with kwargs.
        msg = t(
            {"language": "en"},
            "search_quality_summary",
            quality=0.5,
            claim_quality=0.3,
            coverage=80,
        )
        assert "0.50" in msg
        assert "0.30" in msg
        assert "80%" in msg


class TestMessagesIntegrity:

    @pytest.mark.parametrize("key", list(MESSAGES.keys()))
    def test_each_key_has_de_and_en(self, key):
        entry = MESSAGES[key]
        assert "de" in entry, f"Key '{key}' missing DE translation"
        assert "en" in entry, f"Key '{key}' missing EN translation"
        assert isinstance(entry["de"], str) and entry["de"], f"Key '{key}' DE empty"
        assert isinstance(entry["en"], str) and entry["en"], f"Key '{key}' EN empty"

    @pytest.mark.parametrize("key", list(MESSAGES.keys()))
    def test_de_and_en_have_matching_placeholders(self, key):
        # Drift-Schutz: if DE has {n} and {round}, EN must have the same set.
        de_phs = _placeholders(MESSAGES[key]["de"])
        en_phs = _placeholders(MESSAGES[key]["en"])
        assert de_phs == en_phs, (
            f"Key '{key}' has different placeholders between DE/EN: "
            f"DE={de_phs}, EN={en_phs}"
        )

    def test_no_format_spec_collisions(self):
        # Sanity: ``Formatter().parse`` accepts every template (no
        # malformed ``{...}`` literals).
        formatter = string.Formatter()
        for key, entry in MESSAGES.items():
            for lang, tmpl in entry.items():
                try:
                    list(formatter.parse(tmpl))
                except ValueError as exc:  # pragma: no cover — fail the test
                    pytest.fail(f"Key '{key}' lang '{lang}' template malformed: {exc}")
