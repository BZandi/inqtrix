"""Narration lines (plan B2): grounded German prose from run artifacts.

The transcript renders these payloads verbatim, so the builders must be
deterministic, bounded, and never invent content — and the sanitizer's
allowlist must pass every schema field through untouched.
"""

from __future__ import annotations

from inqtrix.agents.narration import (
    NARRATION_EVENT,
    discovery_narration,
    first_sentences,
    plan_narration,
    section_narration,
    synthesis_narration,
    task_narration,
)
from inqtrix.agents.phase_models import DiscoveryResult
from inqtrix.runtime_logging import sanitize_event_payload


def _discovery(facts: int, gaps: int) -> DiscoveryResult:
    return DiscoveryResult.model_validate(
        {
            "known_facts": [
                {"fact": f"Fakt {i}", "source": "doc:d1#0", "fresh": True}
                for i in range(facts)
            ],
            "gaps": [
                {
                    "gap_id": f"g{i}",
                    "kind": "missing",
                    "description": "Aktuelle Umsetzungslage fehlt.",
                    "recommended_capability": "web_research",
                    "suggested_queries": [],
                    "blocking": False,
                }
                for i in range(gaps)
            ],
            "questions_for_user": [],
            "sufficient_to_plan": True,
        }
    )


def test_first_sentences_cuts_at_sentence_boundary() -> None:
    text = "Erster Satz. Zweiter Satz ist deutlich laenger. Dritter Satz."
    assert first_sentences(text, limit=30) == "Erster Satz."
    # Whole text under the limit passes through unchanged.
    assert first_sentences("Kurz.", limit=30) == "Kurz."
    # A single overlong sentence falls back to a word boundary + marker.
    long = "Wort " * 40
    cut = first_sentences(long, limit=50)
    assert cut.endswith("...")
    assert len(cut) <= 54


def test_discovery_narration_counts_facts_and_gaps() -> None:
    text = discovery_narration(_discovery(facts=3, gaps=2))
    assert "3 belegte Fakten" in text
    assert "2 offene Luecken" in text
    assert "Aktuelle Umsetzungslage fehlt." in text
    assert discovery_narration(None) == ""
    empty = discovery_narration(_discovery(facts=0, gaps=0))
    assert "weder" in empty


def test_plan_and_task_and_synthesis_lines() -> None:
    assert plan_narration("Erst intern sichten, dann Web.", 6) == (
        "Mein Plan (6 Aufgaben): Erst intern sichten, dann Web."
    )
    assert plan_narration("", 4) == "Ich schlage einen Plan mit 4 Aufgaben vor."
    assert task_narration("Interne Sammlung", "Zwoelf Treffer. Details ...") \
        .startswith("Interne Sammlung: Zwoelf Treffer.")
    assert task_narration("Titel", "") == ""
    assert synthesis_narration("Markteinschätzung", 4) == (
        "Ich schreibe jetzt das Memo 'Markteinschätzung' mit 4 Abschnitten."
    )
    assert section_narration("Kernaussagen") == (
        "Abschnitt 'Kernaussagen' geschrieben."
    )


def test_sanitizer_allowlist_passes_narration_payload_through() -> None:
    """The strict schema keeps every field the transcript renders and
    drops surprise keys (gotcha #9 discipline)."""
    payload = {
        "narration_id": "n-plan-1",
        "kind": "plan",
        "text": "Mein Plan (6 Aufgaben): Erst intern sichten.",
        "phase": "planning",
        "final": True,
        "surprise_secret": "nope",
    }
    clean = sanitize_event_payload(NARRATION_EVENT, dict(payload))
    assert clean == {
        "narration_id": "n-plan-1",
        "kind": "plan",
        "text": "Mein Plan (6 Aufgaben): Erst intern sichten.",
        "phase": "planning",
        "final": True,
    }


def test_sanitizer_bounds_narration_independently_from_evidence_text() -> None:
    clean = sanitize_event_payload(
        NARRATION_EVENT,
        {
            "narration_id": "n-bounded",
            "kind": "intent",
            "text": "Wort " * 200,
            "phase": "execution",
        },
    )

    assert clean["text"].endswith(" ...")
    assert len(clean["text"]) <= 400
