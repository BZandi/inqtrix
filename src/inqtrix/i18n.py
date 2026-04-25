"""Bilingual (DE/EN) progress messages and UI-language detection.

The agent's user-facing progress strings (streamed as SSE chunks during a
research run) need to follow the language of the question. Detection runs
in two stages:

1. A fast local heuristic (:func:`detect_ui_language`) used by
   :func:`inqtrix.state.initial_state` so the very first progress event
   ("Analysiere Frage..." / "Analyzing question...") is already in the
   right language.
2. The LLM-based classification in :func:`inqtrix.nodes.classify` later
   overwrites ``state["language"]`` with a more precise reading.

Translations are looked up via :func:`t`. The fallback chain is:
``state["language"] == "de"`` → DE, anything else (including ``""``,
``"en"``, ``"fr"``, ...) → EN.
"""

from __future__ import annotations

import re
from typing import Any

# Umlauts and ß are an unambiguous DE signal.
_DE_CHARS = re.compile(r"[äöüÄÖÜß]")

# Frequent DE stopwords (short, lowercased tokens).
# Bewusst nur Wörter, die im EN selten/nie vorkommen.
_DE_STOPWORDS = frozenset({
    "der", "die", "das", "den", "dem", "des",
    "und", "oder", "aber", "denn", "weil", "wenn", "dass", "ob",
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "sein", "hat", "haben", "hatte", "hatten",
    "ein", "eine", "einen", "einer", "eines", "einem",
    "kein", "keine", "keinen", "keiner", "keines", "keinem",
    "wie", "warum", "wieso", "weshalb", "wann", "wer", "wen", "wem",
    "wo", "was", "welche", "welcher", "welches", "wieviel", "wieviele",
    "ich", "du", "er", "es", "wir", "ihr", "sie",
    "mich", "dich", "ihn", "uns", "euch",
    "mein", "meine", "meiner", "meines", "meinem", "meinen",
    "dein", "deine", "deiner", "deines", "deinem", "deinen",
    "sein", "seine", "seiner", "seines", "seinem", "seinen",
    "ihre", "ihrer", "ihres", "ihrem", "ihren",
    "unser", "unsere", "unserer", "unseres", "unserem", "unseren",
    "euer", "eure", "eurer", "eures", "eurem", "euren",
    "nicht", "noch", "schon", "auch", "nur", "mehr", "viel", "viele",
    "mit", "ohne", "bei", "von", "vom", "zum", "zur", "zu",
    "fuer", "für", "über", "ueber", "unter", "vor", "nach", "auf", "an",
    "im", "ins", "am", "ans", "beim",
})

# English stopword tie-breakers when no DE marker fires.
_EN_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "because",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did", "doing",
    "how", "why", "when", "where", "what", "which", "who", "whom", "whose",
    "this", "that", "these", "those",
    "with", "without", "of", "for", "to", "from", "by", "at", "in", "on",
    "i", "you", "he", "she", "we", "they", "it",
    "my", "your", "his", "her", "our", "their", "its",
    "should", "would", "could", "can", "may", "might", "must",
    "not", "no", "any", "some", "all", "many", "much",
})

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def detect_ui_language(text: str) -> str:
    """Return ``"de"`` or ``"en"`` based on a fast local heuristic.

    1. Empty input → ``"en"`` (fallback per spec).
    2. Any DE-specific character (umlaut or ß) → ``"de"``.
    3. Tokenize lowercased text on word boundaries and count
       stopword hits in either language; ``de_hits > en_hits`` → ``"de"``,
       otherwise → ``"en"``.
    """
    if not text:
        return "en"
    if _DE_CHARS.search(text):
        return "de"
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return "en"
    de_hits = sum(1 for tok in tokens if tok in _DE_STOPWORDS)
    en_hits = sum(1 for tok in tokens if tok in _EN_STOPWORDS)
    if de_hits > en_hits:
        return "de"
    return "en"


# ---------------------------------------------------------------------------
# Translation table
# ---------------------------------------------------------------------------

MESSAGES: dict[str, dict[str, str]] = {
    # ---- Sprache / Recency / Type-Labels (eingebettet in classify-Hints) ----
    "lang_name_de": {"de": "Deutsch", "en": "German"},
    "lang_name_en": {"de": "Englisch", "en": "English"},
    "lang_name_fr": {"de": "Franzoesisch", "en": "French"},
    "lang_name_es": {"de": "Spanisch", "en": "Spanish"},
    "lang_name_it": {"de": "Italienisch", "en": "Italian"},
    "lang_name_pt": {"de": "Portugiesisch", "en": "Portuguese"},
    "recency_hour": {"de": "letzte Stunde", "en": "last hour"},
    "recency_day": {"de": "heute", "en": "today"},
    "recency_week": {"de": "diese Woche", "en": "this week"},
    "recency_month": {"de": "diesen Monat", "en": "this month"},
    "recency_label": {
        "de": "Aktualitaet: {label}",
        "en": "Recency: {label}",
    },
    "type_academic": {"de": "Akademisch", "en": "Academic"},
    "type_news": {"de": "Nachrichten", "en": "News"},
    "search_in_lang": {
        "de": "Suche auf {lang_name}",
        "en": "Search in {lang_name}",
    },

    # ---- compose_answer_sections (nodes.py) ----
    "compose_section_token_limit": {
        "de": (
            "Warnung: Abschnitt '{heading}' hat Token-Limit erreicht "
            "(finish_reason={finish_reason}, request_max_tokens={request_max}, "
            "completion_tokens={completion})"
        ),
        "en": (
            "Warning: section '{heading}' hit token limit "
            "(finish_reason={finish_reason}, request_max_tokens={request_max}, "
            "completion_tokens={completion})"
        ),
    },
    "compose_section_truncation": {
        "de": "Warnung: Abschnitt '{heading}' zeigt Trunkierungsanzeichen: {reasons}",
        "en": "Warning: section '{heading}' shows truncation signals: {reasons}",
    },
    "compose_aborted": {
        "de": (
            "Antwort-Synthese abgebrochen: {n} aufeinanderfolgende leere "
            "Sections (zuletzt '{heading}')"
        ),
        "en": (
            "Answer composition aborted: {n} consecutive empty sections "
            "(last '{heading}')"
        ),
    },

    # ---- classify (nodes.py) ----
    "classify_start": {
        "de": "Analysiere Frage...",
        "en": "Analyzing question...",
    },
    "classify_warning_hint": {
        "de": "Hinweis: {warning}",
        "en": "Note: {warning}",
    },
    "classify_followup_detected": {
        "de": "Vertiefungsfrage erkannt — nutze bisherige Recherche",
        "en": "Follow-up question detected — using prior research",
    },
    "classify_new_topic": {
        "de": "Neues Thema erkannt — starte frische Recherche",
        "en": "New topic detected — starting fresh research",
    },
    "classify_direct_answer": {
        "de": "Direktantwort ohne Websuche",
        "en": "Direct answer without web search",
    },
    "classify_search_required": {
        "de": "Websuche erforderlich{hints}",
        "en": "Web search required{hints}",
    },
    "classify_subquestions": {
        "de": "Frage in {n} Teilfragen zerlegt: {subs}",
        "en": "Question split into {n} sub-questions: {subs}",
    },
    "classify_failed": {
        "de": "Klassifikation fehlgeschlagen ({label}) — nutze konservative Defaults",
        "en": "Classification failed ({label}) — using conservative defaults",
    },

    # ---- plan (nodes.py) ----
    "plan_start": {
        "de": "Plane Suchanfragen (Runde {round}/{max_rounds})...",
        "en": "Planning search queries (round {round}/{max_rounds})...",
    },
    "plan_failed": {
        "de": "Planung fehlgeschlagen ({label}) — verwende Original-Frage als Fallback-Query",
        "en": "Planning failed ({label}) — using original question as fallback query",
    },
    "plan_new_queries": {
        "de": "{added} neue Suchanfragen generiert{strategy_hint}",
        "en": "{added} new search queries generated{strategy_hint}",
    },
    "plan_no_more_queries": {
        "de": "Keine neuen Suchanfragen moeglich — beende Recherche",
        "en": "No new search queries possible — ending research",
    },

    # ---- search (nodes.py) ----
    "search_start": {
        "de": "Durchsuche {n} Quellen (Runde {round}/{max_rounds})...",
        "en": "Searching {n} sources (round {round}/{max_rounds})...",
    },
    "search_failed_n_of_m": {
        "de": "{failed} von {total} Suchanfragen fehlgeschlagen (leere Ergebnisse zurueckgegeben)",
        "en": "{failed} of {total} searches failed (returned empty results)",
    },
    "search_empty_n_of_m": {
        "de": "{empty} von {total} Suchanfragen lieferten keine Ergebnisse",
        "en": "{empty} of {total} searches returned no results",
    },
    "summarize_failed": {
        "de": "{failed} von {total} Zusammenfassungen fehlgeschlagen (Rohtext verwendet)",
        "en": "{failed} of {total} summaries failed (using raw text)",
    },
    "claim_extract_failed": {
        "de": "{failed} von {total} Claim-Extraktionen fehlgeschlagen (uebersprungen)",
        "en": "{failed} of {total} claim extractions failed (skipped)",
    },
    "search_sources_processed": {
        "de": "{n} Quellen verarbeitet, {citations} Referenzen gesammelt",
        "en": "{n} sources processed, {citations} references collected",
    },
    "search_related_questions": {
        "de": "{n} verwandte Fragen aus Suchergebnissen erkannt",
        "en": "{n} related questions detected from search results",
    },
    "search_quality_summary": {
        "de": (
            "Quellenqualitaet {quality:.2f}, Claim-Qualitaet {claim_quality:.2f}, "
            "Aspektabdeckung {coverage}%"
        ),
        "en": (
            "Source quality {quality:.2f}, claim quality {claim_quality:.2f}, "
            "aspect coverage {coverage}%"
        ),
    },

    # ---- evaluate (nodes.py) ----
    "evaluate_start": {
        "de": "Bewerte Informationsqualitaet (nach Runde {round}/{max_rounds})...",
        "en": "Evaluating information quality (after round {round}/{max_rounds})...",
    },
    "evaluate_confidence_missing": {
        "de": "Bewertung unvollstaendig (CONFIDENCE-Feld fehlt) — nutze Default 5",
        "en": "Evaluation incomplete (CONFIDENCE field missing) — using default 5",
    },
    "evaluate_failed": {
        "de": "Qualitaetsbewertung fehlgeschlagen ({label}) — konservative Confidence-Begrenzung",
        "en": "Quality evaluation failed ({label}) — using conservative confidence cap",
    },
    "research_finished": {
        "de": "Recherche abgeschlossen (Confidence: {conf}/10, Runden: {round})",
        "en": "Research finished (Confidence: {conf}/10, Rounds: {round})",
    },
    "confidence_continue": {
        "de": "Confidence {conf}/10 — weitere Recherche noetig",
        "en": "Confidence {conf}/10 — further research needed",
    },
    "min_rounds_continue": {
        "de": "min_rounds={min_rounds} noch nicht erreicht (aktuell {round}); setze Recherche fort",
        "en": "min_rounds={min_rounds} not yet reached (currently {round}); continuing research",
    },

    # ---- answer (nodes.py) ----
    "answer_start": {
        "de": "Formuliere Antwort (nach {n} {round_label})...",
        "en": "Formulating answer (after {n} {round_label})...",
    },
    "answer_round_singular": {"de": "Runde", "en": "round"},
    "answer_round_plural": {"de": "Runden", "en": "rounds"},
    "answer_fallback_model": {
        "de": "Finale Antwort fehlgeschlagen — Fallback-Modell {model}",
        "en": "Final answer failed — fallback model {model}",
    },
    "answer_incomplete_detected": {
        "de": "Finale Antwort als unvollstaendig erkannt",
        "en": "Final answer detected as incomplete",
    },
    "answer_links_removed": {
        "de": "{n} nicht-zugelassene Links entfernt",
        "en": "{n} non-allowed links removed",
    },

    # ---- graph.py ----
    "run_aborted": {
        "de": "Recherche abgebrochen: {exc}",
        "en": "Research aborted: {exc}",
    },
    "run_failed": {
        "de": "Recherche fehlgeschlagen: {exc}",
        "en": "Research failed: {exc}",
    },
    "chat_mode_failed": {
        "de": "Chat-Modus fehlgeschlagen: {exc}",
        "en": "Chat mode failed: {exc}",
    },

    # ---- stop_criteria.py ----
    "contradictions_severe": {
        "de": "Schwere Widersprueche erkannt — Confidence stark begrenzt",
        "en": "Severe contradictions detected — confidence strongly capped",
    },
    "contradictions_light": {
        "de": "Leichte Widersprueche erkannt (z.B. Datumsabweichungen)",
        "en": "Minor contradictions detected (e.g. date mismatches)",
    },
    "irrelevant_filtered": {
        "de": "{n} irrelevante Quellen gefiltert",
        "en": "{n} irrelevant sources filtered",
    },
    "multiple_explanations": {
        "de": "Mehrere moegliche Erklaerungen erkannt",
        "en": "Multiple possible explanations detected",
    },
    "falsification_released": {
        "de": "Confidence wieder hoch — Falsifikations-Modus deaktiviert",
        "en": "Confidence recovered — falsification mode deactivated",
    },
    "falsification_started": {
        "de": "Niedrige Evidenz — starte Falsifikations-Recherche",
        "en": "Low evidence — starting falsification research",
    },
    "stagnation_premise_false": {
        "de": (
            "Umfangreiche Recherche abgeschlossen — "
            "Praemisse der Frage wahrscheinlich falsch"
        ),
        "en": (
            "Extensive research complete — "
            "question premise likely false"
        ),
    },
    "utility_weak_continue": {
        "de": (
            "Informationsgewinn stagniert, aber Evidenzlage "
            "noch zu schwach — suche weiter"
        ),
        "en": (
            "Information gain stagnating, but evidence "
            "still too weak — searching further"
        ),
    },
    "utility_stop": {
        "de": "Informationsgewinn stagniert — beende Recherche",
        "en": "Information gain stagnating — ending research",
    },
    "plateau_stop": {
        "de": "Confidence {conf}/10 stabil — Recherche abgeschlossen",
        "en": "Confidence {conf}/10 stable — research complete",
    },

    # ---- streaming.py SSE error chunks (user-facing) ----
    "sse_request_timeout": {
        "de": "\n---\n\n⚠️ **Fehler bei der Recherche:** Request-Timeout erreicht",
        "en": "\n---\n\n⚠️ **Research error:** request timeout reached",
    },
    "sse_agent_error": {
        "de": "\n---\n\n⚠️ **Fehler bei der Recherche:** {err}",
        "en": "\n---\n\n⚠️ **Research error:** {err}",
    },
}


def t(state: dict[str, Any], key: str, **kwargs: Any) -> str:
    """Translate ``key`` to the language stored in ``state["language"]``.

    Returns the localized string, with ``**kwargs`` substituted via
    :meth:`str.format`. Fallback chain:
    ``state["language"] == "de"`` → DE; anything else → EN.

    If ``key`` is not registered in :data:`MESSAGES`, the key itself is
    returned (defensive — surfaces typos as visible bugs in the stream).
    If ``kwargs`` are missing for placeholders in the template, the raw
    template is returned untransformed (no exception).
    """
    raw_lang = (state.get("language") or "").lower() if state else ""
    lang = "de" if raw_lang == "de" else "en"
    entry = MESSAGES.get(key)
    if entry is None:
        return key
    template = entry.get(lang) or entry.get("en") or entry.get("de") or key
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template
    return template
