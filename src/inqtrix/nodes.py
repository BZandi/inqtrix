"""State machine node functions for the research agent.

Each node takes the state dict plus providers and strategies as keyword
arguments.  The five nodes correspond to the five phases of the research
loop: classify, plan, search, evaluate, answer.

Extracted from ``_original_agent.py`` and adapted to use the provider /
strategy / settings abstractions defined in the ``inqtrix`` package.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import OpenAIError

from inqtrix.domains import LANG_NAMES, LOW_QUALITY_DOMAINS
from inqtrix.exceptions import AgentRateLimited, AgentTimeout, AnthropicAPIError, BedrockAPIError
from inqtrix.json_helpers import parse_json_object, parse_json_string_list
from inqtrix.prompts import (
    EVALUATE_FORMAT_SUFFIX,
    build_answer_system_prompt,
)
from inqtrix.providers.base import ProviderContext, _check_deadline
from inqtrix.settings import AgentSettings
from inqtrix.state import append_iteration_log, emit_progress, track_tokens
from inqtrix.strategies import StrategyContext
from inqtrix.text import is_none_value, tokenize
from inqtrix.urls import (
    count_allowed_links,
    extract_urls,
    normalize_url,
    sanitize_answer_links,
    today,
)

log = logging.getLogger("inqtrix")


# ======================================================================= #
# 1. classify
# ======================================================================= #


def classify(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Analyse the incoming question and seed the initial research state.

    Args:
        s: Mutable AgentState-compatible dict. Reads the question,
            follow-up markers, and deadline; writes language, query type,
            risk flags, aspect hints, and reset markers for new topics.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for risk scoring and downstream
            claim/context handling.
        settings: Agent behavior settings used for risk escalation and
            timeout handling.

    Returns:
        The mutated state dict with classification results.

    Raises:
        AgentRateLimited: Propagated when the upstream classification
            model hard-fails on rate limiting.

    Example:
        >>> classify(state, providers=providers, strategies=strategies, settings=settings)
        {'query_type': 'general', 'language': 'de', ...}
    """
    emit_progress(s, "Analysiere Frage...")
    _t0 = time.monotonic()
    _followup_seeded = bool(s.get("_is_followup"))
    s["risk_score"] = strategies.risk_scoring.score(s["question"])
    s["high_risk"] = s["risk_score"] >= settings.high_risk_score_threshold
    classify_model = (
        providers.llm.models.reasoning_model
        if (s["high_risk"] and settings.high_risk_classify_escalate)
        else providers.llm.models.effective_classify_model
    )

    # Follow-up context for the classify prompt
    _followup_prompt_ctx = ""
    if s.get("_is_followup") and s.get("_prev_question"):
        _followup_prompt_ctx = (
            f"=== KONTEXT: VORHERIGE RECHERCHE ===\n"
            f"Vorherige Frage: {s['_prev_question']}\n"
            f"Vorherige Antwort (Auszug): {s['_prev_answer'][:300]}\n"
            f"Der Nutzer stellt jetzt eine neue Frage. Bestimme ob sie sich "
            f"thematisch auf die vorherige Recherche bezieht (Follow-up/Vertiefung) "
            f"oder ein komplett neues Thema ist.\n\n"
        )

    try:
        _check_deadline(s["deadline"])
        d = providers.llm.complete(
            f"Heutiges Datum: {today()}\n\n"
            f"{_followup_prompt_ctx}"
            f"Analysiere diese Frage in ZWEI Teilen:\n\n"
            f"=== TEIL 1: KLASSIFIKATION ===\n"
            f"1. Braucht sie eine aktuelle Websuche? "
            f"(Aktuelle Ereignisse, Preise, Statistiken, neue Technologien, "
            f"veraenderliche Fakten → IMMER Suche)\n"
            f"2. In welcher Sprache ist die Frage geschrieben?\n"
            f"3. In welcher Sprache findet man die besten Suchergebnisse? "
            f"(z.B. Programmierung/Tech/Wissenschaft → oft Englisch, "
            f"lokale Themen/Politik/Recht → Sprache der Frage)\n"
            f"4. Wie aktuell muessen die Ergebnisse sein?\n"
            f"   - NONE: Zeitlose Fakten (Mathematik, Geschichte, Definitionen)\n"
            f"   - MONTH: Aktuelle Entwicklungen, neueste Versionen\n"
            f"   - WEEK: Nachrichten der letzten Tage, aktuelle Ereignisse\n"
            f"   - DAY: Breaking News, Live-Daten, heutige Ereignisse\n"
            f"5. Welcher Suchtyp passt am besten?\n"
            f"   - GENERAL: Standard-Websuche\n"
            f"   - ACADEMIC: Wissenschaftliche Fragen, Studien, Papers\n"
            f"   - NEWS: Nachrichten, aktuelle Ereignisse, Meldungen\n\n"
            f"=== TEIL 2: DEKOMPOSITION ===\n"
            f"Zerlege die Frage in 1-3 unabhaengige Teilfragen fuer gezieltere Recherche.\n"
            f"Wenn die Frage einfach genug ist, gib sie unveraendert als einzelne Teilfrage zurueck.\n\n"
            f"ZEITLICHE VERANKERUNG:\n"
            f"- Interpretiere relative Zeitangaben (vor kurzem, neulich, letztens, kuerzlich) "
            f"immer relativ zum heutigen Datum ({today()}).\n"
            f"- 'vor kurzem' = letzte 2-4 Wochen vor dem heutigen Datum.\n"
            f"- FUEGE KEINE konkreten Jahreszahlen ein die du nicht aus der Frage kennst.\n"
            f"- Statt '2025' oder '2026' zu raten, nutze 'recent' oder das Datum.\n\n"
            f"Frage: {s['question']}\n\n"
            f"Antworte EXAKT in diesem Format:\n"
            f"DECISION: SEARCH oder DIRECT\n"
            f"LANGUAGE: Sprachcode der Frage (z.B. de, en, fr)\n"
            f"SEARCH_LANGUAGE: Sprachcode fuer optimale Suche (z.B. en, de)\n"
            f"RECENCY: NONE oder DAY oder WEEK oder MONTH\n"
            f"TYPE: GENERAL oder ACADEMIC oder NEWS\n"
            + (f"FOLLOWUP: YES oder NO (bezieht sich die Frage auf die vorherige Recherche?)\n"
               if s.get("_is_followup") else "")
            + f"SUB_QUESTIONS: JSON Array von 1-3 Teilfragen als Strings",
            deadline=s["deadline"],
            model=classify_model,
            state=s,
        )
        s["done"] = bool(re.search(r"DECISION:\s*DIRECT", d, re.IGNORECASE))

        # Extract language
        m_lang = re.search(r"LANGUAGE:\s*(\w+)", d)
        s["language"] = m_lang.group(1).strip().lower()[:2] if m_lang else "de"

        m_search_lang = re.search(r"SEARCH_LANGUAGE:\s*(\w+)", d)
        s["search_language"] = m_search_lang.group(
            1).strip().lower()[:2] if m_search_lang else s["language"]

        # Extract recency requirement
        m_recency = re.search(r"RECENCY:\s*(\w+)", d)
        recency_raw = m_recency.group(1).strip().upper() if m_recency else "NONE"
        recency_map = {"DAY": "day", "WEEK": "week", "MONTH": "month", "NONE": ""}
        s["recency"] = recency_map.get(recency_raw, "")

        # Extract query type
        m_type = re.search(r"TYPE:\s*(\w+)", d)
        type_raw = m_type.group(1).strip().upper() if m_type else "GENERAL"
        type_map = {"ACADEMIC": "academic", "NEWS": "news", "GENERAL": "general"}
        s["query_type"] = type_map.get(type_raw, "general")

        # Fallback: keyword-based detection in case LLM misses academic questions
        if s["query_type"] != "academic":
            q_lower = s["question"].lower()
            academic_keywords = (
                "paper", "studie", "study", "preprint", "doi",
                "publikation", "publication", "arxiv", "veroeffentlich",
                "publish", "journal", "conference", "peer-review",
            )
            if any(kw in q_lower for kw in academic_keywords):
                _prev_type = s["query_type"]
                s["query_type"] = "academic"
                log.info("TRACE classify: type override %s->academic (keyword fallback)", _prev_type)

        # Extract sub-questions (part 2 of the combined call)
        m_sub = re.search(r"SUB_QUESTIONS:\s*(\[.*)", d, re.DOTALL)
        sub_q_text = m_sub.group(1) if m_sub else ""

        # Follow-up detection: LLM answered the FOLLOWUP field
        if s.get("_is_followup"):
            m_followup = re.search(r"FOLLOWUP:\s*(YES|NO|JA|NEIN)", d, re.IGNORECASE)
            is_followup = bool(m_followup and m_followup.group(1).upper() in ("YES", "JA"))

            if is_followup:
                # Deepening question: keep seeded data, merge new aspects
                log.info("TRACE classify: follow-up detected, keeping seeded research data")
                emit_progress(s, "Vertiefungsfrage erkannt — nutze bisherige Recherche")
                # New sub-questions for the follow-up question
                s["sub_questions"] = parse_json_string_list(
                    sub_q_text, fallback=[s["question"]], max_items=3)
                # Add new aspects from the follow-up to the existing ones
                new_aspects = strategies.risk_scoring.derive_required_aspects(
                    s["question"], s["query_type"])
                existing = set(s.get("required_aspects", []))
                for asp in new_aspects:
                    if asp not in existing:
                        s["required_aspects"].append(asp)
                        s["uncovered_aspects"].append(asp)
                # Recompute aspect coverage with extended aspects
                if s["required_aspects"]:
                    covered = len(s["required_aspects"]) - len(s["uncovered_aspects"])
                    s["aspect_coverage"] = max(0.0, covered / len(s["required_aspects"]))
            else:
                # New topic: reset all seeded data
                log.info("TRACE classify: new topic detected, clearing seeded data")
                emit_progress(s, "Neues Thema erkannt — starte frische Recherche")
                s["sub_questions"] = parse_json_string_list(
                    sub_q_text, fallback=[s["question"]], max_items=3)
                s["required_aspects"] = strategies.risk_scoring.derive_required_aspects(
                    s["question"], s["query_type"])
                s["uncovered_aspects"] = list(s["required_aspects"])
                s["aspect_coverage"] = 0.0
                s["all_citations"] = []
                s["context"] = []
                s["consolidated_claims"] = []
                s["claim_ledger"] = []
                s["queries"] = []
                s["source_tier_counts"] = {
                    "primary": 0, "mainstream": 0, "stakeholder": 0, "unknown": 0, "low": 0}
                s["source_quality_score"] = 0.0
                s["claim_status_counts"] = {"verified": 0, "contested": 0, "unverified": 0}
                s["claim_quality_score"] = 0.0
                s["_prev_question"] = ""
                s["_prev_answer"] = ""
            # One-shot flag -- reset after evaluation
            s["_is_followup"] = False
        else:
            # Standard path: no follow-up, normal initialisation
            s["sub_questions"] = parse_json_string_list(
                sub_q_text, fallback=[s["question"]], max_items=3)
            s["required_aspects"] = strategies.risk_scoring.derive_required_aspects(
                s["question"], s["query_type"])
            s["uncovered_aspects"] = list(s["required_aspects"])
            s["aspect_coverage"] = 0.0

        # Trace logging
        log.info(
            "TRACE classify: decision=%s lang=%s search_lang=%s recency=%s type=%s sub_q=%d risk=%d high_risk=%s model=%s",
            "DIRECT" if s["done"] else "SEARCH",
            s["language"], s["search_language"], s["recency"] or "NONE", s["query_type"],
            len(s["sub_questions"]), s["risk_score"], s["high_risk"], classify_model,
        )

        if s["done"]:
            emit_progress(s, "Direktantwort ohne Websuche")
        else:
            hints: list[str] = []
            if s["search_language"] != s["language"]:
                hints.append(
                    f"Suche auf {LANG_NAMES.get(s['search_language'], s['search_language'])}")
            if s["recency"]:
                recency_labels = {"day": "heute",
                                  "week": "diese Woche", "month": "diesen Monat"}
                hints.append(f"Aktualitaet: {recency_labels.get(s['recency'], s['recency'])}")
            if s["query_type"] != "general":
                type_labels = {"academic": "Akademisch", "news": "Nachrichten"}
                hints.append(type_labels.get(s["query_type"], s["query_type"]))

            hint_str = f" ({', '.join(hints)})" if hints else ""
            emit_progress(s, f"Websuche erforderlich{hint_str}")

            if len(s["sub_questions"]) > 1:
                emit_progress(s, f"Frage in {len(s['sub_questions'])} Teilfragen zerlegt")
    except AgentRateLimited:
        raise
    except (OpenAIError, AgentTimeout, AnthropicAPIError, BedrockAPIError):
        # Fail-safe: on classification error do NOT fall back to direct answer.
        # Conservatively continue researching with robust defaults.
        s["done"] = False
        s["language"] = s.get("language") or "de"
        s["search_language"] = s.get("search_language") or s["language"]
        s["query_type"] = strategies.risk_scoring.infer_query_type(s["question"])
        s["recency"] = "month" if s["query_type"] == "news" else ""
        s["sub_questions"] = [s["question"]]
        s["required_aspects"] = strategies.risk_scoring.derive_required_aspects(
            s["question"], s["query_type"])
        s["uncovered_aspects"] = list(s["required_aspects"])
        s["aspect_coverage"] = 0.0

    append_iteration_log(s, {
        "node": "classify",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "decision": "DIRECT" if s["done"] else "SEARCH",
        "question_length": len(s.get("question", "")),
        "history_length": len(s.get("history", "")),
        "lang": s["language"],
        "search_lang": s["search_language"],
        "recency": s["recency"] or "NONE",
        "type": s["query_type"],
        "sub_questions": s["sub_questions"],
        "sub_question_count": len(s["sub_questions"]),
        "risk_score": s["risk_score"],
        "high_risk": s["high_risk"],
        "followup_seeded": _followup_seeded,
        "model": classify_model,
        "required_aspects": s.get("required_aspects", []),
    }, testing_mode=settings.testing_mode)
    return s


# ======================================================================= #
# 2. plan
# ======================================================================= #


def plan(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Generate the next batch of research queries.

    Args:
        s: Mutable AgentState-compatible dict. Reads the current round,
            required aspects, gaps, and related questions; writes new
            planned queries and related planning metadata.
        providers: Active LLM and search providers.
        strategies: Runtime strategies used to derive quality terms and
            other planning hints.
        settings: Agent behavior settings controlling round limits and
            first-round breadth.

    Returns:
        The mutated state dict with updated query planning.

    Raises:
        AgentRateLimited: Propagated when the planning model hard-fails
            on upstream rate limiting.

    Example:
        >>> plan(state, providers=providers, strategies=strategies, settings=settings)
        {'queries': ['gkv reform 2026', ...], ...}
    """
    emit_progress(s, f"Plane Suchanfragen (Runde {s['round'] + 1}/{settings.max_rounds})...")
    _t0 = time.monotonic()
    try:
        _check_deadline(s["deadline"])

        # Build prompt with sub-questions and gap info
        sub_q_info = ""
        if s["sub_questions"]:
            sub_q_info = f"Teilfragen: {json.dumps(s['sub_questions'], ensure_ascii=False)}\n"

        required_info = ""
        if s.get("required_aspects"):
            required_info = (
                f"Pflichtaspekte fuer die Antwort:\n"
                f"{json.dumps(s['required_aspects'], ensure_ascii=False)}\n"
            )

        uncovered_info = ""
        if s.get("uncovered_aspects"):
            uncovered_info = (
                f"Noch NICHT ausreichend abgedeckt:\n"
                f"{json.dumps(s['uncovered_aspects'], ensure_ascii=False)}\n"
                f"Mindestens eine Query MUSS gezielt die offenen Aspekte abdecken.\n"
            )

        gap_info = ""
        if s["gaps"]:
            gap_info = f"Noch fehlende Informationen: {s['gaps']}\n"

        # Use related questions from Perplexity as inspiration
        related_info = ""
        if s["related_questions"]:
            related_info = (
                f"Verwandte Fragen (von der Suchmaschine vorgeschlagen):\n"
                f"{json.dumps(s['related_questions'][:5], ensure_ascii=False)}\n"
                f"Nutze diese als Inspiration, aber kopiere sie nicht woertlich.\n"
            )

        # Determine search language
        search_lang = s.get("search_language", s.get("language", "de"))
        lang_instruction = ""
        if search_lang == "en":
            lang_instruction = (
                "WICHTIG: Formuliere die Suchqueries auf ENGLISCH, "
                "da fuer dieses Thema englische Quellen besser sind.\n"
            )
        elif search_lang != s.get("language", "de"):
            lang_instruction = (
                f"WICHTIG: Formuliere die Suchqueries auf "
                f"{LANG_NAMES.get(search_lang, search_lang)}.\n"
            )

        # Perspective diversity (STORM-inspired)
        perspective_instruction = ""
        if s["round"] > 0:
            perspective_instruction = (
                "PERSPEKTIV-DIVERSITAET: Betrachte das Thema aus einer ANDEREN Perspektive "
                "als die bisherigen Queries. Moegliche Perspektiven:\n"
                "- Technisch/Mechanistisch: Wie funktioniert es genau?\n"
                "- Praktisch/Anwendung: Wie wird es eingesetzt?\n"
                "- Kritisch/Limitierungen: Was sind die Grenzen und Probleme?\n"
                "- Vergleichend: Wie steht es im Vergleich zu Alternativen?\n"
                "- Historisch/Kontext: Wie hat es sich entwickelt?\n"
                "- Aktuell/Zukunft: Was sind die neuesten Entwicklungen?\n\n"
            )

        # Alternative hypothesis search
        alternative_instruction = ""
        if s["round"] == 1:
            alternative_instruction = (
                "WICHTIG — ALTERNATIVE HYPOTHESEN:\n"
                "Mindestens eine deiner Queries MUSS nach ALTERNATIVEN Ereignissen/Antworten suchen.\n"
                f"Heutiges Datum: {today()}. Die bisherigen Ergebnisse koennten ein AELTERES Ereignis "
                "beschreiben, das nicht das ist was der Nutzer meint.\n"
                "Suche gezielt nach dem AKTUELLSTEN passenden Ereignis — "
                "z.B. 'stock market crash AI February 2026' oder 'latest AI selloff this week'.\n"
                "Wenn die Frage 'vor kurzem/neulich/letztens' sagt, muss mindestens eine Query "
                f"explizit den Zeitraum der letzten 2-4 Wochen vor {today()} abdecken.\n\n"
            )

        # Competing events: force targeted comparison queries
        competing_instruction = ""
        competing = s.get("competing_events", "")
        if competing:
            competing_instruction = (
                f"WICHTIG — KONKURRIERENDE ERKLAERUNGEN:\n"
                f"Die Evaluierung hat folgende moegliche Ereignisse/Antworten identifiziert:\n"
                f"{competing}\n\n"
                f"Deine Queries MUESSEN gezielt klaeren welches Ereignis AKTUELLER und RELEVANTER ist.\n"
                f"Suche nach DIREKTEN Vergleichen, exakten Daten, und spezifischen Details "
                f"die eine eindeutige Zuordnung ermoeglichen.\n"
                f"Mindestens eine Query muss das NEUESTE der konkurrierenden Ereignisse "
                f"mit explizitem Datum/Zeitraum suchen.\n\n"
            )

        # Aggressive reformulation on low confidence after round 1
        reformulation_instruction = ""
        if s["round"] >= 2 and s.get("final_confidence", 5) <= 4:
            reformulation_instruction = (
                "ACHTUNG: Die bisherigen Suchen haben kaum relevante Ergebnisse geliefert "
                f"(Confidence: {s.get('final_confidence', '?')}/10 nach {s['round']} Runden).\n"
                "Du MUSST die Suchstrategie GRUNDLEGEND aendern:\n"
                "1. HINTERFRAGE DIE PRAEMISSE: Vielleicht existiert das Beschriebene gar nicht, "
                "oder der Nutzer verwechselt/vermischt verschiedene Dinge. "
                "Suche stattdessen nach dem was TATSAECHLICH existiert.\n"
                "2. Suche nach dem BREITEREN Thema: z.B. statt 'DeepSeek Paper ueber X' "
                "suche 'DeepSeek neueste Papers 2026 Liste aller Veroeffentlichungen'\n"
                "3. Suche nach den beteiligten Personen/Organisationen und deren neueste Arbeiten\n"
                "4. Formuliere KOMPLETT um — andere Begriffe, Synonyme, uebergeordnete Kategorien\n"
                "5. Suche nach Diskussionen/Nachrichten UEBER das Thema statt nach dem Thema selbst\n\n"
            )

        # Falsification mode (FVA-RAG-inspired)
        falsification_instruction = ""
        if s.get("falsification_triggered", False):
            falsification_instruction = (
                "FALSIFIKATIONS-MODUS AKTIV:\n"
                "Die bisherige Recherche hat wiederholt KEINE ueberzeugenden Belege fuer die "
                "Behauptung/Praemisse in der Frage gefunden. Jetzt suchen wir gezielt nach "
                "GEGEN-EVIDENZ um die Praemisse zu testen.\n\n"
                "Mindestens 2 deiner 3 Queries MUESSEN Falsifikations-Queries sein:\n"
                "- '[Behauptung] debunked' oder '[Behauptung] myth'\n"
                "- '[Thema] does not exist' oder '[Thema] never happened'\n"
                "- '[Thema] hoax' oder '[Thema] refuted'\n"
                "- '[Thema] misinformation' oder '[Thema] false claim'\n\n"
                "Die dritte Query MUSS nach dem TATSAECHLICH existierenden naechstliegenden "
                "Sachverhalt suchen: z.B. statt 'Gemini eingestellt' -> "
                "'Google Gemini aktueller Status 2026' oder 'Was hat [Organisation] "
                "TATSAECHLICH veroeffentlicht?'\n\n"
                "ZIEL: Entweder finden wir Belege dass die Praemisse falsch ist "
                "(-> hochkonfidente Antwort 'existiert nicht'), oder wir finden doch "
                "noch den richtigen Sachverhalt.\n\n"
            )

        # Follow-up context
        followup_plan_ctx = ""
        if s.get("_prev_question"):
            followup_plan_ctx = (
                f"KONTEXT: Dies ist eine Vertiefungsfrage zu: {s['_prev_question']}\n"
                f"Bisherige Recherche enthielt {len(s['context'])} Informationsbloecke "
                f"und {len(s['all_citations'])} Quellen.\n"
                f"Generiere Queries die GEZIELT die neue Frage beantworten, "
                f"nicht die bereits recherchierten Aspekte wiederholen.\n\n"
            )

        q = providers.llm.complete(
            f"Heutiges Datum: {today()}\n\n"
            f"{followup_plan_ctx}"
            f"Erzeuge {'5-6 diverse Suchqueries die verschiedene Aspekte, Hypothesen und Perspektiven der Frage breit abdecken' if s['round'] == 0 else '2-3 praezise, spezifische Suchqueries'} fuer eine Websuche.\n"
            f"Jede Query sollte 5-15 Woerter lang sein und konkreten Kontext enthalten.\n"
            f"SCHLECHT: 'KI Entwicklung' (zu vage)\n"
            f"GUT: 'neueste Durchbrueche kuenstliche Intelligenz 2025 Sprachmodelle' (spezifisch)\n\n"
            f"{reformulation_instruction}"
            f"{falsification_instruction}"
            f"{competing_instruction}"
            f"{alternative_instruction}"
            f"{perspective_instruction}"
            f"{lang_instruction}"
            f"Frage: {s['question']}\n"
            f"{sub_q_info}"
            f"{required_info}"
            f"{uncovered_info}"
            f"{gap_info}"
            f"{related_info}"
            f"Bisherige Queries: {s['queries']}\n"
            f"Bisherige Ergebnisse: {len(s['context'])} Informationsbloecke\n\n"
            f"Generiere Queries die NEUE Informationen liefern, nicht schon Bekanntes wiederholen.\n"
            f"Antworte NUR mit einem JSON Array von Strings. Beispiel: [\"query1\", \"query2\"]",
            deadline=s["deadline"],
            state=s,
        )
    except AgentRateLimited:
        raise
    except (OpenAIError, AgentTimeout, AnthropicAPIError, BedrockAPIError):
        q = ""

    _max_items = settings.first_round_queries if s["round"] == 0 else 3
    new_q = parse_json_string_list(
        q,
        fallback=[s["question"]],
        max_items=_max_items,
    )

    # Quality-chase: force at least one primary and mainstream query for
    # policy / news questions (DE).
    search_lang = s.get("search_language", s.get("language", "de"))
    is_policyish = bool(
        re.search(
            r"\b(privatis\w*|gkv|krankenkass\w*|gesetz\w*|verordnung\w*|beitrag\w*|kosten|haushalt\w*)\b",
            (s.get("question", "") or "").lower(),
        )
    )
    if is_policyish and (search_lang or "").lower() == "de":
        tiers = s.get("source_tier_counts", {}) or {}
        need_primary = (
            s["round"] == 0
            or int(tiers.get("primary", 0)) == 0
            or int(s.get("claim_needs_primary_total", 0)) > int(s.get("claim_needs_primary_verified", 0))
        )
        need_mainstream = (
            s["round"] == 0
            or int(tiers.get("mainstream", 0)) == 0
            or float(s.get("claim_quality_score", 0.0) or 0.0) < 0.35
        )
        new_q = strategies.risk_scoring.inject_quality_site_queries(
            new_q,
            search_lang=search_lang,
            question=s.get("question", ""),
            query_type=s.get("query_type", "general"),
            need_primary=need_primary,
            need_mainstream=need_mainstream,
            max_items=_max_items,
        )

    # Deduplicate, preserve order
    seen = set(s["queries"])
    added = 0
    for query in new_q:
        if query not in seen:
            s["queries"].append(query)
            seen.add(query)
            added += 1

    log.info(
        "TRACE plan: round=%d new_queries=%s total=%d",
        s["round"], json.dumps(new_q, ensure_ascii=False), len(s["queries"]),
    )

    # If no new queries: answer directly (prevents infinite loop)
    if added == 0:
        log.info("Keine neuen Suchqueries generiert, beende Recherche")
        s["done"] = True

    append_iteration_log(s, {
        "node": "plan",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "round": s["round"],
        "new_queries": new_q,
        "new_query_count": len(new_q),
        "added_queries": added,
        "done_no_new_queries": added == 0,
        "quality_site_queries": [q for q in new_q if (q or "").lower().startswith("site:")],
        "total_queries": len(s["queries"]),
        "required_aspects": s.get("required_aspects", []),
        "uncovered_aspects": s.get("uncovered_aspects", []),
    }, testing_mode=settings.testing_mode)
    return s


# ======================================================================= #
# 3. search
# ======================================================================= #


def search(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Execute the current query batch and merge search evidence into state.

    Args:
        s: Mutable AgentState-compatible dict. Reads queued queries,
            offsets, and deadline; writes context blocks, citations,
            token counters, claims, and round progress.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for claim extraction,
            consolidation, and pruning.
        settings: Agent behavior settings controlling batch width,
            timeouts, and logging/test instrumentation.

    Returns:
        The mutated state dict after search, summarization, and claim
        extraction complete or short-circuit.

    Raises:
        AgentRateLimited: Propagated when a provider surfaces a fatal
            rate limit that must abort the run.

    Example:
        >>> search(state, providers=providers, strategies=strategies, settings=settings)
        {'all_citations': ['https://...'], 'context': ['...'], ...}
    """
    _t0 = time.monotonic()
    # Dynamic batch size: broader first round for better coverage
    _batch = settings.first_round_queries if s["round"] == 0 else 3
    offset = s["search_offset"]
    new_q = s["queries"][offset:offset + _batch]
    s["search_offset"] = offset + len(new_q)
    emit_progress(
        s,
        f"Durchsuche {len(new_q)} Quellen (Runde {s['round'] + 1}/{settings.max_rounds})...",
    )

    if not new_q:
        # No queries left -> go straight to answer
        s["done"] = True
        s["round"] += 1
        append_iteration_log(s, {
            "node": "search",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "round": s["round"] - 1,
            "queries_executed": 0,
            "queries": [],
            "sources_found": 0,
            "total_citations": len(s["all_citations"]),
            "context_blocks": len(s["context"]),
            "skipped": "no_queries",
        }, testing_mode=settings.testing_mode)
        return s

    try:
        _check_deadline(s["deadline"])
    except AgentTimeout:
        s["done"] = True
        s["round"] += 1
        append_iteration_log(s, {
            "node": "search",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "round": s["round"] - 1,
            "queries_executed": 0,
            "queries": [],
            "sources_found": 0,
            "total_citations": len(s["all_citations"]),
            "context_blocks": len(s["context"]),
            "skipped": "deadline_exceeded",
        }, testing_mode=settings.testing_mode)
        return s

    # Build search parameters from classify results
    search_kwargs: dict[str, Any] = {
        "search_context_size": "high",  # Always max depth
        "deadline": s["deadline"],
    }

    # Recency filter
    recency = s.get("recency", "")
    if recency:
        search_kwargs["recency_filter"] = recency

    # Language filter via API (more reliable than prompt instruction)
    search_lang = s.get("search_language", "")
    if search_lang:
        search_kwargs["language_filter"] = [search_lang]

    # Query type -> search_mode
    query_type = s.get("query_type", "general")
    if query_type == "academic":
        search_kwargs["search_mode"] = "academic"

    # Default: exclude low-quality domains.
    # When the query contains an explicit `site:...`, we use the
    # Perplexity domain filter as an allow-list for that domain.
    _base_domain_filter = LOW_QUALITY_DOMAINS
    _collect_query_details = settings.testing_mode or log.isEnabledFor(logging.DEBUG)

    def _consume_nonfatal_notice(obj: object) -> str | None:
        consumer = getattr(obj, "consume_nonfatal_notice", None)
        if callable(consumer):
            return consumer()
        return None

    def _domain_filter_for_query(q: str) -> list[str] | None:
        ql = (q or "").lower()
        m = re.search(r"(?:^|\s)site:([^\s]+)", ql)
        if not m:
            return _base_domain_filter
        dom = m.group(1).strip()
        dom = dom.replace("https://", "").replace("http://", "")
        dom = dom.split("/")[0].strip()
        dom = dom.strip(" ,.;:()[]{}<>\\\"'")
        if not dom:
            return _base_domain_filter
        return [dom]

    # Request related questions in first round
    if s["round"] == 0:
        search_kwargs["return_related"] = True

    _query_domain_filters = [_domain_filter_for_query(q) for q in new_q]

    # Parallel search
    def _search_one(item: tuple[str, list[str] | None]) -> dict[str, Any]:
        q, domain_filter = item
        result = providers.search.search(
            q, domain_filter=domain_filter, **search_kwargs)
        warning = _consume_nonfatal_notice(providers.search)
        if warning:
            result = dict(result)
            result["_nonfatal_notice"] = warning
        return result

    _n_workers = min(len(new_q), settings.first_round_queries)

    with ThreadPoolExecutor(max_workers=_n_workers) as ex:
        results = list(ex.map(_search_one, zip(new_q, _query_domain_filters)))

    _query_details: list[dict[str, Any]] = []
    if _collect_query_details:
        for _qi, q in enumerate(new_q):
            r = results[_qi]
            _detail: dict[str, Any] = {
                "query": q,
                "domain_filter": _query_domain_filters[_qi] or [],
                "provider_notice": r.get("_nonfatal_notice") or "",
                "answer_length": len(r.get("answer", "") or ""),
                "citation_count": len(r.get("citations", []) or []),
                "related_question_count": len(r.get("related_questions", []) or []),
                "prompt_tokens": r.get("_prompt_tokens", 0),
                "completion_tokens": r.get("_completion_tokens", 0),
            }
            if r.get("citations"):
                _detail["urls"] = [normalize_url(u) for u in r.get("citations", [])[:5]]
            _query_details.append(_detail)

    _search_fallbacks = sum(1 for r in results if r.get("_nonfatal_notice"))
    if _search_fallbacks:
        emit_progress(
            s,
            f"{_search_fallbacks} Suchanfragen liefen mit Provider-Fallback weiter.",
        )

    # Aggregate token usage from Sonar searches
    _search_prompt_tokens = 0
    _search_completion_tokens = 0
    for r in results:
        _search_prompt_tokens += r.get("_prompt_tokens", 0)
        _search_completion_tokens += r.get("_completion_tokens", 0)

    # Phase 1: Parallel summarize -- all independent Claude calls concurrently
    _summarize_inputs: list[tuple[int, str]] = []
    for _qi, r in enumerate(results):
        if r["answer"]:
            _summarize_inputs.append((_qi, r["answer"]))

    _summarize_results: dict[int, tuple[str, int, int]] = {}
    _summarize_fallbacks = 0
    _summarize_warnings: dict[int, str] = {}
    if _summarize_inputs:
        def _do_summarize(item: tuple[int, str]) -> tuple[int, tuple[str, int, int], str | None]:
            idx, text = item
            result_tuple = providers.llm.summarize_parallel(text, deadline=s["deadline"])
            return (idx, result_tuple, _consume_nonfatal_notice(providers.llm))

        with ThreadPoolExecutor(max_workers=min(len(_summarize_inputs), _n_workers)) as ex:
            for idx, result_tuple, warning in ex.map(_do_summarize, _summarize_inputs):
                _summarize_results[idx] = result_tuple
                if warning:
                    _summarize_fallbacks += 1
                    _summarize_warnings[idx] = warning

    if _summarize_fallbacks:
        emit_progress(
            s,
            f"{_summarize_fallbacks} Quellen-Zusammenfassungen liefen im Fallback-Modus.",
        )

    # Aggregate summarize tokens
    _sum_prompt_tokens = 0
    _sum_completion_tokens = 0
    for facts, pt, ct in _summarize_results.values():
        _sum_prompt_tokens += pt
        _sum_completion_tokens += ct

    # Phase 1b: Parallel claim extraction for later consolidation
    _claim_inputs: list[tuple[int, str, list[str]]] = []
    for _qi, r in enumerate(results):
        if r["answer"]:
            _claim_inputs.append((_qi, r["answer"], r.get("citations", [])))

    _claim_results: dict[int, tuple[list[dict[str, Any]], int, int]] = {}
    _claim_fallbacks = 0
    _claim_warnings: dict[int, str] = {}
    if _claim_inputs:
        def _do_claim_extract(
            item: tuple[int, str, list[str]],
        ) -> tuple[int, tuple[list[dict[str, Any]], int, int], str | None]:
            idx, text, citations = item
            result_tuple = strategies.claim_extraction.extract(
                text,
                citations,
                s.get("question", ""),
                deadline=s["deadline"],
            )
            return (idx, result_tuple, _consume_nonfatal_notice(strategies.claim_extraction))

        with ThreadPoolExecutor(max_workers=min(len(_claim_inputs), _n_workers)) as ex:
            for idx, result_tuple, warning in ex.map(_do_claim_extract, _claim_inputs):
                _claim_results[idx] = result_tuple
                if warning:
                    _claim_fallbacks += 1
                    _claim_warnings[idx] = warning

    if _claim_fallbacks:
        emit_progress(
            s,
            f"Bei {_claim_fallbacks} Quellen konnte keine Claim-Extraktion durchgefuehrt werden.",
        )

    _claim_prompt_tokens = 0
    _claim_completion_tokens = 0
    for _, pt, ct in _claim_results.values():
        _claim_prompt_tokens += pt
        _claim_completion_tokens += ct

    # Phase 2: Sequential context assembly (state access, not parallelisable)
    focus_stems = strategies.claim_consolidation.focus_stems_from_question(
        s.get("question", ""))
    sources_found = 0
    _sources_summary: list[dict[str, Any]] = []
    for _qi, r in enumerate(results):
        if not r["answer"]:
            continue

        # Get summarize result from phase 1
        if _qi in _summarize_results:
            facts = _summarize_results[_qi][0]
        else:
            facts = r["answer"][:800]

        block = facts
        if r["citations"]:
            # Normalise URLs and collect globally (deduplicated)
            for url in r["citations"][:8]:
                normalized = normalize_url(url)
                if normalized not in s["all_citations"]:
                    s["all_citations"].append(normalized)
            sources = "\n".join(f"- {normalize_url(url)}" for url in r["citations"][:8])
            block += f"\n\nQuellen:\n{sources}"
        s["context"].append(block)
        sources_found += 1

        # Fill claim ledger with structured assertions
        extracted_claims = _claim_results.get(_qi, ([], 0, 0))[0]
        kept_claims = 0
        for claim in extracted_claims:
            claim_text = str(claim.get("claim_text", "")).strip()
            if len(claim_text) < 12:
                continue
            if not strategies.claim_consolidation.claim_matches_focus_stems(claim_text, focus_stems):
                continue
            signature = str(claim.get("signature", "")).strip(
            ) or strategies.claim_consolidation.claim_signature(claim_text)
            if not signature:
                continue
            entry = {
                "claim_text": claim_text,
                "claim_type": str(claim.get("claim_type", "fact")),
                "polarity": str(claim.get("polarity", "affirmed")),
                "needs_primary": bool(claim.get("needs_primary", False)),
                "source_urls": [normalize_url(u) for u in claim.get("source_urls", []) if u][:4],
                "published_date": str(claim.get("published_date", "unknown")),
                "signature": signature,
                "round": s["round"],
                "query": new_q[_qi] if _qi < len(new_q) else "",
            }
            s["claim_ledger"].append(entry)
            kept_claims += 1

        # Cap ledger size to keep prompt and RAM stable
        if len(s["claim_ledger"]) > 400:
            s["claim_ledger"] = s["claim_ledger"][-400:]

        if _collect_query_details:
            _entry = dict(_query_details[_qi]) if _qi < len(_query_details) else {
                "query": new_q[_qi] if _qi < len(new_q) else "?",
            }
            _entry["summary"] = facts
            _entry["claims_extracted"] = len(extracted_claims)
            _entry["claims_kept"] = kept_claims
            if _qi in _summarize_warnings:
                _entry["summarize_notice"] = _summarize_warnings[_qi]
            if _qi in _claim_warnings:
                _entry["claim_notice"] = _claim_warnings[_qi]
            if extracted_claims:
                _entry["claims_sample"] = [
                    str(c.get("claim_text", "")).strip()
                    for c in extracted_claims[:3]
                    if str(c.get("claim_text", "")).strip()
                ]
            _sources_summary.append(_entry)

        # Collect related questions
        for rq in r.get("related_questions", []):
            if rq not in s["related_questions"]:
                s["related_questions"].append(rq)

    emit_progress(
        s, f"{sources_found} Quellen verarbeitet, {len(s['all_citations'])} Referenzen gesammelt")

    # Update source quality and aspect coverage
    tier_counts, quality_score = strategies.source_tiering.quality_from_urls(s["all_citations"])
    s["source_tier_counts"] = tier_counts
    s["source_quality_score"] = quality_score
    consolidated_claims_all = strategies.claim_consolidation.consolidate(
        s.get("claim_ledger", []))
    consolidated_claims = strategies.claim_consolidation.materialize(consolidated_claims_all)
    s["consolidated_claims"] = consolidated_claims
    claim_counts, claim_quality, np_total, np_verified = strategies.claim_consolidation.quality_metrics(
        consolidated_claims)
    s["claim_status_counts"] = claim_counts
    s["claim_quality_score"] = claim_quality
    s["claim_needs_primary_total"] = np_total
    s["claim_needs_primary_verified"] = np_verified

    log.info(
        "TRACE search: round=%d queries=%s sources_found=%d total_citations=%d context_blocks=%d "
        "claims=%d claim_quality=%.2f",
        s["round"], json.dumps(new_q, ensure_ascii=False),
        sources_found, len(s["all_citations"]), len(s["context"]),
        len(consolidated_claims), claim_quality,
    )

    # Relevance-based context pruning instead of FIFO
    s["context"] = strategies.context_pruning.prune(
        s["context"],
        question=s["question"],
        sub_questions=s.get("sub_questions", []),
        max_blocks=settings.max_context,
        n_new=sources_found,
    )
    uncovered, coverage = strategies.risk_scoring.estimate_aspect_coverage(
        s.get("required_aspects", []),
        s["context"],
    )
    s["uncovered_aspects"] = uncovered
    s["aspect_coverage"] = coverage
    s["round"] += 1

    emit_progress(
        s,
        f"Quellenqualitaet {quality_score:.2f}, Claim-Qualitaet {claim_quality:.2f}, "
        f"Aspektabdeckung {int(coverage * 100)}%",
    )

    # Aggregate token usage from Sonar + Summarize + Claim extraction
    s["total_prompt_tokens"] += _search_prompt_tokens + \
        _sum_prompt_tokens + _claim_prompt_tokens
    s["total_completion_tokens"] += (
        _search_completion_tokens + _sum_completion_tokens + _claim_completion_tokens
    )

    append_iteration_log(s, {
        "node": "search",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "round": s["round"] - 1,
        "worker_count": _n_workers,
        "queries_executed": len(new_q),
        "queries": new_q,
        "search_parameters": {
            "search_context_size": search_kwargs.get("search_context_size", "high"),
            "recency_filter": search_kwargs.get("recency_filter") or "",
            "language_filter": search_kwargs.get("language_filter", []),
            "search_mode": search_kwargs.get("search_mode") or "",
            "return_related": bool(search_kwargs.get("return_related")),
        },
        "search_fallbacks": _search_fallbacks,
        "summarize_fallbacks": _summarize_fallbacks,
        "claim_fallbacks": _claim_fallbacks,
        "sources_found": sources_found,
        "total_citations": len(s["all_citations"]),
        "context_blocks": len(s["context"]),
        "source_tier_counts": s.get("source_tier_counts", {}),
        "source_quality_score": s.get("source_quality_score", 0.0),
        "claim_ledger_size": len(s.get("claim_ledger", [])),
        "consolidated_claims_count": len(s.get("consolidated_claims", [])),
        "claim_status_counts": s.get("claim_status_counts", {}),
        "claim_quality_score": s.get("claim_quality_score", 0.0),
        "claim_needs_primary_total": s.get("claim_needs_primary_total", 0),
        "claim_needs_primary_verified": s.get("claim_needs_primary_verified", 0),
        "aspect_coverage": s.get("aspect_coverage", 0.0),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        "sources_summary": _sources_summary,
    }, testing_mode=settings.testing_mode)
    return s


# ======================================================================= #
# 4. evaluate
# ======================================================================= #


def evaluate(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Evaluate evidence quality, stop criteria, and remaining gaps.

    Args:
        s: Mutable AgentState-compatible dict. Reads accumulated
            evidence, claims, and prior confidence; writes quality
            metrics, stop decisions, and follow-up gap information.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for source quality, claim
            consolidation, risk coverage, and stop criteria.
        settings: Agent behavior settings controlling escalation and
            stopping thresholds.

    Returns:
        The mutated state dict with refreshed quality metrics and stop
        status.

    Raises:
        AgentRateLimited: Propagated when the evaluation model hard-fails
            on upstream rate limiting.

    Example:
        >>> evaluate(state, providers=providers, strategies=strategies, settings=settings)
        {'final_confidence': 8, 'done': True, ...}
    """
    emit_progress(
        s,
        f"Bewerte Informationsqualitaet (nach Runde {s['round']}/{settings.max_rounds})...",
    )
    _t0 = time.monotonic()
    _stagnation_detected = False
    if s.get("done"):
        append_iteration_log(s, {
            "node": "evaluate",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "confidence": s.get("final_confidence", 0),
            "skipped": "already_done",
        }, testing_mode=settings.testing_mode)
        return s
    try:
        _check_deadline(s["deadline"])
    except AgentTimeout:
        s["done"] = True
        append_iteration_log(s, {
            "node": "evaluate",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _t0, 3),
            "confidence": s.get("final_confidence", 0),
            "skipped": "deadline_exceeded",
        }, testing_mode=settings.testing_mode)
        return s

    # Recompute quality / coverage metrics (robust against drift)
    tier_counts, quality_score = strategies.source_tiering.quality_from_urls(
        s.get("all_citations", []))
    s["source_tier_counts"] = tier_counts
    s["source_quality_score"] = quality_score
    uncovered, coverage = strategies.risk_scoring.estimate_aspect_coverage(
        s.get("required_aspects", []),
        s.get("context", []),
    )
    s["uncovered_aspects"] = uncovered
    s["aspect_coverage"] = coverage
    consolidated_claims_all = strategies.claim_consolidation.consolidate(
        s.get("claim_ledger", []))
    consolidated_claims = strategies.claim_consolidation.materialize(consolidated_claims_all)
    s["consolidated_claims"] = consolidated_claims
    claim_counts, claim_quality, claim_np_total, claim_np_verified = strategies.claim_consolidation.quality_metrics(
        consolidated_claims
    )
    s["claim_status_counts"] = claim_counts
    s["claim_quality_score"] = claim_quality
    s["claim_needs_primary_total"] = claim_np_total
    s["claim_needs_primary_verified"] = claim_np_verified

    evaluate_model = (
        providers.llm.models.reasoning_model
        if (s.get("high_risk", False) and settings.high_risk_evaluate_escalate)
        else providers.llm.models.effective_evaluate_model
    )

    # Hint for negative evidence
    negative_evidence_hint = ""
    if s["round"] >= 2:
        _prev_conf = s.get("final_confidence", 0)
        _n_citations = len(s.get("all_citations", []))
        negative_evidence_hint = (
            "\n\nWICHTIG — NEGATIVE EVIDENZ:\n"
            f"Es wurden bereits {s['round']} Suchrunden mit {len(s['queries'])} Queries durchgefuehrt "
            f"und {_n_citations} Quellen gesammelt.\n"
        )
        if _prev_conf > 0 and _prev_conf <= 4:
            negative_evidence_hint += (
                f"Die Confidence war in der vorherigen Runde ebenfalls nur {_prev_conf}/10.\n"
                f"Wenn sich trotz {_n_citations} durchsuchter Quellen nichts Substantielles "
                f"verbessert hat, ist das ein STARKES Signal: "
                f"Die Praemisse der Frage ist wahrscheinlich FALSCH.\n"
                f"Setze in diesem Fall CONFIDENCE auf 7-9 — 'Es existiert nicht' ist eine "
                f"hochkonfidente Erkenntnis nach umfangreicher Recherche.\n"
            )
        negative_evidence_hint += (
            "Wenn die Recherche KONSISTENT keine Belege fuer die Behauptung/Annahme in der Frage findet, "
            "dann IST das ein Ergebnis. 'Es existiert nicht' oder 'Die Annahme ist falsch' "
            "sind valide Antworten mit HOHER Confidence (7-9).\n"
            "Bewerte also nicht nur ob du gefunden hast WAS gefragt wurde, "
            "sondern auch ob du genug gesucht hast um sicher zu sagen dass es NICHT existiert.\n"
        )

    # --- LLM call: evaluate information quality ---
    _eval_raw = ""
    quality_hint = (
        "\nQUELLENQUALITAET:\n"
        f"- primary={tier_counts.get('primary', 0)}, "
        f"mainstream={tier_counts.get('mainstream', 0)}, "
        f"stakeholder={tier_counts.get('stakeholder', 0)}, "
        f"unknown={tier_counts.get('unknown', 0)}, "
        f"low={tier_counts.get('low', 0)}.\n"
        f"- Gesamt-Qualitaetsscore: {quality_score:.2f} (0-1).\n"
        "- Wenn zentrale Aussagen nur durch Stakeholder- oder Low-Quality-Quellen belegt sind, "
        "reduziere CONFIDENCE und setze GAPS entsprechend.\n"
        "- Trenne strikt zwischen neutralem Fakt und Akteursbehauptung "
        "(z.B. Parteiverband, Branchenverband, Lobbyorganisation).\n\n"
    )
    aspect_hint = ""
    if s.get("required_aspects"):
        aspect_hint = (
            "ASPEKTABDECKUNG:\n"
            f"- Pflichtaspekte: {json.dumps(s['required_aspects'], ensure_ascii=False)}\n"
            f"- Noch offen: {json.dumps(s['uncovered_aspects'], ensure_ascii=False)}\n"
            f"- Coverage: {int(s['aspect_coverage'] * 100)}%.\n"
            "- Wenn Pflichtaspekte offen sind, kann STATUS nicht SUFFICIENT sein.\n\n"
        )
    claim_hint = (
        "CLAIM-LEDGER:\n"
        f"- verified={claim_counts.get('verified', 0)}, "
        f"contested={claim_counts.get('contested', 0)}, "
        f"unverified={claim_counts.get('unverified', 0)}.\n"
        f"- Claim-Qualitaetsscore: {claim_quality:.2f} (0-1).\n"
        f"- Primaerpflichtige Claims verifiziert: {claim_np_verified}/{claim_np_total}.\n"
        "- Falls zentrale Claims contested/unverified sind, reduziere CONFIDENCE und "
        "setze STATUS auf INSUFFICIENT.\n"
        "- Nutze die folgende konsolidierte Claim-Liste fuer die Bewertung:\n"
        + strategies.claim_consolidation.claims_prompt_view(consolidated_claims, max_items=14)
        + "\n\n"
    )
    try:
        eval_prompt = (
            f"Heutiges Datum: {today()}\n\n"
            f"Bewerte ob die recherchierten Informationen ausreichen, "
            f"um die Frage vollstaendig und korrekt zu beantworten.\n\n"
            f"ZEITLICHE KONSISTENZ:\n"
            f"- Wenn die Frage relative Zeitangaben enthaelt ('vor kurzem', 'neulich', 'letztens'), "
            f"pruefe ob die gefundenen Ereignisse zeitlich zum heutigen Datum ({today()}) passen.\n"
            f"- Ein Ereignis von vor 12+ Monaten ist NICHT 'vor kurzem'.\n"
            f"- Wenn die gefundenen Ereignisse zeitlich nicht passen, setze GAPS "
            f"auf 'Zeitlich aktuelleres Ereignis nicht gefunden' und CONFIDENCE maximal 5.\n\n"
            f"MEHRERE PASSENDE EREIGNISSE:\n"
            f"- Wenn mehrere Ereignisse auf die Beschreibung passen koennten, "
            f"setze GAPS auf 'Moeglicherweise ein anderes/aktuelleres Ereignis gemeint' "
            f"und reduziere CONFIDENCE.\n\n"
            f"KONKURRIERENDE ERKLAERUNGEN:\n"
            f"- Wenn du in den Recherche-Ergebnissen VERSCHIEDENE moegliche Ereignisse/Antworten findest, "
            f"die auf die Frage passen koennten, liste sie auf.\n"
            f"- Antworte in der Zeile COMPETING_EVENTS mit einer kurzen Auflistung: "
            f"'Event A (Datum) vs Event B (Datum)' oder 'Keine'.\n\n"
            + quality_hint
            + aspect_hint
            + claim_hint
            + f"Frage: {s['question']}\n\n"
            + f"Recherche-Ergebnisse (nummeriert):\n"
            + "\n".join(f"[Block {i + 1}]:\n{block}" for i, block in enumerate(s["context"]))
            + negative_evidence_hint
            + EVALUATE_FORMAT_SUFFIX
        )
        a = providers.llm.complete(
            eval_prompt,
            deadline=s["deadline"],
            model=evaluate_model,
            state=s,
        )
        _eval_raw = a

        # --- Parse base values from LLM response ---
        m_conf = re.search(r"CONFIDENCE:\s*(\d+)", a)
        conf = int(m_conf.group(1)) if m_conf else 5

        m_gaps = re.search(r"GAPS:\s*(.+?)(?:\n|$)", a)
        gaps = m_gaps.group(1).strip() if m_gaps else ""
        s["gaps"] = "" if is_none_value(gaps) else gaps

        # --- Apply heuristics ---
        conf = strategies.stop_criteria.check_contradictions(s, a, conf)
        strategies.stop_criteria.filter_irrelevant_blocks(s, a)
        conf = strategies.stop_criteria.extract_competing_events(s, a, conf)
        conf = strategies.stop_criteria.extract_evidence_scores(s, a, conf)

    except AgentRateLimited:
        raise
    except (OpenAIError, AgentTimeout, AnthropicAPIError, BedrockAPIError):
        # No fail-open: on evaluate error stay conservative.
        conf = min(max(s.get("final_confidence", 0), 5), settings.confidence_stop - 2)
        if not s.get("gaps"):
            s["gaps"] = "Automatische Qualitaetsbewertung unvollstaendig; Antwort vorsichtig formulieren."

    # Guardrails: couple confidence to source quality and aspect coverage.
    q_lower = s["question"].lower()
    needs_primary = bool(
        re.search(r"\b(prozent|mrd|mio|euro|gesetz|regel|politik|beitrag|kosten)\b", q_lower))
    primary_n = int(tier_counts.get("primary", 0))
    mainstream_n = int(tier_counts.get("mainstream", 0))
    low_n = int(tier_counts.get("low", 0))
    uncovered_n = len(s.get("uncovered_aspects", []))
    verified_claims = int(claim_counts.get("verified", 0))
    contested_claims = int(claim_counts.get("contested", 0))
    unverified_claims = int(claim_counts.get("unverified", 0))

    if not s.get("all_citations"):
        conf = min(conf, 6)
        if not s.get("gaps"):
            s["gaps"] = "Keine belastbaren Quellen gefunden."
    if low_n > (primary_n + mainstream_n) and conf > 7:
        conf = 7
    if needs_primary and primary_n == 0 and conf > 8:
        conf = 8
        if not s.get("gaps"):
            s["gaps"] = "Zentrale Zahlen/Regelungen nicht mit Primaerquelle belegt."
    if uncovered_n > 0 and conf > 8:
        conf = 8
        if not s.get("gaps"):
            s["gaps"] = f"Pflichtaspekte offen: {', '.join(s['uncovered_aspects'][:2])}"
    if contested_claims >= 2 and conf > 7:
        conf = 7
        if not s.get("gaps"):
            s["gaps"] = "Mehrere zentrale Aussagen sind zwischen Quellen umstritten."

    # --- Post-LLM stop heuristics ---
    _prev_conf = s.get("final_confidence", 0)
    _n_citations = len(s.get("all_citations", []))

    _falsification_just_triggered = strategies.stop_criteria.check_falsification(
        s, conf, _prev_conf)
    conf, _stagnation_detected = strategies.stop_criteria.check_stagnation(
        s, conf, _prev_conf, _n_citations, _falsification_just_triggered)
    _utility, _utility_stop = strategies.stop_criteria.compute_utility(
        s, conf, _prev_conf, _n_citations)

    s["final_confidence"] = conf

    _plateau_stop = strategies.stop_criteria.check_plateau(
        s, conf, _prev_conf, _stagnation_detected)

    # --- Final stop logic ---
    log.info(
        "TRACE evaluate: round=%d confidence=%d/%d gaps='%s' context_blocks=%d "
        "quality=%.2f claim_quality=%.2f claims(v/c/u)=%d/%d/%d model=%s done=%s",
        s["round"], conf, settings.confidence_stop,
        s.get("gaps", "")[:100], len(s["context"]),
        quality_score, claim_quality,
        verified_claims, contested_claims, unverified_claims,
        evaluate_model,
        conf >= settings.confidence_stop or s["round"] >= settings.max_rounds or s["done"],
    )

    if s["done"] or conf >= settings.confidence_stop or s["round"] >= settings.max_rounds:
        s["done"] = True
        emit_progress(
            s, f"Recherche abgeschlossen (Confidence: {conf}/10, Runden: {s['round']})")
    else:
        emit_progress(s, f"Confidence {conf}/10 — weitere Recherche noetig")

    _eval_log_entry: dict[str, Any] = {
        "node": "evaluate",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "round": s["round"],
        "confidence": conf,
        "confidence_stop_target": settings.confidence_stop,
        "gaps": s.get("gaps", ""),
        "competing_events": s.get("competing_events", ""),
        "stagnation_detected": _stagnation_detected,
        "falsification_triggered": s.get("falsification_triggered", False),
        "evidence_consistency": s.get("evidence_consistency", 0),
        "evidence_sufficiency": s.get("evidence_sufficiency", 0),
        "verified_claims": verified_claims,
        "contested_claims": contested_claims,
        "unverified_claims": unverified_claims,
        "source_tier_counts": s.get("source_tier_counts", {}),
        "source_quality_score": s.get("source_quality_score", 0.0),
        "claim_status_counts": s.get("claim_status_counts", {}),
        "claim_quality_score": s.get("claim_quality_score", 0.0),
        "consolidated_claims_count": len(s.get("consolidated_claims", [])),
        "claim_needs_primary_total": s.get("claim_needs_primary_total", 0),
        "claim_needs_primary_verified": s.get("claim_needs_primary_verified", 0),
        "aspect_coverage": s.get("aspect_coverage", 0.0),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        "model": evaluate_model,
        "utility_score": _utility,
        "utility_stop": _utility_stop,
        "plateau_stop": _plateau_stop,
        "context_blocks": len(s["context"]),
        "stop_by_confidence": conf >= settings.confidence_stop,
        "stop_by_round_limit": s["round"] >= settings.max_rounds,
        "stop_by_existing_done": bool(s.get("done")),
        "done": s["done"],
    }
    if settings.testing_mode and _eval_raw:
        _eval_log_entry["reasoning"] = _eval_raw
    append_iteration_log(s, _eval_log_entry, testing_mode=settings.testing_mode)
    return s


# ======================================================================= #
# 5. answer
# ======================================================================= #


def answer(
    s: dict,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict:
    """Formulate the final user-facing answer from the collected evidence.

    Args:
        s: Mutable AgentState-compatible dict. Reads context, claim
            metrics, language, citations, and deadline; writes the final
            answer text and answer-node runtime metadata.
        providers: Active LLM and search providers.
        strategies: Runtime strategies for answer citation selection and
            claim formatting.
        settings: Agent behavior settings controlling citation limits and
            fallback behavior.

    Returns:
        The mutated state dict with the final answer text populated.

    Raises:
        AgentRateLimited: Propagated when final answer generation hits a
            fatal upstream rate limit.

    Example:
        >>> answer(state, providers=providers, strategies=strategies, settings=settings)
        {'answer': 'Aktueller Stand ...', ...}
    """
    n_rounds = s.get("round", 0)
    round_label = "Runde" if n_rounds == 1 else "Runden"
    emit_progress(s, f"Formuliere Antwort (nach {n_rounds} {round_label})...")
    _t0 = time.monotonic()
    ctx = "\n\n---\n\n".join(s["context"]) if s["context"] else ""

    # Determine answer language
    lang = s.get("language", "de")
    answer_lang = LANG_NAMES.get(lang, lang)

    # Build numbered source list (fast-path: claim-bound selection)
    consolidated_claims = s.get("consolidated_claims", [])
    citations = s.get("all_citations", [])
    prompt_citations = strategies.claim_consolidation.select_answer_citations(
        consolidated_claims,
        citations,
        max_items=settings.answer_prompt_citations_max,
    )
    prompt_citations_used_fallback = False
    if not prompt_citations and citations:
        fallback_n = min(25, settings.answer_prompt_citations_max)
        prompt_citations = citations[:fallback_n]
        prompt_citations_used_fallback = True

    # Assemble state_data dict for build_answer_system_prompt
    state_data: dict[str, Any] = {
        "today_str": today(),
        "answer_lang": answer_lang,
        "context": ctx,
        "prompt_citations": prompt_citations,
        "all_citations": citations,
        "source_tier_counts": s.get("source_tier_counts", {}),
        "source_quality_score": s.get("source_quality_score", 0.0),
        "claim_status_counts": s.get("claim_status_counts", {}),
        "claim_quality_score": s.get("claim_quality_score", 0.0),
        "claim_needs_primary_total": s.get("claim_needs_primary_total", 0),
        "claim_needs_primary_verified": s.get("claim_needs_primary_verified", 0),
        "consolidated_claims": consolidated_claims,
        "claims_prompt_view_fn": strategies.claim_consolidation.claims_prompt_view,
        "required_aspects": s.get("required_aspects"),
        "uncovered_aspects": s.get("uncovered_aspects", []),
        "competing_events": s.get("competing_events", ""),
        "history": s.get("history", ""),
        "_prev_question": s.get("_prev_question", ""),
        "_prev_answer": s.get("_prev_answer", ""),
    }
    system = build_answer_system_prompt(state_data)
    fallback_model = (
        providers.llm.models.effective_evaluate_model
        if (
            providers.llm.models.effective_evaluate_model
            and providers.llm.models.effective_evaluate_model != providers.llm.models.reasoning_model
        )
        else None
    )
    fallback_attempted = False
    fallback_succeeded = False

    try:
        s["answer"] = providers.llm.complete(
            s["question"], system=system, deadline=s["deadline"], state=s)
    except AgentTimeout:
        # On timeout: use context directly as answer
        if ctx:
            s["answer"] = (
                "Die Recherche konnte aus Zeitgruenden nicht vollstaendig abgeschlossen werden. "
                f"Hier die bisherigen Ergebnisse:\n\n{ctx}"
            )
        else:
            s["answer"] = "Die Anfrage konnte aufgrund eines Zeitlimits nicht bearbeitet werden. Bitte erneut versuchen."
    except (OpenAIError, AnthropicAPIError, BedrockAPIError) as e:
        log.error("Finale Antwort fehlgeschlagen: %s", e)
        if fallback_model:
            try:
                fallback_attempted = True
                emit_progress(
                    s, f"Finale Antwort fehlgeschlagen — Fallback-Modell {fallback_model}")
                s["answer"] = providers.llm.complete(
                    s["question"],
                    system=system,
                    deadline=s["deadline"],
                    model=fallback_model,
                    state=s,
                )
                fallback_succeeded = True
            except (OpenAIError, AgentTimeout, AnthropicAPIError, BedrockAPIError) as e2:
                log.error("Finale Antwort-Fallback fehlgeschlagen (%s): %s", fallback_model, e2)
                if ctx:
                    s["answer"] = (
                        "Bei der Formulierung der Antwort ist ein Fehler aufgetreten. "
                        f"Hier die Recherche-Ergebnisse:\n\n{ctx}"
                    )
                else:
                    s["answer"] = "Bei der Verarbeitung ist ein Fehler aufgetreten. Bitte erneut versuchen."
        elif ctx:
            s["answer"] = (
                "Bei der Formulierung der Antwort ist ein Fehler aufgetreten. "
                f"Hier die Recherche-Ergebnisse:\n\n{ctx}"
            )
        else:
            s["answer"] = "Bei der Verarbeitung ist ein Fehler aufgetreten. Bitte erneut versuchen."

    # Quick citation guardrail: remove non-allowed links.
    removed_link_count = 0
    allowed_citation_urls = set(prompt_citations)
    appended_sources_footer = False
    allowed_link_count = 0
    if allowed_citation_urls and s.get("answer"):
        s["answer"], removed_link_count = sanitize_answer_links(
            s["answer"], allowed_citation_urls)
        if removed_link_count:
            log.info("TRACE answer: removed %d non-allowed links", removed_link_count)
            emit_progress(s, f"{removed_link_count} nicht-zugelassene Links entfernt")
        allowed_link_count = count_allowed_links(s["answer"], allowed_citation_urls)
        if allowed_link_count == 0:
            fallback_bar = " | ".join(
                f"[{i}]({url})" for i, url in enumerate(prompt_citations[:5], 1)
            )
            if fallback_bar:
                s["answer"] += f"\n\n**Quellen:** {fallback_bar}"
                appended_sources_footer = True

    # Append stats footer
    elapsed = time.monotonic() - s.get("start_time", time.monotonic())
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    time_str = f"{minutes}:{seconds:02d} min" if minutes > 0 else f"{seconds}s"
    conf = s.get("final_confidence", 0)
    n_sources = len(s.get("all_citations", []))
    n_rounds = s.get("round", 0)
    n_queries = len(s.get("queries", []))

    stats_parts = []
    if n_sources:
        if 0 < allowed_link_count < n_sources:
            stats_parts.append(f"{n_sources} Quellen ({allowed_link_count} verlinkt)")
        else:
            stats_parts.append(f"{n_sources} Quellen")
    if n_queries:
        stats_parts.append(f"{n_queries} Suchen")
    if n_rounds:
        stats_parts.append(f"{n_rounds} {'Runde' if n_rounds == 1 else 'Runden'}")
    stats_parts.append(time_str)
    if conf:
        stats_parts.append(f"Confidence {conf}/10")

    stats_line = " · ".join(stats_parts)
    s["answer"] += f"\n\n---\n*{stats_line}*"

    log.info(
        "TRACE answer: length=%d citations=%d prompt_citations=%d linked=%d rounds=%d elapsed=%.1fs confidence=%d",
        len(s["answer"]), n_sources, len(
            prompt_citations), allowed_link_count, n_rounds, elapsed, conf,
    )
    log.debug("ANSWER text:\n%s", s["answer"])

    append_iteration_log(s, {
        "node": "answer",
        "timestamp": time.time(),
        "duration_s": round(time.monotonic() - _t0, 3),
        "answer_length": len(s["answer"]),
        "citation_count": n_sources,
        "prompt_citation_count": len(prompt_citations),
        "prompt_citations": prompt_citations[:10],
        "prompt_citations_used_fallback": prompt_citations_used_fallback,
        "removed_non_allowed_links": removed_link_count,
        "allowed_link_count": allowed_link_count,
        "appended_sources_footer": appended_sources_footer,
        "fallback_model": fallback_model or "",
        "fallback_attempted": fallback_attempted,
        "fallback_succeeded": fallback_succeeded,
        "stats_line": stats_line,
        "rounds": n_rounds,
        "elapsed_total_s": round(elapsed, 1),
        "confidence": conf,
    }, testing_mode=settings.testing_mode)

    emit_progress(s, "done")
    return s
