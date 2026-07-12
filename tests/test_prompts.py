"""Tests for prompt variants used by the final answer node."""

from __future__ import annotations

from types import SimpleNamespace

from inqtrix.nodes import (
    _domain_filter_for_query_text,
    _select_additional_links,
)
from inqtrix.prompts import (
    build_answer_section_system_prompt,
    build_answer_section_user_prompt,
    build_answer_system_prompt,
)
from inqtrix.report_profiles import ReportProfile, tuning_for_report_profile
from inqtrix.strategies._risk_scoring import KeywordRiskScorer
from inqtrix.strategies._source_tiering import DefaultSourceTiering


def _base_state(report_profile: ReportProfile) -> dict[str, object]:
    return {
        "today_str": "2026-04-13",
        "answer_lang": "Deutsch",
        "evidence_overview": "",
        "allowed_citations": [],
        "source_tier_counts": {},
        "claim_status_counts": {},
        "report_profile": report_profile,
    }


def test_build_answer_system_prompt_compact_profile():
    prompt = build_answer_system_prompt(_base_state(ReportProfile.COMPACT))

    assert "**Kurzfazit**" in prompt
    assert "So ausfuehrlich wie Frage und Evidenzlage es erfordern" in prompt
    assert "600-1200 Woerter" not in prompt
    assert "Priorisiere vollstaendige, sauber abgeschlossene Abschnitte" in prompt
    assert "**Executive Summary**" not in prompt


def test_domain_filter_for_query_text_extracts_multiple_site_domains():
    domain_filter = _domain_filter_for_query_text(
        "site:reuters.com OR site:bloomberg.com Tesla CapEx",
        base_domain_filter=["-reddit.com"],
    )

    assert domain_filter == ["reuters.com", "bloomberg.com"]


def test_domain_filter_for_query_text_uses_base_filter_without_site_operator():
    domain_filter = _domain_filter_for_query_text(
        "Tesla CapEx news",
        base_domain_filter=["-reddit.com"],
    )

    assert domain_filter == ["-reddit.com"]


def test_domain_filter_for_query_text_has_no_default_blocklist():
    domain_filter = _domain_filter_for_query_text("Tesla CapEx news")

    assert domain_filter == []


def test_build_answer_system_prompt_deep_profile():
    prompt = build_answer_system_prompt(_base_state(ReportProfile.DEEP))

    assert "**Executive Summary**" in prompt
    assert "So ausfuehrlich wie Frage und Evidenzlage es erfordern" in prompt
    assert "1800-2400 Woerter" not in prompt
    assert "Priorisiere vollstaendige, sauber abgeschlossene Abschnitte" in prompt
    assert "**Risiken / Unsicherheiten**" in prompt
    assert "keine belastbare Evidenz vorliegt" in prompt


def test_build_answer_system_prompt_uses_runtime_managed_reference_sections():
    state = _base_state(ReportProfile.DEEP)
    state["evidence_overview"] = "RECHERCHE-ERGEBNIS R1\n[E1] Quelle"
    state["allowed_citations"] = ["https://example.com/report"]

    prompt = build_answer_system_prompt(state)

    assert "Erzeuge KEINEN eigenen Referenz-, Quellen- oder Linkabschnitt" in prompt
    assert "[E1] [E2]" in prompt
    assert "Am Ende der Antwort fuege eine Quellenleiste ein" not in prompt


def test_build_answer_system_prompt_includes_evidence_overview_contract():
    state = _base_state(ReportProfile.COMPACT)
    state["allowed_citations"] = ["https://example.com/report", "https://example.com/check"]
    state["evidence_overview"] = (
        "RECHERCHE-ERGEBNIS R1\n"
        "[E1] Meta Investor Relations\n"
        "  URL: https://example.com/report\n"
        "  Datum: 2026-05-01 | Einstufung: primary | Beleglage: cross-checked\n"
        "  Aussagen dieser Quelle:\n"
        "  - Meta expects 115-135 USD billion capex."
    )

    prompt = build_answer_system_prompt(state)

    assert "EVIDENZ-UEBERSICHT" in prompt
    assert "cross-checked" in prompt and "primary-source" in prompt
    assert "single-source verified" in prompt
    assert "source-context" in prompt
    # Citation rule: bare [E#] label, no URL (URLs are attached post-synthesis).
    assert "[E12]" in prompt
    assert "[E12](URL)" not in prompt
    assert "RECHERCHE-ERGEBNIS R1" in prompt


def test_build_answer_system_prompt_hint_when_no_evidence_overview():
    state = _base_state(ReportProfile.COMPACT)
    state["evidence_overview"] = ""
    state["allowed_citations"] = ["https://example.com/report"]
    state["claim_status_counts"] = {"verified": 0, "contested": 0, "unverified": 0}
    state["source_tier_counts"] = {"primary": 1, "mainstream": 0, "stakeholder": 0, "unknown": 0, "low": 0}

    prompt = build_answer_system_prompt(state)

    assert "HINWEIS ZUR EVIDENZLAGE" in prompt
    assert "quellen-attribuiert" in prompt


def test_news_contract_detects_relative_day_window():
    scorer = KeywordRiskScorer()

    assert scorer.infer_answer_contract(
        "Was waren die wichtigsten KI-Entwicklungen der letzten 7 Tage?"
    ) == "news_briefing"


def test_additional_links_skip_low_quality_domains():
    strategies = SimpleNamespace(source_tiering=DefaultSourceTiering())

    links = _select_additional_links(
        [
            "https://aitoolsrecap.com/Blog/ai-news-may-2026",
            "https://openai.com/index/introducing-gpt-5-5",
            "https://toolscompare.ai/news/may-2026",
            "https://techcrunch.com/2026/05/05/openai-releases-gpt-5-5-instant-a-new-default-model-for-chatgpt",
        ],
        excluded_urls=set(),
        prompt_citation_urls=set(),
        strategies=strategies,
    )

    assert "https://openai.com/index/introducing-gpt-5-5" in links
    assert "https://techcrunch.com/2026/05/05/openai-releases-gpt-5-5-instant-a-new-default-model-for-chatgpt" in links
    assert all("aitoolsrecap.com" not in url for url in links)
    assert all("toolscompare.ai" not in url for url in links)


def test_answer_prompt_explains_evidence_overview_beleglage_levels():
    state = _base_state(ReportProfile.COMPACT)
    state["allowed_citations"] = ["https://example.com/report"]
    state["evidence_overview"] = (
        "RECHERCHE-ERGEBNIS R1\n"
        "[E1] Quelle\n"
        "  URL: https://example.com/report\n"
        "  Datum: 2026-05-01 | Einstufung: primary | Beleglage: source-context"
    )

    prompt = build_answer_system_prompt(state)

    assert "`cross-checked` und `primary-source`" in prompt
    assert "`single-source verified`" in prompt
    assert "`contested`" in prompt
    assert "`source-context`" in prompt
    assert "inline attribuieren" in prompt


def test_answer_prompt_tells_model_to_use_per_source_substance():
    state = _base_state(ReportProfile.COMPACT)
    state["allowed_citations"] = ["https://example.com/report"]
    state["evidence_overview"] = (
        "RECHERCHE-ERGEBNIS R1\n"
        "[E1] Quelle\n"
        "  Aussagen dieser Quelle:\n"
        "  - 115-135 USD billion capex erwartet."
    )

    prompt = build_answer_system_prompt(state)

    assert "konkreten Aussagen und Belegausschnitte" in prompt
    assert "nicht nur Titel" in prompt
    assert "Provider-Synthese ist Orientierung, keine eigenstaendige Quelle" in prompt


def test_section_prompt_contains_report_so_far_and_used_evidence_labels():
    prompt = build_answer_section_user_prompt(
        "Was ist passiert?",
        heading="Analyse",
        instruction="Vertiefe die Fakten.",
        completed_headings=["Executive Summary"],
        report_so_far_summary="Executive Summary: Kurzbefund.",
        used_evidence_labels=["E1", "E2"],
    )

    assert "Bisherige Report-Zusammenfassung:" in prompt
    assert "Executive Summary: Kurzbefund." in prompt
    assert "Bereits verwendete Evidence-Labels:" in prompt
    assert "E1, E2" in prompt


def test_build_answer_section_system_prompt_reuses_common_rules():
    state = _base_state(ReportProfile.DEEP)
    state["evidence_overview"] = "RECHERCHE-ERGEBNIS R1\n[E1] Quelle"
    state["allowed_citations"] = ["https://example.com/report"]

    prompt = build_answer_section_system_prompt(
        state,
        heading="Analyse",
        instruction="Schreibe sinnvolle Unterabschnitte mit Zahlen und Zusammenhaengen.",
        length_guidance="Tiefe nach Evidenzlage mit `###`-Unterabschnitten",
        section_position=3,
        section_total=6,
    )

    assert "Du schreibst NUR EINEN Abschnitt" in prompt
    assert "Fuege die Hauptueberschrift `## Analyse` NICHT selbst hinzu" in prompt
    assert "Erzeuge KEINEN eigenen Referenz-, Quellen- oder Linkabschnitt" in prompt
    assert "CLAIM-KALIBRIERUNG" not in prompt


def _weak_evidence_state(report_profile: ReportProfile) -> dict[str, object]:
    """State where TRANSPARENZPFLICHT and citation rules should fire."""
    state = _base_state(report_profile)
    state["claim_status_counts"] = {"verified": 1, "contested": 0, "unverified": 5}
    state["claim_needs_primary_total"] = 3
    state["claim_needs_primary_verified"] = 1
    state["source_tier_counts"] = {"primary": 1, "mainstream": 1, "stakeholder": 0, "unknown": 2, "low": 0}
    state["evidence_overview"] = "RECHERCHE-ERGEBNIS R1\n[E1] Quelle"
    state["allowed_citations"] = ["https://example.com/a", "https://example.com/b"]
    state["required_aspects"] = ["Status quo", "Position der Akteure"]
    state["uncovered_aspects"] = ["Position der Akteure"]
    return state


def test_section_executive_summary_omits_unsicherheiten_subsection_directive():
    """Executive Summary must NOT be told to add a '## Unsicherheiten' block.

    Regression: the global TRANSPARENZPFLICHT block previously injected this
    directive into every section, which made the Executive Summary section
    add an unsolicited uncertainty bullet list where it does not belong.
    """
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_section_system_prompt(
        state,
        heading="Executive Summary",
        instruction="Beantworte die Frage direkt.",
        length_guidance="knapp und dicht, keine Bulletpoints",
        section_position=1,
        section_total=6,
    )

    # The transparency block itself still appears (status info is useful)
    assert "TRANSPARENZPFLICHT BEI UNSICHERER EVIDENZ" in prompt
    # but the structural directive is suppressed for this section
    assert "Fuege einen Abschnitt '## Unsicherheiten / Offene Punkte'" not in prompt
    # and the suppression note is present
    assert "Fuege in DIESEM Abschnitt KEINEN eigenen 'Unsicherheiten'-Block" in prompt


def test_section_risiken_keeps_unsicherheiten_subsection_directive():
    """The Risiken / Unsicherheiten section is the legitimate host."""
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_section_system_prompt(
        state,
        heading="Risiken / Unsicherheiten",
        instruction="Liste praezise Bulletpoints zu Evidenzgrenzen.",
        length_guidance="nur echte Risiken und Unsicherheiten",
        section_position=5,
        section_total=6,
    )

    assert "TRANSPARENZPFLICHT BEI UNSICHERER EVIDENZ" in prompt
    assert "Fuege einen Abschnitt '## Unsicherheiten / Offene Punkte'" in prompt
    assert "Fuege in DIESEM Abschnitt KEINEN eigenen 'Unsicherheiten'-Block" not in prompt


def test_section_fazit_does_not_host_unsicherheiten_block():
    """The Fazit section in DEEP must NOT host the transparency sub-block.

    Regression: the section after 'Risiken / Unsicherheiten' would otherwise
    duplicate the block, producing two '## Unsicherheiten / Offene Punkte'
    sub-headings in the rendered answer.
    """
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_section_system_prompt(
        state,
        heading="Fazit / Ausblick",
        instruction="Ziehe eine belastbare Gesamteinordnung.",
        length_guidance="kurz, klar und abgeschlossen",
        section_position=6,
        section_total=6,
    )

    assert "TRANSPARENZPFLICHT BEI UNSICHERER EVIDENZ" in prompt
    assert "Fuege einen Abschnitt '## Unsicherheiten / Offene Punkte'" not in prompt
    assert "Fuege in DIESEM Abschnitt KEINEN eigenen 'Unsicherheiten'-Block" in prompt


def test_section_einordnung_compact_hosts_unsicherheiten_block():
    """In COMPACT (no Risiken section) the closing 'Einordnung / Ausblick' hosts it."""
    state = _weak_evidence_state(ReportProfile.COMPACT)

    prompt = build_answer_section_system_prompt(
        state,
        heading="Einordnung / Ausblick",
        instruction="Ordne die Befunde knapp ein.",
        length_guidance="kurz und abgeschlossen",
        section_position=4,
        section_total=4,
    )

    assert "Fuege einen Abschnitt '## Unsicherheiten / Offene Punkte'" in prompt


def test_monolithic_mode_keeps_legacy_unsicherheiten_directive():
    """Backwards compatibility: the monolithic prompt path is unchanged."""
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_system_prompt(state)

    assert "Fuege einen Abschnitt '## Unsicherheiten / Offene Punkte'" in prompt


def test_section_citation_rules_use_per_section_wording():
    """In section mode the citation rules use generic per-section wording."""
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_section_system_prompt(
        state,
        heading="Hintergrund / Kontext",
        instruction="Beschreibe Ausgangspunkt und zeitlichen Rahmen.",
        length_guidance="kompakt, aber mit den noetigen Kontextankern",
        section_position=2,
        section_total=6,
    )

    assert "ZITATIONS-REGELN" in prompt
    assert "In diesem Abschnitt" in prompt


def test_monolithic_citation_rules_use_global_wording():
    """Monolithic mode uses the report-wide citation wording."""
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_system_prompt(state)

    assert "ZITATIONS-REGELN" in prompt
    assert "In Kernaussagen und Detailabschnitten" in prompt


def test_section_abdeckungsregel_has_inline_marker_for_non_risiken_section():
    """ABDECKUNGSREGEL should not push the LLM to spawn a 'Risiken' sub-block in
    a non-Risiken section."""
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_section_system_prompt(
        state,
        heading="Analyse",
        instruction="Vertiefe Kernaspekte.",
        length_guidance="Tiefe nach Evidenzlage statt Wortlimit",
        section_position=3,
        section_total=6,
    )

    assert "ABDECKUNGSREGEL" in prompt
    assert "kennzeichne sie kurz inline mit '(unbestaetigt)'" in prompt
    # The "nenne sie unter 'Risiken / Unsicherheiten'" directive must not appear
    # in this section because Analyse is not the Risiken host.
    assert "nenne sie transparent unter" not in prompt


def test_language_directive_is_in_target_language_for_english():
    """Regression: 'Antworte auf Englisch' alone gets ignored by the LLM
    because all other instructions in the system prompt are in German.
    The directive must therefore be repeated in the target language.
    """
    state = _base_state(ReportProfile.COMPACT)
    state["answer_lang"] = "Englisch"

    prompt = build_answer_system_prompt(state)

    assert "ALWAYS respond in English" in prompt
    # The legacy German-only directive must NOT appear when the answer
    # language is English — otherwise the LLM gets a mixed signal.
    assert "Antworte IMMER auf Englisch" not in prompt


def test_language_directive_keeps_german_when_answer_lang_is_deutsch():
    """Default German answer keeps the German wording."""
    state = _base_state(ReportProfile.COMPACT)
    state["answer_lang"] = "Deutsch"

    prompt = build_answer_system_prompt(state)

    assert "Antworte IMMER auf Deutsch" in prompt
    assert "ALWAYS respond in English" not in prompt


def test_language_directive_for_french():
    state = _base_state(ReportProfile.COMPACT)
    state["answer_lang"] = "Franzoesisch"

    prompt = build_answer_system_prompt(state)

    assert "Réponds TOUJOURS en français" in prompt


def test_language_directive_falls_back_for_unknown_language():
    """Unknown language: the German wording is reused with the supplied label."""
    state = _base_state(ReportProfile.COMPACT)
    state["answer_lang"] = "Esperanto"

    prompt = build_answer_system_prompt(state)

    assert "Antworte IMMER auf Esperanto" in prompt


def test_language_directive_applies_to_section_prompt_too():
    """The section composer must also enforce the target language."""
    state = _base_state(ReportProfile.DEEP)
    state["answer_lang"] = "Englisch"

    prompt = build_answer_section_system_prompt(
        state,
        heading="Executive Summary",
        instruction="Beantworte die Frage direkt.",
        length_guidance="knapp und dicht",
        section_position=1,
        section_total=6,
    )

    assert "ALWAYS respond in English" in prompt


def test_deep_profile_first_round_queries_are_bounded_to_eight():
    """Deep starts with the operator-approved eight-query evidence wave."""
    overrides = tuning_for_report_profile(ReportProfile.DEEP).settings_overrides
    assert overrides.get("first_round_queries") == 8


def test_report_profiles_do_not_carry_provider_token_budgets():
    """Provider output budgets belong to provider constructors, not profiles."""
    compact = tuning_for_report_profile(ReportProfile.COMPACT)
    deep = tuning_for_report_profile(ReportProfile.DEEP)

    assert not hasattr(compact, "default_max_tokens")
    assert not hasattr(deep, "default_max_tokens")
    assert all(not hasattr(s, "max_output_tokens") for s in compact.answer_sections)
    assert all(not hasattr(s, "max_output_tokens") for s in deep.answer_sections)


def test_report_profiles_do_not_apply_hidden_answer_citation_caps():
    compact = tuning_for_report_profile(ReportProfile.COMPACT)
    deep = tuning_for_report_profile(ReportProfile.DEEP)

    assert not hasattr(compact, "answer_body_citation_cap")
    assert not hasattr(deep, "answer_body_citation_cap")
    assert not hasattr(compact, "answer_citation_block_char_budget")
    assert not hasattr(deep, "answer_citation_block_char_budget")


def test_deep_profile_keeps_broader_pre_report_evidence_view():
    deep = tuning_for_report_profile(ReportProfile.DEEP)

    assert deep.claim_max_items == 24
    assert deep.materialize_max_unverified == deep.materialize_max_total
