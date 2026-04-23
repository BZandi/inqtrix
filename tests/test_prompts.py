"""Tests for prompt variants used by the final answer node."""

from __future__ import annotations

from inqtrix.nodes import _limit_prompt_citations_by_char_budget
from inqtrix.prompts import build_answer_section_system_prompt, build_answer_system_prompt
from inqtrix.report_profiles import ReportProfile, tuning_for_report_profile


def _base_state(report_profile: ReportProfile) -> dict[str, object]:
    return {
        "today_str": "2026-04-13",
        "answer_lang": "Deutsch",
        "context": "Block 1",
        "all_citations": [],
        "prompt_citations": [],
        "source_tier_counts": {},
        "claim_status_counts": {},
        "consolidated_claims": [],
        "report_profile": report_profile,
    }


def test_build_answer_system_prompt_compact_profile():
    prompt = build_answer_system_prompt(_base_state(ReportProfile.COMPACT))

    assert "**Kurzfazit**" in prompt
    assert "600-1200 Woerter" in prompt
    assert "Priorisiere vollstaendige, sauber abgeschlossene Abschnitte" in prompt
    assert "**Executive Summary**" not in prompt


def test_build_answer_system_prompt_deep_profile():
    prompt = build_answer_system_prompt(_base_state(ReportProfile.DEEP))

    assert "**Executive Summary**" in prompt
    assert "1800-2400 Woerter" in prompt
    assert "Priorisiere vollstaendige, sauber abgeschlossene Abschnitte" in prompt
    assert "**Risiken / Unsicherheiten**" in prompt
    assert "keine belastbare Evidenz vorliegt" in prompt


def test_build_answer_system_prompt_uses_runtime_managed_reference_sections():
    state = _base_state(ReportProfile.DEEP)
    state["prompt_citations"] = ["https://example.com/report"]
    state["all_citations"] = ["https://example.com/report"]

    prompt = build_answer_system_prompt(state)

    assert "Erzeuge KEINEN eigenen Referenz-, Quellen- oder Linkabschnitt" in prompt
    assert "Am Ende der Antwort fuege eine Quellenleiste ein" not in prompt


def test_build_answer_section_system_prompt_reuses_common_rules():
    state = _base_state(ReportProfile.DEEP)
    state["prompt_citations"] = ["https://example.com/report"]
    state["all_citations"] = ["https://example.com/report", "https://example.com/extra"]

    prompt = build_answer_section_system_prompt(
        state,
        heading="Analyse",
        instruction="Schreibe 3-5 Unterabschnitte mit Zahlen und Zusammenhaengen.",
        length_guidance="ca. 700-1000 Woerter mit `###`-Unterabschnitten",
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
    state["prompt_citations"] = ["https://example.com/a", "https://example.com/b"]
    state["all_citations"] = ["https://example.com/a", "https://example.com/b"]
    state["required_aspects"] = ["Status quo", "Position der Akteure"]
    state["uncovered_aspects"] = ["Position der Akteure"]
    return state


def test_section_executive_summary_omits_unsicherheiten_subsection_directive():
    """Executive Summary must NOT be told to add a '## Unsicherheiten' block.

    Regression: the global TRANSPARENZPFLICHT block previously injected this
    directive into every section, which made the Executive Summary section
    overspend its 1500-token budget by writing an extra unsolicited bullet
    list.
    """
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_section_system_prompt(
        state,
        heading="Executive Summary",
        instruction="Beantworte die Frage in 3-5 Saetzen.",
        length_guidance="ca. 140-220 Woerter, keine Bulletpoints",
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
        instruction="Liste 3-5 Bulletpoints zu Evidenzgrenzen.",
        length_guidance="ca. 120-220 Woerter als Bulletpoints",
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
        length_guidance="ca. 100-160 Woerter, 2-4 Saetze",
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
        length_guidance="ca. 120-180 Woerter",
        section_position=4,
        section_total=4,
    )

    assert "Fuege einen Abschnitt '## Unsicherheiten / Offene Punkte'" in prompt


def test_monolithic_mode_keeps_legacy_unsicherheiten_directive():
    """Backwards compatibility: the monolithic prompt path is unchanged."""
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_system_prompt(state)

    assert "Fuege einen Abschnitt '## Unsicherheiten / Offene Punkte'" in prompt


def test_section_citation_rules_avoid_naming_other_sections():
    """In section mode the citation rules must not name Kernaussagen / Detailanalyse."""
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_section_system_prompt(
        state,
        heading="Hintergrund / Kontext",
        instruction="Beschreibe Ausgangspunkt und zeitlichen Rahmen.",
        length_guidance="ca. 180-260 Woerter",
        section_position=2,
        section_total=6,
    )

    assert "ZITATIONS-REGELN" in prompt
    # Section-mode wording
    assert "Innerhalb dieser Section" in prompt
    # Legacy section-naming wording must be suppressed
    assert "In **Kernaussagen**" not in prompt
    assert "In **Detailanalyse**" not in prompt


def test_monolithic_citation_rules_keep_section_naming():
    """Backwards compatibility: monolithic mode references the old section names."""
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_system_prompt(state)

    assert "In **Kernaussagen**" in prompt
    assert "In **Detailanalyse**" in prompt
    assert "Innerhalb dieser Section" not in prompt


def test_section_abdeckungsregel_has_inline_marker_for_non_risiken_section():
    """ABDECKUNGSREGEL should not push the LLM to spawn a 'Risiken' sub-block in
    a non-Risiken section."""
    state = _weak_evidence_state(ReportProfile.DEEP)

    prompt = build_answer_section_system_prompt(
        state,
        heading="Analyse",
        instruction="Vertiefe Kernaspekte.",
        length_guidance="ca. 700-1000 Woerter",
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
        instruction="Beantworte die Frage in 3-5 Saetzen.",
        length_guidance="ca. 140-220 Woerter",
        section_position=1,
        section_total=6,
    )

    assert "ALWAYS respond in English" in prompt


def test_deep_profile_first_round_queries_minimum():
    """Phase 13: DEEP profile must seed the first round with at least 10 queries.

    Rationale: when an over-confident evaluator stops the loop after Round 0
    (e.g. GPT-4.1 reaching confidence_stop=9 immediately), Round 0 is the
    ONLY chance to broaden the source pool. The previous default of 8
    queries left the answer with ~11 unique domains; 10 lifts it to ~14-15
    without exceeding sensible parallelization.
    """
    overrides = tuning_for_report_profile(ReportProfile.DEEP).settings_overrides
    assert overrides.get("first_round_queries", 0) >= 10, (
        "DEEP must seed the first round with >= 10 queries; otherwise an "
        "early evaluator stop would leave the answer under-diversified."
    )


def test_section_limits_have_safety_margin_for_no_thinking_path():
    """Regression: Section max_output_tokens must give the LLM headroom even
    when extended thinking is not enabled (no auto-raise). Live measurements
    on 18.04.2026 showed the prior 'Kernaussagen=2000' cap landing at 95 %
    utilization in COMPACT no-thinking mode — borderline truncation. The
    bumped values below are the floor we want to defend.
    """
    compact = {
        s.heading: s.max_output_tokens
        for s in tuning_for_report_profile(ReportProfile.COMPACT).answer_sections
    }
    deep = {
        s.heading: s.max_output_tokens
        for s in tuning_for_report_profile(ReportProfile.DEEP).answer_sections
    }

    # COMPACT — Phase 9.2 minima.
    assert compact["Kurzfazit"] >= 1500
    assert compact["Kernaussagen"] >= 2500, (
        "Kernaussagen sat at 95% util in no-thinking mode — keep the safety margin."
    )
    assert compact["Detailanalyse"] >= 4000
    assert compact["Einordnung / Ausblick"] >= 1800

    # DEEP — Phase 9.2 minima.
    assert deep["Executive Summary"] >= 1800
    assert deep["Hintergrund / Kontext"] >= 2600
    assert deep["Analyse"] >= 6000
    assert deep["Perspektiven / Positionen"] >= 4000
    assert deep["Risiken / Unsicherheiten"] >= 2000
    assert deep["Fazit / Ausblick"] >= 1600


def test_limit_prompt_citations_by_char_budget_trims_long_source_list():
    citations = [
        f"https://example.com/research/{idx}/{'a' * 40}"
        for idx in range(12)
    ]

    trimmed = _limit_prompt_citations_by_char_budget(citations, char_budget=260)

    assert 1 <= len(trimmed) < len(citations)
    assert trimmed == citations[: len(trimmed)]
