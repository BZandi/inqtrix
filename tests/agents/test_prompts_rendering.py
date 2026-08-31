"""Drift tripwire for the rendering-capabilities source of truth.

The block tells the model what the FRONTEND renderer actually supports
(MarkdownRenderer.tsx: remark-gfm, rehype-katex, bounded Shiki highlighting,
MermaidFigure). If a capability is ever removed from the renderer, this
test is the reminder to update BOTH places — and if someone rewords the
block, the consuming prompts must keep carrying it.
"""

from __future__ import annotations

from inqtrix.agents.prompts import (
    agent_answer_system_prompt,
    agent_synthesis_system_prompt,
    build_agent_answer_prompt,
    build_agent_outline_prompt,
    build_agent_section_prompt,
    rendering_capabilities_block,
)

RENDERER_FEATURES = ("Tabellen", "KaTeX", "mermaid", "Code")


def test_rendering_block_names_every_renderer_feature():
    block = rendering_capabilities_block()
    for feature in RENDERER_FEATURES:
        assert feature in block, feature
    # Explicit non-capabilities stay named (skipHtml is on, no emojis).
    assert "Kein HTML" in block
    assert "keine Emojis" in block


def test_rendering_block_states_usage_rules():
    """P7: the palette carries RULES, not just a feature list — the
    mandatory-table trigger and the no-decoration constraint."""
    block = rendering_capabilities_block()
    assert "ist eine Tabelle Pflicht" in block
    assert "Nichts Dekoratives" in block
    # Owner decision: no data plots for now.
    assert "xychart" not in block


def test_answer_and_synthesis_prompts_carry_the_block():
    block = rendering_capabilities_block()
    assert block in agent_answer_system_prompt()
    assert block in agent_synthesis_system_prompt()


def test_synthesis_prompts_require_a_small_nonredundant_citation_set():
    prompts = (
        agent_answer_system_prompt(),
        agent_synthesis_system_prompt(),
        build_agent_outline_prompt("Auftrag", ["Kriterium"], "[W1] Fakt"),
        build_agent_section_prompt(
            "Auftrag", "Kapitel", "Fokus", "[W1] Fakt", ""
        ),
        build_agent_answer_prompt("Auftrag", "[W1] Fakt", ""),
    )

    for prompt in prompts:
        assert "kleinste hinreichende" in prompt
        assert "nicht redundante" in prompt
    assert "1-3 Labels" in prompts[1]
    assert "niemals alle verfuegbaren Labels" in prompts[3]


# --- F-P0-QUELLEN: sources appear exactly once (the curated list) ------------


def test_no_model_source_sections_is_word_identical_on_every_surface():
    """P6A/F-P0-QUELLEN: THE prohibition sentence rides every answer
    surface verbatim — chat answer (system + closing), memo section,
    kernel system prompt, deep revision and the quick-web lane. A
    reworded copy on one surface is drift, not style."""
    from inqtrix.agents.prompts import (
        NO_MODEL_SOURCE_SECTIONS,
        build_agent_kernel_system_prompt,
        build_deep_revision_prompt,
        quick_web_answer_rules,
    )

    assert "KEINEN eigenen Quellen-" in NO_MODEL_SOURCE_SECTIONS
    assert "keine Roh-URLs" in NO_MODEL_SOURCE_SECTIONS
    surfaces = {
        "answer_system": agent_answer_system_prompt(),
        "synthesis_system": agent_synthesis_system_prompt(),
        "answer_prompt": build_agent_answer_prompt("Frage?", "(keine)", ""),
        "section_prompt": build_agent_section_prompt(
            "Frage?", "Abschnitt", "Fokus", "(keine)", ""
        ),
        "kernel_system": build_agent_kernel_system_prompt(),
        "deep_revision": build_deep_revision_prompt("Auftrag", "Bundle", []),
        "quick_web": quick_web_answer_rules(),
    }
    for name, text in surfaces.items():
        assert NO_MODEL_SOURCE_SECTIONS in text, name


def test_kernel_citation_block_defines_the_label_style():
    from inqtrix.agents.prompts import build_agent_kernel_system_prompt

    prompt = build_agent_kernel_system_prompt()
    assert "Zitierweise:" in prompt
    assert "[K1] [W2], nie [K1][W2]" in prompt


def test_deep_review_rubric_does_not_reward_model_source_lists():
    """The 'oder Quellen' escape hatch rewarded model-authored source
    sections in the Deep rubric — labels are the only accepted form."""
    from inqtrix.agents.prompts import build_deep_review_prompt

    prompt = build_deep_review_prompt("Auftrag", "Bundle")
    assert "Belege-Labels" in prompt
    assert "oder Quellen" not in prompt


def test_quick_web_lane_cites_labels_and_appends_no_source_section():
    """The quick lane cites with W-labels (no URL-linking instruction)
    and the server no longer appends a '### Quellen' section — the
    curated reference list is the one source surface."""
    import inspect

    from inqtrix.agents.kernel.algorithm import KernelAgentAlgorithm
    from inqtrix.agents.prompts import quick_web_answer_rules

    rules = quick_web_answer_rules()
    assert "[W1], [W2]" in rules
    assert "Verlinke Aussagen" not in rules
    source = inspect.getsource(
        KernelAgentAlgorithm._synthesize_quick_web_answer
    )
    assert "### Quellen" not in source
    assert "quick_web_answer_rules" in source


def test_registry_line_leads_with_the_file_name():
    """P9 (K4): named documents render `- name.md — Titel (...)`; kinds
    without a name keep the plain line, and the header carries the
    address-by-name instruction."""
    from inqtrix.agents.prompts import build_agent_session_context_sections

    section = build_agent_session_context_sections(
        artifact_registry=(
            {
                "artifact_id": "art_1",
                "kind": "deliverable",
                "title": "Marktbericht",
                "revision": 3,
                "updated_by": "agent",
                "name": "marktbericht.md",
            },
            {
                "artifact_id": "art_2",
                "kind": "evidence_bundle",
                "title": "Belege",
                "revision": 1,
                "updated_by": "agent",
            },
        ),
    )
    assert (
        "- marktbericht.md — Marktbericht (artifact_id art_1, Revision 3"
        in section
    )
    assert "- Belege (artifact_id art_2" in section
    assert "gegenueber dem Nutzer" in section
    assert "beim Dateinamen" in section


def test_kernel_prompt_carries_the_file_name_discipline():
    """P9 (K3): the model must address documents by file name, never by
    artifact_id — pinned on the composed kernel system prompt."""
    from inqtrix.agents.prompts import build_agent_kernel_system_prompt

    prompt = build_agent_kernel_system_prompt()
    assert "bei ihrem Dateinamen" in prompt
    assert "nie bei der artifact_id" in prompt
