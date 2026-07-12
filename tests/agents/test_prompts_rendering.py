"""Drift tripwire of the rendering-capabilities SSOT (plan M1 S5).

The block tells the model what the FRONTEND renderer actually supports
(MarkdownRenderer.tsx: remark-gfm, rehype-katex, rehype-pretty-code,
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
