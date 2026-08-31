"""Phase 9 — the memo critic (§4), built on the S1 pattern discipline.

Deterministic facts (citation coverage, quote verification,
contradiction mentions) are PRECOMPUTED and handed in — the critic
judges, it never measures (Prinzip 7: scores that exist get consumed).
Verdict ``revise`` buys EXACTLY one revision round.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inqtrix.agents.patterns._structured import StructuredOutcome, structured_call
from inqtrix.agents.phase_models import AgentCriticReport
from inqtrix.agents.prompts import (
    agent_critic_system_prompt,
    build_agent_critic_prompt,
)

if TYPE_CHECKING:
    from inqtrix.providers.base import LLMProvider


def precomputed_facts(
    *,
    coverage: dict[str, int],
    quote_checks: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    memo_markdown: str,
) -> str:
    """The deterministic fact block the critic prompt embeds."""
    unverified = [
        check["quote"] for check in quote_checks if not check["verified"]
    ]
    mentioned = sum(
        1
        for contradiction in contradictions
        if contradiction.get("internal_position", "")[:40] in memo_markdown
        or contradiction.get("external_position", "")[:40] in memo_markdown
    )
    lines = [
        f"Absaetze: {coverage.get('paragraphs', 0)}, davon mit Beleg: "
        f"{coverage.get('cited_paragraphs', 0)}",
        f"Verwendete Belege-Labels: {coverage.get('labels_used', 0)}",
        f"Woertliche Zitate geprueft: {len(quote_checks)}, "
        f"davon NICHT verifiziert: {len(unverified)}",
        f"Bekannte Widersprueche: {len(contradictions)}, im Memo "
        f"angesprochen: {mentioned}",
    ]
    return "\n".join(lines)


def run_critic(
    llm: "LLMProvider",
    *,
    memo_markdown: str,
    success_criteria: list[str],
    facts: str,
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
    user_guidance: str = "",
    skills_block: str = "",
) -> StructuredOutcome:
    """The fast-tier critic call; value is an AgentCriticReport."""
    return structured_call(
        llm,
        prompt=build_agent_critic_prompt(
            memo_markdown,
            success_criteria,
            facts,
            user_guidance=user_guidance,
            skills_block=skills_block,
        ),
        model_cls=AgentCriticReport,
        node="agent_critic",
        system=agent_critic_system_prompt(),
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
