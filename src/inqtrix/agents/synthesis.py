"""Phase 8 — memo synthesis into the canvas artifact (§4).

Outline first, then ONE call per section; every finished section flushes
into the memo artifact (revision++) so the canvas streams section-wise
(decision E12 — no delta protocol, the artifact row is the truth).
Citation validation, quote grounding, and coverage stats live in
``report_quality`` (shared with the kernel, Prinzip 4); this module
re-exports the names its callers and tests already import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inqtrix.agents.markdown import normalize_agent_markdown
from inqtrix.agents.patterns._structured import StructuredOutcome, structured_call
from inqtrix.agents.phase_models import (
    ReportOutline,
    SectionText,
)
from inqtrix.agents.prompts import (
    agent_answer_system_prompt,
    agent_synthesis_system_prompt,
    build_agent_answer_prompt,
    build_agent_outline_prompt,
    build_agent_section_prompt,
)
from inqtrix.agents.report_quality import (
    CitationValidationFailed,
    citation_coverage,
    cited_references,
    unknown_citation_labels,
    validate_and_repair_citations,
    verify_quotes,
)

if TYPE_CHECKING:
    from inqtrix.providers.base import LLMProvider

__all__ = [
    "CitationValidationFailed",
    "assemble_memo",
    "citation_coverage",
    "cited_references",
    "run_outline",
    "unknown_citation_labels",
    "verify_quotes",
    "write_chat_answer",
    "write_section",
]


def run_outline(
    llm: "LLMProvider",
    *,
    question: str,
    success_criteria: list[str],
    evidence_digest: str,
    prior_memo: str = "",
    skills_block: str = "",
    user_guidance: str = "",
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
) -> StructuredOutcome:
    """The outline call; value is a ReportOutline.

    ``prior_memo`` carries the session memo of an earlier turn (E15): when
    present the outline CONTINUES it instead of starting from scratch.
    """
    return structured_call(
        llm,
        prompt=build_agent_outline_prompt(
            question,
            success_criteria,
            evidence_digest,
            prior_memo=prior_memo,
            user_guidance=user_guidance,
            skills_block=skills_block,
        ),
        model_cls=ReportOutline,
        node="agent_synthesis",
        system=agent_synthesis_system_prompt(),
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def write_section(
    llm: "LLMProvider",
    *,
    question: str,
    section_title: str,
    section_focus: str,
    evidence_digest: str,
    contradictions_digest: str,
    skills_block: str = "",
    user_guidance: str = "",
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
    known_labels: list[str] | None = None,
) -> tuple[str, dict[str, int]]:
    """One section's prose; returns ``(markdown, usage)``.

    Structured on purpose: ``LLMProvider.complete`` returns bare text and
    DISCARDS token metadata, which would leave every section call
    unmetered on real providers — the structured path reports usage.
    """
    outcome = structured_call(
        llm,
        prompt=build_agent_section_prompt(
            question,
            section_title,
            section_focus,
            evidence_digest,
            contradictions_digest,
            skills_block=skills_block,
            user_guidance=user_guidance,
        ),
        model_cls=SectionText,
        node="agent_synthesis",
        system=agent_synthesis_system_prompt(),
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    section = outcome.value
    text = section.markdown if isinstance(section, SectionText) else ""
    normalized = normalize_agent_markdown(text.strip())
    if known_labels is None:
        return normalized, outcome.usage
    return validate_and_repair_citations(
        llm,
        markdown=normalized,
        known_labels=known_labels,
        usage=outcome.usage,
        system=agent_synthesis_system_prompt(),
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def write_chat_answer(
    llm: "LLMProvider",
    *,
    question: str,
    evidence_digest: str,
    contradictions_digest: str,
    history: str,
    prior_memo: str,
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
    skills_block: str = "",
    user_guidance: str = "",
    known_labels: list[str] | None = None,
) -> tuple[str, dict[str, int]]:
    """The ONE chat-form answer call (plan M1 S3); ``(markdown, usage)``.

    The chat deliverable skips the outline/section loop: one direct
    conversational answer over the same evidence digest, citation labels
    still mandatory. Structured for the same reason as
    :func:`write_section` — bare ``complete()`` discards token metadata.
    """
    outcome = structured_call(
        llm,
        prompt=build_agent_answer_prompt(
            question,
            evidence_digest,
            contradictions_digest,
            history=history,
            prior_memo=prior_memo,
            skills_block=skills_block,
            user_guidance=user_guidance,
        ),
        model_cls=SectionText,
        node="agent_answer",
        system=agent_answer_system_prompt(),
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    section = outcome.value
    text = section.markdown if isinstance(section, SectionText) else ""
    normalized = normalize_agent_markdown(text.strip())
    if known_labels is None:
        return normalized, outcome.usage
    return validate_and_repair_citations(
        llm,
        markdown=normalized,
        known_labels=known_labels,
        usage=outcome.usage,
        system=agent_answer_system_prompt(),
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def assemble_memo(
    title: str, sections: list[tuple[str, str]]
) -> str:
    """Join finished sections into the memo body."""
    parts = [f"# {title}"]
    for section_title, body in sections:
        parts.append(f"## {section_title}\n\n{body}")
    return normalize_agent_markdown("\n\n".join(parts))
