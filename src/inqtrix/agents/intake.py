"""Phase 0 — intake and readiness routing (§4).

One fast-tier structured call over the assignment plus a DETERMINISTIC
readiness router: the LLM describes the assignment, code decides where
to go next (ask the user first, plan straight away, or discover).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inqtrix.agents.patterns._structured import StructuredOutcome, structured_call
from inqtrix.agents.phase_models import AssignmentProfile
from inqtrix.agents.prompts import (
    agent_intake_system_prompt,
    build_agent_intake_prompt,
)

if TYPE_CHECKING:
    from inqtrix.providers.base import LLMProvider


def run_intake(
    llm: "LLMProvider",
    *,
    question: str,
    history: str,
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
    skills_block: str = "",
    artifact_registry: tuple[dict, ...] | list[dict] = (),
    last_response_form: str = "",
    prior_evidence_count: int = 0,
) -> StructuredOutcome:
    """The Phase-0 structured call; value is an AssignmentProfile."""
    return structured_call(
        llm,
        prompt=build_agent_intake_prompt(
            question,
            history,
            skills_block=skills_block,
            artifact_registry=artifact_registry,
            last_response_form=last_response_form,
            prior_evidence_count=prior_evidence_count,
        ),
        model_cls=AssignmentProfile,
        node="agent_intake",
        system=agent_intake_system_prompt(),
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def route_readiness(profile: AssignmentProfile | None) -> str:
    """Deterministic readiness router (no LLM, §4 Phase 0).

    ``ask_user_first`` when the assignment is ambiguous or carries
    clarification questions; ``plan_now`` for a clear, purely-web
    assignment without internal holdings; ``discover_first`` otherwise
    (the default). A failed intake (``None`` profile) discovers first —
    the safest route, and the failure is already marked loudly by the
    structured fallback.
    """
    if profile is None:
        return "discover_first"
    if (
        profile.scope_clarity == "ambiguous"
        or profile.clarification_questions
    ):
        return "ask_user_first"
    if (
        profile.scope_clarity == "clear"
        and profile.needs_web
        and not profile.needs_internal
        and not profile.needs_files
    ):
        return "plan_now"
    return "discover_first"
