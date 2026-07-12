"""Inqtrix middleware enforcing kernel-owned execution contracts."""

from __future__ import annotations

import logging
import hashlib
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger("inqtrix")

TOOL_LIMIT_REACHED_EVENT = "inqtrix.agent.tool_limit.reached"
SKILL_INPUTS_RESOLVED_MARKER = "[skill_inputs_resolved]"


def _decision_answer(
    question: dict[str, Any], decision: dict[str, Any]
) -> str:
    entry = (decision.get("answers") or {}).get(question.get("id", "")) or {}
    labels = {
        option["id"]: option["label"]
        for option in question.get("options", [])
    }
    parts = [
        labels[item]
        for item in entry.get("option_ids", [])
        if item in labels
    ]
    text = str(entry.get("text") or "").strip()
    if text:
        parts.append(text)
    if parts:
        return ", ".join(parts)
    return str(decision.get("answer") or "").strip()


def resolve_kernel_skill_inputs(
    skill: Any,
    *,
    clarification_scope: str,
) -> dict[str, str]:
    """Resolve required skill points before activating its instructions.

    The same resolver serves attached skills and dynamic ``load_skill``.
    Missing points become deterministic clarification rows and LangGraph
    interrupts; the skill remains inactive until every batch is answered.
    """
    from langgraph.types import interrupt

    from inqtrix.agents.clarification import build_clarification, sanitize_questions
    from inqtrix.agents.control_ports import ClarificationNotFound
    from inqtrix.agents.kernel.deps import kernel_deps, run_coro
    from inqtrix.agents.phase_models import (
        ClarificationOptionModel,
        ClarificationQuestionModel,
    )
    from inqtrix.agents.skills_runtime import (
        check_skill_points,
        skill_point_key,
        unanswered_required_points,
    )
    from inqtrix.model_routing import (
        describe_resolution,
        describe_unresolved_resolution,
    )

    deps = kernel_deps()
    provider_models = getattr(deps.llm, "models", None)
    resolution = (
        describe_unresolved_resolution("agent_skill_point_check", "")
        if provider_models is None
        else describe_resolution(
            "agent_skill_point_check",
            provider_models,
            "",
            requested_model="",
            requested_effort="",
        )
    )
    deps.emit("inqtrix.node.model_resolution", resolution)
    answers, usage = check_skill_points(
        deps.llm,
        skill=skill,
        question=deps.question,
        history=deps.session_history,
        model=resolution.get("model") or None,
        reasoning_effort=resolution.get("effort") or None,
        timeout=deps.timeout,
    )
    deps.book_usage(
        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    )
    missing = unanswered_required_points(skill, answers)
    for batch_index in range(0, len(missing), 3):
        points = missing[batch_index : batch_index + 3]
        questions = sanitize_questions(
            [
                ClarificationQuestionModel(
                    prompt=str(point.get("question") or ""),
                    options=[
                        ClarificationOptionModel(
                            label=str(option.get("label") or ""),
                            description=str(option.get("description") or ""),
                        )
                        for option in point.get("options", [])
                    ],
                    multi_select=bool(point.get("multi_select")),
                )
                for point in points
            ]
        )
        for question, point in zip(questions, points, strict=True):
            question["skill_key"] = skill_point_key(point)
        scope_hash = hashlib.sha1(
            clarification_scope.encode("utf-8")
        ).hexdigest()[:10]
        clarification_id = (
            f"clr_{deps.run_id[-12:]}_skill_{scope_hash}_{batch_index // 3}"
        )
        try:
            record = run_coro(
                deps.control.get_clarification(deps.run_id, clarification_id)
            )
        except ClarificationNotFound:
            record = run_coro(
                deps.control.create_clarification(
                    build_clarification(
                        deps.run_id,
                        questions=questions,
                        clarification_id=clarification_id,
                    )
                )
            )
            deps.emit(
                "inqtrix.agent.clarification.requested",
                {
                    "clarification_id": record.clarification_id,
                    "question": record.question,
                    "options": [dict(item) for item in record.options],
                    "question_count": len(record.questions),
                },
            )
        if record.status == "answered":
            decision = {
                "answer": record.answer,
                "option_id": record.option_id,
                "answers": dict(record.answers),
            }
        else:
            decision = interrupt(
                {"kind": "clarification", "id": clarification_id}
            )
        for question in record.questions:
            value = _decision_answer(dict(question), dict(decision or {}))
            if value:
                answers[str(question.get("skill_key") or "")] = value
    return answers


class KernelSkillInputMiddleware(AgentMiddleware):
    """Gate attached skills and inject only fully resolved skill blocks."""

    def before_model(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        del runtime
        messages = state.get("messages") or []
        if any(
            SKILL_INPUTS_RESOLVED_MARKER
            in str(getattr(message, "content", "") or "")
            for message in messages
        ):
            return None
        from inqtrix.agents.kernel.deps import kernel_deps
        from inqtrix.agents.skills_runtime import build_skills_block

        try:
            deps = kernel_deps()
        except RuntimeError:
            # Low-level harness contract tests intentionally assemble a
            # throwaway graph without per-run deps.
            return None
        if not deps.skills:
            return None
        resolved: dict[str, dict[str, str]] = {}
        for skill in deps.skills:
            resolved[skill.id] = resolve_kernel_skill_inputs(
                skill,
                clarification_scope=f"attached:{skill.id}",
            )
        deps.skill_answers.update(resolved)
        block = build_skills_block(deps.skills, resolved)
        return {
            "messages": [
                HumanMessage(
                    content=f"{SKILL_INPUTS_RESOLVED_MARKER}\n{block}"
                )
            ]
        }


class KernelToolBudgetExceeded(RuntimeError):
    """The model requested a tool-call batch beyond the run limit."""

    def __init__(self, *, attempted: int, limit: int, batch_size: int) -> None:
        self.attempted = attempted
        self.limit = limit
        self.batch_size = batch_size
        super().__init__(
            "Kernel-Tool-Budget ueberschritten: "
            f"{attempted} angefordert, maximal {limit}."
        )


class KernelToolBudgetMiddleware(AgentMiddleware):
    """Reject an overflowing model batch before any tool is dispatched.

    Counts are derived from checkpointed AI messages rather than mutable
    process state.  Re-entry and park/resume therefore produce the same total
    without double-counting, including multiple calls in one assistant turn.

    Args:
        max_tool_calls: Positive run-wide call ceiling.  The entire model
            batch is rejected when its cumulative total exceeds this value.
    """

    def __init__(self, max_tool_calls: int) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        self.max_tool_calls = max_tool_calls

    def after_model(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        """Validate the latest model batch at the pre-dispatch boundary."""
        del runtime
        messages = state.get("messages") or []
        ai_messages = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        if not ai_messages:
            return None
        latest_calls = list(ai_messages[-1].tool_calls or [])
        if not latest_calls:
            return None
        attempted = sum(
            len(message.tool_calls or []) for message in ai_messages
        )
        if attempted <= self.max_tool_calls:
            return None

        payload = {
            "attempted": attempted,
            "limit": self.max_tool_calls,
            "batch_size": len(latest_calls),
        }
        log.warning(
            "Kernel-Tool-Budget ueberschritten: attempted=%d limit=%d "
            "batch_size=%d; kompletter Batch wird nicht ausgefuehrt.",
            attempted,
            self.max_tool_calls,
            len(latest_calls),
        )
        from inqtrix.agents.kernel.deps import kernel_deps

        try:
            deps = kernel_deps()
        except RuntimeError:
            deps = None
        if deps is not None:
            deps.emit(TOOL_LIMIT_REACHED_EVENT, payload)
        raise KernelToolBudgetExceeded(
            attempted=attempted,
            limit=self.max_tool_calls,
            batch_size=len(latest_calls),
        )
