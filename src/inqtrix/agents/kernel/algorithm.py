"""The cognitive kernel algorithm.

``mode=agent_kernel`` is a SECOND registered ``AgentAlgorithm`` next to
the deterministic phase machine: an LLM tool-calling loop
on deepagents, executed strictly through the platform seams — same
RunService -> queue -> worker path, park/resume against control rows
(R5), sync ``graph.stream`` only, shared checkpointer with
``thread_id=run_id``.

Walking-skeleton surface: the ``ask_user`` tool with a full park/resume
round trip. Read tools, policy gates (compiled ``interrupt_on``
variants), canvas writes, and child runs land in M2 steps 6-8; the
system prompt stays a minimal interim constant until
``build_agent_kernel_system_prompt`` (step 9).
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
import threading
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, Any

from inqtrix.constants import REASONING_TIMEOUT
from inqtrix.agents.checkpoint_guard import ensure_checkpoint_restart_safe
from inqtrix.agents.control_ports import ApprovalNotFound, ApprovalRecord
from inqtrix.agents.kernel.chat_bridge import build_tool_chat_model
from inqtrix.agents.kernel.deps import (
    KernelDeps,
    kernel_deps,
    run_coro,
    set_kernel_deps,
)
from inqtrix.agents.kernel.interrupts import (
    CHILDREN_INTERRUPT,
    CLARIFICATION_INTERRUPT,
    TOOL_APPROVAL_INTERRUPT,
    translate_kernel_interrupt,
)
from inqtrix.agents.kernel.policy import interrupt_config_for
from inqtrix.agents.kernel.tools import build_kernel_tools
from inqtrix.agents.limit_contract import (
    LIMIT_CHOICE_CANCEL,
    LIMIT_CHOICE_PARTIAL,
    LIMIT_DECIDED_EVENT,
    LIMIT_REACHED_EVENT,
    QUICK_WEB_SEARCH_LIMIT,
    AgentLimitGate,
    LimitChoice,
    create_or_get_limit_gate,
    effective_extended_limit,
    latest_terminal_limit_choice,
    next_extended_limit,
)
from inqtrix.agents.markdown import normalize_agent_markdown
from inqtrix.agents.patterns._structured import observe_structured_retries
from inqtrix.agents.telemetry import model_retry_activity, provider_retry_activity
from inqtrix.agents.skills_runtime import (
    skill_model_pins,
    strictest_requires_plan,
)
from inqtrix.agents.web_execution_policy import derive_web_research_policy
from inqtrix.execution_authority import pinned_knowledge_collection_ids
from inqtrix.i18n import detect_ui_language
from inqtrix.model_routing import (
    describe_resolution,
    describe_unresolved_resolution,
)
from inqtrix.providers.base import observe_provider_retries
from inqtrix.urls import normalize_url, scrub_credential_urls

if TYPE_CHECKING:
    from inqtrix.agents.checkpointing import CheckpointerHandle
    from inqtrix.agents.control_ports import AgentControlStore
    from inqtrix.core.context import RunContext, RuntimeContext
    from inqtrix.core.results import AgentResult, RunRequest
    from inqtrix.providers.base import ChatTurn
    from inqtrix.settings import AgentPlatformSettings

log = logging.getLogger("inqtrix")

_KERNEL_NODE = "agent_kernel"

_WAITING_STATUS_BY_ORIGIN = {
    CLARIFICATION_INTERRUPT: "waiting_for_input",
    TOOL_APPROVAL_INTERRUPT: "waiting_for_approval",
    CHILDREN_INTERRUPT: "waiting_for_children",
}

PHASE_CHANGED_EVENT = "inqtrix.agent.phase.changed"
TOOL_STARTED_EVENT = "inqtrix.agent.tool.started"
TOOL_FINISHED_EVENT = "inqtrix.agent.tool.finished"
TODO_UPDATED_EVENT = "inqtrix.agent.todo.updated"
NARRATION_EVENT = "inqtrix.agent.narration"

_ARGS_PREVIEW_LIMIT = 200
"""Character cap of the redacted tool-args preview in events."""

def _tool_intent_narration(
    tool_calls: list[dict[str, Any]],
    *,
    language: str,
) -> str:
    """Build one localized status from tool identity, never model prose.

    Provider messages may attach a short factual answer or a complete
    Markdown draft to a tool request.  Neither is trustworthy process
    narration.  The transcript therefore derives its status solely from the
    capability names already accepted by the kernel policy; the answer body
    remains owned exclusively by the answer publisher.
    """

    names = {
        str(call.get("name", "") or "")
        for call in tool_calls
        if isinstance(call, dict)
    }
    if names and names <= {
        "search_project_knowledge",
        "read_project_document",
        "web_instant",
    }:
        return (
            "Ich prüfe jetzt die benötigten Quellen."
            if language == "de"
            else "I’m checking the required sources now."
        )
    if names and names <= {"write_canvas", "read_canvas", "propose_editor_patch"}:
        return (
            "Ich bearbeite jetzt den angeforderten Arbeitsstand."
            if language == "de"
            else "I’m working on the requested artifact now."
        )
    if names and names <= {"run_web_research", "run_deep_mission", "delegate_batch"}:
        return (
            "Ich vertiefe jetzt die erforderlichen Arbeitsstränge."
            if language == "de"
            else "I’m expanding the required workstreams now."
        )
    return (
        "Ich führe jetzt den nächsten erforderlichen Arbeitsschritt aus."
        if language == "de"
        else "I’m carrying out the next required step now."
    )

_SUPERSTEPS_PER_TOOL_TURN = 8
"""LangGraph super-steps consumed by ONE model turn that calls a tool.

The kernel's ``recursion_limit`` counts super-steps, not model turns, and
every middleware hook is its own graph node: ``before_model`` (skill
inputs, sufficiency) -> model -> ``after_model`` (child-batch guard, tool
budget, todo, and HITL when the policy gates) -> the tool node. Measured
against the compiled production graph: 8 per tool turn, identical across
the policy variants (the HITL node does not change the price).

Budgets below are expressed as TOOL TURNS multiplied by this constant,
so the ceilings say what they mean. ``tests/agents/
test_harness_kernel_contract.py::test_supersteps_per_tool_turn_is_pinned``
measures the compiled graph and fails when a middleware change makes this
value wrong — the number is derived, never guessed (the sufficiency
middleware raised it from 7 to 8 exactly this way)."""

_ANSWER_TURN_SUPERSTEPS = 9
"""Super-steps of the bare final answer turn (measured alongside the
per-tool-turn price; the answer turn carries the graph's entry/exit
overhead, so it costs ONE more than a tool turn). A ceiling formula is
always ``_ANSWER_TURN_SUPERSTEPS + turns * _SUPERSTEPS_PER_TOOL_TURN``
— pinned by the same contract test."""

_SCHNELL_TOOL_TURNS = 3
"""Tool turns the ``schnell`` tier may spend BESIDE the answer turn:
the ONE published ``web_instant`` call, one knowledge/todo turn (both
allowed in schnell), and one slack turn for a failed or blocked call.
The tier publishes ``web_instant_budget=1``; a clamp that cannot afford
that call plus the answer would make every schnell tool run a
guaranteed ``GraphRecursionError`` (published != enforced) — the
pre-recalibration literal ``8`` did exactly that."""


def _emit_kernel_model_retry(
    deps: KernelDeps,
    node: str,
    notice: dict[str, Any],
) -> None:
    """Emit one Kernel model retry through the shared activity channel."""
    deps.emit(
        "inqtrix.agent.activity",
        model_retry_activity(notice, node=node),
    )


class _DepsChatProvider:
    """Duck-typed ``LLMProvider.chat`` delegate reading kernel deps.

    The compiled kernel graph (and thus its bound chat model) is shared
    across runs, but model override, effort, and timeout are PER
    SEGMENT — this delegate resolves them from the deps ContextVar at
    call time, so one compiled graph serves every run and stack. The
    model-call boundary doubles as the kernel's abort chokepoint
    (cancel event + token budget); usage booking happens ONCE via the
    bridge's ``usage_hook``, not here.
    """

    def chat(
        self,
        messages: Any,
        *,
        tools: Any = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        timeout: float = REASONING_TIMEOUT,
        **kwargs: Any,
    ) -> "ChatTurn":
        deps = kernel_deps()
        deps.check_abort()
        with observe_provider_retries(
            deps.llm,
            lambda notice: _emit_kernel_model_retry(
                deps, "agent_kernel", notice
            ),
        ):
            return deps.llm.chat(
                messages,
                tools=tools,
                model=deps.model,
                reasoning_effort=deps.reasoning_effort,
                timeout=deps.timeout,
                **kwargs,
            )

    def supports_tool_calls(self, *, model: str | None = None) -> bool:
        return True


class KernelAgentAlgorithm:
    """The conversational kernel over the platform seams.

    Args:
        control_store: Clarification/approval rows (park truth, R5).
        checkpointer: The E8 checkpointer handle; REQUIRED — the kernel
            never registers without one (container gate).
        platform: Server-side agent limits (never prompted).
        capability_registry: Read-only tool surface (wave-1
            capabilities); ``None`` degrades every capability tool to
            a visible not-available result.
        permission_service: Resolves the owner's visibility context per
            segment (E5: the agent's tools see exactly what its OWNER
            sees); ``None`` keeps the historical unscoped view of the
            anonymous/static modes.
        run_service: Child-run submission seam (research fan-out +
            deep-mission delegation); ``None`` degrades the child tools
            to a visible not-available result.
        resolver: Builds child ``ResolvedAgentContext``s (E18);
            required together with *run_service*.
    """

    id = "agent_kernel"
    display_name = "Agent-Kernel"

    def __init__(
        self,
        *,
        control_store: "AgentControlStore",
        checkpointer: "CheckpointerHandle",
        platform: "AgentPlatformSettings",
        capability_registry: Any = None,
        permission_service: Any = None,
        run_service: Any = None,
        resolver: Any = None,
        skill_service: Any = None,
        agent_memory_service: Any = None,
    ) -> None:
        self._control = control_store
        self._checkpointer = checkpointer
        self._platform = platform
        self._capabilities = capability_registry
        self._permissions = permission_service
        self._run_service = run_service
        self._resolver = resolver
        self._skill_service = skill_service
        # The SAME long-term memory service the mission engine uses (F9):
        # one memory stack, opt-in per user, never evidentiary.
        self._memory = agent_memory_service
        # One compiled graph per policy: interrupt_on is compile-time
        # and the policy is fixed per run.
        self._graphs: dict[tuple[str, str, str, str, int], Any] = {}
        self._graph_lock = threading.Lock()

    def capabilities(self) -> dict[str, Any]:
        """Registry manifest entry (no research-graph streaming)."""
        return {
            "requires": ["llm"],
            "streams_events": True,
            "supports_chat_completions": False,
            "terminal_node": _KERNEL_NODE,
            "produces": ["markdown"],
            "interrupts": ["clarification", "approval"],
        }

    # -- execution --------------------------------------------------------- #

    def run(
        self,
        request: "RunRequest",
        *,
        runtime: "RuntimeContext",
        context: "RunContext",
    ) -> "AgentResult":
        """Execute or RESUME one kernel run (segment-aware, R5)."""
        from inqtrix.core.results import AgentResult

        if context.run_id is None or context.park is None:
            raise RuntimeError(
                "agent_kernel laeuft nur ueber /v1/runs (Park-Faehigkeit "
                "und run_id erforderlich)."
            )
        llm = context.providers.llm
        if not llm.supports_tool_calls():
            # Per-request stacks may resolve a different provider than
            # the registration-time default — fail loudly, never degrade
            # to a tool-less loop.
            raise RuntimeError(
                "Der aufgeloeste LLM-Provider unterstuetzt kein natives "
                "Tool-Calling; mode=agent_kernel ist fuer diesen Stack "
                "nicht verfuegbar."
            )
        run_id = context.run_id
        autonomy = (
            request.autonomy or self._platform.default_autonomy
        ).strip() or "balanced"
        depth = context.agent_settings.depth
        if request.execution_directive and depth != "normal":
            log.info(
                "Execution-Directive %s erzwingt depth=normal "
                "(angefordert: %s).",
                request.execution_directive,
                depth,
            )
            depth = "normal"
        skill_records = self._load_request_skills(request, context)
        pin_tier, pin_effort = skill_model_pins(skill_records)
        deps = self._build_deps(
            run_id,
            llm,
            context,
            request,
            pin_tier=pin_tier,
            pin_effort=pin_effort,
            depth=depth,
        )
        deps.depth = depth
        deps.question = scrub_credential_urls(request.question)
        deps.session_id = request.session_id or ""
        deps.run_service = self._run_service
        deps.resolver = self._resolver
        deps.principal = context.principal
        deps.autonomy = autonomy
        deps.skill_service = self._skill_service
        deps.stack_name = getattr(context, "stack_name", "") or ""
        # Long-term memory (F9): the mission engine's service + opt-in
        # semantics — resolved once per segment; a preference-read error
        # degrades to False inside opt_in_enabled (never silently on).
        deps.memory = self._memory
        if self._memory is not None and context.principal is not None:
            deps.memory_opt_in = run_coro(
                self._memory.opt_in_enabled(context.principal)
            )
        from inqtrix.agents.source_policy import effective_source_policy

        deps.source_policy = effective_source_policy(
            request.source_policy, request.execution_directive
        )
        deps.execution_directive = request.execution_directive
        tier = getattr(context.agent_settings, "agent_tier", "") or ""
        research_policy = derive_web_research_policy(
            depth=depth,
            admitted_directive="web_research" in request.tool_directives,
            tier=tier or None,
        )
        deps.tier = tier
        deps.web_research_allowed = research_policy.allowed
        deps.web_research_profile = research_policy.profile
        deps.hydrate_evidence()
        deps.capability_context = dataclass_replace(
            deps.capability_context,
            question=deps.question,
        )
        for record in skill_records:
            deps.activate_skill(record)
        if (
            request.execution_directive == "quick_web"
            and strictest_requires_plan(deps.skills) == "always"
        ):
            raise RuntimeError(
                "quick_web ist mit einem angehaengten Skill, der einen "
                "Plan erzwingt, nicht vereinbar."
            )
        if autonomy == "autonomous" and (
            strictest_requires_plan(deps.skills) == "always"
        ):
            # The kernel has no plan phase — `requires_plan: always`
            # (the one lever a skill author has to force human review)
            # maps onto the gated policy variant instead of silently
            # no-opping in Auto. Deterministic per segment: derived
            # from the persisted request skills, so every resume picks
            # the same compiled graph.
            log.info(
                "Run %s: Skill mit requires_plan=always erzwingt den "
                "Standard-Freigabemodus (autonomous -> balanced).",
                run_id,
            )
            autonomy = "balanced"
            deps.autonomy = autonomy
        if request.execution_directive == "quick_web":
            self._hydrate_session_context(request, deps)
            with observe_structured_retries(
                lambda node, notice: _emit_kernel_model_retry(
                    deps, node, notice
                )
            ):
                return self._run_quick_web(request, context, deps)
        (
            base_tool_limit,
            tool_ceiling,
            base_step_limit,
            step_ceiling,
        ) = configured_kernel_limits(self._platform, depth=depth, tier=tier)
        deps.tool_call_ceiling = tool_ceiling
        deps.tool_call_limit = effective_extended_limit(
            self._control,
            run_id=run_id,
            kind="tool_calls",
            base=base_tool_limit,
            ceiling=tool_ceiling,
            run_async=run_coro,
        )
        deps.step_ceiling = step_ceiling
        deps.step_limit = effective_extended_limit(
            self._control,
            run_id=run_id,
            kind="steps",
            base=base_step_limit,
            ceiling=step_ceiling,
            run_async=run_coro,
        )
        graph = self._compiled_graph(
            autonomy,
            source_policy=deps.source_policy,
            execution_directive=request.execution_directive,
            max_tool_calls=deps.tool_call_limit,
        )
        state_config = {"configurable": {"thread_id": run_id}}
        existing = graph.get_state(state_config)
        deps.prior_usage = _checkpointed_usage(existing)
        deps.tool_use_counts = _checkpointed_tool_use_counts(existing)
        deps.tool_calls_used = _checkpointed_tool_call_count(existing)
        (
            deps.emitted_tool_start_ids,
            deps.emitted_tool_finish_ids,
        ) = _checkpointed_tool_event_ids(existing)
        deps.checkpointed_steps = _checkpointed_steps(existing)
        self._reactivate_loaded_skills(deps, existing)

        terminal_limit = latest_terminal_limit_choice(
            self._control, run_id=run_id, run_async=run_coro
        )
        if terminal_limit is not None:
            gate, choice, record = terminal_limit
            return self._apply_terminal_limit_choice(
                request,
                deps,
                gate=gate,
                choice=choice,
                clarification_id=record.clarification_id,
            )

        remaining_steps = deps.step_limit - deps.checkpointed_steps
        if remaining_steps <= 0 and bool(getattr(existing, "next", ())):
            gate = AgentLimitGate(
                kind="steps",
                current=deps.step_limit,
                proposed=next_extended_limit(
                    current=deps.step_limit,
                    ceiling=deps.step_ceiling,
                ),
                ceiling=deps.step_ceiling,
                used=deps.checkpointed_steps,
            )
            return self._park_for_limit(request, context, deps, gate=gate)

        config = {
            "configurable": {"thread_id": run_id},
            # LangGraph applies this bound to one invocation. Subtracting
            # the checkpoint's cumulative step coordinate turns it into the
            # published run-wide allowance across approvals and resumes.
            "recursion_limit": max(1, remaining_steps),
        }
        graph_input = self._graph_input(request, run_id, existing, deps)

        from inqtrix.exceptions import AgentCancelled, AgentTokenBudgetExceeded

        deps.emit(
            PHASE_CHANGED_EVENT,
            {
                "phase": "execution",
                "previous_phase": "",
                "snapshot": {
                    "current_node": _KERNEL_NODE,
                    "phase": "execution",
                    "execution": self._execution_payload(
                        request, deps, consent_reason=_kernel_consent_reason(deps)
                    ),
                },
            },
        )
        set_kernel_deps(deps)
        try:
            with observe_structured_retries(
                lambda node, notice: _emit_kernel_model_retry(
                    deps, node, notice
                )
            ):
                interrupts = [
                    item
                    for update in graph.stream(
                        graph_input, config=config, stream_mode="updates"
                    )
                    if isinstance(update, dict)
                    for item in _observe_update(deps, update)
                ]
        except AgentCancelled as exc:
            # Cancel/budget stop at the model boundary: the segment's
            # spend is returned (execute_run_request books it before
            # cancelling), the checkpoint stays for post-mortems.
            if isinstance(exc, AgentTokenBudgetExceeded):
                token_used = sum(deps.prior_usage.values()) + sum(
                    deps.usage.values()
                )
                token_payload = {
                    "kind": "tokens",
                    "used": max(0, token_used),
                    "limit": deps.token_budget,
                    "ceiling": deps.token_budget,
                    "extendable": False,
                    "recoverable": False,
                    "reason": "operator_ceiling_exactly_once_required",
                    "state": "failed_at_model_boundary",
                }
                deps.emit(LIMIT_REACHED_EVENT, token_payload)
                deps.emit(
                    NARRATION_EVENT,
                    {
                        "narration_id": f"n-limit-tokens-{deps.run_id[-12:]}",
                        "kind": "limit",
                        "text": (
                            "Das feste serverseitige Tokenlimit ist erreicht. "
                            "Der Lauf wurde ohne automatische Teilantwort beendet; "
                            "eine Erweiterung ist für diesen Lauf nicht möglich."
                        ),
                        "phase": "execution",
                        "final": True,
                    },
                )
            return AgentResult(
                answer="",
                result_type="agent_kernel_result",
                raw={
                    "answer": "",
                    "usage": dict(deps.usage),
                    "result_state": {
                        "cancelled": True,
                        "cancel_reason": (
                            "token_budget_exceeded"
                            if isinstance(exc, AgentTokenBudgetExceeded)
                            else "client_requested_cancel"
                        ),
                        "execution": self._execution_payload(
                            request,
                            deps,
                            consent_reason=_kernel_consent_reason(deps),
                        ),
                    },
                },
            )
        except Exception as exc:
            # Both loop ceilings stop at checkpoint-safe boundaries. They
            # therefore reuse the native clarification wait/resume contract
            # instead of becoming a generic failure or a silent partial.
            from langgraph.errors import GraphRecursionError

            from inqtrix.agents.kernel.middleware import (
                KernelToolBudgetExceeded,
            )

            if isinstance(exc, GraphRecursionError):
                stopped = graph.get_state(state_config)
                deps.checkpointed_steps = max(
                    deps.step_limit, _checkpointed_steps(stopped)
                )
                return self._park_for_limit(
                    request,
                    context,
                    deps,
                    gate=AgentLimitGate(
                        kind="steps",
                        current=deps.step_limit,
                        proposed=next_extended_limit(
                            current=deps.step_limit,
                            ceiling=deps.step_ceiling,
                        ),
                        ceiling=deps.step_ceiling,
                        used=deps.checkpointed_steps,
                    ),
                )
            if isinstance(exc, KernelToolBudgetExceeded):
                deps.tool_calls_used = max(
                    0, exc.attempted - exc.batch_size
                )
                return self._park_for_limit(
                    request,
                    context,
                    deps,
                    gate=AgentLimitGate(
                        kind="tool_calls",
                        current=deps.tool_call_limit,
                        proposed=next_extended_limit(
                            current=deps.tool_call_limit,
                            ceiling=deps.tool_call_ceiling,
                            required=exc.attempted,
                        ),
                        ceiling=deps.tool_call_ceiling,
                        used=deps.tool_calls_used,
                    ),
                )
            raise
        finally:
            set_kernel_deps(None)

        if interrupts:
            if len(interrupts) > 1:
                # Parallel gates have no defined resume mapping yet —
                # parking would strand the run half-answered (loud
                # failure is safer than an undefined resume).
                raise RuntimeError(
                    "Mehrere parallele Kernel-Interrupts in einem "
                    "Segment werden noch nicht unterstuetzt."
                )
            origin, payload = translate_kernel_interrupt(
                interrupts[0].value
            )
            if origin == TOOL_APPROVAL_INTERRUPT:
                self._ensure_tool_approval(
                    run_id,
                    deps,
                    interrupt_id=str(interrupts[0].id),
                    payload=payload,
                )
            context.park(_WAITING_STATUS_BY_ORIGIN[origin])
            return AgentResult(
                answer="",
                result_type="agent_kernel_parked",
                raw={
                    "answer": "",
                    "usage": dict(deps.usage),
                    "result_state": {
                        "parked": True,
                        "cancelled": False,
                        "execution": self._execution_payload(
                            request,
                            deps,
                            consent_reason=_kernel_consent_reason(deps),
                        ),
                    },
                },
            )

        state = graph.get_state(config)
        deps.checkpointed_steps = _checkpointed_steps(state)
        deps.tool_calls_used = _checkpointed_tool_call_count(state)
        answer = _final_answer(state.values.get("messages") or [])
        if not answer:
            self._checkpointer.delete_thread(run_id)
            raise RuntimeError(
                "Der Kernel endete ohne Antworttext (finish ohne "
                "AI-Nachricht)."
            )
        canvases = self._current_run_canvases(deps)
        deps.effective_response_form = "canvas" if canvases else "chat"
        if depth == "deep":
            # BEFORE delete_thread: a redelivered segment that died
            # inside the verify pass must still find its checkpoint.
            try:
                with observe_structured_retries(
                    lambda node, notice: _emit_kernel_model_retry(
                        deps, node, notice
                    )
                ):
                    answer = self._deep_verify(
                        deps,
                        request,
                        answer,
                        canvases,
                        messages=list(state.values.get("messages") or []),
                    )
            except AgentCancelled as exc:
                # Same contract as the model-boundary cancel above:
                # spend is booked by execute_run_request, the run ends
                # CANCELLED (never a generic failure).
                return AgentResult(
                    answer="",
                    result_type="agent_kernel_result",
                    raw={
                        "answer": "",
                        "usage": dict(deps.usage),
                        "result_state": {
                            "cancelled": True,
                            "cancel_reason": (
                                "token_budget_exceeded"
                                if isinstance(exc, AgentTokenBudgetExceeded)
                                else "client_requested_cancel"
                            ),
                            "execution": self._execution_payload(
                                request,
                                deps,
                                consent_reason=_kernel_consent_reason(deps),
                            ),
                        },
                    },
                )
        answer = _validate_kernel_answer_citations(
            deps,
            answer,
            list(deps.evidence_refs.values()),
        )
        answer_claim_bindings: list[dict[str, Any]] = []
        # Answer-artifact persistence belongs to the native RunService
        # publisher.  The kernel returns the post-verify Markdown and its
        # reference state; publication begins only after this algorithm has
        # completed, so no client can observe the final body before streaming.
        _stage_kernel_memory(deps, answer)
        self._checkpointer.delete_thread(run_id)
        deps.emit(
            PHASE_CHANGED_EVENT,
            {
                "phase": "done",
                "previous_phase": "execution",
                "snapshot": {
                    "current_node": _KERNEL_NODE,
                    "phase": "done",
                    "execution": self._execution_payload(
                        request, deps, consent_reason=_kernel_consent_reason(deps)
                    ),
                },
            },
        )
        # F5: the surfaced reference list follows the answer's citations
        # (cited-only, basis fallback via _result_references).  The central
        # publisher persists this exact exported projection on the answer
        # artifact. all/allowed_citations stay the FULL ledger URLs: they are
        # source-ordering inputs, not the user-facing reference list.
        report_references = _result_references(
            answer, list(deps.evidence_refs.values())
        )
        return AgentResult(
            answer=answer,
            result_type="agent_kernel_result",
            raw={
                "answer": answer,
                "usage": dict(deps.usage),
                "result_state": {
                    "answer": answer,
                    "cancelled": False,
                    "report_references": report_references,
                    "answer_claim_bindings": answer_claim_bindings,
                    "all_citations": [
                        str(item.get("url") or "")
                        for item in deps.evidence_refs.values()
                        if item.get("url")
                    ],
                    "allowed_citations": [
                        str(item.get("url") or "")
                        for item in deps.evidence_refs.values()
                        if item.get("url")
                    ],
                    "execution": self._execution_payload(
                        request, deps, consent_reason=_kernel_consent_reason(deps)
                    ),
                },
            },
        )

    def _park_for_limit(
        self,
        request: "RunRequest",
        context: "RunContext",
        deps: KernelDeps,
        *,
        gate: AgentLimitGate,
    ) -> "AgentResult":
        """Park one checkpoint-safe limit through the native input gate."""
        from inqtrix.core.results import AgentResult

        record, created = create_or_get_limit_gate(
            self._control,
            run_id=deps.run_id,
            gate=gate,
            run_async=run_coro,
        )
        payload = gate.payload(clarification_id=record.clarification_id)
        deps.emit(LIMIT_REACHED_EVENT, {**payload, "state": "waiting_for_input"})
        if created:
            deps.emit(
                "inqtrix.agent.clarification.requested",
                {
                    "clarification_id": record.clarification_id,
                    "question": record.question,
                    "options": [dict(item) for item in record.options],
                    "question_count": 1,
                },
            )
            deps.emit(
                NARRATION_EVENT,
                {
                    "narration_id": f"n-limit-{record.clarification_id}",
                    "kind": "limit",
                    "text": record.question,
                    "phase": "execution",
                    "final": True,
                },
            )
        context.park("waiting_for_input")
        return AgentResult(
            answer="",
            result_type="agent_kernel_parked",
            raw={
                "answer": "",
                "usage": dict(deps.usage),
                "result_state": {
                    "parked": True,
                    "cancelled": False,
                    "limit": payload,
                    "execution": self._execution_payload(
                        request,
                        deps,
                        consent_reason=_kernel_consent_reason(deps),
                    ),
                },
            },
        )

    def _apply_terminal_limit_choice(
        self,
        request: "RunRequest",
        deps: KernelDeps,
        *,
        gate: AgentLimitGate,
        choice: LimitChoice,
        clarification_id: str,
    ) -> "AgentResult":
        """Apply an explicit partial/cancel choice exactly once on resume."""
        from inqtrix.core.results import AgentResult

        payload = {
            **gate.payload(clarification_id=clarification_id),
            "choice": choice,
            "state": "decided",
        }
        deps.emit(LIMIT_DECIDED_EVENT, payload)
        if choice == LIMIT_CHOICE_CANCEL:
            self._checkpointer.delete_thread(deps.run_id)
            return AgentResult(
                answer="",
                result_type="agent_kernel_result",
                raw={
                    "answer": "",
                    "usage": dict(deps.usage),
                    "result_state": {
                        "cancelled": True,
                        "cancel_reason": "limit_cancelled_by_user",
                        "limit": payload,
                        "execution": self._execution_payload(
                            request,
                            deps,
                            consent_reason=_kernel_consent_reason(deps),
                        ),
                    },
                },
            )
        if choice != LIMIT_CHOICE_PARTIAL:
            raise RuntimeError(f"Unbekannte terminale Limit-Entscheidung: {choice!r}")

        answer = _partial_limit_answer(gate, list(deps.evidence_refs.values()))
        answer = _validate_kernel_answer_citations(
            deps,
            answer,
            list(deps.evidence_refs.values()),
        )
        answer_claim_bindings: list[dict[str, Any]] = []
        deps.effective_response_form = "chat"
        self._checkpointer.delete_thread(deps.run_id)
        deps.emit(
            PHASE_CHANGED_EVENT,
            {
                "phase": "done",
                "previous_phase": "execution",
                "snapshot": {
                    "current_node": _KERNEL_NODE,
                    "phase": "done",
                    "execution": self._execution_payload(
                        request,
                        deps,
                        consent_reason=_kernel_consent_reason(deps),
                    ),
                    "limit": payload,
                },
            },
        )
        references = _result_references(answer, list(deps.evidence_refs.values()))
        return AgentResult(
            answer=answer,
            result_type="agent_kernel_result",
            raw={
                "answer": answer,
                "usage": dict(deps.usage),
                "result_state": {
                    "answer": answer,
                    "cancelled": False,
                    "partial": True,
                    "partial_reason": "limit_reached",
                    "limit": payload,
                    "report_references": references,
                    "answer_claim_bindings": answer_claim_bindings,
                    "all_citations": [
                        str(item.get("url") or "")
                        for item in deps.evidence_refs.values()
                        if item.get("url")
                    ],
                    "allowed_citations": [
                        str(item.get("url") or "")
                        for item in deps.evidence_refs.values()
                        if item.get("url")
                    ],
                    "execution": self._execution_payload(
                        request,
                        deps,
                        consent_reason=_kernel_consent_reason(deps),
                    ),
                },
            },
        )

    def _execution_payload(
        self,
        request: "RunRequest",
        deps: KernelDeps,
        *,
        consent_reason: str,
    ) -> dict[str, object]:
        """Canonical Agent Desk execution block for this kernel run."""
        from inqtrix.agents.source_policy import execution_payload

        limits: dict[str, object] = {}
        if request.execution_directive == "quick_web":
            limits["web_searches"] = {
                "used": max(0, int(deps.tool_use_counts.get("web", 0) or 0)),
                "limit": QUICK_WEB_SEARCH_LIMIT,
                "ceiling": QUICK_WEB_SEARCH_LIMIT,
                "recoverable": False,
                "extendable": False,
                "reason": "direct_route_single_search",
            }
        if deps.tool_call_limit > 0:
            limits["tool_calls"] = {
                "used": max(0, deps.tool_calls_used),
                "limit": deps.tool_call_limit,
                "ceiling": max(deps.tool_call_limit, deps.tool_call_ceiling),
                "recoverable": True,
                "extendable": deps.tool_call_limit < deps.tool_call_ceiling,
            }
        if deps.step_limit > 0:
            limits["steps"] = {
                "used": max(0, deps.checkpointed_steps),
                "limit": deps.step_limit,
                "ceiling": max(deps.step_limit, deps.step_ceiling),
                "recoverable": True,
                "extendable": deps.step_limit < deps.step_ceiling,
            }
        token_used = sum(deps.prior_usage.values()) + sum(deps.usage.values())
        if deps.token_budget > 0:
            limits["tokens"] = {
                "used": max(0, token_used),
                "limit": deps.token_budget,
                "ceiling": deps.token_budget,
                "recoverable": False,
                "extendable": False,
                "reason": "operator_ceiling_exactly_once_required",
            }

        return execution_payload(
            execution_directive=request.execution_directive,
            effective_mode="agent_kernel",
            response_form=(
                deps.effective_response_form
                or (
                    "chat"
                    if request.execution_directive
                    else request.response_form or "auto"
                )
            ),
            depth=deps.depth,
            model=deps.model,
            reasoning_effort=deps.reasoning_effort,
            source_policy=deps.source_policy,
            consent_reason=consent_reason,
            tool_use_counts=deps.tool_use_counts,
            limits=limits,
        )

    def _run_quick_web(
        self,
        request: "RunRequest",
        context: "RunContext",
        deps: KernelDeps,
    ) -> "AgentResult":
        """Execute the isolated one-search quick-web lane.

        The path intentionally bypasses the kernel graph: no plan, child,
        RAG, canvas, or model-selected second tool can appear. Balanced is
        already scoped consent because the user explicitly requested
        ``quick_web``; strict creates the existing read-only tool approval
        row and resumes with its reviewed query; autonomous runs directly.
        """
        from inqtrix.agents.patterns._structured import structured_call
        from inqtrix.agents.phase_models import QuickWebQuery
        from inqtrix.core.results import AgentResult

        try:
            deps.require_tool_allowed("web_instant")
        except PermissionError as exc:
            raise RuntimeError(str(exc)) from exc
        if deps.capability_registry is None:
            raise RuntimeError(
                "Schnell-Web angefordert, aber web.search.instant ist "
                "nicht registriert."
            )

        deps.emit(
            PHASE_CHANGED_EVENT,
            {
                "phase": "execution",
                "previous_phase": "",
                "snapshot": {
                    "current_node": _KERNEL_NODE,
                    "phase": "execution",
                    "execution": self._execution_payload(
                        request,
                        deps,
                        consent_reason=_kernel_consent_reason(deps),
                    ),
                },
            },
        )

        query = ""
        recency = ""
        consent_reason = "explicit_directive"
        if deps.autonomy == "strict":
            approval_id = _quick_web_approval_id(deps.run_id)
            try:
                approval = run_coro(
                    self._control.get_approval(deps.run_id, approval_id)
                )
            except ApprovalNotFound:
                query, recency = self._derive_quick_web_query(
                    request, deps, structured_call, QuickWebQuery
                )
                # The action args must match the web_instant tool schema
                # (query only) so an edit re-validates cleanly; recency is a
                # quick-web search refinement, not a tool arg, so it rides in
                # the approval payload where an args-only edit can't strip it.
                actions = [
                    {
                        "tool": "web_instant",
                        "args": {"query": query},
                        "summary": "Eine direkte Websuche ausfuehren.",
                    }
                ]
                approval = run_coro(
                    self._control.create_approval(
                        ApprovalRecord(
                            approval_id=approval_id,
                            run_id=deps.run_id,
                            kind="tool",
                            payload={"actions": actions, "recency": recency},
                            interrupt_key="quick_web",
                        )
                    )
                )
                deps.emit(
                    "inqtrix.agent.approval.requested",
                    {
                        "approval_id": approval.approval_id,
                        "kind": "tool",
                        "actions": actions,
                    },
                )
            if approval.status == "pending":
                if context.park is None:
                    raise RuntimeError(
                        "Strict quick_web braucht eine park-faehige Run-"
                        "Ausfuehrung."
                    )
                context.park("waiting_for_approval")
                return AgentResult(
                    answer="",
                    result_type="agent_kernel_parked",
                    raw={
                        "answer": "",
                        "usage": dict(deps.usage),
                        "result_state": {
                            "parked": True,
                            "cancelled": False,
                            "execution": self._execution_payload(
                                request,
                                deps,
                                consent_reason="strict_approval_required",
                            ),
                        },
                    },
                )
            if approval.decision == "reject":
                answer = "Die direkte Websuche wurde nicht freigegeben."
                return self._quick_web_result(
                    request=request,
                    answer=answer,
                    deps=deps,
                    references=[],
                    query="",
                    consent_reason="strict_rejected",
                )
            action_set = (
                approval.decision_payload.get("actions")
                if approval.decision == "edit"
                else approval.payload.get("actions")
            )
            query, action_recency = _validated_quick_web_action(action_set)
            # recency lives in the approval payload (not the tool args), so an
            # edited query keeps the originally-derived recency; the action-arg
            # value is only a fallback for any legacy pre-fix approval row.
            recency = str(approval.payload.get("recency") or action_recency or "")
            consent_reason = "strict_approval"
        else:
            query, recency = self._derive_quick_web_query(
                request, deps, structured_call, QuickWebQuery
            )

        deps.check_abort()
        deps.emit(
            TOOL_STARTED_EVENT,
            {
                "tool": "web_instant",
                "tool_call_id": "quick_web",
                "args_preview": query[:_ARGS_PREVIEW_LIMIT],
            },
        )
        payload: dict[str, Any] = {"query": query, "max_sources": 8}
        if recency:
            payload["recency"] = recency
        output = run_coro(
            deps.capability_registry.invoke(
                "web.search.instant", payload, deps.capability_context
            )
        )
        deps.book_usage(output.prompt_tokens, output.completion_tokens)
        deps.record_source_tool_use("web")
        all_references = deps.register_instant_web_search(output)
        deps.emit(
            TOOL_FINISHED_EVENT,
            {
                "tool": "web_instant",
                "tool_call_id": "quick_web",
                "status": "success",
                "result_preview": normalize_agent_markdown(
                    str(output.answer or "")
                )[:_ARGS_PREVIEW_LIMIT],
            },
        )

        references = all_references
        answer = self._synthesize_quick_web_answer(
            request=request,
            deps=deps,
            query=query,
            provider_answer=str(output.answer or ""),
            references=references,
        )
        return self._quick_web_result(
            request=request,
            answer=answer,
            deps=deps,
            references=references,
            query=query,
            consent_reason=consent_reason,
        )

    def _derive_quick_web_query(
        self,
        request: "RunRequest",
        deps: KernelDeps,
        structured_call: Any,
        model_cls: Any,
    ) -> tuple[str, str]:
        """Derive one search query; visibly fall back to the question."""
        deps.check_abort()
        context_block = deps.session_history[-4000:]
        prompt = (
            "Formuliere genau EINE eigenstaendige Web-Suchanfrage fuer die "
            "aktuelle Nutzerfrage. Nutze den Verlauf nur zur Aufloesung von "
            "Bezugnahmen. Setze recency auf day/week/month/year nur bei "
            "einem ausdruecklich aktuellen Zeitbezug, sonst auf ''.\n\n"
            f"AKTUELLE FRAGE:\n{deps.question}\n\n"
            f"RELEVANTER VERLAUF:\n{context_block or '(kein Verlauf)'}"
        )
        outcome = structured_call(
            deps.llm,
            prompt=prompt,
            model_cls=model_cls,
            node="agent_quick_web_query",
            model=deps.model,
            reasoning_effort=deps.reasoning_effort,
            timeout=deps.timeout,
        )
        deps.book_usage(
            outcome.usage.get("prompt_tokens", 0),
            outcome.usage.get("completion_tokens", 0),
        )
        value = outcome.value
        if (
            value is None
            or not value.query.strip()
            or value.recency not in (
                "",
                "day",
                "week",
                "month",
                "year",
            )
        ):
            log.warning(
                "Quick-Web-Query konnte nicht valide abgeleitet werden "
                "(Marker %s) — sichtbarer Fallback auf die Nutzerfrage.",
                outcome.marker,
            )
            deps.emit(
                "inqtrix.agent.quick_web.fallback",
                {
                    "stage": "query",
                    "marker": outcome.marker,
                    "fallback": "original_question",
                },
            )
            return deps.question.strip(), ""
        return value.query.strip(), value.recency

    def _synthesize_quick_web_answer(
        self,
        *,
        request: "RunRequest",
        deps: KernelDeps,
        query: str,
        provider_answer: str,
        references: list[dict[str, Any]],
    ) -> str:
        """Ground the chat answer exclusively in the instant-search output."""
        deps.check_abort()
        source_parts: list[str] = []
        for item in references:
            if item.get("excerpt"):
                support_label = "Quellenauszug"
                support = item["excerpt"]
            elif item.get("provider_snippet"):
                support_label = "Vom Websuchdienst geliefertes Snippet"
                support = item["provider_snippet"]
            elif item.get("grounded_support"):
                support_label = (
                    "Dieser URL im Azure-Suchergebnis zugeordneter "
                    "Antwortkontext"
                )
                support = item["grounded_support"]
            else:
                support_label = "Zuordnungsstatus"
                support = (
                    "Azure lieferte die URL, aber keinen eindeutig dieser "
                    "Quelle zuordenbaren Einzelabschnitt."
                )
            source_parts.append(
                f"[{item['label']}] {item.get('title') or item['url']}\n"
                f"URL: {item['url']}\n{support_label}: {support}"
            )
        source_block = "\n".join(source_parts) or "(keine Quellen geliefert)"
        with observe_provider_retries(
            deps.llm,
            lambda notice: _emit_kernel_model_retry(
                deps, "agent_quick_web_answer", notice
            ),
        ):
            from inqtrix.agents.prompts import (
                _KERNEL_SECURITY,
                untrusted_fence,
            )

            # Built before the prompt: a backslash inside a nested f-string
            # expression is a syntax error on the minimum supported Python.
            fenced_web_content = untrusted_fence(
                f"{provider_answer}\n\n{source_block}", "web"
            )

            response = deps.llm.complete_with_metadata(
                (
                    "Beantworte die Nutzerfrage knapp und direkt in ihrer "
                    "Sprache. Verwende ausschliesslich das abgegrenzte "
                    "Azure-Websuchergebnis und die von Azure gelieferten "
                    "Quellen. Die Provider-Antwort ist das geerdete Ergebnis "
                    "dieser Suche und darf einschließlich darin genannter "
                    "Zahlen, Preise und Daten verwendet werden. Erfinde "
                    "nichts hinzu. Verlinke Aussagen nur mit URLs, die im "
                    "Suchergebnis vorkommen. Wenn Azure mehrere Links einem "
                    "gemeinsamen Antwortabschnitt zuordnet, behaupte keine "
                    "exklusive Eins-zu-eins-Herkunft. Benenne echte Lücken "
                    "oder Widersprüche offen, aber entferne vorhandene "
                    "Providerinformationen nicht allein wegen einer "
                    "Quellenklassifikation. Leite aus fehlenden Treffern "
                    "niemals Abwesenheit ab.\n\n"
                    f"NUTZERFRAGE:\n{deps.question}\n\n"
                    f"RELEVANTER VERLAUF:\n"
                    f"{deps.session_history[-4000:] or '(kein Verlauf)'}\n\n"
                    f"SUCHANFRAGE:\n{query}\n\n"
                    # Same fence + SICHERHEIT anchor as the graph lane: the
                    # quick lane synthesizes DIRECTLY from external content
                    # and must not be the one unfenced injection surface.
                    f"{fenced_web_content}"
                ),
                system=(
                    "Du formulierst eine direkt belegte Schnell-Web-Antwort. "
                    "Fuehre kein eigenes Wissen als Fakt ein.\n\n"
                    f"{_KERNEL_SECURITY}"
                ),
                model=deps.model,
                reasoning_effort=deps.reasoning_effort,
                timeout=deps.timeout,
            )
        deps.book_usage(response.prompt_tokens, response.completion_tokens)
        answer = str(response.content or "").strip()
        if not answer:
            log.warning(
                "Quick-Web-Antwortsynthese lieferte keinen Text — "
                "sichtbarer Fallback auf die geerdete Provider-Antwort."
            )
            deps.emit(
                "inqtrix.agent.quick_web.fallback",
                {
                    "stage": "answer",
                    "fallback": "provider_answer",
                },
            )
            answer = (
                "Die eigenständige Aufbereitung ist fehlgeschlagen. "
                "Hier ist das von Azure gelieferte Websuchergebnis:\n\n"
                + provider_answer.strip()
            )
        source_lines = "\n".join(
            f"- [{item.get('title') or item['url']}]({item['url']})"
            for item in references
        )
        if source_lines:
            answer = f"{answer}\n\n### Quellen\n{source_lines}"
        return normalize_agent_markdown(answer)

    def _quick_web_result(
        self,
        *,
        request: "RunRequest",
        answer: str,
        deps: KernelDeps,
        references: list[dict[str, Any]],
        query: str,
        consent_reason: str,
    ) -> "AgentResult":
        """Build the normal kernel result for the isolated quick lane."""
        from inqtrix.core.results import AgentResult

        execution = self._execution_payload(
            request, deps, consent_reason=consent_reason
        )

        answer = _validate_kernel_answer_citations(deps, answer, references)
        answer_claim_bindings: list[dict[str, Any]] = []

        # The native RunService publisher materializes this answer after the
        # algorithm returns, including strict-reject receipts.
        if consent_reason != "strict_rejected":
            # A reject receipt is no memory substrate — staging would spend
            # an LLM call on "Die direkte Websuche wurde nicht freigegeben."
            _stage_kernel_memory(deps, answer)
        deps.emit(
            PHASE_CHANGED_EVENT,
            {
                "phase": "done",
                "previous_phase": "execution",
                "snapshot": {
                    "current_node": _KERNEL_NODE,
                    "phase": "done",
                    "execution": execution,
                },
            },
        )
        # Same cited-only/basis contract as the normal lane.  The central
        # publisher consumes this exported list without reinterpreting it.
        report_references = _result_references(answer, references)
        return AgentResult(
            answer=answer,
            result_type="agent_kernel_result",
            raw={
                "answer": answer,
                "usage": dict(deps.usage),
                "result_state": {
                    "answer": answer,
                    "cancelled": False,
                    "report_references": report_references,
                    "answer_claim_bindings": answer_claim_bindings,
                    "all_citations": [
                        str(item.get("url") or "")
                        for item in references
                        if item.get("url")
                    ],
                    "allowed_citations": [
                        str(item.get("url") or "")
                        for item in references
                        if item.get("url")
                    ],
                    "queries": [query] if query else [],
                    "quick_web_query": query,
                    "execution": execution,
                },
            },
        )

    def _current_run_canvases(self, deps: KernelDeps) -> list[Any]:
        """Load every canvas created or updated by the current run."""
        from inqtrix.pagination import decode_cursor

        canvases: list[Any] = []
        after: tuple[float, str] | None = None
        while True:
            rows, cursor = run_coro(
                self._control.list_artifacts(
                    deps.run_id,
                    kind="deliverable",
                    limit=50,
                    after=after,
                )
            )
            for row in rows:
                full, _revisions = run_coro(
                    self._control.get_artifact(
                        deps.run_id, row.artifact_id
                    )
                )
                canvases.append(full)
            if not cursor:
                break
            after = decode_cursor(cursor)
        return canvases

    def _deep_verify(
        self,
        deps: KernelDeps,
        request: "RunRequest",
        answer: str,
        canvases: list[Any],
        *,
        messages: list[Any],
    ) -> str:
        """Review the full assignment and all outputs, then revise once."""
        from inqtrix.agents.control_ports import (
            ArtifactBatchRevision,
            ArtifactNotFound,
            ArtifactRevisionConflict,
        )
        from inqtrix.agents.patterns._structured import structured_call
        from inqtrix.agents.phase_models import (
            DeepRevisionBundle,
            DeepReviewVerdict,
        )
        from inqtrix.agents.prompts import (
            build_deep_review_prompt,
            build_deep_revision_prompt,
        )

        def _review_narration(text: str, *, final: bool) -> None:
            deps.emit(
                NARRATION_EVENT,
                {
                    "narration_id": "kernel_deep_review",
                    "kind": "synthesis",
                    "phase": "execution",
                    "text": f"Verifikations-Durchlauf: {text}",
                    "final": final,
                },
            )

        from inqtrix.agents.kernel.middleware import (
            SKILL_INPUTS_RESOLVED_FLAG,
            SKILL_INPUTS_RESOLVED_MARKER,
        )

        resolved_skill_blocks: list[str] = []
        for message in messages:
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                continue
            message_type = type(message).__name__
            # F3 hardening: trust is anchored on SERVER-SET metadata —
            # the middleware's additional_kwargs flag or the load_skill
            # tool identity — never on user-forgeable content text. A
            # user question containing the literal marker is inert here.
            trusted_resolved_block = (
                message_type == "HumanMessage"
                and bool(
                    (getattr(message, "additional_kwargs", None) or {}).get(
                        SKILL_INPUTS_RESOLVED_FLAG
                    )
                )
            ) or (
                message_type == "ToolMessage"
                and getattr(message, "name", "") == "load_skill"
                and SKILL_INPUTS_RESOLVED_MARKER in content
            )
            if trusted_resolved_block:
                resolved_skill_blocks.append(content)
        from inqtrix.agents.source_policy import effective_source_policy

        effective_policy = effective_source_policy(
            deps.source_policy, deps.execution_directive
        )
        policy_payload = {
            "source_policy": {
                "web": effective_policy.web,
                "knowledge": effective_policy.knowledge,
            },
            "execution_directive": deps.execution_directive,
        }
        effective_assignment = (
            f"{self._user_message(request, deps)}\n\n"
            "EFFECTIVE SERVER POLICY:\n"
            f"{json.dumps(policy_payload, ensure_ascii=False, sort_keys=True)}"
        )
        if resolved_skill_blocks:
            effective_assignment += (
                "\n\nRESOLVED SKILL INPUTS AND INSTRUCTIONS:\n"
                + "\n\n".join(resolved_skill_blocks)
            )
        output_payload = {
            "chat": answer,
            "artifacts": [
                {
                    "artifact_id": item.artifact_id,
                    "revision": item.revision,
                    "content_markdown": item.content_markdown,
                    "refs": [dict(ref) for ref in item.refs],
                }
                for item in canvases
            ],
        }
        output_bundle = json.dumps(
            output_payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        _review_narration("Pruefung laeuft.", final=False)
        deps.check_abort()
        provider_models = getattr(deps.llm, "models", None)
        if provider_models is None:
            desc = describe_unresolved_resolution("agent_deep_review", "")
        else:
            # Tier map only — the request's model/tier override targets
            # the BRAIN, not the cheap rubric check (R2/R3 pattern).
            desc = describe_resolution(
                "agent_deep_review",
                provider_models,
                "",
                requested_model="",
                requested_effort="",
            )
        deps.emit("inqtrix.node.model_resolution", desc)
        try:
            outcome = structured_call(
                deps.llm,
                prompt=build_deep_review_prompt(
                    effective_assignment,
                    output_bundle,
                ),
                model_cls=DeepReviewVerdict,
                node="agent_deep_review",
                model=desc.get("model") or None,
                reasoning_effort=desc.get("effort") or None,
                timeout=deps.timeout,
            )
        except Exception as exc:  # noqa: BLE001 — a finished answer exists
            # A hard provider failure (timeout/5xx) must not fail a run
            # that already HAS its answer — the pass is an upgrade, not
            # a dependency.
            log.warning(
                "Deep-Review-Aufruf fehlgeschlagen (error_type=%s) — die Antwort "
                "bleibt unveraendert.",
                type(exc).__name__,
            )
            _review_narration(
                "Providerfehler — bestehende Outputs bleiben unveraendert.",
                final=True,
            )
            return answer
        deps.book_usage(
            outcome.usage.get("prompt_tokens", 0),
            outcome.usage.get("completion_tokens", 0),
        )
        verdict = outcome.value
        if verdict is None:
            log.warning(
                "Deep-Review ohne valides Ergebnis — die Antwort bleibt "
                "unveraendert."
            )
            _review_narration(
                "Parsefehler — bestehende Outputs bleiben unveraendert.",
                final=True,
            )
            return answer
        findings = list(verdict.findings)
        if not all(
            (
                verdict.complete,
                verdict.grounded,
                verdict.contradictions_named,
            )
        ) and not findings:
            log.warning(
                "Deep-Review meldete einen negativen Befund ohne "
                "reparierbare Findings."
            )
            _review_narration(
                "inkonsistentes Review — bestehende Outputs bleiben "
                "unveraendert.",
                final=True,
            )
            return answer
        known = {item.artifact_id: item for item in canvases}
        if any(
            (finding.target == "chat" and finding.artifact_id)
            or (
                finding.target == "artifact"
                and finding.artifact_id not in known
            )
            or not finding.finding.strip()
            for finding in findings
        ):
            log.warning("Deep-Review nannte ein unbekanntes/ungueltiges Ziel.")
            _review_narration(
                "ungueltiges Befundziel — bestehende Outputs bleiben unveraendert.",
                final=True,
            )
            return answer

        if not findings:
            _review_narration("keine Befunde, Antwort besteht.", final=True)
            return answer
        _review_narration(
            f"{len(findings)} Befund(e) — Ueberarbeitung laeuft.",
            final=False,
        )
        deps.check_abort()
        try:
            revision = structured_call(
                deps.llm,
                prompt=build_deep_revision_prompt(
                    effective_assignment,
                    output_bundle,
                    [item.model_dump() for item in findings],
                ),
                model_cls=DeepRevisionBundle,
                node="agent_kernel",
                model=deps.model,
                reasoning_effort=deps.reasoning_effort,
                timeout=deps.timeout,
            )
        except Exception as exc:  # noqa: BLE001 — keep the finished answer
            log.warning(
                "Deep-Revision fehlgeschlagen (error_type=%s) — die urspruengliche "
                "Antwort bleibt bestehen.",
                type(exc).__name__,
            )
            _review_narration(
                f"{len(findings)} Befund(e), Ueberarbeitung "
                "fehlgeschlagen — die urspruengliche Antwort bleibt.",
                final=True,
            )
            return answer
        deps.book_usage(
            revision.usage.get("prompt_tokens", 0),
            revision.usage.get("completion_tokens", 0),
        )
        bundle = revision.value
        revised = (
            normalize_agent_markdown(bundle.chat_markdown.strip())
            if isinstance(bundle, DeepRevisionBundle)
            else ""
        )
        if not revised:
            log.warning(
                "Deep-Revision ohne Inhalt — die urspruengliche Antwort "
                "bleibt bestehen."
            )
            _review_narration(
                f"{len(findings)} Befund(e), Ueberarbeitung ohne "
                "Ergebnis — die urspruengliche Antwort bleibt.",
                final=True,
            )
            return answer
        chat_targeted = any(item.target == "chat" for item in findings)
        if not chat_targeted and revised != answer:
            log.warning(
                "Deep-Revision veraenderte Chat ohne Chat-Befund."
            )
            _review_narration(
                "ungueltige Chat-Revision — bestehende Outputs bleiben "
                "unveraendert.",
                final=True,
            )
            return answer
        if chat_targeted and revised == answer:
            log.warning("Deep-Revision liess einen Chat-Befund unveraendert.")
            _review_narration(
                "Ueberarbeitung ohne Ergebnis — bestehende Outputs bleiben "
                "unveraendert.",
                final=True,
            )
            return answer
        artifact_updates = list(bundle.artifacts)
        normalized_artifact_markdown = {
            item.artifact_id: normalize_agent_markdown(item.markdown.strip())
            for item in artifact_updates
        }
        targeted_ids = {
            item.artifact_id
            for item in findings
            if item.target == "artifact"
        }
        ids = [item.artifact_id for item in artifact_updates]
        invalid_revision = (
            len(ids) != len(set(ids))
            or any(
                item.artifact_id not in targeted_ids
                for item in artifact_updates
            )
            or any(item.artifact_id not in known for item in artifact_updates)
            or any(
                known[item.artifact_id].revision != item.expected_revision
                for item in artifact_updates
                if item.artifact_id in known
            )
            or any(
                not normalized_artifact_markdown[item.artifact_id]
                for item in artifact_updates
            )
            or any(
                normalized_artifact_markdown[item.artifact_id]
                == known[item.artifact_id].content_markdown
                for item in artifact_updates
                if item.artifact_id in known
            )
        )
        if invalid_revision:
            log.warning("Deep-Revision-Bundle verletzt den Output-Vertrag.")
            _review_narration(
                "ungueltige Revision — bestehende Outputs bleiben unveraendert.",
                final=True,
            )
            return answer
        if targeted_ids != set(ids):
            log.warning("Deep-Revision liess Canvas-Befunde unbearbeitet.")
            _review_narration(
                "unvollstaendige Revision — bestehende Outputs bleiben unveraendert.",
                final=True,
            )
            return answer
        try:
            run_coro(
                self._control.revise_session_artifacts_atomically(
                    run_id=deps.run_id,
                    session_id=deps.session_id or None,
                    revisions=[
                        ArtifactBatchRevision(
                            artifact_id=item.artifact_id,
                            expected_revision=item.expected_revision,
                            content_markdown=normalized_artifact_markdown[
                                item.artifact_id
                            ],
                        )
                        for item in artifact_updates
                    ],
                )
            )
        except (ArtifactNotFound, ArtifactRevisionConflict) as exc:
            log.warning(
                "Deep-Revision CAS-Konflikt (error_type=%s).",
                type(exc).__name__,
            )
            _review_narration(
                "CAS-Konflikt — bestehende Outputs bleiben unveraendert.",
                final=True,
            )
            return answer
        except Exception as exc:  # noqa: BLE001 — preserve reviewed outputs
            log.warning(
                "Deep-Revision Store-Fehler (error_type=%s).",
                type(exc).__name__,
            )
            _review_narration(
                "Storefehler — bestehende Outputs bleiben unveraendert.",
                final=True,
            )
            return answer
        _review_narration(
            f"{len(findings)} Befund(e) behoben — Outputs ueberarbeitet.",
            final=True,
        )
        return revised

    # -- segment assembly ---------------------------------------------------#

    def _build_deps(
        self,
        run_id: str,
        llm: Any,
        context: "RunContext",
        request: "RunRequest",
        *,
        pin_tier: str = "",
        pin_effort: str = "",
        depth: str = "normal",
    ) -> KernelDeps:
        settings = context.agent_settings
        provider_models = getattr(llm, "models", None)
        requested_tier = (getattr(settings, "model_tier", "") or "").strip()
        requested_model = (getattr(settings, "model", "") or "").strip()
        requested_effort = (getattr(settings, "effort", "") or "").strip()
        if (pin_tier and not requested_tier) or (
            pin_effort and not requested_effort
        ):
            # R4 precedence: explicit user override > skill pin > depth
            # > tier map. The pin only fills what the request left empty.
            log.info(
                "Skill-Pin aktiv: model_tier=%r effort=%r",
                pin_tier or "(kein)",
                pin_effort or "(kein)",
            )
        requested_tier = requested_tier or pin_tier
        requested_effort = requested_effort or pin_effort
        if depth == "deep" and not requested_effort:
            # Deep mode: the kernel node runs on high
            # reasoning effort unless the request or a skill pin chose
            # one — the tier stays with the tier map (agent_kernel is
            # high-tier already).
            requested_effort = "high"
            log.info("Deep-Modus: reasoning_effort=high fuer den Kernel.")
        if provider_models is None and not requested_model:
            desc = describe_unresolved_resolution(
                _KERNEL_NODE, requested_tier
            )
        else:
            desc = describe_resolution(
                _KERNEL_NODE,
                provider_models,
                requested_tier,
                requested_model=requested_model,
                requested_effort=requested_effort,
            )
        deps = KernelDeps(
            run_id=run_id,
            control=self._control,
            platform=self._platform,
            llm=llm,
            model=desc.get("model") or None,
            reasoning_effort=desc.get("effort") or None,
            timeout=float(getattr(settings, "reasoning_timeout", REASONING_TIMEOUT)),
            event_sink=context.event_sink,
            capability_registry=self._capabilities,
            capability_context=self._capability_context(context, request),
            cancel_token=context.cancel_token,
            token_budget=int(context.token_budget or 0),
        )
        # Context management is resolved per run (the compiled graph is
        # shared): the compaction trigger follows the RESOLVED model's
        # card, the offload threshold is a platform constant.
        deps.context_trigger_tokens = resolve_context_trigger_tokens(
            self._platform, deps.model
        )
        deps.context_offload_chars = int(
            getattr(self._platform, "kernel_context_offload_chars", 0) or 0
        )
        deps.emit("inqtrix.node.model_resolution", desc)
        return deps

    def _load_request_skills(
        self, request: "RunRequest", context: "RunContext"
    ) -> list[Any]:
        """Load the router-admitted attached skills (loud, never partial).

        Same trust model as the phase machine: the router admitted the
        ids with the caller's grants, the segment executes what was
        admitted (``get_admitted``). Loading is separate from
        activation so the skill model pins (R4) can inform the deps
        BEFORE the model resolution.
        """
        if not request.skill_ids:
            return []
        if self._skill_service is None:
            raise RuntimeError(
                "Skills sind angehaengt, aber auf diesem Server nicht "
                "eingerichtet."
            )
        records: list[Any] = []
        for skill_id in request.skill_ids:
            try:
                visible_to = getattr(
                    self._capability_context(context, request), "visible_to", None
                )
                record, _access = run_coro(
                    self._skill_service.get_visible(
                        skill_id,
                        tenant_id=(
                            context.principal.tenant_id
                            if context.principal is not None
                            else "default"
                        ),
                        visible_to=visible_to,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — loud, never partial
                raise RuntimeError(
                    f"Angehaengter Skill {skill_id} ist nicht mehr "
                    "verfuegbar — Lauf abgebrochen."
                ) from exc
            expected_revision = request.skill_revisions.get(skill_id)
            if (
                expected_revision is None
                or record.revision != expected_revision
            ):
                raise RuntimeError(
                    f"Angehaengter Skill {skill_id} hat sich seit der "
                    "Freigabe geaendert — Lauf abgebrochen."
                )
            records.append(record)
        return records

    def _reactivate_loaded_skills(
        self, deps: KernelDeps, snapshot: Any
    ) -> None:
        """Re-arm restrictions from load_skill markers in the transcript.

        A skill loaded BEFORE a park narrowed the tool surface; the
        resume segment must not forget that (a forgotten allowlist
        would silently widen what the model may call).
        """
        from inqtrix.agents.kernel.tools import SKILL_LOADED_MARKER

        if snapshot is None or not snapshot.values:
            return
        prefix = SKILL_LOADED_MARKER.split("{", 1)[0]
        for message in snapshot.values.get("messages") or []:
            if getattr(message, "type", "") != "tool":
                continue
            # Trust the PRODUCING TOOL, not the content prefix: other
            # tools relay provider-controlled text verbatim (web
            # answers), so a marker-shaped result from them must never
            # activate a skill.
            if getattr(message, "name", "") != "load_skill":
                continue
            content = str(getattr(message, "content", "") or "")
            if not content.startswith(prefix):
                continue
            marker_body = content[len(prefix):].split("]", 1)[0]
            skill_id, separator, admitted_revision = marker_body.rpartition("@")
            if not skill_id or not separator or not admitted_revision:
                log.warning("Geladener Skill ohne Revisionsmarker im Checkpoint.")
                raise RuntimeError(
                    "Geladener Skill besitzt keinen sicheren "
                    "Revisionsmarker — Resume wird abgebrochen."
                )
            if self._skill_service is None:
                log.warning(
                    "Geladener Skill %s kann ohne Skill-Service nicht "
                    "reaktiviert werden.",
                    skill_id,
                )
                raise RuntimeError(
                    f"Geladener Skill {skill_id} kann ohne Skill-Service "
                    "nicht sicher reaktiviert werden."
                )
            try:
                record, _access = run_coro(
                    self._skill_service.get_visible(
                        skill_id,
                        tenant_id=(
                            deps.principal.tenant_id
                            if deps.principal is not None
                            else "default"
                        ),
                        visible_to=getattr(
                            deps.capability_context, "visible_to", None
                        ),
                    )
                )
            except Exception:  # noqa: BLE001 — visible via the warning
                log.warning(
                    "Geladener Skill %s ist beim Resume nicht mehr "
                    "verfuegbar — Resume wird fail-closed abgebrochen.",
                    skill_id,
                )
                raise RuntimeError(
                    f"Geladener Skill {skill_id} kann beim Resume nicht "
                    "sicher reaktiviert werden."
                )
            if str(record.revision) != admitted_revision:
                log.warning(
                    "Geladener Skill %s wechselte Revision (%s -> %s).",
                    skill_id,
                    admitted_revision,
                    record.revision,
                )
                raise RuntimeError(
                    f"Geladener Skill {skill_id} hat sich seit der "
                    "Aktivierung geaendert — Resume wird abgebrochen."
                )
            deps.activate_skill(record)

    def _skills_context(self, deps: KernelDeps) -> str:
        """Attached-skill instructions + the model-facing disclosure list.

        The disclosure block (plan `3.3`) names ONLY the caller-visible
        ``model_allowed`` autocomplete skills under a deterministic
        character budget; overflow is a VISIBLE "... und N weitere"
        line, never a silent cut. Shared-in skills never appear (their
        grants are not resolvable here, and recipients are structurally
        ``user_only`` anyway).
        """
        parts: list[str] = []
        if deps.skill_service is not None and deps.principal is not None:
            try:
                visible = run_coro(
                    deps.skill_service.list_visible(
                        tenant_id=deps.principal.tenant_id,
                        visible_to=getattr(
                            deps.capability_context, "visible_to", None
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001 — a lookup outage costs the list
                log.warning(
                    "Skill-Disclosure nicht verfuegbar "
                    "(error_type=%s).",
                    type(exc).__name__,
                )
                visible = []
            activated = {skill.id for skill in deps.skills}
            candidates = [
                record
                for record, _shared in visible
                if record.invocation == "model_allowed"
                and record.include_in_autocomplete
                and record.id not in activated
            ]
            budget = self._platform.skills_disclosure_budget_chars
            lines: list[str] = []
            used = 0
            dropped = 0
            for record in candidates:
                line = (
                    f"- /{record.label} ({record.id}): "
                    f"{record.description or record.title}"
                    + (
                        f" — Wann: {record.when_to_use}"
                        if record.when_to_use
                        else ""
                    )
                )
                if used + len(line) > budget:
                    dropped += 1
                    continue
                lines.append(line)
                used += len(line)
            if dropped:
                lines.append(f"- ... und {dropped} weitere")
            if lines:
                parts.append(
                    "Verfuegbare Skills (bei Bedarf mit load_skill "
                    "aktivieren):\n" + "\n".join(lines)
                )
        return "\n\n".join(parts)

    def _capability_context(
        self, context: "RunContext", request: "RunRequest"
    ) -> Any:
        """Per-segment tool identity and server-pinned knowledge boundary.

        Resolved once per segment like the workspace `_RunDeps`;
        ``visible_to=None`` stays the historical see-everything view of
        the anonymous/static modes only.
        """
        from inqtrix.capabilities import CapabilityContext

        visible_to = None
        if self._permissions is not None and context.principal is not None:
            visible_to = run_coro(
                self._permissions.resolve_user_context(context.principal)
            )
        knowledge_collection_ids = pinned_knowledge_collection_ids(
            request.knowledge_filters,
            scoped_principal=bool(
                context.principal is not None
                and context.principal.user_id is not None
            ),
        )
        return CapabilityContext(
            principal=context.principal,
            visible_to=visible_to,
            workspace_id=context.workspace_id,
            run_id=context.run_id,
            question=request.question,
            knowledge_collection_ids=knowledge_collection_ids,
            search_provider=context.providers.search,
            authority_check=context.authority_check,
            on_provider_retry=(
                lambda notice: context.event_sink(
                    "inqtrix.agent.activity",
                    provider_retry_activity(notice),
                )
                if context.event_sink is not None
                else None
            ),
        )

    def _graph_input(
        self,
        request: "RunRequest",
        run_id: str,
        existing: Any,
        deps: KernelDeps,
    ) -> Any:
        if existing is None or not existing.values:
            if self._run_service is None:
                raise RuntimeError(
                    "Checkpoint-Status kann ohne Run-Store nicht sicher "
                    "geprueft werden; der Lauf wird nicht neu gestartet."
                )
            ensure_checkpoint_restart_safe(
                run_id,
                control=self._control,
                run_store=self._run_service.run_store,
                run_async=run_coro,
            )
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": self._user_message(request, deps),
                    }
                ]
            }
        if existing.tasks and any(
            task.interrupts for task in existing.tasks
        ):
            from langgraph.types import Command

            return Command(resume=self._resume_value(run_id, existing))
        # Crash between segments without a pending interrupt: the
        # checkpoint fast-forwards past completed nodes on its own.
        return None

    def _user_message(
        self, request: "RunRequest", deps: KernelDeps
    ) -> str:
        """The per-run user message: K1 session context + assignment.

        Context rides the user message because the compiled graph (and
        its system prompt) is shared across runs. An explicit
        ``request.history`` (API callers) wins over the builder, same
        precedence as the phase machine.
        """
        from inqtrix.agents.prompts import build_kernel_user_message

        self._hydrate_session_context(request, deps)
        history_block = deps.session_history
        registry = deps.artifact_registry
        last_form = deps.last_response_form
        prior_evidence_count = deps.prior_evidence_count
        from inqtrix.agents.skills_runtime import build_tool_directives_line

        message = build_kernel_user_message(
            deps.question,
            history_block=history_block,
            artifact_registry=registry,
            last_response_form=last_form,
            prior_evidence_count=prior_evidence_count,
            memory_briefing=deps.memory_briefing,
            response_form=request.response_form or "",
            autonomy=deps.autonomy,
            depth=deps.depth,
            tier=deps.tier,
            skills_block=self._skills_context(deps),
            tool_directives_line=build_tool_directives_line(
                request.tool_directives or ()
            ),
        )
        if request.execution_directive == "knowledge_only":
            message += (
                "\n\nVERBINDLICHE AUSFUEHRUNGSROUTE: Nutze fuer diese "
                "Antwort ausschliesslich das Projektwissen. Suche mit "
                "search_project_knowledge und lies bei Bedarf gefundene "
                "Dokumente. Nutze keine Web-, Missions-, Canvas- oder "
                "Editor-Werkzeuge. Wenn das Projektwissen nicht reicht, "
                "benenne die Luecke statt eigenes Weltwissen als Fakt "
                "einzufuehren."
            )
        return message

    def _hydrate_session_context(
        self, request: "RunRequest", deps: KernelDeps
    ) -> None:
        """Load K1-K4 once for every kernel route, including quick web."""
        from inqtrix.agents.session_context import build_session_context

        history_block = request.history or ""
        registry: tuple[dict[str, Any], ...] = deps.artifact_registry
        last_form = deps.last_response_form
        prior_evidence_count = deps.prior_evidence_count
        if deps.session_id and self._run_service is not None:
            pack = build_session_context(
                deps.session_id,
                run_store=self._run_service.run_store,
                control=self._control,
                run_async=run_coro,
                visible_to=getattr(
                    deps.capability_context, "visible_to", None
                ),
                current_run_id=deps.run_id,
            )
            if not history_block:
                history_block = pack.history_block
            registry = pack.artifact_registry
            last_form = pack.last_response_form
            prior_evidence_count = pack.prior_evidence_count
        deps.session_history = history_block
        deps.artifact_registry = registry
        deps.last_response_form = last_form
        deps.prior_evidence_count = prior_evidence_count
        self._recall_memory_briefing(request, deps)

    def _recall_memory_briefing(
        self, request: "RunRequest", deps: KernelDeps
    ) -> None:
        """Load the non-evidentiary K5 memory briefing (mission parity).

        Same gates as ``_load_memory_briefing`` in the mission engine:
        service wired, principal present, per-user opt-in (privacy
        default OFF), provider available and eligible. Every outcome is
        visible — ``deps.memory_status`` mirrors the mission vocabulary
        and a recall failure narrates instead of silently disabling.
        """
        if deps.memory_recalled:
            # Once per segment: _hydrate_session_context runs again for
            # the deep-verify assignment rebuild — a second recall would
            # double the briefing round-trip for every deep run.
            return
        deps.memory_recalled = True
        service = deps.memory
        principal = deps.principal
        if service is None or principal is None or not deps.memory_opt_in:
            deps.memory_status = "disabled"
            return
        status = service.status(principal)
        if not status.get("available") or not status.get(
            "principal_eligible"
        ):
            deps.memory_status = "disabled"
            return
        from inqtrix.agents.memory_ports import AgentMemoryUnavailable

        try:
            briefing, memory_status = run_coro(
                service.recall_briefing(
                    principal=principal,
                    query=deps.question or "",
                    limit=5,
                )
            )
        except AgentMemoryUnavailable:
            briefing, memory_status = "", "unavailable"
        deps.memory_briefing = briefing
        deps.memory_status = memory_status
        if memory_status == "used":
            log.info("Kernel-Memory-Briefing geladen (K5, nicht zitierbar).")
        elif memory_status == "unavailable":
            log.warning(
                "Kernel-Memory nicht verfuegbar — Lauf ohne "
                "Langzeit-Kontext."
            )
            deps.emit(
                NARRATION_EVENT,
                {
                    "narration_id": "kernel_memory_recall",
                    "kind": "status",
                    "phase": "intake",
                    "text": (
                        "Langzeit-Memory ist fuer diesen Lauf nicht "
                        "nutzbar."
                    ),
                    "final": True,
                },
            )

    def _resume_value(self, run_id: str, snapshot: Any) -> Any:
        """Resolve the resume payload from the control store (rule R5)."""
        raw: Any = None
        interrupt_id = ""
        for task in snapshot.tasks:
            for intr in task.interrupts:
                raw = intr.value
                interrupt_id = str(intr.id)
                break
        origin, payload = translate_kernel_interrupt(raw)
        if origin == CLARIFICATION_INTERRUPT:
            record = run_coro(
                self._control.get_clarification(run_id, payload["id"])
            )
            return {
                "kind": "clarification",
                "status": record.status,
                "answer": record.answer,
                "option_id": record.option_id,
                "answers": dict(record.answers),
            }
        if origin == CHILDREN_INTERRUPT:
            # No row backs a children wait (R5): the tool re-reads its
            # child's terminal outcome from the run store itself.
            return {"kind": "children"}
        approval = run_coro(
            self._control.get_approval(
                run_id, _tool_approval_id(run_id, interrupt_id)
            )
        )
        if approval.status == "pending":
            # The worker only wakes on a decision; a pending row here
            # means the wake path broke — resuming blind would grant
            # something no one approved.
            raise RuntimeError(
                "Tool-Genehmigung ist noch offen — Resume ohne "
                "Entscheidung ist nicht moeglich."
            )
        return {
            "decisions": _hitl_decisions(
                approval, action_count=len(payload["action_requests"])
            )
        }

    def _ensure_tool_approval(
        self,
        run_id: str,
        deps: KernelDeps,
        *,
        interrupt_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Idempotently create the ``kind="tool"`` approval row (R5).

        The approval id derives from the langgraph interrupt id, which
        is STABLE across re-entry and agent rebuilds (frozen by the
        harness contract tests) — a crash between park and decision
        re-enters here and finds the existing row instead of asking
        twice.
        """
        approval_id = _tool_approval_id(run_id, interrupt_id)
        try:
            run_coro(self._control.get_approval(run_id, approval_id))
            return
        except ApprovalNotFound:
            pass
        actions = [
            {
                "tool": str(request.get("name", "")),
                # The args ARE the approval content (a web query stands
                # here verbatim, plan `2.3`).
                "args": dict(request.get("args") or {}),
                "summary": str(request.get("description", "")),
            }
            for request in payload["action_requests"]
        ]
        record = run_coro(
            self._control.create_approval(
                ApprovalRecord(
                    approval_id=approval_id,
                    run_id=run_id,
                    kind="tool",
                    payload={"actions": actions},
                    interrupt_key=interrupt_id,
                )
            )
        )
        deps.emit(
            "inqtrix.agent.approval.requested",
            {
                "approval_id": record.approval_id,
                "kind": "tool",
                "actions": actions,
            },
        )

    # -- graph --------------------------------------------------------------#

    def _compiled_graph(
        self,
        autonomy: str,
        *,
        source_policy: Any,
        max_tool_calls: int,
        execution_directive: str = "",
    ) -> Any:
        """The compiled kernel graph of one policy variant (cached).

        Stateless and shared across runs: the chat model delegates to
        the deps ContextVar, so no per-run state is baked in — only the
        policy's ``interrupt_on`` and enforced source tool surface differ per
        variant (both are compile-time).
        """
        key = (
            autonomy,
            source_policy.web,
            source_policy.knowledge,
            execution_directive,
            max_tool_calls,
        )
        with self._graph_lock:
            if key not in self._graphs:
                from inqtrix.agents.harness import build_kernel_agent

                from inqtrix.agents.prompts import (
                    build_agent_kernel_system_prompt,
                )
                from inqtrix.agents.source_policy import kernel_tool_allowed

                tools = [
                    tool
                    for tool in build_kernel_tools()
                    if kernel_tool_allowed(
                        str(getattr(tool, "name", "")),
                        policy=source_policy,
                        execution_directive=execution_directive,
                    )
                ]

                self._graphs[key] = build_kernel_agent(
                    build_tool_chat_model(
                        _DepsChatProvider(),
                        # THE one usage-booking channel: per generation
                        # into the current segment's accumulator.
                        usage_hook=lambda prompt, completion: kernel_deps()
                        .book_usage(prompt, completion),
                    ),
                    tools=tools,
                    system_prompt=build_agent_kernel_system_prompt(),
                    interrupt_on=interrupt_config_for(autonomy),
                    checkpointer=self._checkpointer.saver(),
                    max_tool_calls=max_tool_calls,
                    # Summarization is graph-compile-time; its TRIGGER is
                    # per-run via the deps ContextVar, so the cache key
                    # stays model-independent.
                    context_keep_messages=(
                        self._platform.kernel_context_keep_messages
                    ),
                )
            return self._graphs[key]


def _final_answer(messages: list[Any]) -> str:
    """The last AI message text of a finished kernel loop."""
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return normalize_agent_markdown(content)
    return ""


def configured_kernel_limits(
    platform: Any, *, depth: str, tier: str
) -> tuple[int, int, int, int]:
    """Return base/ceiling pairs from operator settings for one run.

    The ``schnell`` tier remains intentionally non-extendable: extending its
    step budget would silently turn the advertised seconds-scale route into a
    normal research run. Users can still accept the explicit partial result
    or cancel and submit with another tier.
    """
    deep = depth == "deep"
    tool_base = int(
        platform.kernel_max_tool_calls_deep
        if deep
        else platform.kernel_max_tool_calls
    )
    tool_ceiling = max(
        tool_base,
        int(
            getattr(
                platform,
                (
                    "kernel_max_tool_calls_extension_ceiling_deep"
                    if deep
                    else "kernel_max_tool_calls_extension_ceiling"
                ),
                tool_base,
            )
        ),
    )
    step_base = int(
        platform.kernel_max_iterations_deep
        if deep
        else platform.kernel_max_iterations
    )
    step_ceiling = max(
        step_base,
        int(
            getattr(
                platform,
                (
                    "kernel_max_iterations_extension_ceiling_deep"
                    if deep
                    else "kernel_max_iterations_extension_ceiling"
                ),
                step_base,
            )
        ),
    )
    if tier == "schnell":
        step_base = min(
            step_base,
            _ANSWER_TURN_SUPERSTEPS
            + _SUPERSTEPS_PER_TOOL_TURN * _SCHNELL_TOOL_TURNS,
        )
        step_ceiling = step_base
        tool_ceiling = tool_base
    return tool_base, tool_ceiling, step_base, step_ceiling


def _partial_limit_answer(
    gate: AgentLimitGate, references: list[dict[str, Any]]
) -> str:
    """Build a non-generative, explicitly incomplete answer receipt.

    A limit choice must not trigger another unbounded model call. The receipt
    therefore reports only persisted evidence metadata; it makes no new claim
    and points users to the complete evidence ledger for inspection.
    """
    limit_name = (
        "Werkzeuglimit" if gate.kind == "tool_calls" else "Schrittlimit"
    )
    lines = [
        "## Teilstand – ausdrücklich übernommen",
        "",
        (
            f"Der Lauf wurde auf deine Entscheidung als Teilstand beendet, "
            f"nachdem das konfigurierte {limit_name} erreicht wurde. "
            "Es wurde keine vollständige Synthese vorgetäuscht."
        ),
        "",
        "### Gesicherte Belegbasis",
        "",
    ]
    if not references:
        lines.append(
            "Bis zu diesem Punkt wurde noch keine zitierfähige Quelle gespeichert."
        )
    else:
        for reference in references[:30]:
            label = str(reference.get("label") or "Quelle").strip()
            title = str(
                reference.get("title")
                or reference.get("document_title")
                or reference.get("url")
                or "Quelle ohne Titel"
            )
            title = " ".join(title.split()).replace("[", "\\[").replace(
                "]", "\\]"
            )
            lines.append(f"- [{label}] {title}")
        omitted = len(references) - 30
        if omitted > 0:
            lines.append(
                f"- Weitere {omitted} Quellen bleiben im Belegprotokoll erhalten."
            )
    lines.extend(
        [
            "",
            "### Offene Lücke",
            "",
            (
                "Bewertung, Widerspruchsprüfung und finale Antwort wurden nicht "
                "vollständig abgeschlossen. Für eine belastbare Endantwort muss "
                "der Auftrag mit einem ausreichend hohen, vom Betreiber erlaubten "
                "Limit fortgesetzt oder neu eingegrenzt werden."
            ),
        ]
    )
    return normalize_agent_markdown("\n".join(lines))


_CONTEXT_TRIGGER_FLOOR_TOKENS = 128_000
"""Auto-trigger floor AND the no-card fallback: compacting earlier than
128k would churn the prompt cache for runs that fit comfortably, and a
model without a card gets a conservative threshold instead of the
deepagents 170k default that the kernel's ceilings never reach."""


def resolve_context_trigger_tokens(
    platform: Any, model_id: str | None
) -> int:
    """Per-run compaction threshold: explicit pin > model card > floor.

    Pure resolver (no I/O): an explicit
    ``kernel_context_trigger_tokens`` pin wins verbatim; otherwise the
    resolved model's card yields ``context_window_tokens x fraction``
    bounded below by the floor; an unknown model uses the floor.
    """
    pinned = int(getattr(platform, "kernel_context_trigger_tokens", 0) or 0)
    if pinned > 0:
        return pinned
    fraction = float(
        getattr(platform, "kernel_context_trigger_fraction", 0.75) or 0.75
    )
    from inqtrix.model_cards import resolve_model_card

    card = resolve_model_card(model_id) if model_id else None
    window = int(getattr(card, "context_window_tokens", 0) or 0)
    if window <= 0:
        return _CONTEXT_TRIGGER_FLOOR_TOKENS
    if window <= _CONTEXT_TRIGGER_FLOOR_TOKENS:
        # A floor at/above the model's whole window would mean "never
        # compact before provider overflow" — small-window models use the
        # plain fraction instead.
        return int(window * fraction)
    return max(_CONTEXT_TRIGGER_FLOOR_TOKENS, int(window * fraction))


def _stage_kernel_memory(deps: KernelDeps, answer: str) -> None:
    """Stage candidate-only long-term memories after a successful run.

    Mission parity (``_stage_memory_candidates``): candidates only —
    nothing is written to memory without the user's later review. The
    kernel's substrate is its final answer (it has no memo); the shared
    ``run_memory_reflection`` + ``stage_candidates`` path is reused, so
    there is exactly ONE reflection/staging implementation. Optional by
    contract: every failure logs and returns, it can never fail a run
    that already has its answer.
    """
    service = deps.memory
    principal = deps.principal
    if (
        service is None
        or principal is None
        or not deps.memory_opt_in
        or not answer.strip()
    ):
        return
    status = service.status(principal)
    if (
        status.get("provider") == "none"
        or status.get("mode") == "off"
        or not status.get("principal_eligible")
    ):
        return
    from inqtrix.agents import memory_reflection
    from inqtrix.agents.phase_models import MemoryReflection

    provider_models = getattr(deps.llm, "models", None)
    model: str | None = None
    effort: str | None = None
    if provider_models is not None:
        desc = describe_resolution(
            "agent_memory_reflection",
            provider_models,
            "",
            requested_model="",
            requested_effort="",
        )
        model = desc.get("model") or None
        effort = desc.get("effort") or None
    try:
        outcome = memory_reflection.run_memory_reflection(
            deps.llm,
            question=deps.question,
            memo_markdown=answer,
            critic_digest="",
            task_digest="",
            model=model,
            reasoning_effort=effort,
            timeout=deps.timeout,
        )
    except Exception as exc:  # noqa: BLE001 - candidate staging is optional
        log.warning(
            "Kernel-Memory-Reflection fehlgeschlagen (error_type=%s).",
            type(exc).__name__,
        )
        # Mission parity: the failure is an EVENT, not just a log line
        # (_stage_memory_candidates emits the same activity kind).
        deps.emit(
            "inqtrix.agent.activity",
            {
                "kind": "memory_unavailable",
                "label": "Memory unavailable",
                "detail": "Memory-Kandidaten konnten nicht erzeugt werden.",
            },
        )
        return
    deps.book_usage(
        outcome.usage.get("prompt_tokens", 0),
        outcome.usage.get("completion_tokens", 0),
    )
    reflection = outcome.value
    if not isinstance(reflection, MemoryReflection):
        return
    if not reflection.candidates:
        return
    try:
        staged = run_coro(
            service.stage_candidates(
                principal=principal,
                candidates=[
                    candidate.model_dump()
                    for candidate in reflection.candidates
                ],
                source_run_id=deps.run_id,
            )
        )
    except Exception as exc:  # noqa: BLE001 - candidate staging is optional
        log.warning(
            "Kernel-Memory-Staging fehlgeschlagen (error_type=%s).",
            type(exc).__name__,
        )
        return
    log.info(
        "Kernel-Memory: %d Kandidaten zur Pruefung vorgemerkt.",
        len(staged) if isinstance(staged, list) else 0,
    )


_INLINE_CITATION_LABEL = re.compile(r"\[([KW]\d+)\]")
_INLINE_MARKDOWN_URL = re.compile(r"\[[^\]\n]+\]\((https?://[^\s)]+)\)")
_UNRESOLVED_CITATION_MARKER = re.compile(
    r"\[(?:unbelegt|unsupported):\s*[KW]\d+\]",
    re.IGNORECASE,
)
_UNRESOLVED_HARD_CLAIM_MARKER = re.compile(
    r"\[(?:unbelegt:\s*harte\s+aussage|unsupported:\s*hard\s+claim)\]",
    re.IGNORECASE,
)


def _result_references(
    answer: str, ledger: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """References the final answer surfaces: cited-only when it cites.

    An answer that carries ``[K#]/[W#]`` labels lists EXACTLY the cited
    subset (F5: no read-everything reference dumps). An answer without
    citation labels keeps the full trusted ledger — those sources were
    the basis of the work, and an empty list would hide them (quick and
    conversational answers rarely cite inline). One helper for both the
    answer artifact and ``result_state.report_references`` so the run
    result and the rendered artifact can never disagree.  Any presence of
    unknown inline labels disables the uncited-answer basis fallback: mixed
    answers retain only their known cited subset, and unknown-only answers
    surface no falsely attributed references.
    """
    from inqtrix.agents.report_quality import (
        cited_references,
        unknown_citation_labels,
    )

    unknown = unknown_citation_labels(answer, ledger)
    unresolved_marker = bool(
        _UNRESOLVED_CITATION_MARKER.search(answer)
        or _UNRESOLVED_HARD_CLAIM_MARKER.search(answer)
    )
    if unknown:
        log.warning(
            "Antwort zitiert unbekannte Belege-Labels "
            "(unknown_count=%d).",
            len(unknown),
        )
    cited = cited_references(answer, ledger)
    if unknown or unresolved_marker:
        return cited
    linked_urls = {
        normalize_url(match.group(1))
        for match in _INLINE_MARKDOWN_URL.finditer(answer)
        if normalize_url(match.group(1))
    }
    if linked_urls:
        return [
            dict(item)
            for item in ledger
            if normalize_url(str(item.get("url") or "")) in linked_urls
        ]
    return cited if cited else [dict(item) for item in ledger]


def _mark_unresolved_citation_labels(
    answer: str,
    unknown: list[str],
    *,
    language: str,
) -> str:
    """Replace unresolved citation syntax with an explicit visible marker."""
    unresolved = set(unknown)
    marker = "unbelegt" if language == "de" else "unsupported"
    return _INLINE_CITATION_LABEL.sub(
        lambda match: (
            f"[{marker}: {match.group(1)}]"
            if match.group(1) in unresolved
            else match.group(0)
        ),
        answer,
    )


def _validate_kernel_answer_citations(
    deps: KernelDeps,
    answer: str,
    ledger: list[dict[str, Any]],
) -> str:
    """Boundedly repair final chat citations or mark unresolved labels.

    Canvas writes return a citation error to the bounded kernel loop, giving
    the model one chance to correct the tool call.  A tool-free final chat
    answer has no next loop turn, so it uses the shared synthesis repair once.
    If that call fails or still invents labels, only the unsupported citation
    syntax is replaced with a visible ``unbelegt`` marker; valid labels and
    prose remain unchanged.
    """
    from inqtrix.agents.prompts import agent_answer_system_prompt
    from inqtrix.agents.report_quality import (
        CitationValidationFailed,
        cited_references,
        unknown_citation_labels,
        validate_and_repair_citations,
    )
    from inqtrix.exceptions import AgentCancelled
    from inqtrix.execution_authority import AuthorizationRevoked

    unknown = unknown_citation_labels(answer, ledger)
    if not unknown:
        return answer
    language = detect_ui_language(deps.question)

    deps.check_abort()
    known_labels = [
        str(reference.get("label") or "")
        for reference in ledger
        if str(reference.get("label") or "")
    ]
    try:
        repaired, usage = validate_and_repair_citations(
            deps.llm,
            markdown=answer,
            known_labels=known_labels,
            usage={},
            system=agent_answer_system_prompt(),
            model=deps.model,
            reasoning_effort=deps.reasoning_effort,
            timeout=deps.timeout,
        )
    except CitationValidationFailed as exc:
        deps.book_usage(
            exc.usage.get("prompt_tokens", 0),
            exc.usage.get("completion_tokens", 0),
        )
        remaining = unknown_citation_labels(answer, ledger)
        marked = _mark_unresolved_citation_labels(
            answer,
            remaining,
            language=language,
        )
        deps.emit(
            "inqtrix.agent.citation.validation",
            {
                "status": "degraded",
                "unknown_labels": remaining,
                "resolution": "marked_unsubstantiated",
            },
        )
        return marked
    except (AgentCancelled, AuthorizationRevoked):
        raise
    except Exception as exc:  # noqa: BLE001 — deterministic safe degradation
        log.warning(
            "Bounded citation repair failed (error_type=%s); unresolved "
            "labels are marked.",
            type(exc).__name__,
        )
        remaining = unknown_citation_labels(answer, ledger)
        marked = _mark_unresolved_citation_labels(
            answer,
            remaining,
            language=language,
        )
        deps.emit(
            "inqtrix.agent.citation.validation",
            {
                "status": "degraded",
                "unknown_labels": remaining,
                "resolution": "marked_unsubstantiated",
            },
        )
        return marked

    deps.book_usage(
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )
    deps.emit(
        "inqtrix.agent.citation.validation",
        {
            "status": "repaired",
            "unknown_labels": unknown,
            "resolution": "bounded_repair",
        },
    )
    if not cited_references(repaired, ledger):
        # A valid repair may remove an unsupported label when no honest
        # replacement exists. Preserve that fact in the rendered answer so
        # the reference projector cannot mistake it for an originally
        # citation-free answer and attach the whole ledger.
        marker = "unbelegt" if language == "de" else "unsupported"
        markers = ", ".join(f"[{marker}: {label}]" for label in unknown)
        notice = (
            f"> Belegstatus: Nicht auflösbare Zitationslabels wurden "
            f"entfernt: {markers}."
            if language == "de"
            else f"> Evidence status: Unresolvable citation labels were "
            f"removed: {markers}."
        )
        repaired = normalize_agent_markdown(f"{repaired}\n\n{notice}")
    return repaired


def _kernel_consent_reason(deps: KernelDeps) -> str:
    """Canonical consent-policy label before any per-call decision."""
    if deps.execution_directive == "quick_web":
        return (
            "strict_approval_required"
            if deps.autonomy == "strict"
            else "explicit_directive"
        )
    return (
        "autonomous_policy"
        if deps.autonomy == "autonomous"
        else "permission_policy"
    )


def _observe_update(
    deps: KernelDeps, update: dict[str, Any]
) -> list[Any]:
    """Emit follow-the-agent events from one stream update; return interrupts.

    THE one place the raw update stream is interpreted: tool identity ->
    deterministic localized narration and started event (redacted args
    preview), tool results -> finished, todo state -> todo.updated. Model
    prose never becomes narration. Events are signals (R1) — clients refetch,
    so a replayed duplicate is harmless.
    """
    if "__interrupt__" in update:
        return list(update["__interrupt__"])
    for delta in update.values():
        if not isinstance(delta, dict):
            continue
        for message in delta.get("messages") or []:
            role = getattr(message, "type", "")
            if role == "ai":
                tool_calls = getattr(message, "tool_calls", None) or []
                message_id = str(getattr(message, "id", "") or "")
                invocation_ids = [
                    _tool_call_invocation_id(message_id, call, index)
                    for index, call in enumerate(tool_calls)
                    if isinstance(call, dict)
                ]
                has_new_invocation = any(
                    invocation_id not in deps.emitted_tool_start_ids
                    for invocation_id in invocation_ids
                )
                # Provider prose accompanying a tool call is untrusted answer
                # material, not process state. Narration is derived from the
                # accepted capability identity; tool-free AI content travels
                # exclusively through the answer publication/output channel.
                if tool_calls and has_new_invocation:
                    narration = _tool_intent_narration(
                        tool_calls,
                        language=detect_ui_language(deps.question),
                    )
                    narration_identity = {
                        "message_id": message_id,
                        "tool_calls": [
                            {
                                "id": str(call.get("id", "") or ""),
                                "name": str(call.get("name", "") or ""),
                            }
                            for call in tool_calls
                            if isinstance(call, dict)
                        ],
                    }
                    digest = hashlib.sha1(
                        json.dumps(
                            narration_identity,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()[:8]
                    deps.emit(
                        NARRATION_EVENT,
                        {
                            "narration_id": f"kernel_{digest}",
                            "kind": "intent",
                            "text": narration,
                            "phase": "execution",
                        },
                    )
                for index, call in enumerate(tool_calls):
                    if not isinstance(call, dict):
                        continue
                    invocation_id = _tool_call_invocation_id(
                        message_id, call, index
                    )
                    if invocation_id in deps.emitted_tool_start_ids:
                        continue
                    deps.emitted_tool_start_ids.add(invocation_id)
                    args = call.get("args", {})
                    args = args if isinstance(args, dict) else {}
                    query = args.get("query", "")
                    # ONE preview form across both lanes (the quick-web
                    # lane emits the bare query): a query-only call
                    # previews as plain text, anything else as JSON.
                    preview = (
                        query
                        if isinstance(query, str)
                        and query.strip()
                        and len(args) == 1
                        else json.dumps(args, ensure_ascii=False)
                    )
                    if len(preview) > _ARGS_PREVIEW_LIMIT:
                        preview = (
                            preview[:_ARGS_PREVIEW_LIMIT] + "…"
                        )
                    deps.emit(
                        TOOL_STARTED_EVENT,
                        {
                            "tool": str(call.get("name", "")),
                            "tool_call_id": invocation_id,
                            "invocation_id": invocation_id,
                            "args_preview": preview,
                        },
                    )
            elif role == "tool":
                content = getattr(message, "content", "")
                invocation_id = str(
                    getattr(message, "tool_call_id", "") or ""
                )
                if not invocation_id:
                    invocation_id = str(getattr(message, "id", "") or "")
                if invocation_id in deps.emitted_tool_finish_ids:
                    continue
                deps.emitted_tool_finish_ids.add(invocation_id)
                deps.emit(
                    TOOL_FINISHED_EVENT,
                    {
                        "tool": str(getattr(message, "name", "") or ""),
                        "tool_call_id": invocation_id,
                        "invocation_id": invocation_id,
                        "status": str(
                            getattr(message, "status", "") or "success"
                        ),
                        "result_preview": str(content)[
                            :_ARGS_PREVIEW_LIMIT
                        ],
                    },
                )
        todos = delta.get("todos")
        if isinstance(todos, list):
            deps.emit(
                TODO_UPDATED_EVENT,
                {
                    "todos": [
                        {
                            "content": str(item.get("content", "")),
                            "status": str(item.get("status", "")),
                        }
                        for item in todos
                        if isinstance(item, dict)
                    ]
                },
            )
    return []


def _tool_call_invocation_id(
    message_id: str,
    call: dict[str, Any],
    ordinal: int,
) -> str:
    """Return the provider id or a deterministic fallback invocation id."""
    explicit = str(call.get("id", "") or "")
    if explicit:
        return explicit
    identity = {
        "args": call.get("args", {}),
        "message_id": message_id,
        "name": str(call.get("name", "") or ""),
        "ordinal": ordinal,
    }
    digest = hashlib.sha1(
        json.dumps(
            identity,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"call_{digest}"


def _tool_approval_id(run_id: str, interrupt_id: str) -> str:
    """Deterministic approval id of one HITL gate.

    The langgraph interrupt id is stable across re-entry and rebuilt
    agents (harness contract test) and already hash-shaped — no extra
    digest needed."""
    return f"apr_{run_id[-12:]}_tool_{interrupt_id[:12]}"


def _quick_web_approval_id(run_id: str) -> str:
    """Deterministic approval id for the strict quick-web lane."""
    return f"apr_{run_id[-12:]}_quick_web"


def _validated_quick_web_action(actions: Any) -> tuple[str, str]:
    """Extract the reviewed quick-web query or fail loudly.

    The generic approval edit endpoint guarantees one action and an args
    object, but this lane additionally validates its own capability schema so
    a malformed edited query can never reach the external provider.
    """
    if not isinstance(actions, list) or len(actions) != 1:
        raise RuntimeError(
            "Quick-Web-Genehmigung enthaelt nicht genau eine Aktion."
        )
    action = actions[0]
    if not isinstance(action, dict) or action.get("tool") != "web_instant":
        raise RuntimeError(
            "Quick-Web-Genehmigung enthaelt ein unerwartetes Werkzeug."
        )
    args = action.get("args")
    if not isinstance(args, dict):
        raise RuntimeError(
            "Quick-Web-Genehmigung enthaelt keine gueltigen Argumente."
        )
    query = args.get("query")
    recency = args.get("recency", "")
    if not isinstance(query, str) or not query.strip():
        raise RuntimeError(
            "Quick-Web-Genehmigung enthaelt keine Suchanfrage."
        )
    if recency not in ("", "day", "week", "month", "year"):
        raise RuntimeError(
            "Quick-Web-Genehmigung enthaelt einen ungueltigen "
            "Recency-Filter."
        )
    return query.strip(), str(recency)


def _hitl_decisions(
    approval: Any, *, action_count: int
) -> list[dict[str, Any]]:
    """Map one decided approval row onto HITL resume decisions.

    Approve/reject fan out over ALL actions of the gate (one row
    approves the whole batch); an edit is validated to exactly one
    action by the service, so its mapping is single-action by
    construction.
    """
    if approval.decision == "approve":
        return [{"type": "approve"} for _ in range(action_count)]
    if approval.decision == "reject":
        decision: dict[str, Any] = {"type": "reject"}
        if approval.note:
            decision["message"] = approval.note
        return [dict(decision) for _ in range(action_count)]
    if approval.decision == "edit":
        edited = (approval.decision_payload.get("actions") or [{}])[0]
        return [
            {
                "type": "edit",
                "edited_action": {
                    "name": str(edited.get("tool", "")),
                    "args": dict(edited.get("args") or {}),
                },
            }
        ]
    raise RuntimeError(
        f"Unbekannte Tool-Entscheidung {approval.decision!r} fuer "
        f"Genehmigung {approval.approval_id}."
    )


def _checkpointed_usage(snapshot: Any) -> dict[str, int]:
    """Token totals of all EARLIER segments, from checkpointed messages.

    The bridge stamps ``usage_metadata`` on every AI message, so the
    checkpoint itself carries the run-cumulative spend — no second
    bookkeeping channel for the token-budget check.
    """
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    if snapshot is None or not snapshot.values:
        return totals
    for message in snapshot.values.get("messages") or []:
        meta = getattr(message, "usage_metadata", None) or {}
        totals["prompt_tokens"] += int(meta.get("input_tokens", 0) or 0)
        totals["completion_tokens"] += int(
            meta.get("output_tokens", 0) or 0
        )
    return totals


def _checkpointed_steps(snapshot: Any) -> int:
    """Return the cumulative committed LangGraph super-step coordinate."""
    if snapshot is None:
        return 0
    metadata = getattr(snapshot, "metadata", None) or {}
    try:
        # LangGraph starts at -1 before the first committed graph step.
        return max(0, int(metadata.get("step", -1)) + 1)
    except (TypeError, ValueError):
        return 0


def _checkpointed_tool_call_count(snapshot: Any) -> int:
    """Count all model-requested calls from checkpointed AI messages."""
    if snapshot is None or not snapshot.values:
        return 0
    total = 0
    for message in snapshot.values.get("messages") or []:
        if isinstance(message, dict):
            calls = message.get("tool_calls") or []
            role = message.get("role") or message.get("type")
        else:
            calls = getattr(message, "tool_calls", None) or []
            role = getattr(message, "type", "")
        if role in {"ai", "assistant"}:
            total += len(calls)
    return total


def _checkpointed_tool_event_ids(
    snapshot: Any,
) -> tuple[set[str], set[str]]:
    """Reconstruct durable logical tool events from checkpointed messages."""
    starts: set[str] = set()
    finishes: set[str] = set()
    if snapshot is None or not snapshot.values:
        return starts, finishes
    for message in snapshot.values.get("messages") or []:
        if isinstance(message, dict):
            role = str(message.get("role") or message.get("type") or "")
            message_id = str(message.get("id", "") or "")
            calls = message.get("tool_calls") or []
            tool_call_id = str(message.get("tool_call_id", "") or "")
        else:
            role = str(getattr(message, "type", "") or "")
            message_id = str(getattr(message, "id", "") or "")
            calls = getattr(message, "tool_calls", None) or []
            tool_call_id = str(
                getattr(message, "tool_call_id", "") or ""
            )
        if role in {"ai", "assistant"}:
            for index, call in enumerate(calls):
                if isinstance(call, dict):
                    starts.add(
                        _tool_call_invocation_id(message_id, call, index)
                    )
        elif role == "tool":
            finishes.add(tool_call_id or message_id)
    finishes.discard("")
    return starts, finishes


def _checkpointed_tool_use_counts(snapshot: Any) -> dict[str, int]:
    """Rehydrate cumulative source-tool usage from persisted tool messages.

    SUCCESS-only, mirroring the live counter (which increments after a
    successful invoke): failed calls persist as visible failure texts
    (``SOURCE_TOOL_FAILURE_PREFIXES``) or error-status ToolMessages and
    must not consume budgets or arm the sufficiency judge on resume.
    """
    from inqtrix.agents.kernel.tools import SOURCE_TOOL_FAILURE_PREFIXES
    from inqtrix.agents.source_policy import (
        KNOWLEDGE_TOOL_NAMES,
        WEB_TOOL_NAMES,
    )

    counts = {"web": 0, "knowledge": 0}
    if snapshot is None or not snapshot.values:
        return counts
    for message in snapshot.values.get("messages") or []:
        if getattr(message, "type", "") != "tool":
            continue
        if str(getattr(message, "status", "") or "") == "error":
            continue
        content = str(getattr(message, "content", "") or "")
        if content.startswith(SOURCE_TOOL_FAILURE_PREFIXES):
            continue
        name = str(getattr(message, "name", "") or "")
        if name in WEB_TOOL_NAMES:
            counts["web"] += 1
        elif name in KNOWLEDGE_TOOL_NAMES:
            counts["knowledge"] += 1
    return counts
