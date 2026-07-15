"""The cognitive kernel algorithm (plan M2 step 5 — walking skeleton).

``mode=agent_kernel`` is a SECOND registered ``AgentAlgorithm`` next to
the deterministic phase machine (ADR-PLAT-2): an LLM tool-calling loop
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
import logging
import threading
from typing import TYPE_CHECKING, Any

from inqtrix.constants import REASONING_TIMEOUT
from inqtrix.agents.checkpoint_guard import ensure_checkpoint_restart_safe
from inqtrix.agents.control_ports import ApprovalNotFound, ApprovalRecord
from inqtrix.agents.evidence import enrich_instant_evidence
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
from inqtrix.agents.markdown import normalize_agent_markdown
from inqtrix.agents.patterns._structured import observe_structured_retries
from inqtrix.agents.telemetry import model_retry_activity, provider_retry_activity
from inqtrix.agents.skills_runtime import (
    skill_model_pins,
    strictest_requires_plan,
)
from inqtrix.agents.web_execution_policy import derive_web_research_policy
from inqtrix.execution_authority import pinned_knowledge_collection_ids
from inqtrix.model_routing import (
    describe_resolution,
    describe_unresolved_resolution,
)
from inqtrix.providers.base import observe_provider_retries

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
    """The conversational kernel over the platform seams (ADR-PLAT-2).

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
    ) -> None:
        self._control = control_store
        self._checkpointer = checkpointer
        self._platform = platform
        self._capabilities = capability_registry
        self._permissions = permission_service
        self._run_service = run_service
        self._resolver = resolver
        self._skill_service = skill_service
        # One compiled graph per policy: interrupt_on is compile-time
        # and the policy is fixed per run (plan M2 `2.2`).
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
        deps.question = request.question
        deps.session_id = request.session_id or ""
        deps.run_service = self._run_service
        deps.resolver = self._resolver
        deps.principal = context.principal
        deps.autonomy = autonomy
        deps.skill_service = self._skill_service
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
        deps.explicit_web_research = research_policy.allowed
        deps.web_research_profile = research_policy.profile
        deps.hydrate_evidence()
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
        graph = self._compiled_graph(
            autonomy,
            source_policy=deps.source_policy,
            execution_directive=request.execution_directive,
            max_tool_calls=(
                self._platform.kernel_max_tool_calls_deep
                if depth == "deep"
                else self._platform.kernel_max_tool_calls
            ),
        )
        config = {
            "configurable": {"thread_id": run_id},
            # Hard iteration bound (plan M2 `2.7`): deepagents lifts
            # langgraph's default to 9999 — a runaway tool loop must
            # fail LOUDLY (GraphRecursionError -> failed run) instead.
            # Deep raises the ceiling (M4), it never removes it.
            # The schnell tier additionally clamps the ceiling: a
            # seconds-scale run has no business iterating dozens of
            # times (deterministic budget, never prompt-only).
            "recursion_limit": (
                self._platform.kernel_max_iterations_deep
                if depth == "deep"
                else min(
                    self._platform.kernel_max_iterations,
                    8 if tier == "schnell" else (
                        self._platform.kernel_max_iterations
                    ),
                )
            ),
        }

        existing = graph.get_state(config)
        deps.prior_usage = _checkpointed_usage(existing)
        deps.tool_use_counts = _checkpointed_tool_use_counts(existing)
        self._reactivate_loaded_skills(deps, existing)
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
        finally:
            set_kernel_deps(None)

        if interrupts:
            if len(interrupts) > 1:
                # Parallel gates have no defined resume mapping yet —
                # parking would strand the run half-answered (loud
                # failure beats an undefined resume, plan M2 skeleton).
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
        return AgentResult(
            answer=answer,
            result_type="agent_kernel_result",
            raw={
                "answer": answer,
                "usage": dict(deps.usage),
                "result_state": {
                    "answer": answer,
                    "cancelled": False,
                    "report_references": list(
                        deps.evidence_refs.values()
                    ),
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

    def _execution_payload(
        self,
        request: "RunRequest",
        deps: KernelDeps,
        *,
        consent_reason: str,
    ) -> dict[str, object]:
        """Canonical Agent Desk execution block for this kernel run."""
        from inqtrix.agents.source_policy import execution_payload

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
                actions = [
                    {
                        "tool": "web_instant",
                        "args": {"query": query, "recency": recency},
                        "summary": "Eine direkte Websuche ausfuehren.",
                    }
                ]
                approval = run_coro(
                    self._control.create_approval(
                        ApprovalRecord(
                            approval_id=approval_id,
                            run_id=deps.run_id,
                            kind="tool",
                            payload={"actions": actions},
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
            query, recency = _validated_quick_web_action(action_set)
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

        references = deps.register_references(
            enrich_instant_evidence(
                str(output.answer or ""),
                [
                    {
                        "url": source.url,
                        "title": source.title or None,
                        "excerpt": source.snippet or None,
                    }
                    for source in output.sources
                ],
            )
        )
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
            f"AKTUELLE FRAGE:\n{request.question}\n\n"
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
            return request.question.strip(), ""
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
            elif item.get("grounded_support"):
                support_label = "Geerdeter Antwortkontext (kein Quellenauszug)"
                support = item["grounded_support"]
            else:
                support_label = "Quellenauszug"
                support = "(nicht vorhanden)"
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
            response = deps.llm.complete_with_metadata(
                (
                    "Beantworte die Nutzerfrage knapp und direkt in ihrer "
                    "Sprache. Verwende ausschliesslich das abgegrenzte "
                    "Websuchergebnis und die gelieferten Quellen. Verlinke "
                    "Tatsachen mit den passenden URLs; benenne fehlende oder "
                    "widerspruechliche Evidenz offen.\n\n"
                    f"NUTZERFRAGE:\n{request.question}\n\n"
                    f"RELEVANTER VERLAUF:\n"
                    f"{deps.session_history[-4000:] or '(kein Verlauf)'}\n\n"
                    f"SUCHANFRAGE:\n{query}\n\n"
                    "BEGINN WEBSUCHERGEBNIS\n"
                    f"{provider_answer}\n\n{source_block}\n"
                    "ENDE WEBSUCHERGEBNIS"
                ),
                system=(
                    "Du formulierst eine direkt belegte Schnell-Web-Antwort. "
                    "Fuehre kein eigenes Wissen als Fakt ein."
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
            answer = provider_answer.strip()
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
        return AgentResult(
            answer=answer,
            result_type="agent_kernel_result",
            raw={
                "answer": answer,
                "usage": dict(deps.usage),
                "result_state": {
                    "answer": answer,
                    "cancelled": False,
                    "report_references": references,
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
            SKILL_INPUTS_RESOLVED_MARKER,
        )

        resolved_skill_blocks: list[str] = []
        for message in messages:
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                continue
            message_type = type(message).__name__
            trusted_resolved_block = (
                message_type == "HumanMessage"
                and content.startswith(SKILL_INPUTS_RESOLVED_MARKER)
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
                "Deep-Review-Aufruf fehlgeschlagen (%s) — die Antwort "
                "bleibt unveraendert.",
                exc,
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
                "Deep-Revision fehlgeschlagen (%s) — die urspruengliche "
                "Antwort bleibt bestehen.",
                exc,
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
            log.warning("Deep-Revision CAS-Konflikt: %s", exc)
            _review_narration(
                "CAS-Konflikt — bestehende Outputs bleiben unveraendert.",
                final=True,
            )
            return answer
        except Exception as exc:  # noqa: BLE001 — preserve reviewed outputs
            log.warning("Deep-Revision Store-Fehler: %s", exc)
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
            # Deep (plan M4 `4.1.1`): the kernel node runs on high
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
            except Exception:  # noqa: BLE001 — a lookup outage costs the list
                log.warning(
                    "Skill-Disclosure nicht verfuegbar (Lookup-Fehler).",
                    exc_info=True,
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
            knowledge_collection_ids=knowledge_collection_ids,
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
            request.question,
            history_block=history_block,
            artifact_registry=registry,
            last_response_form=last_form,
            prior_evidence_count=prior_evidence_count,
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

    THE one place the raw update stream is interpreted: intent prose ->
    narration (deterministic content-hash id, safe under replay), tool
    calls -> started (redacted args preview), tool results -> finished,
    todo state -> todo.updated. Events are signals (R1) — clients
    refetch, so a replayed duplicate is harmless.
    """
    import hashlib
    import json

    if "__interrupt__" in update:
        return list(update["__interrupt__"])
    for delta in update.values():
        if not isinstance(delta, dict):
            continue
        for message in delta.get("messages") or []:
            role = getattr(message, "type", "")
            if role == "ai":
                text = getattr(message, "content", "")
                if isinstance(text, str) and text.strip():
                    digest = hashlib.sha1(
                        text.encode("utf-8")
                    ).hexdigest()[:8]
                    deps.emit(
                        NARRATION_EVENT,
                        {
                            "narration_id": f"kernel_{digest}",
                            "kind": "intent",
                            "text": text.strip(),
                            "phase": "execution",
                        },
                    )
                for call in getattr(message, "tool_calls", None) or []:
                    preview = json.dumps(
                        call.get("args", {}), ensure_ascii=False
                    )
                    if len(preview) > _ARGS_PREVIEW_LIMIT:
                        preview = (
                            preview[:_ARGS_PREVIEW_LIMIT] + "…"
                        )
                    deps.emit(
                        TOOL_STARTED_EVENT,
                        {
                            "tool": str(call.get("name", "")),
                            "tool_call_id": str(call.get("id", "")),
                            "args_preview": preview,
                        },
                    )
            elif role == "tool":
                content = getattr(message, "content", "")
                deps.emit(
                    TOOL_FINISHED_EVENT,
                    {
                        "tool": str(getattr(message, "name", "") or ""),
                        "tool_call_id": str(
                            getattr(message, "tool_call_id", "") or ""
                        ),
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


def _checkpointed_tool_use_counts(snapshot: Any) -> dict[str, int]:
    """Rehydrate cumulative source-tool usage from persisted tool messages."""
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
        name = str(getattr(message, "name", "") or "")
        if name in WEB_TOOL_NAMES:
            counts["web"] += 1
        elif name in KNOWLEDGE_TOOL_NAMES:
            counts["knowledge"] += 1
    return counts
