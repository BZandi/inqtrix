"""Native run submission through the algorithm registry.

Owns the run work-callable that used to live inline in the
``POST /v1/runs`` route body: it dispatches the resolved mode through
the registry, translates the algorithm result into the export payload,
and drives the run-handle lifecycle (events, answer deltas, snapshot,
completion). Read-side operations stay on the
:class:`~inqtrix.server.runs.RunStore` and are called directly by the
runs router — the store already is the in-memory repository.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

from inqtrix.core.context import RunContext, RuntimeContext
from inqtrix.core.results import (
    CanvasContext,
    RunRequest,
    SourcePolicy,
    WebRecency,
)
from inqtrix.execution_authority import AuthorizationRevoked, guard_provider_context
from inqtrix.execution_failures import RunExecutionFailure
from inqtrix.quota.models import QuotaDimension, QuotaSubject, consumed_tokens
from inqtrix.result import ResearchResult
from inqtrix.services.agent_answer_publisher import AgentAnswerPublisher
from inqtrix.services.agent_context import ResolvedAgentContext
from inqtrix.state import build_run_snapshot

if TYPE_CHECKING:
    from inqtrix.agents.control_ports import AgentControlStore
    from inqtrix.auth.principal import Principal
    from inqtrix.core.algorithms import AgentAlgorithm, AlgorithmRegistry
    from inqtrix.server.runs import RunHandle, RunStore
    from inqtrix.services.execution_dependency_authority import (
        ExecutionDependencyAuthorizer,
    )
    from inqtrix.services.quota_service import QuotaService


def execute_run_request(
    handle: "RunHandle",
    *,
    algorithm: "AgentAlgorithm",
    run_request: RunRequest,
    resolved: ResolvedAgentContext,
    runtime: RuntimeContext,
    principal: "Principal | None",
    quota_service: "QuotaService | None" = None,
    quota_subject: "QuotaSubject | None" = None,
    token_budget: int | None = None,
    workspace_id: str | None = None,
    authority_check: Callable[[], None] | None = None,
    answer_publisher: AgentAnswerPublisher | None = None,
) -> None:
    """Execute one resolved run request against its algorithm.

    The single execution body shared by the in-process run thread and
    the queue worker: dispatches the algorithm, honours cancellation
    observed at graph node boundaries, shapes the export payload, and
    drives the handle lifecycle (answer deltas, snapshot, completion).
    Exceptions propagate to the caller, which owns the failure path.

    Args:
        handle: Store-backed run handle (events, complete, cancel).
        algorithm: The registry entry for the request's mode.
        run_request: The validated run request to execute.
        resolved: Stack/override/mode resolution for this request.
        runtime: App-level runtime context.
        principal: Submitting identity; ``None`` in worker processes,
            which carry the explicit quota account in *quota_subject*
            instead.
        quota_service: Optional usage meter. When wired (quotas on), the
            run's real LLM-token consumption is booked through its
            synchronous bridge AFTER execution (non-fatal — a recording
            failure never loses the run), and the per-run token budget
            aborts a runaway run at a node boundary. ``None`` for
            unmetered deployments — byte-identical to the historical
            path.
        quota_subject: Explicit quota account for the worker path,
            reconstructed from the run row's persisted effective-actor UUID
            and tenant. When ``None`` (the in-process path), the account is
            derived from *principal*. This makes token accounting fire
            regardless of which process executes the run.
        workspace_id: Persisted project namespace of the native run. Agent
            algorithms pass it unchanged to delegated children; it is not an
            authorization decision.
        answer_publisher: Shared durable publisher for native Agent Desk
            modes.  Those modes fail loudly when it is absent; other modes
            retain the event-only answer lifecycle.
    """
    t0 = time.monotonic()

    # Enrich the run root span (opened by the execution boundary —
    # worker _execute or in-process _run_worker) with the Langfuse trace
    # fields: explicit langfuse.* keys are REQUIRED because this span is
    # a child of the submitter's context, and Langfuse takes trace-level
    # fields only from the root span or these keys.
    from inqtrix.auth.log_redaction import stable_pseudonym
    from inqtrix.observability import semconv
    from inqtrix.observability.otel import (
        current_trace_id_hex,
        enrich_current_span,
        mark_current_span_error,
    )

    subject_user_id = (
        getattr(principal, "user_id", None)
        or getattr(quota_subject, "user_id", None)
    )
    # getattr throughout: telemetry must never impose fields on the
    # request/handle doubles embedded callers and tests pass in.
    run_mode = str(getattr(run_request, "mode", "") or "")
    # Metrics feature label + ledger subject for every provider call
    # inside this segment (token counters per product feature; raw
    # booking identity for llm_usage rows). The executing loop clears
    # both vars in its finally — worker threads are reused.
    from inqtrix.observability.context import (
        bind_feature,
        bind_usage_subject,
    )

    bind_feature(run_mode)
    bind_usage_subject(
        getattr(principal, "tenant_id", None)
        or getattr(quota_subject, "tenant_id", None),
        subject_user_id,
        workspace_id,
    )
    enrich_current_span(
        {
            semconv.LANGFUSE_TRACE_NAME: f"run:{run_mode}",
            semconv.LANGFUSE_USER_ID: (
                stable_pseudonym("usr", subject_user_id)
                if subject_user_id is not None
                else ""
            ),
            semconv.LANGFUSE_SESSION_ID: (
                getattr(run_request, "session_id", "") or ""
            ),
            semconv.INQTRIX_WORKSPACE: workspace_id or "",
            "inqtrix.mode": run_mode,
            "inqtrix.stack": getattr(resolved, "stack_name", "") or "",
        }
    )
    # Persist the trace id as a durable run event so the admin surface
    # and the trace-export API can find the trace WITHOUT a lookup —
    # emitted once per execution segment (retries overwrite by recency).
    trace_id_hex = current_trace_id_hex()
    emit_event = getattr(handle, "emit", None)
    if trace_id_hex is not None and callable(emit_event):
        emit_event("inqtrix.run.trace", {"trace_id": trace_id_hex})

    providers = resolved.providers
    if authority_check is not None:
        authority_check()
        providers = guard_provider_context(providers, authority_check)

    run_context = RunContext(
        providers=providers,
        strategies=resolved.strategies,
        agent_settings=resolved.agent_settings,
        principal=principal,
        run_id=handle.run_id,
        workspace_id=workspace_id,
        cancel_token=handle.cancel_event,
        event_sink=handle.emit,
        # Opt-in hard per-run token cap (0 = off). Independent of the
        # monthly token quota: it bounds a single run, the quota blocks
        # the next one. The optional narrowing comes only from trusted
        # server callers; planner-authored task budgets are never input.
        token_budget=_effective_budget(
            runtime.settings.quota.max_tokens_per_run, token_budget
        ),
        park=handle.wait,
        authority_check=authority_check,
        # Children inherit the parent's provider stack through the
        # resolver (F7c) — same inheritance rule as workspace_id.
        stack_name=getattr(resolved, "stack_name", "") or "",
    )
    agent_result = algorithm.run(
        run_request,
        runtime=runtime,
        context=run_context,
    )
    result = agent_result.raw
    result_state = result.get("result_state", {}) or {}
    run_usage = result.get("usage", {}) or {}
    enrich_current_span(
        {
            semconv.GEN_AI_USAGE_INPUT_TOKENS: int(
                run_usage.get("prompt_tokens", 0) or 0
            ),
            semconv.GEN_AI_USAGE_OUTPUT_TOKENS: int(
                run_usage.get("completion_tokens", 0) or 0
            ),
            "inqtrix.outcome": (
                "cancelled"
                if (
                    agent_result.cancelled
                    or handle.cancel_event.is_set()
                )
                else ("parked" if handle.parked else "completed")
            ),
        }
    )
    # Book the LLM tokens this run actually consumed before any early
    # return: a budget-cancelled, client-cancelled, or PARKED run still
    # spent what it spent, and that spend must count toward the monthly
    # quota (it is what blocks the NEXT submission).
    if quota_service is not None:
        quota_account = (
            quota_subject
            if quota_subject is not None
            else quota_service.subject_for(principal)
        )
        quota_service.record_blocking(
            quota_account,
            QuotaDimension.LLM_TOKENS,
            consumed_tokens(result.get("usage")),
        )
    if handle.parked:
        # The algorithm parked the run (agent interrupt): the segment is
        # over, the store retained the closure/payload for the resume —
        # completing or cancelling here would destroy the wait.
        return
    if handle.cancel_event.is_set():
        handle.cancel("client_requested_cancel")
        return
    if agent_result.cancelled:
        if agent_result.cancel_reason == "token_budget_exceeded":
            handle.fail(
                "Der Lauf hat das serverseitige Tokenbudget erreicht.",
                error_type="token_budget_exceeded",
            )
            mark_current_span_error("token_budget_exceeded")
            enrich_current_span({"inqtrix.outcome": "failed"})
        else:
            handle.cancel(
                agent_result.cancel_reason or "client_requested_cancel"
            )
        return

    terminal_failure = result_state.get("_terminal_failure")
    if isinstance(terminal_failure, dict):
        raise RunExecutionFailure(
            str(terminal_failure.get("type") or "server_error"),
            str(
                terminal_failure.get("message")
                or "The algorithm reported a terminal execution failure."
            ),
        )

    research_result = ResearchResult.from_raw(result)
    segment_elapsed = round(time.monotonic() - t0, 2)
    total_elapsed_reader = getattr(handle, "total_elapsed_seconds", None)
    total_elapsed = (
        round(float(total_elapsed_reader()), 2)
        if callable(total_elapsed_reader)
        else segment_elapsed
    )
    # The durable store starts the timer before dispatch and never resets it
    # on resume. It therefore includes queued-visible execution and explicit
    # waiting segments. The monotonic segment value remains a lower bound for
    # custom non-store handles used by embedded callers and tests.
    research_result.metrics.elapsed_seconds = max(segment_elapsed, total_elapsed)
    answer = research_result.answer
    payload = research_result.to_export_payload()
    usage = result.get("usage", {})
    payload["usage"] = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": (
            usage.get("prompt_tokens", 0)
            + usage.get("completion_tokens", 0)
        ),
    }
    if authority_check is not None:
        authority_check()
    mode = str(getattr(run_request, "mode", "") or "")
    if mode in {"agent_kernel", "workspace_agent"}:
        _prune_uncollected_agent_metrics(payload)
        if answer_publisher is None:
            raise RunExecutionFailure(
                "answer_publication_unavailable",
                "Die persistente Agent-Antwortpublikation ist nicht verfuegbar.",
            )
        exported_references = payload.get("references", [])
        if not isinstance(exported_references, list):
            raise RunExecutionFailure(
                "answer_publication_invalid_references",
                "Die exportierten Antwortreferenzen sind nicht serialisierbar.",
            )
        answer_publisher.publish(
            handle,
            answer,
            references=exported_references,
            question=run_request.question,
        )
    else:
        handle.emit_answer(answer)
    if authority_check is not None:
        authority_check()
    current_node = str(algorithm.capabilities().get("terminal_node", "answer"))
    handle.complete(
        payload,
        snapshot=build_run_snapshot(
            result_state,
            current_node=current_node,
            last_message="completed",
        ),
    )


#: Research metrics an AGENT run genuinely produces. Everything else in
#: ``ResearchMetrics`` is computed by the research-desk graph, which an
#: agent run never executes.
_AGENT_REAL_METRIC_KEYS = frozenset(
    {"elapsed_seconds", "prompt_tokens", "completion_tokens"}
)


def _prune_uncollected_agent_metrics(payload: dict[str, Any]) -> None:
    """Drop research metrics an agent run never measured (P10-K6).

    Agent results ride the shared research serialization, so every
    research counter reached the wire as a hard zero: a mission with
    sixteen knowledge searches and seven citations reported
    ``total_queries: 0`` and ``total_citations: 0``. Zeros are a claim,
    not an absence — omitting the keys says "not measured here", which
    is the truth. What an agent run does measure (wall clock, tokens)
    stays.
    """
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return
    payload["metrics"] = {
        key: value
        for key, value in metrics.items()
        if key in _AGENT_REAL_METRIC_KEYS
    }


def _effective_budget(run_cap: int, override: int | None) -> int:
    """The tighter of the deployment cap and a caller override."""
    if not override or override <= 0:
        return run_cap
    if not run_cap or run_cap <= 0:
        return override
    return min(run_cap, override)


class RunService:
    """Submit native runs that execute via the algorithm registry.

    Args:
        registry: The algorithm registry; the resolved request mode
            selects the algorithm executed by the run worker.
        runtime: App-level runtime context handed through to the
            algorithm.
        run_store: The in-memory run registry/queue that owns
            dispatch, events, and retention.
        dependency_authorizer: Safepoint checker for the segment actor
            and every dependency pinned into the request. Required for a
            scoped actor — execution fails closed without it. It owns the
            actor directory, so the run thread never reaches the HTTP
            loop's pooled identity engine.
        answer_artifact_store: Authoritative agent-control store used by the
            shared answer publisher. Native Agent Desk modes fail loudly when
            this dependency is absent.
    """

    def __init__(
        self,
        *,
        registry: "AlgorithmRegistry",
        runtime: RuntimeContext,
        run_store: "RunStore",
        quota_service: "QuotaService | None" = None,
        dependency_authorizer: "ExecutionDependencyAuthorizer | None" = None,
        answer_artifact_store: "AgentControlStore | None" = None,
    ) -> None:
        self._registry = registry
        self._runtime = runtime
        self._run_store = run_store
        self._quota_service = quota_service
        self._dependency_authorizer = dependency_authorizer
        self._answer_publisher = (
            AgentAnswerPublisher(answer_artifact_store)
            if answer_artifact_store is not None
            else None
        )

    @property
    def dependency_authorizer(self) -> "ExecutionDependencyAuthorizer | None":
        """Shared pinned-dependency checker for API and queue workers."""
        return self._dependency_authorizer

    @property
    def run_store(self) -> "RunStore":
        """The underlying run store (read-side surface for the router)."""
        return self._run_store

    @property
    def answer_publisher(self) -> AgentAnswerPublisher | None:
        """Shared native Agent Desk answer publisher, when configured."""
        return self._answer_publisher

    def submit(
        self,
        *,
        question: str,
        history: str,
        messages: list[dict[str, Any]],
        resolved: ResolvedAgentContext,
        workspace_id: str | None,
        principal: "Principal | None" = None,
        kind: str = "standard",
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
        session_id: str | None = None,
        autonomy: str = "",
        document_id: str = "",
        response_form: str = "",
        token_budget: int | None = None,
        origin_key: str = "",
        skill_ids: list[str] | None = None,
        skill_revisions: dict[str, int] | None = None,
        tool_directives: list[str] | None = None,
        source_policy: dict[str, str] | SourcePolicy | None = None,
        web_recency: WebRecency | None = None,
        execution_directive: str = "",
        canvas_context: "CanvasContext | None" = None,
        report_requirement: str = "",
        attached_reports: list[dict[str, Any]] | None = None,
        parent_task_id: str = "",
        parent_task_attempt: int = 0,
    ) -> dict[str, Any]:
        """Queue one native run and return its public summary.

        Args:
            question: Normalized research question.
            history: Pre-formatted conversation history block.
            messages: Raw chat messages (carried on the run request).
            resolved: Stack/override/mode resolution for this request.
            workspace_id: Client-side UI namespace (NOT an
                authorization input).
            principal: Verified request identity, threaded into the
                run context for attribution.
            kind: Role in an agent tree (``standard`` — the default,
                emits no extra summary keys — or ``agent`` /
                ``agent_child``).
            parent_run_id: Direct parent for ``agent_child`` runs; the
                parent's cancel cascades over this link.
            root_run_id: Tree root (equals ``parent_run_id`` for depth-1
                trees); lets deep descendants resolve their root run
                without walking parents.
            session_id: Saved Agent or Knowledge session grouping the run.
            autonomy: Workspace-agent permission mode, carried on
                the run request and the worker replay payload; empty for
                non-agent modes.
            document_id: Target editor document of a workspace-agent
                patch assignment; empty when the run has no edit
                target. Carried like ``autonomy``.
            response_form: Workspace-agent output-form override
                (``chat``/``canvas``); empty delegates to the
                intake profile. Carried like ``autonomy``.
            token_budget: Optional trusted per-run LLM-token ceiling that
                narrows the deployment cap. Planner-authored task budgets
                are compatibility-only data and never flow into this field.
            origin_key: Idempotency key of the submitting kernel tool call or
                mission plan-task attempt. Persisted in the replay body and
                exposed on child summaries so checkpoint re-execution finds
                the already-submitted run. Empty for every other caller.
            skill_ids: Router-admitted skill ids; visibility
                and count cap already enforced; carried on the request
                and the worker replay payload.
            skill_revisions: Admission-time integer revision for each
                attached skill; runtime loading fails closed on drift.
            tool_directives: Whitelisted composer tool hints,
                carried like ``skill_ids``.
            source_policy: Availability of web and project-knowledge tools.
                ``None`` preserves the historical both-available surface.
                The normalized policy is persisted for worker replay and
                inherited by agent children.
            web_recency: Provider-neutral recency filter for delegated web
                research. ``None`` leaves recency inference to the research
                graph. Persisted for worker replay.
            execution_directive: Optional one-shot server-enforced route
                (``quick_web`` or ``knowledge_only``). Empty uses normal
                routing.
            report_requirement: Server-composed result requirement set
                before the run (free text plus attached library rules,
                each with its origin marker). Empty when none was set.
            attached_reports: Research reports the user attached,
                already resolved server-side to
                ``{report_id, title, reference_count}``. Only the names
                travel; the kernel tool fetches the bodies.
            canvas_context: Router-validated canvas attachment of an
                agent-kernel submission (open document + queued selection
                comments). Persisted in the replay body, injected into
                the kernel user message, frozen with the first segment's
                checkpoint. Never inherited by child runs and never part
                of summaries; ``None`` everywhere else.
            parent_task_id: Internal plan-task correlation for an agent
                child. Persisted only in the durable replay payload so the
                run store can project child progress onto the parent task.
            parent_task_attempt: Logical task attempt corresponding to the
                child. Independent of the run-row claim/fencing attempt.

        Raises:
            inqtrix.server.runs.RunQueueFull: When the waiting queue
                has no free slot (the router maps this to HTTP 429).
        """
        algorithm = self._registry.get(resolved.mode)
        normalized_source_policy = (
            source_policy
            if isinstance(source_policy, SourcePolicy)
            else SourcePolicy.model_validate(source_policy or {})
        )
        effective_agent_overrides = {
            **resolved.agent_overrides,
            **(
                {
                    "depth": resolved.agent_settings.depth,
                    # The selected Stufe travels with the run (worker
                    # replay, summaries, edit-path policy) exactly like
                    # depth. Only echoed when selected: a no-tier run
                    # replays byte-identically to today, and a tier run
                    # replays the CONSISTENT (tier, bridged depth) pair,
                    # which the override validator accepts.
                    **(
                        {"agent_tier": tier_value}
                        if (
                            tier_value := getattr(
                                resolved.agent_settings, "agent_tier", ""
                            )
                        )
                        else {}
                    ),
                }
                if kind in {"agent", "agent_child"}
                else {}
            ),
        }
        run_request = RunRequest(
            mode=resolved.mode,
            question=question,
            history=history,
            messages=messages,
            agent_overrides=effective_agent_overrides,
            knowledge_filters=resolved.knowledge_filters,
            autonomy=autonomy,
            session_id=session_id or "",
            document_id=document_id,
            response_form=response_form,
            skill_ids=tuple(skill_ids or ()),
            skill_revisions=dict(skill_revisions or {}),
            tool_directives=tuple(tool_directives or ()),
            source_policy=normalized_source_policy,
            web_recency=web_recency,
            execution_directive=execution_directive,
            canvas_context=canvas_context,
            report_requirement=report_requirement,
            attached_reports=tuple(attached_reports or ()),
        )

        def _work(handle: "RunHandle") -> None:
            effective_principal = handle.effective_principal(principal)
            has_scoped_actor = (
                effective_principal is not None
                and effective_principal.user_id is not None
            )

            def _check_authority() -> None:
                self._run_store.check_execution_authority(handle.run_id)
                if self._dependency_authorizer is None:
                    # Only reachable for a scoped actor (see the gate
                    # below): silently skipping the actor and pinned
                    # dependency probes would be a fallback, not a check.
                    raise AuthorizationRevoked(
                        "run execution has no dependency authorizer"
                    )
                self._dependency_authorizer.check(
                    run_request,
                    effective_principal,
                )

            execute_run_request(
                handle,
                algorithm=algorithm,
                run_request=run_request,
                resolved=resolved,
                runtime=self._runtime,
                principal=effective_principal,
                quota_service=self._quota_service,
                token_budget=token_budget,
                workspace_id=workspace_id,
                authority_check=(
                    _check_authority if has_scoped_actor else None
                ),
                answer_publisher=self._answer_publisher,
            )

        # Everything a worker process needs to re-resolve and execute
        # this run from the persisted row alone; the in-memory backend
        # ignores it (the work closure stays in-process).
        request_payload: dict[str, Any] = {
            "question": question,
            "history": history,
            "messages": messages,
            "body": {
                "mode": resolved.mode,
                "agent_overrides": effective_agent_overrides,
                "knowledge_filters": resolved.knowledge_filters,
                **({"autonomy": autonomy} if autonomy else {}),
                **({"session_id": session_id} if session_id else {}),
                **({"document_id": document_id} if document_id else {}),
                **(
                    {"response_form": response_form}
                    if response_form
                    else {}
                ),
                **(
                    {"token_budget": token_budget}
                    if token_budget
                    else {}
                ),
                **({"origin_key": origin_key} if origin_key else {}),
                **(
                    {"parent_task_id": parent_task_id}
                    if parent_task_id
                    else {}
                ),
                **(
                    {"parent_task_attempt": int(parent_task_attempt)}
                    if parent_task_attempt > 0
                    else {}
                ),
                **({"skill_ids": list(skill_ids)} if skill_ids else {}),
                **(
                    {"skill_revisions": dict(skill_revisions)}
                    if skill_revisions
                    else {}
                ),
                **(
                    {"tool_directives": list(tool_directives)}
                    if tool_directives
                    else {}
                ),
                **(
                    {
                        "source_policy": normalized_source_policy.model_dump(
                            mode="json"
                        )
                    }
                    if source_policy is not None
                    else {}
                ),
                **(
                    {"web_recency": web_recency}
                    if web_recency is not None
                    else {}
                ),
                **(
                    {"execution_directive": execution_directive}
                    if execution_directive
                    else {}
                ),
                **(
                    {
                        "canvas_context": canvas_context.model_dump(
                            mode="json"
                        )
                    }
                    if canvas_context is not None
                    else {}
                ),
                **(
                    {"report_requirement": report_requirement}
                    if report_requirement
                    else {}
                ),
                **(
                    {
                        "attached_reports": [
                            dict(report) for report in attached_reports
                        ]
                    }
                    if attached_reports
                    else {}
                ),
                **(
                    {"stack": resolved.stack_name}
                    if resolved.stack_name
                    else {}
                ),
            },
        }
        return self._run_store.submit(
            question=question,
            stack_name=resolved.stack_name or "default",
            work=_work,
            # One effective override contract feeds in-process execution,
            # worker replay, and summaries. The summary additionally carries
            # UI/session metadata that already travels as dedicated replay
            # fields.
            agent_overrides={
                **effective_agent_overrides,
                **({"autonomy": autonomy} if autonomy else {}),
                **(
                    {"response_form": response_form}
                    if response_form
                    else {}
                ),
                **(
                    {
                        "source_policy": normalized_source_policy.model_dump(
                            mode="json"
                        )
                    }
                    if source_policy is not None
                    else {}
                ),
                **(
                    {"execution_directive": execution_directive}
                    if execution_directive
                    else {}
                ),
            },
            mode=resolved.mode,
            workspace_id=workspace_id,
            created_by_user_id=principal.user_id if principal is not None else None,
            created_by_tenant_id=(
                principal.tenant_id if principal is not None else None
            ),
            execution_scopes=(
                principal.scopes if principal is not None else frozenset()
            ),
            request_payload=request_payload,
            kind=kind,
            parent_run_id=parent_run_id,
            root_run_id=root_run_id,
            session_id=session_id,
            origin_key=origin_key or None,
        )
