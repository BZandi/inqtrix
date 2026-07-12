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
from typing import TYPE_CHECKING, Any

from inqtrix.core.context import RunContext, RuntimeContext
from inqtrix.core.results import RunRequest, SourcePolicy, WebRecency
from inqtrix.execution_failures import RunExecutionFailure
from inqtrix.quota.models import QuotaDimension, QuotaSubject, consumed_tokens
from inqtrix.result import ResearchResult
from inqtrix.services.agent_context import ResolvedAgentContext
from inqtrix.state import build_run_snapshot

if TYPE_CHECKING:
    from inqtrix.auth.principal import Principal
    from inqtrix.core.algorithms import AgentAlgorithm, AlgorithmRegistry
    from inqtrix.server.runs import RunHandle, RunStore
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
            which carry the metered subject in *quota_subject* instead.
        quota_service: Optional usage meter. When wired (quotas on), the
            run's real LLM-token consumption is booked through its
            synchronous bridge AFTER execution (non-fatal — a recording
            failure never loses the run), and the per-run token budget
            aborts a runaway run at a node boundary. ``None`` for
            unmetered deployments — byte-identical to the historical
            path.
        quota_subject: Explicit metered subject for the worker path,
            reconstructed from the run row's persisted (sub, tenant).
            When ``None`` (the in-process path) the subject is derived
            from *principal*. This is what makes token accounting fire
            regardless of which process executes the run.
        workspace_id: Persisted project namespace of the native run. Agent
            algorithms pass it unchanged to delegated children; it is not an
            authorization decision.
    """
    t0 = time.monotonic()

    def _event_sink(event_type: str, payload: dict[str, Any]) -> None:
        handle.emit(event_type, payload)

    run_context = RunContext(
        providers=resolved.providers,
        strategies=resolved.strategies,
        agent_settings=resolved.agent_settings,
        principal=principal,
        run_id=handle.run_id,
        workspace_id=workspace_id,
        cancel_token=handle.cancel_event,
        event_sink=_event_sink,
        # Opt-in hard per-run token cap (0 = off). Independent of the
        # monthly token quota: it bounds a single run, the quota blocks
        # the next one. The optional narrowing comes only from trusted
        # server callers; planner-authored task budgets are never input.
        token_budget=_effective_budget(
            runtime.settings.quota.max_tokens_per_run, token_budget
        ),
        park=handle.wait,
    )
    agent_result = algorithm.run(
        run_request,
        runtime=runtime,
        context=run_context,
    )
    result = agent_result.raw
    result_state = result.get("result_state", {}) or {}
    # Book the LLM tokens this run actually consumed before any early
    # return: a budget-cancelled, client-cancelled, or PARKED run still
    # spent what it spent, and that spend must count toward the monthly
    # quota (it is what blocks the NEXT submission).
    if quota_service is not None:
        subject = (
            quota_subject
            if quota_subject is not None
            else quota_service.subject_for(principal)
        )
        quota_service.record_blocking(
            subject,
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
    research_result.metrics.elapsed_seconds = round(time.monotonic() - t0, 2)
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
    handle.emit_answer(answer)
    current_node = str(algorithm.capabilities().get("terminal_node", "answer"))
    handle.complete(
        payload,
        snapshot=build_run_snapshot(
            result_state,
            current_node=current_node,
            last_message="completed",
        ),
    )


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
    """

    def __init__(
        self,
        *,
        registry: "AlgorithmRegistry",
        runtime: RuntimeContext,
        run_store: "RunStore",
        quota_service: "QuotaService | None" = None,
    ) -> None:
        self._registry = registry
        self._runtime = runtime
        self._run_store = run_store
        self._quota_service = quota_service

    @property
    def run_store(self) -> "RunStore":
        """The underlying run store (read-side surface for the router)."""
        return self._run_store

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
        skill_revisions: dict[str, float] | None = None,
        tool_directives: list[str] | None = None,
        source_policy: dict[str, str] | SourcePolicy | None = None,
        web_recency: WebRecency | None = None,
        execution_directive: str = "",
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
            session_id: Agent-desk session grouping the run belongs to.
            autonomy: Workspace-agent permission mode (E16), carried on
                the run request and the worker replay payload; empty for
                non-agent modes.
            document_id: Target editor document of a workspace-agent
                patch assignment (M7); empty when the run has no edit
                target. Carried like ``autonomy``.
            response_form: Workspace-agent output-form override
                (plan M1, ``chat``/``canvas``); empty delegates to the
                intake profile. Carried like ``autonomy``.
            token_budget: Optional trusted per-run LLM-token ceiling that
                narrows the deployment cap. Planner-authored task budgets
                are compatibility-only data and never flow into this field.
            origin_key: Idempotency key of the submitting kernel tool call or
                mission plan-task attempt. Persisted in the replay body and
                exposed on child summaries so checkpoint re-execution finds
                the already-submitted run. Empty for every other caller.
            skill_ids: Router-ADMITTED skill ids (plan M3) — visibility
                and count cap already enforced; carried on the request
                and the worker replay payload.
            skill_revisions: Admission-time ``updated_at`` pin for each
                attached skill; runtime loading fails closed on drift.
            tool_directives: Whitelisted composer tool hints (plan M3),
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
        )

        def _work(handle: "RunHandle") -> None:
            execute_run_request(
                handle,
                algorithm=algorithm,
                run_request=run_request,
                resolved=resolved,
                runtime=self._runtime,
                principal=principal,
                quota_service=self._quota_service,
                token_budget=token_budget,
                workspace_id=workspace_id,
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
            created_by_sub=principal.sub if principal is not None else None,
            created_by_tenant_id=(
                principal.tenant_id if principal is not None else None
            ),
            request_payload=request_payload,
            kind=kind,
            parent_run_id=parent_run_id,
            root_run_id=root_run_id,
            session_id=session_id,
            origin_key=origin_key or None,
        )
