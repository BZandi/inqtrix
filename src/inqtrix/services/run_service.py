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
from inqtrix.core.results import RunRequest
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
        cancel_token=handle.cancel_event,
        event_sink=_event_sink,
        # Opt-in hard per-run token cap (0 = off). Independent of the
        # monthly token quota: it bounds a single run, the quota blocks
        # the next one.
        token_budget=runtime.settings.quota.max_tokens_per_run,
    )
    agent_result = algorithm.run(
        run_request,
        runtime=runtime,
        context=run_context,
    )
    result = agent_result.raw
    result_state = result.get("result_state", {}) or {}
    # Book the LLM tokens this run actually consumed before any early
    # return: a budget-cancelled or client-cancelled run still spent
    # what it spent, and that spend must count toward the monthly quota
    # (it is what blocks the NEXT submission).
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
    if handle.cancel_event.is_set() or agent_result.cancelled:
        handle.cancel("client_requested_cancel")
        return

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

        Raises:
            inqtrix.server.runs.RunQueueFull: When the waiting queue
                has no free slot (the router maps this to HTTP 429).
        """
        algorithm = self._registry.get(resolved.mode)
        run_request = RunRequest(
            mode=resolved.mode,
            question=question,
            history=history,
            messages=messages,
            agent_overrides=resolved.agent_overrides,
            knowledge_filters=resolved.knowledge_filters,
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
                "agent_overrides": resolved.agent_overrides,
                "knowledge_filters": resolved.knowledge_filters,
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
            agent_overrides=resolved.agent_overrides,
            mode=resolved.mode,
            workspace_id=workspace_id,
            created_by_sub=principal.sub if principal is not None else None,
            created_by_tenant_id=(
                principal.tenant_id if principal is not None else None
            ),
            request_payload=request_payload,
        )
