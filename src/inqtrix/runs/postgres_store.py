"""Durable run store: records, events, and results in Postgres.

Same public surface as the in-memory
:class:`~inqtrix.server.runs.RunStore` — the runs router and
:class:`~inqtrix.services.run_service.RunService` cannot tell the
backends apart. Two execution modes:

* ``queue is None`` (``INQTRIX_STORAGE_BACKEND=postgres`` alone):
  records, events, and results are durable; execution stays in this
  process with the same thread-dispatch semantics as the memory store.
* ``queue`` set (``INQTRIX_QUEUE_BACKEND=valkey``): accepted runs are
  persisted and dispatched to the job stream; ``inqtrix-worker``
  processes claim and execute them. The run row is the source of
  truth — a lost stream message is recoverable from Postgres, and the
  guarded status transitions make at-least-once delivery idempotent.

The storage layer is async (asyncpg) while this surface is sync (the
routers and the run-handle call it from worker threads); the store
therefore owns one background event loop and funnels every database
operation through it. Sequence numbers are allocated via
``UPDATE runs SET event_seq = event_seq + 1 RETURNING event_seq`` —
serialized on the row lock, gap-free across processes, which is what
keeps the SSE stream byte-compatible with the in-memory store.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Any, Mapping

from sqlalchemy import and_, delete, func, insert, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from inqtrix.auth.permissions import AuditEntry, SharePermission
from inqtrix.runs.durable_store import (
    DEFAULT_TENANT,
    DurableJobStoreBase,
    PollingJobSubscription,
    _LocalJob,
)
from inqtrix.runs.shared import (
    access_annotation as _access_annotation,
    build_run_summary,
    expand_run_event,
)
from inqtrix.runtime_logging import new_run_id
from inqtrix.server.runs import (
    TERMINAL_RUN_STATUSES,
    RunActive,
    RunHandle,
    RunNotFound,
    RunQueueFull,
    RunStatus,
    RunWork,
)
from inqtrix.storage.runs_orm import run_events, runs
from inqtrix.urls import sanitize_error

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from inqtrix.auth.permissions import AuditSink
    from inqtrix.auth.principal import UserContext
    from inqtrix.runs.valkey_queue import ValkeyRunQueue

log = logging.getLogger("inqtrix")

_TERMINAL_VALUES = tuple(status.value for status in TERMINAL_RUN_STATUSES)

_STUCK_ROW_MAX_AGE_SECONDS = 7 * 86_400.0
"""Hard retention cap for rows that never reached a terminal state —
request payloads carry user conversation content and must not live
forever just because a worker died mid-run."""

_TERMINAL_EVENT_TYPES = frozenset(
    {"inqtrix.run.completed", "inqtrix.run.failed", "inqtrix.run.cancelled"}
)


@dataclass(frozen=True)
class _RowView:
    """Flat view of one ``runs`` row for the shared summary builder."""

    run_id: str
    status: str
    question: str
    stack_name: str
    workspace_id: str | None
    mode: str
    agent_overrides: dict[str, Any]
    created_at: float
    started_at: float | None
    finished_at: float | None
    snapshot: dict[str, Any]
    error: dict[str, Any] | None


@dataclass(frozen=True)
class ClaimedRun:
    """Result of a successful worker claim on a queued run."""

    run_id: str
    tenant_id: str
    attempt: int
    request_payload: dict[str, Any]
    # Persisted attribution of the submitter, so the worker can meter
    # the run against the right quota subject without a live principal.
    created_by_sub: str | None = None
    created_by_tenant_id: str | None = None


class PostgresRunStore(DurableJobStoreBase):
    """Durable run registry with the in-memory store's public surface.

    Args:
        engine: Async engine OWNED by this store. asyncpg pools are
            event-loop-affine: every connection must live on the
            store's background loop, so the engine is deliberately NOT
            shared with the identity/file backends (which run on the
            HTTP server's loop) — sharing one pool across loops
            corrupts connections.
        app_role: Restricted database role assumed per transaction
            (``SET LOCAL ROLE``) so forced RLS applies.
        queue: Job queue for worker dispatch; ``None`` keeps execution
            in this process (durable records, unchanged threading).
        max_concurrent: In-process execution slots (no-queue mode) and
            part of the queue-saturation formula.
        max_queue_size: Waiting-run bound; exceeding it with all slots
            busy rejects submissions (HTTP 429 upstream).
        completed_ttl_seconds: Retention for terminal runs; lazy
            cleanup deletes older rows (events cascade).
        worker_id: Identity stamped into ``claimed_by`` for runs this
            process executes or fences.
        audit: Optional audit sink; visibility denials are persisted
            here in addition to the operator log line (closing the
            gap the in-memory store deliberately deferred to G6).

    Tenancy: run rows live in the single deployment tenant
    (``default``) at the RLS layer — exactly like the in-memory store,
    per-user visibility is the ``(created_by_sub, created_by_tenant_id)``
    predicate, never the row tenant. True multi-tenant run storage
    arrives with the OIDC rollout and gets its own decision.
    """

    _loop_thread_name = "inqtrix-runs-db"
    _dispatch_thread_prefix = "inqtrix-run"
    _job_kind = "Native run"

    def __init__(
        self,
        *,
        engine: "AsyncEngine",
        app_role: str,
        queue: "ValkeyRunQueue | None" = None,
        max_concurrent: int,
        max_queue_size: int,
        completed_ttl_seconds: int,
        worker_id: str,
        audit: "AuditSink | None" = None,
    ) -> None:
        # The engine/session/loop/dispatch plumbing lives in
        # DurableJobStoreBase; this store adds only its sizing,
        # retention, and audit state.
        super().__init__(
            engine=engine,
            app_role=app_role,
            worker_id=worker_id,
            queue=queue,
            max_concurrent=max_concurrent,
        )
        self._max_queue_size = max_queue_size
        self._completed_ttl_seconds = completed_ttl_seconds
        self._audit = audit

    # -- public surface (RunStore parity) -------------------------------- #

    def submit(
        self,
        *,
        question: str,
        stack_name: str,
        work: RunWork,
        agent_overrides: dict[str, Any] | None = None,
        mode: str = "research",
        workspace_id: str | None = None,
        created_by_sub: str | None = None,
        created_by_tenant_id: str | None = None,
        request_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one queued run, then dispatch locally or enqueue.

        Raises:
            RunQueueFull: When the waiting queue is full and every
                execution slot is busy (queue-mode counts are
                cluster-wide via the database).
        """
        summary = self._call(
            self._submit_db(
                tenant_id=DEFAULT_TENANT,
                question=question[:500],
                stack_name=stack_name,
                agent_overrides=dict(agent_overrides or {}),
                mode=mode,
                workspace_id=workspace_id,
                created_by_sub=created_by_sub,
                created_by_tenant_id=created_by_tenant_id,
                request_payload=request_payload,
            )
        )
        run_id = summary["run_id"]
        if self._queue is not None:
            try:
                self._queue.enqueue(
                    run_id=run_id, tenant_id=DEFAULT_TENANT
                )
            except Exception:  # noqa: BLE001 — row is committed; visible
                # The run row is the source of truth: the worker
                # reconciler re-dispatches stale queued rows, so a
                # broker blip must not turn an accepted run into a 500.
                log.warning(
                    "Dispatch-Nachricht fuer Run %s konnte nicht "
                    "gesendet werden — der Reconciler holt das nach.",
                    run_id,
                    exc_info=True,
                )
        else:
            with self._lock:
                self._local[run_id] = _LocalJob(work=work)
                self._pending.append(run_id)
                self._dispatch_locked()
        return summary

    def import_completed_run(
        self,
        *,
        run_id: str,
        question: str,
        stack_name: str,
        result: dict[str, Any],
        status: str = "completed",
        mode: str = "research",
        agent_overrides: dict[str, Any] | None = None,
        snapshot: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        created_at: float | None = None,
        workspace_id: str | None = None,
        created_by_sub: str | None = None,
        created_by_tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist an already-terminal imported run durably. See the port.

        Never queues or dispatches — the row is written terminal in one shot.

        Raises:
            ValueError: When *status* is not a terminal status.
        """
        if status not in _TERMINAL_VALUES:
            raise ValueError(
                f"import_completed_run requires a terminal status, got {status!r}"
            )
        return self._call(
            self._import_completed_run_db(
                tenant_id=DEFAULT_TENANT,
                run_id=run_id,
                question=question[:500],
                stack_name=stack_name,
                result=result,
                status=status,
                mode=mode,
                agent_overrides=dict(agent_overrides or {}),
                snapshot=dict(snapshot or {}),
                error=dict(error) if error else None,
                created_at=created_at,
                workspace_id=workspace_id,
                created_by_sub=created_by_sub,
                created_by_tenant_id=created_by_tenant_id,
            )
        )

    def get(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        """Return a public summary for *run_id*."""
        return self._call(
            self._summary_db(run_id, workspace_id, visible_to, also_visible)
        )

    def list(
        self,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[dict[str, Any]]:
        """Return public summaries, newest first."""
        return self._call(
            self._list_db(workspace_id, visible_to, also_visible)
        )

    def owner_sub(self, run_id: str) -> str | None:
        """The run's creator regardless of visibility (share layer)."""
        return self._call(self._owner_sub_db(run_id))

    async def _owner_sub_db(self, run_id: str) -> str | None:
        async with self._session("default") as session:
            row = (
                await session.execute(
                    select(runs.c.created_by_sub).where(
                        runs.c.run_id == run_id
                    )
                )
            ).first()
        return row[0] if row is not None else None

    async def _delete_db(
        self,
        run_id: str,
        workspace_id: str | None,
        requester_sub: str | None,
    ) -> None:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            row = (
                await session.execute(
                    select(runs).where(runs.c.run_id == run_id)
                )
            ).mappings().first()
            if row is None:
                raise RunNotFound(run_id)
            if (
                (
                    row["created_by_sub"] is not None
                    and row["created_by_sub"] != requester_sub
                )
                or not _workspace_matches_row(row, workspace_id)
            ):
                # Owner-only for runs with a recorded creator; a legacy run
                # (created_by_sub None) is gated by its workspace alone, so it
                # stays deletable rather than orphaned forever.
                log.warning(
                    "authz denied: run %s delete refused for sub=%s",
                    run_id,
                    requester_sub or "",
                )
                raise RunNotFound(run_id)
            if row["status"] not in {
                status.value for status in TERMINAL_RUN_STATUSES
            }:
                raise RunActive(run_id)
            await session.execute(
                delete(runs).where(runs.c.run_id == run_id)
            )

    def result(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        """Return the stored result payload for a completed run."""
        return self._call(
            self._result_db(run_id, workspace_id, visible_to, also_visible)
        )

    def cancel(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        """Request cancellation for a queued or running run.

        Queued runs transition to ``cancelled`` immediately (guarded —
        a concurrent worker claim wins or loses atomically); running
        runs get ``cancel_requested`` set, observed by the executing
        process at the next graph node boundary.
        """
        summary = self._call(
            self._cancel_db(run_id, workspace_id, visible_to, also_visible)
        )
        with self._lock:
            local = self._local.get(run_id)
            if local is not None:
                local.cancel_event.set()
                if summary["status"] == RunStatus.CANCELLED.value:
                    try:
                        self._pending.remove(run_id)
                    except ValueError:
                        pass
                    local.work = None
        return summary

    def delete(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        requester_sub: str | None = None,
    ) -> None:
        """Permanently remove one terminal run (owner-only).

        Mirrors the in-memory store: creator identity gates the delete (not
        share visibility), terminal-only; the ``run_events`` rows cascade
        with the parent (FK ``ondelete=CASCADE``). ``RunNotFound`` for
        unknown / non-owner / cross-namespace ids; ``RunActive`` for a
        still-active run.
        """
        self._call(self._delete_db(run_id, workspace_id, requester_sub))
        with self._lock:
            self._local.pop(run_id, None)
            try:
                self._pending.remove(run_id)
            except ValueError:
                pass

    def subscribe(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> PollingJobSubscription:
        """Subscribe to a run's event stream with full stored replay."""
        tenant_id, replay = self._call(
            self._replay_db(run_id, workspace_id, visible_to, also_visible)
        )
        return PollingJobSubscription(
            self,
            run_id,
            tenant_id,
            replay,
            terminal_events=_TERMINAL_EVENT_TYPES,
            thread_label="inqtrix-run-events",
        )

    def unsubscribe(self, run_id: str, queue: Queue) -> None:
        """Parity no-op: polling subscriptions detach via ``close()``."""

    def emit(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        fence_attempt: int | None = None,
    ) -> None:
        """Append one event (plus snapshot companion) to the run.

        With *fence_attempt* set (worker handles), events from a
        superseded attempt are dropped with a visible log instead of
        interleaving with the superseding attempt's stream.
        """
        self._call(
            self._emit_db(
                run_id, event_type, payload or {}, fence_attempt
            )
        )

    def complete(
        self,
        run_id: str,
        result: dict[str, Any],
        *,
        snapshot: dict[str, Any] | None = None,
        fence_attempt: int | None = None,
    ) -> bool:
        """Store the final result and mark the run completed.

        Returns:
            ``True`` when the terminal transition landed; ``False``
            when it was absorbed (already terminal or fenced out by a
            superseding claim) — the caller must NOT ack the dispatch
            message in that case.
        """
        return self._call(
            self._terminal_db(
                run_id,
                RunStatus.COMPLETED,
                fence_attempt=fence_attempt,
                result=dict(result),
                snapshot=dict(snapshot) if snapshot else None,
                event_builder=lambda row_snapshot: (
                    "inqtrix.run.completed",
                    {
                        "status": "completed",
                        "metrics": (
                            result.get("metrics")
                            if isinstance(result.get("metrics"), dict)
                            else {}
                        ),
                        "result_url": f"/v1/runs/{run_id}/result",
                        "snapshot": row_snapshot,
                    },
                ),
            )
        )

    def fail(
        self,
        run_id: str,
        message: str,
        *,
        error_type: str = "server_error",
        fence_attempt: int | None = None,
    ) -> bool:
        """Mark a run failed with a sanitized error payload.

        Returns:
            ``True`` when the transition landed (see :meth:`complete`).
        """
        error = {"message": sanitize_error(message), "type": error_type}
        return self._call(
            self._terminal_db(
                run_id,
                RunStatus.FAILED,
                fence_attempt=fence_attempt,
                error=error,
                event_builder=lambda row_snapshot: (
                    "inqtrix.run.failed",
                    {
                        "status": "failed",
                        "error": error,
                        "snapshot": row_snapshot,
                    },
                ),
            )
        )

    def mark_cancelled(
        self,
        run_id: str,
        *,
        reason: str,
        fence_attempt: int | None = None,
    ) -> bool:
        """Mark a running run cancelled after its worker observed it.

        Returns:
            ``True`` when the transition landed (see :meth:`complete`).
        """
        return self._call(
            self._terminal_db(
                run_id,
                RunStatus.CANCELLED,
                fence_attempt=fence_attempt,
                event_builder=lambda row_snapshot: (
                    "inqtrix.run.cancelled",
                    {
                        "status": "cancelled",
                        "reason": reason,
                        "snapshot": row_snapshot,
                    },
                ),
            )
        )

    # -- worker surface --------------------------------------------------- #

    def claim_for_execution(
        self,
        run_id: str,
        tenant_id: str,
        *,
        allow_takeover: bool,
    ) -> ClaimedRun | None:
        """Atomically claim one run for execution.

        Args:
            run_id: Run to claim.
            tenant_id: Tenant carried in the dispatch message.
            allow_takeover: Permit claiming a run already marked
                ``running`` — only correct for redeliveries (the
                previous worker stopped heartbeating); fresh duplicate
                messages must NOT take over a healthy execution.

        Returns:
            The claim with its fencing ``attempt``, or ``None`` when
            the run is terminal, missing, or running elsewhere.
        """
        return self._call(
            self._claim_db(run_id, tenant_id, allow_takeover=allow_takeover)
        )

    def cancel_requested_runs(
        self, run_ids: dict[str, str]
    ) -> set[str]:
        """Subset of ``run_ids`` (id -> tenant) with a pending cancel."""
        return self._call(self._cancel_requested_db(run_ids))

    def stale_queued_runs(self, *, older_than_seconds: float) -> list[tuple[str, str]]:
        """Queued ``(run_id, tenant_id)`` pairs older than the threshold.

        Re-enqueue feed for the worker's reconciler: a crash between
        the run-row insert and the stream ``XADD`` leaves the row
        queued with no message; re-dispatching is safe because fresh
        duplicates never take over a healthy execution.
        """
        return self._call(self._stale_queued_db(older_than_seconds))

    # -- in-process execution hooks (no-queue mode) ---------------------- #

    def _make_handle(self, run_id: str, cancel_event) -> RunHandle:
        return RunHandle(self, run_id, cancel_event)

    def _auto_complete(self, run_id: str) -> None:
        self._call(
            self._terminal_db(
                run_id,
                RunStatus.COMPLETED,
                fence_attempt=None,
                warn_on_noop=False,
                event_builder=lambda row_snapshot: (
                    "inqtrix.run.completed",
                    {"status": "completed", "snapshot": row_snapshot},
                ),
            )
        )

    # -- subscription poll bridge ----------------------------------------- #

    def _events_after(
        self, run_id: str, tenant_id: str, after_sequence: int
    ) -> list[dict[str, Any]]:
        return self._call(
            self._events_after_db(run_id, tenant_id, after_sequence)
        )

    # -- async database operations ----------------------------------------- #

    async def _submit_db(self, *, tenant_id: str, **fields: Any) -> dict[str, Any]:
        async with self._session(tenant_id) as session:
            await self._cleanup_db(session)
            queued = await session.scalar(
                select(func.count())
                .select_from(runs)
                .where(runs.c.status == RunStatus.QUEUED.value)
            )
            running = await session.scalar(
                select(func.count())
                .select_from(runs)
                .where(runs.c.status == RunStatus.RUNNING.value)
            )
            if (
                (queued or 0) >= self._max_queue_size
                and (running or 0) >= self._max_concurrent
            ):
                raise RunQueueFull("native run queue is full")

            created_at = time.time()
            run_id = await self._insert_with_unique_id(
                session, tenant_id=tenant_id, created_at=created_at, **fields
            )
            position = await self._queue_position_db(session, created_at)
            await self._append_events_db(
                session,
                run_id,
                tenant_id,
                expand_run_event(
                    "inqtrix.run.queued",
                    {"status": "queued", "queue_position": position},
                    status=RunStatus.QUEUED.value,
                )[1],
            )
            row = await self._row_db(session, run_id)
            return build_run_summary(row, queue_position=position)

    async def _import_completed_run_db(
        self,
        *,
        tenant_id: str,
        run_id: str,
        question: str,
        stack_name: str,
        result: dict[str, Any],
        status: str,
        mode: str,
        agent_overrides: dict[str, Any],
        snapshot: dict[str, Any],
        error: dict[str, Any] | None,
        created_at: float | None,
        workspace_id: str | None,
        created_by_sub: str | None,
        created_by_tenant_id: str | None,
    ) -> dict[str, Any]:
        now = time.time()
        values: dict[str, Any] = {
            "tenant_id": tenant_id,
            "status": status,
            "question": question,
            "stack_name": stack_name,
            "workspace_id": workspace_id,
            "mode": mode,
            "agent_overrides": agent_overrides,
            "created_by_sub": created_by_sub,
            "created_by_tenant_id": created_by_tenant_id,
            # created_at keeps the report's original date (display/ordering);
            # started_at/finished_at = now so the durable-retention clock starts
            # at import -- an old report is not pruned on the next cleanup.
            "created_at": created_at if created_at is not None else now,
            "started_at": now,
            "finished_at": now,
            "snapshot": snapshot,
            # Only a completed run carries a result payload; failed/cancelled
            # imports keep the error, matching the live terminal transitions.
            "result": result if status == RunStatus.COMPLETED.value else None,
            "error": error,
        }
        async with self._session(tenant_id) as session:
            await self._cleanup_db(session)
            landed = (
                await session.execute(
                    pg_insert(runs)
                    .values(run_id=run_id, **values)
                    .on_conflict_do_nothing(index_elements=["run_id"])
                    .returning(runs.c.run_id)
                )
            ).scalar_one_or_none()
            if landed is None:
                # The id is taken. If it is the caller's OWN run, the snapshot
                # is immutable -> idempotent return. If another principal owns
                # it, never overwrite/leak it: allocate a fresh id instead.
                owner = (
                    await session.execute(
                        select(
                            runs.c.created_by_sub, runs.c.created_by_tenant_id
                        ).where(runs.c.run_id == run_id)
                    )
                ).first()
                if (
                    owner is not None
                    and owner.created_by_sub == created_by_sub
                    and owner.created_by_tenant_id == created_by_tenant_id
                ):
                    row = await self._row_db(session, run_id)
                    return build_run_summary(row, queue_position=None)
                log.warning(
                    "Imported run id %s already owned by another principal; "
                    "allocating a new id.",
                    run_id,
                )
                run_id = await self._insert_terminal_with_unique_id(
                    session, values=values
                )
            row = await self._row_db(session, run_id)
            return build_run_summary(row, queue_position=None)

    async def _insert_terminal_with_unique_id(
        self, session: "AsyncSession", *, values: dict[str, Any]
    ) -> str:
        for _ in range(8):
            run_id = new_run_id()
            landed = (
                await session.execute(
                    pg_insert(runs)
                    .values(run_id=run_id, **values)
                    .on_conflict_do_nothing(index_elements=["run_id"])
                    .returning(runs.c.run_id)
                )
            ).scalar_one_or_none()
            if landed is not None:
                return run_id
            log.warning("Native run id collision detected; retrying allocation.")
        raise RuntimeError("could not allocate a unique native run id")

    async def _insert_with_unique_id(
        self,
        session: "AsyncSession",
        *,
        tenant_id: str,
        created_at: float,
        **fields: Any,
    ) -> str:
        for _ in range(8):
            run_id = new_run_id()
            result = await session.execute(
                pg_insert(runs)
                .values(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    status=RunStatus.QUEUED.value,
                    question=fields["question"],
                    stack_name=fields["stack_name"],
                    workspace_id=fields["workspace_id"],
                    mode=fields["mode"],
                    agent_overrides=fields["agent_overrides"],
                    request_payload=fields["request_payload"],
                    created_by_sub=fields["created_by_sub"],
                    created_by_tenant_id=fields["created_by_tenant_id"],
                    created_at=created_at,
                )
                .on_conflict_do_nothing(index_elements=["run_id"])
                .returning(runs.c.run_id)
            )
            if result.scalar_one_or_none() is not None:
                return run_id
            log.warning("Native run id collision detected; retrying allocation.")
        raise RuntimeError("could not allocate a unique native run id")

    async def _queue_position_db(
        self, session: "AsyncSession", created_at: float
    ) -> int:
        earlier = await session.scalar(
            select(func.count())
            .select_from(runs)
            .where(
                runs.c.status == RunStatus.QUEUED.value,
                runs.c.created_at < created_at,
            )
        )
        return int(earlier or 0) + 1

    async def _row_db(
        self, session: "AsyncSession", run_id: str
    ) -> _RowView:
        row = (
            await session.execute(select(runs).where(runs.c.run_id == run_id))
        ).mappings().first()
        if row is None:
            raise RunNotFound(run_id)
        return _row_view(row)

    async def _visible_row_db(
        self,
        session: "AsyncSession",
        run_id: str,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[Any, "SharePermission | None"]:
        """The row plus the share grant that admitted it (if any).

        Shared-in runs bypass the workspace namespace filter — they
        carry the grantor's workspace id.
        """
        row = (
            await session.execute(select(runs).where(runs.c.run_id == run_id))
        ).mappings().first()
        if row is None:
            raise RunNotFound(run_id)
        if _visible_row(row, visible_to):
            if not _workspace_matches_row(row, workspace_id):
                raise RunNotFound(run_id)
            return row, None
        shared = (
            also_visible.get(run_id) if also_visible is not None else None
        )
        if shared is not None:
            return row, shared
        log.warning(
            "authz denied: run %s hidden from sub=%s tenant=%s",
            run_id,
            visible_to.principal.sub if visible_to else "",
            visible_to.principal.tenant_id if visible_to else "",
        )
        if self._audit is not None and visible_to is not None:
            await self._audit.record(
                AuditEntry(
                    tenant_id=visible_to.principal.tenant_id,
                    actor_sub=visible_to.principal.sub,
                    action="authz.denied",
                    resource_type="run",
                    resource_id=run_id,
                    detail={"surface": "runs"},
                )
            )
        raise RunNotFound(run_id)

    async def _summary_db(
        self,
        run_id: str,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            row, shared = await self._visible_row_db(
                session, run_id, workspace_id, visible_to, also_visible
            )
            position = (
                await self._queue_position_db(session, row["created_at"])
                if row["status"] == RunStatus.QUEUED.value
                else None
            )
            return build_run_summary(
                _row_view(row),
                queue_position=position,
                access=_access_annotation(shared),
            )

    async def _list_db(
        self,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[dict[str, Any]]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            query = select(runs).order_by(runs.c.created_at.desc())
            shared_ids = list(also_visible) if also_visible else []
            if visible_to is not None:
                owned = and_(
                    runs.c.created_by_sub == visible_to.principal.sub,
                    runs.c.created_by_tenant_id
                    == visible_to.principal.tenant_id,
                )
                if workspace_id is not None:
                    owned = and_(
                        owned, runs.c.workspace_id == workspace_id
                    )
                if shared_ids:
                    # Shared-in rows join REGARDLESS of the workspace
                    # namespace filter (they carry the grantor's id).
                    query = query.where(
                        or_(owned, runs.c.run_id.in_(shared_ids))
                    )
                else:
                    query = query.where(owned)
            elif workspace_id is not None:
                query = query.where(runs.c.workspace_id == workspace_id)
            rows = (await session.execute(query)).mappings().all()
            summaries = []
            for row in rows:
                position = (
                    await self._queue_position_db(session, row["created_at"])
                    if row["status"] == RunStatus.QUEUED.value
                    else None
                )
                shared = (
                    also_visible.get(row["run_id"])
                    if also_visible is not None
                    and not _visible_row(row, visible_to)
                    else None
                )
                summaries.append(
                    build_run_summary(
                        _row_view(row),
                        queue_position=position,
                        access=_access_annotation(shared),
                    )
                )
            return summaries

    async def _result_db(
        self,
        run_id: str,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            row, _shared = await self._visible_row_db(
                session, run_id, workspace_id, visible_to, also_visible
            )
            if row["result"] is None:
                raise RunNotFound(run_id)
            return {
                "run_id": run_id,
                "status": row["status"],
                **row["result"],
            }

    async def _cancel_db(
        self,
        run_id: str,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            row, shared = await self._visible_row_db(
                session, run_id, workspace_id, visible_to, also_visible
            )
            if shared is not None and not shared.at_least(
                SharePermission.EDIT
            ):
                # A view-only invitee watching a run must not be able
                # to kill it; the denial is the indistinct 404.
                raise RunNotFound(run_id)
            status = row["status"]
            if status == RunStatus.QUEUED.value:
                cancelled = (
                    await session.execute(
                        update(runs)
                        .where(
                            runs.c.run_id == run_id,
                            runs.c.status == RunStatus.QUEUED.value,
                        )
                        .values(
                            status=RunStatus.CANCELLED.value,
                            cancel_requested=True,
                            finished_at=time.time(),
                        )
                        .returning(runs.c.run_id)
                    )
                ).scalar_one_or_none()
                if cancelled is not None:
                    await self._append_events_db(
                        session,
                        run_id,
                        row["tenant_id"],
                        expand_run_event(
                            "inqtrix.run.cancelled",
                            {
                                "status": "cancelled",
                                "reason": "cancelled_before_start",
                            },
                            status=RunStatus.CANCELLED.value,
                        )[1],
                    )
                else:
                    # Lost the CAS to a concurrent claim: the run is
                    # running now — degrade to the two-phase cancel
                    # request instead of returning a stale summary
                    # that cancelled nothing.
                    status = RunStatus.RUNNING.value
            if status == RunStatus.RUNNING.value:
                # Guarded like the terminal writes: the row may have
                # gone terminal between the read and this statement —
                # a cancel_requested event AFTER the terminal event
                # would corrupt the log's ends-terminal invariant.
                requested = (
                    await session.execute(
                        update(runs)
                        .where(
                            runs.c.run_id == run_id,
                            runs.c.status == RunStatus.RUNNING.value,
                        )
                        .values(cancel_requested=True)
                        .returning(runs.c.run_id)
                    )
                ).scalar_one_or_none()
                if requested is not None:
                    await self._append_events_db(
                        session,
                        run_id,
                        row["tenant_id"],
                        expand_run_event(
                            "inqtrix.run.cancel_requested",
                            {
                                "status": "running",
                                "reason": "client_requested_cancel",
                            },
                            status=RunStatus.RUNNING.value,
                        )[1],
                    )
            fresh = await self._row_db(session, run_id)
            position = (
                await self._queue_position_db(session, fresh.created_at)
                if fresh.status == RunStatus.QUEUED.value
                else None
            )
            return build_run_summary(fresh, queue_position=position)

    async def _replay_db(
        self,
        run_id: str,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            row, _shared = await self._visible_row_db(
                session, run_id, workspace_id, visible_to, also_visible
            )
            events = await self._fetch_events(session, run_id, 0)
            return row["tenant_id"], events

    async def _events_after_db(
        self, run_id: str, tenant_id: str, after_sequence: int
    ) -> list[dict[str, Any]]:
        async with self._session(tenant_id) as session:
            return await self._fetch_events(session, run_id, after_sequence)

    async def _fetch_events(
        self, session: "AsyncSession", run_id: str, after_sequence: int
    ) -> list[dict[str, Any]]:
        rows = (
            await session.execute(
                select(run_events)
                .where(
                    run_events.c.run_id == run_id,
                    run_events.c.sequence > after_sequence,
                )
                .order_by(run_events.c.sequence)
            )
        ).mappings().all()
        return [
            {
                "type": row["type"],
                "run_id": run_id,
                "sequence": row["sequence"],
                "created_at": row["created_at"],
                "data": row["data"],
            }
            for row in rows
        ]

    async def _emit_db(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        fence_attempt: int | None = None,
    ) -> None:
        async with self._session(DEFAULT_TENANT) as session:
            if fence_attempt is not None:
                # Atomic fence: the sequence allocation itself carries
                # the ownership condition, so a takeover between a
                # check and the write cannot interleave zombie events.
                fenced_seq = (
                    await session.execute(
                        update(runs)
                        .where(
                            runs.c.run_id == run_id,
                            runs.c.claimed_by == self._worker_id,
                            runs.c.attempt == fence_attempt,
                        )
                        .values(event_seq=runs.c.event_seq + 0)
                        .returning(runs.c.event_seq)
                    )
                ).scalar_one_or_none()
                if fenced_seq is None:
                    log.warning(
                        "Event %s fuer Run %s verworfen — der Lauf "
                        "gehoert inzwischen einem anderen Worker-Versuch.",
                        event_type,
                        run_id,
                    )
                    return
            row = await self._row_db(session, run_id)
            new_snapshot, events = expand_run_event(
                event_type, payload, status=row.status
            )
            if new_snapshot is not None:
                await session.execute(
                    update(runs)
                    .where(runs.c.run_id == run_id)
                    .values(snapshot=new_snapshot)
                )
            await self._append_events_db(
                session, run_id, DEFAULT_TENANT, events
            )

    async def _append_events_db(
        self,
        session: "AsyncSession",
        run_id: str,
        tenant_id: str,
        events: list[tuple[str, dict[str, Any]]],
    ) -> None:
        for event_type, data in events:
            sequence = (
                await session.execute(
                    update(runs)
                    .where(runs.c.run_id == run_id)
                    .values(event_seq=runs.c.event_seq + 1)
                    .returning(runs.c.event_seq)
                )
            ).scalar_one()
            await session.execute(
                insert(run_events).values(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    sequence=sequence,
                    type=event_type,
                    created_at=time.time(),
                    data=data,
                )
            )

    async def _terminal_db(
        self,
        run_id: str,
        status: RunStatus,
        *,
        fence_attempt: int | None,
        event_builder,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        snapshot: dict[str, Any] | None = None,
        warn_on_noop: bool = True,
    ) -> bool:
        async with self._session(DEFAULT_TENANT) as session:
            values: dict[str, Any] = {
                "status": status.value,
                "finished_at": time.time(),
            }
            if result is not None:
                values["result"] = result
            if error is not None:
                values["error"] = error
            if snapshot is not None:
                values["snapshot"] = snapshot
            query = (
                update(runs)
                .where(
                    runs.c.run_id == run_id,
                    runs.c.status.notin_(_TERMINAL_VALUES),
                )
                .values(**values)
                .returning(runs.c.snapshot, runs.c.tenant_id)
            )
            if fence_attempt is not None:
                query = query.where(
                    runs.c.claimed_by == self._worker_id,
                    runs.c.attempt == fence_attempt,
                )
            row = (await session.execute(query)).first()
            if row is None:
                # Already terminal, missing, or fenced out (a reclaimed
                # zombie) — absorbing states stay absorbing. The
                # discarded write is operator-visible, EXCEPT for the
                # auto-complete safety net whose no-op is the expected
                # happy path (the work already completed itself).
                if warn_on_noop:
                    log.warning(
                        "Terminal-Schreibvorgang fuer Run %s verworfen "
                        "(bereits terminal oder von einem anderen Worker "
                        "uebernommen).",
                        run_id,
                    )
                return False
            event_type, payload = event_builder(dict(row[0] or {}))
            _, events = expand_run_event(
                event_type, payload, status=status.value
            )
            await self._append_events_db(session, run_id, row[1], events)
            return True

    async def _claim_db(
        self, run_id: str, tenant_id: str, *, allow_takeover: bool
    ) -> ClaimedRun | None:
        async with self._session(tenant_id) as session:
            allowed = [RunStatus.QUEUED.value]
            if allow_takeover:
                allowed.append(RunStatus.RUNNING.value)
            row = (
                await session.execute(
                    update(runs)
                    .where(
                        runs.c.run_id == run_id,
                        runs.c.status.in_(allowed),
                    )
                    .values(
                        status=RunStatus.RUNNING.value,
                        claimed_by=self._worker_id,
                        attempt=runs.c.attempt + 1,
                        started_at=time.time(),
                    )
                    .returning(
                        runs.c.attempt,
                        runs.c.request_payload,
                        runs.c.snapshot,
                        runs.c.created_by_sub,
                        runs.c.created_by_tenant_id,
                    )
                )
            ).first()
            if row is None:
                return None
            await self._append_events_db(
                session,
                run_id,
                tenant_id,
                expand_run_event(
                    "inqtrix.run.started",
                    {"status": "running", "snapshot": dict(row[2] or {})},
                    status=RunStatus.RUNNING.value,
                )[1],
            )
            return ClaimedRun(
                run_id=run_id,
                tenant_id=tenant_id,
                attempt=int(row[0]),
                request_payload=dict(row[1] or {}),
                created_by_sub=row[3],
                created_by_tenant_id=row[4],
            )

    async def _cancel_requested_db(
        self, run_ids: dict[str, str]
    ) -> set[str]:
        if not run_ids:
            return set()
        async with self._session(DEFAULT_TENANT) as session:
            rows = (
                await session.execute(
                    select(runs.c.run_id).where(
                        runs.c.run_id.in_(list(run_ids)),
                        runs.c.cancel_requested.is_(True),
                    )
                )
            ).scalars().all()
            return set(rows)

    async def _stale_queued_db(
        self, older_than_seconds: float
    ) -> list[tuple[str, str]]:
        async with self._session(DEFAULT_TENANT) as session:
            rows = (
                await session.execute(
                    select(runs.c.run_id, runs.c.tenant_id).where(
                        runs.c.status == RunStatus.QUEUED.value,
                        runs.c.created_at < time.time() - older_than_seconds,
                    )
                )
            ).all()
            return [(row[0], row[1]) for row in rows]

    async def _cleanup_db(self, session: "AsyncSession") -> None:
        if self._sweep_orphans:
            self._sweep_orphans = False
            await self._recover_orphans_db(session)
        await session.execute(
            delete(runs).where(
                runs.c.status.in_(_TERMINAL_VALUES),
                runs.c.finished_at.isnot(None),
                runs.c.finished_at
                < time.time() - self._completed_ttl_seconds,
            )
        )
        # Retention failsafe: rows stuck non-terminal (dead worker,
        # lost dispatch, operator error) must not retain request
        # payloads — user conversation content — indefinitely.
        stuck = (
            (
                await session.execute(
                    delete(runs)
                    .where(
                        runs.c.status.notin_(_TERMINAL_VALUES),
                        runs.c.created_at
                        < time.time() - _STUCK_ROW_MAX_AGE_SECONDS,
                    )
                    .returning(runs.c.run_id)
                )
            )
            .scalars()
            .all()
        )
        if stuck:
            log.warning(
                "%d Run-Zeilen nach %d Tagen ohne Abschluss geloescht: %s",
                len(stuck),
                int(_STUCK_ROW_MAX_AGE_SECONDS // 86_400),
                ", ".join(stuck[:5]),
            )

    async def _recover_orphans_db(self, session: "AsyncSession") -> None:
        """Fail queued/running rows left behind by a previous process.

        Only meaningful in no-queue mode: in-process execution cannot
        survive a restart (the work closures are gone), so the rows
        would otherwise count against admission capacity forever and
        present as eternally running to clients. Queue mode never
        sweeps — workers own those rows. Runs inside the caller's
        transaction (the first lazy cleanup).

        Assumes a SINGLE API process in no-queue durable mode (the
        documented deployment shape): a second process sharing the
        database would have its in-flight runs swept here. Multi
        replica deployments use the queue backend.
        """
        error = {
            "message": "Ein Server-Neustart hat den Lauf unterbrochen.",
            "type": "server_restarted",
        }
        rows = (
            await session.execute(
                update(runs)
                .where(
                    runs.c.status.in_(
                        (
                            RunStatus.QUEUED.value,
                            RunStatus.RUNNING.value,
                        )
                    )
                )
                .values(
                    status=RunStatus.FAILED.value,
                    finished_at=time.time(),
                    error=error,
                )
                .returning(runs.c.run_id, runs.c.tenant_id)
            )
        ).all()
        for run_id, tenant_id in rows:
            log.warning(
                "Verwaister Run %s nach Neustart als fehlgeschlagen "
                "markiert.",
                run_id,
            )
            await self._append_events_db(
                session,
                run_id,
                tenant_id,
                expand_run_event(
                    "inqtrix.run.failed",
                    {
                        "status": "failed",
                        "error": error,
                        "snapshot": {},
                    },
                    status=RunStatus.FAILED.value,
                )[1],
            )

def _row_view(row: Any) -> _RowView:
    return _RowView(
        run_id=row["run_id"],
        status=row["status"],
        question=row["question"],
        stack_name=row["stack_name"],
        workspace_id=row["workspace_id"],
        mode=row["mode"],
        agent_overrides=dict(row["agent_overrides"] or {}),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        snapshot=dict(row["snapshot"] or {}),
        error=dict(row["error"]) if row["error"] else None,
    )


def _workspace_matches_row(row: Any, workspace_id: str | None) -> bool:
    return workspace_id is None or row["workspace_id"] == workspace_id


def _visible_row(row: Any, visible_to: "UserContext | None") -> bool:
    """SQL-row twin of the in-memory visibility predicate."""
    if visible_to is None:
        return True
    return (
        row["created_by_sub"] is not None
        and row["created_by_sub"] == visible_to.principal.sub
        and row["created_by_tenant_id"] == visible_to.principal.tenant_id
    )


