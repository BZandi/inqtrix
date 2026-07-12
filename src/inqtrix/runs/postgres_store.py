"""Durable run store: records, events, and results in Postgres.

Same public surface as the in-memory
:class:`~inqtrix.server.runs.RunStore` — the runs router and
:class:`~inqtrix.services.run_service.RunService` cannot tell the
backends apart. Two execution modes:

* ``queue`` and ``dispatch_queue`` are both ``None``
  (``INQTRIX_STORAGE_BACKEND=postgres`` alone):
  records, events, and results are durable; execution stays in this
  process with the same thread-dispatch semantics as the memory store.
* an external dispatch queue set (``INQTRIX_QUEUE_BACKEND=valkey``): accepted runs are
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
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue, SimpleQueue
from typing import TYPE_CHECKING, Any, Mapping

from sqlalchemy import (
    and_,
    delete,
    event,
    func,
    insert,
    or_,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, IntegrityError

from inqtrix.auth.permissions import AuditEntry, SharePermission
from inqtrix.execution_failures import terminate_native_run
from inqtrix.runs.durable_store import (
    DEFAULT_TENANT,
    DurableJobStoreBase,
    PollingJobSubscription,
    _LocalJob,
)
from inqtrix.pagination import encode_cursor
from inqtrix.runs.ports import RunStoreMetrics
from inqtrix.runs.shared import (
    CHILD_PROGRESS_EVENT,
    access_annotation as _access_annotation,
    build_child_progress_payload,
    build_run_summary,
    expand_run_event,
    should_project_child_event,
)
from inqtrix.runtime_logging import new_run_id
from inqtrix.server.runs import (
    TERMINAL_RUN_STATUSES,
    WAITING_RUN_STATUSES,
    RunActive,
    RunHandle,
    RunNotFound,
    RunParentInactive,
    RunPerUserLimit,
    RunQueueFull,
    RunSessionActive,
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

_WAITING_VALUES = tuple(status.value for status in WAITING_RUN_STATUSES)

_ACTIVE_AGENT_SESSION_CONSTRAINT = "uq_runs_active_agent_session"

_STUCK_ROW_MAX_AGE_SECONDS = 7 * 86_400.0
"""Hard retention cap for rows that never reached a terminal state —
request payloads carry user conversation content and must not live
forever just because a worker died mid-run."""

_CANCEL_DEADLOCK_RETRIES = 3
"""Attempts for the synchronous cancel transaction before a Postgres
deadlock (SQLSTATE 40P01) is surfaced. The cancel cascade and a concurrent
child terminal write take the parent+child row locks in opposite orders, so
a rare abort is expected; the terminal side is queue-redelivered but cancel
is a request with no redelivery, so it retries the whole transaction here."""

_CANCEL_DEADLOCK_BACKOFF_SECONDS = 0.05
"""Base linear backoff between cancel deadlock retries (scaled by attempt).
Short by design — the contending write commits in milliseconds, so the
lock the retry needs frees almost immediately."""

_TERMINAL_EVENT_TYPES = frozenset(
    {"inqtrix.run.completed", "inqtrix.run.failed", "inqtrix.run.cancelled"}
)

_CLEANUP_HANDOFF_KEY = "inqtrix_run_cleanup_handoffs"


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
    kind: str = "standard"
    parent_run_id: str | None = None
    root_run_id: str | None = None
    session_id: str | None = None
    origin_key: str | None = None


@dataclass(frozen=True)
class ClaimedRun:
    """Result of a successful worker claim on a queued run."""

    run_id: str
    tenant_id: str
    attempt: int
    request_payload: dict[str, Any]
    workspace_id: str | None = None
    kind: str = "standard"
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
            ownership in this process. The claim-mode worker deliberately
            leaves this ``None`` because it claims through
            :class:`WorkerLoop`.
        dispatch_queue: Optional external queue used only for submissions,
            resumes, and child-triggered parent wakes. Defaults to *queue*;
            the claim-mode worker passes Valkey here while keeping *queue*
            ``None`` so ownership/fencing semantics do not change.
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
        recover_orphans: Whether this instance may blanket-fail
            queued/running rows left by a previous process. ``None``
            infers from ``queue`` (no-queue single API process sweeps,
            queue mode never); the queue-backed WORKER passes an
            explicit ``False`` because its ``queue=None`` is claim-mode
            wiring and stream reclaim owns crash recovery there.

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
        dispatch_queue: "ValkeyRunQueue | None" = None,
        max_concurrent: int,
        max_queue_size: int,
        completed_ttl_seconds: int,
        worker_id: str,
        audit: "AuditSink | None" = None,
        waiting_ttl_seconds: float = 7 * 24 * 3600.0,
        max_concurrent_per_user: int | None = None,
        recover_orphans: bool | None = None,
    ) -> None:
        # Validate BEFORE super().__init__ spawns the background loop
        # thread and adopts the engine — a config error must not leak
        # either.
        if float(waiting_ttl_seconds) <= 0:
            raise ValueError(
                f"waiting_ttl_seconds must be > 0, got {waiting_ttl_seconds}"
            )
        # The engine/session/loop/dispatch plumbing lives in
        # DurableJobStoreBase; this store adds only its sizing,
        # retention, and audit state.
        super().__init__(
            engine=engine,
            app_role=app_role,
            worker_id=worker_id,
            queue=queue,
            max_concurrent=max_concurrent,
            recover_orphans=recover_orphans,
        )
        self._dispatch_queue = (
            dispatch_queue if dispatch_queue is not None else queue
        )
        self._max_queue_size = max_queue_size
        self._completed_ttl_seconds = completed_ttl_seconds
        self._audit = audit
        self._waiting_ttl_seconds = float(waiting_ttl_seconds)
        self._max_concurrent_per_user = max_concurrent_per_user
        # TTL-swept waiting run ids, handed over from the cleanup
        # coroutine (which must never take self._lock — see
        # _cleanup_db) to the sync surface, which releases their
        # retained no-queue closures in _release_swept_locals().
        self._swept_waiting: SimpleQueue[str] = SimpleQueue()
        # Parents flipped waiting_for_children -> queued by a child's
        # terminal write (or the park-time self-heal). Same handoff
        # pattern as _swept_waiting: the coroutine only records the id
        # post-commit, the sync mutators perform the actual dispatch in
        # _dispatch_woken_parents() — never from the store loop.
        self._parents_to_wake: SimpleQueue[str] = SimpleQueue()

    def _release_swept_locals(self) -> None:
        """Apply post-commit handoffs produced by lazy cleanup.

        The cleanup transaction records swept ids and parents to wake only
        from SQLAlchemy's ``after_commit`` hook. The synchronous caller may
        therefore safely release retained closures and dispatch resumed
        parents without observing state that later rolls back.
        """
        swept: list[str] = []
        while True:
            try:
                swept.append(self._swept_waiting.get_nowait())
            except Empty:
                break
        if swept:
            with self._lock:
                for run_id in swept:
                    local = self._local.pop(run_id, None)
                    if local is not None:
                        local.cancel_event.set()
                        local.work = None
        self._dispatch_woken_parents()

    def _register_cleanup_handoffs(
        self,
        session: "AsyncSession",
        *,
        swept_run_ids: list[str],
        woken_parent_ids: list[str],
    ) -> None:
        """Publish cleanup side effects only after its transaction commits."""
        if not swept_run_ids and not woken_parent_ids:
            return
        sync_session = session.sync_session
        pending = sync_session.info.setdefault(
            _CLEANUP_HANDOFF_KEY,
            {"swept": [], "parents": [], "registered": False},
        )
        pending["swept"].extend(swept_run_ids)
        pending["parents"].extend(woken_parent_ids)
        if pending["registered"]:
            return
        pending["registered"] = True

        @event.listens_for(sync_session, "after_commit", once=True)
        def _publish(committed_session: Any) -> None:
            committed = committed_session.info.pop(
                _CLEANUP_HANDOFF_KEY, {}
            )
            for run_id in committed.get("swept", ()):
                self._swept_waiting.put(run_id)
            for parent_id in committed.get("parents", ()):
                self._parents_to_wake.put(parent_id)

    def _dispatch_woken_parents(self) -> None:
        """Dispatch parents whose last child's terminal write woke them.

        The DB rows already flipped ``waiting_for_children -> queued``
        inside the child's terminal transaction (or the park-time
        self-heal); this performs the mode-appropriate dispatch on the
        SYNC surface — the store loop must never take ``self._lock``
        (see ``_cleanup_db``) nor block on a queue round-trip.

        Queue mode enqueues a fresh dispatch message (any worker resumes
        from payload + checkpoint); a lost enqueue is healed by the
        reconciler, exactly like :meth:`resume_run`. No-queue mode
        re-appends the retained closure, honoring the park handshake —
        and a parent whose closure is gone (process restarted while
        parked) is logged loudly: it cannot resume in-process and ends
        via the waiting/stuck lifecycle, never silently.
        """
        while True:
            try:
                parent_id = self._parents_to_wake.get_nowait()
            except Empty:
                break
            if self._dispatch_queue is not None:
                try:
                    self._dispatch_queue.enqueue(
                        run_id=parent_id, tenant_id=DEFAULT_TENANT
                    )
                except Exception:  # noqa: BLE001 — row committed; reconciler heals
                    log.warning(
                        "Kind-Abschluss-Dispatch fuer Eltern-Run %s konnte "
                        "nicht gesendet werden — der Reconciler holt das "
                        "nach.",
                        parent_id,
                        exc_info=True,
                    )
                continue
            with self._lock:
                local = self._local.get(parent_id)
                if local is None or local.work is None:
                    log.warning(
                        "Eltern-Run %s wurde von seinen Kindern geweckt, "
                        "aber es ist keine Ausfuehrung mehr vorhanden "
                        "(Neustart im no-queue-Modus) — der Lauf kann "
                        "nicht fortgesetzt werden.",
                        parent_id,
                    )
                    continue
                if local.park_in_flight:
                    local.resume_requested = True
                else:
                    local.parked = False
                    self._pending.append(parent_id)
                    self._dispatch_locked()

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
        kind: str = "standard",
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
        session_id: str | None = None,
        origin_key: str | None = None,
    ) -> dict[str, Any]:
        """Persist one queued run, then dispatch locally or enqueue.

        ``origin_key`` needs no column here: the run service embeds it
        into ``request_payload["body"]`` and the row view lifts it back
        out for summaries.

        Raises:
            RunQueueFull: When the waiting queue is full and every
                execution slot is busy (queue-mode counts are
                cluster-wide via the database).
        """
        durable_payload = dict(request_payload or {})
        if origin_key:
            body = dict(durable_payload.get("body") or {})
            body["origin_key"] = origin_key
            durable_payload["body"] = body
        summary, created = self._call(
            self._submit_db(
                tenant_id=DEFAULT_TENANT,
                question=question[:500],
                stack_name=stack_name,
                agent_overrides=dict(agent_overrides or {}),
                mode=mode,
                workspace_id=workspace_id,
                created_by_sub=created_by_sub,
                created_by_tenant_id=created_by_tenant_id,
                request_payload=durable_payload or None,
                kind=kind,
                parent_run_id=parent_run_id,
                root_run_id=root_run_id,
                session_id=session_id,
            )
        )
        self._release_swept_locals()
        if not created:
            return summary
        run_id = summary["run_id"]
        if self._dispatch_queue is not None:
            try:
                self._dispatch_queue.enqueue(
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
        summary = self._call(
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
        self._release_swept_locals()
        return summary

    def get(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        """Return a public summary for *run_id*."""
        summary = self._call(
            self._summary_db(run_id, workspace_id, visible_to, also_visible)
        )
        self._release_swept_locals()
        return summary

    def list(
        self,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[dict[str, Any]]:
        """Return public summaries, newest first (unbounded)."""
        summaries = self._call(
            self._list_db(workspace_id, visible_to, also_visible)
        )
        self._release_swept_locals()
        return summaries

    def list_session_runs(
        self,
        session_id: str,
        *,
        visible_to: "UserContext | None" = None,
    ) -> list[dict[str, Any]]:
        """Visible summaries of one agent session, oldest first (K1)."""
        summaries = self._call(
            self._list_session_runs_db(session_id, visible_to)
        )
        self._release_swept_locals()
        return summaries

    def session_owners(
        self, session_id: str
    ) -> set[tuple[str | None, str | None]]:
        """Return every recorded owner identity for ``session_id``."""
        return self._call(self._session_owners_db(session_id))

    async def _session_owners_db(
        self, session_id: str
    ) -> set[tuple[str | None, str | None]]:
        async with self._session(DEFAULT_TENANT) as session:
            rows = (
                await session.execute(
                    select(
                        runs.c.created_by_tenant_id,
                        runs.c.created_by_sub,
                    )
                    .where(runs.c.session_id == session_id)
                    .distinct()
                )
            ).all()
            return {(row[0], row[1]) for row in rows}

    async def _list_session_runs_db(
        self,
        session_id: str,
        visible_to: "UserContext | None",
    ) -> list[dict[str, Any]]:
        async with self._session(DEFAULT_TENANT) as session:
            query = self._apply_run_visibility(
                select(runs)
                .where(runs.c.session_id == session_id)
                .order_by(
                    runs.c.created_at.asc(), runs.c.run_id.asc()
                ),
                None,
                visible_to,
                None,
            )
            rows = (await session.execute(query)).mappings().all()
            return [
                build_run_summary(_row_view(row), queue_position=None)
                for row in rows
            ]

    def list_page(
        self,
        *,
        limit: int,
        after: tuple[float, str] | None = None,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """One keyset page of visible summaries + the next cursor.

        Bounded read for the HTTP list endpoint (2.2): the durable table
        is 90-day-retained, so an active user's history is otherwise
        materialised whole on every poll. Keyed on the existing
        ``ix_runs_tenant_created_id`` (created_at, run_id) index.
        """
        result = self._call(
            self._list_page_db(
                workspace_id, visible_to, also_visible, limit, after
            )
        )
        self._release_swept_locals()
        return result

    def metrics_snapshot(self) -> RunStoreMetrics:
        """Current QUEUED/RUNNING counts for the ``/metrics`` collector.

        ``capacity`` is ``None``: in the durable topology the worker
        fleet owns the execution slots, not this API process, so there is
        no single in-process ceiling to report.
        """
        return self._call(self._metrics_snapshot_db())

    async def _metrics_snapshot_db(self) -> RunStoreMetrics:
        async with self._session("default") as session:
            rows = (
                await session.execute(
                    select(runs.c.status, func.count())
                    .where(
                        runs.c.status.in_(
                            (
                                RunStatus.QUEUED.value,
                                RunStatus.RUNNING.value,
                            )
                        )
                    )
                    .group_by(runs.c.status)
                )
            ).all()
        counts = {status: count for status, count in rows}
        return RunStoreMetrics(
            queued=counts.get(RunStatus.QUEUED.value, 0),
            active=counts.get(RunStatus.RUNNING.value, 0),
            capacity=None,
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

    def title(self, run_id: str) -> str | None:
        """The run's question as a share-surface title, regardless of
        visibility — a pending-share recipient must see it to decide. ``None``
        when the run no longer exists, so the inbox skips it."""
        return self._call(self._title_db(run_id))

    async def _title_db(self, run_id: str) -> str | None:
        async with self._session("default") as session:
            row = (
                await session.execute(
                    select(runs.c.question).where(runs.c.run_id == run_id)
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
        result = self._call(
            self._result_db(run_id, workspace_id, visible_to, also_visible)
        )
        self._release_swept_locals()
        return result

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
        summary, _affected = self.cancel_tree(
            run_id,
            workspace_id=workspace_id,
            visible_to=visible_to,
            also_visible=also_visible,
        )
        return summary

    def cancel_tree(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Cancel one durable run tree and expose the committed affected ids."""
        for attempt in range(_CANCEL_DEADLOCK_RETRIES):
            try:
                summary, cascaded_children = self._call(
                    self._cancel_db(
                        run_id, workspace_id, visible_to, also_visible
                    )
                )
                break
            except DBAPIError as err:
                # The cancel cascade locks parent->child while a concurrent
                # child terminal write locks child->parent; the opposite
                # orders can deadlock and Postgres aborts one side. Each
                # attempt is a fresh transaction (no partial state survives an
                # abort: the woken-parent append happens only post-commit), so
                # retry the whole cancel a bounded number of times before
                # surfacing a 500 to the caller. Catch DBAPIError, not
                # OperationalError: the asyncpg dialect maps a PostgresError
                # (the base of DeadlockDetectedError) to the bare DBAPI Error,
                # so SQLAlchemy wraps a real 40P01 as DBAPIError — an
                # OperationalError-only catch would never fire in production.
                # The sqlstate discriminator below keeps the guard narrow
                # (only 40P01 retries; every other DBAPIError re-raises), and
                # this also covers a sync psycopg driver, whose deadlock is an
                # OperationalError (a DBAPIError subclass).
                sqlstate = getattr(err.orig, "sqlstate", None) or getattr(
                    err.orig, "pgcode", None
                )
                if (
                    sqlstate != "40P01"
                    or attempt == _CANCEL_DEADLOCK_RETRIES - 1
                ):
                    raise
                log.warning(
                    "cancel deadlock (40P01) on run_id=%s; retry %s/%s",
                    run_id,
                    attempt + 1,
                    _CANCEL_DEADLOCK_RETRIES - 1,
                )
                time.sleep(_CANCEL_DEADLOCK_BACKOFF_SECONDS * (attempt + 1))
        self._release_swept_locals()
        self._dispatch_woken_parents()
        with self._lock:
            for cancelled_id in (run_id, *cascaded_children):
                local = self._local.get(cancelled_id)
                if local is None:
                    continue
                local.cancel_event.set()
                if (
                    cancelled_id == run_id
                    and summary["status"] != RunStatus.CANCELLED.value
                ):
                    # Root is running: the executing worker observes
                    # the cancel event and its finally cleans up.
                    continue
                was_pending = True
                try:
                    self._pending.remove(cancelled_id)
                except ValueError:
                    was_pending = False
                local.work = None
                if was_pending or local.parked:
                    # Nothing executes this run (queued locally or
                    # parked waiting) — no worker finally will release
                    # the retained closure, so do it here.
                    self._local.pop(cancelled_id, None)
        return summary, (run_id, *cascaded_children)

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
        self._release_swept_locals()
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
        self._release_swept_locals()
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
        landed = self._call(
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
        self._dispatch_woken_parents()
        return landed

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
        landed = self._call(
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
        self._dispatch_woken_parents()
        return landed

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
        landed = self._call(
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
        self._dispatch_woken_parents()
        return landed

    def mark_waiting(
        self,
        run_id: str,
        *,
        status: RunStatus | str,
        fence_attempt: int | None = None,
    ) -> None:
        """Park a RUNNING run in a waiting status (agent interrupt).

        The executing closure returns right after; the local job
        (no-queue mode) is retained by the dispatch worker so
        :meth:`resume_run` can re-dispatch the same closure. Only the
        two waiting statuses are accepted, and only from RUNNING. When
        a cancel request is already pending, the run is CANCELLED
        instead of parked (reason ``cancelled_while_waiting``) — a
        cancelled assignment must not sit in a waiting status until
        its TTL (memory-store parity).
        """
        waiting = RunStatus(status)
        if waiting not in WAITING_RUN_STATUSES:
            raise ValueError(f"not a waiting status: {status!r}")
        with self._lock:
            local = self._local.get(run_id)
            if local is not None:
                # Pre-arm BEFORE the row becomes waiting-visible: a
                # resume racing the commit must already see the
                # in-flight flag, or it would dispatch a second worker
                # while this one is still unwinding.
                local.parked = True
                local.park_in_flight = True
        try:
            parked = self._call(
                self._mark_waiting_db(run_id, waiting, fence_attempt)
            )
        except BaseException:
            with self._lock:
                local = self._local.get(run_id)
                if local is not None:
                    local.parked = False
                    local.park_in_flight = False
            raise
        # Park-time self-heal handoff: when every child was already
        # terminal, _mark_waiting_db flipped the row straight back to
        # queued — dispatch it now (park_in_flight defers the local
        # re-dispatch to this worker's unwind, exactly like resume_run).
        self._dispatch_woken_parents()
        if not parked:
            # Resolved as a cancel (a cancel request was pending):
            # nothing is retained, the unwind cleans up normally.
            with self._lock:
                local = self._local.get(run_id)
                if local is not None:
                    local.parked = False
                    local.park_in_flight = False

    def resume_run(
        self, run_id: str, *, control_write: Any = None
    ) -> dict[str, Any]:
        """Move a waiting run back to QUEUED and dispatch it.

        Queue mode re-enqueues (any worker resumes from the persisted
        payload/checkpoint); no-queue mode re-appends the retained local
        closure — a waiting row without one (process restarted) fails
        loudly instead of hanging.

        *control_write* is the rule-R9 seam: an optional coroutine
        function ``await control_write(session)`` executed INSIDE the
        ``waiting -> queued`` transaction, right after the status CAS
        lands — the agent control store records approval/clarification
        decisions through it so a crash can never separate the decision
        from the resume. When it raises, the whole transaction (including
        the status flip) rolls back.

        Raises:
            RunNotFound: Unknown id.
            RunActive: The run is not in a waiting status, or no-queue
                mode retained no closure to re-dispatch.
            Whatever *control_write* raises (transaction rolled back).
        """
        if self._dispatch_queue is None:
            # Check the retained closure BEFORE flipping the row: a
            # queued row nothing can execute would hang until the
            # stuck-row failsafe.
            with self._lock:
                local = self._local.get(run_id)
            if local is None or local.work is None:
                # Distinguish the failure loudly, with memory-parity
                # error types: unknown id -> RunNotFound (raised by the
                # row probe), not waiting -> RunActive with the status,
                # waiting without a closure (process restarted) ->
                # RunActive naming the restart.
                current_status = self._call(self._status_db(run_id))
                if current_status not in _WAITING_VALUES:
                    raise RunActive(
                        f"run {run_id} is not waiting "
                        f"(status {current_status})"
                    )
                raise RunActive(
                    f"run {run_id} has no retained work to resume "
                    "(no-queue mode after a restart)"
                )
        summary = self._call(self._resume_db(run_id, control_write))
        self._release_swept_locals()
        if self._dispatch_queue is not None:
            try:
                self._dispatch_queue.enqueue(
                    run_id=run_id, tenant_id=DEFAULT_TENANT
                )
            except Exception:  # noqa: BLE001 — row committed; reconciler heals
                log.warning(
                    "Resume-Dispatch fuer Run %s konnte nicht gesendet "
                    "werden — der Reconciler holt das nach.",
                    run_id,
                    exc_info=True,
                )
            return summary
        with self._lock:
            local = self._local.get(run_id)
            if local is None or local.work is None:
                # Cancelled/TTL-swept in the window between the check
                # and the row flip — _resume_db would have raised, so
                # this is unreachable in practice; guard it anyway.
                raise RunActive(f"run {run_id} lost its work while resuming")
            if local.park_in_flight:
                # The parking worker has not unwound yet: dispatching
                # now would run the same closure on two threads. The
                # unwind performs the deferred re-dispatch.
                local.resume_requested = True
            else:
                local.parked = False
                self._pending.append(run_id)
                self._dispatch_locked()
        return summary

    def children(self, run_id: str) -> list[dict[str, Any]]:
        """Summaries of this run's direct children, newest first.

        Authorization happens on the PARENT (the route resolves it via
        :meth:`get` first); children inherit that access (plan rule R7).
        """
        children = self._call(self._children_db(run_id))
        self._release_swept_locals()
        return children

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

    def dispatch_status(self, run_id: str, tenant_id: str) -> str | None:
        """Authoritative row status for worker duplicate classification.

        This intentionally skips cleanup and caller visibility: it is a
        worker-only read under the dispatch message's tenant. ``None`` means
        the row disappeared; storage errors propagate so uncertainty never
        turns a legitimate successor into an ACK.
        """
        return self._call(self._dispatch_status_db(run_id, tenant_id))

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

    def _terminate_work_exception(
        self, handle: RunHandle, entity_id: str, exc: BaseException
    ) -> None:
        """Preserve typed run failures in durable no-queue mode."""
        del entity_id
        terminate_native_run(handle, exc)

    def _auto_complete(self, run_id: str) -> None:
        # exclude_waiting: a work closure that parked its run (agent
        # interrupt) returns normally — the safety net must not
        # terminal-write over the parked row (memory-store parity).
        self._call(
            self._terminal_db(
                run_id,
                RunStatus.COMPLETED,
                fence_attempt=None,
                warn_on_noop=False,
                exclude_waiting=True,
                event_builder=lambda row_snapshot: (
                    "inqtrix.run.completed",
                    {"status": "completed", "snapshot": row_snapshot},
                ),
            )
        )
        self._dispatch_woken_parents()

    # -- subscription poll bridge ----------------------------------------- #

    def _events_after(
        self, run_id: str, tenant_id: str, after_sequence: int
    ) -> list[dict[str, Any]]:
        return self._call(
            self._events_after_db(run_id, tenant_id, after_sequence)
        )

    # -- async database operations ----------------------------------------- #

    async def _admit_agent_child_db(
        self,
        session: "AsyncSession",
        fields: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Canonicalize one child and serialize it against tree cancellation.

        The canonical root and idempotency identity both come from durable
        parent state. Lock order is root -> direct parent, matching cancel, so
        either an insert commits before the cascade and is included or the
        cancel flag lands first and admission is rejected.
        """
        if fields.get("kind") != "agent_child":
            return None
        parent_run_id = str(fields.get("parent_run_id") or "")
        if not parent_run_id:
            raise RunParentInactive("agent child has no parent run")
        parent_probe = (
            await session.execute(
                select(runs.c.root_run_id).where(
                    runs.c.run_id == parent_run_id
                )
            )
        ).first()
        if parent_probe is None:
            raise RunParentInactive("agent child parent is missing")
        canonical_root_id = str(parent_probe[0] or parent_run_id)
        root = (
            await session.execute(
                select(
                    runs.c.run_id,
                    runs.c.status,
                    runs.c.cancel_requested,
                    runs.c.root_run_id,
                )
                .where(runs.c.run_id == canonical_root_id)
                .with_for_update()
            )
        ).mappings().first()
        if root is None:
            raise RunParentInactive("agent child root is missing")
        if canonical_root_id == parent_run_id:
            parent = root
        else:
            parent = (
                await session.execute(
                    select(
                        runs.c.run_id,
                        runs.c.status,
                        runs.c.cancel_requested,
                        runs.c.root_run_id,
                    )
                    .where(runs.c.run_id == parent_run_id)
                    .with_for_update()
                )
            ).mappings().first()
            if parent is None:
                raise RunParentInactive("agent child parent is missing")

        request_body = (fields.get("request_payload") or {}).get("body") or {}
        origin_key = (
            str(request_body.get("origin_key") or "")
            if isinstance(request_body, dict)
            else ""
        )
        if origin_key:
            existing_id = (
                await session.execute(
                    select(runs.c.run_id)
                    .where(
                        runs.c.kind == "agent_child",
                        runs.c.parent_run_id == parent_run_id,
                        func.json_extract_path_text(
                            runs.c.request_payload,
                            "body",
                            "origin_key",
                        )
                        == origin_key,
                    )
                    .order_by(runs.c.created_at, runs.c.run_id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing_id is not None:
                existing = await self._row_db(session, existing_id)
                position = (
                    await self._queue_position_db(session, existing.created_at)
                    if existing.status == RunStatus.QUEUED.value
                    else None
                )
                return build_run_summary(existing, queue_position=position)

        inactive = {
            *_TERMINAL_VALUES,
            RunStatus.EXPIRED.value,
        }
        if (
            str(parent["status"]) in inactive
            or bool(parent["cancel_requested"])
            or str(root["status"]) in inactive
            or bool(root["cancel_requested"])
        ):
            raise RunParentInactive(
                "agent child parent or root is no longer active"
            )
        fields["root_run_id"] = canonical_root_id
        return None

    async def _submit_db(
        self, *, tenant_id: str, **fields: Any
    ) -> tuple[dict[str, Any], bool]:
        async with self._session(tenant_id) as session:
            await self._cleanup_db(session)
            existing = await self._admit_agent_child_db(session, fields)
            if existing is not None:
                return existing, False
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
            submitting_sub = fields.get("created_by_sub")
            if (
                self._max_concurrent_per_user is not None
                and submitting_sub is not None
            ):
                # Fairness bound UNDER the global cap, counted in the SAME
                # transaction as the insert. COUNTED: QUEUED+RUNNING
                # standard runs and agent CHILDREN (the runs that occupy
                # an execution slot). EXCLUDED: WAITING runs (parked,
                # slot-free) AND agent PARENTS (kind='agent'), which park
                # immediately and must not contend against their own
                # children for the user's budget. Scope is created_by_sub
                # only (single-tenant today; see the memory-store note).
                #
                # APPROXIMATE bound: under READ COMMITTED the COUNT takes
                # no lock on the counted rows, so N concurrent submits by
                # one sub can each observe cap-1 and all pass, transiently
                # reaching up to cap+N-1. That is acceptable for a
                # fairness bound (the memory store IS exact under its
                # lock); a hard guarantee would need SELECT FOR UPDATE on
                # a per-sub sentinel or SERIALIZABLE retry, at real
                # hot-path cost. The docstring calls this out.
                in_flight = await session.scalar(
                    select(func.count())
                    .select_from(runs)
                    .where(
                        runs.c.created_by_sub == submitting_sub,
                        runs.c.kind != "agent",
                        runs.c.status.in_(
                            (
                                RunStatus.QUEUED.value,
                                RunStatus.RUNNING.value,
                            )
                        ),
                    )
                )
                if (in_flight or 0) >= self._max_concurrent_per_user:
                    raise RunPerUserLimit(
                        "per-user in-flight run cap reached"
                    )

            created_at = time.time()
            try:
                run_id = await self._insert_with_unique_id(
                    session,
                    tenant_id=tenant_id,
                    created_at=created_at,
                    **fields,
                )
            except IntegrityError as exc:
                if _ACTIVE_AGENT_SESSION_CONSTRAINT in str(exc.orig):
                    raise RunSessionActive(fields.get("session_id")) from exc
                raise
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
            await self._project_child_progress_db(
                session,
                child_run_id=run_id,
                kind=str(fields.get("kind") or "standard"),
                parent_run_id=fields.get("parent_run_id"),
                request_payload=fields.get("request_payload") or {},
                run_status=RunStatus.QUEUED.value,
                event_type="inqtrix.run.queued",
                payload={"status": "queued", "queue_position": position},
                snapshot={},
            )
            row = await self._row_db(session, run_id)
            return build_run_summary(row, queue_position=position), True

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
                    kind=fields["kind"],
                    parent_run_id=fields["parent_run_id"],
                    root_run_id=fields["root_run_id"],
                    session_id=fields["session_id"],
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

    async def _status_db(self, run_id: str) -> str:
        """Just the status string; raises ``RunNotFound`` when missing."""
        async with self._session(DEFAULT_TENANT) as session:
            status = (
                await session.execute(
                    select(runs.c.status).where(runs.c.run_id == run_id)
                )
            ).scalar_one_or_none()
            if status is None:
                raise RunNotFound(run_id)
            return status

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

    @staticmethod
    def _apply_run_visibility(
        query,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None",
    ):
        """Apply the owner/workspace/shared-in visibility WHERE (shared).

        The single definition of who may see a run, used by both the
        unbounded ``_list_db`` and the paginated ``_list_page_db`` so the
        two can never drift.
        """
        shared_ids = list(also_visible) if also_visible else []
        if visible_to is not None:
            owned = and_(
                runs.c.created_by_sub == visible_to.principal.sub,
                runs.c.created_by_tenant_id
                == visible_to.principal.tenant_id,
            )
            if workspace_id is not None:
                owned = and_(owned, runs.c.workspace_id == workspace_id)
            if shared_ids:
                # Shared-in rows join REGARDLESS of the workspace
                # namespace filter (they carry the grantor's id).
                return query.where(or_(owned, runs.c.run_id.in_(shared_ids)))
            return query.where(owned)
        if workspace_id is not None:
            return query.where(runs.c.workspace_id == workspace_id)
        return query

    async def _queue_positions_for(
        self, session: "AsyncSession", run_ids: list[str]
    ) -> dict[str, int]:
        """Global queue positions for the QUEUED rows in *run_ids*.

        ONE window-function scan over the (small, capacity-bounded) set
        of QUEUED rows, replacing the old per-row COUNT(*) N+1. Positions
        are GLOBAL (1-based over every queued run), so a queued run on a
        later page still reports its true position.
        """
        if not run_ids:
            return {}
        ranked = (
            select(
                runs.c.run_id,
                func.row_number()
                .over(order_by=(runs.c.created_at, runs.c.run_id))
                .label("position"),
            )
            .where(runs.c.status == RunStatus.QUEUED.value)
            .subquery()
        )
        rows = (
            await session.execute(
                select(ranked.c.run_id, ranked.c.position).where(
                    ranked.c.run_id.in_(run_ids)
                )
            )
        ).all()
        return {run_id: int(position) for run_id, position in rows}

    async def _list_db(
        self,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[dict[str, Any]]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            query = self._apply_run_visibility(
                # run_id tiebreaker matches list_page's keyset order, so the
                # unbounded internal read and the paginated HTTP endpoint agree
                # on the relative order of runs sharing a created_at epoch.
                select(runs).order_by(
                    runs.c.created_at.desc(), runs.c.run_id.desc()
                ),
                workspace_id,
                visible_to,
                also_visible,
            )
            rows = (await session.execute(query)).mappings().all()
            positions = await self._queue_positions_for(
                session,
                [
                    row["run_id"]
                    for row in rows
                    if row["status"] == RunStatus.QUEUED.value
                ],
            )
            summaries = []
            for row in rows:
                shared = (
                    also_visible.get(row["run_id"])
                    if also_visible is not None
                    and not _visible_row(row, visible_to)
                    else None
                )
                summaries.append(
                    build_run_summary(
                        _row_view(row),
                        queue_position=positions.get(row["run_id"]),
                        access=_access_annotation(shared),
                    )
                )
            return summaries

    async def _list_page_db(
        self,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None",
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            query = self._apply_run_visibility(
                select(runs),
                workspace_id,
                visible_to,
                also_visible,
            )
            if after is not None:
                # Keyset over the (created_at, run_id) index — the run_id
                # tiebreaker is mandatory (created_at is a float epoch and
                # collides on bulk inserts, so a created_at-only cursor
                # would skip or repeat rows).
                query = query.where(
                    tuple_(runs.c.created_at, runs.c.run_id)
                    < tuple_(after[0], after[1])
                )
            query = query.order_by(
                runs.c.created_at.desc(), runs.c.run_id.desc()
            ).limit(limit + 1)
            rows = (await session.execute(query)).mappings().all()
            window = list(rows[: limit + 1])
            page_rows = window[:limit]
            next_cursor = (
                encode_cursor(
                    page_rows[-1]["created_at"], page_rows[-1]["run_id"]
                )
                if len(window) > limit and page_rows
                else None
            )
            positions = await self._queue_positions_for(
                session,
                [
                    row["run_id"]
                    for row in page_rows
                    if row["status"] == RunStatus.QUEUED.value
                ],
            )
            summaries = []
            for row in page_rows:
                shared = (
                    also_visible.get(row["run_id"])
                    if also_visible is not None
                    and not _visible_row(row, visible_to)
                    else None
                )
                summaries.append(
                    build_run_summary(
                        _row_view(row),
                        queue_position=positions.get(row["run_id"]),
                        access=_access_annotation(shared),
                    )
                )
            return summaries, next_cursor

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
    ) -> tuple[dict[str, Any], list[str]]:
        """Cancel one run tree and return every descendant id.

        Active descendants receive the cancellation transition. Terminal
        descendants are still returned so control-row reconciliation sees
        the same complete tree as the in-memory store.
        """
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
            canonical_root_id = str(row["root_run_id"] or run_id)
            root_locked = (
                await session.execute(
                    select(runs.c.run_id)
                    .where(runs.c.run_id == canonical_root_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if root_locked is None:
                raise RunNotFound(run_id)
            woken = await self._cancel_row_db(
                session, run_id, row["tenant_id"], row["status"]
            )
            # Cascade: an agent run cancels its whole tree — children are
            # real runs a client may not even know about, so the parent
            # cancel is responsible for them (authorization was the
            # parent's; children inherit, plan rule R7). The per-child
            # wake probe no-ops here: the parent went terminal first.
            descendants = select(
                runs.c.run_id,
                runs.c.tenant_id,
                runs.c.status,
                runs.c.parent_run_id,
            ).where(runs.c.parent_run_id == run_id).cte(
                "run_descendants", recursive=True
            )
            descendants = descendants.union_all(
                select(
                    runs.c.run_id,
                    runs.c.tenant_id,
                    runs.c.status,
                    runs.c.parent_run_id,
                ).join(
                    descendants,
                    runs.c.parent_run_id == descendants.c.run_id,
                )
            )
            child_rows = (
                await session.execute(
                    select(
                        descendants.c.run_id,
                        descendants.c.tenant_id,
                        descendants.c.status,
                    )
                )
            ).all()
            cascaded: list[str] = []
            for child_id, child_tenant, child_status in child_rows:
                cascaded.append(child_id)
                if child_status not in _TERMINAL_VALUES:
                    await self._cancel_row_db(
                        session, child_id, child_tenant, child_status
                    )
            fresh = await self._row_db(session, run_id)
            position = (
                await self._queue_position_db(session, fresh.created_at)
                if fresh.status == RunStatus.QUEUED.value
                else None
            )
            summary = build_run_summary(fresh, queue_position=position)
        if woken:
            self._parents_to_wake.put(woken)
        return summary, cascaded

    async def _cancel_row_db(
        self,
        session: "AsyncSession",
        run_id: str,
        tenant_id: str,
        status: str,
    ) -> str | None:
        """Apply the status-appropriate cancel transition to one row.

        Every CAS can lose to a concurrent transition (claim, park,
        resume); each miss re-reads and degrades to the branch for the
        FRESH status instead of a silent no-op, so no run slips
        through a cancel because it changed state underneath it.

        Returns:
            The parent run id when this cancel terminally ended an
            agent child whose last sibling was already terminal (the
            wake probe fired — the caller hands the id to the
            post-commit dispatch); ``None`` otherwise. Immediate QUEUED
            and WAITING cancellations terminate here; a RUNNING child
            terminates later via ``_terminal_db``, which carries its
            own probe.
        """
        if status in _WAITING_VALUES:
            landed, woken_parent = await self._cancel_waiting_row_db(
                session, run_id, tenant_id
            )
            if landed:
                return woken_parent
            # Lost the CAS to a concurrent resume (or the run went
            # terminal, which is absorbing): degrade to the fresh
            # status — a resumed run must still receive the cancel.
            fresh = (
                await session.execute(
                    select(runs.c.status).where(runs.c.run_id == run_id)
                )
            ).scalar_one_or_none()
            if fresh not in (
                RunStatus.QUEUED.value,
                RunStatus.RUNNING.value,
            ):
                return None
            status = fresh
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
                    .returning(
                        runs.c.kind,
                        runs.c.parent_run_id,
                        runs.c.request_payload,
                        runs.c.snapshot,
                        runs.c.attempt,
                    )
                )
            ).first()
            if cancelled is not None:
                return await self._record_terminal_run_db(
                    session,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    kind=str(cancelled[0] or "standard"),
                    parent_run_id=cancelled[1],
                    request_payload=dict(cancelled[2] or {}),
                    status=RunStatus.CANCELLED.value,
                    event_type="inqtrix.run.cancelled",
                    payload={
                        "status": "cancelled",
                        "reason": "cancelled_before_start",
                    },
                    snapshot=dict(cancelled[3] or {}),
                    attempt=int(cancelled[4] or 0) or None,
                )
            else:
                # Lost the CAS to a concurrent claim: the run is
                # running now — degrade to the two-phase cancel
                # request instead of a stale no-op.
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
                    tenant_id,
                    expand_run_event(
                        "inqtrix.run.cancel_requested",
                        {
                            "status": "running",
                            "reason": "client_requested_cancel",
                        },
                        status=RunStatus.RUNNING.value,
                    )[1],
                )
                return None
            # Lost the CAS: the run went terminal (absorbing, nothing
            # to do) OR it just PARKED itself — a parked run must not
            # slip through a cancel silently, so re-read and resolve
            # the wait immediately.
            fresh = (
                await session.execute(
                    select(runs.c.status).where(runs.c.run_id == run_id)
                )
            ).scalar_one_or_none()
            if fresh in _WAITING_VALUES:
                _landed, woken_parent = await self._cancel_waiting_row_db(
                    session, run_id, tenant_id
                )
                return woken_parent
        return None

    async def _cancel_waiting_row_db(
        self, session: "AsyncSession", run_id: str, tenant_id: str
    ) -> tuple[bool, str | None]:
        """Cancel one waiting row immediately (nothing is executing).

        Returns:
            ``(landed, woken_parent)``. ``landed=False`` means the row left
            the waiting statuses concurrently and the caller must degrade to
            the fresh status. A non-null parent id means this was its last
            active child and the post-commit dispatcher must wake it.
        """
        waited = (
            await session.execute(
                update(runs)
                .where(
                    runs.c.run_id == run_id,
                    runs.c.status.in_(_WAITING_VALUES),
                )
                .values(
                    status=RunStatus.CANCELLED.value,
                    cancel_requested=True,
                    waiting_since=None,
                    finished_at=time.time(),
                )
                .returning(
                    runs.c.run_id,
                    runs.c.kind,
                    runs.c.parent_run_id,
                    runs.c.request_payload,
                    runs.c.snapshot,
                    runs.c.attempt,
                )
            )
        ).first()
        if waited is None:
            return False, None
        woken_parent = await self._record_terminal_run_db(
            session,
            run_id=run_id,
            tenant_id=tenant_id,
            kind=str(waited[1] or "standard"),
            parent_run_id=waited[2],
            request_payload=dict(waited[3] or {}),
            status=RunStatus.CANCELLED.value,
            event_type="inqtrix.run.cancelled",
            payload={
                "status": "cancelled",
                "reason": "cancelled_while_waiting",
            },
            snapshot=dict(waited[4] or {}),
            attempt=int(waited[5] or 0) or None,
        )
        return True, woken_parent

    async def _mark_waiting_db(
        self,
        run_id: str,
        waiting: RunStatus,
        fence_attempt: int | None = None,
    ) -> bool:
        """Park one RUNNING row; returns whether it parked.

        ``False`` means a pending cancel request won instead: the row
        went terminal ``cancelled`` (reason ``cancelled_while_waiting``)
        — parking it would leave a cancelled assignment waiting until
        its TTL (memory-store parity).
        """
        woken_parent: str | None = None
        parked = False
        async with self._session(DEFAULT_TENANT) as session:
            query = (
                update(runs)
                .where(
                    runs.c.run_id == run_id,
                    runs.c.status == RunStatus.RUNNING.value,
                    runs.c.cancel_requested.is_(False),
                )
                .values(
                    status=waiting.value,
                    waiting_since=time.time(),
                )
                .returning(runs.c.snapshot, runs.c.tenant_id)
            )
            if fence_attempt is not None:
                # Queue-worker park: a reclaimed zombie must not park a
                # run the live attempt owns (same fence as the terminal
                # writes).
                query = query.where(
                    runs.c.claimed_by == self._worker_id,
                    runs.c.attempt == fence_attempt,
                )
            row = (await session.execute(query)).first()
            if row is not None:
                parked = True
                _, events = expand_run_event(
                    "inqtrix.run.waiting",
                    {"status": waiting.value, "snapshot": dict(row[0] or {})},
                    status=waiting.value,
                )
                await self._append_events_db(session, run_id, row[1], events)
                if waiting is RunStatus.WAITING_FOR_CHILDREN:
                    # Lost-wakeup self-heal, in the SAME transaction as
                    # the park: the last child may have gone terminal
                    # while this run was still unwinding towards the
                    # park — its wake probe then found the parent
                    # RUNNING and no-oped. Re-probing here closes the
                    # window: either the terminal write commits first
                    # (this probe sees zero outstanding and flips
                    # straight back to queued) or the park commits
                    # first (the terminal probe sees the waiting row).
                    woken_parent = await self._maybe_wake_parent_db(
                        session, parent_run_id=run_id
                    )
                if woken_parent is None:
                    return True
                # Self-heal hit: fall out of the session block so the
                # transaction commits, then hand the id to the
                # post-commit dispatch below.
            else:
                woken_parent = await self._resolve_waiting_miss_db(
                    session, run_id, fence_attempt
                )
                if woken_parent is None:
                    return False
        # The self-heal and cancelled-child paths both wake only after their
        # transaction committed. A root cancellation has no parent handoff.
        if woken_parent:
            self._parents_to_wake.put(woken_parent)
        return parked

    async def _resolve_waiting_miss_db(
        self,
        session: "AsyncSession",
        run_id: str,
        fence_attempt: int | None,
    ) -> str | None:
        """Resolve a missed park CAS: cancel, fence violation, or bug.

        Returns the parent id when cancelling an agent child woke its parent,
        otherwise ``None``. Every non-cancel miss raises loudly.
        """
        # The summary view (_row_db) omits the fencing pair, so the
        # fresh read selects it directly.
        current = (
            await session.execute(
                select(
                    runs.c.status, runs.c.claimed_by, runs.c.attempt
                ).where(runs.c.run_id == run_id)
            )
        ).first()
        if current is None:
            raise RunNotFound(run_id)
        if (
            fence_attempt is not None
            and current.status == RunStatus.RUNNING.value
            and (
                current.claimed_by != self._worker_id
                or current.attempt != fence_attempt
            )
        ):
            # Fenced miss on a RUNNING row owned by ANOTHER attempt:
            # a zombie must not fall through into the (unfenced)
            # cancel resolution below — that would kill the LIVE
            # attempt's execution. A fenced miss on the OWN attempt
            # means cancel_requested blocked the park; that case
            # falls through and resolves as cancel (memory parity).
            raise RunActive(
                f"run {run_id} is owned by another worker attempt"
            )
        if current.status == RunStatus.RUNNING.value:
            # RUNNING but cancel_requested: resolve the pending
            # cancel now instead of parking.
            cancelled = (
                await session.execute(
                    update(runs)
                    .where(
                        runs.c.run_id == run_id,
                        runs.c.status == RunStatus.RUNNING.value,
                    )
                    .values(
                        status=RunStatus.CANCELLED.value,
                        finished_at=time.time(),
                    )
                    .returning(
                        runs.c.snapshot,
                        runs.c.tenant_id,
                        runs.c.kind,
                        runs.c.parent_run_id,
                        runs.c.request_payload,
                        runs.c.attempt,
                    )
                )
            ).first()
            if cancelled is not None:
                payload = {
                    "status": "cancelled",
                    "reason": "cancelled_while_waiting",
                    "snapshot": dict(cancelled[0] or {}),
                }
                return await self._record_terminal_run_db(
                    session,
                    run_id=run_id,
                    tenant_id=cancelled[1],
                    kind=str(cancelled[2] or "standard"),
                    parent_run_id=cancelled[3],
                    request_payload=dict(cancelled[4] or {}),
                    status=RunStatus.CANCELLED.value,
                    event_type="inqtrix.run.cancelled",
                    payload=payload,
                    snapshot=dict(cancelled[0] or {}),
                    attempt=int(cancelled[5] or 0) or None,
                )
            current = await self._row_db(session, run_id)
        # Not RUNNING (terminal or a caller bug) — loud, memory
        # parity: RunNotFound came from _row_db above if missing.
        raise RunActive(
            f"run {run_id} cannot wait from status {current.status}"
        )

    async def _resume_db(
        self, run_id: str, control_write: Any = None
    ) -> dict[str, Any]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            row = (
                await session.execute(
                    update(runs)
                    .where(
                        runs.c.run_id == run_id,
                        runs.c.status.in_(_WAITING_VALUES),
                    )
                    .values(
                        status=RunStatus.QUEUED.value,
                        waiting_since=None,
                    )
                    .returning(runs.c.tenant_id, runs.c.created_at)
                )
            ).first()
            if row is None:
                current = await self._row_db(session, run_id)
                raise RunActive(
                    f"run {run_id} is not waiting (status {current.status})"
                )
            if control_write is not None:
                # Rule R9: the control-store decision commits or rolls
                # back together with the status flip above.
                await control_write(session)
            position = await self._queue_position_db(session, row[1])
            _, events = expand_run_event(
                "inqtrix.run.queued",
                {
                    "status": "queued",
                    "queue_position": position,
                    "resumed": True,
                },
                status=RunStatus.QUEUED.value,
            )
            await self._append_events_db(session, run_id, row[0], events)
            fresh = await self._row_db(session, run_id)
            return build_run_summary(fresh, queue_position=position)

    async def _children_db(self, run_id: str) -> list[dict[str, Any]]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            rows = (
                (
                    await session.execute(
                        select(runs)
                        .where(runs.c.parent_run_id == run_id)
                        .order_by(runs.c.created_at.desc())
                    )
                )
                .mappings()
                .all()
            )
            # ONE window-function scan for the queued children's positions,
            # not a COUNT per child — the /children endpoint is polled by the
            # Agent Desk, so a wide fan-out must not issue N queue COUNTs
            # (mirrors _list_db / _list_page_db).
            positions = await self._queue_positions_for(
                session,
                [
                    row["run_id"]
                    for row in rows
                    if row["status"] == RunStatus.QUEUED.value
                ],
            )
            return [
                build_run_summary(
                    _row_view(row),
                    queue_position=positions.get(row["run_id"]),
                )
                for row in rows
            ]

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
            row = (
                await session.execute(
                    select(runs).where(runs.c.run_id == run_id)
                )
            ).mappings().first()
            if row is None:
                raise RunNotFound(run_id)
            if row["status"] in _TERMINAL_VALUES:
                # The log's ends-terminal invariant is what stops SSE
                # streams; post-terminal signals are dropped LOUDLY
                # (memory parity) — finished-run clients reconcile via
                # the REST reads.
                log.warning(
                    "Event %s fuer Run %s verworfen: der Lauf ist bereits "
                    "terminal (%s), das Event-Log endet mit dem "
                    "Terminal-Event.",
                    event_type,
                    run_id,
                    row["status"],
                )
                return
            new_snapshot, events = expand_run_event(
                event_type, payload, status=row["status"]
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
            await self._project_child_progress_db(
                session,
                child_run_id=run_id,
                kind=str(row["kind"] or "standard"),
                parent_run_id=row["parent_run_id"],
                request_payload=dict(row["request_payload"] or {}),
                run_status=str(row["status"]),
                event_type=event_type,
                payload=payload,
                snapshot=(
                    new_snapshot
                    if new_snapshot is not None
                    else dict(row["snapshot"] or {})
                ),
                attempt=int(row["attempt"] or 0) or None,
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

    async def _project_child_progress_db(
        self,
        session: "AsyncSession",
        *,
        child_run_id: str,
        kind: str,
        parent_run_id: str | None,
        request_payload: dict[str, Any],
        run_status: str,
        event_type: str,
        payload: dict[str, Any],
        snapshot: dict[str, Any],
        attempt: int | None = None,
    ) -> None:
        """Append one bounded child projection to its parent event log."""
        body = request_payload.get("body") or {}
        parent_task_id = (
            str(body.get("parent_task_id") or "")
            if isinstance(body, dict)
            else ""
        )
        logical_attempt = (
            int(body.get("parent_task_attempt", 0) or 0)
            if isinstance(body, dict)
            else 0
        )
        if (
            kind != "agent_child"
            or not parent_run_id
            or not parent_task_id
            or not should_project_child_event(event_type)
        ):
            return
        parent = (
            await session.execute(
                select(runs.c.status, runs.c.tenant_id).where(
                    runs.c.run_id == parent_run_id,
                    runs.c.status.notin_(_TERMINAL_VALUES),
                ).with_for_update()
            )
        ).first()
        if parent is None:
            return
        projected = build_child_progress_payload(
            child_run_id=child_run_id,
            parent_task_id=parent_task_id,
            run_status=run_status,
            event_type=event_type,
            payload=payload,
            snapshot=snapshot,
            attempt=logical_attempt or attempt,
        )
        await self._append_events_db(
            session,
            parent_run_id,
            parent[1],
            expand_run_event(
                CHILD_PROGRESS_EVENT,
                projected,
                status=parent[0],
            )[1],
        )

    async def _record_terminal_run_db(
        self,
        session: "AsyncSession",
        *,
        run_id: str,
        tenant_id: str,
        kind: str,
        parent_run_id: str | None,
        request_payload: dict[str, Any],
        status: str,
        event_type: str,
        payload: dict[str, Any],
        snapshot: dict[str, Any],
        attempt: int | None,
    ) -> str | None:
        """Persist one terminal signal, child projection, and parent wake."""
        await self._append_events_db(
            session,
            run_id,
            tenant_id,
            expand_run_event(event_type, payload, status=status)[1],
        )
        await self._project_child_progress_db(
            session,
            child_run_id=run_id,
            kind=kind,
            parent_run_id=parent_run_id,
            request_payload=request_payload,
            run_status=status,
            event_type=event_type,
            payload=payload,
            snapshot=snapshot,
            attempt=attempt,
        )
        if kind == "agent_child" and parent_run_id:
            return await self._maybe_wake_parent_db(
                session, parent_run_id=parent_run_id
            )
        return None

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
        exclude_waiting: bool = False,
    ) -> bool:
        woken_parent: str | None = None
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
                .returning(
                    runs.c.snapshot,
                    runs.c.tenant_id,
                    runs.c.kind,
                    runs.c.parent_run_id,
                    runs.c.request_payload,
                    runs.c.attempt,
                )
            )
            if exclude_waiting:
                # Auto-complete safety net only: a parked (waiting) run
                # is not "still running work that forgot to complete" —
                # completing it would destroy the interrupt.
                query = query.where(runs.c.status.notin_(_WAITING_VALUES))
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
            woken_parent = await self._record_terminal_run_db(
                session,
                run_id=run_id,
                tenant_id=row[1],
                kind=str(row[2] or "standard"),
                parent_run_id=row[3],
                request_payload=dict(row[4] or {}),
                status=status.value,
                event_type=event_type,
                payload=payload,
                snapshot=dict(row[0] or {}),
                attempt=int(row[5] or 0) or None,
            )
        if woken_parent:
            # Post-commit handoff: the sync mutators dispatch (queue
            # enqueue / local re-append) — never this loop coroutine.
            self._parents_to_wake.put(woken_parent)
        return True

    async def _maybe_wake_parent_db(
        self, session: "AsyncSession", *, parent_run_id: str
    ) -> str | None:
        """Flip a children-parked parent to QUEUED once no child is left.

        Runs inside the child's terminal transaction (or the park-time
        self-heal): the sibling probe and the parent CAS commit together
        with the child's transition. The CAS keys on
        ``waiting_for_children`` alone (resume is attempt-agnostic,
        exactly like ``_resume_db``); a parent still RUNNING — the child
        finished before the parent finished parking — makes this a no-op,
        closed by the park-time self-heal in ``_mark_waiting_db``.

        The FIRST statement locks the parent row (``FOR NO KEY UPDATE``).
        This is load-bearing under multi-worker queue mode: two of the
        parent's children terminating in CONCURRENT transactions on
        DIFFERENT workers run under READ COMMITTED, where each cannot see
        the other's uncommitted terminal write. Without the lock, both
        sibling probes would count the other child as still outstanding
        (=1) and BOTH decline to wake — a lost wake-up that strands the
        parent until the waiting TTL. The row lock serialises the two
        wake deciders on the parent: the later one blocks until the
        earlier commits, then re-reads the true committed count (0) and
        wakes. NO KEY UPDATE suffices — no PK/FK column is touched — and
        is lighter than FOR UPDATE. Lock ordering: a terminating child
        locks its own child row first, then this shared parent row; the
        park self-heal already holds the parent row (its own park UPDATE)
        and re-locks it REENTRANTLY in the same transaction; the cancel
        cascade likewise already holds the parent row (the parent's own
        cancel UPDATE) before the per-child probe re-locks it reentrantly.
        So this adds no NEW lock-order cycle. (The pre-existing
        cancel-vs-terminal contention on the parent row — present since
        the original CAS also write-locked it — can still surface as a
        Postgres deadlock abort under an exact race. The terminal write is
        redelivered by the queue; the synchronous cancel path, which has no
        redelivery, retries the whole transaction a bounded number of times
        on SQLSTATE 40P01 — see ``cancel`` and ``_CANCEL_DEADLOCK_RETRIES``.)

        Returns:
            The parent id when the flip landed (caller hands it to the
            post-commit dispatch), ``None`` otherwise.
        """
        locked = (
            await session.execute(
                select(runs.c.run_id)
                .where(runs.c.run_id == parent_run_id)
                .with_for_update(key_share=True)
            )
        ).scalar_one_or_none()
        if locked is None:
            return None
        outstanding = await session.scalar(
            select(func.count())
            .select_from(runs)
            .where(
                runs.c.parent_run_id == parent_run_id,
                runs.c.status.notin_(_TERMINAL_VALUES),
            )
        )
        if outstanding:
            return None
        row = (
            await session.execute(
                update(runs)
                .where(
                    runs.c.run_id == parent_run_id,
                    runs.c.status
                    == RunStatus.WAITING_FOR_CHILDREN.value,
                )
                .values(
                    status=RunStatus.QUEUED.value,
                    waiting_since=None,
                )
                .returning(runs.c.tenant_id, runs.c.created_at)
            )
        ).first()
        if row is None:
            return None
        position = await self._queue_position_db(session, row[1])
        _, events = expand_run_event(
            "inqtrix.run.queued",
            {
                "status": "queued",
                "queue_position": position,
                "resumed": True,
            },
            status=RunStatus.QUEUED.value,
        )
        await self._append_events_db(session, parent_run_id, row[0], events)
        return parent_run_id

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
                        runs.c.workspace_id,
                        runs.c.kind,
                        runs.c.parent_run_id,
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
            await self._project_child_progress_db(
                session,
                child_run_id=run_id,
                kind=str(row[6] or "standard"),
                parent_run_id=row[7],
                request_payload=dict(row[1] or {}),
                run_status=RunStatus.RUNNING.value,
                event_type="inqtrix.run.started",
                payload={"status": "running", "snapshot": dict(row[2] or {})},
                snapshot=dict(row[2] or {}),
                attempt=int(row[0]),
            )
            return ClaimedRun(
                run_id=run_id,
                tenant_id=tenant_id,
                attempt=int(row[0]),
                request_payload=dict(row[1] or {}),
                kind=str(row[6] or "standard"),
                created_by_sub=row[3],
                created_by_tenant_id=row[4],
                workspace_id=row[5],
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

    async def _dispatch_status_db(
        self, run_id: str, tenant_id: str
    ) -> str | None:
        async with self._session(tenant_id) as session:
            return (
                await session.execute(
                    select(runs.c.status).where(runs.c.run_id == run_id)
                )
            ).scalar_one_or_none()

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
        woken_parents: list[str] = []
        if self._sweep_orphans:
            self._sweep_orphans = False
            woken_parents.extend(await self._recover_orphans_db(session))
        # Waiting TTL: a parked run is auto-cancelled with a visible
        # reason naming WHAT was awaited — a human decision
        # (``approval_timeout``) or child runs that never all ended
        # (``children_timeout``) — never silently forever (memory-store
        # parity). Runs BEFORE the stuck-row failsafe so a timed-out
        # wait goes terminal cleanly instead of being deleted as stuck.
        # One sweep per reason class: RETURNING reflects the row AFTER
        # the update, so the pre-update status must be part of the
        # WHERE, not read back.
        timed_out: list[tuple[Any, str]] = []
        for statuses, reason in (
            (
                (
                    RunStatus.WAITING_FOR_APPROVAL.value,
                    RunStatus.WAITING_FOR_INPUT.value,
                ),
                "approval_timeout",
            ),
            ((RunStatus.WAITING_FOR_CHILDREN.value,), "children_timeout"),
        ):
            swept = (
                (
                    await session.execute(
                        update(runs)
                        .where(
                            runs.c.status.in_(statuses),
                            runs.c.waiting_since.isnot(None),
                            runs.c.waiting_since
                            < time.time() - self._waiting_ttl_seconds,
                        )
                        .values(
                            status=RunStatus.CANCELLED.value,
                            cancel_requested=True,
                            waiting_since=None,
                            finished_at=time.time(),
                        )
                        .returning(
                            runs.c.run_id,
                            runs.c.tenant_id,
                            runs.c.kind,
                            runs.c.parent_run_id,
                            runs.c.request_payload,
                            runs.c.snapshot,
                            runs.c.attempt,
                        )
                    )
                )
            ).all()
            timed_out.extend((row, reason) for row in swept)
        for waited, reason in timed_out:
            waited_id = waited[0]
            log.warning(
                "Run %s wartete laenger als %d Sekunden und wurde "
                "automatisch abgebrochen (%s).",
                waited_id,
                int(self._waiting_ttl_seconds),
                reason,
            )
            payload = {
                "status": "cancelled",
                "reason": reason,
                "snapshot": dict(waited[5] or {}),
            }
            woken_parent = await self._record_terminal_run_db(
                session,
                run_id=waited_id,
                tenant_id=waited[1],
                kind=str(waited[2] or "standard"),
                parent_run_id=waited[3],
                request_payload=dict(waited[4] or {}),
                status=RunStatus.CANCELLED.value,
                event_type="inqtrix.run.cancelled",
                payload=payload,
                snapshot=dict(waited[5] or {}),
                attempt=int(waited[6] or 0) or None,
            )
            if woken_parent:
                woken_parents.append(woken_parent)
        self._register_cleanup_handoffs(
            session,
            swept_run_ids=[row[0] for row, _reason in timed_out],
            woken_parent_ids=woken_parents,
        )
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
        # payloads — user conversation content — indefinitely. Waiting
        # rows are NOT stuck: they are excluded here because they have
        # their own, VISIBLE lifecycle end (the waiting-TTL sweep above
        # cancels them, then the terminal TTL deletes them) — this
        # DELETE keying on created_at would otherwise fire first and
        # erase a parked run without any event.
        stuck = (
            (
                await session.execute(
                    delete(runs)
                    .where(
                        runs.c.status.notin_(_TERMINAL_VALUES),
                        runs.c.status.notin_(_WAITING_VALUES),
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

    async def _recover_orphans_db(
        self, session: "AsyncSession"
    ) -> list[str]:
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
                .returning(
                    runs.c.run_id,
                    runs.c.tenant_id,
                    runs.c.kind,
                    runs.c.parent_run_id,
                    runs.c.request_payload,
                    runs.c.snapshot,
                    runs.c.attempt,
                )
            )
        ).all()
        woken_parents: list[str] = []
        for row in rows:
            run_id = row[0]
            log.warning(
                "Verwaister Run %s nach Neustart als fehlgeschlagen "
                "markiert.",
                run_id,
            )
            payload = {
                "status": "failed",
                "error": error,
                "snapshot": dict(row[5] or {}),
            }
            woken_parent = await self._record_terminal_run_db(
                session,
                run_id=run_id,
                tenant_id=row[1],
                kind=str(row[2] or "standard"),
                parent_run_id=row[3],
                request_payload=dict(row[4] or {}),
                status=RunStatus.FAILED.value,
                event_type="inqtrix.run.failed",
                payload=payload,
                snapshot=dict(row[5] or {}),
                attempt=int(row[6] or 0) or None,
            )
            if woken_parent:
                woken_parents.append(woken_parent)
        return woken_parents

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
        kind=row["kind"] or "standard",
        parent_run_id=row["parent_run_id"],
        root_run_id=row["root_run_id"],
        session_id=row["session_id"],
        # Persisted inside the replay payload (no column): the child's
        # idempotency key for kernel tool re-execution (M2 step 8).
        origin_key=((row["request_payload"] or {}).get("body") or {}).get(
            "origin_key"
        ),
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
