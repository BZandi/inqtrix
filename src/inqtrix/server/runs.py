"""In-memory run registry and queue for native Inqtrix UI clients.

The OpenAI-compatible chat endpoint remains request/response oriented.
This module backs the native ``/v1/runs`` surface: it accepts research
jobs, caps active provider work, keeps a bounded FIFO queue, and exposes
short-lived event buffers for browser UIs.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, Mapping

from inqtrix.auth.permissions import SharePermission
from inqtrix.exceptions import RunNotFound
from inqtrix.execution_failures import terminate_native_run
from inqtrix.pagination import keyset_page
from inqtrix.runs.ports import RunStoreMetrics
from inqtrix.runs.shared import (
    CHILD_PROGRESS_EVENT,
    access_annotation,
    build_child_progress_payload,
    build_run_summary,
    expand_run_event,
    should_project_child_event,
    status_value,
)
from inqtrix.runtime_logging import new_run_id

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext
from inqtrix.settings import ServerSettings
from inqtrix.text import iter_word_chunks
from inqtrix.urls import sanitize_error

log = logging.getLogger("inqtrix")

RunWork = Callable[["RunHandle"], None]


class RunStatus(StrEnum):
    """Lifecycle status for a native in-memory run.

    The ``waiting_*`` statuses are the parked, NON-terminal states of
    an agent run suspended mid-graph: the run holds no execution slot,
    is excluded from orphan/stuck sweeps (it is legitimately idle), and
    resumes through ``resume_run``. ``waiting_for_approval`` and
    ``waiting_for_input`` wait on a human decision;
    ``waiting_for_children`` waits on the terminal write of the last
    child research run the agent submitted (the store wakes the parent
    itself — no human involved). An unanswered wait is auto-cancelled
    after ``waiting_ttl_seconds`` (visible, reason ``approval_timeout``
    or ``children_timeout``) — never silently forever.
    """

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_CHILDREN = "waiting_for_children"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)

WAITING_RUN_STATUSES = frozenset(
    {
        RunStatus.WAITING_FOR_APPROVAL,
        RunStatus.WAITING_FOR_INPUT,
        RunStatus.WAITING_FOR_CHILDREN,
    }
)


class RunQueueFull(RuntimeError):
    """Raised when the native run queue has no free slot."""


class RunPerUserLimit(RunQueueFull):
    """Raised when the submitting subject hit its in-flight run cap.

    Subclass of :class:`RunQueueFull` so every existing 429 path keeps
    working; the router distinguishes it to tell the caller THEIR cap
    was hit (retry after own runs finish), not the shared queue.
    Counted statuses are QUEUED+RUNNING — a parked (waiting) run holds
    no execution slot and must not eat its owner's fairness budget.
    """


class RunActive(RuntimeError):
    """Raised when a delete targets a run that is still queued or running.

    Deletion is terminal-only: removing a record an executing worker still
    holds would let its final write resurrect a half-gone run. The caller
    cancels first, then deletes once the run reaches a terminal state.
    """


class RunSessionActive(RuntimeError):
    """Raised when a session already has an active root agent run."""


class RunParentInactive(RuntimeError):
    """Raised when a child submit races a cancelled or terminal ancestor."""


@dataclass
class RunRecord:
    """Mutable server-side state for one native run."""

    run_id: str
    question: str
    stack_name: str
    workspace_id: str | None
    created_at: float
    work: RunWork | None = field(repr=False)
    created_by_sub: str | None = None
    """Verified subject that submitted the run (authorization fact,
    server-resolved from the principal — unlike ``workspace_id``,
    which is the client-supplied UI namespace). ``None`` only for
    records created before the field existed."""
    created_by_tenant_id: str | None = None
    """Tenant of the submitting principal. Carried alongside the sub
    because OIDC subjects are only unique per issuer/tenant — a sub
    collision across tenants must never grant visibility."""
    agent_overrides: dict[str, Any] = field(default_factory=dict)
    mode: str = "research"
    kind: str = "standard"
    """Run role in an agent tree: ``standard`` (every historical run),
    ``agent`` (a workspace-agent root) or ``agent_child`` (a research
    run spawned by an agent task). Summaries omit the default."""
    parent_run_id: str | None = None
    """The spawning agent run for ``agent_child`` rows; ``None`` else."""
    root_run_id: str | None = None
    """The tree root (== parent for one-level trees); ``None`` else."""
    session_id: str | None = None
    """Agent-desk session this run belongs to; ``None`` else."""
    origin_key: str | None = None
    """Idempotency key of the SUBMITTING tool call for agent children
    (M2 step 8): a kernel tool re-executing after park/resume finds its
    already-submitted child by this key instead of spawning a second
    one. Summaries expose it only when set."""
    parent_task_id: str | None = None
    """Internal plan-task correlation copied from the durable replay body.
    It is never part of the public run summary; the store uses it only to
    project child progress onto the parent run's task card."""
    parent_task_attempt: int | None = None
    """Logical parent-task attempt; independent of run claim fencing."""
    status: RunStatus = RunStatus.QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    finished_monotonic: float | None = None
    waiting_since: float | None = None
    """Unix time the run entered a waiting status (TTL anchor);
    cleared on resume."""
    park_in_flight: bool = False
    """Set between ``mark_waiting`` and the parking worker's unwind;
    a resume in that window defers its dispatch (``resume_requested``)
    to the unwind instead of racing the still-live worker."""
    resume_requested: bool = False
    """A resume arrived while ``park_in_flight``; the parking worker's
    unwind performs the deferred re-dispatch."""
    snapshot: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    event_seq: int = 0
    events: deque[dict[str, Any]] = field(default_factory=deque, repr=False)
    subscribers: list[Queue] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class RunSubscription:
    """Live event subscription with buffered replay."""

    run_id: str
    queue: Queue
    replay: list[dict[str, Any]]
    store: "RunStore"

    def close(self) -> None:
        """Detach the subscriber queue from the run store."""
        self.store.unsubscribe(self.run_id, self.queue)


class RunHandle:
    """Worker-side handle for updating one run without exposing the store."""

    def __init__(self, store: "RunStore", run_id: str, cancel_event: threading.Event) -> None:
        self._store = store
        self.run_id = run_id
        self.cancel_event = cancel_event
        self.parked = False
        """This execution parked its run via :meth:`wait` — the worker
        loop reads it to skip the auto-complete safety net and to
        finish the park handoff in its unwind."""

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Emit one structured event for this run."""
        self._store.emit(self.run_id, event_type, payload or {})

    def emit_answer(self, answer: str) -> None:
        """Emit final answer text as word-aligned output delta events."""
        for chunk in iter_word_chunks(answer or ""):
            self.emit("inqtrix.output_text.delta", {"delta": chunk})

    def complete(self, result: dict[str, Any], snapshot: dict[str, Any] | None = None) -> None:
        """Mark the run completed and store its short-lived result payload."""
        self._store.complete(self.run_id, result, snapshot=snapshot)

    def fail(self, message: str, *, error_type: str = "server_error") -> None:
        """Mark the run failed."""
        self._store.fail(self.run_id, message, error_type=error_type)

    def cancel(self, reason: str = "cancelled") -> None:
        """Mark the run cancelled after the worker observed the cancel event."""
        self._store.mark_cancelled(self.run_id, reason=reason)

    def wait(self, status: "RunStatus | str") -> None:
        """Park the run in a waiting status (agent interrupt observed).

        The executing closure returns right after calling this; the
        store keeps the closure for :meth:`RunStore.resume_run`. When a
        cancel request is already pending, the run is cancelled instead
        of parked (the closure still returns normally) — a cancelled
        assignment must not sit in a waiting status until its TTL.
        """
        self._store.mark_waiting(self.run_id, status=status)
        self.parked = True


class RunStore:
    """Thread-safe in-memory queue and registry for native run endpoints.

    Args:
        max_concurrent: Maximum number of actively executing research jobs.
            Additional accepted jobs stay in the FIFO queue.
        max_queue_size: Maximum number of waiting jobs. Active jobs do not
            count against this number.
        completed_ttl_seconds: How long terminal records remain queryable
            after completion. Queued/running runs are never TTL-evicted.
        event_buffer_size: Number of recent events retained per run for
            late SSE subscribers.
        waiting_ttl_seconds: How long a run may sit in a waiting status
            before it is auto-cancelled with the visible
            ``approval_timeout`` reason. Default seven days — a parked
            approval must never linger silently forever.
    """

    def __init__(
        self,
        *,
        max_concurrent: int,
        max_queue_size: int,
        completed_ttl_seconds: int,
        event_buffer_size: int,
        waiting_ttl_seconds: float = 7 * 24 * 3600.0,
        max_concurrent_per_user: int | None = None,
    ) -> None:
        self._max_concurrent = _require_minimum(
            "max_concurrent", max_concurrent, minimum=1
        )
        self._max_queue_size = _require_minimum(
            "max_queue_size", max_queue_size, minimum=0
        )
        self._completed_ttl_seconds = _require_minimum(
            "completed_ttl_seconds", completed_ttl_seconds, minimum=0
        )
        self._event_buffer_size = _require_minimum(
            "event_buffer_size", event_buffer_size, minimum=1
        )
        if float(waiting_ttl_seconds) <= 0:
            raise ValueError(
                f"waiting_ttl_seconds must be > 0, got {waiting_ttl_seconds}"
            )
        self._waiting_ttl_seconds = float(waiting_ttl_seconds)
        if max_concurrent_per_user is not None:
            _require_minimum(
                "max_concurrent_per_user", max_concurrent_per_user, minimum=1
            )
        self._max_concurrent_per_user = max_concurrent_per_user
        self._records: dict[str, RunRecord] = {}
        self._pending: deque[str] = deque()
        self._running_count = 0
        self._lock = threading.RLock()

    @classmethod
    def from_settings(cls, settings: ServerSettings) -> "RunStore":
        """Build a run store from HTTP server settings."""
        return cls(
            max_concurrent=settings.run_max_concurrent or settings.max_concurrent,
            max_queue_size=settings.run_queue_max_size,
            completed_ttl_seconds=settings.run_completed_ttl_seconds,
            event_buffer_size=settings.run_event_buffer_size,
            max_concurrent_per_user=settings.run_max_concurrent_per_user,
        )

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
        """Create a queued run and dispatch it if capacity is available.

        Args:
            request_payload: Re-execution payload persisted by durable
                backends so worker processes can rebuild the run from
                the row alone. Deliberately ignored here — in-memory
                execution keeps the work closure in-process.
            kind: Run role in an agent tree (``standard`` default keeps
                every historical caller byte-identical).
            parent_run_id: Spawning agent run for ``agent_child`` rows.
            root_run_id: Tree root for child rows.
            session_id: Agent-desk session grouping.

        Returns:
            Public run summary suitable for HTTP responses.

        Raises:
            RunQueueFull: When the waiting queue is already full.
        """
        request_body = (
            (request_payload or {}).get("body")
            if isinstance(request_payload, dict)
            else {}
        )
        parent_task_id = (
            str(request_body.get("parent_task_id") or "")
            if isinstance(request_body, dict)
            else ""
        )
        parent_task_attempt = (
            int(request_body.get("parent_task_attempt", 0) or 0)
            if isinstance(request_body, dict)
            else 0
        )
        del request_payload
        with self._lock:
            self._cleanup_locked()
            if kind == "agent_child":
                if not parent_run_id:
                    raise RunParentInactive("agent child has no parent run")
                parent = self._records.get(parent_run_id)
                if parent is None:
                    raise RunParentInactive("agent child parent is missing")
                canonical_root_id = parent.root_run_id or parent.run_id
                root = self._records.get(canonical_root_id)
                if root is None:
                    raise RunParentInactive("agent child root is missing")
                if origin_key:
                    existing = next(
                        (
                            record
                            for record in self._records.values()
                            if record.kind == "agent_child"
                            and record.parent_run_id == parent_run_id
                            and record.origin_key == origin_key
                        ),
                        None,
                    )
                    if existing is not None:
                        return self._summary_locked(existing)
                blocked = (
                    parent.status in TERMINAL_RUN_STATUSES
                    or parent.status is RunStatus.EXPIRED
                    or parent.cancel_event.is_set()
                    or root.status in TERMINAL_RUN_STATUSES
                    or root.status is RunStatus.EXPIRED
                    or root.cancel_event.is_set()
                )
                if blocked:
                    raise RunParentInactive(
                        "agent child parent or root is no longer active"
                    )
                # The parent row, not a caller-provided value, owns lineage.
                root_run_id = canonical_root_id
            if kind == "agent" and parent_run_id is None and session_id:
                active_statuses = {
                    RunStatus.QUEUED,
                    RunStatus.RUNNING,
                    *WAITING_RUN_STATUSES,
                }
                if any(
                    record.kind == "agent"
                    and record.parent_run_id is None
                    and record.session_id == session_id
                    and record.status in active_statuses
                    for record in self._records.values()
                ):
                    raise RunSessionActive(session_id)
            if len(self._pending) >= self._max_queue_size and self._running_count >= self._max_concurrent:
                raise RunQueueFull("native run queue is full")
            if (
                self._max_concurrent_per_user is not None
                and created_by_sub is not None
            ):
                # Fairness bound UNDER the global cap: a recount over the
                # (TTL-bounded) records, leak-free by construction and
                # EXACT here (the whole submit holds self._lock, so two
                # submits by one sub serialise — unlike the Postgres path,
                # which is an approximate bound under READ COMMITTED).
                #
                # What is COUNTED: QUEUED+RUNNING standard runs and agent
                # CHILDREN — the runs that actually occupy an execution
                # slot. EXCLUDED: WAITING runs (parked, slot-free) AND
                # agent PARENTS (kind='agent'), which park almost
                # immediately and would otherwise contend against their
                # OWN children for the user's budget (self-starvation).
                # So one agent tree costs the user its children, not the
                # orchestrator on top.
                #
                # Scope is created_by_sub only (not tenant): run storage
                # is single-tenant today; the (sub, tenant) identity pair
                # matters only for the reserved multi-tenant OIDC path,
                # tracked for when it lands.
                in_flight = sum(
                    1
                    for record in self._records.values()
                    if record.created_by_sub == created_by_sub
                    and record.kind != "agent"
                    and record.status
                    in (RunStatus.QUEUED, RunStatus.RUNNING)
                )
                if in_flight >= self._max_concurrent_per_user:
                    raise RunPerUserLimit(
                        "per-user in-flight run cap reached"
                    )

            run_id = self._new_unique_run_id_locked()
            record = RunRecord(
                run_id=run_id,
                question=question[:500],
                stack_name=stack_name,
                workspace_id=workspace_id,
                created_at=time.time(),
                created_by_sub=created_by_sub,
                created_by_tenant_id=created_by_tenant_id,
                work=work,
                agent_overrides=dict(agent_overrides or {}),
                mode=mode,
                kind=kind,
                parent_run_id=parent_run_id,
                root_run_id=root_run_id,
                session_id=session_id,
                origin_key=origin_key,
                parent_task_id=parent_task_id or None,
                parent_task_attempt=(parent_task_attempt or None),
            )
            record.events = deque(maxlen=self._event_buffer_size)
            self._records[run_id] = record
            self._pending.append(run_id)
            self._emit_locked(
                record,
                "inqtrix.run.queued",
                {
                    "status": "queued",
                    "queue_position": self._queue_position_locked(run_id),
                },
            )
            self._dispatch_locked()
            return self._summary_locked(record)

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
        """Persist an ALREADY-TERMINAL run carried in from a project file.

        Unlike :meth:`submit` (which queues a fresh run for execution), this
        stores a completed report snapshot directly so a loaded project's
        reports survive a reload + follow the user, scoped to the caller. The
        client-supplied ``run_id`` is kept when free so the local and server
        records stay aligned (no remap); a re-import of the caller's OWN run is
        an idempotent no-op (snapshots are immutable). If the id is already held
        by ANOTHER principal, a fresh id is allocated rather than overwriting or
        leaking that row (No Silent Fallbacks).

        ``created_at`` keeps the report's ORIGINAL date for display + ordering,
        but ``finished_at`` is set to the import time so the durable-retention
        clock starts now: a report older than the retention window must NOT be
        pruned on the next cleanup just because its original run finished long
        ago (that would re-lose it). ``finished_at`` is therefore not a
        parameter.

        Raises:
            ValueError: When *status* is not a terminal status.
        """
        terminal = _coerce_status(status)
        if terminal not in TERMINAL_RUN_STATUSES:
            raise ValueError(
                f"import_completed_run requires a terminal status, got {status!r}"
            )
        now = time.time()
        with self._lock:
            self._cleanup_locked()
            existing = self._records.get(run_id)
            if existing is not None and not (
                existing.created_by_sub == created_by_sub
                and existing.created_by_tenant_id == created_by_tenant_id
            ):
                log.warning(
                    "Imported run id %s already owned by another principal; "
                    "allocating a new id.",
                    run_id,
                )
                run_id = self._new_unique_run_id_locked()
                existing = None
            if existing is not None:
                return self._summary_locked(existing)
            record = RunRecord(
                run_id=run_id,
                question=question[:500],
                stack_name=stack_name,
                workspace_id=workspace_id,
                created_at=created_at if created_at is not None else now,
                created_by_sub=created_by_sub,
                created_by_tenant_id=created_by_tenant_id,
                work=None,
                agent_overrides=dict(agent_overrides or {}),
                mode=mode,
            )
            record.events = deque(maxlen=self._event_buffer_size)
            record.status = terminal
            # Retention clock = import time (not the original finish), so an old
            # report is kept the full window from when it was imported.
            record.started_at = now
            record.finished_at = now
            record.finished_monotonic = time.monotonic()
            record.snapshot = dict(snapshot or {})
            record.result = (
                dict(result) if terminal == RunStatus.COMPLETED else None
            )
            record.error = dict(error) if error else None
            self._records[run_id] = record
            return self._summary_locked(record)

    def owner_sub(self, run_id: str) -> str | None:
        """The run's creator regardless of visibility (share layer)."""
        with self._lock:
            record = self._records.get(run_id)
            return record.created_by_sub if record is not None else None

    def title(self, run_id: str) -> str | None:
        """The run's question as a share-surface title, regardless of
        visibility — a pending-share recipient must see it to decide. ``None``
        when the run no longer exists (e.g. pruned), so the inbox skips it."""
        with self._lock:
            record = self._records.get(run_id)
            return record.question if record is not None else None

    def get(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        """Return a public summary for *run_id*."""
        with self._lock:
            self._cleanup_locked()
            record, shared = self._record_locked(
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
            return self._summary_locked(record, shared=shared)

    def list(
        self,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[dict[str, Any]]:
        """Return public summaries for all in-memory runs.

        Shared-in runs join the listing REGARDLESS of the workspace
        namespace filter (they carry the grantor's workspace id) and
        carry the additive ``access`` annotation.
        """
        with self._lock:
            self._cleanup_locked()
            summaries = []
            for record in sorted(
                self._records.values(),
                # (created_at, run_id) matches list_page's keyset order so the
                # unbounded read and the paginated endpoint agree on ties.
                key=lambda item: (item.created_at, item.run_id),
                reverse=True,
            ):
                if _workspace_matches(
                    record, workspace_id
                ) and _visible_to_matches(record, visible_to):
                    summaries.append(self._summary_locked(record))
                    continue
                shared = (
                    also_visible.get(record.run_id)
                    if also_visible is not None
                    else None
                )
                if shared is not None:
                    summaries.append(
                        self._summary_locked(record, shared=shared)
                    )
            return summaries

    def list_session_runs(
        self,
        session_id: str,
        *,
        visible_to: "UserContext | None" = None,
    ) -> list[dict[str, Any]]:
        """Visible summaries of one agent session, oldest first (K1)."""
        with self._lock:
            self._cleanup_locked()
            return [
                self._summary_locked(record)
                for record in sorted(
                    (
                        record
                        for record in self._records.values()
                        if record.session_id == session_id
                        and _visible_to_matches(record, visible_to)
                    ),
                    key=lambda item: (item.created_at, item.run_id),
                )
            ]

    def session_owners(
        self, session_id: str
    ) -> set[tuple[str | None, str | None]]:
        """Return every recorded owner identity for ``session_id``."""
        with self._lock:
            self._cleanup_locked()
            return {
                (record.created_by_tenant_id, record.created_by_sub)
                for record in self._records.values()
                if record.session_id == session_id
            }

    def list_page(
        self,
        *,
        limit: int,
        after: tuple[float, str] | None = None,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """One keyset page of visible runs + the next cursor (wire parity).

        Mirrors the durable store's ``list_page`` wire shape: filter-before-
        slice so pages stay full, newest-first over ``(created_at, run_id)``.
        Note this in-process tier sorts the whole visible set per call
        (O(n log n)); the durable store is the truly index-bounded path, and
        multi-replica scale requires that backend anyway (the in-memory store
        is per-pod). The page bound here caps the RESPONSE size, not the scan.
        """
        with self._lock:
            self._cleanup_locked()
            visible: list[tuple[RunRecord, "SharePermission | None"]] = []
            for record in self._records.values():
                if _workspace_matches(
                    record, workspace_id
                ) and _visible_to_matches(record, visible_to):
                    visible.append((record, None))
                    continue
                shared = (
                    also_visible.get(record.run_id)
                    if also_visible is not None
                    else None
                )
                if shared is not None:
                    visible.append((record, shared))
            visible.sort(
                key=lambda item: (item[0].created_at, item[0].run_id),
                reverse=True,
            )
            page, next_cursor = keyset_page(
                visible,
                limit=limit,
                after=after,
                created_at_of=lambda item: item[0].created_at,
                id_of=lambda item: item[0].run_id,
            )
            summaries = [
                self._summary_locked(record, shared=shared)
                for record, shared in page
            ]
            return summaries, next_cursor

    def metrics_snapshot(self) -> "RunStoreMetrics":
        """In-process queue load under the store lock (see the port)."""
        with self._lock:
            return RunStoreMetrics(
                queued=len(self._pending),
                active=self._running_count,
                capacity=self._max_concurrent,
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
        with self._lock:
            self._cleanup_locked()
            record, _shared = self._record_locked(
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
            if record.result is None:
                raise RunNotFound(run_id)
            return {
                "run_id": run_id,
                "status": record.status.value,
                **record.result,
            }

    def cancel(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        """Request cancellation for a queued or running run.

        Shared-in recipients need at least an ``edit`` grant — a
        view-only invitee watching a run must not be able to kill it;
        the denial is the indistinct 404.
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
        """Cancel one run tree and return ids touched under the store lock."""
        with self._lock:
            self._cleanup_locked()
            record, shared = self._record_locked(
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
            if shared is not None and not shared.at_least(
                SharePermission.EDIT
            ):
                raise RunNotFound(run_id)
            summary = self._cancel_record_locked(record)
            # Walk the actual parent links so nested Kernel -> Mission ->
            # Research trees are cancelled as one unit. The store lock also
            # serializes child admission against this traversal.
            frontier = [run_id]
            seen = {run_id}
            affected = [run_id]
            while frontier:
                parent_id = frontier.pop()
                children = [
                    child
                    for child in self._records.values()
                    if child.parent_run_id == parent_id
                    and child.run_id not in seen
                ]
                for child in children:
                    seen.add(child.run_id)
                    affected.append(child.run_id)
                    frontier.append(child.run_id)
                    self._cancel_record_locked(child)
            return summary, tuple(affected)

    def _cancel_record_locked(self, record: RunRecord) -> dict[str, Any]:
        """Cancel one record in place (queued/waiting/running semantics)."""
        if record.status == RunStatus.QUEUED:
            self._remove_pending_locked(record.run_id)
            record.cancel_event.set()
            self._mark_terminal_locked(record, RunStatus.CANCELLED)
            self._emit_locked(
                record,
                "inqtrix.run.cancelled",
                {"status": "cancelled", "reason": "cancelled_before_start"},
            )
            record.work = None
            return self._summary_locked(record)
        if record.status in WAITING_RUN_STATUSES:
            # Nothing is executing — the wait resolves immediately.
            record.cancel_event.set()
            self._mark_terminal_locked(record, RunStatus.CANCELLED)
            record.waiting_since = None
            self._emit_locked(
                record,
                "inqtrix.run.cancelled",
                {"status": "cancelled", "reason": "cancelled_while_waiting"},
            )
            record.work = None
            return self._summary_locked(record)
        if record.status == RunStatus.RUNNING:
            record.cancel_event.set()
            self._emit_locked(
                record,
                "inqtrix.run.cancel_requested",
                {"status": "running", "reason": "client_requested_cancel"},
            )
            return self._summary_locked(record)
        return self._summary_locked(record)

    def delete(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        requester_sub: str | None = None,
    ) -> None:
        """Permanently remove one terminal run (owner-only).

        Stronger than :meth:`cancel`: a shared-in recipient — even with an
        ``edit`` grant — must never delete the owner's run, so the gate is
        creator identity, not share visibility. A non-owner or a
        cross-namespace caller gets the indistinct ``RunNotFound`` (denial
        equals absence). Only terminal runs are deletable; an active run
        raises ``RunActive`` so the executing worker cannot write into a
        record that vanished mid-run.
        """
        with self._lock:
            self._cleanup_locked()
            record = self._records.get(run_id)
            if record is None:
                raise RunNotFound(run_id)
            if (
                (
                    record.created_by_sub is not None
                    and record.created_by_sub != requester_sub
                )
                or not _workspace_matches(record, workspace_id)
            ):
                # Owner-only for runs that HAVE a recorded creator; a legacy
                # pre-scoping run (created_by_sub is None) has no owner signal
                # but its workspace, so the namespace match alone gates it —
                # otherwise such a run would be undeletable by anyone.
                log.warning(
                    "authz denied: run %s delete refused for sub=%s",
                    run_id,
                    requester_sub or "",
                )
                raise RunNotFound(run_id)
            if record.status not in TERMINAL_RUN_STATUSES:
                raise RunActive(run_id)
            del self._records[run_id]

    def subscribe(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> RunSubscription:
        """Subscribe to a run's event stream, replaying buffered events."""
        with self._lock:
            self._cleanup_locked()
            record, _shared = self._record_locked(
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
            queue: Queue = Queue()
            record.subscribers.append(queue)
            return RunSubscription(
                run_id=run_id,
                queue=queue,
                replay=list(record.events),
                store=self,
            )

    def unsubscribe(self, run_id: str, queue: Queue) -> None:
        """Remove a queue from the subscriber list if it is still present."""
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return
            try:
                record.subscribers.remove(queue)
            except ValueError:
                return

    def emit(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        fence_attempt: int | None = None,
    ) -> None:
        """Emit one event to the run buffer and live subscribers.

        Terminal runs accept no further events: the log's ends-terminal
        invariant is what tells SSE streams (live and replayed) to stop.
        A post-terminal signal (e.g. a user editing an artifact after the
        run finished) is dropped LOUDLY — clients of finished runs
        reconcile via the REST reads, not the closed event stream.

        Args:
            fence_attempt: Accepted for :class:`RunStorePort` parity;
                ignored (see :meth:`mark_waiting` — the in-process store
                has no claim/reclaim to fence).
        """
        with self._lock:
            record, _shared = self._record_locked(run_id)
            if record.status in TERMINAL_RUN_STATUSES:
                log.warning(
                    "Event %s fuer Run %s verworfen: der Lauf ist bereits "
                    "terminal (%s), das Event-Log endet mit dem "
                    "Terminal-Event.",
                    event_type,
                    run_id,
                    record.status.value,
                )
                return
            self._emit_locked(record, event_type, payload or {})

    def complete(
        self,
        run_id: str,
        result: dict[str, Any],
        *,
        snapshot: dict[str, Any] | None = None,
        fence_attempt: int | None = None,
    ) -> None:
        """Store the final result and mark the run completed.

        Args:
            fence_attempt: Accepted for :class:`RunStorePort` parity;
                ignored (see :meth:`mark_waiting`).
        """
        with self._lock:
            record, _shared = self._record_locked(run_id)
            if record.status in TERMINAL_RUN_STATUSES:
                return
            if snapshot:
                record.snapshot = dict(snapshot)
            record.result = dict(result)
            self._mark_terminal_locked(record, RunStatus.COMPLETED)
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            self._emit_locked(
                record,
                "inqtrix.run.completed",
                {
                    "status": "completed",
                    "metrics": metrics,
                    "result_url": f"/v1/runs/{run_id}/result",
                    "snapshot": record.snapshot,
                },
            )

    def fail(
        self,
        run_id: str,
        message: str,
        *,
        error_type: str = "server_error",
        fence_attempt: int | None = None,
    ) -> None:
        """Mark a run failed with a sanitized error payload.

        Args:
            fence_attempt: Accepted for :class:`RunStorePort` parity;
                ignored (see :meth:`mark_waiting`).
        """
        with self._lock:
            record, _shared = self._record_locked(run_id)
            if record.status in TERMINAL_RUN_STATUSES:
                return
            record.error = {
                "message": sanitize_error(message),
                "type": error_type,
            }
            self._mark_terminal_locked(record, RunStatus.FAILED)
            self._emit_locked(
                record,
                "inqtrix.run.failed",
                {"status": "failed", "error": record.error, "snapshot": record.snapshot},
            )

    def mark_cancelled(
        self, run_id: str, *, reason: str, fence_attempt: int | None = None
    ) -> None:
        """Mark a running run cancelled after its worker exits.

        Args:
            fence_attempt: Accepted for :class:`RunStorePort` parity;
                ignored (see :meth:`mark_waiting`).
        """
        with self._lock:
            record, _shared = self._record_locked(run_id)
            if record.status in TERMINAL_RUN_STATUSES:
                return
            self._mark_terminal_locked(record, RunStatus.CANCELLED)
            self._emit_locked(
                record,
                "inqtrix.run.cancelled",
                {"status": "cancelled", "reason": reason, "snapshot": record.snapshot},
            )

    def mark_waiting(
        self,
        run_id: str,
        *,
        status: RunStatus | str,
        fence_attempt: int | None = None,
    ) -> None:
        """Park a RUNNING run in a waiting status (agent interrupt).

        The executing closure returns after calling this; the run keeps
        its work closure so :meth:`resume_run` can re-dispatch it. Only
        waiting statuses are accepted, and only from RUNNING — anything
        else is a caller bug, raised loudly. Parking on children
        immediately self-heals the lost-wakeup race: when every child is
        ALREADY terminal at park time (the last one finished while this
        run was still unwinding towards the park), the run flips straight
        back to QUEUED instead of waiting for a wake-up that fired.

        Args:
            fence_attempt: Accepted for :class:`RunStorePort` parity with
                the durable backend (the worker calls both stores through
                the port and always passes it). Deliberately unused here:
                the in-process store executes closures on a single owner
                with no claim/reclaim, so there is no zombie attempt to
                fence out.
        """
        waiting = _coerce_status(status)
        if waiting not in WAITING_RUN_STATUSES:
            raise ValueError(f"not a waiting status: {status!r}")
        with self._lock:
            record, _shared = self._record_locked(run_id)
            if record.status is not RunStatus.RUNNING:
                raise RunActive(
                    f"run {run_id} cannot wait from status {record.status.value}"
                )
            if record.cancel_event.is_set():
                # A cancel is already pending: parking would leave a
                # cancelled assignment sitting in a waiting status until
                # its TTL. Resolve the cancel now instead; the closure
                # returns normally either way.
                self._mark_terminal_locked(record, RunStatus.CANCELLED)
                self._emit_locked(
                    record,
                    "inqtrix.run.cancelled",
                    {
                        "status": "cancelled",
                        "reason": "cancelled_while_waiting",
                        "snapshot": record.snapshot,
                    },
                )
                return
            record.status = waiting
            record.waiting_since = time.time()
            record.park_in_flight = True
            self._emit_locked(
                record,
                "inqtrix.run.waiting",
                {"status": waiting.value, "snapshot": record.snapshot},
            )
            if waiting is RunStatus.WAITING_FOR_CHILDREN:
                # Lost-wakeup self-heal: the last child may have gone
                # terminal BEFORE this park landed — its wake probe then
                # found the parent still RUNNING and no-oped. Re-probe
                # now; park_in_flight is set, so a hit defers the
                # re-dispatch to the unwinding worker.
                self._wake_parent_if_children_done_locked(run_id)

    def resume_run(self, run_id: str) -> dict[str, Any]:
        """Move a waiting run back to QUEUED and dispatch it.

        The decision endpoints (approval/clarification) call this after
        recording the decision; execution resumes as a fresh dispatch of
        the retained work closure (the M5 closure re-enters at its
        checkpoint).

        Raises:
            RunNotFound: Unknown id.
            RunActive: The run is not in a waiting status.
        """
        with self._lock:
            record, _shared = self._record_locked(run_id)
            if record.status not in WAITING_RUN_STATUSES:
                raise RunActive(
                    f"run {run_id} is not waiting (status {status_value(record.status)})"
                )
            if record.work is None:
                # In-memory closures never survive a process restart; a
                # waiting record without work is unresumable and failing
                # loudly beats a silent hang.
                raise RunActive(f"run {run_id} has no retained work to resume")
            record.status = RunStatus.QUEUED
            record.waiting_since = None
            if record.park_in_flight:
                # The parking worker has not unwound yet: dispatching
                # now would run the same closure on two threads. The
                # unwind performs the deferred re-dispatch.
                record.resume_requested = True
            else:
                self._pending.append(run_id)
            self._emit_locked(
                record,
                "inqtrix.run.queued",
                {
                    "status": "queued",
                    "queue_position": self._queue_position_locked(run_id),
                    "resumed": True,
                },
            )
            self._dispatch_locked()
            return self._summary_locked(record)

    def children(self, run_id: str) -> list[dict[str, Any]]:
        """Summaries of this run's direct children, newest first.

        Authorization happens on the PARENT (the route resolves it via
        :meth:`get` before calling this); children inherit that access
        (plan rule R7), so no per-child visibility filter here.
        """
        with self._lock:
            self._cleanup_locked()
            records = sorted(
                (
                    record
                    for record in self._records.values()
                    if record.parent_run_id == run_id
                ),
                key=lambda record: record.created_at,
                reverse=True,
            )
            return [self._summary_locked(record) for record in records]

    def _run_worker(self, run_id: str, work: RunWork, cancel_event: threading.Event) -> None:
        handle = RunHandle(self, run_id, cancel_event)
        crashed = False
        try:
            work(handle)
            with self._lock:
                record = self._records.get(run_id)
                if (
                    record is not None
                    and not handle.parked
                    and record.status not in TERMINAL_RUN_STATUSES
                    and record.status not in WAITING_RUN_STATUSES
                ):
                    # ``handle.parked`` guards the resumed-in-window
                    # case: a resume may have flipped the run back to
                    # QUEUED before this line — completing it here
                    # would destroy the interrupt.
                    self._mark_terminal_locked(record, RunStatus.COMPLETED)
                    self._emit_locked(
                        record,
                        "inqtrix.run.completed",
                        {"status": "completed", "snapshot": record.snapshot},
                    )
        except Exception as exc:  # noqa: BLE001 - run workers must terminate cleanly
            crashed = True
            log.exception("Native run %s failed", run_id)
            terminate_native_run(handle, exc)
        finally:
            # The park handoff completes HERE: a resume that arrived
            # before this unwind parked its request in
            # ``resume_requested`` and is dispatched now.
            with self._lock:
                record = self._records.get(run_id)
                if record is not None:
                    parked_alive = (
                        handle.parked
                        and not crashed
                        and (
                            record.status in WAITING_RUN_STATUSES
                            or (
                                record.status is RunStatus.QUEUED
                                and record.resume_requested
                            )
                        )
                    )
                    if parked_alive:
                        # A parked run keeps its closure — resume_run
                        # re-dispatches the same segment-aware work.
                        record.park_in_flight = False
                        if record.resume_requested:
                            record.resume_requested = False
                            self._pending.append(run_id)
                    else:
                        record.work = None
                        record.park_in_flight = False
                        record.resume_requested = False
                self._running_count = max(0, self._running_count - 1)
                self._dispatch_locked()

    def _dispatch_locked(self) -> None:
        while self._running_count < self._max_concurrent and self._pending:
            run_id = self._pending.popleft()
            record = self._records.get(run_id)
            if record is None or record.status != RunStatus.QUEUED or record.work is None:
                continue
            record.status = RunStatus.RUNNING
            record.started_at = time.time()
            self._running_count += 1
            self._emit_locked(
                record,
                "inqtrix.run.started",
                {"status": "running", "snapshot": record.snapshot},
            )
            thread = threading.Thread(
                target=self._run_worker,
                args=(run_id, record.work, record.cancel_event),
                name=f"inqtrix-run-{run_id}",
                daemon=True,
            )
            thread.start()

    def _cleanup_locked(self) -> None:
        now = time.monotonic()
        # A wait must never linger silently forever: past the TTL the run
        # is auto-cancelled with a visible reason (event + warning), then
        # ages out via the normal terminal TTL. The reason names WHAT was
        # awaited: a human decision (``approval_timeout``) or child runs
        # that never all terminated (``children_timeout``).
        wall_now = time.time()
        for record in list(self._records.values()):
            if (
                record.status in WAITING_RUN_STATUSES
                and record.waiting_since is not None
                and (wall_now - record.waiting_since) > self._waiting_ttl_seconds
            ):
                reason = (
                    "children_timeout"
                    if record.status is RunStatus.WAITING_FOR_CHILDREN
                    else "approval_timeout"
                )
                log.warning(
                    "Run %s wartete laenger als %d s und wurde automatisch "
                    "abgebrochen (%s).",
                    record.run_id,
                    int(self._waiting_ttl_seconds),
                    reason,
                )
                record.cancel_event.set()
                self._mark_terminal_locked(record, RunStatus.CANCELLED)
                record.waiting_since = None
                record.work = None
                self._emit_locked(
                    record,
                    "inqtrix.run.cancelled",
                    {"status": "cancelled", "reason": reason},
                )
        expired = [
            run_id
            for run_id, record in self._records.items()
            if record.status in TERMINAL_RUN_STATUSES
            and record.finished_monotonic is not None
            and (now - record.finished_monotonic) > self._completed_ttl_seconds
            and not record.subscribers
        ]
        for run_id in expired:
            del self._records[run_id]

    def _record_locked(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[RunRecord, "SharePermission | None"]:
        """The record plus the share grant that admitted it (if any).

        Shared-in runs BYPASS the workspace namespace filter — they
        carry the GRANTOR's workspace id, which would otherwise hide
        every shared run from recipients filtering their own
        namespace.
        """
        record = self._records.get(run_id)
        if record is None:
            raise RunNotFound(run_id)
        shared = (
            also_visible.get(run_id) if also_visible is not None else None
        )
        if _visible_to_matches(record, visible_to):
            if not _workspace_matches(record, workspace_id):
                raise RunNotFound(run_id)
            return record, None
        if shared is not None:
            return record, shared
        # The client sees the indistinct 404; the denial itself must
        # stay operator-visible (Designprinzip 1). Persisting it to
        # the audit log arrives with the durable run port — this
        # store is sync/threaded, the audit sink is async.
        log.warning(
            "authz denied: run %s hidden from sub=%s tenant=%s",
            run_id,
            visible_to.principal.sub if visible_to else "",
            visible_to.principal.tenant_id if visible_to else "",
        )
        raise RunNotFound(run_id)

    def _new_unique_run_id_locked(self) -> str:
        for _ in range(8):
            run_id = new_run_id()
            if run_id not in self._records:
                return run_id
            log.warning("Native run id collision detected; retrying allocation.")
        raise RuntimeError("could not allocate a unique native run id")

    def _mark_terminal_locked(self, record: RunRecord, status: RunStatus | str) -> None:
        record.status = _coerce_status(status)
        record.finished_at = time.time()
        record.finished_monotonic = time.monotonic()
        # Every terminal transition of an agent child funnels through
        # here (complete/fail/cancel/TTL alike), so this is THE choke
        # point for waking a parent parked on its children.
        if record.kind == "agent_child" and record.parent_run_id:
            self._wake_parent_if_children_done_locked(record.parent_run_id)

    def _wake_parent_if_children_done_locked(self, parent_run_id: str) -> None:
        """Resume a children-parked parent once its last child ended.

        No-ops unless the parent currently sits in
        ``waiting_for_children`` AND every sibling is terminal — the
        CAS-like status check makes at-least-once invocation safe (a
        second call finds the parent already QUEUED). The park handshake
        is honored exactly like :meth:`resume_run`: a parent whose
        parking worker has not unwound yet defers its re-dispatch to
        that unwind via ``resume_requested``.
        """
        parent = self._records.get(parent_run_id)
        if parent is None or parent.status is not RunStatus.WAITING_FOR_CHILDREN:
            return
        if any(
            sibling.parent_run_id == parent_run_id
            and sibling.status not in TERMINAL_RUN_STATUSES
            for sibling in self._records.values()
        ):
            return
        parent.status = RunStatus.QUEUED
        parent.waiting_since = None
        if parent.park_in_flight:
            parent.resume_requested = True
        else:
            self._pending.append(parent_run_id)
        self._emit_locked(
            parent,
            "inqtrix.run.queued",
            {
                "status": "queued",
                "queue_position": self._queue_position_locked(parent_run_id),
                "resumed": True,
            },
        )
        self._dispatch_locked()

    def _queue_position_locked(self, run_id: str) -> int | None:
        try:
            return list(self._pending).index(run_id) + 1
        except ValueError:
            return None

    def _remove_pending_locked(self, run_id: str) -> None:
        try:
            self._pending.remove(run_id)
        except ValueError:
            return

    def _summary_locked(
        self,
        record: RunRecord,
        *,
        shared: "SharePermission | None" = None,
    ) -> dict[str, Any]:
        return build_run_summary(
            record,
            queue_position=self._queue_position_locked(record.run_id),
            access=access_annotation(shared),
        )

    def _emit_locked(
        self,
        record: RunRecord,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        new_snapshot, events = expand_run_event(
            event_type, payload, status=record.status.value
        )
        if new_snapshot is not None:
            record.snapshot = new_snapshot
        for expanded_type, clean_payload in events:
            self._append_event_locked(record, expanded_type, clean_payload)
        self._project_child_progress_locked(
            record,
            event_type=event_type,
            payload=payload,
        )

    def _project_child_progress_locked(
        self,
        child: RunRecord,
        *,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Append a bounded child status projection to its parent stream."""
        if (
            child.kind != "agent_child"
            or not child.parent_run_id
            or not child.parent_task_id
            or not should_project_child_event(event_type)
        ):
            return
        parent = self._records.get(child.parent_run_id)
        if parent is None or parent.status in TERMINAL_RUN_STATUSES:
            return
        projected = build_child_progress_payload(
            child_run_id=child.run_id,
            parent_task_id=child.parent_task_id,
            run_status=child.status.value,
            event_type=event_type,
            payload=payload,
            snapshot=child.snapshot,
            attempt=child.parent_task_attempt,
        )
        _, events = expand_run_event(
            CHILD_PROGRESS_EVENT,
            projected,
            status=parent.status.value,
        )
        for projected_type, clean_payload in events:
            self._append_event_locked(
                parent, projected_type, clean_payload
            )

    def _append_event_locked(
        self,
        record: RunRecord,
        event_type: str,
        clean_payload: dict[str, Any],
    ) -> None:
        record.event_seq += 1
        event = {
            "type": event_type,
            "run_id": record.run_id,
            "sequence": record.event_seq,
            "created_at": time.time(),
            "data": clean_payload,
        }
        record.events.append(event)
        for subscriber in list(record.subscribers):
            subscriber.put(event)


def format_sse_event(event: dict[str, Any]) -> str:
    """Render one run event as a Server-Sent Event frame."""
    event_type = str(event.get("type") or "message")
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"


def _require_minimum(name: str, value: int, *, minimum: int) -> int:
    """Coerce an integer setting and reject invalid values loudly."""
    coerced = int(value)
    if coerced < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {coerced}")
    return coerced


def _workspace_matches(record: RunRecord, workspace_id: str | None) -> bool:
    """Return whether *record* belongs to the optional workspace namespace."""
    return workspace_id is None or record.workspace_id == workspace_id


def _visible_to_matches(record: RunRecord, visible_to: "UserContext | None") -> bool:
    """Authorization visibility predicate for one run record.

    ``None`` means "no scoping" — the legacy anonymous/static
    principals see every run, preserving single-tenant behaviour
    bit-for-bit (the :class:`~inqtrix.auth.permissions.PermissionService`
    yields ``None`` exactly for those). A scoped principal only sees
    runs it created, matched on (tenant, sub) — sub alone is only
    unique per issuer, so a cross-tenant sub collision must not grant
    visibility. Pre-scoping records (``created_by_sub is None``) stay
    invisible to scoped principals rather than leaking across users.
    Workspace-shared run visibility arrives with the content/sharing
    layer — creator-only is the deliberately conservative v1 rule.
    """
    if visible_to is None:
        return True
    return (
        record.created_by_sub is not None
        and record.created_by_sub == visible_to.principal.sub
        and record.created_by_tenant_id == visible_to.principal.tenant_id
    )


def _coerce_status(status: RunStatus | str) -> RunStatus:
    """Convert status strings to RunStatus and fail on unknown values."""
    if isinstance(status, RunStatus):
        return status
    return RunStatus(status)
