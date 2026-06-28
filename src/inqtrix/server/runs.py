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
from inqtrix.runs.shared import (
    access_annotation,
    build_run_summary,
    expand_run_event,
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
    """Lifecycle status for a native in-memory run."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)


class RunQueueFull(RuntimeError):
    """Raised when the native run queue has no free slot."""


class RunNotFound(KeyError):
    """Raised when a requested run id is not present in memory."""


class RunActive(RuntimeError):
    """Raised when a delete targets a run that is still queued or running.

    Deletion is terminal-only: removing a record an executing worker still
    holds would let its final write resurrect a half-gone run. The caller
    cancels first, then deletes once the run reaches a terminal state.
    """


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
    status: RunStatus = RunStatus.QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    finished_monotonic: float | None = None
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
    """

    def __init__(
        self,
        *,
        max_concurrent: int,
        max_queue_size: int,
        completed_ttl_seconds: int,
        event_buffer_size: int,
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
    ) -> dict[str, Any]:
        """Create a queued run and dispatch it if capacity is available.

        Args:
            request_payload: Re-execution payload persisted by durable
                backends so worker processes can rebuild the run from
                the row alone. Deliberately ignored here — in-memory
                execution keeps the work closure in-process.

        Returns:
            Public run summary suitable for HTTP responses.

        Raises:
            RunQueueFull: When the waiting queue is already full.
        """
        del request_payload
        with self._lock:
            self._cleanup_locked()
            if len(self._pending) >= self._max_queue_size and self._running_count >= self._max_concurrent:
                raise RunQueueFull("native run queue is full")

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
                key=lambda item: item.created_at,
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
            if record.status == RunStatus.QUEUED:
                self._remove_pending_locked(run_id)
                record.cancel_event.set()
                self._mark_terminal_locked(record, RunStatus.CANCELLED)
                self._emit_locked(
                    record,
                    "inqtrix.run.cancelled",
                    {"status": "cancelled", "reason": "cancelled_before_start"},
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

    def emit(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Emit one event to the run buffer and live subscribers."""
        with self._lock:
            record, _shared = self._record_locked(run_id)
            self._emit_locked(record, event_type, payload or {})

    def complete(
        self,
        run_id: str,
        result: dict[str, Any],
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Store the final result and mark the run completed."""
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

    def fail(self, run_id: str, message: str, *, error_type: str = "server_error") -> None:
        """Mark a run failed with a sanitized error payload."""
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

    def mark_cancelled(self, run_id: str, *, reason: str) -> None:
        """Mark a running run cancelled after its worker exits."""
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

    def _run_worker(self, run_id: str, work: RunWork, cancel_event: threading.Event) -> None:
        handle = RunHandle(self, run_id, cancel_event)
        try:
            work(handle)
            with self._lock:
                record = self._records.get(run_id)
                if record is not None and record.status not in TERMINAL_RUN_STATUSES:
                    self._mark_terminal_locked(record, RunStatus.COMPLETED)
                    self._emit_locked(
                        record,
                        "inqtrix.run.completed",
                        {"status": "completed", "snapshot": record.snapshot},
                    )
        except Exception as exc:  # noqa: BLE001 - run workers must terminate cleanly
            log.exception("Native run %s failed", run_id)
            self.fail(run_id, sanitize_error(exc))
        finally:
            with self._lock:
                record = self._records.get(run_id)
                if record is not None:
                    record.work = None
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
