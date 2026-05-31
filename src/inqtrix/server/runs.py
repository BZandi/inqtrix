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
from typing import Any, Callable

from inqtrix.runtime_logging import new_run_id, sanitize_event_payload
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


@dataclass
class RunRecord:
    """Mutable server-side state for one native run."""

    run_id: str
    question: str
    stack_name: str
    workspace_id: str | None
    created_at: float
    work: RunWork | None = field(repr=False)
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
    ) -> dict[str, Any]:
        """Create a queued run and dispatch it if capacity is available.

        Returns:
            Public run summary suitable for HTTP responses.

        Raises:
            RunQueueFull: When the waiting queue is already full.
        """
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

    def get(self, run_id: str, *, workspace_id: str | None = None) -> dict[str, Any]:
        """Return a public summary for *run_id*."""
        with self._lock:
            self._cleanup_locked()
            record = self._record_locked(run_id, workspace_id=workspace_id)
            return self._summary_locked(record)

    def list(self, *, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """Return public summaries for all in-memory runs."""
        with self._lock:
            self._cleanup_locked()
            return [
                self._summary_locked(record)
                for record in sorted(
                    self._records.values(),
                    key=lambda item: item.created_at,
                    reverse=True,
                )
                if _workspace_matches(record, workspace_id)
            ]

    def result(self, run_id: str, *, workspace_id: str | None = None) -> dict[str, Any]:
        """Return the stored result payload for a completed run."""
        with self._lock:
            self._cleanup_locked()
            record = self._record_locked(run_id, workspace_id=workspace_id)
            if record.result is None:
                raise RunNotFound(run_id)
            return {
                "run_id": run_id,
                "status": record.status.value,
                **record.result,
            }

    def cancel(self, run_id: str, *, workspace_id: str | None = None) -> dict[str, Any]:
        """Request cancellation for a queued or running run."""
        with self._lock:
            self._cleanup_locked()
            record = self._record_locked(run_id, workspace_id=workspace_id)
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

    def subscribe(self, run_id: str, *, workspace_id: str | None = None) -> RunSubscription:
        """Subscribe to a run's event stream, replaying buffered events."""
        with self._lock:
            self._cleanup_locked()
            record = self._record_locked(run_id, workspace_id=workspace_id)
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
            record = self._record_locked(run_id)
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
            record = self._record_locked(run_id)
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
            record = self._record_locked(run_id)
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
            record = self._record_locked(run_id)
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
    ) -> RunRecord:
        record = self._records.get(run_id)
        if record is None or not _workspace_matches(record, workspace_id):
            raise RunNotFound(run_id)
        return record

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

    def _summary_locked(self, record: RunRecord) -> dict[str, Any]:
        elapsed = None
        if record.started_at is not None:
            end = record.finished_at or time.time()
            elapsed = round(max(0.0, end - record.started_at), 2)
        return {
            "run_id": record.run_id,
            "status": record.status.value,
            "queue_position": self._queue_position_locked(record.run_id),
            "question": record.question,
            "stack": record.stack_name,
            "workspace_id": record.workspace_id,
            "mode": record.mode,
            "agent_overrides": dict(record.agent_overrides),
            "created_at": record.created_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "elapsed_seconds": elapsed,
            "snapshot": dict(record.snapshot),
            "error": dict(record.error) if record.error else None,
            "events_url": f"/v1/runs/{record.run_id}/events",
            "result_url": f"/v1/runs/{record.run_id}/result",
        }

    def _emit_locked(
        self,
        record: RunRecord,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        clean_payload = sanitize_event_payload(event_type, dict(payload))
        snapshot = clean_payload.get("snapshot")
        if isinstance(snapshot, dict):
            record.snapshot = dict(snapshot)
            if event_type != "inqtrix.run.snapshot":
                snapshot_payload = sanitize_event_payload(
                    "inqtrix.run.snapshot",
                    {
                        "status": record.status.value,
                        "snapshot": record.snapshot,
                    },
                )
                self._append_event_locked(
                    record,
                    "inqtrix.run.snapshot",
                    snapshot_payload,
                )
        self._append_event_locked(record, event_type, clean_payload)

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


def _coerce_status(status: RunStatus | str) -> RunStatus:
    """Convert status strings to RunStatus and fail on unknown values."""
    if isinstance(status, RunStatus):
        return status
    return RunStatus(status)
