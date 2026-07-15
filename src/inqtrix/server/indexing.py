"""In-memory registry and queue for background vector-index reindex jobs.

The sibling of :mod:`inqtrix.server.runs` for a different long-running
operation: re-embedding a knowledge collection's documents in the
background so a browser can start a reindex, close, and return to a
still-running job. It mirrors the run store's proven shape — daemon-thread
dispatch, a bounded FIFO queue, per-job event buffers with replay, and
canonical ``(tenant_id, user_id)`` visibility — but stays its own type
because reindex records carry progress fields runs lack, are keyed by
``collection_id``, and retain a per-collection history rather than a flat
TTL set.

A reindex is rebuild-in-place: each document keeps its identity and only
its vectors are recomputed (see
:meth:`~inqtrix.services.knowledge_service.KnowledgeService.reembed_document`),
so a cancelled job leaves the collection consistent (some documents
re-embedded, the rest on their previous vectors) — there is no half-built
collection to clean up.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable

from inqtrix.auth.permissions import SharePermission
from inqtrix.execution_authority import AuthorizationRevoked
from inqtrix.server.runs import format_sse_event
from inqtrix.urls import sanitize_error

if TYPE_CHECKING:
    from inqtrix.auth.memory_authority import MemoryAuthorityCoordinator
    from inqtrix.auth.principal import UserContext
    from inqtrix.settings import KnowledgeSettings

log = logging.getLogger("inqtrix")

__all__ = [
    "ACTIVE_INDEXING_STATUSES",
    "ACTIVE_INDEXING_STATUS_VALUES",
    "IndexingJobConflict",
    "IndexingJobHandle",
    "IndexingJobNotFound",
    "IndexingJobRecord",
    "IndexingJobStatus",
    "IndexingJobStore",
    "IndexingQueueFull",
    "TERMINAL_INDEXING_EVENTS",
    "TERMINAL_INDEXING_STATUSES",
    "build_indexing_event",
    "build_indexing_job_summary",
    "format_sse_event",
    "new_indexing_job_id",
]

IndexingWork = Callable[["IndexingJobHandle"], None]


class IndexingJobStatus(StrEnum):
    """Lifecycle status for a background reindex job."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_INDEXING_STATUSES = frozenset(
    {
        IndexingJobStatus.COMPLETED,
        IndexingJobStatus.FAILED,
        IndexingJobStatus.CANCELLED,
    }
)

ACTIVE_INDEXING_STATUSES = frozenset(
    {
        IndexingJobStatus.QUEUED,
        IndexingJobStatus.RUNNING,
        IndexingJobStatus.CANCELLING,
    }
)
"""States that reserve a collection's exclusive maintenance boundary."""

ACTIVE_INDEXING_STATUS_VALUES = tuple(
    status.value for status in ACTIVE_INDEXING_STATUSES
)
"""Database-ready representation of :data:`ACTIVE_INDEXING_STATUSES`."""

TERMINAL_INDEXING_EVENTS = frozenset(
    {
        "inqtrix.index.completed",
        "inqtrix.index.failed",
        "inqtrix.index.cancelled",
    }
)


class IndexingQueueFull(RuntimeError):
    """Raised when the reindex queue has no free slot."""


class IndexingJobConflict(RuntimeError):
    """Raised when a collection already has an active reindex job.

    Reindex is serialized per collection: a second concurrent run on the
    same collection would race two re-embed passes over the same
    documents, so the router maps this to HTTP 409.
    """


class IndexingJobNotFound(KeyError):
    """Raised when a requested reindex job id is not present in memory."""


def new_indexing_job_id() -> str:
    """Return an opaque, log-safe identifier for one reindex job."""
    return f"ix_{uuid.uuid4().hex}"


@dataclass
class IndexingJobRecord:
    """Mutable server-side state for one reindex job."""

    job_id: str
    collection_id: str
    collection_name: str
    embedding_model: str
    created_at: float
    work: IndexingWork | None = field(repr=False, default=None)
    index_id: str | None = None
    """Client-side vector-index id echoed back so the browser can map a
    job (keyed by ``job_id``) onto the right index on resume — events
    never carry the client's id otherwise."""
    workspace_id: str | None = None
    created_by_user_id: uuid.UUID | None = None
    """Canonical user UUID that started the job.

    The value is server-resolved authorization state. ``None`` is reserved
    for unscoped principals.
    """
    created_by_tenant_id: str | None = None
    """Tenant paired with ``created_by_user_id`` for job visibility."""
    status: IndexingJobStatus = IndexingJobStatus.QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    finished_monotonic: float | None = None
    total_documents: int = 0
    completed_documents: int = 0
    current_document_title: str = ""
    error: dict[str, Any] | None = None
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    event_seq: int = 0
    events: deque[dict[str, Any]] = field(default_factory=deque, repr=False)
    subscribers: list[Queue] = field(default_factory=list, repr=False)

    @property
    def percent(self) -> int:
        """Whole-percent progress, derived from completed/total documents.

        Terminal completion reads 100 even for an empty collection (a
        reindex with nothing to do is wholly done); otherwise the floor
        is 0 so a fresh job never flashes a stale value.
        """
        if self.total_documents > 0:
            return round(self.completed_documents / self.total_documents * 100)
        return 100 if self.status == IndexingJobStatus.COMPLETED else 0


def _job_snapshot(record: IndexingJobRecord) -> dict[str, Any]:
    """The progress snapshot carried by events and the public summary.

    One shape so the SSE ``data.snapshot`` and the summary cannot drift;
    ``progress_estimate`` is the 0..1 form mirrored from the run
    snapshot so the frontend's existing progress reader works unchanged.
    """
    return {
        "completed_documents": record.completed_documents,
        "total_documents": record.total_documents,
        "progress_estimate": round(record.percent / 100, 4),
        "current_document_title": record.current_document_title,
    }


def build_indexing_event(
    *,
    job_id: str,
    sequence: int,
    event_type: str,
    created_at: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """The one SSE event envelope shape for reindex jobs.

    Both the in-memory store and the durable Postgres store build their
    frames through this helper so the wire stays byte-identical across
    backends (the frontend ``indexingTypes.ts`` contract).
    """
    return {
        "type": event_type,
        "job_id": job_id,
        "sequence": sequence,
        "created_at": created_at,
        "data": dict(payload),
    }


def build_indexing_job_summary(
    record: IndexingJobRecord, *, queue_position: int | None = None
) -> dict[str, Any]:
    """Public reindex-job summary for HTTP responses (one wire shape).

    ``queue_position`` is the 1-based slot a still-queued job occupies in
    the FIFO wait line (``None`` once running/terminal), mirroring the run
    summary so the UI can show a waiting state instead of a stalled 0 %.
    """
    elapsed = None
    if record.started_at is not None:
        end = record.finished_at or time.time()
        elapsed = round(max(0.0, end - record.started_at), 2)
    return {
        "job_id": record.job_id,
        "collection_id": record.collection_id,
        "collection_name": record.collection_name,
        "embedding_model": record.embedding_model,
        "index_id": record.index_id,
        "status": record.status.value,
        "queue_position": queue_position,
        "workspace_id": record.workspace_id,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "elapsed_seconds": elapsed,
        "total_documents": record.total_documents,
        "completed_documents": record.completed_documents,
        "percent": record.percent,
        "snapshot": _job_snapshot(record),
        "error": dict(record.error) if record.error else None,
        "events_url": f"/v1/knowledge/indexing-jobs/{record.job_id}/events",
    }


@dataclass(frozen=True)
class IndexingJobSubscription:
    """Live event subscription with buffered replay."""

    job_id: str
    queue: Queue
    replay: list[dict[str, Any]]
    store: "IndexingJobStore"

    def close(self) -> None:
        """Detach the subscriber queue from the store."""
        self.store.unsubscribe(self.job_id, self.queue)


class IndexingJobHandle:
    """Worker-side handle for advancing one job without exposing the store."""

    def __init__(
        self,
        store: "IndexingJobStore",
        job_id: str,
        cancel_event: threading.Event,
    ) -> None:
        self._store = store
        self.job_id = job_id
        self.cancel_event = cancel_event

    @property
    def cancelled(self) -> bool:
        """Whether cancellation was requested for this job."""
        return self.cancel_event.is_set()

    def begin(self, total_documents: int) -> None:
        """Record the total document count once it is known."""
        self._store.set_total(self.job_id, total_documents)

    def progress(
        self, *, completed_documents: int, current_document_title: str = ""
    ) -> None:
        """Emit one progress step (documents re-embedded so far)."""
        self._store.progress(
            self.job_id,
            completed_documents=completed_documents,
            current_document_title=current_document_title,
        )

    def complete(self) -> None:
        """Mark the job completed."""
        self._store.complete(self.job_id)

    def fail(self, message: str, *, error_type: str = "server_error") -> None:
        """Mark the job failed with a sanitized error payload."""
        self._store.fail(self.job_id, message, error_type=error_type)

    def cancel(self, reason: str = "cancelled") -> None:
        """Mark the job cancelled after the worker observed the request."""
        self._store.mark_cancelled(self.job_id, reason=reason)

    def document_completed(self, document_id: str) -> None:
        """Emit a per-document 'embedded' event for one finished document."""
        self._store.document_completed(self.job_id, document_id)


class IndexingJobStore:
    """Thread-safe in-memory queue and registry for reindex jobs.

    Args:
        max_concurrent: Maximum number of actively executing reindex
            jobs. Additional accepted jobs wait in the FIFO queue.
        max_queue_size: Maximum number of waiting jobs. Active jobs do
            not count against this number.
        completed_ttl_seconds: How long terminal records remain
            queryable after completion. Queued/running jobs are never
            TTL-evicted.
        event_buffer_size: Number of recent events retained per job for
            late SSE subscribers.
        history_limit: Maximum number of terminal records retained per
            collection (the inline "last N" history); older terminal
            records for a collection are evicted beyond this count.
    """

    def __init__(
        self,
        *,
        max_concurrent: int,
        max_queue_size: int,
        completed_ttl_seconds: int,
        event_buffer_size: int,
        history_limit: int,
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
        self._history_limit = _require_minimum(
            "history_limit", history_limit, minimum=0
        )
        self._records: dict[str, IndexingJobRecord] = {}
        self._pending: deque[str] = deque()
        self._running_count = 0
        self._lock = threading.RLock()
        self._authority: MemoryAuthorityCoordinator | None = None

    def bind_authority_coordinator(
        self, coordinator: "MemoryAuthorityCoordinator"
    ) -> None:
        """Use the process-wide authority lock for job and collection writes."""
        self._authority = coordinator
        self._lock = coordinator.lock

    def _job_authority_context_locked(
        self,
        record: IndexingJobRecord,
        *,
        actor_user_id: uuid.UUID | None = None,
    ) -> Any:
        actor = actor_user_id or record.created_by_user_id
        if self._authority is None or actor is None:
            return nullcontext()
        if not record.created_by_tenant_id:
            raise AuthorizationRevoked("reindex actor has no tenant authority")
        return self._authority.registered_resource_access_guard(
            tenant_id=record.created_by_tenant_id,
            actor_user_id=actor,
            resource_type="knowledge_collection",
            resource_id=record.collection_id,
            minimum=SharePermission.EDIT,
        )

    @contextmanager
    def _submission_authority_context(
        self,
        *,
        tenant_id: str | None,
        actor_user_id: uuid.UUID | None,
        collection_id: str,
    ) -> Iterator[None]:
        if self._authority is None or actor_user_id is None:
            yield
            return
        if not tenant_id:
            raise AuthorizationRevoked("reindex actor has no tenant authority")
        with self._authority.registered_resource_access_guard(
            tenant_id=tenant_id,
            actor_user_id=actor_user_id,
            resource_type="knowledge_collection",
            resource_id=collection_id,
            minimum=SharePermission.EDIT,
        ):
            yield

    def _append_job_effect_locked(
        self,
        record: IndexingJobRecord,
        *,
        action: str,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        if self._authority is None or not record.created_by_tenant_id:
            return
        self._authority.append_registered_resource_effects(
            tenant_id=record.created_by_tenant_id,
            actor_user_id=actor_user_id or record.created_by_user_id,
            action=action,
            resource_type="knowledge_collection",
            resource_id=record.collection_id,
            scope="knowledge",
        )

    @classmethod
    def from_settings(cls, settings: "KnowledgeSettings") -> "IndexingJobStore":
        """Build a reindex job store from knowledge settings."""
        return cls(
            max_concurrent=settings.reindex_max_concurrent,
            max_queue_size=settings.reindex_queue_max_size,
            completed_ttl_seconds=settings.reindex_completed_ttl_seconds,
            event_buffer_size=settings.reindex_event_buffer_size,
            history_limit=settings.reindex_history_limit,
        )

    def submit(
        self,
        *,
        collection_id: str,
        collection_name: str,
        embedding_model: str,
        work: IndexingWork,
        index_id: str | None = None,
        workspace_id: str | None = None,
        created_by_user_id: uuid.UUID | None = None,
        created_by_tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a queued reindex job and dispatch it if capacity allows.

        Raises:
            IndexingJobConflict: The collection already has a queued or
                running reindex job (one active run per collection).
            IndexingQueueFull: The waiting queue is already full.
        """
        with self._lock, self._submission_authority_context(
            tenant_id=created_by_tenant_id,
            actor_user_id=created_by_user_id,
            collection_id=collection_id,
        ):
            self._cleanup_locked()
            if self._active_job_for_collection_locked(collection_id) is not None:
                raise IndexingJobConflict(collection_id)
            if (
                len(self._pending) >= self._max_queue_size
                and self._running_count >= self._max_concurrent
            ):
                raise IndexingQueueFull("reindex queue is full")
            job_id = self._new_unique_job_id_locked()
            record = IndexingJobRecord(
                job_id=job_id,
                collection_id=collection_id,
                collection_name=collection_name,
                embedding_model=embedding_model,
                created_at=time.time(),
                work=work,
                index_id=index_id,
                workspace_id=workspace_id,
                created_by_user_id=created_by_user_id,
                created_by_tenant_id=created_by_tenant_id,
            )
            record.events = deque(maxlen=self._event_buffer_size)
            self._records[job_id] = record
            self._pending.append(job_id)
            self._emit_locked(
                record,
                "inqtrix.index.queued",
                {
                    "status": "queued",
                    "queue_position": self._queue_position_locked(job_id),
                },
            )
            self._append_job_effect_locked(record, action="indexing.submitted")
            self._dispatch_locked()
            return self._summary(record)

    def get(
        self,
        job_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> dict[str, Any]:
        """Return a public summary for *job_id*."""
        with self._lock:
            self._cleanup_locked()
            record = self._record_locked(
                job_id, workspace_id=workspace_id, visible_to=visible_to
            )
            return self._summary(record)

    def list(
        self,
        *,
        collection_id: str | None = None,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> list[dict[str, Any]]:
        """Return summaries for visible jobs, newest first.

        Optionally narrowed to one *collection_id* (the per-collection
        history/resume view). Visibility matches the run store:
        ``None`` *visible_to* sees every job (unscoped principals), a
        scoped principal sees only its own.
        """
        with self._lock:
            self._cleanup_locked()
            summaries = []
            for record in sorted(
                self._records.values(),
                key=lambda item: item.created_at,
                reverse=True,
            ):
                if collection_id is not None and record.collection_id != collection_id:
                    continue
                if _workspace_matches(
                    record, workspace_id
                ) and self._visible_to_matches_locked(record, visible_to):
                    summaries.append(self._summary(record))
            return summaries

    def has_active_job(self, collection_id: str) -> bool:
        """Whether *collection_id* is reserved for reindex maintenance."""
        with self._lock:
            self._cleanup_locked()
            return self._active_job_for_collection_locked(collection_id) is not None

    def run_collection_mutation(
        self, collection_id: str, mutation: Callable[[], Any]
    ) -> Any:
        """Run one in-memory mutation iff no reindex job is active.

        The existing job-store lock is the memory backend's collection-row
        boundary. The callback must contain only the final store mutation;
        parsing and embedding happen before it.

        Raises:
            CollectionMaintenanceActive: An active job owns the collection.
        """
        from inqtrix.knowledge.stores.ports import CollectionMaintenanceActive

        with self._lock:
            self._cleanup_locked()
            if self._active_job_for_collection_locked(collection_id) is not None:
                raise CollectionMaintenanceActive(collection_id)
            return mutation()

    def cancel(
        self,
        job_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        actor_user_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Request cancellation for a queued or running reindex job."""
        with self._lock:
            self._cleanup_locked()
            record = self._record_locked(
                job_id, workspace_id=workspace_id, visible_to=visible_to
            )
            try:
                with self._job_authority_context_locked(
                    record, actor_user_id=actor_user_id
                ):
                    canonical = self._records[job_id]
                    if canonical.status == IndexingJobStatus.QUEUED:
                        self._remove_pending_locked(job_id)
                        canonical.cancel_event.set()
                        self._mark_terminal_locked(
                            canonical, IndexingJobStatus.CANCELLED
                        )
                        self._emit_locked(
                            canonical,
                            "inqtrix.index.cancelled",
                            {
                                "status": "cancelled",
                                "reason": "cancelled_before_start",
                                "snapshot": _job_snapshot(canonical),
                            },
                        )
                        canonical.work = None
                        self._append_job_effect_locked(
                            canonical,
                            action="indexing.cancelled",
                            actor_user_id=actor_user_id,
                        )
                        return self._summary(canonical)
                    if canonical.status == IndexingJobStatus.RUNNING:
                        canonical.cancel_event.set()
                        canonical.status = IndexingJobStatus.CANCELLING
                        self._emit_locked(
                            canonical,
                            "inqtrix.index.cancel_requested",
                            {
                                "status": "cancelling",
                                "reason": "client_requested_cancel",
                            },
                        )
                        self._append_job_effect_locked(
                            canonical,
                            action="indexing.cancel_requested",
                            actor_user_id=actor_user_id,
                        )
                    return self._summary(canonical)
            except AuthorizationRevoked as exc:
                raise IndexingJobNotFound(job_id) from exc

    def subscribe(
        self,
        job_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> IndexingJobSubscription:
        """Subscribe to a job's event stream, replaying buffered events."""
        with self._lock:
            self._cleanup_locked()
            record = self._record_locked(
                job_id, workspace_id=workspace_id, visible_to=visible_to
            )
            queue: Queue = Queue()
            record.subscribers.append(queue)
            return IndexingJobSubscription(
                job_id=job_id,
                queue=queue,
                replay=list(record.events),
                store=self,
            )

    def unsubscribe(self, job_id: str, queue: Queue) -> None:
        """Remove a queue from the subscriber list if still present."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            try:
                record.subscribers.remove(queue)
            except ValueError:
                return

    def set_total(self, job_id: str, total_documents: int) -> None:
        """Record the job's total document count and emit a progress step."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.status in TERMINAL_INDEXING_STATUSES:
                return
            with self._job_authority_context_locked(record):
                record.total_documents = max(0, int(total_documents))
                self._emit_locked(
                    record,
                    "inqtrix.index.progress",
                    {"snapshot": _job_snapshot(record)},
                )

    def progress(
        self,
        job_id: str,
        *,
        completed_documents: int,
        current_document_title: str = "",
    ) -> None:
        """Update completed-document progress and emit a progress event."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.status in TERMINAL_INDEXING_STATUSES:
                return
            with self._job_authority_context_locked(record):
                record.completed_documents = max(0, int(completed_documents))
                record.current_document_title = current_document_title
                self._emit_locked(
                    record,
                    "inqtrix.index.progress",
                    {"snapshot": _job_snapshot(record)},
                )

    def complete(self, job_id: str) -> None:
        """Mark the job completed."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.status in TERMINAL_INDEXING_STATUSES:
                return
            with self._job_authority_context_locked(record):
                if record.status == IndexingJobStatus.CANCELLING:
                    self._mark_terminal_locked(
                        record, IndexingJobStatus.CANCELLED
                    )
                    self._emit_locked(
                        record,
                        "inqtrix.index.cancelled",
                        {
                            "status": "cancelled",
                            "reason": "client_requested_cancel",
                            "snapshot": _job_snapshot(record),
                        },
                    )
                    self._append_job_effect_locked(
                        record, action="indexing.cancelled"
                    )
                    return
                record.current_document_title = ""
                self._mark_terminal_locked(record, IndexingJobStatus.COMPLETED)
                self._emit_locked(
                    record,
                    "inqtrix.index.completed",
                    {"status": "completed", "snapshot": _job_snapshot(record)},
                )
                self._append_job_effect_locked(
                    record, action="indexing.completed"
                )

    def fail(
        self,
        job_id: str,
        message: str,
        *,
        error_type: str = "server_error",
    ) -> None:
        """Mark the job failed with a sanitized error payload."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.status in TERMINAL_INDEXING_STATUSES:
                return
            record.error = {"message": sanitize_error(message), "type": error_type}
            self._mark_terminal_locked(record, IndexingJobStatus.FAILED)
            self._emit_locked(
                record,
                "inqtrix.index.failed",
                {
                    "status": "failed",
                    "error": record.error,
                    "snapshot": _job_snapshot(record),
                },
            )
            self._append_job_effect_locked(record, action="indexing.failed")

    def mark_cancelled(self, job_id: str, *, reason: str) -> None:
        """Mark a running job cancelled after its worker exits."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.status in TERMINAL_INDEXING_STATUSES:
                return
            self._mark_terminal_locked(record, IndexingJobStatus.CANCELLED)
            self._emit_locked(
                record,
                "inqtrix.index.cancelled",
                {
                    "status": "cancelled",
                    "reason": reason,
                    "snapshot": _job_snapshot(record),
                },
            )
            self._append_job_effect_locked(record, action="indexing.cancelled")

    def document_completed(self, job_id: str, document_id: str) -> None:
        """Emit a per-document completion event (one document finished embedding).

        A standalone, non-terminal event carrying the backend document id so the
        frontend can flip that one file's row to "Indexiert" the moment it lands,
        rather than all files flipping together on completion. It is NOT part of
        the progress snapshot — the document-count bar is unchanged.
        """
        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.status in TERMINAL_INDEXING_STATUSES:
                return
            with self._job_authority_context_locked(record):
                self._emit_locked(
                    record,
                    "inqtrix.index.document_completed",
                    {"document_id": document_id, "outcome": "embedded"},
                )

    # -- internals -------------------------------------------------------- #

    def _run_worker(
        self, job_id: str, work: IndexingWork, cancel_event: threading.Event
    ) -> None:
        handle = IndexingJobHandle(self, job_id, cancel_event)
        try:
            work(handle)
            self.complete(job_id)
        except Exception as exc:  # noqa: BLE001 - workers must terminate cleanly
            log.exception("Reindex job %s failed", job_id)
            from inqtrix.execution_failures import classify_execution_failure

            self.fail(
                job_id,
                sanitize_error(exc),
                error_type=classify_execution_failure(exc),
            )
        finally:
            with self._lock:
                record = self._records.get(job_id)
                if record is not None:
                    record.work = None
                self._running_count = max(0, self._running_count - 1)
                self._dispatch_locked()

    def _dispatch_locked(self) -> None:
        while self._running_count < self._max_concurrent and self._pending:
            job_id = self._pending.popleft()
            record = self._records.get(job_id)
            if (
                record is None
                or record.status != IndexingJobStatus.QUEUED
                or record.work is None
            ):
                continue
            record.status = IndexingJobStatus.RUNNING
            record.started_at = time.time()
            self._running_count += 1
            self._emit_locked(
                record,
                "inqtrix.index.started",
                {"status": "running", "snapshot": _job_snapshot(record)},
            )
            thread = threading.Thread(
                target=self._run_worker,
                args=(job_id, record.work, record.cancel_event),
                name=f"inqtrix-reindex-{job_id}",
                daemon=True,
            )
            thread.start()

    def _cleanup_locked(self) -> None:
        now = time.monotonic()
        ttl_expired = {
            job_id
            for job_id, record in self._records.items()
            if record.status in TERMINAL_INDEXING_STATUSES
            and record.finished_monotonic is not None
            and (now - record.finished_monotonic) > self._completed_ttl_seconds
            and not record.subscribers
        }
        beyond_history = self._beyond_history_locked()
        for job_id in ttl_expired | beyond_history:
            del self._records[job_id]

    def _beyond_history_locked(self) -> set[str]:
        """Terminal records past the per-collection history cap, oldest first.

        Records with live subscribers are kept so an in-flight SSE
        replay is never yanked mid-stream.
        """
        per_collection: dict[str, list[IndexingJobRecord]] = {}
        for record in self._records.values():
            if record.status not in TERMINAL_INDEXING_STATUSES:
                continue
            per_collection.setdefault(record.collection_id, []).append(record)
        doomed: set[str] = set()
        for records in per_collection.values():
            records.sort(
                key=lambda item: item.finished_at or item.created_at,
                reverse=True,
            )
            for record in records[self._history_limit :]:
                if not record.subscribers:
                    doomed.add(record.job_id)
        return doomed

    def _active_job_for_collection_locked(
        self, collection_id: str
    ) -> IndexingJobRecord | None:
        for record in self._records.values():
            if (
                record.collection_id == collection_id
                and record.status in ACTIVE_INDEXING_STATUSES
            ):
                return record
        return None

    def _record_locked(
        self,
        job_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> IndexingJobRecord:
        record = self._records.get(job_id)
        if record is None:
            raise IndexingJobNotFound(job_id)
        if self._visible_to_matches_locked(record, visible_to):
            if not _workspace_matches(record, workspace_id):
                raise IndexingJobNotFound(job_id)
            return record
        # The client sees the indistinct 404; the denial stays
        # operator-visible (Designprinzip 1).
        log.warning(
            "authz denied: reindex job %s hidden from user_id=%s tenant=%s",
            job_id,
            visible_to.principal.user_id if visible_to else "",
            visible_to.principal.tenant_id if visible_to else "",
        )
        raise IndexingJobNotFound(job_id)

    def _visible_to_matches_locked(
        self,
        record: IndexingJobRecord,
        visible_to: "UserContext | None",
    ) -> bool:
        if _visible_to_matches(record, visible_to):
            return True
        if self._authority is None or visible_to is None:
            return False
        principal = visible_to.principal
        if principal.user_id is None:
            return False
        try:
            with self._authority.registered_resource_access_guard(
                tenant_id=principal.tenant_id,
                actor_user_id=principal.user_id,
                resource_type="knowledge_collection",
                resource_id=record.collection_id,
                minimum=SharePermission.VIEW,
            ):
                return True
        except AuthorizationRevoked:
            return False

    def _new_unique_job_id_locked(self) -> str:
        for _ in range(8):
            job_id = new_indexing_job_id()
            if job_id not in self._records:
                return job_id
            log.warning("Reindex job id collision detected; retrying allocation.")
        raise RuntimeError("could not allocate a unique reindex job id")

    def _mark_terminal_locked(
        self, record: IndexingJobRecord, status: IndexingJobStatus
    ) -> None:
        record.status = status
        record.finished_at = time.time()
        record.finished_monotonic = time.monotonic()

    def _queue_position_locked(self, job_id: str) -> int | None:
        try:
            return list(self._pending).index(job_id) + 1
        except ValueError:
            return None

    def _summary(self, record: IndexingJobRecord) -> dict[str, Any]:
        """Build the public summary, attaching the FIFO position for a
        still-queued job (``None`` once running/terminal)."""
        position = (
            self._queue_position_locked(record.job_id)
            if record.status == IndexingJobStatus.QUEUED
            else None
        )
        return build_indexing_job_summary(record, queue_position=position)

    def _remove_pending_locked(self, job_id: str) -> None:
        try:
            self._pending.remove(job_id)
        except ValueError:
            return

    def _emit_locked(
        self,
        record: IndexingJobRecord,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        record.event_seq += 1
        event = build_indexing_event(
            job_id=record.job_id,
            sequence=record.event_seq,
            event_type=event_type,
            created_at=time.time(),
            payload=payload,
        )
        record.events.append(event)
        for subscriber in list(record.subscribers):
            subscriber.put(event)


def _require_minimum(name: str, value: int, *, minimum: int) -> int:
    """Coerce an integer setting and reject invalid values loudly."""
    coerced = int(value)
    if coerced < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {coerced}")
    return coerced


def _workspace_matches(
    record: IndexingJobRecord, workspace_id: str | None
) -> bool:
    """Whether *record* belongs to the optional workspace namespace."""
    return workspace_id is None or record.workspace_id == workspace_id


def _visible_to_matches(
    record: IndexingJobRecord, visible_to: "UserContext | None"
) -> bool:
    """Authorization visibility predicate for one reindex job record.

    Mirrors the run store: ``None`` means no scoping (anonymous/static
    principals see every job). A scoped principal sees only jobs created by
    its canonical UUID in the same tenant. Ownerless records stay invisible
    to scoped principals rather than leaking across users.
    """
    if visible_to is None:
        return True
    return (
        record.created_by_user_id is not None
        and record.created_by_user_id == visible_to.principal.user_id
        and record.created_by_tenant_id == visible_to.principal.tenant_id
    )
