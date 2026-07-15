"""Durable reindex-job store: records and events in Postgres.

Same public surface as the in-memory
:class:`~inqtrix.server.indexing.IndexingJobStore` — the reindex router
and :class:`~inqtrix.services.indexing_service.IndexingService` cannot
tell the backends apart. The durable twin of
:class:`~inqtrix.runs.postgres_store.PostgresRunStore`, built from the
same parts and sharing its worker stack; the differences are the
reindex domain itself: per-document progress columns, one-active-job
serialization per collection (a partial unique index), a per-collection
history cap, and no share layer.

Two execution modes (mirroring the run store):

* ``queue is None`` (``INQTRIX_STORAGE_BACKEND=postgres`` alone):
  records and events are durable; execution stays in this process with
  the same daemon-thread dispatch as the in-memory store.
* ``queue`` set (``INQTRIX_QUEUE_BACKEND=valkey``): accepted jobs are
  persisted and dispatched to the reindex stream; ``inqtrix-worker``
  processes claim and execute them, re-embedding from the canonical
  Postgres text. The job row is the source of truth.

The storage layer is async (asyncpg) while this surface is sync (the
router and the job handle call it from worker threads); the store owns
one background event loop and funnels every database operation through
it. Sequence numbers are allocated via
``UPDATE indexing_jobs SET event_seq = event_seq + 1 RETURNING event_seq``
— gap-free across processes, which keeps the SSE stream byte-compatible
with the in-memory store.

The schema-agnostic plumbing — the ``_call``/``close``/``_session``
async-loop bridge, the background-loop construction, the no-queue
in-process dispatch loop, and the DB-polling SSE subscription — lives
once in :class:`~inqtrix.runs.durable_store.DurableJobStoreBase` and
:class:`~inqtrix.runs.durable_store.PollingJobSubscription`, shared with
the run store. Only the schema-bearing bodies stay here: the SQL, the
visibility predicate, and this store's history-cap / active-collection /
progress semantics.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Any

from sqlalchemy import case, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from inqtrix.auth.permissions import SharePermission
from inqtrix.execution_authority import AuthorizationRevoked
from inqtrix.runs.durable_store import (
    DEFAULT_TENANT,
    DurableJobStoreBase,
    PollingJobSubscription,
    _LocalJob,
)
from inqtrix.server.indexing import (
    ACTIVE_INDEXING_STATUS_VALUES,
    TERMINAL_INDEXING_EVENTS,
    TERMINAL_INDEXING_STATUSES,
    IndexingJobConflict,
    IndexingJobHandle,
    IndexingJobNotFound,
    IndexingJobRecord,
    IndexingJobStatus,
    IndexingQueueFull,
    IndexingWork,
    build_indexing_event,
    build_indexing_job_summary,
    new_indexing_job_id,
)
from inqtrix.knowledge.stores.ports import CollectionNotFound
from inqtrix.storage.indexing_orm import indexing_job_events, indexing_jobs
from inqtrix.storage.knowledge_orm import knowledge_collections
from inqtrix.storage.resource_access import (
    LockedResourceAccess,
    append_resource_effects,
    lock_resource_access,
)
from inqtrix.urls import sanitize_error

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from inqtrix.auth.principal import UserContext
    from inqtrix.runs.indexing_queue import ValkeyIndexingQueue

log = logging.getLogger("inqtrix")

_TERMINAL_VALUES = tuple(status.value for status in TERMINAL_INDEXING_STATUSES)
_ACTIVE_VALUES = ACTIVE_INDEXING_STATUS_VALUES

_STUCK_ROW_MAX_AGE_SECONDS = 7 * 86_400.0
"""Hard retention cap for jobs that never reached a terminal state — a
worker died mid-run and the row would otherwise hold an active slot and
present as eternally running forever."""

_ACTIVE_JOB_CONSTRAINT = "uq_indexing_jobs_active_collection"


@dataclass(frozen=True)
class ClaimedIndexingJob:
    """Result of a successful worker claim on a queued reindex job."""

    job_id: str
    tenant_id: str
    attempt: int
    collection_id: str
    embedding_model: str
    # Persisted attribution of the submitter, so the worker can meter the
    # re-embed against the canonical user UUID without a live principal.
    created_by_user_id: uuid.UUID | None = None
    created_by_tenant_id: str | None = None
    cancel_requested: bool = False


@dataclass(frozen=True)
class _MaintenanceAction:
    """One retention or recovery mutation applied under lifecycle locks."""

    action: str
    error: dict[str, str] | None = None
    request_cancel: bool = False


class PostgresIndexingJobStore(DurableJobStoreBase):
    """Durable reindex registry with the in-memory store's public surface.

    Args:
        engine: Async engine OWNED by this store. asyncpg pools are
            event-loop-affine: every connection must live on the store's
            background loop, so the engine is deliberately NOT shared
            with the identity/file/knowledge backends.
        app_role: Restricted database role assumed per transaction
            (``SET LOCAL ROLE``) so forced RLS applies.
        queue: Reindex dispatch queue; ``None`` keeps execution in this
            process (durable records, unchanged threading).
        max_concurrent: In-process execution slots (no-queue mode) and
            part of the queue-saturation formula.
        max_queue_size: Waiting-job bound; exceeding it with all slots
            busy rejects submissions (HTTP 429 upstream).
        completed_ttl_seconds: Retention for terminal jobs; lazy cleanup
            deletes older rows (events cascade).
        history_limit: Maximum terminal jobs retained per collection
            (the inline "last N" history); older terminal rows for a
            collection are deleted beyond this count.
        worker_id: Identity stamped into ``claimed_by`` for jobs this
            process executes or fences.
        recover_orphans: Whether this instance may blanket-fail
            queued/running rows left by a previous process. ``None``
            infers from ``queue`` (no-queue single API process sweeps,
            queue mode never); the queue-backed WORKER passes an
            explicit ``False`` — its ``queue=None`` is claim-mode
            wiring, stream reclaim owns crash recovery there.

    Tenancy: job rows live in the single deployment tenant (``default``)
    at the RLS layer — per-user visibility is the
    ``(created_by_user_id, created_by_tenant_id)`` predicate, exactly like
    the run store and the in-memory reindex store.
    """

    _loop_thread_name = "inqtrix-index-db"
    _dispatch_thread_prefix = "inqtrix-reindex"
    _job_kind = "Durable reindex job"

    def __init__(
        self,
        *,
        engine: "AsyncEngine",
        app_role: str,
        queue: "ValkeyIndexingQueue | None" = None,
        max_concurrent: int,
        max_queue_size: int,
        completed_ttl_seconds: int,
        history_limit: int,
        worker_id: str,
        restrict_to_workspace_members: bool = False,
        recover_orphans: bool | None = None,
    ) -> None:
        # The engine/session/loop/dispatch plumbing lives in
        # DurableJobStoreBase; this store adds only its sizing and
        # retention state.
        super().__init__(
            engine=engine,
            app_role=app_role,
            worker_id=worker_id,
            queue=queue,
            max_concurrent=max_concurrent,
            recover_orphans=recover_orphans,
        )
        self._max_queue_size = max_queue_size
        self._completed_ttl_seconds = completed_ttl_seconds
        self._history_limit = history_limit
        self._restrict_to_workspace_members = restrict_to_workspace_members

    # -- public surface (IndexingJobStore parity) ------------------------- #

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
        """Persist one queued reindex job, then dispatch locally or enqueue.

        Raises:
            IndexingJobConflict: The collection already has an active
                reindex job (one active run per collection).
            IndexingQueueFull: The waiting queue is full and every slot
                is busy (queue-mode counts are cluster-wide via the
                database, exactly like the run store).
        """
        summary = self._call(
            self._submit_db(
                collection_id=collection_id,
                collection_name=collection_name,
                embedding_model=embedding_model,
                index_id=index_id,
                workspace_id=workspace_id,
                created_by_user_id=created_by_user_id,
                created_by_tenant_id=created_by_tenant_id,
            )
        )
        job_id = summary["job_id"]
        if self._queue is not None:
            try:
                self._queue.enqueue(job_id=job_id, tenant_id=DEFAULT_TENANT)
            except Exception:  # noqa: BLE001 — row is committed; visible
                log.warning(
                    "Dispatch-Nachricht fuer Reindex-Job %s konnte nicht "
                    "gesendet werden — der Reconciler holt das nach.",
                    job_id,
                    exc_info=True,
                )
        else:
            with self._lock:
                self._local[job_id] = _LocalJob(work=work)
                self._pending.append(job_id)
                self._dispatch_locked()
        return summary

    def get(
        self,
        job_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> dict[str, Any]:
        """Return a public summary for *job_id*."""
        return self._call(self._summary_db(job_id, workspace_id, visible_to))

    def list(
        self,
        *,
        collection_id: str | None = None,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> list[dict[str, Any]]:
        """Return summaries for visible jobs, newest first."""
        return self._call(
            self._list_db(collection_id, workspace_id, visible_to)
        )

    def has_active_job(self, collection_id: str) -> bool:
        """Whether *collection_id* has a queued/running/cancelling job."""
        return self._call(self._has_active_job_db(collection_id))

    def cancel(
        self,
        job_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        actor_user_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Request cancellation for a queued or running reindex job."""
        summary = self._call(
            self._cancel_db(job_id, workspace_id, visible_to, actor_user_id)
        )
        with self._lock:
            local = self._local.get(job_id)
            if local is not None:
                local.cancel_event.set()
                if summary["status"] == IndexingJobStatus.CANCELLED.value:
                    try:
                        self._pending.remove(job_id)
                    except ValueError:
                        pass
                    local.work = None
        return summary

    def subscribe(
        self,
        job_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> PollingJobSubscription:
        """Subscribe to a job's event stream with full stored replay."""
        tenant_id, replay = self._call(
            self._replay_db(job_id, workspace_id, visible_to)
        )
        return PollingJobSubscription(
            self,
            job_id,
            tenant_id,
            replay,
            terminal_events=TERMINAL_INDEXING_EVENTS,
            thread_label="inqtrix-index-events",
        )

    def unsubscribe(self, job_id: str, queue: Queue) -> None:
        """Parity no-op: polling subscriptions detach via ``close()``."""

    def set_total(
        self, job_id: str, total_documents: int, *, fence_attempt: int | None = None
    ) -> None:
        """Record the job's total document count and emit a progress step."""
        self._call(
            self._progress_db(
                job_id,
                total_documents=max(0, int(total_documents)),
                fence_attempt=fence_attempt,
            )
        )

    def progress(
        self,
        job_id: str,
        *,
        completed_documents: int,
        current_document_title: str = "",
        fence_attempt: int | None = None,
    ) -> None:
        """Update completed-document progress and emit a progress event."""
        self._call(
            self._progress_db(
                job_id,
                completed_documents=max(0, int(completed_documents)),
                current_document_title=current_document_title,
                fence_attempt=fence_attempt,
            )
        )

    def complete(self, job_id: str, *, fence_attempt: int | None = None) -> bool:
        """Mark the job completed.

        Returns:
            ``True`` when the terminal transition landed; ``False`` when
            absorbed (already terminal or fenced out) — the worker must
            NOT ack the dispatch message in that case.
        """
        return self._call(
            self._terminal_db(
                job_id,
                IndexingJobStatus.COMPLETED,
                fence_attempt=fence_attempt,
                clear_title=True,
                event_type="inqtrix.index.completed",
                extra={"status": "completed"},
            )
        )

    def fail(
        self,
        job_id: str,
        message: str,
        *,
        error_type: str = "server_error",
        fence_attempt: int | None = None,
    ) -> bool:
        """Mark the job failed with a sanitized error payload."""
        error = {"message": sanitize_error(message), "type": error_type}
        return self._call(
            self._terminal_db(
                job_id,
                IndexingJobStatus.FAILED,
                fence_attempt=fence_attempt,
                error=error,
                event_type="inqtrix.index.failed",
                extra={"status": "failed", "error": error},
            )
        )

    def mark_cancelled(
        self, job_id: str, *, reason: str, fence_attempt: int | None = None
    ) -> bool:
        """Mark a running job cancelled after its worker observed it."""
        return self._call(
            self._terminal_db(
                job_id,
                IndexingJobStatus.CANCELLED,
                fence_attempt=fence_attempt,
                event_type="inqtrix.index.cancelled",
                extra={"status": "cancelled", "reason": reason},
            )
        )

    def document_completed(
        self, job_id: str, document_id: str, *, fence_attempt: int | None = None
    ) -> None:
        """Emit a per-document completion event (one document finished embedding)."""
        self._call(
            self._document_completed_db(
                job_id, document_id, fence_attempt=fence_attempt
            )
        )

    # -- worker surface --------------------------------------------------- #

    def claim_for_execution(
        self, job_id: str, tenant_id: str, *, allow_takeover: bool
    ) -> ClaimedIndexingJob | None:
        """Atomically claim one reindex job for execution."""
        return self._call(
            self._claim_db(job_id, tenant_id, allow_takeover=allow_takeover)
        )

    def cancel_requested_jobs(self, job_ids: dict[str, str]) -> set[str]:
        """Subset of ``job_ids`` (id -> tenant) with a pending cancel."""
        return self._call(self._cancel_requested_db(job_ids))

    def stale_queued_jobs(
        self, *, older_than_seconds: float
    ) -> list[tuple[str, str]]:
        """Queued ``(job_id, tenant_id)`` pairs older than the threshold."""
        return self._call(self._stale_queued_db(older_than_seconds))

    # -- in-process execution hooks (no-queue mode) ---------------------- #

    def _make_handle(self, job_id: str, cancel_event) -> IndexingJobHandle:
        return IndexingJobHandle(self, job_id, cancel_event)

    def _auto_complete(self, job_id: str) -> None:
        # Usually a no-op (the work body already completed the job);
        # warn_on_noop=False keeps the genuine fenced-out warning on the
        # public complete()/fail() path meaningful.
        self._call(
            self._terminal_db(
                job_id,
                IndexingJobStatus.COMPLETED,
                fence_attempt=None,
                clear_title=True,
                event_type="inqtrix.index.completed",
                extra={"status": "completed"},
                warn_on_noop=False,
            )
        )

    # -- subscription poll bridge ----------------------------------------- #

    def _events_after(
        self, job_id: str, tenant_id: str, after_sequence: int
    ) -> list[dict[str, Any]]:
        return self._call(
            self._events_after_db(job_id, tenant_id, after_sequence)
        )

    # -- async database operations ---------------------------------------- #

    async def _has_active_job_db(self, collection_id: str) -> bool:
        async with self._session(DEFAULT_TENANT) as session:
            active = await session.scalar(
                select(indexing_jobs.c.job_id)
                .where(
                    indexing_jobs.c.collection_id == collection_id,
                    indexing_jobs.c.status.in_(_ACTIVE_VALUES),
                )
                .limit(1)
            )
            return active is not None

    async def _lock_collection_access_for_job(
        self,
        session: "AsyncSession",
        *,
        job_id: str,
        actor_user_id: uuid.UUID | None = None,
    ) -> tuple[Any, LockedResourceAccess] | None:
        """Lock parent collection access, then the immutable job pointer."""
        pointer = (
            await session.execute(
                select(
                    indexing_jobs.c.collection_id,
                    indexing_jobs.c.created_by_user_id,
                ).where(indexing_jobs.c.job_id == job_id)
            )
        ).first()
        if pointer is None:
            return None
        effective_actor = (
            actor_user_id
            if actor_user_id is not None
            else pointer.created_by_user_id
        )
        access = await lock_resource_access(
            session,
            tenant_id=DEFAULT_TENANT,
            actor_user_id=effective_actor,
            resource_type="knowledge_collection",
            resource_table=knowledge_collections,
            id_column=knowledge_collections.c.id,
            resource_id=pointer.collection_id,
            owner_column=knowledge_collections.c.created_by_user_id,
            minimum=SharePermission.EDIT,
            restrict_to_workspace_members=self._restrict_to_workspace_members,
        )
        if access is None:
            return None
        row = (
            await session.execute(
                select(indexing_jobs)
                .where(indexing_jobs.c.job_id == job_id)
                .with_for_update()
            )
        ).mappings().first()
        if row is None or row["collection_id"] != pointer.collection_id:
            return None
        return row, access

    async def _append_collection_effects(
        self,
        session: "AsyncSession",
        *,
        row: Any,
        access: LockedResourceAccess,
        action: str,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        """Invalidate the parent collection views in the same transaction."""
        await append_resource_effects(
            session,
            tenant_id=row["tenant_id"],
            actor_user_id=(
                actor_user_id
                if actor_user_id is not None
                else row["created_by_user_id"]
            ),
            owner_user_id=access.owner_user_id,
            action=action,
            resource_type="knowledge_collection",
            resource_id=row["collection_id"],
            scope="indexing",
        )

    async def _submit_db(self, **fields: Any) -> dict[str, Any]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            collection_id = fields["collection_id"]
            access = await lock_resource_access(
                session,
                tenant_id=DEFAULT_TENANT,
                actor_user_id=fields["created_by_user_id"],
                resource_type="knowledge_collection",
                resource_table=knowledge_collections,
                id_column=knowledge_collections.c.id,
                resource_id=collection_id,
                owner_column=knowledge_collections.c.created_by_user_id,
                minimum=SharePermission.EDIT,
                restrict_to_workspace_members=self._restrict_to_workspace_members,
            )
            if access is None:
                raise CollectionNotFound(collection_id)
            active = await session.scalar(
                select(func.count())
                .select_from(indexing_jobs)
                .where(
                    indexing_jobs.c.collection_id == collection_id,
                    indexing_jobs.c.status.in_(_ACTIVE_VALUES),
                )
            )
            if active:
                raise IndexingJobConflict(collection_id)
            queued = await session.scalar(
                select(func.count())
                .select_from(indexing_jobs)
                .where(
                    indexing_jobs.c.status == IndexingJobStatus.QUEUED.value
                )
            )
            running = await session.scalar(
                select(func.count())
                .select_from(indexing_jobs)
                .where(
                    indexing_jobs.c.status.in_(
                        (
                            IndexingJobStatus.RUNNING.value,
                            IndexingJobStatus.CANCELLING.value,
                        )
                    )
                )
            )
            if (
                (queued or 0) >= self._max_queue_size
                and (running or 0) >= self._max_concurrent
            ):
                raise IndexingQueueFull("reindex queue is full")

            created_at = time.time()
            try:
                job_id = await self._insert_with_unique_id(
                    session, created_at=created_at, **fields
                )
            except IntegrityError as exc:
                # The partial unique index is the race backstop: a second
                # submit that slipped past the count check collides here.
                if _ACTIVE_JOB_CONSTRAINT in str(exc.orig):
                    raise IndexingJobConflict(fields["collection_id"]) from exc
                raise
            position = await self._queue_position_db(session, created_at)
            await self._append_events_db(
                session,
                job_id,
                DEFAULT_TENANT,
                "inqtrix.index.queued",
                {"status": "queued", "queue_position": position},
            )
            row = await self._row_db(session, job_id)
            await self._append_collection_effects(
                session,
                row=row,
                access=access,
                action="indexing.submitted",
            )
            return await self._row_summary(session, row)

    async def _insert_with_unique_id(
        self, session: "AsyncSession", *, created_at: float, **fields: Any
    ) -> str:
        for _ in range(8):
            job_id = new_indexing_job_id()
            result = await session.execute(
                pg_insert(indexing_jobs)
                .values(
                    job_id=job_id,
                    tenant_id=DEFAULT_TENANT,
                    status=IndexingJobStatus.QUEUED.value,
                    collection_id=fields["collection_id"],
                    collection_name=fields["collection_name"],
                    embedding_model=fields["embedding_model"],
                    index_id=fields["index_id"],
                    workspace_id=fields["workspace_id"],
                    created_by_user_id=fields["created_by_user_id"],
                    created_by_tenant_id=fields["created_by_tenant_id"],
                    created_at=created_at,
                )
                .on_conflict_do_nothing(index_elements=["job_id"])
                .returning(indexing_jobs.c.job_id)
            )
            if result.scalar_one_or_none() is not None:
                return job_id
            log.warning("Reindex job id collision detected; retrying.")
        raise RuntimeError("could not allocate a unique reindex job id")

    async def _queue_position_db(
        self, session: "AsyncSession", created_at: float
    ) -> int:
        earlier = await session.scalar(
            select(func.count())
            .select_from(indexing_jobs)
            .where(
                indexing_jobs.c.status == IndexingJobStatus.QUEUED.value,
                indexing_jobs.c.created_at < created_at,
            )
        )
        return int(earlier or 0) + 1

    async def _row_summary(self, session: "AsyncSession", row) -> dict[str, Any]:
        """Public summary for one row, with the FIFO position attached for a
        still-queued job (``None`` once running/terminal)."""
        position = (
            await self._queue_position_db(session, row["created_at"])
            if row["status"] == IndexingJobStatus.QUEUED.value
            else None
        )
        return build_indexing_job_summary(
            _record_from_row(row), queue_position=position
        )

    async def _row_db(self, session: "AsyncSession", job_id: str):
        row = (
            await session.execute(
                select(indexing_jobs).where(indexing_jobs.c.job_id == job_id)
            )
        ).mappings().first()
        if row is None:
            raise IndexingJobNotFound(job_id)
        return row

    async def _visible_row_db(
        self,
        session: "AsyncSession",
        job_id: str,
        workspace_id: str | None,
        visible_to: "UserContext | None",
    ):
        row = await self._row_db(session, job_id)
        if _visible_row(row, visible_to):
            if not _workspace_matches_row(row, workspace_id):
                raise IndexingJobNotFound(job_id)
            return row
        log.warning(
            "authz denied: reindex job %s hidden from user_id=%s tenant=%s",
            job_id,
            visible_to.principal.user_id if visible_to else "",
            visible_to.principal.tenant_id if visible_to else "",
        )
        raise IndexingJobNotFound(job_id)

    async def _summary_db(
        self,
        job_id: str,
        workspace_id: str | None,
        visible_to: "UserContext | None",
    ) -> dict[str, Any]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            row = await self._visible_row_db(
                session, job_id, workspace_id, visible_to
            )
            return await self._row_summary(session, row)

    async def _list_db(
        self,
        collection_id: str | None,
        workspace_id: str | None,
        visible_to: "UserContext | None",
    ) -> list[dict[str, Any]]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            query = select(indexing_jobs).order_by(
                indexing_jobs.c.created_at.desc()
            )
            if collection_id is not None:
                query = query.where(
                    indexing_jobs.c.collection_id == collection_id
                )
            if visible_to is not None:
                query = query.where(
                    indexing_jobs.c.created_by_user_id == visible_to.principal.user_id,
                    indexing_jobs.c.created_by_tenant_id
                    == visible_to.principal.tenant_id,
                )
                if workspace_id is not None:
                    query = query.where(
                        indexing_jobs.c.workspace_id == workspace_id
                    )
            elif workspace_id is not None:
                query = query.where(
                    indexing_jobs.c.workspace_id == workspace_id
                )
            rows = (await session.execute(query)).mappings().all()
            return [await self._row_summary(session, row) for row in rows]

    async def _cancel_db(
        self,
        job_id: str,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        actor_user_id: uuid.UUID | None,
    ) -> dict[str, Any]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            locked = await self._lock_collection_access_for_job(
                session,
                job_id=job_id,
                actor_user_id=actor_user_id,
            )
            if locked is None:
                raise IndexingJobNotFound(job_id)
            row, access = locked
            if not _workspace_matches_row(row, workspace_id):
                raise IndexingJobNotFound(job_id)
            if visible_to is not None and actor_user_id is None and not _visible_row(
                row, visible_to
            ):
                raise IndexingJobNotFound(job_id)
            status = row["status"]
            if status == IndexingJobStatus.QUEUED.value:
                cancelled = (
                    await session.execute(
                        update(indexing_jobs)
                        .where(
                            indexing_jobs.c.job_id == job_id,
                            indexing_jobs.c.status
                            == IndexingJobStatus.QUEUED.value,
                        )
                        .values(
                            status=IndexingJobStatus.CANCELLED.value,
                            cancel_requested=True,
                            finished_at=time.time(),
                        )
                        .returning(indexing_jobs.c.job_id)
                    )
                ).scalar_one_or_none()
                if cancelled is not None:
                    await self._append_events_db(
                        session,
                        job_id,
                        row["tenant_id"],
                        "inqtrix.index.cancelled",
                        {
                            "status": "cancelled",
                            "reason": "cancelled_before_start",
                            "snapshot": _snapshot_from_row(
                                await self._row_db(session, job_id)
                            ),
                        },
                    )
                    await self._append_collection_effects(
                        session,
                        row=await self._row_db(session, job_id),
                        access=access,
                        action="indexing.cancelled",
                        actor_user_id=actor_user_id,
                    )
                else:
                    # Lost the CAS to a concurrent claim: the job runs
                    # now — degrade to the two-phase cancel request.
                    status = IndexingJobStatus.RUNNING.value
            if status == IndexingJobStatus.RUNNING.value:
                requested = (
                    await session.execute(
                        update(indexing_jobs)
                        .where(
                            indexing_jobs.c.job_id == job_id,
                            indexing_jobs.c.status
                            == IndexingJobStatus.RUNNING.value,
                        )
                        .values(
                            status=IndexingJobStatus.CANCELLING.value,
                            cancel_requested=True,
                        )
                        .returning(indexing_jobs.c.job_id)
                    )
                ).scalar_one_or_none()
                if requested is not None:
                    await self._append_events_db(
                        session,
                        job_id,
                        row["tenant_id"],
                        "inqtrix.index.cancel_requested",
                        {
                            "status": "cancelling",
                            "reason": "client_requested_cancel",
                        },
                    )
                    await self._append_collection_effects(
                        session,
                        row=await self._row_db(session, job_id),
                        access=access,
                        action="indexing.cancel_requested",
                        actor_user_id=actor_user_id,
                    )
            fresh = await self._row_db(session, job_id)
            return await self._row_summary(session, fresh)

    async def _replay_db(
        self,
        job_id: str,
        workspace_id: str | None,
        visible_to: "UserContext | None",
    ) -> tuple[str, list[dict[str, Any]]]:
        async with self._session(DEFAULT_TENANT) as session:
            await self._cleanup_db(session)
            row = await self._visible_row_db(
                session, job_id, workspace_id, visible_to
            )
            events = await self._fetch_events(session, job_id, 0)
            return row["tenant_id"], events

    async def _events_after_db(
        self, job_id: str, tenant_id: str, after_sequence: int
    ) -> list[dict[str, Any]]:
        async with self._session(tenant_id) as session:
            return await self._fetch_events(session, job_id, after_sequence)

    async def _fetch_events(
        self, session: "AsyncSession", job_id: str, after_sequence: int
    ) -> list[dict[str, Any]]:
        rows = (
            await session.execute(
                select(indexing_job_events)
                .where(
                    indexing_job_events.c.job_id == job_id,
                    indexing_job_events.c.sequence > after_sequence,
                )
                .order_by(indexing_job_events.c.sequence)
            )
        ).mappings().all()
        return [
            build_indexing_event(
                job_id=job_id,
                sequence=row["sequence"],
                event_type=row["type"],
                created_at=row["created_at"],
                payload=row["data"],
            )
            for row in rows
        ]

    async def _append_events_db(
        self,
        session: "AsyncSession",
        job_id: str,
        tenant_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        sequence = (
            await session.execute(
                update(indexing_jobs)
                .where(indexing_jobs.c.job_id == job_id)
                .values(event_seq=indexing_jobs.c.event_seq + 1)
                .returning(indexing_jobs.c.event_seq)
            )
        ).scalar_one()
        await session.execute(
            insert(indexing_job_events).values(
                job_id=job_id,
                tenant_id=tenant_id,
                sequence=sequence,
                type=event_type,
                created_at=time.time(),
                data=dict(payload),
            )
        )

    async def _fence_ok(
        self,
        session: "AsyncSession",
        job_id: str,
        fence_attempt: int | None,
    ) -> bool:
        """Atomic ownership re-check before a fenced non-terminal write.

        Mirrors the run store: the no-op ``event_seq + 0`` update returns
        a row only while this worker attempt still owns the job, so a
        superseded attempt's progress is dropped with a visible log
        instead of interleaving with the new owner's stream.
        """
        if fence_attempt is None:
            return True
        owned = (
            await session.execute(
                update(indexing_jobs)
                .where(
                    indexing_jobs.c.job_id == job_id,
                    indexing_jobs.c.claimed_by == self._worker_id,
                    indexing_jobs.c.attempt == fence_attempt,
                )
                .values(event_seq=indexing_jobs.c.event_seq + 0)
                .returning(indexing_jobs.c.event_seq)
            )
        ).scalar_one_or_none()
        if owned is None:
            log.warning(
                "Progress fuer Reindex-Job %s verworfen — der Lauf "
                "gehoert inzwischen einem anderen Worker-Versuch.",
                job_id,
            )
            return False
        return True

    async def _progress_db(
        self,
        job_id: str,
        *,
        completed_documents: int | None = None,
        current_document_title: str | None = None,
        total_documents: int | None = None,
        fence_attempt: int | None = None,
    ) -> None:
        async with self._session(DEFAULT_TENANT) as session:
            locked = await self._lock_collection_access_for_job(
                session, job_id=job_id
            )
            if locked is None:
                raise AuthorizationRevoked(
                    "reindex requester lost collection edit access"
                )
            _job, _access = locked
            if not await self._fence_ok(session, job_id, fence_attempt):
                return
            values: dict[str, Any] = {}
            if total_documents is not None:
                values["total_documents"] = total_documents
            if completed_documents is not None:
                values["completed_documents"] = completed_documents
            if current_document_title is not None:
                values["current_document_title"] = current_document_title
            updated = (
                await session.execute(
                    update(indexing_jobs)
                    .where(
                        indexing_jobs.c.job_id == job_id,
                        indexing_jobs.c.status.notin_(_TERMINAL_VALUES),
                    )
                    .values(**values)
                    .returning(indexing_jobs.c.job_id)
                )
            ).scalar_one_or_none()
            if updated is None:
                return
            row = await self._row_db(session, job_id)
            await self._append_events_db(
                session,
                job_id,
                row["tenant_id"],
                "inqtrix.index.progress",
                {"snapshot": _snapshot_from_row(row)},
            )

    async def _document_completed_db(
        self,
        job_id: str,
        document_id: str,
        *,
        fence_attempt: int | None = None,
    ) -> None:
        async with self._session(DEFAULT_TENANT) as session:
            locked = await self._lock_collection_access_for_job(
                session, job_id=job_id
            )
            if locked is None:
                raise AuthorizationRevoked(
                    "reindex requester lost collection edit access"
                )
            _job, _access = locked
            if not await self._fence_ok(session, job_id, fence_attempt):
                return
            # A standalone event (no column write); guard against a late event
            # after the job went terminal, mirroring _progress_db's status gate.
            tenant_id = (
                await session.execute(
                    select(indexing_jobs.c.tenant_id).where(
                        indexing_jobs.c.job_id == job_id,
                        indexing_jobs.c.status.notin_(_TERMINAL_VALUES),
                    )
                )
            ).scalar_one_or_none()
            if tenant_id is None:
                return
            await self._append_events_db(
                session,
                job_id,
                tenant_id,
                "inqtrix.index.document_completed",
                {"document_id": document_id, "outcome": "embedded"},
            )

    async def _terminal_db(
        self,
        job_id: str,
        status: IndexingJobStatus,
        *,
        fence_attempt: int | None,
        event_type: str,
        extra: dict[str, Any],
        error: dict[str, Any] | None = None,
        clear_title: bool = False,
        warn_on_noop: bool = True,
    ) -> bool:
        async with self._session(DEFAULT_TENANT) as session:
            locked = await self._lock_collection_access_for_job(
                session, job_id=job_id
            )
            if locked is None:
                return await self._terminalize_revoked_job_db(
                    session,
                    job_id=job_id,
                    fence_attempt=fence_attempt,
                )
            _locked_row, access = locked
            values: dict[str, Any] = {
                "status": (
                    case(
                        (
                            (
                                indexing_jobs.c.cancel_requested.is_(True)
                                | (
                                    indexing_jobs.c.status
                                    == IndexingJobStatus.CANCELLING.value
                                )
                            ),
                            IndexingJobStatus.CANCELLED.value,
                        ),
                        else_=IndexingJobStatus.COMPLETED.value,
                    )
                    if status == IndexingJobStatus.COMPLETED
                    else status.value
                ),
                "finished_at": time.time(),
            }
            if error is not None:
                values["error"] = error
            if clear_title:
                values["current_document_title"] = ""
            query = (
                update(indexing_jobs)
                .where(
                    indexing_jobs.c.job_id == job_id,
                    indexing_jobs.c.status.notin_(_TERMINAL_VALUES),
                )
                .values(**values)
                .returning(indexing_jobs.c.tenant_id)
            )
            if fence_attempt is not None:
                query = query.where(
                    indexing_jobs.c.claimed_by == self._worker_id,
                    indexing_jobs.c.attempt == fence_attempt,
                )
            tenant_id = (await session.execute(query)).scalar_one_or_none()
            if tenant_id is None:
                # Already terminal, missing, or fenced out (a reclaimed
                # zombie) — absorbing states stay absorbing. The discarded
                # write is operator-visible, EXCEPT the auto-complete
                # safety net whose no-op is the expected happy path.
                if warn_on_noop:
                    log.warning(
                        "Terminal-Schreibvorgang fuer Reindex-Job %s "
                        "verworfen (bereits terminal oder von einem anderen "
                        "Worker uebernommen).",
                        job_id,
                    )
                return False
            row = await self._row_db(session, job_id)
            if (
                status == IndexingJobStatus.COMPLETED
                and row["status"] == IndexingJobStatus.CANCELLED.value
            ):
                event_type = "inqtrix.index.cancelled"
                extra = {
                    "status": "cancelled",
                    "reason": "client_requested_cancel",
                }
            await self._append_events_db(
                session,
                job_id,
                tenant_id,
                event_type,
                {**extra, "snapshot": _snapshot_from_row(row)},
            )
            await self._append_collection_effects(
                session,
                row=row,
                access=access,
                action=f"indexing.{row['status']}",
            )
            return True

    async def _terminalize_revoked_job_db(
        self,
        session: "AsyncSession",
        *,
        job_id: str,
        fence_attempt: int | None,
    ) -> bool:
        """Fail a job whose requester lost collection edit authority."""
        error = {
            "message": "Collection authorization was revoked during reindexing.",
            "type": "authorization_revoked",
        }
        query = (
            update(indexing_jobs)
            .where(
                indexing_jobs.c.job_id == job_id,
                indexing_jobs.c.status.notin_(_TERMINAL_VALUES),
            )
            .values(
                status=IndexingJobStatus.FAILED.value,
                finished_at=time.time(),
                current_document_title="",
                error=error,
            )
            .returning(
                indexing_jobs.c.tenant_id,
                indexing_jobs.c.collection_id,
                indexing_jobs.c.created_by_user_id,
            )
        )
        if fence_attempt is not None:
            query = query.where(
                indexing_jobs.c.claimed_by == self._worker_id,
                indexing_jobs.c.attempt == fence_attempt,
            )
        landed = (await session.execute(query)).first()
        if landed is None:
            return False
        row = await self._row_db(session, job_id)
        await self._append_events_db(
            session,
            job_id,
            landed.tenant_id,
            "inqtrix.index.failed",
            {
                "status": "failed",
                "error": error,
                "snapshot": _snapshot_from_row(row),
            },
        )
        owner_user_id = (
            await session.execute(
                select(knowledge_collections.c.created_by_user_id).where(
                    knowledge_collections.c.tenant_id == landed.tenant_id,
                    knowledge_collections.c.id == landed.collection_id,
                )
            )
        ).scalar_one_or_none()
        requester_targets = (
            (landed.created_by_user_id,)
            if landed.created_by_user_id is not None
            else ()
        )
        await append_resource_effects(
            session,
            tenant_id=landed.tenant_id,
            actor_user_id=landed.created_by_user_id,
            owner_user_id=owner_user_id,
            action="indexing.authorization_revoked",
            resource_type="knowledge_collection",
            resource_id=landed.collection_id,
            scope="indexing",
            additional_targets=requester_targets,
        )
        return True

    async def _claim_db(
        self, job_id: str, tenant_id: str, *, allow_takeover: bool
    ) -> ClaimedIndexingJob | None:
        async with self._session(tenant_id) as session:
            locked = await self._lock_collection_access_for_job(
                session, job_id=job_id
            )
            if locked is None:
                await self._terminalize_revoked_job_db(
                    session,
                    job_id=job_id,
                    fence_attempt=None,
                )
                return None
            allowed = [IndexingJobStatus.QUEUED.value]
            if allow_takeover:
                allowed.extend(
                    (
                        IndexingJobStatus.RUNNING.value,
                        IndexingJobStatus.CANCELLING.value,
                    )
                )
            row = (
                await session.execute(
                    update(indexing_jobs)
                    .where(
                        indexing_jobs.c.job_id == job_id,
                        indexing_jobs.c.status.in_(allowed),
                    )
                    .values(
                        status=case(
                            (
                                indexing_jobs.c.cancel_requested.is_(True),
                                IndexingJobStatus.CANCELLING.value,
                            ),
                            else_=IndexingJobStatus.RUNNING.value,
                        ),
                        claimed_by=self._worker_id,
                        attempt=indexing_jobs.c.attempt + 1,
                        started_at=time.time(),
                    )
                    .returning(
                        indexing_jobs.c.attempt,
                        indexing_jobs.c.collection_id,
                        indexing_jobs.c.embedding_model,
                        indexing_jobs.c.created_by_user_id,
                        indexing_jobs.c.created_by_tenant_id,
                        indexing_jobs.c.cancel_requested,
                    )
                )
            ).first()
            if row is None:
                return None
            started_row = await self._row_db(session, job_id)
            await self._append_events_db(
                session,
                job_id,
                tenant_id,
                "inqtrix.index.started",
                {
                    "status": started_row["status"],
                    "snapshot": _snapshot_from_row(started_row),
                },
            )
            return ClaimedIndexingJob(
                job_id=job_id,
                tenant_id=tenant_id,
                attempt=int(row[0]),
                collection_id=row[1],
                embedding_model=row[2],
                created_by_user_id=row[3],
                created_by_tenant_id=row[4],
                cancel_requested=bool(row[5]),
            )

    async def _cancel_requested_db(self, job_ids: dict[str, str]) -> set[str]:
        if not job_ids:
            return set()
        async with self._session(DEFAULT_TENANT) as session:
            rows = (
                await session.execute(
                    select(indexing_jobs.c.job_id).where(
                        indexing_jobs.c.job_id.in_(list(job_ids)),
                        indexing_jobs.c.cancel_requested.is_(True),
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
                    select(
                        indexing_jobs.c.job_id, indexing_jobs.c.tenant_id
                    ).where(
                        indexing_jobs.c.status
                        == IndexingJobStatus.QUEUED.value,
                        indexing_jobs.c.created_at
                        < time.time() - older_than_seconds,
                    )
                )
            ).all()
            return [(row[0], row[1]) for row in rows]

    async def _cleanup_db(self, session: "AsyncSession") -> None:
        """Apply recovery and retention through one locked lifecycle path.

        Collection rows are locked in canonical order before any job row.
        This is the same collection-to-job order used by normal lifecycle
        writes, so cleanup cannot deadlock with a claim, progress write, or
        terminal transition. Active rows are failed rather than deleted;
        terminal history is invalidated and audited before deletion.
        """
        now = time.time()
        recover_orphans = self._sweep_orphans
        recovery_rows = []
        if recover_orphans:
            recovery_rows = (
                await session.execute(
                    select(
                        indexing_jobs.c.job_id,
                        indexing_jobs.c.collection_id,
                    ).where(indexing_jobs.c.status.in_(_ACTIVE_VALUES))
                )
            ).all()
        recovery_ids = {row.job_id for row in recovery_rows}

        candidate_rows = await self._retention_candidate_rows_db(
            session,
            now=now,
        )
        history_rows = await self._history_overflow_rows_db(session)
        candidate_pairs = {
            (row.job_id, row.collection_id)
            for row in (*recovery_rows, *candidate_rows, *history_rows)
        }
        if not candidate_pairs:
            if recover_orphans:
                self._sweep_orphans = False
            return

        collection_ids = tuple(
            sorted({collection_id for _job_id, collection_id in candidate_pairs})
        )
        locked_collections = (
            await session.execute(
                select(
                    knowledge_collections.c.id,
                    knowledge_collections.c.created_by_user_id,
                )
                .where(
                    knowledge_collections.c.tenant_id == DEFAULT_TENANT,
                    knowledge_collections.c.id.in_(collection_ids),
                )
                .order_by(knowledge_collections.c.id)
                .with_for_update()
            )
        ).all()
        owner_by_collection = {
            row.id: row.created_by_user_id for row in locked_collections
        }

        # Re-evaluate ordinary retention after the collection locks land.
        # Recovery deliberately keeps its initial id snapshot: jobs accepted
        # concurrently by this new process are not previous-process orphans.
        candidate_rows = await self._retention_candidate_rows_db(
            session,
            now=now,
            collection_ids=collection_ids,
        )
        history_rows = await self._history_overflow_rows_db(
            session,
            collection_ids=collection_ids,
        )
        history_ids = {row.job_id for row in history_rows}
        job_ids = tuple(
            sorted(
                recovery_ids
                | {row.job_id for row in candidate_rows}
                | history_ids
            )
        )
        locked_jobs = (
            await session.execute(
                select(indexing_jobs)
                .where(indexing_jobs.c.job_id.in_(job_ids))
                .order_by(indexing_jobs.c.job_id)
                .with_for_update()
            )
        ).mappings().all()

        terminal_cutoff = now - self._completed_ttl_seconds
        stuck_cutoff = now - _STUCK_ROW_MAX_AGE_SECONDS
        for row in locked_jobs:
            action = self._maintenance_action_for_row(
                row,
                recovery_ids=recovery_ids,
                history_ids=history_ids,
                terminal_cutoff=terminal_cutoff,
                stuck_cutoff=stuck_cutoff,
            )
            if action is None:
                continue
            await self._apply_maintenance_action_db(
                session,
                row=row,
                owner_user_id=owner_by_collection.get(row["collection_id"]),
                action=action,
                now=now,
            )
        if recover_orphans:
            # Retry on the next transaction if any recovery mutation above
            # raises and rolls this transaction back.
            self._sweep_orphans = False

    async def _retention_candidate_rows_db(
        self,
        session: "AsyncSession",
        *,
        now: float,
        collection_ids: tuple[str, ...] | None = None,
    ) -> list[Any]:
        """Return TTL-expired or non-terminal timeout candidates."""
        terminal_expired = (
            indexing_jobs.c.status.in_(_TERMINAL_VALUES)
            & indexing_jobs.c.finished_at.isnot(None)
            & (
                indexing_jobs.c.finished_at
                < now - self._completed_ttl_seconds
            )
        )
        active_stuck = (
            indexing_jobs.c.status.notin_(_TERMINAL_VALUES)
            & (
                indexing_jobs.c.created_at
                < now - _STUCK_ROW_MAX_AGE_SECONDS
            )
        )
        statement = select(
            indexing_jobs.c.job_id,
            indexing_jobs.c.collection_id,
        ).where(terminal_expired | active_stuck)
        if collection_ids is not None:
            statement = statement.where(
                indexing_jobs.c.collection_id.in_(collection_ids)
            )
        return (await session.execute(statement)).all()

    async def _history_overflow_rows_db(
        self,
        session: "AsyncSession",
        *,
        collection_ids: tuple[str, ...] | None = None,
    ) -> list[Any]:
        """Return terminal jobs beyond the per-collection history cap."""
        ranked_statement = select(
            indexing_jobs.c.job_id,
            indexing_jobs.c.collection_id,
            func.row_number()
            .over(
                partition_by=indexing_jobs.c.collection_id,
                order_by=(
                    func.coalesce(
                        indexing_jobs.c.finished_at,
                        indexing_jobs.c.created_at,
                    ).desc(),
                    indexing_jobs.c.job_id.desc(),
                ),
            )
            .label("rn"),
        ).where(indexing_jobs.c.status.in_(_TERMINAL_VALUES))
        if collection_ids is not None:
            ranked_statement = ranked_statement.where(
                indexing_jobs.c.collection_id.in_(collection_ids)
            )
        ranked = ranked_statement.subquery()
        return (
            await session.execute(
                select(ranked.c.job_id, ranked.c.collection_id).where(
                    ranked.c.rn > self._history_limit
                )
            )
        ).all()

    @staticmethod
    def _maintenance_action_for_row(
        row: Any,
        *,
        recovery_ids: set[str],
        history_ids: set[str],
        terminal_cutoff: float,
        stuck_cutoff: float,
    ) -> _MaintenanceAction | None:
        """Choose one mutation after all lifecycle locks have landed."""
        status = row["status"]
        job_id = row["job_id"]
        if job_id in recovery_ids and status in _ACTIVE_VALUES:
            return _MaintenanceAction(
                action="indexing.server_restarted",
                error={
                    "message": (
                        "Ein Server-Neustart hat die Indizierung unterbrochen."
                    ),
                    "type": "server_restarted",
                },
            )
        if (
            status not in _TERMINAL_VALUES
            and row["created_at"] < stuck_cutoff
        ):
            return _MaintenanceAction(
                action="indexing.stuck_timeout",
                error={
                    "message": (
                        "The reindex job exceeded the maximum lifecycle age."
                    ),
                    "type": "stuck_job_timeout",
                },
                request_cancel=True,
            )
        if (
            status in _TERMINAL_VALUES
            and row["finished_at"] is not None
            and row["finished_at"] < terminal_cutoff
        ):
            return _MaintenanceAction(action="indexing.retention_deleted")
        if status in _TERMINAL_VALUES and job_id in history_ids:
            return _MaintenanceAction(action="indexing.history_evicted")
        return None

    async def _apply_maintenance_action_db(
        self,
        session: "AsyncSession",
        *,
        row: Any,
        owner_user_id: uuid.UUID | None,
        action: _MaintenanceAction,
        now: float,
    ) -> None:
        """Apply one audited terminalization or deletion atomically."""
        if action.error is not None:
            values: dict[str, Any] = {
                "status": IndexingJobStatus.FAILED.value,
                "finished_at": now,
                "current_document_title": "",
                "error": action.error,
            }
            if action.request_cancel:
                values["cancel_requested"] = True
            landed = (
                await session.execute(
                    update(indexing_jobs)
                    .where(
                        indexing_jobs.c.job_id == row["job_id"],
                        indexing_jobs.c.status.notin_(_TERMINAL_VALUES),
                    )
                    .values(**values)
                    .returning(indexing_jobs.c.job_id)
                )
            ).scalar_one_or_none()
            if landed is None:
                return
            row = await self._row_db(session, row["job_id"])
            await self._append_events_db(
                session,
                row["job_id"],
                row["tenant_id"],
                "inqtrix.index.failed",
                {
                    "status": "failed",
                    "error": action.error,
                    "snapshot": _snapshot_from_row(row),
                },
            )
            if action.request_cancel:
                with self._lock:
                    local = self._local.get(row["job_id"])
                    if local is not None:
                        local.cancel_event.set()

        requester_targets = (
            (row["created_by_user_id"],)
            if row["created_by_user_id"] is not None
            else ()
        )
        await append_resource_effects(
            session,
            tenant_id=row["tenant_id"],
            actor_user_id=None,
            owner_user_id=owner_user_id,
            action=action.action,
            resource_type="knowledge_collection",
            resource_id=row["collection_id"],
            scope="indexing",
            additional_targets=requester_targets,
        )
        if action.error is None:
            await session.execute(
                delete(indexing_jobs).where(
                    indexing_jobs.c.job_id == row["job_id"]
                )
            )
        log.warning(
            "Reindex lifecycle maintenance applied %s to job %s.",
            action.action,
            row["job_id"],
        )


def _record_from_row(row: Any) -> IndexingJobRecord:
    """Hydrate an :class:`IndexingJobRecord` from one ``indexing_jobs`` row.

    Only the persisted fields matter to ``build_indexing_job_summary``
    (the in-proc ``cancel_event``/``events``/``subscribers`` defaults are
    unused for summary building); reusing the dataclass keeps the wire
    shape byte-identical to the in-memory store.
    """
    return IndexingJobRecord(
        job_id=row["job_id"],
        collection_id=row["collection_id"],
        collection_name=row["collection_name"],
        embedding_model=row["embedding_model"],
        created_at=row["created_at"],
        index_id=row["index_id"],
        workspace_id=row["workspace_id"],
        created_by_user_id=row["created_by_user_id"],
        created_by_tenant_id=row["created_by_tenant_id"],
        status=IndexingJobStatus(row["status"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        total_documents=row["total_documents"],
        completed_documents=row["completed_documents"],
        current_document_title=row["current_document_title"],
        error=dict(row["error"]) if row["error"] else None,
        event_seq=row["event_seq"],
    )


def _snapshot_from_row(row: Any) -> dict[str, Any]:
    """The progress snapshot for one row (same shape as the in-memory store)."""
    return build_indexing_job_summary(_record_from_row(row))["snapshot"]


def _workspace_matches_row(row: Any, workspace_id: str | None) -> bool:
    return workspace_id is None or row["workspace_id"] == workspace_id


def _visible_row(row: Any, visible_to: "UserContext | None") -> bool:
    """SQL-row twin of the in-memory visibility predicate."""
    if visible_to is None:
        return True
    return (
        row["created_by_user_id"] is not None
        and row["created_by_user_id"] == visible_to.principal.user_id
        and row["created_by_tenant_id"] == visible_to.principal.tenant_id
    )
