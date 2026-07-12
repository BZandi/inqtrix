"""The reindex worker loop: durable background re-embedding.

The reindex twin of :class:`~inqtrix.worker.loop.WorkerLoop`. It reuses
the entire generic claim/fence/heartbeat/reclaim/reconcile/cancel
machinery in :class:`~inqtrix.worker.loop.BaseWorkerLoop` and supplies
only the reindex specifics: the dispatch id field, the stale/cancel
store calls on the reindex stream, and an execution body that runs the
shared :func:`~inqtrix.services.indexing_service.execute_reindex_job`
against a fenced job handle.

The worker has no live principal, so it meters the re-embed against the
``QuotaSubject`` reconstructed from the persisted ``created_by_*``
attribution on the claimed row — exactly as the run worker does.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from inqtrix.quota.models import QuotaSubject
from inqtrix.server.indexing import IndexingJobHandle
from inqtrix.services.indexing_service import execute_reindex_job
from inqtrix.urls import sanitize_error
from inqtrix.worker.loop import _RECONCILE_MIN_AGE_SECONDS, BaseWorkerLoop

if TYPE_CHECKING:
    from inqtrix.runs.indexing_postgres import (
        ClaimedIndexingJob,
        PostgresIndexingJobStore,
    )
    from inqtrix.runs.indexing_queue import QueuedIndexingJob, ValkeyIndexingQueue
    from inqtrix.services.knowledge_service import KnowledgeService
    from inqtrix.services.quota_service import QuotaService

log = logging.getLogger("inqtrix")


class FencedIndexingJobHandle(IndexingJobHandle):
    """Reindex job handle whose writes carry the claim fence.

    The fence ``(claimed_by, attempt)`` makes a reclaimed zombie's late
    progress/terminal writes a visible no-op instead of corrupting the
    superseding attempt's stream. ``terminal_landed`` records whether
    THIS attempt's terminal write actually landed — the worker acks the
    dispatch message only when it did.
    """

    def __init__(
        self,
        store: "PostgresIndexingJobStore",
        job_id: str,
        cancel_event: threading.Event,
        attempt: int,
    ) -> None:
        super().__init__(store, job_id, cancel_event)
        self._fence_attempt = attempt
        self.terminal_landed = False

    def begin(self, total_documents: int) -> None:
        """Record the total document count, fenced to this attempt."""
        self._store.set_total(
            self.job_id, total_documents, fence_attempt=self._fence_attempt
        )

    def progress(
        self, *, completed_documents: int, current_document_title: str = ""
    ) -> None:
        """Emit one progress step, fenced to this attempt."""
        self._store.progress(
            self.job_id,
            completed_documents=completed_documents,
            current_document_title=current_document_title,
            fence_attempt=self._fence_attempt,
        )

    def document_completed(self, document_id: str) -> None:
        """Emit a per-document completion event, fenced to this attempt."""
        self._store.document_completed(
            self.job_id, document_id, fence_attempt=self._fence_attempt
        )

    def complete(self) -> None:
        """Mark the job completed, fenced to this attempt."""
        self.terminal_landed = self._store.complete(
            self.job_id, fence_attempt=self._fence_attempt
        )

    def fail(self, message: str, *, error_type: str = "server_error") -> None:
        """Mark the job failed, fenced to this attempt."""
        self.terminal_landed = self._store.fail(
            self.job_id,
            message,
            error_type=error_type,
            fence_attempt=self._fence_attempt,
        )

    def cancel(self, reason: str = "cancelled") -> None:
        """Mark the job cancelled, fenced to this attempt."""
        self.terminal_landed = self._store.mark_cancelled(
            self.job_id, reason=reason, fence_attempt=self._fence_attempt
        )


class IndexingWorkerLoop(BaseWorkerLoop["QueuedIndexingJob", "ClaimedIndexingJob"]):
    """Claim-and-execute loop for durable reindex jobs.

    Args:
        store: Postgres reindex-job store (claims, events, terminal
            writes).
        queue: Reindex Valkey queue bound to this worker's consumer name.
        knowledge_service: The collection/document service whose store
            and re-embed pipeline the worker drives.
        concurrency: Parallel re-embeds in this process.
        max_attempts: Delivery budget before dead-lettering.
        heartbeat_seconds: Idle-reset interval for in-flight entries.
        claim_idle_seconds: Reclaim threshold for entries whose owner
            stopped heartbeating.
        quota_service: Optional usage meter for incremental
            embedding-token accounting (keyed by the persisted subject).
    """

    def __init__(
        self,
        *,
        store: "PostgresIndexingJobStore",
        queue: "ValkeyIndexingQueue",
        knowledge_service: "KnowledgeService",
        concurrency: int,
        max_attempts: int,
        heartbeat_seconds: float,
        claim_idle_seconds: float,
        quota_service: "QuotaService | None" = None,
    ) -> None:
        super().__init__(
            store=store,
            queue=queue,
            concurrency=concurrency,
            max_attempts=max_attempts,
            heartbeat_seconds=heartbeat_seconds,
            claim_idle_seconds=claim_idle_seconds,
            thread_prefix="inqtrix-reindex",
        )
        self._knowledge_service = knowledge_service
        self._quota_service = quota_service

    def _entity_id(self, job: "QueuedIndexingJob") -> str:
        return job.job_id

    def _stale_dispatch(self) -> list[tuple[str, str]]:
        return self._store.stale_queued_jobs(
            older_than_seconds=_RECONCILE_MIN_AGE_SECONDS
        )

    def _cancel_requested(self, watched: dict[str, str]) -> set[str]:
        return self._store.cancel_requested_jobs(watched)

    def _enqueue_dispatch(self, entity_id: str, tenant_id: str) -> None:
        self._queue.enqueue(job_id=entity_id, tenant_id=tenant_id)

    def _execute(
        self,
        job: "QueuedIndexingJob",
        claimed: "ClaimedIndexingJob",
        cancel_event: threading.Event,
    ) -> None:
        handle = FencedIndexingJobHandle(
            self._store, job.job_id, cancel_event, claimed.attempt
        )
        old_message_acked = False
        try:
            try:
                # Reconstruct the metered subject from the persisted job
                # attribution — the worker has no live principal, but the
                # embedding-token spend must still count toward the
                # submitter's monthly quota.
                quota_subject = None
                if (
                    self._quota_service is not None
                    and claimed.created_by_sub
                    and claimed.created_by_tenant_id
                ):
                    quota_subject = QuotaSubject(
                        tenant_id=claimed.created_by_tenant_id,
                        sub=claimed.created_by_sub,
                    )
                execute_reindex_job(
                    handle,
                    knowledge_service=self._knowledge_service,
                    collection_id=claimed.collection_id,
                    embedding_model=claimed.embedding_model,
                    quota_service=self._quota_service,
                    quota_subject=quota_subject,
                )
            except Exception as exc:  # noqa: BLE001 — terminal-write then ack
                log.exception("Worker-Reindex %s fehlgeschlagen", job.job_id)
                handle.fail(sanitize_error(exc))
            if handle.terminal_landed:
                # Terminal state is committed; only now may the stream
                # forget the job.
                self._queue.ack(job.message_id)
                old_message_acked = True
            else:
                # Fenced out: a superseding attempt owns the job AND this
                # very message id — acking here would strip the new
                # owner's crash-recovery entry.
                log.warning(
                    "Worker %s: Reindex-Job %s wurde waehrend der "
                    "Ausfuehrung von einem anderen Worker uebernommen — "
                    "Ergebnis verworfen, Nachricht bleibt beim neuen Owner.",
                    self._store.worker_id,
                    job.job_id,
                )
        except Exception:  # noqa: BLE001 — Futures here are unobserved
            log.exception(
                "Worker %s: Abschlussphase fuer Reindex-Job %s "
                "fehlgeschlagen — Redelivery uebernimmt.",
                self._store.worker_id,
                job.job_id,
            )
        finally:
            self._finish_active(
                job, allow_successor=old_message_acked
            )
