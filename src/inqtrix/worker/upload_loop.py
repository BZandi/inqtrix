"""Claim loop for durable original-file upload recovery."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from inqtrix.runs.upload_operations import UploadAttemptSuperseded
from inqtrix.services.upload_operation_service import (
    UploadBytesRequired,
    UploadExecutionDeferred,
)
from inqtrix.worker.loop import _RECONCILE_MIN_AGE_SECONDS, BaseWorkerLoop

if TYPE_CHECKING:
    from inqtrix.runs.upload_postgres import (
        ClaimedUploadOperation,
        PostgresUploadOperationStore,
    )
    from inqtrix.runs.upload_queue import QueuedUploadOperation, ValkeyUploadQueue
    from inqtrix.services.upload_operation_service import UploadOperationService

log = logging.getLogger("inqtrix")


class UploadWorkerLoop(
    BaseWorkerLoop["QueuedUploadOperation", "ClaimedUploadOperation"]
):
    def __init__(
        self,
        *,
        store: "PostgresUploadOperationStore",
        queue: "ValkeyUploadQueue",
        service: "UploadOperationService",
        concurrency: int,
        max_attempts: int,
        heartbeat_seconds: float,
        claim_idle_seconds: float,
        claim_guard=None,
    ) -> None:
        super().__init__(
            store=store,
            queue=queue,
            concurrency=concurrency,
            max_attempts=max_attempts,
            heartbeat_seconds=heartbeat_seconds,
            claim_idle_seconds=claim_idle_seconds,
            claim_guard=claim_guard,
            thread_prefix="inqtrix-upload",
        )
        self._service = service

    def _entity_id(self, job: "QueuedUploadOperation") -> str:
        return job.operation_id

    def _stale_dispatch(self) -> list[tuple[str, str]]:
        return self._store.stale_dispatches(
            older_than_seconds=_RECONCILE_MIN_AGE_SECONDS
        )

    def _cancel_requested(self, watched: dict[str, str]) -> set[str]:
        del watched
        return set()

    def _enqueue_dispatch(self, entity_id: str, tenant_id: str) -> None:
        self._queue.enqueue(operation_id=entity_id, tenant_id=tenant_id)

    def _execute(
        self,
        job: "QueuedUploadOperation",
        claimed: "ClaimedUploadOperation",
        cancel_event: threading.Event,
    ) -> None:
        del cancel_event
        ack = False
        try:
            try:
                self._service.execute_claimed(claimed)
            except (UploadBytesRequired, UploadExecutionDeferred):
                # Both states were persisted before control reached here.
                pass
            except UploadAttemptSuperseded:
                log.info(
                    "Upload-Operation %s wurde von einem neueren Versuch uebernommen.",
                    job.operation_id,
                )
                return
            except Exception as exc:
                # The service persists a typed failure or retry before raising.
                log.warning(
                    "Upload-Operation %s endete mit einem sichtbaren Fehler "
                    "(error_type=%s).",
                    job.operation_id,
                    type(exc).__name__,
                )
            self._queue.ack(job.message_id)
            ack = True
        finally:
            self._finish_active(job, allow_successor=ack)
