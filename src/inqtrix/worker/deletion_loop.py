"""Claim loop for durable aggregate deletion operations."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.execution_failures import classify_execution_failure
from inqtrix.runs.deletion_operations import (
    DeletionAttemptSuperseded,
    DeletionJobHandle,
    DeletionStage,
)
from inqtrix.urls import sanitize_error
from inqtrix.worker.loop import BaseWorkerLoop, _RECONCILE_MIN_AGE_SECONDS

if TYPE_CHECKING:
    from inqtrix.runs.deletion_postgres import (
        ClaimedDeletionOperation,
        PostgresDeletionOperationStore,
    )
    from inqtrix.runs.deletion_queue import (
        QueuedDeletionOperation,
        ValkeyDeletionQueue,
    )
    from inqtrix.services.asset_deletion_service import AssetDeletionService

log = logging.getLogger("inqtrix")


class FencedDeletionJobHandle(DeletionJobHandle):
    def __init__(
        self,
        store: "PostgresDeletionOperationStore",
        operation_id: str,
        attempt: int,
    ) -> None:
        super().__init__(store, operation_id)
        self._attempt = attempt
        self.manages_asset_lifecycle = True

    def assert_current(self) -> None:
        if not self._store.is_attempt_current(
            self.operation_id, fence_attempt=self._attempt
        ):
            raise DeletionAttemptSuperseded(self.operation_id)

    def checkpoint_source_cleanup(
        self, asset_id: str, plan: dict[str, object]
    ) -> None:
        landed = self._store.checkpoint_source_cleanup(
            self.operation_id,
            asset_id=asset_id,
            plan=plan,
            fence_attempt=self._attempt,
        )
        if not landed:
            raise DeletionAttemptSuperseded(self.operation_id)

    def source_deletion_permit(self, scope):
        self.assert_current()
        return self._store.source_deletion_permit(
            self.operation_id,
            scope=scope,
            fence_attempt=self._attempt,
        )

    def progress(
        self,
        stage: DeletionStage,
        *,
        completed_items: int,
        total_items: int,
    ) -> None:
        landed = self._store.progress(
            self.operation_id,
            stage=stage,
            completed_items=completed_items,
            total_items=total_items,
            fence_attempt=self._attempt,
        )
        if not landed:
            raise DeletionAttemptSuperseded(self.operation_id)

    def complete(self) -> None:
        self.terminal_landed = self._store.complete(
            self.operation_id, fence_attempt=self._attempt
        )
        if not self.terminal_landed:
            raise DeletionAttemptSuperseded(self.operation_id)

    def fail(self, message: str, *, error_type: str = "server_error") -> None:
        self.terminal_landed = self._store.fail(
            self.operation_id,
            message,
            error_type=error_type,
            fence_attempt=self._attempt,
        )


class DeletionWorkerLoop(
    BaseWorkerLoop["QueuedDeletionOperation", "ClaimedDeletionOperation"]
):
    def __init__(
        self,
        *,
        store: "PostgresDeletionOperationStore",
        queue: "ValkeyDeletionQueue",
        service: "AssetDeletionService",
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
            thread_prefix="inqtrix-delete",
        )
        self._service = service

    def _entity_id(self, job: "QueuedDeletionOperation") -> str:
        return job.operation_id

    def _stale_dispatch(self) -> list[tuple[str, str]]:
        return self._store.stale_queued_operations(
            older_than_seconds=_RECONCILE_MIN_AGE_SECONDS
        )

    def _cancel_requested(self, watched: dict[str, str]) -> set[str]:
        return self._store.cancel_requested_operations(watched)

    def _enqueue_dispatch(self, entity_id: str, tenant_id: str) -> None:
        self._queue.enqueue(operation_id=entity_id, tenant_id=tenant_id)

    def _execute(
        self,
        job: "QueuedDeletionOperation",
        claimed: "ClaimedDeletionOperation",
        cancel_event: threading.Event,
    ) -> None:
        del cancel_event  # deletion is intentionally not reversible mid-saga
        handle = FencedDeletionJobHandle(
            self._store,
            job.operation_id,
            claimed.attempt,
        )
        old_message_acked = False
        try:
            principal = Principal(
                user_id=claimed.created_by_user_id,
                kind=(
                    "oidc_session"
                    if claimed.created_by_user_id is not None
                    else "anonymous"
                ),
                tenant_id=claimed.tenant_id,
                role=(
                    "member"
                    if claimed.created_by_user_id is not None
                    else "owner"
                ),
            )
            visible_to = UserContext(
                principal=principal,
                workspace_ids=(
                    (claimed.workspace_id,) if claimed.workspace_id else ()
                ),
            )
            try:
                self._service.execute(
                    handle,
                    manifest=claimed.manifest,
                    vector_index_context=claimed.vector_index_context,
                    knowledge_context=claimed.knowledge_context,
                    session_context=claimed.session_context,
                    target_kind=claimed.target_kind,
                    target_id=claimed.target_id,
                    principal=principal,
                    visible_to=visible_to,
                    workspace_id=claimed.workspace_id,
                )
            except DeletionAttemptSuperseded:
                log.info(
                    "Worker-Loeschoperation %s wurde durch einen neueren "
                    "Versuch abgeloest.",
                    job.operation_id,
                )
            except Exception as exc:  # noqa: BLE001 - terminal write then ACK
                log.error(
                    "Worker-Loeschoperation %s fehlgeschlagen "
                    "(error_type=%s)",
                    job.operation_id,
                    type(exc).__name__,
                )
                handle.fail(
                    sanitize_error(exc),
                    error_type=classify_execution_failure(exc),
                )
            if handle.terminal_landed:
                self._queue.ack(job.message_id)
                old_message_acked = True
            else:
                log.warning(
                    "Worker %s: Loeschoperation %s wurde durch einen "
                    "neueren Versuch gezaunt; Nachricht bleibt beim neuen Owner.",
                    self._store.worker_id,
                    job.operation_id,
                )
        except Exception as exc:
            log.error(
                "Worker %s: Abschlussphase fuer Loeschoperation %s "
                "fehlgeschlagen; Redelivery uebernimmt (error_type=%s).",
                self._store.worker_id,
                job.operation_id,
                type(exc).__name__,
            )
        finally:
            self._finish_active(job, allow_successor=old_message_acked)
