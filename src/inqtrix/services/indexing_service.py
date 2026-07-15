"""Background reindex submission and the per-document re-embed worker.

The reindex counterpart of :mod:`inqtrix.services.run_service`: it owns
the work-callable that re-embeds a collection's documents one by one,
checking the cancel token between documents, emitting progress, and
booking embedding-token usage incrementally so a cancelled run never
over-charges. Read-side operations stay on the
:class:`~inqtrix.server.indexing.IndexingJobStore` and are called by the
router directly.

A reindex is rebuild-in-place (each document keeps its id, only its
vectors are recomputed), so there is no half-built collection to unwind
on cancel or failure — the collection stays consistent throughout.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable

from inqtrix.knowledge.stores.ports import (
    CollectionNotFound,
    DocumentNotFound,
    KnowledgeCollection,
)
from inqtrix.quota.models import QuotaDimension, estimate_tokens
from inqtrix.sync_bridge import run_coro_sync

if TYPE_CHECKING:
    from inqtrix.auth.principal import Principal
    from inqtrix.quota.models import QuotaSubject
    from inqtrix.server.indexing import IndexingJobHandle, IndexingJobStore
    from inqtrix.services.knowledge_service import KnowledgeService
    from inqtrix.services.quota_service import QuotaService
    from inqtrix.services.execution_dependency_authority import (
        CollectionEditAuthorizer,
    )

log = logging.getLogger("inqtrix")


class ReindexUnsupported(RuntimeError):
    """Raised when the active vector store cannot re-embed in place.

    A store without
    :meth:`~inqtrix.knowledge.stores.ports.KnowledgeStore.reembed_document`
    cannot back a background reindex; the router maps this to HTTP 501
    so the missing capability is visible, never a silent no-op
    (Designprinzip 1 / capability-gated degradation).
    """


def store_supports_reembed(knowledge_service: "KnowledgeService") -> bool:
    """Whether the store can re-embed behind a mutation boundary."""
    store = knowledge_service.knowledge.store
    return callable(getattr(store, "reembed_document", None)) and bool(
        getattr(store, "supports_safe_reindex", True)
    )


def execute_reindex_job(
    handle: "IndexingJobHandle",
    *,
    knowledge_service: "KnowledgeService",
    collection_id: str,
    embedding_model: str,
    quota_service: "QuotaService | None" = None,
    quota_subject: "QuotaSubject | None" = None,
    authority_check: Callable[[], None] | None = None,
    actor_user_id: uuid.UUID | None = None,
) -> None:
    """Re-embed every document in one collection, emitting progress.

    The single execution body shared by the in-process dispatch thread:
    it enumerates the collection's documents, re-embeds each through the
    same pipeline as first-time ingestion, checks the cancel token
    between documents, and reports document-level progress. Exceptions
    propagate to the store worker, which owns the failure path.

    Args:
        handle: Store-backed job handle (progress, complete, cancel).
        collection_id: The collection whose documents are re-embedded.
        embedding_model: The collection's immutable embedding model id.
        quota_service: Optional usage meter. When wired, the embedded
            text of each document is booked against ``EMBEDDING_TOKENS``
            as it completes (incremental, so a cancelled run only counts
            what it actually embedded). ``None`` for unmetered
            deployments.
        quota_subject: Explicit quota account, reconstructed from the job's
            canonical user UUID and tenant — token accounting fires
            regardless of which thread executes the job.
    """
    store = knowledge_service.knowledge.store
    if authority_check is not None:
        authority_check()
    # The reindex worker is synchronous; the knowledge store/service are
    # async. Bridge each call to completion on this worker thread.
    documents = run_coro_sync(store.list_documents(collection_id))
    handle.begin(len(documents))
    for index, listed_document in enumerate(documents):
        if authority_check is not None:
            authority_check()
        if handle.cancelled:
            handle.cancel("client_requested_cancel")
            return
        try:
            # Enumeration fixes only the work list. The canonical document is
            # reloaded immediately before embedding so a preceding mutation
            # that committed before the maintenance boundary cannot feed stale
            # title/text/metadata into this pass.
            document = run_coro_sync(store.get_document(listed_document.id))
            if authority_check is not None:
                authority_check()
            handle.progress(
                completed_documents=index,
                current_document_title=document.title,
            )
            run_coro_sync(
                knowledge_service.reembed_document(
                    document=document,
                    embedding_model=embedding_model,
                    authority_check=authority_check,
                    actor_user_id=actor_user_id,
                )
            )
        except DocumentNotFound:
            # Deleted between enumeration and re-embed: visible, not
            # fatal — the remaining documents still get re-embedded.
            log.warning(
                "Reindex %s: document %s vanished mid-run; skipping",
                collection_id,
                listed_document.id,
            )
            continue
        # This document is now re-embedded — emit a per-document event so the
        # UI flips just this file's row, not all files together on completion.
        handle.document_completed(document.id)
        if quota_service is not None and quota_subject is not None:
            quota_service.record_blocking(
                quota_subject,
                QuotaDimension.EMBEDDING_TOKENS,
                estimate_tokens(document.text),
            )
    if handle.cancelled:
        handle.cancel("client_requested_cancel")
        return
    if authority_check is not None:
        authority_check()
    handle.progress(
        completed_documents=len(documents), current_document_title=""
    )
    handle.complete()


class IndexingService:
    """Submit background reindex jobs over the knowledge collections.

    Args:
        knowledge_service: The collection/document service whose store
            and re-embed pipeline the worker drives.
        job_store: The in-memory reindex registry/queue that owns
            dispatch, events, and retention.
        quota_service: Optional usage meter threaded into the worker for
            incremental embedding-token accounting.
    """

    def __init__(
        self,
        *,
        knowledge_service: "KnowledgeService",
        job_store: "IndexingJobStore",
        quota_service: "QuotaService | None" = None,
        authority: "CollectionEditAuthorizer | None" = None,
    ) -> None:
        self._knowledge_service = knowledge_service
        self._job_store = job_store
        self._quota_service = quota_service
        self._authority = authority
        knowledge_service.bind_collection_maintenance(
            active_check=job_store.has_active_job,
            mutation_runner=getattr(job_store, "run_collection_mutation", None),
        )

    @property
    def job_store(self) -> "IndexingJobStore":
        """The underlying job store (read-side surface for the router)."""
        return self._job_store

    @property
    def authority(self) -> "CollectionEditAuthorizer | None":
        """Live collection-edit checker shared with queue workers."""
        return self._authority

    def submit(
        self,
        *,
        collection: KnowledgeCollection,
        index_id: str | None = None,
        workspace_id: str | None = None,
        principal: "Principal | None" = None,
    ) -> dict[str, Any]:
        """Queue one reindex job for *collection* and return its summary.

        Access to the collection is the router's concern (checked before
        this call); this method only wires the work closure and the
        canonical quota attribution.

        Raises:
            ReindexUnsupported: The active vector store cannot re-embed
                in place.
            inqtrix.server.indexing.IndexingJobConflict: The collection
                already has an active reindex job (mapped to HTTP 409).
            inqtrix.server.indexing.IndexingQueueFull: The waiting queue
                is full (mapped to HTTP 429).
        """
        if not store_supports_reembed(self._knowledge_service):
            raise ReindexUnsupported(
                "the active knowledge store cannot safely serialize reindexing"
            )
        quota_subject = (
            self._quota_service.subject_for(principal)
            if self._quota_service is not None
            else None
        )
        collection_id = collection.id
        embedding_model = collection.embedding_model

        def _check_authority() -> None:
            if self._authority is not None:
                self._authority.check(collection_id, principal)

        def _work(handle: "IndexingJobHandle") -> None:
            execute_reindex_job(
                handle,
                knowledge_service=self._knowledge_service,
                collection_id=collection_id,
                embedding_model=embedding_model,
                quota_service=self._quota_service,
                quota_subject=quota_subject,
                authority_check=_check_authority,
                actor_user_id=(
                    principal.user_id if principal is not None else None
                ),
            )

        return self._job_store.submit(
            collection_id=collection_id,
            collection_name=collection.name,
            embedding_model=embedding_model,
            work=_work,
            index_id=index_id,
            workspace_id=workspace_id,
            created_by_user_id=principal.user_id if principal is not None else None,
            created_by_tenant_id=(
                principal.tenant_id if principal is not None else None
            ),
        )
