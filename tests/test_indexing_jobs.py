"""Tests for the background reindex (re-embed) job subsystem.

Covers the in-memory :class:`IndexingJobStore` (dispatch, per-collection
serialization, queue cap, cancel, visibility, history retention) and the
:func:`execute_reindex_job` worker (progress, incremental quota,
cancel-between-documents, unsupported-store guard).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable

import pytest

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import DocumentNotFound, KnowledgeProviderContext
from inqtrix.quota.models import QuotaDimension, QuotaSubject
from inqtrix.server.indexing import (
    IndexingJobConflict,
    IndexingJobHandle,
    IndexingJobNotFound,
    IndexingJobStore,
    IndexingQueueFull,
    format_sse_event,
)
from inqtrix.services.indexing_service import (
    IndexingService,
    ReindexUnsupported,
    execute_reindex_job,
)
from inqtrix.services.knowledge_service import KnowledgeService
from inqtrix.settings import KnowledgeSettings

from tests.test_knowledge_engine import StubEmbeddings, make_knowledge_context


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    """Wait until *predicate* becomes true or fail the test."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition was not reached before timeout")


def _store(
    *,
    max_concurrent: int = 1,
    max_queue_size: int = 5,
    completed_ttl_seconds: int = 30,
    event_buffer_size: int = 50,
    history_limit: int = 10,
) -> IndexingJobStore:
    """Create a small reindex job store for unit tests."""
    return IndexingJobStore(
        max_concurrent=max_concurrent,
        max_queue_size=max_queue_size,
        completed_ttl_seconds=completed_ttl_seconds,
        event_buffer_size=event_buffer_size,
        history_limit=history_limit,
    )


def _complete_now(handle: IndexingJobHandle) -> None:
    """Trivial work that finishes immediately with no documents."""
    handle.begin(0)
    handle.complete()


def _submit(store: IndexingJobStore, *, collection_id: str, work=_complete_now, **kw):
    """Submit with sensible defaults for the store-only tests."""
    return store.submit(
        collection_id=collection_id,
        collection_name=kw.pop("collection_name", "Collection"),
        embedding_model=kw.pop("embedding_model", "stub-embed-8"),
        work=work,
        **kw,
    )


# ------------------------------------------------------------------ #
# Store: construction / settings
# ------------------------------------------------------------------ #


def test_from_settings_uses_reindex_knobs() -> None:
    settings = KnowledgeSettings(
        INQTRIX_REINDEX_MAX_CONCURRENT=2,
        INQTRIX_REINDEX_QUEUE_MAX_SIZE=7,
        INQTRIX_REINDEX_COMPLETED_TTL_SECONDS=99,
        INQTRIX_REINDEX_EVENT_BUFFER_SIZE=11,
        INQTRIX_REINDEX_HISTORY_LIMIT=3,
    )
    store = IndexingJobStore.from_settings(settings)
    assert store._max_concurrent == 2
    assert store._max_queue_size == 7
    assert store._completed_ttl_seconds == 99
    assert store._event_buffer_size == 11
    assert store._history_limit == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_concurrent": 0}, "max_concurrent must be >= 1"),
        ({"max_queue_size": -1}, "max_queue_size must be >= 0"),
        ({"completed_ttl_seconds": -1}, "completed_ttl_seconds must be >= 0"),
        ({"event_buffer_size": 0}, "event_buffer_size must be >= 1"),
        ({"history_limit": -1}, "history_limit must be >= 0"),
    ],
)
def test_store_rejects_invalid_limits(kwargs: dict, message: str) -> None:
    options = {
        "max_concurrent": 1,
        "max_queue_size": 1,
        "completed_ttl_seconds": 30,
        "event_buffer_size": 10,
        "history_limit": 5,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        IndexingJobStore(**options)


# ------------------------------------------------------------------ #
# Store: dispatch / lifecycle
# ------------------------------------------------------------------ #


def test_submit_dispatches_and_completes() -> None:
    store = _store()
    summary = _submit(store, collection_id="kc1")
    assert summary["status"] in {"queued", "running", "completed"}
    assert summary["collection_id"] == "kc1"
    assert summary["events_url"] == f"/v1/knowledge/indexing-jobs/{summary['job_id']}/events"
    _wait_until(lambda: store.get(summary["job_id"])["status"] == "completed")
    final = store.get(summary["job_id"])
    assert final["percent"] == 100
    assert final["snapshot"]["progress_estimate"] == 1.0


def test_one_active_job_per_collection_conflicts() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking(handle: IndexingJobHandle) -> None:
        handle.begin(1)
        started.set()
        release.wait(timeout=2)
        handle.progress(completed_documents=1)
        handle.complete()

    store = _store(max_concurrent=1, max_queue_size=5)
    first = _submit(store, collection_id="kcA", work=blocking)
    _wait_until(started.is_set)
    with pytest.raises(IndexingJobConflict):
        _submit(store, collection_id="kcA")
    release.set()
    _wait_until(lambda: store.get(first["job_id"])["status"] == "completed")
    # A terminal job no longer blocks a fresh reindex of the same collection.
    again = _submit(store, collection_id="kcA")
    _wait_until(lambda: store.get(again["job_id"])["status"] == "completed")


def test_queue_full_raises() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking(handle: IndexingJobHandle) -> None:
        handle.begin(1)
        started.set()
        release.wait(timeout=2)
        handle.complete()

    store = _store(max_concurrent=1, max_queue_size=0)
    _submit(store, collection_id="kcA", work=blocking)
    _wait_until(started.is_set)
    with pytest.raises(IndexingQueueFull):
        _submit(store, collection_id="kcB")
    release.set()


def test_cancel_queued_job_marks_cancelled() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking(handle: IndexingJobHandle) -> None:
        started.set()
        release.wait(timeout=2)
        handle.complete()

    store = _store(max_concurrent=1, max_queue_size=5)
    running = _submit(store, collection_id="kcA", work=blocking)
    _wait_until(started.is_set)
    queued = _submit(store, collection_id="kcB")
    assert store.get(queued["job_id"])["status"] == "queued"
    cancelled = store.cancel(queued["job_id"])
    assert cancelled["status"] == "cancelled"
    release.set()
    _wait_until(lambda: store.get(running["job_id"])["status"] == "completed")


def test_cancel_running_job_is_observed_by_worker() -> None:
    started = threading.Event()

    def cooperative(handle: IndexingJobHandle) -> None:
        handle.begin(3)
        started.set()
        # Spin until cancellation is requested, then exit cleanly.
        for _ in range(500):
            if handle.cancelled:
                handle.cancel("client_requested_cancel")
                return
            time.sleep(0.005)
        handle.complete()

    store = _store(max_concurrent=1)
    summary = _submit(store, collection_id="kcA", work=cooperative)
    _wait_until(started.is_set)
    store.cancel(summary["job_id"])
    _wait_until(lambda: store.get(summary["job_id"])["status"] == "cancelled")


# ------------------------------------------------------------------ #
# Store: visibility / retention / events
# ------------------------------------------------------------------ #


def test_visibility_scopes_jobs_to_creator() -> None:
    store = _store()
    owner = UserContext(principal=Principal(sub="owner", kind="oidc_session"))
    stranger = UserContext(principal=Principal(sub="stranger", kind="oidc_session"))
    summary = _submit(
        store,
        collection_id="kc1",
        created_by_sub="owner",
        created_by_tenant_id="default",
    )
    _wait_until(lambda: store.get(summary["job_id"])["status"] == "completed")
    assert len(store.list(visible_to=owner)) == 1
    assert store.list(visible_to=stranger) == []
    assert store.get(summary["job_id"], visible_to=owner)["status"] == "completed"
    with pytest.raises(IndexingJobNotFound):
        store.get(summary["job_id"], visible_to=stranger)


def test_history_capped_per_collection() -> None:
    store = _store(history_limit=2)
    seen: list[str] = []
    for _ in range(3):
        summary = _submit(store, collection_id="kcH")
        job_id = summary["job_id"]
        _wait_until(
            lambda jid=job_id: any(
                job["job_id"] == jid and job["status"] == "completed"
                for job in store.list(collection_id="kcH")
            )
        )
        seen.append(job_id)
    remaining = {job["job_id"] for job in store.list(collection_id="kcH")}
    assert len(remaining) == 2
    # The two NEWEST terminal records survive; the oldest is evicted.
    assert seen[0] not in remaining
    assert seen[1] in remaining
    assert seen[2] in remaining


def test_subscribe_replays_lifecycle_events() -> None:
    store = _store()
    summary = _submit(store, collection_id="kc1")
    _wait_until(lambda: store.get(summary["job_id"])["status"] == "completed")
    subscription = store.subscribe(summary["job_id"])
    types = [event["type"] for event in subscription.replay]
    subscription.close()
    assert "inqtrix.index.queued" in types
    assert "inqtrix.index.started" in types
    assert "inqtrix.index.completed" in types
    assert types[-1] == "inqtrix.index.completed"


def test_format_sse_event_renders_frame() -> None:
    frame = format_sse_event(
        {"type": "inqtrix.index.progress", "job_id": "ix_1", "sequence": 2, "data": {}}
    )
    assert frame.startswith("event: inqtrix.index.progress\n")
    assert "\n\n" in frame


# ------------------------------------------------------------------ #
# Worker: execute_reindex_job + IndexingService
# ------------------------------------------------------------------ #


def _service_with_docs(*texts: str) -> tuple[KnowledgeService, StubEmbeddings, object]:
    embeddings = StubEmbeddings()
    context = make_knowledge_context(embeddings=embeddings)
    service = KnowledgeService(
        knowledge=context, chunk_max_chars=2_000, max_document_chars=100_000
    )

    async def _seed():
        collection = await service.create_collection(name="C")
        for index, text in enumerate(texts):
            await service.add_document(
                collection_id=collection.id, title=f"Doc {index}", text=text
            )
        return collection

    collection = asyncio.run(_seed())
    return service, embeddings, collection


def test_reindex_reembeds_every_document() -> None:
    service, embeddings, collection = _service_with_docs("alpha beta", "gamma delta")
    calls_before = embeddings.document_calls
    store = _store()
    indexing = IndexingService(knowledge_service=service, job_store=store)
    summary = indexing.submit(collection=collection)
    _wait_until(lambda: store.get(summary["job_id"])["status"] == "completed")
    final = store.get(summary["job_id"])
    assert final["total_documents"] == 2
    assert final["completed_documents"] == 2
    assert final["percent"] == 100
    # Both documents were embedded a second time by the reindex.
    assert embeddings.document_calls == calls_before + 2


def test_reindex_emits_one_document_completed_event_per_document() -> None:
    service, _embeddings, collection = _service_with_docs("alpha beta", "gamma delta")
    store = _store()
    indexing = IndexingService(knowledge_service=service, job_store=store)
    summary = indexing.submit(collection=collection)
    _wait_until(lambda: store.get(summary["job_id"])["status"] == "completed")
    replay = store.subscribe(summary["job_id"]).replay
    doc_events = [
        event for event in replay
        if event["type"] == "inqtrix.index.document_completed"
    ]
    # One per re-embedded document, each carrying its backend document id.
    assert len(doc_events) == 2
    assert all(event["data"]["outcome"] == "embedded" for event in doc_events)
    assert all(event["data"]["document_id"] for event in doc_events)
    # The per-document events land before the terminal completed event.
    types = [event["type"] for event in replay]
    last_doc = max(
        index for index, kind in enumerate(types)
        if kind == "inqtrix.index.document_completed"
    )
    assert types.index("inqtrix.index.completed") > last_doc


def test_reindex_emits_no_document_completed_for_a_vanished_document(monkeypatch) -> None:
    service, _embeddings, collection = _service_with_docs("alpha", "beta")
    store = _store()
    # One document is deleted between enumeration and re-embed (DocumentNotFound):
    # it must be skipped WITHOUT emitting a per-document event — only the
    # surviving document flips its file row.
    real_reembed = service.reembed_document

    async def vanishing_reembed(*, document, embedding_model):
        if document.title == "Doc 0":
            raise DocumentNotFound(document.id)
        return await real_reembed(document=document, embedding_model=embedding_model)

    monkeypatch.setattr(service, "reembed_document", vanishing_reembed)
    indexing = IndexingService(knowledge_service=service, job_store=store)
    summary = indexing.submit(collection=collection)
    _wait_until(lambda: store.get(summary["job_id"])["status"] == "completed")
    replay = store.subscribe(summary["job_id"]).replay
    doc_events = [
        event for event in replay
        if event["type"] == "inqtrix.index.document_completed"
    ]
    assert len(doc_events) == 1


def test_reindex_records_embedding_quota_per_document() -> None:
    service, _embeddings, collection = _service_with_docs("alpha beta", "gamma delta")
    principal = Principal(sub="user-1", kind="oidc_session")

    class FakeQuota:
        def __init__(self) -> None:
            self.records: list[tuple] = []

        def subject_for(self, who):
            return QuotaSubject(tenant_id="default", sub="user-1") if who else None

        def record_blocking(self, subject, dimension, amount) -> None:
            self.records.append((subject, dimension, amount))

    quota = FakeQuota()
    store = _store()
    indexing = IndexingService(
        knowledge_service=service, job_store=store, quota_service=quota
    )
    summary = indexing.submit(collection=collection, principal=principal)
    _wait_until(lambda: store.get(summary["job_id"])["status"] == "completed")
    embedding_records = [
        rec for rec in quota.records if rec[1] == QuotaDimension.EMBEDDING_TOKENS
    ]
    assert len(embedding_records) == 2
    assert all(amount > 0 for _subject, _dim, amount in embedding_records)


def test_reindex_cancel_between_documents_stops_early() -> None:
    service, embeddings, collection = _service_with_docs("alpha", "beta", "gamma")
    calls_before = embeddings.document_calls

    class FakeHandle:
        def __init__(self) -> None:
            self.progress_calls: list[int] = []
            self.document_completed_ids: list[str] = []
            self.completed = False
            self.cancel_reason: str | None = None

        @property
        def cancelled(self) -> bool:
            # Cancel becomes visible after the first document is processed.
            return len(self.progress_calls) >= 1

        def begin(self, total: int) -> None:
            self.total = total

        def progress(self, *, completed_documents: int, current_document_title: str = "") -> None:
            self.progress_calls.append(completed_documents)

        def document_completed(self, document_id: str) -> None:
            self.document_completed_ids.append(document_id)

        def complete(self) -> None:
            self.completed = True

        def cancel(self, reason: str = "cancelled") -> None:
            self.cancel_reason = reason

    handle = FakeHandle()
    execute_reindex_job(
        handle,
        knowledge_service=service,
        collection_id=collection.id,
        embedding_model=collection.embedding_model,
    )
    assert handle.completed is False
    assert handle.cancel_reason == "client_requested_cancel"
    # Only the first document was re-embedded before the cancel took effect.
    assert embeddings.document_calls == calls_before + 1
    # And exactly that one document emitted a per-document completion event.
    assert len(handle.document_completed_ids) == 1


def test_submit_raises_when_store_lacks_reembed() -> None:
    class NoReembedStore(MemoryKnowledgeStore):
        reembed_document = None  # type: ignore[assignment]

    context = KnowledgeProviderContext(
        embeddings=StubEmbeddings(), store=NoReembedStore()
    )
    service = KnowledgeService(
        knowledge=context, chunk_max_chars=2_000, max_document_chars=100_000
    )
    collection = asyncio.run(service.create_collection(name="C"))
    indexing = IndexingService(knowledge_service=service, job_store=_store())
    with pytest.raises(ReindexUnsupported):
        indexing.submit(collection=collection)
