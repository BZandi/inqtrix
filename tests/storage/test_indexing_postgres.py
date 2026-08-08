"""Postgres integration tests for the durable reindex-job store (gated).

Same gating as the run-store suite (``INQTRIX_TEST_DATABASE_URL``,
restricted app role, RLS). The parity assertions mirror the in-memory
:class:`~inqtrix.server.indexing.IndexingJobStore` contract — the summary
wire shape, gap-free 1-based event sequences, terminal-state absorption,
claim fencing, collection-generation serialization, immutable-revision
submission idempotency, and the per-collection history cap.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.contextualization_circuit import ContextualizationCircuitPermit
from inqtrix.runs.indexing_postgres import PostgresIndexingJobStore
from inqtrix.server.indexing import (
    IndexingJobConflict,
    IndexingJobNotFound,
    IndexingResumeUnavailable,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.postgres

APP_ROLE = "inqtrix_app"
USER_A = uuid.UUID("11111111-1111-4111-8111-111111111111")
USER_B = uuid.UUID("22222222-2222-4222-8222-222222222222")
USER_C = uuid.UUID("44444444-4444-4444-8444-444444444444")
COLLECTION_IDS = {
    "col-1",
    "col-2",
    "col-3",
    "col-a",
    "col-b",
    "col-c",
    "col-doc",
    "col-h",
    "col-owned",
    "col-x",
}
OWNED_COLLECTION_ID = "col-owned"

SUMMARY_KEYS = {
    "job_id",
    "collection_id",
    "collection_name",
    "embedding_model",
    "operation_kind",
    "document_id",
    "revision_id",
    "index_id",
    "status",
    "queue_position",
    "workspace_id",
    "created_at",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "total_documents",
    "completed_documents",
    "percent",
    "snapshot",
    "error",
    "phase",
    "current_batch",
    "total_batches",
    "checkpoint",
    "generation_id",
    "fence_token",
    "events_url",
    "last_event_sequence",
}


@pytest.fixture(scope="session", autouse=True)
def indexing_schema_migrated():
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture()
async def engine():
    engine = build_engine(TEST_DATABASE_URL)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def store(engine):
    factory = build_session_factory(engine)
    async with factory() as session:
        async with session.begin():
            bypasses = (
                await session.execute(
                    text(
                        "SELECT rolsuper OR rolbypassrls FROM pg_roles "
                        "WHERE rolname = current_user"
                    )
                )
            ).scalar_one()
            if not bypasses:
                pytest.fail(
                    "INQTRIX_TEST_DATABASE_URL must connect as a "
                    "superuser/BYPASSRLS user for cross-tenant cleanup."
                )
            await session.execute(text("DELETE FROM indexing_job_events"))
            await session.execute(text("DELETE FROM indexing_jobs"))
            await session.execute(
                text("DELETE FROM contextualization_provider_circuits")
            )
            for collection_id in COLLECTION_IDS:
                await session.execute(
                    text(
                        "DELETE FROM user_events WHERE "
                        "resource_type = 'knowledge_collection' "
                        "AND resource_id = :resource_id"
                    ),
                    {"resource_id": collection_id},
                )
                await session.execute(
                    text(
                        "DELETE FROM audit_log WHERE "
                        "resource_type = 'knowledge_collection' "
                        "AND resource_id = :resource_id"
                    ),
                    {"resource_id": collection_id},
                )
                await session.execute(
                    text(
                        "DELETE FROM resource_shares WHERE "
                        "resource_type = 'knowledge_collection' "
                        "AND resource_id = :resource_id"
                    ),
                    {"resource_id": collection_id},
                )
            for user_id, subject in (
                (USER_A, "indexing-owner"),
                (USER_B, "indexing-recipient"),
                (USER_C, "indexing-unshared"),
            ):
                await session.execute(
                    text(
                        "INSERT INTO users "
                        "(id, tenant_id, issuer, subject, email, "
                        "email_verified) VALUES "
                        "(:id, 'default', 'pytest-indexing', :subject, "
                        ":email, true) ON CONFLICT (id) DO UPDATE SET "
                        "disabled_at = NULL"
                    ),
                    {
                        "id": user_id,
                        "subject": subject,
                        "email": f"{subject}@example.test",
                    },
                )
            for collection_id in COLLECTION_IDS:
                owner_user_id = (
                    USER_A if collection_id == OWNED_COLLECTION_ID else None
                )
                await session.execute(
                    text(
                        "INSERT INTO knowledge_collections "
                        "(id, tenant_id, name, embedding_model, embedding_dim, "
                        "created_by_user_id, created_at) VALUES "
                        "(:id, 'default', :name, 'text-embedding-3-large', 8, "
                        ":owner_user_id, :created_at) ON CONFLICT (id) "
                        "DO UPDATE SET created_by_user_id = "
                        "EXCLUDED.created_by_user_id"
                    ),
                    {
                        "id": collection_id,
                        "name": f"Fixture {collection_id}",
                        "owner_user_id": owner_user_id,
                        "created_at": time.time(),
                    },
                )
            await session.execute(
                text(
                    "INSERT INTO resource_shares "
                    "(id, tenant_id, recipient_user_id, resource_type, "
                    "resource_id, permission, granted_by_user_id, "
                    "accepted_at) VALUES "
                    "(:id, 'default', :recipient_user_id, "
                    "'knowledge_collection', :resource_id, 'edit', "
                    ":owner_user_id, now())"
                ),
                {
                    "id": uuid.UUID("33333333-3333-4333-8333-333333333333"),
                    "recipient_user_id": USER_B,
                    "resource_id": OWNED_COLLECTION_ID,
                    "owner_user_id": USER_A,
                },
            )
    store = PostgresIndexingJobStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        history_limit=2,
        worker_id="pytest-index-worker",
    )
    yield store
    store.close()


def wait_for_status(store, job_id, statuses, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summary = store.get(job_id)
        if summary["status"] in statuses:
            return summary
        time.sleep(0.05)
    pytest.fail(f"job {job_id} never reached {statuses}")


def expire_contextualization_circuit(
    store,
    *,
    provider_key: str,
    model: str,
    expire_probe: bool = False,
) -> None:
    async def _expire() -> None:
        async with store._session("default") as session:
            values = (
                "probe_lease_until = 0, updated_at = 0"
                if expire_probe
                else "cooldown_until = 0, updated_at = 0"
            )
            await session.execute(
                text(
                    "UPDATE contextualization_provider_circuits "
                    f"SET {values} "
                    "WHERE tenant_id = 'default' "
                    "AND provider_key = :provider_key AND model = :model"
                ),
                {"provider_key": provider_key, "model": model},
            )

    store._call(_expire())


def submit_reembed(store, *, collection_id="col-a", work=None, **kwargs):
    def _default_work(handle):
        handle.begin(2)
        handle.progress(completed_documents=1, current_document_title="doc-1")
        handle.progress(completed_documents=2, current_document_title="doc-2")
        handle.complete()

    return store.submit(
        collection_id=collection_id,
        collection_name=kwargs.pop("collection_name", "Collection A"),
        embedding_model=kwargs.pop("embedding_model", "text-embedding-3-large"),
        work=work or _default_work,
        **kwargs,
    )


def effect_cursor(store: PostgresIndexingJobStore) -> int:
    """Return the last user-invalidation id visible to the store tenant."""

    async def _read() -> int:
        async with store._session("default") as session:
            value = await session.scalar(
                text("SELECT COALESCE(MAX(id), 0) FROM user_events")
            )
            return int(value or 0)

    return store._call(_read())


def maintenance_effects(
    store: PostgresIndexingJobStore,
    *,
    action: str,
    resource_id: str,
    after_event_id: int,
) -> tuple[int, set[uuid.UUID]]:
    """Read one maintenance audit count and its new invalidation targets."""

    async def _read() -> tuple[int, set[uuid.UUID]]:
        async with store._session("default") as session:
            audit_count = await session.scalar(
                text(
                    "SELECT count(*) FROM audit_log WHERE action = :action "
                    "AND resource_type = 'knowledge_collection' "
                    "AND resource_id = :resource_id"
                ),
                {"action": action, "resource_id": resource_id},
            )
            targets = (
                await session.execute(
                    text(
                        "SELECT target_user_id FROM user_events "
                        "WHERE id > :after_event_id "
                        "AND scope = 'indexing' "
                        "AND resource_type = 'knowledge_collection' "
                        "AND resource_id = :resource_id"
                    ),
                    {
                        "after_event_id": after_event_id,
                        "resource_id": resource_id,
                    },
                )
            ).scalars()
            return int(audit_count or 0), set(targets)

    return store._call(_read())


def age_job(
    store: PostgresIndexingJobStore,
    job_id: str,
    *,
    created_at: float | None = None,
    finished_at: float | None = None,
) -> None:
    """Backdate selected lifecycle timestamps for retention tests."""

    async def _update() -> None:
        values: dict[str, float] = {}
        if created_at is not None:
            values["created_at"] = created_at
        if finished_at is not None:
            values["finished_at"] = finished_at
        assignments = ", ".join(f"{column} = :{column}" for column in values)
        async with store._session("default") as session:
            await session.execute(
                text(
                    f"UPDATE indexing_jobs SET {assignments} "
                    "WHERE job_id = :job_id"
                ),
                {"job_id": job_id, **values},
            )

    store._call(_update())


def test_submit_executes_and_keeps_the_wire_shape(store):
    summary = submit_reembed(store)
    assert set(summary.keys()) == SUMMARY_KEYS
    assert summary["status"] in {"queued", "running"}

    final = wait_for_status(store, summary["job_id"], {"completed"})
    assert set(final.keys()) == SUMMARY_KEYS
    assert final["percent"] == 100
    assert final["completed_documents"] == 2
    assert final["total_documents"] == 2
    assert final["error"] is None


def test_in_process_document_job_carries_its_claim_attempt_to_publication(store):
    observed: dict[str, object] = {}

    def inspect_fence(handle):
        observed["job_id"] = handle.fence_job_id
        observed["attempt"] = handle.fence_attempt
        handle.begin(1)
        handle.progress(completed_documents=1)
        handle.complete()

    summary = submit_reembed(
        store,
        collection_id="col-doc",
        operation_kind="document_revision",
        document_id="kd_claim_fence",
        revision_id="rev_claim_fence",
        work=inspect_fence,
    )
    wait_for_status(store, summary["job_id"], {"completed"})

    assert observed == {
        "job_id": summary["job_id"],
        "attempt": 1,
    }


def test_document_delta_and_generation_coexist_but_generations_serialize(store):
    document_started = threading.Event()
    generation_started = threading.Event()
    release = threading.Event()

    def blocking(started):
        def work(handle):
            handle.begin(1)
            started.set()
            release.wait(timeout=5)
            handle.progress(completed_documents=1)

        return work

    document = submit_reembed(
        store,
        collection_id="col-doc",
        operation_kind="document_revision",
        document_id="kd_postgres_delta",
        revision_id="rev_postgres_delta",
        work=blocking(document_started),
    )
    assert document_started.wait(timeout=5)
    generation = submit_reembed(
        store,
        collection_id="col-doc",
        operation_kind="collection_generation",
        generation_id="gen_postgres_shared",
        work=blocking(generation_started),
    )
    assert generation_started.wait(timeout=5)

    with pytest.raises(IndexingJobConflict):
        submit_reembed(
            store,
            collection_id="col-doc",
            operation_kind="collection_generation",
            generation_id="gen_postgres_conflict",
        )

    release.set()
    wait_for_status(store, document["job_id"], {"completed"})
    wait_for_status(store, generation["job_id"], {"completed"})


def test_concurrent_revision_retries_return_one_durable_job(store):
    callers_ready = threading.Barrier(3)
    work_started = threading.Event()
    release = threading.Event()
    result_lock = threading.Lock()
    results: list[dict] = []
    errors: list[BaseException] = []
    work_calls = 0

    def work(handle):
        nonlocal work_calls
        work_calls += 1
        handle.begin(1)
        work_started.set()
        release.wait(timeout=5)
        handle.progress(completed_documents=1)

    def submit_retry():
        try:
            callers_ready.wait(timeout=5)
            summary = submit_reembed(
                store,
                collection_id=OWNED_COLLECTION_ID,
                operation_kind="document_revision",
                document_id="kd_pg_retry",
                revision_id="rev_pg_retry",
                created_by_user_id=USER_A,
                created_by_tenant_id="default",
                work=work,
            )
            with result_lock:
                results.append(summary)
        except BaseException as exc:  # pragma: no cover - asserted below
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=submit_retry) for _ in range(2)]
    for thread in threads:
        thread.start()
    callers_ready.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    try:
        assert errors == []
        assert len(results) == 2
        assert results[0]["job_id"] == results[1]["job_id"]
        assert work_started.wait(timeout=5)
        assert work_calls == 1
        collaborator_retry = submit_reembed(
            store,
            collection_id=OWNED_COLLECTION_ID,
            operation_kind="document_revision",
            document_id="kd_pg_retry",
            revision_id="rev_pg_retry",
            created_by_user_id=USER_B,
            created_by_tenant_id="default",
        )
        assert collaborator_retry["job_id"] == results[0]["job_id"]
    finally:
        release.set()
    wait_for_status(store, results[0]["job_id"], {"completed"})


def test_terminal_revision_failure_and_cancel_release_durable_slot(store):
    def fail(handle):
        handle.begin(1)
        raise RuntimeError("embedding failed")

    failed = submit_reembed(
        store,
        collection_id="col-doc",
        operation_kind="document_revision",
        document_id="kd_pg_terminal",
        revision_id="rev_pg_terminal",
        work=fail,
    )
    wait_for_status(store, failed["job_id"], {"failed"})
    after_failure = submit_reembed(
        store,
        collection_id="col-doc",
        operation_kind="document_revision",
        document_id="kd_pg_terminal",
        revision_id="rev_pg_terminal",
    )
    assert after_failure["job_id"] != failed["job_id"]
    wait_for_status(store, after_failure["job_id"], {"completed"})

    started = threading.Event()
    release = threading.Event()

    def block(handle):
        handle.begin(1)
        started.set()
        release.wait(timeout=5)

    cancelling = submit_reembed(
        store,
        collection_id="col-doc",
        operation_kind="document_revision",
        document_id="kd_pg_cancel",
        revision_id="rev_pg_cancel",
        work=block,
    )
    assert started.wait(timeout=5)
    store.cancel(cancelling["job_id"])
    release.set()
    wait_for_status(store, cancelling["job_id"], {"cancelled"})
    after_cancel = submit_reembed(
        store,
        collection_id="col-doc",
        operation_kind="document_revision",
        document_id="kd_pg_cancel",
        revision_id="rev_pg_cancel",
    )
    assert after_cancel["job_id"] != cancelling["job_id"]
    wait_for_status(store, after_cancel["job_id"], {"completed"})


def test_event_stream_is_gap_free_with_progress(store):
    summary = submit_reembed(store)
    wait_for_status(store, summary["job_id"], {"completed"})

    subscription = store.subscribe(summary["job_id"])
    try:
        events = subscription.replay
    finally:
        subscription.close()

    sequences = [event["sequence"] for event in events]
    assert sequences == list(range(1, len(sequences) + 1))
    types = [event["type"] for event in events]
    assert types[0] == "inqtrix.index.queued"
    assert types[-1] == "inqtrix.index.completed"
    for event in events:
        assert set(event.keys()) == {
            "type",
            "job_id",
            "sequence",
            "created_at",
            "data",
        }


def test_failed_work_lands_as_sanitized_failure(store):
    def boom(handle):
        handle.begin(1)
        raise RuntimeError("embedding backend exploded")

    summary = submit_reembed(store, work=boom)
    final = wait_for_status(store, summary["job_id"], {"failed"})
    assert final["error"] == {
        "message": "embedding backend exploded",
        "type": "server_error",
    }


def test_one_active_job_per_collection_conflicts(store):
    release = threading.Event()

    def slow(handle):
        handle.begin(1)
        release.wait(10)
        handle.complete()

    first = submit_reembed(store, collection_id="col-x", work=slow)
    try:
        wait_for_status(store, first["job_id"], {"running"})
        with pytest.raises(IndexingJobConflict):
            submit_reembed(store, collection_id="col-x", work=slow)
    finally:
        release.set()
        wait_for_status(store, first["job_id"], {"completed"})
    # Once the first is terminal a fresh job for the same collection is fine.
    second = submit_reembed(store, collection_id="col-x")
    wait_for_status(store, second["job_id"], {"completed"})


def test_per_collection_history_cap_evicts_oldest(store):
    """history_limit=2: only the two newest terminal jobs per collection
    survive the lazy cleanup."""
    ids = []
    for _ in range(2):
        summary = submit_reembed(
            store,
            collection_id=OWNED_COLLECTION_ID,
            created_by_user_id=USER_A,
            created_by_tenant_id="default",
        )
        wait_for_status(store, summary["job_id"], {"completed"})
        ids.append(summary["job_id"])

    cursor = effect_cursor(store)
    summary = submit_reembed(
        store,
        collection_id=OWNED_COLLECTION_ID,
        created_by_user_id=USER_A,
        created_by_tenant_id="default",
    )
    wait_for_status(store, summary["job_id"], {"completed"})
    ids.append(summary["job_id"])
    surviving = {
        row["job_id"]
        for row in store.list(collection_id=OWNED_COLLECTION_ID)
    }
    assert ids[0] not in surviving
    assert {ids[1], ids[2]} <= surviving
    audit_count, targets = maintenance_effects(
        store,
        action="indexing.history_evicted",
        resource_id=OWNED_COLLECTION_ID,
        after_event_id=cursor,
    )
    assert audit_count == 1
    assert targets == {USER_A, USER_B}


def test_terminal_ttl_deletion_is_audited_and_invalidates_shared_views(
    store,
) -> None:
    summary = submit_reembed(
        store,
        collection_id=OWNED_COLLECTION_ID,
        created_by_user_id=USER_A,
        created_by_tenant_id="default",
    )
    wait_for_status(store, summary["job_id"], {"completed"})
    age_job(
        store,
        summary["job_id"],
        finished_at=time.time() - 301,
    )
    cursor = effect_cursor(store)

    assert store.list(collection_id=OWNED_COLLECTION_ID) == []
    with pytest.raises(IndexingJobNotFound):
        store.get(summary["job_id"])
    audit_count, targets = maintenance_effects(
        store,
        action="indexing.retention_deleted",
        resource_id=OWNED_COLLECTION_ID,
        after_event_id=cursor,
    )
    assert audit_count == 1
    assert targets == {USER_A, USER_B}


def test_running_job_age_is_not_an_implicit_lifecycle_deadline(store) -> None:
    release = threading.Event()
    cancel_observed = threading.Event()

    def slow(handle):
        handle.begin(1)
        while not handle.cancelled and not release.wait(0.01):
            pass
        if handle.cancelled:
            cancel_observed.set()
            handle.cancel("client_requested_cancel")
        else:
            handle.complete()

    summary = submit_reembed(
        store,
        collection_id=OWNED_COLLECTION_ID,
        work=slow,
        created_by_user_id=USER_A,
        created_by_tenant_id="default",
    )
    try:
        wait_for_status(store, summary["job_id"], {"running"})
        age_job(
            store,
            summary["job_id"],
            created_at=time.time() - (8 * 86_400),
        )
        cursor = effect_cursor(store)

        rows = store.list(collection_id=OWNED_COLLECTION_ID)
        current = next(row for row in rows if row["job_id"] == summary["job_id"])
        assert current["status"] == "running"
        assert current["error"] is None
        assert store.has_active_job(OWNED_COLLECTION_ID) is True
        assert cancel_observed.is_set() is False

        subscription = store.subscribe(summary["job_id"])
        try:
            assert all(
                event["type"] != "inqtrix.index.failed"
                for event in subscription.replay
            )
        finally:
            subscription.close()
        audit_count, targets = maintenance_effects(
            store,
            action="indexing.stuck_timeout",
            resource_id=OWNED_COLLECTION_ID,
            after_event_id=cursor,
        )
        assert audit_count == 0
        assert targets == set()
    finally:
        release.set()

    wait_for_status(store, summary["job_id"], {"completed"})


def test_paused_job_survives_age_cleanup_and_resumes_from_checkpoint(
    store,
) -> None:
    executions = 0

    def pause_once(handle):
        nonlocal executions
        executions += 1
        handle.begin(1)
        if executions == 1:
            handle.checkpoint_context_batch(
                "kd_pause_retention",
                {
                    "batch_index": 0,
                    "batch_size": 1,
                    "contexts": ["retained"],
                    "document_id": "kd_pause_retention",
                    "prompt_hash": "prompt-retention",
                    "total_batches": 2,
                },
            )
            handle.pause_dependency("provider unavailable")
            return
        handle.progress(completed_documents=1)
        handle.complete()

    summary = submit_reembed(
        store,
        collection_id=OWNED_COLLECTION_ID,
        work=pause_once,
        created_by_user_id=USER_A,
        created_by_tenant_id="default",
    )
    paused = wait_for_status(store, summary["job_id"], {"paused_dependency"})
    retained_checkpoint = paused["checkpoint"]
    age_job(
        store,
        summary["job_id"],
        created_at=time.time() - (365 * 86_400),
    )

    after_cleanup = store.get(summary["job_id"])
    assert after_cleanup["status"] == "paused_dependency"
    assert after_cleanup["checkpoint"] == retained_checkpoint
    assert store.has_active_job(OWNED_COLLECTION_ID) is True

    resumed = store.resume(summary["job_id"])
    assert resumed["status"] in {"queued", "running"}
    completed = wait_for_status(store, summary["job_id"], {"completed"})
    assert completed["checkpoint"] == retained_checkpoint
    assert executions == 2


def test_parallel_document_checkpoints_do_not_overwrite_each_other(store) -> None:
    observed: dict[str, list[dict]] = {}

    def work(handle):
        handle.begin(2)
        barrier = threading.Barrier(3)

        def checkpoint(document_id: str, context: str) -> None:
            barrier.wait()
            handle.checkpoint_context_batch(
                document_id,
                {
                    "batch_number": 1,
                    "batch_size": 1,
                    "contexts": [context],
                    "document_id": document_id,
                    "prompt_hash": f"prompt-{document_id}",
                    "total_batches": 2,
                },
            )

        workers = [
            threading.Thread(target=checkpoint, args=("kd_pg_parallel_a", "a")),
            threading.Thread(target=checkpoint, args=("kd_pg_parallel_b", "b")),
        ]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join()

        observed["a_before"] = handle.context_batch_checkpoints("kd_pg_parallel_a")
        observed["b_before"] = handle.context_batch_checkpoints("kd_pg_parallel_b")
        handle.document_progress(
            "kd_pg_parallel_a",
            "contextualization",
            current_batch=1,
            total_batches=2,
        )
        handle.document_progress(
            "kd_pg_parallel_b",
            "contextualization",
            current_batch=1,
            total_batches=2,
        )
        handle.checkpoint_document("kd_pg_parallel_a")
        observed["a_after"] = handle.context_batch_checkpoints("kd_pg_parallel_a")
        observed["b_after"] = handle.context_batch_checkpoints("kd_pg_parallel_b")
        handle.pause_dependency("retain unfinished document")

    summary = submit_reembed(
        store,
        collection_id=OWNED_COLLECTION_ID,
        work=work,
        created_by_user_id=USER_A,
        created_by_tenant_id="default",
    )
    paused = wait_for_status(store, summary["job_id"], {"paused_dependency"})

    assert len(observed["a_before"]) == 1
    assert len(observed["b_before"]) == 1
    assert observed["a_after"] == []
    assert observed["b_after"] == observed["b_before"]
    assert paused["checkpoint"]["contextualization"] == {
        "active_documents": 1,
        "completed_batches": 1,
        "document_id": "kd_pg_parallel_b",
        "total_batches": 2,
    }
    assert paused["checkpoint"]["document_progress"] == {
        "kd_pg_parallel_b": {
            "current_batch": 1,
            "phase": "contextualization",
            "total_batches": 2,
        }
    }


def test_cancel_of_queued_job_is_immediate(store):
    release = threading.Event()

    def slow(handle):
        handle.begin(1)
        release.wait(10)
        handle.complete()

    first = submit_reembed(store, collection_id="col-1", work=slow)
    second = submit_reembed(store, collection_id="col-2", work=slow)
    queued = submit_reembed(store, collection_id="col-3", work=slow)
    try:
        cancelled = store.cancel(queued["job_id"])
        assert cancelled["status"] == "cancelled"
        events = store.subscribe(queued["job_id"])
        try:
            assert events.replay[-1]["type"] == "inqtrix.index.cancelled"
            assert (
                events.replay[-1]["data"]["reason"]
                == "cancelled_before_start"
            )
        finally:
            events.close()
    finally:
        release.set()
        wait_for_status(store, first["job_id"], {"completed"})
        wait_for_status(store, second["job_id"], {"completed"})


def test_cancel_running_keeps_collection_reserved_until_worker_exits(store):
    release = threading.Event()

    def slow(handle):
        handle.begin(1)
        release.wait(10)
        handle.complete()

    first = submit_reembed(store, collection_id="col-x", work=slow)
    try:
        wait_for_status(store, first["job_id"], {"running"})
        cancelling = store.cancel(first["job_id"])
        assert cancelling["status"] == "cancelling"
        assert store.has_active_job("col-x") is True
        with pytest.raises(IndexingJobConflict):
            submit_reembed(store, collection_id="col-x")
    finally:
        release.set()

    wait_for_status(store, first["job_id"], {"cancelled"})
    assert store.has_active_job("col-x") is False
    second = submit_reembed(store, collection_id="col-x")
    wait_for_status(store, second["job_id"], {"completed"})


def test_scoped_visibility_denies_with_404_semantics(store):
    summary = submit_reembed(
        store,
        collection_id=OWNED_COLLECTION_ID,
        created_by_user_id=USER_A,
        created_by_tenant_id="default",
    )
    wait_for_status(store, summary["job_id"], {"completed"})

    foreign = UserContext(
        principal=Principal(
            user_id=USER_C, kind="oidc_session", tenant_id="default"
        ),
        workspace_ids=(),
    )
    with pytest.raises(IndexingJobNotFound):
        store.get(summary["job_id"], visible_to=foreign)
    assert store.list(visible_to=foreign) == []

    owner = UserContext(
        principal=Principal(
            user_id=USER_A, kind="oidc_session", tenant_id="default"
        ),
        workspace_ids=(),
    )
    assert store.get(summary["job_id"], visible_to=owner)["status"] == (
        "completed"
    )


def test_claim_fencing_discards_zombie_terminal_writes(store):
    release = threading.Event()

    def slow(handle):
        handle.begin(1)
        release.wait(10)
        handle.complete()

    try:
        # Fill both slots so the third job STAYS queued and the
        # worker-claim path can be exercised directly.
        submit_reembed(store, collection_id="col-a", work=slow)
        submit_reembed(store, collection_id="col-b", work=slow)
        third = submit_reembed(store, collection_id="col-c", work=slow)
        job_id = third["job_id"]
        assert store.get(job_id)["status"] == "queued"

        first = store.claim_for_execution(
            job_id, "default", allow_takeover=False
        )
        assert first is not None and first.attempt == 1
        # A fresh duplicate must NOT steal a healthy job...
        assert (
            store.claim_for_execution(
                job_id, "default", allow_takeover=False
            )
            is None
        )
        # ...but a reclaim (owner stopped heartbeating) takes over.
        second = store.claim_for_execution(
            job_id, "default", allow_takeover=True
        )
        assert second is not None and second.attempt == 2

        # The zombie's write (old fence) is a discarded no-op...
        assert store.complete(job_id, fence_attempt=1) is False
        assert store.get(job_id)["status"] == "running"
        # ...while the current attempt's write lands.
        assert store.complete(job_id, fence_attempt=2) is True
        assert store.get(job_id)["status"] == "completed"
    finally:
        release.set()


def test_document_completed_event_round_trips(store):
    """The per-document event's JSON payload survives Postgres unchanged
    (no schema column — it lives in the generic event ``data``)."""
    def work(handle):
        handle.begin(1)
        handle.document_started("kd_round")
        handle.document_progress(
            "kd_round",
            "contextualization",
            current_batch=2,
            total_batches=5,
        )
        handle.document_completed("kd_round")
        handle.complete()

    summary = submit_reembed(store, collection_id="col-doc", work=work)
    wait_for_status(store, summary["job_id"], {"completed"})

    subscription = store.subscribe(summary["job_id"])
    try:
        events = subscription.replay
    finally:
        subscription.close()
    doc_events = [
        event for event in events
        if event["type"] == "inqtrix.index.document_completed"
    ]
    started_events = [
        event for event in events
        if event["type"] == "inqtrix.index.document_started"
    ]
    progress_events = [
        event for event in events
        if event["type"] == "inqtrix.index.document_progress"
    ]
    assert len(started_events) == 1
    assert started_events[0]["data"] == {"document_id": "kd_round"}
    assert len(progress_events) == 1
    assert progress_events[0]["data"]["document_id"] == "kd_round"
    assert progress_events[0]["data"]["phase"] == "contextualization"
    assert progress_events[0]["data"]["current_batch"] == 2
    assert progress_events[0]["data"]["total_batches"] == 5
    assert len(doc_events) == 1
    assert doc_events[0]["data"] == {"document_id": "kd_round", "outcome": "embedded"}


def test_document_completed_respects_the_claim_fence(store):
    """A reclaimed zombie's per-document event (old fence) is dropped, the
    current attempt's lands — same fence as progress/terminal writes."""
    release = threading.Event()

    def slow(handle):
        handle.begin(1)
        release.wait(10)
        handle.complete()

    def doc_event_count(job_id):
        subscription = store.subscribe(job_id)
        try:
            return sum(
                1 for event in subscription.replay
                if event["type"] == "inqtrix.index.document_completed"
            )
        finally:
            subscription.close()

    try:
        submit_reembed(store, collection_id="col-a", work=slow)
        submit_reembed(store, collection_id="col-b", work=slow)
        third = submit_reembed(store, collection_id="col-c", work=slow)
        job_id = third["job_id"]
        assert store.get(job_id)["status"] == "queued"

        assert store.claim_for_execution(job_id, "default", allow_takeover=False).attempt == 1
        assert store.claim_for_execution(job_id, "default", allow_takeover=True).attempt == 2

        # The zombie's event (old fence) is a discarded no-op...
        store.document_completed(job_id, "kd_zombie", fence_attempt=1)
        assert doc_event_count(job_id) == 0
        # ...while the current attempt's event lands.
        store.document_completed(job_id, "kd_live", fence_attempt=2)
        assert doc_event_count(job_id) == 1
    finally:
        release.set()


@pytest.mark.parametrize("pause_kind", ["dependency", "validation"])
def test_cancel_wins_concurrent_pause_cas(store, pause_kind: str) -> None:
    started = threading.Event()
    release = threading.Event()

    def pause_after_cancel(handle):
        handle.begin(1)
        started.set()
        release.wait(10)
        if pause_kind == "dependency":
            handle.pause_dependency(
                "provider unavailable",
                error_type="contextualization_provider_unavailable",
            )
        else:
            handle.pause_validation("provider response invalid")

    summary = submit_reembed(
        store,
        collection_id=OWNED_COLLECTION_ID,
        work=pause_after_cancel,
        created_by_user_id=USER_A,
        created_by_tenant_id="default",
    )
    assert started.wait(timeout=2)
    cancelling = store.cancel(summary["job_id"])
    assert cancelling["status"] == "cancelling"
    release.set()

    cancelled = wait_for_status(store, summary["job_id"], {"cancelled"})
    assert cancelled["error"] is None
    subscription = store.subscribe(summary["job_id"])
    try:
        event_types = [event["type"] for event in subscription.replay]
    finally:
        subscription.close()
    assert "inqtrix.index.cancelled" in event_types
    assert "inqtrix.index.paused_dependency" not in event_types
    assert "inqtrix.index.paused_validation" not in event_types


def test_provider_model_circuit_grants_one_half_open_probe_across_stores(
    store,
) -> None:
    provider_key = "azure"
    model = "fast-deployment"
    initial = store.acquire_contextualization_circuit(
        provider_key=provider_key,
        model=model,
        cooldown_seconds=60,
        probe_lease_seconds=120,
    )
    assert initial is not None
    store.record_contextualization_circuit_failure(
        initial,
        error_type="contextualization_provider_timeout",
    )
    assert (
        store.acquire_contextualization_circuit(
            provider_key=provider_key,
            model=model,
            cooldown_seconds=60,
            probe_lease_seconds=120,
        )
        is None
    )
    expire_contextualization_circuit(
        store,
        provider_key=provider_key,
        model=model,
    )

    peer = PostgresIndexingJobStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        recover_orphans=False,
        max_concurrent=1,
        max_queue_size=10,
        completed_ttl_seconds=300,
        history_limit=2,
        worker_id="pytest-index-circuit-peer",
    )
    barrier = threading.Barrier(2)

    def acquire(authority) -> ContextualizationCircuitPermit | None:
        barrier.wait()
        return authority.acquire_contextualization_circuit(
            provider_key=provider_key,
            model=model,
            cooldown_seconds=60,
            probe_lease_seconds=120,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(acquire, store)
            second = executor.submit(acquire, peer)
            permits = [first.result(timeout=5), second.result(timeout=5)]
        granted = [permit for permit in permits if permit is not None]
        assert len(granted) == 1
        assert granted[0].probe_token is not None
    finally:
        peer.close()


def test_provider_circuit_reclaims_expired_probe_and_fences_stale_token(
    store,
) -> None:
    provider_key = "anthropic"
    model = "haiku"
    initial = store.acquire_contextualization_circuit(
        provider_key=provider_key,
        model=model,
        cooldown_seconds=60,
        probe_lease_seconds=120,
    )
    assert initial is not None
    store.record_contextualization_circuit_failure(
        initial,
        error_type="contextualization_provider_unavailable",
    )
    expire_contextualization_circuit(
        store,
        provider_key=provider_key,
        model=model,
    )
    crashed = store.acquire_contextualization_circuit(
        provider_key=provider_key,
        model=model,
        cooldown_seconds=60,
        probe_lease_seconds=120,
    )
    assert crashed is not None and crashed.probe_token is not None
    wrong = replace(crashed, probe_token="not-the-current-probe")
    store.record_contextualization_circuit_success(wrong)
    assert (
        store.acquire_contextualization_circuit(
            provider_key=provider_key,
            model=model,
            cooldown_seconds=60,
            probe_lease_seconds=120,
        )
        is None
    )

    expire_contextualization_circuit(
        store,
        provider_key=provider_key,
        model=model,
        expire_probe=True,
    )
    replacement = store.acquire_contextualization_circuit(
        provider_key=provider_key,
        model=model,
        cooldown_seconds=60,
        probe_lease_seconds=120,
    )
    assert replacement is not None
    assert replacement.probe_token != crashed.probe_token
    store.record_contextualization_circuit_failure(
        crashed,
        error_type="contextualization_provider_timeout",
    )
    assert (
        store.acquire_contextualization_circuit(
            provider_key=provider_key,
            model=model,
            cooldown_seconds=60,
            probe_lease_seconds=120,
        )
        is None
    )
    store.record_contextualization_circuit_success(replacement)
    reopened = store.acquire_contextualization_circuit(
        provider_key=provider_key,
        model=model,
        cooldown_seconds=60,
        probe_lease_seconds=120,
    )
    assert reopened is not None
    assert reopened.probe_token is None


def test_restart_recovery_fails_lost_execution_but_preserves_paused_work(store):
    release = threading.Event()

    def pause_for_dependency(handle):
        handle.begin(1)
        handle.checkpoint_context_batch(
            "kd_restart_pause",
            {
                "batch_index": 0,
                "batch_size": 1,
                "contexts": ["retained"],
                "document_id": "kd_restart_pause",
                "prompt_hash": "prompt-restart",
                "total_batches": 2,
            },
        )
        handle.pause_dependency("provider unavailable")

    paused = submit_reembed(
        store,
        collection_id="col-h",
        work=pause_for_dependency,
    )
    paused_before = wait_for_status(
        store, paused["job_id"], {"paused_dependency"}
    )

    def slow(handle):
        handle.begin(1)
        release.wait(10)
        handle.complete()

    try:
        running = submit_reembed(
            store,
            collection_id=OWNED_COLLECTION_ID,
            work=slow,
            created_by_user_id=USER_A,
            created_by_tenant_id="default",
        )
        second = submit_reembed(store, collection_id="col-b", work=slow)
        third = submit_reembed(store, collection_id="col-c", work=slow)
        assert store.get(third["job_id"])["status"] == "queued"
        cursor = effect_cursor(store)

        restarted = PostgresIndexingJobStore(
            engine=build_engine(TEST_DATABASE_URL),
            app_role=APP_ROLE,
            queue=None,
            max_concurrent=2,
            max_queue_size=10,
            completed_ttl_seconds=300,
            history_limit=2,
            worker_id="pytest-index-orphan",
        )
        try:
            swept = restarted.get(third["job_id"])
            assert swept["status"] == "failed"
            assert swept["error"]["type"] == "server_restarted"
            assert restarted.get(running["job_id"])["status"] == "failed"
            assert restarted.get(second["job_id"])["status"] == "failed"
            paused_after = restarted.get(paused["job_id"])
            assert paused_after["status"] == "paused_dependency"
            assert paused_after["checkpoint"] == paused_before["checkpoint"]
            with pytest.raises(IndexingResumeUnavailable):
                restarted.resume(paused["job_id"])
            assert restarted.get(paused["job_id"])["status"] == (
                "paused_dependency"
            )

            resumed_calls = 0

            def rebound_work(handle):
                nonlocal resumed_calls
                resumed_calls += 1
                handle.begin(1)
                handle.progress(completed_documents=1)
                handle.complete()

            resumed = restarted.resume(
                paused["job_id"],
                work=rebound_work,
            )
            assert resumed["status"] in {"queued", "running"}
            wait_for_status(restarted, paused["job_id"], {"completed"})
            assert resumed_calls == 1
            audit_count, targets = maintenance_effects(
                restarted,
                action="indexing.server_restarted",
                resource_id=OWNED_COLLECTION_ID,
                after_event_id=cursor,
            )
            assert audit_count == 1
            assert targets == {USER_A, USER_B}
        finally:
            restarted.close()
    finally:
        release.set()
