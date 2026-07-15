"""Postgres integration tests for the durable reindex-job store (gated).

Same gating as the run-store suite (``INQTRIX_TEST_DATABASE_URL``,
restricted app role, RLS). The parity assertions mirror the in-memory
:class:`~inqtrix.server.indexing.IndexingJobStore` contract — the
summary wire shape, gap-free 1-based event sequences, terminal-state
absorption, and claim fencing — plus the two behaviours the run store
lacks: one-active-job-per-collection serialization and the
per-collection history cap.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.runs.indexing_postgres import PostgresIndexingJobStore
from inqtrix.server.indexing import (
    IndexingJobConflict,
    IndexingJobNotFound,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"
USER_A = uuid.UUID("11111111-1111-4111-8111-111111111111")
USER_B = uuid.UUID("22222222-2222-4222-8222-222222222222")
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
    "events_url",
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
    for _ in range(3):
        summary = submit_reembed(
            store,
            collection_id=OWNED_COLLECTION_ID,
            created_by_user_id=USER_A,
            created_by_tenant_id="default",
        )
        wait_for_status(store, summary["job_id"], {"completed"})
        ids.append(summary["job_id"])

    cursor = effect_cursor(store)
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


def test_stuck_active_job_is_failed_not_deleted_and_requests_local_stop(
    store,
) -> None:
    release = threading.Event()
    cancel_observed = threading.Event()

    def slow(handle):
        handle.begin(1)
        while not handle.cancelled and not release.wait(0.01):
            pass
        if handle.cancelled:
            cancel_observed.set()
            handle.cancel("lifecycle_timeout")
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
        failed = next(row for row in rows if row["job_id"] == summary["job_id"])
        assert failed["status"] == "failed"
        assert failed["error"]["type"] == "stuck_job_timeout"
        assert store.has_active_job(OWNED_COLLECTION_ID) is False
        assert cancel_observed.wait(2)

        subscription = store.subscribe(summary["job_id"])
        try:
            assert subscription.replay[-1]["type"] == "inqtrix.index.failed"
            assert (
                subscription.replay[-1]["data"]["error"]["type"]
                == "stuck_job_timeout"
            )
        finally:
            subscription.close()
        audit_count, targets = maintenance_effects(
            store,
            action="indexing.stuck_timeout",
            resource_id=OWNED_COLLECTION_ID,
            after_event_id=cursor,
        )
        assert audit_count == 1
        assert targets == {USER_A, USER_B}
    finally:
        release.set()


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
        created_by_user_id=str(USER_A),
        created_by_tenant_id="default",
    )
    wait_for_status(store, summary["job_id"], {"completed"})

    foreign = UserContext(
        principal=Principal(
            user_id=USER_B, kind="oidc_session", tenant_id="default"
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


def test_orphan_sweep_fails_stale_rows_on_first_touch(store):
    release = threading.Event()

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
