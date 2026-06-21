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
        summary = submit_reembed(store, collection_id="col-h")
        wait_for_status(store, summary["job_id"], {"completed"})
        ids.append(summary["job_id"])

    surviving = {row["job_id"] for row in store.list(collection_id="col-h")}
    assert ids[0] not in surviving
    assert {ids[1], ids[2]} <= surviving


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


def test_scoped_visibility_denies_with_404_semantics(store):
    summary = submit_reembed(
        store, created_by_sub="user-a", created_by_tenant_id="default"
    )
    wait_for_status(store, summary["job_id"], {"completed"})

    foreign = UserContext(
        principal=Principal(
            sub="user-b", kind="oidc_session", tenant_id="default"
        ),
        workspace_ids=(),
        groups=(),
    )
    with pytest.raises(IndexingJobNotFound):
        store.get(summary["job_id"], visible_to=foreign)
    assert store.list(visible_to=foreign) == []

    owner = UserContext(
        principal=Principal(
            sub="user-a", kind="oidc_session", tenant_id="default"
        ),
        workspace_ids=(),
        groups=(),
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


def test_orphan_sweep_fails_stale_rows_on_first_touch(store):
    release = threading.Event()

    def slow(handle):
        handle.begin(1)
        release.wait(10)
        handle.complete()

    try:
        running = submit_reembed(store, collection_id="col-a", work=slow)
        second = submit_reembed(store, collection_id="col-b", work=slow)
        third = submit_reembed(store, collection_id="col-c", work=slow)
        assert store.get(third["job_id"])["status"] == "queued"

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
        finally:
            restarted.close()
    finally:
        release.set()
