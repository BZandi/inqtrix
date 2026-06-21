"""Postgres integration tests for the durable run store (gated suite).

Same gating and conventions as the identity/content suites: a
disposable database via ``INQTRIX_TEST_DATABASE_URL``, operations under
the restricted app role, RLS as the second defense layer. The parity
assertions mirror the in-memory contract: the 16-key summary shape,
gap-free 1-based event sequences, the snapshot companion BEFORE its
carrier, terminal-state absorption, and claim fencing.
"""

from __future__ import annotations

import os
import time

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.runs.postgres_store import PostgresRunStore
from inqtrix.server.runs import RunNotFound
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"

RUN_SUMMARY_KEYS = {
    "run_id",
    "status",
    "queue_position",
    "question",
    "stack",
    "workspace_id",
    "mode",
    "agent_overrides",
    "created_at",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "snapshot",
    "error",
    "events_url",
    "result_url",
}


@pytest.fixture(scope="session", autouse=True)
def runs_schema_migrated():
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
            await session.execute(text("DELETE FROM run_events"))
            await session.execute(text("DELETE FROM runs"))
    store = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-worker",
    )
    yield store
    store.close()


def wait_for_status(store, run_id, statuses, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summary = store.get(run_id)
        if summary["status"] in statuses:
            return summary
        time.sleep(0.05)
    pytest.fail(f"run {run_id} never reached {statuses}")


def submit_noop(store, *, work=None, **kwargs):
    def _default_work(handle):
        handle.emit(
            "inqtrix.node.started",
            {"node": "classify", "snapshot": {"current_node": "classify"}},
        )
        handle.complete(
            {"answer": "fertig", "metrics": {"rounds": 1}},
            snapshot={"current_node": "answer", "done": True},
        )

    return store.submit(
        question=kwargs.pop("question", "Wie ist die Haftung geregelt?"),
        stack_name="default",
        work=work or _default_work,
        request_payload={"question": "x", "body": {"mode": "research"}},
        **kwargs,
    )


def test_submit_executes_and_keeps_the_wire_shape(store):
    summary = submit_noop(store)
    assert set(summary.keys()) == RUN_SUMMARY_KEYS
    assert summary["status"] in {"queued", "running"}

    final = wait_for_status(store, summary["run_id"], {"completed"})
    assert set(final.keys()) == RUN_SUMMARY_KEYS
    assert final["snapshot"]["done"] is True
    assert final["error"] is None
    assert final["queue_position"] is None

    result = store.result(summary["run_id"])
    assert result["run_id"] == summary["run_id"]
    assert result["status"] == "completed"
    assert result["answer"] == "fertig"


def test_import_completed_run_is_idempotent_and_owner_scoped(store):
    summary = store.import_completed_run(
        run_id="run_imported_pg",
        question="imported report",
        stack_name="default",
        result={"answer": "the report body", "metrics": {"rounds": 1}},
        created_at=1000.0,
        workspace_id="ws_owner",
        created_by_sub="owner-1",
        created_by_tenant_id="default",
    )
    assert set(summary.keys()) == RUN_SUMMARY_KEYS
    assert summary["run_id"] == "run_imported_pg"
    assert summary["status"] == "completed"
    assert store.result("run_imported_pg")["answer"] == "the report body"

    # Idempotent re-import for the same owner: same row, body untouched.
    again = store.import_completed_run(
        run_id="run_imported_pg",
        question="imported report",
        stack_name="default",
        result={"answer": "ignored on re-import"},
        created_by_sub="owner-1",
        created_by_tenant_id="default",
    )
    assert again["run_id"] == "run_imported_pg"
    assert store.result("run_imported_pg")["answer"] == "the report body"

    # A foreign principal importing the SAME id gets a fresh id; A stays intact.
    foreign = store.import_completed_run(
        run_id="run_imported_pg",
        question="B",
        stack_name="default",
        result={"answer": "owner B body"},
        created_by_sub="owner-2",
        created_by_tenant_id="default",
    )
    assert foreign["run_id"] != "run_imported_pg"
    assert store.result("run_imported_pg")["answer"] == "the report body"
    assert store.result(foreign["run_id"])["answer"] == "owner B body"


def test_delete_removes_terminal_run_owner_scoped(store):
    store.import_completed_run(
        run_id="run_delete_pg",
        question="report",
        stack_name="default",
        result={"answer": "the body", "metrics": {"rounds": 1}},
        workspace_id="ws_owner",
        created_by_sub="owner-1",
        created_by_tenant_id="default",
    )

    # Owner-only, namespace-scoped: a foreign sub and the wrong workspace are
    # both the indistinct 404, and the row stays.
    with pytest.raises(RunNotFound):
        store.delete(
            "run_delete_pg", workspace_id="ws_owner", requester_sub="intruder"
        )
    with pytest.raises(RunNotFound):
        store.delete(
            "run_delete_pg", workspace_id="ws_other", requester_sub="owner-1"
        )
    assert (
        store.get("run_delete_pg", workspace_id="ws_owner")["run_id"]
        == "run_delete_pg"
    )

    # The owner deletes it durably; the row (and its cascaded events) are gone.
    store.delete(
        "run_delete_pg", workspace_id="ws_owner", requester_sub="owner-1"
    )
    with pytest.raises(RunNotFound):
        store.get("run_delete_pg", workspace_id="ws_owner")


def test_event_stream_is_gap_free_with_companions_first(store):
    summary = submit_noop(store)
    wait_for_status(store, summary["run_id"], {"completed"})

    subscription = store.subscribe(summary["run_id"])
    try:
        events = subscription.replay
    finally:
        subscription.close()

    sequences = [event["sequence"] for event in events]
    assert sequences == list(range(1, len(sequences) + 1))
    types = [event["type"] for event in events]
    assert types[0] == "inqtrix.run.queued"
    # Snapshot companions precede their carrier events.
    for index, event_type in enumerate(types):
        if event_type in {"inqtrix.run.started", "inqtrix.run.completed"}:
            assert types[index - 1] == "inqtrix.run.snapshot"
    assert types[-1] == "inqtrix.run.completed"
    for event in events:
        assert set(event.keys()) == {
            "type",
            "run_id",
            "sequence",
            "created_at",
            "data",
        }


def test_failed_work_lands_as_sanitized_failure(store):
    def boom(handle):
        raise RuntimeError("provider exploded")

    summary = submit_noop(store, work=boom)
    final = wait_for_status(store, summary["run_id"], {"failed"})
    assert final["error"] == {
        "message": "provider exploded",
        "type": "server_error",
    }
    with pytest.raises(RunNotFound):
        store.result(summary["run_id"])


def test_cancel_of_queued_run_is_immediate(store):
    import threading

    release = threading.Event()

    def slow(handle):
        release.wait(10)
        handle.complete({"answer": "spaet"})

    first = submit_noop(store, work=slow)
    second = submit_noop(store, work=slow)
    queued = submit_noop(store, work=slow)
    try:
        cancelled = store.cancel(queued["run_id"])
        assert cancelled["status"] == "cancelled"
        events = store.subscribe(queued["run_id"])
        try:
            assert events.replay[-1]["type"] == "inqtrix.run.cancelled"
            assert (
                events.replay[-1]["data"]["reason"]
                == "cancelled_before_start"
            )
        finally:
            events.close()
    finally:
        release.set()
        wait_for_status(store, first["run_id"], {"completed"})
        wait_for_status(store, second["run_id"], {"completed"})


def test_scoped_visibility_denies_with_404_semantics(store):
    summary = submit_noop(
        store, created_by_sub="user-a", created_by_tenant_id="default"
    )
    wait_for_status(store, summary["run_id"], {"completed"})

    foreign = UserContext(
        principal=Principal(
            sub="user-b", kind="oidc_session", tenant_id="default"
        ),
        workspace_ids=(),
        groups=(),
    )
    with pytest.raises(RunNotFound):
        store.get(summary["run_id"], visible_to=foreign)
    assert store.list(visible_to=foreign) == []

    owner = UserContext(
        principal=Principal(
            sub="user-a", kind="oidc_session", tenant_id="default"
        ),
        workspace_ids=(),
        groups=(),
    )
    assert store.get(summary["run_id"], visible_to=owner)["status"] == (
        "completed"
    )


def test_claim_fencing_discards_zombie_terminal_writes(store):
    import threading

    release = threading.Event()

    def slow(handle):
        release.wait(10)
        handle.complete({"answer": "spaet"})

    try:
        # Fill both execution slots so the third run STAYS queued and
        # the worker-claim path can be exercised directly.
        submit_noop(store, work=slow)
        submit_noop(store, work=slow)
        third = submit_noop(store, work=slow)
        run_id = third["run_id"]
        assert store.get(run_id)["status"] == "queued"

        first = store.claim_for_execution(
            run_id, "default", allow_takeover=False
        )
        assert first is not None and first.attempt == 1
        # A fresh duplicate message must NOT steal a healthy run...
        assert (
            store.claim_for_execution(
                run_id, "default", allow_takeover=False
            )
            is None
        )
        # ...but a reclaim (owner stopped heartbeating) takes over.
        second = store.claim_for_execution(
            run_id, "default", allow_takeover=True
        )
        assert second is not None and second.attempt == 2

        # The zombie's write (old fence) is a discarded no-op...
        store.complete(run_id, {"answer": "zombie"}, fence_attempt=1)
        assert store.get(run_id)["status"] == "running"
        # ...while the current attempt's write lands.
        store.complete(run_id, {"answer": "echt"}, fence_attempt=2)
        assert store.get(run_id)["status"] == "completed"
        assert store.result(run_id)["answer"] == "echt"
    finally:
        release.set()


def test_orphan_sweep_fails_stale_rows_on_first_touch(store):
    """Queued/running rows from a dead process must not stay
    eternally running: a fresh no-queue store sweeps them visibly."""
    import threading

    release = threading.Event()

    def slow(handle):
        release.wait(10)
        handle.complete({"answer": "spaet"})

    try:
        running = submit_noop(store, work=slow)
        second = submit_noop(store, work=slow)
        third = submit_noop(store, work=slow)
        assert store.get(third["run_id"])["status"] == "queued"

        # Simulated restart: a second no-queue store over the same DB.
        restarted = PostgresRunStore(
            engine=build_engine(TEST_DATABASE_URL),
            app_role=APP_ROLE,
            queue=None,
            max_concurrent=2,
            max_queue_size=10,
            completed_ttl_seconds=300,
            worker_id="pytest-orphan",
        )
        try:
            swept = restarted.get(third["run_id"])
            assert swept["status"] == "failed"
            assert swept["error"]["type"] == "server_restarted"
            events = restarted.subscribe(third["run_id"])
            try:
                assert events.replay[-1]["type"] == "inqtrix.run.failed"
            finally:
                events.close()
            assert restarted.get(running["run_id"])["status"] == "failed"
            assert restarted.get(second["run_id"])["status"] == "failed"
        finally:
            restarted.close()
    finally:
        release.set()


def test_stuck_row_cap_spares_terminal_rows(store):
    """The retention failsafe targets NON-terminal rows only —
    terminal rows are governed by the regular TTL."""
    import asyncio as aio

    from sqlalchemy import update as sa_update

    from inqtrix.storage.runs_orm import runs as runs_table

    summary = submit_noop(store)
    wait_for_status(store, summary["run_id"], {"completed"})

    async def backdate():
        async with store._session("default") as session:
            await session.execute(
                sa_update(runs_table)
                .where(runs_table.c.run_id == summary["run_id"])
                .values(created_at=time.time() - 30 * 86_400)
            )

    aio.run_coroutine_threadsafe(backdate(), store._loop).result()
    # Any read runs the lazy cleanup; the terminal row must survive
    # the stuck-row cap (its finished_at is fresh, the TTL governs).
    assert store.get(summary["run_id"])["status"] == "completed"


def test_enqueue_failure_keeps_the_accepted_run(engine):
    """A broker blip after the row commit must not turn an accepted
    run into a 500 — the reconciler re-dispatches it later."""

    class BrokenQueue:
        def enqueue(self, *, run_id, tenant_id):
            raise ConnectionError("broker weg")

    broken = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=BrokenQueue(),
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-enqueue",
    )
    try:
        summary = broken.submit(
            question="Broker-Test",
            stack_name="default",
            work=lambda handle: None,
            request_payload={"question": "x", "body": {"mode": "research"}},
        )
        assert summary["status"] == "queued"
        assert broken.get(summary["run_id"])["status"] == "queued"
        assert broken.stale_queued_runs(older_than_seconds=0) != []
    finally:
        broken.close()


def test_ttl_cleanup_removes_old_terminal_runs(store):
    store = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=1,
        worker_id="pytest-ttl",
    )
    try:
        summary = submit_noop(store)
        # The run completes, ages past the 1s retention, and the lazy
        # cleanup on a later read evicts it — eviction shows up as the
        # canonical RunNotFound, indistinguishable from never-existed.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                store.get(summary["run_id"])
            except RunNotFound:
                break
            time.sleep(0.2)
        else:
            pytest.fail("terminal run was never TTL-evicted")
        assert all(
            item["run_id"] != summary["run_id"] for item in store.list()
        )
    finally:
        store.close()


def _scoped(sub: str) -> UserContext:
    return UserContext(
        principal=Principal(
            sub=sub, kind="oidc_session", tenant_id="default"
        ),
        workspace_ids=(),
        groups=(),
    )


def test_shared_in_grants_admit_reads_and_gate_cancel(store):
    """WP-C-C parity: the also_visible contract on the durable store.

    Mirrors tests/test_runs_sharing.py — shared-in rows bypass the
    workspace filter, carry the additive access annotation, admit
    result/replay, and cancel needs at least an edit grant.
    """
    from inqtrix.auth.permissions import SharePermission

    shared = submit_noop(
        store,
        created_by_sub="user-owner",
        created_by_tenant_id="default",
        workspace_id="ws-owner",
    )
    own = submit_noop(
        store,
        created_by_sub="user-recipient",
        created_by_tenant_id="default",
        workspace_id="ws-recipient",
    )
    wait_for_status(store, shared["run_id"], {"completed"})
    wait_for_status(store, own["run_id"], {"completed"})
    recipient = _scoped("user-recipient")
    grants = {shared["run_id"]: SharePermission.VIEW}

    summary = store.get(
        shared["run_id"], visible_to=recipient, also_visible=grants
    )
    assert summary["access"] == {"via": "share", "permission": "view"}
    assert "access" not in store.get(own["run_id"], visible_to=recipient)

    listed = store.list(
        workspace_id="ws-recipient",
        visible_to=recipient,
        also_visible=grants,
    )
    by_id = {item["run_id"]: item for item in listed}
    assert set(by_id) == {own["run_id"], shared["run_id"]}
    assert by_id[shared["run_id"]]["workspace_id"] == "ws-owner"
    assert by_id[shared["run_id"]]["access"]["permission"] == "view"
    assert "access" not in by_id[own["run_id"]]

    result = store.result(
        shared["run_id"], visible_to=recipient, also_visible=grants
    )
    assert result["answer"] == "fertig"

    subscription = store.subscribe(
        shared["run_id"], visible_to=recipient, also_visible=grants
    )
    try:
        assert any(
            event["type"] == "inqtrix.run.completed"
            for event in subscription.replay
        )
    finally:
        subscription.close()

    with pytest.raises(RunNotFound):
        store.cancel(
            shared["run_id"], visible_to=recipient, also_visible=grants
        )
    cancelled = store.cancel(
        shared["run_id"],
        visible_to=recipient,
        also_visible={shared["run_id"]: SharePermission.EDIT},
    )
    assert cancelled["status"] == "completed"

    with pytest.raises(RunNotFound):
        store.get(shared["run_id"], visible_to=_scoped("user-stranger"))
