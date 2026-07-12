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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.exceptions import AgentProviderTimeout, AgentRateLimited
from inqtrix.runs.postgres_store import PostgresRunStore
from inqtrix.server.runs import (
    RunActive,
    RunNotFound,
    RunParentInactive,
    RunSessionActive,
)
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


class _RecordingDispatchQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    def enqueue(self, *, run_id: str, tenant_id: str) -> None:
        self.enqueued.append((run_id, tenant_id))


def test_worker_claim_store_dispatches_children_and_parent_wakes(
    store,
) -> None:
    """Claim ownership stays local while dispatch returns to Valkey."""
    del store  # fixture provides isolated migrated rows for this test
    queue = _RecordingDispatchQueue()
    dispatching = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        dispatch_queue=queue,
        recover_orphans=False,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-claim-dispatch",
    )
    try:
        parent = dispatching.submit(
            question="parent",
            stack_name="default",
            work=lambda _handle: None,
            workspace_id="ws-agent",
            kind="agent",
            session_id="session-dispatch",
            request_payload={
                "question": "parent",
                "body": {"mode": "workspace_agent"},
            },
        )
        assert (
            dispatching.dispatch_status(parent["run_id"], "default")
            == "queued"
        )
        assert dispatching.dispatch_status(parent["run_id"], "other") is None
        parent_claim = dispatching.claim_for_execution(
            parent["run_id"], "default", allow_takeover=False
        )
        assert parent_claim is not None
        assert parent_claim.workspace_id == "ws-agent"
        assert (
            dispatching.dispatch_status(parent["run_id"], "default")
            == "running"
        )
        child = dispatching.submit(
            question="child",
            stack_name="default",
            work=lambda _handle: None,
            workspace_id="ws-agent",
            kind="agent_child",
            parent_run_id=parent["run_id"],
            root_run_id=parent["run_id"],
            session_id="session-dispatch",
            request_payload={
                "question": "child",
                "body": {
                    "mode": "research",
                    "parent_task_id": "task-dispatch",
                    "parent_task_attempt": 2,
                },
            },
        )

        def child_progress() -> list[dict]:
            subscription = dispatching.subscribe(parent["run_id"])
            try:
                return [
                    event
                    for event in subscription.replay
                    if event["type"] == "inqtrix.agent.child.progress"
                ]
            finally:
                subscription.close()

        queued_count = len(child_progress())
        assert queued_count == 1
        dispatching.emit(
            child["run_id"],
            "inqtrix.output_text.delta",
            {"delta": "not copied to parent"},
        )
        assert len(child_progress()) == queued_count

        child_claim = dispatching.claim_for_execution(
            child["run_id"], "default", allow_takeover=False
        )
        assert child_claim is not None
        assert child_claim.workspace_id == "ws-agent"
        dispatching.mark_waiting(
            parent["run_id"],
            status="waiting_for_children",
            fence_attempt=parent_claim.attempt,
        )
        assert (
            dispatching.dispatch_status(parent["run_id"], "default")
            == "waiting_for_children"
        )
        dispatching.emit(
            child["run_id"],
            "inqtrix.node.started",
            {"node": "search", "snapshot": {"current_node": "search"}},
            fence_attempt=child_claim.attempt,
        )
        assert dispatching.complete(
            child["run_id"],
            {"answer": "done", "metrics": {"sources": 3}},
            snapshot={"current_node": "answer", "done": True},
            fence_attempt=child_claim.attempt,
        )

        assert dispatching.get(parent["run_id"])["status"] == "queued"
        assert (
            dispatching.dispatch_status(parent["run_id"], "default")
            == "queued"
        )
        assert [run_id for run_id, _tenant in queue.enqueued] == [
            parent["run_id"],
            child["run_id"],
            parent["run_id"],
        ]
        projected = child_progress()
        assert projected[-1]["data"]["run_status"] == "completed"
        assert projected[-1]["data"]["task_id"] == "task-dispatch"
        assert projected[-1]["data"]["attempt"] == 2
    finally:
        dispatching.close()


@pytest.mark.asyncio
async def test_migration_scrubs_active_legacy_child_token_budget(
    store,
) -> None:
    """0043 removes planner caps before a worker can replay the child."""
    del store
    queue = _RecordingDispatchQueue()
    durable = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        dispatch_queue=queue,
        recover_orphans=False,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-legacy-budget",
    )
    root = durable.submit(
        question="root",
        stack_name="default",
        work=lambda _handle: None,
        kind="agent",
        request_payload={"body": {"mode": "workspace_agent"}},
    )
    child = durable.submit(
        question="legacy child",
        stack_name="default",
        work=lambda _handle: None,
        kind="agent_child",
        parent_run_id=root["run_id"],
        request_payload={
            "body": {
                "mode": "research",
                "token_budget": 1800,
                "parent_task_id": "task-legacy",
            }
        },
    )
    migration = import_module(
        "inqtrix.storage.migrations.versions."
        "0043_agent_task_execution_contract"
    )
    engine = build_engine(TEST_DATABASE_URL)
    factory = build_session_factory(engine)
    try:
        async with factory() as session:
            async with session.begin():
                before = (
                    await session.execute(
                        text(
                            "SELECT request_payload::jsonb->'body'->>"
                            "'token_budget' FROM runs "
                            "WHERE run_id = :run_id"
                        ),
                        {"run_id": child["run_id"]},
                    )
                ).scalar_one()
                await session.execute(
                    text(migration._LEGACY_CHILD_BUDGET_BACKFILL_SQL)
                )
                after = (
                    await session.execute(
                        text(
                            "SELECT request_payload::jsonb->'body'->>"
                            "'token_budget' FROM runs "
                            "WHERE run_id = :run_id"
                        ),
                        {"run_id": child["run_id"]},
                    )
                ).scalar_one()
    finally:
        durable.close()
        await engine.dispose()

    assert before == "1800"
    assert after is None


def test_nested_children_use_canonical_root_and_cancel_recursively(
    store,
) -> None:
    """Durable lineage is parent-owned and root cancel covers all depths."""
    release = threading.Event()

    def blocking_work(handle) -> None:
        release.wait(timeout=10)
        if handle.cancel_event.is_set():
            handle.cancel("client_requested_cancel")
        else:
            handle.complete({"answer": "done"})

    root = store.submit(
        question="root",
        stack_name="default",
        work=blocking_work,
        kind="agent",
        request_payload={"body": {"mode": "workspace_agent"}},
    )
    finished_child = store.submit(
        question="finished child",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "finished"}),
        kind="agent_child",
        parent_run_id=root["run_id"],
        root_run_id=root["run_id"],
        request_payload={"body": {"mode": "research"}},
    )
    wait_for_status(store, finished_child["run_id"], {"completed"})
    child = store.submit(
        question="child",
        stack_name="default",
        work=blocking_work,
        kind="agent_child",
        parent_run_id=root["run_id"],
        root_run_id="caller-controlled-wrong-root",
        request_payload={"body": {"mode": "research"}},
    )
    grandchild = store.submit(
        question="grandchild",
        stack_name="default",
        work=blocking_work,
        kind="agent_child",
        parent_run_id=child["run_id"],
        root_run_id=child["run_id"],
        request_payload={"body": {"mode": "research"}},
    )
    try:
        assert child["root_run_id"] == root["run_id"]
        assert grandchild["root_run_id"] == root["run_id"]
        _summary, affected = store.cancel_tree(root["run_id"])
        assert set(affected) == {
            root["run_id"],
            finished_child["run_id"],
            child["run_id"],
            grandchild["run_id"],
        }
        release.set()
        for run in (root, child, grandchild):
            wait_for_status(store, run["run_id"], {"cancelled"})
    finally:
        release.set()


def test_origin_key_submit_or_find_is_atomic_postgres(store) -> None:
    release = threading.Event()

    def root_work(handle) -> None:
        release.wait(timeout=10)
        if handle.cancel_event.is_set():
            handle.cancel("client_requested_cancel")
        else:
            handle.complete({"answer": "done"})

    root = store.submit(
        question="root",
        stack_name="default",
        work=root_work,
        kind="agent",
        request_payload={"body": {"mode": "workspace_agent"}},
    )
    barrier = threading.Barrier(3)

    def submit_same_origin() -> dict:
        barrier.wait()
        return store.submit(
            question="same child",
            stack_name="default",
            work=lambda handle: handle.complete({"answer": "child"}),
            kind="agent_child",
            parent_run_id=root["run_id"],
            origin_key="task-1:attempt-1",
            request_payload={"body": {"mode": "research"}},
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit_same_origin) for _ in range(2)]
            barrier.wait()
            results = [future.result(timeout=10) for future in futures]
        assert results[0]["run_id"] == results[1]["run_id"]
        assert len(store.children(root["run_id"])) == 1
    finally:
        store.cancel(root["run_id"])
        release.set()
        wait_for_status(store, root["run_id"], {"cancelled"})


def test_child_submit_racing_root_cancel_is_serialized_postgres(store) -> None:
    """Either admission wins and is cascaded, or cancellation rejects it."""
    release = threading.Event()

    def blocking_work(handle) -> None:
        release.wait(timeout=10)
        if handle.cancel_event.is_set():
            handle.cancel("client_requested_cancel")
        else:
            handle.complete({"answer": "done"})

    root = store.submit(
        question="root",
        stack_name="default",
        work=blocking_work,
        kind="agent",
        request_payload={"body": {"mode": "workspace_agent"}},
    )
    barrier = threading.Barrier(3)

    def submit_child() -> dict:
        barrier.wait()
        return store.submit(
            question="racing child",
            stack_name="default",
            work=blocking_work,
            kind="agent_child",
            parent_run_id=root["run_id"],
            origin_key="racing-origin",
            request_payload={"body": {"mode": "research"}},
        )

    def cancel_root() -> dict:
        barrier.wait()
        return store.cancel(root["run_id"])

    submitted: dict | None = None
    rejected = False
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            submit_future = pool.submit(submit_child)
            cancel_future = pool.submit(cancel_root)
            barrier.wait()
            cancel_future.result(timeout=10)
            try:
                submitted = submit_future.result(timeout=10)
            except RunParentInactive:
                rejected = True
        release.set()
        wait_for_status(store, root["run_id"], {"cancelled"})
        assert (submitted is not None) is not rejected
        if submitted is not None:
            wait_for_status(store, submitted["run_id"], {"cancelled"})
        else:
            assert store.children(root["run_id"]) == []
    finally:
        release.set()


def test_cancelled_child_park_miss_projects_and_wakes_parent(store) -> None:
    del store
    queue = _RecordingDispatchQueue()
    dispatching = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        dispatch_queue=queue,
        recover_orphans=False,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-cancelled-park",
    )
    try:
        parent = dispatching.submit(
            question="parent",
            stack_name="default",
            work=lambda _handle: None,
            kind="agent",
            session_id="session-cancelled-park",
            request_payload={"body": {"mode": "workspace_agent"}},
        )
        parent_claim = dispatching.claim_for_execution(
            parent["run_id"], "default", allow_takeover=False
        )
        assert parent_claim is not None
        child = dispatching.submit(
            question="child",
            stack_name="default",
            work=lambda _handle: None,
            kind="agent_child",
            parent_run_id=parent["run_id"],
            root_run_id=parent["run_id"],
            request_payload={
                "body": {
                    "parent_task_id": "task-cancelled-park",
                    "parent_task_attempt": 1,
                }
            },
        )
        child_claim = dispatching.claim_for_execution(
            child["run_id"], "default", allow_takeover=False
        )
        assert child_claim is not None
        dispatching.mark_waiting(
            parent["run_id"],
            status="waiting_for_children",
            fence_attempt=parent_claim.attempt,
        )

        cancelled = dispatching.cancel(child["run_id"])
        assert cancelled["status"] == "running"
        dispatching.mark_waiting(
            child["run_id"],
            status="waiting_for_input",
            fence_attempt=child_claim.attempt,
        )

        assert dispatching.get(child["run_id"])["status"] == "cancelled"
        assert dispatching.get(parent["run_id"])["status"] == "queued"
        child_events = dispatching.subscribe(child["run_id"])
        try:
            assert child_events.replay[-1]["type"] == "inqtrix.run.cancelled"
        finally:
            child_events.close()
        parent_events = dispatching.subscribe(parent["run_id"])
        try:
            projected = [
                event
                for event in parent_events.replay
                if event["type"] == "inqtrix.agent.child.progress"
            ]
            assert projected[-1]["data"]["run_status"] == "cancelled"
            assert projected[-1]["data"]["task_id"] == (
                "task-cancelled-park"
            )
        finally:
            parent_events.close()
        assert queue.enqueued[-1] == (parent["run_id"], "default")
    finally:
        dispatching.close()


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


def test_parallel_root_agent_session_claim_accepts_exactly_one(store):
    """The partial unique index is the cross-worker execution lease."""
    hold = threading.Event()
    barrier = threading.Barrier(2)

    def work(handle):
        hold.wait(timeout=10.0)
        handle.complete({"answer": "ok", "metrics": {}})

    def submit() -> str:
        barrier.wait(timeout=5.0)
        return store.submit(
            question="Agent",
            stack_name="default",
            work=work,
            request_payload={"question": "Agent"},
            kind="agent",
            session_id="sess-pg-exclusive",
        )["run_id"]

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit) for _ in range(2)]
            outcomes: list[object] = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=10.0))
                except Exception as exc:  # noqa: BLE001 - contract assertion
                    outcomes.append(exc)
        assert sum(isinstance(item, str) for item in outcomes) == 1
        assert sum(isinstance(item, RunSessionActive) for item in outcomes) == 1
    finally:
        hold.set()


def test_metrics_snapshot_counts_running_rows(store):
    """The durable metrics snapshot counts real RUNNING rows; no capacity.

    Capacity is None on the durable backend (the worker fleet owns the
    slots), and the RUNNING count comes from a live grouped COUNT — so a
    genuinely executing run shows up as active=1.
    """
    import threading

    idle = store.metrics_snapshot()
    assert idle.active == 0
    assert idle.queued == 0
    assert idle.capacity is None

    release = threading.Event()

    def blocking_work(handle):
        handle.emit(
            "inqtrix.node.started",
            {"node": "classify", "snapshot": {"current_node": "classify"}},
        )
        release.wait(timeout=10)
        handle.complete(
            {"answer": "fertig"}, snapshot={"current_node": "answer", "done": True}
        )

    summary = submit_noop(store, work=blocking_work)
    try:
        wait_for_status(store, summary["run_id"], {"running"})
        snap = store.metrics_snapshot()
        assert snap.active == 1
        assert snap.capacity is None
    finally:
        release.set()
    wait_for_status(store, summary["run_id"], {"completed"})
    assert store.metrics_snapshot().active == 0


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


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (
            AgentRateLimited("model-a", RuntimeError("429")),
            "rate_limited",
        ),
        (AgentProviderTimeout("provider request"), "provider_timeout"),
    ],
)
def test_no_queue_work_preserves_typed_native_failures(
    store,
    failure: Exception,
    error_type: str,
) -> None:
    def boom(_handle):
        raise failure

    summary = submit_noop(store, work=boom)
    final = wait_for_status(store, summary["run_id"], {"failed"})

    assert final["error"]["type"] == error_type


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


def test_worker_shape_store_never_sweeps_foreign_rows(store):
    """The queue-backed worker builds its store with ``queue=None``
    (claim-mode wiring) — with ``recover_orphans=False`` its first DB
    touch must NOT fail runs another process still owns. Regression for
    the deployment-wide "Verwaister Run" sweep on worker start."""
    import threading

    release = threading.Event()

    def slow(handle):
        release.wait(10)
        handle.complete({"answer": "spaet"})

    try:
        running = submit_noop(store, work=slow)
        wait_for_status(store, running["run_id"], {"running"})

        worker_shaped = PostgresRunStore(
            engine=build_engine(TEST_DATABASE_URL),
            app_role=APP_ROLE,
            queue=None,
            recover_orphans=False,
            max_concurrent=2,
            max_queue_size=10,
            completed_ttl_seconds=300,
            worker_id="pytest-worker-shape",
        )
        try:
            # get() runs the lazy cleanup — the foreign RUNNING row
            # must survive it untouched.
            assert (
                worker_shaped.get(running["run_id"])["status"] == "running"
            )
        finally:
            worker_shaped.close()
    finally:
        release.set()
    wait_for_status(store, running["run_id"], {"completed"})


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


# ---------------------------------------------------------------------------
# M3: agent run tree + waiting statuses (lockstep with the memory store)
# ---------------------------------------------------------------------------


def test_agent_tree_keys_and_children_listing(store):
    calls = {"count": 0}

    def parked_parent(handle):
        calls["count"] += 1
        if calls["count"] == 1:
            handle.wait("waiting_for_approval")
            return
        handle.complete({"answer": "fortgesetzt"})

    parent = submit_noop(
        store,
        work=parked_parent,
        kind="agent",
        session_id="sess-pg",
    )
    wait_for_status(store, parent["run_id"], {"waiting_for_approval"})
    assert parent["kind"] == "agent"
    assert parent["children_url"] == f"/v1/runs/{parent['run_id']}/children"
    assert parent["session_id"] == "sess-pg"

    first = submit_noop(
        store,
        question="erste Teilaufgabe",
        kind="agent_child",
        parent_run_id=parent["run_id"],
        root_run_id=parent["run_id"],
    )
    time.sleep(0.02)
    second = submit_noop(
        store,
        question="zweite Teilaufgabe",
        kind="agent_child",
        parent_run_id=parent["run_id"],
        root_run_id=parent["run_id"],
    )
    for summary in (first, second):
        wait_for_status(store, summary["run_id"], {"completed"})

    assert first["parent_run_id"] == parent["run_id"]
    assert first["root_run_id"] == parent["run_id"]
    children = store.children(parent["run_id"])
    assert [child["run_id"] for child in children] == [
        second["run_id"],
        first["run_id"],
    ]
    store.resume_run(parent["run_id"])
    wait_for_status(store, parent["run_id"], {"completed"})
    # Standard runs stay byte-identical: no agent-tree keys.
    standard = submit_noop(store, question="normaler Lauf")
    assert set(standard.keys()) == RUN_SUMMARY_KEYS
    wait_for_status(store, standard["run_id"], {"completed"})


def test_waiting_lifecycle_parks_resumes_and_completes(store):
    calls = {"count": 0}

    def segmented(handle):
        calls["count"] += 1
        if calls["count"] == 1:
            handle.wait("waiting_for_approval")
            return
        handle.complete({"answer": "fortgesetzt"})

    summary = submit_noop(store, work=segmented)
    run_id = summary["run_id"]
    wait_for_status(store, run_id, {"waiting_for_approval"})
    # The auto-complete safety net must NOT complete the parked run.
    time.sleep(0.2)
    assert store.get(run_id)["status"] == "waiting_for_approval"

    resumed = store.resume_run(run_id)
    assert resumed["status"] in {"queued", "running"}
    wait_for_status(store, run_id, {"completed"})
    assert calls["count"] == 2
    assert store.result(run_id)["answer"] == "fortgesetzt"

    events = store.subscribe(run_id)
    try:
        types = [event["type"] for event in events.replay]
        assert "inqtrix.run.waiting" in types
        queued = [
            event
            for event in events.replay
            if event["type"] == "inqtrix.run.queued"
        ]
        assert queued[-1]["data"].get("resumed") is True
        sequences = [event["sequence"] for event in events.replay]
        assert sequences == list(range(1, len(sequences) + 1))
    finally:
        events.close()


def test_cancel_while_waiting_cascades_over_children(store):
    parent = submit_noop(
        store,
        work=lambda handle: handle.wait("waiting_for_input"),
        kind="agent",
    )
    wait_for_status(store, parent["run_id"], {"waiting_for_input"})
    child = submit_noop(
        store,
        question="Teilaufgabe",
        work=lambda handle: handle.wait("waiting_for_input"),
        kind="agent_child",
        parent_run_id=parent["run_id"],
        root_run_id=parent["run_id"],
    )
    wait_for_status(store, child["run_id"], {"waiting_for_input"})

    cancelled = store.cancel(parent["run_id"])

    assert cancelled["status"] == "cancelled"
    assert store.get(child["run_id"])["status"] == "cancelled"
    for run_id in (parent["run_id"], child["run_id"]):
        events = store.subscribe(run_id)
        try:
            assert events.replay[-1]["type"] == "inqtrix.run.cancelled"
            assert (
                events.replay[-1]["data"]["reason"]
                == "cancelled_while_waiting"
            )
        finally:
            events.close()
    with pytest.raises(RunActive):
        store.resume_run(parent["run_id"])


def test_waiting_ttl_auto_cancels_with_approval_timeout(engine):
    impatient = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-worker-ttl",
        waiting_ttl_seconds=0.4,
    )
    try:
        summary = submit_noop(
            impatient,
            work=lambda handle: handle.wait("waiting_for_approval"),
        )
        run_id = summary["run_id"]
        wait_for_status(impatient, run_id, {"waiting_for_approval"})
        time.sleep(0.5)

        # Any store touch runs the sweep.
        final = wait_for_status(impatient, run_id, {"cancelled"})
        assert final["status"] == "cancelled"
        events = impatient.subscribe(run_id)
        try:
            assert events.replay[-1]["type"] == "inqtrix.run.cancelled"
            assert events.replay[-1]["data"]["reason"] == "approval_timeout"
        finally:
            events.close()
    finally:
        impatient.close()


def test_waiting_child_ttl_projects_terminal_and_wakes_parent(store):
    del store
    impatient = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-child-ttl",
        waiting_ttl_seconds=10.0,
    )
    calls = {"count": 0}
    child_id = {"value": ""}

    def child_work(handle):
        handle.wait("waiting_for_input")

    def parent_work(handle):
        calls["count"] += 1
        if calls["count"] == 1:
            child = impatient.submit(
                question="Kind-Recherche",
                stack_name="default",
                work=child_work,
                kind="agent_child",
                parent_run_id=handle.run_id,
                root_run_id=handle.run_id,
                request_payload={
                    "body": {
                        "parent_task_id": "task-child-ttl",
                        "parent_task_attempt": 1,
                    }
                },
            )
            child_id["value"] = child["run_id"]
            handle.wait("waiting_for_children")
            return
        handle.complete({"answer": "resumed", "metrics": {}})

    try:
        parent = impatient.submit(
            question="Agent-Auftrag",
            stack_name="default",
            work=parent_work,
            kind="agent",
            request_payload={"body": {"mode": "workspace_agent"}},
        )
        parent_id = parent["run_id"]
        wait_for_status(impatient, parent_id, {"waiting_for_children"})
        wait_for_status(impatient, child_id["value"], {"waiting_for_input"})

        import asyncio

        async def _age_child_wait() -> None:
            aging_engine = build_engine(TEST_DATABASE_URL)
            factory = build_session_factory(aging_engine)
            try:
                async with factory() as session:
                    async with session.begin():
                        await session.execute(
                            text(
                                "UPDATE runs SET waiting_since = :old "
                                "WHERE run_id = :run_id"
                            ),
                            {
                                "old": time.time() - 20.0,
                                "run_id": child_id["value"],
                            },
                        )
            finally:
                await aging_engine.dispose()

        asyncio.run(_age_child_wait())
        assert impatient.get(child_id["value"])["status"] == "cancelled"
        wait_for_status(impatient, parent_id, {"completed"})

        child_events = impatient.subscribe(child_id["value"])
        try:
            assert child_events.replay[-1]["type"] == "inqtrix.run.cancelled"
            assert child_events.replay[-1]["data"]["reason"] == (
                "approval_timeout"
            )
        finally:
            child_events.close()

        events = impatient.subscribe(parent_id)
        try:
            projected = [
                event
                for event in events.replay
                if event["type"] == "inqtrix.agent.child.progress"
            ]
            assert projected[-1]["data"]["run_status"] == "cancelled"
            resumed = [
                event
                for event in events.replay
                if event["type"] == "inqtrix.run.queued"
                and event["data"].get("resumed")
            ]
            assert projected[-1]["sequence"] < resumed[-1]["sequence"]
        finally:
            events.close()
        assert calls["count"] == 2
    finally:
        impatient.close()


def test_orphan_sweep_spares_waiting_rows(store):
    summary = submit_noop(
        store, work=lambda handle: handle.wait("waiting_for_approval")
    )
    run_id = summary["run_id"]
    wait_for_status(store, run_id, {"waiting_for_approval"})

    # A second process (fresh store instance) sweeps queued/running
    # orphans on first touch — a parked run must survive it: in queue
    # mode any worker can resume it later.
    second = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-worker-2",
    )
    try:
        assert second.get(run_id)["status"] == "waiting_for_approval"
        # But the closure died with the first process: resuming from
        # the restarted no-queue store fails loudly instead of hanging.
        with pytest.raises(RunActive):
            second.resume_run(run_id)
    finally:
        second.close()


def test_stuck_row_failsafe_spares_waiting_rows(store, engine):
    """A parked run aged past the stuck threshold must NOT be deleted.

    The stuck failsafe keys on created_at while the waiting TTL keys on
    waiting_since — without the waiting exclusion the failsafe would
    always fire first and erase the parked run without any event.
    """
    calls = {"count": 0}

    def segmented(handle):
        calls["count"] += 1
        if calls["count"] == 1:
            handle.wait("waiting_for_approval")
            return
        handle.complete({"answer": "nach dem Sweep fortgesetzt"})

    summary = submit_noop(store, work=segmented)
    run_id = summary["run_id"]
    wait_for_status(store, run_id, {"waiting_for_approval"})

    # Age the row past the stuck threshold; keep the wait fresh. A
    # dedicated engine: the fixture engine is bound to another loop.
    import asyncio

    async def _age():
        aging_engine = build_engine(TEST_DATABASE_URL)
        try:
            factory = build_session_factory(aging_engine)
            async with factory() as session:
                async with session.begin():
                    await session.execute(
                        text(
                            "UPDATE runs SET created_at = :old "
                            "WHERE run_id = :rid"
                        ),
                        {"old": time.time() - 30 * 86_400, "rid": run_id},
                    )
        finally:
            await aging_engine.dispose()

    asyncio.run(_age())

    # Any store touch runs the cleanup, including the stuck failsafe.
    assert store.get(run_id)["status"] == "waiting_for_approval"
    resumed = store.resume_run(run_id)
    assert resumed["status"] in {"queued", "running"}
    wait_for_status(store, run_id, {"completed"})


def test_waiting_ttl_sweep_does_not_deadlock_concurrent_dispatch(engine):
    """TTL sweep firing while runs dispatch must not freeze the store.

    Regression for the review P1: the sweep runs on the store's event
    loop; releasing local closures there while a dispatcher holds the
    store lock blocked on that loop deadlocked every later call.
    """
    impatient = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=2,
        max_queue_size=20,
        completed_ttl_seconds=300,
        worker_id="pytest-worker-deadlock",
        waiting_ttl_seconds=0.4,
    )
    try:
        parked = submit_noop(
            impatient,
            work=lambda handle: handle.wait("waiting_for_approval"),
        )
        wait_for_status(impatient, parked["run_id"], {"waiting_for_approval"})
        time.sleep(0.5)

        # Concurrent submitters (dispatch holds the store lock while
        # blocking on the store loop) and readers (their cleanup sweep
        # runs ON that loop and records the timed-out parked run).
        # With the regression this interleaving wedges the whole store.
        import threading

        errors: list[BaseException] = []

        def _submit_and_wait() -> None:
            try:
                summary = submit_noop(impatient)
                wait_for_status(impatient, summary["run_id"], {"completed"})
            except BaseException as exc:  # noqa: BLE001 — surfaced below
                errors.append(exc)

        def _read_loop() -> None:
            try:
                for _ in range(20):
                    impatient.list()
                    time.sleep(0.01)
            except BaseException as exc:  # noqa: BLE001 — surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=_submit_and_wait) for _ in range(4)]
        threads += [threading.Thread(target=_read_loop) for _ in range(2)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 15
        for thread in threads:
            thread.join(timeout=max(0.1, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            pytest.fail("store deadlocked: workers never finished")
        assert not errors, errors
        final = wait_for_status(impatient, parked["run_id"], {"cancelled"})
        assert final["status"] == "cancelled"
    finally:
        impatient.close()


def test_mark_waiting_rejections_match_memory(store):
    summary = submit_noop(store)
    run_id = summary["run_id"]
    wait_for_status(store, run_id, {"completed"})

    with pytest.raises(ValueError, match="not a waiting status"):
        store.mark_waiting(run_id, status="running")
    with pytest.raises(RunActive):
        store.mark_waiting(run_id, status="waiting_for_input")
    with pytest.raises(RunNotFound):
        store.mark_waiting("run_unbekannt", status="waiting_for_input")


def test_fenced_park_with_pending_cancel_resolves_as_cancel(store):
    """A fenced park whose OWN attempt lost the CAS to a pending cancel
    must resolve as cancel (memory parity) — a fenced miss is only a
    zombie signal when claimed_by/attempt point at ANOTHER attempt."""
    import threading

    release = threading.Event()

    def slow(handle):
        release.wait(10)
        handle.complete({"answer": "spaet"})

    try:
        submit_noop(store, work=slow)
        submit_noop(store, work=slow)
        own = submit_noop(store, work=slow)["run_id"]
        zombie = submit_noop(store, work=slow)["run_id"]

        claim = store.claim_for_execution(
            own, "default", allow_takeover=False
        )
        assert claim is not None and claim.attempt == 1
        # Two-phase request against the running claim...
        store.cancel(own)
        # ...then the worker parks: the pending cancel wins, the run
        # ends CANCELLED — not RunActive -> worker fail() -> FAILED.
        store.mark_waiting(
            own, status="waiting_for_approval", fence_attempt=1
        )
        assert store.get(own)["status"] == "cancelled"
        events = store.subscribe(own)
        try:
            assert events.replay[-1]["type"] == "inqtrix.run.cancelled"
            assert (
                events.replay[-1]["data"]["reason"]
                == "cancelled_while_waiting"
            )
        finally:
            events.close()

        # The zombie guard stays: a superseded attempt's park neither
        # parks nor cancels the live attempt's run.
        first = store.claim_for_execution(
            zombie, "default", allow_takeover=False
        )
        assert first is not None and first.attempt == 1
        second = store.claim_for_execution(
            zombie, "default", allow_takeover=True
        )
        assert second is not None and second.attempt == 2
        with pytest.raises(RunActive, match="another worker attempt"):
            store.mark_waiting(
                zombie, status="waiting_for_approval", fence_attempt=1
            )
        assert store.get(zombie)["status"] == "running"
    finally:
        release.set()


def test_resume_error_types_match_memory(store):
    with pytest.raises(RunNotFound):
        store.resume_run("run_unbekannt")

    summary = submit_noop(store)
    run_id = summary["run_id"]
    wait_for_status(store, run_id, {"completed"})
    with pytest.raises(RunActive, match="is not waiting"):
        store.resume_run(run_id)


def test_emit_after_terminal_is_dropped_loudly(store):
    summary = submit_noop(store)
    run_id = summary["run_id"]
    wait_for_status(store, run_id, {"completed"})

    store.emit(run_id, "inqtrix.agent.artifact.updated", {"revision": 2})

    events = store.subscribe(run_id)
    try:
        assert events.replay[-1]["type"] == "inqtrix.run.completed"
    finally:
        events.close()


def test_child_projection_cannot_follow_parent_terminal_event(store):
    """Parent lock orders a racing child projection before terminalization."""
    del store
    import asyncio
    from types import MethodType

    queue = _RecordingDispatchQueue()
    projector = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        dispatch_queue=queue,
        recover_orphans=False,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-projector",
    )
    terminator = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        dispatch_queue=queue,
        recover_orphans=False,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-terminator",
    )
    projection_selected = threading.Event()
    release_projection = threading.Event()
    errors: list[BaseException] = []
    try:
        parent = projector.submit(
            question="parent",
            stack_name="default",
            work=lambda _handle: None,
            kind="agent",
            session_id="session-projection-order",
            request_payload={"body": {"mode": "workspace_agent"}},
        )
        child = projector.submit(
            question="child",
            stack_name="default",
            work=lambda _handle: None,
            kind="agent_child",
            parent_run_id=parent["run_id"],
            root_run_id=parent["run_id"],
            request_payload={
                "body": {"parent_task_id": "task-projection-order"}
            },
        )
        original_append = projector._append_events_db

        async def _gated_append(
            self,
            session,
            run_id,
            tenant_id,
            events,
        ):
            del self
            if run_id == parent["run_id"] and any(
                event_type == "inqtrix.agent.child.progress"
                for event_type, _payload in events
            ):
                projection_selected.set()
                assert await asyncio.to_thread(
                    release_projection.wait, 10.0
                )
            await original_append(session, run_id, tenant_id, events)

        projector._append_events_db = MethodType(  # type: ignore[method-assign]
            _gated_append, projector
        )

        def _emit_child() -> None:
            try:
                projector.emit(
                    child["run_id"],
                    "inqtrix.node.started",
                    {"node": "search", "snapshot": {"current_node": "search"}},
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def _fail_parent() -> None:
            try:
                terminator.fail(parent["run_id"], "terminal race")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        child_thread = threading.Thread(target=_emit_child)
        child_thread.start()
        assert projection_selected.wait(timeout=10.0)
        terminal_thread = threading.Thread(target=_fail_parent)
        terminal_thread.start()
        time.sleep(0.1)
        assert terminal_thread.is_alive()
        release_projection.set()
        child_thread.join(timeout=10.0)
        terminal_thread.join(timeout=10.0)
        assert not child_thread.is_alive()
        assert not terminal_thread.is_alive()
        assert not errors, errors

        events = terminator.subscribe(parent["run_id"])
        try:
            types = [event["type"] for event in events.replay]
            assert "inqtrix.agent.child.progress" in types
            assert types[-1] == "inqtrix.run.failed"
        finally:
            events.close()
    finally:
        release_projection.set()
        projector.close()
        terminator.close()


# -- A1: children park-and-resume (waiting_for_children) --------------------- #


class _SegmentedParent:
    """Two-segment parent work: submit children + park, then complete.

    Mirrors the checkpointed agent algorithm: the first dispatch
    submits the children and parks ``waiting_for_children``; the
    re-dispatch (after the store woke the run) completes.
    """

    def __init__(self, store, child_works, *, park_gate=None):
        self.store = store
        self.child_works = child_works
        self.park_gate = park_gate
        self.child_ids = []
        self.segments = 0
        self.run_id = ""

    def __call__(self, handle):
        self.run_id = handle.run_id
        self.segments += 1
        if self.segments == 1:
            for work in self.child_works:
                summary = self.store.submit(
                    question="Kind-Recherche",
                    stack_name="default",
                    work=work,
                    request_payload={"question": "kind"},
                    kind="agent_child",
                    parent_run_id=self.run_id,
                    root_run_id=self.run_id,
                )
                self.child_ids.append(summary["run_id"])
            if self.park_gate is not None:
                assert self.park_gate.wait(timeout=10.0)
            handle.wait("waiting_for_children")
            return
        handle.complete({"answer": "fertig", "metrics": {}})


def test_children_wake_resumes_parent_after_last_child(store):
    import threading

    release_second = threading.Event()

    def child_one(handle):
        handle.complete({"answer": "eins", "metrics": {}})

    def child_two(handle):
        assert release_second.wait(timeout=10.0)
        handle.complete({"answer": "zwei", "metrics": {}})

    parent = _SegmentedParent(store, [child_one, child_two])
    run_id = store.submit(
        question="Agent-Auftrag",
        stack_name="default",
        work=parent,
        request_payload={"question": "agent"},
        kind="agent",
    )["run_id"]
    try:
        wait_for_status(store, run_id, {"waiting_for_children"})
        wait_for_status(store, parent.child_ids[0], {"completed"})
        # One child outstanding: the parent must still wait.
        assert store.get(run_id)["status"] == "waiting_for_children"
        release_second.set()
        wait_for_status(store, run_id, {"completed"})
        assert parent.segments == 2
        events = store.subscribe(run_id)
        try:
            resumed = [
                event
                for event in events.replay
                if event["type"] == "inqtrix.run.queued"
                and event["data"].get("resumed")
            ]
            assert resumed, "the wake must emit queued{resumed:true}"
        finally:
            events.close()
    finally:
        release_second.set()


def test_cancelling_waiting_child_wakes_parent(store):
    def child_work(handle):
        handle.wait("waiting_for_input")

    parent = _SegmentedParent(store, [child_work])
    run_id = store.submit(
        question="Agent-Auftrag",
        stack_name="default",
        work=parent,
        request_payload={"question": "agent"},
        kind="agent",
    )["run_id"]
    wait_for_status(store, run_id, {"waiting_for_children"})
    child_id = parent.child_ids[0]
    wait_for_status(store, child_id, {"waiting_for_input"})

    store.cancel(child_id)

    wait_for_status(store, run_id, {"completed"})
    assert parent.segments == 2


def test_children_park_self_heals_when_child_already_terminal(store):
    import threading

    park_gate = threading.Event()

    def child_work(handle):
        handle.complete({"answer": "kind fertig", "metrics": {}})

    parent = _SegmentedParent(store, [child_work], park_gate=park_gate)
    run_id = store.submit(
        question="Agent-Auftrag",
        stack_name="default",
        work=parent,
        request_payload={"question": "agent"},
        kind="agent",
    )["run_id"]
    # Let the child terminate FIRST: its wake probe finds the parent
    # RUNNING and no-ops — the park-time self-heal must close the gap.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if parent.child_ids and store.get(parent.child_ids[0])[
            "status"
        ] == "completed":
            break
        time.sleep(0.05)
    else:
        pytest.fail("child never completed")
    park_gate.set()
    wait_for_status(store, run_id, {"completed"})
    assert parent.segments == 2


def test_cancel_of_children_parked_parent_cascades(store):
    import threading

    hold_child = threading.Event()

    def child_work(handle):
        hold_child.wait(timeout=10.0)
        if handle.cancel_event.is_set():
            handle.cancel("cancelled")
            return
        handle.complete({"answer": "kind", "metrics": {}})

    parent = _SegmentedParent(store, [child_work])
    run_id = store.submit(
        question="Agent-Auftrag",
        stack_name="default",
        work=parent,
        request_payload={"question": "agent"},
        kind="agent",
    )["run_id"]
    try:
        wait_for_status(store, run_id, {"waiting_for_children"})
        store.cancel(run_id)
        hold_child.set()
        wait_for_status(store, run_id, {"cancelled"})
        wait_for_status(store, parent.child_ids[0], {"cancelled"})
        # The cascaded child terminal write must not resurrect the
        # parent (the wake CAS keys on waiting_for_children).
        time.sleep(0.2)
        assert store.get(run_id)["status"] == "cancelled"
        assert parent.segments == 1
    finally:
        hold_child.set()


def test_per_user_cap_admission_counts_queued_and_running(engine):
    """1.3: durable per-user fairness bound; parked runs excluded."""
    import threading

    from inqtrix.server.runs import RunPerUserLimit

    store = PostgresRunStore(
        engine=engine,
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=3,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-cap",
        max_concurrent_per_user=2,
    )
    hold = threading.Event()

    def slow(handle):
        hold.wait(timeout=10.0)
        handle.complete({"answer": "ok", "metrics": {}})

    def submit(sub):
        return store.submit(
            question="F",
            stack_name="default",
            work=slow,
            request_payload={"question": "F"},
            created_by_sub=sub,
        )

    try:
        submit("user-a")
        submit("user-a")
        with pytest.raises(RunPerUserLimit):
            submit("user-a")
        # Distinct subject unaffected; anonymous never capped.
        submit("user-b")
        store.submit(
            question="F",
            stack_name="default",
            work=slow,
            request_payload={"question": "F"},
        )
    finally:
        hold.set()
        store.close()


def test_list_page_keyset_walks_history_and_positions_queued(store):
    """2.2: durable keyset paging + batched queue-position (no N+1)."""
    from inqtrix.pagination import decode_cursor

    # 5 completed runs (fast noop) — deterministic newest-first ordering.
    completed_ids = []
    for _ in range(5):
        completed_ids.append(submit_noop(store)["run_id"])
    for rid in completed_ids:
        wait_for_status(store, rid, {"completed"})

    seen = []
    after = None
    for _ in range(10):
        summaries, cursor = store.list_page(limit=2, after=after)
        seen.extend(s["run_id"] for s in summaries)
        if cursor is None:
            break
        after = decode_cursor(cursor)
    # Every run exactly once, newest-first.
    assert set(seen) == set(completed_ids)
    assert len(seen) == len(set(seen))
    assert seen == list(reversed(completed_ids))


def test_list_page_reports_true_global_queue_position_across_pages(store):
    import threading

    release = threading.Event()

    def blocking(handle):
        release.wait(timeout=10.0)
        handle.complete({"answer": "ok", "metrics": {}})

    # max_concurrent=2 -> 2 run, the rest queue. Submit 5.
    ids = []
    for i in range(5):
        ids.append(
            store.submit(
                question=f"q{i}",
                stack_name="default",
                work=blocking,
                request_payload={"question": "x"},
            )["run_id"]
        )
    try:
        # Let the store settle (2 running, 3 queued).
        wait_for_status(store, ids[0], {"running"})
        # Page size 2 forces the queued runs onto later pages; their
        # position must stay GLOBAL, not page-local.
        positions = {}
        after = None
        from inqtrix.pagination import decode_cursor

        for _ in range(10):
            summaries, cursor = store.list_page(limit=2, after=after)
            for s in summaries:
                if s["queue_position"] is not None:
                    positions[s["run_id"]] = s["queue_position"]
            if cursor is None:
                break
            after = decode_cursor(cursor)
        # The queued runs (submitted last, newest) carry ascending global
        # positions 1..3 regardless of which page they landed on.
        assert sorted(positions.values()) == [1, 2, 3]
    finally:
        release.set()
