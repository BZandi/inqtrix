"""Tests for the native in-memory run queue."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import pytest

import inqtrix.server.runs as runs_module
from inqtrix.server.runs import (
    RunActive,
    RunHandle,
    RunNotFound,
    RunQueueFull,
    RunStatus,
    RunStore,
    format_sse_event,
)
from inqtrix.settings import ServerSettings


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
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
    max_queue_size: int = 1,
    completed_ttl_seconds: int = 30,
    event_buffer_size: int = 10,
) -> RunStore:
    """Create a small run store for unit tests."""
    return RunStore(
        max_concurrent=max_concurrent,
        max_queue_size=max_queue_size,
        completed_ttl_seconds=completed_ttl_seconds,
        event_buffer_size=event_buffer_size,
    )


def test_from_settings_uses_native_run_limits() -> None:
    settings = ServerSettings(
        MAX_CONCURRENT=2,
        RUN_MAX_CONCURRENT=1,
        RUN_QUEUE_MAX_SIZE=7,
        RUN_COMPLETED_TTL_SECONDS=9,
        RUN_EVENT_BUFFER_SIZE=11,
    )

    store = RunStore.from_settings(settings)

    assert store._max_concurrent == 1
    assert store.submit(question="q", stack_name="default", work=lambda handle: handle.complete({}))[
        "status"
    ] in {"queued", "running", "completed"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_concurrent": 0}, "max_concurrent must be >= 1"),
        ({"max_queue_size": -1}, "max_queue_size must be >= 0"),
        ({"completed_ttl_seconds": -1}, "completed_ttl_seconds must be >= 0"),
        ({"event_buffer_size": 0}, "event_buffer_size must be >= 1"),
    ],
)
def test_store_rejects_invalid_limits(kwargs: dict[str, int], message: str) -> None:
    options = {
        "max_concurrent": 1,
        "max_queue_size": 1,
        "completed_ttl_seconds": 30,
        "event_buffer_size": 10,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        RunStore(**options)


def test_submit_dispatches_first_run_and_keeps_second_queued() -> None:
    first_started = threading.Event()
    release_first = threading.Event()

    def blocking_work(handle: RunHandle) -> None:
        first_started.set()
        release_first.wait(timeout=1)
        handle.complete({"answer": "done"})

    store = _store(max_concurrent=1, max_queue_size=2)

    first = store.submit(question="first", stack_name="default", work=blocking_work)
    _wait_until(lambda: first_started.is_set())
    second = store.submit(
        question="second",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "second"}),
        agent_overrides={"max_rounds": 2},
    )

    assert store.get(first["run_id"])["status"] == "running"
    assert second["status"] == "queued"
    assert second["queue_position"] == 1
    assert second["agent_overrides"] == {"max_rounds": 2}

    release_first.set()
    _wait_until(lambda: store.get(second["run_id"])["status"] == "completed")


def test_submit_summary_includes_run_mode() -> None:
    store = _store(max_concurrent=1, max_queue_size=1)

    summary = store.submit(
        question="direct",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "done"}),
        mode="direct_llm",
    )

    assert summary["mode"] == "direct_llm"
    _wait_until(lambda: store.get(summary["run_id"])["status"] == "completed")
    assert store.get(summary["run_id"])["mode"] == "direct_llm"


def test_submit_retries_run_id_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = iter(["run_duplicate", "run_duplicate", "run_unique"])
    monkeypatch.setattr(runs_module, "new_run_id", lambda: next(ids))
    store = _store(max_concurrent=1, max_queue_size=2)

    first = store.submit(
        question="first",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "first"}),
    )
    second = store.submit(
        question="second",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "second"}),
    )

    assert first["run_id"] == "run_duplicate"
    assert second["run_id"] == "run_unique"
    _wait_until(lambda: store.get(second["run_id"])["status"] == "completed")


def test_list_and_get_can_filter_by_workspace_id() -> None:
    store = _store(max_concurrent=1, max_queue_size=2)
    run_a = store.submit(
        question="workspace a",
        stack_name="default",
        workspace_id="ws_browser_a",
        work=lambda handle: handle.complete({"answer": "a"}),
    )
    run_b = store.submit(
        question="workspace b",
        stack_name="default",
        workspace_id="ws_browser_b",
        work=lambda handle: handle.complete({"answer": "b"}),
    )

    summaries = store.list(workspace_id="ws_browser_a")

    assert [summary["run_id"] for summary in summaries] == [run_a["run_id"]]
    assert store.get(run_a["run_id"], workspace_id="ws_browser_a")["workspace_id"] == "ws_browser_a"
    with pytest.raises(RunNotFound):
        store.get(run_b["run_id"], workspace_id="ws_browser_a")


def test_delete_removes_terminal_run_for_owner() -> None:
    store = _store(max_queue_size=2)
    store.import_completed_run(
        run_id="run_del_1",
        question="report",
        stack_name="default",
        result={"answer": "body"},
        workspace_id="ws_owner",
        created_by_sub="owner-1",
        created_by_tenant_id="default",
    )

    store.delete("run_del_1", workspace_id="ws_owner", requester_sub="owner-1")

    # The run is gone from the durable surface, so a reload cannot re-hydrate
    # it (the regression behind "deleted report comes back").
    assert store.list(workspace_id="ws_owner") == []
    with pytest.raises(RunNotFound):
        store.get("run_del_1", workspace_id="ws_owner")
    # Delete is not idempotent: a repeat is a clean 404, not a crash.
    with pytest.raises(RunNotFound):
        store.delete("run_del_1", workspace_id="ws_owner", requester_sub="owner-1")


def test_delete_refuses_non_owner_and_cross_workspace() -> None:
    store = _store(max_queue_size=2)
    store.import_completed_run(
        run_id="run_del_2",
        question="report",
        stack_name="default",
        result={"answer": "body"},
        workspace_id="ws_owner",
        created_by_sub="owner-1",
        created_by_tenant_id="default",
    )

    # Owner-only is stronger than cancel: a different creator gets the
    # indistinct RunNotFound and the run survives.
    with pytest.raises(RunNotFound):
        store.delete(
            "run_del_2", workspace_id="ws_owner", requester_sub="intruder"
        )
    # The right owner from the wrong namespace is denied too.
    with pytest.raises(RunNotFound):
        store.delete(
            "run_del_2", workspace_id="ws_other", requester_sub="owner-1"
        )
    assert [s["run_id"] for s in store.list(workspace_id="ws_owner")] == [
        "run_del_2"
    ]


def test_delete_allows_legacy_run_without_recorded_owner() -> None:
    store = _store(max_queue_size=2)
    store.import_completed_run(
        run_id="run_legacy",
        question="legacy report",
        stack_name="default",
        result={"answer": "body"},
        workspace_id="ws_legacy",
        created_by_sub=None,
        created_by_tenant_id="default",
    )

    # A pre-scoping run has no recorded creator; gating it on the workspace
    # alone keeps it deletable instead of orphaned forever, while a foreign
    # namespace is still denied.
    with pytest.raises(RunNotFound):
        store.delete(
            "run_legacy", workspace_id="ws_other", requester_sub="__anonymous__"
        )
    store.delete(
        "run_legacy", workspace_id="ws_legacy", requester_sub="__anonymous__"
    )
    assert store.list(workspace_id="ws_legacy") == []


def test_delete_refuses_a_still_active_run() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_work(handle: RunHandle) -> None:
        started.set()
        release.wait(timeout=1)
        handle.complete({"answer": "done"})

    store = _store(max_queue_size=2)
    summary = store.submit(
        question="running",
        stack_name="default",
        workspace_id="ws_owner",
        created_by_sub="owner-1",
        work=blocking_work,
    )
    _wait_until(started.is_set)

    # Deleting a record an executing worker still holds would let its final
    # write resurrect a half-gone run, so it is refused until terminal.
    with pytest.raises(RunActive):
        store.delete(
            summary["run_id"], workspace_id="ws_owner", requester_sub="owner-1"
        )

    release.set()
    _wait_until(
        lambda: store.get(summary["run_id"], workspace_id="ws_owner")["status"]
        == "completed"
    )
    store.delete(
        summary["run_id"], workspace_id="ws_owner", requester_sub="owner-1"
    )
    assert store.list(workspace_id="ws_owner") == []


def test_import_completed_run_persists_and_is_idempotent() -> None:
    store = _store(max_queue_size=2)
    first = store.import_completed_run(
        run_id="run_report_1",
        question="imported report",
        stack_name="default",
        result={"answer": "the report body", "metrics": {}},
        created_at=1000.0,
        workspace_id="ws_owner",
        created_by_sub="owner-1",
        created_by_tenant_id="default",
    )
    assert first["run_id"] == "run_report_1"
    assert first["status"] == "completed"
    # Lists + scopes to the owner's workspace, and the body is fetchable.
    assert [s["run_id"] for s in store.list(workspace_id="ws_owner")] == [
        "run_report_1"
    ]
    assert store.result("run_report_1")["answer"] == "the report body"
    # Re-importing the OWNER's own run is an idempotent no-op (one row).
    again = store.import_completed_run(
        run_id="run_report_1",
        question="imported report",
        stack_name="default",
        result={"answer": "ignored on re-import"},
        created_by_sub="owner-1",
        created_by_tenant_id="default",
    )
    assert again["run_id"] == "run_report_1"
    assert len(store.list()) == 1


def test_import_completed_run_never_overwrites_a_foreign_owner() -> None:
    store = _store(max_queue_size=2)
    store.import_completed_run(
        run_id="run_shared_id",
        question="A",
        stack_name="default",
        result={"answer": "owner A body"},
        created_by_sub="owner-a",
        created_by_tenant_id="default",
    )
    # A different principal importing the SAME id must NOT clobber or leak A's
    # run; it gets a fresh id instead (No Silent Fallbacks, no cross-user loss).
    other = store.import_completed_run(
        run_id="run_shared_id",
        question="B",
        stack_name="default",
        result={"answer": "owner B body"},
        created_by_sub="owner-b",
        created_by_tenant_id="default",
    )
    assert other["run_id"] != "run_shared_id"
    assert store.result("run_shared_id")["answer"] == "owner A body"
    assert store.result(other["run_id"])["answer"] == "owner B body"


def test_import_completed_run_rejects_non_terminal_status() -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.import_completed_run(
            run_id="x",
            question="q",
            stack_name="default",
            result={},
            status="running",
        )


def test_queue_full_rejects_extra_waiting_run() -> None:
    release = threading.Event()
    store = _store(max_concurrent=1, max_queue_size=1)
    store.submit(
        question="running",
        stack_name="default",
        work=lambda handle: release.wait(timeout=1),
    )
    store.submit(
        question="queued",
        stack_name="default",
        work=lambda handle: handle.complete({}),
    )

    with pytest.raises(RunQueueFull):
        store.submit(
            question="overflow",
            stack_name="default",
            work=lambda handle: handle.complete({}),
        )

    release.set()


def test_cancel_queued_run_marks_it_terminal() -> None:
    release = threading.Event()
    store = _store(max_concurrent=1, max_queue_size=1)
    store.submit(
        question="running",
        stack_name="default",
        work=lambda handle: release.wait(timeout=1),
    )
    queued = store.submit(
        question="queued",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "never"}),
    )

    cancelled = store.cancel(queued["run_id"])

    assert cancelled["status"] == "cancelled"
    assert cancelled["queue_position"] is None
    assert store._records[queued["run_id"]].status is RunStatus.CANCELLED
    release.set()


def test_cancel_running_run_sets_worker_cancel_event() -> None:
    observed_cancel = threading.Event()

    def cancellable_work(handle: RunHandle) -> None:
        if handle.cancel_event.wait(timeout=1):
            observed_cancel.set()
            handle.cancel("client_requested_cancel")

    store = _store()
    run = store.submit(question="q", stack_name="default", work=cancellable_work)
    _wait_until(lambda: store.get(run["run_id"])["status"] == "running")

    cancelled = store.cancel(run["run_id"])

    assert cancelled["status"] == "running"
    _wait_until(lambda: observed_cancel.is_set())
    _wait_until(lambda: store.get(run["run_id"])["status"] == "cancelled")


def test_completed_result_and_ttl_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monotonic = 1000.0
    monkeypatch.setattr(runs_module.time, "monotonic", lambda: monotonic)
    store = _store(completed_ttl_seconds=5)
    run = store.submit(
        question="q",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "ok", "metrics": {"rounds": 1}}),
    )
    _wait_until(lambda: store.get(run["run_id"])["status"] == "completed")

    result = store.result(run["run_id"])

    assert result["run_id"] == run["run_id"]
    assert result["answer"] == "ok"
    monotonic = 1006.0
    _wait_until(lambda: _raises_not_found(lambda: store.get(run["run_id"])))


def test_subscribe_replays_buffered_events_and_receives_live_events() -> None:
    store = _store()
    run = store.submit(
        question="q",
        stack_name="default",
        work=lambda handle: handle.emit("inqtrix.progress.message", {"message": "ready"}),
    )
    _wait_until(lambda: store.get(run["run_id"])["status"] == "completed")

    subscription = store.subscribe(run["run_id"])
    try:
        event_types = [event["type"] for event in subscription.replay]
        assert "inqtrix.run.queued" in event_types
        assert "inqtrix.progress.message" in event_types

        store.emit(run["run_id"], "inqtrix.debug", {"snapshot": {"round": 1}})
        snapshot_event: dict[str, Any] = subscription.queue.get(timeout=1)
        live_event: dict[str, Any] = subscription.queue.get(timeout=1)
        assert snapshot_event["type"] == "inqtrix.run.snapshot"
        assert snapshot_event["data"]["snapshot"]["round"] == 1
        assert live_event["type"] == "inqtrix.debug"
        assert live_event["data"]["snapshot"]["round"] == 1
        assert store.get(run["run_id"])["snapshot"]["round"] == 1
    finally:
        subscription.close()


def test_format_sse_event_uses_event_type_and_json_payload() -> None:
    frame = format_sse_event(
        {
            "type": "inqtrix.run.started",
            "run_id": "run_1",
            "sequence": 1,
            "data": {"status": "running"},
        }
    )

    assert frame.startswith("event: inqtrix.run.started\n")
    assert '"status": "running"' in frame
    assert frame.endswith("\n\n")


def _raises_not_found(action: Callable[[], object]) -> bool:
    try:
        action()
    except RunNotFound:
        return True
    return False
