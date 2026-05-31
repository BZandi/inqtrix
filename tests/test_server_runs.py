"""Tests for the native in-memory run queue."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import pytest

import inqtrix.server.runs as runs_module
from inqtrix.server.runs import (
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
