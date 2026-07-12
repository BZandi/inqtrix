"""Offline worker-loop tests: claim guards, fencing, ack ordering.

The loop is exercised job-by-job (``_start``/``_execute``) against
recording stubs — the full resolver/registry path runs with the
contract-suite provider stubs and a monkeypatched graph, so the worker
executes EXACTLY the code path the HTTP in-process runner uses.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any

import pytest

from inqtrix.exceptions import (
    AgentRateLimited,
    AgentTimeout,
    AzureOpenAIAPIError,
)
from inqtrix.providers.base import ProviderContext
from inqtrix.runs.postgres_store import ClaimedRun
from inqtrix.runs.valkey_queue import QueuedJob
from inqtrix.server.container import build_container
from inqtrix.settings import Settings
from inqtrix.worker.loop import FencedRunHandle, WorkerLoop

from tests.contract._app import StubLLM, StubSearch, minimal_agent_result


class StubStore:
    """Recording stand-in for the worker-facing store surface."""

    def __init__(
        self,
        *,
        claim_result: ClaimedRun | None,
        timeline: list[str] | None = None,
    ) -> None:
        self.claim_result = claim_result
        self.calls: list[tuple[str, Any]] = []
        self.timeline = timeline if timeline is not None else []
        self.worker_id = "worker-test"
        self.dispatch_status_value: str | None = "running"
        self.dispatch_status_error: BaseException | None = None

    def claim_for_execution(self, run_id, tenant_id, *, allow_takeover):
        self.calls.append(("claim", run_id, allow_takeover))
        return self.claim_result

    def emit(self, run_id, event_type, payload=None, *, fence_attempt=None):
        self.calls.append(("emit", event_type, payload, fence_attempt))

    terminal_lands = True
    """Scripted outcome of terminal writes — ``False`` simulates a
    fenced-out zombie attempt."""

    def complete(self, run_id, result, *, snapshot=None, fence_attempt=None):
        self.calls.append(("complete", run_id, fence_attempt))
        self.timeline.append("complete")
        return self.terminal_lands

    def fail(self, run_id, message, *, error_type="server_error", fence_attempt=None):
        self.calls.append(("fail", run_id, message, error_type, fence_attempt))
        return self.terminal_lands

    def mark_cancelled(self, run_id, *, reason, fence_attempt=None):
        self.calls.append(("cancelled", run_id, reason, fence_attempt))
        return self.terminal_lands

    def cancel_requested_runs(self, run_ids):
        return set()

    def stale_queued_runs(self, *, older_than_seconds):
        return []

    def dispatch_status(self, run_id, tenant_id):
        self.calls.append(("dispatch_status", run_id, tenant_id))
        if self.dispatch_status_error is not None:
            raise self.dispatch_status_error
        return self.dispatch_status_value

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


class StubQueue:
    """Recording stand-in for the queue surface the loop touches."""

    def __init__(self, timeline: list[str] | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.timeline = timeline if timeline is not None else []

    def ack(self, message_id):
        self.calls.append(("ack", message_id))
        self.timeline.append("ack")

    def dead_letter(self, job, *, reason):
        self.calls.append(("dead_letter", job.run_id, reason))

    def enqueue(self, *, run_id, tenant_id):
        self.calls.append(("enqueue", run_id))

    def heartbeat(self, message_ids):
        self.calls.append(("heartbeat", tuple(message_ids)))


def make_loop(store, queue, monkeypatch) -> WorkerLoop:
    def fake_graph(question, **kwargs):
        return minimal_agent_result()

    monkeypatch.setattr(
        "inqtrix.research.web_research.run_web_graph", fake_graph
    )
    container = build_container(
        providers=ProviderContext(llm=StubLLM(), search=StubSearch()),
        strategies=None,
        settings=Settings(),
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    return WorkerLoop(
        store=store,
        queue=queue,
        resolver=container.resolver,
        registry=container.registry,
        runtime=container.runtime,
        concurrency=1,
        max_attempts=3,
        heartbeat_seconds=15,
        claim_idle_seconds=90,
    )


def job(delivery_count: int = 1) -> QueuedJob:
    return QueuedJob(
        message_id="1-0",
        run_id="run_w1",
        tenant_id="default",
        delivery_count=delivery_count,
    )


def successor_job(message_id: str = "2-0") -> QueuedJob:
    return QueuedJob(
        message_id=message_id,
        run_id="run_w1",
        tenant_id="default",
        delivery_count=1,
    )


def claimed(
    payload: dict | None = None,
    *,
    workspace_id: str | None = None,
    kind: str = "standard",
) -> ClaimedRun:
    return ClaimedRun(
        run_id="run_w1",
        tenant_id="default",
        attempt=1,
        request_payload=(
            payload
            if payload is not None
            else {
                "question": "Wie ist die Haftung geregelt?",
                "history": "",
                "messages": [],
                "body": {
                    "mode": "research",
                    "agent_overrides": {},
                    "knowledge_filters": {},
                },
            }
        ),
        workspace_id=workspace_id,
        kind=kind,
    )


def run_one(store, queue, monkeypatch, *, the_job=None) -> WorkerLoop:
    loop = make_loop(store, queue, monkeypatch)
    loop._start(the_job or job(), takeover=False)
    assert loop.drain(timeout=10), "job did not finish in time"
    return loop


def test_successful_execution_completes_then_acks(monkeypatch):
    timeline: list[str] = []
    store = StubStore(claim_result=claimed(), timeline=timeline)
    queue = StubQueue(timeline=timeline)

    run_one(store, queue, monkeypatch)

    assert ("claim", "run_w1", False) in store.calls
    assert ("complete", "run_w1", 1) in store.calls
    assert queue.calls == [("ack", "1-0")]
    # Terminal write precedes the ack (at-least-once ordering) — the
    # shared timeline makes the cross-object order assertable.
    assert timeline == ["complete", "ack"]


@pytest.mark.parametrize(
    ("exc", "error_type"),
    [
        (
            AgentRateLimited("model-a", RuntimeError("429")),
            "rate_limited",
        ),
        (AgentTimeout("deadline"), "run_timeout"),
        (ConnectionError("reset"), "temporary_transport"),
        (
            AzureOpenAIAPIError(
                model="deployment", status_code=503, message="unavailable"
            ),
            "upstream_5xx",
        ),
    ],
    ids=["rate-limit", "timeout", "transport", "upstream-5xx"],
)
def test_worker_persists_stable_execution_failure_type(
    monkeypatch, exc, error_type
):
    def fail_execution(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(
        "inqtrix.worker.loop.execute_run_request", fail_execution
    )
    store = StubStore(claim_result=claimed())
    queue = StubQueue()

    run_one(store, queue, monkeypatch)

    failure = next(call for call in store.calls if call[0] == "fail")
    assert failure[3] == error_type
    assert queue.calls == [("ack", "1-0")]


def test_worker_rehydrates_source_policy_and_execution_directive(monkeypatch):
    """Durable replay reconstructs the exact admitted Agent Desk policy."""
    captured = []

    def fake_execute(handle, *, run_request, **kwargs):
        captured.append((run_request, kwargs))
        handle.complete(minimal_agent_result())

    monkeypatch.setattr(
        "inqtrix.worker.loop.execute_run_request", fake_execute
    )
    store = StubStore(
        claim_result=claimed(
            payload={
                "question": "Nur diese Nachricht im Web suchen.",
                "history": "",
                "messages": [],
                "body": {
                    "mode": "research",
                    "agent_overrides": {},
                    "knowledge_filters": {},
                    "source_policy": {
                        "web": "disabled",
                        "knowledge": "available",
                    },
                    "web_recency": "year",
                    "execution_directive": "quick_web",
                },
            },
            workspace_id="ws-agent",
        )
    )
    queue = StubQueue()

    run_one(store, queue, monkeypatch)

    assert len(captured) == 1
    assert captured[0][0].source_policy.model_dump() == {
        "web": "disabled",
        "knowledge": "available",
    }
    assert captured[0][0].execution_directive == "quick_web"
    assert captured[0][0].web_recency == "year"
    assert captured[0][1]["workspace_id"] == "ws-agent"


def test_worker_ignores_legacy_child_budget_and_emits_stable_notice(
    monkeypatch,
):
    """Pre-migration child caps never narrow the operator run limit."""
    captured: list[dict[str, Any]] = []

    def fake_execute(handle, **kwargs):
        captured.append(kwargs)
        handle.complete(minimal_agent_result())

    monkeypatch.setattr(
        "inqtrix.worker.loop.execute_run_request", fake_execute
    )
    store = StubStore(
        claim_result=claimed(
            payload={
                "question": "Legacy child",
                "history": "",
                "messages": [],
                "body": {
                    "mode": "research",
                    "agent_overrides": {},
                    "knowledge_filters": {},
                    "token_budget": 1800,
                    "parent_task_id": "task-legacy",
                },
            },
            kind="agent_child",
        )
    )
    queue = StubQueue()

    run_one(store, queue, monkeypatch)

    assert captured[0]["token_budget"] is None
    notices = [
        call
        for call in store.calls
        if call[0] == "emit"
        and call[1] == "inqtrix.agent.activity"
    ]
    assert len(notices) == 1
    assert notices[0][2] == {
        "activity_id": "legacy-child-budget:run_w1",
        "scope": "task",
        "phase": "execution",
        "operation": "task.legacy_budget_ignored",
        "detail": "Veraltetes Task-Budget wird ignoriert",
        "status": "completed",
        "task_id": "task-legacy",
        "fallback": True,
    }


def test_unclaimable_job_is_acked_and_skipped(monkeypatch):
    store = StubStore(claim_result=None)
    queue = StubQueue()

    run_one(store, queue, monkeypatch)

    assert store.names() == ["claim"]
    assert queue.calls == [("ack", "1-0")]


def test_exhausted_delivery_budget_dead_letters_and_fails(monkeypatch):
    store = StubStore(claim_result=claimed())
    queue = StubQueue()

    run_one(store, queue, monkeypatch, the_job=job(delivery_count=4))

    fails = [call for call in store.calls if call[0] == "fail"]
    assert fails and fails[0][3] == "max_retries_exceeded"
    assert ("dead_letter", "run_w1", "max_attempts_exceeded") in queue.calls
    assert "claim" not in store.names()


def test_empty_request_payload_fails_loudly(monkeypatch):
    store = StubStore(claim_result=claimed(payload={}))
    queue = StubQueue()

    run_one(store, queue, monkeypatch)

    fails = [call for call in store.calls if call[0] == "fail"]
    assert fails and "request_payload" in fails[0][2]
    assert fails[0][4] == 1
    assert ("ack", "1-0") in queue.calls


def test_redelivered_job_claims_with_takeover(monkeypatch):
    store = StubStore(claim_result=claimed())
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)

    loop._start(job(delivery_count=2), takeover=True)
    assert loop.drain(timeout=10)

    assert ("claim", "run_w1", True) in store.calls


def test_fenced_handle_threads_the_attempt_into_terminal_writes():
    store = StubStore(claim_result=None)
    import threading

    handle = FencedRunHandle(store, "run_w1", threading.Event(), attempt=7)
    handle.complete({"answer": "x"})
    handle.fail("kaputt")
    handle.cancel("client_requested_cancel")

    assert ("complete", "run_w1", 7) in store.calls
    assert any(
        call[0] == "fail" and call[4] == 7 for call in store.calls
    )
    assert ("cancelled", "run_w1", "client_requested_cancel", 7) in store.calls


@pytest.mark.parametrize("stale", [[("run_s1", "default")], []])
def test_reconciler_re_enqueues_stale_queued_rows(monkeypatch, stale):
    store = StubStore(claim_result=None)
    store.stale_queued_runs = lambda *, older_than_seconds: stale
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)

    loop._reconcile()

    expected = [("enqueue", run_id) for run_id, _tenant in stale]
    assert queue.calls == expected


def test_reconciler_cooldown_suppresses_duplicate_floods(monkeypatch):
    store = StubStore(claim_result=None)
    store.stale_queued_runs = lambda *, older_than_seconds: [
        ("run_s1", "default")
    ]
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)

    loop._reconcile()
    loop._reconcile()

    assert queue.calls == [("enqueue", "run_s1")]


def test_self_reclaim_of_own_entry_is_never_acked(monkeypatch):
    """XAUTOCLAIM can hand a worker its OWN in-flight entry back;
    acking it would destroy the run's only crash-recovery entry."""
    store = StubStore(claim_result=claimed())
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)
    from inqtrix.worker.loop import _ActiveJob

    with loop._lock:
        loop._active["run_w1"] = _ActiveJob(
            job=job(), cancel_event=threading.Event()
        )

    loop._start(job(), takeover=True)  # same message id "1-0"

    assert queue.calls == []
    assert "claim" not in store.names()


def test_duplicate_dispatch_for_active_run_is_acked_not_dropped(monkeypatch):
    """A silently dropped duplicate would idle in the PEL and mature
    into a takeover of the healthy execution — it must be acked."""
    store = StubStore(claim_result=claimed())
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)
    from inqtrix.worker.loop import _ActiveJob

    with loop._lock:
        loop._active["run_w1"] = _ActiveJob(
            job=job(), cancel_event=threading.Event()
        )

    duplicate = QueuedJob(
        message_id="2-0",
        run_id="run_w1",
        tenant_id="default",
        delivery_count=1,
    )
    loop._start(duplicate, takeover=False)

    assert queue.calls == [("ack", "2-0")]
    assert ("dispatch_status", "run_w1", "default") in store.calls
    assert "claim" not in store.names()


def test_unsegmented_worker_keeps_historical_duplicate_ack(monkeypatch):
    """The Base default used by reindex never invents successor semantics."""
    from inqtrix.worker.loop import BaseWorkerLoop, _ActiveJob

    class _UnsegmentedLoop(BaseWorkerLoop[QueuedJob, ClaimedRun]):
        def _entity_id(self, queued: QueuedJob) -> str:
            return queued.run_id

        def _execute(self, queued, claimed_run, cancel_event) -> None:
            del queued, claimed_run, cancel_event

        def _stale_dispatch(self):
            return []

        def _cancel_requested(self, watched):
            del watched
            return set()

        def _enqueue_dispatch(self, entity_id, tenant_id) -> None:
            del entity_id, tenant_id

    store = StubStore(claim_result=claimed())
    queue = StubQueue()
    loop = _UnsegmentedLoop(
        store=store,
        queue=queue,
        concurrency=1,
        max_attempts=3,
        heartbeat_seconds=15,
        claim_idle_seconds=90,
    )
    with loop._lock:
        loop._active["run_w1"] = _ActiveJob(
            job=job(), cancel_event=threading.Event()
        )

    loop._start(successor_job(), takeover=False)

    assert queue.calls == [("ack", "2-0")]
    assert "dispatch_status" not in store.names()


def test_queued_successor_is_held_hearted_and_claimed_after_old_ack(
    monkeypatch,
):
    """A wake racing the old segment's unwind is never ACKed as a duplicate."""
    store = StubStore(claim_result=claimed())
    store.dispatch_status_value = "queued"
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)
    from inqtrix.worker.loop import _ActiveJob

    old = job()
    successor = successor_job()
    with loop._lock:
        loop._active[old.run_id] = _ActiveJob(
            job=old, cancel_event=threading.Event()
        )

    loop._start(successor, takeover=False)

    assert queue.calls == []
    assert loop._heartbeat_message_ids() == ["1-0", "2-0"]
    with loop._lock:
        assert loop._active[old.run_id].successor is successor

    submitted: list[tuple[Any, ...]] = []

    def _submit(*args):
        submitted.append(args)
        return Future()

    monkeypatch.setattr(loop._executor, "submit", _submit)
    loop._finish_active(old, allow_successor=True)

    assert ("claim", "run_w1", False) in store.calls
    assert len(submitted) == 1
    with loop._lock:
        active = loop._active[old.run_id]
        assert active.job is successor
        assert active.successor is None
        assert active.handoff_in_progress is False
    # The placeholder remains capacity/drain-visible until the successor's
    # executor finally releases it.
    assert loop.drain(timeout=0) is False


def test_second_successor_duplicate_is_acked_while_first_is_held(monkeypatch):
    store = StubStore(claim_result=claimed())
    store.dispatch_status_value = "queued"
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)
    from inqtrix.worker.loop import _ActiveJob

    with loop._lock:
        loop._active["run_w1"] = _ActiveJob(
            job=job(), cancel_event=threading.Event()
        )
    loop._start(successor_job("2-0"), takeover=False)
    loop._start(successor_job("3-0"), takeover=False)

    assert queue.calls == [("ack", "3-0")]
    assert store.names().count("dispatch_status") == 1


def test_successor_status_read_error_never_degrades_to_ack(monkeypatch):
    store = StubStore(claim_result=claimed())
    store.dispatch_status_error = RuntimeError("database unavailable")
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)
    from inqtrix.worker.loop import _ActiveJob

    with loop._lock:
        loop._active["run_w1"] = _ActiveJob(
            job=job(), cancel_event=threading.Event()
        )

    with pytest.raises(RuntimeError, match="database unavailable"):
        loop._start(successor_job(), takeover=False)

    assert queue.calls == []
    with loop._lock:
        assert loop._active["run_w1"].successor is None


def test_successor_stays_unacked_when_old_message_ack_failed(monkeypatch):
    store = StubStore(claim_result=claimed())
    store.dispatch_status_value = "queued"
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)
    from inqtrix.worker.loop import _ActiveJob

    old = job()
    successor = successor_job()
    with loop._lock:
        loop._active["run_w1"] = _ActiveJob(
            job=old,
            cancel_event=threading.Event(),
            successor=successor,
        )

    loop._finish_active(old, allow_successor=False)

    assert queue.calls == []
    assert "claim" not in store.names()
    with loop._lock:
        assert "run_w1" not in loop._active


def test_cancelled_successor_is_acked_without_execution(monkeypatch):
    store = StubStore(claim_result=None)
    store.dispatch_status_value = "queued"
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)
    from inqtrix.worker.loop import _ActiveJob

    old = job()
    successor = successor_job()
    with loop._lock:
        loop._active["run_w1"] = _ActiveJob(
            job=old,
            cancel_event=threading.Event(),
            successor=successor,
        )

    loop._finish_active(old, allow_successor=True)

    assert queue.calls == [("ack", "2-0")]
    assert ("claim", "run_w1", False) in store.calls
    with loop._lock:
        assert "run_w1" not in loop._active


def test_fast_successor_can_park_again_before_executor_submit_returns(
    monkeypatch,
):
    """A third segment is not mistaken for a duplicate of its predecessor."""
    store = StubStore(claim_result=claimed())
    store.dispatch_status_value = "queued"
    queue = StubQueue()
    loop = make_loop(store, queue, monkeypatch)
    from inqtrix.worker.loop import _ActiveJob

    old = job()
    second = successor_job("2-0")
    third = successor_job("3-0")
    with loop._lock:
        loop._active["run_w1"] = _ActiveJob(
            job=old,
            cancel_event=threading.Event(),
            successor=second,
        )

    def _submit(*_args):
        # ThreadPoolExecutor may run the submitted callable before submit()
        # returns. Model that exact interleaving: segment 2 already parked and
        # its row is queued when dispatch 3 reaches the claim loop.
        loop._start(third, takeover=False)
        return Future()

    monkeypatch.setattr(loop._executor, "submit", _submit)
    loop._finish_active(old, allow_successor=True)

    assert queue.calls == []
    with loop._lock:
        active = loop._active["run_w1"]
        assert active.job is second
        assert active.successor is third
        assert active.handoff_in_progress is False
    assert loop._heartbeat_message_ids() == ["2-0", "3-0"]


def test_fenced_out_attempt_does_not_ack_the_message(monkeypatch):
    """When a superseding worker owns the run (and the message), the
    zombie's ack would strip the new owner's crash-recovery entry."""
    store = StubStore(claim_result=claimed())
    store.terminal_lands = False
    queue = StubQueue()

    loop = make_loop(store, queue, monkeypatch)
    loop._start(job(), takeover=False)
    assert loop.drain(timeout=10)

    assert ("complete", "run_w1", 1) in store.calls
    assert ("ack", "1-0") not in queue.calls
