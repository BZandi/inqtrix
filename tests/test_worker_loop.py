"""Offline worker-loop tests: claim guards, fencing, ack ordering.

The loop is exercised job-by-job (``_start``/``_execute``) against
recording stubs — the full resolver/registry path runs with the
contract-suite provider stubs and a monkeypatched graph, so the worker
executes EXACTLY the code path the HTTP in-process runner uses.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

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

    def claim_for_execution(self, run_id, tenant_id, *, allow_takeover):
        self.calls.append(("claim", run_id, allow_takeover))
        return self.claim_result

    def emit(self, run_id, event_type, payload=None, *, fence_attempt=None):
        self.calls.append(("emit", event_type))

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


def claimed(payload: dict | None = None) -> ClaimedRun:
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
    import threading

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
    import threading

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
    assert "claim" not in store.names()


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
