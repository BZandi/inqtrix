"""Durable Agent Desk answer publication ordering and fencing."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from inqtrix.agents.control_memory import MemoryAgentControlStore
from inqtrix.agents.control_ports import ArtifactRevisionConflict
from inqtrix.execution_failures import RunExecutionFailure
from inqtrix.server.runs import RunHandle
from inqtrix.services.agent_answer_publisher import AgentAnswerPublisher
from inqtrix.sync_bridge import run_coro_sync
from inqtrix.worker.loop import FencedRunHandle


class _EventStore:
    def __init__(self, control: MemoryAgentControlStore) -> None:
        self.control = control
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.snapshots: list[tuple[str, str, str, int]] = []

    def emit(
        self,
        _run_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        fence_attempt: int | None = None,
    ) -> None:
        del fence_attempt
        self.events.append((event_type, dict(payload)))
        artifact_id = str(payload.get("artifact_id") or "")
        if artifact_id:
            artifact, _revisions = run_coro_sync(
                self.control.get_artifact(_run_id, artifact_id)
            )
            self.snapshots.append(
                (
                    event_type,
                    artifact.status,
                    artifact.content_markdown,
                    artifact.revision,
                )
            )


def _handle(
    control: MemoryAgentControlStore, run_id: str = "run_publication_order"
) -> tuple[RunHandle, _EventStore]:
    events = _EventStore(control)
    return RunHandle(events, run_id, threading.Event()), events  # type: ignore[arg-type]


def test_answer_body_is_hidden_until_final_revision_precedes_ready() -> None:
    control = MemoryAgentControlStore()
    handle, events = _handle(control)
    references = [
        {
            "label": "W2",
            "url": "https://example.test/source",
            "title": "Primärquelle",
            "tier": "primary",
        }
    ]

    AgentAnswerPublisher(control).publish(
        handle,
        "Belastbare Antwort [W2].",
        references=references,
    )

    event_types = [event_type for event_type, _payload in events.events]
    assert event_types[0:2] == [
        "inqtrix.agent.artifact.created",
        "inqtrix.answer.started",
    ]
    assert event_types[-2:] == [
        "inqtrix.agent.artifact.updated",
        "inqtrix.answer.ready",
    ]
    started = next(
        snapshot
        for snapshot in events.snapshots
        if snapshot[0] == "inqtrix.answer.started"
    )
    assert started[1:] == ("writing", "", 1)
    for snapshot in events.snapshots:
        if snapshot[0] == "inqtrix.output_text.delta":
            assert snapshot[1:] == ("writing", "", 1)
    ready = next(
        snapshot
        for snapshot in events.snapshots
        if snapshot[0] == "inqtrix.answer.ready"
    )
    assert ready[1:] == ("ready", "Belastbare Antwort [W2].", 2)

    artifact, _revisions = run_coro_sync(
        control.get_artifact(handle.run_id, events.events[0][1]["artifact_id"])
    )
    assert list(artifact.refs) == references


def test_redelivery_clears_previous_body_before_restreaming() -> None:
    control = MemoryAgentControlStore()
    publisher = AgentAnswerPublisher(control)
    first, _events = _handle(control, "run_publication_retry")
    publisher.publish(first, "Erste Antwort.", references=[])

    retry, retry_events = _handle(control, "run_publication_retry")
    publisher.publish(retry, "Korrigierte Antwort.", references=[])

    assert retry_events.events[0][0] == "inqtrix.agent.artifact.updated"
    started = next(
        snapshot
        for snapshot in retry_events.snapshots
        if snapshot[0] == "inqtrix.answer.started"
    )
    assert started[1:] == ("writing", "", 3)
    artifact, _revisions = run_coro_sync(
        control.get_artifact(
            retry.run_id, retry_events.events[0][1]["artifact_id"]
        )
    )
    assert artifact.status == "ready"
    assert artifact.content_markdown == "Korrigierte Antwort."
    assert artifact.revision == 4


class _FinalizeConflictStore(MemoryAgentControlStore):
    async def upsert_artifact(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        if kwargs.get("kind") == "answer" and kwargs.get("status") == "ready":
            raise ArtifactRevisionConflict(1)
        return await super().upsert_artifact(**kwargs)


def test_finalize_conflict_interrupts_stream_and_never_emits_ready() -> None:
    control = _FinalizeConflictStore()
    handle, events = _handle(control, "run_publication_conflict")

    with pytest.raises(RunExecutionFailure) as failure:
        AgentAnswerPublisher(control).publish(
            handle,
            "Diese Antwort darf nicht vorzeitig sichtbar sein.",
            references=[],
        )

    assert failure.value.error_type == "answer_publication_conflict"
    assert events.events[-1][0] == "inqtrix.answer.interrupted"
    assert events.events[-1][1]["stage"] == "finalizing"
    assert not any(
        event_type == "inqtrix.answer.ready"
        for event_type, _payload in events.events
    )
    artifact, _revisions = run_coro_sync(
        control.get_artifact(
            handle.run_id, events.events[0][1]["artifact_id"]
        )
    )
    assert artifact.status == "writing"
    assert artifact.content_markdown == ""


class _AttemptRecordingStore(MemoryAgentControlStore):
    def __init__(self) -> None:
        super().__init__()
        self.attempts: list[int | None] = []

    async def upsert_artifact(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.attempts.append(kwargs.get("expected_run_attempt"))
        return await super().upsert_artifact(**kwargs)


def test_fenced_worker_attempt_reaches_both_artifact_transactions() -> None:
    control = _AttemptRecordingStore()
    events = _EventStore(control)
    handle = FencedRunHandle(
        events,  # type: ignore[arg-type]
        "run_publication_fenced",
        threading.Event(),
        attempt=7,
    )

    AgentAnswerPublisher(control).publish(handle, "Antwort.", references=[])

    assert control.attempts == [7, 7]
