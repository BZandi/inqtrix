"""Native-run submission contracts that must survive worker replay."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from inqtrix.settings import AgentSettings
from inqtrix.services.run_service import RunService


class _Registry:
    def get(self, _mode: str) -> object:
        return object()


class _Store:
    def __init__(self) -> None:
        self.submission: dict[str, Any] = {}

    def submit(self, **kwargs: Any) -> dict[str, Any]:
        self.submission = kwargs
        return {"run_id": "run-effective-settings"}


def _resolved(*, depth: str) -> SimpleNamespace:
    return SimpleNamespace(
        mode="workspace_agent",
        stack_name="",
        agent_overrides={},
        knowledge_filters={},
        agent_settings=AgentSettings(depth=depth),
    )


def test_agent_submission_persists_effective_depth_for_later_plan_edits() -> None:
    store = _Store()
    service = RunService(
        registry=_Registry(),
        runtime=SimpleNamespace(),
        run_store=store,
    )

    service.submit(
        question="Research",
        history="",
        messages=[],
        resolved=_resolved(depth="deep"),
        workspace_id="workspace-a",
        kind="agent",
    )

    assert store.submission["agent_overrides"]["depth"] == "deep"
    assert (
        store.submission["request_payload"]["body"]["agent_overrides"]
        == {"depth": "deep"}
    )


def test_standard_submission_keeps_explicit_override_summary_shape() -> None:
    store = _Store()
    service = RunService(
        registry=_Registry(),
        runtime=SimpleNamespace(),
        run_store=store,
    )

    service.submit(
        question="Research",
        history="",
        messages=[],
        resolved=_resolved(depth="deep"),
        workspace_id=None,
    )

    assert store.submission["agent_overrides"] == {}


def test_delegated_research_recency_is_persisted_for_worker_replay() -> None:
    store = _Store()
    service = RunService(
        registry=_Registry(),
        runtime=SimpleNamespace(),
        run_store=store,
    )

    service.submit(
        question="Research",
        history="",
        messages=[],
        resolved=_resolved(depth="deep"),
        workspace_id="workspace-a",
        kind="agent_child",
        web_recency="year",
    )

    assert store.submission["request_payload"]["body"]["web_recency"] == "year"
