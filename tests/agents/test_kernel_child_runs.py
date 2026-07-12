"""Kernel child-run tools: slot-free waits + origin_key idempotency (M2-8).

``run_web_research`` submits a REAL child research run, parks the parent
``waiting_for_children`` (no control row — child run rows are the
truth, R5), and the resumed tool re-execution finds its child via
``origin_key`` instead of spawning a second one.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("deepagents")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import inqtrix.research.web_research as web_research_module
from inqtrix.providers.base import ChatTurn, LLMProvider, ToolCallRequest
from inqtrix.server.routes import create_router, register_routes
from inqtrix.settings import (
    AgentPlatformSettings,
    AgentSettings,
    ModelSettings,
    ServerSettings,
    Settings,
    StorageSettings,
)


class ScriptedToolLLM(LLMProvider):
    def __init__(self, turns: list[ChatTurn]) -> None:
        self.models = ModelSettings(
            reasoning_model="base-model",
            tier_high_model="high-model",
            tier_mid_model="mid-model",
            tier_fast_model="fast-model",
        )
        self._turns = list(turns)
        self.chat_calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        raise AssertionError("kernel must use chat()")

    def is_available(self) -> bool:
        return True

    def supports_tool_calls(self, *, model: str | None = None) -> bool:
        return True

    def chat(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> ChatTurn:
        self.chat_calls.append({"messages": list(messages), "tools": tools})
        if not self._turns:
            raise AssertionError("scripted provider ran out of turns")
        return self._turns.pop(0)


def _tool_turn(call_id: str, name: str, arguments: dict) -> ChatTurn:
    return ChatTurn(
        text="",
        tool_calls=(
            ToolCallRequest(id=call_id, name=name, arguments=arguments),
        ),
        finish_reason="tool_calls",
        model="high-model",
        prompt_tokens=10,
        completion_tokens=5,
        raw=None,
    )


def _text_turn(text: str) -> ChatTurn:
    return ChatTurn(
        text=text,
        tool_calls=(),
        finish_reason="stop",
        model="high-model",
        prompt_tokens=10,
        completion_tokens=5,
        raw=None,
    )


def fake_child_graph(question: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "answer": f"Kindbericht zu: {question}",
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        "result_state": {
            "answer": f"Kindbericht zu: {question}",
            "round": 1,
            "report_references": [
                {
                    "label": "E1",
                    "url": "https://example.com/quelle",
                    "title": "Primaerquelle",
                    "tier": "mainstream",
                }
            ],
            "consolidated_claims": [],
        },
    }


def make_client(
    monkeypatch: pytest.MonkeyPatch, llm: ScriptedToolLLM
) -> TestClient:
    monkeypatch.setattr(
        web_research_module, "run_web_graph", fake_child_graph
    )
    app = FastAPI()
    router = create_router()
    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=ServerSettings(),
        storage=StorageSettings(backend="memory", database_url=""),
    )
    settings.agent_platform = AgentPlatformSettings(
        INQTRIX_AGENT_ALLOW_VOLATILE=True,
        INQTRIX_AGENT_KERNEL_ENABLED=True,
    )
    register_routes(
        router,
        providers=SimpleNamespace(llm=llm, search=None),
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(4),
    )
    app.include_router(router)
    return TestClient(app)


def wait_status(
    client: TestClient, run_id: str, statuses: set[str], *, timeout: float = 15.0
) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        summary = client.get(f"/v1/runs/{run_id}").json()
        if summary["status"] in statuses:
            return summary
        time.sleep(0.02)
    pytest.fail(f"run {run_id} never reached {statuses}")


def test_run_web_research_parks_and_returns_child_report(monkeypatch):
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_research1",
                "run_web_research",
                {"question": "Marktlage Klimaanlagen 2026"},
            ),
            _text_turn("Antwort auf Basis des Kindberichts."),
        ]
    )
    client = make_client(monkeypatch, llm)
    with client:
        client.headers["x-inqtrix-workspace-id"] = "ws_kernel_contract"
        response = client.post(
            "/v1/runs",
            json={
                "question": "Analysiere die Marktlage.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "session_id": "sess-child",
                "tool_directives": ["web_research"],
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]

        wait_status(client, run_id, {"completed"})
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        # Exactly ONE child despite the tool re-executing on resume —
        # the origin_key lookup found the submitted run.
        assert len(children) == 1
        child = children[0]
        assert child["kind"] == "agent_child"
        assert child["workspace_id"] == "ws_kernel_contract"
        assert child["origin_key"] == "call_research1"
        assert child["status"] == "completed"
        assert child["agent_overrides"]["report_profile"] == "compact"
        replay = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        projected = [
            event["data"]
            for event in replay
            if event["type"] == "inqtrix.agent.child.progress"
        ]
        assert projected
        assert {event["task_id"] for event in projected} == {
            "call_research1"
        }
        assert {event["attempt"] for event in projected} == {1}
        assert {event["child_run_id"] for event in projected} == {
            child["run_id"]
        }

        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"] == "Antwort auf Basis des Kindberichts."
        assert result["references"][0]["label"] == "W1"
        assert (
            result["references"][0]["url"]
            == "https://example.com/quelle"
        )
        tool_replies = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert "Kindbericht zu: Marktlage Klimaanlagen 2026" in (
            tool_replies[0]["content"]
        )
        assert "Primaerquelle" in tool_replies[0]["content"]


def test_run_web_research_gates_in_balanced(monkeypatch):
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_research2",
                "run_web_research",
                {"question": "Externe Recherche"},
            ),
            _text_turn("Fertig."),
        ]
    )
    client = make_client(monkeypatch, llm)
    with client:
        run_id = client.post(
            "/v1/runs",
            json={
                "question": "Recherchiere.",
                "mode": "agent_kernel",
                "autonomy": "balanced",
                "tool_directives": ["web_research"],
            },
        ).json()["run_id"]
        wait_status(client, run_id, {"waiting_for_approval"})
        # Nothing was submitted before consent.
        assert client.get(f"/v1/runs/{run_id}/children").json()["data"] == []
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        action = approvals[0]["payload"]["actions"][0]
        assert action["tool"] == "run_web_research"
        assert action["args"]["question"] == "Externe Recherche"

        client.post(
            f"/v1/runs/{run_id}/approvals/{approvals[0]['approval_id']}",
            json={"decision": "approve"},
        )
        wait_status(client, run_id, {"completed"})
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 1


def test_failed_child_is_a_visible_tool_result(monkeypatch):
    def broken_child_graph(question: str, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Recherche-Backend nicht erreichbar")

    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_research3",
                "run_web_research",
                {"question": "kaputt"},
            ),
            _text_turn("Ohne Kindbericht beantwortet."),
        ]
    )
    client = make_client(monkeypatch, llm)
    monkeypatch.setattr(
        web_research_module, "run_web_graph", broken_child_graph
    )
    with client:
        run_id = client.post(
            "/v1/runs",
            json={
                "question": "Recherchiere.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "tool_directives": ["web_research"],
            },
        ).json()["run_id"]
        wait_status(client, run_id, {"completed"})
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"] == "Ohne Kindbericht beantwortet."
        tool_replies = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert "fehlgeschlagen" in tool_replies[0]["content"]


def test_normal_kernel_blocks_unrequested_research_child(monkeypatch):
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_unrequested",
                "run_web_research",
                {"question": "Ungefragter Recherchelauf"},
            ),
            _text_turn("Die nicht freigegebene Recherche wurde nicht genutzt."),
        ]
    )
    client = make_client(monkeypatch, llm)
    with client:
        run_id = client.post(
            "/v1/runs",
            json={
                "question": "Beantworte eine einzelne Frage.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
            },
        ).json()["run_id"]
        wait_status(client, run_id, {"completed"})
        assert client.get(f"/v1/runs/{run_id}/children").json()["data"] == []
        tool_replies = [
            message
            for message in llm.chat_calls[1]["messages"]
            if message.get("role") == "tool"
        ]
        assert "nicht ausdruecklich freigegeben" in tool_replies[0]["content"]
