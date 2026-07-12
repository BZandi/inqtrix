"""Kernel walking skeleton: ask_user park/resume end-to-end (M2 step 5).

Drives ``mode=agent_kernel`` through the FULL platform path (router ->
RunService -> worker -> park -> clarification answer -> resume) with a
scripted tool-calling provider — the same trajectory a real user takes,
no seam skipped. Plus the pure interrupt-translation contract.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("deepagents")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.agents.kernel.interrupts import (
    CLARIFICATION_INTERRUPT,
    TOOL_APPROVAL_INTERRUPT,
    ask_user_clarification_id,
    translate_kernel_interrupt,
)
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
    """Sequential ChatTurn script with full call recording."""

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
        raise AssertionError("kernel must use chat(), not complete()")

    def is_available(self) -> bool:
        return True

    def supports_tool_calls(self, *, model: str | None = None) -> bool:
        return True

    def chat(
        self, messages: Any, *, tools: Any = None, model: Any = None, **kwargs: Any
    ) -> ChatTurn:
        self.chat_calls.append(
            {"messages": list(messages), "tools": tools, "model": model}
        )
        if not self._turns:
            raise AssertionError("scripted provider ran out of turns")
        return self._turns.pop(0)


class RetryNotifyingToolLLM(ScriptedToolLLM):
    """Kernel provider that exposes one retry through the shared seam."""

    def __init__(self, turns: list[ChatTurn]) -> None:
        super().__init__(turns)
        self._retry_callback: Any = None
        self._retry_emitted = False

    @contextmanager
    def observe_retries(self, callback: Any) -> Any:
        previous = self._retry_callback
        self._retry_callback = callback
        try:
            yield self
        finally:
            self._retry_callback = previous

    def chat(
        self, messages: Any, *, tools: Any = None, model: Any = None, **kwargs: Any
    ) -> ChatTurn:
        if not self._retry_emitted and self._retry_callback is not None:
            self._retry_emitted = True
            self._retry_callback({
                "attempt": 1,
                "max_attempts": 3,
                "delay_seconds": 0.5,
                "error_code": "rate_limited",
                "operation": "chat",
            })
        return super().chat(
            messages,
            tools=tools,
            model=model,
            **kwargs,
        )


def _ask_turn() -> ChatTurn:
    return ChatTurn(
        text="",
        tool_calls=(
            ToolCallRequest(
                id="call_ask0001",
                name="ask_user",
                arguments={
                    "question": "Welches Format soll die Antwort haben?",
                    "options": ["Kompakt", "Ausfuehrlich", "Stichpunkte"],
                    "default_assumption": "Kompakt",
                },
            ),
        ),
        finish_reason="tool_calls",
        model="high-model",
        prompt_tokens=40,
        completion_tokens=12,
        raw=None,
    )


def _answer_turn() -> ChatTurn:
    return ChatTurn(
        text="Hier ist die kompakte Antwort auf deinen Auftrag.",
        tool_calls=(),
        finish_reason="stop",
        model="high-model",
        prompt_tokens=60,
        completion_tokens=25,
        raw=None,
    )


def make_kernel_client(
    llm: ScriptedToolLLM,
    *,
    max_tokens_per_run: int = 0,
    max_tool_calls: int = 30,
) -> TestClient:
    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=ServerSettings(),
        storage=StorageSettings(backend="memory", database_url=""),
    )
    settings.agent_platform = AgentPlatformSettings(
        INQTRIX_AGENT_ALLOW_VOLATILE=True,
        INQTRIX_AGENT_KERNEL_ENABLED=True,
        INQTRIX_AGENT_KERNEL_MAX_TOOL_CALLS=max_tool_calls,
    )
    if max_tokens_per_run:
        settings.quota.max_tokens_per_run = max_tokens_per_run
    return _client_for(settings, llm)


def _client_for(settings: Settings, llm: ScriptedToolLLM) -> TestClient:
    app = FastAPI()
    router = create_router()
    register_routes(
        router,
        providers=SimpleNamespace(llm=llm, search=None),
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(2),
    )
    app.include_router(router)
    client = TestClient(app)
    client.llm = llm  # type: ignore[attr-defined]
    return client


def wait_status(
    client: TestClient, run_id: str, statuses: set[str], *, timeout: float = 10.0
) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        summary = client.get(f"/v1/runs/{run_id}").json()
        if summary["status"] in statuses:
            return summary
        time.sleep(0.02)
    pytest.fail(f"run {run_id} never reached {statuses}")


def run_events(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    with client.stream("GET", f"/v1/runs/{run_id}/events") as stream:
        body = stream.read().decode("utf-8")
    return [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_default_mode_flip_requires_registered_kernel():
    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(DEPTH="deep"),
        server=ServerSettings(),
        storage=StorageSettings(backend="memory", database_url=""),
    )
    settings.agent_platform = AgentPlatformSettings(
        INQTRIX_AGENT_ALLOW_VOLATILE=True,
        INQTRIX_AGENT_KERNEL_ENABLED=True,
        INQTRIX_AGENT_DEFAULT_MODE="agent_kernel",
    )
    with _client_for(settings, ScriptedToolLLM([])) as client:
        agent = client.get("/v1/capabilities").json()["agent"]
        assert agent["default_mode"] == "agent_kernel"
        assert agent["default_depth"] == "deep"

    # Kernel configured as default but NOT registered (rollout switch
    # off): capabilities fall back so the desk never submits a 400 mode.
    settings2 = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=ServerSettings(),
        storage=StorageSettings(backend="memory", database_url=""),
    )
    settings2.agent_platform = AgentPlatformSettings(
        INQTRIX_AGENT_ALLOW_VOLATILE=True,
        INQTRIX_AGENT_DEFAULT_MODE="agent_kernel",
    )
    with _client_for(settings2, ScriptedToolLLM([])) as client:
        agent = client.get("/v1/capabilities").json()["agent"]
        assert agent["default_mode"] == "workspace_agent"
        assert agent["default_depth"] == "normal"


def test_kernel_model_retries_use_agent_activity_channel():
    llm = RetryNotifyingToolLLM([_answer_turn()])
    client = make_kernel_client(llm)
    with client:
        response = client.post(
            "/v1/runs",
            json={"question": "Fasse den EU AI Act zusammen.", "mode": "agent_kernel"},
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"completed"})
        retries = [
            event["data"]
            for event in run_events(client, run_id)
            if event["type"] == "inqtrix.agent.activity"
            and event["data"].get("retry")
        ]

    assert len(retries) == 1
    assert retries[0]["scope"] == "run"
    assert retries[0]["phase"] == "execution"
    assert retries[0]["purpose"] == "Agent-Antwort wird fortgesetzt"
    assert retries[0]["retry"]["max_attempts"] == 3


def test_ask_user_park_resume_trajectory():
    llm = ScriptedToolLLM([_ask_turn(), _answer_turn()])
    client = make_kernel_client(llm)
    with client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["features"]["agent_kernel"] is True
        # The kernel is the Agent Desk front door once its independent
        # registration gate passes.
        assert capabilities["agent"]["default_mode"] == "agent_kernel"

        response = client.post(
            "/v1/runs",
            json={"question": "Fasse den EU AI Act zusammen.", "mode": "agent_kernel"},
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]

        wait_status(client, run_id, {"waiting_for_input"})
        rows = client.get(f"/v1/runs/{run_id}/clarifications").json()["data"]
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == "pending"
        # Deterministic id derived from run + frozen tool_call_id.
        assert row["clarification_id"] == ask_user_clarification_id(
            run_id, "call_ask0001"
        )
        assert row["question"] == "Welches Format soll die Antwort haben?"
        assert [o["label"] for o in row["questions"][0]["options"]] == [
            "Kompakt",
            "Ausfuehrlich",
            "Stichpunkte",
        ]
        assert row["default_assumption"] == "Kompakt"

        answered = client.post(
            f"/v1/runs/{run_id}/clarifications/{row['clarification_id']}",
            json={"option_id": "q1_o2"},
        )
        assert answered.status_code == 200, answered.text

        wait_status(client, run_id, {"completed"})
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"] == (
            "Hier ist die kompakte Antwort auf deinen Auftrag."
        )

        # The resumed segment saw the user's answer as the tool result.
        assert len(llm.chat_calls) == 2
        resumed_messages = llm.chat_calls[1]["messages"]
        tool_replies = [
            m for m in resumed_messages if m.get("role") == "tool"
        ]
        assert len(tool_replies) == 1
        assert "Antwort des Nutzers: Ausfuehrlich" in tool_replies[0]["content"]
        # The ask_user tool schema went to the provider on both calls.
        tool_names = {
            t["function"]["name"] for t in (llm.chat_calls[0]["tools"] or [])
        }
        assert "ask_user" in tool_names


def test_two_ask_user_rounds_use_distinct_rows():
    """Regression: prefix-sharing tool_call_ids must not collide.

    Round 2 reusing round 1's ANSWERED row would skip the create, park
    the run as waiting_for_input with nothing pending, and strand it.
    """
    second_ask = ChatTurn(
        text="",
        tool_calls=(
            ToolCallRequest(
                id="call_ask0002",
                name="ask_user",
                arguments={
                    "question": "Welche Sprache?",
                    "options": ["Deutsch", "Englisch"],
                    "default_assumption": "Deutsch",
                },
            ),
        ),
        finish_reason="tool_calls",
        model="high-model",
        prompt_tokens=20,
        completion_tokens=8,
        raw=None,
    )
    llm = ScriptedToolLLM([_ask_turn(), second_ask, _answer_turn()])
    client = make_kernel_client(llm)
    with client:
        run_id = client.post(
            "/v1/runs", json={"question": "Auftrag.", "mode": "agent_kernel"}
        ).json()["run_id"]

        wait_status(client, run_id, {"waiting_for_input"})
        first_rows = client.get(
            f"/v1/runs/{run_id}/clarifications"
        ).json()["data"]
        client.post(
            f"/v1/runs/{run_id}/clarifications/"
            f"{first_rows[0]['clarification_id']}",
            json={"answer": "Kompakt."},
        )

        wait_status(client, run_id, {"waiting_for_input"})
        rows = client.get(f"/v1/runs/{run_id}/clarifications").json()["data"]
        assert len(rows) == 2
        pending = [r for r in rows if r["status"] == "pending"]
        assert len(pending) == 1
        assert pending[0]["question"] == "Welche Sprache?"
        client.post(
            f"/v1/runs/{run_id}/clarifications/"
            f"{pending[0]['clarification_id']}",
            json={"answer": "Deutsch."},
        )

        wait_status(client, run_id, {"completed"})


def test_token_budget_stops_the_kernel_across_segments():
    """The per-run cap counts checkpointed spend from earlier segments.

    Segment 1 books 52 tokens (over the 30-token cap); the resumed
    segment's first model call must abort BEFORE reaching the provider and
    terminate with the distinct token-budget failure type.
    """
    llm = ScriptedToolLLM([_ask_turn(), _answer_turn()])
    client = make_kernel_client(llm, max_tokens_per_run=30)
    with client:
        run_id = client.post(
            "/v1/runs", json={"question": "Auftrag.", "mode": "agent_kernel"}
        ).json()["run_id"]
        wait_status(client, run_id, {"waiting_for_input"})
        rows = client.get(f"/v1/runs/{run_id}/clarifications").json()["data"]
        client.post(
            f"/v1/runs/{run_id}/clarifications/"
            f"{rows[0]['clarification_id']}",
            json={"answer": "Egal."},
        )
        summary = wait_status(client, run_id, {"failed", "completed"})
        assert summary["status"] == "failed"
        assert summary["error"]["type"] == "token_budget_exceeded"
        # The second scripted turn was never consumed: the abort fired
        # at the model boundary, not after another paid call.
        assert len(llm.chat_calls) == 1


def test_kernel_disabled_by_default_is_unknown_mode():
    llm = ScriptedToolLLM([])
    app = FastAPI()
    router = create_router()
    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=ServerSettings(),
        storage=StorageSettings(backend="memory", database_url=""),
    )
    settings.agent_platform = AgentPlatformSettings(
        INQTRIX_AGENT_ALLOW_VOLATILE=True
    )
    register_routes(
        router,
        providers=SimpleNamespace(llm=llm, search=None),
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(2),
    )
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/runs", json={"question": "x", "mode": "agent_kernel"}
        )
        assert response.status_code == 400
        features = client.get("/v1/capabilities").json()["features"]
        assert features["agent_kernel"] is False


def test_kernel_gate_warns_without_tool_calls(caplog):
    class NoToolsLLM(ScriptedToolLLM):
        def supports_tool_calls(self, *, model: str | None = None) -> bool:
            return False

    llm = NoToolsLLM([])
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
    with caplog.at_level("WARNING", logger="inqtrix"):
        register_routes(
            router,
            providers=SimpleNamespace(llm=llm, search=None),
            strategies=SimpleNamespace(),
            settings=settings,
            semaphore_factory=lambda: __import__("asyncio").Semaphore(2),
        )
    app.include_router(router)
    assert any(
        "Agent-Kernel" in record.message for record in caplog.records
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/runs", json={"question": "x", "mode": "agent_kernel"}
        )
        assert response.status_code == 400


def test_ask_user_discards_empty_question_without_parking():
    """A whitespace question must not park the run on an empty prompt."""
    from inqtrix.agents.control_memory import MemoryAgentControlStore
    from inqtrix.agents.kernel.deps import (
        KernelDeps,
        run_coro,
        set_kernel_deps,
    )
    from inqtrix.agents.kernel.tools import build_kernel_tools

    store = MemoryAgentControlStore()
    set_kernel_deps(
        KernelDeps(
            run_id="run_guard",
            control=store,
            platform=None,  # type: ignore[arg-type]
            llm=None,  # type: ignore[arg-type]
            model=None,
            reasoning_effort=None,
            timeout=1.0,
        )
    )
    try:
        ask_user = build_kernel_tools()[0]
        message = ask_user.invoke(
            {
                "name": "ask_user",
                "type": "tool_call",
                "id": "call_guard1",
                "args": {
                    "question": "   ",
                    "options": [],
                    "default_assumption": "Kompakt",
                },
            }
        )
    finally:
        set_kernel_deps(None)
    result = str(message.content)
    assert "verworfen" in result
    assert "Kompakt" in result
    assert run_coro(store.list_clarifications("run_guard")) == []


def test_translate_kernel_interrupt_discriminates_origins():
    origin, payload = translate_kernel_interrupt(
        {"kind": "clarification", "id": "clr_abc"}
    )
    assert (origin, payload) == (CLARIFICATION_INTERRUPT, {"id": "clr_abc"})

    origin, payload = translate_kernel_interrupt(
        {
            "action_requests": [{"name": "web_instant", "args": {}}],
            "review_configs": [],
        }
    )
    assert origin == TOOL_APPROVAL_INTERRUPT
    assert payload["action_requests"][0]["name"] == "web_instant"

    with pytest.raises(ValueError):
        translate_kernel_interrupt({"kind": "unbekannt"})
    with pytest.raises(ValueError):
        translate_kernel_interrupt("garbage")
    # An actionless HITL payload has nothing to approve — routing it
    # would park a run no decision can wake.
    with pytest.raises(ValueError):
        translate_kernel_interrupt(
            {"action_requests": [], "review_configs": []}
        )


def test_ask_user_clarification_id_is_deterministic_and_collision_safe():
    first = ask_user_clarification_id("run_1234567890ab", "call_abcdef12")
    assert first == ask_user_clarification_id(
        "run_1234567890ab", "call_abcdef12"
    )
    # Provider ids share long constant prefixes (call_..., toolu_01...);
    # the id must still differ per call — a collision reuses round 1's
    # answered row and strands round 2 parked with nothing pending.
    prefix_shared = [
        ask_user_clarification_id("run_1234567890ab", tool_call_id)
        for tool_call_id in (
            "toolu_01Aaaaaaaaaaaaaaaaaaaaaa",
            "toolu_01Bbbbbbbbbbbbbbbbbbbbbb",
            "call_abc111111111111111111",
            "call_abc222222222222222222",
        )
    ]
    assert len(set(prefix_shared)) == len(prefix_shared)
