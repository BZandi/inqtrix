"""Kernel system prompt, session continuity, and follow events (M2-9).

The prompt drift test is a tripwire: the rendering SSOT and every
registered tool must be covered by the prompt's discipline rules. The
trajectory tests secure K1 continuity (a follow-up turn SEES the prior
answer) and the follow-the-agent event stream.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("deepagents")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.agents.prompts import (
    build_agent_kernel_system_prompt,
    build_kernel_user_message,
    rendering_capabilities_block,
)
from inqtrix.providers.base import ChatTurn, LLMProvider, ToolCallRequest
from inqtrix.search_result import GroundedSearchResult, GroundedSource
from inqtrix.server.routes import create_router, register_routes
from inqtrix.settings import (
    AgentPlatformSettings,
    AgentSettings,
    ModelSettings,
    ServerSettings,
    Settings,
    StorageSettings,
)


def test_kernel_system_prompt_covers_tools_and_rendering():
    prompt = build_agent_kernel_system_prompt()
    # The rendering SSOT is embedded verbatim (M1 S5 drift rule).
    assert rendering_capabilities_block() in prompt
    for tool_name in (
        "ask_user",
        "search_project_knowledge",
        "read_project_document",
        "web_instant",
        "write_canvas",
        "propose_editor_patch",
        "run_web_research",
        "run_deep_mission",
        "write_todos",
    ):
        assert tool_name in prompt, f"{tool_name} fehlt im Kernel-Prompt"
    for rule in (
        "Ausgabeform",
        "Rueckfragen",
        "Werkzeugdisziplin",
        "Aktualitaet",
    ):
        assert rule in prompt


def test_kernel_cognition_prompt_rules_are_present():
    """The prompt pins recency routing, query discipline, and clarification."""
    prompt = build_agent_kernel_system_prompt()
    # 2.1 recency awareness — training knowledge is dated, web_instant
    # covers everything time-critical.
    assert "Trainingswissen kann veraltet sein" in prompt
    assert "Zeitkritische" in prompt
    # 2.4 query discipline: precise SEARCH query, one goal, shown
    # verbatim at the approval gate.
    assert "SUCHQUERY, keine Gespraechsfrage" in prompt
    assert "woertlich zur Freigabe angezeigt und exakt so gesucht" in prompt
    # 2.3 positive clarification trigger with every guardrail kept.
    assert "die BESSERE Arbeit" in prompt
    assert "Hoechstens zwei Rueckfrage-Runden" in prompt
    assert "Auto-Modus" in prompt


def test_kernel_user_message_composes_context_sections():
    message = build_kernel_user_message(
        "Ueberarbeite die Mail.",
        history_block="Nutzer: Entwirf eine Mail.\nAgent: Erledigt.",
        artifact_registry=(
            {
                "artifact_id": "art_1",
                "kind": "deliverable",
                "title": "E-Mail-Entwurf",
                "revision": 2,
                "updated_by": "agent",
            },
        ),
        last_response_form="canvas",
        response_form="chat",
        autonomy="autonomous",
    )
    assert "Bisheriger Verlauf" in message
    assert "artifact_id art_1, Revision 2" in message
    assert "Letzte Ausgabeform dieser Sitzung: canvas." in message
    assert "Chat-Antwort" in message
    assert "Auto" in message
    assert message.endswith("Auftrag:\nUeberarbeite die Mail.")
    # The functional run date leads every kernel turn because recency
    # routing keys its web-versus-memory decision on this line.
    from inqtrix.urls import today

    assert message.startswith(f"Heute ist {today()}.")

    bare = build_kernel_user_message("Nur die Frage.")
    assert bare == f"Heute ist {today()}.\n\nAuftrag:\nNur die Frage."


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


class RecordingSearch:
    def is_available(self) -> bool:
        return True

    def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
        return GroundedSearchResult(
            answer=f"Web zu {query}",
            sources=[
                GroundedSource(
                    url="https://example.com", title="Q", snippet="s"
                )
            ],
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


def _web_turn(
    text: str = "Ich suche kurz im Web.",
    *,
    call_id: str = "call_ev1",
    query: str = "Eventtest",
) -> ChatTurn:
    return ChatTurn(
        text=text,
        tool_calls=(
            ToolCallRequest(
                id=call_id,
                name="web_instant",
                arguments={"query": query},
            ),
        ),
        finish_reason="tool_calls",
        model="high-model",
        prompt_tokens=10,
        completion_tokens=5,
        raw=None,
    )


def make_client(llm: ScriptedToolLLM) -> TestClient:
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
    container = register_routes(
        router,
        providers=SimpleNamespace(llm=llm, search=RecordingSearch()),
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(2),
    )
    app.include_router(router)
    client = TestClient(app)
    client.container = container  # type: ignore[attr-defined]
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


def test_follow_up_turn_sees_the_prior_answer():
    llm = ScriptedToolLLM(
        [
            _text_turn("Die Hauptstadt ist Canberra."),
            _text_turn("Wie zuvor gesagt: Canberra."),
        ]
    )
    client = make_client(llm)
    with client:
        first = client.post(
            "/v1/runs",
            json={
                "question": "Hauptstadt Australiens?",
                "mode": "agent_kernel",
                "session_id": "sess-k1",
            },
        ).json()["run_id"]
        wait_status(client, first, {"completed"})

        second = client.post(
            "/v1/runs",
            json={
                "question": "Wiederhole deine letzte Antwort.",
                "mode": "agent_kernel",
                "session_id": "sess-k1",
            },
        ).json()["run_id"]
        wait_status(client, second, {"completed"})

        user_message = llm.chat_calls[1]["messages"][-1]["content"]
        assert "Bisheriger Verlauf" in user_message
        assert "Hauptstadt Australiens?" in user_message
        assert "Die Hauptstadt ist Canberra." in user_message


def test_follow_events_cover_tools_narration_and_phases():
    llm = ScriptedToolLLM([_web_turn(), _text_turn("Fertig.")])
    client = make_client(llm)
    with client:
        run_id = client.post(
            "/v1/runs",
            json={
                "question": "Suche bitte die aktuelle Quelle.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
            },
        ).json()["run_id"]
        wait_status(client, run_id, {"completed"})

        events = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        by_type: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            by_type.setdefault(event["type"], []).append(event["data"])

        assert [
            entry["phase"]
            for entry in by_type["inqtrix.agent.phase.changed"]
        ] == ["execution", "done"]
        started = by_type["inqtrix.agent.tool.started"]
        assert started[0]["tool"] == "web_instant"
        assert started[0]["invocation_id"] == started[0]["tool_call_id"]
        assert "Eventtest" in started[0]["args_preview"]
        finished = by_type["inqtrix.agent.tool.finished"]
        assert finished[0]["tool"] == "web_instant"
        assert finished[0]["invocation_id"] == started[0]["invocation_id"]
        narrations = by_type["inqtrix.agent.narration"]
        # Only tool-accompanying intent belongs in narration. The final
        # tool-free AI markdown is published through the answer channel.
        assert len(narrations) == 1
        assert narrations[0]["text"] == "Ich prüfe jetzt die benötigten Quellen."
        assert narrations[0]["narration_id"].startswith("kernel_")
        assert "inqtrix.node.model_resolution" in by_type


def test_approval_resume_does_not_replay_logical_tool_events():
    llm = ScriptedToolLLM([_web_turn(), _text_turn("Fertig.")])
    client = make_client(llm)
    with client:
        run_id = client.post(
            "/v1/runs",
            json={
                "question": "Suche bitte die aktuelle Quelle.",
                "mode": "agent_kernel",
                "autonomy": "strict",
            },
        ).json()["run_id"]
        wait_status(client, run_id, {"waiting_for_approval"})
        approval = client.get(
            f"/v1/runs/{run_id}/approvals"
        ).json()["data"][0]
        response = client.post(
            f"/v1/runs/{run_id}/approvals/{approval['approval_id']}",
            json={"decision": "approve"},
        )
        assert response.status_code == 200
        wait_status(client, run_id, {"completed"})

        events = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        started = [
            event["data"]
            for event in events
            if event["type"] == "inqtrix.agent.tool.started"
        ]
        finished = [
            event["data"]
            for event in events
            if event["type"] == "inqtrix.agent.tool.finished"
        ]
        intent_narrations = [
            event["data"]
            for event in events
            if event["type"] == "inqtrix.agent.narration"
            and event["data"].get("kind") == "intent"
        ]

        assert len(started) == len(finished) == len(intent_narrations) == 1
        assert started[0]["invocation_id"] == finished[0]["invocation_id"]


@pytest.mark.parametrize(
    "unsafe_intent",
    [
        (
            "| Region | Preis |\n|---|---:|\n| Global | 5 USD |\n\n"
            "Damit lautet die fertige Antwort bereits jetzt sehr ausfuehrlich."
        ),
        "Ich beginne mit der Recherche. " + ("Antwortentwurf " * 40),
        "Ich pruefe **jetzt** die [Preisliste](https://example.com).",
        "Die Antwort lautet 5 USD pro 1M Token. Ich pruefe noch die Quelle.",
    ],
)
def test_tool_accompanying_answer_draft_never_enters_plain_narration(
    unsafe_intent: str,
) -> None:
    llm = ScriptedToolLLM([_web_turn(unsafe_intent), _text_turn("Fertig.")])
    client = make_client(llm)
    with client:
        run_id = client.post(
            "/v1/runs",
            json={
                "question": "Suche bitte die aktuelle Quelle.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
            },
        ).json()["run_id"]
        wait_status(client, run_id, {"completed"})

        events = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        narrations = [
            event["data"]
            for event in events
            if event["type"] == "inqtrix.agent.narration"
        ]

        assert [item["text"] for item in narrations] == [
            "Ich prüfe jetzt die benötigten Quellen."
        ]
        assert all("|" not in item["text"] for item in narrations)
        assert all("**" not in item["text"] for item in narrations)


def test_tool_narration_is_localized_and_replay_ids_are_per_invocation() -> None:
    llm = ScriptedToolLLM(
        [
            _web_turn(
                "The answer is already 5 USD.",
                call_id="call_source_one",
                query="first source",
            ),
            _web_turn(
                "The answer is still 5 USD.",
                call_id="call_source_two",
                query="second source",
            ),
            _text_turn("Done."),
        ]
    )
    client = make_client(llm)
    with client:
        run_id = client.post(
            "/v1/runs",
            json={
                "question": "Search the current price and verify it.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
            },
        ).json()["run_id"]
        wait_status(client, run_id, {"completed"})

        events = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        narrations = [
            event["data"]
            for event in events
            if event["type"] == "inqtrix.agent.narration"
        ]

        assert [item["text"] for item in narrations] == [
            "I’m checking the required sources now.",
            "I’m checking the required sources now.",
        ]
        assert len({item["narration_id"] for item in narrations}) == 2


def test_schnell_graph_run_makes_one_web_instant_call():
    """The schnell tier's clamped recursion ceiling affords the tier's ONE
    published ``web_instant`` call plus the answer turn.

    Regression for the recalibration bug: the ceiling used to be a literal
    ``8`` super-steps — not even the bare answer turn — so a schnell run
    that (correctly) used its published ``web_instant_budget=1`` died in
    ``GraphRecursionError``. The clamp is now derived
    (``_ANSWER_TURN_SUPERSTEPS + _SCHNELL_TOOL_TURNS * _SUPERSTEPS_PER_
    TOOL_TURN``); this drives a real schnell GRAPH run (not the quick_web
    lane) with a todo turn BEFORE the search — the ordinary two-tool-turn
    trajectory that also died under the first recalibration attempt
    (one-turn clamp) — and pins that the published call is executable.
    """

    def _todo_turn() -> ChatTurn:
        return ChatTurn(
            text="",
            tool_calls=(
                ToolCallRequest(
                    id="call_suffplan",
                    name="write_todos",
                    arguments={
                        "todos": [
                            {"content": "Suchen", "status": "in_progress"},
                        ]
                    },
                ),
            ),
            finish_reason="tool_calls",
            model="high-model",
            prompt_tokens=10,
            completion_tokens=5,
            raw=None,
        )

    llm = ScriptedToolLLM([_todo_turn(), _web_turn(), _text_turn("Fertig.")])
    client = make_client(llm)
    with client:
        run_id = client.post(
            "/v1/runs",
            json={
                "question": "Was gibt es Neues dazu?",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "agent_overrides": {"agent_tier": "schnell"},
            },
        ).json()["run_id"]
        summary = wait_status(client, run_id, {"completed", "failed"})
        assert summary["status"] == "completed", (
            "schnell graph run failed — the clamped recursion ceiling no "
            f"longer affords the published web_instant call: {summary}"
        )

        events = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        started = [
            event["data"]
            for event in events
            if event["type"] == "inqtrix.agent.tool.started"
        ]
        # write_todos rides the same wire event; the ONE published web
        # call is what the clamp must afford.
        assert [
            entry["tool"] for entry in started if entry["tool"] != "write_todos"
        ] == ["web_instant"]
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"] == "Fertig."
