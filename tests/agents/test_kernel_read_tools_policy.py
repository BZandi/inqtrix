"""Kernel read tools + policy gates (plan M2 step 6).

The policy matrix (which tool interrupts in which mode), and the full
tool-approval trajectory over the platform path: gated ``web_instant``
parks as ``kind="tool"`` with the QUERY VERBATIM in the approval
payload; approve executes, edit executes with replaced args, reject
skips and the model continues visibly.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("deepagents")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from inqtrix.capabilities.contracts import (
    CapabilityContext,
    CapabilityDefinition,
    Effect,
)
from inqtrix.agents.kernel.policy import interrupt_config_for
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


# -- policy matrix --------------------------------------------------------- #


def test_policy_matrix_per_autonomy():
    # Write-effect tools gate in EVERY mode (E14) — including Auto.
    autonomous = interrupt_config_for("autonomous")
    assert set(autonomous) == {"propose_editor_patch"}

    balanced = interrupt_config_for("balanced")
    assert set(balanced) == {
        "web_instant",
        "search_project_knowledge",
        "run_web_research",
        "run_deep_mission",
        "load_skill",
        "propose_editor_patch",
    }
    assert "when" not in balanced["web_instant"]
    assert balanced["web_instant"]["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
    ]
    when = balanced["search_project_knowledge"]["when"]
    scoped = SimpleNamespace(
        tool_call={"args": {"query": "x", "collection_ids": ["col_1"]}}
    )
    unscoped = SimpleNamespace(tool_call={"args": {"query": "x"}})
    assert when(scoped) is False
    assert when(unscoped) is True

    strict = interrupt_config_for("strict")
    assert set(strict) == {
        "web_instant",
        "search_project_knowledge",
        "read_project_document",
        "read_canvas",
        "write_canvas",
        "run_web_research",
        "run_deep_mission",
        "load_skill",
        "propose_editor_patch",
    }
    assert all("when" not in config for config in strict.values())

    with pytest.raises(ValueError):
        interrupt_config_for("yolo")


# -- trajectory fixtures ---------------------------------------------------- #


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
    def __init__(self) -> None:
        self.queries: list[str] = []

    def is_available(self) -> bool:
        return True

    def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
        self.queries.append(query)
        return GroundedSearchResult(
            answer=f"Web-Antwort zu {query}",
            sources=[
                GroundedSource(
                    url="https://example.com/a",
                    title="Quelle A",
                    snippet="Auszug.",
                )
            ],
            prompt_tokens=17,
            completion_tokens=9,
        )


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


def make_client(llm: ScriptedToolLLM, search: Any) -> TestClient:
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
        providers=SimpleNamespace(llm=llm, search=search),
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


def _submit(client: TestClient, *, autonomy: str) -> str:
    response = client.post(
        "/v1/runs",
        json={
            "question": "Was ist neu beim EU AI Act?",
            "mode": "agent_kernel",
            "autonomy": autonomy,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def _register_empty_knowledge_search(client: TestClient) -> None:
    """Register just enough knowledge capability for directive admission."""

    class _Input(BaseModel):
        query: str
        collection_ids: list[str] = Field(default_factory=list)
        top_k: int = 8

    class _Output(BaseModel):
        hits: list[dict[str, Any]] = Field(default_factory=list)

    async def _handler(
        payload: BaseModel, context: CapabilityContext
    ) -> BaseModel:
        del payload, context
        return _Output()

    client.container.capability_registry.register(  # type: ignore[attr-defined]
        CapabilityDefinition(
            id="knowledge.search",
            summary="Search project knowledge.",
            input_model=_Input,
            output_model=_Output,
            effect=Effect.READ,
            idempotent=True,
            handler=_handler,
        )
    )


def _pending_tool_approval(client: TestClient, run_id: str) -> dict[str, Any]:
    rows = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
    pending = [r for r in rows if r["status"] == "pending"]
    assert len(pending) == 1, rows
    assert pending[0]["kind"] == "tool"
    return pending[0]


def _offered_tool_names(llm: ScriptedToolLLM) -> set[str]:
    tools = llm.chat_calls[0]["tools"] or []
    return {
        str((item.get("function") or {}).get("name") or item.get("name") or "")
        for item in tools
        if isinstance(item, dict)
    }


# -- trajectories ------------------------------------------------------------#


def test_balanced_web_instant_gates_and_approve_executes():
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_web1", "web_instant", {"query": "EU AI Act 2026"}
            ),
            _text_turn("Antwort mit Webwissen."),
        ]
    )
    search = RecordingSearch()
    client = make_client(llm, search)
    with client:
        run_id = _submit(client, autonomy="balanced")
        wait_status(client, run_id, {"waiting_for_approval"})
        # No search ran before the consent.
        assert search.queries == []
        approval = _pending_tool_approval(client, run_id)
        # The query stands VERBATIM in the approval content.
        action = approval["payload"]["actions"][0]
        assert action["tool"] == "web_instant"
        assert action["args"] == {"query": "EU AI Act 2026"}

        decided = client.post(
            f"/v1/runs/{run_id}/approvals/{approval['approval_id']}",
            json={"decision": "approve"},
        )
        assert decided.status_code == 200, decided.text
        wait_status(client, run_id, {"completed"})
        assert search.queries == ["EU AI Act 2026"]
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"] == "Antwort mit Webwissen."
        # The resumed model saw the tool result.
        tool_replies = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert "Web-Antwort zu EU AI Act 2026" in tool_replies[0]["content"]


def test_edit_decision_replaces_the_query():
    llm = ScriptedToolLLM(
        [
            _tool_turn("call_web2", "web_instant", {"query": "vage Suche"}),
            _text_turn("Fertig."),
        ]
    )
    search = RecordingSearch()
    client = make_client(llm, search)
    with client:
        run_id = _submit(client, autonomy="balanced")
        wait_status(client, run_id, {"waiting_for_approval"})
        approval = _pending_tool_approval(client, run_id)
        decided = client.post(
            f"/v1/runs/{run_id}/approvals/{approval['approval_id']}",
            json={
                "decision": "edit",
                "actions": [
                    {
                        "tool": "web_instant",
                        "args": {"query": "EU AI Act Fristen 2027"},
                    }
                ],
            },
        )
        assert decided.status_code == 200, decided.text
        wait_status(client, run_id, {"completed"})
        assert search.queries == ["EU AI Act Fristen 2027"]


def test_reject_decision_skips_tool_and_model_continues():
    llm = ScriptedToolLLM(
        [
            _tool_turn("call_web3", "web_instant", {"query": "extern"}),
            _text_turn("Ohne Websuche beantwortet."),
        ]
    )
    search = RecordingSearch()
    client = make_client(llm, search)
    with client:
        run_id = _submit(client, autonomy="balanced")
        wait_status(client, run_id, {"waiting_for_approval"})
        approval = _pending_tool_approval(client, run_id)
        client.post(
            f"/v1/runs/{run_id}/approvals/{approval['approval_id']}",
            json={"decision": "reject", "note": "Keine externe Suche."},
        )
        wait_status(client, run_id, {"completed"})
        assert search.queries == []
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"] == "Ohne Websuche beantwortet."


def test_autonomous_web_instant_runs_ungated():
    llm = ScriptedToolLLM(
        [
            _tool_turn("call_web4", "web_instant", {"query": "frei"}),
            _text_turn("Direkt erledigt."),
        ]
    )
    search = RecordingSearch()
    client = make_client(llm, search)
    with client:
        run_id = _submit(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        assert search.queries == ["frei"]
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert approvals == []
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["usage"]["prompt_tokens"] == 37
        assert result["usage"]["completion_tokens"] == 19
        assert result["usage"]["total_tokens"] == 56


def test_web_instant_tool_and_final_answer_normalize_currency_markdown():
    class CurrencySearch(RecordingSearch):
        def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
            self.queries.append(query)
            return GroundedSearchResult(
                answer=(
                    "Der Markt erreicht US-$1.5T "
                    "([Quelle](https://example.com/a))."
                ),
                sources=[
                    GroundedSource(
                        url="https://example.com/a",
                        title="Quelle A",
                        snippet="",
                    )
                ],
            )

    llm = ScriptedToolLLM(
        [
            _tool_turn("call_web_currency", "web_instant", {"query": "frei"}),
            _text_turn("Der Markt erreicht $1.5T; Formel $x$."),
        ]
    )
    client = make_client(llm, CurrencySearch())
    with client:
        run_id = _submit(client, autonomy="autonomous")
        wait_status(client, run_id, {"completed"})
        tool_reply = [
            item
            for item in llm.chat_calls[1]["messages"]
            if item.get("role") == "tool"
        ][-1]["content"]
        assert r"US-\$1.5T" in tool_reply
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"] == r"Der Markt erreicht \$1.5T; Formel $x$."


def test_source_policy_removes_web_from_normal_kernel_surface():
    llm = ScriptedToolLLM([_text_turn("Ohne Web beantwortet.")])
    search = RecordingSearch()
    client = make_client(llm, search)
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Nur ohne externe Quellen.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "source_policy": {
                    "web": "disabled",
                    "knowledge": "available",
                },
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"completed"})
        offered = _offered_tool_names(llm)
        assert "web_instant" not in offered
        assert "run_web_research" not in offered
        assert search.queries == []
        execution = client.get(f"/v1/runs/{run_id}/result").json()[
            "execution"
        ]
        assert execution["source_policy"]["web"] == "disabled"
        assert execution["tool_use_counts"]["web"] == 0


def test_knowledge_only_blocks_web_but_keeps_knowledge_approval():
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_knowledge_only",
                "search_project_knowledge",
                {"query": "interne Richtlinie"},
            ),
            _text_turn("Aus Projektwissen beantwortet."),
        ]
    )
    search = RecordingSearch()
    client = make_client(llm, search)
    with client:
        _register_empty_knowledge_search(client)
        response = client.post(
            "/v1/runs",
            json={
                "question": "Was sagt unsere interne Richtlinie?",
                "mode": "workspace_agent",
                "autonomy": "balanced",
                "execution_directive": "knowledge_only",
                "source_policy": {
                    "web": "available",
                    "knowledge": "disabled",
                },
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"waiting_for_approval"})

        offered = _offered_tool_names(llm)
        assert "web_instant" not in offered
        assert "run_web_research" not in offered
        assert "search_project_knowledge" in offered
        approval = _pending_tool_approval(client, run_id)
        assert (
            approval["payload"]["actions"][0]["tool"]
            == "search_project_knowledge"
        )
        assert search.queries == []


def test_balanced_scoped_knowledge_search_runs_ungated():
    """The when-predicate: only the UN-scoped project sweep gates.

    No knowledge service is wired, so the executed tool returns a
    VISIBLE not-available text — the gating semantics are what this
    test secures, and the model must acknowledge the gap.
    """
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_rag1",
                "search_project_knowledge",
                {"query": "intern", "collection_ids": ["col_1"]},
            ),
            _text_turn("Interne Suche versucht."),
        ]
    )
    client = make_client(llm, RecordingSearch())
    with client:
        run_id = _submit(client, autonomy="balanced")
        wait_status(client, run_id, {"completed"})
        approvals = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        assert approvals == []
        tool_replies = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert "nicht eingerichtet" in tool_replies[0]["content"]


def test_balanced_unscoped_knowledge_search_gates():
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_rag2",
                "search_project_knowledge",
                {"query": "alles durchsuchen"},
            ),
            _text_turn("Nach Freigabe gesucht."),
        ]
    )
    client = make_client(llm, RecordingSearch())
    with client:
        run_id = _submit(client, autonomy="balanced")
        wait_status(client, run_id, {"waiting_for_approval"})
        approval = _pending_tool_approval(client, run_id)
        assert (
            approval["payload"]["actions"][0]["tool"]
            == "search_project_knowledge"
        )
        client.post(
            f"/v1/runs/{run_id}/approvals/{approval['approval_id']}",
            json={"decision": "approve"},
        )
        wait_status(client, run_id, {"completed"})
