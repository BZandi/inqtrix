"""Kernel canvas tools: write_canvas CAS + always-gated patches (M2-7).

write_canvas creates/updates session-scoped ``deliverable`` artifacts
with optimistic concurrency (visible conflict text, never a clobber);
``propose_editor_patch`` parks for approval in EVERY mode (E14).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("deepagents")

from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    register_routes(
        router,
        providers=SimpleNamespace(llm=llm, search=None),
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(2),
    )
    app.include_router(router)
    return TestClient(app)


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


def _submit(client: TestClient, *, autonomy: str = "autonomous") -> str:
    response = client.post(
        "/v1/runs",
        json={
            "question": "Entwirf eine E-Mail.",
            "mode": "agent_kernel",
            "autonomy": autonomy,
            "session_id": "sess-canvas",
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def _artifacts(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    return client.get(f"/v1/runs/{run_id}/artifacts").json()["data"]


def test_write_canvas_creates_then_updates_with_cas():
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_canvas1",
                "write_canvas",
                {
                    "title": "E-Mail-Entwurf",
                    "content_markdown": "# Entwurf V1",
                    "deliverable_kind": "email",
                },
            ),
            # The model updates using id + revision from the tool result.
            None,  # placeholder, replaced below after first result known
        ]
    )
    # Two-phase scripting is impossible without knowing the artifact id;
    # instead: the update turn is generated from the FIRST tool reply by
    # a callable-aware provider.

    class TwoPhase(ScriptedToolLLM):
        def chat(self, messages, *, tools=None, **kwargs):
            self.chat_calls.append(
                {"messages": list(messages), "tools": tools}
            )
            turn_index = len(self.chat_calls)
            if turn_index == 1:
                return _tool_turn(
                    "call_canvas1",
                    "write_canvas",
                    {
                        "title": "E-Mail-Entwurf",
                        "content_markdown": "# Entwurf V1",
                        "deliverable_kind": "email",
                    },
                )
            if turn_index == 2:
                reply = [
                    m for m in messages if m.get("role") == "tool"
                ][-1]["content"]
                artifact_id = reply.split("artifact_id ")[1].split(",")[0]
                return _tool_turn(
                    "call_canvas2",
                    "write_canvas",
                    {
                        "title": "E-Mail-Entwurf",
                        "content_markdown": "# Entwurf V2",
                        "deliverable_kind": "email",
                        "artifact_id": artifact_id,
                        "expected_revision": 1,
                    },
                )
            return _text_turn("Entwurf steht im Canvas.")

    llm = TwoPhase([])
    client = make_client(llm)
    with client:
        run_id = _submit(client)
        wait_status(client, run_id, {"completed"})
        rows = _artifacts(client, run_id)
        deliverables = [r for r in rows if r["kind"] == "deliverable"]
        assert len(deliverables) == 1
        row = deliverables[0]
        assert row["revision"] == 2
        detail = client.get(
            f"/v1/runs/{run_id}/artifacts/{row['artifact_id']}"
        ).json()
        assert detail["content_markdown"] == "# Entwurf V2"
        assert detail["payload"] == {"deliverable_kind": "email"}


def test_write_canvas_normalizes_currency_but_preserves_math_and_code():
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_currency_canvas",
                "write_canvas",
                {
                    "title": "Markt-Memo",
                    "content_markdown": (
                        "Umsatz: US-$1.5T. Formel: $x$. "
                        "Code: `$raw`."
                    ),
                    "deliverable_kind": "memo",
                },
            ),
            _text_turn("Memo steht im Canvas."),
        ]
    )
    client = make_client(llm)
    with client:
        run_id = _submit(client)
        wait_status(client, run_id, {"completed"})
        row = next(
            item
            for item in _artifacts(client, run_id)
            if item["kind"] == "deliverable"
        )
        detail = client.get(
            f"/v1/runs/{run_id}/artifacts/{row['artifact_id']}"
        ).json()

    assert detail["content_markdown"] == (
        r"Umsatz: US-\$1.5T. Formel: $x$. Code: `$raw`."
    )


def test_write_canvas_stale_revision_is_a_visible_conflict():
    class StaleUpdate(ScriptedToolLLM):
        def chat(self, messages, *, tools=None, **kwargs):
            self.chat_calls.append(
                {"messages": list(messages), "tools": tools}
            )
            turn_index = len(self.chat_calls)
            if turn_index == 1:
                return _tool_turn(
                    "call_c1",
                    "write_canvas",
                    {
                        "title": "Memo",
                        "content_markdown": "# V1",
                        "deliverable_kind": "memo",
                    },
                )
            if turn_index == 2:
                reply = [
                    m for m in messages if m.get("role") == "tool"
                ][-1]["content"]
                artifact_id = reply.split("artifact_id ")[1].split(",")[0]
                return _tool_turn(
                    "call_c2",
                    "write_canvas",
                    {
                        "title": "Memo",
                        "content_markdown": "# stale",
                        "deliverable_kind": "memo",
                        "artifact_id": artifact_id,
                        "expected_revision": 7,
                    },
                )
            return _text_turn("Konflikt erkannt.")

    llm = StaleUpdate([])
    client = make_client(llm)
    with client:
        run_id = _submit(client)
        wait_status(client, run_id, {"completed"})
        rows = [
            r
            for r in _artifacts(client, run_id)
            if r["kind"] == "deliverable"
        ]
        # The stale write did NOT clobber: still revision 1.
        assert rows[0]["revision"] == 1
        conflict_reply = [
            m
            for m in llm.chat_calls[2]["messages"]
            if m.get("role") == "tool"
        ][-1]["content"]
        assert "Revisionskonflikt" in conflict_reply
        assert "Revision 1" in conflict_reply


def test_write_canvas_cannot_update_another_sessions_artifact():
    class CrossSessionAttempt(ScriptedToolLLM):
        def chat(self, messages, *, tools=None, **kwargs):
            self.chat_calls.append({"messages": list(messages), "tools": tools})
            index = len(self.chat_calls)
            if index == 1:
                return _tool_turn(
                    "call_owner_create",
                    "write_canvas",
                    {"title": "Owner", "content_markdown": "# Original"},
                )
            if index == 2:
                reply = [
                    item for item in messages if item.get("role") == "tool"
                ][-1]["content"]
                self.victim_id = reply.split("artifact_id ")[1].split(",")[0]
                return _text_turn("Owner fertig.")
            if index == 3:
                return _tool_turn(
                    "call_foreign_update",
                    "write_canvas",
                    {
                        "title": "Foreign",
                        "content_markdown": "# Ueberschrieben",
                        "artifact_id": self.victim_id,
                        "expected_revision": 1,
                        "reference_ids": [],
                    },
                )
            return _text_turn("Fremdzugriff abgelehnt.")

    llm = CrossSessionAttempt([])
    client = make_client(llm)
    with client:
        owner_run = _submit(client)
        wait_status(client, owner_run, {"completed"})
        foreign = client.post(
            "/v1/runs",
            json={
                "question": "Versuche ein fremdes Canvas zu aendern.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "session_id": "sess-foreign",
            },
        )
        assert foreign.status_code == 202, foreign.text
        wait_status(client, foreign.json()["run_id"], {"completed"})

        detail = client.get(
            f"/v1/runs/{owner_run}/artifacts/{llm.victim_id}"
        ).json()
        assert detail["revision"] == 1
        assert detail["content_markdown"] == "# Original"
        reply = [
            item
            for item in llm.chat_calls[3]["messages"]
            if item.get("role") == "tool"
        ][-1]["content"]
        assert "nicht gefunden" in reply


def test_invalid_deliverable_kind_is_a_visible_tool_error():
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_bad",
                "write_canvas",
                {
                    "title": "X",
                    "content_markdown": "# X",
                    "deliverable_kind": "poster",
                },
            ),
            _text_turn("Korrigiert."),
        ]
    )
    client = make_client(llm)
    with client:
        run_id = _submit(client)
        wait_status(client, run_id, {"completed"})
        assert (
            [r for r in _artifacts(client, run_id) if r["kind"] == "deliverable"]
            == []
        )
        reply = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ][-1]["content"]
        assert "Unbekannte Dokumentart" in reply


def test_write_canvas_rejects_unknown_citation_labels_loudly():
    """P7 parity: a canvas text citing labels that are not among the
    ATTACHED references is a visible tool error, never a silent write."""
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_uncited",
                "write_canvas",
                {
                    "title": "Memo",
                    "content_markdown": "Behauptung mit Beleg [W1].",
                    "deliverable_kind": "memo",
                },
            ),
            _tool_turn(
                "call_fixed",
                "write_canvas",
                {
                    "title": "Memo",
                    "content_markdown": "Behauptung ohne Beleg-Label.",
                    "deliverable_kind": "memo",
                },
            ),
            _text_turn("Memo steht im Canvas."),
        ]
    )
    client = make_client(llm)
    with client:
        run_id = _submit(client)
        wait_status(client, run_id, {"completed"})
        reply = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ][-1]["content"]
        assert "unbekannte Labels" in reply
        assert "W1" in reply
        deliverables = [
            r for r in _artifacts(client, run_id) if r["kind"] == "deliverable"
        ]
        assert len(deliverables) == 1


def test_propose_editor_patch_gates_even_in_autonomous():
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_patch1",
                "propose_editor_patch",
                {
                    "document_id": "doc_1",
                    "edits": [
                        {
                            "position": "replace",
                            "find": "alt",
                            "text": "neu",
                        }
                    ],
                    "summary": "Begriff ersetzen",
                },
            ),
            _text_turn("Patch vorgeschlagen."),
        ]
    )
    client = make_client(llm)
    with client:
        run_id = _submit(client, autonomy="autonomous")
        wait_status(client, run_id, {"waiting_for_approval"})
        rows = client.get(f"/v1/runs/{run_id}/approvals").json()["data"]
        pending = [r for r in rows if r["status"] == "pending"]
        assert pending[0]["kind"] == "tool"
        action = pending[0]["payload"]["actions"][0]
        assert action["tool"] == "propose_editor_patch"
        assert action["args"]["document_id"] == "doc_1"

        client.post(
            f"/v1/runs/{run_id}/approvals/{pending[0]['approval_id']}",
            json={"decision": "approve"},
        )
        wait_status(client, run_id, {"completed"})
        # The capability ran after consent and denied VISIBLY (the demo
        # deployment has no such editor document).
        reply = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ][-1]["content"]
        assert "Werkzeug-Fehler" in reply
        assert "editor.document_not_found" in reply
