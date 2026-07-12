"""Kernel skill integration: load_skill + allowlist enforcement (M3-6).

Trajectories: the disclosure list reaches the model, ``load_skill``
activates a model_allowed skill (marker + instructions as the tool
result) and refuses user_only ones, the allowed_tools union blocks
foreign tools VISIBLY, and — the security-critical one — a restriction
acquired before a park survives the resume via the checkpointed marker.
"""

from __future__ import annotations

import asyncio
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

SKILL_PAYLOAD = {
    "label": "email-stil",
    "title": "E-Mail-Stil",
    "description": "Formuliert E-Mails im Hausstil.",
    "when_to_use": "Wenn eine E-Mail entworfen werden soll.",
    "instructions_markdown": "Schreibe E-Mails knapp und freundlich.",
    "clarification_points": [],
    "deliverable": "email",
    "allowed_tools": ["write_canvas", "search_project_knowledge"],
    "requires_plan": "auto",
    "invocation": "model_allowed",
}


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


class SkillCheckingLLM(ScriptedToolLLM):
    """Kernel script with a deterministic unanswered SkillPointCheck."""

    def supports_structured_output(self, *, model: str | None = None) -> bool:
        return True

    def complete_structured(
        self, prompt: str, *, schema: Any, schema_name: str, **kwargs: Any
    ) -> Any:
        assert schema_name == "SkillPointCheck"
        return SimpleNamespace(
            parsed={
                "points": [
                    {
                        "id": "p1",
                        "answered": False,
                        "answer_from_context": "",
                    }
                ]
            },
            prompt_tokens=3,
            completion_tokens=2,
        )


class DeepSkillCheckingLLM(SkillCheckingLLM):
    """Skill gate plus a passing Deep review that records its prompt."""

    def __init__(self, turns: list[ChatTurn]) -> None:
        super().__init__(turns)
        self.review_prompt = ""

    def complete_structured(
        self, prompt: str, *, schema: Any, schema_name: str, **kwargs: Any
    ) -> Any:
        if schema_name == "SkillPointCheck":
            return super().complete_structured(
                prompt, schema=schema, schema_name=schema_name, **kwargs
            )
        assert schema_name == "DeepReviewVerdict"
        self.review_prompt = prompt
        return SimpleNamespace(
            parsed={
                "complete": True,
                "grounded": True,
                "contradictions_named": True,
                "findings": [],
            },
            prompt_tokens=3,
            completion_tokens=2,
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
        providers=SimpleNamespace(llm=llm, search=None),
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(2),
    )
    app.include_router(router)
    client = TestClient(app)
    client.container = container  # type: ignore[attr-defined]
    return client


def _create_skill(client, **overrides) -> str:
    record = asyncio.run(
        client.container.skill_service.create(
            {**SKILL_PAYLOAD, **overrides},
            tenant_id="default",
            owner_sub=None,
        )
    )
    return record.id


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


def _submit(
    client, *, autonomy: str = "autonomous", skill_ids=None,
    depth: str | None = None,
) -> str:
    body: dict[str, Any] = {
        "question": "Entwirf eine E-Mail an das Team.",
        "mode": "agent_kernel",
        "autonomy": autonomy,
    }
    if skill_ids:
        body["skill_ids"] = skill_ids
    if depth:
        body["agent_overrides"] = {"depth": depth}
    response = client.post("/v1/runs", json=body)
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def test_disclosure_load_and_allowlist_block():
    llm = ScriptedToolLLM(
        [
            _tool_turn("call_load1", "load_skill", {"skill_id": "PENDING"}),
            # After activation the allowlist forbids web_instant.
            _tool_turn("call_web1", "web_instant", {"query": "extern"}),
            _text_turn("Ohne Websuche erledigt."),
        ]
    )
    client = make_client(llm)
    with client:
        skill_id = _create_skill(client)
        # Patch the scripted args now that the real id exists.
        llm._turns[0] = _tool_turn(
            "call_load1", "load_skill", {"skill_id": skill_id}
        )
        run_id = _submit(client)
        wait_status(client, run_id, {"completed"})

        # Disclosure reached the FIRST user message under the budget.
        first_user = llm.chat_calls[0]["messages"][-1]["content"]
        assert "/email-stil" in first_user
        assert skill_id in first_user

        # load_skill returned marker + instructions.
        load_reply = [
            m for m in llm.chat_calls[1]["messages"] if m.get("role") == "tool"
        ][-1]["content"]
        assert load_reply.startswith(f"[skill_geladen:{skill_id}@")
        assert "knapp und freundlich" in load_reply

        # The allowlist blocked web_instant VISIBLY.
        block_reply = [
            m for m in llm.chat_calls[2]["messages"] if m.get("role") == "tool"
        ][-1]["content"]
        assert "nicht erlaubt" in block_reply
        assert "write_canvas" in block_reply


def test_user_only_skill_refuses_model_activation():
    llm = ScriptedToolLLM(
        [
            _tool_turn("call_load2", "load_skill", {"skill_id": "PENDING"}),
            _text_turn("Verstanden."),
        ]
    )
    client = make_client(llm)
    with client:
        skill_id = _create_skill(
            client, label="privat", invocation="user_only"
        )
        llm._turns[0] = _tool_turn(
            "call_load2", "load_skill", {"skill_id": skill_id}
        )
        run_id = _submit(client)
        wait_status(client, run_id, {"completed"})
        reply = [
            m for m in llm.chat_calls[1]["messages"] if m.get("role") == "tool"
        ][-1]["content"]
        assert "nur vom Nutzer" in reply
        # user_only skills never appear in the disclosure either.
        first_user = llm.chat_calls[0]["messages"][-1]["content"]
        assert "/privat" not in first_user


def test_restriction_survives_park_and_resume():
    """Security invariant: a loaded allowlist re-arms after a park."""
    llm = ScriptedToolLLM(
        [
            _tool_turn("call_load3", "load_skill", {"skill_id": "PENDING"}),
            _tool_turn(
                "call_ask1",
                "ask_user",
                {
                    "question": "An wen genau?",
                    "options": ["Team", "Abteilung"],
                },
            ),
            # Resumed segment: the model tries a forbidden tool.
            _tool_turn("call_web2", "web_instant", {"query": "extern"}),
            _text_turn("Fertig ohne Websuche."),
        ]
    )
    client = make_client(llm)
    with client:
        skill_id = _create_skill(client)
        llm._turns[0] = _tool_turn(
            "call_load3", "load_skill", {"skill_id": skill_id}
        )
        run_id = _submit(client)
        wait_status(client, run_id, {"waiting_for_input"})
        rows = client.get(f"/v1/runs/{run_id}/clarifications").json()["data"]
        client.post(
            f"/v1/runs/{run_id}/clarifications/{rows[0]['clarification_id']}",
            json={"answer": "Ans Team."},
        )
        wait_status(client, run_id, {"completed"})
        # The post-resume web_instant was blocked by the RE-ARMED union.
        final_messages = llm.chat_calls[-1]["messages"]
        block_reply = [
            m
            for m in final_messages
            if m.get("role") == "tool"
            and "web_instant" in str(m.get("content", ""))
        ]
        assert block_reply, "blocked tool reply missing after resume"
        assert "nicht erlaubt" in block_reply[-1]["content"]


def test_dynamic_skill_revision_drift_fails_closed_on_resume():
    llm = ScriptedToolLLM(
        [
            _tool_turn("call_load_drift", "load_skill", {"skill_id": "PENDING"}),
            _tool_turn(
                "call_ask_drift",
                "ask_user",
                {"question": "Fortfahren?", "options": ["Ja", "Nein"]},
            ),
            _text_turn("Darf nicht erreicht werden."),
        ]
    )
    client = make_client(llm)
    with client:
        skill_id = _create_skill(client)
        llm._turns[0] = _tool_turn(
            "call_load_drift", "load_skill", {"skill_id": skill_id}
        )
        run_id = _submit(client)
        wait_status(client, run_id, {"waiting_for_input"})

        current = asyncio.run(
            client.container.skill_service.get_admitted(
                skill_id, tenant_id="default"
            )
        )
        asyncio.run(
            client.container.skill_service.update(
                skill_id,
                {**SKILL_PAYLOAD, "allowed_tools": ["web_instant"]},
                tenant_id="default",
                expected_updated_at=current.updated_at,
            )
        )
        row = client.get(
            f"/v1/runs/{run_id}/clarifications"
        ).json()["data"][0]
        response = client.post(
            f"/v1/runs/{run_id}/clarifications/{row['clarification_id']}",
            json={"answer": "Ja"},
        )
        assert response.status_code == 200, response.text
        failed = wait_status(client, run_id, {"failed", "completed"})
        assert failed["status"] == "failed"
        assert "seit der Aktivierung geaendert" in failed["error"]["message"]


def test_attached_skill_instructions_ride_the_user_message():
    llm = ScriptedToolLLM([_text_turn("E-Mail steht im Canvas.")])
    client = make_client(llm)
    with client:
        skill_id = _create_skill(client)
        run_id = _submit(client, skill_ids=[skill_id])
        wait_status(client, run_id, {"completed"})
        first_user = llm.chat_calls[0]["messages"][-1]["content"]
        assert "[Skill 'email-stil'" in first_user
        assert "knapp und freundlich" in first_user


def test_attached_skill_required_input_parks_before_main_model():
    llm = SkillCheckingLLM([_text_turn("Fertig fuer Marketing.")])
    client = make_client(llm)
    points = [
        {
            "name": "zielgruppe",
            "question": "Welche Zielgruppe?",
            "options": [],
            "required": True,
            "default_assumption": "",
        }
    ]
    with client:
        skill_id = _create_skill(
            client,
            instructions_markdown="Schreibe fuer {{zielgruppe}}.",
            clarification_points=points,
        )
        run_id = _submit(client, skill_ids=[skill_id])
        wait_status(client, run_id, {"waiting_for_input"})
        assert llm.chat_calls == []
        rows = client.get(f"/v1/runs/{run_id}/clarifications").json()["data"]
        response = client.post(
            f"/v1/runs/{run_id}/clarifications/{rows[0]['clarification_id']}",
            json={"answer": "Marketing"},
        )
        assert response.status_code == 200, response.text
        wait_status(client, run_id, {"completed"})
        assert "Schreibe fuer Marketing" in str(llm.chat_calls[0]["messages"])


def test_dynamic_skill_activates_only_after_required_input():
    llm = SkillCheckingLLM(
        [
            _tool_turn("call_required", "load_skill", {"skill_id": "PENDING"}),
            _text_turn("Fertig fuer Marketing."),
        ]
    )
    client = make_client(llm)
    points = [
        {
            "name": "zielgruppe",
            "question": "Welche Zielgruppe?",
            "options": [],
            "required": True,
            "default_assumption": "",
        }
    ]
    with client:
        skill_id = _create_skill(
            client,
            instructions_markdown="Schreibe fuer {{zielgruppe}}.",
            clarification_points=points,
        )
        llm._turns[0] = _tool_turn(
            "call_required", "load_skill", {"skill_id": skill_id}
        )
        run_id = _submit(client)
        wait_status(client, run_id, {"waiting_for_input"})
        assert len(llm.chat_calls) == 1
        rows = client.get(f"/v1/runs/{run_id}/clarifications").json()["data"]
        response = client.post(
            f"/v1/runs/{run_id}/clarifications/{rows[0]['clarification_id']}",
            json={"answer": "Marketing"},
        )
        assert response.status_code == 200, response.text
        wait_status(client, run_id, {"completed"})
        reply = [
            message
            for message in llm.chat_calls[-1]["messages"]
            if message.get("role") == "tool"
        ][-1]["content"]
        assert "Schreibe fuer Marketing" in reply


def test_deep_review_receives_resolved_attached_skill_inputs():
    llm = DeepSkillCheckingLLM([_text_turn("Fertig fuer Marketing.")])
    client = make_client(llm)
    points = [
        {
            "name": "zielgruppe",
            "question": "Welche Zielgruppe?",
            "options": [],
            "required": True,
            "default_assumption": "",
        }
    ]
    with client:
        skill_id = _create_skill(
            client,
            instructions_markdown="Schreibe fuer {{zielgruppe}}.",
            clarification_points=points,
        )
        run_id = _submit(client, skill_ids=[skill_id], depth="deep")
        wait_status(client, run_id, {"waiting_for_input"})
        row = client.get(
            f"/v1/runs/{run_id}/clarifications"
        ).json()["data"][0]
        response = client.post(
            f"/v1/runs/{run_id}/clarifications/{row['clarification_id']}",
            json={"answer": "Marketing"},
        )
        assert response.status_code == 200, response.text
        wait_status(client, run_id, {"completed"})
        assert "RESOLVED SKILL INPUTS" in llm.review_prompt
        assert "Schreibe fuer Marketing" in llm.review_prompt
