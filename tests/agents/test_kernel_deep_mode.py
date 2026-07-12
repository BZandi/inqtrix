"""Deep mode on the kernel (plan M4): budgets, prompts, verification.

Deep must verify MEASURABLY, not just cost more: the trajectory pins
that a deep run issues the rubric review call, that findings trigger
exactly ONE revision whose text becomes the final answer, and that a
normal run never touches the review path. The unit tests pin the
deterministic derivations (effort precedence, iteration ceiling,
prompt line, child profiles).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import re

import pytest

pytest.importorskip("deepagents")

from inqtrix.agents.prompts import build_kernel_user_message
from inqtrix.agents.kernel.algorithm import KernelAgentAlgorithm
from inqtrix.pagination import encode_cursor
from inqtrix.providers.base import ChatTurn, ToolCallRequest

from tests.agents.test_kernel_prompt_events import (
    ScriptedToolLLM,
    _text_turn,
    make_client,
    wait_status,
)


class StructuredScriptedLLM(ScriptedToolLLM):
    """ScriptedToolLLM plus a scripted native structured-output path."""

    def __init__(self, turns: list[Any], structured: list[Any]) -> None:
        super().__init__(turns)
        self._structured = list(structured)
        self.structured_calls: list[dict[str, Any]] = []

    def supports_structured_output(self, *, model: str | None = None) -> bool:
        return True

    def complete_structured(
        self, prompt: str, *, schema: Any, schema_name: str, **kwargs: Any
    ) -> Any:
        self.structured_calls.append(
            {
                "prompt": prompt,
                "schema_name": schema_name,
                "model": kwargs.get("model"),
                "reasoning_effort": kwargs.get("reasoning_effort"),
            }
        )
        if not self._structured:
            raise AssertionError("scripted structured provider ran out")
        return SimpleNamespace(
            parsed=self._structured.pop(0),
            prompt_tokens=7,
            completion_tokens=3,
        )


class CanvasReviewLLM(StructuredScriptedLLM):
    """Target the concrete canvas id observed in the review bundle."""

    def complete_structured(
        self, prompt: str, *, schema: Any, schema_name: str, **kwargs: Any
    ) -> Any:
        self.structured_calls.append(
            {"prompt": prompt, "schema_name": schema_name, **kwargs}
        )
        if schema_name == "DeepReviewVerdict":
            match = re.search(r'"artifact_id": "([^"]+)"', prompt)
            assert match, prompt
            self.canvas_id = match.group(1)
            parsed = {
                "complete": True,
                "grounded": True,
                "contradictions_named": False,
                "findings": [
                    {
                        "target": "artifact",
                        "artifact_id": self.canvas_id,
                        "finding": "Canvas nennt die Unsicherheit nicht.",
                    }
                ],
            }
        else:
            parsed = {
                "chat_markdown": "Chat bleibt unveraendert.",
                "artifacts": [
                    {
                        "artifact_id": self.canvas_id,
                        "expected_revision": 1,
                        "markdown": "# Canvas\n\nUnsicherheit benannt.",
                    }
                ],
            }
        return SimpleNamespace(parsed=parsed, prompt_tokens=7, completion_tokens=3)


def _canvas_turn() -> ChatTurn:
    return ChatTurn(
        text="",
        tool_calls=(
            ToolCallRequest(
                id="call_canvas_deep",
                name="write_canvas",
                arguments={
                    "title": "Canvas",
                    "content_markdown": "# Canvas\n\nUnsicherheit fehlt.",
                },
            ),
        ),
        finish_reason="tool_calls",
        model="high-model",
        prompt_tokens=10,
        completion_tokens=5,
        raw=None,
    )


def test_deep_run_reviews_and_revises_once():
    llm = StructuredScriptedLLM(
        [_text_turn("Erste Antwort ohne Belege.")],
        [
            {
                "complete": True,
                "grounded": False,
                "contradictions_named": True,
                "findings": [
                    {
                        "target": "chat",
                        "artifact_id": "",
                        "finding": "Faktische Aussagen tragen keine Belege.",
                    }
                ],
            },
            {
                "chat_markdown": "Ueberarbeitete Antwort mit Beleg [W1].",
                "artifacts": [],
            },
        ],
    )
    client = make_client(llm)
    with client:
        run = client.post(
            "/v1/runs",
            json={
                "question": "Wie ist die Lage?",
                "mode": "agent_kernel",
                "agent_overrides": {"depth": "deep"},
            },
        ).json()
        summary = wait_status(client, run["run_id"], {"completed", "failed"})
        assert summary["status"] == "completed", summary
        result = client.get(f"/v1/runs/{run['run_id']}/result").json()
        assert result["answer"] == "Ueberarbeitete Antwort mit Beleg [W1]."
        # Exactly review + revision, in that order, on the right models:
        # the rubric check runs mid-tier, the revision on the kernel's
        # own high-tier resolution (with the deep effort default).
        assert [c["schema_name"] for c in llm.structured_calls] == [
            "DeepReviewVerdict",
            "DeepRevisionBundle",
        ]
        review, revision = llm.structured_calls
        assert review["model"] == "mid-model"
        assert "Rubrik" in review["prompt"]
        assert revision["model"] == "high-model"
        assert revision["reasoning_effort"] == "high"
        assert "Faktische Aussagen tragen keine Belege." in revision["prompt"]
        # The deep prompt line reached the model's user message.
        first_user = str(llm.chat_calls[0]["messages"])
        assert "Deep-Modus" in first_user
        events = client.get(
            f"/v1/runs/{run['run_id']}/events",
            params={"format": "json"},
        ).json()["data"]
        review_narrations = [
            event
            for event in events
            if event.get("data", {}).get("narration_id")
            == "kernel_deep_review"
        ]
        # Same narration_id twice (clients upsert): first the running
        # state, then the actual OUTCOME — never a promised revision
        # that silently failed.
        assert [n["data"]["final"] for n in review_narrations] == [
            False,
            False,
            True,
        ]
        assert "1 Befund" in review_narrations[1]["data"]["text"]
        assert "ueberarbeitet" in review_narrations[2]["data"]["text"]


def test_deep_revision_receives_every_review_finding():
    findings = [
        {
            "target": "chat",
            "artifact_id": "",
            "finding": f"Befund Nummer {index}",
        }
        for index in range(9)
    ]
    llm = StructuredScriptedLLM(
        [_text_turn("Erste Antwort.")],
        [
            {
                "complete": False,
                "grounded": False,
                "contradictions_named": False,
                "findings": findings,
            },
            {"chat_markdown": "Alle neun Befunde behoben.", "artifacts": []},
        ],
    )
    client = make_client(llm)
    with client:
        run = client.post(
            "/v1/runs",
            json={
                "question": "Pruefe alles.",
                "mode": "agent_kernel",
                "agent_overrides": {"depth": "deep"},
            },
        ).json()
        summary = wait_status(client, run["run_id"], {"completed", "failed"})
        assert summary["status"] == "completed", summary
        assert "Befund Nummer 8" in llm.structured_calls[1]["prompt"]


def test_deep_run_without_findings_keeps_the_answer():
    llm = StructuredScriptedLLM(
        [_text_turn("Belegte Antwort [W1].")],
        [
            {
                "complete": True,
                "grounded": True,
                "contradictions_named": True,
                "findings": [],
            }
        ],
    )
    client = make_client(llm)
    with client:
        run = client.post(
            "/v1/runs",
            json={
                "question": "Kurzfrage?",
                "mode": "agent_kernel",
                "agent_overrides": {"depth": "deep"},
            },
        ).json()
        summary = wait_status(client, run["run_id"], {"completed", "failed"})
        assert summary["status"] == "completed", summary
        result = client.get(f"/v1/runs/{run['run_id']}/result").json()
        assert result["answer"] == "Belegte Antwort [W1]."
        # Review ran, revision did NOT.
        assert [c["schema_name"] for c in llm.structured_calls] == [
            "DeepReviewVerdict"
        ]


def test_deep_negative_verdict_without_findings_is_fail_closed():
    llm = StructuredScriptedLLM(
        [_text_turn("Bestehende Antwort.")],
        [
            {
                "complete": False,
                "grounded": True,
                "contradictions_named": True,
                "findings": [],
            }
        ],
    )
    client = make_client(llm)
    with client:
        run = client.post(
            "/v1/runs",
            json={
                "question": "Pruefe vollstaendig.",
                "mode": "agent_kernel",
                "agent_overrides": {"depth": "deep"},
            },
        ).json()
        summary = wait_status(client, run["run_id"], {"completed", "failed"})
        assert summary["status"] == "completed", summary
        result = client.get(f"/v1/runs/{run['run_id']}/result").json()
        assert result["answer"] == "Bestehende Antwort."
        assert [call["schema_name"] for call in llm.structured_calls] == [
            "DeepReviewVerdict"
        ]
        events = client.get(
            f"/v1/runs/{run['run_id']}/events",
            params={"format": "json"},
        ).json()["data"]
        terminal = [
            item["data"]["text"]
            for item in events
            if item.get("data", {}).get("narration_id")
            == "kernel_deep_review"
            and item["data"].get("final")
        ]
        assert any("inkonsistentes Review" in text for text in terminal)


def test_deep_chat_finding_with_unchanged_revision_is_rejected():
    llm = StructuredScriptedLLM(
        [_text_turn("Unveraendert.")],
        [
            {
                "complete": False,
                "grounded": True,
                "contradictions_named": True,
                "findings": [
                    {
                        "target": "chat",
                        "artifact_id": "",
                        "finding": "Ein Teil fehlt.",
                    }
                ],
            },
            {"chat_markdown": "Unveraendert.", "artifacts": []},
        ],
    )
    client = make_client(llm)
    with client:
        run = client.post(
            "/v1/runs",
            json={
                "question": "Vollstaendig antworten.",
                "mode": "agent_kernel",
                "agent_overrides": {"depth": "deep"},
            },
        ).json()
        summary = wait_status(client, run["run_id"], {"completed", "failed"})
        assert summary["status"] == "completed", summary
        result = client.get(f"/v1/runs/{run['run_id']}/result").json()
        assert result["answer"] == "Unveraendert."
        events = client.get(
            f"/v1/runs/{run['run_id']}/events",
            params={"format": "json"},
        ).json()["data"]
        assert any(
            "ohne Ergebnis" in item.get("data", {}).get("text", "")
            and item.get("data", {}).get("final")
            for item in events
        )


def test_deep_review_receives_effective_source_policy():
    llm = StructuredScriptedLLM(
        [_text_turn("Antwort.")],
        [
            {
                "complete": True,
                "grounded": True,
                "contradictions_named": True,
                "findings": [],
            }
        ],
    )
    client = make_client(llm)
    with client:
        run = client.post(
            "/v1/runs",
            json={
                "question": "Nur intern.",
                "mode": "agent_kernel",
                "source_policy": {
                    "web": "disabled",
                    "knowledge": "available",
                },
                "agent_overrides": {"depth": "deep"},
            },
        ).json()
        summary = wait_status(client, run["run_id"], {"completed", "failed"})
        assert summary["status"] == "completed", summary
        prompt = llm.structured_calls[0]["prompt"]
        assert '"web": "disabled"' in prompt
        assert '"knowledge": "available"' in prompt


def test_deep_canvas_finding_updates_the_target_with_batch_cas():
    llm = CanvasReviewLLM(
        [_canvas_turn(), _text_turn("Chat bleibt unveraendert.")],
        [],
    )
    client = make_client(llm)
    with client:
        run = client.post(
            "/v1/runs",
            json={
                "question": "Erstelle und pruefe ein Canvas.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "session_id": "sess-deep-canvas",
                "agent_overrides": {"depth": "deep"},
            },
        ).json()
        summary = wait_status(client, run["run_id"], {"completed", "failed"})
        assert summary["status"] == "completed", summary
        artifacts = client.get(f"/v1/runs/{run['run_id']}/artifacts").json()["data"]
        canvas = next(item for item in artifacts if item["kind"] == "deliverable")
        detail = client.get(
            f"/v1/runs/{run['run_id']}/artifacts/{canvas['artifact_id']}"
        ).json()
        assert detail["revision"] == 2
        assert "Unsicherheit benannt" in detail["content_markdown"]
        review_prompt = llm.structured_calls[0]["prompt"]
        assert "Unsicherheit fehlt" in review_prompt
        assert "Vollstaendiger effektiver Auftrag" in review_prompt


def test_deep_store_failure_has_terminal_narration_and_preserves_outputs():
    llm = CanvasReviewLLM(
        [_canvas_turn(), _text_turn("Chat bleibt unveraendert.")],
        [],
    )
    client = make_client(llm)

    async def fail_store(**kwargs):
        del kwargs
        raise RuntimeError("store offline")

    client.container.agent_control_service.store.revise_session_artifacts_atomically = (  # type: ignore[attr-defined]
        fail_store
    )
    with client:
        run = client.post(
            "/v1/runs",
            json={
                "question": "Erstelle und pruefe ein Canvas.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "session_id": "sess-deep-store-failure",
                "agent_overrides": {"depth": "deep"},
            },
        ).json()
        summary = wait_status(client, run["run_id"], {"completed", "failed"})
        assert summary["status"] == "completed", summary
        result = client.get(f"/v1/runs/{run['run_id']}/result").json()
        assert result["answer"] == "Chat bleibt unveraendert."
        events = client.get(
            f"/v1/runs/{run['run_id']}/events", params={"format": "json"}
        ).json()["data"]
        assert any(
            "Storefehler" in item.get("data", {}).get("text", "")
            and item.get("data", {}).get("final")
            for item in events
        )


def test_normal_run_never_touches_the_review_path():
    llm = StructuredScriptedLLM([_text_turn("Direkte Antwort.")], [])
    client = make_client(llm)
    with client:
        run = client.post(
            "/v1/runs",
            json={"question": "Frage?", "mode": "agent_kernel"},
        ).json()
        summary = wait_status(client, run["run_id"], {"completed", "failed"})
        assert summary["status"] == "completed", summary
        assert llm.structured_calls == []
        first_user = str(llm.chat_calls[0]["messages"])
        assert "Deep-Modus" not in first_user


def test_deep_prompt_line_and_user_message_composition():
    deep = build_kernel_user_message("Auftrag.", depth="deep")
    assert "Deep-Modus" in deep
    assert "run_deep_mission" in deep
    normal = build_kernel_user_message("Auftrag.")
    assert "Deep-Modus" not in normal


def test_deep_canvas_bundle_paginates_beyond_fifty_outputs():
    rows = [SimpleNamespace(artifact_id=f"art_{index}") for index in range(51)]

    class PagedControl:
        async def list_artifacts(self, run_id, *, kind, limit, after=None):
            assert run_id == "run_deep"
            assert kind == "deliverable"
            assert limit == 50
            if after is None:
                return rows[:50], encode_cursor(1.0, "art_49")
            assert after == (1.0, "art_49")
            return rows[50:], None

        async def get_artifact(self, run_id, artifact_id):
            assert run_id == "run_deep"
            return SimpleNamespace(artifact_id=artifact_id), []

    algorithm = object.__new__(KernelAgentAlgorithm)
    algorithm._control = PagedControl()
    canvases = algorithm._current_run_canvases(
        SimpleNamespace(run_id="run_deep")
    )

    assert [item.artifact_id for item in canvases] == [
        f"art_{index}" for index in range(51)
    ]


def test_deep_child_research_uses_the_deep_wire_profile(monkeypatch):
    """Deep forces the DEEP report profile on child research — as the
    LITERAL ReportProfile value ('deep'); the German label 'gruendlich'
    would 400 the child resolve and fail the whole run (review F1)."""
    from tests.agents.test_kernel_child_runs import (
        ScriptedToolLLM as ChildScriptedLLM,
        _text_turn as child_text_turn,
        _tool_turn as child_tool_turn,
        make_client as make_child_client,
        wait_status as wait_child_status,
    )

    llm = ChildScriptedLLM(
        [
            child_tool_turn(
                "call_deepchild",
                "run_web_research",
                {"question": "Marktlage 2026"},
            ),
            child_text_turn("Fertig auf Basis des Kindberichts."),
        ]
    )
    client = make_child_client(monkeypatch, llm)
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Analysiere die Marktlage gruendlich.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "session_id": "sess-deep-child",
                "agent_overrides": {"depth": "deep"},
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        summary = wait_child_status(client, run_id, {"completed", "failed"})
        assert summary["status"] == "completed", summary
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 1
        assert children[0]["agent_overrides"]["report_profile"] == "deep"
