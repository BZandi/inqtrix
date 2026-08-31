"""Full-path contracts for Agent Desk sources and one-shot directives."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("deepagents")

from inqtrix.providers.base import LLMProvider, LLMResponse
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


class QuickWebLLM(LLMProvider):
    """Two-call quick-web provider recording model/effort propagation."""

    def __init__(
        self,
        *,
        answer_text: str = "Die Mannschaft gewann heute das Finale.",
    ) -> None:
        self.models = ModelSettings(
            reasoning_model="base-model",
            tier_high_model="high-model",
            tier_mid_model="mid-model",
            tier_fast_model="fast-model",
        )
        self.calls: list[dict[str, Any]] = []
        self.answer_text = answer_text

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.complete_with_metadata(prompt, **kwargs).content

    def complete_with_metadata(
        self, prompt: str, **kwargs: Any
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "model": kwargs.get("model"),
                "effort": kwargs.get("reasoning_effort"),
            }
        )
        content = (
            '{"query":"FIFA Weltmeisterschaft Sieger heute",'
            '"recency":"day"}'
            if "Formuliere genau EINE" in prompt
            else self.answer_text
        )
        return LLMResponse(
            content=content,
            prompt_tokens=11,
            completion_tokens=7,
            model=str(kwargs.get("model") or ""),
        )

    def is_available(self) -> bool:
        return True

    def supports_tool_calls(self, *, model: str | None = None) -> bool:
        return True


class RetryNotifyingQuickWebLLM(QuickWebLLM):
    """Expose one retry from the grounded-answer model call."""

    def __init__(self) -> None:
        super().__init__()
        self._retry_callback: Any = None

    @contextmanager
    def observe_retries(self, callback: Any) -> Any:
        previous = self._retry_callback
        self._retry_callback = callback
        try:
            yield self
        finally:
            self._retry_callback = previous

    def complete_with_metadata(
        self, prompt: str, **kwargs: Any
    ) -> LLMResponse:
        if len(self.calls) == 1 and self._retry_callback is not None:
            self._retry_callback({
                "attempt": 1,
                "max_attempts": 3,
                "delay_seconds": 0.1,
                "error_code": "provider_timeout",
                "operation": "complete",
            })
        return super().complete_with_metadata(prompt, **kwargs)


class CountingSearch:
    """Instant-search stub whose call count is the route invariant."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
        self.calls.append({"query": query, **kwargs})
        return GroundedSearchResult(
            answer="Grounded provider answer.",
            sources=[
                GroundedSource(
                    url="https://example.test/final",
                    title="Final result",
                    snippet="The team won the final today.",
                    rank=1,
                )
            ],
        )

    def is_available(self) -> bool:
        return True


def _client(llm: QuickWebLLM, search: CountingSearch) -> TestClient:
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
    app = FastAPI()
    router = create_router()
    register_routes(
        router,
        providers=SimpleNamespace(llm=llm, search=search),
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(2),
    )
    app.include_router(router)
    return TestClient(app)


def _wait_status(
    client: TestClient,
    run_id: str,
    statuses: set[str],
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        summary = client.get(f"/v1/runs/{run_id}").json()
        if summary["status"] in statuses:
            return summary
        time.sleep(0.02)
    pytest.fail(f"run {run_id} never reached {statuses}")


def _run_events(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    with client.stream("GET", f"/v1/runs/{run_id}/events") as stream:
        body = stream.read().decode("utf-8")
    return [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _quick_body(*, autonomy: str = "balanced") -> dict[str, Any]:
    return {
        "question": "Wer gewann heute das Finale?",
        # The directive, not this stale UI selection, owns routing.
        "mode": "workspace_agent",
        "autonomy": autonomy,
        "response_form": "canvas",
        "agent_overrides": {
            "depth": "deep",
            "model": "picked-model",
            "effort": "high",
        },
        # A one-shot directive temporarily overrides the session source
        # preference without mutating it in the stored request summary.
        "source_policy": {"web": "disabled", "knowledge": "available"},
        "execution_directive": "quick_web",
    }


def test_quick_web_is_one_search_without_plan_child_rag_or_canvas() -> None:
    llm = QuickWebLLM()
    search = CountingSearch()
    with _client(llm, search) as client:
        response = client.post("/v1/runs", json=_quick_body())
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        summary = _wait_status(client, run_id, {"completed"})

        assert summary["mode"] == "agent_kernel"
        assert summary["agent_overrides"]["depth"] == "normal"
        assert summary["agent_overrides"]["response_form"] == "chat"
        assert summary["agent_overrides"]["source_policy"] == {
            "web": "disabled",
            "knowledge": "available",
        }
        assert summary["agent_overrides"]["execution_directive"] == "quick_web"
        assert len(search.calls) == 1
        assert search.calls[0]["recency_filter"] == "day"
        assert client.get(f"/v1/runs/{run_id}/children").json()["data"] == []
        assert client.get(f"/v1/runs/{run_id}/plan").status_code == 404

        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"].startswith("Die Mannschaft gewann")
        assert result["references"][0]["url"] == "https://example.test/final"
        expected_execution = {
            "execution_directive": "quick_web",
            "effective_mode": "agent_kernel",
            "response_form": "chat",
            "depth": "normal",
            "model": "picked-model",
            "reasoning_effort": "high",
            "source_policy": {
                "web": "available",
                "knowledge": "disabled",
            },
            "consent_reason": "explicit_directive",
            "tool_use_counts": {"web": 1, "knowledge": 0},
            "limits": {
                "web_searches": {
                    "used": 1,
                    "limit": 1,
                    "ceiling": 1,
                    "recoverable": False,
                    "extendable": False,
                    "reason": "direct_route_single_search",
                }
            },
            "tool_grants": [],
        }
        assert summary["snapshot"]["execution"] == expected_execution
        assert result["execution"] == expected_execution
        # Query derivation and grounded synthesis use the selected model.
        assert [call["model"] for call in llm.calls] == [
            "picked-model",
            "picked-model",
        ]
        assert [call["effort"] for call in llm.calls] == ["high", "high"]


def test_quick_web_preserves_provider_grounded_numbers_and_sources() -> None:
    llm = QuickWebLLM(
        answer_text="Der Listenpreis beträgt 1.50 USD pro 1M Token."
    )
    search = CountingSearch()
    body = {
        **_quick_body(),
        "question": "Was kostet das Modell exakt pro 1M Token?",
    }

    with _client(llm, search) as client:
        response = client.post("/v1/runs", json=body)
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        _wait_status(client, run_id, {"completed"})

        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert "1.50" in result["answer"]
        assert "[unbelegt: harte Aussage]" not in result["answer"]
        assert [item["url"] for item in result["references"]] == [
            "https://example.test/final"
        ]

    answer_prompt = llm.calls[1]["prompt"]
    assert "einschliesslich darin genannter Zahlen, Preise und Daten" in answer_prompt
    assert "entferne vorhandene Providerinformationen nicht" in answer_prompt
    assert "Grounded provider answer." in answer_prompt


def test_quick_web_answer_retries_use_agent_activity_channel() -> None:
    llm = RetryNotifyingQuickWebLLM()
    with _client(llm, CountingSearch()) as client:
        response = client.post("/v1/runs", json=_quick_body())
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        _wait_status(client, run_id, {"completed"})
        retries = [
            event["data"]
            for event in _run_events(client, run_id)
            if event["type"] == "inqtrix.agent.activity"
            and event["data"].get("retry")
        ]

    assert len(retries) == 1
    assert retries[0]["purpose"] == "Direkte Webantwort wird formuliert"


def test_quick_web_follow_up_uses_server_built_session_history() -> None:
    llm = QuickWebLLM()
    search = CountingSearch()
    with _client(llm, search) as client:
        first_body = {
            **_quick_body(),
            "question": "Wer gewann das Finale?",
            "session_id": "sess-quick-follow-up",
        }
        first = client.post("/v1/runs", json=first_body)
        assert first.status_code == 202, first.text
        _wait_status(client, first.json()["run_id"], {"completed"})

        llm.calls.clear()
        follow_up = client.post(
            "/v1/runs",
            json={
                **_quick_body(),
                "question": "Und wie war es heute?",
                "session_id": "sess-quick-follow-up",
            },
        )
        assert follow_up.status_code == 202, follow_up.text
        _wait_status(client, follow_up.json()["run_id"], {"completed"})

        query_prompt = llm.calls[0]["prompt"]
        answer_prompt = llm.calls[1]["prompt"]
        assert "Nutzer: Wer gewann das Finale?" in query_prompt
        assert "Agent: Die Mannschaft gewann" in query_prompt
        assert "Nutzer: Wer gewann das Finale?" in answer_prompt


def test_strict_quick_web_gates_reviewed_query_then_runs_once() -> None:
    llm = QuickWebLLM()
    search = CountingSearch()
    with _client(llm, search) as client:
        response = client.post(
            "/v1/runs", json=_quick_body(autonomy="strict")
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        pending = _wait_status(client, run_id, {"waiting_for_approval"})
        assert pending["snapshot"]["execution"]["consent_reason"] == (
            "strict_approval_required"
        )
        assert pending["snapshot"]["execution"]["tool_use_counts"] == {
            "web": 0,
            "knowledge": 0,
        }
        assert search.calls == []
        approvals = client.get(
            f"/v1/runs/{run_id}/approvals"
        ).json()["data"]
        assert len(approvals) == 1
        action = approvals[0]["payload"]["actions"][0]
        assert action["tool"] == "web_instant"
        assert action["args"]["query"] == "FIFA Weltmeisterschaft Sieger heute"

        decided = client.post(
            f"/v1/runs/{run_id}/approvals/{approvals[0]['approval_id']}",
            json={"decision": "approve"},
        )
        assert decided.status_code == 200, decided.text
        completed = _wait_status(client, run_id, {"completed"})
        assert len(search.calls) == 1
        assert completed["snapshot"]["execution"]["consent_reason"] == (
            "strict_approval"
        )
        assert completed["snapshot"]["execution"]["tool_use_counts"] == {
            "web": 1,
            "knowledge": 0,
        }
        # Resume reuses the persisted reviewed query; it is not regenerated.
        assert len(llm.calls) == 2


def test_strict_quick_web_edit_runs_the_edited_query() -> None:
    """Editing the gated quick-web query resumes with the EDITED query.

    Regression: the approval action once carried a spurious ``recency`` arg
    that ``web_instant`` (query-only) does not accept, so ``decision: edit``
    was rejected as "Unbekannte Tool-Felder: recency" and the edit path never
    worked end-to-end. recency now rides in the approval payload, so an
    args-only edit re-validates cleanly against the tool schema.
    """
    llm = QuickWebLLM()
    search = CountingSearch()
    with _client(llm, search) as client:
        run_id = client.post(
            "/v1/runs", json=_quick_body(autonomy="strict")
        ).json()["run_id"]
        _wait_status(client, run_id, {"waiting_for_approval"})
        approval = client.get(
            f"/v1/runs/{run_id}/approvals"
        ).json()["data"][0]
        # recency is NOT a web_instant tool arg — the gate exposes only query.
        assert "recency" not in approval["payload"]["actions"][0]["args"]

        decided = client.post(
            f"/v1/runs/{run_id}/approvals/{approval['approval_id']}",
            json={
                "decision": "edit",
                "actions": [
                    {"tool": "web_instant", "args": {"query": "EU AI Act Governance-Pflichten 2026"}}
                ],
            },
        )
        assert decided.status_code == 200, decided.text

        _wait_status(client, run_id, {"completed"})
        assert len(search.calls) == 1
        assert search.calls[0]["query"] == "EU AI Act Governance-Pflichten 2026"
        # The edited query kept the originally-derived recency (payload-carried).
        assert search.calls[0]["recency_filter"] == "day"


def test_capabilities_and_legacy_directive_conflict_are_explicit() -> None:
    with _client(QuickWebLLM(), CountingSearch()) as client:
        agent = client.get("/v1/capabilities").json()["agent"]
        assert agent["source_controls"] == [
            {"id": "web", "default": "available", "available": True},
            {"id": "knowledge", "default": "available", "available": False},
        ]
        assert agent["execution_directives"] == [
            {"id": "quick_web", "available": True},
            {"id": "knowledge_only", "available": False},
        ]

        response = client.post(
            "/v1/runs",
            json={
                "question": "x",
                "execution_directive": "quick_web",
                "tool_directives": [],
            },
        )
        assert response.status_code == 400
        assert "nicht gleichzeitig" in response.json()["error"]["message"]


def test_the_quick_lane_refuses_a_result_requirement_it_cannot_honor() -> None:
    """The quick lane returns from `_run_quick_web` before the kernel
    user message is ever built (kernel/algorithm.py: the early return at
    `execution_directive == "quick_web"`), so a requirement sent with it
    would be validated, composed, persisted — and never reach a single
    prompt. Same refusal as canvas_context, for the same reason: a silent
    drop is exactly what this feature exists to prevent."""
    with _client(QuickWebLLM(), CountingSearch()) as client:
        body = _quick_body()
        body["report_guidance"] = "Antworte in genau drei Stichpunkten."
        response = client.post("/v1/runs", json=body)
        assert response.status_code == 400
        message = response.json()["error"]["message"]
        assert "report_guidance" in message
        assert "execution_directive" in message
