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
    # The child owns a complete V2 evidence ledger. Its presentation answer
    # cites the child label, but only the independently bound claim may cross
    # into the parent report.
    claim_text = f"Kindbericht zu: {question}"
    source_url = "https://www.bafin.de/SharedDocs/Tests/quelle.html"
    child_run_id = str(kwargs.get("run_id") or f"run_{abs(hash(question))}")
    return {
        "answer": f"{claim_text} [E1]",
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        "result_state": {
            "_run_id": child_run_id,
            "question": question,
            "answer": f"{claim_text} [E1]",
            "round": 1,
            "query_records": [
                {
                    "invocation_id": "invocation_1",
                    "query_id": "query_1",
                    "round": 0,
                    "query": question,
                    "provider": "StubSearch",
                }
            ],
            "source_records": {
                "source_1": {
                    "source_id": "source_1",
                    "url": source_url,
                    "canonical_url": source_url,
                    "domain": "bafin.de",
                    "tier": "primary",
                    "tier_reason": "test_regulator",
                }
            },
            "provider_citation_records": [
                {
                    "citation_id": "citation_1",
                    "query_id": "query_1",
                    "source_id": "source_1",
                    "canonical_url": source_url,
                    "title": "Primaerquelle",
                    "snippet": "",
                }
            ],
            "evidence_ledger": [
                {
                    "evidence_id": "evidence_1",
                    "source_id": "source_1",
                    "claims": [
                        {
                            "raw_claim_id": "raw_claim_1",
                            "claim_text": claim_text,
                            "evidence_snippet": claim_text,
                            "claim_type": "fact",
                            "needs_primary": True,
                            "source_ids": ["source_1"],
                            "source_urls": [source_url],
                        }
                    ],
                }
            ],
            "consolidated_claims_full": [
                {
                    "claim_id": "claim_1",
                    "claim_text": claim_text,
                    "claim_type": "fact",
                    "needs_primary": True,
                    "member_claim_ids": ["raw_claim_1"],
                    "status": "verified",
                }
            ],
            "required_aspects": ["Kindbericht"],
            "uncovered_aspects": [],
            "gaps": "",
            "_stop_reason": "confidence_reached",
            "report_references": [
                {
                    "label": "E1",
                    "url": source_url,
                    "title": "Primaerquelle",
                    "tier": "primary",
                }
            ],
        },
    }


def make_client(
    monkeypatch: pytest.MonkeyPatch,
    llm: ScriptedToolLLM,
    *,
    max_iterations: int | None = None,
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
    platform_kwargs: dict[str, Any] = dict(
        INQTRIX_AGENT_ALLOW_VOLATILE=True,
        INQTRIX_AGENT_KERNEL_ENABLED=True,
    )
    if max_iterations is not None:
        platform_kwargs["INQTRIX_AGENT_KERNEL_MAX_ITERATIONS"] = (
            max_iterations
        )
    settings.agent_platform = AgentPlatformSettings(**platform_kwargs)
    container = register_routes(
        router,
        providers=SimpleNamespace(llm=llm, search=None),
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: __import__("asyncio").Semaphore(4),
    )
    app.include_router(router)
    client = TestClient(app)
    client.container = container  # type: ignore[attr-defined]
    return client


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
            == "https://www.bafin.de/SharedDocs/Tests/quelle.html"
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
        # F5: the merged child ref surfaces its citable parent label, the
        # renumbering is visible, and the child TEXT's citations are
        # translated IN THE TOOL (a verbatim-copied old label would
        # resolve to the wrong parent source — the model is never
        # trusted with the mapping).
        content = tool_replies[0]["content"]
        assert "Beleg [W1] — reference_id: ref_" in content
        assert "[E1] -> [W1]" in content
        assert "Marktlage Klimaanlagen 2026 [W1]" in content
        assert "2026 [E1]" not in content


def test_child_report_preserves_provider_result_for_parent_synthesis(
    monkeypatch,
):
    false_claim = "Input costs 0.90 USD per 1M tokens."
    source_url = "https://prices.azure.com/api/retail/prices"

    def false_numeric_child_graph(
        question: str, **kwargs: Any
    ) -> dict[str, Any]:
        child_run_id = str(kwargs.get("run_id") or "run_false_numeric")
        return {
            "answer": f"{false_claim} [E1]",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "result_state": {
                "_run_id": child_run_id,
                "question": question,
                "answer": f"{false_claim} [E1]",
                "round": 1,
                "query_records": [
                    {
                        "invocation_id": "invocation_false",
                        "query_id": "query_false",
                        "round": 0,
                        "query": question,
                        "provider": "StubSearch",
                    }
                ],
                "source_records": {
                    "source_false": {
                        "source_id": "source_false",
                        "url": source_url,
                        "canonical_url": source_url,
                        "domain": "prices.azure.com",
                        "tier": "primary",
                        "tier_reason": "test_first_party",
                    }
                },
                "provider_citation_records": [
                    {
                        "citation_id": "citation_false",
                        "query_id": "query_false",
                        "source_id": "source_false",
                        "canonical_url": source_url,
                        "title": "Azure Retail Prices",
                        "snippet": false_claim,
                    }
                ],
                "evidence_ledger": [
                    {
                        "evidence_id": "evidence_false",
                        "source_id": "source_false",
                        "claims": [
                            {
                                "raw_claim_id": "raw_false",
                                "claim_text": false_claim,
                                "evidence_snippet": false_claim,
                                "claim_type": "fact",
                                "needs_primary": True,
                                "source_ids": ["source_false"],
                                "source_urls": [source_url],
                            }
                        ],
                    }
                ],
                "consolidated_claims_full": [
                    {
                        "claim_id": "claim_false",
                        "claim_text": false_claim,
                        "claim_type": "fact",
                        "needs_primary": True,
                        "member_claim_ids": ["raw_false"],
                        "status": "unverified",
                    }
                ],
                "required_aspects": ["Input price"],
                "uncovered_aspects": ["Input price"],
                "gaps": "Original price row missing",
                "_stop_reason": "max_rounds",
                "report_references": [
                    {
                        "label": "E1",
                        "url": source_url,
                        "title": "Azure Retail Prices",
                        "tier": "primary",
                    }
                ],
            },
        }

    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_false_numeric",
                "run_web_research",
                {"question": "What is the exact model price?"},
            ),
            _text_turn("Die Recherche ist noch nicht ausreichend belegt."),
        ]
    )
    client = make_client(monkeypatch, llm)
    monkeypatch.setattr(
        web_research_module,
        "run_web_graph",
        false_numeric_child_graph,
    )
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Prüfe den exakten Modellpreis.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "tool_directives": ["web_research"],
            },
        )
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"completed"})
        tool_content = next(
            message["content"]
            for message in llm.chat_calls[1]["messages"]
            if message.get("role") == "tool"
        )
        result = client.get(f"/v1/runs/{run_id}/result").json()
        evidence = client.get(
            f"/v1/runs/{run_id}/artifacts/"
            f"art_{run_id[-12:]}_evidence"
        ).json()

    assert false_claim in tool_content
    assert '<unvertrauenswuerdiger_inhalt quelle="unterauftrag">' in tool_content
    assert "[W1]" in tool_content
    assert "Azure Retail Prices" in tool_content
    assert result["references"][0]["url"] == source_url
    assert result["references"][0]["query_id"] == "query_false"
    assert result["references"][0]["citation_id"] == "citation_false"
    ledger = evidence["payload"]["web_search_ledger"]
    search = ledger["searches"]["query_false"]
    assert search["query"] == "What is the exact model price?"
    assert search["provider"] == "StubSearch"
    assert search["citations"][0]["snippet"] == false_claim
    assert search["citations"][0]["url"] == source_url


def test_legacy_child_report_reaches_parent_without_page_read_contract(
    monkeypatch,
):
    legacy_answer = "Legacy answer claims the price is 0.90 USD. [E1]"
    legacy_url = "https://example.com/legacy-price"

    def legacy_child_graph(
        question: str, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "answer": legacy_answer,
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "result_state": {
                "answer": legacy_answer,
                "round": 1,
                "report_references": [
                    {
                        "label": "E1",
                        "url": legacy_url,
                        "title": "Legacy price page",
                        "tier": "primary",
                    }
                ],
                "consolidated_claims": [
                    {
                        "claim_text": "The price is 0.90 USD.",
                        "status": "verified",
                    }
                ],
            },
        }

    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_legacy",
                "run_web_research",
                {"question": "Legacy price lookup"},
            ),
            _text_turn("Für diese Recherche fehlt ein Evidenzvertrag."),
        ]
    )
    client = make_client(monkeypatch, llm)
    monkeypatch.setattr(
        web_research_module, "run_web_graph", legacy_child_graph
    )
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Prüfe den Legacy-Preis.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "tool_directives": ["web_research"],
            },
        )
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"completed"})
        tool_content = next(
            message["content"]
            for message in llm.chat_calls[1]["messages"]
            if message.get("role") == "tool"
        )
        result = client.get(f"/v1/runs/{run_id}/result").json()
        artifacts = client.get(
            f"/v1/runs/{run_id}/artifacts"
        ).json()["data"]

    assert "Legacy answer claims the price is 0.90 USD. [W1]" in tool_content
    assert "Legacy price page" in tool_content
    assert '<unvertrauenswuerdiger_inhalt quelle="unterauftrag">' in tool_content
    assert result["references"][0]["url"] == legacy_url
    assert result["references"][0]["title"] == "Legacy price page"
    assert any(
        artifact["kind"] == "evidence_bundle" for artifact in artifacts
    )


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


def test_normal_kernel_can_adaptively_run_shared_research_child(monkeypatch):
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_unrequested",
                "run_web_research",
                {"question": "Ungefragter Recherchelauf"},
            ),
            _text_turn("Die Recherche wurde genutzt."),
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
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 1
        tool_replies = [
            message
            for message in llm.chat_calls[1]["messages"]
            if message.get("role") == "tool"
        ]
        assert "Werkzeug blockiert" not in tool_replies[0]["content"]


# -- Phase 7: delegate_batch (parallel fan-out) --------------------------- #


def test_delegate_batch_submits_all_before_single_park(monkeypatch):
    """Both children exist BEFORE the one park; results come back in
    assignment order as condensed (<=300-word) summaries with sources."""
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_batch1",
                "delegate_batch",
                {
                    "assignments": [
                        {"objective": "Strang A untersuchen", "mode": "research"},
                        {"objective": "Strang B untersuchen", "mode": "research"},
                    ]
                },
            ),
            _text_turn("Antwort auf Basis beider Unterauftraege."),
        ]
    )
    client = make_client(monkeypatch, llm)
    with client:
        client.headers["x-inqtrix-workspace-id"] = "ws_batch"
        response = client.post(
            "/v1/runs",
            json={
                "question": "Analysiere beide Straenge.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "session_id": "sess-batch",
                "tool_directives": ["web_research"],
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]

        wait_status(client, run_id, {"completed"})
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 2
        by_origin = {child["origin_key"]: child for child in children}
        assert set(by_origin) == {"call_batch1:0", "call_batch1:1"}
        # (Autonomy inheritance is a submit-side contract shared with the
        # single-child tools; the summary does not expose the field.)

        tool_replies = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ]
        content = tool_replies[0]["content"]
        # Assignment order, one block per child, condensed summary text.
        first = content.find("## Unterauftrag 1")
        second = content.find("## Unterauftrag 2")
        assert 0 <= first < second, content[:300]
        assert "Kindbericht zu: Strang A untersuchen" in content
        assert "Kindbericht zu: Strang B untersuchen" in content


def _batch_with_preparation_turns() -> list[ChatTurn]:
    """A realistic single-segment batch trajectory: the model plans and
    refines todos, then fans out — all BEFORE the one park. This is the
    live failure class of the recursion recalibration: preparation turns
    plus the fan-out turn in one segment."""
    plan = [
        {"content": "Straenge identifizieren", "status": "in_progress"},
        {"content": "Fan-out starten", "status": "pending"},
        {"content": "Synthese schreiben", "status": "pending"},
    ]
    refined = [dict(plan[0], status="completed"), plan[1], plan[2]]
    started = [refined[0], dict(plan[1], status="in_progress"), plan[2]]
    return [
        _tool_turn("call_prep1", "write_todos", {"todos": plan}),
        _tool_turn("call_prep2", "write_todos", {"todos": refined}),
        _tool_turn("call_prep3", "write_todos", {"todos": started}),
        _tool_turn(
            "call_batch3",
            "delegate_batch",
            {
                "assignments": [
                    {"objective": "Strang A untersuchen", "mode": "research"},
                    {"objective": "Strang B untersuchen", "mode": "research"},
                ]
            },
        ),
        _text_turn("Synthese aus beiden Unterauftraegen."),
    ]


def test_delegate_batch_two_children_plus_synthesis_completes(monkeypatch):
    """The default recursion ceiling permits a prepared two-child batch.

    A batch run whose segment holds preparation turns plus the fan-out
    must complete under the DEFAULT iteration ceiling. One tool turn
    costs 8 super-steps and the answer turn 9 (both pinned in
    ``test_supersteps_per_tool_turn_is_pinned``), so this 4-tool-turn
    trajectory needs 41. The second half proves that a ceiling of 24
    fails before fan-out, so the success assertion is not tautological.
    """
    llm = ScriptedToolLLM(_batch_with_preparation_turns())
    client = make_client(monkeypatch, llm)
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Analysiere beide Straenge gruendlich.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "tool_directives": ["web_research"],
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        summary = wait_status(client, run_id, {"completed", "failed"})
        assert summary["status"] == "completed", (
            "batch trajectory died under the DEFAULT recursion ceiling "
            f"— the recalibration regressed: {summary}"
        )

        children = client.get(
            f"/v1/runs/{run_id}/children"
        ).json()["data"]
        assert {child["origin_key"] for child in children} == {
            "call_batch3:0",
            "call_batch3:1",
        }
        assert {child["status"] for child in children} == {"completed"}
        result = client.get(f"/v1/runs/{run_id}/result").json()
        assert result["answer"] == "Synthese aus beiden Unterauftraegen."

    # The same trajectory under a ceiling of 24 reaches the durable limit
    # decision before delegate_batch dispatches. It must remain visibly
    # paused with zero children until the user makes an explicit choice.
    old_llm = ScriptedToolLLM(_batch_with_preparation_turns())
    old_client = make_client(monkeypatch, old_llm, max_iterations=24)
    with old_client:
        run_id = old_client.post(
            "/v1/runs",
            json={
                "question": "Analysiere beide Straenge gruendlich.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "tool_directives": ["web_research"],
            },
        ).json()["run_id"]
        summary = wait_status(old_client, run_id, {"waiting_for_input"})
        assert summary["status"] == "waiting_for_input"
        assert old_client.get(
            f"/v1/runs/{run_id}/children"
        ).json()["data"] == []

        events = old_client.get(
            f"/v1/runs/{run_id}/events",
            params={"format": "json"},
        ).json()["data"]
        limit_events = [
            event
            for event in events
            if event["type"] == "inqtrix.agent.limit.reached"
        ]
        assert len(limit_events) == 1
        assert limit_events[0]["data"] == {
            "clarification_id": limit_events[0]["data"]["clarification_id"],
            "kind": "steps",
            "used": 25,
            "limit": 24,
            "next_limit": 48,
            "ceiling": 145,
            "extendable": True,
            "choices": ["extend", "partial", "cancel"],
            "state": "waiting_for_input",
        }

        clarifications = old_client.get(
            f"/v1/runs/{run_id}/clarifications"
        ).json()["data"]
        assert len(clarifications) == 1
        decision = old_client.post(
            f"/v1/runs/{run_id}/clarifications/"
            f"{clarifications[0]['clarification_id']}",
            json={"option_id": "partial"},
        )
        assert decision.status_code == 200, decision.text
        completed = wait_status(old_client, run_id, {"completed", "failed"})
        assert completed["status"] == "completed", completed
        result = old_client.get(f"/v1/runs/{run_id}/result").json()
        assert "Teilstand" in result["answer"]
        decided = [
            event
            for event in old_client.get(
                f"/v1/runs/{run_id}/events",
                params={"format": "json"},
            ).json()["data"]
            if event["type"] == "inqtrix.agent.limit.decided"
        ]
        assert [event["data"]["choice"] for event in decided] == ["partial"]


def test_delegate_batch_reports_partial_failure_per_assignment(monkeypatch):
    """One failed child yields a visible per-assignment failure line while
    the sibling's result survives (all-wait join, no batch abort)."""
    calls = {"count": 0}

    def flaky_child_graph(question: str, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        if "B" in question:
            raise RuntimeError("Provider explodierte")
        return fake_child_graph(question, **kwargs)

    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_batch2",
                "delegate_batch",
                {
                    "assignments": [
                        {"objective": "Strang A", "mode": "research"},
                        {"objective": "Strang B", "mode": "research"},
                    ]
                },
            ),
            _text_turn("Antwort mit benannter Luecke."),
        ]
    )
    client = make_client(monkeypatch, llm)
    # AFTER make_client: it patches run_web_graph itself and would
    # otherwise overwrite the flaky variant.
    monkeypatch.setattr(
        web_research_module, "run_web_graph", flaky_child_graph
    )
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Beide Straenge.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "tool_directives": ["web_research"],
            },
        )
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"completed"})
        tool_replies = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ]
        content = tool_replies[0]["content"]
        assert "Kindbericht zu: Strang A" in content
        assert "ist fehlgeschlagen" in content


def test_child_batch_guard_voids_child_next_to_ask_user():
    """The guard voids a child next to ``ask_user`` before dispatch.

    Without the guard, the child would submit and then both bodies would
    interrupt, leaving the child without a responsible parent path.
    """
    from langchain_core.messages import AIMessage

    from inqtrix.agents.kernel.middleware import (
        KernelChildBatchGuardMiddleware,
    )

    turn = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_ask_mix",
                "name": "ask_user",
                "args": {"question": "Welcher Markt?"},
            },
            {
                "id": "call_child_mix",
                "name": "run_web_research",
                "args": {"question": "Marktlage"},
            },
        ],
    )
    update = KernelChildBatchGuardMiddleware().after_model(
        {"messages": [turn]}, None
    )
    assert update is not None
    # Return shape: the AIMessage stays first, corrective answers follow.
    corrective = update["messages"][1:]
    assert [message.tool_call_id for message in corrective] == [
        "call_child_mix"
    ]
    assert "Rueckfrage" in corrective[0].content
    # The gate predicate skips these turns too (single answering
    # authority — HITL must not park an approval the guard voids).
    from inqtrix.agents.kernel.policy import _single_child_dispatch
    from types import SimpleNamespace

    request = SimpleNamespace(state={"messages": [turn]})
    assert _single_child_dispatch(request) is False


def test_child_batch_guard_blocks_parallel_single_child_calls(monkeypatch):
    """Two single-child tool calls in ONE turn dispatch nothing: zero
    submissions, corrective ToolMessages, the next turn proceeds."""
    llm = ScriptedToolLLM(
        [
            ChatTurn(
                text="",
                tool_calls=(
                    ToolCallRequest(
                        id="call_par1",
                        name="run_deep_mission",
                        arguments={"assignment": "A"},
                    ),
                    ToolCallRequest(
                        id="call_par2",
                        name="run_deep_mission",
                        arguments={"assignment": "B"},
                    ),
                ),
                finish_reason="tool_calls",
                model="high-model",
                prompt_tokens=10,
                completion_tokens=5,
                raw=None,
            ),
            _text_turn("Verstanden — ich nutze delegate_batch."),
        ]
    )
    client = make_client(monkeypatch, llm)
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Zwei parallele Auftraege.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
            },
        )
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"completed"})
        # NOTHING was submitted (pre-dispatch guard).
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert children == []
        corrective = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert len(corrective) == 2
        assert all(
            "delegate_batch" in str(m.get("content", ""))
            for m in corrective
        )
        events = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        assert any(
            e["type"] == "inqtrix.agent.child_batch.guarded"
            for e in events
        )
        # The interception is VISIBLE (No Silent Fallbacks): a narration
        # tells the user why nothing dispatched.
        assert any(
            e["type"] == "inqtrix.agent.narration"
            and str(e["data"].get("narration_id", "")).startswith(
                "batchguard_"
            )
            for e in events
        )


def test_parent_failure_cancels_running_children(monkeypatch):
    """A failed parent cancels its live children (orphan cleanup)."""
    import threading

    release = threading.Event()

    def hanging_child_graph(question: str, **kwargs: Any) -> dict[str, Any]:
        release.wait(timeout=10)
        return fake_child_graph(question, **kwargs)

    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_orph1",
                "run_web_research",
                {"question": "Haengender Auftrag"},
            ),
        ]
    )
    client = make_client(monkeypatch, llm)
    monkeypatch.setattr(
        web_research_module, "run_web_graph", hanging_child_graph
    )
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Starte den Auftrag.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "tool_directives": ["web_research"],
            },
        )
        run_id = response.json()["run_id"]
        # Parent parks on the child; the child hangs in its worker.
        wait_status(client, run_id, {"waiting_for_children"})
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 1

        # Force the parent into FAILED via the store seam.
        store = client.container.run_store  # type: ignore[attr-defined]
        store.fail(run_id, "Simulierter Elternfehler")
        release.set()

        summary = wait_status(client, run_id, {"failed"})
        assert summary["status"] == "failed"
        child_row = wait_status(
            client,
            children[0]["run_id"],
            {"cancelled", "failed", "completed"},
        )
        assert child_row["status"] == "cancelled"


def test_delegate_batch_gates_in_balanced_with_verbatim_assignments(
    monkeypatch,
):
    """Standard mode parks ONE kind='tool' approval whose single action
    carries every assignment verbatim; approve submits and resumes."""
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_batch3",
                "delegate_batch",
                {
                    "assignments": [
                        {"objective": "Strang A", "mode": "research"},
                        {"objective": "Strang B", "mode": "research"},
                    ]
                },
            ),
            _text_turn("Antwort nach Freigabe."),
        ]
    )
    client = make_client(monkeypatch, llm)
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Beide Straenge.",
                "mode": "agent_kernel",
                "autonomy": "balanced",
                "tool_directives": ["web_research"],
            },
        )
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"waiting_for_approval"})
        # Gate BEFORE dispatch: nothing was submitted yet.
        assert client.get(
            f"/v1/runs/{run_id}/children"
        ).json()["data"] == []
        approvals = client.get(
            f"/v1/runs/{run_id}/approvals"
        ).json()["data"]
        pending = [a for a in approvals if a["status"] == "pending"]
        assert len(pending) == 1 and pending[0]["kind"] == "tool"
        action = pending[0]["payload"]["actions"][0]
        assert action["tool"] == "delegate_batch"
        assert action["args"]["assignments"] == [
            {"objective": "Strang A", "mode": "research"},
            {"objective": "Strang B", "mode": "research"},
        ]

        decided = client.post(
            f"/v1/runs/{run_id}/approvals/{pending[0]['approval_id']}",
            json={"decision": "approve"},
        )
        assert decided.status_code == 200, decided.text
        wait_status(client, run_id, {"completed"})
        children = client.get(f"/v1/runs/{run_id}/children").json()["data"]
        assert len(children) == 2


def test_balanced_parallel_child_calls_void_without_gate(monkeypatch):
    """Two child-tool calls in ONE balanced turn are voided by the guard
    WITHOUT parking a gate.

    The gate policy's ``when`` predicate skips a multi-child turn, so the
    approve-then-void bug is gone (an approval whose actions the guard
    then refuses) and so is the reject double-answer. Discriminator: with
    the OLD unconditional gate, HITL would batch both calls into one
    interrupt and the run would hang at ``waiting_for_approval`` — this
    test would time out at ``completed``.
    """
    llm = ScriptedToolLLM(
        [
            ChatTurn(
                text="",
                tool_calls=(
                    ToolCallRequest(
                        id="call_bp1",
                        name="run_deep_mission",
                        arguments={"assignment": "A"},
                    ),
                    ToolCallRequest(
                        id="call_bp2",
                        name="run_deep_mission",
                        arguments={"assignment": "B"},
                    ),
                ),
                finish_reason="tool_calls",
                model="high-model",
                prompt_tokens=10,
                completion_tokens=5,
                raw=None,
            ),
            _text_turn("Verstanden — ich nutze delegate_batch."),
        ]
    )
    client = make_client(monkeypatch, llm)
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Zwei parallele Auftraege.",
                "mode": "agent_kernel",
                "autonomy": "balanced",
            },
        )
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"completed"})
        # No gate was EVER parked (the predicate skipped it).
        assert client.get(
            f"/v1/runs/{run_id}/approvals"
        ).json()["data"] == []
        # Nothing submitted; both calls answered by the guard.
        assert client.get(
            f"/v1/runs/{run_id}/children"
        ).json()["data"] == []
        corrective = [
            m
            for m in llm.chat_calls[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert len(corrective) == 2
        events = client.get(
            f"/v1/runs/{run_id}/events?format=json"
        ).json()["data"]
        assert any(
            e["type"] == "inqtrix.agent.child_batch.guarded"
            for e in events
        )


def test_child_submissions_inherit_parent_autonomy(monkeypatch):
    """Children inherit the PARENT's autonomy at submit time.

    Closes the plan's ``run_web_research`` gap where children defaulted
    to no autonomy. Stack inheritance rides the same resolve payload but
    needs a multi-stack env (single-stack test mode resolves to ``""``),
    so it is asserted in the live PG E2E, not here.
    """
    llm = ScriptedToolLLM(
        [
            _tool_turn(
                "call_inh1", "run_web_research", {"question": "Frage"}
            ),
            _text_turn("Fertig."),
        ]
    )
    client = make_client(monkeypatch, llm)
    captured: list[dict[str, Any]] = []
    real_submit = client.container.run_service.submit

    def spy(**kwargs: Any) -> Any:
        if kwargs.get("kind") == "agent_child":
            captured.append(kwargs)
        return real_submit(**kwargs)

    client.container.run_service.submit = spy  # type: ignore[attr-defined]
    with client:
        response = client.post(
            "/v1/runs",
            json={
                "question": "Recherche bitte.",
                "mode": "agent_kernel",
                "autonomy": "autonomous",
                "tool_directives": ["web_research"],
            },
        )
        run_id = response.json()["run_id"]
        wait_status(client, run_id, {"completed"})
    assert captured, "no child submission was captured"
    assert all(call["autonomy"] == "autonomous" for call in captured)
    assert captured[0]["origin_key"] == "call_inh1"
    assert captured[0]["parent_run_id"] == run_id
