"""Tests for HTTP routes and request-level safeguards."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import inqtrix.server.app as app_module
import inqtrix.server.routes as routes_module
from inqtrix.legal import legal_metadata
from inqtrix.providers.base import ProviderContext
from inqtrix.search_result import GroundedSearchResult
from inqtrix.server.app import create_app
from inqtrix.server.routes import create_router, register_routes
from inqtrix.settings import AgentSettings, ModelSettings, ServerSettings, Settings


class _DummyLLM:
    def __init__(self, available: bool = True) -> None:
        self._available = available

    def complete(self, *args, **kwargs):
        return "ok"


    def is_available(self) -> bool:
        return self._available


class _DummySearch:
    def __init__(self, available: bool = True) -> None:
        self._available = available

    def search(self, *args, **kwargs):
        return GroundedSearchResult()

    def is_available(self) -> bool:
        return self._available


def _make_app(
    *,
    llm_available: bool = True,
    search_available: bool = True,
    agent_max_total_seconds: int = 300,
    server_settings: ServerSettings | None = None,
) -> TestClient:
    app = FastAPI()
    router = create_router()

    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=server_settings or ServerSettings(),
    )
    settings.agent.max_total_seconds = agent_max_total_seconds
    providers = SimpleNamespace(
        llm=_DummyLLM(available=llm_available),
        search=_DummySearch(available=search_available),
    )

    register_routes(
        router,
        providers=providers,
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    app.include_router(router)
    return TestClient(app)


def test_health_reports_provider_aware_status():
    client = _make_app(llm_available=True, search_available=False)

    response = client.get("/health")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["llm"]["provider"] == "_DummyLLM"
    assert payload["llm"]["status"] == "ready"
    assert payload["search"]["provider"] == "_DummySearch"
    assert payload["search"]["status"] == "unavailable"
    assert payload["report_profile"] == "compact"
    assert payload["legal"] == {
        "project": "Inqtrix",
        "license": "AGPL-3.0-only",
        "source_url": "https://github.com/BZandi/inqtrix",
        "copyright": "Copyright (c) 2026 Babak Zandi.",
        "notice": (
            "Inqtrix - Copyright (c) 2026 Babak Zandi - "
            "https://github.com/BZandi/inqtrix"
        ),
        "warranty_notice": (
            "This software is provided without warranty under AGPL-3.0-only; "
            "see LICENSE for details."
        ),
    }


def test_legal_metadata_includes_warranty_notice():
    payload = legal_metadata()

    assert payload["license"] == "AGPL-3.0-only"
    assert payload["warranty_notice"] == (
        "This software is provided without warranty under AGPL-3.0-only; "
        "see LICENSE for details."
    )


def test_chat_completions_returns_timeout_response(monkeypatch):
    client = _make_app(agent_max_total_seconds=-29)

    def fake_run(*args, **kwargs):
        time.sleep(1.2)
        return {"answer": "Zu spaet", "result_state": {}}

    monkeypatch.setattr(routes_module, "agent_run", fake_run)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hallo"}],
            "stream": False,
        },
    )

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "timeout_error"


def test_native_runs_endpoint_returns_events_and_result(monkeypatch):
    client = _make_app()

    def fake_run(question, **kwargs):
        event_sink = kwargs["run_event_sink"]
        assert question == "Was ist neu?"
        assert kwargs["run_id"].startswith("run_")
        assert kwargs["settings"].max_rounds == 2
        event_sink(
            "inqtrix.node.started",
            {
                "node": "classify",
                "snapshot": {"current_node": "classify", "active_round": 0},
            },
        )
        return {
            "answer": "Antwort mit Quelle [1].",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "result_state": {
                "answer": "Antwort mit Quelle [1].",
                "round": 1,
                "queries": ["test query"],
                "all_citations": ["https://example.com/source"],
                "report_references": [
                    {"label": "E1", "url": "https://example.com/source", "tier": "unknown"},
                ],
                "source_records": {"src_1": {"url": "https://example.com/source"}},
                "final_confidence": 8,
                "source_tier_counts": {
                    "primary": 0,
                    "mainstream": 0,
                    "stakeholder": 0,
                    "unknown": 1,
                    "low": 0,
                },
                "claim_status_counts": {
                    "verified": 0,
                    "contested": 0,
                    "unverified": 0,
                },
                "source_quality_score": 0.25,
                "claim_quality_score": 0.0,
                "evidence_ledger": [{"id": "ev_1"}],
                "consolidated_claims": [{"status": "unverified"}],
                "aspect_coverage": 0.5,
                "evidence_consistency": 6,
                "evidence_sufficiency": 7,
            },
        }

    monkeypatch.setattr(routes_module, "agent_run", fake_run)

    response = client.post(
        "/v1/runs",
        json={
            "question": "Was ist neu?",
            "agent_overrides": {"max_rounds": 2},
        },
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert response.json()["agent_overrides"] == {"max_rounds": 2}

    deadline = time.time() + 1
    while time.time() < deadline:
        summary = client.get(f"/v1/runs/{run_id}").json()
        if summary["status"] == "completed":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("native run did not complete")

    result = client.get(f"/v1/runs/{run_id}/result")
    assert result.status_code == 200
    payload = result.json()
    assert payload["answer"] == "Antwort mit Quelle [1]."
    assert payload["metrics"]["rounds"] == 1
    assert payload["metrics"]["total_queries"] == 1
    assert payload["references"] == [
        {"label": "E1", "url": "https://example.com/source", "tier": "unknown"}
    ]
    assert payload["usage"]["total_tokens"] == 18
    summary = client.get(f"/v1/runs/{run_id}").json()
    snapshot = summary["snapshot"]
    assert snapshot["source_tier_counts"]["unknown"] == 1
    assert snapshot["source_quality_score"] == 0.25
    assert snapshot["claim_status_counts"]["unverified"] == 0
    assert snapshot["claim_quality_score"] == 0.0
    assert snapshot["evidence_record_count"] == 1
    assert snapshot["consolidated_claim_count"] == 1
    assert snapshot["aspect_coverage"] == 0.5
    assert snapshot["evidence_consistency"] == 6
    assert snapshot["evidence_sufficiency"] == 7

    with client.stream("GET", f"/v1/runs/{run_id}/events") as stream_response:
        body = stream_response.read().decode("utf-8")

    assert "event: inqtrix.run.queued" in body
    assert "event: inqtrix.run.snapshot" in body
    assert "event: inqtrix.node.started" in body
    assert "event: inqtrix.output_text.delta" in body
    assert "event: inqtrix.run.completed" in body


def test_native_runs_are_filterable_by_workspace_id(monkeypatch):
    client = _make_app()

    def fake_run(question, **kwargs):
        return {
            "answer": f"Antwort: {question}",
            "result_state": {
                "answer": f"Antwort: {question}",
                "round": 1,
                "queries": [],
                "all_citations": [],
            },
        }

    monkeypatch.setattr(routes_module, "agent_run", fake_run)

    response_a = client.post(
        "/v1/runs",
        headers={"X-Inqtrix-Workspace-Id": "ws_browser_a"},
        json={"question": "Workspace A"},
    )
    response_b = client.post(
        "/v1/runs",
        headers={"X-Inqtrix-Workspace-Id": "ws_browser_b"},
        json={"question": "Workspace B"},
    )

    assert response_a.status_code == 202
    assert response_b.status_code == 202
    run_a = response_a.json()["run_id"]
    run_b = response_b.json()["run_id"]

    visible_a = client.get(
        "/v1/runs",
        headers={"X-Inqtrix-Workspace-Id": "ws_browser_a"},
    )
    assert visible_a.status_code == 200
    assert [item["run_id"] for item in visible_a.json()["data"]] == [run_a]
    assert visible_a.json()["data"][0]["workspace_id"] == "ws_browser_a"

    hidden = client.get(
        f"/v1/runs/{run_b}",
        headers={"X-Inqtrix-Workspace-Id": "ws_browser_a"},
    )
    assert hidden.status_code == 404

    unscoped = client.get("/v1/runs")
    assert unscoped.status_code == 200
    assert {item["run_id"] for item in unscoped.json()["data"]} >= {run_a, run_b}


def test_native_runs_reject_invalid_workspace_id():
    client = _make_app()

    response = client.get(
        "/v1/runs",
        headers={"X-Inqtrix-Workspace-Id": "../bad"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


# ------------------------------------------------------------------ #
# create_app — Baukasten injection (ADR-WS-1) and lifespan (ADR-WS-2)
# ------------------------------------------------------------------ #


def test_create_app_accepts_provider_injection(monkeypatch):
    """create_app(providers=...) must reuse the injected pair."""
    provider_factory_calls: list[str] = []

    def fake_create_providers(*args, **kwargs):
        provider_factory_calls.append("create_providers")
        raise AssertionError("create_providers must not run with injected providers")

    monkeypatch.setattr(app_module, "create_providers", fake_create_providers)

    providers = ProviderContext(llm=_DummyLLM(), search=_DummySearch())
    app = create_app(settings=Settings(), providers=providers)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["llm"]["provider"] == "_DummyLLM"
    assert response.json()["search"]["provider"] == "_DummySearch"
    assert provider_factory_calls == [], (
        f"Provider auto-create path was reached unexpectedly: {provider_factory_calls}"
    )


def test_create_app_strategies_use_provider_claim_extract_model(monkeypatch):
    """Bug A regression: strategies must read claim_extract_model from the
    LLM provider's own ``models`` attribute, not from the global
    ``settings.models`` block. Otherwise the LiteLLM-flavoured default
    leaks into Anthropic / Bedrock / Azure providers and breaks claim
    extraction with HTTP 400/404 on ``claude-opus-4.6-agent``.
    """
    captured: dict[str, str] = {}

    real_create_default = app_module.create_default_strategies

    def spy_create_default(settings_arg, **kwargs):  # noqa: ANN001
        captured["claim_extract_model"] = kwargs.get("claim_extract_model", "")
        return real_create_default(settings_arg, **kwargs)

    monkeypatch.setattr(app_module, "create_default_strategies", spy_create_default)

    class _LLMWithProviderModel(_DummyLLM):
        def __init__(self) -> None:
            super().__init__(available=True)
            self.models = ModelSettings(
                reasoning_model="claude-opus-4-6",
                claim_extract_model="claude-haiku-4-5",
            )

    providers = ProviderContext(llm=_LLMWithProviderModel(), search=_DummySearch())
    create_app(settings=Settings(), providers=providers)

    assert captured["claim_extract_model"] == "claude-haiku-4-5", (
        "Strategies-Layer must use the provider's claim_extract_model "
        f"(claude-haiku-4-5), not the LiteLLM default; got "
        f"{captured['claim_extract_model']!r}."
    )


def test_health_models_payload_uses_provider_models():
    """Bug C regression: /health must show the LLM provider's actual
    model identifiers, not the global ``settings.models`` defaults.
    """
    class _LLMWithProviderModel(_DummyLLM):
        def __init__(self) -> None:
            super().__init__(available=True)
            self.models = ModelSettings(
                reasoning_model="claude-opus-4-6",
                claim_extract_model="claude-haiku-4-5",
            )

    class _SearchWithModel(_DummySearch):
        def __init__(self) -> None:
            super().__init__(available=True)
            # ADR-WS-12: search providers expose their identifier via
            # the standardized ``search_model`` property (mirrors the
            # ``LLMProvider.models`` Constructor-First contract).
            self.search_model = "sonar-pro"

    app = FastAPI()
    router = create_router()
    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=ServerSettings(),
    )
    providers = SimpleNamespace(
        llm=_LLMWithProviderModel(),
        search=_SearchWithModel(),
    )
    register_routes(
        router,
        providers=providers,
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    app.include_router(router)

    payload = TestClient(app).get("/health").json()

    assert payload["reasoning_model"] == "claude-opus-4-6"
    assert payload["claim_extract_model"] == "claude-haiku-4-5"
    # Roles without explicit per-role model fall back to reasoning_model
    # via the provider's effective_* properties — never to the global
    # LiteLLM default ``claude-opus-4.6-agent``.
    assert payload["classify_model"] == "claude-opus-4-6"
    assert payload["evaluate_model"] == "claude-opus-4-6"
    assert payload["search_model"] == "sonar-pro"

    # node_models exposes the per-node resolution with provenance, so an
    # operator can see which call site uses which model and *why*: answer
    # falls back to reasoning_model (no tier set -> the default grips, and it
    # says so), while claim_extract is a per-node override.
    node_models = payload["node_models"]
    assert node_models["answer"]["model"] == "claude-opus-4-6"
    assert node_models["answer"]["model_source"] == "reasoning_model_default"
    assert node_models["claim_extract"]["model"] == "claude-haiku-4-5"
    assert node_models["claim_extract"]["model_source"] == "per_node_override"
    chat_options = payload["chat_model_options"]
    assert [option["tier"] for option in chat_options] == ["high", "mid", "fast"]
    assert [option["node"] for option in chat_options] == ["direct_chat"] * 3
    assert chat_options[1]["model"] == "claude-opus-4-6"


def test_health_models_payload_respects_model_tier():
    class _LLMWithTierModels(_DummyLLM):
        def __init__(self) -> None:
            super().__init__(available=True)
            self.models = ModelSettings(
                reasoning_model="fallback",
                tier_high_model="high-model",
                tier_mid_model="mid-model",
                tier_fast_model="fast-model",
            )

    app = FastAPI()
    router = create_router()
    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(model_tier="fast"),
        server=ServerSettings(),
    )
    providers = SimpleNamespace(llm=_LLMWithTierModels(), search=_DummySearch())
    register_routes(
        router,
        providers=providers,
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    app.include_router(router)

    payload = TestClient(app).get("/health").json()

    assert payload["node_models"]["answer"]["model"] == "fast-model"
    assert payload["node_models"]["answer"]["requested_tier"] == "fast"
    assert payload["classify_model"] == "fast-model"
    assert payload["claim_extract_model"] == "fast-model"
    assert payload["evaluate_model"] == "fast-model"
    assert [option["model"] for option in payload["chat_model_options"]] == [
        "high-model",
        "mid-model",
        "fast-model",
    ]
    assert [option["requested_tier"] for option in payload["chat_model_options"]] == [
        "high",
        "mid",
        "fast",
    ]


def test_health_models_payload_does_not_leak_settings_defaults_without_provider_models():
    app = FastAPI()
    router = create_router()
    settings = Settings(
        models=ModelSettings(
            reasoning_model="settings-reasoning",
            classify_model="settings-classify",
            claim_extract_model="settings-claim",
            evaluate_model="settings-evaluate",
        ),
        agent=AgentSettings(),
        server=ServerSettings(),
    )
    providers = SimpleNamespace(llm=_DummyLLM(), search=_DummySearch())
    register_routes(
        router,
        providers=providers,
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    app.include_router(router)

    payload = TestClient(app).get("/health").json()

    assert payload["reasoning_model"] == ""
    assert payload["classify_model"] == ""
    assert payload["claim_extract_model"] == ""
    assert payload["evaluate_model"] == ""
    assert payload["node_models"]["answer"]["model_source"] == "provider_models_missing"
    assert payload["node_models"]["answer"]["effort_source"] == "provider_default_unseen"
    assert all(option["model"] == "" for option in payload["chat_model_options"])
    assert all(
        option["model_source"] == "provider_models_missing"
        for option in payload["chat_model_options"]
    )


def test_health_search_model_uses_provider_property_not_settings_default():
    """ADR-WS-12 regression: when the search provider exposes a
    ``search_model`` property it MUST surface in /health verbatim,
    never the global ``settings.models.search_model`` default. This is
    the root-cause fix for the Azure-Live-Test surprise where
    ``AzureOpenAIWebSearch`` had no recognised attribute and the
    operator saw ``perplexity-sonar-pro-agent`` on an Azure-only stack.
    """
    class _AzureLikeSearch(_DummySearch):
        def __init__(self) -> None:
            super().__init__(available=True)
            # No legacy attribute names (model / _model / agent_name) —
            # simulates AzureOpenAIWebSearch pre-fix; only the new
            # standardized property is set.
            self.search_model = "gpt-4.1+web_search_tool"

    app = FastAPI()
    router = create_router()
    settings = Settings(
        models=ModelSettings(),  # leaves the LiteLLM default search_model
        agent=AgentSettings(),
        server=ServerSettings(),
    )
    providers = SimpleNamespace(llm=_DummyLLM(), search=_AzureLikeSearch())
    register_routes(
        router,
        providers=providers,
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    app.include_router(router)

    payload = TestClient(app).get("/health").json()
    assert payload["search_model"] == "gpt-4.1+web_search_tool", (
        "Provider-exposed search_model must win over settings.models.search_model"
    )


def test_health_search_model_falls_back_to_settings_when_provider_silent():
    """When a third-party search provider lacks the ``search_model``
    property entirely, the helper falls back to ``settings.models.search_model``.
    This keeps backwards compatibility for code outside the inqtrix
    repo that pre-dates ADR-WS-12.
    """
    class _SilentSearch(_DummySearch):
        # Inherits no search_model attr; getattr will return "" / falsy
        pass

    app = FastAPI()
    router = create_router()
    settings = Settings(
        models=ModelSettings(search_model="legacy-default-sentinel"),
        agent=AgentSettings(),
        server=ServerSettings(),
    )
    providers = SimpleNamespace(llm=_DummyLLM(), search=_SilentSearch())
    register_routes(
        router,
        providers=providers,
        strategies=SimpleNamespace(),
        settings=settings,
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    app.include_router(router)

    payload = TestClient(app).get("/health").json()
    assert payload["search_model"] == "legacy-default-sentinel"


def test_create_app_lifespan_logs_provider_init(caplog):
    """ASGI lifespan must emit startup + shutdown logs with provider labels."""
    providers = ProviderContext(llm=_DummyLLM(), search=_DummySearch())
    app = create_app(settings=Settings(), providers=providers)

    # configure_logging() sets propagate=False on the inqtrix logger so the
    # standard root-handler caplog setup does not see records. Attach the
    # capture handler directly to the inqtrix logger for this test.
    inqtrix_logger = logging.getLogger("inqtrix")
    inqtrix_logger.addHandler(caplog.handler)
    previous_level = inqtrix_logger.level
    inqtrix_logger.setLevel(logging.INFO)
    try:
        with caplog.at_level(logging.INFO, logger="inqtrix"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        inqtrix_logger.removeHandler(caplog.handler)
        inqtrix_logger.setLevel(previous_level)

    messages = [rec.getMessage() for rec in caplog.records]
    startup = [m for m in messages if "Inqtrix server starting" in m]
    shutdown = [m for m in messages if "Inqtrix server stopping" in m]
    assert startup, f"Startup log not emitted; got: {messages}"
    assert "_DummyLLM" in startup[0]
    assert "_DummySearch" in startup[0]
    assert shutdown, f"Shutdown log not emitted; got: {messages}"


# ------------------------------------------------------------------ #
# Payload caps (ADR-WS DoS guard)
# ------------------------------------------------------------------ #


def test_chat_completions_rejects_too_many_messages():
    """``max_message_count`` rejects array-bomb payloads with HTTP 413."""
    server = ServerSettings(INQTRIX_MAX_MESSAGE_COUNT=10)
    client = _make_app(server_settings=server)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": f"msg {i}"} for i in range(11)
            ],
            "stream": False,
        },
    )

    assert response.status_code == 413
    payload = response.json()
    assert payload["error"]["type"] == "payload_too_large"
    assert "11" in payload["error"]["message"]
    assert "10" in payload["error"]["message"]


def test_chat_completions_rejects_oversized_total_tokens():
    """``max_total_input_tokens`` rejects oversized bodies with HTTP 413."""
    # 10_000 token cap = 40_000 chars at 4 chars per token.
    server = ServerSettings(INQTRIX_MAX_TOTAL_INPUT_TOKENS=10_000)
    client = _make_app(server_settings=server)

    huge_content = "x" * 60_000  # ~15_000 tokens — exceeds the cap.
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": huge_content}],
            "stream": False,
        },
    )

    assert response.status_code == 413
    payload = response.json()
    assert payload["error"]["type"] == "payload_too_large"
    assert "tokens" in payload["error"]["message"].lower()


def test_chat_completions_accepts_typical_payload(monkeypatch):
    """Realistic multi-turn payloads pass the cap unchanged."""
    monkeypatch.setattr(
        routes_module,
        "agent_run",
        lambda *args, **kwargs: {
            "answer": "ok",
            "result_state": {},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        },
    )

    client = _make_app()  # Defaults: 200 messages, 500k tokens.

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Question " + str(i) * 200}
                for i in range(20)
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
