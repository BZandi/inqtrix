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
from inqtrix.providers.base import ProviderContext
from inqtrix.server.app import create_app
from inqtrix.server.routes import create_router, register_routes
from inqtrix.server.session import SessionStore
from inqtrix.settings import AgentSettings, ModelSettings, ServerSettings, Settings


class _DummyLLM:
    def __init__(self, available: bool = True) -> None:
        self._available = available

    def complete(self, *args, **kwargs):
        return "ok"

    def summarize_parallel(self, *args, **kwargs):
        return ("", 0, 0)

    def is_available(self) -> bool:
        return self._available


class _DummySearch:
    def __init__(self, available: bool = True) -> None:
        self._available = available

    def search(self, *args, **kwargs):
        return {
            "answer": "",
            "citations": [],
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

    def is_available(self) -> bool:
        return self._available


def _make_app(
    *,
    llm_available: bool = True,
    search_available: bool = True,
    agent_max_total_seconds: int = 300,
) -> TestClient:
    app = FastAPI()
    router = create_router()

    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=ServerSettings(),
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
        session_store=SessionStore(),
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


# ------------------------------------------------------------------ #
# create_app — Baukasten injection (ADR-WS-1) and lifespan (ADR-WS-2)
# ------------------------------------------------------------------ #


def test_create_app_accepts_provider_injection(monkeypatch):
    """create_app(providers=...) must skip YAML and reuse the injected pair."""
    yaml_calls: list[str] = []

    def fake_load_config(*args, **kwargs):
        yaml_calls.append("load_config")
        from inqtrix.config import InqtrixConfig

        return InqtrixConfig()

    def fake_create_providers(*args, **kwargs):
        yaml_calls.append("create_providers")
        raise AssertionError("create_providers must not run with injected providers")

    monkeypatch.setattr(app_module, "load_config", fake_load_config)
    monkeypatch.setattr(app_module, "create_providers", fake_create_providers)

    providers = ProviderContext(llm=_DummyLLM(), search=_DummySearch())
    app = create_app(settings=Settings(), providers=providers)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["llm"]["provider"] == "_DummyLLM"
    assert response.json()["search"]["provider"] == "_DummySearch"
    # YAML must not be touched in injection mode.
    assert yaml_calls == [], (
        f"YAML/Auto-create path was reached unexpectedly: {yaml_calls}"
    )


def test_create_app_strategies_use_provider_summarize_model(monkeypatch):
    """Bug A regression: strategies must read summarize_model from the
    LLM provider's own ``models`` attribute, not from the global
    ``settings.models`` block. Otherwise the LiteLLM-flavoured default
    leaks into Anthropic / Bedrock / Azure providers and breaks claim
    extraction with HTTP 400/404 on ``claude-opus-4.6-agent``.
    """
    captured: dict[str, str] = {}

    real_create_default = app_module.create_default_strategies

    def spy_create_default(settings_arg, **kwargs):  # noqa: ANN001
        captured["summarize_model"] = kwargs.get("summarize_model", "")
        return real_create_default(settings_arg, **kwargs)

    monkeypatch.setattr(app_module, "create_default_strategies", spy_create_default)

    class _LLMWithProviderModel(_DummyLLM):
        def __init__(self) -> None:
            super().__init__(available=True)
            self.models = ModelSettings(
                reasoning_model="claude-opus-4-6",
                summarize_model="claude-haiku-4-5",
            )

    providers = ProviderContext(llm=_LLMWithProviderModel(), search=_DummySearch())
    create_app(settings=Settings(), providers=providers)

    assert captured["summarize_model"] == "claude-haiku-4-5", (
        "Strategies-Layer must use the provider's summarize_model "
        f"(claude-haiku-4-5), not the LiteLLM default; got "
        f"{captured['summarize_model']!r}."
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
                summarize_model="claude-haiku-4-5",
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
        session_store=SessionStore(),
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    app.include_router(router)

    payload = TestClient(app).get("/health").json()

    assert payload["reasoning_model"] == "claude-opus-4-6"
    assert payload["summarize_model"] == "claude-haiku-4-5"
    # Roles without explicit per-role model fall back to reasoning_model
    # via the provider's effective_* properties — never to the global
    # LiteLLM default ``claude-opus-4.6-agent``.
    assert payload["classify_model"] == "claude-opus-4-6"
    assert payload["evaluate_model"] == "claude-opus-4-6"
    assert payload["search_model"] == "sonar-pro"


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
        session_store=SessionStore(),
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
        session_store=SessionStore(),
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
