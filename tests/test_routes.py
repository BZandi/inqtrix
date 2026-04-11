"""Tests for HTTP routes and request-level safeguards."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import inqtrix.routes as routes_module
from inqtrix.routes import create_router, register_routes
from inqtrix.session import SessionStore
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
    assert payload["llm"]["status"] == "connected"
    assert payload["search"]["provider"] == "_DummySearch"
    assert payload["search"]["status"] == "unavailable"


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
