"""Tests for per-request agent overrides (ADR-WS-6).

Covers the whitelist Pydantic model, range validation, the merge
helper ``apply_overrides`` (including the three profile-switch
scenarios A/B/C from the ADR), and the end-to-end integration
through ``/v1/chat/completions``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import inqtrix.server.routes as routes_module
from inqtrix.report_profiles import ReportProfile
from inqtrix.server.overrides import (
    AgentOverridesRequest,
    apply_overrides,
    parse_overrides_payload,
)
from inqtrix.server.routes import create_router, register_routes
from inqtrix.server.session import SessionStore
from inqtrix.settings import AgentSettings, ModelSettings, ServerSettings, Settings


# ------------------------------------------------------------------ #
# Whitelist Pydantic model
# ------------------------------------------------------------------ #


def test_overrides_request_accepts_whitelist_fields():
    """All whitelist fields must validate when supplied with sane values."""
    overrides = AgentOverridesRequest(
        max_rounds=5,
        min_rounds=2,
        confidence_stop=9,
        report_profile=ReportProfile.DEEP,
        max_total_seconds=540,
        first_round_queries=10,
        max_context=24,
    )
    assert overrides.max_rounds == 5
    assert overrides.report_profile == ReportProfile.DEEP


def test_overrides_request_rejects_unknown_field():
    """Unknown keys must raise (model_config extra='forbid')."""
    with pytest.raises(ValidationError):
        AgentOverridesRequest.model_validate({"unknown_field": 1})


def test_overrides_request_validates_ranges():
    """Range-violating values must raise."""
    with pytest.raises(ValidationError):
        AgentOverridesRequest.model_validate({"max_rounds": 99})
    with pytest.raises(ValidationError):
        AgentOverridesRequest.model_validate({"max_total_seconds": 5})  # ge=30
    with pytest.raises(ValidationError):
        AgentOverridesRequest.model_validate({"confidence_stop": 0})  # ge=1


def test_parse_overrides_payload_returns_none_when_absent():
    assert parse_overrides_payload(None) is None


def test_parse_overrides_payload_rejects_non_dict():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        parse_overrides_payload("not a dict")
    assert excinfo.value.status_code == 400


def test_parse_overrides_payload_rejects_unknown_field():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        parse_overrides_payload({"definitely_not_real": 1})
    assert excinfo.value.status_code == 400


# ------------------------------------------------------------------ #
# apply_overrides — base behaviour
# ------------------------------------------------------------------ #


def test_apply_overrides_no_op_when_none():
    base = AgentSettings()
    assert apply_overrides(base, None) is base


def test_apply_overrides_no_op_when_empty():
    base = AgentSettings()
    overrides = AgentOverridesRequest()
    assert apply_overrides(base, overrides) is base


def test_apply_overrides_merges_into_agent_settings():
    base = AgentSettings()
    overrides = AgentOverridesRequest(max_rounds=2, confidence_stop=6)
    patched = apply_overrides(base, overrides)
    assert patched is not base
    assert patched.max_rounds == 2
    assert patched.confidence_stop == 6


# ------------------------------------------------------------------ #
# Profile-switch semantics — Scenarios A / B / C from ADR-WS-6
# ------------------------------------------------------------------ #


def test_apply_overrides_preserves_operator_explicit_fields():
    """Scenario A — operator-explicit max_rounds must survive a DEEP switch."""
    # Operator explicitly sets max_rounds=6 against the COMPACT default.
    base = AgentSettings(report_profile=ReportProfile.COMPACT, max_rounds=6)
    overrides = AgentOverridesRequest(report_profile=ReportProfile.DEEP)
    patched = apply_overrides(base, overrides)

    assert patched.report_profile == ReportProfile.DEEP
    # Operator-explicit value wins over the DEEP profile default of 5.
    assert patched.max_rounds == 6
    # Non-explicit fields take on DEEP defaults.
    assert patched.confidence_stop == 9
    assert patched.max_context == 24


def test_apply_overrides_applies_full_profile_defaults():
    """Scenario B — pure COMPACT defaults flip wholesale to DEEP."""
    # Use Pydantic-default construction so model_fields_set stays empty.
    base = AgentSettings.model_validate({})
    overrides = AgentOverridesRequest(report_profile=ReportProfile.DEEP)
    patched = apply_overrides(base, overrides)

    assert patched.report_profile == ReportProfile.DEEP
    # All DEEP-profile defaults filled in.
    assert patched.max_rounds == 5
    assert patched.confidence_stop == 9
    assert patched.max_context == 24
    assert patched.first_round_queries == 10


def test_apply_overrides_user_explicit_wins_over_profile_defaults():
    """Scenario C — user-explicit max_rounds beats DEEP-profile preset."""
    base = AgentSettings.model_validate({})
    overrides = AgentOverridesRequest(
        report_profile=ReportProfile.DEEP, max_rounds=3
    )
    patched = apply_overrides(base, overrides)

    assert patched.report_profile == ReportProfile.DEEP
    # User-explicit value wins over the DEEP default of 5.
    assert patched.max_rounds == 3
    # Other DEEP defaults are still applied.
    assert patched.confidence_stop == 9
    assert patched.max_context == 24


# ------------------------------------------------------------------ #
# Integration through /v1/chat/completions
# ------------------------------------------------------------------ #


class _DummyLLM:
    def complete(self, *args, **kwargs):
        return "ok"

    def summarize_parallel(self, *args, **kwargs):
        return ("", 0, 0)

    def is_available(self) -> bool:
        return True


class _DummySearch:
    def search(self, *args, **kwargs):
        return {
            "answer": "",
            "citations": [],
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

    def is_available(self) -> bool:
        return True


def _make_app() -> tuple[TestClient, dict[str, Any]]:
    """Build a TestClient with a stub agent_run that records the settings."""
    captured: dict[str, Any] = {}

    def fake_run(question, *, history, prev_session, providers, strategies, settings):
        captured["settings"] = settings
        captured["question"] = question
        return {
            "answer": "Antwort",
            "result_state": {},
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    app = FastAPI()
    router = create_router()
    settings = Settings(models=ModelSettings(), agent=AgentSettings(), server=ServerSettings())
    providers = SimpleNamespace(llm=_DummyLLM(), search=_DummySearch())

    register_routes(
        router,
        providers=providers,
        strategies=SimpleNamespace(),
        settings=settings,
        session_store=SessionStore(),
        semaphore_factory=lambda: asyncio.Semaphore(1),
    )
    app.include_router(router)

    client = TestClient(app)
    client.app.dependency_overrides = {}  # noqa: SLF001 — placeholder
    return client, captured, fake_run


def test_chat_completions_with_overrides_routes_through_agent_run(monkeypatch):
    client, captured, fake_run = _make_app()
    monkeypatch.setattr(routes_module, "agent_run", fake_run)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hallo"}],
            "stream": False,
            "agent_overrides": {"max_rounds": 2, "confidence_stop": 7},
        },
    )

    assert response.status_code == 200
    forwarded = captured["settings"]
    assert forwarded.max_rounds == 2
    assert forwarded.confidence_stop == 7


def test_chat_completions_with_profile_switch_routes_deep_settings(monkeypatch):
    """End-to-end Scenario B: pure COMPACT base + DEEP override."""
    client, captured, fake_run = _make_app()
    monkeypatch.setattr(routes_module, "agent_run", fake_run)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hallo"}],
            "stream": False,
            "agent_overrides": {"report_profile": "deep"},
        },
    )

    assert response.status_code == 200
    forwarded = captured["settings"]
    assert forwarded.report_profile == ReportProfile.DEEP
    # DEEP profile defaults must have cascaded.
    assert forwarded.max_rounds == 5
    assert forwarded.confidence_stop == 9
    assert forwarded.max_context == 24


def test_chat_completions_invalid_override_returns_400(monkeypatch):
    client, captured, fake_run = _make_app()
    monkeypatch.setattr(routes_module, "agent_run", fake_run)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hallo"}],
            "stream": False,
            "agent_overrides": {"max_rounds": 99},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    # agent_run must NOT be invoked when validation fails.
    assert "settings" not in captured


def test_chat_completions_without_overrides_uses_server_defaults(monkeypatch):
    """No agent_overrides field → server-default AgentSettings reaches agent_run."""
    client, captured, fake_run = _make_app()
    monkeypatch.setattr(routes_module, "agent_run", fake_run)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hallo"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    forwarded = captured["settings"]
    # Server default profile is COMPACT.
    assert forwarded.report_profile == ReportProfile.COMPACT
