"""Tests for the multi-stack server (ADR-MS-1..MS-4)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import inqtrix.server.routes as routes_module
from inqtrix.providers.base import ProviderContext
from inqtrix.server.session import SessionStore  # noqa: F401 — for completeness
from inqtrix.server.stacks import (
    StackBundle,
    _DiscoveryCache,
    _validate_stacks,
    create_multi_stack_app,
)
from inqtrix.settings import AgentSettings, ModelSettings, ServerSettings, Settings


# ------------------------------------------------------------------ #
# Stub providers (network-free)
# ------------------------------------------------------------------ #


class _StubLLM:
    def __init__(self, label: str = "default") -> None:
        self._label = label

    def complete(self, *a, **kw):
        return self._label

    def summarize_parallel(self, *a, **kw):
        return ("", 0, 0)

    def is_available(self):
        return True


class _StubSearch:
    def __init__(self, label: str = "default") -> None:
        self._label = label

    def search(self, *a, **kw):
        return {
            "answer": "",
            "citations": [],
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

    def is_available(self):
        return True


def _bundle(label: str, *, description: str = "") -> StackBundle:
    return StackBundle(
        providers=ProviderContext(llm=_StubLLM(label), search=_StubSearch(label)),
        description=description,
    )


def _make_settings() -> Settings:
    return Settings(models=ModelSettings(), agent=AgentSettings(), server=ServerSettings())


# ------------------------------------------------------------------ #
# StackBundle + validation
# ------------------------------------------------------------------ #


def test_stack_bundle_is_frozen():
    bundle = _bundle("a")
    with pytest.raises(Exception):  # FrozenInstanceError or TypeError
        bundle.description = "mutated"


def test_validate_stacks_rejects_empty():
    with pytest.raises(ValueError, match="at least one stack"):
        _validate_stacks({}, "x")


def test_validate_stacks_rejects_unknown_default():
    with pytest.raises(ValueError, match="not in stacks"):
        _validate_stacks({"a": _bundle("a")}, "missing")


def test_validate_stacks_rejects_invalid_name():
    with pytest.raises(ValueError, match="must match"):
        _validate_stacks({"BadName": _bundle("a")}, "BadName")


def test_validate_stacks_accepts_lowercase_underscored_names():
    _validate_stacks(
        {"valid_name_123": _bundle("a")}, "valid_name_123"
    )  # must not raise


# ------------------------------------------------------------------ #
# Discovery endpoint
# ------------------------------------------------------------------ #


def test_discovery_endpoint_lists_all_stacks_with_ready_flag():
    settings = _make_settings()
    stacks = {
        "stack_a": _bundle("a", description="Stack A description"),
        "stack_b": _bundle("b", description="Stack B description"),
    }
    app = create_multi_stack_app(
        settings=settings, stacks=stacks, default_stack="stack_a"
    )
    with TestClient(app) as client:
        response = client.get("/v1/stacks")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default"] == "stack_a"
    assert len(payload["stacks"]) == 2
    names = {s["name"] for s in payload["stacks"]}
    assert names == {"stack_a", "stack_b"}
    for s in payload["stacks"]:
        assert s["ready"] is True
        assert s["llm"] == "_StubLLM"
        assert s["search"] == "_StubSearch"


def test_discovery_payload_includes_per_stack_models_block():
    """Bug C regression for multi-stack: every entry in /v1/stacks must
    expose a ``models`` block built from the bundle's own provider, so a
    UI can render an honest per-stack model chip without an extra
    /health round-trip.
    """

    class _LLMWithModels(_StubLLM):
        def __init__(self, label: str, summarize: str) -> None:
            super().__init__(label=label)
            self.models = ModelSettings(
                reasoning_model=f"reason-{label}",
                summarize_model=summarize,
            )

    class _SearchWithModel(_StubSearch):
        def __init__(self, label: str, model: str) -> None:
            super().__init__(label=label)
            # ADR-WS-12: standardized search_model property used by
            # ``_stack_models_payload`` (no more attribute heuristic).
            self.search_model = model

    settings = _make_settings()
    bundle_a = StackBundle(
        providers=ProviderContext(
            llm=_LLMWithModels("a", "haiku-a"),
            search=_SearchWithModel("a", "sonar-a"),
        ),
        description="A",
    )
    bundle_b = StackBundle(
        providers=ProviderContext(
            llm=_LLMWithModels("b", "haiku-b"),
            search=_SearchWithModel("b", "sonar-b"),
        ),
        description="B",
    )
    stacks = {"stack_a": bundle_a, "stack_b": bundle_b}
    app = create_multi_stack_app(
        settings=settings, stacks=stacks, default_stack="stack_a"
    )
    with TestClient(app) as client:
        payload = client.get("/v1/stacks").json()

    by_name = {s["name"]: s for s in payload["stacks"]}
    assert by_name["stack_a"]["models"] == {
        "reasoning_model": "reason-a",
        "summarize_model": "haiku-a",
        "classify_model": "reason-a",
        "evaluate_model": "reason-a",
        "search_model": "sonar-a",
    }
    assert by_name["stack_b"]["models"]["summarize_model"] == "haiku-b"
    assert by_name["stack_b"]["models"]["search_model"] == "sonar-b"


def test_discovery_cache_returns_same_payload_within_ttl():
    """The same StackBundle.is_available() must not be hammered."""
    cache = _DiscoveryCache()
    stacks = {"a": _bundle("a"), "b": _bundle("b")}
    p1 = cache.get(stacks=stacks, default_stack="a")
    p2 = cache.get(stacks=stacks, default_stack="a")
    assert p1 is p2  # cached object identity


# ------------------------------------------------------------------ #
# Multi-stack chat-completions routing
# ------------------------------------------------------------------ #


def _patched_agent_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch agent_run to record the providers/settings it received."""
    captured: dict[str, Any] = {}

    def fake_run(question, *, history, prev_session, providers, strategies, settings):
        captured["llm_label"] = providers.llm._label
        captured["settings_max_rounds"] = settings.max_rounds
        return {
            "answer": f"answer from {providers.llm._label}",
            "result_state": {},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    monkeypatch.setattr(routes_module, "agent_run", fake_run)
    return captured


def test_chat_completions_routes_to_default_stack_when_no_field(monkeypatch):
    captured = _patched_agent_run(monkeypatch)
    settings = _make_settings()
    stacks = {
        "alpha": _bundle("alpha", description="Alpha"),
        "beta": _bundle("beta", description="Beta"),
    }
    app = create_multi_stack_app(settings=settings, stacks=stacks, default_stack="alpha")
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
        )
    assert response.status_code == 200
    assert captured["llm_label"] == "alpha"


def test_chat_completions_routes_to_named_stack(monkeypatch):
    captured = _patched_agent_run(monkeypatch)
    settings = _make_settings()
    stacks = {"alpha": _bundle("alpha"), "beta": _bundle("beta")}
    app = create_multi_stack_app(settings=settings, stacks=stacks, default_stack="alpha")
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "stack": "beta",
            },
        )
    assert response.status_code == 200
    assert captured["llm_label"] == "beta"


def test_chat_completions_unknown_stack_returns_400_with_available(monkeypatch):
    _patched_agent_run(monkeypatch)
    settings = _make_settings()
    stacks = {"alpha": _bundle("alpha"), "beta": _bundle("beta")}
    app = create_multi_stack_app(settings=settings, stacks=stacks, default_stack="alpha")
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "stack": "does_not_exist",
            },
        )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["available_stacks"] == ["alpha", "beta"]


def test_chat_completions_per_stack_agent_settings_override_global(monkeypatch):
    """A StackBundle with its own agent_settings overrides settings.agent."""
    captured = _patched_agent_run(monkeypatch)
    settings = _make_settings()
    custom_settings = AgentSettings(max_rounds=2)  # global default is 4
    stacks = {
        "tight": StackBundle(
            providers=ProviderContext(llm=_StubLLM("tight"), search=_StubSearch("tight")),
            agent_settings=custom_settings,
        ),
    }
    app = create_multi_stack_app(settings=settings, stacks=stacks, default_stack="tight")
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
        )
    assert response.status_code == 200
    assert captured["settings_max_rounds"] == 2


def test_cross_stack_follow_up_creates_distinct_session(monkeypatch):
    """Same conversation against two stacks must yield two different sessions."""
    settings = _make_settings()
    stacks = {"alpha": _bundle("alpha"), "beta": _bundle("beta")}

    saved: list[tuple[str, str]] = []  # (stack_label, session_id)

    def fake_run(question, *, history, prev_session, providers, strategies, settings):
        return {
            "answer": "ok",
            "result_state": {"all_citations": [], "context": []},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    monkeypatch.setattr(routes_module, "agent_run", fake_run)
    real_save = SessionStore.save

    def spy_save(self, session_id, *args, **kwargs):
        saved.append(session_id)
        return real_save(self, session_id, *args, **kwargs)

    monkeypatch.setattr(SessionStore, "save", spy_save)

    app = create_multi_stack_app(settings=settings, stacks=stacks, default_stack="alpha")
    conv = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"messages": conv, "stream": False, "stack": "alpha"},
        )
        client.post(
            "/v1/chat/completions",
            json={"messages": conv, "stream": False, "stack": "beta"},
        )

    assert len(saved) == 2
    assert saved[0] != saved[1], "Different stacks must produce different session ids"
