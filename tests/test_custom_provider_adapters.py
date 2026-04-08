"""Tests for the built-in direct provider adapters."""

from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

from inqtrix.exceptions import AnthropicAPIError
from inqtrix.providers_anthropic import AnthropicLLM
from inqtrix.providers_brave import BraveSearch
from inqtrix.strategies import LLMClaimExtractor


def _http_error(
    status_code: int,
    *,
    body: str = "",
    headers: dict[str, str] | None = None,
) -> HTTPError:
    msg = Message()
    for key, value in (headers or {}).items():
        msg[key] = value
    return HTTPError(
        url="https://api.anthropic.com/v1/messages",
        code=status_code,
        msg="error",
        hdrs=msg,
        fp=BytesIO(body.encode("utf-8")),
    )


def test_brave_search_maps_response_and_filters(monkeypatch):
    search = BraveSearch(api_key="brave-key")
    captured: dict[str, object] = {}

    def fake_request_json(*, params, timeout):
        captured["params"] = params
        return {
            "web": {
                "results": [
                    {
                        "url": "https://example.com/report",
                        "title": "Titel",
                        "description": "Beschreibung",
                        "extra_snippets": ["Snippet A"],
                    }
                ]
            }
        }

    monkeypatch.setattr(search, "_request_json", fake_request_json)

    result = search.search(
        "gkv reform",
        recency_filter="week",
        language_filter=["de"],
        domain_filter=["bundesgesundheitsministerium.de", "-reddit.com"],
    )

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["freshness"] == "pw"
    assert params["search_lang"] == "de"
    assert "site:bundesgesundheitsministerium.de" in params["q"]
    assert "-site:reddit.com" in params["q"]
    assert result["citations"] == ["https://example.com/report"]
    assert "Titel" in result["answer"]
    assert result["_prompt_tokens"] == 0
    assert result["_completion_tokens"] == 0


def test_anthropic_llm_parses_response_and_tracks_tokens(monkeypatch):
    llm = AnthropicLLM(
        api_key="anthropic-key",
        default_model="claude-3-7-sonnet-latest",
        summarize_model="claude-3-5-haiku-latest",
    )

    def fake_request_json(*, payload, timeout, deadline=None):
        assert payload["model"] == "claude-3-7-sonnet-latest"
        return {
            "content": [
                {"type": "text", "text": "Hallo"},
                {"type": "text", "text": " Welt"},
            ],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)

    state = {"total_prompt_tokens": 0, "total_completion_tokens": 0}
    response = llm.complete_with_metadata("Frage", state=state)

    assert response.content == "Hallo Welt"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 7
    assert state["total_prompt_tokens"] == 12
    assert state["total_completion_tokens"] == 7


def test_anthropic_llm_summarize_uses_summarize_model(monkeypatch):
    llm = AnthropicLLM(
        api_key="anthropic-key",
        default_model="claude-3-7-sonnet-latest",
        summarize_model="claude-3-5-haiku-latest",
    )

    def fake_request_json(*, payload, timeout, deadline=None):
        assert payload["model"] == "claude-3-5-haiku-latest"
        return {
            "content": [{"type": "text", "text": "Kurzfassung"}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)

    summary = llm.summarize_parallel("Langer Text")

    assert summary == ("Kurzfassung", 3, 2)


def test_anthropic_temperature_in_payload(monkeypatch):
    llm = AnthropicLLM(api_key="key", temperature=0.3)
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.complete("test")
    assert captured["payload"]["temperature"] == 0.3
    assert "thinking" not in captured["payload"]


def test_anthropic_thinking_in_payload(monkeypatch):
    thinking_cfg = {"type": "adaptive"}
    llm = AnthropicLLM(api_key="key", thinking=thinking_cfg)
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    resp = llm.complete_with_metadata("test")
    assert captured["payload"]["thinking"] == {"type": "adaptive"}
    assert "temperature" not in captured["payload"]
    # _extract_text must skip thinking blocks
    assert resp.content == "ok"


def test_anthropic_thinking_in_payload_for_model_override(monkeypatch):
    llm = AnthropicLLM(
        api_key="key",
        classify_model="claude-sonnet-4-6",
        thinking={"type": "adaptive"},
    )
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.complete("test", model=llm.models.effective_classify_model)

    assert captured["payload"]["model"] == "claude-sonnet-4-6"
    assert captured["payload"]["thinking"] == {"type": "adaptive"}


def test_anthropic_temperature_and_thinking_raises():
    import pytest

    with pytest.raises(ValueError, match="mutually exclusive"):
        AnthropicLLM(api_key="key", temperature=0.5, thinking={"type": "adaptive"})


def test_anthropic_thinking_not_in_summarize(monkeypatch):
    llm = AnthropicLLM(api_key="key", thinking={"type": "adaptive"})
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "kurz"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.summarize_parallel("Langer Text")
    assert "thinking" not in captured["payload"]


def test_anthropic_thinking_adjusts_max_tokens(monkeypatch):
    llm = AnthropicLLM(
        api_key="key",
        default_max_tokens=1024,
        thinking={"type": "enabled", "budget_tokens": 8000},
    )
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.complete("test")
    # budget_tokens (8000) >= default_max_tokens (1024) → raised to 9024,
    # then clamped up to _THINKING_MIN_MAX_TOKENS (16384) by the floor check.
    from inqtrix.providers_anthropic import _THINKING_MIN_MAX_TOKENS
    assert captured["payload"]["max_tokens"] == _THINKING_MIN_MAX_TOKENS


def test_anthropic_retries_transient_http_error_before_success(monkeypatch):
    import inqtrix.providers_anthropic as anthropic_module

    llm = AnthropicLLM(api_key="key")
    calls = {"count": 0}

    class _FakeResponse:
        def __init__(self, body: str):
            self._body = body.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._body

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] < 4:
            raise _http_error(
                529,
                body='{"type":"error","error":{"type":"overloaded_error","message":"temporary overload"},"request_id":"req_retry"}',
                headers={"request-id": "req_retry"},
            )
        return _FakeResponse('{"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}}')

    monkeypatch.setattr(anthropic_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(anthropic_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(anthropic_module.random, "uniform", lambda a, b: 1.0)

    assert llm.complete("test") == "ok"
    assert calls["count"] == 4


def test_anthropic_error_includes_request_id_and_body_details(monkeypatch):
    import pytest
    import inqtrix.providers_anthropic as anthropic_module

    llm = AnthropicLLM(api_key="key")

    def fake_urlopen(request, timeout):
        raise _http_error(
            529,
            body='{"type":"error","error":{"type":"overloaded_error","message":"cluster busy"},"request_id":"req_123"}',
            headers={"request-id": "req_123"},
        )

    monkeypatch.setattr(anthropic_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(anthropic_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(anthropic_module.random, "uniform", lambda a, b: 1.0)

    with pytest.raises(AnthropicAPIError) as exc_info:
        llm.complete("test")

    error = exc_info.value
    assert error.status_code == 529
    assert error.error_type == "overloaded_error"
    assert error.request_id == "req_123"
    assert "cluster busy" in str(error)
    assert "request-id=req_123" in str(error)


def test_anthropic_summarize_retries_then_falls_back(monkeypatch):
    import inqtrix.providers_anthropic as anthropic_module

    llm = AnthropicLLM(api_key="key")
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        raise _http_error(
            529,
            body='{"type":"error","error":{"type":"overloaded_error","message":"busy"},"request_id":"req_sum"}',
            headers={"request-id": "req_sum"},
        )

    monkeypatch.setattr(anthropic_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(anthropic_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(anthropic_module.random, "uniform", lambda a, b: 1.0)

    summary = llm.summarize_parallel("Langer Testtext")

    assert summary == ("Langer Testtext"[:800], 0, 0)
    assert calls["count"] == 5


def test_anthropic_claim_extraction_suppresses_thinking(monkeypatch):
    llm = AnthropicLLM(
        api_key="key",
        summarize_model="claude-haiku-4-5",
        thinking={"type": "adaptive"},
    )
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": '{"claims": []}'}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    extractor = LLMClaimExtractor(llm, summarize_model="claude-haiku-4-5")
    monkeypatch.setattr(llm, "_request_json", fake_request_json)

    claims, prompt_tokens, completion_tokens = extractor.extract(
        "Kurzer Text",
        ["https://example.com/report"],
        "Was ist passiert?",
    )

    assert claims == []
    assert prompt_tokens == 1
    assert completion_tokens == 1
    assert captured["payload"]["model"] == "claude-haiku-4-5"
    assert "thinking" not in captured["payload"]


def test_configured_llm_provider_delegates_without_thinking(monkeypatch):
    from inqtrix.providers import ConfiguredLLMProvider
    from inqtrix.settings import ModelSettings

    llm = AnthropicLLM(
        api_key="key",
        summarize_model="claude-haiku-4-5",
        thinking={"type": "adaptive"},
    )
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": '{"claims": []}'}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)

    wrapped = ConfiguredLLMProvider(llm, ModelSettings(
        reasoning_model="claude-sonnet-4-6",
        summarize_model="claude-haiku-4-5",
    ))
    extractor = LLMClaimExtractor(wrapped, summarize_model="claude-haiku-4-5")

    extractor.extract(
        "Kurzer Text",
        ["https://example.com/report"],
        "Was ist passiert?",
    )

    assert "thinking" not in captured["payload"]


def test_anthropic_adaptive_thinking_auto_raises_max_tokens(monkeypatch):
    llm = AnthropicLLM(
        api_key="key",
        default_max_tokens=1024,
        thinking={"type": "adaptive"},
    )
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.complete("test")
    # adaptive thinking (no budget_tokens) with low default_max_tokens
    # → auto-raised to _THINKING_MIN_MAX_TOKENS
    from inqtrix.providers_anthropic import _THINKING_MIN_MAX_TOKENS
    assert captured["payload"]["max_tokens"] == _THINKING_MIN_MAX_TOKENS


def test_anthropic_explicit_budget_above_minimum_not_clamped(monkeypatch):
    llm = AnthropicLLM(
        api_key="key",
        default_max_tokens=1024,
        thinking={"type": "enabled", "budget_tokens": 20000},
    )
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.complete("test")
    # budget_tokens=20000 → max_tokens = 20000+1024 = 21024, which is > minimum
    assert captured["payload"]["max_tokens"] == 20000 + 1024
