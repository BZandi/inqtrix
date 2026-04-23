"""Tests for the built-in direct provider adapters."""

from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

from inqtrix.exceptions import AnthropicAPIError
from inqtrix.providers.anthropic import AnthropicLLM
from inqtrix.providers.base import SummarizeOptions
from inqtrix.providers.brave import BraveSearch
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


def test_anthropic_complete_passes_max_output_tokens(monkeypatch):
    llm = AnthropicLLM(api_key="anthropic-key")
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.complete("Frage", max_output_tokens=77)

    assert captured["payload"]["max_tokens"] == 77


def test_anthropic_summarize_uses_custom_options(monkeypatch):
    llm = AnthropicLLM(api_key="anthropic-key")
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "Kurzfassung"}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.summarize_parallel(
        "Langer Text",
        options=SummarizeOptions(
            prompt_template="DEEP:\n",
            input_char_limit=4,
            max_output_tokens=55,
        ),
    )

    assert captured["payload"]["messages"][0]["content"] == "DEEP:\nLang"
    assert captured["payload"]["max_tokens"] == 55


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
    # Phase 8: with thinking enabled, request_max_tokens must reflect the
    # post-clamp value (>= 16384), not the small caller-provided default,
    # so downstream token-utilization logging stays meaningful.
    assert resp.request_max_tokens >= 16384


def test_anthropic_request_max_tokens_without_thinking(monkeypatch):
    """Without thinking, request_max_tokens echoes the caller-supplied value."""
    llm = AnthropicLLM(api_key="key", default_max_tokens=1024)
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    resp = llm.complete_with_metadata("test", max_output_tokens=2000)
    # No thinking → no auto-raise → request_max_tokens == 2000
    assert resp.request_max_tokens == 2000
    assert captured["payload"]["max_tokens"] == 2000


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


def test_anthropic_effort_in_output_config_with_thinking(monkeypatch):
    """Regression: 'effort' must be wrapped in output_config, NOT top-level.

    Live test against the Anthropic Messages API showed that a top-level
    ``effort`` field gets rejected with HTTP 400
    "effort: Extra inputs are not permitted". The correct form is
    ``output_config: {"effort": "..."}``.
    """
    llm = AnthropicLLM(
        api_key="key",
        thinking={"type": "adaptive"},
        effort="medium",
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
    payload = captured["payload"]
    # Top-level 'effort' would be rejected by Anthropic — must NOT be there.
    assert "effort" not in payload
    # Correct location:
    assert payload["output_config"] == {"effort": "medium"}
    # And thinking still configured separately:
    assert payload["thinking"] == {"type": "adaptive"}


def test_anthropic_effort_works_without_thinking(monkeypatch):
    """effort can be used standalone (controls overall token spend)."""
    llm = AnthropicLLM(api_key="key", effort="low")
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.complete("test")
    assert captured["payload"]["output_config"] == {"effort": "low"}
    assert "thinking" not in captured["payload"]


def test_anthropic_effort_accepts_xhigh_and_max():
    """xhigh + max are valid effort levels (Opus 4.7 / Mythos)."""
    AnthropicLLM(api_key="key", effort="xhigh")
    AnthropicLLM(api_key="key", effort="max")


def test_anthropic_effort_invalid_value_raises():
    import pytest

    with pytest.raises(ValueError, match="effort must be one of"):
        AnthropicLLM(api_key="key", effort="ultra-mega")


def test_anthropic_effort_suppressed_inside_without_thinking(monkeypatch):
    """Critical regression: effort must NOT be sent in helper paths.

    Live-test discovery: Claude Haiku 4.5 (typical summarize/claim-extract
    model) rejects ``output_config.effort`` with HTTP 400. The
    ``without_thinking`` context manager — used by claim_extraction and
    available to summarize_parallel — must therefore suppress effort, not
    just thinking.
    """
    llm = AnthropicLLM(
        api_key="key",
        thinking={"type": "adaptive"},
        effort="medium",
    )
    captured: dict[str, object] = {}

    def fake_request_json(*, payload, timeout, deadline=None):
        captured["payload"] = payload
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)

    # Outside without_thinking: both fields present.
    llm.complete("outside")
    assert captured["payload"]["thinking"] == {"type": "adaptive"}
    assert captured["payload"]["output_config"] == {"effort": "medium"}

    # Inside without_thinking: BOTH must be absent.
    captured.clear()
    with llm.without_thinking():
        llm.complete("inside")
    assert "thinking" not in captured["payload"]
    assert "output_config" not in captured["payload"]
    assert "effort" not in captured["payload"]


def test_anthropic_effort_skipped_for_haiku_via_per_call_model(monkeypatch):
    """Phase 12: when effort is configured, calls to Haiku must omit it.

    Helper paths use ``model=summarize_model`` per call. If summarize_model
    is Haiku, the per-call payload must NOT carry output_config.effort,
    even when the surrounding session has effort configured. This is the
    blacklist behaviour that makes Sonnet-helper still get effort while
    Haiku-helper does not.
    """
    llm = AnthropicLLM(
        api_key="key",
        default_model="claude-opus-4-6",
        summarize_model="claude-haiku-4-5",
        thinking={"type": "adaptive"},
        effort="medium",
    )
    captured: list[dict[str, object]] = []

    def fake_request_json(*, payload, timeout, deadline=None):
        captured.append(payload)
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)

    # Reasoning call (Opus) → effort sent.
    llm.complete("question", model="claude-opus-4-6")
    assert captured[-1].get("output_config") == {"effort": "medium"}

    # Helper call to Haiku via complete_with_metadata → effort skipped
    # because the per-call model is on the blacklist.
    llm.complete_with_metadata("snippet", model="claude-haiku-4-5")
    assert "output_config" not in captured[-1]
    assert captured[-1]["model"] == "claude-haiku-4-5"


def test_anthropic_effort_kept_for_sonnet_via_per_call_model(monkeypatch):
    """Phase 12: a Sonnet helper-path call DOES receive effort."""
    llm = AnthropicLLM(
        api_key="key",
        default_model="claude-opus-4-6",
        summarize_model="claude-sonnet-4-6",
        thinking={"type": "adaptive"},
        effort="medium",
    )
    captured: list[dict[str, object]] = []

    def fake_request_json(*, payload, timeout, deadline=None):
        captured.append(payload)
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)
    llm.complete_with_metadata("snippet", model="claude-sonnet-4-6")
    # Sonnet supports effort → output_config present.
    assert captured[-1]["output_config"] == {"effort": "medium"}


def test_anthropic_effort_config_warnings_emitted_for_haiku_role(caplog):
    """Configuring effort + a Haiku role yields a warning the operator can read."""
    import logging

    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        llm = AnthropicLLM(
            api_key="key",
            default_model="claude-opus-4-6",
            summarize_model="claude-haiku-4-5",
            thinking={"type": "adaptive"},
            effort="medium",
        )
    warnings = llm.consume_effort_config_warnings()
    assert len(warnings) == 1
    assert "summarize_model" in warnings[0]
    assert "claude-haiku-4-5" in warnings[0]
    assert "effort='medium'" in warnings[0]
    # Same warning text reached the log
    assert any("summarize_model" in rec.getMessage() and "claude-haiku-4-5" in rec.getMessage()
               for rec in caplog.records)
    # Consume drains the list.
    assert llm.consume_effort_config_warnings() == []


def test_anthropic_effort_no_warning_when_all_models_support_it(caplog):
    """No warning fires when every configured role uses an effort-capable model."""
    import logging

    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        llm = AnthropicLLM(
            api_key="key",
            default_model="claude-opus-4-6",
            summarize_model="claude-sonnet-4-6",
            evaluate_model="claude-sonnet-4-6",
            thinking={"type": "adaptive"},
            effort="medium",
        )
    assert llm.consume_effort_config_warnings() == []


def test_anthropic_effort_no_warning_without_effort_configured():
    """Effort-incompatibility warning only fires when effort is actually set."""
    llm = AnthropicLLM(
        api_key="key",
        summarize_model="claude-haiku-4-5",
        thinking={"type": "adaptive"},
    )
    assert llm.consume_effort_config_warnings() == []


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
    from inqtrix.providers.anthropic import _THINKING_MIN_MAX_TOKENS
    assert captured["payload"]["max_tokens"] == _THINKING_MIN_MAX_TOKENS


def test_anthropic_retries_transient_http_error_before_success(monkeypatch):
    import inqtrix.providers.anthropic as anthropic_module
    import inqtrix.providers.base as base_module

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
    monkeypatch.setattr(base_module.random, "uniform", lambda a, b: 1.0)

    assert llm.complete("test") == "ok"
    assert calls["count"] == 4


def test_anthropic_error_includes_request_id_and_body_details(monkeypatch):
    import pytest
    import inqtrix.providers.anthropic as anthropic_module
    import inqtrix.providers.base as base_module

    llm = AnthropicLLM(api_key="key")

    def fake_urlopen(request, timeout):
        raise _http_error(
            529,
            body='{"type":"error","error":{"type":"overloaded_error","message":"cluster busy"},"request_id":"req_123"}',
            headers={"request-id": "req_123"},
        )

    monkeypatch.setattr(anthropic_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(anthropic_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(base_module.random, "uniform", lambda a, b: 1.0)

    with pytest.raises(AnthropicAPIError) as exc_info:
        llm.complete("test")

    error = exc_info.value
    assert error.status_code == 529
    assert error.error_type == "overloaded_error"
    assert error.request_id == "req_123"
    assert "cluster busy" in str(error)
    assert "request-id=req_123" in str(error)


def test_anthropic_summarize_retries_then_falls_back(monkeypatch):
    import inqtrix.providers.anthropic as anthropic_module
    import inqtrix.providers.base as base_module

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
    monkeypatch.setattr(base_module.random, "uniform", lambda a, b: 1.0)

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
    from inqtrix.providers.base import ConfiguredLLMProvider
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
    from inqtrix.providers.anthropic import _THINKING_MIN_MAX_TOKENS
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
