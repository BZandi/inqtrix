"""Tests for provider response normalization."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from perplexity import APIError, RateLimitError

from inqtrix.exceptions import AgentRateLimited
from inqtrix.providers.base import get_search_provider_capabilities
from inqtrix.providers.litellm import LiteLLM
from inqtrix.providers.perplexity import PerplexitySearch
from inqtrix.providers.base import _bounded_timeout


def test_litellm_provider_handles_sse_string_response() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = (
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        'data: [DONE]\n\n'
    )
    provider = LiteLLM(api_key="test-key", default_model="gpt-4o")
    provider._client = client  # inject mock

    state = {"total_prompt_tokens": 0, "total_completion_tokens": 0}
    response = provider.complete_with_metadata("Hello", state=state)

    assert response.content == "Hello"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 2
    assert response.finish_reason == "stop"
    assert state["total_prompt_tokens"] == 3
    assert state["total_completion_tokens"] == 2


def test_litellm_complete_passes_max_output_tokens() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))],
        usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        model_dump=MagicMock(return_value={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }),
    )
    provider = LiteLLM(api_key="test-key", default_model="gpt-4o")
    provider._client = client

    provider.complete("Hello", max_output_tokens=42)

    call_kwargs = client.chat.completions.create.call_args
    assert call_kwargs.kwargs["max_tokens"] == 42


def test_litellm_complete_with_metadata_propagates_finish_reason() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"), finish_reason="length")],
        usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        model_dump=MagicMock(return_value={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }),
    )
    provider = LiteLLM(api_key="test-key", default_model="gpt-4o")
    provider._client = client

    response = provider.complete_with_metadata("Hello")

    assert response.content == "ok"
    assert response.finish_reason == "length"
    assert response.raw is not None
    assert response.raw["choices"][0]["finish_reason"] == "length"


def test_litellm_transient_status_retries_with_visible_notice() -> None:
    from openai import APIStatusError

    response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"), finish_reason="stop")],
        usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        model_dump=MagicMock(return_value={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }),
    )
    error_response = MagicMock()
    error_response.status_code = 503
    error_response.headers = {}
    transient_error = APIStatusError(
        message="service unavailable",
        response=error_response,
        body=None,
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [transient_error, response]
    provider = LiteLLM(api_key="test-key", default_model="gpt-4o")
    provider._client = client
    observed = []

    with patch("inqtrix.providers.base._sleep_before_retry") as sleep:
        with provider.observe_retries(lambda notice: observed.append(notice)):
            result = provider.complete_with_metadata("Hello")

    assert result.content == "ok"
    assert client.chat.completions.create.call_count == 2
    assert sleep.call_count == 1
    assert len(observed) == 1
    assert observed[0]["provider"] == "LiteLLM"
    assert observed[0]["error_code"] == "HTTP 503"
    assert observed[0]["progress_emitted"] is True


def _agent_response(*, answer, results, input_tokens=11, output_tokens=28):
    """Build a fake Perplexity Agent API response mirroring the SDK schema."""
    search_results = SimpleNamespace(
        type="search_results",
        queries=["q"],
        results=[
            SimpleNamespace(
                id=item["id"],
                url=item["url"],
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                date=item.get("date", ""),
                last_updated=item.get("last_updated", ""),
                source=item.get("source", ""),
            )
            for item in results
        ],
    )
    message = SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text=answer, annotations=[])],
    )
    return SimpleNamespace(
        id="resp_1",
        model="sonar",
        status="completed",
        output=[search_results, message],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _agent_search(response=None, *, side_effect=None, **kwargs):
    """Construct a PerplexitySearch backed by a stub Agent client."""
    client = MagicMock()
    if side_effect is not None:
        client.responses.create.side_effect = side_effect
    else:
        client.responses.create.return_value = response
    params = {"api_key": "test-key", "_client": client}
    params.update(kwargs)
    return PerplexitySearch(**params), client


def test_perplexity_agent_parses_answer_and_full_sources() -> None:
    long_snippet = "Revenue rose 12 percent. " * 300  # ~7500 chars, must survive uncut
    response = _agent_response(
        answer="NVIDIA reported record revenue[2][3].",
        results=[
            {"id": 1, "url": "https://a.example/report", "title": "A", "snippet": "short A"},
            {
                "id": 2,
                "url": "https://b.example/news",
                "title": "B",
                "snippet": long_snippet,
                "date": "2026-05-20",
                "last_updated": "2026-05-22",
            },
        ],
    )
    search, _ = _agent_search(response)

    result = search.search("NVIDIA Quartalszahlen")

    assert result.answer == "NVIDIA reported record revenue[2][3]."
    assert len(result.sources) == 2
    assert result.sources[1].rank == 2  # Perplexity result id preserved as rank
    assert len(result.sources[1].snippet) == len(long_snippet)  # no cap
    assert result.sources[1].last_updated == "2026-05-22"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 28
    assert result.citation_urls == ["https://a.example/report", "https://b.example/news"]


def test_perplexity_agent_falls_back_to_output_text_when_no_message() -> None:
    # Some Agent responses carry sources but no `message` item; the answer must
    # fall back to the top-level `output_text` so the synthesis is not lost.
    search_results = SimpleNamespace(
        type="search_results",
        results=[
            SimpleNamespace(
                id=1,
                url="https://a.example/report",
                title="A",
                snippet="s",
                date="",
                last_updated="",
                source="",
            )
        ],
    )
    response = SimpleNamespace(
        id="resp_1",
        model="sonar",
        status="completed",
        output=[search_results],  # no message item
        output_text="Fallback synthesis text.",
        usage=SimpleNamespace(input_tokens=5, output_tokens=7),
    )
    search, _ = _agent_search(response)

    result = search.search("frage")

    assert result.answer == "Fallback synthesis text."
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://a.example/report"


def test_perplexity_agent_uses_positional_rank_for_non_integer_id() -> None:
    # A non-integer source id must not crash; rank falls back to position.
    response = _agent_response(
        answer="ok",
        results=[{"id": "abc", "url": "https://a.example/r", "title": "A", "snippet": "s"}],
    )
    search, _ = _agent_search(response)

    result = search.search("frage")

    assert len(result.sources) == 1
    assert result.sources[0].rank == 1  # positional fallback when id is not an int


def test_perplexity_agent_explicit_model_overrides_preset() -> None:
    search, client = _agent_search(_agent_response(answer="ok", results=[]), model="sonar")
    search.search("frage")
    call = client.responses.create.call_args
    assert call.kwargs["model"] == "sonar"
    assert call.kwargs["tools"] == [{"type": "web_search"}]
    assert "preset" not in call.kwargs


def test_perplexity_agent_defaults_to_fast_search_preset() -> None:
    # Out of the box (no model/preset given) the provider uses the fast-search
    # preset, which bundles the inline-citation system prompt.
    search, client = _agent_search(_agent_response(answer="ok", results=[]))
    search.search("frage")
    call = client.responses.create.call_args
    assert call.kwargs["preset"] == "fast-search"
    assert "model" not in call.kwargs
    assert search.search_model == "fast-search"


# -- PerplexitySearch Agent API error and hint handling -------------------


def test_perplexity_agent_domain_filter_injected_into_input() -> None:
    search, client = _agent_search(_agent_response(answer="ok", results=[]))
    search.search("frage", domain_filter=["nature.com", "-pinterest.com"])
    call = client.responses.create.call_args
    assert "nature.com" in call.kwargs["input"]
    assert "pinterest.com" in call.kwargs["input"]


def test_perplexity_agent_empty_response_sets_notice() -> None:
    search, _ = _agent_search(_agent_response(answer="", results=[]))
    result = search.search("nichts")
    assert result.answer == ""
    assert result.sources == []
    notice = search.consume_nonfatal_notice()
    assert notice is not None and "lieferte keine Textantwort" in notice


def test_perplexity_agent_rate_limit_escalates() -> None:
    req = httpx.Request("POST", "https://api.perplexity.ai/responses")
    resp = httpx.Response(429, request=req)
    search, _ = _agent_search(side_effect=RateLimitError("rate", response=resp, body=None))
    with pytest.raises(AgentRateLimited):
        search.search("frage")


def test_perplexity_agent_api_error_degrades_to_empty() -> None:
    req = httpx.Request("POST", "https://api.perplexity.ai/responses")
    search, _ = _agent_search(side_effect=APIError("boom", request=req, body=None))
    result = search.search("frage")
    assert result.answer == ""
    assert result.sources == []
    notice = search.consume_nonfatal_notice()
    assert notice is not None and "Perplexity-Suche fehlgeschlagen" in notice


def test_bounded_timeout_respects_small_remaining_deadline() -> None:
    deadline = time.monotonic() + 0.2
    bounded = _bounded_timeout(120, deadline)

    assert 0 < bounded <= 0.2


def test_search_provider_capabilities_default_to_all_hints() -> None:
    capabilities = get_search_provider_capabilities(object())

    assert capabilities.supports("search_context_size") is True
    assert capabilities.supports("recency_filter") is True
    assert capabilities.supports("language_filter") is True
    assert capabilities.supports("domain_filter") is True
    assert capabilities.supports("search_mode") is True
    assert capabilities.supports("return_related") is True


def test_search_provider_capabilities_resolve_provider_attribute() -> None:
    class _PartialSearchProvider:
        supported_search_parameters = frozenset({
            "search_context_size",
            "recency_filter",
            "language_filter",
            "domain_filter",
        })

    provider = _PartialSearchProvider()
    capabilities = get_search_provider_capabilities(provider)

    assert capabilities.supports("search_context_size") is True
    assert capabilities.supports("domain_filter") is True
    assert capabilities.supports("search_mode") is False
    assert capabilities.supports("return_related") is False
