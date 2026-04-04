"""Tests for provider response normalization."""

from __future__ import annotations

from unittest.mock import MagicMock

from inqtrix.providers import LiteLLMProvider, PerplexitySearch
from inqtrix.settings import AgentSettings, ModelSettings


def _make_models() -> ModelSettings:
    return ModelSettings(
        reasoning_model="gpt-4o",
        search_model="sonar-pro",
        classify_model="gpt-4o-mini",
        summarize_model="gpt-4o-mini",
        evaluate_model="gpt-4o-mini",
    )


def test_litellm_provider_handles_sse_string_response() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = (
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        'data: [DONE]\n\n'
    )
    provider = LiteLLMProvider(
        client=client,
        models=_make_models(),
        agent_settings=AgentSettings(),
    )

    state = {"total_prompt_tokens": 0, "total_completion_tokens": 0}
    response = provider.complete_with_metadata("Hello", state=state)

    assert response.content == "Hello"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 2
    assert state["total_prompt_tokens"] == 3
    assert state["total_completion_tokens"] == 2


def test_perplexity_search_handles_sse_string_response() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = (
        'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Antwort"}}],"citations":["https://example.com"],"related_questions":["Was dann?"]}\n\n'
        'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" fertig"}}],"usage":{"prompt_tokens":7,"completion_tokens":11}}\n\n'
        'data: [DONE]\n\n'
    )
    search = PerplexitySearch(
        client=client,
        model="sonar-pro",
        cache_maxsize=16,
        cache_ttl=60,
    )

    result = search.search("Testsuche")

    assert result["answer"] == "Antwort fertig"
    assert result["citations"] == ["https://example.com"]
    assert result["related_questions"] == ["Was dann?"]
    assert result["_prompt_tokens"] == 7
    assert result["_completion_tokens"] == 11
