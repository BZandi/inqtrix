"""Tests for provider response normalization."""

from __future__ import annotations

from unittest.mock import MagicMock

from inqtrix.providers import LiteLLM, PerplexitySearch


def test_litellm_provider_handles_sse_string_response() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = (
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        'data: [DONE]\n\n'
    )
    provider = LiteLLM(api_key="test-key", default_model="gpt-4o")
    provider._client = client  # inject mock

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
        api_key="test-key",
        model="sonar-pro",
        cache_maxsize=16,
        cache_ttl=60,
        _client=client,
    )

    result = search.search("Testsuche")

    assert result["answer"] == "Antwort fertig"
    assert result["citations"] == ["https://example.com"]
    assert result["related_questions"] == ["Was dann?"]
    assert result["_prompt_tokens"] == 7
    assert result["_completion_tokens"] == 11


# -- PerplexitySearch dual-mode (direct vs proxy) extra_body tests --------


def _make_search(base_url: str, direct_mode: bool | None = None) -> tuple[PerplexitySearch, MagicMock]:
    """Create a PerplexitySearch with a mock client."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))],
        usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        model_dump=MagicMock(return_value={
            "choices": [{"message": {"content": "ok"}}],
            "citations": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }),
    )
    kwargs: dict = {
        "api_key": "test",
        "base_url": base_url,
        "model": "sonar-pro",
        "_client": client,
    }
    if direct_mode is not None:
        kwargs["direct_mode"] = direct_mode
    return PerplexitySearch(**kwargs), client


def test_direct_mode_auto_detect_perplexity_url() -> None:
    search, _ = _make_search("https://api.perplexity.ai")
    assert search._direct_mode is True


def test_direct_mode_auto_detect_proxy_url() -> None:
    search, _ = _make_search("http://localhost:4000/v1")
    assert search._direct_mode is False


def test_direct_mode_explicit_override() -> None:
    search, _ = _make_search("http://localhost:4000/v1", direct_mode=True)
    assert search._direct_mode is True

    search2, _ = _make_search("https://api.perplexity.ai", direct_mode=False)
    assert search2._direct_mode is False


def test_proxy_mode_nests_under_web_search_options() -> None:
    """LiteLLM proxy path: params nested inside web_search_options."""
    search, client = _make_search("http://localhost:4000/v1")
    search.search(
        "test query",
        search_context_size="high",
        recency_filter="month",
        language_filter=["de"],
        domain_filter=["-pinterest.com"],
        search_mode="academic",
        return_related=True,
    )
    call_kwargs = client.chat.completions.create.call_args
    extra = call_kwargs.kwargs.get("extra_body") or call_kwargs[1].get("extra_body")
    assert "web_search_options" in extra
    wso = extra["web_search_options"]
    assert wso["search_context_size"] == "high"
    assert wso["search_recency_filter"] == "month"
    assert wso["search_language_filter"] == ["de"]
    assert wso["search_domain_filter"] == ["-pinterest.com"]
    assert wso["search_mode"] == "academic"
    assert wso["return_related_questions"] is True
    assert wso["num_search_results"] == 20


def test_direct_mode_flat_extra_body() -> None:
    """Direct Perplexity API: params as flat top-level keys."""
    search, client = _make_search("https://api.perplexity.ai")
    search.search(
        "test query",
        search_context_size="high",
        recency_filter="week",
        language_filter=["en"],
        domain_filter=["nature.com"],
        search_mode="academic",
        return_related=True,
    )
    call_kwargs = client.chat.completions.create.call_args
    extra = call_kwargs.kwargs.get("extra_body") or call_kwargs[1].get("extra_body")
    # No web_search_options nesting
    assert "web_search_options" not in extra
    assert extra["search_context_size"] == "high"
    assert extra["search_recency_filter"] == "week"
    assert extra["search_language_filter"] == ["en"]
    assert extra["search_domain_filter"] == ["nature.com"]
    assert extra["search_mode"] == "academic"
    assert extra["return_related_questions"] is True
    # num_search_results is NOT sent in direct mode
    assert "num_search_results" not in extra


def test_direct_mode_omits_optional_params_when_not_set() -> None:
    """Direct mode should not include keys for unset optional params."""
    search, client = _make_search("https://api.perplexity.ai")
    search.search("test query")
    call_kwargs = client.chat.completions.create.call_args
    extra = call_kwargs.kwargs.get("extra_body") or call_kwargs[1].get("extra_body")
    assert extra["search_context_size"] == "high"
    assert "search_recency_filter" not in extra
    assert "search_language_filter" not in extra
    assert "search_domain_filter" not in extra
    assert "search_mode" not in extra
    assert "return_related_questions" not in extra
