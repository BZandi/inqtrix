"""VCR-replay tests for the Perplexity Sonar search provider.

Cassettes live under ``tests/fixtures/cassettes/perplexity/``. The
provider speaks the OpenAI chat-completions protocol, so the cassettes
mirror the JSON shape of Sonar responses through both a LiteLLM proxy
and the direct ``api.perplexity.ai`` endpoint.

Cache notes: every test instantiates a fresh ``PerplexitySearch``
because the in-memory TTLCache would otherwise short-circuit the second
call with the cached result of the first.
"""

from __future__ import annotations

import pathlib

import pytest

from inqtrix.exceptions import AgentRateLimited
from inqtrix.providers.perplexity import PerplexitySearch

pytestmark = pytest.mark.replay


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    """Pin Perplexity cassettes under ``tests/fixtures/cassettes/perplexity/``."""
    return str(
        pathlib.Path(__file__).resolve().parent.parent
        / "fixtures"
        / "cassettes"
        / "perplexity"
    )


def _build_proxy(*, max_retries: int | None = None) -> PerplexitySearch:
    """Construct a Perplexity provider routed through a local LiteLLM proxy."""
    provider = PerplexitySearch(
        api_key="test-key",
        base_url="http://localhost:4000/v1",
        model="perplexity-sonar-pro-agent",
    )
    if max_retries is not None:
        provider._client = provider._client.with_options(max_retries=max_retries)
    return provider


def _build_direct(*, max_retries: int | None = None) -> PerplexitySearch:
    """Construct a Perplexity provider talking directly to api.perplexity.ai."""
    provider = PerplexitySearch(
        api_key="test-key",
        base_url="https://api.perplexity.ai",
        model="sonar-pro",
    )
    if max_retries is not None:
        provider._client = provider._client.with_options(max_retries=max_retries)
    return provider


@pytest.mark.vcr
def test_search_success_proxy_replay() -> None:
    """A normal Sonar response (proxy mode) yields content + citations."""
    provider = _build_proxy()

    result = provider.search("Was ist Inqtrix?")

    assert "Inqtrix" in result["answer"]
    assert result["citations"] == [
        "https://example.com/inqtrix-audit",
        "https://example.com/inqtrix-providers",
    ]
    assert result["related_questions"] == [
        "Wie funktioniert die LangGraph-Integration?",
        "Welche Stop-Kriterien nutzt Inqtrix?",
    ]
    assert result["_prompt_tokens"] == 14
    assert result["_completion_tokens"] == 28
    assert provider.consume_nonfatal_notice() is None


@pytest.mark.vcr
def test_search_success_direct_replay() -> None:
    """Direct-mode requests against api.perplexity.ai produce the same shape."""
    provider = _build_direct()
    assert provider._direct_mode is True

    result = provider.search("Direct path test")

    assert "Direkter Perplexity-Pfad" in result["answer"]
    assert result["citations"] == ["https://example.com/perplexity-direct"]
    assert result["_prompt_tokens"] == 9
    assert result["_completion_tokens"] == 17


@pytest.mark.vcr
def test_empty_response_replay() -> None:
    """Empty content response sets a nonfatal notice but returns successfully."""
    provider = _build_proxy()

    result = provider.search("Sehr seltsame Frage ohne Treffer")

    assert result["answer"] == ""
    assert result["citations"] == []
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "lieferte keine Textantwort" in notice


@pytest.mark.vcr
def test_rate_limit_replay() -> None:
    """A 429 is escalated as ``AgentRateLimited`` (fatal for the run)."""
    provider = _build_proxy(max_retries=0)

    with pytest.raises(AgentRateLimited):
        provider.search("Triggert Rate-Limit")


@pytest.mark.vcr
def test_api_error_replay() -> None:
    """A non-429 5xx degrades to the empty result and stores a notice."""
    provider = _build_proxy(max_retries=0)

    result = provider.search("Triggert 502 vom Upstream")

    assert result["answer"] == ""
    assert result["citations"] == []
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "Perplexity-Suche fehlgeschlagen" in notice
