"""VCR-replay tests for the Brave Search provider (urllib transport).

Brave's transport is plain ``urllib.request.urlopen`` over GET, so VCR
intercepts the call through ``http.client.HTTPConnection``. The
cassettes encode the JSON shape Brave's Web Search v1 endpoint returns
(``web.results[].{url,title,description,extra_snippets}``).
"""

from __future__ import annotations

import pathlib

import pytest

from inqtrix.exceptions import AgentRateLimited
from inqtrix.providers.brave import BraveSearch

pytestmark = pytest.mark.replay


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    """Pin Brave cassettes under ``tests/fixtures/cassettes/brave/``."""
    return str(
        pathlib.Path(__file__).resolve().parent.parent
        / "fixtures"
        / "cassettes"
        / "brave"
    )


def _build_provider(*, result_count: int = 10) -> BraveSearch:
    """Construct a Brave provider against the canonical Brave endpoint."""
    return BraveSearch(api_key="test-key", result_count=result_count)


@pytest.mark.vcr
def test_search_success_replay() -> None:
    """A normal Brave response yields concatenated answer + citations."""
    provider = _build_provider()

    result = provider.search("Inqtrix Architektur")

    assert "Inqtrix Architektur" in result["answer"]
    assert "LangGraph" in result["answer"]
    assert "Provider als Plug-ins" in result["answer"]
    assert result["citations"] == [
        "https://example.com/inqtrix-architektur",
        "https://example.com/inqtrix-providers",
    ]
    assert result["related_questions"] == []
    assert result["_prompt_tokens"] == 0
    assert result["_completion_tokens"] == 0
    assert provider.consume_nonfatal_notice() is None


@pytest.mark.vcr
def test_empty_replay() -> None:
    """An empty Brave response stores a notice but does not raise."""
    provider = _build_provider()

    result = provider.search("ganz seltsame Frage")

    assert result["answer"] == ""
    assert result["citations"] == []
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "lieferte keine Textantwort" in notice


@pytest.mark.vcr
def test_rate_limit_replay() -> None:
    """A 429 from Brave is escalated immediately as ``AgentRateLimited``."""
    provider = _build_provider()

    with pytest.raises(AgentRateLimited):
        provider.search("triggert rate-limit")


@pytest.mark.vcr
def test_api_error_replay() -> None:
    """A non-429 HTTPError degrades to the empty result + nonfatal notice."""
    provider = _build_provider()

    result = provider.search("triggert 500er")

    assert result["answer"] == ""
    assert result["citations"] == []
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "Brave-Suche fehlgeschlagen" in notice


@pytest.mark.vcr
def test_domain_filter_replay() -> None:
    """``domain_filter`` injects ``site:`` operators into the query string.

    The cassette URI carries the expected ``site:`` / ``-site:`` tokens, so
    matching the cassette is itself the assertion that the operator
    injection happened correctly. The response payload is then a normal
    Brave shape with one result.
    """
    provider = _build_provider(result_count=8)

    result = provider.search(
        "Inqtrix",
        search_context_size="medium",
        domain_filter=["example.com", "-pinterest.com"],
    )

    assert result["citations"] == ["https://example.com/inqtrix-domain"]
    assert "Inqtrix Domain Filter" in result["answer"]
