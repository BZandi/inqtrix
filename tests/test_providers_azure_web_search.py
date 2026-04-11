"""Tests for the Azure Foundry Web Search provider adapter (Responses API)."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from inqtrix.exceptions import (
    AgentRateLimited,
    AgentTimeout,
    AzureFoundryWebSearchAPIError,
)


# ---------------------------------------------------------------------------
# Helpers — mock the Responses API return objects
# ---------------------------------------------------------------------------


def _url_citation(url: str, title: str = ""):
    """Build a mock url_citation annotation."""
    return SimpleNamespace(type="url_citation", url=url, title=title,
                           start_index=0, end_index=0)


def _output_text(text: str, annotations: list | None = None):
    """Build a mock ResponseOutputText content item."""
    return SimpleNamespace(type="output_text", text=text,
                           annotations=annotations or [])


def _output_message(text: str, annotations: list | None = None):
    """Build a mock ResponseOutputMessage item."""
    return SimpleNamespace(type="message",
                           content=[_output_text(text, annotations)])


def _response(
    text: str = "",
    annotations: list | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
):
    """Build a mock Responses API Response object."""
    output = [_output_message(text, annotations)] if text else []
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    resp = SimpleNamespace(output=output, usage=usage, output_text=text)
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_foundry_web():
    """Patch OpenAI + azure-identity so tests run without Azure credentials."""
    mock_openai_client = MagicMock()

    # Default happy-path wiring
    mock_openai_client.responses.create.return_value = _response(
        "Die GKV-Reform bringt folgende Aenderungen...",
        annotations=[
            _url_citation("https://example.com/reform", "Reform"),
            _url_citation("https://example.com/details", "Details"),
        ],
    )

    mock_cred_instance = MagicMock()
    mock_cred_instance.get_token.return_value = SimpleNamespace(token="fake-token")

    # azure.identity is now imported lazily inside _resolve_credential,
    # so we patch it on the actual azure.identity module.
    with patch("inqtrix.providers.azure_web_search.OpenAI") as mock_openai_cls, \
            patch("azure.identity.DefaultAzureCredential") as mock_cred:
        mock_openai_cls.return_value = mock_openai_client
        mock_cred.return_value = mock_cred_instance

        from inqtrix.providers.azure_web_search import AzureFoundryWebSearch
        yield AzureFoundryWebSearch, mock_openai_client


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_requires_project_endpoint(mock_foundry_web):
    Cls, _ = mock_foundry_web
    with pytest.raises(ValueError, match="project_endpoint"):
        Cls(project_endpoint="", agent_name="my-agent")


def test_requires_agent_name(mock_foundry_web):
    Cls, _ = mock_foundry_web
    with pytest.raises(ValueError, match="agent_name"):
        Cls(project_endpoint="https://test.ai.azure.com/api", agent_name="")


def test_is_available_when_configured(mock_foundry_web):
    Cls, _ = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )
    assert provider.is_available()


def test_construction_with_service_principal(mock_foundry_web):
    Cls, _ = mock_foundry_web
    with patch("azure.identity.ClientSecretCredential") as mock_sp:
        mock_sp.return_value = MagicMock()
        mock_sp.return_value.get_token.return_value = SimpleNamespace(token="sp-tok")
        provider = Cls(
            project_endpoint="https://test.ai.azure.com/api",
            agent_name="web-search-agent",
            tenant_id="tenant-123",
            client_id="client-456",
            client_secret="secret-789",
        )
        mock_sp.assert_called_once_with(
            tenant_id="tenant-123",
            client_id="client-456",
            client_secret="secret-789",
        )
        assert provider.is_available()


def test_construction_with_explicit_credential(mock_foundry_web):
    Cls, _ = mock_foundry_web
    custom_cred = MagicMock()
    custom_cred.get_token.return_value = SimpleNamespace(token="custom-tok")
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
        credential=custom_cred,
    )
    assert provider._credential is custom_cred


def test_construction_with_api_key():
    """api_key path does not need azure-identity at all."""
    with patch("inqtrix.providers.azure_web_search.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        from inqtrix.providers.azure_web_search import AzureFoundryWebSearch
        provider = AzureFoundryWebSearch(
            project_endpoint="https://test.ai.azure.com/api",
            agent_name="web-search-agent",
            api_key="test-key-123",
        )
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key-123"
        assert provider._credential is None
        assert provider.is_available()


def test_api_key_takes_priority_over_sp():
    """When api_key is set, tenant/client/secret are ignored."""
    with patch("inqtrix.providers.azure_web_search.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        from inqtrix.providers.azure_web_search import AzureFoundryWebSearch
        provider = AzureFoundryWebSearch(
            project_endpoint="https://test.ai.azure.com/api",
            agent_name="web-search-agent",
            api_key="my-key",
            tenant_id="should-be-ignored",
            client_id="should-be-ignored",
            client_secret="should-be-ignored",
        )
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "my-key"
        assert provider._credential is None


def test_openai_client_base_url(mock_foundry_web):
    Cls, _ = mock_foundry_web
    with patch("inqtrix.providers.azure_web_search.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        Cls(
            project_endpoint="https://test.ai.azure.com/api",
            agent_name="web-search-agent",
        )
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "https://test.ai.azure.com/api/openai/v1/"


def test_trailing_slash_stripped_from_endpoint(mock_foundry_web):
    Cls, _ = mock_foundry_web
    with patch("inqtrix.providers.azure_web_search.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        Cls(
            project_endpoint="https://test.ai.azure.com/api/",
            agent_name="web-search-agent",
        )
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "https://test.ai.azure.com/api/openai/v1/"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_search_returns_answer_and_citations(mock_foundry_web):
    Cls, _ = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    result = provider.search("GKV Reform aktueller Stand")

    assert "GKV-Reform" in result["answer"]
    assert "https://example.com/reform" in result["citations"]
    assert "https://example.com/details" in result["citations"]
    assert result["related_questions"] == []
    assert isinstance(result["_prompt_tokens"], int)
    assert isinstance(result["_completion_tokens"], int)


def test_search_calls_responses_create(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    provider.search("Test query")

    mock_client.responses.create.assert_called_once()
    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["input"] == [{"role": "user", "content": "Test query"}]
    agent_ref = call_kwargs["extra_body"]["agent_reference"]
    assert agent_ref["name"] == "web-search-agent"
    assert agent_ref["type"] == "agent_reference"


def test_search_includes_agent_version(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
        agent_version="2",
    )

    provider.search("Test query")

    call_kwargs = mock_client.responses.create.call_args.kwargs
    agent_ref = call_kwargs["extra_body"]["agent_reference"]
    assert agent_ref["version"] == "2"


def test_search_omits_version_when_empty(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    provider.search("Test query")

    call_kwargs = mock_client.responses.create.call_args.kwargs
    agent_ref = call_kwargs["extra_body"]["agent_reference"]
    assert "version" not in agent_ref


def test_search_empty_response(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    mock_client.responses.create.return_value = _response("")

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    result = provider.search("Empty query")
    assert result["answer"] == ""
    assert result["citations"] == []


def test_search_no_annotations_falls_back_to_url_extraction(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    mock_client.responses.create.return_value = _response(
        "Laut https://www.bmi.bund.de/reform ist die Reform in Kraft.",
        annotations=[],
    )

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    result = provider.search("Reform")
    assert "https://www.bmi.bund.de/reform" in result["citations"]


def test_search_deduplicates_citations(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    mock_client.responses.create.return_value = _response(
        "Doppelte Quelle...",
        annotations=[
            _url_citation("https://example.com/same"),
            _url_citation("https://example.com/same"),
        ],
    )

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    result = provider.search("Test")
    assert result["citations"] == ["https://example.com/same"]


def test_search_token_counts(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    mock_client.responses.create.return_value = _response(
        "Test", input_tokens=42, output_tokens=99,
    )

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    result = provider.search("Test")
    assert result["_prompt_tokens"] == 42
    assert result["_completion_tokens"] == 99


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


def test_domain_filter_appended_to_query(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    provider.search("KI Regulierung", domain_filter=["bmi.bund.de"])

    call_kwargs = mock_client.responses.create.call_args.kwargs
    content = call_kwargs["input"][0]["content"]
    assert "site:bmi.bund.de" in content


def test_domain_filter_exclusion(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    provider.search("KI", domain_filter=["-pinterest.com", "-reddit.com"])

    call_kwargs = mock_client.responses.create.call_args.kwargs
    content = call_kwargs["input"][0]["content"]
    assert "-site:pinterest.com" in content
    assert "-site:reddit.com" in content


def test_unsupported_params_ignored_gracefully(mock_foundry_web):
    Cls, _ = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    result = provider.search(
        "Test",
        search_context_size="low",
        search_mode="academic",
        return_related=True,
    )
    assert result["related_questions"] == []


def test_recency_hint_in_user_input(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    provider.search("Test", recency_filter="day")

    call_kwargs = mock_client.responses.create.call_args.kwargs
    content = call_kwargs["input"][0]["content"]
    assert "24 Stunden" in content


def test_language_hint_in_user_input(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    provider.search("Test", language_filter=["en"])

    call_kwargs = mock_client.responses.create.call_args.kwargs
    content = call_kwargs["input"][0]["content"]
    assert "en" in content


def test_no_hint_when_no_filters(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    provider.search("Test")

    call_kwargs = mock_client.responses.create.call_args.kwargs
    content = call_kwargs["input"][0]["content"]
    assert content == "Test"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_sdk_exception_returns_empty_with_notice(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    mock_client.responses.create.side_effect = RuntimeError("connection refused")

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    result = provider.search("Test")
    assert result["answer"] == ""
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "fehlgeschlagen" in notice


def test_rate_limit_exception_re_raised(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    mock_client.responses.create.side_effect = RuntimeError("Rate limit exceeded")

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    with pytest.raises(AgentRateLimited):
        provider.search("Test")


def test_timeout_exception_re_raised(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    mock_client.responses.create.side_effect = RuntimeError("Request timed out")

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    with pytest.raises(AgentTimeout):
        provider.search("Test")


def test_deadline_exceeded_raises_agent_timeout(mock_foundry_web):
    Cls, _ = mock_foundry_web
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    with pytest.raises(AgentTimeout):
        provider.search("Test", deadline=time.monotonic() - 10)


def test_generic_exception_wrapped_in_api_error(mock_foundry_web):
    Cls, mock_client = mock_foundry_web
    mock_client.responses.create.side_effect = ValueError("bad value")

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="web-search-agent",
    )

    # Generic exceptions are caught by the outer handler, not re-raised
    result = provider.search("Test")
    assert result["answer"] == ""
    notice = provider.consume_nonfatal_notice()
    assert notice is not None


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


def test_apply_domain_filters_inclusion():
    from inqtrix.providers.base import _apply_domain_filters
    result = _apply_domain_filters(
        "query", ["example.com", "test.de"]
    )
    assert result == "query site:example.com site:test.de"


def test_apply_domain_filters_exclusion():
    from inqtrix.providers.base import _apply_domain_filters
    result = _apply_domain_filters(
        "query", ["-spam.com"]
    )
    assert result == "query -site:spam.com"


def test_apply_domain_filters_empty():
    from inqtrix.providers.base import _apply_domain_filters
    assert _apply_domain_filters("query", None) == "query"
    assert _apply_domain_filters("query", []) == "query"


def test_build_instructions_recency():
    from inqtrix.providers.base import _build_recency_language_hints
    result = _build_recency_language_hints("week", None)
    assert "Woche" in result


def test_build_instructions_language():
    from inqtrix.providers.base import _build_recency_language_hints
    result = _build_recency_language_hints(None, ["de"])
    assert "de" in result


def test_build_instructions_none_when_no_hints():
    from inqtrix.providers.base import _build_recency_language_hints
    assert _build_recency_language_hints(None, None) is None


# ---------------------------------------------------------------------------
# Exception class
# ---------------------------------------------------------------------------


def test_api_error_formatting():
    err = AzureFoundryWebSearchAPIError(
        agent_name="my-agent",
        status_code=429,
        error_code="RateLimitExceeded",
        message="Too many requests",
    )
    s = str(err)
    assert "my-agent" in s
    assert "429" in s
    assert "RateLimitExceeded" in s
    assert "Too many requests" in s


def test_api_error_minimal():
    err = AzureFoundryWebSearchAPIError(agent_name="a", message="boom")
    assert "boom" in str(err)


# ---------------------------------------------------------------------------
# _parse_response edge cases
# ---------------------------------------------------------------------------


def test_parse_response_no_output():
    from inqtrix.providers.azure_web_search import AzureFoundryWebSearch
    resp = SimpleNamespace(output_text="", output=[], usage=None)
    result = AzureFoundryWebSearch._parse_response(resp)
    assert result["answer"] == ""
    assert result["citations"] == []
    assert result["_prompt_tokens"] == 0
    assert result["_completion_tokens"] == 0


def test_parse_response_non_message_items_skipped():
    from inqtrix.providers.azure_web_search import AzureFoundryWebSearch
    resp = SimpleNamespace(
        output_text="answer",
        output=[
            SimpleNamespace(type="tool_call", content=[]),
            _output_message("answer", [_url_citation("https://a.com")]),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    result = AzureFoundryWebSearch._parse_response(resp)
    assert result["citations"] == ["https://a.com"]
