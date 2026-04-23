"""Tests for the native Azure OpenAI Responses web_search provider."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai import APIStatusError, RateLimitError

from inqtrix.exceptions import (
    AgentRateLimited,
    AgentTimeout,
    AzureOpenAIWebSearchAPIError,
)


def _url_citation(url: str, title: str = ""):
    return SimpleNamespace(type="url_citation", url=url, title=title)


def _output_text(text: str, annotations: list | None = None):
    return SimpleNamespace(
        type="output_text",
        text=text,
        annotations=annotations or [],
    )


def _output_message(text: str, annotations: list | None = None):
    return SimpleNamespace(
        type="message",
        content=[_output_text(text, annotations)],
    )


def _web_search_call(sources: list | None = None, query: str = "query"):
    return SimpleNamespace(
        type="web_search_call",
        action=SimpleNamespace(type="search", query=query, sources=sources or []),
    )


def _response(
    text: str = "",
    annotations: list | None = None,
    sources: list | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
):
    output: list = []
    if sources is not None:
        output.append(_web_search_call(sources=sources))
    if text:
        output.append(_output_message(text, annotations))
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(output=output, usage=usage, output_text=text)


@pytest.fixture()
def mock_azure_openai_web_search():
    mock_client = MagicMock()
    mock_client.responses.create.return_value = _response(
        text="Aktuelle Entwicklungen zeigen...",
        annotations=[_url_citation("https://example.com/story", "Story")],
    )

    with patch("inqtrix.providers.azure_openai_web_search.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = mock_client
        from inqtrix.providers.azure_openai_web_search import AzureOpenAIWebSearch
        yield AzureOpenAIWebSearch, mock_client


def test_requires_exactly_one_endpoint(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    with pytest.raises(ValueError, match="exactly one"):
        Cls(
            azure_endpoint="https://test.openai.azure.com/",
            base_url="https://test.openai.azure.com/openai/v1/",
            api_key="test-key",
        )


def test_requires_auth_mode(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    with pytest.raises(ValueError, match="must be provided"):
        Cls(azure_endpoint="https://test.openai.azure.com/")


def test_rejects_ai_project_endpoint(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    with pytest.raises(ValueError, match="AI Project endpoint"):
        Cls(
            azure_endpoint="https://my-project.services.ai.azure.com/api",
            api_key="test-key",
        )


def test_uses_v1_base_url_from_endpoint(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    with patch("inqtrix.providers.azure_openai_web_search.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        Cls(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["base_url"] == "https://test.openai.azure.com/openai/v1/"
    assert call_kwargs["api_key"] == "test-key"


def test_accepts_token_provider(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    token_provider = MagicMock(return_value="bearer-token")

    with patch("inqtrix.providers.azure_openai_web_search.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        Cls(
            azure_endpoint="https://test.openai.azure.com/",
            azure_ad_token_provider=token_provider,
        )

    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["api_key"] is token_provider


def test_service_principal_auth_builds_token_provider(mock_azure_openai_web_search):
    """tenant_id+client_id+client_secret build an internal token provider."""
    Cls, _ = mock_azure_openai_web_search
    fake_token_provider = MagicMock(return_value="bearer-token")

    with (
        patch("inqtrix.providers.azure_openai_web_search.OpenAI") as mock_openai,
        patch(
            "inqtrix.providers.azure_openai_web_search.build_azure_openai_token_provider",
            return_value=fake_token_provider,
        ) as mock_build,
    ):
        mock_openai.return_value = MagicMock()
        Cls(
            azure_endpoint="https://test.openai.azure.com/",
            tenant_id="tenant-123",
            client_id="client-456",
            client_secret="secret-789",
        )

    mock_build.assert_called_once_with(
        credential=None,
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret-789",
        scope="https://cognitiveservices.azure.com/.default",
    )
    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["api_key"] is fake_token_provider


def test_explicit_credential_builds_token_provider(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    custom_credential = MagicMock(name="custom-credential")
    fake_token_provider = MagicMock(return_value="bearer-token")

    with (
        patch("inqtrix.providers.azure_openai_web_search.OpenAI") as mock_openai,
        patch(
            "inqtrix.providers.azure_openai_web_search.build_azure_openai_token_provider",
            return_value=fake_token_provider,
        ) as mock_build,
    ):
        mock_openai.return_value = MagicMock()
        Cls(
            azure_endpoint="https://test.openai.azure.com/",
            credential=custom_credential,
        )

    assert mock_build.call_args.kwargs["credential"] is custom_credential
    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["api_key"] is fake_token_provider


def test_partial_service_principal_fields_raise(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    with pytest.raises(ValueError, match="must all be provided together"):
        Cls(
            azure_endpoint="https://test.openai.azure.com/",
            tenant_id="t",
            client_secret="s",
        )


def test_api_key_and_service_principal_mutually_exclusive(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    with pytest.raises(ValueError, match="mutually exclusive"):
        Cls(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="static-key",
            tenant_id="t",
            client_id="c",
            client_secret="s",
        )


def test_credential_and_token_provider_mutually_exclusive(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    with pytest.raises(ValueError, match="mutually exclusive"):
        Cls(
            azure_endpoint="https://test.openai.azure.com/",
            credential=MagicMock(),
            azure_ad_token_provider=lambda: "tok",
        )


def test_custom_token_scope_propagates(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search

    with (
        patch("inqtrix.providers.azure_openai_web_search.OpenAI") as mock_openai,
        patch(
            "inqtrix.providers.azure_openai_web_search.build_azure_openai_token_provider",
            return_value=MagicMock(),
        ) as mock_build,
    ):
        mock_openai.return_value = MagicMock()
        Cls(
            azure_endpoint="https://test.openai.azure.com/",
            credential=MagicMock(),
            token_scope="https://custom.scope/.default",
        )

    assert mock_build.call_args.kwargs["scope"] == "https://custom.scope/.default"


def test_search_calls_responses_create_with_web_search_tool(mock_azure_openai_web_search):
    Cls, mock_client = mock_azure_openai_web_search
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        default_model="gpt-4.1-deployment",
    )

    provider.search("Please browse the web")

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4.1-deployment"
    assert call_kwargs["input"] == "Please browse the web"
    assert call_kwargs["tools"] == [{"type": "web_search"}]
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["include"] == ["web_search_call.action.sources"]


def test_search_includes_user_location(mock_azure_openai_web_search):
    Cls, mock_client = mock_azure_openai_web_search
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        user_location={"type": "approximate", "country": "DE"},
    )

    provider.search("Browse")

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["tools"] == [
        {
            "type": "web_search",
            "user_location": {"type": "approximate", "country": "DE"},
        }
    ]


def test_positive_domain_filter_maps_to_allowed_domains(mock_azure_openai_web_search):
    Cls, mock_client = mock_azure_openai_web_search
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    provider.search("Browse", domain_filter=["pubmed.ncbi.nlm.nih.gov"])

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["tools"] == [
        {
            "type": "web_search",
            "filters": {
                "allowed_domains": ["pubmed.ncbi.nlm.nih.gov"],
            },
        }
    ]


def test_negative_domain_filter_uses_query_fallback(mock_azure_openai_web_search):
    Cls, mock_client = mock_azure_openai_web_search
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    provider.search("Browse", domain_filter=["-pinterest.com", "-reddit.com"])

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["input"] == "Browse -site:pinterest.com -site:reddit.com"


def test_search_returns_annotations_as_citations(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    result = provider.search("Browse")

    assert result["answer"] == "Aktuelle Entwicklungen zeigen..."
    assert result["citations"] == ["https://example.com/story"]
    assert result["related_questions"] == []
    assert result["_prompt_tokens"] == 10
    assert result["_completion_tokens"] == 20


def test_sources_fallback_used_when_annotations_missing(mock_azure_openai_web_search):
    Cls, mock_client = mock_azure_openai_web_search
    mock_client.responses.create.return_value = _response(
        text="Antwort ohne Annotationen",
        annotations=[],
        sources=[
            {"url": "https://example.com/source-1"},
            SimpleNamespace(url="https://example.com/source-2"),
        ],
    )
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    result = provider.search("Browse")

    assert result["citations"] == [
        "https://example.com/source-1",
        "https://example.com/source-2",
    ]


def test_supported_search_parameters_are_conservative(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    assert provider.supported_search_parameters == frozenset({"domain_filter"})


def test_rate_limit_raises_agent_rate_limited(mock_azure_openai_web_search):
    Cls, mock_client = mock_azure_openai_web_search
    mock_client.responses.create.side_effect = RateLimitError(
        "rate limit",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    with pytest.raises(AgentRateLimited):
        provider.search("Browse")


def test_deadline_exceeded_raises_agent_timeout(mock_azure_openai_web_search):
    Cls, _ = mock_azure_openai_web_search
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    with pytest.raises(AgentTimeout):
        provider.search("Browse", deadline=time.monotonic() - 10)


def test_api_status_error_is_wrapped(mock_azure_openai_web_search):
    Cls, mock_client = mock_azure_openai_web_search
    response = MagicMock()
    response.headers = {"apim-request-id": "req-123"}
    exc = APIStatusError(
        "bad request",
        response=response,
        body={"error": {"code": "invalid_request", "message": "Bad request"}},
    )
    mock_client.responses.create.side_effect = exc
    provider = Cls(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    with pytest.raises(AzureOpenAIWebSearchAPIError) as caught:
        provider.search("Browse")

    assert "invalid_request" in str(caught.value)
    assert "req-123" in str(caught.value)
