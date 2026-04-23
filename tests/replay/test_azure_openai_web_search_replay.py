"""VCR-replay tests for the native Azure OpenAI ``web_search`` tool.

The provider drives the Responses API (``responses.create``) on the
deployment's built-in web-search tool. Cassettes encode the
``output[]`` array shape with both ``message`` items (text + url_citation
annotations) and ``web_search_call`` items (action.sources).
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from inqtrix.exceptions import AgentRateLimited, AzureOpenAIWebSearchAPIError
from inqtrix.providers.azure_openai_web_search import AzureOpenAIWebSearch

pytestmark = pytest.mark.replay

_ENDPOINT = "https://test-resource.openai.azure.com/"


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    """Pin AzureOpenAIWebSearch cassettes under ``tests/fixtures/cassettes/azure_openai_web_search/``."""
    return str(
        pathlib.Path(__file__).resolve().parent.parent
        / "fixtures"
        / "cassettes"
        / "azure_openai_web_search"
    )


def _build_with_api_key(*, max_retries: int | None = None) -> AzureOpenAIWebSearch:
    provider = AzureOpenAIWebSearch(
        azure_endpoint=_ENDPOINT,
        api_key="test-key",
        default_model="my-gpt41-search-deployment",
    )
    if max_retries is not None:
        provider._client = provider._client.with_options(max_retries=max_retries)
    return provider


@pytest.mark.vcr
def test_search_success_api_key_replay() -> None:
    """A normal Responses-API result yields answer + folded citations."""
    provider = _build_with_api_key()

    result = provider.search("Was leistet Azure OpenAI WebSearch?")

    assert "Azure OpenAI WebSearch" in result["answer"]
    # Action sources + annotations both surface in citations.
    assert "https://example.com/azure-websearch-1" in result["citations"]
    assert "https://example.com/azure-websearch-2" in result["citations"]
    assert result["_prompt_tokens"] == 12
    assert result["_completion_tokens"] == 21
    assert result["related_questions"] == []


@pytest.mark.vcr
def test_search_success_sp_auth_replay() -> None:
    """Service-Principal auth produces a valid Responses-API result."""
    fake_token_provider = MagicMock(return_value="fake-bearer")

    with patch("azure.identity.ClientSecretCredential") as mock_cred_cls, \
            patch("azure.identity.get_bearer_token_provider", return_value=fake_token_provider):
        mock_cred_cls.return_value = MagicMock()
        provider = AzureOpenAIWebSearch(
            azure_endpoint=_ENDPOINT,
            tenant_id="fake-tenant",
            client_id="fake-client-id",
            client_secret="fake-client-secret",
            default_model="sp-gpt41-search-deployment",
        )

    result = provider.search("SP-Auth fuer WebSearch")
    assert "Service-Principal-Auth" in result["answer"]


@pytest.mark.vcr("test_search_success_api_key_replay.yaml")
def test_search_success_token_provider_replay() -> None:
    """Custom ``azure_ad_token_provider`` callable wires through unchanged."""
    fake_token_provider = MagicMock(return_value="static-bearer")

    provider = AzureOpenAIWebSearch(
        azure_endpoint=_ENDPOINT,
        azure_ad_token_provider=fake_token_provider,
        default_model="my-gpt41-search-deployment",
    )

    result = provider.search("token-provider WebSearch")
    assert result["citations"]


@pytest.mark.vcr("test_search_success_api_key_replay.yaml")
def test_search_success_credential_replay() -> None:
    """Pre-built ``credential`` object produces a valid result."""
    fake_credential = MagicMock()
    fake_token_provider = MagicMock(return_value="cred-bearer")

    with patch("azure.identity.get_bearer_token_provider", return_value=fake_token_provider):
        provider = AzureOpenAIWebSearch(
            azure_endpoint=_ENDPOINT,
            credential=fake_credential,
            default_model="my-gpt41-search-deployment",
        )

    result = provider.search("credential WebSearch")
    assert result["citations"]


@pytest.mark.vcr
def test_empty_response_replay() -> None:
    """Empty Responses-API output sets a notice but does not raise."""
    provider = _build_with_api_key()

    result = provider.search("ganz seltsame Frage")

    assert result["answer"] == ""
    assert result["citations"] == []
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "lieferte keine Textantwort" in notice


@pytest.mark.vcr
def test_rate_limit_replay() -> None:
    """A 429 escalates as ``AgentRateLimited``."""
    provider = _build_with_api_key(max_retries=0)

    with pytest.raises(AgentRateLimited):
        provider.search("Triggert Rate-Limit")


@pytest.mark.vcr
def test_api_error_replay() -> None:
    """A non-429 surfaces as ``AzureOpenAIWebSearchAPIError`` with details."""
    provider = _build_with_api_key(max_retries=0)

    with pytest.raises(AzureOpenAIWebSearchAPIError) as excinfo:
        provider.search("Triggert invalid-tool-choice")
    err = excinfo.value
    assert err.status_code == 400
    assert err.error_code == "InvalidToolChoice"
    assert err.request_id == "req-aows-replay-400-1"
