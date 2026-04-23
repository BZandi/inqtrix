"""VCR-replay tests for AzureFoundryWebSearch (Responses API + agent_reference).

The provider speaks the Responses API on a project-scoped Foundry
endpoint and references the agent through ``extra_body``. Cassettes
encode the same Responses-API output[] shape as
``AzureOpenAIWebSearch``, only the URL path differs (project endpoint
instead of resource endpoint).

Auth note: when constructed without ``api_key`` the provider eagerly
calls ``credential.get_token(...)`` to mint a static bearer at
construction time. The SP-auth test patches both
``ClientSecretCredential`` (lazily imported in
``resolve_azure_credential``) and the credential's ``get_token``
return shape.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from inqtrix.exceptions import AgentRateLimited
from inqtrix.providers.azure_web_search import AzureFoundryWebSearch

pytestmark = pytest.mark.replay


_PROJECT_ENDPOINT = "https://test.services.ai.azure.com/api/projects/demo"


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    """Pin AzureFoundryWebSearch cassettes under ``tests/fixtures/cassettes/azure_foundry_web_search/``."""
    return str(
        pathlib.Path(__file__).resolve().parent.parent
        / "fixtures"
        / "cassettes"
        / "azure_foundry_web_search"
    )


def _build_with_api_key(*, max_retries: int | None = None) -> AzureFoundryWebSearch:
    provider = AzureFoundryWebSearch(
        project_endpoint=_PROJECT_ENDPOINT,
        agent_name="web-search-agent",
        api_key="test-key",
    )
    if max_retries is not None:
        provider._client = provider._client.with_options(max_retries=max_retries)
    return provider


@pytest.mark.vcr
def test_search_success_api_key_replay() -> None:
    provider = _build_with_api_key()

    result = provider.search("Was leistet Foundry-WebSearch?")

    assert "Foundry-WebSearch" in result["answer"]
    assert "https://example.com/foundry-web-search-1" in result["citations"]
    assert "https://example.com/foundry-web-search-2" in result["citations"]
    assert result["_prompt_tokens"] == 14
    assert result["_completion_tokens"] == 22


@pytest.mark.vcr
def test_search_success_sp_auth_replay() -> None:
    """SP-auth path mints a bearer at construction time."""
    fake_token = SimpleNamespace(token="fake-sp-bearer")
    fake_credential = MagicMock()
    fake_credential.get_token.return_value = fake_token

    with patch("azure.identity.ClientSecretCredential", return_value=fake_credential):
        provider = AzureFoundryWebSearch(
            project_endpoint=_PROJECT_ENDPOINT,
            agent_name="web-search-agent",
            tenant_id="fake-tenant",
            client_id="fake-client-id",
            client_secret="fake-client-secret",
        )

    result = provider.search("SP-Auth Foundry")
    assert "Service-Principal-Auth" in result["answer"]


@pytest.mark.vcr("test_search_success_api_key_replay.yaml")
def test_search_success_credential_replay() -> None:
    """Pre-built ``credential`` mode reuses the api-key cassette."""
    fake_credential = MagicMock()
    fake_credential.get_token.return_value = SimpleNamespace(token="cred-bearer")

    provider = AzureFoundryWebSearch(
        project_endpoint=_PROJECT_ENDPOINT,
        agent_name="web-search-agent",
        credential=fake_credential,
    )

    result = provider.search("credential Foundry")
    assert "Foundry-WebSearch" in result["answer"]


@pytest.mark.vcr("test_search_success_api_key_replay.yaml")
def test_search_success_default_credential_replay() -> None:
    """``DefaultAzureCredential`` fallback (no auth args supplied)."""
    fake_default_cred = MagicMock()
    fake_default_cred.get_token.return_value = SimpleNamespace(token="default-bearer")

    with patch("azure.identity.DefaultAzureCredential", return_value=fake_default_cred):
        provider = AzureFoundryWebSearch(
            project_endpoint=_PROJECT_ENDPOINT,
            agent_name="web-search-agent",
        )

    result = provider.search("default-azure-cred Foundry")
    assert "Foundry-WebSearch" in result["answer"]


@pytest.mark.vcr
def test_empty_response_replay() -> None:
    provider = _build_with_api_key()

    result = provider.search("seltsame Frage")

    assert result["answer"] == ""
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "lieferte keine Textantwort" in notice


@pytest.mark.vcr
def test_rate_limit_replay() -> None:
    provider = _build_with_api_key(max_retries=0)

    with pytest.raises(AgentRateLimited):
        provider.search("triggert rate-limit")


@pytest.mark.vcr
def test_api_error_replay() -> None:
    """A 404 AgentNotFound is caught and degraded to empty + nonfatal notice.

    Note: ``search()`` deliberately swallows non-rate-limit
    ``AzureFoundryWebSearchAPIError`` and returns the empty-result
    dict so a single failing agent does not abort the whole research
    run. The protective contract is exercised here.
    """
    provider = _build_with_api_key(max_retries=0)

    result = provider.search("triggert agent-not-found")

    assert result["answer"] == ""
    assert result["citations"] == []
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "Azure-Foundry-WebSearch fehlgeschlagen" in notice
