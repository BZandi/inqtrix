"""Replay tests for AzureFoundryBingSearch (modern + legacy paths).

The modern Responses-API path is covered by VCR cassettes under
``tests/fixtures/cassettes/azure_foundry_bing/`` (httpx transport,
identical wire shape to ``AzureFoundryWebSearch``).

The legacy ``AIProjectClient.agents.create_thread_and_process_run``
path goes through azure-core's transport pipeline, which VCR cannot
intercept reliably (see ``docs/development/testing-strategy.md`` for
the rationale). That path is therefore covered by a focused MagicMock
test that mirrors the existing ``tests/test_providers_azure_bing.py``
pattern. The MagicMock test is explicitly NOT a cassette but is part
of this module to keep all Foundry-Bing replay coverage in one place.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from azure.ai.agents.models import MessageRole

from inqtrix.exceptions import AgentRateLimited
from inqtrix.providers.azure_bing import AzureFoundryBingSearch

pytestmark = pytest.mark.replay

_PROJECT_ENDPOINT = "https://test.services.ai.azure.com/api/projects/demo"


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    """Pin Foundry-Bing cassettes under ``tests/fixtures/cassettes/azure_foundry_bing/``."""
    return str(
        pathlib.Path(__file__).resolve().parent.parent
        / "fixtures"
        / "cassettes"
        / "azure_foundry_bing"
    )


def _build_with_api_key(*, max_retries: int | None = None) -> AzureFoundryBingSearch:
    provider = AzureFoundryBingSearch(
        project_endpoint=_PROJECT_ENDPOINT,
        agent_name="foundry-bing-agent",
        api_key="test-key",
    )
    if max_retries is not None:
        provider._client = provider._client.with_options(max_retries=max_retries)
    return provider


# ---------------------------------------------------------------------------
# Modern path (VCR cassettes)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_search_success_modern_replay() -> None:
    """Modern Responses-API path returns answer + Bing citations."""
    provider = _build_with_api_key()

    result = provider.search("Foundry-Bing modern path")

    assert "Foundry-Bing-Agent" in result["answer"]
    assert "https://example.com/bing-foundry-1" in result["citations"]
    assert "https://example.com/bing-foundry-2" in result["citations"]


@pytest.mark.vcr
def test_search_success_sp_auth_replay() -> None:
    """SP-auth mints bearer in constructor, then uses the modern Responses API."""
    fake_credential = MagicMock()
    fake_credential.get_token.return_value = SimpleNamespace(token="fake-sp-bearer")

    with patch(
        "inqtrix.providers.azure_bing.ClientSecretCredential",
        return_value=fake_credential,
    ), patch("inqtrix.providers.azure_bing.AIProjectClient") as mock_proj_cls:
        mock_proj_cls.return_value = MagicMock()
        provider = AzureFoundryBingSearch(
            project_endpoint=_PROJECT_ENDPOINT,
            agent_name="foundry-bing-agent",
            tenant_id="fake-tenant",
            client_id="fake-client-id",
            client_secret="fake-client-secret",
        )

    result = provider.search("Bing SP-Auth")
    assert "Service-Principal-Auth" in result["answer"]


@pytest.mark.vcr("test_search_success_modern_replay.yaml")
def test_search_success_credential_replay() -> None:
    """Pre-built ``credential`` mode reuses the modern-path cassette."""
    fake_credential = MagicMock()
    fake_credential.get_token.return_value = SimpleNamespace(token="cred-bearer")

    with patch("inqtrix.providers.azure_bing.AIProjectClient") as mock_proj_cls:
        mock_proj_cls.return_value = MagicMock()
        provider = AzureFoundryBingSearch(
            project_endpoint=_PROJECT_ENDPOINT,
            agent_name="foundry-bing-agent",
            credential=fake_credential,
        )

    result = provider.search("Bing credential")
    assert "Foundry-Bing-Agent" in result["answer"]


@pytest.mark.vcr
def test_empty_response_replay() -> None:
    provider = _build_with_api_key()

    result = provider.search("seltsame Bing-Frage")

    assert result["answer"] == ""
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "lieferte keine Textantwort" in notice


@pytest.mark.vcr
def test_rate_limit_replay() -> None:
    provider = _build_with_api_key(max_retries=0)

    with pytest.raises(AgentRateLimited):
        provider.search("triggert bing rate-limit")


@pytest.mark.vcr
def test_api_error_replay() -> None:
    """Non-429 error degrades to empty-result + nonfatal notice."""
    provider = _build_with_api_key(max_retries=0)

    result = provider.search("triggert invalid agent reference")

    assert result["answer"] == ""
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "Azure-Foundry-Bing-Suche fehlgeschlagen" in notice


# ---------------------------------------------------------------------------
# Legacy path (MagicMock, intentionally not VCR; see module docstring)
# ---------------------------------------------------------------------------


def test_legacy_thread_run_path_returns_answer_with_mock() -> None:
    """Legacy AIProjectClient path goes through azure-core, not httpx.

    This test mocks ``AIProjectClient`` directly because azure-core's
    transport pipeline is not VCR-friendly. It covers the legacy code
    path that resolves an opaque ``agent_id`` and runs Thread/Run/Messages
    against the project client.
    """
    fake_credential = MagicMock()
    fake_credential.get_token.return_value = SimpleNamespace(token="legacy-bearer")

    project_client = MagicMock()

    legacy_message = SimpleNamespace(
        role=MessageRole.AGENT,
        content=[
            SimpleNamespace(
                text=SimpleNamespace(
                    value="Legacy Bing path antwort.",
                    annotations=[
                        SimpleNamespace(url="https://example.com/legacy-bing-1"),
                    ],
                )
            )
        ],
    )

    project_client.agents = SimpleNamespace(
        get_agent=MagicMock(side_effect=Exception("name resolution skipped")),
        create_thread_and_process_run=MagicMock(
            return_value=SimpleNamespace(
                status="completed",
                last_error=None,
                thread_id="thread-legacy-1",
            )
        ),
        messages=SimpleNamespace(
            list=MagicMock(return_value=[legacy_message])
        ),
    )

    with patch(
        "inqtrix.providers.azure_bing.ClientSecretCredential",
        return_value=fake_credential,
    ), patch(
        "inqtrix.providers.azure_bing.AIProjectClient",
        return_value=project_client,
    ):
        provider = AzureFoundryBingSearch(
            project_endpoint=_PROJECT_ENDPOINT,
            agent_id="agent-legacy-id-1",
            tenant_id="fake-tenant",
            client_id="fake-client-id",
            client_secret="fake-client-secret",
        )

    result = provider.search("Legacy Bing query")

    assert "Legacy Bing path antwort" in result["answer"]
    assert "https://example.com/legacy-bing-1" in result["citations"]
