"""Tests for the standardized ``SearchProvider.search_model`` property (ADR-WS-12).

Each in-tree search provider must expose a stable, operator-friendly
identifier via ``search_model``. The ABC default is intentionally loud
(``"<ClassName>(unknown)"``) so a missing override surfaces immediately
in the ``GET /health`` payload rather than silently leaking the global
``Settings.models.search_model`` default.
"""

from __future__ import annotations

import pytest

from inqtrix.providers.base import SearchProvider


# ------------------------------------------------------------------ #
# ABC default — loud "(unknown)" hint
# ------------------------------------------------------------------ #


def test_searchprovider_abc_default_is_loud():
    """A subclass that forgets to override must produce a loud identifier."""
    class _ForgotToOverride(SearchProvider):
        def search(self, query, **_kw):
            return {
                "answer": "",
                "citations": [],
                "related_questions": [],
                "_prompt_tokens": 0,
                "_completion_tokens": 0,
            }

        def is_available(self) -> bool:
            return True

    assert _ForgotToOverride().search_model == "_ForgotToOverride(unknown)"


# ------------------------------------------------------------------ #
# PerplexitySearch
# ------------------------------------------------------------------ #


def test_perplexity_search_model_returns_constructor_model():
    from inqtrix.providers.perplexity import PerplexitySearch

    provider = PerplexitySearch(
        api_key="test-key",
        base_url="https://api.perplexity.ai",
        model="sonar-pro",
    )
    assert provider.search_model == "sonar-pro"


# ------------------------------------------------------------------ #
# BraveSearch
# ------------------------------------------------------------------ #


def test_brave_search_model_constant():
    from inqtrix.providers.brave import BraveSearch

    provider = BraveSearch(api_key="test-key")
    assert provider.search_model == "brave-search-api"


# ------------------------------------------------------------------ #
# AzureOpenAIWebSearch
# ------------------------------------------------------------------ #


def test_azure_openai_web_search_model_includes_tool_suffix():
    from inqtrix.providers.azure_openai_web_search import AzureOpenAIWebSearch

    provider = AzureOpenAIWebSearch(
        azure_endpoint="https://example.openai.azure.com/",
        api_key="test-key",
        default_model="gpt-4.1",
    )
    assert provider.search_model == "gpt-4.1+web_search_tool"


# ------------------------------------------------------------------ #
# AzureFoundryWebSearch
# ------------------------------------------------------------------ #


def test_azure_foundry_web_search_model_format():
    from inqtrix.providers.azure_web_search import AzureFoundryWebSearch

    provider = AzureFoundryWebSearch(
        project_endpoint="https://example.services.ai.azure.com/api/projects/p1",
        agent_name="bing-grounding-agent",
        agent_version="v3",
        api_key="test-key",
    )
    assert provider.search_model == "foundry-web:bing-grounding-agent@v3"


def test_azure_foundry_web_search_model_defaults_to_latest_when_no_version():
    from inqtrix.providers.azure_web_search import AzureFoundryWebSearch

    provider = AzureFoundryWebSearch(
        project_endpoint="https://example.services.ai.azure.com/api/projects/p1",
        agent_name="my-web-agent",
        api_key="test-key",
    )
    assert provider.search_model == "foundry-web:my-web-agent@latest"


# ------------------------------------------------------------------ #
# AzureFoundryBingSearch
# ------------------------------------------------------------------ #


def test_azure_foundry_bing_search_model_format():
    from inqtrix.providers.azure_bing import AzureFoundryBingSearch

    provider = AzureFoundryBingSearch(
        project_endpoint="https://example.services.ai.azure.com/api/projects/p1",
        agent_name="bing-grounding-agent",
        agent_version="v2",
        api_key="test-key",
    )
    assert provider.search_model == "foundry-bing:bing-grounding-agent@v2"


def test_azure_foundry_bing_search_model_defaults_to_latest_when_no_version():
    from inqtrix.providers.azure_bing import AzureFoundryBingSearch

    provider = AzureFoundryBingSearch(
        project_endpoint="https://example.services.ai.azure.com/api/projects/p1",
        agent_name="my-bing-agent",
        api_key="test-key",
    )
    assert provider.search_model == "foundry-bing:my-bing-agent@latest"


def test_azure_foundry_bing_search_model_falls_back_to_agent_id_when_no_name(monkeypatch):
    """Legacy constructor path: only ``agent_id`` is supplied.

    Constructed without going through the full Service-Principal auth
    bootstrap (which would call out to Entra and fail in offline tests);
    the property only depends on ``self._agent_name`` / ``self._agent_id``
    / ``self._agent_version`` — set those directly via instance dict to
    bypass the constructor's eager credential probe.
    """
    from inqtrix.providers.azure_bing import AzureFoundryBingSearch

    instance = AzureFoundryBingSearch.__new__(AzureFoundryBingSearch)
    instance._agent_name = ""
    instance._agent_id = "legacy-agent-id-123"
    instance._agent_version = ""
    assert instance.search_model == "foundry-bing:legacy-agent-id-123@latest"
