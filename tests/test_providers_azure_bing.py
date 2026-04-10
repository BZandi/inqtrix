"""Tests for the Azure Foundry Bing Search provider adapter."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from inqtrix.exceptions import (
    AgentRateLimited,
    AgentTimeout,
    AzureFoundryBingAPIError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use string sentinels that match what the provider compares against.
# The real MessageRole.AGENT is an enum value; we import it from the
# provider module so mock messages use the identical object.
from azure.ai.agents.models import MessageRole as _MR


def _text_content(text: str, annotations: list | None = None):
    """Build a mock message content item with text and optional annotations."""
    item = SimpleNamespace()
    item.text = SimpleNamespace()
    item.text.value = text
    item.text.annotations = annotations or []
    return item


def _url_annotation(url: str):
    """Build a mock Bing grounding annotation with a URL."""
    return SimpleNamespace(url=url)


def _agent_message(text: str, annotations: list | None = None):
    """Build a mock Agent message."""
    return SimpleNamespace(
        role=_MR.AGENT,
        content=[_text_content(text, annotations)],
    )


def _completed_run(thread_id: str = "thread-test-123"):
    return SimpleNamespace(status="completed", last_error=None, thread_id=thread_id)


def _failed_run(error: str = "internal_error", thread_id: str = "thread-test-123"):
    return SimpleNamespace(status="failed", last_error=error, thread_id=thread_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_foundry():
    """Patch AIProjectClient so tests run without Azure credentials."""
    mock_client = MagicMock()

    # Default happy-path wiring
    mock_client.agents.create_thread_and_process_run.return_value = _completed_run()
    mock_client.agents.messages.list.return_value = [
        _agent_message(
            "Die GKV-Reform bringt folgende Aenderungen...",
            annotations=[
                _url_annotation("https://example.com/reform"),
                _url_annotation("https://example.com/details"),
            ],
        ),
    ]

    with patch("inqtrix.providers_azure_bing.AIProjectClient") as mock_cls, \
         patch("inqtrix.providers_azure_bing.DefaultAzureCredential") as mock_cred:
        mock_cls.return_value = mock_client
        mock_cred.return_value = MagicMock()

        from inqtrix.providers_azure_bing import AzureFoundryBingSearch
        yield AzureFoundryBingSearch, mock_client


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_requires_project_endpoint(mock_foundry):
    Cls, _ = mock_foundry
    with pytest.raises(ValueError, match="project_endpoint"):
        Cls(project_endpoint="", agent_id="agent-1")


def test_requires_agent_id(mock_foundry):
    Cls, _ = mock_foundry
    with pytest.raises(ValueError, match="agent_id"):
        Cls(project_endpoint="https://test.ai.azure.com/api", agent_id="")


def test_is_available_when_configured(mock_foundry):
    Cls, _ = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )
    assert provider.is_available()


def test_construction_with_service_principal(mock_foundry):
    Cls, _ = mock_foundry
    with patch("inqtrix.providers_azure_bing.ClientSecretCredential") as mock_sp:
        mock_sp.return_value = MagicMock()
        provider = Cls(
            project_endpoint="https://test.ai.azure.com/api",
            agent_id="agent-1",
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


def test_construction_with_explicit_credential(mock_foundry):
    Cls, _ = mock_foundry
    custom_cred = MagicMock()
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
        credential=custom_cred,
    )
    assert provider._credential is custom_cred


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_search_returns_answer_and_citations(mock_foundry):
    Cls, mock_client = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    result = provider.search("GKV Reform aktueller Stand")

    assert "GKV-Reform" in result["answer"]
    assert "https://example.com/reform" in result["citations"]
    assert "https://example.com/details" in result["citations"]
    assert result["related_questions"] == []
    assert isinstance(result["_prompt_tokens"], int)
    assert isinstance(result["_completion_tokens"], int)


def test_search_calls_create_thread_and_process_run(mock_foundry):
    Cls, mock_client = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    provider.search("Test query")

    mock_client.agents.create_thread_and_process_run.assert_called_once()
    call_kwargs = mock_client.agents.create_thread_and_process_run.call_args.kwargs
    assert call_kwargs["agent_id"] == "agent-1"
    # Thread contains the query as initial message
    thread = call_kwargs["thread"]
    assert thread["messages"][0]["content"] == "Test query"
    assert thread["messages"][0]["role"] == "user"


def test_search_empty_response(mock_foundry):
    Cls, mock_client = mock_foundry
    mock_client.agents.messages.list.return_value = [
        _agent_message(""),
    ]

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    result = provider.search("Empty query")
    assert result["answer"] == ""
    assert result["citations"] == []


def test_search_no_annotations_falls_back_to_url_extraction(mock_foundry):
    Cls, mock_client = mock_foundry
    mock_client.agents.messages.list.return_value = [
        _agent_message(
            "Laut https://www.bmi.bund.de/reform ist die Reform in Kraft.",
            annotations=[],
        ),
    ]

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    result = provider.search("Reform")
    assert "https://www.bmi.bund.de/reform" in result["citations"]


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


def test_domain_filter_appended_to_query(mock_foundry):
    Cls, mock_client = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    provider.search("KI Regulierung", domain_filter=["bmi.bund.de"])

    call_kwargs = mock_client.agents.create_thread_and_process_run.call_args.kwargs
    thread = call_kwargs["thread"]
    assert "site:bmi.bund.de" in thread["messages"][0]["content"]


def test_domain_filter_exclusion(mock_foundry):
    Cls, mock_client = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    provider.search("KI", domain_filter=["-pinterest.com", "-reddit.com"])

    call_kwargs = mock_client.agents.create_thread_and_process_run.call_args.kwargs
    thread = call_kwargs["thread"]
    content = thread["messages"][0]["content"]
    assert "-site:pinterest.com" in content
    assert "-site:reddit.com" in content


def test_unsupported_params_ignored_gracefully(mock_foundry):
    Cls, _ = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    # Should not raise
    result = provider.search(
        "Test",
        search_context_size="low",
        search_mode="academic",
        return_related=True,
    )
    assert result["related_questions"] == []


def test_additional_instructions_for_recency(mock_foundry):
    Cls, mock_client = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    provider.search("Test", recency_filter="day")

    call_kwargs = mock_client.agents.create_thread_and_process_run.call_args.kwargs
    assert "24 Stunden" in call_kwargs.get("instructions", "")


def test_additional_instructions_for_language(mock_foundry):
    Cls, mock_client = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    provider.search("Test", language_filter=["en"])

    call_kwargs = mock_client.agents.create_thread_and_process_run.call_args.kwargs
    assert "en" in call_kwargs.get("instructions", "")


def test_no_additional_instructions_when_no_hints(mock_foundry):
    Cls, mock_client = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    provider.search("Test")

    call_kwargs = mock_client.agents.create_thread_and_process_run.call_args.kwargs
    assert "instructions" not in call_kwargs


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_failed_run_returns_empty_with_notice(mock_foundry):
    Cls, mock_client = mock_foundry
    mock_client.agents.create_thread_and_process_run.return_value = _failed_run("quota_exceeded")

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    result = provider.search("Test")
    assert result["answer"] == ""
    assert result["citations"] == []
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "fehlgeschlagen" in notice


def test_sdk_exception_returns_empty_with_notice(mock_foundry):
    Cls, mock_client = mock_foundry
    mock_client.agents.create_thread_and_process_run.side_effect = RuntimeError("connection refused")

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    result = provider.search("Test")
    assert result["answer"] == ""
    notice = provider.consume_nonfatal_notice()
    assert notice is not None


def test_deadline_exceeded_raises_agent_timeout(mock_foundry):
    Cls, _ = mock_foundry
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    # Deadline in the past
    with pytest.raises(AgentTimeout):
        provider.search("Test", deadline=time.monotonic() - 10)


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


def test_apply_domain_filters_inclusion():
    from inqtrix.providers_azure_bing import AzureFoundryBingSearch
    result = AzureFoundryBingSearch._apply_domain_filters(
        "query", ["example.com", "test.de"]
    )
    assert result == "query site:example.com site:test.de"


def test_apply_domain_filters_exclusion():
    from inqtrix.providers_azure_bing import AzureFoundryBingSearch
    result = AzureFoundryBingSearch._apply_domain_filters(
        "query", ["-spam.com"]
    )
    assert result == "query -site:spam.com"


def test_apply_domain_filters_empty():
    from inqtrix.providers_azure_bing import AzureFoundryBingSearch
    assert AzureFoundryBingSearch._apply_domain_filters("query", None) == "query"
    assert AzureFoundryBingSearch._apply_domain_filters("query", []) == "query"


def test_build_additional_instructions_recency():
    from inqtrix.providers_azure_bing import AzureFoundryBingSearch
    result = AzureFoundryBingSearch._build_additional_instructions("week", None)
    assert "Woche" in result


def test_build_additional_instructions_language():
    from inqtrix.providers_azure_bing import AzureFoundryBingSearch
    result = AzureFoundryBingSearch._build_additional_instructions(None, ["de"])
    assert "de" in result


def test_build_additional_instructions_combined():
    from inqtrix.providers_azure_bing import AzureFoundryBingSearch
    result = AzureFoundryBingSearch._build_additional_instructions("month", ["en"])
    assert "Monats" in result
    assert "en" in result


def test_build_additional_instructions_none():
    from inqtrix.providers_azure_bing import AzureFoundryBingSearch
    assert AzureFoundryBingSearch._build_additional_instructions(None, None) is None
    assert AzureFoundryBingSearch._build_additional_instructions("", []) is None


# ---------------------------------------------------------------------------
# create_agent classmethod
# ---------------------------------------------------------------------------


def test_create_agent_returns_provider(mock_foundry):
    Cls, mock_client = mock_foundry

    # Mock the agent creation response
    created_agent = SimpleNamespace(id="agent-new-123")
    mock_client.agents.create_agent.return_value = created_agent

    with patch("inqtrix.providers_azure_bing.BingGroundingTool") as mock_tool:
        mock_tool.return_value = MagicMock(definitions=[{"type": "bing_grounding"}])

        provider = Cls.create_agent(
            project_endpoint="https://test.ai.azure.com/api",
            bing_connection_id="/subscriptions/.../connections/bing",
            model="gpt-4o",
            market="de-DE",
            freshness="Week",
        )

    assert provider._agent_id == "agent-new-123"
    assert provider.is_available()
    mock_client.agents.create_agent.assert_called_once()
