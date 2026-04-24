"""Tests for the Azure Foundry Bing Search provider adapter."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai import APIStatusError, RateLimitError

from inqtrix.exceptions import (
    AgentRateLimited,
    AgentTimeout,
    AzureFoundryBingAPIError,
)

from azure.ai.agents.models import MessageRole as _MR


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


def _response(
    text: str = "",
    annotations: list | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
):
    output: list = []
    if text:
        output.append(_output_message(text, annotations))
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(output=output, usage=usage, output_text=text)


def _legacy_text_content(text: str, annotations: list | None = None):
    item = SimpleNamespace()
    item.text = SimpleNamespace()
    item.text.value = text
    item.text.annotations = annotations or []
    return item


def _legacy_url_annotation(url: str):
    return SimpleNamespace(url=url)


def _legacy_agent_message(text: str, annotations: list | None = None):
    return SimpleNamespace(
        role=_MR.AGENT,
        content=[_legacy_text_content(text, annotations)],
    )


def _completed_run(thread_id: str = "thread-test-123"):
    return SimpleNamespace(status="completed", last_error=None, thread_id=thread_id)


@pytest.fixture()
def mock_foundry_bing():
    project_client = MagicMock()
    openai_client = MagicMock()
    credential = MagicMock()
    credential.get_token.return_value = SimpleNamespace(token="fake-token")

    project_client.agents = SimpleNamespace(
        get_agent=MagicMock(return_value=SimpleNamespace(
            id="agent-1",
            name="bing-agent",
            version="7",
        )),
        create_version=MagicMock(return_value=SimpleNamespace(
            id="agent-new-123",
            name="bing-agent",
            version="2",
        )),
        create_agent=MagicMock(return_value=SimpleNamespace(
            id="agent-new-123",
            name="bing-agent",
        )),
        create_thread_and_process_run=MagicMock(return_value=_completed_run()),
        messages=SimpleNamespace(list=MagicMock(return_value=[
            _legacy_agent_message(
                "Die GKV-Reform bringt folgende Aenderungen...",
                annotations=[
                    _legacy_url_annotation("https://example.com/reform"),
                    _legacy_url_annotation("https://example.com/details"),
                ],
            )
        ])),
    )

    openai_client.responses.create.return_value = _response(
        text="Die GKV-Reform bringt folgende Aenderungen...",
        annotations=[
            _url_citation("https://example.com/reform", "Reform"),
            _url_citation("https://example.com/details", "Details"),
        ],
    )

    with patch("inqtrix.providers.azure_bing.AIProjectClient") as mock_project_cls, \
            patch("inqtrix.providers.azure_bing.OpenAI") as mock_openai_cls, \
            patch("inqtrix.providers.azure_bing.DefaultAzureCredential") as mock_default_cred:
        mock_project_cls.return_value = project_client
        mock_openai_cls.return_value = openai_client
        mock_default_cred.return_value = credential

        from inqtrix.providers.azure_bing import AzureFoundryBingSearch

        yield AzureFoundryBingSearch, project_client, openai_client, credential, mock_project_cls, mock_openai_cls


def test_requires_project_endpoint(mock_foundry_bing):
    Cls, *_ = mock_foundry_bing
    with pytest.raises(ValueError, match="project_endpoint"):
        Cls(project_endpoint="", agent_name="bing-agent")


def test_requires_agent_name_or_id(mock_foundry_bing):
    Cls, *_ = mock_foundry_bing
    with pytest.raises(ValueError, match="agent_name oder agent_id"):
        Cls(project_endpoint="https://test.ai.azure.com/api")


def test_is_available_when_configured(mock_foundry_bing):
    Cls, *_ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )
    assert provider.is_available()


def test_construction_with_service_principal(mock_foundry_bing):
    Cls, project_client, _, _, mock_project_cls, _ = mock_foundry_bing
    with patch("inqtrix.providers.azure_bing.ClientSecretCredential") as mock_sp:
        sp_cred = MagicMock()
        sp_cred.get_token.return_value = SimpleNamespace(token="sp-token")
        mock_sp.return_value = sp_cred

        provider = Cls(
            project_endpoint="https://test.ai.azure.com/api",
            agent_name="bing-agent",
            tenant_id="tenant-123",
            client_id="client-456",
            client_secret="secret-789",
        )

        mock_sp.assert_called_once_with(
            tenant_id="tenant-123",
            client_id="client-456",
            client_secret="secret-789",
        )
        mock_project_cls.assert_called()
        assert provider._project_client is project_client


def test_construction_with_api_key(mock_foundry_bing):
    Cls, _, _, _, mock_project_cls, mock_openai_cls = mock_foundry_bing
    mock_project_cls.reset_mock()
    mock_openai_cls.reset_mock()

    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
        api_key="project-key",
    )

    assert provider._credential is None
    assert provider._project_client is None
    call_kwargs = mock_openai_cls.call_args.kwargs
    assert call_kwargs["base_url"] == "https://test.ai.azure.com/api/openai/v1/"
    assert call_kwargs["api_key"] == "project-key"
    mock_project_cls.assert_not_called()


def test_agent_id_only_requires_credentials_when_using_api_key(mock_foundry_bing):
    Cls, *_ = mock_foundry_bing
    with pytest.raises(ValueError, match="agent_name ist erforderlich"):
        Cls(
            project_endpoint="https://test.ai.azure.com/api",
            agent_id="agent-1",
            api_key="project-key",
        )


def test_entra_id_passes_token_string_not_callable(mock_foundry_bing):
    Cls, _, _, _, _, mock_openai_cls = mock_foundry_bing
    mock_openai_cls.reset_mock()

    Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    call_kwargs = mock_openai_cls.call_args.kwargs
    assert isinstance(call_kwargs["api_key"], str)
    assert call_kwargs["api_key"] == "fake-token"


def test_search_returns_answer_and_citations(mock_foundry_bing):
    Cls, _, _, _, _, _ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    result = provider.search("GKV Reform aktueller Stand")

    assert "GKV-Reform" in result["answer"]
    assert result["citations"] == [
        "https://example.com/reform",
        "https://example.com/details",
    ]
    assert result["related_questions"] == []
    assert result["_prompt_tokens"] == 10
    assert result["_completion_tokens"] == 20


def test_search_calls_responses_create_with_agent_reference(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    provider.search("Test query")

    call_kwargs = openai_client.responses.create.call_args.kwargs
    assert call_kwargs["input"] == "Test query"
    assert call_kwargs["tool_choice"] == "required"
    assert call_kwargs["extra_body"] == {
        "agent_reference": {
            "name": "bing-agent",
            "type": "agent_reference",
        }
    }


def test_search_includes_agent_version(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
        agent_version="7",
    )

    provider.search("Test query")

    call_kwargs = openai_client.responses.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "agent_reference": {
            "name": "bing-agent",
            "type": "agent_reference",
            "version": "7",
        }
    }


def test_legacy_agent_id_is_resolved_to_agent_reference(mock_foundry_bing):
    Cls, project_client, openai_client, _, _, _ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    provider.search("Test query")

    project_client.agents.get_agent.assert_called_with("agent-1")
    call_kwargs = openai_client.responses.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "agent_reference": {
            "name": "bing-agent",
            "type": "agent_reference",
            "version": "7",
        }
    }


def test_responses_404_falls_back_to_legacy_thread_run(mock_foundry_bing):
    Cls, project_client, openai_client, _, _, _ = mock_foundry_bing
    response404 = MagicMock()
    response404.status_code = 404
    response404.headers = {}
    err404 = APIStatusError(
        "not found",
        response=response404,
        body={"error": {"code": "not_found", "message": "Agent not found"}},
    )
    openai_client.responses.create.side_effect = [err404, err404]
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    result = provider.search("Test query")

    assert openai_client.responses.create.call_count == 2
    project_client.agents.create_thread_and_process_run.assert_called_once()
    assert "GKV-Reform" in result["answer"]


def test_responses_404_retries_without_auto_resolved_version(mock_foundry_bing):
    Cls, project_client, openai_client, _, _, _ = mock_foundry_bing
    response404 = MagicMock()
    response404.status_code = 404
    response404.headers = {}
    err404 = APIStatusError(
        "not found",
        response=response404,
        body={"error": {"code": "not_found", "message": "version not found"}},
    )
    openai_client.responses.create.side_effect = [
        err404,
        _response(
            text="Die GKV-Reform bringt folgende Aenderungen...",
            annotations=[
                _url_citation("https://example.com/reform", "Reform"),
            ],
        ),
    ]
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    result = provider.search("Test query")

    assert openai_client.responses.create.call_count == 2
    second = openai_client.responses.create.call_args_list[1].kwargs
    assert second["extra_body"]["agent_reference"] == {
        "name": "bing-agent",
        "type": "agent_reference",
    }
    project_client.agents.create_thread_and_process_run.assert_not_called()
    assert "GKV-Reform" in result["answer"]


def test_execution_mode_legacy_skips_get_agent(mock_foundry_bing):
    Cls, project_client, openai_client, _, _, _ = mock_foundry_bing
    openai_client.responses.create.reset_mock()
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
        execution_mode="legacy",
    )

    result = provider.search("Test query")

    project_client.agents.get_agent.assert_not_called()
    openai_client.responses.create.assert_not_called()
    project_client.agents.create_thread_and_process_run.assert_called_once()
    assert "GKV-Reform" in result["answer"]


def test_execution_mode_responses_does_not_fallback_on_404(mock_foundry_bing):
    Cls, project_client, openai_client, _, _, _ = mock_foundry_bing
    response404 = MagicMock()
    response404.status_code = 404
    response404.headers = {}
    openai_client.responses.create.side_effect = APIStatusError(
        "not found",
        response=response404,
        body={"error": {"code": "not_found", "message": "Agent not found"}},
    )
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
        execution_mode="responses",
    )

    result = provider.search("Test query")

    openai_client.responses.create.assert_called_once()
    project_client.agents.create_thread_and_process_run.assert_not_called()
    assert result["answer"] == ""
    assert provider.consume_nonfatal_notice() is not None


def test_execution_mode_legacy_requires_agent_id(mock_foundry_bing):
    Cls, *_ = mock_foundry_bing
    with pytest.raises(ValueError, match="execution_mode='legacy' requires agent_id"):
        Cls(
            project_endpoint="https://test.ai.azure.com/api",
            agent_name="bing-agent",
            execution_mode="legacy",
        )


def test_execution_mode_legacy_rejects_api_key_only(mock_foundry_bing):
    Cls, *_ = mock_foundry_bing
    with pytest.raises(ValueError, match="execution_mode='legacy' requires credential"):
        Cls(
            project_endpoint="https://test.ai.azure.com/api",
            agent_id="agent-1",
            api_key="project-key",
            execution_mode="legacy",
        )


def test_unresolvable_agent_id_falls_back_to_legacy_thread_run(mock_foundry_bing):
    Cls, project_client, openai_client, _, _, _ = mock_foundry_bing
    project_client.agents.get_agent.side_effect = RuntimeError("not found")
    openai_client.responses.create.reset_mock()
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_id="agent-1",
    )

    result = provider.search("Test query")

    openai_client.responses.create.assert_not_called()
    project_client.agents.create_thread_and_process_run.assert_called_once()
    assert "GKV-Reform" in result["answer"]


def test_search_empty_response(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    openai_client.responses.create.return_value = _response(text="", annotations=[])
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    result = provider.search("Empty query")

    assert result["answer"] == ""
    assert result["citations"] == []


def test_search_no_annotations_falls_back_to_url_extraction(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    openai_client.responses.create.return_value = _response(
        text="Laut https://www.bmi.bund.de/reform ist die Reform in Kraft.",
        annotations=[],
    )
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    result = provider.search("Reform")

    assert result["citations"] == ["https://www.bmi.bund.de/reform"]


def test_domain_filter_appended_to_query(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    provider.search("KI Regulierung", domain_filter=["bmi.bund.de"])

    call_kwargs = openai_client.responses.create.call_args.kwargs
    assert "site:bmi.bund.de" in call_kwargs["input"]


def test_domain_filter_exclusion(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    provider.search("KI", domain_filter=["-pinterest.com", "-reddit.com"])

    call_kwargs = openai_client.responses.create.call_args.kwargs
    assert "-site:pinterest.com" in call_kwargs["input"]
    assert "-site:reddit.com" in call_kwargs["input"]


def test_recency_and_language_hints_are_prefixed(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    provider.search("Test", recency_filter="day", language_filter=["en"])

    call_kwargs = openai_client.responses.create.call_args.kwargs
    assert "24 Stunden" in call_kwargs["input"]
    assert "en" in call_kwargs["input"]
    assert call_kwargs["input"].endswith("\n\nTest")


def test_unsupported_params_ignored_gracefully(mock_foundry_bing):
    Cls, *_ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    result = provider.search(
        "Test",
        search_context_size="low",
        search_mode="academic",
        return_related=True,
    )

    assert result["related_questions"] == []


def test_sdk_exception_returns_empty_with_notice(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    openai_client.responses.create.side_effect = RuntimeError("connection refused")
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    result = provider.search("Test")

    assert result["answer"] == ""
    notice = provider.consume_nonfatal_notice()
    assert notice is not None
    assert "fehlgeschlagen" in notice


def test_rate_limit_exception_re_raised(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    openai_client.responses.create.side_effect = RateLimitError(
        "rate limit",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    with pytest.raises(AgentRateLimited):
        provider.search("Test")


def test_api_status_error_non_429_is_wrapped(mock_foundry_bing):
    Cls, _, openai_client, _, _, _ = mock_foundry_bing
    response = MagicMock()
    response.headers = {"apim-request-id": "req-123"}
    openai_client.responses.create.side_effect = APIStatusError(
        "bad request",
        response=response,
        body={"error": {"code": "invalid_request", "message": "Bad request"}},
    )
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    with pytest.raises(AzureFoundryBingAPIError) as caught:
        provider._execute_responses_search("Test", None, 60.0)

    assert "invalid_request" in str(caught.value)
    assert "req-123" in str(caught.value)


def test_deadline_exceeded_raises_agent_timeout(mock_foundry_bing):
    Cls, *_ = mock_foundry_bing
    provider = Cls(
        project_endpoint="https://test.ai.azure.com/api",
        agent_name="bing-agent",
    )

    with pytest.raises(AgentTimeout):
        provider.search("Test", deadline=time.monotonic() - 10)


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


def test_build_additional_instructions_recency():
    from inqtrix.providers.base import _build_recency_language_hints

    result = _build_recency_language_hints("week", None)
    assert "Woche" in result


def test_build_additional_instructions_language():
    from inqtrix.providers.base import _build_recency_language_hints

    result = _build_recency_language_hints(None, ["de"])
    assert "de" in result


def test_build_additional_instructions_combined():
    from inqtrix.providers.base import _build_recency_language_hints

    result = _build_recency_language_hints("month", ["en"])
    assert "Monats" in result
    assert "en" in result


def test_build_additional_instructions_none():
    from inqtrix.providers.base import _build_recency_language_hints

    assert _build_recency_language_hints(None, None) is None
    assert _build_recency_language_hints("", []) is None


def test_create_agent_rejects_api_key(mock_foundry_bing):
    Cls, *_ = mock_foundry_bing

    with pytest.raises(ValueError, match=r"create_agent\(\) benoetigt Azure-Credentials"):
        Cls.create_agent(
            project_endpoint="https://test.ai.azure.com/api",
            bing_connection_id="/subscriptions/.../connections/bing",
            api_key="project-key",
        )


def test_create_agent_returns_provider_from_legacy_sdk(mock_foundry_bing):
    Cls, project_client, _, _, _, _ = mock_foundry_bing
    project_client.agents = SimpleNamespace(
        create_agent=MagicMock(return_value=SimpleNamespace(
            id="agent-new-123",
            name="bing-agent",
        )),
    )

    with patch("inqtrix.providers.azure_bing.BingGroundingTool") as mock_tool:
        mock_tool.return_value = MagicMock(definitions=[{"type": "bing_grounding"}])

        provider = Cls.create_agent(
            project_endpoint="https://test.ai.azure.com/api",
            bing_connection_id="/subscriptions/.../connections/bing",
            model="gpt-4o",
            market="de-DE",
            freshness="Week",
        )

    assert provider._agent_name == "bing-agent"
    assert provider._agent_id == "agent-new-123"
    assert provider.is_available()
    project_client.agents.create_agent.assert_called_once()
