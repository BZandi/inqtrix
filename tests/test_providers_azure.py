"""Tests for the Azure OpenAI LLM provider adapter."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from inqtrix.exceptions import AgentRateLimited, AgentTimeout, AzureOpenAIAPIError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_completion_response(
    content: str = "Hallo Welt",
    prompt_tokens: int = 12,
    completion_tokens: int = 7,
):
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    choice.delta = None

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model_dump.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return response


@pytest.fixture()
def mock_azure_client():
    """Patch OpenAI so AzureOpenAILLM can be instantiated without real credentials."""
    mock_client = MagicMock()

    with patch("inqtrix.providers_azure.OpenAI") as mock_cls:
        mock_cls.return_value = mock_client
        from inqtrix.providers_azure import AzureOpenAILLM
        yield AzureOpenAILLM, mock_client


def test_client_uses_v1_base_url_from_endpoint(mock_azure_client):
    AzureOpenAILLM, _ = mock_azure_client

    with patch("inqtrix.providers_azure.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        AzureOpenAILLM(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["base_url"] == "https://test.openai.azure.com/openai/v1/"
    assert call_kwargs["api_key"] == "test-key"


def test_client_accepts_explicit_base_url(mock_azure_client):
    AzureOpenAILLM, _ = mock_azure_client

    with patch("inqtrix.providers_azure.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        AzureOpenAILLM(
            base_url="https://test.openai.azure.com/openai/v1/",
            api_key="test-key",
        )

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["base_url"] == "https://test.openai.azure.com/openai/v1/"


def test_ai_project_endpoint_rejected(mock_azure_client):
    AzureOpenAILLM, _ = mock_azure_client

    with pytest.raises(ValueError, match="AI Project endpoint"):
        AzureOpenAILLM(
            azure_endpoint="https://my-project.services.ai.azure.com/api",
            api_key="test-key",
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_complete_returns_text(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response("Hallo Welt")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        default_model="gpt-4o",
    )
    result = llm.complete("Frage")

    assert result == "Hallo Welt"
    mock_client.chat.completions.create.assert_called_once()


def test_complete_with_metadata_returns_tokens(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response(
        "Antwort", 10, 5)

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )
    state = {"total_prompt_tokens": 0, "total_completion_tokens": 0}
    resp = llm.complete_with_metadata("Frage", state=state)

    assert resp.content == "Antwort"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 5
    assert state["total_prompt_tokens"] == 10
    assert state["total_completion_tokens"] == 5


def test_complete_with_system_prompt(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response("ok")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )
    llm.complete("Frage", system="Du bist ein Forscher.")

    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
    assert messages[0] == {"role": "system", "content": "Du bist ein Forscher."}
    assert messages[1] == {"role": "user", "content": "Frage"}


def test_summarize_parallel_happy_path(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response(
        "Kurzfassung", 3, 2)

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )
    summary, p_tok, c_tok = llm.summarize_parallel("Langer Text")

    assert summary == "Kurzfassung"
    assert p_tok == 3
    assert c_tok == 2


def test_summarize_parallel_empty_text(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )
    result = llm.summarize_parallel("   ")

    assert result == ("", 0, 0)
    mock_client.chat.completions.create.assert_not_called()


def test_is_available_returns_true(mock_azure_client):
    AzureOpenAILLM, _ = mock_azure_client

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )
    assert llm.is_available() is True


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------


def test_models_property(mock_azure_client):
    AzureOpenAILLM, _ = mock_azure_client

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        default_model="gpt-4o",
        classify_model="gpt-4o-mini",
        summarize_model="gpt-4o-mini",
        evaluate_model="gpt-4o-mini",
    )
    assert llm.models.reasoning_model == "gpt-4o"
    assert llm.models.classify_model == "gpt-4o-mini"
    assert llm.models.summarize_model == "gpt-4o-mini"
    assert llm.models.evaluate_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


def test_temperature_in_params(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response("ok")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        temperature=0.3,
    )
    llm.complete("test")

    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("temperature") == 0.3


def test_completion_uses_max_completion_tokens_by_default(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response("ok")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        default_max_tokens=123,
    )
    llm.complete("test")

    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("max_completion_tokens") == 123
    assert "max_tokens" not in call_kwargs.kwargs


def test_completion_can_use_max_tokens_when_configured(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response("ok")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        default_max_tokens=321,
        token_budget_parameter="max_tokens",
    )
    llm.complete("test")

    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("max_tokens") == 321
    assert "max_completion_tokens" not in call_kwargs.kwargs


def test_no_temperature_when_not_set(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response("ok")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )
    llm.complete("test")

    call_kwargs = mock_client.chat.completions.create.call_args
    assert "temperature" not in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# without_thinking (noop for Azure/GPT)
# ---------------------------------------------------------------------------


def test_without_thinking_is_noop(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response("ok")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    with llm.without_thinking() as ctx:
        assert ctx is llm
        result = llm.complete("test")

    assert result == "ok"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_rate_limit_raises_agent_rate_limited(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client

    from openai import RateLimitError
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    error = RateLimitError(
        message="rate limit",
        response=mock_response,
        body=None,
    )
    mock_client.chat.completions.create.side_effect = error

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    with pytest.raises(AgentRateLimited):
        llm.complete("test")


def test_api_status_error_429_raises_agent_rate_limited(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client

    from openai import APIStatusError

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    error = APIStatusError(
        message="too many requests",
        response=mock_response,
        body=None,
    )
    mock_client.chat.completions.create.side_effect = error

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    with pytest.raises(AgentRateLimited):
        llm.complete("test")


def test_api_status_error_non_429_raises_azure_api_error(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client

    from openai import APIStatusError

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.headers = {"apim-request-id": "req-404"}
    error = APIStatusError(
        message="deployment not found",
        response=mock_response,
        body={"error": {"code": "DeploymentNotFound", "message": "deployment not found"}},
    )
    mock_client.chat.completions.create.side_effect = error

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    with pytest.raises(AzureOpenAIAPIError) as exc_info:
        llm.complete("test")

    err = exc_info.value
    assert err.status_code == 404
    assert err.model == "gpt-4o"
    assert err.error_code == "DeploymentNotFound"
    assert err.request_id == "req-404"
    assert "Azure-OpenAI-Aufruf fehlgeschlagen" in str(err)


def test_openai_error_raises_azure_api_error(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client

    from openai import OpenAIError

    mock_client.chat.completions.create.side_effect = OpenAIError("connection failed")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    with pytest.raises(AzureOpenAIAPIError) as exc_info:
        llm.complete("test")

    assert "connection failed" in str(exc_info.value)


def test_summarize_falls_back_on_error(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client

    from openai import OpenAIError

    mock_client.chat.completions.create.side_effect = OpenAIError("service error")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )
    text, p_tok, c_tok = llm.summarize_parallel("Langer Testtext " * 100)

    assert text == ("Langer Testtext " * 100)[:800]
    assert p_tok == 0
    assert c_tok == 0
    notice = llm.consume_nonfatal_notice()
    assert notice is not None
    assert "Fallback" in notice


def test_summarize_reraises_rate_limited(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client

    from openai import RateLimitError

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    error = RateLimitError(
        message="rate limit",
        response=mock_response,
        body=None,
    )
    mock_client.chat.completions.create.side_effect = error

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )

    with pytest.raises(AgentRateLimited):
        llm.summarize_parallel("Langer Text")


# ---------------------------------------------------------------------------
# Deadline enforcement
# ---------------------------------------------------------------------------


def test_deadline_raises_agent_timeout(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
    )
    past_deadline = time.monotonic() - 10

    with pytest.raises(AgentTimeout):
        llm.complete("test", deadline=past_deadline)


# ---------------------------------------------------------------------------
# Authentication validation
# ---------------------------------------------------------------------------


def test_api_key_and_token_provider_raises(mock_azure_client):
    AzureOpenAILLM, _ = mock_azure_client

    with pytest.raises(ValueError, match="mutually exclusive"):
        AzureOpenAILLM(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            azure_ad_token_provider=lambda: "token",
        )


def test_no_auth_raises(mock_azure_client):
    AzureOpenAILLM, _ = mock_azure_client

    with pytest.raises(ValueError, match="must be provided"):
        AzureOpenAILLM(
            azure_endpoint="https://test.openai.azure.com/",
        )


def test_endpoint_and_base_url_together_raises(mock_azure_client):
    AzureOpenAILLM, _ = mock_azure_client

    with pytest.raises(ValueError, match="exactly one"):
        AzureOpenAILLM(
            azure_endpoint="https://test.openai.azure.com/",
            base_url="https://test.openai.azure.com/openai/v1/",
            api_key="test-key",
        )


def test_token_provider_auth(mock_azure_client):
    AzureOpenAILLM, mock_client = mock_azure_client
    mock_client.chat.completions.create.return_value = _chat_completion_response("ok")

    llm = AzureOpenAILLM(
        azure_endpoint="https://test.openai.azure.com/",
        azure_ad_token_provider=lambda: "token",
    )
    result = llm.complete("test")
    assert result == "ok"


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------


def test_proxy_creates_http_client(mock_azure_client):
    AzureOpenAILLM, _ = mock_azure_client

    with patch("inqtrix.providers_azure.DefaultHttpxClient") as mock_httpx_cls, \
            patch("inqtrix.providers_azure.OpenAI") as mock_openai_cls:
        mock_httpx_cls.return_value = MagicMock()
        mock_openai_cls.return_value = MagicMock()

        AzureOpenAILLM(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            proxy_url="http://proxy.corp.local:8080",
        )

    call_kwargs = mock_openai_cls.call_args.kwargs
    assert "http_client" in call_kwargs
    assert call_kwargs["http_client"] is not None


# ---------------------------------------------------------------------------
# AzureOpenAIAPIError
# ---------------------------------------------------------------------------


def test_azure_openai_api_error_message_format():
    err = AzureOpenAIAPIError(
        model="gpt-4o",
        status_code=404,
        error_code="DeploymentNotFound",
        message="deployment not found",
        request_id="req-abc",
    )
    assert "Azure-OpenAI-Aufruf fehlgeschlagen" in str(err)
    assert "gpt-4o" in str(err)
    assert "HTTP 404" in str(err)
    assert "DeploymentNotFound" in str(err)
    assert "request-id=req-abc" in str(err)
    assert "deployment not found" in str(err)


def test_azure_openai_api_error_minimal():
    err = AzureOpenAIAPIError(
        model="gpt-4o-mini",
        message="something went wrong",
    )
    assert "gpt-4o-mini" in str(err)
    assert "something went wrong" in str(err)
