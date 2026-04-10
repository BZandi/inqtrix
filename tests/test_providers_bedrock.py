"""Tests for the Bedrock LLM provider adapter."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from inqtrix.exceptions import AgentRateLimited, AgentTimeout, BedrockAPIError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client_error(code: str, message: str = "error", status_code: int = 400, request_id: str = "req-123"):
    """Build a botocore ClientError with realistic response structure."""
    from botocore.exceptions import ClientError

    return ClientError(
        error_response={
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {
                "HTTPStatusCode": status_code,
                "RequestId": request_id,
            },
        },
        operation_name="Converse",
    )


def _converse_response(text: str = "Hallo Welt", input_tokens: int = 12, output_tokens: int = 7):
    """Build a realistic Bedrock Converse response dict."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
        "stopReason": "end_turn",
    }


def _converse_response_with_thinking(text: str = "ok", thinking_text: str = "let me think..."):
    """Converse response containing both reasoning and text blocks."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": thinking_text, "signature": "sig"}}},
                    {"text": text},
                ],
            }
        },
        "usage": {"inputTokens": 20, "outputTokens": 15},
        "stopReason": "end_turn",
    }


@pytest.fixture()
def mock_boto3():
    """Patch boto3 so BedrockLLM can be imported and instantiated without real AWS credentials."""
    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client

    with patch("inqtrix.providers_bedrock.boto3") as mock_boto3_mod, \
            patch("inqtrix.providers_bedrock.BotoConfig"):
        mock_boto3_mod.Session.return_value = mock_session
        from inqtrix.providers_bedrock import BedrockLLM
        yield BedrockLLM, mock_client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_complete_returns_text(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("Hallo Welt")

    llm = BedrockLLM(default_model="eu.anthropic.claude-sonnet-4-6")
    result = llm.complete("Frage")

    assert result == "Hallo Welt"
    mock_client.converse.assert_called_once()


def test_complete_with_metadata_returns_tokens(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("Antwort", 10, 5)

    llm = BedrockLLM(default_model="eu.anthropic.claude-sonnet-4-6")
    state = {"total_prompt_tokens": 0, "total_completion_tokens": 0}
    resp = llm.complete_with_metadata("Frage", state=state)

    assert resp.content == "Antwort"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 5
    assert state["total_prompt_tokens"] == 10
    assert state["total_completion_tokens"] == 5


def test_complete_with_system_prompt(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM()
    llm.complete("Frage", system="Du bist ein Forscher.")

    call_kwargs = mock_client.converse.call_args
    params = call_kwargs.kwargs if call_kwargs.kwargs else {}
    assert "system" in params
    assert params["system"] == [{"text": "Du bist ein Forscher."}]


def test_summarize_parallel_happy_path(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("Kurzfassung", 3, 2)

    llm = BedrockLLM()
    summary, p_tok, c_tok = llm.summarize_parallel("Langer Text")

    assert summary == "Kurzfassung"
    assert p_tok == 3
    assert c_tok == 2


def test_summarize_parallel_empty_text(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    llm = BedrockLLM()

    result = llm.summarize_parallel("   ")
    assert result == ("", 0, 0)
    mock_client.converse.assert_not_called()


def test_is_available_returns_true(mock_boto3):
    BedrockLLM, _ = mock_boto3
    llm = BedrockLLM()
    assert llm.is_available() is True


# ---------------------------------------------------------------------------
# Thinking
# ---------------------------------------------------------------------------


def test_thinking_in_params(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response_with_thinking("ok")

    llm = BedrockLLM(thinking={"type": "adaptive"})
    resp = llm.complete_with_metadata("test")

    call_kwargs = mock_client.converse.call_args
    # additionalModelRequestFields should contain thinking
    assert "additionalModelRequestFields" in (call_kwargs.kwargs or {})
    assert call_kwargs.kwargs["additionalModelRequestFields"]["thinking"] == {
        "type": "adaptive"}
    # _extract_text must skip reasoning blocks
    assert resp.content == "ok"


def test_thinking_suppressed_in_summarize(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("kurz")

    llm = BedrockLLM(thinking={"type": "adaptive"})
    llm.summarize_parallel("Langer Text")

    call_kwargs = mock_client.converse.call_args
    params = call_kwargs.kwargs if call_kwargs.kwargs else {}
    # summarize_parallel builds its own params without thinking —
    # additionalModelRequestFields must not appear.
    assert "additionalModelRequestFields" not in params


def test_without_thinking_context_manager(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM(thinking={"type": "adaptive"})

    # Outside context manager: thinking enabled
    assert llm._thinking_enabled() is True

    with llm.without_thinking():
        assert llm._thinking_enabled() is False
        llm.complete("test")
        call_kwargs = mock_client.converse.call_args
        # Should NOT have additionalModelRequestFields
        assert "additionalModelRequestFields" not in (call_kwargs.kwargs or {})

    # After context manager: thinking enabled again
    assert llm._thinking_enabled() is True


def test_thinking_auto_raises_max_tokens(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    from inqtrix.providers_bedrock import _THINKING_MIN_MAX_TOKENS

    llm = BedrockLLM(
        default_max_tokens=1024,
        thinking={"type": "enabled", "budget_tokens": 8000},
    )
    llm.complete("test")

    call_kwargs = mock_client.converse.call_args
    inference_config = call_kwargs.kwargs.get("inferenceConfig", {})
    # budget_tokens (8000) >= default_max_tokens (1024) → raised to 9024,
    # then clamped up to _THINKING_MIN_MAX_TOKENS (16384) by the floor check.
    assert inference_config["maxTokens"] == _THINKING_MIN_MAX_TOKENS


def test_temperature_and_thinking_raises(mock_boto3):
    BedrockLLM, _ = mock_boto3
    with pytest.raises(ValueError, match="mutually exclusive"):
        BedrockLLM(temperature=0.5, thinking={"type": "adaptive"})


def test_temperature_in_params(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM(temperature=0.3)
    llm.complete("test")

    call_kwargs = mock_client.converse.call_args
    inference_config = call_kwargs.kwargs.get("inferenceConfig", {})
    assert inference_config["temperature"] == 0.3
    assert "additionalModelRequestFields" not in (call_kwargs.kwargs or {})


# ---------------------------------------------------------------------------
# Retry & error handling
# ---------------------------------------------------------------------------


def test_retries_transient_error_then_succeeds(mock_boto3):
    BedrockLLM, mock_client = mock_boto3

    from botocore.exceptions import ClientError

    transient = _make_client_error("ServiceUnavailableException", "service busy", 503)
    mock_client.converse.side_effect = [transient, transient, _converse_response("ok")]

    llm = BedrockLLM()
    # Patch sleep to avoid real delays
    with patch("inqtrix.providers_bedrock.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 1000
        mock_time.sleep = MagicMock()
        result = llm.complete("test")

    assert result == "ok"
    assert mock_client.converse.call_count == 3


def test_throttling_retries_then_raises_rate_limited(mock_boto3):
    BedrockLLM, mock_client = mock_boto3

    throttle = _make_client_error("ThrottlingException", "rate exceeded", 429)
    mock_client.converse.side_effect = [throttle] * 5

    llm = BedrockLLM()
    with patch("inqtrix.providers_bedrock.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 1000
        mock_time.sleep = MagicMock()
        with pytest.raises(AgentRateLimited):
            llm.complete("test")

    assert mock_client.converse.call_count == 5


def test_non_retryable_error_raises_bedrock_api_error(mock_boto3):
    BedrockLLM, mock_client = mock_boto3

    validation_error = _make_client_error(
        "ValidationException", "invalid input", 400, "req-val-456")
    mock_client.converse.side_effect = validation_error

    llm = BedrockLLM()
    with pytest.raises(BedrockAPIError) as exc_info:
        llm.complete("test")

    error = exc_info.value
    assert error.error_code == "ValidationException"
    assert error.status_code == 400
    assert error.request_id == "req-val-456"
    assert "invalid input" in str(error)


def test_summarize_falls_back_on_error(mock_boto3):
    BedrockLLM, mock_client = mock_boto3

    error = _make_client_error("ServiceUnavailableException", "busy", 503)
    mock_client.converse.side_effect = error

    llm = BedrockLLM()
    with patch("inqtrix.providers_bedrock.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 1000
        mock_time.sleep = MagicMock()
        text, p_tok, c_tok = llm.summarize_parallel("Langer Testtext " * 100)

    assert text == ("Langer Testtext " * 100)[:800]
    assert p_tok == 0
    assert c_tok == 0
    notice = llm.consume_nonfatal_notice()
    assert notice is not None
    assert "Fallback" in notice


def test_summarize_reraises_rate_limited(mock_boto3):
    BedrockLLM, mock_client = mock_boto3

    throttle = _make_client_error("ThrottlingException", "rate exceeded", 429)
    mock_client.converse.side_effect = [throttle] * 5

    llm = BedrockLLM()
    with patch("inqtrix.providers_bedrock.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 1000
        mock_time.sleep = MagicMock()
        with pytest.raises(AgentRateLimited):
            llm.summarize_parallel("Langer Text")


# ---------------------------------------------------------------------------
# Deadline enforcement
# ---------------------------------------------------------------------------


def test_deadline_raises_agent_timeout(mock_boto3):
    BedrockLLM, mock_client = mock_boto3

    llm = BedrockLLM()
    # Set a deadline that's already in the past
    past_deadline = time.monotonic() - 10

    with pytest.raises(AgentTimeout):
        llm.complete("test", deadline=past_deadline)


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------


def test_models_property(mock_boto3):
    BedrockLLM, _ = mock_boto3
    llm = BedrockLLM(
        default_model="eu.anthropic.claude-opus-4-6-v1",
        classify_model="eu.anthropic.claude-sonnet-4-6",
        summarize_model="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        evaluate_model="eu.anthropic.claude-sonnet-4-6",
    )
    assert llm.models.reasoning_model == "eu.anthropic.claude-opus-4-6-v1"
    assert llm.models.classify_model == "eu.anthropic.claude-sonnet-4-6"
    assert llm.models.summarize_model == "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert llm.models.evaluate_model == "eu.anthropic.claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# BedrockAPIError
# ---------------------------------------------------------------------------


def test_bedrock_api_error_message_format():
    err = BedrockAPIError(
        model="eu.anthropic.claude-opus-4-6-v1",
        error_code="ValidationException",
        status_code=400,
        message="invalid model id",
        request_id="req-abc",
    )
    assert "Bedrock-Aufruf fehlgeschlagen" in str(err)
    assert "eu.anthropic.claude-opus-4-6-v1" in str(err)
    assert "ValidationException" in str(err)
    assert "HTTP 400" in str(err)
    assert "request-id=req-abc" in str(err)
    assert "invalid model id" in str(err)
