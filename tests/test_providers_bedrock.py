"""Tests for the Bedrock LLM provider adapter."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from inqtrix.exceptions import AgentRateLimited, AgentTimeout, BedrockAPIError
from inqtrix.providers.base import SummarizeOptions


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

    with patch("inqtrix.providers.bedrock.boto3") as mock_boto3_mod, \
            patch("inqtrix.providers.bedrock.BotoConfig"):
        mock_boto3_mod.Session.return_value = mock_session
        from inqtrix.providers.bedrock import BedrockLLM
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
    assert resp.finish_reason == "end_turn"
    assert resp.raw is not None
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

    from inqtrix.providers.bedrock import _THINKING_MIN_MAX_TOKENS

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


def test_bedrock_effort_in_output_config_with_thinking(mock_boto3):
    """effort travels via additionalModelRequestFields.output_config alongside thinking."""
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM(thinking={"type": "adaptive"}, effort="medium")
    llm.complete("test")

    call_kwargs = mock_client.converse.call_args
    additional = call_kwargs.kwargs.get("additionalModelRequestFields", {})
    assert additional.get("thinking") == {"type": "adaptive"}
    assert additional.get("output_config") == {"effort": "medium"}


def test_bedrock_effort_works_without_thinking(mock_boto3):
    """effort can be used standalone (controls overall token spend)."""
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM(effort="low")
    llm.complete("test")

    call_kwargs = mock_client.converse.call_args
    additional = call_kwargs.kwargs.get("additionalModelRequestFields", {})
    assert additional == {"output_config": {"effort": "low"}}


def test_bedrock_effort_invalid_value_raises(mock_boto3):
    BedrockLLM, _ = mock_boto3
    with pytest.raises(ValueError, match="effort must be one of"):
        BedrockLLM(effort="ultra-mega")


def test_bedrock_effort_accepts_xhigh_and_max(mock_boto3):
    """xhigh + max are valid effort levels (Opus 4.7 / Mythos)."""
    BedrockLLM, _ = mock_boto3
    BedrockLLM(effort="xhigh")
    BedrockLLM(effort="max")


def test_bedrock_summarize_does_not_include_effort(mock_boto3):
    """summarize_parallel runs without thinking AND without effort overhead.

    Helper threads (summarize / claim extraction) should stay lean —
    they do not benefit from output_config.effort.
    """
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("kurz")

    llm = BedrockLLM(thinking={"type": "adaptive"}, effort="high")
    llm.summarize_parallel("Langer Text")

    call_kwargs = mock_client.converse.call_args
    params = call_kwargs.kwargs if call_kwargs.kwargs else {}
    # Neither thinking nor effort/output_config appear in summarize calls.
    assert "additionalModelRequestFields" not in params


def test_bedrock_effort_suppressed_inside_without_thinking(mock_boto3):
    """Regression: effort must be suppressed in helper paths (claim extraction).

    Same principle as the Anthropic-direct test: Bedrock helper models
    (e.g. Sonnet 4.5 for summarize) may reject effort. ``without_thinking``
    must therefore strip it from the payload.
    """
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM(thinking={"type": "adaptive"}, effort="medium")

    # Outside: both thinking and effort present.
    llm.complete("outside")
    additional = mock_client.converse.call_args.kwargs.get(
        "additionalModelRequestFields", {}
    )
    assert additional.get("thinking") == {"type": "adaptive"}
    assert additional.get("output_config") == {"effort": "medium"}

    # Inside without_thinking: BOTH must be absent.
    mock_client.converse.reset_mock()
    mock_client.converse.return_value = _converse_response("ok")
    with llm.without_thinking():
        llm.complete("inside")
    params = mock_client.converse.call_args.kwargs
    assert "additionalModelRequestFields" not in params


def test_bedrock_effort_skipped_for_haiku_per_call_model(mock_boto3):
    """Phase 12: Bedrock Haiku per-call model must omit output_config.effort.

    Same blacklist mechanic as in AnthropicLLM. With effort configured on
    the session, calls targeted at Haiku via the per-call ``model`` arg
    must not include output_config — otherwise Bedrock returns 400.
    """
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM(
        default_model="eu.anthropic.claude-opus-4-6-v1",
        summarize_model="eu.anthropic.claude-haiku-4-5",
        thinking={"type": "adaptive"},
        effort="medium",
    )

    # Reasoning call (Opus, default) → effort sent.
    llm.complete("question")
    params = mock_client.converse.call_args.kwargs
    additional = params.get("additionalModelRequestFields", {})
    assert additional.get("output_config") == {"effort": "medium"}

    # Helper call to Haiku via complete_with_metadata model override → no effort.
    mock_client.converse.reset_mock()
    mock_client.converse.return_value = _converse_response("ok")
    llm.complete_with_metadata("snippet", model="eu.anthropic.claude-haiku-4-5")
    params = mock_client.converse.call_args.kwargs
    additional = params.get("additionalModelRequestFields", {})
    assert "output_config" not in additional


def test_bedrock_effort_kept_for_sonnet_per_call_model(mock_boto3):
    """Phase 12: Bedrock Sonnet helper call DOES receive effort."""
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM(
        default_model="eu.anthropic.claude-opus-4-6-v1",
        summarize_model="eu.anthropic.claude-sonnet-4-6",
        thinking={"type": "adaptive"},
        effort="medium",
    )
    llm.complete_with_metadata("snippet", model="eu.anthropic.claude-sonnet-4-6")
    params = mock_client.converse.call_args.kwargs
    additional = params.get("additionalModelRequestFields", {})
    assert additional.get("output_config") == {"effort": "medium"}


def test_bedrock_effort_config_warnings_emitted_for_haiku_role(mock_boto3, caplog):
    """Configuring effort + a Haiku role on Bedrock yields a warning."""
    import logging

    BedrockLLM, _ = mock_boto3
    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        llm = BedrockLLM(
            default_model="eu.anthropic.claude-opus-4-6-v1",
            summarize_model="eu.anthropic.claude-haiku-4-5",
            thinking={"type": "adaptive"},
            effort="medium",
        )
    warnings = llm.consume_effort_config_warnings()
    assert len(warnings) == 1
    assert "summarize_model" in warnings[0]
    assert "haiku" in warnings[0].lower()
    # And the same line reached the log.
    assert any("haiku" in rec.getMessage().lower() for rec in caplog.records)
    # Consume drains the list.
    assert llm.consume_effort_config_warnings() == []


def test_bedrock_effort_no_warning_when_all_models_support_it(mock_boto3, caplog):
    """No warning when every configured role uses an effort-capable Bedrock model."""
    import logging

    BedrockLLM, _ = mock_boto3
    with caplog.at_level(logging.WARNING, logger="inqtrix"):
        llm = BedrockLLM(
            default_model="eu.anthropic.claude-opus-4-6-v1",
            summarize_model="eu.anthropic.claude-sonnet-4-6",
            thinking={"type": "adaptive"},
            effort="medium",
        )
    assert llm.consume_effort_config_warnings() == []


def test_temperature_in_params(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM(temperature=0.3)
    llm.complete("test")

    call_kwargs = mock_client.converse.call_args
    inference_config = call_kwargs.kwargs.get("inferenceConfig", {})
    assert inference_config["temperature"] == 0.3
    assert "additionalModelRequestFields" not in (call_kwargs.kwargs or {})


def test_complete_allows_per_call_output_budget_override(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("ok")

    llm = BedrockLLM(default_max_tokens=1024)
    llm.complete("test", max_output_tokens=77)

    call_kwargs = mock_client.converse.call_args
    inference_config = call_kwargs.kwargs.get("inferenceConfig", {})
    assert inference_config["maxTokens"] == 77


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
    with patch("inqtrix.providers.bedrock.time") as mock_time:
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
    with patch("inqtrix.providers.bedrock.time") as mock_time:
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
    with patch("inqtrix.providers.bedrock.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 1000
        mock_time.sleep = MagicMock()
        text, p_tok, c_tok = llm.summarize_parallel("Langer Testtext " * 100)

    assert text == ("Langer Testtext " * 100)[:800]
    assert p_tok == 0
    assert c_tok == 0
    notice = llm.consume_nonfatal_notice()
    assert notice is not None
    assert "Fallback" in notice


def test_summarize_parallel_uses_custom_options(mock_boto3):
    BedrockLLM, mock_client = mock_boto3
    mock_client.converse.return_value = _converse_response("Kurzfassung", 3, 2)

    llm = BedrockLLM()
    llm.summarize_parallel(
        "Langer Text",
        options=SummarizeOptions(
            prompt_template="DEEP:\n",
            input_char_limit=4,
            max_output_tokens=55,
        ),
    )

    call_kwargs = mock_client.converse.call_args
    assert call_kwargs.kwargs["messages"][0]["content"][0]["text"] == "DEEP:\nLang"
    assert call_kwargs.kwargs["inferenceConfig"]["maxTokens"] == 55


def test_summarize_reraises_rate_limited(mock_boto3):
    BedrockLLM, mock_client = mock_boto3

    throttle = _make_client_error("ThrottlingException", "rate exceeded", 429)
    mock_client.converse.side_effect = [throttle] * 5

    llm = BedrockLLM()
    with patch("inqtrix.providers.bedrock.time") as mock_time:
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
