"""Direct Amazon Bedrock adapter for the LLMProvider interface.

This module calls the Bedrock **Converse** API via ``boto3`` — no
LiteLLM proxy.  The Converse API is AWS's recommended model-agnostic
inference endpoint for Bedrock; it provides structured request/response
formats, native reasoning (thinking) support, and consistent error
handling across all supported models.

Authentication is handled through **AWS named profiles** (configured
in ``~/.aws/config`` / ``~/.aws/credentials``).  Pass the profile
name to the constructor or let boto3 resolve credentials from the
standard AWS credential chain (env vars, instance role, etc.).

Key design decisions mirror :mod:`inqtrix.providers.anthropic`:

* **Retry with jitter** — Bedrock returns ``ThrottlingException``
  under sustained load.  The provider wraps boto3 calls in its own
  retry loop with jittered exponential backoff.  boto3 built-in
  retries are **disabled** to avoid double-retry and to allow
  deadline-aware backoff.

* **Thinking isolation** — Extended thinking is valuable for complex
  reasoning calls but wastes tokens on short helper calls.  The
  same thread-local ``without_thinking`` context manager from
  :class:`AnthropicLLM` is replicated here.

* **Token-budget auto-raise** — Bedrock (Claude models) counts
  thinking tokens inside ``maxTokens``.  The provider auto-raises
  the budget to a safe minimum when thinking is enabled.

Requires ``boto3``::

    uv sync
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from inqtrix.constants import REASONING_TIMEOUT, SUMMARIZE_TIMEOUT
from inqtrix.exceptions import AgentRateLimited, AgentTimeout, BedrockAPIError
from inqtrix.prompts import SUMMARIZE_PROMPT
from inqtrix.providers.base import (
    LLMProvider,
    LLMResponse,
    ThinkingSuppressionMixin,
    _NonFatalNoticeMixin,
    _THINKING_MIN_MAX_TOKENS,
    _check_deadline,
    _retry_delay_seconds,
    _sleep_before_retry,
)
from inqtrix.settings import ModelSettings
from inqtrix.state import track_tokens

import boto3  # type: ignore[import-untyped]
from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
# type: ignore[import-untyped]
from botocore.exceptions import ConnectionError as BotoConnectionError

log = logging.getLogger("inqtrix")

# ---------------------------------------------------------------------------
# Retry & backoff constants
#
# Bedrock returns ThrottlingException under sustained load — especially
# when many parallel summarise/claim-extraction threads hit the API at
# once.  The following constants control the retry loop in
# _converse_with_retry.
#
# Shared backoff constants (_BACKOFF_BASE_SECONDS, _BACKOFF_MAX_SECONDS,
# _JITTER_RANGE, _THINKING_MIN_MAX_TOKENS) are imported from base.py.
# ---------------------------------------------------------------------------
_RETRYABLE_BEDROCK_ERRORS = frozenset({
    "ThrottlingException",
    "ModelTimeoutException",
    "InternalServerException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
})
_MAX_BEDROCK_ATTEMPTS = 5


class BedrockLLM(ThinkingSuppressionMixin, _NonFatalNoticeMixin, LLMProvider):
    """Call Amazon Bedrock Converse directly via boto3.

    Use this provider when Claude or other supported models should run on
    Amazon Bedrock rather than on direct Anthropic or LiteLLM-backed
    infrastructure. It is the right choice when AWS regions, named
    profiles, or Bedrock-specific access controls are operational
    requirements.

    Attributes:
        _default_model (str): Primary Bedrock model ID for reasoning
            calls.
        _summarize_model (str): Bedrock model ID used for summarize-model
            helper calls.
        _default_max_tokens (int): Output-token budget for reasoning
            requests before thinking-related auto-raise.
        _summarize_max_tokens (int): Output-token budget for helper
            summarization requests.
        _temperature (float | None): Optional sampling temperature.
        _thinking (dict[str, Any] | None): Extended-thinking config used
            on reasoning calls when not suppressed.
        _thread_state (threading.local): Thread-local storage for
            thinking suppression depth.
        _models (ModelSettings): Effective role-to-model mapping exposed
            to the runtime.
        _client (Any): boto3 Bedrock Runtime client configured without
            built-in retries.
    """

    def __init__(
        self,
        *,
        profile_name: str | None = None,
        region_name: str = "eu-central-1",
        default_model: str = "eu.anthropic.claude-sonnet-4-6",
        classify_model: str = "",
        summarize_model: str = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        evaluate_model: str = "",
        default_max_tokens: int = 1024,
        summarize_max_tokens: int = 512,
        temperature: float | None = None,
        thinking: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Use the constructor when Bedrock should handle reasoning calls via
        the Converse API. The provider disables boto3 retries and runs its
        own deadline-aware retry loop, so constructor-level choices mainly
        decide credentials, region, model roles, and token budgets.

        Args:
            profile_name: Optional AWS named profile from
                ``~/.aws/config`` or ``~/.aws/credentials``. When omitted,
                boto3 falls back to the default AWS credential chain.
            region_name: AWS region for the Bedrock endpoint. The default
                is ``"eu-central-1"``.
            default_model: Primary Bedrock model ID for classify, plan,
                evaluate fallback, and answer calls.
            classify_model: Optional cheaper override for classification.
                When omitted, classification falls back to
                ``default_model``.
            summarize_model: Bedrock model ID used for parallel
                summarization and claim extraction.
            evaluate_model: Optional override for evidence evaluation.
                When omitted, evaluation falls back to ``default_model``.
            default_max_tokens: Output-token budget for reasoning calls
                before any thinking-related auto-raise. The default is
                ``1024``.
            summarize_max_tokens: Output-token budget for summarize-model
                helper calls. The default is ``512``.
            temperature: Optional sampling temperature. Do not set this
                together with ``thinking`` because Bedrock Claude models
                reject that combination.
            thinking: Optional Bedrock thinking configuration forwarded via
                ``additionalModelRequestFields``.

        Raises:
            ValueError: If both ``temperature`` and ``thinking`` are set.

        Example:
            >>> from inqtrix import BedrockLLM
            >>> llm = BedrockLLM(
            ...     profile_name="default",
            ...     region_name="eu-central-1",
            ...     default_model="eu.anthropic.claude-sonnet-4-6",
            ... )
            >>> llm.models.reasoning_model
            'eu.anthropic.claude-sonnet-4-6'
        """
        if temperature is not None and thinking is not None:
            raise ValueError(
                "temperature and thinking are mutually exclusive — "
                "the Bedrock API (Claude models) rejects requests that set both."
            )
        self._default_model = default_model
        self._summarize_model = summarize_model
        self._default_max_tokens = default_max_tokens
        self._summarize_max_tokens = summarize_max_tokens
        self._temperature = temperature
        self._thinking = thinking
        self._thread_state = threading.local()
        self._models = ModelSettings(
            reasoning_model=default_model,
            search_model="",
            classify_model=classify_model,
            summarize_model=summarize_model,
            evaluate_model=evaluate_model,
        )

        # Disable boto3 built-in retries — we run our own retry loop
        # with deadline enforcement and jittered backoff.
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
        self._client = session.client(
            "bedrock-runtime",
            config=BotoConfig(
                retries={"max_attempts": 0},
                read_timeout=180,
            ),
        )

    @property
    def models(self) -> ModelSettings:
        """Return the effective role-to-model mapping for the runtime.

        Returns:
            ModelSettings: Resolved Bedrock model IDs used by graph nodes.
        """
        return self._models

    # ------------------------------------------------------------------ #
    # Thinking suppression — provided by ThinkingSuppressionMixin
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_error_details(exc: ClientError) -> dict[str, Any]:
        """Extract structured error information from a boto3 ClientError."""
        response = getattr(exc, "response", None) or {}
        error = response.get("Error", {})
        metadata = response.get("ResponseMetadata", {})
        return {
            "error_code": error.get("Code", ""),
            "message": error.get("Message", ""),
            "status_code": metadata.get("HTTPStatusCode"),
            "request_id": metadata.get("RequestId"),
        }

    @staticmethod
    def _build_api_error(
        *,
        model: str,
        details: dict[str, Any] | None = None,
        message: str = "",
        original: Exception,
    ) -> BedrockAPIError:
        details = details or {}
        sc = details.get("status_code")
        return BedrockAPIError(
            model=model,
            error_code=str(details.get("error_code") or "").strip(),
            status_code=sc if isinstance(sc, int) else None,
            message=message.strip() or str(details.get("message") or "").strip() or str(original),
            request_id=str(details.get("request_id") or "").strip() or None,
            original=original,
        )

    def _converse_with_retry(
        self,
        *,
        params: dict[str, Any],
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Call Bedrock Converse with retry and deadline enforcement.

        Implements jittered exponential backoff for transient errors
        (ThrottlingException, ServiceUnavailableException, etc.).
        The client-level ``read_timeout`` (set at creation) prevents
        individual calls from hanging; deadline enforcement handles
        the tighter agent time budget.
        """
        use_model = str(params.get("modelId") or self._default_model)

        for attempt in range(_MAX_BEDROCK_ATTEMPTS):
            if deadline is not None:
                _check_deadline(deadline)

            try:
                response = self._client.converse(**params)
                return response if isinstance(response, dict) else {}
            except ClientError as exc:
                details = self._extract_error_details(exc)
                error_code = details.get("error_code", "")

                if error_code == "ThrottlingException":
                    if attempt >= (_MAX_BEDROCK_ATTEMPTS - 1):
                        raise AgentRateLimited(use_model, exc) from exc
                    # Retryable — fall through to backoff below.

                api_error = self._build_api_error(
                    model=use_model, details=details, original=exc)

                if error_code in _RETRYABLE_BEDROCK_ERRORS and attempt < (_MAX_BEDROCK_ATTEMPTS - 1):
                    delay = _retry_delay_seconds(attempt)
                    log.warning(
                        "Bedrock transient error (%s, code=%s, request-id=%s, attempt=%d/%d). Retrying in %.2fs.",
                        use_model,
                        error_code,
                        details.get("request_id") or "-",
                        attempt + 1,
                        _MAX_BEDROCK_ATTEMPTS,
                        delay,
                    )
                    _sleep_before_retry(delay, deadline)
                    continue

                raise api_error from exc
            except BotoConnectionError as exc:
                api_error = self._build_api_error(model=use_model, original=exc)
                if attempt < (_MAX_BEDROCK_ATTEMPTS - 1):
                    delay = _retry_delay_seconds(attempt)
                    log.warning(
                        "Bedrock transport error (%s, attempt=%d/%d). Retrying in %.2fs: %s",
                        use_model,
                        attempt + 1,
                        _MAX_BEDROCK_ATTEMPTS,
                        delay,
                        exc,
                    )
                    _sleep_before_retry(delay, deadline)
                    continue
                raise api_error from exc

        raise self._build_api_error(  # pragma: no cover
            model=use_model,
            message="Bedrock request exhausted retries without a final response.",
            original=RuntimeError("retries exhausted"),
        )

    # ------------------------------------------------------------------ #
    # Response extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        """Extract visible text from a Bedrock Converse response.

        Collects only ``{"text": ...}`` content blocks — reasoning
        (thinking) content blocks are intentionally skipped.
        """
        parts: list[str] = []
        output = response.get("output", {})
        message = output.get("message", {}) if isinstance(output, dict) else {}
        content = message.get("content", []) if isinstance(message, dict) else []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_usage(response: dict[str, Any]) -> tuple[int, int]:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return (0, 0)
        return (
            int(usage.get("inputTokens") or 0),
            int(usage.get("outputTokens") or 0),
        )

    # ------------------------------------------------------------------ #
    # LLMProvider interface
    # ------------------------------------------------------------------ #

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        """Generate text through Bedrock and discard token metadata.

        Args:
            prompt: User-facing input text.
            system: Optional system instruction.
            model: Optional per-call Bedrock model override. When omitted,
                the provider uses ``self._default_model``.
            timeout: Per-call timeout budget in seconds.
            state: Optional mutable agent state for token tracking.
            deadline: Optional absolute monotonic deadline for the full
                run.

        Returns:
            str: Visible assistant text extracted from the Converse
            response.

        Raises:
            AgentTimeout: If the full run deadline has elapsed.
            AgentRateLimited: If Bedrock throttles the request fatally.
            BedrockAPIError: If a non-retryable Bedrock error occurs.
        """
        return self.complete_with_metadata(
            prompt,
            system=system,
            model=model,
            timeout=timeout,
            state=state,
            deadline=deadline,
        ).content

    def complete_with_metadata(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        """Generate text and token metadata through Bedrock Converse.

        Use this method for reasoning calls when the runtime wants both
        visible content and token usage. When thinking is enabled, the
        method raises ``maxTokens`` so hidden reasoning does not crowd out
        the visible answer. Request execution itself is delegated to
        ``_converse_with_retry()``, which owns deadline-aware retry logic
        for throttling and transient Bedrock failures.

        Args:
            prompt: User-facing input text.
            system: Optional system instruction.
            model: Optional per-call model override. When omitted, the
                provider uses ``self._default_model``.
            timeout: Per-call timeout budget in seconds. Bedrock uses the
                provider's retry loop instead of a dedicated per-request
                timeout field.
            state: Optional mutable agent state that receives token counts
                through ``track_tokens()`` when provided.
            deadline: Optional absolute monotonic deadline for the full
                run.

        Returns:
            LLMResponse: Structured response containing visible content,
            token counts, and the effective model label.

        Raises:
            AgentTimeout: If the full run deadline has already elapsed.
            AgentRateLimited: If Bedrock throttles the request after retry
                exhaustion.
            BedrockAPIError: If a non-retryable Bedrock or transport error
                occurs.
        """
        if deadline is not None:
            _check_deadline(deadline)

        use_model = model or self._default_model
        max_tokens = self._default_max_tokens
        use_thinking = self._thinking_enabled()
        if use_thinking:
            budget = self._thinking.get("budget_tokens")
            if isinstance(budget, int) and budget >= max_tokens:
                max_tokens = budget + 1024
            if max_tokens < _THINKING_MIN_MAX_TOKENS:
                log.debug(
                    "maxTokens auto-raised from %d to %d (thinking enabled)",
                    max_tokens,
                    _THINKING_MIN_MAX_TOKENS,
                )
                max_tokens = _THINKING_MIN_MAX_TOKENS

        params: dict[str, Any] = {
            "modelId": use_model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            params["system"] = [{"text": system}]
        if self._temperature is not None:
            params["inferenceConfig"]["temperature"] = self._temperature
        if use_thinking:
            params["additionalModelRequestFields"] = {"thinking": self._thinking}

        raw = self._converse_with_retry(
            params=params,
            deadline=deadline,
        )

        prompt_tokens, completion_tokens = self._extract_usage(raw)
        response = LLMResponse(
            content=self._extract_text(raw),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=use_model,
        )
        if state is not None:
            track_tokens(state, response)
        return response

    def summarize_parallel(
        self,
        text: str,
        deadline: float | None = None,
    ) -> tuple[str, int, int]:
        """Summarize search text in a thread-safe helper path.

        Helper summarization uses the configured summarize model and the
        same retry loop as reasoning calls. On non-rate-limit failure the
        method degrades locally to truncated raw text and stores a
        nonfatal notice for the search node.

        Args:
            text: Raw search-result text to condense. Blank input returns
                an empty summary payload immediately.
            deadline: Optional absolute monotonic deadline for the full
                run.

        Returns:
            tuple[str, int, int]: ``(facts_text, prompt_tokens,
            completion_tokens)``. On fallback, the text becomes the first
            800 characters of the raw input and both token counts are ``0``.

        Raises:
            AgentTimeout: If the global deadline has already elapsed.
            AgentRateLimited: If Bedrock throttles helper summarization
                fatally.
        """
        if not text.strip():
            return ("", 0, 0)
        self._clear_nonfatal_notice()
        if deadline is not None:
            _check_deadline(deadline)

        params: dict[str, Any] = {
            "modelId": self._summarize_model,
            "messages": [{"role": "user", "content": [{"text": f"{SUMMARIZE_PROMPT}{text[:6000]}"}]}],
            "inferenceConfig": {"maxTokens": self._summarize_max_tokens},
        }
        if self._temperature is not None:
            params["inferenceConfig"]["temperature"] = self._temperature

        try:
            raw = self._converse_with_retry(
                params=params,
                deadline=deadline,
            )
        except AgentRateLimited:
            raise
        except (BedrockAPIError, AgentTimeout) as exc:
            self._set_nonfatal_notice(
                f"Bedrock-Summarize fehlgeschlagen ({self._summarize_model}); Fallback auf Rohtext."
            )
            log.error("Bedrock-Summarize fehlgeschlagen (%s): %s", self._summarize_model, exc)
            return (text[:800], 0, 0)

        prompt_tokens, completion_tokens = self._extract_usage(raw)
        return (
            self._extract_text(raw),
            prompt_tokens,
            completion_tokens,
        )

    def is_available(self) -> bool:
        """Report whether the provider is configured to attempt requests.

        Returns:
            bool: ``True`` when the Bedrock Runtime client exists,
            otherwise ``False``.
        """
        return self._client is not None
