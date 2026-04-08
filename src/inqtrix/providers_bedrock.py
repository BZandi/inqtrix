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

Key design decisions mirror :mod:`providers_anthropic`:

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

Requires ``boto3`` (optional dependency)::

    uv sync --extra bedrock
"""

from __future__ import annotations

import logging
import random
import threading
import time
from contextlib import contextmanager
from typing import Any

from inqtrix.constants import REASONING_TIMEOUT, SUMMARIZE_TIMEOUT
from inqtrix.exceptions import AgentRateLimited, AgentTimeout, BedrockAPIError
from inqtrix.prompts import SUMMARIZE_PROMPT
from inqtrix.providers import LLMProvider, LLMResponse, _NonFatalNoticeMixin, _check_deadline
from inqtrix.settings import ModelSettings
from inqtrix.state import track_tokens

try:
    import boto3  # type: ignore[import-untyped]
    from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]
    from botocore.exceptions import ClientError  # type: ignore[import-untyped]
    # type: ignore[import-untyped]
    from botocore.exceptions import ConnectionError as BotoConnectionError

    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False

log = logging.getLogger("inqtrix")

# ---------------------------------------------------------------------------
# Retry & backoff constants
#
# Bedrock returns ThrottlingException under sustained load — especially
# when many parallel summarise/claim-extraction threads hit the API at
# once.  The following constants control the retry loop in
# _converse_with_retry.
#
# Values are identical to providers_anthropic for consistency.
# See that module's docstring for the rationale behind each constant.
# ---------------------------------------------------------------------------
_RETRYABLE_BEDROCK_ERRORS = frozenset({
    "ThrottlingException",
    "ModelTimeoutException",
    "InternalServerException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
})
_MAX_BEDROCK_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 8.0
_JITTER_RANGE = (0.5, 1.5)
_THINKING_MIN_MAX_TOKENS = 16_384


class BedrockLLM(_NonFatalNoticeMixin, LLMProvider):
    """Call Amazon Bedrock Converse API directly via boto3.

    Parameters
    ----------
    profile_name:
        AWS named profile from ``~/.aws/config``.  ``None`` uses the
        default credential chain (env vars, instance role, etc.).
    region_name:
        AWS region for the Bedrock endpoint.
    default_model:
        Bedrock model ID for reasoning calls (classify, plan, evaluate,
        answer).
    classify_model:
        Optional override for question classification.  Falls back to
        *default_model*.
    summarize_model:
        Cheaper model for search-result summarisation.
    evaluate_model:
        Optional override for evidence evaluation.  Falls back to
        *default_model*.
    default_max_tokens:
        Output-token budget for reasoning calls.  When *thinking* is
        set with a ``budget_tokens`` value that exceeds this limit,
        ``maxTokens`` is automatically raised to
        ``budget_tokens + 1024``.
    summarize_max_tokens:
        Output-token budget for summarisation calls.
    temperature:
        Sampling temperature.  **Mutually exclusive with *thinking***
        — the Bedrock API (for Claude models) rejects requests that
        set both.
    thinking:
        Extended-thinking configuration dict forwarded via
        ``additionalModelRequestFields``.  Recommended::

            thinking={"type": "adaptive"}
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
        if not _HAS_BOTO3:
            raise ImportError(
                "boto3 is required for BedrockLLM.  "
                "Install it with: uv sync --extra bedrock"
            )
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
        return self._models

    # ------------------------------------------------------------------ #
    # Thinking suppression
    # ------------------------------------------------------------------ #

    @contextmanager
    def without_thinking(self):
        depth = int(getattr(self._thread_state, "suppress_thinking_depth", 0))
        self._thread_state.suppress_thinking_depth = depth + 1
        try:
            yield self
        finally:
            if depth:
                self._thread_state.suppress_thinking_depth = depth
            else:
                try:
                    delattr(self._thread_state, "suppress_thinking_depth")
                except AttributeError:
                    pass

    def _thinking_enabled(self) -> bool:
        return self._thinking is not None and int(
            getattr(self._thread_state, "suppress_thinking_depth", 0)
        ) == 0

    # ------------------------------------------------------------------ #
    # Retry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _retry_delay_seconds(attempt: int) -> float:
        base = min(_BACKOFF_BASE_SECONDS * (2 ** attempt), _BACKOFF_MAX_SECONDS)
        return base * random.uniform(*_JITTER_RANGE)

    @staticmethod
    def _sleep_before_retry(delay: float, deadline: float | None = None) -> None:
        if delay <= 0:
            return
        if deadline is not None:
            _check_deadline(deadline)
            delay = min(delay, max(0.0, deadline - time.monotonic()))
        if delay > 0:
            time.sleep(delay)

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
                    delay = self._retry_delay_seconds(attempt)
                    log.warning(
                        "Bedrock transient error (%s, code=%s, request-id=%s, attempt=%d/%d). Retrying in %.2fs.",
                        use_model,
                        error_code,
                        details.get("request_id") or "-",
                        attempt + 1,
                        _MAX_BEDROCK_ATTEMPTS,
                        delay,
                    )
                    self._sleep_before_retry(delay, deadline)
                    continue

                raise api_error from exc
            except BotoConnectionError as exc:
                api_error = self._build_api_error(model=use_model, original=exc)
                if attempt < (_MAX_BEDROCK_ATTEMPTS - 1):
                    delay = self._retry_delay_seconds(attempt)
                    log.warning(
                        "Bedrock transport error (%s, attempt=%d/%d). Retrying in %.2fs: %s",
                        use_model,
                        attempt + 1,
                        _MAX_BEDROCK_ATTEMPTS,
                        delay,
                        exc,
                    )
                    self._sleep_before_retry(delay, deadline)
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
        """Thread-safe fact extraction from a search result.

        Falls back to raw text on failure and sets a nonfatal notice.
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
        return True
