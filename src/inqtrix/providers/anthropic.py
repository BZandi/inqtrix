"""Direct Anthropic adapter for the LLMProvider interface.

This module calls the Anthropic Messages API via ``urllib`` — no SDK, no
LiteLLM proxy.  The direct approach gives full control over headers,
retry behaviour, and extended-thinking configuration, but it also means
we must handle transient failures, error parsing, and token-budget
adjustments ourselves.

Key design decisions:

* **Retry with jitter** — Anthropic returns HTTP 529 ("Overloaded")
  when capacity is tight.  The research agent fires many parallel
  requests (summarize + claim-extraction for every search result),
  which can trigger 529 bursts.  A simple fixed-interval retry makes
  things worse (thundering herd), so we use exponential backoff with
  random jitter so parallel threads spread their retries over time.

* **Thinking isolation** — Extended thinking (``thinking={"type":
  "adaptive"}``) is valuable for complex reasoning calls (classify,
  plan, evaluate, answer), but it wastes tokens on short helper
  calls like summarise and claim extraction.  A thread-local
  suppression mechanism (``without_thinking`` context manager) lets
  callers disable thinking for specific code paths without mutating
  shared config.

* **Token-budget auto-raise** — Anthropic counts thinking tokens
  *inside* ``max_tokens``.  With the default 1024 budget the model
  would spend most tokens thinking and truncate the visible answer.
  The provider auto-raises ``max_tokens`` to a safe minimum when
  thinking is enabled.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inqtrix.constants import REASONING_TIMEOUT, SUMMARIZE_TIMEOUT
from inqtrix.exceptions import AgentRateLimited, AgentTimeout, AnthropicAPIError
from inqtrix.prompts import SUMMARIZE_PROMPT
from inqtrix.providers.base import (
    LLMProvider,
    LLMResponse,
    ThinkingSuppressionMixin,
    _NonFatalNoticeMixin,
    _THINKING_MIN_MAX_TOKENS,
    _bounded_timeout,
    _check_deadline,
    _retry_delay_seconds,
    _sleep_before_retry,
)
from inqtrix.settings import ModelSettings
from inqtrix.state import track_tokens

log = logging.getLogger("inqtrix")

# ---------------------------------------------------------------------------
# Retry & backoff constants
#
# Anthropic returns 529 ("Overloaded") under sustained load — especially
# when many parallel summarise/claim-extraction threads hit the API at
# once.  The following constants control the retry loop in _request_json.
#
# Shared backoff constants (_BACKOFF_BASE_SECONDS, _BACKOFF_MAX_SECONDS,
# _JITTER_RANGE, _THINKING_MIN_MAX_TOKENS) are imported from base.py.
#
# _RETRYABLE_HTTP_STATUS: status codes that trigger a retry instead of
#     an immediate hard failure.  529 is Anthropic-specific; 500/502/503/
#     504 cover standard transient infrastructure errors.
#
# _MAX_ANTHROPIC_ATTEMPTS: total tries (initial + retries).  5 gives
#     enough room for a sustained 529 burst (~15–20 s total with backoff)
#     without wasting the global time budget.
# ---------------------------------------------------------------------------
_RETRYABLE_HTTP_STATUS = frozenset({500, 502, 503, 504, 529})
_MAX_ANTHROPIC_ATTEMPTS = 5


class AnthropicLLM(ThinkingSuppressionMixin, _NonFatalNoticeMixin, LLMProvider):
    """Call the Anthropic Messages API directly without LiteLLM.

    Parameters
    ----------
    api_key:
        Anthropic API key.
    base_url:
        Messages endpoint URL.
    anthropic_version:
        Value of the ``anthropic-version`` header.
    default_model:
        Model for reasoning calls (classify, plan, evaluate, answer).
    classify_model:
        Optional override for question classification. Falls back to
        *default_model*.
    summarize_model:
        Cheaper model for search-result summarisation.
    evaluate_model:
        Optional override for evidence evaluation. Falls back to
        *default_model*.
    default_max_tokens:
        Output-token budget for reasoning calls.  When *thinking* is
        set with a ``budget_tokens`` value that exceeds this limit,
        ``max_tokens`` is automatically raised to
        ``budget_tokens + 1024``.
    summarize_max_tokens:
        Output-token budget for summarisation calls.
    user_agent:
        ``User-Agent`` header value.
    temperature:
        Sampling temperature (0.0–1.0).  **Mutually exclusive with
        *thinking*** — the Anthropic API rejects requests that set
        both.
    thinking:
        Extended-thinking configuration dict forwarded verbatim to
        the API.  Recommended for Claude 4.6 models::

            thinking={"type": "adaptive"}

        Legacy / Claude 3.7 (manual budget)::

            thinking={"type": "enabled", "budget_tokens": 10000}

        Thinking is applied to reasoning calls only, not to
        summarize-model helper calls such as ``summarize_parallel``
        and claim extraction.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com/v1/messages",
        anthropic_version: str = "2023-06-01",
        default_model: str = "claude-sonnet-4-6",
        classify_model: str = "",
        summarize_model: str = "claude-haiku-4-5",
        evaluate_model: str = "",
        default_max_tokens: int = 1024,
        summarize_max_tokens: int = 512,
        user_agent: str = "inqtrix/0.1",
        temperature: float | None = None,
        thinking: dict[str, Any] | None = None,
    ) -> None:
        if temperature is not None and thinking is not None:
            raise ValueError(
                "temperature and thinking are mutually exclusive — "
                "the Anthropic API rejects requests that set both."
            )
        self._api_key = api_key
        self._base_url = base_url
        self._anthropic_version = anthropic_version
        self._default_model = default_model
        self._summarize_model = summarize_model
        self._default_max_tokens = default_max_tokens
        self._summarize_max_tokens = summarize_max_tokens
        self._user_agent = user_agent
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

    @property
    def models(self) -> ModelSettings:
        return self._models

    # ------------------------------------------------------------------ #
    # Thinking suppression — provided by ThinkingSuppressionMixin
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Retry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_retryable_http_status(status_code: int) -> bool:
        return status_code in _RETRYABLE_HTTP_STATUS

    @staticmethod
    def _extract_http_error_details(exc: HTTPError) -> dict[str, str | int | None]:
        """Parse structured error information from an Anthropic HTTP error.

        Anthropic returns errors as JSON with request-id in headers and
        an ``{"error": {"type": ..., "message": ...}}`` body.  This
        method extracts all available details so that ``AnthropicAPIError``
        can surface them for debugging — including the request-id which
        is essential for support tickets.
        """
        headers = getattr(exc, "headers", None) or getattr(exc, "hdrs", None)
        request_id = None
        retry_after = None
        if headers is not None:
            request_id = headers.get("request-id") or headers.get("anthropic-request-id")
            retry_after = headers.get("retry-after")

        raw_body = ""
        try:
            raw_bytes = exc.read()
        except Exception:
            raw_bytes = b""
        if raw_bytes:
            raw_body = raw_bytes.decode("utf-8", errors="replace")

        error_type = ""
        message = ""
        if raw_body:
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                message = raw_body.strip()
            else:
                if isinstance(payload, dict):
                    body_request_id = payload.get("request_id")
                    if isinstance(body_request_id, str) and body_request_id.strip():
                        request_id = request_id or body_request_id.strip()
                    error = payload.get("error")
                    if isinstance(error, dict):
                        raw_type = error.get("type")
                        raw_message = error.get("message")
                        if isinstance(raw_type, str):
                            error_type = raw_type.strip()
                        if isinstance(raw_message, str):
                            message = raw_message.strip()
                    if not message:
                        top_level_message = payload.get("message")
                        if isinstance(top_level_message, str):
                            message = top_level_message.strip()

        return {
            "status_code": exc.code,
            "request_id": request_id,
            "retry_after": retry_after,
            "error_type": error_type,
            "message": message,
        }

    @staticmethod
    def _build_api_error(
        *,
        model: str,
        details: dict[str, str | int | None] | None = None,
        message: str = "",
        original: Exception,
    ) -> AnthropicAPIError:
        details = details or {}
        sc = details.get("status_code")
        return AnthropicAPIError(
            model=model,
            status_code=sc if isinstance(sc, int) else None,
            error_type=str(details.get("error_type") or "").strip(),
            message=message.strip() or str(details.get("message") or "").strip() or str(original),
            request_id=str(details.get("request_id") or "").strip() or None,
            retry_after=str(details.get("retry_after") or "").strip() or None,
            original=original,
        )

    def _request_json(
        self,
        *,
        payload: dict[str, Any],
        timeout: float,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Send a POST to the Anthropic Messages API with retry.

        This is the central HTTP method shared by ``complete_with_metadata``
        and ``summarize_parallel``.  It implements:

        1. Deadline enforcement — abort early if the agent time budget
           has been exceeded.
        2. Retry with jittered backoff for transient server errors
           (500/502/503/504/529).  HTTP 429 (rate-limit) is *not*
           retried but escalated as ``AgentRateLimited`` so the graph
           can abort the entire run.
        3. Structured error extraction for non-retryable failures,
           surfacing request-id and error details in the exception.
        """
        use_model = str(payload.get("model") or self._default_model)

        for attempt in range(_MAX_ANTHROPIC_ATTEMPTS):
            if deadline is not None:
                _check_deadline(deadline)

            request = Request(
                self._base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": self._user_agent,
                    "x-api-key": self._api_key,
                    "anthropic-version": self._anthropic_version,
                },
                method="POST",
            )

            try:
                with urlopen(request, timeout=_bounded_timeout(timeout, deadline)) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
            except HTTPError as exc:
                details = self._extract_http_error_details(exc)
                if exc.code == 429:
                    raise AgentRateLimited(use_model, exc)

                api_error = self._build_api_error(
                    model=use_model, details=details, original=exc)
                if self._is_retryable_http_status(exc.code) and attempt < (_MAX_ANTHROPIC_ATTEMPTS - 1):
                    delay = _retry_delay_seconds(attempt, details.get("retry_after"))
                    log.warning(
                        "Anthropic transient HTTP error (%s, status=%s, type=%s, request-id=%s, attempt=%d/%d). Retrying in %.2fs.",
                        use_model,
                        exc.code,
                        details.get("error_type") or "unknown",
                        details.get("request_id") or "-",
                        attempt + 1,
                        _MAX_ANTHROPIC_ATTEMPTS,
                        delay,
                    )
                    _sleep_before_retry(delay, deadline)
                    continue
                raise api_error from exc
            except (URLError, OSError) as exc:
                api_error = self._build_api_error(model=use_model, original=exc)
                if attempt < (_MAX_ANTHROPIC_ATTEMPTS - 1):
                    delay = _retry_delay_seconds(attempt)
                    log.warning(
                        "Anthropic transport error (%s, attempt=%d/%d). Retrying in %.2fs: %s",
                        use_model,
                        attempt + 1,
                        _MAX_ANTHROPIC_ATTEMPTS,
                        delay,
                        exc,
                    )
                    _sleep_before_retry(delay, deadline)
                    continue
                raise api_error from exc
            except ValueError as exc:
                raise self._build_api_error(model=use_model, original=exc) from exc

        # Unreachable: every iteration ends with return, raise, or continue
        # (continue only when attempt < MAX - 1). Keep as defensive safeguard.
        raise self._build_api_error(  # pragma: no cover
            model=use_model,
            message="Anthropic request exhausted retries without a final response.",
            original=RuntimeError("retries exhausted"),
        )

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        """Extract visible text from an Anthropic response.

        The Messages API returns a ``content`` array with typed blocks.
        We only collect ``{"type": "text"}`` blocks — thinking blocks
        (``{"type": "thinking"}``) are intentionally skipped so they
        never leak into the user-visible output.
        """
        parts: list[str] = []
        for block in payload.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_usage(payload: dict[str, Any]) -> tuple[int, int]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return (0, 0)
        return (
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )

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
            # Anthropic counts thinking tokens *inside* max_tokens.
            # Two adjustments ensure the answer isn't truncated:
            #
            # 1. Explicit budget: if the caller set budget_tokens and
            #    it exceeds max_tokens, raise to budget + 1024 so there
            #    is room for the visible answer.
            # 2. Floor: even for adaptive thinking (no explicit budget),
            #    enforce _THINKING_MIN_MAX_TOKENS so the model has
            #    enough room for both reasoning and output.
            budget = self._thinking.get("budget_tokens")
            if isinstance(budget, int) and budget >= max_tokens:
                max_tokens = budget + 1024
            if max_tokens < _THINKING_MIN_MAX_TOKENS:
                log.debug(
                    "max_tokens auto-raised from %d to %d (thinking enabled)",
                    max_tokens,
                    _THINKING_MIN_MAX_TOKENS,
                )
                max_tokens = _THINKING_MIN_MAX_TOKENS
        payload: dict[str, Any] = {
            "model": use_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if use_thinking:
            payload["thinking"] = self._thinking

        raw = self._request_json(
            payload=payload,
            timeout=timeout,
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

        Called from multiple threads in the search node.  On failure
        (transient or hard), falls back to the first 800 chars of raw
        text and sets a nonfatal notice so the search node can report
        the degradation to the user.

        Thinking is NOT used here — callers (LLMClaimExtractor) wrap
        this in ``without_thinking()`` to avoid wasting tokens.
        """
        if not text.strip():
            return ("", 0, 0)
        self._clear_nonfatal_notice()
        if deadline is not None:
            _check_deadline(deadline)

        payload: dict[str, Any] = {
            "model": self._summarize_model,
            "max_tokens": self._summarize_max_tokens,
            "messages": [{"role": "user", "content": f"{SUMMARIZE_PROMPT}{text[:6000]}"}],
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature

        try:
            raw = self._request_json(
                payload=payload,
                timeout=SUMMARIZE_TIMEOUT,
                deadline=deadline,
            )
        except AgentRateLimited:
            raise
        except (AnthropicAPIError, AgentTimeout) as exc:
            self._set_nonfatal_notice(
                f"Anthropic-Summarize fehlgeschlagen ({self._summarize_model}); Fallback auf Rohtext."
            )
            log.error("Anthropic-Summarize fehlgeschlagen (%s): %s", self._summarize_model, exc)
            return (text[:800], 0, 0)

        prompt_tokens, completion_tokens = self._extract_usage(raw)
        return (
            self._extract_text(raw),
            prompt_tokens,
            completion_tokens,
        )

    def is_available(self) -> bool:
        return bool(self._api_key)
