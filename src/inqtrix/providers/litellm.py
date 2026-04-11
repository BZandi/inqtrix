"""LiteLLM provider — LLM completions via any OpenAI-compatible endpoint.

This is the default LLM provider used when no explicit provider is
configured.  It wraps the OpenAI Python SDK and supports any endpoint
that speaks the OpenAI chat completions protocol (LiteLLM proxy,
OpenRouter, vLLM, Ollama, etc.).
"""

from __future__ import annotations

import logging
from openai import OpenAI, OpenAIError, RateLimitError, APIStatusError

from inqtrix.constants import REASONING_TIMEOUT, SUMMARIZE_TIMEOUT
from inqtrix.exceptions import AgentTimeout, AgentRateLimited
from inqtrix.prompts import SUMMARIZE_PROMPT
from inqtrix.providers.base import (
    LLMProvider,
    LLMResponse,
    _NonFatalNoticeMixin,
    _SDK_MAX_RETRIES,
    _check_deadline,
    _bounded_timeout,
    _normalize_completion_response,
)
from inqtrix.settings import ModelSettings
from inqtrix.state import track_tokens

log = logging.getLogger("inqtrix")


class LiteLLM(_NonFatalNoticeMixin, LLMProvider):
    """Route LLM calls through a LiteLLM or OpenAI-compatible endpoint.

    Use this provider when your reasoning models are exposed behind a
    LiteLLM proxy or any other endpoint that implements the OpenAI chat
    completions protocol. It is the default provider for the env-based
    auto-create path and is usually the simplest option when one gateway
    should front multiple upstream models.

    Attributes:
        _client (OpenAI): Shared SDK client used for reasoning and
            summarize-model requests.
        _models (ModelSettings): Effective model mapping for reasoning,
            classify, summarize, and evaluate roles.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "http://localhost:4000/v1",
        default_model: str = "gpt-4o",
        classify_model: str = "",
        summarize_model: str = "",
        evaluate_model: str = "",
    ) -> None:
        """Initialize the LiteLLM-backed provider.

        Use the constructor when your models are reachable through a
        single OpenAI-compatible base URL such as LiteLLM, OpenRouter,
        vLLM, or Ollama. The role-specific model arguments let you keep a
        strong default reasoning model while moving classification,
        summarization, or evaluation to cheaper deployments.

        Args:
            api_key: API key for the LiteLLM or OpenAI-compatible
                endpoint. This argument is required.
            base_url: Base URL of the endpoint. The default is
                ``"http://localhost:4000/v1"``. Use a different value
                when the proxy is hosted elsewhere.
            default_model: Primary model for reasoning, planning,
                evaluation fallback, and final answer synthesis. The
                default is ``"gpt-4o"``.
            classify_model: Optional cheaper override for question
                classification. When omitted, classification falls back
                to ``default_model``.
            summarize_model: Optional cheaper override for parallel
                summarization and claim extraction. When omitted, helper
                threads also use ``default_model``.
            evaluate_model: Optional override for evidence evaluation.
                When omitted, evaluation falls back to ``default_model``.

        Example:
            >>> from inqtrix import LiteLLM
            >>> llm = LiteLLM(
            ...     api_key="test-key",
            ...     base_url="http://localhost:4000/v1",
            ...     default_model="gpt-4o",
            ...     summarize_model="gpt-4o-mini",
            ... )
            >>> llm.models.reasoning_model
            'gpt-4o'
        """
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, max_retries=_SDK_MAX_RETRIES,
        )
        self._models = ModelSettings(
            reasoning_model=default_model,
            search_model="",
            classify_model=classify_model,
            summarize_model=summarize_model,
            evaluate_model=evaluate_model,
        )

    # -- public interface --------------------------------------------------

    @property
    def models(self) -> ModelSettings:
        """Return the effective role-to-model mapping for the runtime.

        Returns:
            ModelSettings: Resolved model names that graph nodes use when
            selecting classify, summarize, evaluate, or reasoning calls.
        """
        return self._models

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
        """Generate text via LiteLLM and discard token metadata.

        Use this convenience wrapper when the caller only needs visible
        response text. It delegates to ``complete_with_metadata()`` so the
        provider keeps one code path for error handling and token
        tracking.

        Args:
            prompt: User-facing input text.
            system: Optional system message. The default is ``None``.
            model: Optional per-call model override. When omitted, the
                provider uses its default reasoning model.
            timeout: Per-call timeout budget in seconds. The default is
                ``REASONING_TIMEOUT``.
            state: Optional mutable agent state for token tracking. Omit
                this in helper threads or when no token aggregation is
                needed.
            deadline: Optional absolute monotonic deadline for the full
                agent run.

        Returns:
            str: Visible assistant text for the completion.

        Raises:
            AgentTimeout: If the remaining agent budget is exhausted.
            AgentRateLimited: If the endpoint returns a fatal rate-limit
                error.
            OpenAIError: If the SDK surfaces a non-rate-limit backend
                failure.
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
        """Generate text and token metadata through the shared SDK client.

        Use this method for normal reasoning calls when the caller wants
        both the visible content and token accounting. The OpenAI SDK owns
        the retry loop here via ``max_retries=_SDK_MAX_RETRIES``; this
        method mainly assembles the request, clamps the timeout against the
        remaining deadline, and maps fatal rate limits into
        ``AgentRateLimited``.

        Args:
            prompt: User-facing input text.
            system: Optional system instruction. The default is ``None``.
            model: Optional per-call model override. If omitted, the
                provider uses ``self._models.reasoning_model``.
            timeout: Per-call timeout budget in seconds before deadline
                clamping. The default is ``REASONING_TIMEOUT``.
            state: Optional mutable agent state that receives token
                accounting through ``track_tokens()`` when provided.
            deadline: Optional absolute monotonic deadline for the full
                run. When present, the request timeout is reduced to the
                smaller of ``timeout`` and the remaining run budget.

        Returns:
            LLMResponse: Structured response containing visible content,
            token counts, and the effective model label.

        Raises:
            AgentTimeout: If the full run deadline has already elapsed.
            AgentRateLimited: If the backend returns HTTP 429 or the SDK
                raises ``RateLimitError``.
            APIStatusError: If the backend responds with a non-429 HTTP
                error.
            OpenAIError: If the SDK raises any other client or transport
                error.
        """
        if deadline is not None:
            _check_deadline(deadline)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        use_model = model or self._models.reasoning_model

        try:
            r = self._client.chat.completions.create(
                model=use_model,
                messages=messages,
                timeout=_bounded_timeout(timeout, deadline),
                stream=False,
            )
            normalized = _normalize_completion_response(r)
            if state is not None:
                track_tokens(state, normalized)
            return LLMResponse(
                content=normalized.content,
                prompt_tokens=normalized.prompt_tokens,
                completion_tokens=normalized.completion_tokens,
                model=use_model,
            )
        except RateLimitError as e:
            log.error("FATAL Rate-Limit (%s): %s", use_model, e)
            raise AgentRateLimited(use_model, e)
        except APIStatusError as e:
            if e.status_code == 429:
                log.error("FATAL Rate-Limit (%s): %s", use_model, e)
                raise AgentRateLimited(use_model, e)
            log.error("LLM-Aufruf fehlgeschlagen (%s): %s", use_model, e)
            raise
        except OpenAIError as e:
            log.error("LLM-Aufruf fehlgeschlagen (%s): %s", use_model, e)
            raise

    def summarize_parallel(
        self, text: str, deadline: float | None = None
    ) -> tuple[str, int, int]:
        """Summarize search text in a thread-safe helper path.

        The search node calls this method from worker threads, so it does
        not accept a shared ``state`` object. On provider failure it
        degrades locally to truncated raw text and stores a nonfatal notice
        that the search node can surface to the user.

        Args:
            text: Raw search-result text to condense. Blank input returns
                an empty tuple payload immediately.
            deadline: Optional absolute monotonic deadline for the full
                run. The request timeout is clamped to the remaining time.

        Returns:
            tuple[str, int, int]: ``(facts_text, prompt_tokens,
            completion_tokens)``. On fallback, the text becomes the first
            800 characters of the raw input and both token counts are ``0``.

        Raises:
            AgentTimeout: If the global run deadline has already elapsed
                before the request starts.
        """
        if not text.strip():
            return ("", 0, 0)
        self._clear_nonfatal_notice()
        if deadline is not None:
            _check_deadline(deadline)

        summarize_model = self._models.effective_summarize_model
        prompt = f"{SUMMARIZE_PROMPT}{text[:6000]}"

        try:
            r = self._client.chat.completions.create(
                model=summarize_model,
                messages=[{"role": "user", "content": prompt}],
                timeout=_bounded_timeout(SUMMARIZE_TIMEOUT, deadline),
                stream=False,
            )
            normalized = _normalize_completion_response(r)
            return (
                normalized.content,
                normalized.prompt_tokens,
                normalized.completion_tokens,
            )
        except (OpenAIError, AgentTimeout):
            self._set_nonfatal_notice(
                f"Zusammenfassung via {summarize_model} fehlgeschlagen; Fallback auf Rohtext."
            )
            return (text[:800], 0, 0)

    def is_available(self) -> bool:
        """Report whether the provider is configured to attempt requests.

        Returns:
            bool: ``True`` when the shared SDK client exists, otherwise
            ``False``.
        """
        return self._client is not None


# Backwards-compatible alias used by config_bridge and internal code.
LiteLLMProvider = LiteLLM
