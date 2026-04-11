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
    """LLM completions via any LiteLLM- or OpenAI-compatible endpoint.

    This is the default LLM provider.  Instantiate it directly in your
    script, analogous to ``AnthropicLLM`` or any other provider::

        from inqtrix import LiteLLM

        llm = LiteLLM(
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url="http://localhost:4000/v1",
            default_model="gpt-4o",
            summarize_model="gpt-4o-mini",
        )

    Parameters
    ----------
    api_key:
        API key for the LiteLLM / OpenAI-compatible endpoint.
    base_url:
        Base URL of the endpoint (e.g. ``http://localhost:4000/v1``).
    default_model:
        Model used for reasoning, planning, and answer synthesis.
    classify_model:
        Model for question classification.  Falls back to *default_model*.
    summarize_model:
        Model for summarisation and claim extraction.  Falls back to
        *default_model*.
    evaluate_model:
        Model for evidence evaluation.  Falls back to *default_model*.
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
        """Expose model configuration for node access."""
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
        """Call an LLM via LiteLLM and return the response text."""
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
        """Call an LLM via LiteLLM. Handles timeout and error cases."""
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
        """Thread-safe fact extraction without state access.

        Returns (facts, prompt_tokens, completion_tokens).
        No *state* parameter so there are no race conditions when called
        from multiple threads.
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
        return self._client is not None


# Backwards-compatible alias used by config_bridge and internal code.
LiteLLMProvider = LiteLLM
