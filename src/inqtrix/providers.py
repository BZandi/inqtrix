"""LLM and search provider abstractions.

Defines abstract base classes (LLMProvider, SearchProvider), concrete
implementations (LiteLLM, PerplexitySearch), response dataclasses,
and a factory function to wire everything together.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from cachetools import TTLCache
from openai import OpenAI, OpenAIError, RateLimitError, APIStatusError

from inqtrix.constants import REASONING_TIMEOUT, SEARCH_TIMEOUT, SUMMARIZE_TIMEOUT
from inqtrix.exceptions import AgentTimeout, AgentRateLimited
from inqtrix.prompts import SUMMARIZE_PROMPT
from inqtrix.settings import Settings, ModelSettings, AgentSettings
from inqtrix.state import track_tokens
from inqtrix.urls import extract_urls

log = logging.getLogger("inqtrix")

# =========================================================
# Internal deadline helpers
# =========================================================


def _check_deadline(deadline: float) -> None:
    """Raise AgentTimeout when the deadline has passed."""
    if time.monotonic() > deadline:
        raise AgentTimeout(
            "Agent-Zeitlimit ueberschritten. "
            "Antwort wird mit bisherigem Kontext generiert."
        )


def _bounded_timeout(
    default_timeout: int | float, deadline: float | None = None
) -> float:
    """Clamp API timeout to the remaining agent time budget."""
    if deadline is None:
        return float(default_timeout)
    _check_deadline(deadline)
    remaining = deadline - time.monotonic()
    return max(1.0, min(float(default_timeout), remaining))


# =========================================================
# Response dataclasses
# =========================================================


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured wrapper around an LLM completion result."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Single search result entry."""

    url: str
    title: str = ""
    snippet: str = ""


@dataclass(frozen=True, slots=True)
class SummarizeResult:
    """Result of a parallel summarization call."""

    facts: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True, slots=True)
class _NormalizedCompletion:
    """Internal normalized representation of a chat completion."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] | None = None


def _extract_content_value(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _extract_choice_content(choice: Any) -> str:
    for field_name in ("message", "delta"):
        if isinstance(choice, dict):
            payload = choice.get(field_name)
        else:
            payload = getattr(choice, field_name, None)
        if payload is None:
            continue
        if isinstance(payload, dict):
            content = payload.get("content")
        else:
            content = getattr(payload, "content", None)
        text = _extract_content_value(content)
        if text:
            return text
    return ""


def _extract_usage_from_payload(payload: dict[str, Any]) -> tuple[int, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return (0, 0)
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def _extract_usage_from_response(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return (0, 0)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _normalize_completion_payload(payload: dict[str, Any]) -> _NormalizedCompletion:
    content_parts: list[str] = []
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            text = _extract_choice_content(choice)
            if text:
                content_parts.append(text)
    prompt_tokens, completion_tokens = _extract_usage_from_payload(payload)
    return _NormalizedCompletion(
        content="".join(content_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        raw=payload,
    )


def _normalize_text_completion(response: str) -> _NormalizedCompletion:
    response_text = response.strip()
    if not response_text:
        return _NormalizedCompletion(content="", raw={})

    if not response_text.startswith("data:"):
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return _NormalizedCompletion(content=response_text, raw={})
        if isinstance(payload, dict):
            return _normalize_completion_payload(payload)
        return _NormalizedCompletion(content=response_text, raw={})

    content_parts: list[str] = []
    citations: list[str] = []
    related_questions: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    model = ""

    for line in response.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        normalized = _normalize_completion_payload(payload)
        if normalized.content:
            content_parts.append(normalized.content)
        if normalized.prompt_tokens or normalized.completion_tokens:
            prompt_tokens = normalized.prompt_tokens
            completion_tokens = normalized.completion_tokens

        payload_citations = payload.get("citations")
        if isinstance(payload_citations, list):
            for citation in payload_citations:
                text = str(citation)
                if text and text not in citations:
                    citations.append(text)

        payload_related = payload.get("related_questions")
        if isinstance(payload_related, list):
            for question in payload_related:
                text = str(question)
                if text and text not in related_questions:
                    related_questions.append(text)

        payload_model = payload.get("model")
        if isinstance(payload_model, str) and payload_model:
            model = payload_model

    raw: dict[str, Any] = {}
    if citations:
        raw["citations"] = citations
    if related_questions:
        raw["related_questions"] = related_questions
    if model:
        raw["model"] = model
    if prompt_tokens or completion_tokens:
        raw["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    return _NormalizedCompletion(
        content="".join(content_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        raw=raw,
    )


def _normalize_completion_response(response: Any) -> _NormalizedCompletion:
    if isinstance(response, str):
        return _normalize_text_completion(response)

    content_parts: list[str] = []
    choices = getattr(response, "choices", None)
    if choices:
        for choice in choices:
            text = _extract_choice_content(choice)
            if text:
                content_parts.append(text)

    prompt_tokens, completion_tokens = _extract_usage_from_response(response)
    raw: dict[str, Any] = {}
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except TypeError:
            dumped = None
        if isinstance(dumped, dict):
            raw = dumped
            if not content_parts:
                normalized = _normalize_completion_payload(dumped)
                if normalized.content:
                    content_parts.append(normalized.content)
            if not prompt_tokens and not completion_tokens:
                prompt_tokens, completion_tokens = _extract_usage_from_payload(dumped)

    return _NormalizedCompletion(
        content="".join(content_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        raw=raw,
    )


# =========================================================
# Abstract base classes
# =========================================================


class LLMProvider(ABC):
    """Abstract interface for LLM completions and summarization."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        """Send a prompt and return the completion text.

        Parameters
        ----------
        prompt:
            The user message content.
        system:
            Optional system message.
        model:
            Override the default reasoning model for this call.
        timeout:
            Per-call timeout in seconds (may be shortened by deadline).
        state:
            Optional agent state dict for token tracking.
        deadline:
            Absolute monotonic deadline; triggers AgentTimeout if exceeded.

        Returns
        -------
        str
            The assistant message content (empty string on empty response).
        """

    def complete_with_metadata(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        """Return completion text plus optional token metadata.

        Custom providers only need to implement :meth:`complete`.
        Providers that can surface token counts may override this method.
        """
        return LLMResponse(
            content=self.complete(
                prompt,
                system=system,
                model=model,
                timeout=timeout,
                state=state,
                deadline=deadline,
            ),
            model=model or "",
        )

    @abstractmethod
    def summarize_parallel(
        self, text: str, deadline: float | None = None
    ) -> tuple[str, int, int]:
        """Thread-safe fact extraction without state access.

        Returns
        -------
        tuple[str, int, int]
            (facts, prompt_tokens, completion_tokens)
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the provider is ready to serve requests."""


class SearchProvider(ABC):
    """Abstract interface for web search."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        search_context_size: str = "high",
        recency_filter: str | None = None,
        language_filter: list[str] | None = None,
        domain_filter: list[str] | None = None,
        search_mode: str | None = None,
        return_related: bool = False,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Execute a web search and return structured results.

        Returns
        -------
        dict[str, Any]
            Keys: answer, citations, related_questions,
                  _prompt_tokens, _completion_tokens
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the provider is ready to serve requests."""


class ConfiguredLLMProvider(LLMProvider):
    """Adapter that attaches model metadata to a custom LLM provider.

    The graph nodes select classify/evaluate/fallback models through
    ``providers.llm.models``. Wrapping custom providers here preserves the
    lightweight Baukasten contract while still giving the runtime access to
    the effective model names from configuration.
    """

    def __init__(self, provider: LLMProvider, models: ModelSettings) -> None:
        self._provider = provider
        self._models = models

    @property
    def models(self) -> ModelSettings:
        return self._models

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        return self._provider.complete(
            prompt,
            system=system,
            model=model or self._models.reasoning_model,
            timeout=timeout,
            state=state,
            deadline=deadline,
        )

    def complete_with_metadata(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        return self._provider.complete_with_metadata(
            prompt,
            system=system,
            model=model or self._models.reasoning_model,
            timeout=timeout,
            state=state,
            deadline=deadline,
        )

    def summarize_parallel(
        self, text: str, deadline: float | None = None
    ) -> tuple[str, int, int]:
        return self._provider.summarize_parallel(text, deadline=deadline)

    def is_available(self) -> bool:
        return self._provider.is_available()


# =========================================================
# LiteLLM — user-facing LLM provider for OpenAI-compatible endpoints
# =========================================================


class LiteLLM(LLMProvider):
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
        self._client = OpenAI(base_url=base_url, api_key=api_key)
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
            log.error("Claude-Aufruf fehlgeschlagen (%s): %s", use_model, e)
            raise
        except OpenAIError as e:
            log.error("Claude-Aufruf fehlgeschlagen (%s): %s", use_model, e)
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
        except AgentRateLimited:
            raise
        except (OpenAIError, AgentTimeout):
            return (text[:800], 0, 0)

    def is_available(self) -> bool:
        return self._client is not None


# Backwards-compatible alias used by config_bridge and internal code.
LiteLLMProvider = LiteLLM


# =========================================================
# PerplexitySearch — user-facing search provider
# =========================================================


class PerplexitySearch(SearchProvider):
    """Web search via Perplexity Sonar — works with both LiteLLM proxy and the direct Perplexity API.

    .. note::

       This class targets the Perplexity **Sonar API** (chat completions
       endpoint).  Other Perplexity products — such as Deep Research or
       the internal Agent API — use different parameters and endpoints
       and are **not** supported by this provider.

    Via LiteLLM proxy::

        search = PerplexitySearch(
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url="http://localhost:4000/v1",
            model="perplexity-sonar-pro-agent",
        )

    Direct Perplexity API::

        search = PerplexitySearch(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
            model="sonar-pro",
        )

    Parameters
    ----------
    api_key:
        API key for the endpoint.
    base_url:
        Base URL.  Use ``http://localhost:4000/v1`` for a LiteLLM proxy,
        or ``https://api.perplexity.ai`` for the direct Perplexity Sonar API
        (no ``/v1`` suffix — the OpenAI SDK appends ``/chat/completions``).
    model:
        Search model identifier.  LiteLLM example:
        ``perplexity-sonar-pro-agent``.  Direct Perplexity: ``sonar-pro``.
    cache_maxsize:
        Maximum number of cached search results.
    cache_ttl:
        Cache time-to-live in seconds.
    request_params:
        Extra request parameters forwarded to the chat completions call.
    direct_mode:
        Controls how search parameters are formatted in ``extra_body``.
        ``True``: flat top-level keys (direct Perplexity API).
        ``False``: nested inside ``web_search_options`` (LiteLLM proxy).
        ``None`` (default): auto-detect from *base_url* — URLs containing
        ``perplexity.ai`` use direct mode.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "http://localhost:4000/v1",
        model: str = "perplexity-sonar-pro-agent",
        cache_maxsize: int = 256,
        cache_ttl: int = 3600,
        request_params: dict[str, Any] | None = None,
        direct_mode: bool | None = None,
        # Internal: accept a pre-built client (used by create_providers / config_bridge).
        _client: OpenAI | None = None,
    ) -> None:
        self._client = _client or OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._cache: TTLCache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self._cache_lock = threading.Lock()
        self._request_params = dict(request_params or {})
        self._direct_mode = (
            direct_mode if direct_mode is not None
            else "perplexity.ai" in base_url
        )

    # -- internal helpers --------------------------------------------------

    def _build_extra_body(
        self,
        *,
        search_context_size: str,
        search_mode: str | None,
        recency_filter: str | None,
        language_filter: list[str] | None,
        domain_filter: list[str] | None,
        return_related: bool,
    ) -> dict[str, Any]:
        """Build ``extra_body`` for the chat completions call.

        Direct Perplexity API: flat top-level keys.
        LiteLLM proxy: all options nested inside ``web_search_options``.
        """
        if self._direct_mode:
            body: dict[str, Any] = {
                "search_context_size": search_context_size,
            }
            if search_mode:
                body["search_mode"] = search_mode
            if recency_filter:
                body["search_recency_filter"] = recency_filter
            if language_filter:
                body["search_language_filter"] = language_filter
            if domain_filter:
                body["search_domain_filter"] = domain_filter
            if return_related:
                body["return_related_questions"] = True
            return body

        # LiteLLM proxy path: nest inside web_search_options.
        # IMPORTANT: web_search_options must be COMPLETE because LiteLLM
        # REPLACES (not deep-merges) the config defaults.
        web_opts: dict[str, Any] = {
            "search_context_size": search_context_size,
            "search_mode": search_mode or "web",
            "num_search_results": 20,
        }
        if recency_filter:
            web_opts["search_recency_filter"] = recency_filter
        if language_filter:
            web_opts["search_language_filter"] = language_filter
        if domain_filter:
            web_opts["search_domain_filter"] = domain_filter
        if return_related:
            web_opts["return_related_questions"] = True
        return {"web_search_options": web_opts}

    # -- public interface --------------------------------------------------

    def search(
        self,
        query: str,
        *,
        search_context_size: str = "high",
        recency_filter: str | None = None,
        language_filter: list[str] | None = None,
        domain_filter: list[str] | None = None,
        search_mode: str | None = None,
        return_related: bool = False,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Execute a Perplexity Sonar search via LiteLLM.

        Uses Perplexity-specific parameters for higher quality:
        - search_context_size: "low"/"medium"/"high" (web search depth)
        - recency_filter: "day"/"week"/"month"/"year"
        - language_filter: ISO 639-1 codes e.g. ["en","de"]
        - domain_filter: domains to include/exclude (max 20)
        - search_mode: "academic" for scholarly sources
        - return_related: include related questions in response
        """
        # Cache key includes all search parameters
        cache_parts = [
            query,
            search_context_size,
            str(recency_filter),
            str(language_filter),
            str(domain_filter),
            str(search_mode),
        ]
        key = hashlib.sha256("|".join(cache_parts).encode()).hexdigest()

        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)

        # Build Perplexity-specific extra_body.
        # Format depends on whether we talk to the direct Perplexity API
        # or go through a LiteLLM proxy.
        extra = self._build_extra_body(
            search_context_size=search_context_size,
            search_mode=search_mode,
            recency_filter=recency_filter,
            language_filter=language_filter,
            domain_filter=domain_filter,
            return_related=return_related,
        )

        _empty: dict[str, Any] = {
            "answer": "",
            "citations": [],
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

        try:
            request_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [{"role": "user", "content": query}],
                "timeout": _bounded_timeout(SEARCH_TIMEOUT, deadline),
                "stream": False,
                "extra_body": extra,
            }
            for param_key, param_val in self._request_params.items():
                if param_key in {"model", "messages", "timeout", "stream"}:
                    continue
                if param_key == "extra_body" and isinstance(param_val, dict):
                    merged_extra = dict(request_kwargs["extra_body"])
                    merged_extra.update(param_val)
                    request_kwargs["extra_body"] = merged_extra
                    continue
                request_kwargs[param_key] = param_val
            r = self._client.chat.completions.create(**request_kwargs)
            normalized = _normalize_completion_response(r)
            content = normalized.content
        except RateLimitError as e:
            log.error("FATAL Rate-Limit (Perplexity '%s'): %s", query, e)
            raise AgentRateLimited(self._model, e)
        except APIStatusError as e:
            if e.status_code == 429:
                log.error("FATAL Rate-Limit (Perplexity '%s'): %s", query, e)
                raise AgentRateLimited(self._model, e)
            log.error(
                "Perplexity-Suche fehlgeschlagen fuer '%s': %s", query, e
            )
            return _empty
        except OpenAIError as e:
            log.error(
                "Perplexity-Suche fehlgeschlagen fuer '%s': %s", query, e
            )
            return _empty

        # Extract structured citations from the API response
        citations: list[str] = []
        raw = normalized.raw or {}

        # Perplexity returns citations as a top-level field
        api_citations = raw.get("citations", [])
        if api_citations and isinstance(api_citations, list):
            citations = [str(c) for c in api_citations if c]

        # Fallback: extract URLs from the answer text
        if not citations:
            citations = extract_urls(content)

        # Extract related questions (useful for query planning)
        related: list[str] = []
        api_related = raw.get("related_questions", [])
        if api_related and isinstance(api_related, list):
            related = [str(q) for q in api_related if q]

        data: dict[str, Any] = {
            "answer": content,
            "citations": citations,
            "related_questions": related,
            "_prompt_tokens": normalized.prompt_tokens,
            "_completion_tokens": normalized.completion_tokens,
        }

        # Token usage from the API response
        if hasattr(r, "usage") and r.usage:
            data["_prompt_tokens"] = (
                getattr(r.usage, "prompt_tokens", 0) or 0
            )
            data["_completion_tokens"] = (
                getattr(r.usage, "completion_tokens", 0) or 0
            )

        with self._cache_lock:
            self._cache[key] = data
        return data

    def is_available(self) -> bool:
        return self._client is not None


# =========================================================
# ProviderContext and factory
# =========================================================


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """Container holding the active LLM and search providers."""

    llm: LLMProvider
    search: SearchProvider


def create_providers(settings: Settings) -> ProviderContext:
    """Create both providers from *settings* (env-var / auto-create path).

    This is the single entry point for wiring up provider instances
    when no explicit providers are given.  The OpenAI client is shared
    between LLM and search providers for connection pooling.
    """
    client = OpenAI(
        base_url=settings.server.litellm_base_url,
        api_key=settings.server.litellm_api_key,
    )

    llm = LiteLLM(
        api_key=settings.server.litellm_api_key,
        base_url=settings.server.litellm_base_url,
        default_model=settings.models.reasoning_model,
        classify_model=settings.models.classify_model,
        summarize_model=settings.models.summarize_model,
        evaluate_model=settings.models.evaluate_model,
    )
    # Share the same client instance for connection pooling.
    llm._client = client

    search = PerplexitySearch(
        api_key=settings.server.litellm_api_key,
        base_url=settings.server.litellm_base_url,
        model=settings.models.search_model,
        cache_maxsize=settings.agent.search_cache_maxsize,
        cache_ttl=settings.agent.search_cache_ttl,
        _client=client,
    )

    return ProviderContext(llm=llm, search=search)
