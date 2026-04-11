"""Perplexity Sonar search provider.

Web search via Perplexity's Sonar API — works with both a LiteLLM proxy
and the direct Perplexity API endpoint.  Results are cached with a TTL
to avoid redundant API calls within the same research run.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
from typing import Any

from cachetools import TTLCache
from openai import OpenAI, OpenAIError, RateLimitError, APIStatusError

from inqtrix.constants import SEARCH_TIMEOUT
from inqtrix.exceptions import AgentRateLimited
from inqtrix.providers.base import (
    SearchProvider,
    _NonFatalNoticeMixin,
    _SDK_MAX_RETRIES,
    _bounded_timeout,
    _normalize_completion_response,
)
from inqtrix.urls import extract_urls

log = logging.getLogger("inqtrix")


class PerplexitySearch(_NonFatalNoticeMixin, SearchProvider):
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
        self._client = _client or OpenAI(
            base_url=base_url, api_key=api_key, max_retries=_SDK_MAX_RETRIES,
        )
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
        self._clear_nonfatal_notice()
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
            self._set_nonfatal_notice(
                f"Perplexity-Suche fehlgeschlagen fuer Query '{query[:80]}'; leeres Ergebnis wird weiterverwendet."
            )
            return _empty
        except OpenAIError as e:
            log.error(
                "Perplexity-Suche fehlgeschlagen fuer '%s': %s", query, e
            )
            self._set_nonfatal_notice(
                f"Perplexity-Suche fehlgeschlagen fuer Query '{query[:80]}'; leeres Ergebnis wird weiterverwendet."
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
