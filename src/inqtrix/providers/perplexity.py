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
    """Query Perplexity Sonar through either direct or proxy-backed APIs.

    Use this provider when web search should come from the Perplexity
    Sonar API. It supports both direct calls to ``api.perplexity.ai`` and
    indirect calls through a LiteLLM or other OpenAI-compatible proxy.
    This provider is specific to Sonar-style chat-completions search; it
    does not target Perplexity Deep Research or other product surfaces.

    Attributes:
        _client (OpenAI): Shared SDK client used for Sonar requests.
        _model (str): Effective Perplexity or proxy model identifier.
        _cache (TTLCache): In-memory cache of normalized search results.
        _cache_lock (threading.Lock): Lock guarding cache access for
            multi-threaded search runs.
        _request_params (dict[str, Any]): Extra request parameters merged
            into the SDK call after reserved keys are filtered out.
        _direct_mode (bool): Whether request options are formatted for the
            direct Perplexity API rather than a LiteLLM-style proxy.
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
        """Initialize the Perplexity Sonar provider.

        Use the constructor when search should be backed by Perplexity
        Sonar. The provider can talk to Perplexity directly or through a
        proxy, with ``direct_mode`` controlling how search options are
        encoded in ``extra_body``.

        Args:
            api_key: API key for the direct Perplexity endpoint or for the
                proxy that fronts it.
            base_url: Endpoint base URL. Use ``"http://localhost:4000/v1"``
                for a LiteLLM proxy or ``"https://api.perplexity.ai"`` for
                direct Perplexity access.
            model: Search model identifier. Typical values are
                ``"perplexity-sonar-pro-agent"`` through LiteLLM and
                ``"sonar-pro"`` for direct access.
            cache_maxsize: Maximum number of normalized search responses
                kept in the in-memory cache. The default is ``256``.
            cache_ttl: Cache time-to-live in seconds. The default is
                ``3600``.
            request_params: Optional extra SDK parameters merged into each
                request after reserved keys are filtered out.
            direct_mode: Optional explicit override for request formatting.
                Use ``True`` for the direct Perplexity API, ``False`` for a
                LiteLLM-style proxy, or omit it to auto-detect from
                ``base_url``.
            _client: Optional prebuilt SDK client used internally by the
                provider factory and config bridge.

        Example:
            >>> from inqtrix import PerplexitySearch
            >>> search = PerplexitySearch(
            ...     api_key="test-key",
            ...     base_url="https://api.perplexity.ai",
            ...     model="sonar-pro",
            ...     direct_mode=True,
            ... )
            >>> search.is_available()
            True
        """
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
        """Execute a Perplexity Sonar search and normalize the response.

        Use this method when the graph wants Perplexity-backed web search.
        Unlike the Azure search providers, Perplexity supports most
        Inqtrix search hints natively, so this backend is usually the best
        fit when you want strong per-call control over recency, domain,
        language, and academic search behavior.

        Args:
            query: User-facing search query text.
            search_context_size: Perplexity search-depth hint. Supported
                values are typically ``"low"``, ``"medium"``, and
                ``"high"``. The default is ``"high"``.
            recency_filter: Optional freshness filter such as ``"day"``,
                ``"week"``, ``"month"``, or ``"year"``.
            language_filter: Optional ISO 639-1 language codes such as
                ``["de"]``. Perplexity accepts the list directly.
            domain_filter: Optional domain include/exclude list. Direct
                Perplexity mode forwards this natively; proxy mode nests it
                inside ``web_search_options``.
            search_mode: Optional Perplexity mode such as ``"academic"``.
                Omit this when standard web search is sufficient.
            return_related: Whether related questions should be requested
                when the backend supports them. The default is ``False``.
            deadline: Optional absolute monotonic deadline for the full
                agent run.

        Returns:
            dict[str, Any]: Normalized result with ``answer`` from the
            Sonar response, ``citations`` as full URLs, optional
            ``related_questions``, and token counts from the API when the
            backend reports them. Cached results are returned in the same
            shape as fresh responses.

        Raises:
            AgentRateLimited: If the backend returns HTTP 429 or the SDK
                raises ``RateLimitError``.
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

        with self._cache_lock:
            self._cache[key] = data
        return data

    def is_available(self) -> bool:
        """Report whether the provider is configured to attempt requests.

        Returns:
            bool: ``True`` when the SDK client exists, otherwise ``False``.
        """
        return self._client is not None
