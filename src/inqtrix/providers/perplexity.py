"""Perplexity Agent API search provider.

Web search via Perplexity's Agent API through the native ``perplexityai``
SDK (``client.responses.create``). Each query returns a synthesized answer
with inline ``[id]`` citations plus a per-source ``search_results`` list
(id, url, title, full snippet, date) -- both surfaced through
:class:`~inqtrix.search_result.GroundedSearchResult`. The integer source
``id`` is preserved as :attr:`GroundedSource.rank`, which is the anchor the
answer's inline ``[id]`` citations reference. Results are cached with a TTL
to avoid redundant API calls within one research run.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
from typing import Any

from cachetools import TTLCache
from perplexity import APIError, APIStatusError, Perplexity, RateLimitError

from inqtrix.constants import SEARCH_TIMEOUT
from inqtrix.exceptions import AgentRateLimited
from inqtrix.providers.base import (
    SearchProvider,
    _NonFatalNoticeMixin,
    _SDK_MAX_RETRIES,
    _apply_domain_filters,
    _bounded_timeout,
    _build_recency_language_hints,
)
from inqtrix.search_result import GroundedSearchResult, GroundedSource

log = logging.getLogger("inqtrix")


class PerplexitySearch(_NonFatalNoticeMixin, SearchProvider):
    """Query the Perplexity Agent API and normalize the grounded result.

    Use this provider when web search should come from Perplexity's Agent
    API. One call returns both a synthesized, inline-cited answer and the
    structured per-source ``search_results`` that back it, so the evidence
    pipeline gets rich per-source snippets and a ready cross-matched
    synthesis in a single round trip.

    Attributes:
        _client (Perplexity): Native Perplexity SDK client.
        _model (str): Agent reasoning model (e.g. ``"sonar"``).
        _preset (str | None): Optional Agent preset (e.g. a fast-search
            preset) selected instead of an explicit model.
        _instructions (str | None): Optional system instructions passed to
            the Agent.
        _cache (TTLCache): In-memory cache of normalized search results.
        _cache_lock (threading.Lock): Lock guarding cache access for
            multi-threaded search runs.
        _request_params (dict[str, Any]): Extra request parameters merged
            into the SDK call after reserved keys are filtered out.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        preset: str = "fast-search",
        model: str | None = None,
        instructions: str | None = None,
        cache_maxsize: int = 256,
        cache_ttl: int = 3600,
        request_params: dict[str, Any] | None = None,
        # Internal: accept a pre-built client from provider factories/tests.
        _client: Perplexity | None = None,
    ) -> None:
        """Initialize the Perplexity Agent API provider.

        Args:
            api_key: API key for the Perplexity API.
            base_url: Optional API base URL override. Leave ``None`` to use
                the SDK default (``https://api.perplexity.ai``); set it only
                to target a Perplexity-compatible proxy.
            preset: Agent preset that drives the search. The default
                ``"fast-search"`` works out of the box: it bundles web search
                and the system prompt that yields inline ``[n]`` citations.
                Other values: ``"pro-search"``, ``"deep-research"``.
            model: Optional explicit Agent model (e.g. ``"perplexity/sonar"``,
                ``"openai/gpt-5.5"``). When set it overrides ``preset`` and you
                are responsible for citation behaviour via ``instructions``.
            instructions: Optional system instructions for the Agent.
            cache_maxsize: Maximum number of normalized search responses
                kept in the in-memory cache. The default is ``256``.
            cache_ttl: Cache time-to-live in seconds. The default is
                ``3600``.
            request_params: Optional extra SDK parameters merged into each
                ``responses.create`` call after reserved keys are filtered.
            _client: Optional prebuilt SDK client used internally by the
                provider factory, config bridge, and tests.

        Example:
            >>> from inqtrix import PerplexitySearch
            >>> search = PerplexitySearch(api_key="test-key")  # fast-search preset
            >>> search.is_available()
            True
        """
        if _client is not None:
            self._client = _client
        else:
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "max_retries": _SDK_MAX_RETRIES,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = Perplexity(**client_kwargs)
        self._model = model
        self._preset = preset
        self._instructions = instructions
        self._cache: TTLCache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self._cache_lock = threading.Lock()
        self._request_params = dict(request_params or {})

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
    ) -> GroundedSearchResult:
        """Execute a Perplexity Agent API search and normalize the response.

        Args:
            query: User-facing search query text.
            search_context_size: Accepted for ABC compatibility; the Agent
                decides its own context depth, so this value is not used.
            recency_filter: Optional freshness hint, applied as best-effort
                prompt guidance prepended to the query.
            language_filter: Optional language hints, applied as best-effort
                prompt guidance.
            domain_filter: Optional allow/deny domain list, injected as
                ``site:`` / ``-site:`` operators into the query text.
            search_mode: Accepted for ABC compatibility; not used.
            return_related: Accepted for ABC compatibility; the Agent API
                response does not surface related questions in this shape.
            deadline: Optional absolute monotonic deadline for the run.

        Returns:
            GroundedSearchResult: The synthesized answer plus per-source
            records with full snippets. Cached results are deep-copied.

        Raises:
            AgentRateLimited: If the backend returns HTTP 429 or the SDK
                raises ``RateLimitError``.
        """
        self._clear_nonfatal_notice()
        cache_parts = [
            query,
            str(recency_filter),
            str(language_filter),
            str(domain_filter),
            str(self._model),
            str(self._preset),
        ]
        key = hashlib.sha256("|".join(cache_parts).encode()).hexdigest()
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)

        effective_query = _apply_domain_filters(query, domain_filter)
        hint = _build_recency_language_hints(recency_filter, language_filter)
        user_input = f"{hint}\n\n{effective_query}" if hint else effective_query

        create_kwargs: dict[str, Any] = {
            "input": user_input,
            "tools": [{"type": "web_search"}],
            "stream": False,
            "timeout": _bounded_timeout(SEARCH_TIMEOUT, deadline),
        }
        # An explicit model wins; otherwise the preset (default "fast-search")
        # drives the agent and bundles the inline-citation system prompt.
        if self._model:
            create_kwargs["model"] = self._model
        else:
            create_kwargs["preset"] = self._preset
        if self._instructions:
            create_kwargs["instructions"] = self._instructions
        for param_key, param_val in self._request_params.items():
            if param_key in {"input", "tools", "stream"}:
                continue
            create_kwargs[param_key] = param_val

        try:
            response = self._client.responses.create(**create_kwargs)
        except RateLimitError as exc:
            log.error("FATAL Rate-Limit (Perplexity '%s'): %s", query, exc)
            raise AgentRateLimited(self._model, exc) from exc
        except APIStatusError as exc:
            if exc.status_code == 429:
                log.error("FATAL Rate-Limit (Perplexity '%s'): %s", query, exc)
                raise AgentRateLimited(self._model, exc) from exc
            log.error("Perplexity-Suche fehlgeschlagen fuer '%s': %s", query, exc)
            self._set_nonfatal_notice(
                f"Perplexity-Suche fehlgeschlagen fuer Query '{query[:80]}'; "
                "leeres Ergebnis wird weiterverwendet."
            )
            return GroundedSearchResult()
        except APIError as exc:
            log.error("Perplexity-Suche fehlgeschlagen fuer '%s': %s", query, exc)
            self._set_nonfatal_notice(
                f"Perplexity-Suche fehlgeschlagen fuer Query '{query[:80]}'; "
                "leeres Ergebnis wird weiterverwendet."
            )
            return GroundedSearchResult()

        result = self._parse_response(response)
        if not result.answer and not result.sources:
            self._set_nonfatal_notice(
                f"Perplexity-Suche fuer '{query[:80]}' lieferte keine Textantwort"
            )

        with self._cache_lock:
            self._cache[key] = result
        return copy.deepcopy(result)

    @staticmethod
    def _parse_response(response: Any) -> GroundedSearchResult:
        """Extract the answer and per-source records from an Agent response."""
        answer = ""
        sources: list[GroundedSource] = []
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "search_results":
                for result in getattr(item, "results", []) or []:
                    url = str(getattr(result, "url", "") or "").strip()
                    if not url:
                        continue
                    raw_id = getattr(result, "id", 0)
                    try:
                        rank = int(raw_id)
                    except (TypeError, ValueError):
                        rank = len(sources) + 1
                    sources.append(
                        GroundedSource(
                            url=url,
                            title=str(getattr(result, "title", "") or ""),
                            snippet=str(getattr(result, "snippet", "") or ""),
                            date=str(getattr(result, "date", "") or ""),
                            last_updated=str(getattr(result, "last_updated", "") or ""),
                            rank=rank,
                            origin="search_results",
                        )
                    )
            elif item_type == "message" and not answer:
                parts = [
                    str(getattr(part, "text", "") or "")
                    for part in getattr(item, "content", []) or []
                    if getattr(part, "type", "output_text") == "output_text"
                ]
                answer = "".join(parts).strip()

        if not answer:
            answer = str(getattr(response, "output_text", "") or "").strip()

        usage = getattr(response, "usage", None)
        return GroundedSearchResult(
            answer=answer,
            sources=sources,
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

    def is_available(self) -> bool:
        """Report whether the provider is configured to attempt requests.

        Returns:
            bool: ``True`` when the internal SDK client was constructed.
        """
        return self._client is not None

    @property
    def search_model(self) -> str:
        """Operator-facing identifier of the Agent backend in use.

        The constructor-supplied model (when set) or preset identifier,
        surfaced in ``GET /health`` and ``GET /v1/stacks``.
        """
        return self._model or self._preset
