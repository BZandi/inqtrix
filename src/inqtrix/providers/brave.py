"""Direct Brave Search adapter for the SearchProvider interface.

Calls the Brave Web Search API (``/res/v1/web/search``) via ``urllib``
— no SDK dependency.  Results are mapped to the :class:`SearchProvider`
contract (``answer``, ``citations``, ``related_questions``, token
counts).

Unlike :class:`PerplexitySearch`, Brave returns raw web results (title +
description + extra snippets) rather than an LLM-generated summary.  The
provider concatenates all snippet blocks into a single ``answer`` string,
which the graph's ``search`` node then passes through the LLM summariser.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from inqtrix.constants import SEARCH_TIMEOUT
from inqtrix.exceptions import AgentRateLimited, AgentTimeout
from inqtrix.providers.base import SearchProvider, _NonFatalNoticeMixin, _apply_domain_filters, _bounded_timeout, _check_deadline

log = logging.getLogger("inqtrix")


class BraveSearch(_NonFatalNoticeMixin, SearchProvider):
    """Query the Brave Web Search API directly without LiteLLM.

    Use this provider when you want a simple direct-search backend with no
    extra SDK dependency and no LLM-generated search summary at the search
    stage itself. Unlike Perplexity, Brave returns raw result snippets,
    which Inqtrix later summarizes through the active LLM provider.

    Attributes:
        _api_key (str): Brave Search API subscription token.
        _base_url (str): Brave Web Search endpoint URL.
        _result_count (int): Maximum number of results requested per query.
        _extra_params (dict[str, Any]): Extra query parameters merged into
            every request.
        _user_agent (str): User-Agent header for direct Brave requests.
    """

    supported_search_parameters = frozenset({
        "search_context_size",
        "recency_filter",
        "language_filter",
        "domain_filter",
    })

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.search.brave.com/res/v1/web/search",
        result_count: int = 10,
        extra_params: dict[str, Any] | None = None,
        user_agent: str = "inqtrix/0.1",
    ) -> None:
        """Initialize the Brave Search provider.

        Use the constructor when a lightweight direct REST search backend
        is sufficient and you are comfortable with Inqtrix performing the
        later summarization step itself.

        Args:
            api_key: Brave Search API subscription token.
            base_url: Brave web-search endpoint URL. The default is
                ``"https://api.search.brave.com/res/v1/web/search"``.
            result_count: Maximum number of results requested per query.
                The default is ``10`` and values smaller than ``1`` are
                clamped up internally.
            extra_params: Optional additional query parameters forwarded to
                every request.
            user_agent: User-Agent header value. The default is
                ``"inqtrix/0.1"``.

        Example:
            >>> from inqtrix import BraveSearch
            >>> search = BraveSearch(api_key="test-key", result_count=8)
            >>> search.is_available()
            True
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("?")
        self._result_count = max(1, int(result_count))
        self._extra_params = dict(extra_params or {})
        self._user_agent = user_agent

    @staticmethod
    def _map_freshness(recency_filter: str | None) -> str | None:
        """Map an Inqtrix recency hint to the Brave ``freshness`` parameter.

        Brave does not expose a sub-day ``freshness`` value, so ``"hour"``
        falls back to ``"pd"`` (past day) — the most narrow real bucket
        Brave supports.
        """
        return {
            "hour": "pd",
            "day": "pd",
            "week": "pw",
            "month": "pm",
            "year": "py",
        }.get((recency_filter or "").strip().lower())

    @staticmethod
    def _result_count_for_context(search_context_size: str) -> int:
        """Map ``search_context_size`` to a Brave result count."""
        return {
            "low": 5,
            "medium": 8,
            "high": 10,
        }.get((search_context_size or "high").strip().lower(), 10)

    def _request_json(
        self,
        *,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """Send a GET request to the Brave Search API and return JSON."""
        url = f"{self._base_url}?{urlencode(params, doseq=True)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
                "X-Subscription-Token": self._api_key,
            },
            method="GET",
        )
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        return data if isinstance(data, dict) else {}

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
        """Execute a Brave search request and normalize the response.

        Use this method when you want direct web results rather than an
        API-generated answer. Brave supports only part of the generic
        Inqtrix search contract, so unsupported hints are intentionally
        ignored instead of blocking the run.

        Args:
            query: User-facing search query text.
            search_context_size: Inqtrix hint for how much web context to
                request. Brave maps this to a result count of roughly 5, 8,
                or 10 results for ``low``, ``medium``, and ``high``.
            recency_filter: Optional freshness hint such as ``day``,
                ``week``, ``month``, or ``year``. Brave maps this to its
                ``freshness`` query parameter.
            language_filter: Optional language hint list. Brave uses the
                first entry as ``search_lang`` when present.
            domain_filter: Optional domain include/exclude list. Brave does
                not support this natively, so the provider injects
                ``site:`` and ``-site:`` operators into the query string.
            search_mode: Unsupported by Brave. Passing a value has no
                effect and should generally be avoided.
            return_related: Unsupported by Brave. Passing ``True`` does not
                populate ``related_questions``.
            deadline: Optional absolute monotonic deadline for the full
                agent run.

        Returns:
            dict[str, Any]: Normalized result with ``answer`` containing
            concatenated Brave title/description/snippet text,
            ``citations`` as full result URLs, ``related_questions`` as an
            empty list, and zero token counts because Brave does not report
            LLM usage.

        Raises:
            AgentTimeout: If the full run deadline has already elapsed.
            AgentRateLimited: If Brave responds with HTTP 429.
        """
        _empty: dict[str, Any] = {
            "answer": "",
            "citations": [],
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

        self._clear_nonfatal_notice()
        if not self._api_key:
            self._set_nonfatal_notice("Brave-API-Key fehlt — Suche uebersprungen")
            return _empty
        if deadline is not None:
            _check_deadline(deadline)

        effective_query = _apply_domain_filters(query, domain_filter)
        params: dict[str, Any] = {
            "q": effective_query,
            "count": min(
                self._result_count,
                self._result_count_for_context(search_context_size),
            ),
        }

        freshness = self._map_freshness(recency_filter)
        if freshness:
            params["freshness"] = freshness
        if language_filter:
            params["search_lang"] = language_filter[0]

        # Brave does not expose direct equivalents for all Inqtrix hints.
        # Unsupported knobs (search_mode, return_related) are filtered out by
        # the search node via supported_search_parameters; they remain in the
        # signature only for ABC compatibility.

        params.update(self._extra_params)

        try:
            payload = self._request_json(
                params=params,
                timeout=_bounded_timeout(SEARCH_TIMEOUT, deadline),
            )
        except HTTPError as exc:
            if exc.code == 429:
                raise AgentRateLimited("brave-search", exc)
            log.error("Brave-Suche fehlgeschlagen fuer '%s': %s", query, exc)
            self._set_nonfatal_notice(
                f"Brave-Suche fehlgeschlagen fuer Query '{query[:80]}'; leeres Ergebnis wird weiterverwendet."
            )
            return _empty
        except (URLError, OSError, ValueError, AgentTimeout) as exc:
            log.error("Brave-Suche fehlgeschlagen fuer '%s': %s", query, exc)
            self._set_nonfatal_notice(
                f"Brave-Suche fehlgeschlagen fuer Query '{query[:80]}'; leeres Ergebnis wird weiterverwendet."
            )
            return _empty

        web_results = ((payload.get("web") or {}).get("results") or [])
        citations: list[str] = []
        snippets: list[str] = []

        for item in web_results[: self._result_count]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()

            extra_snippets = item.get("extra_snippets", [])
            extra_parts: list[str] = []
            if isinstance(extra_snippets, list):
                for snippet in extra_snippets[:2]:
                    text = str(snippet).strip()
                    if text:
                        extra_parts.append(text)

            block = "\n".join(
                part for part in [title, description, *extra_parts] if part
            )
            if block:
                snippets.append(block)
            if url and url not in citations:
                citations.append(url)

        answer = "\n\n".join(snippets)
        if not answer:
            self._set_nonfatal_notice(
                f"Brave-Suche fuer '{query[:80]}' lieferte keine Textantwort"
            )
        return {
            "answer": answer,
            "citations": citations,
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

    def is_available(self) -> bool:
        """Report whether the provider has enough config to attempt requests.

        Configuration here means: a non-empty Brave API key was
        supplied to the constructor. Brave key validity is not
        pre-validated; invalid keys surface as HTTP 401 on the first
        ``search()`` call.

        Returns:
            ``True`` when ``api_key`` is non-empty, otherwise ``False``.
        """
        return bool(self._api_key)

    @property
    def search_model(self) -> str:
        """Stable identifier of the Brave Search backend.

        Brave Search has no per-request "model" concept (it returns
        results from a single index per account tier). The constant
        string here surfaces in the operator-facing health and
        discovery payloads so a UI can label the stack honestly
        instead of falling back to the global ``Settings`` default.
        """
        return "brave-search-api"
