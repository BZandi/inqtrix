"""Direct Brave Search adapter for the SearchProvider interface."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from inqtrix.constants import SEARCH_TIMEOUT
from inqtrix.exceptions import AgentRateLimited, AgentTimeout
from inqtrix.providers import SearchProvider, _NonFatalNoticeMixin, _bounded_timeout, _check_deadline

log = logging.getLogger("inqtrix")


class BraveSearch(_NonFatalNoticeMixin, SearchProvider):
    """Query Brave Search directly without going through LiteLLM."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.search.brave.com/res/v1/web/search",
        result_count: int = 10,
        extra_params: dict[str, Any] | None = None,
        user_agent: str = "inqtrix/0.1",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("?")
        self._result_count = max(1, int(result_count))
        self._extra_params = dict(extra_params or {})
        self._user_agent = user_agent

    @staticmethod
    def _map_freshness(recency_filter: str | None) -> str | None:
        return {
            "day": "pd",
            "week": "pw",
            "month": "pm",
            "year": "py",
        }.get((recency_filter or "").strip().lower())

    @staticmethod
    def _result_count_for_context(search_context_size: str) -> int:
        return {
            "low": 5,
            "medium": 8,
            "high": 10,
        }.get((search_context_size or "high").strip().lower(), 10)

    @staticmethod
    def _apply_domain_filters(query: str, domain_filter: list[str] | None) -> str:
        if not domain_filter:
            return query

        suffix_parts: list[str] = []
        for raw_domain in domain_filter[:10]:
            domain = str(raw_domain or "").strip()
            if not domain:
                continue
            is_exclusion = domain.startswith("-")
            if is_exclusion:
                domain = domain[1:].strip()
            if not domain:
                continue
            token = f"site:{domain}"
            if is_exclusion:
                token = f"-site:{domain}"
            suffix_parts.append(token)

        if not suffix_parts:
            return query
        return f"{query} {' '.join(suffix_parts)}"

    def _request_json(
        self,
        *,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
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
        """Execute a Brave Search request and map it to the search contract."""
        _empty: dict[str, Any] = {
            "answer": "",
            "citations": [],
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

        self._clear_nonfatal_notice()
        if not self._api_key:
            return _empty
        if deadline is not None:
            _check_deadline(deadline)

        effective_query = self._apply_domain_filters(query, domain_filter)
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
        # Unsupported knobs are intentionally ignored rather than blocking the run.
        _ = search_mode
        _ = return_related

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

        return {
            "answer": "\n\n".join(snippets),
            "citations": citations,
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

    def is_available(self) -> bool:
        return bool(self._api_key)
