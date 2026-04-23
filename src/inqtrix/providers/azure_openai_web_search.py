"""Azure OpenAI Responses ``web_search`` adapter for the SearchProvider interface.

This provider uses the native Azure OpenAI Responses API web-search tool
(``tools=[{"type": "web_search"}]``). Unlike
``AzureFoundryWebSearch``, it does not require a pre-created Foundry
agent or an ``agent_reference``. It is the direct Azure OpenAI path
documented for the current ``web_search`` tool.

Supported generic Inqtrix search hints are intentionally conservative:

* ``domain_filter``: positive domains are mapped to
  ``filters.allowed_domains``; negative domains are appended to the
  query text via ``-site:`` operators as a best-effort fallback.
* ``deadline``: clamped to the remaining run budget.

The current public Azure documentation does not provide a stable,
explicit per-request mapping for the other generic Inqtrix hints, so the
provider does not silently translate them into prompt hints.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from openai import APIStatusError, OpenAI, OpenAIError, RateLimitError

from inqtrix.exceptions import (
    AgentRateLimited,
    AgentTimeout,
    AzureOpenAIWebSearchAPIError,
)
from inqtrix.providers._azure_common import (
    AZURE_OPENAI_DEFAULT_SCOPE,
    build_azure_openai_token_provider,
    extract_azure_api_error_details,
    normalize_openai_v1_base_url,
)
from inqtrix.providers.base import (
    SearchProvider,
    _NonFatalNoticeMixin,
    _SDK_MAX_RETRIES,
    _bounded_timeout,
    _check_deadline,
)
from inqtrix.urls import extract_urls

log = logging.getLogger("inqtrix")

_EMPTY_RESULT: dict[str, Any] = {
    "answer": "",
    "citations": [],
    "related_questions": [],
    "_prompt_tokens": 0,
    "_completion_tokens": 0,
}

# Responses-API include flag that asks Azure to attach the per-tool
# action sources (the URLs the model actually fetched while answering)
# to the response. Without this flag the response only carries inline
# annotations, which is harder to reconcile with citation lists. Kept
# as a single-string constant rather than a list because the OpenAI SDK
# accepts both shapes for this field but emits cleaner request payloads
# for the string form.
_DEFAULT_INCLUDE = "web_search_call.action.sources"


class AzureOpenAIWebSearch(_NonFatalNoticeMixin, SearchProvider):
    """Query the native Azure OpenAI Responses API ``web_search`` tool.

    Use this provider when web grounding should happen through the
    native Azure OpenAI ``web_search`` tool exposed by the Responses
    runtime, instead of through a pre-created Foundry agent
    (:class:`AzureFoundryWebSearch`) or Bing-grounded agent
    (:class:`AzureFoundryBingSearch`). The native tool runs entirely
    inside the Azure OpenAI deployment — no project endpoint, no
    agent reference — so it is the lightest-weight Azure search path.

    Authentication accepts the same four mutually exclusive modes as
    :class:`~inqtrix.providers.AzureOpenAILLM`: static ``api_key``,
    pre-built bearer-``azure_ad_token_provider``, generic Azure
    ``credential`` object, or Service Principal (``tenant_id`` +
    ``client_id`` + ``client_secret``). The constructor raises
    ``ValueError`` if zero or more than one auth mode is supplied.

    Tool grounding is governed by ``tool_choice`` (default ``"auto"``;
    set ``"required"`` to force the tool call, ``"none"`` to disable
    it) and the optional ``user_location`` payload which biases the
    geographic relevance of search hits. Both pass through unmodified
    to the Responses API.

    Attributes:
        supported_search_parameters: Hints honoured by this provider
            beyond the bare query. Currently only ``"domain_filter"``
            (translated to ``site:`` operators in the query) — Azure's
            native ``web_search`` tool does not yet expose recency,
            language or context-size knobs at request time.
    """

    supported_search_parameters = frozenset({"domain_filter"})

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_ad_token_provider: Callable[[], str] | None = None,
        credential: Any | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_scope: str = AZURE_OPENAI_DEFAULT_SCOPE,
        default_model: str = "gpt-4.1",
        timeout: float = 60.0,
        tool_choice: str | None = "auto",
        user_location: Mapping[str, Any] | None = None,
        include_action_sources: bool = True,
        default_headers: Mapping[str, str] | None = None,
        request_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the native Azure OpenAI web-search provider.

        Use the constructor when search should run through the
        deployment's built-in ``web_search`` tool (no Foundry agent
        needed). Exactly one endpoint input and exactly one
        authentication path must be supplied; the constructor enforces
        these constraints up front so misconfiguration fails before
        the first request.

        Args:
            azure_endpoint: Azure OpenAI resource endpoint such as
                ``"https://my-resource.openai.azure.com/"``. When
                provided, ``base_url`` must be omitted. The provider
                normalises this to the ``/openai/v1/`` shape internally.
            base_url: Full Azure OpenAI v1 base URL such as
                ``"https://my-resource.openai.azure.com/openai/v1/"``.
                When provided, ``azure_endpoint`` must be omitted.
            api_key: Static Azure OpenAI API key. Mutually exclusive
                with ``azure_ad_token_provider``, ``credential`` and
                the Service Principal fields. Use for the simplest
                setup; for CI/CD or Enterprise-Auth prefer the
                Service-Principal args or ``credential``.
            azure_ad_token_provider: Bearer-token provider from
                ``azure.identity.get_bearer_token_provider(...)``. Use
                this for Managed Identity, AzureCliCredential, or any
                custom credential type. Mutually exclusive with the
                other auth modes.
            credential: Optional pre-built ``azure.identity`` credential
                object (any ``TokenCredential``-like instance). When
                supplied, the provider builds an internal bearer-token
                provider from it. Mutually exclusive with the other
                auth modes.
            tenant_id: Entra tenant ID for automatic Service Principal
                auth. Must be supplied together with ``client_id`` and
                ``client_secret``. Mutually exclusive with the other
                auth modes.
            client_id: Entra client ID for automatic Service Principal
                auth. See ``tenant_id``.
            client_secret: Entra client secret for automatic Service
                Principal auth. See ``tenant_id``.
            token_scope: OAuth scope used when building an internal
                token provider from ``credential`` or the Service
                Principal fields. Defaults to the Azure OpenAI scope
                (``https://cognitiveservices.azure.com/.default``).
            default_model: Azure **deployment name** (not the
                underlying model name) hosting the ``web_search``
                tool. The default ``"gpt-4.1"`` is a placeholder and
                must usually be overridden in production. If the
                deployment is not reachable, requests fail with
                :class:`AzureOpenAIWebSearchAPIError` (HTTP 404).
            timeout: Default per-call timeout in seconds. The default
                is ``60.0``.
            tool_choice: Responses API ``tool_choice`` setting. One of
                ``"auto"`` (default — model decides), ``"required"``
                (force tool call, recommended for deterministic
                grounding), ``"none"`` (disable the tool, leaves the
                model ungrounded), or ``None``. Any other value raises
                ``ValueError``.
            user_location: Optional tool-level ``user_location``
                payload, e.g.
                ``{"type": "approximate", "country": "DE"}``. Forwarded
                verbatim into the tool config.
            include_action_sources: When ``True`` (default), requests
                that the response include the action-sources URLs the
                tool actually consumed. Set ``False`` to reduce
                response size when only inline annotations are needed.
            default_headers: Optional headers forwarded on every
                request (e.g. preview feature headers).
            request_params: Optional extra parameters merged into
                ``responses.create`` after reserved SDK keys
                (``model``, ``messages``, ``timeout``, ``stream``,
                ``input``) are filtered out.

        Raises:
            ValueError: If neither or both of ``azure_endpoint`` and
                ``base_url`` are provided, if more than one of the auth
                modes is supplied, if none is supplied, if the Service
                Principal fields are partially supplied, or if
                ``tool_choice`` is not in the allowed set.

        Example:
            >>> from inqtrix import AzureOpenAIWebSearch
            >>> search = AzureOpenAIWebSearch(
            ...     azure_endpoint="https://example.openai.azure.com/",
            ...     api_key="test-key",
            ...     default_model="my-gpt4o-search-deployment",
            ... )
            >>> search.is_available()
            True
        """
        if bool(azure_endpoint) == bool(base_url):
            raise ValueError(
                "Provide exactly one of azure_endpoint or base_url."
            )

        sp_fields = (tenant_id, client_id, client_secret)
        sp_any = any(sp_fields)
        sp_all = all(sp_fields)
        if sp_any and not sp_all:
            raise ValueError(
                "tenant_id, client_id and client_secret must all be "
                "provided together for Service Principal auth."
            )

        auth_modes_chosen = sum(
            1
            for present in (
                bool(api_key),
                bool(azure_ad_token_provider),
                bool(credential),
                sp_all,
            )
            if present
        )
        if auth_modes_chosen > 1:
            raise ValueError(
                "api_key, azure_ad_token_provider, credential and the "
                "Service Principal fields (tenant_id/client_id/client_secret) "
                "are mutually exclusive — pass exactly one auth mode."
            )
        if auth_modes_chosen == 0:
            raise ValueError(
                "An auth mode must be provided: api_key, azure_ad_token_provider, "
                "credential, or tenant_id+client_id+client_secret."
            )

        if azure_ad_token_provider is None and (credential is not None or sp_all):
            azure_ad_token_provider = build_azure_openai_token_provider(
                credential=credential,
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
                scope=token_scope,
            )

        if tool_choice not in {None, "auto", "required", "none"}:
            raise ValueError("tool_choice must be one of auto, required, none, or None.")

        self._default_model = default_model
        self._timeout = timeout
        self._tool_choice = tool_choice
        self._user_location = dict(user_location or {})
        self._include_action_sources = include_action_sources
        self._request_params = dict(request_params or {})

        client_kwargs: dict[str, Any] = {
            "base_url": self._normalize_base_url(base_url or azure_endpoint or ""),
            "api_key": azure_ad_token_provider or api_key or "",
            "timeout": timeout,
            "max_retries": _SDK_MAX_RETRIES,
        }
        if default_headers:
            client_kwargs["default_headers"] = dict(default_headers)

        self._client = OpenAI(**client_kwargs)

    _normalize_base_url = staticmethod(normalize_openai_v1_base_url)
    _extract_api_error_details = staticmethod(extract_azure_api_error_details)

    def _merge_request_params(
        self,
        base_kwargs: dict[str, Any],
        extra_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        reserved = {
            "model",
            "input",
            "timeout",
            "tools",
            "tool_choice",
            "include",
        }
        merged = dict(base_kwargs)
        for key, value in extra_params.items():
            if key in reserved:
                if key == "include" and isinstance(value, list):
                    existing = list(merged.get("include", []))
                    for item in value:
                        if item not in existing:
                            existing.append(item)
                    merged["include"] = existing
                continue
            merged[key] = value
        return merged

    @staticmethod
    def _normalize_allowed_domain(raw_domain: str) -> str:
        cleaned = raw_domain.strip()
        cleaned = cleaned.replace("https://", "").replace("http://", "")
        return cleaned.rstrip("/").strip()

    @staticmethod
    def _normalize_excluded_domain(raw_domain: str) -> str:
        cleaned = AzureOpenAIWebSearch._normalize_allowed_domain(raw_domain)
        return cleaned.split("/")[0].strip()

    @staticmethod
    def _split_domain_filters(
        domain_filter: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        allowed: list[str] = []
        excluded: list[str] = []
        for entry in domain_filter or []:
            text = str(entry or "").strip()
            if not text:
                continue
            if text.startswith("-"):
                normalized = AzureOpenAIWebSearch._normalize_excluded_domain(text[1:])
                if normalized and normalized not in excluded:
                    excluded.append(normalized)
                continue
            normalized = AzureOpenAIWebSearch._normalize_allowed_domain(text)
            if normalized and normalized not in allowed:
                allowed.append(normalized)
        return allowed[:100], excluded

    def _build_request(
        self,
        query: str,
        domain_filter: list[str] | None,
        timeout: float,
    ) -> dict[str, Any]:
        allowed_domains, excluded_domains = self._split_domain_filters(domain_filter)
        tool: dict[str, Any] = {"type": "web_search"}
        if self._user_location:
            tool["user_location"] = dict(self._user_location)
        if allowed_domains:
            tool["filters"] = {"allowed_domains": allowed_domains}

        effective_query = query.strip()
        if excluded_domains:
            suffix = " ".join(f"-site:{domain}" for domain in excluded_domains)
            effective_query = f"{effective_query} {suffix}".strip()

        request_kwargs: dict[str, Any] = {
            "model": self._default_model,
            "input": effective_query,
            "tools": [tool],
            "timeout": timeout,
        }
        if self._tool_choice is not None:
            request_kwargs["tool_choice"] = self._tool_choice
        if self._include_action_sources:
            request_kwargs["include"] = [_DEFAULT_INCLUDE]
        return self._merge_request_params(request_kwargs, self._request_params)

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
        """Execute one native Azure OpenAI web-search request.

        Routes the query through the deployment's ``web_search`` tool
        via the Responses runtime, then collapses the multi-part
        Responses payload into the standard Inqtrix search-result
        shape. Unsupported generic hints are intentionally ignored
        because Azure's native tool does not document a stable
        per-request mapping for them today.

        Args:
            query: Natural-language search query forwarded as the
                user input. Pre-processed to embed any
                ``domain_filter`` as ``site:`` operators.
            search_context_size: ABC-compatibility parameter — ignored
                by this provider; filtered out by the search node
                before this call.
            recency_filter: ABC-compatibility parameter — ignored
                (Azure ``web_search`` has no per-request recency knob).
            language_filter: ABC-compatibility parameter — ignored
                (no per-request language knob exposed).
            domain_filter: Optional list of domains. Embedded into
                the effective query as
                ``site:domain1 OR site:domain2``; honoured best-effort
                by the underlying Bing index.
            search_mode: ABC-compatibility parameter — ignored.
            return_related: ABC-compatibility parameter — ignored.
            deadline: Optional absolute monotonic deadline. The
                per-call timeout is clamped to the remaining budget;
                exceeding the deadline raises :class:`AgentTimeout`.

        Returns:
            Search-result dict with keys ``answer`` (model-generated
            text), ``citations`` (list of source URLs), ``related_questions``
            (always empty for this provider), ``_prompt_tokens`` and
            ``_completion_tokens`` (from the Responses ``usage`` block
            when present, otherwise ``0``). On non-deadline / non-
            rate-limit failure, the provider returns ``_EMPTY_RESULT``
            and stores a non-fatal notice for the search node.

        Raises:
            AgentTimeout: When the absolute run deadline has elapsed
                or the SDK times out the request.
            AgentRateLimited: When Azure returns HTTP 429 or the SDK
                raises ``RateLimitError``.
            AzureOpenAIWebSearchAPIError: When Azure returns a non-
                rate-limit API error.
        """
        self._clear_nonfatal_notice()

        if deadline is not None:
            _check_deadline(deadline)

        # Unsupported hints (search_context_size, recency_filter, language_filter,
        # search_mode, return_related) are filtered out by the search node via
        # supported_search_parameters; they remain in the signature only for
        # ABC compatibility.

        timeout = _bounded_timeout(self._timeout, deadline)

        try:
            request_kwargs = self._build_request(query, domain_filter, timeout)
            response = self._client.responses.create(**request_kwargs)
        except RateLimitError as exc:
            raise AgentRateLimited(self._default_model, exc) from exc
        except APIStatusError as exc:
            details = self._extract_api_error_details(exc)
            if details["status_code"] == 429:
                raise AgentRateLimited(self._default_model, exc) from exc
            raise AzureOpenAIWebSearchAPIError(
                model=self._default_model,
                status_code=details["status_code"],
                error_code=details["error_code"],
                message=details["message"] or str(exc),
                request_id=details["request_id"],
                original=exc,
            ) from exc
        except OpenAIError as exc:
            exc_text = str(exc).lower()
            if "timeout" in exc_text or "timed out" in exc_text:
                raise AgentTimeout(
                    f"Azure-OpenAI-WebSearch Timeout fuer '{query[:80]}'"
                ) from exc
            log.error(
                "Azure-OpenAI-WebSearch fehlgeschlagen fuer '%s': %s",
                query[:80],
                exc,
            )
            self._set_nonfatal_notice(
                f"Azure-OpenAI-WebSearch fehlgeschlagen fuer Query "
                f"'{query[:80]}': {exc}; leeres Ergebnis wird weiterverwendet."
            )
            return dict(_EMPTY_RESULT)

        result = self._parse_response(response)
        if not result.get("answer"):
            self._set_nonfatal_notice(
                f"Azure-OpenAI-WebSearch fuer '{query[:80]}' lieferte keine Textantwort"
            )
        return result

    @staticmethod
    def _append_url(url: str, citations: list[str], seen_urls: set[str]) -> None:
        cleaned = str(url or "").strip()
        if cleaned and cleaned not in seen_urls:
            seen_urls.add(cleaned)
            citations.append(cleaned)

    @classmethod
    def _append_action_sources(
        cls,
        sources: Any,
        citations: list[str],
        seen_urls: set[str],
    ) -> None:
        if not isinstance(sources, list):
            return
        for source in sources:
            if isinstance(source, str):
                cls._append_url(source, citations, seen_urls)
                continue
            if isinstance(source, dict):
                cls._append_url(source.get("url", ""), citations, seen_urls)
                continue
            cls._append_url(getattr(source, "url", ""), citations, seen_urls)

    @classmethod
    def _parse_response(cls, response: Any) -> dict[str, Any]:
        answer = getattr(response, "output_text", "") or ""
        citations: list[str] = []
        seen_urls: set[str] = set()

        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) != "output_text":
                        continue
                    for ann in getattr(content, "annotations", []):
                        if getattr(ann, "type", None) != "url_citation":
                            continue
                        cls._append_url(getattr(ann, "url", ""), citations, seen_urls)
                continue

            if item_type != "web_search_call":
                continue

            action = getattr(item, "action", None)
            if isinstance(action, dict):
                cls._append_action_sources(action.get("sources"), citations, seen_urls)
                continue
            cls._append_action_sources(
                getattr(action, "sources", None),
                citations,
                seen_urls,
            )

        if not citations and answer:
            citations = extract_urls(answer)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        return {
            "answer": answer,
            "citations": citations,
            "related_questions": [],
            "_prompt_tokens": input_tokens or 0,
            "_completion_tokens": output_tokens or 0,
        }

    def is_available(self) -> bool:
        """Report whether the provider has enough config to attempt requests.

        Configuration here means: a non-empty ``default_model``
        (Azure deployment name) was supplied to the constructor.
        Endpoint and auth are validated by the constructor itself
        (it raises ``ValueError`` on bad input), so by the time an
        instance exists the only field still subject to runtime
        truthiness is the deployment name. Backend reachability and
        deployment validity are not pre-validated.

        Returns:
            ``True`` when ``default_model`` is non-empty, otherwise
            ``False``.
        """
        return bool(self._default_model)

    @property
    def search_model(self) -> str:
        """Composite identifier showing both the Azure deployment and the search tool.

        Format: ``"<deployment>+web_search_tool"`` (e.g.
        ``"gpt-4.1+web_search_tool"``). The ``+web_search_tool``
        suffix tells the operator that search runs through the
        deployment's built-in Responses-API ``web_search`` tool —
        important to disambiguate from a plain ``AzureOpenAILLM``
        which uses the same deployment name but does no web search.
        """
        return f"{self._default_model}+web_search_tool"
