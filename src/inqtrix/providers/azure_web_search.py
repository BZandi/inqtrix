"""Azure Foundry Web Search adapter for the SearchProvider interface.

Uses the OpenAI Responses API (``client.responses.create``) to query an
Azure Foundry Web Search agent.  Unlike :class:`AzureFoundryBingSearch`
(which drives the Agent-Service Thread/Message/Run pipeline), this
provider talks directly to the project-scoped ``/openai/v1/responses``
endpoint — the same path shape described by newer
``AIProjectClient.get_openai_client()`` docs, but constructed manually so
we stay compatible with ``openai >=1.66, <2.0``.

The agent must be created beforehand (via Azure Portal or CLI).  Pass
its ``agent_name`` (and optionally ``agent_version``) to the constructor.

Three authentication modes are supported (tried in this order):

1. **API key** — pass ``api_key`` to the constructor (simplest path).
2. **Service Principal** — pass ``tenant_id``, ``client_id``,
   ``client_secret`` to the constructor.
3. **DefaultAzureCredential** — fallback when neither of the above is
   given; works with ``az login``, Managed Identity, VS Code sign-in,
   etc.

Install the required SDKs::

    uv sync
"""

from __future__ import annotations

import logging
from typing import Any

from openai import APIStatusError, OpenAI, OpenAIError, RateLimitError

from inqtrix.exceptions import (
    AgentRateLimited,
    AgentTimeout,
    AzureFoundryWebSearchAPIError,
)
from inqtrix.providers._azure_common import (
    extract_azure_api_error_details,
    resolve_azure_credential,
)
from inqtrix.providers.base import (
    SearchProvider,
    _NonFatalNoticeMixin,
    _SDK_MAX_RETRIES,
    _apply_domain_filters,
    _bounded_timeout,
    _build_recency_language_hints,
    _check_deadline,
)
from inqtrix.urls import extract_urls

log = logging.getLogger("inqtrix")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AUTH_SCOPE = "https://ai.azure.com/.default"

_EMPTY_RESULT: dict[str, Any] = {
    "answer": "",
    "citations": [],
    "related_questions": [],
    "_prompt_tokens": 0,
    "_completion_tokens": 0,
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AzureFoundryWebSearch(_NonFatalNoticeMixin, SearchProvider):
    """Query the web via the Azure Foundry Responses API and Web Search tool.

    Use this provider when search should run through a pre-created Azure
    Foundry agent referenced by name and optional version. It is the newer
    Azure search path compared with ``AzureFoundryBingSearch`` and works
    through the project-scoped Responses API instead of the older
    Thread/Run agent service path.

    Attributes:
        _project_endpoint (str): Azure AI Foundry project endpoint.
        _agent_name (str): Agent name referenced in the Responses API.
        _agent_version (str): Optional agent version override.
        _timeout (float): Per-call timeout budget used before deadline
            clamping.
        _credential (Any | None): Resolved Azure credential when the
            provider authenticates via Entra ID instead of static API key.
        _client (OpenAI): OpenAI SDK client pointing at the project-scoped
            Foundry ``/openai/v1/`` endpoint.
    """

    supported_search_parameters = frozenset({
        "recency_filter",
        "language_filter",
        "domain_filter",
    })

    def __init__(
        self,
        *,
        project_endpoint: str,
        agent_name: str,
        agent_version: str = "",
        api_key: str | None = None,
        credential: Any | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the Azure Foundry Web Search provider.

        Use the constructor when a Web Search-capable Foundry agent
        already exists and should be addressed by name through the
        Responses API. Authentication is resolved in this order: static
        ``api_key``, explicit ``credential``, Service Principal fields,
        then ``DefaultAzureCredential``.

        Args:
            project_endpoint: Azure AI Foundry project endpoint.
            agent_name: Name of the existing Web Search agent.
            agent_version: Optional version override. Leave empty to let
                Foundry resolve the default or latest version.
            api_key: Optional static Foundry project API key.
            credential: Optional prebuilt Azure credential object.
            tenant_id: Optional Entra tenant ID for automatic Service
                Principal auth.
            client_id: Optional Entra client ID for automatic Service
                Principal auth.
            client_secret: Optional Entra client secret for automatic
                Service Principal auth.
            timeout: Default timeout budget in seconds for one search call.

        Raises:
            ValueError: If ``project_endpoint`` or ``agent_name`` is empty.

        Example:
            >>> from inqtrix import AzureFoundryWebSearch
            >>> search = AzureFoundryWebSearch(
            ...     project_endpoint="https://example.services.ai.azure.com/api/projects/demo",
            ...     agent_name="web-search-agent",
            ...     api_key="test-key",
            ... )
            >>> search.is_available()
            True
        """
        if not project_endpoint:
            raise ValueError("project_endpoint ist erforderlich")
        if not agent_name:
            raise ValueError("agent_name ist erforderlich")

        self._project_endpoint = project_endpoint.rstrip("/")
        self._agent_name = agent_name
        self._agent_version = agent_version
        self._timeout = timeout

        # Azure Foundry's project-scoped Responses API requires the v1 path.
        base_url = f"{self._project_endpoint}/openai/v1/"

        if api_key:
            # ── Path 1: static API key ──────────────────────────────
            self._credential = None
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=_SDK_MAX_RETRIES,
            )
        else:
            # ── Path 2/3/4: Entra ID token (credential / SP / default) ─
            self._credential = self._resolve_credential(
                credential=credential,
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
            # Obtain a bearer token at construction time.  Tokens are
            # typically valid for 60-75 minutes which is sufficient for
            # a research run.  We cannot pass a callable here because
            # the OpenAI SDK expects a plain string for ``api_key``.
            token = self._credential.get_token(_AUTH_SCOPE).token
            self._client = OpenAI(
                base_url=base_url,
                api_key=token,
                timeout=timeout,
                max_retries=_SDK_MAX_RETRIES,
            )

    # ------------------------------------------------------------------
    # Credential resolution (shared pattern with AzureFoundryBingSearch)
    # ------------------------------------------------------------------

    _resolve_credential = staticmethod(resolve_azure_credential)
    _extract_api_error_details = staticmethod(extract_azure_api_error_details)

    # ------------------------------------------------------------------
    # Query helpers (reuse BraveSearch / BingSearch patterns)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # SearchProvider interface
    # ------------------------------------------------------------------

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
        """Execute a search through the Foundry Responses API.

        Use this method when the runtime should query a Web Search agent
        by agent reference instead of by opaque agent ID. Several generic
        Inqtrix search hints are only best-effort for this backend because
        the Responses API does not expose first-class fields for them.

        Args:
            query: User-facing search query text.
            search_context_size: Unsupported at runtime for this backend.
                The agent's own configuration decides how much web context
                it uses, so this value is ignored.
            recency_filter: Optional recency hint. The provider turns it
                into best-effort prompt guidance rather than a guaranteed
                backend filter.
            language_filter: Optional language hint list. The provider uses
                it only for best-effort prompt guidance.
            domain_filter: Optional domain include/exclude list. The
                provider injects ``site:`` and ``-site:`` operators into the
                query string because the Responses API does not offer a
                dedicated domain-filter field here.
            search_mode: Unsupported for this backend. Passing a value has
                no effect.
            return_related: Unsupported for this backend. Passing ``True``
                does not populate ``related_questions``.
            deadline: Optional absolute monotonic deadline for the full
                agent run.

        Returns:
            dict[str, Any]: Normalized result with ``answer`` from the
            Responses API output text, ``citations`` from URL-citation
            annotations or URL fallback extraction, ``related_questions``
            as an empty list, and usage counts copied into
            ``_prompt_tokens`` and ``_completion_tokens`` when available.

        Raises:
            AgentTimeout: If the global deadline has already elapsed.
            AgentRateLimited: If the backend surfaces a fatal rate limit.
        """
        self._clear_nonfatal_notice()

        if deadline is not None:
            _check_deadline(deadline)

        # Unsupported hints (search_context_size, search_mode, return_related)
        # are filtered out by the search node via supported_search_parameters;
        # they remain in the signature only for ABC compatibility.

        effective_query = _apply_domain_filters(query, domain_filter)
        hint = _build_recency_language_hints(recency_filter, language_filter)

        timeout = _bounded_timeout(self._timeout, deadline)

        try:
            result = self._execute_search(effective_query, hint, timeout)
        except AgentTimeout:
            raise
        except AgentRateLimited:
            raise
        except Exception as exc:
            log.error(
                "Azure-Foundry-WebSearch fehlgeschlagen fuer '%s': %s",
                query[:80],
                exc,
            )
            self._set_nonfatal_notice(
                f"Azure-Foundry-WebSearch fehlgeschlagen fuer Query "
                f"'{query[:80]}': {exc}; leeres Ergebnis wird weiterverwendet."
            )
            return dict(_EMPTY_RESULT)

        if not result.get("answer"):
            self._set_nonfatal_notice(
                f"Azure-Foundry-WebSearch fuer '{query[:80]}' lieferte keine Textantwort"
            )
        return result

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute_search(
        self,
        query: str,
        hint: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        """Call ``client.responses.create`` with the agent reference."""
        # Build user input — prepend hint if present
        if hint:
            user_content = f"{hint}\n\n{query}"
        else:
            user_content = query

        agent_ref: dict[str, str] = {
            "name": self._agent_name,
            "type": "agent_reference",
        }
        if self._agent_version:
            agent_ref["version"] = self._agent_version

        try:
            response = self._client.responses.create(
                input=[{"role": "user", "content": user_content}],
                extra_body={"agent_reference": agent_ref},
                timeout=timeout,
            )
        except RateLimitError as exc:
            raise AgentRateLimited(self._agent_name, exc) from exc
        except APIStatusError as exc:
            details = self._extract_api_error_details(exc)
            if details["status_code"] == 429:
                raise AgentRateLimited(self._agent_name, exc) from exc
            raise AzureFoundryWebSearchAPIError(
                agent_name=self._agent_name,
                message=details["message"] or str(exc),
                original=exc,
            ) from exc
        except OpenAIError as exc:
            exc_text = str(exc).lower()
            if "timeout" in exc_text or "timed out" in exc_text:
                raise AgentTimeout(
                    f"Azure-Foundry-WebSearch Timeout fuer '{query[:80]}'"
                ) from exc
            raise AzureFoundryWebSearchAPIError(
                agent_name=self._agent_name,
                message=str(exc),
                original=exc,
            ) from exc

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: Any) -> dict[str, Any]:
        """Extract answer text and citation URLs from a Responses API reply."""
        answer = getattr(response, "output_text", "") or ""
        citations: list[str] = []
        seen_urls: set[str] = set()

        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) != "output_text":
                    continue
                for ann in getattr(content, "annotations", []):
                    if getattr(ann, "type", None) != "url_citation":
                        continue
                    url = getattr(ann, "url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        citations.append(url)

        # Fallback: extract URLs from answer text
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

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Report whether the provider is configured to attempt requests.

        Returns:
            bool: ``True`` when both the agent name and project endpoint
            are present, otherwise ``False``.
        """
        return bool(self._agent_name and self._project_endpoint)

    @property
    def search_model(self) -> str:
        """Foundry web-search agent identifier shown to operators.

        Format: ``"foundry-web:<agent_name>@<version_or_latest>"``
        (e.g. ``"foundry-web:bing-grounding-agent@v3"`` or
        ``"foundry-web:bing-grounding-agent@latest"`` when no version
        was pinned). The ``foundry-web`` prefix disambiguates from the
        Bing grounding variant (which uses ``foundry-bing:``).
        """
        version = self._agent_version or "latest"
        return f"foundry-web:{self._agent_name}@{version}"
