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
from typing import Any, Callable

from openai import OpenAI

from inqtrix.exceptions import (
    AgentRateLimited,
    AgentTimeout,
    AzureFoundryWebSearchAPIError,
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

_AUTH_SCOPE = "https://cognitiveservices.azure.com/.default"

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
    """Query the web via the Azure Foundry Responses API (Web Search Tool).

    The agent must already exist.  Pass its ``agent_name`` (and optionally
    ``agent_version``) to the constructor.

    Authentication (checked in order of priority):

    * **api_key** — a static API key string from the Azure Foundry portal
      (Project Settings → Keys & Endpoint).  Simplest option.
    * **credential** — any ``azure-identity`` credential object you built
      yourself (e.g. ``ManagedIdentityCredential``).
    * **tenant_id + client_id + client_secret** — builds a
      ``ClientSecretCredential`` (Service Principal) automatically.
    * **None of the above** — falls back to ``DefaultAzureCredential``
      which tries ``az login``, Managed Identity, VS Code sign-in, etc.
    """

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
            self._client = OpenAI(
                base_url=base_url,
                api_key=self._make_token_provider(),
                timeout=timeout,
                max_retries=_SDK_MAX_RETRIES,
            )

    # ------------------------------------------------------------------
    # Credential resolution (shared pattern with AzureFoundryBingSearch)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_credential(
        *,
        credential: Any | None,
        tenant_id: str | None,
        client_id: str | None,
        client_secret: str | None,
    ) -> Any:
        """Build or return an Azure credential object.

        Only called when no ``api_key`` was provided.  The
        ``azure-identity`` package is imported lazily so users who
        authenticate via API key do not need it installed.
        """
        if credential is not None:
            return credential

        from azure.identity import ClientSecretCredential, DefaultAzureCredential

        if tenant_id and client_id and client_secret:
            return ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
        return DefaultAzureCredential()

    def _make_token_provider(self) -> Callable[[], str]:
        """Return a callable that yields a fresh bearer token per request."""
        cred = self._credential
        scope = _AUTH_SCOPE

        def _get_token() -> str:
            return cred.get_token(scope).token

        return _get_token

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
        """Execute a web search via Azure Foundry Web Search (Responses API)."""
        self._clear_nonfatal_notice()

        if deadline is not None:
            _check_deadline(deadline)

        # Parameters without direct Responses-API equivalents
        _ = search_context_size
        _ = search_mode
        _ = return_related

        effective_query = _apply_domain_filters(query, domain_filter)
        hint = _build_recency_language_hints(recency_filter, language_filter)

        timeout = _bounded_timeout(self._timeout, deadline)

        try:
            return self._execute_search(effective_query, hint, timeout)
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
        except Exception as exc:
            exc_str = str(exc).lower()
            if "rate" in exc_str and "limit" in exc_str:
                raise AgentRateLimited(self._agent_name, exc) from exc
            if "timeout" in exc_str or "timed out" in exc_str:
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
        """Return True when the provider is ready to serve requests."""
        return bool(self._agent_name and self._project_endpoint)
