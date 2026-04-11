"""Azure Foundry Bing Search adapter for the SearchProvider interface.

Uses an Azure AI Foundry Agent with Bing Grounding to perform web searches.
The agent must be created beforehand (via Azure Portal, CLI, or the
``create_agent`` classmethod).  At runtime only a Thread + Message + Run
are created per search call -- the agent itself is reused.

Search parameters like freshness, market, and count are fixed at agent
creation time.  The provider bridges the gap to the SearchProvider
interface by injecting ``site:`` operators into the query text for domain
filtering and passing ``instructions`` as non-deterministic hints for
recency and language.

Install the required SDK::

    uv sync
"""

from __future__ import annotations

import logging
from typing import Any

from inqtrix.exceptions import AgentRateLimited, AgentTimeout, AzureFoundryBingAPIError
from inqtrix.providers.base import (
    SearchProvider,
    _NonFatalNoticeMixin,
    _apply_domain_filters,
    _bounded_timeout,
    _build_recency_language_hints,
    _check_deadline,
)
from inqtrix.urls import extract_urls

log = logging.getLogger("inqtrix")

# ---------------------------------------------------------------------------
# Guarded SDK imports
# ---------------------------------------------------------------------------
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    BingGroundingTool,
    AgentThreadCreationOptions,
    ThreadMessageOptions,
    MessageRole,
)
from azure.identity import ClientSecretCredential, DefaultAzureCredential

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMPTY_RESULT: dict[str, Any] = {
    "answer": "",
    "citations": [],
    "related_questions": [],
    "_prompt_tokens": 0,
    "_completion_tokens": 0,
}

DEFAULT_INSTRUCTIONS = (
    "Du bist ein Recherche-Assistent. "
    "Beantworte jede Frage ausschliesslich auf Basis aktueller Websuche-Ergebnisse. "
    "Nenne immer deine Quellen mit vollstaendigen URLs. "
    "Gib die Informationen so wieder, wie sie in den Quellen stehen, "
    "ohne eigene Interpretation hinzuzufuegen."
)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AzureFoundryBingSearch(_NonFatalNoticeMixin, SearchProvider):
    """Query the web via an Azure Foundry agent with Bing Grounding.

    Use this provider when search should run through a pre-created Azure
    AI Foundry agent that has the Bing Grounding tool attached. This is a
    good fit for Azure-native deployments that want Bing grounding but do
    not want to overload the reasoning provider with search-specific tool
    behavior.

    Attributes:
        _project_endpoint (str): Azure AI Foundry project endpoint.
        _agent_id (str): ID of the pre-created Bing agent.
        _timeout (float): Per-call timeout budget used before deadline
            clamping.
        _credential (Any): Resolved Azure credential used by the Foundry
            SDK client.
        _client (AIProjectClient): Azure AI Projects SDK client used to
            create runs and list messages.
    """

    def __init__(
        self,
        *,
        project_endpoint: str,
        agent_id: str,
        credential: Any | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the Azure Foundry Bing search provider.

        Use the constructor when a Bing Grounding agent already exists in
        Azure AI Foundry and the runtime should only reuse that agent by
        ID. Authentication is resolved in this order: explicit
        ``credential``, Service Principal fields, then
        ``DefaultAzureCredential``.

        Args:
            project_endpoint: Azure AI Foundry project endpoint.
            agent_id: ID of the existing Bing Grounding agent.
            credential: Optional prebuilt Azure credential object.
            tenant_id: Optional Entra tenant ID used only when building a
                ``ClientSecretCredential`` internally.
            client_id: Optional Entra client ID used only when building a
                ``ClientSecretCredential`` internally.
            client_secret: Optional Entra client secret used only when
                building a ``ClientSecretCredential`` internally.
            timeout: Default timeout budget in seconds for one search call.

        Raises:
            ValueError: If ``project_endpoint`` or ``agent_id`` is empty.

        Example:
            >>> from inqtrix import AzureFoundryBingSearch
            >>> search = AzureFoundryBingSearch(
            ...     project_endpoint="https://example.services.ai.azure.com/api/projects/demo",
            ...     agent_id="agent_123",
            ... )
            >>> search.is_available()
            True
        """
        if not project_endpoint:
            raise ValueError("project_endpoint ist erforderlich")
        if not agent_id:
            raise ValueError("agent_id ist erforderlich")

        self._project_endpoint = project_endpoint
        self._agent_id = agent_id
        self._timeout = timeout
        self._credential = self._resolve_credential(
            credential=credential,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
        self._client = AIProjectClient(
            endpoint=self._project_endpoint,
            credential=self._credential,
        )

    # ------------------------------------------------------------------
    # Credential resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_credential(
        *,
        credential: Any | None,
        tenant_id: str | None,
        client_id: str | None,
        client_secret: str | None,
    ) -> Any:
        """Build or return an Azure credential object."""
        if credential is not None:
            return credential
        if tenant_id and client_id and client_secret:
            return ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
        return DefaultAzureCredential()

    # ------------------------------------------------------------------
    # Agent creation convenience
    # ------------------------------------------------------------------

    @classmethod
    def create_agent(
        cls,
        *,
        project_endpoint: str,
        bing_connection_id: str,
        model: str = "gpt-4o",
        instructions: str = DEFAULT_INSTRUCTIONS,
        agent_name: str = "inqtrix-bing-search",
        freshness: str | None = None,
        market: str | None = None,
        set_lang: str | None = None,
        count: int = 10,
        credential: Any | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = 60.0,
    ) -> "AzureFoundryBingSearch":
        """Create a Bing Grounding agent and return an initialized provider.

        Use this convenience factory for one-time setup or experiments. In
        production, create the agent once and then switch to the regular
        constructor with the persisted ``agent_id`` because agent creation
        is slower and fixes several search settings permanently.

        Args:
            project_endpoint: Azure AI Foundry project endpoint.
            bing_connection_id: Connected-resource ID for the Bing
                Grounding connection inside the Foundry project.
            model: Model deployment used by the Bing agent itself.
            instructions: Agent instructions stored with the agent at
                creation time.
            agent_name: Human-readable agent name to create.
            freshness: Optional Bing freshness mode fixed at agent-creation
                time rather than per request.
            market: Optional Bing market fixed at agent-creation time.
            set_lang: Optional language preference fixed at agent-creation
                time.
            count: Number of Bing results the agent should ground on.
            credential: Optional prebuilt Azure credential object.
            tenant_id: Optional Entra tenant ID for automatic Service
                Principal auth.
            client_id: Optional Entra client ID for automatic Service
                Principal auth.
            client_secret: Optional Entra client secret for automatic
                Service Principal auth.
            timeout: Default timeout budget in seconds for later runtime
                searches.

        Returns:
            AzureFoundryBingSearch: Initialized provider pointing at the
            newly created agent.
        """
        resolved_credential = cls._resolve_credential(
            credential=credential,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

        client = AIProjectClient(
            endpoint=project_endpoint,
            credential=resolved_credential,
        )

        # Build Bing tool -- parameters are fixed at agent creation time
        bing_kwargs: dict[str, Any] = {"connection_id": bing_connection_id, "count": count}
        if market:
            bing_kwargs["market"] = market
        if set_lang:
            bing_kwargs["set_lang"] = set_lang
        if freshness:
            bing_kwargs["freshness"] = freshness

        bing_tool = BingGroundingTool(**bing_kwargs)

        agent = client.agents.create_agent(
            model=model,
            name=agent_name,
            instructions=instructions,
            tools=bing_tool.definitions,
            description="Inqtrix Bing-Grounding Search Agent",
        )
        log.info("Azure Foundry Bing Agent erstellt: %s", agent.id)

        return cls(
            project_endpoint=project_endpoint,
            agent_id=agent.id,
            credential=resolved_credential,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Query helpers (reuse BraveSearch pattern)
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
        """Execute a search through the Azure Foundry Bing agent.

        Use this method when the runtime should delegate web search to a
        Foundry agent with Bing Grounding. Several generic Inqtrix search
        hints are not first-class Bing runtime parameters here because the
        real Bing settings are fixed when the agent is created.

        Args:
            query: User-facing search query text.
            search_context_size: Unsupported at runtime for this backend.
                The value is ignored because Bing result count is fixed when
                the agent is created.
            recency_filter: Optional recency hint. The provider converts it
                into additional instructions for the agent, but the result
                is best effort rather than guaranteed filtering.
            language_filter: Optional language hint list. The provider uses
                it only for best-effort runtime instructions.
            domain_filter: Optional domain include/exclude list. The
                provider injects ``site:`` and ``-site:`` operators into the
                query because Bing grounding does not expose a dedicated
                domain-filter parameter here.
            search_mode: Unsupported at runtime for this backend. Passing a
                value has no effect.
            return_related: Unsupported for this backend. Passing ``True``
                does not populate ``related_questions``.
            deadline: Optional absolute monotonic deadline for the full
                agent run.

        Returns:
            dict[str, Any]: Normalized result with ``answer`` from the
            agent message text, ``citations`` from grounding annotations or
            URL fallback extraction, ``related_questions`` as an empty
            list, and zero token counts because the Foundry agent path does
            not expose compatible usage metrics here.

        Raises:
            AgentTimeout: If the global deadline has already elapsed.
            AgentRateLimited: If the backend surfaces a fatal rate limit.
        """
        self._clear_nonfatal_notice()

        if deadline is not None:
            _check_deadline(deadline)

        # Parameters without Bing equivalents -- silently ignored
        _ = search_context_size
        _ = search_mode
        _ = return_related

        effective_query = _apply_domain_filters(query, domain_filter)
        additional = _build_recency_language_hints(
            recency_filter, language_filter
        )

        timeout = _bounded_timeout(self._timeout, deadline)

        try:
            return self._execute_search(effective_query, additional, timeout)
        except AgentTimeout:
            raise
        except AgentRateLimited:
            raise
        except Exception as exc:
            log.error(
                "Azure-Foundry-Bing-Suche fehlgeschlagen fuer '%s': %s",
                query[:80],
                exc,
            )
            self._set_nonfatal_notice(
                f"Azure-Foundry-Bing-Suche fehlgeschlagen fuer Query "
                f"'{query[:80]}': {exc}; leeres Ergebnis wird weiterverwendet."
            )
            return dict(_EMPTY_RESULT)

    def _execute_search(
        self,
        query: str,
        additional_instructions: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        """Run Thread+Message+Run in one call and extract the response."""
        # Build thread with the user query as initial message
        thread = AgentThreadCreationOptions(
            messages=[ThreadMessageOptions(role="user", content=query)]
        )

        run_kwargs: dict[str, Any] = {
            "agent_id": self._agent_id,
            "thread": thread,
        }
        if additional_instructions:
            run_kwargs["instructions"] = additional_instructions

        run = self._client.agents.create_thread_and_process_run(**run_kwargs)

        if run.status != "completed":
            error_msg = getattr(run, "last_error", None) or run.status
            raise AzureFoundryBingAPIError(
                agent_id=self._agent_id,
                message=f"Run nicht erfolgreich: {error_msg}",
            )

        # Extract answer and citations from the thread
        messages = self._client.agents.messages.list(thread_id=run.thread_id)
        return self._parse_agent_response(messages)

    def _parse_agent_response(self, messages: Any) -> dict[str, Any]:
        """Extract answer text and citation URLs from the agent response."""
        answer_parts: list[str] = []
        citations: list[str] = []
        seen_urls: set[str] = set()

        for msg in messages:
            if msg.role != MessageRole.AGENT:
                continue
            for item in msg.content:
                if not hasattr(item, "text"):
                    continue
                answer_parts.append(item.text.value)

                # Structured annotations (Bing grounding URLs)
                for ann in getattr(item.text, "annotations", None) or []:
                    url = getattr(ann, "url", None)
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        citations.append(url)

        answer = "\n\n".join(answer_parts)

        # Fallback: extract URLs from answer text if no annotations found
        if not citations and answer:
            citations = extract_urls(answer)

        return {
            "answer": answer,
            "citations": citations,
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

    def is_available(self) -> bool:
        """Report whether the provider is configured to attempt requests.

        Returns:
            bool: ``True`` when both the agent ID and project endpoint are
            present, otherwise ``False``.
        """
        return bool(self._agent_id and self._project_endpoint)
