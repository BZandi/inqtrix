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
from inqtrix.providers import SearchProvider, _NonFatalNoticeMixin, _bounded_timeout, _check_deadline
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
    """Query the web via an Azure Foundry Agent with Bing Grounding.

    The agent must already exist.  Pass its ``agent_id`` to the constructor.
    Use the :meth:`create_agent` classmethod to create one programmatically.
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
        """Create a Foundry Bing agent and return an initialised provider.

        This is a convenience factory.  Agent creation takes 2-5 seconds.
        For production use, create the agent once and pass its ID to the
        constructor.
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

    @staticmethod
    def _apply_domain_filters(query: str, domain_filter: list[str] | None) -> str:
        """Inject ``site:`` / ``-site:`` operators into the query string."""
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

    @staticmethod
    def _build_additional_instructions(
        recency_filter: str | None,
        language_filter: list[str] | None,
    ) -> str | None:
        """Build best-effort hints for the agent LLM (non-deterministic)."""
        parts: list[str] = []

        recency = (recency_filter or "").strip().lower()
        if recency == "day":
            parts.append("Fokussiere dich auf Ergebnisse der letzten 24 Stunden.")
        elif recency == "week":
            parts.append("Fokussiere dich auf Ergebnisse der letzten Woche.")
        elif recency == "month":
            parts.append("Fokussiere dich auf Ergebnisse des letzten Monats.")
        elif recency == "year":
            parts.append("Fokussiere dich auf Ergebnisse des letzten Jahres.")

        if language_filter:
            lang = language_filter[0]
            parts.append(
                f"Antworte auf {lang} und bevorzuge Quellen in dieser Sprache."
            )

        return " ".join(parts) if parts else None

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
        """Execute a web search via Azure Foundry Bing Grounding agent."""
        self._clear_nonfatal_notice()

        if deadline is not None:
            _check_deadline(deadline)

        # Parameters without Bing equivalents -- silently ignored
        _ = search_context_size
        _ = search_mode
        _ = return_related

        effective_query = self._apply_domain_filters(query, domain_filter)
        additional = self._build_additional_instructions(
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
        """Return True when the provider is ready to serve requests."""
        return bool(self._agent_id and self._project_endpoint)
