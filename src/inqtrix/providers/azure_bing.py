"""Azure Foundry Bing Search adapter for the SearchProvider interface.

Uses the project-scoped OpenAI Responses API to invoke a pre-created
Azure AI Foundry agent that has Bing Grounding attached. The modern
runtime path references the agent by name and optional version through
``extra_body={"agent_reference": ...}``.

For backwards compatibility the provider still accepts a legacy
``agent_id``. When credential-based auth is available it resolves that
ID to an agent name through ``AIProjectClient.agents.get_agent()`` and
then uses the modern Responses API path. When ``execution_mode`` is
``"auto"`` (default), a Responses ``HTTP 404`` for ``agent_reference``
falls back to the older Thread/Run path if an ``agent_id`` is known,
and a second Responses attempt omits ``version`` when that version was
only auto-filled from ``get_agent()`` (not explicitly set by the
caller). Only when resolution is not possible does the provider use
Thread/Run immediately (no ``agent_name``).

Search parameters like freshness, market, and count are still fixed at
agent creation time. The provider bridges the gap to the SearchProvider
interface by injecting ``site:`` operators into the query text for
domain filtering and passing recency/language as best-effort prompt
hints.

Install the required SDKs::

    uv sync
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from openai import APIStatusError, OpenAI, OpenAIError, RateLimitError

from inqtrix.exceptions import AgentRateLimited, AgentTimeout, AzureFoundryBingAPIError
from inqtrix.providers._azure_common import extract_azure_api_error_details
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
# Guarded SDK imports
# ---------------------------------------------------------------------------
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    AgentThreadCreationOptions,
    BingGroundingTool,
    MessageRole,
    ThreadMessageOptions,
)
from azure.identity import ClientSecretCredential, DefaultAzureCredential

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Azure AI Foundry OAuth scope used to mint bearer tokens for the Foundry
# project endpoint. This scope serves the Responses runtime (modern path)
# and the legacy AIProjectClient (legacy path); both surfaces accept the
# same audience. Tokens minted under this scope live ~60-75 min — keep
# the provider as a per-process singleton (not per-request) so the same
# bearer is reused across calls; new bearers are minted automatically by
# the credential when the cached one approaches expiry.
_AUTH_SCOPE = "https://ai.azure.com/.default"

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


class AzureFoundryBingSearch(_NonFatalNoticeMixin, SearchProvider):
    """Query the web via an Azure Foundry agent with Bing Grounding.

    Use this provider when search should run through a pre-created Azure
    AI Foundry agent that has the Bing Grounding tool attached. The
    modern runtime path uses the project-scoped Responses API with an
    ``agent_reference``. Existing callers that only know the legacy
    opaque ``agent_id`` are still supported when the provider can resolve
    that ID back to an agent name.
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
        agent_name: str = "",
        agent_version: str = "",
        agent_id: str = "",
        api_key: str | None = None,
        credential: Any | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = 60.0,
        tool_choice: str | None = "required",
        execution_mode: Literal["auto", "responses", "legacy"] = "auto",
    ) -> None:
        """Initialize the Azure Foundry Bing search provider.

        Use ``agent_name`` and optional ``agent_version`` for the modern
        Foundry runtime path. The legacy ``agent_id`` remains accepted for
        compatibility; when credential-based auth is available the
        provider resolves it to a name and then uses the Responses API.

        Args:
            project_endpoint: Azure AI Foundry project endpoint.
            agent_name: Name of the existing Bing Grounding agent.
            agent_version: Optional version override for the agent
                reference. Leave empty to use the default/latest version.
            agent_id: Optional legacy opaque agent ID. Prefer
                ``agent_name`` for new code.
            api_key: Optional static Foundry project API key.
            credential: Optional prebuilt Azure credential object.
            tenant_id: Optional Entra tenant ID used only when building a
                ``ClientSecretCredential`` internally.
            client_id: Optional Entra client ID used only when building a
                ``ClientSecretCredential`` internally.
            client_secret: Optional Entra client secret used only when
                building a ``ClientSecretCredential`` internally.
            timeout: Default timeout budget in seconds for one search call.
            tool_choice: Optional Responses API ``tool_choice`` setting.
                Defaults to ``"required"`` so the search agent uses its
                Bing tool deterministically.
            execution_mode: ``"auto"`` (default) tries the Responses API
                when an ``agent_name`` is available and on ``HTTP 404``
                may retry without an auto-resolved ``agent_version`` or
                fall back to Thread/Run when ``agent_id`` is set.
                ``"responses"`` disables those fallbacks. ``"legacy"``
                forces Thread/Run and requires ``agent_id`` plus
                credential-based auth (no API-key-only mode).

        Raises:
            ValueError: If ``project_endpoint`` is empty, if neither
                ``agent_name`` nor ``agent_id`` is supplied, if
                ``agent_id``-only setup is combined with API-key auth, or
                if ``tool_choice`` is invalid.
        """
        if not project_endpoint:
            raise ValueError("project_endpoint ist erforderlich")
        if not agent_name and not agent_id:
            raise ValueError("agent_name oder agent_id ist erforderlich")
        if tool_choice not in {None, "auto", "required", "none"}:
            raise ValueError("tool_choice must be one of auto, required, none, or None.")
        if execution_mode not in {"auto", "responses", "legacy"}:
            raise ValueError(
                "execution_mode must be one of auto, responses, legacy."
            )
        if execution_mode == "legacy" and not str(agent_id or "").strip():
            raise ValueError("execution_mode='legacy' requires agent_id")
        if execution_mode == "legacy" and api_key:
            raise ValueError(
                "execution_mode='legacy' requires credential-based auth; "
                "Thread/Run is not available with api_key-only mode."
            )

        self._project_endpoint = project_endpoint.rstrip("/")
        self._agent_name = agent_name.strip()
        self._agent_version = str(agent_version or "").strip()
        self._agent_id = agent_id.strip()
        self._agent_version_explicit = bool(str(agent_version or "").strip())
        self._execution_mode = execution_mode
        self._timeout = timeout
        self._tool_choice = tool_choice
        self._project_client: AIProjectClient | None = None

        if api_key and not self._agent_name and self._agent_id:
            raise ValueError(
                "agent_name ist erforderlich bei API-Key-Auth; "
                "agent_id-only Aufloesung benoetigt Azure-Credentials."
            )

        base_url = f"{self._project_endpoint}/openai/v1/"

        if api_key:
            self._credential = None
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=_SDK_MAX_RETRIES,
            )
        else:
            self._credential = self._resolve_credential(
                credential=credential,
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
            self._project_client = AIProjectClient(
                endpoint=self._project_endpoint,
                credential=self._credential,
            )
            token = self._credential.get_token(_AUTH_SCOPE).token
            self._client = OpenAI(
                base_url=base_url,
                api_key=token,
                timeout=timeout,
                max_retries=_SDK_MAX_RETRIES,
            )
            if self._execution_mode != "legacy":
                self._resolve_agent_reference_from_id()

    @staticmethod
    def _resolve_credential(
        *,
        credential: Any | None,
        tenant_id: str | None,
        client_id: str | None,
        client_secret: str | None,
    ) -> Any:
        """Build or return an Azure credential object.

        Kept as a local staticmethod (duplicated with
        ``_azure_common.resolve_azure_credential``) because the test
        suite patches ``inqtrix.providers.azure_bing.ClientSecretCredential``
        and ``inqtrix.providers.azure_bing.DefaultAzureCredential``
        through the module namespace — a shared helper would bypass
        those patches.
        """
        if credential is not None:
            return credential
        if tenant_id and client_id and client_secret:
            return ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
        return DefaultAzureCredential()

    _extract_api_error_details = staticmethod(extract_azure_api_error_details)

    def _agent_identifier(self) -> str:
        """Return the most useful identifier for logs and errors."""
        return self._agent_name or self._agent_id

    def _resolve_agent_reference_from_id(self) -> None:
        """Resolve legacy ``agent_id`` to ``agent_name`` when possible."""
        if self._agent_name or not self._agent_id or self._project_client is None:
            return

        try:
            agent = self._project_client.agents.get_agent(self._agent_id)
        except Exception as exc:
            log.warning(
                "Konnte Azure-Foundry-Bing agent_id '%s' nicht zu agent_name aufloesen: %s",
                self._agent_id,
                exc,
            )
            return

        resolved_name = str(getattr(agent, "name", "") or "").strip()
        if resolved_name:
            self._agent_name = resolved_name
        resolved_version = str(getattr(agent, "version", "") or "").strip()
        if resolved_version and not self._agent_version:
            self._agent_version = resolved_version

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
        api_key: str | None = None,
        credential: Any | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = 60.0,
        tool_choice: str | None = "required",
    ) -> "AzureFoundryBingSearch":
        """Create a Bing Grounding agent and return an initialized provider.

        The helper prefers the newer versioned agent-creation API when it
        is available in the installed SDK. On older SDK versions it falls
        back to the classic ``create_agent`` method but still returns a
        provider configured for the newer runtime path whenever an agent
        name is available.
        """
        if api_key:
            raise ValueError(
                "create_agent() benoetigt Azure-Credentials; API-Key-Auth wird "
                "nur fuer bereits vorhandene agent_name-basierte Laufzeitaufrufe unterstuetzt."
            )

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

        if hasattr(client.agents, "create_version"):
            try:
                from azure.ai.projects.models import PromptAgentDefinition
                from azure.ai.agents.models import (
                    BingGroundingSearchConfiguration,
                    BingGroundingSearchToolParameters,
                )

                configuration_kwargs: dict[str, Any] = {
                    "project_connection_id": bing_connection_id,
                }
                if market:
                    configuration_kwargs["market"] = market
                if set_lang:
                    configuration_kwargs["set_lang"] = set_lang
                if freshness:
                    configuration_kwargs["freshness"] = freshness
                if count:
                    configuration_kwargs["count"] = count

                bing_tool = BingGroundingTool(
                    bing_grounding=BingGroundingSearchToolParameters(
                        search_configurations=[
                            BingGroundingSearchConfiguration(**configuration_kwargs)
                        ]
                    )
                )
                agent = client.agents.create_version(
                    agent_name=agent_name,
                    definition=PromptAgentDefinition(
                        model=model,
                        instructions=instructions,
                        tools=[bing_tool],
                    ),
                )
                log.info(
                    "Azure Foundry Bing Agent-Version erstellt: %s v%s",
                    getattr(agent, "name", agent_name),
                    getattr(agent, "version", ""),
                )
                return cls(
                    project_endpoint=project_endpoint,
                    agent_name=str(getattr(agent, "name", agent_name) or agent_name),
                    agent_version=str(getattr(agent, "version", "") or ""),
                    agent_id=str(getattr(agent, "id", "") or ""),
                    credential=resolved_credential,
                    timeout=timeout,
                    tool_choice=tool_choice,
                    execution_mode="auto",
                )
            except ImportError:
                log.info(
                    "azure-ai-projects ohne PromptAgentDefinition erkannt; "
                    "falle auf legacy create_agent() zurueck."
                )

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
        log.info("Azure Foundry Bing Agent erstellt: %s", getattr(agent, "id", ""))

        return cls(
            project_endpoint=project_endpoint,
            agent_name=str(getattr(agent, "name", agent_name) or agent_name),
            agent_version=str(getattr(agent, "version", "") or ""),
            agent_id=str(getattr(agent, "id", "") or ""),
            credential=resolved_credential,
            timeout=timeout,
            tool_choice=tool_choice,
            execution_mode="auto",
        )

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
        """Execute one Bing-grounded search through the Foundry agent.

        Triggers the pre-created Foundry agent's Bing tool via the
        modern Responses runtime (when an ``agent_name`` is available)
        or the legacy AIProjectClient path (when only ``agent_id`` was
        supplied and credential-based auth could not resolve a name).
        Citations are returned verbatim when the agent surfaces them;
        otherwise the provider falls back to URL extraction from the
        answer text.

        Args:
            query: Natural-language search query. Forwarded to the
                agent as the user message; pre-processed to embed any
                ``domain_filter`` as ``site:`` operators because the
                Foundry Bing tool does not expose a structured
                domain-restriction field.
            search_context_size: ABC-compatibility parameter — ignored
                by this provider because the Foundry Bing tool does
                not expose a context-size knob. Filtered out by the
                search node via ``supported_search_parameters`` before
                this call, so production code never sets it.
            recency_filter: Optional recency hint (``"day"``, ``"week"``,
                ``"month"``, ``"year"``). Translated to a German
                instruction appended to ``additional_instructions`` for
                the agent (the Foundry Bing tool has no native recency
                parameter).
            language_filter: Optional list of language codes.
                Currently only the first entry is used and translated
                to a German instruction (e.g. ``"Antworte auf de"``);
                multi-language hints are not supported by Bing tool
                grounding.
            domain_filter: Optional list of domains. Embedded into the
                effective query as ``site:domain1 OR site:domain2``;
                Bing's site-operator support honours this best-effort.
            search_mode: ABC-compatibility parameter — ignored by this
                provider (no academic mode equivalent on Bing tool).
            return_related: ABC-compatibility parameter — ignored;
                Bing tool grounding does not surface related questions.
            deadline: Optional absolute monotonic deadline. The
                per-call timeout is clamped to the remaining budget;
                exceeding the deadline raises :class:`AgentTimeout`.

        Returns:
            Search-result dict with keys ``answer`` (model-generated
            text with inline grounding), ``citations`` (list of source
            URLs), ``related_questions`` (always empty for this
            provider), ``_prompt_tokens`` and ``_completion_tokens``
            (always 0 — Foundry Bing does not surface token counts to
            the agent invocation). On any non-deadline / non-rate-
            limit failure the provider returns ``_EMPTY_RESULT``
            (empty answer, no citations) and stores a non-fatal notice
            on itself for the search node to surface.

        Raises:
            AgentTimeout: When the absolute run deadline has elapsed.
            AgentRateLimited: When the underlying SDK escalates a
                fatal rate limit (rare for Foundry Bing).
        """
        self._clear_nonfatal_notice()

        if deadline is not None:
            _check_deadline(deadline)

        # Unsupported hints (search_context_size, search_mode, return_related)
        # are filtered out by the search node via supported_search_parameters;
        # they remain in the signature only for ABC compatibility.

        effective_query = _apply_domain_filters(query, domain_filter)
        additional = _build_recency_language_hints(
            recency_filter, language_filter
        )

        timeout = _bounded_timeout(self._timeout, deadline)

        try:
            result = self._execute_search(effective_query, additional, timeout)
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

        if not result.get("answer"):
            self._set_nonfatal_notice(
                f"Azure-Foundry-Bing-Suche fuer '{query[:80]}' lieferte keine Textantwort"
            )
        return result

    def _execute_search(
        self,
        query: str,
        additional_instructions: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        """Execute via Responses API, falling back to legacy Thread/Run."""
        if self._execution_mode == "legacy":
            if self._agent_id and self._project_client is not None:
                log.info(
                    "Azure-Foundry-Bing nutzt legacy Thread/Run (execution_mode=legacy) "
                    "fuer agent_id '%s'",
                    self._agent_id,
                )
                return self._execute_legacy_search(
                    query,
                    additional_instructions,
                    timeout,
                )
            raise AzureFoundryBingAPIError(
                agent_id=self._agent_identifier(),
                message="execution_mode='legacy' benoetigt agent_id und AIProjectClient.",
            )

        self._resolve_agent_reference_from_id()

        if not self._agent_name:
            if self._agent_id and self._project_client is not None:
                log.info(
                    "Azure-Foundry-Bing nutzt legacy Thread/Run-Fallback fuer agent_id '%s'",
                    self._agent_id,
                )
                return self._execute_legacy_search(
                    query,
                    additional_instructions,
                    timeout,
                )

            raise AzureFoundryBingAPIError(
                agent_id=self._agent_identifier(),
                message="Keine verwendbare Agent-Referenz fuer Azure Foundry Bing konfiguriert.",
            )

        if self._execution_mode == "responses":
            return self._execute_responses_search(
                query,
                additional_instructions,
                timeout,
            )

        try:
            return self._execute_responses_search(
                query,
                additional_instructions,
                timeout,
            )
        except AzureFoundryBingAPIError as first_exc:
            if first_exc.status_code != 404:
                raise

            if not self._agent_version_explicit and self._agent_version:
                saved_version = self._agent_version
                self._agent_version = ""
                try:
                    return self._execute_responses_search(
                        query,
                        additional_instructions,
                        timeout,
                    )
                except AzureFoundryBingAPIError as second_exc:
                    if second_exc.status_code != 404:
                        raise
                finally:
                    self._agent_version = saved_version

            if self._agent_id and self._project_client is not None:
                log.info(
                    "Azure-Foundry-Bing: Responses HTTP 404 fuer agent_reference, "
                    "nutze legacy Thread/Run fuer agent_id '%s'",
                    self._agent_id,
                )
                return self._execute_legacy_search(
                    query,
                    additional_instructions,
                    timeout,
                )

            raise

    def _execute_responses_search(
        self,
        query: str,
        additional_instructions: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        """Call ``client.responses.create`` with an agent reference."""
        if additional_instructions:
            user_input = f"{additional_instructions}\n\n{query}"
        else:
            user_input = query

        agent_ref: dict[str, str] = {
            "name": self._agent_name,
            "type": "agent_reference",
        }
        if self._agent_version:
            agent_ref["version"] = self._agent_version

        request_kwargs: dict[str, Any] = {
            "input": user_input,
            "extra_body": {"agent_reference": agent_ref},
            "timeout": timeout,
        }
        if self._tool_choice is not None:
            request_kwargs["tool_choice"] = self._tool_choice

        try:
            response = self._client.responses.create(**request_kwargs)
        except RateLimitError as exc:
            raise AgentRateLimited(self._agent_identifier(), exc) from exc
        except APIStatusError as exc:
            details = self._extract_api_error_details(exc)
            if details["status_code"] == 429:
                raise AgentRateLimited(self._agent_identifier(), exc) from exc
            raise AzureFoundryBingAPIError(
                agent_id=self._agent_identifier(),
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
                    f"Azure-Foundry-Bing Timeout fuer '{query[:80]}'"
                ) from exc
            raise AzureFoundryBingAPIError(
                agent_id=self._agent_identifier(),
                message=str(exc),
                original=exc,
            ) from exc

        return self._parse_response(response)

    def _execute_legacy_search(
        self,
        query: str,
        additional_instructions: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        """Run the legacy Thread/Run flow for agent_id-only compatibility."""
        del timeout

        if self._project_client is None:
            raise AzureFoundryBingAPIError(
                agent_id=self._agent_identifier(),
                message="Legacy-Fallback benoetigt einen AIProjectClient.",
            )

        thread = AgentThreadCreationOptions(
            messages=[ThreadMessageOptions(role="user", content=query)]
        )

        run_kwargs: dict[str, Any] = {
            "agent_id": self._agent_id,
            "thread": thread,
        }
        if additional_instructions:
            run_kwargs["instructions"] = additional_instructions

        run = self._project_client.agents.create_thread_and_process_run(**run_kwargs)

        if run.status != "completed":
            error_msg = getattr(run, "last_error", None) or run.status
            raise AzureFoundryBingAPIError(
                agent_id=self._agent_identifier(),
                message=f"Run nicht erfolgreich: {error_msg}",
            )

        messages = self._project_client.agents.messages.list(thread_id=run.thread_id)
        return self._parse_agent_response(messages)

    @staticmethod
    def _parse_response(response: Any) -> dict[str, Any]:
        """Extract answer text and citations from a Responses API reply."""
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

    def _parse_agent_response(self, messages: Any) -> dict[str, Any]:
        """Extract answer text and citation URLs from the legacy agent response."""
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

                for ann in getattr(item.text, "annotations", None) or []:
                    url = getattr(ann, "url", None)
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        citations.append(url)

        answer = "\n\n".join(answer_parts)

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
        """Report whether the provider has enough config to call Foundry.

        Configuration here means: a non-empty ``project_endpoint`` is
        set AND at least one of ``agent_name`` or ``agent_id`` is set.
        Auth resolution (api-key / credential / Service Principal /
        DefaultAzureCredential) is validated lazily on the first
        ``search()`` call; reachability of the Foundry project is
        likewise not pre-validated.

        Returns:
            ``True`` when both endpoint and at least one agent
            identifier are set, otherwise ``False``.
        """
        return bool(self._project_endpoint and (self._agent_name or self._agent_id))

    @property
    def search_model(self) -> str:
        """Foundry Bing-grounding agent identifier shown to operators.

        Format: ``"foundry-bing:<agent_name_or_id>@<version_or_latest>"``
        (e.g. ``"foundry-bing:my-bing-agent@v2"``). When constructed
        only with an ``agent_id`` (legacy path) the id is used in
        place of the name. The ``foundry-bing`` prefix disambiguates
        from the Foundry web-search variant (``foundry-web:``).
        """
        identifier = self._agent_name or self._agent_id or "unknown"
        version = self._agent_version or "latest"
        return f"foundry-bing:{identifier}@{version}"
