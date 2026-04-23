"""Bridge between YAML configuration and existing Settings/Providers.

Translates an :class:`InqtrixConfig` (from YAML) into the existing
:class:`Settings` and :class:`ProviderContext` objects that the rest
of the application depends on.

Key components:

* :class:`ModelResolver` — resolves named models to ``(OpenAI client, model_id)``
  pairs, caching clients per unique ``(base_url, api_key)``.
* :func:`config_to_settings` — produces a :class:`Settings` from YAML overrides.
* :func:`create_providers_from_config` — creates a :class:`ProviderContext`
  with a :class:`MultiClientLLMProvider` that routes each call to the correct
  provider endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from inqtrix.config import AgentConfig, InqtrixConfig
from inqtrix.providers.azure_openai_web_search import AzureOpenAIWebSearch
from inqtrix.providers.azure_web_search import AzureFoundryWebSearch
from inqtrix.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderContext,
    SearchProvider,
    SummarizeOptions,
    _NonFatalNoticeMixin,
    _bounded_timeout,
    _check_deadline,
    _normalize_completion_response,
)
from inqtrix.providers.perplexity import PerplexitySearch
from inqtrix.constants import REASONING_TIMEOUT
from inqtrix.exceptions import AgentRateLimited, AgentTimeout
from inqtrix.prompts import SUMMARIZE_PROMPT
from inqtrix.settings import (
    AgentSettings,
    ModelSettings,
    ServerSettings,
    Settings,
)
from inqtrix.state import track_tokens

log = logging.getLogger("inqtrix")

_SEARCH_PROVIDER_ALIASES = {
    "perplexity": "perplexity",
    "azure_openai_web_search": "azure_openai_web_search",
    "azure_foundry_web_search": "azure_foundry_web_search",
    "azure_web_search": "azure_foundry_web_search",
}


# ------------------------------------------------------------------ #
# Resolved model
# ------------------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A model resolved to its provider client and actual model ID."""

    client: OpenAI
    model_id: str
    params: dict[str, Any]


# ------------------------------------------------------------------ #
# ModelResolver
# ------------------------------------------------------------------ #


class ModelResolver:
    """Resolve named models to ``(OpenAI client, model_id)`` tuples.

    Caches :class:`OpenAI` client instances per unique
    ``(base_url, api_key)`` pair so that multiple models on the same
    provider share one HTTP connection pool.
    """

    def __init__(self, config: InqtrixConfig) -> None:
        self._config = config
        self._clients: dict[tuple[str, str], OpenAI] = {}

    def _get_client(self, base_url: str, api_key: str) -> OpenAI:
        """Return a cached or new OpenAI client for the given endpoint."""
        key = (base_url, api_key)
        if key not in self._clients:
            self._clients[key] = OpenAI(base_url=base_url, api_key=api_key)
        return self._clients[key]

    def resolve(self, model_name: str) -> ResolvedModel:
        """Resolve *model_name* to a :class:`ResolvedModel`.

        Raises :class:`KeyError` if the model or provider is not defined.
        """
        model_cfg = self._config.models[model_name]
        provider_cfg = self._config.providers[model_cfg.provider]
        client = self._get_client(provider_cfg.base_url, provider_cfg.api_key)
        return ResolvedModel(
            client=client,
            model_id=model_cfg.model_id,
            params=model_cfg.params,
        )

    def has_model(self, model_name: str) -> bool:
        """Return True if *model_name* is defined in the config."""
        return model_name in self._config.models


# ------------------------------------------------------------------ #
# MultiClientLLMProvider
# ------------------------------------------------------------------ #


class MultiClientLLMProvider(_NonFatalNoticeMixin, LLMProvider):
    """LLM provider that routes each call to the correct API client.

    When reasoning runs on OpenAI and classify on Anthropic via LiteLLM,
    ``complete(model="gpt-4o")`` uses the OpenAI client while
    ``complete(model="claude-haiku")`` uses the Anthropic client.

    The provider uses named model references from the YAML config.
    The ``model`` parameter in :meth:`complete` is a *named model key*
    (e.g. ``"gpt-4o"``) that gets resolved via :class:`ModelResolver`
    to the actual ``(client, model_id)``.
    """

    def __init__(
        self,
        resolver: ModelResolver,
        models: ModelSettings,
        agent_settings: AgentSettings,
    ) -> None:
        self._resolver = resolver
        self._models = models
        self._agent_settings = agent_settings

    @property
    def models(self) -> ModelSettings:
        """Expose model configuration for node access."""
        return self._models

    @staticmethod
    def _request_kwargs(
        *,
        resolved: ResolvedModel,
        messages: list[dict[str, str]],
        timeout: float,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build completion kwargs including model-level parameters."""
        request_kwargs: dict[str, Any] = {
            "model": resolved.model_id,
            "messages": messages,
            "timeout": timeout,
            "stream": False,
        }
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max_output_tokens
        for key, value in resolved.params.items():
            if key in {"model", "messages", "timeout", "stream", "max_tokens"}:
                continue
            if key == "extra_body" and isinstance(value, dict):
                request_kwargs["extra_body"] = dict(value)
                continue
            request_kwargs[key] = value
        return request_kwargs

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        return self.complete_with_metadata(
            prompt,
            system=system,
            model=model,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            state=state,
            deadline=deadline,
        ).content

    def complete_with_metadata(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        if deadline is not None:
            _check_deadline(deadline)

        # Determine which named model to use
        use_model_name = model or self._models.reasoning_model

        # Resolve to client + model_id
        resolved = self._resolver.resolve(use_model_name)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            from openai import OpenAIError, RateLimitError, APIStatusError

            r = resolved.client.chat.completions.create(
                **self._request_kwargs(
                    resolved=resolved,
                    messages=messages,
                    timeout=_bounded_timeout(timeout, deadline),
                    max_output_tokens=max_output_tokens,
                ),
            )
            normalized = _normalize_completion_response(r)
            if state is not None:
                track_tokens(state, normalized)
            return LLMResponse(
                content=normalized.content,
                prompt_tokens=normalized.prompt_tokens,
                completion_tokens=normalized.completion_tokens,
                model=resolved.model_id,
            )
        except RateLimitError as e:
            log.error("FATAL Rate-Limit (%s): %s", resolved.model_id, e)
            raise AgentRateLimited(resolved.model_id, e)
        except APIStatusError as e:
            if e.status_code == 429:
                log.error("FATAL Rate-Limit (%s): %s", resolved.model_id, e)
                raise AgentRateLimited(resolved.model_id, e)
            log.error("LLM call failed (%s): %s", resolved.model_id, e)
            raise
        except OpenAIError as e:
            log.error("LLM call failed (%s): %s", resolved.model_id, e)
            raise

    def summarize_parallel(
        self,
        text: str,
        deadline: float | None = None,
        options: SummarizeOptions | None = None,
    ) -> tuple[str, int, int]:
        self._clear_nonfatal_notice()
        if not text.strip():
            return ("", 0, 0)
        if deadline is not None:
            _check_deadline(deadline)

        summarize_model = self._models.effective_summarize_model
        resolved = self._resolver.resolve(summarize_model)
        prompt_template = options.prompt_template if options and options.prompt_template else SUMMARIZE_PROMPT
        input_char_limit = options.input_char_limit if options and options.input_char_limit else 6000
        fallback_char_limit = (
            options.fallback_char_limit if options and options.fallback_char_limit else 800
        )
        prompt = f"{prompt_template}{text[:input_char_limit]}"

        try:
            from openai import OpenAIError

            r = resolved.client.chat.completions.create(
                **self._request_kwargs(
                    resolved=resolved,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=_bounded_timeout(
                        self._agent_settings.summarize_timeout,
                        deadline,
                    ),
                    max_output_tokens=(
                        options.max_output_tokens
                        if options and options.max_output_tokens is not None
                        else None
                    ),
                ),
            )
            normalized = _normalize_completion_response(r)
            return (
                normalized.content,
                normalized.prompt_tokens,
                normalized.completion_tokens,
            )
        except AgentRateLimited:
            raise
        except (OpenAIError, AgentTimeout):
            self._set_nonfatal_notice(
                f"Zusammenfassung via {summarize_model} fehlgeschlagen; Fallback auf Rohtext."
            )
            return (text[:fallback_char_limit], 0, 0)

    def is_available(self) -> bool:
        required_models = {
            self._models.reasoning_model,
            self._models.effective_classify_model,
            self._models.effective_summarize_model,
            self._models.effective_evaluate_model,
        }
        required_models.discard("")
        return bool(required_models) and all(
            self._resolver.has_model(model_name) for model_name in required_models
        )


# ------------------------------------------------------------------ #
# Config → Settings
# ------------------------------------------------------------------ #


def _field_alias(cls: type, field_name: str) -> str:
    """Return the alias for a field if it has one, else the field name.

    Keeping overrides alias-aware ensures env-var names and Python-side
    field names stay interchangeable across YAML and library paths.
    """
    info = cls.model_fields.get(field_name)
    if info and info.alias:
        return info.alias
    return field_name


def _apply_agent_overrides(
    base: AgentSettings, overrides: Any
) -> AgentSettings:
    """Apply non-None fields from AgentBehaviorConfig onto AgentSettings."""
    data = base.model_dump(by_alias=True)
    explicit_fields = set(base.model_fields_set)
    for field_name in type(overrides).model_fields:
        val = getattr(overrides, field_name)
        if val is not None:
            alias = _field_alias(AgentSettings, field_name)
            data[alias] = val
            explicit_fields.add(field_name)
    return AgentSettings(**data).with_report_profile_defaults(
        explicit_fields=explicit_fields,
    )


def _apply_server_overrides(
    base: ServerSettings, overrides: Any
) -> ServerSettings:
    """Apply non-None fields from ServerConfig onto ServerSettings."""
    data = base.model_dump(by_alias=True)
    for field_name in type(overrides).model_fields:
        val = getattr(overrides, field_name)
        if val is not None:
            alias = _field_alias(ServerSettings, field_name)
            data[alias] = val
    return ServerSettings(**data)


def config_to_settings(
    config: InqtrixConfig, agent_name: str = "default"
) -> Settings:
    """Convert an :class:`InqtrixConfig` into a :class:`Settings` instance.

    Role→model mappings from the YAML agent config are translated into
    the :class:`ModelSettings` fields (reasoning_model, search_model, etc.).
    Server and agent behavior overrides are applied on top of the
    Pydantic-default / env-var values.

    Parameters
    ----------
    config:
        The parsed YAML configuration.
    agent_name:
        Which agent block to use for role→model mapping.  Defaults to
        ``"default"``.
    """
    # Start from env-var defaults
    model_settings = ModelSettings()
    agent_settings = AgentSettings()
    server_settings = ServerSettings()

    # Apply role→model mapping if agent exists
    agent_cfg = config.agents.get(agent_name)
    if agent_cfg:
        roles = agent_cfg.roles
        # Build kwargs using env-var aliases for stable round-tripping.
        model_kwargs: dict[str, str] = {}
        if roles.reasoning:
            model_kwargs["REASONING_MODEL"] = roles.reasoning
        if roles.search:
            model_kwargs["SEARCH_MODEL"] = roles.search
        if roles.classify:
            model_kwargs["CLASSIFY_MODEL"] = roles.classify
        if roles.summarize:
            model_kwargs["SUMMARIZE_MODEL"] = roles.summarize
        if roles.evaluate:
            model_kwargs["EVALUATE_MODEL"] = roles.evaluate

        if model_kwargs:
            # Merge with current defaults (alias-keyed)
            base_data = model_settings.model_dump(by_alias=True)
            base_data.update(model_kwargs)
            model_settings = ModelSettings(**base_data)

        # Apply agent behavior overrides
        agent_settings = _apply_agent_overrides(
            agent_settings, agent_cfg.settings
        )

    # Apply server overrides
    server_settings = _apply_server_overrides(
        server_settings, config.server
    )

    return Settings(
        models=model_settings,
        agent=agent_settings,
        server=server_settings,
    )


# ------------------------------------------------------------------ #
# Config → ProviderContext
# ------------------------------------------------------------------ #


def _normalize_search_provider_kind(raw_kind: str) -> str:
    """Normalize supported search-provider selector aliases."""
    normalized = raw_kind.strip().lower().replace("-", "_")
    if not normalized:
        return ""
    try:
        return _SEARCH_PROVIDER_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_SEARCH_PROVIDER_ALIASES))
        raise ValueError(
            f"Unsupported search_provider '{raw_kind}'. Supported values: {supported}"
        ) from exc


def _detect_search_provider_kind(base_url: str) -> str:
    """Infer the search adapter from the configured provider endpoint."""
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    if host.endswith(".openai.azure.com"):
        return "azure_openai_web_search"
    if host.endswith(".services.ai.azure.com") and "/api/projects/" in path:
        return "azure_foundry_web_search"
    return "perplexity"


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Convert permissive YAML-style truthy values to bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _create_search_provider_from_model(
    *,
    config: InqtrixConfig,
    settings: Settings,
    search_model_name: str,
    resolver: ModelResolver,
) -> SearchProvider:
    """Instantiate the configured search adapter for one named model."""
    model_cfg = config.models[search_model_name]
    provider_cfg = config.providers[model_cfg.provider]
    resolved_search = resolver.resolve(search_model_name)
    search_params = dict(resolved_search.params)

    explicit_kind = _normalize_search_provider_kind(
        str(search_params.pop("search_provider", ""))
    )
    provider_kind = explicit_kind or _detect_search_provider_kind(provider_cfg.base_url)

    if provider_kind == "azure_openai_web_search":
        tool_choice = search_params.pop("tool_choice", "auto")
        user_location = search_params.pop("user_location", None)
        include_action_sources = _coerce_bool(
            search_params.pop("include_action_sources", True),
            default=True,
        )
        default_headers = search_params.pop("default_headers", None)
        timeout = float(search_params.pop("timeout", 60.0))
        return AzureOpenAIWebSearch(
            base_url=provider_cfg.base_url,
            api_key=provider_cfg.api_key,
            default_model=resolved_search.model_id,
            timeout=timeout,
            tool_choice=tool_choice,
            user_location=user_location,
            include_action_sources=include_action_sources,
            default_headers=default_headers,
            request_params=search_params,
        )

    if provider_kind == "azure_foundry_web_search":
        timeout = float(search_params.pop("timeout", 60.0))
        agent_version = str(search_params.pop("agent_version", "")).strip()
        if search_params:
            log.warning(
                "Ignoring unsupported Azure Foundry Web Search config params for '%s': %s",
                search_model_name,
                ", ".join(sorted(search_params)),
            )
        return AzureFoundryWebSearch(
            project_endpoint=provider_cfg.base_url,
            agent_name=resolved_search.model_id,
            agent_version=agent_version,
            api_key=provider_cfg.api_key or None,
            timeout=timeout,
        )

    return PerplexitySearch(
        api_key="",
        model=resolved_search.model_id,
        cache_maxsize=settings.agent.search_cache_maxsize,
        cache_ttl=settings.agent.search_cache_ttl,
        request_params=search_params,
        _client=resolved_search.client,
    )


def create_providers_from_config(
    config: InqtrixConfig,
    settings: Settings | None = None,
    agent_name: str = "default",
) -> ProviderContext:
    """Create a :class:`ProviderContext` from a YAML config.

    Uses :class:`MultiClientLLMProvider` so different roles can route
    to different API endpoints.

    Parameters
    ----------
    config:
        The parsed YAML configuration with providers/models/agents.
    settings:
        Optional pre-built settings.  If ``None``, derived from config.
    agent_name:
        Which agent block to use for role→model mapping.
    """
    if settings is None:
        settings = config_to_settings(config, agent_name)

    resolver = ModelResolver(config)

    # LLM provider: multi-client routing
    llm = MultiClientLLMProvider(
        resolver=resolver,
        models=settings.models,
        agent_settings=settings.agent,
    )

    # Search provider: resolve the search model's provider
    agent_cfg = config.agents.get(agent_name)
    search_model_name = ""
    if agent_cfg and agent_cfg.roles.search:
        search_model_name = agent_cfg.roles.search
    else:
        search_model_name = settings.models.search_model

    # Build the search provider
    if search_model_name and resolver.has_model(search_model_name):
        search = _create_search_provider_from_model(
            config=config,
            settings=settings,
            search_model_name=search_model_name,
            resolver=resolver,
        )
    else:
        # Fallback: use the first provider defined for search
        # (this handles edge cases where search model isn't in config)
        from inqtrix.providers import create_providers

        fallback_ctx = create_providers(settings)
        search = fallback_ctx.search

    return ProviderContext(llm=llm, search=search)
