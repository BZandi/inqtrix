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

from openai import OpenAI

from inqtrix.config import AgentConfig, InqtrixConfig
from inqtrix.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderContext,
    SearchProvider,
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
    ) -> dict[str, Any]:
        """Build completion kwargs including model-level parameters."""
        request_kwargs: dict[str, Any] = {
            "model": resolved.model_id,
            "messages": messages,
            "timeout": timeout,
            "stream": False,
        }
        for key, value in resolved.params.items():
            if key in {"model", "messages", "timeout", "stream"}:
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
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        return self.complete_with_metadata(
            prompt,
            system=system,
            model=model,
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
        self, text: str, deadline: float | None = None
    ) -> tuple[str, int, int]:
        if not text.strip():
            return ("", 0, 0)
        self._clear_nonfatal_notice()
        if deadline is not None:
            _check_deadline(deadline)

        summarize_model = self._models.effective_summarize_model
        resolved = self._resolver.resolve(summarize_model)
        prompt = f"{SUMMARIZE_PROMPT}{text[:6000]}"

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
            return (text[:800], 0, 0)

    def is_available(self) -> bool:
        return True


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
    for field_name in type(overrides).model_fields:
        val = getattr(overrides, field_name)
        if val is not None:
            alias = _field_alias(AgentSettings, field_name)
            data[alias] = val
    return AgentSettings(**data)


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
        resolved_search = resolver.resolve(search_model_name)
        search: SearchProvider = PerplexitySearch(
            api_key="",
            model=resolved_search.model_id,
            cache_maxsize=settings.agent.search_cache_maxsize,
            cache_ttl=settings.agent.search_cache_ttl,
            request_params=resolved_search.params,
            _client=resolved_search.client,
        )
    else:
        # Fallback: use the first provider defined for search
        # (this handles edge cases where search model isn't in config)
        from inqtrix.providers import create_providers

        fallback_ctx = create_providers(settings)
        search = fallback_ctx.search

    return ProviderContext(llm=llm, search=search)
