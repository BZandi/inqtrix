"""Tests for config bridge — YAML config to Settings/Providers translation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inqtrix.config import (
    AgentBehaviorConfig,
    AgentConfig,
    FallbackConfig,
    InqtrixConfig,
    ModelConfig,
    ProviderConfig,
    RoleAssignment,
    ServerConfig,
)
from inqtrix.config_bridge import (
    ModelResolver,
    MultiClientLLMProvider,
    ResolvedModel,
    config_to_settings,
    create_providers_from_config,
)
from inqtrix.settings import AgentSettings, ModelSettings, ServerSettings


# ------------------------------------------------------------------ #
# Helper: build a minimal valid InqtrixConfig
# ------------------------------------------------------------------ #


def _make_config(
    *,
    providers: dict | None = None,
    models: dict | None = None,
    agents: dict | None = None,
    server: ServerConfig | None = None,
) -> InqtrixConfig:
    """Build a valid InqtrixConfig with sensible defaults."""
    _providers = providers or {
        "openai": ProviderConfig(
            base_url="https://api.openai.com/v1", api_key="sk-test"
        ),
        "perplexity": ProviderConfig(
            base_url="https://api.perplexity.ai", api_key="pplx-test"
        ),
    }
    _models = models or {
        "gpt-4o": ModelConfig(provider="openai", model_id="gpt-4o"),
        "gpt-4o-mini": ModelConfig(
            provider="openai",
            model_id="gpt-4o-mini",
            params={"temperature": 0.0},
        ),
        "sonar-pro": ModelConfig(provider="perplexity", model_id="sonar-pro"),
    }
    _agents = agents or {
        "default": AgentConfig(
            roles=RoleAssignment(
                reasoning="gpt-4o",
                classify="gpt-4o-mini",
                summarize="gpt-4o-mini",
                evaluate="gpt-4o-mini",
                search="sonar-pro",
            ),
            fallbacks=FallbackConfig(reasoning=["gpt-4o-mini"]),
            settings=AgentBehaviorConfig(max_rounds=6, confidence_stop=9),
        ),
    }
    return InqtrixConfig(
        providers=_providers,
        models=_models,
        agents=_agents,
        server=server or ServerConfig(),
    )


# ------------------------------------------------------------------ #
# ModelResolver
# ------------------------------------------------------------------ #


class TestModelResolver:

    def test_resolve_returns_correct_model_id(self):
        config = _make_config()
        resolver = ModelResolver(config)
        resolved = resolver.resolve("gpt-4o")
        assert resolved.model_id == "gpt-4o"
        assert resolved.params == {}

    def test_resolve_includes_params(self):
        config = _make_config()
        resolver = ModelResolver(config)
        resolved = resolver.resolve("gpt-4o-mini")
        assert resolved.params == {"temperature": 0.0}

    def test_resolve_caches_clients(self):
        """Two models on same provider should share one OpenAI client."""
        config = _make_config()
        resolver = ModelResolver(config)
        r1 = resolver.resolve("gpt-4o")
        r2 = resolver.resolve("gpt-4o-mini")
        assert r1.client is r2.client  # same object

    def test_resolve_different_providers_different_clients(self):
        config = _make_config()
        resolver = ModelResolver(config)
        r_openai = resolver.resolve("gpt-4o")
        r_pplx = resolver.resolve("sonar-pro")
        assert r_openai.client is not r_pplx.client

    def test_resolve_unknown_model_raises(self):
        config = _make_config()
        resolver = ModelResolver(config)
        with pytest.raises(KeyError):
            resolver.resolve("nonexistent-model")

    def test_has_model(self):
        config = _make_config()
        resolver = ModelResolver(config)
        assert resolver.has_model("gpt-4o") is True
        assert resolver.has_model("nonexistent") is False


# ------------------------------------------------------------------ #
# config_to_settings
# ------------------------------------------------------------------ #


class TestConfigToSettings:

    def test_role_mapping_to_model_settings(self):
        config = _make_config()
        settings = config_to_settings(config)
        assert settings.models.reasoning_model == "gpt-4o"
        assert settings.models.search_model == "sonar-pro"
        assert settings.models.classify_model == "gpt-4o-mini"
        assert settings.models.summarize_model == "gpt-4o-mini"
        assert settings.models.evaluate_model == "gpt-4o-mini"

    def test_agent_behavior_overrides(self):
        config = _make_config()
        settings = config_to_settings(config)
        assert settings.agent.max_rounds == 6
        assert settings.agent.confidence_stop == 9
        # Non-overridden fields should keep defaults
        assert settings.agent.max_context == AgentSettings().max_context

    def test_server_overrides(self):
        config = _make_config(
            server=ServerConfig(max_concurrent=10, session_ttl_seconds=7200)
        )
        settings = config_to_settings(config)
        assert settings.server.max_concurrent == 10
        assert settings.server.session_ttl_seconds == 7200
        # Non-overridden
        assert settings.server.max_messages_history == ServerSettings().max_messages_history

    def test_empty_config_returns_defaults(self):
        config = InqtrixConfig()
        settings = config_to_settings(config)
        # Should be equivalent to Settings()
        defaults = ModelSettings()
        assert settings.models.reasoning_model == defaults.reasoning_model
        assert settings.models.search_model == defaults.search_model

    def test_nonexistent_agent_returns_defaults(self):
        config = _make_config()
        settings = config_to_settings(config, agent_name="nonexistent")
        defaults = ModelSettings()
        assert settings.models.reasoning_model == defaults.reasoning_model

    def test_partial_roles_only_search(self):
        """Agent with only search role should keep default reasoning."""
        config = InqtrixConfig(
            providers={
                "pplx": ProviderConfig(
                    base_url="https://api.perplexity.ai", api_key="k"
                ),
            },
            models={
                "sonar": ModelConfig(provider="pplx", model_id="sonar-pro"),
            },
            agents={
                "default": AgentConfig(
                    roles=RoleAssignment(search="sonar"),
                ),
            },
        )
        settings = config_to_settings(config)
        assert settings.models.search_model == "sonar"
        # reasoning should stay default since it was empty
        defaults = ModelSettings()
        assert settings.models.reasoning_model == defaults.reasoning_model


# ------------------------------------------------------------------ #
# MultiClientLLMProvider
# ------------------------------------------------------------------ #


class TestMultiClientLLMProvider:

    def _make_provider(self) -> tuple[MultiClientLLMProvider, MagicMock]:
        """Create a MultiClientLLMProvider with a mocked resolver."""
        config = _make_config()
        resolver = ModelResolver(config)

        # Mock the OpenAI client's completions.create
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        # Patch all clients in the resolver to use our mock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        # Replace all cached clients
        for key in list(resolver._clients.keys()):
            resolver._clients[key] = mock_client
        # Pre-resolve to populate cache, then replace
        resolver.resolve("gpt-4o")
        resolver.resolve("sonar-pro")
        for key in list(resolver._clients.keys()):
            resolver._clients[key] = mock_client

        settings = config_to_settings(config)
        provider = MultiClientLLMProvider(
            resolver=resolver,
            models=settings.models,
            agent_settings=settings.agent,
        )
        return provider, mock_client

    def test_complete_returns_content(self):
        provider, mock = self._make_provider()
        result = provider.complete("Hello")
        assert result == "test response"

    def test_complete_uses_resolved_model_id(self):
        provider, mock = self._make_provider()
        provider.complete("Hello", model="gpt-4o")
        call_kwargs = mock.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    def test_complete_passes_model_params(self):
        provider, mock = self._make_provider()
        provider.complete("Hello", model="gpt-4o-mini")
        call_kwargs = mock.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    def test_complete_default_model(self):
        provider, mock = self._make_provider()
        provider.complete("Hello")
        call_kwargs = mock.chat.completions.create.call_args
        # Default is reasoning model = gpt-4o
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    def test_complete_with_system_message(self):
        provider, mock = self._make_provider()
        provider.complete("Hello", system="You are helpful")
        call_kwargs = mock.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_complete_handles_sse_string_response(self):
        provider, mock = self._make_provider()
        mock.chat.completions.create.return_value = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"test "}}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"response"}}],"usage":{"prompt_tokens":10,"completion_tokens":20}}\n\n'
            'data: [DONE]\n\n'
        )

        state = {"total_prompt_tokens": 0, "total_completion_tokens": 0}
        result = provider.complete_with_metadata("Hello", state=state)

        assert result.content == "test response"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert state["total_prompt_tokens"] == 10
        assert state["total_completion_tokens"] == 20

    def test_models_property(self):
        provider, _ = self._make_provider()
        assert provider.models.reasoning_model == "gpt-4o"
        assert provider.models.search_model == "sonar-pro"

    def test_is_available(self):
        provider, _ = self._make_provider()
        assert provider.is_available() is True

    def test_summarize_parallel_empty_text(self):
        provider, _ = self._make_provider()
        facts, pt, ct = provider.summarize_parallel("")
        assert facts == ""
        assert pt == 0
        assert ct == 0

    def test_summarize_parallel_passes_model_params(self):
        provider, mock = self._make_provider()
        provider.summarize_parallel("Some text")
        call_kwargs = mock.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0


# ------------------------------------------------------------------ #
# create_providers_from_config
# ------------------------------------------------------------------ #


class TestCreateProvidersFromConfig:

    def test_returns_provider_context(self):
        config = _make_config()
        ctx = create_providers_from_config(config)
        assert ctx.llm is not None
        assert ctx.search is not None

    def test_llm_is_multi_client(self):
        config = _make_config()
        ctx = create_providers_from_config(config)
        assert isinstance(ctx.llm, MultiClientLLMProvider)

    def test_search_uses_resolved_model(self):
        config = _make_config()
        ctx = create_providers_from_config(config)
        # The search provider should exist and be a PerplexitySearch
        from inqtrix.providers import PerplexitySearch
        assert isinstance(ctx.search, PerplexitySearch)

    def test_search_receives_model_params(self):
        config = _make_config(
            models={
                "gpt-4o": ModelConfig(provider="openai", model_id="gpt-4o"),
                "gpt-4o-mini": ModelConfig(
                    provider="openai",
                    model_id="gpt-4o-mini",
                    params={"temperature": 0.0},
                ),
                "sonar-pro": ModelConfig(
                    provider="perplexity",
                    model_id="sonar-pro",
                    params={"presence_penalty": 0.1},
                ),
            },
        )
        ctx = create_providers_from_config(config)
        assert getattr(ctx.search, "_request_params", {}) == {"presence_penalty": 0.1}

    def test_custom_settings_passed_through(self):
        config = _make_config()
        from inqtrix.settings import Settings
        custom_settings = Settings(
            agent=AgentSettings(max_rounds=99),
        )
        ctx = create_providers_from_config(config, settings=custom_settings)
        assert ctx.llm.models.reasoning_model == custom_settings.models.reasoning_model
