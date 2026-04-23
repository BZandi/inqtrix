"""Tests for YAML configuration schema, loader, and env-var resolution."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from inqtrix.config import (
    AgentBehaviorConfig,
    AgentConfig,
    FallbackConfig,
    InqtrixConfig,
    ModelConfig,
    ParityConfig,
    ProviderConfig,
    RoleAssignment,
    ServerConfig,
    _resolve_env_vars,
    load_config,
)


# ------------------------------------------------------------------ #
# Environment variable resolution
# ------------------------------------------------------------------ #


class TestResolveEnvVars:

    def test_simple_replacement(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        assert _resolve_env_vars("key=${MY_KEY}") == "key=secret123"

    def test_multiple_replacements(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        result = _resolve_env_vars("http://${HOST}:${PORT}/v1")
        assert result == "http://localhost:8080/v1"

    def test_missing_env_var_raises(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
        with pytest.raises(ValueError, match="NONEXISTENT_VAR_12345"):
            _resolve_env_vars("key=${NONEXISTENT_VAR_12345}")

    def test_no_placeholders_passthrough(self):
        text = "plain text without any variables"
        assert _resolve_env_vars(text) == text

    def test_empty_string(self):
        assert _resolve_env_vars("") == ""

    def test_underscores_in_var_name(self, monkeypatch):
        monkeypatch.setenv("MY_LONG_VAR_NAME", "val")
        assert _resolve_env_vars("${MY_LONG_VAR_NAME}") == "val"

    def test_adjacent_vars(self, monkeypatch):
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert _resolve_env_vars("${A}${B}") == "12"

    def test_partial_match_not_replaced(self):
        """Bare $VAR (without braces) should NOT be resolved."""
        result = _resolve_env_vars("$NOT_REPLACED")
        assert result == "$NOT_REPLACED"


# ------------------------------------------------------------------ #
# Pydantic model unit tests
# ------------------------------------------------------------------ #


class TestProviderConfig:

    def test_minimal(self):
        p = ProviderConfig(base_url="http://localhost:4000/v1")
        assert p.base_url == "http://localhost:4000/v1"
        assert p.api_key == ""

    def test_with_api_key(self):
        p = ProviderConfig(base_url="https://api.openai.com/v1", api_key="sk-test")
        assert p.api_key == "sk-test"


class TestModelConfig:

    def test_minimal(self):
        m = ModelConfig(provider="openai", model_id="gpt-4o")
        assert m.params == {}

    def test_with_params(self):
        m = ModelConfig(
            provider="openai",
            model_id="gpt-4o",
            params={"temperature": 0.0, "max_tokens": 4096},
        )
        assert m.params["temperature"] == 0.0
        assert m.params["max_tokens"] == 4096


class TestRoleAssignment:

    def test_defaults_are_empty(self):
        r = RoleAssignment()
        assert r.reasoning == ""
        assert r.classify == ""
        assert r.summarize == ""
        assert r.evaluate == ""
        assert r.search == ""


class TestFallbackConfig:

    def test_defaults_are_empty_lists(self):
        f = FallbackConfig()
        assert f.reasoning == []
        assert f.classify == []

    def test_with_chains(self):
        f = FallbackConfig(reasoning=["model-a", "model-b"])
        assert f.reasoning == ["model-a", "model-b"]


class TestAgentBehaviorConfig:

    def test_defaults_are_none(self):
        a = AgentBehaviorConfig()
        assert a.max_rounds is None
        assert a.confidence_stop is None
        assert a.testing_mode is None

    def test_partial_override(self):
        a = AgentBehaviorConfig(max_rounds=6, confidence_stop=9)
        assert a.max_rounds == 6
        assert a.confidence_stop == 9
        assert a.max_context is None  # not overridden


class TestServerConfig:

    def test_defaults_are_none(self):
        s = ServerConfig()
        assert s.max_concurrent is None
        assert s.session_ttl_seconds is None


class TestParityConfig:

    def test_defaults(self):
        p = ParityConfig()
        assert p.analysis_model == ""
        assert p.analysis_timeout == 120


# ------------------------------------------------------------------ #
# InqtrixConfig validation
# ------------------------------------------------------------------ #


class TestInqtrixConfig:

    def test_empty_config(self):
        """Empty config signals env-var-only mode."""
        cfg = InqtrixConfig()
        assert cfg.providers == {}
        assert cfg.models == {}
        assert cfg.agents == {}

    def test_valid_full_config(self):
        cfg = InqtrixConfig(
            providers={
                "openai": ProviderConfig(
                    base_url="https://api.openai.com/v1", api_key="sk-test"
                ),
            },
            models={
                "gpt-4o": ModelConfig(provider="openai", model_id="gpt-4o"),
            },
            agents={
                "default": AgentConfig(
                    roles=RoleAssignment(reasoning="gpt-4o"),
                    settings=AgentBehaviorConfig(max_rounds=5),
                ),
            },
        )
        assert "openai" in cfg.providers
        assert "gpt-4o" in cfg.models
        assert "default" in cfg.agents

    def test_model_references_unknown_provider(self):
        with pytest.raises(ValidationError, match="unknown provider.*'missing'"):
            InqtrixConfig(
                providers={},
                models={
                    "bad-model": ModelConfig(
                        provider="missing", model_id="x"
                    ),
                },
            )

    def test_role_references_unknown_model(self):
        with pytest.raises(ValidationError, match="unknown model.*'nonexistent'"):
            InqtrixConfig(
                providers={
                    "p": ProviderConfig(base_url="http://localhost"),
                },
                models={
                    "m": ModelConfig(provider="p", model_id="x"),
                },
                agents={
                    "a": AgentConfig(
                        roles=RoleAssignment(reasoning="nonexistent"),
                    ),
                },
            )

    def test_fallback_references_unknown_model(self):
        with pytest.raises(ValidationError, match="fallback.*unknown model.*'ghost'"):
            InqtrixConfig(
                providers={
                    "p": ProviderConfig(base_url="http://localhost"),
                },
                models={
                    "m": ModelConfig(provider="p", model_id="x"),
                },
                agents={
                    "a": AgentConfig(
                        roles=RoleAssignment(reasoning="m"),
                        fallbacks=FallbackConfig(reasoning=["ghost"]),
                    ),
                },
            )

    def test_empty_role_skips_validation(self):
        """Empty role string means 'fall back to reasoning' — no validation error."""
        cfg = InqtrixConfig(
            providers={
                "p": ProviderConfig(base_url="http://localhost"),
            },
            models={
                "m": ModelConfig(provider="p", model_id="x"),
            },
            agents={
                "a": AgentConfig(
                    roles=RoleAssignment(reasoning="m", classify=""),
                ),
            },
        )
        assert cfg.agents["a"].roles.classify == ""

    def test_server_settings_in_config(self):
        cfg = InqtrixConfig(
            server=ServerConfig(max_concurrent=10, session_ttl_seconds=3600),
        )
        assert cfg.server.max_concurrent == 10
        assert cfg.server.session_ttl_seconds == 3600

    def test_parity_settings_in_config(self):
        cfg = InqtrixConfig(
            parity=ParityConfig(analysis_model="analysis-model", analysis_timeout=180),
        )
        assert cfg.parity.analysis_model == "analysis-model"
        assert cfg.parity.analysis_timeout == 180


# ------------------------------------------------------------------ #
# YAML loading (load_config)
# ------------------------------------------------------------------ #


class TestLoadConfig:

    def test_no_file_returns_empty_config(self, tmp_path, monkeypatch):
        """No YAML file → empty config (env-var mode)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("INQTRIX_CONFIG", raising=False)
        cfg = load_config()
        assert cfg.providers == {}
        assert cfg.models == {}

    def test_explicit_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-test")
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            providers:
              openai:
                base_url: "https://api.openai.com/v1"
                api_key: "${TEST_API_KEY}"
            models:
              gpt-4o:
                provider: openai
                model_id: "gpt-4o"
        """))
        cfg = load_config(yaml_file)
        assert cfg.providers["openai"].api_key == "sk-test"
        assert cfg.models["gpt-4o"].model_id == "gpt-4o"

    def test_env_var_path(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "env-config.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            providers:
              local:
                base_url: "http://localhost:11434/v1"
                api_key: "ollama"
        """))
        monkeypatch.setenv("INQTRIX_CONFIG", str(yaml_file))
        cfg = load_config()
        assert "local" in cfg.providers
        assert cfg.providers["local"].api_key == "ollama"

    def test_default_path_inqtrix_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("INQTRIX_CONFIG", raising=False)
        (tmp_path / "inqtrix.yaml").write_text(textwrap.dedent("""\
            providers:
              p:
                base_url: "http://localhost"
        """))
        cfg = load_config()
        assert "p" in cfg.providers

    def test_default_path_inqtrix_yml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("INQTRIX_CONFIG", raising=False)
        (tmp_path / "inqtrix.yml").write_text(textwrap.dedent("""\
            providers:
              q:
                base_url: "http://localhost"
        """))
        cfg = load_config()
        assert "q" in cfg.providers

    def test_default_path_dot_inqtrix_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("INQTRIX_CONFIG", raising=False)
        (tmp_path / ".inqtrix.yaml").write_text(textwrap.dedent("""\
            providers:
              r:
                base_url: "http://localhost"
        """))
        cfg = load_config()
        assert "r" in cfg.providers

    def test_explicit_path_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_empty_yaml_returns_empty_config(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cfg = load_config(yaml_file)
        assert cfg.providers == {}
        assert cfg.models == {}

    def test_env_var_resolution_in_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_BASE_URL", "https://api.anthropic.com/v1")
        monkeypatch.setenv("MY_API_KEY", "ant-key-123")
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            providers:
              anthropic:
                base_url: "${MY_BASE_URL}"
                api_key: "${MY_API_KEY}"
        """))
        cfg = load_config(yaml_file)
        assert cfg.providers["anthropic"].base_url == "https://api.anthropic.com/v1"
        assert cfg.providers["anthropic"].api_key == "ant-key-123"

    def test_dotenv_resolution_in_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DOTENV_ONLY_API_KEY", raising=False)
        (tmp_path / ".env").write_text(
            "DOTENV_ONLY_API_KEY=dotenv-key\n",
            encoding="utf-8",
        )
        (tmp_path / "inqtrix.yaml").write_text(textwrap.dedent("""\
            providers:
              openai:
                base_url: "https://api.openai.com/v1"
                api_key: "${DOTENV_ONLY_API_KEY}"
        """))

        cfg = load_config()

        assert cfg.providers["openai"].api_key == "dotenv-key"

    def test_process_env_takes_precedence_over_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DOTENV_OVERRIDE_KEY", "from-process-env")
        (tmp_path / ".env").write_text(
            "DOTENV_OVERRIDE_KEY=from-dotenv\n",
            encoding="utf-8",
        )
        (tmp_path / "inqtrix.yaml").write_text(textwrap.dedent("""\
            providers:
              openai:
                base_url: "https://api.openai.com/v1"
                api_key: "${DOTENV_OVERRIDE_KEY}"
        """))

        cfg = load_config()

        assert cfg.providers["openai"].api_key == "from-process-env"

    def test_missing_env_var_in_yaml_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TOTALLY_MISSING_VAR", raising=False)
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            providers:
              x:
                base_url: "http://localhost"
                api_key: "${TOTALLY_MISSING_VAR}"
        """))
        with pytest.raises(ValueError, match="TOTALLY_MISSING_VAR"):
            load_config(yaml_file)

    def test_full_roundtrip(self, tmp_path, monkeypatch):
        """Full YAML with providers, models, agents, server parses correctly."""
        monkeypatch.setenv("OAI_KEY", "sk-round")
        monkeypatch.setenv("PPLX_KEY", "pplx-round")
        yaml_file = tmp_path / "full.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            providers:
              openai:
                base_url: "https://api.openai.com/v1"
                api_key: "${OAI_KEY}"
              perplexity:
                base_url: "https://api.perplexity.ai"
                api_key: "${PPLX_KEY}"

            models:
              gpt-4o:
                provider: openai
                model_id: "gpt-4o"
                params:
                  temperature: 0.0
              sonar-pro:
                provider: perplexity
                model_id: "sonar-pro"

            agents:
              default:
                roles:
                  reasoning: gpt-4o
                  classify: gpt-4o
                  summarize: gpt-4o
                  evaluate: gpt-4o
                  search: sonar-pro
                fallbacks:
                  reasoning:
                    - sonar-pro
                settings:
                  max_rounds: 6
                  confidence_stop: 9

            server:
              max_concurrent: 5
              session_ttl_seconds: 7200

            parity:
              analysis_model: gpt-4o
              analysis_timeout: 180
        """))
        cfg = load_config(yaml_file)

        # Providers
        assert cfg.providers["openai"].api_key == "sk-round"
        assert cfg.providers["perplexity"].base_url == "https://api.perplexity.ai"

        # Models
        assert cfg.models["gpt-4o"].provider == "openai"
        assert cfg.models["gpt-4o"].params["temperature"] == 0.0
        assert cfg.models["sonar-pro"].provider == "perplexity"

        # Agent roles
        agent = cfg.agents["default"]
        assert agent.roles.reasoning == "gpt-4o"
        assert agent.roles.search == "sonar-pro"

        # Agent fallbacks
        assert agent.fallbacks.reasoning == ["sonar-pro"]

        # Agent settings
        assert agent.settings.max_rounds == 6
        assert agent.settings.confidence_stop == 9
        assert agent.settings.max_context is None  # not set

        # Server
        assert cfg.server.max_concurrent == 5
        assert cfg.server.session_ttl_seconds == 7200

        # Parity tooling
        assert cfg.parity.analysis_model == "gpt-4o"
        assert cfg.parity.analysis_timeout == 180

    def test_string_path_accepted(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "str-path.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            providers:
              s:
                base_url: "http://localhost"
        """))
        cfg = load_config(str(yaml_file))
        assert "s" in cfg.providers

    def test_priority_explicit_over_env(self, tmp_path, monkeypatch):
        """Explicit path argument takes precedence over INQTRIX_CONFIG env var."""
        env_file = tmp_path / "env.yaml"
        env_file.write_text(textwrap.dedent("""\
            providers:
              from_env:
                base_url: "http://env"
        """))
        explicit_file = tmp_path / "explicit.yaml"
        explicit_file.write_text(textwrap.dedent("""\
            providers:
              from_explicit:
                base_url: "http://explicit"
        """))
        monkeypatch.setenv("INQTRIX_CONFIG", str(env_file))
        cfg = load_config(explicit_file)
        assert "from_explicit" in cfg.providers
        assert "from_env" not in cfg.providers

    def test_priority_env_over_default(self, tmp_path, monkeypatch):
        """INQTRIX_CONFIG env var takes precedence over default file paths."""
        monkeypatch.chdir(tmp_path)
        # Create both default file and env-specified file
        (tmp_path / "inqtrix.yaml").write_text(textwrap.dedent("""\
            providers:
              from_default:
                base_url: "http://default"
        """))
        env_file = tmp_path / "custom.yaml"
        env_file.write_text(textwrap.dedent("""\
            providers:
              from_env_var:
                base_url: "http://custom"
        """))
        monkeypatch.setenv("INQTRIX_CONFIG", str(env_file))
        cfg = load_config()
        assert "from_env_var" in cfg.providers
        assert "from_default" not in cfg.providers

    def test_validation_error_bad_model_ref_in_yaml(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            providers:
              p:
                base_url: "http://localhost"
            models:
              m:
                provider: ghost_provider
                model_id: "x"
        """))
        with pytest.raises(ValidationError, match="unknown provider"):
            load_config(yaml_file)
