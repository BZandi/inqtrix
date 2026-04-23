"""YAML configuration schema and loader for Inqtrix.

Provides an optional ``inqtrix.yaml`` configuration file that defines
multiple LLM/search providers, named models, and role-to-model mappings.
When no YAML file is found, the system falls back to environment-variable
configuration (full backwards compatibility).

API keys are **never** stored in YAML — use ``${ENV_VAR}`` references
which are resolved at load time. For local development, ``.env`` is
auto-loaded before resolving placeholders.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, model_validator

from inqtrix.report_profiles import ReportProfile


# ------------------------------------------------------------------ #
# Environment variable resolution
# ------------------------------------------------------------------ #

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_DOTENV_NAME = ".env"


def _load_dotenv_file(path: Path, *, protected_keys: set[str]) -> None:
    """Load key-value pairs from *path* into ``os.environ``.

    Existing process-level environment variables always win. Values loaded
    from a later local ``.env`` file may override values from an earlier one.
    """
    if not path.exists() or not path.is_file():
        return

    for name, value in dotenv_values(path).items():
        if value is None or name in protected_keys:
            continue
        os.environ[name] = value


def _resolve_env_vars(text: str) -> str:
    """Replace ``${VAR}`` placeholders with their environment values.

    Raises :class:`ValueError` if a referenced variable is not set.
    """

    def _replace(m: re.Match) -> str:
        name = m.group(1)
        val = os.environ.get(name)
        if val is None:
            raise ValueError(
                f"Environment variable '{name}' is not set "
                f"(referenced in config as ${{{name}}})"
            )
        return val

    return _ENV_RE.sub(_replace, text)


def _resolve_env_in_data(obj: Any) -> Any:
    """Recursively resolve ``${VAR}`` in parsed YAML data.

    Operates on the **parsed** data structure rather than raw YAML text,
    so environment variable values cannot inject YAML metacharacters.
    """
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_in_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_in_data(item) for item in obj]
    return obj


# ------------------------------------------------------------------ #
# Provider configuration
# ------------------------------------------------------------------ #


class ProviderConfig(BaseModel):
    """A named provider connection endpoint.

    Example YAML::

        providers:
          openai:
            base_url: "https://api.openai.com/v1"
            api_key: "${OPENAI_API_KEY}"
    """

    base_url: str
    api_key: str = ""


# ------------------------------------------------------------------ #
# Model configuration
# ------------------------------------------------------------------ #


class ModelConfig(BaseModel):
    """A named model referencing a provider.

    Example YAML::

        models:
          gpt-4o:
            provider: openai
            model_id: "gpt-4o"
            params:
              temperature: 0.0
    """

    provider: str
    model_id: str
    params: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# Agent configuration
# ------------------------------------------------------------------ #


class RoleAssignment(BaseModel):
    """Maps agent roles to named models.

    Empty string means "fall back to reasoning model".
    """

    reasoning: str = ""
    classify: str = ""
    summarize: str = ""
    evaluate: str = ""
    search: str = ""


class FallbackConfig(BaseModel):
    """Optional per-role fallback chains (list of model names)."""

    reasoning: list[str] = Field(default_factory=list)
    classify: list[str] = Field(default_factory=list)
    summarize: list[str] = Field(default_factory=list)
    evaluate: list[str] = Field(default_factory=list)
    search: list[str] = Field(default_factory=list)


class AgentBehaviorConfig(BaseModel):
    """Agent-level settings that override :class:`AgentSettings` defaults.

    Fields set to ``None`` are not overridden — the Pydantic default
    (or environment variable) takes precedence.
    """

    report_profile: ReportProfile | None = None
    max_rounds: int | None = None
    confidence_stop: int | None = None
    max_context: int | None = None
    first_round_queries: int | None = None
    answer_prompt_citations_max: int | None = None
    reasoning_timeout: int | None = None
    search_timeout: int | None = None
    summarize_timeout: int | None = None
    max_total_seconds: int | None = None
    max_question_length: int | None = None
    high_risk_score_threshold: int | None = None
    high_risk_classify_escalate: bool | None = None
    high_risk_evaluate_escalate: bool | None = None
    search_cache_maxsize: int | None = None
    search_cache_ttl: int | None = None
    testing_mode: bool | None = None


class AgentConfig(BaseModel):
    """A named agent with its own role→model mapping and settings."""

    roles: RoleAssignment = Field(default_factory=RoleAssignment)
    fallbacks: FallbackConfig = Field(default_factory=FallbackConfig)
    settings: AgentBehaviorConfig = Field(default_factory=AgentBehaviorConfig)


# ------------------------------------------------------------------ #
# Server configuration
# ------------------------------------------------------------------ #


class ServerConfig(BaseModel):
    """Server-level settings (override :class:`ServerSettings` defaults)."""

    max_concurrent: int | None = None
    max_messages_history: int | None = None
    session_ttl_seconds: int | None = None
    session_max_count: int | None = None
    session_max_context_blocks: int | None = None
    session_max_claim_ledger: int | None = None


class ParityConfig(BaseModel):
    """Optional parity-tooling configuration for diagnostic analysis."""

    analysis_model: str = ""
    analysis_timeout: int = 120


# ------------------------------------------------------------------ #
# Root configuration
# ------------------------------------------------------------------ #


class InqtrixConfig(BaseModel):
    """Root YAML configuration for Inqtrix.

    All sections are optional.  An empty :class:`InqtrixConfig` (no
    providers, no models, no agents) signals the env-var-only mode.
    """

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    server: ServerConfig = Field(default_factory=ServerConfig)
    parity: ParityConfig = Field(default_factory=ParityConfig)

    @model_validator(mode="after")
    def _validate_references(self) -> "InqtrixConfig":
        """Ensure all model→provider and role→model references are valid."""
        # Model → provider
        for model_name, model_cfg in self.models.items():
            if model_cfg.provider not in self.providers:
                raise ValueError(
                    f"Model '{model_name}' references unknown provider "
                    f"'{model_cfg.provider}'"
                )
        # Role → model
        for agent_name, agent_cfg in self.agents.items():
            for role_name in ("reasoning", "classify", "summarize",
                              "evaluate", "search"):
                model_ref = getattr(agent_cfg.roles, role_name, "")
                if model_ref and model_ref not in self.models:
                    raise ValueError(
                        f"Agent '{agent_name}' role '{role_name}' references "
                        f"unknown model '{model_ref}'"
                    )
            # Fallback → model
            for role_name, chain in agent_cfg.fallbacks.model_dump().items():
                for fb_ref in chain:
                    if fb_ref not in self.models:
                        raise ValueError(
                            f"Agent '{agent_name}' fallback for '{role_name}' "
                            f"references unknown model '{fb_ref}'"
                        )
        return self


# ------------------------------------------------------------------ #
# Loader
# ------------------------------------------------------------------ #

_DEFAULT_PATHS = [
    Path("inqtrix.yaml"),
    Path("inqtrix.yml"),
    Path(".inqtrix.yaml"),
]


def load_config(path: str | Path | None = None) -> InqtrixConfig:
    """Load YAML configuration with ``.env`` and ``${ENV_VAR}`` resolution.

    Resolution order:

    1. Local ``.env`` in the current directory
    2. Explicit *path* argument
    3. ``INQTRIX_CONFIG`` environment variable
    4. Default paths in current directory (``inqtrix.yaml``, ``inqtrix.yml``,
       ``.inqtrix.yaml``)
    5. Adjacent ``.env`` next to the resolved config file, if different
    6. No file found → empty config (all defaults, env-var mode)
    """
    protected_keys = set(os.environ)
    _load_dotenv_file(Path.cwd() / _DOTENV_NAME, protected_keys=protected_keys)

    if path is None:
        env_path = os.environ.get("INQTRIX_CONFIG")
        if env_path:
            path = Path(env_path)

    if path is None:
        for default in _DEFAULT_PATHS:
            if default.exists():
                path = default
                break

    if path is None:
        return InqtrixConfig()

    path = Path(path)

    config_env_path = path.parent / _DOTENV_NAME
    cwd_env_path = Path.cwd() / _DOTENV_NAME
    if config_env_path != cwd_env_path:
        _load_dotenv_file(config_env_path, protected_keys=protected_keys)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")

    # Parse YAML first, then resolve ${ENV_VAR} on the parsed data.
    # This prevents env-var values (e.g. API keys containing `: `)
    # from injecting YAML metacharacters.
    data = yaml.safe_load(raw_text)
    if data is None:
        return InqtrixConfig()

    resolved_data = _resolve_env_in_data(data)
    return InqtrixConfig.model_validate(resolved_data)
