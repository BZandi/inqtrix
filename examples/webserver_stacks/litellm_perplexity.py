"""Webserver-Spiegel zu provider_stacks/litellm_perplexity.py.

Mounts the OpenAI-compatible Inqtrix server on top of an explicit
LiteLLM + PerplexitySearch provider stack. Provider construction is
**1:1 identical** to ``examples/provider_stacks/litellm_perplexity.py``;
only the run block differs (it injects the providers into
``create_app(...)`` and starts ``uvicorn`` instead of calling
``agent.research(...)``).

Architecture
------------
A single LiteLLM proxy serves both the language model and the
Perplexity Sonar search API — that is why ``LiteLLM`` and
``PerplexitySearch`` share ``base_url`` and ``api_key``. Every request
goes through the LiteLLM endpoint, which routes it to the correct
upstream provider based on the model name.

Required environment variables
------------------------------
- ``LITELLM_API_KEY``                 — proxy auth, also reused for Perplexity
- ``LITELLM_BASE_URL``                — defaults to ``http://localhost:4000/v1``

Optional logging:
- ``INQTRIX_LOG_ENABLED`` / ``INQTRIX_LOG_LEVEL`` / ``INQTRIX_LOG_CONSOLE``
- ``INQTRIX_LOG_INCLUDE_WEB`` (default ``true``) — mirror uvicorn / FastAPI
  logs into the same file when file logging is enabled
- ``INQTRIX_LOG_WEB_LEVEL`` (default ``INFO``) — separate verbosity knob
  for the uvicorn / FastAPI loggers

Optional server bind:
- ``INQTRIX_SERVER_HOST`` (default ``0.0.0.0``)
- ``INQTRIX_SERVER_PORT`` (default ``5100``)

Optional security (off-by-default):
- ``INQTRIX_SERVER_TLS_KEYFILE`` + ``INQTRIX_SERVER_TLS_CERTFILE``  — HTTPS via uvicorn
- ``INQTRIX_SERVER_API_KEY``          — Bearer-API-key gate on /v1/chat/completions
- ``INQTRIX_SERVER_CORS_ORIGINS``     — comma-separated CORS whitelist

Run with::

    uv sync
    uv run python examples/webserver_stacks/litellm_perplexity.py

Then call::

    curl http://localhost:5100/health
    curl -X POST http://localhost:5100/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Hallo"}],
             "agent_overrides":{"report_profile":"deep"}}'
"""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from inqtrix import (
    AgentConfig,
    LiteLLM,
    PerplexitySearch,
    ReportProfile,
    ResearchAgent,
)
from inqtrix.logging_config import build_uvicorn_log_config, configure_logging
from inqtrix.providers.base import ProviderContext
from inqtrix.server import create_app
from inqtrix.server.security import resolve_tls_paths
from inqtrix.settings import (
    AgentSettings,
    ModelSettings,
    ServerSettings,
    Settings,
)


load_dotenv()

# Logging mirrors the provider_stacks pattern — controlled via env so
# the example stays quiet unless requested.
_INQTRIX_LOG_PATH = configure_logging(
    enabled=os.getenv("INQTRIX_LOG_ENABLED", "").lower() == "true",
    level=os.getenv("INQTRIX_LOG_LEVEL", "INFO"),
    console=os.getenv("INQTRIX_LOG_CONSOLE", "").lower() == "true",
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_providers() -> ProviderContext:
    """Build LiteLLM + PerplexitySearch — identical to the library example.

    Diff against ``provider_stacks/litellm_perplexity.py`` is intentionally
    confined to the surrounding run-block; this function is byte-for-byte
    the same Baukasten construction so behaviour stays comparable.
    """
    api_key = _require_env("LITELLM_API_KEY")
    base_url = os.environ.get(
        "LITELLM_BASE_URL", "http://localhost:4000/v1"
    ).strip()

    llm = LiteLLM(
        api_key=api_key,
        base_url=base_url,
        default_model="claude-opus-4.6-agent",
        classify_model="claude-sonnet-4.6",
        summarize_model="claude-sonnet-4.6",
        evaluate_model="claude-sonnet-4.6",
    )
    search = PerplexitySearch(
        api_key=api_key,
        base_url=base_url,
        model="perplexity-sonar-pro-agent",
    )
    return ProviderContext(llm=llm, search=search)


def _build_settings() -> Settings:
    """Return the full Settings stack used to drive the FastAPI server.

    All security / lifecycle defaults flow through ``ServerSettings``
    (and therefore through the ``INQTRIX_SERVER_*`` env-vars). The
    behavioural envelope mirrors the AgentConfig values used in the
    sibling library example, so a side-by-side run produces the same
    research behaviour for the same question.
    """
    agent = AgentSettings(
        report_profile=ReportProfile.DEEP,
        max_rounds=4,
        confidence_stop=8,
        max_context=12,
        first_round_queries=6,
        answer_prompt_citations_max=60,
        max_total_seconds=600,
        max_question_length=10_000,
        reasoning_timeout=120,
        search_timeout=60,
        summarize_timeout=60,
        high_risk_score_threshold=4,
        high_risk_classify_escalate=True,
        high_risk_evaluate_escalate=True,
        search_cache_maxsize=256,
        search_cache_ttl=3600,
    )
    return Settings(
        models=ModelSettings(),  # not used in Baukasten injection mode
        agent=agent,
        server=ServerSettings(),  # picks up INQTRIX_SERVER_* env vars
    )


def build_app() -> FastAPI:
    """Test-friendly entry point: build the wired FastAPI app.

    Used by the test suite (which calls this without ever invoking
    ``uvicorn.run``). The same function is what ``main()`` consumes
    so the production and test paths stay symmetric.
    """
    providers = _build_providers()
    settings = _build_settings()
    return create_app(settings=settings, providers=providers)


def main() -> None:
    """Build the app and start uvicorn (with optional TLS)."""
    app = build_app()
    settings = _build_settings()
    tls = resolve_tls_paths(settings.server)
    uvicorn_kwargs: dict[str, Any] = dict(
        host=os.getenv("INQTRIX_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("INQTRIX_SERVER_PORT", "5100")),
        workers=1,
        timeout_keep_alive=300,
    )
    if tls is not None:
        uvicorn_kwargs["ssl_keyfile"], uvicorn_kwargs["ssl_certfile"] = tls
    if os.getenv("INQTRIX_LOG_INCLUDE_WEB", "true").lower() != "false":
        uvicorn_kwargs["log_config"] = build_uvicorn_log_config(
            _INQTRIX_LOG_PATH,
            web_level=os.getenv("INQTRIX_LOG_WEB_LEVEL", "INFO"),
        )
    uvicorn.run(app, **uvicorn_kwargs)


# Optional: hint that an Inqtrix-`AgentConfig` / `ResearchAgent` could
# also be instantiated for in-process use; the webserver itself does
# not rely on these symbols. Kept as imports for symmetry with the
# library example so the diff against provider_stacks is minimal.
_ = (AgentConfig, ResearchAgent)


if __name__ == "__main__":
    main()
