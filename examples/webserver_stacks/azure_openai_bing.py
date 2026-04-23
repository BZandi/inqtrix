"""Webserver-Spiegel zu provider_stacks/azure_openai_bing.py.

Mounts the OpenAI-compatible Inqtrix server on top of an
``AzureOpenAILLM`` + ``AzureFoundryBingSearch`` stack (classic Bing
Grounding agent path through the project-scoped Responses API).
Provider construction is **1:1 identical** to the library example,
including the four Azure auth modes for the LLM and the
api-key-or-Service-Principal options for the Bing project endpoint.

Required environment variables
------------------------------
- ``AZURE_OPENAI_API_KEY``           (LLM — Option A default)
- ``AZURE_OPENAI_ENDPOINT``
- ``AZURE_OPENAI_DEPLOYMENT_NAME``
- ``AZURE_AI_PROJECT_ENDPOINT``
- ``BING_AGENT_NAME`` *(or legacy ``BING_AGENT_ID``)*

Optional (parity with provider_stacks):
- ``BING_AGENT_VERSION``
- ``BING_AGENT_ID``                   (legacy fallback)
- ``AZURE_AI_PROJECT_API_KEY``
- ``AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME``
- ``AZURE_TENANT_ID`` / ``AZURE_CLIENT_ID`` / ``AZURE_CLIENT_SECRET``

Foundry token caveat
--------------------
The Bing search provider mints a bearer token at constructor time
against the ``https://ai.azure.com/.default`` scope (~60-75 min
lifetime). The credential refreshes automatically; long-running server
processes do not need extra refresh logic. If the token cache enters
the <10 s expiry window mid-request the next call may transiently 401 —
the documented mitigation is a process restart.

Optional logging / server / security: see other webserver examples.

Run with::

    uv sync
    uv run python examples/webserver_stacks/azure_openai_bing.py
"""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from inqtrix import (
    AgentConfig,
    AzureFoundryBingSearch,
    AzureOpenAILLM,
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


def _optional_env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _build_providers() -> ProviderContext:
    """Build AzureOpenAILLM + AzureFoundryBingSearch — identical to the library example."""
    azure_endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
    azure_deployment = _require_env("AZURE_OPENAI_DEPLOYMENT_NAME")
    project_endpoint = _require_env("AZURE_AI_PROJECT_ENDPOINT")
    bing_agent_name = _optional_env("BING_AGENT_NAME")
    bing_agent_version = _optional_env("BING_AGENT_VERSION")
    bing_agent_id = _optional_env("BING_AGENT_ID")
    if not bing_agent_name and not bing_agent_id:
        raise RuntimeError(
            "Missing required environment variable: BING_AGENT_NAME "
            "(or legacy BING_AGENT_ID)"
        )

    azure_summarize_deployment = os.environ.get(
        "AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME", ""
    ).strip()

    # -- LLM Provider ---------------------------------------------------------
    #
    # Option A: API Key (default)
    azure_api_key = _require_env("AZURE_OPENAI_API_KEY")

    llm = AzureOpenAILLM(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        default_model=azure_deployment,
        summarize_model=azure_summarize_deployment,
    )

    # -- Option B: Service Principal — direct constructor args ----------------
    # llm = AzureOpenAILLM(
    #     azure_endpoint=azure_endpoint,
    #     tenant_id=os.environ["AZURE_TENANT_ID"],
    #     client_id=os.environ["AZURE_CLIENT_ID"],
    #     client_secret=os.environ["AZURE_CLIENT_SECRET"],
    #     default_model=azure_deployment,
    #     summarize_model=azure_summarize_deployment,
    # )

    # -- Option C: Service Principal — manual token provider ------------------
    # from azure.identity import ClientSecretCredential, get_bearer_token_provider
    # credential = ClientSecretCredential(
    #     tenant_id=os.environ["AZURE_TENANT_ID"],
    #     client_id=os.environ["AZURE_CLIENT_ID"],
    #     client_secret=os.environ["AZURE_CLIENT_SECRET"],
    # )
    # token_provider = get_bearer_token_provider(
    #     credential, "https://cognitiveservices.azure.com/.default",
    # )
    # llm = AzureOpenAILLM(
    #     azure_endpoint=azure_endpoint,
    #     azure_ad_token_provider=token_provider,
    #     default_model=azure_deployment,
    #     summarize_model=azure_summarize_deployment,
    # )

    # -- Option D: Existing token provider ------------------------------------
    # llm = AzureOpenAILLM(
    #     azure_endpoint=azure_endpoint,
    #     azure_ad_token_provider=existing_token_provider,
    #     default_model=azure_deployment,
    #     summarize_model=azure_summarize_deployment,
    # )

    # -- Option E: DefaultAzureCredential -------------------------------------
    # from azure.identity import DefaultAzureCredential
    # llm = AzureOpenAILLM(
    #     azure_endpoint=azure_endpoint,
    #     credential=DefaultAzureCredential(),
    #     default_model=azure_deployment,
    #     summarize_model=azure_summarize_deployment,
    # )

    # -- Search Provider ------------------------------------------------------
    #
    # Option A: Use an existing agent (recommended for production).
    project_api_key = _optional_env("AZURE_AI_PROJECT_API_KEY")
    tenant_id = _optional_env("AZURE_TENANT_ID")
    client_id = _optional_env("AZURE_CLIENT_ID")
    client_secret = _optional_env("AZURE_CLIENT_SECRET")

    search_kwargs: dict[str, Any] = {"project_endpoint": project_endpoint}
    if bing_agent_name:
        search_kwargs["agent_name"] = bing_agent_name
    if bing_agent_version:
        search_kwargs["agent_version"] = bing_agent_version
    if bing_agent_id:
        search_kwargs["agent_id"] = bing_agent_id
    if project_api_key:
        search_kwargs["api_key"] = project_api_key
    elif tenant_id and client_id and client_secret:
        search_kwargs["tenant_id"] = tenant_id
        search_kwargs["client_id"] = client_id
        search_kwargs["client_secret"] = client_secret

    search = AzureFoundryBingSearch(**search_kwargs)

    # -- Option B: Create the agent on the fly (one-time setup) ---------------
    #
    # search = AzureFoundryBingSearch.create_agent(
    #     project_endpoint=project_endpoint,
    #     bing_connection_id=os.environ["BING_PROJECT_CONNECTION_ID"],
    #     model="gpt-4o",
    #     market="de-DE",
    #     set_lang="de",
    #     freshness="Week",
    #     count=10,
    # )

    return ProviderContext(llm=llm, search=search)


def _build_settings() -> Settings:
    agent = AgentSettings(
        report_profile=ReportProfile.DEEP,
        max_rounds=4,
        confidence_stop=8,
        max_context=12,
        first_round_queries=6,
        answer_prompt_citations_max=60,
        max_total_seconds=300,
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
        models=ModelSettings(),
        agent=agent,
        server=ServerSettings(),
    )


def build_app() -> FastAPI:
    providers = _build_providers()
    settings = _build_settings()
    return create_app(settings=settings, providers=providers)


def main() -> None:
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


_ = (AgentConfig, ResearchAgent)


if __name__ == "__main__":
    main()
