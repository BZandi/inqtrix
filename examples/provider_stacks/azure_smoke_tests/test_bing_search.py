"""Minimal Azure Foundry Bing grounding smoke test.

Use this before the combined Azure example when you want to verify only
the search path via ``AzureFoundryBingSearch``.

What this checks:
- Azure AI Foundry project endpoint is correct
- the configured Bing grounding agent is reachable
- a single search call returns an answer and citations

Required environment variables:
- AZURE_AI_PROJECT_ENDPOINT
- BING_AGENT_NAME or BING_AGENT_ID

One-time agent creation (optional): set ``BING_SMOKE_CREATE_AGENT=true`` to call
``AzureFoundryBingSearch.create_agent()`` before the search. Then you only need
Service Principal credentials plus ``BING_PROJECT_CONNECTION_ID``; you do not
need ``BING_AGENT_NAME`` / ``BING_AGENT_ID`` until later runs.

Optional:
- BING_AGENT_VERSION
- AZURE_AI_PROJECT_API_KEY (mutually exclusive with Service Principal below)
- AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET (when not using project API key)
- BING_EXECUTION_MODE: ``auto`` (default), ``responses``, or ``legacy`` (see provider docs)
- BING_SMOKE_CREATE_AGENT: ``true`` to create agent first (requires SP + connection id)
- BING_PROJECT_CONNECTION_ID (required when ``BING_SMOKE_CREATE_AGENT=true``)
- BING_CREATE_AGENT_MODEL (default ``gpt-4o`` — must be deployed in the Foundry project)
- BING_CREATE_AGENT_NAME (default ``inqtrix-bing-smoke``)
- BING_CREATE_AGENT_MARKET, BING_CREATE_AGENT_SET_LANG, BING_CREATE_AGENT_FRESHNESS, BING_CREATE_AGENT_COUNT
- AZURE_BING_TEST_QUERY
- INQTRIX_LOG_ENABLED
- INQTRIX_LOG_LEVEL
- INQTRIX_LOG_CONSOLE

Notes:
- freshness, market, set_lang, and count are fixed when the agent is created
- recency_filter, language_filter, and domain_filter are only runtime hints
- If Azure returns an error but the provider falls back to an empty result,
  this script prints the provider's nonfatal notice so the root cause stays visible

Run with::

    uv run python examples/provider_stacks/azure_smoke_tests/test_bing_search.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from inqtrix import AzureFoundryBingAPIError, AzureFoundryBingSearch
from inqtrix.exceptions import AgentRateLimited, AgentTimeout
from inqtrix.logging_config import configure_logging


load_dotenv()

_log_path = configure_logging(
    enabled=os.getenv("INQTRIX_LOG_ENABLED", "").lower() == "true",
    level=os.getenv("INQTRIX_LOG_LEVEL", "INFO"),
    console=os.getenv("INQTRIX_LOG_CONSOLE", "").lower() == "true",
)
if _log_path:
    print(f"Logging to {_log_path}")

QUERY = os.getenv(
    "AZURE_BING_TEST_QUERY",
    "Was ist der aktuelle Stand der GKV-Reform in Deutschland?",
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_env(name: str) -> str:
    return os.environ.get(name, "").strip()


def main() -> None:
    project_endpoint = _require_env("AZURE_AI_PROJECT_ENDPOINT")
    smoke_create = _optional_env("BING_SMOKE_CREATE_AGENT").lower() == "true"

    if smoke_create:
        bing_connection_id = _require_env("BING_PROJECT_CONNECTION_ID")
        tenant_id = _require_env("AZURE_TENANT_ID")
        client_id = _require_env("AZURE_CLIENT_ID")
        client_secret = _require_env("AZURE_CLIENT_SECRET")
        create_model = _optional_env("BING_CREATE_AGENT_MODEL") or "gpt-4o"
        create_name = _optional_env("BING_CREATE_AGENT_NAME") or "inqtrix-bing-smoke"
        market = _optional_env("BING_CREATE_AGENT_MARKET") or "de-DE"
        set_lang = _optional_env("BING_CREATE_AGENT_SET_LANG") or "de"
        freshness = _optional_env("BING_CREATE_AGENT_FRESHNESS") or "Week"
        count_raw = _optional_env("BING_CREATE_AGENT_COUNT") or "10"
        try:
            count = int(count_raw)
        except ValueError as exc:
            raise RuntimeError(
                f"BING_CREATE_AGENT_COUNT must be an integer, got {count_raw!r}"
            ) from exc

        search = AzureFoundryBingSearch.create_agent(
            project_endpoint=project_endpoint,
            bing_connection_id=bing_connection_id,
            model=create_model,
            agent_name=create_name,
            market=market,
            set_lang=set_lang,
            freshness=freshness,
            count=count,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
        next_steps_lines = [
            "Agent created. Add these to your .env for later runs (Option A):",
            f"BING_AGENT_NAME={search._agent_name}",
        ]
        if search._agent_version:
            next_steps_lines.append(f"BING_AGENT_VERSION={search._agent_version}")
        else:
            next_steps_lines.append(
                "# BING_AGENT_VERSION=  (omit — empty uses default/latest)"
            )
        if search._agent_id:
            next_steps_lines.append(
                f"# legacy id (optional): BING_AGENT_ID={search._agent_id}"
            )
        next_steps_lines.extend(
            [
                "",
                "Unset BING_SMOKE_CREATE_AGENT (or set to false) before re-running "
                "so you do not create duplicate agents each time.",
            ]
        )
        Console().print(
            Panel(
                "\n".join(next_steps_lines),
                title="Next steps",
                expand=False,
            )
        )
    else:
        bing_agent_name = _optional_env("BING_AGENT_NAME")
        bing_agent_version = _optional_env("BING_AGENT_VERSION")
        bing_agent_id = _optional_env("BING_AGENT_ID")
        if not bing_agent_name and not bing_agent_id:
            raise RuntimeError(
                "Missing required environment variable: BING_AGENT_NAME "
                "(or legacy BING_AGENT_ID). "
                "Alternatively set BING_SMOKE_CREATE_AGENT=true once to create an agent."
            )

        search_kwargs: dict[str, object] = {
            "project_endpoint": project_endpoint,
        }
        if bing_agent_name:
            search_kwargs["agent_name"] = bing_agent_name
        if bing_agent_version:
            search_kwargs["agent_version"] = bing_agent_version
        if bing_agent_id:
            search_kwargs["agent_id"] = bing_agent_id

        project_api_key = _optional_env("AZURE_AI_PROJECT_API_KEY")
        tenant_id = _optional_env("AZURE_TENANT_ID")
        client_id = _optional_env("AZURE_CLIENT_ID")
        client_secret = _optional_env("AZURE_CLIENT_SECRET")
        if project_api_key:
            search_kwargs["api_key"] = project_api_key
        elif tenant_id and client_id and client_secret:
            search_kwargs["tenant_id"] = tenant_id
            search_kwargs["client_id"] = client_id
            search_kwargs["client_secret"] = client_secret

        exec_mode = _optional_env("BING_EXECUTION_MODE").lower()
        if exec_mode in {"auto", "responses", "legacy"}:
            search_kwargs["execution_mode"] = exec_mode  # type: ignore[assignment]

        search = AzureFoundryBingSearch(**search_kwargs)

    result = search.search(
        QUERY,
        recency_filter="month",
        language_filter=["de"],
    )

    nonfatal_notice = search.consume_nonfatal_notice()
    citations = result.get("citations") or []
    citations_text = "\n".join(f"- {url}" for url in citations) or "<no citations returned>"

    console = Console()
    if nonfatal_notice:
        console.print(
            Panel(
                nonfatal_notice,
                title="Provider notice",
                expand=False,
            )
        )
    console.print(
        Panel(
            result.get("answer") or "<empty answer>",
            title="Azure Foundry Bing answer",
            expand=False,
        )
    )
    console.print(
        Panel(
            citations_text,
            title=f"Citations ({len(citations)})",
            expand=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (
        RuntimeError,
        ValueError,
        AgentTimeout,
        AgentRateLimited,
        AzureFoundryBingAPIError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
