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

Optional:
- BING_AGENT_VERSION
- AZURE_AI_PROJECT_API_KEY
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
    bing_agent_name = _optional_env("BING_AGENT_NAME")
    bing_agent_version = _optional_env("BING_AGENT_VERSION")
    bing_agent_id = _optional_env("BING_AGENT_ID")
    if not bing_agent_name and not bing_agent_id:
        raise RuntimeError(
            "Missing required environment variable: BING_AGENT_NAME "
            "(or legacy BING_AGENT_ID)"
        )

    search_kwargs = {
        "project_endpoint": project_endpoint,
    }
    if bing_agent_name:
        search_kwargs["agent_name"] = bing_agent_name
    if bing_agent_version:
        search_kwargs["agent_version"] = bing_agent_version
    if bing_agent_id:
        search_kwargs["agent_id"] = bing_agent_id

    project_api_key = _optional_env("AZURE_AI_PROJECT_API_KEY")
    if project_api_key:
        search_kwargs["api_key"] = project_api_key

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
