"""Minimal Azure Foundry Web Search smoke test.

Use this before the combined Azure example when you want to verify only
the search path via ``AzureFoundryWebSearch`` (Responses API).

For the full stack (``AzureOpenAILLM`` + ``AzureFoundryWebSearch``), see:

- ``examples/example_4/basic_azure_web.py``

What this checks:
- Azure AI Foundry project endpoint is correct
- the configured Web Search agent is reachable
- a single search call returns an answer and citations

Required environment variables:
- AZURE_AI_PROJECT_ENDPOINT
- WEB_SEARCH_AGENT_NAME

Optional:
- WEB_SEARCH_AGENT_VERSION
- AZURE_WEB_SEARCH_TEST_QUERY
- INQTRIX_LOG_ENABLED
- INQTRIX_LOG_LEVEL
- INQTRIX_LOG_CONSOLE

Authentication — three options (pick ONE):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1 — API Key (simplest)**
    Set in .env::

        AZURE_AI_PROJECT_API_KEY=abc123...

    Found in Azure Portal → your Foundry project → Project Settings
    → Keys & Endpoint.

**Option 2 — Service Principal**
    Set in .env::

        AZURE_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        AZURE_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        AZURE_CLIENT_SECRET=your-secret-value

    Created via Azure Portal → App Registrations → New Registration
    → Certificates & Secrets.

**Option 3 — DefaultAzureCredential (fallback)**
    Requires NO extra env vars.  Uses the first available credential:
    ``az login`` (Azure CLI), Managed Identity, VS Code Azure sign-in, etc.
    Install Azure CLI: ``brew install azure-cli && az login``

Notes:
- recency_filter, language_filter, and domain_filter are runtime hints
- If Azure returns an error but the provider falls back to an empty result,
  this script prints the provider's nonfatal notice so the root cause stays visible
- Some prompts can trigger Azure content filtering or false positives.
    If that happens, override ``AZURE_WEB_SEARCH_TEST_QUERY`` with a simpler,
    current-events style query.

Run with::

    uv run python examples/example_4/basic_test_component_azure_web_search.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from inqtrix import AzureFoundryWebSearch, AzureFoundryWebSearchAPIError
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
    "AZURE_WEB_SEARCH_TEST_QUERY",
    "Was sind die heutigen Schlagzeilen bei Tagesschau und ZDFheute?",
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    project_endpoint = _require_env("AZURE_AI_PROJECT_ENDPOINT")
    agent_name = _require_env("WEB_SEARCH_AGENT_NAME")
    agent_version = os.getenv("WEB_SEARCH_AGENT_VERSION", "")

    # ── Auth: pick the first available credential ────────────────────
    api_key = os.getenv("AZURE_AI_PROJECT_API_KEY")
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")

    # Option 1 — API Key (if AZURE_AI_PROJECT_API_KEY is set in .env)
    # → Simplest. No azure-identity or az login needed.
    search = AzureFoundryWebSearch(
        project_endpoint=project_endpoint,
        agent_name=agent_name,
        agent_version=agent_version,
        api_key=api_key,                # None → falls through to Entra ID
        tenant_id=tenant_id,            # None → ignored when api_key is set
        client_id=client_id,            # None → ignored when api_key is set
        client_secret=client_secret,    # None → ignored when api_key is set
    )
    # The constructor resolves auth automatically in this priority:
    #   1. api_key set          → uses static API key
    #   2. tenant/client/secret → uses Service Principal (ClientSecretCredential)
    #   3. none of the above    → uses DefaultAzureCredential (az login etc.)

    # --- Alternative: hardcoded API key (uncomment to test) ---
    # search = AzureFoundryWebSearch(
    #     project_endpoint=project_endpoint,
    #     agent_name=agent_name,
    #     agent_version=agent_version,
    #     api_key="your-api-key-from-azure-portal",
    # )

    # --- Alternative: hardcoded Service Principal (uncomment to test) ---
    # search = AzureFoundryWebSearch(
    #     project_endpoint=project_endpoint,
    #     agent_name=agent_name,
    #     agent_version=agent_version,
    #     tenant_id="your-tenant-id",
    #     client_id="your-client-id",
    #     client_secret="your-client-secret",
    # )

    # --- Alternative: DefaultAzureCredential only (uncomment to test) ---
    # Requires: brew install azure-cli && az login
    # search = AzureFoundryWebSearch(
    #     project_endpoint=project_endpoint,
    #     agent_name=agent_name,
    #     agent_version=agent_version,
    # )

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
        if "content_filter" in nonfatal_notice:
            console.print(
                Panel(
                    "Azure hat den Prompt per Content Filter blockiert. "
                    "Das ist kein Auth- oder Endpoint-Fehler. "
                    "Setze fuer einen Smoke-Test z. B. "
                    "AZURE_WEB_SEARCH_TEST_QUERY='Tell me what you can help with.' "
                    "oder eine einfache aktuelle Nachrichtenfrage. "
                    "Wenn selbst harmlose Queries blockiert werden, pruefe auch die Agent-"
                    "Instruktionen im Foundry-Portal.",
                    title="Hint",
                    expand=False,
                )
            )
    console.print(
        Panel(
            result.get("answer") or "<empty answer>",
            title="Azure Foundry Web Search answer",
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
    console.print(
        Panel(
            "\n".join(
                [
                    f"Agent: {agent_name}" + (f" v{agent_version}" if agent_version else ""),
                    f"Prompt tokens: {result.get('_prompt_tokens', 0)}",
                    f"Completion tokens: {result.get('_completion_tokens', 0)}",
                ]
            ),
            title="Metadata",
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
        AzureFoundryWebSearchAPIError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
