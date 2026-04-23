"""Minimal native Azure OpenAI ``web_search`` smoke test.

Use this when you want to verify the search path via
``AzureOpenAIWebSearch`` -- the native Azure OpenAI Responses-API
``web_search`` tool.  This is the simplest of all three Azure search
paths, because it does NOT require a Foundry project, a Foundry agent,
or a separate Bing resource.

===============================================================================
WHAT THIS SCRIPT CREATES IN AZURE: NOTHING.
===============================================================================
Running this script makes a single HTTPS request against the Azure
OpenAI Responses API with a body of the form::

    {
        "model": "<your-deployment-name>",
        "input": "<the query>",
        "tools": [{"type": "web_search"}],
        "include": ["web_search_call.action.sources"]
    }

Azure handles the search server-side inside the model (internally via
the Bing backend) and returns the grounded answer plus the cited URLs.
No new Azure resource, no agent, no project, no connection -- nothing
persistent -- is provisioned by this script.  The only side effects are:

- one entry in your Azure cost analysis under "Azure OpenAI" (a few
  cents per call -- token costs are higher than a normal LLM call
  because Azure injects the fetched page contents into the prompt)
- an internal Bing-transaction counter (not exposed in the portal as
  long as you do not own a separate Grounding-with-Bing resource)

If your Azure subscription is allowed to use the Bing backend
(true for pay-as-you-go and most enterprise subscriptions; possibly
NOT true for pure sponsored/trial subscriptions), this is the
absolutely shortest path from "I have an Azure OpenAI deployment" to
"I have a working web search".

===============================================================================
PREREQUISITES (only ONE thing -- the LLM smoke test must already pass)
===============================================================================

This script reuses exactly the same Azure OpenAI resource + model
deployment as ``test_llm.py``.  If you have not yet done the one-time
Azure setup, do it FIRST by following the docstring at the top of
``test_llm.py`` (sections "ONE-TIME AZURE SETUP" steps 1-5) and run
``test_llm.py`` once.  Once that prints a green answer panel, this
script is ready to run.

In short:
- IF ``test_llm.py`` works -> you have everything you need here.
- NO additional Azure components must be created.
- NO ``.env`` variables beyond those of ``test_llm.py`` are required.

The deployed model MUST support the ``web_search`` tool.  As of writing
that is true for ``gpt-4.1``, ``gpt-4.1-mini``, and ``gpt-4o`` family
deployments.  Older or specialised deployments may not.

===============================================================================
RELATED SMOKE TESTS
===============================================================================

- ``test_llm.py``                -- AzureOpenAILLM (LLM only; prerequisite)
- ``test_foundry_web_search.py`` -- AzureFoundryWebSearch (Foundry agent
                                     path, needs a Foundry project +
                                     Web-Search agent)
- ``test_bing_search.py``        -- AzureFoundryBingSearch (Bing-grounded
                                     agent, needs a separate Bing resource
                                     + Foundry connection + agent)

===============================================================================
WHAT THIS SCRIPT CHECKS
===============================================================================
- Azure OpenAI endpoint is correct
- API key works (or AAD token provider)
- the deployment supports the ``web_search`` tool
- the subscription is allowed to use the Bing backend
- a single search call returns an answer and citations

===============================================================================
ENVIRONMENT VARIABLES
===============================================================================

Required (already set in .env if test_llm.py passed):
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_DEPLOYMENT_NAME   (must be a model with web_search support,
                                  e.g. gpt-4.1 / gpt-4.1-mini / gpt-4o)

Optional:
- AZURE_OPENAI_WEB_SEARCH_TEST_QUERY     (override the default demo query)
- AZURE_OPENAI_WEB_SEARCH_USER_COUNTRY   (ISO country code for user_location,
                                          default: DE)
- AZURE_OPENAI_WEB_SEARCH_DOMAIN_FILTER  (comma-separated allow/deny list, e.g.
                                          "tagesschau.de,bundestag.de,-reddit.com")
- INQTRIX_LOG_ENABLED
- INQTRIX_LOG_LEVEL
- INQTRIX_LOG_CONSOLE

===============================================================================
TROUBLESHOOTING
===============================================================================
- ``tool 'web_search' is not supported`` -> the deployment's underlying
  model does not support the web_search tool.  Deploy gpt-4.1 (or
  gpt-4o) and point AZURE_OPENAI_DEPLOYMENT_NAME at that deployment.
- Errors mentioning ``Bing``, ``grounding``, ``subscription``,
  ``not eligible`` -> your Azure subscription is not allowed to use
  the Bing backend.  Upgrade the trial to Pay-as-you-go in
  "Cost Management + Billing" (your remaining $200 credit is
  preserved).
- ``content_filter`` -> the prompt was blocked by Azure's content
  filter.  Override AZURE_OPENAI_WEB_SEARCH_TEST_QUERY with a simpler,
  current-events style query.  The script also prints a hint panel in
  this case.
- High prompt tokens (~20-30K) are NORMAL: Azure injects fetched page
  contents into the prompt to ground the answer.

Run with::

    uv run python examples/provider_stacks/azure_smoke_tests/test_openai_web_search.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from inqtrix import AzureOpenAIWebSearch, AzureOpenAIWebSearchAPIError
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
    "AZURE_OPENAI_WEB_SEARCH_TEST_QUERY",
    "Was sind heute die Top-Schlagzeilen bei tagesschau.de und zdfheute.de?",
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _parse_domain_filter(raw: str) -> list[str] | None:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def main() -> None:
    azure_endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
    azure_api_key = _require_env("AZURE_OPENAI_API_KEY")
    azure_deployment = _require_env("AZURE_OPENAI_DEPLOYMENT_NAME")

    user_country = os.getenv("AZURE_OPENAI_WEB_SEARCH_USER_COUNTRY", "DE").strip()
    domain_filter = _parse_domain_filter(
        os.getenv("AZURE_OPENAI_WEB_SEARCH_DOMAIN_FILTER", "")
    )

    user_location = (
        {"type": "approximate", "country": user_country} if user_country else None
    )

    search = AzureOpenAIWebSearch(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        default_model=azure_deployment,
        user_location=user_location,
        # tool_choice="auto" is the default; set "required" to force a
        # web_search call even for trivial questions.
    )

    result = search.search(
        QUERY,
        domain_filter=domain_filter,
    )

    nonfatal_notice = search.consume_nonfatal_notice()
    citations = result.get("citations") or []
    citations_text = (
        "\n".join(f"- {url}" for url in citations) or "<no citations returned>"
    )

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
                    "AZURE_OPENAI_WEB_SEARCH_TEST_QUERY='Wer ist der aktuelle "
                    "Bundeskanzler von Deutschland?' oder eine einfache "
                    "aktuelle Nachrichtenfrage.",
                    title="Hint",
                    expand=False,
                )
            )
    console.print(
        Panel(
            result.get("answer") or "<empty answer>",
            title="Azure OpenAI web_search answer",
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
    metadata_lines = [
        f"Deployment: {azure_deployment}",
        f"User location: {user_country or '<none>'}",
        f"Domain filter: {','.join(domain_filter) if domain_filter else '<none>'}",
        f"Prompt tokens: {result.get('_prompt_tokens', 0)}",
        f"Completion tokens: {result.get('_completion_tokens', 0)}",
    ]
    console.print(
        Panel(
            "\n".join(metadata_lines),
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
        AzureOpenAIWebSearchAPIError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
