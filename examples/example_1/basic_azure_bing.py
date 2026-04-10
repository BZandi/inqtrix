"""Azure OpenAI v1 + Azure Foundry Bing Search example.

Architecture
------------
This example calls two independent Azure services directly:

1. **Azure OpenAI v1 Chat Completions API** -- via ``AzureOpenAILLM``,
   using the official ``openai`` SDK's generic ``OpenAI`` client
   against the Azure ``/openai/v1/`` endpoint.  Used for all reasoning
   tasks: question classification, query planning, evidence evaluation,
   parallel summarisation, and final answer synthesis.

2. **Azure AI Foundry Agent with Bing Grounding** -- via
   ``AzureFoundryBingSearch``, using the ``azure-ai-projects`` SDK.
   A pre-created Foundry Agent with Bing Grounding performs web
   searches.  Used ONLY for web search -- not for reasoning.

Why two separate providers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike ``basic_azure_perplexity.py`` (where Perplexity handles search
via a simple API call), Azure Foundry Bing uses an AI agent that
autonomously invokes Bing.  This agent cannot be reused for LLM
reasoning because:

- BingGroundingTool is permanently attached -- the agent ALWAYS
  performs web search, even for pure reasoning tasks.
- The classify/plan/evaluate/answer nodes need structured reasoning
  (e.g. JSON query lists, DECISION keywords) that web search would
  corrupt.
- Search parameters (freshness, market, count) are fixed at agent
  creation time -- they cannot be changed per call.

A single unified endpoint is therefore not possible.  This example
uses Azure OpenAI for reasoning and a separate Foundry Bing agent
for search.

Comparison with Perplexity
~~~~~~~~~~~~~~~~~~~~~~~~~~
+-------------------------+-----------------------------+---------------------------+
| Aspect                  | Perplexity (Sonar API)      | Azure Foundry Bing        |
+=========================+=============================+===========================+
| Search trigger          | Explicit API call with      | LLM agent decides         |
|                         | structured parameters       | autonomously from prompt  |
+-------------------------+-----------------------------+---------------------------+
| Parameter control       | Per call (recency, lang,    | Fixed at agent creation   |
|                         | domain, academic mode)      | (freshness, market, count)|
+-------------------------+-----------------------------+---------------------------+
| Domain filtering        | Native API parameter        | Best-effort via site:     |
|                         |                             | operators in query text   |
+-------------------------+-----------------------------+---------------------------+
| Related questions       | Returned by API             | Not available             |
+-------------------------+-----------------------------+---------------------------+
| Academic mode           | Supported (search_mode)     | Not available             |
+-------------------------+-----------------------------+---------------------------+
| Response type           | LLM-summarised with         | LLM-generated answer with |
|                         | structured citations        | Bing grounding annotations|
+-------------------------+-----------------------------+---------------------------+
| Setup complexity        | API key only                | Azure Portal: Bing        |
|                         |                             | resource + Foundry project|
|                         |                             | + model deployment +      |
|                         |                             | connection + agent        |
+-------------------------+-----------------------------+---------------------------+

Prerequisites (one-time Azure Portal setup)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Bing resource:** Azure Portal > "Ressource erstellen" > search
   for "Grounding with Bing Search" > create in your subscription.

2. **AI Foundry project:** Azure Portal > "Azure AI Foundry" > create
   a Hub (if needed) > create a Project within the Hub > deploy a
   model (e.g. gpt-4o) under "Models + Endpoints".

3. **Connect Bing:** In the Foundry project, go to "Connected
   Resources" / "Connections" > add the Bing resource from step 1.
   Note the **Connection ID** (format:
   ``/subscriptions/.../connections/my-bing-connection``).

4. **Create the search agent:** Either via Azure Portal / CLI, or
   programmatically with ``AzureFoundryBingSearch.create_agent()``
   (see commented Option B below).  Note the **Agent ID**.

5. **(Optional) Service Principal:** In Entra ID (Azure AD), create
   an App Registration + Client Secret.  Assign the role "Azure AI
   Developer" to the service principal on the Foundry project scope.

Required environment variables (in .env or process env):
- AZURE_OPENAI_API_KEY          (for LLM -- API-key auth)
- AZURE_OPENAI_ENDPOINT         (e.g. https://mein-openai.openai.azure.com/)
- AZURE_OPENAI_DEPLOYMENT_NAME  (the deployment name, NOT the model name)
- AZURE_AI_PROJECT_ENDPOINT     (e.g. https://mein-projekt.services.ai.azure.com/api)
- BING_AGENT_ID                 (agent ID from step 4 above)

Optional:
- AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME  (cheaper deployment for summarisation)
- AZURE_TENANT_ID, AZURE_CLIENT_ID_DEV, AZURE_CLIENT_SECRET_DEV  (Service Principal)
- BING_PROJECT_CONNECTION_ID    (only needed for create_agent() path)

Run with::

    uv sync
    uv run python examples/example_1/basic_azure_bing.py
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from inqtrix import (
    AgentConfig,
    AzureOpenAILLM,
    AzureFoundryBingSearch,
    ResearchAgent,
)

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


load_dotenv()

# -- Logging ------------------------------------------------------------------
from inqtrix.logging_config import configure_logging

_log_path = configure_logging(
    enabled=os.getenv("INQTRIX_LOG_ENABLED", "").lower() == "true",
    level=os.getenv("INQTRIX_LOG_LEVEL", "INFO"),
    console=os.getenv("INQTRIX_LOG_CONSOLE", "").lower() == "true",
)
if _log_path:
    print(f"Logging to {_log_path}")

QUESTION = "Was ist der aktuelle Stand der GKV-Reform?"


def _print_result(result) -> None:
    """Pretty-print a ResearchResult."""
    metrics_line = (
        f"Confidence: {result.metrics.confidence}/10  |  "
        f"Sources: {result.metrics.total_citations}  |  "
        f"Rounds: {result.metrics.rounds}"
    )
    console = Console()
    console.print(Markdown(result.answer))
    console.print()
    console.print(Panel(metrics_line, title="Metrics", expand=False))


# -- Output mode --------------------------------------------------------------
USE_STREAMING = True
INCLUDE_PROGRESS = True


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    azure_endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
    azure_deployment = _require_env("AZURE_OPENAI_DEPLOYMENT_NAME")
    project_endpoint = _require_env("AZURE_AI_PROJECT_ENDPOINT")
    bing_agent_id = _require_env("BING_AGENT_ID")

    azure_summarize_deployment = os.environ.get(
        "AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME", ""
    ).strip()

    # -- LLM Provider ---------------------------------------------------------
    #
    # AzureOpenAILLM -- identical to basic_azure_perplexity.py.
    # See that example for detailed auth documentation.
    #
    # Option A: API Key (default)
    azure_api_key = _require_env("AZURE_OPENAI_API_KEY")

    llm = AzureOpenAILLM(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        default_model=azure_deployment,
        summarize_model=azure_summarize_deployment,
    )

    # -- Option B: Service Principal (Entra ID) --------------------------------
    #
    # from azure.identity import ClientSecretCredential, get_bearer_token_provider
    #
    # credential = ClientSecretCredential(
    #     tenant_id=os.environ["AZURE_TENANT_ID"],
    #     client_id=os.environ["AZURE_CLIENT_ID_DEV"],
    #     client_secret=os.environ["AZURE_CLIENT_SECRET_DEV"],
    # )
    # token_provider = get_bearer_token_provider(
    #     credential,
    #     "https://ai.azure.com/.default",
    # )
    #
    # llm = AzureOpenAILLM(
    #     azure_endpoint=azure_endpoint,
    #     azure_ad_token_provider=token_provider,
    #     default_model=azure_deployment,
    #     summarize_model=azure_summarize_deployment,
    # )

    # -- Search Provider -------------------------------------------------------
    #
    # AzureFoundryBingSearch replaces PerplexitySearch from the
    # basic_azure_perplexity.py example.  Key differences:
    #
    # - PerplexitySearch: one API call per search, all parameters per call.
    # - AzureFoundryBingSearch: reuses a pre-created Azure Foundry Agent
    #   with Bing Grounding.  Each search() call creates a Thread +
    #   Message + Run, the agent autonomously queries Bing, and returns
    #   a grounded answer with source URLs as annotations.
    #
    # Search parameters like freshness, market, and result count are
    # fixed when the agent is created.  At runtime, the provider can
    # only influence the search through:
    #   - The query text itself (primary control)
    #   - site:/−site: operators for domain filtering (best-effort)
    #   - additional_instructions hints for recency/language (non-deterministic)
    #
    # Parameters that have no Bing equivalent (search_mode="academic",
    # return_related, search_context_size) are silently ignored, following
    # the same pattern as BraveSearch.
    #
    # Option A: Use an existing agent (recommended for production).
    # The agent was created beforehand via Azure Portal, CLI, or the
    # create_agent() classmethod.  Only the agent ID is needed here.
    #
    search = AzureFoundryBingSearch(
        project_endpoint=project_endpoint,
        agent_id=bing_agent_id,
        # For Service Principal auth, pass credential or individual fields:
        # tenant_id=os.environ["AZURE_TENANT_ID"],
        # client_id=os.environ["AZURE_CLIENT_ID_DEV"],
        # client_secret=os.environ["AZURE_CLIENT_SECRET_DEV"],
    )

    # -- Option B: Create the agent on the fly (one-time setup) ----------------
    #
    # Takes 2-5 seconds.  The returned provider instance already has the
    # agent_id set.  Save the agent ID (printed below) for future runs
    # so you can use Option A instead of re-creating the agent each time.
    #
    # The search parameters below are fixed once the agent is created:
    #   - market:    Bing market for regional results (e.g. "de-DE")
    #   - set_lang:  Response language preference (e.g. "de")
    #   - freshness: Content recency filter ("Day", "Week", "Month")
    #   - count:     Number of Bing results to ground on (1-50)
    #
    # search = AzureFoundryBingSearch.create_agent(
    #     project_endpoint=project_endpoint,
    #     bing_connection_id=os.environ["BING_PROJECT_CONNECTION_ID"],
    #     model="gpt-4o",
    #     market="de-DE",
    #     set_lang="de",
    #     freshness="Week",
    #     count=10,
    #     # credential or tenant_id/client_id/client_secret
    # )
    # print(f"Agent created with ID: {search._agent_id}  -- save this!")

    # -- Agent -----------------------------------------------------------------
    agent = ResearchAgent(AgentConfig(
        llm=llm,
        search=search,
        max_rounds=4,
        max_total_seconds=300,
    ))

    # -- Run -------------------------------------------------------------------
    if USE_STREAMING:
        answer_buf: list[str] = []
        in_answer = False
        for chunk in agent.stream(QUESTION, include_progress=INCLUDE_PROGRESS):
            if not in_answer and chunk == "---\n":
                in_answer = True
                continue
            if in_answer:
                answer_buf.append(chunk)
            else:
                print(chunk, end="", flush=True)

        full_answer = "".join(answer_buf)
        if full_answer:
            console = Console()
            print()
            console.print(Markdown(full_answer))
    else:
        result = agent.research(QUESTION)
        _print_result(result)


if __name__ == "__main__":
    main()
