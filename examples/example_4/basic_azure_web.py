"""Azure OpenAI v1 + Azure Foundry Web Search example.

Architecture
------------
This example combines two independent Azure components directly:

1. **Sprachmodell / Reasoning** -- via ``AzureOpenAILLM``
2. **Websuche / Grounding** -- via ``AzureFoundryWebSearch``

Recommended validation order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before using this combined example, validate both components separately:

- ``examples/example_4/basic_test_component_azure_llm.py``
- ``examples/example_4/basic_test_component_azure_web_search.py``
- ``examples/example_4/basic_test_component_azure_bing_search.py``

Why these smoke tests?

- ``basic_test_component_azure_llm.py`` validates the Azure OpenAI reasoning path.
- ``basic_test_component_azure_web_search.py`` validates the newer Foundry Web Search
  tool path (Responses API) used by THIS file.
- ``basic_test_component_azure_bing_search.py`` validates the classic Bing Grounding
  search-agent path.  It is useful for comparison, but it is NOT the
  search provider used by this file.

This combined script itself uses:
``AzureOpenAILLM`` + ``AzureFoundryWebSearch``.

This example calls two independent Azure services directly:

1. **Azure OpenAI v1 Chat Completions API** -- via ``AzureOpenAILLM``,
   using the official ``openai`` SDK's generic ``OpenAI`` client
   against the Azure ``/openai/v1/`` endpoint.  Used for all reasoning
   tasks: question classification, query planning, evidence evaluation,
   parallel summarisation, and final answer synthesis.

2. **Azure AI Foundry Web Search Agent** -- via
   ``AzureFoundryWebSearch``, using the OpenAI ``responses`` API
   against the project-scoped Foundry ``/openai/v1/`` endpoint.
   A pre-created Foundry agent with the Web Search tool performs web
   searches.  Used ONLY for web search -- not for reasoning.

Why two separate providers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Even though both components ultimately use OpenAI-compatible clients,
the Web Search agent should not be reused for reasoning because:

- The agent's instructions and attached tool are specialized for web search.
- The classify/plan/evaluate/answer nodes need structured reasoning
  (e.g. JSON query lists, DECISION keywords) that agent-side web search
  would corrupt.
- Search behaviour is mostly defined by the agent and its tool config,
  not by the research graph's reasoning prompts.

Comparison with Azure Foundry Bing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+-------------------------+-----------------------------+---------------------------+
| Aspect                  | Azure Foundry Web Search    | Azure Foundry Bing        |
+=========================+=============================+===========================+
| Search path             | Responses API               | Agent Thread/Run API      |
|                         | with agent_reference        | with pre-created agent ID |
+-------------------------+-----------------------------+---------------------------+
| Agent reference         | name + optional version     | opaque agent_id           |
+-------------------------+-----------------------------+---------------------------+
| Auth options            | API key or Entra ID         | Entra ID / credential     |
+-------------------------+-----------------------------+---------------------------+
| Domain filtering        | Best-effort via site:       | Best-effort via site:     |
|                         | operators in query text     | operators in query text   |
+-------------------------+-----------------------------+---------------------------+
| Recency/language        | Runtime hints in prompt     | Runtime hints in prompt   |
+-------------------------+-----------------------------+---------------------------+
| Related questions       | Not available               | Not available             |
+-------------------------+-----------------------------+---------------------------+
| search_mode             | Ignored                     | Ignored                   |
+-------------------------+-----------------------------+---------------------------+
| search_context_size     | Ignored at runtime          | Ignored at runtime        |
|                         | (agent config decides)      | (agent config decides)    |
+-------------------------+-----------------------------+---------------------------+

Prerequisites (one-time Azure Portal setup)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **AI Foundry project:** Azure Portal > "Azure AI Foundry" > create
   a Hub (if needed) > create a Project within the Hub > deploy a
   model (e.g. gpt-4o or gpt-5-mini) under "Models + Endpoints".

2. **Create a Web Search agent:** In the Foundry project, create an
   agent with the Web Search tool enabled.  Note the **Agent Name** and,
   if relevant, the **Agent Version**.

3. **Authentication:** Choose one:
   - **API key** from Project Settings > Keys & Endpoint
   - **Service Principal** with project access
   - **DefaultAzureCredential** (local ``az login`` / Managed Identity)

Required environment variables (in .env or process env):
- AZURE_OPENAI_API_KEY          (for LLM -- API-key auth)
- AZURE_OPENAI_ENDPOINT         (e.g. https://mein-openai.openai.azure.com/)
- AZURE_OPENAI_DEPLOYMENT_NAME  (the deployment name, NOT the model name)
- AZURE_AI_PROJECT_ENDPOINT     (e.g. https://mein-projekt.services.ai.azure.com/api/projects/mein-projekt)
- WEB_SEARCH_AGENT_NAME         (agent name from Foundry)

Optional:
- WEB_SEARCH_AGENT_VERSION      (specific agent version; if empty, latest/default)
- AZURE_AI_PROJECT_API_KEY      (simplest auth path for web search)
- AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET  (Service Principal)
- AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME  (cheaper deployment for summarisation)
- AZURE_WEB_STACK_QUESTION      (override the default demo question)

Run with::

    uv sync
    uv run python examples/example_4/basic_azure_web.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from inqtrix import (
    AgentConfig,
    AzureFoundryWebSearch,
    AzureOpenAILLM,
    ResearchAgent,
)


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
    agent_name = _require_env("WEB_SEARCH_AGENT_NAME")
    agent_version = os.getenv("WEB_SEARCH_AGENT_VERSION", "")

    azure_summarize_deployment = os.environ.get(
        "AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME", ""
    ).strip()
    azure_default_max_tokens = int(
        os.environ.get("AZURE_OPENAI_DEFAULT_MAX_TOKENS", "1024").strip() or "4096"
    )

    # -- LLM Provider ---------------------------------------------------------
    #
    # AzureOpenAILLM is the same component that is smoke-tested in:
    #   examples/example_4/basic_test_component_azure_llm.py
    # Test that file first if the combined setup fails.
    #
    # Option A: API Key (default)
    azure_api_key = _require_env("AZURE_OPENAI_API_KEY")

    llm = AzureOpenAILLM(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        default_model=azure_deployment,
        default_max_tokens=16384,
        summarize_max_tokens=16384
        # summarize_model=azure_summarize_deployment,
    )

    # -- Option B: Service Principal (Entra ID) --------------------------------
    #
    # from azure.identity import ClientSecretCredential, get_bearer_token_provider
    #
    # credential = ClientSecretCredential(
    #     tenant_id=os.environ["AZURE_TENANT_ID"],
    #     client_id=os.environ["AZURE_CLIENT_ID"],
    #     client_secret=os.environ["AZURE_CLIENT_SECRET"],
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

    # -- Option C: Existing token provider -------------------------------------
    #
    # llm = AzureOpenAILLM(
    #     azure_endpoint=azure_endpoint,
    #     azure_ad_token_provider=existing_token_provider,
    #     default_model=azure_deployment,
    #     summarize_model=azure_summarize_deployment,
    # )

    # -- Search Provider -------------------------------------------------------
    #
    # AzureFoundryWebSearch is the same component that is smoke-tested in:
    #   examples/example_4/basic_test_component_azure_web_search.py
    # Test that file first if the combined setup fails.
    #
    # This provider uses the Foundry Responses API with agent_reference.
    # The research algorithm passes the standard SearchProvider parameters,
    # but this backend only supports some of them directly:
    #
    # - domain_filter       → best-effort via site:/-site: operators
    # - recency_filter      → best-effort prompt hint
    # - language_filter     → best-effort prompt hint
    # - search_context_size → ignored at runtime
    # - search_mode         → ignored
    # - return_related      → ignored
    #
    # Option A: Auto-resolve auth from environment (recommended)
    # Priority:
    #   1. AZURE_AI_PROJECT_API_KEY
    #   2. AZURE_TENANT_ID + AZURE_CLIENT_ID + AZURE_CLIENT_SECRET
    #   3. DefaultAzureCredential
    #
    web_api_key = os.getenv("AZURE_AI_PROJECT_API_KEY", "").strip()
    tenant_id = os.getenv("AZURE_TENANT_ID", "").strip()
    client_id = os.getenv("AZURE_CLIENT_ID", "").strip()
    client_secret = os.getenv("AZURE_CLIENT_SECRET", "").strip()

    search_kwargs = {
        "project_endpoint": project_endpoint,
        "agent_name": agent_name,
        "agent_version": agent_version,
    }
    if web_api_key:
        search_kwargs["api_key"] = web_api_key
    elif tenant_id and client_id and client_secret:
        search_kwargs["tenant_id"] = tenant_id
        search_kwargs["client_id"] = client_id
        search_kwargs["client_secret"] = client_secret

    search = AzureFoundryWebSearch(**search_kwargs)

    # -- Option B: Hardcoded API key -------------------------------------------
    #
    # search = AzureFoundryWebSearch(
    #     project_endpoint=project_endpoint,
    #     agent_name=agent_name,
    #     agent_version=agent_version,
    #     api_key="your-api-key-from-azure-portal",
    # )

    # -- Option C: Hardcoded Service Principal ---------------------------------
    #
    # search = AzureFoundryWebSearch(
    #     project_endpoint=project_endpoint,
    #     agent_name=agent_name,
    #     agent_version=agent_version,
    #     tenant_id="your-tenant-id",
    #     client_id="your-client-id",
    #     client_secret="your-client-secret",
    # )

    # -- Option D: DefaultAzureCredential only ---------------------------------
    #
    # Requires: brew install azure-cli && az login
    # search = AzureFoundryWebSearch(
    #     project_endpoint=project_endpoint,
    #     agent_name=agent_name,
    #     agent_version=agent_version,
    # )

    # -- Agent -----------------------------------------------------------------
    agent = ResearchAgent(AgentConfig(
        llm=llm,
        search=search,
        max_rounds=4,
        max_total_seconds=300,
        confidence_stop=9
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
