"""Azure OpenAI v1 + Azure Foundry Web Search example  --  VARIANTE B (Foundry Agent).

==============================================================================
WHICH AZURE WEB-SEARCH VARIANT THIS FILE COVERS
==============================================================================
This script uses the Azure FOUNDRY Web Search Agent path (provider class
``AzureFoundryWebSearch``, smoke-tested in ``test_foundry_web_search.py``).  A
pre-created Foundry agent equipped with the Web Search tool is invoked
via the project-scoped Responses API and ``agent_reference``.

The sibling stack file ``azure_openai_web_search.py`` covers a different
variant: the NATIVE Azure OpenAI Responses-API ``web_search`` tool
(provider class ``AzureOpenAIWebSearch``, smoke-tested in
``test_openai_web_search.py``).  That variant needs no Foundry project,
no agent, and no separate Bing resource -- only an Azure OpenAI
deployment.

Quick way to verify which path a stack file uses: check the Search-Provider
import at the top of the file.

- imports ``AzureFoundryWebSearch`` -> Variante B (Foundry agent, this file)
- imports ``AzureOpenAIWebSearch``  -> Variante C (native, see
                                       ``azure_openai_web_search.py``)

==============================================================================
WHICH SMOKE TESTS MUST PASS BEFORE RUNNING THIS FILE
==============================================================================
Run, in order, and only proceed if each prints a green panel:

1. ``examples/provider_stacks/azure_smoke_tests/test_llm.py``
   -- proves the Azure OpenAI reasoning path (this file uses it for
   classify / plan / summarize / evaluate / answer).
2. ``examples/provider_stacks/azure_smoke_tests/test_foundry_web_search.py``
   -- proves the Azure Foundry Web Search Agent path (this file uses
   it for every web-search call).

The third smoke test in the folder is for a DIFFERENT path:

- ``examples/provider_stacks/azure_smoke_tests/test_bing_search.py``
  -- validates the classic Bing-grounded agent (Variante A) and is
  NOT used by this file.  Skip it unless you want to compare paths.

Architecture
------------
This example combines two independent Azure components directly:

1. **Sprachmodell / Reasoning** -- via ``AzureOpenAILLM``
2. **Websuche / Grounding** -- via ``AzureFoundryWebSearch``

This combined script itself uses:
``AzureOpenAILLM`` + ``AzureFoundryWebSearch``.
It does NOT use the native Azure OpenAI ``web_search`` tool path from
``AzureOpenAIWebSearch`` (see "Variante C" note at the bottom).

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
| Search path             | Responses API               | Responses API             |
|                         | with agent_reference        | with Bing-grounded agent  |
+-------------------------+-----------------------------+---------------------------+
| Agent reference         | name + optional version     | name + optional version   |
+-------------------------+-----------------------------+---------------------------+
| Auth options            | API key or Entra ID         | API key or Entra ID       |
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
- AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME  (cheaper deployment for summarisation;
                                            see "Summarize behaviour" below)
- AZURE_WEB_STACK_QUESTION      (override the default demo question)

Summarize behaviour
~~~~~~~~~~~~~~~~~~~
The LLM constructor in this script reads
``AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME`` and forwards it as
``summarize_model``:

- If set (e.g. ``gpt-4.1-mini``), all parallel result-summarisation
  and claim-extraction calls use that cheaper deployment, while
  classify / plan / evaluate / answer keep using the strong
  ``default_model``.  Per Inqtrix run this typically saves ~5x on
  the summarize-role token costs.
- If empty or unset, summarisation falls back to the default
  ``default_model`` (no cost benefit, no behavioural change).

The same env-driven pattern works for ``classify_model`` and
``evaluate_model`` if you want even finer-grained role splitting --
those are not wired up here on purpose, to keep the example minimal.

Variante C alternative (native AzureOpenAIWebSearch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to run Inqtrix with the NATIVE Azure OpenAI ``web_search``
tool path instead (no Foundry project, no agent, no separate
``AZURE_AI_PROJECT_ENDPOINT`` / ``WEB_SEARCH_AGENT_NAME``), use the
sibling stack file:

- ``examples/provider_stacks/azure_openai_web_search.py``

That file mirrors this one's structure but instantiates
``AzureOpenAIWebSearch`` against the same Azure OpenAI deployment as
the LLM.  YAML configuration for the same variant is documented in
``README.md`` lines 453-480.

Terminal rendering
------------------
The agent returns Markdown, which looks good in a chat UI but is hard
to read as raw text in a terminal.  This example uses ``rich`` to
render Markdown with colours, formatted headers, bullet lists, and
clickable links (in terminals that support OSC 8 hyperlinks such as
iTerm2, Windows Terminal, or GNOME Terminal).

``rich`` is a core dependency and always available after::

    uv sync

Run with::

    uv sync
    uv run python examples/provider_stacks/azure_foundry_web_search.py
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

# ── Logging ──────────────────────────────────────────────────────────
# File-based logging with automatic secret redaction.  Disabled by
# default — enable via environment variables to keep terminal clean:
#
#   INQTRIX_LOG_ENABLED=true   — write logs to logs/inqtrix_*.log
#   INQTRIX_LOG_LEVEL=DEBUG    — DEBUG / INFO / WARNING (default: INFO)
#   INQTRIX_LOG_CONSOLE=true   — also print WARNING+ to stderr
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


# ── Output mode ──────────────────────────────────────────────────────
# True  → streaming (live progress messages + word-by-word answer)
# False → blocking  (waits for the full result, then prints at once)
USE_STREAMING = True

# Only relevant when USE_STREAMING is True:
# True  → show intermediate progress messages before the answer
#          e.g. "Analysiere Frage…", "Plane Suchanfragen (Runde 1/4)…"
# False → stream only the final answer text, no status updates
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

    # -- LLM Provider ---------------------------------------------------------
    #
    # AzureOpenAILLM is the same component that is smoke-tested in:
    #   examples/provider_stacks/azure_smoke_tests/test_llm.py
    # Test that file first if the combined setup fails.
    #
    # summarize_model is the cheap-deployment switch documented in the
    # module docstring under "Summarize behaviour".  Passing an empty
    # string is treated by AzureOpenAILLM as "use default_model" and
    # is therefore safe even when the env variable is unset.
    #
    # Option A: API Key (default)
    azure_api_key = _require_env("AZURE_OPENAI_API_KEY")

    llm = AzureOpenAILLM(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        default_model=azure_deployment,
        summarize_model=azure_summarize_deployment,
        default_max_tokens=16384,
        summarize_max_tokens=16384,
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
    #     "https://cognitiveservices.azure.com/.default",
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
    #   examples/provider_stacks/azure_smoke_tests/test_foundry_web_search.py
    # Test that file first if the combined setup fails.
    #
    # This provider uses the Foundry Responses API with agent_reference.
    # If you want the native Azure OpenAI tool path instead, instantiate
    # AzureOpenAIWebSearch (or set search_provider=azure_openai_web_search in YAML).
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

    # ── AgentConfig — all available options ──────────────────────────
    #
    # Only llm + search are required for explicit setup.  Every other
    # field has a sensible default.  Uncomment and change only what you
    # need.  If llm or search are omitted (None), they are auto-created
    # from environment variables / .env.
    agent = ResearchAgent(AgentConfig(
        # -- Providers (None = auto-create from env) --
        llm=llm,
        search=search,

        # -- Behaviour --
        max_rounds=4,                       # max research-loop iterations before forced stop
        # confidence threshold (1-10) — stop early when reached
        confidence_stop=9,
        # max context blocks retained across rounds (older ones pruned)
        max_context=12,
        first_round_queries=6,              # number of parallel search queries in first round
        answer_prompt_citations_max=60,     # max citation URLs forwarded to the answer-synthesis prompt
        # hard wall-clock deadline for the entire run (seconds).
        # GPT-4o through Azure is typically fast — 300s is usually enough.
        # Increase to 600 for very complex questions or slower deployments.
        max_total_seconds=300,
        max_question_length=10_000,         # reject questions longer than this (characters)

        # -- Timeouts (per individual LLM/search call, in seconds) --
        reasoning_timeout=120,              # timeout for reasoning / planning / answer calls
        search_timeout=60,                  # timeout for each web-search call
        summarize_timeout=60,               # timeout for each summarise / claim-extraction call

        # -- Risk escalation ──────────────────────────────────────────
        #
        # The risk score (0-10) is ALWAYS computed automatically by the
        # RiskScoringStrategy (regex-based on keywords like "Gesetz",
        # "Steuer", "Medizin", etc.).
        #
        # With AzureOpenAILLM these flags matter only if you configured
        # classify_model and/or evaluate_model above.  In that case,
        # high-risk questions can escalate those roles back up to the
        # stronger default_model.  Summarization always uses
        # summarize_model regardless of risk.
        high_risk_score_threshold=4,        # risk score ≥ this triggers high_risk = True
        high_risk_classify_escalate=True,   # if classify_model is set, high-risk uses default_model instead
        high_risk_evaluate_escalate=True,   # if evaluate_model is set, high-risk uses default_model instead

        # -- Search cache --
        search_cache_maxsize=256,           # max cached search results (LRU eviction)
        search_cache_ttl=3600,              # cache time-to-live in seconds

        # -- Strategies (None = use built-in defaults) ────────────────
        # Each strategy is a pluggable ABC.  Pass your own implementation
        # to override the default algorithm for that concern.
        # source_tiering=None,              # URL → quality-tier mapping
        # claim_extraction=None,            # extract structured claims from search results
        # claim_consolidation=None,         # deduplicate and verify claims across rounds
        # context_pruning=None,             # drop low-relevance context blocks
        # risk_scoring=None,                # score question risk (0-10)
        # stop_criteria=None,               # multi-signal heuristic: keep researching or stop?
    ))

    # ── Run ──────────────────────────────────────────────────────────
    if USE_STREAMING:
        # Collect chunks: progress lines go to stdout immediately,
        # answer text is buffered for a final rich render pass.
        answer_buf: list[str] = []
        in_answer = False
        for chunk in agent.stream(QUESTION, include_progress=INCLUDE_PROGRESS):
            if not in_answer and chunk == "---\n":
                in_answer = True
                continue
            if in_answer:
                answer_buf.append(chunk)
            else:
                # Progress lines — always printed raw
                print(chunk, end="", flush=True)

        full_answer = "".join(answer_buf)
        if full_answer:
            console = Console()
            print()  # newline after progress block
            console.print(Markdown(full_answer))
    else:
        result = agent.research(QUESTION)
        _print_result(result)


if __name__ == "__main__":
    main()
