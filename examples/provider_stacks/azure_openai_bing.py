"""Azure OpenAI v1 + Azure Foundry Bing Search example.

Architecture
------------
This example combines two independent Azure components directly:

1. **Sprachmodell / Reasoning** -- via ``AzureOpenAILLM``
2. **Websuche / Grounding** -- via ``AzureFoundryBingSearch``

Recommended validation order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before using this combined example, validate both components separately:

- ``examples/provider_stacks/azure_smoke_tests/test_llm.py``
- ``examples/provider_stacks/azure_smoke_tests/test_bing_search.py``
- ``examples/provider_stacks/azure_smoke_tests/test_foundry_web_search.py``

Related combined example:

- ``examples/provider_stacks/azure_foundry_web_search.py`` shows the same full stack,
  but with ``AzureFoundryWebSearch`` instead of ``AzureFoundryBingSearch``.
- ``examples/provider_stacks/azure_openai_web_search.py`` shows the native
  Azure OpenAI ``web_search`` tool path via ``AzureOpenAIWebSearch`` (no
  Foundry agent needed).

Why three smoke tests?

- ``azure_smoke_tests/test_llm.py`` validates the Azure OpenAI reasoning path.
- ``azure_smoke_tests/test_bing_search.py`` validates the classic Bing Grounding
    search-agent path used by THIS file.
- ``azure_smoke_tests/test_foundry_web_search.py`` validates the newer Foundry
    Web Search tool path (Responses API).  It is useful for comparison, but it is
    NOT the search provider used by this file.

This combined script itself uses the first two components only:
``AzureOpenAILLM`` + ``AzureFoundryBingSearch``.

This example calls two independent Azure services directly:

1. **Azure OpenAI v1 Chat Completions API** -- via ``AzureOpenAILLM``,
   using the official ``openai`` SDK's generic ``OpenAI`` client
   against the Azure ``/openai/v1/`` endpoint.  Used for all reasoning
   tasks: question classification, query planning, evidence evaluation,
   parallel summarisation, and final answer synthesis.

2. **Azure AI Foundry Bing-grounded agent** -- via
    ``AzureFoundryBingSearch``, using the project-scoped OpenAI
    ``responses`` API with ``agent_reference``.
    A pre-created Foundry agent with Bing Grounding performs web
    searches.  Used ONLY for web search -- not for reasoning.

Why two separate providers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike ``azure_openai_perplexity.py`` (where Perplexity handles search
via a simple API call), Azure Foundry Bing uses an AI agent that
autonomously invokes Bing.  This agent cannot be reused for LLM
reasoning because:

- The Bing Grounding tool is attached to the agent definition, so the
    runtime path is specialised for search rather than structured graph
    reasoning.
- The classify/plan/evaluate/answer nodes need structured reasoning
  (e.g. JSON query lists, DECISION keywords) that web search would
  corrupt.
- Search parameters (freshness, market, count) are fixed when the agent
    version is created -- they cannot be changed per call.

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
    (see commented Option B below).  Note the **Agent Name** and, if you
    pin one explicitly, the **Agent Version**.  A legacy **Agent ID** is
    still accepted for compatibility, but new setups should prefer the
    name-based path.

5. **(Optional) Service Principal:** In Entra ID (Azure AD), create
   an App Registration + Client Secret.  Assign the role "Azure AI
   Developer" to the service principal on the Foundry project scope.

Required environment variables (in .env or process env):
- AZURE_OPENAI_API_KEY          (for LLM -- API-key auth)
- AZURE_OPENAI_ENDPOINT         (e.g. https://mein-openai.openai.azure.com/)
- AZURE_OPENAI_DEPLOYMENT_NAME  (the deployment name, NOT the model name)
- AZURE_AI_PROJECT_ENDPOINT     (e.g. https://mein-projekt.services.ai.azure.com/api/projects/mein-projekt)
- BING_AGENT_NAME               (agent name from step 4 above)

Optional:
- BING_AGENT_VERSION            (specific agent version; if empty, latest/default)
- BING_AGENT_ID                 (legacy fallback if you only have the old opaque agent ID)
- AZURE_AI_PROJECT_API_KEY      (simplest auth path for Bing runtime calls)
- AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME  (cheaper deployment for summarisation)
- AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET  (Service Principal)
- BING_PROJECT_CONNECTION_ID    (only needed for create_agent() path)

Run with::

    uv sync
    uv run python examples/provider_stacks/azure_openai_bing.py
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


def _optional_env(name: str) -> str:
    return os.environ.get(name, "").strip()


def main() -> None:
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
    # AzureOpenAILLM is the same component that is smoke-tested in:
    #   examples/provider_stacks/azure_smoke_tests/test_llm.py
    # Test that file first if the combined setup fails.
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
    #
    # The provider builds the ClientSecretCredential and token provider
    # internally — no azure.identity imports needed in your code.
    #
    # llm = AzureOpenAILLM(
    #     azure_endpoint=azure_endpoint,
    #     tenant_id=os.environ["AZURE_TENANT_ID"],
    #     client_id=os.environ["AZURE_CLIENT_ID"],
    #     client_secret=os.environ["AZURE_CLIENT_SECRET"],
    #     default_model=azure_deployment,
    #     summarize_model=azure_summarize_deployment,
    # )

    # -- Option C: Service Principal — manual token provider ------------------
    #
    # Use this if you need custom credential settings (token caching,
    # regional authority, additional scopes, etc.).
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

    # -- Option D: Existing token provider -------------------------------------
    #
    # If you already created a bearer-token provider elsewhere in your code,
    # pass it directly to the constructor.  This is useful in larger
    # Baukasten setups where auth is centralised.
    #
    # llm = AzureOpenAILLM(
    #     azure_endpoint=azure_endpoint,
    #     azure_ad_token_provider=existing_token_provider,
    #     default_model=azure_deployment,
    #     summarize_model=azure_summarize_deployment,
    # )

    # -- Option E: DefaultAzureCredential (local dev / Managed Identity) ------
    #
    # from azure.identity import DefaultAzureCredential
    #
    # llm = AzureOpenAILLM(
    #     azure_endpoint=azure_endpoint,
    #     credential=DefaultAzureCredential(),
    #     default_model=azure_deployment,
    #     summarize_model=azure_summarize_deployment,
    # )

    # -- Search Provider -------------------------------------------------------
    #
    # AzureFoundryBingSearch is the same component that is smoke-tested in:
    #   examples/provider_stacks/azure_smoke_tests/test_bing_search.py
    # Test that file first if the combined setup fails.
    #
    # This file intentionally uses the classic Bing Grounding agent path,
    # not the newer Responses-API web-search path from:
    #   examples/provider_stacks/azure_smoke_tests/test_foundry_web_search.py
    #
    # Compared with PerplexitySearch, key differences are:
    #
    # - PerplexitySearch: one API call per search, all parameters per call.
    # - AzureFoundryBingSearch: reuses a pre-created Azure Foundry agent
    #   with Bing Grounding. Each search() call uses the project-scoped
    #   Responses API with an agent_reference, and the agent returns a
    #   grounded answer with source URLs as annotations.
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
    # The preferred runtime identifier is the agent name plus optional
    # version.  The legacy opaque agent ID is still accepted when you
    # need compatibility with older setups.
    #
    project_api_key = _optional_env("AZURE_AI_PROJECT_API_KEY")
    tenant_id = _optional_env("AZURE_TENANT_ID")
    client_id = _optional_env("AZURE_CLIENT_ID")
    client_secret = _optional_env("AZURE_CLIENT_SECRET")

    search_kwargs = {
        "project_endpoint": project_endpoint,
    }
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

    # -- Option B: Create the agent on the fly (one-time setup) ----------------
    #
    # Takes 2-5 seconds.  The returned provider instance already has the
    # agent_name set and may also expose version / legacy ID metadata.
    # Save the printed agent name (and version, if present) for future
    # runs so you can use Option A instead of re-creating the agent each
    # time.
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
    # print(
    #     "Agent created:",
    #     search._agent_name,
    #     f"version={search._agent_version or 'latest/default'}",
    #     f"legacy_id={search._agent_id or '-'}",
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
        confidence_stop=8,
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
