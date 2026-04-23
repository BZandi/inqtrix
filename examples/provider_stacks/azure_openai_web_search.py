"""Azure OpenAI v1 + native web_search example  --  VARIANTE C (no Foundry agent).

==============================================================================
WHICH AZURE WEB-SEARCH VARIANT THIS FILE COVERS
==============================================================================
This script uses the NATIVE Azure OpenAI Responses-API ``web_search``
tool path (provider class ``AzureOpenAIWebSearch``, smoke-tested in
``test_openai_web_search.py``).  Web search is invoked directly through
the Azure OpenAI endpoint with ``tools=[{"type": "web_search"}]`` --
no Foundry project, no Foundry agent, and no separate Bing resource
are required.

The sibling stack file ``azure_foundry_web_search.py`` covers a
different variant: the Azure FOUNDRY Web Search Agent path (provider
class ``AzureFoundryWebSearch``).  That variant needs a pre-created
Foundry agent and the project-scoped endpoint.

Quick way to verify which path a stack file uses: check the Search-Provider
import at the top of the file.

- imports ``AzureOpenAIWebSearch``  -> Variante C (native, this file)
- imports ``AzureFoundryWebSearch`` -> Variante B (Foundry agent, see
                                        ``azure_foundry_web_search.py``)

==============================================================================
WHICH SMOKE TESTS MUST PASS BEFORE RUNNING THIS FILE
==============================================================================
Run, in order, and only proceed if each prints a green panel:

1. ``examples/provider_stacks/azure_smoke_tests/test_llm.py``
   -- proves the Azure OpenAI reasoning path (this file uses it for
   classify / plan / summarize / evaluate / answer).
2. ``examples/provider_stacks/azure_smoke_tests/test_openai_web_search.py``
   -- proves the native ``web_search`` tool path (this file uses it
   for every web-search call).

The other two smoke tests in the folder are NOT required for this
file:

- ``test_foundry_web_search.py`` -- Foundry Web Search agent (Variante B)
- ``test_bing_search.py``        -- Bing-grounded agent (Variante A)

Architecture
------------
This example combines two independent uses of the SAME Azure OpenAI
resource:

1. **Sprachmodell / Reasoning** -- via ``AzureOpenAILLM``
2. **Websuche / Grounding** -- via ``AzureOpenAIWebSearch``

Both providers point at the same Azure OpenAI endpoint and use the
same model deployment -- one for chat-completion-style reasoning, the
other for the Responses API's ``web_search`` tool.  Compared with
Variante B, this is the simplest possible Azure web stack: ONE
resource, ONE deployment, ZERO additional Azure components.

Why two providers if they share one Azure resource?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The reasoning and search paths are still kept as two distinct
provider instances because they call DIFFERENT API surfaces on the
same endpoint:

- ``AzureOpenAILLM`` calls ``chat.completions.create`` (no tools).
- ``AzureOpenAIWebSearch`` calls ``responses.create`` with
  ``tools=[{"type": "web_search"}]``.

Mixing them in a single provider would couple search-tool semantics
into every reasoning call (or vice versa) and corrupt Inqtrix's
classify / plan / evaluate prompts that expect plain text output.

Comparison with Variante B (Foundry Web Search)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+-------------------------+-----------------------------+---------------------------+
| Aspect                  | Variante C (this file)      | Variante B                |
|                         | -> AzureOpenAIWebSearch     | -> AzureFoundryWebSearch  |
+=========================+=============================+===========================+
| Required Azure          | 1 OpenAI resource           | 1 OpenAI resource +       |
| components              | + 1 model deployment        | 1 Foundry project +       |
|                         |                             | 1 model deployment in     |
|                         |                             |   that project +          |
|                         |                             | 1 Web Search agent        |
+-------------------------+-----------------------------+---------------------------+
| Endpoint                | ``...openai.azure.com/``    | ``...services.ai.azure.   |
|                         |                             | com/api/projects/...``    |
+-------------------------+-----------------------------+---------------------------+
| Per-call configurability| ``user_location``,          | Limited: most settings    |
|                         | ``allowed_domains``,        | are fixed at agent        |
|                         | ``tool_choice`` per request | creation time             |
+-------------------------+-----------------------------+---------------------------+
| Versioning, governance, | None                        | YES via Foundry portal    |
| multi-tool agents       |                             |                           |
+-------------------------+-----------------------------+---------------------------+
| Bing backend            | Same backend internally     | Same backend internally   |
+-------------------------+-----------------------------+---------------------------+

Use this file when you want the absolutely shortest path from
"I have an Azure OpenAI deployment" to "I have a working Inqtrix
research run".  Use ``azure_foundry_web_search.py`` if you need the
Foundry features in the right column (versioning, central agent
governance, multi-tool agents).

Required environment variables (in .env or process env):
- AZURE_OPENAI_API_KEY          (LLM + search; same key for both)
- AZURE_OPENAI_ENDPOINT         (e.g. https://mein-openai.openai.azure.com/)
- AZURE_OPENAI_DEPLOYMENT_NAME  (must support web_search; e.g. gpt-4.1 / gpt-4o)

Optional:
- AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME  (cheaper deployment for summarisation;
                                            see "Summarize behaviour" below)
- AZURE_OPENAI_WEB_SEARCH_USER_COUNTRY    (ISO country for user_location;
                                            default: DE)
- AZURE_WEB_STACK_QUESTION                (override the default demo question)
- HTTPS_PROXY                             (corporate proxy environments)

For Service Principal auth (instead of API key):
- AZURE_TENANT_ID
- AZURE_CLIENT_ID
- AZURE_CLIENT_SECRET

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

Subscription eligibility note
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Even though no separate Bing resource is required, the native
``web_search`` tool internally uses Microsoft's Grounding-with-Bing
backend.  Microsoft restricts that backend to paid / pay-as-you-go
subscriptions.  Pure sponsored or trial-only subscriptions may get
an explicit licence error on the FIRST search call.  In that case,
upgrade the subscription in "Cost Management + Billing" to
Pay-as-you-go (your remaining trial credit is preserved).

Terminal rendering
------------------
The agent returns Markdown, which looks good in a chat UI but is
hard to read as raw text in a terminal.  This example uses ``rich``
to render Markdown with colours, formatted headers, bullet lists,
and clickable links (in terminals that support OSC 8 hyperlinks
such as iTerm2, Windows Terminal, or GNOME Terminal).

``rich`` is a core dependency and always available after::

    uv sync

Run with::

    uv sync
    uv run python examples/provider_stacks/azure_openai_web_search.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from inqtrix import (
    AgentConfig,
    AzureOpenAILLM,
    AzureOpenAIWebSearch,
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

QUESTION = os.getenv(
    "AZURE_WEB_STACK_QUESTION",
    "Was ist der aktuelle Stand der GKV-Reform?",
)


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
    azure_api_key = _require_env("AZURE_OPENAI_API_KEY")

    azure_summarize_deployment = os.environ.get(
        "AZURE_OPENAI_SUMMARIZE_DEPLOYMENT_NAME", ""
    ).strip()

    user_country = os.getenv("AZURE_OPENAI_WEB_SEARCH_USER_COUNTRY", "DE").strip()
    user_location = (
        {"type": "approximate", "country": user_country} if user_country else None
    )

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
    llm = AzureOpenAILLM(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        default_model=azure_deployment,
        summarize_model=azure_summarize_deployment,
        default_max_tokens=16384,
        summarize_max_tokens=16384,
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

    # -- Option D: Existing token provider ------------------------------------
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
    # AzureOpenAIWebSearch is the same component that is smoke-tested in:
    #   examples/provider_stacks/azure_smoke_tests/test_openai_web_search.py
    # Test that file first if the combined setup fails.
    #
    # This provider talks to the SAME Azure OpenAI endpoint as the LLM
    # but calls responses.create with the native web_search tool.
    # The deployment passed as default_model must support web_search
    # (gpt-4.1, gpt-4.1-mini, gpt-4o family).  Reusing the LLM's
    # deployment is the normal case and saves having to manage a
    # second deployment.
    #
    # Supported per-call hints (others are silently ignored):
    #
    # - domain_filter -> mapped to filters.allowed_domains; negative
    #                    entries (-domain.com) become -site: operators
    #                    appended to the query
    # - deadline      -> clamped to remaining run budget
    #
    # Other generic Inqtrix hints (recency_filter, language_filter,
    # search_context_size, search_mode, return_related) are not yet
    # mapped because the public Azure docs do not provide a stable
    # per-request mapping for them in the native path.
    #
    # Option A: API Key (default; same key as the LLM above)
    search = AzureOpenAIWebSearch(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        default_model=azure_deployment,
        user_location=user_location,
        # tool_choice="auto" is the default; set "required" to force
        # a web_search call even for trivial questions, or "none" to
        # disable the tool while still going through the Responses API.
        # tool_choice="required",
    )

    # -- Option B: Service Principal — direct constructor args ----------------
    #
    # The provider builds the ClientSecretCredential and token provider
    # internally — no azure.identity imports needed in your code.
    #
    # search = AzureOpenAIWebSearch(
    #     azure_endpoint=azure_endpoint,
    #     tenant_id=os.environ["AZURE_TENANT_ID"],
    #     client_id=os.environ["AZURE_CLIENT_ID"],
    #     client_secret=os.environ["AZURE_CLIENT_SECRET"],
    #     default_model=azure_deployment,
    #     user_location=user_location,
    # )

    # -- Option C: Service Principal — manual token provider ------------------
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
    # search = AzureOpenAIWebSearch(
    #     azure_endpoint=azure_endpoint,
    #     azure_ad_token_provider=token_provider,
    #     default_model=azure_deployment,
    #     user_location=user_location,
    # )

    # -- Option D: DefaultAzureCredential (local dev / Managed Identity) ------
    #
    # from azure.identity import DefaultAzureCredential
    #
    # search = AzureOpenAIWebSearch(
    #     azure_endpoint=azure_endpoint,
    #     credential=DefaultAzureCredential(),
    #     default_model=azure_deployment,
    #     user_location=user_location,
    # )

    # -- Option E: Hardcoded API key -------------------------------------------
    #
    # search = AzureOpenAIWebSearch(
    #     azure_endpoint=azure_endpoint,
    #     api_key="your-api-key-from-azure-portal",
    #     default_model=azure_deployment,
    #     user_location=user_location,
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
        # GPT-4.1 / GPT-4o through Azure are typically fast — 300s is usually enough.
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
