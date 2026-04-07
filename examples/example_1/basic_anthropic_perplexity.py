"""Direct Anthropic + Perplexity example (no LiteLLM proxy).

Architecture
------------
This example calls two independent APIs directly:

1. **Anthropic Messages API** — via ``AnthropicLLM``, using a native
   ``urllib`` adapter.  Claude models are called at
   ``https://api.anthropic.com/v1/messages`` without any proxy or SDK
   in between.

2. **Perplexity Sonar API** — via ``PerplexitySearch``, using the
   OpenAI Python SDK pointed at ``https://api.perplexity.ai``.
   Perplexity's Sonar endpoint is fully OpenAI Chat Completions
   compatible, so the SDK works out of the box.

Each provider has its own API key and base URL — there is no shared
proxy.  Compare this with the sibling ``basic_litellm_perplexity.py``
where a single LiteLLM proxy routes both LLM and search traffic.

Use this example when:
- you have direct API keys for both Anthropic and Perplexity
- you do NOT want to run a LiteLLM proxy
- you want the simplest possible two-provider setup

Required environment variables (in .env or process env):
- ANTHROPIC_API_KEY
- PERPLEXITY_API_KEY

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

    uv run python examples/example_1/basic_anthropic_perplexity.py
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from inqtrix import AgentConfig, AnthropicLLM, PerplexitySearch, ResearchAgent

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


load_dotenv()

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
    anthropic_key = _require_env("ANTHROPIC_API_KEY")
    perplexity_key = _require_env("PERPLEXITY_API_KEY")

    # ── LLM Provider ────────────────────────────────────────────────
    #
    # AnthropicLLM calls the Anthropic Messages API directly
    # (https://api.anthropic.com/v1/messages) without a proxy.
    #
    # default_model: used for ALL reasoning roles — classify, plan,
    #   evaluate, and final answer synthesis.  Unlike LiteLLM, there
    #   are no separate classify_model / evaluate_model overrides.
    #
    # summarize_model: the cheaper, faster model used exclusively for
    #   search-result summarization and claim extraction (called in
    #   parallel threads via summarize_parallel).
    #
    # Optional tuning (defaults shown):
    #   default_max_tokens=1024    — output budget for reasoning calls
    #   summarize_max_tokens=512   — output budget for summarization
    #
    # Extended thinking (optional) — uncomment ONE of the following:
    #
    #   Adaptive thinking (recommended for Claude 4.6 models):
    #   thinking={"type": "adaptive"},
    #
    #   Manual budget (Claude 3.7 / precise control):
    #   thinking={"type": "enabled", "budget_tokens": 10000},
    #
    # When thinking is enabled, set default_max_tokens high enough
    # (e.g. 16000).  If budget_tokens >= default_max_tokens the
    # provider auto-raises max_tokens to budget_tokens + 1024.
    #
    # temperature (optional, 0.0–1.0) — sampling temperature.
    # NOTE: temperature and thinking are MUTUALLY EXCLUSIVE.
    #   temperature=0.3,
    llm = AnthropicLLM(
        api_key=anthropic_key,
        default_model="claude-sonnet-4-6",
        summarize_model="claude-haiku-4-5",
    )

    # ── Search Provider ─────────────────────────────────────────────
    #
    # PerplexitySearch pointed directly at the Perplexity Sonar API.
    # This provider is designed specifically for the Sonar API — other
    # Perplexity products (Deep Research, Agent API) use different
    # parameters and endpoints.
    #
    # base_url is "https://api.perplexity.ai" (no /v1 suffix — the
    # OpenAI SDK appends /chat/completions internally).
    #
    # The provider auto-detects direct mode from the URL and formats
    # search parameters (recency, language, domain filters, etc.) as
    # flat top-level extra_body keys — which is what the Perplexity
    # API expects.  Through a LiteLLM proxy these would be nested
    # inside web_search_options instead.
    #
    # Model names are Perplexity-native here (sonar-pro, sonar),
    # not LiteLLM aliases (perplexity-sonar-pro-agent).
    search = PerplexitySearch(
        api_key=perplexity_key,
        base_url="https://api.perplexity.ai",
        model="sonar-pro",
    )

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
        # hard wall-clock deadline for the entire run (seconds)
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
        # With AnthropicLLM the escalation flags below are effectively
        # no-ops: classify and evaluate already use default_model
        # (there are no cheaper per-role models to escalate FROM).
        # They exist for compatibility with LiteLLM, which supports
        # classify_model / evaluate_model overrides.
        #
        # Summarization always uses summarize_model regardless of risk.
        high_risk_score_threshold=4,        # risk score ≥ this triggers high_risk = True
        high_risk_classify_escalate=True,   # no-op with AnthropicLLM
        high_risk_evaluate_escalate=True,   # no-op with AnthropicLLM

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
