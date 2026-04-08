"""Explicit LiteLLM + PerplexitySearch example.

Architecture
------------
In this setup, a single LiteLLM proxy serves as the gateway for BOTH
the language models (e.g. Claude) AND the Perplexity Sonar search API.
That is why ``LiteLLM`` and ``PerplexitySearch`` share the same
``base_url`` and ``api_key`` — every request goes through the LiteLLM
endpoint, which routes it to the correct upstream provider based on the
model name.

This is different from calling the Perplexity API directly.  If you
want direct Perplexity access without a proxy, you would set a separate
``base_url`` (e.g. ``https://api.perplexity.ai``) and a dedicated
Perplexity API key on the ``PerplexitySearch`` provider instead.

Use this example when:
- reasoning and search models are reachable through one LiteLLM- or OpenAI-compatible endpoint
- you want to see how providers are configured explicitly (Baukasten pattern)

Required environment variables (in .env or process env):
- LITELLM_API_KEY
- optionally LITELLM_BASE_URL (defaults to http://localhost:4000/v1)

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

    uv run python examples/example_1/basic_litellm_perplexity.py
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from inqtrix import AgentConfig, LiteLLM, PerplexitySearch, ResearchAgent

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
    api_key = _require_env("LITELLM_API_KEY")
    # other option: os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1").strip()
    base_url = "http://127.0.0.1:4000/v1"

    # ── LLM Provider ────────────────────────────────────────────────
    #
    # default_model: the primary model used for reasoning, query
    #   planning, and final answer synthesis.  This is where the
    #   strongest model usually belongs.
    #
    # classify_model / summarize_model / evaluate_model: optional
    #   per-role overrides.  If left empty (""), each falls back to
    #   default_model.
    #
    #   classify_model: good place for a smaller model if the question
    #   decomposition is straightforward and you mainly want to save
    #   cost on the first routing step.
    #
    #   summarize_model: usually the best place to save money because
    #   it runs in parallel on many search results.  If claim
    #   extraction gets too shallow or noisy, move this role up.
    #
    #   evaluate_model: useful for a slightly cheaper model than
    #   default_model, but keep it strong enough for evidence weighing
    #   and stop decisions.
    llm = LiteLLM(
        api_key=api_key,
        base_url=base_url,
        default_model="claude-opus-4.6-agent",
        classify_model="claude-sonnet-4.6",
        summarize_model="claude-sonnet-4.6",
        evaluate_model="claude-sonnet-4.6",
    )

    # ── Search Provider ─────────────────────────────────────────────
    search = PerplexitySearch(
        api_key=api_key,
        base_url=base_url,
        model="perplexity-sonar-pro-agent",
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
        # hard wall-clock deadline for the entire run (seconds).
        # Opus through a proxy needs more time than Sonnet — 600s is a
        # safe default.  Reduce to 300 for Sonnet-only setups.
        max_total_seconds=600,
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
        # high_risk_score_threshold decides WHEN a question counts as
        # high-risk.  The two boolean flags below then decide WHETHER
        # model escalation actually happens for those high-risk cases.
        #
        # Example with the defaults below:
        #   - Question gets risk_score = 6 → high_risk = True (≥ 4)
        #   - high_risk_classify_escalate = True
        #     → classify node uses the strong default_model instead of
        #       the cheaper classify_model
        #   - high_risk_evaluate_escalate = True
        #     → evaluate node uses the strong default_model instead of
        #       the cheaper evaluate_model
        #
        # Set either flag to False to save cost: the smaller model is
        # then used even for high-risk questions in that role.
        high_risk_score_threshold=4,        # risk score ≥ this triggers high_risk = True
        high_risk_classify_escalate=True,   # escalate classify to default_model on high-risk
        high_risk_evaluate_escalate=True,   # escalate evaluate to default_model on high-risk

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
