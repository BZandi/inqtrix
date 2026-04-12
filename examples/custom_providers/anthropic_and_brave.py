"""Use Anthropic and Brave directly without LiteLLM.

Architecture
------------
This example calls two independent APIs directly:

1. **Anthropic Messages API** — via ``AnthropicLLM``.  Claude models
   handle all reasoning roles (classify, plan, evaluate, answer) plus
   parallel summarisation.

2. **Brave Search API** — via ``BraveSearch``.  Brave's Web Search
   endpoint returns organic results with titles, descriptions, and
   URLs.  The research algorithm uses this as its grounding source.

Each provider has its own API key — there is no shared proxy.

Use this example when:
- you want a fully custom provider setup through the Baukasten system
- Anthropic should handle reasoning/classify/summarize/evaluate
- Brave Search should handle web search
- you do NOT want to run a LiteLLM proxy

Required environment variables (in .env or process env):
- ANTHROPIC_API_KEY
- BRAVE_API_KEY

Run with::

    uv run python examples/custom_providers/anthropic_and_brave.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from inqtrix import AgentConfig, AnthropicLLM, BraveSearch, ResearchAgent


QUESTION = "Was ist der aktuelle Stand der GKV-Reform?"
# ── Output mode ──────────────────────────────────────────────────────
# True  → streaming (live progress messages + word-by-word answer)
# False → blocking  (waits for the full result, then prints at once)
USE_STREAMING = False

# Only relevant when USE_STREAMING is True:
# True  → show intermediate progress messages before the answer
# False → stream only the final answer text, no status updates
INCLUDE_PROGRESS = True


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


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    agent = ResearchAgent(AgentConfig(
        llm=AnthropicLLM(
            api_key=_require_env("ANTHROPIC_API_KEY"),
            # Strong model for reasoning, planning, and final answer.
            default_model="claude-sonnet-4-6",
            # Optional per-role overrides; leave unset to fall back to
            # default_model.
            # classify_model="claude-haiku-4-5",
            # summarize_model is usually the best cost lever because it
            # runs on many search results in parallel.
            summarize_model="claude-haiku-4-5",
            # evaluate_model="claude-sonnet-4-6",
            # thinking={"type": "adaptive"},
        ),
        search=BraveSearch(
            api_key=_require_env("BRAVE_API_KEY"),
        ),
        max_rounds=4,                       # max research-loop iterations
        # confidence threshold (1-10) — stop early when reached
        confidence_stop=8,
        max_total_seconds=300,              # hard wall-clock deadline (seconds)
        # All other options (timeouts, risk escalation, search cache,
        # strategies) have sensible defaults.  See the provider_stacks/
        # examples for full AgentConfig documentation.
    ))

    if USE_STREAMING:
        for chunk in agent.stream(QUESTION, include_progress=INCLUDE_PROGRESS):
            print(chunk, end="", flush=True)
        return

    result = agent.research(QUESTION)
    print(result.answer)
    print()
    print(f"Confidence: {result.metrics.confidence}/10")
    print(f"Sources: {result.metrics.total_citations}")
    print(f"Rounds: {result.metrics.rounds}")


if __name__ == "__main__":
    main()
