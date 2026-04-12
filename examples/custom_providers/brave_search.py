"""Use Brave Search directly while keeping the LLM side on env-based auto setup.

Architecture
------------
This example demonstrates the **partial override** pattern:

- **Search** — ``BraveSearch`` with a direct Brave API key.
  The Brave Web Search endpoint returns organic results with titles,
  descriptions, and URLs.

- **LLM** — omitted (``None``).  The agent auto-creates the LLM
  provider from environment variables / ``.env`` on first
  ``.research()`` call, typically ``LiteLLM`` through the configured
  proxy.

This is the inverse of ``anthropic_with_env_search.py``: only the
search side changes, while the LLM backend stays on env-based setup.

Use this example when:
- you already have a working env-based LLM setup
- you want to replace only the search backend with Brave Search

Required environment variables (in .env or process env):
- BRAVE_API_KEY
- the normal env-based LLM variables (LITELLM_BASE_URL, LITELLM_API_KEY, REASONING_MODEL)

Run with::

    uv run python examples/custom_providers/brave_search.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from inqtrix import AgentConfig, BraveSearch, ResearchAgent


QUESTION = "Was ist der aktuelle Stand der GKV-Reform?"


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
    # ── Search Provider ─────────────────────────────────────────────
    #
    # BraveSearch calls the Brave Web Search API directly.
    # Unlike PerplexitySearch, Brave returns raw organic results
    # (title + description + URL) without LLM summarisation — the
    # research agent's summarize step handles that part.
    agent = ResearchAgent(AgentConfig(
        search=BraveSearch(api_key=_require_env("BRAVE_API_KEY")),
        # All other options (timeouts, risk escalation, search cache,
        # strategies) have sensible defaults.  See the provider_stacks/
        # examples for full AgentConfig documentation.
    ))
    result = agent.research(QUESTION)

    print(result.answer)
    print()
    print(f"Confidence: {result.metrics.confidence}/10")
    print(f"Sources: {result.metrics.total_citations}")
    print(f"Rounds: {result.metrics.rounds}")


if __name__ == "__main__":
    main()
