"""Use Brave Search directly while keeping the LLM side on env-based auto setup.

Use this example when:
- you already have a working env-based LLM setup
- you want to replace only the search backend with Brave Search

Required environment variables:
- BRAVE_API_KEY
- the normal env-based LLM variables, for example LITELLM_BASE_URL, LITELLM_API_KEY, REASONING_MODEL

Important detail:
- this script reads environment variables directly
- for local development it explicitly loads `.env` via `load_dotenv()`

Run with:
    uv run python examples/custom_brave_search.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from inqtrix import AgentConfig, BraveSearch, ResearchAgent


QUESTION = "Was ist der aktuelle Stand der GKV-Reform?"


load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    agent = ResearchAgent(AgentConfig(
        search=BraveSearch(api_key=_require_env("BRAVE_API_KEY")),
    ))
    result = agent.research(QUESTION)

    print(result.answer)
    print()
    print(f"Confidence: {result.metrics.confidence}/10")
    print(f"Sources: {result.metrics.total_citations}")
    print(f"Rounds: {result.metrics.rounds}")


if __name__ == "__main__":
    main()
