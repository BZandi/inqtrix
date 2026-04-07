"""Use Anthropic directly for LLM calls while keeping search on env-based setup.

Use this example when:
- you want to remove LiteLLM from the LLM side only
- your search backend should still come from the normal env-based provider path

Required environment variables:
- ANTHROPIC_API_KEY
- the normal env-based search variables, usually LITELLM_BASE_URL, LITELLM_API_KEY, SEARCH_MODEL

Important detail:
- this script reads environment variables directly
- for local development it explicitly loads `.env` via `load_dotenv()`

Run with:
    uv run python examples/custom_anthropic_with_env_search.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from inqtrix import AgentConfig, AnthropicLLM, ResearchAgent


QUESTION = "Was ist der aktuelle Stand der GKV-Reform?"


load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    agent = ResearchAgent(AgentConfig(
        llm=AnthropicLLM(
            api_key=_require_env("ANTHROPIC_API_KEY"),
            default_model="claude-3-7-sonnet-latest",
            summarize_model="claude-3-5-haiku-latest",
        ),
    ))
    result = agent.research(QUESTION)

    print(result.answer)
    print()
    print(f"Confidence: {result.metrics.confidence}/10")
    print(f"Sources: {result.metrics.total_citations}")
    print(f"Rounds: {result.metrics.rounds}")


if __name__ == "__main__":
    main()
