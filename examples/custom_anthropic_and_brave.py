"""Use Anthropic and Brave directly without LiteLLM.

Use this example when:
- you want a fully custom provider setup through the Baukasten system
- Anthropic should handle reasoning/classify/summarize/evaluate
- Brave Search should handle web search

Required environment variables:
- ANTHROPIC_API_KEY
- BRAVE_API_KEY

Important detail:
- this script reads environment variables directly
- for local development it explicitly loads `.env` via `load_dotenv()`

Run with:
    uv run python examples/custom_anthropic_and_brave.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from inqtrix import AgentConfig, AnthropicLLM, BraveSearch, ResearchAgent


QUESTION = "Was ist der aktuelle Stand der GKV-Reform?"
USE_STREAMING = False
INCLUDE_PROGRESS = True


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
        search=BraveSearch(
            api_key=_require_env("BRAVE_API_KEY"),
        ),
        reasoning_model="claude-3-7-sonnet-latest",
        classify_model="claude-3-5-haiku-latest",
        summarize_model="claude-3-5-haiku-latest",
        evaluate_model="claude-3-5-haiku-latest",
        max_rounds=4,
        confidence_stop=8,
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
