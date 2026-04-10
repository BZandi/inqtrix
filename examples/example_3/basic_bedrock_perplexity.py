"""Direct Amazon Bedrock + Perplexity example (no LiteLLM proxy).

Architecture
------------
This example calls two independent APIs directly:

1. **Amazon Bedrock Converse API** — via ``BedrockLLM``, using a native
   ``boto3`` adapter.  Claude models are called through the Bedrock
   Runtime endpoint in your configured AWS region.

2. **Perplexity Sonar API** — via ``PerplexitySearch``, using the
   OpenAI Python SDK pointed at ``https://api.perplexity.ai``.

Authentication for Bedrock uses **AWS named profiles** configured in
``~/.aws/config`` and ``~/.aws/credentials``.  The profile name and
region are read from environment variables (or ``.env``).

Use this example when:
- you have an AWS account with Bedrock model access enabled
- you have a direct Perplexity API key
- you do NOT want to run a LiteLLM proxy
- you want to use Claude models hosted on Bedrock (e.g. EU region)

Required environment variables (in .env or process env):
- PERPLEXITY_API_KEY
- AWS_PROFILE          (optional — defaults to the default profile)
- AWS_REGION           (optional — defaults to eu-central-1)

Prerequisites:
- ``boto3`` must be installed: ``uv sync``
- The AWS profile must have ``bedrock:InvokeModel`` permission
- The requested models must be enabled in the target region

Run with::

    uv run python examples/example_3/basic_bedrock_perplexity.py
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from inqtrix import AgentConfig, BedrockLLM, PerplexitySearch, ResearchAgent

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
USE_STREAMING = True
INCLUDE_PROGRESS = True


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    perplexity_key = _require_env("PERPLEXITY_API_KEY")
    aws_profile = os.environ.get("AWS_PROFILE", "").strip() or None
    aws_region = os.environ.get("AWS_REGION", "").strip() or "eu-central-1"

    # ── LLM Provider ────────────────────────────────────────────────
    #
    # BedrockLLM calls the Amazon Bedrock Converse API via boto3.
    # Authentication is handled through AWS named profiles.
    #
    # default_model: Bedrock model ID for reasoning calls (classify,
    #   plan, evaluate, answer).  Uses the EU cross-region inference
    #   profile for Opus.
    #
    # summarize_model: cheaper model for search-result summarisation
    #   and claim extraction (called in parallel threads).
    #
    # Bedrock model IDs (EU region):
    #   eu.anthropic.claude-opus-4-6-v1
    #   eu.anthropic.claude-sonnet-4-6
    #   eu.anthropic.claude-sonnet-4-5-20250929-v1:0
    #
    # Extended thinking (optional):
    #   thinking={"type": "adaptive"}
    #
    # NOTE: temperature and thinking are MUTUALLY EXCLUSIVE.
    llm = BedrockLLM(
        profile_name=aws_profile,
        region_name=aws_region,
        default_model="eu.anthropic.claude-opus-4-6-v1",
        # classify_model="eu.anthropic.claude-sonnet-4-6",
        summarize_model="eu.anthropic.claude-sonnet-4-6",
        # evaluate_model="eu.anthropic.claude-sonnet-4-6",
        thinking={"type": "adaptive"},
    )

    # ── Search Provider ─────────────────────────────────────────────
    search = PerplexitySearch(
        api_key=perplexity_key,
        base_url="https://api.perplexity.ai",
        model="sonar-pro",
    )

    # ── Agent ────────────────────────────────────────────────────────
    agent = ResearchAgent(AgentConfig(
        llm=llm,
        search=search,
        max_rounds=4,
        confidence_stop=8,
        max_total_seconds=600,
    ))

    # ── Run ──────────────────────────────────────────────────────────
    if USE_STREAMING:
        answer_buf: list[str] = []
        in_answer = False
        for chunk in agent.stream(QUESTION, include_progress=INCLUDE_PROGRESS):
            if not in_answer and chunk == "---\n":
                in_answer = True
                continue
            if in_answer:
                answer_buf.append(chunk)
            else:
                print(chunk, end="", flush=True)

        full_answer = "".join(answer_buf)
        if full_answer:
            console = Console()
            print()
            console.print(Markdown(full_answer))
    else:
        result = agent.research(QUESTION)
        _print_result(result)


if __name__ == "__main__":
    main()
