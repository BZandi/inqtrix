"""Minimal Azure OpenAI smoke test.

Use this before the larger Azure examples when you want to verify only
the direct LLM call path via ``AzureOpenAILLM``.

What this checks:
- Azure OpenAI endpoint is correct
- API key works
- deployment name is valid
- the model can return a basic chat completion

Required environment variables:
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_DEPLOYMENT_NAME

Optional:
- AZURE_OPENAI_TEST_PROMPT
- INQTRIX_LOG_ENABLED
- INQTRIX_LOG_LEVEL
- INQTRIX_LOG_CONSOLE

Run with::

    uv run python examples/example_4/basic_test_component_azure_llm.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from inqtrix import AzureOpenAIAPIError, AzureOpenAILLM
from inqtrix.exceptions import AgentRateLimited, AgentTimeout
from inqtrix.logging_config import configure_logging


load_dotenv()

_log_path = configure_logging(
    enabled=os.getenv("INQTRIX_LOG_ENABLED", "").lower() == "true",
    level=os.getenv("INQTRIX_LOG_LEVEL", "INFO"),
    console=os.getenv("INQTRIX_LOG_CONSOLE", "").lower() == "true",
)
if _log_path:
    print(f"Logging to {_log_path}")

PROMPT = os.getenv(
    "AZURE_OPENAI_TEST_PROMPT",
    "Antworte auf Deutsch in genau einem Satz: Dieser Test prueft, ob Azure OpenAI korrekt aufgerufen wird.",
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    azure_endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
    azure_deployment = _require_env("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_api_key = _require_env("AZURE_OPENAI_API_KEY")

    llm = AzureOpenAILLM(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        default_model=azure_deployment,
    )

    response = llm.complete_with_metadata(PROMPT)

    console = Console()
    console.print(
        Panel(
            response.content or "<empty response>",
            title="Azure OpenAI response",
            expand=False,
        )
    )
    console.print(
        Panel(
            "\n".join(
                [
                    f"Deployment: {response.model or azure_deployment}",
                    f"Prompt tokens: {response.prompt_tokens}",
                    f"Completion tokens: {response.completion_tokens}",
                ]
            ),
            title="Metadata",
            expand=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, AgentTimeout, AgentRateLimited, AzureOpenAIAPIError) as exc:
        raise SystemExit(str(exc)) from exc
