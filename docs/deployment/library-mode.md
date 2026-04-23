# Library mode

> Files: `src/inqtrix/agent.py`, `src/inqtrix/__init__.py`

## Scope

How to embed Inqtrix in a Python script or application as a library. Three common entry paths are documented: env-based auto-creation, explicit Baukasten providers, and manual YAML bridging. For the HTTP server see [Web server mode](webserver-mode.md).

## Prerequisites

- Python ≥ 3.12.
- Editable install with dev extras: `uv sync --extra dev` or `pip install -e ".[dev]"`.
- At least one set of provider credentials in `.env` or the process environment.

See [Installation](../getting-started/installation.md) for the full bootstrap.

## Option A: `.env` or process environment

`ResearchAgent()` auto-loads a local `.env` from the current working directory when providers are auto-created. `inqtrix.yaml` is **not** auto-loaded in library mode.

Use this path when both reasoning model and search model are reachable through the same LiteLLM- or OpenAI-compatible endpoint.

```dotenv
# .env
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_API_KEY=sk-...
REASONING_MODEL=gpt-4o
SEARCH_MODEL=perplexity-sonar-pro-agent
CLASSIFY_MODEL=gpt-4o-mini
SUMMARIZE_MODEL=gpt-4o-mini
EVALUATE_MODEL=gpt-4o-mini
REPORT_PROFILE=deep
```

```python
from inqtrix import ResearchAgent

agent = ResearchAgent()
result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")

print(result.answer)
print(f"Confidence: {result.metrics.confidence}/10")
print(f"Sources: {result.metrics.total_citations}")
print(f"Rounds: {result.metrics.rounds}")
```

If `CLASSIFY_MODEL`, `SUMMARIZE_MODEL`, or `EVALUATE_MODEL` are omitted they fall back to `REASONING_MODEL`. Set `REPORT_PROFILE=deep` for a longer review-style answer.

## Option B: explicit providers (Baukasten pattern)

Use this path when you want to define providers and models directly in Python. No `.env` file is required by the providers themselves (Constructor-First); however, your script may still read environment variables from `.env` via `load_dotenv()`.

```python
import os
from dotenv import load_dotenv

from inqtrix import AgentConfig, LiteLLM, PerplexitySearch, ReportProfile, ResearchAgent


load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


llm = LiteLLM(
    api_key=_require_env("LITELLM_API_KEY"),
    base_url=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1"),
    default_model="gpt-4o",
    classify_model="gpt-4o-mini",
    summarize_model="gpt-4o-mini",
    evaluate_model="gpt-4o-mini",
)

search = PerplexitySearch(
    api_key=_require_env("LITELLM_API_KEY"),
    base_url=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1"),
    model="perplexity-sonar-pro-agent",
)

agent = ResearchAgent(AgentConfig(
    llm=llm,
    search=search,
    report_profile=ReportProfile.DEEP,
    max_rounds=4,
    confidence_stop=8,
    max_total_seconds=300,
))

result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")
print(result.answer)
```

All model names live on provider constructors (`LiteLLM`, `PerplexitySearch`, `AnthropicLLM`, ...), not on `AgentConfig`. See [Providers overview](../providers/overview.md) and [Agent config](../configuration/agent-config.md).

Rule of thumb:

- If `ResearchAgent()` creates providers internally, `.env` is auto-loaded by the settings layer.
- If your own script reads `os.environ[...]`, your script must either call `load_dotenv()` or rely on already-exported environment variables.

## Option C: manual YAML bridging

Use this path when you want the same multi-provider routing as the HTTP server from a library script, for example direct OpenAI for reasoning plus direct Perplexity for search.

```python
from inqtrix import AgentConfig, ReportProfile, ResearchAgent
from inqtrix.config import load_config
from inqtrix.config_bridge import config_to_settings, create_providers_from_config


config = load_config("inqtrix.yaml")
settings = config_to_settings(config)
providers = create_providers_from_config(config, settings)

agent = ResearchAgent(AgentConfig(
    llm=providers.llm,
    search=providers.search,
    report_profile=ReportProfile.DEEP,
    **settings.agent.model_dump(),
))
```

Secrets still belong in `.env` or the real process environment, not in Python. `load_config()` auto-loads a local `.env` before resolving `${ENV_VAR}` placeholders.

## Streaming

`research(...)` waits until the full loop is finished and returns a structured `ResearchResult`. `stream(...)` is the incremental variant for scripts and CLIs.

```python
from inqtrix import ResearchAgent

agent = ResearchAgent()
for chunk in agent.stream("Meine Frage"):
    print(chunk, end="", flush=True)
```

By default it emits live status messages first, then the final answer chunks. Pass `include_progress=False` to omit the intermediate updates:

```python
for chunk in agent.stream("Meine Frage", include_progress=False):
    print(chunk, end="", flush=True)
```

Progress messages look like `Analysiere Frage...`, `Plane Suchanfragen (Runde 1/4)...`, `Durchsuche 6 Quellen (Runde 1/4)...`, `Bewerte Informationsqualitaet (nach Runde 1/4)...`, `Formuliere Antwort (nach 3 Runden)...`.

### Example: pick blocking vs streaming

```python
from inqtrix import ResearchAgent


QUESTION = "Was ist der aktuelle Stand der GKV-Reform?"
USE_STREAMING = True
INCLUDE_PROGRESS = True


def main() -> None:
    agent = ResearchAgent()

    if USE_STREAMING:
        for chunk in agent.stream(QUESTION, include_progress=INCLUDE_PROGRESS):
            print(chunk, end="", flush=True)
        return

    result = agent.research(QUESTION)
    print(result.answer)
    print(f"Confidence: {result.metrics.confidence}/10")
    print(f"Sources: {result.metrics.total_citations}")
    print(f"Rounds: {result.metrics.rounds}")


if __name__ == "__main__":
    main()
```

## Optional: Streamlit UI (`webapp.py`)

`webapp.py` at the repo root is a Streamlit chat interface that talks to a running HTTP server. It is intentionally minimal and not part of the supported surface; start it with `streamlit run webapp.py` after the server is up. It is documented here because `.env.example` references it — new users sometimes discover it before the docs.

## Related docs

- [Providers overview](../providers/overview.md)
- [Writing a custom provider](../providers/writing-a-custom-provider.md)
- [Agent config](../configuration/agent-config.md)
- [Web server mode](webserver-mode.md)
