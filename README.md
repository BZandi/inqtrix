<div align="center">
  <img src="inqtrix-logo.svg" width="100%">
</div>
<p></p>

> [!WARNING]
> **Experimental Software / Reference and Integration Foundation**
> This repository is an experimental codebase and integration foundation for self-hosted or locally operated deployments. It does **not** provide a complete production-ready security configuration, hardened deployment profile, or any assurance that it is suitable for direct use in internet-facing, multi-user, regulated, or otherwise high-risk environments.
>
> Configurations, defaults, example values, example scripts, and helper paths included in this repository may be useful for development, testing, or integration work, but must not be assumed to be secure, complete, or appropriate for production use without independent review and adaptation.
>
> Secure configuration, hardening, deployment architecture, access control, secret handling, logging, monitoring, compliance, and day-to-day operation remain the sole responsibility of the operator. The current test suite covers substantial internal logic, interface behavior, and regression scenarios, but it is **not** evidence of production readiness or fully validated live integrations. Perform your own technical and security review before using this project in integration, staging, test, or production environments.

# Inqtrix

Codebase and self-hostable Python package for an iterative AI research agent with parallel web search, claim verification, and source tiering.

Inqtrix provides the library, HTTP server, and integration building blocks that you can run yourself against your own configured model and search providers.

When deployed, it runs a bounded multi-round research loop: it decomposes questions, searches the web in parallel, extracts and verifies claims against multiple sources, and synthesises a cited answer once evidence quality thresholds are met.

## Features

- **Iterative research loop** with configurable confidence thresholds and max rounds
- **Parallel web search** with LLM-based summarisation and structured claim extraction
- **Claim verification** — claims are consolidated, deduplicated, and classified as *verified*, *contested*, or *unverified*
- **Source tiering** — URLs are classified into quality tiers (primary, mainstream, stakeholder, unknown, low)
- **Aspect coverage tracking** — ensures all facets of a question are researched
- **9 stopping heuristics** — confidence, utility plateau, stagnation detection, falsification mode, and more
- **Pluggable architecture** — swap LLM providers, search engines, and strategies independently
- **Pydantic configuration** — type-safe, serialisable, IDE-friendly
- **OpenAI-compatible HTTP API** — drop-in replacement for `/v1/chat/completions`

## Installation

**Requirements:** Python 3.12+

### From a Fresh Clone

Create a dedicated Python 3.12 environment first. You can use any environment manager; `uv` and `conda` are both fine.

```bash
git clone https://github.com/BZandi/inqtrix.git
cd inqtrix

# Option A: uv
uv sync --extra dev
source .venv/bin/activate

# Option B: conda
conda create -n inqtrix python=3.12
conda activate inqtrix
pip install -e ".[dev]"
```

Editable install is the recommended workflow for local development and testing:

- `pip install -e .` installs the package in editable mode for normal local use
- `pip install -e ".[dev]"` does the same and also installs the test dependencies
- code changes under `src/inqtrix/` are picked up immediately without reinstalling after every edit

Because this repository uses a `src/` layout, a plain clone is not importable by default. In other words: being inside the project directory is not enough on its own for `import inqtrix`, `python -m inqtrix`, or the `inqtrix-parity` entry point to work reliably. Installing the project in editable mode links the environment to your working tree, which is exactly what you want while developing.

If you only want a quick experiment without installing, you can manually set `PYTHONPATH=src`, but that is a temporary workaround rather than the recommended setup.

### First Local Check

This is the fastest offline regression check after cloning. It runs the local `pytest` suite only, does **not** call real model or search providers, and does not require API keys.

```bash
uv run pytest tests/ -v
```

### Runnable Examples

The repository includes runnable example scripts under `examples/`. Each file starts with a short explanation of when to use it, which environment variables it expects, and how to run it.

- `examples/library_basic_env.py`: blocking library usage with the normal env-based provider setup
- `examples/library_streaming.py`: streaming library usage with optional progress messages
- `examples/yaml_library_mode.py`: library usage with YAML provider/model routing instead of the HTTP server
- `examples/custom_brave_search.py`: direct Brave Search plus env-based LLM setup
- `examples/custom_anthropic_with_env_search.py`: direct Anthropic LLM plus env-based search setup
- `examples/custom_anthropic_and_brave.py`: full custom setup without LiteLLM

Run any example like this:

```bash
uv run python examples/library_basic_env.py
```

Running an example script or your own `main.py` is **not** part of the automated test suite. If valid providers are configured, that is a real agent run against the configured backends.

For a consumer-style install outside active development, use a normal non-editable install instead:

```bash
pip install .

# Editable install without test extras
pip install -e .

# Editable install with test extras
pip install -e ".[dev]"
```

## Runtime Modes

| Mode | Optional files | Where models are defined | How to start |
|------|----------------|--------------------------|--------------|
| Python library via `.env` or process env | `.env` | environment variables | your own script, e.g. `uv run python main.py` |
| Python library via `AgentConfig` | none | Python code | your own script, e.g. `uv run python main.py` |
| HTTP server in env-only mode | `.env` | environment variables | `uv run python -m inqtrix` |
| HTTP server in YAML mode | `inqtrix.yaml`, `inqtrix.yml`, or `.inqtrix.yaml`, plus optional `.env` for secrets | YAML `roles:` plus environment variables for secrets | `uv run python -m inqtrix` |

`main.py` is only needed if you want your own library script. The HTTP server starts directly via `python -m inqtrix` and does not require a separate `main.py`.

For local development, `.env` is convenient in both library and server mode. Exported process environment variables always take precedence over values from `.env`.
In library mode, explicit scalar fields in `AgentConfig` override the values loaded from process env or `.env` when providers are auto-created.

## Quick Start

### As a Python Library

#### Option A: `.env` or process environment

`ResearchAgent()` auto-loads a local `.env` from the current working directory. `inqtrix.yaml` is **not** auto-loaded in library mode.

Use this path when both your reasoning model and your search model are reachable through the same LiteLLM- or OpenAI-compatible endpoint.

```dotenv
# .env
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_API_KEY=sk-...
REASONING_MODEL=gpt-4o
SEARCH_MODEL=perplexity-sonar-pro-agent
CLASSIFY_MODEL=gpt-4o-mini
SUMMARIZE_MODEL=gpt-4o-mini
EVALUATE_MODEL=gpt-4o-mini
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

If you prefer not to use `.env`, export the same variables in your shell before running the script.

If `CLASSIFY_MODEL`, `SUMMARIZE_MODEL`, or `EVALUATE_MODEL` are omitted, they fall back to `REASONING_MODEL`.

#### Option B: explicit providers (Baukasten pattern)

Use this path when you want to define providers and models directly in Python instead of `.env`. No `.env` file is required here.

In practice, the usual best practice is still: keep secrets in `.env` or real process environment variables and keep only the model/behavior choices in Python.

Because this script reads environment variables itself, local `.env` loading should be explicit via `load_dotenv()`.

```python
import os

from dotenv import load_dotenv

from inqtrix import ResearchAgent, AgentConfig, LiteLLM, PerplexitySearch


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
    max_rounds=4,
    confidence_stop=8,
    max_total_seconds=300,
))

result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")

print(result.answer)
print(f"Confidence: {result.metrics.confidence}/10")
```

All model names live on the provider constructors (`LiteLLM`, `PerplexitySearch`, `AnthropicLLM`), not on `AgentConfig`. This matches the Baukasten principle: each provider is a fully self-contained building block.

Rule of thumb:

- If `ResearchAgent()` creates providers internally, `.env` is auto-loaded by the settings layer.
- If your own script reads `os.environ[...]`, your script must either call `load_dotenv()` or rely on already-exported environment variables.

#### Option C: manual YAML bridging in library mode

Use this path when you want the same multi-provider routing as the HTTP server, for example direct OpenAI for reasoning plus direct Perplexity for search.

```python
from inqtrix import AgentConfig, ResearchAgent
from inqtrix.config import load_config
from inqtrix.config_bridge import config_to_settings, create_providers_from_config

config = load_config("inqtrix.yaml")
settings = config_to_settings(config)
providers = create_providers_from_config(config, settings)

agent = ResearchAgent(AgentConfig(
    llm=providers.llm,
    search=providers.search,
    **settings.agent.model_dump(),
))
```

In this path, secrets still belong in `.env` or the real process environment, not in Python.

`load_config()` auto-loads a local `.env` before resolving `${ENV_VAR}` placeholders in `inqtrix.yaml`, so this pattern is already explicit about where the tokens come from.

This is the advanced path. For the common YAML workflow, running the HTTP server is simpler because YAML is auto-detected there.

### With Streaming

`research()` waits until the full research loop is finished and then returns a structured `ResearchResult`.

`stream()` is the incremental variant for scripts and CLIs. By default it emits live status messages first, then the final answer as text chunks.

```python
from inqtrix import ResearchAgent

agent = ResearchAgent()
for chunk in agent.stream("Meine Frage"):
    print(chunk, end="", flush=True)
```

Typical progress messages look like this:

- `Analysiere Frage...`
- `Plane Suchanfragen (Runde 1/4)...`
- `Durchsuche 6 Quellen (Runde 1/4)...`
- `Bewerte Informationsqualitaet (nach Runde 1/4)...`
- `Formuliere Antwort (nach 3 Runden)...`

If you want only the final answer chunks without the intermediate status updates, disable progress explicitly:

```python
from inqtrix import ResearchAgent

agent = ResearchAgent()
for chunk in agent.stream("Meine Frage", include_progress=False):
    print(chunk, end="", flush=True)
```

Use `include_progress=True` for terminal tools and demos. Use `include_progress=False` when another program consumes the stream and should only receive answer text.

Example on how to use the different parameters:
```python
from inqtrix import ResearchAgent


QUESTION = "Was ist der aktuelle Stand der GKV-Reform?"
USE_STREAMING = True
INCLUDE_PROGRESS = True


def main() -> None:
    agent = ResearchAgent()

    if USE_STREAMING:
        for chunk in agent.stream(
            QUESTION,
            include_progress=INCLUDE_PROGRESS,
        ):
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

### As an HTTP Server

No `main.py` is required for the server. Start it directly with `python -m inqtrix`.

#### Option A: env-only server mode

Use this path when one LiteLLM- or OpenAI-compatible endpoint can serve both the reasoning models and the search model.

```bash
cp .env.example .env
# edit .env

# Start server
python -m inqtrix
```

Example `.env`:

```dotenv
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_API_KEY=sk-...
REASONING_MODEL=gpt-4o
SEARCH_MODEL=perplexity-sonar-pro-agent
CLASSIFY_MODEL=gpt-4o-mini
SUMMARIZE_MODEL=gpt-4o-mini
EVALUATE_MODEL=gpt-4o-mini
```

You can also set the same variables directly in the current shell instead of using `.env`:

```bash
export LITELLM_BASE_URL="http://localhost:4000/v1"
export LITELLM_API_KEY="sk-..."
export REASONING_MODEL="gpt-4o"
export SEARCH_MODEL="perplexity-sonar-pro-agent"
export CLASSIFY_MODEL="gpt-4o-mini"
export SUMMARIZE_MODEL="gpt-4o-mini"
export EVALUATE_MODEL="gpt-4o-mini"

python -m inqtrix
```

#### Option B: server mode with YAML routing

Use this path when different roles should hit different providers or different API endpoints. In server mode, `python -m inqtrix` auto-detects `inqtrix.yaml`, `inqtrix.yml`, or `.inqtrix.yaml` in the current working directory.

If your config file lives elsewhere, set `INQTRIX_CONFIG=/path/to/custom.yaml` before starting the server.

```bash
cp .env.example .env
cp inqtrix.yaml.example inqtrix.yaml

# edit .env and inqtrix.yaml

python -m inqtrix
```

Secrets belong in `.env`; YAML should only reference them via `${ENV_VAR}` placeholders.

```dotenv
# .env
OPENAI_API_KEY=sk-...
PERPLEXITY_API_KEY=pplx-...
```

```yaml
# inqtrix.yaml
providers:
    openai:
        base_url: "https://api.openai.com/v1"
        api_key: "${OPENAI_API_KEY}"

    perplexity:
        base_url: "https://api.perplexity.ai"
        api_key: "${PERPLEXITY_API_KEY}"

models:
    strong:
        provider: openai
        model_id: "gpt-4o"
        params:
            temperature: 0.0

    small:
        provider: openai
        model_id: "gpt-4o-mini"

    web:
        provider: perplexity
        model_id: "sonar-pro"

agents:
    default:
        roles:
            reasoning: strong
            classify: small
            summarize: small
            evaluate: small
            search: web
        settings:
            max_rounds: 4
            confidence_stop: 8
            high_risk_classify_escalate: true
            high_risk_evaluate_escalate: true
```

Role mapping in the default pipeline:

- `search`: web search calls
- `classify`: initial classification and decomposition; may escalate to `reasoning` for high-risk questions
- `summarize`: parallel result summarization and claim extraction
- `evaluate`: evidence sufficiency check; may escalate to `reasoning` for high-risk questions
- `reasoning`: query planning and final answer synthesis

If `classify`, `summarize`, or `evaluate` are omitted, they fall back to `reasoning`.

The active model routing is defined by `agents.<name>.roles`. The optional `fallbacks:` schema is validated, but it is not currently used as a runtime retry chain.

When no YAML file is found in server mode, the system falls back to environment-variable configuration. `.env` is auto-loaded for local development in both library and server mode. The YAML schema is defined in `config.py`; the bridge to internal settings is in `config_bridge.py`.

The server exposes an OpenAI-compatible API at `http://localhost:5100/v1/chat/completions`. For staging or production, inject the same variables via the real process environment or a secret manager instead of relying on `.env`.

### HTTP Streaming

The HTTP endpoint also supports streaming via standard OpenAI-style SSE.

```bash
curl -N http://localhost:5100/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "research-agent",
        "messages": [
            {"role": "user", "content": "Was ist der aktuelle Stand der GKV-Reform?"}
        ],
        "stream": true
    }'
```

When `stream` is `true`, progress updates are included by default before the final answer chunks. If you want SSE answer chunks without the intermediate status messages, pass `"include_progress": false`:

```bash
curl -N http://localhost:5100/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "research-agent",
        "messages": [
            {"role": "user", "content": "Was ist der aktuelle Stand der GKV-Reform?"}
        ],
        "stream": true,
        "include_progress": false
    }'
```

Library streaming yields plain text chunks. HTTP streaming yields SSE chunks in the OpenAI-compatible `data: {...}` format.

## Provider Interfaces

There are two different integration layers in Inqtrix:

- The built-in OpenAI-/LiteLLM-compatible providers expect an OpenAI-style chat completions wire format.
- Custom `LLMProvider` and `SearchProvider` implementations only need to satisfy the Python method interface. Inside your adapter, the upstream API can look completely different.

### LLM Response Format

If you use the built-in `LiteLLM` provider or the YAML-based `MultiClientLLMProvider`, Inqtrix reads the response text from one of these OpenAI-compatible shapes:

- normal chat completion: `choices[0].message.content`
- SSE chunk stream encoded as text: `choices[*].delta.content`
- optional token metadata: `usage.prompt_tokens` and `usage.completion_tokens`

Minimal non-stream example:

```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Dies ist die Antwort."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 48
    }
}
```

Minimal SSE-style example. This is also accepted even if an upstream proxy returns the entire SSE payload as a plain string:

```text
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Dies "}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ist die Antwort."}}],"usage":{"prompt_tokens":120,"completion_tokens":48}}

data: [DONE]
```

If you implement your own `LLMProvider`, none of the raw HTTP payload format matters to Inqtrix. The internal interface is simply:

- `complete(...) -> str`
- `summarize_parallel(...) -> tuple[str, int, int]`
- optional `complete_with_metadata(...) -> LLMResponse` if you want token counts on normal completions

### Search Response Format

The built-in search adapter is `PerplexitySearch`. It also uses OpenAI-compatible chat completions, but it expects search-specific metadata in addition to the answer text.

The current request shape sent to the search model is:

```python
client.chat.completions.create(
        model="<search-model>",
        messages=[{"role": "user", "content": query}],
        timeout=...,
        stream=False,
        extra_body={
                "web_search_options": {
                        "search_context_size": "high",
                        "search_mode": "web",
                        "num_search_results": 20,
                }
        },
)
```

Depending on the node state, the adapter may also add these optional fields inside `web_search_options`:

- `search_recency_filter`
- `search_language_filter`
- `search_domain_filter`
- `return_related_questions`
- `search_mode="academic"` for academic queries

On the response side, Inqtrix uses:

- answer text from `choices[0].message.content` or SSE `delta.content`
- top-level `citations: [...]` if present
- top-level `related_questions: [...]` if present
- optional `usage.prompt_tokens` / `usage.completion_tokens`

Minimal search response example:

```json
{
    "id": "chatcmpl-search-123",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Kurzantwort mit Web-Ergebnissen."
            },
            "finish_reason": "stop"
        }
    ],
    "citations": [
        "https://example.com/report",
        "https://example.org/briefing"
    ],
    "related_questions": [
        "Welche Reformschritte sind noch offen?"
    ],
    "usage": {
        "prompt_tokens": 300,
        "completion_tokens": 120
    }
}
```

If `citations` are missing, Inqtrix falls back to extracting URLs from the answer text. That fallback works, but explicit `citations` are preferable because they are much more reliable for source tracking.

### Current Sonar Parameters

For the current Perplexity Sonar path, the built-in adapter sets these defaults:

- `stream=False`
- `messages=[{"role": "user", "content": query}]`
- `extra_body.web_search_options.search_context_size="high"`
- `extra_body.web_search_options.search_mode="web"`
- `extra_body.web_search_options.num_search_results=20`

Then the runtime may add recency, language, domain, and related-question flags depending on the classified question.

In YAML mode, `models.<name>.params` for the search model are forwarded as extra request kwargs. One important caveat: `extra_body` is only shallow-merged. If you override `extra_body.web_search_options`, provide the full nested object rather than only one nested key.

### What This Means for Brave, Azure Foundry, or Other Grounded Search APIs

This is the important distinction:

- If the backend is genuinely OpenAI chat-completions compatible and its extra search knobs can be expressed as normal kwargs or `extra_body`, you may be able to adapt it with model params.
- If the backend uses different request fields, different grounding configuration, tool-calling, or returns citations in another shape, simply swapping `search_model` is usually not enough.

For Brave Search, Bing-grounded Azure Foundry, or similar backends, the safe path is usually a dedicated `SearchProvider` adapter that translates the upstream API into Inqtrix's expected internal search shape:

```python
{
        "answer": "Summarised answer text",
        "citations": ["https://source1.com", "https://source2.com"],
        "related_questions": [],
        "_prompt_tokens": 0,
        "_completion_tokens": 0,
}
```

That advice is not speculation; it follows from the current code path. The built-in search adapter is explicitly Perplexity-shaped today.

## Result Structure

```python
result = agent.research("...")

result.answer                       # Markdown-formatted answer
result.metrics.confidence           # 1-10 confidence score
result.metrics.rounds               # Number of research rounds
result.metrics.elapsed_seconds      # Total time
result.metrics.total_citations      # Number of sources found
result.metrics.total_queries        # Number of search queries
result.metrics.aspect_coverage      # 0.0-1.0 completeness
result.metrics.sources.quality_score  # 0.0-1.0 source quality
result.metrics.claims.quality_score   # 0.0-1.0 claim verification rate
result.top_sources                  # list[Source] with url + tier
result.top_claims                   # list[Claim] with status, evidence counts, primary-need, source tiers

# Full JSON serialisation
print(result.model_dump_json(indent=2))

# Flexible public export view
from inqtrix import ResearchResultExportOptions

payload = result.to_export_payload(ResearchResultExportOptions(
    include_sources=False,
    max_claims=5,
))
```

## Architecture Overview

```
inqtrix/
├── __init__.py          # Public API exports
├── __main__.py          # Entry point for `python -m inqtrix`
├── agent.py             # ResearchAgent + AgentConfig (main entry point)
├── result.py            # ResearchResult, Source, Claim, Metrics (Pydantic)
├── graph.py             # LangGraph state machine orchestration
├── nodes.py             # 5 pipeline nodes (classify, plan, search, evaluate, answer)
├── providers.py         # LLM + Search provider ABCs and implementations
├── strategies.py        # 6 pluggable strategy ABCs and defaults
├── settings.py          # Pydantic Settings (env-var configuration)
├── config.py            # YAML configuration schema and loader
├── config_bridge.py     # Bridge: YAML config -> Settings + Providers
├── state.py             # AgentState TypedDict (48 fields)
├── prompts.py           # German-language prompt templates
├── session.py           # In-memory session store for follow-ups
├── streaming.py         # SSE streaming utilities
├── routes.py            # FastAPI HTTP endpoints
├── app.py               # FastAPI application factory
├── domains.py           # Domain whitelists for source tiering
├── text.py              # Tokenisation, stopwords, normalisation
├── urls.py              # URL normalisation, extraction, sanitisation
├── json_helpers.py      # Robust JSON parsing from LLM output
├── constants.py         # Default timeouts and thresholds
├── parity.py            # Baseline regression testing utilities
└── exceptions.py        # AgentTimeout, AgentRateLimited
```

See [Agent-Architecture.md](Agent-Architecture.md) for the full technical reference.

### Where to Change What

| Goal | Files to Touch | Architecture Reference |
|------|---------------|----------------------|
| Add a new search backend | `providers.py` (implement `SearchProvider` ABC) | [Section 5 — Provider Abstractions](Agent-Architecture.md#5-provider-abstractions) |
| Add a new LLM backend | `providers.py` (implement `LLMProvider` ABC) | [Section 5](Agent-Architecture.md#5-provider-abstractions) |
| Change source quality tiers | `strategies.py` (implement `SourceTieringStrategy`), `domains.py` (domain lists) | [Section 13 — Source Tiering](Agent-Architecture.md#13-source-tiering) |
| Customise claim extraction | `strategies.py` (implement `ClaimExtractionStrategy`) | [Section 14 — Claims](Agent-Architecture.md#14-claim-extraction-and-consolidation) |
| Customise claim dedup/consolidation | `strategies.py` (implement `ClaimConsolidationStrategy`) | [Section 14](Agent-Architecture.md#14-claim-extraction-and-consolidation) |
| Change context pruning logic | `strategies.py` (implement `ContextPruningStrategy`) | [Section 6 — Strategies](Agent-Architecture.md#6-strategy-abstractions) |
| Change risk scoring | `strategies.py` (implement `RiskScoringStrategy`) | [Section 8 — Classify](Agent-Architecture.md#8-node-1-classify) |
| Change stop/continue heuristics | `strategies.py` (implement `StopCriteriaStrategy`, 10 methods) | [Section 16 — Stop Logic](Agent-Architecture.md#16-evaluation-and-stop-logic) |
| Add/rewire a graph node | `nodes.py` (node function), `graph.py` (wiring) | [Section 7 — State Machine](Agent-Architecture.md#7-state-machine-and-agent-state) |
| Change prompt templates | `prompts.py` | [Section 12 — Answer](Agent-Architecture.md#12-node-5-answer) |
| Add new state fields | `state.py` (add to `AgentState` TypedDict) | [Section 7](Agent-Architecture.md#7-state-machine-and-agent-state) |
| Add a new HTTP endpoint | `routes.py`, `app.py` | [Section 19 — HTTP Server](Agent-Architecture.md#19-http-server-layer) |
| Change timeouts/thresholds | `constants.py` (defaults), `settings.py` (env-var config), `config.py` (YAML schema) | [Section 4 — Configuration](Agent-Architecture.md#4-configuration-system), [Section 17 — Timeouts](Agent-Architecture.md#17-timeout-and-error-architecture) |
| Change session/follow-up behaviour | `session.py` | [Section 18 — Sessions](Agent-Architecture.md#18-follow-ups-and-session-reuse) |
| Add domain allow/blocklists | `domains.py` | [Section 10 — Search](Agent-Architecture.md#10-node-3-search) |
| Add regression baselines | `parity.py`, `tests/integration/` | [Agent-Architecture.md](Agent-Architecture.md) |

All strategy and provider customisations are passed via `AgentConfig` — no subclassing of `ResearchAgent` required. See the examples below.

## Extending Inqtrix

### Custom Search Provider

Implement the `SearchProvider` ABC to integrate any search engine. This is the recommended path for Brave Search, Bing-grounded Azure Foundry, or any other backend whose request or response format is not Perplexity-like:

```python
from inqtrix import ResearchAgent, AgentConfig, SearchProvider

class BingSearch(SearchProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key

    def search(
        self,
        query: str,
        *,
        search_context_size: str = "high",
        recency_filter: str | None = None,
        language_filter: list[str] | None = None,
        domain_filter: list[str] | None = None,
        search_mode: str | None = None,
        return_related: bool = False,
        deadline: float | None = None,
    ) -> dict:
        # Call Bing API, return:
        return {
            "answer": "Summarised answer text",
            "citations": ["https://source1.com", "https://source2.com"],
            "related_questions": [],
            "_prompt_tokens": 0,
            "_completion_tokens": 0,
        }

    def is_available(self) -> bool:
        return bool(self._api_key)

agent = ResearchAgent(AgentConfig(search=BingSearch(api_key="...")))
```

The algorithm does not require every search hint to be supported perfectly. These inputs are best understood as optional guidance:

- `query` is the essential input.
- `recency_filter`, `language_filter`, `domain_filter`, `search_mode`, and `return_related` are quality-improving hints.
- If your backend cannot express one of those knobs, you can ignore it or map it best-effort.

The required part is the return shape:

```python
{
    "answer": "some textual search result payload",
    "citations": ["https://source1", "https://source2"],
    "related_questions": [],
    "_prompt_tokens": 0,
    "_completion_tokens": 0,
}
```

Concrete example: direct Brave Search adapter without LiteLLM:

```python
import os

from dotenv import load_dotenv

from inqtrix import AgentConfig, BraveSearch, ResearchAgent


load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


agent = ResearchAgent(AgentConfig(
    search=BraveSearch(api_key=_require_env("BRAVE_API_KEY")),
))
```

This works with the current algorithm even though Brave does not look like the Perplexity path. The main tradeoff is that you usually get snippets instead of a model-generated search answer, so downstream summarisation and claim extraction may be somewhat weaker.

### Custom LLM Provider

Implement the `LLMProvider` ABC to use any language model:

```python
from inqtrix import ResearchAgent, AgentConfig, LLMProvider

class MyLLM(LLMProvider):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        # Call your LLM, return the response text
        return "LLM response"

    def summarize_parallel(
        self,
        text: str,
        deadline: float | None = None,
    ) -> tuple[str, int, int]:
        # Return (summary, prompt_tokens, completion_tokens)
        return ("Summary", 100, 50)

    def is_available(self) -> bool:
        return True

agent = ResearchAgent(AgentConfig(llm=MyLLM()))
```

Concrete example: direct Anthropic adapter without LiteLLM:

```python
import os

from dotenv import load_dotenv

from inqtrix import AgentConfig, AnthropicLLM, ResearchAgent


load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


agent = ResearchAgent(AgentConfig(
    llm=AnthropicLLM(
        api_key=_require_env("ANTHROPIC_API_KEY"),
        default_model="claude-3-7-sonnet-latest",
        summarize_model="claude-3-5-haiku-latest",
    ),
))
```

For full end-to-end runs, Inqtrix needs model-routing metadata for `reasoning`, `classify`, `summarize`, and `evaluate`. If your custom LLM provider does not expose a `.models` property itself, `ResearchAgent` wraps it automatically with default `ModelSettings` from the environment.

That means the Baukasten idea does work with the current algorithm. The only strict requirement is that your provider translates the upstream API into the small internal interface that the nodes and strategies expect.

### Step By Step: Use Custom Providers In `main.py`

Yes, this is supported now.

You can wire a custom search provider, a custom LLM provider, or both directly from your own `main.py`.

1. Put the external API keys into `.env` or the real process environment.

```dotenv
BRAVE_API_KEY=...
ANTHROPIC_API_KEY=...
```

2. Create `main.py` and import the built-in direct adapters.

```python
import os

from dotenv import load_dotenv

from inqtrix import AgentConfig, AnthropicLLM, BraveSearch, ResearchAgent


load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


agent = ResearchAgent(AgentConfig(
    llm=AnthropicLLM(
        api_key=_require_env("ANTHROPIC_API_KEY"),
        default_model="claude-3-7-sonnet-latest",
        summarize_model="claude-3-5-haiku-latest",
    ),
    search=BraveSearch(
        api_key=_require_env("BRAVE_API_KEY"),
    ),
    max_rounds=4,
    confidence_stop=8,
))

result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")

print(result.answer)
print(f"Confidence: {result.metrics.confidence}/10")
print(f"Sources: {result.metrics.total_citations}")
print(f"Rounds: {result.metrics.rounds}")
```

3. Run it with:

```bash
uv run python main.py
```

4. If you want streaming instead of a blocking result object, switch only the last part:

```python
for chunk in agent.stream(
    "Was ist der aktuelle Stand der GKV-Reform?",
    include_progress=True,
):
    print(chunk, end="", flush=True)
```

You can also mix custom and built-in providers:

- only custom search: set `search=BraveSearch(...)` and leave `llm=None`
- only custom LLM: set `llm=AnthropicLLM(...)` and leave `search=None`

In those mixed setups, the missing side is still auto-created from environment variables or explicit `AgentConfig` values.

This distinction matters:

- `ResearchAgent()` env mode: `.env` is auto-loaded internally.
- `load_config()` YAML mode: `.env` is auto-loaded internally before `${ENV_VAR}` resolution.
- custom-provider scripts that call `os.environ[...]`: load `.env` yourself with `load_dotenv()` or export the variables in the shell first.

### Custom Strategy

Each algorithmic concern is a pluggable strategy. Implement any ABC and pass it to `AgentConfig`:

```python
from inqtrix import ResearchAgent, AgentConfig, SourceTieringStrategy

class MySourceTiering(SourceTieringStrategy):
    """Custom source tier logic — e.g. trust internal wiki."""

    def tier_for_url(self, url: str) -> str:
        if "internal-wiki.example.com" in url:
            return "primary"
        return "unknown"

    def quality_from_urls(self, urls: list[str]) -> tuple[dict[str, int], float]:
        counts = {"primary": 0, "mainstream": 0, "stakeholder": 0, "unknown": 0, "low": 0}
        for url in urls:
            counts[self.tier_for_url(url)] += 1
        total = len(urls) or 1
        weights = {"primary": 1.0, "mainstream": 0.8, "stakeholder": 0.45, "unknown": 0.35, "low": 0.1}
        score = sum(weights[self.tier_for_url(u)] for u in urls) / total
        return counts, score

agent = ResearchAgent(AgentConfig(
    source_tiering=MySourceTiering(),
))
```

> **Note:** Some ABCs like `StopCriteriaStrategy` have many abstract methods (10 methods covering the full heuristic cascade). See [Agent-Architecture.md](Agent-Architecture.md) Section 16 for the full method list. Simpler ABCs like `SourceTieringStrategy` (2 methods) or `ContextPruningStrategy` (1 method) are easier starting points.

Available strategy ABCs:

| ABC | Concern | Default Implementation |
|-----|---------|----------------------|
| `SourceTieringStrategy` | Classify URLs into quality tiers | `DefaultSourceTiering` |
| `ClaimExtractionStrategy` | Extract structured claims from search results | `LLMClaimExtractor` |
| `ClaimConsolidationStrategy` | Deduplicate and verify claims | `DefaultClaimConsolidator` |
| `ContextPruningStrategy` | Prune context blocks by relevance | `RelevanceBasedPruning` |
| `RiskScoringStrategy` | Score question risk, derive aspects | `KeywordRiskScorer` |
| `StopCriteriaStrategy` | Decide when to stop researching | `MultiSignalStopCriteria` |

### Custom Graph Topology

The LangGraph state machine is configurable via `GraphConfig`:

```python
from inqtrix.graph import GraphConfig, build_graph, default_graph_config
from inqtrix.providers import ProviderContext
from inqtrix.strategies import StrategyContext

# Get default config, then modify
config = default_graph_config(providers, strategies, settings)

# Add a custom node
config.nodes["fact_check"] = my_fact_check_node
config.edges.append(("evaluate", "fact_check"))
# Rewire: fact_check -> answer instead of evaluate -> answer

graph = build_graph(config)
```

## Configuration Reference

For local development, place these values in `.env`; it is auto-loaded. In shared environments, pass the same variables through the real process environment, your container orchestrator, or a dedicated secret manager.

### Configuration Sources

Inqtrix supports all common ways of supplying environment variables:

1. Real process environment variables, for example via `export ...` in your shell, CI/CD secrets, Docker `-e`, Compose `environment:`, Kubernetes `env:`, or your hosting platform's secret store.
2. A local `.env` file for development only.
3. Built-in defaults for non-sensitive values.

If the same variable exists in multiple places, the real process environment wins over `.env`. That means `export LITELLM_API_KEY=...` temporarily overrides the value in `.env`, which is usually exactly what you want for debugging, CI jobs, and production deployments.

### Deployment Guidance

Recommended setup by environment:

1. Local development: use `.env` plus `inqtrix.yaml` if you want structured provider and model routing.
2. One-off shell runs: use `export` in the terminal when you want temporary overrides without editing `.env`.
3. CI/CD: store secrets in the CI system's secret store and expose them as environment variables in the job.
4. Containers and orchestration: inject environment variables from the platform, for example Docker Compose env vars, Kubernetes Secrets, or your cloud runtime's secret integration.

Do not commit `.env` files, do not put plain-text secrets into YAML, and do not rely on checked-in config files for production credentials.

### Environment Variables

**Models:**

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_MODEL` | `claude-opus-4.6-agent` | Primary LLM for reasoning |
| `SEARCH_MODEL` | `perplexity-sonar-pro-agent` | Web search model |
| `CLASSIFY_MODEL` | *(reasoning)* | Optional dedicated classify model |
| `SUMMARIZE_MODEL` | *(reasoning)* | Optional dedicated summarise model |
| `EVALUATE_MODEL` | *(reasoning)* | Optional dedicated evaluate model |

**Server connection:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://litellm-proxy:4000/v1` | LLM gateway URL |
| `LITELLM_API_KEY` | `sk-placeholder` | LLM gateway API key |

**Agent behaviour:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ROUNDS` | `4` | Maximum research loop iterations |
| `CONFIDENCE_STOP` | `8` | Confidence threshold to stop (1-10) |
| `MAX_CONTEXT` | `12` | Max context blocks retained |
| `FIRST_ROUND_QUERIES` | `6` | Queries in first round |
| `ANSWER_PROMPT_CITATIONS_MAX` | `60` | Max citations in answer prompt |
| `MAX_QUESTION_LENGTH` | `10000` | Max input question length (chars) |
| `TESTING_MODE` | `false` | Enable test endpoint |

**Timeouts:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TOTAL_SECONDS` | `300` | Hard deadline for entire run |
| `REASONING_TIMEOUT` | `120` | Per-call LLM timeout (seconds) |
| `SEARCH_TIMEOUT` | `60` | Per-call search timeout (seconds) |
| `SUMMARIZE_TIMEOUT` | `60` | Per-call summarise/claims timeout |

**Risk scoring:**

| Variable | Default | Description |
|----------|---------|-------------|
| `HIGH_RISK_SCORE_THRESHOLD` | `4` | Risk score to trigger model escalation |
| `HIGH_RISK_CLASSIFY_ESCALATE` | `true` | Use reasoning model for classify when high-risk |
| `HIGH_RISK_EVALUATE_ESCALATE` | `true` | Use reasoning model for evaluate when high-risk |

**Search cache:**

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_CACHE_MAXSIZE` | `256` | Max cached search results |
| `SEARCH_CACHE_TTL` | `3600` | Cache TTL in seconds |

**HTTP server (only relevant when running as server):**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT` | `3` | Max concurrent agent runs |
| `MAX_MESSAGES_HISTORY` | `20` | Max messages for history extraction |
| `SESSION_TTL_SECONDS` | `1800` | Session TTL (30 minutes) |
| `SESSION_MAX_COUNT` | `20` | Max sessions in memory (LRU eviction) |
| `SESSION_MAX_CONTEXT_BLOCKS` | `8` | Max context blocks per session |
| `SESSION_MAX_CLAIM_LEDGER` | `50` | Max claim ledger entries per session |

### AgentConfig Fields

`AgentConfig` mirrors the library-relevant agent, timeout, cache and
provider settings above. Model names live on provider constructors, not on
`AgentConfig`. Server-only deployment settings such as
`MAX_CONCURRENT` and `SESSION_*` stay on the HTTP server side.

```python
from inqtrix import AgentConfig, LiteLLM

AgentConfig(
    llm=LiteLLM(api_key="...", default_model="gpt-4o"),
    max_rounds=3,
    confidence_stop=7,
    answer_prompt_citations_max=40,
)
```

## Testing

Testing in this repository currently has multiple layers. They serve different purposes and should not be confused with each other.

| Layer | Command | Real provider calls? | What it covers | Prerequisites |
|-------|---------|----------------------|----------------|---------------|
| Automated local test suite | `uv run pytest tests/ -v` | No | Core logic, config loading, YAML bridge, provider normalisation, streaming behavior, graph wiring, result shaping, parity/report tooling, custom adapters via stubs/monkeypatching | Editable install with dev dependencies |
| Parity asset validation | `inqtrix-parity contract` | No | Checks the canonical question set and local parity asset structure under `tests/integration/` | Editable install |
| Local parity run against the HTTP test endpoint | `inqtrix-parity run --endpoint http://127.0.0.1:5100` | Yes | Executes the canonical questions against a running local Inqtrix server via `/v1/test/run` and stores structured run artifacts | Running server, valid provider config, `TESTING_MODE=true` |
| Manual live smoke test | `uv run python main.py` or `uv run python examples/...` | Yes | Real end-to-end agent run with your configured providers; useful for manual validation, debugging, and checking actual external integrations | Valid provider config and reachable upstream services |

Important distinction:

- `pytest` is the automated regression suite. It primarily tests internal behavior with fixtures, stubs, monkeypatching, and local reference data.
- `tests/integration/` currently stores canonical parity questions and baselines for the parity tooling. It is not a fully automated live-provider test harness on its own.
- Running `main.py`, a script in `examples/`, or a parity run against `/v1/test/run` can perform real external calls when providers are configured.

In other words: running a script is not the same as running the automated tests. A script run is a manual or semi-manual live check.

### Recommended Validation Flow

For local work, the practical order is:

1. Run `uv run pytest tests/ -v` for the fast offline regression check.
2. Run `inqtrix-parity contract` if you changed parity assets under `tests/integration/`.
3. Start the local server in testing mode if you want a structured end-to-end check.
4. Run `inqtrix-parity run --endpoint http://127.0.0.1:5100` for a local HTTP-based parity run.
5. Run `uv run python main.py` or one of the example scripts when you want a direct manual live smoke test against your actual provider setup.

Minimal setup for a local HTTP parity run:

```bash
# Terminal 1
export TESTING_MODE=true
uv run python -m inqtrix

# Terminal 2
inqtrix-parity run --endpoint http://127.0.0.1:5100
```

This parity run goes through the local server and can hit the real configured providers. It is therefore slower and more failure-prone than `pytest`, but it gives you a much better end-to-end signal.

### What `pytest` Covers Today

The current automated suite focuses on repository-local correctness and regressions, especially:

- configuration loading from environment variables, `.env`, and YAML
- YAML-to-runtime bridging and model/provider resolution
- provider response normalization and adapter interfaces
- streaming behavior and progress propagation
- graph wiring, orchestration export shape, and result serialization
- source tiering, claim consolidation, context pruning, text and URL utilities
- parity comparison and report generation logic

The suite does **not** currently guarantee that every documented provider combination has been exercised against the real external service. That gap is exactly why the repository status warning above is explicit.

```bash
uv run pytest tests/ -v
```

## Parity Tooling

The canonical parity questions live in `tests/integration/questions.json`.

```bash
# Validate local parity assets
inqtrix-parity contract

# Import the reference baselines from the litellm repo into Inqtrix
inqtrix-parity import-baselines /path/to/litellm/research-agent/tests/baselines

# Run the canonical suite against a local server in TESTING_MODE
inqtrix-parity run --endpoint http://127.0.0.1:5100

# Compare a saved run against baseline artifacts
inqtrix-parity compare tests/integration/runs/run_YYYY-MM-DD_HH-MM-SS.json \
    --baseline-dir /path/to/baselines

# Optional: add an LLM-assisted diagnostic report over queries, logs and answers
inqtrix-parity compare tests/integration/runs/run_YYYY-MM-DD_HH-MM-SS.json \
    --llm-analysis
```

For editable installs, the same commands are available through `python scripts/parity_runner.py`.

`compare` always performs a deterministic baseline-vs-run check first. `--llm-analysis` adds a second, optional diagnostic layer that uses the configured analysis model to inspect iteration logs, query trajectories, stop logic and answer quality.

Each `compare` run now emits three deterministic artifacts into the report directory:

- `report_*.json`: full machine-readable compare result with per-question issues, structured metric checks, and exported top-claim snapshots.
- `report_*.md`: markdown summary with overview and flagged-check tables plus claim snapshot sections for quick audits.
- `report_*.csv`: flat `question x metric x status` export for spreadsheets, diffing and cross-run analysis, including compact baseline/current claim snapshots.

You can configure the analysis model through the YAML Pydantic interface:

```yaml
parity:
    analysis_model: gpt4
    analysis_timeout: 180
```

## License

Copyright (c) 2026 Babak Zandi.

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for the full license text, warranty disclaimer, and limitation of liability.

## Acknowledgments

Inqtrix is built on the following open-source libraries:

| Library | License | Purpose |
|---------|---------|---------|
| [FastAPI](https://github.com/tiangolo/fastapi) | MIT | HTTP server and API endpoints |
| [Uvicorn](https://github.com/encode/uvicorn) | BSD-3-Clause | ASGI server |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | Apache-2.0 | LLM and search provider communication |
| [LangGraph](https://github.com/langchain-ai/langgraph) | MIT | State machine orchestration |
| [Pydantic](https://github.com/pydantic/pydantic) / [Pydantic Settings](https://github.com/pydantic/pydantic-settings) | MIT | Data validation and configuration |
| [cachetools](https://github.com/tkem/cachetools) | MIT | TTL-based search result caching |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | YAML configuration loading |

**Dev dependencies:**

| Library | License | Purpose |
|---------|---------|---------|
| [pytest](https://github.com/pytest-dev/pytest) | MIT | Test framework |
| [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio) | Apache-2.0 | Async test support |
| [Requests](https://github.com/psf/requests) | Apache-2.0 | HTTP client for integration tests |

## Third-Party Services and Output Notice

When configured to use external model, search, or API providers, this project may transmit prompts, context, search queries, and related request data to those third-party services.

Use of third-party services is governed by their respective terms, privacy policies, and data-processing practices. Users and operators are solely responsible for ensuring that their use of this project and any connected services complies with applicable law, contractual obligations, confidentiality requirements, and internal policies. Do not assume that any provider integration, default configuration, or example workflow included in this repository satisfies your legal, security, or data-protection obligations.

Outputs generated by this project or by connected third-party providers are provided for informational purposes only and do not constitute legal, medical, financial, or other professional advice. Independent verification remains the responsibility of the user.

## AI Disclosure

This project was developed with assistance from AI tools:

- **[Claude Code](https://www.anthropic.com/)** (Anthropic)
- **[GitHub Copilot](https://github.com/features/copilot)** (GitHub / Microsoft)
- **[ChatGPT](https://openai.com/chatgpt)** (OpenAI)

This disclosure is provided for transparency only. Use of this project remains subject to the terms of the [MIT License](LICENSE), including the "as is" disclaimer and limitation of liability.
