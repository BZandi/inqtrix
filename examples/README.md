# Examples

## Scope

Runnable examples for the three main ways to use Inqtrix: quickstart scripts, explicit provider stacks, and HTTP server stacks. These scripts are intentionally small entry points; provider credentials still come from your environment or a local `.env` file.

## Quickstart

| Script | Use it for |
|--------|------------|
| [`quickstart/basic_env.py`](quickstart/basic_env.py) | Minimal `ResearchAgent()` run from env-based configuration. |
| [`quickstart/yaml_config.py`](quickstart/yaml_config.py) | Loading `inqtrix.yaml` and bridging it into providers/settings. |
| [`quickstart/streaming.py`](quickstart/streaming.py) | Iterating over `agent.stream(...)` chunks. |

Run with:

```bash
uv run python examples/quickstart/basic_env.py
```

## Provider stacks

Provider stacks run one research question in-process through explicit Baukasten constructors:

| Script | LLM | Search |
|--------|-----|--------|
| [`provider_stacks/litellm_perplexity.py`](provider_stacks/litellm_perplexity.py) | LiteLLM | Perplexity |
| [`provider_stacks/anthropic_perplexity.py`](provider_stacks/anthropic_perplexity.py) | Anthropic | Perplexity |
| [`provider_stacks/anthropic_perplexity_chat.py`](provider_stacks/anthropic_perplexity_chat.py) | Anthropic | Perplexity |
| [`provider_stacks/bedrock_perplexity.py`](provider_stacks/bedrock_perplexity.py) | Bedrock | Perplexity |
| [`provider_stacks/azure_openai_perplexity.py`](provider_stacks/azure_openai_perplexity.py) | Azure OpenAI | Perplexity |
| [`provider_stacks/azure_openai_web_search.py`](provider_stacks/azure_openai_web_search.py) | Azure OpenAI | Azure OpenAI web search tool |
| [`provider_stacks/azure_openai_bing.py`](provider_stacks/azure_openai_bing.py) | Azure OpenAI | Azure Foundry Bing Grounding |
| [`provider_stacks/azure_foundry_web_search.py`](provider_stacks/azure_foundry_web_search.py) | Azure OpenAI | Azure Foundry Web Search |

The interactive Anthropic + Perplexity script opens a terminal REPL and keeps chat history in the process:

```bash
uv run python examples/provider_stacks/anthropic_perplexity_chat.py
```

## Custom providers

| Script | Shows |
|--------|-------|
| [`custom_providers/brave_search.py`](custom_providers/brave_search.py) | Direct `BraveSearch` wiring. |
| [`custom_providers/anthropic_and_brave.py`](custom_providers/anthropic_and_brave.py) | Anthropic LLM with Brave search. |
| [`custom_providers/anthropic_with_env_search.py`](custom_providers/anthropic_with_env_search.py) | Custom LLM with auto-created search provider. |

Use these when you want to copy the constructor-first pattern into your own script.

## Webserver stacks

`examples/webserver_stacks/` exposes the same provider combinations over the OpenAI-compatible HTTP API. Start one stack:

```bash
uv run python examples/webserver_stacks/anthropic_perplexity.py
```

Or start the multi-stack server, which registers every stack whose required env vars are present:

```bash
uv run python examples/webserver_stacks/multi_stack.py
```

The operational reference for env vars, logging, TLS, API keys, CORS, per-request overrides, multi-stack routing, and cancel behaviour is [`webserver_stacks/README.md`](webserver_stacks/README.md).

## Azure smoke tests

The scripts under [`provider_stacks/azure_smoke_tests/`](provider_stacks/azure_smoke_tests/) make isolated live calls against Azure providers. They are not part of the offline pytest suite and require real Azure configuration:

```bash
uv run python examples/provider_stacks/azure_smoke_tests/test_llm.py
uv run python examples/provider_stacks/azure_smoke_tests/test_bing_search.py
uv run python examples/provider_stacks/azure_smoke_tests/test_foundry_web_search.py
uv run python examples/provider_stacks/azure_smoke_tests/test_openai_web_search.py
```

## Related docs

- [Docs hub](../docs/README.md)
- [Library mode](../docs/deployment/library-mode.md)
- [Web server mode](../docs/deployment/webserver-mode.md)
- [Writing a custom provider](../docs/providers/writing-a-custom-provider.md)
