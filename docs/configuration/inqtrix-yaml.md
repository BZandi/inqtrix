# `inqtrix.yaml`

> Files: `src/inqtrix/config.py`, `src/inqtrix/config_bridge.py`, `inqtrix.yaml.example`

## Scope

YAML configuration for the HTTP server — the structured way to wire multiple providers, different models per role, and per-search-adapter parameters. YAML is only auto-detected in server mode; library scripts can still use YAML but must bridge manually.

## When YAML is used

- `python -m inqtrix` (or an `examples/webserver_stacks/*.py` script that boots the same server) auto-detects `inqtrix.yaml`, `inqtrix.yml`, or `.inqtrix.yaml` in the current working directory. `INQTRIX_CONFIG=/path/to/config.yaml` overrides auto-detection.
- `load_config(path)` + `config_bridge.config_to_settings(...)` + `create_providers_from_config(...)` is the library-mode entry point. See the example at the bottom of this page.
- `ResearchAgent()` **does not** auto-load YAML in library mode.

## Precedence

YAML sits between `AgentConfig` (library) and `Settings` (env) in the configuration hierarchy:

1. Explicit `AgentConfig` scalar fields (library only).
2. YAML (server mode, if present and well-formed).
3. `Settings` (from `.env` and process env).
4. Code defaults.

Secrets always live in `.env`; YAML references them via `${ENV_VAR}` placeholders resolved at load time.

## Minimal example

```yaml
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

## Role mapping

The server's default pipeline knows five roles:

| Role | Used for |
|------|----------|
| `reasoning` | Query planning and final answer synthesis |
| `classify` | Initial classification and decomposition (may escalate to `reasoning` for high-risk questions) |
| `summarize` | Parallel summarisation and claim extraction |
| `evaluate` | Evidence sufficiency check (may escalate to `reasoning` for high-risk questions) |
| `search` | Web search provider / model |

If `classify`, `summarize`, or `evaluate` are omitted, they fall back to `reasoning` via the `effective_*` accessors in `ModelSettings`. The optional top-level `fallbacks:` block is validated by the schema but is **not** consumed as a runtime retry chain today.

## Search-adapter selection

Search model entries drive the search adapter. Supported selectors in `models.<name>.params.search_provider`:

| Selector | Adapter |
|----------|---------|
| `perplexity` | `PerplexitySearch` |
| `azure_openai_web_search` | `AzureOpenAIWebSearch` |
| `azure_foundry_web_search` | `AzureFoundryWebSearch` |
| `azure_web_search` | Alias for `azure_foundry_web_search` |

When `search_provider` is omitted, the bridge auto-detects:

- Endpoints under `.openai.azure.com` → `AzureOpenAIWebSearch`.
- Endpoints under `.services.ai.azure.com/api/projects/...` → `AzureFoundryWebSearch`.
- Everything else → `PerplexitySearch`.

### Native Azure OpenAI web search via YAML

```yaml
providers:
    azure-openai:
        base_url: "https://my-resource.openai.azure.com/"
        api_key: "${AZURE_OPENAI_API_KEY}"

models:
    strong:
        provider: azure-openai
        model_id: "gpt-4.1-reasoning"

    web:
        provider: azure-openai
        model_id: "gpt-4.1-search"
        params:
            tool_choice: required
            user_location:
                type: approximate
                country: DE

agents:
    default:
        roles:
            reasoning: strong
            search: web
```

### `extra_body` is shallow-merged

The Perplexity adapter forwards `params.extra_body` to the OpenAI SDK as a shallow merge. If you override `extra_body.web_search_options`, provide the full nested object — otherwise unspecified nested keys are dropped.

## Library-mode bridging

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

`load_config()` auto-loads a local `.env` before resolving placeholders. For the common YAML workflow, running the HTTP server is simpler because YAML is auto-detected there.

## Related docs

- [Agent config](agent-config.md)
- [Settings and env](settings-and-env.md)
- [Providers overview](../providers/overview.md)
- [Web server mode](../deployment/webserver-mode.md)
