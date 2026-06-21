# Provider recipes (mix and match)

> Files: `src/inqtrix/providers/__init__.py`, `src/inqtrix/settings.py`, `examples/webserver_stacks/`

## Scope

Copy-paste `.env` recipes for every supported LLM + search provider combination, configured purely through environment variables — no Python. This is the env-driven front door for the HTTP server (`python -m inqtrix`) and the Stack-mode compose stack. Library-mode users who build providers in Python keep using the explicit constructors (see [Library mode](../deployment/library-mode.md) and the scripts in [`examples/provider_stacks/`](../../examples/provider_stacks/)); this page is for the server/Stack-mode path.

**What these settings give you.** With one LLM + one search provider configured, the server runs the full agent: iterative web research with live SSE events (`/v1/runs`), the OpenAI-compatible chat endpoint (`/v1/chat/completions`), and the editor/text-improvement endpoints. The *search* provider only does web search — the *LLM* does the reasoning, planning, and answer synthesis, so the server is far more than "just web search". Features that need extra infrastructure — knowledge/RAG over your own documents, file uploads, multi-user/sharing — are **not** enabled by these recipes; see [More configuration](#more-configuration-production) below. The full endpoint surface is documented in [Web server mode](../deployment/webserver-mode.md).

## Two independent axes

The provider stack has two **independent** selectors. Any LLM pairs with any search provider:

```dotenv
INQTRIX_LLM_PROVIDER=litellm      # litellm | anthropic | azure | bedrock   (default: litellm)
INQTRIX_SEARCH_PROVIDER=perplexity # perplexity | azure_foundry              (default: perplexity)
```

Leaving both unset reproduces the historical LiteLLM + Perplexity stack exactly. An unknown value fails loudly at startup; a selected provider that is missing a required credential fails loudly at startup too (never a silent fallback to another provider).

### The matrix

LLM down the side, search across the top. Each cell links to a runnable webserver example stack (the Python equivalent of the `.env` recipe, useful for library mode or as a reference):

| LLM ↓ / Search → | `perplexity` | `azure_foundry` |
|---|---|---|
| **`litellm`** (default) | ✅ [`litellm_perplexity.py`](../../examples/webserver_stacks/litellm_perplexity.py) | ⚙ env-supported |
| **`anthropic`** | ✅ [`anthropic_perplexity.py`](../../examples/webserver_stacks/anthropic_perplexity.py) | ⚙ env-supported |
| **`azure`** | ✅ [`azure_openai_perplexity.py`](../../examples/webserver_stacks/azure_openai_perplexity.py) | ✅ [`azure_foundry_web_search.py`](../../examples/webserver_stacks/azure_foundry_web_search.py) |
| **`bedrock`** | ✅ [`bedrock_perplexity.py`](../../examples/webserver_stacks/bedrock_perplexity.py) | ⚙ env-supported |

✅ = shipped example stack · ⚙ = supported via env, no dedicated example script. Every cell is a valid runtime combination.

## Recipe shape

Each recipe is: pick **one** LLM block + **one** search block, set the model names, start the server, verify. `INQTRIX_SELECTABLE_CHAT_MODELS` (comma-separated) populates the in-app model picker and appears in every recipe so the surface is uniform — drop it if you do not want an explicit picker.

## Recipes

### LiteLLM + Perplexity (default)

Any model behind an OpenAI-compatible gateway (LiteLLM proxy, OpenRouter, vLLM, Ollama) + Perplexity search. This is the zero-config default — both selectors may be omitted.

```dotenv
# LLM axis
INQTRIX_LLM_PROVIDER=litellm
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_API_KEY=sk-...
# Search axis (independent of the LLM axis)
INQTRIX_SEARCH_PROVIDER=perplexity
PERPLEXITY_API_KEY=pplx-...
# Models
REASONING_MODEL=gpt-4o
SEARCH_MODEL=perplexity-sonar-pro-agent
# Comma-separated ids for the in-app model picker (optional)
INQTRIX_SELECTABLE_CHAT_MODELS=gpt-4o,gpt-4o-mini
```

Python equivalent: [`examples/webserver_stacks/litellm_perplexity.py`](../../examples/webserver_stacks/litellm_perplexity.py).

### Anthropic + Perplexity

Claude models via the direct Messages API + Perplexity search.

```dotenv
# LLM axis
INQTRIX_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
# Search axis
INQTRIX_SEARCH_PROVIDER=perplexity
PERPLEXITY_API_KEY=pplx-...
# Models
REASONING_MODEL=claude-opus-4-8
TIER_HIGH_MODEL=claude-opus-4-8
TIER_MID_MODEL=claude-sonnet-4-6
TIER_FAST_MODEL=claude-haiku-4-5
TIER_HIGH_EFFORT=medium
# Comma-separated ids for the in-app model picker (optional)
INQTRIX_SELECTABLE_CHAT_MODELS=claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5
```

Python equivalent: [`examples/webserver_stacks/anthropic_perplexity.py`](../../examples/webserver_stacks/anthropic_perplexity.py).

### Azure OpenAI + Perplexity

GPT models on Azure (the model name is the **deployment** name) + Perplexity search. Authenticate with an API key **or** an Entra Service Principal (all three SP variables together).

```dotenv
# LLM axis
INQTRIX_LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=...
#   ... or Service Principal instead of the key:
# AZURE_TENANT_ID=...
# AZURE_CLIENT_ID=...
# AZURE_CLIENT_SECRET=...
# Search axis
INQTRIX_SEARCH_PROVIDER=perplexity
PERPLEXITY_API_KEY=pplx-...
# Models (deployment names, not model ids)
REASONING_MODEL=gpt-5.4
TIER_HIGH_MODEL=gpt-5.4
TIER_FAST_MODEL=gpt-5.4-mini
# Comma-separated deployment names for the in-app model picker (optional)
INQTRIX_SELECTABLE_CHAT_MODELS=gpt-5.4,gpt-5.4-mini
```

Python equivalent: [`examples/webserver_stacks/azure_openai_perplexity.py`](../../examples/webserver_stacks/azure_openai_perplexity.py).

### Azure OpenAI + Azure Foundry Web Search (all-Azure)

GPT models on Azure + the Azure AI Foundry Web Search agent. The cleanest enterprise combo: one Entra Service Principal authenticates **both** axes.

```dotenv
# LLM axis
INQTRIX_LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# Search axis
INQTRIX_SEARCH_PROVIDER=azure_foundry
AZURE_AI_PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
WEB_SEARCH_AGENT_NAME=web-search-agent
# WEB_SEARCH_AGENT_VERSION=2          # optional pin
# Shared Service Principal (authenticates both the LLM and the search agent):
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
#   ... or per-axis API keys instead:
# AZURE_OPENAI_API_KEY=...
# AZURE_AI_PROJECT_API_KEY=...
# Models
REASONING_MODEL=gpt-5.4
# Comma-separated deployment names for the in-app model picker (optional)
INQTRIX_SELECTABLE_CHAT_MODELS=gpt-5.4,gpt-5.4-mini
```

Python equivalent: [`examples/webserver_stacks/azure_foundry_web_search.py`](../../examples/webserver_stacks/azure_foundry_web_search.py).

### Bedrock + Perplexity

Claude models on AWS Bedrock (credentials via the standard AWS chain — named profile, env vars, or instance role) + Perplexity search.

```dotenv
# LLM axis
INQTRIX_LLM_PROVIDER=bedrock
AWS_PROFILE=your-profile           # optional; omit to use the default AWS chain
AWS_REGION=eu-central-1
# Search axis
INQTRIX_SEARCH_PROVIDER=perplexity
PERPLEXITY_API_KEY=pplx-...
# Models (Bedrock model ids)
REASONING_MODEL=eu.anthropic.claude-opus-4-8-v1
TIER_HIGH_MODEL=eu.anthropic.claude-opus-4-8-v1
TIER_MID_MODEL=eu.anthropic.claude-sonnet-4-6
TIER_FAST_MODEL=eu.anthropic.claude-haiku-4-5
# Comma-separated model ids for the in-app model picker (optional)
INQTRIX_SELECTABLE_CHAT_MODELS=eu.anthropic.claude-opus-4-8-v1,eu.anthropic.claude-sonnet-4-6
```

Python equivalent: [`examples/webserver_stacks/bedrock_perplexity.py`](../../examples/webserver_stacks/bedrock_perplexity.py).

The `⚙` cells (e.g. Anthropic + Azure Foundry, Bedrock + Azure Foundry) work the same way: combine the chosen LLM block with the `azure_foundry` search block.

## Model knobs (the full env surface)

Every model-related knob is reachable from `.env` — the server never needs Python to expose the full experience.

| Variable | Applies to | Meaning |
|---|---|---|
| `REASONING_MODEL` | all LLMs | Primary model + fallback for unset roles |
| `TIER_HIGH_MODEL` / `TIER_MID_MODEL` / `TIER_FAST_MODEL` | all LLMs | Per-tier models (answer / plan+evaluate / classify+claim-extract) |
| `TIER_HIGH_EFFORT` / `TIER_MID_EFFORT` / `TIER_FAST_EFFORT` | all LLMs | Per-tier reasoning effort (`none`..`xhigh`) |
| `CLASSIFY_MODEL` / `CLAIM_EXTRACT_MODEL` / `EVALUATE_MODEL` / `PLAN_MODEL` / `ANSWER_MODEL` / `DIRECT_CHAT_MODEL` | all LLMs | Optional per-node overrides |
| `INQTRIX_SELECTABLE_CHAT_MODELS` | all LLMs | Comma-separated model ids offered in the UI model picker (feeds `/health.models_catalog`), e.g. `gpt-4o,gpt-4o-mini` |
| `INQTRIX_TEMPERATURE` | anthropic, azure, bedrock | Sampling temperature (LiteLLM ignores it — a warning is logged if set) |
| `INQTRIX_TOKEN_BUDGET_PARAMETER` | litellm, azure | `max_tokens` or `max_completion_tokens` (the latter for OpenAI o-series) |
| `SEARCH_MODEL` | perplexity | Explicit search model |
| `INQTRIX_SEARCH_PRESET` | perplexity | `fast-search` / `pro-search` / `deep-research` |
| `INQTRIX_SEARCH_INSTRUCTIONS` | perplexity | System instructions for the search agent |

Notes:

- **Context window and output budget are not env vars.** They come from the model-card catalogue ([`model_cards.py`](../../src/inqtrix/model_cards.py)); the constructor values are a fallback only for models without a card. See [Model cards](../configuration/model-cards.md).
- **Web search has no high/mid/fast tiers** like the LLM. Search is one model (`SEARCH_MODEL`) plus the Perplexity preset/instructions.
- A knob set for a provider that cannot honour it (e.g. `INQTRIX_TEMPERATURE` under `litellm`) is **not** silently dropped — the server logs a visible `CONFIG: ... ignoring` warning at startup.

## More configuration (production)

The recipes above cover the providers and models. Everything else lives in dedicated pages so this one stays focused — the full set of environment variables is the reference, not just what is shown here:

- **All env variables** (storage, auth, knowledge, queue, logging, limits): [Settings and env](../configuration/settings-and-env.md).
- **Infrastructure & features** (Postgres, S3 object store, Qdrant/RAG, Valkey workers — which feature needs which component): [Platform components](../getting-started/platform-components.md).
- **Authentication** (`none` / `apikey` / `local` / `oidc` / `ldap`, and how users are created): [Auth modes](../deployment/auth-modes.md).
- **Running the full stack with all of the above**: [Stack quickstart](stack-quickstart.md) and [Runbooks](../deployment/runbooks.md).

## Verify

```bash
uv run python -m inqtrix
# in a second shell:
curl http://localhost:5100/health        # llm.provider / search.provider + model identity
curl http://localhost:5100/v1/capabilities
```

`/health` reports the actually-built provider and its resolved models; if `INQTRIX_SELECTABLE_CHAT_MODELS` is set, `models_catalog` lists the picker entries. A missing required credential surfaces as a loud startup error naming the variable. The full request/response shapes for every endpoint are in [Web server mode](../deployment/webserver-mode.md); the SSE event schema is in [Run events](../observability/run-events.md).

## Related docs

- [First research run](first-research-run.md)
- [Web server mode](../deployment/webserver-mode.md) — the endpoint surface
- [Settings and env](../configuration/settings-and-env.md) — every variable
- [Platform components](platform-components.md) — RAG, files, workers
- [Providers overview](../providers/overview.md)
- [Model cards](../configuration/model-cards.md)
