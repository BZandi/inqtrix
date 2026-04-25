# Providers overview

> Package: `src/inqtrix/providers/`

## Scope

The provider package is the Baukasten surface. Every LLM backend and every search backend is an implementation of one of two ABCs and can be swapped independently. This page lists the built-ins, the contracts they satisfy, and the conventions custom providers must follow.

## Provider matrix

| Role | Provider class | Module | Backend |
|------|----------------|--------|---------|
| LLM | `LiteLLM` | `providers/litellm.py` | Any OpenAI-compatible endpoint |
| LLM | `AnthropicLLM` | `providers/anthropic.py` | Anthropic Messages API |
| LLM | `AzureOpenAILLM` | `providers/azure.py` | Azure OpenAI Chat Completions |
| LLM | `BedrockLLM` | `providers/bedrock.py` | Amazon Bedrock Converse API |
| Search | `PerplexitySearch` | `providers/perplexity.py` | Perplexity Sonar API (TTL cache) |
| Search | `BraveSearch` | `providers/brave.py` | Brave Web Search API (snippets only) |
| Search | `AzureOpenAIWebSearch` | `providers/azure_openai_web_search.py` | Native Azure OpenAI Responses-API `web_search` tool |
| Search | `AzureFoundryBingSearch` | `providers/azure_bing.py` | Azure Foundry Bing Grounding agent |
| Search | `AzureFoundryWebSearch` | `providers/azure_web_search.py` | Azure Foundry Web Search agent |

All nine providers are exported from the top-level `inqtrix` package.

## `LLMProvider` ABC

```python
class LLMProvider(ABC):
    def complete(self, prompt, *, system=None, model=None,
                 timeout=120.0, state=None, deadline=None) -> str: ...
    def summarize_parallel(self, text, deadline=None) -> tuple[str, int, int]: ...
    def is_available(self) -> bool: ...
```

Optional `complete_with_metadata(...) -> LLMResponse` lets an adapter return token counts on a normal completion.

### The `models` property

Every LLM implementation exposes `.models` — a `ModelSettings`-shaped object that captures the model names chosen at construction time. Code that needs to know which model a given role resolves to reads it **constructor-first**:

```python
provider.models.effective_classify_model    # falls back to reasoning
provider.models.effective_summarize_model   # falls back to reasoning
provider.models.effective_evaluate_model    # falls back to reasoning
provider.models.reasoning_model
```

The `resolve_summarize_model(llm, fallback)` helper in `inqtrix.strategies` is the canonical way to read the summarize model; it emits a warning marker if neither constructor path nor reasoning fallback yields a value. Use that helper when you write custom strategy wiring so provider-specific model names do not leak back to global `Settings` defaults.

## `SearchProvider` ABC

```python
class SearchProvider(ABC):
    def search(self, query, *, search_context_size="high",
               recency_filter=None, language_filter=None,
               domain_filter=None, search_mode=None,
               return_related=False, deadline=None) -> dict: ...
    def is_available(self) -> bool: ...
```

### Return shape (mandatory)

```python
{
    "answer": "Summarised answer text",
    "citations": ["https://source1.com", "https://source2.com"],
    "related_questions": [],
    "_prompt_tokens": 0,
    "_completion_tokens": 0,
}
```

The optional hints (`recency_filter`, `language_filter`, `domain_filter`, `search_mode`, `return_related`) are quality-improving but not required. A custom adapter can ignore a hint that the underlying backend cannot express.

### The `search_model` property

Every search provider exposes a `search_model: str` property so `/health` and `/v1/stacks` can show a meaningful identifier. The default on the ABC returns `"<ClassName>(unknown)"` — that loud string makes a missing override immediately visible. Overrides used by the built-ins:

| Class | `search_model` |
|-------|----------------|
| `PerplexitySearch` | the configured Sonar model id (for example `sonar-pro`) |
| `BraveSearch` | `"brave-search-api"` (constant — Brave has no model concept) |
| `AzureOpenAIWebSearch` | `f"{default_model}+web_search_tool"` |
| `AzureFoundryWebSearch` | `f"foundry-web:{name}@{version_or_latest}"` |
| `AzureFoundryBingSearch` | `f"foundry-bing:{name_or_id}@{version_or_latest}"` |

### Search capabilities

Search backends do not all support the same generic hints. A provider can declare the subset it accepts so the search node filters unsupported arguments before calling `search(...)`:

```python
class MySearch(SearchProvider):
    supported_search_parameters = frozenset({
        "search_context_size",
        "recency_filter",
        "language_filter",
        "return_related",
    })
```

For dynamic metadata, expose a `search_capabilities` property returning `SearchProviderCapabilities`. If neither field exists, Inqtrix assumes all hints are supported for backwards compatibility with older custom adapters.

## `ProviderContext`

Frozen dataclass bundling the active providers:

```python
@dataclass(frozen=True)
class ProviderContext:
    llm: LLMProvider
    search: SearchProvider
```

Every node receives `providers` via dependency injection; tests replace providers with stubs against this interface only.

## Constructor-first convention

All built-in providers honour the Constructor-First principle: they do not read environment variables directly. The example scripts under `examples/provider_stacks/` and `examples/webserver_stacks/` are the only place where `.env` translates to constructor arguments. This invariant is enforced by 82 parametrised tests in `tests/test_docstring_completeness.py` and by the Baukasten architecture guard in the replay tests.

If you integrate a new provider, follow the same pattern:

- Read no env var inside the provider module.
- Accept every secret, endpoint, deployment, and model id as a named constructor argument.
- Use `sanitize_log_message(...)` (from `runtime_logging.py`) before logging anything that may contain credentials.

## Per-provider pages

- [LiteLLM](litellm.md)
- [Anthropic](anthropic.md)
- [Bedrock](bedrock.md)
- [Azure OpenAI](azure-openai.md)
- [Azure OpenAI Web Search](azure-openai-web-search.md)
- [Azure Foundry Bing](azure-foundry-bing.md)
- [Azure Foundry Web Search](azure-foundry-web-search.md)
- [Perplexity](perplexity.md)
- [Brave](brave.md)
- [Writing a custom provider](writing-a-custom-provider.md)

## Related docs

- [Agent config](../configuration/agent-config.md)
- [Settings and env](../configuration/settings-and-env.md)
- [Architecture overview](../architecture/overview.md)
