# Writing a custom provider

## Scope

How to integrate a backend that is not covered by the built-in providers. The walkthrough covers both `SearchProvider` and `LLMProvider`, shows concrete `main.py` wiring, and flags the conventions the custom adapter must follow.

## Checklist before you start

- Look at [Providers overview](overview.md) first — a dedicated adapter is only needed when model params and `extra_body` cannot express what you want.
- Confirm that no in-tree provider covers your target. Custom Azure flavours should probably subclass an existing adapter rather than starting from scratch.
- Plan for the Constructor-First convention: your adapter must not read environment variables; pass every secret and endpoint as a constructor argument.

## `SearchProvider`

```python
from inqtrix import ResearchAgent, AgentConfig, SearchProvider


class BingSearch(SearchProvider):
    def __init__(self, api_key: str) -> None:
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

### Hints are optional, return shape is not

Inqtrix passes the hints as keyword arguments but does not require every backend to honour them. `query` is the only required input; `recency_filter`, `language_filter`, `domain_filter`, `search_mode`, `return_related` are best-effort. The return shape above is **mandatory** — the downstream code breaks if a key is missing.

If your backend rejects unsupported parameters, declare the accepted subset. The search node will filter every other hint before calling your provider:

```python
class BingSearch(SearchProvider):
    supported_search_parameters = frozenset({
        "search_context_size",
        "recency_filter",
        "language_filter",
        "return_related",
    })
```

Use `search_capabilities` when the set is dynamic:

```python
from inqtrix.providers.base import SearchProviderCapabilities


class MySearch(SearchProvider):
    @property
    def search_capabilities(self) -> SearchProviderCapabilities:
        return SearchProviderCapabilities(
            supported_parameters=frozenset({"domain_filter"})
        )
```

### Add a `search_model` property

`SearchProvider.search_model` has a loud default (`"<ClassName>(unknown)"`) so `/health` and `/v1/stacks` stay useful even for external subclasses. Override it to name the backend (see the built-in providers in [Providers overview](overview.md)):

```python
@property
def search_model(self) -> str:
    return "bing-websearch-v7"
```

### Direct Brave example (no LiteLLM)

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

The script reads `BRAVE_API_KEY` in user code, not inside the adapter — that keeps the Constructor-First invariant intact.

## `LLMProvider`

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
        return "LLM response"

    def summarize_parallel(
        self,
        text: str,
        deadline: float | None = None,
    ) -> tuple[str, int, int]:
        return ("Summary", 100, 50)

    def is_available(self) -> bool:
        return True


agent = ResearchAgent(AgentConfig(llm=MyLLM()))
```

### Exposing a `.models` property

`ResearchAgent` expects the provider to tell the strategies layer which model each role resolves to. If your adapter exposes the `ModelSettings`-shaped `.models` property, the strategies consume it directly. If not, `ResearchAgent` wraps the provider with default `ModelSettings` from the environment (legacy safety net); however, that path can misreport model names on non-LiteLLM stacks. Provide `.models` explicitly.

### Direct Anthropic example (no LiteLLM)

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
        default_model="claude-sonnet-4-6",
        classify_model="claude-haiku-4-5",
        summarize_model="claude-haiku-4-5",
        evaluate_model="claude-sonnet-4-6",
        # thinking={"type": "adaptive"},
    ),
))
```

## Mixing custom and built-in providers

You can set only one side and leave the other `None`:

- Only custom search: `search=BraveSearch(...)` with `llm=None`. `ResearchAgent` will auto-create the LLM from environment variables.
- Only custom LLM: `llm=AnthropicLLM(...)` with `search=None`. Same auto-creation for the search side.

The distinction matters when it comes to `.env` loading:

- `ResearchAgent()` env mode: `.env` is auto-loaded internally by the settings layer.
- `load_config()` YAML mode: `.env` is auto-loaded before `${ENV_VAR}` resolution.
- Custom-provider scripts that call `os.environ[...]` themselves must call `load_dotenv()` or rely on exported shell variables.

## Full custom `main.py`

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
        default_model="claude-sonnet-4-6",
        classify_model="claude-haiku-4-5",
        summarize_model="claude-haiku-4-5",
    ),
    search=BraveSearch(api_key=_require_env("BRAVE_API_KEY")),
    max_rounds=4,
    confidence_stop=8,
))

result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")

print(result.answer)
print(f"Confidence: {result.metrics.confidence}/10")
print(f"Sources: {result.metrics.total_citations}")
print(f"Rounds: {result.metrics.rounds}")
```

Run it with:

```bash
uv run python main.py
```

For streaming output, replace the last block with a `for chunk in agent.stream(...)` loop.

## Related docs

- [Providers overview](overview.md)
- [Strategies](../architecture/strategies.md)
- [Agent config](../configuration/agent-config.md)
- [Library mode](../deployment/library-mode.md)
