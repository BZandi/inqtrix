# Brave Search

> File: `src/inqtrix/providers/brave.py`

## Scope

`BraveSearch` is a direct adapter for the Brave Web Search API. It returns snippets and URLs — there is no LLM-generated answer text on this backend — so it is a good choice when you want raw search results and let Inqtrix's downstream summarisation and claim extraction do the synthesis work.

## When to use it

- You want a privacy-oriented web search API.
- You accept weaker downstream synthesis (no Sonar-style pre-summary) in exchange for cheaper per-call cost.
- You need a backup search provider for resilience alongside Perplexity or an Azure search.

## Constructor

```python
from inqtrix import BraveSearch


search = BraveSearch(
    api_key="BSA-...",
    timeout_seconds=30,
)
```

| Parameter | Purpose |
|-----------|---------|
| `api_key` | Brave API key. |
| `timeout_seconds` | Per-call wall-clock timeout; capped by the run deadline. |
| `cache_maxsize` / `cache_ttl_seconds` | TTL-based LRU cache parameters; defaults mirror `PerplexitySearch`. |

## `search_model` property

Returns the constant `"brave-search-api"`. Brave has no model concept, but the string still lets `/health` and `/v1/stacks` distinguish the provider.

## Response normalisation

The adapter assembles the standard return dict by concatenating snippet text from the top results:

- `answer` — `"\n\n".join(snippet for each result)`. This is raw snippet text, not an LLM summary.
- `citations` — the result URLs in rank order.
- `related_questions` — always empty (the Brave API does not expose one).
- `_prompt_tokens` / `_completion_tokens` — always `0`.

Because the `answer` field is snippet-only, downstream claim extraction and source tiering tend to report slightly lower quality scores than with a Sonar-style provider. The claim ledger still fills up because claims are extracted from the snippet text by the summarize model.

## Errors and retries

- Transport errors (`urllib`-level) are wrapped as `BraveSearchAPIError` with status code.
- HTTP 429 / quota: raise `AgentRateLimited`.
- Per-query failure is non-fatal at the node level; the failing query is dropped and the other queries continue.

## Example

```python
from inqtrix import AgentConfig, AnthropicLLM, BraveSearch, ResearchAgent

agent = ResearchAgent(AgentConfig(
    llm=AnthropicLLM(api_key="sk-ant-...", default_model="claude-sonnet-4-6"),
    search=BraveSearch(api_key="BSA-..."),
))
```

See `examples/custom_providers/brave_search.py` for a drop-in example that keeps the LLM side env-based.

## Related docs

- [Providers overview](overview.md)
- [Perplexity](perplexity.md)
- [Writing a custom provider](writing-a-custom-provider.md)
