# Perplexity

> File: `src/inqtrix/providers/perplexity.py`

## Scope

`PerplexitySearch` is the default `SearchProvider`. It calls the native
Perplexity Agent API through the `perplexityai` SDK and supplies Inqtrix with a
typed `GroundedSearchResult`: one full provider synthesis plus per-source
records from `search_results`. Perplexity source ids are preserved as
`GroundedSource.rank` so inline answer citations such as `[2]` can be bound to
claims deterministically.

## When to use it

- You want high-quality citation-rich search with minimal configuration.
- You want Perplexity's native Agent API instead of an OpenAI-compatible proxy.
- You want Inqtrix's search cache (TTL-based LRU) without building your own.

## Constructor

```python
from inqtrix import PerplexitySearch


search = PerplexitySearch(
    api_key="pplx-...",
    preset="fast-search",                        # default; provides cited web-search answers
    model=None,                                  # optional explicit Agent model override
    cache_maxsize=256,
    cache_ttl=3600,
)
```

| Parameter | Purpose |
|-----------|---------|
| `api_key` | Perplexity API key. Server-mode auto-creation reads `PERPLEXITY_API_KEY`. |
| `preset` | Agent preset. Default `fast-search` is preferred because it returns cited web-search output out of the box. |
| `model` | Optional explicit Agent model. When set, it overrides `preset`. |
| `instructions` | Optional Agent instructions appended to the request. |
| `base_url` | Optional Perplexity-compatible endpoint override. Leave empty for the SDK default. |
| `cache_maxsize` / `cache_ttl` | TTL-based LRU cache parameters. Key = SHA-256 of query + supported hints. |

## `search_model` property

Returns the explicit model when configured, otherwise the preset label, for
example `perplexity-agent:fast-search`.

## Request shape sent to the API

```python
client.responses.create(
    input=query,
    tools=[{"type": "web_search"}],
    stream=False,
    preset="fast-search",  # or model=...
    instructions=...,
    timeout=...,
)
```

Recency, language, and domain hints are folded into the user input when the
Agent API has no native parameter for that hint.

## Response shape expected from the API

The adapter reads:

- **Answer text** — from `output_text` or message content.
- **Sources** — `output[].type == "search_results"` with `results[]` fields
  `id`, `url`, `title`, `snippet`, `date`, `last_updated`, and `source`.
- **Tokens** — `usage.input_tokens` / `usage.output_tokens`.

Minimal response example:

```json
{
  "output_text": "NVIDIA reported ... [2]",
  "output": [
    {
      "type": "search_results",
      "results": [
        {"id": 2, "url": "https://example.com/report", "title": "...", "snippet": "..."}
      ]
    }
  ],
  "usage": {"input_tokens": 300, "output_tokens": 120}
}
```

## Errors and retries

- Perplexity SDK retry loop drives transient retries.
- HTTP 429 / quota: raise `AgentRateLimited`.
- Other API errors degrade to an empty `GroundedSearchResult` with a visible
  non-fatal notice so the search node can continue and report partial failure.

## Related docs

- [Providers overview](overview.md)
- [Writing a custom provider](writing-a-custom-provider.md)
- [Nodes](../architecture/nodes.md)
