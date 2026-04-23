# Perplexity

> File: `src/inqtrix/providers/perplexity.py`

## Scope

`PerplexitySearch` is the default `SearchProvider`. It calls Perplexity Sonar Pro through an OpenAI-compatible chat-completions endpoint and supplies Inqtrix with answer text, citations, related questions, and token usage. This page covers the request shape, the response shape, and the knobs exposed to callers.

## When to use it

- You want high-quality citation-rich search with minimal configuration.
- You already proxy OpenAI-compatible traffic through LiteLLM and want to reuse the same proxy for search.
- You want Inqtrix's search cache (TTL-based LRU) without building your own.

## Constructor

```python
from inqtrix import PerplexitySearch


search = PerplexitySearch(
    api_key="pplx-...",
    base_url="https://api.perplexity.ai",        # or your LiteLLM proxy
    model="sonar-pro",                           # or "perplexity-sonar-pro-agent" via LiteLLM
    cache_maxsize=256,
    cache_ttl_seconds=3600,
    timeout_seconds=60,
)
```

| Parameter | Purpose |
|-----------|---------|
| `api_key` | Perplexity API key or the matching proxy key. |
| `base_url` | Perplexity's native endpoint or an OpenAI-compatible proxy. Default targets the native endpoint. |
| `model` | Sonar model id. `sonar`, `sonar-pro`, `sonar-reasoning` and their proxy aliases are all accepted. |
| `cache_maxsize` / `cache_ttl_seconds` | TTL-based LRU cache parameters. Key = SHA-256 of `query + params`. |
| `timeout_seconds` | Per-call wall-clock timeout; capped by the run deadline. |

## `search_model` property

Returns the configured Sonar model id verbatim (for example `"sonar-pro"`).

## Request shape sent to the API

```python
client.chat.completions.create(
    model=model,
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

Depending on the node state, the adapter can additionally set any of:

- `search_recency_filter`
- `search_language_filter`
- `search_domain_filter`
- `return_related_questions`
- `search_mode="academic"` for academic queries

These come from the hints Inqtrix passes into `search(...)`; see [Nodes](../architecture/nodes.md).

## Response shape expected from the API

The adapter reads:

- **Answer text** — from `choices[0].message.content` (non-stream) or from SSE `delta.content` chunks.
- **Citations** — top-level `citations: [...]` when present. If missing, the adapter falls back to URL extraction from the answer text (less reliable — explicit `citations` are preferred).
- **Related questions** — top-level `related_questions: [...]` when present.
- **Tokens** — `usage.prompt_tokens` / `usage.completion_tokens`.

Minimal response example:

```json
{
  "id": "chatcmpl-search-123",
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}}],
  "citations": ["https://example.com/report", "https://example.org/briefing"],
  "related_questions": ["Welche Reformschritte sind noch offen?"],
  "usage": {"prompt_tokens": 300, "completion_tokens": 120}
}
```

## YAML-mode caveat: `extra_body` is shallow-merged

In YAML mode, `models.<name>.params` values are forwarded as extra request kwargs. The merge is **shallow** — if you override `extra_body.web_search_options`, you must provide the full nested object, not only one nested key. This mirrors the OpenAI SDK behaviour.

See [inqtrix.yaml](../configuration/inqtrix-yaml.md) for the YAML wiring.

## Errors and retries

- OpenAI SDK retry loop drives transient retries (5xx, connection failures).
- HTTP 429 / quota: raise `AgentRateLimited`.
- Other errors: raise `PerplexityAPIError`.
- Per-query failure is non-fatal at the node level: the search node drops the failing query and continues with the rest.

## Related docs

- [Providers overview](overview.md)
- [Writing a custom provider](writing-a-custom-provider.md)
- [Nodes](../architecture/nodes.md)
- [inqtrix.yaml](../configuration/inqtrix-yaml.md)
