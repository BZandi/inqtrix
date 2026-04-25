# Timeouts and errors

> Files: `providers/base.py`, `providers/*`, `exceptions.py`, `constants.py`

## Scope

The deadline model, per-layer timeouts, the exception hierarchy, and the graceful-degradation rules per node. This is the reference for "what happens when a provider times out or fails".

## Deadline model

```
Global:  deadline = time.monotonic() + MAX_TOTAL_SECONDS (default 300s)
Per-call: effective_timeout = min(default_timeout, remaining_until_deadline)
```

Every node entry calls `_check_deadline(state["deadline"])`; if the deadline is already past, the node raises `AgentTimeout`. Every provider call receives the effective per-call timeout so a late-round call cannot silently burn the remaining budget.

| Layer | Env variable | Default | Purpose |
|-------|--------------|---------|---------|
| Agent total | `MAX_TOTAL_SECONDS` | 300s | Hard wall-clock budget for the whole run |
| LLM reasoning | `REASONING_TIMEOUT` | 120s | Per-call budget for classify/plan/evaluate/answer |
| Perplexity / other search | `SEARCH_TIMEOUT` | 60s | Per-call budget for a single `search()` |
| Summarise / claim extraction | `SUMMARIZE_TIMEOUT` | 60s | Per-call budget for summarize_parallel and claim extraction |

All four are configurable via environment variables (see [Settings and env](../configuration/settings-and-env.md)) or via `AgentConfig`.

## Exception hierarchy

| Exception | Parent | Trigger | Behaviour |
|-----------|--------|---------|-----------|
| `AgentTimeout` | `RuntimeError` | `time.monotonic() > deadline` at a node boundary or inside a provider | Graceful: `answer` is called with accumulated context even when downstream nodes have not completed. |
| `AgentRateLimited` | `RuntimeError` | HTTP 429 or daily token limit | Immediate abort. Partial token counts are preserved on the result. |
| `AgentCancelled` | `RuntimeError` | `_cancel_event` set (via disconnect watcher) | Abort at next node boundary; session save is skipped. |
| `AnthropicAPIError`, `AzureOpenAIAPIError`, `AzureFoundryBingAPIError`, `AzureFoundryWebSearchAPIError`, `BedrockAPIError`, `BraveSearchAPIError`, `PerplexityAPIError` | `RuntimeError` | Per-provider HTTP error (400s, 500s, schema mismatches) | Graceful degradation per node (see below). |

Each provider raises its own dedicated error type. All of them are exported from the top-level package so library consumers can catch them by type.

## Graceful-degradation rules per node

- **classify fails** — heuristic type inference; single sub-question = question verbatim.
- **plan fails** — fallback to `[question]` as single query.
- **search fails for a query** — skip that query, continue with others.
- **claim extraction fails for a source** — keep the search result and summary, continue without structured claims for that source (iteration-log marker `_claim_extraction_fallback`).
- **evaluate fails** — keep previous confidence, conservative gaps; iteration-log marker `_evaluate_fallback`.
- **answer fails** — return raw context without synthesis; the fallback answer is a German notice.

The invariant across all of them: a partial result is always returned to the caller. The agent never terminates without producing either an answer string or a typed exception.

## Provider retry behaviour

Two provider clients implement their own retry loops on top of the deadline model:

- `AnthropicLLM` — up to 5 attempts with exponential backoff and jitter on 5xx and 529 (overloaded). Rate-limit responses raise `AgentRateLimited` directly.
- `BedrockLLM` — up to 5 attempts on `ThrottlingException`; after the last attempt the error is translated to `AgentRateLimited`.

The OpenAI SDK drives retries for `LiteLLM`, `AzureOpenAILLM`, `AzureOpenAIWebSearch`, `AzureFoundryWebSearch`, and `AzureFoundryBingSearch`. Replay tests disable the retry loop to keep cassettes compact; production callers leave the default intact.

## Deadline interaction with retries

Each retry consults the deadline. If the remaining time is smaller than the next backoff interval, the provider aborts immediately with the underlying error instead of sleeping past the deadline. This keeps `MAX_TOTAL_SECONDS` as a real wall-clock bound.

## Related docs

- [Logging](logging.md)
- [Debugging runs](debugging-runs.md)
- [Settings and env](../configuration/settings-and-env.md)
- [Providers overview](../providers/overview.md)
