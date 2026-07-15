# Timeouts and errors

> Files: `providers/base.py`, `providers/*`, `exceptions.py`, `constants.py`

## Scope

The deadline model, per-layer timeouts, the exception hierarchy, and the graceful-degradation rules per node. This is the reference for "what happens when a provider times out or fails".

## Deadline model

```
Run:       deadline = time.monotonic() + MAX_TOTAL_SECONDS (default 3600s)
Operation: deadline = min(now + configured_timeout, run_deadline)
```

Every node entry calls `_check_deadline(state["deadline"])`; if the deadline is already past, the node raises `AgentTimeout`. Every provider call receives the effective per-call timeout so a late-round call cannot silently burn the remaining budget.

| Layer | Env variable | Default | Purpose |
|-------|--------------|---------|---------|
| Active research run | `MAX_TOTAL_SECONDS` | 3600s | Outer wall-clock budget checked at graph boundaries and passed into provider calls. Time parked for Agent Desk approval/input is not active research time. |
| LLM reasoning | `REASONING_TIMEOUT` | 600s | Logical-operation budget for classify/plan/evaluate/answer, including retries and backoff. |
| Editor assistant | `EDITOR_ASSISTANT_TIMEOUT` | 600s | Logical-operation budget for editor suggest/instruct calls. |
| Perplexity / other search | `SEARCH_TIMEOUT` | 600s | Logical-operation budget for one `search()`, including retries and backoff. |
| Claim extraction | `CLAIM_EXTRACT_TIMEOUT` | 600s | Logical-operation budget for one extraction, including retries and backoff. |

All four are configurable via environment variables (see [Settings and env](../configuration/settings-and-env.md)) or via `AgentConfig`.

## Exception hierarchy

| Exception | Parent | Trigger | Behaviour |
|-----------|--------|---------|-----------|
| `AgentTimeout` | `RuntimeError` | `time.monotonic() > deadline` at a node boundary or inside a provider | Graceful: `answer` is called with accumulated context even when downstream nodes have not completed. |
| `AgentProviderTimeout` | `AgentTimeout` | One logical provider operation exhausts its own timeout before the outer run deadline | Visible provider-operation failure; retry metadata records the operation, configured/effective budget, and attempt. |
| `AgentRateLimited` | `RuntimeError` | HTTP 429 remains after the shared three-attempt budget, or the provider reports a hard daily/token cap | Visible terminal provider failure. Partial token counts are preserved on the result. |
| `AgentCancelled` | `RuntimeError` | `_cancel_event` set (explicit cancel or disconnect watcher) | Abort at the next cancellation checkpoint (node entry, provider retry attempt, backoff sleep, fan-out coordination, answer section) and return a cancelled result state. Residual latency: the remainder of one in-flight provider HTTP attempt. |
| `AnthropicAPIError`, `AzureOpenAIAPIError`, `AzureFoundryWebSearchAPIError`, `BedrockAPIError`, `PerplexityAPIError` | `RuntimeError` | Per-provider HTTP error (400s, 500s, schema mismatches) | Graceful degradation per node (see below). |

Each provider raises its own dedicated error type. All of them are exported from the top-level package so library consumers can catch them by type.

## Graceful-degradation rules per node

- **classify fails** — heuristic type inference; single sub-question = question verbatim.
- **plan fails** — fallback to `[question]` as single query.
- **search fails for a query** — skip that query, continue with others.
- **claim extraction fails for a source** — keep the search result and summary, but mark the source-level extraction as `ALGO-FAIL claim_extraction` via progress, warning, and iteration-log fields.
- **claim extraction fails for the whole run** — in forensic/deep runs, block normal report synthesis and return a short diagnostic report instead of a normal hard-fact report.
- **evaluate fails** — keep previous confidence, conservative gaps; iteration-log marker `_evaluate_fallback`.
- **answer fails** — return raw context without synthesis; the fallback answer is a German notice.

The invariant across all of them: a partial result is always returned to the caller. The agent never terminates without producing either an answer string or a typed exception.

## Provider retry behaviour

The built-in providers implement one visible retry authority on top of the
deadline model. A logical operation has at most three attempts in total; a
mixed sequence of transport, 5xx, and 429 failures cannot reset that counter:

- `AnthropicLLM` — up to 3 total attempts with exponential backoff and jitter on transient transport, 5xx/529, and 429 responses.
- `BedrockLLM` — up to 3 total attempts on transient Converse/transport errors; throttling after the final attempt becomes `AgentRateLimited`.
- `AzureOpenAILLM` and `LiteLLM` — OpenAI SDK retries are disabled and Inqtrix retries transient 408/409/5xx plus SDK timeout/connection errors itself.
- `AzureFoundryWebSearch` — OpenAI SDK retries are disabled and Inqtrix retries transient Responses API 408/409/5xx plus SDK timeout/connection errors itself.

Every retry emits a warning log and a live progress/activity event. Research
search retries include the parallel query position (for example
`Websuche 2/6`); Agent Desk instant-search retries carry the task/query,
attempt `n/3`, delay, error code, and configured/effective timeout in the
existing `inqtrix.agent.activity` stream. Mission phase calls and Kernel model
calls use that same stream with a human-readable phase purpose; no separate
model-progress channel is created.

## Deadline interaction with retries

All attempts and backoff sleeps share the operation deadline computed before
attempt one. A retry never receives another 600 seconds. The outer run
deadline may shorten this budget further; whichever deadline expires first
determines whether `AgentProviderTimeout` or `AgentTimeout` is surfaced.

## Related docs

- [Logging](logging.md)
- [Debugging runs](debugging-runs.md)
- [Settings and env](../configuration/settings-and-env.md)
- [Providers overview](../providers/overview.md)
