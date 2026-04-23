# Anthropic

> File: `src/inqtrix/providers/anthropic.py`

## Scope

`AnthropicLLM` is a direct adapter for the Anthropic Messages API. It does **not** route through LiteLLM, so it is the right choice when you want to use Anthropic-specific features (extended thinking, request-level retry tuning) or avoid an extra proxy hop.

## When to use it

- You already have an Anthropic API key and want the shortest path to Claude.
- You want to enable extended thinking (`thinking={"type": "adaptive"}`) on reasoning-heavy roles.
- You want the adapter's own retry loop (5 attempts, exponential backoff with jitter on 5xx / 529) rather than the OpenAI SDK retry semantics.

## Constructor

```python
from inqtrix import AnthropicLLM


llm = AnthropicLLM(
    api_key="sk-ant-...",
    default_model="claude-sonnet-4-6",
    classify_model="claude-haiku-4-5",   # optional, cheaper role
    summarize_model="claude-haiku-4-5",  # usually the biggest cost lever
    evaluate_model="claude-sonnet-4-6",  # optional per-role override
    # thinking={"type": "adaptive"},     # optional reasoning budget
)
```

| Parameter | Purpose |
|-----------|---------|
| `api_key` | Anthropic API key. Read from `.env` in the example scripts. |
| `default_model` | Used for classify fallback, plan, answer, evaluate fallback. |
| `classify_model` / `evaluate_model` / `summarize_model` | Optional per-role overrides. |
| `thinking` | Forwarded on reasoning calls (classify, evaluate, plan, answer). **Not** used for summarize. If the model does not support thinking, Anthropic rejects the request with a 400. |
| `request_max_tokens` | Hard cap on `max_tokens`; enables honest token-utilisation logging. |
| `timeout_seconds` | Per-call timeout default; capped by the run deadline. |

## Extended thinking

`thinking` is forwarded verbatim to the Messages API. Supported shapes include:

- `{"type": "adaptive"}` — the API picks a thinking budget.
- `{"type": "enabled", "budget_tokens": 5000}` — explicit budget.

### Haiku quirk

Claude Haiku 4.5 rejects the `effort` parameter with HTTP 400 and expects a different `output_config.effort` layout. Inqtrix suppresses `effort` for Haiku roles via `_EFFORT_UNSUPPORTED_FRAGMENTS` and surfaces the suppression via `consume_effort_config_warnings` — the warning is loud on purpose (Design Principle 1). If you call `AnthropicLLM` with a Haiku role and your code supplies `effort`, expect a warning marker per call; the run still succeeds.

## Retry behaviour

- HTTP 5xx or 529 (overloaded): up to 5 attempts with exponential backoff and jitter.
- HTTP 429 / daily token limit: raise `AgentRateLimited` immediately without retry.
- Other errors: raise `AnthropicAPIError` with status code and body.

Every retry consults the run deadline so retries do not push the run past `MAX_TOTAL_SECONDS`.

## Example stacks

Library:

```python
from inqtrix import AgentConfig, AnthropicLLM, PerplexitySearch, ResearchAgent

agent = ResearchAgent(AgentConfig(
    llm=AnthropicLLM(
        api_key="sk-ant-...",
        default_model="claude-sonnet-4-6",
        summarize_model="claude-haiku-4-5",
    ),
    search=PerplexitySearch(api_key="pplx-..."),
))
```

Server: `examples/webserver_stacks/anthropic_perplexity.py`.

## Related docs

- [Providers overview](overview.md)
- [LiteLLM](litellm.md)
- [Perplexity](perplexity.md)
- [Debugging runs](../observability/debugging-runs.md)
