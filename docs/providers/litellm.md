# LiteLLM

> File: `src/inqtrix/providers/litellm.py`

## Scope

`LiteLLM` is the default LLM provider. It talks to any OpenAI-compatible chat-completions endpoint via the official `openai` Python SDK, so it works with a hosted LiteLLM proxy, OpenRouter, vLLM, LocalAI, or any other OpenAI-shaped gateway.

## When to use it

- You run a LiteLLM proxy that handles routing, retries, and quota for you.
- You want a single provider that can reach both reasoning models and search-capable models through one endpoint.
- You are prototyping and want the shortest path to a working setup.

If your backend is **not** OpenAI-shaped, implement `LLMProvider` directly instead. See [Writing a custom provider](writing-a-custom-provider.md).

## Constructor

```python
from inqtrix import LiteLLM


llm = LiteLLM(
    api_key="sk-...",
    base_url="http://localhost:4000/v1",
    default_model="gpt-4o",
    classify_model="gpt-4o-mini",
    summarize_model="gpt-4o-mini",
    evaluate_model="gpt-4o-mini",
)
```

| Parameter | Purpose |
|-----------|---------|
| `api_key` | Proxy API key; passed straight to the OpenAI SDK. |
| `base_url` | Proxy URL including `/v1`. Default targets `http://litellm-proxy:4000/v1`. |
| `default_model` | Fallback model used for `reasoning`, `plan`, `answer`. |
| `classify_model`, `summarize_model`, `evaluate_model` | Optional per-role overrides; each falls back to `default_model` when omitted. |
| `request_max_tokens` | Hard cap forwarded as `max_completion_tokens` (or the legacy `max_tokens`) to let the adapter record an honest token-utilisation ratio. |

## Example stack

Library mode:

```python
from inqtrix import AgentConfig, LiteLLM, PerplexitySearch, ResearchAgent

agent = ResearchAgent(AgentConfig(
    llm=LiteLLM(api_key="sk-...", base_url="http://localhost:4000/v1",
                default_model="gpt-4o"),
    search=PerplexitySearch(api_key="sk-...", base_url="http://localhost:4000/v1",
                            model="perplexity-sonar-pro-agent"),
    max_rounds=4,
    confidence_stop=8,
))
```

Server mode: `examples/webserver_stacks/litellm_perplexity.py` is the byte-for-byte server-side counterpart; see [Web server mode](../deployment/webserver-mode.md).

## Response shape expected from the endpoint

The adapter reads the text from any of the standard OpenAI-compatible shapes:

- Non-stream: `choices[0].message.content`.
- SSE chunk stream encoded as text: `choices[*].delta.content`.
- Optional token metadata: `usage.prompt_tokens` and `usage.completion_tokens`.

Both plain JSON responses and SSE payloads encoded as a single string are accepted, so proxies that collapse an SSE stream into one blob still work.

## Limitations

- Retries and rate-limit handling come from the OpenAI SDK; the adapter does not add its own retry loop.
- Thinking or reasoning-budget parameters are not part of this adapter. Use `AnthropicLLM` or a future provider if you need them.
- Token-usage metadata is only as good as the upstream proxy: a proxy that strips `usage` from the response will report zeros.

## Related docs

- [Providers overview](overview.md)
- [Perplexity](perplexity.md)
- [Writing a custom provider](writing-a-custom-provider.md)
- [Web server mode](../deployment/webserver-mode.md)
