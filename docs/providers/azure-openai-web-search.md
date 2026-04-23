# Azure OpenAI Web Search (native tool)

> File: `src/inqtrix/providers/azure_openai_web_search.py`

## Scope

`AzureOpenAIWebSearch` uses the native Azure OpenAI Responses-API `web_search` tool. It does **not** require a Foundry agent; it talks directly to the same Azure OpenAI resource as [Azure OpenAI LLM](azure-openai.md).

## When to use it

- You already use Azure OpenAI for reasoning and want search from the same resource without pulling in Foundry.
- You need web search but your security team does not approve the Foundry agent path.
- You want first-class citation metadata through Responses-API tool calls.

## Constructor

```python
from inqtrix import AzureOpenAIWebSearch


search = AzureOpenAIWebSearch(
    azure_endpoint="https://my-resource.openai.azure.com/",
    api_key="...",
    default_model="my-gpt4o-search-deployment",
    user_country="DE",                                    # optional location hint
    user_location={"type": "approximate", "country": "DE"},  # alternative
)
```

The constructor accepts the same four authentication modes as [Azure OpenAI LLM](azure-openai.md): API key, Service Principal (`tenant_id` / `client_id` / `client_secret`), pre-built `credential`, or pre-built `azure_ad_token_provider`. The modes are mutually exclusive.

| Parameter | Purpose |
|-----------|---------|
| `default_model` | Azure **deployment name** that exposes the `web_search` tool. |
| `user_country` | Two-letter country code used to shape `user_location` automatically. |
| `user_location` | Full location dict forwarded to the tool. Mutually exclusive with `user_country`. |
| `tool_choice` | `"auto"` by default; set `"required"` to force the tool invocation. |

## `search_model` property

Returns `f"{default_model}+web_search_tool"` so `/health` and `/v1/stacks` show a string that combines the underlying deployment with the tool identity.

## Response normalisation

The adapter collects the Responses-API tool output:

- Answer text is assembled from the message content (including inline citation markers).
- `citations` is populated from every tool call that returned URL metadata.
- `related_questions` comes from the Responses `suggested_actions` block when present.
- Token counts come from `response.usage.input_tokens` and `.output_tokens`.

If the tool call returns no URLs the answer is returned as-is with an empty citations list; the run continues and the downstream source-quality cap will reflect the missing citations.

## Errors and degradation

- `AzureOpenAIWebSearchAPIError` — wraps Responses-API errors with status code and body.
- Non-API exceptions (network, parse) are caught by a final `except Exception` block and degraded to `_EMPTY_RESULT`. The node keeps running with an empty search result; the iteration log records the fallback.

## Related docs

- [Providers overview](overview.md)
- [Azure OpenAI LLM](azure-openai.md)
- [Azure Foundry Web Search](azure-foundry-web-search.md)
- [Enterprise Azure](../deployment/enterprise-azure.md)
