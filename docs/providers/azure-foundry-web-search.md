# Azure Foundry Web Search

> File: `src/inqtrix/providers/azure_web_search.py`

## Scope

`AzureFoundryWebSearch` invokes a pre-created Azure Foundry agent that uses the Foundry Web Search tool. It runs over the Responses API, which makes it VCR-recordable; unlike [Azure Foundry Bing](azure-foundry-bing.md), there is no legacy AIProjectClient path.

## When to use it

- Your organisation mandates Foundry-managed web search instead of direct Bing Grounding.
- You want all agent calls to flow through Foundry for audit and cost attribution.
- You already have a Foundry project with the Web Search tool configured.

## Constructor

```python
from inqtrix import AzureFoundryWebSearch


search = AzureFoundryWebSearch(
    project_endpoint="https://my-project.services.ai.azure.com/api/projects/my-project",
    tenant_id="...",
    client_id="...",
    client_secret="...",
    web_search_agent_name="web-search-agent",
    # web_search_agent_version="...",        # optional version pin
)
```

| Parameter | Purpose |
|-----------|---------|
| `project_endpoint` | Foundry project endpoint (`.services.ai.azure.com/api/projects/...`). |
| `web_search_agent_name` | Pre-created agent name. The adapter resolves it to an agent reference on first use. |
| `web_search_agent_version` | Optional version pin; omitting it uses the latest published version. |
| `tenant_id` / `client_id` / `client_secret` / `credential` / `azure_ad_token_provider` | One of the four auth modes (see [Azure OpenAI](azure-openai.md)). |

## `search_model` property

Returns `f"foundry-web:{name}@{version_or_latest}"` so operators see the agent identity and version in `/health` and `/v1/stacks`.

## Token lifetime

Identical to [Azure Foundry Bing](azure-foundry-bing.md): cached bearer tokens are valid for ~60–75 minutes. Long-running servers occasionally see transient 401s when a near-expired token is handed out for a long request. See [Enterprise Azure](../deployment/enterprise-azure.md) for the recommended container-restart cadence.

## Response shape

The adapter collects Responses-API tool output and normalises it to the standard search-return dict:

- `answer` — assembled from the agent's final message content.
- `citations` — URLs reported by the Web Search tool.
- `related_questions` — populated when the agent returns them.
- `_prompt_tokens` / `_completion_tokens` — from the Responses `usage` block.

## Errors and degradation

- `AzureFoundryWebSearchAPIError` — wraps Foundry Responses errors with status code and body.
- Any other exception is caught by a final `except Exception` block, recorded as a non-fatal notice, and the adapter returns `_EMPTY_RESULT`. The node keeps running with no citations; the iteration log records the fallback.

## Related docs

- [Providers overview](overview.md)
- [Azure OpenAI LLM](azure-openai.md)
- [Azure Foundry Bing](azure-foundry-bing.md)
- [Enterprise Azure](../deployment/enterprise-azure.md)
