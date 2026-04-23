# Azure Foundry Bing Grounding

> File: `src/inqtrix/providers/azure_bing.py`

## Scope

`AzureFoundryBingSearch` invokes a pre-created Azure Foundry agent that uses Bing Grounding as its tool. Two runtime paths are supported:

- **Modern path** — `openai` SDK → Foundry Responses API. VCR-recordable and used by the built-in tests.
- **Legacy path** — `azure-ai-projects` `AIProjectClient.agents.*`. Kept for Foundry projects whose agents were registered before the Responses-API surface was generally available. The azure-core transport is not VCR-friendly; the legacy tests use `unittest.mock.MagicMock`.

## When to use it

- Your organisation standardised on Bing Grounding inside Azure Foundry.
- You must route all web traffic through Foundry-managed agents for audit reasons.
- You run an existing Foundry agent that already has the Bing tool configured and you want to reuse it.

## Constructor

```python
from inqtrix import AzureFoundryBingSearch


search = AzureFoundryBingSearch(
    project_endpoint="https://my-project.services.ai.azure.com/api/projects/my-project",
    tenant_id="...",
    client_id="...",
    client_secret="...",
    bing_agent_name="bing-grounding-agent",   # preferred
    # bing_agent_id="agent-xxx",              # alternative: pre-existing agent id
    # bing_agent_version="...",               # optional agent version
    bing_project_connection_id="...",         # optional Foundry connection ref
)
```

| Parameter | Purpose |
|-----------|---------|
| `project_endpoint` | Foundry project endpoint (`.services.ai.azure.com/api/projects/...`). |
| `bing_agent_name` / `bing_agent_id` | Identify the pre-created agent. Name is resolved to id on first call. |
| `bing_agent_version` | Optional version pin; omitting it uses the latest. |
| `bing_project_connection_id` | Foundry connection to Bing. Must exist in the project. |
| `tenant_id` / `client_id` / `client_secret` / `credential` / `azure_ad_token_provider` | One of the four auth modes (see [Azure OpenAI](azure-openai.md)). |

## `search_model` property

Returns `f"foundry-bing:{name_or_id}@{version_or_latest}"` so operators see the agent identity and version in `/health` and `/v1/stacks`.

## Token lifetime

Foundry bearer tokens have a cached lifetime of approximately 60–75 minutes. Long-running servers will see occasional transient 401s when a nearly expired token is handed out for a long request (see Gotcha #17 in the internal notes); the simplest mitigation is to accept the transient failure and rely on the next request to refresh. See [Enterprise Azure](../deployment/enterprise-azure.md) for container-restart guidance.

## Response shape

The adapter produces the standard search-return dict. Citations come from the Bing tool's URL metadata; `related_questions` comes from the tool's "people also ask" array when present.

## Errors and degradation

- `AzureFoundryBingAPIError` — wraps Foundry API errors with status code and body.
- Any other exception (azure-core transport failure, parse error, agent-not-found) is caught by a final `except Exception` block, recorded as a non-fatal notice, and the adapter returns `_EMPTY_RESULT`. The node keeps running with zero citations; the iteration log records the fallback.

## Legacy-path caveat

The legacy `AIProjectClient.agents.thread` / `run` path is exercised by a single MagicMock-based test (`tests/replay/test_azure_foundry_bing_replay.py::test_legacy_thread_run_path_returns_answer_with_mock`). That pattern is intentional: azure-core's transport pipeline is not reliably VCR-interceptable, and a hand-rolled patch would be more brittle than the MagicMock.

## Related docs

- [Providers overview](overview.md)
- [Azure OpenAI LLM](azure-openai.md)
- [Azure Foundry Web Search](azure-foundry-web-search.md)
- [Enterprise Azure](../deployment/enterprise-azure.md)
