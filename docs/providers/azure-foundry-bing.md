# Azure Foundry Bing Grounding

> File: `src/inqtrix/providers/azure_bing.py`

## Scope

`AzureFoundryBingSearch` invokes a pre-created Azure Foundry agent that uses Bing Grounding as its tool. Two runtime paths are supported:

- **Modern path** — `openai` SDK → Foundry Responses API (`responses.create` with `extra_body.agent_reference`). VCR-recordable and used by the built-in tests.
- **Legacy path** — `azure-ai-projects` `AIProjectClient.agents.create_thread_and_process_run`. Used when the agent is not (or not yet) addressable via `agent_reference`, or when `execution_mode="legacy"` is set.

With `execution_mode="auto"` (default), a Responses **HTTP 404** for the agent reference triggers: (1) a second Responses attempt **without** `version` when that version was only auto-filled from `get_agent()` (not passed explicitly by you); (2) if still failing, **legacy Thread/Run** when an `agent_id` is known and credential-based auth built an `AIProjectClient`.

## When to use it

- Your organisation standardised on Bing Grounding inside Azure Foundry.
- You must route all web traffic through Foundry-managed agents for audit reasons.
- You run an existing Foundry agent that already has the Bing tool configured and you want to reuse it.

## Constructor

Parameter names match the Python API (there is **no** `bing_*` prefix on the constructor).

```python
from inqtrix import AzureFoundryBingSearch


search = AzureFoundryBingSearch(
    project_endpoint="https://my-project.services.ai.azure.com/api/projects/my-project",
    agent_name="bing-grounding-agent",
    # agent_version="2",          # optional explicit version pin
    # agent_id="asst_...",        # optional legacy id (often used with fallbacks)
    # api_key="...",              # optional Foundry project API key
    tenant_id="...",
    client_id="...",
    client_secret="...",
    # execution_mode="auto",      # auto | responses | legacy (default: auto)
)
```

| Parameter | Purpose |
|-----------|---------|
| `project_endpoint` | Foundry project endpoint (`…services.ai.azure.com/api/projects/…`). |
| `agent_name` | Preferred identifier for the Responses `agent_reference.name` field. |
| `agent_id` | Legacy opaque id. With Entra auth, `get_agent(id)` may resolve `agent_name` / `agent_version` for the modern path; `auto` mode can fall back to Thread/Run using this id. |
| `agent_version` | Optional `agent_reference.version`. Omit to let Foundry pick the default/latest. |
| `api_key` | Optional static Foundry project API key (OpenAI client `api_key`). Mutually exclusive with the Service-Principal trio below for the credential branch; **legacy Thread/Run is not available** in API-key-only mode. |
| `credential` | Optional prebuilt Azure credential (advanced). |
| `tenant_id` / `client_id` / `client_secret` | Service Principal → `ClientSecretCredential` when all three are set. |
| `timeout` | Per-call HTTP timeout (seconds). |
| `tool_choice` | Passed through to `responses.create` (`"required"` default). |
| `execution_mode` | `"auto"` (default): Responses first, 404 fallbacks as above. `"responses"`: modern path only (no 404 fallbacks). `"legacy"`: Thread/Run only; **requires** `agent_id` and credential-based auth (no `api_key`-only). |

## Creating an agent once (`create_agent`)

Programmatic creation uses **`bing_connection_id`** (full Foundry connection resource id for Bing), not the runtime constructor:

```python
search = AzureFoundryBingSearch.create_agent(
    project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    bing_connection_id=os.environ["BING_PROJECT_CONNECTION_ID"],
    model="gpt-4o",  # must match a model deployment in the same Foundry project
    agent_name="inqtrix-bing-search",
    tenant_id=os.environ["AZURE_TENANT_ID"],
    client_id=os.environ["AZURE_CLIENT_ID"],
    client_secret=os.environ["AZURE_CLIENT_SECRET"],
)
```

`create_agent()` **does not** accept `api_key`; it requires Azure credentials. On recent SDKs it prefers `agents.create_version` (prompt agent) so the returned instance is typically Responses-ready.

Smoke test helper: set `BING_SMOKE_CREATE_AGENT=true` in `examples/provider_stacks/azure_smoke_tests/test_bing_search.py` (see script docstring and `.env.example`).

## `search_model` property

Returns `f"foundry-bing:{name_or_id}@{version_or_latest}"` so operators see the agent identity and version in `/health` and `/v1/stacks`.

## Token lifetime

Foundry bearer tokens have a cached lifetime of approximately 60–75 minutes. Long-running servers will see occasional transient 401s when a nearly expired token is handed out for a long request (see Gotcha #17 in the internal notes); the simplest mitigation is to accept the transient failure and rely on the next request to refresh. See [Enterprise Azure](../deployment/enterprise-azure.md) for container-restart guidance.

## Response shape

The adapter produces the standard search-return dict. Citations come from the Bing tool's URL metadata when the runtime surfaces URL annotations; otherwise the provider may fall back to URL extraction from answer text.

## Troubleshooting (common HTTP 404)

| Symptom | Likely cause | What to try |
|--------|----------------|------------|
| `Agent … with version not found` after using **only** `agent_id` | `get_agent` returned a **version string** that the Responses registry does not recognise. | Use `execution_mode="auto"` (default): provider retries without auto-resolved version, then Thread/Run if `agent_id` is set. Or pass **`agent_name`** and omit **`agent_version`**. |
| Same 404 when using **`agent_name` only** | Agent exists in UI but is a **classic** agent, wrong **project endpoint**, or UI label ≠ API `name`. | Confirm `AZURE_AI_PROJECT_ENDPOINT` matches the project that lists the agent. Use `AIProjectClient.agents.get_agent` / list APIs to read the exact `name`. Create a versioned agent via `create_agent()` or Foundry UI. |
| Need Thread/Run only | Stable compatibility with legacy agents. | `execution_mode="legacy"` + `agent_id` + Service Principal (not API key). |

## Errors and degradation

- `AzureFoundryBingAPIError` — wraps Foundry API errors with status code and body when raised from the Responses path.
- Other exceptions from `search()` are caught by a final `except Exception` block, recorded as a non-fatal notice, and the adapter returns `_EMPTY_RESULT`. The node keeps running with zero citations; the iteration log records the fallback.

## Legacy-path caveat

The legacy `AIProjectClient.agents` Thread/Run path is exercised by MagicMock-based tests (`tests/test_providers_azure_bing.py`, `tests/replay/test_azure_foundry_bing_replay.py`). That pattern is intentional: azure-core's transport pipeline is not reliably VCR-interceptable.

## Related docs

- [Providers overview](overview.md)
- [Azure OpenAI LLM](azure-openai.md)
- [Azure Foundry Web Search](azure-foundry-web-search.md)
- [Enterprise Azure](../deployment/enterprise-azure.md)
