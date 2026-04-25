# Azure OpenAI (LLM)

> File: `src/inqtrix/providers/azure.py`

## Scope

`AzureOpenAILLM` is the adapter for the Azure OpenAI Chat Completions surface (`.openai.azure.com/openai/v1/...`). This page summarises the deployment-relevant choices.

## When to use it

- Your organisation runs GPT-class models through an Azure OpenAI resource.
- You need enterprise auth (Service Principal, Managed Identity, custom token provider) rather than a static API key.
- You must stay within an Azure region for data-residency reasons.

For Azure **search** backends see [Azure OpenAI Web Search](azure-openai-web-search.md), [Azure Foundry Web Search](azure-foundry-web-search.md), and [Azure Foundry Bing](azure-foundry-bing.md).

## Four authentication modes

All four are mutually exclusive. Supplying more than one raises `ValueError`.

### 1. API key

```python
AzureOpenAILLM(
    azure_endpoint="https://my-resource.openai.azure.com/",
    api_key="...",
    default_model="my-gpt4o-deployment",
)
```

Simplest path. Good for single-developer setups and quick experiments.

### 2. Service Principal

```python
AzureOpenAILLM(
    azure_endpoint="https://my-resource.openai.azure.com/",
    tenant_id="...",
    client_id="...",
    client_secret="...",
    default_model="my-gpt4o-deployment",
)
```

Canonical for CI/CD and servers that cannot use Managed Identity. Internally builds a `ClientSecretCredential` and wraps it in a token provider that caches with ~60–75 minute token lifetime.

### 3. Pre-built credential (Managed Identity, DefaultAzureCredential, ...)

```python
from azure.identity import DefaultAzureCredential

AzureOpenAILLM(
    azure_endpoint="https://my-resource.openai.azure.com/",
    credential=DefaultAzureCredential(),
    default_model="my-gpt4o-deployment",
)
```

Production-recommended when Inqtrix runs inside Azure (AKS, App Service, VMs with Managed Identity). Same token-lifetime caveat applies; see [Enterprise Azure](../deployment/enterprise-azure.md) for the long-running-server strategy.

### 4. Pre-built token provider

```python
AzureOpenAILLM(
    azure_endpoint="https://my-resource.openai.azure.com/",
    azure_ad_token_provider=my_custom_token_provider,
    default_model="my-gpt4o-deployment",
)
```

Use when you already issue bearer tokens through a custom code path. The callable must return a non-empty string; Inqtrix calls it on every request, so implement your own cache.

## Constructor (full)

```python
AzureOpenAILLM(
    *,
    azure_endpoint: str,
    api_key: str | None = None,
    tenant_id: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    credential: TokenCredential | None = None,
    azure_ad_token_provider: Callable[[], str] | None = None,
    default_model: str = "gpt-4o",
    classify_model: str | None = None,
    summarize_model: str | None = None,
    evaluate_model: str | None = None,
    request_max_tokens: int | None = None,
    token_budget_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens",
    api_version: str = "v1",
    timeout_seconds: float = 120.0,
)
```

### Key parameters

- `azure_endpoint` — resource URL in the form `https://<resource>.openai.azure.com/`. Both the bare resource URL and the explicit `/openai/v1/` URL are accepted. An Azure AI Project endpoint (`.../api`) is rejected (different API surface).
- `default_model` — **deployment name**, not the underlying model name. Must match a deployment in the target resource. The placeholder default `"gpt-4o"` almost always needs to be overridden.
- `token_budget_parameter` — newer deployments expect `"max_completion_tokens"` (default). Switch to `"max_tokens"` only when a specific legacy deployment still requires it — using the wrong one yields a `400 unsupported parameter` from the API.

## Errors

- `AzureOpenAIAPIError(status_code=404)` — deployment name does not exist in the resource.
- `AzureOpenAIAPIError(status_code=400)` — `token_budget_parameter` mismatch, or prompt/model constraint violation.
- `AgentRateLimited` — HTTP 429 or token-per-minute quota exceeded.

## Related docs

- [Providers overview](overview.md)
- [Enterprise Azure](../deployment/enterprise-azure.md)
- [Azure OpenAI Web Search](azure-openai-web-search.md)
- [Azure Foundry Bing](azure-foundry-bing.md)
- [Azure Foundry Web Search](azure-foundry-web-search.md)
