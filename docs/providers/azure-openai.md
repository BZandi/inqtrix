# Azure OpenAI (LLM)

> File: `src/inqtrix/providers/azure.py`

## Scope

`AzureOpenAILLM` is the adapter for the Azure OpenAI Chat Completions surface (`.openai.azure.com/openai/v1/...`). This page covers the deployment-relevant choices.

## When to use it

- Your organisation runs GPT-class models through an Azure OpenAI resource.
- You need enterprise auth (Service Principal, Managed Identity, custom token provider) rather than a static API key.
- You must stay within an Azure region for data-residency reasons.

For Azure **search** backends see [Azure Foundry Web Search](azure-foundry-web-search.md).

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
    claim_extract_model: str | None = None,
    evaluate_model: str | None = None,
    request_max_tokens: int | None = None,
    token_budget_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens",
    request_params: Mapping[str, Any] | None = None,
    default_reasoning_effort: str | None = None,
    api_version: str = "v1",
    timeout_seconds: float = 120.0,
)
```

### Key parameters

- `azure_endpoint` — resource URL in the form `https://<resource>.openai.azure.com/`. Both the bare resource URL and the explicit `/openai/v1/` URL are accepted. An Azure AI Project endpoint (`.../api`) is rejected (different API surface).
- `default_model` — **deployment name**, not the underlying model name. Must match a deployment in the target resource. The placeholder default `"gpt-4o"` almost always needs to be overridden.
- `token_budget_parameter` — newer deployments expect `"max_completion_tokens"` (default). Switch to `"max_tokens"` only when a specific legacy deployment still requires it — using the wrong one yields a `400 unsupported parameter` from the API.
- `default_reasoning_effort` — see [Reasoning effort](#reasoning-effort) below.

## Reasoning effort

Azure GPT-5 series and o-series deployments accept a `reasoning_effort`
parameter that trades latency against reasoning depth. Inqtrix exposes this
for the default reasoning path; claim extraction remains a regular completion
call and should use a cheaper deployment through `claim_extract_model` when
cost matters:

```python
AzureOpenAILLM(
    azure_endpoint="https://my-resource.openai.azure.com/",
    api_key="...",
    default_model="gpt-5",
    claim_extract_model="gpt-5-nano",
    default_reasoning_effort="low",        # classify, plan, evaluate, answer
)
```

### Allowed values

`"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`. The
provider validates the value set; per-value-per-model compatibility is
handled by Azure (it returns HTTP 400 with a clear message when a
deployment rejects a value).

### Per-model caveats

- `"none"` disables reasoning entirely. Only on `gpt-5.1+` deployments — other models return 400.
- `"minimal"` is "few reasoning tokens", **not** none. Only on the **original** GPT-5 (not `gpt-5.1+`, not `gpt-5-codex`).
- `"xhigh"` only on `gpt-5.1-codex-max`.
- `gpt-5-pro` accepts only `"high"`.
- `gpt-5.1+` defaults to `"none"`. After upgrading from `gpt-5` to `gpt-5.1`, set `default_reasoning_effort="medium"` (or higher) explicitly if you want reasoning to occur.
- `o-series` (`o1`, `o3`, `o4-mini`) accepts only `low`/`medium`/`high`.
- `o1-mini` does not support `reasoning_effort` at all.

The provider warns at construction time when the deployment name contains a known non-reasoning fragment (`gpt-4o`, `gpt-3.5`, `embedding`, `whisper`, `tts`, `dall-e`, `o1-mini`, ...). Unfamiliar names pass through silently — Azure is the authoritative source.

### Mutex with `temperature`

Reasoning models reject `temperature`. Setting both `temperature` and any `*_reasoning_effort` raises `ValueError` in the constructor — analog to Anthropic's `temperature`/`thinking` mutex.

## Errors

- `AzureOpenAIAPIError(status_code=404)` — deployment name does not exist in the resource.
- `AzureOpenAIAPIError(status_code=400)` — `token_budget_parameter` mismatch, prompt/model constraint violation, deployment does not support the requested `reasoning_effort` value (e.g. `"xhigh"` on a non-codex-max deployment), or `temperature` set on a reasoning deployment.
- `ValueError` (constructor) — invalid `default_reasoning_effort` value, or `temperature` together with `default_reasoning_effort`.
- `AgentRateLimited` — HTTP 429 or token-per-minute quota exceeded.

SDK retries are disabled for LLM calls. The provider retries transient 408/409/5xx and SDK timeout/connection errors itself so attempts are visible in logs and live progress.

## Related docs

- [Providers overview](overview.md)
- [Enterprise Azure](../deployment/enterprise-azure.md)
- [Azure Foundry Web Search](azure-foundry-web-search.md)
