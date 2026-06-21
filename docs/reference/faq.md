# FAQ

## Why do I see Anthropic 404 errors with a model name like `claude-opus-4.6-agent`?

That model name is the default on `ModelSettings()`; it is a LiteLLM alias, not a real Anthropic model id. Older code paths that read `settings.models.effective_claim_extract_model` could leak that default into the claim-extraction strategy and the Anthropic backend rejected it. Current code reads constructor-first via `resolve_claim_extract_model(llm, fallback=...)`. If you wrote a custom strategy, make sure you use the same helper.

See [Debugging runs](../observability/debugging-runs.md) for the log-marker walkthrough.

## I pressed Cancel in my UI but the run kept going for another minute

Cancel is enforced at node boundaries, not mid-provider-call. A running Anthropic Opus call with a 60-second reasoning budget will complete before the cancel takes effect. The SSE stream closes immediately; the agent continues in the background until the next boundary. Reduce `REASONING_TIMEOUT` to shorten the worst case.

See [Web server mode](../deployment/webserver-mode.md) and [Progress events](../observability/progress-events.md).

## Which Azure authentication mode should I pick?

- **API key** — simplest, good for quick experiments and single-developer setups.
- **Service Principal** (`tenant_id` + `client_id` + `client_secret`) — canonical for CI/CD and for servers that cannot use Managed Identity.
- **Managed Identity** (pass a `DefaultAzureCredential` as `credential=...`) — production-recommended when Inqtrix runs in Azure (AKS, App Service, VMs with MI).
- **Pre-built token provider** — if your platform already issues bearer tokens via a custom code path.

All four are constructor arguments; the provider never reads Azure env vars directly. See [Azure OpenAI provider](../providers/azure-openai.md) and [Enterprise Azure](../deployment/enterprise-azure.md).

## How much does a single run cost?

Depends on your provider mix, model selection, and question. The tools to measure it:

- `ResearchResult.metrics.prompt_tokens` and `.completion_tokens` capture aggregate token usage per run (where the provider returns usage metadata).
- The iteration log records per-call usage in testing mode.
- The parity CLI computes per-question deltas in its analysis report.

See [Result schema](../architecture/result-schema.md) and [Parity tooling](../development/parity-tooling.md).

## Why are the prompt templates in German?

The default user base is German-speaking. Prompt strings in `src/inqtrix/prompts.py` are the single exception to the English-only convention (the other exceptions are UI-facing HTTP error strings and demo questions in `examples/`). There is no public prompt-dictionary field on `AgentConfig` today; to switch to English, fork or edit the prompt templates, or wrap the relevant provider/strategy in your application.

## Can I build a React UI with live research cards?

Yes. The repository now includes the foundation for a React + Vite + shadcn app in `apps/research-desk`. Use the native `/v1/runs` API instead of `/v1/chat/completions`: `POST /v1/runs` creates a queued run, `GET /v1/runs/{run_id}/events` streams structured progress snapshots, and `GET /v1/runs/{run_id}/result` returns the final markdown report plus metrics/sources/claims. See [React UI](../deployment/react-ui.md) and [Run events](../observability/run-events.md).

Completed run records are intentionally short-lived in memory (`RUN_COMPLETED_TTL_SECONDS`, default 300). They do not survive server restart and are not a durable report archive. Persist the result payload in your application database if the UI must restore completed reports after refresh.

## Can I ship Inqtrix as a service to end-users?

The repository is explicitly experimental (see the disclaimer in the root `README.md`), but Stack mode ships real multi-user authentication: native email/password (`local`), LDAP/AD bind (`ldap`), and OIDC SSO (`oidc`), each with per-user access tokens and login brute-force throttling (`INQTRIX_LOGIN_RATE_LIMIT_*`). See [Auth modes](../deployment/auth-modes.md) and [Deploy to production](../how-to/deploy-to-production.md).

Still out of scope, and what a reverse proxy / WAF in front must add: general per-request (per-IP) rate limiting beyond the login throttle, and durable cross-worker sharing of completed native run results — finished runs live in memory with a TTL (`RUN_COMPLETED_TTL_SECONDS`), so persist the result payload yourself if a UI must restore reports after a restart. Always terminate TLS in front for remote exposure.

See [Security hardening](../deployment/security-hardening.md).

## How do I add a new search backend?

Implement `SearchProvider`. The ABC has one method (`search(...)`) and must return a typed `GroundedSearchResult` with `GroundedSource` rows for provenance. Pass an instance to `AgentConfig(search=...)`. See [Writing a custom provider](../providers/writing-a-custom-provider.md).

## Why does the evaluation score sometimes show `0`?

`_confidence_parsed` is a boolean marker, not the raw confidence value. The numeric score is `final_confidence`.

If `final_confidence=0` appears in exported metrics, the run did not complete a normal evaluator pass (for example early failure before evaluate parsing). Inspect the nearest evaluate entry for `_evaluate_fallback`, parse flags, and `stop_cascade` details.

See [Iteration log](../observability/iteration-log.md) and [Debugging runs](../observability/debugging-runs.md).

## Does the agent retry on API errors?

Yes, for the built-in LLM providers:

- `AnthropicLLM` retries 5xx/529 and transport errors with exponential backoff.
- `BedrockLLM` retries transient Converse errors and selected transport errors with exponential backoff.
- `AzureOpenAILLM` and `LiteLLM` disable hidden OpenAI SDK retries and run an Inqtrix-owned retry loop for transient 408/409/5xx and SDK timeout/connection errors.

Every LLM retry consults the run deadline, logs a warning, and emits live progress so long waits are visible in the UI. HTTP 429 remains fatal (`AgentRateLimited`) instead of being silently retried. See [Timeouts and errors](../observability/timeouts-and-errors.md).

## How can I validate my setup without running a full research question?

Three levels, cheapest first:

1. `uv run pytest tests/ -v` — fully offline regression suite. Use `uv run pytest tests/ --collect-only -q` for the current count.
2. `uv run python examples/provider_stacks/azure_smoke_tests/test_llm.py` — isolated Azure OpenAI validation, one call.
3. A single research call via the example scripts or parity CLI — real end-to-end.

See [Running tests](../development/running-tests.md).

## How do I keep `provider_stacks/` and `webserver_stacks/` in sync?

They share the same provider construction byte-for-byte by convention; only the run block differs (library vs uvicorn). When you edit one, edit the other. The dedicated test file `tests/test_webserver_examples.py` verifies the symmetry. See [Library mode](../deployment/library-mode.md) and [Web server mode](../deployment/webserver-mode.md).

## Related docs

- [Glossary](glossary.md)
- [Troubleshooting](troubleshooting.md)
- [Debugging runs](../observability/debugging-runs.md)
- [Architecture overview](../architecture/overview.md)
