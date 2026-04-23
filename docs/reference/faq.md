# FAQ

## Why do I see Anthropic 404 errors with a model name like `claude-opus-4.6-agent`?

That model name is the default on `ModelSettings()`; it is a LiteLLM alias, not a real Anthropic model id. Before ADR-WS-8, code paths that read `settings.models.effective_summarize_model` leaked that default into the claim-extraction strategy and the Anthropic backend rejected it. Current code reads constructor-first via `resolve_summarize_model(llm, fallback=...)`. If you wrote a custom strategy, make sure you use the same helper.

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

- `ResearchResult.metrics.total_prompt_tokens` and `.total_completion_tokens` capture aggregate token usage per run (where the provider returns usage metadata).
- The iteration log records per-call usage in testing mode.
- The parity CLI computes per-question deltas in its analysis report.

See [Result schema](../architecture/result-schema.md) and [Parity tooling](../development/parity-tooling.md).

## Why are the prompt templates in German?

The default user base is German-speaking. Prompt strings in `src/inqtrix/prompts.py` are the single exception to the English-only convention (the other exceptions are UI-facing HTTP error strings and demo questions in `examples/`). To switch to English, fork the prompt templates; all node code reads them by key, so a custom prompt dictionary plugs in cleanly.

## Is there a Streamlit UI?

Yes, `webapp.py` at the repository root is a Streamlit chat interface that talks to the HTTP server. It is intentionally minimal and not part of the supported surface. Start it with `streamlit run webapp.py` after the server is running. See [Library mode](../deployment/library-mode.md) for the reference to it.

## Can I ship Inqtrix as a service to end-users?

The repository is explicitly experimental (see the disclaimer in the root `README.md`). The `examples/webserver_stacks/` bundle ships opt-in TLS, API-key, and CORS hardening, but there is no rate limiter, OAuth2/OIDC layer, per-IP throttling, or multi-worker session sharing. Deploying to external users requires a reverse proxy and additional hardening that is explicitly out of scope.

See [Security hardening](../deployment/security-hardening.md).

## How do I add a new search backend?

Implement `SearchProvider`. The ABC has one method (`search(...)`) with a fixed return dict shape (`answer`, `citations`, `related_questions`, `_prompt_tokens`, `_completion_tokens`). Pass an instance to `AgentConfig(search=...)`. See [Writing a custom provider](../providers/writing-a-custom-provider.md).

## Why does the evaluation score sometimes show `0`?

The evaluate node always emits the raw LLM-reported confidence as `_confidence_parsed` and the post-cascade value as `final_confidence`. If only `final_confidence=0` appears, the LLM call likely failed and the node used the `_evaluate_fallback` path. Check the log for the failing call.

See [Iteration log](../observability/iteration-log.md) and [Debugging runs](../observability/debugging-runs.md).

## Does the agent retry on API errors?

Partially:

- `AnthropicLLM` — up to 5 attempts with exponential backoff on 5xx/529.
- `BedrockLLM` — up to 5 attempts on `ThrottlingException`; escalates to `AgentRateLimited` on exhaustion.
- OpenAI SDK clients (`LiteLLM`, `AzureOpenAILLM`, Azure Foundry providers) use the SDK's default retry loop.

Every retry consults the run deadline so retries cannot push past `MAX_TOTAL_SECONDS`. See [Timeouts and errors](../observability/timeouts-and-errors.md).

## How can I validate my setup without running a full research question?

Three levels, cheapest first:

1. `uv run pytest tests/ -v` — fully offline regression suite (~800 tests).
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
