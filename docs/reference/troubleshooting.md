# Troubleshooting

Symptom → cause → fix matrix. For deeper mechanism descriptions, each row links to the explanatory page.

## Startup and configuration

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ResearchAgent()` raises `RuntimeError` about missing provider credentials | No `.env` loaded and no explicit `AgentConfig(llm=..., search=...)` | Call `load_dotenv()` in your script, or pass providers explicitly. See [Library mode](../deployment/library-mode.md). |
| HTTP server logs are empty, console only | `INQTRIX_LOG_ENABLED=false` (default) or file path unset | Set `INQTRIX_LOG_ENABLED=true` and `INQTRIX_LOG_FILE=/path/to/log`. See [Logging](../observability/logging.md). |
| Server starts but `/health` shows wrong `search_model` | Custom `SearchProvider` subclass without an override for the `search_model` property (shows `"<ClassName>(unknown)"`) | Implement the property on the subclass. See [Providers overview](../providers/overview.md). |
| Server starts but `/health` shows LiteLLM default models on a non-LiteLLM stack | Legacy code reading `settings.models.*` instead of the provider | Upgrade; current `/health` reads constructor-first. If writing a new endpoint, use `resolve_summarize_model(llm, fallback)`. See [Debugging runs](../observability/debugging-runs.md). |

## Azure-specific

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `AzureOpenAIAPIError(status_code=400, ...unsupported parameter...)` | `token_budget_parameter` mismatch for the deployment | Switch between `"max_completion_tokens"` and `"max_tokens"` on the provider constructor. See [Azure OpenAI](../providers/azure-openai.md). |
| `AzureOpenAIAPIError(status_code=404)` on first call | Deployment name incorrect | The `default_model` argument must equal the **deployment name** in your Azure resource, not the underlying model id. |
| Foundry Bing calls succeed for 60 minutes, then 401 | Cached token with <10 s lifetime returned by `ClientSecretCredential` / `DefaultAzureCredential` | Long-running servers: accept occasional transient 401 and let the next request refresh; for sub-minute reliability, restart the container periodically. See [Enterprise Azure](../deployment/enterprise-azure.md). |
| `consume_effort_config_warnings` logs rejection on Anthropic Haiku | Haiku does not accept `effort`; Inqtrix warns loudly but keeps running | Either move the call to a Sonnet/Opus role or drop the `effort` kwarg for the Haiku role. |

## Run behaviour

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Run stops at confidence 8 and never goes higher | Aspect coverage cap is active; at least one required aspect is uncovered | Inspect `required_aspects` vs `uncovered_aspects`; supply or wait for evidence. See [Aspect coverage](../scoring-and-stopping/aspect-coverage.md). |
| Run stops at confidence 7 with only mainstream sources | Missing-primary cap | Inject a primary-tier source via `site:`-style queries or a custom strategy. See [Source tiering](../scoring-and-stopping/source-tiering.md). |
| Run loops past round 3 with no progress | Falsification not yet armed (need `prev_conf > 0` and `prev_conf <= 4`) | Reduce `confidence_stop` or wait; stagnation typically terminates by round 4. See [Falsification](../scoring-and-stopping/falsification.md). |
| `_claim_extraction_fallback` warnings in every run | Summarize model name leaked from `ModelSettings` defaults | Use `resolve_summarize_model(llm, fallback)` instead of `settings.models.effective_summarize_model`. Current strategies factory already does. |

## HTTP API

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `curl --max-time N` does not actually cancel the run | Cancel is best-effort at node boundaries, not mid-call | Reduce `REASONING_TIMEOUT`; the next node boundary will honour the cancel. See [Web server mode](../deployment/webserver-mode.md). |
| 429 returned with available capacity | `MAX_CONCURRENT` reached or semaphore leaked | Increase the setting; check the access log for long-running runs. |
| SSE stream ends abruptly | Client closed the connection; watcher task detected disconnect and set `cancel_event` | Reconnect on the client side or disable progress streaming with `"include_progress": false`. |
| CORS request blocked in browser with credentials | `INQTRIX_SERVER_CORS_ORIGINS=*` with credentials enabled — browsers reject wildcard + credentials | List explicit origins. See [Security hardening](../deployment/security-hardening.md). |

## Tests and local development

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `uv run pytest` fails immediately with import errors | Not installed in editable mode | `uv sync --extra dev` or `pip install -e ".[dev]"`. See [Installation](../getting-started/installation.md). |
| Replay tests require API keys | Mis-set `INQTRIX_RECORD_MODE` | Unset it; default is `none` (offline replay). See [Testing strategy](../development/testing-strategy.md). |
| Two logger tests fail when the full suite runs but pass in isolation | Known test-order pollution from logger-configuration tests | Run the affected test with `-k` in isolation; the underlying code is correct. |

## Related docs

- [Debugging runs](../observability/debugging-runs.md)
- [FAQ](faq.md)
- [Logging](../observability/logging.md)
- [Timeouts and errors](../observability/timeouts-and-errors.md)
