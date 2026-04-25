# Debugging runs

## Scope

The checklist operators reach for when a run misbehaves. Every entry links to the underlying mechanism. If a symptom is not covered here, look at [Iteration log](iteration-log.md) first — it records almost every decision the agent makes.

## Step zero: enable the file log

All debugging flows below assume logs are on disk. The minimum setup:

```bash
export INQTRIX_LOG_ENABLED=true
export INQTRIX_LOG_LEVEL=INFO
export INQTRIX_LOG_FILE=./logs/inqtrix.log
```

For the HTTP server, the example webserver scripts additionally mirror uvicorn output into the same file via `build_uvicorn_log_config(...)` (see [Logging](logging.md)). Without the file sink, `grep`-based post-mortem is not possible.

## Symptom: "My run finds no sources"

1. Check the search cache hit-rate in the iteration log: `_search_cache_hits` vs `_search_cache_misses`. A run with many hits but still no `citations` points at a stale cache.
2. Read the `queries` field on each `search` entry. If the queries look off-topic, the plan node likely mis-parsed the classification output. Look for `_classify_fallback`.
3. Inspect the provider log lines for 4xx/5xx. `PerplexityAPIError` / `BraveSearchAPIError` / `AzureOpenAIAPIError` lines indicate the backend rejected the call — credentials or model-name issues.
4. If `all_citations` is empty but no error was raised, the provider returned an empty body; the iteration log lists this as `_search_results_dropped=N` with `reason="empty_answer"`.

## Symptom: "Confidence stays capped, run will not stop"

1. Read `_confidence_parsed` vs `final_confidence` per round. A large delta indicates a cap is binding.
2. Check which cap fired via the per-round `cap_reason` field: `no_citations`, `low_dominance`, `missing_primary`, `uncovered_aspects`, `contested_claims`, or one of the Group A reasons.
3. `uncovered_aspects` is the most common "invisible" cap. Check `required_aspects` vs `uncovered_aspects` and add evidence targeted at the remaining aspects.
4. If the trajectory is flat low, falsification and stagnation should eventually trigger. See [Falsification](../scoring-and-stopping/falsification.md); if neither fires, verify `round >= 2` and that `prev_conf` is actually being threaded (it is recorded in the iteration log).

## Symptom: "Claim extraction fallback warnings"

The marker `_claim_extraction_fallback` with `model=<name>` means the extractor called the summarize model and the backend rejected it. Two common root causes:

- **Model name leakage from `ModelSettings()` defaults.** The strategies layer previously read `settings.models.effective_summarize_model` (LiteLLM defaults), which failed on Anthropic/Bedrock/Azure stacks. Current code uses `resolve_summarize_model(llm, fallback=...)`. If you wrote a custom strategy, make sure you use the same helper.
- **Deployment misconfiguration on Azure.** Check that the deployment name in `AzureOpenAILLM(default_model=...)` or `summarize_model=...` exists in the target resource.

## Symptom: "Cancel does not stop the run"

The cancel-on-disconnect mechanism is best-effort at node boundaries. A currently running provider call will complete before the cancel takes effect; typical latency is 5-60 seconds depending on the active call. If you need guaranteed sub-second cancel:

- Reduce `REASONING_TIMEOUT` so the in-flight call finishes sooner.
- Force an explicit cancel through your reverse proxy (client-side).
- In-flight HTTP cancellation through the agent is out of scope today (open follow-up).

## Symptom: "HTTP 429 from the server with slots free"

Check `MAX_CONCURRENT` (default 3) and compare to the number of active research runs visible in the access log. If the number of active runs equals `MAX_CONCURRENT`, increase the setting. If not, the semaphore may have leaked — the normal cause would be an exception escaping outside the `stream_response` context manager. Run the cancel-on-disconnect tests and inspect the unified uvicorn/Inqtrix log.

## Symptom: "Answer contains German text the UI does not render"

The default answer prompt is German. There is no public `AgentConfig` field for prompt dictionaries today. To switch to English answers, fork or edit the LLM-facing prompt templates in `src/inqtrix/prompts.py`, or wrap the relevant provider/strategy in your own application code. The progress-message strings are independent and remain German unless you change the source strings (see [Progress events](progress-events.md)).

## Symptom: "HTTP `/health` shows wrong model name"

`/health` reads model identities from the active providers. If you see an odd search model string, check whether the response matches `<ClassName>(unknown)` — that means a custom `SearchProvider` subclass has not implemented the `search_model` property yet (see [Providers overview](../providers/overview.md)).

## Related docs

- [Logging](logging.md)
- [Iteration log](iteration-log.md)
- [Timeouts and errors](timeouts-and-errors.md)
- [Troubleshooting](../reference/troubleshooting.md)
