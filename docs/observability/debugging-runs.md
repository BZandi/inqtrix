# Debugging runs

## Scope

The checklist operators reach for when a run misbehaves. Every entry links to the underlying mechanism. If a symptom is not covered here, look at [Iteration log](iteration-log.md) first — it records almost every decision the agent makes.

## Step zero: enable the file log

All debugging flows below assume logs are on disk. The minimum setup:

```bash
export INQTRIX_LOG_ENABLED=true
export INQTRIX_LOG_LEVEL=INFO
```

For the HTTP server, the example webserver scripts additionally mirror uvicorn output into the same file via `build_uvicorn_log_config(...)` (see [Logging](logging.md)). Without the file sink, `grep`-based post-mortem is not possible.

For full source-to-claim-to-answer reconstruction, enable forensic events:

```bash
export INQTRIX_LOG_ENABLED=true
export INQTRIX_LOG_LEVEL=DEBUG
export OBSERVABILITY_PROFILE=forensic
```

## Symptom: "My run finds no sources"

1. Check the `search` summary entry: `queries`, `sources_found`, `_search_results_kept`, `_search_results_dropped`, and provider notices. In forensic mode, follow `query_record_ids` to `source_record` and `provider_citation_record` events.
2. Read the `queries` field on each `search` entry. If the queries look off-topic, the plan node likely mis-parsed the classification output. Look for `_classify_fallback`.
3. Inspect the provider log lines for 4xx/5xx. `PerplexityAPIError` / `AzureOpenAIAPIError` lines indicate the backend rejected the call — credentials or model-name issues.
4. If `all_citations` is empty but no error was raised, the provider returned an empty body. In forensic mode, `source_record.access_status` and provider notices show whether the provider returned no answer or only legacy citations.

## Symptom: "Confidence stays capped, run will not stop"

1. Read `_confidence_parsed` vs `final_confidence` per round. A large delta indicates a cap is binding.
2. Check `guardrail_reasons` and `stop_cascade` on the evaluate entry. `stop_cascade.final_stop_reason` is the canonical stop outcome.
3. `uncovered_aspects` is the most common "invisible" cap. Check `required_aspects` vs `uncovered_aspects` and add evidence targeted at the remaining aspects.
4. If the trajectory is flat low, falsification and stagnation should eventually trigger. See [Falsification](../scoring-and-stopping/falsification.md); if neither fires, verify `round >= 2` and that `prev_conf` is actually being threaded — the evaluate iteration-log entry now records it as a top-level field alongside `stop_cascade`.

## Symptom: "I cannot trace final answer sentences to sources"

1. Re-run with `OBSERVABILITY_PROFILE=forensic` and `INQTRIX_LOG_LEVEL=DEBUG`.
2. Follow `provider_citation_record` to `source_record`, then `claim_record`, then `claim_merge`.
3. Inspect `answer_claim_binding` events. A `binding_status="citation_without_claim"` means the final answer linked a selected citation that was not represented in the consolidated claim ledger.
4. If bindings are missing entirely, check whether the answer contains allowed Markdown links after `_answer_links_sanitized`; no links means there is no answer-level citation anchor to bind.

## Symptom: "Claim extraction fallback warnings"

The marker `_claim_extraction_fallback` with `model=<name>` means the extractor called the claim-extraction model and the backend rejected it. Two common root causes:

- **Model name leakage from `ModelSettings()` defaults.** The strategies layer previously read `settings.models.effective_claim_extract_model` (LiteLLM defaults), which failed on Anthropic/Bedrock/Azure stacks. Current code uses `resolve_claim_extract_model(llm, fallback=...)`. If you wrote a custom strategy, make sure you use the same helper.
- **Deployment misconfiguration on Azure.** Check that the deployment name in `AzureOpenAILLM(default_model=...)` or `claim_extract_model=...` exists in the target resource.

## Symptom: "Cancel does not stop the run"

The cancel mechanisms are best-effort at node boundaries. `/v1/chat/completions` uses disconnect cancellation; `/v1/runs/{run_id}/cancel` sets the same per-run cancel event for native runs. A currently running provider call will complete before the cancel takes effect; typical latency is 5-60 seconds depending on the active call. If you need guaranteed sub-second cancel:

- Reduce `REASONING_TIMEOUT` so the in-flight call finishes sooner.
- Force an explicit cancel through your reverse proxy (client-side).
- In-flight HTTP cancellation through the agent is out of scope today (open follow-up).

## Symptom: "HTTP 429 from the server with slots free"

Check which HTTP surface returned 429:

- `/v1/chat/completions`: compare `MAX_CONCURRENT` (default 6) to active research requests in the access log. If active requests equal the cap, increase the setting. If not, the semaphore may have leaked — the normal cause would be an exception escaping outside the `stream_response` context manager.
- `/v1/runs`: inspect `GET /v1/runs`. If active runs equal `RUN_MAX_CONCURRENT` (or `MAX_CONCURRENT` when `RUN_MAX_CONCURRENT` is unset) and queued runs equal `RUN_QUEUE_MAX_SIZE`, the queue is full. Increase `RUN_QUEUE_MAX_SIZE`, increase the active native run cap, or add front-door rate limiting. If the list looks empty but 429 persists, check whether terminal records are still inside `RUN_COMPLETED_TTL_SECONDS` with open event subscribers.

Run the cancel-on-disconnect and run-store tests, then inspect the unified uvicorn/Inqtrix log.

## Symptom: "Answer contains German text the UI does not render"

The default answer prompt is German. There is no public `AgentConfig` field for prompt dictionaries today. To switch to English answers, fork or edit the LLM-facing prompt templates in `src/inqtrix/prompts.py`, or wrap the relevant provider/strategy in your own application code. The progress-message strings are independent and remain German unless you change the source strings (see [Progress events](progress-events.md)).

## Symptom: "HTTP `/health` shows wrong model name"

`/health` reads model identities from the active providers. If you see an odd search model string, check whether the response matches `<ClassName>(unknown)` — that means a custom `SearchProvider` subclass has not implemented the `search_model` property yet (see [Providers overview](../providers/overview.md)).

## Related docs

- [Logging](logging.md)
- [Iteration log](iteration-log.md)
- [Run events](run-events.md)
- [Timeouts and errors](timeouts-and-errors.md)
- [Troubleshooting](../reference/troubleshooting.md)
