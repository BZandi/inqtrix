# Iteration log

> Files: `src/inqtrix/state.py` (`append_iteration_log`), `src/inqtrix/nodes.py`

## Scope

The structured, per-round log that Inqtrix appends to `state["iteration_log"]` in testing mode. Unlike [Progress events](progress-events.md), this log is machine-consumable: the parity tooling, the analysis reports, and any ad-hoc debugging scripts read it directly.

## When the log is populated

The iteration log is written unconditionally by the nodes. In production (non-testing mode) the log is attached to the `ResearchResult` export only when `testing_mode` is enabled (see [Settings and env](../configuration/settings-and-env.md), `TESTING_MODE=true`). The cost of writing is negligible; the cost of serialising into a result object is what `testing_mode` gates.

## Entry shape

Each node appends one dict per invocation. The fields vary per node but share a common header:

```json
{
  "node": "evaluate",
  "round": 2,
  "elapsed_ms": 734,
  ...
}
```

`round` is the logical research round, not the call index. A follow-up run starts again at round 0 when the topic has changed.

## Markers emitted

The markers below are the canonical hook points for operators. Each marker is both a log line (level WARNING or INFO) and an iteration-log field; see [Logging](logging.md) for the level mapping.

### Classify

| Marker | Meaning |
|--------|---------|
| `_classify_fallback` | LLM call failed or returned unparseable output; the node used heuristic type inference and a single-question fallback. |
| `_classify_parsed` | All five fields (DECISION, LANGUAGE, RECENCY, TYPE, SUB_QUESTIONS) parsed successfully. |

### Plan

| Marker | Meaning |
|--------|---------|
| `_plan_fallback` | LLM call failed; the node used `[question]` as the single query. |
| `_plan_injected_quality_sites` | Number of `site:`-prefixed queries injected for German policy detection. |
| `_plan_stored_queries` | Number of new queries added after deduplication. |

### Search

| Marker | Meaning |
|--------|---------|
| `_claim_extraction_fallback` | Claim extraction failed for at least one source; lists the failing model name. Keeps the run going. |
| `_search_cache_hits` / `_search_cache_misses` | Counts per round. |
| `_search_results_kept` / `_search_results_dropped` | After context pruning. |

### Evaluate

| Marker | Meaning |
|--------|---------|
| `_confidence_parsed` | Raw integer that the LLM produced. |
| `_evidence_consistency_parsed` | Sanity signal used in Group A of the cascade. |
| `_evidence_sufficiency_parsed` | Sanity signal feeding utility. |
| `_evaluate_fallback` | LLM call failed; previous confidence retained. |
| `_stop_reason` | When `done=True`, names the stop rule (confidence, max_rounds, plateau, utility, stagnation, falsification, cancelled, direct_answer). |

### Answer

| Marker | Meaning |
|--------|---------|
| `_answer_fallback` | LLM call failed; the raw context was returned instead of a synthesised answer. |
| `_answer_citations_selected` | How many URLs the citation selector kept. |
| `_answer_links_sanitized` | How many stray markdown links were stripped. |

## Reading the log

The parity CLI (`inqtrix-parity compare --llm-analysis`) reads the iteration log when it produces its diagnostic report. For ad-hoc inspection:

```python
result = agent.research("...")
for entry in result.metrics.iteration_log:  # requires testing_mode=True
    print(entry["node"], entry.get("_stop_reason", ""))
```

For a run captured via the HTTP `/v1/test/run` endpoint, the JSON payload contains the same list under `state.iteration_log`.

## Design principle

The iteration log is the **single source of truth** for "what did the agent decide, and why". A new branch in a node — new fallback, new cap, new stop heuristic — must add a marker here. Failing to do so hides behaviour from operators and violates Design Principle 1 (no silent fallbacks).

## Related docs

- [Logging](logging.md)
- [Progress events](progress-events.md)
- [Debugging runs](debugging-runs.md)
- [Parity tooling](../development/parity-tooling.md)
