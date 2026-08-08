# Iteration log

> Files: `src/inqtrix/state.py` (`append_iteration_log`), `src/inqtrix/nodes.py`

## Scope

The structured, per-round log that Inqtrix appends to `state["iteration_logs"]` (`AgentState` key, `list[dict[str, Any]]`) in testing mode. Unlike [Progress events](progress-events.md), this log is machine-consumable: the parity tooling, the analysis reports, and any ad-hoc debugging scripts read it directly.

## When the log is populated

Node summaries are always emitted to structured runtime logging. The in-memory list `state["iteration_logs"]` is populated in testing mode only (see [Settings and env](../configuration/settings-and-env.md), `TESTING_MODE=true`). It is exported by test/parity entry points as a top-level `iteration_logs` list; it is not part of the public `ResearchResult.metrics` Pydantic model.

## Entry shape

Each node appends one summary dict per invocation. Forensic mode may add additional event dicts for source, citation, claim, stop, and answer lineage. All entries share a common header:

```json
{
  "event": "iteration_summary",
  "node": "evaluate",
  "run_id": "run_...",
  "event_seq": 17,
  "round": 2,
  "duration_s": 0.734,
  ...
}
```

`round` is the logical research round, not the call index. Every new
`graph.run()` starts at round 0.

## Markers emitted

The markers below are the canonical hook points for operators. Each marker is both a log line (level WARNING or INFO) and an iteration-log field; see [Logging](logging.md) for the level mapping.

### Classify

| Marker | Meaning |
|--------|---------|
| `_classify_fallback` | LLM call failed; the node used heuristic type inference and a single-question fallback. |
| `_classify_parsed` | The node completed without the fallback path. |

### Plan

| Marker | Meaning |
|--------|---------|
| `_plan_fallback` | LLM call failed; the node used `[question]` as the single query. |
| `_plan_stored_queries` | Number of new queries added after deduplication. |
| `query_slot_count` / `query_slot_types` | Planned fan-out slots that drove one targeted query each. |
| `crosscheck_target_count` | Number of claims selected for independent later-round verification. |

### Search

| Marker | Meaning |
|--------|---------|
| `_claim_extraction_fallback` | Claim extraction failed for at least one source. Full-run failure in forensic/deep mode emits `ALGO-FAIL claim_extraction` and blocks normal report synthesis. |
| `claim_extraction_modes` | Per-round counts of `structured_output`, `legacy_text_json`, or other diagnostic modes used by claim extraction. |
| `claim_extraction_raw_claim_count` / `claim_extraction_normalized_claim_count` / `claim_extraction_filtered_claim_count` | Per-query counters in `query_summary` that show whether valid-empty extraction came from an empty model response or from local filtering after schema-valid claims were returned. |
| `_search_results_kept` / `_search_results_dropped` | Context blocks retained or pruned after search assembly. |
| `query_record_ids` / `source_record_ids` / `provider_citation_record_ids` | IDs linking the search summary to provider-neutral forensic records. |
| `evidence_record_count` / `report_eligible_evidence_count` | Counts of EvidenceRecords and (subset) report-eligible records after search. |
| `evidence_depth_gap` | Diagnostic showing whether report evidence is too single-source-heavy for plateau stopping. |
| `algorithm_failure_count` / `blocking_algorithm_failure_count` | Number of visible core-path failures recorded so far, and how many of them block final-report synthesis. |
| `cancel_abandoned_work` | Run cancellation interrupted a fan-out (search or claim extraction): `abandoned` queued calls were cancelled before starting, `in_flight` running calls were awaited (they stop at the provider cancel probe), out of `total`. Also emitted as a warning progress message. |

### Evaluate

| Marker | Meaning |
|--------|---------|
| `_confidence_parsed` | Whether the LLM-produced confidence integer was parsed. |
| `_evidence_consistency_parsed` | Sanity signal used in Group A of the cascade. |
| `_evidence_sufficiency_parsed` | Sanity signal feeding utility. |
| `_evaluate_fallback` | LLM call failed; confidence is derived from a conservative fallback clamp and gaps are set defensively. |
| `prev_conf` | The previous round's `final_confidence` as it entered the cascade. Used by the plateau / stagnation rules and by guardrails. |
| `evidence_depth_gap` | Same structural evidence-depth diagnostic as search, recomputed before evaluation. |
| `report_eligible_evidence_count` | Count of report-eligible EvidenceRecords. Used by the report-evidence floor stop suppression (`min_report_eligible_evidence`). |
| `_stop_reason` | When `done=True`, names the final stop rule (`confidence_stop`, `round_limit`, `plateau_stop`, `utility_stop`, `deadline_exceeded`, `already_done`, or another normalized tag). |
| `stop_cascade` | Full structured cascade inputs and the final stop decision. |

### Answer

| Marker | Meaning |
|--------|---------|
| `_answer_fallback` | `True` when any answer-synthesis fallback path was taken (timeout, primary-model API error without configured fallback model, or fallback model also failed). The answer body itself begins with a visible `> [!WARNING] Antwort-Synthese-Fallback aktiv` block in this case. |
| `_answer_fallback_kind` | Discriminator for the fallback path: `""` (success), `timeout`, `no_fallback_model`, `fallback_model_failed`. |
| `_answer_fallback_reason` | Human-readable explanation including provider error class, error message, and which fallback model (if any) was attempted. |
| `_answer_citations_selected` | How many URLs the citation selector kept. |
| `_answer_links_sanitized` | How many stray markdown links were stripped. |
| `rendered_record_count` / `omitted_record_count` | Records rendered into the answer prompt's evidence overview vs records dropped by the char budget (visible as a "HINWEIS: …" footer in the overview). |
| `evidence_overview_chars` | Total character length of the rendered evidence overview that went into the section system prompt. |
| `answer_prompt_diagnostics` | Extra answer-prompt density event emitted in testing mode or forensic observability. Counts evidence overview chars, citations, rendered records, and verification-label mix without logging full prompts in normal mode. |
| `answer_claim_binding_count` | Count of final answer citation-to-claim bindings. |
| `answer_evidence_binding_count` | Count of final answer-segment to EvidenceRecord audit rows. |

## Forensic events

Set `OBSERVABILITY_PROFILE=forensic` to populate detailed lineage in the protected iteration-log/audit path. At `DEBUG` level the ordinary logger receives only a content-minimized projection: IDs, lifecycle/status fields, models, counters, usage and timings. Exact queries, URLs, provider prose, snippets, claim/evidence text and prompt views remain solely in the authorized redacted audit representation; raw provider request bodies, headers, SDK responses and credentials enter neither sink.

| Event | Meaning |
|-------|---------|
| `query_record` | Protected audit: exact query plus IDs and parameters. Operational log: query/source/citation IDs, round, index, provider. |
| `source_record` | Protected audit: source URL/domain and provenance. Operational log: source IDs, provider, tier and access status. |
| `provider_citation_record` | Protected audit: title/snippet/URL provenance. Operational log: query/source/citation IDs, rank, origin and provider. |
| `query_summary` | Per-query summary, claim counts, and claim-extraction mode (`structured_output` vs. `legacy_text_json`). |
| `evidence_record` | Protected audit: source passages, snippets and raw supports. Operational log: evidence/query/source/citation IDs, tier, provider and counts/status. |
| `claim_record` | Protected audit: raw extracted claim and support. Operational log: claim/query/evidence/source/citation IDs and categorical status. |
| `claim_merge` | Consolidated claim with member raw-claim IDs, status, support/contradiction, source IDs. |
| `evidence_verification_projection` | Aggregate projection of consolidated verification back onto EvidenceRecords. |
| `evidence_overview_render` | EvidenceOverview render counts (`rendered_record_count`, `omitted_record_count`, visible `allowed_urls` size, visible label count, `label_by_evidence_id` size) and evidence-depth diagnostics for answer synthesis. |
| `score_snapshot` | Per-phase diagnostic snapshot used for progress and stop diagnostics; it records values that are also present on `result_state`. |
| `stop_cascade` | Evaluate stop cascade inputs and final normalized reason. |
| `citation_selection` | Answer prompt citation selection and trimming. |
| `answer_prompt_diagnostics` | Compact answer-prompt density counters, including prompt evidence unit counts and claimless evidence counts. |
| `answer_claim_binding` | Final answer segment/citation/source/claim binding. |
| `answer_sentence_audit` | Final answer segment to EvidenceRecord audit (`binding_status` in `matched` / `source_context` / `unknown_citation`). |

## Reading the log

The parity CLI (`inqtrix-parity compare --llm-analysis`) reads the iteration log when it produces its diagnostic report. For ad-hoc inspection, use a test/parity result payload or the raw `result_state` returned by internal test helpers:

```python
import requests

payload = requests.post(
    "http://localhost:5100/v1/test/run",
    json={"question": "..."},
    timeout=300,
).json()
for entry in payload["iteration_logs"]:
    print(entry["node"], entry.get("_stop_reason", ""))
```

For a run captured via the HTTP `/v1/test/run` endpoint, the JSON payload contains the same list under the top-level `iteration_logs` key.

### Stop/continue triage recipe

Use the latest `evaluate` entry and read in this order:

1. `prev_conf`, `_confidence_parsed`, `_evidence_*_parsed` (input quality).
2. `guardrail_reasons` (deterministic confidence caps).
3. `stop_cascade` (`utility`, `plateau_stop`, `stagnation_detected`, final gate).
4. `_stop_reason` and `stop_cascade.suppressed_by_min_rounds` (final decision vs. suppressed decision).

This sequence is enough to explain why a run stopped or continued without reading node code.

## Design principle

The iteration log is the **single source of truth** for "what did the agent decide, and why". A new branch in a node — new fallback, new cap, new stop heuristic — must add a marker here. Failing to do so hides behaviour from operators and violates Design Principle 1 (no silent fallbacks).

## Related docs

- [Logging](logging.md)
- [Evidence pipeline](../architecture/evidence-pipeline.md)
- [Score ledger](../scoring-and-stopping/score-ledger.md)
- [Progress events](progress-events.md)
- [Debugging runs](debugging-runs.md)
- [Forensic cookbook](forensic-cookbook.md)
- [Parity tooling](../development/parity-tooling.md)
