# Forensic cookbook

> Files: `src/inqtrix/runtime_logging.py`, `src/inqtrix/nodes.py`, `src/inqtrix/state.py`

## Scope

How to reconstruct an answer end-to-end from forensic logs: which events are emitted, which fields each event carries, and how to follow IDs from `query_record` through `query_summary`, `claim_merge`, answer prompt inputs, generated sections, and finally to `answer_claim_binding`. Use this page when you need to explain "where did sentence X come from" or "why did the run stop in round N".

This page is operator-facing. The architectural rationale and ADR list live in [Architecture overview](../architecture/overview.md) and the local `.cursor/memory/architecture-decisions.md` (gitignored).

## Enabling forensic mode

Forensic events ride on the standard logger. For a forensic **file log**, all three switches are required: file logging on, logger threshold at `DEBUG`, and the forensic observability profile:

```bash
export INQTRIX_LOG_ENABLED=true
export INQTRIX_LOG_LEVEL=DEBUG
export OBSERVABILITY_PROFILE=forensic
```

If `OBSERVABILITY_PROFILE=forensic` is set without `INQTRIX_LOG_LEVEL=DEBUG`, the detailed events are produced inside the run but their `ITERATION ...: {...}` file-log lines remain below the logger threshold. The events are also placed into `state["iteration_logs"]` (`AgentState` key, `list[dict[str, Any]]`) whenever testing mode is active (`TESTING_MODE=true` env, `AgentSettings(testing_mode=True)`, or HTTP `/v1/test/run`). Both sinks share the same allowlist and redaction pipeline (`runtime_logging.sanitize_event_payload`), so URL query parameters such as `api_key=...`, `token=...`, bearer tokens, and provider raw payloads never reach either sink. See [Logging](logging.md) for the redaction details.

There is no required second log artifact. The forensic file log is the audit trail: compact `TRACE ...` lines are human-readable signposts, and structured `ITERATION ...: {...}` lines in the same `logs/inqtrix_*.log` file carry the reconstructable payloads.

## Event catalog

Every forensic event carries a common envelope: `event`, `event_seq`, `node`, `run_id`, `timestamp`. The fields documented below are the per-event payload fields that live alongside the envelope.

### `run_start`

```json
{
  "event": "run_start",
  "run_mode": "library",
  "run_id": "run_2026-05-09T08-12-44Z_a1b2c3",
  "question_length": 87,
  "history_length": 0,
  "llm": {"provider": "AzureOpenAILLM", "reasoning_model": "gpt-5.1", "claim_extract_model": "gpt-5.1-mini"},
  "search": {"provider": "AzureFoundryWebSearch", "engine": "foundry-web:search-agent@latest"},
  "settings": {"observability_profile": "forensic", "max_rounds": 4, "confidence_stop": 9, "max_total_seconds": 300, "testing_mode": false}
}
```

### `query_record`

```json
{
  "event": "query_record",
  "query_id": "qry_round-0_idx-1_b7e2",
  "round": 0,
  "query_index": 1,
  "query": "Wirkstoff X Phase III Studie 2025",
  "domain_filter": ["site:nih.gov"],
  "provider": "AzureFoundryWebSearch",
  "source_ids": ["src_1c4f", "src_92aa"],
  "citation_ids": ["cit_q-b7e2_s-1c4f", "cit_q-b7e2_s-92aa"]
}
```

### `source_record`

```json
{
  "event": "source_record",
  "source_id": "src_1c4f",
  "url": "https://www.nih.gov/news-events/news-releases/wirkstoff-x-phase-iii",
  "canonical_url": "https://www.nih.gov/news-events/news-releases/wirkstoff-x-phase-iii",
  "domain": "nih.gov",
  "provider": "AzureFoundryWebSearch",
  "first_seen_query_id": "qry_round-0_idx-1_b7e2",
  "first_seen_rank": 1,
  "origin": "search_result",
  "tier": "primary",
  "tier_reason": "regulator-domain",
  "access_status": "ok"
}
```

`source_record` is emitted **once per unique source ID**. Subsequent queries that hit the same canonical URL produce only `provider_citation_record` events; the dedup happens in the search node (see `src/inqtrix/nodes.py`).

### `provider_citation_record`

```json
{
  "event": "provider_citation_record",
  "citation_id": "cit_q-b7e2_s-1c4f",
  "query_id": "qry_round-0_idx-1_b7e2",
  "source_id": "src_1c4f",
  "url": "https://www.nih.gov/news-events/news-releases/wirkstoff-x-phase-iii",
  "canonical_url": "https://www.nih.gov/news-events/news-releases/wirkstoff-x-phase-iii",
  "rank": 1,
  "origin": "search_result",
  "provider": "AzureFoundryWebSearch",
  "title": "Wirkstoff X Phase III Studie",
  "snippet": "Die Studie zeigte..."
}
```

### `query_summary`

`query_summary` restores the per-query information that operators need to understand what the algorithm actually processed before claim extraction and final answer composition.

```json
{
  "event": "query_summary",
  "query_id": "qry_round-0_idx-1_b7e2",
  "round": 0,
  "query_index": 1,
  "query": "Wirkstoff X Phase III Studie 2025",
  "summary": "- Die Studie erreichte ...\n- Die Nebenwirkungen ...",
  "claims_extracted": 4,
  "claims_kept": 3,
  "claim_extraction_mode": "structured_output",
  "claim_extraction_schema": "inqtrix_claim_extraction_v1",
  "claim_extraction_structured_supported": true,
  "claim_extraction_valid_empty": false,
  "claim_extraction_raw_claim_count": 4,
  "claim_extraction_normalized_claim_count": 4,
  "claim_extraction_filtered_claim_count": 0,
  "evidence_context_char_count": 1880,
  "evidence_context_source_count": 2,
  "claims_sample": ["Die Studie erreichte den primaeren Endpunkt."],
  "source_ids": ["src_1c4f"],
  "citation_ids": ["cit_q-b7e2_s-1c4f"],
  "urls": ["https://www.nih.gov/news-events/news-releases/wirkstoff-x-phase-iii"],
  "prompt_tokens": 8420,
  "completion_tokens": 611,
  "provider_notice": ""
}
```

Use this event to inspect the context-level information that later enters
`s["context"]`. Claims are the verification layer; `summary` and the rendered
EvidenceRecord source context are the richer information layer used for the
report body. `claim_extraction_valid_empty=true` means the extractor returned a
valid empty `claims` array while source context still existed; parse/API or
token-limit failures appear separately through `claim_notice`, `claim_fallbacks`,
the `_claim_extraction_fallback` marker, and `algorithm_failure` events. In
forensic/deep runs, a run-wide claim-extraction failure blocks normal report
synthesis. `claim_extraction_mode` shows whether the query used provider-native
schema enforcement (`structured_output`) or the legacy text-JSON prompt
(`legacy_text_json`). The raw/normalized/filtered claim counters distinguish a
model-valid empty response (`raw=0`) from a response that contained schema-valid
claim objects that local normalization later discarded. After a provider
migration, use these fields together with `scripts/debug_research_log.py` to
confirm that the intended adapter path is active before comparing failure
counts.

### `claim_record`

```json
{
  "event": "claim_record",
  "raw_claim_id": "rcl_b7e2_001",
  "query_id": "qry_round-0_idx-1_b7e2",
  "signature": "wirkstoff x phase iii primaerendpunkt",
  "claim_text": "Die Phase-III-Studie zu Wirkstoff X erreichte den primaeren Endpunkt im Mai 2025.",
  "evidence_snippet": "Die Studie zu Wirkstoff X erreichte im Mai 2025 ihren primaeren Endpunkt.",
  "claim_type": "factual",
  "polarity": "support",
  "needs_primary": true,
  "source_ids": ["src_1c4f"],
  "citation_ids": ["cit_q-b7e2_s-1c4f"],
  "source_urls": ["https://www.nih.gov/news-events/news-releases/wirkstoff-x-phase-iii"],
  "published_date": "2025-05-09",
  "round": 0
}
```

`claim_text` stays atomic on purpose. A paragraph can contain several statements with different evidence status; the claim ledger needs one checkable statement per row. `evidence_snippet` is the short supporting text span that makes the audit trail readable without turning claims into paragraph-sized bundles.

### `claim_merge`

```json
{
  "event": "claim_merge",
  "claim_id": "clm_wirkstoff-x-phase-iii_a3b1",
  "signature": "wirkstoff x phase iii primaerendpunkt",
  "member_claim_ids": ["rcl_b7e2_001", "rcl_d92e_004"],
  "status": "verified",
  "status_reason": "two-sources-agree",
  "evidence_snippets": ["Die Studie ... erreichte den primaeren Endpunkt.", "Die FDA bestaetigte ..."],
  "support_count": 2,
  "contradict_count": 0,
  "source_ids": ["src_1c4f", "src_92aa"],
  "citation_ids": ["cit_q-b7e2_s-1c4f", "cit_q-b7e2_s-92aa"],
  "source_urls": ["https://www.nih.gov/news-events/news-releases/wirkstoff-x-phase-iii", "https://www.fda.gov/news-events/press-announcements/wirkstoff-x"],
  "round_first_seen": 0,
  "round_last_updated": 1
}
```

### `stop_cascade`

```json
{
  "event": "stop_cascade",
  "round": 1,
  "max_rounds": 4,
  "min_rounds": 1,
  "confidence": 9,
  "confidence_stop_target": 9,
  "utility_score": 0.62,
  "utility_stop": false,
  "plateau_stop": false,
  "stagnation_detected": false,
  "falsification_triggered": false,
  "done_after_utility": false,
  "done_after_plateau": false,
  "confidence_stop": true,
  "round_limit": false,
  "rendered_record_count": 6,
  "omitted_record_count": 0,
  "report_eligible_evidence_count": 6,
  "min_report_eligible_evidence": 3,
  "evidence_depth_gap_active": false,
  "suppressed_by_min_rounds": false,
  "suppressed_by_report_evidence_floor": false,
  "suppressed_stop_reason": "",
  "final_stop_reason": "confidence_stop",
  "final_done": true
}
```

### `citation_selection`

```json
{
  "event": "citation_selection",
  "selected_prompt_citations": ["https://www.nih.gov/...", "https://www.fda.gov/..."],
  "body_prompt_citations": ["https://www.nih.gov/...", "https://www.fda.gov/..."],
  "prompt_citations_used_fallback": false,
  "prompt_citations_trimmed_for_body": 0,
  "prompt_citations_trimmed_by_budget": 0,
  "answer_claim_binding_count": 3
}
```

### `answer_prompt_inputs`

```json
{
  "event": "answer_prompt_inputs",
  "report_profile": "deep",
  "model": "gpt-5.1",
  "context_count": 8,
  "context_previews": [{"context_id": "ctx_a1b2", "context_preview": "Die Studie ..."}],
  "context_char_count": 14620,
  "report_evidence_bundle_count": 3,
  "prompt_evidence_unit_count": 8,
  "prompt_evidence_unit_types": {"verified_claim": 3, "contested_claim": 1, "source_context": 3, "unverified_candidate": 1},
  "selected_prompt_citations": ["https://www.nih.gov/..."],
  "body_prompt_citations": ["https://www.nih.gov/..."],
  "consolidated_claim_count": 7,
  "claim_prompt_max_items": 16,
  "claim_prompt_view": "[1] status=verified ... Evidence: ...",
  "section_plan": [{"heading": "Executive Summary", "required": true}]
}
```

This is the answer-side checkpoint: it records the sanitized context previews,
prompt evidence density, claim view, citation set, profile, model, and section
plan that the final report composer actually received. The full prompt evidence
text is capped by the runtime logging allowlist; use the counters here and
`answer_prompt_diagnostics` to understand evidence density without relying on
raw provider payloads.

### `answer_prompt_diagnostics`

```json
{
  "event": "answer_prompt_diagnostics",
  "context_block_count": 8,
  "context_chars": 14620,
  "report_evidence_chars": 22480,
  "unverified_evidence_chars": 0,
  "prompt_citation_count": 18,
  "all_citation_count": 42,
  "evidence_record_count": 28,
  "report_bundle_count": 5,
  "selected_report_bundle_count": 5,
  "prompt_evidence_unit_count": 12,
  "prompt_source_context_unit_count": 5,
  "prompt_unverified_unit_count": 2,
  "claimless_evidence_count": 14,
  "source_context_only_count": 10
}
```

This event is emitted in testing mode or forensic observability. It is the
fastest way to see whether a thin report came from claim extraction, bundle
materialisation, source-context promotion, citation selection, or prompt
rendering budget.

### `answer_section`

```json
{
  "event": "answer_section",
  "heading": "Analyse",
  "position": 3,
  "model": "gpt-5.1",
  "content_length": 4210,
  "content_preview": "## Analyse\nDie wichtigsten Befunde ...",
  "finish_reason": "stop",
  "limit_hit": false,
  "incomplete": false,
  "prompt_tokens": 12400,
  "completion_tokens": 1100,
  "section_evidence_unit_count": 6,
  "used_evidence_labels": ["E1.1", "S4.1"]
}
```

Section events make the final composition step auditable without storing raw provider request/response payloads.

### `answer_claim_binding`

```json
{
  "event": "answer_claim_binding",
  "binding_id": "bind_seg-2_src-1c4f_clm-a3b1",
  "answer_segment_id": "segment_2",
  "answer_segment_preview": "Die Studie erreichte den primaeren Endpunkt im Mai 2025 [1](https://www.nih.gov/...).",
  "citation_url": "https://www.nih.gov/news-events/news-releases/wirkstoff-x-phase-iii",
  "source_id": "src_1c4f",
  "citation_id": "cit_q-b7e2_s-1c4f",
  "claim_id": "clm_wirkstoff-x-phase-iii_a3b1",
  "claim_status": "verified",
  "binding_status": "matched"
}
```

`binding_status` values:

- `matched` -- claim signature or three+ content tokens overlap the answer segment text. Claims at this status drive `claim["used_in_answer"]`.
- `source_only_binding` -- the URL is cited in this segment, but no claim that shares the URL plausibly carries the segment text. Lineage is preserved without false-positive matches.
- `citation_without_claim` -- the URL is cited in this segment and no consolidated claim references the URL at all (legacy, manually injected, or post-claim-extraction citation).

The binding is computed against the **answer body only**, captured before the appendix sections (references / further reading) are appended. Reference-link URLs in the appendix do not produce bindings.

### `run_end`

```json
{
  "event": "run_end",
  "run_id": "run_2026-05-09T08-12-44Z_a1b2c3",
  "run_mode": "library",
  "status": "ok",
  "reason": "confidence_stop",
  "elapsed_s": 47.812,
  "round": 1,
  "done": true,
  "cancelled": false,
  "final_confidence": 9,
  "total_citations": 8,
  "evidence_record_count": 14,
  "report_eligible_evidence_count": 8,
  "consolidated_claims_count": 7,
  "total_prompt_tokens": 12530,
  "total_completion_tokens": 4180
}
```

## Source -> claim -> answer walkthrough

A minimal run: one question, two queries hitting one shared NIH URL plus one new FDA URL, two consolidated claims (one of which ends up in the answer).

| Step | Event | Key IDs |
|------|-------|---------|
| 1 | `query_record` round=0 | `query_id=qry...b7e2` -> `source_ids=[src_1c4f]`, `citation_ids=[cit_q-b7e2_s-1c4f]` |
| 2 | `source_record` (NIH) | `source_id=src_1c4f` (emitted once) |
| 3 | `provider_citation_record` | `citation_id=cit_q-b7e2_s-1c4f` linking `query_id` to `source_id` |
| 4 | `query_summary` | `query_id=qry...b7e2`, `summary=...`, `claims_extracted=...` |
| 5 | `claim_record` | `raw_claim_id=rcl_b7e2_001` -> `source_ids=[src_1c4f]`, `citation_ids=[cit_q-b7e2_s-1c4f]`, `evidence_snippet=...` |
| 6 | `query_record` round=1 | `source_ids=[src_1c4f, src_92aa]`. The NIH URL is reused. |
| 7 | `source_record` (FDA) | `source_id=src_92aa` (NIH is **not** re-emitted; check the dedup contract). |
| 8 | `provider_citation_record` | `citation_id=cit_q-d92e_s-1c4f` (per-query) |
| 9 | `claim_record` | `raw_claim_id=rcl_d92e_004` -> `source_ids=[src_1c4f, src_92aa]` |
| 10 | `claim_merge` | `claim_id=clm...a3b1`, `member_claim_ids=[rcl_b7e2_001, rcl_d92e_004]`, `status=verified` |
| 11 | `stop_cascade` | `final_stop_reason=confidence_stop`, `final_done=true` |
| 12 | `citation_selection` | The NIH and FDA URLs are kept for the answer body. |
| 13 | `answer_prompt_inputs` | Context previews, claim prompt view, citations, and section plan sent to the composer. |
| 14 | `answer_section` | Generated section metadata and preview. |
| 15 | `answer_claim_binding` | `claim_id=clm...a3b1`, `citation_id=cit_q-b7e2_s-1c4f`, `binding_status=matched` |
| 16 | `run_end` | `total_citations=2`, `consolidated_claims_count=1` |

To trace an answer sentence backwards, start with `answer_claim_binding`: the segment ID locates the sentence in the body, `claim_id` joins to `claim_merge`, `member_claim_ids` join to `claim_record` rows, and each `claim_record` carries `query_id` plus `source_ids`/`citation_ids` to recover the originating `provider_citation_record` and `source_record`. To trace a stop decision, start with `stop_cascade` and read `final_stop_reason` plus the per-rule booleans.

## Querying logs

### `jq` recipes

Bindings (matched vs source-only) by claim:

```bash
jq -r 'select(.event=="answer_claim_binding") | "\(.binding_status)\t\(.citation_id)\t\(.claim_id)\t\(.citation_url)"' run.log.jsonl
```

Trace a single source through the lineage:

```bash
SRC=src_1c4f
jq --arg sid "$SRC" '
  select(
    (.event=="source_record" and .source_id==$sid)
    or ((.event=="provider_citation_record" or .event=="claim_record" or .event=="claim_merge")
        and ((.source_ids // [])[]? == $sid))
    or (.event=="answer_claim_binding" and .source_id==$sid)
  )
' run.log.jsonl
```

Verify single-`source_record` invariant (one row per unique URL):

```bash
jq -r 'select(.event=="source_record") | .canonical_url' run.log.jsonl | sort | uniq -c | sort -nr | head
```

Quick stop reason histogram:

```bash
jq -r 'select(.event=="stop_cascade") | .final_stop_reason' run.log.jsonl | sort | uniq -c | sort -nr
```

### Python recipe

The structured events live inside the standard `inqtrix` log lines under prefixes such as `RUN metadata: {...}` and `ITERATION search: {...}`. To parse a non-JSONL file, extract the JSON object at the end of each line:

```python
import json
import re
from collections import defaultdict
from pathlib import Path

EVENT_LINE = re.compile(r"\| inqtrix \| [^:]+: (\{.*\})$")

bindings_by_claim: dict[str, list[dict]] = defaultdict(list)

for line in Path("logs/inqtrix_20260509_144946.log").read_text().splitlines():
    match = EVENT_LINE.search(line)
    if not match:
        continue
    payload = json.loads(match.group(1))
    if payload.get("event") != "answer_claim_binding":
        continue
    bindings_by_claim[payload.get("claim_id", "")].append(payload)

for claim_id, rows in bindings_by_claim.items():
    matched = sum(1 for r in rows if r["binding_status"] == "matched")
    source_only = sum(1 for r in rows if r["binding_status"] == "source_only_binding")
    print(f"{claim_id}\tmatched={matched}\tsource_only={source_only}")
```

For runs captured via `/v1/test/run` or parity tooling, the structured events are already deserialised under the top-level `iteration_logs` key; iterate the list directly.

## Custom provider provenance

Search providers return a typed `GroundedSearchResult`. Each `GroundedSource`
inside it becomes normalized `source_record` and `provider_citation_record`
events, carrying `origin`, `rank`, `title`, `snippet`, and URL provenance
without exposing raw SDK payloads.

```python
from inqtrix.providers.base import SearchProvider
from inqtrix.search_result import GroundedSearchResult, GroundedSource


class MyCustomSearch(SearchProvider):
    def search(self, query: str, *, recency: str = "", language: str = "") -> GroundedSearchResult:
        raw_results = self._call_backend(query, recency=recency, language=language)

        sources = [
            GroundedSource(
                url=item["url"],
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                rank=index + 1,
                origin="search_result",
            )
            for index, item in enumerate(raw_results)
        ]

        return GroundedSearchResult(
            answer=self._compose_answer(raw_results),
            sources=sources,
        )
```

`runtime_logging.normalize_source_provenance` reads the typed source fields and
drops everything else before logging. See [Writing a custom provider](../providers/writing-a-custom-provider.md) for the full contract.

## Related docs

- [Logging](logging.md)
- [Iteration log](iteration-log.md)
- [Debugging runs](debugging-runs.md)
- [Writing a custom provider](../providers/writing-a-custom-provider.md)
