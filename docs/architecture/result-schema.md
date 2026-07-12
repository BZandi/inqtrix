# Result schema

> File: `result.py`

## Scope

The typed Pydantic model returned by `ResearchAgent.research()`. If you consume Inqtrix as a library or over the HTTP API, this page lists every field, every metric, and the export helper.

## `ResearchResult`

```python
result = agent.research("...")

result.answer                         # str, Markdown-formatted
result.metrics.confidence             # int, 0-10
result.metrics.rounds                 # int
result.metrics.elapsed_seconds        # float
result.metrics.total_citations        # int
result.metrics.total_queries          # int
result.metrics.prompt_tokens          # int
result.metrics.completion_tokens      # int
result.metrics.aspect_coverage        # float, 0.0-1.0
result.metrics.sources.quality_score  # float, 0.0-1.0
result.metrics.claims.quality_score   # float, 0.0-1.0
result.metrics.evidence_contract_status  # "clean" | "needs_review" | "unknown"
result.top_sources                    # list[Source]
result.references                     # list[ReportReference]
result.top_claims                     # list[Claim]
result.execution                      # AgentExecution | None
```

`ResearchResult.from_raw()` bridges the internal state dict (TypedDict) to the typed Pydantic model. You normally do not call it directly — `ResearchAgent.research()` returns the typed object. The native HTTP endpoint `GET /v1/runs/{run_id}/result` returns the same public projection via `ResearchResult.to_export_payload()`.

## Internal-to-public projection

`ResearchResult` is deliberately smaller than `AgentState`. It is safe for
applications and HTTP clients, but it is not a full evidence ledger export.

| Public field | Internal source | Projection rule |
|---|---|---|
| `answer` | `raw["answer"]` / `AgentState.answer` | Markdown string after link sanitation, evidence audit, and footer assembly. |
| `metrics.rounds` | `AgentState.round` | Number of completed search rounds. |
| `metrics.total_queries` | `AgentState.queries` | Length of the deduplicated query list. |
| `metrics.total_citations` | `AgentState.all_citations` | Count of known citation URLs, not count of verified evidence units. |
| `metrics.confidence` | `AgentState.final_confidence` | Evaluator confidence after guardrails and stop heuristics. |
| `metrics.sources` | `source_tier_counts`, `source_quality_score` | Aggregate source inventory quality. |
| `metrics.claims` | `claim_status_counts`, `claim_quality_score` | Aggregate consolidated-claim quality. |
| `metrics.evidence_contract_status` | `answer_claim_bindings`, `answer_evidence_bindings` | `clean` (a cited sentence carries a consolidated claim, nothing cited is unsubstantiated), `needs_review` (claim carried but some citation unsubstantiated), `source_context_only` (sources cited, no claim carried), `algorithm_failed` (synthesis blocked), `unknown` (no audit). |
| `top_sources` | `answer`, `EvidenceOverview.allowed_urls`, `evidence_ledger`, `all_citations`, `source_records` | Capped source URLs ordered by actual answer use first, then visible EvidenceOverview URLs, then remaining report-eligible/discovered citations. Tiers prefer normalized `source_records` and fall back to default tiering. |
| `references` | `AgentState.report_references` | Exact structured counterpart of the rendered `## Referenzen` appendix. It is not capped by `from_raw()` and should be used when a UI must match the report reference list. |
| `top_claims` | `AgentState.consolidated_claims` | First capped claims converted through `Claim.from_consolidated()`. |
| `execution` | `result_state.execution` | Canonical effective Agent Desk route, model/effort, source policy, consent reason, and successful tool-use counters. Present for current agent runs; `None` for non-agent and legacy results. |

For debugging weak answers, inspect the internal `result_state` returned by
`graph.run()` or the observability logs. For application code, prefer the typed
public result.

### `ResearchMetrics`

| Field | Type | Description |
|-------|------|-------------|
| `confidence` | `int` | Final confidence after the evaluate cascade, range 0–10 (`0` means no completed evaluator pass). |
| `rounds` | `int` | Number of `PLAN → SEARCH → EVALUATE` passes that actually executed. |
| `elapsed_seconds` | `float` | Monotonic wall time from `research()` call to final answer. |
| `total_citations` | `int` | Distinct URLs collected across all search rounds. |
| `total_queries` | `int` | Distinct queries executed after dedup. |
| `prompt_tokens` | `int` | Aggregate prompt-token usage across all LLM calls where the provider returned usage metadata. |
| `completion_tokens` | `int` | Aggregate completion-token usage across all LLM calls where the provider returned usage metadata. |
| `aspect_coverage` | `float` | `aspects_covered / total_aspects`, 0.0–1.0. |
| `evidence_consistency` | `int` | Evaluator consistency score, range 0–10. Parse failures inside `evaluate` store `5`; `0` in the exported result means the raw state did not contain the field. |
| `evidence_sufficiency` | `int` | Evaluator sufficiency score, range 0–10. Parse failures inside `evaluate` store `5`; `0` in the exported result means the raw state did not contain the field. |
| `sources` | `SourceMetrics` | Nested Pydantic model with `tier_counts` and `quality_score`. See [Source tiering](../scoring-and-stopping/source-tiering.md). |
| `claims` | `ClaimMetrics` | Nested Pydantic model with `status_counts` and `quality_score`. See [Claims](../scoring-and-stopping/claims.md). |
| `answer_bound_claims_count` | `int` | Claim-level answer bindings that matched a consolidated claim (a cited sentence that plausibly carries it). URL-level matches are tracked separately and do not count here. |
| `unbound_answer_citations_count` | `int` | Answer citations that resolved to no EvidenceRecord (`unknown_citation`). |
| `verified_claims_used_count` | `int` | Verified consolidated claims marked as used in the final answer. |
| `evidence_contract_status` | `str` | `clean`, `needs_review`, `source_context_only`, `algorithm_failed`, or `unknown`, decided by the claim-level binding of cited answer sentences. |

### `Source`

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str` | Canonicalised URL. |
| `tier` | `str` | `primary`, `mainstream`, `stakeholder`, `unknown`, or `low`. |

### `ReportReference`

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | Human-visible label selected for the report appendix, for example `E3`, `1`, or `Quelle 4`. |
| `url` | `str` | Canonicalised URL rendered in the final report's `## Referenzen` section. |
| `tier` | `str` | Source-tier classification using the same tier labels as `Source.tier`. |

Use `references` for exact report/evidence views. `top_sources` remains a ranked
overview capped at 60 sources and may intentionally differ from the complete
report appendix.

### `AgentExecution`

| Field | Type | Description |
|-------|------|-------------|
| `execution_directive` | `str` | `quick_web`, `knowledge_only`, or an empty string for ordinary routing. |
| `effective_mode` | `str` | Algorithm that actually executed, currently `agent_kernel` or `workspace_agent`. |
| `response_form` | `str` | Effective `auto`, `chat`, or `canvas` delivery form. |
| `depth` | `str` | Effective `normal` or `deep` depth. |
| `model` | `str` | Resolved model id; empty when the provider default remains unresolved. |
| `reasoning_effort` | `str` | Resolved effort token; empty when the provider default applies. |
| `source_policy` | `dict[str, str]` | Effective `web` and `knowledge` availability after a one-shot directive. |
| `consent_reason` | `str` | Stable machine reason for permission/approval state. |
| `tool_use_counts` | `dict[str, int]` | Successful source operations keyed by `web` and `knowledge`; zero is explicit. |

Availability and use are intentionally separate: a source can be
`available` while its count remains zero. The complete block is also copied
into state-bearing native-run snapshots for live transparency; see [Run
events](../observability/run-events.md#run-summary).

### `Claim`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Atomic claim statement (post-consolidation). |
| `status` | `str` | `verified`, `contested`, or `unverified`. See [Claims](../scoring-and-stopping/claims.md). |
| `claim_type` | `str` | `fact`, `actor_claim`, or `forecast`. |
| `support_count` | `int` | Number of independent sources that affirm the claim. |
| `contradict_count` | `int` | Number of sources that negate the claim. |
| `needs_primary` | `bool` | Whether the claim needs stronger evidence, usually primary-tier support or independent non-low cross-checking. |
| `status_reason` | `str` | Short reason for the verification status. |
| `source_tier_counts` | `dict[str, int]` | Breakdown by source tier (primary/mainstream/stakeholder/unknown/low). |
| `sources` | `list[str]` | Normalised URLs backing this claim. |

Example public consumption:

```python
result = agent.research("Which companies changed 2026 AI capex guidance?")

for claim in result.top_claims:
    print(claim.status, claim.support_count, claim.text)

if result.metrics.evidence_contract_status != "clean":
    print("Answer evidence audit needs review")
```

## Export helper

`ResearchResult.to_export_payload(options)` produces a lightweight public view, used by parity tooling and UI integrations:

```python
from inqtrix import ResearchResultExportOptions

payload = result.to_export_payload(ResearchResultExportOptions(
    include_sources=False,
    max_claims=5,
))
```

`ResearchResultExportOptions` supports:

- `include_sources: bool` — include the `top_sources` block.
- `include_references: bool` — include the exact `references` block.
- `include_claims: bool` — include the `top_claims` block.
- `max_claims: int | None` — cap the number of claims in the export.
- `max_references: int | None` — cap the number of report references in this export only. `None` keeps every report reference.
- `max_sources: int | None` — cap the number of sources in the export.

When `execution` is present it is always exported as a complete block; the
source/tool transparency contract is not controlled by the evidence-list
include/cap options.

The native run API stores this export payload only for `RUN_COMPLETED_TTL_SECONDS` in process memory. A durable application UI should persist the payload in an application database after reading `/v1/runs/{run_id}/result`.

## Full JSON serialisation

`result.model_dump_json(indent=2)` serialises the complete structure, including every metric. Non-serialisable runtime internals (cancel event, deadline, thread pools) are never part of `ResearchResult`; they live on `AgentState` and are stripped before typing.

## Related docs

- [Public API layer](public-api.md)
- [Evidence pipeline](evidence-pipeline.md)
- [Source tiering](../scoring-and-stopping/source-tiering.md)
- [Claims](../scoring-and-stopping/claims.md)
- [Parity tooling](../development/parity-tooling.md)
- [Run events](../observability/run-events.md)
