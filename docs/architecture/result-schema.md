# Result schema

> File: `result.py`

## Scope

The typed Pydantic model returned by `ResearchAgent.research()`. If you consume Inqtrix as a library or over the HTTP API, this page lists every field, every metric, and the export helper.

## `ResearchResult`

```python
result = agent.research("...")

result.answer                         # str, Markdown-formatted
result.metrics.confidence             # int, 1-10
result.metrics.rounds                 # int
result.metrics.elapsed_seconds        # float
result.metrics.total_citations        # int
result.metrics.total_queries          # int
result.metrics.aspect_coverage        # float, 0.0-1.0
result.metrics.sources.quality_score  # float, 0.0-1.0
result.metrics.claims.quality_score   # float, 0.0-1.0
result.top_sources                    # list[Source]
result.top_claims                     # list[Claim]
```

`ResearchResult.from_raw()` bridges the internal state dict (TypedDict) to the typed Pydantic model. You normally do not call it directly — `ResearchAgent.research()` returns the typed object.

### `ResearchMetrics`

| Field | Type | Description |
|-------|------|-------------|
| `confidence` | `int` | Final LLM-reported confidence capped by the heuristic cascade, range 1–10. |
| `rounds` | `int` | Number of `PLAN → SEARCH → EVALUATE` passes that actually executed. |
| `elapsed_seconds` | `float` | Monotonic wall time from `research()` call to final answer. |
| `total_citations` | `int` | Distinct URLs collected across all search rounds. |
| `total_queries` | `int` | Distinct queries executed after dedup. |
| `aspect_coverage` | `float` | `aspects_covered / total_aspects`, 0.0–1.0. |
| `sources` | `SourcesMetrics` | Nested: `tier_counts`, `quality_score`. See [Source tiering](../scoring-and-stopping/source-tiering.md). |
| `claims` | `ClaimsMetrics` | Nested: `status_counts`, `quality_score`. See [Claims](../scoring-and-stopping/claims.md). |

### `Source`

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str` | Canonicalised URL. |
| `tier` | `str` | `primary`, `mainstream`, `stakeholder`, `unknown`, or `low`. |
| `support_count` | `int` | How many claims reference this URL as an evidence source. |

### `Claim`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Atomic claim statement (post-consolidation). |
| `status` | `str` | `verified`, `contested`, or `unverified`. See [Claims](../scoring-and-stopping/claims.md). |
| `support` | `int` | Number of independent sources that affirm the claim. |
| `contradict` | `int` | Number of sources that negate the claim. |
| `needs_primary` | `bool` | Whether the claim requires a primary-tier source for verification. |
| `primary_tier_count` | `int` | Breakdown by source tier (primary/mainstream/stakeholder/unknown/low). |
| `source_urls` | `list[str]` | Normalised URLs backing this claim. |

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
- `include_claims: bool` — include the `top_claims` block.
- `max_claims: int | None` — cap the number of claims in the export.
- `max_sources: int | None` — cap the number of sources in the export.

## Full JSON serialisation

`result.model_dump_json(indent=2)` serialises the complete structure, including every metric. Non-serialisable runtime internals (cancel event, deadline, thread pools) are never part of `ResearchResult`; they live on `AgentState` and are stripped before typing.

## Related docs

- [Public API layer](public-api.md)
- [Source tiering](../scoring-and-stopping/source-tiering.md)
- [Claims](../scoring-and-stopping/claims.md)
- [Parity tooling](../development/parity-tooling.md)
