# AgentConfig

> File: `src/inqtrix/agent.py`

## Scope

`AgentConfig` is the only configuration object `ResearchAgent` consumes. It covers agent behaviour (rounds, confidence, report profile), timeouts, cache settings, and optional provider/strategy injection. Server-only deployment settings stay on `ServerSettings` (see [Settings and env](settings-and-env.md)).

This page describes the Python library config in `src/inqtrix/agent.py`.
HTTP server mode uses the same agent defaults through `Settings` and optional
provider injection in `create_app(...)`.

Current boundary: `AgentConfig` parametrizes the default
`classify -> plan -> search -> evaluate -> answer` algorithm. It does not yet
contain an `algorithm` field and does not select a different graph topology.

## Minimal usage

```python
from inqtrix import AgentConfig, LiteLLM, ReportProfile

AgentConfig(
    llm=LiteLLM(api_key="...", default_model="gpt-4o"),
    report_profile=ReportProfile.DEEP,
    max_rounds=3,
    confidence_stop=7,
    answer_prompt_citations_max=40,
)
```

Model names live on provider constructors, not on `AgentConfig`. Leaving `llm=None` or `search=None` triggers auto-creation from environment variables on the first `research()` call.

## Configuration hierarchy

Precedence, highest wins:

1. Explicit scalar fields on `AgentConfig` — always override everything else.
2. `Settings` (Pydantic, loaded from `.env` and process environment; see [Settings and env](settings-and-env.md)) when providers are auto-created.
3. Code defaults.

Provider and model names live on provider constructors. For complex stacks,
construct providers explicitly and pass them through `AgentConfig(llm=..., search=...)`
or `create_app(providers=...)`.

## How fields affect the procedure

| Field family | Changes control flow? | Effect |
|---|---:|---|
| Providers (`llm`, `search`) | No topology change | Decide which external backends perform LLM and search calls. |
| Strategies | No topology change | Change policy inside nodes: tiering, extraction, consolidation, risk/aspects, stop heuristics. |
| Loop settings (`max_rounds`, `min_rounds`, `confidence_stop`, `first_round_queries`) | Yes | Change when the loop continues, stops, or how broad round 0 is. |
| Citation budget (`answer_prompt_citations_max`) | Indirect | `answer_prompt_citations_max` is the only answer-body citation cap. |
| Timeouts | No | Bound provider calls and whole-run wall time. |
| Model tiers + per-node model overrides | Indirect | Map each node to a high/mid/fast model and reasoning effort. Configured on the provider; see [LLM calls](../architecture/llm-calls.md). |
| Report profile | Indirect | Applies a bundle of answer-depth and evidence-retention defaults unless explicitly overridden. |

## Fields

### Providers and strategies

| Field | Type | Purpose |
|-------|------|---------|
| `llm` | `LLMProvider \| None` | Explicit LLM provider. `None` → auto-create from `Settings`. |
| `search` | `SearchProvider \| None` | Explicit search provider. Same auto-create behaviour. |
| `source_tiering` | `SourceTieringStrategy \| None` | Replace the default tiering. See [Source tiering](../scoring-and-stopping/source-tiering.md). |
| `claim_extraction` | `ClaimExtractionStrategy \| None` | Replace the LLM claim extractor. |
| `claim_consolidation` | `ClaimConsolidationStrategy \| None` | Replace claim dedup / consolidation. |
| `risk_scoring` | `RiskScoringStrategy \| None` | Replace risk scoring and aspect derivation. |
| `stop_criteria` | `StopCriteriaStrategy \| None` | Replace the stop heuristic cascade. |

### Agent behaviour

| Field | Default | Purpose |
|-------|---------|---------|
| `report_profile` | `ReportProfile.COMPACT` | Answer depth: `COMPACT` keeps the concise format; `DEEP` raises evidence context and answer budgets. See [Report profiles](report-profiles.md). |
| `max_rounds` | `2` | Maximum research-loop iterations. |
| `min_rounds` | `1` | Minimum research rounds before an already-triggered stop may be accepted; earlier stops are suppressed unless `max_rounds` is reached. |
| `confidence_stop` | `7` | Stop threshold for `final_confidence` in evaluate. See [Confidence](../scoring-and-stopping/confidence.md). |
| `first_round_queries` | `6` | Query count for round 0. Round 1+ uses `max(6, first_round_queries - 2)` so planning slots and executed search calls stay aligned. |
| `answer_prompt_citations_max` | `60` | Visible upper bound on URLs in the answer prompt. DEEP raises this to `500`; no hidden body cap or citation-block character budget applies afterward. |
| `max_question_length` | `60_000` | Validation guardrail on user input. Generous because the chat composer inlines attached file content into the message. |

### Timeouts

| Field | Default (s) | Purpose |
|-------|-------------|---------|
| `max_total_seconds` | `3600` | Outer active-run budget. See [Timeouts and errors](../observability/timeouts-and-errors.md). |
| `reasoning_timeout` | `600` | Logical reasoning-operation budget, shared by at most three attempts. |
| `editor_assistant_timeout` | `600` | Logical editor-operation budget, independent from research reasoning. |
| `search_timeout` | `600` | Logical search-operation budget, shared by at most three attempts. |
| `claim_extract_timeout` | `600` | Logical claim-extraction budget, shared by at most three attempts. |

### Risk scoring

| Field | Default | Purpose |
|-------|---------|---------|
| `high_risk_score_threshold` | `4` | Risk score at and above which a question is flagged `high_risk`. The flag is an observability signal only (forensic events, `/health`, follow-up preservation); it does not change model selection and drives no query/answer heuristic. Use model tiers or a per-node model override to put demanding questions on a stronger model — see [LLM calls](../architecture/llm-calls.md). |

### Model tier

| Field | Default | Purpose |
|-------|---------|---------|
| `model_tier` | `""` | Per-run model-tier selection (`high` / `mid` / `fast`); replaces the default per-node tier assignment for the run (an explicit per-node model override still wins). The tier *models* themselves are configured on the provider (constructor args / env), not on `AgentConfig` — see [LLM calls](../architecture/llm-calls.md). |

### Search cache

| Field | Default | Purpose |
|-------|---------|---------|
| `search_cache_maxsize` | `256` | LRU capacity. |
| `search_cache_ttl` | `3600` | Entry lifetime in seconds. |

### Convenience

| Field | Default | Purpose |
|-------|---------|---------|
| `testing_mode` | `False` | Exposes `/v1/test/run` in server mode and enables top-level `iteration_logs` export in test/parity payloads. |

## Three common setups

### A: Compact default

```python
agent = ResearchAgent()  # env-based, compact profile
```

### B: Deep report with explicit providers

```python
agent = ResearchAgent(AgentConfig(
    llm=AnthropicLLM(api_key="sk-ant-...", default_model="claude-sonnet-4-6"),
    search=PerplexitySearch(api_key="pplx-..."),
    report_profile=ReportProfile.DEEP,
    max_rounds=4,
    confidence_stop=8,
))
```

### C: Strategy override

Use this when the default algorithm is right but one policy needs to change.
The graph still runs the same five nodes.

```python
from inqtrix import AgentConfig, ResearchAgent, SourceTieringStrategy


class InternalDocsArePrimary(SourceTieringStrategy):
    def tier_for_url(self, url: str) -> str:
        return "primary" if "docs.example.com" in url else "unknown"

    def quality_from_urls(self, urls: list[str]) -> tuple[dict[str, int], float]:
        counts = {"primary": 0, "mainstream": 0, "stakeholder": 0, "unknown": 0, "low": 0}
        for url in urls:
            counts[self.tier_for_url(url)] += 1
        score = counts["primary"] / max(1, len(urls))
        return counts, score


agent = ResearchAgent(AgentConfig(source_tiering=InternalDocsArePrimary()))
```

### D: Strict budget

```python
agent = ResearchAgent(AgentConfig(
    llm=LiteLLM(api_key="sk-...", default_model="gpt-4o-mini"),
    search=PerplexitySearch(api_key="..."),
    max_rounds=2,
    confidence_stop=7,
    max_total_seconds=120,
    reasoning_timeout=45,
    claim_extract_timeout=30,
))
```

## Per-request overrides (HTTP only)

Server callers can override a whitelisted subset of fields per request via `body["agent_overrides"]`:

- `max_rounds`, `min_rounds`
- `confidence_stop`
- `report_profile`
- `max_total_seconds`
- `first_round_queries`
- `skip_search`

Unknown keys return HTTP 400. Provider- and model-level fields are intentionally **not** overridable per request. See [Web server mode](../deployment/webserver-mode.md).

Example:

```json
{
  "model": "research-agent",
  "messages": [{"role": "user", "content": "Explain Retrieval Augmented Generation"}],
  "agent_overrides": {
    "report_profile": "compact",
    "skip_search": true
  }
}
```

`skip_search=true` bypasses the plan/search/evaluate loop for that request and produces an uncited direct LLM answer. It remains available for compatibility with existing callers. New HTTP/API clients should prefer the top-level `mode` field: `mode="direct_llm"` for direct provider chat, or `mode="research"` to force the full graph even when a server default has `skip_search=true`.

## Related docs

- [Settings and env](settings-and-env.md)
- [Report profiles](report-profiles.md)
- [Providers overview](../providers/overview.md)
- [Strategies](../architecture/strategies.md)
