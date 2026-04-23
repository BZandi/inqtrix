# AgentConfig

> File: `src/inqtrix/agent.py`

## Scope

`AgentConfig` is the only configuration object `ResearchAgent` consumes. It covers agent behaviour (rounds, confidence, report profile), timeouts, cache settings, and optional provider/strategy injection. Server-only deployment settings stay on `ServerSettings` (see [Settings and env](settings-and-env.md)).

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
2. `inqtrix.yaml` (if present and in server mode; see [inqtrix.yaml](inqtrix-yaml.md)).
3. `Settings` (Pydantic, loaded from `.env` and process environment; see [Settings and env](settings-and-env.md)).
4. Code defaults.

In library mode, `ResearchAgent()` loads `.env` automatically but does **not** auto-detect `inqtrix.yaml`. Use `load_config(...)` and `create_providers_from_config(...)` if you need YAML routing in a library script (see [inqtrix.yaml](inqtrix-yaml.md)).

## Fields

### Providers and strategies

| Field | Type | Purpose |
|-------|------|---------|
| `llm` | `LLMProvider \| None` | Explicit LLM provider. `None` → auto-create from `Settings`. |
| `search` | `SearchProvider \| None` | Explicit search provider. Same auto-create behaviour. |
| `source_tiering` | `SourceTieringStrategy \| None` | Replace the default tiering. See [Source tiering](../scoring-and-stopping/source-tiering.md). |
| `claim_extraction` | `ClaimExtractionStrategy \| None` | Replace the LLM claim extractor. |
| `claim_consolidation` | `ClaimConsolidationStrategy \| None` | Replace claim dedup / consolidation. |
| `context_pruning` | `ContextPruningStrategy \| None` | Replace context pruning. |
| `risk_scoring` | `RiskScoringStrategy \| None` | Replace risk scoring, aspect derivation, and `site:` injection. |
| `stop_criteria` | `StopCriteriaStrategy \| None` | Replace the stop heuristic cascade. |

### Agent behaviour

| Field | Default | Purpose |
|-------|---------|---------|
| `report_profile` | `ReportProfile.COMPACT` | Answer depth: `COMPACT` keeps the concise format; `DEEP` raises summarize/context/answer budgets. See [Report profiles](report-profiles.md). |
| `max_rounds` | `4` | Maximum research-loop iterations. |
| `min_rounds` | `1` | Minimum research rounds before stop heuristics may trigger. |
| `confidence_stop` | `8` | Integer 1–10; stop threshold for `final_confidence`. See [Confidence](../scoring-and-stopping/confidence.md). |
| `first_round_queries` | `6` | Query count for round 0. Round 1+ uses 2–3 queries. |
| `max_context` | `12` | Maximum retained context blocks across rounds. |
| `answer_prompt_citations_max` | `60` | Upper bound on URLs in the answer prompt. |
| `max_question_length` | `10_000` | Validation guardrail on user input. |

### Timeouts

| Field | Default (s) | Purpose |
|-------|-------------|---------|
| `max_total_seconds` | `300` | Hard wall-clock deadline for the whole run. See [Timeouts and errors](../observability/timeouts-and-errors.md). |
| `reasoning_timeout` | `120` | Per-call LLM budget. |
| `search_timeout` | `60` | Per-call search budget. |
| `summarize_timeout` | `60` | Per-call summarise / claim extraction budget. |

### Risk scoring

| Field | Default | Purpose |
|-------|---------|---------|
| `high_risk_score_threshold` | `4` | Risk score at which classify and evaluate escalate to `reasoning_model`. |
| `high_risk_classify_escalate` | `True` | Toggle classify escalation. |
| `high_risk_evaluate_escalate` | `True` | Toggle evaluate escalation. |
| `enable_de_policy_bias` | `True` | Adds the German-policy regex bonus (`+2`) to the deterministic risk score. |

### Search cache

| Field | Default | Purpose |
|-------|---------|---------|
| `search_cache_maxsize` | `256` | LRU capacity. |
| `search_cache_ttl` | `3600` | Entry lifetime in seconds. |

### Convenience

| Field | Default | Purpose |
|-------|---------|---------|
| `testing_mode` | `False` | Exposes `/v1/test/run`, enables iteration-log export on results. |

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

### C: Strict budget

```python
agent = ResearchAgent(AgentConfig(
    llm=LiteLLM(api_key="sk-...", default_model="gpt-4o-mini"),
    search=PerplexitySearch(api_key="..."),
    max_rounds=2,
    confidence_stop=7,
    max_total_seconds=120,
    reasoning_timeout=45,
    summarize_timeout=30,
))
```

## Per-request overrides (HTTP only)

Server callers can override a whitelisted subset of fields per request via `body["agent_overrides"]`:

- `max_rounds`, `min_rounds`
- `confidence_stop`
- `report_profile`
- `max_total_seconds`
- `first_round_queries`
- `max_context`

Unknown keys return HTTP 400. Provider-, model-, and session-level fields are intentionally **not** overridable per request. See [Web server mode](../deployment/webserver-mode.md).

## Related docs

- [Settings and env](settings-and-env.md)
- [inqtrix.yaml](inqtrix-yaml.md)
- [Report profiles](report-profiles.md)
- [Providers overview](../providers/overview.md)
- [Strategies](../architecture/strategies.md)
