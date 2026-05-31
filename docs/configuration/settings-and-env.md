# Settings and environment variables

> Files: `src/inqtrix/settings.py`, `.env.example`

## Scope

The full env-variable reference. `Settings` is a Pydantic `BaseSettings` container that reads process environment variables and optionally a local `.env` file. It feeds auto-created providers and acts as the default source for `AgentConfig` in library mode.

## Configuration sources

Inqtrix accepts environment variables from:

1. Real process environment variables (`export VAR=...`, CI/CD secrets, Docker `-e`, Kubernetes `env:`, cloud secret managers).
2. A local `.env` file for development only.
3. Built-in defaults for non-sensitive values.

When the same variable exists in both process env and `.env`, the process environment wins. That is deliberately the behaviour you want for debugging and CI.

## Deployment guidance

- Local development — `.env`.
- One-off shell runs — `export` in the terminal for temporary overrides.
- CI/CD — store secrets in the CI secret store and expose them as environment variables in the job.
- Containers and orchestration — inject via Docker Compose env, Kubernetes Secrets, or a cloud secret integration.

Do not commit `.env` and do not rely on checked-in config files for production credentials.

Minimal env-only LiteLLM setup:

```dotenv
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_API_KEY=sk-...
REASONING_MODEL=gpt-4o
SEARCH_MODEL=perplexity-sonar-pro-agent
REPORT_PROFILE=compact
```

Direct-chat mode without web search:

```dotenv
SKIP_SEARCH=true
```

`SKIP_SEARCH=true` is usually set per request by the Streamlit UI rather than globally; a global setting turns every auto-created run into direct LLM chat without citations.

## Models

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_MODEL` | `claude-opus-4.6-agent` | Primary LLM for reasoning (legacy LiteLLM-flavoured default; see note below). |
| `SEARCH_MODEL` | `perplexity-sonar-pro-agent` | Web search model (legacy LiteLLM-flavoured default). |
| `CLASSIFY_MODEL` | *(tier/reasoning)* | Optional dedicated classify model (per-node override). |
| `CLAIM_EXTRACT_MODEL` | *(tier/reasoning)* | Optional dedicated claim-extraction model (per-node override). |
| `EVALUATE_MODEL` | *(tier/reasoning)* | Optional dedicated evaluate model (per-node override). |
| `PLAN_MODEL` | *(tier/reasoning)* | Optional dedicated plan model (per-node override). |
| `ANSWER_MODEL` | *(tier/reasoning)* | Optional dedicated answer model (per-node override). |
| `DIRECT_CHAT_MODEL` | *(tier/reasoning)* | Optional dedicated direct-chat model (per-node override). |
| `TIER_HIGH_MODEL` / `TIER_MID_MODEL` / `TIER_FAST_MODEL` | *(reasoning)* | Models for the high / mid / fast tiers. Nodes map to a tier automatically: answer→high, plan/evaluate/direct_chat→mid, classify/claim_extract→fast. |
| `TIER_HIGH_EFFORT` / `TIER_MID_EFFORT` / `TIER_FAST_EFFORT` | *(empty)* | Per-tier reasoning effort: `""` inherit, `none` off, or `minimal`/`low`/`medium`/`high`/`xhigh`. |
| `MODEL_TIER` | *(empty)* | Per-run tier selection (`high`/`mid`/`fast`); replaces the default per-node tier assignment. Also a per-request override. |

> **Tiers.** Resolution order per node: `<node>_model` → `tier_<tier>_model` → `reasoning_model`. Full reference, including the per-provider reasoning-effort mapping, in [LLM calls](../architecture/llm-calls.md).

> **Note.** The `ModelSettings()` default of `claude-opus-4.6-agent` is a LiteLLM alias, not a real Anthropic model id. Code paths that read `settings.models.*` on non-LiteLLM stacks would leak this string into the backend (producing 400/404 errors). Current runtime code reads model names constructor-first via `provider.models.effective_*_model` and `resolve_claim_extract_model(llm, fallback)`. If you write a new endpoint or strategy, follow that rule.

## Server connection (env-only LiteLLM mode)

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://litellm-proxy:4000/v1` | LLM gateway URL. |
| `LITELLM_API_KEY` | `sk-placeholder` | LLM gateway API key. |

These are only relevant for the default `LiteLLM` auto-creation path. Provider-specific stacks (Anthropic, Bedrock, Azure, Perplexity) read their own env variables in the example scripts; the providers themselves do not (Constructor-First).

## Agent behaviour

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ROUNDS` | `4` | Maximum research-loop iterations. |
| `MIN_ROUNDS` | `1` | Minimum rounds before an already-triggered stop may be accepted; earlier stops are suppressed unless `MAX_ROUNDS` is reached. |
| `CONFIDENCE_STOP` | `8` | Confidence stop threshold used by evaluate (`final_confidence >= CONFIDENCE_STOP`). |
| `REPORT_PROFILE` | `compact` | `compact` or `deep`. |
| `FIRST_ROUND_QUERIES` | `6` | Query count for round 0. |
| `ANSWER_PROMPT_CITATIONS_MAX` | `60` | Max citations in the answer prompt. |
| `MAX_QUESTION_LENGTH` | `10000` | Max input question length (characters). |
| `TESTING_MODE` | `false` | Enable `/v1/test/run` and sanitized top-level `iteration_logs` export in test/parity payloads. Never enable in production. With `OBSERVABILITY_PROFILE=forensic`, the export includes forensic events even if file logging is off. |
| `SKIP_SEARCH` | `false` | Bypass plan/search/evaluate and answer directly with the LLM. No citations, `round=0`. |

## Timeouts

| Variable | Default (s) | Description |
|----------|-------------|-------------|
| `MAX_TOTAL_SECONDS` | `300` | Wall-clock deadline for the whole run. |
| `REASONING_TIMEOUT` | `120` | Per-call LLM timeout. |
| `SEARCH_TIMEOUT` | `60` | Per-call search timeout. |
| `CLAIM_EXTRACT_TIMEOUT` | `60` | Per-call claim-extraction timeout. |

## Risk scoring

| Variable | Default | Description |
|----------|---------|-------------|
| `HIGH_RISK_SCORE_THRESHOLD` | `4` | Risk score at/above which a question is flagged `high_risk`. Observability signal only (forensic events, `/health`, follow-up preservation); it does not change model selection (use the model tiers above) and drives no query/answer heuristic. |

## Search cache

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_CACHE_MAXSIZE` | `256` | LRU capacity. |
| `SEARCH_CACHE_TTL` | `3600` | TTL in seconds. |

## HTTP server (`ServerSettings`)

Only relevant when running as a server (`python -m inqtrix` or the `examples/webserver_stacks/*.py` scripts).

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT` | `3` | Max concurrent `/v1/chat/completions` requests. The native run queue uses this value only when `RUN_MAX_CONCURRENT` is unset. |
| `RUN_MAX_CONCURRENT` | *(unset)* | Optional active-worker cap for native `/v1/runs`. When unset, native runs reuse `MAX_CONCURRENT`; queued jobs are still controlled separately by `RUN_QUEUE_MAX_SIZE`. |
| `RUN_QUEUE_MAX_SIZE` | `50` | Max native `/v1/runs` jobs waiting in memory. Active jobs do not count against this limit. |
| `RUN_COMPLETED_TTL_SECONDS` | `300` | How long completed, failed, or cancelled native run records and buffered events remain queryable in memory. |
| `RUN_EVENT_BUFFER_SIZE` | `200` | Recent structured events retained per native run for late SSE subscribers. |
| `MAX_MESSAGES_HISTORY` | `20` | Max messages extracted from chat history. |

### Opt-in security (all off by default)

| Variable | Purpose |
|----------|---------|
| `INQTRIX_SERVER_TLS_KEYFILE` / `INQTRIX_SERVER_TLS_CERTFILE` | TLS key/cert pair. Both required; partial setup raises a `RuntimeError` (no silent fallback). |
| `INQTRIX_SERVER_API_KEY` | Enables Bearer token auth on `/v1/chat/completions`, `/v1/runs*`, and `/v1/test/run`. `/health` and `/v1/models` stay public for liveness / discovery. Uses `hmac.compare_digest`. |
| `INQTRIX_SERVER_CORS_ORIGINS` | Comma-list of origins. `*` is allowed but WARNs (browsers reject `*` together with credentials). |

See [Security hardening](../deployment/security-hardening.md).

## Streamlit UI connection

Only relevant for `webapp.py` (Streamlit frontend).

| Variable | Default | Description |
|----------|---------|-------------|
| `INQTRIX_WEBAPP_BASE_URL` | *(unset)* | Server URL used by the Streamlit UI (for example `http://localhost:5100` or `http://192.168.1.42:5100`). When unset, the UI waits for manual URL input in the sidebar and does not auto-probe localhost. |
| `INQTRIX_WEBAPP_API_KEY` | *(unset)* | Optional bearer token prefill for the Streamlit UI. Sent as `Authorization: Bearer ...` on chat calls. |

### Host and port

| Variable | Default | Description |
|----------|---------|-------------|
| `INQTRIX_SERVER_HOST` | `0.0.0.0` | uvicorn bind address. |
| `INQTRIX_SERVER_PORT` | `5100` | uvicorn port. |

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `INQTRIX_LOG_ENABLED` | `false` | Master switch for persistent file logs under `logs/`. |
| `INQTRIX_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Structured runtime `EVENT` lines are emitted at `DEBUG`. |
| `INQTRIX_LOG_CONSOLE` | `false` | Mirror WARNING+ records from the `inqtrix` logger to stderr. |
| `INQTRIX_LOG_WEB_LEVEL` | `INFO` | Level for uvicorn / FastAPI logs when mirrored via `build_uvicorn_log_config`. |
| `INQTRIX_LOG_INCLUDE_WEB` | `true` | Opt-out for uvicorn mirroring. |
| `OBSERVABILITY_PROFILE` | `summary` | `summary`, `debug`, or `forensic`. `forensic` produces full source/citation/claim/answer lineage; pair it with `INQTRIX_LOG_ENABLED=true` and `INQTRIX_LOG_LEVEL=DEBUG` to see those events in the file log. `debug` is currently reserved for future mid-level detail. |

For file output, the switches compose as follows:

- Summary file log: `INQTRIX_LOG_ENABLED=true` + `INQTRIX_LOG_LEVEL=INFO`.
- Algorithm trace: `INQTRIX_LOG_ENABLED=true` + `INQTRIX_LOG_LEVEL=DEBUG`.
- Forensic file log: `INQTRIX_LOG_ENABLED=true` + `INQTRIX_LOG_LEVEL=DEBUG` + `OBSERVABILITY_PROFILE=forensic`.
- Exported iteration logs for parity/testing: `TESTING_MODE=true` (optionally plus `OBSERVABILITY_PROFILE=forensic`).

See [Logging](../observability/logging.md) and [Forensic cookbook](../observability/forensic-cookbook.md).

## Related docs

- [Agent config](agent-config.md)
- [Report profiles](report-profiles.md)
- [Security hardening](../deployment/security-hardening.md)
- [Streamlit UI](../deployment/streamlit-ui.md)
