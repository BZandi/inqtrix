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

- Local development — `.env` plus optional `inqtrix.yaml`.
- One-off shell runs — `export` in the terminal for temporary overrides.
- CI/CD — store secrets in the CI secret store and expose them as environment variables in the job.
- Containers and orchestration — inject via Docker Compose env, Kubernetes Secrets, or a cloud secret integration.

Do not commit `.env`, do not put plaintext secrets in YAML, and do not rely on checked-in config files for production credentials.

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
ENABLE_DE_POLICY_BIAS=false
```

`SKIP_SEARCH=true` is usually set per request by the Streamlit UI rather than globally; a global setting turns every auto-created run into direct LLM chat without citations.

## Models

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_MODEL` | `claude-opus-4.6-agent` | Primary LLM for reasoning (legacy LiteLLM-flavoured default; see note below). |
| `SEARCH_MODEL` | `perplexity-sonar-pro-agent` | Web search model (legacy LiteLLM-flavoured default). |
| `CLASSIFY_MODEL` | *(reasoning)* | Optional dedicated classify model. |
| `SUMMARIZE_MODEL` | *(reasoning)* | Optional dedicated summarise model. |
| `EVALUATE_MODEL` | *(reasoning)* | Optional dedicated evaluate model. |

> **Note.** The `ModelSettings()` default of `claude-opus-4.6-agent` is a LiteLLM alias, not a real Anthropic model id. Code paths that read `settings.models.*` on non-LiteLLM stacks would leak this string into the backend (producing 400/404 errors). Current runtime code reads model names constructor-first via `provider.models.effective_*_model` and `resolve_summarize_model(llm, fallback)`. If you write a new endpoint or strategy, follow that rule.

## Server connection (env-only LiteLLM mode)

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://litellm-proxy:4000/v1` | LLM gateway URL. |
| `LITELLM_API_KEY` | `sk-placeholder` | LLM gateway API key. |

These are only relevant for the default `LiteLLM` auto-creation path. Provider-specific stacks (Anthropic, Bedrock, Azure, Perplexity, Brave) read their own env variables in the example scripts; the providers themselves do not (Constructor-First).

## Agent behaviour

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ROUNDS` | `4` | Maximum research-loop iterations. |
| `MIN_ROUNDS` | `1` | Minimum rounds before stop heuristics may trigger. |
| `CONFIDENCE_STOP` | `8` | Confidence threshold (1–10). |
| `MAX_CONTEXT` | `12` | Max context blocks retained across rounds. |
| `REPORT_PROFILE` | `compact` | `compact` or `deep`. |
| `FIRST_ROUND_QUERIES` | `6` | Query count for round 0. |
| `ANSWER_PROMPT_CITATIONS_MAX` | `60` | Max citations in the answer prompt. |
| `MAX_QUESTION_LENGTH` | `10000` | Max input question length (characters). |
| `TESTING_MODE` | `false` | Enable `/v1/test/run` and iteration-log export. |
| `SKIP_SEARCH` | `false` | Bypass plan/search/evaluate and answer directly with the LLM. No citations, `round=0`. |

## Timeouts

| Variable | Default (s) | Description |
|----------|-------------|-------------|
| `MAX_TOTAL_SECONDS` | `300` | Wall-clock deadline for the whole run. |
| `REASONING_TIMEOUT` | `120` | Per-call LLM timeout. |
| `SEARCH_TIMEOUT` | `60` | Per-call search timeout. |
| `SUMMARIZE_TIMEOUT` | `60` | Per-call summarise / claim extraction timeout. |

## Risk scoring

| Variable | Default | Description |
|----------|---------|-------------|
| `HIGH_RISK_SCORE_THRESHOLD` | `4` | Risk score at which classify / evaluate escalate to `reasoning_model`. |
| `HIGH_RISK_CLASSIFY_ESCALATE` | `true` | Toggle classify escalation. |
| `HIGH_RISK_EVALUATE_ESCALATE` | `true` | Toggle evaluate escalation. |
| `ENABLE_DE_POLICY_BIAS` | `true` | Include the German-policy regex bonus in the score. |

## Search cache

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_CACHE_MAXSIZE` | `256` | LRU capacity. |
| `SEARCH_CACHE_TTL` | `3600` | TTL in seconds. |

## HTTP server (`ServerSettings`)

Only relevant when running as a server (`python -m inqtrix` or the `examples/webserver_stacks/*.py` scripts).

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT` | `3` | Max concurrent agent runs; HTTP 429 when saturated. |
| `MAX_MESSAGES_HISTORY` | `20` | Max messages extracted from chat history. |
| `SESSION_TTL_SECONDS` | `1800` | Session TTL (30 minutes). |
| `SESSION_MAX_COUNT` | `20` | Max concurrent sessions (LRU eviction). |
| `SESSION_MAX_CONTEXT_BLOCKS` | `8` | Max context blocks per session. |
| `SESSION_MAX_CLAIM_LEDGER` | `50` | Max claim-ledger entries per session. |

### Opt-in security (all off by default)

| Variable | Purpose |
|----------|---------|
| `INQTRIX_SERVER_TLS_KEYFILE` / `INQTRIX_SERVER_TLS_CERTFILE` | TLS key/cert pair. Both required; partial setup raises a `RuntimeError` (no silent fallback). |
| `INQTRIX_SERVER_API_KEY` | Enables Bearer token auth on `/v1/chat/completions` and `/v1/test/run`. `/health` and `/v1/models` stay public for liveness / discovery. Uses `hmac.compare_digest`. |
| `INQTRIX_SERVER_CORS_ORIGINS` | Comma-list of origins. `*` is allowed but WARNs (browsers reject `*` together with credentials). |

See [Security hardening](../deployment/security-hardening.md).

### Host and port

| Variable | Default | Description |
|----------|---------|-------------|
| `INQTRIX_SERVER_HOST` | `0.0.0.0` | uvicorn bind address. |
| `INQTRIX_SERVER_PORT` | `5100` | uvicorn port. |

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `INQTRIX_LOG_ENABLED` | `false` | Master switch. |
| `INQTRIX_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `INQTRIX_LOG_CONSOLE` | `true` | Mirror to stderr. |
| `INQTRIX_LOG_FILE` | *(unset)* | File sink path. |
| `INQTRIX_LOG_WEB_LEVEL` | `INFO` | Level for uvicorn / FastAPI logs when mirrored via `build_uvicorn_log_config`. |
| `INQTRIX_LOG_INCLUDE_WEB` | `true` | Opt-out for uvicorn mirroring. |

See [Logging](../observability/logging.md).

## Related docs

- [Agent config](agent-config.md)
- [inqtrix.yaml](inqtrix-yaml.md)
- [Report profiles](report-profiles.md)
- [Security hardening](../deployment/security-hardening.md)
- [Streamlit UI](../deployment/streamlit-ui.md)
