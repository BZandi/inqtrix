# Troubleshooting

Symptom → cause → fix matrix. For deeper mechanism descriptions, each row links to the explanatory page.

## Startup and configuration

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ResearchAgent()` raises `RuntimeError` about missing provider credentials | No `.env` loaded and no explicit `AgentConfig(llm=..., search=...)` | Call `load_dotenv()` in your script, or pass providers explicitly. See [Library mode](../deployment/library-mode.md). |
| HTTP server logs are empty, console only | `INQTRIX_LOG_ENABLED=false` (default) | Set `INQTRIX_LOG_ENABLED=true`. Example scripts write a timestamped file under `logs/`. See [Logging](../observability/logging.md). |
| Server starts but `/health` shows wrong `search_model` | Custom `SearchProvider` subclass without an override for the `search_model` property (shows `"<ClassName>(unknown)"`) | Implement the property on the subclass. See [Providers overview](../providers/overview.md). |
| Server starts but `/health` shows LiteLLM default models on a non-LiteLLM stack | Legacy code reading `settings.models.*` instead of the provider | Upgrade; current `/health` reads constructor-first. If writing a new endpoint, use `resolve_claim_extract_model(llm, fallback)`. See [Debugging runs](../observability/debugging-runs.md). |

Quick log setup:

```bash
export INQTRIX_LOG_ENABLED=true
export INQTRIX_LOG_LEVEL=INFO
```

## Azure-specific

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `AzureOpenAIAPIError(status_code=400, ...unsupported parameter...)` | `token_budget_parameter` mismatch for the deployment | Switch between `"max_completion_tokens"` and `"max_tokens"` on the provider constructor. See [Azure OpenAI](../providers/azure-openai.md). |
| `AzureOpenAIAPIError(status_code=404)` on first call | Deployment name incorrect | The `default_model` argument must equal the **deployment name** in your Azure resource, not the underlying model id. |
| Foundry web-search calls succeed for 60 minutes, then 401 | Cached token with <10 s lifetime returned by `ClientSecretCredential` / `DefaultAzureCredential` | Long-running servers: accept occasional transient 401 and let the next request refresh; for sub-minute reliability, restart the container periodically. See [Enterprise Azure](../deployment/enterprise-azure.md). |
| `consume_effort_config_warnings` logs rejection on Anthropic Haiku | Haiku does not accept `effort`; Inqtrix warns loudly but keeps running | Either move the call to a Sonnet/Opus role or drop the `effort` kwarg for the Haiku role. |

## Run behaviour

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Run stops at confidence 8 and never goes higher | Aspect coverage cap is active; at least one required aspect is uncovered | Inspect `required_aspects` vs `uncovered_aspects`; supply or wait for evidence. See [Aspect coverage](../scoring-and-stopping/aspect-coverage.md). |
| Run remains capped because a central claim needs primary support | The provider-grounded results contain no primary-tier source for that claim | Inspect the exact query, provider answer, and returned citations in the evidence Canvas; refine the query toward the issuer, regulator, filing, dataset, or original study. See [Claims](../scoring-and-stopping/claims.md). |
| Run loops past round 3 with no progress | Falsification not yet armed (need `prev_conf > 0` and `prev_conf <= 4`) | Reduce `confidence_stop` or wait; stagnation typically terminates by round 4. See [Falsification](../scoring-and-stopping/falsification.md). |
| `_claim_extraction_fallback` warnings in every run | Claim-extraction model name leaked from `ModelSettings` defaults | Use `resolve_claim_extract_model(llm, fallback)` instead of `settings.models.effective_claim_extract_model`. Current strategies factory already does. |

## HTTP API

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `curl --max-time N` does not actually cancel the run | Cancel is best-effort at node boundaries, not mid-call | Reduce `REASONING_TIMEOUT`; the next node boundary will honour the cancel. See [Web server mode](../deployment/webserver-mode.md). |
| 429 from `/v1/chat/completions` with available capacity | `MAX_CONCURRENT` reached or semaphore leaked | Increase the setting; check the access log for long-running runs. |
| 429 from `/v1/runs` | Native run FIFO queue full (`RUN_QUEUE_MAX_SIZE`) while active runs already equal `MAX_CONCURRENT` | Increase `RUN_QUEUE_MAX_SIZE`, add front-door rate limiting, or wait for queued jobs. See [Run events](../observability/run-events.md). |
| SSE stream ends abruptly | Client closed the connection; watcher task detected disconnect and set `cancel_event` | Reconnect on the client side or disable progress streaming with `"include_progress": false`. |
| CORS request blocked in browser with credentials | `INQTRIX_SERVER_CORS_ORIGINS=*` with credentials enabled — browsers reject wildcard + credentials | List explicit origins. See [Security hardening](../deployment/security-hardening.md). |

Minimal HTTP smoke:

```bash
curl http://localhost:5100/health
curl -N http://localhost:5100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"research-agent","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## Authentication and Stack mode

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Fresh `local` deploy never shows the owner-setup screen (`GET /api/setup/status` returns `needs_owner: false`) | Not in `local` mode, an owner already exists, or the DB is not migrated | Confirm `INQTRIX_AUTH_MODE=local` — the gate is inert in every other mode. On a fresh DB run the `migrate` step first. See [Create and manage users](../how-to/create-and-manage-users.md). |
| `POST /api/setup/owner` returns 409 | The instance owner already exists (the gate is one-shot) | Sign in with `POST /api/auth/login/local` instead; the owner is created exactly once. |
| Login suddenly returns 429, even with the correct password | Login brute-force lockout tripped (`INQTRIX_LOGIN_RATE_LIMIT_*`), keyed per identifier + client IP | Wait `INQTRIX_LOGIN_RATE_LIMIT_LOCKOUT_SECONDS` (60) or raise the thresholds. Behind proxies, make each trusted edge append/overwrite forwarding data and set `INQTRIX_TRUSTED_PROXY_HOPS` to the exact chain depth; the bundled gateway contributes the right-most trusted hop. See [Auth modes](../deployment/auth-modes.md). |
| `ldap` login always 401 | Wrong bind DN/password, search base, or user filter — search-then-bind cannot locate or re-bind the user | Verify `INQTRIX_LDAP_BIND_DN`/`_PASSWORD`, `INQTRIX_LDAP_USER_SEARCH_BASE`, `INQTRIX_LDAP_USER_SEARCH_FILTER` against the directory; the 401 is uniform by design (no enumeration oracle). See [Connect to an existing LDAP](../how-to/connect-to-existing-ldap.md). |
| 401 on a `Bearer ipat_...` request that worked before | Token revoked/expired, or the memory storage backend lost it on restart | PATs persist only with `INQTRIX_STORAGE_BACKEND=postgres`; the memory default WARNs at startup that tokens vanish on restart. Mint a new one under Settings → Account. |
| `docker compose ... up` exits at the `migrate` step, or `api` never turns healthy | Missing value in the selected secret file, a bad visible `INQTRIX_DATABASE_URL`, or a migration error | Render the selected Compose model safely with `inqtrix-deploy config --redact`; this checks the pair contract, while the full placeholder/profile/topology preflight runs before `inqtrix-deploy up` creates containers. Then inspect `docker compose -f deploy/compose/compose.stack.yaml --env-file deploy/.env.stack.secrets --env-file deploy/.env.stack logs postgres migrate api`. Confirm the secret file is mode `0600`, has no `CHANGE_ME` values, and its password resolves through the visible DSN expression. Editing the file alone does not rotate an initialized PostgreSQL role; use the documented `db rotate-password` operation. See [Stack quickstart](../getting-started/stack-quickstart.md). |

## Tests and local development

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `uv run pytest` fails immediately with import errors | Not installed in editable mode | `uv sync --extra dev` or `python -m pip install -e ".[dev]"`. See [Installation](../getting-started/installation.md). |
| Replay tests require API keys | Mis-set `INQTRIX_RECORD_MODE` | Unset it; default is `none` (offline replay). See [Testing strategy](../development/testing-strategy.md). |
| Full suite count differs from a number in older docs | The suite grows as provider and server coverage expands | Trust `uv run pytest tests/ --collect-only -q`, or `python -m pytest tests/ --collect-only -q` in the pip-installed environment, for the current count. |

## Related docs

- [Debugging runs](../observability/debugging-runs.md)
- [FAQ](faq.md)
- [Logging](../observability/logging.md)
- [Timeouts and errors](../observability/timeouts-and-errors.md)
