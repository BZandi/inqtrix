# Settings and environment variables

> Files: `src/inqtrix/settings.py`, `src/inqtrix/server/container.py`, `.env.example`

## Scope

This page is the single source of truth for environment variables: it lists **every** variable the code reads. `Settings` is a Pydantic `BaseSettings` container with ten groups — Models, Providers, Agent, Server, Auth, Storage, Queue, Knowledge, Quota, Sharing — each loaded from process environment variables and optionally a local `.env` file. The groups are the only env-coupled surface: the providers and stores they configure receive every value via constructor arguments (Constructor-First). Two further categories sit outside `Settings` and are documented at the end of this page: [process-level variables](#process-level-variables-outside-settings) read by the server/worker bootstrap, and [development and test-only variables](#development-and-test-only-variables) read by scripts, the eval harness, and the test suite.

Deep-dive pages (provider recipes, auth modes, logging, knowledge profiles) cover usage and walkthroughs; when they name a variable they link back here for the authoritative definition. The committed `.env.example` and `deploy/.env.stack.example` templates are practical starting points, not the reference — this page is.

## Configuration sources

1. Real process environment variables (`export VAR=...`, CI/CD secrets, Docker `-e`, Kubernetes `env:`).
2. A local `.env` file for development only.
3. Built-in defaults for non-sensitive values.

When the same variable exists in both process env and `.env`, the process environment wins. Do not commit `.env` and do not rely on checked-in config files for production credentials.

Minimal env-only LiteLLM setup:

```dotenv
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_API_KEY=sk-...
REASONING_MODEL=gpt-4o
SEARCH_MODEL=perplexity-sonar-pro-agent
REPORT_PROFILE=compact
```

## Models (`ModelSettings`)

Routes each agent role (classify, plan, search, evaluate, answer, claim extraction, direct chat) to a model name. Resolution order per node: `<node>_model` -> `tier_<tier>_model` -> `reasoning_model`. Default node-to-tier mapping: answer -> high; plan/evaluate/direct_chat -> mid; classify/claim_extract -> fast. Full reference, including the per-provider reasoning-effort mapping, in [LLM calls](../architecture/llm-calls.md).

| Variable | Default | Effect |
|----------|---------|--------|
| `REASONING_MODEL` | `claude-opus-4.6-agent` | Primary reasoning model and fallback for every unset role. Format is provider-dependent: OpenAI/LiteLLM model id, Anthropic model name, or Azure deployment name. |
| `SEARCH_MODEL` | *(empty)* | Model called by `PerplexitySearch` or a LiteLLM-routed search adapter. Ignored by non-LLM search providers. |
| `TIER_HIGH_MODEL` / `TIER_MID_MODEL` / `TIER_FAST_MODEL` | *(reasoning)* | Models for the high / mid / fast tiers. Empty falls back to `REASONING_MODEL`. |
| `TIER_HIGH_EFFORT` / `TIER_MID_EFFORT` / `TIER_FAST_EFFORT` | *(empty)* | Per-tier reasoning effort: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`. Empty inherits the provider constructor default. For Anthropic any non-empty/non-`none` value turns on adaptive thinking. |
| `CLASSIFY_MODEL`, `CLAIM_EXTRACT_MODEL`, `EVALUATE_MODEL`, `PLAN_MODEL`, `ANSWER_MODEL`, `DIRECT_CHAT_MODEL` | *(tier/reasoning)* | Per-node overrides; an explicit per-node model always wins over the tier. `CLAIM_EXTRACT_MODEL` is the highest-volume role (one call per search hit) and the largest cost lever. |

> **Note.** The `REASONING_MODEL` default is a LiteLLM alias, not a real Anthropic model id. Runtime code reads model names constructor-first via `provider.models.effective_*_model`; new endpoints and strategies must follow that rule rather than reading `settings.models.*` directly.

**Interactions.** The per-request routing overrides `MODEL_TIER`, `MODEL_OVERRIDE`, and `EFFORT_OVERRIDE` live in the Agent block (below) but act on this block's tier router. The Knowledge block adds two fast-tier call sites (`knowledge_rerank`, `knowledge_decompose`) that resolve through the same tiers.

**When this block is unset.** Every role runs on `REASONING_MODEL`. Functional, but the high-volume claim-extraction role then pays frontier-model cost; setting `TIER_FAST_MODEL` is the standard optimisation.

## Providers (`ProviderSettings`)

Selects WHICH provider builds each axis of the auto-created server stack, plus the per-provider construction knobs that have no home in the Models block. Two independent axes (any LLM pairs with any search backend); unset reproduces the byte-identical LiteLLM + Perplexity default. Per-combination `.env` recipes live in [Provider recipes](../getting-started/provider-recipes.md) — this block is the variable reference.

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_LLM_PROVIDER` | `litellm` | `litellm`, `anthropic`, `azure`, or `bedrock`. An unknown value fails at `Settings()` construction; a missing required credential for the chosen provider fails loudly at startup — never a silent fallback to litellm. |
| `INQTRIX_SEARCH_PROVIDER` | `perplexity` | `perplexity` or `azure_foundry`. Independent of the LLM axis. |
| `INQTRIX_SELECTABLE_CHAT_MODELS` | *(empty)* | Comma-separated chat models offered in the UI model picker (feeds `/health.models_catalog`). Empty offers only the resolved default model. |
| `INQTRIX_TEMPERATURE` | *(unset)* | Sampling temperature for the `anthropic`/`azure`/`bedrock` providers. `litellm` ignores it and logs a visible warning when set. |
| `INQTRIX_TOKEN_BUDGET_PARAMETER` | *(empty)* | Output-budget request field for `litellm`/`azure`: `max_tokens` or `max_completion_tokens` (the latter for OpenAI o-series). Anthropic/Bedrock ignore it (logged when set). |
| `INQTRIX_SEARCH_PRESET` | *(empty → `fast-search`)* | Perplexity agent preset: `fast-search`, `pro-search`, or `deep-research`. Ignored by `azure_foundry` (logged when set). Web search has no high/mid/fast tiers — this preset plus `SEARCH_MODEL` are the search knobs. |
| `INQTRIX_SEARCH_INSTRUCTIONS` | *(empty)* | Optional system instructions for the Perplexity search agent; ignored by `azure_foundry`. |

Further tuning (the chosen provider's credentials): `anthropic` reads `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`; `azure` reads `AZURE_OPENAI_ENDPOINT` plus `AZURE_OPENAI_API_KEY` or the `AZURE_TENANT_ID`/`AZURE_CLIENT_ID`/`AZURE_CLIENT_SECRET` service-principal trio; `bedrock` reads `AWS_REGION` (default `eu-central-1`) / `AWS_PROFILE`; `azure_foundry` search reads `AZURE_AI_PROJECT_ENDPOINT` plus `WEB_SEARCH_AGENT_NAME` (and the optional `WEB_SEARCH_AGENT_VERSION` pin), authenticating with `AZURE_AI_PROJECT_API_KEY` or the same `AZURE_TENANT_ID`/`AZURE_CLIENT_ID`/`AZURE_CLIENT_SECRET` trio. Per-tier reasoning effort and tier models stay in the Models block (`TIER_*_EFFORT`, `TIER_*_MODEL`); the selector passes them to whichever provider is chosen. Context window and output budget are NOT env vars — the model cards (`src/inqtrix/model_cards.py`) are the authoritative source. Inqtrix passes provider credentials explicitly (constructor-first) and does not rely on the underlying SDKs' own credential env vars (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `PPLX_API_KEY`); use `LITELLM_API_KEY`/`LITELLM_BASE_URL` and `PERPLEXITY_API_KEY` instead. The one exception is Bedrock, which uses boto3's standard AWS chain (`AWS_ACCESS_KEY_ID` etc.) when `AWS_PROFILE` is not set.

**Interactions.** Unset selectors reproduce the legacy `LiteLLM` + `PerplexitySearch` bodies exactly; there is deliberately no inference from which keys are present (that would be a silent fallback). A behavioural knob set for a provider that ignores it (e.g. `INQTRIX_TEMPERATURE` with `litellm`) is logged, never silently dropped.

**When this block is unset.** The server builds `LiteLLM` for reasoning and `PerplexitySearch` for search from the Server block's gateway/keys, identical to pre-selector behaviour.

## Agent (`AgentSettings`)

Tunes a single research run: loop bounds, stop thresholds, timeouts, input limits, and the search cache. `REPORT_PROFILE` doubles as a preset trigger — assigning `compact` or `deep` applies a bundle of overrides for any field not explicitly set, so explicit env values always win over the profile.

| Variable | Default | Effect |
|----------|---------|--------|
| `REPORT_PROFILE` | `compact` | `compact` or `deep`. `compact` presets `max_rounds=2`, `min_rounds=1`, `confidence_stop=7`, `first_round_queries=6`; `deep` presets `max_rounds=4`, `min_rounds=2`, `confidence_stop=8`, `first_round_queries=10`, `answer_prompt_citations_max=500`, `reasoning_timeout=900`, `editor_assistant_timeout=900`, `claim_extract_timeout=600`, `search_timeout=300`, `max_total_seconds=1800`. See [Report profiles](report-profiles.md). |
| `MAX_ROUNDS` | `2` | Hard upper bound for the research loop (`4` under DEEP). |
| `MIN_ROUNDS` | `1` | Suppresses early stops until this round; clamped to `MAX_ROUNDS` at request time. |
| `CONFIDENCE_STOP` | `7` | Minimum evaluator confidence (1-10) at which the stop cascade may emit `done` (`8` under DEEP). |
| `MAX_TOTAL_SECONDS` | `300` | Wall-clock deadline for the whole run, checked at node boundaries; also the base for the chat and SSE HTTP wait (this value + `REQUEST_WAIT_MARGIN_SECONDS`). `1800` under DEEP. |
| `REASONING_TIMEOUT` | `120` | Per-call timeout (seconds) for reasoning LLM calls (classify/plan/evaluate/answer). `900` under DEEP. |
| `EDITOR_ASSISTANT_TIMEOUT` | `120` | Per-call timeout (seconds) for editor suggest/instruct calls, decoupled from `REASONING_TIMEOUT`. Raise it if long editor instructions with large attachments hit a 504, without lengthening research reasoning calls. `900` under DEEP. |
| `SEARCH_TIMEOUT` | `60` | Per-call timeout (seconds) for search-provider calls; keep below `MAX_TOTAL_SECONDS / FIRST_ROUND_QUERIES` so one slow query cannot consume the run budget. `300` under DEEP. |
| `CLAIM_EXTRACT_TIMEOUT` | `60` | Per-call timeout (seconds) for claim-extraction calls (one per search hit) and the `/v1/text` improvement call. `600` under DEEP. |
| `SKIP_SEARCH` | `false` | Bypass plan/search/evaluate and answer directly from the LLM with conversation history. No citations, `round` stays `0`. |
| `TESTING_MODE` | `false` | Expose `/v1/test/run` (used by `inqtrix-parity run`). Never enable in production: no rate limiting, returns full iteration logs. |
| `OBSERVABILITY_PROFILE` | `summary` | `summary`, `debug` (reserved), or `forensic` (full source/citation/claim/stop/answer lineage events). See [Logging](../observability/logging.md). |

Further tuning: `FIRST_ROUND_QUERIES` (6), `ANSWER_PROMPT_CITATIONS_MAX` (60), `REQUIRED_CONTEXT_WINDOW_TOKENS` (128000), `MAX_QUESTION_LENGTH` (60000 — generous because the chat composer inlines attached file content), `HIGH_RISK_SCORE_THRESHOLD` (4 — observability signal only), `SEARCH_CACHE_MAXSIZE` (256; `0` disables), `SEARCH_CACHE_TTL` (3600), `MODEL_TIER` / `MODEL_OVERRIDE` / `EFFORT_OVERRIDE` (per-request model routing, normally sent by API clients rather than set globally). The per-call/run timeouts and how the HTTP and client waits derive from them are detailed under "Timeout dependency chain" below.

**Interactions.** `REPORT_PROFILE` rewrites the other loop fields unless they were set explicitly. `MODEL_TIER` replaces the default node-to-tier mapping for every LLM call site of a run; `MODEL_OVERRIDE`/`EFFORT_OVERRIDE` bypass tier routing for the direct-chat call only.

**Timeout dependency chain.** Three layers bound any single operation, and the same outer-wait shape deliberately hangs off different base budgets depending on the endpoint — making that explicit is the point of this section. Every per-call timeout is additionally clamped to the time left in `MAX_TOTAL_SECONDS` (`_bounded_timeout`), so no single call can outlive the run.

```text
Base per-call / run budgets (env-configurable)
  MAX_TOTAL_SECONDS ........ whole-run wall clock
  REASONING_TIMEOUT ........ one reasoning call (classify/plan/evaluate/answer)
  EDITOR_ASSISTANT_TIMEOUT . one editor suggest/instruct call
                             (defaults to the REASONING_TIMEOUT default)
  SEARCH_TIMEOUT ........... one search-provider call
  CLAIM_EXTRACT_TIMEOUT .... one claim-extraction call; the /v1/text call

Derived HTTP waits  (inner budget + REQUEST_WAIT_MARGIN_SECONDS, 30s; each is
                     also capped at the chat wait, so none outlives the run)
  chat + SSE streaming     <- MAX_TOTAL_SECONDS
  editor suggest/instruct  <- EDITOR_ASSISTANT_TIMEOUT  (NOT CLAIM_EXTRACT_TIMEOUT:
                                                        editor work is a full
                                                        generation, not a tight hit)
  /v1/text improvement     <- CLAIM_EXTRACT_TIMEOUT

Client abort timeouts (browser)
  editor AbortController   = server editor wait + client margin
  chat-chain step abort    = server chat  wait  + client margin
  Both are DISCOVERED from GET /v1/capabilities ("timeouts" block), never
  hardcoded, so raising a server budget lengthens the client abort in step.
```

`REQUEST_WAIT_MARGIN_SECONDS` (30s, an internal constant in `services/request_parsing.py`, not an env var) is the grace each HTTP wait adds over its inner per-call budget so the inner call raises its specific provider error before the outer `asyncio.wait_for` fires a generic 504.

**When this block is unset.** There is no off switch — unset means the COMPACT defaults. The one degradation toggle is `SKIP_SEARCH=true`, which turns every run into a direct LLM answer without web research or citations; UI clients normally send it per request instead of setting it globally.

## Server (`ServerSettings`)

Configures the FastAPI surface started by `python -m inqtrix`: the upstream LLM gateway for auto-created providers, request concurrency, native-run retention, and transport hardening. Library-mode users (instantiating `ResearchAgent` directly) can ignore this group entirely.

| Variable | Default | Effect |
|----------|---------|--------|
| `LITELLM_BASE_URL` | `http://litellm-proxy:4000/v1` | OpenAI-compatible gateway for the auto-created `LiteLLM` provider. Must include `/v1`. |
| `LITELLM_API_KEY` | `sk-placeholder` | Bearer key for the gateway. The obvious placeholder makes misconfiguration fail loudly on the first upstream call. |
| `PERPLEXITY_API_KEY` | *(empty)* | Key for the auto-created `PerplexitySearch` provider (its own endpoint, not the gateway). |
| `MAX_CONCURRENT` | `6` | Concurrent `/v1/chat/completions` requests. Native runs reuse this unless `RUN_MAX_CONCURRENT` is set. Each active run holds one thread (the research graph is synchronous), so this also bounds the in-process thread pool; the real ceiling is the provider rate limit + host CPU/RAM. |
| `RUN_MAX_CONCURRENT` | *(unset)* | Optional separate active-job cap for native `/v1/runs`. |
| `RUN_QUEUE_MAX_SIZE` | `50` | Waiting native runs; a full queue returns HTTP 429 on `POST /v1/runs`. |
| `INQTRIX_SERVER_API_KEY` | *(empty)* | Static Bearer gate on chat, text-improvement, test-run, and native run routes (`hmac.compare_digest`). `/health` and `/v1/models` stay public. Also drives auth-mode inference (see Auth). |
| `INQTRIX_SERVER_CORS_ORIGINS` | *(empty)* | Comma-list of allowed origins; installs `CORSMiddleware` with credentials. `*` is accepted but WARNs (browsers reject wildcard with credentials). |
| `INQTRIX_ENABLE_OPENAPI` | `false` | Serve `/openapi.json`, `/docs`, `/redoc`. Documentation routes only; never changes API behaviour. |
| `INQTRIX_PUBLIC_BASE_URL` | *(empty)* | Externally reachable base URL. Set: knowledge citations become clickable `/v1/sources/...` links. Empty: citations keep the internal `inqtrix://` scheme — a visible degradation, never a guessed hostname. |
| `INQTRIX_MAX_TOTAL_INPUT_TOKENS` | `500000` | Approximate-token DoS cap on `question` + `messages[]` (estimated `len(text) // 4`). |

Further tuning: `RUN_COMPLETED_TTL_SECONDS` (300 — how long finished native runs stay queryable in memory), `RUN_EVENT_BUFFER_SIZE` (200 — replay buffer for late SSE subscribers), `MAX_MESSAGES_HISTORY` (20), `INQTRIX_MAX_MESSAGE_COUNT` (200 — HTTP 413 above), `PERPLEXITY_BASE_URL`, `INQTRIX_SERVER_TLS_KEYFILE` / `INQTRIX_SERVER_TLS_CERTFILE` (both or neither; partial setup raises `RuntimeError`). See [Security hardening](../deployment/security-hardening.md).

**Interactions.** `INQTRIX_SERVER_API_KEY` is the inference input for `INQTRIX_AUTH_MODE=infer`. `INQTRIX_PUBLIC_BASE_URL` feeds two other blocks: knowledge citation links and the derived OIDC callback URL. `LITELLM_BASE_URL`/`LITELLM_API_KEY` are reused as the default embedding endpoint by the Knowledge block.

**When this block is OFF.** In pure library mode the whole group is ignored. With everything at defaults the server runs open (no auth gate, no CORS headers, no TLS, no OpenAPI schema) against a placeholder gateway that fails on the first LLM call.

## Auth (`AuthSettings`)

Selects how every HTTP request resolves to a `Principal`. Five modes exist: `none` (anonymous principal), `apikey` (static Bearer gate, byte-identical to the legacy behaviour), `oidc` (browser session via authorization code + PKCE through a backend-for-frontend; Dex is the reference IdP in the dev compose stack), `local` (native email/password accounts with a first-run owner setup — the **default**), and `ldap` (search-then-bind against an existing directory). The three cookie-session modes (`oidc`/`local`/`ldap`) share the same session/CSRF/PAT machinery; `local`/`ldap` ride synthetic issuers and so reuse it verbatim. Inqtrix speaks only generic OIDC for `oidc` — discovery plus claim mapping, no hardwired provider. Per-mode walkthroughs: [Auth modes](../deployment/auth-modes.md), [Create and manage users](../how-to/create-and-manage-users.md), [Connect to an existing LDAP](../how-to/connect-to-existing-ldap.md).

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_AUTH_MODE` | `infer` | `none`, `apikey`, `oidc`, `local`, `ldap`, or `infer`. `infer` derives the mode for backwards compatibility: non-empty `INQTRIX_SERVER_API_KEY` means `apikey`, empty means `none`. An explicit value wins over inference. |
| `INQTRIX_OIDC_ISSUER` | *(empty)* | Issuer URL; discovery comes from `{issuer}/.well-known/openid-configuration` and must echo this exact value. Required for `oidc`. |
| `INQTRIX_OIDC_CLIENT_ID` / `INQTRIX_OIDC_CLIENT_SECRET` | *(empty)* | Confidential client registered at the IdP; tokens never reach the browser. Required for `oidc`. |
| `INQTRIX_SESSION_SECRET` | *(empty)* | Server-side secret for CSRF-token derivation (signed double-submit). Required for `oidc`, `local`, and `ldap`. |
| `INQTRIX_PAT_PEPPER` | *(empty)* | HMAC pepper for personal access tokens. Required for `oidc`, `local`, and `ldap`; rotation invalidates every issued token. |
| `INQTRIX_REGISTRATION` | `open` | First-login admission: `invite` requires a matching open invitation (and the postgres storage backend); `open` keeps the historical behaviour. |
| `INQTRIX_PAT_MAX_PER_USER` | `10` | Active-token cap per user (sprawl guardrail). |
| `INQTRIX_PAT_DEFAULT_TTL_DAYS` | `0` | Default token lifetime when no explicit expiry is given; `0` = non-expiring. |
| `INQTRIX_OIDC_REDIRECT_URL` | *(derived)* | Callback URL registered at the IdP (byte-for-byte match). Empty derives `{INQTRIX_PUBLIC_BASE_URL}/api/auth/callback`. |
| `INQTRIX_OIDC_ALLOWED_GROUPS` | *(empty)* | Comma-separated group allowlist; non-matching logins get a visible, audit-logged 403 at the callback. Empty admits every authenticated user. |
| `INQTRIX_SESSION_MAX_AGE_SECONDS` | `28800` | Absolute session lifetime (8 h); expiry resolves to 401 and the SPA re-runs the login redirect. |
| `INQTRIX_LOCAL_REGISTRATION` | `closed` | `local` only: `closed` (owner + admin-created accounts) or `open` (mounts a public self-signup route, logged loudly). |
| `INQTRIX_LDAP_URL` | *(empty)* | `ldap` bind target, e.g. `ldaps://ldap.example.com:636`. With `INQTRIX_LDAP_BIND_DN` / `INQTRIX_LDAP_BIND_PASSWORD` and `INQTRIX_LDAP_USER_SEARCH_BASE` it forms the search-then-bind core (all required for `ldap`). Attribute and TLS knobs are in the LDAP further-tuning note below; the admin-group knob is `INQTRIX_LDAP_ADMIN_GROUP_DN` (next rows). |
| `INQTRIX_OIDC_ADMIN_ROLES` / `INQTRIX_OIDC_ADMIN_GROUPS` | *(empty)* | Comma-separated role/group claim values that grant instance-admin on `oidc` login (grant-only — a non-match never demotes). The `ldap` analogue is `INQTRIX_LDAP_ADMIN_GROUP_DN`. |
| `INQTRIX_LOGIN_RATE_LIMIT_ENABLED` | `true` | Login brute-force throttle for `local`/`ldap`, keyed per identifier + client IP (sliding window + lockout). Tune with `INQTRIX_LOGIN_RATE_LIMIT_MAX_ATTEMPTS` (10), `INQTRIX_LOGIN_RATE_LIMIT_WINDOW_SECONDS` (300), `INQTRIX_LOGIN_RATE_LIMIT_LOCKOUT_SECONDS` (60). |

Further tuning (IdP exchangeability): `INQTRIX_OIDC_SCOPES` (`openid profile email`; add `groups` for Okta/Dex), `INQTRIX_OIDC_USERNAME_CLAIM` (`preferred_username`, dot paths descend into nested claims), `INQTRIX_OIDC_EMAIL_CLAIM` (`email`), `INQTRIX_OIDC_GROUPS_CLAIM` (`groups`), `INQTRIX_OIDC_ROLES_CLAIM` (`roles`; the claim used for admin elevation, dot paths supported), `INQTRIX_OIDC_ALLOWED_DOMAINS` (*(empty)*; comma-separated email-domain allowlist orthogonal to the group allowlist — a login without a listed email domain gets a visible 403, and a login without any email is rejected fail-closed), `INQTRIX_OIDC_CLAIM_SEPARATORS` (`" ,"`; characters a string-valued group/role claim is split on, a JSON array is used as-is), `INQTRIX_OIDC_GROUPS_STRIP_PATH_PREFIX` (`false`; strip a single leading `/` from Keycloak full-path groups), `INQTRIX_OIDC_PROVIDER_NAME` (*(empty)*; SSO login-button label surfaced by the auth-config endpoint), `INQTRIX_OIDC_SKIP_EMAIL_VERIFIED` (`false`; required for Entra ID), `INQTRIX_OIDC_DISCOVERY_URL`, `INQTRIX_OIDC_USERINFO_FALLBACK` (`true`), `INQTRIX_OIDC_CA_CERT`, `INQTRIX_OIDC_INSECURE_DEV_COOKIES` (`false`; loopback-HTTP dev only, WARNs at startup).

Further tuning (LDAP attribute and TLS mapping): `INQTRIX_LDAP_USER_SEARCH_FILTER` (`(uid={username})`; AD commonly uses `(sAMAccountName={username})`, the login name is escaped before formatting), `INQTRIX_LDAP_EMAIL_ATTR` (`mail`; falls back to the login username), `INQTRIX_LDAP_DISPLAY_NAME_ATTR` (`cn`; falls back to email), `INQTRIX_LDAP_ID_ATTR` (`entryUUID`; the stable subject anchor — `objectGUID` for Active Directory, both survive renames where `uid` does not; falls back to the user DN), `INQTRIX_LDAP_FIRST_LOGIN_OWNER` (`true`; the first LDAP login becomes instance-admin if none exists yet), `INQTRIX_LDAP_START_TLS` (`false`; issue StartTLS on an `ldap://` connection before binding), `INQTRIX_LDAP_CA_CERT` (*(empty)*; PEM CA bundle for ldaps/StartTLS verification), `INQTRIX_LDAP_TLS_VALIDATE` (`true`; `false` skips certificate verification on a trusted dev network and WARNs). Per-IdP walkthrough: [Connect to an existing LDAP](../how-to/connect-to-existing-ldap.md).

**Interactions.** Mode inference reads `INQTRIX_SERVER_API_KEY` from the Server block. Misconfiguration fails loudly at startup: `apikey` without a configured key and `oidc` without issuer + client id + client secret + session secret are rejected; `local`/`ldap` without `INQTRIX_SESSION_SECRET` + `INQTRIX_PAT_PEPPER` (and `ldap` without its URL / bind DN+password / search base) are rejected; `none` with a configured key disables the gate deliberately and logs a WARNING. The cookie-session modes register the `/api/auth/*` routes — `oidc` adds `login|callback`, `local`/`ldap` add `POST /api/auth/login/local|ldap` (and `local` the `/api/setup/*` owner gate) — sharing the `session|logout` and PAT routes; the dev compose stack starts Dex only under the `oidc` profile. All three want the postgres storage backend for durable accounts/sessions (memory works but logins evaporate on restart).

**When this block is OFF.** Mode `none`: every request resolves to the anonymous principal and all routes are open — acceptable only on trusted networks. The mode is reported by the auth provider, not by re-reading the raw `INQTRIX_AUTH_MODE` value.

## Storage (`StorageSettings`)

Selects persistence for the platform layer (identity, file registry, run records) and for uploaded binary blobs (ObjectStore). The two selectors are independent: relational state goes to `memory` or `postgres`, blobs go to `local` disk or any `s3`-compatible endpoint.

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_STORAGE_BACKEND` | `memory` | `memory` (no external services, pytest default) or `postgres`. There is deliberately no SQLite option — the schema relies on Postgres row-level security. |
| `INQTRIX_DATABASE_URL` | *(empty)* | Async SQLAlchemy URL, e.g. `postgresql+asyncpg://inqtrix:...@127.0.0.1:5432/inqtrix`. Required for `postgres`; empty + `postgres` fails loudly at startup. The database must be migrated (`inqtrix-migrate`). |
| `INQTRIX_DATABASE_APP_ROLE` | `inqtrix_app` | Role switched to via `SET LOCAL ROLE` per transaction so row-level security applies even for owner/superuser connections. Empty disables the switch. |
| `INQTRIX_OBJECT_STORE_BACKEND` | `local` | `local` writes content-addressed blobs below `INQTRIX_OBJECT_STORE_PATH`; `s3` targets any S3-compatible endpoint (SeaweedFS in the dev compose stack) and requires the `INQTRIX_S3_*` fields. File metadata always lives in the file registry, never in the blob store. |
| `INQTRIX_S3_ENDPOINT_URL` | *(empty)* | S3 endpoint, e.g. `http://127.0.0.1:8333` for SeaweedFS. Required for `s3`; path-style addressing is always used. |
| `INQTRIX_MAX_FILE_BYTES` | `104857600` | Per-upload cap (100 MiB), enforced while spooling; HTTP 413 above. |

Further tuning: `INQTRIX_OBJECT_STORE_PATH` (`data/object-store`), `INQTRIX_S3_BUCKET` (`inqtrix-files`, created on startup; keys are namespaced `tenants/<tenant>/files/<uuid>`), `INQTRIX_S3_ACCESS_KEY`, `INQTRIX_S3_SECRET_KEY`, `INQTRIX_S3_REGION` (`us-east-1`).

**Interactions.** `INQTRIX_QUEUE_BACKEND=valkey` requires `INQTRIX_STORAGE_BACKEND=postgres` (enforced loudly in `build_container` — the run row is the source of truth for distributed execution). The dev compose stack at `deploy/compose/compose.dev.yaml` provides postgres and SeaweedFS.

**When this block is OFF.** With `memory`, identity facts, the file registry, and native run records live in process memory and vanish on restart; finished runs only outlive completion by `RUN_COMPLETED_TTL_SECONDS`. Uploaded blobs still hit the local disk path, but their registry entries do not survive a restart.

## Queue (`QueueSettings`)

Moves native-run execution out of the API process. Two orthogonal upgrades keep the zero-infrastructure default intact: `storage=postgres` makes run records durable (execution stays in-process), and `queue=valkey` additionally dispatches execution to `inqtrix-worker` processes via a Valkey Stream with at-least-once delivery.

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_QUEUE_BACKEND` | `memory` | `memory` executes runs in-process; `valkey` dispatches to workers and requires `INQTRIX_STORAGE_BACKEND=postgres` plus a non-empty `INQTRIX_VALKEY_URL`. |
| `INQTRIX_VALKEY_URL` | *(empty)* | Connection URL (`redis://` scheme, e.g. `redis://127.0.0.1:6379/0`). Required for `valkey`; ignored otherwise. |
| `INQTRIX_WORKER_CONCURRENCY` | `2` | Runs one worker process executes concurrently. Each run blocks one thread for its full duration (the research graph is synchronous). |
| `INQTRIX_WORKER_MAX_ATTEMPTS` | `3` | Delivery attempts before a run is dead-lettered and marked failed. Redelivery of finished runs is a no-op. |

Further tuning: `INQTRIX_WORKER_HEARTBEAT_SECONDS` (15.0 — workers re-claim their in-flight stream entries so long runs are not stolen), `INQTRIX_WORKER_CLAIM_IDLE_SECONDS` (90.0 — idle threshold for reclaiming entries from crashed workers; sized to heartbeat loss, not run duration).

**Interactions.** `valkey` without `postgres` storage or without a Valkey URL fails loudly at startup, on both sides: `build_container` rejects it in the API server, and `inqtrix-worker` refuses to start without `INQTRIX_STORAGE_BACKEND=postgres` plus `INQTRIX_QUEUE_BACKEND=valkey`. The stream carries only dispatch messages; run state lives in the Postgres run row. The dev compose stack provides the Valkey service.

**When this block is OFF.** With `memory`, runs execute inside the API server, capped by `RUN_MAX_CONCURRENT` / `MAX_CONCURRENT`; there are no worker processes, no retries across crashes, and a server restart aborts in-flight runs.

## Knowledge (`KnowledgeSettings`)

The internal document-retrieval engine: collections, document ingestion, hybrid retrieval, and the `mode=knowledge` answer pipeline (gate, grounding, retrieval profiles). Disabled by default so the historical deployment shape stays untouched.

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_KNOWLEDGE_ENABLED` | `false` | Master switch. Off: no knowledge/sources routes, no embedding provider, `mode=knowledge` is not a registered algorithm. |
| `INQTRIX_VECTOR_BACKEND` | `memory` | `memory` (in-process, lost on restart) or `qdrant` (persistent, hybrid dense + BM25 retrieval; requires the `knowledge-qdrant` extra and `INQTRIX_QDRANT_URL`). |
| `INQTRIX_QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant REST endpoint; default matches the dev compose stack. Set `INQTRIX_QDRANT_API_KEY` everywhere except pure loopback dev — self-hosted Qdrant is unauthenticated by default. |
| `INQTRIX_KNOWLEDGE_SPARSE` | `bm25_german` | Lexical branch for the qdrant backend: client-side BM25 sparse vectors fused with the dense branch via RRF. `off` runs dense-only. Ignored by the memory backend. |
| `INQTRIX_RERANKER_PROVIDER` | `none` | Rerank stage after retrieval. `none` skips it (a visible capability flag, never a silent downgrade); `cohere` calls a Cohere-rerank-schema endpoint (native or Azure AI Foundry serverless) and requires `INQTRIX_RERANKER_BASE_URL` / `INQTRIX_RERANKER_API_KEY` / `INQTRIX_RERANKER_MODEL`; `llm` ranks listwise through the deployment's own LLM. |
| `INQTRIX_DOCUMENT_PARSER` | `markitdown` | File ingestion: converts PDF/DOCX/PPTX/XLSX/HTML to Markdown in pure Python. `none` disables file ingestion (the text-only API stays available). |
| `INQTRIX_KNOWLEDGE_GATE` | `on` | Sufficiency gate for `mode=knowledge`: a fast-tier LLM call judges the evidence and may trigger another retrieval pass; insufficient evidence yields an honest no-evidence answer. |
| `INQTRIX_KNOWLEDGE_GROUNDING` | `on` | Quote-then-answer grounding: verbatim quotes are required before the answer, verified deterministically (no extra LLM call), and stripped from the user-facing text. |
| `INQTRIX_KNOWLEDGE_GATE_MAX_ROUNDS` | `3` | Hard operator cap (1-5) on gate rewrite-and-retrieve rounds for every retrieval profile; `tief` requests up to this many, `standard` always uses one. |
| `INQTRIX_KNOWLEDGE_CONTEXTUALIZE` | `off` | Contextual retrieval: one batched fast-tier LLM call per document at ingestion prepends a situating context per chunk. Existing documents are unaffected — re-ingest to apply. |
| `INQTRIX_EMBEDDING_MODEL` | `text-embedding-3-small` | Default embedding model for new collections. Each collection stores its model immutably at creation. |

Further tuning: `INQTRIX_RERANK_CANDIDATE_DEPTH` (40 — pool retrieved before rerank reduces to top_k), `INQTRIX_KNOWLEDGE_TOP_K` (8; per-request override via `knowledge_filters.top_k`), `INQTRIX_EMBEDDING_PROVIDER` (`openai_compatible` or `azure`; `azure` reads `INQTRIX_EMBEDDING_AZURE_ENDPOINT` / `INQTRIX_EMBEDDING_AZURE_API_KEY` / `INQTRIX_EMBEDDING_AZURE_API_VERSION` (`2024-10-21`) with fallbacks to the established `AZURE_AI_PROJECT_ENDPOINT` / `AZURE_AI_PROJECT_API_KEY` / `AZURE_OPENAI_API_KEY` variables), `INQTRIX_EMBEDDING_BASE_URL` / `INQTRIX_EMBEDDING_API_KEY` (empty reuses `LITELLM_BASE_URL` / `LITELLM_API_KEY`), `INQTRIX_SELECTABLE_EMBEDDING_MODELS` (empty hides the collection-creation picker), `INQTRIX_KNOWLEDGE_CHUNK_MAX_CHARS` (2000), `INQTRIX_KNOWLEDGE_MAX_DOCUMENT_CHARS` (2000000).

Background reindex (re-embed) tuning: `INQTRIX_REINDEX_MAX_CONCURRENT` (6 — how many DIFFERENT collections re-embed at once; a single collection is always serialized, one active reindex per collection; concurrent reindexes add up on the embedding endpoint and compete with live query-embedding, so lower it if bulk reindex starves interactive search — in worker mode the real parallelism is `INQTRIX_WORKER_CONCURRENCY` and this governs admission only), `INQTRIX_REINDEX_QUEUE_MAX_SIZE` (50 — waiting reindex jobs; full → 429), `INQTRIX_REINDEX_COMPLETED_TTL_SECONDS` (3600 — terminal-record retention, in both the in-memory and the durable Postgres store), `INQTRIX_REINDEX_HISTORY_LIMIT` (10 — terminal records kept per collection for the inline run history), `INQTRIX_REINDEX_EVENT_BUFFER_SIZE` (200 — recent events retained per job for late SSE subscribers, in-memory tier).

**Interactions.** The embedding endpoint defaults to the Server block's LiteLLM gateway, so a standard proxy deployment needs no extra embedding configuration. `INQTRIX_RERANKER_PROVIDER=llm` runs through the deployment's own LLM provider (fast tier) — no rerank API contract needed, but roughly an order of magnitude costlier and slower than a cross-encoder, hard-capped at 20 candidates per query with a visible log line. `INQTRIX_PUBLIC_BASE_URL` (Server) turns knowledge citations into clickable `/v1/sources/...` links. The dev compose stack provides Qdrant.

**When this block is OFF.** No knowledge or sources routes are registered, no embedding provider is constructed, and requests naming `mode=knowledge` get the standard mode-validation 400. The `/v1/capabilities` manifest reports the absence, so the research-desk UI hides the knowledge workspace instead of rendering dead controls.

## Quota (`QuotaSettings`)

Per-user usage quotas for multi-user deployments — the operator-ceiling layer of a two-level rule: the admin UI sets the middle layer (a tenant default and per-user overrides) within these bounds. Each flow dimension has a `*_default` (the out-of-box per-user allowance) and a `*_max` (the hard ceiling no admin-set value may exceed). A value of `0` means UNLIMITED, so the all-zero default leaves every deployment byte-identical until an operator sets real numbers. Quotas apply ONLY when enabled and the active auth mode is one of the cookie-session modes (`oidc`, `local`, `ldap`); the anonymous/static principals are never metered.

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_QUOTA_ENABLED` | `false` | Master switch. Even when on, quotas bind only in `oidc`, `local`, or `ldap` mode — the single-operator `none`/`apikey` modes are never metered. Advertised in `/v1/capabilities` as `features.quota`. |
| `INQTRIX_QUOTA_RUNS_PER_MONTH` | `0` | Default native research runs a user may start per calendar month. `0` = unlimited. Checked exactly at submission. |
| `INQTRIX_QUOTA_RUNS_PER_MONTH_MAX` | `0` | Hard ceiling for the monthly-run allowance; no admin default or override may exceed it. `0` = no ceiling. |
| `INQTRIX_QUOTA_LLM_TOKENS_PER_MONTH` | `0` | Default LLM tokens (prompt + completion, summed across runs, chat, editor) per user per calendar month. `0` = unlimited. Recorded post-hoc: the current call finishes and the NEXT submission is blocked once the budget is reached. |
| `INQTRIX_QUOTA_LLM_TOKENS_PER_MONTH_MAX` | `0` | Hard ceiling for the monthly LLM-token allowance. `0` = no ceiling. |
| `INQTRIX_QUOTA_EMBEDDING_TOKENS_PER_MONTH` | `0` | Default embedding input tokens per user per calendar month (document ingestion). `0` = unlimited. Same block-next model as the other flow dimensions. |
| `INQTRIX_QUOTA_EMBEDDING_TOKENS_PER_MONTH_MAX` | `0` | Hard ceiling for the monthly embedding-token allowance. `0` = no ceiling. |
| `INQTRIX_QUOTA_STORED_BYTES` | `0` | Default object-store occupancy a user may hold, in bytes. A STOCK quota (rises on upload, falls on delete, never resets). `0` = unlimited. |
| `INQTRIX_QUOTA_STORED_BYTES_MAX` | `0` | Hard ceiling for per-user object-store occupancy in bytes. `0` = no ceiling. |
| `INQTRIX_QUOTA_MAX_TOKENS_PER_RUN` | `0` | Optional HARD per-run token budget. `0` = off. When set, a single run that crosses the budget mid-flight is cancelled gracefully at the next graph-node boundary (partial result returned, never a mid-call kill). Independent of the monthly token quota. |

**Interactions.** Enforcement reads the ACTIVE auth mode (so it follows `INQTRIX_AUTH_MODE=infer`), not the raw value. The admin UI sets the tenant default and per-user overrides within each `*_max`. Embedding-token runaway on a single ingestion is already bounded by `INQTRIX_KNOWLEDGE_MAX_DOCUMENT_CHARS` and `INQTRIX_MAX_FILE_BYTES`. See [Create and manage users](../how-to/create-and-manage-users.md) and [Authentication modes](../deployment/auth-modes.md).

**When this block is OFF.** With `INQTRIX_QUOTA_ENABLED=false` (default) or any non-cookie-session auth mode (`none`/`apikey`), nothing is metered and behaviour is byte-identical to deployments before quotas existed.

## Sharing (`SharingSettings`)

Resource-sharing policy for the cookie-session multi-user modes (`oidc`/`local`/`ldap`). Sharing is otherwise tenant-wide: any authenticated user may be a share target and the share typeahead searches the whole tenant.

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_SHARING_RESTRICT_TO_WORKSPACE_MEMBERS` | `false` | When `true`, a user may only share a resource with people they share at least one workspace with, and the share typeahead is scoped the same way. A grant-time (write) restriction only — turning it on never revokes an existing grant. Default `false` keeps sharing tenant-wide, byte-identical to deployments before this setting existed. Only meaningful in the cookie-session modes that mount the sharing surface; the single-operator `none`/`apikey` modes never mount it. |

**When this block is OFF.** Default `false` = tenant-wide sharing, identical to before this setting existed.

## Process-level variables (outside `Settings`)

A few variables are read by the process bootstrap rather than the Pydantic groups. The server/worker bind and logging knobs:

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_SERVER_HOST` | `0.0.0.0` | Bind host for uvicorn in `python -m inqtrix`. |
| `INQTRIX_SERVER_PORT` | `5100` | Bind port for uvicorn in `python -m inqtrix`. |
| `INQTRIX_LOG_ENABLED` | `false` | Master switch for persistent file logging (`true` activates it); otherwise the library only attaches a `NullHandler`. Read by both the server and `inqtrix-worker`. |
| `INQTRIX_LOG_LEVEL` | `INFO` | Logging level for the `inqtrix` logger: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Structured runtime events are emitted at `DEBUG`. |
| `INQTRIX_LOG_CONSOLE` | `false` | `true` mirrors WARNING+ records onto stderr in addition to any file sink. |
| `INQTRIX_LOG_INCLUDE_WEB` | `true` | When file logging is on, also route uvicorn/FastAPI logs into the same file. Set `false` to opt out. |
| `INQTRIX_LOG_WEB_LEVEL` | `INFO` | Level for the uvicorn/FastAPI loggers (used by `build_uvicorn_log_config`). |

Full logging behaviour, file paths, and forensic-event interaction with `OBSERVABILITY_PROFILE`: [Logging](../observability/logging.md).

## The two-level rule

Knowledge env switches are the OPERATOR CEILING; the per-request retrieval profile (`schnell` | `standard` | `gruendlich` | `tief` | `auto`) selects within it. `INQTRIX_KNOWLEDGE_GATE=off`, `INQTRIX_KNOWLEDGE_GROUNDING=off`, `INQTRIX_RERANKER_PROVIDER=none`, and `INQTRIX_KNOWLEDGE_GATE_MAX_ROUNDS` clamp every profile, and every clamp is visible (`degraded_stages` in events, result state, and `/v1/capabilities`). Full matrix and stage notes in [Retrieval profiles](knowledge-profiles.md).

## Development and test-only variables

These variables are read by helper scripts, the evaluation harness, and the test suite — never by the server or worker at runtime. They are listed here for completeness; none belongs in a production deployment. Background: [Running tests](../development/running-tests.md), [Local infrastructure](../development/local-infrastructure.md), [Testing strategy](../development/testing-strategy.md).

Replay tests (`tests/replay/`):

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_RECORD_MODE` | *(unset → replay-only)* | vcrpy record mode for the cassette replay tests: `once`, `new_episodes`, `all`, or `none`. Re-records cassettes against a real backend when set (see the `uv run` recipe in [Running tests](../development/running-tests.md)). |

Optional external services for integration tests (each empty value skips its suite):

| Variable | Default | Effect |
|----------|---------|--------|
| `INQTRIX_TEST_DATABASE_URL` | *(empty)* | Postgres URL enabling the `tests/storage/*_postgres.py` suite. |
| `INQTRIX_TEST_QDRANT_URL` | *(empty)* | Qdrant endpoint enabling the qdrant-store tests. |
| `INQTRIX_TEST_QDRANT_API_KEY` | *(empty)* | API key for the test Qdrant instance. |
| `INQTRIX_TEST_S3_ENDPOINT` | *(empty)* | S3-compatible endpoint enabling the object-store S3 tests. |
| `INQTRIX_TEST_S3_ACCESS_KEY` | `inqtrix-dev-access` | Access key for the test S3 endpoint. |
| `INQTRIX_TEST_S3_SECRET_KEY` | `inqtrix-dev-secret` | Secret key for the test S3 endpoint. |

Evaluation harness (`tests/eval/`): `INQTRIX_EVAL_GOLDEN_SET` (`base`), `INQTRIX_EVAL_KNOWLEDGE_PROFILE` (`standard`), `INQTRIX_EVAL_VECTOR_BACKEND` (`memory`), `INQTRIX_EVAL_QDRANT_URL` (`http://127.0.0.1:6333`), `INQTRIX_EVAL_QDRANT_API_KEY` (*(empty)*), `INQTRIX_EVAL_SPARSE` (`bm25_german`), `INQTRIX_EVAL_RERANKER` (`none`), `INQTRIX_EVAL_CONTEXTUALIZE` (`off`), `INQTRIX_EVAL_EMBEDDING_PROVIDER` (`openai_compatible`), `INQTRIX_EVAL_EMBEDDING_MODEL` (`text-embedding-3-small`), `INQTRIX_EVAL_EMBEDDING_BASE_URL` (*(empty)*), `INQTRIX_EVAL_EMBEDDING_API_KEY` (*(empty)*), `INQTRIX_EVAL_AZURE_ENDPOINT` (*(empty)*), `INQTRIX_EVAL_AZURE_API_KEY` (*(empty)*). These mirror the corresponding runtime knobs but isolate an eval run from any deployment configuration.

Research Desk launcher (`scripts/run_research_desk.py`, dev convenience): `INQTRIX_DIST_DIR` (*(empty)*; override the built React `dist/` location), `INQTRIX_BACKEND_URL` (`http://localhost:5100`; backend origin the launcher proxies to), `RESEARCH_DESK_HOST` (`127.0.0.1`), `RESEARCH_DESK_PORT` (`8080`).

Search debug script (`scripts/debug_search_dataflow.py`): `INQTRIX_PERPLEXITY_INSTRUCTIONS` (*(unset)*), `INQTRIX_PERPLEXITY_MODEL` (*(unset)*), `INQTRIX_PERPLEXITY_PRESET` (`fast-search`). Script-local overrides for ad-hoc Perplexity dataflow debugging; the runtime equivalents are `INQTRIX_SEARCH_INSTRUCTIONS` / `SEARCH_MODEL` / `INQTRIX_SEARCH_PRESET` in the Providers block.

Live prompt-flow debug script (`scripts/live_prompt_flow_debug.py`): `INQTRIX_LOG_DIR` (default `logs`; the `configure_logging(log_dir=...)` target, wired from env only by this script — the server/worker always use `logs`). The script also defaults `OBSERVABILITY_PROFILE` to `forensic` for its own run.

## Related docs

- [Agent config](agent-config.md)
- [Report profiles](report-profiles.md)
- [Retrieval profiles](knowledge-profiles.md)
- [Security hardening](../deployment/security-hardening.md)
- [Logging](../observability/logging.md)
