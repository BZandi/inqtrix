# Web server mode

> Files: `src/inqtrix/server/app.py`, `src/inqtrix/server/routes.py`, `src/inqtrix/server/runs.py`, `src/inqtrix/server/streaming.py`, `src/inqtrix/server/stacks.py`

## Scope

How Inqtrix exposes the agent as an OpenAI-compatible HTTP service. This page summarises the endpoints, lifecycle, concurrency, and cancel semantics. The authoritative operational reference for the eight example stacks lives in [`examples/webserver_stacks/README.md`](../../examples/webserver_stacks/README.md) — this doc links to it for deployment-specific details.

## Endpoints

| Route | Method | Auth | Purpose |
|-------|--------|------|---------|
| `/health` | GET | Open | Liveness probe, active-provider summary, and project legal/source metadata. |
| `/v1/models` | GET | Open | OpenAI-style model discovery. Returns `research-agent`. |
| `/v1/stacks` | GET | Open | Multi-stack-only discovery (created by `create_multi_stack_app`). 5-second cache. |
| `/v1/chat/completions` | POST | Optional Bearer | Main research endpoint, streaming and non-streaming. |
| `/v1/text/improvements` | POST | Optional Bearer | LLM-backed improvement suggestions for browser text fields. |
| `/v1/editor/suggest` | POST | Optional Bearer | LLM-backed rewrite for a selected/commented Markdown block in the React editor. Accepts an optional additive `attachments` array of reference documents. |
| `/v1/editor/instruct` | POST | Optional Bearer | LLM-backed document-level editor instruction returning content-anchored edit proposals. Accepts an optional additive `attachments` array of reference documents. |
| `/v1/runs` | POST/GET | Optional Bearer | Native UI run API with queueing, live structured events, cancellation, and final report fetch. |
| `/v1/runs/{run_id}` | GET | Optional Bearer | Current run summary for a queued, running, or short-lived terminal run. |
| `/v1/runs/{run_id}/events` | GET | Optional Bearer | Buffered + live Server-Sent Events for a native run. |
| `/v1/runs/{run_id}/result` | GET | Optional Bearer | Final `ResearchResult` export after completion. |
| `/v1/runs/{run_id}/cancel` | POST | Optional Bearer | Cancel a queued run or request cancellation for a running run. |
| `/v1/test/run` | POST | Optional Bearer | Structured test endpoint, only when `TESTING_MODE=true`. |

The Bearer layer activates when `INQTRIX_SERVER_API_KEY` is set; see [Security hardening](security-hardening.md).

## Authentication and TLS

### Server-side activation — no code change required

Every script under `examples/webserver_stacks/*.py` is already wired for both layers:

- **API-key gate.** `create_app()` calls `make_api_key_dependency(settings.server)` on startup ([`src/inqtrix/server/app.py`](../../src/inqtrix/server/app.py)). Once `INQTRIX_SERVER_API_KEY` is set, the dependency is automatically attached to `/v1/chat/completions`, `/v1/text/improvements`, `/v1/runs*`, and `/v1/test/run`. `/health`, `/v1/models`, and `/v1/stacks` stay open. Comparison is timing-safe (`hmac.compare_digest`).
- **TLS.** Each script calls `resolve_tls_paths(settings.server)` and forwards `ssl_keyfile` / `ssl_certfile` to `uvicorn.run(...)`. Once `INQTRIX_SERVER_TLS_KEYFILE` and `INQTRIX_SERVER_TLS_CERTFILE` are set, the same script binds HTTPS instead of HTTP. Both env vars are required — partial configuration raises `RuntimeError` on startup (no silent downgrade).

> **`python -m inqtrix` caveat.** The default entry point activates the API-key gate via `create_app()`, but it does **not** wire TLS into uvicorn. For TLS, either run one of the `examples/webserver_stacks/*.py` scripts directly, or terminate TLS at a reverse proxy (nginx, Traefik, Azure Application Gateway).

### End-to-end example: API key + TLS

Take [`examples/webserver_stacks/azure_openai_perplexity.py`](../../examples/webserver_stacks/azure_openai_perplexity.py) as the reference. The script itself stays untouched; only the environment changes.

1. Generate a local cert (only for development — use a real CA for anything externally reachable):

   ```bash
   mkcert -install
   mkcert -key-file key.pem -cert-file cert.pem localhost
   ```

2. Populate `.env` (or export the variables in the shell):

   ```dotenv
   # Provider credentials (already required by this stack)
   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
   AZURE_OPENAI_API_KEY=<azure-key>
   PERPLEXITY_API_KEY=<perplexity-key>

   # Bearer-token gate — any sufficiently random string
   INQTRIX_SERVER_API_KEY=dev-secret-xxxxx

   # TLS pair — both required, partial config raises RuntimeError
   INQTRIX_SERVER_TLS_KEYFILE=./key.pem
   INQTRIX_SERVER_TLS_CERTFILE=./cert.pem
   ```

3. Start the server:

   ```bash
   uv run python examples/webserver_stacks/azure_openai_perplexity.py
   ```

   The startup log line should read `... | api_key_gate=on | cors=...` and uvicorn should announce HTTPS on port 5100. Any deviation is a deployment mistake — fail the smoke test fast.

### Client-side request examples

The examples below assume HTTP on `localhost:5100` for clarity. For TLS, swap the scheme to `https://` and add `--cacert ./cert.pem` (curl) or `verify="./cert.pem"` (httpx / requests).

**bash / curl**

```bash
# Authenticated request — returns 200 and streams the answer
curl -N http://localhost:5100/v1/chat/completions \
    -H "Authorization: Bearer dev-secret-xxxxx" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "research-agent",
        "messages": [{"role": "user", "content": "Was ist der Stand der GKV-Reform?"}],
        "stream": true
    }'

# Missing or wrong header — returns 401 with WWW-Authenticate: Bearer
curl -i http://localhost:5100/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}]}'
```

**Python (`httpx`)**

```python
import httpx

API_KEY = "dev-secret-xxxxx"
BASE_URL = "http://localhost:5100"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
body = {
    "model": "research-agent",
    "messages": [{"role": "user", "content": "Was ist der Stand der GKV-Reform?"}],
    "stream": False,
}

with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=1800.0, write=30.0, pool=5.0)) as client:
    resp = client.post(f"{BASE_URL}/v1/chat/completions", headers=headers, json=body)
    resp.raise_for_status()
    print(resp.json()["choices"][0]["message"]["content"])
```

For SSE streaming use `client.stream("POST", url, headers=headers, json={..., "stream": True})` and iterate over `resp.iter_lines()`. The bundled UI client at [`inqtrix_webapp/client.py`](../../inqtrix_webapp/client.py) is a complete reference implementation.

**Python (`requests`)**

```python
import requests

API_KEY = "dev-secret-xxxxx"
BASE_URL = "http://localhost:5100"

resp = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "research-agent",
        "messages": [{"role": "user", "content": "Was ist der Stand der GKV-Reform?"}],
        "stream": False,
    },
    timeout=(5, 1800),
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
```

### Streamlit UI

The bundled UI is a pure HTTP consumer — it does not import the `inqtrix` package. Set the same token on both sides:

```bash
export INQTRIX_SERVER_API_KEY=dev-secret-xxxxx   # server
export INQTRIX_WEBAPP_API_KEY=dev-secret-xxxxx   # UI
```

The UI calls `GET /health` on startup; the response carries `auth_required: true` when the gate is on, and the UI surfaces a token field in the model popover so end users can paste their own token. See [`inqtrix_webapp/client.py`](../../inqtrix_webapp/client.py) for the full client surface.

For the full hardening picture (TLS at uvicorn vs. reverse proxy, CORS allow-list, operator-visible startup banner), see [Security hardening](security-hardening.md).

## Startup

`python -m inqtrix` boots the default server via `__main__.py`. The eight example scripts in `examples/webserver_stacks/*.py` follow the same pattern but inject explicit providers instead of relying only on `Settings`:

```python
from inqtrix.server.app import create_app
from inqtrix.server.stacks import StackBundle, create_multi_stack_app
```

`create_app(*, settings=None, providers=None, strategies=None)` is the single-stack factory. `create_multi_stack_app(*, settings, stacks, default_stack)` is the multi-stack variant. Both keep provider construction outside the route layer so explicit Baukasten stacks and env-driven stacks share the same runtime path.

### Lifespan logging

Both factories wire an ASGI `lifespan` context manager that logs on startup and shutdown. On startup it probes `is_available()` per provider, logs the active security layers (TLS on/off, API key on/off, CORS on/off), and logs the report profile, concurrency, native-run queue size, and native-run completed TTL. On shutdown it writes a compact "server stopping" line. This satisfies the "no silent fallbacks" rule for operator visibility and fires automatically for `TestClient(app)` as well.

### uvicorn log mirroring

The example scripts pass `log_config=build_uvicorn_log_config(log_file, web_level=...)` to `uvicorn.run(...)` so that `uvicorn.error`, `uvicorn.access`, and the `inqtrix` logger all write to the same file. Attaching a mirror handler to `uvicorn.*` at runtime does **not** work because uvicorn's internal `logging.config.dictConfig` replaces the handlers on boot.

## Per-request overrides

Clients can override a whitelisted subset of agent fields per request:

```json
{
  "model": "research-agent",
  "messages": [{"role": "user", "content": "..."}],
  "agent_overrides": {
    "max_rounds": 3,
    "confidence_stop": 7,
    "report_profile": "deep"
  }
}
```

The whitelist is: `max_rounds`, `min_rounds`, `confidence_stop`, `report_profile`, `max_total_seconds`, `first_round_queries`, `skip_search`, and `model_tier`. Unknown keys return HTTP 400. Specific model *names* remain operator concerns and are not overridable; `model_tier` only selects among the operator-configured tiers. See [Agent config](../configuration/agent-config.md) and the recipe in `src/inqtrix/server/overrides.py` for how to extend the whitelist safely.

`skip_search=true` is the direct-chat path used by the Streamlit UI when web search is disabled. It bypasses plan/search/evaluate, calls the LLM provider directly with the question plus conversation history, returns no citations, and leaves `round=0`.

### Model tiers

Model selection is an operator concern, not a per-request override. To route nodes to three tiers — `answer` to high, `plan`/`evaluate`/`direct_chat` to mid, `classify`/`claim_extract` to fast — set the tier environment variables before boot:

```dotenv
TIER_HIGH_MODEL=claude-opus-4-7
TIER_HIGH_EFFORT=medium
TIER_MID_MODEL=claude-sonnet-4-6
TIER_FAST_MODEL=claude-haiku-4-5
# optional per-node model override:
ANSWER_MODEL=claude-opus-4-7
```

Per-tier effort turns reasoning on deliberately, so tiers differ by model alone until set. The per-request `model_tier` override (`"high"`/`"mid"`/`"fast"`) selects a tier for one run — useful with `skip_search` or `mode="direct_llm"` to pick the model class for a direct-chat answer. `/health` and `/v1/stacks` also expose `chat_model_options`, a `direct_chat` descriptor for each tier with the actual model name, effort token, tier, and provenance. The React Research Desk uses that list for its Chat mode picker and still sends only `agent_overrides.model_tier`; specific model names remain server-side operator configuration. For the full call-site-to-tier mapping and per-provider effort behaviour see [LLM calls, model tiers, and reasoning effort](../architecture/llm-calls.md).

## Run mode

Both `/v1/chat/completions` and `/v1/runs` accept an optional top-level `mode` field:

- `research` forces the normal classify/plan/search/evaluate/answer graph.
- `direct_llm` routes the request through the active LLM provider directly, using the same internal path as `skip_search=true`.

If `mode` is omitted, the server keeps the existing behaviour: server/stack settings and `agent_overrides.skip_search` decide. `agent_overrides.skip_search` remains supported for compatibility, but new UI clients should prefer `mode`. Conflicting requests such as `mode="direct_llm"` with `agent_overrides.skip_search=false` return HTTP 400.

OpenAI-compatible direct chat:

```json
{
  "model": "research-agent",
  "mode": "direct_llm",
  "messages": [{"role": "user", "content": "Answer without web research."}],
  "stream": false,
  "agent_overrides": {"model_tier": "fast"}
}
```

Non-streaming chat-completion responses keep the OpenAI-compatible top-level
`model` value (`research-agent`) and add an optional Inqtrix diagnostics block
when the active provider exposes model metadata:

```json
{
  "model": "research-agent",
  "inqtrix": {
    "model_resolution": {
      "node": "direct_chat",
      "model": "claude-haiku-4-5",
      "tier": "fast",
      "effort": "none",
      "model_source": "tier:fast",
      "effort_source": "tier:fast",
      "requested_tier": "fast"
    }
  }
}
```

Native run API direct chat:

```json
{
  "mode": "direct_llm",
  "messages": [{"role": "user", "content": "Chat directly with the active LLM."}]
}
```

## Streaming (SSE)

`/v1/chat/completions` supports OpenAI-style Server-Sent Events:

```bash
curl -N http://localhost:5100/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "research-agent",
        "messages": [
            {"role": "user", "content": "Was ist der aktuelle Stand der GKV-Reform?"}
        ],
        "stream": true
    }'
```

When `stream` is `true`, progress chunks come first (`> Research Step: ...`), followed by a `---` separator and the answer chunks, terminated with `data: [DONE]`. If the direct-chat run resolved model metadata, the server emits a metadata chunk carrying `inqtrix.model_resolution` before the first answer-token chunk. Pass `"include_progress": false` for answer-only SSE.

Library streaming yields plain text chunks (see [Library mode](library-mode.md)). HTTP streaming yields SSE chunks in the OpenAI-compatible `data: {...}` format.

Native browser UIs should prefer `/v1/runs/{run_id}/events` when they need structured state for progress cards. That stream uses named SSE events such as `inqtrix.run.queued`, `inqtrix.run.snapshot`, `inqtrix.node.started`, `inqtrix.progress.message`, `inqtrix.output_text.delta`, and `inqtrix.run.completed`. See [Run events](../observability/run-events.md).

The React Research Desk in `apps/research-desk/` consumes this native run API
directly. In local Vite development it uses the same-origin `/health` and `/v1`
proxy by default, or it can target a separately deployed API by setting
`VITE_INQTRIX_API_BASE_URL` to a complete origin (scheme + host + optional port).
The same variable bakes the backend origin into a production build:

```bash
# Dev server against a non-default backend:
VITE_INQTRIX_API_BASE_URL=http://127.0.0.1:5100 pnpm run ui:dev
# or: VITE_INQTRIX_API_BASE_URL=http://127.0.0.1:5100 npm run ui:dev

# Production bundle with a fixed backend origin (split-origin hosting; backend needs CORS):
VITE_INQTRIX_API_BASE_URL=https://inqtrix-api.example.com pnpm run ui:build
# or: VITE_INQTRIX_API_BASE_URL=https://inqtrix-api.example.com npm run ui:build
```

For same-origin production serving (no baked origin, no CORS), see
[React UI](react-ui.md), which also covers the nginx two-pod topology and the
Python launcher. Do not put `INQTRIX_SERVER_API_KEY` into a `VITE_*` variable; when the server's
health payload reports `auth_required: true`, the React app passes a
runtime-entered Bearer token to protected `/v1/*` requests and to the
fetch-based run-event stream.

Native UI clients should send a stable, non-secret workspace namespace with
`X-Inqtrix-Workspace-Id` or a top-level `workspace_id` on `POST /v1/runs`.
When present, the server stores it on the run record, filters `GET /v1/runs`,
and requires the same namespace for `GET /v1/runs/{run_id}`,
`GET /v1/runs/{run_id}/events`, `GET /v1/runs/{run_id}/result`, and
`POST /v1/runs/{run_id}/cancel`. Omitting the workspace id preserves the
historical unscoped behaviour for scripts and operator debugging. The
workspace id is a browser/project routing namespace, not an auth boundary.

## Concurrency

`MAX_CONCURRENT` (default 3) caps active `/v1/chat/completions` requests. The OpenAI-compatible path keeps its historical behaviour: when the semaphore is saturated, the server returns HTTP 429 instead of queueing.

The native `/v1/runs` path uses `RunStore`, an in-memory FIFO queue. Its active worker cap is `RUN_MAX_CONCURRENT` when set, otherwise it falls back to `MAX_CONCURRENT`. When all native run slots are busy, accepted runs wait up to `RUN_QUEUE_MAX_SIZE` queued jobs (default 50). If that queue is full, `POST /v1/runs` returns HTTP 429. Active jobs do not count against the queue size.

These caps are per HTTP surface, not a global provider-call limiter. If both `/v1/chat/completions` and `/v1/runs` are used at the same time, worst-case active provider work can be the sum of the two caps. Use matching values or a lower `RUN_MAX_CONCURRENT` when the deployment must keep a stricter provider quota envelope.

Terminal native runs remain in memory for `RUN_COMPLETED_TTL_SECONDS` (default 300) together with a bounded event buffer of `RUN_EVENT_BUFFER_SIZE` events (default 200). This gives a React UI enough time to replay events and fetch the final report after completion, but it is not durable persistence. If a product requires completed reports after refresh or server restart, add a database-backed run/result store later.

Compiled LangGraph instances are cached per provider identity, strategy identity, and effective `AgentSettings` fingerprint. The cache is thread-safe and bounded to 32 graph variants per process. That size is not a concurrency limit: active requests keep their own graph reference, and eviction only means a later matching request may need to compile the graph again.

## Cancel

Every streaming response spawns a watcher task that calls `await request.receive()` blocking on `http.disconnect`. On disconnect it sets `cancel_event`, and the next node boundary raises `AgentCancelled`. Latency from disconnect to actual stop equals the remaining duration of the currently running provider call — typically 5-60 seconds. Polling `request.is_disconnected()` was avoided because it can miss disconnects during active SSE writes.

Native runs additionally expose `POST /v1/runs/{run_id}/cancel`. Queued runs transition to `cancelled` immediately. Running runs set the same `cancel_event` used by disconnect cancellation, so they stop at the next node boundary. Hard cancel through in-flight provider calls remains out of scope.

## Request Boundaries

The HTTP server is stateless for research data. Each request runs with a fresh
`AgentState`; prior chat messages are still formatted into the `history` string,
but previous citations, EvidenceRecords, claims, aspects, and confidence are not
carried into the new run.

## Multi-stack serving

`create_multi_stack_app(...)` mounts several provider stacks in one process. Each stack is a `StackBundle(providers, strategies, agent_settings, description)` keyed by a lowercase-alphanumeric name. The request picks a stack via `body["stack"]`; absent values use `default_stack`, while unknown names return HTTP 400 with an `available_stacks` hint. The dedicated example is `examples/webserver_stacks/multi_stack.py`; each `StackBundle` is opt-in through environment-variable gating.

## Health payload

`/health` returns the report profile, the active security layers, per-role model identities resolved from provider constructors, search-model identities via the `SearchProvider.search_model` property, and a `legal` block with the project name, `AGPL-3.0-only` license identifier, source URL, copyright notice, attribution notice, and no-warranty notice. Operators should read these values to confirm the deployment is wired as intended and to surface the source and warranty links in network-facing UIs. `/v1/stacks` exposes the same provider/model information per stack for multi-stack deployments.

## Related docs

- [Examples README](../../examples/webserver_stacks/README.md) — the operational reference with per-stack env-variable tables and run commands.
- [Enterprise Azure](enterprise-azure.md) — Managed Identity, SP, Foundry token lifetime.
- [Security hardening](security-hardening.md) — TLS, Bearer, CORS.
- [Agent config](../configuration/agent-config.md) — per-request overrides.
- [React UI](react-ui.md) — Vite/shadcn frontend setup, build workflow, and same-origin deployment via nginx or the `scripts/run_research_desk.py` launcher.
- [Streamlit UI](streamlit-ui.md) — bundled HTTP frontend and override mapping.
