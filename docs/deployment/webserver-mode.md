# Web server mode

> Files: `src/inqtrix/server/app.py`, `src/inqtrix/server/container.py`, `src/inqtrix/server/routes.py`, `src/inqtrix/server/routers/`, `src/inqtrix/services/`, `src/inqtrix/server/stacks.py`

## Scope

How Inqtrix exposes the agent platform as an HTTP service (v0.2.0): the layering, the endpoint surface per feature, native runs with their SSE event stream, the capability manifest, and the storage/queue backends. The operational reference for the example stacks stays [`examples/webserver_stacks/README.md`](../../examples/webserver_stacks/README.md).

## Layering

```
server/routers/*          thin, one module per surface (chat, runs, knowledge, files, ...)
  -> services/*           orchestration: AgentContextResolver, ChatService, RunService,
                          HealthService, KnowledgeService, FileService
    -> core/              AlgorithmRegistry, AgentAlgorithm, RunRequest/AgentResult, RunContext
      -> engines          research/web_research.py (research, direct_llm) and
                          knowledge/algorithm.py (knowledge)
```

`server/container.py:build_container` is the composition root — the single place where providers, strategies, settings, the algorithm registry, the auth provider, and the application services are assembled into one `AppContainer`. Services never read the environment (Constructor-First); the container hands them everything. `server/routes.py` is a thin registration facade kept for the legacy `register_routes(...)` signature; it builds the container and includes the split routers. `research/web_research.py` is the only module invoking `graph.run` for serving — tests monkeypatch `inqtrix.research.web_research.run_web_graph`.

`create_app(*, settings=None, providers=None, strategies=None)` is the single-stack factory (used by `python -m inqtrix`); `create_multi_stack_app(*, settings, stacks, default_stack)` mounts several `StackBundle`s in one process and adds `GET /v1/stacks` plus the per-request `body["stack"]` selector. Both call `build_auth_provider(settings)` and wire an ASGI lifespan that logs providers, security layers, and queue/concurrency configuration on startup (no silent fallbacks).

## Endpoints

### Meta and discovery (always registered, unauthenticated)

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness, active providers/models, `auth_required` + `auth_mode`, legal block. |
| `/v1/models` | GET | OpenAI-style model discovery (`research-agent`). |
| `/v1/capabilities` | GET | Feature manifest — see [the feature-gating contract](#v1capabilities-the-feature-gating-contract). |
| `/v1/stacks` | GET | Multi-stack deployments only; per-stack provider/model discovery (5-second cache). |

### Chat and editor (gated)

| Route | Method | Purpose |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible endpoint, streaming and non-streaming, optional `mode` field. |
| `/v1/text/improvements` | POST | LLM-backed improvement suggestions for browser text fields. |
| `/v1/editor/suggest` | POST | Rewrite for a selected/commented Markdown block (optional `attachments`). |
| `/v1/editor/instruct` | POST | Document-level editor instruction returning content-anchored edit proposals. |

### Native runs (gated)

| Route | Method | Purpose |
|---|---|---|
| `/v1/runs` | POST | Submit a run (`question`/`messages`, optional `mode`, `agent_overrides`, `knowledge_filters`, `workspace_id`). |
| `/v1/runs` | GET | List runs, filtered by workspace namespace when one is sent. |
| `/v1/runs/{run_id}` | GET | Current run summary. |
| `/v1/runs/{run_id}/events` | GET | Buffered + live SSE event stream — the UI contract. |
| `/v1/runs/{run_id}/result` | GET | Final result export after completion. |
| `/v1/runs/{run_id}/cancel` | POST | Cancel queued immediately; running runs stop at the next node boundary. |
| `/v1/runs/{run_id}` | DELETE | Permanently delete a terminal run (owner-only); 409 while still active. Revokes any shares on the run. |

### Knowledge (gated; registered only when `INQTRIX_KNOWLEDGE_ENABLED=true`)

| Route | Method | Purpose |
|---|---|---|
| `/v1/knowledge/collections` | POST / GET | Create (immutable embedding model) / list collections. |
| `/v1/knowledge/collections/{id}` | DELETE | Delete a collection with all documents. |
| `/v1/knowledge/collections/{id}/documents` | POST / GET | Ingest one document (`text` OR `file_id`) / list documents. |
| `/v1/knowledge/documents/{id}` | DELETE | Delete one document and its chunks. |
| `/v1/knowledge/documents/{id}/text` | GET | Full extracted text (the document viewer's source). |
| `/v1/knowledge/search` | POST | Synchronous retrieval search (debug/evaluation surface). |
| `/v1/sources/{document_id}` | GET | Citable source view — the target of HTTP knowledge citations. |

See [Knowledge engine](../knowledge/overview.md) for the data model, ingestion chain, and answer path.

### Files (gated; ObjectStore-backed)

| Route | Method | Purpose |
|---|---|---|
| `/v1/files` | POST / GET | Streamed upload (spooled, hash/size-accounted, `INQTRIX_MAX_FILE_BYTES` cap → 413) / list. |
| `/v1/files/{file_id}` | GET / DELETE | Metadata / delete. |
| `/v1/files/{file_id}/content` | GET | Streamed download (clients never see store URLs or credentials). |

### Project persistence (gated; durable when `features.project_persistence=true`)

| Route | Method | Purpose |
|---|---|---|
| `/v1/knowledge-session-groups` | GET | List Knowledge Desk folders for the caller/workspace. |
| `/v1/knowledge-session-groups/{group_id}` | PUT / DELETE | Upsert or delete one Knowledge Desk folder. Delete orphans member sessions to `group_id=null`. |
| `/v1/knowledge-sessions` | GET | List Knowledge Desk session metadata. Rows include `group_id`; `items_json` is excluded. |
| `/v1/knowledge-sessions/{session_id}` | GET / PUT / DELETE | Load, upsert, or delete one Knowledge Desk session. PUT accepts `title`, `items_json`, `created_at`, `updated_at`, and `group_id` (`string` or `null`). |

These routes are wired with the project-persistence tier's private
owner/workspace scoping. The volatile memory implementation is available for
offline/tests, but the frontend treats the tier as durable only when the
capability flag is true.

### Instance administration (session-admin gated)

| Route | Method | Purpose |
|---|---|---|
| `/v1/admin/system/runtime` | GET | Sanitized runtime categories for the System settings page: storage, run queue/execution, file store, knowledge backends, API documentation state, and read-only reachability booleans for optional backing services. |

### Auth (registered only when `INQTRIX_AUTH_MODE=oidc`)

| Route | Method | Purpose |
|---|---|---|
| `/api/auth/login` | GET | Start the authorization-code + PKCE flow at the IdP. |
| `/api/auth/callback` | GET | Code exchange, claim mapping, session cookie. |
| `/api/auth/session` | GET | Current session for the SPA. |
| `/api/auth/logout` | POST | End the session. |

### Test

`POST /v1/test/run` exists only when `TESTING_MODE=true` (used by `inqtrix-parity`). Never enable in production.

Registration is a composition decision, not a runtime fallback: `routes.py` includes the files router only when a `FileService` is wired (the default wiring always builds one), the knowledge and sources routers only when the knowledge engine is enabled, and the auth router only in `oidc` mode. A disabled feature has no routes (404), keeping the historical surface untouched.

## Authentication and TLS

Every request resolves to a `Principal` through the active `AuthProvider`; `/health`, `/v1/models`, `/v1/capabilities`, and `/v1/stacks` stay open for probes and pre-credential discovery.

| `INQTRIX_AUTH_MODE` | Effect |
|---|---|
| unset / `infer` (default) | Backwards-compatible inference: non-empty `INQTRIX_SERVER_API_KEY` means `apikey`, empty means `none`. |
| `none` | Open server, even when a key is configured (a WARNING makes the override visible). |
| `apikey` | Static Bearer gate (`Authorization: Bearer <key>`, timing-safe comparison); startup fails loudly without a key. |
| `oidc` | Browser login via the backend-for-frontend above: authorization code + PKCE against any OIDC-compliant IdP (Dex is the dev reference), opaque session cookie, CSRF-protected unsafe methods. Requires issuer, client id/secret, and `INQTRIX_SESSION_SECRET`. |

See [Auth modes](auth-modes.md) for the full mode reference and the OIDC variable set, and [Security hardening](security-hardening.md) for TLS, CORS, and the startup banner. TLS: setting both `INQTRIX_SERVER_TLS_KEYFILE` and `INQTRIX_SERVER_TLS_CERTFILE` makes the example stack scripts bind HTTPS (partial configuration raises at startup); `python -m inqtrix` does not wire TLS into uvicorn — terminate TLS at a reverse proxy or run a stack script.

## Modes and per-request overrides

`/v1/chat/completions` and `/v1/runs` accept a top-level `mode` resolved against the `AlgorithmRegistry`:

- `research` — the classify/plan/search/evaluate/answer web-research graph.
- `direct_llm` — straight to the active LLM provider (same path as the legacy `agent_overrides.skip_search=true`, which remains supported; conflicts return 400).
- `knowledge` — retrieval over knowledge collections, only registered when the engine is enabled. Scope and profile travel in `knowledge_filters` (see [Retrieval profiles](../configuration/knowledge-profiles.md)). On the chat-completions surface `stream=true` is rejected with 400 for this mode — the streamed chat path still executes the research graph directly; native runs are the streaming surface for knowledge.

`agent_overrides` is a strict whitelist (`extra="forbid"`, unknown keys → 400): `max_rounds`, `min_rounds`, `confidence_stop`, `report_profile`, `max_total_seconds`, `first_round_queries`, `skip_search`, `model_tier`, `model`, `effort`. `model_tier` selects among operator-configured tiers; `model`/`effort` pick a concrete model for the direct-chat answer only (the UI model picker) — research-node model names remain operator configuration. See [LLM calls, model tiers, and reasoning effort](../architecture/llm-calls.md) for tier wiring and `/health`'s `chat_model_options` discovery.

```json
{
  "model": "research-agent",
  "mode": "direct_llm",
  "messages": [{"role": "user", "content": "..."}],
  "agent_overrides": {"model_tier": "fast"}
}
```

## Native runs and the SSE event stream (the UI contract)

`POST /v1/runs` accepts a run into a FIFO queue and returns immediately; `GET /v1/runs/{run_id}/events` replays the buffered events (up to `RUN_EVENT_BUFFER_SIZE`, default 200) and then streams live named SSE events. This stream is the contract the React Research Desk renders from:

- Lifecycle: `inqtrix.run.queued`, `inqtrix.run.started`, `inqtrix.run.snapshot`, `inqtrix.run.cancel_requested`, `inqtrix.run.cancelled`, `inqtrix.run.failed`, `inqtrix.run.completed`.
- Progress: `inqtrix.node.started`, `inqtrix.progress.message`, `inqtrix.output_text.delta`.
- Knowledge steps (`mode=knowledge`): `inqtrix.knowledge.profile.resolved`, `decomposition.completed`, `retrieval.completed`, `evidence.truncated`, `gate.evaluated`, `grounding.checked`.

See [Run events](../observability/run-events.md) for payload schemas. With the in-memory store, terminal runs remain fetchable for `RUN_COMPLETED_TTL_SECONDS` (default 300); the Postgres backend makes records, events, and results durable.

Native UI clients should send a stable, non-secret workspace namespace via the `X-Inqtrix-Workspace-Id` header or a top-level `workspace_id` on `POST /v1/runs`. When present, the server stores it on the run record, filters `GET /v1/runs`, and requires the same namespace on the per-run routes. It is a browser/project routing namespace, not an auth boundary; omitting it preserves the unscoped behaviour for scripts.

`/v1/chat/completions` keeps the OpenAI-compatible SSE format (`stream=true`: progress chunks, `---` separator, answer chunks, `data: [DONE]`; `"include_progress": false` for answer-only). The React Research Desk consumes the native run API instead — see [React UI](react-ui.md) for `VITE_INQTRIX_API_BASE_URL` and deployment topologies.

## `/v1/capabilities`: the feature-gating contract

Clients discover features here instead of hardcoding them; the manifest is unauthenticated (the UI needs it before any credential prompt) and exposes only feature identity plus sanitized availability-derived booleans. Shape:

| Block | Content |
|---|---|
| `algorithms` | Manifest of registered algorithm ids (`research`, `direct_llm`, and `knowledge` when enabled) with their capability dicts. |
| `features` | Booleans: `knowledge`, `files`, `openapi`, `embedding_provider`; plus, when knowledge is on: `hybrid_retrieval`, `reranker`, `contextual_retrieval`, `document_parser`. Infrastructure-backed flags are false when the configured object store/vector store is not reachable. |
| `files` | `max_file_bytes` (when files are enabled). |
| `knowledge` | `default_embedding_model`, the annotated `embedding_catalog`, `default_top_k`, `default_profile`, `reranker_provider`, and `profiles[]` — the effective stage plan per retrieval profile including `degraded` stages, derived from the SAME ceiling instance the algorithm runs against. |

A capability flag never downgrades silently: a profile stage the operator ceiling forbids appears in `profiles[].degraded`, and unreachable S3/Qdrant backends clear the affected feature flags so the UI renders what would actually run.

## Storage, queue, and the worker

| Concern | Variable | Backends |
|---|---|---|
| Records (runs, identity, file metadata) | `INQTRIX_STORAGE_BACKEND` | `memory` (default, in-process) / `postgres` (`INQTRIX_DATABASE_URL`, migrated via `inqtrix-migrate`, RLS via `INQTRIX_DATABASE_APP_ROLE`). |
| Binary blobs (uploaded files) | `INQTRIX_OBJECT_STORE_BACKEND` | `local` (content-addressed under `INQTRIX_OBJECT_STORE_PATH`) / `s3` (any S3-compatible store; SeaweedFS in the dev stack). |
| Run execution | `INQTRIX_QUEUE_BACKEND` | `memory` (in-process, default) / `valkey` (Valkey Stream consumed by `inqtrix-worker` processes; requires `INQTRIX_STORAGE_BACKEND=postgres` — the run row is the source of truth, the stream only the dispatch channel). |
| Vectors (knowledge) | `INQTRIX_VECTOR_BACKEND` | `memory` / `qdrant` — see [Knowledge engine](../knowledge/overview.md). |

Contradictory combinations (`valkey` without postgres, `s3` without credentials, `postgres` without a URL) fail loudly at startup. The worker (`uv run inqtrix-worker`) executes runs with at-least-once delivery: `INQTRIX_WORKER_CONCURRENCY` bounds parallel runs per process, heartbeats keep long runs from being reclaimed, and `INQTRIX_WORKER_MAX_ATTEMPTS` dead-letters crash loops. The dev compose stack at `deploy/compose/compose.dev.yaml` provides `postgres`, `seaweedfs`, `qdrant`, and `valkey`, plus `dex` behind the `oidc` profile — see [Local infrastructure](../development/local-infrastructure.md).

When the knowledge engine is enabled, the same `inqtrix-worker` process also consumes a **second Valkey stream** (`inqtrix:index:jobs`, dead-letter `inqtrix:index:dead`) for background vector-index reindex (re-embed) jobs, using the identical fencing/heartbeat/reclaim/dead-letter machinery on a separate consumer group. This makes a reindex durable — it survives a server restart, not just closing the browser. Two consequences for operators: in queue mode (`postgres` + `valkey`) durable reindex **requires at least one running worker** (the API only persists and enqueues the job); and a worker started with knowledge disabled has no reindex consumer (logged at startup: `Knowledge-Engine deaktiviert — kein Reindex-Consumer`, never a silent no-op). Reindex sizing reuses `INQTRIX_REINDEX_*` (concurrency, queue size, TTL, per-collection history) and the worker mechanics reuse `INQTRIX_WORKER_*` — there are no separate indexing-worker knobs. Monitor the `inqtrix:index:dead` dead-letter stream alongside `inqtrix:runs:dead`.

The admin System page reads `GET /v1/admin/system/runtime` for the deployment
shape behind those switches. The endpoint is session-only and instance-admin
gated (personal access tokens cannot call it). It returns backend categories
such as `storage.backend`, `runs.queue`, `files.object_store`,
`knowledge.vector_store`, `knowledge.embedding_provider`, and `api.openapi`,
plus booleans such as `runs.queue_available`,
`files.object_store_available`, and `knowledge.vector_store_available`. These
checks are read-only pings/head requests and deliberately omit database URLs,
object-store paths, bucket names, service endpoints, and credentials.
`runs.execution=worker_dispatch` means runs are configured to dispatch through
Valkey to workers; it is not a live worker-count heartbeat, and it describes
the run path only — the reindex consumer rides the same
`INQTRIX_QUEUE_BACKEND=valkey` switch on its own stream.

## Concurrency and cancel

`MAX_CONCURRENT` (default 6) caps active `/v1/chat/completions` requests; a saturated semaphore returns 429 (no queueing on the OpenAI-compatible path). Native runs use `RUN_MAX_CONCURRENT` (falling back to `MAX_CONCURRENT`) for active workers and queue up to `RUN_QUEUE_MAX_SIZE` jobs (default 50) before `POST /v1/runs` returns 429. Background reindex jobs have their own pair, `INQTRIX_REINDEX_MAX_CONCURRENT` (default 6) and `INQTRIX_REINDEX_QUEUE_MAX_SIZE` (default 50). The caps are per surface, not a global provider limiter — worst case is their sum.

The research graph and document re-embedding are **synchronous work on a bounded thread pool**, not unbounded async coroutines: each active job holds one OS thread for its full duration. That is deliberate and fine for this app's scaling model — the work is I/O-bound on the LLM/embedding/search providers, the concurrency caps keep thread count small, and you scale *out* (more worker processes) rather than *up* (thousands of threads in one process). The practical ceiling on any single process is the upstream provider's rate limit and host CPU/RAM, not the thread model; rewriting the agent graph to async would be a large, risky change across every node, provider, and strategy and would not raise that ceiling (horizontal scaling does).

Streaming disconnects and `POST /v1/runs/{run_id}/cancel` set the same cancel event; execution stops at the next node boundary (typically 5-60 seconds, the remainder of the in-flight provider call). Queued runs cancel immediately. Hard cancel through in-flight provider calls is out of scope.

## Deployment sizing: with or without Valkey

Valkey and the worker are **optional** and not wired into any feature — they are confined to the job-dispatch tier (`runs/` + the worker entry point). Every function (research, knowledge retrieval, chat, editor, reindex) runs the same execution bodies whether dispatched in-process or to a worker. Three tiers, chosen by two switches:

| Mode | `INQTRIX_STORAGE_BACKEND` | `INQTRIX_QUEUE_BACKEND` | Execution | Durable across restart? | Multiple API replicas? |
|---|---|---|---|---|---|
| In-memory | `memory` | `memory` | in the API process | no (lost on restart) | no |
| **Postgres, no broker** | `postgres` | `memory` | in the API process | records yes; in-flight jobs fail visibly on restart | **no — single API process** |
| Postgres + Valkey | `postgres` | `valkey` | `inqtrix-worker` processes | yes (redelivery) | yes |

**Sizing guidance.** For a small-to-mid deployment (say up to ~100 users with bursty, occasional runs/reindexes), **`postgres` + `memory` on a single API process is the right, simpler choice**: durable records, no broker, no worker. Tune `MAX_CONCURRENT` / `INQTRIX_REINDEX_MAX_CONCURRENT` to your provider's rate limit and the host's CPU/RAM. Two constraints define this mode:

- **Bounded concurrency, then a queue.** Beyond the active caps, jobs wait (FIFO) and then return 429. Sustained simultaneous *long* runs are what saturates it, not raw user count.
- **Single API process.** In-process durable mode assumes ONE API process: on restart it marks in-flight rows `failed` (`server_restarted` — visible, not silent), and two replicas sharing one Postgres would sweep each other's in-flight jobs. Running more than one replica, or needing in-flight jobs to survive a restart, is the cue to switch on Valkey + the worker (`--profile workers`).

**When to add Valkey + worker** is therefore about *operational shape*, not a fixed headcount: (a) more than one API replica (HA / load balancing), (b) isolating long jobs from the request-serving process, or (c) in-flight jobs that must survive a restart. With the worker tier, per-worker parallelism is `INQTRIX_WORKER_CONCURRENCY` and you scale by running more worker replicas; the API-side `MAX_CONCURRENT` / `REINDEX_MAX_CONCURRENT` then govern admission only.

The server is stateless for research data: each request runs a fresh `AgentState`; prior messages are formatted into the `history` string, but citations, evidence, and confidence are not carried over.

## Quick check

```bash
curl -N http://localhost:5100/v1/chat/completions \
    -H "Authorization: Bearer dev-secret-xxxxx" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "research-agent",
        "messages": [{"role": "user", "content": "Was ist der Stand der GKV-Reform?"}],
        "stream": true
    }'
```

A missing or wrong Bearer header returns 401 with `WWW-Authenticate: Bearer`. `INQTRIX_ENABLE_OPENAPI=true` additionally serves `/openapi.json`, `/docs`, and `/redoc` (off by default, preserving the historical no-schema surface).

## Related docs

- [Auth modes](auth-modes.md) — `none` | `apikey` | `oidc` in full, including the OIDC BFF variables.
- [Knowledge engine](../knowledge/overview.md) — collections, ingestion, the gated answer path.
- [Settings and environment](../configuration/settings-and-env.md) — every server/storage/queue variable with defaults.
- [Retrieval profiles](../configuration/knowledge-profiles.md) — `knowledge_filters.profile` and the stage matrix.
- [Run events](../observability/run-events.md) — SSE payload schemas for native runs.
- [Examples README](../../examples/webserver_stacks/README.md) — per-stack env tables and run commands.
- [React UI](react-ui.md) — the bundled frontend.
- [Security hardening](security-hardening.md) — TLS, Bearer, CORS, startup banner.
- [Local infrastructure](../development/local-infrastructure.md) — the dev compose stack.
