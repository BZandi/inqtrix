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
| `/health` | GET | Liveness, active providers/models, `auth_required` + `auth_mode`, `legal` block, `ai_disclosure` block (see [AI transparency](../reference/ai-transparency.md)). |
| `/readyz` | GET | Readiness for load balancers: 503 while the database or queue is unreachable (bounded 2s probes), 200 `degraded` when only the vector store is down (knowledge fails per-request, everything else serves). Memory backends answer `ready`/`skipped`. The Helm chart keys its `readinessProbe` on this; liveness stays on `/health`. |
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
| `/v1/runs` | POST | Submit a run (`question`/`messages`, optional `mode`, `agent_overrides`, `knowledge_filters`, `workspace_id`; agent runs also accept `source_policy` and `execution_directive`). |
| `/v1/runs` | GET | List runs, filtered by workspace namespace when one is sent. |
| `/v1/runs/{run_id}` | GET | Current run summary. |
| `/v1/runs/{run_id}/events` | GET | Buffered + live SSE event stream — the UI contract. |
| `/v1/runs/{run_id}/result` | GET | Final result export after completion; Agent Desk results include the canonical effective `execution` block. |
| `/v1/runs/{run_id}/cancel` | POST | Cancel the complete run tree: queued/waiting rows become terminal immediately; running rows stop at the next node boundary. |
| `/v1/runs/{run_id}` | DELETE | Permanently delete a terminal run (owner-only); 409 while still active. Revokes any shares on the run. |

### Agent memory (gated; personal, when the agent platform is mounted)

| Route | Method | Purpose |
|---|---|---|
| `/v1/agent/memory` | GET | List personal accepted memories; optional `?q=&scope=&limit=` searches the same personal namespace via recall. |
| `/v1/agent/memory/{memory_id}` | PATCH / DELETE | Edit or delete one memory. Foreign ids are indistinguishable 404s. |
| `/v1/agent/memory:clear` | POST | Clear all personal memories, optionally one `scope`. |
| `/v1/agent/memory/candidates` | GET | List personal memory candidates staged by completed agent runs. |
| `/v1/agent/memory/candidates/{id}:accept` | POST | Accept a candidate, optionally with edited `content`. |
| `/v1/agent/memory/candidates/{id}:reject` | POST | Reject a candidate. |
| `/v1/agent/memory/feedback` | GET | Read personal run-feedback history; optional `?run_id=&limit=`. |
| `/v1/agent/runs/{run_id}/feedback` | POST | Store personal run feedback: `feedback=positive|negative|neutral`, optional `reason`, optional owner-checked `memory_id`. |

The memory routes never accept client owner fields (`sub`, `tenant_id`,
`user_id`, `owner`, `namespace`, ...). Ownership is derived from the
authenticated principal; anonymous/static-key principals cannot use
long-term memory.

### Knowledge (gated; registered only when `INQTRIX_KNOWLEDGE_ENABLED=true`)

| Route | Method | Purpose |
|---|---|---|
| `/v1/knowledge/collections` | POST / GET | Create (immutable embedding model) / list collections. |
| `/v1/knowledge/collections/{id}` | DELETE | Start the durable collection-and-vector deletion; HTTP 202 returns a `DeletionOperation`. |
| `/v1/knowledge/collections/{id}/document-revisions` | POST | Reserve an immutable text/asset revision and return its server-owned indexing job (HTTP 202). |
| `/v1/knowledge/collections/{id}/documents` | POST / GET | Compatibility ingestion (`text` OR raw `file_id`) over the same revision job / list documents. |
| `/v1/knowledge/documents/{id}` | DELETE | Start the durable document-and-vector deletion; HTTP 202 returns a `DeletionOperation`. |
| `/v1/knowledge/documents/{id}/text` | GET | Full extracted text (the document viewer's source). |
| `/v1/knowledge/documents/{id}/chunks/{chunk_index}` | GET | One chunk plus up to `?context=0..3` neighbour chunks per side (the evidence view). |
| `/v1/knowledge/search` | POST | Synchronous retrieval search (debug/evaluation surface). |
| `/v1/sources/{document_id}` | GET | Citable source view — the target of HTTP knowledge citations. |
| `/v1/knowledge/indexing-jobs*` | GET / POST | Read, cancel, resume, explicitly resume without contextualization, and stream collection-generation or document-revision jobs. |

See [Knowledge engine](../knowledge/overview.md) for the data model, ingestion chain, and answer path.

### Files (gated; ObjectStore-backed)

| Route | Method | Purpose |
|---|---|---|
| `/v1/files` | POST / GET | Streamed upload (spooled, hash/size-accounted, `INQTRIX_MAX_FILE_BYTES` cap → 413) / list. A normal bound upload also returns its durable upload operation and server-owned parse state. |
| `/v1/files/{file_id}` | GET / DELETE | Metadata / start deletion through the linked asset lifecycle. |
| `/v1/files/{file_id}/content` | GET | Streamed download (clients never see store URLs or credentials). |
| `/v1/uploads` / `/v1/uploads/{operation_id}` | GET | List/read durable bound-upload and parse checkpoints. |
| `/v1/uploads/{operation_id}/retry` | POST | Retry the same recoverable upload operation; missing bytes are requested explicitly. |

### Project persistence (gated; durable when `features.project_persistence=true`)

| Route | Method | Purpose |
|---|---|---|
| `/v1/knowledge-session-groups` | GET | List Knowledge Desk folders for the caller/workspace. |
| `/v1/knowledge-session-groups/{group_id}` | PUT / DELETE | Upsert or delete one Knowledge Desk folder. Delete orphans member sessions to `group_id=null`. |
| `/v1/knowledge-sessions` | GET | List Knowledge Desk session metadata. Rows include `group_id`; `items_json` is excluded. |
| `/v1/knowledge-sessions/{session_id}` | GET / PUT / DELETE | Load, upsert, or delete one Knowledge Desk session. PUT accepts `title`, `items_json`, `created_at`, `updated_at`, and `group_id` (`string` or `null`). |
| `/v1/agent-session-groups*` | GET / PUT / DELETE | List and manage private Agent Desk session folders. |
| `/v1/agent-sessions` | GET | List Agent Desk session metadata; `items_json` is excluded so the session body remains load-on-open. |
| `/v1/agent-sessions/{session_id}` | GET / PUT / DELETE | Load, upsert, or delete one Agent Desk session. The UI-owned `items_json` includes the session source policy; PUT also carries `title`, `created_at`, `updated_at`, and optional `group_id`. |
| `/v1/assets/{asset_id}` | GET / PUT / DELETE | Read/upsert a file-library asset or start its aggregate deletion. The DELETE response is HTTP 202 and is not terminal. |
| `/v1/assets/deletion-operations` | POST / GET | Start a stable bulk manifest or list retained deletion receipts for recovery after reload. |
| `/v1/deletion-operations/{operation_id}` | GET | Read authoritative progress for asset, group, section, vector-index, knowledge-document, or knowledge-collection deletion. |
| `/v1/deletion-operations/{operation_id}/retry` | POST | Resume the same failed cleanup manifest (HTTP 202). |
| `/v1/vector-indexes/{index_id}` | DELETE | Fence the index and its backing collection, then start the same durable deletion lifecycle. |
| `/v1/editor/documents/{document_id}` | PUT | Autosave upsert of one editor document. The save must carry a STRICTLY newer `revision` than the stored row (monotonic guard; debounced multi-edit flushes legitimately jump several revisions) — a stale-or-equal counter answers 409 `conflict` with `current_revision`, and the client refetches and rebases instead of silently overwriting a concurrent writer (e.g. an applied agent patch). |
| `/v1/editor/documents/{document_id}/patches` | GET | Patch metadata of one editor document (no edit bodies), newest first; `?status=pending\|accepted\|rejected` filters. |
| `/v1/editor/patches/{patch_id}` | GET | One patch with its anchored edits, warnings, and the document's CURRENT revision. |
| `/v1/editor/patches/{patch_id}:apply` | POST | Apply a pending patch server-side. Body `{"expected_revision": int}` (CAS against the document); stale revision answers 409 with `current_revision` + `revision_before`; replaying the same apply answers 200 with the stored outcome. |
| `/v1/editor/patches/{patch_id}:reject` | POST | Reject a pending patch. Body `{"note"?: string}`; already-applied answers 409, a reject replay 200. |

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
| `/api/auth/session` | GET | Current session for the SPA; a valid session also refreshes the readable CSRF double-submit cookie. |
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
- `knowledge` — retrieval over knowledge collections, only registered when the engine is enabled. Scope and profile travel in `knowledge_filters` (see [Retrieval profiles](../configuration/knowledge-profiles.md)). Streaming (`stream=true`) is supported: streamed chat dispatches through the `AlgorithmRegistry` like every other mode, so the cited answer streams as content chunks. The granular progress (gate rounds, grounding) is emitted only on the native `/v1/runs` SSE surface — the chat-completions stream carries the answer, not the progress narration.
- `workspace_agent` — the staged Mission engine with plan, approval,
  execution, synthesis, and optional canvas/editor delivery. It is registered
  only when the agent-platform durability gate passes.
- `agent_kernel` — the automatic conversational front door. It may answer
  directly, use instant tools, or delegate a Mission. Registration additionally
  requires the kernel rollout gate and a tool-calling-capable provider.

`agent_overrides` is a strict whitelist (`extra="forbid"`, unknown keys → 400): `max_rounds`, `min_rounds`, `confidence_stop`, `report_profile`, `max_total_seconds`, `first_round_queries`, `skip_search`, `model_tier`, `model`, `effort`. `model_tier` selects among operator-configured tiers; `model`/`effort` pick a concrete model for direct chat and the Agent Desk's scoped thinking nodes. In the kernel this includes the conversation turn and both quick-web LLM calls; in a Mission it includes planning, synthesis, and answer nodes while assembly-line classifiers stay on the operator tier map. Ordinary research-graph node names remain operator configuration. See [LLM calls, model tiers, and reasoning effort](../architecture/llm-calls.md) for tier wiring and `/health`'s `chat_model_options` discovery.

```json
{
  "model": "research-agent",
  "mode": "direct_llm",
  "messages": [{"role": "user", "content": "..."}],
  "agent_overrides": {"model_tier": "fast"}
}
```

Agent-native runs may add a session-selected source policy and a one-message
enforced route:

```json
{
  "question": "Who won today's final?",
  "mode": "agent_kernel",
  "source_policy": {"web": "available", "knowledge": "available"},
  "execution_directive": "quick_web"
}
```

Each source value is `available` or `disabled`; omission means both available.
`execution_directive` is `quick_web` (one `web.search.instant` call, no plan,
child, RAG, or canvas) or `knowledge_only` (project-knowledge reads only).
Either route forces `agent_kernel`, `response_form=chat`, and normal depth for
that one message. An explicit model and effort in `agent_overrides` still apply
to quick-web query preparation and grounded answer synthesis. The legacy
`tool_directives` hint surface remains compatible on its own, but sending both
fields is HTTP 400. Directives also reject `document_id`; a requested route
whose capability is not published fails instead of falling back.

Source enforcement is common to the kernel, phase-machine planner/executor,
and child submission. Effective precedence is deployment/identity/strict and
write gates, activated skill restrictions, one-shot directive, session source
policy, then automatic model choice.

Mission plan tasks use one server-validated web contract. A normal plan uses
`web_instant`; its `queries` array must contain exactly one self-contained
question, and that question becomes exactly one capability call. Independent
tasks in the same dependency wave may execute concurrently. `web_research`
means one delegated child whose strings are joint guidance questions. It is
accepted only for Deep, an admitted `tool_directives=["web_research"]`, or a
user-edited plan; the child profile is server-selected as `compact` or `deep`.
Task `params.recency` accepts only `day`, `week`, `month`, `year`, or omission.
Task resource budgets are deployment-owned: non-empty `budget` objects on new
or edited plans fail with `task_budget_server_managed` instead of silently
changing execution limits.

Completed Agent Desk results expose one evidence shape through
`result.references`. Each entry retains its stable `K#`/`W#` label, canonical
URL or knowledge identity, exact `excerpt`/`source_text` when available, and
optional `grounded_support`. The latter is bounded provider-answer context for
a cited web URL, not a quote from the linked page; clients must label it
accordingly. `report_references` remains the canonical internal state key, with
the older Agent `references` key accepted only as an export compatibility
alias.

## Native runs and the SSE event stream (the UI contract)

`POST /v1/runs` accepts a run into a FIFO queue and returns immediately; `GET /v1/runs/{run_id}/events` replays the buffered events (up to `RUN_EVENT_BUFFER_SIZE`, default 200) and then streams live named SSE events. This stream is the contract the React Research Desk renders from:

- Lifecycle: `inqtrix.run.queued`, `inqtrix.run.started`, `inqtrix.run.resumed`, `inqtrix.run.snapshot`, `inqtrix.run.cancel_requested`, `inqtrix.run.cancelled`, `inqtrix.run.failed`, `inqtrix.run.completed`.
- Progress and answer publication: `inqtrix.node.started`, `inqtrix.progress.message`, `inqtrix.answer.started`, `inqtrix.output_text.delta`, `inqtrix.answer.ready`, `inqtrix.answer.interrupted`.
- Knowledge steps (`mode=knowledge`): `inqtrix.knowledge.profile.resolved`, `decomposition.completed`, `retrieval.completed`, `evidence.truncated`, `gate.evaluated`, `grounding.checked`.
- Agent execution facts: every state-bearing agent snapshot exposes the
  canonical `execution` block; kernel tool events report actual usage, and
  quick-web emits one `web_instant` started/finished pair. See [Run
  events](../observability/run-events.md) for the block and consent semantics.

See [Run events](../observability/run-events.md) for payload schemas. With the in-memory store, terminal runs remain fetchable for `RUN_COMPLETED_TTL_SECONDS` (default 300); the Postgres backend makes records, events, and results durable.

Native UI clients should send a stable, non-secret workspace namespace via the `X-Inqtrix-Workspace-Id` header or a top-level `workspace_id` on `POST /v1/runs`. When present, the server stores it on the run record, filters `GET /v1/runs`, and requires the same namespace on the per-run routes. It is a browser/project routing namespace, not an auth boundary; omitting it preserves the unscoped behaviour for scripts.

`/v1/chat/completions` keeps the OpenAI-compatible SSE format (`stream=true`: progress chunks, `---` separator, answer chunks, `data: [DONE]`; `"include_progress": false` for answer-only). The React Research Desk consumes the native run API instead — see [React UI](react-ui.md) for `VITE_INQTRIX_API_BASE_URL` and deployment topologies.

## `/v1/capabilities`: the feature-gating contract

Clients discover features here instead of hardcoding them; the manifest is unauthenticated (the UI needs it before any credential prompt) and exposes only feature identity plus sanitized availability-derived booleans. Shape:

| Block | Content |
|---|---|
| `algorithms` | Manifest of registered algorithm ids (`research`, `direct_llm`, plus `knowledge`, `workspace_agent`, and `agent_kernel` when their gates pass) with their capability dicts. |
| `features` | Booleans: `knowledge`, `files`, `openapi`, `embedding_provider`; plus, when knowledge is on: `hybrid_retrieval`, `reranker`, `contextual_retrieval`, `document_parser`. Infrastructure-backed flags are false when the configured object store/vector store is not reachable. |
| `files` | `max_file_bytes` (when files are enabled). |
| `knowledge` | `default_embedding_model`, the annotated `embedding_catalog`, `default_top_k`, `evidence_k_max` (the final-evidence ceiling that bounds a `final_k` override), `default_profile`, `reranker_provider`, and `profiles[]` — the effective stage plan per retrieval profile including each profile's `final_k_factor` and its `degraded` stages, derived from the SAME ceiling instance the algorithm runs against. |
| `agent` | Agent modes, permission/depth presets, tools, and the source-routing surface. `source_controls[]` publishes `web`/`knowledge` entries with `default` and effective `available`; `execution_directives[]` publishes `quick_web`/`knowledge_only` with effective `available`. Clients must not render a server-disabled route as selectable. |

A capability flag never downgrades silently: a profile stage the operator ceiling forbids appears in `profiles[].degraded`, and unreachable S3/Qdrant backends clear the affected feature flags so the UI renders what would actually run.

## Storage, queue, and the worker

| Concern | Variable | Backends |
|---|---|---|
| Records (runs, identity, file metadata) | `INQTRIX_STORAGE_BACKEND` | `memory` (default, in-process) / `postgres` (`INQTRIX_DATABASE_URL`, migrated via `inqtrix-migrate`, RLS via `INQTRIX_DATABASE_APP_ROLE`). |
| Binary blobs (uploaded files) | `INQTRIX_OBJECT_STORE_BACKEND` | `local` (content-addressed under `INQTRIX_OBJECT_STORE_PATH`) / `s3` (any S3-compatible store; SeaweedFS in the dev stack). |
| Run execution | `INQTRIX_QUEUE_BACKEND` | `memory` (in-process, default) / `valkey` (Valkey Stream consumed by `inqtrix-worker` processes; requires `INQTRIX_STORAGE_BACKEND=postgres` — the run row is the source of truth, the stream only the dispatch channel). |
| Vectors (knowledge) | `INQTRIX_VECTOR_BACKEND` | `memory` / `qdrant` — see [Knowledge engine](../knowledge/overview.md). |

Contradictory combinations (`valkey` without postgres, `s3` without credentials, `postgres` without a URL) fail loudly at startup. The worker (`uv run inqtrix-worker`, or `python -m inqtrix.worker` after a normal pip install) executes runs with at-least-once delivery: `INQTRIX_WORKER_CONCURRENCY` bounds parallel runs per process, heartbeats keep long runs from being reclaimed, and `INQTRIX_WORKER_MAX_ATTEMPTS` dead-letters crash loops. One Valkey queue instance is injected into both the Postgres run store and worker loop, so child submission, child-terminal parent wake, resume, and ordinary dispatch share the same path. Claiming a durable run row and dispatching its stream message remain separate operations: a successor message is held unacknowledged until the older delivery is acknowledged, which prevents a fast resume from being mistaken for a duplicate and dropped. Host-side development selects services from the canonical stack and adds only `deploy/compose/compose.dev-ports.yaml` for the `inqtrix-dev` project name and loopback ports; `dex` remains behind the `oidc` profile — see [Local infrastructure](../development/local-infrastructure.md).

When the knowledge engine is enabled, the same `inqtrix-worker` process also
consumes the dedicated indexing stream (`inqtrix:index:jobs`, dead-letter
`inqtrix:index:dead`) for both collection-generation builds and individual
document revisions. The durable Postgres job owns checkpoints, pause/cancel
state, and publication fencing; Valkey is dispatch only. A worker started with
knowledge disabled has no indexing consumer and logs that state explicitly.
Index admission/history uses `INQTRIX_REINDEX_*`; delivery, reclaim, and
dead-letter mechanics reuse `INQTRIX_WORKER_*`.

Two further streams carry recovery work that must not be represented as a
finished browser mutation: aggregate deletion uses
`inqtrix:deletion:jobs`/`inqtrix:deletion:dead`, and bound upload/finalization
uses `inqtrix:upload:jobs`/`inqtrix:upload:dead`. Their Postgres rows remain the
source of truth, and the same worker process starts the corresponding
consumers. Operators should monitor these dead-letter streams alongside
`inqtrix:index:dead` and `inqtrix:runs:dead`.

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
the run path only. Indexing, deletion, and upload recovery use their own
streams under the same durable queue deployment.

## Concurrency and cancel

`MAX_CONCURRENT` (default 6) caps active `/v1/chat/completions` requests; a saturated semaphore returns 429 (no queueing on the OpenAI-compatible path). Native runs use `RUN_MAX_CONCURRENT` (falling back to `MAX_CONCURRENT`) for active workers and queue up to `RUN_QUEUE_MAX_SIZE` jobs (default 50) before `POST /v1/runs` returns 429. Background reindex jobs have their own pair, `INQTRIX_REINDEX_MAX_CONCURRENT` (default 6) and `INQTRIX_REINDEX_QUEUE_MAX_SIZE` (default 50). The caps are per surface, not a global provider limiter — worst case is their sum.

The research graph and document re-embedding are **synchronous work on a bounded thread pool**, not unbounded async coroutines: each active job holds one OS thread for its full duration. That is deliberate and fine for this app's scaling model — the work is I/O-bound on the LLM/embedding/search providers, the concurrency caps keep thread count small, and you scale *out* (more worker processes) rather than *up* (thousands of threads in one process). The practical ceiling on any single process is the upstream provider's rate limit and host CPU/RAM, not the thread model; rewriting the agent graph to async would be a large, risky change across every node, provider, and strategy and would not raise that ceiling (horizontal scaling does).

Streaming disconnects and `POST /v1/runs/{run_id}/cancel` set the same cancel event; execution stops at the next safe boundary (the current logical provider operation may use up to its configured 600-second budget). Explicit cancellation locks and transitions the complete descendant tree, then reconciles every affected Agent plan task from the terminal run rows. A read of an older partially reconciled plan repeats that idempotent settlement, covering a process failure after the run transaction committed. Queued and waiting rows cancel immediately. Agent Desk additionally exposes `POST /v1/runs/{run_id}/tasks/{task_id}/cancel`: pending work ends immediately, research children reuse run cancellation, and an in-flight synchronous instant request records `cancel_requested`, performs no retry or result commit, and ends naturally. Hard network interruption of that synchronous provider request is out of scope.

## Deployment sizing: with or without Valkey

Valkey and the worker are optional execution infrastructure, not an alternate
product implementation. They dispatch the same run, indexing, deletion, and
upload-recovery bodies whose durable records live in Postgres. Knowledge
retrieval, ordinary chat, and editor reads remain normal request paths. Three
tiers are chosen by the storage and queue switches:

| Mode | `INQTRIX_STORAGE_BACKEND` | `INQTRIX_QUEUE_BACKEND` | Execution | Durable across restart? | Multiple API replicas? |
|---|---|---|---|---|---|
| In-memory | `memory` | `memory` | in the API process | no (lost on restart) | no |
| **Postgres, no broker** | `postgres` | `memory` | in the API process | records and paused checkpoints yes; queued/running/cancelling closures fail visibly on restart; paused indexing work is reconstructed on explicit resume | **no — single API process** |
| Postgres + Valkey | `postgres` | `valkey` | `inqtrix-worker` processes | yes (redelivery) | yes |

**Sizing guidance.** For a small-to-mid deployment (say up to ~100 users with bursty, occasional runs/reindexes), **`postgres` + `memory` on a single API process is the right, simpler choice**: durable records, no broker, no worker. Tune `MAX_CONCURRENT` / `INQTRIX_REINDEX_MAX_CONCURRENT` to your provider's rate limit and the host's CPU/RAM. Two constraints define this mode:

- **Bounded concurrency, then a queue.** Beyond the active caps, jobs wait (FIFO) and then return 429. Sustained simultaneous *long* runs are what saturates it, not raw user count.
- **Single API process.** In-process durable mode assumes ONE API process: on restart it marks lost queued/running/cancelling closures `failed` (`server_restarted` — visible, not silent), while durable `paused_dependency`/`paused_validation` rows and checkpoints remain paused. Explicit resume reconstructs the operation from its canonical document/revision or generation identity before queueing it. Two replicas sharing one Postgres would still sweep each other's in-flight closures. Running more than one replica, or needing actively executing jobs to survive a restart, is the cue to switch on Valkey + the worker (`--profile workers`).

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
- [AI transparency](../reference/ai-transparency.md) — the `ai_disclosure` block and how generated output is marked.
- [Local infrastructure](../development/local-infrastructure.md) — the dev compose stack.
