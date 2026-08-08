# Security hardening

> Files: `src/inqtrix/server/security.py`, `src/inqtrix/server/routes.py`, `src/inqtrix/auth/api_key.py`

## Scope

Opt-in hardening layers the HTTP server ships with: TLS at the uvicorn layer, Bearer API-key authentication, and CORS allow-listing. All three are off by default and driven by `ServerSettings`. Authentication is mode-driven (`none` | `apikey` | `local` | `oidc` | `ldap`); this page covers the `apikey` gate and the security-relevant deployment boundaries — for the full mode matrix, the inference rule, and browser sessions see [Authentication modes](auth-modes.md). This is not a substitute for a real reverse-proxy-fronted production stack — it is the minimum viable hardening for self-hosted experimentation.

> Inqtrix is explicitly experimental. The root `README.md` disclaimer applies. Secure configuration, hardening, deployment architecture, access control, secret handling, and compliance remain the operator's responsibility.

## The three layers

### TLS termination in uvicorn

Set both `INQTRIX_SERVER_TLS_KEYFILE` and `INQTRIX_SERVER_TLS_CERTFILE` to local PEM file paths. Both are required; setting only one raises `RuntimeError` on startup (no silent fallback).

The values are passed to `uvicorn.run(..., ssl_keyfile=..., ssl_certfile=...)`. For production, prefer terminating TLS at a reverse proxy (nginx, Traefik, Azure Application Gateway) whenever possible — built-in TLS exists for small self-hosted setups where a proxy would be overkill.

Editor guest links require an HTTPS `INQTRIX_PUBLIC_BASE_URL` at startup (guest tokens and passwords travel with every request). For plain-HTTP local development only, `INQTRIX_EDITOR_GUEST_LINKS_ALLOW_INSECURE_HTTP=true` converts the hard startup failure into a loud startup WARNING and drops the `Secure` flag from the guest cookies so browsers accept them over http. The switch never covers production, and the Level-3 release verification of guest access keeps running against HTTPS regardless.

### Bearer API key

Set `INQTRIX_SERVER_API_KEY` to a random string. With `INQTRIX_AUTH_MODE` unset, a non-empty key selects the `apikey` mode (the inference rule in [Authentication modes](auth-modes.md)). The server then enforces the Bearer gate on every principal-gated route — `/v1/chat/completions`, `/v1/text/improvements`, `/v1/editor/*`, `/v1/runs*`, `/v1/test/run`, and, when their feature gates register them, `/v1/files*`, `/v1/knowledge/*`, and `/v1/sources/*`:

```http
Authorization: Bearer <key>
```

Comparison uses `hmac.compare_digest` (timing-safe). `/health`, `/v1/models`, and `/v1/capabilities` remain open for liveness and discovery; `/health` also carries the project legal/source metadata used by UIs. `/v1/stacks` (multi-stack apps) also stays open because UIs need it before prompting for credentials.

### CORS allow-list

Set `INQTRIX_SERVER_CORS_ORIGINS` to a comma-separated list of origins:

```
INQTRIX_SERVER_CORS_ORIGINS=https://ui.example.com,https://admin.example.com
```

- `*` is accepted but WARNs on startup, because browsers reject `Access-Control-Allow-Origin: *` together with `allow_credentials=True`.
- The built-in policy allows `Authorization` and `Content-Type` headers so Bearer tokens pass through correctly.

### Editor collaboration WebSocket

The optional editor transport has a separate, fail-closed boundary. Browsers
connect only to same-origin `/collaboration`; the Vite development proxy or
selected production web adapter (Python by default, nginx only when explicitly
selected) forwards it to FastAPI, which validates `Origin` and relays binary
frames to private Node. `INQTRIX_COLLABORATION_ALLOWED_ORIGINS` adds explicit
origins when TLS termination or a trusted external frontend makes derived
same-origin insufficient. CORS configuration does not replace this WebSocket
Origin check.

Node must have no public host port or database credentials. FastAPI and Node
authenticate their private HTTP/WebSocket calls with an independent
`INQTRIX_COLLABORATION_SECRET`; it must not reuse the session secret or enter a
URL, log, metric, ConfigMap, or browser bundle. Document leases are available
only to cookie-backed `local`/`ldap`/`oidc` sessions, expire quickly, and are
rechecked against the live share/account/session state.

The gateway and coordinator enforce binary-only frames, size limits, session
caps, update/awareness rates, schema/generation matching, and a single active
writer fencing epoch. A validation, persistence, or policy failure changes the
editor to read-only or closes it; it never enables offline writes or falls back
to Markdown autosave. See [Deploy editor collaboration](editor-collaboration.md#security-and-limits)
for defaults and close codes.

## What is not covered

All of these are explicit non-goals for this family of settings. They are tracked as follow-up tasks:

- Per-IP or per-key rate limiting.
- Request tracing / correlation IDs (basic logging covers it, but distributed tracing does not).
- Multi-key rotation lists.
- Selfsigned cert helpers (`mkcert`, `openssl req -x509 ...`) — there is no built-in generator.

If any of these is a hard requirement, a reverse proxy or an API gateway in front of Inqtrix is the recommended route.

Covered elsewhere: OIDC browser sessions are a first-class auth mode ([Authentication modes](auth-modes.md)); durable run persistence across restarts is provided by `INQTRIX_STORAGE_BACKEND=postgres` together with the `valkey` queue backend ([Settings and env](../configuration/settings-and-env.md)).

## Operator-visible behaviour

The lifespan log records the active layers on startup in a single line:

```
Inqtrix server starting | llm=... ready=True | search=... ready=True | report_profile=compact | max_concurrent=3 | run_max_concurrent=3 | run_queue_max_size=50 | run_completed_ttl_seconds=300 | api_key_gate=on | auth_mode=apikey | cors=on
```

`api_key_gate` reflects whether the resolved auth mode is `apikey`; `auth_mode` names the resolved mode (`none` | `apikey` | `local` | `oidc` | `ldap`). Any deviation from the expected line is a deployment mistake; fail the smoke test fast.

## Identity, authorization, and revocation boundaries

Multi-user deployments use the local `users.id` UUID as the only authority
identifier. OIDC/LDAP issuer and subject values are authentication-adapter
provenance, not resource ids. Session and PAT verification resolves the user
and checks that the account is still active on every request. Disabling a user
revokes their sessions and PATs; enabling the user later does not reactivate
those credentials.

Resource access is owner-or-direct-share. Accepted shares for runs, knowledge
collections, prompt templates, and skill templates remain `view`/`edit`.
Collaboration editor documents additionally permit `suggest`; this does not
expand the permission vocabulary of other resource types. Workspace roles do
not imply resource rights, and files are owner-bound. Services re-check the
resource owner, active user, current share, permission, and optional
common-workspace restriction at the access boundary.
Long-running work repeats those checks at safepoints, and run, indexing, and
user SSE streams re-check immediately before each data frame. This bounds a
revocation without claiming that bytes already sent to a browser or an
external provider can be recalled.

Postgres mutations that combine authorization, a resource write, audit, and
cache invalidation perform them in one transaction. Security invariants use
targeted row locks: a tenant security row serializes first-admin and
last-admin decisions; a workspace row serializes last-owner changes; resource
and direct-share rows follow one resource-then-share lock order. Operators
should run the PostgreSQL-backed race tests against a disposable database as
part of release qualification; an offline suite that skips those tests does
not validate these concurrency guarantees.

## Database migration and object-store identities

Production PostgreSQL uses a restricted runtime path and a separate direct
migration identity. The migration Secret is available only to the one-shot job;
API/worker must never receive it, and Alembic must not traverse PgBouncer
transaction pooling. `bypass` mode requires a dedicated `BYPASSRLS` login;
`owner` mode requires table ownership, an exclusive maintenance boundary and
an explicit operator assertion. Runtime readiness rejects a stale Alembic head,
a failed `SET ROLE`, or an effective role that is superuser, `BYPASSRLS`, or an
RLS-table owner. The effective role also needs schema `USAGE` without `CREATE`,
the exact DML set without RLS-bypassing `TRUNCATE`, explicit execution of the
tenant-policy function, `USAGE`-only identity sequences and `SELECT`-only access
to the active schema's revision table. Runtime identities may not own the
database or receive database/schema `CREATE`, own, inherit or assume the owners
of those dependencies, nor assume a helper role with forbidden DDL, grant, sequence,
table or column rights. PUBLIC table/column grants and canonical app-role column
ACLs are rejected; unrelated named reporting roles remain operator policy. The
live probe also requires enabled/active RLS and a writable transaction. Missing
dependencies, read-replica DSNs or excess grants keep API and workers not ready.
See [Database migrations](database-migrations.md).

Managed S3 credentials follow the same least-privilege split. Static keys and
STS tokens are injected only into API/worker; workload identity uses only their
component ServiceAccounts. Web, Collaboration, migration and smoke-test pods
remain tokenless and receive neither the S3 Secret nor CA bundle. TLS
verification cannot be disabled, and uploads set no object ACL. See [Object
storage](object-storage.md).

The migration DSN follows a stricter split: Helm rejects reuse of the runtime
application Secret for `migrations.databaseSecret`, and API/worker explicitly
override `INQTRIX_MIGRATION_DATABASE_URL` to an empty value as defense in depth.
For `serviceAccount.create=false`, provide an existing unannotated internal
ServiceAccount name; component API/worker accounts remain the only place for
cloud-identity annotations.

## v0.2 hard-cut upgrade

The canonical-identity/direct-sharing migration is intentionally destructive:
there is no dual-read, dual-write, or mixed-version compatibility window. API,
worker, frontend, and migrations must move together during a maintenance
window.

Install the matching source tree before running host-side migration commands:
use `uv sync --extra dev`, or create and activate a normal virtual environment
and run `python -m pip install -e ".[dev]"`. The commands below show both
execution forms.

1. Stop every API and worker process so no old binary can write during the
   cutover.
2. Take and verify a restorable database backup.
3. Run the read-only audit:

   ```bash
   # uv
   uv run inqtrix-migrate --preflight-v02

   # standard Python/pip
   python -m inqtrix.storage.migrate --preflight-v02
   ```

   It requires the expected pre-v0.2 schema and reports unmappable or
   ambiguous legacy subject references, unsupported active group/file/
   `comment`/`manage` shares, orphaned share targets, and non-terminal runs or
   reindex jobs. Exit status 2 means the migration must not proceed.
4. Resolve unsupported data deliberately. With every API and worker process
   still stopped and its connection pool drained, terminalize old
   non-terminal work explicitly:

   ```bash
   # uv
   uv run inqtrix-migrate \
     --terminalize-v02-work \
     --confirm-services-stopped

   # standard Python/pip
   python -m inqtrix.storage.migrate \
     --terminalize-v02-work \
     --confirm-services-stopped
   ```

   The command reruns the complete preflight under exclusive `NOWAIT` table
   locks, refuses any other database client session, and writes all
   `platform_upgrade` terminal states and events in that same transaction. It
   aborts if any blocker other than non-terminal work remains. The confirmation
   flag is an operator assertion, not a process-killing mechanism; do not use
   it while an API or worker could still be inside an external call. Repeat the
   read-only preflight afterwards and require `ready: true`.
5. Apply migrations 0045 through 0047 with `uv run inqtrix-migrate`, or
   `python -m inqtrix.storage.migrate` in the pip-installed environment.
6. Deploy the matching API, worker, and frontend versions together. Do not
   bring up an old worker against the new schema.
7. Wait for startup workspace-share reconciliation to finish before declaring
   readiness when `INQTRIX_SHARING_RESTRICT_TO_WORKSPACE_MEMBERS=true`.
8. Run a two-user smoke test: login, grant, accept, edit, observe the other
   user's refetch, revoke, and verify that the resource and open streams close
   for the recipient.

There is no supported downgrade across migration 0045. Rollback means stopping
the new binaries, restoring the pre-upgrade backup, and starting the matching
old binaries. A schema downgrade without a database restore is not a safe
rollback.

## Testing hardening locally

1. Generate a throwaway cert (example):

   ```bash
   mkcert -install
   mkcert -key-file key.pem -cert-file cert.pem localhost
   ```

2. Set all three layers in `.env`:

   ```dotenv
   INQTRIX_SERVER_TLS_KEYFILE=./key.pem
   INQTRIX_SERVER_TLS_CERTFILE=./cert.pem
   INQTRIX_SERVER_API_KEY=dev-secret-xxxxx
   INQTRIX_SERVER_CORS_ORIGINS=https://localhost:3000
   ```

3. Start the server and smoke-test with `curl -k`:

   ```bash
   curl -kN https://localhost:5100/health
   curl -kN -H "Authorization: Bearer dev-secret-xxxxx" \
        https://localhost:5100/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"research-agent","messages":[{"role":"user","content":"hi"}],"stream":true}'
   ```

The first call should return 200 without auth; the second should stream.

## Related docs

- [Authentication modes](auth-modes.md)
- [Editor collaboration](editor-collaboration.md)
- [Settings and env](../configuration/settings-and-env.md)
- [Runbooks](runbooks.md)
