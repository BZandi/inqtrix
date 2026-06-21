# Local Infrastructure

## Scope

How to run the optional local PostgreSQL stack with Podman and
Compose: the image policy for third-party containers, starting and
stopping the stack, applying schema migrations, and running the
database-gated integration tests. The platform layer persists identity
facts (workspaces, memberships, shares, audit log) in PostgreSQL when
`INQTRIX_STORAGE_BACKEND=postgres`; the default remains `memory` — no
external services, the full test suite runs offline. This page does
not cover production deployment or the research agent's own
configuration (see [Related docs](#related-docs)).

## Image policy

Binding rules for every container image in this repository:

1. **Trusted-content tiers only.** Docker Official Images
   (`docker.io/library/*`) first; where no official image exists, the
   Docker-Sponsored OSS image published by the upstream project itself
   (e.g. `docker.io/valkey/valkey` later — there is no
   `library/valkey` and there will not be one). Both tiers are
   actively maintained, CVE-rebuilt, and rate-limit exempt or
   high-volume. Nothing gets built from scratch, nothing from
   unknown publishers.
2. **Fully qualified names.** Always `docker.io/library/postgres`,
   never bare `postgres` — Podman has no implicit registry and
   short-name resolution differs per host.
3. **Tag + digest pinned.** `postgres:18.4-trixie@sha256:...` — the
   digest (multi-arch manifest list, not a platform digest) is what is
   enforced; the tag documents intent. Official images are rebuilt
   every few weeks for base-image CVE fixes, so a pinned digest goes
   stale: refresh it deliberately when bumping, via
   `skopeo inspect docker://docker.io/library/postgres:18.4-trixie`
   or the registry API.
4. **One service per container.** No multi-process containers, no
   shared pods for unrelated services — compose orchestrates.

## Why a compose file (not a Podman pod / kube YAML)

`podman kube play` ignores readiness/startup probes and has no
equivalent of `depends_on: condition: service_healthy`; compose is the
runtime-agnostic dev format and the Kubernetes manifests arrive
separately with the deployment phase. `podman compose` delegates to an
external provider — install docker-compose v2 for the best
compose-spec coverage (podman-compose has known health-gating bugs):

```bash
brew install docker-compose   # provider only; the runtime stays Podman
podman machine start          # forwards the API socket
```

## Start / stop

```bash
podman compose -f deploy/compose/compose.dev.yaml up -d
podman compose -f deploy/compose/compose.dev.yaml ps         # wait: healthy
podman compose -f deploy/compose/compose.dev.yaml down       # keep data
podman compose -f deploy/compose/compose.dev.yaml down -v    # drop data
```

Defaults (override via environment): user `inqtrix`, password
`inqtrix-dev-password`, database `inqtrix`, host port `5432` bound to
loopback only. Postgres data lives in the named volume `pgdata`
(inside the Podman machine VM — fast, no macOS bind-mount permission
issues). Note the postgres:18 mount point `/var/lib/postgresql`; do
not reuse pre-18 volumes mounted at `/var/lib/postgresql/data`.

## Migrate and run against Postgres

```bash
export INQTRIX_DATABASE_URL="postgresql+asyncpg://inqtrix:inqtrix-dev-password@127.0.0.1:5432/inqtrix"
uv run inqtrix-migrate                       # apply schema (Alembic head)
INQTRIX_STORAGE_BACKEND=postgres uv run python -m inqtrix
```

The migrations create the restricted `inqtrix_app` role
(NOLOGIN/NOSUPERUSER/NOBYPASSRLS); every application transaction
switches to it via `SET LOCAL ROLE`, so row-level security applies
even though the dev connection user owns the tables. Tenant context is
transaction-local (`set_config('inqtrix.tenant_id', ..., true)`); a
query without it fails loudly in `inqtrix_current_tenant_id()`.

## Database-gated tests

The default suite never needs a database. The Postgres integration
tests run only when a test database is configured:

```bash
export INQTRIX_TEST_DATABASE_URL="postgresql+asyncpg://inqtrix:inqtrix-dev-password@127.0.0.1:5432/inqtrix_test"
podman compose -f deploy/compose/compose.dev.yaml up -d
podman exec -it inqtrix-dev-postgres-1 createdb -U inqtrix inqtrix_test
uv run pytest tests/storage/ -v
```

The fixtures migrate the test database to head and roll the schema
back per session; never point `INQTRIX_TEST_DATABASE_URL` at a
database with real data.

## Durable runs and the worker queue (Valkey)

Run records, events, and results become durable with the Postgres
backend alone — no extra service, execution stays in-process:

```bash
export INQTRIX_DATABASE_URL="postgresql+asyncpg://inqtrix:inqtrix-dev-password@127.0.0.1:5432/inqtrix"
INQTRIX_STORAGE_BACKEND=postgres uv run python -m inqtrix
```

Moving run EXECUTION to separate worker processes additionally needs
the Valkey job stream (service `valkey` in the compose stack, AOF
persistence enabled, loopback-only, password mandatory):

```bash
export INQTRIX_STORAGE_BACKEND=postgres
export INQTRIX_DATABASE_URL="postgresql+asyncpg://inqtrix:inqtrix-dev-password@127.0.0.1:5432/inqtrix"
export INQTRIX_QUEUE_BACKEND=valkey
export INQTRIX_VALKEY_URL="redis://:inqtrix-dev-valkey-password@127.0.0.1:6379/0"

uv run python -m inqtrix          # API replica(s): accept + persist + enqueue
uv run inqtrix-worker             # worker replica(s): claim + execute
```

Semantics worth knowing as an operator:

* The Postgres run row is the source of truth; the stream carries only
  dispatch messages. Delivery is at-least-once — guarded status
  transitions (`queued -> running` compare-and-set, fenced terminal
  writes) make redelivery harmless, and the worker's reconciler
  re-enqueues rows whose dispatch message was lost.
* Workers heartbeat their in-flight stream entries every
  `INQTRIX_WORKER_HEARTBEAT_SECONDS` (default 15); entries idle longer
  than `INQTRIX_WORKER_CLAIM_IDLE_SECONDS` (default 90) are reclaimed
  by another worker. After `INQTRIX_WORKER_MAX_ATTEMPTS` (default 3)
  deliveries a job is dead-lettered to `inqtrix:runs:dead` and the run
  is marked failed.
* Cancellation crosses processes via the run row
  (`cancel_requested`), polled by the executing worker and observed at
  graph node boundaries — exactly the in-memory two-phase semantics.
* SIGTERM stops claiming and drains in-flight runs (90s); undrained
  runs are NOT cancelled — heartbeat silence hands them to another
  worker. Per-worker parallelism: `INQTRIX_WORKER_CONCURRENCY`
  (default 2).
* Without `INQTRIX_QUEUE_BACKEND` (or with the memory storage
  default) nothing changes: runs live in process memory exactly as
  before. `INQTRIX_QUEUE_BACKEND=valkey` without the Postgres backend
  is rejected loudly at startup.

## OIDC login (Dex reference IdP)

`INQTRIX_AUTH_MODE=oidc` turns on the browser login (BFF: tokens never
reach the browser; the session cookie carries an opaque id). Inqtrix
speaks only generic OIDC — Entra ID, Okta, Keycloak, authentik, or the
bundled Dex reference IdP are configuration, not code. The Dex service
is gated behind the `oidc` compose profile so the default stack stays
auth-free:

```bash
podman compose -f deploy/compose/compose.dev.yaml --profile oidc up -d dex

export INQTRIX_AUTH_MODE=oidc
export INQTRIX_OIDC_ISSUER="http://127.0.0.1:5556/dex"
export INQTRIX_OIDC_CLIENT_ID="inqtrix-local"
export INQTRIX_OIDC_CLIENT_SECRET="inqtrix-dev-oidc-secret"
export INQTRIX_OIDC_REDIRECT_URL="http://127.0.0.1:5100/api/auth/callback"
export INQTRIX_SESSION_SECRET="$(openssl rand -hex 32)"
export INQTRIX_OIDC_INSECURE_DEV_COOKIES=true   # plain-http loopback dev ONLY

uv run python -m inqtrix
```

Demo login: `admin@example.com` / `password` (see
`deploy/compose/dex-config.yaml`; mint new bcrypt hashes with
`htpasswd -bnBC 10 "" 'pw' | tr -d ':\n'`). Flow surface:
`GET /api/auth/login` starts the redirect, `GET /api/auth/session`
bootstraps the SPA (identity + CSRF token), unsafe methods then
require the `X-CSRF-Token` header, `POST /api/auth/logout` ends the
session. Sessions/login flows live in memory by default and in
Postgres when `INQTRIX_STORAGE_BACKEND=postgres` (multi-replica
logins; migration 0004). Production deployments use real TLS, leave
`INQTRIX_OIDC_INSECURE_DEV_COOKIES` unset, and get `__Host-`-prefixed
Secure cookies.

## Object storage (SeaweedFS)

The compose stack includes SeaweedFS (`docker.io/chrislusf/seaweedfs`,
the official project image, Apache-2.0) as the S3-compatible blob
store for the file feature. One container runs master + volume +
filer + S3 gateway; the named volume `seaweedfs_data` holds blobs AND
the embedded filer metadata — back up the whole volume, never blobs
alone. The S3 API authenticates against
[`seaweedfs-s3.json`](../../deploy/compose/seaweedfs-s3.json)
(dev credentials; without that file the API would be anonymous).

```bash
export INQTRIX_OBJECT_STORE_BACKEND=s3
export INQTRIX_S3_ENDPOINT_URL="http://127.0.0.1:8333"
export INQTRIX_S3_ACCESS_KEY=inqtrix-dev-access
export INQTRIX_S3_SECRET_KEY=inqtrix-dev-secret
```

Without any object-store configuration the file feature uses the
`local` backend (a directory below `INQTRIX_OBJECT_STORE_PATH`,
default `data/object-store`) — no container needed. The store is
never exposed to clients in either mode: downloads stream through
`GET /v1/files/{id}/content` after the permission check.

The SeaweedFS-gated tests run with:

```bash
INQTRIX_TEST_S3_ENDPOINT="http://127.0.0.1:8333" uv run pytest tests/test_object_store.py -v
```

## Related docs

- [Testing strategy](testing-strategy.md) — where the database-gated
  tests sit in the test pyramid.
- [Docs maintenance](docs-maintenance.md) — conventions this page
  follows.
