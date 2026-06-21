# Platform components — do you need this?

> Files: `deploy/compose/compose.stack.yaml`, `deploy/compose/compose.dev.yaml`, `src/inqtrix/storage/migrate.py`, `src/inqtrix/worker/__main__.py`, `src/inqtrix/settings.py`

## Scope

Inqtrix runs with **zero infrastructure** by default (in-memory storage, in-memory queue, knowledge engine off). Each platform component — Postgres, an object store, Qdrant, Valkey + workers, an OIDC IdP — is opt-in and unlocks specific features. This page answers *which components you actually need*, maps every user-facing feature to its requirements, and shows the manual (Framework-Mode) setup. For the one-command Stack-mode path see [Stack quickstart](stack-quickstart.md).

## Do you need this?

| Your situation | What to run |
|---|---|
| Trying it out / single user, no persistence | Nothing extra — zero-infra default ([First research run](first-research-run.md)) |
| Single-user setup, data survives restart | Stack mode default: **Postgres** (+ API + web) |
| Cited answers over your own documents | + **Qdrant** (`--profile knowledge`) for persistent/hybrid retrieval |
| File uploads | + an **object store** (local volume by default; **S3** via `--profile s3`) |
| Many concurrent runs / runs that survive an API restart | + **Valkey** and worker processes (`--profile workers`) |
| Enterprise SSO | + an **OIDC IdP** (`--profile oidc`; Dex is the reference) |
| Bind logins to an existing directory | + **LDAP** (`--profile ldap`; LLDAP is the dev reference) |

## Feature → requirements matrix

Which feature needs which component and which auth mode. The capability endpoint (`/v1/capabilities`) reports each flag's actual state so the UI degrades visibly; for S3 and Qdrant it also accounts for whether the configured backing service is reachable.

| Feature | Auth mode | Storage | Object store | Qdrant | Valkey/worker | Capability flag |
|---|---|---|---|---|---|---|
| Research runs (core) | any | memory ok | – | – | – | (always on) |
| Durable runs (survive restart) | any | **postgres** | – | – | optional | (implicit) |
| Scaled / queued runs | any | **postgres** | – | – | **valkey + worker** | (implicit) |
| Knowledge / RAG | any | memory ok | – | qdrant (memory ok) | – | `features.knowledge` |
| Hybrid retrieval + reranking | – | – | – | **qdrant** + reranker | – | `features.hybrid_retrieval` / `reranker` |
| File uploads / attachments | any | postgres | **local or s3** | – | – | `features.files` |
| Prompt templates (durable) | any | **postgres** | – | – | – | `features.prompt_templates` |
| Multi-user / invitations | oidc / local / ldap | **postgres** | – | – | – | (implicit) |
| Sharing (runs/collections/templates) | oidc / local / ldap | **postgres** | – | – | – | `features.sharing` |

Note: multi-user, invitations, and sharing require a cookie-session auth mode — `local` (email/password), `ldap`, or `oidc` — plus the Postgres backend. The single-operator `none`/`apikey` modes have no scoped identity and therefore no sharing; for single-user-with-sharing run `local` with one owner.

For administrators, `GET /v1/admin/system/runtime` exposes the same component
choices as a sanitized runtime manifest for the System settings page. It reports
categories only (for example Postgres vs memory, local volume vs S3, Qdrant vs
memory, Valkey worker dispatch vs in-process execution), plus read-only
availability booleans for the optional backing services. It never returns URLs,
paths, bucket names, or credentials. The worker row describes configured
dispatch mode and queue reachability, not live worker replica count.

## Component glossary

Each component states what it enables, the env switch, and what you lose without it.

- **Postgres** (`INQTRIX_STORAGE_BACKEND=postgres`, `INQTRIX_DATABASE_URL=...`) — durable run rows, identity, knowledge metadata, durable prompt templates. Without it: everything is in-memory and lost on restart. The Stack-mode compose defaults to Postgres. Migrations are applied by `inqtrix-migrate` (the compose `migrate` service runs it once before the API starts) and also create the restricted `inqtrix_app` role used for row-level security.
- **Object store** (`INQTRIX_OBJECT_STORE_BACKEND=local|s3`) — storage for uploaded file blobs. `local` (default) writes to a volume; `s3` (`--profile s3` → SeaweedFS, or any S3 endpoint) for shared/scalable storage. If `s3` is configured but unreachable, file uploads are not advertised (`features.files=false`) and the System page shows the object store as not reachable.
- **Qdrant** (`--profile knowledge`, `INQTRIX_VECTOR_BACKEND=qdrant`, `INQTRIX_QDRANT_URL=...`) — persistent vector/document store with hybrid dense + BM25 retrieval. The knowledge engine otherwise uses an in-memory store (lost on restart, dense-only). If Qdrant is configured but unreachable, knowledge and hybrid retrieval are not advertised. Self-hosted Qdrant is unauthenticated by default — set `INQTRIX_QDRANT_API_KEY`.
- **Valkey + worker** (`--profile workers`, `INQTRIX_QUEUE_BACKEND=valkey`, `INQTRIX_VALKEY_URL=...`) — dispatches native runs to separate worker processes for horizontal scaling and restart survival. The default `memory` queue runs in-process. The worker refuses to start without Postgres + Valkey (nothing durable to process).
- **OIDC IdP** (`--profile oidc`, `INQTRIX_AUTH_MODE=oidc`) — browser SSO. Dex is the dev reference; any OIDC provider works. See [Auth modes](../deployment/auth-modes.md).
- **LDAP directory** (`--profile ldap`, `INQTRIX_AUTH_MODE=ldap`) — bind logins against a directory. A throwaway LLDAP is the dev reference; any LDAP/AD works. See [Connect to an existing LDAP](../how-to/connect-to-existing-ldap.md) and the [LDAP stack walkthrough](../../examples/webserver_stacks/ldap_stack.md).

## The "full experience" profile

To turn on every feature in Stack mode, run the `knowledge` + `workers` profiles (add `s3` for shared object storage) and set the matching env in `deploy/.env.stack`:

```bash
docker compose -f deploy/compose/compose.stack.yaml --env-file deploy/.env.stack \
  --profile knowledge --profile workers up -d --build
```

```dotenv
INQTRIX_STORAGE_BACKEND=postgres        # durable runs, templates, identity
INQTRIX_OBJECT_STORE_BACKEND=local      # file uploads (or s3 + --profile s3)
INQTRIX_KNOWLEDGE_ENABLED=true          # knowledge engine
INQTRIX_VECTOR_BACKEND=qdrant           # persistent/hybrid retrieval
INQTRIX_QDRANT_URL=http://qdrant:6333
INQTRIX_QDRANT_API_KEY=...
INQTRIX_QUEUE_BACKEND=valkey            # scaled/durable run execution
INQTRIX_VALKEY_URL=redis://:...@valkey:6379/0
INQTRIX_VALKEY_PASSWORD=...
```

Misconfiguration fails loudly at startup (e.g. `INQTRIX_QUEUE_BACKEND=valkey` without a URL, or `valkey` without `postgres`).

## Manual / host platform (Framework Mode)

If you run the API on the host instead of in the Stack-mode container (library mode, custom providers, integration tests), start only the infrastructure with the dev compose and wire the API yourself:

```bash
# Infrastructure only (no api/web container):
docker compose -f deploy/compose/compose.dev.yaml up -d        # add --profile oidc for Dex
uv sync --extra knowledge-qdrant --extra queue-valkey          # backends that need extras

# Postgres + migrations, then the API on the host:
export INQTRIX_STORAGE_BACKEND=postgres
export INQTRIX_DATABASE_URL="postgresql+asyncpg://inqtrix:inqtrix-dev-password@127.0.0.1:5432/inqtrix"
uv run inqtrix-migrate
uv run python -m inqtrix

# Workers (separate shells, same .env) when INQTRIX_QUEUE_BACKEND=valkey:
uv run inqtrix-worker
```

The dev compose credentials are committed loopback defaults, not secrets — override the `INQTRIX_PG_*`, `INQTRIX_QDRANT_API_KEY`, `INQTRIX_VALKEY_PASSWORD` variables for anything beyond local use. Image policy, Podman setup, and volume details: [Local infrastructure](../development/local-infrastructure.md).

## Verify

```bash
curl http://localhost:5100/health          # or :8080 through the Stack-mode web container
curl http://localhost:5100/v1/capabilities # the feature manifest
```

## Related docs

- [Stack quickstart](stack-quickstart.md)
- [Runbooks](../deployment/runbooks.md)
- [Deployment modes](../deployment/deployment-modes.md)
- [Settings and env](../configuration/settings-and-env.md)
- [Local infrastructure](../development/local-infrastructure.md)
