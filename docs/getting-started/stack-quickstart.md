# Stack quickstart — 5 minutes to Research Desk

> Files: `deploy/compose/compose.stack.yaml`, `deploy/.env.stack.example`, `deploy/docker/Dockerfile.api`, `deploy/docker/Dockerfile.web`, `deploy/docker/Dockerfile.collaboration`

## Scope

The fastest path to a running Research Desk with one command: `docker compose up`. This is **Stack mode** — one compose file, one URL, a built-in web UI, Postgres for durable data. No Python toolchain, no host-side build. For embedding Inqtrix as a library or wiring providers in Python, see [Library mode](../deployment/library-mode.md) instead.

## Prerequisites

- Docker (or Podman with the docker-compose v2 provider).
- One LLM API key and one search API key (e.g. an OpenAI-compatible gateway + Perplexity). See [Provider recipes](provider-recipes.md) for every supported combination.

## 1. Configure

```bash
cp deploy/.env.stack.example deploy/.env.stack
# edit deploy/.env.stack
```

Set the four secrets (`INQTRIX_PG_PASSWORD` — and the matching password in `INQTRIX_DATABASE_URL` — plus `INQTRIX_SESSION_SECRET` and `INQTRIX_PAT_PEPPER`), then fill **one** LLM block and **one** search block. The default is LiteLLM + Perplexity:

```dotenv
INQTRIX_LLM_PROVIDER=litellm
LITELLM_BASE_URL=http://litellm-proxy:4000/v1
LITELLM_API_KEY=sk-...
INQTRIX_SEARCH_PROVIDER=perplexity
PERPLEXITY_API_KEY=pplx-...
REASONING_MODEL=gpt-4o
SEARCH_MODEL=perplexity-sonar-pro-agent
```

To run Azure / Anthropic / Bedrock instead, copy the matching block from [Provider recipes](provider-recipes.md) — it is purely an `.env` change.

The committed stack example also exposes the shared long-operation policy used
by API and worker containers. Keep these values aligned unless you intentionally
tune the deployment:

```dotenv
REASONING_TIMEOUT=600
EDITOR_ASSISTANT_TIMEOUT=600
SEARCH_TIMEOUT=600
CLAIM_EXTRACT_TIMEOUT=600
MAX_TOTAL_SECONDS=3600
INQTRIX_AGENT_MAX_PARALLEL_CHILDREN=6
INQTRIX_AZURE_FOUNDRY_MAX_CONCURRENCY=6
INQTRIX_QUOTA_MAX_TOKENS_PER_RUN=900000
```

The first four values budget one logical operation including every retry;
retries do not receive another full timeout. The optional 900,000-token run
quota stops at the next safe graph boundary and never shortens a stored answer.

That minimum (secrets + one LLM + one search) gets you research + chat. The same `deploy/.env.stack` file has **commented blocks** for the optional pieces — uncomment the ones you want and run the matching profile in step 2:

| Want | Uncomment in `deploy/.env.stack` | Run with |
|---|---|---|
| Cited answers over your documents (RAG) | `INQTRIX_KNOWLEDGE_ENABLED`, `INQTRIX_VECTOR_BACKEND=qdrant`, `INQTRIX_QDRANT_*` | `--profile knowledge` |
| Durable / scaled runs | `INQTRIX_QUEUE_BACKEND=valkey`, `INQTRIX_VALKEY_*` | `--profile workers` |
| Bundled S3 object store (instead of the local volume) | the SeaweedFS static/path/create-if-missing block | `--profile s3` |
| Managed/native S3 | the default-chain or external S3-compatible block; do not start bundled SeaweedFS | no S3 profile |
| SSO login | the OIDC block + `INQTRIX_AUTH_MODE=oidc` | `--profile oidc` |
| Live shared editor documents | `INQTRIX_COLLABORATION_ENABLED=true` + an independent 32+ character `INQTRIX_COLLABORATION_SECRET` | `--profile collaboration` |

Which feature needs which component (and which is on by default) is the matrix in [Platform components](platform-components.md); the full variable list is [Settings and env](../configuration/settings-and-env.md).

## 2. Start

Always pass `--env-file deploy/.env.stack` — it supplies the values Compose needs while reading the file (Postgres password, ports). Compose does not auto-load a file named `.env.stack`.

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.stack up -d --build
```

This builds the API and web images, starts Postgres, runs the schema migration
once through the automatic `migrate` dependency, then starts the API and nginx
web container. A normal install/update never requires a separate manual
`inqtrix-migrate` command. Managed PostgreSQL can place its dedicated direct DSN
in the optional migration-only env file; see [Database migrations](../deployment/database-migrations.md).
First build pulls images and installs dependencies, so it takes a few minutes;
subsequent starts are fast.

```bash
docker compose -f deploy/compose/compose.stack.yaml --env-file deploy/.env.stack ps   # wait for healthy
```

## 3. Open

Open <http://localhost:8080>. The browser talks to a single origin; nginx proxies `/api`, `/v1`, `/health`, and the optional `/collaboration` WebSocket to the API container. Type a question into the composer - the research run streams live.

The web port binds to `127.0.0.1` by default. For a temporary multi-user test
on a trusted local network, expose only the web ingress and open it through the
host's LAN address:

```bash
INQTRIX_WEB_BIND_ADDRESS=0.0.0.0 \
podman compose -f deploy/compose/compose.stack.yaml \
  --env-file deploy/.env.azure.stack \
  --profile knowledge --profile s3 --profile workers up -d --build

# On another device: http://<HOST-LAN-IP>:8080
```

You can instead put `INQTRIX_WEB_BIND_ADDRESS=0.0.0.0` in the selected stack
env file so the original command stays unchanged. Keep `api`, Postgres,
Qdrant, Valkey, and the object store private; Compose publishes only nginx.
Use `local`, `ldap`, or `oidc` auth for distinct users. Plain HTTP requires
`INQTRIX_OIDC_INSECURE_DEV_COOKIES=true` for the cookie-session modes and is
appropriate only on a trusted test LAN. For any broader or persistent rollout,
keep the loopback bind and terminate TLS at a reverse proxy as described in
[Deploy to production](../how-to/deploy-to-production.md).

## 4. Verify

```bash
curl http://localhost:8080/health          # status, llm/search provider + models
curl http://localhost:8080/v1/capabilities # which features are on (knowledge, files, sharing, ...)
```

A missing or wrong credential surfaces as a loud startup error in `docker compose logs api`, naming the variable.

## Users and login

The Stack-mode compose ships with **native accounts** as the default (`INQTRIX_AUTH_MODE=local`). On first visit it walks you through creating the instance owner; from then on the in-app admin area manages users, invitations and access tokens, and people sign in with email and password.

> [!IMPORTANT]
> **Login over plain `http://localhost` can fail silently on a Mac.** The session cookie is hardened with `Secure` and the `__Host-` prefix, which Safari (and Chrome, for the `__Host-` prefix) refuse to store over non-HTTPS. The login request returns success, but the cookie is dropped, so you land back on the login screen with no error message. For local HTTP development, set `INQTRIX_OIDC_INSECURE_DEV_COOKIES=true` in `deploy/.env.stack` and restart the `api` container; it drops the `Secure`/`__Host-` hardening (and logs a startup warning, never use it in production). For any real deployment, terminate TLS in front so the secure cookie works as intended.

Three other modes cover different environments — all in [Auth modes](../deployment/auth-modes.md):

- **`ldap`** — bind against your existing LDAP/Active Directory directory; see [Connect to an existing LDAP](../how-to/connect-to-existing-ldap.md).
- **`oidc`** — enterprise SSO via your own IdP (`--profile oidc`; needs the `INQTRIX_OIDC_*` block and `INQTRIX_PUBLIC_BASE_URL` so the callback URL is derived correctly).
- **`apikey`** / **`none`** — a shared Bearer token, or an open single-operator server with no login.

For remote exposure, terminate TLS in front (see [Security hardening](../deployment/security-hardening.md)).

## Turn on more

The default stack is Postgres + API + web. Add capabilities with compose profiles + the matching env (uncomment the blocks in `deploy/.env.stack`):

```bash
# Knowledge/RAG (Qdrant) and/or durable run workers (Valkey):
docker compose -f deploy/compose/compose.stack.yaml --env-file deploy/.env.stack \
  --profile knowledge --profile workers up -d

# Live editor collaboration (private single Node replica):
docker compose -f deploy/compose/compose.stack.yaml --env-file deploy/.env.stack \
  --profile collaboration up -d --build
```

Which feature needs which component is the decision tree in [Platform components](platform-components.md). Day-to-day start/stop/update/backup commands are in [Runbooks](../deployment/runbooks.md).

To attach an external/managed PostgreSQL instead of the bundled container
(any PostgreSQL 15+ works, including pgvector-enabled images — the extension
stays unused), add the `deploy/compose/compose.external-db.yaml` override and
follow the "External PostgreSQL" block in `deploy/.env.stack.example`:

```bash
docker compose -f deploy/compose/compose.stack.yaml \
  -f deploy/compose/compose.external-db.yaml \
  --env-file deploy/.env.stack up -d --build
```

The migration contract (dedicated migration DSN, RLS mode) is described in
[Database migrations](../deployment/database-migrations.md).

Profiles and env switches must move together: the profile starts the backing
container, while the env block tells the API to use it. If an env block such as
`INQTRIX_VECTOR_BACKEND=qdrant`, `INQTRIX_QUEUE_BACKEND=valkey`, or
`INQTRIX_OBJECT_STORE_BACKEND=s3` is enabled without the matching profile or an
external reachable service, `/v1/capabilities` degrades the affected feature and
the admin System page marks the backend as not reachable.

Collaboration is stricter: enabling its API flag without Postgres, cookie auth,
private service URLs, or a valid independent secret fails FastAPI startup. The
profile without the flag merely starts an unused private service. See
[Deploy editor collaboration](../deployment/editor-collaboration.md).

Enterprise SSO is the `oidc` profile (`--profile oidc`, `INQTRIX_AUTH_MODE=oidc`); it needs the `INQTRIX_OIDC_*` block in `deploy/.env.stack` and `INQTRIX_PUBLIC_BASE_URL=http://localhost:8080` so the callback URL is derived correctly. See [Auth modes](../deployment/auth-modes.md).

## Different setups, one command

Keep a separate env file per setup (e.g. `deploy/.env.azure.stack`). Two things read it: `--env-file` (for `${...}` interpolation) and the services' own `env_file:` (the app's runtime variables, whose path follows `INQTRIX_ENV_FILE`, default `deploy/.env.stack`). Point both at your file by adding one line inside it, then passing `--env-file`:

```bash
# inside deploy/.env.azure.stack, add this line:  INQTRIX_ENV_FILE=deploy/.env.azure.stack
docker compose -f deploy/compose/compose.stack.yaml --env-file deploy/.env.azure.stack up -d
```

## Next steps

- [Provider recipes](provider-recipes.md) — every LLM × search `.env` combination.
- [Platform components](platform-components.md) — do you need Postgres / Qdrant / Valkey / S3? The decision tree + feature matrix.
- [Runbooks](../deployment/runbooks.md) — start, stop, update, backup, restore, reset.
- [Editor collaboration](../deployment/editor-collaboration.md) - optional service, security, and recovery.

## Related docs

- [Overview](overview.md)
- [First research run](first-research-run.md)
- [Deployment modes](../deployment/deployment-modes.md)
- [Editor collaboration](../deployment/editor-collaboration.md)
