# Kubernetes and OpenShift

> Files: `deploy/helm/inqtrix/`, `deploy/docker/Dockerfile.api`, `deploy/docker/Dockerfile.web`, `deploy/nginx/inqtrix.conf.template`

## Scope

How to deploy Inqtrix on Kubernetes or Red Hat OpenShift with the bundled Helm
chart at `deploy/helm/inqtrix/`. This page covers the concepts (what Helm does,
what runs where), building and publishing the images, four step-by-step install
scenarios (turnkey demo, Azure-backed demo, production with an external database,
OpenShift), secret and auth configuration, scaling, upgrades, and
troubleshooting.

It does **not** cover the single-host Docker Compose stack (see
[Stack quickstart](../getting-started/stack-quickstart.md)) or the conceptual
component overview (see [Platform components](../getting-started/platform-components.md)).
Provider selection and the full environment-variable reference live in
[Provider recipes](../getting-started/provider-recipes.md) and
[Settings and env](../configuration/settings-and-env.md).

## Concepts: how a Helm deployment works

If you know Docker Compose, Helm is the Kubernetes equivalent of `docker compose up`:

```text
Compose                              Kubernetes
  docker compose up          ~=        helm install inqtrix ./deploy/helm/inqtrix
  (reads compose.yaml)                 (reads the Helm chart)
  starts the containers                creates the Kubernetes objects (Pods, ...)
```

A **Helm chart** is a parameterised package of Kubernetes manifests. `helm install`
reads it, fills the placeholders from `values.yaml` (and your `--set` overrides),
and creates everything automatically. There is one command that brings the whole
deployment up; that is the normal path, not the exception.

The end-to-end flow:

```text
  [1] source + Dockerfiles
        |  docker build (the images are non-root, OpenShift-ready)
        v
  [2] two images: inqtrix-api, inqtrix-web   --push-->  a container registry
        v
  [3] helm install inqtrix ./deploy/helm/inqtrix      <- the "one command"
        |
        v
  +-------------------- Kubernetes / OpenShift cluster --------------------+
  |  migrate Job (runs ONCE first: inqtrix-migrate) ---+                   |
  |                                                    v                   |
  |  api Deployment   web Deployment   worker Deployment   (our pods)      |
  |        |               |                 |                             |
  |        +---- ConfigMap (settings) + Secret (passwords/keys) ----+      |
  |                                                                        |
  |  optionally bundled:  [Postgres] [Qdrant] [Valkey] [MinIO/S3]          |
  |  OR point at external/managed services instead                         |
  |                                                                        |
  |  Ingress (k8s) / Route (OpenShift)  -> reachable over HTTPS            |
  +------------------------------------------------------------------------+
```

Two groups of components, important to keep apart:

| Group | Components | Who provides them |
|---|---|---|
| **Inqtrix itself** | `api`, `web`, `worker`, the one-shot `migrate` Job | Always the chart (your images). The worker is never "external". |
| **Backing infrastructure** | PostgreSQL, Qdrant (vector store), Valkey (queue), S3 object store (MinIO) | Either bundled by the chart (demo) **or** external/managed (production). Your choice per service. |

The chart bundles Postgres, Qdrant, Valkey and an S3 object store (MinIO) as
optional turnkey services, but not every Docker Compose profile. In particular,
Dex/OIDC and LLDAP are external to the chart: use your own identity/directory
provider (or deploy one with your platform's preferred operator/chart) and point
Inqtrix at it through `config` and `secret`.

Configuration is split exactly as Kubernetes expects: non-secret settings go into
a **ConfigMap** (chart value `config`), secrets (DB URL, session/PAT secrets, API
keys) go into a **Secret** (chart value `secret`). Both are injected into the pods
as the same environment variables you would set in a `.env` file, including
provider-specific names such as `AZURE_OPENAI_ENDPOINT`.

## Prerequisites

- A running cluster and a configured CLI: `kubectl` (Kubernetes) or `oc` (OpenShift).
- [Helm](https://helm.sh/) 3.8 or newer (`helm version`).
- A container registry the cluster can pull from (Docker Hub, GHCR, Quay, the
  OpenShift internal registry, ...). For a local kind/minikube cluster you can
  load images directly instead of pushing.

Decide before you start (each maps to a value below):

- **Platform**: vanilla Kubernetes (Ingress) or OpenShift (Route, `openshift.enabled=true`)?
- **Database and queues**: bundle demo Postgres/Qdrant/Valkey
  (`postgres.enabled=true`, `qdrant.enabled=true`, `valkey.enabled=true`) or
  point at external/managed ones (`INQTRIX_DATABASE_URL`, `INQTRIX_QDRANT_URL`,
  `INQTRIX_VALKEY_URL`).
- **Object store**: local single-pod storage, the bundled S3 (MinIO) via
  `s3.enabled=true`, or an external S3-compatible endpoint?
- **Auth mode**: `none`, `apikey`, `local`, `ldap`, or `oidc`? (See [Auth modes](auth-modes.md).)
- **Knowledge engine / worker**: needed? (Qdrant for knowledge; Valkey + worker for durable runs.)
- **Live editor collaboration**: needed? It requires Postgres, cookie auth, and one private collaboration pod.

## Step 1: build and publish the images, prepare the chart

The chart always deploys API and web images and can deploy the optional
collaboration image. There is no CI publishing them yet, so build and push them
yourself. Run these from the repository root and replace `<REGISTRY>`
(your registry prefix, e.g. `ghcr.io/your-org`) and `<TAG>` (e.g. `0.2.0`):

```bash
# Build (build context is the repo root)
docker build -f deploy/docker/Dockerfile.api -t <REGISTRY>/inqtrix-api:<TAG> .
docker build -f deploy/docker/Dockerfile.web -t <REGISTRY>/inqtrix-web:<TAG> .
docker build -f deploy/docker/Dockerfile.collaboration -t <REGISTRY>/inqtrix-collaboration:<TAG> .

# Push (the cluster pulls from here)
docker push <REGISTRY>/inqtrix-api:<TAG>
docker push <REGISTRY>/inqtrix-web:<TAG>
docker push <REGISTRY>/inqtrix-collaboration:<TAG> # only when enabled
```

For a multi-architecture image (amd64 + arm64) use buildx:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
    -f deploy/docker/Dockerfile.api -t <REGISTRY>/inqtrix-api:<TAG> --push .
```

If your registry is private, create a pull secret and reference it via
`imagePullSecrets` (see the values reference below). The namespace must exist
first (`kubectl create namespace inqtrix`, or `oc new-project inqtrix`):

```bash
kubectl create namespace inqtrix   # skip if it already exists
kubectl -n inqtrix create secret docker-registry regcred \
    --docker-server=<REGISTRY> --docker-username=<USER> --docker-password=<TOKEN>
```

Then pass it to the chart on `helm install` with `--set imagePullSecrets[0].name=regcred`.

Point the chart at the images you pushed, either on the command line
(`--set image.registry=<REGISTRY> --set image.api.tag=<TAG> --set image.web.tag=<TAG>`)
or in a values file. The tag defaults to the chart `appVersion` when left empty.

The chart has no external Helm dependencies, so no `helm dependency update` is
needed. Validate it any time with:

```bash
# A bare lint/render hits the fail-fast guard (default postgres backend, no DB).
# Pass a DB source to validate cleanly; the memory backend with migrations off is
# the simplest and renders a coherent stateless manifest for inspection:
helm lint deploy/helm/inqtrix \
    --set config.INQTRIX_STORAGE_BACKEND=memory --set migrations.enabled=false
helm template inqtrix deploy/helm/inqtrix \
    --set config.INQTRIX_STORAGE_BACKEND=memory --set migrations.enabled=false
```

## Compose profiles to Helm values

Docker Compose can read a whole `.env` file through `INQTRIX_ENV_FILE`. Helm does
not read `.env` files directly. Instead, put every non-secret environment
variable under `config` (or `extraConfig`) and every secret under `secret.data`
or an externally managed Kubernetes Secret.

| Compose / `.env` setting | Helm equivalent |
|---|---|
| `INQTRIX_ENV_FILE=deploy/.env.azure.stack` | Compose-only loader. For Helm, copy the relevant keys into a private values file, `config.*`, `secret.data.*`, or `secret.existingSecret`. |
| `INQTRIX_WEB_PORT=8080` | `service.webPort` controls the Service port; expose it with port-forward, Ingress, or Route. |
| `INQTRIX_PG_PASSWORD` | For bundled demo Postgres use `postgres.auth.password`; for external Postgres put the full `INQTRIX_DATABASE_URL` in the Secret. |
| `INQTRIX_MIGRATION_DATABASE_URL`, `INQTRIX_MIGRATION_RLS_MODE` | Put the direct privileged URL in a dedicated Secret selected by `migrations.databaseSecret.name/key`; its name must differ from the runtime application Secret. `secret.data.INQTRIX_MIGRATION_DATABASE_URL` is rejected. Select authority with `migrations.rlsMode`. The URL is injected only into the one-shot hook Job and explicitly blanked in API/worker. |
| `INQTRIX_QDRANT_URL`, `INQTRIX_VECTOR_BACKEND`, `INQTRIX_KNOWLEDGE_ENABLED` | Auto-wired when `qdrant.enabled=true`; for external Qdrant set them in `config` and put `INQTRIX_QDRANT_API_KEY` in the Secret when needed. |
| `INQTRIX_QUEUE_BACKEND=valkey`, `INQTRIX_VALKEY_URL`, `INQTRIX_VALKEY_PASSWORD` | Auto-wired when `valkey.enabled=true` and `worker.enabled=true`; for external Valkey set `INQTRIX_QUEUE_BACKEND=valkey` in `config` and `INQTRIX_VALKEY_URL` in the Secret. |
| `--profile collaboration`, `INQTRIX_COLLABORATION_ENABLED=true` | `collaboration.enabled=true`; set `image.collaboration.*` and provide `collaboration.secret.existingSecret` (or the main Secret) with `collaboration.secret.key`. The chart derives the private HTTP/WS URLs. |
| `--profile s3` (SeaweedFS), `INQTRIX_OBJECT_STORE_BACKEND=s3`, `INQTRIX_S3_*` | Bundle MinIO with `s3.enabled=true`, use static credentials in the app Secret, or set `AUTH_MODE=default` plus API/worker ServiceAccount overrides for managed workload identity. See [Object storage](object-storage.md). |
| Dex/OIDC and LLDAP profiles | Helm has no bundled Dex or LLDAP service. Use `config.INQTRIX_AUTH_MODE=oidc`/`ldap` and point at an external IdP/directory. |

## Step 2, scenario A: turnkey demo on vanilla Kubernetes

The fastest way to see Inqtrix running. The chart starts Postgres and Qdrant
in-cluster and auto-wires their connections, so no database setup is needed.
**Demo/dev only** (single-pod backing services, not highly available).

> Resource names: the commands here use the release name `inqtrix`, so the
> resources are named `inqtrix-api`, `inqtrix-web`, `inqtrix-postgres`, etc. If
> you install under a different release name `<rel>`, the chart names them
> `<rel>-inqtrix-*` (and the Route is `<rel>-inqtrix`); adjust the `svc/`,
> `deploy/`, `job/` and route names in every command below accordingly.

```bash
helm install inqtrix deploy/helm/inqtrix \
    --namespace inqtrix --create-namespace \
    --set image.registry=<REGISTRY> \
    --set image.api.tag=<TAG> --set image.web.tag=<TAG> \
    --set postgres.enabled=true \
    --set qdrant.enabled=true \
    --set-string secret.data.INQTRIX_SESSION_SECRET=$(openssl rand -hex 32) \
    --set-string secret.data.INQTRIX_PAT_PEPPER=$(openssl rand -hex 32)
```

What happens: the bundled Postgres and Qdrant start, then a post-install
migration job waits for Postgres and creates the schema; the api/web pods may
restart a few times until it finishes, then become ready. `INQTRIX_DATABASE_URL`,
`INQTRIX_QDRANT_URL` and the knowledge/vector backend flags are auto-wired from
the bundled services. The demo Postgres password defaults to `change-me-postgres`;
override it with `--set postgres.auth.password=...` (URL-unreserved characters
`[A-Za-z0-9._~-]` only, since it is embedded in the connection URL). To also demo
the durable worker, see [Scaling](#scaling) — it needs the S3 object store so the
api and worker share artifacts.

For the bundled-Postgres demo, run the install command as shown (no `--wait`).
The migration is a Helm post-install hook, so Helm waits for that hook; then use
Kubernetes readiness to wait for the pods:

```bash
kubectl -n inqtrix wait --for=condition=ready pod \
    -l app.kubernetes.io/instance=inqtrix --timeout=5m
kubectl -n inqtrix get pods
```

No Ingress is enabled by default, so reach it with a port-forward:

```bash
kubectl -n inqtrix port-forward svc/inqtrix-web 8080:8080
# then open http://127.0.0.1:8080/
```

(The default auth mode is `infer` -> `none`, i.e. anonymous, which is fine for a
local trial. Pick a real auth mode for anything shared - see scenario C.)
This scenario is a platform smoke test. Use scenario B for a usable Azure-backed
chat/search/knowledge demo.

## Step 2, scenario B: Azure-backed demo on vanilla Kubernetes

This is closer to a real application stack than the minimal demo: Azure OpenAI
for chat, Azure AI Foundry for web search, Azure embeddings, Qdrant for
knowledge, Valkey + worker for durable runs, and a bundled S3 object store
(MinIO) for shared files. Postgres, Qdrant, Valkey and the MinIO object store are
all bundled for a self-contained demo; use scenario C when those services are
managed externally.

Start from the checked-in example overlay and change the placeholder endpoints,
project path and bucket to your own values:

```bash
deploy/helm/inqtrix/values-azure.example.yaml
```

Install with secrets supplied on the command line (or through a private values
file / external secret manager):

```bash
helm install inqtrix deploy/helm/inqtrix \
    --namespace inqtrix --create-namespace \
    -f deploy/helm/inqtrix/values-azure.example.yaml \
    --set image.registry=<REGISTRY> \
    --set image.api.tag=<TAG> --set image.web.tag=<TAG> \
    --set config.INQTRIX_OIDC_INSECURE_DEV_COOKIES=true \
    --set-string secret.data.INQTRIX_SESSION_SECRET=$(openssl rand -hex 32) \
    --set-string secret.data.INQTRIX_PAT_PEPPER=$(openssl rand -hex 32) \
    --set-string secret.data.AZURE_OPENAI_API_KEY=<AZURE-OPENAI-KEY> \
    --set-string secret.data.AZURE_AI_PROJECT_API_KEY=<AZURE-FOUNDRY-KEY> \
    --set-string secret.data.INQTRIX_EMBEDDING_AZURE_API_KEY=<AZURE-EMBEDDING-KEY>
```

No S3 credentials are needed here: the bundled MinIO (`s3.enabled=true` in the
overlay) auto-wires its root credentials as the app's S3 keys. For anything beyond
a local trial, override them with `--set s3.accessKey=... --set s3.secretKey=...`
(secret key >= 8 characters).

The `INQTRIX_OIDC_INSECURE_DEV_COOKIES=true` override is only for the local
HTTP port-forward shown below. Remove it when you expose the app over HTTPS.

Reach the web UI:

```bash
kubectl -n inqtrix wait --for=condition=ready pod \
    -l app.kubernetes.io/instance=inqtrix --timeout=5m
kubectl -n inqtrix port-forward svc/inqtrix-web 8080:8080
```

For a lighter trial (no object store, queue or worker), disable them and use the
local object store instead:

```bash
--set s3.enabled=false \
--set worker.enabled=false \
--set valkey.enabled=false \
--set config.INQTRIX_OBJECT_STORE_BACKEND=local
```

### What the Azure overlay sets

| Area | Values |
|---|---|
| Chat models | `INQTRIX_LLM_PROVIDER=azure`, `AZURE_OPENAI_ENDPOINT`, `REASONING_MODEL`, `TIER_*_MODEL`, `INQTRIX_SELECTABLE_CHAT_MODELS` |
| Web search | `INQTRIX_SEARCH_PROVIDER=azure_foundry`, `AZURE_AI_PROJECT_ENDPOINT`, `WEB_SEARCH_AGENT_NAME`, `WEB_SEARCH_AGENT_VERSION` |
| Embeddings | `INQTRIX_EMBEDDING_PROVIDER=azure`, `INQTRIX_EMBEDDING_AZURE_ENDPOINT`, `INQTRIX_EMBEDDING_MODEL=text-embedding-3-large` |
| Knowledge | `qdrant.enabled=true` auto-wires `INQTRIX_QDRANT_URL`, `INQTRIX_VECTOR_BACKEND=qdrant`, `INQTRIX_KNOWLEDGE_ENABLED=true` |
| Durable work | `valkey.enabled=true` and `worker.enabled=true` auto-wire `INQTRIX_QUEUE_BACKEND=valkey` and `INQTRIX_VALKEY_URL` |
| Object store | `s3.enabled=true` bundles MinIO and auto-wires `INQTRIX_OBJECT_STORE_BACKEND=s3`, `INQTRIX_S3_ENDPOINT_URL`, `INQTRIX_S3_BUCKET` and the S3 access/secret keys |
| Auth & access | `INQTRIX_AUTH_MODE=local`, `INQTRIX_LOCAL_REGISTRATION=closed` (first user becomes the instance owner via first-run setup), `INQTRIX_QUOTA_ENABLED=true` (per-user quotas; every limit `0` = unlimited until an admin sets one) |

## Step 2, scenario C: production with an external database (vanilla Kubernetes)

Production keeps state in a managed/external PostgreSQL and exposes the app over
an Ingress with TLS. Provide secrets through a Kubernetes Secret you own.

1. Create the namespace and the Secret (replace every `<...>`; generate the two
   random secrets with `openssl rand -hex 32`):

   ```bash
   kubectl create namespace inqtrix   # skip if it already exists
   kubectl -n inqtrix create secret generic inqtrix-secrets \
       --from-literal=INQTRIX_DATABASE_URL='postgresql+asyncpg://<USER>:<PASS>@<DB-HOST>:5432/<DB-NAME>' \
       --from-literal=INQTRIX_SESSION_SECRET='<openssl rand -hex 32>' \
       --from-literal=INQTRIX_PAT_PEPPER='<openssl rand -hex 32>' \
       --from-literal=AZURE_OPENAI_API_KEY='<AZURE-OPENAI-KEY>'
   kubectl -n inqtrix create secret generic inqtrix-migration-database \
       --from-literal=INQTRIX_MIGRATION_DATABASE_URL='postgresql+asyncpg://<MIGRATION-USER>:<PASS>@<DB-HOST>:5432/<DB-NAME>'
   ```

   The runtime Secret holds only application/provider secrets. The second
   Secret is available only to the migration Job and must use a direct database
   connection with the authority described in [Database migrations](database-migrations.md).

2. Install, pointing at the Secret and your Ingress host:

   ```bash
   helm install inqtrix deploy/helm/inqtrix \
       --namespace inqtrix --create-namespace \
       --set image.registry=<REGISTRY> \
       --set image.api.tag=<TAG> --set image.web.tag=<TAG> \
       --set secret.existingSecret=inqtrix-secrets \
       --set migrations.databaseSecret.name=inqtrix-migration-database \
       --set migrations.rlsMode=bypass \
       --set config.INQTRIX_AUTH_MODE=local \
       --set config.INQTRIX_LLM_PROVIDER=azure \
       --set config.AZURE_OPENAI_ENDPOINT=https://<AZURE-OPENAI-RESOURCE>.openai.azure.com/ \
       --set-string config.REASONING_MODEL=gpt-5.4 \
       --set-string config.TIER_HIGH_MODEL=gpt-5.4 \
       --set-string config.TIER_MID_MODEL=gpt-5.4 \
       --set-string config.TIER_FAST_MODEL=gpt-5.4-mini \
       --set config.INQTRIX_PUBLIC_BASE_URL=https://<YOUR-DOMAIN> \
       --set ingress.enabled=true \
       --set ingress.className=<YOUR-INGRESS-CLASS> \
       --set ingress.host=<YOUR-DOMAIN> \
       --set ingress.tls.enabled=true \
       --set ingress.tls.secretName=<YOUR-TLS-SECRET>
   ```

   For an enabled Ingress, the chart derives `INQTRIX_PUBLIC_BASE_URL` from
   `ingress.host` and `ingress.tls.enabled`, and configures nginx to overwrite
   forwarded proto with that trusted boundary. The explicit value above is
   therefore optional when it is identical, and keeps precedence when the
   externally published origin intentionally differs from the Ingress host.

3. To scale or survive restarts, enable the worker and a queue, and use the S3
   object store (see [Scaling](#scaling)).

## Step 2, scenario D: OpenShift

Same as scenario C, with three differences:

- Set `openshift.enabled=true`. The chart then renders an OpenShift **Route**
  instead of an Ingress, and omits `runAsUser`/`runAsGroup`/`fsGroup` so the
  default `restricted-v2` SCC assigns its own arbitrary UID. No SCC change is
  needed: the images run non-root, bind only ports above 1024, drop all
  capabilities and use a read-only root filesystem.
- Use `oc` and an OpenShift Route host (or leave `route.host` empty to let
  OpenShift generate one). Collaboration needs a public trust anchor at render
  time: set `route.host` explicitly, or deploy with collaboration disabled,
  read the generated host, then upgrade with
  `config.INQTRIX_PUBLIC_BASE_URL=https://<GENERATED-HOST>` and
  `collaboration.enabled=true`. The chart fails loudly if collaboration is
  enabled while both values are unknown.
- On OpenShift you **can** bundle Qdrant, Valkey and the S3 store
  (`qdrant.enabled`, `valkey.enabled`, `s3.enabled`) — they run under the
  `restricted-v2` arbitrary UID. The bundled **Postgres** is the exception: the
  official image's `initdb` needs a fixed UID and will not start under
  `restricted-v2`, so use an external or managed database there
  (`INQTRIX_DATABASE_URL`, e.g. Azure Database for PostgreSQL). The example below
  uses `values-openshift.yaml`, which keeps Postgres external. The external
  service must run PostgreSQL 15 or newer; the migration preflight checks this
  before taking schema locks. No PostgreSQL extensions are required —
  pgvector-enabled images work, the extension stays unused (see
  [Database migrations](database-migrations.md)).

```bash
oc new-project inqtrix

oc -n inqtrix create secret generic inqtrix-secrets \
    --from-literal=INQTRIX_DATABASE_URL='postgresql+asyncpg://<USER>:<PASS>@<DB-HOST>:5432/<DB-NAME>' \
    --from-literal=INQTRIX_SESSION_SECRET='<openssl rand -hex 32>' \
    --from-literal=INQTRIX_PAT_PEPPER='<openssl rand -hex 32>' \
    --from-literal=INQTRIX_OIDC_CLIENT_SECRET='<OIDC-CLIENT-SECRET>' \
    --from-literal=AZURE_OPENAI_API_KEY='<AZURE-OPENAI-KEY>'

oc -n inqtrix create secret generic inqtrix-migration-database \
    --from-literal=INQTRIX_MIGRATION_DATABASE_URL='postgresql+asyncpg://<MIGRATION-USER>:<PASS>@<DB-HOST>:5432/<DB-NAME>'

helm install inqtrix deploy/helm/inqtrix \
    --namespace inqtrix \
    -f deploy/helm/inqtrix/values-openshift.yaml \
    --set image.registry=<REGISTRY> \
    --set image.api.tag=<TAG> --set image.web.tag=<TAG> \
    --set secret.existingSecret=inqtrix-secrets \
    --set migrations.databaseSecret.name=inqtrix-migration-database \
    --set migrations.rlsMode=bypass \
    --set config.INQTRIX_AUTH_MODE=oidc \
    --set config.INQTRIX_LLM_PROVIDER=azure \
    --set config.AZURE_OPENAI_ENDPOINT=https://<AZURE-OPENAI-RESOURCE>.openai.azure.com/ \
    --set-string config.REASONING_MODEL=gpt-5.4 \
    --set-string config.TIER_HIGH_MODEL=gpt-5.4 \
    --set-string config.TIER_MID_MODEL=gpt-5.4 \
    --set-string config.TIER_FAST_MODEL=gpt-5.4-mini \
    --set config.INQTRIX_OIDC_ISSUER=https://<IDP>/realms/<REALM> \
    --set config.INQTRIX_OIDC_CLIENT_ID=<OIDC-CLIENT-ID> \
    --set config.INQTRIX_PUBLIC_BASE_URL=https://<YOUR-ROUTE-HOST>
```

`oidc` refuses to start without `INQTRIX_OIDC_ISSUER`, `INQTRIX_OIDC_CLIENT_ID`,
`INQTRIX_OIDC_CLIENT_SECRET`, `INQTRIX_SESSION_SECRET`, `INQTRIX_PAT_PEPPER`, and one
of `INQTRIX_PUBLIC_BASE_URL` or `INQTRIX_OIDC_REDIRECT_URL` (all set above) — see the
[auth modes](auth-modes.md#mode-oidc) reference for the optional claim/group settings.
Find the generated URL with `oc -n inqtrix get route inqtrix -o jsonpath='{.spec.host}'`.

## Configuration in detail

### Secrets

These are the secret keys Inqtrix reads (full reference:
[Settings and env](../configuration/settings-and-env.md)). Put them in your
`existingSecret` (production) or under `secret.data` via `--set-string` (dev).

Important: when `secret.existingSecret` is set, the chart references only that
Secret. It cannot merge chart-derived secret values into your Secret. If you use
an existing Secret together with bundled Postgres/Valkey/MinIO, that Secret must
still contain the derived values itself (`INQTRIX_DATABASE_URL` /
`INQTRIX_VALKEY_URL` / `INQTRIX_S3_ACCESS_KEY` + `INQTRIX_S3_SECRET_KEY`). For
bundled demo installs, prefer the chart-managed `secret.data` path unless you
intentionally own every secret key.

| Secret key | When needed | What it is |
|---|---|---|
| `INQTRIX_DATABASE_URL` | storage backend `postgres` | `postgresql+asyncpg://user:pass@host:5432/db`. Auto-wired when `postgres.enabled=true`. |
| `INQTRIX_MIGRATION_DATABASE_URL` | managed PostgreSQL migration job | Direct migration-only URL stored in `migrations.databaseSecret`; never put it in the app Secret. |
| `INQTRIX_SESSION_SECRET` | auth `local`/`ldap`/`oidc` | >=32 chars; CSRF-token HMAC key. `openssl rand -hex 32`. |
| `INQTRIX_PAT_PEPPER` | auth `local`/`ldap`/`oidc` | >=32 chars; pepper for personal access tokens. Rotating it invalidates all tokens. |
| `INQTRIX_OIDC_CLIENT_SECRET` | auth `oidc` | Confidential OIDC client secret from your IdP. |
| `INQTRIX_LDAP_BIND_PASSWORD` | auth `ldap` | Service-account bind password. |
| `INQTRIX_S3_ACCESS_KEY`, `INQTRIX_S3_SECRET_KEY`, optional `INQTRIX_S3_SESSION_TOKEN` | object store `s3`, static auth | S3 credentials. Auto-wired from `s3.accessKey`/`s3.secretKey` only for bundled MinIO. Omit all three with `INQTRIX_S3_AUTH_MODE=default`. |
| `INQTRIX_QDRANT_API_KEY` | external Qdrant with auth | Qdrant API key. Auto-wired when `qdrant.apiKey` is set on a bundled Qdrant. |
| `LITELLM_API_KEY` | LLM provider `litellm` (default) | Key for the LiteLLM gateway / OpenAI-compatible endpoint. |
| `ANTHROPIC_API_KEY` | LLM provider `anthropic` | Direct Anthropic API key. |
| `AZURE_OPENAI_API_KEY` (or `AZURE_CLIENT_SECRET`) | LLM provider `azure` | Azure OpenAI key (or Entra service-principal secret). |
| `PERPLEXITY_API_KEY` | search provider `perplexity`, native | Perplexity key when not proxied via LiteLLM. |
| `AZURE_AI_PROJECT_API_KEY` | search provider `azure_foundry` | Azure AI Foundry project key. |
| `INQTRIX_EMBEDDING_API_KEY` (or `INQTRIX_EMBEDDING_AZURE_API_KEY`) | knowledge engine | Embeddings endpoint key (empty reuses `LITELLM_API_KEY`). |
| `INQTRIX_RERANKER_API_KEY` | reranker `cohere`/`llm` | Rerank endpoint key. |
| `INQTRIX_SERVER_API_KEY` | auth `apikey` | Static Bearer key gating the server. |

Set only the keys your selected providers/features use. The exhaustive, always-current
list of every secret field is [Settings and env](../configuration/settings-and-env.md).
Provider endpoints, deployment names, model names, bucket names and auth
issuer/client ids are usually non-secret and belong under `config`, for example
`AZURE_OPENAI_ENDPOINT`, `AZURE_AI_PROJECT_ENDPOINT`, `AZURE_CLIENT_ID`,
`AZURE_TENANT_ID`, `INQTRIX_S3_ENDPOINT_URL` and `INQTRIX_S3_BUCKET`.

Generate the random secrets once:

```bash
openssl rand -hex 32   # use the output for INQTRIX_SESSION_SECRET and (a second run) INQTRIX_PAT_PEPPER
```

For production secret management beyond a hand-created Secret, use the External
Secrets Operator, Sealed Secrets, or SOPS, and point `secret.existingSecret` at
the resulting Secret.

### Auth modes

Set the mode in `config.INQTRIX_AUTH_MODE` and provide the mode's secrets. Full
behaviour is in [Auth modes](auth-modes.md); the chart-specific essentials:

`local`, `ldap` and `oidc` need the **postgres** storage backend (to persist
accounts/sessions durably). The chart default `config.INQTRIX_STORAGE_BACKEND`
is already `postgres`, so it just works — do **not** override it to `memory` for
these modes (with `memory`, logins are lost on restart and not shared across
replicas).

- **none / apikey** — no database needed for auth. `apikey` requires
  `INQTRIX_SERVER_API_KEY` in the Secret.
- **local** — email/password accounts (requires the postgres storage backend).
  Needs `INQTRIX_SESSION_SECRET` + `INQTRIX_PAT_PEPPER`.
  If you test local auth through an HTTP port-forward, temporarily set
  `config.INQTRIX_OIDC_INSECURE_DEV_COOKIES=true`; remove it behind HTTPS.

  ```bash
  --set config.INQTRIX_AUTH_MODE=local
  # secret: INQTRIX_SESSION_SECRET, INQTRIX_PAT_PEPPER
  ```

- **oidc** — SSO via your IdP.

  ```bash
  --set config.INQTRIX_AUTH_MODE=oidc \
  --set config.INQTRIX_OIDC_ISSUER=https://<IDP>/realms/<REALM> \
  --set config.INQTRIX_OIDC_CLIENT_ID=<CLIENT-ID> \
  --set config.INQTRIX_PUBLIC_BASE_URL=https://<YOUR-DOMAIN>
  # secret: INQTRIX_OIDC_CLIENT_SECRET, INQTRIX_SESSION_SECRET, INQTRIX_PAT_PEPPER
  # Required to boot: issuer + client-id + client-secret + session-secret +
  # pat-pepper + one of INQTRIX_PUBLIC_BASE_URL / INQTRIX_OIDC_REDIRECT_URL.
  ```

- **ldap** — bind to an existing directory.

  ```bash
  --set config.INQTRIX_AUTH_MODE=ldap \
  --set config.INQTRIX_LDAP_URL=ldaps://<LDAP-HOST>:636 \
  --set config.INQTRIX_LDAP_BIND_DN='<BIND-DN>' \
  --set config.INQTRIX_LDAP_USER_SEARCH_BASE='<BASE-DN>'
  # secret: INQTRIX_LDAP_BIND_PASSWORD, INQTRIX_SESSION_SECRET, INQTRIX_PAT_PEPPER
  ```

### Values reference (most-used knobs)

The commented [`values.yaml`](../../deploy/helm/inqtrix/values.yaml) is the full
reference. The knobs you will reach for most:

| Value | Default | Effect |
|---|---|---|
| `image.registry`, `image.api.tag`, `image.web.tag`, `image.collaboration.tag` | `""`, appVersion | Where to pull images and which tag. The collaboration image is used only when enabled. |
| `imagePullSecrets` | `[]` | Pull-secret names for a private registry, e.g. `--set imagePullSecrets[0].name=regcred`. |
| `openshift.enabled` | `false` | `true` renders a Route and omits the fixed UID/fsGroup for the SCC. |
| `config.<ENV_VAR>` / `extraConfig.<ENV_VAR>` | see file | Non-secret env (ConfigMap). Provider, auth mode, endpoints, object store, tuning. |
| `secret.existingSecret` | `""` | Name of a Secret you created (production). |
| `secret.data.<ENV_VAR>` | `{}` | Chart-managed secret values (dev only). |
| `migrations.databaseSecret.name/key` | `""`, `INQTRIX_MIGRATION_DATABASE_URL` | Dedicated direct migration DSN, injected only into the hook Job. |
| `migrations.rlsMode`, `migrations.ownerMaintenanceConfirmed` | `auto`, `false` | Migration authority and explicit owner-upgrade maintenance assertion. |
| `serviceAccount.api`, `serviceAccount.worker` | shared account | Optional independent names/annotations/token policy for workload identity. Web, Collaboration and migrations retain tokenless identities. |
| `s3.caBundle.existingConfigMap/key/mountPath` | empty | Optional private CA mounted only into API/worker and exposed as `INQTRIX_S3_CA_BUNDLE`. |
| `ingress.enabled`, `ingress.host`, `ingress.tls.*` | `false` | Vanilla-k8s Ingress. |
| `route.enabled`, `route.tls.termination` | `true`, `edge` | OpenShift Route (only when `openshift.enabled`). `edge` is the only termination that works with the stock plain-HTTP web pod; `reencrypt`/`passthrough` need a TLS-serving backend (and `reencrypt` a `route.tls.destinationCACertificate`). |
| `worker.enabled` | `false` | Durable worker Deployment (needs postgres + valkey). |
| `collaboration.enabled` | `false` | Private one-replica `Recreate` Deployment and ClusterIP Service. Requires Postgres, cookie auth, and an independent secret; it has no HPA/PDB/PVC. |
| `api.replicaCount`, `api.autoscaling.*` | `1`, off | Scale the API (see scaling note). |
| `persistence.enabled`, `persistence.size` | `false`, `5Gi` | PVC for the local object store. |
| `postgres.enabled`, `qdrant.enabled`, `valkey.enabled`, `s3.enabled` | `false` | Bundle a demo backing service in-cluster (`s3.enabled` = MinIO object store). qdrant/valkey/s3 are OpenShift-capable; postgres is vanilla-k8s only. |

## Object store

The chart offers three object-store options:

- **Local** (default): the local backend backed by Kubernetes storage (`emptyDir`,
  or a PVC when `persistence.enabled=true`). Single-pod only — a `ReadWriteOnce` PVC
  is not shared across nodes.
- **Bundled S3 (MinIO)**: `--set s3.enabled=true` starts an in-cluster MinIO and
  auto-wires the object-store backend, endpoint, bucket and credentials. Works on
  vanilla Kubernetes **and** OpenShift; the bucket is created on first upload.
  Demo/dev (single-pod, not HA). Override `s3.accessKey`/`s3.secretKey` for anything
  beyond a local trial (secret key >= 8 characters).
- **External S3** (AWS S3, MinIO, R2, SeaweedFS, ...):

  Set `INQTRIX_S3_AUTH_MODE=static` with keys for an S3-compatible
  provider, or `default` with API/worker ServiceAccount annotations for AWS/
  ROSA workload identity. Native AWS leaves `INQTRIX_S3_ENDPOINT_URL` empty;
  managed buckets should set `INQTRIX_S3_BUCKET_PROVISIONING=existing`.

Use shared S3 (bundled or external) for anything with a worker or more than one API
pod. For production, prefer external/managed S3; put static keys in
`secret.existingSecret`, or grant only API/worker a workload identity. The
complete IAM, CA and SSE-KMS contract is [Object storage](object-storage.md).

## Scaling

The simple defaults (in-memory run store, local object store) support exactly
**one** API replica. To run more than one replica or survive restarts, switch to
the durable backends — all of which Inqtrix already supports:

- `config.INQTRIX_STORAGE_BACKEND=postgres` with `INQTRIX_DATABASE_URL` — the run
  state lives in the database, not in one pod's memory.
- `config.INQTRIX_QUEUE_BACKEND=valkey` + `INQTRIX_VALKEY_URL` and
  `worker.enabled=true` — runs are dispatched to worker pods.
- `config.INQTRIX_OBJECT_STORE_BACKEND=s3` with the S3 credentials, or the bundled
  `s3.enabled=true` (MinIO) — a local PVC is `ReadWriteOnce` and is **not** shared
  across nodes, so multi-pod or `worker.enabled` setups need shared object storage.
  Do not combine `worker.enabled` with the local-object-store `persistence.enabled`
  unless both pods are co-located on one node.

Then raise `api.replicaCount` (or enable `api.autoscaling`) and `worker.replicaCount`.

Editor collaboration is an exception to that horizontal-scaling advice.
Version 1 always renders exactly one collaboration replica with `Recreate` and
database fencing. Do not add an HPA, PDB, or second replica; Redis/Valkey is not
a persistence or multi-writer substitute for this service.

## Upgrades and migrations

Schema migrations run automatically as a Helm hook Job (`inqtrix-migrate`): a
`pre-upgrade` hook on every upgrade (and on install, `pre-install` for an external
database or `post-install` when Postgres is bundled). Helm blocks until it
succeeds, so a failed migration fails the release. A normal upgrade is:

```bash
helm upgrade inqtrix deploy/helm/inqtrix -n inqtrix --reuse-values \
    --set image.api.tag=<NEW-TAG> --set image.web.tag=<NEW-TAG>
```

For collaboration-enabled releases, update
`image.collaboration.tag=<NEW-TAG>` in the same maintenance window so the API,
browser schema, Node schema, and migration stay compatible.

Back up the database (and the object store) before upgrades — see [Runbooks](runbooks.md).
For managed PostgreSQL, give only the Job a direct migration Secret and select
`bypass` or `owner`. For owner upgrades,
`migrations.ownerMaintenanceConfirmed=true` authorizes the chart's bounded
quiesce hook; custom charts must preserve that stop -> one-shot job ->
readiness -> start ordering. Manual
`inqtrix-migrate` is break-glass only. See [Database migrations](database-migrations.md).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Pods `CrashLoopBackOff`, logs show a DB/connection error | `INQTRIX_DATABASE_URL` missing or wrong, or storage backend is `postgres` with no DB | Provide the Secret (scenario C) or bundle Postgres (scenario A/B); verify the URL. |
| Pod `CrashLoopBackOff` with `Read-only file system: 'logs'` | `INQTRIX_LOG_ENABLED=true` under the read-only root filesystem | Logs already go to container stdout (captured by the cluster). Leave file logging off; the chart mounts an ephemeral `/app/logs` so the flag does not crash, but stdout is the durable sink. |
| `helm install` hangs on the migrate hook | The database is unreachable from the cluster | Check `kubectl -n inqtrix logs job/inqtrix-migrate`; fix connectivity/credentials. |
| Migration fails with SQLSTATE `28000` / `inqtrix.tenant_id is not set` | A normal managed-PostgreSQL role touched a forced-RLS table, or the custom chart reused the runtime credential | Configure the migration-only Secret and explicit `bypass`/`owner` mode; do not set a tenant env variable. See [Database migrations](database-migrations.md). |
| `/readyz` says database unavailable and product routes return `database_not_ready` | New runtime image is running against an old schema, cannot `SET ROLE`, or the effective role is privileged | Inspect the failed migration Job and role grants. Do not bypass readiness or start workers until the contract passes. |
| Turnkey install with bundled Postgres hangs when `--wait` is added | Helm can wait for normal resources before the post-install migration hook has run | Install scenario A without `--wait`, then run `kubectl wait` as shown above. |
| Bundled services are enabled but `INQTRIX_DATABASE_URL`, `INQTRIX_VALKEY_URL`, `INQTRIX_S3_ACCESS_KEY`/`INQTRIX_S3_SECRET_KEY` or `INQTRIX_QDRANT_API_KEY` is missing | `secret.existingSecret` was set, so the chart cannot render derived secret values into the app Secret | Put the derived secret values into the existing Secret yourself, or use chart-managed `secret.data` for bundled demo installs. |
| S3 uploads fail against `http://seaweedfs:8333` | That hostname is the Compose stack's SeaweedFS; the chart's bundled store is MinIO under a different Service name | Bundle the store with `s3.enabled=true` (auto-wires `INQTRIX_S3_ENDPOINT_URL`), or set `INQTRIX_S3_ENDPOINT_URL` to your external endpoint. |
| Pods rejected on OpenShift (`runAsNonRoot`/SCC error) | `openshift.enabled` not set, or a bundled service image needs a fixed UID | Set `openshift.enabled=true`; use an external database instead of the bundled ones on OpenShift. |
| `ImagePullBackOff` | Registry not reachable or private without a pull secret | Push to a reachable registry; add `imagePullSecrets`. |
| Worker pod stuck `Pending` (Multi-Attach) | `worker.enabled` + `persistence.enabled` share one `ReadWriteOnce` PVC across nodes | Use the S3 object-store backend for the worker (see [Scaling](#scaling)). |
| Collaboration pod is ready but `service_available=false` | API/Node secrets differ, private API is unreachable, or the instance fencing lease was not acquired | Compare Secret references, inspect API and collaboration logs, and verify one collaboration replica only. |

Inspect a deployment:

```bash
kubectl -n inqtrix get pods
kubectl -n inqtrix logs deploy/inqtrix-api
kubectl -n inqtrix logs job/inqtrix-migrate
helm test inqtrix -n inqtrix          # /readyz; also requires S3 when configured
```

## Related docs

- [Deploy editor collaboration](editor-collaboration.md) - secret, readiness, backup, and kill-switch details.
- [Auth modes](auth-modes.md) - full `none`/`apikey`/`local`/`ldap`/`oidc` reference.
- [Settings and env](../configuration/settings-and-env.md) - every environment variable.
- [Runbooks](runbooks.md) - backup, restore, and operational procedures.
