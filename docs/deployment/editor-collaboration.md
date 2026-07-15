# Deploy Editor Collaboration

## Scope

This page configures and operates the optional Node/Hocuspocus service, the
same-origin WebSocket gateway, PostgreSQL persistence, backup/restore,
retention, security limits, and the read-only kill switch. It covers the
bundled Compose and Helm paths plus host development. It does not describe
end-user review controls or claim horizontal availability, offline writes, or
guest access.

## Prerequisites

Collaboration is disabled by default. Enabling it requires all of the
following:

- `INQTRIX_STORAGE_BACKEND=postgres` with migration
  `0048_editor_collaboration` applied;
- cookie-session authentication: `INQTRIX_AUTH_MODE=local`, `ldap`, or `oidc`;
- the canonical PostgreSQL session/user stores used by those auth modes;
- the private Node HTTP and WebSocket URLs;
- an independent `INQTRIX_COLLABORATION_SECRET` of at least 32 UTF-8 bytes,
  different from `INQTRIX_SESSION_SECRET`; and
- one and only one collaboration service replica.

When `INQTRIX_COLLABORATION_ENABLED=true`, an invalid storage/auth setup,
missing URL, weak/reused secret, or contradictory timing value stops startup.
The server does not silently disable the requested feature. With the flag left
false, the legacy editor and all non-collaboration features keep their normal
behavior.

## Network topology

Only the FastAPI gateway is browser-facing:

```text
Browser ws(s)://public-host/collaboration
  -> Vite, nginx, or scripts/run_research_desk.py
  -> FastAPI /collaboration
  -> private ws://collaboration:1234/collaboration
```

FastAPI also calls private Node HTTP operations for conversion, projection,
suggestion publication, and decisions. Node calls FastAPI internal APIs for
lease introspection, state loading, durable updates, snapshots, policy events,
and compaction. Both directions use the collaboration bearer secret. Node has
no database URL, no host port, and no persistent volume.

The frontend production image remains nginx-only. The collaboration service
uses its own multi-stage image based on the same `node:22-bookworm-slim` image
family as the web build. Packages are installed from the workspace lockfile at
image build time; the container does not run `npx` or download packages at
startup.

## Configuration

The minimum host configuration is:

```dotenv
INQTRIX_STORAGE_BACKEND=postgres
INQTRIX_DATABASE_URL=postgresql+asyncpg://inqtrix:change-me@127.0.0.1:5432/inqtrix
INQTRIX_AUTH_MODE=local
INQTRIX_SESSION_SECRET=replace-with-an-independent-session-secret
INQTRIX_PAT_PEPPER=replace-with-an-independent-pat-pepper

INQTRIX_COLLABORATION_ENABLED=true
INQTRIX_COLLABORATION_HTTP_URL=http://127.0.0.1:1234
INQTRIX_COLLABORATION_WS_URL=ws://127.0.0.1:1234/collaboration
INQTRIX_COLLABORATION_SECRET=replace-with-at-least-32-random-characters
```

Start Node with a deliberately narrow environment containing only the private
API URL, collaboration secret, bind/port, and collaboration limits. Do not pass
the complete API `.env` into the Node process:

```bash
corepack pnpm --filter @inqtrix/collaboration-server build
INQTRIX_API_INTERNAL_URL=http://127.0.0.1:5100 \
INQTRIX_COLLABORATION_SECRET="$INQTRIX_COLLABORATION_SECRET" \
INQTRIX_COLLABORATION_TENANT_ID=default \
corepack pnpm --filter @inqtrix/collaboration-server start
```

The tenant value must match the deployment's canonical `Principal.tenant_id`.
Version 1 is intentionally single-tenant per deployment and therefore uses
`default` unless the whole deployment is configured for another canonical
tenant.

Every API and sidecar variable, default, and interaction is listed in
[Settings and environment variables](../configuration/settings-and-env.md#editor-collaboration-collaborationsettings).

## Compose

The `collaboration` profile adds only the private Node container. Set the
feature flag and secret in the same env file used for Compose interpolation
and API `env_file`, then start:

```bash
CF="-f deploy/compose/compose.stack.yaml --env-file deploy/.env.stack"
docker compose $CF --profile collaboration up -d --build
docker compose $CF ps
docker compose $CF logs -f api collaboration
```

The API service already receives the private URLs
`http://collaboration:1234` and
`ws://collaboration:1234/collaboration`; they are inert while the feature flag
is false. The Node container has a read-only root filesystem, drops Linux
capabilities, runs non-root, and publishes no host port.

The loopback stack defaults to HTTP. When a reverse proxy terminates TLS in
front of Compose, set both values in the stack environment and preserve the
public Host at that proxy:

```text
INQTRIX_PUBLIC_BASE_URL=https://desk.example
INQTRIX_EXTERNAL_SCHEME=https
```

The API consumes the full public origin; the web container consumes the scheme
and overwrites any client-supplied forwarding value. A mismatch fails
collaboration origin validation rather than falling back to HTTP.

Do not start the profile without `INQTRIX_COLLABORATION_ENABLED=true`, or set
the feature flag without starting Node. Both are visible misconfigurations:
the former starts an unused private service, while the latter reports the
configured capability as unavailable and collaboration operations fail.

## Helm

The chart defaults to `collaboration.enabled=false`. Build and publish
`inqtrix-collaboration` alongside the API and web images, store the bearer in a
Kubernetes Secret, and enable the component:

```yaml
image:
  collaboration:
    repository: registry.example.com/inqtrix-collaboration
    tag: "0.2.0"

collaboration:
  enabled: true
  secret:
    existingSecret: inqtrix-collaboration
    key: INQTRIX_COLLABORATION_SECRET
```

The Secret value must also be available to the API pod under the same key and
must differ from the session secret. Enabling the value renders a one-replica
Deployment with `Recreate`, a private ClusterIP Service, and no HPA, PDB, or
PVC. The chart derives the private HTTP/WS URLs and feature flag. An enabled
Ingress also derives `INQTRIX_PUBLIC_BASE_URL` and the nginx external scheme
from its host/TLS settings; an explicit `config.INQTRIX_PUBLIC_BASE_URL` keeps
precedence. Do not change the replica count: version 1 has a database fencing
lease for fail-safe replacement, not multi-replica fan-out.

An OpenShift Route with an empty `route.host` receives its hostname only after
admission, too late for Helm to derive the API trust anchor. With collaboration
enabled, the chart therefore requires either an explicit `route.host` or the
final `config.INQTRIX_PUBLIC_BASE_URL` and fails rendering when neither exists.

## Vite, nginx, and the dist launcher

- Vite proxies `/collaboration` with WebSocket support to FastAPI during
  development; the same prefix carries the HTTP instance probe.
- The bundled nginx configuration forwards WebSocket Upgrade traffic for
  `/collaboration` to FastAPI while serving the SPA on the same origin. It
  overwrites forwarded proto/host from the deployment-owned external scheme
  and request Host; an exact HTTP location forwards
  `/collaboration/instance` through the same upstream instead of SPA fallback.
  It never relays client-supplied forwarding chains.
- `scripts/run_research_desk.py` serves an existing `dist/` directory and
  performs a real bidirectional binary WebSocket relay plus an explicit HTTP
  instance-probe proxy to FastAPI. Direct HTTP derives forwarding metadata from
  the ASGI connection. When TLS terminates before the launcher, set the exact
  `INQTRIX_PUBLIC_BASE_URL`; the launcher then overwrites forwarded scheme and
  host from that value and never trusts incoming `X-Forwarded-Proto` or
  `X-Forwarded-Host`.

The Python launcher never starts or targets Node directly. Dist mode is fully
functional when the separately started API and optional collaboration service
are healthy. Without Node, the Research Desk still starts, but collaboration
documents are read-only and the capability reports the service unavailable.

## Readiness and verification

Node exposes private endpoints:

| Endpoint | Meaning |
|---|---|
| `GET /health/live` | Process is running. |
| `GET /health/ready` | Internal FastAPI is reachable and this instance owns the active fencing lease. |
| `GET /metrics` | Sidecar Prometheus text metrics; keep private to the service network. |

The public capability response is the operator/user contract:

```bash
curl -s http://127.0.0.1:5100/v1/capabilities
```

Expect `collaboration.configured=true`,
`collaboration.service_available=true`, transport path `/collaboration`, and
mode `single_replica`. `configured=true` with `service_available=false` means
the API accepted its configuration but Node is not ready or reachable.

Release verification may also call the unauthenticated, content-free probe:

```bash
curl -s http://127.0.0.1:5100/collaboration/instance
```

A ready response has contract `inqtrix-collaboration-instance-v1`, service
`inqtrix-collaboration`, status `ready`, and the current non-empty
`instance_id` plus positive fencing `epoch`. FastAPI reads that identity from
PostgreSQL before and after checking the private Node readiness endpoint. A
missing, expired, changing, or unreachable data plane returns JSON status
`not_ready` with HTTP 503 and is never replaced by fixture/controller data.

## Backup and restore

There is no sidecar data volume to back up. A PostgreSQL backup contains all
collaboration document metadata, binary updates, verified snapshots, Markdown
projections, leases, attribution, and decision rows. Continue backing up the
object store for the rest of Inqtrix, but it is not the collaboration body
source of truth.

For a consistent collaboration backup, stop or drain the API and Node writer,
then take the ordinary PostgreSQL dump. Restore with matching application
binaries, schema/protocol versions, and migrations:

1. Stop API and collaboration processes.
2. Restore PostgreSQL using the normal [Runbooks](runbooks.md#restore)
   procedure.
3. Apply only migrations that belong to the restored application version.
4. Start FastAPI, then start exactly one Node instance.
5. Require Node readiness and `service_available=true`.
6. Open a document and verify that Node reconstructs it from the latest
   snapshot plus the remaining update tail.

Do not restore only `content_markdown` and attempt to recreate Yjs state. That
loses merge history and can duplicate content. A rollback across the
collaboration migration requires a matching pre-migration database backup and
old binaries; schema downgrade alone is not a safe recovery procedure.

## Retention and deletion

Snapshots are requested after 5 seconds of inactivity, 256 durable updates, or
a 1 MiB update tail by default. After a verified covering snapshot:

- binary update payloads may be pruned after 24 hours;
- attribution and decision metadata may be pruned after 90 days;
- only the latest two verified snapshots are retained; and
- tombstoned documents may be physically purged after 90 days.

Compaction is fenced and runs on an independent periodic timer in the active
Node instance, so retention does not depend on a document receiving another
edit or snapshot. Metadata retention is an operational history window, not a
complete version-restore system. Set longer periods before rollout when legal,
audit, or incident-response requirements demand them.

An eligible loaded room retries a failed snapshot autonomously with capped
exponential backoff. The default delay starts at 1 second and caps at 30
seconds; success, unload, shutdown, or loss of snapshot eligibility cancels the
single pending retry slot.

## Kill switch and maintenance

Set `INQTRIX_COLLABORATION_ENABLED=false` and restart FastAPI to disable public
sessions, internal APIs, and the WebSocket gateway. Then stop Node. This is a
read-only kill switch, not a conversion or deletion:

- legacy Markdown documents continue to edit and autosave normally;
- collaboration documents retain their Yjs state in PostgreSQL;
- users can see only the last stored Markdown projection; and
- no client falls back to full-body Markdown writes or offline writes.

Re-enable only with the matching schema/protocol build and one healthy Node
instance. A planned Node restart closes sockets with `1012`; clients become
read-only and reconnect after readiness returns.

## Security and limits

- Terminate TLS at the public proxy so browsers use `wss://`.
- Keep `INQTRIX_PUBLIC_BASE_URL`, the public proxy host, and its TLS mode
  aligned. FastAPI accepts forwarded same-origin only when browser Origin,
  sanitized forwarded proto/host, and that configured public origin all match.
- Keep Node and `/metrics` private; expose only nginx/web and FastAPI's
  same-origin gateway.
- The public instance probe is deliberately unauthenticated and `no-store`; it
  exposes only fixed contract labels plus readiness, process instance ID, and
  fencing epoch. It contains no lease token, user, room, document, or content.
- Allow additional browser origins only through
  `INQTRIX_COLLABORATION_ALLOWED_ORIGINS`; CORS settings do not replace the
  WebSocket Origin check.
- Rotate the collaboration secret by restarting API and Node together. Never
  put it in a URL, ConfigMap, log, metric label, or frontend build.
- Default limits are 2 MiB per frame, 10 MiB per document, five sessions per
  user/document, 30 session issuances per minute, 120 updates per 10 seconds,
  and 20 awareness messages per second.
- Before Hocuspocus processing, each physical socket is limited to 32 reserved
  frames, 8 MiB of reserved frame bytes, and 4 MiB of outbound socket
  backpressure by default. Queue or backpressure exhaustion closes with `4429`;
  an individual oversize frame closes with `1009`.
- Account disable, share revoke/downgrade, generation change, and session
  invalidation propagate through the existing `user_events` feed. A policy
  polling failure emits a warning and metric; current readiness reflects the
  instance fencing lease, so alert on that metric separately. Short lease
  rotation still rechecks live access and bounds stale authorization.

Current sidecar metrics include active connections and rooms, document queue
depth, validation/persistence/durable-ack latency, bounded rejection reasons,
instance readiness/renew failures, internal HTTP latency/errors, policy polling
and revalidation, snapshot success/failure, and compaction results. Labels
exclude tokens, document content, comments, update bytes, and high-cardinality
room or user identifiers. Document metadata exposes persisted and projection
sequences for targeted lag diagnosis; the sidecar does not add document IDs to
Prometheus labels.

## Unsupported deployment shapes

Version 1 does not support multiple Hocuspocus replicas, Redis/Valkey fan-out,
offline writes, public/guest links, or a direct browser-to-Node route. The
Redis extension would distribute live state but would not replace PostgreSQL
persistence. Do not add a second replica until a separately designed HA mode
defines fan-out, leader/fencing behavior, rolling schema compatibility, and
load-tested failure handling.

## Related docs

- [Editor collaboration architecture](../architecture/editor-collaboration.md)
- [Collaborate on editor documents](../how-to/collaborate-on-editor-documents.md)
- [Runbooks](runbooks.md)
- [Security hardening](security-hardening.md)
