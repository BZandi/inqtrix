# Deploy to production

## Scope

What to change when you take the product compose from a laptop to a real
deployment: TLS, secrets, cookie security, backups, and the hardening of each
opt-in backend. It assumes you already have the stack running locally
([Stack quickstart](../getting-started/stack-quickstart.md)) and builds on
the component map ([Platform components](../getting-started/platform-components.md)).
It is not a Kubernetes manifest — the same env contract applies whatever the
orchestrator.

## TLS and the public URL

Terminate TLS at your ingress/load balancer (or the bundled nginx behind one)
and serve the app over HTTPS. The public origin must match how the browser
reaches the app, because cookies and the OIDC callback derive from it:

```dotenv
INQTRIX_PUBLIC_BASE_URL=https://research.example.com
```

Over HTTPS, leave secure cookies ON — that is the default. **Never** set
`INQTRIX_OIDC_INSECURE_DEV_COOKIES=true` in production; it drops the `Secure`
flag and the `__Host-` prefix and exists only for `http://localhost` dev.

## Secrets

These must be set to real, high-entropy values and, for multi-replica
deployments, **shared identically** across every api/worker replica (a session
minted on one replica must validate on another):

| Secret | Why |
|---|---|
| `INQTRIX_SESSION_SECRET` | HMAC for the session/CSRF cookies (local/ldap/oidc). |
| `INQTRIX_PAT_PEPPER` | Pepper for personal-access-token hashing. |
| `INQTRIX_PG_PASSWORD` + the password inside `INQTRIX_DATABASE_URL` | Database auth (must match). |

Generate them with a CSPRNG and inject via your secret manager, not the repo:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"   # one per secret
```

Any `CHANGE_ME_*` placeholder left in place is logged loudly at startup as
insecure (No Silent Fallbacks) — treat that log line as a release blocker.

## Storage durability

Production runs the Postgres backend; migrations are applied by the one-shot
`migrate` service (or `uv run inqtrix-migrate`) before the api starts. Without
Postgres everything is in-memory and lost on restart, including sessions and
the user mirror — so a restart logs everyone out.

## Backups

Back up two things:

- **Postgres** — runs, identity/credentials, knowledge metadata, prompt
  templates, quotas. Use your standard `pg_dump`/snapshot cadence.
- **The object store** — uploaded file blobs. The default `local` backend writes
  to a volume; snapshot it, or move to `--profile s3` (SeaweedFS or any S3
  endpoint) for replicated/scalable storage.

Restore is the inverse: restore Postgres, restore the blob volume/bucket, then
start the stack (migrations are idempotent).

## Hardening the opt-in backends

| Backend | Hardening |
|---|---|
| Qdrant (`--profile knowledge`) | Self-hosted Qdrant is unauthenticated by default — set `INQTRIX_QDRANT_API_KEY` and keep it off the public network. |
| Valkey + workers (`--profile workers`) | Set a strong `INQTRIX_VALKEY_PASSWORD`; the worker refuses to start without Postgres + Valkey. |
| Object store S3 (`--profile s3`) | Real credentials, a private bucket, TLS to the endpoint. |
| OIDC (`--profile oidc` / your IdP) | Register the exact `https://…/api/auth/callback`; the bundled Dex is a dev reference, not for production. |

Misconfiguration fails loudly at startup (e.g. `INQTRIX_QUEUE_BACKEND=valkey`
without a URL, or `valkey` without `postgres`) rather than degrading silently.

## Auth in production

Pick a real auth mode — `local` (email/password, owner setup), `ldap` (your
directory, [guide](connect-to-existing-ldap.md)), or `oidc` (your IdP). The
single-operator `none`/`apikey` modes have no scoped identity and no sharing;
use them only for a locked-down single-user instance.

## Login rate limiting

The credential check is uniform-timing (no email-enumeration oracle) AND the
login endpoints are throttled: after `INQTRIX_LOGIN_RATE_LIMIT_MAX_ATTEMPTS`
failures (default 10) for a `(identifier, source_ip)` within
`INQTRIX_LOGIN_RATE_LIMIT_WINDOW_SECONDS` (default 300) the key is locked for
`INQTRIX_LOGIN_RATE_LIMIT_LOCKOUT_SECONDS` (default 60) and further attempts get
`429`. On by default; tune or disable via `INQTRIX_LOGIN_RATE_LIMIT_ENABLED`.

The counters are **process-local**, so a **multi-replica** deployment should
ALSO rate-limit per-IP at the reverse proxy / WAF (e.g. on `/api/auth/login/*`)
— the in-app limit then complements, not replaces, the edge limit.

The client IP is read from the **right** of `X-Forwarded-For`, at the depth set
by `INQTRIX_TRUSTED_PROXY_HOPS` (default `1`, matching the single bundled proxy
— the nginx `web` container or `scripts/run_research_desk.py`, both of which
append the real peer). The right-most hop is written by that proxy and is not
client-spoofable, so an attacker cannot rotate a forged left-most value to mint
a fresh throttle key per attempt. Set it to the **exact** number of chained
proxies in front of the server (e.g. `2` for an external load balancer in front
of the bundled proxy); a value **higher** than the real chain lets a client
backfill the gap and spoof again. Set it to `0` only when the API server is
exposed directly with no reverse proxy — then just the socket peer is trusted.

## Pre-flight checklist

- [ ] HTTPS at the ingress; `INQTRIX_PUBLIC_BASE_URL` is the public `https://` origin.
- [ ] `INQTRIX_OIDC_INSECURE_DEV_COOKIES` unset/false.
- [ ] `SESSION_SECRET`, `PAT_PEPPER`, `PG_PASSWORD` real and shared across replicas; no `CHANGE_ME_*` in the startup log.
- [ ] Postgres backend + migrations applied; backups scheduled for Postgres and the blob store.
- [ ] Each enabled profile hardened (Qdrant key, Valkey password, S3 creds, OIDC callback).
- [ ] A real auth mode chosen; login endpoints rate-limited at the proxy.

## Related docs

- [Stack quickstart](../getting-started/stack-quickstart.md) — the local starting point.
- [Platform components](../getting-started/platform-components.md) — which backend each feature needs.
- [Runbooks](../deployment/runbooks.md) — start/stop/update/restore operations.
- [Auth modes](../deployment/auth-modes.md) — choosing and wiring auth.
