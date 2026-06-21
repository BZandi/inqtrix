# Security hardening

> Files: `src/inqtrix/server/security.py`, `src/inqtrix/server/routes.py`, `src/inqtrix/auth/api_key.py`

## Scope

Opt-in hardening layers the HTTP server ships with: TLS at the uvicorn layer, Bearer API-key authentication, and CORS allow-listing. All three are off by default and driven by `ServerSettings`. Authentication is mode-driven (`none` | `apikey` | `oidc`); this page covers the `apikey` gate — for the full mode matrix, the inference rule, and OIDC browser sessions see [Authentication modes](auth-modes.md). This is not a substitute for a real reverse-proxy-fronted production stack — it is the minimum viable hardening for self-hosted experimentation.

> Inqtrix is explicitly experimental. The root `README.md` disclaimer applies. Secure configuration, hardening, deployment architecture, access control, secret handling, and compliance remain the operator's responsibility.

## The three layers

### TLS termination in uvicorn

Set both `INQTRIX_SERVER_TLS_KEYFILE` and `INQTRIX_SERVER_TLS_CERTFILE` to local PEM file paths. Both are required; setting only one raises `RuntimeError` on startup (no silent fallback).

The values are passed to `uvicorn.run(..., ssl_keyfile=..., ssl_certfile=...)`. For production, prefer terminating TLS at a reverse proxy (nginx, Traefik, Azure Application Gateway) whenever possible — built-in TLS exists for small self-hosted setups where a proxy would be overkill.

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

`api_key_gate` reflects whether the resolved auth mode is `apikey`; `auth_mode` names the resolved mode (`none` | `apikey` | `oidc`). Any deviation from the expected line is a deployment mistake; fail the smoke test fast.

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
- [Web server mode](webserver-mode.md)
- [Enterprise Azure](enterprise-azure.md)
- [Settings and env](../configuration/settings-and-env.md)
- [Troubleshooting](../reference/troubleshooting.md)
