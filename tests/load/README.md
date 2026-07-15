# Collaboration load gate

The load runner uses direct `ws` and Yjs protocol clients. It validates the
routed room field, authenticated `readonly`/`read-write` scope, complete Yjs
sync step two, strict frame consumption, exact update hashes, and durable ACK
sequence fields.

## Commands

- `pnpm load:collaboration:test` runs semantic protocol, fixture, release
  option, one-send, latency-boundary, token-hygiene, and reconstruction tests.
- `pnpm load:collaboration:check` syntax-checks the runner and library.
- `pnpm load:collaboration:dev -- --fixture PATH` runs the clearly labelled,
  parameterizable developer profile (20 sockets and 5 writers by default).
- `pnpm load:collaboration:release -- --fixture PATH` runs semantic and syntax
  gates, then the fixed release profile.

`load:collaboration` remains a developer alias. A developer protocol smoke may
pass `--skip-api-probe`; text and JSON output label the API gate as skipped.
`--help` succeeds only in developer mode. Supplying it to the release command
fails before any fixture is read or any socket is opened.

## Fixed release profile

Release mode cannot override or weaken these architecture values:

- exactly 1,000 sockets and 100 edit-capable writers;
- exactly 20 non-writer visibility observers;
- visible-update p95 strictly below 250 ms;
- durable-ACK p95 strictly below 500 ms;
- loaded API p95 degradation at or below 20% from baseline;
- at least 30 seconds of sustained writes and at least 10 durably acknowledged
  rounds from every writer;
- authenticated rotation of a fresh 60-second lease on every connected socket,
  followed by scheduled refreshes at `refresh_after` and fresh post-restart
  observer sessions.

Release mode rejects `--allow-insecure-tls`, `--skip-api-probe`, capacity or
latency/duration/round overrides, and a shortened post-sample observation
window. The API probe must be HTTPS `/health`; every WebSocket URL must be WSS
on exact path `/collaboration`, have the same origin including effective port,
and carry that exact origin in the session's allowed `origin` field.
The production instance probe must use HTTPS at exact path
`/collaboration/instance` on that same origin.

The 100 writers continuously send a next update after their previous update is
visible to all 20 observers and durably acknowledged. Writes continue while the
loaded API probes run and until the fixed duration and per-writer round floor
are met. The 20 loaded health samples are scheduled across the full sustained
interval, with the final sample starting at least 30 seconds after the first;
the measured sample span is itself a release gate. Durable acknowledgement
sequences must be positive. Any unexpected socket error or close remains fatal
through the post-sample observation window.

Immediately after all sockets authenticate and sync, the runner reissues every
lease and sends the replacement token through the already-open socket. Each
client must receive a new authenticated scope before writes begin. A background
supervisor refreshes any client reaching `refresh_after` while writers and API
probes continue; an HTTP, schema, timing, or socket reauthentication failure is
fatal. This is an exercised rotation gate, not an expiry-TTL preflight shortcut.
The current lease is checked after every sequential reissue response and again
immediately before its socket authentication frame. Equality with `expires_at`
is expired; a later batch or reauthentication crossing that boundary aborts the
whole rotation and no partial cohort is credited. Post-restart observer reissue
uses the same post-response invariant, and fresh observer authentication checks
the newly issued lease again before connecting.

After measurements, the runner calls the restart control while every original
socket remains open. All original sockets must be terminated by an ungraceful
sidecar process restart and report WebSocket code `1006`, which proves transport
loss without a peer close frame. Planned shutdown codes including `1000` and
`1012`, a socket that stays open, or an unchanged production instance
identity/epoch fails the gate. Only then does the runner obtain newly reissued
sessions and connect 20 fresh observer/Y.Doc clients, wait for authentication
and sync step two, and require every marker from the run exactly once on every
observer. Missing, duplicate, or unexpected run markers fail the gate.

## Fixture contract

Generate the fixture immediately before a run from collaboration-session
responses. Fixture version 2 requires at least 1,000 unexpired sessions for one room, at
least 100 with `access: "edit"`, 20 additional non-writers, `api_probe`,
`instance_probe`, `restart_control`, and `session_reissue`:

```json
{
  "version": 2,
  "base_url": "https://app.example.test",
  "api_probe": {
    "contract": "inqtrix-health-v1",
    "url": "/health"
  },
  "instance_probe": {
    "contract": "inqtrix-collaboration-instance-v1",
    "url": "/collaboration/instance"
  },
  "restart_control": {
    "base_url": "https://control.example.test",
    "authorization_env": "INQTRIX_LOAD_RESTART_TOKEN",
    "restart_path": "/v1/test/collaboration/restart"
  },
  "session_reissue": {
    "authorization_env": "INQTRIX_LOAD_REISSUE_TOKEN",
    "contract": "inqtrix-collaboration-session-reissue-v1",
    "lease_ttl_seconds": 60,
    "url": "https://control.example.test/v1/test/collaboration/sessions/reissue"
  }
}
```

The health probe sends no document content or token. Every sample must return
JSON content type and the `inqtrix-health-v1` FastAPI schema: top-level
`status: "ok"`, non-empty `llm.provider`/`llm.status` and
`search.provider`/`search.status`, string `auth_mode`, and object `legal`.
HTML, redirects into the SPA, generic JSON, and malformed health payloads fail.

The restart endpoint receives `{ "room" }` while the sockets are live and must
terminate the collaboration sidecar process without running its shutdown hooks
or sending WebSocket close frames. It returns HTTP 2xx JSON only after a
replacement process is ready:

```json
{
  "state": "ready",
  "restart_kind": "ungraceful_process"
}
```

The runner does not trust instance values returned by that controller. Before
the control call and after every original socket has failed abnormally, it
independently sends an unauthenticated, content-free GET to the production
`instance_probe` endpoint. The endpoint must be served from the public
FastAPI collaboration data plane at the fixed `/collaboration/instance` path,
not by the fixture controller, and return JSON content type plus
`Cache-Control: no-store` with this schema:

```json
{
  "contract": "inqtrix-collaboration-instance-v1",
  "service": "inqtrix-collaboration",
  "status": "ready",
  "instance_id": "sidecar-b",
  "epoch": 18
}
```

Both observed instance IDs must be non-empty and different, both epochs must be
positive, and the observed post-restart epoch must advance. A controller-hosted
identity endpoint or controller-self-attested before/after fields do not satisfy
the contract. Release preflight fixes the path and requires this probe to share
the exact HTTPS origin, including effective port, with the public WSS gateway;
the fixture cannot redirect it to the separate control service. The named
environment variable supplies the control bearer value, and the control URL
must use HTTPS in release mode.

Each version-2 session is a collaboration-session API response plus a unique,
opaque `reissue_id`. Required fields are `room`, `lease_token`, `expires_at`,
`refresh_after`, `websocket_path` (or `websocket_url`), `access`,
`initial_write_mode`, `protocol_version`, `schema_version`, and `user.id`;
`origin` remains optional. The initial lease only needs to be unexpired: release
qualification depends on exercised reissue and socket reauthentication rather
than a larger minimum initial TTL.

The reissue controller must have generated the fixture and retain, only in
protected memory, the mapping from each `reissue_id` to its cookie-authenticated
user, document, and current lease. For every requested rotation it calls the
real FastAPI collaboration-session endpoint with that authenticated identity,
current lease, and supplied rotation command ID. It must not fabricate leases
or return permanent login credentials. The runner sends no lease token or
document content to the controller:

```json
{
  "contract": "inqtrix-collaboration-session-reissue-v1",
  "lease_ttl_seconds": 60,
  "purpose": "connected_rotation",
  "sessions": [
    {
      "reissue_id": "load-session-0001",
      "rotation_command_id": "00000000-0000-4000-8000-000000000001"
    }
  ]
}
```

`purpose` is `connected_rotation`, `scheduled_rotation`,
`post_restart_observer`, or `fresh_observer` in developer runs. A successful
response has JSON content type, `Cache-Control: no-store`, matching contract and
TTL, `source: "fastapi_collaboration_session"`, and one correlated item per
request. Each item echoes `reissue_id` and `rotation_command_id` and embeds the
unmodified FastAPI session response under `session`. The harness rejects a
reused token; changed room, user, access, protocol, schema, or WebSocket path;
invalid refresh timing; and a supposedly fresh 60-second lease with less than
45 or more than 65 seconds remaining.

Probe, control, and WebSocket URLs may not contain credentials, queries, or
fragments. Response bodies are parsed only for contract validation and never
printed. Control services must redact authorization and lease values from their
own request logging. Tokens remain in memory and are never included in progress,
results, URLs, requests to the reissue control, or errors.

Generated lease/session/token fixtures are ignored under `tests/load/`. Do not
commit them. The checked-in example contains nonfunctional placeholder values.
