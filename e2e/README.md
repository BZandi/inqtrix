# Collaboration browser gate

The Playwright suite targets an external, disposable collaboration stack. It
does not mint credentials or start product services.

## Commands

- `pnpm e2e:tooling:test` runs semantic tests for fixture parsing, release
  preflight, control responses, token-safe errors, and release-fatal skips.
- `pnpm e2e:typecheck` compiles all Playwright tooling.
- `pnpm e2e:list` lists the developer matrix without requiring a fixture.
- `pnpm e2e:dev` runs configured endpoints and explicitly skips unavailable
  external prerequisites with the precise reason.
- `pnpm e2e:release` is the release gate. It runs type and semantic checks,
  requires the complete version-2 fixture, and fails if any selected test is
  skipped. It accepts no forwarded arguments: `--help`, `--list`, reporter
  overrides, grep/project/file selectors, and every other Playwright override
  fail before Playwright starts. Use `e2e:dev` or `e2e:list` for those controls.

`e2e:test` remains a developer alias and is not a release command.

The release launcher resolves Playwright and the repository root from its own
`import.meta.url`, so invoking its absolute path from another working directory
cannot bypass the fixture preflight. It rejects `NODE_OPTIONS`, `PWDEBUG`, every
`PWTEST_*` and `PW_TEST_*` variable, and every unapproved `PLAYWRIGHT_*`
variable. This blocks watch, source-transform, reporter, selector, and related
runner overrides. Fixture/auth variables, `CI`, proxy variables, normal process
runtime variables, `PLAYWRIGHT_BROWSERS_PATH`, and the official browser download
host/timeout/skip variables remain available.

## Release fixture

Copy `e2e/fixture.example.json` outside version control and set
`INQTRIX_E2E_FIXTURE` to its path. Storage-state files under `e2e/.auth/` are
ignored. Release mode requires:

- owner and collaborator storage states, a UUID collaborator user ID, and
  display names;
- a structured `documents.suggestion` object containing the disposable
  `documentId`, literal `expectedPermission: "suggest"`, and
  `expectedAuthorId` equal to the collaborator user ID;
- disposable direct-edit, revocation, reconciliation, sidecar-outage,
  FastAPI-gateway-outage,
  protocol-rejection, concurrent-edit, permission-downgrade, and detached
  export/import documents;
- a private-anchor document containing a non-empty private AI range and private
  comment range for each user, with the exact anchor and private card text in
  `aiAnchorText`, `aiText`, `commentAnchorText`, and `commentText`;
- Vite, nginx, and built-dist HTTP(S) URLs that normalize to three distinct
  origins (scheme, host, and effective port); configured application base paths
  are preserved during navigation. Each endpoint must also expose its shipped,
  observable transport identity: a real Vite `@vite/client` module, an nginx
  `Server` header, or a Uvicorn `Server` header for the dist launcher;
- every fixture-control path and the bearer value in the environment variable
  named by `controls.authorizationEnv`.

The suite runs direct editing, genuine concurrent editing, exact remote-caret
identity/position, transparent selection-overlay exclusivity, suggestion
accept/reject plus lease/author identity and direct-Yjs policy rejection,
genuine Chromium CDP IME composition, permission downgrade and recovery,
revocation close/hidden-404 status, lost-ACK reconciliation, independently
observable sidecar and public FastAPI/gateway outages, Source read-only
projection, private anchors, detached export/import, protocol rejection, and
full editor/Inspector layout checks on all three desktop and mobile transports.
Collaboration-critical scenarios run on both desktop and mobile projects;
mobile also verifies exclusive modal tree and inspector drawers.

Transport identity is not declared by the fixture or controller. Every desktop
and mobile project probes its own base URL, validates an HTML application root,
and checks the hardcoded runtime evidence above. Distinct origins that all
resolve to the same server therefore fail the release gate.

The private-anchor scenario checks real editor decorations and exact ranges,
rebases those ranges after a collaborator inserts before them, reloads while
connected, and proves the other user cannot see either private decoration or
private card. The detached-transfer scenario downloads the real project archive,
imports it through the product UI into a fresh authenticated context, and fails
if the imported copy attempts a collaboration session or WebSocket connection.

## Fault-control contract

All control calls are authenticated `POST` requests with JSON bodies. The
client never prints the URL, response body, authorization header, or bearer
value. Paths must be absolute and contain no query or fragment.

- `armLostAckPath` receives `{ "document_id", "user_id" }`. It arms exactly one
  update to be durably persisted, drops that update's `durable_ack`, and closes
  the affected socket with code `1012`.
- `armOutagePath` receives `{ "document_id", "user_id" }`. It arms exactly one
  update to be durably persisted, prevents its projection, then takes the real
  collaboration sidecar out of service and closes with code `4503`. The public
  FastAPI `/health` endpoint must remain healthy.
- `armGatewayOutagePath` receives `{ "document_id", "user_id" }`. It
  ungracefully terminates or isolates the real public FastAPI collaboration
  gateway while the browser sockets remain open. The external fixture-control
  service must remain reachable. Browser sockets must lose transport without a
  close frame (`1006`), and both `/health` and collaboration-session requests
  must fail by transport error or HTTP 500/502/503/504.
- `operationStatusPath` receives `{ "operation_id" }`.
- `restorePath` receives `{ "operation_id" }` and restores a ready sidecar.
- `restartPath` receives `{ "document_id" }` and returns only after durable
  state can be reconstructed. It is reserved for fixture lifecycle checks.

Every response contains `operation_id` and one of `armed`, `triggered`,
`outage`, `ready`, or `failed` in `state`. Sidecar-outage operation responses
contain `outage_layer: "collaboration_sidecar"`; FastAPI/gateway-outage arm,
status, and restore responses contain `outage_layer: "fastapi_gateway"`. A
triggered lost-ACK response contains
`close_code: 1012` and a positive `durable_sequence`. Controller claims that an
ACK was dropped are deliberately ignored. Before arming the fault, the browser
must visibly be `Saved`/`Gespeichert` with no pending saving state. A WebSocket
observer on the actual `/collaboration` browser socket parses routed stateless
frames and proves that no valid `durable_ack` of any sequence reached the
original client before its observed `1012` close. It records only
socket IDs, event order, close codes, and positive sequence numbers, never
frame content or hashes; any undecodable frame in the fault window fails the
test instead of becoming evidence that no ACK arrived. After a different socket reconnects,
`operationStatusPath` must report `durability_reconciled: true`,
`pending_durability_count: 0`, and a `reconciliation_sequence` greater than or
equal to that durable sequence. The browser must then visibly settle on
`Saved`/`Gespeichert` with no remaining `Saving`/`Wird gespeichert` status
before reload, after which the update must remain exactly once. An outage
response contains `close_code: 4503`, a positive `durable_sequence`, and
`projection_sequence`, with the durable sequence strictly greater than the
projection sequence.

The outage scenario requires collaboration session and projection-flush calls
to return HTTP 503 while ordinary document detail remains available as a stale
projection. Its `projection_updated_at` must remain unchanged until restore.
This is intentionally separate from the FastAPI/gateway outage, where `/health`
itself becomes unavailable and the browser must observe `1006`. After public
gateway restore, `/health` must again match the FastAPI JSON schema, both users
must reconnect, and a pre-outage durable marker must survive exactly once after
reload.

The suggestion scenario first verifies the accepted document share is really
`suggest`, then parses a newly issued collaboration session and requires
`access`, `initial_write_mode`, and `user.id` to match the fixture authority.
Each real suggestion mark must carry that author ID. A raw, fully synced client
then sends a manipulated direct Yjs update through the same suggest lease; the
server must close it with `4403`, issue no matching durable ACK, leave persisted
and projection sequences unchanged, and expose no marker to either user.

Revocation requires the established socket to close with `4403`. A newly
invisible shared document must return HTTP 404 for both detail and session
requests; HTTP 403 is reserved for a document that remains visible but forbids
the attempted operation.

The permission-downgrade scenario changes a live collaborator share from
`edit` to `view` with the fixture's authenticated API. Before the downgrade it
opens and fully syncs a second raw Hocuspocus/Yjs socket with `read-write`
scope. On the server's policy reauthentication challenge, that same socket
sends a real Yjs update before answering. The test requires explicit
`readonly`, authentication denial, or access-revoked close behavior, no matching
durable ACK, and no sequence or document mutation. It then observes the product
editor and a new session become read-only, restores `edit`, and verifies
reconnection and edits.

The Source scenario enters the real collaboration Source view, requires both
write-mode controls to be disabled with the product's read-only reason, proves
a remote CRDT edit appears and disappears in the Source projection, and proves
keyboard input cannot alter UI or canonical document state. Layout assertions
cover the editor main surface, its scroll viewport, the full Inspector surface,
every control that is even partially visible in either surface, desktop
editor/Inspector separation, and mobile dialog bounds. The layout scenario
creates a real suggestion, opens Changes, expands its populated row, and
requires markup modes, author/type filters, per-change decisions, and batch
controls before measuring geometry. A partially clipped control fails bounds
analysis even when its center lies outside the viewport.
The release reporter treats every missing or skipped required scenario in every
transport project as a release failure.

Tests use unique markers and normally remove them. Documents must still be
disposable because an interrupted run can leave data behind.
