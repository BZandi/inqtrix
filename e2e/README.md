# Verification profiles for browser and system testing

Browser, live-system, fault, collaboration-load, and web-edge checks share one
orchestrator:

```bash
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON \
  --experimental-strip-types \
  tests/verification/cli.ts \
  --profile ui-fixture
```

External profiles do not start the product stack. When no external fixture is
supplied, `system-smoke` and `fault-injection` provision Run-ID-bound browser
accounts and documents against the already running active gateway;
`fault-injection` additionally starts a loopback-only controller for the exact
Compose services selected through `--container-engine`. The collaboration
service must have been explicitly recreated with
`INQTRIX_COLLABORATION_VERIFICATION_FAULTS=1`; the default is disabled and the
service must be recreated without that variable after the fault run.
`load-smoke` creates
four Run-ID-bound temporary
accounts, one collaboration document, shares, and 20 short-lived leases
through the public product APIs. `load-soak` creates 24 temporary accounts plus
the owner, runs a 30-minute mixed workload with scoped Podman network shaping,
and verifies real feature activity and resource recovery. The isolated
`edge-conformance` profile builds
both real web targets and starts only run-labelled synthetic local containers.

This harness supplies reusable release evidence; it is not by itself an
Enterprise-ready claim. A real Safari smoke run, production-scale external
capacity, supported network-shaping profiles, and any unavailable live
security/recovery matrix remain visible external gates.

## Profiles and engines

The scenario inventory in
[`tests/verification/scenario-inventory.ts`](../tests/verification/scenario-inventory.ts)
is the single source for profile membership and required Playwright tags.

| Profile | Engine adapters | Purpose |
| --- | --- | --- |
| `ui-fixture` | `ui-fixture-playwright` | Deterministic browser fixtures without an external product stack |
| `system-smoke` | `collaboration-playwright`, `editor-system-live` | Visible multiuser browser paths, role matrix, comments, navigation, and layout |
| `agent-desk` | `agent-desk-live` | Kernel answer experience through the visible Agent Desk: submission, live tool activity, clickable citations, console/network hygiene, cleanup integrity |
| `fault-injection` | `collaboration-playwright` | Revocation, permission downgrade, lost acknowledgement, sidecar restart, outages, private anchors, and protocol rejection |
| `load-smoke` | `collaboration-load` | Run-ID-bound 20-socket/five-writer workload with automatic temporary-resource provisioning |
| `load-soak` | `collaboration-load` | Resource-guarded 30-minute, 25-identity mixed workload with scoped network phases and real product APIs |
| `load-capacity` | `collaboration-load` | Fixed high-capacity, latency, lease-rotation, restart, and reconstruction workload |
| `edge-conformance` | `web-edge-containers` | Black-box parity of the packaged Python and nginx web edges |

The six engines remain focused adapters. Profiles own selection, preflight,
run identity, reporting, and cleanup instead of each engine claiming an
independent completion status.

## npm integration contract

The root `package.json` exposes these stable integration points:

```text
npm run verify:list
npm run verify:tooling
npm run verify:ui-fixture
npm run verify:system-smoke
npm run verify:agent-desk
npm run verify:fault-injection -- --container-engine podman
npm run verify:load-smoke
npm run verify:load-soak
npm run verify:load-capacity -- --fixture /protected/load-fixture.json
npm run verify:edge -- --container-engine podman
# or: npm run verify:edge -- --container-engine docker
```

Their `scripts` entries are:

```json
{
  "verify:orchestrator": "node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types tests/verification/cli.ts",
  "verify:list": "npm run verify:orchestrator -- --list",
  "verify:tooling": "node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --test --experimental-strip-types tests/verification/orchestrator.test.ts tests/verification/container-engine.test.ts tests/verification/repository-hygiene.test.ts tests/verification/fixtures/accounts.test.mjs tests/verification/fixtures/fault-control-server.test.mjs e2e/browser-observer.test.ts e2e/config.test.ts e2e/control.test.ts e2e/layout.test.ts e2e/protocol-session.test.ts e2e/scenario-reporter.test.ts e2e/transport-fingerprint.test.ts && node --test tests/load/collaboration-load.test.mjs && tsc -p e2e/tsconfig.json --noEmit",
  "verify:ui-fixture": "npm run verify:orchestrator -- --profile ui-fixture",
  "verify:system-smoke": "npm run verify:orchestrator -- --profile system-smoke",
  "verify:agent-desk": "npm run verify:orchestrator -- --profile agent-desk",
  "verify:fault-injection": "npm run verify:orchestrator -- --profile fault-injection",
  "verify:load-smoke": "npm run verify:orchestrator -- --profile load-smoke",
  "verify:load-soak": "npm run verify:orchestrator -- --profile load-soak --container-engine podman",
  "verify:load-capacity": "npm run verify:orchestrator -- --profile load-capacity",
  "verify:edge": "npm run verify:orchestrator -- --profile edge-conformance"
}
```

For diagnostics, the same orchestrator can be invoked directly:

```bash
node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON \
  --experimental-strip-types \
  tests/verification/cli.ts \
  --list

node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON \
  --experimental-strip-types \
  tests/verification/cli.ts \
  --profile system-smoke \
  --fixture /protected/e2e-fixture.json
```

Supported orchestrator controls are `--profile`, `--fixture`,
`--container-engine`, `--run-id`, `--preflight-only`, `--list`, and `--json`.
`--fixture` is rejected for `load-smoke` and `load-soak`; their private session
fixtures are generated inside the run directory and removed before completion.
`--container-engine` accepts only an explicit `podman` or `docker`; there is no
automatic selection or fallback. It is mandatory for generated
`fault-injection`, `load-soak`, and `edge-conformance`; load-soak specifically
requires Podman for namespace-scoped shaping. Playwright grep, project, reporter, worker,
and file overrides are intentionally not accepted by this entrypoint.

## Run lifecycle

Every execution has a machine-readable `inqv-…` Run ID. The orchestrator:

1. validates the selected profile and scenario inventory;
2. runs every selected adapter preflight before starting any engine;
3. records child processes or exact run-labelled container resources
   immediately in a cleanup ledger;
4. executes adapters in the profile's declared order;
5. accepts product-resource registrations from a live child over a private IPC
   channel and acknowledges them only after the ledger has been flushed;
6. terminates registered processes on interruption or failure;
7. runs cleanup in reverse registration order;
8. writes the final report and cleanup state.

Reports live under:

```text
e2e/.results/verification/<run-id>/report.json
e2e/.results/verification/<run-id>/cleanup-ledger.json
```

Both files use mode `0600`. The report schema is version 3 and records the
Inqtrix version read from `src/inqtrix/__init__.py`, the profile, selected
engines, scenario records, source revision and dirty-state flag, explicit
status values, preflight results, adapter exit metadata, and cleanup state.
The CLI summary prints the same Inqtrix version. Scenario records use exactly:

- `passed`: the selected engine explicitly reported that scenario as passed;
- `failed`: that scenario ran and failed, or the engine failed before it could
  identify a more precise first failure;
- `blocked`: the owning engine failed preflight and never started;
- `not_run`: the profile selected the scenario, but execution never reached it;
- `not_applicable`: the scenario belongs to another profile.

An exit code of zero without complete explicit scenario results is converted to
an execution failure. Missing results are never inferred as passed, and
scenarios after an early failure remain `not_run`.

Reports never contain
environment dumps, command arguments, fixtures, browser storage state,
credentials, account emails, user/owner/recipient identifiers, cookies, CSRF
values, leases, or tokens. Cleanup-ledger labels describe only the resource
kind and current-run scope, never its product identifier. Sensitive keys,
bearer values, guest-link values, URL secrets, and sensitive environment values
are redacted before persistence. Child stdout and stderr also pass through the
same bounded line redactor before reaching the operator terminal.

Final statuses are unambiguous:

- `preflight_passed`: `--preflight-only` completed and no engine ran;
- `blocked`: a prerequisite failed before execution;
- `passed`: every selected adapter passed and cleanup completed;
- `failed`: an adapter failed;
- `cleanup_failed`: cleanup failed, regardless of the engine result;
- `interrupted`: the run received an interrupt or termination signal.

Exit codes are `0` for `passed`/`preflight_passed`, `2` for `blocked`, `3` for
`cleanup_failed`, `130` for `interrupted`, and `1` for execution failure.

## Web-edge container engine

The opt-in edge profile builds `web-python` and `web-nginx` from the same
`deploy/docker/Dockerfile.web`, starts a synthetic private backend and one
isolated network, and exercises both adapters through random loopback ports.
It covers SPA/static cache behavior, SSE, duplicate cookies, hop-by-hop header
removal, chunked body limits, binary WebSockets, backend recovery,
non-root/read-only execution, and normalized guest/share-link security and log
redaction.

Python removes dynamically nominated request hop fields. Stock nginx has no
generic directive for removing an arbitrary field named by `Connection`, so it
fails such malformed requests closed with HTTP 400; known protocol options
continue normally and fixed hop fields are stripped. The shared security
contract is that a nominated or fixed hop field never reaches the next hop.
An adapter may independently emit `Connection` for its own downstream
HTTP/1.1 link, so conformance distinguishes that from observable upstream
fields such as `Keep-Alive` and `Trailer`.
The trusted application upstream must not emit an arbitrary response field
nominated by `Connection`: Python removes such fields defensively, while stock
nginx can only remove the fixed protocol fields without an additional header
filter module.

No environment or secret file is read. Container commands use argv arrays and
their bounded output remains in memory; raw container logs are neither printed
nor persisted. Containers, network, and final run image tags are registered
before creation and removed in reverse order. A final label query requires
zero residual resources. Missing engine selection or an unreachable selected
engine is `blocked` with exit code 2, never a passing skip.

## UI fixture engine

`apps/research-desk/playwright.frontend.config.ts` runs
`browser-tests/editorCollaborationLifecycle.spec.ts` against a local Vite
fixture. It uses npm to resolve Vite and writes Playwright artifacts beneath the
current Run ID. It requires no application account or external service. The
fixture suite uses no engine-specific browser API, so every scenario must pass
in desktop Chromium, Firefox, and Playwright WebKit.

## Collaboration Playwright engine

The external-stack scenarios live at
`e2e/scenarios/collaboration.system.spec.ts`. `playwright.config.ts` selects the
profile inventory and runs each selected scenario across all three configured
transports with:

- Chromium desktop and mobile;
- Firefox desktop;
- Playwright WebKit desktop.

The genuine IME scenario uses Chromium's CDP composition API and is declared
Chromium-only in the inventory. It is removed from Firefox/WebKit selection,
not skipped at runtime. Playwright WebKit is engine coverage and does not
replace the mandatory real-Safari smoke gate for a supported Safari release.

Without `--fixture`, `system-smoke` creates its own active-gateway fixture and
`fault-injection` does the same when an explicit container engine is selected.
The generated fault controller binds only to `127.0.0.1`, requires both an
ephemeral bearer and the exact Run ID, and can target only the run's declared
documents and the label-verified `web` and `collaboration` services of the
canonical Compose project. It exposes no product API. The sidecar installs its
deterministic fault-file and signal handling only when
`INQTRIX_COLLABORATION_VERIFICATION_FAULTS=1`; unset or `0` means no active
fault seam, and every other value is rejected at startup.

For explicit three-transport coverage, copy `e2e/fixture.example.json` outside
version control and pass its path with `--fixture` or
`INQTRIX_E2E_FIXTURE`. Storage-state files under `e2e/.auth/`
remain ignored and, on POSIX systems, must not be accessible by group or other
users (`chmod 600 …`).

All strict profiles require:

- distinct declared owner and collaborator UUIDs and browser storage-state
  files; the live session endpoint must confirm that both states resolve to
  those two different identities;
- disposable direct-edit, revocation, and suggestion documents;
- Vite, nginx, and Python-gateway URLs on distinct credential-free origins
  (`fixture.transports.python-gateway` or
  `INQTRIX_E2E_PYTHON_GATEWAY_BASE_URL`);
- observable transport identity at every endpoint.

`system-smoke` additionally requires the concurrent-edit and detached-transfer
documents.
`fault-injection` additionally requires the downgrade, reconciliation,
sidecar-outage, public-gateway-outage, protocol, and private-anchor documents
plus authenticated fixture controls. Generated fixtures create these resources,
seed each actor's private comment anchors, and generate the private AI proposal
through the visible UI with a deterministic response limited to that one
provider boundary.

Unavailable strict prerequisites are `blocked`; they are never converted into
skipped success. The scenario reporter also fails if a selected required
scenario is absent or skipped at runtime.

## Live editor system engine

`e2e/engines/editor-system-live.mjs` runs the account-level multiuser matrix. It
uses `INQTRIX_VERIFICATION_RUN_ID` for document IDs and artifact paths, opens
isolated browser contexts, and exercises sharing, permissions, suggestions,
comments, guest links when enabled, navigation, and responsive layout.

Required or optional environment variables:

```text
INQTRIX_E2E_ADMIN_EMAIL          required; no seed-account default
INQTRIX_E2E_TESTER_EMAIL         required; must identify a different account
INQTRIX_E2E_ADMIN_PASSWORD       required; no password default
INQTRIX_E2E_USER_PASSWORD        required; no fallback to the admin password
INQTRIX_E2E_BASE_URL             optional; defaults to http://127.0.0.1:8080
PLAYWRIGHT_EXECUTABLE_PATH       optional; Playwright resolves its browser otherwise
INQTRIX_E2E_REQUIRE_GUEST_LINKS  optional strict capability requirement
```

The engine closes pages and contexts, deletes run documents, disables temporary
users, logs out normal account sessions, and closes the browser in `finally`.
Account, document, share/link, and session lifecycle responsibilities live in
the shared modules under `tests/verification/fixtures/`; the engine file keeps
the scenario interactions.

Product cleanup is durable from the parent orchestrator's perspective:

- documents are registered before creation and are deleted by their owner;
- shares and guest links are registered immediately after creation and are
  cleaned by deleting their Run-ID-bound document;
- temporary-user emails are deterministically Run-ID-bound and users are
  disabled after the run, which also revokes their sessions and personal access
  tokens;
- each authenticated browser context first registers an exact-session cleanup
  target with the parent and then writes its transient `0600` Playwright
  storage-state file under `.cleanup-secrets` before continuing.

## Local collaboration load smoke

`load-smoke` uses the same product-resource IPC and cleanup ledger as the live
system engine. It needs `INQTRIX_E2E_ADMIN_EMAIL`,
`INQTRIX_E2E_ADMIN_PASSWORD`, and `INQTRIX_E2E_USER_PASSWORD`;
`INQTRIX_E2E_BASE_URL` defaults to `http://127.0.0.1:8080`. The owner creates
four deterministic temporary users, shares one Run-ID-bound collaboration
document with Edit access, and issues exactly five leases per temporary
identity. The 20 sessions are ordered across identities so the first five
writer sockets do not all represent one user.

Use HTTPS for the normal hardened cookie contract. A deliberately plain-HTTP
loopback/LAN test must set `INQTRIX_OIDC_INSECURE_DEV_COOKIES=true` in that
temporary test environment; otherwise the login response can succeed while a
standalone API test client correctly refuses to replay the `Secure` cookie.
Never carry this development escape hatch into production.

The generated version-2 fixture contains only protocol fields required by the
load engine. It is atomically written with mode `0600` below the run's private
`.cleanup-secrets` directory, never copied into the report, and deleted after
the child exits. Accounts, sessions, shares, and the document are registered
before or immediately after creation, so parent cleanup still applies after
failure or interruption. `load-capacity` remains operator-provisioned because
its 1,000 sessions, HTTPS control plane, restart control, and lease-reissue
contract are environment-specific.

`load-soak` uses the same lifecycle with 25 distinct identities. Five writers
produce paced durable Yjs updates, five Suggest users create 50 mixed comment
threads, ten readers navigate repeatedly, and five users execute staggered
Research, Knowledge, Agent, and Chat activity. Six five-minute phases apply
normal, delayed, bandwidth-limited, lossy, and normalized network conditions
only to the selected collaboration container. Resource headroom is checked
before every phase; a post-quiet sample proves bounded memory, sockets,
database connections, and restart counts. Provider activity is limited to five
small workflows and a 5 USD usage-ledger budget.

On normal completion the engine performs these actions in `finally` and marks
the records complete. If the child fails, is interrupted, or is killed with
`SIGKILL`, the still-running parent replays the persisted ledger in reverse
order. There is no broad `ed_inqv-*` or `inqv-*@example.invalid` stale sweep,
because that could delete another concurrent verification run.
The harness deliberately exposes no automatic prior-run cleanup mode. If the
parent orchestrator itself is killed and leaves resources behind, an operator
must select that exact Run ID and clean it through the product's owner/admin
APIs; guessing by a shared prefix is not accepted.

The previous embedded load process was removed; load behavior belongs
exclusively to the two load profiles.

## Fault-control contract

Fault controls are authenticated `POST` operations whose URLs, response bodies,
and bearer values are never written to reports. They provide distinct
operations for lost durable acknowledgement, collaboration-sidecar outage,
public FastAPI gateway outage, status, restore, and sidecar restart.

The scenarios require observable effects rather than trusting controller
claims: browser close codes and decoded protocol state, unchanged projections
during outage, independent health behavior, exact-once reconciliation, and
successful recovery. A control response must expose its operation state and
fault layer without returning secrets.
