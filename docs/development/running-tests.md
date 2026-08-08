# Running tests

## Scope

Commands and expectations for running the Inqtrix test suite. The full test strategy (cassette layout, sanitization, recording) lives in [Testing strategy](testing-strategy.md); this page covers the "how do I run it today" view.

## Layers

Testing in this repository has four technical families—offline/unit and replay,
provider parity, browser/system verification, and transport/load
verification—with the concrete entry points below. They serve different
purposes and should not be confused.

| Layer | Command | Real provider calls | What it covers | Prerequisites |
|-------|---------|---------------------|----------------|---------------|
| Automated local test suite | `uv run pytest tests/ -v` or `python -m pytest tests/ -v` | No | Unit tests, replay tests, streaming, config, server routes, provider normalisation. Use collect-only for the current count. | Editable install with dev extras. |
| Infrastructure-bound proof (mandatory before a release) | `INQTRIX_TEST_REQUIRE_INTEGRATION=1 uv run pytest tests/storage tests/test_qdrant_store.py` | No | Persistence, RLS and tenant isolation, fencing, sequence assignment, migration cutover. These tests are marked `postgres`/`qdrant` and are skipped in the run above. | Disposable Postgres in `INQTRIX_TEST_DATABASE_URL` and Qdrant in `INQTRIX_TEST_QDRANT_URL`, see [Local infrastructure](local-infrastructure.md). |
| Replay-only | `uv run pytest tests/replay/ -v` or `python -m pytest tests/replay/ -v` | No | Provider replay with VCR cassettes and `botocore.Stubber`. Offline. | Editable install with dev extras. |
| Parity asset validation | `inqtrix-parity contract` | No | Canonical question set and local parity asset structure under `tests/integration/`. | Editable install. |
| Local parity run against the HTTP test endpoint | `inqtrix-parity run --endpoint http://127.0.0.1:5100` | Yes | Runs canonical questions against a running server via `/v1/test/run`. | Running server, valid provider config, `TESTING_MODE=true`. |
| Manual live smoke test | `uv run python examples/provider_stacks/...` or plain `python examples/provider_stacks/...` | Yes | Real end-to-end against configured providers. | Valid provider config and either supported Python installation. |
| UI fixture profile | `npm run verify:ui-fixture` | No | Deterministic browser interaction without a backend. | `npm ci` and Playwright browser. |
| System/fault profiles and capacity load | `npm run verify:<profile> -- --fixture <private-file>` | Product stack only | Real multiuser UI, resilience, or fixed-capacity WebSocket load according to the selected profile. | Production-like stack and a disposable, uncommitted fixture. |
| Local collaboration load smoke | `npm run verify:load-smoke` | Product stack only | Auto-provisioned 20-socket/five-writer WebSocket smoke with Run-ID-bound cleanup. | Production-like stack plus the documented admin and temporary-user password environment variables. |
| Web-edge conformance | `npm run verify:edge -- --container-engine podman` or `docker` | No | Real Python/nginx images against a synthetic private backend, including normalized guest routes, streaming, WebSockets, limits, recovery, and hardening. | Explicit reachable container engine; no secret files. |

`pytest` is the automated regression gate. `tests/integration/` holds canonical parity questions and baselines for the parity tooling; it is not a fully automated live harness.

## Daily commands

```bash
# Full offline suite
uv run pytest tests/ -v

# Single test file
uv run pytest tests/test_claims.py -v

# Single test by name
uv run pytest tests/test_claims.py -k "test_name" -v

# Replay tests only
uv run pytest tests/replay/ -v

# Current test count without running the suite
uv run pytest tests/ --collect-only -q

# Every command above also works in a pip-installed environment:
python -m pytest tests/ -v
```

## Recording a replay cassette

Replay tests are offline by default. Recording a new cassette requires real API keys and is opt-in via `INQTRIX_RECORD_MODE=once`:

```bash
INQTRIX_RECORD_MODE=once \
  ANTHROPIC_API_KEY=sk-ant-real-key \
  uv run pytest tests/replay/test_anthropic_replay.py::test_complete_success_replay -v

# standard Python/pip:
INQTRIX_RECORD_MODE=once \
  ANTHROPIC_API_KEY=sk-ant-real-key \
  python -m pytest tests/replay/test_anthropic_replay.py::test_complete_success_replay -v
```

See [Testing strategy](testing-strategy.md) for the full record modes (`none`, `once`, `new_episodes`, `all`) and the sanitization rules.

## Recommended validation flow

1. Run `uv run pytest tests/ -v`, or `python -m pytest tests/ -v` in the
   pip-installed environment, for the fast offline regression check.
2. Run the infrastructure-bound proof against disposable services. Without
   `INQTRIX_TEST_REQUIRE_INTEGRATION=1` a missing URL only skips these tests, so
   step 1 alone never proves persistence, isolation, or migration behaviour:
   ```bash
   INQTRIX_TEST_DATABASE_URL=postgresql+asyncpg://user:pass@127.0.0.1:5432/inqtrix_test \
     INQTRIX_TEST_QDRANT_URL=http://127.0.0.1:6333 \
     INQTRIX_TEST_REQUIRE_INTEGRATION=1 \
     uv run pytest tests/storage tests/test_qdrant_store.py
   ```
3. Run `inqtrix-parity contract` if you changed parity assets under `tests/integration/`.
4. Start the local server in testing mode for structured end-to-end checks:
   ```bash
   export TESTING_MODE=true
   # uv
   uv run python -m inqtrix
   # or standard Python/pip
   python -m inqtrix
   ```
5. In a second shell, run the HTTP-based parity run:
   ```bash
   inqtrix-parity run --endpoint http://127.0.0.1:5100
   ```
6. Run `uv run python main.py`, or plain `python main.py` after the pip
   installation, when you want a direct manual live smoke test against your
   actual provider setup.
7. For Azure-specific provider stacks, run the isolated smoke tests first:
   ```bash
   # uv
   uv run python examples/provider_stacks/azure_smoke_tests/test_llm.py
   uv run python examples/provider_stacks/azure_smoke_tests/test_foundry_web_search.py

   # standard Python/pip
   python examples/provider_stacks/azure_smoke_tests/test_llm.py
   python examples/provider_stacks/azure_smoke_tests/test_foundry_web_search.py
   ```

Running an example script is not part of the automated suite; it performs real external calls when providers are configured.

## Browser, system, fault, and load verification

One orchestrator provides seven honest profiles:

```bash
npm run verify:list
npm run verify:tooling
npm run verify:ui-fixture
npm run verify:system-smoke
npm run verify:fault-injection -- --container-engine podman
npm run verify:load-smoke
npm run verify:load-capacity -- --fixture /protected/load-fixture.json
npm run verify:edge -- --container-engine podman
# or select docker explicitly
npm run verify:edge -- --container-engine docker
```

- `ui-fixture` needs no backend.
- `system-smoke` creates distinct temporary users and disposable
  documents/shares against the already running active gateway when no fixture
  is supplied. An explicit fixture remains available for three-transport runs.
- `fault-injection` creates the same private, Run-ID-bound prerequisites and a
  bearer- and Run-ID-protected loopback controller when no fixture is supplied.
  The explicit container engine is used only to target the label-verified web
  gateway and collaboration sidecar of the running canonical Compose project.
  Its deterministic collaboration fault seam is disabled by default. Recreate
  only the collaboration service with
  `INQTRIX_COLLABORATION_VERIFICATION_FAULTS=1` before this profile, then
  recreate it without that variable after the run. The sidecar rejects any
  other value and installs no fault-file or signal handling in normal service.
- `load-smoke` is the bounded local 20-socket/five-writer check. It uses the
  configured admin account to create four Run-ID-bound temporary users, one
  disposable document, shares, and a private short-lived session fixture;
  `--fixture` is intentionally not accepted.
  Use HTTPS normally. For a temporary plain-HTTP loopback/LAN stack, set
  `INQTRIX_OIDC_INSECURE_DEV_COOKIES=true`; never use that escape hatch in
  production.
- `load-capacity` is the stricter fixed-capacity environment gate.
- `edge-conformance` builds both packaged web adapters in isolated,
  run-labelled containers and never auto-selects a container engine.

The orchestrator reports `blocked` when a mandatory prerequisite is absent,
never a passing skip. Every run uses a machine-readable Run ID and writes a
redacted report plus cleanup ledger under
`e2e/.results/verification/<run-id>/`. See
[`e2e/README.md`](../../e2e/README.md) for the exact fixture, scenario,
cleanup, and reporting contracts.

For `edge-conformance`, omitting `--container-engine` or selecting an
unreachable executable produces `blocked` with exit code 2. Image-build or
contract failures produce exit code 1. Containers, the isolated network, and
run-specific final image tags are removed after success, failure, or
interruption; a residual-resource check can change the result to
`cleanup_failed`.

## What the suite covers today

- Configuration loading from environment variables and `.env`.
- Env-to-runtime model / provider resolution.
- Provider response normalisation and adapter interfaces (via unit tests and replay cassettes).
- Streaming behaviour and progress propagation.
- Graph wiring, orchestration export shape, and result serialisation.
- Source tiering, claim consolidation, text and URL utilities.
- Parity comparison and report generation logic.
- Web server lifespan, overrides, security layers, multi-stack routing, cancel-on-disconnect.

The suite does **not** guarantee that every documented provider combination has been exercised against the real external service. That gap is exactly why the repo is marked experimental in the root `README.md`.

## Test count

Avoid pinning docs to a stale exact count. Use
`uv run pytest tests/ --collect-only -q`, or
`python -m pytest tests/ --collect-only -q` in the pip-installed environment.
Treat the command output as authoritative for the current checkout.

## Related docs

- [Testing strategy](testing-strategy.md)
- [Parity tooling](parity-tooling.md)
- [Troubleshooting](../reference/troubleshooting.md)
- [Installation](../getting-started/installation.md)
