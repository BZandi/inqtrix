# Running tests

## Scope

Commands and expectations for running the Inqtrix test suite. The full test strategy (cassette layout, sanitization, recording) lives in [Testing strategy](testing-strategy.md); this page covers the "how do I run it today" view.

## Layers

Testing in this repository has four layers. They serve different purposes and should not be confused.

| Layer | Command | Real provider calls | What it covers | Prerequisites |
|-------|---------|---------------------|----------------|---------------|
| Automated local test suite | `uv run pytest tests/ -v` | No | Unit tests, replay tests, streaming, config, session, provider normalisation. Currently 800+ tests. | Editable install with dev extras. |
| Replay-only | `uv run pytest tests/replay/ -v` | No | Provider replay with VCR cassettes and `botocore.Stubber`. Offline. | Editable install with dev extras. |
| Parity asset validation | `inqtrix-parity contract` | No | Canonical question set and local parity asset structure under `tests/integration/`. | Editable install. |
| Local parity run against the HTTP test endpoint | `inqtrix-parity run --endpoint http://127.0.0.1:5100` | Yes | Runs canonical questions against a running server via `/v1/test/run`. | Running server, valid provider config, `TESTING_MODE=true`. |
| Manual live smoke test | `uv run python examples/provider_stacks/...` | Yes | Real end-to-end against configured providers. | Valid provider config. |

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

# Skip slow integration smoke tests
uv run pytest tests/ -m "not integration" -v
```

## Recording a replay cassette

Replay tests are offline by default. Recording a new cassette requires real API keys and is opt-in via `INQTRIX_RECORD_MODE=once`:

```bash
INQTRIX_RECORD_MODE=once \
  ANTHROPIC_API_KEY=sk-ant-real-key \
  uv run pytest tests/replay/test_anthropic_replay.py::test_complete_success_replay -v
```

See [Testing strategy](testing-strategy.md) for the full record modes (`none`, `once`, `new_episodes`, `all`) and the sanitization rules.

## Recommended validation flow

1. Run `uv run pytest tests/ -v` for the fast offline regression check.
2. Run `inqtrix-parity contract` if you changed parity assets under `tests/integration/`.
3. Start the local server in testing mode for structured end-to-end checks:
   ```bash
   export TESTING_MODE=true
   uv run python -m inqtrix
   ```
4. In a second shell, run the HTTP-based parity run:
   ```bash
   inqtrix-parity run --endpoint http://127.0.0.1:5100
   ```
5. Run `uv run python main.py` or an example script when you want a direct manual live smoke test against your actual provider setup.
6. For Azure-specific provider stacks, run the isolated smoke tests first:
   ```bash
   uv run python examples/provider_stacks/azure_smoke_tests/test_llm.py
   uv run python examples/provider_stacks/azure_smoke_tests/test_bing_search.py
   uv run python examples/provider_stacks/azure_smoke_tests/test_foundry_web_search.py
   uv run python examples/provider_stacks/azure_smoke_tests/test_openai_web_search.py
   ```

Running an example script is not part of the automated suite; it performs real external calls when providers are configured.

## What the suite covers today

- Configuration loading from environment variables, `.env`, and YAML.
- YAML-to-runtime bridging and model / provider resolution.
- Provider response normalisation and adapter interfaces (via unit tests and replay cassettes).
- Streaming behaviour and progress propagation.
- Graph wiring, orchestration export shape, and result serialisation.
- Source tiering, claim consolidation, context pruning, text and URL utilities.
- Parity comparison and report generation logic.
- Web server lifespan, overrides, security layers, multi-stack routing, cancel-on-disconnect.

The suite does **not** guarantee that every documented provider combination has been exercised against the real external service. That gap is exactly why the repo is marked experimental in the root `README.md`.

## Known test-order pollution

Two caplog-based tests (`test_..._effort_config_warnings_emitted_for_haiku_role`) fail when the full suite runs because logger-configuration tests mutate global state. Running those two tests in isolation shows them green. See [Troubleshooting](../reference/troubleshooting.md) and Gotcha #1 in the internal notes.

## Related docs

- [Testing strategy](testing-strategy.md)
- [Parity tooling](parity-tooling.md)
- [Troubleshooting](../reference/troubleshooting.md)
- [Installation](../getting-started/installation.md)
