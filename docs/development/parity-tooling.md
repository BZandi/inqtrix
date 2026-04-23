# Parity tooling

> Files: `src/inqtrix/parity/*`, `scripts/parity_runner.py`

## Scope

The `inqtrix-parity` CLI compares a live agent run against a pre-recorded baseline. It runs the canonical question set against a local HTTP server, stores structured artefacts, and produces deterministic report files (JSON + Markdown + CSV) plus an optional LLM-driven diagnostic pass.

## Canonical questions and baselines

- Canonical questions live in `tests/integration/questions.json`.
- Baseline artefacts live in `tests/integration/runs/` (created on first run) and under the external baseline directory you import from.
- The parity CLI does not invent baselines; you import them from another repository or re-record them intentionally.

## Commands

```bash
# Validate local parity assets
inqtrix-parity contract

# Import the reference baselines from the litellm repo into Inqtrix
inqtrix-parity import-baselines /path/to/litellm/research-agent/tests/baselines

# Run the canonical suite against a local server in TESTING_MODE
inqtrix-parity run --endpoint http://127.0.0.1:5100

# Compare a saved run against baseline artefacts
inqtrix-parity compare tests/integration/runs/run_YYYY-MM-DD_HH-MM-SS.json \
    --baseline-dir /path/to/baselines

# Optional: add an LLM-driven diagnostic pass
inqtrix-parity compare tests/integration/runs/run_YYYY-MM-DD_HH-MM-SS.json \
    --llm-analysis
```

For editable installs, the same commands are available via `python scripts/parity_runner.py`.

## What `compare` produces

Each `compare` run writes three deterministic artefacts into the report directory:

- `report_*.json` — full machine-readable compare result with per-question issues, structured metric checks, and exported top-claim snapshots.
- `report_*.md` — Markdown summary with overview and flagged-check tables, plus claim snapshot sections for quick audits.
- `report_*.csv` — flat `question × metric × status` export for spreadsheets, diffing, and cross-run analysis, including compact baseline/current claim snapshots.

`compare` always performs the deterministic baseline-vs-run check first. `--llm-analysis` adds a second, optional diagnostic layer that uses the configured analysis model to inspect iteration logs, query trajectories, stop logic, and answer quality.

## Configuring the analysis model

The analysis model is configured through the YAML Pydantic interface:

```yaml
parity:
    analysis_model: gpt4
    analysis_timeout: 180
```

Any YAML model entry name is accepted as `analysis_model`. The analysis timeout is a soft per-question cap; the parity CLI aborts the analysis for a single question if it exceeds the timeout but keeps processing the rest.

## Minimal local parity run

```bash
# Terminal 1 — server in testing mode
export TESTING_MODE=true
uv run python -m inqtrix

# Terminal 2 — canonical suite
inqtrix-parity run --endpoint http://127.0.0.1:5100
```

Because the parity run talks to the real configured providers, it is slower and more failure-prone than `pytest` — but it gives a much better end-to-end signal.

## When to re-import baselines

Re-import a baseline when:

- The upstream reference repository publishes a new baseline for the same canonical questions.
- A change in Inqtrix legitimately alters the expected baseline (update the baseline, note the change in the commit, expect reviewer scrutiny).

Never edit baselines in place to "make the test pass" — either the behaviour is an intentional change (document it) or the behaviour is a regression (fix the code).

## Related docs

- [Testing strategy](testing-strategy.md)
- [Running tests](running-tests.md)
- [Iteration log](../observability/iteration-log.md)
- [Result schema](../architecture/result-schema.md)
