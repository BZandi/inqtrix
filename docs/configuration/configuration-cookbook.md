# Configuration cookbook

## Scope

Short, task-oriented entry points for the most common configuration questions. Each scenario links to the canonical reference pages and to runnable examples where they already exist. This page does not duplicate full env var tables (those live in [Settings and env](settings-and-env.md)).

## Scenarios

### I only have LiteLLM + Perplexity via environment variables

1. Copy [`.env.example`](../../.env.example) to `.env` and fill keys (never commit `.env`).
2. Follow [First research run](../getting-started/first-research-run.md) Path A (library) or Path B (HTTP).
3. Cross-check variable names and defaults in [Settings and env](settings-and-env.md); note the **Effect** column when present.

### I need explicit providers in Python (no env auto-magic)

Use [Library mode](../deployment/library-mode.md) and [Agent config](agent-config.md). Prefer constructor-injected providers; `AgentConfig(llm=None, search=None)` still auto-creates from `Settings` on first `research()` unless you pass concrete instances.

### I want to tune stopping, confidence, or rounds

Start with [Stop criteria](../scoring-and-stopping/stop-criteria.md) and [Confidence](../scoring-and-stopping/confidence.md). Every behavioural knob you change in `AgentConfig` or env should be reflected in the same PR under `docs/configuration/` or `docs/scoring-and-stopping/` per [Docs maintenance](../development/docs-maintenance.md).

### I run the HTTP server and need overrides or security context

Read [Web server mode](../deployment/webserver-mode.md) and [Security hardening](../deployment/security-hardening.md). For log and iteration-log workflows, add [Debugging runs](../observability/debugging-runs.md) and [Forensic cookbook](../observability/forensic-cookbook.md).

## Related docs

- [Settings and env](settings-and-env.md) — full env reference with behaviour notes where applicable.
- [Agent config](agent-config.md) — `AgentConfig` fields and minimal Python example.
- [Docs maintenance](../development/docs-maintenance.md) — when to update which `docs/**` page alongside code.
