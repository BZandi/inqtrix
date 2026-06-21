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

### I want to pick the LLM / search provider via `.env` (Azure, Anthropic, Bedrock, Foundry)

Two independent axes — `INQTRIX_LLM_PROVIDER` and `INQTRIX_SEARCH_PROVIDER` — choose the stack without Python. Copy the matching block from [Provider recipes](../getting-started/provider-recipes.md); the per-provider credentials and the model knobs (`INQTRIX_SELECTABLE_CHAT_MODELS`, `INQTRIX_TEMPERATURE`, `INQTRIX_SEARCH_PRESET`, …) live in the Providers block of [Settings and env](settings-and-env.md).

### I want real accounts: email/password or my existing LDAP

Set `INQTRIX_AUTH_MODE=local` for native email/password (the first visit creates the owner) or `ldap` to bind a directory you already run; both need `INQTRIX_SESSION_SECRET` + `INQTRIX_PAT_PEPPER` and the postgres backend. Conceptual overview in [Auth modes](../deployment/auth-modes.md); end-to-end procedures (owner setup, adding users, access tokens) in [Create and manage users](../how-to/create-and-manage-users.md) and [Connect to an existing LDAP](../how-to/connect-to-existing-ldap.md).

## Related docs

- [Settings and env](settings-and-env.md) — full env reference with behaviour notes where applicable.
- [Agent config](agent-config.md) — `AgentConfig` fields and minimal Python example.
- [Docs maintenance](../development/docs-maintenance.md) — when to update which `docs/**` page alongside code.
