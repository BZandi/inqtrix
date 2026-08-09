# Changelog

This changelog follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). It will be populated and maintained from version `0.2.0` onward.

## Unreleased

No released artefacts yet. The repository is marked experimental (see the disclaimer in the root `README.md`); the version (`0.2.0`, defined once as `__version__` in `src/inqtrix/__init__.py`) is a placeholder and will be tagged formally with the first release.

### Added

- **App-wide Markdown block actions.** Mermaid diagrams now preserve their native
  type scale up to the available reading width and offer a shared responsive modal
  viewer, source copy, and high-resolution PNG export. Markdown tables offer exact
  source copy plus full-width PNG and UTF-8 CSV export across chat, knowledge,
  reports, file previews, and Agent Desk.

- **Model cards + direct model selection.** A curated, provider-neutral model-card
  catalogue (`src/inqtrix/model_cards.py`) with cross-provider alias resolution
  (Anthropic / Bedrock region+version / Azure deployment → one card); an optional
  `selectable_models` list on every LLM provider; per-run `agent_overrides.model`
  / `effort` (scoped to direct-chat + editor assist, research keeps tier routing);
  `models_catalog` + `context_window_tokens` on `/health` and `/v1/stacks`. React
  UI: an adaptive model picker (category groups, `i` hover-card with
  KONTEXT/TEMPO/KOSTEN + capability chips, `No think`/`Think`/`Think hard`
  reasoning selector) and a live composer token meter (tokenx) in chat + editor.
  See [Model cards](../configuration/model-cards.md).

### Changed

- **Dependency and build hygiene.** Mermaid moves to 11.16.1, closing the five
  advisories that reach the browser: CSS injection into sibling elements,
  prototype pollution through the configuration and architecture-diagram paths,
  and infinite-loop denial of service in XY and radar charts. The renderer
  already ran with `securityLevel: 'strict'` and `htmlLabels: false`, which
  bounded the impact. Lockfile-only patches cover DOMPurify, PostCSS, nanoid and
  brace-expansion, so `npm audit` reports no vulnerabilities again. The container
  build context no longer carries local E2E results or README media, and the
  collaboration image installs only the two workspaces its bundle needs instead
  of the entire workspace graph.

- **Fast run cancellation.** Cancelling a running research run now takes
  effect within seconds instead of minutes: provider retry ladders check for
  cancellation before every attempt and during backoff sleeps, the search and
  claim-extraction fan-outs abandon queued calls (visible as a warning
  progress message plus the `cancel_abandoned_work` iteration-log marker),
  and answer composition stops between report sections. The residual worst
  case is the remainder of one in-flight provider HTTP attempt. Run summaries
  additively expose `cancel_requested: true` while a cancel is pending, and
  the web client's delete action now cancels an active run, waits (bounded)
  for the terminal transition, and then deletes — resolving the previous
  409 `run_active` dead end after cancelling. Reranker retries are now
  visible on the knowledge event surface (`inqtrix.knowledge.rerank.retry`)
  instead of server-log-only, and the runtime availability probes name the
  probe bound and exception type in their warning instead of an empty
  message. The Research Desk composer and start-screen suggestion buttons
  are visibly disabled while the auth session is still resolving instead of
  silently ignoring clicks.

- **External PostgreSQL as a Compose building block.** The new
  `deploy/compose/compose.external-db.yaml` override detaches the bundled
  database container so the stack runs against a managed/external
  PostgreSQL 15+ (pgvector-enabled images work; the extension stays unused).
  `deploy/.env.stack.example` documents the four-step recipe and the
  `restricted` vs `bundled_legacy` runtime-login policies inline, and the
  docs gained explicit external-Postgres, S3-compatible (incl. Nutanix
  Objects) and custom-chart guidance.

- **Consistent Agent Desk presentation.** Direct answers now render in full like
  chat responses, the compact Settings selector follows the shared small-button
  typography, and newly selected live research logs position the current step
  immediately without an animated full-log scroll.

- **No silent token truncation on the frontend** (Designprinzip 1). Document
  ingest and chat/editor attachment context no longer truncate per document; the
  token meter shows what fits and warns when the context exceeds the selected
  model's window. The backend reference-document clamp stays as the visible
  last-resort guard.

## How to update this file on release

1. Move items from `Unreleased` into a new dated version section.
2. Group under `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.
3. Link to the corresponding PRs and to any new or changed docs pages.
4. Keep entries short; prefer "why" over "what" when the description is not obvious from the headline.

## Related docs

- [Release process](../development/release-process.md)
- [README](../../README.md)
