# Changelog

This changelog follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). It will be populated and maintained from version `0.2.0` onward.

## Unreleased

No released artefacts yet. The repository is marked experimental (see the disclaimer in the root `README.md`); version numbers in `pyproject.toml` (`0.1.0`) are placeholders and will be tagged formally with the first release.

### Added

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
