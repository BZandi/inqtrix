# Docs maintenance

## Scope

How to keep `docs/**` honest over time. The repository uses plain GitHub-rendered Markdown today (no MkDocs, no Sphinx); the conventions below keep structure and cross-links consistent and make a future migration to MkDocs-Material a mechanical move.

## Conventions per page

Every page under `docs/**`:

1. Starts with an H1 title matching the filename (title-cased, spaces allowed).
2. Has a `## Scope` section as the first sub-heading explaining what the page does and does not cover.
3. Uses relative links for internal cross-refs (`[Agent config](../configuration/agent-config.md)`), never absolute URLs to GitHub.
4. Ends with a `## Related docs` block listing 2–4 peer pages with 0–1 sentence teasers.
5. Mermaid diagrams use only the subset that renders on GitHub: `flowchart`, `stateDiagram`, `sequenceDiagram`. Avoid `timeline`, `mindmap`, `C4Context` until confirmed.
6. Code examples use triple backticks with an explicit language tag (`python`, `bash`, `yaml`, `json`, `dotenv`).
7. English-only prose. German strings only inside quoted demo questions or prompt snippets, matching the repository convention.

## Adding a new page

1. Pick the closest existing subdirectory (`architecture/`, `providers/`, `configuration/`, `deployment/`, `observability/`, `scoring-and-stopping/`, `development/`, `getting-started/`, `reference/`).
2. Write the page following the conventions above.
3. Add the new page to the `Related docs` blocks of the pages that should link to it.
4. Update the README's docs-navigation table when the page is a top-level addition.
5. Run a local link check (see below).

## Link check

Use [`lychee`](https://github.com/lycheeverse/lychee) for a fast offline link check. There is no committed CI workflow for link-checking today; the repository's CI configuration is maintainer-owned, and this snippet is the recommended starting point when the project adopts one:

```bash
# Local check
lychee --no-progress --include-fragments \
    README.md 'docs/**/*.md' 'examples/**/README.md'
```

```yaml
# Recommended GitHub Actions snippet (not yet committed)
name: docs
on:
  pull_request:
    paths:
      - 'README.md'
      - 'docs/**'
      - 'examples/**/README.md'
jobs:
  link-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: lycheeverse/lychee-action@v2
        with:
          args: --no-progress --include-fragments README.md docs/**/*.md examples/**/README.md
          fail: true
```

Alternative tools: `markdown-link-check` (Node, widely used, deprecated community action has forks), `linkinator` (Node, dual CLI/module). `lychee` is recommended for speed on CI.

## Mermaid syntax constraints

Stick to the following subset — all four were verified to render on GitHub at the time of writing:

- Node IDs: camelCase or snake_case. No spaces (`UserService` OK, `User Service` not).
- Edge labels with special characters: wrap in quotes (`A -->|"O(1) lookup"| B`).
- Node labels with `(`, `)`, `:`: wrap in double quotes (`A["Process (main)"]`).
- Avoid reserved keywords as node IDs (`end`, `graph`, `subgraph`, `flowchart`).
- Subgraph headers always specify an ID plus a bracketed label: `subgraph gb [Group B]`.
- Do not use explicit colours or `classDef fill:#...` — they render badly in dark mode.
- Do not use `click` events; GitHub strips them.

## Keeping docs in sync with code

Whenever a code change alters public behaviour, update the matching `docs/**` page in the same PR:

| Code area | Matching docs page(s) |
|-----------|-----------------------|
| `src/inqtrix/providers/<x>.py` | `docs/providers/<x>.md` |
| `src/inqtrix/strategies/<x>.py` | `docs/architecture/strategies.md` and/or the dedicated page under `docs/scoring-and-stopping/` |
| `src/inqtrix/settings.py` / `config.py` | `docs/configuration/*.md` |
| `src/inqtrix/server/<x>.py` | `docs/deployment/webserver-mode.md` and/or `docs/deployment/security-hardening.md` |
| `src/inqtrix/agent.py` public API | `docs/architecture/public-api.md`, `docs/architecture/overview.md` |
| `src/inqtrix/nodes.py` | `docs/architecture/nodes.md` |

The internal notes (`.cursor/memory/audit-history.md`, `.cursor/memory/architecture-decisions.md`) are not a substitute for the public docs. ADRs describe the *decision*; `docs/**` describes the *behaviour* a user relies on.

## Related docs

- [Contributing](contributing.md)
- [Coding standards](coding-standards.md)
- [Testing strategy](testing-strategy.md)
