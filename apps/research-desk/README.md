# Inqtrix Research Desk

React + Vite + shadcn-ready frontend shell for the native Inqtrix run API.

This app is intentionally isolated from the Python package. It consumes the
HTTP server through `/health`, `/v1/stacks`, and `/v1/runs*`; it never imports
`src/inqtrix` and never reads provider credentials.

## Prerequisites

- Node.js >= 22.12 (see root `package.json` `engines`).
- One supported package manager:
  - pnpm >= 11.1.1 activated via Corepack (recommended reference path).
  - npm >= 10.9.0 when Corepack/pnpm is unavailable.

## Install

### Recommended: pnpm via Corepack

Corepack pins the package manager declared in the root `package.json`
(`"packageManager": "pnpm@11.1.1"`). No global pnpm install required.

```bash
# One-time activation per machine:
corepack enable

# From the repository root, install dependencies for the workspace:
corepack pnpm install --frozen-lockfile
```

The root `package.json` scripts keep this path on Corepack-backed pnpm when
they are launched through `pnpm run ...`.

### Alternative: globally installed pnpm

```bash
npm install -g pnpm@11.1.1
pnpm install --frozen-lockfile
```

### Alternative: npm

Use npm from the repository root when pnpm/Corepack is not available:

```bash
npm ci
npm run ui:dev
```

This path is supported through the committed root `package-lock.json`, but it
has different supply-chain properties:

- It does not honour `pnpm-workspace.yaml` security settings
  (`minimumReleaseAge`, `blockExoticSubdeps`, `trustPolicy`); newly
  published or exotic transitive dependencies are accepted as-is.
- It resolves from `package-lock.json` rather than the reference
  `pnpm-lock.yaml`.

Do not switch package managers inside an existing `node_modules` tree. Use a
fresh checkout or rebuild `node_modules` before moving between pnpm and npm.

## Development

Start the API server in one terminal:

```bash
uv run python examples/webserver_stacks/multi_stack.py
```

### Default dev server (Vite proxy to `http://localhost:5100`)

```bash
# From the repository root:
pnpm run ui:dev
# or:
npm run ui:dev
# -> http://127.0.0.1:5173
```

The Vite dev server proxies `/health` and `/v1` to `http://localhost:5100`,
so browser fetches stay same-origin during local development while
preserving the production boundary between frontend and API.

### Dev server pointing at a non-default backend

`VITE_INQTRIX_API_BASE_URL` switches both the Vite dev-proxy target
(see `apps/research-desk/vite.config.ts`) and the browser-side default
base URL (`apps/research-desk/src/api/inqtrixClient.ts`).

```bash
VITE_INQTRIX_API_BASE_URL=http://127.0.0.1:6100 pnpm run ui:dev
# or:
VITE_INQTRIX_API_BASE_URL=http://127.0.0.1:6100 npm run ui:dev
```

The value must be a full browser origin (scheme + host + optional port).
Bare IP addresses without scheme are not accepted by the browser fetch
layer. Leave the variable unset during local proxy development.

If the server has `INQTRIX_SERVER_API_KEY` enabled, enter the matching
Bearer token in the app's Settings view at runtime; do not put API keys
into `VITE_*` variables because they are exposed in the browser bundle.

### Available environment variables

| Variable | Stage | Default | Effect |
|---|---|---|---|
| `VITE_INQTRIX_API_BASE_URL` | dev + build | `""` (same-origin) | Backend origin baked into the bundle and used by the Vite dev-proxy. Leave unset for same-origin serving via nginx or `scripts/run_research_desk.py` (see [`docs/deployment/react-ui.md`](../../docs/deployment/react-ui.md)). |

No other `VITE_*` variables are read by the React app today.

On first load the app creates a non-secret browser workspace id, stores it in
`localStorage`, sends it as `X-Inqtrix-Workspace-Id` on run and chat requests,
and writes it to `project.md` on export/save. Loading a project restores that
id, so `/v1/runs` hydration only brings back runs from the same workspace
namespace instead of every in-memory run on a shared API server.

## Live run integration

The composer creates native `/v1/runs` jobs. Web search enabled sends
`mode="research"`; web search disabled sends `mode="direct_llm"`. The app then
streams `events_url`, patches the selected job card from every snapshot event,
and fetches `result_url` after `inqtrix.run.completed` to populate the report,
evidence tab, and chat attachment data structure.

The visible live protocol is intentionally based on user-facing
`inqtrix.progress.message` events plus terminal/error events. Technical
snapshot, node-start/node-finish, and output-delta events remain part of the
internal state patching path but are not rendered as primary timeline steps.
Progress severities drive the compact info/warning icons in the selected card
and the warning treatment in the right-hand agent panel.

Completed runs keep the same visible event records in the project state and in
the exported Markdown frontmatter under `events`, so the report panel can show
the archived agent protocol after the live run has finished.

Completed reports remain attached to the in-memory project state while the app
is open and can be saved or exported with the project. Backend rehydration after
a full reload only works while the server still retains the terminal run under
`RUN_COMPLETED_TTL_SECONDS` (default 300 seconds). Increase that server setting
for longer review windows, or add durable backend run storage before relying on
reloads after that TTL.

## Chat rules and mentions

The Chat mode can store project-scoped prompt rules. Rules are client-side
Markdown snippets with a required slug label such as `executive-brief`; they
are referenced in the composer as `@rules:executive-brief` and injected into
the next `/v1/chat/completions` request as bounded context. They are not a
backend concept and do not change the selected provider stack.

Completed research reports can be referenced the same way through
`@research:<label>`. The composer also exposes both groups through the plus
menu. Sending a message stores the visible user text plus attachment snapshots
in chat history, while the technical context block is rebuilt only for the API
request.

Project export writes completed research reports under `search-history/` and
chat threads under `chat-history/` with compact timestamp-plus-id filenames.
Long titles stay in frontmatter and in the readable `project.md` export index,
not in file names. Existing exports with older title-based file names still
load because import reads the canonical IDs from frontmatter.

The Chat history can group related conversations into project-scoped groups.
Groups are stored in `project.md` through `chat_groups`, `chat_group_order`,
and `chat_thread_group_memberships`; individual chat files remain unchanged
under `chat-history/`. Loading an older project without these fields treats all
threads as ungrouped.

Chat transcripts support lightweight message management inside the selected
conversation. The header selection mode marks individual messages for bulk
deletion, user messages can be edited inline from their hover actions, and
assistant messages expose a branch action that creates a new chat containing
the transcript up to that response. If deletion leaves a user message as the
last transcript item, its hover actions can start a new assistant response
without duplicating the user message.

Project export writes rules to `rules/<label>.md` with frontmatter identifying
`kind: "inqtrix.chat_rule"`. `project.md` stores `rule_order` so the library
order survives a save/load cycle. Older projects without rules still load with
an empty rule library.

Project export also stores UI preferences in `project.md`: locale, theme mode
(`light`, `dark`, or `system`), theme preset, and contrast mode. Loading a
project applies those preferences to the current browser session. It also
stores the non-secret `workspace_id` used for live API namespacing. Runtime-only
secrets such as Bearer tokens remain excluded.

High contrast mode strengthens surfaces and borders without disabling Shiki
syntax colors in chat and report Markdown code blocks.

## Build

### Same-origin build (recommended)

Leave `VITE_INQTRIX_API_BASE_URL` unset. The resulting bundle uses
relative URLs and works behind any nginx / launcher reverse-proxy
without rebuild.

```bash
pnpm run ui:typecheck
pnpm run ui:lint
pnpm run ui:build
# or run the same commands with npm:
npm run ui:typecheck
npm run ui:lint
npm run ui:build
# -> apps/research-desk/dist/
```

### Build for a fixed backend origin

```bash
VITE_INQTRIX_API_BASE_URL=https://inqtrix-api.example.com \
  pnpm run ui:build
# or:
VITE_INQTRIX_API_BASE_URL=https://inqtrix-api.example.com \
  npm run ui:build
```

Use this only when no reverse-proxy is available; otherwise prefer the
same-origin build because it deploys to every environment without
rebuild.

The production bundle is written to `apps/research-desk/dist/`. That
directory is build output and is intentionally not committed. See
[`docs/deployment/react-ui.md`](../../docs/deployment/react-ui.md)
for deployment options including the nginx pattern and the Python
launcher `scripts/run_research_desk.py`.
