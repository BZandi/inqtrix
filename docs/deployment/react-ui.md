# React UI

> Files: `apps/research-desk/`, `package.json`, `pnpm-lock.yaml`,
> `package-lock.json`, `pnpm-workspace.yaml`

## Scope

`apps/research-desk` is the future React frontend for Inqtrix. It is a
separate Vite application inside the repository, not part of the Python
package. The frontend is a pure HTTP consumer: provider construction,
credentials, sessions, stacks, queueing, cancellation, and graph execution
stay inside the FastAPI server.

The current slice is an app-shell plus mock workspace: React, Vite,
TypeScript, shadcn/ui components, Tailwind styling, theme and locale toggles,
root scripts, package-manager configuration, typed API client for the native run
API, research cards, composer controls, progress panels, chat/settings
workspaces, an editor workspace, and a report panel bound to the selected run.
The mock data uses the same view-model boundary that live `/v1/runs` data should
feed later; it is not a second domain model.

The main screen should stay as a small orchestrator:

- `features/researchDesk/ResearchDesk.tsx` owns view switching and reducer
  wiring only.
- `features/researchDesk/components/` owns reusable research-desk UI pieces.
- `features/researchRuns/` owns native run API types and run-specific helpers.
- `features/chat/`, `features/editor/`, `features/settings/`, and
  `features/report/` own their respective feature surfaces.

The rail workspaces are directly reachable shell surfaces and should render
without a Suspense-only loading state. Import Research, Chat, Editor, Settings,
and the Markdown report viewer directly in the app shell; reserve deferred
loading for future heavy, non-primary tools rather than the views exposed in the
left rail.

## Editor workspace

The Editor rail view adds local Markdown documents, folders, live rendered
editing, a Markdown source mode, local inline comments, import from completed
research reports, and a persistent assistant composer under the document.
The document tree mirrors the chat history ergonomics: folders and documents can
be reordered via drag handles, documents can be moved into and out of folders,
and folder/document titles can be renamed inline. Document titles are also
editable from the compact editor toolbar via double-click. The toolbar also offers a
one-click **Export to Word** action that converts the document Markdown to a styled
`.docx` in a LaTeX-report look (serif justified body, numbered headings, title block,
page numbers) entirely client-side via the `docx` library plus the unified/remark parser
-- no server round-trip; math degrades to its LaTeX source text and raw HTML is skipped.
The editor assistant composer and inline comment composer auto-grow up to six
visible rows before scrolling, matching the chat workspace composer behaviour.

The assistant has two LLM-backed server contracts:

- `/v1/editor/suggest` rewrites one selected/commented Markdown block. It is the
  endpoint behind direct comments and collected comment runs.
- `/v1/editor/instruct` handles free document-level instructions from the
  composer. The request sends the current Markdown document plus the instruction;
  the response returns an assistant message and a list of content-anchored edits
  (`replace`, `before`, `after`, or `append`). The UI renders these as document
  changes that can be accepted or rejected individually or as a group.

Both endpoints also accept an optional additive `attachments` array of reference
documents (`{label, content, page_count, size_bytes}`) -- user-uploaded source
files the model may cite from but must not treat as an instruction. The server
renders them into a delimiter-wrapped `<reference_documents>` block whose `[N]`
headers line up with the `[N]` markers the composer writes into the instruction
text (`src/inqtrix/server/reference_documents.py`), clamps them to the model
context budget with a visible truncation warning, and drops any document that
looks like secret material. Omitting `attachments` keeps the prompt
byte-identical to before.

Both endpoints preserve Markdown as the editing format. Links, citation labels,
URLs, and LaTeX should remain Markdown-compatible across suggestions. The
frontend re-anchors each accepted edit against the live editor state before
applying it; if the target text cannot be found, the suggestion becomes stale
instead of applying to an unrelated position.
Document diff mode compares the active document against the saved anchor as
Markdown chunks and renders those chunks through the shared Markdown renderer;
do not flatten the document to plain text, because that drops headings, links,
tables, and LaTeX in the comparison view.

Editor state is part of the project model so it survives export/import:

- `editorDocuments` and `editorDocumentOrder` store Markdown working files.
- `editorFolders` and `editorFolderOrder` store the local document tree.
- `editorComments` stores comment threads and anchors outside the Markdown body.
- `editorUi` stores open document ids, the active document, visible panels, selected
  comment, view mode, and the local assistant draft.

Project export writes editor files under `documents/*.md` with
`kind: inqtrix.editor_document` frontmatter. The Markdown body stays clean; local
comment threads are serialized as metadata so the same file remains usable as
normal Markdown outside Inqtrix.

The Database rail view is a shared file library. Files attached anywhere (the
chat or editor paperclip and drag-and-drop, or a direct upload here) are parsed
client-side -- PDF via `pdfjs-dist`, DOCX via `mammoth`, text/Markdown/CSV
natively, all behind one swappable `FileParser` -- and appear under one of three
sections (`Temporäre Dateien`, `Bibliothek`, `Projekt-Quellen`) with editable
labels, metadata badges, and groups. Over-long documents are truncated at ingest
with a visible warning, and the chat composer shows an aggregate ">50% of the
context window" warning rather than silently cutting content. The chat and editor
composers share one Tiptap-based `MentionComposer`: positional `@files`,
`@filegroups`, and `@research` mentions render as atomic, auto-numbered `[N]`
pills that renumber by reading order and stay in two-way sync with the chip
legend, while `@rules` stay global (no pill). On send, the instruction carries
`[N]` and each attached context block is labelled with the matching `[N]` (a file
group is one block carrying all its members, so its single `[N]` still lines up).
Every chip can be drag-reordered, in both the chat and editor composers: dragging
a positional `[N]` pill chip permutes which source each number points to (the
prose stays put, the `[N]` reassigns), while rule/template chips reorder their
global run -- the two scopes reorder independently. A sticky chat composer toggle switches
attached templates between concatenation (one call, all templates as global
context) and chaining: the message is piped through each template in chip order
as sequential `/v1/chat/completions` calls (sources injected into the first step
only), with a collapsible step trace shown above the final answer. File
library state lives in the project model (`fileAssets`/`fileAssetOrder`,
`fileGroups`/`fileGroupOrder`, `fileLibrarySections`/`fileLibrarySectionOrder`)
and is exported as `files/*.md` (`kind: inqtrix.file_asset`, extracted text in
the body) plus section/group metadata in the project manifest.

The editor engine is the free Tiptap core stack:

- `@tiptap/react`, `@tiptap/core`, and `@tiptap/pm` provide the React and
  ProseMirror integration.
- `@tiptap/starter-kit` and selected MIT-compatible Tiptap extensions provide
  headings, lists, links, underline, highlight, tables for imported Markdown,
  and text alignment. The toolbar intentionally does not create new tables or
  task lists in the MVP.
- `@tiptap/markdown` is the Markdown import/export bridge through
  `contentType: "markdown"` and `editor.getMarkdown()`.

Do not add Monaco, CodeMirror, Tiptap Pro, Tiptap Cloud, private Tiptap registry
packages, or third-party editor starter kits for this workspace. Tiptap examples
may inform interaction patterns, but Inqtrix owns the compact document toolbar,
file tree, comment UI, and assistant composer. Link handling must keep explicit
safe-protocol checks when `@tiptap/extension-link` is touched.

The live editor is intentionally WYSIWYG-first. Do not add an inline "show the
underlying Markdown tokens for the clicked rich text" interaction unless there is
a dedicated design and implementation pass; Tiptap/ProseMirror does not provide
that behavior natively. Use the Source mode for raw Markdown editing.

## Package managers and lockfiles

pnpm via Corepack is the reference path. The root `package.json` pins the
package manager with `packageManager`, and `pnpm-lock.yaml` is committed for
reproducible installs:

```bash
corepack enable
corepack pnpm install --frozen-lockfile
```

The workspace security settings live in `pnpm-workspace.yaml`:

- `minimumReleaseAge: 1440` delays newly published package versions by one day.
- `minimumReleaseAgeStrict: true` fails resolution when no mature version fits.
- `minimumReleaseAgeIgnoreMissingTime: false` fails registries that omit publish
  timestamps instead of silently bypassing the age gate.
- `blockExoticSubdeps: true` prevents transitive dependencies from pulling code
  from exotic sources such as arbitrary git URLs or tarballs.
- `trustPolicy: no-downgrade` fails if a package's trust evidence regresses
  compared to earlier releases.

The root `package.json` also keeps pnpm build-script execution narrow through
`pnpm.onlyBuiltDependencies`, currently allowing only `esbuild`, which Vite
needs during install.

npm is a supported alternative for environments without pnpm/Corepack. Use the
committed root `package-lock.json`:

```bash
npm ci
```

The npm path does not enforce the pnpm-only security settings above. Do not
switch package managers inside an existing `node_modules` tree; use a fresh
checkout or rebuild `node_modules` before moving between pnpm and npm.

## Local icon assets

The UI vendors the Lucide icons it actually uses under
`apps/research-desk/src/assets/icons/lucide/` and exports React components from
`apps/research-desk/src/components/icons/`. App code should import icons from
`@/components/icons`; do not add a new `lucide-react` dependency for one-off
icons. When adding a new icon, copy the matching Lucide SVG into the local
asset folder, add the component export, and keep the bundled Lucide license
notice intact.

## Local development

Start the API server first:

```bash
uv run python examples/webserver_stacks/multi_stack.py
```

Install frontend dependencies and run Vite from the repository root:

```bash
corepack pnpm install --frozen-lockfile
pnpm run ui:dev
# or:
npm ci
npm run ui:dev
```

The Vite dev server listens on `127.0.0.1:5173` and proxies `/health` and `/v1`
to `http://localhost:5100`. That keeps browser calls same-origin during local
development while preserving the production boundary between frontend and API.

## Build

```bash
pnpm run ui:typecheck
pnpm run ui:lint
pnpm run ui:build
# or:
npm run ui:typecheck
npm run ui:lint
npm run ui:build
```

The Vite build output is `apps/research-desk/dist/`. The directory is ignored by
git because it is a generated artifact. Deployment should build it in CI and
publish the resulting static files, or copy them into a release artifact in a
dedicated packaging step.

## Local production preview

The dev server (`ui:dev`) always runs the development React build, which wraps the
app in `React.StrictMode`. StrictMode double-invokes effects, reducers, and render
functions in development; the production build does not. Some defects are therefore
invisible under `ui:dev` and surface only in the production bundle, so verify a
change against the production build before shipping it.

`ui:prod` builds the bundle and serves it through `vite preview` in one step:

```bash
pnpm run ui:prod
# or:
npm run ui:prod
```

`vite preview` listens on `127.0.0.1:4173`.

| Command | Compiles | Hot reload | React mode | Proxies `/v1`, `/health` |
|---|---|---|---|---|
| `ui:dev` | on demand | yes (HMR) | development (StrictMode) | yes, to `localhost:5100` |
| `ui:build` | once, to `dist/` | no | production | build only |
| `ui:preview` | no (serves `dist/`) | no | production | no |
| `ui:prod` | `ui:build` then `ui:preview` | no | production | no |

Two caveats for the preview path:

- `ui:preview` serves the static `dist/` directory only; it does not rebuild. Re-run
  `ui:build` (or use `ui:prod`) after every source change. A stale `dist/` is the most
  common reason a fix appears to have no effect in the preview.
- `vite preview` does not proxy `/v1` or `/health` -- only the dev server does -- so
  the preview is UI-only. To exercise the production bundle against a running backend
  on one origin, use the Python launcher (Path B under *Same-origin serving without
  Node*).

## Runtime API boundary

The React app should use the native run API rather than treating research runs
as chat messages:

| Route | Purpose |
|---|---|
| `GET /health` | Probe server readiness and whether Bearer auth is required. |
| `GET /v1/stacks` | Discover available provider stacks in multi-stack mode. |
| `POST /v1/chat/completions` | Back the free chat workspace through the OpenAI-compatible direct-LLM endpoint. |
| `POST /v1/text/improvements` | Improve chat/editor drafts, editor comments, and prompt-rule templates for explicit user review before replacement. |
| `POST /v1/runs` | Create a queued research run. |
| `GET /v1/runs/{run_id}/events` | Stream structured run events. |
| `POST /v1/runs/{run_id}/cancel` | Cancel a queued run or request cancellation. |
| `GET /v1/runs/{run_id}/result` | Fetch the final report payload (`answer`, `metrics`, `top_sources`, `references`, `top_claims`, `usage`). |

When `INQTRIX_SERVER_API_KEY` is enabled, event streaming must use a
fetch-based SSE reader because browser `EventSource` cannot attach an
`Authorization` header. The server endpoint accepts Bearer auth; the browser
API is the limiting piece.

Composer controls must map to the backend request shape instead of remaining
decorative. The current app emits a local draft using `question`, `stack`, and
`agent_overrides` fields (`confidence_stop`, `report_profile`,
`first_round_queries`, and `skip_search` after client
serialization). When the live submit path is enabled, add the top-level `mode`
field for the native chat/research choice: send `mode="direct_llm"` for LLM
chat without web research, and send `mode="research"` or omit the field for the
normal graph. Keep `agent_overrides.skip_search` only for compatibility with
older clients rather than introducing parallel mock-only fields.

The Magic Stick text-improvement control is an explicit review flow, not an
auto-rewrite. The browser sends only the active field text plus its context
(`chat_input` or `prompt_template`) to `/v1/text/improvements`; the server uses
the configured LLM provider without creating a research run, saving a chat
session, or invoking search. The UI must show the returned candidate with
highlighted changes and require accept/reject before replacing the draft.
Editor assistant drafts and inline comment drafts reuse the `chat_input`
context; do not add a parallel endpoint for field-level polishing unless the
server-side rewrite contract itself changes.

The run-list filter should stay compact: use the shadcn `DropdownMenu` filter
button with a `ListFilter` icon, active filter label, and count instead of a
horizontal tab strip for `All`, `Running`, `Queued`, `Cancelled`, and
`Completed`.
Changing the filter must normalize the selected run to the visible list: keep
the current selection only if it remains visible, otherwise select the first
visible run or clear the selection when the filter is empty. The trigger uses a
local neutral focus/open style; do not change the global shadcn button focus
ring for this compact control.

Reports should render the backend `answer` markdown rather than a second run
summary card. The mock workspace includes a long GFM report fixture to exercise
headings, links, lists, tables, blockquotes, references, light/dark mode, and
mobile wrapping. Live integration should replace the fixture lookup with
`GET /v1/runs/{run_id}/result.answer` and keep the same viewer component.
The Evidence tab should use `GET /v1/runs/{run_id}/result.references` for the
exact report-reference list shown under `## Referenzen`; `top_sources` remains
a ranked source overview and is only a compatibility fallback for older saved
runs.
Only completed runs should show a report. Queued and running runs use the same
right-hand surface. Running runs show a compact live overview plus a full-width
agent protocol built from `/v1/runs/{run_id}/events`; user-facing
`inqtrix.progress.message` events become the visible steps, while technical
snapshot/node/output-delta events patch state but stay out of the primary
timeline. Completed prior steps use quiet markers, active steps use the live
marker, and warnings/errors use compact severity treatment so context-window or
evidence-contract notices stay readable without leaking into card metadata such
as duration. The expanded running card derives small phase visit markers from
the same event stream so repeated planning/search/evaluation rounds are visible
without adding a second run-state field. Collapsed running cards show only the
minimum live context needed for parallel monitoring: animated current phase,
current round, source count when available, and the latest displayable agent
message as a single truncated line. Completed runs keep that agent
protocol available in the report panel through an `Agent steps` tab beside
Preview, Evidence, and Export; the
archived view is static and readable, without live auto-follow or active-step
animation. The panel should derive one explicit mode from the selected run:
empty, queued, running, completed with report, or completed without report.
Transitions between the live protocol and the completed report should keep the
panel frame stable and cross-fade the inner content, rather than remounting the
whole right surface abruptly. The hide/focus actions stay fixed in the
top-right header row in all modes; completed-run tabs sit below that row. The
step list should be a height-bounded internal log that auto-follows the newest
active event, so the right-hand panel does not grow the whole page. Queued runs
show a dedicated waiting state until the backend worker starts and emits
events.

The desktop research/report split uses `react-resizable-panels`. Size values
must be explicit percentage strings (`"58%"`, `"42%"`) because the installed
version treats bare numbers as pixels. Keep resizable panel containers
`min-w-0`, and keep Markdown content inside full-width ScrollArea children so
long paragraphs wrap to the current viewer width while tables retain their own
horizontal overflow.

## Deployment options

The first production path should be a separately served static frontend:

1. Run `corepack enable && corepack pnpm install --frozen-lockfile`.
2. Run `pnpm run ui:build`.
3. Publish `apps/research-desk/dist/` through the chosen static hosting layer.
4. Configure `VITE_INQTRIX_API_BASE_URL` at build time when the API is not
   same-origin. The value must be a complete origin such as
   `https://inqtrix-api.example.com` or `http://127.0.0.1:5100`; a bare IP
   address is not a valid browser fetch base URL.

The source app should stay in `apps/research-desk/`; only generated build
artifacts should ever move into a release package.

## Same-origin serving without Node

The recommended production topology runs the React bundle and the
Inqtrix API as two separate pods. Each setup below makes the browser see
**one origin** — the frontend host — and routes `/v1/*` and `/health`
to the backend internally. With this in place the React app needs no
`VITE_INQTRIX_API_BASE_URL` at build time and no CORS configuration on
the backend, so the same `dist/` artifact deploys to every environment.

### When to use which

| Situation | Recommendation |
|---|---|
| Production with Kubernetes / separate frontend and backend pods | nginx reverse-proxy (path A) |
| Server without container runtime, only Python available | Python launcher (path B) |
| Local verification of the production build | Python launcher (path B) |
| Build in CI, deploy to cluster | nginx reverse-proxy (path A) |

### Path A: nginx reverse-proxy in the frontend pod

Pod topology:

```
+---------------------+         +-----------------------+
|  Frontend pod       |         |  Backend pod          |
|  nginx:stable       |         |  python:3.12-slim     |
|  /usr/share/        |  /v1/*  |  uvicorn :5100        |
|    nginx/html/      | ------> |  python -m inqtrix    |
|    = dist/ contents | /health |                       |
|  /etc/nginx/        | ------> |                       |
|    conf.d/          |         |                       |
|    default.conf     |         |                       |
+--------^------------+         +-----------------------+
         |
       Browser
       (single origin)
```

`nginx.conf` snippet — copy into the frontend pod as
`/etc/nginx/conf.d/default.conf`:

```nginx
# Two-pod deployment: this nginx pod serves the React bundle and
# proxies the API to the Inqtrix backend service. Browser sees a
# single origin, so VITE_INQTRIX_API_BASE_URL stays empty.

server {
    listen 8080;
    server_name _;

    root /usr/share/nginx/html;       # mount dist/ here
    index index.html;

    # SPA fallback: unknown paths return index.html so client-side
    # routing keeps working. API paths below win because they are
    # declared first.
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Long cache for hashed assets (Vite emits content-hashed names).
    location /assets/ {
        access_log off;
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }

    # API reverse-proxy. Replace "inqtrix-backend" with the actual
    # Kubernetes Service name in your namespace.
    location /v1/ {
        proxy_pass http://inqtrix-backend:5100;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Authorization $http_authorization;
        proxy_set_header X-Inqtrix-Workspace-Id $http_x_inqtrix_workspace_id;

        # SSE streaming for /v1/runs/{id}/events. Without these flags
        # the browser sees events only after the run completes because
        # nginx buffers the response body.
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection "";
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    location = /health {
        proxy_pass http://inqtrix-backend:5100;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }
}
```

Key behaviours encoded in the snippet:

| Aspect | Reason |
|---|---|
| `proxy_buffering off` + `Connection ""` + long `proxy_read_timeout` | Required for the SSE endpoint `/v1/runs/{id}/events`. Without these flags the browser receives all events at once after the run finishes. |
| `proxy_set_header Authorization $http_authorization` | Forwards the Bearer token used when `INQTRIX_SERVER_API_KEY` is enabled. |
| `proxy_set_header X-Inqtrix-Workspace-Id` | Forwards the per-browser workspace namespace header so `/v1/runs` hydration scopes to the current project. |
| Listen port `8080` | nginx runs non-root inside the container; a `Service` can fan a public port 80 in front of it. |

#### nginx quickstart (local Docker validation)

Build the React app once, then run the nginx container with the snippet
above:

```bash
corepack enable && corepack pnpm install --frozen-lockfile
pnpm run ui:build
# npm alternative:
npm ci
npm run ui:build

# Save the snippet above as deploy/nginx-inqtrix.conf, then:
docker run --rm -p 8080:8080 \
  -v "$PWD/apps/research-desk/dist:/usr/share/nginx/html:ro" \
  -v "$PWD/deploy/nginx-inqtrix.conf:/etc/nginx/conf.d/default.conf:ro" \
  --add-host inqtrix-backend:host-gateway \
  nginx:stable-alpine
```

`--add-host inqtrix-backend:host-gateway` resolves the
`proxy_pass http://inqtrix-backend:5100` line to the host machine
(where the local backend listens) on macOS and Linux Docker. In a real
cluster, the `inqtrix-backend` hostname resolves via Kubernetes service
DNS instead.

### Path B: Python launcher `scripts/run_research_desk.py`

The repository ships a Python launcher that serves the same role as
nginx for hosts where only Python is available (no container runtime,
no nginx). It is also the easiest way to verify the production build
locally without container tooling.

The launcher uses `httpx` to stream `/v1/*` and `/health` to the
configured backend origin while serving `dist/` as a SPA. SSE
streaming works because the proxy uses `aiter_raw()` without buffering.

#### Launcher quickstart

The launcher requires a pre-built `dist/`. Build it once before
starting the server:

```bash
corepack enable
corepack pnpm install --frozen-lockfile
pnpm run ui:build      # -> apps/research-desk/dist/
# npm alternative:
npm ci
npm run ui:build       # -> apps/research-desk/dist/
```

Then start the launcher (it lives in `scripts/`, runs via `uv`):

```bash
# Default: serve on 127.0.0.1:8080, proxy to http://localhost:5100
uv run python scripts/run_research_desk.py
```

```bash
# Custom backend origin (e.g. backend on a different host or port):
INQTRIX_BACKEND_URL=https://inqtrix-api.example.com \
  uv run python scripts/run_research_desk.py
```

```bash
# Bind to all interfaces on port 80 (e.g. inside a container):
RESEARCH_DESK_HOST=0.0.0.0 \
RESEARCH_DESK_PORT=80 \
  uv run python scripts/run_research_desk.py
```

```bash
# Point at a dist/ outside the repository (release-artifact layout):
INQTRIX_DIST_DIR=/opt/research-desk/dist \
INQTRIX_BACKEND_URL=http://inqtrix-backend.svc.cluster.local:5100 \
  uv run python scripts/run_research_desk.py
```

#### Launcher environment variables

| Variable | Default | Effect |
|---|---|---|
| `RESEARCH_DESK_HOST` | `127.0.0.1` | uvicorn bind host. Use `0.0.0.0` inside containers. |
| `RESEARCH_DESK_PORT` | `8080` | uvicorn bind port. |
| `INQTRIX_BACKEND_URL` | `http://localhost:5100` | Origin the launcher proxies `/v1/*` and `/health` to. |
| `INQTRIX_DIST_DIR` | `<repo>/apps/research-desk/dist` | Override when serving a `dist/` from a release artifact path. |

If the resolved `dist/` directory does not exist, the launcher fails
loudly at startup with a `RuntimeError` naming the resolved path; run
`pnpm run ui:build` or `npm run ui:build` first, or set `INQTRIX_DIST_DIR`
to an existing directory.

### Backend notes (apply to both paths)

- `INQTRIX_SERVER_CORS_ORIGINS` can stay unset. The browser sees a
  single origin, so cross-origin preflight never happens.
- Terminate TLS at the frontend pod, ingress, or sidecar. Backend
  `INQTRIX_SERVER_TLS_*` remains optional and useful only for direct
  ingress.
- `INQTRIX_SERVER_API_KEY` continues to work transparently because
  both proxies forward the `Authorization` header verbatim.
- `RUN_COMPLETED_TTL_SECONDS` and `RUN_EVENT_BUFFER_SIZE` stay backend
  settings — neither path changes their semantics.

## Runtime connection settings

`VITE_INQTRIX_API_BASE_URL` is a public frontend build variable. It may contain
only the API origin. It must never contain provider credentials or the
`INQTRIX_SERVER_API_KEY` value because Vite embeds `VITE_*` variables into the
browser bundle.

When the server reports `auth_required: true` from `GET /health`, the React app
shows a full-screen unlock gate before protected UI actions are usable. The
gate accepts a runtime Bearer token and validates it with `GET /v1/runs`; only
after that probe succeeds does the app store the token in React memory and pass
it to protected `/v1/*` requests and to the fetch-based SSE stream through the
`Authorization` header. The token is not stored in `localStorage`,
`sessionStorage`, project files, Markdown exports, logs, URLs, or `VITE_*`
variables, so a page reload asks for it again.

The unlock gate and Settings workspace both show the repository, documentation,
and license links. When `/health` includes the server `legal` block, the
Settings workspace uses that source URL, license identifier, copyright notice,
attribution notice, and no-warranty notice; otherwise it falls back to the
bundled Inqtrix project metadata. The unlock gate also shows a static
no-warranty usage notice before authentication.

The unlock gate is a UX guard, not the security boundary. The backend Bearer
dependency remains authoritative. Any non-local deployment that uses Bearer
tokens must run behind HTTPS/TLS, use an explicit CORS origin allow-list, and
should add a restrictive Content Security Policy at the hosting/reverse-proxy
layer to reduce XSS token-exfiltration risk.

Each browser workspace also has a non-secret `workspace_id`. The React app
creates it on first load, stores it in `localStorage`, writes it into
`project.md` as `workspace_id`, and sends it with native run and chat requests
as `X-Inqtrix-Workspace-Id`. Loading a project restores that project's
workspace id for subsequent requests. The server then filters `/v1/runs` and
run-specific result, event, and cancel calls by that namespace, so a page reload
does not hydrate every in-memory run from a shared API server. This is not an
authorization boundary: a caller with the Bearer token and another workspace id
can still access that namespace through direct HTTP calls until a real
per-user/session auth layer is added.

The live submit flow is:

1. Serialize the composer controls into `question`, `stack`, top-level `mode`,
   and whitelisted `agent_overrides`.
2. Call `POST /v1/runs` and insert the returned summary into the project
   `ResearchRunRecord` map.
3. Open `events_url` with the fetch-based SSE reader.
4. Patch the run card from every `data.snapshot` payload: phase, rounds,
   sources, queries, confidence, and quality counters. Do not copy
   `snapshot.last_message` into queue/duration metadata after the run starts.
5. On `inqtrix.run.completed`, fetch `result_url` and attach `answer`,
   `metrics`, `top_sources`, `references`, `top_claims`, and `usage` to the
   same run record.
6. On `inqtrix.run.failed` or `inqtrix.run.cancelled`, close the stream and
   surface the terminal status in the card and right-hand protocol panel.

The chat workspace is separate from the research-run lifecycle. It sends
OpenAI-compatible requests to `POST /v1/chat/completions` with
`mode="direct_llm"` and `include_progress=false`, and it omits `stack` so the
server's default stack handles the request. The composer model picker is driven
by `chat_model_options`: prefer the selected `/v1/stacks` entry on multi-stack
servers and use `/health` only for single-stack servers. Missing discovery is
shown as a compact diagnostic instead of a fake provider-default model name.
Assistant messages store the backend-returned `inqtrix.model_resolution` when
available, so the transcript records which model and effort generated each
answer.

`/v1/stacks` is optional for single-stack servers. If the endpoint returns
HTTP 404, the React app treats that as a stable server capability for the
current page session, stops retrying stack discovery, hides stack selection, and
uses `/health` provider/model fields to label the server default. Native run
creation must omit `stack` in this mode so the server default remains
authoritative.

Chat streaming is a transient UI preference and is not written to project
files. Streaming mode consumes SSE `chat.completion.chunk` frames and appends
`choices[0].delta.content` into the active assistant draft until `[DONE]` or a
terminal `finish_reason` arrives. If a stream contains
`inqtrix.model_resolution`, the active assistant draft is updated before the
answer tokens render. Non-streaming mode uses the same endpoint with
`stream=false`, writes the returned assistant message as one block, and stores
the response-level `inqtrix.model_resolution` on that message.
Stopping a chat response aborts the browser fetch only; it is intentionally not
mapped to `/v1/runs/{run_id}/cancel` because no native research run exists.
If a streaming request fails before any assistant content arrives, or if the
server answers the streaming path without `text/event-stream`, the UI retries
the same messages once with `stream=false`. That fallback is visible: the
Streaming toggle is switched off, the composer shows a compact warning, and the
assistant message receives the blocking response instead of a mock reply. If a
stream fails after partial content has arrived, the UI keeps the partial text
and appends a visible error note rather than replaying the request.

The chat transcript should auto-follow while the user is already near the
bottom, including during streaming assistant updates. If the user scrolls up to
read previous messages, the transcript must not pull them back down until they
send another message or switch threads. Chat Markdown supports GFM tables and
task lists plus TeX math rendered through KaTeX. Message hover actions expose a
copy button for the rendered message source, and thread titles are edited
inline by focusing the title, then committing on blur or Enter.

Completed research reports and chat rules attached in chat remain regular chat
attachments in the exported project. For the live request only, the client
prepends bounded context blocks to the current user message so the model can
answer against the selected material without adding a second persisted context
model. Rules are ordered before research reports, and the visible message
remains the user's own text plus compact attachment chips.

Chat rules are project-scoped prompt templates, not backend resources. Each
rule has a required lowercase slug label (`a-z`, digits, hyphens, max 48
characters), a title, and Markdown prompt content. The composer supports
`@rules:<label>` and `@research:<label>` mentions with keyboard completion;
selecting a mention converts it into a chip. Remaining exact mentions in the
draft are resolved defensively on send. Unknown labels block the send with a
composer warning instead of being silently ignored.

Run cancellation is explicit and non-optimistic. The run card exposes a cancel
action for queued and running native API runs. The button calls
`POST /v1/runs/{run_id}/cancel` and keeps the existing SSE stream open. While
the POST is in flight, the card may show a transient `Cancel submitted` state
that is not persisted with project files. Queued runs can return a summary with
`status="cancelled"` immediately. Running runs normally return a still-running
summary, then emit `inqtrix.run.cancel_requested`, and only move to
`cancelled` when `inqtrix.run.cancelled` arrives. Failed cancel POSTs should be
shown as a local card notice; they must not mark the run itself as `failed`.
Cancelled runs do not fetch `/result` and appear under the `Cancelled` filter
with a muted destructive treatment, distinct from technical failures.

This attachment is the React app's runtime project state, not a replacement for
backend persistence. The native server keeps terminal runs in memory only for
`RUN_COMPLETED_TTL_SECONDS` (default 300 seconds). After that TTL, a browser
reload cannot hydrate completed runs from `/v1/runs`; users must save/export the
project before reload, operators must raise the TTL for longer review windows,
or a later database-backed run store must own durable history.

Project export persists completed runs as Markdown files under
`search-history/`. Filenames are compact timestamp-plus-id stems such as
`20260515T075400Z_ro-0245.md` or `20260520T101501Z_run-3f9a2c1.md`; long
question titles stay in frontmatter and in the readable `project.md` export
index. The YAML frontmatter includes the canonical `events` array for the agent
protocol and the report `references` array for the Evidence tab; there is no
parallel `agent_steps` field. On project load, imported events are normalized
so older files without `kind` or `severity` still render in the archived Agent
steps tab, and older reports without `references` rebuild a compatibility list
from the markdown reference section plus `top_sources` tier data. Event records
may also carry an optional UI phase (`analysis`, `planning`, `search`,
`evaluation`, or `answer`); when absent, the Research Desk derives the phase
from known progress-message wording.

Project export persists chat threads under `chat-history/` with the same
compact timestamp-plus-id filename strategy. Chat titles remain in frontmatter
and in the `project.md` export index, so long conversation titles do not inflate
paths.

Chat history groups are client-side project organization only. `project.md`
persists them as `chat_groups`, `chat_group_order`, and
`chat_thread_group_memberships`, while each conversation still lives as a
normal `inqtrix.chat` Markdown file under `chat-history/`. Deleting a group
ungroups its threads rather than deleting conversations. Older projects that do
not contain group fields load with all threads ungrouped.

Chat mode model selection is also client-side project UI state. `project.md`
stores `ui.selectedChatModelTier` as `"high"`, `"mid"`, `"fast"`, or `null`;
old or malformed values load as `null` (server default). The composer reads
`chat_model_options` from `/v1/stacks` for the selected stack and falls back to
`/health` only on single-stack servers. Complete discovery renders the actual
model name plus thinking effort in the floating model chip; missing or
unresolved provider metadata renders `Backend discovery missing` or `Provider
metadata missing`. Sending a chat message still uses `/v1/chat/completions` with
`mode="direct_llm"` and only passes `agent_overrides.model_tier`, never a raw
model id.

Assistant chat messages may carry optional model-resolution attributes in their
`inqtrix:message` block. Imported older chats without those attributes remain
valid and show no per-message model chip. New messages render a muted inline
chip in the assistant header such as `gpt-5.4 · No think` or
`gpt-5.4 · Think med`; switching the picker mid-chat does not rewrite older
assistant messages.

Inside a selected chat, message-level management stays client-side until the
project is saved or exported. Selection mode can bulk-delete transcript
messages, user messages can be edited inline, and assistant responses can spawn
a new chat branch that copies the transcript up to the selected response. When
the last transcript item is a user message, its hover actions can generate the
missing assistant answer by appending only an assistant placeholder and sending
the existing transcript to `/v1/chat/completions`. These operations rewrite only
the affected chat thread in project state; they do not create new backend API
concepts.

Project export persists chat rules separately under `rules/`, one Markdown file
per label. Rule files use `kind: "inqtrix.chat_rule"` frontmatter with
`rule_id`, `label`, `title`, `created_at`, and `updated_at`; the file body is
the prompt text. `project.md` also carries `rule_order` so the rule library
order remains stable. Project load accepts missing rule data as an empty rule
library and keeps existing chat attachment snapshots readable.

`project.md` also persists project UI preferences under `preferences`: locale,
theme mode (`light`, `dark`, or `system`), theme preset, and contrast mode.
Export/save writes the current provider state into that object, and project load
applies it back to the React providers before rendering the loaded workspace.
Runtime-only connection data such as Bearer tokens remains excluded from project
files.
High contrast mode strengthens surfaces and borders without disabling Shiki
syntax colors in chat and report Markdown code blocks.

`project.md` also persists `workspace_id`, which is a routing namespace for
live API hydration rather than a secret.

The report panel's Export tab is intentionally separate from project export. It
downloads only `run.result.markdown` as a plain `.md` report file and must not
include the YAML frontmatter used by project persistence.

Composer mapping:

| Control | Request field | Behaviour |
|---|---|---|
| Question text | `question` | Trimmed user prompt for the native run. |
| Stack | `stack` | Provider stack name in multi-stack mode; ignored by single-stack servers. |
| Web search on | `mode="research"` | Runs the full classify/plan/search/evaluate/answer graph. |
| Web search off | `mode="direct_llm"` | Calls the active LLM provider directly without web search. |
| Report profile | `agent_overrides.report_profile` | Chooses compact or deep answer profile. |
| Confidence target | `agent_overrides.confidence_stop` | Sets the stop threshold for iterative research. |
| First queries | `agent_overrides.first_round_queries` | Controls broad first-round query fan-out. |
| Max/min rounds | `agent_overrides.max_rounds` / `min_rounds` | Bounds the research loop. |

## Related docs

- [Web server mode](webserver-mode.md)
- [Run events](../observability/run-events.md)
- [Streamlit UI](streamlit-ui.md)
