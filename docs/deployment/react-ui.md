# React UI

> Files: `apps/research-desk/`, `package.json`, `package-lock.json`,
> `src/inqtrix_web_gateway/`, `deploy/docker/Dockerfile.web`

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
- `features/agent/` owns the Agent Desk: the automatic cognitive-kernel
  front door (`mode=agent_kernel`), explicit Mission runs
  (`mode=workspace_agent`), the session rail, assignment timeline with the
  plan/clarification gates, and the six canvas views. `features/canvas/` is
  the agent-agnostic polymorphic
  canvas host (registry + follow hook) other workspaces can mount later.
  The entry is capability-gated on `features.workspace_agent` (visible in
  demo mode).

The rail workspaces are directly reachable shell surfaces and should render
without a Suspense-only loading state. Import Research, Chat, Editor, Settings,
and the Markdown report viewer directly in the app shell; reserve deferred
loading for future heavy, non-primary tools rather than the views exposed in the
left rail.

Desk shells have no workspace-wide entry fade. Structural regions use
`StructuralLoadBoundary` with explicit `pending`, `ready`, `refreshing`,
`empty`, and `error` phases. Cached/prefetched data and refreshes over usable
content render directly. A cold target retains the previous complete surface
as inert (or a quiet body when no previous target exists); only a wait that
outlives 800 ms shows the target silhouette. A painted silhouette stays for at
least 300 ms and exits with the shared 150 ms reveal after registered geometry
work and scroll preparation settle. Reduced-motion mode keeps the readiness and
minimum-display ordering but removes shimmer and the exit animation.

Hover and keyboard focus reuse each feature's selection loader to prefetch
threads, knowledge sessions, editor documents, and server collections. Mermaid
diagrams and directly permitted Markdown images block only while a target is
staged; syntax highlighting remains progressive because it preserves geometry.
Known empty states are immediate, and long-running Agent/Research work uses
local progressive activity instead of a region-wide fallback.

## Shared Markdown rendering

Chat, Knowledge Desk, reports, file previews, Agent Canvas, and direct Agent Desk
answers use the same Markdown renderer. Mermaid diagrams preserve their native
type scale up to the available reading width, shrink responsively when needed,
and provide expand, Mermaid-source copy, and high-resolution PNG actions.
Rendered tables provide exact Markdown-source copy plus PNG and UTF-8 CSV
actions. The controls share one responsive hover/focus/touch action rail, so
individual workspaces must not add parallel wrappers or export paths.

All exports are created locally in the browser. Table CSV is serialized from the
visible cells, while PNG generation is loaded only when requested and captures
the full rendered width, including horizontally scrollable table columns. No
Markdown source or rendered content is sent to the server for these actions.

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
  (`replace`, `before`, `after`, or `append`). The route budgets document-wide
  edits against at least a 128k-token editor context floor before applying the
  remaining hard `400_000` character payload guard, so modern local stacks do not
  reject ordinary reports because of a stale or missing provider context-window
  hint. The UI renders these as document changes that can be accepted or rejected
  individually or as a group.

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

### Live editor collaboration

When an editor document has `content_mode=collaboration`, the React lifecycle
uses one Y.Doc and the Hocuspocus provider instead of emitting full Markdown
body changes. Normal StarterKit undo/redo is disabled for that lifecycle; Yjs
owns collaborative history. A remote update never feeds back through
`setContent()` or the legacy body autosave. Owner metadata changes continue
through their separate metadata-revision endpoint.

The right panel becomes an Editor Inspector with **Assistant** and **Changes**
tabs. Edit/Suggest/View permission, review display, active tab, and active
overlay are separate states. Only one overlay domain is visible: Assistant
shows private AI/comment anchors against the final projection, while Changes
shows tracked-change marks in Simple, All, Final, or Original presentation.
Other users appear as named text carets; remote selection fills and mouse
pointers are deliberately hidden.

The UI reports connection state separately from durability. A local update is
saved only after a `durable_ack` for its hash; reconnecting, revoked,
incompatible, or failed sessions become read-only and never fall back to a
Markdown body `PUT`. Source is read-only for collaboration documents. Project
export writes owned collaboration documents as detached Markdown snapshots,
and import always creates a new Markdown-mode document ID. Accepted incoming
shares remain server resources and are not copied into local project folders or
archives. See [Collaborate on editor documents](../how-to/collaborate-on-editor-documents.md)
for the user workflow and [Editor collaboration](editor-collaboration.md) for
deployment.

## Package manager and lockfile

npm is the only supported JavaScript package manager. The committed root
`package-lock.json` is the single dependency source for every workspace:

```bash
npm ci
```

Do not introduce a second lock, manager-specific patches, or an install hook as
a parallel dependency policy. Fix a dependency boundary in application code or
upgrade to an official release.

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
# uv:
uv run python examples/webserver_stacks/multi_stack.py

# or a standard pip/plain-Python environment:
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/webserver_stacks/multi_stack.py
```

Install frontend dependencies and run Vite from the repository root:

```bash
npm ci
npm run ui:dev
```

The Vite dev server listens on `127.0.0.1:5173` and proxies `/health` and `/v1`
to `http://localhost:5100`. That keeps browser calls same-origin during local
development while preserving the production boundary between frontend and API.

## Build

```bash
npm run ui:typecheck
npm run ui:lint
npm run ui:test
npm run ui:build
```

`ui:test` runs the Vitest unit suite (`vitest run`); it is a separate check and
is not part of `ui:build`.

The Vite build output is `apps/research-desk/dist/`. The directory is ignored by
git because it is a generated artifact. The image build compiles it once and
copies the same result into the selected Python or nginx runtime target. Other
packaging paths must likewise build it in a deliberate release step.

## Local production preview

The dev server (`ui:dev`) always runs the development React build, which wraps the
app in `React.StrictMode`. StrictMode double-invokes effects, reducers, and render
functions in development; the production build does not. Some defects are therefore
invisible under `ui:dev` and surface only in the production bundle, so verify a
change against the production build before shipping it.

`ui:prod` builds the bundle and serves it through `vite preview` in one step:

```bash
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
  on one origin, use the standard packaged Python gateway under
  *Same-origin serving without Node*.

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
| `GET /v1/runs/{run_id}/result` | Fetch the final report payload (`answer`, `metrics`, `top_sources`, `references`, `top_claims`, `usage`, plus `execution` for Agent Desk runs). |
| `GET /v1/user/events` | One content-free, user-scoped invalidation SSE channel per tab. |
| `GET /v1/knowledge/collections` | Authoritative owned-plus-accepted-shared server collections with `access`. |
| `GET /v1/prompt-templates` / `GET /v1/skills` | Authoritative server template lists with integer revisions and `access`. |
| `GET /v1/shares/inbox` / `GET /v1/shares/mine` | Sharing lifecycle views; regular resource lists remain the content source. |

When `INQTRIX_SERVER_API_KEY` is enabled, event streaming must use a
fetch-based SSE reader because browser `EventSource` cannot attach an
`Authorization` header. The server endpoint accepts Bearer auth; the browser
API is the limiting piece.

The user invalidation stream uses the same fetch-based SSE parser and one
`AbortController` per tab. It carries only `ready`, `invalidate`, and `reset`
signals; it never carries resource content or permission patches. `Last-Event-ID`
resumes retained events, while every connect/reconnect, browser focus, and
online transition schedules a debounced authoritative refetch. A `ready.user_id`
that differs from the current session forces a full document reload before any
event is processed.

Server lists are replacement views, not additive caches. A successful empty
prompt, skill, or collection response clears that server slice. A run missing
from the refreshed visible list is removed, its stream is aborted, and the
selection is repaired. Share accept/revoke/leave refreshes inbox, mine, counts,
the open share dialog, and the corresponding resource list together. Revocation
closes an open resource; an edit-to-view downgrade keeps readable content but
disables save. Dirty prompt/skill text is never overwritten by a remote
revision: the UI surfaces a conflict and lets the user copy or discard the
local buffer.

The knowledge picker is populated from `/v1/knowledge/collections`; local
vector-index records only preserve browser setup/progress and may map to the
same server collection id. They never create a synthetic shared collection.
For `access.mode="shared"`, the UI exposes indexed text and retrieval according
to `view`/`edit`, but it does not offer download of the uploader's private
original binary. Before a shared editor ingests a document, the UI explains
that the extracted content becomes part of the owner's collection even if the
editor later leaves.

Composer controls must map to the backend request shape instead of remaining
decorative. The current app emits a local draft using `question`, `stack`, and
`agent_overrides` fields (`confidence_stop`, `report_profile`,
`first_round_queries`, and `skip_search` after client
serialization). When the live submit path is enabled, add the top-level `mode`
field for the native chat/research choice: send `mode="direct_llm"` for LLM
chat without web research, and send `mode="research"` or omit the field for the
normal graph. Keep `agent_overrides.skip_search` only for compatibility with
older clients rather than introducing parallel mock-only fields.

### Agent Desk composer and routing

The Agent Desk composer separates four concepts that must not collapse into
one chip row:

1. **Context and edit targets.** The plus control opens one flat picker for
   knowledge collections and editor targets. The chip row contains only
   selected collections, an edit target, and attached skills; web/knowledge
   availability and slash routes never appear as context chips.
2. **Source Dock.** Web and project knowledge are adjacent two-state buttons.
   `available` means the agent MAY choose that source, not that every message
   invokes it. Server absence is disabled and explained; a one-shot route uses
   a distinct forced state without changing the stored session preference.
   The pair is one `role="group"`; each button keeps a stable accessible name,
   reports selection through `aria-pressed`, and exposes the same explanation
   on keyboard focus as on hover.
3. **Execution Capsule.** Run Setup and the existing `ModelTierPicker` are two
   adjacent segments. Run Setup owns permission mode, Automatic/Mission,
   response form, and Normal/Deep; the model segment keeps the complete Auto,
   model-category, concrete-model, model-information, and reasoning-effort
   picker. Narrow layouts change only the trigger to an icon with an accessible
   selected-model label; they open the same full picker. The capsule trigger
   variant is additive, so every non-Agent `ModelTierPicker` use keeps its
   existing trigger and behaviour.
4. **Transparency and submit.** Quota, the independent run-overview trigger,
   and Send remain reachable at every width. The run overview is read-only and
   reports the effective route, source policy, permission gates, model/effort,
   depth, consent reason, and separately the tools actually used.

Before admission the overview shows the prospective composer/capability state.
Once a run exists, `snapshot.execution` is authoritative: its
`effective_mode`, effective `source_policy`, model/effort, `consent_reason`,
and `tool_use_counts` replace client inference. In particular, “Web available”
and “Web used: 1 search” are distinct rows. The overview trigger remains an
independent control; it is not moved into Run Setup on narrow layouts.

The footer is a query container. At 704px and above its execution controls may
show labels and the short focus/hover reveal; from 576px through 703px they use
compact icon triggers with tooltips; below 576px only essential controls stay
inline and secondary quota detail moves into the run overview. Controls must
wrap/reduce their trigger presentation rather than disappear behind clipping.
Touch devices keep icon triggers stable. Reduced-motion mode removes reveal
and transform motion while preserving every state change. Source and capsule
transitions use the shared `appMotion.composer` timing (160ms,
`cubic-bezier(0.22, 1, 0.36, 1)`); available sources never pulse in a loop.

Run Setup reads the server's permission and depth manifests. The ordinary
permission choices are Standard (`balanced`) and Auto (`autonomous`); Strict
is shown only when the server publishes advanced autonomy. Automatic maps to
the kernel, while Mission explicitly selects the phase machine. If the kernel
is not registered, the composer states that only Mission is available instead
of silently relabelling Mission as Automatic.

Agent-session `items_json` stores the small UI-owned object
`{"source_policy":{"web":"available","knowledge":"disabled"}}`; each source
property independently accepts `available` or `disabled`.
New sessions default both sources to `available`. The selected source policy
is sent on every Agent Desk run, while the selected model/tier/effort remains
in the existing Agent UI settings and is serialized in `agent_overrides`.
The effective depth is likewise always serialized, including an explicit
`normal` selection when the server capability default is `deep`; display and
submission therefore consume the same value.

The slash menu has separate Commands and Skills groups. Selecting `/web` or
`/wissen` sets a visible one-shot route; an exact trailing command is also
recognized during submit and removed from the user question. The request maps
as follows:

| Composer choice | Native-run request |
|---|---|
| Automatic | `mode="agent_kernel"` |
| Mission | `mode="workspace_agent"` |
| Web/project-knowledge availability | `source_policy.web` / `.knowledge` = `available` or `disabled` |
| `/web` | `execution_directive="quick_web"` |
| `/wissen` | `execution_directive="knowledge_only"` |
| Chat/Canvas/Auto | `response_form="chat"` / `"canvas"` / `"auto"` |
| Normal/Deep | `agent_overrides.depth="normal"` / `"deep"` |
| Model and reasoning | existing `agent_overrides.model_tier`, `.model`, and `.effort` fields |

An execution directive is one-message state. The UI clears it only after
`POST /v1/runs` is admitted and a run summary is returned; validation, quota,
or transport failure leaves it visible for retry. The persistent session
source policy is never mutated by that reset. `/web` is the deterministic
one-search path, while Automatic remains the ordinary front door that may
choose instant web itself for a simple current-facts question.

The Source Dock and command availability are feature-detected from
`capabilities.agent.source_controls` and
`capabilities.agent.execution_directives`. The demo manifest publishes the
same blocks so source states, slash routes, responsive model trigger, and run
overview can be reviewed without a live agent backend.

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

### Agent Desk execution control centre

The Agent Desk run canvas is the execution overview, not a raw tool log. It
keeps the root phase rail and groups plan work units into active,
attention-required, and completed sections. Each work unit reports its honest
execution type, current user-facing activity, last update, retry attempt only
after a retry, and available query/source/claim counts. Instant search uses an
indeterminate one-call state, knowledge work may show a known `n/N`, and a real
research child uses its graph phases; the UI never fabricates a percentage or
ETA.

Selecting a work unit changes the existing run canvas descriptor to
`{view: "run", runId, taskId}` while retaining the same `run:{runId}` tab. The
detail view provides a visible Back action and restores focus to the source
card; it does not create a second task tab. Errors, fallbacks, insufficient
evidence, partial result summaries, and SSE-to-polling degradation stay visible
in the overview and expand into sanitized technical details. Terminal run
hydration first reads the durable plan/task rows and then replays the existing
event page through the live reducer, so cancellation or reload does not erase
the execution story.

Plan rows label their actual semantics: `Instant search - 1 request`,
`Research agent - Compact/Deep - N guidance questions`, or `Project knowledge
- n/N queries`. Independent tasks expose their shared execution wave. Editable
titles and questions use the existing auto-growing textarea behaviour; the
full plan never truncates approved text, while the compact approval tray may
use a two-line disclosure. Motion reuses the established pulse/flow primitives
and becomes static under reduced motion.

Agent memo citations use the artifact/result reference contract directly.
Internal `K#` entries open the exact knowledge chunk; web `W#` entries show an
exact source excerpt when one exists, otherwise clearly label
`grounded_support` as provider-grounded context rather than a quote from the
page. The report body does not duplicate citation chips above the document.
New agent output is currency-safe at the backend boundary. A small Agent-only
display adapter repairs legacy saved artifacts while preserving URLs, code,
block math, and genuine inline formulas; the shared Markdown renderer remains
unchanged.

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

1. Run `npm ci`.
2. Run `npm run ui:build`.
3. Publish `apps/research-desk/dist/` through the chosen static hosting layer.
4. Configure `VITE_INQTRIX_API_BASE_URL` at build time when the API is not
   same-origin. The value must be a complete origin such as
   `https://inqtrix-api.example.com` or `http://127.0.0.1:5100`; a bare IP
   address is not a valid browser fetch base URL.

The source app should stay in `apps/research-desk/`; only generated build
artifacts should ever move into a release package.

## Same-origin serving without Node

The Python web gateway is the standard runtime for the built React bundle.
nginx remains an explicit alternative for operators that already standardize on
it. Both keep the browser on one origin and route `/api/*`, `/v1/*`, `/health`,
and `/collaboration` to FastAPI. They are mutually exclusive implementations
of the same logical `web` service; Compose never starts both.

| Situation | Selection |
|---|---|
| Default Compose, Kubernetes, bare-metal Python, or local production-build verification | Packaged Python gateway |
| Enterprise platform that explicitly requires nginx behind its own TLS edge | nginx override |
| Static CDN that cannot proxy | Split-origin build with `VITE_INQTRIX_API_BASE_URL`; configure backend CORS |

### Standard: packaged Python web gateway

Pod topology:

```text
+----------------------+            +-----------------------+
|  Frontend pod        |            |  Backend pod          |
|  python:3.12-slim    |   /api/*   |  python:3.12-slim     |
|  inqtrix_web_gateway |   /v1/*    |  uvicorn :5100        |
|  dist/ contents      | ---------> |  python -m inqtrix    |
|  HTTP + WebSocket    |  /health   |                       |
|  proxy :8080         |  /collab-  |                       |
|                      |  oration   |                       |
+--------^-------------+            +-----------------------+
         |
       Browser
       (single origin)
```

[`deploy/docker/Dockerfile.web`](../../deploy/docker/Dockerfile.web) produces
one shared Node build stage and two explicit final targets. The default
`web-python` target contains the built SPA, the packaged
`src/inqtrix_web_gateway`, and dependencies projected from the repository's
single `uv.lock` through the `web-gateway` group. Its runtime contains no Node,
npm, uv, nginx, or API/agent dependency graph. It covers cookie login, SSE
streaming, uploads, the instance probe, the collaboration WebSocket, SPA
fallback, MIME handling, cache policy, forwarded-header trust, and sanitized
guest-link logs.

The runtime accepts these primary values:

| Variable | Meaning | Typical Kubernetes value |
|---|---|---|
| `INQTRIX_BACKEND_URL` | Full API origin the proxy forwards to | `http://inqtrix-api:5100` |
| `INQTRIX_PUBLIC_BASE_URL` | Primary origin contract: exact browser-visible scheme and authority | `https://desk.example` |
| `INQTRIX_EXTERNAL_SCHEME` | Optional scheme-only override; when set with the public base URL it must match that URL | `https` behind a TLS ingress |
| `INQTRIX_PROXY_MAX_BODY_BYTES` | Optional explicit whole-request cap in bytes | normally derived |

The Helm chart deploys this image as the `web` pod and derives public-origin
values from its Ingress or OpenShift Route. Liveness uses `/`; readiness uses
the API-aware `/health`.

#### Container quickstart

Build and run the same image used by Compose and Helm:

```bash
docker build --target web-python \
  -f deploy/docker/Dockerfile.web -t inqtrix-web .
docker run --rm -p 8080:8080 --read-only --tmpfs /tmp \
  -e INQTRIX_BACKEND_URL=http://host.docker.internal:5100 \
  -e INQTRIX_PUBLIC_BASE_URL=http://localhost:8080 \
  inqtrix-web
```

For direct TLS in Compose, mount certificate material from outside the
repository through the supplied override; no second image or web service is
created:

```bash
INQTRIX_PUBLIC_BASE_URL=https://desk.example:8080 \
INQTRIX_WEB_TLS_CERTFILE=/absolute/path/tls.crt \
INQTRIX_WEB_TLS_KEYFILE=/absolute/path/tls.key \
podman compose -f deploy/compose/compose.stack.yaml \
  -f deploy/compose/compose.web-tls.yaml \
  --env-file deploy/.env.stack.secrets \
  --env-file deploy/.env.stack \
  up -d --build
```

The override applies that one required public origin to the web gateway, API,
and worker, so guest links, OIDC callbacks, source URLs, forwarded headers,
and WebSocket Origin checks cannot retain a different value from a paired
runtime file. It also disables the plain-HTTP account- and guest-cookie
escape hatches for the API; Direct TLS always restores `Secure` cookies and
the `__Host-` account-cookie contract.

The normal trusted-LAN stack remains HTTP-capable for signed-in users.
Account-less guest links remain disabled there; they require a real HTTPS/WSS
browser origin through direct TLS or a trusted edge terminator.

#### Direct host start

The same package runs directly on hosts where containers or nginx are not
permitted. It is also the shortest way to verify a production `dist/` locally.
The gateway uses `httpx` to stream `/v1/*`, `/api/*` and `/health` to
the configured backend origin while serving `dist/` as a SPA. It also uses a
WebSocket client to relay binary `/collaboration` frames to FastAPI. SSE
streaming works because the HTTP proxy uses `aiter_raw()` without buffering.

It is the same production runtime without the container boundary. Multi-worker
mode shares one listen socket, direct TLS supports Compose or bare metal, and
the request-body cap derives from the backend file limit.

The browser observes the same contract in either packaging:

- Unknown paths outside `/assets/` serve `index.html` (SPA fallback);
  missing files under `/assets/` stay hard 404s.
- `index.html` (and every other non-asset path) is served with
  `Cache-Control: no-cache` so a redeploy is picked up immediately;
  hashed `/assets/` bundles cache immutably for a year.
- Proxied requests replace client-supplied forwarded Host/Proto metadata with
  the configured `INQTRIX_PUBLIC_BASE_URL`, the optional matching scheme
  override, or the gateway connection as the documented fallback.
  `X-Forwarded-For` remains append-only and `X-Real-IP` records the immediate
  peer, so per-client login rate limiting keeps distinct buckets. The raw
  request target is forwarded byte-identically (encoded path segments and
  repeated query keys survive).
- `/collaboration` preserves the public Host, Origin, cookies, raw query, binary
  frames, and upstream close behavior. It never starts or connects directly to
  the Node service.
- An unreachable backend answers `502 Bad Gateway`, with a
  60-second HTTP connect timeout; an unavailable collaboration gateway closes
  the socket visibly rather than degrading to polling or autosave.

The gateway requires a pre-built `dist/`:

```bash
npm ci
npm run ui:build       # -> apps/research-desk/dist/
```

Both supported Python installation paths execute the same module:

```bash
# uv: minimal locked gateway projection
uv sync --only-group web-gateway
uv run --only-group web-gateway python -m inqtrix_web_gateway \
  --dist-dir apps/research-desk/dist \
  --backend-url http://127.0.0.1:5100 \
  --host 127.0.0.1 --port 8080

# Standard Python/pip
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m inqtrix_web_gateway \
  --dist-dir apps/research-desk/dist \
  --backend-url http://127.0.0.1:5100 \
  --host 127.0.0.1 --port 8080
```

CLI arguments override environment variables; environment variables override
documented defaults. Startup validates `dist/index.html`, the backend origin,
and the TLS certificate/key pair before accepting traffic.

```bash
# Point at a dist/ outside the repository (release-artifact layout):
INQTRIX_DIST_DIR=/opt/research-desk/dist \
INQTRIX_BACKEND_URL=http://inqtrix-backend.svc.cluster.local:5100 \
  python -m inqtrix_web_gateway
```

#### Gateway environment variables

| Variable | Default | Effect |
|---|---|---|
| `INQTRIX_WEB_ADAPTER` | `python` | Image/Compose integrity sentinel. The Python target accepts only `python`; the nginx target accepts only `nginx`. Operators select the adapter through the documented image target or Compose override, not by changing this value in isolation. |
| `RESEARCH_DESK_HOST` | `127.0.0.1` | uvicorn bind host. Use `0.0.0.0` inside containers. |
| `RESEARCH_DESK_PORT` | `8080` | uvicorn bind port. |
| `RESEARCH_DESK_WORKERS` | `1` | Worker processes. Above 1 uses uvicorn's multiprocess supervisor sharing one listen socket; the upstream pool and body caps apply per worker. |
| `WEB_CONCURRENCY` | unset | Not a second worker setting. A conflicting value is ignored with a warning; use only `RESEARCH_DESK_WORKERS`. |
| `RESEARCH_DESK_SSL_CERTFILE` / `RESEARCH_DESK_SSL_KEYFILE` | unset | Optional TLS termination; both must be set together. `RESEARCH_DESK_SSL_KEYFILE_PASSWORD` decrypts an encrypted key file. |
| `INQTRIX_BACKEND_URL` | `http://localhost:5100` | Origin the gateway proxies `/v1/*`, `/api/*`, `/health`, and `/collaboration` to. |
| `INQTRIX_PUBLIC_BASE_URL` | unset | Explicit browser origin when a trusted reverse proxy terminates TLS before the gateway; pins the forwarded scheme AND host. |
| `INQTRIX_EXTERNAL_SCHEME` | unset | Optional scheme-only override: pins `X-Forwarded-Proto` while the forwarded host follows the request. If `INQTRIX_PUBLIC_BASE_URL` is also set, both schemes must match or startup fails. |
| `INQTRIX_MAX_UPSTREAM_CONNECTIONS` | `512` | Per-worker ceiling for pooled backend connections; each browser tab holds one long-lived SSE stream. Sized against the API's own admission caps: chat and native runs each admit up to 100, and every open event stream holds a connection for the run's whole duration on top of that. Exhaustion is a `503` with a warning naming this variable, never a silent wait. Connections open on demand, so the ceiling costs nothing until the load arrives.  Helm: set via `web.maxUpstreamConnections`; the web pod has no `envFrom`, so `config:` entries never reach it. |
| `INQTRIX_PROXY_MAX_BODY_BYTES` | derived | Explicit request-body cap in bytes. Unset derives `INQTRIX_MAX_FILE_BYTES` + 10 MiB; mirror the backend variable in split-container setups or the gateway warns and uses the packaged default. |
| `INQTRIX_DIST_DIR` | `<repo>/apps/research-desk/dist` | Override when serving a `dist/` from a release artifact path. |
| `INQTRIX_COLLABORATION_MAX_FRAME_BYTES` / `INQTRIX_COLLABORATION_MAX_QUEUED_FRAMES` | `2097152` / `32` | Collaboration relay frame-size and queue-depth limits; keep in sync with the collaboration service. |

If the resolved `dist/` directory or `index.html` does not exist, startup fails
loudly and names the resolved path; run `npm run ui:build` first or point
`INQTRIX_DIST_DIR` at an existing release artifact.

### Alternative: nginx

The `web-nginx` target consumes the exact same `ui-build` output. Choose it
only by overlaying `compose.web-nginx.yaml`; the override replaces the
implementation of the existing `web` service:

```bash
podman compose \
  -f deploy/compose/compose.stack.yaml \
  -f deploy/compose/compose.web-nginx.yaml \
  --env-file deploy/.env.stack.secrets \
  --env-file deploy/.env.stack \
  up -d --build
```

`INQTRIX_WEB_NGINX_IMAGE` optionally changes the nginx image name/tag from
`inqtrix-web-nginx:local`; it does not select nginx by itself. The override file
is still required so the `web-nginx` build target and adapter sentinel change
together.

Or build the target directly:

```bash
docker build --target web-nginx \
  -f deploy/docker/Dockerfile.web -t inqtrix-web-nginx .
```

The nginx option is HTTP-only and is intended to sit behind an external TLS
terminator. It must not be combined with `compose.web-tls.yaml`, whose direct
TLS variables belong to the Python gateway. nginx delegates WebSocket
frame/queue limits to FastAPI and the collaboration sidecar; the Python gateway
can enforce them at the edge. The same black-box suite verifies SPA fallback,
caching, streaming, multiple `Set-Cookie` headers, hop-by-hop isolation,
forwarded-origin trust, upload limits, guest-route privacy, recovery, and log
redaction. One malformed-header outcome is intentionally adapter-specific:
Python can remove an arbitrary request field named by `Connection` and continue;
stock nginx has no generic removal directive and rejects that request with 400.
Both prevent the nominated field from reaching the backend. Response-side,
Python also removes arbitrary dynamically nominated fields; nginx removes the
fixed protocol set and requires the trusted application backend not to emit
custom `Connection`-nominated response fields.

### Backend notes (apply to both paths)

- Build the bundle with `VITE_INQTRIX_API_BASE_URL` left **unset**. A baked
  absolute origin makes the browser call that backend directly and bypass the
  proxy, which reintroduces the cross-origin requirement and defeats the
  single-origin setup. The backend origin belongs in
  `INQTRIX_BACKEND_URL`, set at runtime.
- `INQTRIX_SERVER_CORS_ORIGINS` can stay unset. The browser sees a
  single origin, so cross-origin preflight never happens.
- Terminate TLS at the frontend pod, ingress, or sidecar. Backend
  `INQTRIX_SERVER_TLS_*` remains optional and useful only for direct
  ingress.
- `INQTRIX_SERVER_API_KEY` continues to work transparently because
  both proxies forward the `Authorization` header verbatim. It does not grant a
  collaboration lease; live editor sessions require cookie auth.
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
variables, so a page reload asks for it again. The same runtime token field is
also available under Settings > Security while the app is open.

The unlock gate and Settings > Licensing section both show the repository,
documentation, and license links. When `/health` includes the server `legal`
block, the Settings workspace uses that source URL, license identifier,
copyright notice, attribution notice, and no-warranty notice; otherwise it
falls back to the bundled Inqtrix project metadata. The unlock gate also shows
a static no-warranty usage notice before authentication.

`/health` carries a second, independent `ai_disclosure` block beside `legal`,
holding the machine tokens `marker` and `producer` plus two English disclosure
sentences. The app uses the machine tokens (they are the same values it writes
into `data-ai-*` attributes and exported document properties) but renders its
own localized wording for anything a user reads, so a German session never
shows the English server strings. See
[AI transparency](../reference/ai-transparency.md) for the full set of markers
and for what is deliberately left unmarked.

The unlock gate is a UX guard, not the security boundary. The backend Bearer
dependency remains authoritative. Any non-local deployment that uses Bearer
tokens must run behind HTTPS/TLS, use an explicit CORS origin allow-list, and
should add a restrictive Content Security Policy at the hosting/reverse-proxy
layer to reduce XSS token-exfiltration risk.

Cookie-session modes (`local`, `oidc`, `ldap`) bootstrap a canonical
`user.id` UUID from `/api/auth/session`. Local/LDAP login and logout complete
with a full document reload so no request, autosave, or store from the previous
identity can continue in the next user's tree. A failed logout keeps the
current session rendered and shows the error; the client does not pretend that
logout succeeded. Session refresh uses abort and generation guards, so a late
response for user A cannot overwrite a newer user B session. OIDC callback
navigation already provides the same document boundary.

Each browser workspace also has a non-secret `workspace_id`. The React app
creates it on first load, stores it in `localStorage`, writes it into
`project.md` as `workspace_id`, and sends it with native run and chat requests
as `X-Inqtrix-Workspace-Id`. Loading a project restores that project's
workspace id for subsequent requests. The server then filters `/v1/runs` and
run-specific result, event, and cancel calls by that namespace, so a page reload
does not hydrate every run from the API server. This namespace is not an
authorization boundary. In cookie-session modes, ownership and accepted direct
shares are checked against the canonical user UUID independently of the header;
in anonymous/static-key modes only ownerless legacy resources are visible.

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

Completed research reports and Prompt Library entries attached in chat remain
regular chat attachments in the exported project. For the live request only,
the client prepends bounded context blocks to the current user message so the
model can answer against the selected material without adding a second
persisted context model. Prompt Library entries are ordered before research
reports, and the visible message remains the user's own text plus compact
attachment chips.

Prompt Library entries may begin as project-local templates, but in a
server-persistent authenticated workspace their `/v1/prompt-templates` record
is authoritative. Hydration replaces the complete server-backed slice,
including accepted shared templates; a vanished server id means delete or
revocation and is removed locally. Local-only entries are adopted through one
create path rather than maintained as a parallel server model. Updates require
the loaded integer `expected_revision`; a 409 refreshes the current server
version while preserving the user's dirty draft without advancing its base
revision. The user must then either keep that draft as a new owned copy or
discard it for the current server version; a normal retry can never overwrite
the remote winner.

Each entry has a required lowercase slug label (`a-z`, digits,
hyphens, max 48 characters), a title, Markdown prompt content, a category, a
Chat/Editor visibility setting, and an autocomplete setting. The categories are
Instructions, Functions, and Context Packs. Functions are the only entries used
as prompt-chaining steps; Instructions and Context Packs can be attached as
context but are not chained. Context Packs can also link existing Database
files or file groups through a search-based picker with a bounded result list.
The prompt text can place linked files with `{{context}}`; if that placeholder
is omitted, the rendered context blocks are appended at the end. Attaching a
Context Pack stores the rendered snapshot in chat history so older chats do not
drift after Database edits. Disabling autocomplete hides an entry from all
surfaces and disables its Chat/Editor visibility controls. The composer keeps
the backward-compatible `@rules:<label>` mention shortcut and supports
`@research:<label>` mentions with keyboard completion; `@rules:` autocomplete
filters entries per surface and groups visible results by category. The Prompt
Library list is sorted by category, then title. Selecting a mention converts it
into a chip. Remaining exact mentions in the draft are resolved defensively on
send. Unknown labels block the send with a composer warning instead of being
silently ignored.

Skills follow the same authoritative list and revision rule through
`/v1/skills`. Their server-enforced policy is not reconstructed from local
project data. Prompt and skill `access.mode="shared"` allows save only when
`permission="edit"`; delete and share management stay owner-only.

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
backend persistence. The memory run store keeps terminal runs only for
`RUN_COMPLETED_TTL_SECONDS` (default 300 seconds). The Postgres run store is the
durable path and retains terminal runs for
`RUN_DURABLE_RETENTION_SECONDS` (default 90 days). In both cases `/v1/runs` is
the authoritative visible view: an owner sees their records and an accepted
recipient sees shared records annotated with `access`. Project import sends
`source_run_id`; the server always allocates a fresh public `run_id`, preventing
a retained share from attaching to a later import that reuses a client id.

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

Project export persists Prompt Library entries separately under `rules/`, one
Markdown file per label. Rule files keep `kind: "inqtrix.chat_rule"`
frontmatter with `rule_id`, `label`, `title`, `created_at`, and `updated_at`;
new Prompt Library fields (`category`, `visibility`,
`include_in_autocomplete`, and `linked_context_refs`) are additive, and the file
body is the prompt text. `project.md` also carries `rule_order` so the Prompt
Library order remains stable. Project load accepts missing Prompt Library data
as an empty library, defaults older rule files to Instruction entries visible
in Chat and Editor autocomplete, and keeps existing chat attachment snapshots
readable.

`project.md` also persists project UI preferences under `preferences`: locale,
theme mode (`light`, `dark`, or `system`), theme preset, contrast mode, and
the user message bubble tone (`gray`, `mint`, `orange`, `sky`, `violet`, or
`ink`).
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
- [Editor collaboration](editor-collaboration.md)
- [Collaborate on editor documents](../how-to/collaborate-on-editor-documents.md)
