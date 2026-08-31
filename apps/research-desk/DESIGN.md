# Inqtrix Research Desk — Design Language

This document is the **single source of truth** for the visual language of the web app
(`apps/research-desk/`). It exists so that humans and AI agents make the UI look "from one mould"
(*aus einem Guss*) instead of re-inventing sizes per component, which is what causes visual drift.

The concrete rules live as global CSS in
[`src/styles/globals.css`](src/styles/globals.css) (the `@theme` font tokens and the
`@layer components` role classes). This file explains **what each role is and where it is used**, in
human-readable terms.

---

## Design ethos — "Quiet Enterprise"

The feel this app is designed for: **compact, enterprise-serious, information-dense without noise,
with restrained motion that reads as high-quality — never playful.** The concrete roles below
(§1–§8) are the *what*; this section is the *why/feel*, so new UI matches our reference exemplars —
the `@`-mention menu, the model picker, and the reasoning ("thinking") toggle.

**Essence:** *Quiet, dense enterprise UI — neutral with one accent, separated by hairlines rather than
colour, clearly hierarchical, with short, soft, directed motion that shows state and change. Precise,
not decorative.*

### Principles

- **Density with calm.** Compact ≠ cramped. A 4px spacing grid, clear grouping, lots of quiet
  structure. Every element earns its place; no filler, no decorative numbers. Prefer one more line of
  content over one line of air.
- **Colour is function.** Neutral by default. The brand accent only for selection, the primary
  action, and identity; the semantic set (success / warning / destructive / file) only for real
  status. Same colour ⇒ same meaning. No gradients, no colourful tiles, no emojis. (§5)
- **Separation by line & space.** Hairline 1px borders, whitespace, and gentle surface shifts
  (`surface`/`card`) instead of heavy colour blocks. Shadows soft and sparse, never hard. (§6)
- **Hierarchy via size *and* weight.** A real size step (`display 24 > page title 16 > section/card 14 > body 14 >
  list 13 > label/meta 12 > 11 > 10`, §2) **plus** weight; secondary text is `muted-foreground`;
  numbers/IDs are mono + `tabular-nums`. Few sizes, each owned by a role.
- **Motion explains, it doesn't entertain.** Animation shows origin (entry), state (loading,
  active) and change (height, counts). Short, soft, directed. Nothing blinks, bounces or spins forever
  on real content — loops are reserved for "working" signals.
- **Consistency beats creativity.** One popover layout, one active-accent, one hover-card style,
  one empty-state schema everywhere. Recognisability = trust.
- **Accessible by default.** Visible focus ring, full keyboard paths, sufficient contrast,
  hit-targets ≥ icon-button size, `prefers-reduced-motion` respected, dark/light equal.

### Reuse before abstraction

Search for an existing semantic token, motion contract, icon role, text role,
or UI primitive before adding a local value or component. Shared abstractions
represent a shared responsibility, not merely a similar appearance.

A genuinely unique feature composition may remain local while using the
existing visual language wherever its roles apply. When the same semantic role
or interaction contract appears independently in more than one feature,
extend an existing primitive or extract one shared implementation instead of
copying it. Do not introduce a global token or primitive for speculative
future reuse.

The ownership flow is:

```text
Design tokens and motion contracts
        ↓
Shared UI primitives
        ↓
Feature-owned compositions
```

Central primitives expose deliberate variants. Feature code must not fork a
primitive solely to change a size, colour, easing, or state treatment that
belongs to the same semantic role. A new shared token or variant names its
purpose, documents its intended consumers here, and is verified against all
existing consumers. If two consumers require different contracts, keep them
separate instead of accumulating unrelated conditionals.

This rule does not ban local values categorically. Feature-specific spatial
flows, one-off diagrams, and domain-specific animation may stay local when
they do not establish a reusable role.

### Motion (as implemented)

Easing is `cubic-bezier(0.22, 1, 0.36, 1)` (soft settle); UI transitions ≤ ~0.3s. Source of truth:
`src/motion/transitions.ts` (`appMotion`) and the `inqtrix-*` keyframes in `globals.css`.

| Purpose | Effect | Value |
|---|---|---|
| Content / list-row entry | fade + rise (`y 6–8px→0`) | `appMotion.list` 0.18s / `.card` 0.22s / `.panel` 0.26s |
| Control hover / press | colour/shadow transition | ~0.12–0.16s |
| Tooltip / popover | fade + micro-scale (0.97→1) | ~0.13s |
| Running process (active step) | soft pulse — halo/ring/core | `inqtrix-active-node-*` ~2.75s, `inqtrix-segment-breathe`/`-running-dot` 2s — **no blink** |
| Reasoning "thinking" mark | gentle sweep | `inqtrix-thinking-sweep` 1.9s |
| Agent pulse track (working connector) | light packet drifts along the station connector | `inqtrix-agent-flow` 2.4s — loop only while the agent works; static `brand/22` fill is the reduced-motion base |
| Agent composer signature | source-specific orbit/page-scan + a routing packet across the execution capsule | `inqtrix-source-orbit`, `inqtrix-knowledge-scan`, `inqtrix-execution-route` 0.36–0.52s; one shot on hover/focus only, never ambient |
| Metric update | one-shot count/flash | `inqtrix-metric-flash` 0.6s |
| Expand / collapse | soft height | `grid-template-rows 0fr→1fr`, ~0.3s |
| Side panel collapse / expand | width/flex collapse + directed fade-slide (`x ±10px→0`) | `appMotion.panel` 0.26s; left panels originate left, right panels originate right |
| Page-in-page push (canvas list → detail) | incoming layer slides in full width (`x 100%→0`), covered list parallaxes to `x −30%` and stays mounted (`inert`, scroll/focus survive), leading-edge `--shadow-soft` | forward `appMotion.push` 0.3s, back `appMotion.pushExit` 0.25s; reduced motion = 120ms fade; %-based transforms only (resize-proof) |
| Appearance switch | token swap without colour transition | Theme, preset, contrast, and bubble-tone changes suppress transitions for one frame so the shell changes as one state |
| Structural region wait | retain usable truth; delayed silhouette only for a cold target | `StructuralLoadBoundary`: 0.8s delay, 0.3s minimum fallback, `appMotion.reveal` 0.15s only when the fallback painted |
| Skeleton / loading | shimmer | loop **only while loading**; static under reduced motion |

**Rules:** no endless animation on real content (loops only for "working" signals); everything is
`prefers-reduced-motion`-safe (decorative motion off; the visible end-state is the base); off in print/PDF.

A structural region distinguishes `pending`, `ready`, `refreshing`, `empty`,
and `error`. A cached or prefetched target publishes in its normal React commit
without fallback or mount motion. During a cold identity change the previous
complete surface remains visible and inert; without a previous surface the
shell leaves a quiet, stable body. Only work still pending after 0.8 seconds
receives the target-shaped skeleton. Once painted it remains stable for at
least 0.3 seconds, then performs the single 0.15-second veil exit after data,
geometry blockers, and scroll preparation are complete. Empty and error states
are terminal, while background refresh preserves usable content.

Pointer/focus intent warms likely navigation targets through the same
idempotent loader used by selection. Navigation never clears a usable snapshot.
Mermaid diagrams and images with unknown geometry may block only a staged
target; stable text highlighting remains progressive. A long-running Agent or
Research operation switches from initial hydration to local status rows and
activity cards rather than repeatedly skeletonising the transcript.

Inline button progress, pagination and incremental table rows retain their
local indicators because they do not replace a complete region. They still use
the shared colour, skeleton and reduced-motion roles; they must not introduce a
second mount-wide fade.

### Spatial continuity and disclosure

Motion preserves object identity when the same conceptual object changes
state. Use a shared-bound or container transition only when source and
destination represent that same object. Otherwise use a short fade-through or
directed transition. Keep persistent chrome stable, animate transform and
opacity where possible, and never delay interaction until an animation
completes.

Productive motion is the default. Reserve expressive motion for infrequent,
meaningful moments such as a completed run, a major mode change, or an
explicitly invoked AI operation. Morphing communicates continuity; it is not
decoration. Reduced motion replaces spatial movement, scaling, blur, and
repeated animation with an immediate state change or brief opacity transition.

Primary actions, critical status, errors, and required inputs remain visible.
Secondary actions may appear on hover, keyboard focus, selection, or explicit
disclosure.

### Interaction patterns (reuse these — they are the reference look)

- **Popover / autocomplete menu** (the `@`-mention menu): breadcrumb/scope header left + hit-counter
  right → dense grouped rows (icon · primary `t-list` / secondary `t-meta-sm`) with a 2px active accent
  bar → keyboard footer (`Kbd`). Select via `onMouseDown` + `preventDefault`.
- **Dropdown menu** (the model picker): grouped `text-sm` rows under section eyebrows, an info
  affordance, a check on the active item; a segmented reasoning toggle in the footer.
- **Segmented control** (modes / filters / reasoning level): 2–4 short options, active = `bg-background`
  + soft shadow on a `bg-surface` track (§4, `h-7`).
- **Agent source dock + execution capsule:** source availability is a two-button `aria-pressed`
  group in the composer footer; a one-message forced route adds the brand accent and one-shot marker.
  Run setup and the unchanged rich model picker are adjacent flat controls: no enclosing border,
  divided frame, or persistent background. Source labels reveal on hover/focus only when the composer
  container is at least 704px wide; Plus and the Agent model trigger rely on stable accessible names
  and do not open redundant hover tooltips. The controls,
  run overview and send action must never be clipped when the Canvas narrows the timeline. Its icon
  language stays within Lucide's stroke system but uses distinct semantics: globe/orbit for public
  web, book-search/page scan for project knowledge, workflow for run setup, brain-circuit for the
  kernel and brain-cog for model selection. The moving execution rail is a functional routing cue,
  not ambient decoration, and reduced motion removes every transform and tracer.
- **Hover-card** (rich tooltip): small card — title + one sentence + optional 2–3 compact figures/tags
  — for definitions, derivations, source previews.
- **Status / tier badge:** `*-subtle` surface + dot/icon + short label.
- **Empty state:** centred, neutral icon in a circle, one-sentence title + one-sentence hint.
- **List:** group header (dot + label + count) → dense rows → 2px active accent; hover-only actions
  (`opacity-0 group-hover:opacity-100`).
- **Explorer history row** (chat threads, knowledge sessions, editor documents, agent sessions): the
  ONE `ExplorerHistoryRow` primitive (`components/ui/explorer-list.tsx`) — a two-column grid whose
  truncating `t-list-regular` title never shifts: the right-aligned relative age (`t-hint` +
  `tabular-nums`) fades out on hover/focus-within while up to two absolutely positioned `size-6`
  icon actions (`right-7`/`right-1`, `.icon-sm` glyphs, tooltip + aria-label) fade in. Hidden
  actions are `pointer-events-none` so an invisible destructive button cannot catch a touch tap
  aimed at the timestamp. A leading-indicator slot between title and age carries the running
  spinner / gate dot; rename mode swaps the title for `ExplorerHistoryTitleInput` (commit on
  Enter/blur, cancel on Escape). Do not rebuild this grammar inline in a feature.

### Responsive shell

The app keeps the same mental model across Research Desk, Knowledge Desk, Chat, Agent Desk, Editor,
Prompt Library, Database, and Settings.

- **Topbar:** the global topbar is always one row at `var(--header-h)`. Secondary project actions,
  theme, language, and repository actions collapse progressively into the existing dropdown-menu
  primitive. The global overflow trigger is a borderless hamburger icon button, not a framed
  ellipsis. Do not let the topbar wrap or grow vertically.
- **Side panels:** under `lg`, left/right workspace panels become modal overlays via
  `ResponsiveSidePanel`. The app rail remains visible; the drawer covers the workspace area from
  below the topbar. Drawers must use `aria-modal`, `aria-labelledby`, focus trapping, Escape close,
  backdrop close, and `appMotion.panel`. Right-side advanced drawers use the full workspace width on
  phone-sized screens and 80% from `sm` up; left navigation drawers keep the compact explorer width.
- **Canvas reading surface:** every canvas view body (report, plan, evidence, run
  overview, task detail, diff, patch) centers in the ONE `canvasSurfaceClass`
  measure (`max-w-4xl`, `src/features/canvas/CanvasSurface.tsx`) so tab switches
  never jump between widths; change the measure there, never per view. The agent
  transcript column stays `max-w-5xl` (the chat-mode contract).
- **Desktop panels:** at `lg` and up, keep the resizable split-panel model. Persisted desktop panel
  visibility must not force a mobile drawer to open immediately when the viewport becomes narrow.
  Desktop panel widths keep their per-workspace persisted split sizes and limits; a collapsed panel
  must release its split width completely so no empty gutter remains.
- **Panel-header edge contract:** persistent left/right `PanelToggle` controls and responsive-drawer
  close controls sit exactly `px-3` (12px) from the workspace edge in every desk. Do not widen this
  to viewport-dependent `md:px-6`; the position must remain stable between Editor, Chat, Knowledge,
  Agent and Research. Workspace, document, explorer and assistant titles are text-only by default;
  do not prepend a decorative file, folder, chat or robot icon. Icons remain only when they convey
  a distinct state or action.
- **Master/detail workspaces:** Prompt Library and Database show the navigator/list first under `lg`;
  selecting an item or collection moves into the detail surface, which provides a compact Back
  button. Do not stack navigator and detail vertically on small viewports.
- **Settings navigation:** under `lg`, Settings uses a compact section dropdown instead of a
  horizontally scrolling tab row. At `lg` and up, the left Settings rail is the visible navigation.
- **Dense mobile headers:** on very narrow screens (`< sm`), prioritize panel toggles, current title,
  and primary mode controls. Secondary workspace actions move to a local vertical-ellipsis overflow
  menu, hide, or become reachable from the panel/body; they must never overlap the panel toggle hit
  target.
- **Agent composer:** use the named `agent-composer` inline-size container rather than viewport
  breakpoints. Below 704px, Run Setup and model triggers become icon-only while opening the same full
  popovers; below 576px, the quota percentage collapses to its gauge icon while the full quota menu
  remains available.
  Context, both source buttons, both execution segments, run overview and send/stop remain reachable.

### Do / Don't

**Do:** neutral base + one accent · hairlines + `*-subtle` for quiet surfaces · short directed motion ·
mono + `tabular-nums` for numbers/IDs · the consistent popover / hover-card / empty-state patterns ·
keyboard paths · dark & light equal · the `.t-*` roles + control primitives.

**Don't:** gradients as decoration · full-bleed strong colour blocks · emojis (outside brand) ·
bouncing/rotating/looping animation on content · font-size sprawl (use the roles) · filler/decorative
numbers · ad-hoc `text-[..px]` (the guard warns) · a `.t-*` role on a `<Button>` (overridden — §0.7).

---

## 0. Golden rules (read before changing any size/spacing/colour)

1. **Use a role, never an ad-hoc value.** Reach for a `.t-*` text role and an `.icon-*` icon role
   instead of `text-[13px]`, `leading-[1.45]`, `size-[15px]`, etc. The role classes bundle
   size + weight + line-height so a UI function has exactly one owner.
2. **No silent one-off tweaks.** Do not "just bump this to 15px here". If a screen looks wrong, the
   fix is to apply the correct existing role — not to introduce a new value.
3. **A new size/role is a deliberate design decision.** Only add a new role when a genuinely new text
   *function* appears that none of the existing roles fit. When you do, it must (a) fit the scale
   below and (b) be added to `globals.css` **and** documented in this file in the same change.
4. **This file is kept in sync, and a guardrail enforces it.** Any change to a role's value, or any
   new role, updates this file in the same commit — `DESIGN.md` is the single place to verify the whole
   size/role architecture still fits together. An **ESLint guard** (`eslint.config.js`) **warns** on any
   new ad-hoc `text-[..px]`/`leading-[..]` under `src/features/**`, so a size can only enter the app as
   a role defined here. Treat `DESIGN.md` as part of the public contract (see
   `docs/development/docs-maintenance.md`).
5. **The Markdown renderer has its own reading-system contract** — see §9.
6. **Compact by default.** The app favours dense, compact layouts; pick the tighter role when in doubt.
7. **Roles lose to utilities — never put a `.t-*` role on a `<Button>` (or any element that also
   carries a `text-*` utility).** The roles live in `@layer components`, which Tailwind ranks *below*
   utilities, so the shared `Button`'s built-in `text-sm` (or any `text-xs/sm/base`) wins and the role
   silently does nothing. For a **control** label use the control's own size (e.g. `text-xs` on a
   footer button — see §4); for **content**, put the role on the text element itself (a
   `<span>`/`<p>`/heading that has no competing `text-*`).

---

## 1. Fonts

| Token | Value | Where |
|---|---|---|
| `--font-sans` | `"Inter Variable", ui-sans-serif, system-ui, …` | all UI text (`body`) |
| `--font-mono` | `ui-monospace, SFMono-Regular, Menlo, …` | code, run/job IDs, URLs, tabular numbers |

Inter is **self-hosted** via `@fontsource-variable/inter` (imported in
[`src/main.tsx`](src/main.tsx)) so the UI renders identically on every machine. There is **one UI
typeface** (Inter). Functional differentiation is done through size/weight/line-height (the roles
below), **not** through additional font families. Mono is used only for the cases listed above.

---

## 2. Typography roles (`.t-*`)

Defined in `globals.css` `@layer components`. There is a real **size step** between body and
headings — `display 24 > page title 16 > section/card 14`, with body at 14 — so a heading never reads
as the same size as plain body (a same-size heading is weak hierarchy). Only a **whole page's** header
is 16 (`.t-title`); everything *inside* a page (panel/section/content/card headings) is 14 + bold. The
dense tier (13/12/11) covers lists,
labels and metadata; 10 is for eyebrows/hints. Calibrated to Carbon's productive type set
(heading-compact 14/16, body 14). One role = one size + weight + line-height.

| Role | px | Weight | Line-height | Use it for — concrete examples |
|---|---|---|---|---|
| `.t-display` | 24 | 600 | 2.25rem | Page hero / empty-state heading (e.g. Research Desk "Guten Tag.") |
| `.t-title` | 16 | 600 | snug (~1.375) | **Whole-page / view title only** — the single header for an entire page: "Prompt Library", "Datenbank", "Einstellungen". One per view; distinctly larger than everything else. |
| `.t-section` | 14 | 600 | snug | **Section / panel / content heading** — Report panel "Report", Chat "Gespräche" + the chat title, Editor "Dokumente" / "Editor-Assistent", "Alle Dokumente", "Neuer Prompt", report content sections ("Belege/Quellen"), Settings sections, modal titles. Bold, but body-size — a section heading is *not* a whole page. |
| `.t-card` | 14 | 600 | snug | **Card / item / row title** — research run cards, file/doc cards, Settings row titles. Same 14/600 as `.t-section`; the name conveys intent (a card/row vs. a section heading). |
| `.t-body` | 14 | 400 | normal (1.5) | Reading text + inputs (NOT Markdown) — prompt textarea, instructional hints, settings descriptions, search/input text. Line-height 1.5 matches the chat reading body. |
| `.t-list` | 13 | 600 | 1.25rem | **Dense list / menu primary label** — folder headers, nav rows, menu primary option, strongly labelled dense rows |
| `.t-list-regular` | 13 | 400 | 1.25rem | **Dense explorer item label** — normal chat/knowledge/editor history entries where selection/folder context already provides hierarchy |
| `.t-label` | 12 | 600 | — | Form / control labels, folder-group headers ("Neuer Ordner") |
| `.t-meta` | 12 | 400 | — | Subtitle / metadata / helper text / breadcrumb — sidebar subtitles, "Dokumente › …" breadcrumb |
| `.t-meta-sm` | 11 | 400 | — | Dense subtitle / file metadata — file size & path lines, dense secondary text |
| `.t-caption` | 10 | 600 | — | UPPERCASE eyebrows / section dividers ("OHNE ORDNER", mention-menu group headers) |
| `.t-hint` | 10 | 400 | — | Non-uppercase micro text — keyboard legends ("↑↓ Navigieren · Esc"), counts, timestamps |
| `.t-mono` | 11 | 500 | — | Monospace IDs, URLs, tabular numbers |

**Heading hierarchy:** `display 24 > page title 16 > section/card 14 > body 14`. Only a **whole page's**
header is `.t-title` (16) — one per view ("Prompt Library", "Datenbank", "Einstellungen"). Everything
*inside* a page (panel/section headers, content & card/row titles) is 14 + bold: `.t-section` for a
section/panel/content heading, `.t-card` for a card/item title. They share 14/600 (same look); the
split is semantic so the cheat-sheet stays unambiguous. **Never** put `.t-title` (16) on something that
is not a whole-page header.

**Title + subtitle pair:** page title `.t-title` (16) / section heading `.t-section` (14) / card title
`.t-card` (14), with subtitle `.t-meta` (12); in dense lists the subtitle drops to `.t-meta-sm` (11).

**Colour is not part of a role.** Apply colour with utilities (`text-foreground`,
`text-muted-foreground`, `text-brand`, …) on top of the role. `.t-*` roles only set
size + weight + line-height (+ uppercase/tracking for `.t-caption`).

**Pick-the-right-role cheat-sheet** (common situation → role; controls are the exception, see §4):

| Situation | Role |
|---|---|
| Whole-**page** / view title — one per page ("Prompt Library", "Datenbank", "Einstellungen") | `.t-title` |
| Section / panel / content heading ("Report", "Gespräche", chat title, "Dokumente", "Editor-Assistent", "Alle Dokumente", "Neuer Prompt", report sections, Settings sections, modal titles) | `.t-section` |
| Card / item / row title — research run card, file/doc card, Settings row title ("Demo-Modus") | `.t-card` |
| Reading paragraph / description / instructional hint / long-form prompt textarea | `.t-body` |
| List / nav / menu row primary — chat thread, file label, sidebar nav item, category card | `.t-list` |
| Form-field label, group/folder header ("Neuer Ordner") | `.t-label` |
| Subtitle, metadata, breadcrumb, helper text | `.t-meta` (dense lists → `.t-meta-sm`) |
| File metadata (size, path), dense secondary line | `.t-meta-sm` |
| UPPERCASE section eyebrow / divider ("OHNE ORDNER", `@`-menu group header) | `.t-caption` |
| Count, timestamp, keyboard legend, tiny non-uppercase hint | `.t-hint` |
| Run/job id, URL, `@files:` prefix, tabular token | `.t-mono` |
| Hero / empty-state heading ("Womit kann ich helfen?") | `.t-display` |
| Button label, chip, tab, `kbd`, model picker, **any control** | **not** a `.t-*` role → §4 |
| Single-line `<input>`/search field text | `text-sm` (14, utility); long-form `<textarea>` → `.t-body` |

---

## 3. Icon roles (`.icon-*`)

Three tiers. One role = one size; the same UI function must never render two icon sizes.
(The shared `Button` already forces `size-4` on its `svg`, so button glyphs are `.icon-md` by default.)

| Role | px | Use it for |
|---|---|---|
| `.icon-md` | 16 (`size-4`) | Nav rail, toolbars, button glyphs, coloured tone tiles |
| `.icon-sm` | 14 (`size-3.5`) | Chat-input footer (most compact controls), **file/list leading icons** (file explorer), inline & secondary actions, dense metadata icons |
| `.icon-xs` | 12 (`size-3`) | Status dots, chevrons, micro affordances |

Decorative status dots smaller than 12px (e.g. `size-1.5` running dots) and large empty-state hero
glyphs (`size-6`+ inside a big tile) are deliberate special cases, **not** icon roles.

**Adding a new icon glyph is welcome — this is not a closed set.** `lucide-react` is a real
dependency, kept precisely so you can reach for any of its glyphs when nothing in the current set
fits. Adding one is a normal, lightweight change (no design review, no `globals.css` edit, no role
change — the `.icon-*` roles in the table above are what stays fixed, the *list of available glyphs*
is open). Route every icon through the single barrel `src/components/icons/index.tsx`, so feature
code always imports `from '@/components/icons'` (never directly from `lucide-react`). Two ways to add
one, both fine:

- **Re-export** straight from `lucide-react` (the `export { ... } from 'lucide-react'` line near the
  top of the barrel) — quickest, pulls the glyph from the package at build time.
- **Vendor** it via `createIcon('Name', 'lucide-name', <svg paths>)` (the dominant pattern in the
  barrel) — copies the SVG paths in so the glyph is independent of the package version.

Prefer reusing an existing export before adding a near-duplicate; otherwise just add it. (Distinct
from §0's "new size/role" rule, which *does* gate adding a new `.icon-*` tier.)

---

## 4. Control primitives & their canonical sizes

Control **labels** (buttons, chips, tabs, segmented controls, keyboard hints) are owned by shared
control components, **not** by `.t-*` text roles. Their sizes are fixed here so "a chip" / "a tab"
looks identical everywhere.

| Control | Component / canonical | Notes |
|---|---|---|
| Button | `components/ui/button.tsx` — `sm` = `h-8 px-3 text-xs`; `default` = `h-9 px-4 py-2 text-sm`; `icon` = compact `size-7`/`size-8` | Never combine an explicit `h-9` with `size="sm"` — pick one height |
| Select trigger | `components/ui/select.tsx` — `default` = `h-9 text-sm`; `toolbar` = `h-8 text-xs`; `table` = `h-7 text-xs shadow-none` | Use `table` only inside dense data rows/management grids where the select edits a row value rather than acting as a form field |
| Switch | `components/ui/switch.tsx` — `default` = `h-5 w-9`; `table` = `h-4 w-7` | Use `table` for boolean actions inside dense table rows; keep default for Settings rows and larger forms |
| Status badge | `features/settings/parts.tsx` — `default` = `h-7 text-xs`; `table` = `h-5 .t-hint` | Use `table` for scan labels in data rows so status chips match the Quota admin table |
| Pill / filter chip | **`components/ui/chip.tsx`** (`<Chip active dot count>`) — `h-6 px-2.5 rounded-full text-[11px] font-medium`, active = `bg-brand-subtle text-brand` | Use the component, not raw classes. Tone meaning via the optional `dot` (see §5) |
| `kbd` key badge | **`components/ui/kbd.tsx`** (`<Kbd>`) — `h-4 min-w-4 px-1 text-[10px]`, bordered | keyboard hints (e.g. mention-menu footer) |
| Segmented / tab control | `h-7 text-xs font-medium` (convention) | e.g. Editor "Offen/Erledigt", Report tabs, Prompt-Library category + visibility filters. Not yet one primitive (active treatments differ: `bg-background shadow` vs `bg-accent`); a `SegmentedControl` is a future extraction — keep `h-7`/`text-xs`. |
| Toolbar row | all controls share one height (**`h-8`**) | buttons + selects + toggles must match within a row |
| Composer-footer control text (chat-input model picker, effort) | `text-xs` (12) on the `<Button>` | a utility (not a `.t-*` role — it would be overridden, §0.7); sits one step below the 14px footer icons |
| Dropdown / context-menu items | `text-sm` (14) — `components/ui/dropdown-menu.tsx` | every menu item, label, shortcut row (e.g. the model-picker pop-up options) |
| `@`-autocomplete (mention menu) | primary `text-[13px]`, secondary `text-[11px]`, group-header / footer / `kbd` `text-[10px]` | gold-standard `components/ui/mention-menu.tsx` — the 13/11/10 dense scale; do not alter |
| Chat-input context chip | tone-coded pill via `attachmentChipVisual` (`features/files/attachmentChips.ts`) + `lib/tone.ts`; `h-6 rounded-full`, label `text-[11px]` | the chips added when attaching `@research`/`@files`; hue = brand/success/file by kind |

> The chip (`text-[11px]`) and kbd (`text-[10px]`) sizes now live **only** inside their `components/ui`
> primitives, so feature code never writes those raw values. This is enforced: a `no-restricted-syntax`
> ESLint guard in `eslint.config.js` **warns** on any `text-[..px]` / `leading-[..]` under
> `src/features/**/*.tsx` — use a `.t-*` role or a control primitive instead. Do not invent new
> control sizes; extend a primitive.

---

## 5. Colour roles

Colours are OKLCH tokens in `globals.css` (4 presets + dark + high-contrast). Use them by role:

| Role | Token / class | Use |
|---|---|---|
| Primary action | `bg-brand text-brand-foreground hover:bg-brand/90` (brand = purple) | The one prominent CTA per toolbar (Composer send, Save, Upload, Add). **Exactly one** brand-filled button per toolbar; everything else is `outline`/`ghost`. Do **not** use the near-black `bg-primary` for content CTAs. |
| Selected / active | `bg-brand-subtle text-brand` (optionally a `border-l-2` accent bar) | Selected list row, active toggle, active nav item — one pattern everywhere |
| Tones (`lib/tone.ts`) | `brand` / `success` / `file` / `warning` | Semantic accents: mention scopes, prompt categories (instruction=brand, function=success, context=warning), file identity = `file` (cyan) |
| User message bubble | `--user-bubble-*`, `.inqtrix-user-bubble`, `.inqtrix-user-avatar` | User-authored messages in Chat and Knowledge Desk. Default is neutral gray; Settings offers curated `gray` / `mint` / `orange` / `sky` / `violet` / `ink` tones with light, dark, and high-contrast values. Assistant messages and Markdown rendering stay separate. |
| Incognito chat header | `.inqtrix-chat-header--incognito` (token scope, `globals.css`) | Applied to the chat header bar only when incognito is active. Re-maps the subtree's neutral/brand tokens to the **inverted** surface (dark bar in light mode, light bar in dark mode) by deriving everything from `--primary` / `--primary-foreground`, so it stays correct across presets and high-contrast. Signals the "nothing is saved" state; title, icons and badge follow automatically. Not a `.t-*` role — a surface scope; keep in sync with `globals.css`. |
| Muted / disabled | `text-muted-foreground`, disabled `text-muted-foreground/45` | Secondary text and disabled controls |

Semantic foreground tokens (`brand`, `success`, `warning`, `file`, and
`muted-foreground`) are the readable text variants as well as the icon colours.
Normal informative text must use the token at full opacity so it remains WCAG
AA on both the base and matching subtle surface. Opacity modifiers are reserved
for decorative glyphs or genuinely disabled content; they must not be used to
make counts, timestamps, labels, identifiers, statuses, or metadata quieter.

---

## 6. Surface — radius & shadow tiers

| Tier | Radius | Shadow | Use |
|---|---|---|---|
| Control / input | `rounded-md` | — | buttons, inputs, chips |
| Card / panel | `rounded-lg` | `shadow-[0_1px_2px_var(--shadow-hairline)]` | cards, list rows, isolated panels |
| Fluid settings/data surface | no extra radius when already inside a page surface | — | Settings sections, admin/database-style tables and workspace management grids; avoid nested card chrome. Use a single column-header or section-header hairline, then rely on alignment, spacing, muted surfaces, and hover states; fluid tables do not add default row-by-row separators |
| Floating | `rounded-xl` | `shadow-[0_8px_28px_-12px_var(--shadow-soft)]` | composer, chat bubbles |
| Overlay / menu | `rounded-xl` | `shadow-lg` | popovers, mention menu, dropdowns |

---

## 7. List & nesting model

Used by every navigation/explorer list (chat history, file explorer, editor tree, mention menu):

```
Panel header        .t-section (14) + .icon-md (16)    e.g. "Gespräche", "Dokumente"
  (a whole-page header is .t-title (16), e.g. "Datenbank")
  Group/folder hdr   .t-label (12)  + .icon-sm folder + .t-hint count   e.g. "Neuer Ordner"
    Entry            .t-list (13)   — every depth                       e.g. a chat thread, a file
  Section divider    .t-caption (10, UPPERCASE) + .icon-sm              e.g. "OHNE ORDNER"
  Date / meta        .t-hint (10)
```

**Depth is shown by indentation + an accent bar + the quieter group header — not by shrinking the
entry text per level.** Per-level text-shrinking is the classic source of drift (same role, different
size depending on nesting), so all entries stay `.t-list` (13) regardless of depth.

**Category in a grouped list is the group header's job — not a per-row tag.** When rows sit under a
category eyebrow (dot + label + count), do **not** repeat that category as a filled badge on every row
(redundant noise). The allowed per-row cue is a quiet leading tone icon (`.icon-sm` in the category
`toneText`); the colour identity already lives in the group header + the selected row's left accent bar.
(Reference: the Prompt Library list.)

---

## 8. Layout / chrome conventions

| Element | Convention |
|---|---|
| Top panel header bar | `.inqtrix-panel-header` (`var(--header-h)` tall), `px-3`/`px-4` horizontal, title `.t-section`; keep these single-row and move secondary context to body content or hover text (a whole-page header is `.t-title`) |
| `--header-h` | `2.625rem` (42px) — the compact sticky app topbar and panel-header height; the AppRail width and overlay offsets use the same value so the chrome stays geometrically aligned |
| Workspace sidebars | compact `bg-surface/50` navigation rails with `border-r`, dense list rows, and content descriptions in the main panel, not repeated as large sidebar intro blocks; secondary workspace headers align to the app rail's first icon center, sidebar header icons share the same leading icon column as nav rows, and the main panel header hairline aligns with the sidebar header hairline; group anchors use dark `t-caption` text plus outline icons; grouped Settings-nav items may use a subtle indented rail with item nodes on desktop, with the selected row spanning the full group width and the selected node using the brand accent; runtime/status badges belong in the sidebar footer, not beside the title |
| Toolbar row | one shared control height `h-8`; gap `gap-2`; **one** brand CTA |
| Content padding | base `p-4`, wider screens `md:p-6` |
| Settings rows | row title `.t-list`, helper `.t-meta`, section title `.t-section`; rows sit in quiet structured-list sections with optional section-header hairlines, rounded row hover, and no default outer `border-y` frame |
| Admin data sections | full available content width with the page's fixed horizontal padding; do not cap data tables/grids at the Settings form max-width |
| Read-only status surfaces | compact structured sections with `min-h-9` rows, `.t-list` labels, one section-header hairline, table-density status badges, and optional info hover-cards using the model-picker card treatment |
| Spacing rhythm | 4/8px grid — primary gap `gap-2`, tight `gap-1.5`; padding `px-2/3 py-1.5/2` |

---

## 9. Markdown reading-system contract

The Markdown rendering used in the **chat messages, editor document and report panel** is intentionally
**not** governed by these roles. The following form a deliberate reading system
with their own scale (e.g. report body line-height 1.75 vs. chat 1.45 is a
wanted difference):

- [`src/components/markdown/MarkdownRenderer.tsx`](src/components/markdown/MarkdownRenderer.tsx)
- the CSS classes `.editor-prose`, `.report-markdown`, `.chat-markdown` in `globals.css`

Knowledge citation markers (`a[href^="#kref-"]` inside `.chat-markdown`/`.report-markdown`) carry a
dedicated micro-chip style in `globals.css` (small superscript brand pill with built-in inter-chip gap)
so glued clusters like `K6K15` read as discrete badges. This is an **element style on the Markdown
surface**, not a `.t-*` role — no §2/§3 row; keep it in sync with `globals.css`.

The `.t-*` roles apply to **non-Markdown UI text only**.

Security, accessibility, correctness, and performance work may change the
renderer. Such changes require focused regression coverage and a synchronized
update to this contract when they alter the final typography, component
styling, or interaction model. Performance changes may adjust warm-up, syntax
highlighting caches, or the intermediate structural fallback without
introducing a second Markdown look.
All variants use the same synchronous Markdown parse; asynchronous work is limited to Shiki tokens
and Mermaid SVGs inside their frame-one structural shells.

**Mermaid figures:** a ```` ```mermaid ```` fence renders as a diagram via
[`src/components/markdown/MermaidFigure.tsx`](src/components/markdown/MermaidFigure.tsx) — the ONE
integration point is the fence dispatch inside `MarkdownCodePre`, so every renderer consumer (chat,
knowledge, reports, agent canvas, inline answers) gets it. A successful diagram is **unboxed** and
keeps Mermaid's native width up to the available document width: wide diagrams shrink responsively,
while compact diagrams are never enlarged beyond their intended inline type scale. The viewer may
apply only a modest enlargement before the viewport limit; only parser errors retain a warning frame.
The diagram itself is
token-mapped through mermaid's `base` theme with `themeVariables` read from the CSS custom
properties (`--surface`/`--background`/`--foreground`), so nodes render like Inqtrix surfaces in
both light and dark, while connectors and arrowheads use `--foreground` (ink in light mode, white in
dark mode) for publication-grade contrast. It follows the highlighter's warm-up allowance: a stable
hull with a "Diagramm wird erstellt …" hint fills in asynchronously from a shared 256-entry LRU
cache keyed by (code, theme, preset, contrast). Visible figures render immediately; figures within 1200px of the
nearest scroll viewport warm during browser idle time. Render errors show a **warning-tone box**
(border-warning, AlertTriangle) with the parser message plus the source — content never disappears
silently. Security stays `securityLevel: 'strict'`. Labels default to the HTML-free SVG text path
(`htmlLabels: false`); a fence whose source carries `$$…$$` math renders — and only then — with
mermaid's HTML labels, the sole mode in which its KaTeX support produces output. Those math renders
pass a strict DOMPurify policy twice: mermaid's own passes via `dompurifyConfig`, then an app-owned
pass over the final SVG before injection. MathML/KaTeX markup is allowed; network-capable tags and
attributes (`img`/`image`/`video`/`audio`/`source`/`track`, `src`/`srcset`/`poster`/`background`)
are forbidden, so the external-image privacy boundary below cannot be bypassed through a diagram
label, and without a usable sanitizer the math render fails closed into the visible error box.
Do not relax any part of this, and do not add a second sanitize path beside it. Shiki
highlighting follows the same visible-now / near-viewport-idle policy and uses its own 256-entry LRU,
while preserving the frame-one plaintext code shell required by the synchronous chat contract.

**Rendered Markdown blocks** use one shared 16px vertical rhythm and one
`MarkdownBlockFrame` action pattern across chat, knowledge, reports, file previews, Agent Canvas,
and inline Agent Desk answers. Mermaid figures expose expand, source-copy, and high-resolution PNG;
tables expose exact Markdown-source copy plus high-resolution PNG and UTF-8 CSV. The vertical
icon-button rail appears on hover or keyboard focus and stays visible on no-hover devices. It may sit
in the reading-column gutter only when the nearest horizontal clipping boundary has enough measured
space for the complete rail and its hit-testable gap; otherwise it sits inside the block at the top
right. PNG work is loaded only on demand, and export failures must stay visible and announced. Do not
add workspace-specific table or Mermaid wrappers beside this shared integration.

External Markdown images are a privacy boundary: relative and same-origin HTTP(S) sources may load
directly, while cross-origin HTTP(S) sources render an explicit load control first. Approved images
use lazy decoding and `referrerPolicy="no-referrer"`; do not reintroduce automatic cross-origin image
requests or a second image policy in individual consumers.

---

## 10. How to extend

1. First try to fit the need to an existing role. Almost everything maps to one.
2. If a genuinely new text *function* exists (not just "slightly different size"), add a role:
   - add the class to `globals.css` `@layer components`,
   - add a row to §2/§3/§4 here with size + weight + line-height and a concrete "used where",
   - keep it on the scale (no values between existing steps without a strong reason).
3. Migrate existing usages to the role; never leave the same function on two values.
4. Control labels (chips, tabs, kbd) belong in a `components/ui` primitive (§4), not a `.t-*` role —
   extend or add a primitive there so feature code stays free of raw `text-[..px]`.
5. Verify visually (light + dark) and run the npm workspace typecheck, lint,
   and test commands from the repository root.
   The design-lint guard (`eslint.config.js`) warns on any new arbitrary `text-[..px]`/`leading-[..]`
   in `src/features/**` — keep it at zero.
