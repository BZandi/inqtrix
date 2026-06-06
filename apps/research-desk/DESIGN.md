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

- **P1 — Density with calm.** Compact ≠ cramped. A 4px spacing grid, clear grouping, lots of quiet
  structure. Every element earns its place; no filler, no decorative numbers. Prefer one more line of
  content over one line of air.
- **P2 — Colour is function.** Neutral by default. The brand accent only for selection, the primary
  action, and identity; the semantic set (success / warning / destructive / file) only for real
  status. Same colour ⇒ same meaning. No gradients, no colourful tiles, no emojis. (§5)
- **P3 — Separation by line & space.** Hairline 1px borders, whitespace, and gentle surface shifts
  (`surface`/`card`) instead of heavy colour blocks. Shadows soft and sparse, never hard. (§6)
- **P4 — Hierarchy via size *and* weight.** A real size step (`display 24 > page title 16 > section/card 14 > body 14 >
  list 13 > label/meta 12 > 11 > 10`, §2) **plus** weight; secondary text is `muted-foreground`;
  numbers/IDs are mono + `tabular-nums`. Few sizes, each owned by a role.
- **P5 — Motion explains, it doesn't entertain.** Animation shows origin (entry), state (loading,
  active) and change (height, counts). Short, soft, directed. Nothing blinks, bounces or spins forever
  on real content — loops are reserved for "working" signals.
- **P6 — Consistency beats creativity.** One popover layout, one active-accent, one hover-card style,
  one empty-state schema everywhere. Recognisability = trust.
- **P7 — Accessible by default.** Visible focus ring, full keyboard paths, sufficient contrast,
  hit-targets ≥ icon-button size, `prefers-reduced-motion` respected, dark/light equal.

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
| Metric update | one-shot count/flash | `inqtrix-metric-flash` 0.6s |
| Expand / collapse | soft height | `grid-template-rows 0fr→1fr`, ~0.3s |
| Skeleton / loading | shimmer | loop **only while loading** |

**Rules:** no endless animation on real content (loops only for "working" signals); everything is
`prefers-reduced-motion`-safe (decorative motion off; the visible end-state is the base); off in print/PDF.

### Interaction patterns (reuse these — they are the reference look)

- **Popover / autocomplete menu** (the `@`-mention menu): breadcrumb/scope header left + hit-counter
  right → dense grouped rows (icon · primary `t-list` / secondary `t-meta-sm`) with a 2px active accent
  bar → keyboard footer (`Kbd`). Select via `onMouseDown` + `preventDefault`.
- **Dropdown menu** (the model picker): grouped `text-sm` rows under section eyebrows, an info
  affordance, a check on the active item; a segmented reasoning toggle in the footer.
- **Segmented control** (modes / filters / reasoning level): 2–4 short options, active = `bg-background`
  + soft shadow on a `bg-surface` track (§4, `h-7`).
- **Hover-card** (rich tooltip): small card — title + one sentence + optional 2–3 compact figures/tags
  — for definitions, derivations, source previews.
- **Status / tier badge:** `*-subtle` surface + dot/icon + short label.
- **Empty state:** centred, neutral icon in a circle, one-sentence title + one-sentence hint.
- **List:** group header (dot + label + count) → dense rows → 2px active accent; hover-only actions
  (`opacity-0 group-hover:opacity-100`).

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
5. **The Markdown renderer is off-limits** — see §9.
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
| `.t-list` | 13 | 600 | 1.25rem | **Dense list / menu primary label** — chat history thread rows, file-list labels, mention-menu primary option, nav rows |
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

---

## 4. Control primitives & their canonical sizes

Control **labels** (buttons, chips, tabs, segmented controls, keyboard hints) are owned by shared
control components, **not** by `.t-*` text roles. Their sizes are fixed here so "a chip" / "a tab"
looks identical everywhere.

| Control | Component / canonical | Notes |
|---|---|---|
| Button | `components/ui/button.tsx` — `sm` = `h-8 px-3 text-xs`; `default` = `h-9 px-4 py-2 text-sm`; `icon` = compact `size-7`/`size-8` | Never combine an explicit `h-9` with `size="sm"` — pick one height |
| Pill / filter chip | **`components/ui/chip.tsx`** (`<Chip active dot count>`) — `h-6 px-2.5 rounded-full text-[11px] font-medium`, active = `bg-brand-subtle text-brand` | Use the component, not raw classes. Tone meaning via the optional `dot` (see §5) |
| `kbd` key badge | **`components/ui/kbd.tsx`** (`<Kbd>`) — `h-4 min-w-4 px-1 text-[10px]`, bordered | keyboard hints (e.g. mention-menu footer) |
| Segmented / tab control | `h-7 text-xs font-medium` (convention) | e.g. Editor "Offen/Erledigt", Report tabs, Prompt-Library category filter. Not yet one primitive (active treatments differ: `bg-background shadow` vs `bg-accent`); a `SegmentedControl` is a future extraction — keep `h-7`/`text-xs`. |
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
| Muted / disabled | `text-muted-foreground`, disabled `text-muted-foreground/45` | Secondary text and disabled controls |

---

## 6. Surface — radius & shadow tiers

| Tier | Radius | Shadow | Use |
|---|---|---|---|
| Control / input | `rounded-md` | — | buttons, inputs, chips |
| Card / panel | `rounded-lg` | `shadow-[0_1px_2px_var(--shadow-hairline)]` | cards, list rows, sections |
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

---

## 8. Layout / chrome conventions

| Element | Convention |
|---|---|
| Top panel header bar | `h-12` tall, `px-3` horizontal, title `.t-section` (a whole-page header is `.t-title`) |
| `--header-h` | `3rem` — the sticky topbar height (single source; AppRail offset subtracts it) |
| Toolbar row | one shared control height `h-8`; gap `gap-2`; **one** brand CTA |
| Content padding | base `p-4`, wider screens `md:p-6` |
| Spacing rhythm | 4/8px grid — primary gap `gap-2`, tight `gap-1.5`; padding `px-2/3 py-1.5/2` |

---

## 9. The Markdown renderer is off-limits

The Markdown rendering used in the **chat messages, editor document and report panel** is intentionally
**not** governed by these roles. Leave the following untouched — they are a deliberate, well-tuned
reading system with their own scale (e.g. report body line-height 1.75 vs. chat 1.45 is a wanted
difference):

- [`src/components/markdown/MarkdownRenderer.tsx`](src/components/markdown/MarkdownRenderer.tsx)
- the CSS classes `.editor-prose`, `.report-markdown`, `.chat-markdown` in `globals.css`

The `.t-*` roles apply to **non-Markdown UI text only**.

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
5. Verify visually (light + dark) and run `pnpm --filter @inqtrix/research-desk typecheck lint test`.
   The design-lint guard (`eslint.config.js`) warns on any new arbitrary `text-[..px]`/`leading-[..]`
   in `src/features/**` — keep it at zero.
