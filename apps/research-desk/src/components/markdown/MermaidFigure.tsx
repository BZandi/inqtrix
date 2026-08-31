import { useCallback, useRef, useState } from 'react'
import DOMPurify from 'dompurify'

import { AlertTriangle, Copy, Download, Maximize2 } from '@/components/icons'
import {
  MarkdownBlockFrame,
  useMarkdownBlockAction,
  type MarkdownBlockAction,
} from '@/components/markdown/MarkdownBlockFrame'
import {
  copyMarkdownBlockText,
  downloadMarkdownBlockPng,
  MARKDOWN_BLOCK_FILE_NAMES,
} from '@/components/markdown/markdownBlockExport'
import { Dialog } from '@/components/ui/dialog'
import { useLocale } from '@/i18n/LocaleProvider'
import { useTheme, type ContrastMode, type ThemePreset } from '@/theme/ThemeProvider'
import { cn } from '@/lib/utils'
import { useStructuralRenderBlocker } from '@/motion/StructuralLoadBoundary'
import { BoundedLruCache, MARKDOWN_RENDER_CACHE_CAPACITY } from './boundedLruCache'
import { mermaidNaturalWidth, mermaidPreviewMaxWidth } from './mermaidSizing'
import { useMarkdownCacheEntry } from './useMarkdownCacheEntry'
import { useProgressiveMarkdownWork } from './useProgressiveMarkdownWork'

/**
 * Renders a ```mermaid fence as a diagram.
 *
 * Follows the EXACT `markdownTokenCache` discipline of the code
 * highlighter (gotcha #29): chat history re-renders stay synchronous —
 * a cached diagram paints its SVG immediately; an uncached one shows a
 * stable hull ("Diagramm wird erstellt …") while the async render fills
 * the module cache once and notifies subscribers. The mermaid bundle is
 * a lazy chunk (first diagram pays it, the main bundle never does).
 *
 * Security: `securityLevel: 'strict'` stays. Labels default to the SVG
 * text-node path (`htmlLabels: false`) — no HTML at all. A fence whose
 * source carries `$$…$$` math switches that ONE render to mermaid's
 * HTML-label mode, the only mode where its KaTeX support renders; those
 * renders pass a strict DOMPurify policy twice — mermaid's own passes
 * via `dompurifyConfig`, then the app-owned pass over the final SVG
 * before it is injected. MathML/KaTeX markup is allowed; network-capable
 * tags and attributes are forbidden, so the Markdown image privacy
 * boundary cannot be bypassed through a diagram label. Without a usable
 * sanitizer the math render fails CLOSED into the visible error box.
 * Render failures are LOUD by design: a warning box with the parser
 * message plus the source, never a silently missing block.
 */

type MermaidTheme = 'light' | 'dark'

/** Mermaid detects math only via `$$…$$` (its own `katexRegex`); the same
 * pattern gates the HTML-label mode here, so every diagram WITHOUT math
 * keeps today's byte-identical SVG-text rendering. No `g` flag —
 * `.test` on a global regex is stateful. */
const MERMAID_KATEX_PATTERN = /\$\$(.*?)\$\$/

/** Sanitizer policy for HTML-label (math) renders, shared by both layers.
 * The MathML additions keep KaTeX output intact — DOMPurify's default
 * profile drops `semantics`/`annotation` while KEEPING their content,
 * which would leak raw TeX source as stray text. The forbid lists close
 * every auto-fetching vector a label could smuggle in.
 *
 * `style` differs BETWEEN the layers on purpose: mermaid's own label
 * sanitize must keep forbidding `<style>` in label CONTENT (providing a
 * `dompurifyConfig` replaces its default, which is exactly that), while
 * the app-side pass over the final SVG must NOT strip the DOCUMENT-level
 * `<style>` element — that stylesheet is generator-owned (mermaid's
 * theme CSS; without it every node paints as a black box) and it exists
 * identically in the non-math path, so keeping it is parity, not a
 * relaxation. */
const MERMAID_MATH_ADD_TAGS = [
  'foreignobject',
  'semantics',
  'annotation',
  'annotation-xml',
]
const MERMAID_MATH_FORBID_TAGS = [
  'img',
  'image',
  'video',
  'audio',
  'source',
  'track',
  'iframe',
  'object',
  'embed',
]
const MERMAID_LABEL_FORBID_TAGS = ['style', ...MERMAID_MATH_FORBID_TAGS]
const MERMAID_MATH_FORBID_ATTR = ['src', 'srcset', 'poster', 'background', 'ping']

/** App-owned guarantee at the injection boundary: whatever mermaid (or a
 * future mermaid version) emits in HTML-label mode, no network-capable
 * element or attribute reaches `dangerouslySetInnerHTML`. Fails CLOSED:
 * without a usable DOMPurify (no DOM) the diagram becomes a visible
 * render error, never an unsanitized injection. */
function sanitizeMermaidMathSvg(svg: string): string {
  if (!DOMPurify.isSupported) {
    throw new Error('Math-Diagramm nicht darstellbar: SVG-Sanitizer in dieser Umgebung nicht verfügbar.')
  }
  return DOMPurify.sanitize(svg, {
    ADD_TAGS: MERMAID_MATH_ADD_TAGS,
    FORBID_ATTR: MERMAID_MATH_FORBID_ATTR,
    FORBID_TAGS: MERMAID_MATH_FORBID_TAGS,
    // This DOMPurify build treats ONLY `annotation-xml` as an HTML
    // integration point, so it would strip the HTML label content inside
    // `<foreignObject>` as an invalid namespace transition (label boxes
    // measured non-empty by mermaid then render empty). Declaring
    // foreignObject as the integration point it is per SVG spec restores
    // the content — which still passes the full tag/attribute policy
    // above, so no network-capable or scripting markup survives inside.
    HTML_INTEGRATION_POINTS: { 'annotation-xml': true, foreignobject: true },
    USE_PROFILES: { html: true, mathMl: true, svg: true, svgFilters: true },
  })
}

type MermaidCacheEntry =
  | { kind: 'svg'; svg: string }
  | { kind: 'error'; message: string }

const mermaidCache = new BoundedLruCache<string, MermaidCacheEntry>(
  MARKDOWN_RENDER_CACHE_CAPACITY,
)
const mermaidPending = new Set<string>()

let mermaidCounter = 0
let mermaidRenderQueue: Promise<void> = Promise.resolve()

function cacheKey(
  code: string,
  theme: MermaidTheme,
  preset: ThemePreset,
  contrastMode: ContrastMode,
): string {
  return `${theme}\0${preset}\0${contrastMode}\0${code}`
}

function serializeMermaidRender<Result>(render: () => Promise<Result>): Promise<Result> {
  const result = mermaidRenderQueue.then(render)
  mermaidRenderQueue = result.then(
    () => undefined,
    () => undefined,
  )
  return result
}

/** Mermaid's color math (khroma) cannot parse modern color spaces like
 * the app's `oklch(...)` tokens — resolve every token to sRGB hex via a
 * throwaway canvas (the one conversion path every oklch-capable browser
 * guarantees). Falls back LOUDLY predictable: an unparsable value keeps
 * the per-theme fallback instead of crashing the render. */
function toHexColor(value: string, fallback: string): string {
  try {
    const canvas = document.createElement('canvas')
    canvas.width = 1
    canvas.height = 1
    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) return fallback
    context.fillStyle = value
    context.fillRect(0, 0, 1, 1)
    const [r, g, b] = context.getImageData(0, 0, 1, 1).data
    return `#${[r, g, b]
      .map((channel) => channel.toString(16).padStart(2, '0'))
      .join('')}`
  } catch {
    return fallback
  }
}

/** Design-token colors for the diagram (read once per render, so a
 * theme switch — new cache key — picks up the current values). The
 * `base` theme + explicit variables keeps mermaid's default look out
 * of the app entirely: nodes render like Inqtrix surfaces. */
function themeVariablesFor(theme: MermaidTheme): Record<string, string> {
  const styles = getComputedStyle(document.documentElement)
  const dark = theme === 'dark'
  const token = (name: string, fallback: string) => {
    const raw = styles.getPropertyValue(name).trim()
    return raw ? toHexColor(raw, fallback) : fallback
  }
  const surface = token('--surface', dark ? '#1b1e27' : '#f8fafc')
  const background = token('--background', dark ? '#090a0c' : '#ffffff')
  const foreground = token('--foreground', dark ? '#fafafa' : '#17181c')
  return {
    background: 'transparent',
    fontFamily:
      styles.getPropertyValue('--font-sans').trim()
      || 'ui-sans-serif, system-ui, sans-serif',
    fontSize: '13px',
    primaryColor: surface,
    primaryBorderColor: foreground,
    primaryTextColor: foreground,
    lineColor: foreground,
    textColor: foreground,
    secondaryColor: token('--muted', dark ? '#23262f' : '#eef0f3'),
    tertiaryColor: 'transparent',
    clusterBkg: 'transparent',
    edgeLabelBackground: background,
  }
}

const MERMAID_MEASUREMENT_HOST_ID = 'inqtrix-mermaid-measure'

/** The out-of-flow host mermaid lays its diagram out in.
 *
 * Mermaid needs a REAL laid-out node to measure a diagram (`getBBox` has no
 * boxes under `display: none`). Given no container it appends that node to
 * `document.body` — and a body child below the fold gives the DOCUMENT
 * scroll extent. In this fixed-shell app (`main` is `h-svh overflow-hidden`,
 * the panes are the only scrollers) that painted a window scrollbar for the
 * ~175ms of a render and shifted the whole UI sideways by the scrollbar
 * width. Measured: document 900px -> 1050px, workspace 1470px -> 1455px.
 *
 * `position: fixed` is the fix: laid out, so measurement is unchanged, but
 * outside the document flow, where it can never contribute scroll extent.
 * `visibility: hidden` (not `display: none`) keeps the boxes.
 *
 * ONE shared host is safe: `serializeMermaidRender` runs renders one at a
 * time, and mermaid clears the container (`innerHTML = ''`) on entry. */
function mermaidMeasurementHost(): HTMLElement {
  const existing = document.getElementById(MERMAID_MEASUREMENT_HOST_ID)
  if (existing) return existing
  const host = document.createElement('div')
  host.id = MERMAID_MEASUREMENT_HOST_ID
  host.setAttribute('aria-hidden', 'true')
  host.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:0;'
    + 'overflow:hidden;visibility:hidden;pointer-events:none;'
  document.body.appendChild(host)
  return host
}

/** Test-only visibility into a render outcome (the module cache is
 * otherwise private): lets the node suite pin the fail-closed sanitizer
 * contract without a DOM. */
export function peekMermaidRender(
  code: string,
  theme: MermaidTheme,
  preset: ThemePreset,
  contrastMode: ContrastMode,
): MermaidCacheEntry | undefined {
  return mermaidCache.peek(cacheKey(code, theme, preset, contrastMode))
}

/** Exported for the leak-contract test only. */
export async function ensureMermaidRender(
  code: string,
  theme: MermaidTheme,
  preset: ThemePreset,
  contrastMode: ContrastMode,
): Promise<void> {
  const key = cacheKey(code, theme, preset, contrastMode)
  if (mermaidCache.has(key) || mermaidPending.has(key)) return
  mermaidPending.add(key)
  mermaidCounter += 1
  const renderId = `inqtrix-mermaid-${mermaidCounter}`
  try {
    const themeVariables = themeVariablesFor(theme)
    const mermaid = (await import('mermaid')).default
    const wantsMathLabels = MERMAID_KATEX_PATTERN.test(code)
    const { svg } = await serializeMermaidRender(async () => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        htmlLabels: wantsMathLabels,
        ...(wantsMathLabels
          ? {
            dompurifyConfig: {
              ADD_TAGS: MERMAID_MATH_ADD_TAGS,
              FORBID_ATTR: MERMAID_MATH_FORBID_ATTR,
              FORBID_TAGS: MERMAID_LABEL_FORBID_TAGS,
            },
          }
          : {}),
        theme: 'base',
        themeVariables,
        flowchart: {
          curve: 'basis',
          nodeSpacing: 44,
          padding: 4,
          rankSpacing: 52,
        },
        sequence: {
          diagramMarginX: 24,
        },
      })
      return mermaid.render(renderId, code, mermaidMeasurementHost())
    })
    // The cache stores the SANITIZED SVG, so both injection sites (inline
    // figure and expand dialog) can only ever receive the cleaned string.
    mermaidCache.set(key, {
      kind: 'svg',
      svg: wantsMathLabels ? sanitizeMermaidMathSvg(svg) : svg,
    })
  } catch (error) {
    // Loud, never silent: the figure shows the parser message + source.
    const message = error instanceof Error ? error.message : String(error)
    console.warn('Inqtrix mermaid render failed.', error)
    mermaidCache.set(key, { kind: 'error', message })
  } finally {
    mermaidPending.delete(key)
    // Mermaid builds a temporary `#d<renderId>` inside the measurement host
    // and removes it on success — but LEAKS it on a parse error. The host
    // keeps a stray out of the document flow (see mermaidMeasurementHost),
    // so it can no longer move the shell; it is still removed here so the
    // host stays empty between renders instead of retaining dead SVG.
    // Pinned by MermaidFigure.test.ts.
    document.getElementById(`d${renderId}`)?.remove()
  }
}

export function MermaidFigure({ code }: { code: string }) {
  const { t } = useLocale()
  const { contrastMode, preset, resolvedTheme } = useTheme()
  const theme: MermaidTheme = resolvedTheme === 'dark' ? 'dark' : 'light'
  const key = cacheKey(code, theme, preset, contrastMode)
  const figureRef = useRef<HTMLElement | null>(null)
  const diagramRef = useRef<HTMLDivElement | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const copyAction = useMarkdownBlockAction()
  const pngAction = useMarkdownBlockAction()
  const entry = useMarkdownCacheEntry({
    cache: mermaidCache,
    cacheKey: key,
  })
  const naturalWidth = entry?.kind === 'svg'
    ? mermaidNaturalWidth(entry.svg)
    : undefined
  const previewMaxWidth = entry?.kind === 'svg'
    ? mermaidPreviewMaxWidth(entry.svg)
    : undefined
  const blocksStructuralReveal = useStructuralRenderBlocker(entry === undefined)

  const runRender = useCallback(() => {
    void ensureMermaidRender(code, theme, preset, contrastMode)
  }, [code, contrastMode, key, preset, theme])

  useProgressiveMarkdownWork({
    eager: blocksStructuralReveal,
    isReady: entry !== undefined,
    run: runRender,
    targetRef: figureRef,
    workKey: key,
  })

  if (entry?.kind === 'error') {
    return (
      <figure
        className="my-4 overflow-hidden rounded-md border border-warning/40 bg-warning-subtle/30"
        ref={figureRef}
      >
        <div className="flex items-start gap-2 border-b border-warning/30 px-3 py-2">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-warning" />
          <span className="min-w-0 break-words t-meta text-foreground/90">
            {t.markdown.mermaidError.replace('{message}', entry.message)}
          </span>
        </div>
        <pre className="max-w-full overflow-x-auto p-3 font-mono text-xs text-muted-foreground">
          <code>{code}</code>
        </pre>
      </figure>
    )
  }
  // Successful diagrams are intentionally unboxed and keep Mermaid's native
  // responsive max-width. Wide figures shrink to the reading column, while
  // compact figures retain their intended type scale instead of being enlarged.
  // Error output remains boxed because it is an alert.
  const actions: MarkdownBlockAction[] = entry
    ? [
      {
        icon: Maximize2,
        id: 'expand',
        labels: {
          error: t.markdown.actionFailed,
          idle: t.markdown.mermaidExpand,
          pending: t.markdown.actionWorking,
          success: t.markdown.mermaidExpand,
        },
        onClick: () => setPreviewOpen(true),
        status: 'idle',
      },
      {
        icon: Copy,
        id: 'copy',
        labels: {
          error: t.markdown.actionFailed,
          idle: t.markdown.mermaidCopy,
          pending: t.markdown.actionWorking,
          success: t.markdown.mermaidCopied,
        },
        onClick: () => {
          void copyAction.run(
            () => copyMarkdownBlockText(code),
            'Inqtrix Mermaid source copy failed.',
          )
        },
        status: copyAction.status,
      },
      {
        icon: Download,
        id: 'png',
        labels: {
          error: t.markdown.actionFailed,
          idle: t.markdown.mermaidSavePng,
          pending: t.markdown.actionWorking,
          success: t.markdown.pngSaved,
        },
        onClick: () => {
          void pngAction.run(async () => {
            if (!diagramRef.current) {
              throw new Error('Rendered Mermaid element is unavailable.')
            }
            await downloadMarkdownBlockPng(diagramRef.current, MARKDOWN_BLOCK_FILE_NAMES.diagramPng)
          }, 'Inqtrix Mermaid PNG export failed.')
        },
        status: pngAction.status,
      },
    ]
    : []

  return (
    <figure
      className={cn(
        'my-4 w-full overflow-visible',
        !entry && 'min-h-24',
      )}
      ref={figureRef}
    >
      <MarkdownBlockFrame actions={actions}>
        {entry ? (
          <div className="w-full overflow-x-auto">
            <div
              className={cn('mx-auto max-w-full', naturalWidth && 'w-full')}
              ref={diagramRef}
              style={naturalWidth ? { maxWidth: `${naturalWidth}px` } : undefined}
            >
              <div
                className={cn(
                  'inqtrix-mermaid [&_svg]:mx-auto [&_svg]:block [&_svg]:!h-auto',
                  naturalWidth && '[&_svg]:!w-full [&_svg]:!max-w-none',
                )}
                // Mermaid's own strict-mode SVG output (sanitized labels, no
                // scripts); the sandbox contract lives in ensureMermaidRender.
                dangerouslySetInnerHTML={{ __html: entry.svg }}
              />
            </div>
          </div>
        ) : (
          <p className="px-1 py-6 t-meta text-muted-foreground sm:px-3">
            {t.markdown.mermaidPending}
          </p>
        )}
        {entry ? (
          <Dialog
            className="flex max-h-[calc(100svh-4rem)] max-w-[calc(100vw-2rem)] flex-col"
            closeLabel={t.common.close}
            contentClassName="min-h-0 overflow-auto p-4"
            contentProps={{
              'aria-label': t.markdown.mermaidDialogTitle,
              role: 'region',
              tabIndex: 0,
            }}
            onClose={() => setPreviewOpen(false)}
            open={previewOpen}
            title={t.markdown.mermaidDialogTitle}
          >
            <div
              className={cn('mx-auto max-w-full', previewMaxWidth && 'w-full')}
              style={previewMaxWidth ? { maxWidth: `${previewMaxWidth}px` } : undefined}
            >
              <div
                className={cn(
                  'inqtrix-mermaid [&_svg]:mx-auto [&_svg]:block [&_svg]:!h-auto',
                  previewMaxWidth && '[&_svg]:!w-full [&_svg]:!max-w-none',
                )}
                dangerouslySetInnerHTML={{ __html: entry.svg }}
              />
            </div>
          </Dialog>
        ) : null}
      </MarkdownBlockFrame>
    </figure>
  )
}
