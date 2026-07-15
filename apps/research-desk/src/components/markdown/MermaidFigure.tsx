import { useCallback, useRef, useState } from 'react'

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
import { BoundedLruCache, MARKDOWN_RENDER_CACHE_CAPACITY } from './boundedLruCache'
import { mermaidNaturalWidth, mermaidPreviewMaxWidth } from './mermaidSizing'
import { useMarkdownCacheEntry } from './useMarkdownCacheEntry'
import { useProgressiveMarkdownWork } from './useProgressiveMarkdownWork'

/**
 * Renders a ```mermaid fence as a diagram (plan M1 S6).
 *
 * Follows the EXACT `markdownTokenCache` discipline of the code
 * highlighter (gotcha #29): chat history re-renders stay synchronous —
 * a cached diagram paints its SVG immediately; an uncached one shows a
 * stable hull ("Diagramm wird erstellt …") while the async render fills
 * the module cache once and notifies subscribers. The mermaid bundle is
 * a lazy chunk (first diagram pays it, the main bundle never does).
 *
 * Security: `securityLevel: 'strict'` + `htmlLabels: false` — labels are
 * sanitized, no scripts/clicks, no raw HTML (matches the renderer's
 * `skipHtml`). Render failures are LOUD by design: a warning box with
 * the parser message plus the source, never a silently missing block.
 */

type MermaidTheme = 'light' | 'dark'

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

async function ensureMermaidRender(
  code: string,
  theme: MermaidTheme,
  preset: ThemePreset,
  contrastMode: ContrastMode,
): Promise<void> {
  const key = cacheKey(code, theme, preset, contrastMode)
  if (mermaidCache.has(key) || mermaidPending.has(key)) return
  mermaidPending.add(key)
  try {
    const themeVariables = themeVariablesFor(theme)
    const mermaid = (await import('mermaid')).default
    const { svg } = await serializeMermaidRender(async () => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        htmlLabels: false,
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
      mermaidCounter += 1
      return mermaid.render(`inqtrix-mermaid-${mermaidCounter}`, code)
    })
    mermaidCache.set(key, { kind: 'svg', svg })
  } catch (error) {
    // Loud, never silent: the figure shows the parser message + source.
    const message = error instanceof Error ? error.message : String(error)
    console.warn('Inqtrix mermaid render failed.', error)
    mermaidCache.set(key, { kind: 'error', message })
  } finally {
    mermaidPending.delete(key)
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

  const runRender = useCallback(() => {
    void ensureMermaidRender(code, theme, preset, contrastMode)
  }, [code, contrastMode, key, preset, theme])

  useProgressiveMarkdownWork({
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
