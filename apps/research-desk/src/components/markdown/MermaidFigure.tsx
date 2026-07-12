import { useEffect, useState } from 'react'

import { AlertTriangle } from '@/components/icons'
import { useLocale } from '@/i18n/LocaleProvider'
import { useTheme } from '@/theme/ThemeProvider'
import { cn } from '@/lib/utils'

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

const mermaidCache = new Map<string, MermaidCacheEntry>()
const mermaidPending = new Set<string>()
const mermaidListeners = new Map<string, Set<() => void>>()

let mermaidCounter = 0

function cacheKey(code: string, theme: MermaidTheme): string {
  return `${theme}\0${code}`
}

function subscribe(key: string, listener: () => void): () => void {
  const listeners = mermaidListeners.get(key) ?? new Set()
  listeners.add(listener)
  mermaidListeners.set(key, listeners)
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0) mermaidListeners.delete(key)
  }
}

function notify(key: string) {
  for (const listener of mermaidListeners.get(key) ?? []) listener()
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
): Promise<void> {
  const key = cacheKey(code, theme)
  if (mermaidCache.has(key) || mermaidPending.has(key)) return
  mermaidPending.add(key)
  try {
    const mermaid = (await import('mermaid')).default
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      htmlLabels: false,
      theme: 'base',
      themeVariables: themeVariablesFor(theme),
      flowchart: {
        curve: 'basis',
        nodeSpacing: 44,
        padding: 4,
        rankSpacing: 52,
      },
    })
    mermaidCounter += 1
    const { svg } = await mermaid.render(
      `inqtrix-mermaid-${mermaidCounter}`,
      code,
    )
    mermaidCache.set(key, { kind: 'svg', svg })
  } catch (error) {
    // Loud, never silent: the figure shows the parser message + source.
    const message = error instanceof Error ? error.message : String(error)
    console.warn('Inqtrix mermaid render failed.', error)
    mermaidCache.set(key, { kind: 'error', message })
  } finally {
    mermaidPending.delete(key)
    notify(key)
  }
}

export function MermaidFigure({ code }: { code: string }) {
  const { t } = useLocale()
  const { resolvedTheme } = useTheme()
  const theme: MermaidTheme = resolvedTheme === 'dark' ? 'dark' : 'light'
  const key = cacheKey(code, theme)
  const [, setVersion] = useState(0)

  useEffect(() => {
    if (mermaidCache.has(key)) return undefined
    const unsubscribe = subscribe(key, () => {
      setVersion((version) => version + 1)
    })
    void ensureMermaidRender(code, theme)
    return unsubscribe
  }, [code, key, theme])

  const entry = mermaidCache.get(key)
  if (entry?.kind === 'error') {
    return (
      <figure className="overflow-hidden rounded-md border border-warning/40 bg-warning-subtle/30">
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
  // Successful diagrams are intentionally unboxed: the flow uses the full
  // reading width and belongs to the document rather than looking like an
  // embedded widget. Error output remains boxed because it is an alert.
  return (
    <figure
      className={cn(
        'my-4 w-full overflow-visible',
        !entry && 'min-h-24',
      )}
    >
      {entry ? (
        <div
          className="inqtrix-mermaid w-full overflow-x-auto px-1 py-4 sm:px-3 [&_svg]:mx-auto [&_svg]:h-auto [&_svg]:max-w-full"
          // Mermaid's own strict-mode SVG output (sanitized labels, no
          // scripts); the sandbox contract lives in ensureMermaidRender.
          dangerouslySetInnerHTML={{ __html: entry.svg }}
        />
      ) : (
        <p className="px-1 py-6 t-meta text-muted-foreground sm:px-3">
          {t.markdown.mermaidPending}
        </p>
      )}
    </figure>
  )
}
