import { Check, Copy } from '@/components/icons'
import {
  isValidElement,
  memo,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ComponentPropsWithoutRef,
  type ReactNode,
} from 'react'
import Markdown, { MarkdownHooks, type Components } from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import rehypePrettyCode, { type Options as RehypePrettyCodeOptions } from 'rehype-pretty-code'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import type { Highlighter, ThemedToken } from 'shiki'
import { createBundledHighlighter } from 'shiki/core'
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript'
import type { PluggableList } from 'unified'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { useTheme } from '@/theme/ThemeProvider'
import {
  extractMarkdownCodeBlocks,
  plainCodeLanguageFromClassName,
} from './markdownLanguage'

export type MarkdownRendererVariant = 'chat' | 'report'

type MarkdownRendererProps = {
  isStreaming?: boolean
  markdown: string
  variant: MarkdownRendererVariant
}

type StreamingMarkdownPendingKind = 'code' | 'math' | null

const createMarkdownHighlighter = createBundledHighlighter({
  engine: () => createJavaScriptRegexEngine(),
  langs: {
    bash: () => import('shiki/dist/langs/shellscript.mjs'),
    css: () => import('shiki/dist/langs/css.mjs'),
    html: () => import('shiki/dist/langs/html.mjs'),
    javascript: () => import('shiki/dist/langs/javascript.mjs'),
    json: () => import('shiki/dist/langs/json.mjs'),
    jsonc: () => import('shiki/dist/langs/jsonc.mjs'),
    jsx: () => import('shiki/dist/langs/jsx.mjs'),
    markdown: () => import('shiki/dist/langs/markdown.mjs'),
    python: () => import('shiki/dist/langs/python.mjs'),
    sh: () => import('shiki/dist/langs/shellscript.mjs'),
    shellscript: () => import('shiki/dist/langs/shellscript.mjs'),
    tsx: () => import('shiki/dist/langs/tsx.mjs'),
    typescript: () => import('shiki/dist/langs/typescript.mjs'),
  },
  themes: {
    'github-dark': () => import('shiki/dist/themes/github-dark.mjs'),
    'github-light': () => import('shiki/dist/themes/github-light.mjs'),
  },
})

type MarkdownHighlighterOptions = Parameters<typeof createMarkdownHighlighter>[0]
type MarkdownLoadLanguage = Parameters<Highlighter['loadLanguage']>[0]
type MarkdownTokenizeLanguage = NonNullable<Parameters<Highlighter['codeToTokens']>[1]['lang']>
type MarkdownCodeTheme = 'github-dark' | 'github-light'
type MarkdownHighlightedLine = Array<Pick<ThemedToken, 'color' | 'content' | 'fontStyle'>>

let markdownHighlighterPromise: Promise<Highlighter> | null = null
const markdownTokenCache = new Map<string, MarkdownHighlightedLine[]>()
const markdownTokenPending = new Set<string>()
const markdownTokenListeners = new Map<string, Set<() => void>>()

function getMarkdownHighlighter(options: MarkdownHighlighterOptions): Promise<Highlighter> {
  markdownHighlighterPromise ??= createMarkdownHighlighter(options)
    .then((highlighter) => highlighter as unknown as Highlighter)
  return markdownHighlighterPromise
}

export function preloadMarkdownCodeHighlights(
  markdowns: readonly string[],
  resolvedTheme: 'light' | 'dark',
): Promise<void> {
  const theme = markdownCodeTheme(resolvedTheme)
  const jobs = markdowns.flatMap((markdown) =>
    extractMarkdownCodeBlocks(markdown).map((block) =>
      ensureMarkdownCodeHighlight({
        code: block.code,
        language: block.language,
        theme,
      }),
    ),
  )
  return Promise.allSettled(jobs).then(() => undefined)
}

const PRETTY_CODE_OPTIONS: RehypePrettyCodeOptions = {
  bypassInlineCode: true,
  defaultLang: {
    block: 'plaintext',
    inline: 'plaintext',
  },
  getHighlighter: async (options) => {
    return getMarkdownHighlighter(options)
  },
  keepBackground: false,
  theme: {
    dark: 'github-dark',
    light: 'github-light',
  },
}

const MARKDOWN_REMARK_PLUGINS: PluggableList = [remarkGfm, remarkMath]
const MARKDOWN_FALLBACK_REHYPE_PLUGINS: PluggableList = [rehypeKatex]
const MARKDOWN_REHYPE_PLUGINS: PluggableList = [
  rehypeKatex,
  [rehypePrettyCode, PRETTY_CODE_OPTIONS],
]

const MARKDOWN_COMPONENTS_BY_VARIANT: Record<MarkdownRendererVariant, Components> = {
  chat: {
    a: ({ className, ...props }) => (
      <MarkdownLink className={className} variant="chat" {...props} />
    ),
    blockquote: ({ className, ...props }) => (
      <blockquote
        className={cn('border-l-2 border-brand/35 bg-brand-subtle/55 px-3 py-2 text-sm leading-[1.45]', className)}
        {...props}
      />
    ),
    code: InlineCode,
    figure: ({ className, ...props }) => (
      <figure className={cn('my-3 min-w-0', className)} {...props} />
    ),
    h1: ({ className, ...props }) => (
      <h1
        className={cn(
          'mb-3 mt-5 break-words border-b border-border pb-2 text-lg font-semibold leading-7 text-foreground [overflow-wrap:anywhere] first:mt-0',
          className,
        )}
        {...props}
      />
    ),
    h2: ({ className, ...props }) => (
      <h2
        className={cn(
          'mb-2.5 mt-5 break-words border-b border-border pb-1.5 text-base font-semibold leading-7 text-foreground [overflow-wrap:anywhere] first:mt-0',
          className,
        )}
        {...props}
      />
    ),
    h3: ({ className, ...props }) => (
      <h3
        className={cn(
          'mb-2 mt-4 break-words text-[15px] font-semibold leading-6 text-foreground [overflow-wrap:anywhere] first:mt-0',
          className,
        )}
        {...props}
      />
    ),
    h4: ({ className, ...props }) => (
      <h4
        className={cn(
          'mb-1.5 mt-3 break-words text-sm font-semibold leading-6 text-foreground [overflow-wrap:anywhere] first:mt-0',
          className,
        )}
        {...props}
      />
    ),
    h5: ({ className, ...props }) => (
      <h5
        className={cn(
          'mb-1.5 mt-3 break-words text-xs font-semibold uppercase tracking-[0.02em] text-muted-foreground [overflow-wrap:anywhere] first:mt-0',
          className,
        )}
        {...props}
      />
    ),
    h6: ({ className, ...props }) => (
      <h6
        className={cn(
          'mb-1 mt-2 break-words text-xs font-semibold text-muted-foreground [overflow-wrap:anywhere] first:mt-0',
          className,
        )}
        {...props}
      />
    ),
    li: ({ className, ...props }) => (
      <li className={cn('break-words pl-1 leading-[1.45]', className)} {...props} />
    ),
    ol: ({ className, ...props }) => (
      <ol className={cn('my-3 list-decimal space-y-1 pl-5 leading-[1.45]', className)} {...props} />
    ),
    p: ({ className, ...props }) => (
      <p className={cn('my-2 break-words leading-[1.42] [overflow-wrap:anywhere] first:mt-0 last:mb-0', className)} {...props} />
    ),
    pre: (props) => <PrettyCodePre {...props} variant="chat" />,
    span: MarkdownSpan,
    strong: ({ className, ...props }) => (
      <strong className={cn('font-semibold text-foreground', className)} {...props} />
    ),
    table: ({ className, ...props }) => (
      <div className="my-3 max-w-full overflow-x-auto rounded-md border border-border bg-background [scrollbar-width:thin]">
        <table className={cn('w-full min-w-[32rem] border-collapse text-left text-xs leading-[1.4]', className)} {...props} />
      </div>
    ),
    tbody: ({ className, ...props }) => (
      <tbody className={cn('divide-y divide-border', className)} {...props} />
    ),
    td: ({ className, ...props }) => (
      <td className={cn('border-r border-border px-3 py-2 align-top last:border-r-0', className)} {...props} />
    ),
    th: ({ className, ...props }) => (
      <th className={cn('border-r border-border bg-surface px-3 py-2 align-top font-semibold text-foreground last:border-r-0', className)} {...props} />
    ),
    thead: ({ className, ...props }) => (
      <thead className={cn('border-b border-border', className)} {...props} />
    ),
    tr: ({ className, ...props }) => (
      <tr className={cn('hover:bg-surface/60', className)} {...props} />
    ),
    ul: ({ className, ...props }) => (
      <ul className={cn('my-3 list-disc space-y-1 pl-5 leading-[1.45]', className)} {...props} />
    ),
  },
  report: {
    a: ({ className, ...props }) => (
      <MarkdownLink className={className} variant="report" {...props} />
    ),
    blockquote: ({ className, ...props }) => (
      <blockquote
        className={cn(
          'border-l-2 border-brand/35 bg-brand-subtle/55 px-4 py-3 text-sm leading-7 text-foreground',
          className,
        )}
        {...props}
      />
    ),
    code: InlineCode,
    figure: ({ className, ...props }) => (
      <figure className={cn('my-5 min-w-0', className)} {...props} />
    ),
    h1: ({ className, ...props }) => (
      <h1
        className={cn(
          'mb-4 mt-8 break-words border-b border-border pb-2 text-2xl font-semibold leading-9 text-foreground [overflow-wrap:anywhere] first:mt-0',
          className,
        )}
        {...props}
      />
    ),
    h2: ({ className, ...props }) => (
      <h2
        className={cn(
          'mt-8 break-words border-b border-border pb-2 text-xl font-semibold leading-8 text-foreground [overflow-wrap:anywhere] first:mt-0',
          className,
        )}
        {...props}
      />
    ),
    h3: ({ className, ...props }) => (
      <h3
        className={cn(
          'mt-6 break-words text-base font-semibold leading-7 text-foreground [overflow-wrap:anywhere]',
          className,
        )}
        {...props}
      />
    ),
    h4: ({ className, ...props }) => (
      <h4 className={cn('mt-5 break-words text-sm font-semibold leading-7 text-foreground', className)} {...props} />
    ),
    h5: ({ className, ...props }) => (
      <h5 className={cn('mt-4 break-words text-xs font-semibold uppercase text-muted-foreground', className)} {...props} />
    ),
    h6: ({ className, ...props }) => (
      <h6 className={cn('mt-4 break-words text-xs font-semibold text-muted-foreground', className)} {...props} />
    ),
    hr: ({ className, ...props }) => (
      <hr className={cn('my-8 border-border', className)} {...props} />
    ),
    li: ({ className, ...props }) => (
      <li className={cn('break-words pl-1 leading-7 [overflow-wrap:anywhere]', className)} {...props} />
    ),
    ol: ({ className, ...props }) => (
      <ol className={cn('my-4 list-decimal space-y-1 pl-5 text-sm', className)} {...props} />
    ),
    p: ({ className, ...props }) => (
      <p className={cn('my-4 break-words text-sm leading-7 text-foreground [overflow-wrap:anywhere]', className)} {...props} />
    ),
    pre: (props) => <PrettyCodePre {...props} variant="report" />,
    span: MarkdownSpan,
    strong: ({ className, ...props }) => (
      <strong className={cn('font-semibold text-foreground', className)} {...props} />
    ),
    table: ({ className, ...props }) => (
      <div className="my-5 max-w-full overflow-x-auto rounded-lg border border-border [scrollbar-width:thin]">
        <table className={cn('w-full min-w-[560px] border-collapse text-left text-sm', className)} {...props} />
      </div>
    ),
    tbody: ({ className, ...props }) => (
      <tbody className={cn('divide-y divide-border', className)} {...props} />
    ),
    td: ({ className, ...props }) => (
      <td className={cn('border-r border-border px-3 py-2 align-top last:border-r-0', className)} {...props} />
    ),
    th: ({ className, ...props }) => (
      <th
        className={cn(
          'border-r border-border bg-surface px-3 py-2 text-xs font-semibold uppercase text-muted-foreground last:border-r-0',
          className,
        )}
        {...props}
      />
    ),
    thead: ({ className, ...props }) => (
      <thead className={cn('border-b border-border', className)} {...props} />
    ),
    tr: ({ className, ...props }) => (
      <tr className={cn('hover:bg-surface/60', className)} {...props} />
    ),
    ul: ({ className, ...props }) => (
      <ul className={cn('my-4 list-disc space-y-1 pl-5 text-sm', className)} {...props} />
    ),
  },
}

export const MarkdownRenderer = memo(function MarkdownRenderer({
  isStreaming = false,
  markdown,
  variant,
}: MarkdownRendererProps) {
  const { t } = useLocale()
  const normalizedMarkdown = normalizeLatex(markdown)
  const streamParts = isStreaming
    ? splitStreamingMarkdown(normalizedMarkdown)
    : { pendingKind: null, pendingText: '', stableMarkdown: normalizedMarkdown }

  return (
    <ErrorBoundary
      title={t.markdownError.title}
      retryLabel={t.markdownError.retry}
    >
      {streamParts.stableMarkdown && variant === 'chat' && (
        <SynchronousMarkdown markdown={streamParts.stableMarkdown} variant={variant} />
      )}
      {streamParts.stableMarkdown && variant !== 'chat' && (
        <MarkdownHooks
          components={MARKDOWN_COMPONENTS_BY_VARIANT[variant]}
          fallback={<SynchronousMarkdown markdown={streamParts.stableMarkdown} variant={variant} />}
          rehypePlugins={MARKDOWN_REHYPE_PLUGINS}
          remarkPlugins={MARKDOWN_REMARK_PLUGINS}
          skipHtml
        >
          {streamParts.stableMarkdown}
        </MarkdownHooks>
      )}
      {streamParts.pendingText && (
        <StreamingMarkdownTail
          kind={streamParts.pendingKind}
          text={streamParts.pendingText}
          variant={variant}
        />
      )}
    </ErrorBoundary>
  )
})

function MarkdownLink({
  className,
  href,
  node,
  variant,
  ...props
}: ComponentPropsWithoutRef<'a'> & {
  node?: unknown
  variant: MarkdownRendererVariant
}) {
  void node

  const anchor = (
    <a
      {...props}
      className={cn(
        'break-words font-medium text-brand underline underline-offset-4 hover:text-foreground',
        variant === 'report' && '[overflow-wrap:anywhere]',
        className,
      )}
      href={href}
      rel={props.rel ?? 'noreferrer'}
      target={props.target ?? '_blank'}
    />
  )

  if (variant !== 'report' || !href) {
    return anchor
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {anchor}
      </TooltipTrigger>
      <TooltipContent
        align="center"
        className="max-w-[min(34rem,calc(100vw-2rem))] border border-border/40 bg-foreground px-2.5 py-1.5 font-mono text-[11px] leading-4 text-background shadow-[0_8px_24px_var(--shadow-soft)]"
        collisionPadding={12}
        side="top"
        sideOffset={6}
      >
        <span className="block max-w-full truncate">
          {href}
        </span>
      </TooltipContent>
    </Tooltip>
  )
}

function MarkdownSpan({
  node,
  style,
  ...props
}: ComponentPropsWithoutRef<'span'> & {
  node?: unknown
}) {
  const { resolvedTheme } = useTheme()
  void node

  const tokenColor = shikiTokenColor(style, resolvedTheme)
  return (
    <span
      {...props}
      style={tokenColor ? { ...style, color: tokenColor } : style}
    />
  )
}

function InlineCode({ children, className, ...props }: ComponentPropsWithoutRef<'code'>) {
  const isPrettyCodeBlock = 'data-theme' in props || 'data-language' in props
  const isPlainCodeBlock = plainCodeLanguageFromClassName(className) !== null
  return (
    <code
      className={cn(
        isPrettyCodeBlock || isPlainCodeBlock
          ? 'font-mono'
          : 'rounded bg-muted px-1 py-0.5 font-mono text-[0.85em] text-foreground',
        className,
      )}
      {...props}
    >
      {children}
    </code>
  )
}

function PrettyCodePre({
  children,
  className,
  node,
  variant,
  ...props
}: ComponentPropsWithoutRef<'pre'> & {
  node?: {
    properties?: Record<string, unknown>
  }
  variant: MarkdownRendererVariant
}) {
  const { t } = useLocale()
  const [copied, setCopied] = useState(false)
  const preRef = useRef<HTMLPreElement | null>(null)
  const dataProps = props as Record<string, unknown>
  const language = propToString(dataProps['data-language'] ?? node?.properties?.dataLanguage ?? node?.properties?.['data-language'])
    ?? codeLanguageFromReactNode(children)
    ?? 'text'
  const codeText = textFromReactNode(children)

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(readRenderedCodeText(preRef.current, codeText))
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1200)
    } catch (error) {
      console.warn('Inqtrix markdown code copy failed.', error)
    }
  }

  return (
    <div
      className={cn(
        'group/code overflow-hidden border border-border bg-muted/70 text-xs shadow-[0_1px_2px_var(--shadow-hairline)]',
        variant === 'report' ? 'rounded-lg' : 'rounded-md',
      )}
    >
      <div
        className={cn(
          'flex items-center justify-between gap-2 border-b border-border bg-surface px-3',
          variant === 'report' ? 'h-9' : 'h-8',
        )}
      >
        <span className="truncate font-mono text-[11px] font-semibold text-muted-foreground">
          {language}
        </span>
        <Button
          aria-label={copied ? t.chat.copiedCode : t.chat.copyCode}
          className={cn(
            'size-6 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover/code:opacity-100',
            copied && 'text-success opacity-100 hover:text-success',
          )}
          onClick={() => void copyCode()}
          size="icon"
          type="button"
          variant="ghost"
        >
          {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
        </Button>
      </div>
      <pre
        className={cn(
          'max-w-full overflow-x-auto bg-transparent font-mono text-foreground [scrollbar-width:thin]',
          variant === 'report' ? 'p-4 leading-6' : 'p-3 leading-[1.45]',
          className,
        )}
        {...props}
        ref={preRef}
      >
        <HighlightedCode code={codeText} language={language} />
      </pre>
    </div>
  )
}

function HighlightedCode({
  code,
  language,
}: {
  code: string
  language: string
}) {
  const { resolvedTheme } = useTheme()
  const theme = markdownCodeTheme(resolvedTheme)
  const normalizedLanguage = shikiCodeLanguage(language)
  const cacheKey = markdownCodeCacheKey(code, normalizedLanguage, theme)
  const [, setHighlightVersion] = useState(0)

  useEffect(() => {
    if (markdownTokenCache.has(cacheKey)) return undefined

    const unsubscribe = subscribeMarkdownCodeHighlight(cacheKey, () => {
      setHighlightVersion((version) => version + 1)
    })
    void ensureMarkdownCodeHighlight({
      code,
      language: normalizedLanguage,
      theme,
    })
    return unsubscribe
  }, [cacheKey, code, normalizedLanguage, theme])

  const lines = markdownTokenCache.get(cacheKey)
  if (!lines) {
    return <code className={`language-${normalizedLanguage}`}>{code}</code>
  }

  return (
    <code className={`language-${normalizedLanguage}`}>
      {lines.map((line, lineIndex) => (
        <span data-line="" key={lineIndex}>
          {line.map((token, tokenIndex) => (
            <span key={`${lineIndex}-${tokenIndex}`} style={markdownTokenStyle(token)}>
              {token.content}
            </span>
          ))}
          {lineIndex < lines.length - 1 ? '\n' : null}
        </span>
      ))}
    </code>
  )
}

function SynchronousMarkdown({
  markdown,
  variant,
}: {
  markdown: string
  variant: MarkdownRendererVariant
}) {
  return (
    <Markdown
      components={MARKDOWN_COMPONENTS_BY_VARIANT[variant]}
      rehypePlugins={MARKDOWN_FALLBACK_REHYPE_PLUGINS}
      remarkPlugins={MARKDOWN_REMARK_PLUGINS}
      skipHtml
    >
      {markdown}
    </Markdown>
  )
}

function StreamingMarkdownTail({
  kind,
  text,
  variant,
}: {
  kind: StreamingMarkdownPendingKind
  text: string
  variant: MarkdownRendererVariant
}) {
  if (kind === 'code') {
    const pendingCode = parsePendingCodeFence(text)
    return (
      <div
        className={cn(
          'my-3 overflow-hidden border border-border bg-muted/70 text-xs shadow-[0_1px_2px_var(--shadow-hairline)]',
          variant === 'report' ? 'rounded-lg' : 'rounded-md',
        )}
      >
        <div className="flex h-8 items-center border-b border-border bg-surface px-3">
          <span className="truncate font-mono text-[11px] font-semibold text-muted-foreground">
            {pendingCode.language}
          </span>
        </div>
        <pre className="max-w-full overflow-x-auto whitespace-pre-wrap bg-transparent p-3 font-mono leading-[1.45] text-foreground [overflow-wrap:anywhere] [scrollbar-width:thin]">
          {pendingCode.body}
        </pre>
      </div>
    )
  }

  return (
    <span className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]">
      {text}
    </span>
  )
}

function normalizeLatex(markdown: string) {
  return markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, expression: string) => (
      `\n\n$$\n${expression.trim()}\n$$\n\n`
    ))
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expression: string) => (
      `$${expression.trim()}$`
    ))
}

function splitStreamingMarkdown(markdown: string): {
  pendingKind: StreamingMarkdownPendingKind
  pendingText: string
  stableMarkdown: string
} {
  const pending = findFirstUnclosedStreamingConstruct(markdown)
  if (!pending) {
    return {
      pendingKind: null,
      pendingText: '',
      stableMarkdown: markdown,
    }
  }

  return {
    pendingKind: pending.kind,
    pendingText: markdown.slice(pending.start),
    stableMarkdown: markdown.slice(0, pending.start).trimEnd(),
  }
}

function findFirstUnclosedStreamingConstruct(markdown: string): {
  kind: Exclude<StreamingMarkdownPendingKind, null>
  start: number
} | null {
  const candidates: Array<{
    kind: Exclude<StreamingMarkdownPendingKind, null>
    start: number
  }> = []
  let codeFenceStart = -1
  let blockMathStart = -1
  let inlineMathStart = -1

  for (let index = 0; index < markdown.length; index += 1) {
    const isInsideMath = blockMathStart >= 0 || inlineMathStart >= 0
    const fenceMarkerIndex = isInsideMath ? null : codeFenceMarkerIndexAtLineStart(markdown, index)
    if (fenceMarkerIndex !== null) {
      codeFenceStart = codeFenceStart >= 0 ? -1 : index
      index = fenceMarkerIndex + 2
      continue
    }

    if (codeFenceStart >= 0 || isEscaped(markdown, index)) {
      continue
    }

    if (markdown.startsWith('$$', index)) {
      blockMathStart = blockMathStart >= 0 ? -1 : index
      index += 1
      continue
    }

    if (markdown[index] === '$' && markdown[index + 1] !== '$') {
      if (inlineMathStart >= 0 && isInlineMathCloseDelimiter(markdown, index)) {
        inlineMathStart = -1
      } else if (inlineMathStart < 0 && blockMathStart < 0 && isInlineMathOpenDelimiter(markdown, index)) {
        inlineMathStart = index
      }
    }
  }

  if (codeFenceStart >= 0) {
    candidates.push({ kind: 'code', start: codeFenceStart })
  }
  if (blockMathStart >= 0) {
    candidates.push({ kind: 'math', start: blockMathStart })
  }
  if (inlineMathStart >= 0) {
    candidates.push({ kind: 'math', start: inlineMathStart })
  }

  return candidates.sort((left, right) => left.start - right.start)[0] ?? null
}

function codeFenceMarkerIndexAtLineStart(markdown: string, index: number) {
  if (index > 0 && markdown[index - 1] !== '\n') return null

  let markerIndex = index
  while (markdown[markerIndex] === ' ' && markerIndex - index < 4) {
    markerIndex += 1
  }

  return markdown.startsWith('```', markerIndex) ? markerIndex : null
}

function isEscaped(text: string, index: number) {
  let slashCount = 0
  for (let current = index - 1; current >= 0 && text[current] === '\\'; current -= 1) {
    slashCount += 1
  }
  return slashCount % 2 === 1
}

function isInlineMathOpenDelimiter(markdown: string, index: number) {
  const next = markdown[index + 1]
  const previous = markdown[index - 1]
  if (!next || /\s/.test(next)) return false
  if (previous && /[A-Za-z0-9]/.test(previous)) return false
  return true
}

function isInlineMathCloseDelimiter(markdown: string, index: number) {
  const previous = markdown[index - 1]
  if (!previous || /\s/.test(previous)) return false
  return true
}

function parsePendingCodeFence(text: string) {
  const match = /^(?: {0,3})```([^\n`]*)\n?/.exec(text)
  if (!match) {
    return {
      body: text,
      language: 'text',
    }
  }

  const language = match[1].trim().split(/\s+/)[0] || 'text'
  return {
    body: text.slice(match[0].length),
    language,
  }
}

function markdownCodeTheme(resolvedTheme: 'light' | 'dark'): MarkdownCodeTheme {
  return resolvedTheme === 'dark' ? 'github-dark' : 'github-light'
}

function shikiCodeLanguage(language: string): string {
  return language === 'text' ? 'plaintext' : language
}

function markdownCodeCacheKey(code: string, language: string, theme: MarkdownCodeTheme): string {
  return `${theme}\u0000${language}\u0000${code}`
}

function subscribeMarkdownCodeHighlight(cacheKey: string, listener: () => void) {
  const listeners = markdownTokenListeners.get(cacheKey) ?? new Set<() => void>()
  listeners.add(listener)
  markdownTokenListeners.set(cacheKey, listeners)

  return () => {
    listeners.delete(listener)
    if (listeners.size === 0) markdownTokenListeners.delete(cacheKey)
  }
}

function notifyMarkdownCodeHighlight(cacheKey: string) {
  const listeners = markdownTokenListeners.get(cacheKey)
  if (!listeners) return
  for (const listener of listeners) listener()
}

async function ensureMarkdownCodeHighlight({
  code,
  language,
  theme,
}: {
  code: string
  language: string
  theme: MarkdownCodeTheme
}): Promise<void> {
  const normalizedLanguage = shikiCodeLanguage(language)
  const cacheKey = markdownCodeCacheKey(code, normalizedLanguage, theme)
  if (markdownTokenCache.has(cacheKey) || markdownTokenPending.has(cacheKey)) return

  markdownTokenPending.add(cacheKey)
  try {
    const highlighter = await getMarkdownHighlighter({
      langs: ['plaintext'],
      themes: ['github-dark', 'github-light'],
    })
    if (normalizedLanguage !== 'plaintext') {
      await highlighter.loadLanguage(normalizedLanguage as MarkdownLoadLanguage)
    }
    const result = highlighter.codeToTokens(code, {
      lang: normalizedLanguage as MarkdownTokenizeLanguage,
      theme,
      tokenizeMaxLineLength: 900,
      tokenizeTimeLimit: 200,
    })
    markdownTokenCache.set(
      cacheKey,
      result.tokens.map((line) =>
        line.map((token) => ({
          color: token.color,
          content: token.content,
          fontStyle: token.fontStyle,
        })),
      ),
    )
  } catch (error) {
    console.warn('Inqtrix markdown code highlight failed.', error)
  } finally {
    markdownTokenPending.delete(cacheKey)
    notifyMarkdownCodeHighlight(cacheKey)
  }
}

function markdownTokenStyle(token: Pick<ThemedToken, 'color' | 'fontStyle'>): CSSProperties | undefined {
  const style: CSSProperties = {}
  if (token.color) style.color = token.color

  if (typeof token.fontStyle === 'number' && token.fontStyle > 0) {
    if (token.fontStyle & 1) style.fontStyle = 'italic'
    if (token.fontStyle & 2) style.fontWeight = 700
    if (token.fontStyle & 4) style.textDecoration = 'underline'
  }

  return Object.keys(style).length > 0 ? style : undefined
}

function codeLanguageFromReactNode(node: ReactNode): string | null {
  if (Array.isArray(node)) {
    for (const child of node) {
      const language = codeLanguageFromReactNode(child)
      if (language) return language
    }
    return null
  }

  if (isValidElement<{ children?: ReactNode; className?: unknown }>(node)) {
    return plainCodeLanguageFromClassName(node.props.className)
      ?? codeLanguageFromReactNode(node.props.children)
  }

  return null
}

function propToString(value: unknown) {
  return typeof value === 'string' && value.trim() ? value : null
}

function shikiTokenColor(
  style: CSSProperties | undefined,
  resolvedTheme: 'light' | 'dark',
) {
  if (!style) return null

  const styleProperties = style as CSSProperties & Record<string, unknown>
  const light = propToString(styleProperties['--shiki-light'])
  const dark = propToString(styleProperties['--shiki-dark'])

  return resolvedTheme === 'dark'
    ? dark ?? light
    : light ?? dark
}

function textFromReactNode(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textFromReactNode).join('')
  if (isValidElement<{ children?: ReactNode }>(node)) {
    return textFromReactNode(node.props.children)
  }
  return ''
}

function readRenderedCodeText(pre: HTMLPreElement | null, fallback: ReactNode): string {
  const codeElement = pre?.querySelector('code')
  const renderedLines = codeElement
    ? Array.from(codeElement.querySelectorAll<HTMLElement>('[data-line]'))
    : []
  if (renderedLines.length > 0) {
    return renderedLines.map((line) => line.textContent ?? '').join('\n').replace(/\n$/, '')
  }

  return (codeElement?.textContent ?? textFromReactNode(fallback)).replace(/\n$/, '')
}
