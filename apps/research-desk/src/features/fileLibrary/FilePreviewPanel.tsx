import { useCallback, useEffect, useState } from 'react'
import { FileText, X } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { MarkdownRenderer } from '@/components/markdown/MarkdownRenderer'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { fetchServerFileContent, getAsset, type ClientOptions } from '@/api/inqtrixClient'
import type { FileAssetRecord } from '@/features/project/types'
import { OriginalFileTab } from '@/features/files/OriginalFileTab'

type Tab = 'markdown' | 'original'

type BodyState =
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'ready'; text: string }

/**
 * Right-hand overlay that previews one library file: the transformed Markdown
 * (rendered with the app's report renderer, the same styling as chat/reports)
 * and — when the original is server-managed — the original PDF via react-pdf.
 *
 * Same overlay idiom as the knowledge `DocumentViewer`. The Markdown body comes
 * from the local record when present (client-parsed or already loaded) and is
 * otherwise fetched on open (`getAsset`, the load-on-use pattern). The Original
 * tab needs a connected server (`serverFileId` + options); it is disabled for
 * local-only assets.
 */
export function FilePreviewPanel({
  asset,
  onClose,
  options,
}: {
  asset: FileAssetRecord
  onClose: () => void
  /** Server client options, or null when no server is connected (local-only). */
  options: ClientOptions | null
}) {
  const { t } = useLocale()
  const [tab, setTab] = useState<Tab>('markdown')
  const [originalOpened, setOriginalOpened] = useState(false)

  const localText = asset.extractedText
  const [body, setBody] = useState<BodyState>(
    localText.trim() ? { kind: 'ready', text: localText } : { kind: 'loading' },
  )

  useEffect(() => {
    if (localText.trim()) {
      setBody({ kind: 'ready', text: localText })
      return undefined
    }
    if (!options) {
      // Local-only asset with no server to fetch from: nothing more to show.
      setBody({ kind: 'ready', text: '' })
      return undefined
    }
    let cancelled = false
    setBody({ kind: 'loading' })
    getAsset(asset.id, options)
      .then((detail) => {
        if (!cancelled) setBody({ kind: 'ready', text: detail.extracted_text ?? '' })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setBody({
            kind: 'error',
            message: error instanceof Error ? error.message : t.filePreview.markdownError,
          })
        }
      })
    return () => {
      cancelled = true
    }
  }, [asset.id, localText, options, t])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const canShowOriginal = Boolean(asset.serverFileId && options)
  const serverFileId = asset.serverFileId
  const loadOriginal = useCallback((): Promise<{ blob: Blob; contentType: string }> => {
    if (!serverFileId || !options) {
      return Promise.reject(new Error(t.filePreview.originalUnavailable))
    }
    return fetchServerFileContent(serverFileId, options)
  }, [options, serverFileId, t])

  function switchTab(next: Tab) {
    setTab(next)
    if (next === 'original') setOriginalOpened(true)
  }

  const title = asset.label || asset.fileName || t.filePreview.title

  return (
    <div aria-modal="true" className="fixed inset-0 z-50" role="dialog">
      <ResizablePanelGroup className="h-full w-full" orientation="horizontal">
        <ResizablePanel defaultSize="58%" minSize="20%">
          <button
            aria-label={t.filePreview.close}
            className="h-full w-full cursor-default bg-background/60 backdrop-blur-sm"
            onClick={onClose}
            type="button"
          />
        </ResizablePanel>
        <ResizableHandle aria-label={t.filePreview.resize} />
        <ResizablePanel
          className="min-h-0 min-w-0 overflow-hidden"
          defaultSize="42%"
          maxSize="80%"
          minSize="30%"
        >
          <aside className="flex h-full w-full flex-col border-l border-border bg-background shadow-[0_8px_28px_-12px_var(--shadow-soft)]">
            <header className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-3">
          <FileText className="icon-md shrink-0 text-foreground/80" />
          <div className="min-w-0 flex-1">
            <h2 className="truncate t-section text-foreground">{title}</h2>
            <p className="truncate t-meta text-muted-foreground">{asset.fileName}</p>
          </div>

          <div className="grid h-7 shrink-0 grid-cols-2 rounded-md bg-surface p-0.5">
            {(['markdown', 'original'] as const).map((tabKey) => {
              const isActive = tab === tabKey
              const isDisabled = tabKey === 'original' && !canShowOriginal
              return (
                <button
                  aria-pressed={isActive}
                  className={cn(
                    'inline-flex items-center justify-center rounded px-2 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
                    isActive
                      ? 'bg-background text-foreground shadow-[0_1px_2px_var(--shadow-hairline)]'
                      : 'text-muted-foreground hover:text-foreground',
                  )}
                  disabled={isDisabled}
                  key={tabKey}
                  onClick={() => switchTab(tabKey)}
                  title={isDisabled ? t.filePreview.originalUnavailable : undefined}
                  type="button"
                >
                  {tabKey === 'markdown' ? t.filePreview.tabMarkdown : t.filePreview.tabOriginal}
                </button>
              )
            })}
          </div>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={t.filePreview.close}
                className="size-7 shrink-0 text-muted-foreground hover:text-foreground"
                onClick={onClose}
                size="icon"
                type="button"
                variant="ghost"
              >
                <X className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.filePreview.close}</TooltipContent>
          </Tooltip>
        </header>

            <ScrollArea className={cn('min-h-0 flex-1', tab !== 'markdown' && 'hidden')}>
              <div className="w-full px-6 py-6">
                {body.kind === 'loading' && (
                  <p className="t-hint text-muted-foreground">{t.filePreview.markdownLoading}</p>
                )}
                {body.kind === 'error' && (
                  <p className="t-meta text-destructive">{body.message}</p>
                )}
                {body.kind === 'ready' && (
                  body.text.trim() ? (
                    <div className="report-markdown w-full min-w-0 max-w-full [overflow-wrap:anywhere]">
                      <MarkdownRenderer markdown={body.text} variant="report" />
                    </div>
                  ) : (
                    <div className="space-y-1">
                      <p className="t-hint text-muted-foreground">{t.filePreview.markdownEmpty}</p>
                      {asset.parseWarning ? (
                        <p className="t-meta text-warning">{asset.parseWarning}</p>
                      ) : null}
                    </div>
                  )
                )}
              </div>
            </ScrollArea>

            {originalOpened && canShowOriginal && (
              <div className={cn('flex min-h-0 flex-1 flex-col', tab !== 'original' && 'hidden')}>
                <OriginalFileTab fileName={asset.fileName} load={loadOriginal} />
              </div>
            )}
          </aside>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}
