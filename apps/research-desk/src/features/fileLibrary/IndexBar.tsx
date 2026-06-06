import { AlertTriangle, ChevronDown, Info, Layers, Link, RotateCcw } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import type { Locale } from '@/i18n/translations'
import { cn } from '@/lib/utils'
import { EMBED_MODELS, type EmbedModelId, type VectorIndexRecord, type VectorIndexStatus } from '@/features/project/types'
import type { VectorIndexMemberResolved } from '@/features/project/selectors'
import { ConfirmDelete, InlineText } from './controls'
import { indexVectorCount } from './helpers'

const STATUS_STYLES: Record<VectorIndexStatus, { badge: string; dot: string; pulse: boolean }> = {
  ready: { badge: 'border-success/25 bg-success-subtle text-success', dot: 'bg-success', pulse: false },
  indexing: { badge: 'border-brand/25 bg-brand-subtle text-brand', dot: 'bg-brand', pulse: true },
  stale: { badge: 'border-warning/25 bg-warning-subtle text-warning', dot: 'bg-warning', pulse: false },
}

function formatUpdated(iso: string, locale: Locale): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleDateString(locale, { day: '2-digit', month: 'short' })
}

export function IndexBar({
  index,
  members,
  onDelete,
  onModel,
  onReindex,
  onRename,
}: {
  index: VectorIndexRecord
  members: VectorIndexMemberResolved[]
  onDelete: (indexId: string) => void
  onModel: (indexId: string, model: EmbedModelId) => void
  onReindex: (indexId: string) => void
  onRename: (indexId: string, title: string) => void
}) {
  const { locale, t } = useLocale()
  const indexing = index.status === 'indexing'
  const stale = index.status === 'stale'
  const style = STATUS_STYLES[index.status]
  const statusLabel =
    index.status === 'ready'
      ? t.vectorIndex.statusReady
      : index.status === 'indexing'
        ? t.vectorIndex.statusIndexing
        : t.vectorIndex.statusStale
  const stats: [string, string][] = [
    [t.vectorIndex.dimensions, index.dims.toLocaleString(locale)],
    [t.vectorIndex.vectors, indexVectorCount(members).toLocaleString(locale)],
    [t.vectorIndex.documents, members.length.toLocaleString(locale)],
    [t.vectorIndex.updated, formatUpdated(index.updatedAt, locale)],
  ]

  return (
    <div className="rounded-lg border border-border bg-card p-3.5 shadow-[0_1px_2px_var(--shadow-hairline)]">
      <div className="flex flex-wrap items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-lg border border-file/25 bg-file-subtle text-file">
          <Layers className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <InlineText
              ariaLabel={t.vectorIndex.rename}
              className="text-sm font-semibold text-foreground"
              onCommit={(title) => onRename(index.id, title)}
              value={index.title}
            />
            <span className={cn('inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-semibold', style.badge)}>
              <span className={cn('size-1.5 rounded-full', style.dot, style.pulse && 'inqtrix-running-dot')} />
              {statusLabel}
            </span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex shrink-0 cursor-help items-center gap-1 whitespace-nowrap rounded border border-border bg-surface px-1.5 py-0.5 font-mono">
                  <Link className="size-3" />@index:{index.handle}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top">{t.vectorIndex.indexHandleTooltip}</TooltipContent>
            </Tooltip>
            <span className="hidden sm:inline">{t.vectorIndex.referenceHint}</span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button className="h-9 gap-1.5" disabled={indexing} onClick={() => onReindex(index.id)} size="sm" type="button" variant="outline">
            <RotateCcw className={cn('size-4', indexing && 'motion-safe:animate-spin')} />
            {indexing ? t.vectorIndex.reindexing : t.vectorIndex.reindex}
          </Button>
          <ConfirmDelete ariaLabel={t.vectorIndex.remove} hint={t.vectorIndex.removeHint} label={t.vectorIndex.remove} onConfirm={() => onDelete(index.id)} />
        </div>
      </div>

      {indexing ? (
        <div className="mt-3 border-t border-border/70 pt-3">
          <div className="mb-2 flex items-center justify-between gap-2 text-[11px]">
            <span className="min-w-0 truncate font-medium text-foreground">{t.vectorIndex.progressLabel}</span>
            <span className="shrink-0 font-mono text-muted-foreground">{index.model}</span>
          </div>
          <span className="relative block h-1.5 overflow-hidden rounded-full bg-brand/15">
            <span className="inqtrix-segment-breathe absolute inset-0 rounded-full bg-brand" />
          </span>
        </div>
      ) : (
        <div className="mt-3 flex flex-wrap items-end gap-x-6 gap-y-2 border-t border-border/70 pt-3">
          <label className="min-w-0">
            <span className="block text-[10px] font-semibold uppercase tracking-wide text-muted-foreground/80">{t.vectorIndex.embeddingModel}</span>
            <div className="relative mt-1">
              <select
                aria-label={t.vectorIndex.embeddingModel}
                className="h-7 cursor-pointer appearance-none rounded-md border border-border bg-background pl-2 pr-7 font-mono text-[12px] font-semibold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                onChange={(event) => onModel(index.id, event.target.value as EmbedModelId)}
                value={index.model}
              >
                {EMBED_MODELS.map((model) => (
                  <option key={model.id} value={model.id}>{model.label}</option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2 top-1/2 size-3 -translate-y-1/2 text-muted-foreground" />
            </div>
          </label>
          {stats.map(([label, value]) => (
            <div className="min-w-0" key={label}>
              <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground/80">{label}</p>
              <p className="mt-0.5 truncate text-[13px] font-semibold tabular-nums text-foreground">{value}</p>
            </div>
          ))}
        </div>
      )}

      {stale && !indexing ? (
        <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 rounded-md border border-warning/25 bg-warning-subtle px-2.5 py-1.5">
          <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-warning">
            <AlertTriangle className="size-3.5" />
            {t.vectorIndex.staleWarning}
          </span>
          <button
            className="shrink-0 rounded-md border border-warning/30 bg-card px-2 py-1 text-[11px] font-semibold text-warning hover:bg-warning-subtle"
            onClick={() => onReindex(index.id)}
            type="button"
          >
            {t.vectorIndex.staleAction}
          </button>
        </div>
      ) : null}

      <p className="mt-2.5 inline-flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <Info className="size-3 shrink-0" />
        {t.vectorIndex.simulationNote}
      </p>
    </div>
  )
}
