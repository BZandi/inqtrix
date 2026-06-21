import { useState } from 'react'
import { useReducedMotion } from 'motion/react'
import { AlertTriangle, ChevronDown, Clock3, Database, Info, Layers, Link, RotateCcw, Sparkles, XCircle } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { OptionMenuHeader, OptionMenuItem, optionMenuContentClassName } from '@/components/ui/option-menu'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import type { Locale, TranslationDictionary } from '@/i18n/translations'
import { cn } from '@/lib/utils'
import { formatDurationMsShort } from '@/lib/time'
import type {
  EmbedModelDescriptor,
  EmbedModelId,
  IndexingJobLive,
  VectorIndexRecord,
  VectorIndexRunHistoryEntry,
  VectorIndexRunResult,
  VectorIndexStatus,
} from '@/features/project/types'
import type { VectorIndexMemberResolved } from '@/features/project/selectors'
import type { EmbeddingQuota } from '@/features/quota/useEmbeddingQuota'
import { ConfirmDelete } from './controls'
import { formatAddedAt, indexVectorCount } from './helpers'

const STATUS_STYLES: Record<VectorIndexStatus, { badge: string; dot: string; pulse: boolean }> = {
  ready: { badge: 'border-success/25 bg-success-subtle text-success', dot: 'bg-success', pulse: false },
  indexing: { badge: 'border-brand/25 bg-brand-subtle text-brand', dot: 'bg-brand', pulse: true },
  stale: { badge: 'border-warning/25 bg-warning-subtle text-warning', dot: 'bg-warning', pulse: false },
  error: { badge: 'border-destructive/25 bg-destructive-subtle text-destructive', dot: 'bg-destructive', pulse: false },
}

const HISTORY_RESULT_DOT: Record<VectorIndexRunResult, string> = {
  ok: 'bg-success',
  error: 'bg-destructive',
  cancelled: 'bg-warning',
}

function formatUpdated(iso: string, locale: Locale): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleDateString(locale, { day: '2-digit', month: 'short' })
}

export function IndexBar({
  embedModels,
  embeddingQuota = null,
  index,
  live = null,
  members,
  onCancel,
  onDelete,
  onModel,
  onReindex,
  onRebuild,
  serverBacked,
  serverFeatureLabels = null,
}: {
  embedModels: readonly EmbedModelDescriptor[]
  /** The caller's embedding-token usage; ``null`` when quotas don't apply. */
  embeddingQuota?: EmbeddingQuota | null
  index: VectorIndexRecord
  /** Live reindex progress for this index, or ``null`` when idle. */
  live?: IndexingJobLive | null
  members: VectorIndexMemberResolved[]
  onCancel: (indexId: string) => void
  onDelete: (indexId: string) => void
  onModel: (indexId: string, model: EmbedModelId) => void
  onReindex: (indexId: string) => void
  /** "Neu aufbauen": full re-ingest from the original files (migrates old docs to
   * ingest-time provenance). Heavier than the default reindex; offered only for a
   * built server collection. */
  onRebuild: (indexId: string) => void
  serverBacked: boolean
  serverFeatureLabels?: string[] | null
}) {
  const { locale, t } = useLocale()
  const reduceMotion = useReducedMotion() ?? false
  const [historyOpen, setHistoryOpen] = useState(false)
  const indexing = index.status === 'indexing'
  // Not-yet-indexed members: when present, the top action indexes just those new
  // documents (incremental, per-file progress); with none it refreshes the whole
  // collection. So the label/icon adapts to what the click will actually do.
  const pendingCount = members.filter((member) => member.member.state === 'pending').length
  // Reindex re-embeds every chunk, so a reached embedding budget blocks
  // it on the server path (the local simulation never gates).
  const quotaBlocked = serverBacked && (embeddingQuota?.exhausted ?? false)
  const stale = index.status === 'stale'
  const style = STATUS_STYLES[index.status]
  const history = index.history ?? []
  const statusLabel =
    index.status === 'ready'
      ? t.vectorIndex.statusReady
      : index.status === 'indexing'
        ? t.vectorIndex.statusIndexing
        : index.status === 'error'
          ? t.vectorIndex.statusError
          : t.vectorIndex.statusStale
  const stats: [string, string][] = [
    [t.vectorIndex.dimensions, index.dims.toLocaleString(locale)],
    [t.vectorIndex.vectors, indexVectorCount(members).toLocaleString(locale)],
    [t.vectorIndex.documents, members.length.toLocaleString(locale)],
    [t.vectorIndex.updated, formatUpdated(index.updatedAt, locale)],
  ]
  const currentModel: EmbedModelDescriptor =
    embedModels.find((model) => model.id === index.model)
      ?? { dims: index.dims, id: index.model, label: index.model, provider: '' }

  const quotaResetLabel =
    embeddingQuota && embeddingQuota.resetAt > 0
      ? new Date(embeddingQuota.resetAt * 1000).toLocaleDateString(locale, {
          day: 'numeric',
          month: 'long',
          timeZone: 'UTC',
        })
      : ''

  return (
    <div className="rounded-lg border border-border bg-card p-3.5 shadow-[0_1px_2px_var(--shadow-hairline)]">
      <div className="flex flex-wrap items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-lg border border-file/25 bg-file-subtle text-file">
          <Layers className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="t-card text-foreground">{t.vectorIndex.title}</h3>
            <span className={cn('inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 t-meta-sm font-semibold', style.badge)}>
              <span className={cn('size-1.5 rounded-full', style.dot, style.pulse && !reduceMotion && 'inqtrix-running-dot')} />
              {statusLabel}
            </span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 t-meta-sm text-muted-foreground">
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
          {/* When quotaBlocked, the reason is the always-visible exhausted
              banner below (keyboard/SR reachable) — no disabled-button tooltip. */}
          <Button
            className="gap-1.5"
            disabled={indexing || quotaBlocked}
            onClick={() => onReindex(index.id)}
            size="sm"
            type="button"
            variant="outline"
          >
            {!indexing && pendingCount > 0 ? (
              <Sparkles className="size-4" />
            ) : (
              <RotateCcw className={cn('size-4', indexing && 'motion-safe:animate-spin')} />
            )}
            {indexing
              ? t.vectorIndex.reindexing
              : pendingCount > 0
                ? t.vectorIndex.reindexPending.replace('{count}', String(pendingCount))
                : t.vectorIndex.reindex}
          </Button>
          {/* Secondary "Neu aufbauen" (full re-ingest from files) for a built
              collection — migrates old docs to ingest-time provenance. Heavier
              than the default refresh, so it lives in a small menu, not a second
              prominent button. */}
          {!indexing && index.serverCollectionId ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  aria-label={t.vectorIndex.moreActions}
                  className="size-8 px-0 text-muted-foreground hover:text-foreground"
                  disabled={quotaBlocked}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <ChevronDown className="size-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className={optionMenuContentClassName} sideOffset={6}>
                <OptionMenuItem
                  active={false}
                  description={t.vectorIndex.rebuildFromFilesHint}
                  icon={Database}
                  label={t.vectorIndex.rebuildFromFiles}
                  onSelect={() => onRebuild(index.id)}
                />
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
          {indexing ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  aria-label={t.vectorIndex.cancelIndexing}
                  className="size-8 px-0 text-muted-foreground hover:text-destructive"
                  onClick={() => onCancel(index.id)}
                  size="sm"
                  type="button"
                  variant="ghost"
                >
                  <XCircle className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="top">{t.vectorIndex.cancelIndexing}</TooltipContent>
            </Tooltip>
          ) : null}
          <ConfirmDelete ariaLabel={t.vectorIndex.remove} hint={t.vectorIndex.removeHint} label={t.vectorIndex.remove} onConfirm={() => onDelete(index.id)} />
        </div>
      </div>

      {/* One stats row — the running progress lives on the RIGHT of it (no
          separate row), matching the rest of the design language. */}
      <div className="mt-3 flex flex-wrap items-end gap-x-6 gap-y-2 border-t border-border/70 pt-3">
        <div className="min-w-0">
          <span className="block t-caption font-semibold text-muted-foreground/80">{t.vectorIndex.embeddingModel}</span>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                aria-label={t.vectorIndex.embeddingModel}
                className="mt-1 h-7 gap-1.5 px-2 font-mono text-xs font-semibold"
                disabled={indexing}
                size="sm"
                type="button"
                variant="outline"
              >
                <span className="truncate">{currentModel.label}</span>
                <ChevronDown className="text-muted-foreground" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className={optionMenuContentClassName} sideOffset={6}>
              <OptionMenuHeader count={embedModels.length} title={t.vectorIndex.embeddingModel} value={currentModel.label} />
              <div className="py-1">
                {embedModels.map((model) => (
                  <OptionMenuItem
                    active={model.id === index.model}
                    description={`${model.provider} · ${model.dims.toLocaleString(locale)}`}
                    icon={Layers}
                    key={model.id}
                    label={model.label}
                    onSelect={() => onModel(index.id, model.id)}
                  />
                ))}
              </div>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        {stats.map(([label, value]) => (
          <div className="min-w-0" key={label}>
            <p className="t-caption font-semibold text-muted-foreground/80">{label}</p>
            <p className="mt-0.5 truncate t-list font-semibold tabular-nums text-foreground">{value}</p>
          </div>
        ))}
        {indexing ? (
          <div className="ml-auto min-w-[10rem]">
            <RunningIndexProgress live={live} reduceMotion={reduceMotion} t={t} />
          </div>
        ) : null}
      </div>

      {quotaBlocked ? (
        <div className="mt-2.5 flex items-center gap-2 rounded-md border border-destructive/25 bg-destructive-subtle px-2.5 py-1.5">
          <span className="inline-flex min-w-0 items-center gap-1.5 t-meta-sm font-medium text-destructive">
            <AlertTriangle className="size-3.5 shrink-0" />
            <span>{t.vectorIndex.embeddingQuotaBanner(quotaResetLabel)}</span>
          </span>
        </div>
      ) : null}

      {index.status === 'error' ? (
        <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 rounded-md border border-destructive/25 bg-destructive-subtle px-2.5 py-1.5">
          <span className="inline-flex min-w-0 items-center gap-1.5 t-meta-sm font-medium text-destructive">
            <AlertTriangle className="size-3.5 shrink-0" />
            <span className="truncate">{index.lastError || t.vectorIndex.errorNotice}</span>
          </span>
          <button
            className="shrink-0 rounded-md border border-destructive/30 bg-card px-2 py-1 t-meta-sm font-semibold text-destructive hover:bg-destructive-subtle"
            onClick={() => onReindex(index.id)}
            type="button"
          >
            {t.vectorIndex.errorRetry}
          </button>
        </div>
      ) : null}

      {stale && !indexing ? (
        <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 rounded-md border border-warning/25 bg-warning-subtle px-2.5 py-1.5">
          <span className="inline-flex items-center gap-1.5 t-meta-sm font-medium text-warning">
            <AlertTriangle className="size-3.5" />
            {t.vectorIndex.staleWarning}
          </span>
          <button
            className="shrink-0 rounded-md border border-warning/30 bg-card px-2 py-1 t-meta-sm font-semibold text-warning hover:bg-warning-subtle"
            onClick={() => onReindex(index.id)}
            type="button"
          >
            {t.vectorIndex.staleAction}
          </button>
        </div>
      ) : null}

      {history.length > 0 ? (
        <IndexHistory
          entries={history}
          locale={locale}
          onToggle={() => setHistoryOpen((open) => !open)}
          open={historyOpen}
          reduceMotion={reduceMotion}
          t={t}
        />
      ) : null}

      <p className="mt-2.5 inline-flex items-center gap-1.5 t-meta-sm text-muted-foreground">
        <Info className="size-3 shrink-0" />
        {serverBacked ? t.vectorIndex.serverNote : t.vectorIndex.simulationNote}
        {serverBacked && serverFeatureLabels && serverFeatureLabels.length > 0 ? (
          <span className="t-hint block text-muted-foreground">
            {t.vectorIndex.serverFeaturesLabel}: {serverFeatureLabels.join(' · ')}
          </span>
        ) : null}
      </p>
    </div>
  )
}

/** Compact running indicator: percent + documents + a determinate bar.
 * Falls back to an indeterminate breathing bar when no live entry exists
 * (e.g. an ``indexing`` status restored from a manifest without a job). */
function RunningIndexProgress({
  live,
  reduceMotion,
  t,
}: {
  live: IndexingJobLive | null
  reduceMotion: boolean
  t: TranslationDictionary
}) {
  if (!live) {
    return (
      <div className="w-full" aria-atomic="true" aria-live="polite">
        <span className="block text-right t-meta-sm font-medium text-brand">
          {t.vectorIndex.progressLabel}
        </span>
        <span className="mt-1 block h-1.5 w-full overflow-hidden rounded-full bg-brand/15" aria-hidden>
          <span className={cn('block h-full rounded-full bg-brand', !reduceMotion && 'inqtrix-segment-breathe')} />
        </span>
      </div>
    )
  }
  if (live.queuePosition != null) {
    // Waiting for a free slot — show the FIFO position, not a stalled 0 %
    // (matches the research-run queue display). Indeterminate breathing bar.
    return (
      <div className="w-full" aria-atomic="true" aria-live="polite">
        <span className="block text-right t-meta-sm font-medium text-muted-foreground">
          {t.vectorIndex.queuedPosition(live.queuePosition)}
        </span>
        <span className="mt-1 block h-1.5 w-full overflow-hidden rounded-full bg-brand/15" aria-hidden>
          <span className={cn('block h-full rounded-full bg-brand/60', !reduceMotion && 'inqtrix-segment-breathe')} />
        </span>
      </div>
    )
  }
  const percent = Math.min(100, Math.max(0, live.percent))
  return (
    <div className="w-full" aria-atomic="true" aria-live="polite">
      <div className="flex items-center justify-end gap-1.5 t-meta-sm font-semibold text-brand">
        <span className={cn('size-1.5 rounded-full bg-brand', !reduceMotion && 'inqtrix-running-dot')} aria-hidden />
        <span
          className={cn('tabular-nums', !reduceMotion && 'inqtrix-metric-flash')}
          key={`${percent}-${live.completedDocuments}`}
        >
          {t.vectorIndex.progressPercentDocs(percent, live.completedDocuments, live.totalDocuments)}
        </span>
      </div>
      <span className="mt-1 block h-1.5 w-full overflow-hidden rounded-full bg-brand/15" aria-hidden>
        <span
          className="block h-full rounded-full bg-brand transition-[width] duration-300 ease-out motion-reduce:transition-none"
          style={{ width: `${Math.max(3, percent)}%` }}
        />
      </span>
    </div>
  )
}

/** Collapsible "last N runs" list (start, duration, outcome). */
function IndexHistory({
  entries,
  locale,
  onToggle,
  open,
  reduceMotion,
  t,
}: {
  entries: VectorIndexRunHistoryEntry[]
  locale: Locale
  onToggle: () => void
  open: boolean
  reduceMotion: boolean
  t: TranslationDictionary
}) {
  const resultLabel = (result: VectorIndexRunResult): string =>
    result === 'error'
      ? t.vectorIndex.historyResultError
      : result === 'cancelled'
        ? t.vectorIndex.historyResultCancelled
        : t.vectorIndex.historyResultOk
  return (
    <div className="mt-2.5 border-t border-border/70 pt-2.5">
      <button
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 t-hint font-semibold text-muted-foreground hover:text-foreground"
        onClick={onToggle}
        type="button"
      >
        <ChevronDown className={cn('size-3.5 transition-transform', !reduceMotion && 'duration-200', !open && '-rotate-90')} />
        <Clock3 className="size-3.5" />
        <span>{t.vectorIndex.historyHeading}</span>
        <span className="text-muted-foreground/70">({entries.length})</span>
      </button>
      <div
        className={cn(
          'grid transition-[grid-template-rows]',
          !reduceMotion && 'duration-200 ease-out',
          open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]',
        )}
      >
        <div className="overflow-hidden">
          <ul className="mt-2 space-y-1.5">
            {entries.map((entry, position) => (
              <li
                className="flex flex-wrap items-center gap-x-2 gap-y-0.5 t-meta-sm text-muted-foreground"
                key={`${entry.finishedAt}-${position}`}
              >
                <span className={cn('size-1.5 shrink-0 rounded-full', HISTORY_RESULT_DOT[entry.result])} aria-hidden />
                <span className="tabular-nums">{t.vectorIndex.historyStarted(formatAddedAt(entry.startedAt, locale))}</span>
                <span className="text-muted-foreground/60">·</span>
                <span className="tabular-nums">{t.vectorIndex.historyRan(formatDurationMsShort(entry.durationMs))}</span>
                <span className="text-muted-foreground/60">·</span>
                <span className="font-medium text-foreground/80">{resultLabel(entry.result)}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
