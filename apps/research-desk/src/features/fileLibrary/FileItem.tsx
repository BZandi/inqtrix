import { type DragEvent } from 'react'
import { Eye, Folder, Info, Link, Sparkles, X } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type { FileAssetRecord, VectorIndexMemberState } from '@/features/project/types'
import { StatusBadge } from '@/features/settings/parts'
import {
  ConfirmDelete,
  InlineText,
  MoveMenu,
  ParserBadge,
  StatusMark,
  TypeBadge,
  TypeTile,
  type MoveTarget,
} from './controls'
import { chunkEstimate, formatAddedAt, formatAddedAtFull, formatBytes, memberCellState, typeMeta } from './helpers'
import { FILE_DRAG_TYPE } from './constants'

export const EXPLORER_GRID = 'grid grid-cols-[minmax(16rem,1fr)_4.5rem_5rem_5rem_7rem_minmax(6rem,9rem)_6.25rem] items-center gap-3'

export type FileItemProps = {
  asset: FileAssetRecord
  mode: 'library' | 'index'
  breadcrumb?: string | null
  referenceCount?: number
  moveTargets?: MoveTarget[]
  memberState?: VectorIndexMemberState
  /** Whether the member's index is CURRENTLY running ANY embed job — gates the
   * per-row actions (no "remove" / no per-file "index this file" while a run is
   * in flight, "one job per index"). NOT the running-label gate — see `inRun`. */
  indexing?: boolean
  /** Whether THIS specific file is part of the current run's working set
   * (`runningFileIds`) — drives whether the row reads "läuft". */
  inRun?: boolean
  /** This file's server-confirmed live outcome during an active run. */
  liveProgress?: 'embedded' | 'skipped'
  source?: string | null
  onRename?: (fileId: string, label: string) => void
  onMove?: (fileId: string, sectionId: string, groupId: string | null) => void
  onDelete?: (fileId: string) => void
  onRemoveFromIndex?: (fileId: string) => void
  /** Index just THIS file (per-row action), shown for a not-yet-indexed
   * member when no run is in flight. */
  onIndexMember?: (fileId: string) => void
  onDragStart?: (fileId: string) => void
  onDragEnd?: () => void
  onPreview?: (fileId: string) => void
}

function ChunkCell({
  asset,
  state,
  inRun = false,
  liveProgress,
}: {
  asset: FileAssetRecord
  state: VectorIndexMemberState
  /** Whether THIS file is part of the running job's working set
   * (`runningFileIds`). "läuft" is shown ONLY for files the active run actually
   * processes — a file outside the run keeps its real state, so indexing one new
   * document never makes the already-embedded rows read "läuft" (the prior bug). */
  inRun?: boolean
  /** During an active run, this file's server-CONFIRMED outcome so far:
   * `embedded` (done), `skipped` (no text), or undefined (not yet processed →
   * still running). Drives the live, per-file feedback. */
  liveProgress?: 'embedded' | 'skipped'
}) {
  const { locale, t } = useLocale()
  // The server-confirmed live outcome wins (each row flips as it lands); a file
  // in the run but not yet confirmed shows the pulsing "läuft"; everything else
  // keeps its persisted state (No-Silent-Fallbacks: only an in-run file pulses,
  // never forever and never a row outside the run).
  const effective = memberCellState(state, inRun, liveProgress)
  if (effective === 'running') {
    // A brand status pill with the pulsing dot — same table density as the other
    // states and the same tone as the index-level "Indexiere…" badge.
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex h-5 cursor-help items-center gap-1 rounded-md border border-brand/25 bg-brand-subtle px-1.5 t-hint font-medium text-brand">
            <span className="inqtrix-running-dot size-1.5 rounded-full bg-brand" />
            {t.vectorIndex.embeddingRunning}
          </span>
        </TooltipTrigger>
        <TooltipContent side="top">{t.vectorIndex.embeddingRunningTooltip}</TooltipContent>
      </Tooltip>
    )
  }
  if (effective === 'embedded') {
    // Explicit "Indexiert" status (best practice) instead of a bare number; the
    // chunk count moves into the tooltip as the secondary detail.
    const chunks = chunkEstimate(asset)
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex cursor-help">
            <StatusBadge density="table" label={t.vectorIndex.statusReady} tone="success" />
          </span>
        </TooltipTrigger>
        <TooltipContent side="top">{t.vectorIndex.chunks.replace('{count}', chunks.toLocaleString(locale))}</TooltipContent>
      </Tooltip>
    )
  }
  // 'skipped' (live, no text) or 'pending' (queued / unembeddable, not running).
  const noText =
    effective === 'skipped' ||
    (asset.extractedText.trim().length === 0 && !asset.serverFileId)
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex cursor-help">
          <StatusBadge
            density="table"
            label={noText ? t.vectorIndex.embeddingNoText : t.vectorIndex.embeddingPending}
            tone="neutral"
          />
        </span>
      </TooltipTrigger>
      <TooltipContent side="top">
        {noText
          ? t.vectorIndex.embeddingNoTextTooltip
          : t.vectorIndex.embeddingPendingTooltip}
      </TooltipContent>
    </Tooltip>
  )
}

function UsedCell({ count }: { count: number }) {
  const { t } = useLocale()
  if (count <= 0) return <span className="text-muted-foreground/40">{t.fileLibrary.referencedNone}</span>
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex cursor-help items-center gap-1 text-muted-foreground">
          <Link className="size-3" />
          <span className="tabular-nums">{count}×</span>
        </span>
      </TooltipTrigger>
      <TooltipContent side="top">{t.fileLibrary.referencedTooltip.replace('{count}', String(count))}</TooltipContent>
    </Tooltip>
  )
}

/** Date + time the file was added to the library (`createdAt`). The compact
 * cell omits the year; the hover tooltip carries the full stamp. */
function AddedCell({ asset }: { asset: FileAssetRecord }) {
  const { locale, t } = useLocale()
  return (
    <div className="text-right t-meta-sm tabular-nums text-muted-foreground">
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="cursor-help whitespace-nowrap">{formatAddedAt(asset.createdAt, locale)}</span>
        </TooltipTrigger>
        <TooltipContent side="top">
          {t.fileLibrary.addedTooltip.replace('{date}', formatAddedAtFull(asset.createdAt, locale))}
        </TooltipContent>
      </Tooltip>
    </div>
  )
}

function parserLabel(asset: FileAssetRecord, t: ReturnType<typeof useLocale>['t']): string {
  if (asset.parsePending) return t.fileLibrary.parserRunning
  if (asset.parserId === 'markitdown') return t.fileLibrary.parserMarkitdown
  if (asset.parserId === 'client') return t.fileLibrary.parserClient
  return t.fileLibrary.referencedNone
}

function fileHandle(asset: FileAssetRecord): string {
  return `@files:${asset.label}`
}

function FileHandle({ asset }: { asset: FileAssetRecord }) {
  const { t } = useLocale()
  const handle = fileHandle(asset)
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="shrink-0 cursor-help t-mono text-muted-foreground/70 hover:text-foreground/80">@files:</span>
      </TooltipTrigger>
      <TooltipContent
        className="max-w-[320px] border border-border bg-popover p-2.5 text-popover-foreground shadow-lg"
        side="top"
      >
        <p className="truncate font-mono t-label text-foreground">{handle}</p>
        <p className="mt-1 truncate t-meta-sm text-muted-foreground" title={asset.fileName}>
          {asset.fileName}
        </p>
        <p className="mt-1.5 t-hint text-muted-foreground">{t.fileLibrary.handleTooltip}</p>
      </TooltipContent>
    </Tooltip>
  )
}

function FileDetails({
  asset,
  location,
  mode,
}: {
  asset: FileAssetRecord
  location?: string | null
  mode: FileItemProps['mode']
}) {
  const { locale, t } = useLocale()
  const meta = typeMeta(asset)
  const detailRows: [string, string][] = [
    [t.fileLibrary.detailHandle, fileHandle(asset)],
    [t.fileLibrary.detailFileName, asset.fileName],
    [t.fileLibrary.detailLocation, location || t.fileLibrary.referencedNone],
    [t.fileLibrary.detailParser, parserLabel(asset, t)],
    [
      mode === 'index' ? t.fileLibrary.detailChunks : t.fileLibrary.columnPages,
      mode === 'index'
        ? chunkEstimate(asset).toLocaleString(locale)
        : meta.paged && asset.pageCount != null
          ? asset.pageCount.toLocaleString(locale)
          : t.fileLibrary.noPages,
    ],
    [t.fileLibrary.columnSize, formatBytes(asset.sizeBytes, locale)],
  ]
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex shrink-0 cursor-help items-center text-muted-foreground/70 hover:text-foreground">
          <Info className="size-3.5" />
        </span>
      </TooltipTrigger>
      <TooltipContent
        className="max-w-[320px] border border-border bg-popover p-2.5 text-popover-foreground shadow-lg"
        side="top"
      >
        <p className="truncate t-label text-foreground">{asset.label}</p>
        <div className="mt-2 grid gap-1.5">
          {detailRows.map(([label, value]) => (
            <div className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-2 t-meta-sm" key={label}>
              <span className="text-muted-foreground">{label}</span>
              <span className="min-w-0 truncate text-foreground" title={value}>{value}</span>
            </div>
          ))}
          {asset.parseWarning ? (
            <div className="border-t border-border/70 pt-1.5 t-meta-sm text-warning">
              {asset.parseWarning}
            </div>
          ) : null}
        </div>
      </TooltipContent>
    </Tooltip>
  )
}

function NameCell({
  asset,
  breadcrumb,
  mode,
  onRename,
  source,
}: {
  asset: FileAssetRecord
  breadcrumb?: string | null
  mode: FileItemProps['mode']
  onRename?: (fileId: string, label: string) => void
  source?: string | null
}) {
  const { t } = useLocale()
  const titleNode = onRename ? (
    <InlineText
      ariaLabel={t.fileLibrary.rename}
      className="min-w-0 max-w-full t-list text-foreground"
      onCommit={(label) => onRename(asset.id, label)}
      value={asset.label}
    />
  ) : (
    <span className="min-w-0 truncate t-list text-foreground">{asset.label}</span>
  )
  const location = breadcrumb || source || null

  return (
    <div className="flex min-w-0 items-center gap-2">
      <TypeTile asset={asset} size="sm" />
      <div className="grid min-w-0 flex-1 grid-cols-[minmax(0,1fr)_1.25rem] items-center gap-1.5">
        <div className="flex min-w-0 items-center gap-1.5">
          <FileHandle asset={asset} />
          {titleNode}
          <StatusMark asset={asset} />
          {asset.parsePending ? <ParserBadge asset={asset} /> : null}
        </div>
        <span className="flex justify-center">
          <FileDetails asset={asset} location={location} mode={mode} />
        </span>
      </div>
    </div>
  )
}

function RowActions(props: FileItemProps) {
  const { t } = useLocale()
  const { asset, indexing, memberState, mode, moveTargets, onDelete, onIndexMember, onMove, onRemoveFromIndex } = props
  if (mode === 'index') {
    // A not-yet-indexed member gets its own "index this file" action (no run in
    // flight) — indexing just it, no full rebuild.
    const canIndex = memberState === 'pending' && !indexing && Boolean(onIndexMember)
    return (
      <>
        {canIndex ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={t.vectorIndex.indexMember}
                className="size-7 text-brand hover:text-brand"
                onClick={() => onIndexMember?.(asset.id)}
                size="icon"
                type="button"
                variant="ghost"
              >
                <Sparkles className="size-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top">{t.vectorIndex.indexMember}</TooltipContent>
          </Tooltip>
        ) : null}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              aria-label={t.vectorIndex.removeDoc}
              className="size-7 text-muted-foreground hover:text-foreground"
              // Disabled mid-run: removing a still-building member would strand
              // its server document (its id isn't tracked until the run lands).
              disabled={indexing}
              onClick={() => onRemoveFromIndex?.(asset.id)}
              size="icon"
              type="button"
              variant="ghost"
            >
              <X className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="top">{t.vectorIndex.removeDoc}</TooltipContent>
        </Tooltip>
      </>
    )
  }
  return (
    <>
      {props.onPreview ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              aria-label={t.filePreview.open}
              className="size-7 text-muted-foreground hover:text-foreground"
              onClick={() => props.onPreview?.(asset.id)}
              size="icon"
              type="button"
              variant="ghost"
            >
              <Eye className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="top">{t.filePreview.open}</TooltipContent>
        </Tooltip>
      ) : null}
      {moveTargets && onMove ? <MoveMenu asset={asset} onMove={onMove} targets={moveTargets} /> : null}
      {onDelete ? (
        <ConfirmDelete
          ariaLabel={t.fileLibrary.remove}
          hint={t.fileLibrary.removeFileHint.replace('{label}', asset.label)}
          onConfirm={() => onDelete(asset.id)}
        />
      ) : null}
    </>
  )
}

function dragHandlers(props: FileItemProps) {
  const { asset, mode, onDragEnd, onDragStart } = props
  if (mode === 'index') return {}
  return {
    draggable: true,
    onDragEnd,
    onDragStart: (event: DragEvent) => {
      event.dataTransfer.setData(FILE_DRAG_TYPE, asset.id)
      event.dataTransfer.effectAllowed = 'move'
      onDragStart?.(asset.id)
    },
  }
}

export function FileRow(props: FileItemProps) {
  const { locale, t } = useLocale()
  const { asset, breadcrumb, inRun, liveProgress, memberState, mode, referenceCount, source } = props
  const isIndex = mode === 'index'
  const meta = typeMeta(asset)
  const canPreview = mode === 'library' && Boolean(props.onPreview)
  return (
    <div
      className={cn(
        EXPLORER_GRID,
        'group min-h-9 rounded-md border-b border-border/45 px-2 py-1.5 transition-colors hover:bg-surface/55',
        !isIndex && 'cursor-grab active:cursor-grabbing',
      )}
      onClick={canPreview ? () => props.onPreview?.(asset.id) : undefined}
      {...dragHandlers(props)}
    >
      <NameCell asset={asset} breadcrumb={breadcrumb} mode={mode} onRename={props.onRename} source={source} />
      <div className="min-w-0"><TypeBadge asset={asset} /></div>
      <div className={cn('t-meta tabular-nums text-muted-foreground', isIndex ? 'flex justify-start' : 'text-right')}>
        {isIndex ? (
          <ChunkCell asset={asset} inRun={inRun} liveProgress={liveProgress} state={memberState ?? 'pending'} />
        ) : meta.paged && asset.pageCount != null ? (
          asset.pageCount
        ) : (
          <span className="text-muted-foreground/40">{t.fileLibrary.referencedNone}</span>
        )}
      </div>
      <div className="text-right t-meta tabular-nums text-muted-foreground">{formatBytes(asset.sizeBytes, locale)}</div>
      <AddedCell asset={asset} />
      {isIndex ? (
        <div className="flex min-w-0 items-center gap-1 t-meta-sm text-muted-foreground">
          <Folder className="size-3 shrink-0" />
          <span className="truncate">{source ?? t.fileLibrary.referencedNone}</span>
        </div>
      ) : (
        <div className="flex justify-end t-meta-sm"><UsedCell count={referenceCount ?? 0} /></div>
      )}
      <div
        className="flex items-center justify-end gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100"
        onClick={(event) => event.stopPropagation()}
      >
        <RowActions {...props} />
      </div>
    </div>
  )
}

export function FileCard(props: FileItemProps) {
  const { locale, t } = useLocale()
  const { asset, breadcrumb, inRun, liveProgress, memberState, mode, referenceCount } = props
  const isIndex = mode === 'index'
  const meta = typeMeta(asset)
  const canPreview = mode === 'library' && Boolean(props.onPreview)
  return (
    <div
      className={cn(
        'group flex flex-col rounded-lg border border-border bg-card p-3 shadow-[0_1px_2px_var(--shadow-hairline)] transition-colors hover:bg-accent/30',
        !isIndex && 'cursor-grab active:cursor-grabbing',
      )}
      onClick={canPreview ? () => props.onPreview?.(asset.id) : undefined}
      {...dragHandlers(props)}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <TypeTile asset={asset} size="md" />
          <TypeBadge asset={asset} />
        </div>
        <div
          className="flex items-center gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100"
          onClick={(event) => event.stopPropagation()}
        >
          <RowActions {...props} />
        </div>
      </div>
      <div className="mt-2.5 min-w-0">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="shrink-0 t-mono text-muted-foreground/70">@files:</span>
          {props.onRename ? (
            <InlineText
              ariaLabel={t.fileLibrary.rename}
              className="min-w-0 max-w-full t-list text-foreground"
              onCommit={(label) => props.onRename?.(asset.id, label)}
              value={asset.label}
            />
          ) : (
            <span className="min-w-0 truncate t-list text-foreground">{asset.label}</span>
          )}
          <StatusMark asset={asset} />
          <ParserBadge asset={asset} />
        </div>
        <p className="mt-0.5 truncate t-meta-sm text-muted-foreground" title={asset.fileName}>
          {asset.fileName}
          {breadcrumb ? ` · ${breadcrumb}` : ''}
        </p>
        <p className="mt-1 t-hint tabular-nums text-muted-foreground">{formatAddedAt(asset.createdAt, locale)}</p>
      </div>
      <div className="mt-3 flex items-center justify-between gap-2 border-t border-border/60 pt-2 t-meta-sm text-muted-foreground">
        <span className="tabular-nums">
          {formatBytes(asset.sizeBytes, locale)}
          {meta.paged && asset.pageCount != null ? ` · ${asset.pageCount} ${t.fileLibrary.pagesUnit}` : ''}
        </span>
        {isIndex ? (
          <ChunkCell asset={asset} inRun={inRun} liveProgress={liveProgress} state={memberState ?? 'pending'} />
        ) : (
          <UsedCell count={referenceCount ?? 0} />
        )}
      </div>
    </div>
  )
}
