import { type DragEvent } from 'react'
import { Eye, Folder, Link, Sparkles, X } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type { FileAssetRecord, VectorIndexMemberState } from '@/features/project/types'
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
import { chunkEstimate, formatAddedAt, formatAddedAtFull, formatBytes, typeMeta } from './helpers'
import { FILE_DRAG_TYPE } from './constants'

export const LIBRARY_GRID = 'grid grid-cols-[minmax(0,1fr)_4.5rem_3rem_5rem_7rem_3.5rem_4rem] items-center gap-3'
export const INDEX_GRID = 'grid grid-cols-[minmax(0,1fr)_4.5rem_4rem_5rem_7rem_minmax(6rem,9rem)_2.5rem] items-center gap-3'

export type FileItemProps = {
  asset: FileAssetRecord
  mode: 'library' | 'index'
  breadcrumb?: string | null
  referenceCount?: number
  moveTargets?: MoveTarget[]
  memberState?: VectorIndexMemberState
  /** Whether the member's index is CURRENTLY running an embed job — so a
   * `pending` member only reads "embedding…" while a job actually runs, never
   * forever (the prior bug). */
  indexing?: boolean
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
  indexing = false,
  liveProgress,
}: {
  asset: FileAssetRecord
  state: VectorIndexMemberState
  indexing?: boolean
  /** During an active run, this file's server-CONFIRMED outcome so far:
   * `embedded` (done), `skipped` (no text), or undefined (not yet processed →
   * still running). Drives the live, per-file feedback. */
  liveProgress?: 'embedded' | 'skipped'
}) {
  const { locale, t } = useLocale()
  // While a run is in flight the live per-file outcome wins (each row flips as
  // the server confirms it); once it completes the persisted state takes over.
  // A still-unprocessed file shows the pulsing dot ONLY during an active run,
  // never forever (No-Silent-Fallbacks).
  const effective: 'embedded' | 'skipped' | 'running' | 'pending' = indexing
    ? (liveProgress ?? 'running')
    : state
  if (effective === 'running') {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex cursor-help items-center justify-end gap-1 text-brand">
            <span className="inqtrix-running-dot size-1.5 rounded-full bg-brand" />
            <span className="t-meta-sm font-medium">{t.vectorIndex.embeddingRunning}</span>
          </span>
        </TooltipTrigger>
        <TooltipContent side="top">{t.vectorIndex.embeddingRunningTooltip}</TooltipContent>
      </Tooltip>
    )
  }
  if (effective === 'embedded') {
    const chunks = chunkEstimate(asset)
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="cursor-help tabular-nums">{chunks.toLocaleString(locale)}</span>
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
        <span className="inline-flex cursor-help items-center justify-end gap-1 text-muted-foreground">
          <span className="t-meta-sm font-medium">
            {noText ? t.vectorIndex.embeddingNoText : t.vectorIndex.embeddingPending}
          </span>
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

function NameCell({
  asset,
  breadcrumb,
  onRename,
}: {
  asset: FileAssetRecord
  breadcrumb?: string | null
  onRename?: (fileId: string, label: string) => void
}) {
  const { t } = useLocale()
  return (
    <div className="flex min-w-0 items-center gap-2">
      <TypeTile asset={asset} size="sm" />
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="shrink-0 t-mono text-muted-foreground/70">@files:</span>
          {onRename ? (
            <InlineText
              ariaLabel={t.fileLibrary.rename}
              className="min-w-0 max-w-full t-list text-foreground"
              onCommit={(label) => onRename(asset.id, label)}
              value={asset.label}
            />
          ) : (
            <span className="min-w-0 truncate t-list text-foreground">{asset.label}</span>
          )}
          <StatusMark asset={asset} />
          <ParserBadge asset={asset} />
        </div>
        <div className="flex min-w-0 items-center gap-1.5 t-meta-sm text-muted-foreground">
          <span className="min-w-0 flex-1 truncate" title={asset.fileName}>{asset.fileName}</span>
          {breadcrumb ? <span className="shrink-0 whitespace-nowrap text-muted-foreground/60">· {breadcrumb}</span> : null}
        </div>
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
  const { asset, breadcrumb, indexing, liveProgress, memberState, mode, referenceCount, source } = props
  const isIndex = mode === 'index'
  const meta = typeMeta(asset)
  const canPreview = mode === 'library' && Boolean(props.onPreview)
  return (
    <div
      className={cn(
        isIndex ? INDEX_GRID : LIBRARY_GRID,
        'group rounded-md px-2 py-2 transition-colors hover:bg-accent/45',
        !isIndex && 'cursor-grab active:cursor-grabbing',
      )}
      onClick={canPreview ? () => props.onPreview?.(asset.id) : undefined}
      {...dragHandlers(props)}
    >
      <NameCell asset={asset} breadcrumb={breadcrumb} onRename={props.onRename} />
      <div className="min-w-0"><TypeBadge asset={asset} /></div>
      <div className="text-right t-meta tabular-nums text-muted-foreground">
        {isIndex ? (
          <ChunkCell asset={asset} indexing={indexing} liveProgress={liveProgress} state={memberState ?? 'pending'} />
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
        className="flex items-center justify-end gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100"
        onClick={(event) => event.stopPropagation()}
      >
        <RowActions {...props} />
      </div>
    </div>
  )
}

export function FileCard(props: FileItemProps) {
  const { locale, t } = useLocale()
  const { asset, breadcrumb, indexing, liveProgress, memberState, mode, referenceCount } = props
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
          className="flex items-center gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100"
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
          <ChunkCell asset={asset} indexing={indexing} liveProgress={liveProgress} state={memberState ?? 'pending'} />
        ) : (
          <UsedCell count={referenceCount ?? 0} />
        )}
      </div>
    </div>
  )
}

