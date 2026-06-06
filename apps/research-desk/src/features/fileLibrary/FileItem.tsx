import { type DragEvent } from 'react'
import { Folder, GripVertical, Link, X } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type { FileAssetRecord, VectorIndexMemberState } from '@/features/project/types'
import {
  ConfirmDelete,
  InlineText,
  MoveMenu,
  StatusMark,
  TypeBadge,
  TypeTile,
  type MoveTarget,
} from './controls'
import { chunkEstimate, formatBytes, typeMeta } from './helpers'
import { FILE_DRAG_TYPE } from './constants'

export const LIBRARY_GRID = 'grid grid-cols-[minmax(0,1fr)_4.5rem_3rem_5rem_3.5rem_4rem] items-center gap-3'
export const INDEX_GRID = 'grid grid-cols-[minmax(0,1fr)_4.5rem_4rem_5rem_minmax(6rem,9rem)_2.5rem] items-center gap-3'

export type FileItemProps = {
  asset: FileAssetRecord
  mode: 'library' | 'index'
  breadcrumb?: string | null
  referenceCount?: number
  moveTargets?: MoveTarget[]
  memberState?: VectorIndexMemberState
  source?: string | null
  onRename?: (fileId: string, label: string) => void
  onMove?: (fileId: string, sectionId: string, groupId: string | null) => void
  onDelete?: (fileId: string) => void
  onRemoveFromIndex?: (fileId: string) => void
  onDragStart?: (fileId: string) => void
  onDragEnd?: () => void
}

function ChunkCell({ asset, state }: { asset: FileAssetRecord; state: VectorIndexMemberState }) {
  const { locale, t } = useLocale()
  if (state === 'pending') {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex cursor-help items-center justify-end gap-1 text-brand">
            <span className="inqtrix-running-dot size-1.5 rounded-full bg-brand" />
            <span className="text-[11px] font-medium">{t.vectorIndex.embeddingRunning}</span>
          </span>
        </TooltipTrigger>
        <TooltipContent side="top">{t.vectorIndex.embeddingRunningTooltip}</TooltipContent>
      </Tooltip>
    )
  }
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

function NameCell({
  asset,
  breadcrumb,
  draggable,
  onRename,
}: {
  asset: FileAssetRecord
  breadcrumb?: string | null
  draggable: boolean
  onRename?: (fileId: string, label: string) => void
}) {
  const { t } = useLocale()
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      {draggable ? (
        <GripVertical className="size-3.5 shrink-0 text-transparent transition-colors group-hover:text-muted-foreground/50" />
      ) : null}
      <TypeTile asset={asset} size="sm" />
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="shrink-0 font-mono text-[11px] text-muted-foreground/70">@files:</span>
          {onRename ? (
            <InlineText
              ariaLabel={t.fileLibrary.rename}
              className="min-w-0 max-w-full text-[13px] font-semibold text-foreground"
              onCommit={(label) => onRename(asset.id, label)}
              value={asset.label}
            />
          ) : (
            <span className="min-w-0 truncate text-[13px] font-semibold text-foreground">{asset.label}</span>
          )}
          <StatusMark asset={asset} />
        </div>
        <div className="flex min-w-0 items-center gap-1.5 text-[11px] text-muted-foreground">
          <span className="min-w-0 flex-1 truncate" title={asset.fileName}>{asset.fileName}</span>
          {breadcrumb ? <span className="shrink-0 whitespace-nowrap text-muted-foreground/60">· {breadcrumb}</span> : null}
        </div>
      </div>
    </div>
  )
}

function RowActions(props: FileItemProps) {
  const { t } = useLocale()
  const { asset, mode, moveTargets, onDelete, onMove, onRemoveFromIndex } = props
  if (mode === 'index') {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            aria-label={t.vectorIndex.removeDoc}
            className="size-7 text-muted-foreground hover:text-foreground"
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
    )
  }
  return (
    <>
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
  const { asset, breadcrumb, memberState, mode, referenceCount, source } = props
  const isIndex = mode === 'index'
  const meta = typeMeta(asset)
  return (
    <div
      className={cn(
        isIndex ? INDEX_GRID : LIBRARY_GRID,
        'group border-t border-border/60 px-3 py-2 transition-colors first:border-t-0 hover:bg-accent/40',
        !isIndex && 'cursor-grab active:cursor-grabbing',
      )}
      {...dragHandlers(props)}
    >
      <NameCell asset={asset} breadcrumb={breadcrumb} draggable={!isIndex} onRename={props.onRename} />
      <div className="min-w-0"><TypeBadge asset={asset} /></div>
      <div className="text-right text-xs tabular-nums text-muted-foreground">
        {isIndex ? (
          <ChunkCell asset={asset} state={memberState ?? 'pending'} />
        ) : meta.paged && asset.pageCount != null ? (
          asset.pageCount
        ) : (
          <span className="text-muted-foreground/40">{t.fileLibrary.referencedNone}</span>
        )}
      </div>
      <div className="text-right text-xs tabular-nums text-muted-foreground">{formatBytes(asset.sizeBytes, locale)}</div>
      {isIndex ? (
        <div className="flex min-w-0 items-center gap-1 text-[11px] text-muted-foreground">
          <Folder className="size-3 shrink-0" />
          <span className="truncate">{source ?? t.fileLibrary.referencedNone}</span>
        </div>
      ) : (
        <div className="flex justify-end text-[11px]"><UsedCell count={referenceCount ?? 0} /></div>
      )}
      <div className="flex items-center justify-end gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
        <RowActions {...props} />
      </div>
    </div>
  )
}

export function FileCard(props: FileItemProps) {
  const { locale, t } = useLocale()
  const { asset, breadcrumb, memberState, mode, referenceCount } = props
  const isIndex = mode === 'index'
  const meta = typeMeta(asset)
  return (
    <div
      className={cn(
        'group flex flex-col rounded-lg border border-border bg-card p-3 shadow-[0_1px_2px_var(--shadow-hairline)] transition-colors hover:bg-accent/30',
        !isIndex && 'cursor-grab active:cursor-grabbing',
      )}
      {...dragHandlers(props)}
    >
      <div className="flex items-start justify-between gap-2">
        <TypeTile asset={asset} size="md" />
        <div className="flex items-center gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
          <RowActions {...props} />
        </div>
      </div>
      <div className="mt-2.5 min-w-0">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="shrink-0 font-mono text-[11px] text-muted-foreground/70">@files:</span>
          {props.onRename ? (
            <InlineText
              ariaLabel={t.fileLibrary.rename}
              className="min-w-0 max-w-full text-[13px] font-semibold text-foreground"
              onCommit={(label) => props.onRename?.(asset.id, label)}
              value={asset.label}
            />
          ) : (
            <span className="min-w-0 truncate text-[13px] font-semibold text-foreground">{asset.label}</span>
          )}
          <StatusMark asset={asset} />
        </div>
        <p className="mt-0.5 truncate text-[11px] text-muted-foreground" title={asset.fileName}>
          {asset.fileName}
          {breadcrumb ? ` · ${breadcrumb}` : ''}
        </p>
      </div>
      <div className="mt-3 flex items-center justify-between gap-2 border-t border-border/60 pt-2 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-2">
          <TypeBadge asset={asset} />
          <span className="tabular-nums">{formatBytes(asset.sizeBytes, locale)}</span>
        </span>
        {isIndex ? (
          <ChunkCell asset={asset} state={memberState ?? 'pending'} />
        ) : meta.paged && asset.pageCount != null ? (
          <span className="tabular-nums">
            {asset.pageCount} {t.fileLibrary.pagesUnit}
          </span>
        ) : (
          <UsedCell count={referenceCount ?? 0} />
        )}
      </div>
    </div>
  )
}

export function ListHeader({ mode }: { mode: 'library' | 'index' }) {
  const { t } = useLocale()
  const isIndex = mode === 'index'
  return (
    <div
      className={cn(
        isIndex ? INDEX_GRID : LIBRARY_GRID,
        'px-3 pb-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground',
      )}
    >
      <span>{t.fileLibrary.nameColumn}</span>
      <span>{t.fileLibrary.typeColumn}</span>
      <span className="text-right">{isIndex ? t.fileLibrary.chunksColumn : t.fileLibrary.pagesColumn}</span>
      <span className="text-right">{t.fileLibrary.sizeColumn}</span>
      <span className={isIndex ? '' : 'text-right'}>{isIndex ? t.fileLibrary.sourceColumn : t.fileLibrary.referencedColumn}</span>
      <span />
    </div>
  )
}
