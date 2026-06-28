import * as React from 'react'
import { ChevronDown, ChevronRight, LoaderCircle, Search, X } from '@/components/icons'
import { cn } from '@/lib/utils'

export const EXPLORER_REVEAL_STEP = 5
export const EXPLORER_DRAG_THRESHOLD_PX = 4

export function isExplorerActionTarget(target: EventTarget | null) {
  return target instanceof HTMLElement && Boolean(target.closest('[data-explorer-action]'))
}

export function isPastExplorerDragThreshold(
  startX: number,
  startY: number,
  event: PointerEvent,
) {
  return Math.hypot(event.clientX - startX, event.clientY - startY) >= EXPLORER_DRAG_THRESHOLD_PX
}

export function ExplorerSectionLabel({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <p className={cn('px-1.5 pb-1 pt-3 t-label text-muted-foreground', className)}>
      {children}
    </p>
  )
}

export function ExplorerRunningIndicator({
  label,
  className,
}: {
  label: string
  className?: string
}) {
  return (
    <LoaderCircle
      aria-label={label}
      className={cn('icon-sm animate-spin text-brand', className)}
    />
  )
}

export function ExplorerSearchField({
  clearLabel,
  label,
  onChange,
  onClear,
  placeholder,
  value,
}: {
  clearLabel: string
  label: string
  onChange: (value: string) => void
  onClear: () => void
  placeholder: string
  value: string
}) {
  return (
    <div className="border-b border-border px-2 py-1.5">
      <div className="flex items-center gap-1.5 rounded-md bg-background/80 px-2 py-1">
        <Search className="icon-sm shrink-0 text-muted-foreground" />
        <input
          aria-label={label}
          className="t-label min-w-0 flex-1 border-0 bg-transparent text-foreground outline-none placeholder:text-muted-foreground"
          autoComplete="off"
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Escape' && value) {
              event.preventDefault()
              onClear()
            }
          }}
          placeholder={placeholder}
          role="searchbox"
          type="text"
          value={value}
        />
        {value ? (
          <button
            aria-label={clearLabel}
            className="grid size-5 shrink-0 place-items-center rounded-sm text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            onClick={onClear}
            type="button"
          >
            <X className="icon-sm" />
          </button>
        ) : null}
      </div>
    </div>
  )
}

export function ExplorerRevealControls({
  onShowLess,
  onShowMore,
  showLessLabel,
  showMoreLabel,
  step = EXPLORER_REVEAL_STEP,
  total,
  visibleCount,
}: {
  onShowLess: () => void
  onShowMore: () => void
  showLessLabel: string
  showMoreLabel: string
  step?: number
  total: number
  visibleCount: number
}) {
  if (total <= step) return null
  const canShowMore = visibleCount < total
  const canShowLess = visibleCount > step

  return (
    <div className="inqtrix-explorer-reveal flex min-h-7 items-center gap-4 px-1.5 pl-6">
      {canShowMore && (
        <button
          className="text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          onMouseDown={(event) => event.preventDefault()}
          onClick={onShowMore}
          type="button"
        >
          {showMoreLabel}
        </button>
      )}
      {canShowLess && (
        <button
          className="text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          onMouseDown={(event) => event.preventDefault()}
          onClick={onShowLess}
          type="button"
        >
          {showLessLabel}
        </button>
      )}
    </div>
  )
}

export function ExplorerFolderRow({
  actions,
  children,
  className,
  dragHandle,
  onPointerDown,
}: {
  actions?: React.ReactNode
  children: React.ReactNode
  className?: string
  dragHandle?: React.ReactNode
  onPointerDown?: React.PointerEventHandler<HTMLDivElement>
}) {
  return (
    <div
      className={cn(
        'group/explorer-folder flex min-h-8 items-center gap-1 rounded-md px-1.5 transition-colors',
        'hover:bg-surface focus-within:bg-surface',
        onPointerDown && 'cursor-grab active:cursor-grabbing',
        className,
      )}
      onPointerDown={onPointerDown}
    >
      {dragHandle}
      <div className="min-w-0 flex-1">{children}</div>
      {actions ? (
        <div
          className="flex items-center gap-0.5 opacity-0 transition-opacity group-hover/explorer-folder:opacity-100 group-focus-within/explorer-folder:opacity-100"
          data-explorer-action
        >
          {actions}
        </div>
      ) : null}
    </div>
  )
}

export function ExplorerFolderToggle({
  count,
  expanded,
  icon,
  label,
  onDoubleClick,
  onToggle,
  title,
}: {
  count?: React.ReactNode
  expanded: boolean
  icon: React.ReactNode
  label: string
  onDoubleClick?: React.MouseEventHandler<HTMLButtonElement>
  onToggle: () => void
  title: React.ReactNode
}) {
  const Chevron = expanded ? ChevronDown : ChevronRight
  return (
    <button
      aria-expanded={expanded}
      aria-label={label}
      className="flex min-h-8 w-full min-w-0 items-center gap-1.5 rounded-md text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      onDoubleClick={onDoubleClick}
      onClick={onToggle}
      type="button"
    >
      <span className="shrink-0 text-muted-foreground">{icon}</span>
      <span className="min-w-0 truncate t-list text-foreground">{title}</span>
      <Chevron
        aria-hidden="true"
        className="icon-xs shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover/explorer-folder:opacity-100 group-focus-within/explorer-folder:opacity-100"
      />
      {count !== undefined ? (
        <span className="ml-auto shrink-0 t-hint font-semibold tabular-nums text-muted-foreground">
          {count}
        </span>
      ) : null}
    </button>
  )
}

export function ExplorerItemRow({
  active = false,
  children,
  className,
  dragging = false,
  nested = false,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & {
  active?: boolean
  dragging?: boolean
  nested?: boolean
}) {
  return (
    <div
      className={cn(
        'group/explorer-item relative flex min-h-8 min-w-0 items-center rounded-md border border-transparent',
        'px-1.5 transition-colors',
        nested && 'pl-6',
        props.onPointerDown && 'cursor-grab active:cursor-grabbing',
        active
          ? 'border-brand/20 bg-brand-subtle text-brand'
          : 'text-foreground hover:bg-surface focus-within:bg-surface',
        dragging && 'border-brand/40 bg-brand-subtle/80 opacity-70',
        className,
      )}
      {...props}
    >
      {children}
    </div>
  )
}
