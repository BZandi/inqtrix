import { ChevronLeft, ChevronRight, type LucideIcon } from '@/components/icons'
import { cn } from '@/lib/utils'

export type MentionTone = 'brand' | 'success' | 'file' | 'warning'

export type MentionMenuOption = {
  /** Human-readable primary line (category: description, item: title). */
  primary: string
  /** Secondary line (category: prefix, item: `@kind:handle` in mono). */
  secondary: string
  icon: LucideIcon
  tone: MentionTone
  /** Category rows show a drill-in chevron; item rows show the enter hint. */
  isCategory: boolean
  /** Optional group label; consecutive equal values render under one header. */
  group?: string
}

export type MentionMenuScope = {
  /** null = root level (no breadcrumb, no back affordance). */
  kind: string | null
  query: string
  icon?: LucideIcon
  tone?: MentionTone
}

export type MentionMenuLabels = {
  rootTitle: string
  filterPlaceholder: string
  navHint: string
  selectHint: string
  closeHint: string
  backHint: string
}

const toneText: Record<MentionTone, string> = {
  brand: 'text-brand',
  success: 'text-success',
  file: 'text-file',
  warning: 'text-warning',
}

const toneBar: Record<MentionTone, string> = {
  brand: 'bg-brand',
  success: 'bg-success',
  file: 'bg-file',
  warning: 'bg-warning',
}

const toneChip: Record<MentionTone, string> = {
  brand: 'bg-brand-subtle/70 text-brand',
  success: 'bg-success-subtle/70 text-success',
  file: 'bg-file-subtle/70 text-file',
  warning: 'bg-warning-subtle/70 text-warning',
}

const kbd =
  'inline-flex h-4 min-w-4 items-center justify-center rounded border border-border bg-surface px-1 text-[10px] font-sans text-muted-foreground'

/**
 * Presentation-only mention popover shared by every `@`-autocomplete surface
 * (chat composer and editor composer). It owns no trigger, keyboard, or
 * selection logic: the caller supplies the resolved scope, options, active
 * index, and i18n labels, and receives select/hover/back intents back. Options
 * arrive in display order, so `index` is the flat index used for both hover and
 * keyboard navigation; grouping only inserts headers between consecutive runs
 * of the same `group` value and never reorders.
 */
export function MentionMenu({
  scope,
  options,
  activeIndex,
  labels,
  onSelect,
  onHover,
  onBack,
}: {
  scope: MentionMenuScope
  options: MentionMenuOption[]
  activeIndex: number
  labels: MentionMenuLabels
  onSelect: (index: number) => void
  onHover: (index: number) => void
  onBack?: () => void
}) {
  const isRoot = scope.kind == null
  const canGoBack = !isRoot && Boolean(onBack)
  const ScopeIcon = scope.icon

  const groups: { label: string | null; items: { index: number; option: MentionMenuOption }[] }[] = []
  options.forEach((option, index) => {
    const label = option.group ?? null
    const last = groups[groups.length - 1]
    if (last && last.label === label) last.items.push({ index, option })
    else groups.push({ items: [{ index, option }], label })
  })

  return (
    <div className="absolute bottom-full left-0 z-30 mb-2 w-max min-w-[17rem] max-w-md overflow-hidden rounded-xl border border-border bg-popover shadow-lg animate-in fade-in slide-in-from-bottom-1 motion-reduce:animate-none">
      <div className="flex items-center gap-1.5 border-b border-border px-2.5 py-1.5">
        {isRoot ? (
          <span className="text-[11px] font-medium text-muted-foreground">{labels.rootTitle}</span>
        ) : (
          <span className="flex min-w-0 items-center gap-1.5 text-[11px]">
            {canGoBack ? (
              <button
                aria-label={labels.backHint}
                className="-ml-1 flex shrink-0 items-center rounded p-0.5 text-muted-foreground/70 transition-colors hover:bg-accent hover:text-foreground"
                onMouseDown={(event) => {
                  event.preventDefault()
                  onBack?.()
                }}
                title={labels.backHint}
                type="button"
              >
                <ChevronLeft className="size-3.5" />
              </button>
            ) : null}
            <span className={cn('inline-flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 font-medium', scope.tone ? toneChip[scope.tone] : '')}>
              {ScopeIcon ? <ScopeIcon className="size-3" /> : null}
              {scope.kind}
            </span>
            <ChevronRight className="size-3 shrink-0 text-muted-foreground/40" />
            <span className="truncate text-muted-foreground">
              {scope.query || labels.filterPlaceholder}
            </span>
          </span>
        )}
        <span className="ml-auto shrink-0 text-[10px] tabular-nums text-muted-foreground/50">{options.length}</span>
      </div>

      <div className="max-h-64 overflow-y-auto py-1">
        {groups.map((group, groupIndex) => (
          <div key={group.label ?? `group-${groupIndex}`}>
            {group.label ? (
              <div className="px-2.5 pb-0.5 pt-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground/60">
                {group.label}
              </div>
            ) : null}
            {group.items.map(({ index, option }) => {
              const Icon = option.icon
              const active = index === activeIndex
              return (
                <button
                  className={cn(
                    'relative flex w-full min-w-0 items-center gap-2.5 px-2.5 py-1.5 text-left transition-colors',
                    active ? 'bg-accent' : 'hover:bg-accent/50',
                  )}
                  key={`${option.secondary}-${index}`}
                  onMouseDown={(event) => {
                    event.preventDefault()
                    onSelect(index)
                  }}
                  onMouseEnter={() => onHover(index)}
                  type="button"
                >
                  {active ? (
                    <span className={cn('absolute inset-y-1 left-0 w-0.5 rounded-full', toneBar[option.tone])} />
                  ) : null}
                  <Icon className={cn('size-4 shrink-0', active ? toneText[option.tone] : 'text-muted-foreground/70')} />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] font-semibold text-foreground">{option.primary}</span>
                    <span className="block truncate text-[11px] text-muted-foreground">{option.secondary}</span>
                  </span>
                  {option.isCategory ? (
                    <ChevronRight className="size-3.5 shrink-0 text-muted-foreground/40" />
                  ) : active ? (
                    <kbd className={kbd}>↵</kbd>
                  ) : null}
                </button>
              )
            })}
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3 border-t border-border bg-surface/40 px-2.5 py-1.5 text-[10px] text-muted-foreground/70">
        <span className="inline-flex items-center gap-1">
          <kbd className={kbd}>↑</kbd>
          <kbd className={kbd}>↓</kbd>
          {labels.navHint}
        </span>
        <span className="inline-flex items-center gap-1">
          <kbd className={kbd}>↵</kbd>
          {labels.selectHint}
        </span>
        {canGoBack ? (
          <span className="inline-flex items-center gap-1">
            <kbd className={kbd}>⌫</kbd>
            {labels.backHint}
          </span>
        ) : null}
        <span className="ml-auto inline-flex items-center gap-1">
          <kbd className={kbd}>Esc</kbd>
          {labels.closeHint}
        </span>
      </div>
    </div>
  )
}
