import { ListFilter } from '@/components/icons'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type { JobFilter, ResearchJob } from '../types'

type JobFilterMenuProps = {
  activeFilter: JobFilter
  jobs: ResearchJob[]
  onActiveFilterChange: (filter: JobFilter) => void
}

export function JobFilterMenu({
  activeFilter,
  jobs,
  onActiveFilterChange,
}: JobFilterMenuProps) {
  const { t } = useLocale()
  const filters = buildFilterOptions(jobs, t.home.tabs)

  return (
    <div className="flex h-12 shrink-0 items-center gap-1 overflow-x-auto border-b border-border px-4 [scrollbar-width:none]">
      <ListFilter className="size-3.5 shrink-0 text-muted-foreground" />
      {filters.map((filter) => (
        <FilterChip
          active={activeFilter === filter.key}
          count={filter.count}
          key={filter.key}
          label={filter.label}
          onClick={() => onActiveFilterChange(filter.key)}
        />
      ))}
    </div>
  )
}

function FilterChip({
  active,
  count,
  label,
  onClick,
}: {
  active: boolean
  count: number
  label: string
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'inline-flex h-6 shrink-0 items-center gap-1.5 rounded-full border px-2.5 text-[11px] font-medium transition-colors',
        active
          ? 'border-brand/40 bg-brand-subtle text-brand'
          : 'border-border bg-background text-muted-foreground hover:text-foreground',
      )}
      onClick={onClick}
      type="button"
    >
      {label}
      <span className={cn('text-[10px] tabular-nums', active ? 'text-brand/80' : 'text-muted-foreground/80')}>
        {count}
      </span>
    </button>
  )
}

function buildFilterOptions(
  jobs: ResearchJob[],
  labels: Record<JobFilter, string>,
) {
  return [
    { count: jobs.length, key: 'all', label: labels.all },
    {
      count: jobs.filter((job) => job.status === 'running').length,
      key: 'running',
      label: labels.running,
    },
    {
      count: jobs.filter((job) => job.status === 'queued').length,
      key: 'queued',
      label: labels.queued,
    },
    {
      count: jobs.filter((job) => job.status === 'cancelled').length,
      key: 'cancelled',
      label: labels.cancelled,
    },
    {
      count: jobs.filter((job) => job.status === 'completed').length,
      key: 'completed',
      label: labels.completed,
    },
  ] satisfies Array<{
    count: number
    key: JobFilter
    label: string
  }>
}
