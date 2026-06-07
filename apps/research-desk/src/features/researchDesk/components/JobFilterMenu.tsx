import { ListFilter } from '@/components/icons'
import { Chip } from '@/components/ui/chip'
import { useLocale } from '@/i18n/LocaleProvider'
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
        <Chip
          active={activeFilter === filter.key}
          count={filter.count}
          key={filter.key}
          onClick={() => onActiveFilterChange(filter.key)}
        >
          {filter.label}
        </Chip>
      ))}
    </div>
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
