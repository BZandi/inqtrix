import {
  Check,
  ListFilter,
  MessageSquarePlus,
  PanelBottomClose,
} from '@/components/icons'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type { JobFilter, ResearchJob } from '../types'

type JobFilterMenuProps = {
  activeFilter: JobFilter
  isComposerVisible: boolean
  jobs: ResearchJob[]
  onActiveFilterChange: (filter: JobFilter) => void
  onComposerVisibleChange: (isComposerVisible: boolean) => void
}

export function JobFilterMenu({
  activeFilter,
  isComposerVisible,
  jobs,
  onActiveFilterChange,
  onComposerVisibleChange,
}: JobFilterMenuProps) {
  const { t } = useLocale()
  const filters = buildFilterOptions(jobs, t.home.tabs)
  const selectedFilter = filters.find((filter) => filter.key === activeFilter) ?? filters[0]

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-2">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={t.home.filterLabel}
            className="h-9 gap-2 px-3 focus-visible:ring-border data-[state=open]:border-border data-[state=open]:bg-background data-[state=open]:shadow-sm data-[state=open]:ring-1 data-[state=open]:ring-border"
            type="button"
            variant="outline"
          >
            <ListFilter className="size-4" />
            <span>{selectedFilter.label}</span>
            <span className="inline-flex min-w-5 items-center justify-center rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
              {selectedFilter.count}
            </span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-56">
          <DropdownMenuLabel className="text-xs text-muted-foreground">
            {t.home.filterLabel}
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          {filters.map((filter) => (
            <DropdownMenuItem
              className="gap-2"
              key={filter.key}
              onSelect={() => onActiveFilterChange(filter.key)}
            >
              <Check
                className={cn(
                  'size-4 text-brand opacity-0',
                  activeFilter === filter.key && 'opacity-100',
                )}
              />
              <span className="min-w-0 flex-1">{filter.label}</span>
              <span className="inline-flex min-w-5 items-center justify-center rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                {filter.count}
              </span>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            aria-controls="research-composer"
            aria-expanded={isComposerVisible}
            aria-label={isComposerVisible ? t.composer.hide : t.composer.show}
            className="h-8 shrink-0 gap-1.5 px-2.5 text-xs"
            onClick={() => onComposerVisibleChange(!isComposerVisible)}
            type="button"
            variant={isComposerVisible ? 'ghost' : 'outline'}
          >
            {isComposerVisible ? (
              <PanelBottomClose className="size-4" />
            ) : (
              <MessageSquarePlus className="size-4" />
            )}
            <span className="hidden md:inline">
              {isComposerVisible ? t.composer.hide : t.composer.show}
            </span>
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {isComposerVisible ? t.composer.hide : t.composer.show}
        </TooltipContent>
      </Tooltip>
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
