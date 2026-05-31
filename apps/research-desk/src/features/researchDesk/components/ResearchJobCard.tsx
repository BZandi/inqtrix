import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Globe2,
  Info,
  Repeat2,
  Search,
  Trash2,
  XCircle,
  type LucideIcon,
} from '@/components/icons'
import {
  Fragment,
  forwardRef,
  useEffect,
  useState,
  type MouseEvent,
  type ReactNode,
} from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'
import { localizedText, phaseOrder, type JobPhase, type ResearchJob } from '../types'
import {
  phaseIcon,
  phaseLabel,
  queuedPhaseIcon,
  statusBadgeClassName,
  statusIcon,
} from './runDisplay'

type ResearchJobCardProps = {
  cancelError?: string
  isExpanded: boolean
  isCancelSubmitting?: boolean
  isSelected: boolean
  job: ResearchJob
  onCancel: () => void
  onDelete: () => void
  onSelect: () => void
  onToggleExpanded: () => void
}

export const ResearchJobCard = forwardRef<HTMLElement, ResearchJobCardProps>(
  function ResearchJobCard({
    cancelError,
    isExpanded,
    isCancelSubmitting = false,
    isSelected,
    job,
    onCancel,
    onDelete,
    onSelect,
    onToggleExpanded,
  }, ref) {
    const { locale, t } = useLocale()
    const StatusIcon = statusIcon[job.status]
    const reduceMotion = useReducedMotion()
    const runningDuration = useRunningDuration(job.status, job.startedAtIso)
    const canCancel = job.status === 'running' || job.status === 'queued'
    const metadata = [
      `${t.runCard.jobId}: ${job.id}`,
      job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled' || job.status === 'expired'
        ? `${t.runCard.submitted}: ${job.submittedAt}`
        : `${t.runCard.started}: ${job.startedAt ?? job.submittedAt}`,
    ]

    if (job.duration) {
      metadata.push(`${t.runCard.duration}: ${job.duration}`)
    }
    if (job.status === 'queued' && job.queueNote) {
      metadata.push(localizedText(job.queueNote, locale))
    }
    if (job.status === 'running') {
      metadata.push(`${t.runCard.runtime}: ${runningDuration}`)
    }
    if (isCancelSubmitting) {
      metadata.push(t.runCard.cancelSubmitted)
    } else if (canCancel && job.cancelRequested) {
      metadata.push(t.runCard.cancelRequested)
    } else if (cancelError) {
      metadata.push(`${t.runCard.cancelFailed}: ${cancelError}`)
    }
    if (job.error) {
      metadata.push(job.error)
    }

    return (
      <motion.article
        ref={ref}
        layout="position"
        initial={reduceMotion ? false : { opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.98, y: -6 }}
        onClick={onSelect}
        transition={appMotion.card}
        className={cn(
          'relative cursor-default rounded-lg border bg-card p-4 shadow-[0_1px_2px_var(--shadow-hairline)] transition-shadow',
          isSelected
            ? 'border-brand shadow-[0_12px_30px_var(--brand-shadow)]'
            : 'border-border hover:shadow-[0_8px_24px_var(--shadow-soft)]',
          isSelected && 'mb-3',
        )}
      >
        <div className="grid grid-cols-[auto_minmax(0,1fr)] items-start gap-3 sm:grid-cols-[auto_minmax(0,1fr)_auto]">
          <StatusIcon
            className={cn(
              'mt-1 size-5 shrink-0',
              job.status === 'completed' && 'text-success',
              job.status === 'queued' && 'text-muted-foreground',
              job.status === 'running' && 'text-brand',
              job.status === 'failed' && 'text-destructive',
              job.status === 'cancelled' && 'text-destructive/80',
              job.status === 'expired' && 'text-muted-foreground',
              job.status === 'running' && !reduceMotion && 'animate-spin [animation-duration:1.6s]',
            )}
          />
          <div className="min-w-0">
            <h2 className="line-clamp-2 text-sm font-semibold leading-6 text-foreground md:text-base">
              {localizedText(job.title, locale)}
            </h2>
            <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
              {metadata.map((item) => (
                <span className="min-w-0 truncate" key={item}>
                  {item}
                </span>
              ))}
            </div>
            {job.status === 'running' && !isExpanded && (
              <RunningCompactStatus job={job} />
            )}
          </div>
          <div className="col-span-2 ml-8 flex items-center gap-1 sm:col-span-1 sm:ml-0">
            <Badge className={statusBadgeClassName[job.status]} variant="outline">
              {t.status[job.status]}
            </Badge>
            {canCancel && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    aria-label={isCancelSubmitting ? t.runCard.cancelSubmitted : t.runCard.cancel}
                    className="text-muted-foreground hover:text-destructive disabled:text-muted-foreground"
                    disabled={isCancelSubmitting || job.cancelRequested}
                    onClick={(event) => runCardAction(event, onCancel)}
                    size="icon"
                    type="button"
                    variant="ghost"
                  >
                    <XCircle className="size-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {isCancelSubmitting
                    ? t.runCard.cancelSubmitted
                    : job.cancelRequested
                      ? t.runCard.cancelRequested
                      : t.runCard.cancel}
                </TooltipContent>
              </Tooltip>
            )}
            <Button
              aria-label={t.runCard.delete}
              className="text-muted-foreground hover:text-destructive"
              onClick={(event) => runCardAction(event, onDelete)}
              size="icon"
              type="button"
              variant="ghost"
            >
              <Trash2 className="size-4" />
            </Button>
            <Button
              aria-label={t.runCard.open}
              onClick={(event) => runCardAction(event, onSelect)}
              size="icon"
              type="button"
              variant="ghost"
            >
              <ChevronRight className="size-5" />
            </Button>
          </div>
        </div>

        {isSelected && (
          <div
            className={cn(
              'grid overflow-hidden transition-[grid-template-rows] duration-200 ease-out motion-reduce:transition-none',
              isExpanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]',
            )}
          >
            <motion.div
              animate={isExpanded ? { opacity: 1, y: 0 } : { opacity: 0, y: -4 }}
              className="min-h-0 overflow-hidden"
              initial={false}
              transition={appMotion.card}
            >
              {job.status === 'running' ? (
                <RunningJobDetails job={job} />
              ) : (
                <CompactJobDetails job={job} />
              )}
            </motion.div>
          </div>
        )}

        {isSelected && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={isExpanded ? t.runCard.collapse : t.runCard.expand}
                className="absolute -bottom-3 left-1/2 h-5 w-12 -translate-x-1/2 rounded-b-md rounded-t-none border border-t-0 border-border bg-card text-muted-foreground shadow-[0_4px_10px_var(--shadow-hairline)] hover:bg-accent hover:text-foreground"
                onClick={(event) => runCardAction(event, onToggleExpanded)}
                size="icon"
                type="button"
                variant="ghost"
              >
                {isExpanded ? (
                  <ChevronUp className="size-4" />
                ) : (
                  <ChevronDown className="size-4" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {isExpanded ? t.runCard.collapse : t.runCard.expand}
            </TooltipContent>
          </Tooltip>
        )}
      </motion.article>
    )
  },
)

function runCardAction(
  event: MouseEvent<HTMLButtonElement>,
  action: () => void,
) {
  event.stopPropagation()
  action()
}

function RunningCompactStatus({ job }: { job: ResearchJob }) {
  const { locale, t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  const roundInfo = parseRoundMetric(job.metrics.rounds)
  const latestEvent = job.events.at(-1)
  const PhaseIcon = phaseIcon[job.activePhase]
  const MessageIcon = latestEvent?.severity === 'warning' || latestEvent?.severity === 'error'
    ? AlertTriangle
    : Info

  return (
    <div className="mt-2.5 min-w-0 rounded-md border border-border bg-surface/70 px-2.5 py-2">
      <div className="flex min-w-0 flex-wrap items-center gap-1.5">
        <span className="inline-flex h-7 min-w-0 items-center rounded-md border border-brand/20 bg-brand-subtle px-2 text-xs font-semibold text-brand">
          <AnimatePresence initial={false} mode="wait">
            <motion.span
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex min-w-0 items-center gap-1.5"
              exit={reduceMotion ? undefined : { opacity: 0, y: -5 }}
              initial={reduceMotion ? false : { opacity: 0, y: 5 }}
              key={job.activePhase}
              transition={appMotion.list}
            >
              <span className="relative flex size-5 shrink-0 items-center justify-center rounded-full bg-brand/10">
                {!reduceMotion && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-0 rounded-full border border-brand/35 animate-pulse"
                  />
                )}
                <PhaseIcon className="relative z-10 size-3.5" />
              </span>
              <span className="truncate">{phaseLabel(job.activePhase, t)}</span>
            </motion.span>
          </AnimatePresence>
        </span>

        <span className="inline-flex h-7 shrink-0 items-center gap-1 rounded-md border border-border bg-card px-2 text-xs font-semibold text-muted-foreground">
          <Repeat2 className="size-3.5" aria-hidden="true" />
          <span>{t.runCard.currentRound}</span>
          <strong className="font-semibold text-foreground">{roundInfo.current}</strong>
          {roundInfo.max && (
            <span className="font-semibold text-muted-foreground">/ {roundInfo.max}</span>
          )}
        </span>

        {job.metrics.sources > 0 && (
          <span className="inline-flex h-7 shrink-0 items-center gap-1 rounded-md border border-border bg-card px-2 text-xs font-semibold text-muted-foreground">
            <Globe2 className="size-3.5" aria-hidden="true" />
            <strong className="font-semibold text-foreground">{job.metrics.sources}</strong>
            <span>{t.runCard.sources}</span>
          </span>
        )}
      </div>

      {latestEvent && (
        <div className="mt-1.5 flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
          <MessageIcon
            className={cn(
              'size-3.5 shrink-0',
              latestEvent.severity === 'warning' && 'text-warning',
              latestEvent.severity === 'error' && 'text-destructive',
            )}
          />
          <span className="truncate">
            {localizedText(latestEvent.title, locale)}
          </span>
        </div>
      )}
      <CompactPhaseFlow
        activePhase={job.activePhase}
        completedPhases={job.completedPhases}
        reduceMotion={reduceMotion}
      />
    </div>
  )
}

function CompactPhaseFlow({
  activePhase,
  completedPhases,
  reduceMotion,
}: {
  activePhase: JobPhase
  completedPhases: readonly JobPhase[]
  reduceMotion: boolean
}) {
  return (
    <div aria-hidden="true" className="mt-2 flex min-w-0 items-center px-0.5">
      {phaseOrder.map((phase, index) => {
        const isActive = phase === activePhase
        const isDone = completedPhases.includes(phase)
        const nextPhase = phaseOrder[index + 1]
        const isConnectorActive = nextPhase === activePhase
        const isConnectorDone = Boolean(nextPhase && isDone)

        return (
          <Fragment key={phase}>
            <span
              className={cn(
                'relative flex size-4 shrink-0 items-center justify-center rounded-full border bg-card transition-colors',
                isDone && 'border-success/30 bg-success-subtle',
                isActive && 'border-brand/55 bg-brand-subtle shadow-[0_0_0_4px_var(--brand-subtle)]',
                isActive && !reduceMotion && 'inqtrix-active-node-shell',
                !isDone && !isActive && 'border-border bg-background',
              )}
            >
              {isActive && !reduceMotion && (
                <ActivePhasePulse compact />
              )}
              <span
                className={cn(
                  'relative z-10 size-1.5 rounded-full bg-muted-foreground/45',
                  isDone && 'bg-success',
                  isActive && 'bg-brand inqtrix-active-node-core',
                )}
              />
            </span>
            {index < phaseOrder.length - 1 && (
              <CompactFlowConnector
                isActive={isConnectorActive}
                isDone={isConnectorDone}
                reduceMotion={reduceMotion}
              />
            )}
          </Fragment>
        )
      })}
    </div>
  )
}

function CompactFlowConnector({
  isActive,
  isDone,
  reduceMotion,
}: {
  isActive: boolean
  isDone: boolean
  reduceMotion: boolean
}) {
  return (
    <span
      className={cn(
        'relative mx-1 h-px min-w-4 flex-1 overflow-hidden rounded-full bg-border',
        isDone && 'bg-brand/25',
      )}
    >
      {isActive && !reduceMotion && (
        <motion.span
          animate={{ x: ['-80%', '180%'] }}
          className="absolute inset-y-0 left-0 w-3/4 rounded-full bg-gradient-to-r from-transparent via-brand to-transparent shadow-[0_0_12px_var(--brand-shadow)]"
          transition={{ duration: 1.18, ease: [0.45, 0, 0.2, 1], repeat: Infinity }}
        />
      )}
      {isActive && reduceMotion && (
        <span className="absolute inset-0 rounded-full bg-brand/60" />
      )}
    </span>
  )
}

function RunningJobDetails({ job }: { job: ResearchJob }) {
  const { locale, t } = useLocale()
  const visibleEvents = job.events.slice(-4)

  return (
    <div className="mt-3 space-y-3">
      <RunningPhaseFlow
        activePhase={job.activePhase}
        completedPhases={job.completedPhases}
        phaseVisitCounts={job.phaseVisitCounts}
        rounds={job.metrics.rounds}
      />
      <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_270px]">
        <div className="rounded-lg border border-border bg-surface p-3">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-xs font-semibold text-foreground">
              {t.runCard.liveStatus}
            </h3>
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-success">
              <span className="size-1.5 rounded-full bg-success" />
              {t.runCard.live}
            </span>
          </div>
          <ol className="mt-2 space-y-1.5">
            {visibleEvents.map((event, index) => {
              const EventIcon = event.severity === 'warning' || event.severity === 'error'
                ? AlertTriangle
                : Info

              return (
                <li
                  className="grid min-w-0 grid-cols-[50px_14px_minmax(0,1fr)_auto] items-center gap-2 text-xs"
                  key={`${event.time}-${index}`}
                >
                  <span className="text-muted-foreground">{event.time}</span>
                  <EventIcon
                    className={cn(
                      'size-3.5',
                      event.severity === 'warning' && 'text-warning',
                      event.severity === 'error' && 'text-destructive',
                      event.severity !== 'warning'
                        && event.severity !== 'error'
                        && 'text-muted-foreground/75',
                    )}
                  />
                  <span
                    className={cn(
                      'truncate text-muted-foreground',
                      event.active && 'font-semibold text-foreground',
                    )}
                  >
                    {localizedText(event.title, locale)}
                  </span>
                  {event.active && (
                    <span className="flex gap-1 text-brand" aria-hidden="true">
                      <span className="size-1 rounded-full bg-brand animate-pulse" />
                      <span className="size-1 rounded-full bg-brand animate-pulse [animation-delay:120ms]" />
                      <span className="size-1 rounded-full bg-brand animate-pulse [animation-delay:240ms]" />
                    </span>
                  )}
                </li>
              )
            })}
          </ol>
        </div>

        <div className="rounded-lg border border-border bg-card p-3">
          <h3 className="mb-2 text-xs font-semibold text-foreground">
            {t.runCard.metrics}
          </h3>
          <MetricRow icon={Globe2} label={t.runCard.sources} value={job.metrics.sources} />
          <MetricRow icon={Search} label={t.runCard.queries} value={job.metrics.queries} />
          {job.confidence && (
            <MetricRow icon={CheckCircle2} label={t.common.confidence} value={job.confidence} />
          )}
        </div>
      </div>
    </div>
  )
}

function useRunningDuration(status: ResearchJob['status'], startedAtIso?: string) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (status !== 'running') return undefined

    const intervalId = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(intervalId)
  }, [status])

  if (status !== 'running' || !startedAtIso) return '00:00:00'
  return formatDuration((now - new Date(startedAtIso).getTime()) / 1000)
}

function CompactJobDetails({ job }: { job: ResearchJob }) {
  const { t } = useLocale()
  const QueuedIcon = queuedPhaseIcon

  if (job.status === 'queued') {
    return (
      <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
        {phaseOrder.map((phase) => (
          <span className="inline-flex items-center gap-1" key={phase}>
            <QueuedIcon className="size-3" />
            {phaseLabel(phase, t)}
          </span>
        ))}
      </div>
    )
  }

  return (
    <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
      <span>
        {t.runCard.sources} <strong className="text-foreground">{job.metrics.sources}</strong>
      </span>
      {job.score && (
        <span>
          {t.runCard.score}{' '}
          <strong className="font-semibold text-success">{job.score}</strong>
        </span>
      )}
    </div>
  )
}

function RunningPhaseFlow({
  activePhase,
  completedPhases,
  phaseVisitCounts,
  rounds,
}: {
  activePhase: JobPhase
  completedPhases: readonly JobPhase[]
  phaseVisitCounts: Record<JobPhase, number>
  rounds: number | string
}) {
  const { t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  const roundInfo = parseRoundMetric(rounds)

  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2.5">
      <div className="mb-2 flex items-center justify-between gap-3">
        <h3 className="text-xs font-semibold text-foreground">
          {t.runCard.flow}
        </h3>
        <div className="inline-flex h-7 shrink-0 items-center gap-1 rounded-md border border-brand/20 bg-brand-subtle px-2 text-xs text-brand">
          <Repeat2 className="size-3.5" aria-hidden="true" />
          <span className="font-medium">{t.runCard.currentRound}</span>
          <strong className="font-semibold">{roundInfo.current}</strong>
          {roundInfo.max && (
            <span className="font-semibold text-brand/70">/ {roundInfo.max}</span>
          )}
        </div>
      </div>
      <div className="md:hidden">
        <VerticalPhaseFlow
          activePhase={activePhase}
          completedPhases={completedPhases}
          phaseVisitCounts={phaseVisitCounts}
          reduceMotion={reduceMotion}
        />
      </div>
      <div className="hidden overflow-hidden pb-1 md:block">
        <div className="min-w-[560px]">
          <div className="flex items-start pt-4">
            {phaseOrder.map((phase, index) => {
              const nextPhase = phaseOrder[index + 1]
              return (
                <FlowStepGroup
                  activePhase={activePhase}
                  completedPhases={completedPhases}
                  key={phase}
                  phase={phase}
                  phaseVisitCounts={phaseVisitCounts}
                  reduceMotion={reduceMotion}
                  renderConnector={Boolean(nextPhase)}
                />
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

function VerticalPhaseFlow({
  activePhase,
  completedPhases,
  phaseVisitCounts,
  reduceMotion,
}: {
  activePhase: JobPhase
  completedPhases: readonly JobPhase[]
  phaseVisitCounts: Record<JobPhase, number>
  reduceMotion: boolean
}) {
  const { t } = useLocale()

  return (
    <div className="space-y-1">
      {phaseOrder.map((phase, index) => {
        const Icon = completedPhases.includes(phase) ? CheckCircle2 : phaseIcon[phase]
        const isActive = phase === activePhase
        const isDone = completedPhases.includes(phase)
        const isConnectorActive = phaseOrder[index + 1] === activePhase
        return (
          <div key={phase}>
            <div className="grid grid-cols-[32px_minmax(0,1fr)] items-center gap-2">
              <PhaseSignal isActive={isActive} isDone={isDone} reduceMotion={reduceMotion}>
                <PhaseVisitDots
                  count={phaseVisitCounts[phase] ?? 0}
                  isActive={isActive}
                  isDone={isDone}
                />
                <Icon className="relative z-10 size-3.5" />
              </PhaseSignal>
              <span
                className={cn(
                  'truncate text-[11px] font-semibold leading-4 text-muted-foreground',
                  (isActive || isDone) && 'text-foreground',
                  isActive && 'text-brand',
                )}
              >
                {phaseLabel(phase, t)}
              </span>
            </div>
            {index < phaseOrder.length - 1 && (
              <VerticalConnector isActive={isConnectorActive} reduceMotion={reduceMotion} />
            )}
          </div>
        )
      })}
    </div>
  )
}

function PhaseSignal({
  children,
  isActive,
  isDone,
  reduceMotion,
}: {
  children: ReactNode
  isActive: boolean
  isDone: boolean
  reduceMotion: boolean
}) {
  return (
    <span
      className={cn(
        'relative flex size-8 items-center justify-center rounded-full border bg-card text-muted-foreground transition-colors',
        isDone && 'border-success/25 bg-success-subtle text-success',
        isActive && 'border-brand/50 bg-brand-subtle text-brand shadow-[0_0_0_5px_var(--brand-subtle),0_10px_30px_var(--brand-shadow)]',
        isActive && !reduceMotion && 'inqtrix-active-node-shell',
      )}
    >
      {isActive && !reduceMotion && (
        <ActivePhasePulse />
      )}
      {children}
    </span>
  )
}

function ActivePhasePulse({ compact = false }: { compact?: boolean }) {
  return (
    <>
      <span
        aria-hidden="true"
        className={cn(
          'inqtrix-active-node-halo pointer-events-none absolute rounded-full',
          compact ? '-inset-2' : '-inset-2.5',
        )}
      />
      <span
        aria-hidden="true"
        className={cn(
          'inqtrix-active-node-ring pointer-events-none absolute rounded-full',
          compact ? '-inset-1' : '-inset-1.5',
        )}
      />
      <span
        aria-hidden="true"
        className={cn(
          'inqtrix-active-node-ring inqtrix-active-node-ring-delayed pointer-events-none absolute rounded-full',
          compact ? '-inset-1' : '-inset-1.5',
        )}
      />
    </>
  )
}

function PhaseVisitDots({
  count,
  isActive,
  isDone,
}: {
  count: number
  isActive: boolean
  isDone: boolean
}) {
  if (count <= 0) return null

  if (count > 6) {
    return (
      <span
        aria-hidden="true"
        className={cn(
          'absolute -top-4 left-1/2 z-20 flex h-3 min-w-4 -translate-x-1/2 items-center justify-center rounded-full px-1 text-[8px] font-semibold leading-none',
          isActive && 'bg-brand text-brand-foreground shadow-[0_0_0_2px_var(--brand-subtle)]',
          !isActive && isDone && 'bg-success-subtle text-success',
          !isActive && !isDone && 'bg-muted text-muted-foreground',
        )}
      >
        6+
      </span>
    )
  }

  return (
    <span
      aria-hidden="true"
      className="absolute -top-3.5 left-1/2 z-20 flex h-1.5 -translate-x-1/2 items-center gap-0.5"
    >
      {Array.from({ length: count }).map((_, index) => (
        <span
          className={cn(
            'size-1 rounded-full',
            isActive && 'bg-brand shadow-[0_0_0_2px_var(--brand-subtle)]',
            !isActive && isDone && 'bg-success/55',
            !isActive && !isDone && 'bg-muted-foreground/45',
          )}
          key={index}
        />
      ))}
    </span>
  )
}

function VerticalConnector({
  isActive,
  reduceMotion,
}: {
  isActive: boolean
  reduceMotion: boolean
}) {
  return (
    <span className="relative ml-4 block h-4 w-px overflow-hidden rounded-full bg-border">
      {isActive && !reduceMotion && (
        <motion.span
          aria-hidden="true"
          animate={{ y: ['-80%', '180%'] }}
          className="absolute left-0 top-0 h-3/4 w-px rounded-full bg-gradient-to-b from-transparent via-brand to-transparent shadow-[0_0_12px_var(--brand-shadow)]"
          transition={{ duration: 1.18, ease: [0.45, 0, 0.2, 1], repeat: Infinity }}
        />
      )}
      {isActive && reduceMotion && (
        <span className="absolute inset-0 rounded-full bg-brand/50" />
      )}
    </span>
  )
}

function FlowStepGroup({
  activePhase,
  completedPhases,
  phase,
  phaseVisitCounts,
  reduceMotion,
  renderConnector,
}: {
  activePhase: JobPhase
  completedPhases: readonly JobPhase[]
  phase: JobPhase
  phaseVisitCounts: Record<JobPhase, number>
  reduceMotion: boolean
  renderConnector: boolean
}) {
  const phaseIndex = phaseOrder.indexOf(phase)
  const nextPhase = phaseOrder[phaseIndex + 1]
  const isActive = phase === activePhase
  const isDone = completedPhases.includes(phase)
  const isConnectorActive = nextPhase === activePhase
  const isConnectorDone = Boolean(nextPhase && isDone)

  return (
    <>
      <FlowNode
        isActive={isActive}
        isDone={isDone}
        phase={phase}
        visitCount={phaseVisitCounts[phase] ?? 0}
        reduceMotion={reduceMotion}
      />
      {renderConnector && (
        <FlowConnector
          isActive={isConnectorActive}
          isDone={isConnectorDone}
          reduceMotion={reduceMotion}
        />
      )}
    </>
  )
}

function FlowNode({
  isActive,
  isDone,
  phase,
  reduceMotion,
  visitCount,
}: {
  isActive: boolean
  isDone: boolean
  phase: JobPhase
  reduceMotion: boolean
  visitCount: number
}) {
  const { t } = useLocale()
  const Icon = isDone ? CheckCircle2 : phaseIcon[phase]

  return (
    <div className="flex w-20 shrink-0 flex-col items-center gap-1.5 text-center">
      <PhaseSignal isActive={isActive} isDone={isDone} reduceMotion={reduceMotion}>
        <PhaseVisitDots count={visitCount} isActive={isActive} isDone={isDone} />
        <Icon className="relative z-10 size-3.5" />
      </PhaseSignal>
      <span
        className={cn(
          'max-w-full truncate text-[11px] font-semibold leading-4 text-muted-foreground',
          (isActive || isDone) && 'text-foreground',
          isActive && 'text-brand',
        )}
      >
        {phaseLabel(phase, t)}
      </span>
    </div>
  )
}

function FlowConnector({
  isActive,
  isDone,
  reduceMotion,
}: {
  isActive: boolean
  isDone: boolean
  reduceMotion: boolean
}) {
  return (
    <span
      className={cn(
        'relative mx-1 mt-4 h-px w-8 shrink-0 overflow-hidden rounded-full bg-border',
        isDone && 'bg-brand/25',
      )}
    >
      {isActive && !reduceMotion && (
        <motion.span
          aria-hidden="true"
          animate={{ x: ['-80%', '180%'] }}
          className="absolute inset-y-0 left-0 w-3/4 rounded-full bg-gradient-to-r from-transparent via-brand to-transparent shadow-[0_0_12px_var(--brand-shadow)]"
          transition={{ duration: 1.18, ease: [0.45, 0, 0.2, 1], repeat: Infinity }}
        />
      )}
      {isActive && reduceMotion && (
        <span className="absolute inset-0 rounded-full bg-brand/50" />
      )}
    </span>
  )
}

function parseRoundMetric(rounds: number | string) {
  const value = String(rounds)
  const match = value.match(/^\s*(\d+)\s*\/\s*(\d+)\s*$/)

  if (!match) {
    return { current: value, max: null }
  }

  return { current: match[1], max: match[2] }
}

function MetricRow({
  icon: Icon,
  label,
  value,
}: {
  icon: LucideIcon
  label: string
  value: number | string
}) {
  return (
    <div className="grid grid-cols-[20px_minmax(0,1fr)_auto] items-center gap-1.5 py-1 text-xs">
      <Icon className="size-4 text-muted-foreground" />
      <span className="truncate text-muted-foreground">{label}</span>
      <strong className="font-semibold text-foreground">{value}</strong>
    </div>
  )
}

function formatDuration(seconds: number) {
  const wholeSeconds = Math.max(0, Math.round(seconds))
  const hours = Math.floor(wholeSeconds / 3600)
  const minutes = Math.floor((wholeSeconds % 3600) / 60)
  const remainingSeconds = wholeSeconds % 60

  return [hours, minutes, remainingSeconds]
    .map((part) => part.toString().padStart(2, '0'))
    .join(':')
}
