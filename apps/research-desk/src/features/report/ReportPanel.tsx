import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Clock3,
  Download,
  FileText,
  Folder,
  Info,
  LoaderCircle,
  Maximize2,
  MessageSquarePlus,
  Minimize2,
  PanelRightClose,
  PanelRightOpen,
} from '@/components/icons'
import {
  useEffect,
  useRef,
  useState,
  type UIEvent,
} from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { ResearchRunRecord } from '@/features/project/types'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'
import {
  phaseLabel,
  statusBadgeClassName,
} from '../researchDesk/components/runDisplay'
import { MarkdownReport } from './MarkdownReport'

type ReportRestoreRailProps = {
  onShow: () => void
}

type ReportPanelProps = {
  isExpanded: boolean
  onExpandedChange: (isExpanded: boolean) => void
  onHide: () => void
  onUseReportInChat?: (runId: string) => void
  selectedRun: ResearchRunRecord | null
}

type ReportPanelMode =
  | 'completed-with-report'
  | 'completed-without-report'
  | 'empty'
  | 'queued'
  | 'running'
  | 'terminal'

type ReportPanelState = {
  markdown: string | null
  mode: ReportPanelMode
}

export function ReportRestoreRail({ onShow }: ReportRestoreRailProps) {
  const { t } = useLocale()

  return (
    <aside className="flex min-h-12 items-start justify-center rounded-lg border border-border bg-card p-1 shadow-[0_1px_2px_var(--shadow-hairline)] lg:min-h-0">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            aria-label={t.report.show}
            className="mt-1"
            onClick={onShow}
            size="icon"
            type="button"
            variant="ghost"
          >
            <PanelRightOpen className="size-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="left">{t.report.show}</TooltipContent>
      </Tooltip>
    </aside>
  )
}

export function ReportPanel({
  isExpanded,
  onExpandedChange,
  onHide,
  onUseReportInChat,
  selectedRun,
}: ReportPanelProps) {
  const { t } = useLocale()
  const reduceMotion = useReducedMotion()
  const panelState = resolveReportPanelState(selectedRun)
  const isRunningRun = panelState.mode === 'running'
  const isCompletedRun = panelState.mode === 'completed-with-report'
    || panelState.mode === 'completed-without-report'
  const visibleEventCount = selectedRun
    ? selectedRun.events.filter(isDisplayableAgentEvent).length
    : 0
  const panelTitle = isRunningRun
    ? t.report.agentStepsTitle
    : panelState.mode === 'queued'
      ? t.report.queueTitle
      : t.report.title
  const canUseReportInChat = Boolean(
    selectedRun && panelState.mode === 'completed-with-report' && onUseReportInChat,
  )

  return (
    <motion.aside
      initial={reduceMotion ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={appMotion.panel}
      className={cn(
        'min-h-0 w-full min-w-0 max-w-full overflow-hidden rounded-lg border border-border bg-card shadow-[0_1px_2px_var(--shadow-hairline)]',
        isExpanded
          ? 'fixed bottom-4 left-4 right-4 top-20 z-40 w-auto max-w-none shadow-[0_24px_80px_var(--shadow-soft)] md:bottom-5 md:left-[76px] md:right-5 xl:left-[88px] xl:right-8'
          : 'lg:h-full',
      )}
    >
      <Tabs defaultValue="preview" className="flex h-full min-h-[420px] flex-col lg:min-h-0">
        <div className="border-b border-border">
          <div className="flex h-11 items-center justify-between gap-3 px-3">
            <div className="min-w-0">
              <h2 className="truncate text-sm font-semibold text-foreground">
                {panelTitle}
              </h2>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    aria-label={t.report.useInChat}
                    className="size-8"
                    disabled={!canUseReportInChat}
                    onClick={() => {
                      if (!selectedRun || !canUseReportInChat) return
                      onUseReportInChat?.(selectedRun.runId)
                    }}
                    size="icon"
                    type="button"
                    variant="ghost"
                  >
                    <MessageSquarePlus className="size-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{t.report.useInChat}</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    aria-label={t.report.hide}
                    className="size-8"
                    onClick={onHide}
                    size="icon"
                    type="button"
                    variant="ghost"
                  >
                    <PanelRightClose className="size-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{t.report.hide}</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    aria-label={isExpanded ? t.report.collapse : t.report.expand}
                    className="size-8"
                    onClick={() => onExpandedChange(!isExpanded)}
                    size="icon"
                    type="button"
                    variant="ghost"
                  >
                    {isExpanded ? (
                      <Minimize2 className="size-3.5" />
                    ) : (
                      <Maximize2 className="size-3.5" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {isExpanded ? t.report.collapse : t.report.expand}
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
          {selectedRun && (
            <div className="border-t border-border px-4 py-2.5">
              <div className="min-w-0">
                <p className="truncate text-xs text-muted-foreground">
                  {selectedRun.summary.title}
                </p>
                {isRunningRun ? (
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <span className="inline-flex h-6 items-center gap-1.5 rounded-md border border-success/20 bg-success-subtle px-2 text-xs font-semibold text-success">
                      <span className="size-1.5 rounded-full bg-success" />
                      {t.report.live}
                    </span>
                    <span className="inline-flex h-6 items-center rounded-md border border-border bg-surface px-2 text-xs font-semibold text-muted-foreground">
                      {visibleEventCount} {t.report.events}
                    </span>
                  </div>
                ) : (
                  panelState.mode === 'completed-with-report' && selectedRun.source === 'mock' && (
                    <Badge
                      className="mt-1.5 shrink-0 border-border bg-muted text-muted-foreground hover:bg-muted"
                      variant="outline"
                    >
                      {t.report.mockReport}
                    </Badge>
                  )
                )}
              </div>
              {isCompletedRun && (
                <div className="mt-3 flex min-w-0 items-center">
                  <TabsList className="h-9 bg-surface">
                    <TabsTrigger className="h-7 px-2 text-xs" value="preview">
                      {t.report.tabs.preview}
                    </TabsTrigger>
                    <TabsTrigger className="h-7 px-2 text-xs" value="evidence">
                      {t.report.tabs.evidence}
                    </TabsTrigger>
                    <TabsTrigger className="h-7 px-2 text-xs" value="agentSteps">
                      {t.report.tabs.agentSteps}
                    </TabsTrigger>
                    <TabsTrigger className="h-7 px-2 text-xs" value="export">
                      {t.report.tabs.export}
                    </TabsTrigger>
                  </TabsList>
                </div>
              )}
            </div>
          )}
        </div>

        <AnimatePresence initial={false} mode="wait">
          {panelState.mode === 'empty' ? (
            <motion.div
              animate={{ opacity: 1, y: 0 }}
              className="flex min-h-0 flex-1"
              exit={{ opacity: 0, y: -4 }}
              initial={{ opacity: 0, y: 4 }}
              key="empty"
              transition={appMotion.panel}
            >
              <EmptyReportPanel />
            </motion.div>
          ) : isCompletedRun && selectedRun ? (
            <motion.div
              animate={{ opacity: 1, y: 0 }}
              className="min-h-0 flex-1 overflow-hidden"
              exit={{ opacity: 0, y: -4 }}
              initial={{ opacity: 0, y: 4 }}
              key={`${selectedRun.runId}-${panelState.mode}`}
              transition={appMotion.panel}
            >
              <ScrollArea className="h-full min-h-0">
                <TabsContent className="m-0 w-full min-w-0 max-w-full overflow-hidden p-4" value="preview">
                  <ReportPreview run={selectedRun} markdown={panelState.markdown} />
                </TabsContent>
                <TabsContent className="m-0 w-full min-w-0 max-w-full overflow-hidden p-4" value="evidence">
                  <ReportEvidence run={selectedRun} />
                </TabsContent>
                <TabsContent className="m-0 w-full min-w-0 max-w-full overflow-hidden p-4" value="agentSteps">
                  <ReportAgentSteps run={selectedRun} />
                </TabsContent>
                <TabsContent className="m-0 w-full min-w-0 max-w-full overflow-hidden p-4" value="export">
                  <ReportExport markdown={panelState.markdown} run={selectedRun} />
                </TabsContent>
              </ScrollArea>
            </motion.div>
          ) : selectedRun ? (
            <motion.div
              animate={{ opacity: 1, y: 0 }}
              className="min-h-0 flex-1 overflow-hidden"
              exit={{ opacity: 0, y: -4 }}
              initial={{ opacity: 0, y: 4 }}
              key={`${selectedRun.runId}-${panelState.mode}`}
              transition={appMotion.panel}
            >
              {isRunningRun ? (
                <RunStatusPanel run={selectedRun} />
              ) : (
                <ScrollArea className="h-full min-h-0">
                  <RunStatusPanel run={selectedRun} />
                </ScrollArea>
              )}
            </motion.div>
          ) : (
            <motion.div
              animate={{ opacity: 1, y: 0 }}
              className="flex min-h-0 flex-1"
              exit={{ opacity: 0, y: -4 }}
              initial={{ opacity: 0, y: 4 }}
              key="fallback-empty"
              transition={appMotion.panel}
            >
              <EmptyReportPanel />
            </motion.div>
          )}
        </AnimatePresence>
      </Tabs>
    </motion.aside>
  )
}

function resolveReportPanelState(selectedRun: ResearchRunRecord | null): ReportPanelState {
  if (!selectedRun) {
    return { markdown: null, mode: 'empty' }
  }

  if (selectedRun.status === 'queued') {
    return { markdown: null, mode: 'queued' }
  }

  if (selectedRun.status === 'running') {
    return { markdown: null, mode: 'running' }
  }

  if (selectedRun.status !== 'completed') {
    return { markdown: null, mode: 'terminal' }
  }

  const markdown = selectedRun.result?.markdown ?? null

  return markdown
    ? { markdown, mode: 'completed-with-report' }
    : { markdown: null, mode: 'completed-without-report' }
}

function ReportPreview({
  markdown,
  run,
}: {
  markdown: string | null
  run: ResearchRunRecord
}) {
  const { t } = useLocale()
  const [copyFeedback, setCopyFeedback] = useState<'copied' | 'idle'>('idle')
  const copyResetRef = useRef<ReturnType<typeof window.setTimeout> | null>(null)

  useEffect(() => {
    return () => {
      if (copyResetRef.current) {
        window.clearTimeout(copyResetRef.current)
      }
    }
  }, [])

  async function handleCopyMarkdown() {
    if (!markdown) return

    if (copyResetRef.current) {
      window.clearTimeout(copyResetRef.current)
    }

    try {
      await copyTextToClipboard(markdown)
      setCopyFeedback('copied')
      copyResetRef.current = window.setTimeout(() => {
        setCopyFeedback('idle')
        copyResetRef.current = null
      }, 1600)
    } catch {
      setCopyFeedback('idle')
    }
  }

  if (markdown) {
    const copyLabel = copyFeedback === 'copied'
      ? t.report.copiedMarkdown
      : t.report.copyMarkdown

    return (
      <article className="group relative w-full min-w-0 max-w-full overflow-hidden [overflow-wrap:anywhere]">
        <div className="sticky top-2 z-20 flex h-0 justify-end pr-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={copyLabel}
                className={cn(
                  'pointer-events-auto size-8 border border-border/70 bg-card/85 text-muted-foreground opacity-0 shadow-[0_8px_24px_var(--shadow-soft)] backdrop-blur transition hover:bg-surface hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100',
                  copyFeedback === 'copied' && '!opacity-100 border-success/30 bg-success-subtle text-success hover:bg-success-subtle hover:text-success',
                )}
                onClick={() => void handleCopyMarkdown()}
                size="icon"
                type="button"
                variant="ghost"
              >
                {copyFeedback === 'copied' ? (
                  <CheckCircle2 className="size-4" />
                ) : (
                  <Copy className="size-4" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{copyLabel}</TooltipContent>
          </Tooltip>
        </div>
        <span aria-live="polite" className="sr-only">
          {copyFeedback === 'copied' ? t.report.copiedMarkdown : ''}
        </span>
        <MarkdownReport markdown={markdown} />
      </article>
    )
  }

  return (
    <div className="flex min-h-[340px] items-center justify-center p-4 text-center">
      <div className="max-w-md">
        <div className="mx-auto flex size-14 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <FileText className="size-7" />
        </div>
        <h3 className="mt-5 text-base font-semibold text-foreground">
          {run.summary.title}
        </h3>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          {t.report.unavailableDescription}
        </p>
      </div>
    </div>
  )
}

async function copyTextToClipboard(text: string) {
  if (copyTextWithTextArea(text)) {
    return
  }

  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text)
    return
  }

  throw new Error('Unable to copy markdown')
}

function copyTextWithTextArea(text: string) {
  const textArea = document.createElement('textarea')
  textArea.value = text
  textArea.setAttribute('readonly', 'true')
  textArea.style.opacity = '0'
  textArea.style.position = 'fixed'
  textArea.style.top = '0'

  document.body.appendChild(textArea)
  textArea.focus()
  textArea.select()
  textArea.setSelectionRange(0, text.length)

  const didCopy = document.execCommand('copy')
  document.body.removeChild(textArea)

  return didCopy
}

function RunStatusPanel({ run }: { run: ResearchRunRecord }) {
  if (run.status === 'queued') {
    return <QueuedRunPanel run={run} />
  }
  if (run.status !== 'running') {
    return <TerminalRunPanel run={run} />
  }

  return <LiveRunPanel run={run} />
}

function LiveRunPanel({ run }: { run: ResearchRunRecord }) {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <LiveRunOverview run={run} />
      <div className="min-h-0 flex-1 overflow-hidden">
        <AgentEventList run={run} />
      </div>
    </div>
  )
}

function LiveRunOverview({ run }: { run: ResearchRunRecord }) {
  const { t } = useLocale()
  const snapshot = run.snapshot

  return (
    <section className="border-b border-border px-4 py-3">
      <div className="grid gap-2 sm:grid-cols-2">
        <LiveRunMetric label={t.report.currentPhase} value={phaseLabel(run.phaseState.activePhase, t)} />
        <LiveRunMetric label={t.runCard.rounds} value={run.metrics.rounds} />
        <LiveRunMetric label={t.runCard.sources} value={run.metrics.sources} />
        <LiveRunMetric label={t.runCard.queries} value={run.metrics.queries} />
        {snapshot?.confidence ? (
          <LiveRunMetric label={t.common.confidence} value={`${snapshot.confidence} / 10`} />
        ) : null}
        {snapshot?.claim_quality_score !== undefined ? (
          <LiveRunMetric label={t.report.claimQuality} value={formatScore(snapshot.claim_quality_score)} />
        ) : null}
      </div>
    </section>
  )
}

function LiveRunMetric({
  label,
  value,
}: {
  label: string
  value: number | string
}) {
  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2">
      <p className="text-[11px] font-semibold uppercase tracking-normal text-muted-foreground">
        {label}
      </p>
      <p className="mt-1 truncate text-sm font-semibold text-foreground">{value}</p>
    </div>
  )
}

function QueuedRunPanel({ run }: { run: ResearchRunRecord }) {
  const { t } = useLocale()

  return (
    <div className="p-4">
      <section className="rounded-lg border border-dashed border-border bg-background p-6">
        <div className="flex flex-col items-center text-center">
          <div className="flex size-14 items-center justify-center rounded-full bg-muted text-muted-foreground">
            <Clock3 className="size-7" />
          </div>
          <Badge className={cn('mt-4 gap-1.5', statusBadgeClassName[run.status])} variant="outline">
            <Clock3 className="size-3.5" />
            {t.status[run.status]}
          </Badge>
          <h3 className="mt-4 text-base font-semibold leading-7 text-foreground [overflow-wrap:anywhere]">
            {t.report.waitingTitle}
          </h3>
          <p className="mt-2 max-w-md text-sm leading-6 text-muted-foreground">
            {run.summary.queueNote ?? t.report.queueDescription}
          </p>
        </div>
      </section>
    </div>
  )
}

function AgentEventList({ run }: { run: ResearchRunRecord }) {
  return <AgentStepTimeline mode="live" run={run} />
}

function TerminalRunPanel({ run }: { run: ResearchRunRecord }) {
  return (
    <div>
      <LiveRunOverview run={run} />
      <section className="px-4 py-3">
        <AgentStepTimeline mode="archive" run={run} />
      </section>
    </div>
  )
}

function ReportAgentSteps({ run }: { run: ResearchRunRecord }) {
  const { t } = useLocale()

  return (
    <div className="space-y-3">
      <section className="rounded-md border border-border bg-background p-3">
        <h3 className="text-sm font-semibold text-foreground">
          {t.report.agentStepsTitle}
        </h3>
        <p className="mt-1 text-sm leading-6 text-muted-foreground">
          {t.report.agentStepsArchiveDescription}
        </p>
      </section>
      <AgentStepTimeline mode="archive" run={run} />
    </div>
  )
}

function AgentStepTimeline({
  mode,
  run,
}: {
  mode: 'archive' | 'live'
  run: ResearchRunRecord
}) {
  const { t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  const isLive = mode === 'live'
  const events = run.events.filter(isDisplayableAgentEvent)
  const activeEventIndex = isLive ? events.findIndex((event) => event.active) : -1
  const derivedActiveEventIndex = isLive
    ? activeEventIndex >= 0
      ? activeEventIndex
      : events.length - 1
    : -1
  const activeStepRef = useRef<HTMLLIElement | null>(null)
  const logScrollRef = useRef<HTMLDivElement | null>(null)
  const userNavigatedLogRef = useRef(false)
  const [autoFollow, setAutoFollow] = useState(true)
  const [showJumpButton, setShowJumpButton] = useState(false)

  useEffect(() => {
    const activeStep = activeStepRef.current
    const scrollContainer = logScrollRef.current

    if (!isLive || !activeStep || !scrollContainer || !autoFollow) return

    scrollContainer.scrollTo({
      top: Math.max(
        activeStep.offsetTop - scrollContainer.clientHeight + activeStep.offsetHeight + 18,
        0,
      ),
      behavior: 'smooth',
    })
  }, [autoFollow, derivedActiveEventIndex, events.length, isLive])

  function handleEventScroll(event: UIEvent<HTMLDivElement>) {
    if (!isLive || !userNavigatedLogRef.current) return

    const target = event.currentTarget
    const distanceFromBottom = target.scrollHeight - target.scrollTop - target.clientHeight
    const isNearBottom = distanceFromBottom < 56

    setAutoFollow(isNearBottom)
    setShowJumpButton(!isNearBottom)
  }

  function markManualLogNavigation() {
    if (!isLive) return
    userNavigatedLogRef.current = true
  }

  function jumpToCurrentStep() {
    const activeStep = activeStepRef.current
    const scrollContainer = logScrollRef.current

    userNavigatedLogRef.current = false
    setAutoFollow(true)
    setShowJumpButton(false)

    if (!activeStep || !scrollContainer) return

    scrollContainer.scrollTo({
      top: Math.max(
        activeStep.offsetTop - scrollContainer.clientHeight + activeStep.offsetHeight + 18,
        0,
      ),
      behavior: 'smooth',
    })
  }

  return (
    <section
      className={cn(
        'relative',
        isLive && 'flex h-full min-h-0 flex-col overflow-hidden px-4 py-3',
      )}
      onPointerDown={markManualLogNavigation}
      onTouchMove={markManualLogNavigation}
      onWheel={markManualLogNavigation}
    >
      {events.length > 0 ? (
        <div className={cn('relative', isLive && 'min-h-0 flex-1 overflow-hidden')}>
          <div
            className={cn(
              isLive
                ? [
                  'h-full min-h-0 overflow-y-auto overscroll-contain pr-2 [scrollbar-gutter:stable] [scrollbar-width:thin]',
                  '[scrollbar-color:color-mix(in_oklch,var(--muted-foreground)_20%,transparent)_transparent]',
                  '[&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent',
                  '[&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border/50',
                  'hover:[&::-webkit-scrollbar-thumb]:bg-muted-foreground/30',
                ]
                : 'overflow-visible',
            )}
            onScroll={handleEventScroll}
            ref={logScrollRef}
          >
            <ol className="space-y-1">
              {events.map((event, index) => {
                const isActive = isLive && index === derivedActiveEventIndex
                const isDone = isLive && index < derivedActiveEventIndex
                const isWarning = event.severity === 'warning'
                const isError = event.severity === 'error'
                const showAlert = isWarning || isError

                return (
                  <li
                    className="grid grid-cols-[28px_minmax(0,1fr)] gap-3"
                    key={event.id}
                    ref={isActive ? activeStepRef : undefined}
                  >
                    <div className="relative flex justify-center pt-0.5">
                      {index < events.length - 1 && (
                        <span
                          className={cn(
                            'absolute bottom-[-10px] top-7 w-px bg-border',
                            isActive && 'bg-brand/25',
                          )}
                        />
                      )}
                      <span
                        className={cn(
                          'relative z-10 flex size-5 items-center justify-center rounded-full border bg-card text-muted-foreground transition-colors',
                          (isDone || !isLive) && !showAlert && 'border-transparent bg-transparent',
                          isActive && !showAlert && 'border-brand/35 bg-brand-subtle text-brand shadow-[0_0_0_3px_var(--brand-subtle)]',
                          isWarning && 'border-warning/30 bg-warning-subtle text-warning',
                          isError && 'border-destructive/30 bg-destructive/10 text-destructive',
                          !isDone && !isActive && !showAlert && 'border-border',
                        )}
                      >
                        {isActive && !showAlert && !reduceMotion && (
                          <motion.span
                            aria-hidden="true"
                            animate={{ opacity: [0.45, 0, 0.45], scale: [1, 1.55, 1] }}
                            className="absolute inset-0 rounded-full border border-brand/35"
                            transition={{ duration: 1.8, ease: 'easeInOut', repeat: Infinity }}
                          />
                        )}
                        {showAlert ? (
                          <AlertTriangle className="relative size-3.5" />
                        ) : isActive ? (
                          <LoaderCircle
                            className={cn(
                              'relative size-3.5',
                              !reduceMotion && 'animate-spin [animation-duration:1.5s]',
                            )}
                          />
                        ) : (
                          <span
                            className={cn(
                              'relative size-1.5 rounded-full bg-muted-foreground/40',
                              isDone && 'bg-muted-foreground/35',
                            )}
                          />
                        )}
                      </span>
                    </div>
                    <div
                      className={cn(
                        'min-w-0 pb-4 text-sm',
                        isActive && 'rounded-md bg-brand-subtle/35 px-2.5 py-2 shadow-[inset_2px_0_0_var(--brand)]',
                        isWarning && 'rounded-md bg-warning-subtle/45 px-2.5 py-2 shadow-[inset_2px_0_0_var(--warning)]',
                        isError && 'rounded-md bg-destructive/10 px-2.5 py-2 shadow-[inset_2px_0_0_var(--destructive)]',
                      )}
                    >
                      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-start gap-3">
                        <p
                          className={cn(
                            'leading-6 text-foreground [overflow-wrap:anywhere]',
                            isLive && !isActive && !isDone && 'text-muted-foreground',
                          )}
                        >
                          {displayAgentEventTitle(event.title, t)}
                        </p>
                        <span className="flex items-center gap-1 pt-1 text-[11px] font-medium text-muted-foreground">
                          {event.severity === 'warning' ? (
                            <AlertTriangle className="size-3 text-warning" />
                          ) : event.severity === 'error' ? (
                            <AlertTriangle className="size-3 text-destructive" />
                          ) : (
                            <Info className="size-3 text-muted-foreground/70" />
                          )}
                          <time>{formatTime(event.createdAt)}</time>
                        </span>
                      </div>
                      {isActive && isLive && (
                        <span className="mt-1.5 inline-flex rounded-full bg-brand/10 px-2 py-0.5 text-[11px] font-semibold text-brand">
                          {t.report.activeEvent}
                        </span>
                      )}
                    </div>
                  </li>
                )
              })}
            </ol>
          </div>
          {showJumpButton && (
            <Button
              className="absolute bottom-3 right-4 gap-2 shadow-[0_8px_24px_var(--shadow-soft)]"
              onClick={jumpToCurrentStep}
              size="sm"
              type="button"
              variant="secondary"
            >
              <LoaderCircle className="size-3.5" />
              <span>{t.report.jumpToCurrent}</span>
            </Button>
          )}
        </div>
      ) : (
        <p className="mt-3 text-sm leading-6 text-muted-foreground">
          {isLive ? t.report.noEvents : t.report.agentStepsEmpty}
        </p>
      )}
    </section>
  )
}

function ReportEvidence({ run }: { run: ResearchRunRecord }) {
  const { t } = useLocale()
  const metrics = run.result?.metrics
  const references = run.result?.references ?? []
  const topSources = run.result?.topSources ?? []
  const topClaims = run.result?.topClaims ?? []
  const fallbackSources = references.length === 0 ? topSources : []

  if (metrics || references.length > 0 || fallbackSources.length > 0 || topClaims.length > 0) {
    return (
      <div className="space-y-4">
        {metrics && (
          <section className="grid gap-2 sm:grid-cols-2">
            <EvidenceMetric label={t.common.confidence} value={`${metrics.confidence ?? 0} / 10`} />
            <EvidenceMetric label={t.runCard.sources} value={metrics.total_citations ?? 0} />
            <EvidenceMetric label={t.report.sourceQuality} value={formatScore(metrics.sources?.quality_score)} />
            <EvidenceMetric label={t.report.claimQuality} value={formatScore(metrics.claims?.quality_score)} />
            <EvidenceMetric label={t.report.evidenceContract} value={metrics.evidence_contract_status ?? 'unknown'} />
            <EvidenceMetric label={t.report.aspectCoverage} value={formatPercent(metrics.aspect_coverage)} />
          </section>
        )}
        {references.length > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-foreground">{t.report.reportReferences}</h3>
            <ol className="mt-2 space-y-2">
              {references.map((reference, index) => (
                <li
                  className="rounded-md border border-border bg-background p-3 text-sm"
                  key={`${reference.url}-${index}`}
                >
                  <div className="flex min-w-0 items-start gap-2">
                    <span className="mt-0.5 inline-flex h-5 shrink-0 items-center rounded-sm border border-border bg-card px-1.5 text-[10px] font-semibold text-muted-foreground">
                      {reference.label}
                    </span>
                    <a
                      className="min-w-0 break-words font-medium text-foreground hover:text-brand"
                      href={reference.url}
                      rel="noreferrer"
                      target="_blank"
                    >
                      {reference.url}
                    </a>
                  </div>
                  <p className="mt-1 text-xs font-semibold text-muted-foreground">{reference.tier}</p>
                </li>
              ))}
            </ol>
          </section>
        )}
        {fallbackSources.length > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-foreground">{t.runCard.sources}</h3>
            <ol className="mt-2 space-y-2">
              {fallbackSources.map((source) => (
                <li
                  className="rounded-md border border-border bg-background p-3 text-sm"
                  key={source.url}
                >
                  <a
                    className="break-words font-medium text-foreground hover:text-brand"
                    href={source.url}
                    rel="noreferrer"
                    target="_blank"
                  >
                    {source.url}
                  </a>
                  <p className="mt-1 text-xs font-semibold text-muted-foreground">{source.tier}</p>
                </li>
              ))}
            </ol>
          </section>
        )}
        {topClaims.length > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-foreground">{t.report.claims}</h3>
            <ol className="mt-2 space-y-2">
              {topClaims.slice(0, 10).map((claim, index) => (
                <li
                  className="rounded-md border border-border bg-background p-3 text-sm leading-6"
                  key={`${claim.status}-${index}-${claim.text}`}
                >
                  <p className="text-foreground">{claim.text}</p>
                  <p className="mt-1 text-xs font-semibold text-muted-foreground">
                    {claim.status} · {claim.support_count} support / {claim.contradict_count} contradict
                  </p>
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-dashed border-border bg-background p-6 text-sm leading-6 text-muted-foreground">
      {t.report.evidencePending}
    </div>
  )
}

function displayAgentEventTitle(
  title: string,
  t: ReturnType<typeof useLocale>['t'],
) {
  if (title === 'Run completed') return t.report.runCompletedEvent
  if (title === 'Run cancelled') return t.report.runCancelledEvent
  if (title === 'Cancellation requested') return t.report.cancellationRequestedEvent
  if (title === 'Run failed') return t.report.runFailedEvent
  return title
}

function isDisplayableAgentEvent(event: ResearchRunRecord['events'][number]) {
  const title = event.title.trim()
  return !(
    /^run snapshot$/i.test(title)
    || /^queued$/i.test(title)
    || /^run started$/i.test(title)
    || /^started\s+\w+/i.test(title)
    || /^finished\s+\w+/i.test(title)
  )
}

function EvidenceMetric({
  label,
  value,
}: {
  label: string
  value: number | string
}) {
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <p className="text-xs font-semibold text-muted-foreground">{label}</p>
      <p className="mt-1 text-sm font-semibold text-foreground">{value}</p>
    </div>
  )
}

function formatScore(value: number | undefined) {
  if (value === undefined) return 'n/a'
  return value.toFixed(2)
}

function formatPercent(value: number | undefined) {
  if (value === undefined) return 'n/a'
  return `${Math.round(value * 100)}%`
}

function formatTime(iso: string) {
  return new Intl.DateTimeFormat('de-DE', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(iso))
}

function ReportExport({
  markdown,
  run,
}: {
  markdown: string | null
  run: ResearchRunRecord
}) {
  const { t } = useLocale()
  const hasReport = Boolean(markdown)

  function handleExportMarkdown() {
    if (!markdown) return

    downloadMarkdownFile(
      `${markdown.trimEnd()}\n`,
      reportMarkdownFileName(run),
    )
  }

  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <h3 className="text-sm font-semibold text-foreground">{t.report.exportTitle}</h3>
      <p className="mt-2 text-sm leading-6 text-muted-foreground">
        {hasReport ? t.report.exportReadyDescription : t.report.exportDescription}
      </p>
      <Button
        className="mt-4 gap-2"
        disabled={!hasReport}
        onClick={handleExportMarkdown}
        type="button"
        variant="outline"
      >
        <Download className="size-4" />
        <span>{t.report.exportMarkdown}</span>
      </Button>
    </div>
  )
}

function downloadMarkdownFile(markdown: string, fileName: string) {
  const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  link.style.display = 'none'

  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

function reportMarkdownFileName(run: ResearchRunRecord) {
  const date = run.submittedAt.slice(0, 10).replace(/-/g, '')
  const title = sanitizeFileNameSegment(run.summary.title)
  const runId = sanitizeFileNameSegment(run.runId)
  return `${date}_${runId}_${title}.md`
}

function sanitizeFileNameSegment(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 72) || 'report'
}

function EmptyReportPanel() {
  const { t } = useLocale()

  return (
    <div className="flex flex-1 items-center justify-center p-8 text-center">
      <div className="max-w-sm">
        <div className="mx-auto flex size-16 items-center justify-center rounded-full bg-brand-subtle text-brand">
          <Folder className="size-8" />
        </div>
        <h3 className="mt-5 text-lg font-semibold text-foreground">
          {t.report.emptyTitle}
        </h3>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          {t.report.emptyDescription}
        </p>
      </div>
    </div>
  )
}
