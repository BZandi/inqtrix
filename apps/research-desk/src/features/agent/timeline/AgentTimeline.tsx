import { useEffect, useMemo, useState } from 'react'
import type { MouseEvent } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import {
  AlertTriangle,
  BookSearch,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  ExternalLink,
  FileText,
  Globe2,
  ListChecks,
  PenLine,
  Search,
  X,
} from '@/components/icons'
import { Button } from '@/components/ui/button'
import { MarkdownRenderer } from '@/components/markdown/MarkdownRenderer'
import { MarkdownSelectionCopyMenu } from '@/components/markdown/MarkdownSelectionCopyMenu'
import { effortLevelLabel } from '@/lib/modelCard'
import { useLocale } from '@/i18n/LocaleProvider'
import { formatDuration, formatMessageTimestamp } from '@/lib/time'
import { withAiDisclosure } from '@/lib/aiDisclosure'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'
import type { CanvasViewDescriptor } from '@/features/canvas/types'
import type { TranslationDictionary } from '@/i18n/translations'
import { AgentActivityLine } from '../AgentPulseTrack'
import {
  canEditAgentRun,
  isActiveAgentRun,
  isGateAgentRun,
  type AgentArtifactRecord,
  type AgentRunRecord,
  type AgentStepEntry,
} from '../model'
import type {
  AgentApprovalDecisionRequest,
  AgentClarificationAnswerRequest,
} from '../types'
import type { PlanSourceInfo } from '../plan/sourceLabel'
import type { AgentPlanDraft } from '../plan/usePlanApproval'
import { clarificationAnswerSummary } from '../clarificationAnswers'
import {
  agentArtifactReferences,
  agentCitationLabelFromHref,
  linkifyAgentArtifactCitations,
  type AgentArtifactReference,
} from '../artifactCitations'
import {
  agentActivityIconKind,
  activityDisplayText,
  activityStepDisplayText,
} from '../activityPresentation'
import {
  agentTaskResultPreview,
  effectiveAgentTaskStatus,
} from '../plan/taskPresentation'
import { agentRunCompletionRecap } from '../runPresentation'
import { WebEvidenceSourceRow } from '../WebEvidenceSourceRow'
import { copyTextToClipboard } from '@/lib/clipboard'

export type AgentTimelineActions = {
  answerClarification: (
    runId: string,
    clarificationId: string,
    answer: AgentClarificationAnswerRequest,
  ) => Promise<unknown>
  decideApproval: (
    runId: string,
    approvalId: string,
    decision: AgentApprovalDecisionRequest,
  ) => Promise<unknown>
  onCancelRun: (runId: string) => void
  onOpenCanvas: (descriptor: CanvasViewDescriptor) => void
  applyPatch: (
    runId: string,
    patchId: string,
    expectedRevision: number,
  ) => Promise<
    | { kind: 'applied'; revision: number; appliedEditIds: string[] }
    | { kind: 'conflict'; currentRevision: number | null }
  >
  rejectPatch: (
    runId: string,
    patchId: string,
    note: string,
  ) => Promise<unknown>
  /** Shared per-run plan drafts (reducer-owned — one draft, two surfaces). */
  planDrafts: Record<string, AgentPlanDraft>
  setPlanDraft: (runId: string, draft: AgentPlanDraft | null) => void
  /** Collection titles + vector-backend label for the plan's per-task
   * source line (same data plane as the canvas plan view). */
  planSource: PlanSourceInfo
}

/**
 * One session turn as a chat TRANSCRIPT (plan B1): the user's question as
 * a bubble, the agent's steps as a running feed of one-liners (stepLog
 * joined with the control rows), decisions as user-side entries, and a
 * permanently mounted live status line while the run is active. Gates
 * are NOT rendered here — they live in the composer gate tray; the rich
 * surfaces (plan, Verlauf, memo) live in the canvas tabs.
 */
export function AgentRunTurn({
  actions,
  animateEntry = true,
  run,
  transportDegraded = false,
}: {
  actions: AgentTimelineActions
  /** False for runs that already existed when the workspace mounted: a
   * remounted history renders in place instead of replaying its entry
   * animation over the view-level entry. */
  animateEntry?: boolean
  run: AgentRunRecord
  /** Live updates are on the polling fallback — shown as
   * a visible hint, never a silent behavior change. */
  transportDegraded?: boolean
}) {
  const { locale, t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  const isActive = isActiveAgentRun(run.status)
  // Settled turns collapse their step feed to one summary line (plan B1
  // "✓ n Schritte") so the session history stays scannable; the ACTIVE
  // run always streams in full.
  const [historyExpanded, setHistoryExpanded] = useState(false)
  const showSteps = isActive || historyExpanded
  const memoId = run.artifactOrder.find(
    (artifactId) => run.artifacts[artifactId]?.kind === 'memo',
  )
  const answerId = run.artifactOrder.find(
    (artifactId) => run.artifacts[artifactId]?.kind === 'answer',
  )
  const answer = answerId ? run.artifacts[answerId] : undefined
  const tasks = (run.plan?.tasks ?? []).filter(
    (task) => task.toolKind !== 'synthesis',
  )
  const done = tasks.filter(
    (task) => effectiveAgentTaskStatus(task, run.taskStates[task.taskId]) === 'completed',
  ).length
  const failed = tasks.filter((task) => {
    const status = effectiveAgentTaskStatus(task, run.taskStates[task.taskId])
    return status === 'failed' || status === 'insufficient_evidence'
  })
  const lastTaskError = [...failed]
    .reverse()
    .map((task) => run.taskStates[task.taskId]?.error)
    .find(Boolean)
  const completionRecap = agentRunCompletionRecap(
    run,
    tasks,
    memoId ? run.artifacts[memoId] : undefined,
  )

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="space-y-2"
      initial={reduceMotion || !animateEntry ? false : { opacity: 0, y: 6 }}
      transition={appMotion.card}
    >
      {/* The question follows the chat mode's user-message pattern to
          the letter (meta line above, tone-aware bubble below) so Agent
          Desk and Chat read as ONE design language. */}
      <div className="flex min-w-0 justify-end">
        <div className="min-w-0 max-w-[min(72%,44rem)]">
          <div className="mb-1 flex flex-wrap items-center justify-end gap-x-2 gap-y-0.5 t-meta-sm font-semibold text-muted-foreground">
            <span>{t.chat.you}</span>
            <span className="whitespace-nowrap tabular-nums">
              {formatMessageTimestamp(run.createdAt, locale)}
            </span>
            {run.autonomy && (
              <span className="rounded-full border border-border bg-background px-1.5 py-px t-hint font-semibold">
                {autonomyLabel(run.autonomy, t)}
              </span>
            )}
            {run.depth === 'deep' && (
              <span className="rounded-full border border-brand/30 bg-brand-subtle px-1.5 py-px t-hint font-semibold text-brand">
                {t.agent.composer.deep}
              </span>
            )}
          </div>
          <div className="inqtrix-user-bubble rounded-lg px-3 py-2.5 text-sm leading-6 shadow-[0_1px_2px_var(--shadow-hairline)]">
            <p className="whitespace-pre-wrap break-words">{run.question}</p>
          </div>
        </div>
      </div>

      {run.status === 'completed' && (
        <RunCompletionRecap
          memoId={memoId}
          onOpenMemo={(artifactId) => actions.onOpenCanvas({
            artifactId,
            runId: run.runId,
            view: 'document',
          })}
          recap={completionRecap}
          t={t}
        />
      )}

      {!isActive && run.stepLog.length > 0 && (
        <button
          aria-expanded={historyExpanded}
          className="flex items-center gap-1.5 px-1 t-meta text-muted-foreground transition-colors hover:text-foreground"
          onClick={() => setHistoryExpanded((current) => !current)}
          type="button"
        >
          {run.status === 'completed' ? (
            <Check className="icon-xs shrink-0 text-success" />
          ) : (
            <X className="icon-xs shrink-0 text-muted-foreground" />
          )}
          {t.agent.timeline.stepsSummary.replace(
            '{count}',
            String(run.stepLog.length),
          )}
          <ChevronDown
            className={cn(
              'size-3 shrink-0 transition-transform',
              historyExpanded && 'rotate-180',
            )}
          />
        </button>
      )}
      {showSteps && run.stepLog.length > 0 && (
        <div className="space-y-1 px-1">
          {run.stepLog.map((entry, index) => (
            <StreamEntry
              actions={actions}
              entry={entry}
              isLatest={isActive && index === run.stepLog.length - 1}
              key={entry.seq}
              run={run}
              t={t}
            />
          ))}
        </div>
      )}

      {tasks.length > 0
        && (run.phase === 'execution' || run.phase === 'evidence') && (
        <button
          className="flex items-center gap-1.5 px-1 t-meta tabular-nums text-muted-foreground transition-colors hover:text-foreground"
          onClick={() =>
            actions.onOpenCanvas({ runId: run.runId, view: 'plan' })}
          type="button"
        >
          <ListChecks className="icon-xs" />
          {t.agent.timeline.stepsProgress
            .replace('{done}', String(done))
            .replace('{total}', String(tasks.length))}
        </button>
      )}

      {answer && (
        <AgentAnswerBlock actions={actions} answer={answer} run={run} t={t} />
      )}

      {(run.status === 'failed' || run.status === 'cancelled') && (
        <div className="flex items-start gap-2 px-1">
          <AlertTriangle className="mt-0.5 icon-sm shrink-0 text-destructive" />
          <div className="min-w-0">
            <p className="break-words t-meta text-destructive/90">
              {run.status === 'cancelled'
                ? t.agent.timeline.cancelled
                : run.error
                  ? t.agent.timeline.taskFailed.replace('{error}', run.error)
                  : t.agent.timeline.failure}
            </p>
            {tasks.length > 0 && (
              <p className="mt-0.5 t-hint tabular-nums text-muted-foreground">
                {t.agent.timeline.terminalTaskSummary
                  .replace('{done}', String(done))
                  .replace('{total}', String(tasks.length))
                  .replace('{failed}', String(failed.length))}
              </p>
            )}
            {!run.error && lastTaskError && (
              <p className="mt-0.5 break-words t-meta-sm text-destructive/90">
                {lastTaskError}
              </p>
            )}
          </div>
        </div>
      )}

      {isActive && transportDegraded && (
        <p className="flex items-center gap-1.5 px-1 t-hint text-muted-foreground">
          <AlertTriangle className="icon-xs shrink-0 text-warning" />
          {t.agent.timeline.transportDegraded}
        </p>
      )}
      {isActive && (
        <div className="flex items-center gap-2 px-1">
          <AgentActivityLine
            className="min-w-0 flex-1"
            gate={isGateAgentRun(run.status)}
            text={activityText(run, t)}
          />
          {canEditAgentRun(run) && (
            <Button
              aria-label={t.agent.composer.stop}
              className="size-5 shrink-0 text-muted-foreground hover:text-destructive"
              onClick={() => actions.onCancelRun(run.runId)}
              size="icon"
              type="button"
              variant="ghost"
            >
              <X className="icon-xs" />
            </Button>
          )}
        </div>
      )}
    </motion.div>
  )
}

function RunCompletionRecap({
  memoId,
  onOpenMemo,
  recap,
  t,
}: {
  memoId: string | undefined
  onOpenMemo: (artifactId: string) => void
  recap: ReturnType<typeof agentRunCompletionRecap>
  t: TranslationDictionary
}) {
  const facts = [
    recap.taskCount > 0
      ? t.agent.timeline.recapTasks
        .replace('{done}', String(recap.tasksCompleted))
        .replace('{total}', String(recap.taskCount))
      : '',
    recap.referenceCount > 0
      ? t.agent.timeline.recapReferences.replace(
        '{count}',
        String(recap.referenceCount),
      )
      : '',
    recap.elapsedSeconds !== undefined
      ? t.agent.timeline.recapElapsed.replace(
        '{duration}',
        formatDuration(recap.elapsedSeconds),
      )
      : '',
  ].filter(Boolean)
  const process = [
    recap.synthesized ? t.agent.timeline.recapSynthesized : '',
    recap.reviewed ? t.agent.timeline.recapReviewed : '',
  ].filter(Boolean)
  return (
    <section className="mx-1 border-l-2 border-success/55 py-1 pl-3">
      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
        <Check className="icon-sm shrink-0 text-success" />
        <p className="min-w-0 flex-1 t-list text-foreground/90">
          {t.agent.timeline.recapTitle}
        </p>
        {memoId && (
          <Button
            className="h-6 shrink-0 gap-1 bg-brand px-2 text-xs text-brand-foreground hover:bg-brand/90"
            onClick={() => onOpenMemo(memoId)}
            size="sm"
            type="button"
          >
            <FileText className="icon-xs" />
            {t.agent.timeline.openMemo}
          </Button>
        )}
      </div>
      {facts.length > 0 && (
        <p className="mt-1 t-meta tabular-nums text-muted-foreground">
          {facts.join(' · ')}
        </p>
      )}
      {process.length > 0 && (
        <p className="mt-0.5 t-meta-sm text-foreground/75">
          {process.join(' · ')}
        </p>
      )}
    </section>
  )
}

/** Quiet right-aligned receipt for an answer/approval already captured by the
 * gate. It is intentionally not a second user-message bubble. */
function DecisionReceipt({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-2 flex min-w-0 justify-end px-1">
      <div className="w-fit max-w-[min(78%,48rem)] space-y-1 border-r border-border/80 py-0.5 pr-2 text-right text-foreground/75">
        {children}
      </div>
    </div>
  )
}

/**
 * The chat-form deliverable, rendered FLAT like the chat mode's
 * assistant answers (ChatWorkspace pattern — meta header line, body
 * directly on the surface, NO bubble): only the user question keeps a
 * bubble, so Agent Desk and Chat read as one design language.
 */
function AgentAnswerBlock({
  actions,
  answer,
  run,
  t,
}: {
  actions: AgentTimelineActions
  answer: AgentArtifactRecord
  run: AgentRunRecord
  t: TranslationDictionary
}) {
  const { locale } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  const [copied, setCopied] = useState(false)
  const body = answer.contentMarkdown
  const writing = answer.status === 'writing'
  const ready = answer.status === 'ready'
  // The single citation path: the chat answer reuses the
  // canvas machinery \u2014 same refs, same linkify, same label semantics.
  // While writing, labels stay plain text (no chips, no sources yet).
  const references = useMemo(
    () => (writing ? [] : agentArtifactReferences(answer.refs)),
    [answer.refs, writing],
  )
  const linkedBody = useMemo(
    () =>
      body !== undefined && references.length > 0
        ? linkifyAgentArtifactCitations(body, references)
        : body,
    [body, references],
  )
  const onCitationClick = (event: MouseEvent<HTMLDivElement>) => {
    const anchor = (event.target as HTMLElement | null)?.closest('a')
    const label = agentCitationLabelFromHref(anchor?.getAttribute('href'))
    if (!label) return
    const reference = references.find((item) => item.label === label)
    if (!reference) return
    event.preventDefault()
    event.stopPropagation()
    actions.onOpenCanvas({
      artifactId: answer.artifactId,
      label,
      runId: run.runId,
      view: 'evidence',
    })
  }
  const copy = () => {
    if (body === undefined) return
    void copyTextToClipboard(
      withAiDisclosure(body, t.aiTransparency.exportNotice),
    ).then((copied) => {
      if (!copied) return
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    })
  }
  return (
    <div
      className="px-1"
      aria-live={writing ? 'polite' : undefined}
      data-testid="agent-answer"
    >
      <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 t-meta-sm font-semibold text-muted-foreground">
        <span>{t.chat.assistant}</span>
        <span className="font-normal">{t.agent.timeline.answerLabel}</span>
        {run.finishedAt && (
          <span className="whitespace-nowrap tabular-nums">
            {formatMessageTimestamp(run.finishedAt, locale)}
          </span>
        )}
        {run.modelResolution?.model && (
          <span
            className="rounded-full border border-border bg-background px-1.5 py-px t-hint font-medium"
            title={run.modelResolution.modelSource}
          >
            {run.modelResolution.model}
            {run.modelResolution.effort
              ? ` \u00b7 ${effortLevelLabel(run.modelResolution.effort)}`
              : ''}
          </span>
        )}
      </div>
      {body !== undefined ? (
        <MarkdownSelectionCopyMenu
          aiGenerated
          className="chat-markdown max-w-4xl text-sm leading-snug text-foreground"
          markdown={body}
          onClickCapture={onCitationClick}
        >
          <MarkdownRenderer
            isStreaming={writing}
            markdown={linkedBody ?? body}
            variant="chat"
          />
          {writing && (
            <span
              aria-hidden="true"
              className={cn(
                'ml-1 inline-block size-2 rounded-full bg-brand align-middle',
                !reduceMotion && 'animate-pulse',
              )}
            />
          )}
        </MarkdownSelectionCopyMenu>
      ) : (
        <span
          aria-hidden="true"
          className={cn(
            'inline-block size-2 rounded-full bg-brand align-middle',
            !reduceMotion && 'animate-pulse',
          )}
        />
      )}
      {answer.status === 'interrupted' && (
        <p className="mt-1 t-meta text-destructive" role="status">
          {t.agent.timeline.answerInterrupted}
        </p>
      )}
      {ready && references.length > 0 && (
        <AgentAnswerSources
          onOpenEvidence={(label) =>
            actions.onOpenCanvas({
              artifactId: answer.artifactId,
              label,
              runId: run.runId,
              view: 'evidence',
            })}
          references={references}
        />
      )}
      {body !== undefined && ready && (
        <div className="mt-1 flex items-center gap-1">
          <Button
            aria-label={t.agent.timeline.answerCopy}
            className="size-6 text-muted-foreground hover:text-foreground"
            onClick={copy}
            size="icon"
            type="button"
            variant="ghost"
          >
            {copied ? (
              <Check className="icon-xs text-success" />
            ) : (
              <Copy className="icon-xs" />
            )}
          </Button>
          <Button
            aria-label={t.agent.timeline.answerOpenCanvas}
            className="size-6 text-muted-foreground hover:text-foreground"
            onClick={() =>
              actions.onOpenCanvas({
                artifactId: answer.artifactId,
                runId: run.runId,
                view: 'document',
              })}
            size="icon"
            type="button"
            variant="ghost"
          >
            <ExternalLink className="icon-xs" />
          </Button>
        </div>
      )}
    </div>
  )
}

/**
 * Compact source list under the chat answer. Web references retain their
 * external URL and expose the same evidence-trail action/status as the Canvas;
 * Knowledge references open the evidence view directly.
 */
function AgentAnswerSources({
  onOpenEvidence,
  references,
}: {
  onOpenEvidence: (label: string) => void
  references: AgentArtifactReference[]
}) {
  const { t } = useLocale()
  return (
    <section
      className="mt-3 max-w-4xl border-t border-border/70 pt-2"
      data-testid="agent-sources"
    >
      <h3 className="t-meta-sm font-semibold text-muted-foreground">
        {t.knowledge.sources}
      </h3>
      <ul className="mt-1.5 space-y-1.5">
        {references.map((reference) => {
          const web = Boolean(reference.queryId)
            || Boolean(reference.url && isWebHref(reference.url))
          return (
            <li
              className={web ? undefined : 'flex min-w-0 items-start gap-2'}
              key={reference.label}
            >
              {web ? (
                <WebEvidenceSourceRow
                  onInspect={() => onOpenEvidence(reference.label)}
                  reference={{
                    ...reference,
                    domain: hostFromUrl(reference.url ?? ''),
                    key: reference.referenceId ?? reference.label,
                  }}
                />
              ) : (
                <>
                  <FileText className="mt-0.5 icon-sm shrink-0 text-muted-foreground/70" />
                  <span className="mt-0.5 shrink-0 t-mono text-muted-foreground">
                    {reference.label}
                  </span>
                  <button
                    className="min-w-0 flex-1 truncate text-left t-list text-foreground transition-colors hover:text-brand"
                    onClick={() => onOpenEvidence(reference.label)}
                    type="button"
                  >
                    {reference.title}
                  </button>
                </>
              )}
            </li>
          )
        })}
      </ul>
    </section>
  )
}

// Deliberately NO third-party favicon fetch here: loading
// icons.duckduckgo.com/ip3/<domain>.ico would leak every cited source
// hostname to an external service on view — against the product's
// data-minimization line (log pseudonymization, webUrl()). Globe2 it is.

function hostFromUrl(url: string): string | null {
  try {
    return new URL(url).hostname
  } catch {
    return null
  }
}

/** Same scheme guard as `webUrl()` (copyAnswer): tool-provided URLs are
 * trusted data, but only http(s) ever reaches `window.open`/`href`. */
function isWebHref(url: string): boolean {
  return /^https?:\/\//i.test(url)
}

/** One transcript line, dispatched by step kind. */
function StreamEntry({
  actions,
  entry,
  isLatest = false,
  run,
  t,
}: {
  actions: AgentTimelineActions
  entry: AgentStepEntry
  /** Newest line of an ACTIVE run — the only one that types itself. */
  isLatest?: boolean
  run: AgentRunRecord
  t: TranslationDictionary
}) {
  switch (entry.kind) {
    case 'phase': {
      const station = entry.phase
        ? (stationLabel(entry.phase, t) ?? null)
        : null
      if (!station) return null
      return (
        <p className="pt-1.5 t-caption uppercase tracking-wide text-muted-foreground/70">
          {station}
        </p>
      )
    }
    case 'activity': {
      const text = activityStepDisplayText(entry, t)
      if (!text) return null
      // Append-only protocol: a finished invocation KEEPS its row (the
      // observed bug was searches vanishing once their task settled) —
      // the row's own status drives the glyph, never the task status.
      const failed = entry.status === 'failed'
      const fallback = entry.fallback === true
      const settled = entry.status === 'completed'
      const displayText = failed && entry.error
        ? `${text} · ${entry.error}`
        : text
      return (
        <p
          className={cn(
            'flex items-start gap-1.5 t-meta',
            entry.taskId && 'pl-5',
            failed
              ? 'text-destructive/90'
              : fallback
                ? 'text-warning'
                : 'text-muted-foreground',
          )}
          data-testid="agent-activity-item"
        >
          {failed || fallback ? (
            <AlertTriangle className="mt-0.5 icon-xs shrink-0" />
          ) : settled ? (
            <Check className="mt-0.5 icon-xs shrink-0 text-success/80" />
          ) : (
            <ActivityGlyph operation={entry.activityOperation} />
          )}
          <span className="min-w-0 break-words">{displayText}</span>
        </p>
      )
    }
    case 'plan': {
      const label = (
        entry.autoApproved
          ? t.agent.timeline.planRevisedAuto
          : entry.version !== undefined && entry.version > 1
            ? t.agent.timeline.planRevised
            : t.agent.timeline.planProposed
      ).replace('{version}', String(entry.version ?? 1))
      return (
        <p className="flex items-center gap-1.5 t-meta text-foreground/85">
          <ListChecks className="icon-xs shrink-0 text-muted-foreground" />
          <span className="min-w-0 break-words">{label}</span>
          <button
            className="inline-flex shrink-0 items-center gap-1 text-brand transition-colors hover:text-brand/80"
            onClick={() =>
              actions.onOpenCanvas({ runId: run.runId, view: 'plan' })}
            type="button"
          >
            <ExternalLink className="icon-xs" />
            {t.agent.timeline.openInCanvas}
          </button>
        </p>
      )
    }
    case 'task_started':
    case 'task_finished':
    case 'task_failed': {
      const title =
        run.plan?.tasks.find((task) => task.taskId === entry.taskId)?.title
        ?? entry.taskId
        ?? ''
      const running =
        entry.kind === 'task_started'
        && run.taskStates[entry.taskId ?? '']?.status === 'running'
      // Started lines collapse into their outcome line once the task is
      // over — the transcript stays one line per finished step.
      if (entry.kind === 'task_started' && !running) return null
      const task = run.plan?.tasks.find((candidate) => candidate.taskId === entry.taskId)
      const effectiveStatus = task
        ? effectiveAgentTaskStatus(task, run.taskStates[task.taskId])
        : entry.kind === 'task_failed'
          ? 'failed'
          : entry.kind === 'task_finished'
            ? 'completed'
            : 'running'
      const insufficient = effectiveStatus === 'insufficient_evidence'
      const live = entry.taskId ? run.taskStates[entry.taskId] : undefined
      const result = task?.resultSummary || live?.resultSummary || ''
      const preview = result ? agentTaskResultPreview(title, result) : ''
      const terminal = entry.kind !== 'task_started'
      const glyph =
        insufficient ? (
          <AlertTriangle className="icon-xs shrink-0 text-warning" />
        ) : entry.kind === 'task_failed' ? (
          <X className="icon-xs shrink-0 text-destructive" />
        ) : entry.kind === 'task_finished' ? (
          <Check className="icon-xs shrink-0 text-success" />
        ) : (
          <span
            aria-hidden="true"
            className="size-1.5 shrink-0 rounded-full bg-brand inqtrix-running-dot"
          />
        )
      return (
        <button
          className="group relative flex w-full items-center gap-1.5 rounded py-1 pl-0.5 pr-6 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          onClick={() =>
            entry.taskId
            && actions.onOpenCanvas({
              runId: run.runId,
              taskId: entry.taskId,
              view: 'run',
            })}
          type="button"
        >
          {glyph}
          <span
            className={cn(
              'min-w-0 flex-1 break-words t-meta',
              insufficient
                ? 'text-warning'
                : entry.kind === 'task_failed'
                ? 'text-destructive/90'
                : 'text-foreground/85',
            )}
          >
            <span className="block font-medium">{title}</span>
            {terminal && (preview || entry.error || insufficient) && (
              <span className="mt-0.5 block line-clamp-2 pr-1 font-normal text-muted-foreground">
                {entry.error
                  ?? (insufficient
                    ? t.agent.task.statusInsufficientEvidence
                    : preview)}
              </span>
            )}
          </span>
          <ChevronRight
            aria-hidden="true"
            className="absolute right-1.5 top-1/2 icon-xs -translate-y-1/2 text-muted-foreground/55 transition-[color,transform] group-hover:translate-x-0.5 group-hover:text-foreground"
          />
        </button>
      )
    }
    case 'clarification_answered': {
      const clarification = run.clarifications.find(
        (item) => item.clarificationId === entry.clarificationId,
      )
      if (!clarification) return null
      const lines = clarificationAnswerSummary(clarification)
      if (lines.length === 0) return null
      return (
        <DecisionReceipt>
          {lines.map((line) => (
            <p
              className="break-words t-meta text-foreground/90"
              key={line.prompt}
            >
              <span className="text-muted-foreground">{line.prompt}</span>
              {` — ${line.answer}`}
            </p>
          ))}
        </DecisionReceipt>
      )
    }
    case 'approval_decided': {
      const approval = run.approvals.find(
        (item) => item.approvalId === entry.approvalId,
      )
      const label = decisionLabel(
        approval?.kind ?? 'plan',
        entry.detail ?? approval?.status ?? '',
        t,
      )
      if (!label) return null
      return (
        <DecisionReceipt>
          <p className="flex items-center justify-end gap-1.5 t-meta text-foreground/80">
            <PenLine className="icon-xs shrink-0 text-muted-foreground" />
            {label}
          </p>
        </DecisionReceipt>
      )
    }
    case 'narration':
      if (!entry.text) return null
      return <NarrationText animate={isLatest} text={entry.text} />
    default:
      return null
  }
}

/**
 * Narration prose with the "writes itself" feel (plan B2): the NEWEST
 * line of an active run types in; history and replays render instantly
 * (the events are persisted paragraphs, the animation is presentation).
 */
function NarrationText({ animate, text }: { animate: boolean; text: string }) {
  const reduceMotion = Boolean(useReducedMotion())
  const typing = animate && !reduceMotion
  const [visible, setVisible] = useState(typing ? 0 : text.length)
  useEffect(() => {
    if (!typing) {
      setVisible(text.length)
      return undefined
    }
    if (visible >= text.length) return undefined
    const id = window.setTimeout(
      () => setVisible((count) => Math.min(count + 3, text.length)),
      18,
    )
    return () => window.clearTimeout(id)
  }, [text, typing, visible])
  return (
    <p className="whitespace-pre-wrap break-words py-0.5 text-sm leading-snug text-foreground/90">
      {text.slice(0, visible)}
      {typing && visible < text.length && (
        <span aria-hidden="true" className="text-brand">
          ▍
        </span>
      )}
    </p>
  )
}

/** German decision line for the transcript (rows carry kind + status). */
function decisionLabel(
  kind: string,
  status: string,
  t: TranslationDictionary,
): string | null {
  const map = t.agent.timeline.decisions as Record<
    string,
    Record<string, string> | undefined
  >
  const byKind = map[kind] ?? map.plan
  if (!byKind) return null
  return byKind[status] ?? byKind.approved ?? null
}

function ActivityGlyph({
  operation,
}: {
  operation: AgentStepEntry['activityOperation']
}) {
  const kind = agentActivityIconKind(operation)
  if (kind === 'web') {
    return <Globe2 className="mt-0.5 icon-xs shrink-0" />
  }
  if (kind === 'knowledge') {
    return <BookSearch className="mt-0.5 icon-xs shrink-0" />
  }
  return <Search className="mt-0.5 icon-xs shrink-0" />
}

function stationLabel(
  phase: string,
  t: TranslationDictionary,
): string | undefined {
  const stations = t.agent.stations as Record<string, string | undefined>
  return stations[phase]
}

function autonomyLabel(autonomy: string, t: TranslationDictionary): string {
  if (autonomy === 'strict') return t.agent.composer.autonomyStrict
  if (autonomy === 'autonomous') return t.agent.composer.autonomyAutonomous
  return t.agent.composer.autonomyBalanced
}

/** The live activity readout: explicit activity event > phase default. */
export function activityText(
  run: AgentRunRecord,
  t: TranslationDictionary,
): string {
  // A freshly submitted run without a real queue position reads as
  // "starting", not as waiting in line (B4 polish — honest either way:
  // an actual position > 0 keeps the queue label).
  if (run.status === 'queued') {
    return run.queuePosition && run.queuePosition > 0
      ? t.agent.activity.queued
      : t.agent.activity.starting
  }
  if (run.status === 'waiting_for_approval') {
    return t.agent.activity.waitingApproval
  }
  if (run.status === 'waiting_for_input') return t.agent.activity.waitingInput
  // Parallel work is the point of the wave scheduler — the live line
  // says so instead of pretending one operation runs at a time.
  const runningTasks = run.plan
    ? run.plan.tasks.filter((task) => {
      const status = run.taskStates[task.taskId]?.status
      return status === 'running' || status === 'cancel_requested'
    }).length
    : 0
  const parallelPrefix = runningTasks > 1
    ? t.agent.activity.parallelTasks.replace('{count}', String(runningTasks))
    : ''
  if (run.activity?.operation) {
    const operationText = activityDisplayText(run.activity, t)
    return parallelPrefix
      ? `${parallelPrefix} · ${operationText}`
      : operationText
  }
  if (run.activity?.kind === 'searching') {
    return run.activity.detail
      ? `${t.agent.activity.searching} · ${run.activity.detail}`
      : t.agent.activity.searching
  }
  if (run.activity?.kind === 'memory') {
    if (run.activity.status === 'used') return t.agent.activity.memoryUsed
    if (run.activity.status === 'unavailable') return t.agent.activity.memoryUnavailable
    return t.agent.activity.memoryChecked
  }
  if (run.activity?.kind === 'memory_candidate') {
    return t.agent.activity.memoryCandidate
  }
  if (run.activity?.kind === 'memory_unavailable') {
    return t.agent.activity.memoryUnavailable
  }
  if (run.activity?.kind === 'memory_conflict') {
    return t.agent.activity.memoryConflict
  }
  if (run.activity?.kind === 'critic_research') {
    return t.agent.activity.criticResearch
  }
  if (run.activity?.kind === 'critic_research_exhausted') {
    return t.agent.activity.criticResearchExhausted
  }
  if (run.activity?.label) return run.activity.label
  if (run.phase === 'execution' && run.plan) {
    const tasks = run.plan.tasks.filter((task) => task.toolKind !== 'synthesis')
    const done = tasks.filter(
      (task) => run.taskStates[task.taskId]?.status === 'completed',
    ).length
    return t.agent.activity.taskProgress
      .replace('{done}', String(done))
      .replace('{total}', String(tasks.length))
  }
  const key = run.phase as keyof TranslationDictionary['agent']['activity']
  const label = t.agent.activity[key]
  return typeof label === 'string' ? label : t.agent.activity.execution
}
