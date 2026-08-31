import { useEffect, useMemo, useState } from 'react'
import type { MouseEvent } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import {
  AlertTriangle,
  BookSearch,
  Check,
  ChevronRight,
  Copy,
  ExternalLink,
  FileText,
  Globe2,
  ListChecks,
  MessageSquareText,
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
import { childProgressLine } from '../childProgress'
import { diffHunkPlan, type DiffHunkPlan } from './artifactDiffHunks'
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
  agentReferenceAsKnowledge,
  isWebEvidenceReference,
  answerCitationLabels,
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
import {
  agentRunCompletionRecap,
  agentTodoReportAge,
  agentTurnDocumentTarget,
  shouldAnimateAgentNarration,
} from '../runPresentation'
import { CitationGroupList } from '@/features/knowledge/CitationRow'
import {
  citationViews,
  groupCitationsByDocument,
} from '@/features/knowledge/citations'
import { agentRunFailureText } from '../runFailure'
import { decidedReportGuidance } from '../plan/reportGuidance'
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
  /** Child-gate twins: the decision targets the CHILD run id, the local
   * echo lands on the parent record (F-P0-CHILDGATE). */
  answerChildClarification: (
    parentRunId: string,
    childRunId: string,
    clarificationId: string,
    answer: AgentClarificationAnswerRequest,
  ) => Promise<unknown>
  decideChildApproval: (
    parentRunId: string,
    childRunId: string,
    approvalId: string,
    decision: AgentApprovalDecisionRequest,
  ) => Promise<unknown>
  onCancelRun: (runId: string) => void
  onOpenCanvas: (descriptor: CanvasViewDescriptor) => void
  /** Revision-body fetch for the inline chip diff (P9b). Resolves the
   * artifact's CURRENT anchor internally; view-only, never store-bound. */
  loadArtifactRevision: (
    runId: string,
    artifactId: string,
    revision: number,
  ) => Promise<{ content_markdown: string }>
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
  artifactNames,
  historical = false,
  run,
  sessionMemo,
  transportDegraded = false,
}: {
  actions: AgentTimelineActions
  /** Session file names (P9, artifactId -> `name.md`), derived from the
   * anchor-independent session index; absent while it has not loaded —
   * the chips then fall back to plain titles instead of faking names. */
  artifactNames?: Record<string, string>
  /** Persisted/hydrated turns render in place. Exact events carrying the
   * transport's live provenance may still animate as the run continues. */
  historical?: boolean
  run: AgentRunRecord
  /** The session's memo under its CURRENT run anchor (P4 / F-NEU-1): an
   * update re-anchors the artifact server-side, so after a reload an
   * older turn's own artifact listing no longer carries the memo it
   * produced. The memo is unique per session — this is the fallback. */
  sessionMemo?: { artifactId: string; runId: string } | null
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
  // Inline chip diffs (P9b): expansion state lives on the TURN so the
  // file rows keep their open diff when they move between the running
  // fallback slot and the answer block.
  const [expandedDiffs, setExpandedDiffs] = useState<
    Record<string, boolean>
  >({})
  // P9d: the sent-comments line above the question bubble (collapsed
  // by default — quiet, expandable on demand).
  const [contextExpanded, setContextExpanded] = useState(false)
  const memoId = run.artifactOrder.find(
    (artifactId) => run.artifacts[artifactId]?.kind === 'memo',
  )
  const memoTarget = agentTurnDocumentTarget(
    memoId,
    run.runId,
    run.touchedArtifacts,
    sessionMemo,
  ) ?? undefined
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
  // File rows are OUTPUT (P9b): they render inside the answer block —
  // above its copy bar — once an answer exists, and fall back to the
  // turn's tail while the run is still producing one.
  const artifactRows = run.touchedArtifacts.length > 0 ? (
    <ArtifactFileRows
      actions={actions}
      artifactNames={artifactNames}
      expandedDiffs={expandedDiffs}
      historical={historical}
      onToggleDiff={(artifactId) =>
        setExpandedDiffs((current) => ({
          ...current,
          [artifactId]: !current[artifactId],
        }))}
      reduceMotion={reduceMotion}
      run={run}
      t={t}
    />
  ) : null

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="space-y-2"
      initial={reduceMotion || historical ? false : { opacity: 0, y: 6 }}
      transition={appMotion.card}
    >
      {/* The question follows the chat mode's user-message pattern to
          the letter (meta line above, tone-aware bubble below) so Agent
          Desk and Chat read as ONE design language. */}
      {/* Meta line, sent-comments box and bubble are SIBLINGS in a
          right-aligned column (P9f): each sizes to its own content under
          the same cap, so expanding the box never re-widens the bubble
          — the box may simply be wider than the bubble. */}
      <div className="flex min-w-0 flex-col items-end">
        <div className="mb-1 flex max-w-[min(72%,44rem)] flex-wrap items-center justify-end gap-x-2 gap-y-0.5 t-meta-sm font-semibold text-muted-foreground">
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
          {run.canvasContextMeta
            && run.canvasContextMeta.comments.length > 0 && (
            <div className="mb-1.5 max-w-[min(72%,44rem)] rounded-md border border-border/70 bg-surface/60 text-left">
              {/* P9d/P9e: the chat chain-trace design language — full-
                  width header button, bordered rows — records which
                  comments rode this submission (replay-durable). The box
                  sizes to its own content and may exceed the bubble's
                  width when expanded (P9f decoupling). */}
              <button
                aria-expanded={contextExpanded}
                className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left t-meta-sm font-semibold text-muted-foreground transition hover:text-foreground"
                onClick={() => setContextExpanded((current) => !current)}
                type="button"
              >
                <MessageSquareText className="size-3.5 shrink-0" />
                <span className="min-w-0 flex-1 truncate">
                  {t.agent.timeline.sentCanvasComments
                    .replace(
                      '{count}',
                      String(run.canvasContextMeta.comments.length),
                    )
                    .replace(
                      '{name}',
                      artifactNames?.[run.canvasContextMeta.artifactId]
                        || t.agent.canvas.views.document,
                    )}
                </span>
                <ChevronRight
                  className={cn(
                    'size-3.5 shrink-0 transition-transform',
                    contextExpanded && 'rotate-90',
                  )}
                />
              </button>
              {contextExpanded && (
                <div className="border-t border-border/70 px-2.5 py-1">
                  {run.canvasContextMeta.comments.map((comment, index) => (
                    <div
                      className="flex min-w-0 gap-1.5 border-b border-border/40 py-1 last:border-0"
                      key={index}
                    >
                      <span className="grid size-4 shrink-0 place-items-center rounded-[4px] bg-brand-subtle t-hint font-semibold tabular-nums text-brand">
                        {index + 1}
                      </span>
                      <span className="min-w-0 t-meta-sm">
                        <span className="text-muted-foreground">
                          „{comment.quotePreview}“
                        </span>{' '}
                        <span className="text-foreground">
                          {comment.comment}
                        </span>
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          <div className="inqtrix-user-bubble min-w-0 max-w-[min(72%,44rem)] rounded-lg px-3 py-2.5 text-sm leading-6 shadow-[0_1px_2px_var(--shadow-hairline)]">
            <p className="whitespace-pre-wrap break-words">{run.question}</p>
          </div>
      </div>

      {run.status === 'completed' && (
        <RunCompletionRecap
          memo={memoTarget}
          onOpenMemo={(target) => actions.onOpenCanvas({
            artifactId: target.artifactId,
            runId: target.runId,
            view: 'document',
          })}
          onToggleSteps={() =>
            setHistoryExpanded((current) => !current)}
          recap={completionRecap}
          stepCount={run.stepLog.length}
          stepsExpanded={historyExpanded}
          t={t}
        />
      )}

      {/* Completed runs fold the step toggle into the recap's fact line
          (P9b) — the standalone row remains for failed/cancelled runs. */}
      {!isActive && run.status !== 'completed' && run.stepLog.length > 0 && (
        <button
          aria-expanded={historyExpanded}
          className="flex items-center gap-1.5 px-1 t-meta text-muted-foreground transition-colors hover:text-foreground"
          onClick={() => setHistoryExpanded((current) => !current)}
          type="button"
        >
          <X className="icon-xs shrink-0 text-muted-foreground" />
          {run.stepLog.length === 1
            ? t.agent.timeline.stepsSummaryOne
            : t.agent.timeline.stepsSummary.replace(
              '{count}',
              String(run.stepLog.length),
            )}
          <ChevronRight
            className={cn(
              'size-3 shrink-0 transition-transform',
              historyExpanded && 'rotate-90',
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
              historicalRun={historical}
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

      {answer ? (
        <AgentAnswerBlock
          actions={actions}
          answer={answer}
          artifactRows={artifactRows}
          run={run}
          t={t}
        />
      ) : (
        artifactRows && <div className="px-1">{artifactRows}</div>
      )}

      {(run.status === 'failed' || run.status === 'cancelled') && (
        <div className="flex items-start gap-2 px-1">
          <AlertTriangle className="mt-0.5 icon-sm shrink-0 text-destructive" />
          <div className="min-w-0">
            <p className="break-words t-meta text-destructive/90">
              {run.status === 'cancelled'
                ? t.agent.timeline.cancelled
                : run.error
                  ? t.agent.timeline.taskFailed.replace(
                    '{error}',
                    agentRunFailureText(run.error, t),
                  )
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

      {run.error
        && run.status !== 'failed'
        && run.status !== 'cancelled' && (
        <p
          className="flex items-start gap-1.5 px-1 t-meta text-destructive/90"
          data-testid="agent-run-surface-error"
          role="status"
        >
          <AlertTriangle className="mt-0.5 icon-xs shrink-0" />
          <span className="min-w-0 break-words">
            {agentRunFailureText(run.error, t)}
          </span>
        </p>
      )}

      {isActive && transportDegraded && (
        <p className="flex items-center gap-1.5 px-1 t-hint text-muted-foreground">
          <AlertTriangle className="icon-xs shrink-0 text-warning" />
          {t.agent.timeline.transportDegraded}
        </p>
      )}
      {isActive && (run.todos?.length ?? 0) > 0 && (
        <AgentTodoList
          reportAgeSeconds={agentTodoReportAge(
            run.todosAt,
            run.stepLog,
            Date.now() / 1000,
          )}
          t={t}
          todos={run.todos ?? []}
        />
      )}
      {isActive && (
        <div className="flex items-center gap-2 px-1">
          <AgentActivityLine
            className="min-w-0 flex-1"
            gate={isGateAgentRun(run.status)}
            text={activityText(run, t)}
          />
          <ElapsedRuntime startedAt={run.startedAt} />
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

/** The kernel's live task list — ONE compact checklist block that
 * replaces itself per todo.updated event (dsh-style recitation). */
function AgentTodoList({
  reportAgeSeconds,
  t,
  todos,
}: {
  /** Seconds since the model last reported this list, when the run has
   * worked since — the list is a report, not a live readout. */
  reportAgeSeconds: number | null
  t: TranslationDictionary
  todos: { content: string; status: string }[]
}) {
  return (
    <div
      className="mx-1 rounded-lg border border-border/60 bg-muted/30 px-2.5 py-1.5"
      data-testid="agent-todo-list"
    >
      <p className="t-caption uppercase tracking-wide text-muted-foreground/70">
        {t.agent.timeline.todoTitle}
        {reportAgeSeconds !== null && (
          <span data-testid="agent-todo-reported-at">
            {' · '}
            {t.agent.timeline.todoReportedAgo.replace(
              '{duration}',
              formatDuration(reportAgeSeconds),
            )}
          </span>
        )}
      </p>
      <ul className="mt-1 space-y-0.5">
        {todos.map((todo, index) => {
          const done = todo.status === 'completed'
          const active = todo.status === 'in_progress'
          return (
            <li
              className={cn(
                'flex items-start gap-1.5 t-meta',
                done
                  ? 'text-muted-foreground/70 line-through'
                  : active
                    ? 'text-foreground/90'
                    : 'text-muted-foreground',
              )}
              key={`${index}:${todo.content}`}
            >
              {done ? (
                <Check className="mt-0.5 icon-xs shrink-0 text-success/80" />
              ) : active ? (
                <span
                  aria-hidden="true"
                  className="mt-1.5 size-1.5 shrink-0 rounded-full bg-brand inqtrix-running-dot"
                />
              ) : (
                <span
                  aria-hidden="true"
                  className="mt-1.5 size-1.5 shrink-0 rounded-full border border-muted-foreground/50"
                />
              )}
              <span className="min-w-0 break-words">{todo.content}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

/** Live elapsed time of the active run — the P0 audit's "no clock, no
 * progress" finding. A plain 1s interval (rAF pauses in background
 * tabs); the value derives from startedAt, so ticks only repaint. */
function ElapsedRuntime({ startedAt }: { startedAt: string | undefined }) {
  const [, setTick] = useState(0)
  useEffect(() => {
    const id = window.setInterval(() => setTick((n) => n + 1), 1000)
    return () => window.clearInterval(id)
  }, [])
  if (!startedAt) return null
  const seconds = Math.max(
    0,
    Math.floor((Date.now() - Date.parse(startedAt)) / 1000),
  )
  if (!Number.isFinite(seconds)) return null
  return (
    <span className="shrink-0 t-hint tabular-nums text-muted-foreground">
      {formatDuration(seconds)}
    </span>
  )
}

function RunCompletionRecap({
  memo,
  onOpenMemo,
  onToggleSteps,
  recap,
  stepCount = 0,
  stepsExpanded = false,
  t,
}: {
  memo: { artifactId: string; runId: string } | undefined
  onOpenMemo: (target: { artifactId: string; runId: string }) => void
  /** Step-feed disclosure folded into the fact line (P9b). */
  onToggleSteps?: () => void
  recap: ReturnType<typeof agentRunCompletionRecap>
  stepCount?: number
  stepsExpanded?: boolean
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
      {/* Title and facts share ONE flex-wrap row (P9f): a single compact
          line on normal widths that folds back to two when narrow. */}
      <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-1">
        <span className="flex shrink-0 items-center gap-2">
          <Check className="icon-sm shrink-0 text-success" />
          <p className="t-list text-foreground/90">
            {t.agent.timeline.recapTitle}
          </p>
        </span>
        {(facts.length > 0 || stepCount > 0) && (
          <div className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-1 t-meta tabular-nums text-muted-foreground">
            {facts.length > 0 && <span>{facts.join(' · ')}</span>}
            {stepCount > 0 && onToggleSteps && (
              <>
                {facts.length > 0 && <span aria-hidden="true">·</span>}
                <button
                  aria-expanded={stepsExpanded}
                  className="flex items-center gap-1 transition-colors hover:text-foreground"
                  onClick={onToggleSteps}
                  type="button"
                >
                  {stepCount === 1
                    ? t.agent.timeline.stepsSummaryOne
                    : t.agent.timeline.stepsSummary.replace(
                      '{count}',
                      String(stepCount),
                    )}
                  <ChevronRight
                    className={cn(
                      'size-3 shrink-0 transition-transform',
                      stepsExpanded && 'rotate-90',
                    )}
                  />
                </button>
              </>
            )}
          </div>
        )}
        {memo && (
          <Button
            className="ml-auto h-6 shrink-0 gap-1 bg-brand px-2 text-xs text-brand-foreground hover:bg-brand/90"
            onClick={() => onOpenMemo(memo)}
            size="sm"
            type="button"
          >
            <FileText className="icon-xs" />
            {t.agent.timeline.openMemo}
          </Button>
        )}
      </div>
      {process.length > 0 && (
        <p className="mt-0.5 t-meta-sm text-foreground/75">
          {process.join(' · ')}
        </p>
      )}
    </section>
  )
}

/**
 * The turn's file rows (P9b): one row per touched document, the ± badge
 * doubling as a disclosure (chevron right -> down) that expands an
 * inline hunk diff below the rows — the canvas full diff stays
 * reachable from inside the expansion.
 */
function ArtifactFileRows({
  actions,
  artifactNames,
  expandedDiffs,
  historical,
  onToggleDiff,
  reduceMotion,
  run,
  t,
}: {
  actions: AgentTimelineActions
  artifactNames?: Record<string, string>
  expandedDiffs: Record<string, boolean>
  historical: boolean
  onToggleDiff: (artifactId: string) => void
  reduceMotion: boolean
  run: AgentRunRecord
  t: TranslationDictionary
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        {run.touchedArtifacts.map((touched) => {
          // File-row identity (P9): the derived session file name
          // leads; title stays the honest fallback while the session
          // index has not loaded (never a fake unsuffixed name).
          const label = artifactNames?.[touched.artifactId]
            || touched.title
            || run.artifacts[touched.artifactId]?.title
            || t.agent.canvas.views.document
          // The ± badge needs a diffable base revision AND a complete
          // server-counted sum — otherwise it stays honestly absent.
          const showDelta = touched.fromRevision >= 1
            && touched.linesAdded !== undefined
            && touched.linesRemoved !== undefined
          const expanded = Boolean(expandedDiffs[touched.artifactId])
          return (
            <motion.span
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex h-6 min-w-0 max-w-80 items-stretch overflow-hidden rounded-full border border-border bg-surface t-hint font-semibold text-muted-foreground"
              // Same live-boundary rule as the step rows: only a chip
              // that arrived on the live side of its stream rises (4D).
              initial={
                reduceMotion || historical || !touched.arrivedLive
                  ? false
                  : { opacity: 0, y: 4 }
              }
              key={touched.artifactId}
              transition={appMotion.list}
            >
              <button
                className="inline-flex min-w-0 items-center gap-1.5 pl-2.5 pr-2 transition-colors hover:text-foreground"
                onClick={() => actions.onOpenCanvas({
                  artifactId: touched.artifactId,
                  runId: run.runId,
                  view: 'document',
                })}
                // Full name stays reachable at the visual cut (9b).
                title={label}
                type="button"
              >
                <FileText className="icon-xs shrink-0" />
                <span className="truncate">{label}</span>
                <span className="shrink-0 rounded-full border border-border bg-background px-1.5 py-px t-hint font-semibold">
                  {touched.created
                    ? t.agent.timeline.artifactCreated
                    : t.agent.timeline.artifactUpdated}
                </span>
              </button>
              {showDelta && (
                <button
                  aria-expanded={expanded}
                  aria-label={t.agent.timeline.artifactDiff
                    .replace('{from}', String(touched.fromRevision))
                    .replace('{to}', String(touched.revision))}
                  className="inline-flex shrink-0 items-center gap-1 border-l border-border px-2 tabular-nums transition-colors hover:bg-background"
                  onClick={() => onToggleDiff(touched.artifactId)}
                  title={t.agent.timeline.artifactDiff
                    .replace('{from}', String(touched.fromRevision))
                    .replace('{to}', String(touched.revision))}
                  type="button"
                >
                  <span className="text-success">
                    +{touched.linesAdded}
                  </span>
                  <span className="text-destructive">
                    −{touched.linesRemoved}
                  </span>
                  <ChevronRight
                    className={cn(
                      'size-3 shrink-0 transition-transform',
                      expanded && 'rotate-90',
                    )}
                  />
                </button>
              )}
            </motion.span>
          )
        })}
      </div>
      {run.touchedArtifacts
        .filter((touched) => expandedDiffs[touched.artifactId])
        .map((touched) => (
          <ArtifactInlineDiff
            actions={actions}
            artifactId={touched.artifactId}
            fromRevision={touched.fromRevision}
            key={touched.artifactId}
            runId={run.runId}
            t={t}
            toRevision={touched.revision}
          />
        ))}
    </div>
  )
}

/**
 * Inline hunk diff of one turn's revision span (P9b): changed regions
 * only — git-style context, visible "unchanged lines" gaps, an inner
 * scroll bound — with the canvas full-document diff one click away.
 */
function ArtifactInlineDiff({
  actions,
  artifactId,
  fromRevision,
  runId,
  t,
  toRevision,
}: {
  actions: AgentTimelineActions
  artifactId: string
  fromRevision: number
  runId: string
  t: TranslationDictionary
  toRevision: number
}) {
  const [plan, setPlan] = useState<DiffHunkPlan | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setPlan(null)
    setError(null)
    void Promise.all([
      actions.loadArtifactRevision(runId, artifactId, fromRevision),
      actions.loadArtifactRevision(runId, artifactId, toRevision),
    ])
      .then(([from, to]) => {
        if (cancelled) return
        setPlan(diffHunkPlan(from.content_markdown, to.content_markdown))
      })
      .catch(() => {
        if (!cancelled) setError(t.agent.timeline.diffLoadFailed)
      })
    return () => {
      cancelled = true
    }
  }, [actions, artifactId, fromRevision, runId, t, toRevision])

  const gap = (count: number, key: string) => (
    <p
      className="px-2 py-0.5 t-hint text-muted-foreground/80"
      key={key}
    >
      {t.agent.timeline.diffSkippedLines.replace('{count}', String(count))}
    </p>
  )

  return (
    <div className="min-w-0 max-w-4xl rounded-md border border-border bg-surface">
      {error ? (
        <p className="px-2 py-1.5 t-hint text-destructive/90" role="status">
          {error}
        </p>
      ) : plan === null ? (
        <p className="px-2 py-1.5 t-hint text-muted-foreground">…</p>
      ) : (
        <div className="max-h-72 overflow-y-auto py-1 font-mono t-hint leading-5">
          {plan.hunks.map((hunk, hunkIndex) => (
            <div key={hunkIndex}>
              {hunk.skippedBefore > 0
                && gap(hunk.skippedBefore, `gap-${hunkIndex}`)}
              {hunk.lines.map((line, lineIndex) => (
                <p
                  className={cn(
                    'flex min-w-0 gap-2 whitespace-pre-wrap break-words px-2',
                    line.type === 'insert'
                      && 'bg-success/10 text-success',
                    line.type === 'delete'
                      && 'bg-destructive/10 text-destructive',
                    line.type === 'context' && 'text-muted-foreground',
                  )}
                  key={lineIndex}
                >
                  <span aria-hidden="true" className="w-3 shrink-0 select-none">
                    {line.type === 'insert'
                      ? '+'
                      : line.type === 'delete'
                        ? '−'
                        : ' '}
                  </span>
                  <span className="min-w-0 flex-1">{line.text || ' '}</span>
                </p>
              ))}
            </div>
          ))}
          {plan.skippedAfter > 0 && gap(plan.skippedAfter, 'gap-tail')}
        </div>
      )}
      <div className="border-t border-border/70 px-2 py-1">
        <button
          className="t-hint text-muted-foreground transition-colors hover:text-foreground"
          onClick={() => actions.onOpenCanvas({
            artifactId,
            fromRevision,
            runId,
            toRevision,
            view: 'diff',
          })}
          type="button"
        >
          {t.agent.timeline.openFullDiff}
        </button>
      </div>
    </div>
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
  artifactRows,
  run,
  t,
}: {
  actions: AgentTimelineActions
  answer: AgentArtifactRecord
  /** The turn's file rows (P9b) — output belongs to the answer, above
   * its copy bar, not above the run status. */
  artifactRows?: React.ReactNode
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
  // Citations render from the FIRST delta. While writing the real refs
  // do not exist yet, but the server sends the LABELS the finished
  // answer will cite (`answer.started`) — and the linkifier needs only
  // those. Without this the body was rewritten wholesale the moment the
  // answer settled: every `[W1]` became a link in one step, which reads
  // as the message being re-inserted. Same labels before and after, so
  // the markdown handed to the renderer no longer changes at all.
  const citationLabels = useMemo(
    () =>
      answerCitationLabels(writing, answer.publicationRefLabels, references),
    [answer.publicationRefLabels, references, writing],
  )
  const linkedBody = useMemo(
    () =>
      body !== undefined && citationLabels.length > 0
        ? linkifyAgentArtifactCitations(body, citationLabels)
        : body,
    [body, citationLabels],
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
      {artifactRows && <div className="mt-2">{artifactRows}</div>}
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
 * Source list under the chat answer, folded behind a disclosure (P9f):
 * a long reference list (13 rows in the survey session measured 862px)
 * would otherwise dominate the transcript, so the default is collapsed
 * with a visible count — same chevron convention as every other
 * expander. Web references retain their external URL and expose the
 * same evidence-trail action/status as the Canvas; Knowledge references
 * open the evidence view directly.
 */
function AgentAnswerSources({
  onOpenEvidence,
  references,
}: {
  onOpenEvidence: (label: string) => void
  references: AgentArtifactReference[]
}) {
  const { t } = useLocale()
  const [expanded, setExpanded] = useState(false)
  const webReferences = references.filter(isWebEvidenceReference)
  const knowledgeGroups = useMemo(
    () =>
      groupCitationsByDocument(
        citationViews(
          references
            .filter((reference) => !isWebEvidenceReference(reference))
            .map(agentReferenceAsKnowledge),
          [],
          t.knowledge.viewerSection,
        ).map((view) => ({ ...view, canOpen: true })),
      ),
    [references, t.knowledge.viewerSection],
  )
  return (
    <section
      className="mt-3 max-w-4xl border-t border-border/70 pt-2"
      data-testid="agent-sources"
    >
      <button
        aria-expanded={expanded}
        className="flex items-center gap-1 t-meta-sm font-semibold text-muted-foreground transition-colors hover:text-foreground"
        onClick={() => setExpanded((current) => !current)}
        type="button"
      >
        {references.length === 1
          ? t.agent.timeline.sourcesSummaryOne
          : t.agent.timeline.sourcesSummary.replace(
            '{count}',
            String(references.length),
          )}
        <ChevronRight
          className={cn(
            'size-3 shrink-0 transition-transform',
            expanded && 'rotate-90',
          )}
        />
      </button>
      {expanded && (
        <div className="mt-1.5 space-y-2">
          {/* Knowledge citations render EXACTLY like the Knowledge Desk's:
              the cited passage leads, the document name and its section
              sit in the quiet meta line, and several passages of one PDF
              collapse into one group. A flat list repeated the same file
              name for every citation and never said WHERE in it. */}
          {knowledgeGroups.length > 0 && (
            <CitationGroupList
              activeKey={null}
              groups={knowledgeGroups}
              onOpen={(view) => onOpenEvidence(view.label)}
              onOpenDocument={(group) => {
                const first = group.citations[0]
                if (first) onOpenEvidence(first.label)
              }}
            />
          )}
          {webReferences.length > 0 && (
            <ul className="space-y-1.5">
              {webReferences.map((reference) => (
                <li key={reference.label}>
                  <WebEvidenceSourceRow
                    onInspect={() => onOpenEvidence(reference.label)}
                    reference={{
                      ...reference,
                      domain: hostFromUrl(reference.url ?? ''),
                      key: reference.referenceId ?? reference.label,
                    }}
                  />
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
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

/** One transcript line, dispatched by step kind. */
function StreamEntry({
  actions,
  entry,
  historicalRun = false,
  isLatest = false,
  run,
  t,
}: {
  actions: AgentTimelineActions
  entry: AgentStepEntry
  historicalRun?: boolean
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
      const row = (
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
      const childLine = settled || failed
        ? null
        : delegatedChildLine(run, entry, t)
      if (!childLine) return row
      // A delegation is ONE tool row that can own the run for tens of
      // minutes. Its child reports phase and task all along; this is
      // where that reaches the reader.
      return (
        <>
          {row}
          <p
            className="flex items-start gap-1.5 pl-5 t-meta text-muted-foreground/70"
            data-testid="agent-child-progress"
          >
            <span className="min-w-0 break-words">{childLine}</span>
          </p>
        </>
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
      return (
        <NarrationText
          animate={shouldAnimateAgentNarration({
            entry,
            historicalRun,
            isLatest,
          })}
          text={entry.text}
        />
      )
    case 'notice': {
      const label = noticeDisplayText(entry, t)
      if (!label) return null
      return (
        <p
          className="flex items-start gap-1.5 t-meta text-warning"
          data-testid="agent-notice-item"
        >
          <AlertTriangle className="mt-0.5 icon-xs shrink-0" />
          <span className="min-w-0 break-words">{label}</span>
        </p>
      )
    }
    case 'gate_requested': {
      // The request marker stays visible even when its row is gone —
      // unlike the decision receipts, history must never lose WHAT was
      // asked (the rows only enrich the line).
      const clarification = entry.clarificationId
        ? run.clarifications.find(
          (item) => item.clarificationId === entry.clarificationId,
        )
        : undefined
      const approval = entry.approvalId
        ? run.approvals.find(
          (item) => item.approvalId === entry.approvalId,
        )
        : undefined
      const label = entry.clarificationId
        ? t.agent.timeline.gateRequestedClarification
        : gateRequestedLabel(entry.detail ?? approval?.kind ?? '', t)
      const detail = clarification?.question ?? ''
      // P6B scope badge: a decision that granted the tool run-wide
      // stays readable in the history.
      const grantedForRun =
        approval?.decisionPayload?.approval_scope === 'run'
      // The result requirement shaped every section of the report. It
      // lived only in the decision payload, so after the gate closed
      // the user had no way to see what was still in force.
      const guidance = approval ? decidedReportGuidance([approval]) : ''
      return (
        <div className="space-y-0.5">
          <p className="flex items-start gap-1.5 t-meta text-muted-foreground">
            <PenLine className="mt-0.5 icon-xs shrink-0" />
            <span className="min-w-0 break-words">
              {label}
              {detail ? ` · ${detail}` : ''}
              {grantedForRun
                ? ` · ${t.agent.timeline.gateGrantedForRun}`
                : ''}
            </span>
          </p>
          {guidance && <GateGuidanceLine guidance={guidance} />}
        </div>
      )
    }
    default:
      return null
  }
}

/** The result requirement of a decided plan gate, collapsed by default.
 *
 * Collapsed because it can be 2000 characters and the transcript is a
 * conversation, not a form; present because a requirement the user
 * cannot re-read is one they have to remember. */
function GateGuidanceLine({ guidance }: { guidance: string }) {
  const { t } = useLocale()
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="pl-[1.125rem]">
      <button
        aria-expanded={expanded}
        className="flex items-center gap-1 t-hint text-muted-foreground/80 transition-colors hover:text-foreground"
        onClick={() => setExpanded((current) => !current)}
        type="button"
      >
        <ChevronRight
          className={cn(
            'icon-xs transition-transform',
            expanded && 'rotate-90',
          )}
        />
        {t.agent.plan.reportGuidanceEffective}
      </button>
      {expanded && (
        <p className="mt-0.5 whitespace-pre-line break-words t-meta text-foreground/85">
          {guidance}
        </p>
      )}
    </div>
  )
}


/** i18n label of one runtime notice row (the row carries only its code).
 * Exported for unit tests (node env cannot render the component). */
export function noticeDisplayText(
  entry: Pick<AgentStepEntry, 'current' | 'detail' | 'noticeCode' | 'status' | 'total'>,
  t: TranslationDictionary,
): string {
  switch (entry.noticeCode) {
    case 'tool_limit':
      return t.agent.timeline.noticeToolLimit
        .replace('{used}', String(entry.current ?? '—'))
        .replace('{limit}', String(entry.total ?? '—'))
    case 'quick_web_fallback':
      return entry.detail === 'answer'
        ? t.agent.timeline.noticeQuickWebAnswerFallback
        : t.agent.timeline.noticeQuickWebQueryFallback
    case 'citation_validation':
      return entry.status === 'degraded'
        ? t.agent.timeline.noticeCitationsDegraded.replace(
          '{labels}',
          entry.detail ?? '',
        )
        : t.agent.timeline.noticeCitationsRepaired
    case 'sufficiency_gap':
      return t.agent.timeline.noticeSufficiencyGap.replace(
        '{gaps}',
        entry.detail ?? '',
      )
    default:
      return ''
  }
}

function gateRequestedLabel(kind: string, t: TranslationDictionary): string {
  switch (kind) {
    case 'plan':
    case 'replan':
      return t.agent.timeline.gateRequestedPlan
    case 'discovery':
      return t.agent.timeline.gateRequestedDiscovery
    case 'patch':
      return t.agent.timeline.gateRequestedPatch
    case 'tool':
      return t.agent.timeline.gateRequestedTool
    default:
      return t.agent.timeline.gateRequested
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

/** The live line of the child a delegation row started, if it has one.
 *
 * The link already exists: a child's `taskId` IS the delegating tool
 * call's id, which is how the gate tray finds the child's assignment. */
function delegatedChildLine(
  run: AgentRunRecord,
  entry: AgentStepEntry,
  t: TranslationDictionary,
): string | null {
  const key = entry.activityKey
  if (!key?.startsWith('tool:')) return null
  const toolCallId = key.slice('tool:'.length)
  const child = Object.values(run.children).find(
    (candidate) => candidate.taskId.split(':')[0] === toolCallId,
  )
  return child ? childProgressLine(child, t) : null
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
  // Waiting for children is NOT idling: the delegated run is doing the
  // work, and its progress arrives here as `child.progress`. Without
  // this branch the line fell through to the parent's own last tool
  // row, which froze the moment it delegated — one unchanging sentence
  // for twenty minutes, on BOTH surfaces that render this text (the
  // transcript's running line and the follow-execution panel). The run
  // looked hung while it was verifying its twelfth finding.
  if (run.status === 'waiting_for_children') {
    for (const child of Object.values(run.children)) {
      const line = childProgressLine(child, t)
      if (line) return line
    }
  }
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
