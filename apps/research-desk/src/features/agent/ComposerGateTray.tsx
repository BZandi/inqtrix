import { useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { ChevronDown, ExternalLink, PenLine, Search } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Chip } from '@/components/ui/chip'
import { Input } from '@/components/ui/input'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type { TranslationDictionary } from '@/i18n/translations'
import {
  answersRequestFromDraft,
  isRoundComplete,
  setFreeText,
  toggleOption,
  type RoundAnswerDraft,
} from './clarificationAnswers'
import type { AgentClarificationAnswerRequest } from './types'
import {
  isActiveAgentRun,
  type AgentApprovalRecord,
  type AgentClarificationRecord,
  type AgentRunRecord,
} from './model'
import { usePatchReview } from './patch/usePatchReview'
import { usePlanApproval } from './plan/usePlanApproval'
import type { AgentTimelineActions } from './timeline/AgentTimeline'
import { discoveryProbeDisplay } from './activityPresentation'

/**
 * The ONE gate surface (plan B3): every pending human decision of the
 * active run — clarification, plan/replan approval, discovery approval,
 * patch review — grows out of the composer as a tray, is answered at
 * the input locus, and afterwards appears in the transcript as a
 * user-side decision entry (the events append the markers). At most one
 * gate is pending at a time (runs are sequentially gated).
 */
export function ComposerGateTray({
  actions,
  run,
}: {
  actions: AgentTimelineActions
  run: AgentRunRecord | undefined
}) {
  const { t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  const gate = run ? pendingGate(run) : null
  return (
    <AnimatePresence initial={false}>
      {run && gate && (
        <motion.div
          animate={{ height: 'auto', opacity: 1 }}
          className="overflow-hidden"
          exit={reduceMotion ? undefined : { height: 0, opacity: 0 }}
          initial={reduceMotion ? false : { height: 0, opacity: 0 }}
          transition={appMotionSafe(reduceMotion)}
        >
          <div
            className={cn(
              'mx-auto mb-1.5 max-w-5xl rounded-xl border px-3 py-2.5 shadow-[0_1px_2px_var(--shadow-hairline)]',
              gate.kind === 'clarification'
                ? 'border-warning/30 bg-warning-subtle/40'
                : 'border-brand/25 bg-brand-subtle/40',
            )}
          >
            {gate.kind === 'clarification' && (
              <ClarificationGate
                clarification={gate.clarification}
                onAnswer={(answer) =>
                  actions.answerClarification(
                    run.runId,
                    gate.clarification.clarificationId,
                    answer,
                  )}
                t={t}
              />
            )}
            {gate.kind === 'plan' && (
              <PlanGate actions={actions} run={run} t={t} />
            )}
            {gate.kind === 'discovery' && (
              <DiscoveryGate
                approval={gate.approval}
                actions={actions}
                run={run}
                t={t}
              />
            )}
            {gate.kind === 'patch' && (
              <PatchGate actions={actions} run={run} t={t} />
            )}
            {gate.kind === 'tool' && (
              <ToolGate
                approval={gate.approval}
                actions={actions}
                run={run}
                t={t}
              />
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

type PendingGate =
  | { kind: 'clarification'; clarification: AgentClarificationRecord }
  | { kind: 'plan'; approval: AgentApprovalRecord }
  | { kind: 'discovery'; approval: AgentApprovalRecord }
  | { kind: 'patch'; approval: AgentApprovalRecord }
  | { kind: 'tool'; approval: AgentApprovalRecord }

/** The active run's single pending gate; decisions on settled runs 409. */
export function pendingGate(run: AgentRunRecord): PendingGate | null {
  if (!isActiveAgentRun(run.status)) return null
  const clarification = run.clarifications.find(
    (item) => item.status === 'pending',
  )
  if (clarification) return { clarification, kind: 'clarification' }
  const approval = run.approvals.find((item) => item.status === 'pending')
  if (!approval) return null
  if (approval.kind === 'discovery') return { approval, kind: 'discovery' }
  if (approval.kind === 'patch') return { approval, kind: 'patch' }
  if (approval.kind === 'tool') return { approval, kind: 'tool' }
  return { approval, kind: 'plan' }
}

function ClarificationGate({
  clarification,
  onAnswer,
  t,
}: {
  clarification: AgentClarificationRecord
  onAnswer: (answer: AgentClarificationAnswerRequest) => Promise<unknown>
  t: TranslationDictionary
}) {
  const [submitting, setSubmitting] = useState(false)
  const choose = (optionId: string) => {
    if (submitting) return
    setSubmitting(true)
    void onAnswer({ option_id: optionId }).finally(() => setSubmitting(false))
  }
  if (clarification.questions.length > 0) {
    return (
      <StructuredClarificationForm
        clarification={clarification}
        onAnswer={onAnswer}
        t={t}
      />
    )
  }
  return (
    <div>
      <GateTitle pulse title={t.agent.timeline.waitingInputTitle} tone="warning" />
      <p className="mt-1 break-words t-body text-foreground">
        {clarification.question}
      </p>
      {clarification.options.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          {clarification.options.map((option) => (
            <Button
              className="h-6 rounded-full border-border bg-background px-2.5 text-xs"
              disabled={submitting}
              key={option.id}
              onClick={() => choose(option.id)}
              size="sm"
              type="button"
              variant="outline"
            >
              {option.label}
            </Button>
          ))}
          <span className="t-hint text-muted-foreground">
            {t.agent.tray.freeTextHint}
          </span>
        </div>
      )}
      {clarification.options.length === 0 && (
        <p className="mt-1 t-hint text-muted-foreground">
          {t.agent.tray.freeTextHint}
        </p>
      )}
      {clarification.defaultAssumption && (
        <p className="mt-1.5 t-hint text-muted-foreground">
          {t.agent.timeline.defaultAssumption.replace(
            '{assumption}',
            clarification.defaultAssumption,
          )}
        </p>
      )}
    </div>
  )
}

/** The structured round form: one chips row per question, per-question
 * "Sonstiges" free text, ONE submit for the whole round (the server
 * rejects partial answers — the round parks the run exactly once). */
function StructuredClarificationForm({
  clarification,
  onAnswer,
  t,
}: {
  clarification: AgentClarificationRecord
  onAnswer: (answer: AgentClarificationAnswerRequest) => Promise<unknown>
  t: TranslationDictionary
}) {
  const [submitting, setSubmitting] = useState(false)
  const [draft, setDraft] = useState<RoundAnswerDraft>({})
  const [otherOpen, setOtherOpen] = useState<Record<string, boolean>>({})
  const questions = clarification.questions
  const complete = isRoundComplete(questions, draft)
  const submit = () => {
    if (submitting || !complete) return
    setSubmitting(true)
    void onAnswer({
      answers: answersRequestFromDraft(questions, draft),
    }).finally(() => setSubmitting(false))
  }
  return (
    <div>
      <GateTitle pulse title={t.agent.timeline.waitingInputTitle} tone="warning" />
      <div className="mt-1.5 space-y-2.5">
        {questions.map((question) => {
          const entry = draft[question.id]
          const showOther
            = Boolean(otherOpen[question.id]) || question.options.length === 0
          return (
            <div key={question.id}>
              <p className="break-words t-body text-foreground">
                {question.prompt}
                {question.multiSelect && (
                  <span className="ml-1.5 t-hint text-muted-foreground">
                    {t.agent.tray.clarifyMultiHint}
                  </span>
                )}
              </p>
              <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                {question.options.map((option) => (
                  <Chip
                    active={Boolean(entry?.optionIds.includes(option.id))}
                    disabled={submitting}
                    key={option.id}
                    onClick={() =>
                      setDraft((prev) => toggleOption(prev, question, option.id))}
                    title={option.description || undefined}
                  >
                    {option.label}
                  </Chip>
                ))}
                {question.options.length > 0 && (
                  <Chip
                    active={showOther}
                    disabled={submitting}
                    onClick={() =>
                      setOtherOpen((prev) => ({
                        ...prev,
                        [question.id]: !prev[question.id],
                      }))}
                  >
                    {t.agent.tray.clarifyOther}
                  </Chip>
                )}
              </div>
              {showOther && (
                <Input
                  className="mt-1.5 h-8 max-w-md"
                  disabled={submitting}
                  onChange={(event) =>
                    setDraft((prev) =>
                      setFreeText(prev, question.id, event.target.value))}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      submit()
                    }
                  }}
                  placeholder={t.agent.tray.clarifyOtherPlaceholder}
                  value={entry?.text ?? ''}
                />
              )}
            </div>
          )
        })}
      </div>
      <div className="mt-2.5 flex items-center gap-2">
        <Button
          disabled={submitting || !complete}
          onClick={submit}
          size="sm"
          type="button"
        >
          {t.agent.tray.clarifySubmit}
        </Button>
        <span className="t-hint text-muted-foreground">
          {t.agent.tray.freeTextHint}
        </span>
      </div>
      {clarification.defaultAssumption && (
        <p className="mt-1.5 t-hint text-muted-foreground">
          {t.agent.timeline.defaultAssumption.replace(
            '{assumption}',
            clarification.defaultAssumption,
          )}
        </p>
      )}
    </div>
  )
}

function PlanGate({
  actions,
  run,
  t,
}: {
  actions: AgentTimelineActions
  run: AgentRunRecord
  t: TranslationDictionary
}) {
  const [queriesExpanded, setQueriesExpanded] = useState(false)
  // THE one decision lane (shared with the canvas plan tab): approving
  // with a dirty draft becomes `decision:'edit'` carrying the edits —
  // a raw decideApproval here would silently DROP changes made in the
  // plan tab before the user pressed Freigeben in the tray.
  const planApproval = usePlanApproval({
    decideApproval: actions.decideApproval,
    draft: actions.planDrafts[run.runId] ?? null,
    onDraftChange: (draft) => actions.setPlanDraft(run.runId, draft),
    run,
  })
  const decide = (decision: 'approve' | 'reject') => {
    void planApproval.decide(decision)
  }
  const submitting = planApproval.submitting
  const tasks = run.plan?.tasks ?? []
  const counts = {
    internal: tasks.filter(
      (task) =>
        task.toolKind === 'rag_query' || task.toolKind === 'file_analysis',
    ).length,
    synthesis: tasks.filter((task) => task.toolKind === 'synthesis').length,
    web: tasks.filter(
      (task) =>
        task.toolKind === 'web_research' || task.toolKind === 'web_instant',
    ).length,
  }
  // Concrete numbers against rubber-stamping: the user sees WHAT they
  // approve before the primary action even exists.
  const summary = t.agent.tray.planSummary
    .replace('{version}', String(run.plan?.version ?? 1))
    .replace('{tasks}', String(tasks.length))
    .replace('{internal}', String(counts.internal))
    .replace('{web}', String(counts.web))
    .replace('{synthesis}', String(counts.synthesis))
  // Informed web consent (E16 amendment, plan M1 S7): the approved plan
  // is the ONLY surface where Standard mode consents to web searches —
  // the concrete queries therefore show INLINE, not just behind "Plan
  // ansehen".
  const webQueries = tasks
    .filter(
      (task) =>
        task.toolKind === 'web_research' || task.toolKind === 'web_instant',
    )
    .flatMap((task) => task.queries)
  const shownQueries = queriesExpanded ? webQueries : webQueries.slice(0, 1)
  return (
    <div>
      <GateTitle pulse title={t.agent.tray.planTitle} tone="brand" />
      <p className="mt-1 t-meta tabular-nums text-foreground/90">{summary}</p>
      {shownQueries.length > 0 && (
        <div className="mt-1.5 space-y-0.5">
          {shownQueries.map((query) => (
            <p
              className="flex min-w-0 items-center gap-1.5 t-meta text-foreground/85"
              key={query}
            >
              <Search className="icon-xs shrink-0 text-muted-foreground" />
              <span className={cn('min-w-0 break-words', !queriesExpanded && 'line-clamp-2')}>
                {query}
              </span>
            </p>
          ))}
          {webQueries.length > 0 && (
            <button
              aria-expanded={queriesExpanded}
              className="flex items-center gap-1 t-hint font-medium text-muted-foreground transition-colors hover:text-foreground"
              onClick={() => setQueriesExpanded((current) => !current)}
              type="button"
            >
              {queriesExpanded
                ? t.agent.tray.fewerQueries
                : t.agent.tray.expandQueries.replace(
                  '{count}',
                  String(webQueries.length),
                )}
              <ChevronDown
                className={cn(
                  'icon-xs transition-transform',
                  queriesExpanded && 'rotate-180',
                )}
              />
            </button>
          )}
        </div>
      )}
      {planApproval.error && (
        <p className="mt-1 break-words t-meta-sm text-destructive">
          {planApproval.error}
        </p>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <Button
          className="h-7 gap-1 bg-brand px-2.5 text-xs text-brand-foreground hover:bg-brand/90"
          onClick={() =>
            actions.onOpenCanvas({ runId: run.runId, view: 'plan' })}
          size="sm"
          type="button"
        >
          <ExternalLink className="size-3" />
          {t.agent.tray.viewPlan}
        </Button>
        <Button
          className="h-7 px-2.5 text-xs"
          disabled={submitting}
          onClick={() => decide('approve')}
          size="sm"
          type="button"
          variant="outline"
        >
          {t.agent.timeline.approve}
        </Button>
        <Button
          className="h-7 px-2.5 text-xs text-muted-foreground hover:text-destructive"
          disabled={submitting}
          onClick={() => decide('reject')}
          size="sm"
          type="button"
          variant="ghost"
        >
          {t.agent.timeline.reject}
        </Button>
        <span className="t-hint text-muted-foreground">
          {t.agent.tray.editHint}
        </span>
      </div>
    </div>
  )
}

function DiscoveryGate({
  actions,
  approval,
  run,
  t,
}: {
  actions: AgentTimelineActions
  approval: AgentApprovalRecord
  run: AgentRunRecord
  t: TranslationDictionary
}) {
  const [submitting, setSubmitting] = useState(false)
  const probes = Array.isArray(approval.payload.probes)
    ? (approval.payload.probes as Record<string, unknown>[])
    : []
  const decide = (decision: 'approve' | 'reject') => {
    if (submitting) return
    setSubmitting(true)
    void actions
      .decideApproval(run.runId, approval.approvalId, { decision })
      .finally(() => setSubmitting(false))
  }
  return (
    <div>
      <GateTitle pulse title={t.agent.timeline.waitingApprovalTitle} tone="warning" />
      <p className="mt-1 t-meta text-muted-foreground">
        {t.agent.timeline.discoveryApprovalHint}
      </p>
      <ul className="mt-1.5 space-y-0.5">
        {probes.map((probe, index) => (
          <DiscoveryProbeRow key={index} probe={probe} t={t} />
        ))}
      </ul>
      <div className="mt-2 flex items-center gap-1.5">
        <Button
          className="h-7 bg-brand px-2.5 text-xs text-brand-foreground hover:bg-brand/90"
          disabled={submitting}
          onClick={() => decide('approve')}
          size="sm"
          type="button"
        >
          {t.agent.timeline.approve}
        </Button>
        <Button
          className="h-7 px-2.5 text-xs"
          disabled={submitting}
          onClick={() => decide('reject')}
          size="sm"
          type="button"
          variant="outline"
        >
          {t.agent.timeline.reject}
        </Button>
      </div>
    </div>
  )
}

/** The kernel's per-tool consent gate (E16: Standard keeps web contact
 * behind an approval). Rendered as a plan gate before, whose approve
 * button silently did nothing — usePlanApproval only decides
 * plan/replan approvals. */
function ToolGate({
  actions,
  approval,
  run,
  t,
}: {
  actions: AgentTimelineActions
  approval: AgentApprovalRecord
  run: AgentRunRecord
  t: TranslationDictionary
}) {
  const [submitting, setSubmitting] = useState(false)
  const proposed = Array.isArray(approval.payload.actions)
    ? (approval.payload.actions as Record<string, unknown>[])
    : []
  const decide = (decision: 'approve' | 'reject') => {
    if (submitting) return
    setSubmitting(true)
    void actions
      .decideApproval(run.runId, approval.approvalId, { decision })
      .finally(() => setSubmitting(false))
  }
  return (
    <div>
      <GateTitle pulse title={t.agent.timeline.waitingApprovalTitle} tone="warning" />
      <p className="mt-1 t-meta text-muted-foreground">
        {t.agent.timeline.toolApprovalHint}
      </p>
      <ul className="mt-1.5 space-y-0.5">
        {proposed.map((action, index) => (
          <li
            className="flex items-start gap-1.5 t-meta text-foreground/85"
            key={index}
          >
            <Search className="mt-0.5 icon-xs shrink-0 text-muted-foreground" />
            <span className="min-w-0 break-words">
              <span className="font-medium">
                {String(action.tool ?? '')}
              </span>
              {(() => {
                const args = action.args as Record<string, unknown> | undefined
                const query = args && typeof args.query === 'string'
                  ? args.query
                  : ''
                return query ? ` · ${query}` : ''
              })()}
            </span>
          </li>
        ))}
      </ul>
      <div className="mt-2 flex items-center gap-1.5">
        <Button
          className="h-7 bg-brand px-2.5 text-xs text-brand-foreground hover:bg-brand/90"
          disabled={submitting}
          onClick={() => decide('approve')}
          size="sm"
          type="button"
        >
          {t.agent.timeline.approve}
        </Button>
        <Button
          className="h-7 px-2.5 text-xs"
          disabled={submitting}
          onClick={() => decide('reject')}
          size="sm"
          type="button"
          variant="outline"
        >
          {t.agent.timeline.reject}
        </Button>
      </div>
    </div>
  )
}


function DiscoveryProbeRow({
  probe,
  t,
}: {
  probe: Record<string, unknown>
  t: TranslationDictionary
}) {
  const display = discoveryProbeDisplay(probe, t)
  return (
    <li className="flex items-start gap-1.5 t-meta text-foreground/85">
      <Search className="mt-0.5 icon-xs shrink-0 text-muted-foreground" />
      <span className="min-w-0 break-words">
        <span className="font-medium">{display.title}</span>
        {display.detail ? ` · ${display.detail}` : ''}
      </span>
    </li>
  )
}

function PatchGate({
  actions,
  run,
  t,
}: {
  actions: AgentTimelineActions
  run: AgentRunRecord
  t: TranslationDictionary
}) {
  const review = usePatchReview({
    actions: {
      applyPatch: actions.applyPatch,
      decideApproval: actions.decideApproval,
      rejectPatch: actions.rejectPatch,
    },
    run,
  })
  const patch = run.patch
  if (!patch || !run.patchId) return null
  return (
    <div>
      <GateTitle pulse title={t.agent.patch.title} tone="brand" />
      <p className="mt-1 t-meta text-foreground/90">
        {t.agent.patch.editCount.replace(
          '{count}',
          String(patch.edits.length),
        )}
        {patch.summary ? ` · ${patch.summary}` : ''}
      </p>
      {review.notice && (
        <p className="mt-1 t-meta-sm text-warning">{review.notice}</p>
      )}
      <div className="mt-2 flex items-center gap-1.5">
        <Button
          className="h-7 gap-1 bg-brand px-2.5 text-xs text-brand-foreground hover:bg-brand/90"
          onClick={() =>
            run.patchId
            && actions.onOpenCanvas({
              patchId: run.patchId,
              runId: run.runId,
              view: 'patch',
            })}
          size="sm"
          type="button"
        >
          <PenLine className="size-3" />
          {t.agent.tray.view}
        </Button>
        <Button
          className="h-7 px-2.5 text-xs"
          disabled={review.submitting}
          onClick={() => void review.approveAndApply()}
          size="sm"
          type="button"
          variant="outline"
        >
          {t.agent.patch.applyAndApprove}
        </Button>
        <Button
          className="h-7 px-2.5 text-xs text-muted-foreground hover:text-destructive"
          disabled={review.submitting}
          onClick={() => void review.reject()}
          size="sm"
          type="button"
          variant="ghost"
        >
          {t.agent.timeline.reject}
        </Button>
      </div>
    </div>
  )
}

function GateTitle({
  pulse,
  title,
  tone,
}: {
  pulse: boolean
  title: string
  tone: 'brand' | 'warning'
}) {
  const reduceMotion = Boolean(useReducedMotion())
  return (
    <div className="flex items-center gap-1.5">
      <span
        aria-hidden="true"
        className={cn(
          'size-1.5 rounded-full',
          tone === 'warning' ? 'bg-warning' : 'bg-brand',
          pulse && !reduceMotion && 'inqtrix-running-dot',
        )}
      />
      <span className="t-card text-foreground">{title}</span>
    </div>
  )
}

function appMotionSafe(reduceMotion: boolean) {
  return reduceMotion
    ? { duration: 0 }
    : { damping: 30, mass: 0.9, stiffness: 380, type: 'spring' as const }
}
