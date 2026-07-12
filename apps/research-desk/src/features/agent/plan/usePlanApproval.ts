import { useCallback, useMemo, useRef, useState } from 'react'

import type {
  AgentPlanRecord,
  AgentPlanTaskRecord,
  AgentRunRecord,
} from '../model'
import type { AgentApprovalDecisionRequest } from '../types'

/** Editable projection of one plan task (v1 scope: title, query strings,
 * delete, reorder, profile + model tier params; plan §5.4). */
export type AgentPlanTaskDraft = {
  taskId: string
  title: string
  toolKind: AgentPlanTaskRecord['toolKind']
  objective: string
  queries: string[]
  gapIds: string[]
  dependsOn: string[]
  budget: Record<string, unknown>
  params: Record<string, unknown>
  expectedOutput: string
  isFalsification: boolean
}

export type AgentPlanDraft = {
  version: number
  summaryMarkdown: string
  assumptions: string[]
  successCriteria: string[]
  tasks: AgentPlanTaskDraft[]
  /** Decision-scoped user guidance for the REPORT (structure, focus,
   * audience). Travels with approve/edit as `report_guidance`; never
   * part of the wire plan, so guidance alone never turns an approve
   * into an edit. */
  reportGuidance: string
}

export function draftFromPlan(
  plan: AgentPlanRecord,
  depth: string | undefined = undefined,
): AgentPlanDraft {
  return withEffectiveResearchProfile(rawDraftFromPlan(plan), depth)
}

function rawDraftFromPlan(plan: AgentPlanRecord): AgentPlanDraft {
  return {
    version: plan.version,
    summaryMarkdown: plan.summaryMarkdown,
    assumptions: [...plan.assumptions],
    successCriteria: [...plan.successCriteria],
    reportGuidance: '',
    tasks: plan.tasks.map((task) => ({
      taskId: task.taskId,
      title: task.title,
      toolKind: task.toolKind,
      objective: task.objective,
      queries: [...task.queries],
      gapIds: [...task.gapIds],
      dependsOn: [...task.dependsOn],
      // Budgets are server-managed. Legacy rows may still carry one, but the
      // UI has no budget editor and must never round-trip an uneditable value.
      budget: {},
      params: { ...task.params },
      expectedOutput: task.expectedOutput,
      isFalsification: task.isFalsification,
    })),
  }
}

/** The wire plan for `decision: 'edit'` — the SAME ExecutionPlanModel shape
 * the agent's planner emits, so both pass the one backend validator. */
export function draftToWirePlan(draft: AgentPlanDraft): Record<string, unknown> {
  return {
    summary_markdown: draft.summaryMarkdown,
    assumptions: draft.assumptions,
    success_criteria: draft.successCriteria,
    tasks: draft.tasks.map((task) => ({
      id: task.taskId,
      title: task.title,
      tool_kind: task.toolKind,
      objective: task.objective,
      queries: task.queries,
      gap_ids: task.gapIds,
      depends_on: task.dependsOn,
      budget: {},
      params: task.params,
      expected_output: task.expectedOutput,
      is_falsification: task.isFalsification,
    })),
  }
}

function withEffectiveResearchProfile(
  draft: AgentPlanDraft,
  depth: string | undefined,
): AgentPlanDraft {
  // Fill only a MISSING profile: the server already stamped the
  // tier-correct default (e.g. `schnell` under the Gruendlich Stufe) —
  // overwriting it here would silently deepen the search AND turn every
  // plain approve into an edit.
  const fallback = depth === 'deep' ? 'deep' : 'compact'
  let changed = false
  const tasks = draft.tasks.map((task) => {
    if (task.toolKind !== 'web_research' || task.params.profile) return task
    changed = true
    return { ...task, params: { ...task.params, profile: fallback } }
  })
  return changed ? { ...draft, tasks } : draft
}

export function planDraftDiffers(
  draft: AgentPlanDraft,
  plan: AgentPlanRecord,
): boolean {
  return JSON.stringify(draftToWirePlan(draft))
    !== JSON.stringify(draftToWirePlan(rawDraftFromPlan(plan)))
}

/**
 * The ONE plan-approval state machine shared by the compact timeline card
 * and the spacious canvas plan view (plan §5.4 — one implementation, two
 * densities). The edit draft lives in the REDUCER (`agentPlanDrafts`,
 * threaded via `draft`/`onDraftChange`) so edits made on one surface are
 * visible — and submitted — from the other. Idempotent decide guard: a
 * double-click cannot fire two decisions.
 */
export function usePlanApproval({
  decideApproval,
  draft,
  onDraftChange,
  run,
}: {
  decideApproval: (
    runId: string,
    approvalId: string,
    decision: AgentApprovalDecisionRequest,
  ) => Promise<unknown>
  draft: AgentPlanDraft | null
  onDraftChange: (draft: AgentPlanDraft | null) => void
  run: AgentRunRecord
}) {
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const decidedRef = useRef(new Set<string>())

  const pendingApproval = useMemo(
    () =>
      run.approvals.find(
        (approval) =>
          approval.status === 'pending'
          && (approval.kind === 'plan' || approval.kind === 'replan'),
      ) ?? null,
    [run.approvals],
  )

  const plan = run.plan ?? null
  const effectiveDraft = useMemo(() => {
    if (!plan) return null
    const base = draft && draft.version === plan.version
      ? draft
      : draftFromPlan(plan, run.depth)
    return withEffectiveResearchProfile(base, run.depth)
  }, [draft, plan, run.depth])

  const updateDraft = useCallback(
    (update: (draft: AgentPlanDraft) => AgentPlanDraft) => {
      if (!plan) return
      const base =
        draft && draft.version === plan.version
          ? draft
          : draftFromPlan(plan, run.depth)
      onDraftChange(update(withEffectiveResearchProfile(base, run.depth)))
    },
    [draft, onDraftChange, plan, run.depth],
  )

  const decide = useCallback(
    async (decision: 'approve' | 'reject', note?: string) => {
      if (!pendingApproval || !plan) return
      const key = `${pendingApproval.approvalId}:${decision}`
      if (decidedRef.current.has(key) || submitting) return
      decidedRef.current.add(key)
      setSubmitting(true)
      setError(null)
      try {
        const edited =
          decision === 'approve'
          && effectiveDraft !== null
          && planDraftDiffers(effectiveDraft, plan)
        const guidance = (effectiveDraft?.reportGuidance ?? '').trim()
        const request: AgentApprovalDecisionRequest = edited
          ? {
            decision: 'edit',
            note,
            plan: draftToWirePlan(effectiveDraft),
            ...(decision === 'approve' && guidance
              ? { report_guidance: guidance }
              : {}),
          }
          : {
            decision,
            note,
            ...(decision === 'approve' && guidance
              ? { report_guidance: guidance }
              : {}),
          }
        await decideApproval(run.runId, pendingApproval.approvalId, request)
        onDraftChange(null)
      } catch (caught) {
        decidedRef.current.delete(key)
        setError(caught instanceof Error ? caught.message : String(caught))
      } finally {
        setSubmitting(false)
      }
    },
    [
      decideApproval,
      effectiveDraft,
      onDraftChange,
      pendingApproval,
      plan,
      run.runId,
      submitting,
    ],
  )

  return {
    decide,
    draft: effectiveDraft,
    error,
    pendingApproval,
    submitting,
    updateDraft,
  }
}
