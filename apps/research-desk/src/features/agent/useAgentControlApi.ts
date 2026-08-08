import { useCallback, useEffect, useRef } from 'react'
import type { Dispatch } from 'react'

import {
  answerAgentRunClarification,
  applyEditorPatch,
  cancelAgentRunTask,
  decideAgentRunApproval,
  exportAgentRunArtifact,
  getAgentRunArtifact,
  getAgentRunPlan,
  getAgentRunTaskResult,
  getEditorDocument,
  getEditorPatch,
  hasHttpStatus,
  listAgentRunApprovals,
  listAgentRunArtifacts,
  listAgentRunClarifications,
  rejectEditorPatch,
  updateAgentRunArtifact,
  type InqtrixRequestError,
} from '@/api/inqtrixClient'
import type { ResearchDeskAction } from '@/features/researchDesk/state'
import type {
  AgentApprovalDecisionRequest,
  AgentClarificationAnswerRequest,
  AgentTaskResultWire,
} from './types'
import {
  TERMINAL_AGENT_TASK_STATUSES,
  canEditAgentRun,
  type AgentRunRecord,
} from './model'

export type ArtifactSaveResult =
  | { kind: 'saved'; revision: number }
  | { kind: 'conflict'; currentRevision: number | null }
  | { kind: 'locked' }

const TASK_RESULT_CACHE_MAX = 50

/**
 * Control-surface data layer of the Agent Desk. Rows are the truth (rule
 * R1): SSE events only flip `*Stale` flags on the run records — this hook
 * turns every stale flag into ONE fetch and stores the rows via the
 * reducer (which clears the flag, so the effect cannot loop). Decisions
 * and answers POST through here; each response embeds the RESUMED run
 * summary, which is upserted immediately (status waiting -> queued).
 */
export function useAgentControlApi({
  apiKey,
  dispatch,
  enabled,
  runs,
  workspaceId,
}: {
  apiKey: string | undefined
  dispatch: Dispatch<ResearchDeskAction>
  enabled: boolean
  runs: Record<string, AgentRunRecord>
  workspaceId: string
}) {
  const optionsRef = useRef({ apiKey, workspaceId })
  optionsRef.current = { apiKey, workspaceId }
  const inFlightRef = useRef(new Set<string>())
  const runsRef = useRef(runs)
  runsRef.current = runs
  const enabledRef = useRef(enabled)
  enabledRef.current = enabled
  const requireEditableRun = useCallback((runId: string) => {
    if (!canEditAgentRun(runsRef.current[runId])) {
      throw new Error('Shared view-only agent runs are read-only.')
    }
  }, [])
  // Cache entries remember the live attempt they were fetched under so a
  // re-run of the SAME terminal status still invalidates (see loadTaskResult).
  const taskResultCacheRef = useRef(
    new Map<string, { result: AgentTaskResultWire; attempt?: number }>(),
  )
  const taskResultInFlightRef = useRef(
    new Map<string, Promise<AgentTaskResultWire>>(),
  )
  // Keys whose PREFETCH failed once: hover/idle must not hammer a
  // failing endpoint on every event; the on-open fetch stays untouched
  // and re-tries loudly.
  const taskResultPrefetchFailedRef = useRef(new Set<string>())

  useEffect(() => {
    if (!enabled) return
    for (const run of Object.values(runs)) {
      if (run.kind !== 'agent') continue
      if (run.planStale) {
        void refetch(`${run.runId}:plan`, async () => {
          try {
            const plan = await getAgentRunPlan(
              run.runId,
              undefined,
              optionsRef.current,
            )
            dispatch({ plan, runId: run.runId, type: 'setAgentRunPlan' })
          } catch (error) {
            if (hasHttpStatus(error, 404)) {
              // No plan yet (pre-planning) — clear the flag, keep nothing.
              dispatch({ plan: null, runId: run.runId, type: 'setAgentRunPlan' })
              return
            }
            throw error
          }
        })
      }
      if (run.approvalsStale) {
        void refetch(`${run.runId}:approvals`, async () => {
          const approvals = await listAgentRunApprovals(
            run.runId,
            optionsRef.current,
          )
          dispatch({
            approvals,
            runId: run.runId,
            type: 'setAgentRunApprovals',
          })
        })
      }
      if (run.clarificationsStale) {
        void refetch(`${run.runId}:clarifications`, async () => {
          const clarifications = await listAgentRunClarifications(
            run.runId,
            optionsRef.current,
          )
          dispatch({
            clarifications,
            runId: run.runId,
            type: 'setAgentRunClarifications',
          })
        })
      }
      if (run.patchStale && run.patchId) {
        const patchId = run.patchId
        void refetch(`${run.runId}:patch`, async () => {
          const patch = await getEditorPatch(patchId, optionsRef.current)
          dispatch({ patch, runId: run.runId, type: 'setAgentRunPatch' })
        })
      }
      if (run.artifactsStale) {
        void refetch(`${run.runId}:artifacts`, async () => {
          const artifacts = await listAgentRunArtifacts(
            run.runId,
            optionsRef.current,
          )
          dispatch({
            artifacts,
            runId: run.runId,
            type: 'setAgentRunArtifacts',
          })
        })
      }
      // Chat answers render INLINE in the transcript, so their body is
      // auto-fetched (unlike canvas artifacts, which load on tab open).
      // Loop-safe: the reducer drops a stale body when the list shows a
      // newer revision — `contentMarkdown === undefined` is therefore
      // exactly "detail missing or outdated", and the fetched detail
      // makes it defined again.
      if (!run.artifactsStale) {
        const answer = run.artifactOrder
          .map((id) => run.artifacts[id])
          .find((artifact) => artifact?.kind === 'answer')
        if (
          answer
          && answer.status !== 'writing'
          && (
            answer.contentMarkdown === undefined
            || answer.publicationNeedsReconcile === true
          )
        ) {
          void refetch(`${run.runId}:answer`, async () => {
            const artifact = await getAgentRunArtifact(
              run.runId,
              answer.artifactId,
              undefined,
              optionsRef.current,
            )
            dispatch({
              artifact,
              runId: run.runId,
              type: 'setAgentRunArtifactDetail',
            })
          })
        }
      }
    }

    function refetch(key: string, fetcher: () => Promise<void>) {
      if (inFlightRef.current.has(key)) return Promise.resolve()
      inFlightRef.current.add(key)
      return fetcher()
        .catch((error: unknown) => {
          // Loud, never silent: the run card shows the fetch failure. The
          // surface's stale flag clears with it (no tight retry loop).
          const [runId, surface] = key.split(':')
          dispatch({
            message: error instanceof Error ? error.message : String(error),
            runId,
            surface: surface as
              | 'plan'
              | 'approvals'
              | 'clarifications'
              | 'artifacts'
              | 'answer'
              | 'patch',
            type: 'markAgentRunError',
          })
        })
        .finally(() => {
          inFlightRef.current.delete(key)
        })
    }
  }, [dispatch, enabled, runs])

  const decideApproval = useCallback(
    async (
      runId: string,
      approvalId: string,
      decision: AgentApprovalDecisionRequest,
    ) => {
      requireEditableRun(runId)
      const result = await decideAgentRunApproval(
        runId,
        approvalId,
        decision,
        optionsRef.current,
      )
      const { run, ...approval } = result
      dispatch({ summary: run, type: 'upsertAgentRunSummary' })
      dispatch({ approvals: [approval], runId, type: 'setAgentRunApprovals' })
      await refreshPlanAfterApprovalDecision({
        kind: approval.kind,
        load: () => getAgentRunPlan(
          runId,
          undefined,
          optionsRef.current,
        ),
        onLoaded: (plan) => {
          dispatch({ plan, runId, type: 'setAgentRunPlan' })
        },
        onError: (message) => {
          dispatch({
            message,
            runId,
            surface: 'plan',
            type: 'markAgentRunError',
          })
        },
      })
      // The full list (older approvals) refreshes via the decided event;
      // seeding the fresh row keeps the card responsive meanwhile.
      return approval
    },
    [dispatch, requireEditableRun],
  )

  const answerClarification = useCallback(
    async (
      runId: string,
      clarificationId: string,
      answer: AgentClarificationAnswerRequest,
    ) => {
      requireEditableRun(runId)
      const result = await answerAgentRunClarification(
        runId,
        clarificationId,
        answer,
        optionsRef.current,
      )
      const { run, ...clarification } = result
      dispatch({ summary: run, type: 'upsertAgentRunSummary' })
      dispatch({
        clarifications: [clarification],
        runId,
        type: 'setAgentRunClarifications',
      })
      return clarification
    },
    [dispatch, requireEditableRun],
  )

  const loadArtifact = useCallback(
    async (runId: string, artifactId: string, revision?: number) => {
      const artifact = await getAgentRunArtifact(
        runId,
        artifactId,
        revision,
        optionsRef.current,
      )
      // Pinned historical revisions are view-only; only the LATEST body
      // may enter the record (the canvas edits against it).
      if (revision === undefined) {
        dispatch({ artifact, runId, type: 'setAgentRunArtifactDetail' })
      }
      return artifact
    },
    [dispatch],
  )

  const loadTaskResult = useCallback(
    (runId: string, taskId: string): Promise<AgentTaskResultWire> => {
      const key = `${runId}:${taskId}`
      const cached = taskResultCacheRef.current.get(key)
      const liveTask = runsRef.current[runId]?.taskStates[taskId]
      // Serve the cache only while the live task still carries the status
      // AND attempt the result was fetched under — a retried task drops
      // its entry even when it lands on the same terminal status again.
      const cacheValid =
        cached !== undefined
        && (liveTask === undefined
          || (liveTask.status === cached.result.status
            && liveTask.attempt === cached.attempt))
      if (cached && cacheValid) {
        return Promise.resolve(cached.result)
      }
      if (cached) taskResultCacheRef.current.delete(key)
      const inFlight = taskResultInFlightRef.current.get(key)
      if (inFlight) return inFlight
      // Attempt at REQUEST time: a retry racing this fetch must not tag
      // the previous attempt's result with the new attempt number.
      const requestAttempt = liveTask?.attempt
      const request = getAgentRunTaskResult(runId, taskId, optionsRef.current)
        .then((result) => {
          // Success re-arms hover/idle prefetch for this task (the flag
          // cannot clear inside prefetchTaskResult — it early-returns).
          taskResultPrefetchFailedRef.current.delete(key)
          if (TERMINAL_AGENT_TASK_STATUSES.has(result.status)) {
            const cache = taskResultCacheRef.current
            cache.set(key, {
              attempt: requestAttempt,
              result,
            })
            if (cache.size > TASK_RESULT_CACHE_MAX) {
              const oldest = cache.keys().next().value
              if (oldest !== undefined) cache.delete(oldest)
            }
          }
          return result
        })
        .finally(() => {
          taskResultInFlightRef.current.delete(key)
        })
      taskResultInFlightRef.current.set(key, request)
      return request
    },
    [],
  )

  const prefetchTaskResult = useCallback(
    (runId: string, taskId: string) => {
      // Demo/offline: the canvas renders demo runs, but there is no
      // server to prefetch from (the detail view guards the same way
      // via clientOptions).
      if (!enabledRef.current) return
      const key = `${runId}:${taskId}`
      if (taskResultPrefetchFailedRef.current.has(key)) return
      void loadTaskResult(runId, taskId)
        .catch((error: unknown) => {
          // Visible, but not load-bearing: the on-open fetch in the task
          // detail view stays the loud error path and still retries.
          taskResultPrefetchFailedRef.current.add(key)
          console.warn(
            `Task result prefetch failed (${key}):`,
            error instanceof Error ? error.message : error,
          )
        })
    },
    [loadTaskResult],
  )

  const cancelTask = useCallback(
    async (runId: string, taskId: string) => {
      requireEditableRun(runId)
      const result = await cancelAgentRunTask(
        runId,
        taskId,
        optionsRef.current,
      )
      dispatch({
        childRunId: result.child_run_id,
        runId,
        status: result.status,
        taskId,
        type: 'ackAgentTaskCancel',
      })
      try {
        const plan = await getAgentRunPlan(
          runId,
          undefined,
          optionsRef.current,
        )
        dispatch({ plan, runId, type: 'setAgentRunPlan' })
      } catch (error) {
        dispatch({
          message: error instanceof Error ? error.message : String(error),
          runId,
          surface: 'plan',
          type: 'markAgentRunError',
        })
      }
      return result
    },
    [dispatch, requireEditableRun],
  )

  const saveArtifact = useCallback(
    async (
      runId: string,
      artifactId: string,
      contentMarkdown: string,
      expectedRevision: number,
    ): Promise<ArtifactSaveResult> => {
      requireEditableRun(runId)
      try {
        const result = await updateAgentRunArtifact(
          runId,
          artifactId,
          {
            content_markdown: contentMarkdown,
            expected_revision: expectedRevision,
          },
          optionsRef.current,
        )
        await loadArtifact(runId, artifactId)
        return { kind: 'saved', revision: result.revision }
      } catch (error) {
        if (hasHttpStatus(error, 409)) {
          const detail = (error as InqtrixRequestError).detail ?? {}
          if (detail.locked_by === 'agent') return { kind: 'locked' }
          await loadArtifact(runId, artifactId)
          const current = detail.current_revision
          return {
            kind: 'conflict',
            currentRevision: typeof current === 'number' ? current : null,
          }
        }
        throw error
      }
    },
    [loadArtifact, requireEditableRun],
  )

  const exportArtifact = useCallback(
    async (runId: string, artifactId: string, title?: string) => {
      return exportAgentRunArtifact(
        runId,
        artifactId,
        { target: 'editor_document', title },
        optionsRef.current,
      )
    },
    [],
  )

  const applyPatch = useCallback(
    async (
      runId: string,
      patchId: string,
      expectedRevision: number,
    ): Promise<
      | { kind: 'applied'; revision: number; appliedEditIds: string[] }
      | { kind: 'conflict'; currentRevision: number | null }
    > => {
      requireEditableRun(runId)
      const refreshAfterApply = async () => {
        const patch = await getEditorPatch(patchId, optionsRef.current)
        dispatch({ patch, runId, type: 'setAgentRunPatch' })
        // The SERVER document just changed under the local editor copy —
        // pull the fresh body in (never dirties, never bumps updatedAt),
        // or the applied patch stays invisible until reload and the next
        // autosave of the stale copy would silently undo it.
        const document = await getEditorDocument(
          patch.document_id,
          optionsRef.current,
        )
        dispatch({
          contentMarkdown: document.content_markdown ?? '',
          documentId: patch.document_id,
          type: 'setServerEditorDocumentBody',
        })
        return patch
      }
      try {
        const result = await applyEditorPatch(
          patchId,
          expectedRevision,
          optionsRef.current,
        )
        await refreshAfterApply()
        return {
          kind: 'applied',
          revision: result.revision,
          appliedEditIds: result.applied_edit_ids,
        }
      } catch (error) {
        if (hasHttpStatus(error, 409)) {
          const detail = (error as InqtrixRequestError).detail ?? {}
          if (detail.status === 'accepted') {
            // Already applied (other tab / replayed decide) — the stored
            // outcome IS the success, not a conflict.
            const patch = await refreshAfterApply()
            return {
              kind: 'applied',
              revision: patch.applied_revision ?? patch.document_revision,
              appliedEditIds: patch.applied_edit_ids ?? [],
            }
          }
          const patch = await getEditorPatch(patchId, optionsRef.current)
          dispatch({ patch, runId, type: 'setAgentRunPatch' })
          const current = detail.current_revision
          return {
            kind: 'conflict',
            currentRevision:
              typeof current === 'number' ? current : null,
          }
        }
        throw error
      }
    },
    [dispatch, requireEditableRun],
  )

  const rejectPatch = useCallback(
    async (runId: string, patchId: string, note: string) => {
      requireEditableRun(runId)
      const patch = await rejectEditorPatch(
        patchId,
        note,
        optionsRef.current,
      )
      dispatch({ patch, runId, type: 'setAgentRunPatch' })
      return patch
    },
    [dispatch, requireEditableRun],
  )

  return {
    answerClarification,
    applyPatch,
    cancelTask,
    decideApproval,
    exportArtifact,
    loadArtifact,
    loadTaskResult,
    prefetchTaskResult,
    rejectPatch,
    saveArtifact,
  }
}

export async function refreshPlanAfterApprovalDecision<Plan>({
  kind,
  load,
  onError,
  onLoaded,
}: {
  kind: string
  load: () => Promise<Plan>
  onError: (message: string) => void
  onLoaded: (plan: Plan) => void
}): Promise<void> {
  if (kind !== 'plan' && kind !== 'replan') return
  try {
    onLoaded(await load())
  } catch (error) {
    onError(error instanceof Error ? error.message : String(error))
  }
}
