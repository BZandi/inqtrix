import { createContext, useContext } from 'react'
import type { MutableRefObject } from 'react'

import type { ClientOptions } from '@/api/inqtrixClient'
import type { CanvasViewDescriptor } from '@/features/canvas/types'
import type { FileAssetRecord } from '@/features/project/types'
import type { AgentRunRecord } from '../model'
import type { PlanSourceInfo } from '../plan/sourceLabel'
import type { AgentPlanDraft } from '../plan/usePlanApproval'
import type {
  AgentApprovalDecisionRequest,
  AgentTaskCancelWire,
  AgentTaskResultWire,
} from '../types'
import type { ArtifactSaveResult } from '../useAgentControlApi'

/**
 * Data plane of the agent canvas views. A React context (not per-render
 * closures) so the view registry stays a MODULE CONSTANT — otherwise every
 * workspace render would recreate the renderer component types and
 * remount the active view (killing the editor cursor and the task view's
 * SSE subscription).
 */
export type AgentCanvasContextValue = {
  applyPatch: (
    runId: string,
    patchId: string,
    expectedRevision: number,
  ) => Promise<
    | { kind: 'applied'; revision: number; appliedEditIds: string[] }
    | { kind: 'conflict'; currentRevision: number | null }
  >
  clientOptions: ClientOptions | null
  cancelTask: (
    runId: string,
    taskId: string,
  ) => Promise<AgentTaskCancelWire>
  decideApproval: (
    runId: string,
    approvalId: string,
    decision: AgentApprovalDecisionRequest,
  ) => Promise<unknown>
  rejectPatch: (
    runId: string,
    patchId: string,
    note: string,
  ) => Promise<unknown>
  exportArtifact: (
    runId: string,
    artifactId: string,
    title?: string,
  ) => Promise<unknown>
  fileAssets: Record<string, FileAssetRecord>
  loadArtifact: (
    runId: string,
    artifactId: string,
    revision?: number,
  ) => Promise<{ content_markdown: string; revision: number }>
  loadTaskResult: (
    runId: string,
    taskId: string,
  ) => Promise<AgentTaskResultWire>
  /** Fire-and-forget cache warm for a settled task's result (hover /
   * idle) — errors stay silent, the on-open fetch is the loud path. */
  prefetchTaskResult: (runId: string, taskId: string) => void
  openCanvasView: (descriptor: CanvasViewDescriptor) => void
  /** Re-flags the run's plan as stale so the control hook refetches it —
   * the plan view's on-open trigger (rows stay the truth, rule R1). */
  requestPlanRefresh: (runId: string) => void
  /** Registered by the document view while an edit is pending; the
   * composer awaits it before a follow-up submit (plan §5.4 flush rule). */
  pendingSaveRef: MutableRefObject<(() => Promise<void>) | null>
  planDrafts: Record<string, AgentPlanDraft>
  /** Collection titles + vector-backend label for the plan view's
   * per-task source line (same data plane as the timeline card). */
  planSource: PlanSourceInfo
  /** Parent runs currently using the visible polling fallback. */
  pollingRunIds: readonly string[]
  runs: Record<string, AgentRunRecord>
  saveArtifact: (
    runId: string,
    artifactId: string,
    contentMarkdown: string,
    expectedRevision: number,
  ) => Promise<ArtifactSaveResult>
  setPlanDraft: (runId: string, draft: AgentPlanDraft | null) => void
  workspaceId: string
}

export const AgentCanvasReactContext =
  createContext<AgentCanvasContextValue | null>(null)

export function useAgentCanvas(): AgentCanvasContextValue {
  const value = useContext(AgentCanvasReactContext)
  if (!value) {
    throw new Error('Agent canvas views require the AgentCanvasReactContext')
  }
  return value
}
