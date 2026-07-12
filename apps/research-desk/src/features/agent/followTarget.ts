/**
 * Follow-the-agent routing (plan §5.3), the PURE part: derive the view the
 * canvas should show from the run RECORD state — not from raw events, so a
 * reconnect/replay lands on the same target as a live stream. Only
 * lifecycle-level state routes; child progress never does. The timing
 * discipline (dwell, hysteresis, interaction deferral) lives in
 * `useCanvasFollow`; the pin/auto-open discipline lives in the reducer.
 */

import type { CanvasViewDescriptor } from '@/features/canvas/types'
import type { AgentRunRecord } from './model'

export type FollowTarget = {
  descriptor: CanvasViewDescriptor
  /**
   * `synthesis` wins against dwell/hysteresis (the memo streaming in is
   * the moment the user is waiting for); `open-only` targets never open
   * a closed canvas (anti auto-open); `auto-open` may open it once.
   */
  urgency: 'auto-open' | 'open-only' | 'synthesis'
}

export function routeAgentRunToView(
  run: AgentRunRecord,
): FollowTarget | null {
  const memoId = run.artifactOrder.find(
    (artifactId) => run.artifacts[artifactId]?.kind === 'memo',
  )
  const memo = memoId ? run.artifacts[memoId] : undefined

  // The memo being written (or finished) is the strongest target: the
  // first `artifact.created` is the ONE allowed auto-open.
  if (memo && (memo.status === 'writing' || run.status === 'completed')) {
    return {
      descriptor: {
        view: 'document',
        runId: run.runId,
        artifactId: memo.artifactId,
      },
      urgency: memo.status === 'writing' ? 'synthesis' : 'auto-open',
    }
  }

  // Execution follows the control-room overview. Task details are an explicit
  // user drill-down; auto-entering one parallel task hides the other work.
  if (run.phase === 'execution') {
    return {
      descriptor: { view: 'run', runId: run.runId },
      urgency: 'open-only',
    }
  }

  // A pending PATCH gate shows the proposed edits, not the plan — the
  // decision the user owes is about the patch (M7).
  if (
    run.status === 'waiting_for_approval'
    && run.patchId
    && run.approvals.some(
      (approval) =>
        approval.status === 'pending' && approval.kind === 'patch',
    )
  ) {
    return {
      descriptor: {
        view: 'patch',
        runId: run.runId,
        patchId: run.patchId,
      },
      urgency: 'open-only',
    }
  }

  // Plan negotiation: show the plan — but never auto-open for it
  // (plan.proposed/approval.requested route only into an OPEN canvas).
  if (
    run.status === 'waiting_for_approval'
    || run.phase === 'planning'
  ) {
    return {
      descriptor: { view: 'plan', runId: run.runId },
      urgency: 'open-only',
    }
  }

  return null
}
