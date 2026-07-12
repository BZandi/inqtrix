import type {
  AgentArtifactRecord,
  AgentPlanTaskRecord,
  AgentRunRecord,
} from './model'
import { effectiveAgentTaskStatus } from './plan/taskPresentation'

export type AgentRunCompletionRecap = {
  elapsedSeconds: number | undefined
  referenceCount: number
  reviewed: boolean
  synthesized: boolean
  taskCount: number
  tasksCompleted: number
}

/** Evidence-backed facts for a completed run's compact return-state. */
export function agentRunCompletionRecap(
  run: Pick<
    AgentRunRecord,
    'elapsedSeconds' | 'phase' | 'station' | 'stepLog' | 'taskStates'
  >,
  tasks: readonly AgentPlanTaskRecord[],
  memo: AgentArtifactRecord | undefined,
): AgentRunCompletionRecap {
  return {
    elapsedSeconds: run.elapsedSeconds,
    referenceCount: memo?.refsCount ?? 0,
    reviewed: run.station === 'critic'
      || run.phase === 'done'
      || run.stepLog.some(
        (entry) => entry.kind === 'phase' && entry.phase === 'critic',
      ),
    synthesized: Boolean(memo) || run.stepLog.some(
      (entry) => entry.kind === 'phase' && entry.phase === 'synthesis',
    ),
    taskCount: tasks.length,
    tasksCompleted: tasks.filter(
      (task) => effectiveAgentTaskStatus(
        task,
        run.taskStates[task.taskId],
      ) === 'completed',
    ).length,
  }
}

/** Collapsed height cap of the inline chat answer (18rem = `max-h-72`):
 * the full result lives in the canvas; the transcript shows a preview. */
export const ANSWER_CLAMP_CAP_PX = 288

/**
 * Whether the settled inline answer collapses behind the canvas
 * affordance. Streaming answers always render in full (the active run
 * is the live surface); short answers stay untouched — only a settled
 * body taller than the cap clamps.
 */
export function answerClampDecision(
  scrollHeightPx: number | null,
  writing: boolean,
): 'clamped' | 'full' {
  if (writing || scrollHeightPx === null) return 'full'
  return scrollHeightPx > ANSWER_CLAMP_CAP_PX ? 'clamped' : 'full'
}
