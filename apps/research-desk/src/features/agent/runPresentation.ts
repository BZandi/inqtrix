import type {
  AgentArtifactRecord,
  AgentPlanTaskRecord,
  AgentRunRecord,
  AgentStepEntry,
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

/** Add only runs observed before the authoritative listing settled to the
 * history baseline. Runs first observed afterwards are genuinely new turns. */
export function retainHydratedAgentRunIds(
  current: ReadonlySet<string>,
  observedRunIds: Iterable<string>,
  runsHydrated: boolean,
): ReadonlySet<string> {
  if (runsHydrated) return current
  const next = new Set(current)
  for (const runId of observedRunIds) next.add(runId)
  return next
}

/** A narration types only when its exact event arrived live. Legacy/demo
 * entries may animate on a new run, but never when remounted as history. */
export function shouldAnimateAgentNarration({
  entry,
  historicalRun,
  isLatest,
}: {
  entry: Pick<AgentStepEntry, 'arrivedLive'>
  historicalRun: boolean
  isLatest: boolean
}): boolean {
  if (!isLatest || entry.arrivedLive === false) return false
  return entry.arrivedLive === true || !historicalRun
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


/**
 * How long ago the task list was reported, when the run has moved on
 * since — else `null`.
 *
 * The list only changes when the model calls `write_todos`. A delegation
 * is ONE tool call that can own the run for tens of minutes, so a list
 * written before it kept naming a finished step as the current one: in
 * the run that motivated this, "structuring the assignment" stood as
 * in-progress for fifty minutes while a whole mission ran beneath it.
 *
 * The surface does not guess whether the list is still true. It states
 * the fact — when it was last reported — and lets the reader judge.
 * Nothing newer than the list means nothing to add.
 */
export function agentTodoReportAge(
  todosAt: number | undefined,
  stepLog: readonly Pick<AgentStepEntry, 'at'>[],
  now: number,
): number | null {
  if (!todosAt) return null
  const newestStep = stepLog.reduce(
    (latest, entry) => (entry.at > latest ? entry.at : latest),
    0,
  )
  if (newestStep <= todosAt) return null
  return Math.max(0, Math.round(now - todosAt))
}


/**
 * The document THIS turn offers to open, or `null` when it wrote none.
 *
 * A turn's own listing can lose its document: an update re-anchors the
 * artifact row to the newest run that touched it, so after a reload an
 * older turn no longer lists what it produced. That is what the
 * session-wide fallback is for.
 *
 * It must not lend a document to a turn that never wrote one. A kernel
 * turn that only answers in chat — and even offers to write a memo
 * "if you want" — then showed an "open memo" button pointing at some
 * other run's document, in the observed case a DELEGATED CHILD's memo,
 * which the parent session cannot address at all. The button opened
 * nothing.
 *
 * `touchedArtifacts` is the turn's own record of what it wrote (P9) and
 * survives the re-anchoring, so it decides whether the fallback applies.
 */
export function agentTurnDocumentTarget(
  ownArtifactId: string | undefined,
  runId: string,
  touchedArtifacts: readonly unknown[],
  sessionFallback: { artifactId: string; runId: string } | null | undefined,
): { artifactId: string; runId: string } | null {
  if (ownArtifactId) return { artifactId: ownArtifactId, runId }
  if (touchedArtifacts.length === 0) return null
  return sessionFallback ?? null
}
