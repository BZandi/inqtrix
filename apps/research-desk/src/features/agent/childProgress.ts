import type { TranslationDictionary } from '../../i18n/translations'
import type { AgentChildProgressRecord } from './model'

/** The child fields a parent line is allowed to speak from. */
export type ChildProgressView = Pick<
  AgentChildProgressRecord,
  'checkedAnswers' | 'error' | 'snapshot' | 'runStatus' | 'openTasks' | 'message'
>

const BROKEN_OFF: Record<string, 'childFailed' | 'childCancelled'> = {
  cancelled: 'childCancelled',
  failed: 'childFailed',
}

/**
 * What a delegating parent can honestly say about its child RIGHT NOW.
 *
 * A kernel run that hands a deep mission to a child used to show one
 * unchanging line for the child's entire lifetime — fifty minutes, in
 * the run that motivated this. The child was reporting its phase and its
 * tasks all along; nothing carried them across the run boundary, and
 * nothing rendered them.
 *
 * The line speaks only from what actually arrived. No phase and no task
 * means no line: a parent that invents progress is worse than a parent
 * that shows none.
 *
 * A child that COMPLETED gets no line — the delegation row's own check
 * already says so. A child that failed or was cancelled keeps one: the
 * tool call still returns normally in that case, so the row shows a
 * check either way, and without this line a failed subtask reads as a
 * successful one.
 */
export function childProgressLine(
  child: ChildProgressView,
  t: TranslationDictionary,
): string | null {
  const status = child.runStatus ?? ''
  if (status === 'completed') return null
  const brokenOff = BROKEN_OFF[status]
  if (brokenOff) {
    const reason = child.error?.trim()
    const label = t.agent.timeline[brokenOff]
    return `${t.agent.timeline.childRun} · ${reason ? `${label}: ${reason}` : label}`
  }

  const parts: string[] = []
  if (status === 'waiting_for_approval' || status === 'waiting_for_input') {
    // The tray owns the decision; the line only says why nothing moves.
    parts.push(t.agent.timeline.childWaiting)
  } else {
    // `agent.activity`, not `agent.stations`: the line says what the
    // child is DOING ("Führt Aufgaben aus"), and it is the only map that
    // covers every mission phase — `stations` has no `evidence` entry.
    const phases = t.agent.activity as Record<string, string | undefined>
    const phase = child.snapshot?.phase
    const phaseLabel = phase ? phases[phase] : undefined
    if (phaseLabel) parts.push(phaseLabel)
  }

  const open = child.openTasks ?? []
  if (open.length > 1) {
    // A mission runs its wave in parallel. Naming ONE of five as "the"
    // current task read as if the other four did not exist — and the
    // phase hung on the slowest of them for thirteen minutes.
    parts.push(
      t.agent.activity.parallelTasks.replace('{count}', String(open.length)),
    )
  } else if (open.length === 1) {
    // Ordinals are zero-based on the wire and one-based for a reader.
    parts.push(
      t.agent.timeline.childTask.replace('{number}', String(open[0] + 1)),
    )
  }

  // The number that MOVES. Everything above can stand still for ten
  // minutes while the child works; this cannot, and it is the only part
  // of the line that answers "is it still alive".
  const checked = child.checkedAnswers ?? 0
  if (checked > 0) {
    parts.push(
      t.agent.timeline.childEvidence.replace('{count}', String(checked)),
    )
  }

  if (parts.length === 0) {
    const message = child.message?.trim()
    if (!message) return null
    parts.push(message)
  }
  return `${t.agent.timeline.childRun} · ${parts.join(' · ')}`
}
