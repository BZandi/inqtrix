import { describe, expect, it } from 'vitest'

import { translations } from '../../i18n/translations'
import type { AgentRunRecord } from './model'
import { activityText } from './timeline/AgentTimeline'

const t = translations.de

/** The fields `activityText` actually reads — a full run record would
 * bury what the case is about. */
function waitingForChildren(child: Record<string, unknown>): AgentRunRecord {
  return {
    children: {
      run_child: { childRunId: 'run_child', taskId: 'call_1', ...child },
    },
    status: 'waiting_for_children',
    taskStates: {},
  } as unknown as AgentRunRecord
}

describe('activityText while children work', () => {
  it('says what the child is doing, not what the parent last did', () => {
    // The regression: this fell through to the parent's own last tool
    // row, frozen since it delegated — one unchanging sentence for
    // twenty minutes, on both surfaces that render this text.
    expect(
      activityText(
        waitingForChildren({
          runStatus: 'running',
          snapshot: { phase: 'execution' },
          openTasks: [5],
          checkedAnswers: 12,
        }),
        t,
      ),
    ).toBe('Unterauftrag · Führt Aufgaben aus · Aufgabe 6 · 12 Belege geprüft')
  })

  it('names the parallel wave through the same path', () => {
    expect(
      activityText(
        waitingForChildren({
          runStatus: 'running',
          snapshot: { phase: 'execution' },
          openTasks: [0, 1, 2, 3, 4],
        }),
        t,
      ),
    ).toBe('Unterauftrag · Führt Aufgaben aus · 5 Aufgaben parallel')
  })

  it('falls through when the child has reported nothing yet', () => {
    // No invented progress: the parent's own readout takes over again.
    expect(
      activityText(waitingForChildren({ runStatus: 'running' }), t),
    ).not.toContain('Unterauftrag')
  })
})
