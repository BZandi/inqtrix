import { describe, expect, it } from 'vitest'

import type { AgentArtifactRecord, AgentPlanTaskRecord } from './model'
import {
  agentRunCompletionRecap,
  agentTodoReportAge,
  agentTurnDocumentTarget,
  retainHydratedAgentRunIds,
  shouldAnimateAgentNarration,
} from './runPresentation'

const task = (taskId: string, status: string): AgentPlanTaskRecord => ({
  budget: {},
  childRunId: null,
  dependsOn: [],
  expectedOutput: '',
  gapIds: [],
  isFalsification: false,
  objective: '',
  ordinal: 0,
  params: {},
  queries: ['q'],
  resultSummary: '',
  status,
  taskId,
  title: taskId,
  toolKind: 'web_instant',
})

describe('agent run completion recap', () => {
  it('uses durable task rows and artifact metadata for a substantive return state', () => {
    const memo = { refsCount: 38 } as AgentArtifactRecord
    expect(agentRunCompletionRecap({
      elapsedSeconds: 863,
      phase: 'done',
      station: 'critic',
      stepLog: [],
      taskStates: {},
    }, [task('t1', 'completed'), task('t2', 'completed')], memo)).toEqual({
      elapsedSeconds: 863,
      referenceCount: 38,
      reviewed: true,
      synthesized: true,
      taskCount: 2,
      tasksCompleted: 2,
    })
  })
})

describe('agent transcript presentation provenance', () => {
  it('adds runs observed during hydration but not later live turns', () => {
    const initial = new Set(['cached'])
    const hydrating = retainHydratedAgentRunIds(
      initial,
      ['cached', 'persisted-page'],
      false,
    )
    expect([...hydrating]).toEqual(['cached', 'persisted-page'])

    const settled = retainHydratedAgentRunIds(
      hydrating,
      ['cached', 'persisted-page', 'new-live-turn'],
      true,
    )
    expect(settled).toBe(hydrating)
    expect(settled.has('new-live-turn')).toBe(false)
  })

  it('animates only the newest genuinely live narration', () => {
    expect(shouldAnimateAgentNarration({
      entry: { arrivedLive: false },
      historicalRun: false,
      isLatest: true,
    })).toBe(false)
    expect(shouldAnimateAgentNarration({
      entry: { arrivedLive: undefined },
      historicalRun: true,
      isLatest: true,
    })).toBe(false)
    expect(shouldAnimateAgentNarration({
      entry: { arrivedLive: true },
      historicalRun: true,
      isLatest: true,
    })).toBe(true)
    expect(shouldAnimateAgentNarration({
      entry: { arrivedLive: true },
      historicalRun: false,
      isLatest: false,
    })).toBe(false)
  })
})

describe('agentTodoReportAge', () => {
  it('says how old the list is once the run worked past it', () => {
    // The regression: a todo written before a delegation kept naming a
    // finished step as current for fifty minutes.
    expect(
      agentTodoReportAge(1000, [{ at: 1200 }, { at: 4020 }], 4000),
    ).toBe(3000)
  })

  it('adds nothing while the list is the newest thing that happened', () => {
    expect(agentTodoReportAge(1000, [{ at: 900 }], 4000)).toBeNull()
  })

  it('adds nothing when no list was ever reported', () => {
    expect(agentTodoReportAge(undefined, [{ at: 900 }], 4000)).toBeNull()
  })

  it('never reports a negative age', () => {
    expect(agentTodoReportAge(5000, [{ at: 6000 }], 4000)).toBe(0)
  })
})

describe('agentTurnDocumentTarget', () => {
  const fallback = { artifactId: 'art_session', runId: 'run_newer' }
  const touched = [{ artifactId: 'art_own', kind: 'memo' as const }]

  it('offers the turn’s own document', () => {
    expect(agentTurnDocumentTarget('art_own', 'run_1', touched, fallback))
      .toEqual({ artifactId: 'art_own', runId: 'run_1' })
  })

  it('falls back when the turn wrote one but lost its listing', () => {
    // An update re-anchors the row to the newest run that touched it.
    expect(agentTurnDocumentTarget(undefined, 'run_1', touched, fallback))
      .toEqual(fallback)
  })

  it('offers nothing when the turn wrote no document', () => {
    // The regression: a chat-only turn showed "open memo" pointing at a
    // DELEGATED CHILD's memo, which the parent session cannot address —
    // the button opened nothing.
    expect(agentTurnDocumentTarget(undefined, 'run_1', [], fallback))
      .toBeNull()
  })

  it('offers nothing when neither the turn nor the session has one', () => {
    expect(agentTurnDocumentTarget(undefined, 'run_1', touched, null))
      .toBeNull()
  })
})
