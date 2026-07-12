import { describe, expect, it } from 'vitest'

import type { AgentArtifactRecord, AgentPlanTaskRecord } from './model'
import {
  agentRunCompletionRecap,
  answerClampDecision,
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

describe('answerClampDecision', () => {
  it('clamps only settled answers above the cap', () => {
    expect(answerClampDecision(600, false)).toBe('clamped')
    expect(answerClampDecision(200, false)).toBe('full')
  })

  it('never clamps while streaming or before measurement', () => {
    expect(answerClampDecision(600, true)).toBe('full')
    expect(answerClampDecision(null, false)).toBe('full')
  })
})
