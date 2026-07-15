import { describe, expect, it } from 'vitest'

import type { AgentArtifactRecord, AgentPlanTaskRecord } from './model'
import { agentRunCompletionRecap } from './runPresentation'

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
