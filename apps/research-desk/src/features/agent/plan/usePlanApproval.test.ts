import { describe, expect, it } from 'vitest'

import type { AgentPlanRecord } from '../model'
import { draftFromPlan, draftToWirePlan, planDraftDiffers } from './usePlanApproval'

function plan(): AgentPlanRecord {
  return {
    assumptions: [],
    createdAt: 1,
    createdBy: 'agent',
    planId: 'p1',
    reason: '',
    status: 'pending',
    successCriteria: [],
    summaryMarkdown: '',
    tasks: [{
      budget: { timeout_seconds: 99 },
      childRunId: null,
      dependsOn: [],
      expectedOutput: '',
      gapIds: [],
      isFalsification: false,
      objective: 'Research one topic',
      ordinal: 0,
      params: { profile: 'compact' },
      queries: ['Question'],
      resultSummary: '',
      status: 'pending',
      taskId: 't1',
      title: 'Research',
      toolKind: 'web_research',
    }],
    version: 1,
    versions: [],
  }
}

describe('plan draft normalization', () => {
  it('preserves the server-stamped profile and strips legacy budgets', () => {
    // The server already stamped the tier-correct profile; the draft must
    // never silently deepen it (that would also turn approve into edit).
    const deepDraft = draftFromPlan(plan(), 'deep')
    expect(deepDraft.tasks[0].params.profile).toBe('compact')
    expect(deepDraft.tasks[0].budget).toEqual({})
    const wire = draftToWirePlan(deepDraft) as {
      tasks: Array<{ budget: Record<string, unknown>; params: Record<string, unknown> }>
    }
    expect(wire.tasks[0]).toMatchObject({ budget: {}, params: { profile: 'compact' } })
  })

  it('fills only a MISSING research profile from the run depth', () => {
    const bare = plan()
    bare.tasks[0].params = {}
    expect(draftFromPlan(bare, 'deep').tasks[0].params.profile).toBe('deep')
    expect(draftFromPlan(bare).tasks[0].params.profile).toBe('compact')
  })

  it('does not turn budget cleanup alone into an edited approval', () => {
    expect(planDraftDiffers(draftFromPlan(plan()), plan())).toBe(false)
  })

  it('keeps the schnell profile as a plain approve, not an edit', () => {
    const fast = plan()
    fast.tasks[0].params.profile = 'schnell'
    expect(planDraftDiffers(draftFromPlan(fast), fast)).toBe(false)
  })
})
