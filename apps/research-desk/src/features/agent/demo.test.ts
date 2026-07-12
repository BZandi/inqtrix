import { describe, expect, it } from 'vitest'

import { demoPlan, demoRunSummary } from './demo'
import { normalizeAgentExecutionSnapshot } from './executionPolicy'

describe('Agent Desk demo contract twin', () => {
  it('publishes the selected run contract in the canonical execution block', () => {
    const summary = demoRunSummary(
      'RA-demo-test',
      'session-test',
      'Question',
      'balanced',
      'running',
      'deep',
      'agent_kernel',
      'auto',
      { web: 'disabled', knowledge: 'available' },
      null,
      { model: 'model-test', effort: 'high', tier: null },
    )

    expect(summary.mode).toBe('agent_kernel')
    expect(summary.agent_overrides).toMatchObject({
      autonomy: 'balanced',
      depth: 'deep',
      effort: 'high',
      model: 'model-test',
      source_policy: { web: 'disabled', knowledge: 'available' },
    })
    expect(normalizeAgentExecutionSnapshot(summary.snapshot)).toMatchObject({
      consentReason: 'permission_policy',
      depth: 'deep',
      effectiveMode: 'agent_kernel',
      model: 'model-test',
      reasoningEffort: 'high',
      responseForm: 'chat',
      sourcePolicy: { web: 'disabled', knowledge: 'available' },
      toolUseCounts: { web: 0, knowledge: 0 },
    })
  })

  it('applies a strict quick-web directive exactly like the accepted server route', () => {
    const summary = demoRunSummary(
      'RA-demo-web',
      'session-test',
      'Current question',
      'strict',
      'running',
      'deep',
      'workspace_agent',
      'canvas',
      { web: 'disabled', knowledge: 'available' },
      'quick_web',
    )

    expect(summary.mode).toBe('agent_kernel')
    expect(normalizeAgentExecutionSnapshot(summary.snapshot)).toMatchObject({
      consentReason: 'strict_approval_required',
      depth: 'normal',
      effectiveMode: 'agent_kernel',
      executionDirective: 'quick_web',
      responseForm: 'chat',
      sourcePolicy: { web: 'available', knowledge: 'disabled' },
    })
  })

  it('removes disabled sources from the plan and its dependency graph', () => {
    const knowledgeOnly = demoPlan(
      'RA-demo-knowledge',
      [],
      { web: 'disabled', knowledge: 'available' },
    )
    expect(knowledgeOnly.tasks.map((task) => task.tool_kind)).toEqual([
      'rag_query',
      'synthesis',
    ])
    expect(knowledgeOnly.tasks[1].depends_on).toEqual(['t1'])

    const noSources = demoPlan(
      'RA-demo-none',
      [],
      { web: 'disabled', knowledge: 'disabled' },
    )
    expect(noSources.tasks.map((task) => task.tool_kind)).toEqual(['synthesis'])
    expect(noSources.tasks[0].depends_on).toEqual([])
  })
})
