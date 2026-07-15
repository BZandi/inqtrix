import { describe, expect, it } from 'vitest'

import { acknowledgeAgentTaskCancellation, applyAgentRunEvent } from './events'
import { agentRunFromSummary } from './model'
import type { ResearchRunEvent, ResearchRunSummary } from '@/features/researchRuns/types'

function summary(overrides: Partial<ResearchRunSummary> = {}): ResearchRunSummary {
  return {
    access: { mode: 'owner' },
    run_id: 'run_agent_1',
    status: 'running',
    queue_position: null,
    question: 'Erstelle eine Marktanalyse.',
    stack: 'default',
    mode: 'workspace_agent',
    kind: 'agent',
    agent_overrides: { autonomy: 'balanced' },
    created_at: 1_700_000_000,
    started_at: 1_700_000_001,
    finished_at: null,
    elapsed_seconds: null,
    snapshot: { current_node: 'agent', last_message: '' },
    error: null,
    events_url: '/v1/runs/run_agent_1/events',
    result_url: '/v1/runs/run_agent_1/result',
    ...overrides,
  }
}

function event(
  sequence: number,
  type: string,
  data: Record<string, unknown> = {},
): ResearchRunEvent {
  return { type, run_id: 'run_agent_1', sequence, created_at: 1_700_000_100 + sequence, data }
}

describe('applyAgentRunEvent', () => {
  it('applies an HTTP task-cancel acknowledgement without waiting for SSE', () => {
    const record = {
      ...agentRunFromSummary(summary()),
      taskStates: {
        t1: { attempt: 1, status: 'running' as const },
      },
    }

    const acknowledged = acknowledgeAgentTaskCancellation(
      record,
      't1',
      'cancel_requested',
      null,
    )

    expect(acknowledged.taskStates.t1?.status).toBe('cancel_requested')
    expect(acknowledged.planStale).toBe(true)

    const completed = {
      ...record,
      taskStates: {
        t1: { attempt: 1, status: 'completed' as const },
      },
    }
    expect(acknowledgeAgentTaskCancellation(
      completed,
      't1',
      'cancel_requested',
      null,
    ).taskStates.t1?.status).toBe('completed')
    expect(applyAgentRunEvent(
      completed,
      event(1, 'inqtrix.agent.task.cancel_requested', {
        status: 'cancel_requested',
        task_id: 't1',
      }),
    ).taskStates.t1?.status).toBe('completed')
  })

  it('keeps an HTTP cancel acknowledgement monotone across delayed events', () => {
    const running = {
      ...agentRunFromSummary(summary()),
      taskStates: {
        t1: { attempt: 1, status: 'running' as const },
      },
    }
    let record = acknowledgeAgentTaskCancellation(
      running,
      't1',
      'cancel_requested',
      null,
    )

    record = applyAgentRunEvent(
      record,
      event(1, 'inqtrix.agent.task.started', {
        attempt: 1,
        task_id: 't1',
      }),
    )
    expect(record.taskStates.t1?.status).toBe('cancel_requested')

    record = applyAgentRunEvent(
      record,
      event(2, 'inqtrix.agent.child.progress', {
        child_run_id: 'run_child_1',
        run_status: 'running',
        task_id: 't1',
      }),
    )
    expect(record.taskStates.t1?.status).toBe('cancel_requested')

    record = applyAgentRunEvent(
      record,
      event(3, 'inqtrix.agent.child.progress', {
        child_run_id: 'run_child_1',
        run_status: 'completed',
        task_id: 't1',
      }),
    )
    expect(record.taskStates.t1?.status).toBe('cancel_requested')

    record = applyAgentRunEvent(
      record,
      event(4, 'inqtrix.agent.task.finished', {
        status: 'cancelled',
        task_id: 't1',
      }),
    )
    expect(record.taskStates.t1?.status).toBe('cancelled')
  })

  it('advances phase and station, never backwards on gate phases', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.phase.changed', { phase: 'discovery' }))
    expect(record.station).toBe('discovery')
    // clarification is a gate, not a station: track keeps discovery
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.phase.changed', { phase: 'clarification' }))
    expect(record.phase).toBe('clarification')
    expect(record.station).toBe('discovery')
    record = applyAgentRunEvent(record, event(3, 'inqtrix.agent.phase.changed', { phase: 'execution' }))
    expect(record.station).toBe('execution')
  })

  it('ignores replayed frames (sequence gate)', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(5, 'inqtrix.agent.phase.changed', { phase: 'planning' }))
    const replay = applyAgentRunEvent(record, event(5, 'inqtrix.agent.phase.changed', { phase: 'intake' }))
    expect(replay).toBe(record)
    expect(replay.station).toBe('planning')
  })

  it('flips stale flags on control-row signals instead of copying payloads', () => {
    let record = agentRunFromSummary(summary())
    record = { ...record, planStale: false, approvalsStale: false, clarificationsStale: false }
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.plan.proposed', { plan_id: 'p1', version: 1 }))
    expect(record.planStale).toBe(true)
    expect(record.plan).toBeUndefined()
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.approval.requested', { approval_id: 'a1', kind: 'plan' }))
    expect(record.approvalsStale).toBe(true)
    record = applyAgentRunEvent(record, event(3, 'inqtrix.agent.clarification.requested', { clarification_id: 'c1' }))
    expect(record.clarificationsStale).toBe(true)
  })

  it('records auto-approved replan versions for the timeline note', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.plan.revised', { version: 2, auto_approved: true }))
    expect(record.lastAutoApprovedVersion).toBe(2)
    expect(record.planStale).toBe(true)
  })

  it('marks the plan stale when a loaded plan approval is decided', () => {
    let record = agentRunFromSummary(summary())
    record = {
      ...record,
      planStale: false,
      approvals: [{
        approvalId: 'apr-plan',
        kind: 'plan',
        status: 'pending',
        subjectType: 'plan',
        subjectId: 'p1',
        payload: {},
        decision: '',
        note: '',
        createdAt: 1,
        decidedAt: null,
      }],
    }
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.approval.decided', {
      approval_id: 'apr-plan',
      status: 'approved',
    }))
    expect(record.planStale).toBe(true)
    expect(record.approvalsStale).toBe(true)
  })

  it('tracks task lifecycle with child attribution and new map references', () => {
    let record = agentRunFromSummary(summary())
    const before = record.taskStates
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.task.started', { task_id: 't1', ordinal: 0, tool_kind: 'web_research', attempt: 1 }))
    expect(record.taskStates.t1.status).toBe('running')
    expect(record.taskStates).not.toBe(before)
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.child.progress', {
      task_id: 't1',
      child_run_id: 'run_child_1',
      snapshot: { current_node: 'search' },
    }))
    expect(record.children.run_child_1.snapshot?.current_node).toBe('search')
    expect(record.taskStates.t1.childRunId).toBe('run_child_1')
    record = applyAgentRunEvent(record, event(3, 'inqtrix.agent.task.finished', {
      task_id: 't1',
      status: 'completed',
      child_run_id: 'run_child_1',
    }))
    expect(record.taskStates.t1.status).toBe('completed')
    expect(record.planStale).toBe(true)
  })

  it('marks failed tasks with their error and outcome', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.task.failed', {
      task_id: 't2',
      status: 'failed',
      error: 'Deadline erreicht',
    }))
    expect(record.taskStates.t2).toMatchObject({ status: 'failed', error: 'Deadline erreicht' })
  })

  it('keeps cancellation requested distinct until the task settles', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.task.started', {
      task_id: 't-cancel',
      attempt: 1,
    }))
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.task.cancel_requested', {
      task_id: 't-cancel',
      status: 'cancel_requested',
    }))

    expect(record.taskStates['t-cancel'].status).toBe('cancel_requested')
    expect(record.planStale).toBe(true)

    record = applyAgentRunEvent(record, event(3, 'inqtrix.agent.task.finished', {
      task_id: 't-cancel',
      status: 'cancelled',
    }))
    expect(record.taskStates['t-cancel']).toMatchObject({
      status: 'cancelled',
      outcome: 'cancelled',
    })
  })

  it('keeps insufficient evidence distinct from failure and completion', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.task.finished', {
      task_id: 't3',
      status: 'insufficient_evidence',
      error: 'Only one source met the evidence contract.',
    }))
    expect(record.taskStates.t3).toMatchObject({
      status: 'insufficient_evidence',
      outcome: 'insufficient_evidence',
    })
  })

  it('keeps identity-only child progress in preparing state and consumes its projection', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.child.progress', {
      task_id: 't1',
      child_run_id: 'run_child_1',
      snapshot: {},
      run_status: 'running',
      current_node: 'search',
      message: 'Searching primary sources',
      metrics: { sources: 4, queries: 2 },
      attempt: 2,
      updated_at: 1_700_000_101,
      error: { type: 'provider_error', message: 'Child provider unavailable' },
    }))
    expect(record.children.run_child_1.snapshot).toBeUndefined()
    expect(record.children.run_child_1).toMatchObject({
      currentNode: 'search',
      message: 'Searching primary sources',
      metrics: { sources: 4, queries: 2 },
      attempt: 2,
      error: 'Child provider unavailable',
      errorCode: 'provider_error',
    })
    expect(record.taskStates.t1).toMatchObject({
      status: 'running',
      attempt: 2,
      childRunId: 'run_child_1',
      error: 'Child provider unavailable',
      errorCode: 'provider_error',
    })
  })

  it('artifact events only flag staleness — cached rows stay untouched (E13)', () => {
    let record = agentRunFromSummary(summary())
    const cached = {
      artifactId: 'art1', kind: 'memo' as const, title: 'Memo',
      status: 'writing' as const, revision: 1, updatedBy: 'agent' as const,
      refsCount: 0, createdAt: 1, updatedAt: 1, contentMarkdown: 'rev-1 body',
    }
    record = {
      ...record,
      artifactOrder: ['art1'],
      artifacts: { art1: cached },
      artifactsStale: false,
    }
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.artifact.updated', {
      artifact_id: 'art1',
      revision: 3,
      updated_by: 'agent',
    }))
    // Adopting the bumped revision over the OLD body would defeat the
    // list-refetch staleness check and enable the E13 silent overwrite.
    expect(record.artifacts.art1).toBe(cached)
    expect(record.artifacts.art1.revision).toBe(1)
    expect(record.artifactsStale).toBe(true)
  })

  it('maps waiting and terminal run events onto the status', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.run.waiting', { status: 'waiting_for_approval' }))
    expect(record.status).toBe('waiting_for_approval')
    record = applyAgentRunEvent(record, event(2, 'inqtrix.run.waiting', { status: 'waiting_for_input' }))
    expect(record.status).toBe('waiting_for_input')
    record = applyAgentRunEvent(record, event(3, 'inqtrix.run.completed', { status: 'completed' }))
    expect(record.status).toBe('completed')
    expect(record.artifactsStale).toBe(true)
    expect(record.finishedAt).toBe('2023-11-14T22:15:03.000Z')
    expect(record.elapsedSeconds).toBe(102)
  })

  it('starts the run stopwatch from the live boundary when the queued summary had no start', () => {
    let record = agentRunFromSummary(summary({ started_at: null }))
    record = applyAgentRunEvent(record, event(1, 'inqtrix.run.started'))
    record = applyAgentRunEvent(record, event(9, 'inqtrix.run.completed'))

    expect(record.startedAt).toBe('2023-11-14T22:15:01.000Z')
    expect(record.elapsedSeconds).toBe(8)
  })

  it('preserves authoritative terminal timing while replaying persisted events', () => {
    let record = agentRunFromSummary(summary({
      elapsed_seconds: 77,
      finished_at: 1_700_000_078,
      status: 'completed',
    }))
    record = applyAgentRunEvent(record, event(9, 'inqtrix.run.completed'))

    expect(record.finishedAt).toBe('2023-11-14T22:14:38.000Z')
    expect(record.elapsedSeconds).toBe(77)
  })

  it('captures the live activity line', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.activity', { kind: 'searching', probe: 'Marktlage 2026' }))
    expect(record.activity).toMatchObject({ kind: 'searching', detail: 'Marktlage 2026' })
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.activity', {
      detail: 'Current evidence wins over memory.',
      kind: 'memory_conflict',
    }))
    expect(record.activity).toMatchObject({
      detail: 'Current evidence wins over memory.',
      kind: 'memory_conflict',
    })
  })

  it('aggregates repeated semantic operations and projects local task progress', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.activity', {
      kind: 'searching',
      probe: 'knowledge.search',
      task_id: 'k1',
      current: 1,
      total: 4,
    }))
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.activity', {
      kind: 'searching',
      operation: 'knowledge_search',
      task_id: 'k1',
      current: 2,
      total: 4,
    }))
    expect(record.stepLog.filter((entry) => entry.kind === 'activity')).toHaveLength(1)
    expect(record.stepLog.find((entry) => entry.kind === 'activity')).toMatchObject({
      activityCount: 2,
      activityOperation: 'knowledge_search',
      current: 2,
      total: 4,
    })
    expect(record.taskStates.k1.activity).toMatchObject({
      operation: 'knowledge_search',
      current: 2,
      total: 4,
    })
  })

  it('gives every query of a multi-query task its own protocol row', () => {
    // Regression (Verlauf = Portal): query 2..N of one task used to
    // coalesce into row 1's counter and vanish entirely once the task
    // settled — "search 6 of 6" was never visible as a step.
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.activity', {
      kind: 'searching',
      operation: 'knowledge_search',
      task_id: 'k1',
      status: 'started',
      query: 'EU AI Act Fristen',
      current: 1,
      total: 2,
    }))
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.activity', {
      kind: 'searching',
      operation: 'knowledge_search',
      task_id: 'k1',
      status: 'completed',
      query: 'EU AI Act Fristen',
      current: 1,
      total: 2,
    }))
    record = applyAgentRunEvent(record, event(3, 'inqtrix.agent.activity', {
      kind: 'searching',
      operation: 'knowledge_search',
      task_id: 'k1',
      status: 'started',
      query: 'EU AI Act Sanktionen',
      current: 2,
      total: 2,
    }))
    const rows = record.stepLog.filter((entry) => entry.kind === 'activity')
    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({
      detail: 'EU AI Act Fristen',
      status: 'completed',
    })
    expect(rows[1]).toMatchObject({
      detail: 'EU AI Act Sanktionen',
      status: 'started',
    })
    // The settled row KEEPS its place — a later task_finished event
    // must not remove protocol lines (append-only contract).
    record = applyAgentRunEvent(record, event(4, 'inqtrix.agent.task.finished', {
      task_id: 'k1',
      status: 'completed',
    }))
    expect(
      record.stepLog.filter((entry) => entry.kind === 'activity'),
    ).toHaveLength(2)
  })

  it('keeps provider retry notices on their invocation row and settles at task end', () => {
    // Regression: retry frames carry a changed detail text ("… Versuch
    // 2/3 …"); keyed by detail they opened ghost rows that never
    // settled. The bare `query` is the stable invocation identity, and
    // the task's terminal event closes any row still open.
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.activity', {
      kind: 'searching',
      operation: 'knowledge_search',
      task_id: 'k1',
      status: 'started',
      query: 'EU AI Act Fristen',
    }))
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.activity', {
      kind: 'searching',
      operation: 'knowledge_search',
      task_id: 'k1',
      status: 'started',
      query: 'EU AI Act Fristen',
      detail: 'EU AI Act Fristen · Provider-Versuch 2/3 fehlgeschlagen',
      attempt: 2,
    }))
    let rows = record.stepLog.filter((entry) => entry.kind === 'activity')
    expect(rows).toHaveLength(1)
    expect(rows[0].detail).toContain('Provider-Versuch 2/3')
    // Hard-terminal task: the still-open row must not stay "running".
    record = applyAgentRunEvent(record, event(3, 'inqtrix.agent.task.finished', {
      task_id: 'k1',
      status: 'completed',
    }))
    rows = record.stepLog.filter((entry) => entry.kind === 'activity')
    expect(rows).toHaveLength(1)
    expect(rows[0].status).toBe('completed')
  })

  it('keeps six parallel tasks as six distinct settled rows', () => {
    let record = agentRunFromSummary(summary())
    for (let index = 1; index <= 6; index += 1) {
      record = applyAgentRunEvent(record, event(index, 'inqtrix.agent.activity', {
        kind: 'searching',
        operation: 'knowledge_search',
        task_id: `t${index}`,
        status: 'started',
        query: `Frage ${index}`,
      }))
    }
    for (let index = 1; index <= 6; index += 1) {
      record = applyAgentRunEvent(
        record,
        event(6 + index, 'inqtrix.agent.task.finished', {
          task_id: `t${index}`,
          status: 'completed',
        }),
      )
    }
    const rows = record.stepLog.filter((entry) => entry.kind === 'activity')
    expect(rows).toHaveLength(6)
    expect(rows.every((row) => row.status === 'completed')).toBe(true)
  })

  it('does not double-count started and completed frames for one invocation', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.activity', {
      activity_id: 'op-1',
      operation: 'web_instant',
      task_id: 'w1',
      status: 'started',
      query: 'EU AI Act latest changes',
    }))
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.activity', {
      activity_id: 'op-1',
      operation: 'web_instant',
      task_id: 'w1',
      status: 'completed',
      purpose: 'Check the current regulatory state',
      metrics: { result_count: 6 },
      attempt: 1,
    }))
    const activity = record.stepLog.find((entry) => entry.kind === 'activity')
    expect(activity?.activityCount).toBe(1)
    expect(record.taskStates.w1.activityHistory).toHaveLength(1)
    expect(record.taskStates.w1.activityHistory?.[0].status).toBe('completed')
    expect(record.taskStates.w1.activityHistory?.[0].detail).toBe(
      'EU AI Act latest changes',
    )
    expect(record.taskStates.w1.activityHistory?.[0]).toMatchObject({
      attempt: 1,
      metrics: { result_count: 6 },
      purpose: 'Check the current regulatory state',
    })
  })

  it('uses nested activity errors instead of mislabelling the query as failure', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.activity', {
      activity_id: 'op-2',
      operation: 'web_instant',
      task_id: 'w2',
      status: 'failed',
      query: 'AI market size',
      error: { code: 'search_timeout', message: 'Search provider timed out' },
    }))
    expect(record.taskStates.w2.error).toBe('Search provider timed out')
    expect(record.taskStates.w2.errorCode).toBe('search_timeout')
    expect(record.taskStates.w2.error).not.toBe('AI market size')
  })

  it('retains terminal task metrics for local work units', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.task.finished', {
      task_id: 'k1',
      status: 'completed',
      metrics: { sources: 7, queries: 3, claims: 4 },
      result_summary: 'Seven relevant sources support the finding.',
    }))
    expect(record.taskStates.k1.metrics).toEqual({
      sources: 7,
      queries: 3,
      claims: 4,
    })
    expect(record.taskStates.k1.resultSummary).toBe(
      'Seven relevant sources support the finding.',
    )
  })

  it('preserves the earlier activity finish when a parallel wave folds later', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.task.started', {
      task_id: 'w1',
    }))
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.activity', {
      activity_id: 'web-1',
      operation: 'web_instant',
      status: 'started',
      task_id: 'w1',
    }))
    record = applyAgentRunEvent(record, event(19, 'inqtrix.agent.activity', {
      activity_id: 'web-1',
      operation: 'web_instant',
      status: 'completed',
      task_id: 'w1',
    }))
    record = applyAgentRunEvent(record, event(47, 'inqtrix.agent.task.finished', {
      status: 'completed',
      task_id: 'w1',
    }))
    expect(record.taskStates.w1).toMatchObject({
      finishedAt: 1_700_000_119,
      startedAt: 1_700_000_101,
    })
  })

  it('marks a visible fallback on its task state', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.activity', {
      operation: 'task.legacy_budget_ignored',
      detail: 'Legacy task budget ignored',
      task_id: 'w1',
      status: 'completed',
      fallback: true,
    }))
    expect(record.taskStates.w1.fallback).toBe(true)
    expect(record.taskStates.w1.activity?.detail).toBe('Legacy task budget ignored')
  })

  it('returns a failed child task to running on a queued retry projection', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.child.progress', {
      task_id: 't1',
      child_run_id: 'child-1',
      run_status: 'failed',
      attempt: 1,
      error: { code: 'timeout', message: 'First attempt timed out' },
    }))
    record = applyAgentRunEvent(record, event(2, 'inqtrix.agent.child.progress', {
      task_id: 't1',
      child_run_id: 'child-1',
      run_status: 'queued',
      attempt: 2,
    }))
    expect(record.taskStates.t1).toMatchObject({
      status: 'running',
      attempt: 2,
      error: 'First attempt timed out',
      errorCode: 'timeout',
    })
    expect(record.taskStates.t1.finishedAt).toBeUndefined()
  })

  it('preserves an unknown stable operation code for technical detail', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(record, event(1, 'inqtrix.agent.activity', {
      operation: 'vendor.custom.lookup',
      task_id: 'x1',
      status: 'started',
    }))
    expect(record.taskStates.x1.activity).toMatchObject({
      operation: undefined,
      operationCode: 'vendor.custom.lookup',
    })
  })
})

describe('stepLog (transcript lines)', () => {
  it('appends ordered entries for phases, activity, plan and tasks', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(
      record,
      event(1, 'inqtrix.agent.phase.changed', { phase: 'discovery' }),
    )
    record = applyAgentRunEvent(
      record,
      event(2, 'inqtrix.agent.activity', {
        kind: 'searching',
        probe: 'EU AI Act Anforderungen',
      }),
    )
    record = applyAgentRunEvent(
      record,
      event(3, 'inqtrix.agent.plan.proposed', {
        plan_id: 'plan_1',
        version: 1,
      }),
    )
    record = applyAgentRunEvent(
      record,
      event(4, 'inqtrix.agent.task.started', { task_id: 't1', attempt: 1 }),
    )
    record = applyAgentRunEvent(
      record,
      event(5, 'inqtrix.agent.task.failed', {
        task_id: 't1',
        status: 'failed',
        error: 'Sammlung nicht sichtbar oder unbekannt: kc_x',
      }),
    )
    expect(record.stepLog.map((entry) => entry.kind)).toEqual([
      'phase',
      'activity',
      'plan',
      'task_started',
      'task_failed',
    ])
    expect(record.stepLog[1].detail).toBe('EU AI Act Anforderungen')
    expect(record.stepLog[2].version).toBe(1)
    expect(record.stepLog[4].error).toContain('kc_x')
    // Entries carry the event sequence for stable keys/ordering.
    expect(record.stepLog.map((entry) => entry.seq)).toEqual([1, 2, 3, 4, 5])
  })

  it('appends decision markers that join with the control rows', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(
      record,
      event(1, 'inqtrix.agent.clarification.answered', {
        clarification_id: 'cl_1',
      }),
    )
    record = applyAgentRunEvent(
      record,
      event(2, 'inqtrix.agent.approval.decided', {
        approval_id: 'apr_1',
        status: 'approved',
      }),
    )
    expect(record.stepLog).toEqual([
      expect.objectContaining({
        clarificationId: 'cl_1',
        kind: 'clarification_answered',
      }),
      expect.objectContaining({
        approvalId: 'apr_1',
        detail: 'approved',
        kind: 'approval_decided',
      }),
    ])
  })

  it('captures narration prose from the narration event', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(
      record,
      event(1, 'inqtrix.agent.narration', {
        kind: 'discovery',
        narration_id: 'n-discovery',
        phase: 'discovery',
        text: 'Ich habe eine Sammlung mit 12 Dokumenten gefunden.',
      }),
    )
    expect(record.stepLog[0]).toEqual(
      expect.objectContaining({
        kind: 'narration',
        narrationId: 'n-discovery',
        phase: 'discovery',
        text: 'Ich habe eine Sammlung mit 12 Dokumenten gefunden.',
      }),
    )
  })

  it('upserts a re-emitted narration id in place instead of duplicating', () => {
    // The critic replan loop re-runs synthesis and re-emits the stable
    // narration_id 'n-synthesis' with a FRESH higher sequence — the
    // sequence guard does not suppress it, so without id-dedup the line
    // would multiply in the transcript.
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(
      record,
      event(1, 'inqtrix.agent.narration', {
        narration_id: 'n-synthesis',
        text: 'Ich schreibe das Memo mit 3 Abschnitten.',
      }),
    )
    record = applyAgentRunEvent(
      record,
      event(2, 'inqtrix.agent.activity', { kind: 'searching', probe: 'x' }),
    )
    record = applyAgentRunEvent(
      record,
      event(9, 'inqtrix.agent.narration', {
        narration_id: 'n-synthesis',
        text: 'Ich schreibe das Memo mit 4 Abschnitten.',
      }),
    )
    const narrations = record.stepLog.filter(
      (entry) => entry.kind === 'narration',
    )
    expect(narrations).toHaveLength(1)
    // Text updated in place; seq stays at the original for key stability.
    expect(narrations[0].text).toBe('Ich schreibe das Memo mit 4 Abschnitten.')
    expect(narrations[0].seq).toBe(1)
    // The intervening activity line kept its position (no reordering).
    expect(record.stepLog.map((entry) => entry.kind)).toEqual([
      'narration',
      'activity',
    ])
  })

  it('replays are no-ops for the log too', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(
      record,
      event(3, 'inqtrix.agent.phase.changed', { phase: 'planning' }),
    )
    const replay = applyAgentRunEvent(
      record,
      event(3, 'inqtrix.agent.phase.changed', { phase: 'planning' }),
    )
    expect(replay).toBe(record)
    expect(record.stepLog).toHaveLength(1)
  })
})

describe('model_resolution events (R5-light)', () => {
  it('keeps only ANSWER-node resolutions on the record', () => {
    let record = agentRunFromSummary(summary())
    record = applyAgentRunEvent(
      record,
      event(1, 'inqtrix.node.model_resolution', {
        node: 'agent_intake',
        model: 'fast-model',
        effort: '',
        tier: 'fast',
        model_source: 'tier:fast',
      }),
    )
    // Assembly-line provenance never drives the chip.
    expect(record.modelResolution).toBeUndefined()
    record = applyAgentRunEvent(
      record,
      event(2, 'inqtrix.node.model_resolution', {
        node: 'agent_synthesis',
        model: 'opus-x',
        effort: 'xhigh',
        tier: 'high',
        model_source: 'explicit_request',
      }),
    )
    expect(record.modelResolution).toEqual({
      model: 'opus-x',
      effort: 'xhigh',
      tier: 'high',
      modelSource: 'explicit_request',
    })
    // A later answer-node resolution wins (the kernel/answer call is
    // the one the chip should describe).
    record = applyAgentRunEvent(
      record,
      event(3, 'inqtrix.node.model_resolution', {
        node: 'agent_answer',
        model: 'opus-x',
        effort: 'high',
        tier: 'high',
        model_source: 'explicit_request',
      }),
    )
    expect(record.modelResolution?.effort).toBe('high')
  })
})
