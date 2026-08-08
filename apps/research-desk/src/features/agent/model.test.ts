import { describe, expect, it } from 'vitest'

import {
  agentArtifactFromWire,
  agentSessionHistoryTimeIso,
  canEditAgentRun,
  isActiveAgentRun,
  isGateAgentRun,
  restoredAgentSessionId,
} from './model'
import type { ResearchRunStatus } from '@/features/researchRuns/types'

describe('agent artifact detail projection', () => {
  it('retains the evidence payload only after the detail row is fetched', () => {
    const artifact = agentArtifactFromWire({
      artifact_id: 'artifact-evidence',
      content_markdown: '',
      created_at: 1,
      kind: 'evidence_bundle',
      payload: {
        schema_version: 1,
        web_search_ledger: {
          kind: 'web_search_ledger',
          schema_version: 1,
          searches: {
            'query-1': { provider_answer: 'Grounded answer', query_id: 'query-1' },
          },
        },
      },
      refs: [],
      refs_count: 0,
      revision: 1,
      revisions: [],
      run_id: 'run-1',
      session_id: null,
      status: 'ready',
      title: 'Evidence',
      updated_at: 2,
      updated_by: 'agent',
    })

    expect(artifact.payload).toEqual({
      schema_version: 1,
      web_search_ledger: {
        kind: 'web_search_ledger',
        schema_version: 1,
        searches: {
          'query-1': { provider_answer: 'Grounded answer', query_id: 'query-1' },
        },
      },
    })
  })
})

/**
 * THE one active/gate vocabulary for every agent surface (composer,
 * pills, rail, pulse track, canvas, transcript). The regression this
 * locks: surfaces hand-rolled their own status lists and drifted —
 * `waiting_for_children` (the NORMAL parked-on-children mid-execution
 * state) was treated as idle by half of them.
 */
describe('agent run status predicates', () => {
  const expectations: Record<
    ResearchRunStatus,
    { active: boolean; gate: boolean }
  > = {
    queued: { active: true, gate: false },
    running: { active: true, gate: false },
    waiting_for_approval: { active: true, gate: true },
    waiting_for_input: { active: true, gate: true },
    waiting_for_children: { active: true, gate: false },
    completed: { active: false, gate: false },
    failed: { active: false, gate: false },
    cancelled: { active: false, gate: false },
    expired: { active: false, gate: false },
  }

  it('classifies every status exactly once', () => {
    for (const [status, expected] of Object.entries(expectations)) {
      expect(
        isActiveAgentRun(status as ResearchRunStatus),
        `${status} active`,
      ).toBe(expected.active)
      expect(
        isGateAgentRun(status as ResearchRunStatus),
        `${status} gate`,
      ).toBe(expected.gate)
    }
  })

  it('a gate is always active (working = active && !gate)', () => {
    for (const [status, expected] of Object.entries(expectations)) {
      if (expected.gate) {
        expect(isActiveAgentRun(status as ResearchRunStatus)).toBe(true)
      }
    }
  })
})

describe('agent run permission gate', () => {
  const runWithAccess = (
    access: import('@/features/researchRuns/types').ResearchRunAccess,
  ) => ({ access }) as import('./model').AgentRunRecord

  it('blocks every mutation for view shares only', () => {
    expect(canEditAgentRun(undefined)).toBe(false)
    expect(canEditAgentRun(runWithAccess({ mode: 'owner' }))).toBe(true)
    expect(canEditAgentRun(runWithAccess({ mode: 'unscoped' }))).toBe(true)
    expect(canEditAgentRun(runWithAccess({
      mode: 'shared',
      permission: 'edit',
    }))).toBe(true)
    expect(canEditAgentRun(runWithAccess({
      mode: 'shared',
      permission: 'view',
    }))).toBe(false)
  })
})

function makeSession(
  id: string,
  updatedAt: string,
  runIds: string[] = [],
) {
  return {
    id,
    title: id,
    groupId: null,
    createdAt: '2026-07-01T00:00:00.000Z',
    updatedAt,
    runIds,
    sourcePolicy: { web: 'available' as const, knowledge: 'available' as const },
  }
}

describe('restoredAgentSessionId', () => {
  const sessions = {
    old: makeSession('old', '2026-07-01T10:00:00.000Z'),
    recent: makeSession('recent', '2026-07-10T10:00:00.000Z'),
  }

  it('prefers the persisted intent when the session still exists', () => {
    expect(
      restoredAgentSessionId('old', ['recent', 'old'], sessions, {}),
    ).toBe('old')
  })

  it('falls back to the most recently updated session for a stale intent', () => {
    expect(
      restoredAgentSessionId('gone', ['old', 'recent'], sessions, {}),
    ).toBe('recent')
  })

  it('falls back by recency with no intent, and to null with no sessions', () => {
    expect(
      restoredAgentSessionId(null, ['old', 'recent'], sessions, {}),
    ).toBe('recent')
    expect(restoredAgentSessionId(null, [], {}, {})).toBeNull()
  })

  it('conversation recency (latest run) beats a newer rename', () => {
    const withRuns = {
      old: makeSession('old', '2026-07-01T10:00:00.000Z', ['r-old']),
      recent: makeSession('recent', '2026-07-10T10:00:00.000Z'),
    }
    const runs = {
      'r-old': { createdAt: '2026-07-11T09:00:00.000Z' },
    } as unknown as Record<string, import('./model').AgentRunRecord>
    expect(
      restoredAgentSessionId(null, ['old', 'recent'], withRuns, runs),
    ).toBe('old')
  })

  it('never restores a session whose durable deletion is still visible', () => {
    const deleting = {
      ...sessions,
      recent: {
        ...sessions.recent,
        deletion: {
          error: null,
          operationId: 'del_1',
          stage: 'queued',
          status: 'deleting' as const,
        },
      },
    }
    expect(restoredAgentSessionId('recent', ['recent', 'old'], deleting, {})).toBe('old')
  })
})

describe('agentSessionHistoryTimeIso', () => {
  it('uses the latest run createdAt over the session updatedAt', () => {
    const session = makeSession('s', '2026-07-01T10:00:00.000Z', ['r1', 'r2'])
    const runs = {
      r1: { createdAt: '2026-07-02T10:00:00.000Z' },
      r2: { createdAt: '2026-07-09T10:00:00.000Z' },
    } as unknown as Record<string, import('./model').AgentRunRecord>
    expect(agentSessionHistoryTimeIso(session, runs)).toBe(
      '2026-07-09T10:00:00.000Z',
    )
  })

  it('skips unresolvable run ids and falls back to updatedAt', () => {
    const session = makeSession('s', '2026-07-01T10:00:00.000Z', ['missing'])
    expect(agentSessionHistoryTimeIso(session, {})).toBe(
      '2026-07-01T10:00:00.000Z',
    )
  })
})
