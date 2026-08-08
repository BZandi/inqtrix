import { describe, expect, it } from 'vitest'
import type { AgentSessionRecord } from './model'
import {
  agentSessionMetadataFromJson,
  agentSessionFingerprint,
  serverAgentSessionFingerprint,
  serverAgentSessionPayload,
} from './agentSessionSync'

describe('agent session source policy serialization', () => {
  it('round-trips source availability through items_json', () => {
    const payload = serverAgentSessionPayload({
      id: 'session-1',
      title: 'Session',
      groupId: null,
      createdAt: '2026-07-10T10:00:00.000Z',
      updatedAt: '2026-07-10T10:01:00.000Z',
      runIds: [],
      sourcePolicy: { web: 'disabled', knowledge: 'available' },
    })
    expect(agentSessionMetadataFromJson(payload.items_json)).toEqual({
      // modelSelection rides the same blob; a session without a pick reports
      // null so the account preference can seed the composer.
      modelSelection: null,
      sourcePolicy: { web: 'disabled', knowledge: 'available' },
    })
  })

  it('loads older and malformed rows with safe available defaults', () => {
    expect(agentSessionMetadataFromJson('{}').sourcePolicy).toEqual({
      web: 'available',
      knowledge: 'available',
    })
    expect(agentSessionMetadataFromJson('{broken').sourcePolicy).toEqual({
      web: 'available',
      knowledge: 'available',
    })
  })

  it('fingerprints a server row before the reducer has hydrated it', () => {
    const wire = {
      created_at: 1_752_140_800,
      group_id: 'group-1',
      id: 'session-1',
      items_json: JSON.stringify({
        source_policy: { web: 'disabled', knowledge: 'available' },
      }),
      title: 'Session',
      updated_at: 1_752_140_860,
    }

    expect(serverAgentSessionFingerprint(wire)).toBe(
      agentSessionFingerprint({
        createdAt: new Date(wire.created_at * 1000).toISOString(),
        groupId: wire.group_id,
        id: wire.id,
        runIds: [],
        sourcePolicy: { web: 'disabled', knowledge: 'available' },
        title: wire.title,
        updatedAt: new Date(wire.updated_at * 1000).toISOString(),
      }),
    )
  })
})

describe('agent session model stickiness (wire)', () => {
  const base = {
    id: 'session-1',
    title: 'Session',
    groupId: null,
    createdAt: '2026-08-07T10:00:00.000Z',
    updatedAt: '2026-08-07T10:00:00.000Z',
    runIds: [],
    sourcePolicy: { web: 'available', knowledge: 'available' },
  } satisfies AgentSessionRecord

  it('round-trips the picked model through items_json', () => {
    const payload = serverAgentSessionPayload({
      ...base,
      modelSelection: { model: 'claude-opus-4-8', tier: null, effort: 'high' },
    })
    const parsed = agentSessionMetadataFromJson(payload.items_json)
    expect(parsed.modelSelection).toEqual({
      model: 'claude-opus-4-8',
      tier: null,
      effort: 'high',
    })
    expect(parsed.sourcePolicy.web).toBe('available')
  })

  it('reports no selection for a session that never had one', () => {
    const payload = serverAgentSessionPayload(base)
    expect(agentSessionMetadataFromJson(payload.items_json).modelSelection).toBeNull()
  })

  it('stays loadable for rows written before this feature', () => {
    const legacy = JSON.stringify({ source_policy: { web: 'disabled', knowledge: 'available' } })
    const parsed = agentSessionMetadataFromJson(legacy)
    expect(parsed.modelSelection).toBeNull()
    expect(parsed.sourcePolicy.web).toBe('disabled')
  })

  it('drops an unknown tier instead of pinning it', () => {
    const future = JSON.stringify({ model_selection: { model: null, tier: 'turbo', effort: null } })
    expect(agentSessionMetadataFromJson(future).modelSelection).toBeNull()
  })

  it('makes a changed model visible to the autosave diff', () => {
    // The fingerprint is the ONLY thing that decides whether a change is
    // pushed. A field missing from it syncs silently never.
    const a = agentSessionFingerprint(base)
    const b = agentSessionFingerprint({
      ...base,
      modelSelection: { model: null, tier: 'fast', effort: null },
    })
    const c = agentSessionFingerprint({
      ...base,
      modelSelection: { model: null, tier: 'high', effort: null },
    })
    expect(b).not.toBe(a)
    expect(c).not.toBe(b)
  })

  it('sees the same change from the server side of the diff', () => {
    const wire = (itemsJson: string) => ({
      id: 'session-1',
      title: 'Session',
      group_id: null,
      created_at: 1_700_000_000,
      updated_at: 1_700_000_000,
      items_json: itemsJson,
    })
    const withTier = serverAgentSessionFingerprint(wire(JSON.stringify({
      source_policy: { web: 'available', knowledge: 'available' },
      model_selection: { model: null, tier: 'fast', effort: null },
    })))
    const without = serverAgentSessionFingerprint(wire(JSON.stringify({
      source_policy: { web: 'available', knowledge: 'available' },
    })))
    expect(withTier).not.toBe(without)
  })
})
