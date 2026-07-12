import { describe, expect, it } from 'vitest'
import {
  agentSessionMetadataFromJson,
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
})
