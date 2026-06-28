import { describe, expect, it } from 'vitest'
import type {
  ServerKnowledgeSession,
  ServerKnowledgeSessionGroup,
} from '@/api/inqtrixClient'
import type {
  KnowledgeSessionGroupRecord,
  KnowledgeSessionRecord,
  KnowledgeThreadItemRecord,
} from '@/features/project/types'
import {
  fingerprintKnowledgeSession,
  groupRecordFromServer,
  itemsFromServerSession,
  serverKnowledgeSessionGroupPayload,
  serverKnowledgeSessionPayload,
  sessionRecordFromServer,
} from './knowledgeSessionSync'

function serverSession(overrides: Partial<ServerKnowledgeSession> = {}): ServerKnowledgeSession {
  return {
    created_at: 1767225600,
    group_id: null,
    id: 'ks-1',
    items_json: null,
    title: 'Session',
    updated_at: 1767312000,
    ...overrides,
  }
}

function serverGroup(overrides: Partial<ServerKnowledgeSessionGroup> = {}): ServerKnowledgeSessionGroup {
  return {
    created_at: 1767225600,
    id: 'kg-1',
    title: 'Folder',
    updated_at: 1767312000,
    ...overrides,
  }
}

function item(overrides: Partial<KnowledgeThreadItemRecord> = {}): KnowledgeThreadItemRecord {
  return {
    collectionTitles: ['EU Recht'],
    createdAt: '2026-01-01T00:00:00.000Z',
    id: 'ki-1',
    progress: { steps: [] },
    question: 'Welche Pflichten gelten?',
    requestedProfile: null,
    runId: 'run-1',
    sessionId: 'ks-old',
    status: 'completed',
    ...overrides,
  }
}

describe('knowledge session sync conversion', () => {
  it('converts server metadata and membership into local session records', () => {
    expect(sessionRecordFromServer(serverSession({ group_id: 'kg-1' }))).toEqual({
      groupId: 'kg-1',
      record: {
        createdAt: '2026-01-01T00:00:00.000Z',
        id: 'ks-1',
        title: 'Session',
        updatedAt: '2026-01-02T00:00:00.000Z',
      },
    })
  })

  it('converts server group metadata into local group records', () => {
    expect(groupRecordFromServer(serverGroup())).toEqual({
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'kg-1',
      title: 'Folder',
      updatedAt: '2026-01-02T00:00:00.000Z',
    })
  })

  it('hydrates item payloads under the owning server session id', () => {
    const parsed = itemsFromServerSession(serverSession({
      id: 'ks-server',
      items_json: JSON.stringify([{
        ...item({
          collectionIds: ['collection-1'],
          completedAt: '2026-01-01T00:05:00.000Z',
          topK: null,
          finalK: 16,
        }),
        sessionId: undefined,
      }, { id: 4 }]),
    }))
    expect(parsed).toHaveLength(1)
    expect(parsed[0]).toMatchObject({
      collectionIds: ['collection-1'],
      completedAt: '2026-01-01T00:05:00.000Z',
      id: 'ki-1',
      sessionId: 'ks-server',
      topK: null,
      finalK: 16,
    })
  })

  it('hydrates cancelled item payloads', () => {
    const parsed = itemsFromServerSession(serverSession({
      id: 'ks-server',
      items_json: JSON.stringify([item({
        completedAt: undefined,
        status: 'cancelled',
      })]),
    }))

    expect(parsed).toHaveLength(1)
    expect(parsed[0]).toMatchObject({
      id: 'ki-1',
      sessionId: 'ks-server',
      status: 'cancelled',
    })
  })

  it('serializes payloads and fingerprints the data that affects autosave', () => {
    const session: KnowledgeSessionRecord = {
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'ks-1',
      title: 'Session',
      updatedAt: '2026-01-02T00:00:00.000Z',
    }
    const items = [item({
      collectionIds: ['collection-1'],
      completedAt: '2026-01-01T00:05:00.000Z',
      sessionId: 'ks-1',
      topK: 12,
      finalK: 16,
    })]
    const payload = serverKnowledgeSessionPayload(session, items, 'kg-1')

    expect(payload).toMatchObject({
      created_at: 1767225600,
      group_id: 'kg-1',
      title: 'Session',
      updated_at: 1767312000,
    })
    expect(JSON.parse(payload.items_json)).toEqual(items)
    expect(fingerprintKnowledgeSession(session, items, 'kg-1')).toContain('"groupId":"kg-1"')
    expect(fingerprintKnowledgeSession(session, items, null)).not.toEqual(
      fingerprintKnowledgeSession(session, items, 'kg-1'),
    )
  })

  it('serializes group payloads', () => {
    const group: KnowledgeSessionGroupRecord = {
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'kg-1',
      title: 'Folder',
      updatedAt: '2026-01-02T00:00:00.000Z',
    }

    expect(serverKnowledgeSessionGroupPayload(group)).toEqual({
      created_at: 1767225600,
      title: 'Folder',
      updated_at: 1767312000,
    })
  })
})
