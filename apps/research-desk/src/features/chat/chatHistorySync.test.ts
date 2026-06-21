import { describe, expect, it } from 'vitest'

import type {
  ServerChatMessage,
  ServerChatThread,
  ServerChatThreadGroup,
} from '@/api/inqtrixClient'
import type { ChatMessageRecord } from '@/features/project/types'
import {
  fingerprintThread,
  groupRecordFromServer,
  messageRecordFromServer,
  serverGroupPayload,
  serverMessagePayload,
  serverThreadPayload,
  threadNeedsSync,
  threadRecordFromServer,
} from './chatHistorySync'

describe('chatHistorySync converters', () => {
  it('round-trips a thread record and its group membership', () => {
    const server: ServerChatThread = {
      created_at: 1_700_000_000,
      group_id: 'ctg_1',
      id: 'ct_1',
      preview: 'last line',
      source: 'imported',
      title: 'A thread',
      updated_at: 1_700_000_500,
    }
    const { groupId, record } = threadRecordFromServer(server)
    expect(groupId).toBe('ctg_1')
    expect(record.messages).toEqual([]) // messages load lazily on open
    expect(record.source).toBe('imported')
    expect(record.createdAt).toBe(new Date(1_700_000_000 * 1000).toISOString())

    const payload = serverThreadPayload(record, groupId)
    expect(payload.created_at).toBe(server.created_at)
    expect(payload.updated_at).toBe(server.updated_at)
    expect(payload.group_id).toBe('ctg_1')
    expect(payload.source).toBe('imported')
  })

  it('normalizes an unknown thread source instead of trusting it blindly', () => {
    const { record } = threadRecordFromServer({
      created_at: 1,
      group_id: null,
      id: 'ct_x',
      preview: '',
      source: 'totally-bogus',
      title: 'T',
      updated_at: 1,
    })
    expect(record.source).toBe('api')
  })

  it('round-trips a message with attachments/chainTrace/modelResolution verbatim', () => {
    const record: ChatMessageRecord = {
      attachments: [
        {
          attachedAt: '2026-01-01T00:00:00.000Z',
          contentMarkdown: '# Report',
          kind: 'research-report',
          runId: 'run_1',
          title: 'My report',
        },
      ],
      chainTrace: [{ label: 'step', output: 'ok', status: 'ok' }],
      contentMarkdown: 'hello',
      createdAt: '2026-06-01T12:00:00.000Z',
      id: 'cm_1',
      modelResolution: {
        effort: 'high',
        effortSource: 'override',
        model: 'claude',
        modelSource: 'default',
        requestedTier: 'deep',
        tier: 'deep',
      },
      role: 'assistant',
    }
    const payload = serverMessagePayload(record)
    expect(payload.metadata.attachments).toEqual(record.attachments)
    expect(payload.metadata.chainTrace).toEqual(record.chainTrace)
    expect(payload.metadata.modelResolution).toEqual(record.modelResolution)

    const wire: ServerChatMessage = {
      content_markdown: payload.content_markdown,
      created_at: payload.created_at,
      id: payload.id,
      metadata: payload.metadata,
      role: payload.role,
      thread_id: 'ct_1',
    }
    expect(messageRecordFromServer(wire)).toEqual(record)
  })

  it('omits absent optional message fields rather than emitting empty keys', () => {
    const wire: ServerChatMessage = {
      content_markdown: 'plain',
      created_at: 10,
      id: 'cm_2',
      metadata: {},
      role: 'user',
      thread_id: 'ct_1',
    }
    const record = messageRecordFromServer(wire)
    expect('attachments' in record).toBe(false)
    expect('chainTrace' in record).toBe(false)
    expect('modelResolution' in record).toBe(false)
    // And a bare message packs an empty metadata object (no junk keys).
    expect(serverMessagePayload(record).metadata).toEqual({})
  })

  it('round-trips a group record', () => {
    const server: ServerChatThreadGroup = {
      created_at: 100,
      id: 'ctg_1',
      title: 'Group',
      updated_at: 200,
    }
    const record = groupRecordFromServer(server)
    const payload = serverGroupPayload(record)
    expect(payload.created_at).toBe(100)
    expect(payload.updated_at).toBe(200)
    expect(payload.title).toBe('Group')
  })
})

describe('thread fingerprint diff', () => {
  const base = {
    createdAt: '2026-01-01T00:00:00.000Z',
    id: 'ct_1',
    messages: [],
    preview: '',
    source: 'api' as const,
    title: 'T',
    updatedAt: '2026-01-01T00:00:00.000Z',
  }

  it('reports a thread as needing sync when never synced', () => {
    expect(threadNeedsSync(undefined, fingerprintThread(base, null))).toBe(true)
  })

  it('detects an updatedAt change (content/title/message edit)', () => {
    const previous = fingerprintThread(base, null)
    const next = fingerprintThread(
      { ...base, updatedAt: '2026-01-02T00:00:00.000Z' },
      null,
    )
    expect(threadNeedsSync(previous, next)).toBe(true)
  })

  it('detects a group membership change without an updatedAt bump', () => {
    const previous = fingerprintThread(base, null)
    const next = fingerprintThread(base, 'ctg_1')
    expect(threadNeedsSync(previous, next)).toBe(true)
  })

  it('is stable for an unchanged thread (no spurious re-push)', () => {
    const previous = fingerprintThread(base, 'ctg_1')
    const next = fingerprintThread(base, 'ctg_1')
    expect(threadNeedsSync(previous, next)).toBe(false)
  })

  it('flags a local-newer thread for push when seeded with the SERVER fingerprint (P1 regression)', () => {
    // Hydration seeds synced state with WHAT THE SERVER HOLDS (the server
    // record). If the local copy is newer, the diff must fire so the newer
    // local thread is pushed UP — seeding with the LOCAL value (the old bug)
    // made this return false and silently stranded the local edit.
    const serverSeed = fingerprintThread(
      { ...base, updatedAt: '2026-01-01T00:00:00.000Z' },
      null,
    )
    const localNewer = fingerprintThread(
      { ...base, updatedAt: '2026-05-01T00:00:00.000Z' },
      null,
    )
    expect(threadNeedsSync(serverSeed, localNewer)).toBe(true)
  })
})
