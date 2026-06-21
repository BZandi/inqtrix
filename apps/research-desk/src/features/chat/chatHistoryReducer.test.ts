import { describe, expect, it } from 'vitest'

import { createEmptyProjectState } from '@/features/project/seedProject'
import type {
  ChatMessageRecord,
  ChatThreadRecord,
  ProjectState,
} from '@/features/project/types'
import { researchDeskReducer } from '@/features/researchDesk/state'

function thread(
  id: string,
  overrides: Partial<ChatThreadRecord> = {},
): ChatThreadRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    messages: [],
    preview: '',
    source: 'api',
    title: id,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function message(id: string, createdAt: string): ChatMessageRecord {
  return { contentMarkdown: id, createdAt, id, role: 'user' }
}

function withThread(local: ChatThreadRecord): ProjectState {
  const base = createEmptyProjectState()
  return {
    ...base,
    chatThreadOrder: [local.id],
    chatThreads: { [local.id]: local },
    dirty: false,
  }
}

describe('server chat hydration (M6a)', () => {
  it('adds server threads with empty messages + membership, WITHOUT dirtying', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      memberships: { ct_1: 'ctg_1' },
      threads: [thread('ct_1', { title: 'From server' })],
      type: 'upsertServerChatThreads',
    })
    expect(next.chatThreads.ct_1.title).toBe('From server')
    expect(next.chatThreads.ct_1.messages).toEqual([]) // loaded lazily on open
    expect(next.chatThreadGroupMemberships.ct_1).toBe('ctg_1')
    expect(next.chatThreadOrder).toContain('ct_1')
    expect(next.dirty).toBe(false) // server-pushed state never marks dirty
  })

  it('preserves the server keyset order within a hydrated page (no updatedAt re-sort)', () => {
    // Server pages by created_at desc; the batch arrives newest-first. updatedAt
    // is deliberately the REVERSE order -- if the reducer re-sorted by updatedAt
    // the list would flip, which is the cross-page inconsistency we forbid.
    const next = researchDeskReducer(createEmptyProjectState(), {
      memberships: { ct_new: null, ct_old: null },
      threads: [
        thread('ct_new', { createdAt: '2026-03-01T00:00:00.000Z', updatedAt: '2026-01-01T00:00:00.000Z' }),
        thread('ct_old', { createdAt: '2026-02-01T00:00:00.000Z', updatedAt: '2026-09-01T00:00:00.000Z' }),
      ],
      type: 'upsertServerChatThreads',
    })
    expect(next.chatThreadOrder).toEqual(['ct_new', 'ct_old'])
  })

  it('appends an older load-more page to the END, keeping the merged list newest-first', () => {
    const page1 = researchDeskReducer(createEmptyProjectState(), {
      memberships: { ct_a: null, ct_b: null },
      threads: [
        thread('ct_a', { createdAt: '2026-03-01T00:00:00.000Z' }),
        thread('ct_b', { createdAt: '2026-02-01T00:00:00.000Z' }),
      ],
      type: 'upsertServerChatThreads',
    })
    const page2 = researchDeskReducer(page1, {
      append: true,
      memberships: { ct_c: null },
      threads: [thread('ct_c', { createdAt: '2026-01-01T00:00:00.000Z' })],
      type: 'upsertServerChatThreads',
    })
    expect(page2.chatThreadOrder).toEqual(['ct_a', 'ct_b', 'ct_c'])
  })

  it('does NOT clobber a newer local thread with an older server version', () => {
    const local = thread('ct_1', {
      messages: [message('cm_local', '2026-02-01T00:00:00.000Z')],
      title: 'Local edit',
      updatedAt: '2026-02-01T00:00:00.000Z',
    })
    const next = researchDeskReducer(withThread(local), {
      memberships: { ct_1: null },
      threads: [thread('ct_1', { title: 'Stale server', updatedAt: '2026-01-01T00:00:00.000Z' })],
      type: 'upsertServerChatThreads',
    })
    expect(next.chatThreads.ct_1.title).toBe('Local edit') // local wins
    expect(next.chatThreads.ct_1.messages).toHaveLength(1) // messages kept
  })

  it('takes a strictly-newer server version for an existing thread', () => {
    const local = thread('ct_1', { title: 'Old local', updatedAt: '2026-01-01T00:00:00.000Z' })
    const next = researchDeskReducer(withThread(local), {
      memberships: { ct_1: null },
      threads: [thread('ct_1', { title: 'Newer server', updatedAt: '2026-03-01T00:00:00.000Z' })],
      type: 'upsertServerChatThreads',
    })
    expect(next.chatThreads.ct_1.title).toBe('Newer server')
  })

  it('fills a thread\'s messages on load-on-open WITHOUT dirtying', () => {
    const local = thread('ct_1')
    const next = researchDeskReducer(withThread(local), {
      messages: [
        message('cm_2', '2026-01-01T00:02:00.000Z'),
        message('cm_1', '2026-01-01T00:01:00.000Z'),
      ],
      threadId: 'ct_1',
      type: 'upsertServerChatMessages',
    })
    // Merged + ordered chronologically (oldest first).
    expect(next.chatThreads.ct_1.messages.map((m) => m.id)).toEqual(['cm_1', 'cm_2'])
    expect(next.dirty).toBe(false)
  })

  it('merges server groups WITHOUT dirtying', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      groups: [{ createdAt: '2026-01-01T00:00:00.000Z', id: 'ctg_1', title: 'G', updatedAt: '2026-01-01T00:00:00.000Z' }],
      type: 'upsertServerChatThreadGroups',
    })
    expect(next.chatThreadGroups.ctg_1.title).toBe('G')
    expect(next.chatThreadGroupOrder).toContain('ctg_1')
    expect(next.dirty).toBe(false)
  })

  it('setServerSyncEnabled flips the opt-in AND dirties (so it persists)', () => {
    const base = createEmptyProjectState()
    expect(base.serverSyncEnabled).toBe(false)
    const next = researchDeskReducer(base, { enabled: true, type: 'setServerSyncEnabled' })
    expect(next.serverSyncEnabled).toBe(true)
    expect(next.dirty).toBe(true)
  })
})
