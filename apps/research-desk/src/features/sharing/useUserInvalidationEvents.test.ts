import { describe, expect, it } from 'vitest'
import type { UserEvent } from '@/api/inqtrixClient'
import {
  confirmedUserEventCursor,
  userEventAction,
  userEventReconnectDelay,
  verifyLatestUserIdentity,
} from './useUserInvalidationEvents'

describe('userEventAction', () => {
  it('forces a reload before processing a ready frame for another user', () => {
    const event: UserEvent = {
      data: { cursor: '12', user_id: 'user-b' },
      id: null,
      type: 'ready',
    }
    expect(userEventAction(event, 'user-a')).toBe('reload')
    expect(userEventAction(event, 'user-b')).toBe('refetch')
  })

  it('treats invalidate and reset as content-free refetch signals', () => {
    expect(userEventAction({
      data: { resource_id: 'run-1', resource_type: 'run', scope: 'sharing' },
      id: '13',
      type: 'invalidate',
    }, 'user-a')).toBe('refetch')
    expect(userEventAction({
      data: {},
      id: null,
      type: 'reset',
    }, 'user-a')).toBe('refetch')
  })

  it('consumes comment outbox duplicates without waking every resource store', () => {
    for (const scope of [
      'collaboration_comment_changed',
      'collaboration_comment_mention',
    ]) {
      expect(userEventAction({
        data: {
          resource_id: 'ed_1',
          resource_type: 'editor_document',
          scope,
        },
        id: '13',
        type: 'invalidate',
      }, 'user-a')).toBe('consume')
    }
  })

  it('ignores unknown named events and bounds reconnect backoff', () => {
    expect(userEventAction({
      data: {},
      id: null,
      type: 'unknown',
    }, 'user-a')).toBe('ignore')
    expect(userEventReconnectDelay(-1)).toBe(1_000)
    expect(userEventReconnectDelay(20)).toBe(30_000)
  })
})

describe('verifyLatestUserIdentity', () => {
  const authenticated = (id: string) => ({
    authenticated: true as const,
    user: { display_name: null, email: null, id, role: 'user' },
  })

  it('authorizes a refetch only for the same canonical user', async () => {
    await expect(verifyLatestUserIdentity({
      expectedUserId: 'user-a',
      isCurrent: () => true,
      probe: async () => authenticated('user-a'),
    })).resolves.toEqual({ action: 'refetch', error: null })
    await expect(verifyLatestUserIdentity({
      expectedUserId: 'user-a',
      isCurrent: () => true,
      probe: async () => authenticated('user-b'),
    })).resolves.toEqual({ action: 'reload', error: null })
    await expect(verifyLatestUserIdentity({
      expectedUserId: 'user-a',
      isCurrent: () => true,
      probe: async () => ({ authenticated: false }),
    })).resolves.toEqual({ action: 'reload', error: null })
  })

  it('retains without refetching on an explicit network failure', async () => {
    await expect(verifyLatestUserIdentity({
      expectedUserId: 'user-a',
      isCurrent: () => true,
      probe: async () => {
        throw new TypeError('Failed to fetch')
      },
    })).resolves.toEqual({ action: 'retain', error: 'Failed to fetch' })
  })

  it('discards a delayed stale probe after a newer generation wins', async () => {
    let releaseFirst!: (value: ReturnType<typeof authenticated>) => void
    const firstProbe = new Promise<ReturnType<typeof authenticated>>((resolve) => {
      releaseFirst = resolve
    })
    let generation = 1
    const first = verifyLatestUserIdentity({
      expectedUserId: 'user-a',
      isCurrent: () => generation === 1,
      probe: () => firstProbe,
    })

    generation = 2
    const second = verifyLatestUserIdentity({
      expectedUserId: 'user-a',
      isCurrent: () => generation === 2,
      probe: async () => authenticated('user-a'),
    })
    await expect(second).resolves.toEqual({ action: 'refetch', error: null })

    releaseFirst(authenticated('user-b'))
    await expect(first).resolves.toBeNull()
  })
})

describe('confirmedUserEventCursor', () => {
  const invalidate: UserEvent = {
    data: { scope: 'sharing' },
    id: '14',
    type: 'invalidate',
  }

  it('does not consume an invalidation while identity verification is offline', () => {
    expect(confirmedUserEventCursor('13', invalidate, {
      action: 'retain',
      error: 'Failed to fetch',
    })).toBe('13')
  })

  it('commits a confirmed event and clears a confirmed reset cursor', () => {
    const confirmed = { action: 'refetch' as const, error: null }
    expect(confirmedUserEventCursor('13', invalidate, confirmed)).toBe('14')
    expect(confirmedUserEventCursor('14', {
      data: {},
      id: null,
      type: 'reset',
    }, confirmed)).toBeUndefined()
  })
})
