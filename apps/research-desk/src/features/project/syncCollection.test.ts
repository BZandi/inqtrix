import { describe, expect, it } from 'vitest'

import { deleteTolerant404, syncCollection } from './syncCollection'

type Item = { id: string; v: string }

function harness(current: Record<string, Item>, synced: Map<string, string>) {
  const pushed: string[] = []
  const deleted: string[] = []
  const run = () =>
    syncCollection<Item, string>({
      current,
      synced,
      fingerprintOf: (item) => item.v,
      changed: (previous, value) => previous !== value,
      pushOne: async (item) => {
        pushed.push(item.id)
      },
      deleteOne: async (id) => {
        deleted.push(id)
      },
    })
  return { deleted, pushed, run }
}

describe('syncCollection', () => {
  it('pushes new entities and advances the synced fingerprint', async () => {
    const synced = new Map<string, string>()
    const { pushed, run } = harness({ a: { id: 'a', v: '1' } }, synced)
    await run()
    expect(pushed).toEqual(['a'])
    expect(synced.get('a')).toBe('1')
  })

  it('pushes only changed entities, skips unchanged', async () => {
    const synced = new Map([['a', '1'], ['b', '1']])
    const { pushed, run } = harness(
      { a: { id: 'a', v: '1' }, b: { id: 'b', v: '2' } },
      synced,
    )
    await run()
    expect(pushed).toEqual(['b']) // a unchanged, b changed
    expect(synced.get('b')).toBe('2')
  })

  it('deletes entities the local collection no longer has', async () => {
    const synced = new Map([['a', '1'], ['gone', '1']])
    const { deleted, run } = harness({ a: { id: 'a', v: '1' } }, synced)
    await run()
    expect(deleted).toEqual(['gone'])
    expect(synced.has('gone')).toBe(false)
    expect(synced.has('a')).toBe(true)
  })

  it('leaves an excluded id alone in both directions', async () => {
    // The live-observed resurrection: a confirmed DELETE pruned the synced
    // baseline while the stale state snapshot still held the thread. Without
    // the exclusion the diff reads that as NEW -> re-push (the deleted row
    // comes back), and the re-seeded fingerprint makes the next pass issue
    // an uncommanded second DELETE. Excluded-but-synced covers the mirror
    // case: a failed deletion waiting for its manual retry must not be
    // auto-deleted by the flush.
    const synced = new Map([['failed', '1']])
    const pushed: string[] = []
    const deleted: string[] = []
    await syncCollection<Item, string>({
      current: { stale: { id: 'stale', v: '9' } },
      exclude: new Set(['stale', 'failed']),
      synced,
      fingerprintOf: (item) => item.v,
      changed: (previous, value) => previous !== value,
      pushOne: async (item) => {
        pushed.push(item.id)
      },
      deleteOne: async (id) => {
        deleted.push(id)
      },
    })
    expect(pushed).toEqual([])
    expect(deleted).toEqual([])
    expect(synced.has('stale')).toBe(false) // never re-seeded
    expect(synced.get('failed')).toBe('1') // baseline kept for the retry
  })

  it('does not advance the fingerprint when the push throws (retry-safe)', async () => {
    const synced = new Map<string, string>()
    let attempts = 0
    await expect(
      syncCollection<Item, string>({
        current: { a: { id: 'a', v: '1' } },
        synced,
        fingerprintOf: (item) => item.v,
        changed: (previous, value) => previous !== value,
        pushOne: async () => {
          attempts += 1
          throw new Error('network')
        },
        deleteOne: async () => {},
      }),
    ).rejects.toThrow('network')
    expect(attempts).toBe(1)
    expect(synced.has('a')).toBe(false) // not advanced -> retried next pass
  })
})

describe('deleteTolerant404', () => {
  it('treats an already-absent entity as the successful delete state', async () => {
    const notFound = Object.assign(new Error('not found'), { status: 404 })

    await expect(
      deleteTolerant404(async () => { throw notFound }),
    ).resolves.toBeUndefined()
  })

  it('keeps non-404 failures visible to the synchronization badge', async () => {
    const unavailable = Object.assign(new Error('unavailable'), { status: 503 })

    await expect(
      deleteTolerant404(async () => { throw unavailable }),
    ).rejects.toBe(unavailable)
  })
})
