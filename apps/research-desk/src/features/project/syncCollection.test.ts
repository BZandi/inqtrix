import { describe, expect, it } from 'vitest'

import { syncCollection } from './syncCollection'

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
