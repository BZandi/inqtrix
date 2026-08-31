import { describe, expect, it, vi } from 'vitest'

import { withControlFetchDeadline } from './useAgentControlApi'

const MESSAGE = 'Die Entscheidung konnte nicht geladen werden.'

describe('withControlFetchDeadline', () => {
  it('passes a normal answer straight through', async () => {
    await expect(
      withControlFetchDeadline(Promise.resolve('ok'), MESSAGE, 50),
    ).resolves.toBe('ok')
  })

  it('keeps a real failure as it is', async () => {
    await expect(
      withControlFetchDeadline(
        Promise.reject(new Error('404')),
        MESSAGE,
        50,
      ),
    ).rejects.toThrow('404')
  })

  it('turns an endless wait into a stated failure', async () => {
    // The regression: with the browser's connection pool exhausted the
    // fetch neither resolved nor rejected. It stayed marked in flight,
    // so nothing was shown and nothing retried — the surface said
    // "waiting for your approval" while unable to offer one.
    await expect(
      withControlFetchDeadline(new Promise(() => undefined), MESSAGE, 5),
    ).rejects.toThrow(MESSAGE)
  })

  it('clears its timer once the work settled', async () => {
    const cancel = vi.fn()
    await withControlFetchDeadline(
      Promise.resolve('ok'),
      MESSAGE,
      50,
      setTimeout,
      cancel as never,
    )
    expect(cancel).toHaveBeenCalledTimes(1)
  })
})
