import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

// Source pin for effect wiring no node-environment test can mount (this
// repo's vitest lane is DOM-less by design; component rendering is
// verified in the browser). The channel RESOLVES its promise after
// calling onUnavailable, so the .then handler runs one microtask after
// the error state was set — without the guard it overwrote 'error' with
// a calm 'settled', silencing the destructive styling the adjacent
// comment promises. Killing mutant: dropping `&& !unavailable` from the
// settled transition (all 1515 other tests stayed green under exactly
// that mutation).
describe('TaskDetailView child transport wiring', () => {
  const source = readFileSync(
    path.resolve(__dirname, './views.tsx'),
    'utf-8',
  )

  it('latches unavailability before the channel resolves', () => {
    expect(source).toContain('let unavailable = false')
    const handler = source.slice(source.indexOf('onUnavailable: () => {'))
    const handlerBody = handler.slice(0, handler.indexOf('},'))
    expect(handlerBody).toContain('unavailable = true')
    expect(handlerBody).toContain("setTransport('error')")
  })

  it("guards the settled transition on the latch, not only on abort", () => {
    expect(source).toMatch(
      /!controller\.signal\.aborted && !unavailable/,
    )
  })
})
