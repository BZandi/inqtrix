import { describe, expect, it, vi } from 'vitest'

import { refreshPlanAfterApprovalDecision } from './useAgentControlApi'

describe('refreshPlanAfterApprovalDecision', () => {
  it('loads the authoritative plan after plan approval returns', async () => {
    const load = vi.fn(async () => ({ status: 'approved' }))
    const onLoaded = vi.fn()
    await refreshPlanAfterApprovalDecision({
      kind: 'plan',
      load,
      onError: vi.fn(),
      onLoaded,
    })
    expect(load).toHaveBeenCalledOnce()
    expect(onLoaded).toHaveBeenCalledWith({ status: 'approved' })
  })

  it('skips non-plan approvals and surfaces refresh errors', async () => {
    const skippedLoad = vi.fn(async () => ({}))
    await refreshPlanAfterApprovalDecision({
      kind: 'patch',
      load: skippedLoad,
      onError: vi.fn(),
      onLoaded: vi.fn(),
    })
    expect(skippedLoad).not.toHaveBeenCalled()

    const onError = vi.fn()
    await refreshPlanAfterApprovalDecision({
      kind: 'replan',
      load: async () => { throw new Error('Plan refresh failed') },
      onError,
      onLoaded: vi.fn(),
    })
    expect(onError).toHaveBeenCalledWith('Plan refresh failed')
  })
})
