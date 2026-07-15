import type { IntervalScheduler, TimerHandle } from '../src/contracts'
import { InstanceLeaseManager } from '../src/instanceLease'
import { SidecarMetrics } from '../src/metrics'
import { FakeCollaborationApi, deferred, settings, silentLogger } from './helpers'

describe('single-replica instance fencing', () => {
  it('becomes unready and invokes the close callback when renewal is lost', async () => {
    const api = new FakeCollaborationApi()
    const lost = vi.fn()
    let scheduled: (() => void) | null = null
    const handle = {} as TimerHandle
    const intervals: IntervalScheduler = {
      clear: vi.fn(),
      every: (callback) => {
        scheduled = callback
        return handle
      },
    }
    const manager = new InstanceLeaseManager(
      api,
      settings(),
      silentLogger,
      new SidecarMetrics(),
      lost,
      intervals,
    )

    await manager.start()
    expect(manager.isReady()).toBe(true)
    expect(scheduled).not.toBeNull()
    api.renewError = new Error('fence unavailable')
    await manager.renewNow()

    expect(manager.isReady()).toBe(false)
    expect(lost).toHaveBeenCalledOnce()
    expect(() => manager.assertActive()).toThrowError('instance_lease_lost')
    await manager.stop()
  })

  it('invalidates the local fence before an in-flight renewal can finish shutdown', async () => {
    const api = new FakeCollaborationApi()
    const renewal = deferred<typeof api.fence>()
    const manager = new InstanceLeaseManager(
      api,
      settings(),
      silentLogger,
      new SidecarMetrics(),
      () => undefined,
    )
    await manager.start()
    api.renewInstance = async () => renewal.promise
    const renewing = manager.renewNow()
    const stopping = manager.stop()

    expect(manager.isReady()).toBe(false)
    expect(() => manager.assertActive()).toThrowError('instance_lease_lost')
    renewal.resolve(api.fence)
    await Promise.all([renewing, stopping])
    expect(manager.isReady()).toBe(false)
  })
})
