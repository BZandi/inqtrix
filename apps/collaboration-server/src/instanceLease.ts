import type {
  CollaborationApi,
  CollaborationSettings,
  InstanceFence,
  IntervalScheduler,
  SidecarLogger,
} from './contracts'
import { CloseCodes, CollaborationError } from './errors'
import { SidecarMetrics } from './metrics'

const scheduler: IntervalScheduler = {
  clear: (handle) => clearInterval(handle),
  every: (callback, intervalMs) => setInterval(callback, intervalMs),
}

export class InstanceLeaseManager {
  private fence: InstanceFence | null = null
  private inFlight: Promise<void> | null = null
  private interval: ReturnType<IntervalScheduler['every']> | null = null
  private running = false

  constructor(
    private readonly api: CollaborationApi,
    private readonly settings: CollaborationSettings,
    private readonly logger: SidecarLogger,
    private readonly metrics: SidecarMetrics,
    private readonly onLeaseLost: () => void,
    private readonly intervals: IntervalScheduler = scheduler,
    private readonly now: () => number = () => Date.now(),
  ) {}

  async start(): Promise<void> {
    if (this.running) return
    this.running = true
    await this.renewNow()
    if (!this.running) return
    this.interval = this.intervals.every(() => {
      void this.renewNow()
    }, this.settings.instanceRenewSeconds * 1_000)
  }

  async stop(): Promise<void> {
    this.running = false
    if (this.interval) this.intervals.clear(this.interval)
    this.interval = null
    const inFlight = this.inFlight
    this.fence = null
    this.metrics.set('inqtrix_collaboration_instance_ready', 0)
    await inFlight
  }

  async renewNow(): Promise<void> {
    if (!this.running) return
    if (this.inFlight) return this.inFlight
    this.inFlight = this.tick().finally(() => {
      this.inFlight = null
    })
    return this.inFlight
  }

  isReady(): boolean {
    return this.fence !== null && this.fence.leaseExpiresAt > this.now() / 1_000
  }

  assertActive(): InstanceFence {
    if (!this.isReady() || !this.fence) {
      throw new CollaborationError('instance_lease_lost', {
        closeCode: CloseCodes.serviceUnavailable,
        httpStatus: 503,
      })
    }
    return this.fence
  }

  private async tick(): Promise<void> {
    try {
      const next = this.fence
        ? await this.api.renewInstance({
            fence: this.fence,
            leaseSeconds: this.settings.instanceLeaseSeconds,
          })
        : await this.api.acquireInstance({
            instanceId: this.settings.instanceId,
            leaseSeconds: this.settings.instanceLeaseSeconds,
            protocolVersion: this.settings.protocolVersion,
            schemaVersion: this.settings.schemaVersion,
          })
      if (!this.running) return
      if (next.instanceId !== this.settings.instanceId) {
        throw new Error('Instance lease response referenced another instance')
      }
      if (this.fence && next.epoch !== this.fence.epoch) {
        throw new Error('Instance renewal changed the fencing epoch')
      }
      const recovered = this.fence === null
      this.fence = next
      this.metrics.set('inqtrix_collaboration_instance_ready', 1)
      this.metrics.set('inqtrix_collaboration_instance_epoch', next.epoch)
      if (recovered) {
        this.logger.info('instance_lease_acquired', {
          epoch: next.epoch,
          instance_id: next.instanceId,
        })
      }
    } catch {
      if (!this.running) return
      const wasActive = this.fence !== null
      this.fence = null
      this.metrics.set('inqtrix_collaboration_instance_ready', 0)
      this.metrics.increment('inqtrix_collaboration_instance_renew_failures_total')
      this.logger.warn('instance_lease_unavailable', {
        instance_id: this.settings.instanceId,
      })
      if (wasActive) this.onLeaseLost()
    }
  }
}
