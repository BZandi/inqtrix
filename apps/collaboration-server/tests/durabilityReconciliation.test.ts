import { getEditorSchemaFingerprint } from '@inqtrix/editor-schema'

import type { ConnectionContext } from '../src/contracts'
import { reconcileDurability } from '../src/durabilityReconciliation'
import { InstanceLeaseManager } from '../src/instanceLease'
import { SidecarMetrics } from '../src/metrics'
import {
  FakeCollaborationApi,
  USER_ID,
  settings,
  silentLogger,
} from './helpers'

describe('stateless durability reconciliation', () => {
  it('acknowledges found hashes and leaves missing hashes for normal Yjs resync', async () => {
    const fixture = await reconciliationFixture()
    const found = 'a'.repeat(64)
    const missing = 'b'.repeat(64)
    fixture.api.lookupResults = [{ hash: found, sequence: 9 }]

    await expect(reconcileDurability(
      JSON.stringify({ type: 'durability_reconcile', hashes: [found, missing] }),
      fixture.context,
      fixture.sender,
      fixture.api,
      fixture.lease,
      fixture.configured,
    )).resolves.toBe(1)

    expect(fixture.api.lookups).toEqual([{
      documentId: 'ed_test',
      fence: fixture.api.fence,
      generation: 1,
      hashes: [found, missing],
    }])
    expect(fixture.sender.sendStateless).toHaveBeenCalledOnce()
    expect(fixture.sender.sendStateless).toHaveBeenCalledWith(JSON.stringify({
      hash: found,
      sequence: 9,
      type: 'durable_ack',
    }))
    await fixture.close()
  })

  it.each([
    ['malformed JSON', '{'],
    ['extra fields', JSON.stringify({ type: 'durability_reconcile', hashes: ['a'.repeat(64)], extra: true })],
    ['duplicate hashes', JSON.stringify({ type: 'durability_reconcile', hashes: ['a'.repeat(64), 'a'.repeat(64)] })],
    ['invalid hash', JSON.stringify({ type: 'durability_reconcile', hashes: ['not-a-hash'] })],
    ['too many hashes', JSON.stringify({ type: 'durability_reconcile', hashes: ['a'.repeat(64), 'b'.repeat(64)] })],
  ])('rejects %s without calling the lookup API', async (_label, payload) => {
    const fixture = await reconciliationFixture({ reconcileMaxHashes: 1 })

    await expect(reconcileDurability(
      payload,
      fixture.context,
      fixture.sender,
      fixture.api,
      fixture.lease,
      fixture.configured,
    )).rejects.toThrowError('invalid_request')
    expect(fixture.api.lookups).toHaveLength(0)
    expect(fixture.sender.sendStateless).not.toHaveBeenCalled()
    await fixture.close()
  })

  it('rejects an oversized payload before parsing or lookup', async () => {
    const fixture = await reconciliationFixture({ frameLimitBytes: 16 })

    await expect(reconcileDurability(
      JSON.stringify({ type: 'durability_reconcile', hashes: ['a'.repeat(64)] }),
      fixture.context,
      fixture.sender,
      fixture.api,
      fixture.lease,
      fixture.configured,
    )).rejects.toThrowError('message_too_large')
    expect(fixture.api.lookups).toHaveLength(0)
    await fixture.close()
  })
})

async function reconciliationFixture(
  overrides: Parameters<typeof settings>[0] = {},
) {
  const api = new FakeCollaborationApi()
  const configured = settings(overrides)
  const lease = new InstanceLeaseManager(
    api,
    configured,
    silentLogger,
    new SidecarMetrics(),
    () => undefined,
  )
  await lease.start()
  const context: ConnectionContext = {
    access: 'edit',
    documentId: 'ed_test',
    expiresAt: Date.now() / 1_000 + 60,
    generation: 1,
    leaseId: 'lease-1',
    policyCursor: 0,
    protocolVersion: configured.protocolVersion,
    schemaHash: await getEditorSchemaFingerprint(),
    schemaVersion: configured.schemaVersion,
    sessionId: 'session-1',
    tenantId: configured.tenantId,
    user: { color: '#123456', id: USER_ID, name: 'Ada' },
  }
  return {
    api,
    close: () => lease.stop(),
    configured,
    context,
    lease,
    sender: { sendStateless: vi.fn() },
  }
}
