import type { HocuspocusProvider } from '@hocuspocus/provider'
import { EDITOR_SCHEMA_VERSION } from '@inqtrix/editor-schema'
import { describe, expect, it, vi } from 'vitest'
import * as Y from 'yjs'

import type { EditorCollaborationSession } from '@/api/inqtrixClient'
import {
  COLLABORATION_UPDATE_BATCH_MS,
  CollaborationDocumentController,
  acquireLifecycleController,
  collaborationHandleForRequestedDocument,
  collaborationWebSocketUrl,
  consumeLifecycleFailure,
  createHocuspocusProvider,
  flushActiveCollaborationDocuments,
  leaseRefreshDelayMs,
  releaseLifecycleController,
  type CollaborationProviderFactoryOptions,
} from './useCollaborationDocument'

const NOW_MS = 1_000_000

type ConfiguredSession = EditorCollaborationSession & {
  provider_flush_ms: number
  refresh_after: number
}

function session(
  leaseToken: string,
  timing: Partial<Pick<ConfiguredSession, 'provider_flush_ms' | 'refresh_after'>> = {},
): ConfiguredSession {
  return {
    access: 'edit',
    expires_at: (NOW_MS + 60_000) / 1_000,
    initial_write_mode: 'edit',
    lease_token: leaseToken,
    provider_flush_ms: timing.provider_flush_ms ?? COLLABORATION_UPDATE_BATCH_MS,
    protocol_version: 1,
    room: 'inqtrix-editor-v1:document-1:g2',
    schema_version: 1,
    refresh_after: timing.refresh_after ?? (NOW_MS + 30_000) / 1_000,
    user: { color: '#2563EB', id: 'user-1', name: 'Ada' },
    websocket_path: '/collaboration',
  }
}

function createHarness(
  requestSession: (
    leaseToken?: string,
    rotationCommandId?: string,
  ) => Promise<EditorCollaborationSession>,
  hashUpdate = vi.fn(async (update: Uint8Array) => {
    void update
    return 'local-update-hash'
  }),
  options: { autoSync?: boolean } = {},
) {
  let providerOptions: CollaborationProviderFactoryOptions | null = null
  const awarenessUsers: unknown[] = []
  const provider = {} as HocuspocusProvider
  const timers = new Map<number, { callback: () => void; delayMs: number }>()
  const transportUpdates: Uint8Array[] = []
  const replayedUpdates: Uint8Array[] = []
  const rotationCommandIds: string[] = []
  let commandCounter = 0
  let nextTimerId = 1
  const cancelTimer = vi.fn((timer: number) => {
    timers.delete(timer)
  })
  const connect = vi.fn(async () => {
    providerOptions?.events.onAuthenticated()
    if (options.autoSync !== false) providerOptions?.events.onSynced()
  })
  const destroy = vi.fn()
  const disconnect = vi.fn()
  const syncToken = vi.fn(async () => {
    providerOptions?.events.onAuthenticated()
  })
  const sendStateless = vi.fn()
  const scheduleTimer = vi.fn((callback: () => void, delayMs: number) => {
    const timer = nextTimerId
    nextTimerId += 1
    timers.set(timer, { callback, delayMs })
    return timer
  })
  const controller = new CollaborationDocumentController(
    {
      documentId: 'document-1',
      generation: 2,
      initialPersistedSequence: 4,
      requestSession,
      resolveWebSocketUrl: (path) => collaborationWebSocketUrl(path, {
        host: 'desk.test',
        protocol: 'https:',
      }),
      schemaVersion: EDITOR_SCHEMA_VERSION,
    },
    {
      cancelTimer,
      createCommandId: () => {
        commandCounter += 1
        const commandId = `00000000-0000-4000-8000-${String(commandCounter).padStart(12, '0')}`
        rotationCommandIds.push(commandId)
        return commandId
      },
      createDocument: () => new Y.Doc(),
      createProvider: (options) => {
        providerOptions = options
        options.document.on('update', (update, origin) => {
          if (origin !== provider) transportUpdates.push(update)
        })
        return {
          connect,
          destroy,
          disconnect,
          provider,
          replayUpdate: (update) => replayedUpdates.push(Uint8Array.from(update)),
          sendStateless,
          setAwarenessUser: (user) => awarenessUsers.push(user),
          syncToken,
        }
      },
      hashUpdate,
      now: () => NOW_MS,
      scheduleTimer,
    },
  )

  function runTimer(delayMs: number): void {
    const match = [...timers.entries()].find(([, timer]) => timer.delayMs === delayMs)
    if (!match) throw new Error(`No ${delayMs}ms timer is scheduled.`)
    const [timerId, timer] = match
    timers.delete(timerId)
    timer.callback()
  }

  return {
    awarenessUsers,
    connect,
    controller,
    destroy,
    disconnect,
    getProviderOptions: () => providerOptions,
    hashUpdate,
    provider,
    replayedUpdates,
    rotationCommandIds,
    runTimer,
    scheduleTimer,
    sendStateless,
    scheduledDelays: () => [...timers.values()].map((timer) => timer.delayMs),
    syncToken,
    transportUpdates,
    emitSynced: () => providerOptions?.events.onSynced(),
  }
}

describe('collaboration document lifecycle', () => {
  it('returns an inactive B handle synchronously on the hook A-to-B rerender boundary', () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    const documentA = harness.controller.getSnapshot()
    const renderedA = collaborationHandleForRequestedDocument(
      documentA,
      'document-1',
      2,
      true,
    )

    const documentB = collaborationHandleForRequestedDocument(
      renderedA,
      'document-2',
      3,
      true,
    )

    expect(renderedA).toBe(documentA)
    expect(documentB).toMatchObject({
      canEdit: false,
      document: null,
      documentId: 'document-2',
      generation: 3,
      lifecycleStatus: 'inactive',
      provider: null,
      synced: false,
    })
  })

  it('remains read-only while authentication has completed but Yjs sync is delayed', async () => {
    const harness = createHarness(
      vi.fn().mockResolvedValue(session('initial-token')),
      undefined,
      { autoSync: false },
    )

    await harness.controller.start()

    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: false,
      connectionStatus: 'connecting',
      lifecycleStatus: 'syncing',
    })

    harness.emitSynced()

    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: true,
      connectionStatus: 'connected',
      lifecycleStatus: 'saved',
    })
  })

  it('joins without a lease then refreshes the same lease and token-syncs the socket', async () => {
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockResolvedValueOnce(session('rotated-token'))
    const harness = createHarness(requestSession)

    await harness.controller.start()

    expect(requestSession.mock.calls[0]).toEqual([])
    expect(harness.getProviderOptions()?.url).toBe('wss://desk.test/collaboration')
    expect(harness.getProviderOptions()?.getToken()).toBe('initial-token')
    expect(harness.awarenessUsers).toEqual([
      { color: '#2563EB', id: 'user-1', name: 'Ada' },
    ])
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: true,
      connectionStatus: 'connected',
    })

    await harness.controller.refreshLease()

    expect(requestSession.mock.calls[1]).toEqual([
      'initial-token',
      harness.rotationCommandIds[0],
    ])
    expect(harness.getProviderOptions()?.getToken()).toBe('rotated-token')
    expect(harness.syncToken).toHaveBeenCalledOnce()
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: true,
      connectionStatus: 'connected',
    })
    expect(JSON.stringify(harness.controller.getSnapshot())).not.toContain('rotated-token')
  })

  it('becomes reconnecting and read-only when a clean lease refresh fails', async () => {
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockRejectedValueOnce(new Error('network unavailable'))
    const harness = createHarness(requestSession)
    await harness.controller.start()

    await harness.controller.refreshLease()

    expect(requestSession.mock.calls[1]).toEqual([
      'initial-token',
      harness.rotationCommandIds[0],
    ])
    expect(harness.disconnect).toHaveBeenCalled()
    expect(harness.scheduledDelays()).toContain(5_000)
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: false,
      connectionStatus: 'reconnecting',
      error: expect.stringContaining('lease could not be refreshed'),
    })
  })

  it('reuses one lease rotation command id after a lost HTTP response', async () => {
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockRejectedValueOnce(new Error('response lost'))
      .mockResolvedValueOnce(session('reconstructed-token'))
    const harness = createHarness(requestSession)
    await harness.controller.start()

    await harness.controller.refreshLease()
    await harness.controller.refreshLease('reconnect')

    expect(requestSession.mock.calls.slice(1)).toEqual([
      ['initial-token', harness.rotationCommandIds[0]],
      ['initial-token', harness.rotationCommandIds[0]],
    ])
    expect(harness.rotationCommandIds).toHaveLength(1)
    expect(harness.getProviderOptions()?.getToken()).toBe('reconstructed-token')
  })

  it('recovers an expired rotation with a fresh cookie-authenticated lease', async () => {
    const expired = Object.assign(new Error('lease expired'), {
      detail: { reason: 'lease_expired' },
      status: 401,
    })
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockRejectedValueOnce(expired)
      .mockResolvedValueOnce(session('fresh-token'))
    const harness = createHarness(requestSession)
    await harness.controller.start()

    await harness.controller.refreshLease()

    expect(requestSession.mock.calls).toEqual([
      [],
      ['initial-token', harness.rotationCommandIds[0]],
      [],
    ])
    expect(harness.getProviderOptions()?.getToken()).toBe('fresh-token')
    expect(harness.syncToken).toHaveBeenCalledOnce()
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: true,
      connectionStatus: 'connected',
    })
  })

  it('rotates the lease before reconnecting a clean transport', async () => {
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockResolvedValueOnce(session('reconnect-token'))
    const harness = createHarness(requestSession)
    await harness.controller.start()

    harness.getProviderOptions()?.events.onClose(1006, 'network_lost')
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: false,
      connectionStatus: 'reconnecting',
    })

    harness.runTimer(5_000)
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    expect(requestSession.mock.calls).toEqual([
      [],
      ['initial-token', harness.rotationCommandIds[0]],
    ])
    expect(harness.getProviderOptions()?.getToken()).toBe('reconnect-token')
    expect(harness.connect).toHaveBeenCalledTimes(2)
    expect(harness.controller.getSnapshot().connectionStatus).toBe('connected')
  })

  it('stops reconnecting on protocol close conflicts', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()

    harness.getProviderOptions()?.events.onClose(4409, 'schema_conflict')

    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: false,
      connectionStatus: 'incompatible',
    })
  })

  it('batches local updates for at most 50ms and hashes the single transported update', async () => {
    const hashUpdate = vi.fn(async (update: Uint8Array) => {
      void update
      return 'merged-update-hash'
    })
    const harness = createHarness(
      vi.fn().mockResolvedValue(session('initial-token')),
      hashUpdate,
    )
    await harness.controller.start()
    const editorDocument = harness.controller.getSnapshot().document

    editorDocument?.getMap('test').set('title', 'Draft')
    editorDocument?.getMap('test').set('subtitle', 'Notes')

    expect(harness.scheduledDelays()).toContain(COLLABORATION_UPDATE_BATCH_MS)
    expect(COLLABORATION_UPDATE_BATCH_MS).toBeLessThanOrEqual(50)
    expect(harness.transportUpdates).toHaveLength(0)

    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    expect(harness.transportUpdates).toHaveLength(1)
    expect(hashUpdate).toHaveBeenCalledOnce()
    expect([...hashUpdate.mock.calls[0][0]]).toEqual([...harness.transportUpdates[0]])
    expect(harness.getProviderOptions()?.document.getMap('test').toJSON()).toEqual({
      subtitle: 'Notes',
      title: 'Draft',
    })
    expect(harness.controller.getSnapshot()).toMatchObject({
      durabilityStatus: 'pending',
      pendingHashes: ['merged-update-hash'],
    })
  })

  it('reports pending local hashes as saved only after the matching durable ack', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()

    harness.controller.getSnapshot().document?.getMap('test').set('title', 'Draft')
    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    expect(harness.controller.getSnapshot()).toMatchObject({
      durabilityStatus: 'pending',
      pendingHashes: ['local-update-hash'],
    })

    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'local-update-hash',
      sequence: 5,
      type: 'durable_ack',
    }))

    expect(harness.controller.getSnapshot()).toMatchObject({
      activityRevision: 1,
      durabilityStatus: 'saved',
      lastLocalDurableSequence: 5,
      lastPersistedSequence: 4,
      pendingHashes: [],
    })
  })

  it('flushes the browser batch and waits for every durable acknowledgement', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()
    harness.controller.getSnapshot().document?.getMap('test').set('title', 'Export')

    let resolved = false
    const flushed = harness.controller.flushAndAwaitDurability().then(() => {
      resolved = true
    })
    await Promise.resolve()
    await Promise.resolve()

    expect(harness.transportUpdates).toHaveLength(1)
    expect(resolved).toBe(false)
    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'local-update-hash',
      sequence: 8,
      type: 'durable_ack',
    }))
    await flushed
    expect(resolved).toBe(true)
  })

  it('keeps local acknowledgements separate from authoritative peer and decision sequences', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()
    expect(harness.controller.getSnapshot()).toMatchObject({
      activityRevision: 0,
      lastLocalDurableSequence: 0,
      lastPersistedSequence: 4,
    })
    harness.controller.getSnapshot().document?.getMap('test').set('title', 'Local')
    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()
    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'local-update-hash',
      sequence: 9,
      type: 'durable_ack',
    }))

    expect(harness.controller.getSnapshot()).toMatchObject({
      activityRevision: 1,
      lastLocalDurableSequence: 9,
      lastPersistedSequence: 4,
    })
    harness.controller.updateAuthoritativeSequence(12)
    harness.controller.updateAuthoritativeSequence(11)
    harness.controller.getSnapshot().updateAuthoritativeSequence(15)

    expect(harness.controller.getSnapshot()).toMatchObject({
      activityRevision: 3,
      lastLocalDurableSequence: 9,
      lastPersistedSequence: 15,
    })
  })

  it('reconciles retained hashes after reconnect authentication', async () => {
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockResolvedValueOnce(session('reconnect-token'))
    const harness = createHarness(requestSession)
    await harness.controller.start()
    harness.controller.getSnapshot().document?.getMap('test').set('title', 'Draft')
    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    harness.getProviderOptions()?.events.onClose(1006, 'network_lost')
    expect(harness.controller.getSnapshot()).toMatchObject({
      connectionStatus: 'reconnecting',
      durabilityStatus: 'pending',
      pendingHashes: ['local-update-hash'],
    })

    harness.runTimer(5_000)
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    expect(harness.sendStateless).toHaveBeenCalledWith(JSON.stringify({
      type: 'durability_reconcile',
      hashes: ['local-update-hash'],
    }))
    expect(harness.replayedUpdates).toHaveLength(1)
    expect([...harness.replayedUpdates[0]]).toEqual([
      ...harness.transportUpdates[0],
    ])
    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'local-update-hash',
      sequence: 5,
      type: 'durable_ack',
    }))

    expect(requestSession).toHaveBeenCalledTimes(2)
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: true,
      connectionStatus: 'connected',
      durabilityStatus: 'saved',
      pendingHashes: [],
    })
  })

  it('exposes a live monotonic authority revision through retained handle snapshots', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()
    const retainedHandle = harness.controller.getSnapshot()
    const writable = retainedHandle.readAuthority()

    expect(writable).toMatchObject({
      access: 'edit',
      canEdit: true,
      connectionStatus: 'connected',
      lifecycleStatus: 'saved',
      synced: true,
    })

    retainedHandle.document?.getMap('test').set('title', 'Pending')
    expect(retainedHandle.readAuthority().revision).toBe(writable.revision)

    harness.getProviderOptions()?.events.onClose(1006, 'network_lost')
    expect(retainedHandle.readAuthority()).toMatchObject({
      access: 'edit',
      canEdit: false,
      connectionStatus: 'reconnecting',
      lifecycleStatus: 'reconnecting',
      synced: false,
    })
    expect(retainedHandle.readAuthority().revision).toBeGreaterThan(writable.revision)
  })

  it('flushes before release and retains the provider until durable ack', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()
    harness.controller.getSnapshot().document?.getMap('test').set('title', 'Navigate')

    harness.controller.release()

    expect(harness.transportUpdates).toHaveLength(1)
    expect(harness.destroy).not.toHaveBeenCalled()
    await Promise.resolve()
    await Promise.resolve()
    expect(harness.controller.getSnapshot().pendingHashes).toEqual([
      'local-update-hash',
    ])
    expect(harness.destroy).not.toHaveBeenCalled()

    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'local-update-hash',
      sequence: 5,
      type: 'durable_ack',
    }))

    expect(harness.destroy).toHaveBeenCalledOnce()
  })

  it('waits for active collaboration durability before the logout boundary', async () => {
    const key = 'workspace-1:logout-boundary:g2'
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    const controller = acquireLifecycleController(key, () => harness.controller)
    await controller.start()
    controller.getSnapshot().document?.getMap('test').set('title', 'Before logout')

    let durable = false
    const boundary = flushActiveCollaborationDocuments().then(() => {
      durable = true
    })
    await Promise.resolve()
    await Promise.resolve()

    expect(harness.transportUpdates).toHaveLength(1)
    expect(durable).toBe(false)
    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'local-update-hash',
      sequence: 5,
      type: 'durable_ack',
    }))
    await boundary
    expect(durable).toBe(true)
    releaseLifecycleController(key, controller)
  })

  it('reuses a released pending controller and removes it after the acked remount', async () => {
    const key = 'workspace-1:registry-remount:g2'
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    const controller = acquireLifecycleController(key, () => harness.controller)
    await controller.start()
    controller.getSnapshot().document?.getMap('test').set('title', 'Retained')
    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    releaseLifecycleController(key, controller)
    expect(harness.destroy).not.toHaveBeenCalled()
    const remounted = acquireLifecycleController(key, () => {
      throw new Error('pending lifecycle entry should be reused')
    })
    expect(remounted).toBe(controller)

    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'local-update-hash',
      sequence: 5,
      type: 'durable_ack',
    }))
    expect(harness.destroy).not.toHaveBeenCalled()
    releaseLifecycleController(key, remounted)
    expect(harness.destroy).toHaveBeenCalledOnce()

    const replacementHarness = createHarness(
      vi.fn().mockResolvedValue(session('replacement-token')),
    )
    const replacement = acquireLifecycleController(
      key,
      () => replacementHarness.controller,
    )
    expect(replacement).toBe(replacementHarness.controller)
    releaseLifecycleController(key, replacement)
  })

  it('retains a timed-out controller for recovery and cleans the registry after ack', async () => {
    const key = 'workspace-1:registry-timeout:g2'
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    const controller = acquireLifecycleController(key, () => harness.controller)
    await controller.start()
    controller.getSnapshot().document?.getMap('test').set('title', 'Unconfirmed')

    releaseLifecycleController(key, controller)
    expect(harness.transportUpdates).toHaveLength(1)
    harness.runTimer(60_000)

    expect(harness.destroy).not.toHaveBeenCalled()
    expect(harness.controller.getSnapshot()).toMatchObject({
      blockingFailure: expect.stringContaining('could not be confirmed'),
      canEdit: false,
      durabilityStatus: 'error',
    })
    expect(consumeLifecycleFailure(key)).toContain('could not be confirmed')
    const replacementHarness = createHarness(
      vi.fn().mockResolvedValue(session('replacement-token')),
    )
    const remounted = acquireLifecycleController(
      key,
      () => {
        throw new Error('the retained controller must be reused')
      },
    )
    expect(remounted).toBe(controller)
    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'local-update-hash',
      sequence: 5,
      type: 'durable_ack',
    }))
    await Promise.resolve()
    await Promise.resolve()
    expect(remounted.getSnapshot()).toMatchObject({
      blockingFailure: null,
      durabilityStatus: 'saved',
    })
    releaseLifecycleController(key, remounted)
    expect(harness.destroy).toHaveBeenCalledOnce()

    const replacement = acquireLifecycleController(key, () => replacementHarness.controller)
    expect(replacement).toBe(replacementHarness.controller)
    releaseLifecycleController(key, replacement)
  })

  it('mirrors exact provider-origin updates without tracking them as local durability', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()
    const remoteDocument = new Y.Doc()
    remoteDocument.getMap('test').set('remote-title', 'Published')
    const remoteUpdate = Y.encodeStateAsUpdate(remoteDocument)
    const scheduledCallsBeforeRemoteUpdate = harness.scheduleTimer.mock.calls.length

    const transportDocument = harness.getProviderOptions()?.document
    expect(transportDocument).toBeDefined()
    Y.applyUpdate(transportDocument!, remoteUpdate, harness.provider)

    const unrelatedDocument = new Y.Doc()
    unrelatedDocument.getMap('test').set('wrong-origin', 'ignored')
    Y.applyUpdate(
      transportDocument!,
      Y.encodeStateAsUpdate(unrelatedDocument),
      { source: 'not-the-provider' },
    )

    expect(harness.controller.getSnapshot().document?.getMap('test').toJSON()).toEqual({
      'remote-title': 'Published',
    })
    expect(harness.controller.getSnapshot().activityRevision).toBe(1)
    expect(harness.scheduleTimer).toHaveBeenCalledTimes(scheduledCallsBeforeRemoteUpdate)
    expect(harness.hashUpdate).not.toHaveBeenCalled()
    expect(harness.controller.getSnapshot().durabilityStatus).toBe('idle')
    remoteDocument.destroy()
    unrelatedDocument.destroy()
  })

  it('stays read-only after a durable rejection even if a later ack arrives', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()
    harness.controller.getSnapshot().document?.getMap('test').set('title', 'Draft')
    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      code: 'too_large',
      hash: 'local-update-hash',
      type: 'durable_rejection',
    }))
    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'local-update-hash',
      sequence: 5,
      type: 'durable_ack',
    }))

    expect(harness.disconnect).toHaveBeenCalled()
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: false,
      connectionStatus: 'error',
      durabilityStatus: 'error',
      pendingHashes: ['local-update-hash'],
    })

    harness.controller.release()
    expect(harness.destroy).not.toHaveBeenCalled()
    expect(harness.controller.getSnapshot()).toMatchObject({
      blockingFailure: expect.stringContaining('rejected'),
      pendingHashes: ['local-update-hash'],
    })
  })

  it('rejects a document schema conflict before issuing a session', async () => {
    const requestSession = vi.fn().mockResolvedValue(session('unused-token'))
    const controller = new CollaborationDocumentController(
      {
        documentId: 'document-1',
        generation: 2,
        initialPersistedSequence: 0,
        requestSession,
        resolveWebSocketUrl: (path) => path,
        schemaVersion: EDITOR_SCHEMA_VERSION + 1,
      },
      {
        cancelTimer: vi.fn(),
        createCommandId: () => '00000000-0000-4000-8000-000000000001',
        createDocument: () => new Y.Doc(),
        createProvider: () => {
          throw new Error('must not create provider')
        },
        hashUpdate: async () => 'hash',
        now: () => NOW_MS,
        scheduleTimer: vi.fn(() => 1),
      },
    )

    await controller.start()

    expect(requestSession).not.toHaveBeenCalled()
    expect(controller.getSnapshot().connectionStatus).toBe('incompatible')
    controller.destroy()
  })
})

describe('collaboration timing and transport', () => {
  it('refreshes halfway through the lease and keeps websocket auth same-origin', () => {
    expect(leaseRefreshDelayMs(1_060, NOW_MS)).toBe(30_000)
    expect(collaborationWebSocketUrl('collaboration', {
      host: 'desk.test:5173',
      protocol: 'http:',
    })).toBe('ws://desk.test:5173/collaboration')
  })

  it('forwards only the OWN awareness state, never received remote states', () => {
    // Hocuspocus 4.3 re-broadcasts every changed awareness client — even
    // states just received from the server. The sidecar's identity gate
    // closes such connections with 4403 (invalid_request) as soon as a
    // second participant or a stale clientID appears (live incident
    // 2026-07-15), so the adapter must filter to the local clientID.
    const document = new Y.Doc()
    const adapter = createHocuspocusProvider({
      document,
      events: {
        onAuthenticated: () => undefined,
        onAuthenticationFailed: () => undefined,
        onClose: () => undefined,
        onStateless: () => undefined,
        onSynced: () => undefined,
      },
      getToken: () => 'lease-token',
      room: 'inqtrix-editor-v1:doc-1:g1',
      url: 'ws://localhost:9/collaboration',
    })
    const forwarded: Array<{ added: number[]; removed: number[]; updated: number[] }> = []
    adapter.provider.awarenessUpdateHandler = ((changes: {
      added: number[]
      removed: number[]
      updated: number[]
    }) => {
      forwarded.push(changes)
    }) as typeof adapter.provider.awarenessUpdateHandler
    try {
      const awareness = adapter.provider.awareness
      expect(awareness).not.toBeNull()
      // A remote participant's state arriving from the server (its update
      // event names a FOREIGN clientID) must NOT be echoed back.
      awareness!.emit('update', [
        { added: [document.clientID + 1], removed: [], updated: [] },
        'server',
      ])
      expect(forwarded).toHaveLength(0)
      // The local user's own state must still be forwarded.
      awareness!.setLocalStateField('user', { id: 'local', name: 'L' })
      expect(forwarded).toHaveLength(1)
      expect(
        [...forwarded[0].added, ...forwarded[0].updated],
      ).toContain(document.clientID)
    } finally {
      adapter.destroy()
      document.destroy()
    }
  })

  it('attaches the real provider to its external websocket transport', () => {
    // Hocuspocus 4.x only auto-attaches when it manages its own socket;
    // with the adapter's external websocketProvider a missing attach()
    // means the auth token is NEVER sent and every participant stays
    // read-only forever (live incident 2026-07-15). The controller tests
    // replace this factory with fakes, so this pin drives the real one.
    const document = new Y.Doc()
    const adapter = createHocuspocusProvider({
      document,
      events: {
        onAuthenticated: () => undefined,
        onAuthenticationFailed: () => undefined,
        onClose: () => undefined,
        onStateless: () => undefined,
        onSynced: () => undefined,
      },
      getToken: () => 'lease-token',
      room: 'inqtrix-editor-v1:doc-1:g1',
      url: 'ws://localhost:9/collaboration',
    })
    try {
      expect(adapter.provider.isAttached).toBe(true)
    } finally {
      adapter.destroy()
      document.destroy()
    }
  })

  it('honors non-default server flush and refresh timing', async () => {
    const configured = session('configured-token', {
      provider_flush_ms: 23,
      refresh_after: (NOW_MS + 17_000) / 1_000,
    })
    const harness = createHarness(vi.fn().mockResolvedValue(configured))

    await harness.controller.start()
    harness.controller.getSnapshot().document?.getMap('test').set('title', 'Timed')

    expect(harness.scheduledDelays()).toContain(17_000)
    expect(harness.scheduledDelays()).toContain(23)
    harness.runTimer(23)
    expect(harness.transportUpdates).toHaveLength(1)
  })
})
