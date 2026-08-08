import type { HocuspocusProvider } from '@hocuspocus/provider'
import {
  EDITOR_SCHEMA_VERSION,
  INQTRIX_STRUCTURE_SUGGESTION_ATTR,
  SUGGESTION_MARK_NAMES,
} from '@inqtrix/editor-schema'
import { describe, expect, it, vi } from 'vitest'
import * as Y from 'yjs'

import type { EditorCollaborationSession } from '@/api/inqtrixClient'
import {
  COLLABORATION_UPDATE_BATCH_MS,
  CollaborationDocumentController,
  COLLABORATION_AWARENESS_THROTTLE_MS,
  collaborationReconnectDelayMs,
  containsStructureSuggestionAttribute,
  acquireLifecycleController,
  collaborationDocumentLifecycleHasUnconfirmedChanges,
  collaborationHandleForRequestedDocument,
  collaborationWebSocketUrl,
  consumeLifecycleFailure,
  createHocuspocusProvider,
  flushActiveCollaborationDocuments,
  leaseRefreshDelayMs,
  releaseLifecycleController,
  retireCollaborationDocumentLifecycle,
  type CollaborationProviderFactoryOptions,
} from './useCollaborationDocument'

const NOW_MS = 1_000_000

function updateSuggestionFormatKeys(update: Uint8Array): string[] {
  return Y.decodeUpdate(update).structs.flatMap((struct) => (
    struct instanceof Y.Item
    && struct.content instanceof Y.ContentFormat
    && SUGGESTION_MARK_NAMES.has(struct.content.key)
      ? [struct.content.key]
      : []
  ))
}

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
    schema_version: EDITOR_SCHEMA_VERSION,
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
  options: { acknowledgeTokenSync?: boolean; autoSync?: boolean } = {},
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
    if (options.acknowledgeTokenSync !== false) {
      providerOptions?.events.onAuthenticated()
    }
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
      random: () => 0.5,
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

  it('stays ready when periodic token sync has no authentication acknowledgement', async () => {
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockResolvedValueOnce(session('rotated-token'))
    const harness = createHarness(requestSession, undefined, {
      acknowledgeTokenSync: false,
    })
    await harness.controller.start()

    await harness.controller.refreshLease()

    expect(harness.syncToken).toHaveBeenCalledOnce()
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: true,
      connectionStatus: 'connected',
      lifecycleStatus: 'saved',
      synced: true,
    })
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
    expect(harness.scheduledDelays()).toContain(1_000)
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: false,
      connectionStatus: 'reconnecting',
      error: expect.stringContaining('lease could not be refreshed'),
      nextReconnectAt: NOW_MS + 1_000,
      reconnectAttempt: 1,
      recoverability: 'retry',
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

    harness.runTimer(1_000)
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

  it.each(['service_unavailable', 'restarting'])(
    'reissues the lease after the in-band %s recovery signal',
    async (reason) => {
      const requestSession = vi.fn()
        .mockResolvedValueOnce(session('initial-token'))
        .mockResolvedValueOnce(session(`${reason}-token`))
      const harness = createHarness(requestSession)
      await harness.controller.start()
      harness.disconnect.mockClear()

      harness.getProviderOptions()?.events.onClose(1_000, reason)

      expect(harness.disconnect).toHaveBeenCalled()
      expect(harness.controller.getSnapshot()).toMatchObject({
        canEdit: false,
        connectionStatus: 'reconnecting',
        recoverability: 'retry',
      })

      harness.runTimer(1_000)
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()

      expect(requestSession.mock.calls).toEqual([
        [],
        ['initial-token', harness.rotationCommandIds[0]],
      ])
      expect(harness.getProviderOptions()?.getToken()).toBe(`${reason}-token`)
      expect(harness.connect).toHaveBeenCalledTimes(2)
      expect(harness.controller.getSnapshot()).toMatchObject({
        canEdit: true,
        connectionStatus: 'connected',
      })
    },
  )

  it('guards parallel manual retries and cancels the scheduled reconnect', async () => {
    let resolveRetry!: (value: EditorCollaborationSession) => void
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveRetry = resolve
      }))
    const harness = createHarness(requestSession)
    await harness.controller.start()
    harness.getProviderOptions()?.events.onClose(1006, 'network_lost')

    const first = harness.controller.getSnapshot().retryConnection()
    const second = harness.controller.getSnapshot().retryConnection()
    expect(requestSession).toHaveBeenCalledTimes(2)
    expect(harness.scheduledDelays()).not.toContain(1_000)
    expect(harness.controller.getSnapshot()).toMatchObject({
      nextReconnectAt: null,
      reconnectAttempt: 1,
      recoverability: 'retry',
    })

    resolveRetry(session('manual-retry-token'))
    await Promise.all([first, second])

    expect(requestSession).toHaveBeenCalledTimes(2)
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: true,
      connectionStatus: 'connected',
      nextReconnectAt: null,
      reconnectAttempt: 0,
      recoverability: 'none',
    })
  })

  it('classifies an expired authenticated session as a login recovery', async () => {
    const expired = Object.assign(new Error('session expired'), { status: 401 })
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockRejectedValueOnce(expired)
      .mockRejectedValueOnce(expired)
    const harness = createHarness(requestSession)
    await harness.controller.start()

    await harness.controller.refreshLease()

    expect(harness.controller.getSnapshot()).toMatchObject({
      connectionStatus: 'error',
      nextReconnectAt: null,
      recoverability: 'login',
    })
    expect(harness.scheduledDelays()).not.toContain(1_000)
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

  it('classifies an in-band close by its reason, because the code arrives as 1000', async () => {
    // A close the collaboration service sends in band reaches the provider
    // with the code rewritten to 1000 and only the reason preserved. Reading
    // the code first therefore misses every close the service itself starts.
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()

    harness.getProviderOptions()?.events.onClose(1_000, 'update_required')

    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: false,
      connectionStatus: 'incompatible',
    })
  })

  it('disconnects the transport before reconnecting after a rejected update', async () => {
    // The reconnect request is a no-op while the socket is still open, so a
    // close that does not tear down the transport leaves the session silently
    // detached: it keeps accepting input and never receives again.
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()
    harness.disconnect.mockClear()

    harness.getProviderOptions()?.events.onClose(1_000, 'invalid_lease')

    expect(harness.disconnect).toHaveBeenCalled()
    expect(harness.controller.getSnapshot()).toMatchObject({
      connectionStatus: 'reconnecting',
    })
  })

  it('reports a rejected page origin as a transport fault, not as revoked access', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()

    harness.getProviderOptions()?.events.onClose(4_403, 'origin_rejected')

    const snapshot = harness.controller.getSnapshot()
    // Its own terminal state, not 'incompatible': both stop the session, but
    // one tells the operator to update the client and the other to fix an
    // address. Reusing the incompatible state produced the wrong advice.
    expect(snapshot).toMatchObject({
      access: 'edit',
      canEdit: false,
      connectionStatus: 'origin_rejected',
    })
    expect(snapshot.error).toContain('origin')
  })

  it('keeps a suggestion-policy rejection separate from access revocation', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()

    harness.getProviderOptions()?.events.onClose(4403, 'suggestion_policy_violation')

    expect(harness.controller.getSnapshot()).toMatchObject({
      access: 'edit',
      canEdit: false,
      connectionStatus: 'error',
      durabilityStatus: 'error',
      recoverability: 'none',
    })
    expect(harness.controller.getSnapshot().error).toContain(
      'proposed editor action was rejected',
    )
  })

  it('reissues a revoked lease and recovers a live permission downgrade as read-only', async () => {
    const invalidated = Object.assign(new Error('lease invalidated'), {
      detail: { reason: 'lease_revoked' },
      status: 401,
    })
    const downgradedSession = {
      ...session('view-token'),
      access: 'view' as const,
      initial_write_mode: 'view' as const,
    }
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockRejectedValueOnce(invalidated)
      .mockResolvedValueOnce(downgradedSession)
    const harness = createHarness(requestSession)
    await harness.controller.start()

    harness.getProviderOptions()?.events.onClose(4403, 'access_revoked')

    expect(harness.controller.getSnapshot()).toMatchObject({
      access: 'edit',
      canEdit: false,
      connectionStatus: 'reconnecting',
      recoverability: 'retry',
    })

    harness.runTimer(1_000)
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    expect(requestSession.mock.calls).toEqual([
      [],
      ['initial-token', harness.rotationCommandIds[0]],
      [],
    ])
    expect(harness.controller.getSnapshot()).toMatchObject({
      access: 'view',
      canEdit: false,
      connectionStatus: 'read_only',
      lifecycleStatus: 'read_only',
      recoverability: 'none',
      synced: true,
    })
    expect(harness.connect).toHaveBeenCalledTimes(2)
  })

  it('enters terminal access-revoked state only after a fresh lease is denied', async () => {
    const invalidated = Object.assign(new Error('lease invalidated'), {
      detail: { reason: 'lease_revoked' },
      status: 401,
    })
    const denied = Object.assign(new Error('resource unavailable'), {
      status: 404,
    })
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockRejectedValueOnce(invalidated)
      .mockRejectedValueOnce(denied)
    const harness = createHarness(requestSession)
    await harness.controller.start()

    harness.getProviderOptions()?.events.onClose(4403, 'access_revoked')
    harness.runTimer(1_000)
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    expect(requestSession.mock.calls).toEqual([
      [],
      ['initial-token', harness.rotationCommandIds[0]],
      [],
    ])
    expect(harness.controller.getSnapshot()).toMatchObject({
      access: 'view',
      canEdit: false,
      connectionStatus: 'access_revoked',
      lifecycleStatus: 'error',
      recoverability: 'none',
      synced: false,
    })
  })

  it('does not replay unconfirmed writes after access changes', async () => {
    const requestSession = vi.fn().mockResolvedValue(session('initial-token'))
    const harness = createHarness(requestSession)
    await harness.controller.start()
    harness.controller.getSnapshot().document?.getMap('test').set(
      'title',
      'Unconfirmed draft',
    )
    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    harness.getProviderOptions()?.events.onClose(4403, 'access_revoked')

    expect(requestSession).toHaveBeenCalledOnce()
    expect(harness.connect).toHaveBeenCalledOnce()
    expect(harness.replayedUpdates).toHaveLength(0)
    expect(harness.scheduledDelays()).not.toContain(1_000)
    expect(harness.controller.getSnapshot()).toMatchObject({
      canEdit: false,
      connectionStatus: 'error',
      durabilityStatus: 'error',
      hasUnconfirmedLocalChanges: true,
      lifecycleStatus: 'error',
      recoverability: 'none',
      synced: false,
    })
    expect(harness.controller.getSnapshot().error).toContain(
      'local document state is retained',
    )
  })

  it('rejects a periodic write-to-view lease while local writes are unconfirmed', async () => {
    const downgradedSession = {
      ...session('view-token'),
      access: 'view' as const,
      initial_write_mode: 'view' as const,
    }
    const requestSession = vi.fn()
      .mockResolvedValueOnce(session('initial-token'))
      .mockResolvedValueOnce(downgradedSession)
    const harness = createHarness(requestSession)
    await harness.controller.start()
    harness.controller.getSnapshot().document?.getMap('test').set(
      'title',
      'Unconfirmed periodic draft',
    )
    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    await harness.controller.refreshLease()

    expect(requestSession).toHaveBeenCalledTimes(2)
    expect(harness.connect).toHaveBeenCalledOnce()
    expect(harness.syncToken).not.toHaveBeenCalled()
    expect(harness.replayedUpdates).toHaveLength(0)
    expect(harness.controller.getSnapshot()).toMatchObject({
      connectionStatus: 'error',
      durabilityStatus: 'error',
      hasUnconfirmedLocalChanges: true,
    })
  })

  it('coalesces authentication-failed and close callbacks into one access revalidation', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()

    harness.getProviderOptions()?.events.onClose(4403, 'access_revoked')
    harness.getProviderOptions()?.events.onAuthenticationFailed('access_revoked')

    expect(
      harness.scheduledDelays().filter((delay) => delay === 1_000),
    ).toHaveLength(1)
    expect(harness.controller.getSnapshot()).toMatchObject({
      connectionStatus: 'reconnecting',
      reconnectAttempt: 1,
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

  it('flushes before a structure suggestion so the server can inspect the slash token', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()
    const editorDocument = harness.controller.getSnapshot().document
    expect(editorDocument).toBeDefined()

    const root = editorDocument!.getXmlFragment('content')
    const paragraph = new Y.XmlElement('paragraph')
    root.insert(0, [paragraph])
    expect(harness.transportUpdates).toHaveLength(0)

    paragraph.setAttribute(INQTRIX_STRUCTURE_SUGGESTION_ATTR, 'structure')

    expect(harness.transportUpdates).toHaveLength(1)
    expect(containsStructureSuggestionAttribute(harness.transportUpdates[0])).toBe(false)

    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    expect(harness.transportUpdates).toHaveLength(2)
    expect(containsStructureSuggestionAttribute(harness.transportUpdates[1])).toBe(true)
  })

  it('flushes pending direct normalization before the first suggestion mark', async () => {
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    await harness.controller.start()
    const editorDocument = harness.controller.getSnapshot().document
    expect(editorDocument).toBeDefined()

    const root = editorDocument!.getXmlFragment('content')
    const paragraph = new Y.XmlElement('paragraph')
    const text = new Y.XmlText()
    root.insert(0, [paragraph])
    paragraph.insert(0, [text])
    expect(harness.transportUpdates).toHaveLength(0)

    text.insert(0, '/', {
      insertion: {
        authorId: '11111111-1111-4111-8111-111111111111',
        createdAt: NOW_MS,
        kind: 'insertion',
        patchId: '22222222-2222-4222-8222-222222222222',
        suggestionId: '33333333-3333-4333-8333-333333333333',
      },
    })

    expect(harness.transportUpdates).toHaveLength(1)
    expect(updateSuggestionFormatKeys(harness.transportUpdates[0])).toEqual([])

    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    expect(harness.transportUpdates).toHaveLength(2)
    expect(updateSuggestionFormatKeys(harness.transportUpdates[1])).toEqual([
      'insertion',
      'insertion',
    ])
  })

  it('recognizes a repeated structure suggestion when Yjs infers the replaced attribute', () => {
    const document = new Y.Doc()
    const root = document.getXmlFragment('content')
    const paragraph = new Y.XmlElement('paragraph')
    root.insert(0, [paragraph])
    paragraph.setAttribute(INQTRIX_STRUCTURE_SUGGESTION_ATTR, {
      action: 'heading1',
      authorId: '11111111-1111-4111-8111-111111111111',
      createdAt: NOW_MS,
      kind: 'structure',
      patchId: '22222222-2222-4222-8222-222222222222',
      suggestionId: '33333333-3333-4333-8333-333333333333',
    } as never)
    const vector = Y.encodeStateVector(document)

    paragraph.setAttribute(INQTRIX_STRUCTURE_SUGGESTION_ATTR, {
      action: 'paragraph',
      authorId: '11111111-1111-4111-8111-111111111111',
      createdAt: NOW_MS + 1,
      kind: 'structure',
      patchId: '44444444-4444-4444-8444-444444444444',
      suggestionId: '55555555-5555-4555-8555-555555555555',
    } as never)
    const update = Y.encodeStateAsUpdate(document, vector)
    const replacement = Y.decodeUpdate(update).structs.find((struct) => (
      struct instanceof Y.Item
      && struct.content instanceof Y.ContentAny
    ))

    expect(replacement).toMatchObject({ parentSub: null })
    expect(containsStructureSuggestionAttribute(update)).toBe(true)
    document.destroy()
  })

  it('does not retain a phantom hash when Yjs removes an already transported update', async () => {
    const hashUpdate = vi.fn(async (update: Uint8Array) => {
      void update
      return 'must-not-be-called'
    })
    const harness = createHarness(
      vi.fn().mockResolvedValue(session('initial-token')),
      hashUpdate,
    )
    await harness.controller.start()
    const editorDocument = harness.controller.getSnapshot().document
    const transportDocument = harness.getProviderOptions()?.document
    expect(editorDocument).toBeDefined()
    expect(transportDocument).toBeDefined()

    const remoteDocument = new Y.Doc()
    remoteDocument.getMap('test').set('remote-title', 'Already transported')
    const remoteUpdate = Y.encodeStateAsUpdate(remoteDocument)

    // Model the multi-user race directly: the transport has already observed
    // a remote struct while the editor-side batch still contains that struct.
    Y.applyUpdate(transportDocument!, remoteUpdate, { source: 'remote-seed' })
    harness.transportUpdates.length = 0
    Y.applyUpdate(editorDocument!, remoteUpdate, { source: 'editor-replay' })

    const flushed = harness.controller.flushAndAwaitDurability()
    await flushed

    expect(harness.transportUpdates).toHaveLength(0)
    expect(hashUpdate).not.toHaveBeenCalled()
    expect(harness.controller.getSnapshot()).toMatchObject({
      durabilityStatus: 'saved',
      hasUnconfirmedLocalChanges: false,
      pendingHashes: [],
    })
    remoteDocument.destroy()
  })

  it('hashes the Yjs-deduplicated transport payload after concurrent state merge', async () => {
    const hashUpdate = vi.fn(async (update: Uint8Array) => {
      void update
      return 'deduplicated-update-hash'
    })
    const harness = createHarness(
      vi.fn().mockResolvedValue(session('initial-token')),
      hashUpdate,
    )
    await harness.controller.start()
    const editorDocument = harness.controller.getSnapshot().document
    const transportDocument = harness.getProviderOptions()?.document
    expect(editorDocument).toBeDefined()
    expect(transportDocument).toBeDefined()

    const remoteDocument = new Y.Doc()
    remoteDocument.getMap('test').set('remote-title', 'Remote')
    const remoteUpdate = Y.encodeStateAsUpdate(remoteDocument)
    Y.applyUpdate(transportDocument!, remoteUpdate, { source: 'remote-seed' })
    harness.transportUpdates.length = 0

    const combinedDocument = new Y.Doc()
    Y.applyUpdate(combinedDocument, remoteUpdate)
    combinedDocument.getMap('test').set('local-title', 'Local')
    Y.applyUpdate(
      editorDocument!,
      Y.encodeStateAsUpdate(combinedDocument),
      { source: 'combined-editor-update' },
    )
    harness.runTimer(COLLABORATION_UPDATE_BATCH_MS)
    await Promise.resolve()
    await Promise.resolve()

    expect(harness.transportUpdates).toHaveLength(1)
    expect(hashUpdate).toHaveBeenCalledOnce()
    expect([...hashUpdate.mock.calls[0][0]]).toEqual([
      ...harness.transportUpdates[0],
    ])
    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      hash: 'deduplicated-update-hash',
      sequence: 12,
      type: 'durable_ack',
    }))
    expect(harness.controller.getSnapshot()).toMatchObject({
      durabilityStatus: 'saved',
      hasUnconfirmedLocalChanges: false,
      lastLocalDurableSequence: 12,
      pendingHashes: [],
    })
    combinedDocument.destroy()
    remoteDocument.destroy()
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

  it('publishes shared-comment and mention invalidations independently', async () => {
    const harness = createHarness(
      vi.fn().mockResolvedValue(session('initial-token')),
    )
    await harness.controller.start()

    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      document_id: 'document-1',
      type: 'collaboration_comment_changed',
    }))
    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      document_id: 'document-1',
      type: 'collaboration_comment_mentioned',
    }))
    harness.getProviderOptions()?.events.onStateless(JSON.stringify({
      document_id: 'another-document',
      type: 'collaboration_comment_changed',
    }))

    expect(harness.controller.getSnapshot()).toMatchObject({
      commentEventVersion: 2,
      commentMentionEventVersion: 1,
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
      hasUnconfirmedLocalChanges: true,
      pendingHashes: ['local-update-hash'],
    })

    harness.runTimer(1_000)
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
      hasUnconfirmedLocalChanges: false,
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

  it('disposes an exact retired lifecycle after its recovery snapshot is captured', async () => {
    const key = 'workspace-1:document-1:g2'
    const harness = createHarness(vi.fn().mockResolvedValue(session('initial-token')))
    const controller = acquireLifecycleController(key, () => harness.controller)
    await controller.start()
    controller.getSnapshot().document?.getMap('test').set('title', 'Unconfirmed')
    expect(collaborationDocumentLifecycleHasUnconfirmedChanges({
      documentId: 'document-1',
      generation: 2,
      workspaceId: 'workspace-1',
    })).toBe(true)

    expect(retireCollaborationDocumentLifecycle({
      documentId: 'document-1',
      generation: 2,
      workspaceId: 'workspace-1',
    })).toBe(true)
    expect(harness.destroy).toHaveBeenCalledOnce()
    expect(collaborationDocumentLifecycleHasUnconfirmedChanges({
      documentId: 'document-1',
      generation: 2,
      workspaceId: 'workspace-1',
    })).toBe(false)

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
        random: () => 0.5,
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
  it('uses exponential reconnect backoff with bounded jitter', () => {
    expect([
      1, 2, 3, 4, 5, 6, 7,
    ].map((attempt) => collaborationReconnectDelayMs(attempt, 0.5))).toEqual([
      1_000,
      2_000,
      4_000,
      8_000,
      15_000,
      30_000,
      30_000,
    ])
    expect(collaborationReconnectDelayMs(1, 0)).toBe(850)
    expect(collaborationReconnectDelayMs(1, 1)).toBe(1_150)
    expect(collaborationReconnectDelayMs(6, 1)).toBe(30_000)
  })

  it('refreshes halfway through the lease and keeps websocket auth same-origin', () => {
    expect(leaseRefreshDelayMs(1_060, NOW_MS)).toBe(30_000)
    expect(collaborationWebSocketUrl('collaboration', {
      host: 'desk.test:5173',
      protocol: 'http:',
    })).toBe('ws://desk.test:5173/collaboration')
  })

  it('throttles OWN awareness state and never echoes received server states', () => {
    // Hocuspocus 4.3 re-broadcasts every changed awareness client — even
    // states just received from the server. The sidecar's identity gate
    // closes such connections with 4403 (invalid_request) as soon as a
    // second participant or stale clientID appears, so the adapter must
    // filter to the local clientID.
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
    vi.useFakeTimers()
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
      // An echoed state for our own clientID is still a server message and
      // must not be sent back.
      awareness!.emit('update', [
        { added: [], removed: [], updated: [document.clientID] },
        adapter.provider,
      ])
      expect(forwarded).toHaveLength(0)
      forwarded.length = 0
      // A burst of local cursor changes is coalesced below the sidecar's
      // awareness rate limit while preserving the latest state.
      for (let cursor = 0; cursor < 25; cursor += 1) {
        awareness!.setLocalStateField('cursor', cursor)
      }
      expect(forwarded).toHaveLength(0)
      vi.advanceTimersByTime(COLLABORATION_AWARENESS_THROTTLE_MS)
      expect(forwarded).toHaveLength(1)
      expect(forwarded[0].updated).toEqual([document.clientID])
      expect(awareness!.getLocalState()?.cursor).toBe(24)
    } finally {
      adapter.destroy()
      document.destroy()
      vi.useRealTimers()
    }
  })

  it('attaches the real provider to its external websocket transport', () => {
    // Hocuspocus 4.x only auto-attaches when it manages its own socket;
    // with the adapter's external websocketProvider a missing attach()
    // means the auth token is NEVER sent and every participant stays
    // read-only. The controller tests replace this factory with fakes, so
    // this regression test drives the real provider.
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
