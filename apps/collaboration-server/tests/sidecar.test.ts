import WebSocket from 'ws'
import { getSchema } from '@tiptap/core'
import { EditorState } from '@tiptap/pm/state'
import { initProseMirrorDoc, updateYFragment } from '@tiptap/y-tiptap'
import {
  EDITOR_YJS_FRAGMENT,
  createEditorSchemaExtensions,
  editorCollaborationRoom,
  parseEditorMarkdown,
  transformToInqtrixSuggestionTransaction,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import type { IntrospectedLease } from '../src/contracts'
import { hashBytes } from '../src/documentState'
import { CollaborationSidecar } from '../src/sidecar'
import {
  FakeCollaborationApi,
  USER_ID,
  deferred,
  documentState,
  markdownDocument,
  settings,
  silentLogger,
} from './helpers'

const DOCUMENT_ID = 'ed_test'
const ROOM = editorCollaborationRoom(DOCUMENT_ID, 1)
const editorSchema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))

describe('private sidecar listener', () => {
  it('disconnects a non-draining receiver without penalizing concurrent draining broadcasts', () => {
    const configured = settings({ maxQueuedFrames: 2 })
    const sidecar = new CollaborationSidecar(configured, {
      api: new FakeCollaborationApi(),
      logger: silentLogger,
    })
    const harness = sidecar as unknown as OutboundHarness
    const slow = fakeOutboundSocket(false)
    const draining = fakeOutboundSocket(true)
    const payload = new Uint8Array(8)

    for (let index = 0; index < 3; index += 1) {
      harness.sendHocuspocusPayload('slow', slow.socket, payload)
      harness.sendHocuspocusPayload('draining', draining.socket, payload)
    }

    expect(slow.close).toHaveBeenCalledWith(4429, 'rate_limited')
    expect(draining.close).not.toHaveBeenCalled()
    expect(draining.send).toHaveBeenCalledTimes(3)
  })

  it('keeps held broadcasts charged through a flush until send callbacks drain', () => {
    const configured = settings({ maxQueuedFrames: 2 })
    const sidecar = new CollaborationSidecar(configured, {
      api: new FakeCollaborationApi(),
      logger: silentLogger,
    })
    const harness = sidecar as unknown as OutboundHarness
    const receiver = fakeOutboundSocket(false)
    const payload = protocolFrame(ROOM, 0, new Uint8Array([0]))
    let blocked = true
    harness.coordinator.isBroadcastBlocked = () => blocked

    harness.sendHocuspocusPayload('held', receiver.socket, payload)
    harness.sendHocuspocusPayload('held', receiver.socket, payload)
    expect(receiver.send).not.toHaveBeenCalled()

    blocked = false
    harness.flushHeldOutbound(ROOM)
    expect(receiver.send).toHaveBeenCalledTimes(2)
    harness.sendHocuspocusPayload('held', receiver.socket, payload)

    expect(receiver.close).toHaveBeenCalledWith(4429, 'rate_limited')
  })

  it('rejects direct WebSocket access without the gateway bearer secret', async () => {
    const configured = settings({ policyPollMs: 60_000 })
    const sidecar = new CollaborationSidecar(configured, {
      api: new FakeCollaborationApi(),
      logger: silentLogger,
    })
    await sidecar.start()
    const port = sidecar.address?.port
    expect(port).toBeTypeOf('number')

    const status = await rejectedUpgrade(`ws://127.0.0.1:${port}/collaboration`)
    expect(status).toBe(401)
    await sidecar.stop()
  })

  it('exposes readiness, metrics, and secret-protected internal conversion', async () => {
    const configured = settings({ policyPollMs: 60_000 })
    const sidecar = new CollaborationSidecar(configured, {
      api: new FakeCollaborationApi(),
      logger: silentLogger,
    })
    await sidecar.start()
    const base = `http://127.0.0.1:${sidecar.address?.port}`

    const ready = await fetch(`${base}/health/ready`)
    expect(ready.status).toBe(200)
    await expect(ready.json()).resolves.toMatchObject({
      mode: 'single_replica',
      status: 'ready',
    })
    const metrics = await fetch(`${base}/metrics`)
    expect(metrics.status).toBe(200)
    expect(await metrics.text()).toContain('inqtrix_collaboration_instance_ready 1')

    const unauthorized = await fetch(`${base}/internal/convert`, {
      body: JSON.stringify({}),
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
    })
    expect(unauthorized.status).toBe(401)

    const converted = await fetch(`${base}/internal/convert`, {
      body: JSON.stringify({
        document_id: 'ed_test',
        markdown: '# Shared',
        max_document_bytes: configured.documentLimitBytes,
        schema_version: configured.schemaVersion,
      }),
      headers: {
        Authorization: `Bearer ${configured.secret}`,
        'Content-Type': 'application/json',
      },
      method: 'POST',
    })
    expect(converted.status).toBe(200)
    await expect(converted.json()).resolves.toMatchObject({
      projection_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      schema_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      snapshot: {
        state_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      },
    })
    await sidecar.stop()
  })

  it('schedules global maintenance without waiting for a snapshot', async () => {
    const api = new FakeCollaborationApi()
    const configured = settings({
      maintenanceIntervalMs: 10,
      policyPollMs: 60_000,
    })
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })

    await sidecar.start()
    await vi.waitFor(() => expect(api.compactions).not.toHaveLength(0))

    expect(api.snapshots).toHaveLength(0)
    expect(api.compactions[0]).toEqual({ fence: api.fence })
    await sidecar.stop()
  })

  it('closes an authenticated socket when its client lease expires without renewal', async () => {
    const api = new FakeCollaborationApi()
    const configured = settings({ policyPollMs: 60_000 })
    const document = markdownDocument('Lease protected')
    api.loadedState = await documentState(DOCUMENT_ID, document)
    document.destroy()
    api.lease = validLease(configured, api.loadedState.schemaHash, 0.25)
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()
    const socket = await openSocket(sidecar, configured.secret)
    const authenticated = nextMessage(socket)
    const closed = nextClose(socket)

    socket.send(authFrame(ROOM, 'lease-token'))
    await withTimeout(authenticated, 1_000, 'authentication response')

    await expect(withTimeout(closed, 1_000, 'lease close')).resolves.toMatchObject({
      code: 4401,
      reason: 'invalid_lease',
    })
    await sidecar.stop()
  })

  it('allows normal durability reconciliation within its connection budget', async () => {
    const api = new FakeCollaborationApi()
    const configured = settings({
      policyPollMs: 60_000,
      reconcileRateLimit: 2,
      reconcileRateWindowMs: 60_000,
    })
    const document = markdownDocument('Reconcile')
    api.loadedState = await documentState(DOCUMENT_ID, document)
    document.destroy()
    api.lease = validLease(configured, api.loadedState.schemaHash, 60)
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()
    const socket = await openSocket(sidecar, configured.secret)
    const authenticated = nextMessage(socket)
    socket.send(authFrame(ROOM, 'lease-token'))
    await withTimeout(authenticated, 1_000, 'reconcile authentication')

    socket.send(reconcileFrame(ROOM, 'a'.repeat(64)))
    await vi.waitFor(() => expect(api.lookups).toHaveLength(1))
    socket.send(reconcileFrame(ROOM, 'b'.repeat(64)))
    await vi.waitFor(() => expect(api.lookups).toHaveLength(2))
    expect(socket.readyState).toBe(WebSocket.OPEN)

    const closed = nextClose(socket)
    socket.close()
    await withTimeout(closed, 1_000, 'reconcile disconnect')
    await sidecar.stop()
  })

  it('closes a durability reconciliation burst with visible rate-limit semantics', async () => {
    const api = new FakeCollaborationApi()
    const configured = settings({
      policyPollMs: 60_000,
      reconcileRateLimit: 1,
      reconcileRateWindowMs: 60_000,
    })
    const document = markdownDocument('Reconcile burst')
    api.loadedState = await documentState(DOCUMENT_ID, document)
    document.destroy()
    api.lease = validLease(configured, api.loadedState.schemaHash, 60)
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()
    const socket = await openSocket(sidecar, configured.secret)
    const authenticated = nextMessage(socket)
    const closed = nextClose(socket)
    socket.send(authFrame(ROOM, 'lease-token'))
    await withTimeout(authenticated, 1_000, 'reconcile-burst authentication')

    socket.send(reconcileFrame(ROOM, 'a'.repeat(64)))
    await vi.waitFor(() => expect(api.lookups).toHaveLength(1))
    socket.send(reconcileFrame(ROOM, 'b'.repeat(64)))

    await expect(withTimeout(closed, 1_000, 'reconcile rate-limit close'))
      .resolves.toEqual({ code: 4429, reason: 'rate_limited' })
    expect(api.lookups).toHaveLength(1)
    await sidecar.stop()
  })

  it('closes a suggest socket before persisting transient Yjs history', async () => {
    const api = new FakeCollaborationApi()
    const configured = settings({ policyPollMs: 60_000 })
    const document = markdownDocument('First\n\nSecond')
    api.loadedState = await documentState(DOCUMENT_ID, document)
    api.lease = {
      ...validLease(configured, api.loadedState.schemaHash, 60),
      permission: 'suggest',
    }
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()
    const socket = await openSocket(sidecar, configured.secret)
    const authenticated = nextMessage(socket)
    socket.send(authFrame(ROOM, 'lease-token'))
    await withTimeout(authenticated, 1_000, 'transient-history authentication')
    const synchronized = nextMessage(socket)
    socket.send(syncFrame(ROOM, 0, Y.encodeStateVector(document)))
    await withTimeout(synchronized, 1_000, 'transient-history synchronization')

    const closed = nextClose(socket)
    socket.send(syncFrame(ROOM, 2, suggestionWithTransientTextUpdate(document)))

    await expect(withTimeout(closed, 1_000, 'transient-history rejection'))
      .resolves.toEqual({ code: 4403, reason: 'suggestion_policy_violation' })
    expect(api.persisted).toHaveLength(0)

    document.destroy()
    await sidecar.stop()
  })

  it('rejects a burst while authentication is slow before Hocuspocus can queue it', async () => {
    const api = new FakeCollaborationApi()
    const configured = settings({
      maxQueuedFrames: 2,
      policyPollMs: 60_000,
    })
    const document = markdownDocument('Queued')
    api.loadedState = await documentState(DOCUMENT_ID, document)
    document.destroy()
    const introspection = deferred<IntrospectedLease>()
    let introspections = 0
    api.introspectLease = async () => {
      introspections += 1
      return introspection.promise
    }
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()
    const socket = await openSocket(sidecar, configured.secret)
    const closed = nextClose(socket)

    socket.send(authFrame(ROOM, 'slow-token'))
    await vi.waitFor(() => expect(introspections).toBe(1))
    socket.send(protocolFrame(ROOM, 0, new Uint8Array([0])))
    socket.send(protocolFrame(ROOM, 0, new Uint8Array([0])))

    await expect(closed).resolves.toMatchObject({
      code: 4429,
      reason: 'rate_limited',
    })
    introspection.resolve(validLease(configured, api.loadedState.schemaHash, 60))
    await sidecar.stop()
  })

  it('rejects queued bytes before copying a slow-authentication frame', async () => {
    const api = new FakeCollaborationApi()
    const auth = authFrame(ROOM, 'slow-token')
    const queued = protocolFrame(ROOM, 0, new Uint8Array(64))
    const configured = settings({
      maxQueuedBytes: auth.byteLength + queued.byteLength - 1,
      maxQueuedFrames: 8,
      policyPollMs: 60_000,
    })
    const document = markdownDocument('Queued bytes')
    api.loadedState = await documentState(DOCUMENT_ID, document)
    document.destroy()
    const introspection = deferred<IntrospectedLease>()
    let introspections = 0
    api.introspectLease = async () => {
      introspections += 1
      return introspection.promise
    }
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()
    const socket = await openSocket(sidecar, configured.secret)
    const closed = nextClose(socket)

    socket.send(auth)
    await vi.waitFor(() => expect(introspections).toBe(1))
    socket.send(queued)

    await expect(closed).resolves.toMatchObject({
      code: 4429,
      reason: 'rate_limited',
    })
    introspection.resolve(validLease(configured, api.loadedState.schemaHash, 60))
    await sidecar.stop()
  })

  it('quarantines a disconnected commit until a verified reload permits reconnect', async () => {
    const api = new FakeCollaborationApi()
    const configured = settings({
      policyPollMs: 60_000,
      snapshotIdleMs: 50,
      snapshotMaxUpdates: 1,
    })
    const document = markdownDocument('Hello')
    api.loadedState = await documentState(DOCUMENT_ID, document)
    api.lease = validLease(configured, api.loadedState.schemaHash, 60)
    const persistence = deferred<{
      duplicate: boolean
      persistedSequence: number
      sequence: number
    }>()
    api.persistImplementation = () => persistence.promise
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()

    const first = await openSocket(sidecar, configured.secret)
    const authenticated = nextMessage(first)
    first.send(authFrame(ROOM, 'lease-token'))
    await withTimeout(authenticated, 1_000, 'first authentication')
    const synchronized = nextMessage(first)
    first.send(syncFrame(ROOM, 0, Y.encodeStateVector(document)))
    await withTimeout(synchronized, 1_000, 'initial synchronization')

    first.send(syncFrame(ROOM, 2, editorUpdate(document, 'Hello!')))
    await vi.waitFor(() => expect(api.persisted).toHaveLength(1))
    const firstClosed = nextClose(first)
    first.close()
    await withTimeout(firstClosed, 1_000, 'first disconnect')

    const fastReconnect = await openSocket(sidecar, configured.secret)
    const rejected = nextClose(fastReconnect)
    fastReconnect.send(authFrame(ROOM, 'lease-token'))
    await expect(withTimeout(rejected, 1_000, 'quarantine rejection')).resolves.toMatchObject({
      code: 1012,
      reason: 'restarting',
    })
    expect(api.snapshots).toHaveLength(0)

    persistence.resolve({ duplicate: false, persistedSequence: 1, sequence: 1 })
    await vi.waitFor(() => expect(api.loadedState?.persistedSequence).toBe(1))
    await vi.waitFor(() => expect(
      (sidecar as unknown as {
        hocuspocus: { documents: Map<string, unknown> }
      }).hocuspocus.documents.has(ROOM),
    ).toBe(false))

    const recovered = await openSocket(sidecar, configured.secret)
    const recoveredAuthentication = nextMessage(recovered)
    recovered.send(authFrame(ROOM, 'lease-token'))
    await withTimeout(recoveredAuthentication, 1_000, 'recovered authentication')
    expect(api.loads.length).toBeGreaterThanOrEqual(2)
    await vi.waitFor(() => expect(api.snapshots).toHaveLength(1))

    const recoveredClosed = nextClose(recovered)
    recovered.close()
    await withTimeout(recoveredClosed, 1_000, 'recovered disconnect')
    document.destroy()
    await sidecar.stop()
  })

  it.each([
    ['update-count', { snapshotMaxUpdates: 1, snapshotTailBytes: 100 * 1024 * 1024 }],
    ['tail-bytes', { snapshotMaxUpdates: 256, snapshotTailBytes: 1 }],
  ] as const)('snapshots immediately when the %s threshold crosses after durable apply', async (
    _label,
    threshold,
  ) => {
    const api = new FakeCollaborationApi()
    const configured = settings({
      ...threshold,
      policyPollMs: 60_000,
      snapshotIdleMs: 60_000,
    })
    const document = markdownDocument('Hello')
    api.loadedState = await documentState(DOCUMENT_ID, document)
    api.lease = validLease(configured, api.loadedState.schemaHash, 60)
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()
    const socket = await openSocket(sidecar, configured.secret)
    const authenticated = nextMessage(socket)
    socket.send(authFrame(ROOM, 'lease-token'))
    await withTimeout(authenticated, 1_000, `${_label} authentication`)
    const synchronized = nextMessage(socket)
    socket.send(syncFrame(ROOM, 0, Y.encodeStateVector(document)))
    await withTimeout(synchronized, 1_000, `${_label} synchronization`)

    socket.send(syncFrame(ROOM, 2, editorUpdate(document, 'Hello!')))
    await vi.waitFor(() => expect(api.snapshots).toHaveLength(1))
    expect(api.snapshots[0]?.coveredSequence).toBe(1)
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(api.snapshots).toHaveLength(1)

    const closed = nextClose(socket)
    socket.close()
    await withTimeout(closed, 1_000, `${_label} disconnect`)
    document.destroy()
    await sidecar.stop()
  })

  it('retries a failed loaded-tail snapshot autonomously while the room stays dirty', async () => {
    const api = new FakeCollaborationApi()
    const configured = settings({
      policyPollMs: 60_000,
      snapshotMaxUpdates: 1,
      snapshotRetryBaseMs: 10,
      snapshotRetryMaxMs: 20,
    })
    const document = markdownDocument('Hello')
    const tail = editorUpdate(document, 'Hello!')
    const loaded = await documentState(DOCUMENT_ID, document)
    api.loadedState = {
      ...loaded,
      persistedSequence: 1,
      updates: [{ hash: hashBytes(tail), sequence: 1, update: tail }],
    }
    api.lease = validLease(configured, api.loadedState.schemaHash, 60)
    let attempts = 0
    api.snapshotImplementation = async () => {
      attempts += 1
      if (attempts === 1) throw new Error('snapshot unavailable')
    }
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()
    const socket = await openSocket(sidecar, configured.secret)
    const authenticated = nextMessage(socket)

    socket.send(authFrame(ROOM, 'lease-token'))
    await withTimeout(authenticated, 1_000, 'snapshot retry authentication')
    await vi.waitFor(() => expect(api.snapshots).toHaveLength(2))
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(api.snapshots).toHaveLength(2)
    expect(api.snapshots[1]).toMatchObject({
      coveredSequence: 1,
      projectionMarkdown: expect.stringContaining('Hello!'),
    })

    const closed = nextClose(socket)
    socket.close()
    await withTimeout(closed, 1_000, 'snapshot retry disconnect')
    document.destroy()
    await sidecar.stop()
  })

  it('stores an immediate follow-up when an update races an in-flight snapshot', async () => {
    const api = new FakeCollaborationApi()
    const configured = settings({
      policyPollMs: 60_000,
      snapshotMaxUpdates: 1,
    })
    const base = markdownDocument('Hello')
    const firstUpdate = editorUpdate(base, 'Hello!')
    const current = new Y.Doc()
    Y.applyUpdate(current, Y.encodeStateAsUpdate(base))
    Y.applyUpdate(current, firstUpdate)
    const loaded = await documentState(DOCUMENT_ID, base)
    api.loadedState = {
      ...loaded,
      persistedSequence: 1,
      updates: [{ hash: hashBytes(firstUpdate), sequence: 1, update: firstUpdate }],
    }
    api.lease = validLease(configured, api.loadedState.schemaHash, 60)
    api.persistImplementation = async () => ({
      duplicate: false,
      persistedSequence: 2,
      sequence: 2,
    })
    const firstSnapshot = deferred<void>()
    let firstSnapshotPending = true
    api.snapshotImplementation = async (input) => {
      if (input.coveredSequence === 1 && firstSnapshotPending) {
        firstSnapshotPending = false
        await firstSnapshot.promise
      }
    }
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
    })
    await sidecar.start()
    const socket = await openSocket(sidecar, configured.secret)
    const authenticated = nextMessage(socket)
    socket.send(authFrame(ROOM, 'lease-token'))
    await withTimeout(authenticated, 1_000, 'snapshot-race authentication')
    await vi.waitFor(() => expect(api.snapshots).toHaveLength(1))

    const synchronized = nextMessage(socket)
    socket.send(syncFrame(ROOM, 0, Y.encodeStateVector(current)))
    await withTimeout(synchronized, 1_000, 'snapshot-race synchronization')
    socket.send(syncFrame(ROOM, 2, editorUpdate(current, 'Hello!!')))
    await vi.waitFor(() => expect(api.persisted).toHaveLength(1))

    firstSnapshot.resolve()
    await vi.waitFor(() => expect(api.snapshots).toHaveLength(2))
    expect(api.snapshots.map((snapshot) => snapshot.coveredSequence)).toEqual([1, 2])
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(api.snapshots).toHaveLength(2)

    const closed = nextClose(socket)
    socket.close()
    await withTimeout(closed, 1_000, 'snapshot-race disconnect')
    current.destroy()
    base.destroy()
    await sidecar.stop()
  })

  it('bounds shutdown with hung policy, maintenance, and socket teardown', async () => {
    const api = new FakeCollaborationApi()
    let policyCalls = 0
    api.compactImplementation = async () => await new Promise<never>(() => undefined)
    api.policyImplementation = async () => {
      policyCalls += 1
      return await new Promise<never>(() => undefined)
    }
    const configured = settings({
      maintenanceIntervalMs: 60_000,
      policyPollMs: 60_000,
    })
    const sidecar = new CollaborationSidecar(configured, {
      api,
      logger: silentLogger,
      shutdownTimeoutMs: 120,
      socketCloseGraceMs: 20,
    })
    await sidecar.start()
    const client = await openSocket(sidecar, configured.secret)
    const clientClosed = nextClose(client)
    await vi.waitFor(() => expect(api.compactions).toHaveLength(1))
    await vi.waitFor(() => expect(policyCalls).toBe(1))
    const serverSocket = [...(sidecar as unknown as ShutdownHarness).sockets.values()][0]
    if (!serverSocket) throw new Error('shutdown socket fixture is invalid')
    const close = vi.spyOn(serverSocket, 'close').mockImplementation(() => undefined)
    const terminate = vi.spyOn(serverSocket, 'terminate')

    const startedAt = performance.now()
    const stopping = sidecar.stop()
    expect(close).toHaveBeenCalledWith(1012, 'restarting')
    await withTimeout(stopping, 500, 'bounded shutdown')

    expect(terminate).toHaveBeenCalledTimes(1)
    expect(performance.now() - startedAt).toBeLessThan(500)
    await withTimeout(clientClosed, 500, 'forced client teardown')
  })

  it('aborts startup cleanly when stopped during instance lease acquisition', async () => {
    const api = new FakeCollaborationApi()
    const acquisition = deferred<typeof api.fence>()
    let acquisitions = 0
    api.acquireInstance = async () => {
      acquisitions += 1
      return acquisition.promise
    }
    const sidecar = new CollaborationSidecar(settings({ policyPollMs: 60_000 }), {
      api,
      logger: silentLogger,
      shutdownTimeoutMs: 500,
    })
    const starting = sidecar.start()
    await vi.waitFor(() => expect(acquisitions).toBe(1))

    const stopping = sidecar.stop()
    acquisition.resolve(api.fence)
    await withTimeout(Promise.all([starting, stopping]), 1_000, 'lease-startup shutdown')

    const harness = sidecar as unknown as StartupHarness
    expect(sidecar.address).toBeNull()
    expect(harness.leaseManager.isReady()).toBe(false)
    expect(harness.maintenanceTimer).toBeNull()
    expect(harness.policyTimer).toBeNull()
  })

  it('closes a delayed socket bind without ever becoming ready after stop', async () => {
    const sidecar = new CollaborationSidecar(settings({ policyPollMs: 60_000 }), {
      api: new FakeCollaborationApi(),
      logger: silentLogger,
      shutdownTimeoutMs: 500,
    })
    const harness = sidecar as unknown as StartupHarness
    const binding = deferred<void>()
    const entered = deferred<void>()
    const listen = harness.listenHttpServer.bind(sidecar)
    harness.listenHttpServer = async () => {
      entered.resolve()
      await binding.promise
      await listen()
    }
    const starting = sidecar.start()
    await entered.promise

    const stopping = sidecar.stop()
    binding.resolve()
    await withTimeout(Promise.all([starting, stopping]), 1_000, 'bind-startup shutdown')

    expect(sidecar.address).toBeNull()
    expect(harness.leaseManager.isReady()).toBe(false)
    expect(harness.maintenanceTimer).toBeNull()
    expect(harness.policyTimer).toBeNull()
  })
})

function validLease(
  configured: ReturnType<typeof settings>,
  schemaHash: string,
  lifetimeSeconds: number,
): IntrospectedLease {
  return {
    documentId: DOCUMENT_ID,
    expiresAt: Date.now() / 1_000 + lifetimeSeconds,
    generation: 1,
    leaseId: 'lease-1',
    permission: 'edit',
    protocolVersion: configured.protocolVersion,
    schemaHash,
    schemaVersion: configured.schemaVersion,
    sessionId: 'session-1',
    tenantId: configured.tenantId,
    user: { color: '#123456', id: USER_ID, name: 'Ada' },
  }
}

async function openSocket(
  sidecar: CollaborationSidecar,
  secret: string,
): Promise<WebSocket> {
  const socket = new WebSocket(
    `ws://127.0.0.1:${sidecar.address?.port}/collaboration`,
    { headers: { Authorization: `Bearer ${secret}` } },
  )
  await new Promise<void>((resolve, reject) => {
    socket.once('open', resolve)
    socket.once('error', reject)
  })
  return socket
}

function nextMessage(socket: WebSocket): Promise<void> {
  return new Promise((resolve, reject) => {
    socket.once('message', () => resolve())
    socket.once('error', reject)
  })
}

function nextClose(socket: WebSocket): Promise<{ code: number; reason: string }> {
  return new Promise((resolve) => {
    socket.once('close', (code, reason) => {
      resolve({ code, reason: reason.toString('utf8') })
    })
  })
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out`)), timeoutMs)
    void promise.then(
      (value) => {
        clearTimeout(timer)
        resolve(value)
      },
      (error: unknown) => {
        clearTimeout(timer)
        reject(error)
      },
    )
  })
}

function authFrame(room: string, token: string): Uint8Array {
  return protocolFrame(room, 2, concat(varUint(0), varString(token)))
}

function syncFrame(room: string, type: number, update: Uint8Array): Uint8Array {
  return protocolFrame(room, 0, concat(varUint(type), varBytes(update)))
}

function reconcileFrame(room: string, hash: string): Uint8Array {
  return protocolFrame(room, 5, varString(JSON.stringify({
    hashes: [hash],
    type: 'durability_reconcile',
  })))
}

function varBytes(value: Uint8Array): Uint8Array {
  return concat(varUint(value.byteLength), value)
}

function editorUpdate(document: Y.Doc, targetMarkdown: string): Uint8Array {
  const replica = new Y.Doc()
  Y.applyUpdate(replica, Y.encodeStateAsUpdate(document))
  const vector = Y.encodeStateVector(replica)
  const fragment = replica.getXmlFragment(EDITOR_YJS_FRAGMENT)
  const initialized = initProseMirrorDoc(fragment, editorSchema)
  updateYFragment(
    replica,
    fragment,
    editorSchema.nodeFromJSON(parseEditorMarkdown(targetMarkdown)),
    initialized.meta,
  )
  const update = Y.encodeStateAsUpdate(replica, vector)
  replica.destroy()
  return update
}

function suggestionWithTransientTextUpdate(document: Y.Doc): Uint8Array {
  const replica = new Y.Doc()
  Y.applyUpdate(replica, Y.encodeStateAsUpdate(document))
  const vector = Y.encodeStateVector(replica)
  const fragment = replica.getXmlFragment(EDITOR_YJS_FRAGMENT)
  const initialized = initProseMirrorDoc(fragment, editorSchema)
  let textPosition: number | null = null
  initialized.doc.descendants((node, position) => {
    if (textPosition === null && node.isText) textPosition = position
  })
  if (textPosition === null) throw new Error('Transient-history fixture has no text')
  const state = EditorState.create({ schema: editorSchema, doc: initialized.doc })
  const tracked = transformToInqtrixSuggestionTransaction(
    state.tr.insertText('!', textPosition + 1),
    state,
    {
      authorId: USER_ID,
      createdAt: 1_784_112_100,
      patchId: '22222222-2222-4222-8222-222222222222',
    },
    () => '33333333-3333-4333-8333-333333333333',
  )
  updateYFragment(replica, fragment, tracked.doc, initialized.meta)
  const paragraph = fragment.get(1)
  const text = paragraph instanceof Y.XmlElement ? paragraph.get(0) : null
  if (!(text instanceof Y.XmlText)) throw new Error('Transient-history target is invalid')
  const hidden = 'x'.repeat(64 * 1024)
  text.insert(1, hidden)
  text.delete(1, hidden.length)
  const update = Y.encodeStateAsUpdate(replica, vector)
  replica.destroy()
  return update
}

function protocolFrame(room: string, type: number, payload: Uint8Array): Uint8Array {
  return concat(varString(room), varUint(type), payload)
}

function varString(value: string): Uint8Array {
  const encoded = new TextEncoder().encode(value)
  return concat(varUint(encoded.byteLength), encoded)
}

function varUint(value: number): Uint8Array {
  const bytes: number[] = []
  let remaining = value
  do {
    let byte = remaining & 0x7f
    remaining = Math.floor(remaining / 128)
    if (remaining > 0) byte |= 0x80
    bytes.push(byte)
  } while (remaining > 0)
  return new Uint8Array(bytes)
}

function concat(...parts: Uint8Array[]): Uint8Array {
  const result = new Uint8Array(parts.reduce((total, part) => total + part.byteLength, 0))
  let offset = 0
  for (const part of parts) {
    result.set(part, offset)
    offset += part.byteLength
  }
  return result
}

function rejectedUpgrade(url: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url)
    socket.once('open', () => {
      socket.close()
      reject(new Error('WebSocket unexpectedly opened'))
    })
    socket.once('unexpected-response', (_request, response) => {
      response.resume()
      resolve(response.statusCode ?? 0)
    })
    socket.once('error', () => undefined)
  })
}

type OutboundHarness = {
  coordinator: { isBroadcastBlocked(room: string): boolean }
  flushHeldOutbound(room: string): void
  sendHocuspocusPayload(
    transportId: string,
    socket: WebSocket,
    payload: Uint8Array,
  ): void
}

type ShutdownHarness = {
  sockets: Map<string, WebSocket>
}

type StartupHarness = {
  leaseManager: { isReady(): boolean }
  listenHttpServer(): Promise<void>
  maintenanceTimer: ReturnType<typeof setInterval> | null
  policyTimer: ReturnType<typeof setInterval> | null
}

function fakeOutboundSocket(drainImmediately: boolean): {
  close: ReturnType<typeof vi.fn>
  send: ReturnType<typeof vi.fn>
  socket: WebSocket
} {
  const close = vi.fn()
  const send = vi.fn((
    _payload: unknown,
    callback: (error?: Error) => void,
  ) => {
    if (drainImmediately) callback()
  })
  return {
    close,
    send,
    socket: {
      bufferedAmount: 0,
      close,
      readyState: WebSocket.OPEN,
      send,
    } as unknown as WebSocket,
  }
}
