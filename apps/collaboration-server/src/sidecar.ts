import { randomUUID } from 'node:crypto'
import { createServer, type IncomingMessage, type Server as HttpServer } from 'node:http'
import type { AddressInfo } from 'node:net'
import type { Duplex } from 'node:stream'

import {
  Hocuspocus,
  type Connection,
  type Document as HocuspocusDocument,
  type WebSocketLike,
} from '@hocuspocus/server'
import {
  parseEditorCollaborationRoom,
} from '@inqtrix/editor-schema'
import WebSocket, {
  type RawData,
  WebSocketServer,
} from 'ws'
import * as Y from 'yjs'

import { FastApiCollaborationClient } from './apiClient'
import { CollaborationAuthenticator } from './authenticator'
import { enforceAwarenessIdentity } from './awareness'
import type {
  CollaborationApi,
  CollaborationPolicyEvent,
  CollaborationSettings,
  ConnectionContext,
  SidecarLogger,
} from './contracts'
import { DocumentCoordinator } from './documentCoordinator'
import { reconstructDocument } from './documentState'
import { reconcileDurability } from './durabilityReconciliation'
import {
  collaborationError,
  ApiRequestError,
  CloseCodes,
  CollaborationError,
} from './errors'
import { hasBearerSecret, InternalHttpRouter } from './httpRouter'
import { InstanceLeaseManager } from './instanceLease'
import { createJsonLogger } from './logger'
import { SidecarMetrics } from './metrics'
import {
  OutboundFlowController,
  type OutboundReservation,
} from './outboundFlow'
import {
  CollaborationOperations,
  type OperationDocumentAccess,
} from './operations'
import { SessionRegistry, SlidingWindowRateLimiter } from './rateLimit'
import { SnapshotRetryController } from './snapshotRetry'

type InternalContext = {
  internalOperation: true
}

type TransportContext = {
  transportId: string
}

type HocuspocusContext = Partial<ConnectionContext>
  & Partial<InternalContext>
  & Partial<TransportContext>

type HocuspocusPayload = Parameters<WebSocketLike['send']>[0]

type HeldOutboundFrame = {
  payload: HocuspocusPayload
  reservation: OutboundReservation
  socket: WebSocket
  transportId: string
}

type IngressReservation = {
  bytes: number
  released: boolean
  transportId: string
}

type RegisteredConnection = {
  connection: Connection<HocuspocusContext>
  documentName: string
  socketId: string
  transportId: string
}

type TransportTeardown = (code: number, reason: string) => void

export type CollaborationSidecarDependencies = {
  api?: CollaborationApi
  logger?: SidecarLogger
  metrics?: SidecarMetrics
  shutdownTimeoutMs?: number
  socketCloseGraceMs?: number
}

const DEFAULT_SHUTDOWN_TIMEOUT_MS = 25_000
const DEFAULT_SOCKET_CLOSE_GRACE_MS = 1_000

export class CollaborationSidecar {
  private readonly api: CollaborationApi
  private readonly authenticator: CollaborationAuthenticator
  private readonly awarenessLimiter: SlidingWindowRateLimiter
  private readonly coordinator: DocumentCoordinator
  private readonly hocuspocus: Hocuspocus<HocuspocusContext>
  private readonly httpServer: HttpServer
  private readonly heldOutbound = new Map<string, HeldOutboundFrame[]>()
  private readonly authenticatedAddresses = new Set<string>()
  private readonly ingressAuthReservations = new Map<string, IngressReservation[]>()
  private readonly ingressFrames = new WeakMap<Uint8Array, IngressReservation>()
  private readonly ingressUsage = new Map<string, { bytes: number; frames: number }>()
  private readonly leaseManager: InstanceLeaseManager
  private readonly connectionLeaseTimers = new Map<string, ReturnType<typeof setTimeout>>()
  private readonly connections = new Map<string, RegisteredConnection>()
  private readonly logger: SidecarLogger
  private maintenanceInFlight: Promise<void> | null = null
  private maintenanceTimer: ReturnType<typeof setInterval> | null = null
  private readonly metrics: SidecarMetrics
  private readonly operations: CollaborationOperations
  private readonly outboundFlow: OutboundFlowController<HocuspocusPayload>
  private readonly outboundRejected = new Set<string>()
  private policyCursor = 0
  private policyPollInFlight: Promise<void> | null = null
  private policyTimer: ReturnType<typeof setInterval> | null = null
  private readonly policyRevalidationTimers = new Map<
    string,
    ReturnType<typeof setTimeout>
  >()
  private readonly router: InternalHttpRouter
  private readonly reconcileLimiter: SlidingWindowRateLimiter
  private readonly sessions: SessionRegistry
  private readonly shutdownTimeoutMs: number
  private readonly snapshotTasks = new Map<string, Promise<void>>()
  private readonly snapshotRetries: SnapshotRetryController
  private readonly sockets = new Map<string, WebSocket>()
  private readonly socketCloseGraceMs: number
  private startPromise: Promise<void> | null = null
  private stopPromise: Promise<void> | null = null
  private readonly transportConnections = new Map<string, Set<string>>()
  private readonly transportTeardowns = new Map<string, TransportTeardown>()
  private readonly updateLimiter: SlidingWindowRateLimiter
  private readonly webSocketServer: WebSocketServer
  private started = false
  private stopping = false

  constructor(
    private readonly settings: CollaborationSettings,
    dependencies: CollaborationSidecarDependencies = {},
  ) {
    this.logger = dependencies.logger ?? createJsonLogger()
    this.metrics = dependencies.metrics ?? new SidecarMetrics()
    this.shutdownTimeoutMs = dependencies.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS
    this.socketCloseGraceMs = dependencies.socketCloseGraceMs ?? DEFAULT_SOCKET_CLOSE_GRACE_MS
    if (this.shutdownTimeoutMs < 1 || this.socketCloseGraceMs < 0) {
      throw new Error('Shutdown deadlines must be non-negative')
    }
    this.outboundFlow = new OutboundFlowController(
      settings.maxQueuedFrames,
      settings.maxQueuedBytes,
      settings.socketBackpressureBytes,
      WebSocket.OPEN,
      (bytes, frames) => {
        this.metrics.set('inqtrix_collaboration_outbound_queued_bytes', bytes)
        this.metrics.set('inqtrix_collaboration_outbound_queued_frames', frames)
      },
    )
    this.api = dependencies.api ?? new FastApiCollaborationClient(
      settings,
      this.logger,
      this.metrics,
    )
    this.updateLimiter = new SlidingWindowRateLimiter(
      settings.updateRateLimit,
      settings.updateRateWindowMs,
    )
    this.awarenessLimiter = new SlidingWindowRateLimiter(
      settings.awarenessRateLimit,
      settings.awarenessRateWindowMs,
    )
    this.reconcileLimiter = new SlidingWindowRateLimiter(
      settings.reconcileRateLimit,
      settings.reconcileRateWindowMs,
    )
    this.sessions = new SessionRegistry(settings.maxSessionsPerUserDocument)
    this.leaseManager = new InstanceLeaseManager(
      this.api,
      settings,
      this.logger,
      this.metrics,
      () => this.handleLeaseLoss(),
    )
    this.coordinator = new DocumentCoordinator(
      this.api,
      this.leaseManager,
      settings,
      this.logger,
      this.metrics,
      {
        onAuthoritativeApplyFailure: (room) => {
          this.discardHeldOutbound(room)
          const document = this.hocuspocus.documents.get(room)
          if (document) this.closeDocument(document, reloadRequired())
        },
        onAuthoritativeApplySuccess: (room) => this.flushHeldOutbound(room),
      },
    )
    this.authenticator = new CollaborationAuthenticator(
      this.api,
      this.leaseManager,
      settings,
    )
    this.hocuspocus = this.createHocuspocus()
    this.snapshotRetries = new SnapshotRetryController(
      settings.snapshotRetryBaseMs,
      settings.snapshotRetryMaxMs,
      {
        isEligible: (room) => (
          !this.stopping
          && this.hocuspocus.documents.has(room)
          && this.coordinator.hasUnsnapshottedUpdates(room)
        ),
        onFailure: (_room, error) => this.recordSnapshotFailure(error),
        onSuccess: (room) => {
          const document = this.hocuspocus.documents.get(room)
          if (document?.getConnectionsCount() === 0) {
            void this.hocuspocus.unloadDocument(document)
          }
        },
      },
    )
    const acquireDocument: OperationDocumentAccess = (room) => (
      this.acquireOperationDocument(room)
    )
    this.operations = new CollaborationOperations(
      this.api,
      this.coordinator,
      this.leaseManager,
      settings,
      acquireDocument,
      this.logger,
      this.metrics,
    )
    this.router = new InternalHttpRouter(
      settings,
      this.leaseManager,
      this.operations,
      this.metrics,
      this.logger,
      () => this.isReady(),
    )
    this.webSocketServer = new WebSocketServer({
      clientTracking: false,
      maxPayload: settings.frameLimitBytes,
      noServer: true,
      perMessageDeflate: false,
    })
    this.httpServer = createServer((request, response) => {
      void this.router.handle(request, response).catch(() => {
        this.logger.error('http_router_failed')
        if (!response.headersSent) {
          response.writeHead(500, {
            'Cache-Control': 'no-store',
            'Content-Type': 'application/json; charset=utf-8',
          })
        }
        response.end(JSON.stringify({ error: { reason: 'internal_consistency' } }))
      })
    })
    this.httpServer.on('upgrade', (request, socket, head) => {
      this.handleUpgrade(request, socket, head)
    })
    this.metrics.set('inqtrix_collaboration_active_connections', 0)
  }

  get address(): AddressInfo | null {
    const value = this.httpServer.address()
    return value && typeof value === 'object' ? value : null
  }

  start(): Promise<void> {
    if (this.started) return Promise.resolve()
    if (this.startPromise) return this.startPromise
    if (this.stopping) return Promise.reject(new Error('Collaboration sidecar is stopping'))
    this.stopping = false
    const starting = this.performStart().finally(() => {
      if (this.startPromise === starting) this.startPromise = null
    })
    this.startPromise = starting
    return starting
  }

  private async performStart(): Promise<void> {
    try {
      await this.leaseManager.start()
      if (this.stopping) return
      await this.listenHttpServer()
      if (this.stopping) return
      this.started = true
      this.startMaintenance()
      this.startPolicyPolling()
      this.logger.info('collaboration_sidecar_listening', {
        address: this.settings.bindAddress,
        port: this.address?.port ?? this.settings.port,
      })
    } finally {
      if (!this.started) {
        await this.closeHttpServer()
        await this.leaseManager.stop()
      }
    }
  }

  stop(): Promise<void> {
    if (this.stopPromise) return this.stopPromise
    this.stopping = true
    const starting = this.startPromise
    this.stopPromise = this.performStop(starting).finally(() => {
      this.stopPromise = null
    })
    return this.stopPromise
  }

  private async performStop(starting: Promise<void> | null): Promise<void> {
    const deadline = Date.now() + this.shutdownTimeoutMs
    this.snapshotRetries.cancelAll()
    const maintenance = this.stopMaintenance()
    const policyPolling = this.stopPolicyPolling()
    const startupLeaseStop = this.started ? null : this.leaseManager.stop()
    const closeHttp = this.closeHttpServer()
    void closeHttp.catch(() => undefined)
    this.closeAllSockets(CloseCodes.restarting, 'restarting')
    this.clearAllConnectionLeaseTimers()
    await this.awaitShutdownPhase('startup', starting, deadline)
    await this.teardownSockets(deadline)
    if (!this.started) {
      await this.awaitShutdownPhase('instance_fence', startupLeaseStop, deadline)
      await this.awaitShutdownPhase('http_server', this.closeHttpServer(), deadline)
      this.webSocketServer.close()
      this.logger.info('collaboration_sidecar_stopped')
      return
    }
    const gatesDrained = await this.awaitShutdownPhase(
      'document_gates',
      this.coordinator.awaitAll(),
      deadline,
    )
    if (gatesDrained) {
      await this.awaitShutdownPhase(
        'snapshots',
        this.flushSnapshotsForShutdown(),
        deadline,
      )
    }
    await this.awaitShutdownPhase(
      'background_requests',
      Promise.allSettled([maintenance, policyPolling]).then(() => undefined),
      deadline,
    )
    const leaseStop = this.leaseManager.stop()
    await this.awaitShutdownPhase('instance_fence', leaseStop, deadline)
    this.httpServer.closeAllConnections()
    await this.awaitShutdownPhase('http_server', closeHttp, deadline)
    this.webSocketServer.close()
    this.started = false
    this.logger.info('collaboration_sidecar_stopped')
  }

  private listenHttpServer(): Promise<void> {
    return new Promise((resolve, reject) => {
      const onError = (error: Error): void => {
        this.httpServer.off('listening', onListening)
        reject(error)
      }
      const onListening = (): void => {
        this.httpServer.off('error', onError)
        resolve()
      }
      this.httpServer.once('error', onError)
      this.httpServer.once('listening', onListening)
      this.httpServer.listen(this.settings.port, this.settings.bindAddress)
    })
  }

  private closeHttpServer(): Promise<void> {
    if (!this.httpServer.listening) return Promise.resolve()
    return new Promise((resolve, reject) => {
      this.httpServer.close((error) => error ? reject(error) : resolve())
    })
  }

  private createHocuspocus(): Hocuspocus<HocuspocusContext> {
    return new Hocuspocus<HocuspocusContext>({
      debounce: this.settings.snapshotIdleMs,
      maxDebounce: Math.max(this.settings.snapshotIdleMs, 60_000),
      maxPendingDocuments: 8,
      maxUnauthenticatedQueueMessages: 64,
      maxUnauthenticatedQueueSize: this.settings.frameLimitBytes,
      name: 'inqtrix-collaboration',
      quiet: true,
      unloadImmediately: true,
      beforeHandleMessage: async ({ context, update }) => {
        this.releaseIngress(update)
        if (!this.isReady()) throw unavailable()
        if (!isInternalContext(context)) this.enforceConnectionLease(context)
        if (update.byteLength > this.settings.frameLimitBytes) throw tooLarge()
      },
      onConnect: async ({ context }) => {
        if (this.isReady()) return
        const error = unavailable()
        this.closeSocketAfterHook(context, error)
        throw error
      },
      onAuthenticate: async ({
        connectionConfig,
        context: hookContext,
        documentName,
        socketId,
        token,
      }) => {
        try {
          const context = await this.authenticator.authenticate(documentName, token)
          this.coordinator.assertJoinAllowed(documentName)
          if (!this.sessions.add(context.documentId, context.user.id, socketId)) {
            throw rateLimited()
          }
          connectionConfig.readOnly = context.access === 'view'
          return context
        } catch (error) {
          const mapped = collaborationError(error)
          this.closeSocketAfterHook(hookContext, mapped)
          throw mapped
        }
      },
      onTokenSync: async ({ connection, context, documentName, token }) => {
        try {
          const current = connectionContext(context)
          const renewed = await this.authenticator.renew(current, documentName, token)
          connection.context = { ...connection.context, ...renewed }
          connection.readOnly = renewed.access === 'view'
          this.scheduleConnectionLease(connection, documentName, renewed.expiresAt)
          return renewed
        } finally {
          this.clearPolicyRevalidation(connection, documentName)
        }
      },
      onLoadDocument: async ({ context, documentName, socketId }) => {
        const parsed = parseEditorCollaborationRoom(documentName)
        if (!parsed) throw invalidRoom()
        const authenticated = isInternalContext(context)
          ? null
          : connectionContext(context)
        if (
          authenticated
          && (
            authenticated.documentId !== parsed.documentId
            || authenticated.generation !== parsed.generation
          )
        ) {
          throw incompatible()
        }
        try {
          const state = await this.api.loadDocumentState({
            documentId: parsed.documentId,
            fence: this.leaseManager.assertActive(),
            generation: parsed.generation,
          })
          let selectedUpdates = state.updates
          const document = await reconstructDocument(state, {
            documentId: parsed.documentId,
            generation: parsed.generation,
            schemaVersion: this.settings.schemaVersion,
          }, this.settings.documentLimitBytes, {
            onCandidateRejected: ({ candidateIndex, reason }) => {
              this.metrics.increment('inqtrix_collaboration_snapshot_fallbacks_total')
              this.logger.warn('snapshot_candidate_rejected', {
                candidate_index: candidateIndex,
                reason,
              })
            },
            onCandidateSelected: ({ updates }) => {
              selectedUpdates = [...updates]
            },
          })
          try {
            this.coordinator.initialize(documentName, state.persistedSequence, {
              bytes: selectedUpdates.reduce(
                (total, update) => total + update.update.byteLength,
                0,
              ),
              updates: selectedUpdates.length,
            })
            return Y.encodeStateAsUpdate(document)
          } finally {
            document.destroy()
          }
        } catch (error) {
          if (authenticated) {
            this.sessions.delete(authenticated.documentId, authenticated.user.id, socketId)
          }
          throw collaborationError(error)
        }
      },
      afterLoadDocument: async ({ document, documentName }) => {
        if (this.coordinator.shouldSnapshot(documentName)) {
          this.queueSnapshot(documentName, document)
        }
      },
      beforeSync: async ({ connection, context, document, documentName, payload, type }) => {
        if (type === 0) return
        if (type !== 1 && type !== 2) throw incompatible()
        if (!this.updateLimiter.consume(connection.socketId)) throw rateLimited()
        try {
          await this.coordinator.prepareClientUpdate({
            allowNoop: type === 1,
            connectionId: pendingKey(connection.socketId, documentName),
            context: connectionContext(context),
            document,
            room: documentName,
            update: payload,
          })
        } catch (error) {
          const mapped = collaborationError(error)
          if (this.coordinator.requiresReconstruction(documentName)) {
            this.discardHeldOutbound(documentName)
            this.closeDocument(document, reloadRequired())
          }
          this.closeConnectionTransport(connection, mapped)
          throw mapped
        }
      },
      afterHandleMessage: async ({ connection, document, documentName }) => {
        try {
          const ack = this.coordinator.finishClientUpdate(
            pendingKey(connection.socketId, documentName),
            document,
          )
          if (ack) {
            if (this.coordinator.shouldSnapshot(documentName)) {
              this.queueSnapshot(documentName, document)
            }
            this.flushHeldOutbound(documentName)
            connection.sendStateless(JSON.stringify(ack))
          }
        } catch (error) {
          const mapped = collaborationError(error)
          this.logger.error('authoritative_update_apply_failed', {
            reason: mapped.reason,
          })
          this.discardHeldOutbound(documentName)
          this.closeDocument(document, mapped)
        }
      },
      beforeHandleAwareness: async ({ connection, context, document, states }) => {
        if (!connection || !context) return
        this.enforceConnectionLease(context)
        if (!this.awarenessLimiter.consume(connection.socketId)) throw rateLimited()
        assertAwarenessOwnership(document, connection, states)
        enforceAwarenessIdentity(states, connectionContext(context))
      },
      onStateless: async ({ connection, payload }) => {
        try {
          this.enforceConnectionLease(connection.context)
          if (!this.reconcileLimiter.consume(connection.socketId)) throw rateLimited()
          const acknowledged = await reconcileDurability(
            payload,
            connectionContext(connection.context),
            connection,
            this.api,
            this.leaseManager,
            this.settings,
          )
          this.metrics.increment('inqtrix_collaboration_reconcile_requests_total')
          this.metrics.add(
            'inqtrix_collaboration_reconcile_acknowledgements_total',
            acknowledged,
          )
        } catch (error) {
          const mapped = collaborationError(error)
          this.metrics.increment('inqtrix_collaboration_rejections_total', {
            reason: mapped.reason,
          })
          this.logger.warn('durability_reconcile_failed', { reason: mapped.reason })
          this.closeConnectionTransport(connection, mapped)
        }
      },
      onChange: async ({ document, documentName }) => {
        if (this.coordinator.shouldSnapshot(documentName)) {
          this.queueSnapshot(documentName, document)
        }
      },
      onStoreDocument: async ({ document, documentName }) => {
        if (!this.coordinator.hasUnsnapshottedUpdates(documentName)) return
        try {
          await this.snapshot(documentName, document)
        } catch (error) {
          this.handleSnapshotFailure(documentName, error)
          throw error
        }
      },
      connected: async ({ connection, context, documentName, socketId }) => {
        const transportId = context.transportId
        if (hasConnectionContext(context) && transportId) {
          this.registerConnection({
            connection,
            documentName,
            socketId,
            transportId,
          })
          this.authenticatedAddresses.add(authAddressKey(transportId, connection.messageAddress))
          this.releaseInitialAuthIngress(transportId, connection.messageAddress)
          this.scheduleConnectionLease(connection, documentName, context.expiresAt)
        }
        this.metrics.set('inqtrix_collaboration_active_connections', this.sockets.size)
      },
      onDisconnect: async ({ context, document, documentName, socketId }) => {
        const abortedRoom = this.coordinator.abortClientUpdate(
          pendingKey(socketId, documentName),
        )
        if (abortedRoom) {
          this.metrics.increment('inqtrix_collaboration_pending_updates_aborted_total')
          this.logger.warn('persisted_update_apply_aborted', { room: abortedRoom })
          this.discardHeldOutbound(abortedRoom)
          this.closeDocument(document, reloadRequired())
        }
        if (!isInternalContext(context) && hasConnectionContext(context)) {
          this.sessions.delete(context.documentId, context.user.id, socketId)
        }
        this.updateLimiter.delete(socketId)
        this.awarenessLimiter.delete(socketId)
        this.reconcileLimiter.delete(socketId)
        this.discardHeldOutboundForTransport(context.transportId)
        this.unregisterConnection(socketId, documentName)
        this.clearPolicyRevalidationsForSocket(socketId)
      },
      beforeUnloadDocument: async ({ documentName }) => {
        this.snapshotRetries.cancel(documentName)
        await this.coordinator.awaitRoom(documentName)
        if (this.coordinator.requiresReconstruction(documentName)) {
          this.discardHeldOutbound(documentName)
        }
      },
      afterUnloadDocument: async ({ documentName }) => {
        this.snapshotRetries.cancel(documentName)
        this.discardHeldOutbound(documentName)
        this.coordinator.markUnloaded(documentName)
      },
    })
  }

  private async acquireOperationDocument(
    room: string,
  ): Promise<{ document: Y.Doc; release: () => Promise<void> }> {
    const connection = await this.hocuspocus.openDirectConnection(room, {
      internalOperation: true,
    })
    if (!connection.document) {
      throw new CollaborationError('internal_consistency', {
        closeCode: CloseCodes.internalConsistency,
      })
    }
    return {
      document: connection.document,
      release: () => connection.disconnect({ unloadImmediately: true }),
    }
  }

  private handleUpgrade(request: IncomingMessage, socket: Duplex, head: Buffer): void {
    const url = new URL(request.url ?? '/', 'http://collaboration.internal')
    if (
      request.method !== 'GET'
      || url.pathname !== this.settings.websocketPath
      || url.search !== ''
    ) {
      rejectUpgrade(socket, 404, 'Not Found')
      return
    }
    if (!hasBearerSecret(request.headers.authorization, this.settings.secret)) {
      this.metrics.increment('inqtrix_collaboration_websocket_rejections_total', {
        reason: 'unauthorized_gateway',
      })
      rejectUpgrade(socket, 401, 'Unauthorized')
      return
    }
    if (!this.isReady()) {
      rejectUpgrade(socket, 503, 'Service Unavailable')
      return
    }

    try {
      this.webSocketServer.handleUpgrade(request, socket, head, (webSocket) => {
        const transportId = randomUUID()
        const hocusRequest = sanitizedHocuspocusRequest(request, this.settings.websocketPath)
        const guardedSocket = guardedWebSocket(webSocket, (payload) => {
          this.sendHocuspocusPayload(transportId, webSocket, payload)
        })
        const client = this.hocuspocus.handleConnection(
          guardedSocket,
          hocusRequest,
          { transportId },
        )
        this.sockets.set(transportId, webSocket)
        this.metrics.set('inqtrix_collaboration_active_connections', this.sockets.size)
        webSocket.on('message', (data, isBinary) => {
          if (this.stopping) {
            webSocket.close(CloseCodes.restarting, 'restarting')
            return
          }
          if (!isBinary) {
            webSocket.close(CloseCodes.incompatible, 'binary_protocol_required')
            return
          }
          const bytes = rawDataByteLength(data)
          if (bytes > this.settings.frameLimitBytes) {
            webSocket.close(CloseCodes.messageTooLarge, 'message_too_large')
            return
          }
          const reservation = this.reserveIngress(
            transportId,
            bytes,
            webSocket.bufferedAmount,
          )
          if (!reservation) {
            webSocket.close(CloseCodes.rateLimited, 'rate_limited')
            return
          }
          let update: Uint8Array
          try {
            update = rawDataBytes(data)
          } catch {
            this.releaseIngressReservation(reservation)
            webSocket.close(CloseCodes.incompatible, 'invalid_schema')
            return
          }
          const header = hocuspocusFrameHeader(update)
          const addressKey = header ? authAddressKey(transportId, header.address) : null
          if (
            header?.type === 2
            && addressKey
            && !this.authenticatedAddresses.has(addressKey)
          ) {
            const reservations = this.ingressAuthReservations.get(addressKey) ?? []
            reservations.push(reservation)
            this.ingressAuthReservations.set(addressKey, reservations)
          } else {
            this.ingressFrames.set(update, reservation)
          }
          client.handleMessage(update)
        })
        let transportClosed = false
        const teardownTransport: TransportTeardown = (code, reason) => {
          if (transportClosed) return
          transportClosed = true
          this.abortPendingForTransport(transportId)
          this.clearTransportConnections(transportId)
          this.clearIngress(transportId)
          this.sockets.delete(transportId)
          this.discardHeldOutboundForTransport(transportId)
          this.outboundFlow.clear(transportId)
          this.outboundRejected.delete(transportId)
          this.transportTeardowns.delete(transportId)
          this.metrics.set('inqtrix_collaboration_active_connections', this.sockets.size)
          client.handleClose({ code, reason })
        }
        this.transportTeardowns.set(transportId, teardownTransport)
        webSocket.on('close', (code, reason) => {
          teardownTransport(code, reason.toString('utf8'))
        })
        webSocket.on('error', () => {
          this.logger.warn('websocket_transport_error')
        })
      })
    } catch {
      this.logger.warn('websocket_upgrade_failed')
      socket.destroy()
    }
  }

  private isReady(): boolean {
    return this.started && !this.stopping && this.leaseManager.isReady()
  }

  private handleLeaseLoss(): void {
    this.closeAllSockets(CloseCodes.serviceUnavailable, 'instance_lease_lost')
    void this.coordinator.awaitAll().then(() => {
      for (const document of this.hocuspocus.documents.values()) {
        if (document.getConnectionsCount() === 0) {
          void this.hocuspocus.unloadDocument(document)
        }
      }
    })
  }

  private startMaintenance(): void {
    if (this.maintenanceTimer) return
    this.maintenanceTimer = setInterval(() => {
      this.queueMaintenance()
    }, this.settings.maintenanceIntervalMs)
    this.maintenanceTimer.unref()
    this.queueMaintenance()
  }

  private stopMaintenance(): Promise<void> | null {
    if (this.maintenanceTimer) clearInterval(this.maintenanceTimer)
    this.maintenanceTimer = null
    return this.maintenanceInFlight
  }

  private queueMaintenance(): void {
    if (!this.isReady() || this.maintenanceInFlight) return
    const maintenance = this.operations.runMaintenance().finally(() => {
      this.maintenanceInFlight = null
    })
    this.maintenanceInFlight = maintenance
  }

  private startPolicyPolling(): void {
    if (this.policyTimer) return
    this.policyTimer = setInterval(() => {
      this.queuePolicyPoll()
    }, this.settings.policyPollMs)
    this.policyTimer.unref()
    this.queuePolicyPoll()
  }

  private stopPolicyPolling(): Promise<void> | null {
    if (this.policyTimer) clearInterval(this.policyTimer)
    this.policyTimer = null
    for (const timer of this.policyRevalidationTimers.values()) clearTimeout(timer)
    this.policyRevalidationTimers.clear()
    return this.policyPollInFlight
  }

  private queuePolicyPoll(): void {
    if (!this.isReady() || this.policyPollInFlight) return
    const startedAt = performance.now()
    const poll = this.pollPolicyEvents()
      .catch((error) => {
        const mapped = collaborationError(error)
        this.metrics.increment('inqtrix_collaboration_policy_poll_errors_total', {
          reason: mapped.reason,
        })
        this.logger.warn('policy_feed_poll_failed', { reason: mapped.reason })
      })
      .finally(() => {
        this.metrics.observeMilliseconds(
          'inqtrix_collaboration_policy_poll_seconds',
          performance.now() - startedAt,
        )
        this.policyPollInFlight = null
      })
    this.policyPollInFlight = poll
  }

  private async pollPolicyEvents(): Promise<void> {
    const page = await this.api.pollPolicyEvents({
      afterId: this.policyCursor,
      fence: this.leaseManager.assertActive(),
      limit: 500,
    })
    if (this.stopping) return
    if (page.resetRequired) {
      this.metrics.increment('inqtrix_collaboration_policy_resets_total')
      this.logger.warn('policy_feed_reset_required', { cursor: page.cursor })
      this.requestAllConnectionRevalidations()
      this.policyCursor = page.cursor
      this.metrics.set('inqtrix_collaboration_policy_cursor', page.cursor)
      return
    }

    const checkedDocuments = new Map<string, boolean>()
    for (const event of page.events) {
      if (event.resourceType === 'editor_document' && event.resourceId) {
        let generationInvalid = checkedDocuments.get(event.resourceId)
        if (generationInvalid === undefined) {
          generationInvalid = await this.closeGenerationMismatches(event.resourceId)
          checkedDocuments.set(event.resourceId, generationInvalid)
        }
        if (generationInvalid) continue
      }
      this.requestAffectedConnectionRevalidations(event)
    }
    this.policyCursor = page.cursor
    this.metrics.set('inqtrix_collaboration_policy_cursor', page.cursor)
  }

  private async closeGenerationMismatches(documentId: string): Promise<boolean> {
    let closed = false
    const matching = [...this.hocuspocus.documents.values()].filter((document) => (
      parseEditorCollaborationRoom(document.name)?.documentId === documentId
    ))
    for (const document of matching) {
      const parsed = parseEditorCollaborationRoom(document.name)
      if (!parsed) continue
      try {
        const state = await this.api.loadDocumentState({
          documentId,
          fence: this.leaseManager.assertActive(),
          generation: parsed.generation,
        })
        if (state.generation !== parsed.generation) {
          this.closeDocument(document, incompatible())
          closed = true
        }
      } catch (error) {
        const mapped = collaborationError(error)
        if (
          (error instanceof ApiRequestError && (error.status === 404 || error.status === 409))
          || mapped.reason === 'generation_mismatch'
          || mapped.reason === 'update_required'
        ) {
          this.closeDocument(document, incompatible())
          closed = true
          continue
        }
        throw error
      }
    }
    return closed
  }

  private requestAffectedConnectionRevalidations(event: CollaborationPolicyEvent): void {
    for (const document of this.hocuspocus.documents.values()) {
      const parsed = parseEditorCollaborationRoom(document.name)
      if (
        event.resourceType === 'editor_document'
        && event.resourceId !== null
        && parsed?.documentId !== event.resourceId
      ) {
        continue
      }
      for (const connection of document.getConnections()) {
        if (!hasConnectionContext(connection.context)) continue
        if (connection.context.user.id !== event.targetUserId) continue
        this.requestPolicyRevalidation(connection, document.name)
      }
    }
  }

  private requestAllConnectionRevalidations(): void {
    for (const document of this.hocuspocus.documents.values()) {
      for (const connection of document.getConnections()) {
        if (hasConnectionContext(connection.context)) {
          this.requestPolicyRevalidation(connection, document.name)
        }
      }
    }
  }

  private requestPolicyRevalidation(
    connection: Connection<HocuspocusContext>,
    documentName: string,
  ): void {
    const key = pendingKey(connection.socketId, documentName)
    if (this.policyRevalidationTimers.has(key)) return
    if (hasConnectionContext(connection.context)) {
      this.scheduleConnectionLease(
        connection,
        documentName,
        connection.context.expiresAt,
      )
    }
    const timer = setTimeout(() => {
      this.policyRevalidationTimers.delete(key)
      this.metrics.increment('inqtrix_collaboration_policy_revalidation_timeouts_total')
      this.logger.warn('policy_revalidation_timed_out')
      connection.close({
        code: CloseCodes.accessRevoked,
        reason: 'policy_revalidation_timeout',
      })
    }, this.settings.policyRevalidationTimeoutMs)
    timer.unref()
    this.policyRevalidationTimers.set(key, timer)
    this.metrics.increment('inqtrix_collaboration_policy_revalidations_total')
    connection.requestToken()
  }

  private clearPolicyRevalidation(
    connection: Connection<HocuspocusContext>,
    documentName: string,
  ): void {
    const key = pendingKey(connection.socketId, documentName)
    const timer = this.policyRevalidationTimers.get(key)
    if (timer) clearTimeout(timer)
    this.policyRevalidationTimers.delete(key)
  }

  private clearPolicyRevalidationsForSocket(socketId: string): void {
    const prefix = `${socketId}\0`
    for (const [key, timer] of this.policyRevalidationTimers) {
      if (!key.startsWith(prefix)) continue
      clearTimeout(timer)
      this.policyRevalidationTimers.delete(key)
    }
  }

  private registerConnection(registration: RegisteredConnection): void {
    const key = pendingKey(registration.socketId, registration.documentName)
    this.connections.set(key, registration)
    const keys = this.transportConnections.get(registration.transportId) ?? new Set<string>()
    keys.add(key)
    this.transportConnections.set(registration.transportId, keys)
  }

  private enforceConnectionLease(context: HocuspocusContext): void {
    try {
      assertConnectionLease(context)
    } catch (error) {
      const transportId = context.transportId
      const socket = transportId ? this.sockets.get(transportId) : undefined
      if (socket?.readyState === WebSocket.OPEN) {
        socket.close(CloseCodes.leaseInvalid, 'invalid_lease')
      }
      throw error
    }
  }

  private closeConnectionTransport(
    connection: Connection<HocuspocusContext>,
    error: CollaborationError,
  ): void {
    const transportId = connection.context.transportId
    const socket = transportId ? this.sockets.get(transportId) : undefined
    if (socket?.readyState === WebSocket.OPEN) {
      socket.close(error.code, error.reason)
      return
    }
    connection.close({ code: error.code, reason: error.reason })
  }

  private unregisterConnection(socketId: string, documentName: string): void {
    const key = pendingKey(socketId, documentName)
    const registration = this.connections.get(key)
    this.connections.delete(key)
    this.clearConnectionLeaseTimer(key)
    if (!registration) return
    this.authenticatedAddresses.delete(
      authAddressKey(registration.transportId, registration.connection.messageAddress),
    )
    const keys = this.transportConnections.get(registration.transportId)
    keys?.delete(key)
    if (keys?.size === 0) this.transportConnections.delete(registration.transportId)
  }

  private clearTransportConnections(transportId: string): void {
    const keys = this.transportConnections.get(transportId)
    if (!keys) return
    for (const key of keys) {
      this.connections.delete(key)
      this.clearConnectionLeaseTimer(key)
    }
    this.transportConnections.delete(transportId)
  }

  private abortPendingForTransport(transportId: string): void {
    const keys = this.transportConnections.get(transportId)
    if (!keys) return
    for (const key of keys) {
      const room = this.coordinator.abortClientUpdate(key)
      if (!room) continue
      this.metrics.increment('inqtrix_collaboration_pending_updates_aborted_total')
      this.logger.warn('persisted_update_apply_aborted', { room })
      this.discardHeldOutbound(room)
      const document = this.hocuspocus.documents.get(room)
      if (document) this.closeDocument(document, reloadRequired())
    }
  }

  private scheduleConnectionLease(
    connection: Connection<HocuspocusContext>,
    documentName: string,
    expiresAt: number,
  ): void {
    const key = pendingKey(connection.socketId, documentName)
    this.clearConnectionLeaseTimer(key)
    const remainingMs = Math.max(0, Math.ceil(expiresAt * 1_000 - Date.now()))
    const timer = setTimeout(() => {
      this.connectionLeaseTimers.delete(key)
      const registration = this.connections.get(key)
      if (!registration) return
      const context = registration.connection.context
      if (hasConnectionContext(context) && context.expiresAt * 1_000 > Date.now()) {
        this.scheduleConnectionLease(
          registration.connection,
          registration.documentName,
          context.expiresAt,
        )
        return
      }
      this.metrics.increment('inqtrix_collaboration_lease_expirations_total')
      this.logger.warn('connection_lease_expired')
      const socket = this.sockets.get(registration.transportId)
      if (socket?.readyState === WebSocket.OPEN) {
        socket.close(CloseCodes.leaseInvalid, 'invalid_lease')
      } else {
        registration.connection.close({
          code: CloseCodes.leaseInvalid,
          reason: 'invalid_lease',
        })
      }
    }, Math.min(remainingMs, 2_147_483_647))
    timer.unref()
    this.connectionLeaseTimers.set(key, timer)
  }

  private clearConnectionLeaseTimer(key: string): void {
    const timer = this.connectionLeaseTimers.get(key)
    if (timer) clearTimeout(timer)
    this.connectionLeaseTimers.delete(key)
  }

  private clearAllConnectionLeaseTimers(): void {
    for (const timer of this.connectionLeaseTimers.values()) clearTimeout(timer)
    this.connectionLeaseTimers.clear()
  }

  private reserveIngress(
    transportId: string,
    bytes: number,
    bufferedAmount: number,
  ): IngressReservation | null {
    const current = this.ingressUsage.get(transportId) ?? { bytes: 0, frames: 0 }
    if (
      bufferedAmount > this.settings.socketBackpressureBytes
      || current.bytes + bytes > this.settings.maxQueuedBytes
      || current.frames + 1 > this.settings.maxQueuedFrames
    ) {
      this.metrics.increment('inqtrix_collaboration_ingress_rejections_total', {
        reason: 'rate_limited',
      })
      return null
    }
    const reservation: IngressReservation = {
      bytes,
      released: false,
      transportId,
    }
    this.ingressUsage.set(transportId, {
      bytes: current.bytes + bytes,
      frames: current.frames + 1,
    })
    this.updateIngressMetrics()
    return reservation
  }

  private releaseIngress(update: Uint8Array): void {
    const reservation = this.ingressFrames.get(update)
    if (!reservation) return
    this.ingressFrames.delete(update)
    this.releaseIngressReservation(reservation)
  }

  private releaseInitialAuthIngress(transportId: string, address: string): void {
    const key = authAddressKey(transportId, address)
    const reservations = this.ingressAuthReservations.get(key) ?? []
    this.ingressAuthReservations.delete(key)
    for (const reservation of reservations) this.releaseIngressReservation(reservation)
  }

  private releaseIngressReservation(reservation: IngressReservation): void {
    if (reservation.released) return
    reservation.released = true
    const current = this.ingressUsage.get(reservation.transportId)
    if (!current) return
    const next = {
      bytes: Math.max(0, current.bytes - reservation.bytes),
      frames: Math.max(0, current.frames - 1),
    }
    if (next.bytes === 0 && next.frames === 0) {
      this.ingressUsage.delete(reservation.transportId)
    } else {
      this.ingressUsage.set(reservation.transportId, next)
    }
    this.updateIngressMetrics()
  }

  private clearIngress(transportId: string): void {
    this.ingressUsage.delete(transportId)
    const prefix = `${transportId}\0`
    for (const key of this.ingressAuthReservations.keys()) {
      if (key.startsWith(prefix)) this.ingressAuthReservations.delete(key)
    }
    for (const key of this.authenticatedAddresses) {
      if (key.startsWith(prefix)) this.authenticatedAddresses.delete(key)
    }
    this.updateIngressMetrics()
  }

  private updateIngressMetrics(): void {
    let bytes = 0
    let frames = 0
    for (const usage of this.ingressUsage.values()) {
      bytes += usage.bytes
      frames += usage.frames
    }
    this.metrics.set('inqtrix_collaboration_ingress_queued_bytes', bytes)
    this.metrics.set('inqtrix_collaboration_ingress_queued_frames', frames)
  }

  private closeSocketAfterHook(
    context: HocuspocusContext,
    error: CollaborationError,
  ): void {
    const transportId = context.transportId
    if (!transportId) return
    setTimeout(() => {
      const socket = this.sockets.get(transportId)
      if (socket?.readyState === WebSocket.OPEN) {
        socket.close(error.code, error.reason)
      }
    }, 0)
  }

  private closeAllSockets(code: number, reason: string): void {
    for (const socket of this.sockets.values()) {
      if (socket.readyState === WebSocket.OPEN) socket.close(code, reason)
    }
  }

  private async teardownSockets(deadline: number): Promise<void> {
    const gracefulDeadline = Math.min(
      deadline,
      Date.now() + this.socketCloseGraceMs,
    )
    await waitForCondition(() => this.sockets.size === 0, gracefulDeadline)
    if (this.sockets.size === 0) return

    for (const [transportId, socket] of [...this.sockets]) {
      socket.terminate()
      this.transportTeardowns.get(transportId)?.(
        CloseCodes.restarting,
        'restarting',
      )
    }
    await waitForCondition(() => this.sockets.size === 0, deadline)
    if (this.sockets.size > 0) {
      this.logger.warn('shutdown_socket_teardown_timed_out', {
        connections: this.sockets.size,
      })
    }
  }

  private async awaitShutdownPhase(
    phase: string,
    promise: Promise<unknown> | null,
    deadline: number,
  ): Promise<boolean> {
    if (!promise) return true
    const outcome = await settleByDeadline(promise, deadline)
    if (outcome === 'settled') return true
    this.logger.warn(
      outcome === 'timeout' ? 'shutdown_phase_timed_out' : 'shutdown_phase_failed',
      { phase },
    )
    return false
  }

  private closeDocument(document: HocuspocusDocument, error: CollaborationError): void {
    for (const connection of document.getConnections()) {
      connection.close({ code: error.code, reason: error.reason })
    }
  }

  private sendHocuspocusPayload(
    transportId: string,
    socket: WebSocket,
    payload: HocuspocusPayload,
  ): void {
    if (this.stopping || this.outboundRejected.has(transportId)) return
    const bytes = hocuspocusPayloadBytes(payload)
    const reservation = this.outboundFlow.reserve(
      transportId,
      socket,
      bytes,
    )
    if (!reservation) {
      this.rejectSlowReceiver(transportId, socket)
      return
    }
    const room = hocuspocusPayloadRoom(payload)
    if (!room || !this.coordinator.isBroadcastBlocked(room)) {
      this.sendReservedOutbound({ payload, reservation, socket, transportId })
      return
    }
    const frames = this.heldOutbound.get(room) ?? []
    frames.push({ payload, reservation, socket, transportId })
    this.heldOutbound.set(room, frames)
  }

  private flushHeldOutbound(room: string): void {
    const frames = this.takeHeldOutbound(room)
    for (const frame of frames) {
      this.sendReservedOutbound(frame)
    }
  }

  private discardHeldOutbound(room: string): void {
    for (const frame of this.takeHeldOutbound(room)) {
      this.outboundFlow.release(frame.reservation)
    }
  }

  private discardHeldOutboundForTransport(transportId: string | undefined): void {
    if (!transportId) return
    for (const [room, frames] of this.heldOutbound) {
      const retained: HeldOutboundFrame[] = []
      for (const frame of frames) {
        if (frame.transportId === transportId) {
          this.outboundFlow.release(frame.reservation)
        } else {
          retained.push(frame)
        }
      }
      if (retained.length > 0) this.heldOutbound.set(room, retained)
      else this.heldOutbound.delete(room)
    }
  }

  private takeHeldOutbound(room: string): HeldOutboundFrame[] {
    const frames = this.heldOutbound.get(room) ?? []
    this.heldOutbound.delete(room)
    return frames
  }

  private sendReservedOutbound(frame: HeldOutboundFrame): void {
    if (this.outboundRejected.has(frame.transportId)) {
      this.outboundFlow.release(frame.reservation)
      return
    }
    const sent = this.outboundFlow.send(
      frame.reservation,
      frame.socket,
      frame.payload,
      () => this.rejectSlowReceiver(frame.transportId, frame.socket),
    )
    if (!sent && frame.socket.readyState === WebSocket.OPEN) {
      this.rejectSlowReceiver(frame.transportId, frame.socket)
    }
  }

  private rejectSlowReceiver(transportId: string, socket: WebSocket): void {
    if (this.outboundRejected.has(transportId)) return
    this.outboundRejected.add(transportId)
    this.metrics.increment('inqtrix_collaboration_outbound_rejections_total', {
      reason: 'rate_limited',
    })
    this.discardHeldOutboundForTransport(transportId)
    if (socket.readyState === WebSocket.OPEN) {
      socket.close(CloseCodes.rateLimited, 'rate_limited')
    }
  }

  private queueSnapshot(room: string, document: Y.Doc): void {
    void this.snapshot(room, document).catch((error) => {
      this.handleSnapshotFailure(room, error)
    })
  }

  private snapshot(room: string, document: Y.Doc): Promise<void> {
    const existing = this.snapshotTasks.get(room)
    if (existing) return existing
    let completed = false
    const task = this.storeSnapshotsUntilCurrent(room, document)
      .then(() => {
        completed = true
      })
      .finally(() => {
        this.snapshotTasks.delete(room)
        if (
          completed
          && this.hocuspocus.documents.get(room) === document
          && this.coordinator.hasUnsnapshottedUpdates(room)
        ) {
          this.queueSnapshot(room, document)
        }
      })
    this.snapshotTasks.set(room, task)
    return task
  }

  private async storeSnapshotsUntilCurrent(room: string, document: Y.Doc): Promise<void> {
    while (true) {
      const currentSequenceCovered = await this.operations.storeSnapshot(room, document)
      this.metrics.increment('inqtrix_collaboration_snapshots_total')
      if (currentSequenceCovered) this.snapshotRetries.cancel(room)
      await this.coordinator.awaitRoom(room)
      if (!this.coordinator.hasUnsnapshottedUpdates(room)) return
      await eventLoopTurn()
    }
  }

  private handleSnapshotFailure(room: string, error: unknown): void {
    this.recordSnapshotFailure(error)
    this.snapshotRetries.schedule(room, async () => {
      const document = this.hocuspocus.documents.get(room)
      if (!document) return
      await this.snapshot(room, document)
    })
  }

  private recordSnapshotFailure(error: unknown): void {
    const mapped = collaborationError(error)
    this.metrics.increment('inqtrix_collaboration_snapshot_errors_total', {
      reason: mapped.reason,
    })
    this.logger.warn('snapshot_store_failed', { reason: mapped.reason })
  }

  private async flushSnapshotsForShutdown(): Promise<void> {
    if (!this.leaseManager.isReady()) return
    const snapshots = [...this.hocuspocus.documents].map(async ([room, document]) => {
      if (!this.coordinator.hasUnsnapshottedUpdates(room)) return
      try {
        await this.snapshot(room, document)
      } catch (error) {
        const mapped = collaborationError(error)
        this.logger.warn('shutdown_snapshot_failed', { reason: mapped.reason })
      }
    })
    await Promise.all(snapshots)
  }
}

function pendingKey(socketId: string, documentName: string): string {
  return `${socketId}\0${documentName}`
}

function eventLoopTurn(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0))
}

type DeadlineOutcome = 'failed' | 'settled' | 'timeout'

function settleByDeadline(
  promise: Promise<unknown>,
  deadline: number,
): Promise<DeadlineOutcome> {
  const remaining = deadline - Date.now()
  if (remaining <= 0) return Promise.resolve('timeout')
  return new Promise((resolve) => {
    let finished = false
    const finish = (outcome: DeadlineOutcome): void => {
      if (finished) return
      finished = true
      clearTimeout(timer)
      resolve(outcome)
    }
    const timer = setTimeout(() => finish('timeout'), remaining)
    void promise.then(
      () => finish('settled'),
      () => finish('failed'),
    )
  })
}

async function waitForCondition(
  condition: () => boolean,
  deadline: number,
): Promise<void> {
  while (!condition() && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 10))
  }
}

function guardedWebSocket(
  socket: WebSocket,
  send: (payload: HocuspocusPayload) => void,
): WebSocketLike {
  return {
    close: (code, reason) => socket.close(code, reason),
    get readyState() {
      return socket.readyState
    },
    send,
  }
}

function hocuspocusPayloadBytes(payload: HocuspocusPayload): number {
  if (typeof payload === 'string') return Buffer.byteLength(payload, 'utf8')
  if (payload instanceof Blob) return payload.size
  if (ArrayBuffer.isView(payload)) return payload.byteLength
  return payload.byteLength
}

function hocuspocusPayloadRoom(payload: HocuspocusPayload): string | null {
  if (typeof payload === 'string' || payload instanceof Blob) return null
  const bytes = ArrayBuffer.isView(payload)
    ? new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength)
    : new Uint8Array(payload)
  let length = 0
  let shift = 0
  let offset = 0
  while (offset < bytes.byteLength && shift <= 35) {
    const value = bytes[offset++]!
    length |= (value & 0x7f) << shift
    if ((value & 0x80) === 0) break
    shift += 7
  }
  if (length < 1 || length > 512 || offset + length > bytes.byteLength) return null
  const rawKey = new TextDecoder().decode(bytes.subarray(offset, offset + length))
  return rawKey.split('\0', 1)[0] || null
}

function isInternalContext(context: HocuspocusContext): context is InternalContext {
  return context.internalOperation === true
}

function hasConnectionContext(context: HocuspocusContext): context is ConnectionContext {
  return (
    typeof context.documentId === 'string'
    && typeof context.generation === 'number'
    && typeof context.leaseId === 'string'
    && typeof context.expiresAt === 'number'
    && typeof context.tenantId === 'string'
    && typeof context.user?.id === 'string'
  )
}

function connectionContext(context: HocuspocusContext): ConnectionContext {
  if (!hasConnectionContext(context)) {
    throw new CollaborationError('invalid_lease', {
      closeCode: CloseCodes.leaseInvalid,
      httpStatus: 401,
    })
  }
  return context
}

function assertConnectionLease(context: HocuspocusContext): void {
  const authenticated = connectionContext(context)
  if (authenticated.expiresAt > Date.now() / 1_000) return
  throw new CollaborationError('invalid_lease', {
    closeCode: CloseCodes.leaseInvalid,
    httpStatus: 401,
  })
}

function assertAwarenessOwnership(
  document: HocuspocusDocument,
  connection: Connection<HocuspocusContext>,
  states: ReadonlyMap<number, Record<string, unknown>>,
): void {
  const owned = document.getClients(connection)
  for (const clientId of states.keys()) {
    if (owned.size > 0 && !owned.has(clientId)) throw awarenessRejected()
    for (const peer of document.getConnections()) {
      if (peer !== connection && document.getClients(peer).has(clientId)) {
        throw awarenessRejected()
      }
    }
  }
}

function sanitizedHocuspocusRequest(
  request: IncomingMessage,
  path: string,
): Request {
  const headers = new Headers()
  for (const name of ['origin', 'user-agent', 'x-forwarded-host', 'x-forwarded-proto']) {
    const value = request.headers[name]
    if (typeof value === 'string') headers.set(name, value)
  }
  return new Request(`http://collaboration.internal${path}`, { headers })
}

function rawDataBytes(data: RawData): Uint8Array {
  if (data instanceof ArrayBuffer) return new Uint8Array(data.slice(0))
  if (Array.isArray(data)) return new Uint8Array(Buffer.concat(data))
  return new Uint8Array(Buffer.from(data))
}

function rawDataByteLength(data: RawData): number {
  if (data instanceof ArrayBuffer) return data.byteLength
  if (Array.isArray(data)) {
    return data.reduce((total, item) => total + item.byteLength, 0)
  }
  return data.byteLength
}

function hocuspocusFrameHeader(
  update: Uint8Array,
): { address: string; type: number } | null {
  const addressLength = readVarUint(update, 0)
  if (!addressLength) return null
  const typeOffset = addressLength.offset + addressLength.value
  if (typeOffset > update.byteLength) return null
  const type = readVarUint(update, typeOffset)?.value
  if (type === undefined) return null
  return {
    address: new TextDecoder().decode(
      update.subarray(addressLength.offset, typeOffset),
    ),
    type,
  }
}

function authAddressKey(transportId: string, address: string): string {
  return `${transportId}\0${address}`
}

function readVarUint(
  bytes: Uint8Array,
  start: number,
): { offset: number; value: number } | null {
  let value = 0
  let shift = 0
  let offset = start
  while (offset < bytes.byteLength && shift <= 35) {
    const byte = bytes[offset++]!
    value += (byte & 0x7f) * (2 ** shift)
    if ((byte & 0x80) === 0) {
      if (!Number.isSafeInteger(value)) return null
      return { offset, value }
    }
    shift += 7
  }
  return null
}

function rejectUpgrade(socket: Duplex, status: number, reason: string): void {
  socket.end(
    `HTTP/1.1 ${status} ${reason}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`,
  )
}

function invalidRoom(): CollaborationError {
  return new CollaborationError('invalid_room', {
    closeCode: CloseCodes.incompatible,
    httpStatus: 409,
  })
}

function incompatible(): CollaborationError {
  return new CollaborationError('update_required', {
    closeCode: CloseCodes.incompatible,
    httpStatus: 409,
  })
}

function unavailable(): CollaborationError {
  return new CollaborationError('service_unavailable', {
    closeCode: CloseCodes.serviceUnavailable,
    httpStatus: 503,
  })
}

function reloadRequired(): CollaborationError {
  return new CollaborationError('restarting', {
    closeCode: CloseCodes.restarting,
    httpStatus: 503,
  })
}

function tooLarge(): CollaborationError {
  return new CollaborationError('message_too_large', {
    closeCode: CloseCodes.messageTooLarge,
    httpStatus: 413,
  })
}

function rateLimited(): CollaborationError {
  return new CollaborationError('rate_limited', {
    closeCode: CloseCodes.rateLimited,
    httpStatus: 429,
  })
}

function awarenessRejected(): CollaborationError {
  return new CollaborationError('invalid_request', {
    closeCode: CloseCodes.accessRevoked,
    httpStatus: 400,
  })
}
