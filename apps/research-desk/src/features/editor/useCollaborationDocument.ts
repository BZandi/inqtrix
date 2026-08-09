import {
  HocuspocusProvider,
  HocuspocusProviderWebsocket,
} from '@hocuspocus/provider'
import {
  EDITOR_COLLABORATION_PROTOCOL_VERSION,
  EDITOR_SCHEMA_VERSION,
  INQTRIX_STRUCTURE_SUGGESTION_ATTR,
  SUGGESTION_MARK_NAMES,
  isCollaborationDurableAck,
  isStructureSuggestionData,
} from '@inqtrix/editor-schema'
import { useEffect, useMemo, useState } from 'react'
import * as Y from 'yjs'

import {
  createEditorCollaborationSession,
  createGuestEditorCollaborationSession,
  type EditorCollaborationSession,
  type EditorCollaborationUser,
  type EditorGuestAccessSession,
  type InqtrixRequestError,
} from '@/api/inqtrixClient'
import type {
  EditorCollaborationConnectionStatus,
  EditorCollaborationDurabilityStatus,
  EditorDocumentRecord,
} from '@/features/project/types'
import type { CollaborationLiveAuthority } from './collaborationAuthority'
import {
  collaborationCommandId,
  collaborationSha256Hex,
} from './collaborationCrypto'

export const COLLABORATION_UPDATE_BATCH_MS = 50
export const COLLABORATION_RECONNECT_DELAYS_MS = [
  1_000,
  2_000,
  4_000,
  8_000,
  15_000,
  30_000,
] as const
const RELEASE_DURABILITY_TIMEOUT_MS = 60_000
const RELEASE_FAILURE_TTL_MS = 5 * 60_000
const MAX_RELEASE_FAILURES = 20
const MIN_PROVIDER_FLUSH_MS = 10
const MAX_PROVIDER_FLUSH_MS = 1_000
const LOCAL_TRANSPORT_ORIGIN = Object.freeze({ kind: 'local-collaboration-batch' })
const REMOTE_EDITOR_ORIGIN = Object.freeze({ kind: 'remote-collaboration-update' })

export type CollaborationProviderEvents = {
  onAuthenticated: () => void
  onAuthenticationFailed: (reason: string) => void
  onClose: (code: number, reason: string) => void
  onStateless: (payload: string) => void
  onSynced: () => void
}

export type CollaborationProviderAdapter = {
  connect: () => Promise<void> | void
  destroy: () => void
  disconnect: () => void
  provider: HocuspocusProvider
  replayUpdate: (update: Uint8Array) => void
  sendStateless: (payload: string) => void
  setAwarenessUser: (user: EditorCollaborationUser) => void
  syncToken: () => Promise<void> | void
}

export type CollaborationProviderFactoryOptions = {
  document: Y.Doc
  events: CollaborationProviderEvents
  getToken: () => string
  room: string
  url: string
}

export type CollaborationDocumentHandle = {
  access: 'comment' | 'edit' | 'suggest' | 'view' | null
  activityRevision: number
  authorityRevision: number
  canEdit: boolean
  blockingFailure: string | null
  commentEventVersion: number
  commentMentionEventVersion: number
  connectionStatus: EditorCollaborationConnectionStatus
  document: Y.Doc | null
  documentId: string | null
  durabilityStatus: EditorCollaborationDurabilityStatus
  error: string | null
  generation: number | null
  hasUnconfirmedLocalChanges: boolean
  lastPersistedSequence: number
  lastLocalDurableSequence: number
  lifecycleStatus: 'connecting' | 'error' | 'inactive' | 'read_only' | 'reconnecting' | 'saved' | 'syncing'
  lifecycleKey: string
  pendingHashes: readonly string[]
  provider: HocuspocusProvider | null
  readAuthority: () => CollaborationLiveAuthority
  reconnectAttempt: number
  recoverability: 'login' | 'none' | 'reload' | 'retry'
  retryConnection: () => Promise<void>
  nextReconnectAt: number | null
  synced: boolean
  flushAndAwaitDurable: () => Promise<number>
  flushAndAwaitDurability: () => Promise<void>
  setAuthoritativeSequence: (sequence: number) => void
  updateAuthoritativeSequence: (sequence: number) => void
  user: EditorCollaborationUser | null
}

export type CollaborationDocumentControllerOptions = {
  documentId: string
  generation: number
  initialPersistedSequence: number
  requestSession: (
    leaseToken?: string,
    rotationCommandId?: string,
  ) => Promise<EditorCollaborationSession>
  resolveWebSocketUrl: (path: string) => string
  schemaVersion: number
}

export type CollaborationDocumentControllerDependencies = {
  cancelTimer: (timer: number) => void
  createDocument: () => Y.Doc
  createCommandId: () => string
  createProvider: (
    options: CollaborationProviderFactoryOptions,
  ) => CollaborationProviderAdapter
  hashUpdate: (update: Uint8Array) => Promise<string>
  now: () => number
  random: () => number
  scheduleTimer: (callback: () => void, delayMs: number) => number
}

type RefreshMode = 'periodic' | 'reconnect'

type ConfiguredCollaborationSession = EditorCollaborationSession & {
  provider_flush_ms?: number
  refresh_after?: number
}

type LifecycleRegistryEntry = {
  controller: CollaborationDocumentController
  identity: string | null
  lingerSince: number | null
  lingerTimer: ReturnType<typeof setTimeout> | null
  references: number
}

/** How long a fully released lifecycle keeps its live session before the
 * real teardown. Re-entering the editor within this window retains the same
 * controller — no session POST, no websocket rebuild, no resync — which is
 * what keeps view switching from burning the server's per-user-per-document
 * lease budget (5 active leases, 60s TTL). The rotation timer keeps the lease
 * fresh for the whole window because release() is deferred, not softened. */
export const LIFECYCLE_LINGER_MS = 180_000
/** Upper bound on lingering documents; each holds a Y.Doc and an open
 * websocket, so the oldest is torn down when a fourth starts lingering. */
export const MAX_LINGERING_LIFECYCLES = 3

const lifecycleRegistry = new Map<string, LifecycleRegistryEntry>()
const lifecycleFailures = new Map<string, { error: string; expiresAt: number }>()

/** Flush every retained collaboration document before an identity boundary.
 *
 * The registry is the lifecycle authority shared by editor surfaces. Taking a
 * controller snapshot prevents React cleanup from changing the iteration while
 * acknowledgements are in flight; a failed durability boundary rejects the
 * caller so logout can leave the authenticated session intact.
 */
export async function flushActiveCollaborationDocuments(): Promise<void> {
  const controllers = new Set(
    [...lifecycleRegistry.values()].map((entry) => entry.controller),
  )
  await Promise.all(
    [...controllers].map((controller) => controller.flushAndAwaitDurability()),
  )
}

export class CollaborationDocumentController {
  private readonly dependencies: CollaborationDocumentControllerDependencies
  private readonly document: Y.Doc
  private readonly transportDocument: Y.Doc
  private readonly listeners = new Set<(state: CollaborationDocumentHandle) => void>()
  private readonly options: CollaborationDocumentControllerOptions
  private readonly pendingHashes = new Set<string>()
  private readonly pendingUpdates = new Map<string, Uint8Array>()
  private readonly earlyAcks = new Set<string>()
  private readonly durabilityWaiters = new Set<{
    reject: (error: Error) => void
    resolve: () => void
  }>()
  private readonly localUpdateBatch: Uint8Array[] = []
  private awarenessUserKey: string | null = null
  private authenticated = false
  private blockingFailure: string | null = null
  private destroyed = false
  private durabilityFailed = false
  private editorInputAttached = true
  private hashesInFlight = 0
  private ignoreNextClose = false
  private leaseToken: string | null = null
  private providerAdapter: CollaborationProviderAdapter | null = null
  private localUpdateTimer: number | null = null
  private localUpdateBatchHasSuggestionBoundary = false
  private onReleaseFailed: ((error: string) => void) | null = null
  private onReleasedSettled: (() => void) | null = null
  private providerFlushMs = COLLABORATION_UPDATE_BATCH_MS
  private reconciliationNeeded = false
  private replayedPendingHashes = new Set<string>()
  private reconnectTimer: number | null = null
  private released = false
  private releaseTimer: number | null = null
  private refreshInFlight: Promise<void> | null = null
  private refreshTimer: number | null = null
  private room: string | null = null
  private rotationCommandId: string | null = null
  private retryInFlight: Promise<void> | null = null
  private started = false
  private synced = false
  private state: CollaborationDocumentHandle

  constructor(
    options: CollaborationDocumentControllerOptions,
    dependencies: CollaborationDocumentControllerDependencies = browserDependencies,
  ) {
    this.options = options
    this.dependencies = dependencies
    this.document = dependencies.createDocument()
    this.transportDocument = dependencies.createDocument()
    this.state = {
      access: null,
      activityRevision: 0,
      authorityRevision: 0,
      blockingFailure: null,
      canEdit: false,
      commentEventVersion: 0,
      commentMentionEventVersion: 0,
      connectionStatus: 'inactive',
      document: this.document,
      documentId: options.documentId,
      durabilityStatus: 'idle',
      error: null,
      flushAndAwaitDurable: this.flushAndAwaitDurable,
      flushAndAwaitDurability: this.flushAndAwaitDurability,
      lastPersistedSequence: options.initialPersistedSequence,
      lastLocalDurableSequence: 0,
      generation: options.generation,
      hasUnconfirmedLocalChanges: false,
      lifecycleStatus: 'inactive',
      lifecycleKey: `${options.documentId}:g${options.generation}:${this.document.clientID}`,
      pendingHashes: [],
      provider: null,
      readAuthority: this.readAuthority,
      reconnectAttempt: 0,
      recoverability: 'none',
      retryConnection: this.retryConnection,
      nextReconnectAt: null,
      setAuthoritativeSequence: this.updateAuthoritativeSequence,
      synced: false,
      updateAuthoritativeSequence: this.updateAuthoritativeSequence,
      user: null,
    }
    this.document.on('update', this.handleEditorDocumentUpdate)
    this.transportDocument.on('update', this.handleTransportDocumentUpdate)
  }

  getSnapshot(): CollaborationDocumentHandle {
    return this.state
  }

  subscribe(listener: (state: CollaborationDocumentHandle) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  readonly readAuthority = (): CollaborationLiveAuthority => ({
    access: this.state.access,
    blockingFailure: this.state.blockingFailure,
    canEdit: this.state.canEdit,
    connectionStatus: this.state.connectionStatus,
    documentId: this.state.documentId,
    generation: this.state.generation,
    lifecycleStatus: this.state.lifecycleStatus,
    revision: this.state.authorityRevision,
    synced: this.state.synced,
  })

  readonly updateAuthoritativeSequence = (sequence: number): void => {
    if (!Number.isSafeInteger(sequence) || sequence < 0) {
      throw new Error('The authoritative collaboration sequence is invalid.')
    }
    if (sequence <= this.state.lastPersistedSequence) return
    this.publish({
      activityRevision: this.state.activityRevision + 1,
      lastPersistedSequence: sequence,
    })
  }

  readonly flushAndAwaitDurability = async (): Promise<void> => {
    if (this.destroyed) {
      throw new Error('The collaboration controller has already been disposed.')
    }
    this.flushScheduledBatch()
    if (this.durabilityFailed || this.blockingFailure !== null) {
      throw new Error(
        this.blockingFailure
          ?? this.state.error
          ?? 'Collaboration durability is unavailable.',
      )
    }
    if (!this.hasPendingDurability()) return
    await new Promise<void>((resolve, reject) => {
      this.durabilityWaiters.add({ reject, resolve })
    })
  }

  readonly flushAndAwaitDurable = async (): Promise<number> => {
    await this.flushAndAwaitDurability()
    return this.state.lastLocalDurableSequence
  }

  readonly retryConnection = async (): Promise<void> => {
    if (this.retryInFlight) return this.retryInFlight
    if (
      this.destroyed
      || this.state.recoverability !== 'retry'
      || this.durabilityFailed
    ) return
    const retry = this.performManualRetry().finally(() => {
      if (this.retryInFlight === retry) this.retryInFlight = null
    })
    this.retryInFlight = retry
    return retry
  }

  retain(): void {
    if (this.destroyed || !this.released) return
    this.released = false
    this.onReleaseFailed = null
    this.onReleasedSettled = null
    this.clearReleaseTimer()
    if (!this.durabilityFailed && this.blockingFailure === null) {
      this.attachEditorInput()
    }
    const canEdit = this.authenticated
      && this.synced
      && this.blockingFailure === null
      && (
        this.state.access === 'comment'
        || this.state.access === 'edit'
        || this.state.access === 'suggest'
      )
    this.publish({
      canEdit,
      synced: this.authenticated
        && this.synced
        && this.blockingFailure === null,
    })
    this.publishReadiness()
  }

  async start(): Promise<void> {
    if (this.started || this.destroyed) return
    this.started = true
    if (this.options.schemaVersion !== EDITOR_SCHEMA_VERSION) {
      this.enterIncompatible('The document schema is not supported by this client.')
      return
    }
    this.publish({
      connectionStatus: 'connecting',
      error: null,
      lifecycleStatus: 'connecting',
    })
    try {
      const session = await this.options.requestSession()
      if (!this.applySession(session, true)) return
      const providerAdapter = this.dependencies.createProvider({
        document: this.transportDocument,
        events: {
          onAuthenticated: this.handleAuthenticated,
          onAuthenticationFailed: this.handleAuthenticationFailed,
          onClose: this.handleClose,
          onStateless: this.handleStateless,
          onSynced: this.handleSynced,
        },
        getToken: () => this.leaseToken ?? '',
        room: session.room,
        url: this.options.resolveWebSocketUrl(session.websocket_path),
      })
      this.providerAdapter = providerAdapter
      this.setAwarenessUser(session.user)
      this.publish({ provider: providerAdapter.provider })
      await providerAdapter.connect()
    } catch (error) {
      this.handleSessionFailure(error, this.providerAdapter !== null)
    }
  }

  async refreshLease(mode: RefreshMode = 'periodic'): Promise<void> {
    if (this.refreshInFlight) return this.refreshInFlight
    if (this.destroyed || !this.leaseToken) return
    const refresh = this.performRefresh(mode).finally(() => {
      if (this.refreshInFlight === refresh) this.refreshInFlight = null
    })
    this.refreshInFlight = refresh
    return refresh
  }

  /** Push any batched local edits to the transport without releasing. The
   * linger path calls this at view exit so navigation never sits on a
   * half-batched update while the deferred teardown window runs. */
  flushPendingLocalUpdates(): void {
    if (this.destroyed) return
    this.flushScheduledBatch()
  }

  release(
    onSettled?: () => void,
    onFailed?: (error: string) => void,
  ): void {
    if (this.destroyed || this.released) return
    this.flushScheduledBatch()
    this.detachEditorInput()
    this.released = true
    this.onReleasedSettled = onSettled ?? null
    this.onReleaseFailed = onFailed ?? null
    this.publish({ canEdit: false, synced: false })
    if (!this.hasPendingDurability() && !this.durabilityFailed) {
      this.finalizeDestroy()
      return
    }
    if (this.durabilityFailed) {
      this.finalizeFailedRelease(
        this.state.error ?? 'Collaboration durability failed before navigation.',
      )
      return
    }
    this.releaseTimer = this.dependencies.scheduleTimer(
      this.handleReleaseTimeout,
      RELEASE_DURABILITY_TIMEOUT_MS,
    )
  }

  destroy(): void {
    this.release()
  }

  /**
   * Permanently dispose a lifecycle after its unconfirmed local state was
   * copied into a separate recovery artifact. This is intentionally distinct
   * from normal release, which waits for durable acknowledgement and retains a
   * failed controller for retry.
   */
  discardAfterRecoveryCapture(): void {
    this.finalizeDestroy()
  }

  private finalizeDestroy(): void {
    if (this.destroyed) return
    const onSettled = this.onReleasedSettled
    this.onReleaseFailed = null
    this.onReleasedSettled = null
    this.destroyed = true
    this.authenticated = false
    this.synced = false
    this.clearRefreshTimer()
    this.clearReconnectTimer()
    this.clearReleaseTimer()
    this.clearLocalUpdateBatch()
    this.detachEditorInput()
    this.transportDocument.off('update', this.handleTransportDocumentUpdate)
    this.providerAdapter?.destroy()
    this.providerAdapter = null
    this.document.destroy()
    this.transportDocument.destroy()
    this.listeners.clear()
    this.pendingHashes.clear()
    this.pendingUpdates.clear()
    this.earlyAcks.clear()
    this.rejectDurabilityWaiters(
      new Error('The collaboration controller was disposed.'),
    )
    onSettled?.()
  }

  private readonly performRefresh = async (mode: RefreshMode): Promise<void> => {
    const currentToken = this.leaseToken
    if (!currentToken || this.destroyed) return
    this.clearRefreshTimer()
    if (mode === 'reconnect') {
      this.publish({
        canEdit: false,
        connectionStatus: 'reconnecting',
        error: null,
        lifecycleStatus: 'reconnecting',
        nextReconnectAt: null,
        recoverability: 'retry',
      })
    } else {
      // Hocuspocus token sync is deliberately one-way: sendToken() resolves
      // after sending and the server updates the connection context without a
      // second "authenticated" response. Keep the already-authenticated,
      // already-synced transport ready while its lease rotates. Any HTTP,
      // send, or later server-close failure still enters the fail-closed
      // reconnect path below.
      this.publish({ error: null })
    }
    try {
      const rotationCommandId = this.rotationCommandId
        ?? this.dependencies.createCommandId()
      this.rotationCommandId = rotationCommandId
      const session = await this.options.requestSession(
        currentToken,
        rotationCommandId,
      )
      if (!this.applySession(session, false)) return
      this.rotationCommandId = null
      if (!this.providerAdapter) return
      if (mode === 'reconnect') {
        await this.providerAdapter.connect()
      } else {
        await this.providerAdapter.syncToken()
        this.publishReadiness()
      }
    } catch (error) {
      const status = requestStatus(error)
      if (
        (status === 401 || status === 403 || status === 404)
        && !this.destroyed
      ) {
        // A policy event invalidates the old lease for both a real revocation
        // and a still-authorized permission downgrade. Re-issuing once through
        // the cookie-authenticated session boundary is the only trustworthy
        // way to distinguish those cases: a downgrade returns a fresh
        // read-only lease, while a revocation remains a non-disclosing 404.
        this.rotationCommandId = null
        try {
          const session = await this.options.requestSession()
          if (!this.applySession(session, false) || !this.providerAdapter) return
          if (mode === 'reconnect') {
            await this.providerAdapter.connect()
          } else {
            await this.providerAdapter.syncToken()
            this.publishReadiness()
          }
          return
        } catch (recoveryError) {
          this.handleSessionFailure(recoveryError, true)
          return
        }
      }
      this.handleSessionFailure(error, true)
    }
  }

  private readonly performManualRetry = async (): Promise<void> => {
    this.clearReconnectTimer()
    this.publish({
      canEdit: false,
      connectionStatus: 'reconnecting',
      error: null,
      lifecycleStatus: 'reconnecting',
      nextReconnectAt: null,
      recoverability: 'retry',
      synced: false,
    })
    this.disconnectTransport()
    if (this.leaseToken && this.providerAdapter) {
      await this.refreshLease('reconnect')
      return
    }
    this.started = false
    await this.start()
  }

  private applySession(
    session: EditorCollaborationSession,
    initial: boolean,
  ): boolean {
    if (this.destroyed || this.durabilityFailed) return false
    if (
      session.protocol_version !== EDITOR_COLLABORATION_PROTOCOL_VERSION
      || session.schema_version !== this.options.schemaVersion
      || !session.lease_token
      || !session.room
      || !session.websocket_path
    ) {
      this.enterIncompatible('The collaboration protocol is not compatible with this client.')
      return false
    }
    const configuredSession = session as ConfiguredCollaborationSession
    const providerFlushMs = configuredSession.provider_flush_ms
    const refreshAfter = configuredSession.refresh_after
    if (
      providerFlushMs !== undefined
      && (!Number.isInteger(providerFlushMs)
        || providerFlushMs < MIN_PROVIDER_FLUSH_MS
        || providerFlushMs > MAX_PROVIDER_FLUSH_MS)
    ) {
      this.enterIncompatible('The collaboration flush interval is not compatible with this client.')
      return false
    }
    if (
      refreshAfter !== undefined
      && (!Number.isFinite(refreshAfter)
        || refreshAfter * 1_000 <= this.dependencies.now()
        || refreshAfter >= session.expires_at)
    ) {
      this.enterIncompatible('The collaboration refresh interval is not compatible with this client.')
      return false
    }
    if (!initial && this.room !== session.room) {
      this.enterIncompatible('The collaboration room changed while refreshing its lease.')
      return false
    }
    if (this.state.user && this.state.user.id !== session.user.id) {
      this.enterAccessRevoked('Collaboration identity changed while refreshing access.')
      return false
    }
    if (
      !initial
      && this.state.access !== 'view'
      && session.access === 'view'
      && this.hasUnreconciledUpdates()
    ) {
      this.enterAccessChangeWithUnconfirmedUpdates()
      return false
    }
    this.room = session.room
    this.leaseToken = session.lease_token
    this.providerFlushMs = providerFlushMs ?? COLLABORATION_UPDATE_BATCH_MS
    this.scheduleRefresh(session.expires_at, refreshAfter)
    const preserveCanEdit = (
      !initial
      && this.state.canEdit
      && (
        session.access === 'comment'
        || session.access === 'edit'
        || session.access === 'suggest'
      )
    )
    this.publish({
      access: session.access,
      canEdit: preserveCanEdit,
      connectionStatus: initial ? 'connecting' : this.state.connectionStatus,
      error: null,
      lifecycleStatus: initial ? 'connecting' : this.state.lifecycleStatus,
      user: deterministicUser(session.user),
    })
    return true
  }

  private readonly handleAuthenticated = (): void => {
    if (this.destroyed || this.durabilityFailed) return
    this.authenticated = true
    this.ignoreNextClose = false
    if (this.state.user) {
      this.setAwarenessUser(this.state.user)
    }
    this.publish({
      canEdit: false,
      connectionStatus: this.synced
        ? this.readyConnectionStatus()
        : 'connecting',
      error: null,
      lifecycleStatus: this.synced
        ? this.readyLifecycleStatus()
        : 'syncing',
      synced: this.synced,
    })
    this.reconcilePendingUpdates()
    this.publishReadiness()
  }

  private readonly handleSynced = (): void => {
    if (this.destroyed || this.durabilityFailed) return
    this.synced = true
    this.publishReadiness()
  }

  private readonly handleAuthenticationFailed = (reason: string): void => {
    if (this.destroyed) return
    this.authenticated = false
    this.synced = false
    if (isCompatibilityReason(reason)) {
      this.enterIncompatible('The collaboration protocol is not compatible with this client.')
      return
    }
    if (reason.includes('access_revoked')) {
      this.revalidateAccess()
      return
    }
    this.enterReconnect('Collaboration authentication failed; reconnecting read-only.')
  }

  private readonly handleClose = (code: number, reason: string): void => {
    if (this.destroyed) return
    this.authenticated = false
    this.synced = false
    this.replayedPendingHashes.clear()
    if (this.ignoreNextClose) {
      this.ignoreNextClose = false
      return
    }
    // The reason decides and the code only confirms. A close the server sends
    // in band reaches the provider with its code rewritten to 1000 and only
    // the reason preserved, so branching on the code first is blind to every
    // close the collaboration service itself initiates.
    if (isCompatibilityReason(reason) || code === 4409) {
      this.enterIncompatible('The collaboration protocol is not compatible with this client.')
      return
    }
    if (reason.includes('origin_rejected')) {
      // A transport misconfiguration, not a permission change: the address the
      // browser reached is not the one the server publishes as its public
      // origin. Terminal like any other incompatibility, but it must not be
      // reported as a revoked authorization or the operator searches the
      // sharing settings instead of the base URL.
      this.enterOriginRejected(
        'The collaboration server rejected this page origin. The address the browser '
        + 'uses does not match the configured public address of the server.',
      )
      return
    }
    if (reason.includes('suggestion_policy_violation')) {
      this.enterDurabilityFailure(
        'The proposed editor action was rejected by the collaboration policy. '
        + 'Your local document state is retained for diagnosis or backup.',
      )
      return
    }
    if (reason.includes('access_revoked') || code === 4403) {
      // Revalidate rather than declare: the authoritative revocation comes
      // from the access refresh answering 403/404, not from a close frame.
      this.revalidateAccess()
      return
    }
    if (reason.includes('message_too_large') || code === 1009) {
      this.enterDurabilityFailure(
        'The collaboration document exceeded the supported update size.',
      )
      return
    }
    this.enterReconnect('The collaboration connection was interrupted; reconnecting read-only.')
  }

  private readonly handleStateless = (payload: string): void => {
    if (this.destroyed) return
    let message: unknown
    try {
      message = JSON.parse(payload)
    } catch {
      return
    }
    if (isCollaborationDurableAck(message)) {
      if (this.durabilityFailed) return
      if (!this.pendingHashes.delete(message.hash)) {
        if (this.hashesInFlight > 0) this.earlyAcks.add(message.hash)
      }
      this.pendingUpdates.delete(message.hash)
      const lastLocalDurableSequence = Math.max(
        this.state.lastLocalDurableSequence,
        message.sequence,
      )
      this.publish({
        ...(lastLocalDurableSequence > this.state.lastLocalDurableSequence
          ? { activityRevision: this.state.activityRevision + 1 }
          : {}),
        durabilityStatus: this.durabilityFailed
          ? 'error'
          : this.hasPendingDurability() ? 'pending' : 'saved',
        lastLocalDurableSequence,
      })
      if (!this.hasPendingDurability()) {
        this.reconciliationNeeded = false
      }
      this.handleDurabilitySettled()
      return
    }
    if (
      isCollaborationCommentEvent(message, this.options.documentId)
    ) {
      this.publish({
        commentEventVersion: this.state.commentEventVersion + 1,
        ...(message.type === 'collaboration_comment_mentioned'
          ? {
              commentMentionEventVersion:
                this.state.commentMentionEventVersion + 1,
            }
          : {}),
      })
      return
    }
    if (isDurableRejection(message)) {
      this.enterDurabilityFailure(
        `A collaboration update was rejected (${message.code}).`,
      )
    }
  }

  private readonly handleEditorDocumentUpdate = (
    update: Uint8Array,
    origin: unknown,
  ): void => {
    if (
      this.destroyed
      || this.durabilityFailed
      || origin === REMOTE_EDITOR_ORIGIN
    ) return
    const suggestionUpdate = classifySuggestionUpdate(update)
    // Suggestion metadata starts a distinct server-policy transaction. This
    // keeps editor normalization or another direct edit out of the first
    // proposal while retaining normal batching for consecutive suggestion
    // text. A slash structure command adds a second boundary because the
    // server must inspect its short-lived `/query` insertion before Yjs can
    // garbage-collect it into the structure proposal.
    if (
      this.localUpdateBatch.length > 0
      && (
        suggestionUpdate.hasStructure
        || (
          suggestionUpdate.hasSuggestionBoundary
          && !this.localUpdateBatchHasSuggestionBoundary
        )
      )
    ) {
      if (this.localUpdateTimer !== null) {
        this.dependencies.cancelTimer(this.localUpdateTimer)
        this.localUpdateTimer = null
      }
      this.flushLocalUpdateBatch()
    }
    this.localUpdateBatch.push(update)
    this.localUpdateBatchHasSuggestionBoundary = (
      this.localUpdateBatchHasSuggestionBoundary
      || suggestionUpdate.hasSuggestionBoundary
    )
    this.publish({ durabilityStatus: 'pending' })
    if (this.localUpdateTimer !== null) return
    this.localUpdateTimer = this.dependencies.scheduleTimer(
      this.flushLocalUpdateBatch,
      this.providerFlushMs,
    )
  }

  private readonly handleTransportDocumentUpdate = (
    update: Uint8Array,
    origin: unknown,
  ): void => {
    if (
      this.destroyed
      || this.durabilityFailed
      || origin !== this.providerAdapter?.provider
    ) return
    // Hocuspocus 4.3 applies incoming sync messages with the provider itself
    // as origin. Keeping this identity check exact prevents remote changes
    // from entering the local durable-ack queue.
    Y.applyUpdate(this.document, update, REMOTE_EDITOR_ORIGIN)
    this.publish({ activityRevision: this.state.activityRevision + 1 })
  }

  private readonly flushLocalUpdateBatch = (): void => {
    this.localUpdateTimer = null
    if (
      this.destroyed
      || this.durabilityFailed
      || this.localUpdateBatch.length === 0
    ) return
    const update = Y.mergeUpdates(this.localUpdateBatch.splice(0))
    this.localUpdateBatchHasSuggestionBoundary = false
    let transportedUpdate: Uint8Array | null = null
    const captureTransportedUpdate = (
      emittedUpdate: Uint8Array,
      origin: unknown,
    ): void => {
      if (origin === LOCAL_TRANSPORT_ORIGIN) {
        transportedUpdate = Uint8Array.from(emittedUpdate)
      }
    }
    this.transportDocument.on('update', captureTransportedUpdate)
    try {
      Y.applyUpdate(this.transportDocument, update, LOCAL_TRANSPORT_ORIGIN)
    } finally {
      this.transportDocument.off('update', captureTransportedUpdate)
    }
    // The editor document may merge a local transaction with structs that
    // already arrived remotely on the transport document. Yjs removes those
    // redundant structs before emitting the actual provider update. Durable
    // acknowledgements hash that emitted payload, so tracking the pre-merge
    // input can leave a phantom pending hash forever after concurrent edits.
    if (transportedUpdate) {
      this.trackSentUpdate(transportedUpdate)
      return
    }
    this.publish({
      durabilityStatus: this.hasPendingDurability() ? 'pending' : 'saved',
    })
    this.handleDurabilitySettled()
  }

  private trackSentUpdate(update: Uint8Array): void {
    const retainedUpdate = Uint8Array.from(update)
    this.hashesInFlight += 1
    this.publish({ durabilityStatus: 'pending' })
    void this.dependencies.hashUpdate(update).then((hash) => {
      if (this.destroyed) return
      this.hashesInFlight = Math.max(0, this.hashesInFlight - 1)
      if (this.durabilityFailed) return
      if (this.earlyAcks.delete(hash)) {
        if (this.hashesInFlight === 0) this.earlyAcks.clear()
        this.publish({
          durabilityStatus: this.hasPendingDurability() ? 'pending' : 'saved',
        })
        this.handleDurabilitySettled()
        return
      }
      this.pendingHashes.add(hash)
      this.pendingUpdates.set(hash, retainedUpdate)
      if (this.hashesInFlight === 0) this.earlyAcks.clear()
      this.publish({ durabilityStatus: 'pending' })
      this.reconcilePendingUpdates()
      this.handleDurabilitySettled()
    }).catch(() => {
      if (this.destroyed) return
      this.hashesInFlight = Math.max(0, this.hashesInFlight - 1)
      if (this.hashesInFlight === 0) this.earlyAcks.clear()
      this.enterDurabilityFailure(
        'The collaboration update could not be tracked for durable storage.',
      )
    })
  }

  private handleSessionFailure(error: unknown, refresh: boolean): void {
    if (this.destroyed) return
    const status = requestStatus(error)
    const reason = requestReason(error)
    if (status === 409 || isCompatibilityReason(reason)) {
      this.enterIncompatible('The collaboration protocol or schema is not compatible.')
      return
    }
    if (status === 403 || status === 404 || reason === 'access_revoked') {
      this.enterAccessRevoked('Access to this collaboration document is unavailable.')
      return
    }
    if (status === 401) {
      this.clearRefreshTimer()
      this.clearReconnectTimer()
      this.disconnectTransport()
      this.publish({
        canEdit: false,
        connectionStatus: 'error',
        error: 'The collaboration session expired. Sign in again to continue.',
        lifecycleStatus: 'error',
        nextReconnectAt: null,
        recoverability: 'login',
        synced: false,
      })
      return
    }
    if (refresh) {
      this.enterReconnect('The collaboration lease could not be refreshed; reconnecting read-only.')
      return
    }
    // Transient initial-open failures (429 lease budget, network, 5xx) enter
    // the same backoff ladder rotations already use — an initial open that
    // fails on a rate limit self-heals within the lease TTL instead of
    // parking in a terminal error. Every other 4xx is a client defect and
    // stays loud and terminal.
    const transientOpenFailure =
      status === undefined || status === 429 || status >= 500
    if (transientOpenFailure) {
      this.enterReconnect(messageFromError(error))
      return
    }
    this.clearRefreshTimer()
    this.disconnectTransport()
    this.publish({
      canEdit: false,
      connectionStatus: 'error',
      error: messageFromError(error),
      lifecycleStatus: 'error',
      nextReconnectAt: null,
      recoverability: 'retry',
    })
  }

  private enterReconnect(error: string): void {
    if (this.destroyed) return
    this.reconciliationNeeded = this.reconciliationNeeded
      || this.hasUnreconciledUpdates()
    this.clearRefreshTimer()
    this.disconnectTransport()
    const reconnectAttempt = this.state.reconnectAttempt + 1
    const delayMs = collaborationReconnectDelayMs(
      reconnectAttempt,
      this.dependencies.random(),
    )
    const nextReconnectAt = this.dependencies.now() + delayMs
    this.publish({
      canEdit: false,
      connectionStatus: 'reconnecting',
      error,
      lifecycleStatus: 'reconnecting',
      nextReconnectAt,
      reconnectAttempt,
      recoverability: 'retry',
      synced: false,
    })
    this.clearReconnectTimer()
    this.reconnectTimer = this.dependencies.scheduleTimer(() => {
      this.reconnectTimer = null
      this.publish({ nextReconnectAt: null })
      if (this.leaseToken) {
        void this.refreshLease('reconnect')
        return
      }
      // The ladder can also carry a failed INITIAL open (no lease yet):
      // refreshLease would bail without a token, so re-run the open itself.
      this.started = false
      void this.start()
    }, delayMs)
  }

  private revalidateAccess(): void {
    if (this.hasUnreconciledUpdates()) {
      this.enterAccessChangeWithUnconfirmedUpdates()
      return
    }
    if (
      this.state.connectionStatus === 'reconnecting'
      && (this.reconnectTimer !== null || this.refreshInFlight !== null)
    ) return
    this.enterReconnect('Collaboration access changed; revalidating read-only.')
  }

  private enterAccessChangeWithUnconfirmedUpdates(): void {
    this.enterDurabilityFailure(
      'Collaboration access changed before local updates were durably confirmed. '
      + 'Your local document state is retained for recovery or backup.',
    )
  }

  private enterIncompatible(error: string): void {
    this.enterUnrecoverable('incompatible', error)
  }

  private enterOriginRejected(error: string): void {
    this.enterUnrecoverable('origin_rejected', error)
  }

  private enterUnrecoverable(
    connectionStatus: 'incompatible' | 'origin_rejected',
    error: string,
  ): void {
    const durabilityStatus = this.hasUnreconciledUpdates()
      ? 'error'
      : this.state.durabilityStatus
    this.durabilityFailed = true
    this.clearRefreshTimer()
    this.clearReconnectTimer()
    this.disconnectTransport()
    this.publish({
      canEdit: false,
      connectionStatus,
      durabilityStatus,
      error,
      lifecycleStatus: 'error',
      nextReconnectAt: null,
      recoverability: 'reload',
      synced: false,
    })
    this.rejectDurabilityWaiters(new Error(error))
    this.finalizeFailedRelease(error)
  }

  private enterAccessRevoked(error: string): void {
    const durabilityStatus = this.hasUnreconciledUpdates()
      ? 'error'
      : this.state.durabilityStatus
    this.durabilityFailed = true
    this.clearRefreshTimer()
    this.clearReconnectTimer()
    this.disconnectTransport()
    this.publish({
      access: 'view',
      canEdit: false,
      connectionStatus: 'access_revoked',
      durabilityStatus,
      error,
      lifecycleStatus: 'error',
      nextReconnectAt: null,
      recoverability: 'none',
      synced: false,
    })
    this.rejectDurabilityWaiters(new Error(error))
    this.finalizeFailedRelease(error)
  }

  private enterDurabilityFailure(error: string): void {
    this.durabilityFailed = true
    this.clearRefreshTimer()
    this.clearReconnectTimer()
    this.publish({
      canEdit: false,
      connectionStatus: 'error',
      durabilityStatus: 'error',
      error,
      lifecycleStatus: 'error',
      nextReconnectAt: null,
      recoverability: 'none',
      synced: false,
    })
    this.rejectDurabilityWaiters(new Error(error))
    this.disconnectTransport()
    this.finalizeFailedRelease(error)
  }

  private scheduleRefresh(expiresAt: number, refreshAfter?: number): void {
    this.clearRefreshTimer()
    this.refreshTimer = this.dependencies.scheduleTimer(() => {
      this.refreshTimer = null
      void this.refreshLease('periodic')
    }, refreshAfter === undefined
      ? leaseRefreshDelayMs(expiresAt, this.dependencies.now())
      : Math.max(1_000, Math.floor(
          refreshAfter * 1_000 - this.dependencies.now(),
        )))
  }

  private clearRefreshTimer(): void {
    if (this.refreshTimer === null) return
    this.dependencies.cancelTimer(this.refreshTimer)
    this.refreshTimer = null
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer === null) return
    this.dependencies.cancelTimer(this.reconnectTimer)
    this.reconnectTimer = null
  }

  private clearReleaseTimer(): void {
    if (this.releaseTimer === null) return
    this.dependencies.cancelTimer(this.releaseTimer)
    this.releaseTimer = null
  }

  private clearLocalUpdateBatch(): void {
    if (this.localUpdateTimer !== null) {
      this.dependencies.cancelTimer(this.localUpdateTimer)
      this.localUpdateTimer = null
    }
    this.localUpdateBatch.splice(0)
    this.localUpdateBatchHasSuggestionBoundary = false
  }

  private flushScheduledBatch(): void {
    if (this.localUpdateTimer !== null) {
      this.dependencies.cancelTimer(this.localUpdateTimer)
      this.localUpdateTimer = null
    }
    this.flushLocalUpdateBatch()
  }

  private attachEditorInput(): void {
    if (this.editorInputAttached) return
    this.document.on('update', this.handleEditorDocumentUpdate)
    this.editorInputAttached = true
  }

  private detachEditorInput(): void {
    if (!this.editorInputAttached) return
    this.document.off('update', this.handleEditorDocumentUpdate)
    this.editorInputAttached = false
  }

  private disconnectTransport(): void {
    if (!this.providerAdapter) return
    this.authenticated = false
    this.synced = false
    this.replayedPendingHashes.clear()
    this.ignoreNextClose = true
    this.providerAdapter.disconnect()
  }

  private reconcilePendingUpdates(): void {
    if (
      !this.reconciliationNeeded
      || !this.authenticated
      || !this.providerAdapter
      || this.pendingHashes.size === 0
    ) return
    try {
      this.providerAdapter.sendStateless(JSON.stringify({
        type: 'durability_reconcile',
        hashes: [...this.pendingHashes],
      }))
      for (const [hash, update] of this.pendingUpdates) {
        if (this.replayedPendingHashes.has(hash)) continue
        this.providerAdapter.replayUpdate(Uint8Array.from(update))
        this.replayedPendingHashes.add(hash)
      }
    } catch {
      this.enterReconnect(
        'Collaboration durability could not be reconciled; reconnecting read-only.',
      )
    }
  }

  private handleDurabilitySettled(): void {
    if (this.destroyed || this.hasPendingDurability()) return
    this.reconciliationNeeded = false
    this.resolveDurabilityWaiters()
    if (this.blockingFailure !== null && !this.durabilityFailed) {
      this.blockingFailure = null
      if (!this.released) this.attachEditorInput()
      this.publish({
        blockingFailure: null,
        durabilityStatus: 'saved',
        error: null,
      })
      this.publishReadiness()
    }
    if (this.released && !this.durabilityFailed) this.finalizeDestroy()
  }

  private finalizeFailedRelease(error: string): void {
    if (!this.released || this.destroyed) return
    this.clearReleaseTimer()
    if (this.hasPendingDurability()) {
      this.markBlockingFailure(error)
      return
    }
    this.finalizeDestroy()
  }

  private readonly handleReleaseTimeout = (): void => {
    this.releaseTimer = null
    if (this.destroyed || !this.released || !this.hasPendingDurability()) return
    const error = 'Collaboration changes could not be confirmed before navigation completed.'
    this.markBlockingFailure(error)
  }

  private markBlockingFailure(error: string): void {
    if (this.blockingFailure !== null) return
    this.blockingFailure = error
    this.publish({
      blockingFailure: error,
      canEdit: false,
      connectionStatus: 'error',
      durabilityStatus: 'error',
      error,
      lifecycleStatus: 'error',
      nextReconnectAt: null,
      recoverability: 'none',
      synced: false,
    })
    this.rejectDurabilityWaiters(new Error(error))
    this.onReleaseFailed?.(error)
  }

  private publishReadiness(): void {
    if (
      !this.authenticated
      || !this.synced
      || this.destroyed
      || this.durabilityFailed
      || this.blockingFailure !== null
    ) return
    const canEdit = !this.released
      && (
        this.state.access === 'comment'
        || this.state.access === 'edit'
        || this.state.access === 'suggest'
      )
    this.publish({
      canEdit,
      connectionStatus: this.readyConnectionStatus(),
      error: null,
      lifecycleStatus: this.readyLifecycleStatus(),
      nextReconnectAt: null,
      reconnectAttempt: 0,
      recoverability: 'none',
      synced: true,
    })
  }

  private readyConnectionStatus(): EditorCollaborationConnectionStatus {
    return this.state.access === 'view' ? 'read_only' : 'connected'
  }

  private readyLifecycleStatus(): CollaborationDocumentHandle['lifecycleStatus'] {
    return this.state.access === 'view' ? 'read_only' : 'saved'
  }

  private resolveDurabilityWaiters(): void {
    const waiters = [...this.durabilityWaiters]
    this.durabilityWaiters.clear()
    for (const waiter of waiters) waiter.resolve()
  }

  private rejectDurabilityWaiters(error: Error): void {
    const waiters = [...this.durabilityWaiters]
    this.durabilityWaiters.clear()
    for (const waiter of waiters) waiter.reject(error)
  }

  private setAwarenessUser(user: EditorCollaborationUser): void {
    const deterministic = deterministicUser(user)
    const key = `${deterministic.color}\u0000${deterministic.id}\u0000${deterministic.name}`
    if (this.awarenessUserKey === key) return
    this.providerAdapter?.setAwarenessUser(deterministic)
    this.awarenessUserKey = key
  }

  private hasPendingDurability(): boolean {
    return this.localUpdateBatch.length > 0
      || this.hashesInFlight > 0
      || this.pendingHashes.size > 0
  }

  private hasUnreconciledUpdates(): boolean {
    return this.hasPendingDurability()
  }

  private publish(patch: Partial<CollaborationDocumentHandle>): void {
    if (this.destroyed) return
    const nextState = {
      ...this.state,
      ...patch,
      hasUnconfirmedLocalChanges: this.hasPendingDurability(),
      pendingHashes: [...this.pendingHashes],
    }
    this.state = {
      ...nextState,
      authorityRevision: collaborationAuthorityChanged(this.state, nextState)
        ? this.state.authorityRevision + 1
        : this.state.authorityRevision,
    }
    for (const listener of this.listeners) listener(this.state)
  }
}

export function containsStructureSuggestionAttribute(
  update: Uint8Array,
): boolean {
  return classifySuggestionUpdate(update).hasStructure
}

export function containsSuggestionBoundary(update: Uint8Array): boolean {
  return classifySuggestionUpdate(update).hasSuggestionBoundary
}

function classifySuggestionUpdate(update: Uint8Array): {
  hasStructure: boolean
  hasSuggestionBoundary: boolean
} {
  try {
    let hasStructure = false
    let hasSuggestionBoundary = false
    for (const struct of Y.decodeUpdate(update).structs) {
      if (!(struct instanceof Y.Item)) continue
      const structure = (
        struct.parentSub === INQTRIX_STRUCTURE_SUGGESTION_ATTR
        || (
          struct.content instanceof Y.ContentAny
          && struct.content.getContent().length === 1
          && isStructureSuggestionData(struct.content.getContent()[0])
        )
      )
      if (structure) {
        hasStructure = true
        hasSuggestionBoundary = true
      }
      if (
        struct.content instanceof Y.ContentFormat
        && SUGGESTION_MARK_NAMES.has(struct.content.key)
      ) {
        hasSuggestionBoundary = true
      }
    }
    return { hasStructure, hasSuggestionBoundary }
  } catch {
    return { hasStructure: false, hasSuggestionBoundary: false }
  }
}

export function acquireLifecycleController(
  key: string,
  identity: string | null,
  create: () => CollaborationDocumentController,
): CollaborationDocumentController {
  const existing = lifecycleRegistry.get(key)
  if (existing && (existing.references > 0 || existing.identity === identity)) {
    lifecycleFailures.delete(key)
    clearLinger(existing)
    existing.references += 1
    existing.controller.retain()
    return existing.controller
  }
  if (existing) {
    // A lingering controller whose request identity went stale (api key
    // changed under the same registry key) would keep signing rotations with
    // the old closure. Tear it down and start fresh.
    finalizeEntryRelease(key, existing)
  }
  const controller = create()
  lifecycleRegistry.set(key, {
    controller,
    identity,
    lingerSince: null,
    lingerTimer: null,
    references: 1,
  })
  releaseSupersededGenerations(key)
  return controller
}

export function releaseLifecycleController(
  key: string,
  controller: CollaborationDocumentController,
): void {
  const entry = lifecycleRegistry.get(key)
  if (!entry || entry.controller !== controller) {
    controller.release()
    return
  }
  entry.references = Math.max(0, entry.references - 1)
  if (entry.references > 0) return
  if (!controllerCanLinger(controller)) {
    finalizeEntryRelease(key, entry)
    return
  }
  controller.flushPendingLocalUpdates()
  entry.lingerSince = Date.now()
  const timer = setTimeout(() => {
    entry.lingerTimer = null
    finalizeEntryRelease(key, entry)
  }, LIFECYCLE_LINGER_MS)
  // In node (tests) a pending linger timer must not pin the process open.
  if (typeof timer === 'object' && 'unref' in timer) timer.unref()
  entry.lingerTimer = timer
  enforceLingerCap()
}

/** Immediately tear down every lifecycle that is only lingering (no mounted
 * surface). An identity boundary that does not hard-reload the document must
 * not keep rotating leases for the previous identity; the test suite uses it
 * to keep the module-level registry hermetic between cases. */
export function destroyLingeringCollaborationLifecycles(): void {
  for (const [key, entry] of [...lifecycleRegistry.entries()]) {
    if (entry.references > 0 || entry.lingerTimer === null) continue
    finalizeEntryRelease(key, entry)
  }
}

/** Linger only sessions that are healthy or already self-healing. Terminal
 * states (revoked, incompatible, hard error) must NOT be preserved across a
 * re-entry: a fresh session request is the only path that can observe a
 * re-grant or a fixed deployment. */
function controllerCanLinger(
  controller: CollaborationDocumentController,
): boolean {
  const snapshot = controller.getSnapshot()
  return (
    snapshot.blockingFailure === null
    && snapshot.durabilityStatus !== 'error'
    && snapshot.connectionStatus !== 'access_revoked'
    && snapshot.connectionStatus !== 'error'
    && snapshot.connectionStatus !== 'incompatible'
    && snapshot.connectionStatus !== 'origin_rejected'
  )
}

function clearLinger(entry: LifecycleRegistryEntry): void {
  if (entry.lingerTimer !== null) clearTimeout(entry.lingerTimer)
  entry.lingerTimer = null
  entry.lingerSince = null
}

/** The pre-linger release path: hand the controller to release(), which
 * destroys immediately when durability is settled and otherwise waits for
 * the acknowledgement (the entry stays registered so a remount can still
 * retain it during that wait). */
function finalizeEntryRelease(
  key: string,
  entry: LifecycleRegistryEntry,
): void {
  clearLinger(entry)
  entry.controller.release(
    () => {
      const current = lifecycleRegistry.get(key)
      if (current === entry && current.references === 0) {
        lifecycleRegistry.delete(key)
        lifecycleFailures.delete(key)
      }
    },
    (error) => recordLifecycleFailure(key, error),
  )
}

function enforceLingerCap(): void {
  const lingering = [...lifecycleRegistry.entries()]
    .filter(([, entry]) => entry.references === 0 && entry.lingerTimer !== null)
  if (lingering.length <= MAX_LINGERING_LIFECYCLES) return
  lingering.sort(
    (a, b) => (a[1].lingerSince ?? 0) - (b[1].lingerSince ?? 0),
  )
  for (const [key, entry] of lingering
    .slice(0, lingering.length - MAX_LINGERING_LIFECYCLES)) {
    finalizeEntryRelease(key, entry)
  }
}

/** A new generation supersedes every lingering lifecycle of the same
 * document: their leases and rooms belong to a rebuilt history and can only
 * rot into incompatibility. Mounted surfaces (references > 0) are left to
 * the recovery flow. */
function releaseSupersededGenerations(key: string): void {
  const marker = key.lastIndexOf(':g')
  if (marker === -1) return
  const prefix = key.slice(0, marker + 2)
  for (const [candidateKey, entry] of [...lifecycleRegistry.entries()]) {
    if (candidateKey === key || !candidateKey.startsWith(prefix)) continue
    if (entry.references > 0) continue
    finalizeEntryRelease(candidateKey, entry)
  }
}

export function retireCollaborationDocumentLifecycle({
  documentId,
  generation,
  workspaceId,
}: {
  documentId: string
  generation: number
  workspaceId: string
}): boolean {
  const key = `${workspaceId}:${documentId}:g${generation}`
  const entry = lifecycleRegistry.get(key)
  if (!entry) return false
  clearLinger(entry)
  lifecycleRegistry.delete(key)
  lifecycleFailures.delete(key)
  entry.controller.discardAfterRecoveryCapture()
  return true
}

export function collaborationDocumentLifecycleHasUnconfirmedChanges({
  documentId,
  generation,
  workspaceId,
}: {
  documentId: string
  generation: number
  workspaceId: string
}): boolean {
  return lifecycleRegistry
    .get(`${workspaceId}:${documentId}:g${generation}`)
    ?.controller.getSnapshot().hasUnconfirmedLocalChanges === true
}

function recordLifecycleFailure(key: string, error: string): void {
  const now = Date.now()
  for (const [candidate, failure] of lifecycleFailures) {
    if (failure.expiresAt <= now) lifecycleFailures.delete(candidate)
  }
  if (lifecycleFailures.size >= MAX_RELEASE_FAILURES) {
    const oldest = lifecycleFailures.keys().next().value
    if (oldest !== undefined) lifecycleFailures.delete(oldest)
  }
  lifecycleFailures.set(key, {
    error,
    expiresAt: now + RELEASE_FAILURE_TTL_MS,
  })
}

export function consumeLifecycleFailure(key: string): string | null {
  const failure = lifecycleFailures.get(key)
  lifecycleFailures.delete(key)
  if (!failure || failure.expiresAt <= Date.now()) return null
  return failure.error
}

export type UseCollaborationDocumentOptions = {
  active: boolean
  apiKey: string | undefined
  document: EditorDocumentRecord | null
  workspaceId: string
}

export function useCollaborationDocument({
  active,
  apiKey,
  document,
  workspaceId,
}: UseCollaborationDocumentOptions): CollaborationDocumentHandle {
  const [state, setState] = useState<CollaborationDocumentHandle>(() => inactiveHandle(null, null))
  const documentId = document?.id ?? null
  const generation = document?.collaboration?.generation ?? null
  const persistedSequence = document?.collaboration?.persistedSequence ?? 0
  const schemaVersion = document?.collaboration?.schemaVersion ?? null
  const collaborationActive = active
    && document?.contentMode === 'collaboration'
    && generation !== null
    && schemaVersion !== null
  const requestedInactiveHandle = useMemo(
    () => inactiveHandle(documentId, generation),
    [documentId, generation],
  )

  useEffect(() => {
    if (!collaborationActive || !documentId || generation === null || schemaVersion === null) {
      setState(requestedInactiveHandle)
      return
    }
    const lifecycleRegistryKey = `${workspaceId}:${documentId}:g${generation}`
    const controller = acquireLifecycleController(
      lifecycleRegistryKey,
      apiKey ?? null,
      () => new CollaborationDocumentController({
        documentId,
        generation,
        initialPersistedSequence: persistedSequence,
        requestSession: (leaseToken, rotationCommandId) => createEditorCollaborationSession(
          documentId,
          {
            protocol_version: EDITOR_COLLABORATION_PROTOCOL_VERSION,
            schema_version: EDITOR_SCHEMA_VERSION,
            ...(leaseToken === undefined ? {} : { lease_token: leaseToken }),
            ...(rotationCommandId === undefined
              ? {}
              : { rotation_command_id: rotationCommandId }),
          },
          { apiKey, workspaceId },
        ),
        resolveWebSocketUrl: collaborationWebSocketUrl,
        schemaVersion,
      }),
    )
    const unsubscribe = controller.subscribe(setState)
    setState(controller.getSnapshot())
    void controller.start()
    return () => {
      unsubscribe()
      releaseLifecycleController(lifecycleRegistryKey, controller)
    }
  }, [
    active,
    apiKey,
    collaborationActive,
    documentId,
    generation,
    schemaVersion,
    requestedInactiveHandle,
    workspaceId,
  ])

  return collaborationHandleForRequestedDocument(
    state,
    documentId,
    generation,
    collaborationActive,
    requestedInactiveHandle,
  )
}

export function useGuestCollaborationDocument({
  access,
  active,
}: {
  access: EditorGuestAccessSession | null
  active: boolean
}): CollaborationDocumentHandle {
  const documentId = access?.document.id ?? null
  const generation = access?.document.generation ?? null
  const persistedSequence = access?.document.persisted_sequence ?? 0
  const guestId = access?.guest.id ?? null
  const displayName = access?.guest.display_name ?? undefined
  const collaborationActive = (
    active
    && documentId !== null
    && generation !== null
    && guestId !== null
  )
  const requestedInactiveHandle = useMemo(
    () => inactiveHandle(documentId, generation),
    [documentId, generation],
  )
  const [state, setState] = useState<CollaborationDocumentHandle>(
    () => requestedInactiveHandle,
  )

  useEffect(() => {
    if (
      !collaborationActive
      || documentId === null
      || generation === null
      || guestId === null
    ) {
      setState(requestedInactiveHandle)
      return
    }
    const lifecycleRegistryKey = (
      `guest:${guestId}:${documentId}:g${generation}`
    )
    const controller = acquireLifecycleController(
      lifecycleRegistryKey,
      null,
      () => new CollaborationDocumentController({
        documentId,
        generation,
        initialPersistedSequence: persistedSequence,
        requestSession: (leaseToken, rotationCommandId) => (
          createGuestEditorCollaborationSession({
            protocol_version: EDITOR_COLLABORATION_PROTOCOL_VERSION,
            schema_version: EDITOR_SCHEMA_VERSION,
            ...(displayName === undefined
              ? {}
              : { display_name: displayName }),
            ...(leaseToken === undefined
              ? {}
              : { lease_token: leaseToken }),
            ...(rotationCommandId === undefined
              ? {}
              : { rotation_command_id: rotationCommandId }),
          })
        ),
        resolveWebSocketUrl: collaborationWebSocketUrl,
        schemaVersion: EDITOR_SCHEMA_VERSION,
      }),
    )
    const unsubscribe = controller.subscribe(setState)
    setState(controller.getSnapshot())
    void controller.start()
    return () => {
      unsubscribe()
      releaseLifecycleController(lifecycleRegistryKey, controller)
    }
  }, [
    collaborationActive,
    displayName,
    documentId,
    generation,
    guestId,
    persistedSequence,
    requestedInactiveHandle,
  ])

  return collaborationHandleForRequestedDocument(
    state,
    documentId,
    generation,
    collaborationActive,
    requestedInactiveHandle,
  )
}

/** Never expose a retained lifecycle snapshot under a newly requested
 * document while React is waiting to run the switch effect. */
export function collaborationHandleForRequestedDocument(
  handle: CollaborationDocumentHandle,
  documentId: string | null,
  generation: number | null,
  active: boolean,
  inactive = inactiveHandle(documentId, generation),
): CollaborationDocumentHandle {
  if (
    active
    && documentId !== null
    && generation !== null
    && handle.documentId === documentId
    && handle.generation === generation
  ) return handle
  return inactive
}

export function collaborationWebSocketUrl(
  path: string,
  location: Pick<Location, 'host' | 'protocol'> = window.location,
): string {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${protocol}//${location.host}${normalizedPath}`
}

export function leaseRefreshDelayMs(expiresAt: number, nowMs: number): number {
  const remainingMs = expiresAt * 1_000 - nowMs
  return Math.max(1_000, Math.floor(remainingMs / 2))
}

export function collaborationReconnectDelayMs(
  attempt: number,
  randomValue: number,
): number {
  const safeAttempt = Math.max(1, Math.floor(attempt))
  const baseDelay = COLLABORATION_RECONNECT_DELAYS_MS[
    Math.min(safeAttempt - 1, COLLABORATION_RECONNECT_DELAYS_MS.length - 1)
  ] ?? COLLABORATION_RECONNECT_DELAYS_MS.at(-1)!
  const normalizedRandom = Number.isFinite(randomValue)
    ? Math.min(1, Math.max(0, randomValue))
    : 0.5
  const jitteredDelay = Math.round(baseDelay * (0.85 + normalizedRandom * 0.3))
  return Math.min(COLLABORATION_RECONNECT_DELAYS_MS.at(-1)!, jitteredDelay)
}

export const COLLABORATION_AWARENESS_THROTTLE_MS = 75

export function createHocuspocusProvider({
  document,
  events,
  getToken,
  room,
  url,
}: CollaborationProviderFactoryOptions): CollaborationProviderAdapter {
  const websocket = new HocuspocusProviderWebsocket({
    autoConnect: false,
    onClose: () => {
      // Lease rotation belongs to the controller; suppress the websocket's
      // built-in retry so it cannot reconnect with a stale lease first.
      websocket.shouldConnect = false
    },
    url,
  })
  const provider = new HocuspocusProvider({
    document,
    name: room,
    onAuthenticated: events.onAuthenticated,
    onAuthenticationFailed: ({ reason }) => events.onAuthenticationFailed(reason),
    onClose: ({ event }) => events.onClose(event.code, event.reason),
    onStateless: ({ payload }) => events.onStateless(payload),
    onSynced: ({ state }) => {
      if (state) events.onSynced()
    },
    token: getToken,
    websocketProvider: websocket,
  })
  // With an externally managed websocketProvider, Hocuspocus 4.x does NOT
  // auto-attach the provider (manageSocket=false skips attach() in the
  // constructor). Without this attach the provider never registers on the
  // socket, its onOpen never fires, and the authentication token is NEVER
  // sent. The connection would remain unauthenticated until teardown and
  // every participant, including the owner, would stay read-only. Unit
  // fakes replace this factory entirely, so this integration contract needs
  // a test that drives the real provider.
  provider.attach()
  // Hocuspocus 4.3's default awareness handler re-broadcasts EVERY changed
  // client — including states just RECEIVED from the server (no origin
  // filter). The sidecar's identity gate rejects any client message that
  // carries more than the sender's own state (close 4403, seen live with
  // two participants + a stale clientID after a reload) and would rewrite
  // a foreign state to the sender's identity otherwise. Forward only OUR
  // OWN awareness changes; the server relays everyone else authoritatively.
  const awareness = provider.awareness
  let disposeAwarenessForwarding = () => undefined
  if (awareness) {
    awareness.off('update', provider.boundAwarenessUpdateHandler)
    let pendingOwnUpdate = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const flushOwnUpdate = () => {
      timer = null
      if (!pendingOwnUpdate) return
      pendingOwnUpdate = false
      const ownClientId = document.clientID
      const removed = awareness.getLocalState() === null
      provider.awarenessUpdateHandler({
        added: [],
        removed: removed ? [ownClientId] : [],
        updated: removed ? [] : [ownClientId],
      }, 'local')
    }
    const handleAwarenessUpdate = (
      changes: { added: number[]; removed: number[]; updated: number[] },
      origin: unknown,
    ) => {
      // Incoming server messages are applied with the provider as origin.
      // Sending them back can form an awareness echo, including for our own
      // clientID when the server normalized the state.
      if (origin === provider) return
      const ownClientId = document.clientID
      if (
        !changes.added.includes(ownClientId)
        && !changes.removed.includes(ownClientId)
        && !changes.updated.includes(ownClientId)
      ) return
      pendingOwnUpdate = true
      if (timer === null) {
        timer = setTimeout(flushOwnUpdate, COLLABORATION_AWARENESS_THROTTLE_MS)
      }
    }
    awareness.on('update', handleAwarenessUpdate)
    disposeAwarenessForwarding = () => {
      awareness.off('update', handleAwarenessUpdate)
      pendingOwnUpdate = false
      if (timer !== null) {
        clearTimeout(timer)
        timer = null
      }
    }
  }
  return {
    connect: async () => {
      await websocket.connect()
    },
    destroy: () => {
      disposeAwarenessForwarding()
      provider.destroy()
      websocket.destroy()
    },
    disconnect: () => websocket.disconnect(),
    provider,
    replayUpdate: (update) => provider.documentUpdateHandler(update, LOCAL_TRANSPORT_ORIGIN),
    sendStateless: (payload) => provider.sendStateless(payload),
    setAwarenessUser: (user) => provider.setAwarenessField('user', user),
    syncToken: () => provider.sendToken(),
  }
}

const browserDependencies: CollaborationDocumentControllerDependencies = {
  cancelTimer: (timer) => window.clearTimeout(timer),
  createCommandId: collaborationCommandId,
  createDocument: () => new Y.Doc(),
  createProvider: createHocuspocusProvider,
  hashUpdate: collaborationSha256Hex,
  now: () => Date.now(),
  random: secureRandomUnitInterval,
  scheduleTimer: (callback, delayMs) => window.setTimeout(callback, delayMs),
}

function inactiveHandle(
  documentId: string | null,
  generation: number | null,
): CollaborationDocumentHandle {
  const authority: CollaborationLiveAuthority = {
    access: null,
    blockingFailure: null,
    canEdit: false,
    connectionStatus: 'inactive',
    documentId,
    generation,
    lifecycleStatus: 'inactive',
    revision: 0,
    synced: false,
  }
  return {
    access: null,
    activityRevision: 0,
    authorityRevision: 0,
    blockingFailure: null,
    canEdit: false,
    commentEventVersion: 0,
    commentMentionEventVersion: 0,
    connectionStatus: 'inactive',
    document: null,
    documentId,
    durabilityStatus: 'idle',
    error: null,
    flushAndAwaitDurability: async () => undefined,
    lastPersistedSequence: 0,
    lastLocalDurableSequence: 0,
    generation,
    hasUnconfirmedLocalChanges: false,
    lifecycleStatus: 'inactive',
    lifecycleKey: 'inactive',
    pendingHashes: [],
    provider: null,
    readAuthority: () => authority,
    reconnectAttempt: 0,
    recoverability: 'none',
    retryConnection: async () => undefined,
    nextReconnectAt: null,
    setAuthoritativeSequence: () => undefined,
    synced: false,
    flushAndAwaitDurable: async () => 0,
    updateAuthoritativeSequence: () => undefined,
    user: null,
  }
}

function collaborationAuthorityChanged(
  previous: CollaborationDocumentHandle,
  next: CollaborationDocumentHandle,
): boolean {
  return previous.access !== next.access
    || previous.blockingFailure !== next.blockingFailure
    || previous.canEdit !== next.canEdit
    || previous.connectionStatus !== next.connectionStatus
    || previous.documentId !== next.documentId
    || previous.generation !== next.generation
    || previous.lifecycleStatus !== next.lifecycleStatus
    || previous.synced !== next.synced
}

function deterministicUser(user: EditorCollaborationUser): EditorCollaborationUser {
  return { color: user.color, id: user.id, name: user.name }
}

function secureRandomUnitInterval(): number {
  const value = globalThis.crypto.getRandomValues(new Uint32Array(1))[0] ?? 0
  return value / 0x1_0000_0000
}

function requestStatus(error: unknown): number | undefined {
  return error instanceof Error
    ? (error as InqtrixRequestError).status
    : undefined
}

function requestReason(error: unknown): string {
  if (!(error instanceof Error)) return ''
  const reason = (error as InqtrixRequestError).detail?.reason
  return typeof reason === 'string' ? reason : ''
}

function isCompatibilityReason(reason: string): boolean {
  // The reasons the collaboration service and the gateway actually emit for
  // "this client cannot proceed without changing": see CollaborationErrorReason
  // in apps/collaboration-server/src/errors.ts.
  return reason.includes('update_required')
    || reason.includes('invalid_schema')
    || reason.includes('generation_mismatch')
    || reason.includes('binary_frames_required')
}

function messageFromError(error: unknown): string {
  return error instanceof Error
    ? error.message
    : 'Collaboration could not be started.'
}

function isCollaborationCommentEvent(
  value: unknown,
  documentId: string,
): value is {
  document_id: string
  type: 'collaboration_comment_changed' | 'collaboration_comment_mentioned'
} {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Record<string, unknown>
  return candidate.document_id === documentId
    && (
      candidate.type === 'collaboration_comment_changed'
      || candidate.type === 'collaboration_comment_mentioned'
    )
}

function isDurableRejection(value: unknown): value is {
  code: string
  hash: string
  type: 'durable_rejection'
} {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Record<string, unknown>
  return candidate.type === 'durable_rejection'
    && typeof candidate.code === 'string'
    && typeof candidate.hash === 'string'
}
