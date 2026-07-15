import {
  editorYDocToJson,
  getEditorSchemaFingerprint,
  validateCanonicalYjsV1Update,
  validateSuggestionYjsUpdate,
  type CollaborationActorKind,
  type CollaborationChangeKind,
  type CollaborationDurableAck,
  type SuggestionDescriptor,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import type {
  CollaborationApi,
  CollaborationSettings,
  ConnectionContext,
  SidecarLogger,
} from './contracts'
import {
  cloneDocument,
  hashBytes,
  reconstructDocument,
  validateDocument,
} from './documentState'
import { collaborationError, CloseCodes, CollaborationError } from './errors'
import { InstanceLeaseManager } from './instanceLease'
import { SidecarMetrics } from './metrics'
import {
  deriveSuggestionPatchStates,
  validateSuggestionUpdate,
} from './suggestPolicy'

type Release = () => void

type RoomPhase = 'loading' | 'ready' | 'awaiting_apply' | 'reconstruction_required'

type RoomState = {
  authoritativeSequence: number
  gate: SerialGate
  initialized: boolean
  loaded: boolean
  phase: RoomPhase
  tailBytes: number
  updatesSinceSnapshot: number
}

type PendingUpdate = {
  expectedStateHash: string
  hash: string
  persistedSequence: number
  release: Release
  room: string
  sequence: number
  startedAt: number
  updateBytes: number
  wasDuplicate: boolean
}

type PreparingUpdate = {
  aborted: boolean
  persistenceAttempted: boolean
  room: string
}

export type PrepareClientUpdateInput = {
  allowNoop: boolean
  connectionId: string
  context: ConnectionContext
  document: Y.Doc
  room: string
  update: Uint8Array
}

export type CapturedDocument = {
  document: Y.Doc
  sequence: number
}

export type ServerMutationInput = {
  actorKind: CollaborationActorKind
  actorUserId: string
  changeKind: Extract<CollaborationChangeKind, 'decision' | 'suggestion'>
  commandId: string
  commandPayloadHash: string
  decision: 'accept' | 'reject' | null
  document: Y.Doc
  documentId: string
  expectedSequence: number
  generation: number
  mutate: (clone: Y.Doc) => {
    patchIds: string[]
    suggestions: SuggestionDescriptor[]
    suggestionIds: string[]
  }
  requestedPatchIds: string[]
  room: string
}

export type ServerMutationResult = {
  ack: CollaborationDurableAck
  patchIds: string[]
  suggestionIds: string[]
}

export type DocumentCoordinatorCallbacks = {
  onAuthoritativeApplyFailure?: (room: string) => void
  onAuthoritativeApplySuccess?: (room: string) => void
}

export class DocumentCoordinator {
  private readonly pending = new Map<string, PendingUpdate>()
  private readonly preparing = new Map<string, PreparingUpdate>()
  private readonly rooms = new Map<string, RoomState>()

  constructor(
    private readonly api: CollaborationApi,
    private readonly leaseManager: InstanceLeaseManager,
    private readonly settings: CollaborationSettings,
    private readonly logger: SidecarLogger,
    private readonly metrics: SidecarMetrics,
    private readonly callbacks: DocumentCoordinatorCallbacks = {},
  ) {}

  initialize(
    room: string,
    persistedSequence: number,
    tail: { bytes: number; updates: number } = { bytes: 0, updates: 0 },
  ): void {
    const state = this.room(room)
    if (
      state.phase === 'awaiting_apply'
      || (state.phase === 'reconstruction_required' && state.loaded)
      || (state.initialized && state.authoritativeSequence !== persistedSequence)
    ) {
      throw new CollaborationError('internal_consistency', {
        closeCode: CloseCodes.internalConsistency,
      })
    }
    state.authoritativeSequence = persistedSequence
    state.initialized = true
    state.loaded = true
    state.phase = 'ready'
    state.tailBytes = tail.bytes
    state.updatesSinceSnapshot = tail.updates
  }

  assertJoinAllowed(room: string): void {
    const state = this.rooms.get(room)
    if (
      state?.phase === 'awaiting_apply'
      || (state?.phase === 'reconstruction_required' && state.loaded)
    ) {
      throw new CollaborationError('restarting', {
        closeCode: CloseCodes.restarting,
        httpStatus: 503,
      })
    }
  }

  isBroadcastBlocked(room: string): boolean {
    return this.rooms.get(room)?.phase === 'awaiting_apply'
  }

  requiresReconstruction(room: string): boolean {
    return this.rooms.get(room)?.phase === 'reconstruction_required'
  }

  async prepareClientUpdate(input: PrepareClientUpdateInput): Promise<'noop' | 'pending'> {
    if (this.pending.has(input.connectionId) || this.preparing.has(input.connectionId)) {
      throw new CollaborationError('internal_consistency', {
        closeCode: CloseCodes.internalConsistency,
      })
    }
    if (input.update.byteLength > this.settings.frameLimitBytes) {
      throw new CollaborationError('message_too_large', {
        closeCode: CloseCodes.messageTooLarge,
        httpStatus: 413,
      })
    }

    const roomState = this.room(input.room)
    const preparing: PreparingUpdate = {
      aborted: false,
      persistenceAttempted: false,
      room: input.room,
    }
    this.preparing.set(input.connectionId, preparing)
    let release: Release | null = null
    let clone: Y.Doc | null = null
    try {
      release = await roomState.gate.reserve()
      assertNotAborted(preparing)
      const fence = this.leaseManager.assertActive()
      if (!roomState.initialized || roomState.phase !== 'ready') {
        throw new CollaborationError('internal_consistency', {
          closeCode: CloseCodes.internalConsistency,
        })
      }
      if (input.context.expiresAt <= Date.now() / 1_000) {
        throw new CollaborationError('invalid_lease', {
          closeCode: CloseCodes.leaseInvalid,
          httpStatus: 401,
        })
      }
      const schemaHash = await getEditorSchemaFingerprint()
      if (
        input.context.protocolVersion !== this.settings.protocolVersion
        || input.context.schemaVersion !== this.settings.schemaVersion
        || input.context.schemaHash !== schemaHash
      ) {
        throw new CollaborationError('update_required', {
          closeCode: CloseCodes.incompatible,
          httpStatus: 409,
        })
      }

      const validationStartedAt = performance.now()
      const safeUpdate = validateSafeClientUpdate(input.update)
      const before = editorYDocToJson(input.document)
      const beforeStateHash = validateDocument(
        input.document,
        this.settings.documentLimitBytes,
      ).stateHash
      clone = cloneDocument(input.document)
      const appliedUpdates: Uint8Array[] = []
      const captureAppliedUpdate = (update: Uint8Array): void => {
        appliedUpdates.push(update)
      }
      clone.on('update', captureAppliedUpdate)
      try {
        Y.applyUpdate(clone, safeUpdate)
      } catch {
        throw new CollaborationError('invalid_schema', {
          closeCode: CloseCodes.incompatible,
          httpStatus: 409,
        })
      } finally {
        clone.off('update', captureAppliedUpdate)
      }
      const validated = validateDocument(clone, this.settings.documentLimitBytes)
      if (appliedUpdates.length > 1) throw invalidClientUpdate()
      const novelUpdate = appliedUpdates[0] ?? null
      if (novelUpdate && !sameBytes(safeUpdate, novelUpdate)) {
        throw invalidClientUpdate()
      }
      const hash = hashBytes(safeUpdate)
      if (!novelUpdate) {
        if (
          input.context.access === 'view'
          || (input.allowNoop && isCanonicalEmptyUpdate(safeUpdate))
        ) return 'noop'
        assertNotAborted(preparing)
        const durable = await this.api.lookupUpdates({
          documentId: input.context.documentId,
          fence,
          generation: input.context.generation,
          hashes: [hash],
        })
        assertNotAborted(preparing)
        if (durable.length === 0) throw invalidClientUpdate()
        const replay = durable[0]
        if (!replay || durable.length !== 1 || replay.hash !== hash) {
          throw new CollaborationError('internal_consistency', {
            closeCode: CloseCodes.internalConsistency,
          })
        }
        if (replay.sequence > roomState.authoritativeSequence) {
          this.requireReconstruction(input.room)
          throw new CollaborationError('internal_consistency', {
            closeCode: CloseCodes.internalConsistency,
          })
        }
        roomState.phase = 'awaiting_apply'
        this.pending.set(input.connectionId, {
          expectedStateHash: beforeStateHash,
          hash,
          persistedSequence: roomState.authoritativeSequence,
          release,
          room: input.room,
          sequence: replay.sequence,
          startedAt: performance.now(),
          updateBytes: 0,
          wasDuplicate: true,
        })
        release = null
        return 'pending'
      }
      let suggestionOperationsValidated = false
      if (input.context.access === 'suggest') {
        validateSafeSuggestionUpdate(safeUpdate)
        suggestionOperationsValidated = true
      }
      const policy = validateSuggestionUpdate(
        before,
        editorYDocToJson(clone),
        input.context.access,
        input.context.user.id,
        { afterDocument: clone, beforeDocument: input.document },
      )
      if (policy.changeKind === 'suggestion' && !suggestionOperationsValidated) {
        validateSafeSuggestionUpdate(safeUpdate)
      }
      this.metrics.observeMilliseconds(
        'inqtrix_collaboration_update_validation_seconds',
        performance.now() - validationStartedAt,
      )

      const persistenceStartedAt = performance.now()
      assertNotAborted(preparing)
      preparing.persistenceAttempted = true
      const persisted = await this.api.persistUpdate({
        actorKind: 'human',
        actorUserId: input.context.user.id,
        changeKind: policy.changeKind,
        commandPayloadHash: null,
        decision: null,
        documentId: input.context.documentId,
        fence,
        generation: input.context.generation,
        hash,
        leaseId: input.context.leaseId,
        patches: policy.patches,
        suggestions: policy.suggestions,
        suggestionIds: policy.suggestionIds,
        update: safeUpdate,
      })
      assertNotAborted(preparing)
      this.metrics.observeMilliseconds(
        'inqtrix_collaboration_update_persistence_seconds',
        performance.now() - persistenceStartedAt,
      )
      verifyPersistedResult(roomState, persisted)
      roomState.phase = 'awaiting_apply'
      this.pending.set(input.connectionId, {
        expectedStateHash: validated.stateHash,
        hash,
        persistedSequence: persisted.persistedSequence,
        release,
        room: input.room,
        sequence: persisted.sequence,
        startedAt: performance.now(),
        updateBytes: safeUpdate.byteLength,
        wasDuplicate: persisted.duplicate,
      })
      release = null
      return 'pending'
    } catch (error) {
      if (preparing.persistenceAttempted) this.requireReconstruction(input.room)
      const mapped = collaborationError(error)
      this.metrics.increment('inqtrix_collaboration_rejections_total', { reason: mapped.reason })
      throw mapped
    } finally {
      clone?.destroy()
      release?.()
      this.preparing.delete(input.connectionId)
    }
  }

  finishClientUpdate(connectionId: string, document: Y.Doc): CollaborationDurableAck | null {
    const pending = this.pending.get(connectionId)
    if (!pending) return null
    this.pending.delete(connectionId)
    try {
      const roomState = this.rooms.get(pending.room)
      if (!roomState || roomState.phase !== 'awaiting_apply') {
        throw new CollaborationError('internal_consistency', {
          closeCode: CloseCodes.internalConsistency,
        })
      }
      const actual = validateDocument(document, this.settings.documentLimitBytes)
      if (actual.stateHash !== pending.expectedStateHash) {
        this.logger.error('authoritative_apply_mismatch', { room: pending.room })
        throw new CollaborationError('internal_consistency', {
          closeCode: CloseCodes.internalConsistency,
        })
      }
      roomState.authoritativeSequence = pending.persistedSequence
      if (!pending.wasDuplicate) {
        roomState.tailBytes += pending.updateBytes
        roomState.updatesSinceSnapshot += 1
      }
      roomState.phase = 'ready'
      this.metrics.observeMilliseconds(
        'inqtrix_collaboration_durable_ack_seconds',
        performance.now() - pending.startedAt,
      )
      return {
        hash: pending.hash,
        sequence: pending.sequence,
        type: 'durable_ack',
      }
    } catch (error) {
      this.requireReconstruction(pending.room)
      throw error
    } finally {
      pending.release()
    }
  }

  abortClientUpdate(connectionId: string): string | null {
    const pending = this.pending.get(connectionId)
    if (pending) {
      this.pending.delete(connectionId)
      pending.release()
      this.requireReconstruction(pending.room)
      return pending.room
    }
    const preparing = this.preparing.get(connectionId)
    if (!preparing) return null
    preparing.aborted = true
    if (!preparing.persistenceAttempted) return null
    this.requireReconstruction(preparing.room)
    return preparing.room
  }

  async applyServerMutation(input: ServerMutationInput): Promise<ServerMutationResult> {
    const roomState = this.room(input.room)
    const release = await roomState.gate.reserve()
    let clone: Y.Doc | null = null
    let authoritativeApplyStarted = false
    let persistenceAttempted = false
    try {
      const fence = this.leaseManager.assertActive()
      const replay = await this.api.lookupCommand({
        commandId: input.commandId,
        commandPayloadHash: input.commandPayloadHash,
        documentId: input.documentId,
        fence,
        generation: input.generation,
      })
      if (replay) {
        if (
          replay.actorKind !== input.actorKind
          || replay.actorUserId !== input.actorUserId
          || replay.changeKind !== input.changeKind
          || replay.commandId !== input.commandId
          || replay.commandPayloadHash !== input.commandPayloadHash
          || replay.decision !== input.decision
          || replay.generation !== input.generation
          || !sameIds(replay.patchIds, input.requestedPatchIds)
        ) {
          throw new CollaborationError('sequence_conflict', {
            closeCode: CloseCodes.incompatible,
            httpStatus: 409,
          })
        }
        if (
          !roomState.initialized
          || roomState.phase !== 'ready'
          || roomState.authoritativeSequence < replay.sequence
        ) {
          roomState.phase = 'awaiting_apply'
          authoritativeApplyStarted = true
          await this.reconcileCommandReplay(roomState, input, replay.sequence, fence)
          roomState.phase = 'ready'
          authoritativeApplyStarted = false
          this.notifyAuthoritativeApplySuccess(input.room)
        }
        return {
          ack: {
            hash: replay.updateHash,
            sequence: replay.sequence,
            type: 'durable_ack',
          },
          patchIds: replay.patchIds,
          suggestionIds: replay.suggestionIds,
        }
      }
      if (
        !roomState.initialized
        || roomState.phase !== 'ready'
        || roomState.authoritativeSequence !== input.expectedSequence
      ) {
        throw new CollaborationError('sequence_conflict', {
          closeCode: CloseCodes.incompatible,
          httpStatus: 409,
        })
      }
      clone = cloneDocument(input.document)
      const beforeStateHash = validateDocument(
        clone,
        this.settings.documentLimitBytes,
      ).stateHash
      const before = editorYDocToJson(clone)
      const beforeVector = Y.encodeStateVector(clone)
      const mutation = input.mutate(clone)
      const validated = validateDocument(clone, this.settings.documentLimitBytes)
      if (validated.stateHash === beforeStateHash) {
        throw new CollaborationError('decision_conflict', {
          closeCode: CloseCodes.incompatible,
          httpStatus: 409,
        })
      }
      const update = Y.encodeStateAsUpdate(clone, beforeVector)
      const hash = hashBytes(update)
      const patches = deriveSuggestionPatchStates(
        before,
        editorYDocToJson(clone),
        mutation.suggestionIds,
      )
      persistenceAttempted = true
      const persisted = await this.api.persistUpdate({
        actorKind: input.actorKind,
        actorUserId: input.actorUserId,
        changeKind: input.changeKind,
        commandId: input.commandId,
        commandPayloadHash: input.commandPayloadHash,
        decision: input.decision,
        documentId: input.documentId,
        expectedSequence: input.expectedSequence,
        fence,
        generation: input.generation,
        hash,
        leaseId: null,
        patches,
        suggestions: mutation.suggestions,
        suggestionIds: mutation.suggestionIds,
        update,
      })
      verifyPersistedResult(roomState, persisted)
      roomState.phase = 'awaiting_apply'
      authoritativeApplyStarted = true
      Y.applyUpdate(input.document, update, {
        context: { actorUserId: input.actorUserId, commandId: input.commandId },
        source: 'local',
      })
      const authoritative = validateDocument(input.document, this.settings.documentLimitBytes)
      if (authoritative.stateHash !== validated.stateHash) {
        throw new CollaborationError('internal_consistency', {
          closeCode: CloseCodes.internalConsistency,
        })
      }
      roomState.authoritativeSequence = persisted.persistedSequence
      roomState.tailBytes += persisted.duplicate ? 0 : update.byteLength
      roomState.updatesSinceSnapshot += persisted.duplicate ? 0 : 1
      roomState.phase = 'ready'
      authoritativeApplyStarted = false
      this.notifyAuthoritativeApplySuccess(input.room)
      return {
        ack: { hash, sequence: persisted.sequence, type: 'durable_ack' },
        patchIds: mutation.patchIds,
        suggestionIds: mutation.suggestionIds,
      }
    } catch (error) {
      if (persistenceAttempted || authoritativeApplyStarted) {
        this.requireReconstruction(input.room)
        this.notifyAuthoritativeApplyFailure(input.room)
      }
      throw collaborationError(error)
    } finally {
      clone?.destroy()
      release()
    }
  }

  private async reconcileCommandReplay(
    roomState: RoomState,
    input: ServerMutationInput,
    commandSequence: number,
    fence: ReturnType<InstanceLeaseManager['assertActive']>,
  ): Promise<void> {
    const loaded = await this.api.loadDocumentState({
      documentId: input.documentId,
      fence,
      generation: input.generation,
    })
    if (
      loaded.persistedSequence < commandSequence
      || loaded.persistedSequence < roomState.authoritativeSequence
    ) {
      throw new CollaborationError('internal_consistency', {
        closeCode: CloseCodes.internalConsistency,
      })
    }
    let recoveredUpdates = loaded.updates
    const recovered = await reconstructDocument(loaded, {
      documentId: input.documentId,
      generation: input.generation,
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
        recoveredUpdates = [...updates]
      },
    })
    try {
      const expected = validateDocument(recovered, this.settings.documentLimitBytes)
      const update = Y.encodeStateAsUpdate(recovered, Y.encodeStateVector(input.document))
      Y.applyUpdate(input.document, update, {
        context: { commandId: input.commandId },
        source: 'command_replay',
      })
      const authoritative = validateDocument(input.document, this.settings.documentLimitBytes)
      if (authoritative.stateHash !== expected.stateHash) {
        throw new CollaborationError('internal_consistency', {
          closeCode: CloseCodes.internalConsistency,
        })
      }
      roomState.initialized = true
      roomState.loaded = true
      roomState.authoritativeSequence = loaded.persistedSequence
      roomState.tailBytes = recoveredUpdates.reduce(
        (total, item) => total + item.update.byteLength,
        0,
      )
      roomState.updatesSinceSnapshot = recoveredUpdates.length
      this.metrics.increment('inqtrix_collaboration_command_reconciliations_total')
      this.logger.warn('command_replay_reconciled', {
        sequence: commandSequence,
      })
    } finally {
      recovered.destroy()
    }
  }

  getPersistedSequence(room: string): number {
    const state = this.rooms.get(room)
    if (
      !state?.initialized
      || state.phase === 'loading'
      || state.phase === 'reconstruction_required'
    ) {
      throw new CollaborationError('internal_consistency', {
        closeCode: CloseCodes.internalConsistency,
      })
    }
    return state.authoritativeSequence
  }

  async captureDocument(room: string, document: Y.Doc): Promise<CapturedDocument> {
    const roomState = this.room(room)
    const release = await roomState.gate.reserve()
    try {
      this.leaseManager.assertActive()
      if (!roomState.initialized || roomState.phase !== 'ready') {
        throw new CollaborationError('internal_consistency', {
          closeCode: CloseCodes.internalConsistency,
        })
      }
      const captured = cloneDocument(document)
      validateDocument(captured, this.settings.documentLimitBytes)
      return {
        document: captured,
        sequence: roomState.authoritativeSequence,
      }
    } finally {
      release()
    }
  }

  shouldSnapshot(room: string): boolean {
    const state = this.rooms.get(room)
    return Boolean(state?.initialized && state.phase === 'ready' && (
      state.updatesSinceSnapshot >= this.settings.snapshotMaxUpdates
      || state.tailBytes >= this.settings.snapshotTailBytes
    ))
  }

  hasUnsnapshottedUpdates(room: string): boolean {
    const state = this.rooms.get(room)
    return Boolean(
      state?.initialized
      && state.phase === 'ready'
      && (state.updatesSinceSnapshot > 0 || state.tailBytes > 0),
    )
  }

  markSnapshot(room: string, coveredSequence: number): boolean {
    const state = this.rooms.get(room)
    if (
      !state?.initialized
      || state.phase !== 'ready'
      || coveredSequence !== state.authoritativeSequence
    ) return false
    state.tailBytes = 0
    state.updatesSinceSnapshot = 0
    return true
  }

  async awaitRoom(room: string): Promise<void> {
    await this.rooms.get(room)?.gate.awaitIdle()
  }

  async awaitAll(): Promise<void> {
    await Promise.all([...this.rooms.values()].map((state) => state.gate.awaitIdle()))
  }

  markUnloaded(room: string): void {
    const state = this.rooms.get(room)
    if (!state) return
    state.loaded = false
    state.initialized = false
    if (state.phase === 'reconstruction_required') {
      state.tailBytes = 0
      state.updatesSinceSnapshot = 0
    } else if (state.gate.isIdle()) {
      this.rooms.delete(room)
    }
    this.updateQueueMetrics()
  }

  private room(room: string): RoomState {
    const existing = this.rooms.get(room)
    if (existing) return existing
    const created: RoomState = {
      authoritativeSequence: 0,
      gate: new SerialGate(() => this.updateQueueMetrics()),
      initialized: false,
      loaded: false,
      phase: 'loading',
      tailBytes: 0,
      updatesSinceSnapshot: 0,
    }
    this.rooms.set(room, created)
    this.updateQueueMetrics()
    return created
  }

  private requireReconstruction(room: string): void {
    const state = this.room(room)
    state.initialized = false
    state.phase = 'reconstruction_required'
    state.tailBytes = 0
    state.updatesSinceSnapshot = 0
  }

  private notifyAuthoritativeApplyFailure(room: string): void {
    try {
      this.callbacks.onAuthoritativeApplyFailure?.(room)
    } catch {
      this.logger.error('authoritative_apply_callback_failed', { outcome: 'failure' })
    }
  }

  private notifyAuthoritativeApplySuccess(room: string): void {
    try {
      this.callbacks.onAuthoritativeApplySuccess?.(room)
    } catch {
      this.logger.error('authoritative_apply_callback_failed', { outcome: 'success' })
    }
  }

  private updateQueueMetrics(): void {
    this.metrics.set('inqtrix_collaboration_rooms', this.rooms.size)
    this.metrics.set(
      'inqtrix_collaboration_document_queue_depth',
      [...this.rooms.values()].reduce((total, state) => total + state.gate.depth, 0),
    )
  }
}

function assertNotAborted(preparing: PreparingUpdate): void {
  if (!preparing.aborted) return
  throw new CollaborationError('invalid_lease', {
    closeCode: CloseCodes.leaseInvalid,
    httpStatus: 401,
  })
}

function suggestionPolicyViolation(): CollaborationError {
  return new CollaborationError('suggestion_policy_violation', {
    closeCode: CloseCodes.accessRevoked,
    httpStatus: 403,
  })
}

function invalidClientUpdate(): CollaborationError {
  return new CollaborationError('invalid_schema', {
    closeCode: CloseCodes.incompatible,
    httpStatus: 409,
  })
}

function validateSafeClientUpdate(update: Uint8Array): Uint8Array {
  try {
    return validateCanonicalYjsV1Update(update)
  } catch {
    throw invalidClientUpdate()
  }
}

function validateSafeSuggestionUpdate(update: Uint8Array): Uint8Array {
  try {
    return validateSuggestionYjsUpdate(update)
  } catch {
    throw suggestionPolicyViolation()
  }
}

function isCanonicalEmptyUpdate(update: Uint8Array): boolean {
  return update.byteLength === 2 && update[0] === 0 && update[1] === 0
}

function sameBytes(left: Uint8Array, right: Uint8Array): boolean {
  return (
    left.byteLength === right.byteLength
    && left.every((value, index) => value === right[index])
  )
}

function sameIds(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) return false
  const sortedLeft = [...left].sort()
  const sortedRight = [...right].sort()
  return sortedLeft.every((value, index) => value === sortedRight[index])
}

class SerialGate {
  private currentTail: Promise<void> = Promise.resolve()
  depth = 0

  constructor(private readonly onChange: () => void) {}

  async reserve(): Promise<Release> {
    const predecessor = this.currentTail
    let unlock = (): void => undefined
    const held = new Promise<void>((resolve) => {
      unlock = resolve
    })
    this.currentTail = predecessor.then(() => held)
    this.depth += 1
    this.onChange()
    await predecessor
    let released = false
    return () => {
      if (released) return
      released = true
      this.depth -= 1
      unlock()
      this.onChange()
    }
  }

  async awaitIdle(): Promise<void> {
    await this.currentTail
  }

  isIdle(): boolean {
    return this.depth === 0
  }
}

function verifyPersistedResult(
  state: RoomState,
  result: {
    duplicate: boolean
    persistedSequence: number
    sequence: number
  },
): void {
  const valid = result.duplicate
    ? (
        result.sequence <= result.persistedSequence
        && result.persistedSequence === state.authoritativeSequence
      )
    : (
        result.sequence === state.authoritativeSequence + 1
        && result.persistedSequence === result.sequence
      )
  if (!valid) {
    throw new CollaborationError('internal_consistency', {
      closeCode: CloseCodes.internalConsistency,
    })
  }
}
