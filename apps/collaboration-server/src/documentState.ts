import { createHash } from 'node:crypto'

import {
  getEditorSchemaFingerprint,
  validateEditorYDoc,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import type {
  LoadedDocumentCandidate,
  LoadedDocumentState,
  LoadedDocumentUpdate,
} from './contracts'
import {
  collaborationError,
  CloseCodes,
  CollaborationError,
  type CollaborationErrorReason,
} from './errors'

export type ValidatedDocument = {
  document: Y.Doc
  encodedState: Uint8Array
  stateHash: string
}

export type ReconstructionOptions = {
  onCandidateRejected?: (event: {
    candidateIndex: number
    reason: CollaborationErrorReason
  }) => void
  onCandidateSelected?: (event: {
    candidateIndex: number
    updates: readonly LoadedDocumentUpdate[]
  }) => void
}

export async function reconstructDocument(
  state: LoadedDocumentState,
  expected: {
    documentId: string
    generation: number
    schemaVersion: number
  },
  documentLimitBytes: number,
  options: ReconstructionOptions = {},
): Promise<Y.Doc> {
  const schemaHash = await getEditorSchemaFingerprint()
  if (
    state.documentId !== expected.documentId
    || state.generation !== expected.generation
    || state.schemaVersion !== expected.schemaVersion
    || state.schemaHash !== schemaHash
  ) {
    throw new CollaborationError('invalid_schema', {
      closeCode: CloseCodes.incompatible,
      httpStatus: 409,
    })
  }

  const candidates = reconstructionCandidates(state)
  for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
    const candidate = candidates[candidateIndex]!
    const document = new Y.Doc()
    try {
      reconstructCandidate(document, candidate, state.persistedSequence)
      validateDocument(document, documentLimitBytes)
      options.onCandidateSelected?.({ candidateIndex, updates: candidate.updates })
      return document
    } catch (error) {
      document.destroy()
      const mapped = collaborationError(error)
      if (candidateIndex === candidates.length - 1) throw mapped
      options.onCandidateRejected?.({ candidateIndex, reason: mapped.reason })
    }
  }
  throw new CollaborationError('internal_consistency', {
    closeCode: CloseCodes.internalConsistency,
  })
}

type ReconstructionCandidate = Omit<LoadedDocumentCandidate, 'snapshot'> & {
  snapshot: LoadedDocumentCandidate['snapshot'] | null
}

function reconstructionCandidates(state: LoadedDocumentState): ReconstructionCandidate[] {
  if (state.snapshotCandidates !== undefined) {
    if (state.snapshotCandidates.length < 1 || state.snapshotCandidates.length > 2) {
      throw new CollaborationError('internal_consistency', {
        closeCode: CloseCodes.internalConsistency,
      })
    }
    return state.snapshotCandidates
  }
  return [{ snapshot: state.snapshot, updates: state.updates }]
}

function reconstructCandidate(
  document: Y.Doc,
  candidate: ReconstructionCandidate,
  persistedSequence: number,
): void {
  let coveredSequence = 0
  if (candidate.snapshot) {
    Y.applyUpdate(document, candidate.snapshot.stateUpdate)
    coveredSequence = candidate.snapshot.coveredSequence
    if (
      hashBytes(Y.encodeStateAsUpdate(document)) !== candidate.snapshot.stateHash
      || !sameBytes(Y.encodeStateVector(document), candidate.snapshot.stateVector)
    ) {
      throw new CollaborationError('internal_consistency', {
        closeCode: CloseCodes.internalConsistency,
      })
    }
  }
  for (const update of candidate.updates) {
    if (hashBytes(update.update) !== update.hash) {
      throw new CollaborationError('internal_consistency', {
        closeCode: CloseCodes.internalConsistency,
      })
    }
    if (update.sequence !== coveredSequence + 1) {
      throw new CollaborationError('internal_consistency', {
        closeCode: CloseCodes.internalConsistency,
      })
    }
    Y.applyUpdate(document, update.update)
    coveredSequence = update.sequence
  }
  if (coveredSequence !== persistedSequence) {
    throw new CollaborationError('internal_consistency', {
      closeCode: CloseCodes.internalConsistency,
    })
  }
}

function sameBytes(left: Uint8Array, right: Uint8Array): boolean {
  return left.byteLength === right.byteLength && left.every((value, index) => value === right[index])
}

export function cloneDocument(document: Y.Doc): Y.Doc {
  const clone = new Y.Doc()
  Y.applyUpdate(clone, Y.encodeStateAsUpdate(document))
  return clone
}

export function validateDocument(
  document: Y.Doc,
  documentLimitBytes: number,
): ValidatedDocument {
  try {
    validateEditorYDoc(document)
  } catch {
    throw new CollaborationError('invalid_schema', {
      closeCode: CloseCodes.incompatible,
      httpStatus: 409,
    })
  }
  const encodedState = Y.encodeStateAsUpdate(document)
  if (encodedState.byteLength > documentLimitBytes) {
    throw new CollaborationError('document_too_large', {
      closeCode: CloseCodes.messageTooLarge,
      httpStatus: 413,
    })
  }
  return {
    document,
    encodedState,
    stateHash: hashBytes(encodedState),
  }
}

export function hashBytes(value: Uint8Array): string {
  return createHash('sha256').update(value).digest('hex')
}

export function hashString(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex')
}
