export type CollaborationAccess = 'edit' | 'suggest' | 'view'
export type CollaborationActorKind = 'assistant' | 'agent' | 'human' | 'system'
export type CollaborationChangeKind = 'decision' | 'direct' | 'suggestion' | 'system'

export type CollaborationDurableAck = {
  hash: string
  sequence: number
  type: 'durable_ack'
}

export type CollaborationDurableRejection = {
  code: 'forbidden' | 'generation_mismatch' | 'invalid_schema' | 'rate_limited' | 'too_large'
  hash: string
  type: 'durable_rejection'
}

export type CollaborationServerMessage =
  | CollaborationDurableAck
  | CollaborationDurableRejection

export type CollaborationUpdateEnvelope = {
  actorKind: CollaborationActorKind
  changeKind: CollaborationChangeKind
  commandId?: string
  generation: number
  hash: string
  suggestionIds: string[]
  update: Uint8Array
}

export type CollaborationDecisionCommand = {
  commandId: string
  decision: 'accept' | 'reject'
  expectedSequence: number
  patchIds: string[]
}

export type CollaborationStateVector = {
  coveredSequence: number
  generation: number
  stateVector: Uint8Array
}

export type CollaborationSnapshot = CollaborationStateVector & {
  createdAt: number
  projectionHash: string
  schemaHash: string
  schemaVersion: number
  stateHash: string
  stateUpdate: Uint8Array
}

export type CollaborationProjection = {
  contentMarkdown: string
  coveredSequence: number
  generation: number
  projectionHash: string
  updatedAt: number
}

export function isCollaborationDurableAck(value: unknown): value is CollaborationDurableAck {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Partial<CollaborationDurableAck>
  return (
    candidate.type === 'durable_ack'
    && typeof candidate.hash === 'string'
    && Number.isSafeInteger(candidate.sequence)
    && (candidate.sequence ?? 0) > 0
  )
}
