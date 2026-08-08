import type { ServerDeletionOperation } from '@/api/inqtrixClient'

export type SessionDeletionState = {
  error: string | null
  operationId: string
  stage: string
  status: 'deleting' | 'delete_failed'
}

export class SessionDeletionContractError extends Error {}

export function assertSessionDeletionOperation(
  operation: ServerDeletionOperation,
  expectedKind: 'agent_session' | 'knowledge_session',
  expectedSessionId: string,
): void {
  if (
    operation.target_kind !== expectedKind
    || operation.target_id !== expectedSessionId
  ) {
    throw new SessionDeletionContractError(
      'Deletion receipt does not match the requested session',
    )
  }
}

export function sessionDeletionFromWire(value: {
  deletion_error?: string | null
  deletion_operation_id?: string | null
  deletion_stage?: string | null
  lifecycle_status?: 'active' | 'deleting' | 'delete_failed'
}): SessionDeletionState | undefined {
  const operationId = value.deletion_operation_id
  if (!operationId || value.lifecycle_status === 'active' || !value.lifecycle_status) {
    return undefined
  }
  return {
    error: value.deletion_error ?? null,
    operationId,
    stage: value.deletion_stage ?? value.lifecycle_status,
    status: value.lifecycle_status,
  }
}
