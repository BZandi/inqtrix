import { describe, expect, it } from 'vitest'

import type { ServerDeletionOperation } from '@/api/inqtrixClient'
import {
  assertSessionDeletionOperation,
  SessionDeletionContractError,
  sessionDeletionFromWire,
} from './sessionDeletion'

function receipt(
  overrides: Partial<ServerDeletionOperation> = {},
): ServerDeletionOperation {
  return {
    attempt: 1,
    asset_ids: [],
    completed_items: 0,
    created_at: 1,
    error: null,
    finished_at: null,
    operation_id: 'del_1',
    retryable: false,
    stage: 'queued',
    started_at: null,
    status: 'queued',
    target_id: 'as_1',
    target_kind: 'agent_session',
    total_items: 2,
    ...overrides,
  }
}

describe('session deletion projection', () => {
  it('accepts only a receipt for the exact requested session and desk', () => {
    expect(() => assertSessionDeletionOperation(
      receipt(),
      'agent_session',
      'as_1',
    )).not.toThrow()

    expect(() => assertSessionDeletionOperation(
      receipt({ target_id: 'as_other' }),
      'agent_session',
      'as_1',
    )).toThrow(SessionDeletionContractError)
    expect(() => assertSessionDeletionOperation(
      receipt({ target_kind: 'knowledge_session' }),
      'agent_session',
      'as_1',
    )).toThrow(SessionDeletionContractError)
  })

  it('restores a durable failed tombstone from session metadata', () => {
    expect(sessionDeletionFromWire({
      deletion_error: 'dependency unavailable',
      deletion_operation_id: 'del_1',
      deletion_stage: 'delete_failed',
      lifecycle_status: 'delete_failed',
    })).toEqual({
      error: 'dependency unavailable',
      operationId: 'del_1',
      stage: 'delete_failed',
      status: 'delete_failed',
    })
  })
})
