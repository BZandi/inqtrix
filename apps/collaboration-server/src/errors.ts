export const CloseCodes = Object.freeze({
  accessRevoked: 4403,
  incompatible: 4409,
  internalConsistency: 1011,
  leaseInvalid: 4401,
  messageTooLarge: 1009,
  rateLimited: 4429,
  restarting: 1012,
  serviceUnavailable: 4503,
})

export type CollaborationErrorReason =
  | 'access_revoked'
  | 'decision_conflict'
  | 'document_too_large'
  | 'generation_mismatch'
  | 'instance_lease_lost'
  | 'internal_consistency'
  | 'invalid_lease'
  | 'invalid_request'
  | 'invalid_room'
  | 'invalid_schema'
  | 'message_too_large'
  | 'rate_limited'
  | 'restarting'
  | 'sequence_conflict'
  | 'service_unavailable'
  | 'suggestion_conflict'
  | 'suggestion_policy_violation'
  | 'unsupported_suggestion_structure'
  | 'update_required'

export class CollaborationError extends Error {
  readonly code: number
  readonly httpStatus: number
  readonly reason: CollaborationErrorReason

  constructor(
    reason: CollaborationErrorReason,
    options: {
      closeCode: number
      httpStatus?: number
    },
  ) {
    super(reason)
    this.name = 'CollaborationError'
    this.code = options.closeCode
    this.httpStatus = options.httpStatus ?? 500
    this.reason = reason
  }
}

export class ApiRequestError extends Error {
  readonly reason: string
  readonly status: number

  constructor(status: number, reason: string) {
    super(`Internal API request failed with status ${status}`)
    this.name = 'ApiRequestError'
    this.reason = reason
    this.status = status
  }
}

/** Whether the failure proves the persistence transaction did not write.
 *
 * A 4xx answer from the internal API is a decision, not an outage: the request
 * was understood, rejected, and rolled back. Callers use this to distinguish a
 * rejection they can recover from locally from an unknown outcome that leaves
 * the in-memory room possibly ahead of the store. Timeouts, transport errors
 * and 5xx are deliberately NOT deterministic — the write may have landed.
 */
export function isDeterministicRejection(error: unknown): boolean {
  return error instanceof ApiRequestError
    && error.status >= 400
    && error.status < 500
}

export function collaborationError(error: unknown): CollaborationError {
  if (error instanceof CollaborationError) return error
  if (error instanceof ApiRequestError) {
    if (error.status === 401) {
      return new CollaborationError('invalid_lease', {
        closeCode: CloseCodes.leaseInvalid,
        httpStatus: 401,
      })
    }
    if (error.status === 403 || error.status === 404) {
      return new CollaborationError('access_revoked', {
        closeCode: CloseCodes.accessRevoked,
        httpStatus: error.status,
      })
    }
    if (error.status === 409) {
      const reason = error.reason === 'sequence_conflict' || error.reason === 'command_conflict'
        ? 'sequence_conflict'
        : error.reason === 'generation_mismatch'
          ? 'generation_mismatch'
          : 'update_required'
      return new CollaborationError(reason, {
        closeCode: CloseCodes.incompatible,
        httpStatus: 409,
      })
    }
    if (error.status === 413) {
      return new CollaborationError('document_too_large', {
        closeCode: CloseCodes.messageTooLarge,
        httpStatus: 413,
      })
    }
    if (error.status === 429) {
      return new CollaborationError('rate_limited', {
        closeCode: CloseCodes.rateLimited,
        httpStatus: 429,
      })
    }
    if (error.status === 503) {
      return new CollaborationError('service_unavailable', {
        closeCode: CloseCodes.serviceUnavailable,
        httpStatus: 503,
      })
    }
  }
  return new CollaborationError('internal_consistency', {
    closeCode: CloseCodes.internalConsistency,
  })
}
