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
  | 'upstream_conflict'

export class CollaborationError extends Error {
  readonly code: number
  readonly httpStatus: number
  readonly reason: CollaborationErrorReason
  /** Der Grund, den die interne API genannt hat, unveraendert.
   *
   * Der Sidecar bildet auf seine eigene, kleine Vokabelliste ab, weil die
   * den WebSocket-Schliesscode steuert. Diese Abbildung darf den
   * urspruenglichen Grund aber nicht VERNICHTEN: sonst steht am Ende
   * weder im Log noch beim Nutzer, warum wirklich abgelehnt wurde. */
  readonly upstreamReason: string | undefined

  constructor(
    reason: CollaborationErrorReason,
    options: {
      closeCode: number
      httpStatus?: number
      upstreamReason?: string
    },
  ) {
    super(reason)
    this.name = 'CollaborationError'
    this.code = options.closeCode
    this.httpStatus = options.httpStatus ?? 500
    this.reason = reason
    this.upstreamReason = options.upstreamReason
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
      // Die interne API kennt rund vierzig Konfliktgruende; hier werden
      // nur die abgebildet, die eine EIGENE Behandlung haben. Alles
      // uebrige bleibt ein Konflikt ohne erfundene Ursache — der echte
      // Grund reist als upstreamReason mit, statt zu "dein Client ist
      // veraltet" zu werden.
      const reason: CollaborationErrorReason =
        error.reason === 'sequence_conflict' || error.reason === 'command_conflict'
          ? 'sequence_conflict'
          : error.reason === 'generation_mismatch'
            ? 'generation_mismatch'
            : error.reason === 'update_required'
              ? 'update_required'
              : 'upstream_conflict'
      return new CollaborationError(reason, {
        closeCode: CloseCodes.incompatible,
        httpStatus: 409,
        upstreamReason: error.reason,
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
  // Der Auffangzweig: jede Antwort, die oben keinen eigenen Zweig hat --
  // heute vor allem 400 -- landet hier. Sie heisst nach aussen weiterhin
  // internal_consistency, denn aus Sicht des Nutzers IST sie ein interner
  // Fehler. Aber der Grund der internen API darf dabei nicht verloren
  // gehen: ohne ihn meldet der Sidecar "internal_consistency" fuer einen
  // Ablehnungsgrund, den er selbst kennt, und die Fehlersuche beginnt bei
  // null. Genau das hat eine 400-Ablehnung als Raum-Inkonsistenz getarnt.
  return new CollaborationError('internal_consistency', {
    closeCode: CloseCodes.internalConsistency,
    ...(error instanceof ApiRequestError ? { upstreamReason: error.reason } : {}),
  })
}
