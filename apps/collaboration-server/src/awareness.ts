import type { ConnectionContext } from './contracts'
import { CloseCodes, CollaborationError } from './errors'

const MAX_AWARENESS_STATE_BYTES = 16 * 1024

export function enforceAwarenessIdentity(
  states: Map<number, Record<string, unknown>>,
  context: ConnectionContext,
): void {
  if (states.size > 1) throw invalidAwareness()
  for (const [clientId, state] of states) {
    const sanitized: Record<string, unknown> = {
      user: {
        color: context.user.color,
        id: context.user.id,
        name: context.user.name,
      },
    }
    if (state.cursor !== undefined && state.cursor !== null) {
      sanitized.cursor = jsonClone(state.cursor)
    }
    if (Buffer.byteLength(JSON.stringify(sanitized), 'utf8') > MAX_AWARENESS_STATE_BYTES) {
      throw invalidAwareness()
    }
    states.set(clientId, sanitized)
  }
}

function jsonClone(value: unknown): unknown {
  try {
    return JSON.parse(JSON.stringify(value)) as unknown
  } catch {
    throw invalidAwareness()
  }
}

function invalidAwareness(): CollaborationError {
  return new CollaborationError('invalid_request', {
    closeCode: CloseCodes.accessRevoked,
    httpStatus: 400,
  })
}
