import { Buffer } from 'node:buffer'

import type {
  CollaborationApi,
  CollaborationSettings,
  ConnectionContext,
} from './contracts'
import { CloseCodes, CollaborationError } from './errors'
import { InstanceLeaseManager } from './instanceLease'

type StatelessSender = {
  sendStateless(payload: string): void
}

export async function reconcileDurability(
  payload: string,
  context: ConnectionContext,
  sender: StatelessSender,
  api: CollaborationApi,
  leaseManager: InstanceLeaseManager,
  settings: CollaborationSettings,
): Promise<number> {
  if (context.tenantId !== settings.tenantId) throw invalidLease()
  const hashes = parseReconcileMessage(payload, settings)
  const updates = await api.lookupUpdates({
    documentId: context.documentId,
    fence: leaseManager.assertActive(),
    generation: context.generation,
    hashes,
  })
  for (const update of updates) {
    sender.sendStateless(JSON.stringify({
      hash: update.hash,
      sequence: update.sequence,
      type: 'durable_ack',
    }))
  }
  return updates.length
}

function parseReconcileMessage(
  payload: string,
  settings: CollaborationSettings,
): string[] {
  if (Buffer.byteLength(payload, 'utf8') > settings.frameLimitBytes) {
    throw new CollaborationError('message_too_large', {
      closeCode: CloseCodes.messageTooLarge,
      httpStatus: 413,
    })
  }
  let value: unknown
  try {
    value = JSON.parse(payload)
  } catch {
    throw invalidRequest()
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalidRequest()
  const message = value as Record<string, unknown>
  if (
    Object.keys(message).sort().join(',') !== 'hashes,type'
    || message.type !== 'durability_reconcile'
    || !Array.isArray(message.hashes)
    || message.hashes.length < 1
    || message.hashes.length > settings.reconcileMaxHashes
  ) {
    throw invalidRequest()
  }
  const hashes = message.hashes
  if (
    hashes.some((hash) => typeof hash !== 'string' || !/^[a-f0-9]{64}$/.test(hash))
    || new Set(hashes).size !== hashes.length
  ) {
    throw invalidRequest()
  }
  return hashes as string[]
}

function invalidRequest(): CollaborationError {
  return new CollaborationError('invalid_request', {
    closeCode: CloseCodes.accessRevoked,
    httpStatus: 400,
  })
}

function invalidLease(): CollaborationError {
  return new CollaborationError('invalid_lease', {
    closeCode: CloseCodes.leaseInvalid,
    httpStatus: 401,
  })
}
