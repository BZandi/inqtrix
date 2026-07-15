import type { AdminSystemRuntime, AdminUser } from '@/api/inqtrixClient'

/**
 * Pure view-model logic for the admin surface (node-testable, no DOM).
 *
 * The user-table guards MIRROR the server's two invariants so the UI
 * disables an action before the request and explains why, rather than
 * letting the user fire a doomed call and read a 409. They are advisory
 * UX — the server stays the source of truth (atomic last-admin guard).
 */

export type GuardReason = 'self' | 'last_admin'
export type GuardResult = { allowed: true } | { allowed: false; reason: GuardReason }

const ALLOWED: GuardResult = { allowed: true }

/** Whether *user* is the signed-in caller (the "you" row). */
export function isSelf(user: AdminUser, sessionUserId: string | null): boolean {
  return sessionUserId != null && user.id === sessionUserId
}

/** Active instance admins (admin role and not disabled). */
export function activeAdminCount(users: readonly AdminUser[]): number {
  return users.filter((u) => u.instance_role === 'admin' && !u.disabled).length
}

/** Whether removing *user*'s active-admin status leaves zero admins. */
export function isLastActiveAdmin(
  users: readonly AdminUser[],
  user: AdminUser,
): boolean {
  return (
    user.instance_role === 'admin' &&
    !user.disabled &&
    activeAdminCount(users) <= 1
  )
}

/** Can the caller set *user* to *nextRole*? (self-demote + last-admin locked) */
export function canSetRole(
  users: readonly AdminUser[],
  user: AdminUser,
  sessionUserId: string | null,
  nextRole: 'admin' | 'user',
): GuardResult {
  if (nextRole === user.instance_role) return ALLOWED
  if (nextRole === 'user') {
    if (isSelf(user, sessionUserId)) return { allowed: false, reason: 'self' }
    if (isLastActiveAdmin(users, user)) {
      return { allowed: false, reason: 'last_admin' }
    }
  }
  return ALLOWED
}

/** Can the caller disable *user*? (self + last-admin locked; enable is free) */
export function canDisable(
  users: readonly AdminUser[],
  user: AdminUser,
  sessionUserId: string | null,
): GuardResult {
  if (isSelf(user, sessionUserId)) return { allowed: false, reason: 'self' }
  if (isLastActiveAdmin(users, user)) {
    return { allowed: false, reason: 'last_admin' }
  }
  return ALLOWED
}

/** Admins first, then by email — a stable, scannable listing order. */
export function sortUsers(users: readonly AdminUser[]): AdminUser[] {
  return [...users].sort((a, b) => {
    if (a.instance_role !== b.instance_role) {
      return a.instance_role === 'admin' ? -1 : 1
    }
    return (a.email ?? '').localeCompare(b.email ?? '')
  })
}

// --- System / feature-status derivation -----------------------------------

export type FeatureRow = { key: string; on: boolean }

/**
 * Feature rows derived from the OPEN `capabilities.features` map — never a
 * hardcoded list, so a new server flag surfaces automatically (the panel
 * shows "what is on/off" without a frontend release).
 */
export function deriveFeatureRows(
  features: Record<string, boolean> | undefined,
): FeatureRow[] {
  if (!features) return []
  return Object.entries(features)
    .map(([key, on]) => ({ key, on: Boolean(on) }))
    .sort((a, b) => a.key.localeCompare(b.key))
}

/**
 * System-page feature rows: start from the public manifest, then fold in the
 * admin runtime's backend availability so configured-but-unreachable infra
 * does not render as a working capability.
 */
export function deriveSystemFeatureRows(
  features: Record<string, boolean> | undefined,
  runtime: AdminSystemRuntime | null,
): FeatureRow[] {
  return deriveFeatureRows(features).map((feature) => ({
    ...feature,
    on: runtimeFeatureOn(feature, runtime),
  }))
}

function runtimeFeatureOn(
  feature: FeatureRow,
  runtime: AdminSystemRuntime | null,
): boolean {
  if (!runtime || !feature.on) return feature.on
  const filesAvailable = runtime.files.enabled && runtime.files.object_store_available
  const knowledgeAvailable = runtime.knowledge.enabled
    && runtime.knowledge.vector_store_available

  if (feature.key === 'files') return filesAvailable
  if (feature.key === 'knowledge') return knowledgeAvailable
  if (feature.key === 'hybrid_retrieval') {
    return knowledgeAvailable && runtime.knowledge.hybrid_retrieval
  }
  if (feature.key === 'document_parser') {
    return knowledgeAvailable && runtime.knowledge.document_parser !== 'none'
  }
  if (feature.key === 'embedding_provider') {
    return knowledgeAvailable && runtime.knowledge.embedding_provider != null
  }
  if (feature.key === 'reranker') {
    return knowledgeAvailable && runtime.knowledge.reranker !== 'none'
  }
  return feature.on
}

// --- PAT one-time-reveal state machine -------------------------------------

export type PatRevealState =
  | { phase: 'idle' }
  | { phase: 'revealed'; name: string; token: string; tokenId: string }

export type PatRevealAction =
  | { name: string; token: string; tokenId: string; type: 'reveal' }
  | { type: 'dismiss' }

/**
 * The plaintext token is shown EXACTLY ONCE (it is never retrievable
 * again). `reveal` carries it into the one-time banner; `dismiss` drops it
 * from memory. There is intentionally no way back to `revealed` without a
 * fresh `reveal` from a new creation.
 */
export function patRevealReducer(
  state: PatRevealState,
  action: PatRevealAction,
): PatRevealState {
  switch (action.type) {
    case 'reveal':
      return {
        phase: 'revealed',
        name: action.name,
        token: action.token,
        tokenId: action.tokenId,
      }
    case 'dismiss':
      return { phase: 'idle' }
    default:
      return state
  }
}
