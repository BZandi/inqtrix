/**
 * Wire<->record conversion for agent sessions (the knowledgeSessionSync
 * pattern). Agent sessions are light: turns are durable server runs, while
 * `items_json` carries only the small user-owned source policy.
 */

import type { AgentSessionGroupRecord, AgentSessionRecord } from './model'
import type { ServerAgentSession, ServerAgentSessionGroup } from './types'
import {
  normalizeAgentSourcePolicy,
  type AgentSourcePolicy,
} from './executionPolicy'

export function agentSessionFingerprint(
  session: AgentSessionRecord,
): string {
  const sourcePolicy = normalizeAgentSourcePolicy(session.sourcePolicy)
  return [
    session.title,
    session.groupId ?? '',
    session.updatedAt,
    sourcePolicy.web,
    sourcePolicy.knowledge,
  ].join('\u0000')
}

/** Ordered sessions admitted to the persistence API. Shared-run view sessions
 * stay entirely client-derived and are deliberately absent. */
export function persistableAgentSessionsInOrder(
  sessions: Readonly<Record<string, AgentSessionRecord>>,
  sessionOrder: readonly string[],
): AgentSessionRecord[] {
  return sessionOrder.flatMap((sessionId) => {
    const session = sessions[sessionId]
    return session && session.persistable !== false ? [session] : []
  })
}

export function serverAgentSessionPayload(session: AgentSessionRecord): {
  title: string
  items_json: string
  group_id: string | null
  created_at: number
  updated_at: number
} {
  return {
    title: session.title,
    items_json: JSON.stringify({
      source_policy: normalizeAgentSourcePolicy(session.sourcePolicy),
    }),
    group_id: session.groupId,
    created_at: secondsFromIso(session.createdAt),
    updated_at: secondsFromIso(session.updatedAt),
  }
}

/** Parse only the metadata fields this client owns. Unknown and malformed
 * fields are ignored so older and newer server rows remain loadable. */
export function agentSessionMetadataFromJson(value: string | undefined): {
  sourcePolicy: AgentSourcePolicy
} {
  if (!value) return { sourcePolicy: normalizeAgentSourcePolicy(null) }
  try {
    const parsed = JSON.parse(value) as unknown
    const sourcePolicy =
      parsed && typeof parsed === 'object'
        ? (parsed as Record<string, unknown>).source_policy
        : null
    return { sourcePolicy: normalizeAgentSourcePolicy(sourcePolicy) }
  } catch {
    return { sourcePolicy: normalizeAgentSourcePolicy(null) }
  }
}

export function serverAgentSessionGroupPayload(
  group: AgentSessionGroupRecord,
): { title: string; created_at: number; updated_at: number } {
  return {
    title: group.title,
    created_at: secondsFromIso(group.createdAt),
    updated_at: secondsFromIso(group.updatedAt),
  }
}

export function sessionsFromServer(
  sessions: ServerAgentSession[],
  groups: ServerAgentSessionGroup[],
): { sessions: ServerAgentSession[]; groups: ServerAgentSessionGroup[] } {
  return { groups, sessions }
}

function secondsFromIso(value: string): number {
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed / 1000 : Date.now() / 1000
}
