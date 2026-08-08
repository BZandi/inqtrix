/**
 * Wire<->record conversion for agent sessions (the knowledgeSessionSync
 * pattern). Agent sessions are light: turns are durable server runs, while
 * `items_json` carries only the small user-owned source policy.
 */

import type { AgentSessionGroupRecord, AgentSessionRecord } from './model'
import type { ServerAgentSession, ServerAgentSessionGroup } from './types'
import {
  agentModelSelectionKey,
  normalizeAgentModelSelection,
  normalizeAgentSourcePolicy,
  type AgentSessionModelSelection,
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
    // Without this a model change never reaches the server: the autosave
    // diffs on this key alone, so an omitted field fails silently.
    agentModelSelectionKey(session.modelSelection),
  ].join('\u0000')
}

/** Fingerprint the authoritative list response without depending on a reducer
 * update having committed already. This preserves local-newer-wins: the sync
 * baseline describes the server row, while a newer local record remains
 * detectably different and is flushed afterwards. */
export function serverAgentSessionFingerprint(
  session: ServerAgentSession,
): string {
  const metadata = agentSessionMetadataFromJson(session.items_json)
  return [
    session.title,
    session.group_id ?? '',
    new Date(session.updated_at * 1000).toISOString(),
    metadata.sourcePolicy.web,
    metadata.sourcePolicy.knowledge,
    agentModelSelectionKey(metadata.modelSelection),
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
      // Whole-value write like the policy above: the field must ride every
      // save or the stored selection is dropped on the next one.
      model_selection: session.modelSelection ?? null,
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
  modelSelection: AgentSessionModelSelection | null
} {
  const empty = {
    modelSelection: null,
    sourcePolicy: normalizeAgentSourcePolicy(null),
  }
  if (!value) return empty
  try {
    const parsed = JSON.parse(value) as unknown
    const record = parsed && typeof parsed === 'object'
      ? (parsed as Record<string, unknown>)
      : null
    return {
      modelSelection: normalizeAgentModelSelection(record?.model_selection),
      sourcePolicy: normalizeAgentSourcePolicy(record?.source_policy),
    }
  } catch {
    return empty
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
