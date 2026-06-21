/** Server-template <-> chat-rule mapping (pure; the hook stays thin).
 *
 * Sync model (Google-Drive rule): the SERVER record is the one truth
 * for synced rules — hydrate always takes the server state. Updates use
 * OPTIMISTIC concurrency: a save carries the `updated_at` it loaded as
 * a precondition, and a server-side conflict (someone else saved in
 * between) surfaces instead of silently overwriting. Local-only rules
 * (no serverTemplateId) stay untouched until their first save uploads
 * them.
 */

import type { ChatRuleCategory, ChatRuleRecord } from '@/features/project/types'

/** A save was rejected because the template changed server-side since
 * it was loaded — the optimistic-concurrency conflict (HTTP 409).
 *
 * `refreshed` says whether the conflict recovery managed to reload the
 * current version into local state: `true` (the editor now shows the
 * latest, re-save will land) drives a different message than `false`
 * (the refresh fetch itself failed, so nothing was reloaded and a
 * blind re-save would just 409 again). */
export class TemplateConflictError extends Error {
  readonly refreshed: boolean
  constructor(refreshed: boolean) {
    super('prompt template was modified by someone else')
    this.name = 'TemplateConflictError'
    this.refreshed = refreshed
  }
}

export type PromptTemplateAccess = {
  permission: 'edit' | 'view'
  via: 'share'
}

/** Wire shape of one `/v1/prompt-templates` record. */
export type PromptTemplateInfo = {
  access?: PromptTemplateAccess
  category: ChatRuleCategory | null
  content_markdown: string
  created_at: number
  id: string
  include_in_autocomplete: boolean
  label: string
  title: string
  updated_at: number
  visibility: { chat?: boolean; editor?: boolean }
}

export type PromptTemplatePayload = {
  category: ChatRuleCategory | null
  content_markdown: string
  /** Optimistic-concurrency precondition (unix seconds) — the server
   * `updated_at` the editor loaded; omitted on create. */
  expected_updated_at?: number
  include_in_autocomplete: boolean
  label: string
  title: string
  visibility: { chat: boolean; editor: boolean }
}

function isoFromUnix(seconds: number): string {
  return new Date(seconds * 1000).toISOString()
}

/** Map one server template into the local rule shape.
 *
 * The rule id IS the server id (`pt_...`) so hydrate upserts stay
 * idempotent. `linkedContextRefs` remain browser-local (they point at
 * local file assets) — the reducer's upsert REPLACES records
 * wholesale, so the refs of the matching *existing* rule are carried
 * over here explicitly.
 */
export function ruleFromTemplate(
  info: PromptTemplateInfo,
  existing?: ChatRuleRecord,
): ChatRuleRecord {
  return {
    access: info.access,
    category: info.category ?? undefined,
    contentMarkdown: info.content_markdown,
    createdAt: isoFromUnix(info.created_at),
    id: info.id,
    includeInAutocomplete: info.include_in_autocomplete,
    label: info.label,
    linkedContextRefs: existing?.linkedContextRefs,
    serverTemplateId: info.id,
    // The EXACT server timestamp (unix seconds), carried raw as the
    // optimistic-concurrency precondition — never via the ISO
    // `updatedAt`, whose millisecond truncation would mismatch the
    // microsecond float the server stores and trip a false 409.
    serverUpdatedAt: info.updated_at,
    title: info.title,
    updatedAt: isoFromUnix(info.updated_at),
    visibility: {
      chat: info.visibility.chat !== false,
      editor: info.visibility.editor !== false,
    },
  }
}

/** Whether a hydrated rule matches the local record (skip the no-op
 * upsert — it would mark a freshly loaded project dirty). */
export function isSameSyncedRule(
  existing: ChatRuleRecord,
  incoming: ChatRuleRecord,
): boolean {
  return (
    existing.id === incoming.id
    && existing.updatedAt === incoming.updatedAt
    && existing.contentMarkdown === incoming.contentMarkdown
    && existing.title === incoming.title
    && existing.label === incoming.label
    && (existing.category ?? null) === (incoming.category ?? null)
    && (existing.includeInAutocomplete !== false)
      === (incoming.includeInAutocomplete !== false)
    && JSON.stringify(existing.access ?? null)
      === JSON.stringify(incoming.access ?? null)
    && JSON.stringify(existing.visibility ?? null)
      === JSON.stringify(incoming.visibility ?? null)
  )
}

/** The writable server fields of one local rule. */
export function templatePayloadFromRule(rule: ChatRuleRecord): PromptTemplatePayload {
  return {
    category: rule.category ?? null,
    content_markdown: rule.contentMarkdown,
    include_in_autocomplete: rule.includeInAutocomplete !== false,
    label: rule.label,
    title: rule.title,
    visibility: {
      chat: rule.visibility?.chat !== false,
      editor: rule.visibility?.editor !== false,
    },
  }
}

/** Local synced rules whose server record vanished (deleted/revoked). */
export function staleSyncedRuleIds(
  localRules: readonly ChatRuleRecord[],
  serverIds: ReadonlySet<string>,
): string[] {
  return localRules
    .filter(
      (rule) =>
        rule.serverTemplateId !== undefined
        && !serverIds.has(rule.serverTemplateId),
    )
    .map((rule) => rule.id)
}

/** Whether the caller may edit/save this rule (view shares cannot). */
export function canEditRule(rule: ChatRuleRecord | null): boolean {
  return rule?.access?.permission !== 'view'
}

/** Whether the caller may delete this rule (shared-in never deletes). */
export function canDeleteRule(rule: ChatRuleRecord | null): boolean {
  return rule !== null && rule.access === undefined
}
