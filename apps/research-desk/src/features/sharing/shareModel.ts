/** Pure sharing logic — kept free of React so vitest covers it directly. */

import type { ResearchRunAccess } from '@/features/researchRuns/types'
import type { SharedWithMeEntry, UserSearchResult } from './types'

/**
 * Split a job list into own and shared-in entries while preserving
 * order. Shared-in jobs (the server's additive `access` annotation)
 * render under the "Mit mir geteilt" divider, never mixed into the
 * caller's own runs.
 */
export function partitionJobsByAccess<T extends { access?: ResearchRunAccess }>(
  jobs: readonly T[],
): { own: T[]; shared: T[] } {
  const own: T[] = []
  const shared: T[] = []
  for (const job of jobs) {
    if (job.access) shared.push(job)
    else own.push(job)
  }
  return { own, shared }
}

/**
 * Whether a shared-in run may be cancelled by the recipient. Mirrors
 * the server rule (cancel needs at least an edit grant) so the UI
 * never offers a button that would land as 404. Owned runs
 * (no annotation) keep the existing status-based gate.
 */
export function canCancelWithAccess(access: ResearchRunAccess | undefined): boolean {
  return access === undefined || access.permission === 'edit'
}

/** Toggle a user in the dialog's invitee selection (dedup by subject). */
export function toggleSelectedUser(
  selected: readonly UserSearchResult[],
  user: UserSearchResult,
): UserSearchResult[] {
  if (selected.some((entry) => entry.subject === user.subject)) {
    return selected.filter((entry) => entry.subject !== user.subject)
  }
  return [...selected, user]
}

/** Typeahead rows minus everyone who already holds a share or is picked. */
export function selectableSearchResults(
  results: readonly UserSearchResult[],
  existingSubjects: ReadonlySet<string>,
): UserSearchResult[] {
  return results.filter((user) => !existingSubjects.has(user.subject))
}

/** Visible label for one person (display name first, email fallback). */
export function personLabel(
  displayName: string | null | undefined,
  email: string | null | undefined,
  fallback: string,
): string {
  return displayName?.trim() || email?.trim() || fallback
}

/** Map shared-with-me rows by resource id for badge lookups. */
export function sharedWithMeByResourceId(
  entries: readonly SharedWithMeEntry[],
): Map<string, SharedWithMeEntry> {
  return new Map(entries.map((entry) => [entry.resource_id, entry]))
}
