/** Wire types of the `/v1/shares*` + `/v1/users/search` surface. */

export type SharePermissionValue = 'edit' | 'view'

/** One row of `/v1/users/search` — the share-dialog typeahead. */
export type UserSearchResult = {
  display_name: string | null
  email: string | null
  subject: string
}

/** One active share as listed/created by `/v1/shares` (profile-enriched). */
export type ShareRecordInfo = {
  created_at: number
  display_name: string | null
  email: string | null
  granted_by_sub: string
  id: string
  permission: SharePermissionValue
  resource_id: string
  resource_type: string
  subject_id: string
  subject_type: string
}

/** One row of `/v1/shares/shared-with-me` (grantor name joined in). */
export type SharedWithMeEntry = {
  created_at: number
  granted_by_display_name: string | null
  granted_by_sub: string
  permission: SharePermissionValue
  resource_id: string
  resource_type: string
}

export type ShareInvitee = {
  permission: SharePermissionValue
  subjectId: string
}

/**
 * One incoming share row of `/v1/shares/inbox` — title- and grantor-enriched.
 * `accepted_at === null` means pending (awaiting the recipient's consent); a
 * number means accepted (active access).
 */
export type InboxShare = {
  accepted_at: number | null
  created_at: number
  granted_by_display_name: string | null
  granted_by_sub: string
  id: string
  permission: SharePermissionValue
  resource_id: string
  resource_title: string
  resource_type: string
}

/** `/v1/shares/inbox` payload: pending (consent queue) + accepted (shared
 * with me), split server-side on `accepted_at`. */
export type SharingInbox = {
  accepted: InboxShare[]
  pending: InboxShare[]
}

/** One outgoing row of `/v1/shares/mine`: a resource I shared, grouped across
 * its recipients with active and still-pending counts. */
export type OutgoingShare = {
  pending_count: number
  resource_id: string
  resource_title: string
  resource_type: string
  share_count: number
}
