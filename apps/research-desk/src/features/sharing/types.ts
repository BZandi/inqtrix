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
