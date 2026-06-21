import type {
  ShareInvitee,
  SharedWithMeEntry,
  ShareRecordInfo,
  UserSearchResult,
} from './types'

/** The workspace owner shown in demo mode (matches the quota admin demo). */
export const DEMO_OWNER = {
  displayName: 'Olga Owner',
  email: 'olga@example.com',
  subject: 'user-olga',
}

/** The owned run that carries seeded shares + a badge in the demo. */
export const DEMO_SHARED_RUN_ID = 'run_52f64e5c818049cdae2d63feb25dc10b'
/** A run shared INTO the demo workspace (the recipient "shared with me" view). */
export const DEMO_SHARED_IN_RUN_ID = 'run_1e52ccb1a21f45f6a135052f321cd5f3'

// 2026-01-01T00:00:00Z — neutral fixed instant (anonymised demo data).
const DEMO_TS = 1_767_225_600

/** People the share-dialog typeahead resolves in demo (no backend). */
const demoPeople: UserSearchResult[] = [
  { display_name: 'Rita Recipient', email: 'rita@example.com', subject: 'user-rita' },
  { display_name: 'Stefan Schulz', email: 'stefan@example.com', subject: 'user-stefan' },
  { display_name: 'Bianca Brandt', email: 'bianca@example.com', subject: 'user-bianca' },
  { display_name: 'Carlos Mendez', email: 'carlos@example.com', subject: 'user-carlos' },
]

export function searchDemoUsers(query: string): UserSearchResult[] {
  const q = query.trim().toLowerCase()
  if (q.length < 2) return []
  return demoPeople.filter(
    (user) =>
      (user.display_name ?? '').toLowerCase().includes(q) ||
      (user.email ?? '').toLowerCase().includes(q) ||
      user.subject.toLowerCase().includes(q),
  )
}

// In-memory share store so grant/revoke in the dialog feel live offline.
// Keyed by `${resource_type}:${resource_id}`; seeded lazily on first read.
const store = new Map<string, ShareRecordInfo[]>()
let seeded = false
let counter = 1

const storeKey = (resourceType: string, resourceId: string) =>
  `${resourceType}:${resourceId}`

function makeRecord(
  resourceType: string,
  resourceId: string,
  person: UserSearchResult,
  permission: ShareInvitee['permission'],
): ShareRecordInfo {
  counter += 1
  return {
    created_at: DEMO_TS,
    display_name: person.display_name,
    email: person.email,
    granted_by_sub: DEMO_OWNER.subject,
    id: `demo-share-${counter}`,
    permission,
    resource_id: resourceId,
    resource_type: resourceType,
    subject_id: person.subject,
    subject_type: 'user',
  }
}

function ensureSeeded() {
  if (seeded) return
  seeded = true
  store.set(storeKey('run', DEMO_SHARED_RUN_ID), [
    makeRecord('run', DEMO_SHARED_RUN_ID, demoPeople[0], 'view'),
    makeRecord('run', DEMO_SHARED_RUN_ID, demoPeople[1], 'edit'),
  ])
}

/** Drop any in-session grant/revoke so re-entering demo starts fresh — the
 * rebuild-on-toggle invariant the rest of the demo state holds. Called when
 * the demo seed is (re)built. */
export function resetDemoShares() {
  store.clear()
  seeded = false
  counter = 1
}

export function listDemoShares(
  resourceType: string,
  resourceId: string,
): ShareRecordInfo[] {
  ensureSeeded()
  return store.get(storeKey(resourceType, resourceId)) ?? []
}

export function grantDemoShares(
  resourceType: string,
  resourceId: string,
  invitees: ShareInvitee[],
) {
  ensureSeeded()
  const key = storeKey(resourceType, resourceId)
  const people = new Map(demoPeople.map((person) => [person.subject, person]))
  const next = [...(store.get(key) ?? [])]
  for (const invitee of invitees) {
    const person = people.get(invitee.subjectId) ?? {
      display_name: null,
      email: null,
      subject: invitee.subjectId,
    }
    const record = makeRecord(resourceType, resourceId, person, invitee.permission)
    const index = next.findIndex((entry) => entry.subject_id === invitee.subjectId)
    if (index >= 0) next[index] = record
    else next.push(record)
  }
  store.set(key, next)
}

export function revokeDemoShare(shareId: string) {
  for (const [key, records] of store) {
    const next = records.filter((record) => record.id !== shareId)
    if (next.length !== records.length) store.set(key, next)
  }
}

export function demoOutgoingShareCounts(
  resourceType: string,
  resourceIds: readonly string[],
): Record<string, number> {
  ensureSeeded()
  const counts: Record<string, number> = {}
  for (const id of resourceIds) {
    const count = (store.get(storeKey(resourceType, id)) ?? []).length
    if (count > 0) counts[id] = count
  }
  return counts
}

/** Resources shared INTO the demo workspace (the "Mit mir geteilt" group). */
export const demoSharedWithMe: SharedWithMeEntry[] = [
  {
    created_at: DEMO_TS,
    granted_by_display_name: 'Bianca Brandt',
    granted_by_sub: 'user-bianca',
    permission: 'view',
    resource_id: DEMO_SHARED_IN_RUN_ID,
    resource_type: 'run',
  },
]
