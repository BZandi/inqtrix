import { describe, expect, it } from 'vitest'
import {
  canCancelWithAccess,
  partitionJobsByAccess,
  personLabel,
  selectableSearchResults,
  sharedWithMeByResourceId,
  toggleSelectedUser,
} from './shareModel'
import type { SharedWithMeEntry, UserSearchResult } from './types'

const user = (subject: string): UserSearchResult => ({
  display_name: null,
  email: `${subject}@example.com`,
  subject,
})

describe('partitionJobsByAccess', () => {
  it('splits shared-in jobs out while preserving order', () => {
    const jobs = [
      { access: undefined, id: 'a' },
      { access: { permission: 'view' as const, via: 'share' as const }, id: 'b' },
      { access: undefined, id: 'c' },
    ]
    const { own, shared } = partitionJobsByAccess(jobs)
    expect(own.map((job) => job.id)).toEqual(['a', 'c'])
    expect(shared.map((job) => job.id)).toEqual(['b'])
  })

  it('returns empty shared group when nothing is shared in', () => {
    expect(partitionJobsByAccess([{ access: undefined, id: 'a' }]).shared).toEqual([])
  })
})

describe('canCancelWithAccess', () => {
  it('keeps owned runs cancellable', () => {
    expect(canCancelWithAccess(undefined)).toBe(true)
  })

  it('mirrors the server rule: view cannot cancel, edit can', () => {
    expect(canCancelWithAccess({ permission: 'view', via: 'share' })).toBe(false)
    expect(canCancelWithAccess({ permission: 'edit', via: 'share' })).toBe(true)
  })
})

describe('toggleSelectedUser', () => {
  it('adds unknown users and removes already-selected ones', () => {
    const once = toggleSelectedUser([], user('u1'))
    expect(once.map((entry) => entry.subject)).toEqual(['u1'])
    expect(toggleSelectedUser(once, user('u1'))).toEqual([])
  })
})

describe('selectableSearchResults', () => {
  it('hides users who already hold a share or are picked', () => {
    const results = [user('u1'), user('u2')]
    expect(
      selectableSearchResults(results, new Set(['u1'])).map((entry) => entry.subject),
    ).toEqual(['u2'])
  })
})

describe('personLabel', () => {
  it('prefers the display name, then email, then the fallback', () => {
    expect(personLabel('Alice B', 'a@example.com', 'sub')).toBe('Alice B')
    expect(personLabel('  ', 'a@example.com', 'sub')).toBe('a@example.com')
    expect(personLabel(null, null, 'sub')).toBe('sub')
  })
})

describe('sharedWithMeByResourceId', () => {
  it('keys entries by resource id', () => {
    const entry: SharedWithMeEntry = {
      created_at: 1,
      granted_by_display_name: 'Olga Owner',
      granted_by_sub: 'owner',
      permission: 'view',
      resource_id: 'run_1',
      resource_type: 'run',
    }
    expect(sharedWithMeByResourceId([entry]).get('run_1')).toBe(entry)
  })
})
