import { describe, expect, it } from 'vitest'

import { initialsFor } from './avatar'

describe('initialsFor', () => {
  it('uses first and last word of the display name', () => {
    expect(initialsFor('Alice Beispiel', null)).toBe('AB')
    expect(initialsFor('Alice Marie Beispiel', null)).toBe('AB')
  })

  it('uses a single initial for one-word names', () => {
    expect(initialsFor('admin', null)).toBe('A')
  })

  it('falls back to the email local part', () => {
    expect(initialsFor(null, 'carla@example.com')).toBe('C')
    expect(initialsFor('  ', 'carla@example.com')).toBe('C')
  })

  it('never renders empty', () => {
    expect(initialsFor(null, null)).toBe('?')
    expect(initialsFor('', '')).toBe('?')
  })
})
