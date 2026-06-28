import { describe, expect, it } from 'vitest'

import type { ServerAccountPreferences } from '@/api/inqtrixClient'
import type { ProjectPreferences } from '@/features/project/types'
import {
  preferencesFingerprint,
  preferencesFromServer,
  serverAccountPreferencesPayload,
} from './accountPreferencesSync'

const fallback: ProjectPreferences = {
  contrastMode: 'standard',
  locale: 'en',
  theme: 'system',
  themePreset: 'standard',
  userBubbleTone: 'gray',
}

describe('accountPreferencesSync', () => {
  it('round-trips a server row to preferences and back', () => {
    const server: ServerAccountPreferences = {
      contrast_mode: 'high', locale: 'de', theme: 'dark', theme_preset: 'sage',
      user_bubble_tone: 'mint',
      updated_at: 1_700_000_000,
    }
    const prefs = preferencesFromServer(server, fallback)
    expect(prefs).toEqual({
      contrastMode: 'high',
      locale: 'de',
      theme: 'dark',
      themePreset: 'sage',
      userBubbleTone: 'mint',
    })

    const payload = serverAccountPreferencesPayload(prefs, 42)
    expect(payload).toEqual({
      contrast_mode: 'high',
      locale: 'de',
      theme: 'dark',
      theme_preset: 'sage',
      user_bubble_tone: 'mint',
      updated_at: 42,
    })
  })

  it('falls back per field on an out-of-domain server value', () => {
    const prefs = preferencesFromServer(
      {
        contrast_mode: 'ultra',
        locale: 'fr',
        theme: 'neon',
        theme_preset: 'gold',
        user_bubble_tone: 'rainbow',
        updated_at: 1,
      },
      fallback,
    )
    expect(prefs).toEqual(fallback)
  })

  it('falls back to the local bubble tone when an old server row omits it', () => {
    const prefs = preferencesFromServer(
      { contrast_mode: 'high', locale: 'de', theme: 'dark', theme_preset: 'sage', updated_at: 1 },
      { ...fallback, userBubbleTone: 'sky' },
    )
    expect(prefs.userBubbleTone).toBe('sky')
  })

  it('fingerprint changes only when a preference field changes', () => {
    const a = preferencesFingerprint(fallback)
    expect(preferencesFingerprint({ ...fallback })).toBe(a)
    expect(preferencesFingerprint({ ...fallback, theme: 'dark' })).not.toBe(a)
    expect(preferencesFingerprint({ ...fallback, userBubbleTone: 'orange' })).not.toBe(a)
  })
})
