/**
 * Pure conversion between the local ProjectPreferences and the server
 * account-preferences wire shape (M6c account tier).
 *
 * Account preferences (theme / locale / contrast / bubble tone) follow the
 * USER across devices, so they live in their own per-user server row — separate
 * from the project, and deliberately NOT part of the project import. This module is the
 * tiny snake_case<->camelCase bridge; the enum normalization defends against a
 * malformed row (the server CHECK-constrains these, but the client stays
 * defensive like the other converters).
 */

import type { ServerAccountPreferences } from '@/api/inqtrixClient'
import type { Locale } from '@/i18n/translations'
import type { ProjectPreferences } from '@/features/project/types'
import type {
  ContrastMode,
  ThemeMode,
  ThemePreset,
  UserBubbleTone,
} from '@/theme/ThemeProvider'

const VALID_CONTRAST: ReadonlySet<string> = new Set(['standard', 'high'])
const VALID_LOCALE: ReadonlySet<string> = new Set(['de', 'en'])
const VALID_THEME: ReadonlySet<string> = new Set(['light', 'dark', 'system'])
const VALID_PRESET: ReadonlySet<string> = new Set(['standard', 'slate', 'graphite', 'sage'])
const VALID_USER_BUBBLE_TONE: ReadonlySet<string> = new Set([
  'gray',
  'mint',
  'orange',
  'sky',
  'violet',
  'ink',
])

/** One server row -> local preferences, falling back per field to the given
 * defaults when a value is out of domain (so a corrupt row never breaks the UI). */
export function preferencesFromServer(
  server: ServerAccountPreferences,
  fallback: ProjectPreferences,
): ProjectPreferences {
  return {
    contrastMode: VALID_CONTRAST.has(server.contrast_mode)
      ? (server.contrast_mode as ContrastMode)
      : fallback.contrastMode,
    locale: VALID_LOCALE.has(server.locale) ? (server.locale as Locale) : fallback.locale,
    theme: VALID_THEME.has(server.theme) ? (server.theme as ThemeMode) : fallback.theme,
    themePreset: VALID_PRESET.has(server.theme_preset)
      ? (server.theme_preset as ThemePreset)
      : fallback.themePreset,
    userBubbleTone:
      server.user_bubble_tone && VALID_USER_BUBBLE_TONE.has(server.user_bubble_tone)
        ? (server.user_bubble_tone as UserBubbleTone)
        : fallback.userBubbleTone,
  }
}

/** Local preferences -> server payload. ``updatedAt`` is a unix-seconds sync
 * stamp the caller supplies (the row has no other lifecycle). */
export function serverAccountPreferencesPayload(
  preferences: ProjectPreferences,
  updatedAt: number,
): {
  contrast_mode: string
  locale: string
  theme: string
  theme_preset: string
  user_bubble_tone: string
  updated_at: number
} {
  return {
    contrast_mode: preferences.contrastMode,
    locale: preferences.locale,
    theme: preferences.theme,
    theme_preset: preferences.themePreset,
    user_bubble_tone: preferences.userBubbleTone,
    updated_at: updatedAt,
  }
}

/** Equality key for the autosave diff — the account preference fields. */
export function preferencesFingerprint(preferences: ProjectPreferences): string {
  return [
    preferences.contrastMode,
    preferences.locale,
    preferences.theme,
    preferences.themePreset,
    preferences.userBubbleTone,
  ].join('|')
}
