import { describe, expect, it } from 'vitest'

import { editorCopy } from '@/features/editor/editorCopy'

import { translations } from './translations'

/** Every key path in `value`, joined with dots. Arrays and functions are
 * leaves: their identity as a key is what matters here, not their shape. */
function keyPaths(value: unknown, prefix = ''): string[] {
  if (
    typeof value !== 'object'
    || value === null
    || Array.isArray(value)
  ) {
    return prefix ? [prefix] : []
  }
  return Object.entries(value).flatMap(([key, child]) =>
    keyPaths(child, prefix ? `${prefix}.${key}` : key),
  )
}

function missingFrom(reference: unknown, candidate: unknown): string[] {
  const candidateKeys = new Set(keyPaths(candidate))
  return keyPaths(reference).filter((path) => !candidateKeys.has(path))
}

/** The `TranslationDictionary` union type already errors when a CONSUMED key
 * is missing from one locale, but it stays silent about a key that exists on
 * only one side and is never read — the kind that surfaces as an untranslated
 * string months later. These tests make the parity explicit. */
describe('locale parity', () => {
  it('every German key has an English counterpart', () => {
    expect(missingFrom(translations.de, translations.en)).toEqual([])
  })

  it('every English key has a German counterpart', () => {
    expect(missingFrom(translations.en, translations.de)).toEqual([])
  })

  it('every German editor string has an English counterpart', () => {
    expect(missingFrom(editorCopy.de, editorCopy.en)).toEqual([])
  })

  it('every English editor string has a German counterpart', () => {
    expect(missingFrom(editorCopy.en, editorCopy.de)).toEqual([])
  })
})

describe('AI transparency strings', () => {
  it('states that Inqtrix is an AI system on both notice lists', () => {
    expect(translations.de.authLock.notices[0]).toContain('KI-System')
    expect(translations.en.authLock.notices[0]).toContain('AI system')
  })

  it('carries a first-interaction line and an export notice per locale', () => {
    for (const locale of [translations.de, translations.en]) {
      expect(locale.aiTransparency.firstInteraction.trim()).not.toBe('')
      expect(locale.aiTransparency.exportNotice.trim()).not.toBe('')
    }
  })
})
