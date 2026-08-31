import { describe, expect, it } from 'vitest'

import { codeBlockPickerState } from './codeBlockView'

/**
 * P5: the language picker writes ONLY in plain edit mode. In suggest
 * mode the schema guard hard-rejects attribute-only transactions (pin
 * in packages/editor-schema/tests/schema.test.ts) — the picker must be
 * visibly disabled there instead of surfacing that error; comment mode
 * and read-only lifecycles have no write path at all.
 */
describe('codeBlockPickerState', () => {
  it('enables the picker only in plain edit mode', () => {
    expect(codeBlockPickerState('edit', true)).toEqual({
      enabled: true,
      reason: null,
    })
    expect(codeBlockPickerState(undefined, true)).toEqual({
      enabled: true,
      reason: null,
    })
  })

  it('disables with a mode reason in suggest AND comment mode', () => {
    expect(codeBlockPickerState('suggest', true)).toEqual({
      enabled: false,
      reason: 'mode',
    })
    expect(codeBlockPickerState('comment', true)).toEqual({
      enabled: false,
      reason: 'mode',
    })
  })

  it('read-only beats every mode', () => {
    expect(codeBlockPickerState('edit', false)).toEqual({
      enabled: false,
      reason: 'readonly',
    })
    expect(codeBlockPickerState('suggest', false)).toEqual({
      enabled: false,
      reason: 'readonly',
    })
  })
})
