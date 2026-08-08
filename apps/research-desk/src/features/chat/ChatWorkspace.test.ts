import { describe, expect, it } from 'vitest'

import { shouldClearAcceptedChatDraft } from './ChatWorkspace'

describe('chat draft submission ownership', () => {
  it('preserves text and pills until preflight accepts the exact snapshot', () => {
    expect(shouldClearAcceptedChatDraft(
      false,
      'Question',
      'Question',
      ['file:f1'],
      ['file:f1'],
    )).toBe(false)
    expect(shouldClearAcceptedChatDraft(
      true,
      'Question with follow-up',
      'Question',
      ['file:f1'],
      ['file:f1'],
    )).toBe(false)
    expect(shouldClearAcceptedChatDraft(
      true,
      'Question',
      'Question',
      ['file:f1', 'file:f2'],
      ['file:f1'],
    )).toBe(false)
    expect(shouldClearAcceptedChatDraft(
      true,
      'Question',
      'Question',
      ['file:f1'],
      ['file:f1'],
    )).toBe(true)
  })
})
