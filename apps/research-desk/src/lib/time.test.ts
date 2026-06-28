import { describe, expect, it } from 'vitest'
import { formatMessageTimestamp } from './time'

describe('formatMessageTimestamp', () => {
  it('formats German chat-style timestamps with date and local time', () => {
    const label = formatMessageTimestamp('2026-06-27T12:34:00.000Z', 'de')

    expect(label).toMatch(/^27\.06\.2026 · \d{2}:\d{2}$/)
  })

  it('formats English chat-style timestamps with date and local time', () => {
    const label = formatMessageTimestamp('2026-06-27T12:34:00.000Z', 'en')

    expect(label).toMatch(/^Jun 27, 2026 · \d{1,2}:34 (AM|PM)$/)
  })
})
