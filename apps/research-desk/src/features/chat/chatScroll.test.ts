import { describe, expect, it } from 'vitest'

import {
  chatDistanceFromBottom,
  chatScrollModeForUpdate,
  isChatNearBottom,
} from './chatScroll'

describe('chat scroll helpers', () => {
  it('handles fractional bottom distances with the near-bottom threshold', () => {
    const metrics = { clientHeight: 808, scrollHeight: 3640, scrollTop: 2808.5 }

    expect(chatDistanceFromBottom(metrics)).toBe(23.5)
    expect(isChatNearBottom(metrics)).toBe(true)
    expect(isChatNearBottom(metrics, 1)).toBe(false)
  })

  it('forces direct bottom restore for thread switches', () => {
    expect(chatScrollModeForUpdate({
      hasActiveAssistantMessage: false,
      nearBottom: false,
      reduceMotion: false,
      threadChanged: true,
    })).toBe('auto')
  })

  it('only smooth-follows same-thread updates when the user is near the bottom', () => {
    expect(chatScrollModeForUpdate({
      hasActiveAssistantMessage: false,
      nearBottom: true,
      reduceMotion: false,
      threadChanged: false,
    })).toBe('smooth')
    expect(chatScrollModeForUpdate({
      hasActiveAssistantMessage: false,
      nearBottom: false,
      reduceMotion: false,
      threadChanged: false,
    })).toBe('none')
    expect(chatScrollModeForUpdate({
      hasActiveAssistantMessage: true,
      nearBottom: true,
      reduceMotion: false,
      threadChanged: false,
    })).toBe('auto')
  })
})
