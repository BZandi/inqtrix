import { describe, expect, it } from 'vitest'

import { costTier, formatTokens, reasoningPresets, speedLabel } from './modelCard'

describe('reasoningPresets', () => {
  it('hides the selector when a model has no effort control (Haiku)', () => {
    expect(reasoningPresets([])).toEqual([])
  })

  it('offers No think / Think / Think hard when reasoning can be disabled', () => {
    const presets = reasoningPresets(['none', 'low', 'medium', 'high', 'xhigh', 'max'])
    expect(presets.map((p) => p.label)).toEqual(['No think', 'Think', 'Think hard'])
    expect(presets[0].effort).toBe('none')
    expect(presets[2].effort).toBe('max') // top graded level
  })

  it('omits No think for a Pro model that cannot disable reasoning', () => {
    const presets = reasoningPresets(['medium', 'high', 'xhigh'])
    expect(presets.map((p) => p.label)).toEqual(['Think', 'Think hard'])
    expect(presets.some((p) => p.effort === 'none')).toBe(false)
  })
})

describe('costTier', () => {
  it('maps output price to coarse $ tiers with the exact value', () => {
    expect(costTier({ input_per_mtok: 1, output_per_mtok: 5 }).signs).toBe('$')
    expect(costTier({ input_per_mtok: 3, output_per_mtok: 15 }).signs).toBe('$$')
    expect(costTier({ input_per_mtok: 5, output_per_mtok: 25 }).signs).toBe('$$$')
    expect(costTier({ input_per_mtok: 30, output_per_mtok: 180 }).signs).toBe('$$$$')
    expect(costTier({ input_per_mtok: 5, output_per_mtok: 25 }).value).toBe('$25/1M')
  })
})

describe('formatTokens', () => {
  it('formats context windows compactly', () => {
    expect(formatTokens(1_000_000)).toBe('1M')
    expect(formatTokens(1_050_000)).toBe('1.05M')
    expect(formatTokens(200_000)).toBe('200k')
  })

  it('shows exact counts below 1k and thousands above', () => {
    expect(formatTokens(0)).toBe('0')
    expect(formatTokens(20)).toBe('20')
    expect(formatTokens(850)).toBe('850')
    expect(formatTokens(1200)).toBe('1.2k')
  })
})

describe('speedLabel', () => {
  it('capitalises the TEMPO label', () => {
    expect(speedLabel('mittel')).toBe('Mittel')
  })
})
