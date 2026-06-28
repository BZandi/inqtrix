import { describe, expect, it } from 'vitest'
import {
  applyComposerReportProfilePreset,
  buildComposerRequest,
  defaultComposerFormState,
} from './components/Composer'

describe('Research Desk composer profile presets', () => {
  it('defaults to the visible deep preset values sent in the payload', () => {
    const request = buildComposerRequest(
      { ...defaultComposerFormState, question: '  What changed?  ' },
      '  What changed?  ',
      'default',
    )

    expect(request).toMatchObject({
      agentOverrides: {
        confidenceStop: 8,
        firstRoundQueries: 10,
        maxRounds: 4,
        minRounds: 2,
        reportProfile: 'deep',
      },
      mode: 'research',
      question: 'What changed?',
      stack: 'default',
    })
  })

  it('switches profile presets while preserving the draft and strategy toggles', () => {
    const base = {
      ...defaultComposerFormState,
      firstRoundQueries: 4,
      maxRounds: 3,
      question: 'Keep this draft',
      webSearch: false,
    } as const

    expect(applyComposerReportProfilePreset(base, 'compact')).toMatchObject({
      confidenceStop: 7,
      firstRoundQueries: 6,
      maxRounds: 2,
      minRounds: 1,
      question: 'Keep this draft',
      reportProfile: 'compact',
      webSearch: false,
    })
    expect(applyComposerReportProfilePreset(base, 'deep')).toMatchObject({
      confidenceStop: 8,
      firstRoundQueries: 10,
      maxRounds: 4,
      minRounds: 2,
      question: 'Keep this draft',
      reportProfile: 'deep',
      webSearch: false,
    })
  })

  it('serializes ten first-round queries as an explicit user-visible value', () => {
    const request = buildComposerRequest(
      { ...defaultComposerFormState, firstRoundQueries: 10, question: 'Deep run' },
      'Deep run',
      'research',
    )

    expect(request.agentOverrides?.firstRoundQueries).toBe(10)
  })
})
