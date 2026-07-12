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
        firstRoundQueries: 8,
        maxRounds: 4,
        minRounds: 2,
        reportProfile: 'deep',
      },
      mode: 'research',
      question: 'What changed?',
      stack: 'default',
    })
  })

  it('switches profile presets while preserving the draft', () => {
    const base = {
      ...defaultComposerFormState,
      firstRoundQueries: 4,
      maxRounds: 3,
      question: 'Keep this draft',
    } as const

    expect(applyComposerReportProfilePreset(base, 'compact')).toMatchObject({
      confidenceStop: 7,
      firstRoundQueries: 6,
      maxRounds: 2,
      minRounds: 1,
      question: 'Keep this draft',
      reportProfile: 'compact',
    })
    expect(applyComposerReportProfilePreset(base, 'schnell')).toMatchObject({
      maxRounds: 1,
      minRounds: 1,
      question: 'Keep this draft',
      reportProfile: 'schnell',
    })
  })

  it('always submits a full research run', () => {
    const request = buildComposerRequest(
      { ...defaultComposerFormState, question: 'Q' },
      'Q',
      'default',
    )

    expect(request.mode).toBe('research')
  })

  it('serializes the visible eight-query maximum', () => {
    const request = buildComposerRequest(
      { ...defaultComposerFormState, firstRoundQueries: 8, question: 'Deep run' },
      'Deep run',
      'research',
    )

    expect(request.agentOverrides?.firstRoundQueries).toBe(8)
  })
})
