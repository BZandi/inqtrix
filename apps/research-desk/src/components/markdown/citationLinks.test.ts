import { describe, expect, it } from 'vitest'

import { citationLabelFromHref, linkifyCitationLabels } from './citationLinks'

const isAgentLabel = (label: string) => /^[KW]\d+$/.test(label)

describe('generic citation links', () => {
  it('linkifies mixed known web and knowledge labels, including adjacent runs', () => {
    const known = new Set(['K2', 'W1'])
    expect(linkifyCitationLabels(
      'Belegt durch [W1] und K2W1.',
      isAgentLabel,
      known,
    )).toBe(
      'Belegt durch [W1](#kref-W1) und [K2](#kref-K2)[W1](#kref-W1).',
    )
  })

  it('leaves unknown bare labels and unrelated label families untouched', () => {
    expect(linkifyCitationLabels(
      'K9 und [Q1] bleiben Text.',
      isAgentLabel,
      new Set(['K1']),
    )).toBe('K9 und [Q1] bleiben Text.')
  })

  it('can require bracketed labels to exist in the caller ledger', () => {
    expect(linkifyCitationLabels(
      '[W1] [W9]',
      isAgentLabel,
      new Set(['W1']),
      { requireKnownBracketed: true },
    )).toBe('[W1](#kref-W1) [W9]')
  })

  it('extracts only labels accepted by the caller', () => {
    expect(citationLabelFromHref('#kref-W3', isAgentLabel)).toBe('W3')
    expect(citationLabelFromHref('#kref-Q3', isAgentLabel)).toBeNull()
  })
})
