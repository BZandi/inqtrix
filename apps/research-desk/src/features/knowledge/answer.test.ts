import { describe, expect, it } from 'vitest'
import type { ResearchRunResult } from '@/features/researchRuns/types'
import { citationLabelFromHref, knowledgeAnswerFromRunResult, linkifyCitationLabels } from './answer'

const known = new Set(['K1', 'K2', 'K6'])

describe('linkifyCitationLabels', () => {
  it('linkifies bracketed [K#] markers (always)', () => {
    expect(linkifyCitationLabels('siehe [K1] und [K2]')).toBe(
      'siehe [K1](#kref-K1) und [K2](#kref-K2)',
    )
  })

  it('linkifies BARE K# markers that match a known reference label', () => {
    expect(linkifyCitationLabels('Recruiting ist Hochrisiko K1.', known)).toBe(
      'Recruiting ist Hochrisiko [K1](#kref-K1).',
    )
  })

  it('splits compact adjacent K# markers when every label is known', () => {
    expect(linkifyCitationLabels('Belegt durch K1K2.', known)).toBe(
      'Belegt durch [K1](#kref-K1)[K2](#kref-K2).',
    )
  })

  it('leaves compact adjacent K# markers untouched when one label is unknown', () => {
    expect(linkifyCitationLabels('Belegt durch K1K9.', known)).toBe('Belegt durch K1K9.')
  })

  it('leaves a bare K# that is NOT a known label untouched (no dead links)', () => {
    // K9 is not a reference here; K2 as a plain token only links when known.
    expect(linkifyCitationLabels('Kaliumoxid K9 ist egal', known)).toBe(
      'Kaliumoxid K9 ist egal',
    )
  })

  it('does not linkify bare K# without a known-labels set (brackets only)', () => {
    expect(linkifyCitationLabels('nur K1 ohne Klammern')).toBe('nur K1 ohne Klammern')
  })

  it('does not touch tokens already rewritten into #kref links', () => {
    const once = linkifyCitationLabels('[K1] und K2', known)
    // Idempotent: a second pass must not double-wrap the existing link.
    expect(linkifyCitationLabels(once, known)).toBe(once)
    expect(once).toBe('[K1](#kref-K1) und [K2](#kref-K2)')
  })

  it('does not linkify K# embedded in a word', () => {
    expect(linkifyCitationLabels('das Modell K1A', new Set(['K1']))).toBe('das Modell K1A')
  })
})

describe('knowledgeAnswerFromRunResult references', () => {
  it('carries the explicit excerpt + ids from the backend reference (reliable open)', () => {
    const result = {
      answer: 'Antwort [K1].',
      report_references: [
        {
          chunk_index: 7,
          // An unparseable URL shape — the explicit document_id must still win,
          // so the citation opens reliably.
          document_id: 'kd_abc',
          excerpt: 'Vortext. Die Haftung ist begrenzt. Nachtext.',
          label: 'K1',
          source_text: 'Vortext. Die Haftung ist begrenzt. Nachtext.',
          tier: 'primary',
          title: 'AI Act.pdf',
          url: 'inqtrix://unparseable-shape',
        },
      ],
    } as unknown as ResearchRunResult

    const ref = knowledgeAnswerFromRunResult(result).references[0]
    expect(ref.documentId).toBe('kd_abc')
    expect(ref.chunkIndex).toBe(7)
    expect(ref.excerpt).toBe('Vortext. Die Haftung ist begrenzt. Nachtext.')
    expect(ref.sourceText).toBe('Vortext. Die Haftung ist begrenzt. Nachtext.')
    expect(ref.title).toBe('AI Act.pdf')
  })

  it('falls back to the URL-parsed id when no explicit document_id (older payload)', () => {
    const result = {
      answer: 'Antwort [K1].',
      report_references: [
        { label: 'K1', tier: 'primary', url: 'https://x/v1/sources/kd_old?chunk=3' },
      ],
    } as unknown as ResearchRunResult

    const ref = knowledgeAnswerFromRunResult(result).references[0]
    expect(ref.documentId).toBe('kd_old')
    expect(ref.chunkIndex).toBe(3)
    expect(ref.excerpt).toBeNull()
  })
})

describe('citationLabelFromHref', () => {
  it('extracts the label from a #kref href', () => {
    expect(citationLabelFromHref('#kref-K3')).toBe('K3')
  })
  it('returns null for unrelated hrefs', () => {
    expect(citationLabelFromHref('https://example.com')).toBeNull()
    expect(citationLabelFromHref(null)).toBeNull()
  })
})
