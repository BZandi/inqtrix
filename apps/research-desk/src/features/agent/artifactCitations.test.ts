import { describe, expect, it } from 'vitest'

import {
  agentArtifactReferences,
  agentReferenceAsKnowledge,
  linkifyAgentArtifactCitations,
} from './artifactCitations'

describe('agent artifact citations', () => {
  it('normalizes web and RAG provenance, including legacy refs without excerpts', () => {
    const refs = agentArtifactReferences([
      {
        citation_id: 'citation-1',
        citation_ids: ['citation-1'],
        excerpt: 'Supporting web passage',
        grounded_support: 'Provider-grounded market statement',
        label: 'W1',
        provider_snippet: 'Provider citation metadata',
        query_id: 'query-1',
        query_ids: ['query-1'],
        reference_id: 'reference-1',
        source_id: 'source-1',
        source_run_id: 'run-1',
        source_run_ids: ['run-1'],
        title: 'Market report',
        url: 'https://example.com/report',
      },
      {
        chunk_id: 'chunk-1',
        chunk_index: 4,
        collection_id: 'collection-1',
        document_id: 'doc-1',
        generation_id: 'generation-1',
        label: 'K1',
        provenance_status: 'verified_span',
        revision_id: 'revision-1',
        source_span: {
          document_content_hash: 'sha256:document',
          end: 42,
          offset_unit: 'utf8_byte',
          start: 12,
        },
        title: 'Internal study',
      },
    ])
    expect(refs).toEqual([
      expect.objectContaining({
        excerpt: 'Supporting web passage',
        groundedSupport: 'Provider-grounded market statement',
        label: 'W1',
        providerSnippet: 'Provider citation metadata',
        queryId: 'query-1',
        sourceRunId: 'run-1',
      }),
      expect.objectContaining({
        chunkIndex: 4,
        documentId: 'doc-1',
        excerpt: null,
        generationId: 'generation-1',
        groundedSupport: null,
        label: 'K1',
        provenanceStatus: 'verified_span',
        revisionId: 'revision-1',
        sourceSpan: {
          documentContentHash: 'sha256:document',
          end: 42,
          offsetUnit: 'utf8_byte',
          start: 12,
        },
      }),
    ])
    expect(agentReferenceAsKnowledge(refs[1]).url).toBe(
      'inqtrix://documents/doc-1',
    )
  })

  it('linkifies only labels present in the artifact ledger', () => {
    const refs = agentArtifactReferences([
      { label: 'W1', url: 'https://example.com', title: 'Example' },
      { label: 'K2', document_id: 'doc-2', chunk_index: 0 },
    ])
    expect(linkifyAgentArtifactCitations(
      'Aussage [W1] mit K2W1; W9 bleibt Text.',
      refs,
    )).toBe(
      'Aussage [W1](#kref-W1) mit [K2](#kref-K2)[W1](#kref-W1); W9 bleibt Text.',
    )
  })

  it('keeps a provider search result inspectable when no URL was returned', () => {
    const refs = agentArtifactReferences([
      {
        grounded_support: 'Frankreich gewann die WM 2018.',
        label: 'W1',
        query_id: 'query-answer-only',
        query_ids: ['query-answer-only'],
        reference_id: 'reference-answer-only',
        title: 'Fußball-Weltmeisterschaft 2018 Sieger',
      },
    ])

    expect(refs).toEqual([
      expect.objectContaining({
        documentId: null,
        groundedSupport: 'Frankreich gewann die WM 2018.',
        label: 'W1',
        queryId: 'query-answer-only',
        url: null,
      }),
    ])
  })
})

describe('delegated research citations', () => {
  it('keeps an E-labelled reference from a delegated research run', () => {
    // The kernel hands work to research; that run's evidence keeps its own
    // labels. Dropping them emptied the evidence panel on exactly the answers
    // that had done the most work.
    const refs = agentArtifactReferences([
      {
        label: 'E3',
        url: 'https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng',
        query_id: 'qry_36bb4f8bfe38a2',
        source_run_id: 'run_13c5bcac58ff44a59bab399b4090bcfd',
        tier: 'primary',
      },
    ])

    expect(refs).toHaveLength(1)
    expect(refs[0]?.label).toBe('E3')
    expect(refs[0]?.sourceRunId).toBe('run_13c5bcac58ff44a59bab399b4090bcfd')
  })

  it('still drops a label outside the citation vocabulary', () => {
    expect(agentArtifactReferences([
      { label: 'X7', url: 'https://example.invalid/x' },
      { label: 'note', url: 'https://example.invalid/n' },
    ])).toHaveLength(0)
  })
})

describe('citation target', () => {
  const refs = agentArtifactReferences([
    { label: 'E3', url: 'https://eur-lex.europa.eu/x', query_id: 'q1' },
  ])

  it('sends a known citation to the evidence panel, not out of the app', () => {
    // The panel carries the excerpt, the provenance and the verification
    // status; the URL stays reachable from there. A click on evidence should
    // land on the evidence.
    const out = linkifyAgentArtifactCitations(
      'Stand [E3](https://eur-lex.europa.eu/x) gilt die Verordnung.',
      refs,
    )

    expect(out).toContain('[E3](#kref-E3)')
    expect(out).not.toContain('https://eur-lex.europa.eu/x')
  })

  it('leaves an ordinary link alone', () => {
    const out = linkifyAgentArtifactCitations(
      'Siehe [Z9](https://example.invalid/z) und [den Bericht](https://example.invalid/r).',
      refs,
    )

    expect(out).toContain('[Z9](https://example.invalid/z)')
    expect(out).toContain('[den Bericht](https://example.invalid/r)')
  })
})
