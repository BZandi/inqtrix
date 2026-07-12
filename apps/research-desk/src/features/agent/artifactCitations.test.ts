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
        excerpt: 'Supporting web passage',
        grounded_support: 'Provider-grounded market statement',
        label: 'W1',
        title: 'Market report',
        url: 'https://example.com/report',
      },
      {
        chunk_index: 4,
        document_id: 'doc-1',
        label: 'K1',
        title: 'Internal study',
      },
    ])
    expect(refs).toEqual([
      expect.objectContaining({
        excerpt: 'Supporting web passage',
        groundedSupport: 'Provider-grounded market statement',
        label: 'W1',
      }),
      expect.objectContaining({
        chunkIndex: 4,
        documentId: 'doc-1',
        excerpt: null,
        groundedSupport: null,
        label: 'K1',
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
})
