import { describe, expect, it } from 'vitest'

import { taskResultReferenceGroups } from './taskResultReferences'

describe('taskResultReferenceGroups', () => {
  it('deduplicates web sources by URL without creating title groups', () => {
    const groups = taskResultReferenceGroups([
      {
        label: 'W1',
        title: 'Report',
        url: 'https://Example.test/report?utm_source=newsletter#section',
      },
      {
        label: 'W2',
        title: 'Duplicate title',
        url: 'https://example.test/report?fbclid=tracking',
      },
    ])

    expect(groups).toHaveLength(1)
    expect(groups[0]?.kind).toBe('web')
    if (groups[0]?.kind === 'web') {
      expect(groups[0].reference.title).toBe('Report')
      expect(groups[0].reference.domain).toBe('example.test')
      expect(groups[0].reference.label).toBe('W1')
    }
  })

  it('groups several chunks of one document but keeps distinct chunks', () => {
    const groups = taskResultReferenceGroups([
      { document_id: 'doc-1', chunk_index: 0, title: 'Study', excerpt: 'A' },
      { document_id: 'doc-1', chunk_index: 1, title: 'Study', excerpt: 'B' },
      { document_id: 'doc-1', chunk_index: 1, title: 'Study', excerpt: 'B' },
    ])

    expect(groups).toEqual([
      {
        kind: 'document',
        title: 'Study',
        references: [
          expect.objectContaining({ chunkIndex: 0, excerpt: 'A' }),
          expect.objectContaining({ chunkIndex: 1, excerpt: 'B' }),
        ],
      },
    ])
  })
})
