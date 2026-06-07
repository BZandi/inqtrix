import { describe, expect, it } from 'vitest'
import { resolveVectorIndexesFromManifest } from './fileSystem'
import type { FileAssetRecord } from './types'

function makeAsset(id: string): FileAssetRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    extractedText: `${id} content`,
    fileName: `${id}.txt`,
    groupId: null,
    id,
    label: id,
    mimeType: 'text/plain',
    origin: 'library',
    pageCount: null,
    parseStatus: 'parsed',
    parseWarning: null,
    sectionId: 'file-section-temp',
    sizeBytes: 12,
    textTruncated: false,
    title: `${id}.txt`,
    updatedAt: '2026-01-01T00:00:00.000Z',
  }
}

describe('resolveVectorIndexesFromManifest', () => {
  const fileAssets: Record<string, FileAssetRecord> = { f1: makeAsset('f1'), f2: makeAsset('f2') }

  it('returns empty indexes for an old manifest without the vector keys', () => {
    const result = resolveVectorIndexesFromManifest({}, fileAssets)
    expect(result.vectorIndexOrder).toEqual([])
    expect(result.vectorIndexes).toEqual({})
  })

  it('drops members whose source asset is missing', () => {
    const manifest = {
      vector_index_order: ['idx1'],
      vector_indexes: [
        {
          createdAt: '2026-01-01T00:00:00.000Z',
          dims: 3072,
          handle: 'eu',
          id: 'idx1',
          members: [
            { fileId: 'f1', state: 'embedded' },
            { fileId: 'missing', state: 'pending' },
          ],
          model: 'text-embedding-3-large',
          status: 'ready',
          title: 'EU',
          updatedAt: '2026-01-01T00:00:00.000Z',
        },
      ],
    }
    const result = resolveVectorIndexesFromManifest(manifest, fileAssets)
    expect(result.vectorIndexOrder).toEqual(['idx1'])
    expect(result.vectorIndexes.idx1.members).toEqual([{ fileId: 'f1', state: 'embedded' }])
  })

  it('defaults an unknown model/status and reconstructs the order', () => {
    const manifest = {
      vector_indexes: [
        { createdAt: '', id: 'idx1', members: [], model: 'bogus', status: 'weird', title: 'A', updatedAt: '' },
      ],
    }
    const result = resolveVectorIndexesFromManifest(manifest, fileAssets)
    expect(result.vectorIndexOrder).toEqual(['idx1'])
    expect(result.vectorIndexes.idx1.model).toBe('text-embedding-3-large')
    expect(result.vectorIndexes.idx1.dims).toBe(3072)
    expect(result.vectorIndexes.idx1.status).toBe('ready')
  })
})
