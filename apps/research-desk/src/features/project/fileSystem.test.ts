import { describe, expect, it } from 'vitest'
import { resolvePinnedExplorerFromManifest, resolveVectorIndexesFromManifest } from './fileSystem'
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

describe('resolvePinnedExplorerFromManifest', () => {
  it('filters duplicate and stale pinned ids against known project ids', () => {
    const result = resolvePinnedExplorerFromManifest({
      chatThreadIds: ['ct-1', 'ct-1', 'missing'],
      editorDocumentIds: ['doc-1', 42, 'missing'],
      knowledgeSessionIds: ['ks-1', 'missing'],
    }, {
      chatThreadIds: ['ct-1'],
      editorDocumentIds: ['doc-1'],
      knowledgeSessionIds: ['ks-1'],
    })

    expect(result).toEqual({
      chatThreadIds: ['ct-1'],
      editorDocumentIds: ['doc-1'],
      knowledgeSessionIds: ['ks-1'],
    })
  })

  it('keeps knowledge pins when no valid server-session list is available yet', () => {
    const result = resolvePinnedExplorerFromManifest({
      chatThreadIds: ['missing'],
      editorDocumentIds: ['missing'],
      knowledgeSessionIds: ['ks-server'],
    }, {
      chatThreadIds: [],
      editorDocumentIds: [],
    })

    expect(result).toEqual({
      chatThreadIds: [],
      editorDocumentIds: [],
      knowledgeSessionIds: ['ks-server'],
    })
  })
})

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

  it('reconciles a persisted "indexing" status on load (no run survives a reload)', () => {
    // A reindex that crashed mid-run leaves the record at "indexing" with no
    // live job (indexingJobs is never serialized). Loading must reconcile it
    // to the pre-run status so the UI is not wedged and (M6c) its server
    // autosave is not deferred forever: stale if any member still pending...
    const stalish = resolveVectorIndexesFromManifest({
      vector_indexes: [{
        createdAt: '', id: 'idx1', model: 'text-embedding-3-large', status: 'indexing',
        title: 'A', updatedAt: '',
        members: [{ fileId: 'f1', state: 'embedded' }, { fileId: 'f2', state: 'pending' }],
      }],
    }, fileAssets)
    expect(stalish.vectorIndexes.idx1.status).toBe('stale')

    // ...ready when every member is already embedded.
    const readyish = resolveVectorIndexesFromManifest({
      vector_indexes: [{
        createdAt: '', id: 'idx1', model: 'text-embedding-3-large', status: 'indexing',
        title: 'A', updatedAt: '',
        members: [{ fileId: 'f1', state: 'embedded' }],
      }],
    }, fileAssets)
    expect(readyish.vectorIndexes.idx1.status).toBe('ready')
  })

  it('reconstructs history, lastError and serverCollectionId (Frozen-Rebuild guard)', () => {
    const manifest = {
      vector_index_order: ['idx1'],
      vector_indexes: [
        {
          createdAt: '2026-01-01T00:00:00.000Z',
          dims: 3072,
          handle: 'eu',
          history: [
            { documents: 4, durationMs: 47_000, finishedAt: '2026-06-15T08:41:00.000Z', result: 'ok', startedAt: '2026-06-15T08:40:13.000Z' },
            { documents: 0, durationMs: 6_000, error: 'backend down', finishedAt: '2026-06-10T11:20:00.000Z', result: 'error', startedAt: '2026-06-10T11:19:54.000Z' },
          ],
          id: 'idx1',
          lastError: 'backend down',
          members: [{ fileId: 'f1', state: 'embedded' }],
          model: 'text-embedding-3-large',
          serverCollectionId: 'kc_live',
          status: 'ready',
          title: 'EU',
          updatedAt: '2026-01-01T00:00:00.000Z',
        },
      ],
    }
    const idx = resolveVectorIndexesFromManifest(manifest, fileAssets).vectorIndexes.idx1
    expect(idx.serverCollectionId).toBe('kc_live')
    expect(idx.lastError).toBe('backend down')
    expect(idx.history).toHaveLength(2)
    expect(idx.history?.[0]).toMatchObject({ documents: 4, result: 'ok' })
    expect(idx.history?.[1]).toMatchObject({ error: 'backend down', result: 'error' })
  })
})
