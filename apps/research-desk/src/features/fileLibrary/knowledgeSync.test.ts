import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { FileAssetRecord, VectorIndexRecord } from '@/features/project/types'
import { ingestNewVectorIndexMembers, reindexVectorIndexOnServer } from './knowledgeSync'

const mocks = vi.hoisted(() => ({
  addKnowledgeDocument: vi.fn(),
  createKnowledgeCollection: vi.fn(),
  deleteKnowledgeCollection: vi.fn(),
  fetchKnowledgeDocumentText: vi.fn(),
  ingestKnowledgeFile: vi.fn(),
}))

vi.mock('@/api/inqtrixClient', () => ({
  addKnowledgeDocument: mocks.addKnowledgeDocument,
  createKnowledgeCollection: mocks.createKnowledgeCollection,
  deleteKnowledgeCollection: mocks.deleteKnowledgeCollection,
  fetchKnowledgeDocumentText: mocks.fetchKnowledgeDocumentText,
  ingestKnowledgeFile: mocks.ingestKnowledgeFile,
  hasHttpStatus: (error: unknown, status: number) =>
    error instanceof Error && (error as Error & { status?: number }).status === status,
}))

function makeIndex(overrides: Partial<VectorIndexRecord> = {}): VectorIndexRecord {
  return {
    createdAt: '2026-06-10T00:00:00.000Z',
    dims: 1536,
    handle: 'vertraege',
    id: 'vector-index-1',
    members: [{ fileId: 'file-1', state: 'pending' }],
    model: 'text-embedding-3-small',
    status: 'stale',
    title: 'Verträge',
    updatedAt: '2026-06-10T00:00:00.000Z',
    ...overrides,
  }
}

function makeAsset(overrides: Partial<FileAssetRecord> = {}): FileAssetRecord {
  return {
    createdAt: '2026-06-10T00:00:00.000Z',
    extractedText: 'Die Haftung ist begrenzt.',
    fileName: 'vertrag.pdf',
    groupId: null,
    id: 'file-1',
    label: 'rahmenvertrag',
    mimeType: 'application/pdf',
    origin: 'library',
    pageCount: 3,
    parseStatus: 'parsed',
    parseWarning: null,
    sectionId: 'section-1',
    sizeBytes: 1000,
    textTruncated: false,
    title: 'Rahmenvertrag.pdf',
    updatedAt: '2026-06-10T00:00:00.000Z',
    ...overrides,
  }
}

beforeEach(() => {
  mocks.addKnowledgeDocument.mockReset()
  mocks.createKnowledgeCollection.mockReset()
  mocks.deleteKnowledgeCollection.mockReset()
  mocks.fetchKnowledgeDocumentText.mockReset()
  mocks.ingestKnowledgeFile.mockReset()
  mocks.createKnowledgeCollection.mockResolvedValue({ id: 'kc_fresh' })
  mocks.addKnowledgeDocument.mockResolvedValue({ id: 'kd_1' })
  mocks.ingestKnowledgeFile.mockResolvedValue({ id: 'kd_file' })
  mocks.fetchKnowledgeDocumentText.mockResolvedValue({ id: 'kd_file', text: 'MARKITDOWN BODY' })
  mocks.deleteKnowledgeCollection.mockResolvedValue(undefined)
})

describe('reindexVectorIndexOnServer', () => {
  it('creates a collection with the index model and uploads member texts', async () => {
    const result = await reindexVectorIndexOnServer(
      makeIndex(),
      [makeAsset()],
      { apiKey: 'key', workspaceId: 'ws_test_123' },
    )

    expect(mocks.createKnowledgeCollection).toHaveBeenCalledWith(
      { embeddingModel: 'text-embedding-3-small', name: 'Verträge' },
      { apiKey: 'key', workspaceId: 'ws_test_123' },
    )
    expect(mocks.addKnowledgeDocument).toHaveBeenCalledWith(
      'kc_fresh',
      {
        metadata: { fileId: 'file-1', fileName: 'vertrag.pdf' },
        text: 'Die Haftung ist begrenzt.',
        title: 'Rahmenvertrag.pdf',
      },
      { apiKey: 'key', workspaceId: 'ws_test_123' },
    )
    expect(result).toEqual({
      collectionId: 'kc_fresh',
      serverCollectionModel: 'text-embedding-3-small',
      embeddedFileIds: ['file-1'],
      skippedFileIds: [],
      skippedFiles: [],
      uploadedDocuments: 1,
      reparsed: [],
      serverDocumentIds: { 'file-1': 'kd_1' },
    })
  })

  it('deletes the previous server collection before rebuilding', async () => {
    await reindexVectorIndexOnServer(
      makeIndex({ serverCollectionId: 'kc_old' }),
      [makeAsset()],
      {},
    )
    expect(mocks.deleteKnowledgeCollection).toHaveBeenCalledWith('kc_old', {})
  })

  it('tolerates a 404 on the stale previous collection', async () => {
    const gone = Object.assign(new Error('not found'), { status: 404 })
    mocks.deleteKnowledgeCollection.mockRejectedValueOnce(gone)

    const result = await reindexVectorIndexOnServer(
      makeIndex({ serverCollectionId: 'kc_old' }),
      [makeAsset()],
      {},
    )
    expect(result.collectionId).toBe('kc_fresh')
  })

  it('reports members without extracted text instead of dropping them silently', async () => {
    const result = await reindexVectorIndexOnServer(
      makeIndex(),
      [makeAsset(), makeAsset({ extractedText: '  ', id: 'file-2', label: 'Scan ohne Text' })],
      {},
    )
    expect(result.skippedFiles).toEqual(['Scan ohne Text'])
    expect(result.uploadedDocuments).toBe(1)
  })

  it('rebuilds to an empty collection and reports all-textless members as skipped', async () => {
    const result = await reindexVectorIndexOnServer(
      makeIndex({ serverCollectionId: 'kc_old' }),
      [makeAsset({ extractedText: '', label: 'Scan ohne Text' })],
      {},
    )

    expect(mocks.deleteKnowledgeCollection).toHaveBeenCalledWith('kc_old', {})
    expect(mocks.createKnowledgeCollection).toHaveBeenCalledWith(
      { embeddingModel: 'text-embedding-3-small', name: 'Verträge' },
      {},
    )
    expect(mocks.addKnowledgeDocument).not.toHaveBeenCalled()
    expect(mocks.ingestKnowledgeFile).not.toHaveBeenCalled()
    expect(result).toEqual({
      collectionId: 'kc_fresh',
      serverCollectionModel: 'text-embedding-3-small',
      embeddedFileIds: [],
      skippedFileIds: ['file-1'],
      skippedFiles: ['Scan ohne Text'],
      uploadedDocuments: 0,
      reparsed: [],
      serverDocumentIds: {},
    })
  })

  it('ingests via file_id when enabled and the asset has a server file', async () => {
    const result = await reindexVectorIndexOnServer(
      makeIndex(),
      [makeAsset({ serverFileId: 'srv-file-9' })],
      { useFileIngestion: true, workspaceId: 'ws_test_123' },
    )

    expect(mocks.ingestKnowledgeFile).toHaveBeenCalledWith(
      'kc_fresh',
      {
        fileId: 'srv-file-9',
        metadata: { fileId: 'file-1', fileName: 'vertrag.pdf', file_id: 'srv-file-9' },
        title: 'Rahmenvertrag.pdf',
      },
      { useFileIngestion: true, workspaceId: 'ws_test_123' },
    )
    expect(mocks.addKnowledgeDocument).not.toHaveBeenCalled()
    expect(result.uploadedDocuments).toBe(1)
  })

  it('embeds the stored text (skips the redundant re-parse) when it is already MarkItDown-grade', async () => {
    const result = await reindexVectorIndexOnServer(
      makeIndex(),
      [makeAsset({ serverFileId: 'srv-file-9', parserId: 'markitdown' })],
      { useFileIngestion: true, workspaceId: 'ws_test_123' },
    )
    // Upload already produced the server (MarkItDown) text — no second S3
    // fetch + parse; the stored text is embedded directly.
    expect(mocks.ingestKnowledgeFile).not.toHaveBeenCalled()
    expect(mocks.addKnowledgeDocument).toHaveBeenCalledWith(
      'kc_fresh',
      {
        metadata: { fileId: 'file-1', fileName: 'vertrag.pdf', file_id: 'srv-file-9' },
        text: 'Die Haftung ist begrenzt.',
        title: 'Rahmenvertrag.pdf',
      },
      { useFileIngestion: true, workspaceId: 'ws_test_123' },
    )
    expect(result.uploadedDocuments).toBe(1)
    // Already MarkItDown-grade: no re-parse, so nothing to back-fill.
    expect(mocks.fetchKnowledgeDocumentText).not.toHaveBeenCalled()
    expect(result.reparsed).toEqual([])
  })

  it('re-ingests the original file and returns its server text for back-fill when client-parsed', async () => {
    const result = await reindexVectorIndexOnServer(
      makeIndex(),
      [makeAsset({ serverFileId: 'srv-file-9', parserId: 'client' })],
      { useFileIngestion: true },
    )
    // Client-grade text — re-parse the original server-side for higher fidelity,
    // then hand the MarkItDown text back so the library asset upgrades.
    expect(mocks.ingestKnowledgeFile).toHaveBeenCalledTimes(1)
    expect(mocks.addKnowledgeDocument).not.toHaveBeenCalled()
    expect(mocks.fetchKnowledgeDocumentText).toHaveBeenCalledWith('kd_file', {
      useFileIngestion: true,
    })
    expect(result.reparsed).toEqual([{ assetId: 'file-1', text: 'MARKITDOWN BODY' }])
  })

  it('keeps the build succeeding (no back-fill) when reading the server text fails', async () => {
    mocks.fetchKnowledgeDocumentText.mockRejectedValueOnce(new Error('text read failed'))
    const result = await reindexVectorIndexOnServer(
      makeIndex(),
      [makeAsset({ serverFileId: 'srv-file-9', parserId: 'client' })],
      { useFileIngestion: true },
    )
    // Indexing already succeeded; the failed read only forgoes the upgrade.
    expect(result.uploadedDocuments).toBe(1)
    expect(result.reparsed).toEqual([])
  })

  it('counts a text-less asset as uploadable when file ingestion covers it', async () => {
    const result = await reindexVectorIndexOnServer(
      makeIndex(),
      [makeAsset({ extractedText: '', serverFileId: 'srv-file-9' })],
      { useFileIngestion: true },
    )
    expect(result.skippedFiles).toEqual([])
    expect(mocks.ingestKnowledgeFile).toHaveBeenCalledTimes(1)
  })

  it('falls back to the text path without the file ingestion capability', async () => {
    await reindexVectorIndexOnServer(
      makeIndex(),
      [makeAsset({ serverFileId: 'srv-file-9' })],
      {},
    )
    expect(mocks.ingestKnowledgeFile).not.toHaveBeenCalled()
    expect(mocks.addKnowledgeDocument).toHaveBeenCalledTimes(1)
  })

  it('falls back to the text path when the asset never reached the server', async () => {
    await reindexVectorIndexOnServer(
      makeIndex(),
      [makeAsset({ serverFileId: null })],
      { useFileIngestion: true },
    )
    expect(mocks.ingestKnowledgeFile).not.toHaveBeenCalled()
    expect(mocks.addKnowledgeDocument).toHaveBeenCalledTimes(1)
  })

  it('cleans up the half-filled collection when an upload fails', async () => {
    mocks.addKnowledgeDocument.mockRejectedValueOnce(new Error('embedding backend down'))

    await expect(
      reindexVectorIndexOnServer(makeIndex(), [makeAsset()], {}),
    ).rejects.toThrow('embedding backend down')
    expect(mocks.deleteKnowledgeCollection).toHaveBeenCalledWith('kc_fresh', {})
  })
})

describe('ingestNewVectorIndexMembers', () => {
  it('ingests only the new members into the existing collection (no rebuild)', async () => {
    const result = await ingestNewVectorIndexMembers(
      makeIndex({
        serverCollectionId: 'kc_live',
        serverCollectionModel: 'text-embedding-3-small',
      }),
      [makeAsset({ id: 'file-2', fileName: 'neu.pdf', title: 'Neu.pdf' })],
      { apiKey: 'key', workspaceId: 'ws_test_123' },
    )

    // The existing collection is reused — never deleted or recreated.
    expect(mocks.deleteKnowledgeCollection).not.toHaveBeenCalled()
    expect(mocks.createKnowledgeCollection).not.toHaveBeenCalled()
    expect(mocks.addKnowledgeDocument).toHaveBeenCalledWith(
      'kc_live',
      expect.objectContaining({ metadata: { fileId: 'file-2', fileName: 'neu.pdf' } }),
      { apiKey: 'key', workspaceId: 'ws_test_123' },
    )
    expect(result.collectionId).toBe('kc_live')
    expect(result.embeddedFileIds).toEqual(['file-2'])
    expect(result.skippedFileIds).toEqual([])
    // The backend document id is returned per member so removal can delete it.
    expect(result.serverDocumentIds).toEqual({ 'file-2': 'kd_1' })
  })

  it('reports text-less members as skipped instead of dropping them', async () => {
    const result = await ingestNewVectorIndexMembers(
      makeIndex({
        serverCollectionId: 'kc_live',
        serverCollectionModel: 'text-embedding-3-small',
      }),
      [makeAsset({ id: 'file-3', extractedText: '   ', label: 'Scan ohne Text' })],
      {},
    )
    expect(mocks.addKnowledgeDocument).not.toHaveBeenCalled()
    expect(result.embeddedFileIds).toEqual([])
    expect(result.skippedFileIds).toEqual(['file-3'])
    expect(result.skippedFiles).toEqual(['Scan ohne Text'])
  })

  it('fails loudly without an existing server collection', async () => {
    await expect(
      ingestNewVectorIndexMembers(makeIndex({ serverCollectionId: null }), [makeAsset()], {}),
    ).rejects.toThrow('requires an existing server collection')
  })

  it('reports server-confirmed per-member progress via onMemberDone', async () => {
    const events: Array<{ fileId: string; done: number; total: number; embedded: boolean }> = []
    await ingestNewVectorIndexMembers(
      makeIndex({
        serverCollectionId: 'kc_live',
        serverCollectionModel: 'text-embedding-3-small',
      }),
      [
        makeAsset({ id: 'file-a' }),
        makeAsset({ id: 'file-b', extractedText: '   ', label: 'Scan' }),
        makeAsset({ id: 'file-c' }),
      ],
      {},
      (event) => events.push(event),
    )
    // One callback per member, in order, with the REAL outcome (the text-less
    // 'file-b' is reported embedded:false, not silently dropped).
    expect(events).toEqual([
      { fileId: 'file-a', done: 1, total: 3, embedded: true },
      { fileId: 'file-b', done: 2, total: 3, embedded: false },
      { fileId: 'file-c', done: 3, total: 3, embedded: true },
    ])
  })
})
