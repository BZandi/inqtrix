import { describe, expect, it } from 'vitest'
import type { FileAssetRecord } from '@/features/project/types'
import { ingestFiles, scheduleServerParse } from './ingest'
import type { FileParser, ParsedFile } from './parsing'

const stubParsed: ParsedFile = {
  extractedText: 'hello',
  pageCount: null,
  parseStatus: 'parsed',
  parseWarning: null,
  textTruncated: false,
}

const stubParser: FileParser = {
  parse: async () => stubParsed,
  supports: () => true,
}

function textFile(name: string): File {
  return new File(['hello'], name, { type: 'text/plain' })
}

describe('ingestFiles', () => {
  it('shapes each file into an asset with the origin and section', async () => {
    const assets = await ingestFiles(
      [textFile('Report.txt')],
      { kind: 'editor', sectionId: 'sec-1' },
      stubParser,
    )
    expect(assets).toHaveLength(1)
    expect(assets[0]).toMatchObject({
      extractedText: 'hello',
      fileName: 'Report.txt',
      label: 'report',
      origin: 'editor',
      sectionId: 'sec-1',
    })
  })

  it('de-duplicates labels against existing and incoming files', async () => {
    const assets = await ingestFiles(
      [textFile('Doc.txt'), textFile('Doc.txt')],
      { kind: 'chat' },
      stubParser,
      ['doc'],
    )
    expect(assets.map((asset) => asset.label)).toEqual(['doc-2', 'doc-3'])
  })

  it('uploads the original bytes and tags the upload as client-parsed', async () => {
    const assets = await ingestFiles(
      [textFile('Doc.txt')],
      { kind: 'chat' },
      stubParser,
      [],
      async () => 'srv-file-1',
    )
    expect(assets[0].serverFileId).toBe('srv-file-1')
    expect(assets[0].parseWarning).toBeNull()
    // Upload is fast/local: text is the client parse, provenance is 'client'
    // (it upgrades to 'markitdown' later, at vector-index time).
    expect(assets[0].extractedText).toBe('hello')
    expect(assets[0].parserId).toBe('client')
  })

  it('keeps the asset local with a visible warning when the upload fails', async () => {
    const assets = await ingestFiles(
      [textFile('Doc.txt')],
      { kind: 'chat' },
      stubParser,
      [],
      async () => {
        throw new Error('server down')
      },
    )
    expect(assets[0].serverFileId).toBeNull()
    expect(assets[0].parseWarning).toContain('Server-Upload fehlgeschlagen')
    expect(assets[0].parseWarning).toContain('server down')
    expect(assets[0].extractedText).toBe('hello')
    expect(assets[0].parserId).toBe('client')
  })

  it('stays purely local without a server upload function', async () => {
    const assets = await ingestFiles([textFile('Doc.txt')], { kind: 'chat' }, stubParser)
    expect(assets[0].serverFileId).toBeNull()
    expect(assets[0].parserId).toBe('client')
  })
})

function asset(over: Partial<FileAssetRecord> = {}): FileAssetRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    extractedText: 'client text',
    fileName: 'a.pdf',
    groupId: null,
    id: 'a1',
    label: 'a',
    mimeType: 'application/pdf',
    origin: 'library',
    pageCount: null,
    parseStatus: 'parsed',
    parseWarning: null,
    sectionId: 's',
    sizeBytes: 1,
    textTruncated: false,
    title: 'a',
    updatedAt: '2026-01-01T00:00:00.000Z',
    serverFileId: 'fl_1',
    parserId: 'client',
    ...over,
  }
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0))

describe('scheduleServerParse (background upgrade)', () => {
  it('marks pending then delivers the server text for a client-parsed, server-backed asset', async () => {
    const events: string[] = []
    scheduleServerParse([asset()], {
      fetchText: async () => 'SERVER TEXT',
      onPending: (id) => events.push(`pending:${id}`),
      onParsed: (id, text) => events.push(`parsed:${id}:${text}`),
      onFailed: (id) => events.push(`failed:${id}`),
    })
    await flush()
    expect(events).toEqual(['pending:a1', 'parsed:a1:SERVER TEXT'])
  })

  it('clears pending via onFailed when the server parse rejects (e.g. scanned PDF -> 422)', async () => {
    const events: string[] = []
    scheduleServerParse([asset()], {
      fetchText: async () => {
        throw new Error('422')
      },
      onPending: (id) => events.push(`pending:${id}`),
      onParsed: () => events.push('parsed'),
      onFailed: (id) => events.push(`failed:${id}`),
    })
    await flush()
    expect(events).toEqual(['pending:a1', 'failed:a1'])
  })

  it('skips assets with no server file or already parsed by MarkItDown', async () => {
    let fetches = 0
    scheduleServerParse(
      [asset({ id: 'local', serverFileId: null }), asset({ id: 'done', parserId: 'markitdown' })],
      {
        fetchText: async () => {
          fetches += 1
          return 'x'
        },
        onPending: () => undefined,
        onParsed: () => undefined,
        onFailed: () => undefined,
      },
    )
    await flush()
    expect(fetches).toBe(0)
  })
})
