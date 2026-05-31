import { describe, expect, it } from 'vitest'
import { ingestFiles } from './ingest'
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
})
