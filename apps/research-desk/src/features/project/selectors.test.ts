import { describe, expect, it } from 'vitest'
import { createEmptyProjectState } from './seedProject'
import {
  chatAttachmentsFromRefs,
  chatContextRefKey,
  dedupeChatContextRefs,
  referenceDocsFromRefs,
} from './selectors'
import type { FileAssetRecord, FileGroupRecord, ProjectState } from './types'

function makeAsset(id: string, label: string, overrides: Partial<FileAssetRecord> = {}): FileAssetRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    extractedText: `${label} content`,
    fileName: `${label}.txt`,
    groupId: null,
    id,
    label,
    mimeType: 'text/plain',
    origin: 'library',
    pageCount: null,
    parseStatus: 'parsed',
    parseWarning: null,
    sectionId: 'file-section-temp',
    sizeBytes: 12,
    textTruncated: false,
    title: `${label}.txt`,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function stateWith(assets: FileAssetRecord[], groups: FileGroupRecord[] = []): ProjectState {
  const base = createEmptyProjectState()
  return {
    ...base,
    fileAssetOrder: assets.map((asset) => asset.id),
    fileAssets: Object.fromEntries(assets.map((asset) => [asset.id, asset])),
    fileGroupOrder: groups.map((group) => group.id),
    fileGroups: Object.fromEntries(groups.map((group) => [group.id, group])),
  }
}

describe('chatContextRefKey', () => {
  it('produces a stable key per reference kind', () => {
    expect(chatContextRefKey({ fileId: 'f1', kind: 'file-asset' })).toBe('file-asset:f1')
    expect(chatContextRefKey({ groupId: 'g1', kind: 'file-group' })).toBe('file-group:g1')
  })
})

describe('dedupeChatContextRefs', () => {
  it('keeps the first occurrence of each reference', () => {
    const refs = dedupeChatContextRefs([
      { fileId: 'f1', kind: 'file-asset' },
      { fileId: 'f1', kind: 'file-asset' },
      { groupId: 'g1', kind: 'file-group' },
    ])
    expect(refs).toEqual([
      { fileId: 'f1', kind: 'file-asset' },
      { groupId: 'g1', kind: 'file-group' },
    ])
  })
})

describe('chatAttachmentsFromRefs', () => {
  it('resolves a file-asset reference to one attachment', () => {
    const state = stateWith([makeAsset('f1', 'alpha')])
    const attachments = chatAttachmentsFromRefs(state, [{ fileId: 'f1', kind: 'file-asset' }])
    expect(attachments).toHaveLength(1)
    expect(attachments[0]).toMatchObject({ contentMarkdown: 'alpha content', kind: 'file-asset', label: 'alpha' })
  })

  it('expands a file-group reference to one attachment per member', () => {
    const group: FileGroupRecord = {
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'g1',
      sectionId: 'file-section-temp',
      title: 'Dossier',
      updatedAt: '2026-01-01T00:00:00.000Z',
    }
    const state = stateWith(
      [makeAsset('f1', 'alpha', { groupId: 'g1' }), makeAsset('f2', 'beta', { groupId: 'g1' })],
      [group],
    )
    const attachments = chatAttachmentsFromRefs(state, [{ groupId: 'g1', kind: 'file-group' }])
    expect(attachments.map((attachment) => attachment.label)).toEqual(['alpha', 'beta'])
    expect(attachments.every((attachment) => attachment.kind === 'file-group')).toBe(true)
  })
})

describe('referenceDocsFromRefs', () => {
  it('maps file and research-report refs to ReferenceDoc DTOs and skips chat rules', () => {
    const base = stateWith([makeAsset('f1', 'alpha', { pageCount: 3 })])
    const report = {
      result: { markdown: 'report body' },
      runId: 'r1',
      status: 'completed',
      summary: { title: 'My Report' },
    } as unknown as ProjectState['researchRuns'][string]
    const state: ProjectState = {
      ...base,
      researchRunOrder: ['r1'],
      researchRuns: { r1: report },
    }
    const docs = referenceDocsFromRefs(state, [
      { fileId: 'f1', kind: 'file-asset' },
      { kind: 'research-report', runId: 'r1' },
      { kind: 'chat-rule', ruleId: 'does-not-exist' },
    ])
    expect(docs).toEqual([
      { content: 'alpha content', label: 'alpha', pageCount: 3, sizeBytes: 12 },
      { content: 'report body', label: 'my-report', pageCount: null, sizeBytes: undefined },
    ])
  })
})
