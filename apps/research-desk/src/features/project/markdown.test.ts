import { describe, expect, it } from 'vitest'
import { createDefaultFileLibrarySections } from '@/features/files/sections'
import { createEmptyProjectState } from './seedProject'
import {
  parseFileAsset,
  parseProjectManifest,
  serializeFileAsset,
  serializeProjectManifest,
} from './markdown'
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

describe('serializeFileAsset / parseFileAsset', () => {
  it('round-trips an asset with all metadata', () => {
    const asset = makeAsset('f1', 'alpha', {
      groupId: 'g1',
      origin: 'chat',
      pageCount: 3,
      parseStatus: 'partial',
      parseWarning: 'Document shortened',
      sizeBytes: 42,
      textTruncated: true,
    })
    const file = serializeFileAsset(asset)
    expect(file.path).toBe('files/alpha.md')
    expect(parseFileAsset(file.contents)).toEqual(asset)
  })

  it('round-trips a null group and page count', () => {
    const asset = makeAsset('f2', 'beta')
    expect(parseFileAsset(serializeFileAsset(asset).contents)).toEqual(asset)
  })
})

describe('project manifest file library', () => {
  it('serializes and re-parses sections, groups and asset order', () => {
    const sections = createDefaultFileLibrarySections('2026-01-01T00:00:00.000Z')
    const group: FileGroupRecord = {
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'g1',
      sectionId: sections[1].id,
      title: 'Group',
      updatedAt: '2026-01-01T00:00:00.000Z',
    }
    const asset = makeAsset('f1', 'alpha', { groupId: 'g1', sectionId: sections[1].id })
    const state: ProjectState = {
      ...createEmptyProjectState(),
      fileAssetOrder: ['f1'],
      fileAssets: { f1: asset },
      fileGroupOrder: ['g1'],
      fileGroups: { g1: group },
      fileLibrarySectionOrder: sections.map((section) => section.id),
      fileLibrarySections: Object.fromEntries(sections.map((section) => [section.id, section])),
    }

    const data = parseProjectManifest(serializeProjectManifest(state).contents)

    expect(data.file_section_order).toEqual(sections.map((section) => section.id))
    expect((data.file_sections as unknown[]).length).toBe(3)
    expect(data.file_group_order).toEqual(['g1'])
    expect(data.file_asset_order).toEqual(['f1'])
  })
})
