import { describe, expect, it } from 'vitest'
import { createEmptyProjectState } from '@/features/project/seedProject'
import type { FileAssetRecord } from '@/features/project/types'
import { researchDeskReducer } from './state'

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

describe('file-asset reducer actions', () => {
  it('ingests assets into the store and order', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha')],
      type: 'ingestFileAssets',
    })
    expect(next.fileAssetOrder).toContain('f1')
    expect(next.fileAssets.f1.label).toBe('alpha')
    expect(next.dirty).toBe(true)
  })

  it('renames an asset label', () => {
    const seeded = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha')],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(seeded, { fileId: 'f1', label: 'renamed', type: 'renameFileAsset' })
    expect(next.fileAssets.f1.label).toBe('renamed')
  })

  it('moves an asset into an existing section group', () => {
    const created = researchDeskReducer(createEmptyProjectState(), {
      sectionId: 'file-section-library',
      title: 'Group',
      type: 'createFileGroup',
    })
    const groupId = created.fileGroupOrder[0]
    const seeded = researchDeskReducer(created, {
      assets: [makeAsset('f1', 'alpha')],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(seeded, {
      fileId: 'f1',
      groupId,
      sectionId: 'file-section-library',
      type: 'moveFileAsset',
    })
    expect(next.fileAssets.f1.sectionId).toBe('file-section-library')
    expect(next.fileAssets.f1.groupId).toBe(groupId)
  })

  it('drops a move into a group that does not belong to the target section', () => {
    const seeded = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha')],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(seeded, {
      fileId: 'f1',
      groupId: 'does-not-exist',
      sectionId: 'file-section-library',
      type: 'moveFileAsset',
    })
    expect(next.fileAssets.f1.sectionId).toBe('file-section-library')
    expect(next.fileAssets.f1.groupId).toBeNull()
  })

  it('deletes an asset and strips its pending chat reference', () => {
    const seeded = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha')],
      type: 'ingestFileAssets',
    })
    const withRef = {
      ...seeded,
      ui: { ...seeded.ui, pendingChatAttachmentRefs: [{ fileId: 'f1', kind: 'file-asset' as const }] },
    }
    const next = researchDeskReducer(withRef, { fileId: 'f1', type: 'deleteFileAsset' })
    expect(next.fileAssets.f1).toBeUndefined()
    expect(next.fileAssetOrder).not.toContain('f1')
    expect(next.ui.pendingChatAttachmentRefs).toEqual([])
  })
})

describe('file-group reducer actions', () => {
  it('creates a group under a section', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      sectionId: 'file-section-library',
      title: 'New Group',
      type: 'createFileGroup',
    })
    expect(next.fileGroupOrder).toHaveLength(1)
    const groupId = next.fileGroupOrder[0]
    expect(next.fileGroups[groupId]).toMatchObject({ sectionId: 'file-section-library', title: 'New Group' })
  })

  it('reassigns members to no group when their group is deleted', () => {
    const created = researchDeskReducer(createEmptyProjectState(), {
      sectionId: 'file-section-library',
      title: 'Group',
      type: 'createFileGroup',
    })
    const groupId = created.fileGroupOrder[0]
    const withAsset = researchDeskReducer(created, {
      assets: [makeAsset('f1', 'alpha', { groupId, sectionId: 'file-section-library' })],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(withAsset, { groupId, type: 'deleteFileGroup' })
    expect(next.fileGroups[groupId]).toBeUndefined()
    expect(next.fileAssets.f1.groupId).toBeNull()
  })
})

describe('reorderChatContextInDraft', () => {
  function withPending(ruleIds: string[]) {
    const base = createEmptyProjectState()
    return {
      ...base,
      ui: {
        ...base.ui,
        pendingChatAttachmentRefs: ruleIds.map((ruleId) => ({ kind: 'chat-rule' as const, ruleId })),
      },
    }
  }

  it('moves a pending ref to a new index', () => {
    const next = researchDeskReducer(withPending(['a', 'b', 'c']), {
      fromIndex: 0,
      toIndex: 2,
      type: 'reorderChatContextInDraft',
    })
    expect(next.ui.pendingChatAttachmentRefs.map((ref) => (ref as { ruleId: string }).ruleId)).toEqual(['b', 'c', 'a'])
    expect(next.dirty).toBe(true)
  })

  it('ignores no-op and out-of-bounds reorders', () => {
    const seeded = withPending(['a', 'b'])
    expect(researchDeskReducer(seeded, { fromIndex: 0, toIndex: 0, type: 'reorderChatContextInDraft' })).toBe(seeded)
    expect(researchDeskReducer(seeded, { fromIndex: 0, toIndex: 5, type: 'reorderChatContextInDraft' })).toBe(seeded)
  })
})
