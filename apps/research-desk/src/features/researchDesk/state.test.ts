import { describe, expect, it } from 'vitest'
import { createEmptyProjectState } from '@/features/project/seedProject'
import type { ChatRuleRecord, FileAssetRecord } from '@/features/project/types'
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

function makeRule(id: string, label: string, overrides: Partial<ChatRuleRecord> = {}): ChatRuleRecord {
  return {
    contentMarkdown: `${label} prompt`,
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    label,
    title: `${label} prompt`,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

describe('ui visibility reducer actions', () => {
  it('hides and shows the chat history panel', () => {
    const hidden = researchDeskReducer(createEmptyProjectState(), {
      isVisible: false,
      type: 'setChatHistoryVisible',
    })
    expect(hidden.ui.isChatHistoryVisible).toBe(false)
    const shown = researchDeskReducer(hidden, {
      isVisible: true,
      type: 'setChatHistoryVisible',
    })
    expect(shown.ui.isChatHistoryVisible).toBe(true)
  })
})

describe('chat folder reducer actions', () => {
  it('creates a chat thread inside the requested folder', () => {
    const withFolder = researchDeskReducer(createEmptyProjectState(), {
      title: 'Folder',
      type: 'createChatThreadGroup',
    })
    const groupId = withFolder.chatThreadGroupOrder[0]

    const next = researchDeskReducer(withFolder, {
      groupId,
      type: 'createChatThread',
    })
    const threadId = next.ui.selectedChatThreadId

    expect(threadId).toBeTruthy()
    expect(next.chatThreadGroupMemberships[threadId as string]).toBe(groupId)
    expect(next.chatThreadOrder[0]).toBe(threadId)
    expect(next.dirty).toBe(true)
  })
})

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

describe('chat rule reducer actions', () => {
  it('upserts legacy rules with prompt-library defaults', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      rule: makeRule('r1', 'legacy'),
      type: 'upsertChatRule',
    })

    expect(next.chatRules.r1).toMatchObject({
      category: 'instruction',
      includeInAutocomplete: true,
      linkedContextRefs: [],
      visibility: { chat: true, editor: true },
    })
    expect(next.chatRuleOrder).toEqual(['r1'])
  })

  it('keeps only database references on context-pack rules', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      rule: makeRule('r1', 'profile', {
        category: 'context',
        linkedContextRefs: [
          { fileId: 'f1', kind: 'file-asset' },
          { kind: 'chat-rule', ruleId: 'nested-rule' },
          { groupId: 'g1', kind: 'file-group' },
        ],
      }),
      type: 'upsertChatRule',
    })

    expect(next.chatRules.r1.linkedContextRefs).toEqual([
      { fileId: 'f1', kind: 'file-asset' },
      { groupId: 'g1', kind: 'file-group' },
    ])
  })

  it('stores rendered context-pack content on chat attachments', () => {
    const base = createEmptyProjectState()
    const seeded = {
      ...base,
      chatRuleOrder: ['r1'],
      chatRules: {
        r1: makeRule('r1', 'profile', {
          category: 'context',
          contentMarkdown: 'Apply the profile.\n{{context}}',
          linkedContextRefs: [{ fileId: 'f1', kind: 'file-asset' }],
        }),
      },
      fileAssetOrder: ['f1'],
      fileAssets: {
        f1: makeAsset('f1', 'alpha', { extractedText: 'Original profile content.' }),
      },
    }

    const next = researchDeskReducer(seeded, {
      assistantMessageId: 'a1',
      attachmentRefs: [{ kind: 'chat-rule', ruleId: 'r1' }],
      contentMarkdown: 'Use @rules:profile',
      createdAt: '2026-01-03T00:00:00.000Z',
      threadId: 'thread-1',
      type: 'startChatExchange',
      userMessageId: 'u1',
    })

    const attachment = next.chatThreads['thread-1'].messages[0].attachments?.[0]
    expect(attachment).toMatchObject({ kind: 'chat-rule', label: 'profile' })
    expect(attachment?.contentMarkdown).toContain('Apply the profile.')
    expect(attachment?.contentMarkdown).toContain('Original profile content.')
    expect(attachment?.contentMarkdown).not.toContain('{{context}}')
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

describe('vector index reducer actions', () => {
  function withAssets(...labels: string[]) {
    return researchDeskReducer(createEmptyProjectState(), {
      assets: labels.map((label) => makeAsset(label, label, { pageCount: 10 })),
      type: 'ingestFileAssets',
    })
  }

  it('creates an index with members as stale and an empty index as ready', () => {
    const seeded = withAssets('a', 'b')
    const withIndex = researchDeskReducer(seeded, { fileIds: ['a', 'b'], title: 'EU Recht', type: 'createVectorIndex' })
    const index = withIndex.vectorIndexes[withIndex.vectorIndexOrder[0]]
    expect(index.title).toBe('EU Recht')
    expect(index.handle).toBe('eu-recht')
    expect(index.dims).toBe(3072)
    expect(index.status).toBe('stale')
    expect(index.members.map((member) => member.fileId)).toEqual(['a', 'b'])
    expect(index.members.every((member) => member.state === 'pending')).toBe(true)

    const empty = researchDeskReducer(seeded, { fileIds: [], title: 'Empty', type: 'createVectorIndex' })
    expect(empty.vectorIndexes[empty.vectorIndexOrder[0]].status).toBe('ready')
  })

  it('assigns unique handles for duplicate titles', () => {
    let state = withAssets('a')
    state = researchDeskReducer(state, { fileIds: [], title: 'EU', type: 'createVectorIndex' })
    state = researchDeskReducer(state, { fileIds: [], title: 'EU', type: 'createVectorIndex' })
    const handles = state.vectorIndexOrder.map((id) => state.vectorIndexes[id].handle).sort()
    expect(handles).toEqual(['eu', 'eu-2'])
  })

  it('re-slugs the handle on rename', () => {
    let state = researchDeskReducer(withAssets('a'), { fileIds: ['a'], title: 'Old', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, title: 'Neuer Titel', type: 'renameVectorIndex' })
    expect(state.vectorIndexes[id].title).toBe('Neuer Titel')
    expect(state.vectorIndexes[id].handle).toBe('neuer-titel')
  })

  it('adds documents as pending and marks the index stale', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, type: 'markVectorIndexIndexing' })
    state = researchDeskReducer(state, { indexId: id, type: 'completeVectorIndexReindex' })
    expect(state.vectorIndexes[id].status).toBe('ready')
    state = researchDeskReducer(state, { fileIds: ['b'], indexId: id, type: 'addDocsToVectorIndex' })
    expect(state.vectorIndexes[id].status).toBe('stale')
    expect(state.vectorIndexes[id].members.find((member) => member.fileId === 'b')?.state).toBe('pending')
  })

  it('changing the model updates dims, marks stale, and resets members to pending', () => {
    let state = researchDeskReducer(withAssets('a'), { fileIds: ['a'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, type: 'markVectorIndexIndexing' })
    state = researchDeskReducer(state, { indexId: id, type: 'completeVectorIndexReindex' })
    expect(state.vectorIndexes[id].members[0].state).toBe('embedded')
    state = researchDeskReducer(state, { indexId: id, model: 'text-embedding-3-small', type: 'setVectorIndexModel' })
    expect(state.vectorIndexes[id].dims).toBe(1536)
    expect(state.vectorIndexes[id].status).toBe('stale')
    expect(state.vectorIndexes[id].members[0].state).toBe('pending')
  })

  it('reindex marks indexing then ready with embedded members; complete is a no-op otherwise', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    const indexing = researchDeskReducer(state, { indexId: id, type: 'markVectorIndexIndexing' })
    expect(indexing.vectorIndexes[id].status).toBe('indexing')
    state = researchDeskReducer(indexing, { indexId: id, type: 'completeVectorIndexReindex' })
    expect(state.vectorIndexes[id].status).toBe('ready')
    expect(state.vectorIndexes[id].members.every((member) => member.state === 'embedded')).toBe(true)
    expect(researchDeskReducer(state, { indexId: id, type: 'completeVectorIndexReindex' })).toBe(state)
  })

  it('removing the last member returns the index to ready', () => {
    let state = researchDeskReducer(withAssets('a'), { fileIds: ['a'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    expect(state.vectorIndexes[id].status).toBe('stale')
    state = researchDeskReducer(state, { fileId: 'a', indexId: id, type: 'removeDocFromVectorIndex' })
    expect(state.vectorIndexes[id].members).toHaveLength(0)
    expect(state.vectorIndexes[id].status).toBe('ready')
  })

  it('deleting a file drops it from every index membership', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { fileId: 'a', type: 'deleteFileAsset' })
    expect(state.fileAssets.a).toBeUndefined()
    expect(state.vectorIndexes[id].members.map((member) => member.fileId)).toEqual(['b'])
  })

  it('deleting a custom section removes its assets and their index memberships', () => {
    let state = researchDeskReducer(createEmptyProjectState(), { sectionId: '', title: 'Custom', type: 'createFileLibrarySection' })
    const sectionId = state.fileLibrarySectionOrder[state.fileLibrarySectionOrder.length - 1]
    state = researchDeskReducer(state, { assets: [makeAsset('a', 'a', { sectionId })], type: 'ingestFileAssets' })
    state = researchDeskReducer(state, { fileIds: ['a'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { sectionId, type: 'deleteFileLibrarySection' })
    expect(state.fileLibrarySections[sectionId]).toBeUndefined()
    expect(state.fileAssets.a).toBeUndefined()
    expect(state.vectorIndexes[id].members).toHaveLength(0)
  })

  it('refuses to delete the temporary section', () => {
    const state = createEmptyProjectState()
    expect(researchDeskReducer(state, { sectionId: 'file-section-temp', type: 'deleteFileLibrarySection' })).toBe(state)
  })
})
