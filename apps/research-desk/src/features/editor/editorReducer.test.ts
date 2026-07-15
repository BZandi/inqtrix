import { describe, expect, it } from 'vitest'

import { createEmptyProjectState } from '@/features/project/seedProject'
import type {
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorFolderRecord,
  EditorSuggestionRecord,
  ProjectState,
} from '@/features/project/types'
import { researchDeskReducer } from '@/features/researchDesk/state'
import {
  planEditorCommentReconciliation,
  planEditorDocumentAutosave,
} from './useEditorHistoryApi'

function doc(
  id: string,
  overrides: Partial<EditorDocumentRecord> = {},
): EditorDocumentRecord {
  return {
    contentMarkdown: '',
    createdAt: '2026-01-01T00:00:00.000Z',
    folderId: null,
    id,
    revision: 1,
    source: 'blank',
    title: id,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function comment(id: string, documentId: string): EditorCommentThreadRecord {
  return {
    anchor: { from: 0, to: 1, selectedText: 'x', quoteBefore: '', quoteAfter: '' },
    commentMarkdown: id,
    createdAt: '2026-01-01T00:00:00.000Z',
    documentId,
    id,
    kind: 'collect',
    status: 'open',
    updatedAt: '2026-01-01T00:00:00.000Z',
  }
}

function withDoc(local: EditorDocumentRecord): ProjectState {
  const base = createEmptyProjectState()
  return {
    ...base,
    dirty: false,
    editorDocumentOrder: [local.id],
    editorDocuments: { [local.id]: local },
  }
}

describe('server editor hydration (M6b)', () => {
  it('adds server documents with empty body + order, WITHOUT dirtying', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      documents: [doc('ed_1', { title: 'From server', folderId: 'edf_1' })],
      type: 'upsertServerEditorDocuments',
    })
    expect(next.editorDocuments.ed_1.title).toBe('From server')
    expect(next.editorDocuments.ed_1.contentMarkdown).toBe('') // body loads on open
    expect(next.editorDocuments.ed_1.folderId).toBe('edf_1')
    expect(next.editorDocumentOrder).toContain('ed_1')
    expect(next.dirty).toBe(false)
  })

  it('keeps a newer local document body + metadata over an older server version', () => {
    const local = doc('ed_1', {
      contentMarkdown: 'local body',
      title: 'Local edit',
      updatedAt: '2026-02-01T00:00:00.000Z',
    })
    const next = researchDeskReducer(withDoc(local), {
      documents: [doc('ed_1', { title: 'Stale server', updatedAt: '2026-01-01T00:00:00.000Z' })],
      type: 'upsertServerEditorDocuments',
    })
    expect(next.editorDocuments.ed_1.title).toBe('Local edit')
    expect(next.editorDocuments.ed_1.contentMarkdown).toBe('local body')
  })

  it('takes newer server metadata but KEEPS the local (loaded) body', () => {
    const local = doc('ed_1', {
      contentMarkdown: 'loaded body',
      title: 'old',
      updatedAt: '2026-01-01T00:00:00.000Z',
    })
    const next = researchDeskReducer(withDoc(local), {
      documents: [doc('ed_1', { title: 'newer server', updatedAt: '2026-03-01T00:00:00.000Z' })],
      type: 'upsertServerEditorDocuments',
    })
    expect(next.editorDocuments.ed_1.title).toBe('newer server')
    expect(next.editorDocuments.ed_1.contentMarkdown).toBe('loaded body') // body kept
  })

  it('sets a document body on load-on-open WITHOUT dirtying or bumping updatedAt', () => {
    const local = doc('ed_1', { updatedAt: '2026-01-01T00:00:00.000Z' })
    const next = researchDeskReducer(withDoc(local), {
      contentMarkdown: 'fetched body',
      documentId: 'ed_1',
      type: 'setServerEditorDocumentBody',
    })
    expect(next.editorDocuments.ed_1.contentMarkdown).toBe('fetched body')
    expect(next.editorDocuments.ed_1.updatedAt).toBe('2026-01-01T00:00:00.000Z')
    expect(next.dirty).toBe(false)
  })

  it('replaces a collaboration metadata stub with the exact projection detail', () => {
    const local = doc('ed_1', {
      contentMarkdown: '',
      contentMode: 'collaboration',
      updatedAt: '2026-01-01T00:00:00.000Z',
    })
    const detail = doc('ed_1', {
      collaboration: {
        generation: 1,
        persistedSequence: 8,
        projectionSequence: 8,
        projectionUpdatedAt: '2026-01-02T00:00:00.000Z',
        schemaVersion: 1,
      },
      contentMarkdown: '# Exact projection',
      contentMode: 'collaboration',
      updatedAt: '2026-01-02T00:00:00.000Z',
    })

    const next = researchDeskReducer(withDoc(local), {
      document: detail,
      type: 'setServerEditorDocumentDetail',
    })

    expect(next.editorDocuments.ed_1).toEqual(detail)
    expect(next.dirty).toBe(false)
  })

  it('authoritatively removes deleted comments while preserving explicit local work', () => {
    const base = withDoc(doc('ed_1'))
    const state: ProjectState = {
      ...base,
      editorComments: {
        'local-comment': comment('local-comment', 'ed_1'),
        'server-deleted': comment('server-deleted', 'ed_1'),
        'server-kept': comment('server-kept', 'ed_1'),
      },
    }
    const refreshed = {
      ...comment('server-kept', 'ed_1'),
      commentMarkdown: 'server version',
      updatedAt: '2026-01-02T00:00:00.000Z',
    }

    const next = researchDeskReducer(state, {
      comments: [refreshed],
      documentId: 'ed_1',
      preserveCommentIds: ['local-comment'],
      type: 'reconcileServerEditorComments',
    })

    expect(next.editorComments['local-comment']).toBe(state.editorComments['local-comment'])
    expect(next.editorComments['server-deleted']).toBeUndefined()
    expect(next.editorComments['server-kept']).toEqual(refreshed)
    expect(next.dirty).toBe(false)
  })

  it('merges server folders + comments WITHOUT dirtying', () => {
    const folder: EditorFolderRecord = {
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'edf_1',
      title: 'F',
      updatedAt: '2026-01-01T00:00:00.000Z',
    }
    const afterFolders = researchDeskReducer(createEmptyProjectState(), {
      folders: [folder],
      type: 'upsertServerEditorFolders',
    })
    expect(afterFolders.editorFolders.edf_1.title).toBe('F')
    expect(afterFolders.editorFolderOrder).toContain('edf_1')
    expect(afterFolders.dirty).toBe(false)

    const afterComments = researchDeskReducer(afterFolders, {
      comments: [comment('edc_1', 'ed_1')],
      type: 'upsertServerEditorComments',
    })
    expect(afterComments.editorComments.edc_1.documentId).toBe('ed_1')
    expect(afterComments.dirty).toBe(false)
  })
})

describe('collaboration metadata CAS adoption', () => {
  function collaborationDocument(metadataRevision: number): EditorDocumentRecord {
    return doc('ed_1', {
      access: { mode: 'owner', permission: 'edit' },
      collaboration: {
        generation: 1,
        persistedSequence: 0,
        projectionSequence: 0,
        schemaVersion: 1,
      },
      contentMode: 'collaboration',
      metadataRevision,
    })
  }

  it('authoritatively enters collaboration despite a future local timestamp', () => {
    const legacy = doc('ed_1', {
      access: { mode: 'owner', permission: 'edit' },
      contentMarkdown: '# Writable legacy body',
      metadataRevision: 4,
      updatedAt: '2099-01-01T00:00:00.000Z',
    })

    const activated = researchDeskReducer(withDoc(legacy), {
      collaboration: {
        generation: 3,
        persistedSequence: 0,
        projectionSequence: 0,
        schemaVersion: 1,
      },
      documentId: legacy.id,
      metadataRevision: 5,
      type: 'activateEditorDocumentCollaboration',
    })

    expect(activated.editorDocuments.ed_1).toMatchObject({
      collaboration: { generation: 3 },
      contentMarkdown: '# Writable legacy body',
      contentMode: 'collaboration',
      metadataRevision: 5,
      updatedAt: '2099-01-01T00:00:00.000Z',
    })
    expect(planEditorDocumentAutosave(activated.editorDocuments.ed_1)).toMatchObject({
      kind: 'collaboration_metadata',
      payload: { expected_metadata_revision: 5 },
    })
    expect(activated.dirty).toBe(false)
  })

  it('advances activation, autosave, and anchor PATCH bases monotonically', () => {
    let state = withDoc(collaborationDocument(4))
    state = researchDeskReducer(state, {
      documentId: 'ed_1',
      metadataRevision: 5,
      type: 'adoptEditorDocumentMetadataRevision',
    })
    expect(planEditorDocumentAutosave(state.editorDocuments.ed_1)).toMatchObject({
      payload: { expected_metadata_revision: 5 },
    })

    state = researchDeskReducer(state, {
      documentId: 'ed_1',
      metadataRevision: 6,
      type: 'adoptEditorDocumentMetadataRevision',
    })
    expect(planEditorDocumentAutosave(state.editorDocuments.ed_1)).toMatchObject({
      payload: { expected_metadata_revision: 6 },
    })

    state = researchDeskReducer(state, {
      documentId: 'ed_1',
      metadataRevision: 7,
      type: 'adoptEditorDocumentMetadataRevision',
    })
    expect(state.editorDocuments.ed_1.metadataRevision).toBe(7)
  })

  it('keeps rename and anchor CAS on the newest response despite a late older adoption', () => {
    let state = withDoc(collaborationDocument(7))
    state = researchDeskReducer(state, {
      documentId: 'ed_1',
      title: 'Renamed',
      type: 'renameEditorDocument',
    })
    expect(planEditorDocumentAutosave(state.editorDocuments.ed_1)).toMatchObject({
      payload: { expected_metadata_revision: 7, title: 'Renamed.md' },
    })

    state = researchDeskReducer(state, {
      documentId: 'ed_1',
      metadataRevision: 9,
      type: 'adoptEditorDocumentMetadataRevision',
    })
    state = researchDeskReducer(state, {
      documentId: 'ed_1',
      metadataRevision: 8,
      type: 'adoptEditorDocumentMetadataRevision',
    })

    expect(state.editorDocuments.ed_1.metadataRevision).toBe(9)
    expect(planEditorDocumentAutosave(state.editorDocuments.ed_1)).toMatchObject({
      payload: { expected_metadata_revision: 9, title: 'Renamed.md' },
    })
  })
})

describe('private comment outbox', () => {
  it('tracks explicit local work and ignores acknowledgements for an older mutation', () => {
    const localComment = comment('local-comment', 'ed_1')
    let state = researchDeskReducer(withDoc(doc('ed_1')), {
      comment: localComment,
      type: 'createEditorComment',
    })
    expect(state.editorCommentOutbox?.[localComment.id]).toEqual({
      documentId: 'ed_1',
      operation: 'upsert',
      updatedAt: localComment.updatedAt,
    })

    state = researchDeskReducer(state, {
      acknowledgements: [{
        commentId: localComment.id,
        operation: 'upsert',
        updatedAt: '2025-12-31T23:59:59.000Z',
      }],
      type: 'acknowledgeEditorCommentOutbox',
    })
    expect(state.editorCommentOutbox?.[localComment.id]).toBeDefined()

    state = researchDeskReducer(state, {
      acknowledgements: [{
        commentId: localComment.id,
        operation: 'upsert',
        updatedAt: localComment.updatedAt,
      }],
      type: 'acknowledgeEditorCommentOutbox',
    })
    expect(state.editorCommentOutbox?.[localComment.id]).toBeUndefined()
  })

  it('drops stale cache after reset while preserving only explicit local mutations', () => {
    const staleComment = comment('stale-comment', 'ed_1')
    const localComment = {
      ...comment('local-comment', 'ed_1'),
      updatedAt: '2026-01-02T00:00:00.000Z',
    }
    const state = {
      ...withDoc(doc('ed_1')),
      editorCommentOutbox: {
        [localComment.id]: {
          documentId: 'ed_1',
          operation: 'upsert' as const,
          updatedAt: localComment.updatedAt,
        },
      },
      editorComments: {
        [localComment.id]: localComment,
        [staleComment.id]: staleComment,
      },
    }
    const plan = planEditorCommentReconciliation(
      'ed_1',
      state.editorComments,
      [],
      state.editorCommentOutbox,
    )
    const reconciled = researchDeskReducer(state, {
      comments: plan.serverComments,
      documentId: 'ed_1',
      preserveCommentIds: [...plan.preserveCommentIds],
      type: 'reconcileServerEditorComments',
    })

    expect(reconciled.editorCommentOutbox).toEqual(state.editorCommentOutbox)
    expect(reconciled.editorComments).toEqual({ [localComment.id]: localComment })
  })
})

describe('private collaboration suggestion publication', () => {
  it('stores durable shared command identifiers when acceptance completes', () => {
    const base = withDoc(doc('ed_1'))
    const suggestion: EditorSuggestionRecord = {
      anchor: { from: 0, quoteAfter: '', quoteBefore: '', selectedText: 'x', to: 1 },
      blockId: 'block-1',
      createdAt: '2026-01-01T00:00:00.000Z',
      documentId: 'ed_1',
      groupId: 'group-1',
      id: 'suggestion-1',
      originalText: 'x',
      origin: { kind: 'global_run' },
      proposedText: 'y',
      status: 'pending',
      updatedAt: '2026-01-01T00:00:00.000Z',
    }
    const state = {
      ...base,
      editorSuggestions: { [suggestion.id]: suggestion },
    }
    const collaborationPublication = {
      commandId: 'command-1',
      patchId: 'patch-1',
      sequence: 9,
      suggestionIds: ['shared-suggestion-1'],
    }

    const next = researchDeskReducer(state, {
      collaborationPublication,
      suggestionId: suggestion.id,
      type: 'acceptEditorSuggestion',
    })

    expect(next.editorSuggestions[suggestion.id]).toMatchObject({
      collaborationPublication,
      status: 'accepted',
    })
  })
})
