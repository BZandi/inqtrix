import { describe, expect, it } from 'vitest'

import { createEmptyProjectState } from '@/features/project/seedProject'
import type {
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorFolderRecord,
  ProjectState,
} from '@/features/project/types'
import { researchDeskReducer } from '@/features/researchDesk/state'

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
