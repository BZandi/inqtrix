import { describe, expect, it } from 'vitest'

import type {
  ServerEditorComment,
  ServerEditorDocument,
  ServerEditorFolder,
} from '@/api/inqtrixClient'
import {
  commentRecordFromServer,
  documentRecordFromServer,
  folderRecordFromServer,
  serverCommentPayload,
  serverDocumentPayload,
  serverFolderPayload,
} from './editorSync'

describe('editorSync converters', () => {
  it('round-trips a document with body, diff anchor, and source run', () => {
    const server: ServerEditorDocument = {
      id: 'ed_1',
      title: 'My doc',
      content_markdown: '# Heavy body',
      folder_id: 'edf_1',
      source: 'imported-research-report',
      source_run_id: 'run_1',
      revision: 7,
      diff_anchor_markdown: '# old',
      diff_anchor_updated_at: 1_700_000_000,
      created_at: 1_699_000_000,
      updated_at: 1_700_000_500,
    }
    const record = documentRecordFromServer(server)
    expect(record.contentMarkdown).toBe('# Heavy body')
    expect(record.folderId).toBe('edf_1')
    expect(record.source).toBe('imported-research-report')
    expect(record.sourceRunId).toBe('run_1')
    expect(record.revision).toBe(7)
    expect(record.diffAnchorMarkdown).toBe('# old')

    const payload = serverDocumentPayload(record)
    expect(payload.content_markdown).toBe('# Heavy body')
    expect(payload.folder_id).toBe('edf_1')
    expect(payload.source).toBe('imported-research-report')
    expect(payload.source_run_id).toBe('run_1')
    expect(payload.revision).toBe(7)
    expect(payload.created_at).toBe(server.created_at)
    expect(payload.updated_at).toBe(server.updated_at)
    expect(payload.diff_anchor_updated_at).toBe(server.diff_anchor_updated_at)
  })

  it('treats a list row (no content_markdown) as an empty body', () => {
    const record = documentRecordFromServer({
      id: 'ed_2',
      title: 'Listed',
      folder_id: null,
      source: 'blank',
      source_run_id: null,
      revision: 1,
      diff_anchor_markdown: null,
      diff_anchor_updated_at: null,
      created_at: 1,
      updated_at: 1,
    })
    expect(record.contentMarkdown).toBe('')
    expect('sourceRunId' in record).toBe(false)
    expect('diffAnchorMarkdown' in record).toBe(false)
    expect(serverDocumentPayload(record).source_run_id).toBe(null)
  })

  it('normalizes an unknown document source', () => {
    const record = documentRecordFromServer({
      id: 'ed_3', title: 'X', folder_id: null, source: 'nope',
      source_run_id: null, revision: 1, diff_anchor_markdown: null,
      diff_anchor_updated_at: null, created_at: 1, updated_at: 1,
    })
    expect(record.source).toBe('blank')
  })

  it('round-trips a comment with its verbatim anchor', () => {
    const server: ServerEditorComment = {
      id: 'edc_1',
      document_id: 'ed_1',
      comment_markdown: 'check this',
      anchor: { from: 3, to: 9, selectedText: 'hello', quoteBefore: 'a', quoteAfter: 'b' },
      kind: 'evidence_review',
      status: 'open',
      evidence_preset: 'fact_check',
      created_at: 10,
      updated_at: 20,
    }
    const record = commentRecordFromServer(server)
    expect(record.anchor).toEqual(server.anchor)
    expect(record.kind).toBe('evidence_review')
    expect(record.evidencePreset).toBe('fact_check')

    const payload = serverCommentPayload(record)
    expect(payload.anchor).toEqual(server.anchor)
    expect(payload.evidence_preset).toBe('fact_check')
    expect(payload.created_at).toBe(10)
    expect(payload.updated_at).toBe(20)
  })

  it('omits an absent evidence preset on a comment', () => {
    const record = commentRecordFromServer({
      id: 'edc_2', document_id: 'ed_1', comment_markdown: 'x',
      anchor: { from: 0, to: 1, selectedText: 'h' }, kind: 'collect',
      status: 'open', evidence_preset: null, created_at: 1, updated_at: 1,
    })
    expect('evidencePreset' in record).toBe(false)
    expect(serverCommentPayload(record).evidence_preset).toBe(null)
  })

  it('round-trips a folder', () => {
    const server: ServerEditorFolder = {
      id: 'edf_1', title: 'F', created_at: 100, updated_at: 200,
    }
    const record = folderRecordFromServer(server)
    const payload = serverFolderPayload(record)
    expect(payload.created_at).toBe(100)
    expect(payload.updated_at).toBe(200)
    expect(payload.title).toBe('F')
  })
})
