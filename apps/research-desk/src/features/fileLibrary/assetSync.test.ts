import { describe, expect, it } from 'vitest'

import type {
  ServerAsset,
  ServerAssetGroup,
  ServerAssetSection,
} from '@/api/inqtrixClient'
import {
  assetRecordFromServer,
  groupRecordFromServer,
  sectionRecordFromServer,
  serverAssetPayload,
  serverGroupPayload,
  serverSectionPayload,
} from './assetSync'

describe('assetSync converters', () => {
  it('round-trips a section and group', () => {
    const section: ServerAssetSection = {
      id: 'file-section-1', kind: 'custom', title: 'My collection',
      created_at: 100, updated_at: 200,
    }
    const sectionRecord = sectionRecordFromServer(section)
    expect(sectionRecord.kind).toBe('custom')
    const sectionPayload = serverSectionPayload(sectionRecord)
    expect(sectionPayload).toMatchObject({ kind: 'custom', title: 'My collection', created_at: 100, updated_at: 200 })

    const group: ServerAssetGroup = {
      id: 'file-group-1', section_id: 'file-section-1', title: 'G',
      created_at: 5, updated_at: 6,
    }
    const groupRecord = groupRecordFromServer(group)
    expect(groupRecord.sectionId).toBe('file-section-1')
    expect(serverGroupPayload(groupRecord)).toMatchObject({ section_id: 'file-section-1', created_at: 5, updated_at: 6 })
  })

  it('round-trips an asset WITH its extracted text', () => {
    const server: ServerAsset = {
      id: 'file-asset-1', section_id: 'file-section-1', group_id: 'file-group-1',
      title: 'Doc', label: 'Doc.pdf', file_name: 'Doc.pdf', mime_type: 'application/pdf',
      origin: 'library', page_count: 12, parse_status: 'parsed', parse_warning: null,
      text_truncated: true, size_bytes: 4096, server_file_id: 'fl_9',
      parser_id: 'markitdown',
      extracted_text: 'the heavy body', created_at: 1_699_000_000, updated_at: 1_700_000_000,
    }
    const record = assetRecordFromServer(server)
    expect(record.extractedText).toBe('the heavy body')
    expect(record.groupId).toBe('file-group-1')
    expect(record.serverFileId).toBe('fl_9')
    expect(record.parserId).toBe('markitdown')
    expect(record.textTruncated).toBe(true)

    const payload = serverAssetPayload(record)
    expect(payload.extracted_text).toBe('the heavy body')
    expect(payload.section_id).toBe('file-section-1')
    expect(payload.group_id).toBe('file-group-1')
    expect(payload.server_file_id).toBe('fl_9')
    expect(payload.parser_id).toBe('markitdown')
    expect(payload.created_at).toBe(server.created_at)
    expect(payload.updated_at).toBe(server.updated_at)
  })

  it('treats a list row (no extracted_text) as an empty body', () => {
    const record = assetRecordFromServer({
      id: 'file-asset-2', section_id: 'file-section-1', group_id: null,
      title: 'Listed', label: 'l', file_name: 'l.pdf', mime_type: 'application/pdf',
      origin: 'library', page_count: null, parse_status: 'parsed', parse_warning: null,
      text_truncated: false, size_bytes: 0, server_file_id: null,
      created_at: 1, updated_at: 1,
    })
    expect(record.extractedText).toBe('')
    expect(record.groupId).toBe(null)
    expect(record.serverFileId).toBe(null)
    // A list-row record carries a null body -> the payload must not send "" as
    // a real body if pushed without loading (the hook guards this; here we
    // only assert the converter is faithful).
    expect(serverAssetPayload(record).extracted_text).toBe('')
  })

  it('normalizes unknown enum values to safe defaults', () => {
    const record = assetRecordFromServer({
      id: 'file-asset-3', section_id: 's', group_id: null, title: 'x', label: 'x',
      file_name: 'x', mime_type: 'x', origin: 'bogus', page_count: null,
      parse_status: 'weird', parse_warning: null, text_truncated: false,
      size_bytes: 0, server_file_id: null, created_at: 1, updated_at: 1,
    })
    expect(record.origin).toBe('library')
    expect(record.parseStatus).toBe('parsed')
    const section = sectionRecordFromServer({
      id: 's', kind: 'nope', title: 't', created_at: 1, updated_at: 1,
    })
    expect(section.kind).toBe('custom')
  })
})
