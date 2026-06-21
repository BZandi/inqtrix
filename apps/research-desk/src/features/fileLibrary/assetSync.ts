/**
 * Pure conversion + push helpers between the local ProjectState file-library
 * records and the server wire shape (M6c project-persistence tier).
 *
 * The file-library counterpart of features/editor/editorSync.ts: the API
 * client speaks the verbatim server shape (snake_case, unix-seconds floats);
 * this module maps it to/from the ISO-timestamped FileLibrarySectionRecord /
 * FileGroupRecord / FileAssetRecord the reducer uses. An asset's heavy
 * extractedText round-trips as ``extracted_text`` — empty on list rows,
 * present on a single-asset GET (the editor's content_markdown pattern).
 *
 * Pure (no React) so the conversion is unit-testable; the entity fingerprints
 * for the autosave diff are plain ``updatedAt`` strings (every mutation that
 * needs syncing bumps the entity's ``updatedAt``).
 */

import type { ClientOptions } from '@/api/inqtrixClient'
import {
  saveAsset,
  saveAssetGroup,
  saveAssetSection,
  type ServerAsset,
  type ServerAssetGroup,
  type ServerAssetSection,
} from '@/api/inqtrixClient'
import type {
  FileAssetOrigin,
  FileAssetRecord,
  FileGroupRecord,
  FileLibrarySectionRecord,
  FileParseStatus,
  FileSectionKind,
} from '@/features/project/types'
import { isoFromUnixSeconds, unixSecondsFromIso } from '@/lib/time'

const VALID_KINDS: ReadonlySet<string> = new Set(['temporary', 'custom'])
const VALID_ORIGINS: ReadonlySet<string> = new Set(['chat', 'editor', 'library'])
const VALID_PARSE: ReadonlySet<string> = new Set([
  'parsed',
  'partial',
  'unsupported',
  'error',
])

function normalizeKind(kind: string): FileSectionKind {
  return VALID_KINDS.has(kind) ? (kind as FileSectionKind) : 'custom'
}

function normalizeOrigin(origin: string): FileAssetOrigin {
  return VALID_ORIGINS.has(origin) ? (origin as FileAssetOrigin) : 'library'
}

function normalizeParse(status: string): FileParseStatus {
  return VALID_PARSE.has(status) ? (status as FileParseStatus) : 'parsed'
}

// -- server -> record ------------------------------------------------------ #

export function sectionRecordFromServer(
  section: ServerAssetSection,
): FileLibrarySectionRecord {
  return {
    createdAt: isoFromUnixSeconds(section.created_at),
    id: section.id,
    kind: normalizeKind(section.kind),
    title: section.title,
    updatedAt: isoFromUnixSeconds(section.updated_at),
  }
}

export function groupRecordFromServer(group: ServerAssetGroup): FileGroupRecord {
  return {
    createdAt: isoFromUnixSeconds(group.created_at),
    id: group.id,
    sectionId: group.section_id,
    title: group.title,
    updatedAt: isoFromUnixSeconds(group.updated_at),
  }
}

/** One server asset -> its local record. ``extracted_text`` is empty on a
 * list (metadata) row and the real body on a single-asset GET. */
export function assetRecordFromServer(asset: ServerAsset): FileAssetRecord {
  return {
    createdAt: isoFromUnixSeconds(asset.created_at),
    extractedText: asset.extracted_text ?? '',
    fileName: asset.file_name,
    groupId: asset.group_id,
    id: asset.id,
    label: asset.label,
    mimeType: asset.mime_type,
    origin: normalizeOrigin(asset.origin),
    pageCount: asset.page_count,
    parseStatus: normalizeParse(asset.parse_status),
    parseWarning: asset.parse_warning,
    sectionId: asset.section_id,
    sizeBytes: asset.size_bytes,
    textTruncated: asset.text_truncated,
    title: asset.title,
    updatedAt: isoFromUnixSeconds(asset.updated_at),
    serverFileId: asset.server_file_id,
    parserId: asset.parser_id ?? null,
  }
}

// -- record -> server payload ---------------------------------------------- #

export function serverSectionPayload(record: FileLibrarySectionRecord): {
  created_at: number
  kind: string
  title: string
  updated_at: number
} {
  return {
    created_at: unixSecondsFromIso(record.createdAt),
    kind: record.kind,
    title: record.title,
    updated_at: unixSecondsFromIso(record.updatedAt),
  }
}

export function serverGroupPayload(record: FileGroupRecord): {
  created_at: number
  section_id: string
  title: string
  updated_at: number
} {
  return {
    created_at: unixSecondsFromIso(record.createdAt),
    section_id: record.sectionId,
    title: record.title,
    updated_at: unixSecondsFromIso(record.updatedAt),
  }
}

export function serverAssetPayload(record: FileAssetRecord): {
  created_at: number
  extracted_text: string
  file_name: string
  group_id: string | null
  label: string
  mime_type: string
  origin: string
  page_count: number | null
  parse_status: string
  parse_warning: string | null
  section_id: string
  server_file_id: string | null
  parser_id: string | null
  size_bytes: number
  text_truncated: boolean
  title: string
  updated_at: number
} {
  return {
    created_at: unixSecondsFromIso(record.createdAt),
    extracted_text: record.extractedText,
    file_name: record.fileName,
    group_id: record.groupId,
    label: record.label,
    mime_type: record.mimeType,
    origin: record.origin,
    page_count: record.pageCount,
    parse_status: record.parseStatus,
    parse_warning: record.parseWarning,
    section_id: record.sectionId,
    server_file_id: record.serverFileId ?? null,
    parser_id: record.parserId ?? null,
    size_bytes: record.sizeBytes,
    text_truncated: record.textTruncated,
    title: record.title,
    updated_at: unixSecondsFromIso(record.updatedAt),
  }
}

// -- whole-project push (the explicit import) ------------------------------ #

/** Push ALL of a local project's file-library entities to the server (the
 * one-time import). Sections first, then groups (they reference a section),
 * then assets (with bodies — a local project has them loaded). Idempotent
 * server upserts make a re-run safe. */
export async function pushAllAssetEntities(
  args: {
    sections: Record<string, FileLibrarySectionRecord>
    groups: Record<string, FileGroupRecord>
    assets: Record<string, FileAssetRecord>
  },
  options: ClientOptions,
): Promise<void> {
  for (const section of Object.values(args.sections)) {
    await saveAssetSection(section.id, serverSectionPayload(section), options)
  }
  for (const group of Object.values(args.groups)) {
    await saveAssetGroup(group.id, serverGroupPayload(group), options)
  }
  for (const asset of Object.values(args.assets)) {
    await saveAsset(asset.id, serverAssetPayload(asset), options)
  }
}
