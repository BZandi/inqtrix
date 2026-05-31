import type { FileAssetOrigin, FileAssetRecord } from '@/features/project/types'
import { createDefaultFileParser, type FileParser } from './parsing'
import { FILE_SECTION_TEMP_ID } from './sections'

export type IngestOrigin = {
  groupId?: string | null
  kind: FileAssetOrigin
  sectionId?: string
}

function createFileId(): string {
  return `file-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function slugifyFileName(fileName: string): string {
  const base = fileName.replace(/\.[^.]+$/, '')
  const slug = base
    .normalize('NFKD')
    .toLowerCase()
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)
  return slug || 'datei'
}

function uniqueLabel(base: string, used: Set<string>): string {
  if (!used.has(base)) {
    used.add(base)
    return base
  }
  let suffix = 2
  while (used.has(`${base}-${suffix}`)) suffix += 1
  const label = `${base}-${suffix}`
  used.add(label)
  return label
}

/**
 * The single ingest pipeline used by every upload entry point (chat paperclip,
 * chat drag-and-drop, editor drag-and-drop, library upload). Each file is parsed
 * and shaped into a `FileAssetRecord`; the caller dispatches the records and
 * decides what to do with the returned ids. Labels are de-duplicated against
 * `existingLabels` so mention tokens stay unique across the library.
 */
export async function ingestFiles(
  files: readonly File[],
  origin: IngestOrigin,
  parser: FileParser = createDefaultFileParser(),
  existingLabels: readonly string[] = [],
): Promise<FileAssetRecord[]> {
  const used = new Set(existingLabels)
  const sectionId = origin.sectionId ?? FILE_SECTION_TEMP_ID
  const groupId = origin.groupId ?? null
  const records: FileAssetRecord[] = []

  for (const file of files) {
    const parsed = await parser.parse(file)
    const now = new Date().toISOString()
    records.push({
      createdAt: now,
      extractedText: parsed.extractedText,
      fileName: file.name,
      groupId,
      id: createFileId(),
      label: uniqueLabel(slugifyFileName(file.name), used),
      mimeType: file.type || 'application/octet-stream',
      origin: origin.kind,
      pageCount: parsed.pageCount,
      parseStatus: parsed.parseStatus,
      parseWarning: parsed.parseWarning,
      sectionId,
      sizeBytes: file.size,
      textTruncated: parsed.textTruncated,
      title: file.name,
      updatedAt: now,
    })
  }

  return records
}
