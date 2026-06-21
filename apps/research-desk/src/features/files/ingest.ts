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
 * CLIENT-SIDE for instant feedback and shaped into a `FileAssetRecord`; the
 * caller dispatches the records and decides what to do with the returned ids.
 * Labels are de-duplicated against `existingLabels` so mention tokens stay
 * unique across the library.
 *
 * The higher-fidelity server parser (MarkItDown) is deliberately NOT run here:
 * it is slow (S3 round-trip + parse) and would block the file from appearing.
 * Instead the ORIGINAL bytes are uploaded for later use, and the server parse
 * happens lazily at vector-index time, back-filling the text + provenance
 * (see knowledgeSync.reindexVectorIndexOnServer).
 */
export type ServerFileUpload = (file: File) => Promise<string>

export async function ingestFiles(
  files: readonly File[],
  origin: IngestOrigin,
  parser: FileParser = createDefaultFileParser(),
  existingLabels: readonly string[] = [],
  serverUpload?: ServerFileUpload,
): Promise<FileAssetRecord[]> {
  const used = new Set(existingLabels)
  const sectionId = origin.sectionId ?? FILE_SECTION_TEMP_ID
  const groupId = origin.groupId ?? null
  const records: FileAssetRecord[] = []

  for (const file of files) {
    const parsed = await parser.parse(file)
    // Progressive enhancement: when the backend advertises the files
    // feature, the ORIGINAL bytes go to the server too (enables the
    // higher-fidelity server parse + ingestion at vector-index time).
    // A failed upload never blocks local use — the asset stays local
    // with a visible warning, parsed by the client.
    let serverFileId: string | null = null
    let serverWarning: string | null = null
    if (serverUpload) {
      try {
        serverFileId = await serverUpload(file)
      } catch (error) {
        serverWarning = `Server-Upload fehlgeschlagen (${
          error instanceof Error ? error.message : 'unbekannt'
        }) — Datei bleibt lokal.`
      }
    }
    const now = new Date().toISOString()
    records.push({
      createdAt: now,
      extractedText: parsed.extractedText,
      serverFileId,
      // Uploaded here = browser-parsed. The provenance upgrades to
      // 'markitdown' (with the text) when the file is indexed server-side.
      parserId: 'client',
      fileName: file.name,
      groupId,
      id: createFileId(),
      label: uniqueLabel(slugifyFileName(file.name), used),
      mimeType: file.type || 'application/octet-stream',
      origin: origin.kind,
      pageCount: parsed.pageCount,
      parseStatus: parsed.parseStatus,
      parseWarning: serverWarning
        ? parsed.parseWarning
          ? `${parsed.parseWarning} ${serverWarning}`
          : serverWarning
        : parsed.parseWarning,
      sectionId,
      sizeBytes: file.size,
      textTruncated: parsed.textTruncated,
      title: file.name,
      updatedAt: now,
    })
  }

  return records
}

/** Side-effect callbacks for {@link scheduleServerParse}, kept abstract so the
 * helper imports neither the API client nor the reducer (pure + testable). */
export type ServerParseHandlers = {
  /** Fetch the server (MarkItDown) text for an uploaded file id. */
  fetchText: (fileId: string) => Promise<string>
  /** A background parse started for this asset (drives the "Parsing…" badge). */
  onPending: (assetId: string) => void
  /** The server text arrived — upgrade the asset's text + provenance. */
  onParsed: (assetId: string, text: string) => void
  /** The server parse failed/declined — clear the pending marker. */
  onFailed: (assetId: string) => void
}

/**
 * Fire-and-forget background server parse for freshly-uploaded assets. NON-
 * BLOCKING by design: the assets are already dispatched and visible, so this
 * only upgrades the instant in-browser parse to the stronger, browser-
 * independent MarkItDown text once it lands (the in-browser parse may have
 * failed, e.g. pdf.js on Safari). Skips assets with no server file or already
 * server-parsed, so it is safe to call on every ingest result.
 */
export function scheduleServerParse(
  assets: readonly FileAssetRecord[],
  handlers: ServerParseHandlers,
): void {
  for (const asset of assets) {
    if (!asset.serverFileId || asset.parserId === 'markitdown') continue
    handlers.onPending(asset.id)
    handlers
      .fetchText(asset.serverFileId)
      .then((text) => handlers.onParsed(asset.id, text))
      .catch(() => handlers.onFailed(asset.id))
  }
}
