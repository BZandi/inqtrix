import { MAX_DOC_CHARS_SOFT } from './budget'

/** Visible warning when a document's extracted text is capped at ingest. */
export const truncationWarning = 'Textinhalt gekürzt (Dokument zu groß für das Modell-Kontextfenster).'

/** Visible warning when a supported file yields no extractable text. */
export const emptyTextWarning = 'Kein Text extrahierbar (evtl. gescanntes Dokument ohne Textebene).'

/** Cap extracted text at the per-document soft limit, flagging truncation. */
export function clampText(text: string): { extractedText: string; textTruncated: boolean } {
  if (text.length <= MAX_DOC_CHARS_SOFT) return { extractedText: text, textTruncated: false }
  return { extractedText: text.slice(0, MAX_DOC_CHARS_SOFT), textTruncated: true }
}
