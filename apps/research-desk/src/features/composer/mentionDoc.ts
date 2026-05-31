import type { JSONContent } from '@tiptap/core'
import type { ChatContextReferenceRecord } from '@/features/project/types'
import { MENTION_PILL_NAME, type MentionPillKind } from './MentionPill'

/** Mention prefix per positional kind (rules are global and never pills). */
export const PILL_KIND_PREFIX: Record<MentionPillKind, string> = {
  'file-asset': '@files:',
  'file-group': '@filegroups:',
  'research-report': '@research:',
}

export function pillKindToRef(kind: MentionPillKind, id: string): ChatContextReferenceRecord {
  switch (kind) {
    case 'file-asset':
      return { fileId: id, kind: 'file-asset' }
    case 'file-group':
      return { groupId: id, kind: 'file-group' }
    case 'research-report':
      return { kind: 'research-report', runId: id }
  }
}

/** Whether a reference kind is positional (rendered as a `[N]` pill). */
export function isPillKind(kind: ChatContextReferenceRecord['kind']): kind is MentionPillKind {
  return kind === 'file-asset' || kind === 'file-group' || kind === 'research-report'
}

/**
 * Serialize a TipTap document (JSON) to plain text. Each mention pill is replaced
 * via `pillToText`, which receives the pill's kind/id/label and its 0-based
 * reading-order index. Paragraphs are joined with newlines.
 */
export function serializeMentionDoc(
  doc: JSONContent,
  pillToText: (kind: MentionPillKind, id: string, label: string, pillIndex: number) => string,
): string {
  let pillIndex = 0
  const paragraphs = (doc.content ?? []).map((block) => (
    (block.content ?? []).map((node) => {
      if (node.type === MENTION_PILL_NAME) {
        const kind = (node.attrs?.refKind ?? 'file-asset') as MentionPillKind
        const text = pillToText(kind, String(node.attrs?.refId ?? ''), String(node.attrs?.refLabel ?? ''), pillIndex)
        pillIndex += 1
        return text
      }
      if (node.type === 'hardBreak') return '\n'
      return node.text ?? ''
    }).join('')
  ))
  return paragraphs.join('\n').trim()
}

/** The instruction text sent to the model: pills become `[N]` (1-based reading order). */
export function instructionTextFromDoc(doc: JSONContent): string {
  return serializeMentionDoc(doc, (_kind, _id, _label, pillIndex) => `[${pillIndex + 1}]`)
}

/** A round-trippable text with pills as `@kind:label` (used for text improvement). */
export function mentionTextFromDoc(doc: JSONContent): string {
  return serializeMentionDoc(doc, (kind, _id, label) => `${PILL_KIND_PREFIX[kind]}${label}`)
}

export type LabelResolver = (kind: MentionPillKind, label: string) => { id: string; label: string } | null

const TOKEN_KIND: Record<string, MentionPillKind> = {
  files: 'file-asset',
  filegroups: 'file-group',
  research: 'research-report',
}

/**
 * Parse plain text with `@files:`/`@filegroups:`/`@research:` tokens into a
 * TipTap document, turning every resolvable token into a pill. Unknown tokens
 * stay as plain text. Used for the initial draft, paste, and the text-improve
 * round-trip.
 */
export function mentionDocFromText(text: string, resolve: LabelResolver): JSONContent {
  const tokenRe = /@(files|filegroups|research):([a-z0-9-]+)/gi
  const content = text.split('\n').map((line) => {
    const inline: JSONContent[] = []
    let lastIndex = 0
    tokenRe.lastIndex = 0
    let match = tokenRe.exec(line)
    while (match) {
      const kind = TOKEN_KIND[match[1].toLowerCase()]
      const resolved = kind ? resolve(kind, match[2].toLowerCase()) : null
      if (resolved) {
        if (match.index > lastIndex) inline.push({ type: 'text', text: line.slice(lastIndex, match.index) })
        inline.push({ attrs: { refId: resolved.id, refKind: kind, refLabel: resolved.label }, type: MENTION_PILL_NAME })
        lastIndex = match.index + match[0].length
      }
      match = tokenRe.exec(line)
    }
    if (lastIndex < line.length) inline.push({ text: line.slice(lastIndex), type: 'text' })
    return inline.length > 0 ? { content: inline, type: 'paragraph' } : { type: 'paragraph' }
  })
  return { content, type: 'doc' }
}

/** Positional pill references in reading order. */
export function pillRefsFromDoc(doc: JSONContent): ChatContextReferenceRecord[] {
  const refs: ChatContextReferenceRecord[] = []
  for (const block of doc.content ?? []) {
    for (const node of block.content ?? []) {
      if (node.type === MENTION_PILL_NAME) {
        refs.push(pillKindToRef((node.attrs?.refKind ?? 'file-asset') as MentionPillKind, String(node.attrs?.refId ?? '')))
      }
    }
  }
  return refs
}
