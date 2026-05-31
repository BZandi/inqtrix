import type { JSONContent } from '@tiptap/core'
import type { Editor } from '@tiptap/react'
import type { EditorCommentAnchorRecord, EditorCommentKind, EditorCommentThreadRecord } from '@/features/project/types'

export type AnchorLocator = {
  hint: number
  quoteAfter?: string
  quoteBefore?: string
  text: string
}

export type EditorTextRange = {
  from: number
  to: number
}

export function createCommentFromSelection(
  editor: Editor,
  documentId: string,
  commentMarkdown: string,
  kind: EditorCommentKind = 'collect',
): EditorCommentThreadRecord | null {
  const { from, to } = editor.state.selection
  if (from >= to) return null
  const anchor = materializeAnchorForRange(editor, { from, to })
  const now = new Date().toISOString()
  return {
    anchor,
    commentMarkdown,
    createdAt: now,
    documentId,
    id: createLocalId('editor-comment'),
    kind,
    status: 'open',
    updatedAt: now,
  }
}

export function clampAnchor(anchor: EditorCommentAnchorRecord, editor: Editor): EditorTextRange {
  const docSize = editor.state.doc.content.size
  const from = Math.max(1, Math.min(anchor.from, docSize))
  const to = Math.max(from, Math.min(anchor.to, docSize))
  return { from, to }
}

export function materializeAnchorForRange(
  editor: Editor,
  range: EditorTextRange,
  previous?: EditorCommentAnchorRecord,
): EditorCommentAnchorRecord {
  const docSize = editor.state.doc.content.size
  const from = Math.max(1, Math.min(range.from, docSize))
  const to = Math.max(from, Math.min(range.to, docSize))
  return {
    ...previous,
    from,
    quoteAfter: editor.state.doc.textBetween(to, Math.min(docSize, to + 80), ' '),
    quoteBefore: editor.state.doc.textBetween(Math.max(0, from - 80), from, ' '),
    selectedMarkdown: markdownForRange(editor, from, to),
    selectedText: editor.state.doc.textBetween(from, to, ' '),
    to,
  }
}

export function resolveMaterializedAnchor(
  editor: Editor,
  anchor: EditorCommentAnchorRecord,
): { anchor: EditorCommentAnchorRecord; range: EditorTextRange } | null {
  const range = resolveAnchorRange(editor, {
    hint: clampAnchor(anchor, editor).from,
    quoteAfter: anchor.quoteAfter,
    quoteBefore: anchor.quoteBefore,
    text: anchor.selectedText,
  })
  if (!range) return null
  return {
    anchor: materializeAnchorForRange(editor, range, anchor),
    range,
  }
}

export function materializeCommentThread(
  editor: Editor,
  comment: EditorCommentThreadRecord,
): EditorCommentThreadRecord | null {
  const resolved = resolveMaterializedAnchor(editor, comment.anchor)
  if (!resolved) return null
  return { ...comment, anchor: resolved.anchor }
}

/**
 * Resolve a content locator (anchored text plus surrounding context) to a live
 * document range. ProseMirror owns positions; suggestions and the model only
 * speak content, so this helper is the single boundary that turns "what text"
 * into "where in the document".
 */
export function resolveAnchorRange(editor: Editor, locator: AnchorLocator): EditorTextRange | null {
  const index = buildDocumentTextIndex(editor)
  for (const needle of searchNeedlesForText(locator.text)) {
    const candidates: EditorTextRange[] = []
    let matchIndex = index.text.indexOf(needle)
    while (matchIndex >= 0) {
      const range = rangeFromIndexedMatch(index.positions, matchIndex, needle.length)
      if (range) candidates.push(range)
      matchIndex = index.text.indexOf(needle, matchIndex + 1)
    }
    if (candidates.length <= 1) {
      if (candidates[0]) return candidates[0]
      continue
    }
    const picked = pickAnchorCandidate(editor, candidates, locator)
    if (picked) return picked
  }
  return null
}

export function blockWidgetPositionForRange(editor: Editor, range: EditorTextRange): number {
  let widgetAt = range.to
  editor.state.doc.descendants((node, pos) => {
    if (!node.isTextblock) return true
    const nodeStart = pos
    const nodeEnd = pos + node.nodeSize
    if (range.to >= nodeStart && range.to <= nodeEnd) {
      widgetAt = Math.min(editor.state.doc.content.size, nodeEnd)
      return false
    }
    return true
  })
  return widgetAt
}

export function blockInsertionPositionForRange(
  editor: Editor,
  range: EditorTextRange,
  side: 'before' | 'after',
): number {
  let insertionAt = side === 'before' ? range.from : range.to
  editor.state.doc.descendants((node, pos) => {
    if (!node.isTextblock) return true
    const nodeStart = pos
    const nodeEnd = pos + node.nodeSize
    if (range.from >= nodeStart && range.to <= nodeEnd) {
      insertionAt = side === 'before'
        ? nodeStart
        : Math.min(editor.state.doc.content.size, nodeEnd)
      return false
    }
    return true
  })
  return Math.max(0, Math.min(insertionAt, editor.state.doc.content.size))
}

export function markdownForRange(editor: Editor, from: number, to: number): string {
  const slice = editor.state.doc.slice(from, to)
  const json = slice.content.toJSON()
  const content = Array.isArray(json) ? json : [json]
  if (content.length === 0 || !editor.markdown) return editor.state.doc.textBetween(from, to, ' ')
  try {
    return editor.markdown.serialize({ type: 'doc', content } as JSONContent).trim()
  } catch {
    return editor.state.doc.textBetween(from, to, ' ')
  }
}

export function shouldParsePastedMarkdown(value: string): boolean {
  const trimmed = value.trim()
  if (!trimmed) return false
  return /(^|\n)(#{1,6}\s|[-*+]\s+|\d+\.\s+|>\s|```|\|.+\|)/.test(trimmed)
    || /\[[^\]]+\]\([^)]+\)/.test(trimmed)
    || /(^|[^*])\*\*[^*\n]+\*\*([^*]|$)/.test(trimmed)
    || /(^|[^_])__[^_\n]+__([^_]|$)/.test(trimmed)
    || /(^|[^`])`[^`\n]+`([^`]|$)/.test(trimmed)
    || /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([^)]+\\\)|(^|[^$])\$[^$\n]+\$([^$]|$)/.test(trimmed)
}

type DocumentTextIndex = {
  positions: Array<number | null>
  text: string
}

function buildDocumentTextIndex(editor: Editor): DocumentTextIndex {
  const positions: Array<number | null> = []
  let text = ''
  const appendChar = (rawChar: string, pos: number | null) => {
    const char = /\s/.test(rawChar) ? ' ' : rawChar
    if (char === ' ' && (!text || text.endsWith(' '))) return
    text += char
    positions.push(pos)
  }

  editor.state.doc.descendants((node, blockPos) => {
    if (!node.isTextblock) return true
    let blockHasText = false
    node.forEach((child, childOffset) => {
      if (!child.isText || !child.text) return
      if (!blockHasText && text) appendChar(' ', null)
      blockHasText = true
      const base = blockPos + 1 + childOffset
      for (let offset = 0; offset < child.text.length; offset += 1) {
        appendChar(child.text[offset], base + offset)
      }
    })
    return false
  })

  return { positions, text: text.trim() }
}

function rangeFromIndexedMatch(
  positions: Array<number | null>,
  start: number,
  length: number,
): EditorTextRange | null {
  const matchPositions = positions.slice(start, start + length)
  const from = matchPositions.find((pos): pos is number => typeof pos === 'number')
  const last = [...matchPositions].reverse().find((pos): pos is number => typeof pos === 'number')
  return from != null && last != null ? { from, to: last + 1 } : null
}

function normalizeSearchText(value: string): string {
  return value.replace(/\s+/g, ' ').trim()
}

export function markdownToPlainTextForEditor(value: string): string {
  return normalizeSearchText(value
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/```[^\n]*\n?|```/g, ' '))
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\d+[.)]\s+/gm, '')
    .replace(/^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/gm, ' ')
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/([*_~])/g, '')
    .replace(/\|/g, ' '))
}

function searchNeedlesForText(value: string): string[] {
  const candidates = [normalizeSearchText(value), markdownToPlainTextForEditor(value)]
  return [...new Set(candidates)]
    .filter(Boolean)
    .sort((a, b) => b.length - a.length)
}

function pickAnchorCandidate(
  editor: Editor,
  candidates: EditorTextRange[],
  locator: AnchorLocator,
): EditorTextRange {
  const { doc } = editor.state
  const before = markdownToPlainTextForEditor(locator.quoteBefore ?? '')
  const after = markdownToPlainTextForEditor(locator.quoteAfter ?? '')
  const scored = candidates.map((candidate) => {
    let score = 0
    if (before) {
      const context = normalizeSearchText(doc.textBetween(Math.max(0, candidate.from - before.length - 20), candidate.from, ' '))
      if (context.endsWith(before)) score += 2
      else if (before.length >= 12 && context.endsWith(before.slice(-12))) score += 1
    }
    if (after) {
      const context = normalizeSearchText(doc.textBetween(candidate.to, Math.min(doc.content.size, candidate.to + after.length + 20), ' '))
      if (context.startsWith(after)) score += 2
      else if (after.length >= 12 && context.startsWith(after.slice(0, 12))) score += 1
    }
    return { candidate, distance: Math.abs(candidate.from - locator.hint), score }
  })
  scored.sort((a, b) => b.score - a.score || a.distance - b.distance)
  return scored[0].candidate
}

function createLocalId(prefix: string): string {
  if (globalThis.crypto?.randomUUID) return `${prefix}-${globalThis.crypto.randomUUID()}`
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}
