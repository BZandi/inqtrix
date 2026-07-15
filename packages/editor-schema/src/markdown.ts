import type { JSONContent } from '@tiptap/core'
import { getSchema } from '@tiptap/core'
import { MarkdownManager } from '@tiptap/markdown'
import { createEditorSchemaExtensions } from './extensions.js'
import { projectFinalDocument, projectOriginalDocument } from './suggestions.js'

const TIPTAP_TABLE_CELL_ARTIFACT = String.fromCharCode(0x1f)

let markdownManager: MarkdownManager | null = null

export type EditorProjectionMode = 'final' | 'original'

function getMarkdownManager(): MarkdownManager {
  markdownManager ??= new MarkdownManager({
    extensions: createEditorSchemaExtensions({ enableUndoRedo: false }),
    markedOptions: { gfm: true },
  })
  return markdownManager
}

export function normalizeEditorMarkdown(markdown: string): string {
  return sanitizeSerializedEditorMarkdown(markdown)
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, expression: string) => (
      `\n\n$$\n${expression.trim()}\n$$\n\n`
    ))
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expression: string) => (
      `$${expression.trim()}$`
    ))
}

export function sanitizeSerializedEditorMarkdown(markdown: string): string {
  return markdown.split(TIPTAP_TABLE_CELL_ARTIFACT).join('')
}

export function parseEditorMarkdown(markdown: string): JSONContent {
  const parsed = getMarkdownManager().parse(normalizeEditorMarkdown(markdown))
  return canonicalizeEditorJson(parsed)
}

export function serializeEditorJson(
  content: JSONContent,
  projection: EditorProjectionMode = 'final',
): string {
  const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
  const document = schema.nodeFromJSON(content)
  const projected = projection === 'final'
    ? projectFinalDocument(document)
    : projectOriginalDocument(document)
  return sanitizeSerializedEditorMarkdown(getMarkdownManager().serialize(projected.toJSON()))
}

export function canonicalizeEditorJson(content: JSONContent): JSONContent {
  const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
  return schema.nodeFromJSON(content).toJSON()
}
