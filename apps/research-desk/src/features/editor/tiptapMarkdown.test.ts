import type { Editor } from '@tiptap/react'
import { describe, expect, it } from 'vitest'
import { normalizeEditorMarkdownForTiptap, serializeEditorMarkdown } from './tiptap'

/** Unit Separator (U+001F) that `@tiptap/markdown` emits inside populated table
 * cells while serializing. It must never reach persisted markdown — it breaks
 * GFM table re-parsing. */
const SEPARATOR = String.fromCharCode(0x1f)

function editorReturning(markdown: string): Editor {
  return { getMarkdown: () => markdown } as unknown as Editor
}

describe('serializeEditorMarkdown', () => {
  it('strips the @tiptap/markdown table-cell separator artifact on save', () => {
    const editor = editorReturning(`| Hello${SEPARATOR} | World${SEPARATOR} |`)
    expect(serializeEditorMarkdown(editor)).toBe('| Hello | World |')
  })

  it('leaves artifact-free markdown unchanged', () => {
    const markdown = '# Title\n\nBody text with no tables.'
    expect(serializeEditorMarkdown(editorReturning(markdown))).toBe(markdown)
  })
})

describe('normalizeEditorMarkdownForTiptap', () => {
  it('heals legacy documents that still carry the table-cell separator on load', () => {
    expect(normalizeEditorMarkdownForTiptap(`| A${SEPARATOR} | B${SEPARATOR} |`)).toBe('| A | B |')
  })

  it('still normalizes block and inline math delimiters', () => {
    expect(normalizeEditorMarkdownForTiptap('\\[ a^2 \\]')).toBe('\n\n$$\na^2\n$$\n\n')
    expect(normalizeEditorMarkdownForTiptap('\\( b \\)')).toBe('$b$')
  })
})
