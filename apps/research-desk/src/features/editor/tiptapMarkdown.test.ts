import { createEditorSchemaExtensions } from '@inqtrix/editor-schema'
import { Editor as HeadlessEditor } from '@tiptap/core'
import type { Editor } from '@tiptap/react'
import { describe, expect, it } from 'vitest'
import {
  normalizeEditorMarkdownForTiptap,
  serializeEditorFinalProjectionMarkdown,
  serializeEditorMarkdown,
} from './tiptap'

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

  it('projects a collaboration replacement to its canonical final markdown', () => {
    const metadata = {
      authorId: 'user-1',
      createdAt: 10,
      patchId: 'patch-1',
      suggestionId: 'suggestion-1',
    }
    const editor = new HeadlessEditor({
      content: {
        content: [{
          content: [
            {
              marks: [{
                attrs: { ...metadata, id: metadata.suggestionId, kind: 'deletion' },
                type: 'deletion',
              }],
              text: 'old',
              type: 'text',
            },
            {
              marks: [{
                attrs: { ...metadata, id: metadata.suggestionId, kind: 'insertion' },
                type: 'insertion',
              }],
              text: 'new',
              type: 'text',
            },
          ],
          type: 'paragraph',
        }],
        type: 'doc',
      },
      element: null,
      extensions: createEditorSchemaExtensions({ enableUndoRedo: false }),
      injectCSS: false,
    })

    try {
      expect(serializeEditorFinalProjectionMarkdown(editor)).toBe('new')
    } finally {
      editor.destroy()
    }
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
