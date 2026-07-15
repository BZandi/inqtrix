import type { Extensions } from '@tiptap/core'
import Document from '@tiptap/extension-document'
import Highlight from '@tiptap/extension-highlight'
import Link from '@tiptap/extension-link'
import { BlockMath, InlineMath } from '@tiptap/extension-mathematics'
import { Table, TableCell, TableHeader, TableRow } from '@tiptap/extension-table'
import TaskItem from '@tiptap/extension-task-item'
import TaskList from '@tiptap/extension-task-list'
import TextAlign from '@tiptap/extension-text-align'
import { Markdown } from '@tiptap/markdown'
import StarterKit from '@tiptap/starter-kit'
import { EDITOR_SUGGESTION_BLOCK_PARENTS } from './constants.js'
import {
  SuggestionDeletion,
  SuggestionInsertion,
  SuggestionModification,
} from './suggestionMarks.js'

export type EditorSchemaExtensionOptions = {
  enableUndoRedo?: boolean
  resizableTables?: boolean
}

const SUGGESTION_MARKS = 'insertion modification deletion'

const SUGGESTION_STARTER_BLOCK_PARENTS: ReadonlySet<string> = new Set(
  EDITOR_SUGGESTION_BLOCK_PARENTS.filter((name) => (
    name === 'codeBlock'
  )),
)

const SuggestionStarterKit = StarterKit.extend({
  addExtensions() {
    return (this.parent?.() ?? []).map((extension) => (
      SUGGESTION_STARTER_BLOCK_PARENTS.has(extension.name)
        ? extension.extend({ marks: SUGGESTION_MARKS })
        : extension
    ))
  },
})

export function createEditorSchemaExtensions(
  options: EditorSchemaExtensionOptions = {},
): Extensions {
  return [
    SuggestionStarterKit.configure({
      document: false,
      link: false,
      undoRedo: options.enableUndoRedo === false ? false : {},
    }),
    Document,
    Markdown.configure({
      markedOptions: {
        gfm: true,
      },
    }),
    BlockMath.configure({
      katexOptions: {
        displayMode: true,
        throwOnError: false,
      },
    }),
    InlineMath.configure({
      katexOptions: {
        displayMode: false,
        throwOnError: false,
      },
    }),
    Link.configure(safeLinkOptions()),
    Highlight.configure({ multicolor: false }),
    TaskList,
    TaskItem.configure({ nested: true }),
    Table.configure({ resizable: options.resizableTables ?? false }),
    TableRow,
    TableHeader,
    TableCell,
    TextAlign.configure({ types: ['heading', 'paragraph'] }),
    SuggestionDeletion,
    SuggestionInsertion,
    SuggestionModification,
  ]
}

function safeLinkOptions() {
  return {
    HTMLAttributes: {
      rel: 'noopener noreferrer nofollow',
      target: '_blank',
    },
    autolink: true,
    defaultProtocol: 'https',
    isAllowedUri: (url: string, context: { defaultValidate: (url: string) => boolean }) => (
      isSafeEditorUrl(url) && context.defaultValidate(url)
    ),
    linkOnPaste: true,
    openOnClick: false,
    protocols: ['http', 'https', 'mailto'],
    shouldAutoLink: isSafeEditorUrl,
  }
}

function isSafeEditorUrl(value: string): boolean {
  const trimmed = value.trim()
  if (!trimmed) return false
  try {
    const url = new URL(trimmed.includes(':') ? trimmed : `https://${trimmed}`)
    return url.protocol === 'http:' || url.protocol === 'https:' || url.protocol === 'mailto:'
  } catch {
    return false
  }
}
