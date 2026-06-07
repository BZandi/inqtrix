import { Extension, getMarkRange, type Editor, type Extensions } from '@tiptap/core'
import Highlight from '@tiptap/extension-highlight'
import Link from '@tiptap/extension-link'
import { BlockMath, InlineMath } from '@tiptap/extension-mathematics'
import { Table, TableCell, TableHeader, TableRow } from '@tiptap/extension-table'
import { Placeholder } from '@tiptap/extensions'
import TaskItem from '@tiptap/extension-task-item'
import TaskList from '@tiptap/extension-task-list'
import TextAlign from '@tiptap/extension-text-align'
import { Markdown } from '@tiptap/markdown'
import StarterKit from '@tiptap/starter-kit'
import { type Node as ProseMirrorNode } from '@tiptap/pm/model'
import { type EditorState, Plugin, PluginKey, TextSelection } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'
import { ReactRenderer } from '@tiptap/react'
import { EditorSuggestionBlockCard, type EditorSuggestionBlockCardProps } from './EditorSuggestionBlockCard'
import { SlashCommandExtension, type SlashCommandConfig } from './slashCommand'

type CommentDecorationOptions = {
  onClick?: (commentId: string) => void
  onSuggestionAccept?: (suggestionId: string) => void
  onSuggestionCancel?: (suggestionId: string) => void
  onSuggestionEdit?: (suggestionId: string, proposedText: string) => void
  onSuggestionRefine?: (suggestionId: string, instruction: string) => void
  onSuggestionReject?: (suggestionId: string) => void
  onSuggestionSelect?: (suggestionId: string) => void
  /** Localized accessible label/tooltip for the click-to-remove syntax markers
   * (e.g. "Formatierung entfernen"). Empty string renders markers as inert. */
  syntaxMarkerRemoveLabel?: string
  /** Placeholder shown on an empty paragraph (e.g. "Write, or type / for commands"). */
  placeholderEmpty?: string
  /** Placeholder shown on an empty heading (e.g. "Heading"). */
  placeholderHeading?: string
  /** Localized config for the `/` slash command menu. Omit to disable the menu. */
  slash?: SlashCommandConfig
}

export type CommentDecorationItem = {
  from: number
  id: string
  kind: 'collect' | 'inline_edit' | 'evidence_review'
  selected: boolean
  status: 'open' | 'resolved' | 'stale'
  to: number
}

type CommentDecorationState = {
  decorations: DecorationSet
}

export const commentDecorationPluginKey = new PluginKey<CommentDecorationState>('inqtrixCommentDecorations')

function buildCommentDecorations(doc: ProseMirrorNode, items: CommentDecorationItem[]): DecorationSet {
  const maxPos = doc.content.size
  const decorations = items.flatMap((item) => {
    const from = Math.max(0, Math.min(item.from, maxPos))
    const to = Math.max(from, Math.min(item.to, maxPos))
    if (from >= to) return []
    return [Decoration.inline(
      from,
      to,
      {
        class: 'editor-comment-anchor',
        'data-editor-comment-anchor': item.id,
        'data-editor-comment-kind': item.kind,
        'data-editor-comment-status': item.status,
        ...(item.selected ? { 'data-editor-comment-selected': 'true' } : {}),
      },
      { commentId: item.id },
    )]
  })
  return DecorationSet.create(doc, decorations)
}

export const CommentDecorationExtension = Extension.create<CommentDecorationOptions>({
  name: 'commentDecorations',

  addOptions() {
    return {
      onClick: undefined,
    }
  },

  addProseMirrorPlugins() {
    const onClick = this.options.onClick
    return [
      new Plugin<CommentDecorationState>({
        key: commentDecorationPluginKey,
        state: {
          init: () => ({ decorations: DecorationSet.empty }),
          apply(tr, value) {
            const meta = tr.getMeta(commentDecorationPluginKey) as { items: CommentDecorationItem[] } | undefined
            if (meta) {
              return { decorations: buildCommentDecorations(tr.doc, meta.items) }
            }
            if (tr.docChanged) {
              return { decorations: value.decorations.map(tr.mapping, tr.doc) }
            }
            return value
          },
        },
        props: {
          decorations(state) {
            return commentDecorationPluginKey.getState(state)?.decorations ?? DecorationSet.empty
          },
          handleClick(view, pos) {
            if (!onClick) return false
            const set = commentDecorationPluginKey.getState(view.state)?.decorations
            if (!set) return false
            const hits = set.find(pos, pos)
            const commentId = hits.length > 0
              ? (hits[0].spec as { commentId?: string }).commentId
              : undefined
            if (!commentId) return false
            onClick(commentId)
            return false
          },
        },
      }),
    ]
  },
})

export type SuggestionDecorationSegment = {
  text: string
  type: 'delete' | 'equal' | 'insert'
}

export type SuggestionDecorationItem = {
  active: boolean
  acceptLabel: string
  cancelLabel: string
  display: 'block' | 'inline'
  editLabel: string
  error?: string
  from: number
  id: string
  isRunning: boolean
  proposedLabel: string
  proposedText: string
  refineLabel: string
  refinementPlaceholder: string
  rejectLabel: string
  revision: number
  revisionLabel: string
  reviewSurface: 'editor' | 'panel'
  runningLabel: string
  saveLabel: string
  segments: SuggestionDecorationSegment[]
  sendLabel: string
  stopLabel: string
  to: number
  widgetAt?: number
}

type SuggestionDecorationState = {
  decorations: DecorationSet
}

export const suggestionDecorationPluginKey = new PluginKey<SuggestionDecorationState>('inqtrixSuggestionDecorations')

type SuggestionDecorationCallbacks = {
  onAccept?: (suggestionId: string) => void
  onCancel?: (suggestionId: string) => void
  onEdit?: (suggestionId: string, proposedText: string) => void
  onRefine?: (suggestionId: string, instruction: string) => void
  onReject?: (suggestionId: string) => void
  onSelect?: (suggestionId: string) => void
}

type SuggestionWidgetElement = HTMLElement & {
  __inqtrixSuggestionRenderer?: ReactRenderer
}

function buildInsertWidget(text: string): HTMLElement {
  const span = document.createElement('span')
  span.className = 'suggestion-insert'
  span.setAttribute('data-suggestion-insert', 'true')
  span.textContent = text
  return span
}

function buildBlockSuggestionWidget(
  editor: Editor,
  item: SuggestionDecorationItem,
  callbacks: SuggestionDecorationCallbacks,
): HTMLElement {
  const props: EditorSuggestionBlockCardProps = {
    active: item.active,
    labels: {
      accept: item.acceptLabel,
      cancel: item.cancelLabel,
      edit: item.editLabel,
      proposed: item.proposedLabel,
      refine: item.refineLabel,
      refinementPlaceholder: item.refinementPlaceholder,
      reject: item.rejectLabel,
      revision: item.revisionLabel,
      running: item.runningLabel,
      save: item.saveLabel,
      send: item.sendLabel,
      stop: item.stopLabel,
    },
    error: item.error,
    id: item.id,
    isRunning: item.isRunning,
    onAccept: callbacks.onAccept,
    onCancelRun: callbacks.onCancel,
    onEdit: callbacks.onEdit,
    onRefine: callbacks.onRefine,
    onReject: callbacks.onReject,
    onSelect: callbacks.onSelect,
    proposedText: item.proposedText,
    reviewSurface: item.reviewSurface,
    revision: item.revision,
  }
  const renderer = new ReactRenderer(EditorSuggestionBlockCard, {
    editor,
    as: 'div',
    className: 'suggestion-block-widget',
    props,
  })
  const element = renderer.element as SuggestionWidgetElement
  element.setAttribute('data-suggestion-block-card', item.id)
  element.__inqtrixSuggestionRenderer = renderer
  return element
}

function destroySuggestionWidget(node: Node) {
  const element = node instanceof HTMLElement ? (node as SuggestionWidgetElement) : null
  const renderer = element?.__inqtrixSuggestionRenderer
  if (!renderer) return
  element.__inqtrixSuggestionRenderer = undefined
  renderer.destroy()
}

function buildSuggestionDecorations(
  doc: ProseMirrorNode,
  items: SuggestionDecorationItem[],
  callbacks: SuggestionDecorationCallbacks,
  editor: Editor,
): DecorationSet {
  const maxPos = doc.content.size
  const decorations: Decoration[] = []
  for (const item of items) {
    const from = Math.max(0, Math.min(item.from, maxPos))
    const to = Math.max(from, Math.min(item.to, maxPos))
    if (item.display === 'block') {
      if (to > from) {
        decorations.push(Decoration.inline(from, to, {
          class: `suggestion-block-original${item.active ? ' suggestion-block-original-active' : ''}`,
          'data-suggestion-id': item.id,
        }))
      }
      decorations.push(Decoration.widget(item.widgetAt ?? to, () => buildBlockSuggestionWidget(editor, item, callbacks), {
        destroy: destroySuggestionWidget,
        key: `suggestion-block-${item.id}-${item.revision}-${item.proposedText.length}-${item.isRunning ? 'running' : 'idle'}-${item.error ?? ''}`,
        side: 1,
        stopEvent: (event) => event.type !== 'click',
      }))
      continue
    }
    let pos = from
    for (const segment of item.segments) {
      if (segment.type === 'equal') {
        pos = Math.min(maxPos, pos + segment.text.length)
        continue
      }
      if (segment.type === 'delete') {
        const end = Math.min(maxPos, pos + segment.text.length)
        if (end > pos) {
          decorations.push(Decoration.inline(pos, end, { class: 'suggestion-delete', 'data-suggestion-id': item.id }))
        }
        pos = end
        continue
      }
      decorations.push(Decoration.widget(pos, () => buildInsertWidget(segment.text), {
        key: `suggestion-insert-${item.id}-${pos}-${segment.text}`,
        marks: [],
        side: 1,
      }))
    }
  }
  return DecorationSet.create(doc, decorations)
}

export const SuggestionDecorationExtension = Extension.create<CommentDecorationOptions>({
  name: 'suggestionDecorations',

  addOptions() {
    return {
      onClick: undefined,
      onSuggestionAccept: undefined,
      onSuggestionCancel: undefined,
      onSuggestionEdit: undefined,
      onSuggestionRefine: undefined,
      onSuggestionReject: undefined,
      onSuggestionSelect: undefined,
    }
  },

  addProseMirrorPlugins() {
    const callbacks: SuggestionDecorationCallbacks = {
      onAccept: this.options.onSuggestionAccept,
      onCancel: this.options.onSuggestionCancel,
      onEdit: this.options.onSuggestionEdit,
      onRefine: this.options.onSuggestionRefine,
      onReject: this.options.onSuggestionReject,
      onSelect: this.options.onSuggestionSelect,
    }
    const editor = this.editor
    return [
      new Plugin<SuggestionDecorationState>({
        key: suggestionDecorationPluginKey,
        state: {
          init: () => ({ decorations: DecorationSet.empty }),
          apply(tr, value) {
            const meta = tr.getMeta(suggestionDecorationPluginKey) as { items: SuggestionDecorationItem[] } | undefined
            if (meta) {
              return { decorations: buildSuggestionDecorations(tr.doc, meta.items, callbacks, editor) }
            }
            if (tr.docChanged) {
              return { decorations: value.decorations.map(tr.mapping, tr.doc) }
            }
            return value
          },
        },
        props: {
          decorations(state) {
            return suggestionDecorationPluginKey.getState(state)?.decorations ?? DecorationSet.empty
          },
        },
      }),
    ]
  },
})

// ----- Syntax marker reveal (Obsidian Live Preview style) -----
// Tiptap's document has no literal markdown markers (bold is a mark, not `**`).
// This plugin injects the markers as inert widget decorations around the
// formatted span that contains the caret, so they read like Obsidian's revealed
// source. The widgets are click-to-remove: clicking a marker strips that
// formatting. Markers are visual hints, never part of `getMarkdown()` output.

type SyntaxMarkerRevealOptions = {
  removeLabel: string
}

/** Open === close glyph for symmetric inline marks. Verified against the
 * `@tiptap/markdown` serializer so the reveal matches the Source view exactly
 * (italic is a single `*`, code a single backtick). */
const INLINE_MARKER_BY_MARK: Record<string, string> = {
  bold: '**',
  italic: '*',
  strike: '~~',
  code: '`',
  highlight: '==',
}

/** Outer -> inner. Earlier entries render furthest from the text, so overlapping
 * marks (e.g. bold+italic) get a deterministic, symmetric glyph order. */
const SYNTAX_MARK_ORDER = ['highlight', 'bold', 'italic', 'strike', 'code'] as const

type SyntaxMarkerState = {
  decorations: DecorationSet
}

export const syntaxMarkerPluginKey = new PluginKey<SyntaxMarkerState>('inqtrixSyntaxMarkerReveal')

function buildSyntaxMarkerWidget(text: string, removeLabel: string, onRemove: () => void): HTMLElement {
  const span = document.createElement('span')
  span.className = 'editor-syntax-marker'
  span.textContent = text
  span.setAttribute('contenteditable', 'false')
  if (removeLabel) {
    span.setAttribute('role', 'button')
    span.setAttribute('aria-label', removeLabel)
    span.setAttribute('title', removeLabel)
  } else {
    span.setAttribute('aria-hidden', 'true')
  }
  span.addEventListener('mousedown', (event) => {
    event.preventDefault()
    event.stopPropagation()
    onRemove()
  })
  return span
}

function syntaxMarkerWidget(
  pos: number,
  text: string,
  key: string,
  side: number,
  removeLabel: string,
  onRemove: () => void,
): Decoration {
  return Decoration.widget(pos, () => buildSyntaxMarkerWidget(text, removeLabel, onRemove), {
    key,
    marks: [],
    side,
    ignoreSelection: true,
    stopEvent: () => true,
  })
}

function buildSyntaxMarkerDecorations(state: EditorState, editor: Editor, removeLabel: string): DecorationSet {
  const { selection, schema, doc } = state
  // Reveal only for a collapsed caret — not while a multi-character selection is
  // being dragged across spans.
  if (!selection.empty) return DecorationSet.empty

  const $head = selection.$head
  const decorations: Decoration[] = []

  // Heading line: '#'*level before the text. Only on a heading that HAS text —
  // an empty heading shows the "Ueberschrift" placeholder instead, and rendering
  // the '#' widget at the same line-start would overlap it.
  const parent = $head.parent
  if (parent.type.name === 'heading' && parent.content.size > 0) {
    const level = Number(parent.attrs.level ?? 1)
    const headStart = $head.start()
    decorations.push(syntaxMarkerWidget(
      headStart,
      `${'#'.repeat(level)} `,
      `synmark-heading-${level}`,
      -1,
      removeLabel,
      () => editor.chain().focus().setTextSelection(headStart).setParagraph().run(),
    ))
  }

  // Symmetric inline marks.
  let orderIndex = 0
  for (const markName of SYNTAX_MARK_ORDER) {
    const markType = schema.marks[markName]
    if (!markType) continue
    const range = getMarkRange($head, markType)
    if (!range || range.to <= range.from) continue
    const { from, to } = range
    const marker = INLINE_MARKER_BY_MARK[markName]
    const openSide = -10 + orderIndex
    const closeSide = 10 - orderIndex
    orderIndex += 1
    const onRemove = () => editor.chain().focus().setTextSelection({ from, to }).unsetMark(markName).run()
    decorations.push(
      syntaxMarkerWidget(from, marker, `synmark-open-${markName}-${from}`, openSide, removeLabel, onRemove),
      syntaxMarkerWidget(to, marker, `synmark-close-${markName}-${to}`, closeSide, removeLabel, onRemove),
    )
  }

  // Link: [text](url) — asymmetric markers, href read from the mark.
  const linkType = schema.marks.link
  if (linkType) {
    const range = getMarkRange($head, linkType)
    if (range && range.to > range.from) {
      const { from, to } = range
      const node = doc.nodeAt(from)
      const href = (node?.marks.find((mark) => mark.type === linkType)?.attrs.href as string | undefined) ?? ''
      const onRemove = () => editor.chain().focus().setTextSelection({ from, to }).unsetLink().run()
      decorations.push(
        syntaxMarkerWidget(from, '[', `synmark-link-open-${from}`, -11, removeLabel, onRemove),
        syntaxMarkerWidget(to, `](${href})`, `synmark-link-close-${to}`, 11, removeLabel, onRemove),
      )
    }
  }

  if (decorations.length === 0) return DecorationSet.empty
  return DecorationSet.create(doc, decorations)
}

/** Inline keyboard editing of a revealed marker — removes ONE marker character,
 * not the whole formatting, so it matches what the user sees:
 *   heading `##` -> `#` (H2 -> H1 -> paragraph);
 *   emphasis stars `***` -> `**` -> `*` -> none (bold+italic -> bold -> italic -> plain);
 *   strike/code/highlight/link have no valid half-state -> the whole mark comes off.
 * Backspace acts on the marker to the caret's left (open marker / heading hash),
 * Delete on the one to its right (close marker). Only fires exactly on a marker
 * boundary, so editing inside the word stays normal. The caret stays put. */
function removeMarkerAtCaret(editor: Editor, state: EditorState, backspace: boolean): boolean {
  const { selection, schema } = state
  if (!selection.empty) return false
  const $head = selection.$head
  const pos = $head.pos

  // Heading: one hash off → drop the level by one (h1 falls back to paragraph).
  if (backspace && $head.parent.type.name === 'heading' && pos === $head.start()) {
    const level = Number($head.parent.attrs.level ?? 1)
    return level > 1
      ? editor.chain().updateAttributes('heading', { level: level - 1 }).run()
      : editor.commands.setParagraph()
  }

  const atEdge = (range: { from: number; to: number } | undefined) =>
    !!range && (backspace ? pos === range.from : pos === range.to)
  const rangeOf = (name: string): { from: number; to: number } | undefined => {
    const type = schema.marks[name]
    if (!type) return undefined
    const range = getMarkRange($head, type)
    return range ? { from: range.from, to: range.to } : undefined
  }
  const removeWhole = (name: string, range: { from: number; to: number }) => {
    const type = schema.marks[name]
    if (!type) return false
    return editor
      .chain()
      .command(({ tr, dispatch }) => {
        if (dispatch) tr.removeMark(range.from, range.to, type)
        return true
      })
      .run()
  }

  // Outer marks first (left-to-right marker order): link, highlight.
  for (const name of ['link', 'highlight']) {
    const range = rangeOf(name)
    if (atEdge(range)) return removeWhole(name, range as { from: number; to: number })
  }

  // Emphasis ladder: one star off. Star count = (bold?2:0)+(italic?1:0);
  // decrement maps back to bold=count>=2, italic=count is odd.
  const boldType = schema.marks.bold
  const italicType = schema.marks.italic
  const boldRange = rangeOf('bold')
  const italicRange = rangeOf('italic')
  if (boldType && italicType && (atEdge(boldRange) || atEdge(italicRange))) {
    const matched = [boldRange, italicRange].filter(atEdge) as { from: number; to: number }[]
    const from = Math.min(...matched.map((range) => range.from))
    const to = Math.max(...matched.map((range) => range.to))
    const stars = (boldRange ? 2 : 0) + (italicRange ? 1 : 0) - 1
    const wantBold = stars >= 2
    const wantItalic = stars % 2 === 1
    return editor
      .chain()
      .command(({ tr, dispatch }) => {
        if (dispatch) {
          if (Boolean(boldRange) !== wantBold) {
            if (wantBold) tr.addMark(from, to, boldType.create())
            else tr.removeMark(from, to, boldType)
          }
          if (Boolean(italicRange) !== wantItalic) {
            if (wantItalic) tr.addMark(from, to, italicType.create())
            else tr.removeMark(from, to, italicType)
          }
        }
        return true
      })
      .run()
  }

  // Inner marks: strike, code (no clean half-state → whole mark off).
  for (const name of ['strike', 'code']) {
    const range = rangeOf(name)
    if (atEdge(range)) return removeWhole(name, range as { from: number; to: number })
  }
  return false
}

const WORD_CHAR = /[\p{L}\p{N}_]/u

/** Word selection around `pos`, computed from the document text (not the DOM).
 * The reveal markers are widget decorations inserted between characters; the
 * browser's native double-click word selection walks the DOM and breaks at those
 * widget boundaries. Deriving the word from the doc text sidesteps them entirely.
 * Returns null on whitespace / a non-textblock, so the caller falls back to the
 * default handler (correct on plain text). Umlaut-safe via the `\p{L}` class. */
function wordSelectionAt(state: EditorState, pos: number): TextSelection | null {
  const $pos = state.doc.resolve(pos)
  const parent = $pos.parent
  if (!parent.isTextblock) return null
  const text = parent.textContent
  if (!text) return null
  const start = $pos.start()
  const offset = Math.min(Math.max(pos - start, 0), text.length)
  const onWord = offset < text.length && WORD_CHAR.test(text[offset])
  const afterWord = offset > 0 && WORD_CHAR.test(text[offset - 1])
  if (!onWord && !afterWord) return null
  let from = offset
  let to = offset
  while (from > 0 && WORD_CHAR.test(text[from - 1])) from -= 1
  while (to < text.length && WORD_CHAR.test(text[to])) to += 1
  if (from === to) return null
  return TextSelection.create(state.doc, start + from, start + to)
}

export const SyntaxMarkerRevealExtension = Extension.create<SyntaxMarkerRevealOptions>({
  name: 'syntaxMarkerReveal',

  addOptions() {
    return {
      removeLabel: '',
    }
  },

  addProseMirrorPlugins() {
    const editor = this.editor
    const removeLabel = this.options.removeLabel
    return [
      new Plugin<SyntaxMarkerState>({
        key: syntaxMarkerPluginKey,
        state: {
          init: (_config, editorState) => ({
            decorations: buildSyntaxMarkerDecorations(editorState, editor, removeLabel),
          }),
          apply(tr, value, _oldState, newState) {
            if (tr.selectionSet || tr.docChanged) {
              return { decorations: buildSyntaxMarkerDecorations(newState, editor, removeLabel) }
            }
            return value
          },
        },
        props: {
          decorations(pmState) {
            // Live mode only — Source mode keeps the editor mounted but non-editable.
            if (!editor?.isEditable) return DecorationSet.empty
            return syntaxMarkerPluginKey.getState(pmState)?.decorations ?? DecorationSet.empty
          },
          handleKeyDown(view, event) {
            // Inline marker editing: Backspace eats the marker on the left, Delete
            // the one on the right. Runs before the default keymap; falls through
            // (returns false) for everything except a caret on a marker boundary.
            if (!editor?.isEditable || event.isComposing || event.metaKey || event.ctrlKey || event.altKey) return false
            if (event.key === 'Backspace') return removeMarkerAtCaret(editor, view.state, true)
            if (event.key === 'Delete') return removeMarkerAtCaret(editor, view.state, false)
            return false
          },
          handleDoubleClick(view, pos) {
            // Select the word ourselves (from the doc text) so the reveal widgets
            // between characters can't break native double-click selection. Null
            // (whitespace) falls through to the default handler.
            if (!editor?.isEditable) return false
            const selection = wordSelectionAt(view.state, pos)
            if (!selection) return false
            view.dispatch(view.state.tr.setSelection(selection))
            return true
          },
        },
      }),
    ]
  },
})

export function createEditorExtensions(options: CommentDecorationOptions = {}): Extensions {
  return [
    StarterKit.configure({
      link: false,
    }),
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
    Table.configure({ resizable: true }),
    TableRow,
    TableHeader,
    TableCell,
    TextAlign.configure({ types: ['heading', 'paragraph'] }),
    Placeholder.configure({
      showOnlyCurrent: true,
      placeholder: ({ node }) =>
        node.type.name === 'heading' ? options.placeholderHeading ?? '' : options.placeholderEmpty ?? '',
    }),
    CommentDecorationExtension.configure(options),
    SuggestionDecorationExtension.configure(options),
    SyntaxMarkerRevealExtension.configure({ removeLabel: options.syntaxMarkerRemoveLabel ?? '' }),
    ...(options.slash ? [SlashCommandExtension.configure({ config: options.slash })] : []),
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
    isAllowedUri: (url: string, context: { defaultValidate: (url: string) => boolean }) => {
      return isSafeEditorUrl(url) && context.defaultValidate(url)
    },
    linkOnPaste: true,
    openOnClick: false,
    protocols: ['http', 'https', 'mailto'],
    shouldAutoLink: isSafeEditorUrl,
  }
}

function isSafeEditorUrl(value: string) {
  const trimmed = value.trim()
  if (!trimmed) return false
  try {
    const url = new URL(trimmed.includes(':') ? trimmed : `https://${trimmed}`)
    return url.protocol === 'http:' || url.protocol === 'https:' || url.protocol === 'mailto:'
  } catch {
    return false
  }
}

/**
 * Unit Separator (U+001F) that `@tiptap/markdown` emits inside populated table
 * cells while serializing (e.g. `| Hello |`). It is invisible, is never
 * part of the user's text, and breaks GFM table re-parsing when persisted — a
 * round-trip through it collapses the table. Stripped on every serialize and
 * defensively on load so legacy documents saved before this fix self-heal.
 */
const TIPTAP_TABLE_CELL_ARTIFACT = String.fromCharCode(0x1f)

/**
 * Serialize the editor document to markdown for persistence, removing the
 * `@tiptap/markdown` table-cell artifact (`TIPTAP_TABLE_CELL_ARTIFACT`). This is
 * the single save-path entry point: the live/source sync comparison must use it
 * too, otherwise a populated table always looks "changed" (raw `getMarkdown()`
 * still carries the separator) and triggers a needless reparse.
 */
export function serializeEditorMarkdown(editor: Editor): string {
  return editor.getMarkdown().split(TIPTAP_TABLE_CELL_ARTIFACT).join('')
}

export function normalizeEditorMarkdownForTiptap(markdown: string) {
  return markdown
    .split(TIPTAP_TABLE_CELL_ARTIFACT).join('')
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, expression: string) => (
      `\n\n$$\n${expression.trim()}\n$$\n\n`
    ))
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expression: string) => (
      `$${expression.trim()}$`
    ))
}
