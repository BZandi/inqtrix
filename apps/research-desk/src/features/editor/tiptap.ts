import { Extension, type Extensions } from '@tiptap/core'
import Highlight from '@tiptap/extension-highlight'
import Link from '@tiptap/extension-link'
import { BlockMath, InlineMath } from '@tiptap/extension-mathematics'
import { Table, TableCell, TableHeader, TableRow } from '@tiptap/extension-table'
import TaskItem from '@tiptap/extension-task-item'
import TaskList from '@tiptap/extension-task-list'
import TextAlign from '@tiptap/extension-text-align'
import { Markdown } from '@tiptap/markdown'
import StarterKit from '@tiptap/starter-kit'
import { type Node as ProseMirrorNode } from '@tiptap/pm/model'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'
import { createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { EditorSuggestionBlockCard } from './EditorSuggestionBlockCard'

type CommentDecorationOptions = {
  onClick?: (commentId: string) => void
  onSuggestionAccept?: (suggestionId: string) => void
  onSuggestionCancel?: (suggestionId: string) => void
  onSuggestionEdit?: (suggestionId: string, proposedText: string) => void
  onSuggestionRefine?: (suggestionId: string, instruction: string) => void
  onSuggestionReject?: (suggestionId: string) => void
  onSuggestionSelect?: (suggestionId: string) => void
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

type SuggestionMarkdownElement = HTMLElement & {
  __inqtrixMarkdownRoot?: Root
}

function buildInsertWidget(text: string): HTMLElement {
  const span = document.createElement('span')
  span.className = 'suggestion-insert'
  span.setAttribute('data-suggestion-insert', 'true')
  span.textContent = text
  return span
}

function buildBlockSuggestionWidget(
  item: SuggestionDecorationItem,
  callbacks: SuggestionDecorationCallbacks,
): HTMLElement {
  const wrapper = document.createElement('div') as SuggestionMarkdownElement
  wrapper.className = 'suggestion-block-widget'
  wrapper.setAttribute('data-suggestion-block-card', item.id)
  const root = createRoot(wrapper)
  wrapper.__inqtrixMarkdownRoot = root
  root.render(createElement(EditorSuggestionBlockCard, {
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
  }))
  return wrapper
}

function destroySuggestionMarkdownWidget(node: Node) {
  const element = node instanceof HTMLElement ? node as SuggestionMarkdownElement : null
  const root = element?.__inqtrixMarkdownRoot
  if (!root) return
  element.__inqtrixMarkdownRoot = undefined
  queueMicrotask(() => root.unmount())
}

function buildSuggestionDecorations(
  doc: ProseMirrorNode,
  items: SuggestionDecorationItem[],
  callbacks: SuggestionDecorationCallbacks,
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
      decorations.push(Decoration.widget(item.widgetAt ?? to, () => buildBlockSuggestionWidget(item, callbacks), {
        destroy: destroySuggestionMarkdownWidget,
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
    return [
      new Plugin<SuggestionDecorationState>({
        key: suggestionDecorationPluginKey,
        state: {
          init: () => ({ decorations: DecorationSet.empty }),
          apply(tr, value) {
            const meta = tr.getMeta(suggestionDecorationPluginKey) as { items: SuggestionDecorationItem[] } | undefined
            if (meta) {
              return { decorations: buildSuggestionDecorations(tr.doc, meta.items, callbacks) }
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
    CommentDecorationExtension.configure(options),
    SuggestionDecorationExtension.configure(options),
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

export function normalizeEditorMarkdownForTiptap(markdown: string) {
  return markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, expression: string) => (
      `\n\n$$\n${expression.trim()}\n$$\n\n`
    ))
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expression: string) => (
      `$${expression.trim()}$`
    ))
}
