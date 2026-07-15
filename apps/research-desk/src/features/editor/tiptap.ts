import {
  createEditorSchemaExtensions,
  isRemoteYjsTransaction,
  normalizeEditorMarkdown,
  sanitizeSerializedEditorMarkdown,
  serializeEditorJson,
  SUGGESTION_MARK_NAMES,
  transformToInqtrixSuggestionTransaction,
  type SuggestionKind,
  type SuggestionMetadata,
} from '@inqtrix/editor-schema'
import { Extension, getMarkRange, type Editor, type Extensions } from '@tiptap/core'
import { Placeholder } from '@tiptap/extensions'
import { type Mark, type Node as ProseMirrorNode } from '@tiptap/pm/model'
import { type EditorState, Plugin, PluginKey, TextSelection, type Transaction } from '@tiptap/pm/state'
import { Decoration, DecorationSet, type DecorationAttrs, type EditorView } from '@tiptap/pm/view'
import { ReactRenderer } from '@tiptap/react'
import { EditorSuggestionBlockCard, type EditorSuggestionBlockCardProps } from './EditorSuggestionBlockCard'
import { SlashCommandExtension, type SlashCommandConfig } from './slashCommand'

export type CollaborationPresenceUser = {
  color: string
  id: string
  name: string
}

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

export type CollaborationReviewDisplay = 'all' | 'final' | 'original' | 'simple'

export type CollaborationReviewOverlayUpdate = {
  collaboration?: boolean
  display: CollaborationReviewDisplay
  documentId?: string | null
  enabled: boolean
  selectedSuggestionIds?: readonly string[]
  visibleSuggestionIds?: readonly string[]
  writeAuthorId?: string | null
  writeMode?: 'edit' | 'suggest' | 'view'
}

type CollaborationReviewExtensionOptions = {
  initialPolicy?: CollaborationReviewOverlayUpdate
}

type CollaborationReviewExtensionStorage = {
  grouping: CollaborationSuggestionGroupingCoordinator
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
  providerActionsDisabled: boolean
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
    providerActionsDisabled: item.providerActionsDisabled,
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
        key: `suggestion-block-${item.id}-${item.revision}-${item.proposedText.length}-${item.isRunning ? 'running' : 'idle'}-${item.providerActionsDisabled ? 'restricted' : 'enabled'}-${item.error ?? ''}`,
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

type CollaborationReviewMarkKind = 'deletion' | 'insertion' | 'modification'

export type CollaborationReviewMarkPresentation = {
  className: string
  display: CollaborationReviewDisplay
  style: string
}

type CollaborationReviewOverlayState = {
  collaboration: boolean
  decorations: DecorationSet
  display: CollaborationReviewDisplay
  documentId: string | null
  enabled: boolean
  selectedSuggestionIds: ReadonlySet<string>
  visibleSuggestionIds: ReadonlySet<string> | null
  writeAuthorId: string | null
  writeMode: 'edit' | 'suggest' | 'view'
}

const DEFAULT_COLLABORATION_REVIEW_STATE: CollaborationReviewOverlayState = {
  collaboration: false,
  decorations: DecorationSet.empty,
  display: 'final',
  documentId: null,
  enabled: false,
  selectedSuggestionIds: new Set(),
  visibleSuggestionIds: null,
  writeAuthorId: null,
  writeMode: 'edit',
}

export const collaborationReviewPluginKey = new PluginKey<CollaborationReviewOverlayState>(
  'inqtrixCollaborationReview',
)

export const collaborationSuggestionErrorEvent = 'inqtrix:collaboration-suggestion-error'
export const collaborationSuggestionCollisionEvent = 'inqtrix:collaboration-suggestion-collision'

export type CollaborationSuggestionCollision = {
  patchId: string
  suggestionId: string
}

export const COLLABORATION_SUGGESTION_GROUP_IDLE_MS = 5_000

type SuggestionGroupingContext = {
  authorId: string | null
  documentId: string | null
  writeMode: 'edit' | 'suggest' | 'view'
}

type SuggestionGroupingCoordinatorOptions = {
  createPatchId?: () => string
  now?: () => number
}

/** The sole owner of local suggest-mode patch identity. Natural caret movement
 * caused by typing stays grouped; explicit selection transactions and remote
 * boundaries reset it before the next edit. */
export class CollaborationSuggestionGroupingCoordinator {
  private readonly createPatchId: () => string
  private readonly now: () => number
  private context: SuggestionGroupingContext = {
    authorId: null,
    documentId: null,
    writeMode: 'view',
  }
  private group: (SuggestionMetadata & { lastActivityAt: number }) | null = null

  constructor(options: SuggestionGroupingCoordinatorOptions = {}) {
    this.createPatchId = options.createPatchId ?? (() => crypto.randomUUID())
    this.now = options.now ?? (() => Date.now())
  }

  observeContext(context: SuggestionGroupingContext): void {
    if (
      context.authorId !== this.context.authorId
      || context.documentId !== this.context.documentId
      || context.writeMode !== this.context.writeMode
    ) {
      this.reset()
    }
    this.context = context
  }

  metadata(): SuggestionMetadata {
    const { authorId, documentId, writeMode } = this.context
    if (!authorId || !documentId || writeMode !== 'suggest') {
      throw new Error('Suggestion grouping requires an active document and author.')
    }
    const now = this.now()
    if (!this.group || now - this.group.lastActivityAt >= COLLABORATION_SUGGESTION_GROUP_IDLE_MS) {
      this.group = {
        authorId,
        createdAt: now,
        lastActivityAt: now,
        patchId: this.createPatchId(),
      }
    } else {
      this.group.lastActivityAt = now
    }
    return {
      authorId: this.group.authorId,
      createdAt: this.group.createdAt,
      patchId: this.group.patchId,
    }
  }

  reset(): void {
    this.group = null
  }
}

/**
 * Return the complete visual contract for one tracked-change mark. Filtered
 * marks deliberately use the clean final projection so a filter never exposes
 * unrelated review overlays.
 */
export function collaborationReviewMarkPresentation(
  kind: CollaborationReviewMarkKind,
  display: CollaborationReviewDisplay,
  active: boolean,
  visible = true,
): CollaborationReviewMarkPresentation {
  const effectiveDisplay = visible ? display : 'final'
  const activeStyle = active
    ? 'box-shadow: 0 0 0 2px color-mix(in oklab, var(--brand) 45%, transparent);'
    : ''
  const common = `border-radius: 0.2rem; -webkit-box-decoration-break: clone; box-decoration-break: clone; ${activeStyle}`
  let style: string
  if (kind === 'insertion') {
    if (effectiveDisplay === 'original') {
      style = 'display: none;'
    } else if (effectiveDisplay === 'all') {
      style = `${common} background: color-mix(in oklab, var(--success-subtle) 78%, transparent); color: var(--success); text-decoration: none;`
    } else if (effectiveDisplay === 'simple') {
      style = `${common} background: color-mix(in oklab, var(--success-subtle) 52%, transparent); color: inherit; text-decoration: none;`
    } else {
      style = `${common} background: transparent; color: inherit; text-decoration: none;`
    }
  } else if (kind === 'deletion') {
    if (effectiveDisplay === 'all' || (effectiveDisplay === 'simple' && active)) {
      style = `${common} background: color-mix(in oklab, var(--destructive-subtle) 70%, transparent); color: var(--destructive); text-decoration: line-through;`
    } else if (effectiveDisplay === 'original') {
      style = `${common} background: transparent; color: inherit; text-decoration: none;`
    } else {
      style = 'display: none;'
    }
  } else if (effectiveDisplay === 'all' || effectiveDisplay === 'simple') {
    style = `${common} background: color-mix(in oklab, var(--warning-subtle) 58%, transparent); color: inherit; text-decoration: none;`
  } else {
    style = `${common} background: transparent; color: inherit; text-decoration: none;`
  }
  return {
    className: [
      'inqtrix-review-overlay',
      `inqtrix-review-overlay-${kind}`,
      ...(active ? ['inqtrix-review-overlay-active'] : []),
    ].join(' '),
    display: effectiveDisplay,
    style,
  }
}

function collaborationReviewDecorations(
  doc: ProseMirrorNode,
  state: CollaborationReviewOverlayState,
): DecorationSet {
  if (!state.enabled) return DecorationSet.empty
  const decorations: Decoration[] = []
  doc.descendants((node, position) => {
    if (!node.isText || node.nodeSize === 0) return true
    for (const mark of node.marks) {
      if (!SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind)) continue
      const suggestionId = mark.attrs.suggestionId
      if (typeof suggestionId !== 'string' || !suggestionId) continue
      const kind = mark.type.name as CollaborationReviewMarkKind
      const active = state.selectedSuggestionIds.has(suggestionId)
      const visible = state.visibleSuggestionIds?.has(suggestionId) ?? true
      const presentation = collaborationReviewMarkPresentation(
        kind,
        state.display,
        active,
        visible,
      )
      decorations.push(Decoration.inline(position, position + node.nodeSize, {
        class: presentation.className,
        'data-review-active': active ? 'true' : 'false',
        'data-review-display': presentation.display,
        'data-review-overlay': 'change',
        'data-review-suggestion-id': suggestionId,
        style: presentation.style,
      }))
    }
    return true
  })
  return DecorationSet.create(doc, decorations)
}

function withCollaborationReviewDecorations(
  doc: ProseMirrorNode,
  state: CollaborationReviewOverlayState,
): CollaborationReviewOverlayState {
  return {
    ...state,
    decorations: collaborationReviewDecorations(doc, state),
  }
}

function enforceCaretOnlyPresence(root: HTMLElement): void {
  for (const caret of root.querySelectorAll<HTMLElement>('.collaboration-carets__caret')) {
    if (caret.dataset.collaborationCaret === 'text') continue
    const label = caret.querySelector<HTMLElement>('.collaboration-carets__label')
    const color = caret.style.borderColor || label?.style.backgroundColor || 'var(--brand)'
    caret.dataset.collaborationCaret = 'text'
    caret.classList.add('inqtrix-collaboration-caret')
    caret.style.setProperty('border-left', `2px solid ${color}`)
    caret.style.setProperty('border-right', '0')
    caret.style.setProperty('height', '1.2em')
    caret.style.setProperty('margin-left', '-1px')
    caret.style.setProperty('margin-right', '-1px')
    caret.style.setProperty('pointer-events', 'none')
    caret.style.setProperty('position', 'relative')
    if (label) {
      label.classList.add(
        'inqtrix-collaboration-caret-label',
        't-meta-sm',
        'whitespace-nowrap',
      )
      label.style.setProperty('background-color', color)
      label.style.setProperty('color', collaborationCaretLabelColor(color))
      label.style.setProperty('left', '-1px')
      label.style.setProperty('padding', '0.125rem 0.25rem')
      label.style.setProperty('position', 'absolute')
      label.style.setProperty('top', '-1.55rem')
      label.style.setProperty('user-select', 'none')
    }
  }
  for (const selection of root.querySelectorAll<HTMLElement>('.collaboration-carets__selection')) {
    if (
      selection.dataset.collaborationSelection === 'transparent'
      && selection.style.getPropertyValue('background-color') === 'transparent'
    ) continue
    selection.dataset.collaborationSelection = 'transparent'
    selection.classList.add('inqtrix-collaboration-selection')
    selection.style.setProperty('background-color', 'transparent', 'important')
    selection.style.setProperty('box-shadow', 'none', 'important')
  }
}

function emitSuggestionTransformError(view: EditorView, error: unknown): void {
  const detail = error instanceof Error
    ? error.message
    : 'This edit cannot be represented as a suggestion.'
  view.dom.dispatchEvent(new CustomEvent<string>(collaborationSuggestionErrorEvent, { detail }))
}

function emitSuggestionTransformSuccess(view: EditorView): void {
  view.dom.dispatchEvent(new CustomEvent<null>(collaborationSuggestionErrorEvent, { detail: null }))
  view.dom.dispatchEvent(new CustomEvent<null>(collaborationSuggestionCollisionEvent, { detail: null }))
}

function emitSuggestionCollision(
  view: EditorView,
  collision: CollaborationSuggestionCollision,
): void {
  view.dom.dispatchEvent(new CustomEvent<CollaborationSuggestionCollision>(
    collaborationSuggestionCollisionEvent,
    { detail: collision },
  ))
}

function clearSuggestionCollision(view: EditorView): void {
  view.dom.dispatchEvent(new CustomEvent<null>(collaborationSuggestionCollisionEvent, { detail: null }))
}

/** Find an existing suggestion by another author in every pre-step range the
 * transaction will mutate. Insertions inspect active boundary marks as well as
 * replaced ranges, so the detector runs before the Yjs dispatch. */
export function detectForeignSuggestionCollision(
  transaction: Transaction,
  authorId: string,
): CollaborationSuggestionCollision | null {
  for (let index = 0; index < transaction.steps.length; index += 1) {
    const document = transaction.docs[index]
    const step = transaction.steps[index]
    if (!document || !step) continue
    let collision: CollaborationSuggestionCollision | null = null
    step.getMap().forEach((oldStart, oldEnd) => {
      if (collision) return
      collision = foreignSuggestionInRange(document, oldStart, oldEnd, authorId)
    })
    if (collision) return collision
  }
  return null
}

function foreignSuggestionInRange(
  document: ProseMirrorNode,
  from: number,
  to: number,
  authorId: string,
): CollaborationSuggestionCollision | null {
  const marks: Mark[] = []
  const maxPosition = document.content.size
  const safeFrom = Math.max(0, Math.min(from, maxPosition))
  const safeTo = Math.max(safeFrom, Math.min(to, maxPosition))
  if (safeFrom < safeTo) {
    document.nodesBetween(safeFrom, safeTo, (node) => {
      marks.push(...node.marks)
      return true
    })
  } else {
    const position = document.resolve(safeFrom)
    marks.push(...position.marks())
    for (let depth = 0; depth <= position.depth; depth += 1) {
      marks.push(...position.node(depth).marks)
    }
  }
  for (const mark of marks) {
    if (!SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind)) continue
    const markAuthorId = mark.attrs.authorId
    const patchId = mark.attrs.patchId
    const suggestionId = mark.attrs.suggestionId
    if (
      typeof markAuthorId === 'string'
      && markAuthorId !== authorId
      && typeof patchId === 'string'
      && patchId
      && typeof suggestionId === 'string'
      && suggestionId
    ) {
      return { patchId, suggestionId }
    }
  }
  return null
}

function collaborationDispatchTransaction(
  view: EditorView,
  baseDispatch: (transaction: Transaction) => void,
  transaction: Transaction,
  grouping: CollaborationSuggestionGroupingCoordinator,
): void {
  const state = collaborationReviewPluginKey.getState(view.state)
    ?? DEFAULT_COLLABORATION_REVIEW_STATE
  grouping.observeContext({
    authorId: state.writeAuthorId,
    documentId: state.documentId,
    writeMode: state.writeMode,
  })
  if (!state.collaboration) {
    baseDispatch(transaction)
    return
  }
  if (isRemoteYjsTransaction(transaction)) {
    grouping.reset()
    baseDispatch(transaction)
    return
  }
  if (!transaction.docChanged) {
    if (transaction.selectionSet) grouping.reset()
    baseDispatch(transaction)
    return
  }
  if (state.writeMode === 'view') return
  if (!state.writeAuthorId) {
    clearSuggestionCollision(view)
    emitSuggestionTransformError(view, new Error('Suggestion mode requires a verified collaborator.'))
    return
  }
  const collision = detectForeignSuggestionCollision(transaction, state.writeAuthorId)
  if (collision) {
    grouping.reset()
    emitSuggestionCollision(view, collision)
    emitSuggestionTransformError(
      view,
      new Error('This edit overlaps a change from another collaborator. Review that change before editing it.'),
    )
    return
  }
  clearSuggestionCollision(view)
  if (state.writeMode === 'edit') {
    grouping.reset()
    baseDispatch(transaction)
    emitSuggestionTransformSuccess(view)
    return
  }
  try {
    const transformed = transformToInqtrixSuggestionTransaction(
      transaction,
      view.state,
      grouping.metadata(),
    )
    baseDispatch(transformed)
    emitSuggestionTransformSuccess(view)
  } catch (error) {
    grouping.reset()
    emitSuggestionTransformError(view, error)
  }
}

function collaborationReviewStateFromUpdate(
  value: CollaborationReviewOverlayState,
  update: CollaborationReviewOverlayUpdate,
): CollaborationReviewOverlayState {
  return {
    collaboration: update.collaboration ?? value.collaboration,
    decorations: value.decorations,
    display: update.display,
    documentId: update.documentId === undefined ? value.documentId : update.documentId,
    enabled: update.enabled,
    selectedSuggestionIds: new Set(update.selectedSuggestionIds ?? []),
    visibleSuggestionIds: update.visibleSuggestionIds === undefined
      ? null
      : new Set(update.visibleSuggestionIds),
    writeAuthorId: update.writeAuthorId === undefined
      ? value.writeAuthorId
      : update.writeAuthorId,
    writeMode: update.writeMode ?? value.writeMode,
  }
}

export const CollaborationReviewExtension = Extension.create<
  CollaborationReviewExtensionOptions,
  CollaborationReviewExtensionStorage
>({
  name: 'collaborationReview',

  addOptions() {
    return { initialPolicy: undefined }
  },

  addStorage() {
    return { grouping: new CollaborationSuggestionGroupingCoordinator() }
  },

  dispatchTransaction({ transaction, next }) {
    collaborationDispatchTransaction(this.editor.view, next, transaction, this.storage.grouping)
  },

  addProseMirrorPlugins() {
    const initialState = this.options.initialPolicy
      ? collaborationReviewStateFromUpdate(
          DEFAULT_COLLABORATION_REVIEW_STATE,
          this.options.initialPolicy,
        )
      : DEFAULT_COLLABORATION_REVIEW_STATE
    return [
      new Plugin<CollaborationReviewOverlayState>({
        key: collaborationReviewPluginKey,
        state: {
          init: (_config, state) => withCollaborationReviewDecorations(state.doc, initialState),
          apply(transaction, value) {
            const update = transaction.getMeta(collaborationReviewPluginKey) as (
              CollaborationReviewOverlayUpdate | undefined
            )
            if (!update && !transaction.docChanged) return value
            const nextState = update
              ? collaborationReviewStateFromUpdate(value, update)
              : value
            return withCollaborationReviewDecorations(transaction.doc, nextState)
          },
        },
        props: {
          decorations(state) {
            return collaborationReviewPluginKey.getState(state)?.decorations
              ?? DecorationSet.empty
          },
        },
        view(view) {
          enforceCaretOnlyPresence(view.dom)
          const presenceObserver = new MutationObserver(() => {
            enforceCaretOnlyPresence(view.dom)
          })
          presenceObserver.observe(view.dom, {
            attributeFilter: ['style'],
            attributes: true,
            childList: true,
            subtree: true,
          })
          return {
            destroy() {
              presenceObserver.disconnect()
            },
            update(nextView) {
              enforceCaretOnlyPresence(nextView.dom)
            },
          }
        },
      }),
    ]
  },
})

export function caretOnlySelectionRender(user: Record<string, unknown>): DecorationAttrs {
  void user
  return {
    nodeName: 'span',
    class: 'inqtrix-collaboration-selection',
    style: 'background-color: transparent; box-shadow: none;',
    'data-collaboration-selection': 'transparent',
  }
}

export function renderCollaborationCaret(user: Record<string, unknown>): HTMLElement {
  const name = typeof user.name === 'string' && user.name.trim()
    ? user.name.trim().slice(0, 80)
    : 'Collaborator'
  const color = typeof user.color === 'string' && /^#[0-9a-f]{6}$/i.test(user.color)
    ? user.color
    : 'var(--brand)'
  const caret = document.createElement('span')
  caret.className = 'inqtrix-collaboration-caret pointer-events-none relative inline-block align-text-bottom'
  caret.setAttribute('aria-hidden', 'true')
  caret.style.cssText = `border-left: 2px solid ${color}; height: 1.2em; margin-left: -1px; margin-right: -1px;`

  const label = document.createElement('span')
  label.className = 'inqtrix-collaboration-caret-label t-meta-sm absolute bottom-full left-0 z-20 whitespace-nowrap rounded-sm px-1 py-0.5 shadow-sm'
  label.style.backgroundColor = color
  label.style.color = collaborationCaretLabelColor(color)
  label.textContent = name
  caret.append(label)
  return caret
}

export function collaborationCaretLabelColor(color: string): '#111827' | '#ffffff' {
  if (!/^#[0-9a-f]{6}$/i.test(color)) return '#ffffff'
  const red = Number.parseInt(color.slice(1, 3), 16)
  const green = Number.parseInt(color.slice(3, 5), 16)
  const blue = Number.parseInt(color.slice(5, 7), 16)
  const luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
  return luminance > 0.58 ? '#111827' : '#ffffff'
}

export const collaborationCaretOptions = {
  render: renderCollaborationCaret,
  selectionRender: caretOnlySelectionRender,
}

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

export function createEditorExtensions(
  options: CommentDecorationOptions & {
    collaborationReview?: CollaborationReviewOverlayUpdate
  } = {},
): Extensions {
  return [
    ...createEditorSchemaExtensions({ resizableTables: true }),
    CollaborationReviewExtension.configure({ initialPolicy: options.collaborationReview }),
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

/**
 * Unit Separator (U+001F) that `@tiptap/markdown` emits inside populated table
 * cells while serializing (e.g. `| Hello |`). It is invisible, is never
 * part of the user's text, and breaks GFM table re-parsing when persisted — a
 * round-trip through it collapses the table. Stripped on every serialize and
 * defensively on load so legacy documents saved before this fix self-heal.
 */
/**
 * Serialize the editor document to markdown for persistence, removing the
 * `@tiptap/markdown` table-cell artifact. This is
 * the single save-path entry point: the live/source sync comparison must use it
 * too, otherwise a populated table always looks "changed" (raw `getMarkdown()`
 * still carries the separator) and triggers a needless reparse.
 */
export function serializeEditorMarkdown(editor: Editor): string {
  return sanitizeSerializedEditorMarkdown(editor.getMarkdown())
}

/** Serialize the canonical accepted view of a marked collaboration document. */
export function serializeEditorFinalProjectionMarkdown(editor: Editor): string {
  return serializeEditorJson(editor.getJSON(), 'final')
}

export function normalizeEditorMarkdownForTiptap(markdown: string) {
  return normalizeEditorMarkdown(markdown)
}
