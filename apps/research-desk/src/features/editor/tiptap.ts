import {
  INQTRIX_STRUCTURE_SUGGESTION_ATTR,
  createSecureUuid,
  createEditorSchemaExtensions,
  isStructureSuggestionData,
  isRemoteYjsTransaction,
  normalizeEditorMarkdown,
  sanitizeSerializedEditorMarkdown,
  serializeEditorJson,
  SUGGESTION_MARK_NAMES,
  suggestionDescriptors,
  transformToInqtrixSuggestionTransaction,
  type SuggestionMetadata,
} from '@inqtrix/editor-schema'
import type { HocuspocusProvider } from '@hocuspocus/provider'
import { Extension, getMarkRange, type Editor, type Extensions } from '@tiptap/core'
import { Placeholder } from '@tiptap/extensions'
import { type Mark, type Node as ProseMirrorNode } from '@tiptap/pm/model'
import { type EditorState, Plugin, PluginKey, TextSelection, type Transaction } from '@tiptap/pm/state'
import { Decoration, DecorationSet, type DecorationAttrs, type EditorView } from '@tiptap/pm/view'
import { ReactRenderer } from '@tiptap/react'
import {
  redo as yjsRedo,
  undo as yjsUndo,
  yCursorPlugin,
  yUndoPluginKey,
} from '@tiptap/y-tiptap'
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
  /** Localized singular label for a shared comment marker. */
  teamCommentLabel?: string
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
  writeMode?: 'comment' | 'edit' | 'suggest' | 'view'
}

type CollaborationReviewExtensionOptions = {
  initialPolicy?: CollaborationReviewOverlayUpdate
  onSuggestionUndo?: (patchId: string) => Promise<void>
}

type CollaborationReviewExtensionStorage = {
  grouping: CollaborationSuggestionGroupingCoordinator
  suggestionUndoHistory: CollaborationSuggestionUndoHistory
  writeMode: 'comment' | 'edit' | 'suggest' | 'view'
}

export type CommentDecorationItem = {
  from: number
  id: string
  kind: 'collect' | 'inline_edit' | 'evidence_review' | 'team'
  selected: boolean
  status: 'open' | 'resolved' | 'stale'
  to: number
}

type CommentDecorationState = {
  decorations: DecorationSet
}

export const commentDecorationPluginKey = new PluginKey<CommentDecorationState>('inqtrixCommentDecorations')

export type TeamCommentMarkerGroup = {
  count: number
  representativeId: string
  selected: boolean
  to: number
}

/**
 * A document can contain many threads on the same passage. Rendering one
 * widget for every thread turns a single line into a wall of dots, so markers
 * that end at the same document position share one count badge. The selected
 * thread becomes the representative, which keeps editor/inspector scrolling
 * and activation deterministic.
 */
export function groupTeamCommentMarkers(
  items: readonly CommentDecorationItem[],
): TeamCommentMarkerGroup[] {
  const groups = new Map<number, CommentDecorationItem[]>()
  for (const item of items) {
    if (item.kind !== 'team') continue
    const group = groups.get(item.to)
    if (group) group.push(item)
    else groups.set(item.to, [item])
  }
  return [...groups.entries()].map(([to, group]) => {
    const representative = group.find((item) => item.selected) ?? group[0]!
    return {
      count: group.length,
      representativeId: representative.id,
      selected: group.some((item) => item.selected),
      to,
    }
  })
}

function buildCommentDecorations(
  doc: ProseMirrorNode,
  items: CommentDecorationItem[],
  onClick: ((commentId: string) => void) | undefined,
  teamCommentLabel: string,
): DecorationSet {
  const maxPos = doc.content.size
  const normalizedItems = items.map((item) => {
    const from = Math.max(0, Math.min(item.from, maxPos))
    const to = Math.max(from, Math.min(item.to, maxPos))
    return { ...item, from, to }
  })
  const decorations = normalizedItems.flatMap((item) => {
    const attributes = {
      class: 'editor-comment-anchor',
      'data-editor-comment-anchor': item.id,
      'data-editor-comment-kind': item.kind,
      'data-editor-comment-status': item.status,
      ...(item.selected ? { 'data-editor-comment-selected': 'true' } : {}),
    }
    const itemDecorations: Decoration[] = []
    if (item.from < item.to) {
      itemDecorations.push(
        Decoration.inline(item.from, item.to, attributes, { commentId: item.id }),
      )
    }
    return itemDecorations
  })
  for (const group of groupTeamCommentMarkers(normalizedItems)) {
    decorations.push(
      Decoration.widget(group.to, () => {
        const marker = document.createElement('span')
        const visibleCount = group.count > 99 ? '99+' : String(group.count)
        const accessibleLabel = group.count > 1
          ? `${teamCommentLabel}: ${group.count}`
          : teamCommentLabel
        marker.className = 'editor-team-comment-marker'
        marker.dataset.editorCommentAnchor = group.representativeId
        marker.dataset.editorCommentCount = String(group.count)
        marker.dataset.editorCommentSelected = String(group.selected)
        marker.setAttribute('aria-label', accessibleLabel)
        marker.setAttribute('role', 'button')
        marker.setAttribute('title', accessibleLabel)
        marker.tabIndex = 0
        marker.textContent = group.count > 1 ? visibleCount : '●'
        if (onClick) {
          const activate = (event: Event) => {
            event.preventDefault()
            event.stopPropagation()
            onClick(group.representativeId)
          }
          marker.addEventListener('click', activate)
          marker.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') activate(event)
          })
        }
        return marker
      }, {
        commentId: group.representativeId,
        key: `team-comments-${group.to}-${group.count}-${group.representativeId}-${group.selected}`,
        side: 1,
      }),
    )
  }
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
    const teamCommentLabel = this.options.teamCommentLabel ?? 'Team comment'
    return [
      new Plugin<CommentDecorationState>({
        key: commentDecorationPluginKey,
        state: {
          init: () => ({ decorations: DecorationSet.empty }),
          apply(tr, value) {
            const meta = tr.getMeta(commentDecorationPluginKey) as { items: CommentDecorationItem[] } | undefined
            if (meta) {
              return {
                decorations: buildCommentDecorations(
                  tr.doc,
                  meta.items,
                  onClick,
                  teamCommentLabel,
                ),
              }
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
  writeMode: 'comment' | 'edit' | 'suggest' | 'view'
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
  writeMode: 'comment' | 'edit' | 'suggest' | 'view'
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
    this.createPatchId = options.createPatchId ?? createSecureUuid
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
 * Suggest-mode undo is a durable review decision, not a raw Yjs reversal.
 *
 * The stack is intentionally session-local: it contains only patches created
 * by this editor instance. Existing or reloaded suggestions remain reviewable
 * through the Changes inspector and can never be rejected accidentally by a
 * stale browser history entry.
 */
export class CollaborationSuggestionUndoHistory {
  private patchIds: string[] = []
  private pendingPatchId: string | null = null

  record(patchId: string): void {
    if (!patchId || this.patchIds.at(-1) === patchId) return
    this.patchIds.push(patchId)
  }

  current(openPatchIds: ReadonlySet<string>): string | null {
    if (this.pendingPatchId !== null) return null
    for (let index = this.patchIds.length - 1; index >= 0; index -= 1) {
      const patchId = this.patchIds[index]
      if (patchId && openPatchIds.has(patchId)) return patchId
    }
    return null
  }

  begin(patchId: string, openPatchIds: ReadonlySet<string>): boolean {
    if (this.current(openPatchIds) !== patchId) return false
    this.pendingPatchId = patchId
    return true
  }

  fail(patchId: string): void {
    if (this.pendingPatchId === patchId) this.pendingPatchId = null
  }

  reconcile(openPatchIds: ReadonlySet<string>): void {
    this.patchIds = this.patchIds.filter((patchId) => openPatchIds.has(patchId))
    if (
      this.pendingPatchId !== null
      && !openPatchIds.has(this.pendingPatchId)
    ) {
      this.pendingPatchId = null
    }
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
    const structure = node.attrs[INQTRIX_STRUCTURE_SUGGESTION_ATTR]
    if (isStructureSuggestionData(structure)) {
      const active = state.selectedSuggestionIds.has(structure.suggestionId)
      const visible = state.visibleSuggestionIds?.has(structure.suggestionId) ?? true
      const effectiveDisplay = visible ? state.display : 'final'
      decorations.push(Decoration.node(position, position + node.nodeSize, {
        class: cnReviewStructureClass(active, effectiveDisplay),
        'data-review-active': active ? 'true' : 'false',
        'data-review-display': effectiveDisplay,
        'data-review-overlay': 'structure',
        'data-review-structure-action': structure.action,
        'data-review-suggestion-id': structure.suggestionId,
      }))
    }
    if (!node.isText || node.nodeSize === 0) return true
    for (const mark of node.marks) {
      if (!SUGGESTION_MARK_NAMES.has(mark.type.name)) continue
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

function cnReviewStructureClass(
  active: boolean,
  display: CollaborationReviewDisplay,
): string {
  return [
    'inqtrix-review-structure',
    `inqtrix-review-structure-${display}`,
    ...(active ? ['inqtrix-review-structure-active'] : []),
  ].join(' ')
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

const COLLABORATION_CARET_LABEL_MAX_WIDTH_PX = 192
const COLLABORATION_CARET_LABEL_MAX_BOUNDARY_RATIO = 0.7
const COLLABORATION_CARET_LABEL_VIEWPORT_GUTTER_PX = 8

type CollaborationCaretLabelLayoutInput = {
  boundaryLeft: number
  boundaryRight: number
  caretLeft: number
  caretRight: number
  labelWidth: number
}

export type CollaborationCaretLabelLayout = {
  maxWidth: number
  shiftX: number
  side: 'left' | 'right'
}

function collaborationCaretLabelMaxWidth(availableWidth: number): number {
  return Math.min(
    COLLABORATION_CARET_LABEL_MAX_WIDTH_PX,
    Math.floor(Math.max(0, availableWidth) * COLLABORATION_CARET_LABEL_MAX_BOUNDARY_RATIO),
  )
}

export function collaborationCaretLabelLayout({
  boundaryLeft,
  boundaryRight,
  caretLeft,
  caretRight,
  labelWidth,
}: CollaborationCaretLabelLayoutInput): CollaborationCaretLabelLayout {
  const availableWidth = Math.max(0, boundaryRight - boundaryLeft)
  const maxWidth = collaborationCaretLabelMaxWidth(availableWidth)
  const renderedWidth = Math.min(Math.max(0, labelWidth), maxWidth)
  const leftStart = caretLeft
  const rightStart = caretRight - renderedWidth
  const overflow = (start: number) => (
    Math.max(0, boundaryLeft - start)
    + Math.max(0, start + renderedWidth - boundaryRight)
  )
  const side = overflow(leftStart) <= overflow(rightStart) ? 'left' : 'right'
  const preferredStart = side === 'left' ? leftStart : rightStart
  const maximumStart = Math.max(boundaryLeft, boundaryRight - renderedWidth)
  const boundedStart = Math.min(Math.max(preferredStart, boundaryLeft), maximumStart)
  return {
    maxWidth,
    shiftX: boundedStart - preferredStart,
    side,
  }
}

function setStylePropertyIfChanged(
  element: HTMLElement,
  property: string,
  value: string,
  priority = '',
): void {
  if (
    element.style.getPropertyValue(property) === value
    && element.style.getPropertyPriority(property) === priority
  ) return
  element.style.setProperty(property, value, priority)
}

function prepareCollaborationCaretLabel(
  caret: HTMLElement,
  label: HTMLElement,
  color: string,
): void {
  if (caret.dataset.collaborationCaret !== 'text') {
    caret.dataset.collaborationCaret = 'text'
  }
  caret.classList.add('inqtrix-collaboration-caret')
  setStylePropertyIfChanged(caret, 'border-left', `2px solid ${color}`)
  setStylePropertyIfChanged(caret, 'border-right', '0px')
  setStylePropertyIfChanged(caret, 'height', '1.2em')
  setStylePropertyIfChanged(caret, 'margin-left', '-1px')
  setStylePropertyIfChanged(caret, 'margin-right', '-1px')
  setStylePropertyIfChanged(caret, 'pointer-events', 'none')
  setStylePropertyIfChanged(caret, 'position', 'relative')

  label.classList.add(
    'inqtrix-collaboration-caret-label',
    't-meta-sm',
    'absolute',
    'bottom-full',
    'z-20',
    'whitespace-nowrap',
    'rounded-sm',
    'px-1',
    'py-0.5',
    'shadow-sm',
  )
  setStylePropertyIfChanged(label, 'background-color', color)
  setStylePropertyIfChanged(label, 'bottom', '100%')
  setStylePropertyIfChanged(label, 'box-sizing', 'border-box')
  setStylePropertyIfChanged(label, 'color', collaborationCaretLabelColor(color))
  setStylePropertyIfChanged(label, 'display', 'block')
  setStylePropertyIfChanged(label, 'max-width', 'min(12rem, 70vw)')
  setStylePropertyIfChanged(label, 'overflow', 'hidden')
  setStylePropertyIfChanged(label, 'position', 'absolute')
  setStylePropertyIfChanged(label, 'text-overflow', 'ellipsis')
  setStylePropertyIfChanged(label, 'top', 'auto')
  setStylePropertyIfChanged(label, 'user-select', 'none')
  setStylePropertyIfChanged(label, 'white-space', 'nowrap')
}

function positionCollaborationCaretLabel(
  root: HTMLElement,
  caret: HTMLElement,
  label: HTMLElement,
): void {
  const rootRect = root.getBoundingClientRect()
  const viewportWidth = root.ownerDocument.defaultView?.innerWidth ?? rootRect.right
  const boundaryLeft = Math.max(
    rootRect.left,
    COLLABORATION_CARET_LABEL_VIEWPORT_GUTTER_PX,
  )
  const boundaryRight = Math.min(
    rootRect.right,
    viewportWidth - COLLABORATION_CARET_LABEL_VIEWPORT_GUTTER_PX,
  )
  if (boundaryRight <= boundaryLeft) return

  const availableWidth = boundaryRight - boundaryLeft
  setStylePropertyIfChanged(
    label,
    'max-width',
    `${collaborationCaretLabelMaxWidth(availableWidth)}px`,
  )
  const caretRect = caret.getBoundingClientRect()
  const labelRect = label.getBoundingClientRect()
  const layout = collaborationCaretLabelLayout({
    boundaryLeft,
    boundaryRight,
    caretLeft: caretRect.left,
    caretRight: caretRect.right,
    labelWidth: labelRect.width,
  })
  if (label.dataset.collaborationLabelSide !== layout.side) {
    label.dataset.collaborationLabelSide = layout.side
  }
  if (layout.side === 'right') {
    setStylePropertyIfChanged(label, 'left', 'auto')
    setStylePropertyIfChanged(label, 'right', '0px')
  } else {
    setStylePropertyIfChanged(label, 'left', '0px')
    setStylePropertyIfChanged(label, 'right', 'auto')
  }
  const currentShiftX = Number.parseFloat(
    label.dataset.collaborationLabelShiftX ?? '0',
  ) || 0
  const positionedRect = label.getBoundingClientRect()
  const unshiftedLeft = positionedRect.left - currentShiftX
  const unshiftedRight = positionedRect.right - currentShiftX
  let exactShiftX = 0
  if (unshiftedLeft < boundaryLeft) exactShiftX = boundaryLeft - unshiftedLeft
  if (unshiftedRight + exactShiftX > boundaryRight) {
    exactShiftX += boundaryRight - (unshiftedRight + exactShiftX)
  }
  const shiftX = Math.round(exactShiftX * 1_000) / 1_000
  const serializedShiftX = String(shiftX)
  if (label.dataset.collaborationLabelShiftX !== serializedShiftX) {
    label.dataset.collaborationLabelShiftX = serializedShiftX
  }
  setStylePropertyIfChanged(
    label,
    'transform',
    shiftX === 0 ? 'none' : `translateX(${shiftX}px)`,
  )
}

function enforceCaretOnlyPresence(root: HTMLElement): void {
  for (const caret of root.querySelectorAll<HTMLElement>(
    '.collaboration-carets__caret, .inqtrix-collaboration-caret',
  )) {
    const label = caret.querySelector<HTMLElement>(
      '.collaboration-carets__label, .inqtrix-collaboration-caret-label',
    )
    const color = caret.style.borderColor || label?.style.backgroundColor || 'var(--brand)'
    if (label) {
      prepareCollaborationCaretLabel(caret, label, color)
      positionCollaborationCaretLabel(root, caret, label)
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
  const structures: Array<{
    authorId: string
    patchId: string
    suggestionId: string
  }> = []
  const inspectStructure = (node: ProseMirrorNode) => {
    const structure = node.attrs[INQTRIX_STRUCTURE_SUGGESTION_ATTR]
    if (isStructureSuggestionData(structure)) structures.push(structure)
  }
  const maxPosition = document.content.size
  const safeFrom = Math.max(0, Math.min(from, maxPosition))
  const safeTo = Math.max(safeFrom, Math.min(to, maxPosition))
  if (safeFrom < safeTo) {
    document.nodesBetween(safeFrom, safeTo, (node) => {
      marks.push(...node.marks)
      inspectStructure(node)
      return true
    })
  } else {
    const position = document.resolve(safeFrom)
    marks.push(...position.marks())
    for (let depth = 0; depth <= position.depth; depth += 1) {
      const node = position.node(depth)
      marks.push(...node.marks)
      inspectStructure(node)
    }
  }
  for (const mark of marks) {
    if (!SUGGESTION_MARK_NAMES.has(mark.type.name)) continue
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
  const structure = structures.find((item) => item.authorId !== authorId)
  if (structure) {
    return {
      patchId: structure.patchId,
      suggestionId: structure.suggestionId,
    }
  }
  return null
}

function collaborationDispatchTransaction(
  view: EditorView,
  baseDispatch: (transaction: Transaction) => void,
  transaction: Transaction,
  storage: CollaborationReviewExtensionStorage,
): void {
  const { grouping, suggestionUndoHistory } = storage
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
    const openPatchIds = openSuggestionPatchIds(view.state.doc)
    if (openPatchIds) suggestionUndoHistory.reconcile(openPatchIds)
    return
  }
  if (!transaction.docChanged) {
    if (transaction.selectionSet) grouping.reset()
    baseDispatch(transaction)
    return
  }
  if (state.writeMode === 'view' || state.writeMode === 'comment') {
    baseDispatch(transaction)
    return
  }
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
    const metadata = grouping.metadata()
    const transformed = transformToInqtrixSuggestionTransaction(
      transaction,
      view.state,
      metadata,
    )
    // A suggestion is undone through the durable patch-decision endpoint.
    // Keeping it out of Yjs history prevents Cmd/Ctrl+Z from first mutating
    // local state into a form the collaboration policy must reject.
    transformed.setMeta('addToHistory', false)
    baseDispatch(transformed)
    if (openSuggestionPatchIds(view.state.doc)?.has(metadata.patchId)) {
      suggestionUndoHistory.record(metadata.patchId)
    }
    emitSuggestionTransformSuccess(view)
  } catch (error) {
    grouping.reset()
    emitSuggestionTransformError(view, error)
  }
}

function openSuggestionPatchIds(document: ProseMirrorNode): Set<string> | null {
  try {
    return new Set(suggestionDescriptors(document).map((item) => item.patchId))
  } catch {
    return null
  }
}

export function collaborationReviewAllowsTransaction({
  collaboration,
  docChanged,
  remote,
  writeMode,
}: {
  collaboration: boolean
  docChanged: boolean
  remote: boolean
  writeMode: 'comment' | 'edit' | 'suggest' | 'view'
}): boolean {
  return (
    !collaboration
    || !docChanged
    || remote
    || (writeMode !== 'view' && writeMode !== 'comment')
  )
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
    return {
      initialPolicy: undefined,
      onSuggestionUndo: undefined,
    }
  },

  addStorage() {
    return {
      grouping: new CollaborationSuggestionGroupingCoordinator(),
      suggestionUndoHistory: new CollaborationSuggestionUndoHistory(),
      writeMode: this.options.initialPolicy?.writeMode ?? 'edit',
    }
  },

  dispatchTransaction({ transaction, next }) {
    collaborationDispatchTransaction(this.editor.view, next, transaction, this.storage)
  },

  addCommands() {
    if (!this.options.onSuggestionUndo) return {}
    return {
      undo:
        () =>
        ({ dispatch, state, tr }) => {
          const policy = collaborationReviewPluginKey.getState(state)
            ?? DEFAULT_COLLABORATION_REVIEW_STATE
          if (policy.collaboration && policy.writeMode === 'suggest') {
            const openPatchIds = openSuggestionPatchIds(state.doc) ?? new Set<string>()
            const patchId = this.storage.suggestionUndoHistory.current(openPatchIds)
            if (!dispatch) return patchId !== null
            tr.setMeta('preventDispatch', true)
            // Consume the native contenteditable shortcut even when the
            // semantic history is empty; otherwise the browser can bypass the
            // collaboration command path.
            if (
              !patchId
              || !this.storage.suggestionUndoHistory.begin(patchId, openPatchIds)
            ) return true
            emitSuggestionTransformSuccess(this.editor.view)
            let request: Promise<void>
            try {
              request = this.options.onSuggestionUndo!(patchId)
            } catch (error) {
              this.storage.suggestionUndoHistory.fail(patchId)
              emitSuggestionTransformError(this.editor.view, error)
              return true
            }
            void request.catch((error: unknown) => {
              this.storage.suggestionUndoHistory.fail(patchId)
              if (!this.editor.isDestroyed) {
                emitSuggestionTransformError(this.editor.view, error)
              }
            })
            return true
          }

          const undoManager = yUndoPluginKey.getState(state)?.undoManager
          if (!undoManager || undoManager.undoStack.length === 0) return false
          if (!dispatch) return true
          tr.setMeta('preventDispatch', true)
          return yjsUndo(state) ?? false
        },
      redo:
        () =>
        ({ dispatch, state, tr }) => {
          const policy = collaborationReviewPluginKey.getState(state)
            ?? DEFAULT_COLLABORATION_REVIEW_STATE
          if (policy.collaboration && policy.writeMode === 'suggest') {
            if (!dispatch) return false
            tr.setMeta('preventDispatch', true)
            return true
          }

          const undoManager = yUndoPluginKey.getState(state)?.undoManager
          if (!undoManager || undoManager.redoStack.length === 0) return false
          if (!dispatch) return true
          tr.setMeta('preventDispatch', true)
          return yjsRedo(state) ?? false
        },
    }
  },

  addProseMirrorPlugins() {
    const storage = this.storage
    const initialState = this.options.initialPolicy
      ? collaborationReviewStateFromUpdate(
          DEFAULT_COLLABORATION_REVIEW_STATE,
          this.options.initialPolicy,
        )
      : DEFAULT_COLLABORATION_REVIEW_STATE
    storage.writeMode = initialState.writeMode
    return [
      new Plugin<CollaborationReviewOverlayState>({
        key: collaborationReviewPluginKey,
        filterTransaction(transaction, state) {
          const policy = collaborationReviewPluginKey.getState(state)
            ?? DEFAULT_COLLABORATION_REVIEW_STATE
          return collaborationReviewAllowsTransaction({
            collaboration: policy.collaboration,
            docChanged: transaction.docChanged,
            remote: isRemoteYjsTransaction(transaction),
            writeMode: policy.writeMode,
          })
        },
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
            storage.writeMode = nextState.writeMode
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
          let animationFrame: number | null = null
          const schedulePresenceLayout = () => {
            if (animationFrame !== null) return
            animationFrame = requestAnimationFrame(() => {
              animationFrame = null
              enforceCaretOnlyPresence(view.dom)
            })
          }
          const presenceObserver = new MutationObserver(schedulePresenceLayout)
          presenceObserver.observe(view.dom, {
            attributeFilter: ['style'],
            attributes: true,
            childList: true,
            subtree: true,
          })
          const presenceResizeObserver = new ResizeObserver(schedulePresenceLayout)
          presenceResizeObserver.observe(view.dom)
          window.addEventListener('resize', schedulePresenceLayout, { passive: true })
          schedulePresenceLayout()
          return {
            destroy() {
              if (animationFrame !== null) cancelAnimationFrame(animationFrame)
              presenceObserver.disconnect()
              presenceResizeObserver.disconnect()
              window.removeEventListener('resize', schedulePresenceLayout)
            },
            update(nextView) {
              enforceCaretOnlyPresence(nextView.dom)
              schedulePresenceLayout()
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

  const label = document.createElement('span')
  label.className = 'inqtrix-collaboration-caret-label'
  label.textContent = name
  caret.append(label)
  prepareCollaborationCaretLabel(caret, label, color)
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

type InqtrixCollaborationCaretOptions = {
  provider: HocuspocusProvider | null
  user: CollaborationPresenceUser | null
}

export function shouldRenderCollaborationAwarenessState(
  localUserId: string,
  localAwarenessClientId: number,
  candidateClientId: number,
  state: unknown,
): boolean {
  if (candidateClientId === localAwarenessClientId) return false
  if (!state || typeof state !== 'object') return true
  const user = Reflect.get(state, 'user')
  if (!user || typeof user !== 'object') return true
  return Reflect.get(user, 'id') !== localUserId
}

/**
 * The provider deliberately uses a transport Y.Doc while the editor binds a
 * separate authoritative Y.Doc. yCursorPlugin's default filter compares an
 * awareness client id with the editor document client id, so it mistakes the
 * local transport state for a remote collaborator. Besides showing a duplicate
 * self-caret, that widget can occupy position zero in an empty code block and
 * swallow the first typed character. Filter against the provider's awareness
 * client id and the stable user identity instead.
 */
export const InqtrixCollaborationCaret = Extension.create<InqtrixCollaborationCaretOptions>({
  name: 'collaborationCaret',

  addOptions() {
    return {
      provider: null,
      user: null,
    }
  },

  addProseMirrorPlugins() {
    const { provider, user } = this.options
    const awareness = provider?.awareness
    if (!awareness || !user) return []
    awareness.setLocalStateField('user', user)
    const localAwarenessClientId = awareness.clientID
    return [
      yCursorPlugin(awareness, {
        awarenessStateFilter: (_editorClientId, candidateClientId, state) => (
          shouldRenderCollaborationAwarenessState(
            user.id,
            localAwarenessClientId,
            Number(candidateClientId),
            state,
          )
        ),
        cursorBuilder: collaborationCaretOptions.render,
        selectionBuilder: collaborationCaretOptions.selectionRender,
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

export function createEditorExtensions(
  options: CommentDecorationOptions & {
    collaborationReview?: CollaborationReviewOverlayUpdate
    onCollaborationSuggestionUndo?: (patchId: string) => Promise<void>
  } = {},
): Extensions {
  return [
    ...createEditorSchemaExtensions({ resizableTables: true }),
    CollaborationReviewExtension.configure({
      initialPolicy: options.collaborationReview,
      onSuggestionUndo: options.onCollaborationSuggestionUndo,
    }),
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
