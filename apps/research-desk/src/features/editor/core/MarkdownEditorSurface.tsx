/**
 * The markdown editing surface: owns the Tiptap editor instance and switches
 * between the live WYSIWYG view (with bubble menu, block handle and table
 * controls), the raw markdown source editor and the document diff view.
 * Extracted from `EditorWorkspace` so other canvases (e.g. the Agent Desk) can
 * reuse the surface without the editor shell.
 */
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { EditorContent, useEditor, type Editor } from '@tiptap/react'
import { BubbleMenu } from '@tiptap/react/menus'
import { useReducedMotion } from 'motion/react'
import { X } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { resizeTextareaToRows } from '@/features/composer/textareaAutosize'
import type {
  EditorCommentKind,
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorSuggestionRecord,
  ProjectState,
} from '@/features/project/types'
import {
  TextImproveButton,
  TextImproveFloatingLayer,
  useTextImprovement,
  type TextImprovementApiOptions,
} from '@/features/textImprove'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { commentDecorationPluginKey, createEditorExtensions, normalizeEditorMarkdownForTiptap, serializeEditorMarkdown, suggestionDecorationPluginKey } from '../tiptap'
import { BlockHandle } from '../BlockHandle'
import { TableControls } from '../TableControls'
import { SelectionToolbar } from '../SelectionToolbar'
import { MarkdownSourceEditor } from '../MarkdownSourceEditor'
import { DocumentDiffView } from '../DocumentDiffView'
import { suggestionDiffPlan } from '../suggestionDiff'
import {
  blockInsertionPositionForRange,
  blockWidgetPositionForRange,
  clampAnchor,
  createCommentFromSelection,
  resolveMaterializedAnchor,
  resolveAnchorRange,
  shouldParsePastedMarkdown,
} from '../anchoring'
import { COMMENT_KIND_ORDER, commentKindMeta } from '../commentKinds'
import type { EditorCopy } from '../editorCopy'
import { escapeCssIdentifier, resetExternalContentFlag } from '../editorDom'

export type MarkdownEditorSurfaceProps = {
  comments: EditorCommentThreadRecord[]
  copy: EditorCopy
  diffAnchorMarkdown: string | null
  document: EditorDocumentRecord
  /** Hosted inside an already-padded, already-scrolling container (the
   * agent canvas): skip the surface's own ScrollArea, min-height and
   * `px-10 py-8` so toggling read <-> edit never shifts the layout.
   * The Editor desk keeps the default standalone geometry. */
  embedded?: boolean
  isDiffVisible: boolean
  mode: ProjectState['editorUi']['viewMode']
  onChange: (contentMarkdown: string) => void
  onCreateComment: (comment: EditorCommentThreadRecord) => void
  onEditorReady: (editor: Editor | null) => void
  onAcceptSuggestion: (suggestion: EditorSuggestionRecord) => void
  onEditSuggestion: (suggestionId: string, proposedText: string) => void
  onMarkSuggestionStale: (suggestionId: string) => void
  onRefineSuggestion: (suggestionId: string, instruction: string) => Promise<void>
  onRejectSuggestion: (suggestionId: string) => void
  onSelectComment: (commentId: string) => void
  onStopSuggestion: (suggestionId: string) => void
  runningSuggestionIds: readonly string[]
  selectedCommentId: string | null
  suggestionErrors: Record<string, string>
  suggestions: EditorSuggestionRecord[]
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}

export function MarkdownEditorSurface({
  comments,
  copy,
  diffAnchorMarkdown,
  document,
  embedded = false,
  isDiffVisible,
  mode,
  onChange,
  onCreateComment,
  onEditorReady,
  onAcceptSuggestion,
  onEditSuggestion,
  onRejectSuggestion,
  onMarkSuggestionStale,
  onRefineSuggestion,
  onSelectComment,
  onStopSuggestion,
  runningSuggestionIds,
  selectedCommentId,
  suggestionErrors,
  suggestions,
  textImprovement,
}: MarkdownEditorSurfaceProps) {
  const documentIdRef = useRef(document.id)
  const editorInstanceRef = useRef<Editor | null>(null)
  const isApplyingExternalContentRef = useRef(false)
  const onAcceptSuggestionRef = useRef(onAcceptSuggestion)
  const onEditSuggestionRef = useRef(onEditSuggestion)
  const onMarkSuggestionStaleRef = useRef(onMarkSuggestionStale)
  const onRefineSuggestionRef = useRef(onRefineSuggestion)
  const onRejectSuggestionRef = useRef(onRejectSuggestion)
  const onSelectCommentRef = useRef(onSelectComment)
  const onStopSuggestionRef = useRef(onStopSuggestion)
  const suggestionsRef = useRef(suggestions)
  const previousModeRef = useRef(mode)
  const commentsSignature = comments.map((comment) => `${comment.id}:${comment.status}:${comment.kind}:${comment.anchor.from}:${comment.anchor.to}`).join('|')
  const suggestionsSignature = suggestions.map((suggestion) =>
    `${suggestion.id}:${suggestion.revision ?? 1}:${suggestion.editPosition ?? 'replace'}:${suggestion.anchorText ?? ''}:${suggestion.originalText.length}:${suggestion.proposedText}`).join('|')
  const suggestionUiSignature = suggestions.map((suggestion) =>
    `${suggestion.id}:${runningSuggestionIds.includes(suggestion.id) ? 'running' : 'idle'}:${suggestionErrors[suggestion.id] ?? ''}`).join('|')
  const tiptapContentMarkdown = normalizeEditorMarkdownForTiptap(document.contentMarkdown)

  useEffect(() => {
    suggestionsRef.current = suggestions
  }, [suggestions])

  useEffect(() => {
    onAcceptSuggestionRef.current = onAcceptSuggestion
    onEditSuggestionRef.current = onEditSuggestion
    onMarkSuggestionStaleRef.current = onMarkSuggestionStale
    onRefineSuggestionRef.current = onRefineSuggestion
    onRejectSuggestionRef.current = onRejectSuggestion
    onSelectCommentRef.current = onSelectComment
    onStopSuggestionRef.current = onStopSuggestion
  }, [onAcceptSuggestion, onEditSuggestion, onMarkSuggestionStale, onRefineSuggestion, onRejectSuggestion, onSelectComment, onStopSuggestion])

  const editor = useEditor({
    content: tiptapContentMarkdown,
    contentType: 'markdown',
    editable: mode === 'live',
    editorProps: {
      attributes: {
        class: 'editor-prose min-h-full focus:outline-none',
      },
      handlePaste: (_view, event) => {
        const pastedMarkdown = event.clipboardData?.getData('text/plain') ?? ''
        const currentEditor = editorInstanceRef.current
        if (!currentEditor?.isEditable || !shouldParsePastedMarkdown(pastedMarkdown)) return false
        event.preventDefault()
        currentEditor.commands.insertContent(normalizeEditorMarkdownForTiptap(pastedMarkdown), {
          contentType: 'markdown',
        })
        return true
      },
    },
    extensions: createEditorExtensions({
      syntaxMarkerRemoveLabel: copy.removeFormatting,
      placeholderEmpty: copy.placeholderEmpty,
      placeholderHeading: copy.placeholderHeading,
      slash: {
        labels: {
          title: copy.slashTitle,
          empty: copy.slashEmpty,
          navHint: copy.slashNav,
          selectHint: copy.slashSelect,
          closeHint: copy.slashClose,
          groupStyle: copy.slashGroupStyle,
          groupInsert: copy.slashGroupInsert,
          text: copy.slashText,
          heading1: copy.slashHeading1,
          heading2: copy.slashHeading2,
          heading3: copy.slashHeading3,
          bulletList: copy.slashBulletList,
          orderedList: copy.slashOrderedList,
          taskList: copy.slashTaskList,
          blockquote: copy.slashBlockquote,
          codeBlock: copy.slashCodeBlock,
          table: copy.slashTable,
          divider: copy.slashDivider,
        },
      },
      onClick: (commentId) => onSelectCommentRef.current(commentId),
      onSuggestionAccept: (suggestionId) => {
        const suggestion = suggestionsRef.current.find((item) => item.id === suggestionId)
        if (suggestion) onAcceptSuggestionRef.current(suggestion)
      },
      onSuggestionReject: (suggestionId) => onRejectSuggestionRef.current(suggestionId),
      onSuggestionEdit: (suggestionId, proposedText) => onEditSuggestionRef.current(suggestionId, proposedText),
      onSuggestionRefine: (suggestionId, instruction) => {
        void onRefineSuggestionRef.current(suggestionId, instruction)
      },
      onSuggestionCancel: (suggestionId) => onStopSuggestionRef.current(suggestionId),
      onSuggestionSelect: (suggestionId) => {
        const suggestion = suggestionsRef.current.find((item) => item.id === suggestionId)
        if (suggestion?.origin.commentId) onSelectCommentRef.current(suggestion.origin.commentId)
      },
    }),
    immediatelyRender: false,
    onCreate: ({ editor: createdEditor }) => {
      editorInstanceRef.current = createdEditor
    },
    onDestroy: () => {
      editorInstanceRef.current = null
    },
    onUpdate: ({ editor: currentEditor }) => {
      if (isApplyingExternalContentRef.current || !currentEditor.isEditable) return
      onChange(serializeEditorMarkdown(currentEditor))
    },
  })

  useEffect(() => {
    onEditorReady(editor)
    return () => onEditorReady(null)
  }, [editor, onEditorReady])

  useEffect(() => {
    if (!editor || documentIdRef.current === document.id) return
    documentIdRef.current = document.id
    isApplyingExternalContentRef.current = true
    editor.commands.setContent(tiptapContentMarkdown, {
      contentType: 'markdown',
      emitUpdate: false,
    })
    resetExternalContentFlag(isApplyingExternalContentRef)
  }, [document.id, editor, tiptapContentMarkdown])

  useEffect(() => {
    if (!editor) return
    const previousMode = previousModeRef.current
    previousModeRef.current = mode
    if (mode !== 'live') return
    const shouldReparseMarkdown = previousMode === 'source'
    if (!shouldReparseMarkdown && serializeEditorMarkdown(editor) === tiptapContentMarkdown) return
    isApplyingExternalContentRef.current = true
    editor.commands.setContent(tiptapContentMarkdown, {
      contentType: 'markdown',
      emitUpdate: false,
    })
    resetExternalContentFlag(isApplyingExternalContentRef)
  }, [editor, mode, tiptapContentMarkdown])

  useEffect(() => {
    editor?.setEditable(mode === 'live')
  }, [editor, mode])

  useEffect(() => {
    if (!editor || mode !== 'live') return
    const items = comments
      .filter((comment) => comment.status !== 'resolved')
      .map((comment) => {
        const resolved = resolveMaterializedAnchor(editor, comment.anchor)
        if (!resolved) return null
        return {
          from: resolved.range.from,
          id: comment.id,
          kind: comment.kind,
          selected: selectedCommentId === comment.id,
          status: comment.status,
          to: resolved.range.to,
        }
      })
      .filter((item): item is NonNullable<typeof item> => Boolean(item))
      .filter((item) => item.from < item.to)
    isApplyingExternalContentRef.current = true
    editor.view.dispatch(editor.state.tr.setMeta(commentDecorationPluginKey, { items }))
    resetExternalContentFlag(isApplyingExternalContentRef)
    // Re-materialise comment anchors when the CONTENT changes (external
    // rebase/hydrate replaces the body) or the comment set changes. Keyed on
    // contentMarkdown, not revision: revision now tracks the server base and
    // no longer moves per local edit, and the decorations already self-map
    // through transactions for in-place typing (see tiptap commentDecoration
    // plugin), so content is the correct, direct trigger.
  }, [commentsSignature, document.contentMarkdown, editor, mode, selectedCommentId])

  useEffect(() => {
    if (!editor || mode !== 'live') return
    const staleSuggestionIds: string[] = []
    const items = suggestions.flatMap((suggestion) => {
      const target = resolveSuggestionDecorationTarget(editor, suggestion)
      if (!target) {
        staleSuggestionIds.push(suggestion.id)
        return []
      }
      const plan = suggestionDiffPlan(suggestion.originalText, suggestion.proposedText)
      return [{
        acceptLabel: copy.accept,
        active: selectedCommentId === suggestion.origin.commentId,
        display: plan.display,
        editLabel: copy.editSuggestion,
        error: suggestionErrors[suggestion.id],
        from: target.from,
        id: suggestion.id,
        isRunning: runningSuggestionIds.includes(suggestion.id),
        proposedLabel: copy.proposedText,
        proposedText: suggestion.proposedText,
        refineLabel: copy.refineSuggestion,
        refinementPlaceholder: copy.refinementPlaceholder,
        rejectLabel: copy.reject,
        revision: suggestion.revision ?? 1,
        revisionLabel: copy.revision,
        reviewSurface: plan.reviewSurface,
        saveLabel: copy.saveSuggestion,
        segments: plan.segments,
        cancelLabel: copy.cancelEdit,
        sendLabel: copy.sendRefinement,
        runningLabel: copy.refiningSuggestion,
        stopLabel: copy.stopRun,
        to: target.to,
        widgetAt: plan.display === 'block' ? target.widgetAt : undefined,
      }]
    })
    isApplyingExternalContentRef.current = true
    editor.view.dispatch(editor.state.tr.setMeta(suggestionDecorationPluginKey, { items }))
    resetExternalContentFlag(isApplyingExternalContentRef)
    for (const suggestionId of staleSuggestionIds) onMarkSuggestionStaleRef.current(suggestionId)
  }, [copy.accept, copy.cancelEdit, copy.editSuggestion, copy.proposedText, copy.refineSuggestion, copy.refinementPlaceholder, copy.reject, copy.revision, copy.saveSuggestion, copy.sendRefinement, copy.refiningSuggestion, copy.stopRun, document.revision, editor, mode, selectedCommentId, suggestionUiSignature, suggestionsSignature])

  useEffect(() => {
    if (!selectedCommentId) return
    const target = globalThis.document?.querySelector<HTMLElement>(
      `[data-editor-comment-anchor="${escapeCssIdentifier(selectedCommentId)}"]`,
    )
    target?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [selectedCommentId])

  if (mode === 'source') {
    return (
      <MarkdownSourceEditor
        labels={{
          addColumn: copy.addColumn,
          addRow: copy.addRow,
          closeTableEditor: copy.closeTableEditor,
          columnLabel: copy.columnLabel,
          deleteColumn: copy.deleteColumn,
          deleteRow: copy.deleteRow,
          editor: copy.sourceEditor,
          formatTables: copy.formatTables,
          insertOrEditTable: copy.insertOrEditTable,
          lineWrap: copy.sourceLineWrap,
          tableAlignmentCenter: copy.tableAlignmentCenter,
          tableAlignmentLeft: copy.tableAlignmentLeft,
          tableAlignmentRight: copy.tableAlignmentRight,
          tableColumn: copy.tableColumn,
          tableEditor: copy.tableEditor,
          tableLines: copy.tableLines,
          tableRows: copy.tableRows,
        }}
        onChange={onChange}
        value={document.contentMarkdown}
      />
    )
  }

  if (isDiffVisible) {
    return (
      <DocumentDiffView
        anchorMarkdown={diffAnchorMarkdown}
        copy={copy}
        currentMarkdown={document.contentMarkdown}
      />
    )
  }

  const liveBody = (
      <div
        className={
          embedded
            ? 'w-full'
            : 'min-h-[calc(100svh-var(--header-h)-10rem)] w-full px-10 py-8'
        }
      >
        {editor ? (
          <EditorBubbleMenu
            copy={copy}
            editor={editor}
            onCreateComment={(commentMarkdown, kind) => {
              const comment = createCommentFromSelection(editor, document.id, commentMarkdown, kind)
              if (!comment) return
              onCreateComment(comment)
            }}
            textImprovement={textImprovement}
          />
        ) : null}
        {editor && mode === 'live' ? (
          <BlockHandle
            editor={editor}
            labels={{
              ariaLabel: copy.blockHandleAria,
              turnInto: copy.blockTurnInto,
              duplicate: copy.blockDuplicate,
              deleteBlock: copy.blockDelete,
              moveUp: copy.blockMoveUp,
              moveDown: copy.blockMoveDown,
              text: copy.slashText,
              heading1: copy.slashHeading1,
              heading2: copy.slashHeading2,
              heading3: copy.slashHeading3,
              bulletList: copy.slashBulletList,
              orderedList: copy.slashOrderedList,
              taskList: copy.slashTaskList,
              blockquote: copy.slashBlockquote,
              codeBlock: copy.slashCodeBlock,
            }}
          />
        ) : null}
        {editor && mode === 'live' ? (
          <TableControls
            editor={editor}
            labels={{
              columnOptions: copy.tableColumnOptions,
              rowOptions: copy.tableRowOptions,
              addColumn: copy.addColumn,
              addRow: copy.addRow,
              colInsertLeft: copy.tableColInsertLeft,
              colInsertRight: copy.tableColInsertRight,
              colMoveLeft: copy.tableColMoveLeft,
              colMoveRight: copy.tableColMoveRight,
              sortAsc: copy.tableSortAsc,
              sortDesc: copy.tableSortDesc,
              colDuplicate: copy.tableColDuplicate,
              colClear: copy.tableColClear,
              toggleHeaderRow: copy.tableToggleHeaderRow,
              colDelete: copy.deleteColumn,
              rowInsertAbove: copy.tableRowInsertAbove,
              rowInsertBelow: copy.tableRowInsertBelow,
              rowMoveUp: copy.tableRowMoveUp,
              rowMoveDown: copy.tableRowMoveDown,
              rowDuplicate: copy.tableRowDuplicate,
              rowDelete: copy.deleteRow,
            }}
          />
        ) : null}
        <EditorContent className="min-h-full" editor={editor} />
      </div>
  )
  // Embedded hosts own scrolling and padding — a nested ScrollArea here
  // is what made the report shift when entering edit mode.
  if (embedded) return liveBody
  return (
    <ScrollArea className="min-h-0 flex-1 bg-background">{liveBody}</ScrollArea>
  )
}

function resolveSuggestionDecorationTarget(
  editor: Editor,
  suggestion: EditorSuggestionRecord,
): { from: number; to: number; widgetAt: number } | null {
  const position = suggestion.editPosition ?? 'replace'
  if (position === 'append') {
    const end = editor.state.doc.content.size
    return { from: end, to: end, widgetAt: end }
  }
  const anchorText = (suggestion.anchorText ?? suggestion.originalText).trim()
  if (!anchorText) return null
  const range = resolveAnchorRange(editor, {
    hint: clampAnchor(suggestion.anchor, editor).from,
    quoteAfter: suggestion.anchor.quoteAfter,
    quoteBefore: suggestion.anchor.quoteBefore,
    text: anchorText,
  })
  if (!range) return null
  if (position === 'replace') {
    return { ...range, widgetAt: blockWidgetPositionForRange(editor, range) }
  }
  const at = blockInsertionPositionForRange(editor, range, position)
  return { from: at, to: at, widgetAt: at }
}

function EditorBubbleMenu({
  copy,
  editor,
  onCreateComment,
  textImprovement,
}: {
  copy: EditorCopy
  editor: Editor
  onCreateComment: (commentMarkdown: string, kind: EditorCommentKind) => void
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}) {
  const { locale, t } = useLocale()
  const reduceMotion = useReducedMotion()
  const [isCommenting, setIsCommenting] = useState(false)
  const [commentDraft, setCommentDraft] = useState('')
  const [commentKind, setCommentKind] = useState<EditorCommentKind>('collect')
  const [commentImproveError, setCommentImproveError] = useState<string | null>(null)
  const commentTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  // Keeps the bubble open while the "Turn into" dropdown is open (a transaction
  // could otherwise re-run shouldShow and hide it mid-interaction).
  const toolbarInteractingRef = useRef(false)
  const commentTextImprove = useTextImprovement({
    ...textImprovement,
    locale,
    messages: {
      requestFailed: (message) => `${t.textImprove.requestFailed}: ${message}`,
      sensitiveText: t.textImprove.sensitiveText,
      unavailable: t.textImprove.unavailable,
    },
  })

  useLayoutEffect(() => {
    if (!isCommenting) return
    resizeTextareaToRows(commentTextareaRef.current, 6)
  }, [commentDraft, isCommenting])

  function closeCommentComposer() {
    const collapseAt = editor.state.selection.to
    editor.commands.setTextSelection(collapseAt)
    editor.commands.blur()
    setCommentDraft('')
    setCommentKind('collect')
    setCommentImproveError(null)
    commentTextImprove.clearProposal()
    setIsCommenting(false)
  }

  function cancelComment() {
    closeCommentComposer()
  }

  function submitComment() {
    const value = commentDraft.trim()
    if (!value) return
    onCreateComment(value, commentKind)
    closeCommentComposer()
  }

  function handleCommentDraftChange(value: string) {
    setCommentDraft(value)
    setCommentImproveError(null)
    commentTextImprove.clearProposal()
  }

  async function improveCommentDraft() {
    setCommentImproveError(null)
    try {
      await commentTextImprove.improve('chat_input', commentDraft)
    } catch (error) {
      setCommentImproveError(messageFromUnknown(error))
    }
  }

  function acceptCommentImprovement(text: string) {
    handleCommentDraftChange(text)
    window.requestAnimationFrame(() => {
      commentTextareaRef.current?.focus()
      resizeTextareaToRows(commentTextareaRef.current, 6)
    })
  }

  return (
    <BubbleMenu
      editor={editor}
      appendTo={() => globalThis.document.body}
      options={{
        flip: { padding: { bottom: 132, left: 12, right: 12, top: 12 } },
        inline: true,
        offset: 8,
        placement: 'top-start',
        shift: { padding: { bottom: 132, left: 12, right: 12, top: 12 } },
        strategy: 'fixed',
      }}
      shouldShow={({ editor: currentEditor, state }) => {
        const { empty } = state.selection
        return currentEditor.isEditable && (!empty || toolbarInteractingRef.current)
      }}
    >
      <div className="z-50 flex min-w-0 items-center gap-1 rounded-lg border border-border bg-popover p-1 text-popover-foreground shadow-lg">
        {isCommenting ? (
          <form
            className="relative flex w-[26rem] max-w-[calc(100vw-5rem)] flex-col gap-2 p-1.5"
            onKeyDown={(event) => {
              if (event.key === 'Escape') {
                event.preventDefault()
                cancelComment()
              }
            }}
            onSubmit={(event) => {
              event.preventDefault()
              submitComment()
            }}
          >
            <button
              aria-label={copy.cancel}
              className="absolute right-2 top-2 z-10 inline-grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              onClick={cancelComment}
              type="button"
            >
              <X className="size-3.5" />
            </button>
            <TextImproveFloatingLayer
              labels={{
                accept: t.textImprove.accept,
                changes: t.textImprove.changes,
                noChanges: t.textImprove.noChanges,
                reject: t.textImprove.reject,
                title: t.textImprove.title,
                warnings: t.textImprove.warnings,
              }}
              onAccept={acceptCommentImprovement}
              onReject={commentTextImprove.clearProposal}
              proposal={commentTextImprove.proposal}
              reduceMotion={reduceMotion}
            />
            <Textarea
              autoFocus
              className="t-body min-h-16 resize-none border-border/70 bg-background/60 pr-16 focus-visible:ring-1 [scrollbar-width:thin]"
              onChange={(event) => handleCommentDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault()
                  submitComment()
                }
              }}
              placeholder={copy.inlineComment}
              ref={commentTextareaRef}
              value={commentDraft}
            />
            <TextImproveButton
              className="absolute right-9 top-2 z-10"
              disabled={!commentDraft.trim()}
              isLoading={commentTextImprove.isImproving}
              label={t.textImprove.improve}
              loadingLabel={t.textImprove.improving}
              onClick={() => void improveCommentDraft()}
              reduceMotion={reduceMotion}
            />
            {commentImproveError ? (
              <p className="t-meta-sm rounded-md border border-destructive/20 bg-destructive/5 px-2 py-1 text-destructive">
                {commentImproveError}
              </p>
            ) : null}
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1">
                {COMMENT_KIND_ORDER.map((kind) => {
                  const kindMeta = commentKindMeta(kind, copy)
                  const KindIcon = kindMeta.Icon
                  const active = commentKind === kind
                  return (
                    <button
                      aria-pressed={active}
                      className={cn(
                        'inline-flex h-6 shrink-0 items-center gap-1 rounded-full border px-2 t-meta-sm font-medium transition-colors',
                        active
                          ? cn(kindMeta.selectedBorderClass, kindMeta.selectedBgClass, kindMeta.accentText)
                          : 'border-border text-muted-foreground hover:text-foreground',
                      )}
                      key={kind}
                      onClick={() => setCommentKind(kind)}
                      type="button"
                    >
                      <KindIcon className="size-3" />
                      {kindMeta.label}
                    </button>
                  )
                })}
              </div>
              <Button disabled={!commentDraft.trim()} size="sm" type="submit">
                {copy.inlineCommentSubmit}
              </Button>
            </div>
          </form>
        ) : (
          <SelectionToolbar
            editor={editor}
            labels={{
              comment: copy.bubbleComment,
              turnInto: copy.blockTurnInto,
              bold: copy.bubbleBold,
              italic: copy.bubbleItalic,
              underline: copy.bubbleUnderline,
              strike: copy.bubbleStrike,
              code: copy.bubbleCode,
              highlight: copy.bubbleHighlight,
              text: copy.slashText,
              heading1: copy.slashHeading1,
              heading2: copy.slashHeading2,
              heading3: copy.slashHeading3,
              bulletList: copy.slashBulletList,
              orderedList: copy.slashOrderedList,
              taskList: copy.slashTaskList,
              blockquote: copy.slashBlockquote,
              codeBlock: copy.slashCodeBlock,
            }}
            onInteractingChange={(interacting) => {
              toolbarInteractingRef.current = interacting
            }}
            onStartComment={() => setIsCommenting(true)}
          />
        )}
      </div>
    </BubbleMenu>
  )
}

function messageFromUnknown(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}
