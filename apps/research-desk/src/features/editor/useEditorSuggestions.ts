import { useEffect, useMemo, useRef, useState, type Dispatch } from 'react'
import type { Editor } from '@tiptap/react'
import type { ChatModelTier } from '@/features/researchRuns/types'
import type {
  ChatContextReferenceRecord,
  EditorCommentAnchorRecord,
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorSuggestionEditPosition,
  EditorSuggestionGroupRecord,
  EditorSuggestionOrigin,
  EditorSuggestionRecord,
  ProjectState,
} from '@/features/project/types'
import { referenceDocsFromRefs } from '@/features/project/selectors'
import { renderChatRuleAttachmentContent } from '@/features/project/chatRuleRendering'
import type { ResearchDeskAction } from '@/features/researchDesk/state'
import {
  blockInsertionPositionForRange,
  clampAnchor,
  materializeAnchorForRange,
  materializeCommentThread,
  resolveAnchorRange,
  type EditorTextRange,
} from './anchoring'
import { normalizeEditorMarkdownForTiptap } from './tiptap'
import {
  LlmSuggestionProducer,
  type InstructionProposal,
  type SuggestionProducer,
} from './suggestionProducer'
import {
  clearRunning,
  clearRuns,
  markError,
  markErrors,
  markManyRunning,
  markRunning,
  runErrors as runErrorsOf,
  runningIds as runningIdsOf,
  type EditorRunStateMap,
} from './editorRunState'

export type UseEditorSuggestionsArgs = {
  activeDocument: EditorDocumentRecord | null
  activeEditor: Editor | null
  apiKey?: string
  attachedCommentIds: string[]
  attachedRefs: ChatContextReferenceRecord[]
  comments: EditorCommentThreadRecord[]
  dispatch: Dispatch<ResearchDeskAction>
  locale: 'de' | 'en'
  onGlobalSuccess: () => void
  selectedModelTier: ChatModelTier | null
  state: ProjectState
}

export type EditorSuggestionController = {
  clearInstructionFeedback: () => void
  documentSuggestions: EditorSuggestionRecord[]
  handleAcceptSuggestionGroup: (groupId: string) => void
  handleAcceptSuggestion: (suggestion: EditorSuggestionRecord) => void
  handleEditSuggestionProposal: (suggestionId: string, proposedText: string) => void
  handleGlobalRun: (globalInstruction: string) => Promise<void>
  handleInstructionRun: (instruction: string) => Promise<void>
  handleMarkSuggestionStale: (suggestionId: string) => void
  handleRefineSuggestion: (suggestionId: string, instruction: string) => Promise<void>
  handleRejectSuggestion: (suggestionId: string) => void
  handleRejectSuggestionGroup: (groupId: string) => void
  handleRunComment: (comment: EditorCommentThreadRecord) => Promise<void>
  handleStopRun: () => void
  handleStopSuggestionRun: (suggestionId: string) => void
  instructionFeedback: EditorInstructionFeedback | null
  isGlobalRunning: boolean
  runErrors: Record<string, string>
  runningCommentIds: readonly string[]
  runningSuggestionIds: readonly string[]
  suggestionErrors: Record<string, string>
}

export type EditorInstructionFeedback = {
  editCount?: number
  message: string
  state: 'error' | 'result' | 'thinking'
  warnings?: string[]
}

const EDITOR_ASSISTANT_TIMEOUT_MS = 120_000

export function useEditorSuggestions({
  activeDocument,
  activeEditor,
  apiKey,
  attachedCommentIds,
  attachedRefs,
  comments,
  dispatch,
  locale,
  onGlobalSuccess,
  selectedModelTier,
  state,
}: UseEditorSuggestionsArgs): EditorSuggestionController {
  const suggestionProducer = useMemo<SuggestionProducer>(
    () => new LlmSuggestionProducer({
      apiKey,
      locale,
      stack: state.ui.selectedStack,
      workspaceId: state.workspaceId,
    }),
    [apiKey, locale, state.ui.selectedStack, state.workspaceId],
  )
  const ruleSnippet = useMemo(
    () => attachedRefs
      .filter((ref) => ref.kind === 'chat-rule')
      .map((ref) => {
        if (ref.kind !== 'chat-rule') return ''
        const rule = state.chatRules[ref.ruleId]
        return rule ? renderChatRuleAttachmentContent(state, rule, new Date().toISOString()) : ''
      })
      .filter(Boolean)
      .join('\n\n'),
    [attachedRefs, state],
  )
  const attachments = useMemo(
    () => referenceDocsFromRefs(state, attachedRefs),
    [attachedRefs, state.fileAssets, state.fileAssetOrder, state.fileGroups],
  )
  const [commentRuns, setCommentRuns] = useState<EditorRunStateMap>({})
  const [suggestionRuns, setSuggestionRuns] = useState<EditorRunStateMap>({})
  const runningCommentIds = useMemo(() => runningIdsOf(commentRuns), [commentRuns])
  const runningSuggestionIds = useMemo(() => runningIdsOf(suggestionRuns), [suggestionRuns])
  const runErrors = useMemo(() => runErrorsOf(commentRuns), [commentRuns])
  const suggestionErrors = useMemo(() => runErrorsOf(suggestionRuns), [suggestionRuns])
  const [isGlobalRunning, setIsGlobalRunning] = useState(false)
  const [instructionFeedback, setInstructionFeedback] = useState<EditorInstructionFeedback | null>(null)
  const runAbortRef = useRef<AbortController | null>(null)
  const selectedModelTierRef = useRef(selectedModelTier)
  const selectedModelRef = useRef(state.ui.selectedChatModel)
  const selectedEffortRef = useRef(state.ui.selectedChatEffort)

  useEffect(() => {
    selectedModelTierRef.current = selectedModelTier
    selectedModelRef.current = state.ui.selectedChatModel
    selectedEffortRef.current = state.ui.selectedChatEffort
  }, [selectedModelTier, state.ui.selectedChatModel, state.ui.selectedChatEffort])

  useEffect(() => () => runAbortRef.current?.abort(), [])

  const documentSuggestions = useMemo(
    () => Object.values(state.editorSuggestions).filter((suggestion) => suggestion.documentId === activeDocument?.id),
    [activeDocument?.id, state.editorSuggestions],
  )

  async function handleRunComment(comment: EditorCommentThreadRecord) {
    if (!activeDocument) return
    const liveComment = activeEditor ? materializeCommentThread(activeEditor, comment) : comment
    if (!liveComment) {
      dispatch({ commentId: comment.id, status: 'stale', type: 'setEditorCommentStatus' })
      setCommentRuns((map) => markError(map, comment.id, staleAnchorMessage(locale)))
      return
    }
    const origin: EditorSuggestionOrigin = comment.kind === 'evidence_review'
      ? { commentId: comment.id, kind: 'evidence_review', preset: comment.evidencePreset ?? 'add_sources' }
      : { commentId: comment.id, kind: 'inline_edit' }
    dispatch({ commentId: comment.id, type: 'selectEditorComment' })
    runAbortRef.current?.abort()
    const controller = new AbortController()
    runAbortRef.current = controller
    setCommentRuns((map) => markRunning(map, comment.id))
    const clearRunTimeout = startEditorRunTimeout(() => {
      controller.abort()
      setCommentRuns((map) => markError(map, comment.id, editorTimeoutMessage(locale)))
    })
    try {
      const modelTier = selectedModelTierRef.current
      const model = selectedModelRef.current
      const effort = selectedEffortRef.current
      const proposal = await suggestionProducer.produce({
        anchor: liveComment.anchor,
        attachments,
        documentId: liveComment.documentId,
        documentMarkdown: activeDocument.contentMarkdown,
        instruction: liveComment.commentMarkdown,
        modelTier,
        model,
        effort,
        origin,
        originalMarkdown: liveComment.anchor.selectedMarkdown,
        originalText: liveComment.anchor.selectedText,
        signal: controller.signal,
      })
      const now = new Date().toISOString()
      const groupId = createLocalId('editor-suggestion-group')
      const group: EditorSuggestionGroupRecord = { createdAt: now, documentId: comment.documentId, id: groupId, origin }
      dispatch({
        group,
        suggestions: [createSuggestionRecord({ comment: liveComment, groupId, now, origin, proposal })],
        type: 'createEditorSuggestionGroup',
      })
    } catch (error) {
      if (controller.signal.aborted) return
      setCommentRuns((map) => markError(map, comment.id, messageFromError(error)))
    } finally {
      clearRunTimeout()
      setCommentRuns((map) => clearRunning(map, [comment.id]))
    }
  }

  function handleAcceptSuggestion(suggestion: EditorSuggestionRecord) {
    if (!applySuggestionToEditor(suggestion)) {
      dispatch({ suggestionId: suggestion.id, type: 'markEditorSuggestionStale' })
      return
    }
    dispatch({ suggestionId: suggestion.id, type: 'acceptEditorSuggestion' })
  }

  function handleAcceptSuggestionGroup(groupId: string) {
    const groupSuggestions = documentSuggestions
      .filter((suggestion) => suggestion.groupId === groupId && suggestion.status === 'pending')
      .sort((a, b) => a.anchor.from - b.anchor.from || a.createdAt.localeCompare(b.createdAt))
    for (const suggestion of groupSuggestions) {
      if (!applySuggestionToEditor(suggestion)) {
        dispatch({ suggestionId: suggestion.id, type: 'markEditorSuggestionStale' })
        continue
      }
      dispatch({ suggestionId: suggestion.id, type: 'acceptEditorSuggestion' })
    }
  }

  function handleRejectSuggestion(suggestionId: string) {
    dispatch({ suggestionId, type: 'rejectEditorSuggestion' })
  }

  function handleRejectSuggestionGroup(groupId: string) {
    dispatch({ groupId, type: 'rejectEditorSuggestionGroup' })
  }

  function handleEditSuggestionProposal(suggestionId: string, proposedText: string) {
    dispatch({
      proposedText,
      source: 'manual_edit',
      suggestionId,
      type: 'updateEditorSuggestionProposal',
    })
    setSuggestionRuns((map) => clearRuns(map, [suggestionId]))
  }

  async function handleRefineSuggestion(suggestionId: string, instruction: string) {
    if (!activeDocument || isGlobalRunning || runningSuggestionIds.includes(suggestionId)) return
    const trimmedInstruction = instruction.trim()
    const suggestion = documentSuggestions.find((item) => item.id === suggestionId)
    if (!suggestion || suggestion.status !== 'pending' || !trimmedInstruction) return
    runAbortRef.current?.abort()
    const controller = new AbortController()
    runAbortRef.current = controller
    setSuggestionRuns((map) => markRunning(map, suggestionId))
    const clearRunTimeout = startEditorRunTimeout(() => {
      controller.abort()
      setSuggestionRuns((map) => markError(map, suggestionId, editorTimeoutMessage(locale)))
    })
    try {
      const modelTier = selectedModelTierRef.current
      const model = selectedModelRef.current
      const effort = selectedEffortRef.current
      const originalInstruction = suggestion.origin.commentId
        ? state.editorComments[suggestion.origin.commentId]?.commentMarkdown
        : undefined
      const proposal = await suggestionProducer.refine({
        attachments,
        documentMarkdown: activeDocument.contentMarkdown,
        instruction: trimmedInstruction,
        modelTier,
        model,
        effort,
        originalInstruction,
        signal: controller.signal,
        suggestion,
      })
      if (controller.signal.aborted) return
      dispatch({
        changeSummary: proposal.changeSummary,
        instruction: trimmedInstruction,
        proposedText: proposal.proposedText,
        source: 'llm_refine',
        suggestionId,
        type: 'updateEditorSuggestionProposal',
        warnings: proposal.warnings,
      })
    } catch (error) {
      if (controller.signal.aborted) return
      setSuggestionRuns((map) => markError(map, suggestionId, messageFromError(error)))
    } finally {
      clearRunTimeout()
      setSuggestionRuns((map) => clearRunning(map, [suggestionId]))
    }
  }

  function handleMarkSuggestionStale(suggestionId: string) {
    dispatch({ suggestionId, type: 'markEditorSuggestionStale' })
  }

  function handleStopRun() {
    runAbortRef.current?.abort()
    setIsGlobalRunning(false)
    setCommentRuns((map) => clearRunning(map, Object.keys(map)))
    setInstructionFeedback((prev) => (prev?.state === 'thinking' ? null : prev))
  }

  function handleStopSuggestionRun(suggestionId: string) {
    runAbortRef.current?.abort()
    setSuggestionRuns((map) => clearRuns(map, [suggestionId]))
  }

  function clearInstructionFeedback() {
    setInstructionFeedback(null)
  }

  function applySuggestionToEditor(suggestion: EditorSuggestionRecord): boolean {
    if (!activeEditor) return false
    const target = resolveSuggestionTarget(activeEditor, suggestion)
    if (!target) return false
    const content = normalizeEditorMarkdownForTiptap(suggestion.proposedText)
    if (target.kind === 'replace') {
      activeEditor.chain().focus().insertContentAt(
        target.range,
        content,
        { contentType: 'markdown' },
      ).run()
      return true
    }
    activeEditor.chain().focus().insertContentAt(
      target.at,
      content,
      { contentType: 'markdown' },
    ).run()
    return true
  }

  async function handleGlobalRun(globalInstruction: string) {
    if (!activeDocument || isGlobalRunning) return
    const targets = comments.filter((comment) =>
      comment.status === 'open' && comment.kind === 'collect' && attachedCommentIds.includes(comment.id))
    if (targets.length === 0) return
    const draftInstruction = globalInstruction.trim()
    const snippet = ruleSnippet
    runAbortRef.current?.abort()
    const controller = new AbortController()
    runAbortRef.current = controller
    setIsGlobalRunning(true)
    setCommentRuns((map) => markManyRunning(map, targets.map((comment) => comment.id)))

    const documentMarkdown = activeDocument.contentMarkdown
    const documentId = activeDocument.id
    const modelTier = selectedModelTierRef.current
    const model = selectedModelRef.current
    const effort = selectedEffortRef.current
    const now = new Date().toISOString()
    const groupId = createLocalId('editor-suggestion-group')
    const clearRunTimeout = startEditorRunTimeout(() => {
      controller.abort()
      const timeoutErrors = Object.fromEntries(targets.map((comment) => [comment.id, editorTimeoutMessage(locale)]))
      setCommentRuns((map) => markErrors(map, timeoutErrors))
      setIsGlobalRunning(false)
    })

    const produceForComment = async (comment: EditorCommentThreadRecord) => {
      const liveComment = activeEditor ? materializeCommentThread(activeEditor, comment) : comment
      if (!liveComment) throw new Error(staleAnchorMessage(locale))
      const origin: EditorSuggestionOrigin = { commentId: comment.id, kind: 'global_run' }
      const proposal = await suggestionProducer.produce({
        anchor: liveComment.anchor,
        attachments,
        documentId,
        documentMarkdown,
        globalInstruction: draftInstruction || undefined,
        instruction: liveComment.commentMarkdown,
        modelTier,
        model,
        effort,
        origin,
        originalMarkdown: liveComment.anchor.selectedMarkdown,
        originalText: liveComment.anchor.selectedText,
        signal: controller.signal,
        snippet: snippet || undefined,
      })
      return { comment: liveComment, origin, proposal }
    }

    const suggestions: EditorSuggestionRecord[] = []
    const errors: Record<string, string> = {}
    const poolSize = 4
    for (let index = 0; index < targets.length; index += poolSize) {
      const batch = targets.slice(index, index + poolSize)
      const settled = await Promise.allSettled(batch.map(produceForComment))
      settled.forEach((outcome, offset) => {
        const comment = batch[offset]
        if (outcome.status === 'fulfilled') {
          const { origin, proposal } = outcome.value
          suggestions.push(createSuggestionRecord({ comment, documentId, groupId, now, origin, proposal }))
        } else if (!controller.signal.aborted) {
          errors[comment.id] = messageFromError(outcome.reason)
        }
      })
    }

    if (controller.signal.aborted) {
      clearRunTimeout()
      setIsGlobalRunning(false)
      setCommentRuns((map) => clearRunning(map, targets.map((comment) => comment.id)))
      return
    }
    if (suggestions.length > 0) {
      dispatch({
        group: {
          createdAt: now,
          documentId,
          id: groupId,
          origin: { kind: 'global_run' },
        },
        suggestions,
        type: 'createEditorSuggestionGroup',
      })
      const firstCommentId = suggestions[0]?.origin.commentId
      if (firstCommentId) dispatch({ commentId: firstCommentId, type: 'selectEditorComment' })
    }
    if (Object.keys(errors).length > 0) {
      setCommentRuns((map) => markErrors(map, errors))
    } else {
      dispatch({ draft: '', type: 'setEditorAssistantDraft' })
      onGlobalSuccess()
    }
    clearRunTimeout()
    setIsGlobalRunning(false)
    setCommentRuns((map) => clearRunning(map, targets.map((comment) => comment.id)))
  }

  async function handleInstructionRun(instruction: string) {
    if (!activeDocument || isGlobalRunning) return
    const draftInstruction = instruction.trim()
    if (!draftInstruction) return
    const snippet = ruleSnippet
    runAbortRef.current?.abort()
    const controller = new AbortController()
    runAbortRef.current = controller
    setIsGlobalRunning(true)
    setInstructionFeedback({
      message: locale === 'de' ? 'Dokument-Anweisung wird verarbeitet …' : 'Processing document instruction …',
      state: 'thinking',
    })
    const clearRunTimeout = startEditorRunTimeout(() => {
      controller.abort()
      setInstructionFeedback({
        message: editorTimeoutMessage(locale),
        state: 'error',
      })
      setIsGlobalRunning(false)
    })

    try {
      const modelTier = selectedModelTierRef.current
      const model = selectedModelRef.current
      const effort = selectedEffortRef.current
      const proposal = await suggestionProducer.produceInstruction({
        attachments,
        documentMarkdown: activeDocument.contentMarkdown,
        instruction: draftInstruction,
        modelTier,
        model,
        effort,
        signal: controller.signal,
        snippet: snippet || undefined,
      })
      if (controller.signal.aborted) return
      const now = new Date().toISOString()
      const groupId = createLocalId('editor-suggestion-group')
      const suggestions = createInstructionSuggestionRecords({
        activeEditor,
        document: activeDocument,
        groupId,
        now,
        proposal,
      })
      if (suggestions.length > 0) {
        dispatch({
          group: {
            assistantMessage: proposal.assistantMessage,
            createdAt: now,
            documentId: activeDocument.id,
            id: groupId,
            origin: { kind: 'global_run' },
            warnings: proposal.warnings,
          },
          suggestions,
          type: 'createEditorSuggestionGroup',
        })
      }
      dispatch({ draft: '', type: 'setEditorAssistantDraft' })
      onGlobalSuccess()
      setInstructionFeedback({
        editCount: suggestions.length,
        message: proposal.assistantMessage || defaultInstructionResultMessage(locale, suggestions.length),
        state: 'result',
        warnings: proposal.warnings,
      })
    } catch (error) {
      if (controller.signal.aborted) return
      setInstructionFeedback({
        message: messageFromError(error),
        state: 'error',
      })
    } finally {
      clearRunTimeout()
      setIsGlobalRunning(false)
    }
  }

  return {
    clearInstructionFeedback,
    documentSuggestions,
    handleAcceptSuggestionGroup,
    handleAcceptSuggestion,
    handleEditSuggestionProposal,
    handleGlobalRun,
    handleInstructionRun,
    handleMarkSuggestionStale,
    handleRefineSuggestion,
    handleRejectSuggestion,
    handleRejectSuggestionGroup,
    handleRunComment,
    handleStopRun,
    handleStopSuggestionRun,
    instructionFeedback,
    isGlobalRunning,
    runErrors,
    runningCommentIds,
    runningSuggestionIds,
    suggestionErrors,
  }
}

type SuggestionRecordArgs = {
  comment: EditorCommentThreadRecord
  documentId?: string
  groupId: string
  now: string
  origin: EditorSuggestionOrigin
  proposal: {
    changeSummary?: string[]
    evidence?: EditorSuggestionRecord['evidence']
    proposedText: string
    warnings?: string[]
  }
}

type InstructionRecordArgs = {
  activeEditor: Editor | null
  document: EditorDocumentRecord
  groupId: string
  now: string
  proposal: InstructionProposal
}

type SuggestionApplyTarget =
  | { kind: 'insert'; at: number }
  | { kind: 'replace'; range: EditorTextRange }

function createSuggestionRecord({
  comment,
  documentId = comment.documentId,
  groupId,
  now,
  origin,
  proposal,
}: SuggestionRecordArgs): EditorSuggestionRecord {
  return {
    anchor: comment.anchor,
    blockId: comment.anchor.blockId ?? '',
    createdAt: now,
    documentId,
    groupId,
    id: createLocalId('editor-suggestion'),
    originalMarkdown: comment.anchor.selectedMarkdown,
    originalText: comment.anchor.selectedText,
    origin,
    proposedText: proposal.proposedText,
    revision: 1,
    status: 'pending',
    updatedAt: now,
    ...(proposal.changeSummary?.length ? { changeSummary: proposal.changeSummary } : {}),
    ...(proposal.evidence ? { evidence: proposal.evidence } : {}),
    ...(proposal.warnings?.length ? { warnings: proposal.warnings } : {}),
  }
}

function createInstructionSuggestionRecords({
  activeEditor,
  document,
  groupId,
  now,
  proposal,
}: InstructionRecordArgs): EditorSuggestionRecord[] {
  const suggestions: EditorSuggestionRecord[] = []
  proposal.edits.forEach((edit, index) => {
    const position = edit.position
    const anchorText = edit.find.trim()
    const proposedText = edit.text.trim()
    if (!proposedText && position !== 'replace') return
    const anchor = createInstructionAnchor(activeEditor, document.id, edit, index)
    const originalText = position === 'replace' ? anchorText : ''
    suggestions.push({
      anchor,
      anchorText,
      blockId: anchor.blockId ?? `instruction-${index}`,
      changeSummary: edit.note ? [edit.note] : undefined,
      createdAt: now,
      documentId: document.id,
      editPosition: position,
      groupId,
      id: createLocalId('editor-suggestion'),
      originalMarkdown: position === 'replace' ? anchorText : '',
      originalText,
      origin: { kind: 'global_run' },
      proposedText,
      revision: 1,
      status: 'pending',
      updatedAt: now,
      warnings: proposal.warnings?.length ? proposal.warnings : undefined,
    })
  })
  return suggestions
}

function createInstructionAnchor(
  activeEditor: Editor | null,
  documentId: string,
  edit: { find: string; position: EditorSuggestionEditPosition; quote_after: string; quote_before: string },
  index: number,
): EditorCommentAnchorRecord {
  const anchorText = edit.find.trim()
  if (activeEditor && anchorText) {
    const range = resolveAnchorRange(activeEditor, {
      hint: 1,
      quoteAfter: edit.quote_after,
      quoteBefore: edit.quote_before,
      text: anchorText,
    })
    if (range) return materializeAnchorForRange(activeEditor, range)
  }
  const docSize = activeEditor?.state.doc.content.size ?? 1
  return {
    blockId: `${documentId}:instruction:${index}`,
    from: docSize,
    quoteAfter: edit.quote_after,
    quoteBefore: edit.quote_before,
    selectedMarkdown: anchorText,
    selectedText: anchorText,
    to: docSize,
  }
}

function resolveSuggestionTarget(
  editor: Editor,
  suggestion: EditorSuggestionRecord,
): SuggestionApplyTarget | null {
  const position = suggestion.editPosition ?? 'replace'
  if (position === 'append') {
    return { at: editor.state.doc.content.size, kind: 'insert' }
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
  if (position === 'replace') return { kind: 'replace', range }
  return {
    at: blockInsertionPositionForRange(editor, range, position),
    kind: 'insert',
  }
}

function messageFromError(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}

function startEditorRunTimeout(onTimeout: () => void): () => void {
  const timeoutId = globalThis.setTimeout(onTimeout, EDITOR_ASSISTANT_TIMEOUT_MS)
  return () => globalThis.clearTimeout(timeoutId)
}

function editorTimeoutMessage(locale: 'de' | 'en'): string {
  if (locale === 'de') {
    return 'Keine Modellantwort nach 120 Sekunden. Der Lauf wurde abgebrochen; bitte erneut versuchen oder ein schnelleres Modell wählen.'
  }
  return 'No model response after 120 seconds. The run was cancelled; retry or choose a faster model.'
}

function defaultInstructionResultMessage(locale: 'de' | 'en', editCount: number): string {
  if (locale === 'de') {
    return editCount > 0
      ? `${editCount} Dokument-Änderungen vorgeschlagen.`
      : 'Keine konkreten Dokument-Änderungen vorgeschlagen.'
  }
  return editCount > 0
    ? `${editCount} document changes proposed.`
    : 'No concrete document changes proposed.'
}

function staleAnchorMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Textstelle hat sich geändert. Bitte markieren Sie den Text neu.'
    : 'The referenced text changed. Please select the passage again.'
}


function createLocalId(prefix: string): string {
  if (globalThis.crypto?.randomUUID) return `${prefix}-${globalThis.crypto.randomUUID()}`
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}
