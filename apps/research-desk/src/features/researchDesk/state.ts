import type {
  ChatModelTier,
  CreateResearchRunRequest,
  ResearchRunEvent,
  ResearchRunResult,
  ResearchRunSummary,
} from '@/features/researchRuns/types'
import { createEmptyProjectState, createSeedProjectState } from '@/features/project/seedProject'
import type {
  ChatChainStepRecord,
  ChatContextReferenceRecord,
  ChatMessageRecord,
  ChatMessageAttachmentRecord,
  ChatMessageModelResolutionRecord,
  ChatRuleRecord,
  ChatThreadGroupRecord,
  ChatThreadRecord,
  EditorCommentKind,
  EditorCommentStatus,
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorEvidencePreset,
  EditorFolderRecord,
  EditorPanelTab,
  EditorSuggestionGroupRecord,
  EditorSuggestionRecord,
  EditorSuggestionRevisionSource,
  EditorViewMode,
  FileAssetRecord,
  FileGroupRecord,
  ProjectConnection,
  ProjectPreferences,
  ProjectState,
  ResearchRunRecord,
} from '@/features/project/types'
import {
  applyRunEvent,
  attachRunResult,
  mergeRunSummary,
} from '@/features/project/types'
import { moveItem } from '@/features/composer/reorder'
import type { AppView, JobFilter, ResearchJob } from './types'

export const stackOptions = [
  'anthropic_perplexity',
  'azure_web_search',
  'azure_openai_web_search',
]

export type ResearchDeskState = ProjectState

export type ResearchDeskAction =
  | { ref: ChatContextReferenceRecord; type: 'attachChatContextToDraft' }
  | { type: 'attachReportToChatDraft'; runId: string }
  | { type: 'attachReportToNewChat'; runId: string }
  | { ref: ChatContextReferenceRecord; type: 'removeChatContextFromDraft' }
  | { fromIndex: number; toIndex: number; type: 'reorderChatContextInDraft' }
  | { type: 'clearChatDraftAttachment' }
  | { groupId?: string | null; type: 'createChatThread' }
  | { messageId: string; threadId: string; type: 'branchChatThreadFromMessage' }
  | { type: 'createLocalRun'; request: CreateResearchRunRequest }
  | { type: 'cancelLocalRun'; runId: string }
  | { type: 'clearChatThread'; threadId: string }
  | { messageIds: string[]; threadId: string; type: 'deleteChatMessages' }
  | { title: string; type: 'createChatThreadGroup' }
  | { ruleId: string; type: 'deleteChatRule' }
  | { groupId: string; type: 'deleteChatThreadGroup' }
  | { type: 'deleteChatThread'; threadId: string }
  | { type: 'deleteJob'; jobId: string }
  | { folderId?: string | null; type: 'createEditorDocument' }
  | { title: string; type: 'createEditorFolder' }
  | { documentId: string; type: 'deleteEditorDocument' }
  | { folderId: string; type: 'deleteEditorFolder' }
  | { documentId: string; type: 'openEditorDocument' }
  | { documentId: string; type: 'closeEditorDocumentTab' }
  | { documentId: string; type: 'setActiveEditorDocument' }
  | { documentId: string; title: string; type: 'renameEditorDocument' }
  | { folderId: string; title: string; type: 'renameEditorFolder' }
  | { folderId: string; targetIndex: number; type: 'moveEditorFolder' }
  | { documentId: string; folderId: string | null; targetIndex: number; type: 'moveEditorDocumentToFolder' }
  | { comment: EditorCommentThreadRecord; type: 'createEditorComment' }
  | { commentId: string; type: 'resolveEditorComment' }
  | { commentId: string; type: 'deleteEditorComment' }
  | { commentId: string; contentMarkdown: string; type: 'updateEditorCommentText' }
  | { commentId: string; status: EditorCommentStatus; type: 'setEditorCommentStatus' }
  | { commentId: string; kind: EditorCommentKind; type: 'setEditorCommentKind' }
  | { commentId: string; preset: EditorEvidencePreset | null; type: 'setEditorCommentEvidencePreset' }
  | { group: EditorSuggestionGroupRecord; suggestions: EditorSuggestionRecord[]; type: 'createEditorSuggestionGroup' }
  | { suggestionId: string; type: 'acceptEditorSuggestion' }
  | { suggestionId: string; type: 'rejectEditorSuggestion' }
  | { suggestionId: string; type: 'markEditorSuggestionStale' }
  | {
    changeSummary?: string[]
    instruction?: string
    proposedText: string
    source: EditorSuggestionRevisionSource
    suggestionId: string
    type: 'updateEditorSuggestionProposal'
    warnings?: string[]
  }
  | { groupId: string; type: 'acceptEditorSuggestionGroup' }
  | { groupId: string; type: 'rejectEditorSuggestionGroup' }
  | { commentId: string | null; type: 'selectEditorComment' }
  | { documentId: string; contentMarkdown: string; type: 'updateEditorDocumentMarkdown' }
  | { isVisible: boolean; type: 'setEditorAssistantVisible' }
  | { isVisible: boolean; type: 'setEditorCommentPanelVisible' }
  | { isVisible: boolean; type: 'setEditorTreeVisible' }
  | { draft: string; type: 'setEditorAssistantDraft' }
  | { tab: EditorPanelTab; type: 'setEditorPanelTab' }
  | { mode: EditorViewMode; type: 'setEditorViewMode' }
  | { documentId: string; type: 'setEditorDiffAnchor' }
  | { isVisible: boolean; type: 'setEditorDiffVisible' }
  | { runId: string; type: 'importResearchReportToEditor' }
  | { contentMarkdown: string; messageId: string; threadId: string; type: 'editChatUserMessage' }
  | { type: 'hydrateProject'; state: ProjectState }
  | { type: 'appendApiRunEvent'; event: ResearchRunEvent }
  | { type: 'attachApiRunResult'; result: ResearchRunResult }
  | { type: 'markApiRunError'; message: string; runId: string }
  | { type: 'upsertApiRunSummary'; select?: boolean; summary: ResearchRunSummary }
  | {
    assistantMessageId: string
    contentMarkdown: string
    createdAt: string
    attachmentRefs?: ChatContextReferenceRecord[]
    modelResolution?: ChatMessageModelResolutionRecord
    threadId: string
    type: 'startChatExchange'
    userMessageId: string
  }
  | {
    assistantMessageId: string
    createdAt: string
    modelResolution?: ChatMessageModelResolutionRecord
    threadId: string
    type: 'startChatAssistantResponse'
    userMessageId: string
  }
  | {
    assistantMessageId: string
    chainTrace?: ChatChainStepRecord[]
    contentMarkdown: string
    modelResolution?: ChatMessageModelResolutionRecord
    threadId: string
    type: 'setChatAssistantMessageContent'
  }
  | { type: 'setDemoMode'; enabled: boolean }
  | { type: 'markProjectSaved'; connection: ProjectConnection; preferences?: ProjectPreferences; savedAt: string }
  | { groupId: string; targetIndex: number; type: 'moveChatThreadGroup' }
  | { groupId: string | null; targetIndex: number; threadId: string; type: 'moveChatThreadToGroup' }
  | { groupId: string; title: string; type: 'renameChatThreadGroup' }
  | { title: string; threadId: string; type: 'renameChatThread' }
  | { type: 'selectChatThread'; threadId: string }
  | { type: 'selectJob'; jobId: string }
  | { type: 'setActiveFilter'; filter: JobFilter }
  | { type: 'setActiveView'; view: AppView }
  | { type: 'setComposerVisible'; isVisible: boolean }
  | { type: 'setReportExpanded'; isExpanded: boolean }
  | { type: 'setReportVisible'; isVisible: boolean }
  | { type: 'setChatHistoryVisible'; isVisible: boolean }
  | { enabled: boolean; type: 'setChatChainingEnabled' }
  | { type: 'setSelectedChatModelTier'; tier: ChatModelTier | null }
  | { type: 'setSelectedStack'; stack: string }
  | { rule: ChatRuleRecord; type: 'upsertChatRule' }
  | { type: 'toggleJob'; jobId: string }
  | { assets: FileAssetRecord[]; type: 'ingestFileAssets' }
  | { fileId: string; label: string; type: 'renameFileAsset' }
  | { fileId: string; groupId: string | null; sectionId: string; type: 'moveFileAsset' }
  | { fileId: string; type: 'deleteFileAsset' }
  | { sectionId: string; title: string; type: 'createFileGroup' }
  | { groupId: string; title: string; type: 'renameFileGroup' }
  | { groupId: string; type: 'deleteFileGroup' }
  | { sectionId: string; title: string; type: 'renameFileLibrarySection' }

export function initializeResearchDeskState(): ResearchDeskState {
  return createEmptyProjectState()
}

export function researchDeskReducer(
  state: ResearchDeskState,
  action: ResearchDeskAction,
): ResearchDeskState {
  if (action.type === 'hydrateProject') {
    return action.state
  }
  if (action.type === 'setDemoMode') {
    return action.enabled ? createSeedProjectState() : createEmptyProjectState()
  }
  if (action.type === 'markProjectSaved') {
    return {
      ...state,
      connection: action.connection,
      dirty: false,
      preferences: action.preferences ?? state.preferences,
      project: {
        ...state.project,
        updatedAt: action.savedAt,
      },
    }
  }
  if (action.type === 'setActiveView') {
    return { ...state, ui: { ...state.ui, activeView: action.view } }
  }
  if (action.type === 'setActiveFilter') {
    const selectedJobId = resolveVisibleSelection(
      state.researchRunOrder,
      state.researchRuns,
      action.filter,
      state.ui.selectedJobId,
    )

    return {
      ...state,
      ui: {
        ...state.ui,
        activeFilter: action.filter,
        expandedJobId: selectedJobId,
        selectedJobId,
      },
    }
  }
  if (action.type === 'setSelectedStack') {
    return {
      ...state,
      dirty: true,
      ui: { ...state.ui, selectedStack: action.stack },
    }
  }
  if (action.type === 'setSelectedChatModelTier') {
    return {
      ...state,
      dirty: true,
      ui: { ...state.ui, selectedChatModelTier: action.tier },
    }
  }
  if (action.type === 'setChatChainingEnabled') {
    return {
      ...state,
      dirty: true,
      ui: { ...state.ui, chatChainingEnabled: action.enabled },
    }
  }
  if (action.type === 'setReportExpanded') {
    return { ...state, ui: { ...state.ui, isReportExpanded: action.isExpanded } }
  }
  if (action.type === 'setReportVisible') {
    return { ...state, ui: { ...state.ui, isReportVisible: action.isVisible } }
  }
  if (action.type === 'setChatHistoryVisible') {
    return { ...state, ui: { ...state.ui, isChatHistoryVisible: action.isVisible } }
  }
  if (action.type === 'setComposerVisible') {
    return { ...state, ui: { ...state.ui, isComposerVisible: action.isVisible } }
  }
  if (action.type === 'selectJob') {
    return {
      ...state,
      ui: {
        ...state.ui,
        expandedJobId: action.jobId,
        selectedJobId: action.jobId,
      },
    }
  }
  if (action.type === 'toggleJob') {
    return {
      ...state,
      ui: {
        ...state.ui,
        expandedJobId: state.ui.expandedJobId === action.jobId ? null : action.jobId,
        selectedJobId: action.jobId,
      },
    }
  }
  if (action.type === 'deleteJob') {
    const researchRuns = { ...state.researchRuns }
    delete researchRuns[action.jobId]
    const researchRunOrder = state.researchRunOrder.filter((runId) => runId !== action.jobId)
    const selectedJobId = resolveVisibleSelection(
      researchRunOrder,
      researchRuns,
      state.ui.activeFilter,
      state.ui.selectedJobId === action.jobId ? null : state.ui.selectedJobId,
    )

    return {
      ...state,
      dirty: true,
      researchRunOrder,
      researchRuns,
      ui: {
        ...state.ui,
        expandedJobId: selectedJobId,
        pendingChatReportRunId: state.ui.pendingChatReportRunId === action.jobId
          ? null
          : state.ui.pendingChatReportRunId,
        selectedJobId,
      },
    }
  }
  if (action.type === 'createEditorFolder') {
    const now = new Date().toISOString()
    const folder: EditorFolderRecord = {
      createdAt: now,
      id: createId('editor-folder'),
      title: action.title.trim() || 'New folder',
      updatedAt: now,
    }
    return {
      ...state,
      dirty: true,
      editorFolderOrder: [...state.editorFolderOrder, folder.id],
      editorFolders: {
        ...state.editorFolders,
        [folder.id]: folder,
      },
    }
  }
  if (action.type === 'renameEditorFolder') {
    const folder = state.editorFolders[action.folderId]
    const title = action.title.trim()
    if (!folder || !title || folder.title === title) return state
    return {
      ...state,
      dirty: true,
      editorFolders: {
        ...state.editorFolders,
        [folder.id]: {
          ...folder,
          title,
          updatedAt: new Date().toISOString(),
        },
      },
    }
  }
  if (action.type === 'deleteEditorFolder') {
    if (!state.editorFolders[action.folderId]) return state
    const editorFolders = { ...state.editorFolders }
    delete editorFolders[action.folderId]
    const updatedAt = new Date().toISOString()
    return {
      ...state,
      dirty: true,
      editorDocumentOrder: moveSectionDocumentIds(
        state.editorDocumentOrder,
        state.editorDocuments,
        action.folderId,
        null,
        state.editorDocumentOrder.length,
      ),
      editorDocuments: Object.fromEntries(
        Object.entries(state.editorDocuments).map(([documentId, document]) => [
          documentId,
          document.folderId === action.folderId
            ? { ...document, folderId: null, updatedAt }
            : document,
        ]),
      ),
      editorFolderOrder: state.editorFolderOrder.filter((folderId) => folderId !== action.folderId),
      editorFolders,
    }
  }
  if (action.type === 'moveEditorFolder') {
    return moveEditorFolder(state, action.folderId, action.targetIndex)
  }
  if (action.type === 'moveEditorDocumentToFolder') {
    return moveEditorDocumentToFolder(state, action.documentId, action.folderId, action.targetIndex)
  }
  if (action.type === 'createEditorDocument') {
    const document = createEditorDocument({
      folderId: action.folderId ?? null,
      source: 'blank',
    })
    return withEditorDocument(state, document, { activeView: 'editor' })
  }
  if (action.type === 'importResearchReportToEditor') {
    const report = reportFromRun(state, action.runId)
    if (!report) return state
    const document = createEditorDocument({
      contentMarkdown: report.contentMarkdown,
      source: 'imported-research-report',
      sourceRunId: report.runId,
      title: `${report.title}.md`,
    })
    return withEditorDocument(state, document, { activeView: 'editor' })
  }
  if (action.type === 'openEditorDocument') {
    if (!state.editorDocuments[action.documentId]) return state
    return {
      ...state,
      ui: { ...state.ui, activeView: 'editor' },
      editorUi: {
        ...state.editorUi,
        activeDocumentId: action.documentId,
        openDocumentIds: addOpenEditorDocumentId(
          state.editorUi.openDocumentIds,
          action.documentId,
        ),
      },
    }
  }
  if (action.type === 'setActiveEditorDocument') {
    if (!state.editorDocuments[action.documentId]) return state
    return {
      ...state,
      editorUi: {
        ...state.editorUi,
        activeDocumentId: action.documentId,
        openDocumentIds: addOpenEditorDocumentId(
          state.editorUi.openDocumentIds,
          action.documentId,
        ),
      },
    }
  }
  if (action.type === 'closeEditorDocumentTab') {
    const openDocumentIds = state.editorUi.openDocumentIds.filter((id) => id !== action.documentId)
    const activeDocumentId = state.editorUi.activeDocumentId === action.documentId
      ? openDocumentIds[openDocumentIds.length - 1] ?? null
      : state.editorUi.activeDocumentId
    return {
      ...state,
      editorUi: {
        ...state.editorUi,
        activeDocumentId,
        openDocumentIds,
        selectedCommentId: activeDocumentId ? state.editorUi.selectedCommentId : null,
      },
    }
  }
  if (action.type === 'deleteEditorDocument') {
    if (!state.editorDocuments[action.documentId]) return state
    const editorDocuments = { ...state.editorDocuments }
    delete editorDocuments[action.documentId]
    const editorComments = Object.fromEntries(
      Object.entries(state.editorComments).filter(([, comment]) => comment.documentId !== action.documentId),
    )
    const editorDocumentOrder = state.editorDocumentOrder.filter((id) => id !== action.documentId)
    const openDocumentIds = state.editorUi.openDocumentIds.filter((id) => id !== action.documentId)
    const activeDocumentId = state.editorUi.activeDocumentId === action.documentId
      ? openDocumentIds[openDocumentIds.length - 1] ?? editorDocumentOrder[0] ?? null
      : state.editorUi.activeDocumentId
    return {
      ...state,
      dirty: true,
      editorComments,
      editorDocumentOrder,
      editorDocuments,
      editorUi: {
        ...state.editorUi,
        activeDocumentId,
        openDocumentIds: activeDocumentId
          ? addOpenEditorDocumentId(openDocumentIds, activeDocumentId)
          : openDocumentIds,
        selectedCommentId: null,
      },
    }
  }
  if (action.type === 'renameEditorDocument') {
    const document = state.editorDocuments[action.documentId]
    const title = normalizeEditorDocumentTitle(action.title)
    if (!document || document.title === title) return state
    return {
      ...state,
      dirty: true,
      editorDocuments: {
        ...state.editorDocuments,
        [document.id]: {
          ...document,
          title,
          updatedAt: new Date().toISOString(),
        },
      },
    }
  }
  if (action.type === 'updateEditorDocumentMarkdown') {
    const document = state.editorDocuments[action.documentId]
    if (!document || document.contentMarkdown === action.contentMarkdown) return state
    const updatedAt = new Date().toISOString()
    return {
      ...state,
      dirty: true,
      editorDocuments: {
        ...state.editorDocuments,
        [document.id]: {
          ...document,
          contentMarkdown: action.contentMarkdown,
          revision: document.revision + 1,
          updatedAt,
        },
      },
    }
  }
  if (action.type === 'createEditorComment') {
    if (!state.editorDocuments[action.comment.documentId]) return state
    return {
      ...state,
      dirty: true,
      editorComments: {
        ...state.editorComments,
        [action.comment.id]: action.comment,
      },
      editorUi: {
        ...state.editorUi,
        isCommentPanelVisible: true,
        panelTab: 'comments',
        selectedCommentId: action.comment.id,
      },
    }
  }
  if (action.type === 'resolveEditorComment') {
    const comment = state.editorComments[action.commentId]
    if (!comment || comment.status === 'resolved') return state
    const now = new Date().toISOString()
    return {
      ...state,
      dirty: true,
      editorComments: {
        ...state.editorComments,
        [comment.id]: {
          ...comment,
          status: 'resolved',
          updatedAt: now,
        },
      },
      editorSuggestions: retireActiveEditorSuggestionsForComments(
        state.editorSuggestions,
        new Set([comment.id]),
        now,
      ),
    }
  }
  if (action.type === 'setEditorCommentStatus') {
    const comment = state.editorComments[action.commentId]
    if (!comment || comment.status === action.status) return state
    const now = new Date().toISOString()
    return {
      ...state,
      dirty: true,
      editorComments: {
        ...state.editorComments,
        [comment.id]: { ...comment, status: action.status, updatedAt: now },
      },
      editorSuggestions: action.status === 'resolved'
        ? retireActiveEditorSuggestionsForComments(state.editorSuggestions, new Set([comment.id]), now)
        : state.editorSuggestions,
    }
  }
  if (action.type === 'setEditorCommentKind') {
    const comment = state.editorComments[action.commentId]
    if (!comment || comment.kind === action.kind) return state
    const now = new Date().toISOString()
    const evidencePreset = action.kind === 'evidence_review'
      ? comment.evidencePreset ?? 'add_sources'
      : undefined
    return {
      ...state,
      dirty: true,
      editorComments: {
        ...state.editorComments,
        [comment.id]: { ...comment, evidencePreset, kind: action.kind, updatedAt: now },
      },
      editorSuggestions: retireActiveEditorSuggestionsForComments(
        state.editorSuggestions,
        new Set([comment.id]),
        now,
      ),
    }
  }
  if (action.type === 'setEditorCommentEvidencePreset') {
    const comment = state.editorComments[action.commentId]
    if (!comment) return state
    const now = new Date().toISOString()
    return {
      ...state,
      dirty: true,
      editorComments: {
        ...state.editorComments,
        [comment.id]: { ...comment, evidencePreset: action.preset ?? undefined, updatedAt: now },
      },
      editorSuggestions: retireActiveEditorSuggestionsForComments(
        state.editorSuggestions,
        new Set([comment.id]),
        now,
      ),
    }
  }
  if (action.type === 'deleteEditorComment') {
    if (!state.editorComments[action.commentId]) return state
    const editorComments = { ...state.editorComments }
    delete editorComments[action.commentId]
    const editorSuggestions = Object.fromEntries(
      Object.entries(state.editorSuggestions).filter(([, suggestion]) =>
        suggestion.origin.commentId !== action.commentId),
    )
    return {
      ...state,
      dirty: true,
      editorComments,
      editorSuggestions,
      editorUi: {
        ...state.editorUi,
        selectedCommentId: state.editorUi.selectedCommentId === action.commentId
          ? null
          : state.editorUi.selectedCommentId,
      },
    }
  }
  if (action.type === 'updateEditorCommentText') {
    const comment = state.editorComments[action.commentId]
    const text = action.contentMarkdown.trim()
    if (!comment || !text || comment.commentMarkdown === text) return state
    const now = new Date().toISOString()
    return {
      ...state,
      dirty: true,
      editorComments: {
        ...state.editorComments,
        [comment.id]: { ...comment, commentMarkdown: text, updatedAt: now },
      },
      editorSuggestions: retireActiveEditorSuggestionsForComments(
        state.editorSuggestions,
        new Set([comment.id]),
        now,
      ),
    }
  }
  if (action.type === 'createEditorSuggestionGroup') {
    const now = new Date().toISOString()
    const commentIds = new Set(
      action.suggestions
        .map((suggestion) => suggestion.origin.commentId)
        .filter((commentId): commentId is string => Boolean(commentId)),
    )
    let editorSuggestions = retireActiveEditorSuggestionsForComments(
      state.editorSuggestions,
      commentIds,
      now,
    )
    if (action.group.origin.kind === 'global_run' && !action.group.origin.commentId) {
      editorSuggestions = retireActiveDocumentInstructionSuggestions(
        editorSuggestions,
        action.group.documentId,
        now,
      )
    }
    // Spread into a fresh object before adding the new suggestions. When nothing was
    // retired, the retire helpers return the input map by reference, so mutating it here
    // would keep `state.editorSuggestions` referentially equal across the update. Consumers
    // memoize on that reference (e.g. documentSuggestions in useEditorSuggestions), so an
    // in-place mutation leaves the new suggestion invisible until an unrelated re-render --
    // the "first suggestion only shows after a second action" bug in production builds.
    const nextEditorSuggestions = { ...editorSuggestions }
    for (const suggestion of action.suggestions) {
      nextEditorSuggestions[suggestion.id] = suggestion
    }
    return {
      ...state,
      dirty: true,
      editorSuggestionGroups: { ...state.editorSuggestionGroups, [action.group.id]: action.group },
      editorSuggestions: nextEditorSuggestions,
    }
  }
  if (
    action.type === 'acceptEditorSuggestion'
    || action.type === 'rejectEditorSuggestion'
    || action.type === 'markEditorSuggestionStale'
  ) {
    const suggestion = state.editorSuggestions[action.suggestionId]
    if (!suggestion) return state
    const status = action.type === 'acceptEditorSuggestion'
      ? 'accepted'
      : action.type === 'rejectEditorSuggestion'
        ? 'rejected'
        : 'stale'
    if (suggestion.status === status) return state
    const nextState: ProjectState = {
      ...state,
      dirty: true,
      editorSuggestions: {
        ...state.editorSuggestions,
        [suggestion.id]: { ...suggestion, status, updatedAt: new Date().toISOString() },
      },
    }
    if (action.type !== 'acceptEditorSuggestion' || !suggestion.origin.commentId) {
      return nextState
    }
    const comment = state.editorComments[suggestion.origin.commentId]
    if (!comment || comment.status === 'resolved') return nextState
    return {
      ...nextState,
      editorComments: {
        ...nextState.editorComments,
        [comment.id]: { ...comment, status: 'resolved', updatedAt: new Date().toISOString() },
      },
      editorUi: {
        ...nextState.editorUi,
        selectedCommentId: nextOpenEditorCommentId(nextState, comment.id),
      },
    }
  }
  if (action.type === 'updateEditorSuggestionProposal') {
    const suggestion = state.editorSuggestions[action.suggestionId]
    const proposedText = action.proposedText
    if (!suggestion || suggestion.status !== 'pending' || !proposedText.trim()) return state
    const now = new Date().toISOString()
    const previousRevision = suggestion.revision ?? 1
    const historyEntry = {
      changeSummary: suggestion.changeSummary,
      createdAt: now,
      instruction: action.instruction,
      proposedText: suggestion.proposedText,
      source: action.source,
      warnings: suggestion.warnings,
    }
    return {
      ...state,
      dirty: true,
      editorSuggestions: {
        ...state.editorSuggestions,
        [suggestion.id]: {
          ...suggestion,
          changeSummary: action.changeSummary?.length ? action.changeSummary : undefined,
          proposedText,
          revision: previousRevision + 1,
          revisionHistory: [...(suggestion.revisionHistory ?? []), historyEntry],
          updatedAt: now,
          warnings: action.warnings?.length ? action.warnings : undefined,
        },
      },
    }
  }
  if (action.type === 'acceptEditorSuggestionGroup' || action.type === 'rejectEditorSuggestionGroup') {
    const status = action.type === 'acceptEditorSuggestionGroup' ? 'accepted' : 'rejected'
    const now = new Date().toISOString()
    const editorSuggestions = { ...state.editorSuggestions }
    let changed = false
    for (const suggestion of Object.values(state.editorSuggestions)) {
      if (suggestion.groupId !== action.groupId || suggestion.status !== 'pending') continue
      editorSuggestions[suggestion.id] = { ...suggestion, status, updatedAt: now }
      changed = true
    }
    if (!changed) return state
    return { ...state, dirty: true, editorSuggestions }
  }
  if (action.type === 'selectEditorComment') {
    return {
      ...state,
      editorUi: {
        ...state.editorUi,
        isCommentPanelVisible: action.commentId ? true : state.editorUi.isCommentPanelVisible,
        panelTab: action.commentId ? 'comments' : state.editorUi.panelTab,
        selectedCommentId: action.commentId,
      },
    }
  }
  if (action.type === 'setEditorAssistantVisible') {
    return {
      ...state,
      editorUi: { ...state.editorUi, isAssistantVisible: action.isVisible },
    }
  }
  if (action.type === 'setEditorCommentPanelVisible') {
    return {
      ...state,
      editorUi: { ...state.editorUi, isCommentPanelVisible: action.isVisible },
    }
  }
  if (action.type === 'setEditorTreeVisible') {
    return {
      ...state,
      editorUi: { ...state.editorUi, isTreeVisible: action.isVisible },
    }
  }
  if (action.type === 'setEditorAssistantDraft') {
    return {
      ...state,
      dirty: true,
      editorUi: { ...state.editorUi, assistantDraft: action.draft },
    }
  }
  if (action.type === 'setEditorPanelTab') {
    return {
      ...state,
      editorUi: { ...state.editorUi, panelTab: action.tab },
    }
  }
  if (action.type === 'setEditorViewMode') {
    return {
      ...state,
      editorUi: { ...state.editorUi, viewMode: action.mode },
    }
  }
  if (action.type === 'setEditorDiffAnchor') {
    const document = state.editorDocuments[action.documentId]
    if (!document) return state
    const now = new Date().toISOString()
    return {
      ...state,
      dirty: true,
      editorDocuments: {
        ...state.editorDocuments,
        [document.id]: {
          ...document,
          diffAnchorMarkdown: document.contentMarkdown,
          diffAnchorUpdatedAt: now,
        },
      },
    }
  }
  if (action.type === 'setEditorDiffVisible') {
    return {
      ...state,
      editorUi: {
        ...state.editorUi,
        isDiffVisible: action.isVisible,
        viewMode: action.isVisible ? 'live' : state.editorUi.viewMode,
      },
    }
  }
  if (action.type === 'createLocalRun') {
    const run = createLocalResearchRun(
      action.request,
      state.localRunCounter,
      state.ui.selectedStack,
    )
    return {
      ...state,
      dirty: true,
      localRunCounter: state.localRunCounter + 1,
      researchRunOrder: [run.runId, ...state.researchRunOrder],
      researchRuns: {
        ...state.researchRuns,
        [run.runId]: run,
      },
      ui: {
        ...state.ui,
        activeFilter: 'all',
        expandedJobId: run.runId,
        selectedJobId: run.runId,
      },
    }
  }
  if (action.type === 'upsertApiRunSummary') {
    const current = state.researchRuns[action.summary.run_id]
    const run = mergeRunSummary(current, action.summary, state.ui.selectedStack)
    const researchRunOrder = state.researchRunOrder.includes(run.runId)
      ? state.researchRunOrder
      : [run.runId, ...state.researchRunOrder]
    const shouldSelect = action.select || state.ui.selectedJobId === null
    const selectedJobId = shouldSelect ? run.runId : state.ui.selectedJobId

    return {
      ...state,
      dirty: true,
      researchRunOrder,
      researchRuns: {
        ...state.researchRuns,
        [run.runId]: run,
      },
      ui: {
        ...state.ui,
        activeFilter: shouldSelect ? 'all' : state.ui.activeFilter,
        expandedJobId: shouldSelect ? run.runId : state.ui.expandedJobId,
        selectedJobId,
      },
    }
  }
  if (action.type === 'appendApiRunEvent') {
    const current = state.researchRuns[action.event.run_id]
    if (!current) return state
    const run = applyRunEvent(current, action.event)

    return {
      ...state,
      dirty: true,
      researchRuns: {
        ...state.researchRuns,
        [run.runId]: run,
      },
    }
  }
  if (action.type === 'attachApiRunResult') {
    const current = state.researchRuns[action.result.run_id]
    if (!current) return state
    const run = attachRunResult(current, action.result)

    return {
      ...state,
      dirty: true,
      researchRuns: {
        ...state.researchRuns,
        [run.runId]: run,
      },
    }
  }
  if (action.type === 'markApiRunError') {
    const current = state.researchRuns[action.runId]
    if (!current) return state

    return {
      ...state,
      dirty: true,
      researchRuns: {
        ...state.researchRuns,
        [action.runId]: {
          ...current,
          error: action.message,
          status: 'failed',
          summary: {
            ...current.summary,
            queueNote: action.message,
          },
        },
      },
    }
  }
  if (action.type === 'cancelLocalRun') {
    const current = state.researchRuns[action.runId]
    if (!current || (current.status !== 'running' && current.status !== 'queued')) return state
    const now = new Date().toISOString()
    const cancelRequested = current.status === 'running'
      ? [{
        active: false,
        createdAt: now,
        id: createId('event'),
        kind: 'system' as const,
        severity: 'warning' as const,
        title: 'Cancellation requested',
      }]
      : []
    const cancelledEvent = {
      createdAt: now,
      id: createId('event'),
      kind: 'system' as const,
      severity: 'warning' as const,
      title: 'Run cancelled',
    }

    return {
      ...state,
      dirty: true,
      researchRuns: {
        ...state.researchRuns,
        [action.runId]: {
          ...current,
          events: [
            ...current.events.map((event) => ({ ...event, active: false })),
            ...cancelRequested,
            cancelledEvent,
          ],
          finishedAt: now,
          queuePosition: null,
          status: 'cancelled',
          summary: {
            ...current.summary,
            queueNote: undefined,
          },
        },
      },
    }
  }
  if (action.type === 'selectChatThread') {
    return {
      ...state,
      ui: { ...state.ui, selectedChatThreadId: action.threadId },
    }
  }
  if (action.type === 'attachReportToChatDraft') {
    if (!reportFromRun(state, action.runId)) return state

    return {
      ...state,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: addPendingAttachmentRef(
          state.ui.pendingChatAttachmentRefs,
          { kind: 'research-report', runId: action.runId },
        ),
        pendingChatReportRunId: action.runId,
      },
    }
  }
  if (action.type === 'attachChatContextToDraft') {
    if (!contextRefExists(state, action.ref)) return state
    return {
      ...state,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: addPendingAttachmentRef(
          state.ui.pendingChatAttachmentRefs,
          action.ref,
        ),
        pendingChatReportRunId: action.ref.kind === 'research-report'
          ? action.ref.runId
          : state.ui.pendingChatReportRunId,
      },
    }
  }
  if (action.type === 'removeChatContextFromDraft') {
    const pendingChatAttachmentRefs = state.ui.pendingChatAttachmentRefs.filter((ref) => (
      chatContextRefKey(ref) !== chatContextRefKey(action.ref)
    ))
    return {
      ...state,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs,
        pendingChatReportRunId: pendingChatAttachmentRefs.find((ref) => ref.kind === 'research-report')?.runId ?? null,
      },
    }
  }
  if (action.type === 'reorderChatContextInDraft') {
    if (
      action.fromIndex === action.toIndex
      || action.fromIndex < 0
      || action.toIndex < 0
      || action.fromIndex >= state.ui.pendingChatAttachmentRefs.length
      || action.toIndex >= state.ui.pendingChatAttachmentRefs.length
    ) {
      return state
    }

    const pendingChatAttachmentRefs = moveItem(
      state.ui.pendingChatAttachmentRefs,
      action.fromIndex,
      action.toIndex,
    )
    return {
      ...state,
      dirty: true,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs,
        pendingChatReportRunId: pendingChatAttachmentRefs.find((ref) => ref.kind === 'research-report')?.runId ?? null,
      },
    }
  }
  if (action.type === 'attachReportToNewChat') {
    const report = reportFromRun(state, action.runId)
    if (!report) return state

    const thread = createChatThread({
      includeGreeting: false,
      preview: `Report attached: ${report.title}`,
      title: `Chat: ${report.title}`,
    })

    return {
      ...state,
      chatThreadOrder: [thread.id, ...state.chatThreadOrder],
      chatThreads: {
        ...state.chatThreads,
        [thread.id]: thread,
      },
      dirty: true,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: [{ kind: 'research-report', runId: action.runId }],
        activeView: 'chat',
        pendingChatReportRunId: action.runId,
        selectedChatThreadId: thread.id,
      },
    }
  }
  if (action.type === 'clearChatDraftAttachment') {
    return {
      ...state,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: [],
        pendingChatReportRunId: null,
      },
    }
  }
  if (action.type === 'ingestFileAssets') {
    const newIds = action.assets
      .filter((asset) => !state.fileAssets[asset.id])
      .map((asset) => asset.id)
    if (action.assets.length === 0) return state
    const fileAssets = { ...state.fileAssets }
    for (const asset of action.assets) {
      fileAssets[asset.id] = asset
    }
    return {
      ...state,
      dirty: true,
      fileAssetOrder: [...newIds, ...state.fileAssetOrder],
      fileAssets,
    }
  }
  if (action.type === 'renameFileAsset') {
    const asset = state.fileAssets[action.fileId]
    const label = action.label.trim()
    if (!asset || !label || asset.label === label) return state
    return {
      ...state,
      dirty: true,
      fileAssets: {
        ...state.fileAssets,
        [asset.id]: { ...asset, label, updatedAt: new Date().toISOString() },
      },
    }
  }
  if (action.type === 'moveFileAsset') {
    const asset = state.fileAssets[action.fileId]
    if (!asset || !state.fileLibrarySections[action.sectionId]) return state
    const targetGroup = action.groupId ? state.fileGroups[action.groupId] : null
    const groupId = targetGroup && targetGroup.sectionId === action.sectionId ? action.groupId : null
    if (asset.sectionId === action.sectionId && asset.groupId === groupId) return state
    return {
      ...state,
      dirty: true,
      fileAssets: {
        ...state.fileAssets,
        [asset.id]: { ...asset, groupId, sectionId: action.sectionId, updatedAt: new Date().toISOString() },
      },
    }
  }
  if (action.type === 'deleteFileAsset') {
    if (!state.fileAssets[action.fileId]) return state
    const fileAssets = { ...state.fileAssets }
    delete fileAssets[action.fileId]
    const keepRef = (ref: ChatContextReferenceRecord) => ref.kind !== 'file-asset' || ref.fileId !== action.fileId
    return {
      ...state,
      dirty: true,
      fileAssetOrder: state.fileAssetOrder.filter((fileId) => fileId !== action.fileId),
      fileAssets,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: state.ui.pendingChatAttachmentRefs.filter(keepRef),
      },
    }
  }
  if (action.type === 'createFileGroup') {
    if (!state.fileLibrarySections[action.sectionId]) return state
    const now = new Date().toISOString()
    const group: FileGroupRecord = {
      createdAt: now,
      id: createId('file-group'),
      sectionId: action.sectionId,
      title: action.title.trim() || 'Neue Gruppe',
      updatedAt: now,
    }
    return {
      ...state,
      dirty: true,
      fileGroupOrder: [group.id, ...state.fileGroupOrder],
      fileGroups: { ...state.fileGroups, [group.id]: group },
    }
  }
  if (action.type === 'renameFileGroup') {
    const group = state.fileGroups[action.groupId]
    const title = action.title.trim()
    if (!group || !title || group.title === title) return state
    return {
      ...state,
      dirty: true,
      fileGroups: {
        ...state.fileGroups,
        [group.id]: { ...group, title, updatedAt: new Date().toISOString() },
      },
    }
  }
  if (action.type === 'deleteFileGroup') {
    if (!state.fileGroups[action.groupId]) return state
    const fileGroups = { ...state.fileGroups }
    delete fileGroups[action.groupId]
    const now = new Date().toISOString()
    const fileAssets = { ...state.fileAssets }
    for (const asset of Object.values(state.fileAssets)) {
      if (asset.groupId === action.groupId) {
        fileAssets[asset.id] = { ...asset, groupId: null, updatedAt: now }
      }
    }
    return {
      ...state,
      dirty: true,
      fileAssets,
      fileGroupOrder: state.fileGroupOrder.filter((groupId) => groupId !== action.groupId),
      fileGroups,
    }
  }
  if (action.type === 'renameFileLibrarySection') {
    const section = state.fileLibrarySections[action.sectionId]
    const title = action.title.trim()
    if (!section || !title || section.title === title) return state
    return {
      ...state,
      dirty: true,
      fileLibrarySections: {
        ...state.fileLibrarySections,
        [section.id]: { ...section, title, updatedAt: new Date().toISOString() },
      },
    }
  }
  if (action.type === 'upsertChatRule') {
    const now = new Date().toISOString()
    const existing = state.chatRules[action.rule.id]
    const rule = {
      ...action.rule,
      createdAt: existing?.createdAt ?? action.rule.createdAt,
      updatedAt: action.rule.updatedAt || now,
    }
    const chatRuleOrder = state.chatRuleOrder.includes(rule.id)
      ? state.chatRuleOrder
      : [rule.id, ...state.chatRuleOrder]
    return {
      ...state,
      chatRuleOrder,
      chatRules: {
        ...state.chatRules,
        [rule.id]: rule,
      },
      dirty: true,
    }
  }
  if (action.type === 'deleteChatRule') {
    if (!state.chatRules[action.ruleId]) return state
    const chatRules = { ...state.chatRules }
    delete chatRules[action.ruleId]
    return {
      ...state,
      chatRuleOrder: state.chatRuleOrder.filter((ruleId) => ruleId !== action.ruleId),
      chatRules,
      dirty: true,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: state.ui.pendingChatAttachmentRefs.filter((ref) => (
          ref.kind !== 'chat-rule' || ref.ruleId !== action.ruleId
        )),
      },
    }
  }
  if (action.type === 'createChatThreadGroup') {
    const now = new Date().toISOString()
    const group: ChatThreadGroupRecord = {
      createdAt: now,
      id: createId('chat-group'),
      title: action.title.trim() || 'New group',
      updatedAt: now,
    }
    return {
      ...state,
      chatThreadGroupOrder: [group.id, ...state.chatThreadGroupOrder],
      chatThreadGroups: {
        ...state.chatThreadGroups,
        [group.id]: group,
      },
      dirty: true,
    }
  }
  if (action.type === 'renameChatThreadGroup') {
    const group = state.chatThreadGroups[action.groupId]
    const title = action.title.trim()
    if (!group || !title || group.title === title) return state
    return {
      ...state,
      chatThreadGroups: {
        ...state.chatThreadGroups,
        [group.id]: {
          ...group,
          title,
          updatedAt: new Date().toISOString(),
        },
      },
      dirty: true,
    }
  }
  if (action.type === 'deleteChatThreadGroup') {
    if (!state.chatThreadGroups[action.groupId]) return state
    const chatThreadGroups = { ...state.chatThreadGroups }
    delete chatThreadGroups[action.groupId]
    const chatThreadGroupMemberships = Object.fromEntries(
      Object.entries(state.chatThreadGroupMemberships).filter(([, groupId]) => groupId !== action.groupId),
    )
    return {
      ...state,
      chatThreadGroupMemberships,
      chatThreadGroupOrder: state.chatThreadGroupOrder.filter((groupId) => groupId !== action.groupId),
      chatThreadGroups,
      dirty: true,
    }
  }
  if (action.type === 'moveChatThreadGroup') {
    return moveChatThreadGroup(state, action.groupId, action.targetIndex)
  }
  if (action.type === 'moveChatThreadToGroup') {
    return moveChatThreadToGroup(state, action.threadId, action.groupId, action.targetIndex)
  }
  if (action.type === 'createChatThread') {
    const thread = createChatThread()
    const groupId = action.groupId && state.chatThreadGroups[action.groupId]
      ? action.groupId
      : null
    const chatThreadGroupMemberships = groupId
      ? { ...state.chatThreadGroupMemberships, [thread.id]: groupId }
      : state.chatThreadGroupMemberships
    const chatThreadOrder = groupId
      ? insertThreadIntoSection(state, chatThreadGroupMemberships, thread.id, groupId, 0)
      : [thread.id, ...state.chatThreadOrder]

    return {
      ...state,
      chatThreadGroupMemberships,
      chatThreadOrder,
      chatThreads: {
        ...state.chatThreads,
        [thread.id]: thread,
      },
      dirty: true,
      ui: {
        ...state.ui,
        selectedChatThreadId: thread.id,
      },
    }
  }
  if (action.type === 'renameChatThread') {
    const thread = state.chatThreads[action.threadId]
    const title = action.title.trim()
    if (!thread || !title || thread.title === title) return state

    const updatedAt = new Date().toISOString()
    return {
      ...state,
      chatThreads: {
        ...state.chatThreads,
        [thread.id]: {
          ...thread,
          title,
          updatedAt,
        },
      },
      dirty: true,
    }
  }
  if (action.type === 'clearChatThread') {
    const thread = state.chatThreads[action.threadId]
    if (!thread) return state
    const updatedAt = new Date().toISOString()
    return {
      ...state,
      chatThreads: {
        ...state.chatThreads,
        [thread.id]: {
          ...thread,
          messages: [],
          preview: 'No user message yet',
          updatedAt,
        },
      },
      dirty: true,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: [],
        pendingChatReportRunId: null,
      },
    }
  }
  if (action.type === 'deleteChatMessages') {
    return deleteChatMessages(state, action.threadId, action.messageIds)
  }
  if (action.type === 'editChatUserMessage') {
    return editChatUserMessage(state, action.threadId, action.messageId, action.contentMarkdown)
  }
  if (action.type === 'branchChatThreadFromMessage') {
    return branchChatThreadFromMessage(state, action.threadId, action.messageId)
  }
  if (action.type === 'startChatExchange') {
    return startChatExchange(state, action)
  }
  if (action.type === 'startChatAssistantResponse') {
    return startChatAssistantResponse(state, action)
  }
  if (action.type === 'setChatAssistantMessageContent') {
    return setChatAssistantMessageContent(state, action)
  }
  if (action.type === 'deleteChatThread') {
    const chatThreads = { ...state.chatThreads }
    delete chatThreads[action.threadId]
    const chatThreadGroupMemberships = { ...state.chatThreadGroupMemberships }
    delete chatThreadGroupMemberships[action.threadId]
    const chatThreadOrder = state.chatThreadOrder.filter((threadId) => threadId !== action.threadId)
    const selectedChatThreadId = state.ui.selectedChatThreadId === action.threadId
      ? chatThreadOrder[0] ?? null
      : state.ui.selectedChatThreadId

    return {
      ...state,
      chatThreadGroupMemberships,
      chatThreadOrder,
      chatThreads,
      dirty: true,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: [],
        pendingChatReportRunId: null,
        selectedChatThreadId,
      },
    }
  }

  return state
}

function deleteChatMessages(
  state: ResearchDeskState,
  threadId: string,
  messageIds: readonly string[],
): ResearchDeskState {
  const thread = state.chatThreads[threadId]
  if (!thread || messageIds.length === 0) return state
  const messageIdSet = new Set(messageIds)
  const messages = thread.messages.filter((message) => !messageIdSet.has(message.id))
  if (messages.length === thread.messages.length) return state

  const updatedAt = new Date().toISOString()
  return {
    ...state,
    chatThreads: {
      ...state.chatThreads,
      [thread.id]: threadWithMessages(thread, messages, updatedAt),
    },
    dirty: true,
  }
}

function editChatUserMessage(
  state: ResearchDeskState,
  threadId: string,
  messageId: string,
  contentMarkdown: string,
): ResearchDeskState {
  const thread = state.chatThreads[threadId]
  const nextContent = contentMarkdown.trim()
  if (!thread || !nextContent) return state
  const currentMessage = thread.messages.find((message) => message.id === messageId)
  if (!currentMessage || currentMessage.role !== 'user' || currentMessage.contentMarkdown === nextContent) {
    return state
  }

  const updatedAt = new Date().toISOString()
  const messages = thread.messages.map((message) => (
    message.id === messageId
      ? { ...message, contentMarkdown: nextContent }
      : message
  ))
  const autoTitle = titleFromMessage(currentMessage.contentMarkdown)
  const nextTitle = thread.title === autoTitle
    ? titleFromMessage(nextContent)
    : thread.title

  return {
    ...state,
    chatThreads: {
      ...state.chatThreads,
      [thread.id]: {
        ...threadWithMessages(thread, messages, updatedAt),
        title: nextTitle,
      },
    },
    dirty: true,
  }
}

function branchChatThreadFromMessage(
  state: ResearchDeskState,
  threadId: string,
  messageId: string,
): ResearchDeskState {
  const sourceThread = state.chatThreads[threadId]
  if (!sourceThread) return state
  const messageIndex = sourceThread.messages.findIndex((message) => message.id === messageId)
  if (messageIndex === -1) return state

  const messages = sourceThread.messages.slice(0, messageIndex + 1).map(cloneChatMessage)
  const now = new Date().toISOString()
  const firstUserMessage = messages.find((message) => message.role === 'user')
  const thread: ChatThreadRecord = threadWithMessages(
    {
      createdAt: now,
      id: createId('chat'),
      messages,
      preview: 'No user message yet',
      source: 'api',
      title: firstUserMessage
        ? titleFromMessage(firstUserMessage.contentMarkdown)
        : sourceThread.title,
      updatedAt: now,
    },
    messages,
    now,
  )

  return {
    ...state,
    chatThreadOrder: [thread.id, ...state.chatThreadOrder],
    chatThreads: {
      ...state.chatThreads,
      [thread.id]: thread,
    },
    dirty: true,
    ui: {
      ...state.ui,
      selectedChatThreadId: thread.id,
    },
  }
}

function cloneChatMessage(message: ChatMessageRecord): ChatMessageRecord {
  return {
    ...message,
    attachments: message.attachments ? message.attachments.map((attachment) => ({ ...attachment })) : undefined,
    id: createId('msg'),
  }
}

function threadWithMessages(
  thread: ChatThreadRecord,
  messages: ChatMessageRecord[],
  updatedAt: string,
): ChatThreadRecord {
  return {
    ...thread,
    messages,
    preview: chatPreviewFromMessages(messages),
    updatedAt,
  }
}

function chatPreviewFromMessages(messages: readonly ChatMessageRecord[]) {
  return [...messages].reverse().find((message) => message.role === 'user')?.contentMarkdown ?? 'No user message yet'
}

function moveChatThreadGroup(
  state: ResearchDeskState,
  groupId: string,
  targetIndex: number,
): ResearchDeskState {
  if (!state.chatThreadGroups[groupId]) return state
  const orderWithoutGroup = state.chatThreadGroupOrder.filter((candidateId) => candidateId !== groupId)
  const boundedTargetIndex = Math.max(0, Math.min(orderWithoutGroup.length, targetIndex))
  const chatThreadGroupOrder = [...orderWithoutGroup]
  chatThreadGroupOrder.splice(boundedTargetIndex, 0, groupId)

  if (arraysEqual(chatThreadGroupOrder, state.chatThreadGroupOrder)) return state

  return {
    ...state,
    chatThreadGroupOrder,
    dirty: true,
  }
}

function moveChatThreadToGroup(
  state: ResearchDeskState,
  threadId: string,
  requestedGroupId: string | null,
  targetIndex: number,
): ResearchDeskState {
  if (!state.chatThreads[threadId]) return state
  const groupId = requestedGroupId && state.chatThreadGroups[requestedGroupId]
    ? requestedGroupId
    : null
  const currentGroupId = normalizedThreadGroupId(state, threadId)
  const currentSectionThreadIds = threadIdsForGroup(state, currentGroupId)
  const currentIndex = currentSectionThreadIds.indexOf(threadId)
  const targetThreadIds = threadIdsForGroup(state, groupId).filter((id) => id !== threadId)
  const boundedTargetIndex = Math.max(0, Math.min(targetThreadIds.length, targetIndex))
  if (currentGroupId === groupId && currentIndex === boundedTargetIndex) return state

  const chatThreadGroupMemberships = { ...state.chatThreadGroupMemberships }
  if (groupId) {
    chatThreadGroupMemberships[threadId] = groupId
  } else {
    delete chatThreadGroupMemberships[threadId]
  }

  return {
    ...state,
    chatThreadGroupMemberships,
    chatThreadOrder: insertThreadIntoSection(
      state,
      chatThreadGroupMemberships,
      threadId,
      groupId,
      boundedTargetIndex,
    ),
    dirty: true,
  }
}

function insertThreadIntoSection(
  state: ResearchDeskState,
  memberships: Record<string, string | null>,
  threadId: string,
  groupId: string | null,
  targetIndex: number,
) {
  const orderWithoutThread = state.chatThreadOrder.filter((id) => id !== threadId)
  const targetThreadIds = orderWithoutThread.filter((id) => normalizedThreadGroupId(state, id, memberships) === groupId)
  const beforeThreadId = targetThreadIds[targetIndex]
  if (beforeThreadId) {
    return insertBefore(orderWithoutThread, threadId, beforeThreadId)
  }

  const previousThreadId = targetThreadIds[targetIndex - 1]
  if (previousThreadId) {
    return insertAfter(orderWithoutThread, threadId, previousThreadId)
  }

  const sectionInsertionIndex = emptySectionInsertionIndex(state, orderWithoutThread, memberships, groupId)
  const nextOrder = [...orderWithoutThread]
  nextOrder.splice(sectionInsertionIndex, 0, threadId)
  return nextOrder
}

function emptySectionInsertionIndex(
  state: ResearchDeskState,
  orderWithoutThread: string[],
  memberships: Record<string, string | null>,
  groupId: string | null,
) {
  const sectionKeys = [
    ...state.chatThreadGroupOrder.filter((candidateId) => Boolean(state.chatThreadGroups[candidateId])),
    null,
  ]
  const targetSectionIndex = Math.max(0, sectionKeys.findIndex((candidateId) => candidateId === groupId))
  for (const nextGroupId of sectionKeys.slice(targetSectionIndex + 1)) {
    const nextThreadId = orderWithoutThread.find((threadId) => (
      normalizedThreadGroupId(state, threadId, memberships) === nextGroupId
    ))
    if (nextThreadId) return orderWithoutThread.indexOf(nextThreadId)
  }

  for (const previousGroupId of sectionKeys.slice(0, targetSectionIndex).reverse()) {
    const previousThreadId = [...orderWithoutThread].reverse().find((threadId) => (
      normalizedThreadGroupId(state, threadId, memberships) === previousGroupId
    ))
    if (previousThreadId) return orderWithoutThread.indexOf(previousThreadId) + 1
  }

  return orderWithoutThread.length
}

function threadIdsForGroup(
  state: ResearchDeskState,
  groupId: string | null,
) {
  return state.chatThreadOrder.filter((threadId) => (
    normalizedThreadGroupId(state, threadId) === groupId
  ))
}

function normalizedThreadGroupId(
  state: ResearchDeskState,
  threadId: string,
  memberships = state.chatThreadGroupMemberships,
) {
  const groupId = memberships[threadId]
  return groupId && state.chatThreadGroups[groupId] ? groupId : null
}

function insertBefore(items: string[], item: string, beforeItem: string) {
  const next = [...items]
  const index = next.indexOf(beforeItem)
  next.splice(index < 0 ? next.length : index, 0, item)
  return next
}

function insertAfter(items: string[], item: string, afterItem: string) {
  const next = [...items]
  const index = next.indexOf(afterItem)
  next.splice(index < 0 ? next.length : index + 1, 0, item)
  return next
}

function arraysEqual(left: readonly string[], right: readonly string[]) {
  return left.length === right.length && left.every((item, index) => item === right[index])
}

function createLocalResearchRun(
  request: CreateResearchRunRequest,
  counter: number,
  fallbackStack: string,
): ResearchRunRecord {
  const now = new Date().toISOString()
  const maxRounds = request.agentOverrides?.maxRounds ?? 10
  const firstRoundQueries = request.agentOverrides?.firstRoundQueries ?? 0
  const question = request.question.trim()

  return {
    agentOverrides: request.agentOverrides ?? {},
    createdAt: now,
    events: [],
    metrics: {
      queries: firstRoundQueries,
      rounds: `0 / ${maxRounds}`,
      sources: 0,
    },
    phaseState: {
      activePhase: 'analysis',
      completedPhases: [],
    },
    runId: `RO-${counter.toString().padStart(4, '0')}`,
    source: 'mock',
    stack: request.stack ?? fallbackStack,
    status: 'queued',
    submittedAt: now,
    summary: {
      queueNote: 'Local draft',
      title: question,
    },
  }
}

function createChatThread(options: {
  id?: string
  includeGreeting?: boolean
  preview?: string
  source?: ChatThreadRecord['source']
  title?: string
} = {}): ChatThreadRecord {
  const now = new Date().toISOString()
  const includeGreeting = options.includeGreeting ?? true

  return {
    createdAt: now,
    id: options.id ?? createId('chat'),
    messages: includeGreeting ? [
      {
        contentMarkdown: 'New conversation ready. Ask a question or sketch the research you want to derive from it.',
        createdAt: now,
        id: createId('msg'),
        role: 'assistant',
      },
    ] : [],
    preview: options.preview ?? 'No user message yet',
    source: options.source ?? 'api',
    title: options.title ?? 'New conversation',
    updatedAt: now,
  }
}

function startChatExchange(
  state: ResearchDeskState,
  action: Extract<ResearchDeskAction, { type: 'startChatExchange' }>,
): ResearchDeskState {
  const trimmedContent = action.contentMarkdown.trim()
  if (!trimmedContent) return state

  const thread = state.chatThreads[action.threadId] ?? createChatThread({
    id: action.threadId,
    includeGreeting: false,
    preview: trimmedContent,
    source: 'api',
    title: titleFromMessage(trimmedContent),
  })
  const isNewThread = !state.chatThreads[action.threadId]
  const report = state.ui.pendingChatReportRunId
    ? reportFromRun(state, state.ui.pendingChatReportRunId)
    : null
  const attachmentRefs = dedupePendingAttachmentRefs([
    ...state.ui.pendingChatAttachmentRefs,
    ...(action.attachmentRefs ?? []),
  ])
  const attachments = createMessageAttachments(state, attachmentRefs, action.createdAt)
  const legacyAttachments = attachments.length > 0
    ? attachments
    : report
      ? [createReportAttachment(report, action.createdAt)]
      : undefined
  const userMessage = {
    attachments: legacyAttachments,
    contentMarkdown: trimmedContent,
    createdAt: action.createdAt,
    id: action.userMessageId,
    role: 'user' as const,
  }
  const assistantMessage = {
    contentMarkdown: '',
    createdAt: action.createdAt,
    id: action.assistantMessageId,
    modelResolution: action.modelResolution,
    role: 'assistant' as const,
  }
  const nextThread: ChatThreadRecord = {
    ...thread,
    messages: [...thread.messages, userMessage, assistantMessage],
    preview: trimmedContent,
    source: 'api',
    title: thread.messages.some((message) => message.role === 'user')
      ? thread.title
      : titleFromMessage(trimmedContent),
    updatedAt: action.createdAt,
  }

  return {
    ...state,
    chatThreadOrder: isNewThread
      ? [nextThread.id, ...state.chatThreadOrder]
      : state.chatThreadOrder,
    chatThreads: {
      ...state.chatThreads,
      [nextThread.id]: nextThread,
    },
    dirty: true,
    ui: {
      ...state.ui,
      pendingChatAttachmentRefs: [],
      pendingChatReportRunId: null,
      selectedChatThreadId: nextThread.id,
    },
  }
}

function startChatAssistantResponse(
  state: ResearchDeskState,
  action: Extract<ResearchDeskAction, { type: 'startChatAssistantResponse' }>,
): ResearchDeskState {
  const thread = state.chatThreads[action.threadId]
  const lastMessage = thread?.messages[thread.messages.length - 1]
  if (!thread || !lastMessage || lastMessage.id !== action.userMessageId || lastMessage.role !== 'user') {
    return state
  }

  const assistantMessage: ChatMessageRecord = {
    contentMarkdown: '',
    createdAt: action.createdAt,
    id: action.assistantMessageId,
    modelResolution: action.modelResolution,
    role: 'assistant',
  }

  return {
    ...state,
    chatThreads: {
      ...state.chatThreads,
      [thread.id]: threadWithMessages(thread, [...thread.messages, assistantMessage], action.createdAt),
    },
    dirty: true,
  }
}

function setChatAssistantMessageContent(
  state: ResearchDeskState,
  action: Extract<ResearchDeskAction, { type: 'setChatAssistantMessageContent' }>,
): ResearchDeskState {
  const thread = state.chatThreads[action.threadId]
  if (!thread) return state

  const messageIndex = thread.messages.findIndex((message) => message.id === action.assistantMessageId)
  if (messageIndex === -1) return state

  const messages = thread.messages.map((message) => (
    message.id === action.assistantMessageId
      ? {
        ...message,
        chainTrace: action.chainTrace ?? message.chainTrace,
        contentMarkdown: action.contentMarkdown,
        modelResolution: action.modelResolution ?? message.modelResolution,
      }
      : message
  ))
  const updatedAt = new Date().toISOString()

  return {
    ...state,
    chatThreads: {
      ...state.chatThreads,
      [thread.id]: {
        ...thread,
        messages,
        updatedAt,
      },
    },
    dirty: true,
  }
}

function createEditorDocument(options: {
  contentMarkdown?: string
  folderId?: string | null
  source: EditorDocumentRecord['source']
  sourceRunId?: string
  title?: string
}): EditorDocumentRecord {
  const now = new Date().toISOString()
  return {
    contentMarkdown: options.contentMarkdown ?? '',
    createdAt: now,
    folderId: options.folderId ?? null,
    id: createId('editor-doc'),
    revision: 1,
    source: options.source,
    sourceRunId: options.sourceRunId,
    title: normalizeEditorDocumentTitle(options.title ?? 'Untitled.md'),
    updatedAt: now,
  }
}

function withEditorDocument(
  state: ResearchDeskState,
  document: EditorDocumentRecord,
  options: { activeView?: AppView } = {},
): ResearchDeskState {
  return {
    ...state,
    dirty: true,
    editorDocumentOrder: [document.id, ...state.editorDocumentOrder],
    editorDocuments: {
      ...state.editorDocuments,
      [document.id]: document,
    },
    editorUi: {
      ...state.editorUi,
      activeDocumentId: document.id,
      openDocumentIds: addOpenEditorDocumentId(state.editorUi.openDocumentIds, document.id),
      viewMode: 'live',
    },
    ui: {
      ...state.ui,
      activeView: options.activeView ?? state.ui.activeView,
    },
  }
}

function moveEditorFolder(
  state: ResearchDeskState,
  folderId: string,
  targetIndex: number,
): ResearchDeskState {
  if (!state.editorFolders[folderId]) return state
  const orderWithoutFolder = state.editorFolderOrder.filter((candidateId) => candidateId !== folderId)
  const boundedTargetIndex = Math.max(0, Math.min(orderWithoutFolder.length, targetIndex))
  const editorFolderOrder = [...orderWithoutFolder]
  editorFolderOrder.splice(boundedTargetIndex, 0, folderId)

  if (arraysEqual(editorFolderOrder, state.editorFolderOrder)) return state

  return {
    ...state,
    dirty: true,
    editorFolderOrder,
  }
}

function moveEditorDocumentToFolder(
  state: ResearchDeskState,
  documentId: string,
  requestedFolderId: string | null,
  targetIndex: number,
): ResearchDeskState {
  const document = state.editorDocuments[documentId]
  if (!document) return state
  const folderId = requestedFolderId && state.editorFolders[requestedFolderId]
    ? requestedFolderId
    : null
  const currentFolderId = document.folderId && state.editorFolders[document.folderId]
    ? document.folderId
    : null
  const currentSectionDocumentIds = documentIdsForEditorFolder(state.editorDocumentOrder, state.editorDocuments, currentFolderId)
  const currentIndex = currentSectionDocumentIds.indexOf(documentId)
  const targetDocumentIds = documentIdsForEditorFolder(state.editorDocumentOrder, state.editorDocuments, folderId)
    .filter((id) => id !== documentId)
  const boundedTargetIndex = Math.max(0, Math.min(targetDocumentIds.length, targetIndex))
  if (currentFolderId === folderId && currentIndex === boundedTargetIndex) return state

  return {
    ...state,
    dirty: true,
    editorDocumentOrder: moveSectionDocumentIds(
      state.editorDocumentOrder,
      state.editorDocuments,
      currentFolderId,
      folderId,
      boundedTargetIndex,
      documentId,
    ),
    editorDocuments: {
      ...state.editorDocuments,
      [document.id]: {
        ...document,
        folderId,
        updatedAt: new Date().toISOString(),
      },
    },
  }
}

function moveSectionDocumentIds(
  documentOrder: readonly string[],
  documents: ResearchDeskState['editorDocuments'],
  currentFolderId: string | null,
  targetFolderId: string | null,
  targetIndex: number,
  movedDocumentId?: string,
): string[] {
  const movedIds = movedDocumentId
    ? [movedDocumentId]
    : documentOrder.filter((documentId) => documents[documentId]?.folderId === currentFolderId)
  const orderWithoutMovedIds = documentOrder.filter((documentId) => !movedIds.includes(documentId))
  const targetSectionIds = orderWithoutMovedIds.filter((documentId) => {
    const document = documents[documentId]
    return Boolean(document) && (document.folderId ?? null) === targetFolderId
  })
  const boundedTargetIndex = Math.max(0, Math.min(targetSectionIds.length, targetIndex))
  const anchorId = targetSectionIds[boundedTargetIndex] ?? null
  const insertionIndex = anchorId
    ? orderWithoutMovedIds.indexOf(anchorId)
    : lastSectionInsertionIndex(orderWithoutMovedIds, documents, targetFolderId)
  const nextOrder = [...orderWithoutMovedIds]
  nextOrder.splice(insertionIndex, 0, ...movedIds)
  return nextOrder
}

function lastSectionInsertionIndex(
  documentOrder: readonly string[],
  documents: ResearchDeskState['editorDocuments'],
  folderId: string | null,
) {
  const sectionIndexes = documentOrder.flatMap((documentId, index) => (
    (documents[documentId]?.folderId ?? null) === folderId ? [index] : []
  ))
  return sectionIndexes.length > 0
    ? Math.max(...sectionIndexes) + 1
    : documentOrder.length
}

function documentIdsForEditorFolder(
  documentOrder: readonly string[],
  documents: ResearchDeskState['editorDocuments'],
  folderId: string | null,
) {
  return documentOrder.filter((documentId) => (documents[documentId]?.folderId ?? null) === folderId)
}

function addOpenEditorDocumentId(openDocumentIds: readonly string[], documentId: string) {
  return openDocumentIds.includes(documentId)
    ? [...openDocumentIds]
    : [...openDocumentIds, documentId]
}

function normalizeEditorDocumentTitle(value: string) {
  const normalized = value.replace(/\s+/g, ' ').trim() || 'Untitled'
  return normalized.endsWith('.md') ? normalized : `${normalized}.md`
}

function reportFromRun(state: ResearchDeskState, runId: string) {
  const run = state.researchRuns[runId]
  const markdown = run?.result?.markdown
  if (!run || run.status !== 'completed' || !markdown) return null

  return {
    contentMarkdown: markdown,
    label: reportLabel(run.summary.title, run.runId),
    runId: run.runId,
    title: run.summary.title,
  }
}

function createReportAttachment(
  report: {
    contentMarkdown: string
    label?: string
    runId: string
    title: string
  },
  attachedAt: string,
): ChatMessageAttachmentRecord {
  return {
    attachedAt,
    contentMarkdown: report.contentMarkdown,
    kind: 'research-report',
    label: report.label,
    runId: report.runId,
    title: report.title,
  }
}

function createRuleAttachment(
  rule: ChatRuleRecord,
  attachedAt: string,
): ChatMessageAttachmentRecord {
  return {
    attachedAt,
    contentMarkdown: rule.contentMarkdown,
    kind: 'chat-rule',
    label: rule.label,
    ruleId: rule.id,
    title: rule.title,
  }
}

function createMessageAttachments(
  state: ResearchDeskState,
  refs: ChatContextReferenceRecord[],
  attachedAt: string,
): ChatMessageAttachmentRecord[] {
  const seen = new Set<string>()
  return refs.flatMap<ChatMessageAttachmentRecord>((ref) => {
    if (ref.kind === 'file-group') {
      const group = state.fileGroups[ref.groupId]
      if (!group) return []
      const members = state.fileAssetOrder
        .map((fileId) => state.fileAssets[fileId])
        .filter((asset): asset is FileAssetRecord => Boolean(asset))
        .filter((asset) => asset.groupId === ref.groupId)
      return members.flatMap<ChatMessageAttachmentRecord>((asset) => {
        const memberKey = `file-asset:${asset.id}`
        if (seen.has(memberKey)) return []
        seen.add(memberKey)
        return [{
          attachedAt,
          contentMarkdown: asset.extractedText,
          fileId: asset.id,
          groupId: group.id,
          groupLabel: group.title,
          kind: 'file-group' as const,
          label: asset.label,
          pageCount: asset.pageCount,
          sizeBytes: asset.sizeBytes,
          title: asset.title,
        }]
      })
    }
    const key = chatContextRefKey(ref)
    if (seen.has(key)) return []
    seen.add(key)
    if (ref.kind === 'research-report') {
      const report = reportFromRun(state, ref.runId)
      return report ? [createReportAttachment(report, attachedAt)] : []
    }
    if (ref.kind === 'file-asset') {
      const asset = state.fileAssets[ref.fileId]
      if (!asset) return []
      return [{
        attachedAt,
        contentMarkdown: asset.extractedText,
        fileId: asset.id,
        kind: 'file-asset' as const,
        label: asset.label,
        pageCount: asset.pageCount,
        sizeBytes: asset.sizeBytes,
        title: asset.title,
      }]
    }
    const rule = state.chatRules[ref.ruleId]
    return rule ? [createRuleAttachment(rule, attachedAt)] : []
  })
}

function contextRefExists(
  state: ResearchDeskState,
  ref: ChatContextReferenceRecord,
) {
  switch (ref.kind) {
    case 'research-report':
      return Boolean(reportFromRun(state, ref.runId))
    case 'file-asset':
      return Boolean(state.fileAssets[ref.fileId])
    case 'file-group':
      return Boolean(state.fileGroups[ref.groupId])
    case 'chat-rule':
      return Boolean(state.chatRules[ref.ruleId])
  }
}

function addPendingAttachmentRef(
  current: ChatContextReferenceRecord[],
  ref: ChatContextReferenceRecord,
) {
  return dedupePendingAttachmentRefs([...current, ref])
}

function dedupePendingAttachmentRefs(refs: ChatContextReferenceRecord[]) {
  const seen = new Set<string>()
  return refs.filter((ref) => {
    const key = chatContextRefKey(ref)
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

function chatContextRefKey(ref: ChatContextReferenceRecord) {
  switch (ref.kind) {
    case 'research-report':
      return `research-report:${ref.runId}`
    case 'file-asset':
      return `file-asset:${ref.fileId}`
    case 'file-group':
      return `file-group:${ref.groupId}`
    case 'chat-rule':
      return `chat-rule:${ref.ruleId}`
  }
}

function titleFromMessage(contentMarkdown: string) {
  return contentMarkdown
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 72) || 'New conversation'
}

function reportLabel(title: string, runId: string) {
  const normalized = title
    .normalize('NFKD')
    .toLowerCase()
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)
  if (normalized) return normalized
  return runId.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 48) || 'report'
}

function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function nextOpenEditorCommentId(state: ProjectState, currentCommentId: string) {
  const current = state.editorComments[currentCommentId]
  if (!current) return null
  const ordered = Object.values(state.editorComments)
    .filter((comment) => comment.documentId === current.documentId && comment.status !== 'resolved')
    .sort((a, b) => a.anchor.from - b.anchor.from || a.createdAt.localeCompare(b.createdAt))
  const currentIndex = ordered.findIndex((comment) => comment.id === currentCommentId)
  if (currentIndex < 0) return ordered[0]?.id ?? null
  return ordered[currentIndex + 1]?.id ?? ordered[currentIndex - 1]?.id ?? null
}

function retireActiveEditorSuggestionsForComments(
  suggestions: ProjectState['editorSuggestions'],
  commentIds: Set<string>,
  updatedAt: string,
): ProjectState['editorSuggestions'] {
  if (commentIds.size === 0) return suggestions
  let changed = false
  const next = Object.fromEntries(
    Object.entries(suggestions).map(([suggestionId, suggestion]) => {
      const commentId = suggestion.origin.commentId
      const shouldRetire = commentId
        && commentIds.has(commentId)
        && (suggestion.status === 'pending' || suggestion.status === 'stale')
      if (!shouldRetire) return [suggestionId, suggestion]
      changed = true
      return [suggestionId, { ...suggestion, status: 'rejected' as const, updatedAt }]
    }),
  )
  return changed ? next : suggestions
}

function retireActiveDocumentInstructionSuggestions(
  suggestions: ProjectState['editorSuggestions'],
  documentId: string,
  updatedAt: string,
): ProjectState['editorSuggestions'] {
  let changed = false
  const next = Object.fromEntries(
    Object.entries(suggestions).map(([suggestionId, suggestion]) => {
      const shouldRetire = suggestion.documentId === documentId
        && suggestion.origin.kind === 'global_run'
        && !suggestion.origin.commentId
        && (suggestion.status === 'pending' || suggestion.status === 'stale')
      if (!shouldRetire) return [suggestionId, suggestion]
      changed = true
      return [suggestionId, { ...suggestion, status: 'rejected' as const, updatedAt }]
    }),
  )
  return changed ? next : suggestions
}

export function visibleResearchJobs(
  jobs: ResearchJob[],
  activeFilter: JobFilter,
) {
  return activeFilter === 'all'
    ? jobs
    : jobs.filter((job) => job.status === activeFilter)
}

export function selectedResearchJob(
  jobs: ResearchJob[],
  selectedJobId: string | null,
) {
  return jobs.find((job) => job.id === selectedJobId) ?? null
}

export function resolveVisibleSelectionFromJobs(
  jobs: ResearchJob[],
  activeFilter: JobFilter,
  selectedJobId: string | null,
) {
  const visibleJobs = visibleResearchJobs(jobs, activeFilter)
  const selectedJobIsVisible = visibleJobs.some((job) => job.id === selectedJobId)

  return selectedJobIsVisible ? selectedJobId : visibleJobs[0]?.id ?? null
}

function resolveVisibleSelection(
  researchRunOrder: string[],
  researchRuns: Record<string, ResearchRunRecord>,
  activeFilter: JobFilter,
  selectedJobId: string | null,
) {
  const visibleRunIds = activeFilter === 'all'
    ? researchRunOrder
    : researchRunOrder.filter((runId) => researchRuns[runId]?.status === activeFilter)
  const selectedJobIsVisible = visibleRunIds.some((runId) => runId === selectedJobId)

  return selectedJobIsVisible ? selectedJobId : visibleRunIds[0] ?? null
}
