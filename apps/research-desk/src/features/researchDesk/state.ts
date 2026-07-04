import type {
  ChatModelTier,
  CreateResearchRunRequest,
  ResearchRunEvent,
  ResearchRunResult,
  ResearchRunSummary,
} from '@/features/researchRuns/types'
import { createEmptyProjectState, createSeedProjectState } from '@/features/project/seedProject'
import { normalizeChatRule } from '@/features/project/chatRules'
import { renderChatRuleAttachmentContent } from '@/features/project/chatRuleRendering'
import type {
  ChatChainStepRecord,
  ChatContextReferenceRecord,
  ChatMessageRecord,
  ChatMessageAttachmentRecord,
  ChatMessageModelResolutionRecord,
  ChatMessageRequestContextRecord,
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
  EmbedModelId,
  FileAssetRecord,
  FileGroupRecord,
  FileLibrarySectionRecord,
  IndexingJobLive,
  KnowledgeAnswerRecord,
  KnowledgeSessionGroupRecord,
  KnowledgeSessionRecord,
  KnowledgeThreadItemRecord,
  ProjectConnection,
  ProjectPreferences,
  ProjectState,
  ResearchRunRecord,
  VectorIndexMemberRecord,
  VectorIndexMemberState,
  VectorIndexRecord,
  VectorIndexRunHistoryEntry,
  VectorIndexStatus,
} from '@/features/project/types'
import {
  DEFAULT_KNOWLEDGE_SESSION_ID,
  DEFAULT_KNOWLEDGE_SESSION_TITLE,
} from '@/features/project/knowledgeSessionDefaults'
import {
  clampPanelLayoutSize,
  type ProjectPanelLayoutKey,
} from '@/features/project/panelLayout'
import {
  applyRunEvent,
  attachRunResult,
  DEFAULT_EMBED_MODEL_ID,
  EMBED_MODELS,
  mergeRunSummary,
  VECTOR_INDEX_HISTORY_LIMIT,
} from '@/features/project/types'
import { moveItem } from '@/features/composer/reorder'
import { knowledgeAnswerFromRunResult } from '@/features/knowledge/answer'
import { applyKnowledgeRunEvent } from '@/features/knowledge/runSteps'
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
  | { threadId: string; type: 'togglePinnedChatThread' }
  | { type: 'deleteJob'; jobId: string }
  | { folderId?: string | null; type: 'createEditorDocument' }
  | { title: string; type: 'createEditorFolder' }
  | { documentId: string; type: 'deleteEditorDocument' }
  | { documentId: string; type: 'togglePinnedEditorDocument' }
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
    /** Append new ids to the END of the order (older load-more pages) rather
     * than prepending (the page-1 hydrate / newest). */
    append?: boolean
    memberships: Record<string, string | null>
    threads: ChatThreadRecord[]
    type: 'upsertServerChatThreads'
  }
  | { messages: ChatMessageRecord[]; threadId: string; type: 'upsertServerChatMessages' }
  | { groups: ChatThreadGroupRecord[]; type: 'upsertServerChatThreadGroups' }
  | { enabled: boolean; persistLocal?: boolean; type: 'setServerSyncEnabled' }
  | { documents: EditorDocumentRecord[]; type: 'upsertServerEditorDocuments' }
  | { contentMarkdown: string; documentId: string; type: 'setServerEditorDocumentBody' }
  | { folders: EditorFolderRecord[]; type: 'upsertServerEditorFolders' }
  | { comments: EditorCommentThreadRecord[]; type: 'upsertServerEditorComments' }
  | { sections: FileLibrarySectionRecord[]; type: 'upsertServerAssetSections' }
  | { groups: FileGroupRecord[]; type: 'upsertServerAssetGroups' }
  | { assets: FileAssetRecord[]; type: 'upsertServerAssetMetadata' }
  | { assetId: string; extractedText: string; type: 'setServerAssetBody' }
  | { assetId: string; extractedText: string; type: 'upgradeFileAssetParse' }
  | { assetId: string; pending: boolean; type: 'setFileAssetParsePending' }
  | { indexes: VectorIndexRecord[]; type: 'upsertServerVectorIndexes' }
  | {
    assistantMessageId: string
    contentMarkdown: string
    createdAt: string
    attachmentRefs?: ChatContextReferenceRecord[]
    modelResolution?: ChatMessageModelResolutionRecord
    requestContext?: ChatMessageRequestContextRecord
    threadId: string
    type: 'startChatExchange'
    userMessageId: string
  }
  | {
    assistantMessageId: string
    createdAt: string
    modelResolution?: ChatMessageModelResolutionRecord
    requestContext?: ChatMessageRequestContextRecord
    threadId: string
    type: 'startChatAssistantResponse'
    userMessageId: string
  }
  | {
    assistantMessageId: string
    createdAt: string
    modelResolution?: ChatMessageModelResolutionRecord
    requestContext?: ChatMessageRequestContextRecord
    replacedAssistantMessageId: string
    threadId: string
    type: 'startChatAssistantRetry'
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
  | { type: 'setKnowledgeHistoryVisible'; isVisible: boolean }
  | { key: ProjectPanelLayoutKey; size: number; type: 'setPanelLayoutSize' }
  | { enabled: boolean; type: 'setChatChainingEnabled' }
  | { type: 'setSelectedChatModelTier'; tier: ChatModelTier | null }
  | { type: 'setSelectedChatModel'; model: string | null }
  | { type: 'setSelectedChatEffort'; effort: string | null }
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
  | { sectionId: string; title: string; type: 'createFileLibrarySection' }
  | { sectionId: string; type: 'deleteFileLibrarySection' }
  | { dims?: number; fileIds: string[]; model?: EmbedModelId; title: string; type: 'createVectorIndex' }
  | { indexId: string; title: string; type: 'renameVectorIndex' }
  | { indexId: string; type: 'deleteVectorIndex' }
  | { dims?: number; indexId: string; model: EmbedModelId; type: 'setVectorIndexModel' }
  | { fileIds: string[]; indexId: string; type: 'addDocsToVectorIndex' }
  | { fileId: string; indexId: string; type: 'removeDocFromVectorIndex' }
  | { indexId: string; jobId: string; runningFileIds?: string[]; source: IndexingJobLive['source']; totalDocuments: number; type: 'startVectorIndexReindex' }
  | { indexId: string; queuePosition: number | null; type: 'markVectorIndexQueued' }
  | { completedDocuments: number; currentDocumentTitle?: string; embedded?: boolean; fileId?: string; indexId: string; totalDocuments: number; type: 'markVectorIndexProgress' }
  | { indexId: string; serverDocumentId: string; type: 'markVectorIndexDocumentEmbedded' }
  | { embeddedFileIds?: string[]; skippedFileIds?: string[]; indexId: string; serverCollectionId?: string; serverCollectionModel?: string; serverDocumentIds?: Record<string, string>; type: 'completeVectorIndexReindex' }
  | { indexId: string; message: string; type: 'markVectorIndexError' }
  | { indexId: string; type: 'markVectorIndexCancelled' }
  | { title: string; type: 'createKnowledgeSessionGroup' }
  | { groupId: string; type: 'deleteKnowledgeSessionGroup' }
  | { groupId: string; targetIndex: number; type: 'moveKnowledgeSessionGroup' }
  | { groupId: string; title: string; type: 'renameKnowledgeSessionGroup' }
  | { groupId: string | null; sessionId: string; targetIndex: number; type: 'moveKnowledgeSessionToGroup' }
  | { session: KnowledgeSessionRecord; type: 'createKnowledgeSession' }
  | { type: 'deleteKnowledgeSession'; sessionId: string }
  | { sessionId: string; type: 'togglePinnedKnowledgeSession' }
  | { title: string; sessionId: string; type: 'renameKnowledgeSession' }
  | { type: 'selectKnowledgeSession'; sessionId: string }
  | { groups: KnowledgeSessionGroupRecord[]; type: 'upsertServerKnowledgeSessionGroups' }
  | { memberships: Record<string, string | null>; sessions: KnowledgeSessionRecord[]; type: 'upsertServerKnowledgeSessions' }
  | { serverIds: string[]; type: 'pruneLocalPlaceholderKnowledgeSessions' }
  | { items: KnowledgeThreadItemRecord[]; sessionId: string; type: 'setServerKnowledgeSessionItems' }
  | { item: KnowledgeThreadItemRecord; type: 'startKnowledgeAsk' }
  | { itemIds: string[]; type: 'deleteKnowledgeItems' }
  | { sessionId: string; type: 'clearKnowledgeSession' }
  | { item: KnowledgeThreadItemRecord; replacedItemId: string; type: 'restartKnowledgeAsk' }
  | { answer: KnowledgeAnswerRecord; runId: string; type: 'completeKnowledgeItem' }

export function initializeResearchDeskState(): ResearchDeskState {
  return createEmptyProjectState()
}

function toggleExplorerPin(ids: readonly string[], id: string) {
  return ids.includes(id)
    ? ids.filter((currentId) => currentId !== id)
    : [id, ...ids]
}

function removeExplorerPin(ids: readonly string[], id: string) {
  return ids.filter((currentId) => currentId !== id)
}

export function researchDeskReducer(
  state: ResearchDeskState,
  action: ResearchDeskAction,
): ResearchDeskState {
  if (action.type === 'hydrateProject') {
    // Bump the load epoch relative to the OUTGOING state (not the loaded one),
    // so the project-scoped server-sync hooks see a changed identity and
    // re-hydrate from this project's own server state -- never carrying the
    // previous project's synced fingerprints into a delete. The loaded state's
    // own (ephemeral, unserialized) epoch is discarded on purpose.
    return { ...action.state, projectEpoch: state.projectEpoch + 1 }
  }
  if (action.type === 'setDemoMode') {
    const base = action.enabled ? createSeedProjectState() : createEmptyProjectState()
    return { ...base, projectEpoch: state.projectEpoch + 1 }
  }
  if (action.type === 'togglePinnedChatThread') {
    if (!state.chatThreads[action.threadId]) return state
    return {
      ...state,
      dirty: true,
      ui: {
        ...state.ui,
        pinnedExplorer: {
          ...state.ui.pinnedExplorer,
          chatThreadIds: toggleExplorerPin(state.ui.pinnedExplorer.chatThreadIds, action.threadId),
        },
      },
    }
  }
  if (action.type === 'togglePinnedEditorDocument') {
    if (!state.editorDocuments[action.documentId]) return state
    return {
      ...state,
      dirty: true,
      ui: {
        ...state.ui,
        pinnedExplorer: {
          ...state.ui.pinnedExplorer,
          editorDocumentIds: toggleExplorerPin(state.ui.pinnedExplorer.editorDocumentIds, action.documentId),
        },
      },
    }
  }
  if (action.type === 'togglePinnedKnowledgeSession') {
    if (!state.knowledgeSessions[action.sessionId]) return state
    return {
      ...state,
      dirty: true,
      ui: {
        ...state.ui,
        pinnedExplorer: {
          ...state.ui.pinnedExplorer,
          knowledgeSessionIds: toggleExplorerPin(
            state.ui.pinnedExplorer.knowledgeSessionIds,
            action.sessionId,
          ),
        },
      },
    }
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
    // Picking a tier (the fallback picker) clears any explicit model choice.
    return {
      ...state,
      dirty: true,
      ui: {
        ...state.ui,
        selectedChatModelTier: action.tier,
        selectedChatModel: null,
        selectedChatEffort: null,
      },
    }
  }
  if (action.type === 'setSelectedChatModel') {
    // Picking a concrete model clears the tier and resets effort to the
    // model's provider default (the reasoning control re-sets it explicitly).
    return {
      ...state,
      dirty: true,
      ui: {
        ...state.ui,
        selectedChatModel: action.model,
        selectedChatModelTier: null,
        selectedChatEffort: null,
      },
    }
  }
  if (action.type === 'setSelectedChatEffort') {
    return {
      ...state,
      dirty: true,
      ui: { ...state.ui, selectedChatEffort: action.effort },
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
    // Collapse state is a persisted ui field (markdown.ts/fileSystem.ts); mark
    // dirty so a pure toggle is durably saved/pushed and survives reload, like
    // chatChainingEnabled above (not opportunistically left to the next save).
    return { ...state, dirty: true, ui: { ...state.ui, isReportVisible: action.isVisible } }
  }
  if (action.type === 'setChatHistoryVisible') {
    return { ...state, dirty: true, ui: { ...state.ui, isChatHistoryVisible: action.isVisible } }
  }
  if (action.type === 'setKnowledgeHistoryVisible') {
    return { ...state, dirty: true, ui: { ...state.ui, isKnowledgeHistoryVisible: action.isVisible } }
  }
  if (action.type === 'setPanelLayoutSize') {
    const size = clampPanelLayoutSize(action.key, action.size)
    if (state.ui.panelLayout[action.key] === size) return state
    return {
      ...state,
      dirty: true,
      ui: {
        ...state.ui,
        panelLayout: {
          ...state.ui.panelLayout,
          [action.key]: size,
        },
      },
    }
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
    if (!state.researchRuns[action.jobId]) return state
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
      ui: {
        ...state.ui,
        pinnedExplorer: {
          ...state.ui.pinnedExplorer,
          editorDocumentIds: removeExplorerPin(state.ui.pinnedExplorer.editorDocumentIds, action.documentId),
        },
      },
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
      dirty: true,
      editorUi: { ...state.editorUi, isCommentPanelVisible: action.isVisible },
    }
  }
  if (action.type === 'setEditorTreeVisible') {
    return {
      ...state,
      dirty: true,
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
    // Knowledge-mode runs are owned by the Knowledge thread. Keeping them out of
    // the global run store prevents deleted/incognito Q&A from reappearing via
    // run-list hydration or project export.
    if (action.summary.mode === 'knowledge') return state
    const current = state.researchRuns[action.summary.run_id]
    const run = mergeRunSummary(current, action.summary, state.ui.selectedStack)
    const researchRunOrder = state.researchRunOrder.includes(run.runId)
      ? state.researchRunOrder
      : [run.runId, ...state.researchRunOrder]
    // Knowledge-mode runs are surfaced by the Wissen thread, never as
    // the selected research job (they are filtered from the job list).
    const shouldSelect = run.mode !== 'knowledge'
      && (action.select || state.ui.selectedJobId === null)
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
  if (action.type === 'upsertServerChatThreads') {
    // Hydrate thread METADATA from the server (M6a). Additive + local-
    // newer-wins (never clobber unpushed local edits) + NEVER sets dirty
    // (server-pushed state must not trigger a re-save loop). Messages are
    // loaded separately on thread open; a freshly hydrated thread starts
    // with an empty message list.
    if (action.threads.length === 0) return state
    const chatThreads = { ...state.chatThreads }
    const chatThreadGroupMemberships = { ...state.chatThreadGroupMemberships }
    const newIds: string[] = []
    for (const incoming of action.threads) {
      const local = chatThreads[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) {
          chatThreads[incoming.id] = {
            ...local,
            title: incoming.title,
            preview: incoming.preview,
            source: incoming.source,
            createdAt: incoming.createdAt,
            updatedAt: incoming.updatedAt,
          }
          chatThreadGroupMemberships[incoming.id] =
            action.memberships[incoming.id] ?? null
        }
      } else {
        chatThreads[incoming.id] = { ...incoming, messages: [] }
        chatThreadGroupMemberships[incoming.id] =
          action.memberships[incoming.id] ?? null
        newIds.push(incoming.id)
      }
    }
    // Preserve the SERVER's keyset order (created_at desc) within each batch:
    // newIds already follows action.threads, which the hook fills straight from
    // the server page. Re-sorting by another key (e.g. updatedAt) would make the
    // merged list inconsistent across page boundaries -- the server paginates by
    // created_at, so an older-created/recently-updated thread would float above
    // page-1 threads it actually sorts after on the server. Keeping the server
    // order makes the displayed list a faithful prefix of the canonical order
    // across every paginated load. (Sending a message to an existing thread does
    // not re-bubble it either -- see startChatExchange -- so created_at order is
    // the coherent client model, not an approximation of activity-recency.)
    // Page-1 / newest hydrate prepends; an older load-more page appends to the
    // end so the displayed order stays newest-first across paginated loads.
    const chatThreadOrder = newIds.length === 0
      ? state.chatThreadOrder
      : action.append
        ? [...state.chatThreadOrder, ...newIds]
        : [...newIds, ...state.chatThreadOrder]
    return {
      ...state,
      chatThreadGroupMemberships,
      chatThreadOrder,
      chatThreads,
    }
  }
  if (action.type === 'upsertServerChatMessages') {
    // Load-on-open: fill a thread's messages from the server, merged by id
    // and ordered chronologically. Never sets dirty. Preview stays as the
    // metadata-hydrated value.
    const thread = state.chatThreads[action.threadId]
    if (!thread || action.messages.length === 0) return state
    const byId = new Map(thread.messages.map((message) => [message.id, message]))
    for (const message of action.messages) byId.set(message.id, message)
    const messages = [...byId.values()].sort((a, b) =>
      a.createdAt.localeCompare(b.createdAt),
    )
    return {
      ...state,
      chatThreads: {
        ...state.chatThreads,
        [action.threadId]: { ...thread, messages },
      },
    }
  }
  if (action.type === 'upsertServerChatThreadGroups') {
    if (action.groups.length === 0) return state
    const chatThreadGroups = { ...state.chatThreadGroups }
    const newIds: string[] = []
    for (const incoming of action.groups) {
      const local = chatThreadGroups[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) chatThreadGroups[incoming.id] = incoming
      } else {
        chatThreadGroups[incoming.id] = incoming
        newIds.push(incoming.id)
      }
    }
    const sortedNew = newIds.sort((a, b) =>
      chatThreadGroups[b].updatedAt.localeCompare(chatThreadGroups[a].updatedAt),
    )
    const chatThreadGroupOrder = sortedNew.length > 0
      ? [...sortedNew, ...state.chatThreadGroupOrder]
      : state.chatThreadGroupOrder
    return { ...state, chatThreadGroupOrder, chatThreadGroups }
  }
  if (action.type === 'setServerSyncEnabled') {
    if (state.serverSyncEnabled === action.enabled) return state
    // A genuine user opt-in (persistLocal omitted/true) marks the project dirty
    // so the flag persists to the local manifest. The AUTO path
    // (persistLocal:false) derives the flag from the authenticated session on
    // every boot, so it must NOT dirty the project (nothing local to save, and
    // dirtying would prompt a spurious "unsaved changes" save loop).
    const dirty = action.persistLocal === false ? state.dirty : true
    return { ...state, dirty, serverSyncEnabled: action.enabled }
  }
  if (action.type === 'upsertServerEditorDocuments') {
    // Hydrate document METADATA from the server (M6b). Additive + local-
    // newer-wins + KEEPS the local body (the body loads separately via
    // setServerEditorDocumentBody on open) + never sets dirty.
    if (action.documents.length === 0) return state
    const editorDocuments = { ...state.editorDocuments }
    const newIds: string[] = []
    for (const incoming of action.documents) {
      const local = editorDocuments[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) {
          editorDocuments[incoming.id] = {
            ...incoming,
            contentMarkdown: local.contentMarkdown,
          }
        }
      } else {
        editorDocuments[incoming.id] = incoming
        newIds.push(incoming.id)
      }
    }
    const sortedNew = newIds.sort((a, b) =>
      editorDocuments[b].updatedAt.localeCompare(editorDocuments[a].updatedAt),
    )
    const editorDocumentOrder = sortedNew.length > 0
      ? [...sortedNew, ...state.editorDocumentOrder]
      : state.editorDocumentOrder
    return { ...state, editorDocumentOrder, editorDocuments }
  }
  if (action.type === 'setServerEditorDocumentBody') {
    // Load-on-open: fill a document's body from the server. Never changes
    // updatedAt (so the autosave does not read it back as a local edit)
    // and never sets dirty.
    const document = state.editorDocuments[action.documentId]
    if (!document) return state
    return {
      ...state,
      editorDocuments: {
        ...state.editorDocuments,
        [action.documentId]: {
          ...document,
          contentMarkdown: action.contentMarkdown,
        },
      },
    }
  }
  if (action.type === 'upsertServerEditorFolders') {
    if (action.folders.length === 0) return state
    const editorFolders = { ...state.editorFolders }
    const newIds: string[] = []
    for (const incoming of action.folders) {
      const local = editorFolders[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) editorFolders[incoming.id] = incoming
      } else {
        editorFolders[incoming.id] = incoming
        newIds.push(incoming.id)
      }
    }
    const sortedNew = newIds.sort((a, b) =>
      editorFolders[b].updatedAt.localeCompare(editorFolders[a].updatedAt),
    )
    const editorFolderOrder = sortedNew.length > 0
      ? [...sortedNew, ...state.editorFolderOrder]
      : state.editorFolderOrder
    return { ...state, editorFolderOrder, editorFolders }
  }
  if (action.type === 'upsertServerEditorComments') {
    // Load-on-open: merge a document's comments from the server, by id,
    // local-newer-wins, never dirty. (Comments have no order array — the
    // editor selectors sort them by anchor position.)
    if (action.comments.length === 0) return state
    const editorComments = { ...state.editorComments }
    for (const incoming of action.comments) {
      const local = editorComments[incoming.id]
      if (!local || incoming.updatedAt > local.updatedAt) {
        editorComments[incoming.id] = incoming
      }
    }
    return { ...state, editorComments }
  }
  if (action.type === 'upsertServerAssetSections') {
    // Hydrate file-library section metadata (M6c). Additive + local-newer-
    // wins + never dirty. Mirror of upsertServerEditorFolders.
    if (action.sections.length === 0) return state
    const fileLibrarySections = { ...state.fileLibrarySections }
    const newIds: string[] = []
    for (const incoming of action.sections) {
      const local = fileLibrarySections[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) fileLibrarySections[incoming.id] = incoming
      } else {
        fileLibrarySections[incoming.id] = incoming
        newIds.push(incoming.id)
      }
    }
    const sortedNew = newIds.sort((a, b) =>
      fileLibrarySections[b].updatedAt.localeCompare(fileLibrarySections[a].updatedAt),
    )
    const fileLibrarySectionOrder = sortedNew.length > 0
      ? [...sortedNew, ...state.fileLibrarySectionOrder]
      : state.fileLibrarySectionOrder
    return { ...state, fileLibrarySectionOrder, fileLibrarySections }
  }
  if (action.type === 'upsertServerAssetGroups') {
    if (action.groups.length === 0) return state
    const fileGroups = { ...state.fileGroups }
    const newIds: string[] = []
    for (const incoming of action.groups) {
      const local = fileGroups[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) fileGroups[incoming.id] = incoming
      } else {
        fileGroups[incoming.id] = incoming
        newIds.push(incoming.id)
      }
    }
    const sortedNew = newIds.sort((a, b) =>
      fileGroups[b].updatedAt.localeCompare(fileGroups[a].updatedAt),
    )
    const fileGroupOrder = sortedNew.length > 0
      ? [...sortedNew, ...state.fileGroupOrder]
      : state.fileGroupOrder
    return { ...state, fileGroupOrder, fileGroups }
  }
  if (action.type === 'upsertServerAssetMetadata') {
    // Hydrate asset METADATA from the server (M6c). Additive + local-newer-
    // wins + KEEPS the local extractedText (the body loads separately via
    // setServerAssetBody on use) + never sets dirty. Mirror of
    // upsertServerEditorDocuments.
    if (action.assets.length === 0) return state
    const fileAssets = { ...state.fileAssets }
    const newIds: string[] = []
    for (const incoming of action.assets) {
      const local = fileAssets[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) {
          fileAssets[incoming.id] = { ...incoming, extractedText: local.extractedText }
        }
      } else {
        fileAssets[incoming.id] = incoming
        newIds.push(incoming.id)
      }
    }
    const sortedNew = newIds.sort((a, b) =>
      fileAssets[b].updatedAt.localeCompare(fileAssets[a].updatedAt),
    )
    const fileAssetOrder = sortedNew.length > 0
      ? [...sortedNew, ...state.fileAssetOrder]
      : state.fileAssetOrder
    return { ...state, fileAssetOrder, fileAssets }
  }
  if (action.type === 'setServerAssetBody') {
    // Load-on-use: fill an asset's extractedText from the server. Never
    // changes updatedAt (so the autosave does not read it back as a local
    // edit) and never sets dirty. Mirror of setServerEditorDocumentBody.
    const asset = state.fileAssets[action.assetId]
    if (!asset) return state
    return {
      ...state,
      fileAssets: {
        ...state.fileAssets,
        [action.assetId]: { ...asset, extractedText: action.extractedText },
      },
    }
  }
  if (action.type === 'upgradeFileAssetParse') {
    // The background (or index-time) server parse landed: replace the instant
    // client body with the higher-fidelity MarkItDown text and reset the parse
    // result to a clean success — this clears any client-parse error (e.g.
    // pdf.js failing on Safari) that the server just superseded, and ends the
    // pending state. A real edit: bumps updatedAt + dirty so it persists.
    // No-ops if the asset is gone, the text is empty (never blank out a good
    // client parse), or the upgrade already landed.
    const asset = state.fileAssets[action.assetId]
    if (
      !asset
      || !action.extractedText.trim()
      || (asset.parserId === 'markitdown' && asset.extractedText === action.extractedText)
    ) {
      return state
    }
    return {
      ...state,
      dirty: true,
      fileAssets: {
        ...state.fileAssets,
        [asset.id]: {
          ...asset,
          extractedText: action.extractedText,
          parserId: 'markitdown',
          parseStatus: 'parsed',
          parseWarning: null,
          textTruncated: false,
          parsePending: false,
          updatedAt: new Date().toISOString(),
        },
      },
    }
  }
  if (action.type === 'setFileAssetParsePending') {
    // Transient background-parse flag (drives the "Parsing…" badge). Never
    // dirty/persisted — it is a client-only in-flight marker.
    const asset = state.fileAssets[action.assetId]
    if (!asset || Boolean(asset.parsePending) === action.pending) return state
    return {
      ...state,
      fileAssets: {
        ...state.fileAssets,
        [asset.id]: { ...asset, parsePending: action.pending },
      },
    }
  }
  if (action.type === 'upsertServerVectorIndexes') {
    // Hydrate vector-index records from the server (M6c). Additive + local-
    // newer-wins + never dirty. Full records (members + history travel with
    // them); the ephemeral indexingJobs map is a SEPARATE field and is left
    // untouched so a live reindex on this device is never clobbered.
    if (action.indexes.length === 0) return state
    const vectorIndexes = { ...state.vectorIndexes }
    const newIds: string[] = []
    for (const incoming of action.indexes) {
      const local = vectorIndexes[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) vectorIndexes[incoming.id] = incoming
      } else {
        vectorIndexes[incoming.id] = incoming
        newIds.push(incoming.id)
      }
    }
    const sortedNew = newIds.sort((a, b) =>
      vectorIndexes[b].updatedAt.localeCompare(vectorIndexes[a].updatedAt),
    )
    const vectorIndexOrder = sortedNew.length > 0
      ? [...sortedNew, ...state.vectorIndexOrder]
      : state.vectorIndexOrder
    return { ...state, vectorIndexOrder, vectorIndexes }
  }
  if (action.type === 'appendApiRunEvent') {
    // Knowledge thread items consume the same event stream as the run
    // records: demo asks have no run record, live asks have both.
    const withKnowledge = applyEventToKnowledgeItem(state, action.event)
    const current = withKnowledge.researchRuns[action.event.run_id]
    if (!current) return withKnowledge
    const run = applyRunEvent(current, action.event)

    return {
      ...withKnowledge,
      dirty: true,
      researchRuns: {
        ...withKnowledge.researchRuns,
        [run.runId]: run,
      },
    }
  }
  if (action.type === 'attachApiRunResult') {
    const knowledgeItem = knowledgeItemByRunId(state, action.result.run_id)
    const completedAt = new Date().toISOString()
    const withKnowledge = knowledgeItem
      ? touchKnowledgeSessions(
        updateKnowledgeItem({ ...state, dirty: true }, knowledgeItem.id, (item) => ({
          ...item,
          answer: knowledgeAnswerFromRunResult(action.result),
          completedAt,
          progress: {
            ...item.progress,
            steps: item.progress.steps.map((step) => (
              step.status === 'done' ? step : { ...step, status: 'done' as const }
            )),
          },
          status: 'completed',
        })),
        [knowledgeItem.sessionId],
        completedAt,
      )
      : state
    const current = withKnowledge.researchRuns[action.result.run_id]
    if (!current) return withKnowledge
    const run = attachRunResult(current, action.result)

    return {
      ...withKnowledge,
      dirty: true,
      researchRuns: {
        ...withKnowledge.researchRuns,
        [run.runId]: run,
      },
    }
  }
  if (action.type === 'markApiRunError') {
    const knowledgeItem = knowledgeItemByRunId(state, action.runId)
    const updatedAt = new Date().toISOString()
    const withKnowledge = knowledgeItem
      ? touchKnowledgeSessions(
        updateKnowledgeItem({ ...state, dirty: true }, knowledgeItem.id, (item) => (
          item.status === 'completed'
            ? item
            : { ...item, error: action.message, status: 'failed' }
        )),
        [knowledgeItem.sessionId],
        updatedAt,
      )
      : state
    const current = withKnowledge.researchRuns[action.runId]
    if (!current) return withKnowledge

    return {
      ...withKnowledge,
      dirty: true,
      researchRuns: {
        ...withKnowledge.researchRuns,
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
  if (action.type === 'createKnowledgeSession') {
    const exists = state.knowledgeSessions[action.session.id]
    return {
      ...state,
      dirty: true,
      knowledgeSessionOrder: exists
        ? state.knowledgeSessionOrder
        : [action.session.id, ...state.knowledgeSessionOrder],
      knowledgeSessions: {
        ...state.knowledgeSessions,
        [action.session.id]: action.session,
      },
      selectedKnowledgeSessionId: action.session.id,
    }
  }
  if (action.type === 'createKnowledgeSessionGroup') {
    const now = new Date().toISOString()
    const group: KnowledgeSessionGroupRecord = {
      createdAt: now,
      id: createId('knowledge-session-group'),
      title: action.title.trim() || 'New folder',
      updatedAt: now,
    }
    return {
      ...state,
      dirty: true,
      knowledgeSessionGroupOrder: [group.id, ...(state.knowledgeSessionGroupOrder ?? [])],
      knowledgeSessionGroups: {
        ...(state.knowledgeSessionGroups ?? {}),
        [group.id]: group,
      },
    }
  }
  if (action.type === 'renameKnowledgeSessionGroup') {
    const group = state.knowledgeSessionGroups[action.groupId]
    const title = action.title.trim()
    if (!group || !title || group.title === title) return state
    return {
      ...state,
      dirty: true,
      knowledgeSessionGroups: {
        ...state.knowledgeSessionGroups,
        [group.id]: { ...group, title, updatedAt: new Date().toISOString() },
      },
    }
  }
  if (action.type === 'deleteKnowledgeSessionGroup') {
    if (!state.knowledgeSessionGroups[action.groupId]) return state
    const knowledgeSessionGroups = { ...state.knowledgeSessionGroups }
    delete knowledgeSessionGroups[action.groupId]
    const knowledgeSessionGroupMemberships = Object.fromEntries(
      Object.entries(state.knowledgeSessionGroupMemberships ?? {})
        .filter(([, groupId]) => groupId !== action.groupId),
    )
    return {
      ...state,
      dirty: true,
      knowledgeSessionGroupMemberships,
      knowledgeSessionGroupOrder: (state.knowledgeSessionGroupOrder ?? [])
        .filter((groupId) => groupId !== action.groupId),
      knowledgeSessionGroups,
    }
  }
  if (action.type === 'moveKnowledgeSessionGroup') {
    return moveKnowledgeSessionGroup(state, action.groupId, action.targetIndex)
  }
  if (action.type === 'moveKnowledgeSessionToGroup') {
    return moveKnowledgeSessionToGroup(state, action.sessionId, action.groupId, action.targetIndex)
  }
  if (action.type === 'selectKnowledgeSession') {
    if (!state.knowledgeSessions[action.sessionId]) return state
    return { ...state, selectedKnowledgeSessionId: action.sessionId }
  }
  if (action.type === 'renameKnowledgeSession') {
    const session = state.knowledgeSessions[action.sessionId]
    const title = action.title.trim()
    if (!session || !title || session.title === title) return state
    return {
      ...state,
      dirty: true,
      knowledgeSessions: {
        ...state.knowledgeSessions,
        [session.id]: { ...session, title, updatedAt: new Date().toISOString() },
      },
    }
  }
  if (action.type === 'deleteKnowledgeSession') {
    // The last remaining session is deletable too: the empty state shows the
    // composer, and the next ask creates a fresh session (startKnowledgeAsk).
    if (!state.knowledgeSessions[action.sessionId]) return state
    const knowledgeSessions = { ...state.knowledgeSessions }
    delete knowledgeSessions[action.sessionId]
    const knowledgeSessionOrder = state.knowledgeSessionOrder.filter((id) => id !== action.sessionId)
    const knowledgeItems = { ...state.knowledgeItems }
    const knowledgeItemOrder = state.knowledgeItemOrder.filter((itemId) => {
      const item = knowledgeItems[itemId]
      const keep = !item || item.sessionId !== action.sessionId
      if (!keep) delete knowledgeItems[itemId]
      return keep
    })
    const selectedKnowledgeSessionId = state.selectedKnowledgeSessionId === action.sessionId
      ? knowledgeSessionOrder[0] ?? null
      : state.selectedKnowledgeSessionId
    const knowledgeSessionGroupMemberships = { ...(state.knowledgeSessionGroupMemberships ?? {}) }
    delete knowledgeSessionGroupMemberships[action.sessionId]
    return {
      ...state,
      dirty: true,
      knowledgeItemOrder,
      knowledgeItems,
      knowledgeSessionGroupMemberships,
      knowledgeSessionOrder,
      knowledgeSessions,
      selectedKnowledgeSessionId,
      ui: {
        ...state.ui,
        pinnedExplorer: {
          ...state.ui.pinnedExplorer,
          knowledgeSessionIds: removeExplorerPin(state.ui.pinnedExplorer.knowledgeSessionIds, action.sessionId),
        },
      },
    }
  }
  if (action.type === 'upsertServerKnowledgeSessionGroups') {
    if (action.groups.length === 0) return state
    const knowledgeSessionGroups = { ...state.knowledgeSessionGroups }
    const newIds: string[] = []
    for (const incoming of action.groups) {
      const local = knowledgeSessionGroups[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) knowledgeSessionGroups[incoming.id] = incoming
      } else {
        knowledgeSessionGroups[incoming.id] = incoming
        newIds.push(incoming.id)
      }
    }
    const sortedNew = newIds.sort((a, b) =>
      knowledgeSessionGroups[b].updatedAt.localeCompare(knowledgeSessionGroups[a].updatedAt),
    )
    const knowledgeSessionGroupOrder = sortedNew.length > 0
      ? [...sortedNew, ...state.knowledgeSessionGroupOrder]
      : state.knowledgeSessionGroupOrder
    return { ...state, knowledgeSessionGroupOrder, knowledgeSessionGroups }
  }
  if (action.type === 'upsertServerKnowledgeSessions') {
    if (action.sessions.length === 0) return state
    const knowledgeSessions = { ...state.knowledgeSessions }
    const knowledgeSessionGroupMemberships = { ...state.knowledgeSessionGroupMemberships }
    const newIds: string[] = []
    for (const incoming of action.sessions) {
      const local = knowledgeSessions[incoming.id]
      if (local) {
        if (incoming.updatedAt > local.updatedAt) {
          knowledgeSessions[incoming.id] = incoming
          knowledgeSessionGroupMemberships[incoming.id] =
            action.memberships[incoming.id] ?? null
        }
      } else {
        knowledgeSessions[incoming.id] = incoming
        knowledgeSessionGroupMemberships[incoming.id] =
          action.memberships[incoming.id] ?? null
        newIds.push(incoming.id)
      }
    }
    const knowledgeSessionOrder = newIds.length === 0
      ? state.knowledgeSessionOrder
      : [...newIds, ...state.knowledgeSessionOrder]
    const selectedKnowledgeSessionId =
      state.selectedKnowledgeSessionId && knowledgeSessions[state.selectedKnowledgeSessionId]
        ? state.selectedKnowledgeSessionId
        : knowledgeSessionOrder[0] ?? null
    return {
      ...state,
      knowledgeSessionGroupMemberships,
      knowledgeSessionOrder,
      knowledgeSessions,
      selectedKnowledgeSessionId,
    }
  }
  if (action.type === 'pruneLocalPlaceholderKnowledgeSessions') {
    // The server is authoritative for the untouched seed placeholder: drop
    // only that pristine local bootstrap session when it is not on the server.
    // A renamed or user-created empty session is user intent and must sync.
    const serverIds = new Set(action.serverIds)
    const sessionHasItems = (sessionId: string) =>
      state.knowledgeItemOrder.some(
        (itemId) => state.knowledgeItems[itemId]?.sessionId === sessionId,
      )
    const isPristineBootstrap = (sessionId: string) => {
      const session = state.knowledgeSessions[sessionId]
      return Boolean(session)
        && sessionId === DEFAULT_KNOWLEDGE_SESSION_ID
        && session.title === DEFAULT_KNOWLEDGE_SESSION_TITLE
        && !sessionHasItems(sessionId)
    }
    const keep = (sessionId: string) => {
      if (serverIds.has(sessionId) || sessionHasItems(sessionId)) return true
      if (!state.knowledgeSessions[sessionId]) return false
      return !isPristineBootstrap(sessionId)
    }
    const removedIds = state.knowledgeSessionOrder.filter((id) => !keep(id))
    if (removedIds.length === 0) return state
    const knowledgeSessions = { ...state.knowledgeSessions }
    for (const id of removedIds) delete knowledgeSessions[id]
    const knowledgeSessionOrder = state.knowledgeSessionOrder.filter(keep)
    const knowledgeSessionGroupMemberships = { ...(state.knowledgeSessionGroupMemberships ?? {}) }
    for (const id of removedIds) delete knowledgeSessionGroupMemberships[id]
    const selectedKnowledgeSessionId =
      state.selectedKnowledgeSessionId && knowledgeSessions[state.selectedKnowledgeSessionId]
        ? state.selectedKnowledgeSessionId
        : knowledgeSessionOrder[0] ?? null
    // No dirty flag: this runs during hydrate (a server read), not a user edit,
    // so it must not mark the project dirty or trigger a save.
    return {
      ...state,
      knowledgeSessionGroupMemberships,
      knowledgeSessionOrder,
      knowledgeSessions,
      selectedKnowledgeSessionId,
    }
  }
  if (action.type === 'setServerKnowledgeSessionItems') {
    const session = state.knowledgeSessions[action.sessionId]
    if (!session) return state
    const existingOtherIds = state.knowledgeItemOrder.filter((itemId) =>
      state.knowledgeItems[itemId]?.sessionId !== action.sessionId)
    const incomingItems = action.items.map((item) => ({ ...item, sessionId: action.sessionId }))
    return {
      ...state,
      knowledgeItemOrder: [...existingOtherIds, ...incomingItems.map((item) => item.id)],
      knowledgeItems: {
        ...Object.fromEntries(existingOtherIds.flatMap((itemId) => {
          const item = state.knowledgeItems[itemId]
          return item ? [[itemId, item] as const] : []
        })),
        ...Object.fromEntries(incomingItems.map((item) => [item.id, item])),
      },
    }
  }
  if (action.type === 'deleteKnowledgeItems') {
    const itemIds = new Set(action.itemIds)
    const existingIds = action.itemIds.filter((itemId) => Boolean(state.knowledgeItems[itemId]))
    if (existingIds.length === 0) return state
    const sessionIds = new Set(existingIds.map((itemId) => state.knowledgeItems[itemId].sessionId))
    const knowledgeItems = { ...state.knowledgeItems }
    for (const itemId of existingIds) delete knowledgeItems[itemId]
    const updatedAt = new Date().toISOString()
    return touchKnowledgeSessions({
      ...state,
      dirty: true,
      knowledgeItemOrder: state.knowledgeItemOrder.filter((itemId) => !itemIds.has(itemId)),
      knowledgeItems,
    }, sessionIds, updatedAt)
  }
  if (action.type === 'clearKnowledgeSession') {
    if (!state.knowledgeSessions[action.sessionId]) return state
    const itemIds = state.knowledgeItemOrder.filter((itemId) =>
      state.knowledgeItems[itemId]?.sessionId === action.sessionId)
    if (itemIds.length === 0) return state
    const itemIdSet = new Set(itemIds)
    const knowledgeItems = { ...state.knowledgeItems }
    for (const itemId of itemIds) delete knowledgeItems[itemId]
    const updatedAt = new Date().toISOString()
    return touchKnowledgeSessions({
      ...state,
      dirty: true,
      knowledgeItemOrder: state.knowledgeItemOrder.filter((itemId) => !itemIdSet.has(itemId)),
      knowledgeItems,
    }, [action.sessionId], updatedAt)
  }
  if (action.type === 'startKnowledgeAsk') {
    const session = state.knowledgeSessions[action.item.sessionId]
    const sessionHasItems = state.knowledgeItemOrder.some((itemId) =>
      state.knowledgeItems[itemId]?.sessionId === action.item.sessionId)
    const updatedSession = session
      ? {
        ...session,
        title: !sessionHasItems && isPlaceholderKnowledgeSessionTitle(session.title)
          ? knowledgeSessionTitleFromQuestion(action.item.question)
          : session.title,
        updatedAt: action.item.createdAt,
      }
      : {
        createdAt: action.item.createdAt,
        id: action.item.sessionId,
        title: knowledgeSessionTitleFromQuestion(action.item.question),
        updatedAt: action.item.createdAt,
      }
    const knowledgeSessionOrder = state.knowledgeSessionOrder.includes(updatedSession.id)
      ? state.knowledgeSessionOrder
      : [updatedSession.id, ...state.knowledgeSessionOrder]
    return {
      ...state,
      dirty: true,
      knowledgeItemOrder: [...state.knowledgeItemOrder, action.item.id],
      knowledgeItems: {
        ...state.knowledgeItems,
        [action.item.id]: action.item,
      },
      knowledgeSessionOrder,
      knowledgeSessions: {
        ...state.knowledgeSessions,
        [updatedSession.id]: updatedSession,
      },
      selectedKnowledgeSessionId: updatedSession.id,
    }
  }
  if (action.type === 'restartKnowledgeAsk') {
    const current = state.knowledgeItems[action.replacedItemId]
    if (!current) return state
    const restarted: KnowledgeThreadItemRecord = {
      collectionTitles: action.item.collectionTitles,
      createdAt: action.item.createdAt,
      id: current.id,
      progress: { steps: [] },
      question: action.item.question,
      requestedProfile: action.item.requestedProfile,
      runId: action.item.runId,
      sessionId: current.sessionId,
      status: 'running',
    }
    if (action.item.collectionIds) restarted.collectionIds = action.item.collectionIds
    if (action.item.topK !== undefined) restarted.topK = action.item.topK
    if (action.item.finalK !== undefined) restarted.finalK = action.item.finalK
    return touchKnowledgeSessions({
      ...state,
      dirty: true,
      knowledgeItems: {
        ...state.knowledgeItems,
        [current.id]: restarted,
      },
      selectedKnowledgeSessionId: current.sessionId,
    }, [current.sessionId], action.item.createdAt)
  }
  if (action.type === 'completeKnowledgeItem') {
    const item = knowledgeItemByRunId(state, action.runId)
    if (!item) return state
    const completedAt = new Date().toISOString()
    return touchKnowledgeSessions(
      updateKnowledgeItem({
        ...state,
        dirty: true,
      }, item.id, (current) => ({
        ...current,
        answer: action.answer,
        completedAt,
        progress: {
          ...current.progress,
          steps: current.progress.steps.map((step) => (
            step.status === 'done' ? step : { ...step, status: 'done' as const }
          )),
        },
        status: 'completed',
      })),
      [item.sessionId],
      completedAt,
    )
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
    const { vectorIndexes } = dropFilesFromVectorIndexes(
      state.vectorIndexes,
      new Set([action.fileId]),
      new Date().toISOString(),
    )
    return {
      ...state,
      dirty: true,
      fileAssetOrder: state.fileAssetOrder.filter((fileId) => fileId !== action.fileId),
      fileAssets,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: state.ui.pendingChatAttachmentRefs.filter(keepRef),
      },
      vectorIndexes,
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
  if (action.type === 'createFileLibrarySection') {
    const now = new Date().toISOString()
    const section: FileLibrarySectionRecord = {
      createdAt: now,
      id: createId('file-section'),
      kind: 'custom',
      title: action.title.trim() || 'Neue Sammlung',
      updatedAt: now,
    }
    return {
      ...state,
      dirty: true,
      fileLibrarySectionOrder: [...state.fileLibrarySectionOrder, section.id],
      fileLibrarySections: { ...state.fileLibrarySections, [section.id]: section },
    }
  }
  if (action.type === 'deleteFileLibrarySection') {
    const section = state.fileLibrarySections[action.sectionId]
    if (!section || section.kind === 'temporary') return state
    const now = new Date().toISOString()
    const removedFileIds = new Set(
      Object.values(state.fileAssets)
        .filter((asset) => asset.sectionId === action.sectionId)
        .map((asset) => asset.id),
    )
    const removedGroupIds = new Set(
      Object.values(state.fileGroups)
        .filter((group) => group.sectionId === action.sectionId)
        .map((group) => group.id),
    )
    const fileAssets = { ...state.fileAssets }
    for (const fileId of removedFileIds) delete fileAssets[fileId]
    const fileGroups = { ...state.fileGroups }
    for (const groupId of removedGroupIds) delete fileGroups[groupId]
    const fileLibrarySections = { ...state.fileLibrarySections }
    delete fileLibrarySections[action.sectionId]
    const { vectorIndexes } = dropFilesFromVectorIndexes(state.vectorIndexes, removedFileIds, now)
    const keepRef = (ref: ChatContextReferenceRecord) => {
      if (ref.kind === 'file-asset') return !removedFileIds.has(ref.fileId)
      if (ref.kind === 'file-group') return !removedGroupIds.has(ref.groupId)
      return true
    }
    return {
      ...state,
      dirty: true,
      fileAssetOrder: state.fileAssetOrder.filter((fileId) => !removedFileIds.has(fileId)),
      fileAssets,
      fileGroupOrder: state.fileGroupOrder.filter((groupId) => !removedGroupIds.has(groupId)),
      fileGroups,
      fileLibrarySectionOrder: state.fileLibrarySectionOrder.filter((sectionId) => sectionId !== action.sectionId),
      fileLibrarySections,
      ui: {
        ...state.ui,
        pendingChatAttachmentRefs: state.ui.pendingChatAttachmentRefs.filter(keepRef),
      },
      vectorIndexes,
    }
  }
  if (action.type === 'createVectorIndex') {
    const now = new Date().toISOString()
    const seen = new Set<string>()
    const members: VectorIndexMemberRecord[] = []
    for (const fileId of action.fileIds) {
      if (seen.has(fileId) || !state.fileAssets[fileId]) continue
      seen.add(fileId)
      members.push({ fileId, state: 'pending' })
    }
    const id = createId('vector-index')
    const model = action.model ?? DEFAULT_EMBED_MODEL_ID
    const index: VectorIndexRecord = {
      createdAt: now,
      dims: action.dims ?? dimsForEmbedModel(model),
      handle: uniqueVectorHandle(slugifyVectorHandle(action.title), state),
      id,
      members,
      model,
      status: members.length > 0 ? 'stale' : 'ready',
      title: action.title.trim() || 'Neuer Vektor-Index',
      updatedAt: now,
    }
    return {
      ...state,
      dirty: true,
      vectorIndexOrder: [id, ...state.vectorIndexOrder],
      vectorIndexes: { ...state.vectorIndexes, [id]: index },
    }
  }
  if (action.type === 'renameVectorIndex') {
    const index = state.vectorIndexes[action.indexId]
    const title = action.title.trim()
    if (!index || !title || index.title === title) return state
    return writeVectorIndex(state, {
      ...index,
      handle: uniqueVectorHandle(slugifyVectorHandle(title), state, index.id),
      title,
      updatedAt: new Date().toISOString(),
    })
  }
  if (action.type === 'deleteVectorIndex') {
    if (!state.vectorIndexes[action.indexId]) return state
    const vectorIndexes = { ...state.vectorIndexes }
    delete vectorIndexes[action.indexId]
    return {
      ...state,
      dirty: true,
      vectorIndexOrder: state.vectorIndexOrder.filter((indexId) => indexId !== action.indexId),
      vectorIndexes,
    }
  }
  if (action.type === 'setVectorIndexModel') {
    const index = state.vectorIndexes[action.indexId]
    if (!index || index.model === action.model) return state
    const hasMembers = index.members.length > 0
    return writeVectorIndex(state, {
      ...index,
      dims: action.dims ?? dimsForEmbedModel(action.model),
      members: hasMembers
        ? index.members.map((member): VectorIndexMemberRecord => ({ ...member, state: 'pending' }))
        : index.members,
      model: action.model,
      status: hasMembers ? 'stale' : 'ready',
      updatedAt: new Date().toISOString(),
    })
  }
  if (action.type === 'addDocsToVectorIndex') {
    const index = state.vectorIndexes[action.indexId]
    if (!index) return state
    const have = new Set(index.members.map((member) => member.fileId))
    const additions: VectorIndexMemberRecord[] = []
    for (const fileId of action.fileIds) {
      if (have.has(fileId) || !state.fileAssets[fileId]) continue
      have.add(fileId)
      additions.push({ fileId, state: 'pending' })
    }
    if (additions.length === 0) return state
    return writeVectorIndex(state, {
      ...index,
      members: [...index.members, ...additions],
      status: 'stale',
      updatedAt: new Date().toISOString(),
    })
  }
  if (action.type === 'removeDocFromVectorIndex') {
    const index = state.vectorIndexes[action.indexId]
    if (!index || !index.members.some((member) => member.fileId === action.fileId)) return state
    const members = index.members.filter((member) => member.fileId !== action.fileId)
    const status = members.length === 0
      ? 'ready'
      : members.some((member) => member.state === 'pending')
        ? 'stale'
        : index.status
    return writeVectorIndex(state, {
      ...index,
      members,
      status,
      updatedAt: new Date().toISOString(),
    })
  }
  if (action.type === 'startVectorIndexReindex') {
    const index = state.vectorIndexes[action.indexId]
    if (!index) return state
    const now = new Date().toISOString()
    const next = writeVectorIndex(state, {
      ...index,
      lastError: null,
      status: 'indexing',
      updatedAt: now,
    })
    return {
      ...next,
      indexingJobs: {
        ...next.indexingJobs,
        [action.indexId]: {
          completedDocuments: 0,
          jobId: action.jobId,
          percent: 0,
          // The run's working set. A client subset (incremental add) is passed
          // explicitly; the durable server re-embed omits it (the worker, not
          // the client, enumerates the docs) — default to every member, since a
          // re-embed re-vectorizes the whole collection.
          runningFileIds: action.runningFileIds ?? index.members.map((member) => member.fileId),
          source: action.source,
          startedAt: now,
          totalDocuments: action.totalDocuments,
        },
      },
    }
  }
  if (action.type === 'markVectorIndexQueued') {
    // The server job is waiting for a free slot — surface its FIFO
    // position. Ephemeral, like progress (no writeVectorIndex / dirty).
    const live = state.indexingJobs[action.indexId]
    if (!live) return state
    return {
      ...state,
      indexingJobs: {
        ...state.indexingJobs,
        [action.indexId]: { ...live, queuePosition: action.queuePosition },
      },
    }
  }
  if (action.type === 'markVectorIndexProgress') {
    // Hot path: update ONLY the ephemeral live entry — no writeVectorIndex,
    // so the project is never marked dirty by streaming progress.
    const live = state.indexingJobs[action.indexId]
    if (!live) return state
    const percent =
      action.totalDocuments > 0
        ? Math.round((action.completedDocuments / action.totalDocuments) * 100)
        : live.percent
    // Client-build path reports the just-confirmed member (fileId) so the file
    // list can flip that row to its real outcome live. The durable server-job
    // path omits fileId (it streams only counts) — then leave the sets as-is.
    const embeddedFileIds =
      action.fileId && action.embedded
        ? [...(live.embeddedFileIds ?? []), action.fileId]
        : live.embeddedFileIds
    const skippedFileIds =
      action.fileId && !action.embedded
        ? [...(live.skippedFileIds ?? []), action.fileId]
        : live.skippedFileIds
    return {
      ...state,
      indexingJobs: {
        ...state.indexingJobs,
        [action.indexId]: {
          ...live,
          completedDocuments: action.completedDocuments,
          currentDocumentTitle: action.currentDocumentTitle,
          embeddedFileIds,
          percent,
          // Progress means a slot freed up and the job is running now —
          // clear any queued position so the UI leaves the waiting state.
          queuePosition: null,
          skippedFileIds,
          totalDocuments: action.totalDocuments,
        },
      },
    }
  }
  if (action.type === 'markVectorIndexDocumentEmbedded') {
    // The durable server re-embed confirms documents by their backend id (the
    // SSE per-document event). Resolve it to the local file via the member's
    // serverDocumentId and flip just that row live — same ephemeral
    // `embeddedFileIds` channel the client-build path uses, so no dirty.
    const live = state.indexingJobs[action.indexId]
    const index = state.vectorIndexes[action.indexId]
    if (!live || !index) return state
    const matched = index.members.find(
      (member) => member.serverDocumentId === action.serverDocumentId,
    )
    // No tracked id (older index) → cannot map; the row flips at completion.
    if (!matched || live.embeddedFileIds?.includes(matched.fileId)) return state
    return {
      ...state,
      indexingJobs: {
        ...state.indexingJobs,
        [action.indexId]: {
          ...live,
          embeddedFileIds: [...(live.embeddedFileIds ?? []), matched.fileId],
        },
      },
    }
  }
  if (action.type === 'completeVectorIndexReindex') {
    const index = state.vectorIndexes[action.indexId]
    if (!index || index.status !== 'indexing') return state
    const now = new Date().toISOString()
    const live = state.indexingJobs[action.indexId]
    // `embeddedFileIds` is the COMPLETE set of members now in the collection
    // (client-driven build/incremental): members in it are embedded. Members in
    // `skippedFileIds` carried no extractable text — TERMINAL 'skipped' (can
    // never embed), distinct from genuinely-not-yet-ingested members which stay
    // 'pending'. The index is honestly 'ready' once nothing is 'pending'
    // (skipped is an accepted terminal outcome, never a false "ready" for a doc
    // that still owes vectors). When `embeddedFileIds` is absent (durable
    // re-embed of an existing collection) every member was re-embedded.
    const embeddedSet = action.embeddedFileIds ? new Set(action.embeddedFileIds) : null
    const skippedSet = action.skippedFileIds ? new Set(action.skippedFileIds) : null
    // Persist each member's backend document id (merge: this run's map for the
    // just-ingested, existing ids kept) so a later removal deletes the exact doc.
    const docIds = action.serverDocumentIds
    const withDocId = (member: VectorIndexMemberRecord): string | undefined =>
      docIds?.[member.fileId] ?? member.serverDocumentId
    const nextMemberState = (member: VectorIndexMemberRecord): VectorIndexMemberState => {
      if (embeddedSet?.has(member.fileId)) return 'embedded'
      if (skippedSet?.has(member.fileId)) return 'skipped'
      // Not part of THIS run (e.g. an incremental add of other docs): keep a
      // terminal state the member already reached — only a genuinely
      // unprocessed member stays 'pending'. A previously-skipped doc must not
      // silently revert to pending.
      return member.state === 'pending' ? 'pending' : member.state
    }
    const members = embeddedSet
      ? index.members.map((member): VectorIndexMemberRecord => ({
          ...member,
          serverDocumentId: withDocId(member),
          state: nextMemberState(member),
        }))
      : index.members.map((member): VectorIndexMemberRecord => ({
          ...member,
          serverDocumentId: withDocId(member),
          // Durable re-embed re-vectorizes existing documents in place; a
          // no-text 'skipped' member has no document to re-embed, so it stays
          // skipped rather than falsely flipping to embedded.
          state: member.state === 'skipped' ? 'skipped' : 'embedded',
        }))
    // Ready once nothing is genuinely pending; embedded + skipped are both
    // terminal (matches the manifest's vectorIndexStatusOrDefault rule).
    const nothingPending = members.every((member) => member.state !== 'pending')
    const next = writeVectorIndex(state, {
      ...index,
      history: appendVectorIndexHistory(index, {
        documents: embeddedSet ? embeddedSet.size : (live?.totalDocuments ?? index.members.length),
        durationMs: runDurationMs(live, now),
        finishedAt: now,
        result: 'ok',
        startedAt: live?.startedAt ?? now,
      }),
      lastError: null,
      members,
      serverCollectionId: action.serverCollectionId ?? index.serverCollectionId ?? null,
      serverCollectionModel:
        action.serverCollectionModel ?? index.serverCollectionModel ?? null,
      status: nothingPending ? 'ready' : 'stale',
      updatedAt: now,
    })
    return clearIndexingJob(next, action.indexId)
  }
  if (action.type === 'markVectorIndexError') {
    const index = state.vectorIndexes[action.indexId]
    // Same precondition as completion: a terminal transition is only valid
    // while a run is in flight. Without it, a late/duplicate terminal
    // callback (resume race, double cancel) would append a garbage history
    // row and flip a finished index back to error.
    if (!index || index.status !== 'indexing') return state
    const now = new Date().toISOString()
    const live = state.indexingJobs[action.indexId]
    const next = writeVectorIndex(state, {
      ...index,
      history: appendVectorIndexHistory(index, {
        documents: live?.completedDocuments ?? 0,
        durationMs: runDurationMs(live, now),
        error: action.message,
        finishedAt: now,
        result: 'error',
        startedAt: live?.startedAt ?? now,
      }),
      lastError: action.message,
      status: 'error',
      updatedAt: now,
    })
    return clearIndexingJob(next, action.indexId)
  }
  if (action.type === 'markVectorIndexCancelled') {
    const index = state.vectorIndexes[action.indexId]
    if (!index || index.status !== 'indexing') return state
    const now = new Date().toISOString()
    const live = state.indexingJobs[action.indexId]
    // Restore the pre-reindex status: stale if any member still needs
    // embedding, else ready. The record's member states were untouched
    // during the run, so this reflects reality without storing it.
    const status: VectorIndexStatus = index.members.some(
      (member) => member.state === 'pending',
    )
      ? 'stale'
      : 'ready'
    const next = writeVectorIndex(state, {
      ...index,
      history: appendVectorIndexHistory(index, {
        documents: live?.completedDocuments ?? 0,
        durationMs: runDurationMs(live, now),
        finishedAt: now,
        result: 'cancelled',
        startedAt: live?.startedAt ?? now,
      }),
      status,
      updatedAt: now,
    })
    return clearIndexingJob(next, action.indexId)
  }
  if (action.type === 'upsertChatRule') {
    const now = new Date().toISOString()
    const existing = state.chatRules[action.rule.id]
    const rule = normalizeChatRule({
      ...action.rule,
      createdAt: existing?.createdAt ?? action.rule.createdAt,
      updatedAt: action.rule.updatedAt || now,
    })
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
  if (action.type === 'startChatAssistantRetry') {
    return startChatAssistantRetry(state, action)
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
        pinnedExplorer: {
          ...state.ui.pinnedExplorer,
          chatThreadIds: removeExplorerPin(state.ui.pinnedExplorer.chatThreadIds, action.threadId),
        },
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
    requestContext: message.requestContext
      ? { knowledgeCollectionIds: message.requestContext.knowledgeCollectionIds?.slice() }
      : undefined,
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

function moveKnowledgeSessionGroup(
  state: ResearchDeskState,
  groupId: string,
  targetIndex: number,
): ResearchDeskState {
  if (!state.knowledgeSessionGroups[groupId]) return state
  const orderWithoutGroup = state.knowledgeSessionGroupOrder.filter((candidateId) => candidateId !== groupId)
  const boundedTargetIndex = Math.max(0, Math.min(orderWithoutGroup.length, targetIndex))
  const knowledgeSessionGroupOrder = [...orderWithoutGroup]
  knowledgeSessionGroupOrder.splice(boundedTargetIndex, 0, groupId)

  if (arraysEqual(knowledgeSessionGroupOrder, state.knowledgeSessionGroupOrder)) return state

  return {
    ...state,
    dirty: true,
    knowledgeSessionGroupOrder,
  }
}

function moveKnowledgeSessionToGroup(
  state: ResearchDeskState,
  sessionId: string,
  requestedGroupId: string | null,
  targetIndex: number,
): ResearchDeskState {
  if (!state.knowledgeSessions[sessionId]) return state
  const groupId = requestedGroupId && state.knowledgeSessionGroups[requestedGroupId]
    ? requestedGroupId
    : null
  const currentGroupId = normalizedKnowledgeSessionGroupId(state, sessionId)
  const currentSectionSessionIds = knowledgeSessionIdsForGroup(state, currentGroupId)
  const currentIndex = currentSectionSessionIds.indexOf(sessionId)
  const targetSessionIds = knowledgeSessionIdsForGroup(state, groupId).filter((id) => id !== sessionId)
  const boundedTargetIndex = Math.max(0, Math.min(targetSessionIds.length, targetIndex))
  if (currentGroupId === groupId && currentIndex === boundedTargetIndex) return state

  const knowledgeSessionGroupMemberships = { ...(state.knowledgeSessionGroupMemberships ?? {}) }
  if (groupId) {
    knowledgeSessionGroupMemberships[sessionId] = groupId
  } else {
    delete knowledgeSessionGroupMemberships[sessionId]
  }
  const knowledgeSessions = currentGroupId === groupId
    ? state.knowledgeSessions
    : {
      ...state.knowledgeSessions,
      [sessionId]: {
        ...state.knowledgeSessions[sessionId],
        updatedAt: new Date().toISOString(),
      },
    }

  return {
    ...state,
    dirty: true,
    knowledgeSessionGroupMemberships,
    knowledgeSessionOrder: insertKnowledgeSessionIntoSection(
      state,
      knowledgeSessionGroupMemberships,
      sessionId,
      groupId,
      boundedTargetIndex,
    ),
    knowledgeSessions,
  }
}

function insertKnowledgeSessionIntoSection(
  state: ResearchDeskState,
  memberships: Record<string, string | null>,
  sessionId: string,
  groupId: string | null,
  targetIndex: number,
) {
  const orderWithoutSession = state.knowledgeSessionOrder.filter((id) => id !== sessionId)
  const targetSessionIds = orderWithoutSession.filter((id) =>
    normalizedKnowledgeSessionGroupId(state, id, memberships) === groupId)
  const beforeSessionId = targetSessionIds[targetIndex]
  if (beforeSessionId) {
    return insertBefore(orderWithoutSession, sessionId, beforeSessionId)
  }

  const previousSessionId = targetSessionIds[targetIndex - 1]
  if (previousSessionId) {
    return insertAfter(orderWithoutSession, sessionId, previousSessionId)
  }

  const sectionInsertionIndex = emptyKnowledgeSessionSectionInsertionIndex(
    state,
    orderWithoutSession,
    memberships,
    groupId,
  )
  const nextOrder = [...orderWithoutSession]
  nextOrder.splice(sectionInsertionIndex, 0, sessionId)
  return nextOrder
}

function emptyKnowledgeSessionSectionInsertionIndex(
  state: ResearchDeskState,
  orderWithoutSession: string[],
  memberships: Record<string, string | null>,
  groupId: string | null,
) {
  const sectionKeys = [
    ...state.knowledgeSessionGroupOrder.filter((candidateId) => Boolean(state.knowledgeSessionGroups[candidateId])),
    null,
  ]
  const targetSectionIndex = Math.max(0, sectionKeys.findIndex((candidateId) => candidateId === groupId))
  for (const nextGroupId of sectionKeys.slice(targetSectionIndex + 1)) {
    const nextSessionId = orderWithoutSession.find((sessionId) => (
      normalizedKnowledgeSessionGroupId(state, sessionId, memberships) === nextGroupId
    ))
    if (nextSessionId) return orderWithoutSession.indexOf(nextSessionId)
  }

  for (const previousGroupId of sectionKeys.slice(0, targetSectionIndex).reverse()) {
    const previousSessionId = [...orderWithoutSession].reverse().find((sessionId) => (
      normalizedKnowledgeSessionGroupId(state, sessionId, memberships) === previousGroupId
    ))
    if (previousSessionId) return orderWithoutSession.indexOf(previousSessionId) + 1
  }

  return orderWithoutSession.length
}

function knowledgeSessionIdsForGroup(
  state: ResearchDeskState,
  groupId: string | null,
) {
  return state.knowledgeSessionOrder.filter((sessionId) => (
    normalizedKnowledgeSessionGroupId(state, sessionId) === groupId
  ))
}

function normalizedKnowledgeSessionGroupId(
  state: ResearchDeskState,
  sessionId: string,
  memberships = state.knowledgeSessionGroupMemberships,
) {
  const groupId = memberships[sessionId]
  return groupId && state.knowledgeSessionGroups[groupId] ? groupId : null
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
      claims: 0,
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
    requestContext: action.requestContext,
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
    requestContext: action.requestContext,
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

function startChatAssistantRetry(
  state: ResearchDeskState,
  action: Extract<ResearchDeskAction, { type: 'startChatAssistantRetry' }>,
): ResearchDeskState {
  const thread = state.chatThreads[action.threadId]
  if (!thread) return state

  const assistantIndex = thread.messages.findIndex((message) => (
    message.id === action.replacedAssistantMessageId
  ))
  const assistantMessage = assistantIndex >= 0 ? thread.messages[assistantIndex] : undefined
  const userMessage = assistantIndex > 0 ? thread.messages[assistantIndex - 1] : undefined
  if (
    !assistantMessage
    || assistantMessage.role !== 'assistant'
    || !userMessage
    || userMessage.role !== 'user'
    || !userMessage.contentMarkdown.trim()
  ) {
    return state
  }

  const retryMessage: ChatMessageRecord = {
    contentMarkdown: '',
    createdAt: action.createdAt,
    id: action.assistantMessageId,
    modelResolution: action.modelResolution,
    requestContext: action.requestContext,
    role: 'assistant',
  }
  const messages = [...thread.messages.slice(0, assistantIndex), retryMessage]

  return {
    ...state,
    chatThreads: {
      ...state.chatThreads,
      [thread.id]: threadWithMessages(thread, messages, action.createdAt),
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
  contentMarkdown: string,
): ChatMessageAttachmentRecord {
  return {
    attachedAt,
    contentMarkdown,
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
    return rule
      ? [createRuleAttachment(
        normalizeChatRule(rule),
        attachedAt,
        renderChatRuleAttachmentContent(state, rule, attachedAt),
      )]
      : []
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

function knowledgeSessionTitleFromQuestion(question: string): string {
  const title = question.trim().replace(/\s+/g, ' ').slice(0, 48)
  return title || 'Knowledge session'
}

function isPlaceholderKnowledgeSessionTitle(title: string): boolean {
  return [
    'Knowledge session',
    'New knowledge session',
    'New session',
    'Neue Sitzung',
    DEFAULT_KNOWLEDGE_SESSION_TITLE,
  ].includes(title)
}

function slugifyVectorHandle(value: string): string {
  const normalized = value
    .normalize('NFKD')
    .toLowerCase()
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)
  return normalized || 'index'
}

function uniqueVectorHandle(base: string, state: ProjectState, exceptId?: string): string {
  const taken = new Set(
    Object.values(state.vectorIndexes)
      .filter((index) => index.id !== exceptId)
      .map((index) => index.handle),
  )
  if (!taken.has(base)) return base
  let suffix = 2
  while (taken.has(`${base}-${suffix}`)) suffix += 1
  return `${base}-${suffix}`
}

function dimsForEmbedModel(model: EmbedModelId): number {
  return EMBED_MODELS.find((entry) => entry.id === model)?.dims ?? 3072
}

function knowledgeItemByRunId(
  state: ProjectState,
  runId: string,
): KnowledgeThreadItemRecord | null {
  for (const itemId of state.knowledgeItemOrder) {
    const item = state.knowledgeItems[itemId]
    if (item && item.runId === runId) return item
  }
  return null
}

function updateKnowledgeItem(
  state: ProjectState,
  itemId: string,
  update: (item: KnowledgeThreadItemRecord) => KnowledgeThreadItemRecord,
): ProjectState {
  const current = state.knowledgeItems[itemId]
  if (!current) return state
  const next = update(current)
  if (next === current) return state
  return {
    ...state,
    knowledgeItems: {
      ...state.knowledgeItems,
      [itemId]: next,
    },
  }
}

function touchKnowledgeSessions(
  state: ProjectState,
  sessionIds: Iterable<string>,
  updatedAt: string,
): ProjectState {
  const existingIds = [...new Set(sessionIds)].filter((sessionId) => Boolean(state.knowledgeSessions[sessionId]))
  if (existingIds.length === 0) return state

  const knowledgeSessions = { ...state.knowledgeSessions }
  for (const sessionId of existingIds) {
    const session = knowledgeSessions[sessionId]
    knowledgeSessions[sessionId] = { ...session, updatedAt }
  }
  return { ...state, knowledgeSessions }
}

/** Route one run event into the matching knowledge thread item (live
 * runs and demo asks share this path); terminal events also close the
 * item so the card never poses as still running. */
function applyEventToKnowledgeItem(
  state: ProjectState,
  event: ResearchRunEvent,
): ProjectState {
  const item = knowledgeItemByRunId(state, event.run_id)
  if (!item || item.status !== 'running') return state
  const updatedAt = new Date().toISOString()
  const terminalExit = event.type === 'inqtrix.run.failed' || event.type === 'inqtrix.run.cancelled'
  const terminalStatus = event.type === 'inqtrix.run.cancelled' ? 'cancelled' : 'failed'
  const updated = updateKnowledgeItem(terminalExit ? { ...state, dirty: true } : state, item.id, (current) => {
    const progress = applyKnowledgeRunEvent(current.progress, event)
    if (terminalExit) {
      const error = typeof (event.data.error as { message?: unknown } | undefined)?.message === 'string'
        ? String((event.data.error as { message?: unknown }).message)
        : undefined
      return {
        ...current,
        error: error ?? current.error,
        progress,
        status: terminalStatus,
      }
    }
    if (progress === current.progress) return current
    return { ...current, progress }
  })
  return terminalExit
    ? touchKnowledgeSessions(updated, [item.sessionId], updatedAt)
    : updated
}

function writeVectorIndex(state: ProjectState, index: VectorIndexRecord): ProjectState {
  return {
    ...state,
    dirty: true,
    vectorIndexes: { ...state.vectorIndexes, [index.id]: index },
  }
}

/** Prepend one finished run to the index history, newest first, capped. */
function appendVectorIndexHistory(
  index: VectorIndexRecord,
  entry: VectorIndexRunHistoryEntry,
): VectorIndexRunHistoryEntry[] {
  return [entry, ...(index.history ?? [])].slice(0, VECTOR_INDEX_HISTORY_LIMIT)
}

/** Elapsed run time from the live entry's start to *now* (ms, never negative). */
function runDurationMs(live: IndexingJobLive | undefined, now: string): number {
  if (!live) return 0
  return Math.max(0, new Date(now).getTime() - new Date(live.startedAt).getTime())
}

/** Drop the ephemeral live-progress entry for an index (run finished). */
function clearIndexingJob(state: ProjectState, indexId: string): ProjectState {
  if (!(indexId in state.indexingJobs)) return state
  const indexingJobs = { ...state.indexingJobs }
  delete indexingJobs[indexId]
  return { ...state, indexingJobs }
}

/** Removes the given file ids from every vector index's membership and
 * recomputes status. Used by file- and section-deletion cascades; an
 * in-flight `indexing` status is left untouched so a running simulation
 * is not disturbed. */
function dropFilesFromVectorIndexes(
  vectorIndexes: ProjectState['vectorIndexes'],
  removedFileIds: ReadonlySet<string>,
  updatedAt: string,
): { changed: boolean; vectorIndexes: ProjectState['vectorIndexes'] } {
  let changed = false
  const next: ProjectState['vectorIndexes'] = {}
  for (const [id, index] of Object.entries(vectorIndexes)) {
    const members = index.members.filter((member) => !removedFileIds.has(member.fileId))
    if (members.length === index.members.length) {
      next[id] = index
      continue
    }
    changed = true
    next[id] = {
      ...index,
      members,
      status: members.length === 0 && index.status !== 'indexing' ? 'ready' : index.status,
      updatedAt,
    }
  }
  return changed ? { changed, vectorIndexes: next } : { changed, vectorIndexes }
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
