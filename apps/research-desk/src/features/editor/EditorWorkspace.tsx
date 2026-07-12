import {
  useCallback,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  type ReactNode,
} from 'react'
import type { Editor } from '@tiptap/react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import {
  AlertTriangle,
  Anchor,
  Bold,
  BookOpen,
  Check,
  ChevronDown,
  Code2,
  Eye,
  FileDown,
  FileText,
  Folder,
  FolderOpen,
  FolderPlus,
  ListFilter,
  LoaderCircle,
  MessageSquareText,
  MessagesSquare,
  MoreHorizontal,
  PanelBottomClose,
  PanelBottomOpen,
  PanelRightClose,
  Paperclip,
  PencilLine,
  Pin,
  PinOff,
  Redo2,
  SendHorizontal,
  Scale,
  Sparkles,
  SquarePen,
  Trash2,
  Undo2,
  X,
} from '@/components/icons'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Chip } from '@/components/ui/chip'
import { AnimatedPanelBody, AnimatedResizableHandle } from '@/components/ui/animated-panel'
import { useAnimatedResizablePanelCollapse } from '@/components/ui/animated-panel-motion'
import { ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import { PanelToggle } from '@/components/ui/panel-toggle'
import { ResponsiveSidePanel } from '@/components/ui/responsive-side-panel'
import { WelcomeState } from '@/components/ui/welcome-state'
import { ComposerIconButton } from '@/features/composer/ComposerIconButton'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  EXPLORER_REVEAL_STEP,
  ExplorerFolderRow,
  ExplorerFolderToggle,
  ExplorerHistoryRow,
  ExplorerHistoryTitleInput,
  ExplorerRevealControls,
  ExplorerRunningIndicator,
  ExplorerSearchField,
  ExplorerSectionLabel,
  isExplorerActionTarget,
  isPastExplorerDragThreshold,
} from '@/components/ui/explorer-list'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { ComposerStopButton } from '@/features/composer/ComposerStopButton'
import {
  chatAttachmentChipsFromRefs,
  chatAttachmentsFromRefs,
  chatContextRefKey,
  chatRuleOptions,
  dedupeChatContextRefs,
  displayRelativeAge,
  editorCommentsForDocument,
  fileGroupMentionOptions,
  fileMentionOptions,
  projectEditorDocuments,
  projectEditorFolders,
  projectFileAssets,
  selectedEditorDocument,
  type ChatAttachmentChipModel,
  type ChatRuleOption,
  type CompletedReportOption,
  type FileGroupMentionOption,
  type FileMentionOption,
} from '@/features/project/selectors'
import type {
  ChatModelOption,
  ModelCatalogEntry,
  ChatModelTier,
  InqtrixCapabilities,
  NodeModelResolution,
} from '@/features/researchRuns/types'
import { ModelTierPicker } from '@/features/researchRuns/ModelTierPicker'
import { ContextTokenMeter } from '@/features/composer/ContextTokenMeter'
import { QuotaMeter } from '@/features/quota/QuotaMeter'
import { QuotaUsageFooter } from '@/features/quota/QuotaUsageFooter'
import {
  buildContextTokenModel,
  estimateTokensFromText,
  type ContextCategoryInput,
} from '@/features/files/contextTokens'
import type {
  ChatContextReferenceRecord,
  EditorCommentKind,
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorEvidencePreset,
  EditorFolderRecord,
  EditorSuggestionRecord,
  ProjectState,
} from '@/features/project/types'
import type { ResearchDeskAction } from '@/features/researchDesk/state'
import { useLocale } from '@/i18n/LocaleProvider'
import { useMediaQuery } from '@/features/researchDesk/hooks/useMediaQuery'
import { cn } from '@/lib/utils'
import { MarkdownEditorSurface } from './core/MarkdownEditorSurface'
import { COMMENT_KIND_ORDER, commentKindMeta, type CommentKindMeta } from './commentKinds'
import { editorCopy, type EditorCopy } from './editorCopy'
import { escapeCssIdentifier } from './editorDom'
import { suggestionDiffPlan } from './suggestionDiff'
import { resizeTextareaToRows } from '@/features/composer/textareaAutosize'
import {
  TextImproveButton,
  TextImproveFloatingLayer,
  useTextImprovement,
  type TextImprovementApiOptions,
} from '@/features/textImprove'
import { EditorDocumentChangesSection } from './EditorDocumentChangesSection'
import { EditorInstructionFeedbackCard } from './EditorInstructionFeedbackCard'
import { useEditorSuggestions, type EditorInstructionFeedback } from './useEditorSuggestions'
import {
  type MentionCategoryLabels,
  type MentionSources,
} from '@/features/composer/mention'
import { ContextChipLegend } from '@/features/composer/ContextChipLegend'
import { MentionComposer, type MentionComposerHandle } from '@/features/composer/MentionComposer'
import { type LabelResolver } from '@/features/composer/mentionDoc'
import { moveItem } from '@/features/composer/reorder'
import { Dropzone } from '@/features/files/Dropzone'
import { ingestFiles } from '@/features/files/ingest'
import { FILE_SECTION_TEMP_ID } from '@/features/files/sections'

type EditorWorkspaceProps = {
  apiKey?: string
  /** Server capability manifest; forwarded to useEditorSuggestions so the
   * client editor-run abort tracks the server wait. */
  capabilities: InqtrixCapabilities | null
  chatModelOptions: ChatModelOption[]
  chatModelOptionsStatus: 'available' | 'missing' | 'unresolved'
  chatModelCatalog?: ModelCatalogEntry[]
  defaultChatModel: NodeModelResolution | null
  dispatch: Dispatch<ResearchDeskAction>
  /** Loads attached file-asset bodies on demand before an AI run reads them
   * (M6c load-on-use); forwarded to useEditorSuggestions. */
  ensureAssetBodiesLoaded?: (assetIds: readonly string[]) => Promise<Map<string, string>>
  reportOptions: CompletedReportOption[]
  selectedModelTier: ChatModelTier | null
  state: ProjectState
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}

type EditorDocumentDropTarget = {
  folderId: string | null
  targetIndex: number
}

const EDITOR_TREE_PANEL_ID = 'editor-file-tree-panel'
const EDITOR_COMMENTS_PANEL_ID = 'editor-comments-panel'
// The center column and the center+comments subtree of the nested resizable
// groups. The tree collapses in the OUTER group, the comments panel in the
// INNER group -- mirroring the Knowledge Desk so the two side panels never
// share one percentage space and fight during independent collapse.
const EDITOR_CENTER_PANEL_ID = 'editor-center-panel'
const EDITOR_CENTER_COMMENTS_PANEL_ID = 'editor-center-comments-panel'

export default function EditorWorkspace({
  apiKey,
  capabilities,
  chatModelOptions,
  chatModelOptionsStatus,
  chatModelCatalog,
  defaultChatModel,
  dispatch,
  ensureAssetBodiesLoaded,
  reportOptions,
  selectedModelTier,
  state,
  textImprovement,
}: EditorWorkspaceProps) {
  const { locale } = useLocale()
  const copy = editorCopy[locale]
  const isDesktop = useMediaQuery('(min-width: 1024px)')
  const folders = useMemo(() => projectEditorFolders(state), [state.editorFolderOrder, state.editorFolders])
  const documents = useMemo(() => projectEditorDocuments(state), [state.editorDocumentOrder, state.editorDocuments])
  const activeDocument = selectedEditorDocument(state)
  const [activeEditor, setActiveEditor] = useState<Editor | null>(null)
  const [isMobileTreeOpen, setIsMobileTreeOpen] = useState(false)
  const [isMobileCommentsOpen, setIsMobileCommentsOpen] = useState(false)
  const comments = useMemo(
    () => editorCommentsForDocument(state, activeDocument?.id ?? null),
    [activeDocument?.id, state.editorComments],
  )
  const ruleOptions = useMemo(() => chatRuleOptions(state, 'editor'), [state.chatRuleOrder, state.chatRules])
  const fileOptions = useMemo(() => fileMentionOptions(state), [state.fileAssetOrder, state.fileAssets])
  const fileGroupOptions = useMemo(() => fileGroupMentionOptions(state), [state.fileGroupOrder, state.fileGroups])
  const [attachedCommentIds, setAttachedCommentIds] = useState<string[]>([])
  // Editor attachments come from two sources: positional `[N]` pills owned by the
  // composer (files/groups) and rules/dropped files attached out-of-band. They
  // merge -- pills first -- into one deduplicated ref list, mirroring the chat.
  const [pillRefs, setPillRefs] = useState<ChatContextReferenceRecord[]>([])
  const [extraRefs, setExtraRefs] = useState<ChatContextReferenceRecord[]>([])
  const attachedRefs = useMemo(
    () => dedupeChatContextRefs([...pillRefs, ...extraRefs]),
    [pillRefs, extraRefs],
  )
  const attachmentChips = useMemo(
    () => chatAttachmentChipsFromRefs(state, attachedRefs),
    [state, attachedRefs],
  )
  const editorSelectedCard = chatModelCatalog?.find(
    (entry) => entry.model_id === state.ui.selectedChatModel,
  )?.card ?? null
  // Per-category token estimate for the editor composer meter. The whole
  // document is always sent as context, so it is the `conversation` category;
  // the composer draft is added live inside EditorAssistantComposer.
  const editorContextBase = useMemo(() => {
    const attachments = chatAttachmentsFromRefs(state, attachedRefs)
    let documents = 0
    let reports = 0
    let rules = 0
    for (const attachment of attachments) {
      const tokens = estimateTokensFromText(attachment.contentMarkdown ?? '')
      if (attachment.kind === 'research-report') reports += tokens
      else if (attachment.kind === 'chat-rule') rules += tokens
      else documents += tokens
    }
    const conversation = estimateTokensFromText(activeDocument?.contentMarkdown ?? '')
    return { documents, reports, rules, conversation }
  }, [state, attachedRefs, activeDocument])
  const editorContextCapacity = {
    contextWindowTokens: editorSelectedCard?.context_window_tokens ?? null,
    reservedOutputTokens: editorSelectedCard?.max_output_tokens ?? 0,
  }
  const composerRef = useRef<MentionComposerHandle | null>(null)
  const [isAttachActive, setIsAttachActive] = useState(false)

  const handleCreateComment = useCallback((comment: EditorCommentThreadRecord) => {
    dispatch({ comment, type: 'createEditorComment' })
  }, [dispatch])

  const addExtraRef = useCallback((ref: ChatContextReferenceRecord) => {
    setExtraRefs((prev) => dedupeChatContextRefs([...prev, ref]))
  }, [])

  const clearEditorAttachments = useCallback(() => {
    setPillRefs([])
    setExtraRefs([])
    composerRef.current?.clear()
  }, [])

  const handleRemoveEditorChip = useCallback((ref: ChatContextReferenceRecord) => {
    composerRef.current?.removeRef(ref)
    setExtraRefs((prev) => prev.filter((item) => chatContextRefKey(item) !== chatContextRefKey(ref)))
  }, [])

  const editorPillKeys = useMemo(() => pillRefs.map(chatContextRefKey), [pillRefs])
  const editorPendingKeys = useMemo(() => extraRefs.map(chatContextRefKey), [extraRefs])

  const handleReorderEditorPill = useCallback((fromIndex: number, toIndex: number) => {
    composerRef.current?.reorderPill(fromIndex, toIndex)
  }, [])

  const handleReorderEditorPending = useCallback((fromIndex: number, toIndex: number) => {
    setExtraRefs((prev) => moveItem(prev, fromIndex, toIndex))
  }, [])

  const handleAttachEditorFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return
    const existingLabels = projectFileAssets(state).map((asset) => asset.label)
    const assets = await ingestFiles(files, { kind: 'editor', sectionId: FILE_SECTION_TEMP_ID }, undefined, existingLabels)
    if (assets.length === 0) return
    dispatch({ assets, type: 'ingestFileAssets' })
    for (const asset of assets) addExtraRef({ fileId: asset.id, kind: 'file-asset' })
  }, [addExtraRef, dispatch, state])

  const {
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
  } = useEditorSuggestions({
    activeDocument,
    activeEditor,
    apiKey,
    attachedCommentIds,
    attachedRefs,
    capabilities,
    comments,
    dispatch,
    ensureAssetBodiesLoaded,
    locale,
    onGlobalSuccess: () => {
      setAttachedCommentIds([])
      clearEditorAttachments()
      setIsAttachActive(false)
    },
    selectedModelTier,
    state,
  })

  const handleSelectSuggestion = useCallback((suggestionId: string) => {
    if (state.editorUi.isDiffVisible) {
      dispatch({ isVisible: false, type: 'setEditorDiffVisible' })
    }
    globalThis.setTimeout(() => {
      const escapedId = escapeCssIdentifier(suggestionId)
      const target = globalThis.document?.querySelector<HTMLElement>(
        `[data-suggestion-block-card="${escapedId}"], [data-suggestion-id="${escapedId}"]`,
      )
      if (!target) return
      target.scrollIntoView({ block: 'center', behavior: 'smooth' })
      target.classList.add('suggestion-scroll-pulse')
      globalThis.setTimeout(() => target.classList.remove('suggestion-scroll-pulse'), 900)
    }, 0)
  }, [dispatch, state.editorUi.isDiffVisible])

  // Both side panels drag-resize + persist their width (like Knowledge/Chat),
  // while the existing `isTreeVisible`/`isCommentPanelVisible` booleans keep
  // driving collapse/expand (already persisted). Nested groups: the tree lives
  // in the outer group, the comments panel in the inner group.
  const reduceMotion = useReducedMotion()
  const treeVisible = state.editorUi.isTreeVisible
  const commentsVisible = state.editorUi.isCommentPanelVisible
  const treeSize = state.ui.panelLayout.editorTree
  const commentsSize = state.ui.panelLayout.editorComments
  const treePanelMotion = useAnimatedResizablePanelCollapse({
    expanded: treeVisible,
    expandedSize: treeSize,
    reduceMotion,
  })
  const commentsPanelMotion = useAnimatedResizablePanelCollapse({
    expanded: commentsVisible,
    expandedSize: commentsSize,
    reduceMotion,
  })
  const treeLayout = {
    [EDITOR_CENTER_PANEL_ID]: treeVisible ? 100 - treeSize : 100,
    [EDITOR_TREE_PANEL_ID]: treeVisible ? treeSize : 0,
  }
  const commentsLayout = {
    [EDITOR_CENTER_COMMENTS_PANEL_ID]: commentsVisible ? 100 - commentsSize : 100,
    [EDITOR_COMMENTS_PANEL_ID]: commentsVisible ? commentsSize : 0,
  }

  const handleTreeVisibleChange = useCallback((isVisible: boolean) => {
    if (isDesktop) {
      dispatch({ isVisible, type: 'setEditorTreeVisible' })
      return
    }
    setIsMobileTreeOpen(isVisible)
  }, [dispatch, isDesktop])

  const handleCommentsVisibleChange = useCallback((isVisible: boolean) => {
    if (isDesktop) {
      dispatch({ isVisible, type: 'setEditorCommentPanelVisible' })
      return
    }
    setIsMobileCommentsOpen(isVisible)
  }, [dispatch, isDesktop])

  const fileTreePanel = (
    <EditorFileTree
      activeDocumentId={activeDocument?.id ?? null}
      copy={copy}
      dispatch={dispatch}
      documents={documents}
      folders={folders}
      pinnedDocumentIds={state.ui.pinnedExplorer.editorDocumentIds}
      reportOptions={reportOptions}
      runningDocumentId={
        (isGlobalRunning || runningCommentIds.length > 0 || runningSuggestionIds.length > 0)
          ? activeDocument?.id ?? null
          : null
      }
    />
  )

  const commentsPanel = (
    <EditorCommentsPanel
      comments={comments}
      copy={copy}
      dispatch={dispatch}
      onClose={() => handleCommentsVisibleChange(false)}
      onAcceptSuggestionGroup={handleAcceptSuggestionGroup}
      onAcceptSuggestion={handleAcceptSuggestion}
      onRejectSuggestionGroup={handleRejectSuggestionGroup}
      onRejectSuggestion={handleRejectSuggestion}
      onRunComment={handleRunComment}
      onSelectSuggestion={handleSelectSuggestion}
      runErrors={runErrors}
      runningCommentIds={runningCommentIds}
      selectedCommentId={state.editorUi.selectedCommentId}
      suggestions={documentSuggestions}
    />
  )

  const editorContent = (
    <main className="flex h-full w-full min-w-0 flex-col bg-background">
      {activeDocument ? (
        <>
          <EditorTopBar
            commentCount={comments.length}
            copy={copy}
            dispatch={dispatch}
            document={activeDocument}
            editor={activeEditor}
            isCommentPanelVisible={isDesktop ? state.editorUi.isCommentPanelVisible : isMobileCommentsOpen}
            isDiffVisible={state.editorUi.isDiffVisible}
            isDirty={state.dirty}
            isTreeVisible={isDesktop ? state.editorUi.isTreeVisible : isMobileTreeOpen}
            onCommentPanelVisibleChange={handleCommentsVisibleChange}
            onTreeVisibleChange={handleTreeVisibleChange}
            viewMode={state.editorUi.viewMode}
          />
          <div className="flex min-h-0 flex-1 flex-col">
            <MarkdownEditorSurface
              comments={comments}
              copy={copy}
              document={activeDocument}
              diffAnchorMarkdown={activeDocument.diffAnchorMarkdown ?? null}
              isDiffVisible={state.editorUi.isDiffVisible}
              mode={state.editorUi.viewMode}
              onChange={(contentMarkdown) => {
                dispatch({
                  contentMarkdown,
                  documentId: activeDocument.id,
                  type: 'updateEditorDocumentMarkdown',
                })
              }}
              onCreateComment={handleCreateComment}
              onEditorReady={setActiveEditor}
              onAcceptSuggestion={handleAcceptSuggestion}
              onEditSuggestion={handleEditSuggestionProposal}
              onMarkSuggestionStale={handleMarkSuggestionStale}
              onRefineSuggestion={handleRefineSuggestion}
              onRejectSuggestion={handleRejectSuggestion}
              onSelectComment={(commentId) => dispatch({ commentId, type: 'selectEditorComment' })}
              onStopSuggestion={handleStopSuggestionRun}
              runningSuggestionIds={runningSuggestionIds}
              selectedCommentId={state.editorUi.selectedCommentId}
              suggestionErrors={suggestionErrors}
              suggestions={documentSuggestions.filter((suggestion) => suggestion.status === 'pending')}
              textImprovement={textImprovement}
            />
            <EditorAssistantComposer
              attachedCommentIds={attachedCommentIds}
              attachmentChips={attachmentChips}
              chatModelOptions={chatModelOptions}
              chatModelOptionsStatus={chatModelOptionsStatus}
              comments={comments}
              composerRef={composerRef}
              copy={copy}
              defaultChatModel={defaultChatModel}
              dispatch={dispatch}
              draft={state.editorUi.assistantDraft}
              fileGroupOptions={fileGroupOptions}
              fileOptions={fileOptions}
              reportOptions={reportOptions}
              isAttachActive={isAttachActive}
              instructionFeedback={instructionFeedback}
              isRunning={isGlobalRunning}
              isVisible={state.editorUi.isAssistantVisible}
              isWideCanvas={isDesktop ? !state.editorUi.isTreeVisible && !state.editorUi.isCommentPanelVisible : true}
              onAttachFiles={(files) => void handleAttachEditorFiles(files)}
              onAttachRule={(ruleId) => addExtraRef({ kind: 'chat-rule', ruleId })}
              onRefsChange={setPillRefs}
              onReorderPending={handleReorderEditorPending}
              onReorderPill={handleReorderEditorPill}
              onRemoveAttachedComment={(commentId) =>
                setAttachedCommentIds((prev) => prev.filter((id) => id !== commentId))}
              onRemoveChip={handleRemoveEditorChip}
              pendingKeys={editorPendingKeys}
              pillKeys={editorPillKeys}
              onToggleAttach={() => {
                const openCollectIds = comments
                  .filter((comment) => comment.status === 'open' && comment.kind === 'collect')
                  .map((comment) => comment.id)
                setAttachedCommentIds(isAttachActive ? [] : openCollectIds)
                setIsAttachActive(!isAttachActive)
              }}
              onDismissInstructionFeedback={clearInstructionFeedback}
              onSend={() => {
                const instruction = composerRef.current?.getInstructionText().trim() ?? ''
                const hasAttachedComments = comments.some((comment) =>
                  comment.status === 'open'
                  && comment.kind === 'collect'
                  && attachedCommentIds.includes(comment.id))
                if (hasAttachedComments) {
                  void handleGlobalRun(instruction)
                } else {
                  void handleInstructionRun(instruction)
                }
              }}
              onStop={handleStopRun}
              ruleOptions={ruleOptions}
              selectedModelTier={selectedModelTier}
              chatModelCatalog={chatModelCatalog}
              selectedModel={state.ui.selectedChatModel}
              selectedEffort={state.ui.selectedChatEffort}
              editorContextBase={editorContextBase}
              editorContextCapacity={editorContextCapacity}
              textImprovement={textImprovement}
            />
          </div>
        </>
      ) : (
        <div className="relative flex min-h-0 flex-1 bg-background">
          <header className="absolute inset-x-0 top-0 z-10 grid inqtrix-panel-header grid-cols-[auto_1fr_auto] items-center gap-2 border-b border-border bg-background px-3">
            <EditorPanelToggle
              copy={copy}
              dispatch={dispatch}
              onToggle={handleTreeVisibleChange}
              side="left"
              visible={isDesktop ? state.editorUi.isTreeVisible : isMobileTreeOpen}
            />
            <span aria-hidden />
            <EditorPanelToggle
              copy={copy}
              dispatch={dispatch}
              onToggle={handleCommentsVisibleChange}
              side="right"
              visible={isDesktop ? state.editorUi.isCommentPanelVisible : isMobileCommentsOpen}
            />
          </header>
          <EditorEmptyState copy={copy} dispatch={dispatch} reportOptions={reportOptions} />
        </div>
      )}
    </main>
  )

  return (
    <div className="relative flex h-full min-h-0 min-w-0 overflow-hidden bg-canvas text-foreground">
      <ResizablePanelGroup
        className="min-h-0 min-w-0 flex-1 overflow-hidden"
        defaultLayout={treeLayout}
        elementRef={treePanelMotion.groupRef}
        onLayoutChanged={(layout) => {
          const size = layout[EDITOR_TREE_PANEL_ID]
          if (
            treeVisible
            && !treePanelMotion.isProgrammaticLayoutChange()
            && Number.isFinite(size)
            && size > 0
          ) {
            dispatch({ key: 'editorTree', size, type: 'setPanelLayoutSize' })
          }
        }}
        orientation="horizontal"
      >
        {isDesktop && (
          <>
            <ResizablePanel
              className="min-h-0 min-w-0 overflow-hidden"
              collapsedSize="0%"
              collapsible
              defaultSize={treeLayout[EDITOR_TREE_PANEL_ID]}
              id={EDITOR_TREE_PANEL_ID}
              maxSize="34%"
              minSize={treeVisible ? '16%' : '0%'}
              panelRef={treePanelMotion.panelRef}
            >
              <AnimatedPanelBody expanded={treeVisible} side="left">
                {fileTreePanel}
              </AnimatedPanelBody>
            </ResizablePanel>
            <AnimatedResizableHandle aria-label={copy.resizeTree} expanded={treeVisible} />
          </>
        )}
        <ResizablePanel
          className="min-h-0 min-w-0 overflow-hidden"
          defaultSize={treeLayout[EDITOR_CENTER_PANEL_ID]}
          id={EDITOR_CENTER_PANEL_ID}
          // Identity anchor — must stay unconditional so the TipTap editor
          // (instance, selection, undo history) survives the desktop/mobile flip.
          key={EDITOR_CENTER_PANEL_ID}
          minSize="40%"
        >
          <ResizablePanelGroup
            className="min-h-0 w-full overflow-hidden"
            defaultLayout={commentsLayout}
            elementRef={commentsPanelMotion.groupRef}
            onLayoutChanged={(layout) => {
              const size = layout[EDITOR_COMMENTS_PANEL_ID]
              if (
                commentsVisible
                && !commentsPanelMotion.isProgrammaticLayoutChange()
                && Number.isFinite(size)
                && size > 0
              ) {
                dispatch({ key: 'editorComments', size, type: 'setPanelLayoutSize' })
              }
            }}
            orientation="horizontal"
          >
            <ResizablePanel
              className="min-h-0 min-w-0 overflow-hidden"
              defaultSize={commentsLayout[EDITOR_CENTER_COMMENTS_PANEL_ID]}
              id={EDITOR_CENTER_COMMENTS_PANEL_ID}
              // Identity anchor — must stay unconditional (see above).
              key={EDITOR_CENTER_COMMENTS_PANEL_ID}
              minSize="50%"
            >
              {editorContent}
            </ResizablePanel>
            {isDesktop && (
              <>
                <AnimatedResizableHandle aria-label={copy.resizeComments} expanded={commentsVisible} />
                <ResizablePanel
                  className="min-h-0 min-w-0 overflow-hidden"
                  collapsedSize="0%"
                  collapsible
                  defaultSize={commentsLayout[EDITOR_COMMENTS_PANEL_ID]}
                  id={EDITOR_COMMENTS_PANEL_ID}
                  maxSize="38%"
                  minSize={commentsVisible ? '20%' : '0%'}
                  panelRef={commentsPanelMotion.panelRef}
                >
                  <AnimatedPanelBody expanded={commentsVisible} side="right">
                    {commentsPanel}
                  </AnimatedPanelBody>
                </ResizablePanel>
              </>
            )}
          </ResizablePanelGroup>
        </ResizablePanel>
      </ResizablePanelGroup>
      {!isDesktop && (
        <>
          <ResponsiveSidePanel
            closeLabel={copy.hideTree}
            controlsId={EDITOR_TREE_PANEL_ID}
            onOpenChange={setIsMobileTreeOpen}
            open={isMobileTreeOpen}
            showHeader={false}
            side="left"
            title={copy.documents}
          >
            {fileTreePanel}
          </ResponsiveSidePanel>
          <ResponsiveSidePanel
            closeLabel={copy.hideAssistant}
            controlsId={EDITOR_COMMENTS_PANEL_ID}
            onOpenChange={setIsMobileCommentsOpen}
            open={isMobileCommentsOpen}
            showHeader={false}
            side="right"
            title={copy.assistant}
          >
            {commentsPanel}
          </ResponsiveSidePanel>
        </>
      )}
    </div>
  )
}

function EditorFileTree({
  activeDocumentId,
  copy,
  dispatch,
  documents,
  folders,
  pinnedDocumentIds,
  reportOptions,
  runningDocumentId,
}: {
  activeDocumentId: string | null
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  documents: EditorDocumentRecord[]
  folders: EditorFolderRecord[]
  pinnedDocumentIds: readonly string[]
  reportOptions: CompletedReportOption[]
  runningDocumentId: string | null
}) {
  const [expandedFolderIds, setExpandedFolderIds] = useState<ReadonlySet<string>>(() => new Set(folders.map((folder) => folder.id)))
  const [visibleDocumentCounts, setVisibleDocumentCounts] = useState<Record<string, number>>({})
  const [draggedDocumentId, setDraggedDocumentId] = useState<string | null>(null)
  const [draggedFolderId, setDraggedFolderId] = useState<string | null>(null)
  const [documentDropTarget, setDocumentDropTarget] = useState<EditorDocumentDropTarget | null>(null)
  const [editingDocumentId, setEditingDocumentId] = useState<string | null>(null)
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null)
  const [folderDropTargetIndex, setFolderDropTargetIndex] = useState<number | null>(null)
  const [documentTitleDraft, setDocumentTitleDraft] = useState('')
  const [folderTitleDraft, setFolderTitleDraft] = useState('')
  const documentTitleInputRef = useRef<HTMLInputElement | null>(null)
  const folderTitleInputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)
  const suppressDocumentOpenClickRef = useRef(false)
  const suppressFolderToggleClickRef = useRef(false)
  const pinnedDocumentIdSet = new Set(pinnedDocumentIds)
  const pinnedDocuments = documents.filter((document) => pinnedDocumentIdSet.has(document.id))
  const treeDocuments = documents.filter((document) => !pinnedDocumentIdSet.has(document.id))
  const ungroupedDocuments = treeDocuments.filter((document) => !document.folderId || !folders.some((folder) => folder.id === document.folderId))
  const hasFolders = folders.length > 0
  const [searchQuery, setSearchQuery] = useState('')
  const trimmedQuery = searchQuery.trim().toLowerCase()
  const isSearching = trimmedQuery.length > 0
  // Title search runs over the already-loaded documents (the tree hydrates all
  // document metadata up front; only bodies are lazy), so it is a client-side
  // filter across every folder — no server round-trip. Matches keep the
  // document order so the result list reads top-to-bottom like the tree.
  const searchResults = useMemo(
    () => (isSearching
      ? documents.filter((document) => document.title.toLowerCase().includes(trimmedQuery))
      : []),
    [documents, isSearching, trimmedQuery],
  )

  useEffect(() => {
    setExpandedFolderIds((current) => new Set([...current, ...folders.map((folder) => folder.id)]))
  }, [folders])

  function visibleDocumentCount(sectionId: string) {
    return visibleDocumentCounts[sectionId] ?? EXPLORER_REVEAL_STEP
  }

  function showMoreDocuments(sectionId: string, total: number) {
    setVisibleDocumentCounts((current) => ({
      ...current,
      [sectionId]: Math.min((current[sectionId] ?? EXPLORER_REVEAL_STEP) + EXPLORER_REVEAL_STEP, total),
    }))
  }

  function showLessDocuments(sectionId: string) {
    setVisibleDocumentCounts((current) => ({
      ...current,
      [sectionId]: EXPLORER_REVEAL_STEP,
    }))
  }

  useLayoutEffect(() => {
    if (!editingDocumentId) return
    documentTitleInputRef.current?.focus()
    documentTitleInputRef.current?.select()
  }, [editingDocumentId])

  useLayoutEffect(() => {
    if (!editingFolderId) return
    folderTitleInputRef.current?.focus()
    folderTitleInputRef.current?.select()
  }, [editingFolderId])

  function toggleFolder(folderId: string) {
    if (suppressFolderToggleClickRef.current) {
      suppressFolderToggleClickRef.current = false
      return
    }
    setExpandedFolderIds((current) => {
      const next = new Set(current)
      if (next.has(folderId)) {
        next.delete(folderId)
      } else {
        next.add(folderId)
      }
      return next
    })
  }

  function openDocumentFromTree(documentId: string) {
    if (suppressDocumentOpenClickRef.current) {
      suppressDocumentOpenClickRef.current = false
      return
    }
    dispatch({ documentId, type: 'openEditorDocument' })
  }

  function startDocumentTitleEdit(document: EditorDocumentRecord) {
    setEditingDocumentId(document.id)
    setDocumentTitleDraft(document.title)
  }

  function commitDocumentTitleEdit() {
    if (!editingDocumentId) return
    const title = documentTitleDraft.trim()
    if (title) {
      dispatch({ documentId: editingDocumentId, title, type: 'renameEditorDocument' })
    }
    setEditingDocumentId(null)
    setDocumentTitleDraft('')
  }

  function cancelDocumentTitleEdit() {
    setEditingDocumentId(null)
    setDocumentTitleDraft('')
  }

  function startFolderTitleEdit(folder: EditorFolderRecord) {
    setEditingFolderId(folder.id)
    setFolderTitleDraft(folder.title)
  }

  function commitFolderTitleEdit() {
    if (!editingFolderId) return
    const title = folderTitleDraft.trim()
    if (title) {
      dispatch({ folderId: editingFolderId, title, type: 'renameEditorFolder' })
    }
    setEditingFolderId(null)
    setFolderTitleDraft('')
  }

  function cancelFolderTitleEdit() {
    setEditingFolderId(null)
    setFolderTitleDraft('')
  }

  function readFolderDropTarget(clientY: number, excludedFolderId = draggedFolderId) {
    const container = listRef.current
    if (!container) return null
    const folderElements = Array.from(container.querySelectorAll<HTMLElement>('[data-editor-draggable-folder-id]'))
      .filter((folderElement) => (
        folderElement.dataset.editorDraggableFolderId !== excludedFolderId
        && folderElement.getBoundingClientRect().height > 0
      ))
    for (const [index, folderElement] of folderElements.entries()) {
      const rect = folderElement.getBoundingClientRect()
      if (clientY < rect.top + rect.height / 2) return index
    }
    return folderElements.length
  }

  function beginFolderDrag(event: ReactPointerEvent<HTMLElement>, folderId: string) {
    if (event.button !== 0 || isExplorerActionTarget(event.target)) return
    const startX = event.clientX
    const startY = event.clientY
    let didStartDrag = false

    function startDrag(moveEvent: PointerEvent) {
      didStartDrag = true
      suppressFolderToggleClickRef.current = true
      setDraggedFolderId(folderId)
      setFolderDropTargetIndex(readFolderDropTarget(moveEvent.clientY, folderId))
    }

    function handlePointerMove(moveEvent: PointerEvent) {
      if (!didStartDrag) {
        if (!isPastExplorerDragThreshold(startX, startY, moveEvent)) return
        startDrag(moveEvent)
      }
      moveEvent.preventDefault()
      setFolderDropTargetIndex(readFolderDropTarget(moveEvent.clientY, folderId))
    }

    function finishPointerDrag(upEvent: PointerEvent) {
      const nextDropTarget = didStartDrag ? readFolderDropTarget(upEvent.clientY, folderId) : null
      cleanupPointerDrag()
      if (nextDropTarget === null) return
      dispatch({ folderId, targetIndex: nextDropTarget, type: 'moveEditorFolder' })
    }

    function cleanupPointerDrag() {
      document.removeEventListener('pointermove', handlePointerMove)
      document.removeEventListener('pointerup', finishPointerDrag)
      document.removeEventListener('pointercancel', cleanupPointerDrag)
      setDraggedFolderId(null)
      setFolderDropTargetIndex(null)
      if (didStartDrag) {
        window.setTimeout(() => {
          suppressFolderToggleClickRef.current = false
        }, 0)
      }
    }

    document.addEventListener('pointermove', handlePointerMove)
    document.addEventListener('pointerup', finishPointerDrag)
    document.addEventListener('pointercancel', cleanupPointerDrag)
  }

  function readDocumentDropTarget(clientY: number, excludedDocumentId = draggedDocumentId): EditorDocumentDropTarget | null {
    const container = listRef.current
    if (!container) return null
    const sectionElements = Array.from(container.querySelectorAll<HTMLElement>('[data-editor-document-section]'))
      .filter((sectionElement) => sectionElement.getBoundingClientRect().height > 0)
    if (sectionElements.length === 0) return null
    const sectionElement = sectionElements.find((candidate) => {
      const rect = candidate.getBoundingClientRect()
      return clientY >= rect.top - 8 && clientY <= rect.bottom + 8
    }) ?? sectionElements.reduce((nearest, candidate) => {
      const nearestRect = nearest.getBoundingClientRect()
      const candidateRect = candidate.getBoundingClientRect()
      const nearestDistance = Math.min(Math.abs(clientY - nearestRect.top), Math.abs(clientY - nearestRect.bottom))
      const candidateDistance = Math.min(Math.abs(clientY - candidateRect.top), Math.abs(clientY - candidateRect.bottom))
      return candidateDistance < nearestDistance ? candidate : nearest
    })
    const folderKey = sectionElement.dataset.editorFolderId ?? '__ungrouped__'
    const folderId = folderKey === '__ungrouped__' ? null : folderKey
    const documentElements = Array.from(sectionElement.querySelectorAll<HTMLElement>('[data-editor-document-id]'))
      .filter((documentElement) => (
        documentElement.dataset.editorDocumentId !== excludedDocumentId
        && documentElement.getBoundingClientRect().height > 0
      ))
    for (const [index, documentElement] of documentElements.entries()) {
      const rect = documentElement.getBoundingClientRect()
      if (clientY < rect.top + rect.height / 2) return { folderId, targetIndex: index }
    }
    return { folderId, targetIndex: documentElements.length }
  }

  function beginDocumentDrag(event: ReactPointerEvent<HTMLElement>, documentId: string) {
    if (event.button !== 0 || isExplorerActionTarget(event.target)) return
    const startX = event.clientX
    const startY = event.clientY
    let didStartDrag = false

    function startDrag(moveEvent: PointerEvent) {
      didStartDrag = true
      suppressDocumentOpenClickRef.current = true
      setDraggedDocumentId(documentId)
      setDocumentDropTarget(readDocumentDropTarget(moveEvent.clientY, documentId))
    }

    function handlePointerMove(moveEvent: PointerEvent) {
      if (!didStartDrag) {
        if (!isPastExplorerDragThreshold(startX, startY, moveEvent)) return
        startDrag(moveEvent)
      }
      moveEvent.preventDefault()
      setDocumentDropTarget(readDocumentDropTarget(moveEvent.clientY, documentId))
    }

    function finishPointerDrag(upEvent: PointerEvent) {
      const nextDropTarget = didStartDrag ? readDocumentDropTarget(upEvent.clientY, documentId) : null
      cleanupPointerDrag()
      if (!nextDropTarget) return
      dispatch({
        documentId,
        folderId: nextDropTarget.folderId,
        targetIndex: nextDropTarget.targetIndex,
        type: 'moveEditorDocumentToFolder',
      })
    }

    function cleanupPointerDrag() {
      document.removeEventListener('pointermove', handlePointerMove)
      document.removeEventListener('pointerup', finishPointerDrag)
      document.removeEventListener('pointercancel', cleanupPointerDrag)
      setDraggedDocumentId(null)
      setDocumentDropTarget(null)
      if (didStartDrag) {
        window.setTimeout(() => {
          suppressDocumentOpenClickRef.current = false
        }, 0)
      }
    }

    document.addEventListener('pointermove', handlePointerMove)
    document.addEventListener('pointerup', finishPointerDrag)
    document.addEventListener('pointercancel', cleanupPointerDrag)
  }

  const ungroupedVisibleCount = visibleDocumentCount('__ungrouped__')

  return (
    <aside className="inqtrix-contained-panel flex h-full w-full min-w-0 flex-col bg-surface">
      <div className="flex inqtrix-panel-header items-center justify-between border-b border-border px-3">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="t-section truncate">{copy.documents}</h2>
        </div>
        <div className="flex items-center gap-1">
          <ImportReportMenu copy={copy} dispatch={dispatch} reportOptions={reportOptions} />
          <TooltipButton
            label={copy.createFolder}
            onClick={() => dispatch({ title: copy.createFolder, type: 'createEditorFolder' })}
          >
            <FolderPlus className="size-4" />
          </TooltipButton>
          <TooltipButton
            label={copy.createDocument}
            onClick={() => dispatch({ type: 'createEditorDocument' })}
          >
            <SquarePen className="size-4" />
          </TooltipButton>
        </div>
      </div>
      <ExplorerSearchField
        clearLabel={copy.searchClear}
        label={copy.searchDocuments}
        onChange={setSearchQuery}
        onClear={() => setSearchQuery('')}
        placeholder={copy.searchDocuments}
        value={searchQuery}
      />
      <ScrollArea className="min-h-0 flex-1">
        <div className="inqtrix-explorer-list space-y-1 p-2" ref={listRef}>
          {isSearching ? (
            searchResults.length > 0 ? (
              <div className="space-y-0.5">
                {searchResults.map((document) => (
                  <EditorDocumentTreeItem
                    beginDocumentDrag={beginDocumentDrag}
                    cancelTitleEdit={cancelDocumentTitleEdit}
                    commitTitleEdit={commitDocumentTitleEdit}
                    copy={copy}
                    document={document}
                    isActive={activeDocumentId === document.id}
                    isDragging={false}
                    isEditing={editingDocumentId === document.id}
                    isNested={false}
                    isPinned={pinnedDocumentIdSet.has(document.id)}
                    isRunning={runningDocumentId === document.id}
                    key={document.id}
                    onDelete={() => dispatch({ documentId: document.id, type: 'deleteEditorDocument' })}
                    onDraftChange={setDocumentTitleDraft}
                    onOpen={() => openDocumentFromTree(document.id)}
                    onTogglePinned={() => dispatch({ documentId: document.id, type: 'togglePinnedEditorDocument' })}
                    showAfterIndicator={false}
                    showBeforeIndicator={false}
                    startTitleEdit={startDocumentTitleEdit}
                    titleDraft={documentTitleDraft}
                    titleInputRef={documentTitleInputRef}
                  />
                ))}
              </div>
            ) : (
              <p className="px-2 py-6 text-center t-meta-sm text-muted-foreground">{copy.searchNoResults}</p>
            )
          ) : (
          <>
          {pinnedDocuments.length > 0 ? (
            <section className="space-y-0.5">
              <ExplorerSectionLabel className="pt-0">{copy.pinned}</ExplorerSectionLabel>
              {pinnedDocuments.map((document) => (
                <EditorDocumentTreeItem
                  beginDocumentDrag={beginDocumentDrag}
                  cancelTitleEdit={cancelDocumentTitleEdit}
                  commitTitleEdit={commitDocumentTitleEdit}
                  copy={copy}
                  document={document}
                  isActive={activeDocumentId === document.id}
                  isDragging={draggedDocumentId === document.id}
                  isEditing={editingDocumentId === document.id}
                  isNested={false}
                  isPinned
                  isRunning={runningDocumentId === document.id}
                  key={document.id}
                  onDelete={() => dispatch({ documentId: document.id, type: 'deleteEditorDocument' })}
                  onDraftChange={setDocumentTitleDraft}
                  onOpen={() => openDocumentFromTree(document.id)}
                  onTogglePinned={() => dispatch({ documentId: document.id, type: 'togglePinnedEditorDocument' })}
                  showAfterIndicator={false}
                  showBeforeIndicator={false}
                  startTitleEdit={startDocumentTitleEdit}
                  titleDraft={documentTitleDraft}
                  titleInputRef={documentTitleInputRef}
                />
              ))}
            </section>
          ) : null}
          {folders.map((folder, folderIndex) => {
            const isExpanded = expandedFolderIds.has(folder.id)
            const isDraggingFolder = draggedFolderId === folder.id
            const showFolderBeforeIndicator = folderDropTargetIndex === folderIndex
            const showFolderAfterIndicator = folderDropTargetIndex === folders.length && folderIndex === folders.length - 1
            const showDropFrame = documentDropTarget?.folderId === folder.id
            const folderDocuments = treeDocuments.filter((document) => document.folderId === folder.id)
            const visibleCount = visibleDocumentCount(folder.id)
            return (
              <section
                className={cn(
                  'relative transition-colors',
                  showDropFrame && 'bg-brand-subtle/45',
                  isDraggingFolder && 'scale-[0.995] opacity-80 shadow-[0_8px_20px_var(--shadow-soft)] ring-1 ring-ring/40',
                )}
                data-editor-document-section
                data-editor-draggable-folder-id={folder.id}
                data-editor-folder-id={folder.id}
                key={folder.id}
              >
                {showFolderBeforeIndicator ? <DropIndicator className="-top-1" /> : null}
                {showFolderAfterIndicator ? <DropIndicator className="-bottom-1" /> : null}
                <ExplorerFolderRow
                  onPointerDown={(event) => beginFolderDrag(event, folder.id)}
                  actions={(
                    <>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            aria-label={copy.options}
                            className="size-6 shrink-0 text-foreground/55 hover:text-foreground"
                            size="icon"
                            type="button"
                            variant="ghost"
                          >
                            <MoreHorizontal className="icon-sm" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-56">
                          <DropdownMenuItem onSelect={() => startFolderTitleEdit(folder)}>
                            <PencilLine className="icon-sm" />
                            {copy.renameFolder}
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive focus:text-destructive"
                            onSelect={() => dispatch({ folderId: folder.id, type: 'deleteEditorFolder' })}
                          >
                            <Trash2 className="icon-sm" />
                            {copy.deleteFolder}
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                      <button
                        aria-label={copy.createDocument}
                        className="grid size-6 shrink-0 place-items-center rounded-sm text-foreground/50 transition hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        onClick={() => dispatch({ folderId: folder.id, type: 'createEditorDocument' })}
                        type="button"
                      >
                        <SquarePen className="icon-sm" />
                      </button>
                    </>
                  )}
                >
                  {editingFolderId === folder.id ? (
                    <span className="flex min-h-8 min-w-0 items-center gap-1.5" data-explorer-action>
                      <FolderOpen className="icon-sm shrink-0 text-muted-foreground" />
                      <input
                        aria-label={copy.renameFolder}
                        className="min-w-0 flex-1 rounded-sm border-0 bg-background/85 px-1.5 py-0.5 t-list text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        onBlur={commitFolderTitleEdit}
                        onChange={(event) => setFolderTitleDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') {
                            event.preventDefault()
                            commitFolderTitleEdit()
                          }
                          if (event.key === 'Escape') {
                            event.preventDefault()
                            cancelFolderTitleEdit()
                          }
                        }}
                        ref={folderTitleInputRef}
                        value={folderTitleDraft}
                      />
                    </span>
                  ) : (
                    <ExplorerFolderToggle
                      count={folderDocuments.length}
                      expanded={isExpanded}
                      icon={isExpanded ? <FolderOpen className="icon-sm shrink-0" /> : <Folder className="icon-sm shrink-0" />}
                      label={`${isExpanded ? copy.hideTree : copy.showTree}: ${folder.title}`}
                      onDoubleClick={(event) => {
                        event.preventDefault()
                        startFolderTitleEdit(folder)
                      }}
                      onToggle={() => toggleFolder(folder.id)}
                      title={folder.title}
                    />
                  )}
                </ExplorerFolderRow>
                {isExpanded ? (
                  <div className="space-y-0.5">
                    {folderDocuments.slice(0, visibleCount).map((document, index) => (
                      <EditorDocumentTreeItem
                        beginDocumentDrag={beginDocumentDrag}
                        cancelTitleEdit={cancelDocumentTitleEdit}
                        commitTitleEdit={commitDocumentTitleEdit}
                        copy={copy}
                        document={document}
                        isActive={activeDocumentId === document.id}
                        isDragging={draggedDocumentId === document.id}
                        isEditing={editingDocumentId === document.id}
                        isNested
                        isPinned={false}
                        isRunning={runningDocumentId === document.id}
                        key={document.id}
                        onDelete={() => dispatch({ documentId: document.id, type: 'deleteEditorDocument' })}
                        onDraftChange={setDocumentTitleDraft}
                        onOpen={() => openDocumentFromTree(document.id)}
                        onTogglePinned={() => dispatch({ documentId: document.id, type: 'togglePinnedEditorDocument' })}
                        showAfterIndicator={documentDropTarget?.folderId === folder.id && documentDropTarget.targetIndex === Math.min(visibleCount, folderDocuments.length) && index === Math.min(visibleCount, folderDocuments.length) - 1}
                        showBeforeIndicator={documentDropTarget?.folderId === folder.id && documentDropTarget.targetIndex === index}
                        startTitleEdit={startDocumentTitleEdit}
                        titleDraft={documentTitleDraft}
                        titleInputRef={documentTitleInputRef}
                      />
                    ))}
                    <ExplorerRevealControls
                      onShowLess={() => showLessDocuments(folder.id)}
                      onShowMore={() => showMoreDocuments(folder.id, folderDocuments.length)}
                      showLessLabel={copy.showLess}
                      showMoreLabel={copy.showMore}
                      total={folderDocuments.length}
                      visibleCount={visibleCount}
                    />
                    {folderDocuments.length === 0 ? (
                      <p className="rounded-md px-2 py-1.5 t-meta-sm font-medium text-muted-foreground">{copy.dropIntoFolder}</p>
                    ) : null}
                  </div>
                ) : null}
              </section>
            )
          })}
          {ungroupedDocuments.length > 0 || documents.length === 0 || hasFolders ? (
            <section
              className={cn('relative space-y-0.5 rounded-md', documentDropTarget?.folderId === null && 'bg-brand-subtle/30')}
              data-editor-document-section
              data-editor-folder-id="__ungrouped__"
            >
              {hasFolders && <p className="t-caption px-1.5 py-1 text-muted-foreground">{copy.documents}</p>}
              {ungroupedDocuments.slice(0, ungroupedVisibleCount).map((document, index) => (
                <EditorDocumentTreeItem
                  beginDocumentDrag={beginDocumentDrag}
                  cancelTitleEdit={cancelDocumentTitleEdit}
                  commitTitleEdit={commitDocumentTitleEdit}
                  copy={copy}
                  document={document}
                  isActive={activeDocumentId === document.id}
                  isDragging={draggedDocumentId === document.id}
                  isEditing={editingDocumentId === document.id}
                  isNested={false}
                  isPinned={false}
                  isRunning={runningDocumentId === document.id}
                  key={document.id}
                  onDelete={() => dispatch({ documentId: document.id, type: 'deleteEditorDocument' })}
                  onDraftChange={setDocumentTitleDraft}
                  onOpen={() => openDocumentFromTree(document.id)}
                  onTogglePinned={() => dispatch({ documentId: document.id, type: 'togglePinnedEditorDocument' })}
                  showAfterIndicator={documentDropTarget?.folderId === null && documentDropTarget.targetIndex === Math.min(ungroupedVisibleCount, ungroupedDocuments.length) && index === Math.min(ungroupedVisibleCount, ungroupedDocuments.length) - 1}
                  showBeforeIndicator={documentDropTarget?.folderId === null && documentDropTarget.targetIndex === index}
                  startTitleEdit={startDocumentTitleEdit}
                  titleDraft={documentTitleDraft}
                  titleInputRef={documentTitleInputRef}
                />
              ))}
              <ExplorerRevealControls
                onShowLess={() => showLessDocuments('__ungrouped__')}
                onShowMore={() => showMoreDocuments('__ungrouped__', ungroupedDocuments.length)}
                showLessLabel={copy.showLess}
                showMoreLabel={copy.showMore}
                total={ungroupedDocuments.length}
                visibleCount={ungroupedVisibleCount}
              />
              {documents.length === 0 ? (
                <p className="px-2 py-6 text-center text-xs text-muted-foreground">{copy.noDocuments}</p>
              ) : null}
            </section>
          ) : null}
          </>
          )}
        </div>
      </ScrollArea>
      <QuotaUsageFooter dimensions={['llm_tokens']} />
    </aside>
  )
}

function EditorDocumentTreeItem({
  beginDocumentDrag,
  cancelTitleEdit,
  commitTitleEdit,
  copy,
  document,
  isActive,
  isDragging,
  isEditing,
  isNested,
  isPinned,
  isRunning,
  onDelete,
  onDraftChange,
  onOpen,
  onTogglePinned,
  showAfterIndicator,
  showBeforeIndicator,
  startTitleEdit,
  titleDraft,
  titleInputRef,
}: {
  beginDocumentDrag: (event: ReactPointerEvent<HTMLElement>, documentId: string) => void
  cancelTitleEdit: () => void
  commitTitleEdit: () => void
  copy: EditorCopy
  document: EditorDocumentRecord
  isActive: boolean
  isDragging: boolean
  isEditing: boolean
  isNested: boolean
  isPinned: boolean
  isRunning: boolean
  onDelete: () => void
  onDraftChange: (value: string) => void
  onOpen: () => void
  onTogglePinned: () => void
  showAfterIndicator: boolean
  showBeforeIndicator: boolean
  startTitleEdit: (document: EditorDocumentRecord) => void
  titleDraft: string
  titleInputRef: RefObject<HTMLInputElement | null>
}) {
  const { locale } = useLocale()
  const timeLabel = displayRelativeAge(document.updatedAt, locale)

  return (
    <div className="relative" data-editor-document-id={document.id}>
      {showBeforeIndicator ? <DropIndicator className="-top-1" /> : null}
      {showAfterIndicator ? <DropIndicator className="-bottom-1" /> : null}
      <ExplorerHistoryRow
        actions={[
          {
            icon: isPinned ? <PinOff className="icon-sm" /> : <Pin className="icon-sm" />,
            label: isPinned ? copy.unpinDocument : copy.pinDocument,
            onSelect: onTogglePinned,
          },
          {
            destructive: true,
            icon: <Trash2 className="icon-sm" />,
            label: copy.deleteDocument,
            onSelect: onDelete,
          },
        ]}
        active={isActive}
        dragging={isDragging}
        indicator={isRunning ? <ExplorerRunningIndicator label={copy.runningSuggestion} /> : undefined}
        nested={isNested}
        onPointerDown={(event) => beginDocumentDrag(event, document.id)}
        onSelect={onOpen}
        onStartRename={() => startTitleEdit(document)}
        renameEditor={isEditing ? (
          <ExplorerHistoryTitleInput
            inputRef={titleInputRef}
            label={copy.renameDocument}
            onCancel={cancelTitleEdit}
            onChange={onDraftChange}
            onCommit={commitTitleEdit}
            value={titleDraft}
          />
        ) : undefined}
        renameLabel={copy.renameDocument}
        timeLabel={timeLabel}
        title={document.title}
      />
    </div>
  )
}

function DropIndicator({ className }: { className?: string }) {
  return (
    <span className={cn('pointer-events-none absolute left-1 right-1 z-20 h-0.5 rounded-full bg-brand shadow-[0_0_0_1px_var(--background)]', className)} />
  )
}

function ImportReportMenu({
  copy,
  dispatch,
  reportOptions,
  triggerClassName,
  variant = 'icon',
}: {
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  reportOptions: CompletedReportOption[]
  triggerClassName?: string
  variant?: 'button' | 'icon'
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={copy.importReport}
          className={cn(
            variant === 'icon' && 'size-8 rounded-md',
            variant === 'button' && 'justify-center',
            triggerClassName,
          )}
          size={variant === 'button' ? 'default' : 'icon'}
          type="button"
          variant={variant === 'button' ? 'outline' : 'ghost'}
        >
          <BookOpen className="size-4" />
          {variant === 'button' ? <span className="min-w-0 truncate">{copy.importReport}</span> : null}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="w-64 max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl p-0 shadow-lg"
      >
        <div className="flex items-center gap-1.5 border-b border-border px-2.5 py-1.5">
          <span className="t-meta-sm font-medium text-muted-foreground">{copy.importReport}</span>
          <span className="ml-auto t-hint tabular-nums text-muted-foreground/50">
            {reportOptions.length}
          </span>
        </div>
        <div className="py-1">
          {reportOptions.length === 0 ? (
            <div className="px-2.5 py-2 t-meta text-muted-foreground">
              {copy.noReports}
            </div>
          ) : reportOptions.map((report) => (
            <DropdownMenuItem
              className="group relative w-full min-w-0 items-start gap-2.5 rounded-none px-2.5 py-1.5 hover:bg-accent/50 focus:bg-accent/80 data-[highlighted]:bg-accent/80"
              key={report.runId}
              onSelect={() => dispatch({ runId: report.runId, type: 'importResearchReportToEditor' })}
            >
              <span className="absolute inset-y-1 left-0 w-0.5 rounded-full bg-brand opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100 group-data-[highlighted]:opacity-100" />
              <FileText className="mt-0.5 icon-md shrink-0 text-muted-foreground/70 transition-colors group-hover:text-brand group-focus:text-brand group-data-[highlighted]:text-brand" />
              <span className="min-w-0 flex-1">
                <span className="block max-w-full truncate t-list text-foreground">
                  @research:{report.label}
                </span>
                <span className="block max-w-full truncate t-meta-sm text-muted-foreground">
                  {report.title}
                </span>
              </span>
            </DropdownMenuItem>
          ))}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/**
 * The single header control that collapses/expands one editor side panel (file
 * tree on the left, comments on the right). Lives at the leading/trailing edge
 * of the editor header and stays put across the document/empty-state swap, so
 * the toggles never move and a collapsed panel always has a reopen path.
 */
function EditorPanelToggle({
  copy,
  dispatch,
  onToggle,
  side,
  visible,
}: {
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  onToggle?: (visible: boolean) => void
  side: 'left' | 'right'
  visible: boolean
}) {
  return side === 'left' ? (
    <PanelToggle
      collapseLabel={copy.hideTree}
      controlsId={EDITOR_TREE_PANEL_ID}
      expandLabel={copy.showTree}
      expanded={visible}
      onToggle={(next) => {
        if (onToggle) {
          onToggle(next)
          return
        }
        dispatch({ isVisible: next, type: 'setEditorTreeVisible' })
      }}
      side="left"
    />
  ) : (
    <PanelToggle
      collapseLabel={copy.hideAssistant}
      controlsId={EDITOR_COMMENTS_PANEL_ID}
      expandLabel={copy.showAssistant}
      expanded={visible}
      onToggle={(next) => {
        if (onToggle) {
          onToggle(next)
          return
        }
        dispatch({ isVisible: next, type: 'setEditorCommentPanelVisible' })
      }}
      side="right"
    />
  )
}

function EditorTopBar({
  commentCount,
  copy,
  dispatch,
  document,
  editor,
  isCommentPanelVisible,
  isDiffVisible,
  isDirty,
  isTreeVisible,
  onCommentPanelVisibleChange,
  onTreeVisibleChange,
  viewMode,
}: {
  commentCount: number
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  document: EditorDocumentRecord
  editor: Editor | null
  isCommentPanelVisible: boolean
  isDiffVisible: boolean
  isDirty: boolean
  isTreeVisible: boolean
  onCommentPanelVisibleChange?: (visible: boolean) => void
  onTreeVisibleChange?: (visible: boolean) => void
  viewMode: ProjectState['editorUi']['viewMode']
}) {
  const [isEditingTitle, setIsEditingTitle] = useState(false)
  const [titleDraft, setTitleDraft] = useState(document.title)
  const titleInputRef = useRef<HTMLInputElement | null>(null)
  const [isExporting, setIsExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)

  useLayoutEffect(() => {
    if (!isEditingTitle) return
    titleInputRef.current?.focus()
    titleInputRef.current?.select()
  }, [isEditingTitle])

  useEffect(() => {
    if (isEditingTitle) return
    setTitleDraft(document.title)
  }, [document.title, isEditingTitle])

  function commitTitleEdit() {
    const title = titleDraft.trim()
    if (title) {
      dispatch({ documentId: document.id, title, type: 'renameEditorDocument' })
    } else {
      setTitleDraft(document.title)
    }
    setIsEditingTitle(false)
  }

  async function handleExportWord() {
    if (isExporting) return
    setExportError(null)
    setIsExporting(true)
    try {
      const { exportMarkdownToDocx } = await import('./export/docxExport')
      await exportMarkdownToDocx(document.contentMarkdown, document.title)
    } catch (error) {
      setExportError(copy.exportWordFailed)
      console.error('Inqtrix Word export failed.', error)
    } finally {
      setIsExporting(false)
    }
  }

  const provenanceLabel = document.sourceRunId
    ? document.source === 'agent-artifact'
      ? copy.importedFromAgent
      : document.source === 'imported-research-report'
        ? copy.importedFrom
        : null
    : null
  const documentStatusText = provenanceLabel
    ? `${provenanceLabel} ${document.sourceRunId} · ${copy.updated} ${formatEditorTime(document.updatedAt)}`
    : `${copy.updated} ${formatEditorTime(document.updatedAt)}`

  return (
    <header className="grid inqtrix-panel-header grid-cols-[minmax(12rem,1fr)_auto_minmax(12rem,1fr)] items-center gap-2 border-b border-border bg-background px-3">
      <div className="flex min-w-0 items-center gap-2">
        <EditorPanelToggle
          copy={copy}
          dispatch={dispatch}
          onToggle={onTreeVisibleChange}
          side="left"
          visible={isTreeVisible}
        />
        <div className="min-w-0" title={documentStatusText}>
          <div className="flex min-w-0 items-center gap-1.5">
            {isEditingTitle ? (
              <input
                aria-label={copy.renameDocument}
                className="t-section min-w-0 flex-1 rounded-sm border-0 bg-transparent px-0 text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
                onBlur={commitTitleEdit}
                onChange={(event) => setTitleDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault()
                    commitTitleEdit()
                  }
                  if (event.key === 'Escape') {
                    event.preventDefault()
                    setTitleDraft(document.title)
                    setIsEditingTitle(false)
                  }
                }}
                ref={titleInputRef}
                value={titleDraft}
              />
            ) : (
              <button
                className="t-section min-w-0 flex-1 truncate rounded-sm text-left hover:text-brand focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                onDoubleClick={() => setIsEditingTitle(true)}
                title={`${copy.renameDocument} · ${documentStatusText}`}
                type="button"
              >
                {document.title}
              </button>
            )}
            {isDirty ? <span className="size-1.5 shrink-0 rounded-full bg-brand" aria-label="Unsaved changes" /> : null}
          </div>
        </div>
      </div>
      <EditorCommandToolbar editor={editor} isSource={viewMode === 'source'} />
      <div className="flex min-w-0 justify-end gap-0.5">
        <Badge className="h-5 rounded-full px-1.5 t-hint" variant="outline">{commentCount}</Badge>
        <Badge className="h-5 rounded-full px-1.5 t-hint" variant="outline">R{document.revision}</Badge>
        <Separator className="mx-0.5 h-5" orientation="vertical" />
        <TooltipButton
          label={viewMode === 'source' ? copy.live : copy.source}
          onClick={() => dispatch({ mode: viewMode === 'source' ? 'live' : 'source', type: 'setEditorViewMode' })}
        >
          {viewMode === 'source' ? <Eye className="size-4" /> : <Code2 className="size-4" />}
        </TooltipButton>
        <TooltipButton
          className={document.diffAnchorMarkdown ? 'bg-brand-subtle text-brand hover:bg-brand-subtle/80 hover:text-brand' : undefined}
          label={copy.setDiffAnchor}
          onClick={() => dispatch({ documentId: document.id, type: 'setEditorDiffAnchor' })}
        >
          <Anchor className="size-4" />
        </TooltipButton>
        <TooltipButton
          className={isDiffVisible ? 'bg-brand text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground' : undefined}
          label={isDiffVisible ? copy.hideDiff : copy.showDiff}
          onClick={() => dispatch({ isVisible: !isDiffVisible, type: 'setEditorDiffVisible' })}
        >
          <Scale className="size-4" />
        </TooltipButton>
        <Separator className="mx-0.5 h-5" orientation="vertical" />
        <TooltipButton
          disabled={isExporting}
          label={exportError ?? copy.exportWord}
          onClick={() => { void handleExportWord() }}
        >
          {isExporting ? <LoaderCircle className="size-4 animate-spin" /> : <FileDown className="size-4" />}
        </TooltipButton>
        <Separator className="mx-0.5 h-5" orientation="vertical" />
        <TooltipButton
          label={copy.deleteDocument}
          onClick={() => dispatch({ documentId: document.id, type: 'deleteEditorDocument' })}
        >
          <Trash2 className="size-4" />
        </TooltipButton>
        <Separator className="mx-0.5 h-5" orientation="vertical" />
        <EditorPanelToggle
          copy={copy}
          dispatch={dispatch}
          onToggle={onCommentPanelVisibleChange}
          side="right"
          visible={isCommentPanelVisible}
        />
      </div>
    </header>
  )
}

function EditorCommandToolbar({
  editor,
  isSource,
}: {
  editor: Editor | null
  isSource: boolean
}) {
  const disabled = !editor || isSource

  return (
    <div className="flex min-w-0 items-center justify-center gap-0.5 overflow-x-auto px-1 [scrollbar-width:none]">
      <ToolbarButton disabled={disabled} icon={Undo2} label="Undo" onClick={() => editor?.chain().focus().undo().run()} />
      <ToolbarButton disabled={disabled} icon={Redo2} label="Redo" onClick={() => editor?.chain().focus().redo().run()} />
    </div>
  )
}

function EditorAssistantComposer({
  attachedCommentIds,
  attachmentChips,
  chatModelOptions,
  chatModelOptionsStatus,
  comments,
  composerRef,
  copy,
  defaultChatModel,
  dispatch,
  draft,
  fileGroupOptions,
  fileOptions,
  reportOptions,
  isAttachActive,
  instructionFeedback,
  isRunning,
  isVisible,
  isWideCanvas,
  onAttachFiles,
  onAttachRule,
  onRefsChange,
  onReorderPending,
  onReorderPill,
  onRemoveAttachedComment,
  onRemoveChip,
  pendingKeys,
  pillKeys,
  onSend,
  onDismissInstructionFeedback,
  onStop,
  onToggleAttach,
  ruleOptions,
  selectedModelTier,
  chatModelCatalog,
  selectedModel,
  selectedEffort,
  editorContextBase,
  editorContextCapacity,
  textImprovement,
}: {
  attachedCommentIds: string[]
  attachmentChips: ChatAttachmentChipModel[]
  chatModelOptions: ChatModelOption[]
  chatModelOptionsStatus: 'available' | 'missing' | 'unresolved'
  comments: EditorCommentThreadRecord[]
  composerRef: RefObject<MentionComposerHandle | null>
  copy: EditorCopy
  defaultChatModel: NodeModelResolution | null
  dispatch: Dispatch<ResearchDeskAction>
  draft: string
  fileGroupOptions: FileGroupMentionOption[]
  fileOptions: FileMentionOption[]
  reportOptions: CompletedReportOption[]
  isAttachActive: boolean
  instructionFeedback: EditorInstructionFeedback | null
  isRunning: boolean
  isVisible: boolean
  isWideCanvas: boolean
  onAttachFiles: (files: File[]) => void
  onAttachRule: (ruleId: string) => void
  onRefsChange: (refs: ChatContextReferenceRecord[]) => void
  onReorderPending: (fromIndex: number, toIndex: number) => void
  onReorderPill: (fromIndex: number, toIndex: number) => void
  onRemoveAttachedComment: (commentId: string) => void
  onRemoveChip: (ref: ChatContextReferenceRecord) => void
  pendingKeys: string[]
  pillKeys: string[]
  onSend: () => void
  onDismissInstructionFeedback: () => void
  onStop: () => void
  onToggleAttach: () => void
  ruleOptions: ChatRuleOption[]
  selectedModelTier: ChatModelTier | null
  chatModelCatalog?: ModelCatalogEntry[]
  selectedModel: string | null
  selectedEffort: string | null
  editorContextBase: { documents: number; reports: number; rules: number; conversation: number }
  editorContextCapacity: { contextWindowTokens: number | null; reservedOutputTokens: number }
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}) {
  const { locale, t } = useLocale()
  const reduceMotion = useReducedMotion()
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [improveError, setImproveError] = useState<string | null>(null)
  const deferredDraft = useDeferredValue(draft)
  const composerTokens = useMemo(() => estimateTokensFromText(deferredDraft), [deferredDraft])
  const contextTokenModel = buildContextTokenModel(
    [
      { key: 'documents', tone: 'file', tokens: editorContextBase.documents },
      { key: 'reports', tone: 'success', tokens: editorContextBase.reports },
      { key: 'rules', tone: 'success', tokens: editorContextBase.rules },
      { key: 'conversation', tone: 'warning', tokens: editorContextBase.conversation },
      { key: 'composer', tone: 'brand', tokens: composerTokens },
    ] satisfies ContextCategoryInput[],
    editorContextCapacity,
  )
  // Confirm-on-overflow guard, mirroring the chat composer: fire only on a real
  // estimated overflow (capacity already nets out reserved output + safety).
  const contextOverflow =
    contextTokenModel.usedFraction != null && contextTokenModel.usedFraction > 1
  const contextOverflowPct = Math.round((contextTokenModel.usedFraction ?? 0) * 100)
  const [overflowConfirmOpen, setOverflowConfirmOpen] = useState(false)
  const assistantTextImprove = useTextImprovement({
    ...textImprovement,
    locale,
    messages: {
      requestFailed: (message) => `${t.textImprove.requestFailed}: ${message}`,
      sensitiveText: t.textImprove.sensitiveText,
      unavailable: t.textImprove.unavailable,
    },
  })

  const attachedComments = comments.filter(
    (comment) => comment.status === 'open'
      && comment.kind === 'collect'
      && attachedCommentIds.includes(comment.id),
  )
  const canEditorSend = isRunning || attachedComments.length > 0 || draft.trim().length > 0
  const guardedSend = () => {
    if (!isRunning && contextOverflow && canEditorSend) {
      setOverflowConfirmOpen(true)
      return
    }
    onSend()
  }
  const commentNumbers = buildCommentNumbers(comments)
  const mentionCategoryLabels: MentionCategoryLabels = {
    files: t.chat.mentionFilesCategory,
    filegroups: t.chat.mentionFilegroupsCategory,
    research: t.chat.mentionResearchCategory,
    rules: t.chat.mentionRulesCategory,
  }
  const mentionSources: MentionSources = { fileGroupOptions, fileOptions, reportOptions, ruleOptions }
  const resolveMentionLabel: LabelResolver = (kind, label) => {
    if (kind === 'file-asset') {
      const option = fileOptions.find((file) => file.label === label)
      return option ? { id: option.fileId, label: option.label } : null
    }
    if (kind === 'file-group') {
      const option = fileGroupOptions.find((group) => group.label === label)
      return option ? { id: option.groupId, label: option.label } : null
    }
    const option = reportOptions.find((report) => report.label === label)
    return option ? { id: option.runId, label: option.label } : null
  }

  function handleComposerChange() {
    dispatch({ draft: composerRef.current?.getMentionText() ?? '', type: 'setEditorAssistantDraft' })
    setImproveError(null)
    assistantTextImprove.clearProposal()
  }

  async function improveDraft() {
    setImproveError(null)
    try {
      await assistantTextImprove.improve('chat_input', composerRef.current?.getMentionText() ?? draft)
    } catch (error) {
      setImproveError(messageFromUnknown(error))
    }
  }

  function acceptDraftImprovement(text: string) {
    composerRef.current?.setMentionText(text)
    dispatch({ draft: composerRef.current?.getMentionText() ?? text, type: 'setEditorAssistantDraft' })
    assistantTextImprove.clearProposal()
    setImproveError(null)
    window.requestAnimationFrame(() => composerRef.current?.focus())
  }

  // Restore the persisted draft (with its `[N]` pills) once when the composer
  // mounts: the assistant draft survives hiding/showing the panel and switching
  // documents, mirroring the previous textarea behaviour.
  const didRestoreRef = useRef(false)
  useEffect(() => {
    if (didRestoreRef.current) return
    didRestoreRef.current = true
    if (draft) composerRef.current?.setMentionText(draft)
  }, [draft, composerRef])

  if (!isVisible) {
    return (
      <div className="shrink-0 px-4 pb-4 pt-2">
        <Button
          className="h-8 rounded-md"
          onClick={() => dispatch({ isVisible: true, type: 'setEditorAssistantVisible' })}
          size="sm"
          type="button"
          variant="outline"
        >
          <PanelBottomOpen className="size-4" />
          {copy.showAssistant}
        </Button>
      </div>
    )
  }

  return (
    <div className="relative z-10 shrink-0 px-4 pb-4 pt-2">
      {isWideCanvas ? (
        <>
          <div
            aria-hidden="true"
            className="pointer-events-none absolute bottom-0 left-4 top-2 hidden rounded-r-xl bg-background/30 backdrop-blur-xl xl:block xl:right-[calc(50%+28rem)]"
          />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute bottom-0 right-4 top-2 hidden rounded-l-xl bg-background/30 backdrop-blur-xl xl:left-[calc(50%+28rem)] xl:block"
          />
        </>
      ) : null}
      <div className="relative mx-auto max-w-4xl">
        <AnimatePresence initial={false}>
          {isAttachActive ? (
            <motion.div
              animate={{ height: 'auto', opacity: 1 }}
              className="mb-2 overflow-hidden rounded-md border border-brand/30 bg-brand-subtle/30"
              exit={{ height: 0, opacity: 0 }}
              initial={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              <div className="flex items-center gap-1.5 px-3 pb-1 pt-2 t-meta-sm font-semibold text-brand">
                <MessagesSquare className="size-3.5" />
                {attachedComments.length} {copy.attachedComments}
              </div>
              <div className="flex flex-wrap gap-1.5 px-3 pb-2.5">
                <AnimatePresence initial={false}>
                  {attachedComments.map((comment) => (
                    <motion.span
                      animate={{ opacity: 1, scale: 1 }}
                      className="inline-flex max-w-full items-center gap-1 rounded-full border border-brand/30 bg-background px-2 py-0.5 t-meta-sm text-foreground"
                      exit={{ opacity: 0, scale: 0.85 }}
                      initial={{ opacity: 0, scale: 0.85 }}
                      key={comment.id}
                      layout
                      transition={{ duration: 0.15 }}
                    >
                      <span className="grid size-4 shrink-0 place-items-center rounded-[4px] bg-brand-subtle t-hint font-semibold tabular-nums text-brand">
                        {commentNumbers.get(comment.id) ?? 0}
                      </span>
                      <span className="max-w-40 truncate text-muted-foreground">“{comment.anchor.selectedText}”</span>
                      <button
                        aria-label={copy.removeFromQueue}
                        className="text-muted-foreground hover:text-destructive"
                        onClick={() => onRemoveAttachedComment(comment.id)}
                        type="button"
                      >
                        <X className="size-3" />
                      </button>
                    </motion.span>
                  ))}
                </AnimatePresence>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
        <EditorInstructionFeedbackCard
          feedback={instructionFeedback}
          labels={{
            assistantDone: copy.assistantDone,
            assistantThinking: copy.assistantThinking,
            dismiss: copy.reject,
            hide: copy.hideAssistant,
            show: copy.showAssistant,
          }}
          onDismiss={onDismissInstructionFeedback}
        />
        <input
          className="hidden"
          multiple
          onChange={(event) => {
            const files = Array.from(event.target.files ?? [])
            if (files.length > 0) onAttachFiles(files)
            event.target.value = ''
          }}
          ref={fileInputRef}
          type="file"
        />
        <Dropzone disabled={isRunning} label={t.chat.dropFiles} onFiles={onAttachFiles}>
        <div className="relative rounded-xl border border-border bg-card px-3 py-2 shadow-[0_8px_28px_-12px_var(--shadow-soft)] transition-[border-color,box-shadow] duration-150 focus-within:border-brand/60 focus-within:ring-2 focus-within:ring-brand/15">
          <TextImproveFloatingLayer
            labels={{
              accept: t.textImprove.accept,
              changes: t.textImprove.changes,
              noChanges: t.textImprove.noChanges,
              reject: t.textImprove.reject,
              title: t.textImprove.title,
              warnings: t.textImprove.warnings,
            }}
            onAccept={acceptDraftImprovement}
            onReject={assistantTextImprove.clearProposal}
            proposal={assistantTextImprove.proposal}
            reduceMotion={reduceMotion}
          />
          <ContextChipLegend
            chips={attachmentChips}
            labels={{
              removeContext: copy.removeFromQueue,
              reorderHint: t.chat.reorderContextHint,
            }}
            onRemove={onRemoveChip}
            onReorderPending={onReorderPending}
            onReorderPill={onReorderPill}
            pendingKeys={pendingKeys}
            pillKeys={pillKeys}
          />
          <MentionComposer
            ariaLabel={copy.assistantPlaceholder}
            categoryLabels={mentionCategoryLabels}
            contentClassName="min-h-16 pb-2 pl-2 pr-9 pt-2 text-sm leading-6"
            enabledKinds={['research', 'rules', 'files', 'filegroups']}
            maxRows={6}
            mentionSources={mentionSources}
            onAttachRule={onAttachRule}
            onChange={handleComposerChange}
            onRefsChange={onRefsChange}
            onSubmit={guardedSend}
            placeholder={copy.assistantPlaceholder}
            ref={composerRef}
            resolveLabel={resolveMentionLabel}
          />
          <TextImproveButton
            className="absolute right-3 top-2"
            disabled={!draft.trim() || isRunning}
            isLoading={assistantTextImprove.isImproving}
            label={t.textImprove.improve}
            loadingLabel={t.textImprove.improving}
            onClick={() => void improveDraft()}
            reduceMotion={reduceMotion}
          />
          {improveError ? (
            <p className="t-meta-sm mb-1 rounded-md border border-destructive/20 bg-destructive/5 px-2 py-1 text-destructive">
              {improveError}
            </p>
          ) : null}
          <div className="flex items-center justify-between gap-2 border-t border-border/70 pt-1.5">
            <div className="flex min-w-0 items-center gap-1">
              <ComposerIconButton
                icon={PanelBottomClose}
                label={copy.hideAssistant}
                onClick={() => dispatch({ isVisible: false, type: 'setEditorAssistantVisible' })}
              />
              <ComposerIconButton
                active={isAttachActive}
                icon={MessageSquareText}
                label={copy.attachComments}
                onClick={onToggleAttach}
              />
              <ComposerIconButton
                icon={Paperclip}
                label={t.chat.attachFiles}
                onClick={() => fileInputRef.current?.click()}
              />
              <ModelTierPicker
                defaultModel={defaultChatModel}
                disabled={false}
                onChange={(tier) => dispatch({ tier, type: 'setSelectedChatModelTier' })}
                options={chatModelOptions}
                optionsStatus={chatModelOptionsStatus}
                selectedTier={selectedModelTier}
                modelCatalog={chatModelCatalog}
                selectedModel={selectedModel}
                selectedEffort={selectedEffort}
                onModelChange={(model) => dispatch({ model, type: 'setSelectedChatModel' })}
                onEffortChange={(effort) => dispatch({ effort, type: 'setSelectedChatEffort' })}
              />
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <QuotaMeter />
              <ContextTokenMeter
                conversationLabel={t.chat.contextCatDocument}
                disabled={false}
                model={contextTokenModel}
              />
            {!isRunning && contextOverflow && canEditorSend ? (
              <DropdownMenu onOpenChange={setOverflowConfirmOpen} open={overflowConfirmOpen}>
                <DropdownMenuTrigger asChild>
                  <Button
                    aria-label={copy.send}
                    className="size-7 rounded-md bg-brand text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground"
                    size="icon"
                    type="button"
                    variant="default"
                  >
                    <SendHorizontal className="size-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-64 rounded-xl p-0 shadow-lg" side="top" sideOffset={8}>
                  <div className="px-2.5 py-2">
                    <p className="flex items-center gap-1.5 t-meta-sm font-medium text-warning">
                      <AlertTriangle className="size-3.5 shrink-0" />
                      {t.chat.contextOverflowConfirmTitle(contextOverflowPct)}
                    </p>
                    <p className="mt-1 t-hint text-muted-foreground/80">
                      {t.chat.contextOverflowConfirmBody}
                    </p>
                  </div>
                  <div className="flex items-center justify-end gap-1.5 border-t border-border px-2 py-1.5">
                    <Button onClick={() => setOverflowConfirmOpen(false)} size="sm" type="button" variant="ghost">
                      {t.chat.contextOverflowCancel}
                    </Button>
                    <Button
                      className="bg-brand text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground"
                      onClick={() => {
                        setOverflowConfirmOpen(false)
                        onSend()
                      }}
                      size="sm"
                      type="button"
                      variant="default"
                    >
                      {t.chat.contextOverflowConfirmSend}
                    </Button>
                  </div>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              isRunning ? (
                <ComposerStopButton label={copy.stopRun} onClick={onStop} />
              ) : (
                <Button
                  aria-label={copy.send}
                  className={cn(
                    'size-7 rounded-md',
                    attachedComments.length === 0 && draft.trim().length === 0
                      ? 'text-muted-foreground/45'
                      : 'bg-brand text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground',
                  )}
                  disabled={attachedComments.length === 0 && draft.trim().length === 0}
                  onClick={onSend}
                  size="icon"
                  type="button"
                  variant={attachedComments.length === 0 && draft.trim().length === 0 ? 'ghost' : 'default'}
                >
                  <SendHorizontal className="size-4" />
                </Button>
              )
            )}
            </div>
          </div>
        </div>
        </Dropzone>
      </div>
    </div>
  )
}

const EVIDENCE_PRESET_ORDER: EditorEvidencePreset[] = ['add_sources', 'fact_check', 'verify_citations']

function evidencePresetLabel(preset: EditorEvidencePreset, copy: EditorCopy) {
  if (preset === 'fact_check') return copy.presetFactCheck
  if (preset === 'verify_citations') return copy.presetVerifyCitations
  return copy.presetAddSources
}

function EditorCommentsPanel({
  comments,
  copy,
  dispatch,
  onClose,
  onAcceptSuggestionGroup,
  onAcceptSuggestion,
  onRejectSuggestionGroup,
  onRejectSuggestion,
  onRunComment,
  onSelectSuggestion,
  runErrors,
  runningCommentIds,
  selectedCommentId,
  suggestions,
}: {
  comments: EditorCommentThreadRecord[]
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  onClose: () => void
  onAcceptSuggestionGroup: (groupId: string) => void
  onAcceptSuggestion: (suggestion: EditorSuggestionRecord) => void
  onRejectSuggestionGroup: (groupId: string) => void
  onRejectSuggestion: (suggestionId: string) => void
  onRunComment: (comment: EditorCommentThreadRecord) => void
  onSelectSuggestion: (suggestionId: string) => void
  runErrors: Record<string, string>
  runningCommentIds: readonly string[]
  selectedCommentId: string | null
  suggestions: EditorSuggestionRecord[]
}) {
  const [statusTab, setStatusTab] = useState<'open' | 'resolved'>('open')
  const [kindFilter, setKindFilter] = useState<'all' | EditorCommentKind>('all')

  const openComments = comments.filter((comment) => comment.status !== 'resolved')
  const resolvedComments = comments.filter((comment) => comment.status === 'resolved')
  const tabComments = statusTab === 'open' ? openComments : resolvedComments
  const visibleComments = kindFilter === 'all'
    ? tabComments
    : tabComments.filter((comment) => comment.kind === kindFilter)
  const commentNumbers = buildCommentNumbers(comments)
  const documentChangeSuggestions = suggestions
    .filter((suggestion) =>
      (suggestion.status === 'pending' || suggestion.status === 'stale')
      && suggestion.origin.kind === 'global_run'
      && !suggestion.origin.commentId)
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt))

  return (
    <aside className="inqtrix-contained-panel flex h-full w-full min-w-0 flex-col bg-background">
      <div className="flex inqtrix-panel-header items-center justify-between border-b border-border px-3">
        <div className="flex items-center gap-2">
          <h2 className="t-section">{copy.assistant}</h2>
        </div>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              aria-label={copy.hideAssistant}
              className="size-7 shrink-0 text-muted-foreground hover:text-foreground"
              onClick={onClose}
              size="icon"
              type="button"
              variant="ghost"
            >
              <PanelRightClose className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">{copy.hideAssistant}</TooltipContent>
        </Tooltip>
      </div>
      <div className="flex flex-col gap-2 border-b border-border px-3 py-2">
        <div className="grid grid-cols-2 gap-0.5 rounded-md bg-muted/60 p-0.5">
          <CommentStatusTab active={statusTab === 'open'} count={openComments.length} label={copy.tabOpen} onClick={() => setStatusTab('open')} />
          <CommentStatusTab active={statusTab === 'resolved'} count={resolvedComments.length} label={copy.tabResolved} onClick={() => setStatusTab('resolved')} />
        </div>
        <div className="flex items-center gap-1 overflow-x-auto pb-0.5 [scrollbar-width:none]">
          <ListFilter className="size-3.5 shrink-0 text-muted-foreground" />
          <CommentKindChip active={kindFilter === 'all'} label={copy.filterAll} onClick={() => setKindFilter('all')} />
          {COMMENT_KIND_ORDER.map((kind) => {
            const meta = commentKindMeta(kind, copy)
            return (
              <CommentKindChip
                active={kindFilter === kind}
                dotClass={meta.dotClass}
                key={kind}
                label={meta.label}
                onClick={() => setKindFilter(kind)}
              />
            )
          })}
        </div>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-2 p-3">
          <EditorDocumentChangesSection
            labels={{
              accept: copy.accept,
              acceptAll: copy.acceptAll,
              documentChanges: copy.documentChanges,
              proposedChange: copy.proposedChange,
              reject: copy.reject,
              rejectAll: copy.rejectAll,
            }}
            onAcceptGroup={onAcceptSuggestionGroup}
            onAcceptSuggestion={onAcceptSuggestion}
            onRejectGroup={onRejectSuggestionGroup}
            onRejectSuggestion={onRejectSuggestion}
            onSelectSuggestion={onSelectSuggestion}
            suggestions={documentChangeSuggestions}
          />
          {visibleComments.length === 0 ? (
            <p className="rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground">{copy.noComments}</p>
          ) : visibleComments.map((comment) => (
            <EditorCommentCard
              comment={comment}
              commentNumber={commentNumbers.get(comment.id) ?? 0}
              copy={copy}
              dispatch={dispatch}
              isRunning={runningCommentIds.includes(comment.id)}
              isSelected={selectedCommentId === comment.id}
              key={comment.id}
              onAcceptSuggestion={onAcceptSuggestion}
              onRejectSuggestion={onRejectSuggestion}
              onRunComment={onRunComment}
              runError={runErrors[comment.id]}
              suggestion={activeSuggestionFor(suggestions, comment.id)}
            />
          ))}
        </div>
      </ScrollArea>
    </aside>
  )
}

function CommentStatusTab({ active, count, label, onClick }: {
  active: boolean
  count: number
  label: string
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'flex h-7 items-center justify-center gap-1.5 rounded-[5px] text-xs font-medium transition-colors',
        active ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
      )}
      onClick={onClick}
      type="button"
    >
      {label}
      <span className="t-hint tabular-nums text-muted-foreground/80">{count}</span>
    </button>
  )
}

function CommentKindChip({ active, dotClass, label, onClick }: {
  active: boolean
  dotClass?: string
  label: string
  onClick: () => void
}) {
  return (
    <Chip active={active} dot={dotClass} onClick={onClick}>
      {label}
    </Chip>
  )
}

function EditorCommentCard({
  comment,
  commentNumber,
  copy,
  dispatch,
  isRunning,
  isSelected,
  onAcceptSuggestion,
  onRejectSuggestion,
  onRunComment,
  runError,
  suggestion,
}: {
  comment: EditorCommentThreadRecord
  commentNumber: number
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  isRunning: boolean
  isSelected: boolean
  onAcceptSuggestion: (suggestion: EditorSuggestionRecord) => void
  onRejectSuggestion: (suggestionId: string) => void
  onRunComment: (comment: EditorCommentThreadRecord) => void
  runError?: string
  suggestion?: EditorSuggestionRecord
}) {
  const meta = commentKindMeta(comment.kind, copy)
  const isResolved = comment.status === 'resolved'
  const isRunnable = comment.kind === 'inline_edit' || comment.kind === 'evidence_review'
  const [isEditing, setIsEditing] = useState(false)
  const [editDraft, setEditDraft] = useState(comment.commentMarkdown)
  const editTextareaRef = useRef<HTMLTextAreaElement | null>(null)

  useLayoutEffect(() => {
    if (!isEditing) return
    resizeTextareaToRows(editTextareaRef.current, 6)
  }, [editDraft, isEditing])

  function startEditing() {
    setEditDraft(comment.commentMarkdown)
    setIsEditing(true)
    dispatch({ commentId: comment.id, type: 'selectEditorComment' })
  }

  function saveEdit() {
    dispatch({ commentId: comment.id, contentMarkdown: editDraft, type: 'updateEditorCommentText' })
    setIsEditing(false)
  }

  return (
    <div
      className={cn(
        'rounded-md border border-l-[3px] text-sm shadow-sm transition-colors',
        meta.borderClass,
        isSelected
          ? cn(meta.selectedBorderClass, meta.selectedBgClass)
          : cn('border-border', meta.bgClass, 'hover:brightness-110'),
        isResolved && 'opacity-65',
      )}
      onClick={() => dispatch({ commentId: comment.id, type: 'selectEditorComment' })}
    >
      <div className="flex items-center gap-1.5 px-3 pt-2.5">
        <span className={cn('grid size-5 shrink-0 place-items-center rounded-[5px] bg-background/70 t-hint font-semibold tabular-nums', meta.accentText)}>
          {commentNumber}
        </span>
        <CommentKindMenu comment={comment} copy={copy} dispatch={dispatch} meta={meta} />
        <Button
          aria-label={copy.deleteComment}
          className="ml-auto size-6 shrink-0 text-muted-foreground hover:text-destructive"
          onClick={(event) => {
            event.stopPropagation()
            dispatch({ commentId: comment.id, type: 'deleteEditorComment' })
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Trash2 className="size-3.5" />
        </Button>
        <Button
          aria-label={copy.editComment}
          className="size-6 shrink-0 text-muted-foreground hover:text-foreground"
          onClick={(event) => {
            event.stopPropagation()
            startEditing()
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          <PencilLine className="size-3.5" />
        </Button>
        <Button
          aria-label={isResolved ? copy.reopen : copy.resolve}
          className="size-6 shrink-0 text-muted-foreground hover:text-foreground"
          onClick={(event) => {
            event.stopPropagation()
            dispatch({ commentId: comment.id, status: isResolved ? 'open' : 'resolved', type: 'setEditorCommentStatus' })
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          {isResolved ? <Undo2 className="size-3.5" /> : <Check className="size-3.5" />}
        </Button>
        <span className="t-hint shrink-0 text-muted-foreground">{formatEditorTime(comment.updatedAt)}</span>
      </div>
      <div className="px-3 pb-2.5 pt-1">
        <p className={cn('text-muted-foreground', isSelected ? 'line-clamp-2' : 'line-clamp-1')}>
          <span className="text-muted-foreground/60">“</span>
          {compactCommentQuote(comment.anchor.selectedText, isSelected ? 180 : 96)}
          <span className="text-muted-foreground/60">”</span>
        </p>
        {isEditing ? (
          <div className="mt-1.5" onClick={(event) => event.stopPropagation()}>
            <Textarea
              autoFocus
              className="t-body min-h-16 resize-none [scrollbar-width:thin]"
              onChange={(event) => setEditDraft(event.target.value)}
              ref={editTextareaRef}
              value={editDraft}
            />
            <div className="mt-1.5 flex items-center justify-end gap-1.5">
              <Button className="h-7" onClick={() => setIsEditing(false)} size="sm" type="button" variant="ghost">
                {copy.cancel}
              </Button>
              <Button className="h-7" disabled={!editDraft.trim()} onClick={saveEdit} size="sm" type="button">
                {copy.save}
              </Button>
            </div>
          </div>
        ) : (
          <p className={cn('mt-1 text-foreground', isSelected ? 'whitespace-pre-wrap' : 'line-clamp-1')}>
            {comment.commentMarkdown}
          </p>
        )}
        {isSelected && !isEditing && comment.kind === 'evidence_review' ? (
          <EvidencePresetPicker comment={comment} copy={copy} dispatch={dispatch} />
        ) : null}
        {isSelected && !isEditing ? (
          isRunning ? (
            <div className="mt-2.5 flex items-center justify-center gap-2 rounded-md border border-brand/25 bg-brand-subtle/30 py-2 t-meta-sm font-medium text-brand">
              <LoaderCircle className="size-3.5 animate-spin" />
              {copy.runningSuggestion}
            </div>
          ) : suggestion && suggestion.status === 'pending' ? (
            <SuggestionReview
              copy={copy}
              onAccept={onAcceptSuggestion}
              onReject={onRejectSuggestion}
              suggestion={suggestion}
            />
          ) : suggestion && suggestion.status === 'stale' ? (
            <div
              className="mt-2.5 rounded-md border border-warning/30 bg-warning-subtle/40 p-2.5"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="t-meta-sm flex items-center gap-1.5 text-warning">
                <AlertTriangle className="size-3.5 shrink-0" />
                {copy.suggestionStale}
              </div>
              <div className="mt-2 flex items-center justify-end gap-1.5">
                <Button className="h-7" onClick={() => onRejectSuggestion(suggestion.id)} size="sm" type="button" variant="ghost">
                  <X className="size-3.5" />
                  {copy.reject}
                </Button>
                {isRunnable ? (
                  <Button className="h-7" onClick={() => onRunComment(comment)} size="sm" type="button" variant="outline">
                    <Sparkles className="size-3.5" />
                    {copy.regenerate}
                  </Button>
                ) : null}
              </div>
            </div>
          ) : (
            <>
              {runError ? (
                <div
                  className="t-meta-sm mt-2.5 rounded-md border border-destructive/30 bg-destructive-subtle/40 p-2 text-destructive"
                  onClick={(event) => event.stopPropagation()}
                >
                  {runError}
                </div>
              ) : null}
              {isRunnable ? (
                <Button
                  className="mt-2.5 h-7 w-full"
                  onClick={(event) => {
                    event.stopPropagation()
                    onRunComment(comment)
                  }}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <Sparkles className="size-3.5" />
                  {copy.runSuggestion}
                </Button>
              ) : null}
            </>
          )
        ) : null}
      </div>
    </div>
  )
}

function buildCommentNumbers(comments: EditorCommentThreadRecord[]): Map<string, number> {
  const ordered = [...comments].sort((a, b) =>
    a.anchor.from - b.anchor.from || a.createdAt.localeCompare(b.createdAt))
  return new Map(ordered.map((comment, index) => [comment.id, index + 1]))
}

function compactCommentQuote(value: string, maxLength: number): string {
  const text = value.replace(/\s+/g, ' ').trim()
  if (text.length <= maxLength) return text
  return `${text.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`
}

function activeSuggestionFor(
  suggestions: EditorSuggestionRecord[],
  commentId: string,
): EditorSuggestionRecord | undefined {
  return suggestions
    .filter((suggestion) =>
      (suggestion.status === 'pending' || suggestion.status === 'stale')
      && suggestion.origin.commentId === commentId)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0]
}

function SuggestionReview({ copy, onAccept, onReject, suggestion }: {
  copy: EditorCopy
  onAccept: (suggestion: EditorSuggestionRecord) => void
  onReject: (suggestionId: string) => void
  suggestion: EditorSuggestionRecord
}) {
  const plan = suggestionDiffPlan(suggestion.originalText, suggestion.proposedText)
  const reviewInEditor = plan.reviewSurface === 'editor'
  return (
    <div
      className="mt-2.5 rounded-md border border-brand/25 bg-brand-subtle/30 p-2.5"
      onClick={(event) => event.stopPropagation()}
    >
      <div className="flex items-center gap-1.5 t-meta-sm font-medium text-brand">
        <Sparkles className="size-3.5 shrink-0" />
        {reviewInEditor ? copy.reviewInEditor : copy.reviewInPanel}
      </div>
      <div className="mt-2 rounded-md border border-success/20 bg-success-subtle/25 p-2">
        <div className="t-caption text-success">{copy.proposedText}</div>
        <p className="t-meta mt-1 line-clamp-4 whitespace-pre-wrap text-foreground">{suggestion.proposedText}</p>
      </div>
      {suggestion.changeSummary?.length ? (
        <div className="mt-2 border-t border-border/60 pt-1.5">
          <div className="t-caption text-muted-foreground">{copy.changeSummary}</div>
          <ul className="mt-1 space-y-0.5">
            {suggestion.changeSummary.map((item, index) => (
              <li className="t-meta text-muted-foreground" key={`${index}-${item}`}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {suggestion.warnings?.length ? (
        <div className="mt-2 rounded-md border border-warning/25 bg-warning-subtle/35 p-2">
          <div className="t-caption flex items-center gap-1.5 text-warning">
            <AlertTriangle className="size-3" />
            {copy.warnings}
          </div>
          <ul className="mt-1 space-y-0.5">
            {suggestion.warnings.map((item, index) => (
              <li className="t-meta text-warning" key={`${index}-${item}`}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {suggestion.evidence ? (
        <div className="mt-2 border-t border-border/60 pt-1.5">
          <div className="t-caption text-success">{copy.sources}</div>
          <ul className="mt-1 space-y-0.5">
            {suggestion.evidence.sources.map((source) => (
              <li className="t-meta truncate" key={source.url}>
                <a className="text-brand hover:underline" href={source.url} rel="noopener noreferrer" target="_blank">
                  {source.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {reviewInEditor ? null : (
        <div className="mt-2 flex items-center justify-end gap-1.5">
          <Button className="h-7" onClick={() => onReject(suggestion.id)} size="sm" type="button" variant="ghost">
            <X className="size-3.5" />
            {copy.reject}
          </Button>
          <Button
            className="h-7 bg-brand text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground"
            onClick={() => onAccept(suggestion)}
            size="sm"
            type="button"
          >
            <Check className="size-3.5" />
            {copy.accept}
          </Button>
        </div>
      )}
    </div>
  )
}

function CommentKindMenu({ comment, copy, dispatch, meta }: {
  comment: EditorCommentThreadRecord
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  meta: CommentKindMeta
}) {
  const KindIcon = meta.Icon
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          aria-label={copy.changeKind}
          className={cn('inline-flex h-6 shrink-0 items-center gap-1 rounded-full bg-background px-1.5 t-meta-sm font-semibold', meta.accentText)}
          onClick={(event) => event.stopPropagation()}
          type="button"
        >
          <KindIcon className="size-3.5" />
          {meta.label}
          <ChevronDown className="size-3 opacity-60" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-52" onClick={(event) => event.stopPropagation()}>
        <DropdownMenuLabel className="text-xs text-muted-foreground">{copy.changeKind}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup
          onValueChange={(value) => dispatch({ commentId: comment.id, kind: value as EditorCommentKind, type: 'setEditorCommentKind' })}
          value={comment.kind}
        >
          {COMMENT_KIND_ORDER.map((kind) => {
            const kindMeta = commentKindMeta(kind, copy)
            const ItemIcon = kindMeta.Icon
            return (
              <DropdownMenuRadioItem className="gap-2 text-sm" key={kind} value={kind}>
                <ItemIcon className={cn('size-3.5', kindMeta.accentText)} />
                {kindMeta.label}
              </DropdownMenuRadioItem>
            )
          })}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function EvidencePresetPicker({ comment, copy, dispatch }: {
  comment: EditorCommentThreadRecord
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
}) {
  const activePreset = comment.evidencePreset ?? 'add_sources'
  return (
    <div className="mt-2.5 flex flex-col gap-1 rounded-md border border-success/25 bg-success-subtle/40 p-2">
      <span className="t-caption text-success">{copy.preset}</span>
      <div className="flex flex-wrap gap-1">
        {EVIDENCE_PRESET_ORDER.map((preset) => (
          <button
            className={cn(
              'inline-flex h-6 items-center rounded-full border px-2 t-meta-sm font-medium transition-colors',
              activePreset === preset ? 'border-success/50 bg-success-subtle text-success' : 'border-border bg-background text-muted-foreground hover:text-foreground',
            )}
            key={preset}
            onClick={(event) => {
              event.stopPropagation()
              dispatch({ commentId: comment.id, preset, type: 'setEditorCommentEvidencePreset' })
            }}
            type="button"
          >
            {evidencePresetLabel(preset, copy)}
          </button>
        ))}
      </div>
    </div>
  )
}

function EditorEmptyState({
  copy,
  dispatch,
  reportOptions,
}: {
  copy: EditorCopy
  dispatch: Dispatch<ResearchDeskAction>
  reportOptions: CompletedReportOption[]
}) {
  return (
    <div className="flex min-h-0 w-full flex-1 items-center justify-center bg-background px-6 py-8">
      <WelcomeState
        actions={(
          <div className="flex flex-wrap justify-center gap-2">
            <Button
              className="h-8 gap-1.5 rounded-md bg-brand px-3 text-xs text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground"
              onClick={() => dispatch({ type: 'createEditorDocument' })}
              type="button"
              variant="default"
            >
              <SquarePen className="icon-sm" />
              {copy.createDocument}
            </Button>
            <ImportReportMenu
              copy={copy}
              dispatch={dispatch}
              reportOptions={reportOptions}
              triggerClassName="h-8 gap-1.5 rounded-md px-3 text-xs"
              variant="button"
            />
          </div>
        )}
        body={<p>{copy.emptyGuidance}</p>}
        example={copy.emptyExample}
        kicker={copy.emptyKicker}
        subtitle={copy.emptyBody}
        title={copy.emptyTitle}
      />
    </div>
  )
}

function ToolbarButton({
  active,
  disabled,
  icon: Icon,
  label,
  onClick,
}: {
  active?: boolean
  disabled?: boolean
  icon: typeof Bold
  label: string
  onClick: () => void
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          aria-pressed={active}
          className={cn('size-6 rounded-md', active && 'bg-brand-subtle text-brand')}
          disabled={disabled}
          onClick={onClick}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Icon className="size-3.5" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function TooltipButton({
  children,
  className,
  disabled = false,
  label,
  onClick,
}: {
  children: ReactNode
  className?: string
  disabled?: boolean
  label: string
  onClick: () => void
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          className={cn('size-7 rounded-md', className)}
          disabled={disabled}
          onClick={onClick}
          size="icon"
          type="button"
          variant="ghost"
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function formatEditorTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function messageFromUnknown(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}
