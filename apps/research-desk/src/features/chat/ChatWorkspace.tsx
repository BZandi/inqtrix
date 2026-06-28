import {
  AlertTriangle,
  BookOpen,
  Bot,
  BrainCircuit,
  Check,
  ChevronDown,
  ChevronRight,
  CircleUserRound,
  Copy,
  Database,
  Eraser,
  EyeOff,
  FileText,
  GitBranchPlus,
  Library,
  ListChecks,
  ListMinus,
  ListOrdered,
  ListPlus,
  MoreHorizontal,
  MessageSquareText,
  MessageSquarePlus,
  Paperclip,
  PencilLine,
  Plus,
  RefreshCw,
  Save,
  SendHorizontal,
  SlidersHorizontal,
  Trash2,
  type LucideIcon,
  X,
} from '@/components/icons'
import { AnimatePresence, motion } from 'motion/react'
import {
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
} from 'react'
import { MarkdownRenderer } from '@/components/markdown/MarkdownRenderer'
import { MarkdownSelectionCopyMenu } from '@/components/markdown/MarkdownSelectionCopyMenu'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { WelcomeState } from '@/components/ui/welcome-state'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  assistantMenuContentClassName,
  AssistantMenuHeader,
  AssistantMenuIcon,
  AssistantMenuLabel,
} from '@/components/ui/assistant-menu'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  chatAttachmentChipsFromAttachments,
  chatContextRefKey,
  type ChatAttachmentChipModel,
  type ChatRuleOption,
  type ChatHistorySection,
  type CompletedReportOption,
  type FileGroupMentionOption,
  type FileMentionOption,
} from '@/features/project/selectors'
import { attachmentChipVisual } from '@/features/files/attachmentChips'
import { Dropzone } from '@/features/files/Dropzone'
import type {
  ChatChainStepRecord,
  ChatContextReferenceRecord,
  ChatMessageAttachmentRecord,
} from '@/features/project/types'
import type {
  ChatModelOption,
  ModelCatalogEntry,
  ChatModelTier,
  NodeModelResolution,
} from '@/features/researchRuns/types'
import { ModelTierPicker } from '@/features/researchRuns/ModelTierPicker'
import { ContextTokenMeter } from '@/features/composer/ContextTokenMeter'
import { QuotaMeter } from '@/features/quota/QuotaMeter'
import {
  buildContextTokenModel,
  estimateTokensFromText,
  type ContextCategoryInput,
} from '@/features/files/contextTokens'
import {
  modelEffortLabelFromToken,
  modelNameLabel,
  modelTierLabel,
} from '@/features/researchRuns/modelLabels'
import { useLocale } from '@/i18n/LocaleProvider'
import { formatMessageTimestamp } from '@/lib/time'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'
import {
  TextImproveButton,
  TextImproveFloatingLayer,
  useTextImprovement,
  type TextImprovementApiOptions,
} from '@/features/textImprove'
import { PanelToggle } from '@/components/ui/panel-toggle'
import { ComposerIconButton, composerIconButtonClassName } from '@/features/composer/ComposerIconButton'
import { ComposerStopButton } from '@/features/composer/ComposerStopButton'
import { ChatHistoryPanel } from './history/ChatHistoryPanel'
import type { ChatMessage, ChatThread } from './types'
import { ContextChipLegend } from '@/features/composer/ContextChipLegend'
import { MentionComposer, type MentionComposerHandle } from '@/features/composer/MentionComposer'
import { type LabelResolver } from '@/features/composer/mentionDoc'
import { resizeTextareaToRows } from '@/features/composer/textareaAutosize'
import { OptionMenuHeader, OptionMenuItem, optionMenuContentClassName } from '@/components/ui/option-menu'
import type { ChatRetryMode, ChatRetryOptions } from './retry'

type ChatWorkspaceProps = {
  activeAssistantMessageId: string | null
  chatModelOptions: ChatModelOption[]
  chatModelOptionsStatus: 'available' | 'missing' | 'unresolved'
  chatHistorySections: ChatHistorySection[]
  /** On-demand chat history: server has older thread pages not yet loaded. */
  chatHistoryHasMore?: boolean
  /** A load-older page request is in flight (busy state for the button). */
  chatHistoryLoadingMore?: boolean
  /** Load the next page of older threads. */
  onLoadMoreChatHistory?: () => void
  defaultChatModel: NodeModelResolution | null
  fileGroupOptions: FileGroupMentionOption[]
  fileOptions: FileMentionOption[]
  isDesktop: boolean
  isHistoryVisible: boolean
  isIncognito: boolean
  isSending: boolean
  onAttachContext: (ref: ChatContextReferenceRecord) => void
  onAttachFiles: (files: File[]) => void
  onPillRefsChange: (refs: ChatContextReferenceRecord[]) => void
  /** Session-scoped composer draft (text with serialized `[N]` pills) lifted to
   * the parent shell so it survives a workspace unmount on view switch. */
  chatDraft: string
  onChatDraftChange: (draft: string) => void
  onAnswerLastUserMessage: (threadId: string, messageId: string) => void
  onBranchFromMessage: (threadId: string, messageId: string) => void
  onClearThread: () => void
  onCreateThread: (groupId?: string | null) => void
  onCreateThreadGroup: () => void
  onDeleteMessages: (threadId: string, messageIds: string[]) => void
  onDeleteThreadGroup: (groupId: string) => void
  onDeleteThread: (threadId: string) => void
  onEditMessage: (threadId: string, messageId: string, contentMarkdown: string) => void
  onRetryAssistantMessage: (
    threadId: string,
    messageId: string,
    mode: ChatRetryMode,
    options?: ChatRetryOptions,
  ) => void
  chainingEnabled: boolean
  onChainingEnabledChange: (enabled: boolean) => void
  onIncognitoChange: (enabled: boolean) => void
  onHistoryVisibleChange: (isVisible: boolean) => void
  onOpenPromptLibrary: () => void
  onMoveThreadGroup: (groupId: string, targetIndex: number) => void
  onMoveThreadToGroup: (threadId: string, groupId: string | null, targetIndex: number) => void
  onRenameThread: (threadId: string, title: string) => void
  onRenameThreadGroup: (groupId: string, title: string) => void
  onRemoveContext: (ref: ChatContextReferenceRecord) => void
  onReorderContext: (fromIndex: number, toIndex: number) => void
  pendingReorderKeys: string[]
  pillKeys: string[]
  onSendMessage: (
    contentMarkdown: string,
    refs?: ChatContextReferenceRecord[],
    options?: ChatSendOptions,
  ) => void
  onSelectThread: (threadId: string) => void
  onTogglePinnedThread: (threadId: string) => void
  onSelectedModelTierChange: (tier: ChatModelTier | null) => void
  chatModelCatalog?: ModelCatalogEntry[]
  selectedChatModel: string | null
  selectedChatEffort: string | null
  onSelectedChatModelChange: (model: string | null) => void
  onSelectedChatEffortChange: (effort: string | null) => void
  chatContextBase: { documents: number; reports: number; rules: number; conversation: number }
  chatContextCapacity: { contextWindowTokens: number | null; reservedOutputTokens: number }
  /** `null` hides the knowledge scope picker (feature unavailable). */
  knowledgeIndexOptions: KnowledgeIndexOption[] | null
  selectedKnowledgeIndexIds: string[]
  onSelectedKnowledgeIndexIdsChange: (ids: string[]) => void
  onStopGenerating: () => void
  onStreamingEnabledChange: (enabled: boolean) => void
  attachmentBudgetNotice: string | null
  pendingChips: ChatAttachmentChipModel[]
  pinnedThreadIds: readonly string[]
  reduceMotion: boolean | null
  reportOptions: CompletedReportOption[]
  requestError: string | null
  requestNotice: string | null
  runningThreadIds: ReadonlySet<string>
  ruleOptions: ChatRuleOption[]
  selectedModelTier: ChatModelTier | null
  selectedThreadId: string | null
  streamingEnabled: boolean
  temporaryThread: ChatThread | null
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
  threads: ChatThread[]
}

type ChatSendOptions = {
  modelTier?: ChatModelTier
  model?: string | null
  effort?: string | null
  /** Backend knowledge-collection ids; non-empty switches the request
   * to `mode: 'knowledge'` (non-streaming, answers from documents). */
  knowledgeCollectionIds?: string[]
}

/** One ready, server-embedded vector index offered as a chat scope. */
export type KnowledgeIndexOption = {
  collectionId: string
  id: string
  title: string
}

const chatModelTierOrder: ChatModelTier[] = ['high', 'mid', 'fast']

export default function ChatWorkspace({
  activeAssistantMessageId,
  chatModelOptions,
  chatModelOptionsStatus,
  chatHistorySections,
  chatHistoryHasMore,
  chatHistoryLoadingMore,
  onLoadMoreChatHistory,
  defaultChatModel,
  isDesktop,
  isHistoryVisible,
  isIncognito,
  isSending,
  onAttachContext,
  onAnswerLastUserMessage,
  onBranchFromMessage,
  onClearThread,
  onCreateThread,
  onCreateThreadGroup,
  onDeleteMessages,
  onDeleteThreadGroup,
  onDeleteThread,
  onEditMessage,
  onRetryAssistantMessage,
  chainingEnabled,
  onChainingEnabledChange,
  onIncognitoChange,
  onHistoryVisibleChange,
  onOpenPromptLibrary,
  onMoveThreadGroup,
  onMoveThreadToGroup,
  onRenameThread,
  onRenameThreadGroup,
  onRemoveContext,
  onReorderContext,
  pendingReorderKeys,
  pillKeys,
  onSendMessage,
  onSelectThread,
  onTogglePinnedThread,
  onSelectedModelTierChange,
  chatModelCatalog = [],
  selectedChatModel,
  selectedChatEffort,
  onSelectedChatModelChange,
  onSelectedChatEffortChange,
  chatContextBase,
  chatContextCapacity,
  knowledgeIndexOptions,
  selectedKnowledgeIndexIds,
  onSelectedKnowledgeIndexIdsChange,
  onStopGenerating,
  onStreamingEnabledChange,
  onAttachFiles,
  onPillRefsChange,
  chatDraft,
  onChatDraftChange,
  fileGroupOptions,
  fileOptions,
  attachmentBudgetNotice,
  pendingChips,
  pinnedThreadIds,
  reduceMotion,
  reportOptions,
  requestError,
  requestNotice,
  runningThreadIds,
  ruleOptions,
  selectedModelTier,
  selectedThreadId,
  streamingEnabled,
  temporaryThread,
  textImprovement,
  threads,
}: ChatWorkspaceProps) {
  const { locale, t } = useLocale()
  const [composerNotice, setComposerNotice] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [draftCommitPulseKey, setDraftCommitPulseKey] = useState(0)
  const deferredDraft = useDeferredValue(draft)
  const composerTokens = useMemo(() => estimateTokensFromText(deferredDraft), [deferredDraft])
  const contextTokenModel = buildContextTokenModel(
    [
      { key: 'documents', tone: 'file', tokens: chatContextBase.documents },
      { key: 'reports', tone: 'success', tokens: chatContextBase.reports },
      { key: 'rules', tone: 'success', tokens: chatContextBase.rules },
      { key: 'conversation', tone: 'warning', tokens: chatContextBase.conversation },
      { key: 'composer', tone: 'brand', tokens: composerTokens },
    ] satisfies ContextCategoryInput[],
    chatContextCapacity,
  )
  // Send-guard only fires on a real estimated overflow (capacity already nets out
  // reserved output + safety); the estimate is ~96% accurate so we confirm rather
  // than hard-block.
  const contextOverflow =
    contextTokenModel.usedFraction != null && contextTokenModel.usedFraction > 1
  const contextOverflowPct = Math.round((contextTokenModel.usedFraction ?? 0) * 100)
  const [overflowConfirmOpen, setOverflowConfirmOpen] = useState(false)
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null)
  const [isEditingTitle, setIsEditingTitle] = useState(false)
  const [isMessageSelectionMode, setIsMessageSelectionMode] = useState(false)
  const [messageEditDraft, setMessageEditDraft] = useState('')
  const [pillRefs, setPillRefs] = useState<ChatContextReferenceRecord[]>([])
  const [selectedMessageIds, setSelectedMessageIds] = useState<ReadonlySet<string>>(() => new Set())
  const [titleDraft, setTitleDraft] = useState('')
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const composerRef = useRef<MentionComposerHandle | null>(null)
  const didRestoreDraftRef = useRef(false)
  const lastAutoFollowThreadIdRef = useRef<string | null>(null)
  const messageEditTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const messagesScrollAreaRef = useRef<HTMLDivElement | null>(null)
  const shouldAutoFollowChatRef = useRef(true)
  const titleInputRef = useRef<HTMLInputElement | null>(null)
  const chatFileInputRef = useRef<HTMLInputElement | null>(null)
  const selectedThread = isIncognito
    ? temporaryThread
    : threads.find((thread) => thread.id === selectedThreadId) ?? threads[0] ?? null
  const lastMessage = selectedThread?.messages[selectedThread.messages.length - 1]
  const canAnswerLastUserMessage = Boolean(
    selectedThread
    && lastMessage
    && lastMessage.role === 'user'
    && !isSending,
  )
  const canSend = draft.trim().length > 0 && !isSending
  const selectedMessageCount = selectedMessageIds.size
  const canManageMessages = Boolean(selectedThread && selectedThread.messages.length > 0 && !isSending)
  const mentionCategoryLabels = {
    files: t.chat.mentionFilesCategory,
    filegroups: t.chat.mentionFilegroupsCategory,
    research: t.chat.mentionResearchCategory,
    rules: t.chat.mentionRulesCategory,
  }
  const mentionSources = { fileGroupOptions, fileOptions, reportOptions, ruleOptions }
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
  const draftTextImprove = useTextImprovement({
    ...textImprovement,
    locale,
    messages: {
      requestFailed: (message) => `${t.textImprove.requestFailed}: ${message}`,
      sensitiveText: t.textImprove.sensitiveText,
      unavailable: t.textImprove.unavailable,
    },
  })

  useEffect(() => {
    setTitleDraft(selectedThread?.title ?? '')
    setIsEditingTitle(false)
    setEditingMessageId(null)
    setMessageEditDraft('')
    setIsMessageSelectionMode(false)
    setSelectedMessageIds(new Set())
    shouldAutoFollowChatRef.current = true
  }, [selectedThread?.id, selectedThread?.title])

  // Restore the session draft once on mount so text AND its `[N]` pills survive a
  // view switch (the workspace unmounts on switch). Setting the text unconditionally
  // -- even for an empty draft -- forces `onRefsChange` to re-derive the parent pill
  // mirror from the live editor, so no orphaned, unremovable chip can linger.
  useEffect(() => {
    if (didRestoreDraftRef.current) return
    didRestoreDraftRef.current = true
    composerRef.current?.setMentionText(chatDraft)
    setDraft(composerRef.current?.getMentionText() ?? chatDraft)
  }, [chatDraft])

  useEffect(() => {
    if (!selectedThread || selectedMessageIds.size === 0) return
    const currentMessageIds = new Set(selectedThread.messages.map((message) => message.id))
    const nextSelectedIds = new Set([...selectedMessageIds].filter((messageId) => currentMessageIds.has(messageId)))
    if (nextSelectedIds.size === selectedMessageIds.size) return
    setSelectedMessageIds(nextSelectedIds)
    if (nextSelectedIds.size === 0) {
      setIsMessageSelectionMode(false)
    }
  }, [selectedMessageIds, selectedThread])

  useEffect(() => {
    const viewport = messagesScrollAreaRef.current?.querySelector<HTMLElement>('[data-scroll-area-viewport]')
    if (!viewport) return undefined
    const scrollViewport = viewport

    function updateAutoFollow() {
      const distanceFromBottom = scrollViewport.scrollHeight - scrollViewport.scrollTop - scrollViewport.clientHeight
      shouldAutoFollowChatRef.current = distanceFromBottom < 96
    }

    updateAutoFollow()
    scrollViewport.addEventListener('scroll', updateAutoFollow, { passive: true })
    return () => scrollViewport.removeEventListener('scroll', updateAutoFollow)
  }, [selectedThread?.id])

  useLayoutEffect(() => {
    if (!isEditingTitle) return
    titleInputRef.current?.focus()
    titleInputRef.current?.select()
  }, [isEditingTitle])

  useLayoutEffect(() => {
    if (!editingMessageId) return
    messageEditTextareaRef.current?.focus()
    messageEditTextareaRef.current?.select()
    resizeTextareaToRows(messageEditTextareaRef.current, 8)
  }, [editingMessageId])

  useLayoutEffect(() => {
    const selectedId = selectedThread?.id ?? null
    if (lastAutoFollowThreadIdRef.current !== selectedId) {
      shouldAutoFollowChatRef.current = true
      lastAutoFollowThreadIdRef.current = selectedId
    }
    if (!shouldAutoFollowChatRef.current) return
    const behavior = reduceMotion || activeAssistantMessageId ? 'auto' : 'smooth'
    window.requestAnimationFrame(() => {
      chatEndRef.current?.scrollIntoView({ block: 'end', behavior })
    })
  }, [
    activeAssistantMessageId,
    lastMessage?.contentMarkdown,
    lastMessage?.id,
    selectedThread?.id,
    selectedThread?.messages.length,
    reduceMotion,
  ])

  function deleteSelectedThread() {
    if (!selectedThread || isIncognito) return

    onDeleteThread(selectedThread.id)
  }

  function toggleMessageSelectionMode() {
    setIsMessageSelectionMode((current) => {
      if (current) {
        setSelectedMessageIds(new Set())
        return false
      }
      if (!canManageMessages) return false
      setEditingMessageId(null)
      setMessageEditDraft('')
      return true
    })
  }

  function toggleSelectedMessage(messageId: string) {
    if (!isMessageSelectionMode) return
    setSelectedMessageIds((current) => {
      const next = new Set(current)
      if (next.has(messageId)) {
        next.delete(messageId)
      } else {
        next.add(messageId)
      }
      return next
    })
  }

  function deleteSelectedMessages() {
    if (!selectedThread || selectedMessageIds.size === 0) return
    onDeleteMessages(selectedThread.id, [...selectedMessageIds])
    setSelectedMessageIds(new Set())
    setIsMessageSelectionMode(false)
  }

  function startMessageEdit(message: ChatMessage) {
    if (!selectedThread || message.role !== 'user' || isSending) return
    setIsMessageSelectionMode(false)
    setSelectedMessageIds(new Set())
    setEditingMessageId(message.id)
    setMessageEditDraft(message.contentMarkdown)
  }

  function commitMessageEdit() {
    if (!selectedThread || !editingMessageId) return
    const nextContent = messageEditDraft.trim()
    if (!nextContent) return
    onEditMessage(selectedThread.id, editingMessageId, nextContent)
    setEditingMessageId(null)
    setMessageEditDraft('')
  }

  function cancelMessageEdit() {
    setEditingMessageId(null)
    setMessageEditDraft('')
  }

  function handleMessageEditKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Escape') {
      event.preventDefault()
      cancelMessageEdit()
      return
    }
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault()
      commitMessageEdit()
    }
  }

  function branchFromMessage(messageId: string) {
    if (!selectedThread || isIncognito || isSending) return
    onBranchFromMessage(selectedThread.id, messageId)
  }

  function retryAssistantMessage(
    messageId: string,
    mode: ChatRetryMode,
    options?: ChatRetryOptions,
  ) {
    if (!selectedThread || isSending) return
    onRetryAssistantMessage(selectedThread.id, messageId, mode, options)
  }

  function answerLastUserMessage(messageId: string) {
    if (!selectedThread || isSending) return
    onAnswerLastUserMessage(selectedThread.id, messageId)
  }

  function commitTitleEdit() {
    if (!selectedThread) return
    const nextTitle = titleDraft.trim()
    if (nextTitle) {
      onRenameThread(selectedThread.id, nextTitle)
    } else {
      setTitleDraft(selectedThread.title)
    }
    setIsEditingTitle(false)
  }

  function handleSendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    // Route an overflowing send through the confirm popover (Enter key path).
    if (contextOverflow && canSend) {
      setOverflowConfirmOpen(true)
      return
    }
    sendDraft()
  }

  function handleComposerChange() {
    const next = composerRef.current?.getMentionText() ?? ''
    setDraft(next)
    onChatDraftChange(next)
    setComposerNotice(null)
    draftTextImprove.clearProposal()
  }

  async function improveDraft() {
    setComposerNotice(null)
    try {
      await draftTextImprove.improve('chat_input', composerRef.current?.getMentionText() ?? draft)
    } catch (error) {
      setComposerNotice(messageFromUnknown(error))
    }
  }

  function acceptDraftImprovement(text: string) {
    composerRef.current?.setMentionText(text)
    const next = composerRef.current?.getMentionText() ?? text
    setDraft(next)
    onChatDraftChange(next)
    draftTextImprove.clearProposal()
    setDraftCommitPulseKey((key) => key + 1)
    window.requestAnimationFrame(() => composerRef.current?.focus())
  }

  function sendDraft() {
    const instruction = composerRef.current?.getInstructionText().trim() ?? ''
    if (!instruction || isSending) return
    shouldAutoFollowChatRef.current = true
    const knowledgeCollectionIds = (knowledgeIndexOptions ?? [])
      .filter((option) => selectedKnowledgeIndexIds.includes(option.id))
      .map((option) => option.collectionId)
    const modelOptions: ChatSendOptions | undefined = selectedChatModel
      ? { model: selectedChatModel, effort: selectedChatEffort }
      : selectedModelTier
        ? { modelTier: selectedModelTier }
        : undefined
    onSendMessage(
      instruction,
      pillRefs,
      knowledgeCollectionIds.length > 0
        ? { ...modelOptions, knowledgeCollectionIds }
        : modelOptions,
    )
    composerRef.current?.clear()
    setDraft('')
    setPillRefs([])
    onChatDraftChange('')
    setComposerNotice(null)
  }

  function handleComposerRefsChange(refs: ChatContextReferenceRecord[]) {
    setPillRefs(refs)
    onPillRefsChange(refs)
  }

  function handleRemoveChip(ref: ChatContextReferenceRecord) {
    // Target both sources unconditionally: removeRef no-ops when the chip is not a
    // live editor pill, onRemoveContext no-ops when it is not a pending attachment.
    // This keeps the "x" working even right after a remount, when the local pill
    // mirror has not yet re-derived from the editor (mirrors handleRemoveEditorChip).
    composerRef.current?.removeRef(ref)
    onRemoveContext(ref)
  }

  const historyPanel = (
    <ChatHistoryPanel
      chatHistorySections={chatHistorySections}
      hasMoreThreads={chatHistoryHasMore}
      isIncognito={isIncognito}
      isLoadingMoreThreads={chatHistoryLoadingMore}
      locale={locale}
      onLoadMoreThreads={onLoadMoreChatHistory}
      onCreateThread={onCreateThread}
      onCreateThreadGroup={onCreateThreadGroup}
      onDeleteThread={onDeleteThread}
      onDeleteThreadGroup={onDeleteThreadGroup}
      onMoveThreadGroup={onMoveThreadGroup}
      onMoveThreadToGroup={onMoveThreadToGroup}
      onRenameThread={onRenameThread}
      onRenameThreadGroup={onRenameThreadGroup}
      onSelectThread={onSelectThread}
      onTogglePinnedThread={onTogglePinnedThread}
      pinnedThreadIds={pinnedThreadIds}
      reduceMotion={reduceMotion}
      runningThreadIds={runningThreadIds}
      selectedThreadId={selectedThread?.id ?? null}
      threads={threads}
    />
  )

  const conversationPanel = (
        <section className="flex min-h-[620px] min-w-0 flex-col bg-background lg:h-full lg:min-h-0 lg:overflow-hidden">
	          <div
            className={cn(
              'z-10 flex inqtrix-panel-header items-center justify-between gap-2 border-b border-border bg-background px-4 transition-colors md:px-6',
              // Incognito makes a consequential state (nothing is saved) impossible
              // to miss: the header inverts to the opposite-mode neutral surface.
              // The treatment is token-scoped in globals.css, so title/icons/badge
              // follow automatically.
              isIncognito && 'inqtrix-chat-header--incognito',
            )}
          >
            <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
              {isDesktop && (
                <PanelToggle
                  collapseLabel={t.chat.hideHistory}
                  expandLabel={t.chat.showHistory}
                  expanded={isHistoryVisible}
                  onToggle={onHistoryVisibleChange}
                  side="left"
                />
              )}
              <MessageSquareText className="size-4 shrink-0 text-foreground/80" />
              <div className="min-w-0 flex-1 overflow-hidden" title={selectedThread ? selectedThread.preview : undefined}>
                <div className="flex min-w-0 items-center gap-2 overflow-hidden">
                  {isEditingTitle && selectedThread ? (
                    <input
                      aria-label={t.chat.renameTitle}
                      className="min-w-0 flex-1 rounded-sm border-0 bg-transparent px-0 t-section text-foreground outline-none focus-visible:ring-0"
                      onBlur={commitTitleEdit}
                      onChange={(event) => setTitleDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.preventDefault()
                          commitTitleEdit()
                        }
                        if (event.key === 'Escape') {
                          event.preventDefault()
                          setTitleDraft(selectedThread.title)
                          setIsEditingTitle(false)
                        }
                      }}
                      ref={titleInputRef}
                      value={titleDraft}
                    />
                  ) : (
                    <button
                      className="min-w-0 flex-1 truncate rounded-sm text-left t-section text-foreground hover:text-brand focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      onClick={() => selectedThread && setIsEditingTitle(true)}
                      title={selectedThread ? t.chat.renameTitle : undefined}
                      type="button"
                    >
                      {selectedThread ? selectedThread.title : t.chat.title}
                    </button>
                  )}
                  {isIncognito && (
                    <Badge
                      className="max-w-[min(44vw,24rem)] shrink border-brand/25 bg-brand-subtle text-brand hover:bg-brand-subtle"
                      title={t.chat.incognitoActive}
                      variant="outline"
                    >
                      <span className="truncate">{t.chat.incognitoActive}</span>
                    </Badge>
                  )}
                </div>
              </div>
	            </div>
	            <div className="flex shrink-0 items-center gap-2">
	              <Tooltip>
	                <TooltipTrigger asChild>
	                  <Button
                    aria-label={t.chat.incognito}
                    aria-pressed={isIncognito}
                    className={cn(
                      'size-7 text-foreground/75',
                      isIncognito && 'bg-brand-subtle text-brand hover:bg-brand-subtle',
                    )}
                    disabled={isSending}
                    onClick={() => onIncognitoChange(!isIncognito)}
                    size="icon"
                    type="button"
                    variant={isIncognito ? 'secondary' : 'ghost'}
                  >
                    <EyeOff className="size-4" />
                  </Button>
	                </TooltipTrigger>
	                <TooltipContent>{t.chat.incognito}</TooltipContent>
	              </Tooltip>
	              <div className="flex h-7 overflow-hidden rounded-md border border-border bg-card shadow-[0_1px_2px_var(--shadow-hairline)]">
	                <Tooltip>
	                  <TooltipTrigger asChild>
	                    <Button
	                      aria-label={isMessageSelectionMode ? t.chat.exitMessageEditMode : t.chat.editMessages}
	                      aria-pressed={isMessageSelectionMode}
	                      className={cn(
	                        'h-7 w-7 rounded-none border-r border-border text-foreground/75 hover:text-foreground',
	                        isMessageSelectionMode && 'bg-brand-subtle text-brand hover:bg-brand-subtle hover:text-brand',
	                      )}
	                      disabled={!canManageMessages && !isMessageSelectionMode}
	                      onClick={toggleMessageSelectionMode}
	                      size="icon"
	                      type="button"
	                      variant="ghost"
	                    >
	                      <ListChecks className="size-4" />
	                    </Button>
	                  </TooltipTrigger>
	                  <TooltipContent>
	                    {isMessageSelectionMode ? t.chat.exitMessageEditMode : t.chat.editMessages}
	                  </TooltipContent>
	                </Tooltip>
	                <Tooltip>
	                  <TooltipTrigger asChild>
	                    <Button
	                      aria-label={t.chat.clearChat}
                      className="h-7 w-7 rounded-none border-r border-border text-foreground/75 hover:text-foreground"
                      disabled={!selectedThread || isSending}
                      onClick={onClearThread}
                      size="icon"
                      type="button"
                      variant="ghost"
                    >
                      <Eraser className="size-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{t.chat.clearChat}</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      aria-label={t.chat.delete}
                      className="h-7 w-7 rounded-none text-foreground/75 hover:text-destructive"
                      disabled={!selectedThread || isIncognito}
                      onClick={deleteSelectedThread}
                      size="icon"
                      type="button"
                      variant="ghost"
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{t.chat.delete}</TooltipContent>
                </Tooltip>
	              </div>
	            </div>
	          </div>
	          <AnimatePresence initial={false}>
	            {isMessageSelectionMode && (
	              <motion.div
	                animate={{ height: 'auto', opacity: 1 }}
	                className="z-10 overflow-hidden border-b border-border bg-surface/80 px-4 md:px-6"
	                exit={{ height: 0, opacity: 0 }}
	                initial={reduceMotion ? false : { height: 0, opacity: 0 }}
	                transition={appMotion.panel}
	              >
	                <div className="mx-auto flex min-h-11 max-w-5xl items-center justify-between gap-3 py-2">
	                  <div className="flex min-w-0 items-center gap-2">
	                    <span className="flex size-7 items-center justify-center rounded-md border border-border bg-background text-foreground/80">
	                      <ListChecks className="size-3.5" />
	                    </span>
	                    <span className="truncate text-xs font-semibold text-foreground">
	                      {selectedMessageCount} {t.chat.messagesSelected}
	                    </span>
	                  </div>
	                  <div className="flex shrink-0 items-center gap-1">
	                    <Tooltip>
	                      <TooltipTrigger asChild>
	                        <Button
	                          aria-label={t.chat.deleteSelectedMessages}
	                          className="h-8 w-8 text-foreground/75 hover:text-destructive"
	                          disabled={selectedMessageCount === 0 || isSending}
	                          onClick={deleteSelectedMessages}
	                          size="icon"
	                          type="button"
	                          variant="ghost"
	                        >
	                          <Trash2 className="size-4" />
	                        </Button>
	                      </TooltipTrigger>
	                      <TooltipContent>{t.chat.deleteSelectedMessages}</TooltipContent>
	                    </Tooltip>
	                    <Tooltip>
	                      <TooltipTrigger asChild>
	                        <Button
	                          aria-label={t.chat.exitMessageEditMode}
	                          className="h-8 w-8 text-foreground/75 hover:text-foreground"
	                          onClick={toggleMessageSelectionMode}
	                          size="icon"
	                          type="button"
	                          variant="ghost"
	                        >
	                          <X className="size-4" />
	                        </Button>
	                      </TooltipTrigger>
	                      <TooltipContent>{t.chat.exitMessageEditMode}</TooltipContent>
	                    </Tooltip>
	                  </div>
	                </div>
	              </motion.div>
	            )}
	          </AnimatePresence>

	          <ScrollArea
            className={cn(
              'min-h-0 flex-1',
              // When empty, let the Radix viewport wrapper fill its height so the
              // inner `min-h-full` resolves and the hero can center vertically.
              // Only in the empty case, so message scrolling stays unaffected.
              !(selectedThread && selectedThread.messages.length > 0) &&
                '[&_[data-scroll-area-viewport]>div]:h-full',
            )}
            ref={messagesScrollAreaRef}
          >
	            <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col gap-5 px-4 py-6 md:px-8">
              {selectedThread && selectedThread.messages.length > 0 ? (
                selectedThread.messages.map((message, index) => {
                  const previousMessage = selectedThread.messages[index - 1]
                  const canRetryAssistantMessage = !isSending
                    && previousMessage?.role === 'user'
                    && previousMessage.contentMarkdown.trim().length > 0
                  return (
                  <ChatMessageBubble
                    canAnswerLastUserMessage={canAnswerLastUserMessage && message.id === lastMessage?.id}
                    canBranch={!isIncognito && !isSending}
                    canRetryAssistantMessage={canRetryAssistantMessage}
                    chatModelCatalog={chatModelCatalog}
                    chatModelOptions={chatModelOptions}
                    chatModelOptionsStatus={chatModelOptionsStatus}
                    defaultChatModel={defaultChatModel}
                    editDraft={messageEditDraft}
                    editTextareaRef={messageEditTextareaRef}
                    editingMessageId={editingMessageId}
                    isStreaming={message.id === activeAssistantMessageId}
                    isSelected={selectedMessageIds.has(message.id)}
                    isSelectionMode={isMessageSelectionMode}
                    key={message.id}
                    message={message}
                    onAnswerLastUserMessage={answerLastUserMessage}
                    onBranch={branchFromMessage}
                    onCancelEdit={cancelMessageEdit}
                    onCommitEdit={commitMessageEdit}
                    onEdit={startMessageEdit}
                    onEditDraftChange={setMessageEditDraft}
                    onEditKeyDown={handleMessageEditKeyDown}
                    onRetryAssistantMessage={retryAssistantMessage}
                    onToggleSelected={toggleSelectedMessage}
                    reduceMotion={reduceMotion}
                    selectedChatEffort={selectedChatEffort}
                    selectedChatModel={selectedChatModel}
                    selectedModelTier={selectedModelTier}
                  />
                  )
                })
              ) : selectedThread ? (
                <EmptyChatState
                  subtitle={pendingChips.length > 0 ? t.chat.emptyWithContext : t.chat.emptyHint}
                  title={t.chat.emptyTitle}
                />
              ) : (
                <EmptyChatState subtitle={t.chat.emptyHint} title={t.chat.emptyTitle} />
              )}
              <div ref={chatEndRef} />
            </div>
          </ScrollArea>

          <div className="z-10 shrink-0 px-3 pb-4 pt-2 md:px-6">
            <form
              className="mx-auto max-w-5xl"
              onSubmit={handleSendMessage}
            >
              <input
                className="hidden"
                multiple
                onChange={(event) => {
                  const files = Array.from(event.target.files ?? [])
                  if (files.length > 0) onAttachFiles(files)
                  event.target.value = ''
                }}
                ref={chatFileInputRef}
                type="file"
              />
              <Dropzone disabled={isSending} label={t.chat.dropFiles} onFiles={onAttachFiles}>
              <div className="relative overflow-visible rounded-xl border border-border bg-card px-2.5 py-2 shadow-[0_8px_28px_-12px_var(--shadow-soft)] transition-[border-color,box-shadow] duration-150 focus-within:border-brand/60 focus-within:ring-2 focus-within:ring-brand/15">
                {attachmentBudgetNotice && (
                  <div className="mb-2 flex items-center gap-1.5 rounded-md border border-warning/30 bg-warning/10 px-2 py-1 t-meta-sm font-medium text-warning">
                    <AlertTriangle className="size-3.5 shrink-0" />
                    <span className="min-w-0">{attachmentBudgetNotice}</span>
                  </div>
                )}
                <ContextChipLegend
                  chips={pendingChips}
                  labels={{
                    removeContext: t.chat.removeContextAttachment,
                    reorderHint: t.chat.reorderContextHint,
                  }}
                  onRemove={handleRemoveChip}
                  onReorderPending={onReorderContext}
                  onReorderPill={(from, to) => composerRef.current?.reorderPill(from, to)}
                  pendingKeys={pendingReorderKeys}
                  pillKeys={pillKeys}
                />
                <motion.div
                  animate={
                    draftCommitPulseKey > 0 && !reduceMotion
                      ? {
                        backgroundColor: [
                          'color-mix(in oklch, var(--brand) 0%, transparent)',
                          'color-mix(in oklch, var(--brand) 7%, transparent)',
                          'color-mix(in oklch, var(--brand) 0%, transparent)',
                        ],
                        boxShadow: [
                          '0 0 0 0 color-mix(in oklch, var(--brand) 0%, transparent)',
                          '0 0 0 3px color-mix(in oklch, var(--brand) 18%, transparent)',
                          '0 0 0 0 color-mix(in oklch, var(--brand) 0%, transparent)',
                        ],
                      }
                      : {
                        backgroundColor: 'color-mix(in oklch, var(--brand) 0%, transparent)',
                        boxShadow: '0 0 0 0 color-mix(in oklch, var(--brand) 0%, transparent)',
                      }
                  }
                  className="relative min-w-0 rounded-md"
                  transition={{ duration: 0.34, ease: appMotion.panel.ease }}
                >
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
                    onReject={draftTextImprove.clearProposal}
                    proposal={draftTextImprove.proposal}
                    reduceMotion={reduceMotion}
                  />
                  <MentionComposer
                    ariaLabel={t.chat.placeholder}
                    categoryLabels={mentionCategoryLabels}
                    contentClassName="min-h-16 pb-2 pl-2 pr-11 pt-2 text-sm leading-6"
                    enabledKinds={['research', 'rules', 'files', 'filegroups']}
                    mentionSources={mentionSources}
                    onAttachRule={(ruleId) => onAttachContext({ kind: 'chat-rule', ruleId })}
                    onChange={handleComposerChange}
                    onRefsChange={handleComposerRefsChange}
                    onSubmit={sendDraft}
                    placeholder={isIncognito ? t.chat.incognitoPlaceholder : t.chat.placeholder}
                    ref={composerRef}
                    resolveLabel={resolveMentionLabel}
                  />
                  <TextImproveButton
                    className="absolute right-1.5 top-1.5"
                    disabled={!draft.trim() || isSending}
                    isLoading={draftTextImprove.isImproving}
                    label={t.textImprove.improve}
                    loadingLabel={t.textImprove.improving}
                    onClick={() => void improveDraft()}
                    reduceMotion={reduceMotion}
                  />
                </motion.div>
                <div className="mt-1.5 flex items-center justify-between gap-2 border-t border-border/70 pt-1.5">
                  <div className="flex min-w-0 items-center gap-0.5 overflow-hidden">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        aria-label={t.chat.attachContext}
                        className={cn(composerIconButtonClassName, 'shrink-0')}
                        type="button"
                        variant="ghost"
                      >
                        <Plus />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent
                      align="start"
                      className="w-72 max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl p-0 shadow-lg"
                      side="top"
                      sideOffset={8}
                    >
                      <div className="flex items-center gap-1.5 border-b border-border px-2.5 py-1.5">
                        <span className="t-meta-sm font-medium text-muted-foreground">{t.chat.attachContext}</span>
                        <span className="ml-auto t-hint tabular-nums text-muted-foreground/50">
                          {reportOptions.length + ruleOptions.length}
                        </span>
                      </div>
                      <div className="py-1">
                        <div className="px-2.5 pb-0.5 pt-1.5 t-caption text-muted-foreground/60">
                          {t.chat.research}
                        </div>
                        {reportOptions.length > 0 ? (
                          reportOptions.map((report) => (
                            <DropdownMenuItem
                              className="group relative w-full min-w-0 items-start gap-2.5 rounded-none px-2.5 py-1.5 hover:bg-accent/50 focus:bg-accent/80 data-[highlighted]:bg-accent/80"
                              key={report.runId}
                              onSelect={() => onAttachContext({ kind: 'research-report', runId: report.runId })}
                            >
                              <span className="absolute inset-y-1 left-0 w-0.5 rounded-full bg-brand opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100 group-data-[highlighted]:opacity-100" />
                              <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground/70 transition-colors group-hover:text-brand group-focus:text-brand group-data-[highlighted]:text-brand" />
                              <span className="min-w-0 flex-1">
                                <span className="block max-w-full truncate t-list text-foreground">
                                  @research:{report.label}
                                </span>
                                <span className="block max-w-full truncate t-meta-sm text-muted-foreground">
                                  {report.title}
                                </span>
                              </span>
                            </DropdownMenuItem>
                          ))
                        ) : (
                          <div className="px-2.5 py-2 t-meta text-muted-foreground">
                            {t.chat.noReports}
                          </div>
                        )}
                        <DropdownMenuSeparator className="mx-0 my-1" />
                        <div className="px-2.5 pb-0.5 pt-1.5 t-caption text-muted-foreground/60">
                          {t.chat.rules}
                        </div>
                        {ruleOptions.length > 0 ? (
                          ruleOptions.map((rule) => (
                            <DropdownMenuItem
                              className="group relative w-full min-w-0 items-start gap-2.5 rounded-none px-2.5 py-1.5 hover:bg-accent/50 focus:bg-accent/80 data-[highlighted]:bg-accent/80"
                              key={rule.ruleId}
                              onSelect={() => onAttachContext({ kind: 'chat-rule', ruleId: rule.ruleId })}
                            >
                              <span className="absolute inset-y-1 left-0 w-0.5 rounded-full bg-success opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100 group-data-[highlighted]:opacity-100" />
                              <BookOpen className="mt-0.5 size-4 shrink-0 text-muted-foreground/70 transition-colors group-hover:text-success group-focus:text-success group-data-[highlighted]:text-success" />
                              <span className="min-w-0 flex-1">
                                <span className="block max-w-full truncate t-list text-foreground">
                                  @rules:{rule.label}
                                </span>
                                <span className="block max-w-full truncate t-meta-sm text-muted-foreground">
                                  {rule.title}
                                </span>
                              </span>
                            </DropdownMenuItem>
                          ))
                        ) : (
                          <div className="px-2.5 py-2 t-meta text-muted-foreground">
                            {t.chat.noRules}
                          </div>
                        )}
                      </div>
                      <DropdownMenuSeparator className="mx-0 my-0" />
                      <DropdownMenuItem
                        className="group relative items-start gap-2.5 rounded-none px-2.5 py-1.5 hover:bg-accent/50 focus:bg-accent/80 data-[highlighted]:bg-accent/80"
                        onSelect={onOpenPromptLibrary}
                      >
                        <span className="absolute inset-y-1 left-0 w-0.5 rounded-full bg-success opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100 group-data-[highlighted]:opacity-100" />
                        <Library className="mt-0.5 size-4 shrink-0 text-muted-foreground/70 transition-colors group-hover:text-success group-focus:text-success group-data-[highlighted]:text-success" />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate t-list text-foreground">{t.chat.manageRules}</span>
                          <span className="block truncate t-meta-sm text-muted-foreground">
                            {t.chat.manageRulesDescription}
                          </span>
                        </span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <ComposerIconButton
                    className="shrink-0"
                    disabled={isSending}
                    icon={Paperclip}
                    label={t.chat.attachFiles}
                    onClick={() => chatFileInputRef.current?.click()}
                  />
                  <ComposerIconButton
                    active={chainingEnabled}
                    className="shrink-0"
                    disabled={isSending}
                    icon={ListOrdered}
                    label={`${t.chat.chaining} · ${t.chat.chainingTooltip}`}
                    onClick={() => onChainingEnabledChange(!chainingEnabled)}
                  />
                  {knowledgeIndexOptions && knowledgeIndexOptions.length > 0 ? (
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          aria-label={t.chat.knowledgeScope}
                          className={cn(
                            composerIconButtonClassName,
                            'shrink-0',
                            selectedKnowledgeIndexIds.length > 0 && 'bg-brand-subtle text-brand hover:text-brand',
                          )}
                          disabled={isSending}
                          type="button"
                          variant="ghost"
                        >
                          <Database />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="start" className={optionMenuContentClassName} side="top" sideOffset={8}>
                        <OptionMenuHeader
                          count={knowledgeIndexOptions.length}
                          title={t.chat.knowledgeScope}
                          value={
                            selectedKnowledgeIndexIds.length > 0
                              ? t.chat.knowledgeScopeActive.replace('{count}', String(selectedKnowledgeIndexIds.length))
                              : t.chat.knowledgeScopeOff
                          }
                        />
                        <div className="py-1">
                          <OptionMenuItem
                            active={selectedKnowledgeIndexIds.length === 0}
                            description={t.chat.knowledgeScopeOffDescription}
                            icon={MessageSquareText}
                            label={t.chat.knowledgeScopeOff}
                            onSelect={() => onSelectedKnowledgeIndexIdsChange([])}
                          />
                          {knowledgeIndexOptions.map((option) => (
                            <OptionMenuItem
                              active={selectedKnowledgeIndexIds.includes(option.id)}
                              description={t.chat.knowledgeScopeIndexDescription}
                              icon={Database}
                              key={option.id}
                              label={option.title}
                              onSelect={() =>
                                onSelectedKnowledgeIndexIdsChange(
                                  selectedKnowledgeIndexIds.includes(option.id)
                                    ? selectedKnowledgeIndexIds.filter((id) => id !== option.id)
                                    : [...selectedKnowledgeIndexIds, option.id],
                                )
                              }
                            />
                          ))}
                        </div>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  ) : null}
                  <ModelTierPicker
                    defaultModel={defaultChatModel}
                    disabled={isSending}
                    onChange={onSelectedModelTierChange}
                    options={chatModelOptions}
                    optionsStatus={chatModelOptionsStatus}
                    selectedTier={selectedModelTier}
                    modelCatalog={chatModelCatalog}
                    selectedModel={selectedChatModel}
                    selectedEffort={selectedChatEffort}
                    onModelChange={onSelectedChatModelChange}
                    onEffortChange={onSelectedChatEffortChange}
                  />
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        aria-label={t.composer.moreSettings}
                        className={cn(composerIconButtonClassName, 'shrink-0')}
                        type="button"
                        variant="ghost"
                      >
                        <SlidersHorizontal />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start" className={optionMenuContentClassName} side="top" sideOffset={8}>
                      <OptionMenuHeader count={1} title={t.composer.moreSettings} />
                      <div className="py-1">
                      <ChatComposerToggleItem
                        checked={streamingEnabled}
                        description={t.chat.streamingDescription}
                        disabled={isSending}
                        icon={MessageSquareText}
                        label={t.chat.streaming}
                        offLabel={t.chat.streamingOff}
                        onCheckedChange={onStreamingEnabledChange}
                        onLabel={t.chat.streamingOn}
                      />
                      </div>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <QuotaMeter disabled={isSending} />
                    <ContextTokenMeter
                      conversationLabel={t.chat.contextCatHistory}
                      disabled={isSending}
                      model={contextTokenModel}
                    />
                    {isSending ? (
                      <ComposerStopButton
                        label={t.chat.stopGenerating}
                        onClick={onStopGenerating}
                      />
                    ) : contextOverflow ? (
                      <DropdownMenu onOpenChange={setOverflowConfirmOpen} open={overflowConfirmOpen}>
                        <DropdownMenuTrigger asChild>
                          <Button
                            aria-label={t.chat.send}
                            className={cn(
                              'size-7 rounded-md focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-0',
                              canSend
                                ? 'bg-brand text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground'
                                : 'text-muted-foreground/45',
                            )}
                            disabled={!canSend}
                            size="icon"
                            type="button"
                            variant={canSend ? 'default' : 'ghost'}
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
                                sendDraft()
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
                      <Button
                        aria-label={t.chat.send}
                        className={cn(
                          'size-7 rounded-md focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-0',
                          canSend
                            ? 'bg-brand text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground'
                            : 'text-muted-foreground/45',
                        )}
                        disabled={!canSend}
                        size="icon"
                        type="submit"
                        variant={canSend ? 'default' : 'ghost'}
                      >
                        <SendHorizontal className="size-4" />
                      </Button>
                    )}
                  </div>
                </div>
              </div>
              </Dropzone>
              {(requestError || requestNotice || composerNotice) && (
                <div className="mt-1 flex min-h-6 flex-wrap items-center justify-end gap-2 border-t border-border/70 pt-1.5">
                  {requestError && (
                    <span className="min-w-0 truncate t-meta-sm font-medium text-destructive">
                      {requestError}
                    </span>
                  )}
                  {!requestError && requestNotice && (
                    <span className="min-w-0 truncate t-meta-sm font-medium text-warning">
                      {requestNotice}
                    </span>
                  )}
                  {!requestError && !requestNotice && composerNotice && (
                    <span className="min-w-0 truncate t-meta-sm font-medium text-warning">
                      {composerNotice}
                    </span>
                  )}
                </div>
              )}
            </form>
          </div>
        </section>
  )

  return (
    <div className="flex min-h-[calc(100svh-var(--header-h))] w-full lg:h-full lg:min-h-0">
      {isDesktop ? (
        isHistoryVisible ? (
          <ResizablePanelGroup
            className="min-h-0 w-full overflow-hidden bg-background"
            orientation="horizontal"
          >
            <ResizablePanel
              className="min-h-0 min-w-0 overflow-hidden bg-surface/60"
              defaultSize="26%"
              maxSize="42%"
              minSize="18%"
            >
              {historyPanel}
            </ResizablePanel>
            <ResizableHandle aria-label={t.chat.resizeHistory} />
            <ResizablePanel className="min-h-0 min-w-0 overflow-hidden" defaultSize="74%" minSize="58%">
              {conversationPanel}
            </ResizablePanel>
          </ResizablePanelGroup>
        ) : (
          <div className="flex min-h-0 w-full overflow-hidden bg-background">
            <div className="min-h-0 min-w-0 flex-1 overflow-hidden">{conversationPanel}</div>
          </div>
        )
      ) : (
        <motion.section
          initial={reduceMotion ? false : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={appMotion.panel}
          className="flex min-h-0 w-full flex-col overflow-hidden bg-background"
        >
          {historyPanel}
          {conversationPanel}
        </motion.section>
      )}
    </div>
  )
}

function ChatToggleVisual({ checked }: { checked: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        'inline-flex h-5 w-9 shrink-0 items-center rounded-full border-2 border-transparent shadow-sm transition-colors',
        checked ? 'bg-brand' : 'bg-input',
      )}
    >
      <span
        className={cn(
          'block h-4 w-4 rounded-full bg-background shadow-lg ring-0 transition-transform',
          checked ? 'translate-x-4' : 'translate-x-0',
        )}
      />
    </span>
  )
}

function ChatComposerToggleItem({
  checked,
  description,
  disabled,
  icon: Icon,
  label,
  offLabel,
  onCheckedChange,
  onLabel,
}: {
  checked: boolean
  description: string
  disabled?: boolean
  icon: LucideIcon
  label: string
  offLabel: string
  onCheckedChange: (checked: boolean) => void
  onLabel: string
}) {
  return (
    <DropdownMenuItem
      className="group relative items-center gap-2.5 rounded-none px-2.5 py-1.5 hover:bg-accent/50 focus:bg-accent/80 data-[highlighted]:bg-accent/80"
      disabled={disabled}
      onSelect={(event) => {
        event.preventDefault()
        onCheckedChange(!checked)
      }}
    >
      <span
        className={cn(
          'absolute inset-y-1 left-0 w-0.5 rounded-full opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100 group-data-[highlighted]:opacity-100',
          checked ? 'bg-success' : 'bg-muted-foreground/50',
        )}
      />
      <Icon
        className={cn(
          'icon-md shrink-0 transition-colors',
          checked
            ? 'text-success'
            : 'text-muted-foreground/70 group-hover:text-foreground group-focus:text-foreground group-data-[highlighted]:text-foreground',
        )}
      />
      <span className="min-w-0 flex-1 text-left">
        <span className="block truncate t-list text-foreground">{label}</span>
        <span className="block truncate t-meta-sm text-muted-foreground">{description}</span>
      </span>
      <span
        className={cn(
          'shrink-0 rounded-md px-1.5 py-0.5 t-hint font-medium',
          checked ? 'bg-success-subtle text-success' : 'bg-surface text-muted-foreground',
        )}
      >
        {checked ? onLabel : offLabel}
      </span>
      <button
        aria-label={`${label}: ${checked ? onLabel : offLabel}`}
        className="shrink-0 disabled:cursor-not-allowed disabled:opacity-40"
        disabled={disabled}
        onClick={(event) => {
          event.stopPropagation()
          onCheckedChange(!checked)
        }}
        type="button"
      >
        <ChatToggleVisual checked={checked} />
      </button>
    </DropdownMenuItem>
  )
}

function ChatMessageModelChip({
  modelCatalog,
  modelResolution,
}: {
  modelCatalog?: ModelCatalogEntry[]
  modelResolution: ChatMessage['modelResolution']
}) {
  const { t } = useLocale()
  if (!modelResolution?.model) return null
  const label = chatMessageModelLabel(modelResolution, t.chat, modelCatalog)
    ?? modelNameLabel(modelResolution, t.chat.modelUnknown)
  return (
    <span
      className="max-w-[min(42vw,16rem)] truncate rounded-md bg-muted/60 px-1.5 py-0.5 t-hint font-semibold text-muted-foreground"
      title={label}
    >
      {label}
    </span>
  )
}

/**
 * Highlight the `@kind` context tokens in a hint sentence so they read as
 * typeable handles (mono, slightly darker) without leaving the prose flow.
 */
function renderMentionHint(text: string) {
  return text.split(/(@[a-z]+)/gi).map((part, index) =>
    /^@[a-z]+$/i.test(part) ? (
      <span className="font-mono text-foreground/75" key={index}>
        {part}
      </span>
    ) : (
      part
    ),
  )
}

function EmptyChatState({
  subtitle,
  title,
}: {
  subtitle: string
  title: string
}) {
  const { t } = useLocale()

  return (
    <div className="flex flex-1 items-center justify-center px-6 py-8 text-center">
      <WelcomeState
        body={(
          <>
            <p>{t.chat.emptyBody}</p>
            <p>{renderMentionHint(t.chat.emptyGuidance)}</p>
          </>
        )}
        example={t.chat.emptyExample}
        kicker={t.chat.title}
        subtitle={renderMentionHint(subtitle)}
        title={title}
      />
    </div>
  )
}

function ChatMessageBubble({
  canAnswerLastUserMessage,
  canBranch,
  canRetryAssistantMessage,
  chatModelCatalog,
  chatModelOptions,
  chatModelOptionsStatus,
  defaultChatModel,
  editDraft,
  editTextareaRef,
  editingMessageId,
  isSelected,
  isSelectionMode,
  isStreaming,
  message,
  onAnswerLastUserMessage,
  onBranch,
  onCancelEdit,
  onCommitEdit,
  onEdit,
  onEditDraftChange,
  onEditKeyDown,
  onRetryAssistantMessage,
  onToggleSelected,
  reduceMotion,
  selectedChatEffort,
  selectedChatModel,
  selectedModelTier,
}: {
  canAnswerLastUserMessage: boolean
  canBranch: boolean
  canRetryAssistantMessage: boolean
  chatModelCatalog: ModelCatalogEntry[]
  chatModelOptions: ChatModelOption[]
  chatModelOptionsStatus: 'available' | 'missing' | 'unresolved'
  defaultChatModel: NodeModelResolution | null
  editDraft: string
  editTextareaRef: RefObject<HTMLTextAreaElement | null>
  editingMessageId: string | null
  isSelected: boolean
  isSelectionMode: boolean
  isStreaming: boolean
  message: ChatMessage
  onAnswerLastUserMessage: (messageId: string) => void
  onBranch: (messageId: string) => void
  onCancelEdit: () => void
  onCommitEdit: () => void
  onEdit: (message: ChatMessage) => void
  onEditDraftChange: (value: string) => void
  onEditKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void
  onRetryAssistantMessage: (messageId: string, mode: ChatRetryMode, options?: ChatRetryOptions) => void
  onToggleSelected: (messageId: string) => void
  reduceMotion: boolean | null
  selectedChatEffort: string | null
  selectedChatModel: string | null
  selectedModelTier: ChatModelTier | null
}) {
  const { locale, t } = useLocale()
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'
  const isEditing = editingMessageId === message.id
  const Icon = isUser ? CircleUserRound : Bot
  const canCopy = message.contentMarkdown.trim().length > 0
  const canContinueFromHere = !isUser && !isSelectionMode && canBranch && canCopy && !isStreaming
  const canGenerateAnswer = isUser && !isSelectionMode && canAnswerLastUserMessage && canCopy && !isStreaming && !isEditing
  const timestampLabel = formatMessageTimestamp(message.createdAt, locale)
  const selectionLabel = isSelected ? t.chat.messageSelected : t.chat.selectMessage
  const selectionControl = isSelectionMode ? (
    <MessageSelectionControl
      isSelected={isSelected}
      label={isSelected ? t.chat.deselectMessage : t.chat.selectMessage}
      onToggle={() => onToggleSelected(message.id)}
    />
  ) : null
  const selectionRowClassName = cn(
    'group/message -mx-4 rounded-xl border border-transparent px-4 py-1 transition-colors md:-mx-8 md:px-8',
    isSelectionMode && 'cursor-pointer hover:border-border/70 hover:bg-surface/70',
    isSelected && 'border-brand/25 bg-brand-subtle/80 ring-1 ring-brand/20 hover:bg-brand-subtle/80',
  )
  const selectionRowClick = isSelectionMode ? () => onToggleSelected(message.id) : undefined

  async function copyMessage() {
    if (!canCopy) return
    try {
      await navigator.clipboard.writeText(message.contentMarkdown)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1200)
    } catch (error) {
      console.warn('Inqtrix chat message copy failed.', error)
    }
  }

  if (!isUser) {
    return (
      <div
        className={selectionRowClassName}
        onClick={selectionRowClick}
      >
        <div
          className={cn(
            'grid min-w-0 gap-3',
            isSelectionMode ? 'grid-cols-[24px_32px_minmax(0,1fr)]' : 'grid-cols-[32px_minmax(0,1fr)]',
          )}
        >
          {selectionControl}
          <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md border border-border bg-surface text-muted-foreground">
            <Icon className="size-4" />
          </span>
          <div className="min-w-0" aria-live={isStreaming ? 'polite' : undefined}>
            <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 t-meta-sm font-semibold text-muted-foreground">
              <span>{t.chat.assistant}</span>
              <ChatMessageModelChip modelCatalog={chatModelCatalog} modelResolution={message.modelResolution} />
              <span className="whitespace-nowrap tabular-nums">{timestampLabel}</span>
              {isSelectionMode && (
                <MessageSelectionPill isSelected={isSelected} label={selectionLabel} />
              )}
            </div>
            {message.chainTrace && message.chainTrace.length > 0 && (
              <ChatChainTrace steps={message.chainTrace} />
            )}
            {message.contentMarkdown ? (
              <MarkdownSelectionCopyMenu
                className={cn(
                  'chat-markdown max-w-4xl text-sm leading-snug text-foreground',
                  isStreaming && !reduceMotion && 'animate-in fade-in-0 duration-200',
                )}
                markdown={message.contentMarkdown}
              >
                <MarkdownRenderer
                  isStreaming={isStreaming}
                  markdown={message.contentMarkdown}
                  variant="chat"
                />
                {isStreaming && (
                  <span
                    aria-hidden="true"
                    className={cn(
                      'ml-1 inline-block size-2 rounded-full bg-brand align-middle',
                      !reduceMotion && 'animate-pulse',
                    )}
                  />
                )}
              </MarkdownSelectionCopyMenu>
            ) : (
              !(message.chainTrace && message.chainTrace.length > 0) && (
                <GeneratingPlaceholder reduceMotion={reduceMotion} />
              )
            )}
            <ChatMessageAttachments attachments={message.attachments} />
            {!isSelectionMode && (
              <MessageActionRow align="start">
                <MessageActionButton
                  canCopy={canCopy}
                  copied={copied}
                  icon="copy"
                  label={copied ? t.chat.copiedMessage : t.chat.copyMessage}
                  onClick={() => void copyMessage()}
                />
                <AssistantMessageMenu
                  canBranch={canContinueFromHere}
                  canRetry={canRetryAssistantMessage && canCopy && !isStreaming}
                  assistantModelLabel={chatMessageModelLabel(message.modelResolution, t.chat, chatModelCatalog)}
                  chatModelCatalog={chatModelCatalog}
                  chatModelOptions={chatModelOptions}
                  chatModelOptionsStatus={chatModelOptionsStatus}
                  defaultChatModel={defaultChatModel}
                  messageId={message.id}
                  onBranch={onBranch}
                  onRetry={onRetryAssistantMessage}
                  selectedChatEffort={selectedChatEffort}
                  selectedChatModel={selectedChatModel}
                  selectedModelTier={selectedModelTier}
                  timestampLabel={timestampLabel}
                />
              </MessageActionRow>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={selectionRowClassName}
      onClick={selectionRowClick}
    >
      <div className="flex min-w-0 items-start gap-3">
        {selectionControl}
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 justify-end">
            <div className="min-w-0 max-w-[min(72%,44rem)]">
              <div className="mb-1 flex flex-wrap items-center justify-end gap-x-2 gap-y-0.5 t-meta-sm font-semibold text-muted-foreground">
                {isSelectionMode && (
                  <MessageSelectionPill isSelected={isSelected} label={selectionLabel} />
                )}
                <span>{t.chat.you}</span>
                <span className="whitespace-nowrap tabular-nums">{timestampLabel}</span>
              </div>
              <ChatMessageAttachments align="end" attachments={message.attachments} />
              {isEditing ? (
                <ChatMessageEditForm
                  draft={editDraft}
                  onCancel={onCancelEdit}
                  onChange={onEditDraftChange}
                  onCommit={onCommitEdit}
                  onKeyDown={onEditKeyDown}
                  textareaRef={editTextareaRef}
                />
              ) : (
                <div className="inqtrix-user-bubble rounded-lg px-3 py-2.5 text-sm leading-6 shadow-[0_1px_2px_var(--shadow-hairline)]">
                  {message.contentMarkdown}
                </div>
              )}
              {!isSelectionMode && !isEditing && (
                <MessageActionRow align="end">
                  <MessageActionButton
                    canCopy={canCopy}
                    copied={copied}
                    icon="copy"
                    label={copied ? t.chat.copiedMessage : t.chat.copyMessage}
                    onClick={() => void copyMessage()}
                  />
                  {!isStreaming && (
                    <MessageActionButton
                      icon="edit"
                      label={t.chat.editMessage}
                      onClick={() => onEdit(message)}
                    />
                  )}
                  {canGenerateAnswer && (
                    <MessageActionButton
                      icon="answer"
                      label={t.chat.generateAnswer}
                      onClick={() => onAnswerLastUserMessage(message.id)}
                    />
                  )}
                </MessageActionRow>
              )}
            </div>
          </div>
        </div>
        <span className="inqtrix-user-avatar mt-1 flex size-8 shrink-0 items-center justify-center rounded-md border">
          <Icon className="size-4" />
        </span>
      </div>
    </div>
  )
}

function MessageSelectionPill({
  isSelected,
  label,
}: {
  isSelected: boolean
  label: string
}) {
  return (
    <span
      className={cn(
        'rounded-full border border-border bg-background px-1.5 py-0.5 t-hint font-semibold text-muted-foreground',
        isSelected && 'border-brand/30 bg-brand-subtle text-brand',
      )}
    >
      {label}
    </span>
  )
}

function MessageSelectionControl({
  isSelected,
  label,
  onToggle,
}: {
  isSelected: boolean
  label: string
  onToggle: () => void
}) {
  return (
    <Button
      aria-label={label}
      aria-pressed={isSelected}
      className={cn(
        'mt-1 size-6 shrink-0 rounded-full border border-border bg-background text-foreground/75 shadow-[0_1px_2px_var(--shadow-hairline)] hover:border-brand/45 hover:text-brand',
        isSelected && 'border-brand bg-brand text-primary-foreground hover:bg-brand hover:text-primary-foreground',
      )}
      onClick={(event) => {
        event.stopPropagation()
        onToggle()
      }}
      size="icon"
      type="button"
      variant="ghost"
    >
      {isSelected ? <Check className="size-3.5" /> : <span className="size-2 rounded-full border border-current" />}
    </Button>
  )
}

function ChatMessageEditForm({
  draft,
  onCancel,
  onChange,
  onCommit,
  onKeyDown,
  textareaRef,
}: {
  draft: string
  onCancel: () => void
  onChange: (value: string) => void
  onCommit: () => void
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void
  textareaRef: RefObject<HTMLTextAreaElement | null>
}) {
  const { t } = useLocale()
  const canSave = draft.trim().length > 0

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (canSave) {
      onCommit()
    }
  }

  return (
    <form
      className="rounded-lg border border-brand/30 bg-card p-2 shadow-[0_1px_2px_var(--shadow-hairline)]"
      onClick={(event) => event.stopPropagation()}
      onSubmit={handleSubmit}
    >
      <Textarea
        aria-label={t.chat.editMessage}
        className="min-h-24 resize-none border-0 bg-transparent px-2 py-1.5 text-sm leading-6 text-foreground focus-visible:ring-0"
        onChange={(event) => {
          onChange(event.target.value)
          resizeTextareaToRows(event.target, 8)
        }}
        onKeyDown={onKeyDown}
        ref={textareaRef}
        rows={3}
        value={draft}
      />
      <div className="mt-2 flex items-center justify-end gap-1.5 border-t border-border/70 pt-2">
        <Button
          className="h-8 gap-1.5 px-2 text-xs"
          onClick={onCancel}
          type="button"
          variant="ghost"
        >
          <X className="size-3.5" />
          {t.chat.cancelMessageEdit}
        </Button>
        <Button
          className="h-8 gap-1.5 px-2 text-xs"
          disabled={!canSave}
          type="submit"
          variant="secondary"
        >
          <Save className="size-3.5" />
          {t.chat.saveMessageEdit}
        </Button>
      </div>
    </form>
  )
}

function MessageActionRow({
  align,
  children,
}: {
  align: 'end' | 'start'
  children: ReactNode
}) {
  return (
    <div
      className={cn(
        'mt-1 flex min-h-6 items-center gap-1 text-muted-foreground',
        align === 'end' ? 'justify-end' : 'justify-start',
      )}
    >
      {children}
    </div>
  )
}

function retryModelLabel({
  catalog,
  defaultModel,
  model,
  options,
  tier,
  t,
}: {
  catalog: ModelCatalogEntry[]
  defaultModel: NodeModelResolution | null
  model: string | null
  options: ChatModelOption[]
  tier: ChatModelTier | null
  t: ReturnType<typeof useLocale>['t']['chat']
}) {
  if (model) {
    return modelDisplayName(model, catalog, model)
  }
  if (tier) {
    const option = options.find((candidate) => candidate.tier === tier)
    const label = option?.model
      ? modelDisplayName(option.model, catalog, modelNameLabel(option, t.modelUnknown))
      : modelNameLabel(option, t.modelUnknown)
    return `${label} · ${modelTierLabel(tier, t)}`
  }
  const defaultLabel = defaultModel?.model
    ? modelDisplayName(defaultModel.model, catalog, modelNameLabel(defaultModel, t.modelUnknown))
    : modelNameLabel(defaultModel, t.modelUnknown)
  return defaultModel?.model ? `${t.modelServerDefault} · ${defaultLabel}` : t.modelServerDefault
}

function modelDisplayName(
  modelId: string,
  modelCatalog: ModelCatalogEntry[],
  fallback: string,
) {
  return modelCatalog.find((entry) => entry.model_id === modelId)?.card?.display_name ?? fallback
}

function chatMessageModelLabel(
  modelResolution: ChatMessage['modelResolution'],
  t: ReturnType<typeof useLocale>['t']['chat'],
  modelCatalog: ModelCatalogEntry[] = [],
) {
  if (!modelResolution?.model) return null
  const modelLabel = modelDisplayName(
    modelResolution.model,
    modelCatalog,
    modelNameLabel(modelResolution, t.modelUnknown),
  )
  return `${modelLabel} · ${modelEffortLabelFromToken(modelResolution.effort, t)}`
}

function AssistantMessageMenu({
  assistantModelLabel,
  canBranch,
  canRetry,
  chatModelCatalog,
  chatModelOptions,
  chatModelOptionsStatus,
  defaultChatModel,
  messageId,
  onBranch,
  onRetry,
  selectedChatEffort,
  selectedChatModel,
  selectedModelTier,
  timestampLabel,
}: {
  assistantModelLabel: string | null
  canBranch: boolean
  canRetry: boolean
  chatModelCatalog: ModelCatalogEntry[]
  chatModelOptions: ChatModelOption[]
  chatModelOptionsStatus: 'available' | 'missing' | 'unresolved'
  defaultChatModel: NodeModelResolution | null
  messageId: string
  onBranch: (messageId: string) => void
  onRetry: (messageId: string, mode: ChatRetryMode, options?: ChatRetryOptions) => void
  selectedChatEffort: string | null
  selectedChatModel: string | null
  selectedModelTier: ChatModelTier | null
  timestampLabel: string
}) {
  const { t } = useLocale()
  const [retryModel, setRetryModel] = useState<string | null>(selectedChatModel)
  const [retryTier, setRetryTier] = useState<ChatModelTier | null>(selectedModelTier)
  const [retryEffort, setRetryEffort] = useState<string | null>(selectedChatEffort)

  useEffect(() => {
    setRetryModel(selectedChatModel)
    setRetryTier(selectedModelTier)
    setRetryEffort(selectedChatEffort)
  }, [selectedChatEffort, selectedChatModel, selectedModelTier])

  const retryOptions: ChatRetryOptions = {
    effort: retryEffort,
    model: retryModel,
    modelTier: retryModel ? null : retryTier,
  }
  const modelLabel = retryModelLabel({
    catalog: chatModelCatalog,
    defaultModel: defaultChatModel,
    model: retryModel,
    options: chatModelOptions,
    tier: retryTier,
    t: t.chat,
  })

  function retry(mode: ChatRetryMode) {
    if (!canRetry) return
    onRetry(messageId, mode, retryOptions)
  }

  return (
    <DropdownMenu modal={false}>
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button
              aria-label={t.chat.messageOptions}
              className="size-6 text-foreground/60 transition-colors hover:text-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-0 data-[state=open]:bg-accent data-[state=open]:text-foreground"
              onClick={(event) => event.stopPropagation()}
              size="icon"
              type="button"
              variant="ghost"
            >
              <MoreHorizontal className="icon-sm" />
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent>{t.chat.messageOptions}</TooltipContent>
      </Tooltip>
      <DropdownMenuContent
        align="start"
        className={assistantMenuContentClassName}
        onClick={(event) => event.stopPropagation()}
        side="top"
        sideOffset={6}
      >
        <AssistantMenuHeader
          primary={timestampLabel}
          secondary={assistantModelLabel ? t.chat.usedModel(assistantModelLabel) : t.chat.modelUsedMissing}
        />
        <div className="p-1">
          {canBranch && (
            <DropdownMenuItem
              className="group gap-2 rounded-md px-2 py-1.5"
              onSelect={() => onBranch(messageId)}
            >
              <AssistantMenuIcon icon={GitBranchPlus} />
              <AssistantMenuLabel>{t.chat.continueFromHere}</AssistantMenuLabel>
            </DropdownMenuItem>
          )}
          <DropdownMenuSub>
            <DropdownMenuSubTrigger
              className="group gap-2 rounded-md px-2 py-1.5"
              disabled={!canRetry}
            >
              <AssistantMenuIcon icon={RefreshCw} />
              <AssistantMenuLabel>{t.chat.retryMessage}</AssistantMenuLabel>
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent className={assistantMenuContentClassName}>
              <AssistantMenuHeader
                primary={assistantModelLabel ? t.chat.usedModel(assistantModelLabel) : t.chat.retryMessage}
              />
              <div className="p-1">
                <RetryPresetItem icon={RefreshCw} label={t.chat.retryMessage} onSelect={() => retry('plain')} />
                <RetryPresetItem icon={ListPlus} label={t.chat.retryAddDetails} onSelect={() => retry('details')} />
                <RetryPresetItem icon={ListMinus} label={t.chat.retryShorter} onSelect={() => retry('shorter')} />
              </div>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
          <DropdownMenuSeparator className="my-1" />
          <DropdownMenuSub>
            <DropdownMenuSubTrigger
              className="group gap-2 rounded-md px-2 py-1.5"
              disabled={!canRetry || (chatModelOptionsStatus !== 'available' && chatModelCatalog.length === 0)}
            >
              <AssistantMenuIcon icon={BrainCircuit} />
              <AssistantMenuLabel>{t.chat.modelSwitch}</AssistantMenuLabel>
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent
              className="flex max-h-[var(--radix-dropdown-menu-content-available-height)] w-72 max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-lg p-0 shadow-lg"
              collisionPadding={12}
              sideOffset={4}
            >
              <AssistantRetryModelMenu
                catalog={chatModelCatalog}
                defaultModel={defaultChatModel}
                model={retryModel}
                modelLabel={modelLabel}
                onEffortChange={setRetryEffort}
                onModelChange={(model) => {
                  setRetryModel(model)
                  setRetryTier(null)
                  setRetryEffort(null)
                }}
                onTierChange={(tier) => {
                  setRetryModel(null)
                  setRetryTier(tier)
                  setRetryEffort(null)
                }}
                options={chatModelOptions}
                optionsStatus={chatModelOptionsStatus}
                selectedEffort={retryEffort}
                tier={retryTier}
              />
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function RetryPresetItem({
  icon: Icon,
  label,
  onSelect,
}: {
  icon: LucideIcon
  label: string
  onSelect: () => void
}) {
  return (
    <DropdownMenuItem className="group gap-2 rounded-md px-2 py-1.5" onSelect={onSelect}>
      <AssistantMenuIcon icon={Icon} />
      <AssistantMenuLabel>{label}</AssistantMenuLabel>
    </DropdownMenuItem>
  )
}

function AssistantRetryModelMenu({
  catalog,
  defaultModel,
  model,
  modelLabel,
  onEffortChange,
  onModelChange,
  onTierChange,
  options,
  optionsStatus,
  selectedEffort,
  tier,
}: {
  catalog: ModelCatalogEntry[]
  defaultModel: NodeModelResolution | null
  model: string | null
  modelLabel: string
  onEffortChange: (effort: string | null) => void
  onModelChange: (model: string | null) => void
  onTierChange: (tier: ChatModelTier | null) => void
  options: ChatModelOption[]
  optionsStatus: 'available' | 'missing' | 'unresolved'
  selectedEffort: string | null
  tier: ChatModelTier | null
}) {
  const { t } = useLocale()
  const catalogMode = catalog.length > 0
  const selectedCatalogEntry = catalog.find((entry) => entry.model_id === model) ?? null
  const effortLevels = selectedCatalogEntry?.card?.reasoning_levels ?? []
  const defaultModelLabel = defaultModel?.model
    ? modelDisplayName(defaultModel.model, catalog, modelNameLabel(defaultModel, t.chat.modelUnknown))
    : t.chat.modelServerDefaultDescription
  const modelCandidates = catalogMode
    ? catalog.map((entry) => ({
        active: model === entry.model_id,
        key: `model:${entry.model_id}`,
        label: entry.card?.display_name ?? entry.model_id,
        onSelect: () => onModelChange(entry.model_id),
      }))
    : optionsStatus === 'available'
      ? chatModelTierOrder.flatMap((candidateTier) => {
          const option = options.find((item) => item.tier === candidateTier)
          if (!option?.model) return []
          return [{
            active: model == null && tier === candidateTier,
            key: `tier:${candidateTier}`,
            label: modelDisplayName(option.model, catalog, modelNameLabel(option, t.chat.modelUnknown)),
            onSelect: () => onTierChange(candidateTier),
          }]
        })
      : []

  return (
    <>
      <AssistantMenuHeader primary={t.chat.retryModelHeader} secondary={modelLabel} />
      <div className="min-h-0 overflow-x-hidden overflow-y-auto p-1">
        <RetryModelRow
          active={model == null && tier == null}
          label={t.chat.modelServerDefault}
          secondaryLabel={defaultModelLabel}
          onSelect={() => {
            onModelChange(null)
            onTierChange(null)
            onEffortChange(null)
          }}
        />
        {modelCandidates.length > 0 ? (
          <>
            <DropdownMenuSeparator className="mx-0 my-1" />
            {modelCandidates.map((candidate) => (
              <RetryModelRow
                active={candidate.active}
                key={candidate.key}
                label={candidate.label}
                onSelect={candidate.onSelect}
              />
            ))}
          </>
        ) : (
          <DropdownMenuItem disabled className="w-full min-w-0 rounded-none px-2.5 py-2">
            <span className="truncate text-muted-foreground">
              {optionsStatus === 'unresolved' ? t.chat.modelMetadataMissing : t.chat.modelDiscoveryMissing}
            </span>
          </DropdownMenuItem>
        )}
      </div>
      {effortLevels.length > 0 && (
        <div className="border-t border-border bg-surface/40 px-2.5 py-1.5">
          <div className="mb-1 t-caption text-muted-foreground/65">{t.chat.modelReasoningLabel}</div>
          <div className="flex flex-wrap gap-1">
            <RetryEffortButton
              active={selectedEffort == null}
              label={t.chat.modelEffortDefault}
              onSelect={() => onEffortChange(null)}
            />
            {effortLevels.map((effort) => (
              <RetryEffortButton
                active={selectedEffort === effort}
                key={effort}
                label={effort}
                onSelect={() => onEffortChange(effort)}
              />
            ))}
          </div>
        </div>
      )}
    </>
  )
}

function RetryModelRow({
  active,
  label,
  onSelect,
  secondaryLabel,
}: {
  active: boolean
  label: string
  onSelect: () => void
  secondaryLabel?: string
}) {
  return (
    <DropdownMenuItem
      className={cn(
        'group relative flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5',
        active && 'bg-accent',
      )}
      onSelect={(event) => {
        event.preventDefault()
        onSelect()
      }}
    >
      <BrainCircuit className={cn('icon-sm shrink-0', active ? 'text-brand' : 'text-foreground/80')} strokeWidth={2.35} />
      <span className="flex min-w-0 flex-1 items-baseline gap-2">
        <span className={cn('min-w-0 truncate t-list-regular text-foreground', active && 'font-semibold')}>
          {label}
        </span>
        {secondaryLabel ? (
          <span className="min-w-0 shrink truncate t-meta-sm text-muted-foreground">
            {secondaryLabel}
          </span>
        ) : null}
      </span>
      <span className="flex size-4 shrink-0 items-center justify-center">
        {active ? <Check className="size-3.5 text-brand" /> : null}
      </span>
    </DropdownMenuItem>
  )
}

function RetryEffortButton({
  active,
  label,
  onSelect,
}: {
  active: boolean
  label: string
  onSelect: () => void
}) {
  return (
    <button
      className={cn(
        'h-7 rounded-md px-2 text-xs font-medium transition-colors',
        active
          ? 'bg-background text-foreground shadow-[0_1px_2px_var(--shadow-hairline)]'
          : 'text-muted-foreground hover:bg-accent/70 hover:text-foreground',
      )}
      onClick={(event) => {
        event.preventDefault()
        onSelect()
      }}
      type="button"
    >
      {label}
    </button>
  )
}

function MessageActionButton({
  canCopy = true,
  copied = false,
  icon,
  label,
  onClick,
}: {
  canCopy?: boolean
  copied?: boolean
  icon: 'answer' | 'branch' | 'copy' | 'edit'
  label: string
  onClick: () => void
}) {
  const Icon = copied
    ? Check
    : icon === 'edit'
      ? PencilLine
      : icon === 'answer'
        ? SendHorizontal
      : icon === 'branch'
        ? MessageSquarePlus
        : Copy
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          className={cn(
            'size-6 text-foreground/60 transition-colors hover:text-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-0',
            copied && 'text-success hover:text-success',
          )}
          disabled={!canCopy}
          onClick={(event) => {
            event.stopPropagation()
            onClick()
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Icon className="size-3" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function ChatChainTrace({ steps }: { steps: ChatChainStepRecord[] }) {
  const { t } = useLocale()
  const [expanded, setExpanded] = useState(false)
  const [openStep, setOpenStep] = useState<number | null>(null)
  return (
    <div className="mb-2 max-w-4xl rounded-md border border-border/70 bg-surface/60">
      <button
        aria-expanded={expanded}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left t-meta-sm font-semibold text-muted-foreground transition hover:text-foreground"
        onClick={() => setExpanded((value) => !value)}
        type="button"
      >
        <ListOrdered className="size-3.5 shrink-0" />
        <span className="min-w-0 flex-1 truncate">
          {t.chat.chainIntermediate.replace('{count}', String(steps.length))}
        </span>
        <ChevronDown className={cn('size-3.5 shrink-0 transition-transform', expanded && 'rotate-180')} />
      </button>
      {expanded && (
        <div className="border-t border-border/70 px-2.5 py-1">
          {steps.map((step, index) => {
            const open = openStep === index
            const label = t.chat.chainStepLabel
              .replace('{n}', String(index + 1))
              .replace('{total}', String(steps.length))
              .replace('{label}', step.label)
            return (
              <div className="border-b border-border/40 py-1 last:border-0" key={index}>
                <button
                  className={cn(
                    'flex w-full items-center gap-1.5 text-left t-meta-sm font-medium transition',
                    step.status === 'error'
                      ? 'text-destructive'
                      : step.status === 'stopped'
                        ? 'text-warning'
                        : 'text-muted-foreground hover:text-foreground',
                  )}
                  onClick={() => setOpenStep(open ? null : index)}
                  type="button"
                >
                  <ChevronRight className={cn('size-3 shrink-0 transition-transform', open && 'rotate-90')} />
                  <span className="min-w-0 flex-1 truncate">{label}</span>
                </button>
                {open && (
                  <div className="chat-markdown mt-1 max-w-full pl-4 text-xs leading-snug text-foreground/90">
                    <MarkdownRenderer markdown={step.output} variant="chat" />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function GeneratingPlaceholder({ reduceMotion }: { reduceMotion: boolean | null }) {
  const { t } = useLocale()
  return (
    <div className="inline-flex items-center gap-2.5 py-1.5 text-xs text-muted-foreground">
      <span
        className={cn(
          'inqtrix-thinking-mark',
          reduceMotion && 'inqtrix-thinking-mark-static',
        )}
        aria-hidden="true"
      >
        <span className="inqtrix-thinking-node" />
        <span className="inqtrix-thinking-node" />
        <span className="inqtrix-thinking-node" />
        <span className="inqtrix-thinking-node" />
      </span>
      <span className="t-caption tracking-[0.14em] text-muted-foreground/80">
        {t.chat.thinking}
      </span>
    </div>
  )
}

function ChatMessageAttachments({
  align = 'start',
  attachments,
}: {
  align?: 'end' | 'start'
  attachments: ChatMessageAttachmentRecord[] | undefined
}) {
  const chips = chatAttachmentChipsFromAttachments(attachments ?? [])
  if (chips.length === 0) return null

  return (
    <div className={cn(
      'mb-2 flex min-w-0 flex-wrap gap-1.5',
      align === 'end' ? 'justify-end' : 'justify-start',
    )}>
      {chips.map((chip) => {
        const { chipClassName, icon: Icon } = attachmentChipVisual(chip.kind)
        return (
          <span
            className={cn(
              'inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-semibold',
              chipClassName,
            )}
            key={chatContextRefKey(chip.ref)}
            title={chip.title}
          >
            <Icon className="size-3.5 shrink-0" />
            <span className="min-w-0 truncate">{chip.label}</span>
            {chip.fileCount !== null && (
              <span className="shrink-0 t-hint font-bold tabular-nums opacity-75">{chip.fileCount}</span>
            )}
          </span>
        )
      })}
    </div>
  )
}

function messageFromUnknown(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}
