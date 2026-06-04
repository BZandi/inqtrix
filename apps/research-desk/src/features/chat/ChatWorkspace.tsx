import {
  AlertTriangle,
  BookOpen,
  Bot,
  Check,
  ChevronDown,
  ChevronRight,
  CircleUserRound,
  Copy,
  Eraser,
  EyeOff,
  FileText,
  Library,
  ListChecks,
  ListOrdered,
  MessageSquareText,
  MessageSquarePlus,
  MessagesSquare,
  Paperclip,
  PencilLine,
  Plus,
  Save,
  SendHorizontal,
  SlidersHorizontal,
  Square,
  Trash2,
  X,
} from '@/components/icons'
import { AnimatePresence, motion } from 'motion/react'
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type RefObject,
} from 'react'
import { MarkdownRenderer } from '@/components/markdown/MarkdownRenderer'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
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
  ChatRuleRecord,
} from '@/features/project/types'
import type {
  ChatModelOption,
  ChatModelTier,
  NodeModelResolution,
} from '@/features/researchRuns/types'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'
import {
  TextImproveButton,
  TextImproveFloatingLayer,
  useTextImprovement,
  type TextImprovementApiOptions,
} from '@/features/textImprove'
import { PanelRail } from '@/components/ui/panel-rail'
import { ComposerIconButton, composerIconButtonClassName } from '@/features/composer/ComposerIconButton'
import { ChatHistoryPanel } from './history/ChatHistoryPanel'
import { RuleLibraryDialog } from './rules/RuleLibraryDialog'
import type { ChatMessage, ChatThread } from './types'
import { ContextChipLegend } from '@/features/composer/ContextChipLegend'
import { MentionComposer, type MentionComposerHandle } from '@/features/composer/MentionComposer'
import { type LabelResolver } from '@/features/composer/mentionDoc'
import { resizeTextareaToRows } from '@/features/composer/textareaAutosize'

type ChatWorkspaceProps = {
  activeAssistantMessageId: string | null
  chatModelOptions: ChatModelOption[]
  chatModelOptionsStatus: 'available' | 'missing' | 'unresolved'
  chatHistorySections: ChatHistorySection[]
  chatRules: ChatRuleRecord[]
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
  onAnswerLastUserMessage: (threadId: string, messageId: string) => void
  onBranchFromMessage: (threadId: string, messageId: string) => void
  onClearThread: () => void
  onCreateThread: (groupId?: string | null) => void
  onCreateThreadGroup: () => void
  onDeleteMessages: (threadId: string, messageIds: string[]) => void
  onDeleteRule: (ruleId: string) => void
  onDeleteThreadGroup: (groupId: string) => void
  onDeleteThread: (threadId: string) => void
  onEditMessage: (threadId: string, messageId: string, contentMarkdown: string) => void
  chainingEnabled: boolean
  onChainingEnabledChange: (enabled: boolean) => void
  onIncognitoChange: (enabled: boolean) => void
  onHistoryVisibleChange: (isVisible: boolean) => void
  onMoveThreadGroup: (groupId: string, targetIndex: number) => void
  onMoveThreadToGroup: (threadId: string, groupId: string | null, targetIndex: number) => void
  onRenameThread: (threadId: string, title: string) => void
  onRenameThreadGroup: (groupId: string, title: string) => void
  onRemoveContext: (ref: ChatContextReferenceRecord) => void
  onReorderContext: (fromIndex: number, toIndex: number) => void
  pendingReorderKeys: string[]
  pillKeys: string[]
  onSaveRule: (rule: ChatRuleRecord) => void
  onSendMessage: (
    contentMarkdown: string,
    refs?: ChatContextReferenceRecord[],
    options?: ChatSendOptions,
  ) => void
  onSelectThread: (threadId: string) => void
  onSelectedModelTierChange: (tier: ChatModelTier | null) => void
  onStopGenerating: () => void
  onStreamingEnabledChange: (enabled: boolean) => void
  attachmentBudgetNotice: string | null
  pendingChips: ChatAttachmentChipModel[]
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
}

export default function ChatWorkspace({
  activeAssistantMessageId,
  chatModelOptions,
  chatModelOptionsStatus,
  chatHistorySections,
  chatRules,
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
  onDeleteRule,
  onDeleteThreadGroup,
  onDeleteThread,
  onEditMessage,
  chainingEnabled,
  onChainingEnabledChange,
  onIncognitoChange,
  onHistoryVisibleChange,
  onMoveThreadGroup,
  onMoveThreadToGroup,
  onRenameThread,
  onRenameThreadGroup,
  onRemoveContext,
  onReorderContext,
  pendingReorderKeys,
  pillKeys,
  onSaveRule,
  onSendMessage,
  onSelectThread,
  onSelectedModelTierChange,
  onStopGenerating,
  onStreamingEnabledChange,
  onAttachFiles,
  onPillRefsChange,
  fileGroupOptions,
  fileOptions,
  attachmentBudgetNotice,
  pendingChips,
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
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null)
  const [isEditingTitle, setIsEditingTitle] = useState(false)
  const [isMessageSelectionMode, setIsMessageSelectionMode] = useState(false)
  const [isRuleLibraryOpen, setIsRuleLibraryOpen] = useState(false)
  const [messageEditDraft, setMessageEditDraft] = useState('')
  const [pillRefs, setPillRefs] = useState<ChatContextReferenceRecord[]>([])
  const [selectedMessageIds, setSelectedMessageIds] = useState<ReadonlySet<string>>(() => new Set())
  const [titleDraft, setTitleDraft] = useState('')
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const composerRef = useRef<MentionComposerHandle | null>(null)
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
    sendDraft()
  }

  function handleComposerChange() {
    setDraft(composerRef.current?.getMentionText() ?? '')
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
    setDraft(composerRef.current?.getMentionText() ?? text)
    draftTextImprove.clearProposal()
    setDraftCommitPulseKey((key) => key + 1)
    window.requestAnimationFrame(() => composerRef.current?.focus())
  }

  function sendDraft() {
    const instruction = composerRef.current?.getInstructionText().trim() ?? ''
    if (!instruction || isSending) return
    shouldAutoFollowChatRef.current = true
    onSendMessage(
      instruction,
      pillRefs,
      selectedModelTier ? { modelTier: selectedModelTier } : undefined,
    )
    composerRef.current?.clear()
    setDraft('')
    setPillRefs([])
    setComposerNotice(null)
  }

  function handleComposerRefsChange(refs: ChatContextReferenceRecord[]) {
    setPillRefs(refs)
    onPillRefsChange(refs)
  }

  function handleRemoveChip(ref: ChatContextReferenceRecord) {
    if (pillRefs.some((pill) => chatContextRefKey(pill) === chatContextRefKey(ref))) {
      composerRef.current?.removeRef(ref)
    } else {
      onRemoveContext(ref)
    }
  }

  const historyPanel = (
    <ChatHistoryPanel
      chatHistorySections={chatHistorySections}
      isIncognito={isIncognito}
      locale={locale}
      onCreateThread={onCreateThread}
      onCreateThreadGroup={onCreateThreadGroup}
      onDeleteThread={onDeleteThread}
      onDeleteThreadGroup={onDeleteThreadGroup}
      onHide={isDesktop ? () => onHistoryVisibleChange(false) : undefined}
      onMoveThreadGroup={onMoveThreadGroup}
      onMoveThreadToGroup={onMoveThreadToGroup}
      onRenameThread={onRenameThread}
      onRenameThreadGroup={onRenameThreadGroup}
      onSelectThread={onSelectThread}
      reduceMotion={reduceMotion}
      runningThreadIds={runningThreadIds}
      selectedThreadId={selectedThread?.id ?? null}
      threads={threads}
    />
  )

  const conversationPanel = (
        <section className="flex min-h-[620px] min-w-0 flex-col bg-background lg:h-full lg:min-h-0 lg:overflow-hidden">
	          <div className="z-10 flex h-12 shrink-0 items-center justify-between gap-2 border-b border-border bg-background px-4 md:px-6">
            <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
              <MessageSquareText className="size-4 shrink-0 text-foreground/80" />
              <div className="min-w-0 flex-1 overflow-hidden">
                <div className="flex min-w-0 items-center gap-2 overflow-hidden">
                  {isEditingTitle && selectedThread ? (
                    <input
                      aria-label={t.chat.renameTitle}
                      className="min-w-0 flex-1 rounded-sm border-0 bg-transparent px-0 text-sm font-semibold text-foreground outline-none focus-visible:ring-0"
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
                      className="min-w-0 flex-1 truncate rounded-sm text-left text-sm font-semibold text-foreground hover:text-brand focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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
                <p
                  className="max-w-md truncate text-[11px] leading-4 text-muted-foreground"
                  title={selectedThread ? selectedThread.preview : undefined}
                >
                  {selectedThread ? selectedThread.preview : t.chat.empty}
                </p>
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

	          <ScrollArea className="min-h-0 flex-1" ref={messagesScrollAreaRef}>
	            <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col gap-5 px-4 py-6 md:px-8">
              {selectedThread && selectedThread.messages.length > 0 ? (
                selectedThread.messages.map((message) => (
                  <ChatMessageBubble
                    canAnswerLastUserMessage={canAnswerLastUserMessage && message.id === lastMessage?.id}
                    canBranch={!isIncognito && !isSending}
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
                    onToggleSelected={toggleSelectedMessage}
                    reduceMotion={reduceMotion}
                  />
                ))
              ) : selectedThread ? (
                <EmptyChatState
                  label={pendingChips.length > 0 ? t.chat.emptyWithContext : t.chat.empty}
                  type="thread"
                />
              ) : (
                <EmptyChatState label={t.chat.empty} type="all" />
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
              <div className="relative overflow-visible rounded-xl border border-border bg-card px-2.5 py-2 shadow-[0_8px_28px_-12px_var(--shadow-soft)]">
                {attachmentBudgetNotice && (
                  <div className="mb-2 flex items-center gap-1.5 rounded-md border border-warning/30 bg-warning/10 px-2 py-1 text-[11px] font-medium text-warning">
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
                    placeholder={t.chat.placeholder}
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
                      className="w-72 max-w-[calc(100vw-2rem)]"
                      side="top"
                      sideOffset={8}
                    >
                      <DropdownMenuLabel className="text-xs text-muted-foreground">
                        {t.chat.research}
                      </DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      {reportOptions.length > 0 ? (
                        reportOptions.map((report) => (
                          <DropdownMenuItem
                            className="w-full min-w-0 items-start gap-2 py-2"
                            key={report.runId}
                            onSelect={() => onAttachContext({ kind: 'research-report', runId: report.runId })}
                          >
                            <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                            <span className="min-w-0 flex-1">
                              <span className="block max-w-full truncate text-sm font-medium">
                                @research:{report.label}
                              </span>
                              <span className="block max-w-full truncate text-xs text-muted-foreground">
                                {report.title}
                              </span>
                            </span>
                          </DropdownMenuItem>
                        ))
                      ) : (
                        <DropdownMenuItem disabled>
                          {t.chat.noReports}
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuSeparator />
                      <DropdownMenuLabel className="text-xs text-muted-foreground">
                        {t.chat.rules}
                      </DropdownMenuLabel>
                      {ruleOptions.length > 0 ? (
                        ruleOptions.map((rule) => (
                          <DropdownMenuItem
                            className="w-full min-w-0 items-start gap-2 py-2"
                            key={rule.ruleId}
                            onSelect={() => onAttachContext({ kind: 'chat-rule', ruleId: rule.ruleId })}
                          >
                            <BookOpen className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                            <span className="min-w-0 flex-1">
                              <span className="block max-w-full truncate text-sm font-medium">
                                @rules:{rule.label}
                              </span>
                              <span className="block max-w-full truncate text-xs text-muted-foreground">
                                {rule.title}
                              </span>
                            </span>
                          </DropdownMenuItem>
                        ))
                      ) : (
                        <DropdownMenuItem disabled>
                          {t.chat.noRules}
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="gap-2"
                        onSelect={() => setIsRuleLibraryOpen(true)}
                      >
                        <Library className="size-4 text-muted-foreground" />
                        {t.chat.manageRules}
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
                  <ComposerIconButton
                    className="shrink-0"
                    icon={Library}
                    label={t.chat.manageRules}
                    onClick={() => setIsRuleLibraryOpen(true)}
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
                    <DropdownMenuContent align="start" className="w-64" side="top" sideOffset={8}>
                      <DropdownMenuLabel>{t.composer.moreSettings}</DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      <DropdownMenuCheckboxItem
                        checked={streamingEnabled}
                        className="gap-3 py-2 pl-2 pr-2 [&>span:first-child]:hidden"
                        disabled={isSending}
                        onCheckedChange={onStreamingEnabledChange}
                        onSelect={(event) => event.preventDefault()}
                      >
                        <span className="grid min-w-0 flex-1 text-left leading-tight">
                          <span className="truncate text-sm font-medium">{t.chat.streaming}</span>
                          <span className="truncate text-xs text-muted-foreground">
                            {streamingEnabled ? t.chat.streamingOn : t.chat.streamingOff}
                          </span>
                        </span>
                        <ChatToggleVisual checked={streamingEnabled} />
                      </DropdownMenuCheckboxItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <ChatModelPicker
                    defaultModel={defaultChatModel}
                    disabled={isSending}
                    onChange={onSelectedModelTierChange}
                    options={chatModelOptions}
                    optionsStatus={chatModelOptionsStatus}
                    selectedTier={selectedModelTier}
                  />
                  </div>
                  <div className="shrink-0">
                    {isSending ? (
                      <Button
                        aria-label={t.chat.stopGenerating}
                        className="size-7 rounded-md text-muted-foreground hover:bg-accent/70 hover:text-destructive focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-0"
                        onClick={onStopGenerating}
                        size="icon"
                        type="button"
                        variant="ghost"
                      >
                        <Square className="size-4 fill-current" />
                      </Button>
                    ) : (
                      <Button
                        aria-label={t.chat.send}
                        className={cn(
                          'size-7 rounded-md focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-0',
                          canSend
                            ? 'bg-brand text-white hover:bg-brand/90 hover:text-white'
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
                    <span className="min-w-0 truncate text-[11px] font-medium text-destructive">
                      {requestError}
                    </span>
                  )}
                  {!requestError && requestNotice && (
                    <span className="min-w-0 truncate text-[11px] font-medium text-warning">
                      {requestNotice}
                    </span>
                  )}
                  {!requestError && !requestNotice && composerNotice && (
                    <span className="min-w-0 truncate text-[11px] font-medium text-warning">
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
              className="min-h-0 min-w-0 overflow-hidden border-r border-border bg-surface/60"
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
            <PanelRail
              label={t.chat.showHistory}
              onExpand={() => onHistoryVisibleChange(true)}
              side="left"
            />
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
      <RuleLibraryDialog
        isOpen={isRuleLibraryOpen}
        onClose={() => setIsRuleLibraryOpen(false)}
        onDeleteRule={onDeleteRule}
        onSaveRule={onSaveRule}
        reduceMotion={reduceMotion}
        rules={chatRules}
        textImprovement={textImprovement}
      />
    </div>
  )
}

function ChatToggleVisual({ checked }: { checked: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        'inline-flex h-5 w-9 shrink-0 items-center rounded-full border-2 border-transparent shadow-sm transition-colors',
        checked ? 'bg-primary' : 'bg-input',
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

const chatModelTierOrder: ChatModelTier[] = ['high', 'mid', 'fast']

function ChatModelPicker({
  defaultModel,
  disabled,
  onChange,
  options,
  optionsStatus,
  selectedTier,
}: {
  defaultModel: NodeModelResolution | null
  disabled: boolean
  onChange: (tier: ChatModelTier | null) => void
  options: ChatModelOption[]
  optionsStatus: 'available' | 'missing' | 'unresolved'
  selectedTier: ChatModelTier | null
}) {
  const { t } = useLocale()
  const selectedOption = selectedTier ? chatModelOptionForTier(options, selectedTier) : null
  const activeModel = selectedOption ?? defaultModel ?? chatModelOptionForTier(options, 'mid') ?? null
  const unavailableLabel = optionsStatus === 'unresolved'
    ? t.chat.modelMetadataMissing
    : t.chat.modelDiscoveryMissing
  const activeLabel = selectedTier && optionsStatus !== 'available'
    ? `${tierLabel(selectedTier, t)} · ${unavailableLabel}`
    : `${modelNameLabel(activeModel, t.chat.modelUnknown)} · ${effortLabel(activeModel, t)}`
  const pickerValue = selectedTier ?? 'default'

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={t.chat.modelPicker}
          className={cn(
            'h-7 min-w-0 max-w-[min(48vw,17rem)] shrink rounded-md px-1.5 text-[11px] font-semibold text-muted-foreground hover:bg-accent/70 hover:text-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-0',
            'data-[state=open]:bg-accent data-[state=open]:text-foreground',
          )}
          disabled={disabled}
          type="button"
          variant="ghost"
        >
          <span className="min-w-0 truncate">{activeLabel}</span>
          <ChevronDown className="size-3 shrink-0 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="w-80 max-w-[calc(100vw-2rem)]"
        side="top"
        sideOffset={8}
      >
        <DropdownMenuLabel className="text-xs text-muted-foreground">
          {t.chat.modelPicker}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup
          onValueChange={(value) => onChange(isChatModelTier(value) ? value : null)}
          value={pickerValue}
        >
          <DropdownMenuRadioItem className="items-start py-2 pr-3" value="default">
            <span className="grid min-w-0 flex-1 text-left leading-tight">
              <span className="truncate text-sm font-medium">{t.chat.modelServerDefault}</span>
              <span className="truncate text-xs text-muted-foreground">
                {modelDetailLabel(defaultModel, t)}
              </span>
            </span>
          </DropdownMenuRadioItem>
          <DropdownMenuSeparator />
          {optionsStatus === 'available' ? chatModelTierOrder.map((tier) => {
            const option = chatModelOptionForTier(options, tier)
            return (
              <DropdownMenuRadioItem
                className="items-start py-2 pr-3"
                key={tier}
                value={tier}
              >
                <span className="grid min-w-0 flex-1 text-left leading-tight">
                  <span className="flex min-w-0 items-baseline gap-2">
                    <span className="shrink-0 text-sm font-medium">
                      {tierLabel(tier, t)}
                    </span>
                    <span className="min-w-0 truncate text-xs font-medium text-muted-foreground">
                      {modelNameLabel(option, t.chat.modelUnknown)}
                    </span>
                  </span>
                  <span className="truncate text-xs text-muted-foreground">
                    {effortLabel(option, t)}
                  </span>
                </span>
              </DropdownMenuRadioItem>
            )
          }) : (
            <DropdownMenuItem disabled className="items-start py-2">
              <span className="grid min-w-0 flex-1 text-left leading-tight">
                <span className="truncate text-sm font-medium">{unavailableLabel}</span>
                <span className="truncate text-xs text-muted-foreground">
                  {t.chat.modelServerDefault}
                </span>
              </span>
            </DropdownMenuItem>
          )}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function chatModelOptionForTier(
  options: readonly ChatModelOption[],
  tier: ChatModelTier,
): ChatModelOption | null {
  return options.find((option) => option.tier === tier) ?? null
}

function isChatModelTier(value: string): value is ChatModelTier {
  return value === 'high' || value === 'mid' || value === 'fast'
}

function modelNameLabel(
  option: Pick<NodeModelResolution, 'model'> | null | undefined,
  fallback: string,
) {
  const model = option?.model?.trim()
  if (!model) return fallback
  return model.replace(/^.+\//, '')
}

function modelDetailLabel(
  option: NodeModelResolution | null,
  t: ReturnType<typeof useLocale>['t'],
) {
  return `${modelNameLabel(option, t.chat.modelUnknown)} · ${effortLabel(option, t)}`
}

function effortLabel(
  option: Pick<NodeModelResolution, 'effort'> | null | undefined,
  t: ReturnType<typeof useLocale>['t'],
) {
  const effort = option?.effort?.trim().toLowerCase()
  if (!effort) return t.chat.modelEffortDefault
  if (effort === 'none') return t.chat.modelThinkingOff
  return `${t.chat.modelThinkingOn} ${shortEffort(effort)}`
}

function shortEffort(effort: string) {
  if (effort === 'medium') return 'med'
  if (effort === 'minimal') return 'min'
  return effort
}

function tierLabel(tier: ChatModelTier, t: ReturnType<typeof useLocale>['t']) {
  if (tier === 'high') return t.chat.modelTierHigh
  if (tier === 'fast') return t.chat.modelTierFast
  return t.chat.modelTierMid
}

function ChatMessageModelChip({
  modelResolution,
}: {
  modelResolution: ChatMessage['modelResolution']
}) {
  const { t } = useLocale()
  if (!modelResolution?.model) return null
  const label = `${modelNameLabel(modelResolution, t.chat.modelUnknown)} · ${effortLabelFromToken(modelResolution.effort, t)}`
  return (
    <span
      className="max-w-[min(42vw,16rem)] truncate rounded-md bg-muted/60 px-1.5 py-0.5 text-[10px] font-semibold text-muted-foreground"
      title={label}
    >
      {label}
    </span>
  )
}

function effortLabelFromToken(
  effortToken: string | undefined,
  t: ReturnType<typeof useLocale>['t'],
) {
  const effort = effortToken?.trim().toLowerCase()
  if (!effort) return t.chat.modelEffortDefault
  if (effort === 'none') return t.chat.modelThinkingOff
  return `${t.chat.modelThinkingOn} ${shortEffort(effort)}`
}

function EmptyChatState({
  label,
  type,
}: {
  label: string
  type: 'all' | 'thread'
}) {
  const Icon = type === 'thread' ? MessageSquareText : MessagesSquare
  return (
    <div className="flex flex-1 items-center justify-center p-8 text-center">
      <div className="max-w-sm">
        <div className="mx-auto flex size-14 items-center justify-center rounded-full border border-border bg-surface text-muted-foreground">
          <Icon className="size-7" />
        </div>
        <p className="mt-4 text-sm font-medium text-muted-foreground">
          {label}
        </p>
      </div>
    </div>
  )
}

function ChatMessageBubble({
  canAnswerLastUserMessage,
  canBranch,
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
  onToggleSelected,
  reduceMotion,
}: {
  canAnswerLastUserMessage: boolean
  canBranch: boolean
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
  onToggleSelected: (messageId: string) => void
  reduceMotion: boolean | null
}) {
  const { t } = useLocale()
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'
  const isEditing = editingMessageId === message.id
  const Icon = isUser ? CircleUserRound : Bot
  const canCopy = message.contentMarkdown.trim().length > 0
  const canContinueFromHere = !isUser && !isSelectionMode && canBranch && canCopy && !isStreaming
  const canGenerateAnswer = isUser && !isSelectionMode && canAnswerLastUserMessage && canCopy && !isStreaming && !isEditing
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
            <div className="mb-1 flex items-center gap-2 text-[11px] font-semibold text-muted-foreground">
              <span>{t.chat.assistant}</span>
              <ChatMessageModelChip modelResolution={message.modelResolution} />
              <span>{formatTime(message.createdAt)}</span>
              {isSelectionMode && (
                <MessageSelectionPill isSelected={isSelected} label={selectionLabel} />
              )}
              {!isSelectionMode && (
                <MessageActionButton
                  canCopy={canCopy}
                  copied={copied}
                  icon="copy"
                  label={copied ? t.chat.copiedMessage : t.chat.copyMessage}
                  onClick={() => void copyMessage()}
                />
              )}
              {canContinueFromHere && (
                <MessageActionButton
                  icon="branch"
                  label={t.chat.continueFromHere}
                  onClick={() => onBranch(message.id)}
                />
              )}
            </div>
            {message.chainTrace && message.chainTrace.length > 0 && (
              <ChatChainTrace steps={message.chainTrace} />
            )}
            {message.contentMarkdown ? (
              <div
                className={cn(
                  'chat-markdown max-w-4xl text-sm leading-[1.42] text-foreground',
                  isStreaming && !reduceMotion && 'animate-in fade-in-0 duration-200',
                )}
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
              </div>
            ) : (
              !(message.chainTrace && message.chainTrace.length > 0) && (
                <GeneratingPlaceholder reduceMotion={reduceMotion} />
              )
            )}
            <ChatMessageAttachments attachments={message.attachments} />
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
              <div className="mb-1 flex items-center justify-end gap-2 text-[11px] font-semibold text-muted-foreground">
                {!isSelectionMode && (
                  <>
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
                  </>
                )}
                {isSelectionMode && (
                  <MessageSelectionPill isSelected={isSelected} label={selectionLabel} />
                )}
                <span>{t.chat.you}</span>
                <span>{formatTime(message.createdAt)}</span>
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
                <div className="rounded-lg border border-brand/25 bg-brand px-3 py-2.5 text-sm leading-6 text-primary-foreground shadow-[0_1px_2px_var(--shadow-hairline)]">
                  {message.contentMarkdown}
                </div>
              )}
            </div>
          </div>
        </div>
        <span className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-md border border-brand/20 bg-brand-subtle text-brand">
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
        'rounded-full border border-border bg-background px-1.5 py-0.5 text-[10px] font-semibold text-muted-foreground',
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
            'size-6 text-foreground/65 opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover/message:opacity-100',
            copied && 'text-success opacity-100 hover:text-success',
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
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-[11px] font-semibold text-muted-foreground transition hover:text-foreground"
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
                    'flex w-full items-center gap-1.5 text-left text-[11px] font-medium transition',
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
                  <div className="chat-markdown mt-1 max-w-full pl-4 text-xs leading-[1.4] text-foreground/90">
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
      <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/80">
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
              <span className="shrink-0 text-[10px] font-bold tabular-nums opacity-75">{chip.fileCount}</span>
            )}
          </span>
        )
      })}
    </div>
  )
}

function formatTime(iso: string) {
  return new Intl.DateTimeFormat('de-DE', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(iso))
}

function messageFromUnknown(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}
