import {
  ChevronDown,
  ChevronRight,
  Folder,
  FolderPlus,
  FolderOpen,
  GripVertical,
  LoaderCircle,
  MessagesSquare,
  PanelLeftClose,
  SquarePen,
  Trash2,
} from '@/components/icons'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  displayRelativeDate,
  type ChatHistorySection,
} from '@/features/project/selectors'
import { useLocale } from '@/i18n/LocaleProvider'
import type { Locale, TranslationDictionary } from '@/i18n/translations'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'
import { AnimatePresence, motion } from 'motion/react'
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from 'react'
import type { ChatThread } from '../types'

type ChatThreadDropTarget = {
  groupId: string | null
  targetIndex: number
}

type ChatHistoryPanelProps = {
  chatHistorySections: ChatHistorySection[]
  isIncognito: boolean
  locale: Locale
  onCreateThread: (groupId?: string | null) => void
  onCreateThreadGroup: () => void
  onDeleteThread: (threadId: string) => void
  onDeleteThreadGroup: (groupId: string) => void
  onHide?: () => void
  onMoveThreadGroup: (groupId: string, targetIndex: number) => void
  onMoveThreadToGroup: (threadId: string, groupId: string | null, targetIndex: number) => void
  onRenameThread: (threadId: string, title: string) => void
  onRenameThreadGroup: (groupId: string, title: string) => void
  onSelectThread: (threadId: string) => void
  /** Server has older thread pages not yet loaded (on-demand history). */
  hasMoreThreads?: boolean
  /** A load-older page request is in flight (disables the button + shows busy). */
  isLoadingMoreThreads?: boolean
  /** Load the next page of older threads. */
  onLoadMoreThreads?: () => void
  reduceMotion: boolean | null
  runningThreadIds: ReadonlySet<string>
  selectedThreadId: string | null
  threads: ChatThread[]
}

const UNGROUPED_CHAT_SECTION_ID = '__ungrouped__'

export function ChatHistoryPanel({
  chatHistorySections,
  isIncognito,
  locale,
  onCreateThread,
  onCreateThreadGroup,
  onDeleteThread,
  onDeleteThreadGroup,
  onHide,
  onMoveThreadGroup,
  onMoveThreadToGroup,
  onRenameThread,
  onRenameThreadGroup,
  onSelectThread,
  hasMoreThreads,
  isLoadingMoreThreads,
  onLoadMoreThreads,
  reduceMotion,
  runningThreadIds,
  selectedThreadId,
  threads,
}: ChatHistoryPanelProps) {
  const { t } = useLocale()
  const [collapsedGroupIds, setCollapsedGroupIds] = useState<ReadonlySet<string>>(() => new Set())
  const [draggedGroupId, setDraggedGroupId] = useState<string | null>(null)
  const [draggedThreadId, setDraggedThreadId] = useState<string | null>(null)
  const [editingGroupId, setEditingGroupId] = useState<string | null>(null)
  const [editingHistoryThreadId, setEditingHistoryThreadId] = useState<string | null>(null)
  const [groupDropTargetIndex, setGroupDropTargetIndex] = useState<number | null>(null)
  const [groupTitleDraft, setGroupTitleDraft] = useState('')
  const [historyThreadTitleDraft, setHistoryThreadTitleDraft] = useState('')
  const [threadDropTarget, setThreadDropTarget] = useState<ChatThreadDropTarget | null>(null)
  const groupTitleInputRef = useRef<HTMLInputElement | null>(null)
  const historyListRef = useRef<HTMLDivElement | null>(null)
  const historyThreadTitleInputRef = useRef<HTMLInputElement | null>(null)
  const skipGroupTitleCommitRef = useRef(false)
  const skipHistoryThreadTitleCommitRef = useRef(false)

  useLayoutEffect(() => {
    if (!editingGroupId) return
    groupTitleInputRef.current?.focus()
    groupTitleInputRef.current?.select()
  }, [editingGroupId])

  useLayoutEffect(() => {
    if (!editingHistoryThreadId) return
    historyThreadTitleInputRef.current?.focus()
    historyThreadTitleInputRef.current?.select()
  }, [editingHistoryThreadId])

  useEffect(() => {
    if (!editingHistoryThreadId) return
    if (threads.some((thread) => thread.id === editingHistoryThreadId)) return
    setEditingHistoryThreadId(null)
    setHistoryThreadTitleDraft('')
  }, [editingHistoryThreadId, threads])

  useEffect(() => {
    if (!editingGroupId) return
    if (chatHistorySections.some((section) => section.kind === 'group' && section.groupId === editingGroupId)) return
    setEditingGroupId(null)
    setGroupTitleDraft('')
  }, [chatHistorySections, editingGroupId])

  function toggleChatThreadGroup(groupId: string) {
    setCollapsedGroupIds((current) => {
      const next = new Set(current)
      if (next.has(groupId)) {
        next.delete(groupId)
      } else {
        next.add(groupId)
      }
      return next
    })
  }

  function startGroupTitleEdit(groupId: string, title: string) {
    skipGroupTitleCommitRef.current = false
    setEditingGroupId(groupId)
    setGroupTitleDraft(title)
  }

  function commitGroupTitleEdit() {
    if (!editingGroupId) return
    if (skipGroupTitleCommitRef.current) {
      skipGroupTitleCommitRef.current = false
      setEditingGroupId(null)
      setGroupTitleDraft('')
      return
    }
    const nextTitle = groupTitleDraft.trim()
    if (nextTitle) {
      onRenameThreadGroup(editingGroupId, nextTitle)
    }
    setEditingGroupId(null)
    setGroupTitleDraft('')
  }

  function cancelGroupTitleEdit() {
    skipGroupTitleCommitRef.current = true
    setEditingGroupId(null)
    setGroupTitleDraft('')
  }

  function startHistoryThreadTitleEdit(threadId: string, title: string) {
    skipHistoryThreadTitleCommitRef.current = false
    setEditingHistoryThreadId(threadId)
    setHistoryThreadTitleDraft(title)
  }

  function commitHistoryThreadTitleEdit() {
    if (!editingHistoryThreadId) return
    if (skipHistoryThreadTitleCommitRef.current) {
      skipHistoryThreadTitleCommitRef.current = false
      setEditingHistoryThreadId(null)
      setHistoryThreadTitleDraft('')
      return
    }
    const nextTitle = historyThreadTitleDraft.trim()
    if (nextTitle) {
      onRenameThread(editingHistoryThreadId, nextTitle)
    }
    setEditingHistoryThreadId(null)
    setHistoryThreadTitleDraft('')
  }

  function cancelHistoryThreadTitleEdit() {
    skipHistoryThreadTitleCommitRef.current = true
    setEditingHistoryThreadId(null)
    setHistoryThreadTitleDraft('')
  }

  function readGroupDropTarget(clientY: number, excludedGroupId = draggedGroupId) {
    const container = historyListRef.current
    if (!container) return null
    const groupElements = Array.from(container.querySelectorAll<HTMLElement>('[data-chat-history-draggable-group-id]'))
      .filter((groupElement) => (
        groupElement.dataset.chatHistoryDraggableGroupId !== excludedGroupId
        && groupElement.getBoundingClientRect().height > 0
      ))

    for (const [index, groupElement] of groupElements.entries()) {
      const rect = groupElement.getBoundingClientRect()
      if (clientY < rect.top + rect.height / 2) {
        return index
      }
    }

    return groupElements.length
  }

  function beginGroupDrag(event: ReactPointerEvent<HTMLButtonElement>, groupId: string) {
    if (event.button !== 0) return
    event.preventDefault()
    setDraggedGroupId(groupId)
    setGroupDropTargetIndex(readGroupDropTarget(event.clientY, groupId))

    function handlePointerMove(moveEvent: PointerEvent) {
      setGroupDropTargetIndex(readGroupDropTarget(moveEvent.clientY, groupId))
    }

    function finishPointerDrag(upEvent: PointerEvent) {
      const nextDropTarget = readGroupDropTarget(upEvent.clientY, groupId)
      cleanupPointerDrag()
      if (nextDropTarget === null) return
      onMoveThreadGroup(groupId, nextDropTarget)
    }

    function cancelPointerDrag() {
      cleanupPointerDrag()
    }

    function cleanupPointerDrag() {
      document.removeEventListener('pointermove', handlePointerMove)
      document.removeEventListener('pointerup', finishPointerDrag)
      document.removeEventListener('pointercancel', cancelPointerDrag)
      setDraggedGroupId(null)
      setGroupDropTargetIndex(null)
    }

    document.addEventListener('pointermove', handlePointerMove)
    document.addEventListener('pointerup', finishPointerDrag)
    document.addEventListener('pointercancel', cancelPointerDrag)
  }

  function readThreadDropTarget(clientY: number, excludedThreadId = draggedThreadId): ChatThreadDropTarget | null {
    const container = historyListRef.current
    if (!container) return null
    const sectionElements = Array.from(container.querySelectorAll<HTMLElement>('[data-chat-history-section]'))
      .filter((sectionElement) => sectionElement.getBoundingClientRect().height > 0)
    if (sectionElements.length === 0) return null

    const sectionElement = sectionElements.find((candidate) => {
      const rect = candidate.getBoundingClientRect()
      return clientY >= rect.top - 8 && clientY <= rect.bottom + 8
    }) ?? sectionElements.reduce((nearest, candidate) => {
      const nearestRect = nearest.getBoundingClientRect()
      const candidateRect = candidate.getBoundingClientRect()
      const nearestDistance = Math.min(
        Math.abs(clientY - nearestRect.top),
        Math.abs(clientY - nearestRect.bottom),
      )
      const candidateDistance = Math.min(
        Math.abs(clientY - candidateRect.top),
        Math.abs(clientY - candidateRect.bottom),
      )
      return candidateDistance < nearestDistance ? candidate : nearest
    })
    const groupKey = sectionElement.dataset.chatHistoryGroupId ?? UNGROUPED_CHAT_SECTION_ID
    const groupId = groupKey === UNGROUPED_CHAT_SECTION_ID ? null : groupKey
    const threadElements = Array.from(sectionElement.querySelectorAll<HTMLElement>('[data-chat-history-thread-id]'))
      .filter((threadElement) => (
        threadElement.dataset.chatHistoryThreadId !== excludedThreadId
        && threadElement.getBoundingClientRect().height > 0
      ))

    for (const [index, threadElement] of threadElements.entries()) {
      const rect = threadElement.getBoundingClientRect()
      if (clientY < rect.top + rect.height / 2) {
        return { groupId, targetIndex: index }
      }
    }

    return { groupId, targetIndex: threadElements.length }
  }

  function beginThreadDrag(event: ReactPointerEvent<HTMLButtonElement>, threadId: string) {
    if (event.button !== 0) return
    event.preventDefault()
    setDraggedThreadId(threadId)
    setThreadDropTarget(readThreadDropTarget(event.clientY, threadId))

    function handlePointerMove(moveEvent: PointerEvent) {
      setThreadDropTarget(readThreadDropTarget(moveEvent.clientY, threadId))
    }

    function finishPointerDrag(upEvent: PointerEvent) {
      const nextDropTarget = readThreadDropTarget(upEvent.clientY, threadId)
      cleanupPointerDrag()
      if (!nextDropTarget) return
      onMoveThreadToGroup(threadId, nextDropTarget.groupId, nextDropTarget.targetIndex)
    }

    function cancelPointerDrag() {
      cleanupPointerDrag()
    }

    function cleanupPointerDrag() {
      document.removeEventListener('pointermove', handlePointerMove)
      document.removeEventListener('pointerup', finishPointerDrag)
      document.removeEventListener('pointercancel', cancelPointerDrag)
      setDraggedThreadId(null)
      setThreadDropTarget(null)
    }

    document.addEventListener('pointermove', handlePointerMove)
    document.addEventListener('pointerup', finishPointerDrag)
    document.addEventListener('pointercancel', cancelPointerDrag)
  }

  const hasHistoryStructure = threads.length > 0 || chatHistorySections.some((section) => section.kind === 'group')
  const showUngroupedHistoryHeader = chatHistorySections.some((section) => section.kind === 'group')
  const groupSectionIds = chatHistorySections.flatMap((section) => (
    section.kind === 'group' ? [section.groupId] : []
  ))
  const dropIndicatorGroupSectionIds = draggedGroupId
    ? groupSectionIds.filter((groupId) => groupId !== draggedGroupId)
    : groupSectionIds

  return (
    <aside className="flex min-h-0 flex-col border-b border-border bg-surface/60 lg:h-full lg:border-b-0">
      <div className="flex h-12 items-center justify-between gap-2 border-b border-border px-3">
        <div className="flex min-w-0 items-center gap-2">
          <MessagesSquare className="size-4 shrink-0 text-foreground/80" />
          <h1 className="truncate t-section text-foreground">
            {t.chat.history}
          </h1>
        </div>
        <div className="flex items-center gap-1.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={t.chat.newGroup}
                className="size-7 shrink-0 rounded-md"
                onClick={onCreateThreadGroup}
                size="icon"
                type="button"
                variant="ghost"
              >
                <FolderPlus className="size-4 text-foreground/85" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.chat.newGroup}</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={t.chat.new}
                className="size-7 shrink-0 rounded-md"
                onClick={() => onCreateThread()}
                size="icon"
                type="button"
                variant="ghost"
              >
                <SquarePen className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.chat.new}</TooltipContent>
          </Tooltip>
          {onHide ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  aria-label={t.chat.hideHistory}
                  className="size-7 shrink-0 rounded-md"
                  onClick={onHide}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  <PanelLeftClose className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{t.chat.hideHistory}</TooltipContent>
            </Tooltip>
          ) : null}
        </div>
      </div>
      <ScrollArea className="max-h-64 min-h-0 lg:max-h-none lg:flex-1">
        <div className="space-y-1 p-2" ref={historyListRef}>
          {hasHistoryStructure ? (
            chatHistorySections.map((section) => (
              <ChatHistorySectionView
                beginGroupDrag={beginGroupDrag}
                beginThreadDrag={beginThreadDrag}
                cancelGroupTitleEdit={cancelGroupTitleEdit}
                cancelHistoryThreadTitleEdit={cancelHistoryThreadTitleEdit}
                collapsedGroupIds={collapsedGroupIds}
                commitGroupTitleEdit={commitGroupTitleEdit}
                commitHistoryThreadTitleEdit={commitHistoryThreadTitleEdit}
                draggedGroupId={draggedGroupId}
                draggedThreadId={draggedThreadId}
                editingGroupId={editingGroupId}
                editingHistoryThreadId={editingHistoryThreadId}
                groupDropTargetIndex={groupDropTargetIndex}
                groupIndex={section.kind === 'group' ? dropIndicatorGroupSectionIds.indexOf(section.groupId) : -1}
                groupSectionCount={dropIndicatorGroupSectionIds.length}
                groupTitleDraft={groupTitleDraft}
                groupTitleInputRef={groupTitleInputRef}
                historyThreadTitleDraft={historyThreadTitleDraft}
                historyThreadTitleInputRef={historyThreadTitleInputRef}
                isIncognito={isIncognito}
                key={section.kind === 'group' ? section.groupId : UNGROUPED_CHAT_SECTION_ID}
                locale={locale}
                onCreateThread={onCreateThread}
                onDeleteThread={onDeleteThread}
                onDeleteThreadGroup={onDeleteThreadGroup}
                onGroupTitleDraftChange={setGroupTitleDraft}
                onHistoryThreadTitleDraftChange={setHistoryThreadTitleDraft}
                onSelectThread={onSelectThread}
                reduceMotion={reduceMotion}
                runningThreadIds={runningThreadIds}
                section={section}
                selectedThreadId={selectedThreadId}
                showUngroupedHeader={showUngroupedHistoryHeader}
                startGroupTitleEdit={startGroupTitleEdit}
                startHistoryThreadTitleEdit={startHistoryThreadTitleEdit}
                t={t}
                threadDropTarget={threadDropTarget}
                toggleChatThreadGroup={toggleChatThreadGroup}
              />
            ))
          ) : (
            <div className="rounded-md border border-dashed border-border p-4 text-center text-xs text-muted-foreground">
              {t.chat.noThreads}
            </div>
          )}
          {hasMoreThreads && onLoadMoreThreads ? (
            <button
              className="flex w-full items-center justify-center gap-1.5 rounded-md px-2 py-1.5 t-meta-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground disabled:cursor-default disabled:opacity-60 disabled:hover:bg-transparent"
              disabled={isLoadingMoreThreads}
              onClick={() => onLoadMoreThreads()}
              type="button"
            >
              {isLoadingMoreThreads ? (
                <>
                  <LoaderCircle className="icon-xs animate-spin" />
                  {t.chat.loadingOlder}
                </>
              ) : (
                t.chat.loadOlder
              )}
            </button>
          ) : null}
        </div>
      </ScrollArea>
    </aside>
  )
}

function ChatHistorySectionView({
  beginGroupDrag,
  beginThreadDrag,
  cancelGroupTitleEdit,
  cancelHistoryThreadTitleEdit,
  collapsedGroupIds,
  commitGroupTitleEdit,
  commitHistoryThreadTitleEdit,
  draggedGroupId,
  draggedThreadId,
  editingGroupId,
  editingHistoryThreadId,
  groupDropTargetIndex,
  groupIndex,
  groupSectionCount,
  groupTitleDraft,
  groupTitleInputRef,
  historyThreadTitleDraft,
  historyThreadTitleInputRef,
  isIncognito,
  locale,
  onCreateThread,
  onDeleteThread,
  onDeleteThreadGroup,
  onGroupTitleDraftChange,
  onHistoryThreadTitleDraftChange,
  onSelectThread,
  reduceMotion,
  runningThreadIds,
  section,
  selectedThreadId,
  showUngroupedHeader,
  startGroupTitleEdit,
  startHistoryThreadTitleEdit,
  t,
  threadDropTarget,
  toggleChatThreadGroup,
}: {
  beginGroupDrag: (event: ReactPointerEvent<HTMLButtonElement>, groupId: string) => void
  beginThreadDrag: (event: ReactPointerEvent<HTMLButtonElement>, threadId: string) => void
  cancelGroupTitleEdit: () => void
  cancelHistoryThreadTitleEdit: () => void
  collapsedGroupIds: ReadonlySet<string>
  commitGroupTitleEdit: () => void
  commitHistoryThreadTitleEdit: () => void
  draggedGroupId: string | null
  draggedThreadId: string | null
  editingGroupId: string | null
  editingHistoryThreadId: string | null
  groupDropTargetIndex: number | null
  groupIndex: number
  groupSectionCount: number
  groupTitleDraft: string
  groupTitleInputRef: RefObject<HTMLInputElement | null>
  historyThreadTitleDraft: string
  historyThreadTitleInputRef: RefObject<HTMLInputElement | null>
  isIncognito: boolean
  locale: Locale
  onCreateThread: (groupId?: string | null) => void
  onDeleteThread: (threadId: string) => void
  onDeleteThreadGroup: (groupId: string) => void
  onGroupTitleDraftChange: (value: string) => void
  onHistoryThreadTitleDraftChange: (value: string) => void
  onSelectThread: (threadId: string) => void
  reduceMotion: boolean | null
  runningThreadIds: ReadonlySet<string>
  section: ChatHistorySection
  selectedThreadId: string | null
  showUngroupedHeader: boolean
  startGroupTitleEdit: (groupId: string, title: string) => void
  startHistoryThreadTitleEdit: (threadId: string, title: string) => void
  t: TranslationDictionary
  threadDropTarget: ChatThreadDropTarget | null
  toggleChatThreadGroup: (groupId: string) => void
}) {
  const groupId = section.groupId
  const groupKey = groupId ?? UNGROUPED_CHAT_SECTION_ID
  const isCollapsed = groupId ? collapsedGroupIds.has(groupId) : false
  const isDraggingGroup = groupId !== null && draggedGroupId === groupId
  const showDropFrame = threadDropTarget?.groupId === groupId
  const showGroupBeforeIndicator = (
    section.kind === 'group'
    && groupIndex >= 0
    && groupDropTargetIndex === groupIndex
  )
  const showGroupAfterIndicator = (
    section.kind === 'group'
    && groupIndex >= 0
    && groupDropTargetIndex === groupSectionCount
    && groupIndex === groupSectionCount - 1
  )
  const shouldRenderContent = !isCollapsed || (section.kind === 'group' && showDropFrame)
  const threads = isCollapsed ? [] : section.threads
  const title = section.kind === 'group' ? section.group.title : t.chat.ungrouped
  const SectionIcon = section.kind === 'group'
    ? isCollapsed ? Folder : FolderOpen
    : MessagesSquare

  return (
    <motion.div
      className={cn(
        'relative transition-colors',
        showDropFrame && 'rounded-md bg-brand-subtle/45',
        isDraggingGroup && 'scale-[0.995] opacity-80 shadow-[0_8px_20px_var(--shadow-soft)] ring-1 ring-ring/40',
      )}
      data-chat-history-draggable-group-id={section.kind === 'group' ? section.groupId : undefined}
      data-chat-history-group-id={groupKey}
      data-chat-history-section
      layout={!reduceMotion}
      transition={appMotion.panel}
    >
      {showGroupBeforeIndicator && (
        <span className="pointer-events-none absolute -top-1 left-1 right-1 h-0.5 rounded-full bg-brand shadow-[0_0_0_1px_var(--background)]" />
      )}
      {showGroupAfterIndicator && (
        <span className="pointer-events-none absolute -bottom-1 left-1 right-1 h-0.5 rounded-full bg-brand shadow-[0_0_0_1px_var(--background)]" />
      )}
      {section.kind === 'group' && (
        <div className="group/header grid min-h-8 grid-cols-[1.5rem_1rem_minmax(0,1fr)_auto_auto_auto_auto] items-center gap-1 px-1.5 text-foreground/75 transition-colors hover:text-foreground">
          <button
            aria-expanded={!isCollapsed}
            aria-label={`${isCollapsed ? t.chat.expandGroup : t.chat.collapseGroup}: ${section.group.title}`}
            className="grid size-6 shrink-0 place-items-center rounded-sm hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            onClick={() => toggleChatThreadGroup(section.groupId)}
            type="button"
          >
            {isCollapsed ? <ChevronRight className="size-3.5" /> : <ChevronDown className="size-3.5" />}
          </button>
          <SectionIcon className="size-3.5 shrink-0" />
          {editingGroupId === section.groupId ? (
            <input
              aria-label={t.chat.renameGroup}
              className="min-w-0 rounded-sm border-0 bg-background/85 px-1.5 py-0.5 text-xs font-semibold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onBlur={commitGroupTitleEdit}
              onChange={(event) => onGroupTitleDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  commitGroupTitleEdit()
                }
                if (event.key === 'Escape') {
                  event.preventDefault()
                  cancelGroupTitleEdit()
                }
              }}
              ref={groupTitleInputRef}
              value={groupTitleDraft}
            />
          ) : (
            <button
              className="min-w-0 truncate rounded-sm px-1 py-0.5 text-left text-xs font-semibold text-foreground/75 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onClick={() => startGroupTitleEdit(section.groupId, section.group.title)}
              title={t.chat.renameGroup}
              type="button"
            >
              {title}
            </button>
          )}
          <span className="shrink-0 rounded-sm px-1 t-hint font-semibold tabular-nums text-muted-foreground">
            {section.threads.length}
          </span>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={`${t.chat.newInFolder}: ${section.group.title}`}
                className="size-6 shrink-0 text-foreground/50 opacity-0 transition hover:text-foreground focus-visible:opacity-100 group-hover/header:opacity-100"
                onClick={() => onCreateThread(section.groupId)}
                size="icon"
                type="button"
                variant="ghost"
              >
                <SquarePen className="size-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.chat.newInFolder}</TooltipContent>
          </Tooltip>
          <button
            aria-label={`${t.chat.moveGroup}: ${section.group.title}`}
            className="grid size-6 shrink-0 cursor-grab place-items-center rounded-sm text-foreground/50 opacity-0 transition hover:bg-surface hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/header:opacity-100 active:cursor-grabbing"
            onPointerDown={(event) => beginGroupDrag(event, section.groupId)}
            type="button"
          >
            <GripVertical className="size-3.5" />
          </button>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={`${t.chat.deleteGroup}: ${section.group.title}`}
                className="size-6 shrink-0 text-foreground/50 opacity-0 transition hover:text-destructive focus-visible:opacity-100 group-hover/header:opacity-100"
                onClick={() => onDeleteThreadGroup(section.groupId)}
                size="icon"
                type="button"
                variant="ghost"
              >
                <Trash2 className="size-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.chat.deleteGroup}</TooltipContent>
          </Tooltip>
        </div>
      )}

      {section.kind === 'ungrouped' && showUngroupedHeader && (
        <div className="flex min-h-7 items-center gap-2 rounded-md px-1.5 t-caption text-foreground/65">
          <SectionIcon className="icon-sm shrink-0" />
          <span className="truncate">{t.chat.ungrouped}</span>
        </div>
      )}

      <AnimatePresence initial={false}>
        {shouldRenderContent && (
          <motion.div
            animate={{ height: 'auto', opacity: 1 }}
            className="overflow-hidden"
            exit={reduceMotion ? undefined : { height: 0, opacity: 0 }}
            initial={reduceMotion ? false : { height: 0, opacity: 0 }}
            transition={{ duration: 0.16, ease: [0.2, 0, 0, 1] }}
          >
            <div className={cn(
              'space-y-0.5',
              section.kind === 'group' && 'ml-4 border-l border-border/70 pl-2',
            )}>
              {threads.map((thread, index) => {
                const showBeforeIndicator = (
                  threadDropTarget?.groupId === groupId
                  && threadDropTarget.targetIndex === index
                )
                const showAfterIndicator = (
                  threadDropTarget?.groupId === groupId
                  && threadDropTarget.targetIndex === threads.length
                  && index === threads.length - 1
                )
                return (
                  <ChatThreadHistoryItem
                    beginThreadDrag={beginThreadDrag}
                    cancelHistoryThreadTitleEdit={cancelHistoryThreadTitleEdit}
                    commitHistoryThreadTitleEdit={commitHistoryThreadTitleEdit}
                    editingHistoryThreadId={editingHistoryThreadId}
                    historyThreadTitleDraft={historyThreadTitleDraft}
                    historyThreadTitleInputRef={historyThreadTitleInputRef}
                    isActive={selectedThreadId === thread.id}
                    isDragging={draggedThreadId === thread.id}
                    isIncognito={isIncognito}
                    isNested={section.kind === 'group'}
                    isThreadRunning={runningThreadIds.has(thread.id)}
                    key={thread.id}
                    locale={locale}
                    onDeleteThread={onDeleteThread}
                    onHistoryThreadTitleDraftChange={onHistoryThreadTitleDraftChange}
                    onSelectThread={onSelectThread}
                    reduceMotion={reduceMotion}
                    showAfterIndicator={showAfterIndicator}
                    showBeforeIndicator={showBeforeIndicator}
                    startHistoryThreadTitleEdit={startHistoryThreadTitleEdit}
                    t={t}
                    thread={thread}
                  />
                )
              })}
              {section.kind === 'group' && threads.length === 0 && !isCollapsed && (
                <div className="rounded-md px-2 py-1.5 t-meta-sm font-medium text-muted-foreground">
                  {t.chat.emptyGroup}
                </div>
              )}
              {section.kind === 'group' && isCollapsed && showDropFrame && (
                <div className="rounded-md border border-dashed border-brand/30 px-2 py-1.5 text-center t-meta-sm font-semibold text-brand">
                  {t.chat.dropIntoGroup}
                </div>
              )}
              {section.kind === 'ungrouped' && threads.length === 0 && showDropFrame && (
                <div className="rounded-md border border-dashed border-brand/30 px-2 py-1.5 text-center t-meta-sm font-semibold text-brand">
                  {t.chat.dropIntoGroup}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function ChatThreadHistoryItem({
  beginThreadDrag,
  cancelHistoryThreadTitleEdit,
  commitHistoryThreadTitleEdit,
  editingHistoryThreadId,
  historyThreadTitleDraft,
  historyThreadTitleInputRef,
  isActive,
  isDragging,
  isIncognito,
  isNested,
  isThreadRunning,
  locale,
  onDeleteThread,
  onHistoryThreadTitleDraftChange,
  onSelectThread,
  reduceMotion,
  showAfterIndicator,
  showBeforeIndicator,
  startHistoryThreadTitleEdit,
  t,
  thread,
}: {
  beginThreadDrag: (event: ReactPointerEvent<HTMLButtonElement>, threadId: string) => void
  cancelHistoryThreadTitleEdit: () => void
  commitHistoryThreadTitleEdit: () => void
  editingHistoryThreadId: string | null
  historyThreadTitleDraft: string
  historyThreadTitleInputRef: RefObject<HTMLInputElement | null>
  isActive: boolean
  isDragging: boolean
  isIncognito: boolean
  isNested: boolean
  isThreadRunning: boolean
  locale: Locale
  onDeleteThread: (threadId: string) => void
  onHistoryThreadTitleDraftChange: (value: string) => void
  onSelectThread: (threadId: string) => void
  reduceMotion: boolean | null
  showAfterIndicator: boolean
  showBeforeIndicator: boolean
  startHistoryThreadTitleEdit: (threadId: string, title: string) => void
  t: TranslationDictionary
  thread: ChatThread
}) {
  const isEditingTitle = editingHistoryThreadId === thread.id

  return (
    <motion.div
      className={cn(
        'group/thread relative transition-colors',
        isNested
          ? 'bg-transparent hover:text-foreground'
          : 'border-border/60 bg-card/60 shadow-[0_1px_1px_var(--shadow-hairline)] hover:border-border hover:bg-background',
        !isNested && 'rounded-md border',
        isNested && isActive && 'before:absolute before:-left-[9px] before:bottom-1.5 before:top-1.5 before:w-0.5 before:rounded-full before:bg-brand',
        !isNested && isActive && 'border-brand/25 bg-brand-subtle/45 ring-1 ring-brand/10',
        isDragging && 'scale-[0.99] opacity-75 shadow-[0_8px_20px_var(--shadow-soft)] ring-1 ring-ring/50',
      )}
      data-chat-history-thread-id={thread.id}
      layout={!reduceMotion}
      transition={appMotion.panel}
    >
      {showBeforeIndicator && (
        <span className="pointer-events-none absolute -top-1 left-1 right-1 h-0.5 rounded-full bg-brand shadow-[0_0_0_1px_var(--background)]" />
      )}
      {showAfterIndicator && (
        <span className="pointer-events-none absolute -bottom-1 left-1 right-1 h-0.5 rounded-full bg-brand shadow-[0_0_0_1px_var(--background)]" />
      )}
      <button
        aria-label={`${t.chat.moveThread}: ${thread.title}`}
        className={cn(
          'absolute top-1/2 z-10 grid -translate-y-1/2 cursor-grab place-items-center rounded-sm text-foreground/50 opacity-0 transition hover:bg-surface hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/thread:opacity-100 active:cursor-grabbing',
          isNested ? 'right-7 size-6' : 'right-8 size-7',
        )}
        onPointerDown={(event) => beginThreadDrag(event, thread.id)}
        type="button"
      >
        <GripVertical className="size-3.5" />
      </button>
      {isEditingTitle ? (
        <div className={cn(
          'grid w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 text-left',
          isNested ? 'min-h-8 px-2 py-1 pr-14' : 'min-h-10 px-3 py-1.5 pr-16',
        )}>
          <span className="flex min-w-0 items-center gap-2">
            <input
              aria-label={t.chat.renameTitle}
              className={cn(
                'min-w-0 flex-1 rounded-sm border-0 bg-background/85 px-1.5 py-0.5 t-list text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring',
              )}
              onBlur={commitHistoryThreadTitleEdit}
              onChange={(event) => onHistoryThreadTitleDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  commitHistoryThreadTitleEdit()
                }
                if (event.key === 'Escape') {
                  event.preventDefault()
                  cancelHistoryThreadTitleEdit()
                }
              }}
              ref={historyThreadTitleInputRef}
              value={historyThreadTitleDraft}
            />
            {isThreadRunning && <RunningThreadDot label={t.chat.generating} />}
          </span>
          <span className="shrink-0 t-hint text-muted-foreground">
            {displayRelativeDate(chatThreadHistoryTimeIso(thread), locale)}
          </span>
        </div>
      ) : (
        <button
          aria-pressed={isActive}
          className={cn(
            'grid w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isNested ? 'min-h-8 px-2 py-1 pr-14' : 'min-h-10 px-3 py-1.5 pr-16',
          )}
          onClick={() => onSelectThread(thread.id)}
          onDoubleClick={() => startHistoryThreadTitleEdit(thread.id, thread.title)}
          title={t.chat.renameTitle}
          type="button"
        >
          <span className="flex min-w-0 items-center gap-2">
            <span className={cn(
              'block min-w-0 flex-1 truncate t-list',
              isNested ? 'text-foreground/85' : 'text-foreground',
              isActive && 'text-foreground',
            )}>
              {thread.title}
            </span>
            {isThreadRunning && <RunningThreadDot label={t.chat.generating} />}
          </span>
          <span className="shrink-0 t-hint text-muted-foreground">
            {displayRelativeDate(chatThreadHistoryTimeIso(thread), locale)}
          </span>
        </button>
      )}
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            aria-label={`${t.chat.delete}: ${thread.title}`}
            className={cn(
              'absolute top-1/2 -translate-y-1/2 text-foreground/55 opacity-0 transition hover:text-destructive focus-visible:opacity-100 group-hover/thread:opacity-100',
              isNested ? 'right-1 size-6' : 'right-1.5 size-7',
            )}
            disabled={isIncognito}
            onClick={() => onDeleteThread(thread.id)}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Trash2 className="size-3.5" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>{t.chat.delete}</TooltipContent>
      </Tooltip>
    </motion.div>
  )
}

function RunningThreadDot({ label }: { label: string }) {
  return (
    <span
      aria-label={label}
      className="relative flex size-2 shrink-0"
      title={label}
    >
      <span className="absolute inline-flex size-full rounded-full bg-brand/45 opacity-75 motion-safe:animate-ping" />
      <span className="relative inline-flex size-2 rounded-full bg-brand" />
    </span>
  )
}

function chatThreadHistoryTimeIso(thread: ChatThread) {
  return thread.messages[thread.messages.length - 1]?.createdAt ?? thread.updatedAt
}
