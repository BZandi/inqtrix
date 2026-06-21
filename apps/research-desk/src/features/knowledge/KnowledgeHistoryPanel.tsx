import {
  BookOpenCheck,
  ChevronDown,
  ChevronRight,
  Folder,
  FolderOpen,
  FolderPlus,
  GripVertical,
  PanelLeftClose,
  SquarePen,
  Trash2,
} from '@/components/icons'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  displayRelativeDate,
  type KnowledgeSessionHistorySection,
} from '@/features/project/selectors'
import type {
  KnowledgeSessionRecord,
  KnowledgeThreadItemRecord,
} from '@/features/project/types'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'
import { AnimatePresence, motion } from 'motion/react'
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type RefObject,
} from 'react'

type SessionDropTarget = {
  groupId: string | null
  targetIndex: number
}

type KnowledgeHistoryPanelProps = {
  items: KnowledgeThreadItemRecord[]
  onCreateSession: (groupId?: string | null) => void
  onCreateSessionGroup: () => void
  onDeleteSession: (sessionId: string) => void
  onDeleteSessionGroup: (groupId: string) => void
  onHide?: () => void
  onMoveSessionGroup: (groupId: string, targetIndex: number) => void
  onMoveSessionToGroup: (sessionId: string, groupId: string | null, targetIndex: number) => void
  onRenameSession: (sessionId: string, title: string) => void
  onRenameSessionGroup: (groupId: string, title: string) => void
  onSelectSession: (sessionId: string) => void
  sections: KnowledgeSessionHistorySection[]
  selectedSessionId: string | null
  reduceMotion: boolean
  sessions: KnowledgeSessionRecord[]
}

const UNGROUPED_KNOWLEDGE_SECTION_ID = '__ungrouped_knowledge__'

export function KnowledgeHistoryPanel({
  items,
  onCreateSession,
  onCreateSessionGroup,
  onDeleteSession,
  onDeleteSessionGroup,
  onHide,
  onMoveSessionGroup,
  onMoveSessionToGroup,
  onRenameSession,
  onRenameSessionGroup,
  onSelectSession,
  sections,
  selectedSessionId,
  reduceMotion,
  sessions,
}: KnowledgeHistoryPanelProps) {
  const { t } = useLocale()
  const [collapsedGroupIds, setCollapsedGroupIds] = useState<ReadonlySet<string>>(() => new Set())
  const [draggedGroupId, setDraggedGroupId] = useState<string | null>(null)
  const [draggedSessionId, setDraggedSessionId] = useState<string | null>(null)
  const [editingGroupId, setEditingGroupId] = useState<string | null>(null)
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [groupDropTargetIndex, setGroupDropTargetIndex] = useState<number | null>(null)
  const [groupTitleDraft, setGroupTitleDraft] = useState('')
  const [sessionDropTarget, setSessionDropTarget] = useState<SessionDropTarget | null>(null)
  const [sessionTitleDraft, setSessionTitleDraft] = useState('')
  const groupTitleInputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)
  const sessionTitleInputRef = useRef<HTMLInputElement | null>(null)

  const runningSessionIds = new Set<string>()
  for (const item of items) {
    if (item.status === 'running') runningSessionIds.add(item.sessionId)
  }

  useLayoutEffect(() => {
    if (!editingGroupId) return
    groupTitleInputRef.current?.focus()
    groupTitleInputRef.current?.select()
  }, [editingGroupId])

  useLayoutEffect(() => {
    if (!editingSessionId) return
    sessionTitleInputRef.current?.focus()
    sessionTitleInputRef.current?.select()
  }, [editingSessionId])

  useEffect(() => {
    if (!editingSessionId) return
    if (sessions.some((session) => session.id === editingSessionId)) return
    setEditingSessionId(null)
    setSessionTitleDraft('')
  }, [editingSessionId, sessions])

  function toggleGroup(groupId: string) {
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

  function startGroupEdit(groupId: string, title: string) {
    setEditingGroupId(groupId)
    setGroupTitleDraft(title)
  }

  function commitGroupEdit() {
    if (!editingGroupId) return
    const title = groupTitleDraft.trim()
    if (title) onRenameSessionGroup(editingGroupId, title)
    setEditingGroupId(null)
    setGroupTitleDraft('')
  }

  function cancelGroupEdit() {
    setEditingGroupId(null)
    setGroupTitleDraft('')
  }

  function startSessionEdit(session: KnowledgeSessionRecord) {
    setEditingSessionId(session.id)
    setSessionTitleDraft(session.title)
  }

  function commitSessionEdit() {
    if (!editingSessionId) return
    const title = sessionTitleDraft.trim()
    if (title) onRenameSession(editingSessionId, title)
    setEditingSessionId(null)
    setSessionTitleDraft('')
  }

  function cancelSessionEdit() {
    setEditingSessionId(null)
    setSessionTitleDraft('')
  }

  function readGroupDropTarget(clientY: number, excludedGroupId = draggedGroupId) {
    const container = listRef.current
    if (!container) return null
    const groupElements = Array.from(container.querySelectorAll<HTMLElement>('[data-knowledge-history-group-id]'))
      .filter((groupElement) => (
        groupElement.dataset.knowledgeHistoryGroupId !== excludedGroupId
        && groupElement.getBoundingClientRect().height > 0
      ))

    for (const [index, groupElement] of groupElements.entries()) {
      const rect = groupElement.getBoundingClientRect()
      if (clientY < rect.top + rect.height / 2) return index
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
      const targetIndex = readGroupDropTarget(upEvent.clientY, groupId)
      cleanupPointerDrag()
      if (targetIndex !== null) onMoveSessionGroup(groupId, targetIndex)
    }

    function cleanupPointerDrag() {
      document.removeEventListener('pointermove', handlePointerMove)
      document.removeEventListener('pointerup', finishPointerDrag)
      document.removeEventListener('pointercancel', cleanupPointerDrag)
      setDraggedGroupId(null)
      setGroupDropTargetIndex(null)
    }

    document.addEventListener('pointermove', handlePointerMove)
    document.addEventListener('pointerup', finishPointerDrag)
    document.addEventListener('pointercancel', cleanupPointerDrag)
  }

  function readSessionDropTarget(clientY: number, excludedSessionId = draggedSessionId): SessionDropTarget | null {
    const container = listRef.current
    if (!container) return null
    const sectionElements = Array.from(container.querySelectorAll<HTMLElement>('[data-knowledge-history-section]'))
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
    const groupKey = sectionElement.dataset.knowledgeHistorySectionGroupId ?? UNGROUPED_KNOWLEDGE_SECTION_ID
    const groupId = groupKey === UNGROUPED_KNOWLEDGE_SECTION_ID ? null : groupKey
    const sessionElements = Array.from(sectionElement.querySelectorAll<HTMLElement>('[data-knowledge-history-session-id]'))
      .filter((sessionElement) => (
        sessionElement.dataset.knowledgeHistorySessionId !== excludedSessionId
        && sessionElement.getBoundingClientRect().height > 0
      ))

    for (const [index, sessionElement] of sessionElements.entries()) {
      const rect = sessionElement.getBoundingClientRect()
      if (clientY < rect.top + rect.height / 2) return { groupId, targetIndex: index }
    }

    return { groupId, targetIndex: sessionElements.length }
  }

  function beginSessionDrag(event: ReactPointerEvent<HTMLButtonElement>, sessionId: string) {
    if (event.button !== 0) return
    event.preventDefault()
    setDraggedSessionId(sessionId)
    setSessionDropTarget(readSessionDropTarget(event.clientY, sessionId))

    function handlePointerMove(moveEvent: PointerEvent) {
      setSessionDropTarget(readSessionDropTarget(moveEvent.clientY, sessionId))
    }

    function finishPointerDrag(upEvent: PointerEvent) {
      const target = readSessionDropTarget(upEvent.clientY, sessionId)
      cleanupPointerDrag()
      if (target) onMoveSessionToGroup(sessionId, target.groupId, target.targetIndex)
    }

    function cleanupPointerDrag() {
      document.removeEventListener('pointermove', handlePointerMove)
      document.removeEventListener('pointerup', finishPointerDrag)
      document.removeEventListener('pointercancel', cleanupPointerDrag)
      setDraggedSessionId(null)
      setSessionDropTarget(null)
    }

    document.addEventListener('pointermove', handlePointerMove)
    document.addEventListener('pointerup', finishPointerDrag)
    document.addEventListener('pointercancel', cleanupPointerDrag)
  }

  const hasStructure = sessions.length > 0 || sections.some((section) => section.kind === 'group')
  const showUngroupedHeader = sections.some((section) => section.kind === 'group')
  const groupIds = sections.flatMap((section) => (section.kind === 'group' ? [section.groupId] : []))
  const dropGroupIds = draggedGroupId ? groupIds.filter((groupId) => groupId !== draggedGroupId) : groupIds

  return (
    <aside className="flex h-full min-h-0 w-full flex-col border-r border-border bg-surface/60">
      <div className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-border px-3">
        <div className="flex min-w-0 items-center gap-2">
          <BookOpenCheck className="size-4 shrink-0 text-foreground/80" />
          <h2 className="truncate t-section text-foreground">{t.knowledge.sessions}</h2>
        </div>
        <div className="flex items-center gap-1.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={t.knowledge.newFolder}
                className="size-7 shrink-0"
                onClick={onCreateSessionGroup}
                size="icon"
                type="button"
                variant="ghost"
              >
                <FolderPlus className="size-4 text-foreground/85" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.knowledge.newFolder}</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={t.knowledge.newSession}
                className="size-7 shrink-0"
                onClick={() => onCreateSession()}
                size="icon"
                type="button"
                variant="ghost"
              >
                <SquarePen className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.knowledge.newSession}</TooltipContent>
          </Tooltip>
          {onHide && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  aria-label={t.knowledge.hideSessions}
                  className="size-7 shrink-0"
                  onClick={onHide}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  <PanelLeftClose className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{t.knowledge.hideSessions}</TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-1 p-2" ref={listRef}>
          {hasStructure ? (
            sections.map((section) => (
              <KnowledgeHistorySectionView
                beginGroupDrag={beginGroupDrag}
                beginSessionDrag={beginSessionDrag}
                cancelGroupEdit={cancelGroupEdit}
                cancelSessionEdit={cancelSessionEdit}
                collapsedGroupIds={collapsedGroupIds}
                commitGroupEdit={commitGroupEdit}
                commitSessionEdit={commitSessionEdit}
                draggedGroupId={draggedGroupId}
                draggedSessionId={draggedSessionId}
                editingGroupId={editingGroupId}
                editingSessionId={editingSessionId}
                groupDropTargetIndex={groupDropTargetIndex}
                groupIndex={section.kind === 'group' ? dropGroupIds.indexOf(section.groupId) : -1}
                groupTitleDraft={groupTitleDraft}
                groupTitleInputRef={groupTitleInputRef}
                groupCount={dropGroupIds.length}
                key={section.kind === 'group' ? section.groupId : UNGROUPED_KNOWLEDGE_SECTION_ID}
                onCreateSession={onCreateSession}
                onDeleteSession={onDeleteSession}
                onDeleteSessionGroup={onDeleteSessionGroup}
                onGroupTitleDraftChange={setGroupTitleDraft}
                onSelectSession={onSelectSession}
                onSessionTitleDraftChange={setSessionTitleDraft}
                reduceMotion={reduceMotion}
                runningSessionIds={runningSessionIds}
                section={section}
                selectedSessionId={selectedSessionId}
                sessionDropTarget={sessionDropTarget}
                sessionTitleDraft={sessionTitleDraft}
                sessionTitleInputRef={sessionTitleInputRef}
                showUngroupedHeader={showUngroupedHeader}
                startGroupEdit={startGroupEdit}
                startSessionEdit={startSessionEdit}
                toggleGroup={toggleGroup}
              />
            ))
          ) : (
            <div className="rounded-md border border-dashed border-border p-4 text-center t-meta-sm text-muted-foreground">
              {t.knowledge.noSessions}
            </div>
          )}
        </div>
      </ScrollArea>
    </aside>
  )
}

function KnowledgeHistorySectionView({
  beginGroupDrag,
  beginSessionDrag,
  cancelGroupEdit,
  cancelSessionEdit,
  collapsedGroupIds,
  commitGroupEdit,
  commitSessionEdit,
  draggedGroupId,
  draggedSessionId,
  editingGroupId,
  editingSessionId,
  groupDropTargetIndex,
  groupIndex,
  groupTitleDraft,
  groupTitleInputRef,
  groupCount,
  onCreateSession,
  onDeleteSession,
  onDeleteSessionGroup,
  onGroupTitleDraftChange,
  onSelectSession,
  onSessionTitleDraftChange,
  reduceMotion,
  runningSessionIds,
  section,
  selectedSessionId,
  sessionDropTarget,
  sessionTitleDraft,
  sessionTitleInputRef,
  showUngroupedHeader,
  startGroupEdit,
  startSessionEdit,
  toggleGroup,
}: {
  beginGroupDrag: (event: ReactPointerEvent<HTMLButtonElement>, groupId: string) => void
  beginSessionDrag: (event: ReactPointerEvent<HTMLButtonElement>, sessionId: string) => void
  cancelGroupEdit: () => void
  cancelSessionEdit: () => void
  collapsedGroupIds: ReadonlySet<string>
  commitGroupEdit: () => void
  commitSessionEdit: () => void
  draggedGroupId: string | null
  draggedSessionId: string | null
  editingGroupId: string | null
  editingSessionId: string | null
  groupDropTargetIndex: number | null
  groupIndex: number
  groupTitleDraft: string
  groupTitleInputRef: RefObject<HTMLInputElement | null>
  groupCount: number
  onCreateSession: (groupId?: string | null) => void
  onDeleteSession: (sessionId: string) => void
  onDeleteSessionGroup: (groupId: string) => void
  onGroupTitleDraftChange: (value: string) => void
  onSelectSession: (sessionId: string) => void
  onSessionTitleDraftChange: (value: string) => void
  reduceMotion: boolean
  runningSessionIds: ReadonlySet<string>
  section: KnowledgeSessionHistorySection
  selectedSessionId: string | null
  sessionDropTarget: SessionDropTarget | null
  sessionTitleDraft: string
  sessionTitleInputRef: RefObject<HTMLInputElement | null>
  showUngroupedHeader: boolean
  startGroupEdit: (groupId: string, title: string) => void
  startSessionEdit: (session: KnowledgeSessionRecord) => void
  toggleGroup: (groupId: string) => void
}) {
  const { t } = useLocale()
  const groupId = section.groupId
  const groupKey = groupId ?? UNGROUPED_KNOWLEDGE_SECTION_ID
  const isCollapsed = groupId ? collapsedGroupIds.has(groupId) : false
  const isDraggingGroup = groupId !== null && draggedGroupId === groupId
  const showDropFrame = sessionDropTarget?.groupId === groupId
  const showGroupBeforeIndicator = section.kind === 'group' && groupIndex >= 0 && groupDropTargetIndex === groupIndex
  const showGroupAfterIndicator = (
    section.kind === 'group'
    && groupIndex >= 0
    && groupDropTargetIndex === groupCount
    && groupIndex === groupCount - 1
  )
  const sessions = isCollapsed ? [] : section.sessions
  const SectionIcon = section.kind === 'group'
    ? isCollapsed ? Folder : FolderOpen
    : BookOpenCheck

  return (
    <motion.div
      className={cn(
        'relative transition-colors',
        showDropFrame && 'rounded-md bg-brand-subtle/45',
        isDraggingGroup && 'scale-[0.995] opacity-80 shadow-[0_8px_20px_var(--shadow-soft)] ring-1 ring-ring/40',
      )}
      data-knowledge-history-group-id={section.kind === 'group' ? section.groupId : undefined}
      data-knowledge-history-section
      data-knowledge-history-section-group-id={groupKey}
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
            onClick={() => toggleGroup(section.groupId)}
            type="button"
          >
            {isCollapsed ? <ChevronRight className="size-3.5" /> : <ChevronDown className="size-3.5" />}
          </button>
          <SectionIcon className="size-3.5 shrink-0" />
          {editingGroupId === section.groupId ? (
            <input
              aria-label={t.knowledge.renameFolder}
              className="min-w-0 rounded-sm border-0 bg-background/85 px-1.5 py-0.5 text-xs font-semibold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onBlur={commitGroupEdit}
              onChange={(event) => onGroupTitleDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  commitGroupEdit()
                }
                if (event.key === 'Escape') {
                  event.preventDefault()
                  cancelGroupEdit()
                }
              }}
              ref={groupTitleInputRef}
              value={groupTitleDraft}
            />
          ) : (
            <button
              className="min-w-0 truncate rounded-sm px-1 py-0.5 text-left text-xs font-semibold text-foreground/75 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onClick={() => startGroupEdit(section.groupId, section.group.title)}
              title={t.knowledge.renameFolder}
              type="button"
            >
              {section.group.title}
            </button>
          )}
          <span className="shrink-0 rounded-sm px-1 t-hint font-semibold tabular-nums text-muted-foreground">
            {section.sessions.length}
          </span>
          <HistoryIconButton
            label={`${t.knowledge.newInFolder}: ${section.group.title}`}
            onClick={() => onCreateSession(section.groupId)}
          >
            <SquarePen className="size-3.5" />
          </HistoryIconButton>
          <button
            aria-label={`${t.knowledge.moveFolder}: ${section.group.title}`}
            className="grid size-6 shrink-0 cursor-grab place-items-center rounded-sm text-foreground/50 opacity-0 transition hover:bg-surface hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/header:opacity-100 active:cursor-grabbing"
            onPointerDown={(event) => beginGroupDrag(event, section.groupId)}
            type="button"
          >
            <GripVertical className="size-3.5" />
          </button>
          <HistoryIconButton
            destructive
            label={`${t.knowledge.deleteFolder}: ${section.group.title}`}
            onClick={() => onDeleteSessionGroup(section.groupId)}
          >
            <Trash2 className="size-3.5" />
          </HistoryIconButton>
        </div>
      )}

      {section.kind === 'ungrouped' && showUngroupedHeader && (
        <div className="flex min-h-7 items-center gap-2 rounded-md px-1.5 t-caption text-foreground/65">
          <SectionIcon className="icon-sm shrink-0" />
          <span className="truncate">{t.knowledge.ungrouped}</span>
        </div>
      )}

      <AnimatePresence initial={false}>
        {(!isCollapsed || showDropFrame) && (
          <motion.div
            animate={{ height: 'auto', opacity: 1 }}
            className="overflow-hidden"
            exit={reduceMotion ? undefined : { height: 0, opacity: 0 }}
            initial={reduceMotion ? false : { height: 0, opacity: 0 }}
            transition={{ duration: 0.16, ease: [0.2, 0, 0, 1] }}
          >
            <div className={cn('space-y-0.5', section.kind === 'group' && 'ml-4 border-l border-border/70 pl-2')}>
              {sessions.map((session, index) => (
                <KnowledgeSessionHistoryItem
                  beginSessionDrag={beginSessionDrag}
                  commitSessionEdit={commitSessionEdit}
                  cancelSessionEdit={cancelSessionEdit}
                  dragged={draggedSessionId === session.id}
                  editing={editingSessionId === session.id}
                  key={session.id}
                  nested={section.kind === 'group'}
                  onDeleteSession={onDeleteSession}
                  onSelectSession={onSelectSession}
                  onSessionTitleDraftChange={onSessionTitleDraftChange}
                  reduceMotion={reduceMotion}
                  running={runningSessionIds.has(session.id)}
                  selected={selectedSessionId === session.id}
                  sectionGroupId={groupId}
                  session={session}
                  sessionDropTarget={sessionDropTarget}
                  sessionIndex={index}
                  sessionTitleDraft={sessionTitleDraft}
                  sessionTitleInputRef={sessionTitleInputRef}
                  startSessionEdit={startSessionEdit}
                />
              ))}
              {section.kind === 'group' && sessions.length === 0 && !isCollapsed && (
                <div className="rounded-md px-2 py-1.5 t-meta-sm font-medium text-muted-foreground">
                  {t.knowledge.emptyFolder}
                </div>
              )}
              {showDropFrame && sessions.length === 0 && (
                <div className="rounded-md border border-dashed border-brand/30 px-2 py-1.5 text-center t-meta-sm font-semibold text-brand">
                  {t.knowledge.dropIntoFolder}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function KnowledgeSessionHistoryItem({
  beginSessionDrag,
  cancelSessionEdit,
  commitSessionEdit,
  dragged,
  editing,
  nested,
  onDeleteSession,
  onSelectSession,
  onSessionTitleDraftChange,
  reduceMotion,
  running,
  selected,
  sectionGroupId,
  session,
  sessionDropTarget,
  sessionIndex,
  sessionTitleDraft,
  sessionTitleInputRef,
  startSessionEdit,
}: {
  beginSessionDrag: (event: ReactPointerEvent<HTMLButtonElement>, sessionId: string) => void
  cancelSessionEdit: () => void
  commitSessionEdit: () => void
  dragged: boolean
  editing: boolean
  nested: boolean
  onDeleteSession: (sessionId: string) => void
  onSelectSession: (sessionId: string) => void
  onSessionTitleDraftChange: (value: string) => void
  reduceMotion: boolean
  running: boolean
  selected: boolean
  sectionGroupId: string | null
  session: KnowledgeSessionRecord
  sessionDropTarget: SessionDropTarget | null
  sessionIndex: number
  sessionTitleDraft: string
  sessionTitleInputRef: RefObject<HTMLInputElement | null>
  startSessionEdit: (session: KnowledgeSessionRecord) => void
}) {
  const { locale, t } = useLocale()
  const sessionDropApplies = sessionDropTarget?.groupId === sectionGroupId
  const showBeforeIndicator = sessionDropApplies && sessionDropTarget?.targetIndex === sessionIndex
  const showAfterIndicator = sessionDropApplies && sessionDropTarget?.targetIndex === sessionIndex + 1
  const sessionTime = displayRelativeDate(session.updatedAt, locale)

  return (
    <motion.div
      className={cn(
        'group/session relative transition-colors',
        nested
          ? 'bg-transparent hover:text-foreground'
          : 'border-border/60 bg-card/60 shadow-[0_1px_1px_var(--shadow-hairline)] hover:border-border hover:bg-background',
        !nested && 'rounded-md border',
        nested && selected && 'before:absolute before:-left-[9px] before:bottom-1.5 before:top-1.5 before:w-0.5 before:rounded-full before:bg-brand',
        !nested && selected && 'border-brand/25 bg-brand-subtle/45 ring-1 ring-brand/10',
        dragged && 'scale-[0.99] opacity-75 shadow-[0_8px_20px_var(--shadow-soft)] ring-1 ring-ring/50',
      )}
      data-knowledge-history-session-id={session.id}
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
        aria-label={`${t.knowledge.moveSession}: ${session.title}`}
        className={cn(
          'absolute top-1/2 z-10 grid -translate-y-1/2 cursor-grab place-items-center rounded-sm text-foreground/50 opacity-0 transition hover:bg-surface hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/session:opacity-100 active:cursor-grabbing',
          nested ? 'right-7 size-6' : 'right-8 size-7',
        )}
        onPointerDown={(event) => beginSessionDrag(event, session.id)}
        type="button"
      >
        <GripVertical className="size-3.5" />
      </button>
      {editing ? (
        <div className={cn(
          'grid w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 text-left',
          nested ? 'min-h-8 px-2 py-1 pr-14' : 'min-h-10 px-3 py-1.5 pr-16',
        )}>
          <span className="flex min-w-0 items-center gap-2">
            <input
              aria-label={t.knowledge.renameSession}
              className="min-w-0 flex-1 rounded-sm border-0 bg-background/85 px-1.5 py-0.5 t-list text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onBlur={commitSessionEdit}
              onChange={(event) => onSessionTitleDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  commitSessionEdit()
                }
                if (event.key === 'Escape') {
                  event.preventDefault()
                  cancelSessionEdit()
                }
              }}
              ref={sessionTitleInputRef}
              value={sessionTitleDraft}
            />
            {running && <RunningSessionDot label={t.common.running} />}
          </span>
          <span className="shrink-0 t-hint tabular-nums text-muted-foreground">
            {sessionTime}
          </span>
        </div>
      ) : (
        <button
          aria-pressed={selected}
          className={cn(
            'grid w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            nested ? 'min-h-8 px-2 py-1 pr-14' : 'min-h-10 px-3 py-1.5 pr-16',
          )}
          onClick={() => onSelectSession(session.id)}
          onDoubleClick={() => startSessionEdit(session)}
          title={t.knowledge.renameSession}
          type="button"
        >
          <span className="flex min-w-0 items-center gap-2">
            <span className={cn(
              'block min-w-0 flex-1 truncate t-list',
              nested ? 'text-foreground/85' : 'text-foreground',
              selected && 'text-foreground',
            )}>
              {session.title}
            </span>
            {running && <RunningSessionDot label={t.common.running} />}
          </span>
          <span className="shrink-0 t-hint tabular-nums text-muted-foreground">
            {sessionTime}
          </span>
        </button>
      )}
      <Button
        aria-label={`${t.knowledge.deleteSession}: ${session.title}`}
        className={cn(
          'absolute top-1/2 -translate-y-1/2 text-foreground/55 opacity-0 transition hover:text-destructive focus-visible:opacity-100 group-hover/session:opacity-100',
          nested ? 'right-1 size-6' : 'right-1.5 size-7',
        )}
        onClick={() => onDeleteSession(session.id)}
        size="icon"
        type="button"
        variant="ghost"
      >
        <Trash2 className="size-3.5" />
      </Button>
    </motion.div>
  )
}

function RunningSessionDot({ label }: { label: string }) {
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

function HistoryIconButton({
  children,
  destructive,
  label,
  onClick,
}: {
  children: ReactNode
  destructive?: boolean
  label: string
  onClick: () => void
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          className={cn(
            'size-6 shrink-0 text-foreground/50 opacity-0 transition focus-visible:opacity-100 group-hover/header:opacity-100',
            destructive ? 'hover:text-destructive' : 'hover:text-foreground',
          )}
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
