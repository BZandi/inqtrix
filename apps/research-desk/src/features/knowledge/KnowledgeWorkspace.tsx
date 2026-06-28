import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import {
  BookOpenCheck,
  Check,
  Copy,
  Database,
  Eraser,
  EyeOff,
  ListChecks,
  PencilLine,
  Search,
  SendHorizontal,
  Sparkles,
  Trash2,
  X,
} from '@/components/icons'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Chip } from '@/components/ui/chip'
import { PanelToggle } from '@/components/ui/panel-toggle'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { WelcomeState } from '@/components/ui/welcome-state'
import { useLocale } from '@/i18n/LocaleProvider'
import { formatMessageTimestamp } from '@/lib/time'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'
import type {
  KnowledgeReferenceRecord,
  KnowledgeSessionRecord,
  KnowledgeThreadItemRecord,
} from '@/features/project/types'
import type { KnowledgeSessionHistorySection } from '@/features/project/selectors'
import type { KnowledgeSearchHit } from '@/features/researchRuns/types'
import { AnswerCard } from './AnswerCard'
import {
  knowledgeCompletionHandoffId,
  knowledgeItemStatusSnapshot,
} from './completionHandoff'
import { DocumentFindResults, type FindResultsState } from './DocumentFindResults'
import { DocumentViewer } from './DocumentViewer'
import { groupHitsByDocument } from './findGrouping'
import { searchTermsFromQuery } from './highlight'
import { KnowledgeComposer } from './KnowledgeComposer'
import { KnowledgeHistoryPanel } from './KnowledgeHistoryPanel'
import { KnowledgeRunCard } from './KnowledgeRunCard'
import { KnowledgeSourcePanel } from './KnowledgeSourcePanel'
import type { KnowledgeProfileOption } from './profileOptions'
import type {
  DocumentViewerTarget,
  KnowledgeCollectionOption,
  KnowledgeDataSource,
} from './types'

export type KnowledgeMode = 'ask' | 'find'
export type KnowledgeAskOptions = {
  collectionIds?: string[]
  profileId?: string | null
  replaceItemId?: string
  topK?: number | null
  finalK?: number | null
}

/** The embedded right panel of the Ask view: which answer's sources it shows
 * and the document currently open in the reader. */
type AskPanelState = {
  item: KnowledgeThreadItemRecord | null
  target: DocumentViewerTarget | null
}

const FIND_DEBOUNCE_MS = 300
const FIND_MIN_QUERY_LENGTH = 2
const FIND_TOP_K = 20
const KNOWLEDGE_ANSWER_ENTRY_MS = 1800
const KNOWLEDGE_ANSWER_SCROLL_SETTLE_MS = 340

type KnowledgeWorkspaceProps = {
  historyItems: KnowledgeThreadItemRecord[]
  collections: KnowledgeCollectionOption[]
  composerNotice: string | null
  dataSource: KnowledgeDataSource
  defaultProfileId: string | null
  defaultTopK: number
  evidenceKMax: number
  rerankerProvider: string | null
  isAskDisabled: boolean
  isAskRunning: boolean
  isHistoryVisible: boolean
  isIncognito: boolean
  items: KnowledgeThreadItemRecord[]
  mode: KnowledgeMode
  knowledgeQuestion: string
  onKnowledgeQuestionChange: (question: string) => void
  onAsk: (question: string, options?: KnowledgeAskOptions) => void
  onClearSession: () => void
  onCreateSession: (groupId?: string | null) => void
  onCreateSessionGroup: () => void
  onDemoAsk?: () => void
  onDeleteSessionGroup: (groupId: string) => void
  onDeleteSession: (sessionId: string) => void
  onDeleteItems: (itemIds: string[]) => void
  onOpenDatabase?: () => void
  onHistoryVisibleChange: (visible: boolean) => void
  onIncognitoChange: (enabled: boolean) => void
  onModeChange: (mode: KnowledgeMode) => void
  onMoveSessionGroup: (groupId: string, targetIndex: number) => void
  onMoveSessionToGroup: (sessionId: string, groupId: string | null, targetIndex: number) => void
  onProfileChange: (profileId: string | null) => void
  onRenameSessionGroup: (groupId: string, title: string) => void
  onRenameSession: (sessionId: string, title: string) => void
  onSelectSession: (sessionId: string) => void
  onStopAsk: () => void
  onTogglePinnedSession: (sessionId: string) => void
  onSelectedCollectionIdsChange: (ids: string[]) => void
  onTopKChange: (topK: number | null) => void
  onFinalKChange: (finalK: number | null) => void
  profileId: string | null
  profileOptions: KnowledgeProfileOption[]
  pinnedSessionIds: readonly string[]
  selectedCollectionIds: string[]
  sessionSections: KnowledgeSessionHistorySection[]
  selectedSessionId: string | null
  sessions: KnowledgeSessionRecord[]
  topK: number | null
  finalK: number | null
}

/**
 * The "Wissen" workspace: Ask (cited Q&A over knowledge collections),
 * Find (literal retrieval search) and Read (the document viewer
 * overlay both modes open into).
 */
export function KnowledgeWorkspace({
  historyItems,
  collections,
  composerNotice,
  dataSource,
  defaultProfileId,
  defaultTopK,
  evidenceKMax,
  rerankerProvider,
  isAskDisabled,
  isAskRunning,
  isHistoryVisible,
  isIncognito,
  items,
  mode,
  knowledgeQuestion,
  onKnowledgeQuestionChange,
  onAsk,
  onClearSession,
  onCreateSession,
  onCreateSessionGroup,
  onDemoAsk,
  onDeleteSessionGroup,
  onDeleteSession,
  onDeleteItems,
  onHistoryVisibleChange,
  onOpenDatabase,
  onIncognitoChange,
  onModeChange,
  onMoveSessionGroup,
  onMoveSessionToGroup,
  onProfileChange,
  onRenameSessionGroup,
  onRenameSession,
  onSelectSession,
  onStopAsk,
  onTogglePinnedSession,
  onSelectedCollectionIdsChange,
  onTopKChange,
  onFinalKChange,
  profileId,
  profileOptions,
  pinnedSessionIds,
  selectedCollectionIds,
  sessionSections,
  selectedSessionId,
  sessions,
  topK,
  finalK,
}: KnowledgeWorkspaceProps) {
  const { locale, t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  // Find mode keeps the overlay viewer; Ask mode uses the embedded right panel.
  const [viewerTarget, setViewerTarget] = useState<DocumentViewerTarget | null>(null)
  const [panel, setPanel] = useState<AskPanelState | null>(null)
  const [completedHandoffItemId, setCompletedHandoffItemId] = useState<string | null>(null)
  const [isItemSelectionMode, setIsItemSelectionMode] = useState(false)
  const [isRunDockExiting, setIsRunDockExiting] = useState(false)
  const [localNotice, setLocalNotice] = useState<string | null>(null)
  const [rerunTargetItemId, setRerunTargetItemId] = useState<string | null>(null)
  const [selectedItemIds, setSelectedItemIds] = useState<ReadonlySet<string>>(() => new Set())
  const completedHandoffTimeoutRef = useRef<number | null>(null)
  const previousItemStatusesRef = useRef(knowledgeItemStatusSnapshot(items))
  const threadEndRef = useRef<HTMLDivElement | null>(null)
  const itemCount = items.length
  const lastItemStatus = items[items.length - 1]?.status
  const selectedItemCount = selectedItemIds.size
  const canManageItems = items.length > 0 && !isAskRunning
  const hasKnowledgeCollections = collections.length > 0

  function scrollKnowledgeThreadToEnd(behavior: ScrollBehavior) {
    const viewports = document.querySelectorAll<HTMLElement>(
      '[data-knowledge-ask-scroll] [data-scroll-area-viewport]',
    )
    if (viewports.length > 0) {
      viewports.forEach((viewport) => {
        viewport.scrollTo({ top: viewport.scrollHeight, behavior })
      })
      return
    }
    const viewport = threadEndRef.current
      ?.closest('[data-scroll-area-viewport]') as HTMLElement | null | undefined
    if (viewport) {
      viewport.scrollTo({ top: viewport.scrollHeight, behavior })
      return
    }
    threadEndRef.current?.scrollIntoView({ behavior, block: 'end' })
  }

  useLayoutEffect(() => {
    if (mode !== 'ask') return
    const behavior: ScrollBehavior = reduceMotion || lastItemStatus === 'running' ? 'auto' : 'smooth'
    const frameId = window.requestAnimationFrame(() => scrollKnowledgeThreadToEnd(behavior))
    return () => window.cancelAnimationFrame(frameId)
  }, [itemCount, lastItemStatus, mode, reduceMotion])

  useEffect(() => {
    if (mode !== 'ask') {
      previousItemStatusesRef.current = knowledgeItemStatusSnapshot(items)
      setCompletedHandoffItemId(null)
      setIsRunDockExiting(false)
      return undefined
    }

    const completedItemId = knowledgeCompletionHandoffId({
      items,
      previousStatuses: previousItemStatusesRef.current,
    })
    previousItemStatusesRef.current = knowledgeItemStatusSnapshot(items)
    if (!completedItemId) return undefined

    setCompletedHandoffItemId(completedItemId)
    setIsRunDockExiting(true)
    if (completedHandoffTimeoutRef.current !== null) {
      window.clearTimeout(completedHandoffTimeoutRef.current)
    }
    completedHandoffTimeoutRef.current = window.setTimeout(() => {
      setCompletedHandoffItemId((current) => (current === completedItemId ? null : current))
      completedHandoffTimeoutRef.current = null
    }, reduceMotion ? 600 : KNOWLEDGE_ANSWER_ENTRY_MS)

    const behavior: ScrollBehavior = reduceMotion ? 'auto' : 'smooth'
    let secondFrameId: number | null = null
    const firstFrameId = window.requestAnimationFrame(() => {
      scrollKnowledgeThreadToEnd(behavior)
      secondFrameId = window.requestAnimationFrame(() => scrollKnowledgeThreadToEnd(behavior))
    })
    const settleTimeoutId = window.setTimeout(
      () => scrollKnowledgeThreadToEnd(behavior),
      reduceMotion ? 0 : KNOWLEDGE_ANSWER_SCROLL_SETTLE_MS,
    )

    return () => {
      window.cancelAnimationFrame(firstFrameId)
      if (secondFrameId !== null) window.cancelAnimationFrame(secondFrameId)
      window.clearTimeout(settleTimeoutId)
    }
  }, [items, mode, reduceMotion])

  useEffect(() => () => {
    if (completedHandoffTimeoutRef.current !== null) {
      window.clearTimeout(completedHandoffTimeoutRef.current)
    }
  }, [])

  const collectionTitleByBackendId = useMemo(() => {
    const map = new Map<string, string>()
    for (const collection of collections) {
      map.set(collection.collectionId, collection.title)
    }
    return map
  }, [collections])
  const collectionById = useMemo(() => {
    const map = new Map<string, KnowledgeCollectionOption>()
    for (const collection of collections) map.set(collection.id, collection)
    return map
  }, [collections])
  const collectionIdsByTitle = useMemo(() => {
    const map = new Map<string, string[]>()
    for (const collection of collections) {
      map.set(collection.title, [...(map.get(collection.title) ?? []), collection.id])
    }
    return map
  }, [collections])

  useEffect(() => {
    setIsItemSelectionMode(false)
    setSelectedItemIds(new Set())
    setRerunTargetItemId(null)
    setLocalNotice(null)
    setPanel(null)
    setViewerTarget(null)
  }, [mode, selectedSessionId, isIncognito])

  useEffect(() => {
    setPanel((current) => {
      if (!current?.item) return current
      return items.some((item) => item.id === current.item?.id) ? current : null
    })
  }, [items])

  useEffect(() => {
    if (selectedItemIds.size === 0) return
    const available = new Set(items.map((item) => item.id))
    const nextSelectedIds = new Set([...selectedItemIds].filter((itemId) => available.has(itemId)))
    if (nextSelectedIds.size === selectedItemIds.size) return
    setSelectedItemIds(nextSelectedIds)
    if (nextSelectedIds.size === 0) setIsItemSelectionMode(false)
  }, [items, selectedItemIds])

  useEffect(() => {
    if (!rerunTargetItemId) return
    if (items.some((item) => item.id === rerunTargetItemId)) return
    setRerunTargetItemId(null)
  }, [items, rerunTargetItemId])

  function targetFromReference(
    item: KnowledgeThreadItemRecord,
    reference: KnowledgeReferenceRecord,
  ): DocumentViewerTarget | null {
    if (!reference.documentId) return null
    const quote = item.answer?.quotes.find((entry) => entry.label === reference.label)
    return {
      collectionLabel: item.collectionTitles.join(' · ') || undefined,
      documentId: reference.documentId,
      // The grounding quote (verbatim cited span) highlights first; the chunk's
      // source text is the fallback span when grounding was off.
      highlightTargets: quote
        ? [quote.text]
        : reference.sourceText
          ? [reference.sourceText]
          : [],
      title: reference.title,
      excerpt: reference.excerpt ?? null,
      chunkIndex: reference.chunkIndex ?? null,
      verified: quote?.verified,
      pageNumber: reference.pageNumber ?? null,
    }
  }

  // Open the right panel on the Belege tab, focused on the clicked source.
  function openReference(item: KnowledgeThreadItemRecord, reference: KnowledgeReferenceRecord) {
    setPanel({ item, target: targetFromReference(item, reference) })
  }

  function openSnippet(hit: KnowledgeSearchHit, query: string) {
    setViewerTarget({
      collectionLabel: collectionTitleByBackendId.get(hit.collection_id) ?? undefined,
      documentId: hit.document_id,
      // The chunk text may carry a contextualization prefix that is not
      // part of the source document; the query terms act as fallback.
      highlightTargets: [hit.text, ...searchTermsFromQuery(query)],
      title: hit.document_title,
    })
  }

  const runningItem = items.find((item) => item.status === 'running') ?? null
  const latestPanelItem = [...items]
    .reverse()
    .find((item) => item.status === 'completed' && item.answer)
    ?? null
  const effectiveHistoryVisible = isHistoryVisible && !isIncognito
  const rerunTargetItem = rerunTargetItemId
    ? items.find((item) => item.id === rerunTargetItemId) ?? null
    : null
  const selectedItemCountLabel = selectedItemCount === 1
    ? t.knowledge.itemsSelectedOne
    : t.knowledge.itemsSelectedOther

  const historyPanel = (
    <KnowledgeHistoryPanel
      items={historyItems}
      onCreateSessionGroup={onCreateSessionGroup}
      onDeleteSessionGroup={onDeleteSessionGroup}
      onCreateSession={onCreateSession}
      onDeleteSession={onDeleteSession}
      onMoveSessionGroup={onMoveSessionGroup}
      onMoveSessionToGroup={onMoveSessionToGroup}
      onRenameSessionGroup={onRenameSessionGroup}
      onRenameSession={onRenameSession}
      onSelectSession={onSelectSession}
      onTogglePinnedSession={onTogglePinnedSession}
      pinnedSessionIds={pinnedSessionIds}
      reduceMotion={reduceMotion}
      selectedSessionId={selectedSessionId}
      sections={sessionSections}
      sessions={sessions}
    />
  )

  const activePanelItem = panel?.item
    ? items.find((item) => item.id === panel.item?.id) ?? null
    : null
  const activePanel = activePanelItem ? { item: activePanelItem, target: panel?.target ?? null } : null
  const sourcePanel = activePanel ? (
    <KnowledgeSourcePanel
      dataSource={dataSource}
      item={activePanel.item}
      onClose={() => setPanel(null)}
      onSelectReference={(reference) =>
        setPanel((prev) => (prev?.item ? { ...prev, target: targetFromReference(prev.item, reference) } : prev))}
      target={activePanel.target}
    />
  ) : null

  function openLatestPanel() {
    if (latestPanelItem) setPanel({ item: latestPanelItem, target: null })
  }

  function deleteSelectedSession() {
    if (!selectedSessionId || isIncognito || isAskRunning) return
    onDeleteSession(selectedSessionId)
  }

  function toggleItemSelectionMode() {
    setIsItemSelectionMode((current) => {
      if (current) {
        setSelectedItemIds(new Set())
        return false
      }
      if (!canManageItems) return false
      return true
    })
  }

  function toggleSelectedItem(itemId: string) {
    if (!isItemSelectionMode) return
    setSelectedItemIds((current) => {
      const next = new Set(current)
      if (next.has(itemId)) {
        next.delete(itemId)
      } else {
        next.add(itemId)
      }
      return next
    })
  }

  function deleteSelectedItems() {
    if (selectedItemIds.size === 0) return
    onDeleteItems([...selectedItemIds])
    if (rerunTargetItemId && selectedItemIds.has(rerunTargetItemId)) {
      setRerunTargetItemId(null)
    }
    setSelectedItemIds(new Set())
    setIsItemSelectionMode(false)
  }

  function clearCurrentSession() {
    setRerunTargetItemId(null)
    setLocalNotice(null)
    setPanel(null)
    setViewerTarget(null)
    onClearSession()
  }

  function collectionIdsForItem(item: KnowledgeThreadItemRecord): string[] | null {
    if (item.collectionIds?.length) {
      const existing = item.collectionIds.filter((collectionId) => collectionById.has(collectionId))
      if (existing.length === item.collectionIds.length) return existing
    }
    if (item.collectionTitles.length === 0) return null
    const resolved: string[] = []
    for (const title of item.collectionTitles) {
      const ids = collectionIdsByTitle.get(title) ?? []
      if (ids.length !== 1) return null
      resolved.push(ids[0])
    }
    return resolved
  }

  function loadItemForEdit(item: KnowledgeThreadItemRecord) {
    const resolvedCollectionIds = collectionIdsForItem(item)
    setRerunTargetItemId(item.id)
    onKnowledgeQuestionChange(item.question)
    onProfileChange(item.requestedProfile)
    onTopKChange(item.topK ?? null)
    onFinalKChange(item.finalK ?? null)
    if (resolvedCollectionIds) {
      onSelectedCollectionIdsChange(resolvedCollectionIds)
      setLocalNotice(null)
    } else {
      setLocalNotice(t.knowledge.rerunScopeMissing)
    }
  }

  function cancelItemEdit() {
    setRerunTargetItemId(null)
    setLocalNotice(null)
  }

  function replayItem(item: KnowledgeThreadItemRecord) {
    const resolvedCollectionIds = collectionIdsForItem(item)
    if (!resolvedCollectionIds) {
      loadItemForEdit(item)
      return
    }
    setLocalNotice(null)
    setRerunTargetItemId(null)
    onAsk(item.question, {
      collectionIds: resolvedCollectionIds,
      profileId: item.requestedProfile,
      replaceItemId: item.id,
      topK: item.topK ?? null,
      finalK: item.finalK ?? null,
    })
  }

  function submitKnowledgeQuestion(question: string) {
    const replaceItemId = rerunTargetItemId ?? undefined
    setLocalNotice(null)
    setRerunTargetItemId(null)
    onAsk(question, replaceItemId ? { replaceItemId } : undefined)
  }

  const selectedSession = selectedSessionId ? sessions.find((session) => session.id === selectedSessionId) ?? null : null
  const pageTitle = mode === 'ask'
    ? isIncognito
      ? t.knowledge.title
      : selectedSession?.title ?? t.knowledge.title
    : t.knowledge.title

  const workspaceHeader = (
    <div
      className={cn(
        'z-10 flex inqtrix-panel-header items-center justify-between gap-2 border-b border-border bg-background px-4 transition-colors md:px-6',
        mode === 'ask' && isIncognito && 'inqtrix-chat-header--incognito',
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
        {mode === 'ask' && !isIncognito && (
          <PanelToggle
            collapseLabel={t.knowledge.hideSessions}
            expandLabel={t.knowledge.showSessions}
            expanded={effectiveHistoryVisible}
            onToggle={onHistoryVisibleChange}
            side="left"
          />
        )}
        <BookOpenCheck className="size-4 shrink-0 text-foreground/80" />
        <div className="min-w-0 flex-1 overflow-hidden">
          <div className="flex min-w-0 items-center gap-2 overflow-hidden">
            <h1 className="truncate t-section text-foreground">
              {pageTitle}
            </h1>
            {mode === 'ask' && isIncognito && (
              <Badge
                className="max-w-[min(44vw,24rem)] shrink border-brand/25 bg-brand-subtle text-brand hover:bg-brand-subtle"
                title={t.knowledge.incognitoActive}
                variant="outline"
              >
                <span className="truncate">{t.knowledge.incognitoActive}</span>
              </Badge>
            )}
          </div>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {onDemoAsk && mode === 'ask' && (
          <Button
            className="gap-1.5"
            disabled={isAskRunning}
            onClick={onDemoAsk}
            size="sm"
            type="button"
            variant="outline"
          >
            <Sparkles className="icon-sm" />
            {t.knowledge.demoSearchStart}
          </Button>
        )}
        {mode === 'ask' && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={t.knowledge.incognito}
                aria-pressed={isIncognito}
                className={cn(
                  'size-7 text-foreground/75',
                  isIncognito && 'bg-brand-subtle text-brand hover:bg-brand-subtle',
                )}
                disabled={isAskRunning}
                onClick={() => onIncognitoChange(!isIncognito)}
                size="icon"
                type="button"
                variant={isIncognito ? 'secondary' : 'ghost'}
              >
                <EyeOff className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.knowledge.incognito}</TooltipContent>
          </Tooltip>
        )}
        {mode === 'ask' && (
          <div className="flex h-7 overflow-hidden rounded-md border border-border bg-card shadow-[0_1px_2px_var(--shadow-hairline)]">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  aria-label={isItemSelectionMode ? t.knowledge.exitItemSelection : t.knowledge.selectItems}
                  aria-pressed={isItemSelectionMode}
                  className={cn(
                    'h-7 w-7 rounded-none border-r border-border text-foreground/75 hover:text-foreground',
                    isItemSelectionMode && 'bg-brand-subtle text-brand hover:bg-brand-subtle hover:text-brand',
                  )}
                  disabled={!canManageItems && !isItemSelectionMode}
                  onClick={toggleItemSelectionMode}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  <ListChecks className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {isItemSelectionMode ? t.knowledge.exitItemSelection : t.knowledge.selectItems}
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  aria-label={t.knowledge.clearSession}
                  className="h-7 w-7 rounded-none border-r border-border text-foreground/75 hover:text-foreground"
                  disabled={items.length === 0 || isAskRunning}
                  onClick={clearCurrentSession}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  <Eraser className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{t.knowledge.clearSession}</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
	                  aria-label={t.knowledge.deleteSession}
	                  className="h-7 w-7 rounded-none text-foreground/75 hover:text-destructive"
	                  disabled={!selectedSessionId || isIncognito || isAskRunning}
                  onClick={deleteSelectedSession}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  <Trash2 className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{t.knowledge.deleteSession}</TooltipContent>
            </Tooltip>
          </div>
        )}
        <div className="grid h-7 grid-cols-2 rounded-md bg-surface p-0.5">
          {(['ask', 'find'] as const).map((modeKey) => {
            const isActive = mode === modeKey
            return (
              <button
                aria-pressed={isActive}
                className={cn(
                  'inline-flex items-center justify-center gap-1.5 rounded px-3 text-xs font-medium transition-colors',
                  isActive
                    ? 'bg-background text-foreground shadow-[0_1px_2px_var(--shadow-hairline)]'
                    : 'text-muted-foreground hover:text-foreground',
                )}
                key={modeKey}
                onClick={() => onModeChange(modeKey)}
                type="button"
              >
                {modeKey === 'ask' ? t.knowledge.modeAsk : t.knowledge.modeFind}
              </button>
            )
          })}
        </div>
        {mode === 'ask' && (sourcePanel || latestPanelItem) && (
          <PanelToggle
            collapseLabel={t.knowledge.panelCollapse}
            expandLabel={t.knowledge.showPanel}
            expanded={Boolean(sourcePanel)}
            onToggle={(next) => (next ? openLatestPanel() : setPanel(null))}
            side="right"
          />
        )}
      </div>
    </div>
  )

  const askCenter = (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col">
      {workspaceHeader}
      <AnimatePresence initial={false}>
        {isItemSelectionMode && (
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
                  {selectedItemCount} {selectedItemCountLabel}
                </span>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      aria-label={t.knowledge.deleteSelectedItems}
                      className="h-8 w-8 text-foreground/75 hover:text-destructive"
                      disabled={selectedItemCount === 0 || isAskRunning}
                      onClick={deleteSelectedItems}
                      size="icon"
                      type="button"
                      variant="ghost"
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{t.knowledge.deleteSelectedItems}</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      aria-label={t.knowledge.exitItemSelection}
                      className="h-8 w-8 text-foreground/75 hover:text-foreground"
                      onClick={toggleItemSelectionMode}
                      size="icon"
                      type="button"
                      variant="ghost"
                    >
                      <X className="size-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{t.knowledge.exitItemSelection}</TooltipContent>
                </Tooltip>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <ScrollArea
        className={cn(
          'min-h-0 flex-1',
          items.length === 0 && '[&_[data-scroll-area-viewport]>div]:h-full',
        )}
        data-knowledge-ask-scroll=""
      >
        <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col gap-5 px-4 py-6 md:px-8">
          {items.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center px-6 py-8 text-center">
              <WelcomeState
                actions={!hasKnowledgeCollections && onOpenDatabase ? (
                  <Button
                    className="h-8 gap-1.5 rounded-md bg-brand px-3 text-xs text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground"
                    onClick={onOpenDatabase}
                    type="button"
                    variant="default"
                  >
                    <Database className="icon-sm" />
                    {t.knowledge.openDatabase}
                  </Button>
                ) : null}
                body={(
                  <>
                    <p>
                      {hasKnowledgeCollections
                        ? t.knowledge.emptyBody
                        : t.knowledge.emptyNoCollectionsHint}
                    </p>
                    {hasKnowledgeCollections ? <p>{t.knowledge.emptyGuidance}</p> : null}
                  </>
                )}
                example={hasKnowledgeCollections ? t.knowledge.emptyExample : undefined}
                kicker={t.knowledge.emptyKicker}
                subtitle={hasKnowledgeCollections ? t.knowledge.emptyHint : t.knowledge.emptyNoCollections}
                title={t.knowledge.emptyTitle}
              />
            </div>
          ) : (
            items.map((item) => {
              const isSelected = selectedItemIds.has(item.id)
              const selectionLabel = isSelected ? t.knowledge.itemSelected : t.knowledge.selectItem
              const selectionControl = isItemSelectionMode ? (
                <KnowledgeSelectionControl
                  isSelected={isSelected}
                  label={isSelected ? t.knowledge.deselectItem : t.knowledge.selectItem}
                  onToggle={() => toggleSelectedItem(item.id)}
                />
              ) : null
              const selectionRowClassName = cn(
                'group/knowledge -mx-4 rounded-xl border border-transparent px-4 py-1 transition-colors md:-mx-8 md:px-8',
                isItemSelectionMode && 'cursor-pointer hover:border-border/70 hover:bg-surface/70',
                isSelected && 'border-brand/25 bg-brand-subtle/80 ring-1 ring-brand/20 hover:bg-brand-subtle/80',
              )
              return (
                <div
                  aria-selected={isItemSelectionMode ? isSelected : undefined}
                  className={selectionRowClassName}
                  key={item.id}
                  onClick={isItemSelectionMode ? () => toggleSelectedItem(item.id) : undefined}
                >
                  <div className="flex min-w-0 items-start gap-3">
                    {selectionControl}
                    <div className="min-w-0 flex-1 space-y-3">
                      <KnowledgeQuestionBubble
                        collectionLabel={item.collectionTitles.join(' · ')}
                        copiedLabel={t.knowledge.questionCopied}
                        copyLabel={t.knowledge.copyQuestion}
                        disabled={isAskRunning || item.status === 'running'}
                        editLabel={t.knowledge.editQuestion}
                        isSelected={isSelected}
                        isSelectionMode={isItemSelectionMode}
                        onEdit={() => loadItemForEdit(item)}
                        onReplay={() => replayItem(item)}
                        question={item.question}
                        replayLabel={t.knowledge.rerunQuestion}
                        selectionLabel={selectionLabel}
                        timestampLabel={formatMessageTimestamp(item.createdAt, locale)}
                      />
                {item.status === 'completed' && item.answer ? (
                  <AnswerCard
                    answer={item.answer}
                    collectionCount={Math.max(item.collectionTitles.length, 1)}
                    completedAtLabel={formatMessageTimestamp(item.completedAt ?? item.createdAt, locale)}
                    highlightEntry={item.id === completedHandoffItemId}
                    onOpenReference={(reference) => openReference(item, reference)}
                    steps={item.progress.steps}
                  />
                ) : item.status === 'failed' || item.status === 'cancelled' ? (
                  <KnowledgeRunCard
                    collectionCount={Math.max(item.collectionTitles.length, 1)}
                    item={item}
                  />
                ) : null}
                    </div>
                  </div>
                </div>
              )
            })
          )}
          <div ref={threadEndRef} />
        </div>
      </ScrollArea>
      <div className="z-10 shrink-0 px-3 pb-4 pt-2 md:px-6">
        <div className="mx-auto max-w-5xl">
          <AnimatePresence initial={false} onExitComplete={() => setIsRunDockExiting(false)}>
            {runningItem && (
              <motion.div
                animate={{ height: 'auto', opacity: 1, y: 0 }}
                className="overflow-hidden"
                exit={reduceMotion ? { height: 0, opacity: 0 } : { filter: 'blur(2px)', height: 0, opacity: 0, y: 10 }}
                initial={false}
                key={runningItem.id}
                transition={appMotion.panel}
              >
                <KnowledgeRunCard
                  collectionCount={Math.max(runningItem.collectionTitles.length, 1)}
                  item={runningItem}
                  presentation="dock"
                />
              </motion.div>
            )}
          </AnimatePresence>
          <KnowledgeComposer
            className="w-full"
            collections={collections}
            connectedTop={Boolean(runningItem) || isRunDockExiting}
            defaultProfileId={defaultProfileId}
            defaultTopK={defaultTopK}
            evidenceKMax={evidenceKMax}
            rerankerProvider={rerankerProvider}
            disabled={isAskDisabled}
            running={isAskRunning}
            notice={localNotice ?? composerNotice}
            draftQuestion={knowledgeQuestion}
            isReplacing={Boolean(rerunTargetItem)}
            onCancelReplace={cancelItemEdit}
            onDraftQuestionChange={onKnowledgeQuestionChange}
            onProfileChange={onProfileChange}
            onSelectedCollectionIdsChange={onSelectedCollectionIdsChange}
            onStop={onStopAsk}
            onSubmit={submitKnowledgeQuestion}
            onTopKChange={onTopKChange}
            onFinalKChange={onFinalKChange}
            profileOptions={profileOptions}
            selectedCollectionIds={selectedCollectionIds}
            selectedProfileId={profileId}
            topK={topK}
            finalK={finalK}
          />
        </div>
      </div>
    </div>
  )

  const findCenter = (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col">
      {workspaceHeader}
      <KnowledgeFindMode
        collections={collections}
        collectionTitleFor={(collectionId) => collectionTitleByBackendId.get(collectionId) ?? null}
        dataSource={dataSource}
        onOpenSnippet={openSnippet}
        onSelectedCollectionIdsChange={onSelectedCollectionIdsChange}
        selectedCollectionIds={selectedCollectionIds}
      />
    </div>
  )

  const centerAndRight = (
    <ResizablePanelGroup className="min-h-0 w-full overflow-hidden" orientation="horizontal">
      <ResizablePanel className="min-h-0 min-w-0 overflow-hidden" defaultSize={sourcePanel ? '64%' : '100%'} minSize="48%">
        {askCenter}
      </ResizablePanel>
      {sourcePanel && (
        <>
          <ResizableHandle aria-label={t.knowledge.panelResize} />
          <ResizablePanel className="min-h-0 min-w-0 overflow-hidden" defaultSize="36%" maxSize="48%" minSize="28%">
            {sourcePanel}
          </ResizablePanel>
        </>
      )}
    </ResizablePanelGroup>
  )

  const askDesktop = effectiveHistoryVisible ? (
    <div className="hidden min-h-0 w-full flex-1 overflow-hidden bg-background lg:flex">
      <ResizablePanelGroup className="min-h-0 min-w-0 flex-1 overflow-hidden" orientation="horizontal">
        <ResizablePanel className="min-h-0 min-w-0 overflow-hidden" defaultSize="26%" maxSize="42%" minSize="18%">
          {historyPanel}
        </ResizablePanel>
        <ResizableHandle aria-label={t.knowledge.resizeSessions} />
        <ResizablePanel className="min-h-0 min-w-0 overflow-hidden" defaultSize="74%" minSize="58%">
          {centerAndRight}
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  ) : (
    <div className="hidden min-h-0 w-full flex-1 overflow-hidden bg-background lg:flex">
      <div className="min-h-0 min-w-0 flex-1 overflow-hidden">{centerAndRight}</div>
    </div>
  )

  const askMobile = (
    <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden lg:hidden">
      {effectiveHistoryVisible && <div className="h-56 shrink-0 border-b border-border">{historyPanel}</div>}
      <div className="min-h-0 flex-1 overflow-hidden">{askCenter}</div>
      {sourcePanel && (
        <div className="absolute inset-0 z-30 bg-background">
          {sourcePanel}
        </div>
      )}
    </div>
  )

  return (
    <section className="flex min-h-[620px] min-w-0 flex-col bg-background lg:h-full lg:min-h-0 lg:overflow-hidden">
      {mode === 'ask' ? (
        <>
          {askDesktop}
          {askMobile}
        </>
      ) : (
        findCenter
      )}

      {viewerTarget && (
        <DocumentViewer
          dataSource={dataSource}
          onClose={() => setViewerTarget(null)}
          target={viewerTarget}
        />
      )}
    </section>
  )
}

function KnowledgeQuestionBubble({
  collectionLabel,
  copiedLabel,
  copyLabel,
  disabled,
  editLabel,
  isSelected,
  isSelectionMode,
  onEdit,
  onReplay,
  question,
  replayLabel,
  selectionLabel,
  timestampLabel,
}: {
  collectionLabel: string
  copiedLabel: string
  copyLabel: string
  disabled: boolean
  editLabel: string
  isSelected: boolean
  isSelectionMode: boolean
  onEdit: () => void
  onReplay: () => void
  question: string
  replayLabel: string
  selectionLabel: string
  timestampLabel: string
}) {
  const [copied, setCopied] = useState(false)

  async function copyQuestion() {
    try {
      await navigator.clipboard.writeText(question)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1200)
    } catch (error) {
      console.warn('Inqtrix knowledge question copy failed.', error)
    }
  }

  return (
    <div className="flex min-w-0 justify-end">
      <div className="min-w-0 max-w-[min(80%,40rem)]">
        <div className="inqtrix-user-bubble rounded-lg px-3 py-2.5 text-sm leading-6 shadow-[0_1px_2px_var(--shadow-hairline)]">
          {question}
        </div>
        <div className="mt-1 flex min-w-0 items-center justify-end gap-1 t-meta-sm text-muted-foreground">
          <span className="shrink-0 whitespace-nowrap tabular-nums">{timestampLabel}</span>
          {collectionLabel && (
            <>
              <span className="shrink-0 text-muted-foreground/45">·</span>
              <Database className="icon-xs shrink-0" />
              <span className="min-w-0 truncate">{collectionLabel}</span>
            </>
          )}
          {isSelectionMode ? (
            <KnowledgeSelectionPill isSelected={isSelected} label={selectionLabel} />
          ) : (
            <div className="ml-0.5 flex shrink-0 items-center gap-0.5">
              <KnowledgeQuestionAction
                icon={copied ? <Check className="size-3" /> : <Copy className="size-3" />}
                label={copied ? copiedLabel : copyLabel}
                onClick={() => void copyQuestion()}
                success={copied}
              />
              <KnowledgeQuestionAction
                disabled={disabled}
                icon={<PencilLine className="size-3" />}
                label={editLabel}
                onClick={onEdit}
              />
              <KnowledgeQuestionAction
                disabled={disabled}
                icon={<SendHorizontal className="size-3" />}
                label={replayLabel}
                onClick={onReplay}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function KnowledgeQuestionAction({
  disabled = false,
  icon,
  label,
  onClick,
  success = false,
}: {
  disabled?: boolean
  icon: ReactNode
  label: string
  onClick: () => void
  success?: boolean
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          className={cn(
            'size-6 text-foreground/65 hover:text-foreground',
            success && 'text-success hover:text-success',
          )}
          disabled={disabled}
          onClick={(event) => {
            event.stopPropagation()
            onClick()
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          {icon}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function KnowledgeSelectionPill({
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

function KnowledgeSelectionControl({
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

function KnowledgeFindMode({
  collections,
  collectionTitleFor,
  dataSource,
  onOpenSnippet,
  onSelectedCollectionIdsChange,
  selectedCollectionIds,
}: {
  collections: KnowledgeCollectionOption[]
  collectionTitleFor: (collectionId: string) => string | null
  dataSource: KnowledgeDataSource
  onOpenSnippet: (hit: KnowledgeSearchHit, query: string) => void
  onSelectedCollectionIdsChange: (ids: string[]) => void
  selectedCollectionIds: string[]
}) {
  const { t } = useLocale()
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<KnowledgeSearchHit[]>([])
  const [state, setState] = useState<FindResultsState>('idle')
  const [error, setError] = useState<string | null>(null)
  const requestIdRef = useRef(0)

  const backendCollectionIds = useMemo(
    () => collections
      .filter((collection) => selectedCollectionIds.includes(collection.id))
      .map((collection) => collection.collectionId),
    [collections, selectedCollectionIds],
  )

  useEffect(() => {
    const trimmed = query.trim()
    if (trimmed === '') {
      setState('idle')
      setHits([])
      return undefined
    }
    if (trimmed.length < FIND_MIN_QUERY_LENGTH) {
      setState('short')
      setHits([])
      return undefined
    }

    setState('searching')
    const requestId = (requestIdRef.current += 1)
    const timeoutId = window.setTimeout(() => {
      dataSource
        .search(trimmed, backendCollectionIds, FIND_TOP_K)
        .then((nextHits) => {
          if (requestIdRef.current !== requestId) return
          setHits(nextHits)
          setError(null)
          setState('ready')
        })
        .catch((searchError: unknown) => {
          if (requestIdRef.current !== requestId) return
          setHits([])
          setError(searchError instanceof Error ? searchError.message : null)
          setState('error')
        })
    }, FIND_DEBOUNCE_MS)
    return () => window.clearTimeout(timeoutId)
  }, [backendCollectionIds, dataSource, query])

  const groups = useMemo(() => groupHitsByDocument(hits), [hits])

  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className="mx-auto w-full max-w-5xl px-4 py-5 md:px-8">
        <div className="flex h-9 items-center gap-2 rounded-md border border-border bg-card px-2.5 shadow-[0_1px_2px_var(--shadow-hairline)] transition-[border-color] focus-within:border-brand/60">
          <Search className="icon-sm shrink-0 text-muted-foreground/70" />
          <input
            aria-label={t.knowledge.findPlaceholder}
            className="h-full min-w-0 flex-1 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground/70"
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t.knowledge.findPlaceholder}
            type="search"
            value={query}
          />
        </div>

        {collections.length > 0 && (
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            {collections.map((collection) => {
              const isActive = selectedCollectionIds.includes(collection.id)
              return (
                <CollectionFilterChip
                  isActive={isActive}
                  key={collection.id}
                  onToggle={() => onSelectedCollectionIdsChange(
                    isActive
                      ? selectedCollectionIds.filter((id) => id !== collection.id)
                      : [...selectedCollectionIds, collection.id],
                  )}
                  title={collection.title}
                />
              )
            })}
          </div>
        )}

        <div className="mt-4">
          <DocumentFindResults
            collectionTitleFor={collectionTitleFor}
            error={error}
            groups={groups}
            onOpenSnippet={(hit) => onOpenSnippet(hit, query)}
            query={query}
            state={state}
          />
        </div>
      </div>
    </ScrollArea>
  )
}

function CollectionFilterChip({
  isActive,
  onToggle,
  title,
}: {
  isActive: boolean
  onToggle: () => void
  title: string
}) {
  return (
    <Chip active={isActive} dot={isActive ? 'bg-brand' : undefined} onClick={onToggle}>
      <span className="max-w-44 truncate">{title}</span>
    </Chip>
  )
}
