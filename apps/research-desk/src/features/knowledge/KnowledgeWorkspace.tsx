import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { BookOpenCheck, Database, Search, Sparkles } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Chip } from '@/components/ui/chip'
import { PanelRail } from '@/components/ui/panel-rail'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useLocale } from '@/i18n/LocaleProvider'
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

/** The embedded right panel of the Ask view: which answer's sources/steps it
 * shows, the active tab, and the document currently open in the reader. */
type AskPanelState = {
  item: KnowledgeThreadItemRecord | null
  tab: 'sources' | 'steps'
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
  isAskDisabled: boolean
  isAskRunning: boolean
  isHistoryVisible: boolean
  items: KnowledgeThreadItemRecord[]
  mode: KnowledgeMode
  knowledgeQuestion: string
  onKnowledgeQuestionChange: (question: string) => void
  onAsk: (question: string) => void
  onCreateSession: (groupId?: string | null) => void
  onCreateSessionGroup: () => void
  onDemoAsk?: () => void
  onDeleteSessionGroup: (groupId: string) => void
  onDeleteSession: (sessionId: string) => void
  onHistoryVisibleChange: (visible: boolean) => void
  onModeChange: (mode: KnowledgeMode) => void
  onMoveSessionGroup: (groupId: string, targetIndex: number) => void
  onMoveSessionToGroup: (sessionId: string, groupId: string | null, targetIndex: number) => void
  onProfileChange: (profileId: string | null) => void
  onRenameSessionGroup: (groupId: string, title: string) => void
  onRenameSession: (sessionId: string, title: string) => void
  onSelectSession: (sessionId: string) => void
  onSelectedCollectionIdsChange: (ids: string[]) => void
  onTopKChange: (topK: number | null) => void
  profileId: string | null
  profileOptions: KnowledgeProfileOption[]
  selectedCollectionIds: string[]
  sessionSections: KnowledgeSessionHistorySection[]
  selectedSessionId: string | null
  sessions: KnowledgeSessionRecord[]
  topK: number | null
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
  isAskDisabled,
  isAskRunning,
  isHistoryVisible,
  items,
  mode,
  knowledgeQuestion,
  onKnowledgeQuestionChange,
  onAsk,
  onCreateSession,
  onCreateSessionGroup,
  onDemoAsk,
  onDeleteSessionGroup,
  onDeleteSession,
  onHistoryVisibleChange,
  onModeChange,
  onMoveSessionGroup,
  onMoveSessionToGroup,
  onProfileChange,
  onRenameSessionGroup,
  onRenameSession,
  onSelectSession,
  onSelectedCollectionIdsChange,
  onTopKChange,
  profileId,
  profileOptions,
  selectedCollectionIds,
  sessionSections,
  selectedSessionId,
  sessions,
  topK,
}: KnowledgeWorkspaceProps) {
  const { t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  // Find mode keeps the overlay viewer; Ask mode uses the embedded right panel.
  const [viewerTarget, setViewerTarget] = useState<DocumentViewerTarget | null>(null)
  const [panel, setPanel] = useState<AskPanelState | null>(null)
  const [completedHandoffItemId, setCompletedHandoffItemId] = useState<string | null>(null)
  const [isRunDockExiting, setIsRunDockExiting] = useState(false)
  const completedHandoffTimeoutRef = useRef<number | null>(null)
  const previousItemStatusesRef = useRef(knowledgeItemStatusSnapshot(items))
  const threadEndRef = useRef<HTMLDivElement | null>(null)
  const itemCount = items.length
  const lastItemStatus = items[items.length - 1]?.status

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
    setPanel({ item, tab: 'sources', target: targetFromReference(item, reference) })
  }

  // Open the right panel on the Schritte tab to review this answer's agent run.
  function openSteps(item: KnowledgeThreadItemRecord) {
    setPanel((prev) => ({
      item,
      tab: 'steps',
      target: prev && prev.item?.id === item.id ? prev.target : null,
    }))
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

  const historyPanel = (
    <KnowledgeHistoryPanel
      items={historyItems}
      onCreateSessionGroup={onCreateSessionGroup}
      onDeleteSessionGroup={onDeleteSessionGroup}
      onCreateSession={onCreateSession}
      onDeleteSession={onDeleteSession}
      onHide={() => onHistoryVisibleChange(false)}
      onMoveSessionGroup={onMoveSessionGroup}
      onMoveSessionToGroup={onMoveSessionToGroup}
      onRenameSessionGroup={onRenameSessionGroup}
      onRenameSession={onRenameSession}
      onSelectSession={onSelectSession}
      reduceMotion={reduceMotion}
      selectedSessionId={selectedSessionId}
      sections={sessionSections}
      sessions={sessions}
    />
  )

  const sourcePanel = panel ? (
    <KnowledgeSourcePanel
      collectionCount={Math.max(panel.item?.collectionTitles.length ?? 1, 1)}
      dataSource={dataSource}
      item={panel.item}
      onClose={() => setPanel(null)}
      onSelectReference={(reference) =>
        setPanel((prev) => (prev?.item ? { ...prev, target: targetFromReference(prev.item, reference) } : prev))}
      onTabChange={(nextTab) => setPanel((prev) => (prev ? { ...prev, tab: nextTab } : prev))}
      tab={panel.tab}
      target={panel.target}
    />
  ) : null

  function openLatestPanel() {
    if (latestPanelItem) setPanel({ item: latestPanelItem, tab: 'sources', target: null })
  }

  const selectedSession = selectedSessionId ? sessions.find((session) => session.id === selectedSessionId) ?? null : null
  const selectedSessionQuestionCount = selectedSessionId
    ? historyItems.filter((item) => item.sessionId === selectedSessionId).length
    : 0
  const pageTitle = mode === 'ask'
    ? selectedSession?.title ?? t.knowledge.title
    : t.knowledge.title
  const pageSubtitle = mode === 'ask'
    ? selectedSession
      ? t.knowledge.sessionItemCount.replace('{count}', String(selectedSessionQuestionCount))
      : t.knowledge.emptyHint
    : t.knowledge.findStartHint

  const workspaceHeader = (
    <div className="z-10 flex h-12 shrink-0 items-center justify-between gap-2 border-b border-border bg-background px-4 md:px-6">
      <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
        <BookOpenCheck className="size-4 shrink-0 text-foreground/80" />
        <div className="min-w-0 flex-1 overflow-hidden">
          <h1 className="truncate t-section text-foreground">
            {pageTitle}
          </h1>
          <p className="max-w-md truncate t-meta-sm text-muted-foreground">
            {pageSubtitle}
          </p>
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
      </div>
    </div>
  )

  const askCenter = (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col">
      {workspaceHeader}
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
              <span className="flex size-10 items-center justify-center rounded-full border border-border bg-surface text-muted-foreground">
                <BookOpenCheck className="size-5" />
              </span>
              <p className="mt-3 t-section text-foreground">{t.knowledge.emptyTitle}</p>
              <p className="mt-1 max-w-md t-meta text-muted-foreground">
                {collections.length === 0 ? t.knowledge.emptyNoCollections : t.knowledge.emptyHint}
              </p>
            </div>
          ) : (
            items.map((item) => (
              <div className="space-y-3" key={item.id}>
                <div className="flex min-w-0 justify-end">
                  <div className="min-w-0 max-w-[min(80%,40rem)]">
                    <div className="rounded-lg border border-brand/25 bg-brand px-3 py-2.5 text-sm leading-6 text-primary-foreground shadow-[0_1px_2px_var(--shadow-hairline)]">
                      {item.question}
                    </div>
                    {item.collectionTitles.length > 0 && (
                      <p className="mt-1 flex items-center justify-end gap-1 t-meta-sm text-muted-foreground">
                        <Database className="icon-xs shrink-0" />
                        <span className="truncate">{item.collectionTitles.join(' · ')}</span>
                      </p>
                    )}
                  </div>
                </div>
                {item.status === 'completed' && item.answer ? (
                  <AnswerCard
                    answer={item.answer}
                    highlightEntry={item.id === completedHandoffItemId}
                    onOpenReference={(reference) => openReference(item, reference)}
                    onOpenSteps={item.progress.steps.length > 0 ? () => openSteps(item) : undefined}
                  />
                ) : item.status === 'failed' ? (
                  <KnowledgeRunCard
                    collectionCount={Math.max(item.collectionTitles.length, 1)}
                    item={item}
                  />
                ) : null}
              </div>
            ))
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
            disabled={isAskDisabled}
            running={isAskRunning}
            notice={composerNotice}
            draftQuestion={knowledgeQuestion}
            onDraftQuestionChange={onKnowledgeQuestionChange}
            onProfileChange={onProfileChange}
            onSelectedCollectionIdsChange={onSelectedCollectionIdsChange}
            onSubmit={onAsk}
            onTopKChange={onTopKChange}
            profileOptions={profileOptions}
            selectedCollectionIds={selectedCollectionIds}
            selectedProfileId={profileId}
            topK={topK}
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
      <ResizablePanel className="min-h-0 min-w-0 overflow-hidden" defaultSize={panel ? '64%' : '100%'} minSize="48%">
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

  const rightPanelRail = !sourcePanel && latestPanelItem
    ? <PanelRail label={t.knowledge.showPanel} onExpand={openLatestPanel} side="right" />
    : null

  const askContentWithPanelRail = (
    <div className="flex h-full min-h-0 w-full overflow-hidden">
      <div className="min-h-0 min-w-0 flex-1 overflow-hidden">{centerAndRight}</div>
      {rightPanelRail}
    </div>
  )

  const askDesktop = isHistoryVisible ? (
    <div className="hidden min-h-0 w-full flex-1 overflow-hidden bg-background lg:flex">
      <ResizablePanelGroup className="min-h-0 min-w-0 flex-1 overflow-hidden" orientation="horizontal">
        <ResizablePanel className="min-h-0 min-w-0 overflow-hidden" defaultSize="26%" maxSize="42%" minSize="18%">
          {historyPanel}
        </ResizablePanel>
        <ResizableHandle aria-label={t.knowledge.resizeSessions} />
        <ResizablePanel className="min-h-0 min-w-0 overflow-hidden" defaultSize="74%" minSize="58%">
          {askContentWithPanelRail}
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  ) : (
    <div className="hidden min-h-0 w-full flex-1 overflow-hidden bg-background lg:flex">
      <PanelRail label={t.knowledge.showSessions} onExpand={() => onHistoryVisibleChange(true)} side="left" />
      <div className="min-h-0 min-w-0 flex-1 overflow-hidden">{askContentWithPanelRail}</div>
    </div>
  )

  const askMobile = (
    <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden lg:hidden">
      {isHistoryVisible && <div className="h-56 shrink-0 border-b border-border">{historyPanel}</div>}
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
