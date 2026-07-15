import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type {
  CollaborationChangeKind,
  SuggestionKind,
} from '@inqtrix/editor-schema'

import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock3,
  ListChecks,
  LoaderCircle,
  LockKeyhole,
  PanelRightClose,
  Users,
  X,
} from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Dialog } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type { CollaborationReviewDisplay } from '../tiptap'
import {
  adjacentChangeId,
  compactChangeText,
  filterInspectorChanges,
  filterInspectorHistory,
  type EditorCollaborationStatusModel,
  type EditorChangesView,
  type EditorInspectorTab,
  type EditorWriteMode,
  type InspectorChange,
  type InspectorHistoryEntry,
  type InspectorHistoryFilters,
  type InspectorOpenFilters,
  type InspectorParticipant,
} from './model'

type InspectorDecision = 'accept' | 'reject'

export type EditorInspectorProps = {
  activeTab: EditorInspectorTab
  assistant: ReactNode
  canDecide: boolean
  changes: readonly InspectorChange[]
  changesError: string | null
  changesView: EditorChangesView
  collaborationStatus: EditorCollaborationStatusModel
  decisionError: string | null
  display: CollaborationReviewDisplay
  history: readonly InspectorHistoryEntry[]
  historyError: string | null
  historyFilters: InspectorHistoryFilters
  historyLoading: boolean
  isDecisionPending: boolean
  onActiveTabChange: (tab: EditorInspectorTab) => void
  onChangesViewChange: (view: EditorChangesView) => void
  onClose: () => void
  onDecision: (decision: InspectorDecision, patchIds: string[]) => void
  onDisplayChange: (display: CollaborationReviewDisplay) => void
  onHistoryFiltersChange: (filters: InspectorHistoryFilters) => void
  onOpenFiltersChange: (filters: InspectorOpenFilters) => void
  onSelectedChangeIdChange: (changeId: string | null) => void
  openFilters: InspectorOpenFilters
  selectedChangeId: string | null
}

const copy = {
  de: {
    accept: 'Annehmen',
    acceptAll: 'Alle annehmen',
    all: 'Alle',
    allAuthors: 'Alle Personen',
    allTypes: 'Alle Arten',
    assistant: 'Assistenz',
    author: 'Person',
    batchAcceptBody: 'Alle gefilterten offenen Änderungen annehmen?',
    batchAcceptTitle: 'Änderungen annehmen',
    batchRejectBody: 'Alle gefilterten offenen Änderungen ablehnen?',
    batchRejectTitle: 'Änderungen ablehnen',
    cancel: 'Abbrechen',
    changes: 'Änderungen',
    close: 'Inspector schließen',
    status: {
      access_revoked: 'Zugriff entzogen',
      error: 'Synchronisierungsfehler',
      inactive: 'Lokal',
      read_only: 'Schreibgeschützt',
      reconnecting: 'Verbindung wird wiederhergestellt',
      saved: 'Gespeichert',
      saving: 'Wird gespeichert',
      syncing: 'Wird synchronisiert',
      update_required: 'Update erforderlich',
    },
    projection: 'Bestätigter Stand',
    participants: 'Teilnehmende',
    deletion: 'Löschung',
    display: 'Anzeige',
    edit: 'Bearbeiten',
    final: 'Final',
    history: 'Verlauf',
    insertion: 'Einfügung',
    loading: 'Änderungen werden geladen',
    modification: 'Änderung',
    next: 'Nächste Änderung',
    noChanges: 'Keine offenen Änderungen.',
    noHistory: 'Kein Verlauf für diese Filter.',
    open: 'Offen',
    original: 'Original',
    previous: 'Vorherige Änderung',
    proposed: 'Vorgeschlagen',
    reject: 'Ablehnen',
    rejectAll: 'Alle ablehnen',
    simple: 'Einfach',
    sourceReadOnly: 'Quelltext ist in der Zusammenarbeit schreibgeschützt.',
    suggest: 'Vorschlagen',
    suggestLocked: 'Diese Freigabe erlaubt nur Vorschläge.',
    type: 'Art',
    viewLocked: 'Diese Freigabe ist schreibgeschützt.',
  },
  en: {
    accept: 'Accept',
    acceptAll: 'Accept all',
    all: 'All',
    allAuthors: 'All people',
    allTypes: 'All types',
    assistant: 'Assistant',
    author: 'Person',
    batchAcceptBody: 'Accept all filtered open changes?',
    batchAcceptTitle: 'Accept changes',
    batchRejectBody: 'Reject all filtered open changes?',
    batchRejectTitle: 'Reject changes',
    cancel: 'Cancel',
    changes: 'Changes',
    close: 'Close inspector',
    status: {
      access_revoked: 'Access revoked',
      error: 'Sync error',
      inactive: 'Local',
      read_only: 'Read-only',
      reconnecting: 'Reconnecting',
      saved: 'Saved',
      saving: 'Saving',
      syncing: 'Syncing',
      update_required: 'Update required',
    },
    projection: 'Confirmed version',
    participants: 'Participants',
    deletion: 'Deletion',
    display: 'Display',
    edit: 'Edit',
    final: 'Final',
    history: 'History',
    insertion: 'Insertion',
    loading: 'Loading changes',
    modification: 'Change',
    next: 'Next change',
    noChanges: 'No open changes.',
    noHistory: 'No history for these filters.',
    open: 'Open',
    original: 'Original',
    previous: 'Previous change',
    proposed: 'Proposed',
    reject: 'Reject',
    rejectAll: 'Reject all',
    simple: 'Simple',
    sourceReadOnly: 'Source is read-only during collaboration.',
    suggest: 'Suggest',
    suggestLocked: 'This share permits suggestions only.',
    type: 'Type',
    viewLocked: 'This share is read-only.',
  },
} as const

export function EditorInspector({
  activeTab,
  assistant,
  canDecide,
  changes,
  changesError,
  changesView,
  collaborationStatus,
  decisionError,
  display,
  history,
  historyError,
  historyFilters,
  historyLoading,
  isDecisionPending,
  onActiveTabChange,
  onChangesViewChange,
  onClose,
  onDecision,
  onDisplayChange,
  onHistoryFiltersChange,
  onOpenFiltersChange,
  onSelectedChangeIdChange,
  openFilters,
  selectedChangeId,
}: EditorInspectorProps) {
  const { locale } = useLocale()
  const labels = copy[locale]
  const filteredChanges = useMemo(
    () => filterInspectorChanges(changes, openFilters),
    [changes, openFilters],
  )
  const filteredHistory = useMemo(
    () => filterInspectorHistory(history, historyFilters),
    [history, historyFilters],
  )
  const openAuthors = useMemo(
    () => uniqueParticipants(changes.map((change) => change.author)),
    [changes],
  )
  const historyActors = useMemo(
    () => uniqueParticipants(history.map((entry) => entry.actor)),
    [history],
  )
  const [expandedChangeId, setExpandedChangeId] = useState<string | null>(null)
  const [expandedHistoryId, setExpandedHistoryId] = useState<string | null>(null)
  const [batchDecision, setBatchDecision] = useState<InspectorDecision | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!selectedChangeId || activeTab !== 'changes' || changesView !== 'open') return
    listRef.current
      ?.querySelector<HTMLElement>(`[data-inspector-change-id="${CSS.escape(selectedChangeId)}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [activeTab, changesView, selectedChangeId])

  useEffect(() => {
    if (canDecide || batchDecision === null) return
    setBatchDecision(null)
  }, [batchDecision, canDecide])

  const navigate = (direction: -1 | 1) => {
    const changeId = adjacentChangeId(filteredChanges, selectedChangeId, direction)
    onSelectedChangeIdChange(changeId)
    if (changeId) setExpandedChangeId(changeId)
  }

  const confirmBatch = () => {
    if (!batchDecision || !canDecide || filteredChanges.length === 0) return
    onDecision(batchDecision, filteredChanges.map((change) => change.id))
    setBatchDecision(null)
  }

  return (
    <Tabs
      className="inqtrix-contained-panel flex h-full w-full min-w-0 flex-col bg-background"
      onValueChange={(value) => onActiveTabChange(value as EditorInspectorTab)}
      value={activeTab}
    >
      <div className="flex inqtrix-panel-header items-center gap-2 border-b border-border px-3">
        <TabsList className="grid h-8 min-w-0 flex-1 grid-cols-2 rounded-md p-0.5">
          <TabsTrigger className="h-7 min-w-0 px-2 text-xs" value="assistant">
            {labels.assistant}
          </TabsTrigger>
          <TabsTrigger className="h-7 min-w-0 gap-1 px-2 text-xs" value="changes">
            {labels.changes}
            {changes.length > 0 ? (
              <span className="t-hint tabular-nums text-muted-foreground">{changes.length}</span>
            ) : null}
          </TabsTrigger>
        </TabsList>
        <IconButton label={labels.close} onClick={onClose}>
          <PanelRightClose className="icon-sm" />
        </IconButton>
      </div>

      <EditorCollaborationStatus model={collaborationStatus} variant="inspector" />

      <TabsContent
        className="m-0 min-h-0 flex-1 data-[state=inactive]:hidden"
        forceMount
        value="assistant"
      >
        {assistant}
      </TabsContent>
      <TabsContent
        className="m-0 flex min-h-0 flex-1 flex-col data-[state=inactive]:hidden"
        forceMount
        value="changes"
      >
        <div className="flex flex-col gap-2 border-b border-border px-3 py-2">
          <div className="grid grid-cols-2 gap-0.5 rounded-md bg-muted/60 p-0.5">
            <SegmentButton
              active={changesView === 'open'}
              count={changes.length}
              label={labels.open}
              onClick={() => onChangesViewChange('open')}
            />
            <SegmentButton
              active={changesView === 'history'}
              label={labels.history}
              onClick={() => onChangesViewChange('history')}
            />
          </div>

          {changesView === 'open' ? (
            <>
              <div aria-label={labels.display} className="grid grid-cols-4 gap-0.5 rounded-md bg-muted/60 p-0.5">
                {(['simple', 'all', 'final', 'original'] as const).map((mode) => (
                  <SegmentButton
                    active={display === mode}
                    key={mode}
                    label={labels[mode]}
                    onClick={() => onDisplayChange(mode)}
                  />
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <InspectorSelect
                  label={labels.author}
                  onValueChange={(value) => onOpenFiltersChange({
                    ...openFilters,
                    authorId: value === 'all' ? null : value,
                  })}
                  options={openAuthors.map((participant) => ({
                    label: participant.name,
                    value: participant.id,
                  }))}
                  placeholder={labels.allAuthors}
                  value={openFilters.authorId ?? 'all'}
                />
                <InspectorSelect
                  label={labels.type}
                  onValueChange={(value) => onOpenFiltersChange({
                    ...openFilters,
                    type: value === 'all' ? null : value as SuggestionKind,
                  })}
                  options={suggestionTypeOptions(labels)}
                  placeholder={labels.allTypes}
                  value={openFilters.type ?? 'all'}
                />
              </div>
              <div className="flex h-7 items-center justify-between gap-2">
                <span className="t-meta-sm truncate text-muted-foreground">
                  {filteredChanges.length} {labels.open.toLocaleLowerCase(locale)}
                </span>
                <div className="flex shrink-0 items-center gap-0.5">
                  <IconButton
                    disabled={filteredChanges.length === 0}
                    label={labels.previous}
                    onClick={() => navigate(-1)}
                  >
                    <ChevronLeft className="icon-sm" />
                  </IconButton>
                  <IconButton
                    disabled={filteredChanges.length === 0}
                    label={labels.next}
                    onClick={() => navigate(1)}
                  >
                    <ChevronRight className="icon-sm" />
                  </IconButton>
                </div>
              </div>
            </>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              <InspectorSelect
                label={labels.author}
                onValueChange={(value) => onHistoryFiltersChange({
                  ...historyFilters,
                  actorId: value === 'all' ? null : value,
                })}
                options={historyActors.map((participant) => ({
                  label: participant.name,
                  value: participant.id,
                }))}
                placeholder={labels.allAuthors}
                value={historyFilters.actorId ?? 'all'}
              />
              <InspectorSelect
                label={labels.type}
                onValueChange={(value) => onHistoryFiltersChange({
                  ...historyFilters,
                  type: value === 'all' ? null : value as CollaborationChangeKind,
                })}
                options={historyTypeOptions(labels)}
                placeholder={labels.allTypes}
                value={historyFilters.type ?? 'all'}
              />
            </div>
          )}
        </div>

        <ScrollArea className="min-h-0 flex-1">
          <div ref={listRef}>
            {changesView === 'open' ? (
              <OpenChanges
                changes={filteredChanges}
                canDecide={canDecide}
                error={changesError ?? decisionError}
                expandedId={expandedChangeId}
                isDecisionPending={isDecisionPending}
                labels={labels}
                locale={locale}
                onDecision={onDecision}
                onExpandedChange={setExpandedChangeId}
                onSelectedChange={onSelectedChangeIdChange}
                selectedId={selectedChangeId}
              />
            ) : (
              <HistoryChanges
                entries={filteredHistory}
                error={historyError}
                expandedId={expandedHistoryId}
                isLoading={historyLoading}
                labels={labels}
                locale={locale}
                onExpandedChange={setExpandedHistoryId}
              />
            )}
          </div>
        </ScrollArea>

        {changesView === 'open' && filteredChanges.length > 0 ? (
          <div className="grid grid-cols-2 gap-2 border-t border-border p-3">
            <Button
              disabled={isDecisionPending || !canDecide}
              onClick={() => setBatchDecision('reject')}
              size="sm"
              type="button"
              variant="outline"
            >
              {canDecide ? <X /> : <LockKeyhole />}
              {labels.rejectAll}
            </Button>
            <Button
              disabled={isDecisionPending || !canDecide}
              onClick={() => setBatchDecision('accept')}
              size="sm"
              type="button"
            >
              {canDecide ? <Check /> : <LockKeyhole />}
              {labels.acceptAll}
            </Button>
          </div>
        ) : null}

        <Dialog
          closeLabel={labels.cancel}
          description={batchDecision === 'accept' ? labels.batchAcceptBody : labels.batchRejectBody}
          dismissable={!isDecisionPending}
          footer={(
            <>
              <Button onClick={() => setBatchDecision(null)} size="sm" type="button" variant="outline">
                {labels.cancel}
              </Button>
              <Button
                disabled={isDecisionPending || !canDecide}
                onClick={confirmBatch}
                size="sm"
                type="button"
                variant={batchDecision === 'reject' ? 'destructive' : 'default'}
              >
                {isDecisionPending ? <LoaderCircle className="animate-spin" /> : null}
                {batchDecision === 'accept' ? labels.acceptAll : labels.rejectAll}
              </Button>
            </>
          )}
          onClose={() => setBatchDecision(null)}
          open={batchDecision !== null}
          title={batchDecision === 'accept' ? labels.batchAcceptTitle : labels.batchRejectTitle}
        >
          <p className="t-meta text-muted-foreground">
            {filteredChanges.length} {labels.changes.toLocaleLowerCase(locale)}
          </p>
        </Dialog>
      </TabsContent>
    </Tabs>
  )
}

export function EditorWriteModeControl({
  access,
  canEdit,
  collaborationActive,
  mode,
  onModeChange,
  sourceReadOnly,
}: {
  access: 'edit' | 'suggest' | 'view' | null
  canEdit: boolean
  collaborationActive: boolean
  mode: EditorWriteMode
  onModeChange: (mode: Exclude<EditorWriteMode, 'view'>) => void
  sourceReadOnly: boolean
}) {
  const { locale } = useLocale()
  const labels = copy[locale]
  if (!collaborationActive) return null
  const editLocked = sourceReadOnly || access !== 'edit' || !canEdit
  const suggestLocked = sourceReadOnly || access === 'view' || access === null || !canEdit
  const lockLabel = sourceReadOnly
    ? labels.sourceReadOnly
    : access === 'view' || access === null
      ? labels.viewLocked
      : labels.suggestLocked

  return (
    <div className="flex h-7 shrink-0 items-center rounded-md bg-muted/60 p-0.5" role="group">
      <ModeButton
        active={mode === 'edit'}
        disabled={editLocked}
        label={labels.edit}
        lockLabel={editLocked ? lockLabel : null}
        onClick={() => onModeChange('edit')}
      />
      <ModeButton
        active={mode === 'suggest'}
        disabled={suggestLocked}
        label={labels.suggest}
        lockLabel={suggestLocked ? lockLabel : null}
        onClick={() => onModeChange('suggest')}
      />
      {mode === 'view' ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="flex size-6 items-center justify-center text-muted-foreground" tabIndex={0}>
              <LockKeyhole className="icon-xs" />
            </span>
          </TooltipTrigger>
          <TooltipContent>{lockLabel}</TooltipContent>
        </Tooltip>
      ) : null}
    </div>
  )
}

export function EditorCollaborationStatus({
  model,
  variant,
}: {
  model: EditorCollaborationStatusModel
  variant: 'inspector' | 'topbar'
}) {
  const { locale } = useLocale()
  const labels = copy[locale]
  const statusLabel = model.notice ?? labels.status[model.kind]
  const projectionLabel = model.projectionConfirmedAt
    ? `${labels.projection}: ${new Date(model.projectionConfirmedAt).toLocaleString(locale)}`
    : null
  const participantLabel = model.participants.length === 0
    ? `0 ${labels.participants}`
    : `${model.participants.length} ${labels.participants}: ${model.participants
      .map((participant) => participant.name)
      .join(', ')}`
  const fullLabel = [statusLabel, projectionLabel, participantLabel].filter(Boolean).join('. ')
  const avatarSize = variant === 'topbar' ? 'size-5' : 'size-6'
  return (
    <div
      aria-label={fullLabel}
      aria-live="polite"
      className={cn(
        'flex shrink-0 items-center justify-between gap-2',
        variant === 'inspector'
          ? 'h-9 border-b border-border px-3'
          : 'h-7 max-w-56 rounded-md bg-muted/60 px-1.5',
      )}
      role="status"
      title={fullLabel}
    >
      <div className="flex min-w-0 items-center gap-1.5">
        <span
          aria-hidden
          className={cn(
            'size-2 shrink-0 rounded-full',
            model.kind === 'saved' ? 'bg-success'
              : model.kind === 'saving'
                || model.kind === 'syncing'
                || model.kind === 'reconnecting'
                ? 'bg-warning'
                : model.kind === 'read_only' || model.kind === 'inactive'
                  ? 'bg-muted-foreground/50'
                  : 'bg-destructive',
          )}
        />
        <span className="t-meta-sm truncate text-muted-foreground">
          {statusLabel}
          {projectionLabel ? ` · ${projectionLabel}` : ''}
        </span>
      </div>
      <div aria-hidden className="flex shrink-0 items-center" data-participant-count={model.participants.length}>
        {model.participants.length === 0 ? <Users className="icon-sm text-muted-foreground" /> : null}
        {model.visibleParticipants.map((participant, index) => (
          <span
            className={cn(
              't-hint flex items-center justify-center rounded-full border-2 border-background font-medium',
              avatarSize,
              index > 0 && '-ml-1.5',
            )}
            key={participant.id}
            data-participant-id={participant.id}
            style={{ backgroundColor: participant.color, color: avatarTextColor(participant.color) }}
            title={participant.name}
          >
            {initials(participant.name)}
          </span>
        ))}
        {model.participantOverflow > 0 ? (
          <span className={cn(
            't-hint -ml-1.5 flex items-center justify-center rounded-full border-2 border-background bg-muted text-muted-foreground',
            avatarSize,
          )}>
            +{model.participantOverflow}
          </span>
        ) : null}
      </div>
    </div>
  )
}

function OpenChanges({
  changes,
  canDecide,
  error,
  expandedId,
  isDecisionPending,
  labels,
  locale,
  onDecision,
  onExpandedChange,
  onSelectedChange,
  selectedId,
}: {
  changes: readonly InspectorChange[]
  canDecide: boolean
  error: string | null
  expandedId: string | null
  isDecisionPending: boolean
  labels: typeof copy.de | typeof copy.en
  locale: 'de' | 'en'
  onDecision: (decision: InspectorDecision, patchIds: string[]) => void
  onExpandedChange: (id: string | null) => void
  onSelectedChange: (id: string | null) => void
  selectedId: string | null
}) {
  if (changes.length === 0) {
    return error
      ? <InspectorState icon={<AlertTriangle className="icon-md" />} message={error} tone="error" />
      : <InspectorState icon={<ListChecks className="icon-md" />} message={labels.noChanges} />
  }
  return (
    <>
      {error ? (
        <div className="flex items-start gap-2 border-b border-destructive/30 bg-destructive-subtle px-3 py-2 text-destructive">
          <AlertTriangle className="icon-sm mt-0.5 shrink-0" />
          <p className="t-meta">{error}</p>
        </div>
      ) : null}
      {changes.map((change) => {
    const expanded = expandedId === change.id
    const selected = selectedId === change.id
    return (
      <div
        className={cn(
          'border-b border-border transition-colors last:border-b-0',
          selected && 'bg-brand-subtle/45',
        )}
        data-inspector-change-id={change.id}
        key={change.id}
      >
        <button
          aria-expanded={expanded}
          className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-start gap-2 px-3 py-2 text-left outline-none hover:bg-surface focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring"
          onClick={() => {
            onSelectedChange(change.id)
            onExpandedChange(expanded ? null : change.id)
          }}
          type="button"
        >
          <span className="min-w-0">
            <span className="flex min-w-0 items-center gap-1.5">
              <span className="t-list truncate">{compactChangeText(change) || suggestionTypeLabel(change.type, labels)}</span>
              <span className="t-hint shrink-0 rounded-sm bg-muted px-1 py-0.5 text-muted-foreground">
                {suggestionTypeLabel(change.type, labels)}
              </span>
            </span>
            <span className="t-meta-sm mt-0.5 flex min-w-0 items-center gap-1.5 text-muted-foreground">
              <span className="truncate">{change.author.name}</span>
              <span aria-hidden>·</span>
              <span className="shrink-0">{formatInspectorTime(change.createdAt, locale)}</span>
            </span>
          </span>
          <ChevronDown className={cn('icon-sm mt-0.5 text-muted-foreground transition-transform', expanded && 'rotate-180')} />
        </button>
        {expanded ? (
          <div className="space-y-2 border-t border-border/70 px-3 py-2">
            {change.originalText ? (
              <ChangeText label={labels.original} text={change.originalText} tone="original" />
            ) : null}
            {change.proposedText ? (
              <ChangeText label={labels.proposed} text={change.proposedText} tone="proposed" />
            ) : null}
            <div className="flex justify-end gap-1">
              <DecisionButton
                decision="reject"
                disabled={isDecisionPending || !canDecide}
                label={labels.reject}
                onClick={() => onDecision('reject', [change.id])}
              />
              <DecisionButton
                decision="accept"
                disabled={isDecisionPending || !canDecide}
                label={labels.accept}
                onClick={() => onDecision('accept', [change.id])}
              />
            </div>
          </div>
        ) : null}
      </div>
    )
      })}
    </>
  )
}

function HistoryChanges({
  entries,
  error,
  expandedId,
  isLoading,
  labels,
  locale,
  onExpandedChange,
}: {
  entries: readonly InspectorHistoryEntry[]
  error: string | null
  expandedId: string | null
  isLoading: boolean
  labels: typeof copy.de | typeof copy.en
  locale: 'de' | 'en'
  onExpandedChange: (id: string | null) => void
}) {
  if (isLoading) return <InspectorState icon={<LoaderCircle className="icon-md animate-spin" />} message={labels.loading} />
  if (error) return <InspectorState icon={<AlertTriangle className="icon-md" />} message={error} tone="error" />
  if (entries.length === 0) return <InspectorState icon={<Clock3 className="icon-md" />} message={labels.noHistory} />
  return entries.map((entry) => {
    const expanded = entry.id === expandedId
    return (
      <div className="border-b border-border last:border-b-0" key={entry.id}>
        <button
          aria-expanded={expanded}
          className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-start gap-2 px-3 py-2 text-left outline-none hover:bg-surface focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring"
          onClick={() => onExpandedChange(expanded ? null : entry.id)}
          type="button"
        >
          <span className="min-w-0">
            <span className="t-list block truncate">{historyTypeLabel(entry.type, labels)}</span>
            <span className="t-meta-sm mt-0.5 flex min-w-0 items-center gap-1.5 text-muted-foreground">
              <span className="truncate">{entry.actor.name}</span>
              <span aria-hidden>·</span>
              <span className="shrink-0">{formatInspectorTime(entry.createdAt, locale)}</span>
            </span>
          </span>
          <ChevronDown className={cn('icon-sm mt-0.5 text-muted-foreground transition-transform', expanded && 'rotate-180')} />
        </button>
        {expanded ? (
          <div className="t-meta-sm border-t border-border/70 px-3 py-2 text-muted-foreground">
            <span className="tabular-nums">{entry.fromSequence}–{entry.toSequence}</span>
            {entry.suggestionIds.length > 0 ? (
              <span> · {entry.suggestionIds.length} {labels.changes.toLocaleLowerCase(locale)}</span>
            ) : null}
          </div>
        ) : null}
      </div>
    )
  })
}

function InspectorSelect({
  label,
  onValueChange,
  options,
  placeholder,
  value,
}: {
  label: string
  onValueChange: (value: string) => void
  options: readonly { label: string; value: string }[]
  placeholder: string
  value: string
}) {
  return (
    <Select onValueChange={onValueChange} value={value}>
      <SelectTrigger aria-label={label} density="toolbar">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="all">{placeholder}</SelectItem>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function SegmentButton({ active, count, label, onClick }: {
  active: boolean
  count?: number
  label: string
  onClick: () => void
}) {
  return (
    <button
      aria-pressed={active}
      className={cn(
        'flex h-7 min-w-0 items-center justify-center gap-1 rounded-md px-1 text-xs font-medium transition-colors',
        active ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
      )}
      onClick={onClick}
      type="button"
    >
      <span className="truncate">{label}</span>
      {count !== undefined ? <span className="t-hint tabular-nums">{count}</span> : null}
    </button>
  )
}

function ModeButton({ active, disabled, label, lockLabel, onClick }: {
  active: boolean
  disabled: boolean
  label: string
  lockLabel: string | null
  onClick: () => void
}) {
  const button = (
    <button
      aria-pressed={active}
      className={cn(
        'flex h-6 items-center gap-1 rounded-sm px-2 text-xs font-medium transition-colors',
        active ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
        disabled && 'cursor-not-allowed opacity-50',
      )}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      {disabled ? <LockKeyhole className="icon-xs" /> : null}
      {label}
    </button>
  )
  if (!lockLabel) return button
  return (
    <Tooltip>
      <TooltipTrigger asChild><span className="inline-flex" tabIndex={0}>{button}</span></TooltipTrigger>
      <TooltipContent>{lockLabel}</TooltipContent>
    </Tooltip>
  )
}

function IconButton({ children, disabled, label, onClick }: {
  children: ReactNode
  disabled?: boolean
  label: string
  onClick: () => void
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          className="size-7 shrink-0 text-muted-foreground hover:text-foreground"
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

function DecisionButton({ decision, disabled, label, onClick }: {
  decision: InspectorDecision
  disabled: boolean
  label: string
  onClick: () => void
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          className={cn('size-7', decision === 'accept' && 'text-success hover:text-success')}
          disabled={disabled}
          onClick={onClick}
          size="icon"
          type="button"
          variant="ghost"
        >
          {decision === 'accept' ? <Check className="icon-sm" /> : <X className="icon-sm" />}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function ChangeText({ label, text, tone }: {
  label: string
  text: string
  tone: 'original' | 'proposed'
}) {
  return (
    <div>
      <span className="t-label text-muted-foreground">{label}</span>
      <p className={cn(
        't-meta mt-1 whitespace-pre-wrap break-words border-l-2 pl-2 text-foreground',
        tone === 'proposed' ? 'border-success' : 'border-destructive',
      )}>
        {text}
      </p>
    </div>
  )
}

function InspectorState({ icon, message, tone = 'muted' }: {
  icon: ReactNode
  message: string
  tone?: 'error' | 'muted'
}) {
  return (
    <div className={cn(
      'flex min-h-32 flex-col items-center justify-center gap-2 px-6 py-8 text-center',
      tone === 'error' ? 'text-destructive' : 'text-muted-foreground',
    )}>
      <span>{icon}</span>
      <p className="t-meta">{message}</p>
    </div>
  )
}

function uniqueParticipants(participants: readonly InspectorParticipant[]): InspectorParticipant[] {
  return [...new Map(participants.map((participant) => [participant.id, participant])).values()]
    .sort((left, right) => left.name.localeCompare(right.name))
}

function suggestionTypeOptions(labels: typeof copy.de | typeof copy.en) {
  return (['insertion', 'deletion', 'modification'] as const).map((type) => ({
    label: suggestionTypeLabel(type, labels),
    value: type,
  }))
}

function historyTypeOptions(labels: typeof copy.de | typeof copy.en) {
  return (['direct', 'suggestion', 'decision', 'system'] as const).map((type) => ({
    label: historyTypeLabel(type, labels),
    value: type,
  }))
}

function suggestionTypeLabel(
  type: SuggestionKind,
  labels: typeof copy.de | typeof copy.en,
): string {
  return labels[type]
}

function historyTypeLabel(
  type: CollaborationChangeKind,
  labels: typeof copy.de | typeof copy.en,
): string {
  const historyLabels = labels === copy.de
    ? { decision: 'Entscheidung', direct: 'Direkte Änderung', suggestion: 'Vorschlag', system: 'System' }
    : { decision: 'Decision', direct: 'Direct edit', suggestion: 'Suggestion', system: 'System' }
  return historyLabels[type]
}

function formatInspectorTime(timestamp: number, locale: 'de' | 'en'): string {
  return new Intl.DateTimeFormat(locale === 'de' ? 'de-DE' : 'en-US', {
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    month: 'short',
  }).format(timestamp)
}

function initials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toLocaleUpperCase())
    .join('') || '?'
}

function avatarTextColor(color: string): '#111827' | '#ffffff' {
  const red = Number.parseInt(color.slice(1, 3), 16)
  const green = Number.parseInt(color.slice(3, 5), 16)
  const blue = Number.parseInt(color.slice(5, 7), 16)
  const luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
  return luminance > 0.58 ? '#111827' : '#ffffff'
}
