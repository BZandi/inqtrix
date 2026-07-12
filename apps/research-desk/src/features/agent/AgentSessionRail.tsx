import { useMemo, useState } from 'react'

import { FolderPlus, Pin, PinOff, SquarePen, Trash2 } from '@/components/icons'
import { Button } from '@/components/ui/button'
import {
  ExplorerFolderRow,
  ExplorerHistoryRow,
  ExplorerHistoryTitleInput,
  ExplorerSearchField,
  ExplorerSectionLabel,
  ExplorerRunningIndicator,
} from '@/components/ui/explorer-list'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { displayRelativeAge } from '@/features/project/selectors'
import { useLocale } from '@/i18n/LocaleProvider'
import {
  agentSessionHistoryTimeIso,
  isActiveAgentRun,
  isGateAgentRun,
  type AgentRunRecord,
  type AgentSessionGroupRecord,
  type AgentSessionRecord,
} from './model'

/**
 * Session rail of the Agent Desk (KnowledgeHistoryPanel structure, shared
 * explorer primitives). Rows show a micro live indicator while a session's
 * latest run is active; a parked run (waiting) shows the warning dot —
 * ambient signal for "needs you".
 */
export function AgentSessionRail({
  onCreateSession,
  onCreateSessionGroup,
  onDeleteSession,
  onRenameSession,
  onSelectSession,
  onTogglePinnedSession,
  pinnedSessionIds,
  runs,
  selectedSessionId,
  sessionGroupOrder,
  sessionGroups,
  sessionOrder,
  sessions,
  syncError = null,
}: {
  onCreateSession: () => void
  onCreateSessionGroup: () => void
  onDeleteSession: (sessionId: string) => void
  onRenameSession: (sessionId: string, title: string) => void
  onSelectSession: (sessionId: string) => void
  onTogglePinnedSession: (sessionId: string) => void
  pinnedSessionIds: readonly string[]
  runs: Record<string, AgentRunRecord>
  selectedSessionId: string | null
  sessionGroupOrder: string[]
  sessionGroups: Record<string, AgentSessionGroupRecord>
  sessionOrder: string[]
  sessions: Record<string, AgentSessionRecord>
  /** Session-sync failure — shown loudly under the header, never dropped. */
  syncError?: string | null
}) {
  const { locale, t } = useLocale()
  const [searchQuery, setSearchQuery] = useState('')
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [titleDraft, setTitleDraft] = useState('')

  const orderedSessions = useMemo(
    () =>
      sessionOrder
        .map((id) => sessions[id])
        .filter((session): session is AgentSessionRecord => Boolean(session)),
    [sessionOrder, sessions],
  )
  const query = searchQuery.trim().toLowerCase()
  const visibleSessions = query
    ? orderedSessions.filter((session) =>
      session.title.toLowerCase().includes(query))
    : orderedSessions
  const pinned = visibleSessions.filter((session) =>
    pinnedSessionIds.includes(session.id))
  const grouped = new Map<string | null, AgentSessionRecord[]>()
  for (const session of visibleSessions) {
    if (pinnedSessionIds.includes(session.id)) continue
    const key = session.groupId && sessionGroups[session.groupId]
      ? session.groupId
      : null
    const bucket = grouped.get(key) ?? []
    bucket.push(session)
    grouped.set(key, bucket)
  }

  const commitEdit = () => {
    if (editingSessionId && titleDraft.trim()) {
      onRenameSession(editingSessionId, titleDraft.trim())
    }
    setEditingSessionId(null)
  }

  const renderSession = (session: AgentSessionRecord, nested: boolean) => {
    const latestRun = session.runIds
      .map((runId) => runs[runId])
      .filter(Boolean)
      .at(-1)
    const gate = latestRun !== undefined && isGateAgentRun(latestRun.status)
    // Working = active minus gates: a children-wait keeps the live dot
    // (the children ARE working), a human gate shows the warning dot.
    const working =
      latestRun !== undefined
      && isActiveAgentRun(latestRun.status)
      && !gate
    const isPinned = pinnedSessionIds.includes(session.id)
    const editing = editingSessionId === session.id
    const timeLabel = displayRelativeAge(
      agentSessionHistoryTimeIso(session, runs),
      locale,
    )
    return (
      <ExplorerHistoryRow
        actions={[
          {
            icon: isPinned ? <PinOff className="icon-sm" /> : <Pin className="icon-sm" />,
            label: isPinned ? t.agent.sessions.unpin : t.agent.sessions.pin,
            onSelect: () => onTogglePinnedSession(session.id),
          },
          {
            destructive: true,
            icon: <Trash2 className="icon-sm" />,
            label: t.agent.sessions.delete,
            onSelect: () => onDeleteSession(session.id),
          },
        ]}
        active={selectedSessionId === session.id}
        indicator={
          gate ? (
            <span
              aria-hidden="true"
              className="size-1.5 shrink-0 rounded-full bg-warning inqtrix-running-dot"
            />
          ) : working ? (
            <ExplorerRunningIndicator label={t.status.running} />
          ) : undefined
        }
        key={session.id}
        nested={nested}
        onSelect={() => onSelectSession(session.id)}
        onStartRename={() => {
          setEditingSessionId(session.id)
          setTitleDraft(session.title)
        }}
        renameEditor={editing ? (
          <ExplorerHistoryTitleInput
            autoFocus
            label={t.agent.sessions.rename}
            onCancel={() => setEditingSessionId(null)}
            onChange={setTitleDraft}
            onCommit={commitEdit}
            value={titleDraft}
          />
        ) : undefined}
        renameLabel={t.agent.sessions.rename}
        timeLabel={timeLabel}
        title={session.title}
      />
    )
  }

  return (
    <aside className="flex h-full min-h-0 w-full flex-col border-r border-border bg-surface/60">
      <div className="flex inqtrix-panel-header items-center justify-between gap-2 border-b border-border px-3">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="truncate t-section text-foreground">
            {t.agent.sessions.title}
          </h2>
        </div>
        <div className="flex items-center gap-1.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={t.agent.sessions.createGroup}
                className="size-7 shrink-0"
                onClick={onCreateSessionGroup}
                size="icon"
                type="button"
                variant="ghost"
              >
                <FolderPlus className="size-4 text-foreground/85" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.agent.sessions.createGroup}</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={t.agent.sessions.create}
                className="size-7 shrink-0"
                onClick={onCreateSession}
                size="icon"
                type="button"
                variant="ghost"
              >
                <SquarePen className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t.agent.sessions.create}</TooltipContent>
          </Tooltip>
        </div>
      </div>
      {syncError && (
        <p className="border-b border-border px-3 py-1.5 t-meta-sm text-destructive">
          {syncError}
        </p>
      )}
      <ExplorerSearchField
        clearLabel={t.knowledge.searchClear}
        label={t.agent.sessions.searchPlaceholder}
        onChange={setSearchQuery}
        onClear={() => setSearchQuery('')}
        placeholder={t.agent.sessions.searchPlaceholder}
        value={searchQuery}
      />
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-1 p-2">
          {visibleSessions.length === 0 && (
            <p className="px-1.5 py-2 t-meta text-muted-foreground">
              {t.agent.sessions.empty}
            </p>
          )}
          {pinned.length > 0 && (
            <div className="space-y-0.5">
              <ExplorerSectionLabel>{t.agent.canvas.pinned}</ExplorerSectionLabel>
              {pinned.map((session) => renderSession(session, false))}
            </div>
          )}
          {sessionGroupOrder.map((groupId) => {
            const group = sessionGroups[groupId]
            const members = grouped.get(groupId) ?? []
            if (!group || members.length === 0) return null
            return (
              <div className="space-y-0.5" key={groupId}>
                <ExplorerFolderRow>
                  <span className="truncate t-label text-muted-foreground">{group.title}</span>
                </ExplorerFolderRow>
                {members.map((session) => renderSession(session, true))}
              </div>
            )
          })}
          <div className="space-y-0.5">
            {(grouped.get(null) ?? []).map((session) =>
              renderSession(session, false))}
          </div>
        </div>
      </ScrollArea>
    </aside>
  )
}
