import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch } from 'react'

import {
  deleteKnowledgeSession,
  deleteKnowledgeSessionGroup,
  getKnowledgeSession,
  listKnowledgeSessionGroups,
  listKnowledgeSessions,
  saveKnowledgeSession,
  saveKnowledgeSessionGroup,
} from '@/api/inqtrixClient'
import type {
  KnowledgeSessionGroupRecord,
  KnowledgeSessionRecord,
  KnowledgeThreadItemRecord,
} from '@/features/project/types'
import {
  DEFAULT_KNOWLEDGE_SESSION_ID,
  DEFAULT_KNOWLEDGE_SESSION_TITLE,
} from '@/features/project/knowledgeSessionDefaults'
import { syncCollection } from '@/features/project/syncCollection'
import {
  useProjectSyncLifecycle,
  type SyncLifecycleToken,
} from '@/features/project/useProjectSyncLifecycle'
import type { ResearchDeskAction } from '@/features/researchDesk/state'
import {
  fingerprintKnowledgeSession,
  groupRecordFromServer,
  itemsFromServerSession,
  serverKnowledgeSessionGroupPayload,
  serverKnowledgeSessionPayload,
  sessionRecordFromServer,
} from './knowledgeSessionSync'

const AUTOSAVE_DEBOUNCE_MS = 1_500

function messageFromError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

type UseKnowledgeSessionsApiOptions = {
  apiKey: string | undefined
  dispatch: Dispatch<ResearchDeskAction>
  items: Record<string, KnowledgeThreadItemRecord>
  itemOrder: string[]
  projectEpoch: number
  selectedSessionId: string | null
  sessionGroupMemberships: Record<string, string | null>
  sessionGroups: Record<string, KnowledgeSessionGroupRecord>
  sessions: Record<string, KnowledgeSessionRecord>
  sessionOrder: string[]
  syncActive: boolean
  workspaceId: string
}

function isPristineBootstrapSession(
  session: KnowledgeSessionRecord,
  items: readonly KnowledgeThreadItemRecord[],
): boolean {
  return session.id === DEFAULT_KNOWLEDGE_SESSION_ID
    && session.title === DEFAULT_KNOWLEDGE_SESSION_TITLE
    && items.length === 0
}

export function useKnowledgeSessionsApi({
  apiKey,
  dispatch,
  items,
  itemOrder,
  projectEpoch,
  selectedSessionId,
  sessionGroupMemberships,
  sessionGroups,
  sessions,
  sessionOrder,
  syncActive,
  workspaceId,
}: UseKnowledgeSessionsApiOptions): { error: string | null } {
  const [error, setError] = useState<string | null>(null)
  const [hydrated, setHydrated] = useState(false)

  const sessionsRef = useRef(sessions)
  sessionsRef.current = sessions
  const sessionOrderRef = useRef(sessionOrder)
  sessionOrderRef.current = sessionOrder
  const groupsRef = useRef(sessionGroups)
  groupsRef.current = sessionGroups
  const membershipsRef = useRef(sessionGroupMemberships)
  membershipsRef.current = sessionGroupMemberships
  const itemsRef = useRef(items)
  itemsRef.current = items
  const itemOrderRef = useRef(itemOrder)
  itemOrderRef.current = itemOrder
  const selectedSessionIdRef = useRef(selectedSessionId)
  selectedSessionIdRef.current = selectedSessionId
  const optionsRef = useRef({ apiKey, workspaceId })
  optionsRef.current = { apiKey, workspaceId }
  const syncActiveRef = useRef(syncActive)
  syncActiveRef.current = syncActive

  const syncedRef = useRef(new Map<string, string>())
  const syncedGroupsRef = useRef(new Map<string, string>())
  const serverKnownRef = useRef(new Set<string>())
  const loadedRef = useRef(new Set<string>())
  const flushingRef = useRef(false)
  const flushPendingRef = useRef(false)

  const itemsForSession = useCallback((sessionId: string) => {
    return itemOrderRef.current
      .map((itemId) => itemsRef.current[itemId])
      .filter((item): item is KnowledgeThreadItemRecord => (
        Boolean(item) && item.sessionId === sessionId
      ))
  }, [])

  const reset = useCallback(() => {
    syncedRef.current.clear()
    syncedGroupsRef.current.clear()
    serverKnownRef.current.clear()
    loadedRef.current.clear()
    flushingRef.current = false
    flushPendingRef.current = false
    setHydrated(false)
  }, [])

  const hydrate = useCallback((token: SyncLifecycleToken) => {
    void (async () => {
      try {
        const serverGroups = await listKnowledgeSessionGroups(optionsRef.current)
        if (token.cancelled) return
        const groupRecords = serverGroups.map(groupRecordFromServer)
        if (groupRecords.length > 0) {
          dispatch({
            groups: groupRecords,
            type: 'upsertServerKnowledgeSessionGroups',
          })
        }
        for (const group of groupRecords) {
          syncedGroupsRef.current.set(group.id, group.updatedAt)
        }

        const serverSessions = await listKnowledgeSessions(optionsRef.current)
        if (token.cancelled) return
        const converted = serverSessions.map(sessionRecordFromServer)
        const records = converted.map(({ record }) => record)
        const memberships = Object.fromEntries(
          converted.map(({ groupId, record }) => [record.id, groupId]),
        )
        if (records.length > 0) {
          dispatch({
            memberships,
            sessions: records,
            type: 'upsertServerKnowledgeSessions',
          })
        }
        serverKnownRef.current = new Set(records.map((record) => record.id))
        for (const record of records) {
          syncedRef.current.set(
            record.id,
            fingerprintKnowledgeSession(record, [], memberships[record.id] ?? null),
          )
        }
        // Server is authoritative for the seed bootstrap default only: drop
        // the pristine local placeholder when it is not on the server. A user-
        // created or renamed empty session is real intent and syncs below.
        dispatch({
          serverIds: [...serverKnownRef.current],
          type: 'pruneLocalPlaceholderKnowledgeSessions',
        })
        const selected = selectedSessionIdRef.current
        if (records.length > 0 && (!selected || !serverKnownRef.current.has(selected))) {
          dispatch({ sessionId: records[0].id, type: 'selectKnowledgeSession' })
        }
        setHydrated(true)
        setError(null)
      } catch (caught) {
        if (!token.cancelled) setError(messageFromError(caught))
      }
    })()
  }, [dispatch])

  useProjectSyncLifecycle({
    active: syncActive,
    hydrate,
    identity: `${workspaceId}:${projectEpoch}`,
    reset,
  })

  useEffect(() => {
    if (!syncActive || !hydrated || !selectedSessionId) return
    if (!serverKnownRef.current.has(selectedSessionId)) return
    if (loadedRef.current.has(selectedSessionId)) return
    loadedRef.current.add(selectedSessionId)
    let cancelled = false
    void (async () => {
      try {
        const serverSession = await getKnowledgeSession(selectedSessionId, optionsRef.current)
        if (cancelled) return
        const { groupId, record } = sessionRecordFromServer(serverSession)
        const local = sessionsRef.current[selectedSessionId]
        const serverItems = itemsFromServerSession(serverSession)
        syncedRef.current.set(
          selectedSessionId,
          fingerprintKnowledgeSession(record, serverItems, groupId),
        )
        if (!local || record.updatedAt >= local.updatedAt) {
          dispatch({
            memberships: { [record.id]: groupId },
            sessions: [record],
            type: 'upsertServerKnowledgeSessions',
          })
          dispatch({
            items: serverItems,
            sessionId: selectedSessionId,
            type: 'setServerKnowledgeSessionItems',
          })
        }
        setError(null)
      } catch (caught) {
        loadedRef.current.delete(selectedSessionId)
        if (!cancelled) setError(messageFromError(caught))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [dispatch, hydrated, selectedSessionId, syncActive])

  const pushGroup = useCallback(async (group: KnowledgeSessionGroupRecord) => {
    await saveKnowledgeSessionGroup(
      group.id,
      serverKnowledgeSessionGroupPayload(group),
      optionsRef.current,
    )
  }, [])

  const pushSession = useCallback(async (session: KnowledgeSessionRecord) => {
    let itemsForPayload = itemsForSession(session.id)
    const groupId = membershipsRef.current[session.id] ?? null
    if (serverKnownRef.current.has(session.id) && !loadedRef.current.has(session.id)) {
      const serverSession = await getKnowledgeSession(session.id, optionsRef.current)
      itemsForPayload = itemsFromServerSession(serverSession)
      loadedRef.current.add(session.id)
      dispatch({
        items: itemsForPayload,
        sessionId: session.id,
        type: 'setServerKnowledgeSessionItems',
      })
    }
    await saveKnowledgeSession(
      session.id,
      serverKnowledgeSessionPayload(session, itemsForPayload, groupId),
      optionsRef.current,
    )
    serverKnownRef.current.add(session.id)
    loadedRef.current.add(session.id)
    syncedRef.current.set(
      session.id,
      fingerprintKnowledgeSession(session, itemsForPayload, groupId),
    )
  }, [dispatch, itemsForSession])

  const flush = useCallback(async () => {
    if (!syncActiveRef.current || !hydrated) return
    if (flushingRef.current) {
      flushPendingRef.current = true
      return
    }
    flushingRef.current = true
    try {
      await syncCollection<KnowledgeSessionGroupRecord, string>({
        current: groupsRef.current,
        synced: syncedGroupsRef.current,
        fingerprintOf: (group) => group.updatedAt,
        changed: (previous, current) => previous !== current,
        pushOne: pushGroup,
        deleteOne: (id) => deleteKnowledgeSessionGroup(id, optionsRef.current),
      })

      const currentSessions = sessionsRef.current
      for (const sessionId of sessionOrderRef.current) {
        const session = currentSessions[sessionId]
        if (!session) continue
        const itemsForFingerprint = itemsForSession(sessionId)
        const groupId = membershipsRef.current[sessionId] ?? null
        // Keep only the untouched seed placeholder local. Renamed or user-
        // created empty sessions are meaningful user state and sync as rows.
        if (
          isPristineBootstrapSession(session, itemsForFingerprint)
          && !serverKnownRef.current.has(sessionId)
        ) {
          continue
        }
        const fingerprint = fingerprintKnowledgeSession(session, itemsForFingerprint, groupId)
        if (syncedRef.current.get(sessionId) !== fingerprint) {
          await pushSession(session)
        }
      }
      for (const sessionId of [...syncedRef.current.keys()]) {
        if (!(sessionId in currentSessions)) {
          await deleteKnowledgeSession(sessionId, optionsRef.current)
          syncedRef.current.delete(sessionId)
          serverKnownRef.current.delete(sessionId)
          loadedRef.current.delete(sessionId)
        }
      }
      setError(null)
    } catch (caught) {
      setError(messageFromError(caught))
    } finally {
      flushingRef.current = false
      if (flushPendingRef.current) {
        flushPendingRef.current = false
        void flush()
      }
    }
  }, [hydrated, itemsForSession, pushGroup, pushSession])

  useEffect(() => {
    if (!syncActive || !hydrated) return undefined
    const anyRunning = itemOrder.some(
      (itemId) => items[itemId]?.status === 'running',
    )
    if (!anyRunning) {
      // Everything has settled (an answer just arrived, a rename, a delete):
      // persist immediately so a reload right after cannot lose it. flush() is
      // fingerprint-guarded, so an eager call is a no-op when nothing changed.
      void flush()
      return undefined
    }
    // While an ask is still streaming, batch interim changes behind the debounce.
    const timer = window.setTimeout(() => {
      void flush()
    }, AUTOSAVE_DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [
    flush,
    hydrated,
    itemOrder,
    items,
    sessionGroupMemberships,
    sessionGroups,
    sessionOrder,
    sessions,
    syncActive,
  ])

  useEffect(() => {
    if (!syncActive) return undefined
    // Backgrounding the tab is the reliable moment to persist before a likely
    // reload/close — same client path as the autosave, no separate beacon.
    // `flush` is in the deps, so the listener is rebound if it changes (no stale
    // closure), and flush() itself no-ops before hydrate completes.
    const onVisibilityChange = () => {
      if (document.visibilityState === 'hidden') void flush()
    }
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => document.removeEventListener('visibilitychange', onVisibilityChange)
  }, [flush, syncActive])

  return { error }
}
