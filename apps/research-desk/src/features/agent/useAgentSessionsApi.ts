import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch } from 'react'

import {
  deleteAgentSession,
  deleteAgentSessionGroup,
  getAgentSession,
  listAgentSessionGroups,
  listAgentSessions,
  saveAgentSession,
  saveAgentSessionGroup,
} from '@/api/inqtrixClient'
import { syncCollection } from '@/features/project/syncCollection'
import {
  useProjectSyncLifecycle,
  type SyncLifecycleToken,
} from '@/features/project/useProjectSyncLifecycle'
import type { ResearchDeskAction } from '@/features/researchDesk/state'
import type { AgentSessionGroupRecord, AgentSessionRecord } from './model'
import {
  agentSessionFingerprint,
  serverAgentSessionGroupPayload,
  serverAgentSessionPayload,
} from './agentSessionSync'

const AUTOSAVE_DEBOUNCE_MS = 1_500

/**
 * Server sync for agent sessions (clone of `useKnowledgeSessionsApi`,
 * simplified): sessions are metadata rows only — the turns are durable
 * server RUNS hydrated separately via the run list. Hydrates on activation,
 * pushes fingerprint-guarded changes debounced, deletes removed rows.
 */
export function useAgentSessionsApi({
  apiKey,
  dispatch,
  projectEpoch,
  sessionGroups,
  sessionOrder,
  selectedSessionId,
  sessions,
  syncActive,
  workspaceId,
}: {
  apiKey: string | undefined
  dispatch: Dispatch<ResearchDeskAction>
  projectEpoch: number
  sessionGroups: Record<string, AgentSessionGroupRecord>
  sessionOrder: string[]
  selectedSessionId: string | null
  sessions: Record<string, AgentSessionRecord>
  syncActive: boolean
  workspaceId: string
}): { error: string | null; settled: boolean } {
  const [error, setError] = useState<string | null>(null)
  const [hydrated, setHydrated] = useState(false)
  // "Settled", not "succeeded": the initial listing finished either way.
  // The workspace gates its loading skeleton on THIS (a failed listing
  // must end the skeleton and surface via `error`), while `hydrated`
  // stays success-only — flush must never push/delete against a
  // baseline that was never seeded (it would resurrect deleted rows).
  const [settled, setSettled] = useState(false)

  const sessionsRef = useRef(sessions)
  sessionsRef.current = sessions
  const sessionOrderRef = useRef(sessionOrder)
  sessionOrderRef.current = sessionOrder
  const groupsRef = useRef(sessionGroups)
  groupsRef.current = sessionGroups
  const optionsRef = useRef({ apiKey, workspaceId })
  optionsRef.current = { apiKey, workspaceId }
  const syncActiveRef = useRef(syncActive)
  syncActiveRef.current = syncActive

  const syncedRef = useRef(new Map<string, string>())
  const syncedGroupsRef = useRef(new Map<string, string>())
  const flushingRef = useRef(false)
  const flushPendingRef = useRef(false)

  const reset = useCallback(() => {
    syncedRef.current.clear()
    syncedGroupsRef.current.clear()
    flushingRef.current = false
    flushPendingRef.current = false
    setHydrated(false)
    setSettled(false)
  }, [])

  const hydrate = useCallback(
    (token: SyncLifecycleToken) => {
      void (async () => {
        try {
          const [serverGroups, serverSessions] = await Promise.all([
            listAgentSessionGroups(optionsRef.current),
            listAgentSessions(optionsRef.current),
          ])
          if (token.cancelled) return
          dispatch({
            groups: serverGroups,
            sessions: serverSessions,
            type: 'upsertServerAgentSessions',
          })
          for (const group of serverGroups) {
            syncedGroupsRef.current.set(group.id, String(group.updated_at))
          }
          for (const wire of serverSessions) {
            const record = sessionsRef.current[wire.id]
            if (record) {
              syncedRef.current.set(wire.id, agentSessionFingerprint(record))
            }
          }
          setHydrated(true)
          setSettled(true)
          setError(null)
        } catch (caught) {
          if (!token.cancelled) {
            setSettled(true)
            setError(
              caught instanceof Error ? caught.message : String(caught),
            )
          }
        }
      })()
    },
    [dispatch],
  )

  useProjectSyncLifecycle({
    active: syncActive,
    hydrate,
    identity: `${workspaceId}:${projectEpoch}`,
    reset,
  })

  const flush = useCallback(async () => {
    if (!syncActiveRef.current || !hydrated) return
    if (flushingRef.current) {
      flushPendingRef.current = true
      return
    }
    flushingRef.current = true
    try {
      await syncCollection<AgentSessionGroupRecord, string>({
        current: groupsRef.current,
        synced: syncedGroupsRef.current,
        fingerprintOf: (group) => group.updatedAt,
        changed: (previous, current) => previous !== current,
        pushOne: async (group) => {
          await saveAgentSessionGroup(
            group.id,
            serverAgentSessionGroupPayload(group),
            optionsRef.current,
          )
        },
        deleteOne: (id) => deleteAgentSessionGroup(id, optionsRef.current),
      })

      const currentSessions = sessionsRef.current
      for (const sessionId of sessionOrderRef.current) {
        const session = currentSessions[sessionId]
        if (!session) continue
        const fingerprint = agentSessionFingerprint(session)
        if (syncedRef.current.get(sessionId) !== fingerprint) {
          await saveAgentSession(
            sessionId,
            serverAgentSessionPayload(session),
            optionsRef.current,
          )
          syncedRef.current.set(sessionId, fingerprint)
        }
      }
      for (const sessionId of [...syncedRef.current.keys()]) {
        if (!(sessionId in currentSessions)) {
          await deleteAgentSession(sessionId, optionsRef.current)
          syncedRef.current.delete(sessionId)
        }
      }
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      flushingRef.current = false
      if (flushPendingRef.current) {
        flushPendingRef.current = false
        void flush()
      }
    }
  }, [hydrated])

  useEffect(() => {
    if (!syncActive || !hydrated) return undefined
    const timer = window.setTimeout(() => {
      void flush()
    }, AUTOSAVE_DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [flush, hydrated, sessionGroups, sessionOrder, sessions, syncActive])

  // List rows are intentionally metadata-only. Fetch the selected row's full
  // items_json so its source policy is hydrated before the user submits.
  useEffect(() => {
    if (!syncActive || !hydrated || !selectedSessionId) return undefined
    let cancelled = false
    void getAgentSession(selectedSessionId, optionsRef.current)
      .then((session) => {
        if (cancelled) return
        dispatch({ groups: [], sessions: [session], type: 'upsertServerAgentSessions' })
      })
      .catch((caught) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught))
        }
      })
    return () => {
      cancelled = true
    }
  }, [dispatch, hydrated, selectedSessionId, syncActive])

  useEffect(() => {
    if (!syncActive) return undefined
    const onVisibilityChange = () => {
      if (document.visibilityState === 'hidden') void flush()
    }
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () =>
      document.removeEventListener('visibilitychange', onVisibilityChange)
  }, [flush, syncActive])

  return { error, settled }
}
