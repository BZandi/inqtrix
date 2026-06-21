/**
 * Chat-history server sync (M6a project-persistence tier).
 *
 * The chat analogue of useResearchRunApi (resume-on-mount) + useTemplateSync
 * (local<->server reconciliation), but persistence-only: it never calls a
 * model. The LLM submission path in ResearchDesk is untouched — this hook
 * is a PARALLEL layer that keeps the conversation record on the server in
 * step with the local reducer, exactly the way template sync runs beside
 * how rules are used.
 *
 * Three behaviours, all gated on the durable ``project_persistence``
 * capability AND ``serverSyncEnabled`` (automatic for an authenticated
 * cookie-session user — derived from the session in ResearchDesk; the import
 * button only sets it for the apikey / local-first tiers):
 *   1. Hydrate on mount: list groups + threads, merge their metadata into
 *      the reducer (server-pushed, never marks the project dirty). Messages
 *      load lazily, per thread, on open.
 *   2. Load-on-open: when a thread without loaded messages is selected,
 *      fetch one newest-first page and merge it in.
 *   3. Autosave: a debounced diff (via the shared syncCollection helper)
 *      pushes new/changed threads + their messages, new/changed groups, and
 *      DELETEs whatever vanished locally. Idempotent server upserts make a
 *      retry or a coalesced burst safe.
 *
 * It does NOT own the import button: the explicit opt-in push is the
 * project-level useProjectServerImport (which pushes chat AND editor in one
 * flow). This hook only hydrates + autosaves once the project is opted in
 * (syncActive), seeding its synced fingerprint to WHAT THE SERVER HOLDS so a
 * local-newer entity is pushed up rather than stranded (the M6a P1 lesson).
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch } from 'react'

import {
  appendChatMessages,
  deleteChatThread,
  deleteChatThreadGroup,
  listChatMessages,
  listChatThreadGroups,
  listChatThreads,
  saveChatThread,
  saveChatThreadGroup,
} from '@/api/inqtrixClient'
import {
  fingerprintThread,
  groupRecordFromServer,
  messageRecordFromServer,
  serverGroupPayload,
  serverMessagePayload,
  serverThreadPayload,
  threadNeedsSync,
  threadRecordFromServer,
  type ThreadFingerprint,
} from '@/features/chat/chatHistorySync'
import type {
  ChatThreadGroupRecord,
  ChatThreadRecord,
} from '@/features/project/types'
import { syncCollection } from '@/features/project/syncCollection'
import {
  useProjectSyncLifecycle,
  type SyncLifecycleToken,
} from '@/features/project/useProjectSyncLifecycle'
import type { ResearchDeskAction } from '@/features/researchDesk/state'

const AUTOSAVE_DEBOUNCE_MS = 1_500
// Threads load on-demand (page one fast, then cursor-based load-more in the
// history sidebar), so the page is sidebar-sized rather than a bulk walk.
const THREAD_PAGE_LIMIT = 50
const MESSAGE_PAGE_LIMIT = 200

function messageFromError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

type UseChatHistoryApiOptions = {
  apiKey: string | undefined
  chatThreadGroupMemberships: Record<string, string | null>
  chatThreadGroups: Record<string, ChatThreadGroupRecord>
  chatThreads: Record<string, ChatThreadRecord>
  dispatch: Dispatch<ResearchDeskAction>
  /** In-session load counter (bumped on every wholesale project replace). Part
   * of the lifecycle identity so a switch to another synced project re-hydrates
   * from its own server state instead of inheriting this one's synced map. */
  projectEpoch: number
  selectedThreadId: string | null
  /** The user opted this project into server sync (``serverSyncEnabled``
   * AND the durable capability AND not demo): hydrate + load-on-open +
   * autosave run. The explicit import push is project-level
   * (useProjectServerImport), not this hook. */
  syncActive: boolean
  workspaceId: string
}

export type ChatHistoryApiHandle = {
  error: string | null
  /** Whether the server has older thread pages not yet loaded. */
  hasMoreThreads: boolean
  /** A load-older page request is in flight (drives the button's busy state). */
  isLoadingMore: boolean
  /** Load the next page of older threads (cursor-based; appends to the list). */
  loadMoreThreads: () => Promise<void>
}

export function useChatHistoryApi({
  apiKey,
  chatThreadGroupMemberships,
  chatThreadGroups,
  chatThreads,
  dispatch,
  projectEpoch,
  selectedThreadId,
  syncActive,
  workspaceId,
}: UseChatHistoryApiOptions): ChatHistoryApiHandle {
  const [error, setError] = useState<string | null>(null)
  // Mirrors hydratedRef into render state so the load-on-open and autosave
  // effects RE-RUN once hydration completes (a ref mutation alone would
  // not), while hydratedRef keeps the value readable from the debounced
  // flush callback outside React's render cycle.
  const [hydrated, setHydrated] = useState(false)

  // Latest state read by the async flush without stale closures.
  const threadsRef = useRef(chatThreads)
  threadsRef.current = chatThreads
  const groupsRef = useRef(chatThreadGroups)
  groupsRef.current = chatThreadGroups
  const membershipsRef = useRef(chatThreadGroupMemberships)
  membershipsRef.current = chatThreadGroupMemberships

  const optionsRef = useRef({ apiKey, workspaceId })
  optionsRef.current = { apiKey, workspaceId }

  // "What the server has", seeded by hydrate/import and advanced by every
  // successful push, so the diff only writes genuinely-changed entities.
  const syncedThreadsRef = useRef(new Map<string, ThreadFingerprint>())
  const syncedGroupsRef = useRef(new Map<string, string>())
  const loadedThreadsRef = useRef(new Set<string>())
  // The next-page cursor for on-demand thread loading; undefined = no more.
  const threadCursorRef = useRef<string | undefined>(undefined)
  const loadingMoreRef = useRef(false)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [hasMoreThreads, setHasMoreThreads] = useState(false)
  const hydratedRef = useRef(false)
  const flushingRef = useRef(false)
  const flushPendingRef = useRef(false)
  const syncActiveRef = useRef(syncActive)
  syncActiveRef.current = syncActive

  // -- pushing one entity (syncCollection advances the synced fingerprint) #

  const pushThread = useCallback(async (thread: ChatThreadRecord) => {
    const options = optionsRef.current
    const groupId = membershipsRef.current[thread.id] ?? null
    await saveChatThread(thread.id, serverThreadPayload(thread, groupId), options)
    if (thread.messages.length > 0) {
      await appendChatMessages(
        thread.id,
        thread.messages.map(serverMessagePayload),
        options,
      )
    }
  }, [])

  const pushGroup = useCallback(async (group: ChatThreadGroupRecord) => {
    await saveChatThreadGroup(
      group.id,
      serverGroupPayload(group),
      optionsRef.current,
    )
  }, [])

  // -- autosave flush (debounced, serialized) ---------------------------- #

  const flush = useCallback(async () => {
    if (!syncActiveRef.current || !hydratedRef.current) return
    if (flushingRef.current) {
      flushPendingRef.current = true
      return
    }
    flushingRef.current = true
    try {
      const memberships = membershipsRef.current
      await syncCollection<ChatThreadRecord, ThreadFingerprint>({
        current: threadsRef.current,
        synced: syncedThreadsRef.current,
        fingerprintOf: (thread) =>
          fingerprintThread(thread, memberships[thread.id] ?? null),
        changed: threadNeedsSync,
        pushOne: pushThread,
        deleteOne: async (id) => {
          await deleteChatThread(id, optionsRef.current)
          loadedThreadsRef.current.delete(id)
        },
      })
      await syncCollection<ChatThreadGroupRecord, string>({
        current: groupsRef.current,
        synced: syncedGroupsRef.current,
        fingerprintOf: (group) => group.updatedAt,
        changed: (previous, current) => previous !== current,
        pushOne: pushGroup,
        deleteOne: (id) => deleteChatThreadGroup(id, optionsRef.current),
      })
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
  }, [pushThread, pushGroup])

  // -- reset + hydrate lifecycle (re-armed on project identity) ---------- #

  // Drop the prior project's sync state. Runs before every (re)hydrate and on
  // leaving sync, so a switch to another synced project never carries this
  // project's synced fingerprints into a cross-project delete.
  const reset = useCallback(() => {
    hydratedRef.current = false
    setHydrated(false)
    syncedThreadsRef.current.clear()
    syncedGroupsRef.current.clear()
    loadedThreadsRef.current.clear()
    threadCursorRef.current = undefined
    loadingMoreRef.current = false
    setIsLoadingMore(false)
    setHasMoreThreads(false)
  }, [])

  const hydrate = useCallback((token: SyncLifecycleToken) => {
    void (async () => {
      try {
        const options = optionsRef.current
        const serverGroups = await listChatThreadGroups(options)
        const groupRecords = serverGroups.map(groupRecordFromServer)
        const threadRecords: ChatThreadRecord[] = []
        const memberships: Record<string, string | null> = {}
        // Load only the first (newest) page; older pages load on demand via
        // loadMoreThreads. The autosave stays safe because synced is seeded
        // only for loaded threads (below), so delete-detection never touches an
        // un-loaded thread.
        const page = await listChatThreads({ ...options, limit: THREAD_PAGE_LIMIT })
        for (const serverThread of page.data) {
          const { groupId, record } = threadRecordFromServer(serverThread)
          threadRecords.push(record)
          memberships[record.id] = groupId
        }
        if (token.cancelled) return
        if (groupRecords.length > 0) {
          dispatch({ groups: groupRecords, type: 'upsertServerChatThreadGroups' })
        }
        if (threadRecords.length > 0) {
          dispatch({
            memberships,
            threads: threadRecords,
            type: 'upsertServerChatThreads',
          })
        }
        // Seed each fingerprint to WHAT THE SERVER HOLDS (the server
        // record), never the local value. When the local copy is newer the
        // reducer keeps it, so local != seeded-server and the next autosave
        // pushes the newer local thread UP (local-newer-wins on the server,
        // not merely locally). When the server is newer or the thread is
        // new, local == server after the merge, so no spurious re-push.
        for (const record of threadRecords) {
          syncedThreadsRef.current.set(
            record.id,
            fingerprintThread(record, memberships[record.id] ?? null),
          )
        }
        for (const record of groupRecords) {
          syncedGroupsRef.current.set(record.id, record.updatedAt)
        }
        threadCursorRef.current = page.next_cursor ?? undefined
        setHasMoreThreads(Boolean(page.next_cursor))
        hydratedRef.current = true
        setHydrated(true)
        setError(null)
      } catch (caught) {
        if (!token.cancelled) setError(messageFromError(caught))
      }
    })()
  }, [dispatch])

  useProjectSyncLifecycle({
    active: syncActive,
    identity: `${workspaceId}:${projectEpoch}`,
    reset,
    hydrate,
  })

  // -- load a thread's messages on open ---------------------------------- #

  useEffect(() => {
    // Gate on the `hydrated` STATE (not the ref) so this re-runs when an
    // async hydration completes AFTER the selected thread was already
    // restored from the manifest — otherwise the landed thread would stay
    // visibly empty until the user clicked away and back.
    if (!syncActive || !hydrated || !selectedThreadId) return
    const thread = threadsRef.current[selectedThreadId]
    if (!thread || thread.messages.length > 0) return
    if (loadedThreadsRef.current.has(selectedThreadId)) return
    loadedThreadsRef.current.add(selectedThreadId)
    let cancelled = false
    let applied = false
    void (async () => {
      try {
        const messages: ReturnType<typeof messageRecordFromServer>[] = []
        let cursor: string | undefined
        do {
          const page = await listChatMessages(selectedThreadId, {
            ...optionsRef.current,
            cursor,
            limit: MESSAGE_PAGE_LIMIT,
          })
          for (const serverMessage of page.data) {
            messages.push(messageRecordFromServer(serverMessage))
          }
          cursor = page.next_cursor ?? undefined
        } while (cursor)
        if (cancelled) return
        if (messages.length > 0) {
          dispatch({
            messages,
            threadId: selectedThreadId,
            type: 'upsertServerChatMessages',
          })
        }
        applied = true
        setError(null)
      } catch (caught) {
        if (!cancelled) {
          loadedThreadsRef.current.delete(selectedThreadId)
          setError(messageFromError(caught))
        }
      }
    })()
    return () => {
      cancelled = true
      // Released if interrupted before applying, so re-opening re-fetches.
      if (!applied) loadedThreadsRef.current.delete(selectedThreadId)
    }
  }, [syncActive, hydrated, selectedThreadId, dispatch])

  // -- debounced autosave trigger ---------------------------------------- #

  // Depends on `hydrated` too, so the FIRST flush runs right after hydration
  // even when no local edit followed — that reconciling flush pushes any
  // local-newer-than-server threads up (paired with the server-seeded
  // fingerprint above).
  useEffect(() => {
    if (!syncActive || !hydrated) return
    const timer = setTimeout(() => {
      void flush()
    }, AUTOSAVE_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [
    chatThreads,
    chatThreadGroups,
    chatThreadGroupMemberships,
    syncActive,
    hydrated,
    flush,
  ])

  // -- load older threads on demand ------------------------------------- #

  const loadMoreThreads = useCallback(async () => {
    if (!syncActiveRef.current || !hydratedRef.current) return
    if (loadingMoreRef.current) return
    const cursor = threadCursorRef.current
    if (!cursor) return
    loadingMoreRef.current = true
    setIsLoadingMore(true)
    try {
      const page = await listChatThreads({
        ...optionsRef.current,
        cursor,
        limit: THREAD_PAGE_LIMIT,
      })
      // The sync session may have been torn down while the page was in flight
      // (project closed / left server-sync). Mirror the hydrate effect's
      // cancellation discipline: do not dispatch or re-seed into a session the
      // re-arm effect has already cleared, which would re-pollute synced state.
      if (!syncActiveRef.current || !hydratedRef.current) return
      const threadRecords: ChatThreadRecord[] = []
      const memberships: Record<string, string | null> = {}
      for (const serverThread of page.data) {
        const { groupId, record } = threadRecordFromServer(serverThread)
        threadRecords.push(record)
        memberships[record.id] = groupId
      }
      if (threadRecords.length > 0) {
        // append: older page goes to the END of the order (stays newest-first).
        dispatch({ append: true, memberships, threads: threadRecords, type: 'upsertServerChatThreads' })
        // Seed synced ONLY for these newly loaded threads (delete-detection
        // stays scoped to loaded threads — never an un-loaded one).
        for (const record of threadRecords) {
          syncedThreadsRef.current.set(
            record.id,
            fingerprintThread(record, memberships[record.id] ?? null),
          )
        }
      }
      threadCursorRef.current = page.next_cursor ?? undefined
      setHasMoreThreads(Boolean(page.next_cursor))
      setError(null)
    } catch (caught) {
      setError(messageFromError(caught))
    } finally {
      loadingMoreRef.current = false
      setIsLoadingMore(false)
    }
  }, [dispatch])

  return { error, hasMoreThreads, isLoadingMore, loadMoreThreads }
}
