import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import {
  cancelResearchRun,
  createResearchRun,
  deleteResearchRun,
  fetchRunEventsPage,
  fetchResearchRunResult,
  fetchResearchRunSummary,
  hasHttpStatus,
  listResearchRuns,
} from '@/api/inqtrixClient'
import { subscribeRunEvents } from './runEventChannel'
import type {
  CreateResearchRunRequest,
  ResearchRunEvent,
  ResearchRunResult,
  ResearchRunSummary,
} from './types'
import { isTerminalRunStatus } from './utils'

/** Runs per keyset page during history hydration; the loop follows
 * `next_cursor` so the full working set still hydrates while each server
 * query stays bounded (matches the chat/asset/editor hydration idiom). */
const RUN_PAGE_LIMIT = 100

type LiveRunCallbacks = {
  onEvent: (event: ResearchRunEvent) => void
  onResult: (result: ResearchRunResult) => void
  onRunError: (runId: string, message: string) => void
  onSummary: (summary: ResearchRunSummary, options?: { select?: boolean }) => void
}

type PerRunCallbacks = Partial<LiveRunCallbacks>

type UseResearchRunApiOptions = LiveRunCallbacks & {
  apiKey?: string
  enabled: boolean
  /** Whether the caller is admitted to list/run (the single auth gate the
   * parent already resolves: anonymous `none`, `apikey` with a key, or a live
   * cookie session). Flipping false->true (e.g. an in-app local/ldap login)
   * re-runs run-list hydration without a remount; false aborts live streams. */
  canList: boolean
  /** Re-run the authoritative list after a user invalidation wake-up. */
  refreshToken?: number
  /** Replace every server-backed run after the complete paginated listing
   * succeeds. Failed/aborted listings never prune local state. */
  onReplace: (summaries: ResearchRunSummary[]) => void
  /** The workspace namespace every run operation scopes to -- the per-user
   * project namespace when authenticated, the browser-local id otherwise.
   * Resolved by the parent (after server discovery + the auth session), so a run
   * is created/listed/cancelled under the same namespace the project data uses
   * and reports follow the user across devices. Flips to the namespace in
   * lockstep with `canList`, so the first list/submit already uses it. */
  workspaceId: string
}

/**
 * Run operations (list/create/cancel/result/stream) for the research desk,
 * scoped to a workspace namespace the PARENT resolves. Server discovery
 * (health/capabilities/stacks) lives in {@link useServerDiscovery}; splitting it
 * out lets the parent resolve the auth session and the per-user namespace before
 * this hook runs, so the run scope is the namespace from the first list/submit
 * with no in-hook session re-probe and no browser-id window.
 */
export function useResearchRunApi({
  apiKey,
  canList,
  enabled,
  onEvent,
  onResult,
  onRunError,
  onReplace,
  onSummary,
  refreshToken = 0,
  workspaceId,
}: UseResearchRunApiOptions) {
  const [lastError, setLastError] = useState<string | null>(null)
  // Initial run-list hydration has SETTLED (success or error): workspaces
  // render a loading skeleton instead of an empty state until this flips,
  // so a reload does not flash "no runs" while pages stream in.
  const [runsHydrated, setRunsHydrated] = useState(false)
  // Runs currently on the polling fallback (plan M1 T1) — the timeline
  // shows a visible degradation hint for these.
  const [pollingRunIds, setPollingRunIds] = useState<string[]>([])
  const streamsRef = useRef(new Map<string, AbortController>())
  const replayedTerminalRunIdsRef = useRef(new Set<string>())
  const perRunCallbacksRef = useRef(new Map<string, PerRunCallbacks>())
  const streamScopeRef = useRef<{
    apiKey: string | undefined
    workspaceId: string
  } | null>(null)
  const callbacksRef = useRef<LiveRunCallbacks>({
    onEvent,
    onResult,
    onRunError,
    onSummary,
  })

  useEffect(() => {
    callbacksRef.current = {
      onEvent,
      onResult,
      onRunError,
      onSummary,
    }
  }, [onEvent, onResult, onRunError, onSummary])

  const loadResult = useCallback(async (runId: string) => {
    try {
      const result = await fetchResearchRunResult(runId, { apiKey, workspaceId })
      const callbacks = perRunCallbacksRef.current.get(runId) ?? callbacksRef.current
      callbacks.onResult?.(result)
    } catch (error) {
      const callbacks = perRunCallbacksRef.current.get(runId) ?? callbacksRef.current
      callbacks.onRunError?.(runId, messageFromError(error))
    } finally {
      perRunCallbacksRef.current.delete(runId)
    }
  }, [apiKey, workspaceId])

  const replayTerminalAgentEvents = useCallback(async (
    summary: ResearchRunSummary,
  ) => {
    if (replayedTerminalRunIdsRef.current.has(summary.run_id)) return
    replayedTerminalRunIdsRef.current.add(summary.run_id)
    try {
      await replayTerminalEventPages({
        fetchPage: (afterSequence) => fetchRunEventsPage(
          summary.events_url,
          afterSequence,
          { apiKey, workspaceId },
        ),
        onEvent: (event) => {
          const callbacks = perRunCallbacksRef.current.get(summary.run_id)
            ?? callbacksRef.current
          callbacks.onEvent?.(event)
        },
      })
    } catch (error) {
      const callbacks = perRunCallbacksRef.current.get(summary.run_id)
        ?? callbacksRef.current
      callbacks.onRunError?.(summary.run_id, messageFromError(error))
    } finally {
      if (summary.status === 'completed') {
        void loadResult(summary.run_id)
      } else {
        perRunCallbacksRef.current.delete(summary.run_id)
      }
    }
  }, [apiKey, loadResult, workspaceId])

  const startStream = useCallback((summary: ResearchRunSummary) => {
    // Child runs are projected through their parent stream. A child channel is
    // opened only when its work unit is explicitly inspected.
    if (summary.kind === 'agent_child') return
    if (streamsRef.current.has(summary.run_id)) return
    if (terminalStatus(summary.status)) {
      if (shouldReplayTerminalAgentEvents(summary)) {
        void replayTerminalAgentEvents(summary)
      } else if (summary.status === 'completed') {
        void loadResult(summary.run_id)
      } else {
        perRunCallbacksRef.current.delete(summary.run_id)
      }
      return
    }

    const controller = new AbortController()
    streamsRef.current.set(summary.run_id, controller)
    void subscribeRunEvents({
      eventsUrl: summary.events_url,
      options: { apiKey, workspaceId },
      signal: controller.signal,
      // Visible degradation (plan M1 T1): the timeline shows a hint
      // while the run is on the polling fallback; recovery clears it.
      onTransportChange: (transport) => {
        setPollingRunIds((current) => {
          const isPolling = transport === 'polling'
          if (current.includes(summary.run_id) === isPolling) return current
          return isPolling
            ? [...current, summary.run_id]
            : current.filter((id) => id !== summary.run_id)
        })
      },
      onEvent: (event) => {
        const callbacks = perRunCallbacksRef.current.get(event.run_id) ?? callbacksRef.current
        callbacks.onEvent?.(event)
        if (event.type === 'inqtrix.run.completed') {
          void loadResult(event.run_id)
        } else if (event.type === 'inqtrix.run.failed' || event.type === 'inqtrix.run.cancelled') {
          perRunCallbacksRef.current.delete(event.run_id)
        }
      },
    }).catch((error) => {
      if (controller.signal.aborted) return
      const callbacks = perRunCallbacksRef.current.get(summary.run_id) ?? callbacksRef.current
      callbacks.onRunError?.(summary.run_id, messageFromError(error))
      perRunCallbacksRef.current.delete(summary.run_id)
    }).finally(() => {
      streamsRef.current.delete(summary.run_id)
      setPollingRunIds((current) =>
        current.includes(summary.run_id)
          ? current.filter((id) => id !== summary.run_id)
          : current)
    })
  }, [apiKey, loadResult, replayTerminalAgentEvents, workspaceId])

  const submitRun = useCallback(async (
    request: CreateResearchRunRequest,
    options?: {
      /** Select the new run in the research workspace (default true);
       * knowledge asks pass false — their surface is the Wissen thread. */
      select?: boolean
      /** Invoked with the accepted summary BEFORE the event stream
       * starts, so callers can register run-id-keyed state without
       * racing the first SSE event. */
      onCreated?: (summary: ResearchRunSummary) => void
      /** Per-run callback override for ephemeral surfaces such as incognito
       * Knowledge asks. Omitted callbacks are intentionally not filled from the
       * global store callbacks, so those runs stay out of persisted state. */
      callbacks?: PerRunCallbacks
      /** Prevent the accepted summary from entering the global run store. */
      suppressSummary?: boolean
    },
  ): Promise<ResearchRunSummary | null> => {
    try {
      setLastError(null)
      const summary = await createResearchRun(request, { apiKey, workspaceId })
      if (options?.callbacks) {
        perRunCallbacksRef.current.set(summary.run_id, options.callbacks)
      }
      if (!options?.suppressSummary) {
        callbacksRef.current.onSummary(summary, { select: options?.select ?? true })
      } else {
        options?.callbacks?.onSummary?.(summary, { select: options?.select ?? true })
      }
      options?.onCreated?.(summary)
      startStream(summary)
      return summary
    } catch (error) {
      const message = messageFromError(error)
      setLastError(message)
      console.warn('Inqtrix run creation failed.', error)
      return null
    }
  }, [apiKey, startStream, workspaceId])

  const cancelRun = useCallback(async (runId: string) => {
    try {
      setLastError(null)
      const summary = await cancelResearchRun(runId, { apiKey, workspaceId })
      callbacksRef.current.onSummary(summary)
      startStream(summary)
    } catch (error) {
      const message = messageFromError(error)
      setLastError(message)
      throw new Error(message, { cause: error })
    }
  }, [apiKey, startStream, workspaceId])

  const deleteRun = useCallback(async (runId: string, options?: { cancelIfActive?: boolean }) => {
    // Stop any live stream first, then delete durably on the server. The
    // caller removes it from local state only after this resolves, so a
    // failed delete never leaves the UI claiming a run is gone while the
    // store still has it (it would re-appear on the next reload).
    const controller = streamsRef.current.get(runId)
    if (controller) {
      controller.abort()
      streamsRef.current.delete(runId)
    }
    perRunCallbacksRef.current.delete(runId)
    try {
      setLastError(null)
      await deleteResearchRun(runId, { apiKey, workspaceId })
    } catch (error) {
      // A 404 means the run is already absent server-side; for a delete that
      // is idempotent success, not a failure to surface. Anything else (e.g.
      // 409 while still active) is a real error the caller must see.
      if (hasHttpStatus(error, 404)) return
      if (options?.cancelIfActive && hasHttpStatus(error, 409)) {
        let cancelled: ResearchRunSummary | null = null
        try {
          try {
            cancelled = await cancelResearchRun(runId, { apiKey, workspaceId })
          } catch (cancelError) {
            if (!hasHttpStatus(cancelError, 404)) throw cancelError
            return
          }
          // Keep the card honest while we wait: the summary now carries
          // cancel_requested, which renders the "cancelling" badge.
          callbacksRef.current.onSummary(cancelled)
          if (!isTerminalRunStatus(cancelled.status)) {
            // A cancel of a RUNNING run is asynchronous (the worker stops
            // at its next checkpoint); only queued/waiting runs cancel to
            // a terminal state synchronously. Wait for the transition
            // instead of retrying the delete into another 409.
            let terminal: ResearchRunSummary | null
            try {
              terminal = await waitForRunTerminal({
                fetchSummary: () =>
                  fetchResearchRunSummary(runId, { apiKey, workspaceId }),
              })
            } catch (waitError) {
              // The run vanished while waiting (expired/foreign delete):
              // the goal state is reached.
              if (hasHttpStatus(waitError, 404)) return
              throw waitError
            }
            if (terminal === null) throw new RunStillCancellingError(runId)
          }
          try {
            await deleteResearchRun(runId, { apiKey, workspaceId })
            return
          } catch (deleteError) {
            if (hasHttpStatus(deleteError, 404)) return
            throw deleteError
          }
        } catch (flowError) {
          // The run survives a failed cancel-and-delete flow, but its live
          // stream was aborted above — reattach it so the card keeps
          // following the run (and flips to "cancelled" once the worker
          // stops) instead of freezing on the "cancelling" badge.
          if (cancelled !== null) startStream(cancelled)
          if (flowError instanceof RunStillCancellingError) {
            // Surfaced once by the caller on the run card; no banner.
            throw flowError
          }
          const message = messageFromError(flowError)
          setLastError(message)
          throw new Error(message, { cause: flowError })
        }
      }
      const message = messageFromError(error)
      setLastError(message)
      throw new Error(message, { cause: error })
    }
  }, [apiKey, startStream, workspaceId])

  useEffect(() => {
    if (!enabled || !canList) {
      // Not admitted (disabled, no apikey, or no/lost session): abort any live
      // streams and list nothing. Re-running with canList true re-hydrates.
      for (const controller of streamsRef.current.values()) {
        controller.abort()
      }
      streamsRef.current.clear()
      perRunCallbacksRef.current.clear()
      replayedTerminalRunIdsRef.current.clear()
      streamScopeRef.current = null
      setRunsHydrated(false)
      return undefined
    }

    let ignore = false
    const listController = new AbortController()
    const previousScope = streamScopeRef.current
    const scopeChanged = previousScope === null
      || previousScope.apiKey !== apiKey
      || previousScope.workspaceId !== workspaceId
    streamScopeRef.current = { apiKey, workspaceId }
    if (scopeChanged) {
      for (const controller of streamsRef.current.values()) controller.abort()
      streamsRef.current.clear()
      perRunCallbacksRef.current.clear()
      replayedTerminalRunIdsRef.current.clear()
      setRunsHydrated(false)
    }

    async function hydrate() {
      try {
        let cursor: string | undefined
        const summaries: ResearchRunSummary[] = []
        do {
          const page = await listResearchRuns({
            apiKey,
            cursor,
            limit: RUN_PAGE_LIMIT,
            signal: listController.signal,
            workspaceId,
          })
          if (ignore) return
          summaries.push(...page.data)
          cursor = page.next_cursor ?? undefined
        } while (cursor)
        if (ignore) return
        onReplace(summaries)
        const visibleIds = new Set(summaries.map((summary) => summary.run_id))
        for (const [runId, controller] of streamsRef.current) {
          if (visibleIds.has(runId)) continue
          controller.abort()
          streamsRef.current.delete(runId)
          perRunCallbacksRef.current.delete(runId)
        }
        for (const summary of summaries) startStream(summary)
        setLastError(null)
      } catch (error) {
        if (!ignore && !listController.signal.aborted) {
          setLastError(messageFromError(error))
        }
      } finally {
        // "Settled", not "succeeded": a failed listing surfaces via
        // lastError — the loading state must still end either way.
        if (!ignore) setRunsHydrated(true)
      }
    }

    void hydrate()

    return () => {
      ignore = true
      listController.abort()
    }
  }, [apiKey, canList, enabled, onReplace, refreshToken, startStream, workspaceId])

  useEffect(() => {
    return () => {
      for (const controller of streamsRef.current.values()) {
        controller.abort()
      }
      streamsRef.current.clear()
      perRunCallbacksRef.current.clear()
      replayedTerminalRunIdsRef.current.clear()
    }
  }, [])

  return {
    cancelRun,
    deleteRun,
    lastError,
    pollingRunIds,
    runsHydrated,
    submitRun,
  }
}

function terminalStatus(status: ResearchRunSummary['status']) {
  return status === 'completed'
    || status === 'failed'
    || status === 'cancelled'
    || status === 'expired'
}

/** Terminal root-agent summaries need their durable event story replayed after
 * hydration. Child runs stay parent-projected; standard runs need only their
 * result payload. */
export function shouldReplayTerminalAgentEvents(
  summary: Pick<ResearchRunSummary, 'kind' | 'mode' | 'status'>,
): boolean {
  return summary.kind !== 'agent_child'
    && (summary.mode === 'workspace_agent' || summary.mode === 'agent_kernel')
    && terminalStatus(summary.status)
}

export async function replayTerminalEventPages({
  fetchPage,
  onEvent,
}: {
  fetchPage: (afterSequence: number | null) => Promise<{
    data: ResearchRunEvent[]
    terminal: boolean
  }>
  onEvent: (event: ResearchRunEvent) => void
}): Promise<void> {
  let afterSequence: number | null = null
  for (;;) {
    const page = await fetchPage(afterSequence)
    for (const event of page.data) onEvent(event)
    if (page.terminal) return
    const latest = page.data.at(-1)?.sequence
    if (latest === undefined || latest === afterSequence) {
      throw new Error('Terminal agent event replay ended before a terminal page.')
    }
    afterSequence = latest
  }
}

/** Thrown by `deleteRun` when a cancelled run did not reach a terminal
 * state within the bounded wait. The message is diagnostic only: callers
 * detect the class and render their own localized copy. */
export class RunStillCancellingError extends Error {
  constructor(runId: string) {
    super(`run ${runId} is still cancelling; retry the delete once it stopped`)
    this.name = 'RunStillCancellingError'
  }
}

/** Poll one run's summary until it reaches a terminal status.
 *
 * Returns the terminal summary, or `null` once `maxWaitMs` is exhausted.
 * The bound caps only how long the UI auto-waits before surfacing a
 * visible "still cancelling" error — it never truncates a server-side
 * operation, so it intentionally needs no capabilities-derived budget
 * (unlike request timeouts, see researchRuns/clientTimeouts.ts). Expected
 * post-cancel latency is seconds; 30s covers a slow in-flight provider
 * attempt without leaving the user staring at a silent spinner forever.
 */
export async function waitForRunTerminal({
  fetchSummary,
  pollMs = 2000,
  maxWaitMs = 30000,
  sleep = (ms: number) =>
    new Promise<void>((resolve) => {
      window.setTimeout(resolve, ms)
    }),
}: {
  fetchSummary: () => Promise<ResearchRunSummary>
  pollMs?: number
  maxWaitMs?: number
  sleep?: (ms: number) => Promise<void>
}): Promise<ResearchRunSummary | null> {
  let waitedMs = 0
  for (;;) {
    const summary = await fetchSummary()
    if (isTerminalRunStatus(summary.status)) return summary
    if (waitedMs >= maxWaitMs) return null
    await sleep(pollMs)
    waitedMs += pollMs
  }
}

function messageFromError(error: unknown) {
  return error instanceof Error ? error.message : 'Inqtrix request failed.'
}
