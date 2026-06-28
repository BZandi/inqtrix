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
  fetchResearchRunResult,
  hasHttpStatus,
  listResearchRuns,
  streamResearchRunEvents,
} from '@/api/inqtrixClient'
import type {
  CreateResearchRunRequest,
  ResearchRunEvent,
  ResearchRunResult,
  ResearchRunSummary,
} from './types'

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
  onSummary,
  workspaceId,
}: UseResearchRunApiOptions) {
  const [lastError, setLastError] = useState<string | null>(null)
  const streamsRef = useRef(new Map<string, AbortController>())
  const perRunCallbacksRef = useRef(new Map<string, PerRunCallbacks>())
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

  const startStream = useCallback((summary: ResearchRunSummary) => {
    if (streamsRef.current.has(summary.run_id)) return
    if (terminalStatus(summary.status)) {
      if (summary.status === 'completed') {
        void loadResult(summary.run_id)
      } else {
        perRunCallbacksRef.current.delete(summary.run_id)
      }
      return
    }

    const controller = new AbortController()
    streamsRef.current.set(summary.run_id, controller)
    void streamResearchRunEvents(summary.events_url, {
      apiKey,
      signal: controller.signal,
      workspaceId,
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
    })
  }, [apiKey, loadResult, workspaceId])

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
        try {
          await cancelResearchRun(runId, { apiKey, workspaceId })
        } catch (cancelError) {
          if (!hasHttpStatus(cancelError, 404)) throw cancelError
          return
        }
        try {
          await deleteResearchRun(runId, { apiKey, workspaceId })
          return
        } catch (deleteError) {
          if (hasHttpStatus(deleteError, 404)) return
          throw deleteError
        }
      }
      const message = messageFromError(error)
      setLastError(message)
      throw new Error(message, { cause: error })
    }
  }, [apiKey, workspaceId])

  useEffect(() => {
    if (!enabled || !canList) {
      // Not admitted (disabled, no apikey, or no/lost session): abort any live
      // streams and list nothing. Re-running with canList true re-hydrates.
      for (const controller of streamsRef.current.values()) {
        controller.abort()
      }
      streamsRef.current.clear()
      perRunCallbacksRef.current.clear()
      return undefined
    }

    let ignore = false
    for (const controller of streamsRef.current.values()) {
      controller.abort()
    }
    streamsRef.current.clear()
    perRunCallbacksRef.current.clear()

    async function hydrate() {
      try {
        const summaries = await listResearchRuns({ apiKey, workspaceId })
        if (ignore) return
        for (const summary of summaries) {
          callbacksRef.current.onSummary(summary)
          startStream(summary)
        }
      } catch (error) {
        if (!ignore) setLastError(messageFromError(error))
      }
    }

    void hydrate()

    return () => {
      ignore = true
    }
  }, [apiKey, canList, enabled, startStream, workspaceId])

  useEffect(() => {
    return () => {
      for (const controller of streamsRef.current.values()) {
        controller.abort()
      }
      streamsRef.current.clear()
      perRunCallbacksRef.current.clear()
    }
  }, [])

  return {
    cancelRun,
    deleteRun,
    lastError,
    submitRun,
  }
}

function terminalStatus(status: ResearchRunSummary['status']) {
  return status === 'completed'
    || status === 'failed'
    || status === 'cancelled'
    || status === 'expired'
}

function messageFromError(error: unknown) {
  return error instanceof Error ? error.message : 'Inqtrix request failed.'
}
