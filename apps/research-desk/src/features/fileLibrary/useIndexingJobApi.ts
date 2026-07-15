import { useCallback, useEffect, useRef } from 'react'
import {
  cancelIndexingJob,
  hasHttpStatus,
  listIndexingJobs,
  startIndexingJob,
  streamIndexingJobEvents,
} from '@/api/inqtrixClient'
import {
  isTerminalIndexingStatus,
  type IndexingJobEvent,
  type IndexingJobSnapshot,
  type IndexingJobSummary,
} from './indexingTypes'

/**
 * Drives the server-backed reindex lifecycle, mirroring
 * {@link useResearchRunApi}: start a job, stream its progress via SSE,
 * cancel it, and re-attach to in-flight jobs on mount (so a reindex
 * survives closing and reopening the app). Events are keyed by job id and
 * resolved through the authoritative parent `collection_id`; callers may map
 * that collection onto local presentation state. Progress is throttled so a
 * large re-embed never floods the store.
 */
type IndexingCallbacks = {
  onCancelled: (collectionId: string) => void
  onComplete: (collectionId: string) => void
  /** A single document finished embedding (server document id) — flips just
   * that file's row live, so a re-embed no longer flips all files together. */
  onDocumentCompleted: (collectionId: string, documentId: string) => void
  onError: (collectionId: string, message: string) => void
  onProgress: (
    collectionId: string,
    completedDocuments: number,
    totalDocuments: number,
    currentDocumentTitle?: string,
  ) => void
  onQueued: (collectionId: string, queuePosition: number | null) => void
  onStart: (
    collectionId: string,
    jobId: string,
    totalDocuments: number,
    status: 'cancelling' | 'queued' | 'running',
  ) => void
}

type UseIndexingJobApiOptions = IndexingCallbacks & {
  apiKey?: string
  enabled: boolean
  /** Cookie-session auth flip re-runs resume hydration (no remount). */
  sessionAuthed?: boolean
  /** User invalidation generation: re-list jobs started by another authorized
   * editor without adding a second polling or per-resource stream. */
  refreshToken?: number
  workspaceId: string
}

const PROGRESS_THROTTLE_MS = 150

export function useIndexingJobApi({
  apiKey,
  enabled,
  onCancelled,
  onComplete,
  onDocumentCompleted,
  onError,
  onProgress,
  onQueued,
  onStart,
  refreshToken = 0,
  sessionAuthed,
  workspaceId,
}: UseIndexingJobApiOptions) {
  const streamsRef = useRef(new Map<string, AbortController>())
  const jobCollectionRef = useRef(new Map<string, string>())
  const lastProgressRef = useRef(new Map<string, number>())
  const callbacksRef = useRef<IndexingCallbacks>({
    onCancelled,
    onComplete,
    onDocumentCompleted,
    onError,
    onProgress,
    onQueued,
    onStart,
  })

  useEffect(() => {
    callbacksRef.current = {
      onCancelled,
      onComplete,
      onDocumentCompleted,
      onError,
      onProgress,
      onQueued,
      onStart,
    }
  }, [onCancelled, onComplete, onDocumentCompleted, onError, onProgress, onQueued, onStart])

  const handleEvent = useCallback((event: IndexingJobEvent) => {
    const collectionId = jobCollectionRef.current.get(event.job_id)
    if (!collectionId) return
    const snapshot = (event.data?.snapshot ?? {}) as IndexingJobSnapshot
    const emitProgress = () =>
      callbacksRef.current.onProgress(
        collectionId,
        snapshot.completed_documents ?? 0,
        snapshot.total_documents ?? 0,
        snapshot.current_document_title,
      )
    if (
      event.type === 'inqtrix.index.progress'
      || event.type === 'inqtrix.index.started'
    ) {
      const now = Date.now()
      const last = lastProgressRef.current.get(event.job_id) ?? 0
      if (now - last < PROGRESS_THROTTLE_MS) return
      lastProgressRef.current.set(event.job_id, now)
      emitProgress()
    } else if (event.type === 'inqtrix.index.completed') {
      // Flush the terminal snapshot un-throttled first: the final progress
      // tick may have been dropped by the throttle, and the history entry
      // reads the live counts when it is recorded.
      emitProgress()
      callbacksRef.current.onComplete(collectionId)
    } else if (event.type === 'inqtrix.index.failed') {
      emitProgress()
      const error = (event.data?.error ?? {}) as { message?: string }
      callbacksRef.current.onError(collectionId, error.message ?? 'Indizierung fehlgeschlagen.')
    } else if (event.type === 'inqtrix.index.cancelled') {
      emitProgress()
      callbacksRef.current.onCancelled(collectionId)
    } else if (event.type === 'inqtrix.index.queued') {
      const position = event.data?.queue_position
      callbacksRef.current.onQueued(
        collectionId,
        typeof position === 'number' ? position : null,
      )
    } else if (event.type === 'inqtrix.index.document_completed') {
      // Per-document flip — NOT throttled (each file should land) and it carries
      // no counts, so it never touches the progress bar.
      const documentId = event.data?.document_id
      if (typeof documentId === 'string') {
        callbacksRef.current.onDocumentCompleted(collectionId, documentId)
      }
    }
  }, [])

  const startStream = useCallback((summary: IndexingJobSummary) => {
    jobCollectionRef.current.set(summary.job_id, summary.collection_id)
    if (streamsRef.current.has(summary.job_id)) return
    if (isTerminalIndexingStatus(summary.status)) {
      // Terminal already (resume of a finished job): reflect the outcome.
      if (summary.status === 'completed') {
        callbacksRef.current.onComplete(summary.collection_id)
      } else if (summary.status === 'failed') {
        callbacksRef.current.onError(
          summary.collection_id,
          summary.error?.message ?? 'Indizierung fehlgeschlagen.',
        )
      } else if (summary.status === 'cancelled') {
        callbacksRef.current.onCancelled(summary.collection_id)
      }
      return
    }
    const controller = new AbortController()
    streamsRef.current.set(summary.job_id, controller)
    const collectionId = summary.collection_id
    void streamIndexingJobEvents(summary.events_url, {
      apiKey,
      signal: controller.signal,
      workspaceId,
      onEvent: handleEvent,
    })
      .catch((error) => {
        if (controller.signal.aborted) return
        callbacksRef.current.onError(collectionId, messageFromError(error))
      })
      .finally(() => {
        streamsRef.current.delete(summary.job_id)
        lastProgressRef.current.delete(summary.job_id)
      })
  }, [apiKey, handleEvent, workspaceId])

  const startReindex = useCallback(async (collectionId: string) => {
    const summary = await startIndexingJob(
      collectionId,
      {},
      { apiKey, workspaceId },
    )
    callbacksRef.current.onStart(
      collectionId,
      summary.job_id,
      summary.total_documents,
      activeStatus(summary.status),
    )
    startStream(summary)
    return summary
  }, [apiKey, startStream, workspaceId])

  const cancelReindex = useCallback(async (jobId: string) => {
    const summary = await cancelIndexingJob(jobId, { apiKey, workspaceId })
    // Re-attach to observe the terminal cancelled event (or the queued
    // job's immediate cancellation).
    startStream(summary)
  }, [apiKey, startStream, workspaceId])

  useEffect(() => {
    if (!enabled) {
      for (const controller of streamsRef.current.values()) controller.abort()
      streamsRef.current.clear()
      jobCollectionRef.current.clear()
      lastProgressRef.current.clear()
      return undefined
    }
    let ignore = false
    for (const controller of streamsRef.current.values()) controller.abort()
    streamsRef.current.clear()

    async function hydrate() {
      try {
        const jobs = await listIndexingJobs({ apiKey, workspaceId })
        if (ignore) return
        for (const summary of jobs) {
          if (!isTerminalIndexingStatus(summary.status)) {
            callbacksRef.current.onStart(
              summary.collection_id,
              summary.job_id,
              summary.total_documents,
              activeStatus(summary.status),
            )
          }
          startStream(summary)
        }
      } catch (error) {
        // 404 = no indexing surface (knowledge disabled / older backend):
        // expected, leave reindex affordances idle. Any other failure is a
        // real fault and must be visible (No-Silent-Fallbacks).
        if (!ignore && !hasHttpStatus(error, 404)) {
          console.warn('Inqtrix indexing-job resume failed.', error)
        }
      }
    }

    void hydrate()
    return () => {
      ignore = true
    }
  }, [apiKey, enabled, refreshToken, sessionAuthed, startStream, workspaceId])

  useEffect(() => {
    return () => {
      for (const controller of streamsRef.current.values()) controller.abort()
      streamsRef.current.clear()
    }
  }, [])

  return { cancelReindex, startReindex }
}

function messageFromError(error: unknown) {
  return error instanceof Error ? error.message : 'Inqtrix request failed.'
}

function activeStatus(status: IndexingJobSummary['status']): 'cancelling' | 'queued' | 'running' {
  if (status === 'cancelling') return 'cancelling'
  if (status === 'queued') return 'queued'
  return 'running'
}
