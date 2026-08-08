import { useCallback, useEffect, useRef, useState } from 'react'

import {
  getAssetDeletionOperation,
  retryAssetDeletionOperation,
  type ClientOptions,
  type ServerDeletionOperation,
} from '@/api/inqtrixClient'
import {
  assertSessionDeletionOperation,
  SessionDeletionContractError,
  type SessionDeletionState,
} from './sessionDeletion'

const FIRST_POLL_DELAY_MS = 300
const MAX_RETRY_DELAY_MS = 5_000

type SessionRow = { deletion?: SessionDeletionState }

function errorMessage(value: unknown): string {
  return value instanceof Error ? value.message : String(value)
}

function wait(delayMs: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(signal.reason)
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, delayMs)
    signal.addEventListener('abort', () => {
      window.clearTimeout(timer)
      reject(signal.reason)
    }, { once: true })
  })
}

/** Shared Agent/Knowledge session-deletion client over the aggregate deletion
 * ledger. It projects durable tombstones, resumes polling after reload, and
 * only removes a row after the server's terminal receipt. */
export function useSessionDeletionApi({
  enabled,
  onComplete,
  onState,
  options,
  scopeKey,
  sessions,
  start,
  targetKind,
}: {
  enabled: boolean
  onComplete: (sessionId: string, operationId: string | null) => void
  onState: (sessionId: string, state: SessionDeletionState) => void
  options: ClientOptions
  scopeKey: string
  sessions: Readonly<Record<string, SessionRow>>
  start: (
    sessionId: string,
    options: ClientOptions,
  ) => Promise<ServerDeletionOperation | null>
  targetKind: 'agent_session' | 'knowledge_session'
}): {
  deleteSession: (sessionId: string) => Promise<void>
  error: string | null
  retrySession: (sessionId: string) => Promise<void>
} {
  const [error, setError] = useState<string | null>(null)
  const controllersRef = useRef(new Map<string, AbortController>())
  const retryTimersRef = useRef(new Map<string, number>())
  const requestsRef = useRef(new Set<string>())
  const scopeRef = useRef(scopeKey)
  const callbacksRef = useRef({ onComplete, onState, options, start })
  callbacksRef.current = { onComplete, onState, options, start }

  const stop = useCallback((operationId: string) => {
    controllersRef.current.get(operationId)?.abort()
    controllersRef.current.delete(operationId)
    const retryTimer = retryTimersRef.current.get(operationId)
    if (retryTimer !== undefined) window.clearTimeout(retryTimer)
    retryTimersRef.current.delete(operationId)
  }, [])

  const apply = useCallback((
    operation: ServerDeletionOperation,
    expectedSessionId: string,
    scope: string,
  ) => {
    if (scopeRef.current !== scope) return false
    assertSessionDeletionOperation(operation, targetKind, expectedSessionId)
    if (operation.status === 'deleted') {
      stop(operation.operation_id)
      callbacksRef.current.onComplete(expectedSessionId, operation.operation_id)
      return true
    }
    callbacksRef.current.onState(expectedSessionId, {
      error: operation.error?.message ?? null,
      operationId: operation.operation_id,
      stage: operation.stage,
      status: operation.status === 'delete_failed' ? 'delete_failed' : 'deleting',
    })
    if (operation.status === 'delete_failed') stop(operation.operation_id)
    return true
  }, [stop, targetKind])

  const poll = useCallback((
    operationId: string,
    expectedSessionId: string,
    scope: string,
    initialDelay = FIRST_POLL_DELAY_MS,
  ) => {
    if (
      !enabled
      || scopeRef.current !== scope
      || controllersRef.current.has(operationId)
    ) return
    const controller = new AbortController()
    controllersRef.current.set(operationId, controller)
    void (async () => {
      let delayMs = initialDelay
      try {
        while (!controller.signal.aborted && scopeRef.current === scope) {
          await wait(delayMs, controller.signal)
          const operation = await getAssetDeletionOperation(operationId, {
            ...callbacksRef.current.options,
            signal: controller.signal,
          })
          if (!apply(operation, expectedSessionId, scope)) return
          setError(null)
          if (operation.status === 'deleted' || operation.status === 'delete_failed') return
          delayMs = FIRST_POLL_DELAY_MS
        }
      } catch (caught) {
        if (controller.signal.aborted || scopeRef.current !== scope) return
        setError(errorMessage(caught))
        controllersRef.current.delete(operationId)
        if (caught instanceof SessionDeletionContractError) return
        const nextDelay = Math.min(MAX_RETRY_DELAY_MS, delayMs * 2)
        const timer = window.setTimeout(() => {
          retryTimersRef.current.delete(operationId)
          poll(operationId, expectedSessionId, scope, nextDelay)
        }, nextDelay)
        retryTimersRef.current.set(operationId, timer)
      } finally {
        if (controllersRef.current.get(operationId) === controller) {
          controllersRef.current.delete(operationId)
        }
      }
    })()
  }, [apply, enabled])

  const track = useCallback((
    operation: ServerDeletionOperation,
    expectedSessionId: string,
  ) => {
    const scope = scopeRef.current
    if (!apply(operation, expectedSessionId, scope)) return
    if (operation.status === 'queued' || operation.status === 'running') {
      poll(operation.operation_id, expectedSessionId, scope)
    }
  }, [apply, poll])

  const deleteSession = useCallback(async (sessionId: string) => {
    if (requestsRef.current.has(sessionId)) return
    if (!enabled) {
      callbacksRef.current.onComplete(sessionId, null)
      return
    }
    requestsRef.current.add(sessionId)
    try {
      const operation = await callbacksRef.current.start(
        sessionId,
        callbacksRef.current.options,
      )
      if (operation === null) {
        callbacksRef.current.onComplete(sessionId, null)
        return
      }
      setError(null)
      track(operation, sessionId)
    } catch (caught) {
      setError(errorMessage(caught))
      throw caught
    } finally {
      requestsRef.current.delete(sessionId)
    }
  }, [enabled, track])

  const retrySession = useCallback(async (sessionId: string) => {
    const operationId = sessions[sessionId]?.deletion?.operationId
    if (!enabled || !operationId || requestsRef.current.has(sessionId)) return
    requestsRef.current.add(sessionId)
    try {
      const operation = await retryAssetDeletionOperation(
        operationId,
        callbacksRef.current.options,
      )
      setError(null)
      track(operation, sessionId)
    } catch (caught) {
      setError(errorMessage(caught))
      throw caught
    } finally {
      requestsRef.current.delete(sessionId)
    }
  }, [enabled, sessions, track])

  useEffect(() => {
    const scope = scopeKey
    scopeRef.current = scope
    for (const controller of controllersRef.current.values()) controller.abort()
    controllersRef.current.clear()
    for (const timer of retryTimersRef.current.values()) window.clearTimeout(timer)
    retryTimersRef.current.clear()
    requestsRef.current.clear()
    setError(null)
    return () => {
      for (const controller of controllersRef.current.values()) controller.abort()
      controllersRef.current.clear()
      for (const timer of retryTimersRef.current.values()) window.clearTimeout(timer)
      retryTimersRef.current.clear()
      requestsRef.current.clear()
    }
  }, [scopeKey])

  useEffect(() => {
    if (!enabled) return
    const scope = scopeRef.current
    for (const [sessionId, session] of Object.entries(sessions)) {
      if (session.deletion?.status === 'deleting') {
        poll(session.deletion.operationId, sessionId, scope)
      }
    }
  }, [enabled, poll, sessions])

  return { deleteSession, error, retrySession }
}
