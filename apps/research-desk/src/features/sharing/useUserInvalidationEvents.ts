import { useEffect, useState } from 'react'
import {
  fetchAuthSession,
  hasHttpStatus,
  streamUserEvents,
  type AuthSessionInfo,
  type UserEvent,
} from '@/api/inqtrixClient'

export type UserEventAction = 'consume' | 'ignore' | 'refetch' | 'reload'

const COLLABORATION_TRANSPORT_SCOPES = new Set([
  'collaboration_comment_changed',
  'collaboration_comment_mention',
])

/**
 * Translate the content-free user event into a shell action. The ready frame
 * is also the cross-tab/account boundary: a stream for another user must never
 * wake stores under the currently rendered identity.
 */
export function userEventAction(
  event: UserEvent,
  expectedUserId: string,
): UserEventAction {
  if (event.type === 'ready') {
    return event.data.user_id === expectedUserId ? 'refetch' : 'reload'
  }
  if (
    event.type === 'invalidate'
    && COLLABORATION_TRANSPORT_SCOPES.has(event.data.scope)
  ) return 'consume'
  if (event.type === 'invalidate' || event.type === 'reset') return 'refetch'
  return 'ignore'
}

/** Bounded reconnect delay; exported so the transport policy stays testable. */
export function userEventReconnectDelay(attempt: number): number {
  return Math.min(30_000, 1_000 * 2 ** Math.min(Math.max(attempt, 0), 5))
}

export type IdentityVerification = {
  action: 'refetch' | 'reload' | 'retain'
  error: string | null
}

/** Advance the replay cursor only after the current identity was confirmed. */
export function confirmedUserEventCursor(
  current: string | undefined,
  event: UserEvent,
  verification: IdentityVerification,
): string | undefined {
  if (verification.action !== 'refetch') return current
  if (event.type === 'reset') return undefined
  return event.id ?? current
}

/** Resolve one canonical-session probe only if it is still the latest probe.
 * A network failure retains the already-rendered identity but authorizes no
 * resource read; malformed/HTTP/anonymous/mismatched responses reload because
 * the active browser identity is no longer trustworthy. */
export async function verifyLatestUserIdentity({
  expectedUserId,
  isCurrent,
  probe,
}: {
  expectedUserId: string
  isCurrent: () => boolean
  probe: () => Promise<AuthSessionInfo>
}): Promise<IdentityVerification | null> {
  try {
    const session = await probe()
    if (!isCurrent()) return null
    if (!session.authenticated || session.user.id !== expectedUserId) {
      return { action: 'reload', error: null }
    }
    return { action: 'refetch', error: null }
  } catch (error) {
    if (!isCurrent()) return null
    if (error instanceof TypeError) {
      return {
        action: 'retain',
        error: error.message || 'User identity verification is offline.',
      }
    }
    return { action: 'reload', error: null }
  }
}

/**
 * Maintain the one user-scoped invalidation stream. Its only output is a
 * monotonically increasing revision: consumers re-read their authoritative
 * list endpoints. No resource content is ever trusted or patched from SSE.
 */
export function useUserInvalidationEvents({
  enabled,
  userId,
}: {
  enabled: boolean
  userId: string | null
}): { error: string | null; revision: number } {
  const [revision, setRevision] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!enabled || !userId) {
      setError(null)
      return
    }

    const controller = new AbortController()
    let lastEventId: string | undefined
    let reconnectAttempt = 0
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let wakeTimer: ReturnType<typeof setTimeout> | null = null
    let identityRetryTimer: ReturnType<typeof setTimeout> | null = null
    let stopped = false
    let identityProbeGeneration = 0
    let identityRetryAttempt = 0
    let pendingInvalidation: UserEvent | null = null

    const scheduleWake = () => {
      if (wakeTimer !== null) clearTimeout(wakeTimer)
      wakeTimer = setTimeout(() => {
        wakeTimer = null
        if (!stopped) setRevision((current) => current + 1)
      }, 75)
    }

    const verifyIdentityAndWake = async () => {
      if (identityRetryTimer !== null) {
        clearTimeout(identityRetryTimer)
        identityRetryTimer = null
      }
      const generation = identityProbeGeneration + 1
      identityProbeGeneration = generation
      const verification = await verifyLatestUserIdentity({
        expectedUserId: userId,
        isCurrent: () =>
          !stopped
          && !controller.signal.aborted
          && generation === identityProbeGeneration,
        probe: () => fetchAuthSession({ signal: controller.signal }),
      })
      if (verification === null) return
      if (verification.action === 'reload') {
        stopped = true
        controller.abort()
        window.location.reload()
        return
      }
      if (verification.action === 'retain') {
        setError(verification.error)
        console.warn(
          'User identity could not be verified; resource refresh was withheld.',
          verification.error,
        )
        if (pendingInvalidation !== null) {
          const delay = userEventReconnectDelay(identityRetryAttempt)
          identityRetryAttempt += 1
          identityRetryTimer = setTimeout(() => {
            identityRetryTimer = null
            void verifyIdentityAndWake()
          }, delay)
        }
        return
      }
      identityRetryAttempt = 0
      if (pendingInvalidation !== null) {
        lastEventId = confirmedUserEventCursor(
          lastEventId,
          pendingInvalidation,
          verification,
        )
        pendingInvalidation = null
      }
      setError(null)
      scheduleWake()
    }

    const handleFocus = () => {
      if (document.visibilityState === 'visible') void verifyIdentityAndWake()
    }
    const handleOnline = () => void verifyIdentityAndWake()
    window.addEventListener('focus', handleFocus)
    window.addEventListener('online', handleOnline)

    const connect = async () => {
      if (stopped || controller.signal.aborted) return
      try {
        await streamUserEvents({
          lastEventId,
          signal: controller.signal,
          onEvent: (event) => {
            if (stopped || controller.signal.aborted) return
            const action = userEventAction(event, userId)
            if (action === 'reload') {
              stopped = true
              controller.abort()
              window.location.reload()
              return
            }
            if (action === 'consume') {
              // Collaboration comment events are already delivered to the
              // document room by the sidecar. Consuming the duplicate
              // user-outbox coordinate here prevents one comment from
              // reloading runs, skills, knowledge, folders, documents and
              // auth state in every open participant tab.
              lastEventId = event.id ?? lastEventId
              return
            }
            if (event.type === 'ready') {
              identityProbeGeneration += 1
              pendingInvalidation = null
              identityRetryAttempt = 0
              if (identityRetryTimer !== null) {
                clearTimeout(identityRetryTimer)
                identityRetryTimer = null
              }
              lastEventId = event.id ?? event.data.cursor
            }
            if (action === 'refetch') {
              reconnectAttempt = 0
              if (event.type === 'ready') {
                setError(null)
                scheduleWake()
              } else {
                // The browser cookie may have changed in another tab after
                // this stream was opened. Re-probe before any list request so
                // another account can never merge into the rendered stores.
                pendingInvalidation = event
                void verifyIdentityAndWake()
              }
            } else {
              console.warn('Unknown user invalidation event ignored.', event)
            }
          },
        })
        if (!stopped && !controller.signal.aborted) {
          setError('User invalidation stream closed; reconnecting.')
          console.warn('User invalidation stream closed; reconnecting.')
        }
      } catch (error) {
        if (stopped || controller.signal.aborted) return
        if (hasHttpStatus(error, 401) || hasHttpStatus(error, 403)) {
          stopped = true
          controller.abort()
          window.location.reload()
          return
        }
        setError(error instanceof Error ? error.message : String(error))
        console.warn('User invalidation stream failed; reconnecting.', error)
      }

      if (stopped || controller.signal.aborted) return
      const delay = userEventReconnectDelay(reconnectAttempt)
      reconnectAttempt += 1
      reconnectTimer = setTimeout(() => void connect(), delay)
    }

    void connect()
    return () => {
      stopped = true
      controller.abort()
      if (reconnectTimer !== null) clearTimeout(reconnectTimer)
      if (wakeTimer !== null) clearTimeout(wakeTimer)
      if (identityRetryTimer !== null) clearTimeout(identityRetryTimer)
      window.removeEventListener('focus', handleFocus)
      window.removeEventListener('online', handleOnline)
    }
  }, [enabled, userId])

  return { error, revision }
}
