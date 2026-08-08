/**
 * Run-event transport with a polling fallback.
 *
 * The app is fetch-first with wake signals (rule R1: events flip stale
 * flags, rows are truth), so the CONTENT never depends on streaming —
 * only the wake channel does. Primary transport is SSE; when the stream
 * fails to open or dies without a terminal event (SSE-buffering
 * proxies, flaky networks), the channel degrades VISIBLY to keyset
 * polling over the same replay buffer (`?format=json&after=N`) and
 * periodically retries SSE in the background, switching back on
 * success. Sequence numbers make the handoff lossless in both
 * directions (events are applied idempotently downstream).
 *
 * The core is dependency-injected so the switching contract is
 * unit-testable in the node-only vitest setup.
 */

import {
  fetchRunEventsPage,
  streamResearchRunEvents,
  type ClientOptions,
} from '@/api/inqtrixClient'
import type { ResearchRunEvent } from './types'

export type RunEventTransport = 'sse' | 'polling'

export type RunEventsPage = {
  events: ResearchRunEvent[]
  terminal: boolean
}

export type RunEventChannelDeps = {
  /** Open the SSE stream after *afterSequence*, delivering every event
   * through *deliver*; resolves when the stream ENDS (`sawTerminal`
   * distinguishes an ordered end from a transport cut); throws when it
   * cannot open / dies hard. */
  streamSse: (
    afterSequence: number | null,
    deliver: (event: ResearchRunEvent) => void,
    attempt: { signal: AbortSignal; onActivity: () => void },
  ) => Promise<{ sawTerminal: boolean }>
  /** One polling page over the same replay buffer. */
  pollOnce: (afterSequence: number | null) => Promise<RunEventsPage>
  onEvent: (event: ResearchRunEvent) => void
  /** Visible degradation hook (No Silent Fallbacks): fired only after
   * the transport is proven active or polling takes over. */
  onTransportChange?: (transport: RunEventTransport) => void
  signal: AbortSignal
  /** Poll cadence while degraded (default 3000 ms). */
  pollIntervalMs?: number
  /** Polls between background SSE retry attempts (default 10). */
  pollsPerSseRetry?: number
  /** Maximum wait for the first SSE bytes (default 15000 ms). */
  firstActivityTimeoutMs?: number
  /** Maximum quiet period after SSE became live (default 30000 ms). */
  inactivityTimeoutMs?: number
  /** Injectable for tests. */
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>
}

function defaultSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = window.setTimeout(done, ms)
    function done() {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }
    function onAbort() {
      window.clearTimeout(timer)
      done()
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

/**
 * Subscribe until a terminal event or abort. Never rejects after the
 * first successful delivery path — transport errors degrade instead.
 * The initial SSE open failure ALSO degrades (the poller then reports
 * a genuine outage loudly by throwing once polling fails too).
 */
export async function subscribeRunEventsWithFallback(
  deps: RunEventChannelDeps,
): Promise<void> {
  const {
    onEvent,
    onTransportChange,
    signal,
    pollIntervalMs = 3000,
    pollsPerSseRetry = 10,
    firstActivityTimeoutMs = 15_000,
    inactivityTimeoutMs = 30_000,
    sleep = defaultSleep,
  } = deps
  let lastSequence: number | null = null
  let reportedTransport: RunEventTransport | null = null
  const reportTransport = (transport: RunEventTransport) => {
    if (reportedTransport === transport) return
    reportedTransport = transport
    onTransportChange?.(transport)
  }
  const deliver = (event: ResearchRunEvent) => {
    if (typeof event.sequence === 'number') {
      lastSequence
        = lastSequence === null
          ? event.sequence
          : Math.max(lastSequence, event.sequence)
    }
    onEvent(event)
  }

  const trySse = async (): Promise<'terminal' | 'ended' | 'failed'> => {
    if (signal.aborted) return 'terminal'
    const attemptController = new AbortController()
    let attemptActive = true
    let activitySeen = false
    let watchdog: number | null = null
    let rejectInterruption: (reason: unknown) => void = () => undefined
    const interruption = new Promise<never>((_resolve, reject) => {
      rejectInterruption = reject
    })
    const clearWatchdog = () => {
      if (watchdog === null) return
      globalThis.clearTimeout(watchdog)
      watchdog = null
    }
    const interrupt = (message: string) => {
      if (!attemptActive) return
      const reason = new DOMException(message, 'TimeoutError')
      attemptController.abort(reason)
      rejectInterruption(reason)
    }
    const armWatchdog = (timeoutMs: number, message: string) => {
      clearWatchdog()
      watchdog = globalThis.setTimeout(
        () => interrupt(message),
        timeoutMs,
      ) as unknown as number
    }
    const onActivity = () => {
      if (!attemptActive) return
      if (!activitySeen) {
        activitySeen = true
        reportTransport('sse')
      }
      armWatchdog(
        inactivityTimeoutMs,
        'Inqtrix run event stream became inactive.',
      )
    }
    const onParentAbort = () => {
      if (!attemptActive) return
      const reason = new DOMException(
        'Inqtrix run event subscription was aborted.',
        'AbortError',
      )
      attemptController.abort(reason)
      rejectInterruption(reason)
    }
    signal.addEventListener('abort', onParentAbort, { once: true })
    armWatchdog(
      firstActivityTimeoutMs,
      'Inqtrix run event stream produced no initial activity.',
    )
    try {
      const { sawTerminal } = await Promise.race([
        deps.streamSse(
          lastSequence,
          (event) => {
            if (!attemptActive) return
            onActivity()
            deliver(event)
          },
          { signal: attemptController.signal, onActivity },
        ),
        interruption,
      ])
      return sawTerminal ? 'terminal' : 'ended'
    } catch (error) {
      if (signal.aborted) return 'terminal'
      console.warn('Inqtrix run event stream failed; polling instead.', error)
      return 'failed'
    } finally {
      attemptActive = false
      clearWatchdog()
      signal.removeEventListener('abort', onParentAbort)
      if (!attemptController.signal.aborted) attemptController.abort()
    }
  }

  let outcome = await trySse()
  while (outcome !== 'terminal' && !signal.aborted) {
    reportTransport('polling')
    let pollsSinceRetry = 0
    for (;;) {
      if (signal.aborted) return
      const page = await deps.pollOnce(lastSequence)
      for (const event of page.events) deliver(event)
      if (page.terminal) return
      pollsSinceRetry += 1
      if (pollsSinceRetry >= pollsPerSseRetry) break
      await sleep(pollIntervalMs, signal)
    }
    if (signal.aborted) return
    outcome = await trySse()
  }
}

const TERMINAL_EVENT_TYPES = new Set([
  'inqtrix.run.completed',
  'inqtrix.run.failed',
  'inqtrix.run.cancelled',
])

/** The client-bound channel: SSE via the shared reader, polling via the
 * `?format=json` page — the ONE subscription entry point for run event
 * consumers (Designprinzip 4). */
export function subscribeRunEvents(params: {
  eventsUrl: string
  options: ClientOptions
  signal: AbortSignal
  onEvent: (event: ResearchRunEvent) => void
  onTransportChange?: (transport: RunEventTransport) => void
}): Promise<void> {
  const separator = params.eventsUrl.includes('?') ? '&' : '?'
  return subscribeRunEventsWithFallback({
    streamSse: async (afterSequence, deliver, attempt) => {
      let sawTerminal = false
      const url
        = afterSequence === null
          ? params.eventsUrl
          : `${params.eventsUrl}${separator}after=${afterSequence}`
      await streamResearchRunEvents(url, {
        ...params.options,
        signal: attempt.signal,
        onActivity: attempt.onActivity,
        onEvent: (event) => {
          if (TERMINAL_EVENT_TYPES.has(event.type)) sawTerminal = true
          deliver(event)
        },
      })
      return { sawTerminal }
    },
    pollOnce: async (afterSequence) => {
      const page = await fetchRunEventsPage(params.eventsUrl, afterSequence, {
        ...params.options,
        signal: params.signal,
      })
      return { events: page.data, terminal: page.terminal }
    },
    onEvent: params.onEvent,
    onTransportChange: params.onTransportChange,
    signal: params.signal,
  })
}
