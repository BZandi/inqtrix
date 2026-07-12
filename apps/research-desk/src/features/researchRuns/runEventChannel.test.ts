import { describe, expect, it } from 'vitest'

import {
  subscribeRunEventsWithFallback,
  type RunEventChannelDeps,
  type RunEventTransport,
} from './runEventChannel'
import type { ResearchRunEvent } from './types'

function event(sequence: number, type = 'inqtrix.agent.activity'): ResearchRunEvent {
  return {
    type,
    run_id: 'run_1',
    sequence,
    created_at: sequence,
    data: {},
  } as ResearchRunEvent
}

const instantSleep: NonNullable<RunEventChannelDeps['sleep']> = () =>
  Promise.resolve()

describe('subscribeRunEventsWithFallback', () => {
  it('stays on SSE when the stream ends with a terminal event', async () => {
    const seen: number[] = []
    const transports: RunEventTransport[] = []
    await subscribeRunEventsWithFallback({
      streamSse: async (_after, deliver) => {
        deliver(event(1))
        deliver(event(2, 'inqtrix.run.completed'))
        return { sawTerminal: true }
      },
      pollOnce: async () => {
        throw new Error('must not poll')
      },
      onEvent: (e) => seen.push(e.sequence),
      onTransportChange: (t) => transports.push(t),
      signal: new AbortController().signal,
      sleep: instantSleep,
    })
    expect(seen).toEqual([1, 2])
    expect(transports).toEqual(['sse'])
  })

  it('degrades to polling after an SSE failure and resumes at the last sequence', async () => {
    const seen: number[] = []
    const transports: RunEventTransport[] = []
    const polledAfter: (number | null)[] = []
    let sseAttempts = 0
    await subscribeRunEventsWithFallback({
      streamSse: async (_after, deliver) => {
        sseAttempts += 1
        deliver(event(1))
        deliver(event(2))
        throw new Error('proxy cut the stream')
      },
      pollOnce: async (after) => {
        polledAfter.push(after)
        return {
          events: [event(3), event(4, 'inqtrix.run.completed')],
          terminal: true,
        }
      },
      onEvent: (e) => seen.push(e.sequence),
      onTransportChange: (t) => transports.push(t),
      signal: new AbortController().signal,
      sleep: instantSleep,
    })
    expect(sseAttempts).toBe(1)
    // No event lost, no duplicate needed: the poll starts AFTER the
    // last delivered sequence.
    expect(polledAfter).toEqual([2])
    expect(seen).toEqual([1, 2, 3, 4])
    expect(transports).toEqual(['sse', 'polling'])
  })

  it('retries SSE after pollsPerSseRetry polls and switches back on success', async () => {
    const transports: RunEventTransport[] = []
    const seen: number[] = []
    let sseAttempts = 0
    let polls = 0
    await subscribeRunEventsWithFallback({
      streamSse: async (_after, deliver) => {
        sseAttempts += 1
        if (sseAttempts === 1) throw new Error('down')
        deliver(event(10, 'inqtrix.run.completed'))
        return { sawTerminal: true }
      },
      pollOnce: async () => {
        polls += 1
        return { events: [event(polls)], terminal: false }
      },
      onEvent: (e) => seen.push(e.sequence),
      onTransportChange: (t) => transports.push(t),
      signal: new AbortController().signal,
      pollsPerSseRetry: 2,
      sleep: instantSleep,
    })
    expect(sseAttempts).toBe(2)
    expect(polls).toBe(2)
    expect(seen).toEqual([1, 2, 10])
    expect(transports).toEqual(['polling', 'sse'])
  })

  it('treats a clean SSE end WITHOUT terminal as a transport cut', async () => {
    const transports: RunEventTransport[] = []
    await subscribeRunEventsWithFallback({
      streamSse: async () => ({ sawTerminal: false }),
      pollOnce: async () => ({
        events: [event(1, 'inqtrix.run.completed')],
        terminal: true,
      }),
      onEvent: () => undefined,
      onTransportChange: (t) => transports.push(t),
      signal: new AbortController().signal,
      sleep: instantSleep,
    })
    expect(transports).toEqual(['polling'])
  })

  it('aborts a stream that never produces activity and falls back to polling', async () => {
    const transports: RunEventTransport[] = []
    let attemptAborted = false
    await subscribeRunEventsWithFallback({
      streamSse: async (_after, _deliver, attempt) =>
        new Promise((_resolve, reject) => {
          attempt.signal.addEventListener('abort', () => {
            attemptAborted = true
            reject(attempt.signal.reason)
          }, { once: true })
        }),
      pollOnce: async () => ({
        events: [event(1, 'inqtrix.run.completed')],
        terminal: true,
      }),
      onEvent: () => undefined,
      onTransportChange: (transport) => transports.push(transport),
      signal: new AbortController().signal,
      firstActivityTimeoutMs: 1,
      sleep: instantSleep,
    })
    expect(attemptAborted).toBe(true)
    expect(transports).toEqual(['polling'])
  })

  it('falls back after an active stream stalls and polls after the last sequence', async () => {
    const transports: RunEventTransport[] = []
    const polledAfter: Array<number | null> = []
    await subscribeRunEventsWithFallback({
      streamSse: async (_after, deliver, attempt) => {
        deliver(event(7))
        return new Promise((_resolve, reject) => {
          attempt.signal.addEventListener(
            'abort',
            () => reject(attempt.signal.reason),
            { once: true },
          )
        })
      },
      pollOnce: async (after) => {
        polledAfter.push(after)
        return {
          events: [event(8, 'inqtrix.run.completed')],
          terminal: true,
        }
      },
      onEvent: () => undefined,
      onTransportChange: (transport) => transports.push(transport),
      signal: new AbortController().signal,
      inactivityTimeoutMs: 1,
      sleep: instantSleep,
    })
    expect(polledAfter).toEqual([7])
    expect(transports).toEqual(['sse', 'polling'])
  })

  it('treats a comment heartbeat as real SSE activity', async () => {
    const transports: RunEventTransport[] = []
    await subscribeRunEventsWithFallback({
      streamSse: async (_after, _deliver, attempt) => {
        attempt.onActivity()
        return new Promise((_resolve, reject) => {
          attempt.signal.addEventListener(
            'abort',
            () => reject(attempt.signal.reason),
            { once: true },
          )
        })
      },
      pollOnce: async () => ({
        events: [event(1, 'inqtrix.run.completed')],
        terminal: true,
      }),
      onEvent: () => undefined,
      onTransportChange: (transport) => transports.push(transport),
      signal: new AbortController().signal,
      inactivityTimeoutMs: 1,
      sleep: instantSleep,
    })
    expect(transports).toEqual(['sse', 'polling'])
  })

  it('keeps reporting polling while a reconnect has no activity', async () => {
    const transports: RunEventTransport[] = []
    let attempts = 0
    let polls = 0
    await subscribeRunEventsWithFallback({
      streamSse: async (_after, _deliver, attempt) => {
        attempts += 1
        if (attempts === 1) throw new Error('down')
        return new Promise((_resolve, reject) => {
          attempt.signal.addEventListener(
            'abort',
            () => reject(attempt.signal.reason),
            { once: true },
          )
        })
      },
      pollOnce: async () => {
        polls += 1
        return {
          events: polls === 2 ? [event(2, 'inqtrix.run.completed')] : [],
          terminal: polls === 2,
        }
      },
      onEvent: () => undefined,
      onTransportChange: (transport) => transports.push(transport),
      signal: new AbortController().signal,
      firstActivityTimeoutMs: 1,
      pollsPerSseRetry: 1,
      sleep: instantSleep,
    })
    expect(attempts).toBe(2)
    expect(transports).toEqual(['polling'])
  })

  it('stops promptly on abort while polling', async () => {
    const controller = new AbortController()
    let polls = 0
    await subscribeRunEventsWithFallback({
      streamSse: async () => {
        throw new Error('down')
      },
      pollOnce: async () => {
        polls += 1
        controller.abort()
        return { events: [], terminal: false }
      },
      onEvent: () => undefined,
      signal: controller.signal,
      sleep: instantSleep,
    })
    expect(polls).toBe(1)
  })

  it('does not open a stream when the parent signal is already aborted', async () => {
    const controller = new AbortController()
    controller.abort()
    let attempts = 0
    await subscribeRunEventsWithFallback({
      streamSse: async () => {
        attempts += 1
        return { sawTerminal: false }
      },
      pollOnce: async () => {
        throw new Error('must not poll')
      },
      onEvent: () => undefined,
      signal: controller.signal,
      sleep: instantSleep,
    })
    expect(attempts).toBe(0)
  })
})
