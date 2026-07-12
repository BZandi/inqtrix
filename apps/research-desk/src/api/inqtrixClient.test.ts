import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  acceptAgentMemoryCandidate,
  clearAgentMemories,
  deleteAgentMemory,
  listAgentMemories,
  listAgentMemoryCandidates,
  listAgentMemoryFeedback,
  rejectAgentMemoryCandidate,
  submitAgentRunFeedback,
  streamServerSentEvents,
  updateAgentMemory,
} from './inqtrixClient'

describe('agent memory client', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sends memory search through the existing list endpoint', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ data: [], object: 'list', status: {} }))
    vi.stubGlobal('fetch', fetchMock)

    await listAgentMemories(
      { baseUrl: 'http://api.test' },
      { limit: 5, q: 'decision table', scope: 'user' },
    )

    expect(fetchMock).toHaveBeenCalledWith(
      'http://api.test/v1/agent/memory?q=decision+table&scope=user&limit=5',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('reads feedback history and posts run feedback without owner fields', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ data: [], object: 'list' }))
    vi.stubGlobal('fetch', fetchMock)

    await listAgentMemoryFeedback(
      { baseUrl: 'http://api.test' },
      { limit: 10, runId: 'run_1' },
    )
    await submitAgentRunFeedback(
      'run_1',
      { feedback: 'positive', memory_id: 'mem_1', reason: 'useful' },
      { baseUrl: 'http://api.test' },
    )

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://api.test/v1/agent/memory/feedback?run_id=run_1&limit=10',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://api.test/v1/agent/runs/run_1/feedback',
      expect.objectContaining({
        body: JSON.stringify({
          feedback: 'positive',
          memory_id: 'mem_1',
          reason: 'useful',
        }),
        method: 'POST',
      }),
    )
  })

  it('wires the memory mutation endpoints to their verbs and paths', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({}))
    vi.stubGlobal('fetch', fetchMock)
    const options = { baseUrl: 'http://api.test' }

    await updateAgentMemory(
      'mem 1',
      { category: 'preference', content: 'x', scope: 'user' },
      options,
    )
    await deleteAgentMemory('mem 1', options)
    await clearAgentMemories(options)
    await listAgentMemoryCandidates(options)
    await acceptAgentMemoryCandidate('cand 1', {}, options)
    await rejectAgentMemoryCandidate('cand 1', options)

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://api.test/v1/agent/memory/mem%201',
      expect.objectContaining({
        body: JSON.stringify({
          category: 'preference',
          content: 'x',
          scope: 'user',
        }),
        method: 'PATCH',
      }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://api.test/v1/agent/memory/mem%201',
      expect.objectContaining({ method: 'DELETE' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      'http://api.test/v1/agent/memory:clear',
      expect.objectContaining({ body: JSON.stringify({}), method: 'POST' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      'http://api.test/v1/agent/memory/candidates',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      'http://api.test/v1/agent/memory/candidates/cand%201:accept',
      expect.objectContaining({ body: JSON.stringify({}), method: 'POST' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      6,
      'http://api.test/v1/agent/memory/candidates/cand%201:reject',
      expect.objectContaining({ body: JSON.stringify({}), method: 'POST' }),
    )
  })
})

describe('SSE liveness', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('reports comment-only heartbeat bytes as transport activity', async () => {
    const fetchMock = vi.fn(async () => new Response(': keepalive\n\n', {
      headers: { 'Content-Type': 'text/event-stream' },
      status: 200,
    }))
    vi.stubGlobal('fetch', fetchMock)
    const onActivity = vi.fn()
    const onEvent = vi.fn()

    await streamServerSentEvents('/v1/runs/run_1/events', {
      baseUrl: 'http://api.test',
      onActivity,
      onEvent,
    })

    expect(onActivity).toHaveBeenCalled()
    expect(onEvent).not.toHaveBeenCalled()
  })
})

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: { 'Content-Type': 'application/json' },
    status: 200,
  })
}
