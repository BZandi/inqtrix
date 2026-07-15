import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  acceptAgentMemoryCandidate,
  clearAgentMemories,
  createEditorCollaborationSession,
  createShares,
  decideEditorCollaborationPatches,
  deleteAgentMemory,
  enableEditorDocumentCollaboration,
  fetchMyShares,
  fetchServerFileInfo,
  fetchSharingInbox,
  flushEditorCollaborationProjection,
  listAgentMemories,
  listAgentMemoryCandidates,
  listAgentMemoryFeedback,
  listEditorCollaborationActivity,
  listEditorDocuments,
  listKnowledgeDocuments,
  patchEditorDocumentMetadata,
  publishEditorCollaborationSuggestion,
  rejectAgentMemoryCandidate,
  setExpectedUserIdentity,
  submitAgentRunFeedback,
  streamServerSentEvents,
  streamUserEvents,
  updateAgentMemory,
  updatePromptTemplate,
  updateShare,
  updateSkill,
} from './inqtrixClient'

describe('browser principal generation', () => {
  afterEach(() => {
    setExpectedUserIdentity(null)
    vi.unstubAllGlobals()
  })

  it('binds API requests to the authenticated user rendered by the SPA', async () => {
    const fetchMock = vi.fn(async (
      _input: RequestInfo | URL,
      _init?: RequestInit,
    ) => {
      void _input
      void _init
      return jsonResponse({ id: 'file-1' })
    })
    vi.stubGlobal('fetch', fetchMock)
    setExpectedUserIdentity('00000000-0000-4000-8000-000000000001')

    await fetchServerFileInfo('file-1', { baseUrl: 'http://api.test' })

    const headers = fetchMock.mock.calls[0][1]?.headers as Headers
    expect(headers.get('X-Inqtrix-Expected-User-Id')).toBe(
      '00000000-0000-4000-8000-000000000001',
    )
  })

  it('surfaces the stable principal-changed response to non-browser callers', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(
      {
        error: {
          message: 'session changed',
          type: 'principal_changed',
        },
      },
      409,
    )))
    setExpectedUserIdentity('00000000-0000-4000-8000-000000000001')

    await expect(
      fetchServerFileInfo('file-1', { baseUrl: 'http://api.test' }),
    ).rejects.toMatchObject({ name: 'principal_changed', status: 409 })
  })
})

describe('editor collaboration client contract', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('omits the lease on initial join and rotates it only in the refresh body', async () => {
    const fetchMock = vi.fn(async (
      _input: RequestInfo | URL,
      _init?: RequestInit,
    ) => {
      void _input
      void _init
      return jsonResponse({})
    })
    vi.stubGlobal('fetch', fetchMock)
    const options = { baseUrl: 'http://api.test' }

    await createEditorCollaborationSession(
      'doc-1',
      { protocol_version: 1, schema_version: 1 },
      options,
    )
    await createEditorCollaborationSession(
      'doc-1',
      {
        lease_token: 'current-private-token',
        protocol_version: 1,
        rotation_command_id: '00000000-0000-4000-8000-000000000004',
        schema_version: 1,
      },
      options,
    )

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://api.test/v1/editor/documents/doc-1/collaboration/session',
      expect.objectContaining({
        body: JSON.stringify({ protocol_version: 1, schema_version: 1 }),
        method: 'POST',
      }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://api.test/v1/editor/documents/doc-1/collaboration/session',
      expect.objectContaining({
        body: JSON.stringify({
          lease_token: 'current-private-token',
          protocol_version: 1,
          rotation_command_id: '00000000-0000-4000-8000-000000000004',
          schema_version: 1,
        }),
        method: 'POST',
      }),
    )
    for (const [url] of fetchMock.mock.calls) {
      expect(String(url)).not.toContain('current-private-token')
    }
  })

  it('keeps document lists owned-only by default and opts history hydration into all scope', async () => {
    const fetchMock = vi.fn(async (
      _input: RequestInfo | URL,
      _init?: RequestInit,
    ) => {
      void _input
      void _init
      return jsonResponse({ data: [], next_cursor: null })
    })
    vi.stubGlobal('fetch', fetchMock)
    const options = {
      baseUrl: 'http://api.test',
      workspaceId: 'workspace-1',
    }

    await listEditorDocuments(options)
    await listEditorDocuments({ ...options, limit: 200, scope: 'all' })

    expect(fetchMock.mock.calls[0][0]).toBe('http://api.test/v1/editor/documents')
    expect(fetchMock.mock.calls[1][0]).toBe(
      'http://api.test/v1/editor/documents?limit=200&scope=all',
    )
    const headers = fetchMock.mock.calls[1][1]?.headers as Headers
    expect(headers.get('X-Inqtrix-Workspace-Id')).toBe('workspace-1')
  })

  it('uses the additive metadata and collaboration endpoints without body PUTs', async () => {
    const fetchMock = vi.fn(async (
      _input: RequestInfo | URL,
      _init?: RequestInit,
    ) => {
      void _input
      void _init
      return jsonResponse({ data: [], next_cursor: null })
    })
    vi.stubGlobal('fetch', fetchMock)
    const options = { baseUrl: 'http://api.test' }

    await patchEditorDocumentMetadata(
      'doc-1',
      {
        diff_anchor_markdown: '# Projected anchor',
        diff_anchor_updated_at: 123.5,
        expected_metadata_revision: 3,
        folder_id: null,
        title: 'Shared notes',
      },
      options,
    )
    await enableEditorDocumentCollaboration(
      'doc-1',
      { expected_metadata_revision: 3, expected_revision: 7, schema_version: 1 },
      options,
    )
    await listEditorCollaborationActivity(
      'doc-1',
      { ...options, cursor: '12', limit: 25 },
    )
    await flushEditorCollaborationProjection('doc-1', options)
    await publishEditorCollaborationSuggestion(
      'doc-1',
      {
        actor_kind: 'assistant',
        command_id: '00000000-0000-4000-8000-000000000002',
        expected_sequence: 12,
        patch_id: '00000000-0000-4000-8000-000000000003',
        target_markdown: '# Shared suggestion',
      },
      options,
    )
    await decideEditorCollaborationPatches(
      'doc-1',
      {
        decision: 'accept',
        decision_id: '00000000-0000-4000-8000-000000000001',
        expected_sequence: 12,
        patch_ids: ['00000000-0000-4000-8000-000000000005'],
      },
      options,
    )

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://api.test/v1/editor/documents/doc-1',
      expect.objectContaining({
        body: JSON.stringify({
          diff_anchor_markdown: '# Projected anchor',
          diff_anchor_updated_at: 123.5,
          expected_metadata_revision: 3,
          folder_id: null,
          title: 'Shared notes',
        }),
        method: 'PATCH',
      }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://api.test/v1/editor/documents/doc-1/collaboration:enable',
      expect.objectContaining({
        body: JSON.stringify({
          expected_metadata_revision: 3,
          expected_revision: 7,
          schema_version: 1,
        }),
        method: 'POST',
      }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      'http://api.test/v1/editor/documents/doc-1/activity?cursor=12&limit=25',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      'http://api.test/v1/editor/documents/doc-1/collaboration/projection:flush',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      'http://api.test/v1/editor/documents/doc-1/suggestions:publish',
      expect.objectContaining({
        body: JSON.stringify({
          actor_kind: 'assistant',
          command_id: '00000000-0000-4000-8000-000000000002',
          expected_sequence: 12,
          patch_id: '00000000-0000-4000-8000-000000000003',
          target_markdown: '# Shared suggestion',
        }),
        method: 'POST',
      }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      6,
      'http://api.test/v1/editor/documents/doc-1/patches:decide',
      expect.objectContaining({
        body: JSON.stringify({
          decision: 'accept',
          decision_id: '00000000-0000-4000-8000-000000000001',
          expected_sequence: 12,
          patch_ids: ['00000000-0000-4000-8000-000000000005'],
        }),
        method: 'POST',
      }),
    )
    expect(fetchMock.mock.calls.map(([, init]) => init?.method)).not.toContain('PUT')
  })
})

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
    const fetchMock = vi.fn(async (
      _input: RequestInfo | URL,
      _init?: RequestInit,
    ) => {
      void _input
      void _init
      return jsonResponse({})
    })
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

describe('sharing client contract', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('uses only inbox and mine for sharing lifecycle listings', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({
      data: [],
      object: 'list',
    }))
    vi.stubGlobal('fetch', fetchMock)

    await fetchSharingInbox({ baseUrl: 'http://api.test' })
    await fetchMyShares({ baseUrl: 'http://api.test' })

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://api.test/v1/shares/inbox',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://api.test/v1/shares/mine',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('grants by canonical user id without subject aliases', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({
      data: [],
      object: 'list',
    }))
    vi.stubGlobal('fetch', fetchMock)

    await createShares(
      'run',
      'run_1',
      [{ permission: 'edit', userId: '00000000-0000-4000-8000-000000000002' }],
      { baseUrl: 'http://api.test' },
    )

    expect(fetchMock).toHaveBeenCalledWith(
      'http://api.test/v1/shares',
      expect.objectContaining({
        body: JSON.stringify({
          invitees: [{
            permission: 'edit',
            user_id: '00000000-0000-4000-8000-000000000002',
          }],
          resource_id: 'run_1',
          resource_type: 'run',
        }),
        method: 'POST',
      }),
    )
  })

  it('updates a share with its exact optimistic revision', async () => {
    const updated = {
      accepted_at: 1,
      created_at: 1,
      display_name: null,
      email: null,
      granted_by_user_id: '00000000-0000-4000-8000-000000000001',
      id: 'share-1',
      permission: 'edit',
      recipient_user_id: '00000000-0000-4000-8000-000000000002',
      resource_id: 'run_1',
      resource_type: 'run',
      revision: 4,
    }
    const fetchMock = vi.fn(async () => jsonResponse({ data: updated, object: 'share' }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(updateShare(
      'share-1',
      { expectedRevision: 3, permission: 'edit' },
      { baseUrl: 'http://api.test' },
    )).resolves.toEqual(updated)
    expect(fetchMock).toHaveBeenCalledWith(
      'http://api.test/v1/shares/share-1',
      expect.objectContaining({
        body: JSON.stringify({ expected_revision: 3, permission: 'edit' }),
        method: 'PATCH',
      }),
    )
  })
})

describe('knowledge collection client contract', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('pages documents by collection id and probes original access without downloading it', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ data: [], next_cursor: null }))
    vi.stubGlobal('fetch', fetchMock)

    await listKnowledgeDocuments('kc 1', {
      baseUrl: 'http://api.test',
      cursor: 'next row',
      limit: 50,
    })
    await fetchServerFileInfo('fl 1', { baseUrl: 'http://api.test' })

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://api.test/v1/knowledge/collections/kc 1/documents?cursor=next+row&limit=50',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://api.test/v1/files/fl 1',
      expect.objectContaining({ method: 'GET' }),
    )
  })
})

describe('shared content optimistic-concurrency contract', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sends integer revisions for prompt-template and skill updates', async () => {
    const fetchMock = vi.fn(async (
      _input: RequestInfo | URL,
      _init?: RequestInit,
    ) => {
      void _input
      void _init
      return jsonResponse({})
    })
    vi.stubGlobal('fetch', fetchMock)
    const options = { baseUrl: 'http://api.test' }

    await updatePromptTemplate('pt 1', {
      category: null,
      content_markdown: 'Summarize.',
      expected_revision: 7,
      include_in_autocomplete: true,
      label: 'summary',
      title: 'Summary',
      visibility: { chat: true, editor: false },
    }, options)
    await updateSkill('sk 1', {
      allowed_tools: [],
      argument_hint: '',
      clarification_points: [],
      deliverable: '',
      description: 'Summarize a document.',
      effort: '',
      expected_revision: 11,
      include_in_autocomplete: true,
      instructions_markdown: 'Summarize.',
      invocation: 'user_only',
      label: 'summary',
      model_tier: '',
      requires_plan: 'auto',
      title: 'Summary',
      when_to_use: 'When a summary is requested.',
    }, options)

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://api.test/v1/prompt-templates/pt 1',
      expect.objectContaining({
        body: expect.stringContaining('"expected_revision":7'),
        method: 'PUT',
      }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://api.test/v1/skills/sk 1',
      expect.objectContaining({
        body: expect.stringContaining('"expected_revision":11'),
        method: 'PUT',
      }),
    )
    for (const call of fetchMock.mock.calls) {
      expect(String(call[1]?.body)).not.toContain('expected_updated_at')
    }
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

  it('parses named user events and resumes with Last-Event-ID', async () => {
    const body = [
      'event: ready\ndata: {"user_id":"user-1","cursor":"40"}',
      'id: 41\nevent: invalidate\ndata: {"scope":"sharing","resource_type":"run"}',
      'event: reset\ndata: {}',
      '',
    ].join('\n\n')
    const fetchMock = vi.fn(async (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => {
      void input
      void init
      return new Response(body, {
        headers: { 'Content-Type': 'text/event-stream' },
        status: 200,
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    const events: unknown[] = []

    await streamUserEvents({
      baseUrl: 'http://api.test',
      lastEventId: '40',
      onEvent: (event) => events.push(event),
    })

    expect(events).toEqual([
      {
        data: { cursor: '40', user_id: 'user-1' },
        id: null,
        type: 'ready',
      },
      {
        data: { resource_type: 'run', scope: 'sharing' },
        id: '41',
        type: 'invalidate',
      },
      { data: {}, id: null, type: 'reset' },
    ])
    const request = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect((request.headers as Headers).get('Last-Event-ID')).toBe('40')
  })
})

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    headers: { 'Content-Type': 'application/json' },
    status,
  })
}
