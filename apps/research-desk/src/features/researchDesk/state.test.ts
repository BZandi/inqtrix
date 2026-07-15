import { describe, expect, it } from 'vitest'
import type { ResearchRunEvent, ResearchRunSummary } from '@/features/researchRuns/types'
import { DEFAULT_PANEL_LAYOUT } from '@/features/project/panelLayout'
import { createEmptyProjectState } from '@/features/project/seedProject'
import type {
  ChatRuleRecord,
  ChatThreadRecord,
  EditorDocumentRecord,
  EditorFolderRecord,
  FileAssetRecord,
  KnowledgeSessionGroupRecord,
  KnowledgeSessionRecord,
  KnowledgeThreadItemRecord,
  ProjectState,
  ResearchRunRecord,
} from '@/features/project/types'
import { applyRunEvent, fromRunSummary, mergeRunSummary } from '@/features/project/types'
import { persistableAgentSessionsInOrder } from '@/features/agent/agentSessionSync'
import { researchDeskReducer } from './state'

function makeRunEvent(overrides: Partial<ResearchRunEvent> = {}): ResearchRunEvent {
  return {
    created_at: Date.parse('2026-01-01T00:00:05.000Z') / 1000,
    data: {},
    run_id: 'run-1',
    sequence: 1,
    type: 'inqtrix.run.started',
    ...overrides,
  }
}

describe('applyRunEvent startedAt', () => {
  it('derives startedAt from the running-transition event when the summary had none', () => {
    // A fresh run is created `queued` with started_at null; the running event
    // carries no started_at, so without deriving it the live timer sticks at
    // 00:00:00 (status flips to running while startedAt stays undefined).
    const queued = fromRunSummary(makeRunSummary(), 'test-stack')
    expect(queued.startedAt).toBeUndefined()

    const next = applyRunEvent(queued, makeRunEvent())
    expect(next.status).toBe('running')
    expect(next.startedAt).toBe('2026-01-01T00:00:05.000Z')
  })

  it('does not overwrite an already-set startedAt on later running events', () => {
    const running = applyRunEvent(fromRunSummary(makeRunSummary(), 'test-stack'), makeRunEvent())
    const later = applyRunEvent(
      running,
      makeRunEvent({
        created_at: Date.parse('2026-01-01T00:00:42.000Z') / 1000,
        data: { status: 'running' },
        sequence: 2,
        type: 'inqtrix.progress',
      }),
    )
    expect(later.startedAt).toBe('2026-01-01T00:00:05.000Z')
  })

  it('preserves an authoritative summary startedAt from a hydrated running run', () => {
    // listResearchRuns already carries started_at for in-flight runs — an event
    // must never clobber that with a (later) event timestamp.
    const hydrated = fromRunSummary(
      makeRunSummary({ started_at: Date.parse('2026-01-01T00:00:01.000Z') / 1000, status: 'running' }),
      'test-stack',
    )
    expect(hydrated.startedAt).toBe('2026-01-01T00:00:01.000Z')
    const next = applyRunEvent(hydrated, makeRunEvent({ created_at: Date.parse('2026-01-01T00:00:09.000Z') / 1000 }))
    expect(next.startedAt).toBe('2026-01-01T00:00:01.000Z')
  })
})

function makeAsset(id: string, label: string, overrides: Partial<FileAssetRecord> = {}): FileAssetRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    extractedText: `${label} content`,
    fileName: `${label}.txt`,
    groupId: null,
    id,
    label,
    mimeType: 'text/plain',
    origin: 'library',
    pageCount: null,
    parseStatus: 'parsed',
    parseWarning: null,
    sectionId: 'file-section-temp',
    sizeBytes: 12,
    textTruncated: false,
    title: `${label}.txt`,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function makeRule(id: string, label: string, overrides: Partial<ChatRuleRecord> = {}): ChatRuleRecord {
  return {
    contentMarkdown: `${label} prompt`,
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    label,
    title: `${label} prompt`,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function makeKnowledgeSession(
  id: string,
  title: string,
  overrides: Partial<KnowledgeSessionRecord> = {},
): KnowledgeSessionRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    title,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function makeKnowledgeGroup(
  id: string,
  title: string,
  overrides: Partial<KnowledgeSessionGroupRecord> = {},
): KnowledgeSessionGroupRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    title,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function makeKnowledgeItem(
  id: string,
  sessionId: string,
  overrides: Partial<KnowledgeThreadItemRecord> = {},
): KnowledgeThreadItemRecord {
  return {
    collectionTitles: ['EU Recht'],
    createdAt: '2026-01-02T00:00:00.000Z',
    id,
    progress: { steps: [] },
    question: 'Welche Pflichten gelten?',
    requestedProfile: null,
    runId: `run-${id}`,
    sessionId,
    status: 'running',
    ...overrides,
  }
}

function makeRunSummary(overrides: Partial<ResearchRunSummary> = {}): ResearchRunSummary {
  return {
    access: { mode: 'owner' },
    agent_overrides: {},
    created_at: Date.parse('2026-01-01T00:00:00.000Z') / 1000,
    elapsed_seconds: null,
    error: null,
    events_url: '/v1/runs/run-1/events',
    finished_at: null,
    mode: 'research',
    question: 'Research question',
    queue_position: null,
    result_url: '/v1/runs/run-1/result',
    run_id: 'run-1',
    snapshot: {},
    stack: 'test-stack',
    started_at: null,
    status: 'queued',
    ...overrides,
  }
}

describe('authoritative run replacement', () => {
  it('adopts the server-generated id for an imported run and its selection', () => {
    const imported = {
      ...fromRunSummary(makeRunSummary({ run_id: 'external-run' }), 'test-stack'),
      source: 'imported' as const,
    }
    const state = {
      ...createEmptyProjectState(),
      researchRunOrder: [imported.runId],
      researchRuns: { [imported.runId]: imported },
      ui: {
        ...createEmptyProjectState().ui,
        expandedJobId: imported.runId,
        pendingChatReportRunId: imported.runId,
        selectedJobId: imported.runId,
      },
    }

    const adopted = researchDeskReducer(state, {
      sourceRunId: imported.runId,
      summary: makeRunSummary({ run_id: 'run_server_allocated' }),
      type: 'adoptImportedApiRun',
    })

    expect(adopted.researchRuns['external-run']).toBeUndefined()
    expect(adopted.researchRuns.run_server_allocated).toMatchObject({
      runId: 'run_server_allocated',
      source: 'api',
    })
    expect(adopted.researchRunOrder).toEqual(['run_server_allocated'])
    expect(adopted.ui).toMatchObject({
      expandedJobId: 'run_server_allocated',
      pendingChatReportRunId: 'run_server_allocated',
      selectedJobId: 'run_server_allocated',
    })
  })

  it('prunes missing API runs while preserving imported rows and server order', () => {
    let state = researchDeskReducer(createEmptyProjectState(), {
      summary: makeRunSummary({ run_id: 'run-revoked' }),
      type: 'upsertApiRunSummary',
    })
    const imported: ResearchRunRecord = {
      ...fromRunSummary(makeRunSummary({ run_id: 'run-imported' }), 'test-stack'),
      source: 'imported',
    }
    state = {
      ...state,
      researchRunOrder: ['run-revoked', imported.runId],
      researchRuns: { ...state.researchRuns, [imported.runId]: imported },
    }

    const replaced = researchDeskReducer(state, {
      summaries: [
        makeRunSummary({ run_id: 'run-newest' }),
        makeRunSummary({ run_id: 'run-older' }),
      ],
      type: 'replaceApiRunSummaries',
    })

    expect(replaced.researchRuns['run-revoked']).toBeUndefined()
    expect(replaced.researchRuns['run-imported']).toBe(imported)
    expect(replaced.researchRunOrder).toEqual([
      'run-newest',
      'run-older',
      'run-imported',
    ])
    expect(replaced.ui.selectedJobId).toBe('run-newest')
  })

  it('keeps live event state for present runs and removes vanished agent shells', () => {
    let state = researchDeskReducer(createEmptyProjectState(), {
      summary: makeRunSummary({ run_id: 'run-live' }),
      type: 'upsertApiRunSummary',
    })
    state = {
      ...state,
      researchRuns: {
        ...state.researchRuns,
        'run-live': {
          ...state.researchRuns['run-live'],
          events: [{
            createdAt: '2026-01-01T00:00:05.000Z',
            id: 'event-live',
            kind: 'system',
            severity: 'info',
            title: 'Live detail',
          }],
        },
      },
    }
    state = researchDeskReducer(state, {
      summary: makeRunSummary({
        kind: 'agent',
        mode: 'workspace_agent',
        run_id: 'run-agent-revoked',
      }),
      type: 'upsertApiRunSummary',
    })

    const replaced = researchDeskReducer(state, {
      summaries: [makeRunSummary({ run_id: 'run-live', status: 'running' })],
      type: 'replaceApiRunSummaries',
    })

    expect(replaced.researchRuns['run-live'].events).toHaveLength(1)
    expect(replaced.agentRuns['run-agent-revoked']).toBeUndefined()
    expect(replaced.agentSessions['run-agent-revoked']).toBeUndefined()
  })

  it('retains loaded agent children for a visible root and prunes the full removed tree', () => {
    const root = makeRunSummary({
      kind: 'agent',
      mode: 'workspace_agent',
      run_id: 'run-agent-root',
      session_id: 'session-agent',
    })
    const child = makeRunSummary({
      kind: 'agent_child',
      mode: 'workspace_agent',
      parent_run_id: root.run_id,
      root_run_id: root.run_id,
      run_id: 'run-agent-child',
      session_id: 'session-agent',
    })
    let state = researchDeskReducer(createEmptyProjectState(), {
      summary: root,
      type: 'upsertAgentRunSummary',
    })
    state = researchDeskReducer(state, {
      summary: child,
      type: 'upsertAgentRunSummary',
    })

    const retained = researchDeskReducer(state, {
      summaries: [root],
      type: 'replaceApiRunSummaries',
    })
    expect(retained.agentRuns[root.run_id]).toBeDefined()
    expect(retained.agentRuns[child.run_id]).toMatchObject({
      parentRunId: root.run_id,
      rootRunId: root.run_id,
    })

    const pruned = researchDeskReducer(retained, {
      summaries: [],
      type: 'replaceApiRunSummaries',
    })
    expect(pruned.agentRuns[root.run_id]).toBeUndefined()
    expect(pruned.agentRuns[child.run_id]).toBeUndefined()
  })

  it('keeps shared agent runs in a non-persistable view session and prunes it on revoke', () => {
    const shared = makeRunSummary({
      access: { mode: 'shared', permission: 'view' },
      kind: 'agent',
      mode: 'workspace_agent',
      run_id: 'run-agent-shared',
      session_id: 'owner-session-secret',
    })
    let state = researchDeskReducer(createEmptyProjectState(), {
      summary: shared,
      type: 'upsertApiRunSummary',
    })
    const derivedId = state.agentRuns[shared.run_id].sessionId as string

    expect(derivedId).toBe('shared-agent-view:run-agent-shared')
    expect(derivedId).not.toContain(shared.session_id as string)
    expect(state.agentSessions[derivedId]).toMatchObject({
      persistable: false,
      runIds: [shared.run_id],
    })
    expect(persistableAgentSessionsInOrder(
      state.agentSessions,
      state.agentSessionOrder,
    )).toEqual([])

    state = researchDeskReducer(state, {
      summaries: [shared],
      type: 'replaceApiRunSummaries',
    })
    expect(state.agentSessionOrder.filter((id) => id === derivedId)).toEqual([
      derivedId,
    ])
    expect(persistableAgentSessionsInOrder(
      state.agentSessions,
      state.agentSessionOrder,
    )).toEqual([])

    state = researchDeskReducer(state, {
      summaries: [],
      type: 'replaceApiRunSummaries',
    })
    expect(state.agentRuns[shared.run_id]).toBeUndefined()
    expect(state.agentSessions[derivedId]).toBeUndefined()
    expect(state.agentSessionOrder).not.toContain(derivedId)
  })

  it('keeps owned agent runs on their existing persistable session path', () => {
    const owned = makeRunSummary({
      kind: 'agent',
      mode: 'workspace_agent',
      run_id: 'run-agent-owned',
      session_id: 'session-owned',
    })
    const state = researchDeskReducer(createEmptyProjectState(), {
      summary: owned,
      type: 'upsertApiRunSummary',
    })

    expect(state.agentRuns[owned.run_id].sessionId).toBe('session-owned')
    expect(persistableAgentSessionsInOrder(
      state.agentSessions,
      state.agentSessionOrder,
    ).map((session) => session.id)).toEqual(['session-owned'])
  })
})

function makeChatThread(id: string, title: string, overrides: Partial<ChatThreadRecord> = {}): ChatThreadRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    messages: [],
    preview: 'Preview',
    source: 'mock',
    title,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function makeEditorFolder(id: string, title: string, overrides: Partial<EditorFolderRecord> = {}): EditorFolderRecord {
  return {
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    title,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

function makeEditorDocument(
  id: string,
  title: string,
  overrides: Partial<EditorDocumentRecord> = {},
): EditorDocumentRecord {
  return {
    contentMarkdown: `# ${title}`,
    createdAt: '2026-01-01T00:00:00.000Z',
    folderId: null,
    id,
    revision: 1,
    source: 'blank',
    title,
    updatedAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

describe('setResearchRunAutocomplete', () => {
  it('toggles a run\'s @-mention availability and marks the project dirty', () => {
    const base = createEmptyProjectState()
    base.researchRuns = {
      'run-1': { events: [], runId: 'run-1', status: 'completed', summary: { title: 'r' } } as unknown as ResearchRunRecord,
    }
    base.researchRunOrder = ['run-1']

    const hidden = researchDeskReducer(base, {
      includeInAutocomplete: false,
      runId: 'run-1',
      type: 'setResearchRunAutocomplete',
    })
    expect(hidden.researchRuns['run-1'].includeInAutocomplete).toBe(false)
    expect(hidden.dirty).toBe(true)

    // An unknown run id is a no-op and returns the same state reference.
    const noop = researchDeskReducer(hidden, {
      includeInAutocomplete: true,
      runId: 'missing',
      type: 'setResearchRunAutocomplete',
    })
    expect(noop).toBe(hidden)
  })
})

describe('mergeRunSummary', () => {
  it('preserves the local @-mention availability across an API summary refresh', () => {
    const prior = {
      events: [],
      includeInAutocomplete: false,
      runId: 'run-1',
      status: 'completed',
      summary: { title: 'r' },
    } as unknown as ResearchRunRecord

    const merged = mergeRunSummary(prior, makeRunSummary({ run_id: 'run-1', status: 'completed' }), 'fallback-stack')

    // A run-list refresh must not silently re-enable a report the user hid.
    expect(merged.includeInAutocomplete).toBe(false)
  })
})

describe('ui visibility reducer actions', () => {
  it('hides and shows the chat history panel', () => {
    const hidden = researchDeskReducer(createEmptyProjectState(), {
      isVisible: false,
      type: 'setChatHistoryVisible',
    })
    expect(hidden.ui.isChatHistoryVisible).toBe(false)
    const shown = researchDeskReducer(hidden, {
      isVisible: true,
      type: 'setChatHistoryVisible',
    })
    expect(shown.ui.isChatHistoryVisible).toBe(true)
  })

  it('marks the project dirty for every collapsible-panel toggle (durable persistence)', () => {
    // Each collapse flag is a persisted ui/editorUi field; a pure toggle must
    // dirty the project so it is saved/pushed and survives reload. A revert that
    // drops dirty:true from any of these would otherwise pass silently.
    const collapseActions = [
      'setChatHistoryVisible',
      'setKnowledgeHistoryVisible',
      'setReportVisible',
      'setEditorTreeVisible',
      'setEditorCommentPanelVisible',
    ] as const
    for (const type of collapseActions) {
      const next = researchDeskReducer(createEmptyProjectState(), { isVisible: false, type })
      expect(next.dirty, `${type} must dirty the project`).toBe(true)
    }
  })

  it('hides and shows the knowledge history panel', () => {
    const hidden = researchDeskReducer(createEmptyProjectState(), {
      isVisible: false,
      type: 'setKnowledgeHistoryVisible',
    })
    expect(hidden.ui.isKnowledgeHistoryVisible).toBe(false)
    const shown = researchDeskReducer(hidden, {
      isVisible: true,
      type: 'setKnowledgeHistoryVisible',
    })
    expect(shown.ui.isKnowledgeHistoryVisible).toBe(true)
  })

  it('persists clamped resizable panel sizes', () => {
    const base = createEmptyProjectState()
    expect(base.ui.panelLayout).toEqual(DEFAULT_PANEL_LAYOUT)

    const resized = researchDeskReducer(base, {
      key: 'chatHistory',
      size: 37,
      type: 'setPanelLayoutSize',
    })
    expect(resized.dirty).toBe(true)
    expect(resized.ui.panelLayout.chatHistory).toBe(37)

    const tooNarrow = researchDeskReducer(resized, {
      key: 'chatHistory',
      size: 0,
      type: 'setPanelLayoutSize',
    })
    expect(tooNarrow.ui.panelLayout.chatHistory).toBe(18)

    const tooWide = researchDeskReducer(base, {
      key: 'researchReport',
      size: 90,
      type: 'setPanelLayoutSize',
    })
    expect(tooWide.ui.panelLayout.researchReport).toBe(58)

    const editorResized = researchDeskReducer(base, {
      key: 'editorComments',
      size: 30,
      type: 'setPanelLayoutSize',
    })
    expect(editorResized.dirty).toBe(true)
    expect(editorResized.ui.panelLayout.editorComments).toBe(30)

    const editorTooNarrow = researchDeskReducer(base, {
      key: 'editorComments',
      size: 5,
      type: 'setPanelLayoutSize',
    })
    expect(editorTooNarrow.ui.panelLayout.editorComments).toBe(20)
  })
})

describe('api run summary reducer actions', () => {
  it('keeps knowledge runs out of the global research run store', () => {
    const base = createEmptyProjectState()

    const reduced = researchDeskReducer(base, {
      summary: makeRunSummary({ mode: 'knowledge', run_id: 'run-knowledge' }),
      type: 'upsertApiRunSummary',
    })

    expect(reduced).toBe(base)
    expect(reduced.researchRuns['run-knowledge']).toBeUndefined()
    expect(reduced.researchRunOrder).not.toContain('run-knowledge')
  })
})

describe('pinned explorer reducer actions', () => {
  it('toggles pinned chat threads, knowledge sessions and editor documents', () => {
    const base = createEmptyProjectState()
    const sessionId = base.selectedKnowledgeSessionId as string
    const seeded = {
      ...base,
      chatThreadOrder: ['ct-1'],
      chatThreads: { 'ct-1': makeChatThread('ct-1', 'Chat') },
      editorDocumentOrder: ['doc-1'],
      editorDocuments: { 'doc-1': makeEditorDocument('doc-1', 'Doc') },
    }

    const withChatPin = researchDeskReducer(seeded, {
      threadId: 'ct-1',
      type: 'togglePinnedChatThread',
    })
    const withKnowledgePin = researchDeskReducer(withChatPin, {
      sessionId,
      type: 'togglePinnedKnowledgeSession',
    })
    const withDocumentPin = researchDeskReducer(withKnowledgePin, {
      documentId: 'doc-1',
      type: 'togglePinnedEditorDocument',
    })

    expect(withDocumentPin.ui.pinnedExplorer.chatThreadIds).toEqual(['ct-1'])
    expect(withDocumentPin.ui.pinnedExplorer.knowledgeSessionIds).toEqual([sessionId])
    expect(withDocumentPin.ui.pinnedExplorer.editorDocumentIds).toEqual(['doc-1'])
    expect(withDocumentPin.dirty).toBe(true)

    const unpinned = researchDeskReducer(withDocumentPin, {
      threadId: 'ct-1',
      type: 'togglePinnedChatThread',
    })
    expect(unpinned.ui.pinnedExplorer.chatThreadIds).toEqual([])
  })

  it('removes stale pins when pinned entries are deleted', () => {
    const base = createEmptyProjectState()
    const sessionId = base.selectedKnowledgeSessionId as string
    const seeded = {
      ...base,
      chatThreadOrder: ['ct-1'],
      chatThreads: { 'ct-1': makeChatThread('ct-1', 'Chat') },
      editorDocumentOrder: ['doc-1'],
      editorDocuments: { 'doc-1': makeEditorDocument('doc-1', 'Doc') },
      editorUi: { ...base.editorUi, activeDocumentId: 'doc-1', openDocumentIds: ['doc-1'] },
      ui: {
        ...base.ui,
        pinnedExplorer: {
          chatThreadIds: ['ct-1'],
          editorDocumentIds: ['doc-1'],
          knowledgeSessionIds: [sessionId],
          agentSessionIds: [],
        },
        selectedChatThreadId: 'ct-1',
      },
    }

    const withoutChat = researchDeskReducer(seeded, {
      threadId: 'ct-1',
      type: 'deleteChatThread',
    })
    const withoutKnowledge = researchDeskReducer(withoutChat, {
      sessionId,
      type: 'deleteKnowledgeSession',
    })
    const withoutDocument = researchDeskReducer(withoutKnowledge, {
      documentId: 'doc-1',
      type: 'deleteEditorDocument',
    })

    expect(withoutDocument.ui.pinnedExplorer).toEqual({
      chatThreadIds: [],
      editorDocumentIds: [],
      knowledgeSessionIds: [],
      agentSessionIds: [],
    })
  })
})

describe('knowledge session reducer actions', () => {
  it('creates, selects, and renames knowledge sessions', () => {
    const base = createEmptyProjectState()
    const created = researchDeskReducer(base, {
      session: makeKnowledgeSession('ks-2', 'Client review'),
      type: 'createKnowledgeSession',
    })
    expect(created.selectedKnowledgeSessionId).toBe('ks-2')
    expect(created.knowledgeSessionOrder[0]).toBe('ks-2')

    const renamed = researchDeskReducer(created, {
      sessionId: 'ks-2',
      title: 'Board memo',
      type: 'renameKnowledgeSession',
    })
    expect(renamed.knowledgeSessions['ks-2'].title).toBe('Board memo')

    const selected = researchDeskReducer(renamed, {
      sessionId: base.selectedKnowledgeSessionId as string,
      type: 'selectKnowledgeSession',
    })
    expect(selected.selectedKnowledgeSessionId).toBe(base.selectedKnowledgeSessionId)
  })

  it('keeps questions scoped to their knowledge session and removes them on delete', () => {
    const base = createEmptyProjectState()
    const defaultSessionId = base.selectedKnowledgeSessionId as string
    const withSecondSession = researchDeskReducer(base, {
      session: makeKnowledgeSession('ks-2', 'New session'),
      type: 'createKnowledgeSession',
    })
    const withDefaultItem = researchDeskReducer(withSecondSession, {
      item: makeKnowledgeItem('ki-1', defaultSessionId),
      type: 'startKnowledgeAsk',
    })
    const withSecondItem = researchDeskReducer(withDefaultItem, {
      item: makeKnowledgeItem('ki-2', 'ks-2', { question: 'Wie ist Artikel 6 belegt?' }),
      type: 'startKnowledgeAsk',
    })

    expect(withSecondItem.knowledgeItems['ki-1'].sessionId).toBe(defaultSessionId)
    expect(withSecondItem.knowledgeItems['ki-2'].sessionId).toBe('ks-2')
    expect(withSecondItem.knowledgeSessions['ks-2'].title).toBe('Wie ist Artikel 6 belegt?')

    const deleted = researchDeskReducer(withSecondItem, {
      sessionId: 'ks-2',
      type: 'deleteKnowledgeSession',
    })
    expect(deleted.knowledgeSessions['ks-2']).toBeUndefined()
    expect(deleted.knowledgeItems['ki-2']).toBeUndefined()
    expect(deleted.knowledgeItems['ki-1']).toBeDefined()
    expect(deleted.selectedKnowledgeSessionId).toBe(defaultSessionId)
  })

  it('deletes selected knowledge Q&A entries and touches their sessions', () => {
    const base = createEmptyProjectState()
    const defaultSessionId = base.selectedKnowledgeSessionId as string
    const seeded = {
      ...base,
      knowledgeItemOrder: ['ki-1', 'ki-2'],
      knowledgeItems: {
        'ki-1': makeKnowledgeItem('ki-1', defaultSessionId),
        'ki-2': makeKnowledgeItem('ki-2', defaultSessionId, { question: 'Second question' }),
      },
      knowledgeSessions: {
        ...base.knowledgeSessions,
        [defaultSessionId]: { ...base.knowledgeSessions[defaultSessionId], updatedAt: '2026-01-01T00:00:00.000Z' },
      },
    }

    const deleted = researchDeskReducer(seeded, {
      itemIds: ['ki-1'],
      type: 'deleteKnowledgeItems',
    })

    expect(deleted.knowledgeItemOrder).toEqual(['ki-2'])
    expect(deleted.knowledgeItems['ki-1']).toBeUndefined()
    expect(deleted.knowledgeItems['ki-2']).toBeDefined()
    expect(deleted.knowledgeSessions[defaultSessionId].updatedAt).not.toBe('2026-01-01T00:00:00.000Z')
    expect(deleted.dirty).toBe(true)
  })

  it('clears one knowledge session without removing sibling session entries', () => {
    const base = createEmptyProjectState()
    const defaultSessionId = base.selectedKnowledgeSessionId as string
    const seeded = {
      ...base,
      knowledgeItemOrder: ['ki-1', 'ki-2'],
      knowledgeItems: {
        'ki-1': makeKnowledgeItem('ki-1', defaultSessionId),
        'ki-2': makeKnowledgeItem('ki-2', 'ks-2'),
      },
      knowledgeSessionOrder: [defaultSessionId, 'ks-2'],
      knowledgeSessions: {
        ...base.knowledgeSessions,
        [defaultSessionId]: { ...base.knowledgeSessions[defaultSessionId], updatedAt: '2026-01-01T00:00:00.000Z' },
        'ks-2': makeKnowledgeSession('ks-2', 'Second'),
      },
    }

    const cleared = researchDeskReducer(seeded, {
      sessionId: defaultSessionId,
      type: 'clearKnowledgeSession',
    })

    expect(cleared.knowledgeItemOrder).toEqual(['ki-2'])
    expect(cleared.knowledgeItems['ki-1']).toBeUndefined()
    expect(cleared.knowledgeItems['ki-2']).toBeDefined()
    expect(cleared.knowledgeSessions[defaultSessionId]).toBeDefined()
    expect(cleared.knowledgeSessions[defaultSessionId].updatedAt).not.toBe('2026-01-01T00:00:00.000Z')
    expect(cleared.selectedKnowledgeSessionId).toBe(defaultSessionId)
    expect(cleared.dirty).toBe(true)
  })

  it('restarts a knowledge ask in place and clears the previous answer payload', () => {
    const base = createEmptyProjectState()
    const defaultSessionId = base.selectedKnowledgeSessionId as string
    const seeded = {
      ...base,
      knowledgeItemOrder: ['ki-1'],
      knowledgeItems: {
        'ki-1': makeKnowledgeItem('ki-1', defaultSessionId, {
          answer: {
            answerMarkdown: 'Old answer',
            degradedStages: [],
            references: [],
            refusal: false,
            quotes: [],
          },
          collectionIds: ['collection-old'],
          completedAt: '2026-01-03T00:00:00.000Z',
          error: 'old error',
          status: 'completed',
          topK: 5,
        }),
      },
    }

    const restarted = researchDeskReducer(seeded, {
      item: makeKnowledgeItem('ki-new', defaultSessionId, {
        collectionIds: ['collection-new'],
        collectionTitles: ['EU Recht', 'AI Act'],
        createdAt: '2026-01-04T00:00:00.000Z',
        question: 'Updated question',
        requestedProfile: 'tief',
        runId: 'run-new',
        topK: null,
      }),
      replacedItemId: 'ki-1',
      type: 'restartKnowledgeAsk',
    })

    expect(restarted.knowledgeItemOrder).toEqual(['ki-1'])
    expect(restarted.knowledgeItems['ki-1']).toMatchObject({
      collectionIds: ['collection-new'],
      collectionTitles: ['EU Recht', 'AI Act'],
      createdAt: '2026-01-04T00:00:00.000Z',
      id: 'ki-1',
      progress: { steps: [] },
      question: 'Updated question',
      requestedProfile: 'tief',
      runId: 'run-new',
      sessionId: defaultSessionId,
      status: 'running',
      topK: null,
    })
    expect(restarted.knowledgeItems['ki-1'].answer).toBeUndefined()
    expect(restarted.knowledgeItems['ki-1'].completedAt).toBeUndefined()
    expect(restarted.knowledgeItems['ki-1'].error).toBeUndefined()
    expect(restarted.knowledgeSessions[defaultSessionId].updatedAt).toBe('2026-01-04T00:00:00.000Z')
    expect(restarted.dirty).toBe(true)
  })

  it('sets completedAt when a knowledge item completes', () => {
    const base = createEmptyProjectState()
    const defaultSessionId = base.selectedKnowledgeSessionId as string
    const seeded = {
      ...base,
      knowledgeItemOrder: ['ki-1'],
      knowledgeItems: {
        'ki-1': makeKnowledgeItem('ki-1', defaultSessionId, {
          progress: {
            steps: [{ facts: {}, id: 'answer', kind: 'answer', status: 'running' }],
          },
        }),
      },
    }

    const completed = researchDeskReducer(seeded, {
      answer: {
        answerMarkdown: 'Fresh answer',
        degradedStages: [],
        references: [],
        refusal: false,
        quotes: [],
      },
      runId: 'run-ki-1',
      type: 'completeKnowledgeItem',
    })

    expect(completed.knowledgeItems['ki-1'].status).toBe('completed')
    expect(completed.knowledgeItems['ki-1'].completedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/)
    expect(completed.knowledgeItems['ki-1'].progress.steps[0].status).toBe('done')
    expect(completed.knowledgeSessions[defaultSessionId].updatedAt).toBe(completed.knowledgeItems['ki-1'].completedAt)
    expect(completed.dirty).toBe(true)
  })

  it('marks a running knowledge item as cancelled when its run is cancelled', () => {
    const base = createEmptyProjectState()
    const defaultSessionId = base.selectedKnowledgeSessionId as string
    const seeded = {
      ...base,
      knowledgeItemOrder: ['ki-1'],
      knowledgeItems: {
        'ki-1': makeKnowledgeItem('ki-1', defaultSessionId, {
          progress: {
            steps: [{ facts: {}, id: 'answer', kind: 'answer', status: 'running' }],
          },
        }),
      },
      knowledgeSessions: {
        ...base.knowledgeSessions,
        [defaultSessionId]: {
          ...base.knowledgeSessions[defaultSessionId],
          updatedAt: '2026-01-01T00:00:00.000Z',
        },
      },
    }
    const event: ResearchRunEvent = {
      created_at: Date.parse('2026-01-02T00:05:00.000Z') / 1000,
      data: { message: 'Request cancelled', status: 'cancelled' },
      run_id: 'run-ki-1',
      sequence: 5,
      type: 'inqtrix.run.cancelled',
    }

    const cancelled = researchDeskReducer(seeded, {
      event,
      type: 'appendApiRunEvent',
    })

    expect(cancelled.knowledgeItems['ki-1'].status).toBe('cancelled')
    expect(cancelled.knowledgeItems['ki-1'].progress.steps[0].status).toBe('done')
    expect(cancelled.knowledgeItems['ki-1'].error).toBeUndefined()
    expect(cancelled.knowledgeSessions[defaultSessionId].updatedAt).not.toBe('2026-01-01T00:00:00.000Z')
  })

  it('deletes the last remaining session, leaving an empty state', () => {
    const base = createEmptyProjectState()
    const onlyId = base.selectedKnowledgeSessionId as string
    expect(base.knowledgeSessionOrder).toEqual([onlyId])

    const deleted = researchDeskReducer(base, {
      sessionId: onlyId,
      type: 'deleteKnowledgeSession',
    })

    // The old length<=1 guard blocked this; now the last session is deletable
    // and the next ask creates a fresh one.
    expect(deleted.knowledgeSessions[onlyId]).toBeUndefined()
    expect(deleted.knowledgeSessionOrder).toEqual([])
    expect(deleted.selectedKnowledgeSessionId).toBeNull()
  })

  it('prunes the pristine bootstrap default on an empty-server hydrate', () => {
    const base = createEmptyProjectState()
    const defaultId = base.selectedKnowledgeSessionId as string

    const pruned = researchDeskReducer(base, {
      serverIds: [],
      type: 'pruneLocalPlaceholderKnowledgeSessions',
    })

    // No phantom "Neue Sitzung" lingers (and so none gets re-synced as a ghost).
    expect(pruned.knowledgeSessions[defaultId]).toBeUndefined()
    expect(pruned.knowledgeSessionOrder).toEqual([])
    expect(pruned.selectedKnowledgeSessionId).toBeNull()
  })

  it('keeps a renamed empty bootstrap session on an empty-server hydrate', () => {
    const base = createEmptyProjectState()
    const defaultId = base.selectedKnowledgeSessionId as string
    const renamed = researchDeskReducer(base, {
      sessionId: defaultId,
      title: 'Test',
      type: 'renameKnowledgeSession',
    })

    const pruned = researchDeskReducer(renamed, {
      serverIds: [],
      type: 'pruneLocalPlaceholderKnowledgeSessions',
    })

    expect(pruned.knowledgeSessions[defaultId]?.title).toBe('Test')
    expect(pruned.knowledgeSessionOrder).toEqual([defaultId])
  })

  it('keeps a user-created empty session on an empty-server hydrate', () => {
    const base = createEmptyProjectState()
    const defaultId = base.selectedKnowledgeSessionId as string
    const created = researchDeskReducer(base, {
      session: makeKnowledgeSession('ks-2', 'Test'),
      type: 'createKnowledgeSession',
    })

    const pruned = researchDeskReducer(created, {
      serverIds: [],
      type: 'pruneLocalPlaceholderKnowledgeSessions',
    })

    expect(pruned.knowledgeSessions[defaultId]).toBeUndefined()
    expect(pruned.knowledgeSessions['ks-2']?.title).toBe('Test')
    expect(pruned.knowledgeSessionOrder).toEqual(['ks-2'])
    expect(pruned.selectedKnowledgeSessionId).toBe('ks-2')
  })

  it('keeps a placeholder session that has local items pending sync', () => {
    const base = createEmptyProjectState()
    const defaultId = base.selectedKnowledgeSessionId as string
    const withItem = researchDeskReducer(base, {
      item: makeKnowledgeItem('ki-1', defaultId),
      type: 'startKnowledgeAsk',
    })

    const pruned = researchDeskReducer(withItem, {
      serverIds: [],
      type: 'pruneLocalPlaceholderKnowledgeSessions',
    })

    expect(pruned.knowledgeSessions[defaultId]).toBeDefined()
  })

  it('hydrates server knowledge groups and session memberships', () => {
    const base = createEmptyProjectState()
    const withGroup = researchDeskReducer(base, {
      groups: [makeKnowledgeGroup('kg-1', 'Folder')],
      type: 'upsertServerKnowledgeSessionGroups',
    })
    const withSession = researchDeskReducer(withGroup, {
      memberships: { 'ks-server': 'kg-1' },
      sessions: [
        makeKnowledgeSession('ks-server', 'Server session', {
          updatedAt: '2026-01-03T00:00:00.000Z',
        }),
      ],
      type: 'upsertServerKnowledgeSessions',
    })

    expect(withSession.dirty).toBe(base.dirty)
    expect(withSession.knowledgeSessionGroupOrder[0]).toBe('kg-1')
    expect(withSession.knowledgeSessionGroups['kg-1']?.title).toBe('Folder')
    expect(withSession.knowledgeSessions['ks-server']?.title).toBe('Server session')
    expect(withSession.knowledgeSessionGroupMemberships['ks-server']).toBe('kg-1')
  })

  it('keeps a server-known session even when its items are not loaded yet', () => {
    const base = createEmptyProjectState()
    const defaultId = base.selectedKnowledgeSessionId as string

    const pruned = researchDeskReducer(base, {
      serverIds: [defaultId],
      type: 'pruneLocalPlaceholderKnowledgeSessions',
    })

    expect(pruned.knowledgeSessions[defaultId]).toBeDefined()
    expect(pruned.knowledgeSessionOrder).toEqual([defaultId])
  })

  it('does not overwrite a custom empty-session title on first ask', () => {
    const withSession = researchDeskReducer(createEmptyProjectState(), {
      session: makeKnowledgeSession('ks-2', 'Client A'),
      type: 'createKnowledgeSession',
    })
    const next = researchDeskReducer(withSession, {
      item: makeKnowledgeItem('ki-1', 'ks-2', { question: 'Welche Risiken gibt es?' }),
      type: 'startKnowledgeAsk',
    })
    expect(next.knowledgeSessions['ks-2'].title).toBe('Client A')
  })

  it('moves knowledge sessions into folders and clears memberships when a folder is deleted', () => {
    const base = createEmptyProjectState()
    const defaultSessionId = base.selectedKnowledgeSessionId as string
    const withFolder = researchDeskReducer(base, {
      title: 'Client files',
      type: 'createKnowledgeSessionGroup',
    })
    const groupId = withFolder.knowledgeSessionGroupOrder[0]
    const withSecondSession = researchDeskReducer(withFolder, {
      session: makeKnowledgeSession('ks-2', 'Evidence review'),
      type: 'createKnowledgeSession',
    })
    const movedSecond = researchDeskReducer(withSecondSession, {
      groupId,
      sessionId: 'ks-2',
      targetIndex: 0,
      type: 'moveKnowledgeSessionToGroup',
    })
    const movedDefault = researchDeskReducer(movedSecond, {
      groupId,
      sessionId: defaultSessionId,
      targetIndex: 0,
      type: 'moveKnowledgeSessionToGroup',
    })

    expect(movedDefault.knowledgeSessionGroupMemberships['ks-2']).toBe(groupId)
    expect(movedDefault.knowledgeSessionGroupMemberships[defaultSessionId]).toBe(groupId)
    expect(movedDefault.knowledgeSessionOrder.slice(0, 2)).toEqual([defaultSessionId, 'ks-2'])

    const deletedFolder = researchDeskReducer(movedDefault, {
      groupId,
      type: 'deleteKnowledgeSessionGroup',
    })

    expect(deletedFolder.knowledgeSessionGroups[groupId]).toBeUndefined()
    expect(deletedFolder.knowledgeSessionGroupMemberships['ks-2']).toBeUndefined()
    expect(deletedFolder.knowledgeSessions['ks-2']).toBeDefined()
    expect(deletedFolder.knowledgeSessions[defaultSessionId]).toBeDefined()
  })

  it('reorders knowledge session folders', () => {
    const withFirstFolder = researchDeskReducer(createEmptyProjectState(), {
      title: 'First',
      type: 'createKnowledgeSessionGroup',
    })
    const firstGroupId = withFirstFolder.knowledgeSessionGroupOrder[0]
    const withSecondFolder = researchDeskReducer(withFirstFolder, {
      title: 'Second',
      type: 'createKnowledgeSessionGroup',
    })
    const secondGroupId = withSecondFolder.knowledgeSessionGroupOrder[0]

    const reordered = researchDeskReducer(withSecondFolder, {
      groupId: firstGroupId,
      targetIndex: 0,
      type: 'moveKnowledgeSessionGroup',
    })

    expect(withSecondFolder.knowledgeSessionGroupOrder).toEqual([secondGroupId, firstGroupId])
    expect(reordered.knowledgeSessionGroupOrder).toEqual([firstGroupId, secondGroupId])
  })
})

describe('chat folder reducer actions', () => {
  it('creates a chat thread inside the requested folder', () => {
    const withFolder = researchDeskReducer(createEmptyProjectState(), {
      title: 'Folder',
      type: 'createChatThreadGroup',
    })
    const groupId = withFolder.chatThreadGroupOrder[0]

    const next = researchDeskReducer(withFolder, {
      groupId,
      type: 'createChatThread',
    })
    const threadId = next.ui.selectedChatThreadId

    expect(threadId).toBeTruthy()
    expect(next.chatThreadGroupMemberships[threadId as string]).toBe(groupId)
    expect(next.chatThreadOrder[0]).toBe(threadId)
    expect(next.dirty).toBe(true)
  })

  it('dissolves a deleted chat folder while keeping its threads', () => {
    const base = createEmptyProjectState()
    const seeded = {
      ...base,
      chatThreadGroupMemberships: { 'ct-1': 'cg-1' },
      chatThreadGroupOrder: ['cg-1'],
      chatThreadGroups: {
        'cg-1': {
          createdAt: '2026-01-01T00:00:00.000Z',
          id: 'cg-1',
          title: 'Folder',
          updatedAt: '2026-01-01T00:00:00.000Z',
        },
      },
      chatThreadOrder: ['ct-1'],
      chatThreads: { 'ct-1': makeChatThread('ct-1', 'Chat') },
    }

    const next = researchDeskReducer(seeded, {
      groupId: 'cg-1',
      type: 'deleteChatThreadGroup',
    })

    expect(next.chatThreadGroups['cg-1']).toBeUndefined()
    expect(next.chatThreadGroupMemberships['ct-1']).toBeUndefined()
    expect(next.chatThreads['ct-1']).toBeDefined()
  })
})

describe('editor folder reducer actions', () => {
  it('dissolves a deleted editor folder while keeping its documents', () => {
    const base = createEmptyProjectState()
    const seeded = {
      ...base,
      editorDocumentOrder: ['doc-1'],
      editorDocuments: { 'doc-1': makeEditorDocument('doc-1', 'Doc', { folderId: 'folder-1' }) },
      editorFolderOrder: ['folder-1'],
      editorFolders: { 'folder-1': makeEditorFolder('folder-1', 'Folder') },
    }

    const next = researchDeskReducer(seeded, {
      folderId: 'folder-1',
      type: 'deleteEditorFolder',
    })

    expect(next.editorFolders['folder-1']).toBeUndefined()
    expect(next.editorDocuments['doc-1']).toMatchObject({ folderId: null })
  })
})

describe('file-asset reducer actions', () => {
  it('ingests assets into the store and order', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha')],
      type: 'ingestFileAssets',
    })
    expect(next.fileAssetOrder).toContain('f1')
    expect(next.fileAssets.f1.label).toBe('alpha')
    expect(next.dirty).toBe(true)
  })

  it('renames an asset label', () => {
    const seeded = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha')],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(seeded, { fileId: 'f1', label: 'renamed', type: 'renameFileAsset' })
    expect(next.fileAssets.f1.label).toBe('renamed')
  })

  it('upgrades a pending/errored client parse to a clean server parse, persisting it', () => {
    const seeded = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha', {
        parserId: 'client',
        extractedText: '',
        parseStatus: 'error',
        parseWarning: "undefined is not a function (near '...e of t...')",
        parsePending: true,
      })],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(seeded, {
      assetId: 'f1',
      extractedText: 'crisp markitdown text',
      type: 'upgradeFileAssetParse',
    })
    expect(next.fileAssets.f1.extractedText).toBe('crisp markitdown text')
    expect(next.fileAssets.f1.parserId).toBe('markitdown')
    // The server parse supersedes the failed client parse: clean result, no
    // lingering error or pending state.
    expect(next.fileAssets.f1.parseStatus).toBe('parsed')
    expect(next.fileAssets.f1.parseWarning).toBeNull()
    expect(next.fileAssets.f1.parsePending).toBe(false)
    expect(next.fileAssets.f1.updatedAt).not.toBe(seeded.fileAssets.f1.updatedAt)
    expect(next.dirty).toBe(true)
  })

  it('toggles the transient parse-pending flag without marking the project dirty', () => {
    const seeded = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha')],
      type: 'ingestFileAssets',
    })
    const clean = { ...seeded, dirty: false }
    const pending = researchDeskReducer(clean, { assetId: 'f1', pending: true, type: 'setFileAssetParsePending' })
    expect(pending.fileAssets.f1.parsePending).toBe(true)
    expect(pending.dirty).toBe(false) // transient, never synced
    const cleared = researchDeskReducer(pending, { assetId: 'f1', pending: false, type: 'setFileAssetParsePending' })
    expect(cleared.fileAssets.f1.parsePending).toBe(false)
  })

  it('never blanks a good client parse when the server text is empty', () => {
    const seeded = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha', { parserId: 'client', extractedText: 'keep me' })],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(seeded, {
      assetId: 'f1',
      extractedText: '   ',
      type: 'upgradeFileAssetParse',
    })
    expect(next.fileAssets.f1.extractedText).toBe('keep me')
    expect(next.fileAssets.f1.parserId).toBe('client')
  })

  it('moves an asset into an existing section group', () => {
    const base = createEmptyProjectState()
    const librarySectionId = base.fileLibrarySectionOrder.find(
      (id) => base.fileLibrarySections[id]?.kind === 'custom',
    )!
    const temporarySectionId = base.fileLibrarySectionOrder.find(
      (id) => base.fileLibrarySections[id]?.kind === 'temporary',
    )!
    const created = researchDeskReducer(base, {
      sectionId: librarySectionId,
      title: 'Group',
      type: 'createFileGroup',
    })
    const groupId = created.fileGroupOrder[0]
    const seeded = researchDeskReducer(created, {
      assets: [makeAsset('f1', 'alpha', { sectionId: temporarySectionId })],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(seeded, {
      fileId: 'f1',
      groupId,
      sectionId: librarySectionId,
      type: 'moveFileAsset',
    })
    expect(next.fileAssets.f1.sectionId).toBe(librarySectionId)
    expect(next.fileAssets.f1.groupId).toBe(groupId)
  })

  it('drops a move into a group that does not belong to the target section', () => {
    const base = createEmptyProjectState()
    const librarySectionId = base.fileLibrarySectionOrder.find(
      (id) => base.fileLibrarySections[id]?.kind === 'custom',
    )!
    const temporarySectionId = base.fileLibrarySectionOrder.find(
      (id) => base.fileLibrarySections[id]?.kind === 'temporary',
    )!
    const seeded = researchDeskReducer(base, {
      assets: [makeAsset('f1', 'alpha', { sectionId: temporarySectionId })],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(seeded, {
      fileId: 'f1',
      groupId: 'does-not-exist',
      sectionId: librarySectionId,
      type: 'moveFileAsset',
    })
    expect(next.fileAssets.f1.sectionId).toBe(librarySectionId)
    expect(next.fileAssets.f1.groupId).toBeNull()
  })

  it('deletes an asset and strips its pending chat reference', () => {
    const seeded = researchDeskReducer(createEmptyProjectState(), {
      assets: [makeAsset('f1', 'alpha')],
      type: 'ingestFileAssets',
    })
    const withRef = {
      ...seeded,
      ui: { ...seeded.ui, pendingChatAttachmentRefs: [{ fileId: 'f1', kind: 'file-asset' as const }] },
    }
    const next = researchDeskReducer(withRef, { fileId: 'f1', type: 'deleteFileAsset' })
    expect(next.fileAssets.f1).toBeUndefined()
    expect(next.fileAssetOrder).not.toContain('f1')
    expect(next.ui.pendingChatAttachmentRefs).toEqual([])
  })
})

describe('file-group reducer actions', () => {
  it('creates a group under a section', () => {
    const base = createEmptyProjectState()
    const librarySectionId = base.fileLibrarySectionOrder.find(
      (id) => base.fileLibrarySections[id]?.kind === 'custom',
    )!
    const next = researchDeskReducer(base, {
      sectionId: librarySectionId,
      title: 'New Group',
      type: 'createFileGroup',
    })
    expect(next.fileGroupOrder).toHaveLength(1)
    const groupId = next.fileGroupOrder[0]
    expect(next.fileGroups[groupId]).toMatchObject({ sectionId: librarySectionId, title: 'New Group' })
  })

  it('reassigns members to no group when their group is deleted', () => {
    const base = createEmptyProjectState()
    const librarySectionId = base.fileLibrarySectionOrder.find(
      (id) => base.fileLibrarySections[id]?.kind === 'custom',
    )!
    const created = researchDeskReducer(base, {
      sectionId: librarySectionId,
      title: 'Group',
      type: 'createFileGroup',
    })
    const groupId = created.fileGroupOrder[0]
    const withAsset = researchDeskReducer(created, {
      assets: [makeAsset('f1', 'alpha', { groupId, sectionId: librarySectionId })],
      type: 'ingestFileAssets',
    })
    const next = researchDeskReducer(withAsset, { groupId, type: 'deleteFileGroup' })
    expect(next.fileGroups[groupId]).toBeUndefined()
    expect(next.fileAssets.f1.groupId).toBeNull()
  })
})

describe('legacy resource identity migration', () => {
  it('rekeys a file section and all local parent references atomically', () => {
    const base = createEmptyProjectState()
    const originalId = base.fileLibrarySectionOrder.find(
      (id) => base.fileLibrarySections[id]?.kind === 'temporary',
    )!
    const legacyId = 'file-section-temp'
    const section = base.fileLibrarySections[originalId]
    const fileLibrarySections: ProjectState['fileLibrarySections'] = {
      ...base.fileLibrarySections,
      [legacyId]: { ...section, id: legacyId },
    }
    delete fileLibrarySections[originalId]
    const state = {
      ...base,
      fileAssetOrder: ['fa-1'],
      fileAssets: {
        'fa-1': makeAsset('fa-1', 'source', { sectionId: legacyId }),
      },
      fileGroupOrder: ['fg-1'],
      fileGroups: {
        'fg-1': {
          createdAt: section.createdAt,
          id: 'fg-1',
          sectionId: legacyId,
          title: 'Sources',
          updatedAt: section.updatedAt,
        },
      },
      fileLibrarySectionOrder: base.fileLibrarySectionOrder.map(
        (id) => id === originalId ? legacyId : id,
      ),
      fileLibrarySections,
    }

    const migrated = researchDeskReducer(state, {
      replacements: { [legacyId]: 'file-section-new' },
      type: 'rekeyFileLibrarySectionIds',
    })

    expect(migrated.fileLibrarySections[legacyId]).toBeUndefined()
    expect(migrated.fileLibrarySections['file-section-new']?.id).toBe('file-section-new')
    expect(migrated.fileGroups['fg-1'].sectionId).toBe('file-section-new')
    expect(migrated.fileAssets['fa-1'].sectionId).toBe('file-section-new')
    expect(migrated.fileLibrarySectionOrder).toContain('file-section-new')
  })

  it('rekeys a knowledge session and every item, membership, selection, and pin', () => {
    const base = createEmptyProjectState()
    const originalId = base.selectedKnowledgeSessionId!
    const legacyId = 'knowledge-session-default'
    const session = base.knowledgeSessions[originalId]
    const state = {
      ...base,
      knowledgeItemOrder: ['ki-1'],
      knowledgeItems: { 'ki-1': makeKnowledgeItem('ki-1', legacyId) },
      knowledgeSessionGroupMemberships: { [legacyId]: 'kg-1' },
      knowledgeSessionOrder: [legacyId],
      knowledgeSessions: { [legacyId]: { ...session, id: legacyId } },
      selectedKnowledgeSessionId: legacyId,
      ui: {
        ...base.ui,
        pinnedExplorer: {
          ...base.ui.pinnedExplorer,
          knowledgeSessionIds: [legacyId],
        },
      },
    }

    const migrated = researchDeskReducer(state, {
      replacements: { [legacyId]: 'ks-new' },
      type: 'rekeyKnowledgeSessionIds',
    })

    expect(migrated.knowledgeSessions[legacyId]).toBeUndefined()
    expect(migrated.knowledgeSessions['ks-new']?.id).toBe('ks-new')
    expect(migrated.knowledgeItems['ki-1'].sessionId).toBe('ks-new')
    expect(migrated.knowledgeSessionGroupMemberships).toEqual({ 'ks-new': 'kg-1' })
    expect(migrated.selectedKnowledgeSessionId).toBe('ks-new')
    expect(migrated.ui.pinnedExplorer.knowledgeSessionIds).toEqual(['ks-new'])
    expect(migrated.dirty).toBe(true)
  })
})

describe('chat rule reducer actions', () => {
  it('upserts legacy rules with prompt-library defaults', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      rule: makeRule('r1', 'legacy'),
      type: 'upsertChatRule',
    })

    expect(next.chatRules.r1).toMatchObject({
      category: 'instruction',
      includeInAutocomplete: true,
      linkedContextRefs: [],
      visibility: { chat: true, editor: true },
    })
    expect(next.chatRuleOrder).toEqual(['r1'])
  })

  it('keeps only database references on context-pack rules', () => {
    const next = researchDeskReducer(createEmptyProjectState(), {
      rule: makeRule('r1', 'profile', {
        category: 'context',
        linkedContextRefs: [
          { fileId: 'f1', kind: 'file-asset' },
          { kind: 'chat-rule', ruleId: 'nested-rule' },
          { groupId: 'g1', kind: 'file-group' },
        ],
      }),
      type: 'upsertChatRule',
    })

    expect(next.chatRules.r1.linkedContextRefs).toEqual([
      { fileId: 'f1', kind: 'file-asset' },
      { groupId: 'g1', kind: 'file-group' },
    ])
  })

  it('stores rendered context-pack content on chat attachments', () => {
    const base = createEmptyProjectState()
    const seeded = {
      ...base,
      chatRuleOrder: ['r1'],
      chatRules: {
        r1: makeRule('r1', 'profile', {
          category: 'context',
          contentMarkdown: 'Apply the profile.\n{{context}}',
          linkedContextRefs: [{ fileId: 'f1', kind: 'file-asset' }],
        }),
      },
      fileAssetOrder: ['f1'],
      fileAssets: {
        f1: makeAsset('f1', 'alpha', { extractedText: 'Original profile content.' }),
      },
    }

    const next = researchDeskReducer(seeded, {
      assistantMessageId: 'a1',
      attachmentRefs: [{ kind: 'chat-rule', ruleId: 'r1' }],
      contentMarkdown: 'Use @rules:profile',
      createdAt: '2026-01-03T00:00:00.000Z',
      threadId: 'thread-1',
      type: 'startChatExchange',
      userMessageId: 'u1',
    })

    const attachment = next.chatThreads['thread-1'].messages[0].attachments?.[0]
    expect(attachment).toMatchObject({ kind: 'chat-rule', label: 'profile' })
    expect(attachment?.contentMarkdown).toContain('Apply the profile.')
    expect(attachment?.contentMarkdown).toContain('Original profile content.')
    expect(attachment?.contentMarkdown).not.toContain('{{context}}')
  })
})

describe('reorderChatContextInDraft', () => {
  function withPending(ruleIds: string[]) {
    const base = createEmptyProjectState()
    return {
      ...base,
      ui: {
        ...base.ui,
        pendingChatAttachmentRefs: ruleIds.map((ruleId) => ({ kind: 'chat-rule' as const, ruleId })),
      },
    }
  }

  it('moves a pending ref to a new index', () => {
    const next = researchDeskReducer(withPending(['a', 'b', 'c']), {
      fromIndex: 0,
      toIndex: 2,
      type: 'reorderChatContextInDraft',
    })
    expect(next.ui.pendingChatAttachmentRefs.map((ref) => (ref as { ruleId: string }).ruleId)).toEqual(['b', 'c', 'a'])
    expect(next.dirty).toBe(true)
  })

  it('ignores no-op and out-of-bounds reorders', () => {
    const seeded = withPending(['a', 'b'])
    expect(researchDeskReducer(seeded, { fromIndex: 0, toIndex: 0, type: 'reorderChatContextInDraft' })).toBe(seeded)
    expect(researchDeskReducer(seeded, { fromIndex: 0, toIndex: 5, type: 'reorderChatContextInDraft' })).toBe(seeded)
  })
})

describe('vector index reducer actions', () => {
  function withAssets(...labels: string[]) {
    return researchDeskReducer(createEmptyProjectState(), {
      assets: labels.map((label) => makeAsset(label, label, { pageCount: 10 })),
      type: 'ingestFileAssets',
    })
  }

  it('creates an index with members as stale and an empty index as ready', () => {
    const seeded = withAssets('a', 'b')
    const withIndex = researchDeskReducer(seeded, { fileIds: ['a', 'b'], title: 'EU Recht', type: 'createVectorIndex' })
    const index = withIndex.vectorIndexes[withIndex.vectorIndexOrder[0]]
    expect(index.title).toBe('EU Recht')
    expect(index.handle).toBe('eu-recht')
    expect(index.dims).toBe(3072)
    expect(index.status).toBe('stale')
    expect(index.members.map((member) => member.fileId)).toEqual(['a', 'b'])
    expect(index.members.every((member) => member.state === 'pending')).toBe(true)

    const empty = researchDeskReducer(seeded, { fileIds: [], title: 'Empty', type: 'createVectorIndex' })
    expect(empty.vectorIndexes[empty.vectorIndexOrder[0]].status).toBe('ready')
  })

  it('assigns unique handles for duplicate titles', () => {
    let state = withAssets('a')
    state = researchDeskReducer(state, { fileIds: [], title: 'EU', type: 'createVectorIndex' })
    state = researchDeskReducer(state, { fileIds: [], title: 'EU', type: 'createVectorIndex' })
    const handles = state.vectorIndexOrder.map((id) => state.vectorIndexes[id].handle).sort()
    expect(handles).toEqual(['eu', 'eu-2'])
  })

  it('re-slugs the handle on rename', () => {
    let state = researchDeskReducer(withAssets('a'), { fileIds: ['a'], title: 'Old', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, title: 'Neuer Titel', type: 'renameVectorIndex' })
    expect(state.vectorIndexes[id].title).toBe('Neuer Titel')
    expect(state.vectorIndexes[id].handle).toBe('neuer-titel')
  })

  it('adds documents as pending and marks the index stale', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'demo', totalDocuments: 1, type: 'startVectorIndexReindex' })
    state = researchDeskReducer(state, { indexId: id, type: 'completeVectorIndexReindex' })
    expect(state.vectorIndexes[id].status).toBe('ready')
    state = researchDeskReducer(state, { fileIds: ['b'], indexId: id, type: 'addDocsToVectorIndex' })
    expect(state.vectorIndexes[id].status).toBe('stale')
    expect(state.vectorIndexes[id].members.find((member) => member.fileId === 'b')?.state).toBe('pending')
  })

  it('changing the model updates dims, marks stale, and resets members to pending', () => {
    let state = researchDeskReducer(withAssets('a'), { fileIds: ['a'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'demo', totalDocuments: 1, type: 'startVectorIndexReindex' })
    state = researchDeskReducer(state, { indexId: id, type: 'completeVectorIndexReindex' })
    expect(state.vectorIndexes[id].members[0].state).toBe('embedded')
    state = researchDeskReducer(state, { indexId: id, model: 'text-embedding-3-small', type: 'setVectorIndexModel' })
    expect(state.vectorIndexes[id].dims).toBe(1536)
    expect(state.vectorIndexes[id].status).toBe('stale')
    expect(state.vectorIndexes[id].members[0].state).toBe('pending')
  })

  it('reindex marks indexing then ready with embedded members; complete is a no-op otherwise', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    const indexing = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'demo', totalDocuments: 2, type: 'startVectorIndexReindex' })
    expect(indexing.vectorIndexes[id].status).toBe('indexing')
    state = researchDeskReducer(indexing, { indexId: id, type: 'completeVectorIndexReindex' })
    expect(state.vectorIndexes[id].status).toBe('ready')
    expect(state.vectorIndexes[id].members.every((member) => member.state === 'embedded')).toBe(true)
    expect(researchDeskReducer(state, { indexId: id, type: 'completeVectorIndexReindex' })).toBe(state)
  })

  it('marks no-text members terminal skipped and reads ready (not perpetually stale)', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'build', totalDocuments: 2, type: 'startVectorIndexReindex' })
    // 'a' embedded; 'b' carried no extractable text -> terminal skipped, never embeds.
    state = researchDeskReducer(state, {
      embeddedFileIds: ['a'],
      skippedFileIds: ['b'],
      indexId: id,
      serverCollectionId: 'kc_live',
      serverCollectionModel: 'text-embedding-3-large',
      type: 'completeVectorIndexReindex',
    })
    const members = state.vectorIndexes[id].members
    expect(members.find((member) => member.fileId === 'a')?.state).toBe('embedded')
    expect(members.find((member) => member.fileId === 'b')?.state).toBe('skipped')
    // Skipped is terminal (a no-text doc can never embed), so nothing is pending
    // and the index reads ready — not a perpetual stale that prompts a futile re-index.
    expect(state.vectorIndexes[id].status).toBe('ready')
    expect(state.vectorIndexes[id].serverCollectionModel).toBe('text-embedding-3-large')
  })

  it('stays stale while a member is genuinely pending (not yet ingested, not skipped)', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'build', totalDocuments: 2, type: 'startVectorIndexReindex' })
    // Only 'a' ingested this run; 'b' was neither embedded nor skipped -> still pending.
    state = researchDeskReducer(state, {
      embeddedFileIds: ['a'],
      indexId: id,
      serverCollectionId: 'kc_live',
      type: 'completeVectorIndexReindex',
    })
    const members = state.vectorIndexes[id].members
    expect(members.find((member) => member.fileId === 'b')?.state).toBe('pending')
    expect(state.vectorIndexes[id].status).toBe('stale')
  })

  it('keeps a previously-skipped member skipped across a later run that does not touch it', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'build', totalDocuments: 2, type: 'startVectorIndexReindex' })
    state = researchDeskReducer(state, { embeddedFileIds: ['a'], skippedFileIds: ['b'], indexId: id, serverCollectionId: 'kc', type: 'completeVectorIndexReindex' })
    // A later incremental run touches only 'a' (b is neither embedded nor skipped
    // THIS run). 'b' must NOT silently revert to pending — skipped is terminal.
    state = researchDeskReducer(state, { indexId: id, jobId: 'j2', source: 'build', totalDocuments: 1, type: 'startVectorIndexReindex' })
    state = researchDeskReducer(state, { embeddedFileIds: ['a'], indexId: id, serverCollectionId: 'kc', type: 'completeVectorIndexReindex' })
    expect(state.vectorIndexes[id].members.find((member) => member.fileId === 'b')?.state).toBe('skipped')
    expect(state.vectorIndexes[id].status).toBe('ready')
  })

  it('marks the complete embedded set ready when every member ingested', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'build', totalDocuments: 2, type: 'startVectorIndexReindex' })
    state = researchDeskReducer(state, {
      embeddedFileIds: ['a', 'b'],
      indexId: id,
      serverCollectionId: 'kc_live',
      serverCollectionModel: 'text-embedding-3-large',
      type: 'completeVectorIndexReindex',
    })
    expect(state.vectorIndexes[id].members.every((member) => member.state === 'embedded')).toBe(true)
    expect(state.vectorIndexes[id].status).toBe('ready')
  })

  it('markVectorIndexProgress advances the bar and records per-file outcomes live', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'build', totalDocuments: 2, type: 'startVectorIndexReindex' })
    state = researchDeskReducer(state, {
      completedDocuments: 1, embedded: true, fileId: 'a', indexId: id,
      totalDocuments: 2, type: 'markVectorIndexProgress',
    })
    state = researchDeskReducer(state, {
      completedDocuments: 2, embedded: false, fileId: 'b', indexId: id,
      totalDocuments: 2, type: 'markVectorIndexProgress',
    })
    const live = state.indexingJobs[id]
    expect(live.completedDocuments).toBe(2)
    expect(live.percent).toBe(100)
    expect(live.embeddedFileIds).toEqual(['a'])
    expect(live.skippedFileIds).toEqual(['b'])
  })

  it('persists each member server-document id on completion (for later removal)', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'build', totalDocuments: 2, type: 'startVectorIndexReindex' })
    state = researchDeskReducer(state, {
      embeddedFileIds: ['a', 'b'],
      indexId: id,
      serverCollectionId: 'kc_live',
      serverCollectionModel: 'text-embedding-3-large',
      serverDocumentIds: { a: 'kd_a', b: 'kd_b' },
      type: 'completeVectorIndexReindex',
    })
    const members = state.vectorIndexes[id].members
    expect(members.find((member) => member.fileId === 'a')?.serverDocumentId).toBe('kd_a')
    expect(members.find((member) => member.fileId === 'b')?.serverDocumentId).toBe('kd_b')
  })

  it('removing the last member returns the index to ready', () => {
    let state = researchDeskReducer(withAssets('a'), { fileIds: ['a'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    expect(state.vectorIndexes[id].status).toBe('stale')
    state = researchDeskReducer(state, { fileId: 'a', indexId: id, type: 'removeDocFromVectorIndex' })
    expect(state.vectorIndexes[id].members).toHaveLength(0)
    expect(state.vectorIndexes[id].status).toBe('ready')
  })

  it('accepts a server-catalog model and explicit dims at creation and model change', () => {
    let state = researchDeskReducer(withAssets('a'), {
      dims: 1024,
      fileIds: ['a'],
      model: 'BAAI/bge-m3',
      title: 'Server',
      type: 'createVectorIndex',
    })
    const id = state.vectorIndexOrder[0]
    expect(state.vectorIndexes[id].model).toBe('BAAI/bge-m3')
    expect(state.vectorIndexes[id].dims).toBe(1024)

    state = researchDeskReducer(state, {
      dims: 3072,
      indexId: id,
      model: 'custom/server-model',
      type: 'setVectorIndexModel',
    })
    expect(state.vectorIndexes[id].model).toBe('custom/server-model')
    expect(state.vectorIndexes[id].dims).toBe(3072)
  })

  it('stores the server collection id on completion and clears errors on retry', () => {
    let state = researchDeskReducer(withAssets('a'), { fileIds: ['a'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]

    state = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'demo', totalDocuments: 1, type: 'startVectorIndexReindex' })
    state = researchDeskReducer(state, {
      indexId: id,
      serverCollectionId: 'kc_live',
      type: 'completeVectorIndexReindex',
    })
    expect(state.vectorIndexes[id].serverCollectionId).toBe('kc_live')
    expect(state.vectorIndexes[id].status).toBe('ready')

    // A terminal error only lands while a run is actually in flight.
    state = researchDeskReducer(state, { indexId: id, jobId: 'j2', source: 'server', totalDocuments: 1, type: 'startVectorIndexReindex' })
    state = researchDeskReducer(state, {
      indexId: id,
      message: 'embedding backend down',
      type: 'markVectorIndexError',
    })
    expect(state.vectorIndexes[id].status).toBe('error')
    expect(state.vectorIndexes[id].lastError).toBe('embedding backend down')

    state = researchDeskReducer(state, { indexId: id, jobId: 'j3', source: 'server', totalDocuments: 1, type: 'startVectorIndexReindex' })
    expect(state.vectorIndexes[id].status).toBe('indexing')
    expect(state.vectorIndexes[id].lastError).toBeNull()
    state = researchDeskReducer(state, { indexId: id, type: 'completeVectorIndexReindex' })
    expect(state.vectorIndexes[id].serverCollectionId).toBe('kc_live')
  })

  it('deleting a file drops it from every index membership', () => {
    let state = researchDeskReducer(withAssets('a', 'b'), { fileIds: ['a', 'b'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { fileId: 'a', type: 'deleteFileAsset' })
    expect(state.fileAssets.a).toBeUndefined()
    expect(state.vectorIndexes[id].members.map((member) => member.fileId)).toEqual(['b'])
  })

  it('deleting a custom section removes its assets and their index memberships', () => {
    let state = researchDeskReducer(createEmptyProjectState(), { sectionId: '', title: 'Custom', type: 'createFileLibrarySection' })
    const sectionId = state.fileLibrarySectionOrder[state.fileLibrarySectionOrder.length - 1]
    state = researchDeskReducer(state, { assets: [makeAsset('a', 'a', { sectionId })], type: 'ingestFileAssets' })
    state = researchDeskReducer(state, { fileIds: ['a'], title: 'X', type: 'createVectorIndex' })
    const id = state.vectorIndexOrder[0]
    state = researchDeskReducer(state, { sectionId, type: 'deleteFileLibrarySection' })
    expect(state.fileLibrarySections[sectionId]).toBeUndefined()
    expect(state.fileAssets.a).toBeUndefined()
    expect(state.vectorIndexes[id].members).toHaveLength(0)
  })

  it('refuses to delete the temporary section', () => {
    const state = createEmptyProjectState()
    const temporarySectionId = state.fileLibrarySectionOrder.find(
      (id) => state.fileLibrarySections[id]?.kind === 'temporary',
    )
    expect(temporarySectionId).toBeDefined()
    expect(researchDeskReducer(state, {
      sectionId: temporarySectionId!,
      type: 'deleteFileLibrarySection',
    })).toBe(state)
  })
})

describe('live indexing-job lifecycle', () => {
  function withIndex(...fileIds: string[]) {
    const seeded = researchDeskReducer(createEmptyProjectState(), {
      assets: fileIds.map((id) => makeAsset(id, id, { pageCount: 10 })),
      type: 'ingestFileAssets',
    })
    const state = researchDeskReducer(seeded, { fileIds, title: 'X', type: 'createVectorIndex' })
    return { id: state.vectorIndexOrder[0], state }
  }

  it('starting a reindex opens a live job and flips the index to indexing', () => {
    const { id, state } = withIndex('a', 'b')
    const started = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'server', totalDocuments: 2, type: 'startVectorIndexReindex' })
    expect(started.vectorIndexes[id].status).toBe('indexing')
    expect(started.indexingJobs[id]).toMatchObject({ completedDocuments: 0, jobId: 'j1', percent: 0, source: 'server', totalDocuments: 2 })
  })

  it('defaults runningFileIds to every member when the action omits it (server re-embed)', () => {
    // The durable server re-embed does not pass a working set — it re-vectorizes
    // the whole collection, so every member must be in the run.
    const { id, state } = withIndex('a', 'b')
    const started = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'server', totalDocuments: 2, type: 'startVectorIndexReindex' })
    expect([...started.indexingJobs[id].runningFileIds].sort()).toEqual(['a', 'b'])
  })

  it('uses the explicit runningFileIds subset (incremental add) so only those files run', () => {
    // Indexing one new document must NOT pull the already-embedded members into
    // the run — only the passed subset is the working set.
    const { id, state } = withIndex('a', 'b')
    const started = researchDeskReducer(state, { indexId: id, jobId: 'j1', runningFileIds: ['b'], source: 'build', totalDocuments: 1, type: 'startVectorIndexReindex' })
    expect(started.indexingJobs[id].runningFileIds).toEqual(['b'])
  })

  it('markVectorIndexDocumentEmbedded maps a server document id to its file and flips it live', () => {
    // The durable server re-embed confirms documents by backend id; the reducer
    // resolves it to the local file via the member's serverDocumentId.
    const { id, state } = withIndex('a', 'b')
    const model = state.vectorIndexes[id].model
    let next = researchDeskReducer(state, { indexId: id, jobId: 'j0', source: 'build', totalDocuments: 2, type: 'startVectorIndexReindex' })
    // A completed build assigns each member its backend document id.
    next = researchDeskReducer(next, {
      embeddedFileIds: ['a', 'b'],
      indexId: id,
      serverCollectionId: 'col-1',
      serverCollectionModel: model,
      serverDocumentIds: { a: 'kd_a', b: 'kd_b' },
      type: 'completeVectorIndexReindex',
    })
    next = researchDeskReducer(next, { indexId: id, jobId: 'j1', source: 'server', totalDocuments: 2, type: 'startVectorIndexReindex' })
    const clean = { ...next, dirty: false }
    const flipped = researchDeskReducer(clean, { indexId: id, serverDocumentId: 'kd_a', type: 'markVectorIndexDocumentEmbedded' })
    // Only file 'a' flips; the project is not dirtied (ephemeral live state).
    expect(flipped.dirty).toBe(false)
    expect(flipped.indexingJobs[id].embeddedFileIds).toContain('a')
    expect(flipped.indexingJobs[id].embeddedFileIds ?? []).not.toContain('b')
  })

  it('markVectorIndexDocumentEmbedded ignores an unknown server document id (older index)', () => {
    // A member without a tracked serverDocumentId cannot be mapped → no-op; the
    // row flips at completion instead. State is returned unchanged.
    const { id, state } = withIndex('a')
    const started = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'server', totalDocuments: 1, type: 'startVectorIndexReindex' })
    const same = researchDeskReducer(started, { indexId: id, serverDocumentId: 'kd_unknown', type: 'markVectorIndexDocumentEmbedded' })
    expect(same).toBe(started)
  })

  it('progress updates the live job without dirtying the project', () => {
    const { id, state } = withIndex('a', 'b')
    const started = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'server', totalDocuments: 2, type: 'startVectorIndexReindex' })
    // Streaming progress is the hot path: it must never mark the project dirty.
    const clean = { ...started, dirty: false }
    const progressed = researchDeskReducer(clean, {
      completedDocuments: 1,
      currentDocumentTitle: 'doc-a',
      indexId: id,
      totalDocuments: 2,
      type: 'markVectorIndexProgress',
    })
    expect(progressed.dirty).toBe(false)
    expect(progressed.indexingJobs[id].percent).toBe(50)
    expect(progressed.indexingJobs[id].currentDocumentTitle).toBe('doc-a')
  })

  it('queued sets the FIFO position (dirty-free) and progress clears it', () => {
    const { id, state } = withIndex('a', 'b')
    const started = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'server', totalDocuments: 2, type: 'startVectorIndexReindex' })
    const clean = { ...started, dirty: false }
    const queued = researchDeskReducer(clean, { indexId: id, queuePosition: 3, type: 'markVectorIndexQueued' })
    expect(queued.dirty).toBe(false)
    expect(queued.indexingJobs[id].queuePosition).toBe(3)
    // A progress tick means a slot freed up → leave the waiting state.
    const running = researchDeskReducer(queued, { completedDocuments: 1, indexId: id, totalDocuments: 2, type: 'markVectorIndexProgress' })
    expect(running.indexingJobs[id].queuePosition).toBeNull()
  })

  it('completion records an ok history entry and clears the live job', () => {
    const { id, state } = withIndex('a', 'b')
    let next = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'server', totalDocuments: 2, type: 'startVectorIndexReindex' })
    next = researchDeskReducer(next, { indexId: id, type: 'completeVectorIndexReindex' })
    expect(next.indexingJobs[id]).toBeUndefined()
    const history = next.vectorIndexes[id].history ?? []
    expect(history).toHaveLength(1)
    expect(history[0]).toMatchObject({ documents: 2, result: 'ok' })
  })

  it('an error records an error entry with the message and clears the live job', () => {
    const { id, state } = withIndex('a', 'b')
    let next = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'server', totalDocuments: 2, type: 'startVectorIndexReindex' })
    next = researchDeskReducer(next, { completedDocuments: 1, indexId: id, totalDocuments: 2, type: 'markVectorIndexProgress' })
    next = researchDeskReducer(next, { indexId: id, message: 'embedding backend down', type: 'markVectorIndexError' })
    expect(next.indexingJobs[id]).toBeUndefined()
    expect(next.vectorIndexes[id].status).toBe('error')
    const entry = (next.vectorIndexes[id].history ?? [])[0]
    expect(entry).toMatchObject({ documents: 1, error: 'embedding backend down', result: 'error' })
  })

  it('cancellation records a cancelled entry, clears the job, and restores stale when members are still pending', () => {
    const { id, state } = withIndex('a', 'b')
    let next = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'demo', totalDocuments: 2, type: 'startVectorIndexReindex' })
    next = researchDeskReducer(next, { indexId: id, type: 'markVectorIndexCancelled' })
    expect(next.indexingJobs[id]).toBeUndefined()
    expect(next.vectorIndexes[id].status).toBe('stale')
    expect((next.vectorIndexes[id].history ?? [])[0]).toMatchObject({ result: 'cancelled' })
  })

  it('cancellation restores ready when every member is already embedded', () => {
    const { id, state } = withIndex('a')
    let next = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'demo', totalDocuments: 1, type: 'startVectorIndexReindex' })
    next = researchDeskReducer(next, { indexId: id, type: 'completeVectorIndexReindex' })
    next = researchDeskReducer(next, { indexId: id, jobId: 'j2', source: 'demo', totalDocuments: 1, type: 'startVectorIndexReindex' })
    next = researchDeskReducer(next, { indexId: id, type: 'markVectorIndexCancelled' })
    expect(next.vectorIndexes[id].status).toBe('ready')
  })

  it('a late terminal callback after a finished run is a no-op (no resurrected run, no garbage history)', () => {
    const { id, state } = withIndex('a', 'b')
    let next = researchDeskReducer(state, { indexId: id, jobId: 'j1', source: 'server', totalDocuments: 2, type: 'startVectorIndexReindex' })
    next = researchDeskReducer(next, { indexId: id, type: 'completeVectorIndexReindex' })
    const settled = next
    // A duplicate error/cancel arriving after completion (resume race) must
    // not flip the finished index nor append a second history entry.
    next = researchDeskReducer(next, { indexId: id, message: 'late failure', type: 'markVectorIndexError' })
    next = researchDeskReducer(next, { indexId: id, type: 'markVectorIndexCancelled' })
    expect(next).toBe(settled)
    expect(next.vectorIndexes[id].status).toBe('ready')
    expect(next.vectorIndexes[id].history ?? []).toHaveLength(1)
  })
})

describe('editor document revision = server base (A2 CAS contract)', () => {
  it('does NOT bump revision on rename/move; only updatedAt moves', () => {
    const base = {
      ...createEmptyProjectState(),
      editorDocumentOrder: ['doc-1'],
      editorDocuments: { 'doc-1': makeEditorDocument('doc-1', 'Doc') },
    }
    // revision is the last-synced server base and stays put across local
    // edits; only updatedAt moves so the debounced autosave fires. The save
    // sends base+1 and the store CAS is against the base — a per-edit bump
    // would be the old, race-prone counter that P1 exploited.
    const renamed = researchDeskReducer(base, {
      documentId: 'doc-1',
      title: 'Neuer Titel',
      type: 'renameEditorDocument',
    })
    expect(renamed.editorDocuments['doc-1'].revision).toBe(1)
    expect(renamed.editorDocuments['doc-1'].updatedAt).not.toBe(
      base.editorDocuments['doc-1'].updatedAt,
    )
    expect(renamed.dirty).toBe(true)

    const moved = researchDeskReducer(renamed, {
      documentId: 'doc-1',
      folderId: 'edf_x',
      targetIndex: 0,
      type: 'moveEditorDocumentToFolder',
    })
    expect(moved.editorDocuments['doc-1'].revision).toBe(1)
  })

  it('leaves revision put on documents orphaned by a folder delete', () => {
    const base = {
      ...createEmptyProjectState(),
      editorDocumentOrder: ['doc-1'],
      editorDocuments: {
        'doc-1': makeEditorDocument('doc-1', 'Doc', { folderId: 'edf_1' }),
      },
      editorFolders: {
        edf_1: {
          createdAt: '2026-01-01T00:00:00.000Z',
          id: 'edf_1',
          title: 'Ordner',
          updatedAt: '2026-01-01T00:00:00.000Z',
        },
      },
      editorFolderOrder: ['edf_1'],
    }
    const next = researchDeskReducer(base, {
      folderId: 'edf_1',
      type: 'deleteEditorFolder',
    })
    expect(next.editorDocuments['doc-1'].folderId).toBeNull()
    expect(next.editorDocuments['doc-1'].revision).toBe(1)
  })

  it('adopts the server revision after a successful save (base tracks server)', () => {
    const base = {
      ...createEmptyProjectState(),
      editorDocumentOrder: ['doc-1'],
      editorDocuments: { 'doc-1': makeEditorDocument('doc-1', 'Doc') },
    }
    // A save based on base 1 advanced the server to 2; adopt it, revision-only.
    const adopted = researchDeskReducer(base, {
      documentId: 'doc-1',
      revision: 2,
      type: 'adoptEditorDocumentRevision',
    })
    expect(adopted.editorDocuments['doc-1'].revision).toBe(2)
    expect(adopted.editorDocuments['doc-1'].updatedAt).toBe(
      base.editorDocuments['doc-1'].updatedAt,
    )
    expect(adopted.dirty).toBe(base.dirty)
    // A late/duplicate adopt must never rewind a fresher base.
    const noop = researchDeskReducer(adopted, {
      documentId: 'doc-1',
      revision: 1,
      type: 'adoptEditorDocumentRevision',
    })
    expect(noop).toBe(adopted)
  })

  it('rebase (no live edit) adopts the server base, keeps updatedAt', () => {
    const base = {
      ...createEmptyProjectState(),
      editorDocumentOrder: ['doc-1'],
      editorDocuments: { 'doc-1': makeEditorDocument('doc-1', 'Doc') },
    }
    const rebased = researchDeskReducer(base, {
      contentMarkdown: 'server body',
      documentId: 'doc-1',
      pushedContentMarkdown: base.editorDocuments['doc-1'].contentMarkdown,
      revision: 7,
      type: 'rebaseServerEditorDocument',
    })
    expect(rebased.editorDocuments['doc-1'].revision).toBe(7)
    expect(rebased.editorDocuments['doc-1'].contentMarkdown).toBe('server body')
    expect(rebased.editorDocuments['doc-1'].updatedAt).toBe(
      base.editorDocuments['doc-1'].updatedAt,
    )
  })

  it('rebase keeps a live keystroke and adopts the server base (no silent loss)', () => {
    const base = {
      ...createEmptyProjectState(),
      editorDocumentOrder: ['doc-1'],
      editorDocuments: { 'doc-1': makeEditorDocument('doc-1', 'Doc') },
    }
    // The flush PUT "# Doc"; during the PUT->GET window the user typed.
    const typed = researchDeskReducer(base, {
      documentId: 'doc-1',
      contentMarkdown: '# Doc + live edit',
      type: 'updateEditorDocumentMarkdown',
    })
    const rebased = researchDeskReducer(typed, {
      contentMarkdown: 'server body from agent patch',
      documentId: 'doc-1',
      pushedContentMarkdown: '# Doc', // what the failed PUT carried
      revision: 7,
      type: 'rebaseServerEditorDocument',
    })
    // The live keystroke is NOT overwritten by the server body…
    expect(rebased.editorDocuments['doc-1'].contentMarkdown).toBe(
      '# Doc + live edit',
    )
    // …and revision rebases onto the server BASE (7); the next flush re-pushes
    // the live edit as base+1 (8), staying dirty. No silent loss.
    expect(rebased.editorDocuments['doc-1'].revision).toBe(7)
    expect(rebased.dirty).toBe(true)
  })
})

describe('agent canvas is reconciled on session switch', () => {
  it('clears canvas tabs when a different session is selected', () => {
    const base = createEmptyProjectState()
    const withSessions = {
      ...base,
      agentSessions: {
        's1': {
          id: 's1', title: 'A', groupId: null, createdAt: '', updatedAt: '', runIds: [],
          sourcePolicy: { web: 'available' as const, knowledge: 'available' as const },
        },
        's2': {
          id: 's2', title: 'B', groupId: null, createdAt: '', updatedAt: '', runIds: [],
          sourcePolicy: { web: 'available' as const, knowledge: 'available' as const },
        },
      },
      selectedAgentSessionId: 's1',
    }
    // Open a pinned plan tab in session s1.
    const withTab = researchDeskReducer(withSessions, {
      descriptor: { runId: 'r1', view: 'plan' },
      source: 'user',
      type: 'openAgentCanvasView',
    })
    expect(withTab.agentCanvas.tabs).toHaveLength(1)
    expect(withTab.agentCanvas.open).toBe(true)

    // Switching to s2 must clear the canvas so s1's tab does not leak.
    const switched = researchDeskReducer(withTab, {
      sessionId: 's2',
      type: 'selectAgentSession',
    })
    expect(switched.selectedAgentSessionId).toBe('s2')
    expect(switched.agentCanvas.tabs).toEqual([])
    expect(switched.agentCanvas.open).toBe(false)
    expect(switched.agentCanvas.activeTabId).toBeNull()

    // Re-selecting the SAME session is a no-op (canvas preserved).
    const withTabAgain = researchDeskReducer(switched, {
      descriptor: { runId: 'r2', view: 'run' },
      source: 'user',
      type: 'openAgentCanvasView',
    })
    const same = researchDeskReducer(withTabAgain, {
      sessionId: 's2',
      type: 'selectAgentSession',
    })
    expect(same).toBe(withTabAgain)
  })

  it('clears canvas when creating (auto-selecting) a new session', () => {
    const base = { ...createEmptyProjectState(), selectedAgentSessionId: 's1' }
    const withTab = researchDeskReducer(base, {
      descriptor: { runId: 'r1', view: 'plan' },
      source: 'user',
      type: 'openAgentCanvasView',
    })
    expect(withTab.agentCanvas.tabs).toHaveLength(1)
    const created = researchDeskReducer(withTab, {
      session: {
        id: 's2',
        title: 'New',
        groupId: null,
        createdAt: '',
        updatedAt: '',
        runIds: [],
        sourcePolicy: { web: 'available', knowledge: 'available' },
      },
      type: 'createAgentSession',
    })
    expect(created.selectedAgentSessionId).toBe('s2')
    expect(created.agentCanvas.tabs).toEqual([])
  })

  it('clears canvas when a select=true run summary switches sessions', () => {
    const base = { ...createEmptyProjectState(), selectedAgentSessionId: 's1' }
    const withTab = researchDeskReducer(base, {
      descriptor: { runId: 'r1', view: 'plan' },
      source: 'user',
      type: 'openAgentCanvasView',
    })
    const summary: ResearchRunSummary = {
      access: { mode: 'owner' },
      run_id: 'r-new',
      status: 'running',
      queue_position: null,
      question: 'Neue Analyse',
      stack: 'default',
      mode: 'workspace_agent',
      kind: 'agent',
      session_id: 's-other',
      agent_overrides: {},
      created_at: 1_700_000_000,
      started_at: 1_700_000_000,
      finished_at: null,
      elapsed_seconds: null,
      snapshot: {},
      error: null,
      events_url: '/v1/runs/r-new/events',
      result_url: '/v1/runs/r-new/result',
    }
    const upserted = researchDeskReducer(withTab, {
      select: true,
      summary,
      type: 'upsertAgentRunSummary',
    })
    expect(upserted.selectedAgentSessionId).toBe('s-other')
    expect(upserted.agentCanvas.tabs).toEqual([])
  })
})

describe('agent plan stale re-flag (plan tab on-open fetch)', () => {
  const summary: ResearchRunSummary = {
    access: { mode: 'owner' },
    run_id: 'r-plan',
    status: 'waiting_for_approval',
    queue_position: null,
    question: 'Planlauf',
    stack: 'default',
    mode: 'workspace_agent',
    kind: 'agent',
    session_id: 's-plan',
    agent_overrides: {},
    created_at: 1_700_000_000,
    started_at: 1_700_000_000,
    finished_at: null,
    elapsed_seconds: null,
    snapshot: {},
    error: null,
    events_url: '/v1/runs/r-plan/events',
    result_url: '/v1/runs/r-plan/result',
  }

  it('re-flags a cleared plan as stale without dirtying the project', () => {
    const base = researchDeskReducer(createEmptyProjectState(), {
      summary,
      type: 'upsertAgentRunSummary',
    })
    // The 404 path clears the flag with no content (pre-planning).
    const cleared = researchDeskReducer(base, {
      plan: null,
      runId: 'r-plan',
      type: 'setAgentRunPlan',
    })
    expect(cleared.agentRuns['r-plan'].planStale).toBe(false)
    expect(cleared.agentRuns['r-plan'].plan).toBeUndefined()
    const reflagged = researchDeskReducer(cleared, {
      runId: 'r-plan',
      type: 'markAgentRunPlanStale',
    })
    expect(reflagged.agentRuns['r-plan'].planStale).toBe(true)
    expect(reflagged.dirty).toBe(false)
  })

  it('is an identity no-op while the flag is already set', () => {
    const base = researchDeskReducer(createEmptyProjectState(), {
      summary,
      type: 'upsertAgentRunSummary',
    })
    expect(base.agentRuns['r-plan'].planStale).toBe(true)
    const again = researchDeskReducer(base, {
      runId: 'r-plan',
      type: 'markAgentRunPlanStale',
    })
    expect(again).toBe(base)
  })
})

describe('gate-tray root fixes (P6): approvals reconcile + exclusive membership', () => {
  const agentSummary = (
    overrides: Partial<ResearchRunSummary> = {},
  ): ResearchRunSummary => ({
    access: { mode: 'owner' },
    run_id: 'r-gate',
    status: 'waiting_for_approval',
    queue_position: null,
    question: 'Gate-Lauf',
    stack: 'default',
    mode: 'workspace_agent',
    kind: 'agent',
    session_id: 's-gate',
    agent_overrides: {},
    created_at: 1_700_000_000,
    started_at: 1_700_000_000,
    finished_at: null,
    elapsed_seconds: null,
    snapshot: {},
    error: null,
    events_url: '/v1/runs/r-gate/events',
    result_url: '/v1/runs/r-gate/result',
    ...overrides,
  })
  const approvalWire = (status: 'pending' | 'approved') => ({
    approval_id: 'ap-1',
    run_id: 'r-gate',
    kind: 'plan' as const,
    status,
    subject_type: 'plan',
    subject_id: 'p1',
    payload: {},
    decision: status === 'approved' ? 'approve' : '',
    note: '',
    decided_by_user_id: null,
    created_at: 1_700_000_000,
    decided_at: status === 'approved' ? 1_700_000_100 : null,
  })

  it('never regresses a decided approval back to pending (stale refetch race)', () => {
    let state = researchDeskReducer(createEmptyProjectState(), {
      summary: agentSummary(),
      type: 'upsertAgentRunSummary',
    })
    state = researchDeskReducer(state, {
      approvals: [approvalWire('approved')],
      runId: 'r-gate',
      type: 'setAgentRunApprovals',
    })
    // The approval.decided event triggers a full-list refetch that can
    // observe the row still pending — it must not re-open the gate.
    state = researchDeskReducer(state, {
      approvals: [approvalWire('pending')],
      runId: 'r-gate',
      type: 'setAgentRunApprovals',
    })
    expect(state.agentRuns['r-gate'].approvals).toHaveLength(1)
    expect(state.agentRuns['r-gate'].approvals[0].status).toBe('approved')
  })

  it('never regresses an answered clarification back to pending', () => {
    const clarificationWire = (status: 'pending' | 'answered') => ({
      clarification_id: 'cl-1',
      run_id: 'r-gate',
      question: 'Welcher Markt?',
      options: [],
      default_assumption: '',
      status,
      answer: status === 'answered' ? 'DACH' : '',
      option_id: '',
      answered_by_user_id: null,
      created_at: 1_700_000_000,
      answered_at: status === 'answered' ? 1_700_000_050 : null,
    })
    let state = researchDeskReducer(createEmptyProjectState(), {
      summary: agentSummary(),
      type: 'upsertAgentRunSummary',
    })
    state = researchDeskReducer(state, {
      clarifications: [clarificationWire('answered')],
      runId: 'r-gate',
      type: 'setAgentRunClarifications',
    })
    // The clarification.answered event refetch can race the commit and
    // observe the round still pending — the gate must stay closed.
    state = researchDeskReducer(state, {
      clarifications: [clarificationWire('pending')],
      runId: 'r-gate',
      type: 'setAgentRunClarifications',
    })
    expect(state.agentRuns['r-gate'].clarifications).toHaveLength(1)
    expect(state.agentRuns['r-gate'].clarifications[0].status).toBe('answered')
  })

  it('sweeps a run out of its phantom session once the real session hydrates', () => {
    // First sighting without session_id: phantom session keyed by runId.
    let state = researchDeskReducer(createEmptyProjectState(), {
      select: true,
      summary: agentSummary({ session_id: undefined }),
      type: 'upsertAgentRunSummary',
    })
    expect(state.agentSessions['r-gate'].runIds).toEqual(['r-gate'])
    expect(state.selectedAgentSessionId).toBe('r-gate')
    // Later summary carries the real session: exclusive membership, the
    // emptied phantom shell disappears and selection follows the run.
    state = researchDeskReducer(state, {
      summary: agentSummary(),
      type: 'upsertAgentRunSummary',
    })
    expect(state.agentSessions['r-gate']).toBeUndefined()
    expect(state.agentSessionOrder).not.toContain('r-gate')
    expect(state.agentSessions['s-gate'].runIds).toEqual(['r-gate'])
    expect(state.selectedAgentSessionId).toBe('s-gate')
    expect(state.ui.selectedAgentSessionId).toBe('s-gate')
  })

  it('keeps an empty-string session_id from stomping the known session', () => {
    let state = researchDeskReducer(createEmptyProjectState(), {
      summary: agentSummary(),
      type: 'upsertAgentRunSummary',
    })
    state = researchDeskReducer(state, {
      summary: agentSummary({ session_id: '', status: 'running' }),
      type: 'upsertAgentRunSummary',
    })
    expect(state.agentRuns['r-gate'].sessionId).toBe('s-gate')
    expect(state.agentSessions['s-gate'].runIds).toEqual(['r-gate'])
    expect(state.agentSessions['r-gate']).toBeUndefined()
  })
})

describe('agent session selection mirrors into persisted ui intent', () => {
  it('select, create, run-summary select and delete all write ui.selectedAgentSessionId', () => {
    const base = {
      ...createEmptyProjectState(),
      agentSessionOrder: ['s1'],
      agentSessions: {
        s1: {
          id: 's1', title: 'A', groupId: null, createdAt: '', updatedAt: '', runIds: [],
          sourcePolicy: { web: 'available' as const, knowledge: 'available' as const },
        },
      },
    }
    const selected = researchDeskReducer(base, {
      sessionId: 's1',
      type: 'selectAgentSession',
    })
    expect(selected.ui.selectedAgentSessionId).toBe('s1')

    const created = researchDeskReducer(selected, {
      session: {
        id: 's2', title: 'B', groupId: null, createdAt: '', updatedAt: '', runIds: [],
        sourcePolicy: { web: 'available' as const, knowledge: 'available' as const },
      },
      type: 'createAgentSession',
    })
    expect(created.ui.selectedAgentSessionId).toBe('s2')

    const deleted = researchDeskReducer(created, {
      sessionId: 's2',
      type: 'deleteAgentSession',
    })
    expect(deleted.selectedAgentSessionId).toBe(deleted.ui.selectedAgentSessionId)
  })
})

describe('agent session source-policy hydration', () => {
  it('does not reset local policy when a metadata-only list row omits items_json', () => {
    const base = createEmptyProjectState()
    const withSession = researchDeskReducer(base, {
      session: {
        id: 's-source',
        title: 'Sources',
        groupId: null,
        createdAt: '2026-07-10T10:00:00.000Z',
        updatedAt: '2026-07-10T10:00:00.000Z',
        runIds: [],
        sourcePolicy: { web: 'disabled', knowledge: 'available' },
      },
      type: 'createAgentSession',
    })
    const metadataOnly = researchDeskReducer(withSession, {
      groups: [],
      sessions: [{
        id: 's-source',
        title: 'Sources',
        group_id: null,
        created_at: Date.parse('2026-07-10T10:00:00.000Z') / 1000,
        updated_at: Date.parse('2026-07-10T10:01:00.000Z') / 1000,
      }],
      type: 'upsertServerAgentSessions',
    })
    expect(metadataOnly.agentSessions['s-source'].sourcePolicy).toEqual({
      web: 'disabled',
      knowledge: 'available',
    })
  })
})

describe('agent model selection actions (R3)', () => {
  it('keeps the chat-picker exclusivity contract on the agent fields', () => {
    let state = createEmptyProjectState()
    state = researchDeskReducer(state, {
      model: 'claude-opus-4-8',
      type: 'setSelectedAgentModel',
    })
    state = researchDeskReducer(state, {
      effort: 'high',
      type: 'setSelectedAgentEffort',
    })
    expect(state.ui.selectedAgentModel).toBe('claude-opus-4-8')
    expect(state.ui.selectedAgentEffort).toBe('high')
    expect(state.ui.selectedAgentModelTier).toBeNull()
    // Tier pick clears model AND effort (effort is model-dependent).
    state = researchDeskReducer(state, {
      tier: 'mid',
      type: 'setSelectedAgentModelTier',
    })
    expect(state.ui.selectedAgentModelTier).toBe('mid')
    expect(state.ui.selectedAgentModel).toBeNull()
    expect(state.ui.selectedAgentEffort).toBeNull()
    // Model pick clears the tier again.
    state = researchDeskReducer(state, {
      model: 'claude-haiku-4-5',
      type: 'setSelectedAgentModel',
    })
    expect(state.ui.selectedAgentModelTier).toBeNull()
    // The AGENT selection never touches the chat fields.
    expect(state.ui.selectedChatModel).toBeNull()
    expect(state.ui.selectedChatModelTier).toBeNull()
  })
})
