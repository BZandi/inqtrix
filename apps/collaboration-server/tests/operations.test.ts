import { getSchema } from '@tiptap/core'
import { EditorState } from '@tiptap/pm/state'
import {
  createEditorSchemaExtensions,
  editorCollaborationRoom,
  editorJsonToYDoc,
  editorYDocToJson,
  parseEditorMarkdown,
  transformToInqtrixSuggestionTransaction,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import { DocumentCoordinator } from '../src/documentCoordinator'
import { hashString } from '../src/documentState'
import { ApiRequestError } from '../src/errors'
import { InstanceLeaseManager } from '../src/instanceLease'
import { SidecarMetrics } from '../src/metrics'
import { CollaborationOperations } from '../src/operations'
import { collectSuggestionRecords } from '../src/suggestPolicy'
import {
  FakeCollaborationApi,
  USER_ID,
  documentState,
  settings,
  silentLogger,
} from './helpers'

const DOCUMENT_ID = 'ed_test'
const ROOM = editorCollaborationRoom(DOCUMENT_ID, 1)
const PATCH_ID = '22222222-2222-4222-8222-222222222222'
const SUGGESTION_ID = '33333333-3333-4333-8333-333333333333'
const COMMAND_ID = '44444444-4444-4444-8444-444444444444'
const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))

describe('FastAPI-to-Node operations', () => {
  it('persists an accept decision with CAS, decision metadata, and the closed patch state', async () => {
    const plain = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: plain })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', 6),
      state,
      { authorId: USER_ID, createdAt: 1_784_112_000, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc
    const document = editorJsonToYDoc(tracked.toJSON())
    const fixture = await operationsFixture(document, 5)

    const response = await fixture.operations.decide(DOCUMENT_ID, {
      actor_user_id: USER_ID,
      command_id: COMMAND_ID,
      decision: 'accept',
      expected_sequence: 5,
      generation: 1,
      patch_ids: [PATCH_ID],
    })

    expect(response).toEqual({
      command_id: COMMAND_ID,
      decision: 'accept',
      patch_ids: [PATCH_ID],
      sequence: 6,
      suggestion_ids: [SUGGESTION_ID],
    })
    expect(fixture.api.persisted[0]).toMatchObject({
      actorKind: 'human',
      changeKind: 'decision',
      commandId: COMMAND_ID,
      decision: 'accept',
      expectedSequence: 5,
      patches: [{
        activeSuggestionIds: [],
        authorId: USER_ID,
        createdAt: 1_784_112_000,
        kinds: [],
        patchId: PATCH_ID,
      }],
      suggestionIds: [SUGGESTION_ID],
    })
    await expect(fixture.operations.decide(DOCUMENT_ID, {
      actor_user_id: USER_ID,
      command_id: COMMAND_ID,
      decision: 'accept',
      expected_sequence: 5,
      generation: 1,
      patch_ids: [PATCH_ID],
    })).resolves.toEqual(response)
    expect(fixture.api.persisted).toHaveLength(1)
    await expect(fixture.operations.decide(DOCUMENT_ID, {
      actor_user_id: USER_ID,
      command_id: COMMAND_ID,
      decision: 'reject',
      expected_sequence: 5,
      generation: 1,
      patch_ids: [PATCH_ID],
    })).rejects.toThrowError('sequence_conflict')
    await fixture.close()
  })

  it('reconciles a committed decision retry after the persistence response is lost', async () => {
    const plain = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: plain })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', 6),
      state,
      { authorId: USER_ID, createdAt: 1_784_112_000, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc
    const document = editorJsonToYDoc(tracked.toJSON())
    const fixture = await operationsFixture(document, 5)
    fixture.api.persistResponseErrorAfterCommit = new ApiRequestError(
      503,
      'internal_api_timeout',
    )
    const request = {
      actor_user_id: USER_ID,
      command_id: COMMAND_ID,
      decision: 'accept',
      expected_sequence: 5,
      generation: 1,
      patch_ids: [PATCH_ID],
    }

    await expect(fixture.operations.decide(DOCUMENT_ID, request))
      .rejects.toThrowError('service_unavailable')
    expect(collectSuggestionRecords(editorYDocToJson(document)).size).toBe(1)
    expect(fixture.coordinator.requiresReconstruction(ROOM)).toBe(true)
    expect(fixture.coordinator.shouldSnapshot(ROOM)).toBe(false)
    expect(() => fixture.coordinator.assertJoinAllowed(ROOM)).toThrowError('restarting')

    await expect(fixture.operations.decide(DOCUMENT_ID, request)).resolves.toEqual({
      command_id: COMMAND_ID,
      decision: 'accept',
      patch_ids: [PATCH_ID],
      sequence: 6,
      suggestion_ids: [SUGGESTION_ID],
    })
    expect(fixture.api.persisted).toHaveLength(1)
    expect(collectSuggestionRecords(editorYDocToJson(document)).size).toBe(0)
    expect(fixture.coordinator.getPersistedSequence(ROOM)).toBe(6)
    await fixture.close()
  })

  it('replays an accepted text modification idempotently as one decision', async () => {
    const plain = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: plain })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('World', 1, 6),
      state,
      { authorId: USER_ID, createdAt: 1_784_112_000, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc
    const document = editorJsonToYDoc(tracked.toJSON())
    const fixture = await operationsFixture(document, 5)
    const request = {
      actor_user_id: USER_ID,
      command_id: COMMAND_ID,
      decision: 'accept',
      expected_sequence: 5,
      generation: 1,
      patch_ids: [PATCH_ID],
    }

    const first = await fixture.operations.decide(DOCUMENT_ID, request)
    await expect(fixture.operations.decide(DOCUMENT_ID, request)).resolves.toEqual(first)

    expect(first).toMatchObject({
      decision: 'accept',
      sequence: 6,
      suggestion_ids: [SUGGESTION_ID],
    })
    expect(fixture.api.persisted).toHaveLength(1)
    expect(collectSuggestionRecords(editorYDocToJson(document)).size).toBe(0)
    await fixture.close()
  })

  it('publishes target Markdown as an assistant suggestion in the same durable flow', async () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    const fixture = await operationsFixture(document, 8)

    const response = await fixture.operations.publishSuggestion(DOCUMENT_ID, {
      actor_kind: 'assistant',
      actor_user_id: USER_ID,
      command_id: COMMAND_ID,
      expected_sequence: 8,
      generation: 1,
      patch_id: PATCH_ID,
      target_markdown: 'Hello world',
    })

    expect(response).toMatchObject({
      command_id: COMMAND_ID,
      patch_id: PATCH_ID,
      sequence: 9,
      suggestion_ids: [expect.stringMatching(/^[0-9a-f-]{36}$/)],
    })
    expect(fixture.api.persisted[0]).toMatchObject({
      actorKind: 'assistant',
      actorUserId: USER_ID,
      changeKind: 'suggestion',
      commandId: COMMAND_ID,
      decision: null,
      expectedSequence: 8,
      patches: [{
        activeSuggestionIds: [expect.stringMatching(/^[0-9a-f-]{36}$/)],
        authorId: USER_ID,
        kinds: ['insertion'],
        patchId: PATCH_ID,
      }],
    })
    await expect(fixture.operations.publishSuggestion(DOCUMENT_ID, {
      actor_kind: 'assistant',
      actor_user_id: USER_ID,
      command_id: COMMAND_ID,
      expected_sequence: 8,
      generation: 1,
      patch_id: PATCH_ID,
      target_markdown: 'Hello world',
    })).resolves.toEqual(response)
    expect(fixture.api.persisted).toHaveLength(1)
    await fixture.close()
  })

  it('reconciles a committed suggestion publication after the response is lost', async () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    const fixture = await operationsFixture(document, 8)
    fixture.api.persistResponseErrorAfterCommit = new ApiRequestError(
      503,
      'internal_api_timeout',
    )
    const request = {
      actor_kind: 'assistant',
      actor_user_id: USER_ID,
      command_id: COMMAND_ID,
      expected_sequence: 8,
      generation: 1,
      patch_id: PATCH_ID,
      target_markdown: 'Hello world',
    }

    await expect(fixture.operations.publishSuggestion(DOCUMENT_ID, request))
      .rejects.toThrowError('service_unavailable')
    expect(collectSuggestionRecords(editorYDocToJson(document)).size).toBe(0)
    expect(fixture.coordinator.requiresReconstruction(ROOM)).toBe(true)
    expect(fixture.coordinator.shouldSnapshot(ROOM)).toBe(false)

    const replay = await fixture.operations.publishSuggestion(DOCUMENT_ID, request)
    expect(replay).toMatchObject({
      command_id: COMMAND_ID,
      patch_id: PATCH_ID,
      sequence: 9,
      suggestion_ids: [expect.stringMatching(/^[0-9a-f-]{36}$/)],
    })
    expect(fixture.api.persisted).toHaveLength(1)
    expect(collectSuggestionRecords(editorYDocToJson(document)).size).toBe(1)
    expect(fixture.coordinator.getPersistedSequence(ROOM)).toBe(9)
    await fixture.close()
  })

  it('returns conversion and projection fields expected by the Python client', async () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('# Shared'))
    const fixture = await operationsFixture(document, 3)

    const conversion = await fixture.operations.convert({
      document_id: DOCUMENT_ID,
      markdown: '# Shared',
      max_document_bytes: 10 * 1024 * 1024,
      schema_version: 2,
    })
    expect(conversion).toMatchObject({
      projection_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      projection_markdown: expect.any(String),
      schema_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      schema_version: 2,
      snapshot: {
        covered_sequence: 0,
        state_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
        state_update_base64: expect.any(String),
        state_vector_base64: expect.any(String),
      },
    })

    const projection = await fixture.operations.project(DOCUMENT_ID, {
      generation: 1,
      include_snapshot: true,
      minimum_sequence: 3,
    })
    expect(projection).toMatchObject({
      generation: 1,
      projection_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      projection_markdown: expect.any(String),
      schema_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      schema_version: 2,
      sequence: 3,
      snapshot: {
        covered_sequence: 3,
        state_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
        state_update_base64: expect.any(String),
        state_vector_base64: expect.any(String),
      },
    })
    await fixture.close()
  })

  it('converts an empty editor document to canonical collaboration state', async () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Seed'))
    const fixture = await operationsFixture(document, 0)

    const conversion = await fixture.operations.convert({
      document_id: DOCUMENT_ID,
      markdown: '',
      max_document_bytes: 10 * 1024 * 1024,
      schema_version: 2,
    })

    expect(conversion).toMatchObject({
      projection_markdown: '',
      schema_version: 2,
      snapshot: {
        covered_sequence: 0,
        state_update_base64: expect.any(String),
      },
    })
    await fixture.close()
  })

  it('stores projection with the snapshot and reports compaction failure without rollback', async () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('# Shared'))
    const fixture = await operationsFixture(document, 3)
    fixture.api.compactImplementation = async () => {
      throw new Error('maintenance unavailable')
    }

    await expect(fixture.operations.storeSnapshot(ROOM, document)).resolves.toBe(true)
    expect(fixture.api.snapshots).toHaveLength(1)
    expect(fixture.api.snapshots[0]).toMatchObject({
      coveredSequence: 3,
      documentId: DOCUMENT_ID,
      generation: 1,
      projectionHash: expect.stringMatching(/^[0-9a-f]{64}$/),
      projectionMarkdown: expect.stringContaining('# Shared'),
    })
    expect(fixture.api.snapshots[0]?.projectionHash).toBe(
      hashString(fixture.api.snapshots[0]?.projectionMarkdown ?? ''),
    )
    expect(fixture.api.compactions).toEqual([{
      documentId: DOCUMENT_ID,
      fence: fixture.api.fence,
      generation: 1,
    }])
    expect(fixture.metrics.render()).toContain(
      'inqtrix_collaboration_compaction_runs_total{status="failure"} 1',
    )
    expect(fixture.warnings).toContain('collaboration_compaction_failed')
    await fixture.close()
  })

  it('runs global maintenance independently of document snapshots', async () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('# Shared'))
    const fixture = await operationsFixture(document, 3)

    await fixture.operations.runMaintenance()

    expect(fixture.api.snapshots).toHaveLength(0)
    expect(fixture.api.compactions).toEqual([{ fence: fixture.api.fence }])
    expect(fixture.metrics.render()).toContain(
      'inqtrix_collaboration_compaction_runs_total{status="success"} 1',
    )
    await fixture.close()
  })
})

async function operationsFixture(document: Y.Doc, sequence: number): Promise<{
  api: FakeCollaborationApi
  close: () => Promise<void>
  coordinator: DocumentCoordinator
  metrics: SidecarMetrics
  operations: CollaborationOperations
  warnings: string[]
}> {
  const api = new FakeCollaborationApi()
  const configured = settings()
  api.persistImplementation = async () => ({
    duplicate: false,
    persistedSequence: sequence + 1,
    sequence: sequence + 1,
  })
  const metrics = new SidecarMetrics()
  const warnings: string[] = []
  const logger = {
    ...silentLogger,
    warn: (event: string): void => {
      warnings.push(event)
    },
  }
  api.loadedState = await documentState(DOCUMENT_ID, document, 1, sequence)
  const lease = new InstanceLeaseManager(api, configured, silentLogger, metrics, () => undefined)
  await lease.start()
  const coordinator = new DocumentCoordinator(api, lease, configured, silentLogger, metrics)
  coordinator.initialize(ROOM, sequence)
  const operations = new CollaborationOperations(
    api,
    coordinator,
    lease,
    configured,
    async (room) => {
      expect(room).toBe(ROOM)
      return { document, release: async () => undefined }
    },
    logger,
    metrics,
  )
  return {
    api,
    close: () => lease.stop(),
    coordinator,
    metrics,
    operations,
    warnings,
  }
}
