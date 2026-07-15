import { getSchema } from '@tiptap/core'
import { EditorState } from '@tiptap/pm/state'
import { initProseMirrorDoc, updateYFragment } from '@tiptap/y-tiptap'
import {
  EDITOR_YJS_FRAGMENT,
  createEditorSchemaExtensions,
  editorCollaborationRoom,
  editorJsonToYDoc,
  editorYDocToJson,
  getEditorSchemaFingerprint,
  parseEditorMarkdown,
  serializeEditorJson,
  transformToInqtrixSuggestionTransaction,
  validateEditorYDoc,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import type { ConnectionContext } from '../src/contracts'
import { DocumentCoordinator } from '../src/documentCoordinator'
import { hashBytes, reconstructDocument } from '../src/documentState'
import { InstanceLeaseManager } from '../src/instanceLease'
import { SidecarMetrics } from '../src/metrics'
import {
  FakeCollaborationApi,
  USER_ID,
  deferred,
  documentState,
  settings,
  silentLogger,
} from './helpers'

const DOCUMENT_ID = 'ed_test'
const ROOM = editorCollaborationRoom(DOCUMENT_ID, 1)
const PATCH_ID = '22222222-2222-4222-8222-222222222222'
const SUGGESTION_ID = '33333333-3333-4333-8333-333333333333'
const EXISTING_PATCH_ID = '55555555-5555-4555-8555-555555555555'
const EXISTING_SUGGESTION_ID = '66666666-6666-4666-8666-666666666666'
const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))

describe('document coordinator', () => {
  it('persists an update before the authoritative document is applied and acknowledged', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    const update = updateFor(document, 'Hello!')
    const persistence = deferred<{ duplicate: boolean; persistedSequence: number; sequence: number }>()
    fixture.api.persistImplementation = () => persistence.promise

    const preparing = fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'connection-1',
      context: fixture.context,
      document,
      room: ROOM,
      update,
    })
    await vi.waitFor(() => expect(fixture.api.persisted).toHaveLength(1))
    expect(markdown(document)).toBe('Hello')

    persistence.resolve({ duplicate: false, persistedSequence: 1, sequence: 1 })
    await expect(preparing).resolves.toBe('pending')
    expect(markdown(document)).toBe('Hello')
    expect(fixture.coordinator.getPersistedSequence(ROOM)).toBe(0)
    expect(fixture.coordinator.shouldSnapshot(ROOM)).toBe(false)
    expect(() => fixture.coordinator.assertJoinAllowed(ROOM)).toThrowError('restarting')
    Y.applyUpdate(document, update)
    expect(fixture.coordinator.finishClientUpdate('connection-1', document)).toEqual({
      hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      sequence: 1,
      type: 'durable_ack',
    })
    expect(fixture.coordinator.getPersistedSequence(ROOM)).toBe(1)
    expect(fixture.api.persisted[0]).toMatchObject({
      changeKind: 'direct',
      decision: null,
      patches: [],
      suggestions: [],
    })
    expect(fixture.api.persisted[0]?.update).toBe(update)
    await fixture.close()
  })

  it('keeps the per-document gate held until authoritative apply completes', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('A'))
    const first = updateFor(document, 'AB')
    const replicaAfterFirst = clone(document)
    Y.applyUpdate(replicaAfterFirst, first)
    const second = updateFor(replicaAfterFirst, 'ABC')
    replicaAfterFirst.destroy()
    const firstPersistence = deferred<{
      duplicate: boolean
      persistedSequence: number
      sequence: number
    }>()
    fixture.api.persistImplementation = async () => {
      if (fixture.api.persisted.length === 1) return firstPersistence.promise
      return { duplicate: false, persistedSequence: 2, sequence: 2 }
    }

    const firstPreparing = fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'connection-1',
      context: fixture.context,
      document,
      room: ROOM,
      update: first,
    })
    await vi.waitFor(() => expect(fixture.api.persisted).toHaveLength(1))
    const secondPreparing = fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'connection-2',
      context: fixture.context,
      document,
      room: ROOM,
      update: second,
    })
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(fixture.api.persisted).toHaveLength(1)

    firstPersistence.resolve({ duplicate: false, persistedSequence: 1, sequence: 1 })
    await firstPreparing
    Y.applyUpdate(document, first)
    fixture.coordinator.finishClientUpdate('connection-1', document)
    await expect(secondPreparing).resolves.toBe('pending')
    expect(fixture.api.persisted).toHaveLength(2)
    Y.applyUpdate(document, second)
    fixture.coordinator.finishClientUpdate('connection-2', document)
    expect(markdown(document)).toBe('ABC')
    await fixture.close()
  })

  it('releases a persisted pending update when disconnect forces an authoritative reload', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    const update = updateFor(document, 'Hello!')

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'disconnected-connection',
      context: fixture.context,
      document,
      room: ROOM,
      update,
    })).resolves.toBe('pending')

    expect(fixture.coordinator.abortClientUpdate('disconnected-connection')).toBe(ROOM)
    expect(fixture.coordinator.requiresReconstruction(ROOM)).toBe(true)
    expect(fixture.coordinator.shouldSnapshot(ROOM)).toBe(false)
    expect(() => fixture.coordinator.assertJoinAllowed(ROOM)).toThrowError('restarting')
    await expect(fixture.coordinator.awaitRoom(ROOM)).resolves.toBeUndefined()
    expect(fixture.coordinator.finishClientUpdate('disconnected-connection', document)).toBeNull()
    await expect(fixture.coordinator.captureDocument(ROOM, document)).rejects.toThrowError(
      'internal_consistency',
    )
    fixture.coordinator.markUnloaded(ROOM)
    expect(() => fixture.coordinator.assertJoinAllowed(ROOM)).not.toThrow()
    document.destroy()
    await fixture.close()
  })

  it('releases the gate when disconnect occurs while persistence is in flight', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    const update = updateFor(document, 'Hello!')
    const persistence = deferred<{
      duplicate: boolean
      persistedSequence: number
      sequence: number
    }>()
    fixture.api.persistImplementation = () => persistence.promise

    const preparing = fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'in-flight-disconnect',
      context: fixture.context,
      document,
      room: ROOM,
      update,
    })
    await vi.waitFor(() => expect(fixture.api.persisted).toHaveLength(1))

    expect(fixture.coordinator.abortClientUpdate('in-flight-disconnect')).toBe(ROOM)
    persistence.resolve({ duplicate: false, persistedSequence: 1, sequence: 1 })

    await expect(preparing).rejects.toThrowError('invalid_lease')
    expect(fixture.coordinator.requiresReconstruction(ROOM)).toBe(true)
    expect(fixture.coordinator.shouldSnapshot(ROOM)).toBe(false)
    await expect(fixture.coordinator.awaitRoom(ROOM)).resolves.toBeUndefined()
    expect(fixture.coordinator.finishClientUpdate('in-flight-disconnect', document)).toBeNull()
    document.destroy()
    await fixture.close()
  })

  it('prohibits snapshots and fast joins until a disconnected commit is reconstructed', async () => {
    const fixture = await coordinatorFixture(0, { snapshotMaxUpdates: 1 })
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    fixture.api.loadedState = await documentState(DOCUMENT_ID, document)
    const update = updateFor(document, 'Hello!')
    const persistence = deferred<{
      duplicate: boolean
      persistedSequence: number
      sequence: number
    }>()
    fixture.api.persistImplementation = () => persistence.promise

    const preparing = fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'disconnect-before-apply',
      context: fixture.context,
      document,
      room: ROOM,
      update,
    })
    await vi.waitFor(() => expect(fixture.api.persisted).toHaveLength(1))
    expect(fixture.coordinator.abortClientUpdate('disconnect-before-apply')).toBe(ROOM)
    expect(fixture.coordinator.shouldSnapshot(ROOM)).toBe(false)
    expect(() => fixture.coordinator.assertJoinAllowed(ROOM)).toThrowError('restarting')

    persistence.resolve({ duplicate: false, persistedSequence: 1, sequence: 1 })
    await expect(preparing).rejects.toThrowError('invalid_lease')
    await fixture.coordinator.awaitRoom(ROOM)
    expect(fixture.coordinator.shouldSnapshot(ROOM)).toBe(false)

    fixture.coordinator.markUnloaded(ROOM)
    expect(() => fixture.coordinator.assertJoinAllowed(ROOM)).not.toThrow()
    const loaded = await fixture.api.loadDocumentState()
    const reconstructed = await reconstructDocument(loaded, {
      documentId: DOCUMENT_ID,
      generation: 1,
      schemaVersion: fixture.context.schemaVersion,
    }, settings().documentLimitBytes)
    fixture.coordinator.initialize(ROOM, loaded.persistedSequence, {
      bytes: loaded.updates.reduce((total, item) => total + item.update.byteLength, 0),
      updates: loaded.updates.length,
    })
    expect(markdown(reconstructed)).toBe('Hello!')
    expect(fixture.coordinator.getPersistedSequence(ROOM)).toBe(1)
    expect(fixture.coordinator.shouldSnapshot(ROOM)).toBe(true)

    reconstructed.destroy()
    document.destroy()
    await fixture.close()
  })

  it('requires reload when persistence may commit before returning an error', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    const update = updateFor(document, 'Hello!')
    fixture.api.persistResponseErrorAfterCommit = new Error('response lost')

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'lost-response',
      context: fixture.context,
      document,
      room: ROOM,
      update,
    })).rejects.toThrowError('internal_consistency')

    expect(fixture.coordinator.requiresReconstruction(ROOM)).toBe(true)
    expect(fixture.coordinator.shouldSnapshot(ROOM)).toBe(false)
    expect(() => fixture.coordinator.assertJoinAllowed(ROOM)).toThrowError('restarting')
    await expect(fixture.coordinator.awaitRoom(ROOM)).resolves.toBeUndefined()
    document.destroy()
    await fixture.close()
  })

  it('returns an older original sequence when a duplicate replay is current globally', async () => {
    const fixture = await coordinatorFixture(5)
    const original = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    const update = suggestionUpdate(original)
    Y.applyUpdate(original, update)
    fixture.api.lookupResults = [{ hash: hashBytes(update), sequence: 2 }]

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'duplicate',
      context: { ...fixture.context, access: 'suggest' },
      document: original,
      room: ROOM,
      update,
    })).resolves.toBe('pending')
    Y.applyUpdate(original, update)
    expect(fixture.coordinator.finishClientUpdate('duplicate', original)?.sequence).toBe(2)
    expect(fixture.coordinator.getPersistedSequence(ROOM)).toBe(5)
    expect(fixture.coordinator.requiresReconstruction(ROOM)).toBe(false)
    expect(() => fixture.coordinator.assertJoinAllowed(ROOM)).not.toThrow()
    expect(fixture.api.lookups).toHaveLength(1)
    expect(fixture.api.persisted).toHaveLength(0)
    await fixture.close()
  })

  it('reconciles a known direct-edit replay from suggest access after generic validation', async () => {
    const fixture = await coordinatorFixture(3)
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    const update = updateFor(document, 'Hello!')
    Y.applyUpdate(document, update)
    fixture.api.lookupResults = [{ hash: hashBytes(update), sequence: 2 }]

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'known-direct-replay',
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update,
    })).resolves.toBe('pending')
    expect(fixture.coordinator.finishClientUpdate('known-direct-replay', document)).toMatchObject({
      sequence: 2,
      type: 'durable_ack',
    })
    expect(fixture.api.lookups).toHaveLength(1)
    expect(fixture.api.persisted).toHaveLength(0)

    document.destroy()
    await fixture.close()
  })

  it('quarantines a duplicate lookup whose durable sequence is ahead of the room', async () => {
    const fixture = await coordinatorFixture(5)
    const original = editorJsonToYDoc(parseEditorMarkdown('Hello'))
    const update = updateFor(original, 'Hello!')
    Y.applyUpdate(original, update)
    fixture.api.lookupResults = [{ hash: hashBytes(update), sequence: 6 }]

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'inconsistent-duplicate',
      context: { ...fixture.context, access: 'suggest' },
      document: original,
      room: ROOM,
      update,
    })).rejects.toThrowError('internal_consistency')

    expect(fixture.coordinator.requiresReconstruction(ROOM)).toBe(true)
    expect(() => fixture.coordinator.assertJoinAllowed(ROOM)).toThrowError('restarting')
    await fixture.close()
  })

  it('still rejects a genuinely new direct edit from a suggest-only lease', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'new-direct-suggest-only',
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update: updateFor(document, 'Hello!'),
    })).rejects.toThrowError('suggestion_policy_violation')
    expect(fixture.api.lookups).toHaveLength(0)
    expect(fixture.api.persisted).toHaveLength(0)
    document.destroy()
    await fixture.close()
  })

  it('reports stale snapshot coverage without clearing the current dirty tail', async () => {
    const fixture = await coordinatorFixture(2)
    fixture.coordinator.initialize(ROOM, 2, { bytes: 16, updates: 2 })

    expect(fixture.coordinator.markSnapshot(ROOM, 1)).toBe(false)
    expect(fixture.coordinator.hasUnsnapshottedUpdates(ROOM)).toBe(true)
    expect(fixture.coordinator.markSnapshot(ROOM, 2)).toBe(true)
    expect(fixture.coordinator.hasUnsnapshottedUpdates(ROOM)).toBe(false)

    await fixture.close()
  })

  it('rejects oversized and malformed Yjs updates before persistence', async () => {
    const fixture = await coordinatorFixture(0, { frameLimitBytes: 4 })
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'too-large',
      context: fixture.context,
      document,
      room: ROOM,
      update: new Uint8Array(5),
    })).rejects.toThrowError('message_too_large')
    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'malformed',
      context: fixture.context,
      document,
      room: ROOM,
      update: new Uint8Array([1, 2, 3]),
    })).rejects.toThrowError('invalid_schema')
    expect(fixture.api.persisted).toHaveLength(0)
    await fixture.close()
  })

  it.each([
    ['suffix bytes', 'suffix', 'edit'],
    ['suffix bytes', 'suffix', 'suggest'],
    ['V2-as-V1 payload', 'v2', 'edit'],
    ['V2-as-V1 payload', 'v2', 'suggest'],
    ['redundant old plus novel merge', 'merged', 'edit'],
    ['redundant old plus novel merge', 'merged', 'suggest'],
    ['redundant delete-set plus novel merge', 'merged-delete', 'edit'],
    ['redundant delete-set plus novel merge', 'merged-delete', 'suggest'],
  ] as const)('rejects %s (%s) for %s access before role policy', async (
    _label,
    attack,
    access,
  ) => {
    const fixture = await coordinatorFixture()
    const { document, update } = genericCanonicalityAttack(attack)

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: `canonical-${attack}-${access}`,
      context: { ...fixture.context, access },
      document,
      room: ROOM,
      update,
    })).rejects.toMatchObject({ code: 4409, reason: 'invalid_schema' })
    expect(fixture.api.lookups).toHaveLength(0)
    expect(fixture.api.persisted).toHaveLength(0)

    document.destroy()
    await fixture.close()
  })

  it.each(['edit', 'suggest'] as const)(
    'repeatedly rejects near-frame-limit suffix bloat for %s access',
    async (access) => {
      const configured = settings()
      const fixture = await coordinatorFixture()
      const document = editorJsonToYDoc(parseEditorMarkdown('Bloat'))
      const novel = updateFor(document, 'Bloat!')
      const bloated = new Uint8Array(configured.frameLimitBytes)
      bloated.set(novel)

      for (let attempt = 0; attempt < 3; attempt += 1) {
        await expect(fixture.coordinator.prepareClientUpdate({
          allowNoop: false,
          connectionId: `bloat-${access}-${attempt}`,
          context: { ...fixture.context, access },
          document,
          room: ROOM,
          update: bloated,
        })).rejects.toMatchObject({ code: 4409, reason: 'invalid_schema' })
      }
      expect(fixture.api.lookups).toHaveLength(0)
      expect(fixture.api.persisted).toHaveLength(0)

      document.destroy()
      await fixture.close()
    },
  )

  it.each(['edit', 'suggest'] as const)(
    'rejects an unknown byte-identical redundant suggestion update for %s access',
    async (access) => {
      const fixture = await coordinatorFixture()
      const document = editorJsonToYDoc(parseEditorMarkdown('Duplicate'))
      const update = suggestionUpdate(document)
      Y.applyUpdate(document, update)

      await expect(fixture.coordinator.prepareClientUpdate({
        allowNoop: false,
        connectionId: `unknown-redundant-${access}`,
        context: { ...fixture.context, access },
        document,
        room: ROOM,
        update,
      })).rejects.toMatchObject({ code: 4409, reason: 'invalid_schema' })
      expect(fixture.api.lookups).toHaveLength(1)
      expect(fixture.api.persisted).toHaveLength(0)

      document.destroy()
      await fixture.close()
    },
  )

  it.each(['edit', 'suggest'] as const)(
    'rejects an unknown canonical empty Type-2 update for %s access',
    async (access) => {
      const fixture = await coordinatorFixture()
      const document = editorJsonToYDoc(parseEditorMarkdown('Unknown no-op'))

      await expect(fixture.coordinator.prepareClientUpdate({
        allowNoop: false,
        connectionId: `unknown-empty-${access}`,
        context: { ...fixture.context, access },
        document,
        room: ROOM,
        update: new Uint8Array([0, 0]),
      })).rejects.toMatchObject({ code: 4409, reason: 'invalid_schema' })
      expect(fixture.api.lookups).toHaveLength(1)
      expect(fixture.api.persisted).toHaveLength(0)

      document.destroy()
      await fixture.close()
    },
  )

  it('allows only the canonical empty V1 update as a protocol sync no-op', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('Sync'))

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: true,
      connectionId: 'canonical-empty-sync',
      context: fixture.context,
      document,
      room: ROOM,
      update: new Uint8Array([0, 0]),
    })).resolves.toBe('noop')
    expect(fixture.api.lookups).toHaveLength(0)
    expect(fixture.api.persisted).toHaveLength(0)

    document.destroy()
    await fixture.close()
  })

  it.each([
    ['top-level subtree replacement', 'abc', [0]],
    ['nested subtree replacement rebased from a replica', '> abc', [0, 0]],
  ])('rejects a same-JSON suggest-mode %s that invalidates anchors', async (
    _label,
    markdownSource,
    path,
  ) => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown(markdownSource))
    const attack = topologyReplacementUpdate(document, path)
    const attacked = clone(document)
    Y.applyUpdate(attacked, attack.update)

    expect(editorYDocToJson(attacked)).toEqual(editorYDocToJson(document))
    expect(Y.createAbsolutePositionFromRelativePosition(
      attack.anchor,
      attacked,
    )).toBeNull()
    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: true,
      connectionId: `topology-${path.join('-')}`,
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update: attack.update,
    })).rejects.toThrowError('suggestion_policy_violation')
    expect(fixture.api.persisted).toHaveLength(0)

    attacked.destroy()
    document.destroy()
    await fixture.close()
  })

  it.each([
    ['huge XmlElement attribute', 'element-attribute'],
    ['huge XmlText attribute', 'text-attribute'],
    ['insert-delete text', 'text-content'],
    ['format churn', 'text-format'],
  ] as const)('rejects a valid suggestion bundled with transient %s history', async (
    _label,
    hiddenHistory,
  ) => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('First\n\nSecond'))
    const update = suggestionWithTransientYjsHistoryUpdate(document, hiddenHistory)
    const attacked = clone(document)
    Y.applyUpdate(attacked, update)

    expect(JSON.stringify(validateEditorYDoc(attacked))).toContain(SUGGESTION_ID)
    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: `transient-${hiddenHistory}`,
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update,
    })).rejects.toMatchObject({
      code: 4403,
      reason: 'suggestion_policy_violation',
    })
    expect(fixture.api.lookups).toHaveLength(0)
    expect(fixture.api.persisted).toHaveLength(0)

    attacked.destroy()
    document.destroy()
    await fixture.close()
  })

  it('rejects a valid suggestion bundled with an unknown top-level Yjs type', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('First\n\nSecond'))
    const update = suggestionWithTransientYjsHistoryUpdate(document, 'unknown-root')
    const attacked = clone(document)
    Y.applyUpdate(attacked, update)

    expect([...attacked.share.keys()]).toContain('rogue')
    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'unknown-root-history',
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update,
    })).rejects.toMatchObject({ code: 4409, reason: 'invalid_schema' })
    expect(fixture.api.lookups).toHaveLength(0)
    expect(fixture.api.persisted).toHaveLength(0)

    attacked.destroy()
    document.destroy()
    await fixture.close()
  })

  it.each([
    ['missing struct dependency', 'struct'],
    ['pending delete dependency', 'delete'],
  ] as const)('rejects a valid suggestion bundled with a %s', async (_label, dependency) => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('First\n\nSecond'))
    const update = suggestionWithMissingCausalUpdate(document, dependency)
    const attacked = clone(document)
    Y.applyUpdate(attacked, update)

    expect(() => validateEditorYDoc(attacked)).toThrow(/unresolved/)
    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: `causal-${dependency}`,
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update,
    })).rejects.toMatchObject({ code: 4409, reason: 'invalid_schema' })
    expect(fixture.api.lookups).toHaveLength(0)
    expect(fixture.api.persisted).toHaveLength(0)

    attacked.destroy()
    document.destroy()
    await fixture.close()
  })

  it.each([
    ['XmlElement attribute', 'element-attribute'],
    ['XmlText attribute', 'text-attribute'],
    ['empty XmlText topology', 'empty-text'],
  ] as const)('rejects a legitimate suggestion bundled with hidden %s', async (
    _label,
    hiddenState,
  ) => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('First\n\nSecond'))
    const update = suggestionWithHiddenYjsStateUpdate(document, hiddenState)
    const attacked = clone(document)
    Y.applyUpdate(attacked, update)

    expect(JSON.stringify(editorYDocToJson(attacked))).toContain(SUGGESTION_ID)
    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: `hidden-${hiddenState}`,
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update,
    })).rejects.toThrowError('invalid_schema')
    expect(fixture.api.persisted).toHaveLength(0)

    attacked.destroy()
    document.destroy()
    await fixture.close()
  })

  it('rejects cloning an unchanged insertion text beside a legitimate new suggestion', async () => {
    const fixture = await coordinatorFixture()
    const plain = schema.node('doc', null, [
      schema.node('paragraph', null, schema.text('First')),
      schema.node('paragraph'),
    ])
    const state = EditorState.create({ schema, doc: plain })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('X', state.doc.child(0).nodeSize + 1),
      state,
      {
        authorId: USER_ID,
        createdAt: 1_784_111_000,
        patchId: EXISTING_PATCH_ID,
      },
      () => EXISTING_SUGGESTION_ID,
    ).doc
    const document = editorJsonToYDoc(tracked.toJSON())
    const sourceText = firstXmlText(xmlElementAt(
      document.getXmlFragment(EDITOR_YJS_FRAGMENT),
      [1],
    ))
    if (!sourceText) throw new Error('Existing suggestion fixture has no text anchor')
    const anchor = Y.createRelativePositionFromTypeIndex(sourceText, 0)

    const replica = clone(document)
    const vector = Y.encodeStateVector(replica)
    applySuggestion(replica)
    replaceFirstXmlText(replica.getXmlFragment(EDITOR_YJS_FRAGMENT), [1])
    const update = Y.encodeStateAsUpdate(replica, vector)
    const attacked = clone(document)
    Y.applyUpdate(attacked, update)

    expect(JSON.stringify(editorYDocToJson(attacked))).toContain(SUGGESTION_ID)
    expect(JSON.stringify(editorYDocToJson(attacked))).toContain(EXISTING_SUGGESTION_ID)
    expect(Y.createAbsolutePositionFromRelativePosition(anchor, attacked)).toBeNull()
    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'combined-insertion-text-topology',
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update,
    })).rejects.toThrowError('suggestion_policy_violation')
    expect(fixture.api.persisted).toHaveLength(0)

    replica.destroy()
    attacked.destroy()
    document.destroy()
    await fixture.close()
  })

  it.each([
    ['top-level subtree replacement', 'First\n\nSecond', [1]],
    ['nested subtree replacement', 'First\n\n> Second', [1, 0]],
    ['unrelated relative-position invalidation', 'First\n\nabcabc', [1]],
  ])('rejects a legitimate suggestion bundled with %s', async (
    _label,
    markdownSource,
    path,
  ) => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown(markdownSource))
    const attack = suggestionWithTopologyReplacementUpdate(document, path)
    const attacked = clone(document)
    Y.applyUpdate(attacked, attack.update)

    expect(editorYDocToJson(attacked)).not.toEqual(editorYDocToJson(document))
    expect(JSON.stringify(editorYDocToJson(attacked))).toContain(SUGGESTION_ID)
    expect(Y.createAbsolutePositionFromRelativePosition(
      attack.anchor,
      attacked,
    )).toBeNull()
    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: `combined-topology-${path.join('-')}`,
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update: attack.update,
    })).rejects.toThrowError('suggestion_policy_violation')
    expect(fixture.api.persisted).toHaveLength(0)

    attacked.destroy()
    document.destroy()
    await fixture.close()
  })

  it('persists a legitimate suggestion that preserves shared Yjs topology', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('First\n\nSecond'))
    const anchoredText = firstXmlText(xmlElementAt(
      document.getXmlFragment(EDITOR_YJS_FRAGMENT),
      [1],
    ))
    if (!anchoredText) throw new Error('Legitimate suggestion fixture has no text anchor')
    const anchor = Y.createRelativePositionFromTypeIndex(anchoredText, 1)
    const update = suggestionUpdate(document)
    const updated = clone(document)
    Y.applyUpdate(updated, update)

    expect(Y.createAbsolutePositionFromRelativePosition(anchor, updated)).not.toBeNull()
    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'legitimate-suggestion',
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update,
    })).resolves.toBe('pending')
    Y.applyUpdate(document, update)
    expect(fixture.coordinator.finishClientUpdate('legitimate-suggestion', document)).toEqual({
      hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      sequence: 1,
      type: 'durable_ack',
    })
    expect(fixture.api.persisted).toHaveLength(1)
    expect(fixture.api.persisted[0]).toMatchObject({
      changeKind: 'suggestion',
      suggestionIds: [SUGGESTION_ID],
    })
    expect(fixture.api.persisted[0]?.update).toBe(update)

    updated.destroy()
    document.destroy()
    await fixture.close()
  })

  it.each(['deletion', 'modification'] as const)(
    'persists a normal %s suggestion without rejecting its causal history',
    async (operation) => {
      const fixture = await coordinatorFixture()
      const document = editorJsonToYDoc(parseEditorMarkdown('First'))
      const update = suggestionUpdate(document, operation)

      await expect(fixture.coordinator.prepareClientUpdate({
        allowNoop: false,
        connectionId: `legitimate-${operation}`,
        context: { ...fixture.context, access: 'suggest' },
        document,
        room: ROOM,
        update,
      })).resolves.toBe('pending')
      Y.applyUpdate(document, update)
      expect(fixture.coordinator.finishClientUpdate(
        `legitimate-${operation}`,
        document,
      )?.sequence).toBe(1)
      expect(fixture.api.persisted).toHaveLength(1)
      expect(fixture.api.persisted[0]).toMatchObject({
        changeKind: 'suggestion',
        suggestions: [{ kind: operation }],
        update,
      })

      document.destroy()
      await fixture.close()
    },
  )

  it('allows a suggest author to edit existing insertion text with a resolved tombstone', async () => {
    const fixture = await coordinatorFixture()
    const document = editorJsonToYDoc(parseEditorMarkdown('First'))
    Y.applyUpdate(document, suggestionUpdate(document, 'insertion', 'XY'))
    const replica = clone(document)
    const vector = Y.encodeStateVector(replica)
    const text = firstXmlText(xmlElementAt(
      replica.getXmlFragment(EDITOR_YJS_FRAGMENT),
      [0],
    ))
    if (!text) throw new Error('Resolved-tombstone fixture has no XML text')
    let offset = 0
    let deletedAt = -1
    for (const part of text.toDelta()) {
      if (typeof part.insert !== 'string') continue
      const index = part.insert.indexOf('Y')
      if (index >= 0) {
        deletedAt = offset + index
        break
      }
      offset += part.insert.length
    }
    if (deletedAt < 0) throw new Error('Resolved-tombstone fixture has no inserted text')
    text.delete(deletedAt, 1)
    const update = Y.encodeStateAsUpdate(replica, vector)
    replica.destroy()

    await expect(fixture.coordinator.prepareClientUpdate({
      allowNoop: false,
      connectionId: 'resolved-suggestion-tombstone',
      context: { ...fixture.context, access: 'suggest' },
      document,
      room: ROOM,
      update,
    })).resolves.toBe('pending')
    Y.applyUpdate(document, update)
    expect(fixture.coordinator.finishClientUpdate(
      'resolved-suggestion-tombstone',
      document,
    )?.sequence).toBe(1)
    expect(fixture.api.persisted).toHaveLength(1)
    expect(fixture.api.persisted[0]).toMatchObject({
      changeKind: 'suggestion',
      suggestionIds: [SUGGESTION_ID],
      update,
    })

    document.destroy()
    await fixture.close()
  })
})

async function coordinatorFixture(
  sequence = 0,
  overrides: Parameters<typeof settings>[0] = {},
): Promise<{
  api: FakeCollaborationApi
  close: () => Promise<void>
  context: ConnectionContext
  coordinator: DocumentCoordinator
}> {
  const api = new FakeCollaborationApi()
  const configured = settings(overrides)
  const metrics = new SidecarMetrics()
  const lease = new InstanceLeaseManager(api, configured, silentLogger, metrics, () => undefined)
  await lease.start()
  const coordinator = new DocumentCoordinator(api, lease, configured, silentLogger, metrics)
  coordinator.initialize(ROOM, sequence)
  return {
    api,
    close: () => lease.stop(),
    context: {
      access: 'edit',
      documentId: DOCUMENT_ID,
      expiresAt: Date.now() / 1_000 + 60,
      generation: 1,
      leaseId: 'lease-1',
      protocolVersion: configured.protocolVersion,
      schemaHash: await getEditorSchemaFingerprint(),
      schemaVersion: configured.schemaVersion,
      sessionId: 'session-1',
      tenantId: 'tenant-1',
      user: { color: '#123456', id: USER_ID, name: 'Ada' },
    },
    coordinator,
  }
}

function updateFor(document: Y.Doc, markdownTarget: string): Uint8Array {
  const replica = clone(document)
  const vector = Y.encodeStateVector(replica)
  const fragment = replica.getXmlFragment(EDITOR_YJS_FRAGMENT)
  const initialized = initProseMirrorDoc(fragment, schema)
  const target = schema.nodeFromJSON(parseEditorMarkdown(markdownTarget))
  updateYFragment(replica, fragment, target, initialized.meta)
  const update = Y.encodeStateAsUpdate(replica, vector)
  replica.destroy()
  return update
}

function updateForV2(document: Y.Doc, markdownTarget: string): Uint8Array {
  const replica = clone(document)
  const vector = Y.encodeStateVector(replica)
  const fragment = replica.getXmlFragment(EDITOR_YJS_FRAGMENT)
  const initialized = initProseMirrorDoc(fragment, schema)
  const target = schema.nodeFromJSON(parseEditorMarkdown(markdownTarget))
  updateYFragment(replica, fragment, target, initialized.meta)
  const update = Y.encodeStateAsUpdateV2(replica, vector)
  replica.destroy()
  return update
}

function genericCanonicalityAttack(
  attack: 'merged' | 'merged-delete' | 'suffix' | 'v2',
): { document: Y.Doc; update: Uint8Array } {
  const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))
  if (attack === 'v2') {
    return { document, update: updateForV2(document, 'Hello!') }
  }
  const first = updateFor(document, attack === 'merged-delete' ? 'ello' : 'Hello!')
  if (attack === 'suffix') {
    const update = new Uint8Array(first.byteLength + 3)
    update.set(first)
    update.set([0xde, 0xad, 0xbe], first.byteLength)
    return { document, update }
  }
  Y.applyUpdate(document, first)
  const second = updateFor(document, attack === 'merged-delete' ? 'ello!' : 'Hello!!')
  return { document, update: Y.mergeUpdates([first, second]) }
}

function clone(document: Y.Doc): Y.Doc {
  const result = new Y.Doc()
  Y.applyUpdate(result, Y.encodeStateAsUpdate(document))
  return result
}

function topologyReplacementUpdate(
  document: Y.Doc,
  path: readonly number[],
): { anchor: Y.RelativePosition; update: Uint8Array } {
  const sourceNode = xmlElementAt(document.getXmlFragment(EDITOR_YJS_FRAGMENT), path)
  const sourceText = firstXmlText(sourceNode)
  if (!sourceText) throw new Error('Topology fixture has no text anchor')
  const anchor = Y.createRelativePositionFromTypeIndex(sourceText, 1)

  const replica = clone(document)
  const vector = Y.encodeStateVector(replica)
  replaceXmlElement(replica.getXmlFragment(EDITOR_YJS_FRAGMENT), path)
  const update = Y.encodeStateAsUpdate(replica, vector)
  replica.destroy()
  return { anchor, update }
}

function suggestionWithTopologyReplacementUpdate(
  document: Y.Doc,
  path: readonly number[],
): { anchor: Y.RelativePosition; update: Uint8Array } {
  const sourceNode = xmlElementAt(document.getXmlFragment(EDITOR_YJS_FRAGMENT), path)
  const sourceText = firstXmlText(sourceNode)
  if (!sourceText) throw new Error('Combined topology fixture has no text anchor')
  const anchor = Y.createRelativePositionFromTypeIndex(sourceText, 1)

  const replica = clone(document)
  const vector = Y.encodeStateVector(replica)
  applySuggestion(replica)
  replaceXmlElement(replica.getXmlFragment(EDITOR_YJS_FRAGMENT), path)
  const update = Y.encodeStateAsUpdate(replica, vector)
  replica.destroy()
  return { anchor, update }
}

type SuggestionOperation = 'deletion' | 'insertion' | 'modification'

function suggestionUpdate(
  document: Y.Doc,
  operation: SuggestionOperation = 'insertion',
  insertedText = '!',
): Uint8Array {
  const replica = clone(document)
  const vector = Y.encodeStateVector(replica)
  applySuggestion(replica, operation, insertedText)
  const update = Y.encodeStateAsUpdate(replica, vector)
  replica.destroy()
  return update
}

function suggestionWithTransientYjsHistoryUpdate(
  document: Y.Doc,
  hiddenHistory:
    | 'element-attribute'
    | 'text-attribute'
    | 'text-content'
    | 'text-format'
    | 'unknown-root',
): Uint8Array {
  const replica = clone(document)
  const vector = Y.encodeStateVector(replica)
  applySuggestion(replica)
  const paragraph = xmlElementAt(replica.getXmlFragment(EDITOR_YJS_FRAGMENT), [1])
  const text = firstXmlText(paragraph)
  if (!text) throw new Error('Transient-history fixture has no XML text')
  const hidden = 'x'.repeat(256 * 1024)
  if (hiddenHistory === 'element-attribute') {
    paragraph.setAttribute('rogue', hidden)
    paragraph.removeAttribute('rogue')
  } else if (hiddenHistory === 'text-attribute') {
    text.setAttribute('rogue', hidden)
    text.removeAttribute('rogue')
  } else if (hiddenHistory === 'text-content') {
    text.insert(1, hidden)
    text.delete(1, hidden.length)
  } else if (hiddenHistory === 'text-format') {
    text.format(0, text.length, { rogue: hidden })
    text.format(0, text.length, { rogue: null })
  } else {
    const rogue = replica.getMap('rogue')
    rogue.set('payload', hidden)
    rogue.delete('payload')
  }
  const update = Y.encodeStateAsUpdate(replica, vector)
  replica.destroy()
  return update
}

function suggestionWithMissingCausalUpdate(
  document: Y.Doc,
  dependency: 'delete' | 'struct',
): Uint8Array {
  const causalReplica = clone(document)
  const text = firstXmlText(xmlElementAt(
    causalReplica.getXmlFragment(EDITOR_YJS_FRAGMENT),
    [1],
  ))
  if (!text) throw new Error('Causal fixture has no XML text')
  text.insert(text.length, 'missing')
  const dependencyVector = Y.encodeStateVector(causalReplica)
  if (dependency === 'struct') text.insert(text.length, '!')
  else text.delete(text.length - 'missing'.length, 'missing'.length)
  const incomplete = Y.encodeStateAsUpdate(causalReplica, dependencyVector)
  causalReplica.destroy()
  return Y.mergeUpdates([suggestionUpdate(document), incomplete])
}

function suggestionWithHiddenYjsStateUpdate(
  document: Y.Doc,
  hiddenState: 'element-attribute' | 'empty-text' | 'text-attribute',
): Uint8Array {
  const replica = clone(document)
  const vector = Y.encodeStateVector(replica)
  applySuggestion(replica)
  const paragraph = xmlElementAt(replica.getXmlFragment(EDITOR_YJS_FRAGMENT), [1])
  if (hiddenState === 'element-attribute') {
    paragraph.setAttribute('rogue', 'hidden')
  } else if (hiddenState === 'text-attribute') {
    const text = firstXmlText(paragraph)
    if (!text) throw new Error('Hidden-state fixture has no XML text')
    text.setAttribute('rogue', 'hidden')
  } else {
    paragraph.insert(paragraph.length, [new Y.XmlText()])
  }
  const update = Y.encodeStateAsUpdate(replica, vector)
  replica.destroy()
  return update
}

function applySuggestion(
  document: Y.Doc,
  operation: SuggestionOperation = 'insertion',
  insertedText = '!',
): void {
  const fragment = document.getXmlFragment(EDITOR_YJS_FRAGMENT)
  const initialized = initProseMirrorDoc(fragment, schema)
  let textPosition: number | null = null
  initialized.doc.descendants((node, position) => {
    if (textPosition === null && node.isText) textPosition = position
  })
  if (textPosition === null) throw new Error('Suggestion fixture has no text position')
  const state = EditorState.create({ schema, doc: initialized.doc })
  const transaction = operation === 'insertion'
    ? state.tr.insertText(insertedText, textPosition + 1)
    : operation === 'deletion'
      ? state.tr.delete(textPosition + 1, textPosition + 2)
      : state.tr.insertText('X', textPosition + 1, textPosition + 2)
  const tracked = transformToInqtrixSuggestionTransaction(
    transaction,
    state,
    { authorId: USER_ID, createdAt: 1_784_112_000, patchId: PATCH_ID },
    () => SUGGESTION_ID,
  )
  updateYFragment(document, fragment, tracked.doc, initialized.meta)
}

function replaceXmlElement(root: Y.XmlFragment, path: readonly number[]): void {
  const parent = xmlParentAt(root, path)
  const index = path.at(-1)
  if (index === undefined) throw new Error('Topology fixture path is empty')
  const current = parent.get(index)
  if (!(current instanceof Y.XmlElement)) {
    throw new Error('Topology fixture does not point to an XML element')
  }
  const replacement = current.clone()
  parent.delete(index, 1)
  parent.insert(index, [replacement])
}

function replaceFirstXmlText(root: Y.XmlFragment, path: readonly number[]): void {
  const element = xmlElementAt(root, path)
  const current = element.get(0)
  if (!(current instanceof Y.XmlText)) {
    throw new Error('Topology fixture does not point to XML text')
  }
  const replacement = current.clone()
  element.delete(0, 1)
  element.insert(0, [replacement])
}

function xmlParentAt(
  root: Y.XmlFragment,
  path: readonly number[],
): Y.XmlFragment | Y.XmlElement {
  if (path.length === 0) throw new Error('Topology fixture path is empty')
  let parent: Y.XmlFragment | Y.XmlElement = root
  for (const index of path.slice(0, -1)) {
    const child: unknown = parent.get(index)
    if (!(child instanceof Y.XmlElement)) {
      throw new Error('Topology fixture path is invalid')
    }
    parent = child
  }
  return parent
}

function xmlElementAt(root: Y.XmlFragment, path: readonly number[]): Y.XmlElement {
  const parent = xmlParentAt(root, path)
  const index = path.at(-1)
  const node = index === undefined ? null : parent.get(index)
  if (!(node instanceof Y.XmlElement)) throw new Error('Topology fixture path is invalid')
  return node
}

function firstXmlText(node: Y.XmlElement): Y.XmlText | null {
  for (let index = 0; index < node.length; index += 1) {
    const child = node.get(index)
    if (child instanceof Y.XmlText) return child
    if (child instanceof Y.XmlElement) {
      const nested = firstXmlText(child)
      if (nested) return nested
    }
  }
  return null
}

function markdown(document: Y.Doc): string {
  return serializeEditorJson(editorYDocToJson(document), 'final').trim()
}
