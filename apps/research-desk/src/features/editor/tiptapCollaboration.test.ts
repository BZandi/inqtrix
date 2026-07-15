import {
  createEditorSchemaExtensions,
  transformToInqtrixSuggestionTransaction,
} from '@inqtrix/editor-schema'
import { getSchema } from '@tiptap/core'
import { EditorState, TextSelection } from '@tiptap/pm/state'
import { describe, expect, it } from 'vitest'

import {
  CollaborationSuggestionGroupingCoordinator,
  detectForeignSuggestionCollision,
} from './tiptap'

describe('collaboration suggest grouping', () => {
  it('reuses the exact patch id and creation time until five seconds of inactivity', () => {
    let now = 1_000
    let nextPatch = 1
    const grouping = new CollaborationSuggestionGroupingCoordinator({
      createPatchId: () => `patch-${nextPatch++}`,
      now: () => now,
    })
    grouping.observeContext({
      authorId: 'user-1',
      documentId: 'doc-1',
      writeMode: 'suggest',
    })

    const first = grouping.metadata()
    now += 4_999
    const continuous = grouping.metadata()
    now += 5_000
    const afterIdle = grouping.metadata()

    expect(continuous).toEqual(first)
    expect(afterIdle).toEqual({
      authorId: 'user-1',
      createdAt: 10_999,
      patchId: 'patch-2',
    })
  })

  it('resets on selection, mode, author, document, and remote boundaries', () => {
    let nextPatch = 1
    const grouping = new CollaborationSuggestionGroupingCoordinator({
      createPatchId: () => `patch-${nextPatch++}`,
      now: () => 100,
    })
    const context = {
      authorId: 'user-1',
      documentId: 'doc-1',
      writeMode: 'suggest' as const,
    }
    grouping.observeContext(context)
    expect(grouping.metadata().patchId).toBe('patch-1')

    grouping.reset()
    expect(grouping.metadata().patchId).toBe('patch-2')
    grouping.observeContext({ ...context, writeMode: 'edit' })
    grouping.observeContext(context)
    expect(grouping.metadata().patchId).toBe('patch-3')
    grouping.observeContext({ ...context, authorId: 'user-2' })
    expect(grouping.metadata().patchId).toBe('patch-4')
    grouping.observeContext({ ...context, documentId: 'doc-2' })
    expect(grouping.metadata().patchId).toBe('patch-5')
  })
})

describe('foreign collaboration suggestion collision', () => {
  const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
  const initialDocument = schema.node('doc', null, [
    schema.node('paragraph', null, schema.text('Hello world')),
  ])
  const initialState = EditorState.create({ schema, doc: initialDocument })
  const foreignInsertion = initialState.tr
    .setSelection(TextSelection.create(initialState.doc, 6))
    .insertText(' shared')
  const tracked = transformToInqtrixSuggestionTransaction(
    foreignInsertion,
    initialState,
    { authorId: 'peer', createdAt: 10, patchId: 'patch-peer' },
    () => 'suggestion-peer',
  )
  const trackedState = initialState.apply(tracked)

  it('returns the authoritative patch and suggestion before an overlapping mutation', () => {
    const overlap = trackedState.tr.insertText('X', 7, 8)

    expect(detectForeignSuggestionCollision(overlap, 'local-user')).toEqual({
      patchId: 'patch-peer',
      suggestionId: 'suggestion-peer',
    })
  })

  it('allows the suggestion author and non-overlapping local edits', () => {
    const overlap = trackedState.tr.insertText('X', 7, 8)
    const separate = trackedState.tr.insertText('Y', 1, 2)

    expect(detectForeignSuggestionCollision(overlap, 'peer')).toBeNull()
    expect(detectForeignSuggestionCollision(separate, 'local-user')).toBeNull()
  })
})
