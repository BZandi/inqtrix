import {
  createEditorSchemaExtensions,
  transformToInqtrixSuggestionTransaction,
} from '@inqtrix/editor-schema'
import { getSchema } from '@tiptap/core'
import { EditorState, TextSelection } from '@tiptap/pm/state'
import { describe, expect, it } from 'vitest'

import {
  CollaborationSuggestionGroupingCoordinator,
  CollaborationSuggestionUndoHistory,
  collaborationReviewAllowsTransaction,
  detectForeignSuggestionCollision,
} from './tiptap'

describe('collaboration transaction policy', () => {
  it('blocks local document mutations for view and comment modes', () => {
    for (const writeMode of ['view', 'comment'] as const) {
      expect(collaborationReviewAllowsTransaction({
        collaboration: true,
        docChanged: true,
        remote: false,
        writeMode,
      })).toBe(false)
    }
  })

  it('allows selection-only, remote, edit, and suggestion transactions', () => {
    expect(collaborationReviewAllowsTransaction({
      collaboration: true,
      docChanged: false,
      remote: false,
      writeMode: 'comment',
    })).toBe(true)
    expect(collaborationReviewAllowsTransaction({
      collaboration: true,
      docChanged: true,
      remote: true,
      writeMode: 'view',
    })).toBe(true)
    for (const writeMode of ['edit', 'suggest'] as const) {
      expect(collaborationReviewAllowsTransaction({
        collaboration: true,
        docChanged: true,
        remote: false,
        writeMode,
      })).toBe(true)
    }
  })
})

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

  it('never allocates suggestion identity in comment-only mode', () => {
    const grouping = new CollaborationSuggestionGroupingCoordinator()
    grouping.observeContext({
      authorId: 'user-1',
      documentId: 'doc-1',
      writeMode: 'comment',
    })

    expect(() => grouping.metadata()).toThrowError(
      'Suggestion grouping requires an active document and author.',
    )
  })
})

describe('collaboration suggestion undo history', () => {
  it('offers only session-local open patches and serializes a pending decision', () => {
    const history = new CollaborationSuggestionUndoHistory()
    history.record('patch-1')
    history.record('patch-1')
    history.record('patch-2')
    const bothOpen = new Set(['patch-1', 'patch-2'])

    expect(history.current(bothOpen)).toBe('patch-2')
    expect(history.begin('patch-2', bothOpen)).toBe(true)
    expect(history.current(bothOpen)).toBeNull()

    history.fail('patch-2')
    expect(history.current(bothOpen)).toBe('patch-2')
  })

  it('drops an authoritatively resolved patch without exposing reloaded history', () => {
    const history = new CollaborationSuggestionUndoHistory()
    history.record('patch-1')
    history.record('patch-2')
    expect(history.begin('patch-2', new Set(['patch-1', 'patch-2']))).toBe(true)

    history.reconcile(new Set(['patch-1']))

    expect(history.current(new Set(['patch-1']))).toBe('patch-1')
    history.reconcile(new Set())
    expect(history.current(new Set())).toBeNull()
    expect(new CollaborationSuggestionUndoHistory().current(new Set(['patch-1']))).toBeNull()
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
