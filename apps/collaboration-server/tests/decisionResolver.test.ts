import { getSchema } from '@tiptap/core'
import type { Node as ProseMirrorNode } from '@tiptap/pm/model'
import { EditorState } from '@tiptap/pm/state'
import {
  createEditorSchemaExtensions,
  editorJsonToYDoc,
  editorYDocToJson,
  parseEditorMarkdown,
  serializeEditorJson,
  suggestionDescriptors,
  transformToInqtrixSuggestionTransaction,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import { resolvePatchDecision } from '../src/decisionResolver'

const PATCH_ID = '22222222-2222-4222-8222-222222222222'
const SUGGESTION_ID = '33333333-3333-4333-8333-333333333333'
const SECOND_PATCH_ID = '44444444-4444-4444-8444-444444444444'
const SECOND_SUGGESTION_ID = '55555555-5555-4555-8555-555555555555'
const USER_ID = '11111111-1111-4111-8111-111111111111'
const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))

describe('patch decision resolver', () => {
  it.each([
    ['text replacement', 'accept', 'World'],
    ['text replacement', 'reject', 'Hello'],
    ['formatting replacement', 'accept', '**Hello**'],
    ['formatting replacement', 'reject', 'Hello'],
  ] as const)('resolves a %s %s atomically', (change, decision, expected) => {
    const tracked = modification(change)
    const document = editorJsonToYDoc(tracked.toJSON())

    const result = resolvePatchDecision(document, {
      decision,
      patchIds: [PATCH_ID],
    })
    const resolved = schema.nodeFromJSON(editorYDocToJson(document))

    expect(serializeEditorJson(resolved.toJSON(), 'final').trim()).toBe(expected)
    expect(suggestionDescriptors(resolved)).toEqual([])
    expect(result).toMatchObject({
      patchIds: [PATCH_ID],
      suggestionIds: [SUGGESTION_ID],
      suggestions: [{
        kind: change === 'text replacement' ? 'replacement' : 'format',
        suggestionId: SUGGESTION_ID,
      }],
    })
    document.destroy()
  })

  it.each([
    ['accept', ''],
    ['reject', 'Hello'],
  ] as const)('resolves an inline deletion at the document end with %s', (
    decision,
    expected,
  ) => {
    const tracked = inlineDeletionAtDocumentEnd()
    const document = editorJsonToYDoc(tracked.toJSON())

    const result = resolvePatchDecision(document, {
      decision,
      patchIds: [PATCH_ID],
    })
    const resolved = schema.nodeFromJSON(editorYDocToJson(document))

    expect(serializeEditorJson(resolved.toJSON(), 'final').trim()).toBe(expected)
    expect(suggestionDescriptors(resolved)).toEqual([])
    expect(result).toMatchObject({
      patchIds: [PATCH_ID],
      suggestionIds: [SUGGESTION_ID],
      suggestions: [{ kind: 'deletion', suggestionId: SUGGESTION_ID }],
    })
    document.destroy()
  })

  it('rejects a partial modification pair without mutating the Y.Doc', () => {
    const tracked = modification('text replacement')
    const partialState = EditorState.create({ schema, doc: tracked })
    let insertionFrom: number | null = null
    let insertionTo: number | null = null
    tracked.descendants((node, position) => {
      if (node.marks.some((mark) => mark.type.name === 'insertion')) {
        insertionFrom = position
        insertionTo = position + node.nodeSize
      }
    })
    if (insertionFrom === null || insertionTo === null) {
      throw new Error('Modification fixture has no insertion half')
    }
    const partial = partialState.apply(partialState.tr.removeMark(
      insertionFrom,
      insertionTo,
      schema.marks.insertion,
    )).doc
    const document = editorJsonToYDoc(partial.toJSON())
    const before = Y.encodeStateAsUpdate(document)

    expect(() => resolvePatchDecision(document, {
      decision: 'accept',
      patchIds: [PATCH_ID],
    })).toThrowError('decision_conflict')
    expect(Y.encodeStateAsUpdate(document)).toEqual(before)
    document.destroy()
  })

  it('rejects a malformed modification pair without mutating the Y.Doc', () => {
    const tracked = modification('formatting replacement')
    const malformedState = EditorState.create({ schema, doc: tracked })
    const insertions: Array<{
      from: number
      mark: NonNullable<ProseMirrorNode['marks'][number]>
      to: number
    }> = []
    tracked.descendants((node, position) => {
      const mark = node.marks.find((candidate) => candidate.type.name === 'insertion')
      if (mark) insertions.push({ from: position, mark, to: position + node.nodeSize })
    })
    const insertion = insertions[0]
    if (!insertion) throw new Error('Modification fixture has no insertion half')
    const malformedMark = insertion.mark.type.create({
      ...insertion.mark.attrs,
      kind: 'insertion',
    })
    const malformed = malformedState.apply(
      malformedState.tr
        .removeMark(insertion.from, insertion.to, insertion.mark)
        .addMark(insertion.from, insertion.to, malformedMark),
    ).doc
    const document = editorJsonToYDoc(malformed.toJSON())
    const before = Y.encodeStateAsUpdate(document)

    expect(() => resolvePatchDecision(document, {
      decision: 'reject',
      patchIds: [PATCH_ID],
    })).toThrowError('decision_conflict')
    expect(Y.encodeStateAsUpdate(document)).toEqual(before)
    document.destroy()
  })

  it('rejects distant modification halves without mutating the Y.Doc', () => {
    const document = editorJsonToYDoc(distantModification().toJSON())
    const before = Y.encodeStateAsUpdate(document)

    expect(() => resolvePatchDecision(document, {
      decision: 'accept',
      patchIds: [PATCH_ID],
    })).toThrowError('decision_conflict')
    expect(Y.encodeStateAsUpdate(document)).toEqual(before)
    document.destroy()
  })

  it.each([
    [PATCH_ID, SECOND_SUGGESTION_ID],
    [SECOND_PATCH_ID, SUGGESTION_ID],
  ])('decides half-open adjacent patch %s without touching its neighbor', (
    selectedPatchId,
    remainingSuggestionId,
  ) => {
    const document = editorJsonToYDoc(adjacentInsertions().toJSON())

    resolvePatchDecision(document, {
      decision: 'accept',
      patchIds: [selectedPatchId],
    })

    expect(suggestionDescriptors(schema.nodeFromJSON(editorYDocToJson(document)))).toEqual([
      expect.objectContaining({ suggestionId: remainingSuggestionId }),
    ])
    document.destroy()
  })
})

function modification(
  change: 'formatting replacement' | 'text replacement',
): ProseMirrorNode {
  const document = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
  const state = EditorState.create({ schema, doc: document })
  const bold = schema.marks.bold
  if (!bold) throw new Error('Editor schema has no bold mark')
  const transaction = change === 'text replacement'
    ? state.tr.insertText('World', 1, 6)
    : state.tr.addMark(1, 6, bold.create())
  return state.apply(transformToInqtrixSuggestionTransaction(
    transaction,
    state,
    { authorId: USER_ID, createdAt: 1_784_112_000, patchId: PATCH_ID },
    () => SUGGESTION_ID,
  )).doc
}

function inlineDeletionAtDocumentEnd(): ProseMirrorNode {
  const document = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
  const state = EditorState.create({ schema, doc: document })
  return state.apply(transformToInqtrixSuggestionTransaction(
    state.tr.delete(1, 6),
    state,
    { authorId: USER_ID, createdAt: 1_784_112_002, patchId: PATCH_ID },
    () => SUGGESTION_ID,
  )).doc
}

function distantModification(): ProseMirrorNode {
  const deletion = schema.marks.deletion
  const insertion = schema.marks.insertion
  if (!deletion || !insertion) throw new Error('Editor schema has no suggestion marks')
  const metadata = {
    authorId: USER_ID,
    createdAt: 1_784_112_001,
    id: SUGGESTION_ID,
    kind: 'modification',
    patchId: PATCH_ID,
    suggestionId: SUGGESTION_ID,
  }
  return schema.node('doc', null, [schema.node('paragraph', null, [
    schema.text('Before', [deletion.create(metadata)]),
    schema.text(' unrelated '),
    schema.text('After', [insertion.create(metadata)]),
  ])])
}

function adjacentInsertions(): ProseMirrorNode {
  const insertion = schema.marks.insertion
  if (!insertion) throw new Error('Editor schema has no insertion mark')
  const mark = (suggestionId: string, patchId: string, createdAt: number) => insertion.create({
    authorId: USER_ID,
    createdAt,
    id: suggestionId,
    kind: 'insertion',
    patchId,
    suggestionId,
  })
  return schema.node('doc', null, [schema.node('paragraph', null, [
    schema.text('A', [mark(SUGGESTION_ID, PATCH_ID, 1_784_112_010)]),
    schema.text('B', [mark(SECOND_SUGGESTION_ID, SECOND_PATCH_ID, 1_784_112_011)]),
  ])])
}
