import { getSchema, type JSONContent } from '@tiptap/core'
import { EditorState, TextSelection } from '@tiptap/pm/state'
import { initProseMirrorDoc, updateYFragment } from '@tiptap/y-tiptap'
import {
  EDITOR_YJS_FRAGMENT,
  createEditorSchemaExtensions,
  editorJsonToYDoc,
  parseEditorMarkdown,
  projectOriginalDocument,
  transformToInqtrixSuggestionTransaction,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import { validateSuggestionUpdate } from '../src/suggestPolicy'
import { USER_ID } from './helpers'

const PATCH_ID = '22222222-2222-4222-8222-222222222222'
const SUGGESTION_ID = '33333333-3333-4333-8333-333333333333'
const OTHER_USER_ID = '44444444-4444-4444-8444-444444444444'
const CREATED_AT = 1_784_112_000
const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))

describe('suggestion policy', () => {
  it('emits authoritative active patch state for a new suggestion', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: before })
    const direct = state.tr
      .setSelection(TextSelection.create(state.doc, 6))
      .insertText('!')
    const tracked = transformToInqtrixSuggestionTransaction(
      direct,
      state,
      { authorId: USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc

    const result = validateMappedSuggestionUpdate(
      before.toJSON(),
      tracked.toJSON(),
      'suggest',
      USER_ID,
    )

    expect(result.changeKind).toBe('suggestion')
    expect(result.suggestionIds).toEqual([SUGGESTION_ID])
    expect(result.patches).toEqual([{
      activeSuggestionIds: [SUGGESTION_ID],
      authorId: USER_ID,
      createdAt: CREATED_AT,
      kinds: ['insertion'],
      patchId: PATCH_ID,
    }])
  })

  it('allows a marked text suggestion to create the Yjs text type in an empty paragraph', () => {
    const before = schema.node('doc', null, [schema.node('paragraph')])
    const state = EditorState.create({ schema, doc: before })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('A', 1),
      state,
      { authorId: USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc

    const result = validateMappedSuggestionUpdate(
      before.toJSON(),
      tracked.toJSON(),
      'suggest',
      USER_ID,
    )

    expect(result.changeKind).toBe('suggestion')
    expect(result.suggestionIds).toEqual([SUGGESTION_ID])
  })

  it('rejects removal of the last active suggestion outside an explicit decision', () => {
    const plain = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: plain })
    const direct = state.tr
      .setSelection(TextSelection.create(state.doc, 6))
      .insertText('!')
    const tracked = transformToInqtrixSuggestionTransaction(
      direct,
      state,
      { authorId: USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc

    expect(() => validateSuggestionUpdate(
      tracked.toJSON(),
      plain.toJSON(),
      'suggest',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('recognizes a replacement as one modification descriptor', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: before })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('World', 1, 6),
      state,
      { authorId: USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc

    const result = validateMappedSuggestionUpdate(
      before.toJSON(),
      tracked.toJSON(),
      'suggest',
      USER_ID,
    )

    expect(result.suggestions).toEqual([expect.objectContaining({
      kind: 'modification',
      suggestionId: SUGGESTION_ID,
    })])
    expect(result.patches[0]?.kinds).toEqual(['modification'])
  })

  it('rejects a newly injected suggestion attributed to another user', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: before })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', 6),
      state,
      { authorId: OTHER_USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc

    expect(() => validateSuggestionUpdate(
      before.toJSON(),
      tracked.toJSON(),
      'suggest',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('allows a direct edit before an unchanged foreign suggestion', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('First\n\nSecond'))
    const state = EditorState.create({ schema, doc: before })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', state.doc.content.size - 1),
      state,
      { authorId: OTHER_USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc
    const trackedState = EditorState.create({ schema, doc: tracked })
    const after = trackedState.apply(trackedState.tr.insertText('Prefix ', 1)).doc

    const result = validateMappedSuggestionUpdate(
      tracked.toJSON(),
      after.toJSON(),
      'edit',
      USER_ID,
    )

    expect(result.changeKind).toBe('direct')
    expect(result.suggestionIds).toEqual([])
  })

  it('allows a same-paragraph direct prefix before an unchanged foreign suggestion', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: before })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', state.doc.content.size - 1),
      state,
      { authorId: OTHER_USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc
    const trackedState = EditorState.create({ schema, doc: tracked })
    const after = trackedState.apply(trackedState.tr.insertText('Prefix ', 1)).doc

    const result = validateMappedSuggestionUpdate(
      tracked.toJSON(),
      after.toJSON(),
      'edit',
      USER_ID,
    )

    expect(result.changeKind).toBe('direct')
    expect(result.suggestionIds).toEqual([])
  })

  it('fails closed when a foreign occurrence has no shared-document position mapping', () => {
    const tracked = trackedInsertion(parseEditorMarkdown('abcabc'), OTHER_USER_ID, 3)

    expect(() => validateSuggestionUpdate(
      tracked.toJSON(),
      tracked.toJSON(),
      'edit',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('allows an unrelated structural insertion before a foreign suggestion', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('First\n\nSecond'))
    const state = EditorState.create({ schema, doc: before })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', state.doc.content.size - 1),
      state,
      { authorId: OTHER_USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc.toJSON()
    const after = structuredClone(tracked)
    const inserted = parseEditorMarkdown('Inserted').content?.[0]
    if (!after.content || !inserted) throw new Error('structural fixture is invalid')
    after.content.unshift(inserted)

    const result = validateMappedSuggestionUpdate(tracked, after, 'edit', USER_ID)

    expect(result.changeKind).toBe('direct')
    expect(result.suggestionIds).toEqual([])
  })

  it('rejects relocation of a foreign insertion to another structural parent', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('First\n\nSecond'))
    const state = EditorState.create({ schema, doc: before })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', state.doc.content.size - 1),
      state,
      { authorId: OTHER_USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc.toJSON()
    const after = structuredClone(tracked)
    const first = after.content?.[0]
    const second = after.content?.[1]
    const foreignIndex = second?.content?.findIndex((node: JSONContent) => (
      node.marks?.some((mark: NonNullable<JSONContent['marks']>[number]) => (
        mark.attrs?.suggestionId === SUGGESTION_ID
      ))
    )) ?? -1
    if (!first || !second?.content || foreignIndex < 0) {
      throw new Error('foreign suggestion fixture is invalid')
    }
    const [foreign] = second.content.splice(foreignIndex, 1)
    if (!foreign) throw new Error('foreign suggestion fixture is invalid')
    first.content = [...(first.content ?? []), foreign]

    expect(() => validateMappedSuggestionUpdate(
      tracked,
      after,
      'edit',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('rejects relocation between structurally identical sibling blocks', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('Same\n\nSame'))
    const state = EditorState.create({ schema, doc: before })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', state.doc.child(0).nodeSize - 1),
      state,
      { authorId: OTHER_USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc.toJSON()
    const after = structuredClone(tracked)
    const first = after.content?.[0]
    const second = after.content?.[1]
    const foreignIndex = first?.content?.findIndex((node: JSONContent) => (
      node.marks?.some((mark: NonNullable<JSONContent['marks']>[number]) => (
        mark.attrs?.suggestionId === SUGGESTION_ID
      ))
    )) ?? -1
    if (!first?.content || !second || foreignIndex < 0) {
      throw new Error('identical sibling fixture is invalid')
    }
    const [foreign] = first.content.splice(foreignIndex, 1)
    if (!foreign) throw new Error('identical sibling fixture is invalid')
    second.content = [...(second.content ?? []), foreign]

    expect(() => validateMappedSuggestionUpdate(
      tracked,
      after,
      'edit',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('rejects relocation of a foreign insertion within its structural parent', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('abc'))
    const state = EditorState.create({ schema, doc: before })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('X', 2),
      state,
      { authorId: OTHER_USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc.toJSON()
    const after = structuredClone(tracked)
    const paragraph = after.content?.[0]
    const foreign = paragraph?.content?.find((node: JSONContent) => (
      node.marks?.some((mark: NonNullable<JSONContent['marks']>[number]) => (
        mark.attrs?.suggestionId === SUGGESTION_ID
      ))
    ))
    if (!paragraph || !foreign) throw new Error('foreign suggestion fixture is invalid')
    paragraph.content = [
      { text: 'ab', type: 'text' },
      foreign,
      { text: 'c', type: 'text' },
    ]

    expect(() => validateMappedSuggestionUpdate(
      tracked,
      after,
      'edit',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('rejects relocation from abc[X]abc to abcabc[X]', () => {
    const tracked = trackedInsertion(parseEditorMarkdown('abcabc'), OTHER_USER_ID, 3)
    const after = relocateSuggestionToParentEnd(tracked.toJSON())

    expect(() => validateMappedSuggestionUpdate(
      tracked.toJSON(),
      after,
      'edit',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('rejects repeated-text relocation when multiple textual anchors look identical', () => {
    const tracked = trackedInsertion(parseEditorMarkdown('abcabcabc'), OTHER_USER_ID, 3)
    const after = relocateSuggestionToParentEnd(tracked.toJSON())

    expect(() => validateMappedSuggestionUpdate(
      tracked.toJSON(),
      after,
      'edit',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('rejects relocation inside nested block content', () => {
    const tracked = trackedInsertion(parseEditorMarkdown('> abcabc'), OTHER_USER_ID, 3)
    const after = relocateSuggestionToParentEnd(tracked.toJSON())

    expect(() => validateMappedSuggestionUpdate(
      tracked.toJSON(),
      after,
      'edit',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('rejects relocation inside a table cell', () => {
    const tracked = trackedInsertion(
      parseEditorMarkdown('| Value |\n| --- |\n| abcabc |'),
      OTHER_USER_ID,
      3,
    )
    const after = relocateSuggestionToParentEnd(tracked.toJSON())

    expect(() => validateMappedSuggestionUpdate(
      tracked.toJSON(),
      after,
      'edit',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('allows an author to edit the content of their own active suggestion', () => {
    const tracked = trackedInsertion(parseEditorMarkdown('abcabc'), USER_ID, 3)
    const after = structuredClone(tracked.toJSON())
    const occurrence = findSuggestionNode(after)
    if (!occurrence) throw new Error('own suggestion fixture is invalid')
    occurrence.text = `${occurrence.text ?? ''}Y`

    const result = validateMappedSuggestionUpdate(
      tracked.toJSON(),
      after,
      'suggest',
      USER_ID,
    )

    expect(result.changeKind).toBe('suggestion')
    expect(result.suggestionIds).toEqual([SUGGESTION_ID])
  })

  it('rejects a mixed direct and suggestion update from an edit user', () => {
    const before = schema.nodeFromJSON(parseEditorMarkdown('First\n\nSecond'))
    const initial = EditorState.create({ schema, doc: before })
    const directState = initial.apply(initial.tr.insertText('Prefix ', 1))
    const mixed = transformToInqtrixSuggestionTransaction(
      directState.tr.insertText('!', directState.doc.content.size - 1),
      directState,
      { authorId: USER_ID, createdAt: CREATED_AT, patchId: PATCH_ID },
      () => SUGGESTION_ID,
    ).doc

    expect(() => validateSuggestionUpdate(
      before.toJSON(),
      mixed.toJSON(),
      'edit',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it.each([
    ['table topology', (document: JSONContent) => {
      const table = document.content?.[0]
      const firstRow = table?.content?.[0]
      if (!table?.content || !firstRow) throw new Error('table fixture is invalid')
      table.content.push(structuredClone(firstRow))
    }],
    ['table cell attributes', (document: JSONContent) => {
      const cell = document.content?.[0]?.content?.[0]?.content?.[0]
      if (!cell) throw new Error('table fixture is invalid')
      cell.attrs = { ...cell.attrs, colspan: 2, colwidth: [160, 160] }
    }],
    ['paragraph attributes', (document: JSONContent) => {
      const paragraph = document.content?.[0]
      if (!paragraph) throw new Error('paragraph fixture is invalid')
      paragraph.attrs = { ...paragraph.attrs, textAlign: 'right' }
    }],
  ])('rejects non-reversible %s changes hidden from suggestion marks', (_label, mutate) => {
    const source = _label.startsWith('table')
      ? '| A | B |\n| --- | --- |\n| one | two |'
      : 'Unchanged text'
    const before = parseEditorMarkdown(source)
    const after = structuredClone(before)
    mutate(after)

    expect(() => validateSuggestionUpdate(
      before,
      after,
      'suggest',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })

  it('rejects a tracked math atom replacement even when its original projection is intact', () => {
    const inlineMath = schema.nodes.inlineMath
    const deletion = schema.marks.deletion
    const insertion = schema.marks.insertion
    if (!inlineMath || !deletion || !insertion) throw new Error('math fixture is invalid')
    const metadata = {
      authorId: USER_ID,
      createdAt: CREATED_AT,
      id: SUGGESTION_ID,
      patchId: PATCH_ID,
      suggestionId: SUGGESTION_ID,
    }
    const before = schema.node('doc', null, [schema.node('paragraph', null, [
      schema.text('Equation: '),
      inlineMath.create({ latex: 'x + 1' }),
      schema.text('.'),
    ])])
    const after = schema.node('doc', null, [schema.node('paragraph', null, [
      schema.text('Equation: '),
      inlineMath.create(
        { latex: 'x + 1' },
        null,
        [deletion.create({ ...metadata, kind: 'modification' })],
      ),
      inlineMath.create(
        { latex: 'x + 2' },
        null,
        [insertion.create({ ...metadata, kind: 'modification' })],
      ),
      schema.text('.'),
    ])])

    expect(projectOriginalDocument(after).eq(before)).toBe(true)

    expect(() => validateSuggestionUpdate(
      before.toJSON(),
      after.toJSON(),
      'suggest',
      USER_ID,
    )).toThrowError('suggestion_policy_violation')
  })
})

function validateMappedSuggestionUpdate(
  before: JSONContent,
  after: JSONContent,
  access: 'edit' | 'suggest' | 'view',
  actorUserId: string,
): ReturnType<typeof validateSuggestionUpdate> {
  const beforeDocument = editorJsonToYDoc(before)
  const afterDocument = new Y.Doc()
  Y.applyUpdate(afterDocument, Y.encodeStateAsUpdate(beforeDocument))
  const fragment = afterDocument.getXmlFragment(EDITOR_YJS_FRAGMENT)
  const initialized = initProseMirrorDoc(fragment, schema)
  updateYFragment(
    afterDocument,
    fragment,
    schema.nodeFromJSON(after),
    initialized.meta,
  )
  try {
    return validateSuggestionUpdate(before, after, access, actorUserId, {
      afterDocument,
      beforeDocument,
    })
  } finally {
    afterDocument.destroy()
    beforeDocument.destroy()
  }
}

function trackedInsertion(
  json: JSONContent,
  authorId: string,
  textOffset: number,
): ReturnType<typeof schema.nodeFromJSON> {
  const document = schema.nodeFromJSON(json)
  let textPosition: number | null = null
  document.descendants((node, position) => {
    if (textPosition === null && node.isText && node.text?.includes('abcabc')) {
      textPosition = position
    }
  })
  if (textPosition === null) throw new Error('suggestion text fixture is invalid')
  const state = EditorState.create({ schema, doc: document })
  return transformToInqtrixSuggestionTransaction(
    state.tr.insertText('X', textPosition + textOffset),
    state,
    { authorId, createdAt: CREATED_AT, patchId: PATCH_ID },
    () => SUGGESTION_ID,
  ).doc
}

function relocateSuggestionToParentEnd(document: JSONContent): JSONContent {
  const after = structuredClone(document)
  const parent = findSuggestionParent(after)
  if (!parent?.content) throw new Error('suggestion relocation fixture is invalid')
  const index = parent.content.findIndex((node) => hasSuggestion(node))
  if (index < 0) throw new Error('suggestion relocation fixture is invalid')
  const [occurrence] = parent.content.splice(index, 1)
  if (!occurrence) throw new Error('suggestion relocation fixture is invalid')
  parent.content.push(occurrence)
  return after
}

function findSuggestionParent(node: JSONContent): JSONContent | null {
  if (node.content?.some((child) => hasSuggestion(child))) return node
  for (const child of node.content ?? []) {
    const found = findSuggestionParent(child)
    if (found) return found
  }
  return null
}

function findSuggestionNode(node: JSONContent): JSONContent | null {
  if (hasSuggestion(node)) return node
  for (const child of node.content ?? []) {
    const found = findSuggestionNode(child)
    if (found) return found
  }
  return null
}

function hasSuggestion(node: JSONContent): boolean {
  return Boolean(node.marks?.some((mark) => mark.attrs?.suggestionId === SUGGESTION_ID))
}
