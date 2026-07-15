import { getSchema } from '@tiptap/core'
import { EditorState } from '@tiptap/pm/state'
import {
  createEditorSchemaExtensions,
  editorJsonToYDoc,
  editorYDocToJson,
  parseEditorMarkdown,
  serializeEditorJson,
  transformToInqtrixSuggestionTransaction,
} from '@inqtrix/editor-schema'

import { publishTargetSuggestion } from '../src/suggestionPublisher'
import { USER_ID } from './helpers'

const PATCH_ID = '22222222-2222-4222-8222-222222222222'
const SUGGESTION_ID = '33333333-3333-4333-8333-333333333333'
const EXISTING_PATCH_ID = '44444444-4444-4444-8444-444444444444'
const EXISTING_SUGGESTION_ID = '55555555-5555-4555-8555-555555555555'
const OTHER_USER_ID = '66666666-6666-4666-8666-666666666666'
const CREATED_AT = 1_784_112_000
const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))

describe('target Markdown suggestion publication', () => {
  it('creates a server-attributed suggestion without replacing the Y.Doc', () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))

    const result = publishTargetSuggestion(document, {
      actorUserId: USER_ID,
      patchId: PATCH_ID,
      targetMarkdown: 'Hello world',
    }, {
      createSuggestionId: () => SUGGESTION_ID,
      nowSeconds: () => CREATED_AT,
    })

    const json = editorYDocToJson(document)
    expect(serializeEditorJson(json, 'original').trim()).toBe('Hello')
    expect(serializeEditorJson(json, 'final').trim()).toBe('Hello world')
    expect(result).toEqual({
      patchIds: [PATCH_ID],
      suggestionIds: [SUGGESTION_ID],
      suggestions: [{
        authorId: USER_ID,
        createdAt: CREATED_AT,
        kind: 'insertion',
        patchId: PATCH_ID,
        suggestionId: SUGGESTION_ID,
      }],
    })
  })

  it('preserves an existing open suggestion outside the target diff', () => {
    const plain = schema.nodeFromJSON(parseEditorMarkdown('Hello\n\nSecond'))
    const state = EditorState.create({ schema, doc: plain })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', 6),
      state,
      {
        authorId: USER_ID,
        createdAt: CREATED_AT - 10,
        patchId: EXISTING_PATCH_ID,
      },
      () => EXISTING_SUGGESTION_ID,
    ).doc
    const document = editorJsonToYDoc(tracked.toJSON())

    publishTargetSuggestion(document, {
      actorUserId: USER_ID,
      patchId: PATCH_ID,
      targetMarkdown: 'Hello!\n\nChanged',
    }, {
      createSuggestionId: () => SUGGESTION_ID,
      nowSeconds: () => CREATED_AT,
    })

    const json = editorYDocToJson(document)
    expect(serializeEditorJson(json, 'original').trim()).toBe('Hello\n\nSecond')
    expect(serializeEditorJson(json, 'final').trim()).toBe('Hello!\n\nChanged')
    expect(JSON.stringify(json)).toContain(EXISTING_SUGGESTION_ID)
    expect(JSON.stringify(json)).toContain(SUGGESTION_ID)
  })

  it('publishes a direct edit before an unchanged foreign suggestion', () => {
    const plain = schema.nodeFromJSON(parseEditorMarkdown('First\n\nSecond'))
    const state = EditorState.create({ schema, doc: plain })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', state.doc.content.size - 1),
      state,
      {
        authorId: OTHER_USER_ID,
        createdAt: CREATED_AT - 10,
        patchId: EXISTING_PATCH_ID,
      },
      () => EXISTING_SUGGESTION_ID,
    ).doc
    const document = editorJsonToYDoc(tracked.toJSON())

    publishTargetSuggestion(document, {
      actorUserId: USER_ID,
      patchId: PATCH_ID,
      targetMarkdown: 'Changed first\n\nSecond!',
    }, {
      createSuggestionId: () => SUGGESTION_ID,
      nowSeconds: () => CREATED_AT,
    })

    const json = editorYDocToJson(document)
    expect(serializeEditorJson(json, 'final').trim()).toBe('Changed first\n\nSecond!')
    expect(JSON.stringify(json)).toContain(EXISTING_SUGGESTION_ID)
    expect(JSON.stringify(json)).toContain(SUGGESTION_ID)
  })

  it('publishes before an unchanged foreign suggestion in the same paragraph', () => {
    const plain = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: plain })
    const tracked = transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', state.doc.content.size - 1),
      state,
      {
        authorId: OTHER_USER_ID,
        createdAt: CREATED_AT - 10,
        patchId: EXISTING_PATCH_ID,
      },
      () => EXISTING_SUGGESTION_ID,
    ).doc
    const document = editorJsonToYDoc(tracked.toJSON())

    publishTargetSuggestion(document, {
      actorUserId: USER_ID,
      patchId: PATCH_ID,
      targetMarkdown: 'Prefix Hello!',
    }, {
      createSuggestionId: () => SUGGESTION_ID,
      nowSeconds: () => CREATED_AT,
    })

    const json = editorYDocToJson(document)
    expect(serializeEditorJson(json, 'final').trim()).toBe('Prefix Hello!')
    expect(JSON.stringify(json)).toContain(EXISTING_SUGGESTION_ID)
    expect(JSON.stringify(json)).toContain(SUGGESTION_ID)
  })

  it('rejects a no-op target rather than creating an empty patch', () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Hello'))

    expect(() => publishTargetSuggestion(document, {
      actorUserId: USER_ID,
      patchId: PATCH_ID,
      targetMarkdown: 'Hello',
    })).toThrowError('suggestion_conflict')
  })

  it('rejects a target that would replace a math atom', () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Equation: $x + 1$.'))

    expect(() => publishTargetSuggestion(document, {
      actorUserId: USER_ID,
      patchId: PATCH_ID,
      targetMarkdown: 'Equation: $x + 2$.',
    })).toThrowError('unsupported_suggestion_structure')
  })

  it('rejects a paragraph insertion that the Yjs codec cannot preserve', () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('First'))

    expect(() => publishTargetSuggestion(document, {
      actorUserId: USER_ID,
      patchId: PATCH_ID,
      targetMarkdown: 'First\n\nSecond',
    })).toThrowError('unsupported_suggestion_structure')
    expect(serializeEditorJson(editorYDocToJson(document), 'final').trim()).toBe('First')
  })
})
