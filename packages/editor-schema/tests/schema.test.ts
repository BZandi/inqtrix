import { getSchema } from '@tiptap/core'
import type { Node as ProseMirrorNode } from '@tiptap/pm/model'
import { EditorState, TextSelection } from '@tiptap/pm/state'
import { initProseMirrorDoc, ySyncPluginKey } from '@tiptap/y-tiptap'
import * as Y from 'yjs'
import { describe, expect, it } from 'vitest'
import {
  EDITOR_BLOCK_SUGGESTIONS_SUPPORTED,
  UnsupportedSuggestionStructureError,
  createEditorSchemaExtensions,
  createRelativePositionAdapter,
  deserializeRelativePosition,
  editorCollaborationRoom,
  editorJsonToYDoc,
  editorSchemaDescriptor,
  editorYDocToJson,
  fingerprintEditorSchemaDescriptor,
  getEditorSchemaFingerprint,
  parseEditorCollaborationRoom,
  parseEditorMarkdown,
  projectFinalJson,
  projectOriginalJson,
  SUGGESTION_MARK_NAMES,
  serializeEditorJson,
  serializeRelativePosition,
  type SuggestionKind,
  suggestionDescriptors,
  transformToInqtrixSuggestionTransaction,
  validateCanonicalYjsV1Update,
  validateEditorYDoc,
  validateSuggestionYjsUpdate,
} from '../src/index'

function suggestionMarkKinds(document: ProseMirrorNode): string[] {
  const kinds = new Set<string>()
  document.descendants((node) => {
    for (const mark of node.marks) {
      if (SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind)) {
        kinds.add(mark.type.name)
      }
    }
  })
  return [...kinds].sort()
}

function suggestionDeclaredKinds(document: ProseMirrorNode): string[] {
  const kinds = new Set<string>()
  document.descendants((node) => {
    for (const mark of node.marks) {
      if (SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind)) {
        kinds.add(String(mark.attrs.kind))
      }
    }
  })
  return [...kinds].sort()
}

function firstParagraphXml(document: Y.Doc): Y.XmlElement {
  const paragraph = document.getXmlFragment('content').get(0)
  if (!(paragraph instanceof Y.XmlElement)) throw new Error('Expected a Yjs paragraph element')
  return paragraph
}

function firstParagraphXmlText(document: Y.Doc): Y.XmlText {
  const text = firstParagraphXml(document).get(0)
  if (!(text instanceof Y.XmlText)) throw new Error('Expected Yjs paragraph text')
  return text
}

function cloneYDoc(document: Y.Doc): Y.Doc {
  const clone = new Y.Doc()
  Y.applyUpdate(clone, Y.encodeStateAsUpdate(document))
  return clone
}

describe('editor schema gate', () => {
  it('round-trips one table-and-mathematics document through Markdown and Yjs', () => {
    const markdown = [
      '# Findings',
      '',
      '| Metric | Value |',
      '| --- | ---: |',
      '| Energy | 42 |',
      '',
      'Inline $E = mc^2$ remains inline.',
      '',
      '$$',
      '\\int_0^1 x^2 \\, dx',
      '$$',
      '',
    ].join('\n')

    const first = parseEditorMarkdown(markdown)
    const yDocument = editorJsonToYDoc(first)
    const restored = editorYDocToJson(yDocument)
    const serialized = serializeEditorJson(restored)
    const second = parseEditorMarkdown(serialized)

    expect(restored).toEqual(first)
    expect(second).toEqual(first)
    expect(serialized).toMatch(/\| Energy \| 42\s+\|/)
    expect(serialized).toContain('$E = mc^2$')
  })

  it('round-trips canonical JSON through the configured Yjs fragment', () => {
    const content = parseEditorMarkdown('## Shared\n\nA **collaborative** paragraph.\n')
    const document = editorJsonToYDoc(content)
    const replica = new Y.Doc()
    Y.applyUpdate(replica, Y.encodeStateAsUpdate(document))

    expect(editorYDocToJson(document)).toEqual(content)
    expect(validateEditorYDoc(replica)).toEqual(content)
    expect([...document.share.keys()]).toEqual(['content'])
    expect([...replica.share.keys()]).toEqual(['content'])
  })

  it('accepts only fully consumed canonical Yjs V1 update bytes', () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Canonical update'))
    const update = Y.encodeStateAsUpdate(document)
    const suffixed = new Uint8Array(update.byteLength + 3)
    suffixed.set(update)
    suffixed.set([0xde, 0xad, 0xbe], update.byteLength)
    const v2 = Y.encodeStateAsUpdateV2(document)

    expect(validateCanonicalYjsV1Update(update)).toBe(update)
    expect(() => validateCanonicalYjsV1Update(suffixed)).toThrow(/fully consumed/)
    expect(() => validateCanonicalYjsV1Update(v2)).toThrow(/canonical Yjs V1/)

    document.destroy()
  })

  it.each([
    ['an invisible XmlElement attribute', (document: Y.Doc) => {
      firstParagraphXml(document).setAttribute('rogue', 'hidden')
    }],
    ['an invisible XmlText attribute', (document: Y.Doc) => {
      firstParagraphXmlText(document).setAttribute('rogue', 'hidden')
    }],
    ['an empty extra XmlText node', (document: Y.Doc) => {
      const paragraph = firstParagraphXml(document)
      paragraph.insert(paragraph.length, [new Y.XmlText()])
    }],
  ] as const)('rejects %s that ProseMirror JSON drops', (_label, mutate) => {
    const content = parseEditorMarkdown('Canonical content')
    const document = editorJsonToYDoc(content)
    mutate(document)

    expect(editorYDocToJson(document)).toEqual(content)
    expect(() => validateEditorYDoc(document)).toThrow(/not canonical/)
    document.destroy()
  })

  it('rejects keyed state materialized onto the editor XmlFragment root', () => {
    const source = new Y.Doc()
    source.getMap('content').set('rogue', 'hidden')
    const document = new Y.Doc()
    Y.applyUpdate(document, Y.encodeStateAsUpdate(source))

    expect(() => validateEditorYDoc(document)).toThrow(/non-canonical structure/)

    document.destroy()
    source.destroy()
  })

  it('retains codec-owned suggestion attributes while ignoring deleted Yjs history', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const plain = schema.nodeFromJSON(parseEditorMarkdown('Hello'))
    const state = EditorState.create({ schema, doc: plain })
    const tracked = state.apply(transformToInqtrixSuggestionTransaction(
      state.tr.insertText('!', 6),
      state,
      {
        authorId: 'user-history',
        createdAt: 1_784_112_003,
        patchId: 'patch-history',
      },
      () => 'suggestion-history',
    )).doc
    const document = editorJsonToYDoc(tracked.toJSON())
    const text = firstParagraphXmlText(document)
    text.insert(text.length, '?')
    text.delete(text.length - 1, 1)
    text.setAttribute('transient', true)
    text.removeAttribute('transient')

    expect(validateEditorYDoc(document)).toEqual(tracked.toJSON())
    document.destroy()
  })

  it.each([
    ['struct dependency', 'struct'],
    ['delete dependency', 'delete'],
  ] as const)('rejects a document with an unresolved Yjs %s', (_label, kind) => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Causal'))
    const replica = cloneYDoc(document)
    const text = firstParagraphXmlText(replica)
    text.insert(text.length, 'missing')
    const dependencyVector = Y.encodeStateVector(replica)
    if (kind === 'struct') text.insert(text.length, '!')
    else text.delete(text.length - 'missing'.length, 'missing'.length)
    const incomplete = Y.encodeStateAsUpdate(replica, dependencyVector)
    const candidate = cloneYDoc(document)
    Y.applyUpdate(candidate, incomplete)

    expect(editorYDocToJson(candidate)).toEqual(editorYDocToJson(document))
    expect(() => validateEditorYDoc(candidate)).toThrow(/unresolved/)

    candidate.destroy()
    replica.destroy()
    document.destroy()
  })

  it.each([
    ['element attribute', (document: Y.Doc) => {
      const paragraph = firstParagraphXml(document)
      paragraph.setAttribute('rogue', 'x'.repeat(64 * 1024))
      paragraph.removeAttribute('rogue')
    }],
    ['text attribute', (document: Y.Doc) => {
      const text = firstParagraphXmlText(document)
      text.setAttribute('rogue', 'x'.repeat(64 * 1024))
      text.removeAttribute('rogue')
    }],
    ['text content', (document: Y.Doc) => {
      const text = firstParagraphXmlText(document)
      text.insert(1, 'x'.repeat(64 * 1024))
      text.delete(1, 64 * 1024)
    }],
    ['text format', (document: Y.Doc) => {
      const text = firstParagraphXmlText(document)
      text.format(0, text.length, { rogue: 'x'.repeat(64 * 1024) })
      text.format(0, text.length, { rogue: null })
    }],
  ] as const)('rejects transient raw Yjs %s history', (_label, mutate) => {
    const document = editorJsonToYDoc(parseEditorMarkdown('History'))
    const replica = cloneYDoc(document)
    const vector = Y.encodeStateVector(replica)
    mutate(replica)
    const update = Y.encodeStateAsUpdate(replica, vector)

    expect(validateEditorYDoc(replica)).toEqual(editorYDocToJson(document))
    expect(() => validateSuggestionYjsUpdate(update)).toThrow(/Suggestion update/)

    replica.destroy()
    document.destroy()
  })

  it('rejects a raw update for an unknown shared root', () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Root'))
    const replica = cloneYDoc(document)
    const vector = Y.encodeStateVector(replica)
    const rogue = replica.getMap('rogue')
    rogue.set('payload', 'hidden')
    rogue.delete('payload')
    const update = Y.encodeStateAsUpdate(replica, vector)

    expect(() => validateSuggestionYjsUpdate(update)).toThrow(/parent/)

    replica.destroy()
    document.destroy()
  })

  it('rejects update-internal tombstones when the sender disables Yjs GC', () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('No GC'))
    const replica = new Y.Doc({ gc: false })
    Y.applyUpdate(replica, Y.encodeStateAsUpdate(document))
    const vector = Y.encodeStateVector(replica)
    const text = firstParagraphXmlText(replica)
    text.insert(1, 'hidden')
    text.delete(1, 'hidden'.length)
    const update = Y.encodeStateAsUpdate(replica, vector)

    expect(() => validateSuggestionYjsUpdate(update)).toThrow(/transient/)

    replica.destroy()
    document.destroy()
  })

  it('allows resolved tombstones for existing content and repeat validation', () => {
    const document = editorJsonToYDoc(parseEditorMarkdown('Delete'))
    const replica = cloneYDoc(document)
    const vector = Y.encodeStateVector(replica)
    firstParagraphXmlText(replica).delete(0, 1)
    const update = Y.encodeStateAsUpdate(replica, vector)

    expect(validateSuggestionYjsUpdate(update)).toBe(update)
    expect(validateSuggestionYjsUpdate(update)).toBe(update)

    replica.destroy()
    document.destroy()
  })

  it('tracks UUID suggestions with authoritative metadata and projects both views', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const document = schema.node('doc', null, [schema.node('paragraph', null, schema.text('Hello'))])
    const state = EditorState.create({ schema, doc: document })
    const insertion = state.tr
      .setSelection(TextSelection.create(state.doc, 6))
      .insertText('!')
    const transformed = transformToInqtrixSuggestionTransaction(
      insertion,
      state,
      {
        authorId: 'user-1',
        createdAt: 1_784_112_000,
        patchId: '95bb8f85-5acd-4f50-a0cc-8af10a8e217b',
      },
      () => '7f8f4a45-867c-4551-92b5-593d7ff6801f',
    )
    const tracked = state.apply(transformed).doc

    expect(suggestionMarkKinds(tracked)).toEqual(['insertion'])
    expect(suggestionDescriptors(tracked)).toEqual([{
      authorId: 'user-1',
      createdAt: 1_784_112_000,
      kind: 'insertion',
      patchId: '95bb8f85-5acd-4f50-a0cc-8af10a8e217b',
      suggestionId: '7f8f4a45-867c-4551-92b5-593d7ff6801f',
    }])
    expect(projectFinalJson(tracked).content?.[0]?.content?.[0]?.text).toBe('Hello!')
    expect(projectOriginalJson(tracked).content?.[0]?.content?.[0]?.text).toBe('Hello')
  })

  it('models a text replacement as one modification with both semantic projections', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const document = schema.node('doc', null, [schema.node('paragraph', null, schema.text('Hello'))])
    const state = EditorState.create({ schema, doc: document })
    const replacement = state.tr.insertText('World', 1, 6)
    const transformed = transformToInqtrixSuggestionTransaction(
      replacement,
      state,
      {
        authorId: 'user-replacement',
        createdAt: 1_784_112_001,
        patchId: 'patch-replacement',
      },
      () => 'suggestion-replacement',
    )
    const tracked = state.apply(transformed).doc

    expect(suggestionMarkKinds(tracked)).toEqual(['deletion', 'insertion'])
    expect(suggestionDeclaredKinds(tracked)).toEqual(['modification'])
    expect(suggestionDescriptors(tracked)).toEqual([{
      authorId: 'user-replacement',
      createdAt: 1_784_112_001,
      kind: 'modification',
      patchId: 'patch-replacement',
      suggestionId: 'suggestion-replacement',
    }])
    expect(projectFinalJson(tracked).content?.[0]?.content?.[0]?.text).toBe('World')
    expect(projectOriginalJson(tracked).content?.[0]?.content?.[0]?.text).toBe('Hello')
    expect(serializeEditorJson(tracked.toJSON(), 'final').trim()).toBe('World')
    expect(serializeEditorJson(tracked.toJSON(), 'original').trim()).toBe('Hello')
    expect(editorYDocToJson(editorJsonToYDoc(tracked.toJSON()))).toEqual(tracked.toJSON())
  })

  it.each([
    ['distant ranges in one paragraph', false],
    ['ranges in separate paragraphs', true],
  ] as const)('rejects modification halves with %s', (_label, separateParents) => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const deletion = schema.marks.deletion
    const insertion = schema.marks.insertion
    if (!deletion || !insertion) throw new Error('Editor schema is missing suggestion marks')
    const metadata = {
      authorId: 'user-distant',
      createdAt: 1_784_112_004,
      id: 'suggestion-distant',
      kind: 'modification',
      patchId: 'patch-distant',
      suggestionId: 'suggestion-distant',
    }
    const deleted = schema.text('Before', [deletion.create(metadata)])
    const inserted = schema.text('After', [insertion.create(metadata)])
    const document = separateParents
      ? schema.node('doc', null, [
          schema.node('paragraph', null, deleted),
          schema.node('paragraph', null, inserted),
        ])
      : schema.node('doc', null, [schema.node('paragraph', null, [
          deleted,
          schema.text(' unrelated '),
          inserted,
        ])])

    expect(() => suggestionDescriptors(document)).toThrow(/adjacent replacement ranges/)
  })

  it('requires both replacement halves to declare modification semantics', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const deletion = schema.marks.deletion
    const insertion = schema.marks.insertion
    if (!deletion || !insertion) throw new Error('Editor schema is missing suggestion marks')
    const metadata = {
      authorId: 'user-semantic-pair',
      createdAt: 1_784_112_005,
      id: 'suggestion-semantic-pair',
      patchId: 'patch-semantic-pair',
      suggestionId: 'suggestion-semantic-pair',
    }
    const document = schema.node('doc', null, [schema.node('paragraph', null, [
      schema.text('Before', [deletion.create({ ...metadata, kind: 'deletion' })]),
      schema.text('After', [insertion.create({ ...metadata, kind: 'insertion' })]),
    ])])

    expect(() => suggestionDescriptors(document)).toThrow(/invalid mark composition/)
  })

  it('models inline formatting as one modification with both semantic projections', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const bold = schema.marks.bold
    if (!bold) throw new Error('Editor schema is missing the bold mark')
    const document = schema.node('doc', null, [schema.node('paragraph', null, schema.text('Hello'))])
    const state = EditorState.create({ schema, doc: document })
    const formatting = state.tr.addMark(1, 6, bold.create())
    const transformed = transformToInqtrixSuggestionTransaction(
      formatting,
      state,
      {
        authorId: 'user-formatting',
        createdAt: 1_784_112_002,
        patchId: 'patch-formatting',
      },
      () => 'suggestion-formatting',
    )
    const tracked = state.apply(transformed).doc

    expect(suggestionMarkKinds(tracked)).toEqual(['deletion', 'insertion'])
    expect(suggestionDeclaredKinds(tracked)).toEqual(['modification'])
    expect(suggestionDescriptors(tracked)).toEqual([{
      authorId: 'user-formatting',
      createdAt: 1_784_112_002,
      kind: 'modification',
      patchId: 'patch-formatting',
      suggestionId: 'suggestion-formatting',
    }])
    expect(serializeEditorJson(tracked.toJSON(), 'final').trim()).toBe('**Hello**')
    expect(serializeEditorJson(tracked.toJSON(), 'original').trim()).toBe('Hello')
  })

  it('rejects conflicting or spoofed authoritative metadata', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const deletion = schema.marks.deletion
    const insertion = schema.marks.insertion
    if (!deletion || !insertion) throw new Error('Editor schema is missing suggestion marks')
    const sharedAttributes = {
      createdAt: 1_784_112_003,
      id: 'suggestion-conflict',
      patchId: 'patch-conflict',
      suggestionId: 'suggestion-conflict',
    }
    const document = schema.node('doc', null, [
      schema.node('paragraph', null, [
        schema.text('Before', [deletion.create({
          ...sharedAttributes,
          authorId: 'user-one',
          kind: 'deletion',
        })]),
        schema.text('After', [insertion.create({
          ...sharedAttributes,
          authorId: 'user-two',
          kind: 'insertion',
        })]),
      ]),
    ])

    expect(() => suggestionDescriptors(document)).toThrow(/inconsistent metadata/)

    const spoofedKind = schema.node('doc', null, [
      schema.node('paragraph', null, [
        schema.text('After', [insertion.create({
          ...sharedAttributes,
          authorId: 'user-one',
          kind: 'deletion',
        })]),
      ]),
    ])
    expect(() => suggestionDescriptors(spoofedKind)).toThrow(/invalid mark composition/)

    const partialModification = schema.node('doc', null, [
      schema.node('paragraph', null, [
        schema.text('Before', [deletion.create({
          ...sharedAttributes,
          authorId: 'user-one',
          kind: 'modification',
        })]),
      ]),
    ])
    expect(() => suggestionDescriptors(partialModification)).toThrow(
      /invalid mark composition/,
    )
  })

  it('never transforms remote Yjs transactions into local suggestions', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const document = schema.node('doc', null, [schema.node('paragraph', null, schema.text('A'))])
    const state = EditorState.create({ schema, doc: document })
    const remote = state.tr.insertText('B', 2).setMeta(ySyncPluginKey, { isChangeOrigin: true })

    const transformed = transformToInqtrixSuggestionTransaction(
      remote,
      state,
      { authorId: 'user-1', createdAt: 1, patchId: 'patch-1' },
      () => 'suggestion-1',
    )

    expect(transformed).toBe(remote)
    expect(state.apply(transformed).doc.textContent).toBe('AB')
  })

  it('rejects block insertions before the Yjs codec can drop their marks', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const document = schema.node('doc', null, [schema.node('paragraph', null, schema.text('First'))])
    const state = EditorState.create({ schema, doc: document })
    const insertedParagraph = schema.node('paragraph', null, schema.text('Second'))
    const insertion = state.tr.insert(state.doc.content.size, insertedParagraph)
    expect(EDITOR_BLOCK_SUGGESTIONS_SUPPORTED).toBe(false)
    expect(() => transformToInqtrixSuggestionTransaction(
      insertion,
      state,
      { authorId: 'user-1', createdAt: 2, patchId: 'patch-block' },
      () => 'suggestion-block',
    )).toThrow(UnsupportedSuggestionStructureError)
  })

  it('rejects block deletions at the document boundary', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const document = schema.node('doc', null, [
      schema.node('paragraph', null, schema.text('First')),
      schema.node('paragraph', null, schema.text('Second')),
    ])
    const state = EditorState.create({ schema, doc: document })
    const deletion = state.tr.delete(
      state.doc.child(0).nodeSize,
      state.doc.content.size,
    )
    expect(() => transformToInqtrixSuggestionTransaction(
      deletion,
      state,
      { authorId: 'user-1', createdAt: 3, patchId: 'patch-delete-block' },
      () => 'suggestion-delete-block',
    )).toThrow(UnsupportedSuggestionStructureError)
  })

  it('rejects block suggestion JSON instead of lossy Yjs round-tripping it', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const insertion = schema.marks.insertion
    if (!insertion) throw new Error('Editor schema is missing insertion suggestions')
    const document = schema.nodeFromJSON({
      type: 'doc',
      content: [{
        type: 'paragraph',
        marks: [{
          type: 'insertion',
          attrs: {
            authorId: 'user-1',
            createdAt: 4,
            id: 'suggestion-block-codec',
            kind: 'insertion',
            patchId: 'patch-block-codec',
            suggestionId: 'suggestion-block-codec',
          },
        }],
        content: [{ type: 'text', text: 'Second' }],
      }],
    })

    expect(schema.nodes.doc?.spec.marks).not.toBe('insertion modification deletion')
    expect(() => editorJsonToYDoc(document.toJSON())).toThrow(
      UnsupportedSuggestionStructureError,
    )
  })

  it('rejects suggestion marks on the document root before Yjs conversion', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const document = schema.nodeFromJSON({
      type: 'doc',
      marks: [{
        type: 'insertion',
        attrs: {
          authorId: 'user-root',
          createdAt: 5,
          id: 'suggestion-root',
          kind: 'insertion',
          patchId: 'patch-root',
          suggestionId: 'suggestion-root',
        },
      }],
      content: [{
        type: 'paragraph',
        content: [{ type: 'text', text: 'Root' }],
      }],
    })

    expect(() => editorJsonToYDoc(document.toJSON())).toThrow(
      UnsupportedSuggestionStructureError,
    )
  })

  it('rejects suggestion marks on inline math atoms before Yjs conversion', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const insertion = schema.marks.insertion
    if (!insertion) throw new Error('Editor schema is missing insertion suggestions')
    const inlineMath = schema.node(
      'inlineMath',
      { latex: 'x^2' },
      undefined,
      [insertion.create({
        authorId: 'user-1',
        createdAt: 6,
        id: 'suggestion-inline-math',
        kind: 'insertion',
        patchId: 'patch-inline-math',
        suggestionId: 'suggestion-inline-math',
      })],
    )
    const document = schema.node('doc', null, [schema.node('paragraph', null, [inlineMath])])

    expect(() => editorJsonToYDoc(document.toJSON())).toThrow(
      UnsupportedSuggestionStructureError,
    )
  })

  it('serializes Yjs relative positions and keeps them stable across concurrent inserts', () => {
    const source = new Y.Doc()
    const text = source.getText('anchor')
    text.insert(0, 'abcd')
    const encoded = serializeRelativePosition(Y.createRelativePositionFromTypeIndex(text, 2))

    const replica = new Y.Doc()
    Y.applyUpdate(replica, Y.encodeStateAsUpdate(source))
    replica.getText('anchor').insert(0, 'XX')
    Y.applyUpdate(source, Y.encodeStateAsUpdate(replica, Y.encodeStateVector(source)))

    const absolute = Y.createAbsolutePositionFromRelativePosition(
      deserializeRelativePosition(encoded),
      source,
    )
    expect(absolute?.index).toBe(4)
  })

  it('adapts real y-tiptap positions across transformer-backed Yjs edits', () => {
    const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))
    const content = parseEditorMarkdown('abcd')
    const document = editorJsonToYDoc(content)
    const fragment = document.getXmlFragment('content')
    const initialized = initProseMirrorDoc(fragment, schema)
    const adapter = createRelativePositionAdapter(document, fragment, initialized.mapping)
    const encoded = adapter.fromProseMirrorPosition(3)

    expect(initialized.doc.toJSON()).toEqual(content)
    expect(adapter.toProseMirrorPosition(encoded)).toBe(3)

    const paragraph = fragment.get(0)
    const text = paragraph instanceof Y.XmlElement ? paragraph.get(0) : null
    if (!(text instanceof Y.XmlText)) throw new Error('Expected transformer-backed Yjs text')
    text.insert(0, 'XX')

    const remapped = initProseMirrorDoc(fragment, schema)
    const remappedAdapter = createRelativePositionAdapter(document, fragment, remapped.mapping)
    expect(remapped.doc.textContent).toBe('XXabcd')
    expect(remappedAdapter.toProseMirrorPosition(encoded)).toBe(5)

    expect(() => createRelativePositionAdapter(
      new Y.Doc(),
      fragment,
      remapped.mapping,
    )).toThrow(/must belong to the supplied Y.Doc/)
  })

  it('fails closed for unexpected or type-confused Yjs shared roots', () => {
    const content = parseEditorMarkdown('Canonical content')
    const unexpectedRoot = editorJsonToYDoc(content)
    unexpectedRoot.getMap('rogue').set('payload', 'not editor content')

    expect(() => validateEditorYDoc(unexpectedRoot)).toThrow(/only the content shared root/)

    const wrongContentType = new Y.Doc()
    wrongContentType.getMap('content').set('payload', 'not an XmlFragment')

    expect(() => validateEditorYDoc(wrongContentType)).toThrow(/must be an XmlFragment/)

    const encodedWrongType = new Y.Doc()
    Y.applyUpdate(encodedWrongType, Y.encodeStateAsUpdate(wrongContentType))
    expect(() => validateEditorYDoc(encodedWrongType)).toThrow(/non-canonical structure/)

    const textRoot = new Y.Doc()
    textRoot.getText('content').insert(0, 'not an XmlFragment')
    const encodedTextRoot = new Y.Doc()
    Y.applyUpdate(encodedTextRoot, Y.encodeStateAsUpdate(textRoot))
    expect(() => validateEditorYDoc(encodedTextRoot)).toThrow(/non-canonical structure/)

    const emptyContent = new Y.Doc()
    emptyContent.getXmlFragment('content')
    expect(() => validateEditorYDoc(emptyContent)).toThrow()
  })

  it('produces a deterministic schema fingerprint and strict room names', async () => {
    await expect(getEditorSchemaFingerprint()).resolves.toMatch(/^[a-f0-9]{64}$/)
    await expect(getEditorSchemaFingerprint()).resolves.toBe(await getEditorSchemaFingerprint())
    expect(editorCollaborationRoom('ed_123', 4)).toBe('inqtrix-editor-v1:ed_123:g4')
    expect(parseEditorCollaborationRoom('inqtrix-editor-v1:ed_123:g4')).toEqual({
      documentId: 'ed_123',
      generation: 4,
    })
    expect(parseEditorCollaborationRoom('inqtrix-editor-v1:ed_123:g0')).toBeNull()
    expect(editorCollaborationRoom('editor-doc-123', 1)).toBe(
      'inqtrix-editor-v1:editor-doc-123:g1',
    )
  })

  it('changes the fingerprint for schema order, rank, shape, and behavior inputs', async () => {
    const descriptor = editorSchemaDescriptor()
    const baseline = await fingerprintEditorSchemaDescriptor(descriptor)
    await expect(getEditorSchemaFingerprint()).resolves.toBe(baseline)

    const reorderedNodes = structuredClone(descriptor)
    const firstNode = reorderedNodes.nodes[0]
    const secondNode = reorderedNodes.nodes[1]
    if (!firstNode || !secondNode) throw new Error('Schema must contain at least two nodes')
    reorderedNodes.nodes.splice(0, 2, secondNode, firstNode)
    reorderedNodes.nodes.forEach((node, order) => { node.order = order })

    const reorderedMarks = structuredClone(descriptor)
    const firstMark = reorderedMarks.marks[0]
    const secondMark = reorderedMarks.marks[1]
    if (!firstMark || !secondMark) throw new Error('Schema must contain at least two marks')
    reorderedMarks.marks.splice(0, 2, secondMark, firstMark)
    reorderedMarks.marks.forEach((mark, order) => {
      mark.order = order
      mark.rank = order
    })

    const changedRank = structuredClone(descriptor)
    const rankedMark = changedRank.marks[0]
    if (!rankedMark || rankedMark.rank === null) throw new Error('Schema must contain ranked marks')
    rankedMark.rank += 1

    const changedShape = structuredClone(descriptor)
    const paragraph = changedShape.nodes.find((node) => node.name === 'paragraph')
    if (!paragraph) throw new Error('Schema must contain a paragraph node')
    paragraph.spec.content = 'text*'

    const changedBehavior = {
      ...descriptor,
      behavior: {
        ...descriptor.behavior,
        suggestionTransform: 'changed-suggestion-behavior',
      },
    }

    const changedVersion = {
      ...descriptor,
      dependencies: {
        ...descriptor.dependencies,
        tiptap: '3.27.2',
      },
    }

    const mutations = await Promise.all([
      fingerprintEditorSchemaDescriptor(reorderedNodes),
      fingerprintEditorSchemaDescriptor(reorderedMarks),
      fingerprintEditorSchemaDescriptor(changedRank),
      fingerprintEditorSchemaDescriptor(changedShape),
      fingerprintEditorSchemaDescriptor(changedBehavior),
      fingerprintEditorSchemaDescriptor(changedVersion),
    ])
    expect(new Set(mutations).size).toBe(mutations.length)
    for (const mutation of mutations) expect(mutation).not.toBe(baseline)
  })
})
