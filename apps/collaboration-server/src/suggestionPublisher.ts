import { randomUUID } from 'node:crypto'

import { applySuggestions } from '@handlewithcare/prosemirror-suggest-changes'
import { getSchema } from '@tiptap/core'
import type { Node as ProseMirrorNode } from '@tiptap/pm/model'
import { EditorState, type Transaction } from '@tiptap/pm/state'
import { initProseMirrorDoc, updateYFragment } from '@tiptap/y-tiptap'
import {
  EDITOR_YJS_FRAGMENT,
  UnsupportedSuggestionStructureError,
  createEditorSchemaExtensions,
  parseEditorMarkdown,
  serializeEditorJson,
  suggestionDescriptors,
  transformToInqtrixSuggestionTransaction,
  type SuggestionDescriptor,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import { CloseCodes, CollaborationError } from './errors'
import {
  collectSuggestionRecords,
  hasSameNonTextAtoms,
  isReversibleSuggestionStructure,
  sameSuggestionRecord,
  suggestionPositionAt,
} from './suggestPolicy'

export type PublishSuggestionInput = {
  actorUserId: string
  patchId: string
  targetMarkdown: string
}

export type PublishSuggestionResult = {
  patchIds: string[]
  suggestions: SuggestionDescriptor[]
  suggestionIds: string[]
}

export type SuggestionPublisherDependencies = {
  createSuggestionId?: () => string
  nowSeconds?: () => number
}

const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))

export function publishTargetSuggestion(
  document: Y.Doc,
  input: PublishSuggestionInput,
  dependencies: SuggestionPublisherDependencies = {},
): PublishSuggestionResult {
  const fragment = document.getXmlFragment(EDITOR_YJS_FRAGMENT)
  const initialized = initProseMirrorDoc(fragment, schema)
  const state = EditorState.create({ doc: initialized.doc, schema })
  const beforeDescriptors = suggestionDescriptors(state.doc)
  if (beforeDescriptors.some((descriptor) => descriptor.patchId === input.patchId)) {
    throw suggestionConflict()
  }

  const finalProjection = projectFinal(state)
  const target = parseRoundTrippableTarget(input.targetMarkdown)
  if (!hasSameNonTextAtoms(finalProjection.state.doc.toJSON(), target.toJSON())) {
    throw unsupportedStructure()
  }
  const canonicalTarget = serializeEditorJson(target.toJSON(), 'final')
  const currentFinal = serializeEditorJson(finalProjection.state.doc.toJSON(), 'final')
  if (canonicalTarget === currentFinal) throw suggestionConflict()

  let direct: Transaction
  try {
    const projectedTransaction = createSingleDiffTransaction(finalProjection.state, target)
    const projectedStep = projectedTransaction.steps[0]
    if (!projectedStep || projectedTransaction.steps.length !== 1) throw unsupportedStructure()
    const mappedStep = projectedStep.map(finalProjection.transaction.mapping.invert())
    if (!mappedStep) throw suggestionConflict()
    direct = state.tr.step(mappedStep)
  } catch (error) {
    if (error instanceof CollaborationError) throw error
    throw unsupportedStructure()
  }

  const createdAt = (dependencies.nowSeconds ?? unixSeconds)()
  if (!Number.isSafeInteger(createdAt) || createdAt < 0) {
    throw new CollaborationError('internal_consistency', {
      closeCode: CloseCodes.internalConsistency,
    })
  }

  let transformed: Transaction
  try {
    transformed = transformToInqtrixSuggestionTransaction(
      direct,
      state,
      {
        authorId: input.actorUserId,
        createdAt,
        patchId: input.patchId,
      },
      dependencies.createSuggestionId ?? randomUUID,
    )
  } catch (error) {
    if (error instanceof UnsupportedSuggestionStructureError) throw unsupportedStructure()
    throw suggestionConflict()
  }
  if (!transformed.docChanged) throw suggestionConflict()

  const next = state.apply(transformed)
  const afterDescriptors = suggestionDescriptors(next.doc)
  const created = afterDescriptors
    .filter((descriptor) => descriptor.patchId === input.patchId)
    .sort((left, right) => left.suggestionId.localeCompare(right.suggestionId))
  if (
    created.length === 0
    || created.some((descriptor) => (
      descriptor.authorId !== input.actorUserId
      || descriptor.createdAt !== createdAt
    ))
  ) {
    throw suggestionConflict()
  }

  assertExistingSuggestionsUnchanged(
    state.doc,
    next.doc,
    input.patchId,
    transformed,
  )
  if (
    !isReversibleSuggestionStructure(state.doc.toJSON(), next.doc.toJSON())
    || serializeEditorJson(next.doc.toJSON(), 'final') !== canonicalTarget
  ) {
    throw suggestionConflict()
  }

  updateYFragment(document, fragment, next.doc, initialized.meta)
  return {
    patchIds: [input.patchId],
    suggestions: created,
    suggestionIds: created.map((descriptor) => descriptor.suggestionId),
  }
}

function projectFinal(state: EditorState): {
  state: EditorState
  transaction: Transaction
} {
  let projection: Transaction | null = null
  const handled = applySuggestions(state, (transaction) => {
    projection = transaction
  })
  if (!handled || !projection) {
    throw new CollaborationError('internal_consistency', {
      closeCode: CloseCodes.internalConsistency,
    })
  }
  const transaction = projection as Transaction
  return { state: state.apply(transaction), transaction }
}

function parseRoundTrippableTarget(markdown: string): ProseMirrorNode {
  try {
    const target = schema.nodeFromJSON(parseEditorMarkdown(markdown))
    const canonical = serializeEditorJson(target.toJSON(), 'final')
    const reparsed = schema.nodeFromJSON(parseEditorMarkdown(canonical))
    if (!target.eq(reparsed)) throw invalidSchema()
    return target
  } catch (error) {
    if (error instanceof CollaborationError) throw error
    throw invalidSchema()
  }
}

function createSingleDiffTransaction(
  state: EditorState,
  target: ProseMirrorNode,
): Transaction {
  const start = state.doc.content.findDiffStart(target.content)
  const ends = state.doc.content.findDiffEnd(target.content)
  if (start === null || ends === null) throw suggestionConflict()
  let endCurrent = ends.a
  let endTarget = ends.b
  const overlap = start - Math.min(endCurrent, endTarget)
  if (overlap > 0) {
    endCurrent += overlap
    endTarget += overlap
  }
  const transaction = state.tr.replace(
    start,
    endCurrent,
    target.slice(start, endTarget),
  )
  if (!transaction.doc.eq(target)) throw unsupportedStructure()
  return transaction
}

function assertExistingSuggestionsUnchanged(
  before: ProseMirrorNode,
  after: ProseMirrorNode,
  newPatchId: string,
  transaction: Transaction,
): void {
  const beforeRecords = collectSuggestionRecords(before.toJSON())
  const afterRecords = collectSuggestionRecords(after.toJSON())
  for (const [id, record] of beforeRecords) {
    if (record.patchId === newPatchId) continue
    const next = afterRecords.get(id)
    if (!next || !sameSuggestionRecord(record, next, (position) => (
      suggestionPositionAt(
        after,
        transaction.mapping.map(position.from, 1),
        transaction.mapping.map(position.to, -1),
      )
    ))) throw suggestionConflict()
  }
  for (const [id, record] of afterRecords) {
    if (record.patchId === newPatchId) continue
    if (!beforeRecords.has(id)) throw suggestionConflict()
  }
}

function unixSeconds(): number {
  return Math.floor(Date.now() / 1_000)
}

function invalidSchema(): CollaborationError {
  return new CollaborationError('invalid_schema', {
    closeCode: CloseCodes.incompatible,
    httpStatus: 409,
  })
}

function suggestionConflict(): CollaborationError {
  return new CollaborationError('suggestion_conflict', {
    closeCode: CloseCodes.incompatible,
    httpStatus: 409,
  })
}

function unsupportedStructure(): CollaborationError {
  return new CollaborationError('unsupported_suggestion_structure', {
    closeCode: CloseCodes.incompatible,
    httpStatus: 409,
  })
}
