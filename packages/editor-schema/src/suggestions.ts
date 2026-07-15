import {
  applySuggestions,
  revertSuggestions,
  suggestChangesKey,
  transformToSuggestionTransaction,
} from '@handlewithcare/prosemirror-suggest-changes'
import type { JSONContent } from '@tiptap/core'
import type { Mark, Node as ProseMirrorNode } from '@tiptap/pm/model'
import { EditorState, type Transaction } from '@tiptap/pm/state'
import type { StepMap } from '@tiptap/pm/transform'
import { ySyncPluginKey } from '@tiptap/y-tiptap'
import { SUGGESTION_MARK_NAMES, type SuggestionKind } from './suggestionMarks.js'

export const INQTRIX_SUGGESTION_BYPASS_META = 'inqtrixSuggestionBypass'

export class UnsupportedSuggestionStructureError extends Error {
  readonly code = 'unsupported_suggestion_structure'

  constructor(message: string) {
    super(message)
    this.name = 'UnsupportedSuggestionStructureError'
  }
}

export type SuggestionMetadata = {
  authorId: string
  createdAt: number
  patchId: string
}

export type SuggestionDescriptor = SuggestionMetadata & {
  kind: SuggestionKind
  suggestionId: string
}

type SuggestionMarkDescriptor = SuggestionMetadata & {
  declaredKind: SuggestionKind
  markKind: SuggestionKind
  suggestionId: string
}

type SuggestionDescriptorAccumulator = SuggestionMetadata & {
  declarations: Array<{
    declaredKind: SuggestionKind
    markKind: SuggestionKind
  }>
  markKinds: Set<SuggestionKind>
  occurrences: SuggestionMarkOccurrence[]
  suggestionId: string
}

type SuggestionMarkOccurrence = {
  from: number
  markKind: SuggestionKind
  parent: ProseMirrorNode
  to: number
}

type MarkLocation = {
  from: number
  mark: Mark
  nodePosition: number | null
  to: number
}

export function isRemoteYjsTransaction(transaction: Transaction): boolean {
  const meta = yjsTransactionMetadata(transaction)
  return meta?.isChangeOrigin === true
}

export function isYjsUndoRedoTransaction(transaction: Transaction): boolean {
  return yjsTransactionMetadata(transaction)?.isUndoRedoOperation === true
}

function yjsTransactionMetadata(transaction: Transaction): {
  isChangeOrigin?: boolean
  isUndoRedoOperation?: boolean
} | undefined {
  return transaction.getMeta(ySyncPluginKey) as {
    isChangeOrigin?: boolean
    isUndoRedoOperation?: boolean
  } | undefined
}

function hasSuggestionCommandBypass(transaction: Transaction): boolean {
  const metadata = transaction.getMeta(suggestChangesKey) as { skip?: unknown } | undefined
  return metadata !== undefined && Object.prototype.hasOwnProperty.call(metadata, 'skip')
}

export function shouldBypassSuggestionTransform(transaction: Transaction): boolean {
  return (
    !transaction.docChanged
    || transaction.getMeta(INQTRIX_SUGGESTION_BYPASS_META) === true
    || isRemoteYjsTransaction(transaction)
    || isYjsUndoRedoTransaction(transaction)
    || hasSuggestionCommandBypass(transaction)
  )
}

export function transformToInqtrixSuggestionTransaction(
  transaction: Transaction,
  state: EditorState,
  metadata: SuggestionMetadata,
  generateSuggestionId: () => string = () => crypto.randomUUID(),
): Transaction {
  if (shouldBypassSuggestionTransform(transaction)) return transaction
  assertSuggestionTransactionSupported(transaction)

  const generatedIds = new Set<string>()
  let transformed: Transaction
  try {
    transformed = transformToSuggestionTransaction(transaction, state, () => {
      const id = generateSuggestionId()
      generatedIds.add(id)
      return id
    })
  } catch (error) {
    if (
      error instanceof Error
      && error.name === 'TransformError'
      && error.message.startsWith('Invalid content for node ')
    ) {
      throw new UnsupportedSuggestionStructureError(
        'The suggestion transform requires node marks that the collaborative schema does not support',
      )
    }
    throw error
  }
  assertEveryGeneratedSuggestionIsRepresented(transformed.doc, generatedIds)
  assertYjsCompatibleSuggestionStructure(transformed.doc)
  return enrichGeneratedSuggestionMarks(transformed, generatedIds, metadata)
}

function assertEveryGeneratedSuggestionIsRepresented(
  document: ProseMirrorNode,
  generatedIds: ReadonlySet<string>,
): void {
  const represented = new Set<string>()
  document.descendants((node) => {
    for (const mark of node.marks) {
      const id = mark.attrs.id
      if (typeof id === 'string' && generatedIds.has(id)) represented.add(id)
    }
  })
  if ([...generatedIds].some((id) => !represented.has(id))) {
    throw new UnsupportedSuggestionStructureError(
      'The suggestion transform produced a structural change without a persistent mark',
    )
  }
}

function assertSuggestionTransactionSupported(transaction: Transaction): void {
  for (let index = 0; index < transaction.steps.length; index += 1) {
    const before = transaction.docs[index]
    const after = transaction.docs[index + 1] ?? transaction.doc
    const step = transaction.steps[index]
    if (!before || !step) continue
    for (const table of tableNodes(before)) {
      const mappedTable = mappedTableNode(after, table.position, step.getMap())
      if (mappedTable && !sameTableTopology(table.node, mappedTable)) {
        throw new UnsupportedSuggestionStructureError(
          'Suggestions cannot change rows, columns, merged cells, or split cells',
        )
      }
    }
  }
}

function tableNodes(document: ProseMirrorNode): Array<{
  node: ProseMirrorNode
  position: number
}> {
  const tables: Array<{ node: ProseMirrorNode; position: number }> = []
  document.descendants((node, position) => {
    if (node.type.name === 'table') tables.push({ node, position })
    return node.type.name !== 'table'
  })
  return tables
}

function mappedTableNode(
  document: ProseMirrorNode,
  position: number,
  mapping: StepMap,
): ProseMirrorNode | null {
  for (const association of [1, -1]) {
    const mapped = mapping.mapResult(position, association)
    if (mapped.deletedAcross) continue
    const node = document.nodeAt(mapped.pos)
    if (node?.type.name === 'table') return node
  }
  return null
}

function sameTableTopology(left: ProseMirrorNode, right: ProseMirrorNode): boolean {
  if (left.childCount !== right.childCount) return false
  for (let rowIndex = 0; rowIndex < left.childCount; rowIndex += 1) {
    const leftRow = left.child(rowIndex)
    const rightRow = right.child(rowIndex)
    if (leftRow.type.name !== rightRow.type.name || leftRow.childCount !== rightRow.childCount) {
      return false
    }
    for (let cellIndex = 0; cellIndex < leftRow.childCount; cellIndex += 1) {
      const leftCell = leftRow.child(cellIndex)
      const rightCell = rightRow.child(cellIndex)
      if (
        leftCell.type.name !== rightCell.type.name
        || leftCell.attrs.colspan !== rightCell.attrs.colspan
        || leftCell.attrs.rowspan !== rightCell.attrs.rowspan
        || JSON.stringify(leftCell.attrs.colwidth) !== JSON.stringify(rightCell.attrs.colwidth)
      ) {
        return false
      }
    }
  }
  return true
}

function enrichGeneratedSuggestionMarks(
  transaction: Transaction,
  generatedIds: ReadonlySet<string>,
  metadata: SuggestionMetadata,
): Transaction {
  const locations: MarkLocation[] = []
  const markKinds = new Map<string, Set<SuggestionKind>>()
  transaction.doc.descendants((node, position) => {
    for (const mark of node.marks) {
      const id = mark.attrs.id
      if (typeof id !== 'string' || !generatedIds.has(id)) continue
      const kind = mark.type.name as SuggestionKind
      const kinds = markKinds.get(id) ?? new Set<SuggestionKind>()
      kinds.add(kind)
      markKinds.set(id, kinds)
      locations.push({
        from: position,
        mark,
        nodePosition: node.isText ? null : position,
        to: position + node.nodeSize,
      })
    }
  })

  for (const location of locations) {
    const id = String(location.mark.attrs.id)
    const kinds = markKinds.get(id)
    if (!kinds) throw new Error(`Suggestion ${id} has no generated mark composition`)
    const kind = resolveSuggestionKind(id, kinds)
    const enriched = location.mark.type.create({
      ...location.mark.attrs,
      authorId: metadata.authorId,
      createdAt: metadata.createdAt,
      id,
      kind,
      patchId: metadata.patchId,
      suggestionId: id,
    })
    if (location.nodePosition === null) {
      transaction.removeMark(location.from, location.to, location.mark)
      transaction.addMark(location.from, location.to, enriched)
      continue
    }
    transaction.removeNodeMark(location.nodePosition, location.mark)
    transaction.addNodeMark(location.nodePosition, enriched)
  }
  return transaction
}

export function suggestionDescriptors(document: ProseMirrorNode): SuggestionDescriptor[] {
  const descriptors = new Map<string, SuggestionDescriptorAccumulator>()
  document.descendants((node, position, parent) => {
    for (const mark of node.marks) {
      if (!SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind)) continue
      const descriptor = suggestionMarkDescriptor(mark)
      if (!parent) throw new Error('Suggestion mark has no structural parent')
      const occurrence: SuggestionMarkOccurrence = {
        from: position,
        markKind: descriptor.markKind,
        parent,
        to: position + node.nodeSize,
      }
      const existing = descriptors.get(descriptor.suggestionId)
      if (existing && !sameSuggestionMetadata(existing, descriptor)) {
        throw new Error(`Suggestion ${descriptor.suggestionId} has inconsistent metadata`)
      }
      if (existing) {
        existing.declarations.push({
          declaredKind: descriptor.declaredKind,
          markKind: descriptor.markKind,
        })
        existing.markKinds.add(descriptor.markKind)
        existing.occurrences.push(occurrence)
        continue
      }
      descriptors.set(descriptor.suggestionId, {
        authorId: descriptor.authorId,
        createdAt: descriptor.createdAt,
        declarations: [{
          declaredKind: descriptor.declaredKind,
          markKind: descriptor.markKind,
        }],
        markKinds: new Set([descriptor.markKind]),
        occurrences: [occurrence],
        patchId: descriptor.patchId,
        suggestionId: descriptor.suggestionId,
      })
    }
  })
  return [...descriptors.values()]
    .map((descriptor) => {
      const kind = resolveSuggestionKind(descriptor.suggestionId, descriptor.markKinds)
      if (!hasValidKindDeclarations(kind, descriptor.declarations)) {
        throw new Error(`Suggestion ${descriptor.suggestionId} has an invalid mark composition`)
      }
      if (
        kind === 'modification'
        && descriptor.markKinds.has('deletion')
        && descriptor.markKinds.has('insertion')
      ) {
        assertAdjacentModificationPair(descriptor.suggestionId, descriptor.occurrences)
      }
      return {
        authorId: descriptor.authorId,
        createdAt: descriptor.createdAt,
        kind,
        patchId: descriptor.patchId,
        suggestionId: descriptor.suggestionId,
      }
    })
    .sort((left, right) => (
      left.createdAt - right.createdAt || left.suggestionId.localeCompare(right.suggestionId)
    ))
}

function suggestionMarkDescriptor(mark: Mark): SuggestionMarkDescriptor {
  const markKind = mark.type.name as SuggestionKind
  const { authorId, createdAt, id, kind, patchId, suggestionId } = mark.attrs
  if (
    !SUGGESTION_MARK_NAMES.has(markKind)
    || typeof id !== 'string'
    || id.length === 0
    || typeof suggestionId !== 'string'
    || suggestionId.length === 0
    || id !== suggestionId
    || !SUGGESTION_MARK_NAMES.has(kind as SuggestionKind)
    || typeof authorId !== 'string'
    || authorId.length === 0
    || typeof createdAt !== 'number'
    || !Number.isFinite(createdAt)
    || typeof patchId !== 'string'
    || patchId.length === 0
  ) {
    throw new Error('Suggestion mark is missing authoritative metadata')
  }
  return {
    authorId,
    createdAt,
    declaredKind: kind as SuggestionKind,
    markKind,
    patchId,
    suggestionId,
  }
}

function hasValidKindDeclarations(
  resolvedKind: SuggestionKind,
  declarations: ReadonlyArray<{
    declaredKind: SuggestionKind
    markKind: SuggestionKind
  }>,
): boolean {
  return declarations.every(({ declaredKind }) => declaredKind === resolvedKind)
}

function assertAdjacentModificationPair(
  suggestionId: string,
  occurrences: readonly SuggestionMarkOccurrence[],
): void {
  const deletions = contiguousSuggestionRanges(occurrences, 'deletion')
  const insertions = contiguousSuggestionRanges(occurrences, 'insertion')
  const deletion = deletions[0]
  const insertion = insertions[0]
  if (
    deletions.length !== 1
    || insertions.length !== 1
    || !deletion
    || !insertion
    || deletion.parent !== insertion.parent
    || (deletion.to !== insertion.from && insertion.to !== deletion.from)
  ) {
    throw new Error(
      `Suggestion ${suggestionId} modification halves must be adjacent replacement ranges`,
    )
  }
}

function contiguousSuggestionRanges(
  occurrences: readonly SuggestionMarkOccurrence[],
  markKind: 'deletion' | 'insertion',
): Array<{ from: number; parent: ProseMirrorNode; to: number }> {
  const ranges: Array<{ from: number; parent: ProseMirrorNode; to: number }> = []
  const matching = occurrences
    .filter((occurrence) => occurrence.markKind === markKind)
    .sort((left, right) => left.from - right.from || left.to - right.to)
  for (const occurrence of matching) {
    const previous = ranges.at(-1)
    if (previous && previous.parent === occurrence.parent && previous.to === occurrence.from) {
      previous.to = occurrence.to
    } else {
      ranges.push({
        from: occurrence.from,
        parent: occurrence.parent,
        to: occurrence.to,
      })
    }
  }
  return ranges
}

function sameSuggestionMetadata(
  left: SuggestionMetadata & { suggestionId: string },
  right: SuggestionMetadata & { suggestionId: string },
): boolean {
  return (
    left.authorId === right.authorId
    && left.createdAt === right.createdAt
    && left.patchId === right.patchId
    && left.suggestionId === right.suggestionId
  )
}

function resolveSuggestionKind(
  suggestionId: string,
  markKinds: ReadonlySet<SuggestionKind>,
): SuggestionKind {
  if (markKinds.size === 1) return [...markKinds][0]!
  if (
    markKinds.size === 2
    && markKinds.has('deletion')
    && markKinds.has('insertion')
  ) {
    return 'modification'
  }
  throw new Error(`Suggestion ${suggestionId} has an unsupported mark composition`)
}

export function assertYjsCompatibleSuggestionStructure(
  document: ProseMirrorNode,
): void {
  let unsupportedNodeType: string | null = null
  const inspect = (node: ProseMirrorNode): boolean => {
    if (
      !node.isText
      && node.marks.some((mark) => (
        SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind)
      ))
    ) {
      unsupportedNodeType = node.type.name
      return false
    }
    return true
  }
  inspect(document)
  document.descendants((node) => {
    if (unsupportedNodeType) return false
    return inspect(node)
  })
  if (unsupportedNodeType) {
    throw new UnsupportedSuggestionStructureError(
      `Node-level suggestions on ${unsupportedNodeType} cannot be represented by the Yjs editor codec`,
    )
  }
}

export function projectFinalDocument(document: ProseMirrorNode): ProseMirrorNode {
  return runProjection(document, applySuggestions)
}

export function projectOriginalDocument(document: ProseMirrorNode): ProseMirrorNode {
  return runProjection(document, revertSuggestions)
}

function runProjection(
  document: ProseMirrorNode,
  command: (state: EditorState, dispatch?: (transaction: Transaction) => void) => boolean,
): ProseMirrorNode {
  let state = EditorState.create({ schema: document.type.schema, doc: document })
  const handled = command(state, (transaction) => {
    state = state.apply(transaction)
  })
  if (!handled) {
    throw new Error('Suggestion projection could not be applied')
  }
  return state.doc
}

export function projectFinalJson(document: ProseMirrorNode): JSONContent {
  return projectFinalDocument(document).toJSON()
}

export function projectOriginalJson(document: ProseMirrorNode): JSONContent {
  return projectOriginalDocument(document).toJSON()
}
