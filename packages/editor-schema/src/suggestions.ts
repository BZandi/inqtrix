import {
  applySuggestions,
  revertSuggestions,
  suggestChangesKey,
  transformToSuggestionTransaction,
} from '@handlewithcare/prosemirror-suggest-changes'
import type { JSONContent } from '@tiptap/core'
import type { Mark, Node as ProseMirrorNode } from '@tiptap/pm/model'
import { EditorState, type Transaction } from '@tiptap/pm/state'
import { findWrapping, type StepMap } from '@tiptap/pm/transform'
import { ySyncPluginKey } from '@tiptap/y-tiptap'
import { createSecureUuid } from './ids.js'
import {
  SUGGESTION_MARK_NAMES,
  type SerializedSuggestionKind,
  type SuggestionKind,
  type SuggestionMarkKind,
} from './suggestionMarks.js'
import {
  INQTRIX_STRUCTURE_COMMAND_META,
  INQTRIX_STRUCTURE_SUGGESTION_ATTR,
  isStructureSuggestionData,
  type StructureSuggestionAction,
  type StructureSuggestionCommand,
  type StructureSuggestionData,
} from './structureSuggestions.js'

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
  declaredKind: SerializedSuggestionKind
  markKind: SuggestionMarkKind
  mark: Mark
  suggestionId: string
}

type SuggestionDescriptorAccumulator = SuggestionMetadata & {
  declarations: Array<{
    declaredKind: SerializedSuggestionKind
    markKind: SuggestionMarkKind
  }>
  markKinds: Set<SuggestionMarkKind>
  marks: Mark[]
  occurrences: SuggestionMarkOccurrence[]
  suggestionId: string
}

type SuggestionMarkOccurrence = {
  from: number
  markKind: SuggestionMarkKind
  parent: ProseMirrorNode
  to: number
}

type MarkLocation = {
  from: number
  mark: Mark
  nodePosition: number | null
  to: number
}

type TrackedSlashSuggestion = SuggestionMetadata & {
  suggestionId: string
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
  generateSuggestionId: () => string = createSecureUuid,
): Transaction {
  if (shouldBypassSuggestionTransform(transaction)) return transaction
  const structureCommand = transaction.getMeta(
    INQTRIX_STRUCTURE_COMMAND_META,
  ) as StructureSuggestionCommand | undefined
  if (structureCommand) {
    return transformStructureCommand(
      transaction,
      state,
      metadata,
      structureCommand,
      generateSuggestionId,
    )
  }
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
  transformed = normalizeReusedSuggestionMarks(transformed, state.doc)
  assertYjsCompatibleSuggestionStructure(transformed.doc)
  transformed = enrichGeneratedSuggestionMarks(
    transformed,
    generatedIds,
    metadata,
    isPureFormattingTransaction(transaction) ? 'format' : null,
  )
  assertSuggestionTransformIsRepresented(
    state.doc,
    transformed.doc,
    generatedIds,
    metadata.authorId,
  )
  return transformed
}

/**
 * `prosemirror-suggest-changes` preserves only the upstream `id` attribute
 * when it extends an adjacent mark. Inqtrix adds authoritative actor, patch,
 * time and semantic attributes, so copy those attributes from the existing
 * mark before validating the transformed document.
 */
function normalizeReusedSuggestionMarks(
  transaction: Transaction,
  before: ProseMirrorNode,
): Transaction {
  const existing = new Map<string, Mark>()
  before.descendants((node) => {
    for (const mark of node.marks) {
      const id = mark.attrs.id
      if (
        SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionMarkKind)
        && typeof id === 'string'
        && id.length > 0
      ) {
        existing.set(`${mark.type.name}:${id}`, mark)
      }
    }
  })
  const locations: MarkLocation[] = []
  transaction.doc.descendants((node, position) => {
    for (const mark of node.marks) {
      const id = mark.attrs.id
      if (
        typeof id !== 'string'
        || !existing.has(`${mark.type.name}:${id}`)
      ) continue
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
    const authoritative = existing.get(`${location.mark.type.name}:${id}`)
    if (!authoritative || location.mark.eq(authoritative)) continue
    if (location.nodePosition === null) {
      transaction.removeMark(location.from, location.to, location.mark)
      transaction.addMark(location.from, location.to, authoritative)
      continue
    }
    transaction.removeNodeMark(location.nodePosition, location.mark)
    transaction.addNodeMark(location.nodePosition, authoritative)
  }
  return transaction
}

function isPureFormattingTransaction(transaction: Transaction): boolean {
  return (
    transaction.steps.length > 0
    && transaction.steps.every((step) => {
      const stepType = step.toJSON().stepType
      return stepType === 'addMark' || stepType === 'removeMark'
    })
  )
}

function assertSuggestionTransformIsRepresented(
  before: ProseMirrorNode,
  after: ProseMirrorNode,
  generatedIds: ReadonlySet<string>,
  authorId: string,
): void {
  const represented = new Set<string>()
  after.descendants((node) => {
    for (const mark of node.marks) {
      const id = mark.attrs.id
      if (typeof id === 'string' && generatedIds.has(id)) represented.add(id)
    }
  })
  if ([...generatedIds].every((id) => represented.has(id))) return

  /*
   * The upstream tracker deliberately reuses an adjacent insertion/deletion
   * mark instead of the freshly allocated id. This is how normal multi-key
   * typing and typing into a newly proposed empty paragraph stay one review
   * item. Accept that optimisation only when:
   *
   *  - an authoritative suggestion by the same actor survives unchanged, and
   *  - rejecting all suggestions produces the exact same original document.
   *
   * The projection equality is the important safety boundary: a default
   * ProseMirror structural step without a durable suggestion representation
   * would alter the original projection and is still rejected.
   */
  try {
    const previous = new Map(
      suggestionDescriptors(before).map((descriptor) => [
        descriptor.suggestionId,
        descriptor,
      ]),
    )
    const reused = suggestionDescriptors(after).some((descriptor) => {
      const existing = previous.get(descriptor.suggestionId)
      return (
        descriptor.authorId === authorId
        && existing?.authorId === descriptor.authorId
        && existing.createdAt === descriptor.createdAt
        && existing.kind === descriptor.kind
        && existing.patchId === descriptor.patchId
      )
    })
    if (
      reused
      && projectOriginalDocument(before).eq(projectOriginalDocument(after))
    ) {
      return
    }
  } catch {
    // Fall through to the stable domain error below.
  }

  throw new UnsupportedSuggestionStructureError(
    'The suggestion transform produced a structural change without a persistent mark',
  )
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
  semanticOverride: Extract<SuggestionKind, 'format'> | null,
): Transaction {
  const locations: MarkLocation[] = []
  const markKinds = new Map<string, Set<SuggestionMarkKind>>()
  transaction.doc.descendants((node, position) => {
    for (const mark of node.marks) {
      const id = mark.attrs.id
      if (typeof id !== 'string' || !generatedIds.has(id)) continue
      const kind = mark.type.name as SuggestionMarkKind
      const kinds = markKinds.get(id) ?? new Set<SuggestionMarkKind>()
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
    const kind = semanticOverride ?? resolveSuggestionKind(
      id,
      kinds,
      locations
        .filter((candidate) => candidate.mark.attrs.id === id)
        .map((candidate) => candidate.mark),
    )
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
      if (!SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionMarkKind)) continue
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
        existing.marks.push(descriptor.mark)
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
        marks: [descriptor.mark],
        occurrences: [occurrence],
        patchId: descriptor.patchId,
        suggestionId: descriptor.suggestionId,
      })
    }
  })
  const markDescriptors = [...descriptors.values()]
    .map((descriptor) => {
      const kind = resolveSuggestionKind(
        descriptor.suggestionId,
        descriptor.markKinds,
        descriptor.marks,
      )
      if (!hasValidKindDeclarations(kind, descriptor.declarations)) {
        throw new Error(`Suggestion ${descriptor.suggestionId} has an invalid mark composition`)
      }
      if (
        (kind === 'replacement' || kind === 'format')
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
  const result = new Map(markDescriptors.map((descriptor) => [
    descriptor.suggestionId,
    descriptor,
  ]))
  document.descendants((node) => {
    const structure = node.attrs[INQTRIX_STRUCTURE_SUGGESTION_ATTR]
    if (structure === null || structure === undefined) return true
    if (!isStructureSuggestionData(structure)) {
      throw new Error('Structural suggestion is missing authoritative metadata')
    }
    if (result.has(structure.suggestionId)) {
      throw new Error(`Suggestion ${structure.suggestionId} has duplicate representations`)
    }
    result.set(structure.suggestionId, {
      authorId: structure.authorId,
      createdAt: structure.createdAt,
      kind: 'structure',
      patchId: structure.patchId,
      suggestionId: structure.suggestionId,
    })
    return true
  })
  return [...result.values()].sort((left, right) => (
      left.createdAt - right.createdAt || left.suggestionId.localeCompare(right.suggestionId)
    ))
}

function suggestionMarkDescriptor(mark: Mark): SuggestionMarkDescriptor {
  const markKind = mark.type.name as SuggestionMarkKind
  const { authorId, createdAt, id, kind, patchId, suggestionId } = mark.attrs
  if (
    !SUGGESTION_MARK_NAMES.has(markKind)
    || typeof id !== 'string'
    || id.length === 0
    || typeof suggestionId !== 'string'
    || suggestionId.length === 0
    || id !== suggestionId
    || !isSerializedSuggestionKind(kind)
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
    declaredKind: kind as SerializedSuggestionKind,
    mark,
    markKind,
    patchId,
    suggestionId,
  }
}

function isSerializedSuggestionKind(value: unknown): value is SerializedSuggestionKind {
  return (
    value === 'deletion'
    || value === 'format'
    || value === 'insertion'
    || value === 'modification'
    || value === 'replacement'
    || value === 'structure'
  )
}

function hasValidKindDeclarations(
  resolvedKind: SuggestionKind,
  declarations: ReadonlyArray<{
    declaredKind: SerializedSuggestionKind
    markKind: SuggestionMarkKind
  }>,
): boolean {
  return declarations.every(({ declaredKind }) => (
    declaredKind === resolvedKind
    || (
      declaredKind === 'modification'
      && ['format', 'replacement', 'structure'].includes(resolvedKind)
    )
  ))
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
  markKinds: ReadonlySet<SuggestionMarkKind>,
  marks: readonly Mark[],
): SuggestionKind {
  const declaredKinds = new Set(
    marks
      .map((mark) => mark.attrs.kind)
      .filter((kind): kind is SuggestionKind => (
        kind === 'deletion'
        || kind === 'format'
        || kind === 'insertion'
        || kind === 'replacement'
        || kind === 'structure'
      )),
  )
  if (declaredKinds.size === 1) {
    const declared = [...declaredKinds][0]!
    if (
      (declared === 'format' || declared === 'replacement')
      && markKinds.size === 2
      && markKinds.has('deletion')
      && markKinds.has('insertion')
    ) return declared
    if (
      declared === 'structure'
      && markKinds.size === 1
      && markKinds.has('modification')
    ) return declared
  }
  if (markKinds.size === 1 && markKinds.has('insertion')) return 'insertion'
  if (markKinds.size === 1 && markKinds.has('deletion')) return 'deletion'
  if (markKinds.size === 1 && markKinds.has('modification')) {
    return marks.some((mark) => mark.attrs.type === 'nodeType')
      ? 'structure'
      : 'format'
  }
  if (
    markKinds.size === 2
    && markKinds.has('deletion')
    && markKinds.has('insertion')
  ) {
    return 'replacement'
  }
  throw new Error(`Suggestion ${suggestionId} has an unsupported mark composition`)
}

function transformStructureCommand(
  proposed: Transaction,
  state: EditorState,
  metadata: SuggestionMetadata,
  command: StructureSuggestionCommand,
  generateSuggestionId: () => string,
): Transaction {
  if (command.action === 'table' || command.action === 'divider') {
    throw new UnsupportedSuggestionStructureError(
      command.action === 'table'
        ? 'Table topology cannot be proposed safely. Switch to Edit mode to insert a table.'
        : 'A divider cannot be reviewed as a reversible structural suggestion.',
    )
  }
  const targetBefore = textblockPositionAt(
    state.doc,
    command.commandRange?.from ?? state.selection.from,
  )
  if (targetBefore === null) {
    throw new UnsupportedSuggestionStructureError(
      'The selected block cannot carry a structural suggestion.',
    )
  }
  const sourceNode = state.doc.nodeAt(targetBefore)
  if (!sourceNode?.isTextblock) {
    throw new UnsupportedSuggestionStructureError(
      'The selected block cannot carry a structural suggestion.',
    )
  }
  const transformed = state.tr
  let discardedCommand: StructureSuggestionData['discardedCommand']
  const trackedSlash = command.commandRange
    ? trackedSlashSuggestion(
        state.doc,
        command.commandRange,
        metadata.authorId,
      )
    : null
  if (command.commandRange) {
    const { from, to } = command.commandRange
    const contentFrom = targetBefore + 1
    const contentTo = targetBefore + sourceNode.nodeSize - 1
    if (
      !Number.isSafeInteger(from)
      || !Number.isSafeInteger(to)
      || from < contentFrom
      || to < from
      || to > contentTo
    ) {
      throw new UnsupportedSuggestionStructureError(
        'The slash command range is outside the selected block.',
      )
    }
    const text = state.doc.textBetween(from, to, '', '')
    if (!text.startsWith('/') || text.length > 96) {
      throw new UnsupportedSuggestionStructureError(
        'The slash command is too large to preserve safely.',
      )
    }
    if (!trackedSlash) {
      discardedCommand = {
        offset: from - contentFrom,
        text,
      }
    }
    transformed.delete(from, to)
  }
  const mappedTarget = transformed.mapping.map(targetBefore, 1)
  const expected = state.tr
  if (command.commandRange) {
    expected.delete(command.commandRange.from, command.commandRange.to)
  }
  const expectedTarget = expected.mapping.map(targetBefore, 1)
  applyStructureAction(expected, expectedTarget, command.action)
  if (!expected.doc.eq(proposed.doc)) {
    throw new UnsupportedSuggestionStructureError(
      'This block transformation cannot yet be represented as one reversible suggestion.',
    )
  }
  const suggestionId = generateSuggestionId()
  if (suggestionId === trackedSlash?.suggestionId) {
    throw new UnsupportedSuggestionStructureError(
      'The structural suggestion identifier must differ from the slash insertion.',
    )
  }
  const structureMetadata = trackedSlash ?? metadata
  const structure: StructureSuggestionData = {
    action: command.action,
    authorId: structureMetadata.authorId,
    createdAt: structureMetadata.createdAt,
    ...(discardedCommand ? { discardedCommand } : {}),
    kind: 'structure',
    patchId: structureMetadata.patchId,
    suggestionId,
  }
  transformed.setNodeAttribute(
    mappedTarget,
    INQTRIX_STRUCTURE_SUGGESTION_ATTR,
    structure,
  )
  assertYjsCompatibleSuggestionStructure(transformed.doc)
  return transformed
}

/**
 * A real slash-menu interaction is two collaborative transactions: typing the
 * slash first creates an insertion suggestion, then selecting a command removes
 * that token and proposes a block transformation. Reuse the first suggestion's
 * patch metadata only when the command range consumes that complete, actor-owned
 * insertion. This keeps the durable patch non-empty and prevents a transient
 * command token from becoming a second review item.
 */
function trackedSlashSuggestion(
  document: ProseMirrorNode,
  range: { from: number; to: number },
  authorId: string,
): TrackedSlashSuggestion | null {
  const { from, to } = range
  const commandText = document.textBetween(from, to, '', '')
  if (
    !Number.isSafeInteger(from)
    || !Number.isSafeInteger(to)
    || from < 0
    || to <= from
    || to > document.content.size
    || !commandText.startsWith('/')
    || commandText.length > 96
  ) return null

  let candidate: TrackedSlashSuggestion | null = null
  let covered = 0
  let invalid = false
  document.nodesBetween(from, to, (node, position) => {
    if (!node.isText) return true
    const overlapFrom = Math.max(from, position)
    const overlapTo = Math.min(to, position + node.nodeSize)
    if (overlapTo <= overlapFrom) return false
    const suggestionMarks = node.marks.filter((mark) => (
      SUGGESTION_MARK_NAMES.has(mark.type.name)
    ))
    const insertion = suggestionMarks.length === 1
      && suggestionMarks[0]?.type.name === 'insertion'
      ? suggestionMarks[0]
      : null
    const attrs = insertion?.attrs
    if (
      !attrs
      || attrs.authorId !== authorId
      || attrs.kind !== 'insertion'
      || typeof attrs.patchId !== 'string'
      || attrs.patchId.length === 0
      || typeof attrs.suggestionId !== 'string'
      || attrs.suggestionId.length === 0
      || attrs.id !== attrs.suggestionId
      || !Number.isSafeInteger(attrs.createdAt)
      || Number(attrs.createdAt) < 0
    ) {
      invalid = true
      return false
    }
    const current: TrackedSlashSuggestion = {
      authorId,
      createdAt: Number(attrs.createdAt),
      patchId: attrs.patchId,
      suggestionId: attrs.suggestionId,
    }
    if (
      candidate
      && (
        candidate.createdAt !== current.createdAt
        || candidate.patchId !== current.patchId
        || candidate.suggestionId !== current.suggestionId
      )
    ) {
      invalid = true
      return false
    }
    candidate = current
    covered += overlapTo - overlapFrom
    return false
  })
  if (invalid || !candidate || covered !== to - from) return null

  let occurrenceSize = 0
  document.descendants((node, position) => {
    if (!node.isText) return true
    const matches = node.marks.some((mark) => (
      mark.type.name === 'insertion'
      && mark.attrs.suggestionId === candidate?.suggestionId
    ))
    if (!matches) return true
    if (position < from || position + node.nodeSize > to) invalid = true
    occurrenceSize += node.nodeSize
    return true
  })
  return !invalid && occurrenceSize === to - from ? candidate : null
}

function textblockPositionAt(document: ProseMirrorNode, position: number): number | null {
  const safePosition = Math.max(0, Math.min(position, document.content.size))
  const resolved = document.resolve(safePosition)
  for (let depth = resolved.depth; depth >= 1; depth -= 1) {
    if (resolved.node(depth).isTextblock) return resolved.before(depth)
  }
  const node = document.nodeAt(safePosition)
  return node?.isTextblock ? safePosition : null
}

function applyStructureAction(
  transaction: Transaction,
  position: number,
  action: StructureSuggestionAction,
): void {
  const node = transaction.doc.nodeAt(position)
  if (!node?.isTextblock) {
    throw new UnsupportedSuggestionStructureError(
      'The structural suggestion target no longer exists.',
    )
  }
  const schema = transaction.doc.type.schema
  if (action === 'paragraph') {
    const paragraph = schema.nodes.paragraph
    if (!paragraph) throw new UnsupportedSuggestionStructureError('Paragraphs are unavailable.')
    transaction.setNodeMarkup(position, paragraph, {
      textAlign: node.attrs.textAlign ?? null,
    })
    return
  }
  if (action === 'codeBlock') {
    const codeBlock = schema.nodes.codeBlock
    const paragraph = schema.nodes.paragraph
    if (!codeBlock || !paragraph) {
      throw new UnsupportedSuggestionStructureError('Code blocks are unavailable.')
    }
    transaction.setNodeMarkup(
      position,
      node.type === codeBlock ? paragraph : codeBlock,
      node.type === codeBlock ? { textAlign: null } : null,
    )
    return
  }
  if (action.startsWith('heading')) {
    const heading = schema.nodes.heading
    const paragraph = schema.nodes.paragraph
    if (!heading || !paragraph) {
      throw new UnsupportedSuggestionStructureError('Headings are unavailable.')
    }
    const level = Number(action.at(-1))
    const targetType = node.type === heading && node.attrs.level === level
      ? paragraph
      : heading
    transaction.setNodeMarkup(position, targetType, targetType === heading
      ? { level, textAlign: node.attrs.textAlign ?? null }
      : { textAlign: node.attrs.textAlign ?? null })
    return
  }
  const wrapperName = action === 'blockquote'
    ? 'blockquote'
    : action === 'bulletList'
      ? 'bulletList'
      : action === 'orderedList'
        ? 'orderedList'
        : 'taskList'
  const wrapper = schema.nodes[wrapperName]
  if (!wrapper) {
    throw new UnsupportedSuggestionStructureError(
      'The requested block structure is unavailable.',
    )
  }
  const from = transaction.doc.resolve(position)
  const to = transaction.doc.resolve(position + node.nodeSize)
  const range = from.blockRange(to)
  const wrapping = range ? findWrapping(range, wrapper) : null
  if (!range || !wrapping) {
    throw new UnsupportedSuggestionStructureError(
      'Only an unwrapped text block can be proposed as this structure.',
    )
  }
  transaction.wrap(range, wrapping)
}

export function resolveStructureSuggestion(
  state: EditorState,
  suggestionId: string,
  decision: 'accept' | 'reject',
): EditorState {
  const locations = structureSuggestionLocations(state.doc)
    .filter((location) => location.data.suggestionId === suggestionId)
  if (locations.length !== 1) {
    throw new Error(`Structural suggestion ${suggestionId} is missing or duplicated`)
  }
  const transaction = state.tr
  const location = locations[0]!
  transaction.setNodeAttribute(
    location.position,
    INQTRIX_STRUCTURE_SUGGESTION_ATTR,
    null,
  )
  if (decision === 'accept') {
    applyStructureAction(transaction, location.position, location.data.action)
  }
  return state.apply(transaction)
}

function projectStructureSuggestions(
  state: EditorState,
  projection: 'final' | 'original',
): EditorState {
  const locations = structureSuggestionLocations(state.doc)
    .sort((left, right) => right.position - left.position)
  let current = state
  for (const location of locations) {
    const live = structureSuggestionLocations(current.doc)
      .find((candidate) => candidate.data.suggestionId === location.data.suggestionId)
    if (!live) throw new Error(`Structural suggestion ${location.data.suggestionId} disappeared`)
    const transaction = current.tr
    transaction.setNodeAttribute(
      live.position,
      INQTRIX_STRUCTURE_SUGGESTION_ATTR,
      null,
    )
    if (projection === 'final') {
      applyStructureAction(transaction, live.position, live.data.action)
    } else if (live.data.discardedCommand) {
      const insertionAt = (
        live.position
        + 1
        + live.data.discardedCommand.offset
      )
      transaction.insertText(live.data.discardedCommand.text, insertionAt)
    }
    current = current.apply(transaction)
  }
  return current
}

function structureSuggestionLocations(document: ProseMirrorNode): Array<{
  data: StructureSuggestionData
  position: number
}> {
  const locations: Array<{ data: StructureSuggestionData; position: number }> = []
  document.descendants((node, position) => {
    const data = node.attrs[INQTRIX_STRUCTURE_SUGGESTION_ATTR]
    if (data === null || data === undefined) return true
    if (!node.isTextblock || !isStructureSuggestionData(data)) {
      throw new Error('Structural suggestion metadata is invalid')
    }
    locations.push({ data, position })
    return true
  })
  return locations
}

export function assertYjsCompatibleSuggestionStructure(
  document: ProseMirrorNode,
): void {
  let unsupportedNodeType: string | null = null
  const inspect = (node: ProseMirrorNode): boolean => {
    if (
      !node.isText
      && node.marks.some((mark) => (
        SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionMarkKind)
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
  const structureIds = new Set<string>()
  document.descendants((node) => {
    const structure = node.attrs[INQTRIX_STRUCTURE_SUGGESTION_ATTR]
    if (structure === null || structure === undefined) return true
    if (
      !node.isTextblock
      || !isStructureSuggestionData(structure)
      || structureIds.has(structure.suggestionId)
    ) {
      throw new UnsupportedSuggestionStructureError(
        'Structural suggestion metadata is invalid or duplicated',
      )
    }
    structureIds.add(structure.suggestionId)
    return true
  })
}

export function projectFinalDocument(document: ProseMirrorNode): ProseMirrorNode {
  return runProjection(document, 'final')
}

export function projectOriginalDocument(document: ProseMirrorNode): ProseMirrorNode {
  return runProjection(document, 'original')
}

function runProjection(
  document: ProseMirrorNode,
  projection: 'final' | 'original',
): ProseMirrorNode {
  let state = EditorState.create({ schema: document.type.schema, doc: document })
  const command = projection === 'final' ? applySuggestions : revertSuggestions
  const handled = command(state, (transaction) => {
    state = state.apply(transaction)
  })
  if (!handled) {
    throw new Error('Suggestion projection could not be applied')
  }
  return projectStructureSuggestions(state, projection).doc
}

export function projectFinalJson(document: ProseMirrorNode): JSONContent {
  return projectFinalDocument(document).toJSON()
}

export function projectOriginalJson(document: ProseMirrorNode): JSONContent {
  return projectOriginalDocument(document).toJSON()
}
