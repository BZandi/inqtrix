import { getSchema, type JSONContent } from '@tiptap/core'
import type { Mark, Node as ProseMirrorNode } from '@tiptap/pm/model'
import {
  absolutePositionToRelativePosition,
  initProseMirrorDoc,
  relativePositionToAbsolutePosition,
} from '@tiptap/y-tiptap'
import {
  EDITOR_YJS_FRAGMENT,
  SUGGESTION_MARK_NAMES,
  createEditorSchemaExtensions,
  projectOriginalDocument,
  suggestionDescriptors,
  type CollaborationAccess,
  type CollaborationChangeKind,
  type SuggestionDescriptor,
  type SuggestionKind,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import type { SuggestionPatchState } from './contracts'
import { CloseCodes, CollaborationError } from './errors'

export type SuggestionOccurrence = {
  markKind: SuggestionKind
  node: unknown
  position: SuggestionPosition
}

export type SuggestionPosition = {
  from: number
  parentPath: StructuralPathEntry[]
  to: number
}

type StructuralPathEntry = {
  attrs: unknown
  index: number
  type: string
}

export type SuggestionPositionMapper = (
  position: SuggestionPosition,
) => SuggestionPosition | null

export type SuggestionDocumentPair = {
  afterDocument: Y.Doc
  beforeDocument: Y.Doc
}

type SuggestionXmlParent = Y.XmlElement | Y.XmlFragment
type SuggestionXmlType = Y.XmlElement | Y.XmlText

export type SuggestionRecord = {
  authorId: string
  createdAt: number
  kind: SuggestionKind
  occurrences: SuggestionOccurrence[]
  patchId: string
  suggestionId: string
}

export type SuggestionPolicyResult = {
  changeKind: CollaborationChangeKind
  patches: SuggestionPatchState[]
  suggestions: SuggestionDescriptor[]
  suggestionIds: string[]
}

export function validateSuggestionUpdate(
  before: JSONContent,
  after: JSONContent,
  access: CollaborationAccess,
  actorUserId: string,
  documents?: SuggestionDocumentPair,
): SuggestionPolicyResult {
  if (access === 'view') throw policyViolation('access_revoked', 403)

  const beforeRecords = collectSuggestionRecords(before)
  const afterRecords = collectSuggestionRecords(after)
  const positionMapper = documents
    ? createYjsSuggestionPositionMapper(before, after, documents)
    : null
  validatePatchGroups(beforeRecords)
  validatePatchGroups(afterRecords)
  validatePatchMetadataTransitions(beforeRecords, afterRecords)
  const semanticChangedIds = changedSuggestionContentIds(beforeRecords, afterRecords)
  const changedIds = changedSuggestionIds(beforeRecords, afterRecords, positionMapper)
  const changeKind = access === 'suggest' || changedIds.length > 0 ? 'suggestion' : 'direct'

  for (const [id, record] of afterRecords) {
    const previous = beforeRecords.get(id)
    if (!previous && record.authorId !== actorUserId) throw policyViolation()
    if (previous && !sameSuggestionMetadata(previous, record)) throw policyViolation()
  }
  for (const [id, record] of beforeRecords) {
    const next = afterRecords.get(id)
    if (!next) throw policyViolation()
    if (record.authorId === actorUserId) continue
    if (!positionMapper || !sameSuggestionRecord(record, next, positionMapper)) {
      throw policyViolation()
    }
  }

  if (changeKind === 'suggestion') {
    if (!documents) throw policyViolation()
    assertSuggestionYjsTopology(documents, new Set(semanticChangedIds))
    if (changedIds.length === 0) throw policyViolation()
    if (!isReversibleSuggestionStructure(before, after)) throw policyViolation()
  }

  const patches = patchStatesForChanges(beforeRecords, afterRecords, changedIds)
  if (patches.some((patch) => patch.activeSuggestionIds.length === 0)) {
    throw policyViolation()
  }

  return {
    changeKind,
    patches,
    suggestions: changedIds.map((id) => descriptor(afterRecords.get(id) ?? beforeRecords.get(id))),
    suggestionIds: changedIds,
  }
}

export function deriveSuggestionPatchStates(
  before: JSONContent,
  after: JSONContent,
  affectedSuggestionIds: readonly string[],
): SuggestionPatchState[] {
  const beforeRecords = collectSuggestionRecords(before)
  const afterRecords = collectSuggestionRecords(after)
  validatePatchGroups(beforeRecords)
  validatePatchGroups(afterRecords)
  validatePatchMetadataTransitions(beforeRecords, afterRecords)
  return patchStatesForChanges(beforeRecords, afterRecords, affectedSuggestionIds)
}

const schema = getSchema(createEditorSchemaExtensions({ enableUndoRedo: false }))

export function collectSuggestionRecords(document: JSONContent): Map<string, SuggestionRecord> {
  try {
    return collectProseMirrorSuggestionRecords(schema.nodeFromJSON(document))
  } catch (error) {
    if (error instanceof CollaborationError) throw error
    throw policyViolation('invalid_schema', 409)
  }
}

function collectProseMirrorSuggestionRecords(
  document: ProseMirrorNode,
): Map<string, SuggestionRecord> {
  const records = new Map<string, SuggestionRecord>()
  for (const descriptor of suggestionDescriptors(document)) {
    if (
      !isUuid(descriptor.suggestionId)
      || !isUuid(descriptor.patchId)
      || !isUuid(descriptor.authorId)
      || !Number.isSafeInteger(descriptor.createdAt)
      || descriptor.createdAt < 0
    ) {
      throw policyViolation('invalid_schema', 409)
    }
    records.set(descriptor.suggestionId, {
      ...descriptor,
      occurrences: [],
    })
  }

  document.descendants((node, position) => {
    const suggestionMarks = node.marks.filter((mark) => (
      SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind)
    ))
    for (const mark of suggestionMarks) {
      const attrs = mark.attrs
      const id = typeof attrs.suggestionId === 'string' ? attrs.suggestionId : null
      const kind = mark.type.name as SuggestionKind
      const record = id ? records.get(id) : null
      if (!record) throw policyViolation('invalid_schema', 409)
      const positionIdentity = suggestionPositionAt(
        document,
        position,
        position + node.nodeSize,
      )
      if (!positionIdentity) throw policyViolation('invalid_schema', 409)
      const occurrence: SuggestionOccurrence = {
        markKind: kind,
        node: suggestionNodeSignature(node, mark),
        position: positionIdentity,
      }
      record.occurrences.push(occurrence)
    }
  })
  if ([...records.values()].some((record) => record.occurrences.length === 0)) {
    throw policyViolation('invalid_schema', 409)
  }
  return records
}

export function isReversibleSuggestionStructure(
  before: JSONContent,
  after: JSONContent,
): boolean {
  try {
    const beforeDocument = schema.nodeFromJSON(before)
    const afterDocument = schema.nodeFromJSON(after)
    return (
      projectOriginalDocument(beforeDocument).eq(projectOriginalDocument(afterDocument))
      && sameNonTextAtoms(beforeDocument, afterDocument)
    )
  } catch {
    return false
  }
}

export function hasSameNonTextAtoms(before: JSONContent, after: JSONContent): boolean {
  try {
    return sameNonTextAtoms(
      schema.nodeFromJSON(before),
      schema.nodeFromJSON(after),
    )
  } catch {
    return false
  }
}

function sameNonTextAtoms(before: ProseMirrorNode, after: ProseMirrorNode): boolean {
  return stableJson(nonTextAtoms(before)) === stableJson(nonTextAtoms(after))
}

function nonTextAtoms(document: ProseMirrorNode): unknown[] {
  const atoms: unknown[] = []
  document.descendants((node) => {
    if (!node.isAtom || node.isText) return
    atoms.push({
      attrs: node.attrs,
      marks: node.marks
        .filter((mark) => !SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind))
        .map((mark) => ({ attrs: mark.attrs, type: mark.type.name })),
      type: node.type.name,
    })
  })
  return atoms
}

function patchStatesForChanges(
  before: ReadonlyMap<string, SuggestionRecord>,
  after: ReadonlyMap<string, SuggestionRecord>,
  affectedSuggestionIds: readonly string[],
): SuggestionPatchState[] {
  const beforeGroups = groupByPatch(before)
  const afterGroups = groupByPatch(after)
  const patchIds = new Set<string>()
  for (const suggestionId of affectedSuggestionIds) {
    const previous = before.get(suggestionId)
    const next = after.get(suggestionId)
    if (!previous && !next) throw policyViolation('invalid_schema', 409)
    if (previous) patchIds.add(previous.patchId)
    if (next) patchIds.add(next.patchId)
  }
  return [...patchIds].sort().map((patchId) => {
    const active = afterGroups.get(patchId) ?? []
    const reference = active[0] ?? beforeGroups.get(patchId)?.[0]
    if (!reference) throw policyViolation('invalid_schema', 409)
    return {
      activeSuggestionIds: active.map((record) => record.suggestionId).sort(),
      authorId: reference.authorId,
      createdAt: reference.createdAt,
      kinds: [...new Set(active.map((record) => record.kind))].sort(suggestionKindOrder),
      patchId,
    }
  })
}

function validatePatchGroups(records: ReadonlyMap<string, SuggestionRecord>): void {
  for (const group of groupByPatch(records).values()) {
    const reference = group[0]
    if (!reference) continue
    for (const record of group) {
      if (
        record.authorId !== reference.authorId
        || record.createdAt !== reference.createdAt
      ) {
        throw policyViolation('invalid_schema', 409)
      }
    }
  }
}

function validatePatchMetadataTransitions(
  before: ReadonlyMap<string, SuggestionRecord>,
  after: ReadonlyMap<string, SuggestionRecord>,
): void {
  const beforeGroups = groupByPatch(before)
  const afterGroups = groupByPatch(after)
  for (const [patchId, beforeGroup] of beforeGroups) {
    const afterGroup = afterGroups.get(patchId)
    if (!afterGroup || !beforeGroup[0] || !afterGroup[0]) continue
    if (
      beforeGroup[0].authorId !== afterGroup[0].authorId
      || beforeGroup[0].createdAt !== afterGroup[0].createdAt
    ) {
      throw policyViolation()
    }
  }
}

function groupByPatch(
  records: ReadonlyMap<string, SuggestionRecord>,
): Map<string, SuggestionRecord[]> {
  const groups = new Map<string, SuggestionRecord[]>()
  for (const record of records.values()) {
    const existing = groups.get(record.patchId)
    if (existing) existing.push(record)
    else groups.set(record.patchId, [record])
  }
  for (const group of groups.values()) {
    group.sort((left, right) => left.suggestionId.localeCompare(right.suggestionId))
  }
  return groups
}

function descriptor(record: SuggestionRecord | undefined): SuggestionDescriptor {
  if (!record) throw policyViolation('invalid_schema', 409)
  return {
    authorId: record.authorId,
    createdAt: record.createdAt,
    kind: record.kind,
    patchId: record.patchId,
    suggestionId: record.suggestionId,
  }
}

function isUuid(value: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value)
}

function changedSuggestionIds(
  before: ReadonlyMap<string, SuggestionRecord>,
  after: ReadonlyMap<string, SuggestionRecord>,
  positionMapper: SuggestionPositionMapper | null = null,
): string[] {
  const ids = new Set([...before.keys(), ...after.keys()])
  return [...ids]
    .filter((id) => {
      const previous = before.get(id)
      const next = after.get(id)
      return !previous || !next || !sameSuggestionRecord(
        previous,
        next,
        positionMapper ?? undefined,
      )
    })
    .sort()
}

function changedSuggestionContentIds(
  before: ReadonlyMap<string, SuggestionRecord>,
  after: ReadonlyMap<string, SuggestionRecord>,
): string[] {
  const ids = new Set([...before.keys(), ...after.keys()])
  return [...ids]
    .filter((id) => {
      const previous = before.get(id)
      const next = after.get(id)
      return !previous || !next || !sameSuggestionContent(previous, next)
    })
    .sort()
}

function sameSuggestionContent(left: SuggestionRecord, right: SuggestionRecord): boolean {
  if (
    !sameSuggestionMetadata(left, right)
    || left.occurrences.length !== right.occurrences.length
  ) return false
  const signatures = (record: SuggestionRecord): string[] => record.occurrences
    .map((occurrence) => stableJson({
      markKind: occurrence.markKind,
      node: occurrence.node,
    }))
    .sort()
  return stableJson(signatures(left)) === stableJson(signatures(right))
}

export function sameSuggestionRecord(
  left: SuggestionRecord,
  right: SuggestionRecord,
  positionMapper: SuggestionPositionMapper = (position) => position,
): boolean {
  if (
    !sameSuggestionMetadata(left, right)
    || left.occurrences.length !== right.occurrences.length
  ) return false

  const unmatched = new Set(right.occurrences.map((_occurrence, index) => index))
  for (const occurrence of left.occurrences) {
    const mappedPosition = positionMapper(occurrence.position)
    if (!mappedPosition) return false
    const matches = [...unmatched].filter((index) => {
      const next = right.occurrences[index]
      return Boolean(
        next
        && occurrence.markKind === next.markKind
        && stableJson(occurrence.node) === stableJson(next.node)
        && sameSuggestionPosition(mappedPosition, next.position)
      )
    })
    if (matches.length !== 1) return false
    unmatched.delete(matches[0]!)
  }
  return unmatched.size === 0
}

function sameSuggestionPosition(left: SuggestionPosition, right: SuggestionPosition): boolean {
  return (
    left.from === right.from
    && left.to === right.to
    && stableJson(left.parentPath) === stableJson(right.parentPath)
  )
}

function sameSuggestionMetadata(left: SuggestionRecord, right: SuggestionRecord): boolean {
  return (
    left.authorId === right.authorId
    && left.createdAt === right.createdAt
    && left.kind === right.kind
    && left.patchId === right.patchId
    && left.suggestionId === right.suggestionId
  )
}

function suggestionKindOrder(left: SuggestionKind, right: SuggestionKind): number {
  const order: Record<SuggestionKind, number> = {
    insertion: 0,
    deletion: 1,
    modification: 2,
  }
  return order[left] - order[right]
}

function suggestionNodeSignature(
  node: ProseMirrorNode,
  targetMark: Mark,
): unknown {
  return {
    attrs: node.attrs,
    marks: node.marks
      .filter((mark) => (
        mark !== targetMark
        && !SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind)
      ))
      .map((mark) => ({ attrs: mark.attrs, type: mark.type.name })),
    text: node.text,
    type: node.type.name,
  }
}

export function suggestionPositionAt(
  document: ProseMirrorNode,
  from: number,
  to: number,
): SuggestionPosition | null {
  if (
    !Number.isSafeInteger(from)
    || !Number.isSafeInteger(to)
    || from < 0
    || to < from
    || to > document.content.size
  ) return null
  try {
    const resolved = document.resolve(from)
    const parentPath: StructuralPathEntry[] = []
    for (let depth = 0; depth <= resolved.depth; depth += 1) {
      const node = resolved.node(depth)
      parentPath.push({
        attrs: node.attrs,
        index: depth === 0 ? 0 : resolved.index(depth - 1),
        type: node.type.name,
      })
    }
    return { from, parentPath, to }
  } catch {
    return null
  }
}

function createYjsSuggestionPositionMapper(
  before: JSONContent,
  after: JSONContent,
  documents: SuggestionDocumentPair,
): SuggestionPositionMapper {
  try {
    const beforeFragment = documents.beforeDocument.getXmlFragment(EDITOR_YJS_FRAGMENT)
    const afterFragment = documents.afterDocument.getXmlFragment(EDITOR_YJS_FRAGMENT)
    const beforeInitialized = initProseMirrorDoc(beforeFragment, schema)
    const afterInitialized = initProseMirrorDoc(afterFragment, schema)
    if (
      !beforeInitialized.doc.eq(schema.nodeFromJSON(before))
      || !afterInitialized.doc.eq(schema.nodeFromJSON(after))
    ) {
      throw policyViolation()
    }
    return (position) => {
      try {
        const relativeFrom = absolutePositionToRelativePosition(
          position.from,
          beforeFragment,
          beforeInitialized.mapping,
        ) as Y.RelativePosition
        const relativeTo = absolutePositionToRelativePosition(
          position.to,
          beforeFragment,
          beforeInitialized.mapping,
        ) as Y.RelativePosition
        const from = relativePositionToAbsolutePosition(
          documents.afterDocument,
          afterFragment,
          relativeFrom,
          afterInitialized.mapping,
        )
        const to = relativePositionToAbsolutePosition(
          documents.afterDocument,
          afterFragment,
          relativeTo,
          afterInitialized.mapping,
        )
        if (from === null || to === null) return null
        return suggestionPositionAt(afterInitialized.doc, from, to)
      } catch {
        return null
      }
    }
  } catch (error) {
    if (error instanceof CollaborationError) throw error
    throw policyViolation()
  }
}

function assertSuggestionYjsTopology(
  documents: SuggestionDocumentPair,
  changedSuggestionIds: ReadonlySet<string>,
): void {
  const beforeRoot = documents.beforeDocument.getXmlFragment(EDITOR_YJS_FRAGMENT)
  const afterRoot = documents.afterDocument.getXmlFragment(EDITOR_YJS_FRAGMENT)
  const beforeTypes = collectSuggestionXmlTypes(beforeRoot)
  const afterTypes = collectSuggestionXmlTypes(afterRoot)
  const mappedTypes = new Map<SuggestionXmlType, SuggestionXmlType>()
  const mappedAfterTypes = new Set<SuggestionXmlType>()

  for (const beforeType of beforeTypes) {
    const afterType = mapSuggestionXmlType(beforeType, documents.afterDocument)
    if (!afterType) {
      if (
        beforeType instanceof Y.XmlText
        && isChangedInsertionText(beforeType, changedSuggestionIds)
      ) continue
      throw policyViolation()
    }
    if (
      (beforeType instanceof Y.XmlElement) !== (afterType instanceof Y.XmlElement)
      || mappedAfterTypes.has(afterType)
    ) {
      throw policyViolation()
    }
    mappedTypes.set(beforeType, afterType)
    mappedAfterTypes.add(afterType)
  }

  for (const [beforeType, afterType] of mappedTypes) {
    const expectedParent = beforeType.parent === beforeRoot
      ? afterRoot
      : isSuggestionXmlType(beforeType.parent)
        ? mappedTypes.get(beforeType.parent)
        : null
    if (!expectedParent || afterType.parent !== expectedParent) throw policyViolation()
  }

  assertMappedChildOrder(beforeRoot, afterRoot, mappedTypes, mappedAfterTypes)
  for (const [beforeType, afterType] of mappedTypes) {
    if (beforeType instanceof Y.XmlElement && afterType instanceof Y.XmlElement) {
      assertMappedChildOrder(beforeType, afterType, mappedTypes, mappedAfterTypes)
    }
  }

  for (const afterType of afterTypes) {
    if (mappedAfterTypes.has(afterType)) continue
    const parent = afterType.parent
    if (
      !(afterType instanceof Y.XmlText)
      || !isChangedInsertionText(afterType, changedSuggestionIds)
      || (
        parent !== afterRoot
        && (!isSuggestionXmlType(parent) || !mappedAfterTypes.has(parent))
      )
    ) {
      throw policyViolation()
    }
  }
}

function collectSuggestionXmlTypes(parent: SuggestionXmlParent): SuggestionXmlType[] {
  const types: SuggestionXmlType[] = []
  for (const child of parent.toArray()) {
    if (!isSuggestionXmlType(child)) throw policyViolation()
    types.push(child)
    if (child instanceof Y.XmlElement) {
      types.push(...collectSuggestionXmlTypes(child))
    }
  }
  return types
}

function mapSuggestionXmlType(
  source: SuggestionXmlType,
  targetDocument: Y.Doc,
): SuggestionXmlType | null {
  const absolute = Y.createAbsolutePositionFromRelativePosition(
    Y.createRelativePositionFromTypeIndex(source, 0),
    targetDocument,
  )
  return isSuggestionXmlType(absolute?.type) ? absolute.type : null
}

function assertMappedChildOrder(
  beforeParent: SuggestionXmlParent,
  afterParent: SuggestionXmlParent,
  mappedTypes: ReadonlyMap<SuggestionXmlType, SuggestionXmlType>,
  mappedAfterTypes: ReadonlySet<SuggestionXmlType>,
): void {
  const expected = beforeParent.toArray()
    .filter(isSuggestionXmlType)
    .flatMap((child) => {
      const mapped = mappedTypes.get(child)
      return mapped ? [mapped] : []
    })
  const actual = afterParent.toArray()
    .filter(isSuggestionXmlType)
    .filter((child) => mappedAfterTypes.has(child))
  if (
    expected.length !== actual.length
    || expected.some((child, index) => child !== actual[index])
  ) {
    throw policyViolation()
  }
}

function isChangedInsertionText(
  text: Y.XmlText,
  changedSuggestionIds: ReadonlySet<string>,
): boolean {
  const delta = text.toDelta() as Array<{
    attributes?: Record<string, unknown>
    insert: unknown
  }>
  return delta.length > 0 && delta.every((part) => {
    if (typeof part.insert !== 'string' || part.insert.length === 0) return false
    const insertion = part.attributes?.insertion
    if (!insertion || typeof insertion !== 'object') return false
    const suggestionId = Reflect.get(insertion, 'suggestionId')
    return typeof suggestionId === 'string' && changedSuggestionIds.has(suggestionId)
  })
}

function isSuggestionXmlType(value: unknown): value is SuggestionXmlType {
  return value instanceof Y.XmlElement || value instanceof Y.XmlText
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`)
      .join(',')}}`
  }
  return JSON.stringify(value)
}

function policyViolation(
  reason: 'access_revoked' | 'invalid_schema' | 'suggestion_policy_violation' = 'suggestion_policy_violation',
  httpStatus = 403,
): CollaborationError {
  return new CollaborationError(reason, {
    closeCode: reason === 'invalid_schema'
      ? CloseCodes.incompatible
      : CloseCodes.accessRevoked,
    httpStatus,
  })
}
