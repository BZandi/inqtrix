import type {
  CollaborationActorKind,
  CollaborationChangeKind,
  SuggestionDescriptor,
  SuggestionKind,
} from '@inqtrix/editor-schema'
import type {
  EditorCollaborationConnectionStatus,
  EditorCollaborationDurabilityStatus,
} from '@/features/project/types'

export type EditorInspectorTab = 'assistant' | 'changes'
export type EditorChangesView = 'history' | 'open'
export type EditorWriteMode = 'edit' | 'suggest' | 'view'

export type InspectorParticipant = {
  color: string
  id: string
  name: string
}

export type EditorCollaborationStatusKind =
  | 'access_revoked'
  | 'error'
  | 'inactive'
  | 'read_only'
  | 'reconnecting'
  | 'saved'
  | 'saving'
  | 'syncing'
  | 'update_required'

export type EditorCollaborationStatusModel = {
  active: boolean
  kind: EditorCollaborationStatusKind
  notice: string | null
  participantOverflow: number
  participants: readonly InspectorParticipant[]
  projectionConfirmedAt: string | null
  visibleParticipants: readonly InspectorParticipant[]
}

export type PendingCollaborationPublicationFocus = {
  documentId: string
  patchId: string
}

export type PendingCollaborationPublicationFocusByDocument = Readonly<
  Record<string, PendingCollaborationPublicationFocus>
>

type CollaborationStatusInput = {
  access: 'edit' | 'suggest' | 'view' | null
  active: boolean
  canEdit: boolean
  connectionStatus: EditorCollaborationConnectionStatus
  durabilityStatus: EditorCollaborationDurabilityStatus
  notice?: string | null
  participants: readonly InspectorParticipant[]
  projectionUpdatedAt?: string | null
  synced: boolean
}

export function buildEditorCollaborationStatusModel({
  access,
  active,
  canEdit,
  connectionStatus,
  durabilityStatus,
  notice = null,
  participants,
  projectionUpdatedAt = null,
  synced,
}: CollaborationStatusInput): EditorCollaborationStatusModel {
  const preview = participantPreview(participants)
  let kind: EditorCollaborationStatusKind = 'saved'
  if (!active) kind = 'inactive'
  else if (connectionStatus === 'access_revoked') kind = 'access_revoked'
  else if (connectionStatus === 'incompatible') kind = 'update_required'
  else if (notice || connectionStatus === 'error' || durabilityStatus === 'error') kind = 'error'
  else if (connectionStatus === 'reconnecting') kind = 'reconnecting'
  else if (connectionStatus === 'connecting' || !synced) kind = 'syncing'
  else if (connectionStatus === 'read_only' || access === 'view' || !canEdit) kind = 'read_only'
  else if (durabilityStatus === 'pending') kind = 'saving'

  const projectionConfirmedAt = projectionUpdatedAt
    && !Number.isNaN(new Date(projectionUpdatedAt).getTime())
    ? new Date(projectionUpdatedAt).toISOString()
    : null

  return {
    active,
    kind,
    notice,
    participantOverflow: preview.overflow,
    participants,
    projectionConfirmedAt,
    visibleParticipants: preview.visible,
  }
}

export function beginCollaborationPublicationFocus(
  documentId: string,
  patchId: string,
): PendingCollaborationPublicationFocus {
  return { documentId, patchId }
}

export function registerCollaborationPublicationFocus(
  pendingByDocument: PendingCollaborationPublicationFocusByDocument,
  documentId: string,
  patchId: string,
): PendingCollaborationPublicationFocusByDocument {
  return {
    ...pendingByDocument,
    [documentId]: beginCollaborationPublicationFocus(documentId, patchId),
  }
}

export function pendingCollaborationPublicationFocusForDocument(
  pendingByDocument: PendingCollaborationPublicationFocusByDocument,
  documentId: string | null,
): PendingCollaborationPublicationFocus | null {
  return documentId ? pendingByDocument[documentId] ?? null : null
}

export function isCollaborationPublicationFocusCurrent(
  pending: PendingCollaborationPublicationFocus,
  documentId: string | null,
): boolean {
  return pending.documentId === documentId
}

export function consumeCollaborationPublicationFocus(
  pending: PendingCollaborationPublicationFocus | null,
  documentId: string | null,
  changes: readonly InspectorChange[],
): {
  focusId: string | null
  pending: PendingCollaborationPublicationFocus | null
} {
  if (!pending || !isCollaborationPublicationFocusCurrent(pending, documentId)) {
    return { focusId: null, pending }
  }
  const publishedChange = changes.find((change) => change.id === pending.patchId)
  if (!publishedChange) return { focusId: null, pending }
  return { focusId: publishedChange.id, pending: null }
}

export type InspectorSuggestionExcerpt = {
  deletionText: string
  insertionText: string
  modificationText: string
  position: number
}

export type InspectorChange = {
  author: InspectorParticipant
  createdAt: number
  id: string
  originalText: string
  position: number
  proposedText: string
  suggestionIds: string[]
  type: SuggestionKind
}

export type InspectorHistoryEntry = {
  actor: InspectorParticipant
  actorKind: CollaborationActorKind
  commandId: string | null
  createdAt: number
  fromSequence: number
  id: string
  suggestionIds: string[]
  toSequence: number
  type: CollaborationChangeKind
}

export type InspectorOpenFilters = {
  authorId: string | null
  type: SuggestionKind | null
}

export type InspectorHistoryFilters = {
  actorId: string | null
  type: CollaborationChangeKind | null
}

export function buildInspectorChanges(
  descriptors: readonly SuggestionDescriptor[],
  excerpts: ReadonlyMap<string, InspectorSuggestionExcerpt>,
  participants: readonly InspectorParticipant[],
  durableSuggestionAuthors: ReadonlyMap<string, InspectorParticipant> = new Map(),
): InspectorChange[] {
  const participantById = new Map(participants.map((participant) => [participant.id, participant]))
  const patches = new Map<string, SuggestionDescriptor[]>()
  for (const descriptor of descriptors) {
    const existing = patches.get(descriptor.patchId)
    if (existing) existing.push(descriptor)
    else patches.set(descriptor.patchId, [descriptor])
  }

  return [...patches.entries()].map(([patchId, patchDescriptors]) => {
    const first = patchDescriptors[0]
    if (!first) throw new Error(`Patch ${patchId} has no suggestions`)
    const patchExcerpts = patchDescriptors.map((descriptor) => excerpts.get(descriptor.suggestionId))
    const originalText = joinExcerptText(patchExcerpts, 'deletionText')
    const insertedText = joinExcerptText(patchExcerpts, 'insertionText')
    const modifiedText = joinExcerptText(patchExcerpts, 'modificationText')
    const positions = patchExcerpts
      .map((excerpt) => excerpt?.position)
      .filter((position): position is number => typeof position === 'number')
    const durableAuthor = patchDescriptors
      .map((descriptor) => durableSuggestionAuthors.get(descriptor.suggestionId))
      .find((participant) => participant !== undefined)
    return {
      author: durableAuthor ?? participantById.get(first.authorId) ?? {
        color: '#6b7280',
        id: first.authorId,
        name: 'Collaborator',
      },
      createdAt: first.createdAt,
      id: patchId,
      originalText: originalText || modifiedText,
      position: positions.length > 0 ? Math.min(...positions) : Number.MAX_SAFE_INTEGER,
      proposedText: insertedText || modifiedText,
      suggestionIds: patchDescriptors.map((descriptor) => descriptor.suggestionId).sort(),
      type: patchType(patchDescriptors.map((descriptor) => descriptor.kind)),
    }
  }).sort((left, right) => (
    left.position - right.position
    || left.createdAt - right.createdAt
    || left.id.localeCompare(right.id)
  ))
}

function joinExcerptText(
  excerpts: readonly (InspectorSuggestionExcerpt | undefined)[],
  key: 'deletionText' | 'insertionText' | 'modificationText',
): string {
  return excerpts
    .map((excerpt) => excerpt?.[key].trim() ?? '')
    .filter(Boolean)
    .join(' ')
}

function patchType(types: readonly SuggestionKind[]): SuggestionKind {
  const distinct = new Set(types)
  if (distinct.size === 1) return types[0] ?? 'modification'
  return 'modification'
}

export function filterInspectorChanges(
  changes: readonly InspectorChange[],
  filters: InspectorOpenFilters,
): InspectorChange[] {
  return changes.filter((change) => (
    (filters.authorId === null || change.author.id === filters.authorId)
    && (filters.type === null || change.type === filters.type)
  ))
}

export function filterInspectorHistory(
  history: readonly InspectorHistoryEntry[],
  filters: InspectorHistoryFilters,
): InspectorHistoryEntry[] {
  return history.filter((entry) => (
    (filters.actorId === null || entry.actor.id === filters.actorId)
    && (filters.type === null || entry.type === filters.type)
  ))
}

export function participantPreview(
  participants: readonly InspectorParticipant[],
  maximum = 3,
): { overflow: number; visible: InspectorParticipant[] } {
  const limit = Math.max(0, Math.floor(maximum))
  return {
    overflow: Math.max(0, participants.length - limit),
    visible: participants.slice(0, limit),
  }
}

export function adjacentChangeId(
  changes: readonly InspectorChange[],
  selectedId: string | null,
  direction: -1 | 1,
): string | null {
  if (changes.length === 0) return null
  const selectedIndex = selectedId === null
    ? (direction === 1 ? -1 : changes.length)
    : changes.findIndex((change) => change.id === selectedId)
  if (selectedIndex < 0 && selectedId !== null) {
    return direction === 1 ? changes[0]?.id ?? null : changes.at(-1)?.id ?? null
  }
  const nextIndex = Math.max(0, Math.min(changes.length - 1, selectedIndex + direction))
  return changes[nextIndex]?.id ?? null
}

export function compactChangeText(change: InspectorChange): string {
  return (change.proposedText || change.originalText).replace(/\s+/g, ' ').trim()
}

export function effectiveEditorWriteMode(
  access: 'edit' | 'suggest' | 'view' | null,
  canEdit: boolean,
  requested: Exclude<EditorWriteMode, 'view'>,
): EditorWriteMode {
  if (!canEdit || access === null || access === 'view') return 'view'
  if (access === 'suggest') return 'suggest'
  return requested
}

export function isOwnedEditorDocument(
  document: { access?: { mode: 'owner' | 'shared' } },
): boolean {
  return document.access?.mode !== 'shared'
}

export function partitionEditorDocumentsByAccess<
  Document extends { access?: { mode: 'owner' | 'shared' } },
>(documents: readonly Document[]): { owned: Document[]; shared: Document[] } {
  const owned: Document[] = []
  const shared: Document[] = []
  for (const document of documents) {
    if (isOwnedEditorDocument(document)) owned.push(document)
    else shared.push(document)
  }
  return { owned, shared }
}
