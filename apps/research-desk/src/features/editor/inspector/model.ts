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

export type EditorInspectorTab = 'assistant' | 'changes' | 'comments'
export type EditorChangesView = 'history' | 'open'
export type EditorWriteMode = 'comment' | 'edit' | 'suggest' | 'view'
export type InspectorHistoryKind = CollaborationChangeKind | 'comment'

export type InspectorParticipant = {
  color: string
  id: string
  name: string
}

export type EditorCollaborationStatusKind =
  | 'access_revoked'
  | 'error'
  | 'inactive'
  | 'origin_rejected'
  | 'read_only'
  | 'reconnecting'
  | 'saved'
  | 'saving'
  | 'syncing'
  | 'reload_required'

export type EditorCollaborationStatusModel = {
  active: boolean
  hasUnconfirmedLocalChanges: boolean
  kind: EditorCollaborationStatusKind
  nextReconnectAt: number | null
  notice: string | null
  participantOverflow: number
  participants: readonly InspectorParticipant[]
  projectionConfirmedAt: string | null
  reconnectAttempt: number
  recoverability: 'login' | 'none' | 'reload' | 'retry'
  visibleParticipants: readonly InspectorParticipant[]
}

const GERMAN_COLLABORATION_NOTICES: Readonly<Record<string, string>> = {
  'Access to this collaboration document is unavailable.':
    'Der Zugriff auf dieses Kollaborationsdokument ist nicht mehr verfügbar.',
  'Access to this collaboration document was revoked.':
    'Der Zugriff auf dieses Kollaborationsdokument wurde entzogen.',
  'Collaboration authentication failed; reconnecting read-only.':
    'Die Kollaborationssitzung konnte nicht bestätigt werden. Inqtrix stellt die Verbindung automatisch wieder her.',
  'Collaboration access changed; revalidating read-only.':
    'Der Kollaborationszugriff hat sich geändert. Inqtrix prüft die aktuelle Berechtigung erneut.',
  'Collaboration durability could not be reconciled; reconnecting read-only.':
    'Nicht bestätigte Änderungen werden erneut abgeglichen. Der Editor bleibt bis dahin schreibgeschützt.',
  'Collaboration identity changed while refreshing access.':
    'Die Identität der Kollaborationssitzung hat sich geändert. Der Zugriff wurde vorsorglich beendet.',
  'The collaboration connection was interrupted; reconnecting read-only.':
    'Die Verbindung zum Kollaborationsdienst wurde unterbrochen. Inqtrix verbindet sich automatisch erneut.',
  'The collaboration lease could not be refreshed; reconnecting read-only.':
    'Die Verbindung zum Kollaborationsdienst ist unterbrochen. Inqtrix verbindet sich automatisch erneut.',
  'The collaboration protocol is not compatible with this client.':
    'Diese App-Version ist nicht mit dem Kollaborationsprotokoll kompatibel.',
  'The collaboration protocol or schema is not compatible.':
    'Diese App-Version ist nicht mit dem Kollaborationsprotokoll oder Dokumentschema kompatibel.',
  'The collaboration service ended this session and will not resume it with the current state. Reload the page to continue.':
    'Der Kollaborationsdienst hat diese Sitzung beendet und nimmt sie mit dem aktuellen Stand nicht wieder auf. Laden Sie die Seite neu, um weiterzuarbeiten.',
  'The collaboration session expired. Sign in again to continue.':
    'Die Anmeldung ist abgelaufen. Melden Sie sich erneut an, um fortzufahren.',
  'The document schema is not supported by this client.':
    'Diese App-Version unterstützt das Dokumentschema nicht.',
}

export function localizeEditorCollaborationNotice(
  notice: string | null,
  locale: 'de' | 'en',
): string | null {
  if (!notice || locale !== 'de') return notice
  return GERMAN_COLLABORATION_NOTICES[notice] ?? notice
}

export type PendingCollaborationPublicationFocus = {
  documentId: string
  patchId: string
}

export type PendingCollaborationPublicationFocusByDocument = Readonly<
  Record<string, PendingCollaborationPublicationFocus>
>

type CollaborationStatusInput = {
  access: 'comment' | 'edit' | 'suggest' | 'view' | null
  active: boolean
  canEdit: boolean
  connectionStatus: EditorCollaborationConnectionStatus
  durabilityStatus: EditorCollaborationDurabilityStatus
  hasUnconfirmedLocalChanges?: boolean
  nextReconnectAt?: number | null
  notice?: string | null
  participants: readonly InspectorParticipant[]
  projectionUpdatedAt?: string | null
  reconnectAttempt?: number
  recoverability?: 'login' | 'none' | 'reload' | 'retry'
  synced: boolean
}

export function buildEditorCollaborationStatusModel({
  access,
  active,
  canEdit,
  connectionStatus,
  durabilityStatus,
  hasUnconfirmedLocalChanges = false,
  nextReconnectAt = null,
  notice = null,
  participants,
  projectionUpdatedAt = null,
  reconnectAttempt = 0,
  recoverability = 'none',
  synced,
}: CollaborationStatusInput): EditorCollaborationStatusModel {
  const preview = participantPreview(participants)
  let kind: EditorCollaborationStatusKind = 'saved'
  if (!active) kind = 'inactive'
  else if (connectionStatus === 'access_revoked') kind = 'access_revoked'
  else if (connectionStatus === 'incompatible') kind = 'reload_required'
  else if (connectionStatus === 'origin_rejected') kind = 'origin_rejected'
  else if (connectionStatus === 'error' || durabilityStatus === 'error') kind = 'error'
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
    hasUnconfirmedLocalChanges,
    kind,
    nextReconnectAt,
    notice,
    participantOverflow: preview.overflow,
    participants,
    projectionConfirmedAt,
    reconnectAttempt,
    recoverability,
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
  commentAction?: 'created' | 'message_deleted' | 'message_edited' | 'reopened' | 'replied' | 'resolved'
  createdAt: number
  fromSequence: number
  id: string
  outcome?: 'accepted' | 'rejected' | null
  summary?: Array<{
    after: string
    before: string
    kind: SuggestionKind | 'direct'
    position: number
  }>
  omittedEditCount?: number
  suggestionIds: string[]
  toSequence: number
  type: InspectorHistoryKind
  updateCount?: number
}

export type InspectorOpenFilters = {
  authorId: string | null
  type: SuggestionKind | null
}

export type InspectorHistoryFilters = {
  actorId: string | null
  type: InspectorHistoryKind | null
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
  if (distinct.size === 1) return types[0] ?? 'replacement'
  if (distinct.has('structure')) return 'structure'
  if (distinct.has('format')) return 'format'
  return 'replacement'
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
  access: 'comment' | 'edit' | 'suggest' | 'view' | null,
  canEdit: boolean,
  requested: Exclude<EditorWriteMode, 'view'>,
): EditorWriteMode {
  if (!canEdit || access === null || access === 'view') return 'view'
  if (access === 'comment') return 'comment'
  if (access === 'suggest') {
    return requested === 'comment' ? 'comment' : 'suggest'
  }
  return requested
}

export function isOwnedEditorDocument(
  document: { access?: { mode: 'owner' | 'shared' } },
): boolean {
  return document.access?.mode !== 'shared'
}

/** What a document row IS, for the rail's interaction grammar:
 * - `owned-private`: mine, not in collaboration — there is no details page for
 *   it, so the row offers no details action.
 * - `owned-shared`: mine and in collaboration — details (access, activity)
 *   exist and the row marks it as shared.
 * - `shared-with-me`: someone else's document shared to me.
 *
 * Derived from exactly the facts that gate the details dialog, so the visible
 * affordance and the destination can never drift apart. A document that
 * carries neither field (a local file, a legacy save) reads as private. */
export type EditorDocumentRowKind = 'owned-private' | 'owned-shared' | 'shared-with-me'

export function editorDocumentRowKind(
  document: { access?: { mode: 'owner' | 'shared' }; contentMode?: string },
): EditorDocumentRowKind {
  if (!isOwnedEditorDocument(document)) return 'shared-with-me'
  return document.contentMode === 'collaboration' ? 'owned-shared' : 'owned-private'
}

export type EditorDocumentRailCapabilities = {
  canDelete: boolean
  canDrag: boolean
  canOpenDetails: boolean
  canPin: boolean
  canRename: boolean
  leadingRole: 'file' | 'people'
}

/** One semantic source for the editor rail's leading role and actions.
 * Rendering code supplies labels/icons, but it may not independently infer
 * ownership or collaboration affordances. */
export function editorDocumentRailCapabilities(
  document: { access?: { mode: 'owner' | 'shared' }; contentMode?: string },
): EditorDocumentRailCapabilities {
  const kind = editorDocumentRowKind(document)
  if (kind === 'shared-with-me') {
    return {
      canDelete: false,
      canDrag: false,
      canOpenDetails: true,
      canPin: false,
      canRename: false,
      leadingRole: 'people',
    }
  }
  return {
    canDelete: true,
    canDrag: true,
    canOpenDetails: kind === 'owned-shared',
    canPin: true,
    canRename: true,
    leadingRole: kind === 'owned-shared' ? 'people' : 'file',
  }
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

/** How long the startup transients may present CALM before they earn color.
 *
 * Opening a document legitimately passes inactive -> syncing -> saved within
 * a few hundred milliseconds. Showing that sequence verbatim flashes a gray
 * dot, an amber dot and three label swaps for a state nobody can act on.
 * Within this grace window both startup transients present as ONE quiet
 * "syncing" with a muted dot; a session still not up after the window shows
 * its real state — and every exceptional kind bypasses the calm entirely
 * (the useCalmCollaborationStatusKind philosophy, applied to startup).
 */
export const COLLABORATION_STARTUP_GRACE_MS = 1_200

const STARTUP_TRANSIENT_KINDS: ReadonlySet<EditorCollaborationStatusKind> = new Set([
  'inactive',
  'syncing',
])

export function startupPresentation(
  kind: EditorCollaborationStatusKind,
  sinceMountMs: number,
  collaborationExpected: boolean,
): { calm: boolean; kind: EditorCollaborationStatusKind } {
  // A local markdown document is FINAL `inactive` ("Lokal") — there is no
  // session coming, so the grace would show 1.2s of "syncing" for a state
  // that was already the truth. Only documents that actually start a
  // collaboration session get the calm window.
  if (!collaborationExpected) return { calm: false, kind }
  if (!STARTUP_TRANSIENT_KINDS.has(kind)) return { calm: false, kind }
  if (sinceMountMs >= COLLABORATION_STARTUP_GRACE_MS) return { calm: false, kind }
  return { calm: true, kind: 'syncing' }
}
