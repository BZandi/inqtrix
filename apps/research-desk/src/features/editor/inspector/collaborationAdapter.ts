import type { HocuspocusProvider } from '@hocuspocus/provider'
import {
  SUGGESTION_MARK_NAMES,
  suggestionDescriptors,
  type CollaborationChangeKind,
  type SuggestionKind,
} from '@inqtrix/editor-schema'
import type { Editor } from '@tiptap/react'
import type { Transaction } from '@tiptap/pm/state'
import { useEffect, useMemo, useReducer, useState } from 'react'

import {
  listEditorCollaborationActivity,
  type EditorCollaborationActivity,
  type EditorCollaborationUser,
} from '@/api/inqtrixClient'
import {
  beginCollaborationAuthorityGuard,
  CollaborationAuthorityError,
  type CollaborationAuthorityGuard,
  type CollaborationAuthoritySource,
} from '../collaborationAuthority'
import type { CollaborationDocumentHandle } from '../useCollaborationDocument'
import {
  buildInspectorChanges,
  type InspectorChange,
  type InspectorHistoryEntry,
  type InspectorParticipant,
  type InspectorSuggestionExcerpt,
} from './model'
import {
  collaborationSuggestionCollisionEvent,
  collaborationSuggestionErrorEvent,
  type CollaborationSuggestionCollision,
} from '../tiptap'

type AwarenessAdapter = {
  getStates?: () => Map<number, unknown>
  off?: (event: 'change' | 'update', listener: () => void) => void
  on?: (event: 'change' | 'update', listener: () => void) => void
  states?: Map<number, unknown>
}

export type InspectorCollaborationSnapshot = {
  changes: InspectorChange[]
  collision: CollaborationSuggestionCollision | null
  error: string | null
  participants: InspectorParticipant[]
}

export function useInspectorCollaborationSnapshot(
  editor: Editor | null,
  collaboration: CollaborationDocumentHandle,
  durableSuggestionAuthors: ReadonlyMap<string, InspectorParticipant> = new Map(),
): InspectorCollaborationSnapshot {
  const [revision, refresh] = useReducer((value: number) => value + 1, 0)
  const [suggestionError, setSuggestionError] = useState<string | null>(null)
  const [collision, setCollision] = useState<CollaborationSuggestionCollision | null>(null)
  const provider = collaboration.provider

  useEffect(() => {
    if (!editor) return
    const handleTransaction = ({ transaction }: { transaction: Transaction }) => {
      if (transaction.docChanged) refresh()
    }
    editor.on('transaction', handleTransaction)
    return () => {
      editor.off('transaction', handleTransaction)
    }
  }, [editor])

  useEffect(() => {
    if (!editor) {
      setSuggestionError(null)
      setCollision(null)
      return
    }
    const handleSuggestionError = (event: Event) => {
      const detail = (event as CustomEvent<string | null>).detail
      setSuggestionError(typeof detail === 'string' && detail ? detail : null)
    }
    const handleSuggestionCollision = (event: Event) => {
      const detail = (event as CustomEvent<CollaborationSuggestionCollision | null>).detail
      setCollision(detail?.patchId && detail.suggestionId ? detail : null)
    }
    editor.view.dom.addEventListener(collaborationSuggestionErrorEvent, handleSuggestionError)
    editor.view.dom.addEventListener(
      collaborationSuggestionCollisionEvent,
      handleSuggestionCollision,
    )
    return () => {
      editor.view.dom.removeEventListener(collaborationSuggestionErrorEvent, handleSuggestionError)
      editor.view.dom.removeEventListener(
        collaborationSuggestionCollisionEvent,
        handleSuggestionCollision,
      )
    }
  }, [editor])

  useEffect(() => {
    const awareness = providerAwareness(provider)
    if (!awareness?.on || !awareness.off) return
    const handleAwareness = () => refresh()
    awareness.on('change', handleAwareness)
    awareness.on('update', handleAwareness)
    return () => {
      awareness.off?.('change', handleAwareness)
      awareness.off?.('update', handleAwareness)
    }
  }, [provider])

  const userSignature = collaboration.user
    ? `${collaboration.user.id}:${collaboration.user.name}:${collaboration.user.color}`
    : ''
  return useMemo(() => {
    const participants = readInspectorParticipants(provider, collaboration.user)
    if (!editor) return { changes: [], collision, error: suggestionError, participants }
    try {
      const descriptors = suggestionDescriptors(editor.state.doc)
      const excerpts = suggestionExcerpts(editor)
      return {
        changes: buildInspectorChanges(
          descriptors,
          excerpts,
          participants,
          durableSuggestionAuthors,
        ),
        collision,
        error: suggestionError,
        participants,
      }
    } catch (error) {
      return {
        changes: [],
        collision,
        error: messageFromError(error),
        participants,
      }
    }
  }, [
    collision,
    durableSuggestionAuthors,
    editor,
    provider,
    revision,
    suggestionError,
    userSignature,
  ])
}

export function useInspectorOpenSuggestionIds(editor: Editor | null): readonly string[] {
  const [revision, refresh] = useReducer((value: number) => value + 1, 0)

  useEffect(() => {
    if (!editor) return
    const handleTransaction = ({ transaction }: { transaction: Transaction }) => {
      if (transaction.docChanged) refresh()
    }
    editor.on('transaction', handleTransaction)
    return () => {
      editor.off('transaction', handleTransaction)
    }
  }, [editor])

  return useMemo(() => {
    if (!editor) return []
    try {
      return [...new Set(
        suggestionDescriptors(editor.state.doc).map((descriptor) => descriptor.suggestionId),
      )].sort()
    } catch {
      return []
    }
  }, [editor, revision])
}

export function readInspectorParticipants(
  provider: HocuspocusProvider | null,
  localUser: EditorCollaborationUser | null,
): InspectorParticipant[] {
  const participants = new Map<string, InspectorParticipant>()
  if (localUser) participants.set(localUser.id, normalizedParticipant(localUser))
  const awareness = providerAwareness(provider)
  const states = awareness?.getStates?.() ?? awareness?.states
  for (const state of states?.values() ?? []) {
    const participant = participantFromAwarenessState(state)
    if (participant) participants.set(participant.id, participant)
  }
  return [...participants.values()].sort((left, right) => {
    if (left.id === localUser?.id) return -1
    if (right.id === localUser?.id) return 1
    return left.name.localeCompare(right.name)
  })
}

function providerAwareness(provider: HocuspocusProvider | null): AwarenessAdapter | null {
  if (!provider) return null
  const candidate = (provider as unknown as { awareness?: AwarenessAdapter }).awareness
  return candidate ?? null
}

function participantFromAwarenessState(state: unknown): InspectorParticipant | null {
  if (!state || typeof state !== 'object') return null
  const user = (state as { user?: unknown }).user
  if (!user || typeof user !== 'object') return null
  const candidate = user as Partial<EditorCollaborationUser>
  if (
    typeof candidate.id !== 'string'
    || typeof candidate.name !== 'string'
    || typeof candidate.color !== 'string'
  ) return null
  return normalizedParticipant({
    color: candidate.color,
    id: candidate.id,
    name: candidate.name,
  })
}

function normalizedParticipant(user: EditorCollaborationUser): InspectorParticipant {
  return {
    color: /^#[0-9a-f]{6}$/i.test(user.color) ? user.color : '#6b7280',
    id: user.id,
    name: user.name.trim() || 'Collaborator',
  }
}

export function suggestionExcerpts(editor: Editor): Map<string, InspectorSuggestionExcerpt> {
  const excerpts = new Map<string, InspectorSuggestionExcerpt>()
  editor.state.doc.descendants((node, position) => {
    for (const mark of node.marks) {
      if (!SUGGESTION_MARK_NAMES.has(mark.type.name as SuggestionKind)) continue
      const suggestionId = mark.attrs.suggestionId
      if (typeof suggestionId !== 'string' || !suggestionId) continue
      const existing = excerpts.get(suggestionId) ?? {
        deletionText: '',
        insertionText: '',
        modificationText: '',
        position,
      }
      const text = node.textContent
      const key = `${mark.type.name}Text` as
        | 'deletionText'
        | 'insertionText'
        | 'modificationText'
      existing[key] += text
      existing.position = Math.min(existing.position, position)
      excerpts.set(suggestionId, existing)
    }
    return true
  })
  return excerpts
}

type UseInspectorActivityOptions = {
  active: boolean
  activityRevision: number
  apiKey?: string
  authorityRevision: number
  documentId: string | null
  generation: number | null
  lifecycleKey: string
  locale: 'de' | 'en'
  openSuggestionIds: readonly string[]
  readAuthority: CollaborationAuthoritySource['readAuthority']
  workspaceId: string
}

type InspectorActivityState = {
  activity: EditorCollaborationActivity[]
  attributionActivity: EditorCollaborationActivity[]
  attributionWarning: string | null
  authorityRevision: number | null
  documentId: string | null
  error: string | null
  isLoading: boolean
  lifecycleKey: string | null
}

export function useInspectorCollaborationActivity({
  active,
  activityRevision,
  apiKey,
  authorityRevision,
  documentId,
  generation,
  lifecycleKey,
  locale,
  openSuggestionIds,
  readAuthority,
  workspaceId,
}: UseInspectorActivityOptions): {
  attributionWarning: string | null
  entries: InspectorHistoryEntry[]
  error: string | null
  isLoading: boolean
  suggestionAuthors: ReadonlyMap<string, InspectorParticipant>
} {
  const [requestState, setRequestState] = useState<InspectorActivityState>({
    activity: [],
    attributionActivity: [],
    attributionWarning: null,
    authorityRevision: null,
    documentId: null,
    error: null,
    isLoading: false,
    lifecycleKey: null,
  })
  const openSuggestionIdsSignature = [...openSuggestionIds].sort().join('\u0000')

  useEffect(() => {
    if (!active || !documentId || generation === null) return
    const controller = new AbortController()
    let authorityGuard: CollaborationAuthorityGuard
    try {
      authorityGuard = beginCollaborationAuthorityGuard(
        { readAuthority },
        { documentId, generation },
        'read',
        locale,
      )
    } catch {
      return () => controller.abort()
    }
    setRequestState((current) => ({
      activity: current.documentId === documentId && current.lifecycleKey === lifecycleKey
        ? current.activity
        : [],
      attributionActivity: current.documentId === documentId && current.lifecycleKey === lifecycleKey
        ? current.attributionActivity
        : [],
      attributionWarning: current.documentId === documentId && current.lifecycleKey === lifecycleKey
        ? current.attributionWarning
        : null,
      authorityRevision,
      documentId,
      error: null,
      isLoading: true,
      lifecycleKey,
    }))
    void loadInspectorCollaborationActivity({
      apiKey,
      authorityGuard,
      documentId,
      openSuggestionIds,
      signal: controller.signal,
      workspaceId,
    }).then((response) => {
      authorityGuard.assertCurrent()
      if (!controller.signal.aborted) {
        setRequestState({
          activity: response.data,
          attributionActivity: response.attributionData,
          attributionWarning: inspectorAttributionWarning(response, locale),
          authorityRevision,
          documentId,
          error: null,
          isLoading: false,
          lifecycleKey,
        })
      }
    }).catch((requestError: unknown) => {
      if (!controller.signal.aborted && !(requestError instanceof CollaborationAuthorityError)) {
        setRequestState((current) => ({
          activity: current.documentId === documentId && current.lifecycleKey === lifecycleKey
            ? current.activity
            : [],
          attributionActivity: current.documentId === documentId && current.lifecycleKey === lifecycleKey
            ? current.attributionActivity
            : [],
          attributionWarning: current.documentId === documentId && current.lifecycleKey === lifecycleKey
            ? current.attributionWarning
            : null,
          authorityRevision,
          documentId,
          error: messageFromError(requestError),
          isLoading: false,
          lifecycleKey,
        }))
      }
    })
    return () => controller.abort()
  }, [
    active,
    activityRevision,
    apiKey,
    authorityRevision,
    documentId,
    generation,
    lifecycleKey,
    locale,
    openSuggestionIdsSignature,
    readAuthority,
    workspaceId,
  ])

  const currentRequest = active
    && requestState.authorityRevision === authorityRevision
    && requestState.documentId === documentId
    && requestState.lifecycleKey === lifecycleKey
  const activity = currentRequest
    ? requestState.activity
    : []
  const attributionActivity = currentRequest
    ? requestState.attributionActivity
    : []
  const entries = useMemo(
    () => normalizeActivity(activity),
    [activity],
  )
  const suggestionAuthors = useMemo(
    () => durableSuggestionAuthorsFromActivity([...activity, ...attributionActivity]),
    [activity, attributionActivity],
  )
  return {
    attributionWarning: currentRequest ? requestState.attributionWarning : null,
    entries,
    error: currentRequest ? requestState.error : null,
    isLoading: currentRequest && requestState.isLoading,
    suggestionAuthors,
  }
}

type InspectorActivityPage = Awaited<ReturnType<typeof listEditorCollaborationActivity>>

const INSPECTOR_ACTIVITY_PAGE_SIZE = 100
// Five pages cap each refresh at 500 canonical rows. The bound is tied to the
// endpoint's established page size, so unique cursors can never cause an
// unbounded attribution walk.
export const INSPECTOR_ATTRIBUTION_MAX_PAGES = 5

type LoadInspectorActivityOptions = {
  apiKey?: string
  authorityGuard?: CollaborationAuthorityGuard | null
  documentId: string
  fetchActivity?: typeof listEditorCollaborationActivity
  openSuggestionIds: readonly string[]
  signal?: AbortSignal
  workspaceId: string
}

export async function loadInspectorCollaborationActivity({
  apiKey,
  authorityGuard = null,
  documentId,
  fetchActivity = listEditorCollaborationActivity,
  openSuggestionIds,
  signal,
  workspaceId,
}: LoadInspectorActivityOptions): Promise<{
  attributionComplete: boolean
  attributionData: EditorCollaborationActivity[]
  data: EditorCollaborationActivity[]
  lookupLimited: boolean
  unresolvedSuggestionCount: number
}> {
  authorityGuard?.assertCurrent()
  const firstPage = await fetchActivity(documentId, {
    apiKey,
    limit: INSPECTOR_ACTIVITY_PAGE_SIZE,
    signal,
    workspaceId,
  })
  authorityGuard?.assertCurrent()
  const missingSuggestionIds = new Set(openSuggestionIds)
  removeAttributedSuggestionIds(missingSuggestionIds, firstPage.data)
  const attributionData: EditorCollaborationActivity[] = []
  const visitedCursors = new Set<string>()
  let cursor = firstPage.next_cursor
  let pagesRead = 1

  while (
    cursor
    && missingSuggestionIds.size > 0
    && pagesRead < INSPECTOR_ATTRIBUTION_MAX_PAGES
    && !visitedCursors.has(cursor)
  ) {
    visitedCursors.add(cursor)
    authorityGuard?.assertCurrent()
    const page: InspectorActivityPage = await fetchActivity(documentId, {
      apiKey,
      cursor,
      limit: INSPECTOR_ACTIVITY_PAGE_SIZE,
      signal,
      workspaceId,
    })
    authorityGuard?.assertCurrent()
    pagesRead += 1
    for (const item of page.data) {
      if (
        item.type === 'suggestion'
        && item.suggestion_ids.some((suggestionId) => missingSuggestionIds.has(suggestionId))
      ) {
        attributionData.push(item)
      }
    }
    removeAttributedSuggestionIds(missingSuggestionIds, page.data)
    cursor = page.next_cursor
  }

  const attributionComplete = missingSuggestionIds.size === 0
  return {
    attributionComplete,
    attributionData,
    data: firstPage.data,
    lookupLimited: !attributionComplete
      && cursor !== null
      && pagesRead >= INSPECTOR_ATTRIBUTION_MAX_PAGES,
    unresolvedSuggestionCount: missingSuggestionIds.size,
  }
}

export function inspectorAttributionWarning(
  result: Pick<
    Awaited<ReturnType<typeof loadInspectorCollaborationActivity>>,
    'attributionComplete' | 'lookupLimited' | 'unresolvedSuggestionCount'
  >,
  locale: 'de' | 'en',
): string | null {
  if (result.attributionComplete || result.unresolvedSuggestionCount === 0) return null
  const count = result.unresolvedSuggestionCount
  const rowLimit = INSPECTOR_ACTIVITY_PAGE_SIZE * INSPECTOR_ATTRIBUTION_MAX_PAGES
  if (locale === 'de') {
    const changes = count === 1 ? 'offene Änderung' : 'offene Änderungen'
    return result.lookupLimited
      ? `Die Autorenzuordnung ist für ${count} ${changes} unvollständig, weil die begrenzte Suche nach ${rowLimit} Aktivitätseinträgen beendet wurde.`
      : `Für ${count} ${changes} ist keine kanonische Autorenzuordnung verfügbar.`
  }
  const changes = count === 1 ? 'open change' : 'open changes'
  return result.lookupLimited
    ? `Author attribution is incomplete for ${count} ${changes} because the bounded ${rowLimit}-row activity lookup was exhausted.`
    : `Canonical author attribution is unavailable for ${count} ${changes}.`
}

function removeAttributedSuggestionIds(
  missingSuggestionIds: Set<string>,
  activity: readonly EditorCollaborationActivity[],
): void {
  for (const item of activity) {
    if (item.type !== 'suggestion') continue
    for (const suggestionId of item.suggestion_ids) missingSuggestionIds.delete(suggestionId)
  }
}

export function normalizeActivity(
  activity: readonly EditorCollaborationActivity[],
): InspectorHistoryEntry[] {
  return activity.map((item) => {
    return {
      actor: activityActor(item),
      actorKind: item.actor_kind,
      commandId: item.command_id,
      createdAt: item.created_at * 1_000,
      fromSequence: item.from_sequence,
      id: `${item.from_sequence}:${item.to_sequence}:${item.command_id ?? item.type}`,
      suggestionIds: item.suggestion_ids,
      toSequence: item.to_sequence,
      type: item.type,
    }
  })
}

/** Derive attribution from canonical suggestion events. Decision events may
 * mention the same suggestion IDs but describe the reviewer, not the author. */
export function durableSuggestionAuthorsFromActivity(
  activity: readonly EditorCollaborationActivity[],
): ReadonlyMap<string, InspectorParticipant> {
  const authors = new Map<string, InspectorParticipant>()
  for (const item of activity) {
    if (item.type !== 'suggestion') continue
    const actor = activityActor(item)
    for (const suggestionId of item.suggestion_ids) {
      if (!authors.has(suggestionId)) authors.set(suggestionId, actor)
    }
  }
  return authors
}

function activityActor(item: EditorCollaborationActivity): InspectorParticipant {
  const name = item.actor.name.trim()
  const id = item.actor.id ?? `${item.actor_kind}:${name || 'unknown'}`
  return {
    color: actorColor(item.actor_kind),
    id,
    name: name || activityActorFallbackName(item.actor_kind),
  }
}

function activityActorFallbackName(kind: EditorCollaborationActivity['actor_kind']): string {
  const names: Record<EditorCollaborationActivity['actor_kind'], string> = {
    assistant: 'Assistant',
    agent: 'Agent',
    human: 'Collaborator',
    system: 'System',
  }
  return names[kind]
}

function actorColor(kind: EditorCollaborationActivity['actor_kind']): string {
  const colors: Record<EditorCollaborationActivity['actor_kind'], string> = {
    assistant: '#7c3aed',
    agent: '#0f766e',
    human: '#6b7280',
    system: '#64748b',
  }
  return colors[kind]
}

export function collaborationChangeKinds(
  entries: readonly InspectorHistoryEntry[],
): CollaborationChangeKind[] {
  return [...new Set(entries.map((entry) => entry.type))]
}

function messageFromError(error: unknown): string {
  return error instanceof Error ? error.message : 'Collaboration data could not be loaded.'
}
