import type { SuggestionDescriptor } from '@inqtrix/editor-schema'
import { describe, expect, it } from 'vitest'

import { collaborationEditorPolicyUpdate } from './editorPolicy'
import {
  adjacentChangeId,
  beginCollaborationPublicationFocus,
  buildEditorCollaborationStatusModel,
  buildInspectorChanges,
  consumeCollaborationPublicationFocus,
  isCollaborationPublicationFocusCurrent,
  localizeEditorCollaborationNotice,
  pendingCollaborationPublicationFocusForDocument,
  registerCollaborationPublicationFocus,
  effectiveEditorWriteMode,
  filterInspectorChanges,
  editorDocumentRowKind,
  editorDocumentRailCapabilities,
  isOwnedEditorDocument,
  participantPreview,
  partitionEditorDocumentsByAccess,
  type InspectorChange,
  startupPresentation,
} from './model'

const participants = [
  { color: '#111111', id: 'ada', name: 'Ada' },
  { color: '#222222', id: 'lin', name: 'Lin' },
  { color: '#333333', id: 'max', name: 'Max' },
  { color: '#444444', id: 'zoe', name: 'Zoe' },
]

const changes: InspectorChange[] = [
  {
    author: participants[0]!,
    createdAt: 10,
    id: 'patch-a',
    originalText: 'old',
    position: 2,
    proposedText: 'new',
    suggestionIds: ['suggestion-a'],
    type: 'replacement',
  },
  {
    author: participants[1]!,
    createdAt: 20,
    id: 'patch-b',
    originalText: '',
    position: 8,
    proposedText: 'addition',
    suggestionIds: ['suggestion-b'],
    type: 'insertion',
  },
]

describe('inspector collaboration model', () => {
  it('groups compound suggestions into one ordered patch row', () => {
    const descriptors: SuggestionDescriptor[] = [
      {
        authorId: 'ada',
        createdAt: 10,
        kind: 'deletion',
        patchId: 'patch-a',
        suggestionId: 'delete-a',
      },
      {
        authorId: 'ada',
        createdAt: 10,
        kind: 'insertion',
        patchId: 'patch-a',
        suggestionId: 'insert-a',
      },
    ]
    const excerpts = new Map([
      ['delete-a', { deletionText: 'before', insertionText: '', modificationText: '', position: 9 }],
      ['insert-a', { deletionText: '', insertionText: 'after', modificationText: '', position: 9 }],
    ])

    expect(buildInspectorChanges(descriptors, excerpts, participants)).toEqual([{
      author: participants[0],
      createdAt: 10,
      id: 'patch-a',
      originalText: 'before',
      position: 9,
      proposedText: 'after',
      suggestionIds: ['delete-a', 'insert-a'],
      type: 'replacement',
    }])
  })

  it('keeps navigation and filters deterministic as remote rows change', () => {
    expect(filterInspectorChanges(changes, { authorId: 'lin', type: null }))
      .toEqual([changes[1]])
    expect(adjacentChangeId(changes, null, 1)).toBe('patch-a')
    expect(adjacentChangeId(changes, 'missing', -1)).toBe('patch-b')
    expect(adjacentChangeId(changes, 'patch-b', 1)).toBe('patch-b')
  })

  it('caps participant avatars at three and reports the overflow', () => {
    expect(participantPreview(participants)).toEqual({
      overflow: 1,
      visible: participants.slice(0, 3),
    })
  })

  it('derives one status model for lifecycle, durability, projection, and avatars', () => {
    const syncing = buildEditorCollaborationStatusModel({
      access: 'edit',
      active: true,
      canEdit: false,
      connectionStatus: 'connected',
      durabilityStatus: 'idle',
      participants,
      projectionUpdatedAt: '2026-07-15T10:00:00.000Z',
      synced: false,
    })
    expect(syncing).toMatchObject({
      projectionConfirmedAt: '2026-07-15T10:00:00.000Z',
      kind: 'syncing',
      participantOverflow: 1,
      visibleParticipants: participants.slice(0, 3),
    })

    const saving = buildEditorCollaborationStatusModel({
      access: 'edit',
      active: true,
      canEdit: true,
      connectionStatus: 'connected',
      durabilityStatus: 'pending',
      participants: [],
      projectionUpdatedAt: '2026-07-15T10:01:00.000Z',
      synced: true,
    })
    expect(saving).toMatchObject({
      kind: 'saving',
      projectionConfirmedAt: '2026-07-15T10:01:00.000Z',
    })
    expect(buildEditorCollaborationStatusModel({
      access: 'edit',
      active: true,
      canEdit: true,
      connectionStatus: 'connected',
      durabilityStatus: 'saved',
      participants: [],
      projectionUpdatedAt: '2026-07-15T10:02:00.000Z',
      synced: true,
    })).toMatchObject({
      kind: 'saved',
      projectionConfirmedAt: '2026-07-15T10:02:00.000Z',
    })
    expect(buildEditorCollaborationStatusModel({
      access: 'view',
      active: true,
      canEdit: false,
      connectionStatus: 'read_only',
      durabilityStatus: 'saved',
      participants: [],
      synced: true,
    }).kind).toBe('read_only')
    expect(buildEditorCollaborationStatusModel({
      access: 'edit',
      active: true,
      canEdit: false,
      connectionStatus: 'incompatible',
      durabilityStatus: 'idle',
      participants: [],
      synced: false,
    }).kind).toBe('update_required')
    // A rejected origin must not borrow the update label: the remedy is an
    // address, not a client version.
    expect(buildEditorCollaborationStatusModel({
      access: 'edit',
      active: true,
      canEdit: false,
      connectionStatus: 'origin_rejected',
      durabilityStatus: 'idle',
      participants: [],
      synced: false,
    }).kind).toBe('origin_rejected')
  })

  it('translates stable collaboration recovery notices without rewriting English', () => {
    const notice =
      'The collaboration lease could not be refreshed; reconnecting read-only.'
    expect(localizeEditorCollaborationNotice(notice, 'de')).toBe(
      'Die Verbindung zum Kollaborationsdienst ist unterbrochen. Inqtrix verbindet sich automatisch erneut.',
    )
    expect(localizeEditorCollaborationNotice(notice, 'en')).toBe(notice)
    expect(localizeEditorCollaborationNotice('diagnostic detail', 'de'))
      .toBe('diagnostic detail')
    expect(localizeEditorCollaborationNotice(
      'Collaboration access changed; revalidating read-only.',
      'de',
    )).toBe(
      'Der Kollaborationszugriff hat sich geändert. Inqtrix prüft die aktuelle Berechtigung erneut.',
    )
  })

  it('holds initiating-user publication focus until the patch arrives, then consumes it once', () => {
    const pending = beginCollaborationPublicationFocus('doc-1', 'patch-b')
    expect(consumeCollaborationPublicationFocus(pending, 'doc-1', [changes[0]!]))
      .toEqual({ focusId: null, pending })

    const arrived = consumeCollaborationPublicationFocus(pending, 'doc-1', changes)
    expect(arrived).toEqual({ focusId: 'patch-b', pending: null })
    expect(consumeCollaborationPublicationFocus(arrived.pending, 'doc-1', changes))
      .toEqual({ focusId: null, pending: null })
    expect(consumeCollaborationPublicationFocus(null, 'doc-1', changes))
      .toEqual({ focusId: null, pending: null })
  })

  it('keeps an A publication pending without switching UI while B is active', () => {
    const pendingByDocument = registerCollaborationPublicationFocus({}, 'doc-a', 'patch-a')
    const pending = pendingCollaborationPublicationFocusForDocument(
      pendingByDocument,
      'doc-a',
    )!

    expect(isCollaborationPublicationFocusCurrent(pending, 'doc-b')).toBe(false)
    expect(pendingCollaborationPublicationFocusForDocument(pendingByDocument, 'doc-b')).toBeNull()
    expect(consumeCollaborationPublicationFocus(pending, 'doc-b', [{
      ...changes[0]!,
      id: 'patch-a',
    }])).toEqual({ focusId: null, pending })
  })

  it('locks write mode to the granted permission', () => {
    expect(effectiveEditorWriteMode('edit', true, 'suggest')).toBe('suggest')
    expect(effectiveEditorWriteMode('edit', true, 'comment')).toBe('comment')
    expect(effectiveEditorWriteMode('suggest', true, 'edit')).toBe('suggest')
    expect(effectiveEditorWriteMode('suggest', true, 'comment')).toBe('comment')
    expect(effectiveEditorWriteMode('view', false, 'edit')).toBe('view')
    expect(effectiveEditorWriteMode('view', false, 'comment')).toBe('view')
  })

  it('keeps shared documents outside the owner hierarchy', () => {
    type DocumentFixture = {
      access?: { mode: 'owner' | 'shared' }
      folderId?: string
      id: string
    }
    const owned: DocumentFixture = { id: 'owned' }
    const shared: DocumentFixture = {
      access: { mode: 'shared' },
      folderId: 'owner-folder',
      id: 'shared',
    }

    expect(partitionEditorDocumentsByAccess([owned, shared])).toEqual({
      owned: [owned],
      shared: [shared],
    })
    expect(isOwnedEditorDocument(owned)).toBe(true)
    expect(isOwnedEditorDocument(shared)).toBe(false)
  })

  it('classifies a row by whether a details page exists for it', () => {
    // Private: no details page — the row must not offer a details action.
    expect(editorDocumentRowKind({ contentMode: 'markdown' })).toBe('owned-private')
    // Shared out by me: details (access, activity) exist.
    expect(editorDocumentRowKind({ contentMode: 'collaboration' })).toBe('owned-shared')
    // Shared with me: details exist and the owner is someone else.
    expect(editorDocumentRowKind({ access: { mode: 'shared' }, contentMode: 'collaboration' })).toBe('shared-with-me')
    expect(editorDocumentRowKind({ access: { mode: 'shared' } })).toBe('shared-with-me')
    // A local/legacy record carries neither fact and reads as private.
    expect(editorDocumentRowKind({})).toBe('owned-private')
  })

  it('derives the full rail grammar from the same ownership classification', () => {
    expect(editorDocumentRailCapabilities({ contentMode: 'markdown' })).toEqual({
      canDelete: true,
      canDrag: true,
      canOpenDetails: false,
      canPin: true,
      canRename: true,
      leadingRole: 'file',
    })
    expect(editorDocumentRailCapabilities({ contentMode: 'collaboration' }))
      .toMatchObject({
        canDelete: true,
        canOpenDetails: true,
        canPin: true,
        leadingRole: 'people',
      })
    expect(editorDocumentRailCapabilities({ access: { mode: 'shared' } }))
      .toEqual({
        canDelete: false,
        canDrag: false,
        canOpenDetails: true,
        canPin: false,
        canRename: false,
        leadingRole: 'people',
      })
  })
})

describe('editor overlay policy', () => {
  it('uses final display outside Changes and restores the chosen display on return', () => {
    const base = {
      collaboration: true,
      changesView: 'open' as const,
      display: 'original' as const,
      documentId: 'doc-1',
      selectedChangeId: 'patch-a',
      visibleChanges: changes,
      writeAuthorId: 'ada',
      writeMode: 'suggest' as const,
    }

    expect(collaborationEditorPolicyUpdate({ ...base, inspectorTab: 'assistant' }))
      .toMatchObject({
        display: 'final',
        selectedSuggestionIds: [],
        visibleSuggestionIds: undefined,
      })
    expect(collaborationEditorPolicyUpdate({ ...base, inspectorTab: 'changes' }))
      .toMatchObject({
        display: 'original',
        selectedSuggestionIds: ['suggestion-a'],
        visibleSuggestionIds: ['suggestion-a', 'suggestion-b'],
      })
  })

  it('activates every suggestion in a selected compound patch', () => {
    const compound = {
      ...changes[0]!,
      suggestionIds: ['delete-a', 'insert-a', 'modify-a'],
    }

    expect(collaborationEditorPolicyUpdate({
      changesView: 'open',
      collaboration: true,
      display: 'simple',
      documentId: 'doc-1',
      inspectorTab: 'changes',
      selectedChangeId: compound.id,
      visibleChanges: [compound],
      writeAuthorId: 'ada',
      writeMode: 'suggest',
    })).toMatchObject({
      selectedSuggestionIds: ['delete-a', 'insert-a', 'modify-a'],
      visibleSuggestionIds: ['delete-a', 'insert-a', 'modify-a'],
    })
  })
})

describe('startupPresentation', () => {
  it('presents both startup transients as one calm syncing state', () => {
    expect(startupPresentation('inactive', 100, true)).toEqual({ calm: true, kind: 'syncing' })
    expect(startupPresentation('syncing', 100, true)).toEqual({ calm: true, kind: 'syncing' })
  })

  it('lets a persisting transient earn its real color after the grace window', () => {
    expect(startupPresentation('syncing', 1_500, true)).toEqual({ calm: false, kind: 'syncing' })
    expect(startupPresentation('inactive', 1_500, true)).toEqual({ calm: false, kind: 'inactive' })
  })

  it('never calms an exceptional state, not even at t=0', () => {
    // The grace exists for expected transients only. An error, a revocation
    // or a forced update must be loud immediately — calming those would be
    // the silent fallback the project forbids.
    for (const kind of ['error', 'access_revoked', 'update_required', 'origin_rejected', 'read_only'] as const) {
      expect(startupPresentation(kind, 0, true)).toEqual({ calm: false, kind })
    }
  })

  it('leaves the settled states untouched', () => {
    expect(startupPresentation('saved', 0, true)).toEqual({ calm: false, kind: 'saved' })
    expect(startupPresentation('saving', 0, true)).toEqual({ calm: false, kind: 'saving' })
  })

  it('never dresses a local document up as syncing', () => {
    // A pure markdown document is FINAL `inactive` ("Lokal"): no session is
    // coming, so the grace must not show 1.2s of "syncing" and then swap the
    // label — that was itself a startup flash (found in the header review).
    expect(startupPresentation('inactive', 0, false)).toEqual({ calm: false, kind: 'inactive' })
    expect(startupPresentation('inactive', 100, false)).toEqual({ calm: false, kind: 'inactive' })
  })
})
