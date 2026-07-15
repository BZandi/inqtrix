import type { SuggestionDescriptor } from '@inqtrix/editor-schema'
import type { HocuspocusProvider } from '@hocuspocus/provider'
import { describe, expect, it, vi } from 'vitest'

import type { EditorCollaborationActivity } from '@/api/inqtrixClient'
import {
  durableSuggestionAuthorsFromActivity,
  INSPECTOR_ATTRIBUTION_MAX_PAGES,
  inspectorAttributionWarning,
  loadInspectorCollaborationActivity,
  normalizeActivity,
  readInspectorParticipants,
} from './collaborationAdapter'
import { buildInspectorChanges } from './model'
import {
  beginCollaborationAuthorityGuard,
  type CollaborationLiveAuthority,
} from '../collaborationAuthority'

const activity: EditorCollaborationActivity[] = [
  {
    actor: { id: 'reviewer', name: 'Grace Reviewer' },
    actor_kind: 'human',
    command_id: 'decision-command',
    created_at: 12,
    from_sequence: 4,
    suggestion_ids: ['suggestion-offline'],
    to_sequence: 5,
    type: 'decision',
  },
  {
    actor: { id: 'offline-author', name: 'Ada Offline' },
    actor_kind: 'human',
    command_id: 'suggestion-command',
    created_at: 10,
    from_sequence: 2,
    suggestion_ids: ['suggestion-offline'],
    to_sequence: 3,
    type: 'suggestion',
  },
]

function readableAuthority(
  overrides: Partial<CollaborationLiveAuthority> = {},
): CollaborationLiveAuthority {
  return {
    access: 'edit',
    blockingFailure: null,
    canEdit: true,
    connectionStatus: 'connected',
    documentId: 'document-1',
    generation: 2,
    lifecycleStatus: 'saved',
    revision: 4,
    synced: true,
    ...overrides,
  }
}

describe('durable collaboration attribution', () => {
  it('keeps canonical attribution when the remote author leaves before activity refresh', () => {
    const descriptors: SuggestionDescriptor[] = [{
      authorId: 'offline-author',
      createdAt: 10,
      kind: 'insertion',
      patchId: 'patch-offline',
      suggestionId: 'suggestion-offline',
    }]
    const excerpts = new Map([[
      'suggestion-offline',
      { deletionText: '', insertionText: 'Durable text', modificationText: '', position: 2 },
    ]])
    const awarenessStates = new Map<number, unknown>([[1, {
      user: { color: '#2563EB', id: 'offline-author', name: 'Ada Live' },
    }]])
    const provider = {
      awareness: { getStates: () => awarenessStates },
    } as unknown as HocuspocusProvider
    expect(readInspectorParticipants(provider, null).map((participant) => participant.name))
      .toEqual(['Ada Live'])

    awarenessStates.clear()
    const participantsAfterLeave = readInspectorParticipants(provider, null)
    const authors = durableSuggestionAuthorsFromActivity(activity)

    expect(participantsAfterLeave).toEqual([])
    expect(authors.get('suggestion-offline')?.name).toBe('Ada Offline')
    expect(buildInspectorChanges(descriptors, excerpts, participantsAfterLeave, authors)[0]?.author)
      .toMatchObject({ id: 'offline-author', name: 'Ada Offline' })
  })

  it('uses canonical activity actor names when every collaborator is offline', () => {
    expect(normalizeActivity(activity).map((entry) => entry.actor.name))
      .toEqual(['Grace Reviewer', 'Ada Offline'])
  })

  it('pages beyond the newest 100 rows to attribute an older still-open suggestion', async () => {
    const recentActivity: EditorCollaborationActivity[] = Array.from(
      { length: 100 },
      (_, index) => ({
        actor: { id: `reviewer-${index}`, name: `Reviewer ${index}` },
        actor_kind: 'human' as const,
        command_id: `command-${index}`,
        created_at: 200 - index,
        from_sequence: 200 - index,
        suggestion_ids: [`closed-${index}`],
        to_sequence: 201 - index,
        type: 'decision' as const,
      }),
    )
    const olderSuggestionEvent = activity[1]!
    const fetchActivity = vi.fn(async (_documentId: string, options: { cursor?: string }) => (
      options.cursor
        ? { data: [olderSuggestionEvent], next_cursor: null, object: 'list' as const }
        : { data: recentActivity, next_cursor: 'older-page', object: 'list' as const }
    )) as never

    const result = await loadInspectorCollaborationActivity({
      documentId: 'document-1',
      fetchActivity,
      openSuggestionIds: ['suggestion-offline'],
      workspaceId: 'workspace-1',
    })

    expect(result.data).toHaveLength(100)
    expect(result.attributionComplete).toBe(true)
    expect(result.attributionData).toEqual([olderSuggestionEvent])
    expect(durableSuggestionAuthorsFromActivity([
      ...result.data,
      ...result.attributionData,
    ]).get('suggestion-offline')?.name).toBe('Ada Offline')
    expect(fetchActivity).toHaveBeenCalledTimes(2)
    expect(fetchActivity).toHaveBeenLastCalledWith(
      'document-1',
      expect.objectContaining({ cursor: 'older-page', limit: 100 }),
    )
  })

  it('stops after page two and does not publish a result when authority is revoked during that await', async () => {
    let authority = readableAuthority()
    const authoritySource = { readAuthority: () => authority }
    const authorityGuard = beginCollaborationAuthorityGuard(
      authoritySource,
      { documentId: 'document-1', generation: 2 },
      'read',
    )
    const fetchActivity = vi.fn(async (_documentId: string, options: { cursor?: string }) => {
      if (!options.cursor) {
        return { data: [], next_cursor: 'page-2', object: 'list' as const }
      }
      authority = readableAuthority({
        access: null,
        canEdit: false,
        connectionStatus: 'access_revoked',
        lifecycleStatus: 'error',
        revision: 5,
        synced: false,
      })
      return { data: [], next_cursor: 'page-3', object: 'list' as const }
    }) as never
    const stateUpdate = vi.fn()

    await expect(loadInspectorCollaborationActivity({
      authorityGuard,
      documentId: 'document-1',
      fetchActivity,
      openSuggestionIds: ['suggestion-offline'],
      workspaceId: 'workspace-1',
    }).then(stateUpdate)).rejects.toThrow('revoked')

    expect(fetchActivity).toHaveBeenCalledTimes(2)
    expect(stateUpdate).not.toHaveBeenCalled()
  })

  it('bounds endless unique cursors and reports incomplete attribution without exposing an id', async () => {
    let page = 0
    const fetchActivity = vi.fn(async () => {
      page += 1
      return { data: [], next_cursor: `cursor-${page}`, object: 'list' as const }
    }) as never

    const result = await loadInspectorCollaborationActivity({
      documentId: 'document-1',
      fetchActivity,
      openSuggestionIds: ['suggestion-secret-user-id'],
      workspaceId: 'workspace-1',
    })
    const warning = inspectorAttributionWarning(result, 'en')

    expect(fetchActivity).toHaveBeenCalledTimes(INSPECTOR_ATTRIBUTION_MAX_PAGES)
    expect(result).toMatchObject({
      attributionComplete: false,
      lookupLimited: true,
      unresolvedSuggestionCount: 1,
    })
    expect(warning).toContain('bounded 500-row activity lookup')
    expect(warning).not.toContain('suggestion-secret-user-id')
  })

  it('never exposes a raw actor id as the visible fallback name', () => {
    const unnamed: EditorCollaborationActivity = {
      ...activity[1]!,
      actor: { id: 'user-opaque-7f3a', name: '   ' },
    }

    expect(normalizeActivity([unnamed])[0]?.actor).toMatchObject({
      id: 'user-opaque-7f3a',
      name: 'Collaborator',
    })
  })
})
