import {
  createEditorSchemaExtensions,
  parseEditorMarkdown,
  type EditorRelativePositionAdapter,
} from '@inqtrix/editor-schema'
import { Editor as HeadlessEditor } from '@tiptap/core'
import type { Editor } from '@tiptap/react'
import { describe, expect, it, vi } from 'vitest'

import type {
  EditorCommentAnchorRecord,
  EditorDocumentRecord,
  EditorSuggestionRecord,
} from '@/features/project/types'
import {
  beginEditorCollaborationAuthorityGuard,
  buildCollaborationSuggestionTargetMarkdown,
  collaborationPublicationFromResponse,
  editorAiReadOnlyReason,
  invokeEditorAiProvider,
  invokePrivateSuggestionPublication,
  prepareCollaborationSuggestionBatch,
  privateSuggestionPublishDisabledReason,
  resolvePrivateSuggestionAnchor,
  resolveEditorAiDocumentContext,
  serializePrivateSuggestionAnchor,
  sortPrivateSuggestionGroup,
} from './useEditorSuggestions'
import type { CollaborationLiveAuthority } from './collaborationAuthority'

function anchor(from: number, to: number): EditorCommentAnchorRecord {
  return {
    from,
    quoteAfter: 'after',
    quoteBefore: 'before',
    selectedText: `range-${from}`,
    to,
  }
}

const serializer: EditorRelativePositionAdapter = {
  fromProseMirrorPosition: (position) => `relative-${position}`,
  toProseMirrorPosition: () => null,
}

describe('private AI suggestion relative anchors', () => {
  it('preserves quote fallback data while resolving live Yjs positions', () => {
    const original = anchor(4, 12)
    const serialized = serializePrivateSuggestionAnchor(original, serializer)
    const resolved = resolvePrivateSuggestionAnchor(serialized.anchor, {
      fromProseMirrorPosition: () => '',
      toProseMirrorPosition: (position) => ({
        'relative-4': 18,
        'relative-12': 26,
      })[position] ?? null,
    })

    expect(serialized.status).toBe('relative')
    expect(resolved).toMatchObject({
      anchor: {
        from: 18,
        quoteAfter: 'after',
        quoteBefore: 'before',
        selectedText: 'range-4',
        to: 26,
      },
      status: 'relative',
    })
  })

  it('falls back to legacy absolute hints when a relative range is invalid', () => {
    const original = anchor(7, 14)
    const serialized = serializePrivateSuggestionAnchor(original, serializer)
    const unresolved = resolvePrivateSuggestionAnchor(serialized.anchor, {
      fromProseMirrorPosition: () => '',
      toProseMirrorPosition: () => null,
    })

    expect(unresolved).toEqual({
      anchor: serialized.anchor,
      reason: 'relative_unresolved',
      status: 'degraded',
    })
    expect(resolvePrivateSuggestionAnchor(original, serializer)).toEqual({
      anchor: original,
      status: 'legacy',
    })
  })

  it('fails required collaboration encoding instead of silently storing an absolute anchor', () => {
    expect(serializePrivateSuggestionAnchor(anchor(2, 6), null, true)).toEqual({
      anchor: anchor(2, 6),
      reason: 'adapter_unavailable',
      status: 'failed',
    })
    expect(serializePrivateSuggestionAnchor(anchor(5, 5), serializer, true)).toEqual({
      anchor: anchor(5, 5),
      reason: 'relative_missing',
      status: 'failed',
    })
  })

  it('orders compound groups by resolved positions instead of stale absolutes', () => {
    const firstStored = serializePrivateSuggestionAnchor(anchor(3, 8), serializer).anchor
    const secondStored = serializePrivateSuggestionAnchor(anchor(20, 25), serializer).anchor
    const livePositions: Record<string, number> = {
      'relative-3': 30,
      'relative-8': 35,
      'relative-20': 5,
      'relative-25': 10,
    }
    const adapter: EditorRelativePositionAdapter = {
      fromProseMirrorPosition: () => '',
      toProseMirrorPosition: (position) => livePositions[position] ?? null,
    }

    const sorted = sortPrivateSuggestionGroup([
      { anchor: firstStored, createdAt: '2026-01-01T00:00:00Z', id: 'first' },
      { anchor: secondStored, createdAt: '2026-01-01T00:00:01Z', id: 'second' },
    ], adapter)

    expect(sorted.map((item) => item.id)).toEqual(['second', 'first'])
  })
})

const BASE_DOCUMENT: EditorDocumentRecord = {
  contentMarkdown: '# Cached body',
  createdAt: '2026-07-15T08:00:00.000Z',
  folderId: null,
  id: 'document-1',
  revision: 1,
  source: 'blank',
  title: 'Draft',
  updatedAt: '2026-07-15T08:00:00.000Z',
}

function writableAuthority(
  overrides: Partial<CollaborationLiveAuthority> = {},
): CollaborationLiveAuthority {
  return {
    access: 'suggest',
    blockingFailure: null,
    canEdit: true,
    connectionStatus: 'connected',
    documentId: BASE_DOCUMENT.id,
    generation: 2,
    lifecycleStatus: 'saved',
    revision: 0,
    synced: true,
    ...overrides,
  }
}

describe('editor AI collaboration projection barrier', () => {
  it('keeps the legacy content path and does not call projection flush', async () => {
    const flush = vi.fn()

    await expect(resolveEditorAiDocumentContext(
      BASE_DOCUMENT,
      null,
      null,
      {},
      'en',
      flush,
    )).resolves.toEqual({ markdown: '# Cached body', sequence: null })
    expect(flush).not.toHaveBeenCalled()
  })

  it('returns the flushed markdown and sequence only when it matches the live final projection', async () => {
    const liveMarkdown = '# Durable body\n\nCurrent text.'
    const editor = {
      getJSON: () => parseEditorMarkdown(liveMarkdown),
    } as Pick<Editor, 'getJSON'>
    const flush = vi.fn(async () => ({
      confirmedAt: '2026-07-15T10:00:00.000Z',
      markdown: liveMarkdown,
      sequence: 17,
    }))

    await expect(resolveEditorAiDocumentContext(
      { ...BASE_DOCUMENT, contentMode: 'collaboration' },
      editor,
      {
        flushAndAwaitDurable: async () => 17,
        readAuthority: () => writableAuthority(),
        setAuthoritativeSequence: () => undefined,
      },
      {},
      'en',
      flush,
    )).resolves.toEqual({ markdown: liveMarkdown, sequence: 17 })
  })

  it('rejects a stale flushed projection without falling back to cached document markdown', async () => {
    const editor = {
      getJSON: () => parseEditorMarkdown('# Live body'),
    } as Pick<Editor, 'getJSON'>

    await expect(resolveEditorAiDocumentContext(
      { ...BASE_DOCUMENT, contentMode: 'collaboration' },
      editor,
      {
        flushAndAwaitDurable: async () => 16,
        readAuthority: () => writableAuthority(),
        setAuthoritativeSequence: () => undefined,
      },
      {},
      'en',
      async () => ({
        confirmedAt: '2026-07-15T10:00:00.000Z',
        markdown: '# Older projection',
        sequence: 16,
      }),
    )).rejects.toThrow('not durable yet')
  })
})

describe('editor AI collaboration access', () => {
  it('does not invoke provider work for view-only collaboration access', async () => {
    const provider = vi.fn(async () => 'provider-result')
    const document: EditorDocumentRecord = {
      ...BASE_DOCUMENT,
      access: { mode: 'shared', permission: 'view' },
      contentMode: 'collaboration',
    }

    expect(editorAiReadOnlyReason(document, null, 'en'))
      .toBe('AI editing is unavailable with view-only access.')
    await expect(invokeEditorAiProvider(document, 'view', 'en', provider))
      .rejects.toThrow('view-only access')
    expect(provider).not.toHaveBeenCalled()
  })

  it('keeps legacy AI behavior unchanged even when legacy share metadata is view-only', async () => {
    const provider = vi.fn(async () => 'provider-result')
    const document: EditorDocumentRecord = {
      ...BASE_DOCUMENT,
      access: { mode: 'shared', permission: 'view' },
    }

    expect(editorAiReadOnlyReason(document, 'view', 'en')).toBeNull()
    await expect(invokeEditorAiProvider(document, 'view', 'en', provider))
      .resolves.toBe('provider-result')
    expect(provider).toHaveBeenCalledOnce()
  })

  it('blocks a cached suggest user after a live view downgrade before context or publish work', async () => {
    const resolveDocumentContext = vi.fn(async () => ({ markdown: '# Current', sequence: 8 }))
    const publish = vi.fn(async (context: { markdown: string; sequence: number | null }) => {
      void context
      return 'published'
    })
    const document: EditorDocumentRecord = {
      ...BASE_DOCUMENT,
      access: { mode: 'shared', permission: 'suggest' },
      collaboration: {
        generation: 2,
        persistedSequence: 8,
        projectionSequence: 8,
        schemaVersion: 1,
      },
      contentMode: 'collaboration',
    }
    const liveViewHandle = {
      access: 'view',
      canEdit: false,
      documentId: document.id,
      generation: 2,
      readAuthority: () => writableAuthority({
        access: 'view',
        canEdit: false,
        connectionStatus: 'read_only',
        lifecycleStatus: 'read_only',
      }),
    } as never

    expect(privateSuggestionPublishDisabledReason(document, liveViewHandle, 'en'))
      .toBe('This collaboration access is read-only.')
    await expect(invokePrivateSuggestionPublication(
      document,
      liveViewHandle,
      'en',
      async () => {
        const context = await resolveDocumentContext()
        return publish(context)
      },
    )).rejects.toThrow('read-only')
    expect(resolveDocumentContext).not.toHaveBeenCalled()
    expect(publish).not.toHaveBeenCalled()
  })

  it('does not invoke provider work after authority downgrades during the projection barrier', async () => {
    let authority = writableAuthority()
    const document: EditorDocumentRecord = {
      ...BASE_DOCUMENT,
      collaboration: {
        generation: 2,
        persistedSequence: 8,
        projectionSequence: 8,
        schemaVersion: 1,
      },
      contentMode: 'collaboration',
    }
    const collaboration = {
      readAuthority: () => authority,
    } as never
    const guard = beginEditorCollaborationAuthorityGuard(
      document,
      collaboration,
      'write',
      'en',
    )
    const provider = vi.fn(async () => 'provider-result')
    const editor = {
      getJSON: () => parseEditorMarkdown('# Durable'),
    } as Pick<Editor, 'getJSON'>

    await expect((async () => {
      const context = await resolveEditorAiDocumentContext(
        document,
        editor,
        {
          flushAndAwaitDurable: async () => 8,
          readAuthority: () => authority,
          setAuthoritativeSequence: () => undefined,
        },
        {},
        'en',
        async () => {
          authority = writableAuthority({
            access: 'view',
            canEdit: false,
            connectionStatus: 'read_only',
            lifecycleStatus: 'read_only',
            revision: 1,
          })
          return {
            confirmedAt: '2026-07-15T10:00:00.000Z',
            markdown: '# Durable',
            sequence: 8,
          }
        },
        guard,
      )
      return invokeEditorAiProvider(document, 'suggest', 'en', provider, guard)
        .then(() => context)
    })()).rejects.toThrow('read-only')
    expect(provider).not.toHaveBeenCalled()
  })

  it('does not continue after access is revoked while the provider is completing', async () => {
    let authority = writableAuthority()
    const document: EditorDocumentRecord = {
      ...BASE_DOCUMENT,
      collaboration: {
        generation: 2,
        persistedSequence: 8,
        projectionSequence: 8,
        schemaVersion: 1,
      },
      contentMode: 'collaboration',
    }
    const collaboration = { readAuthority: () => authority } as never
    const guard = beginEditorCollaborationAuthorityGuard(
      document,
      collaboration,
      'write',
      'en',
    )
    const provider = vi.fn(async () => {
      authority = writableAuthority({
        access: 'view',
        canEdit: false,
        connectionStatus: 'access_revoked',
        lifecycleStatus: 'error',
        revision: 1,
        synced: false,
      })
      return 'provider-result'
    })
    const subsequentSideEffect = vi.fn()

    await expect(
      invokeEditorAiProvider(document, 'suggest', 'en', provider, guard)
        .then(subsequentSideEffect),
    ).rejects.toThrow('revoked')
    expect(provider).toHaveBeenCalledOnce()
    expect(subsequentSideEffect).not.toHaveBeenCalled()
  })
})

describe('collaboration private suggestion publication target', () => {
  it('projects the private edit without mutating the live collaboration editor', () => {
    const suggestion: EditorSuggestionRecord = {
      anchor: {
        from: 1,
        quoteAfter: '',
        quoteBefore: '',
        selectedMarkdown: 'Old text',
        selectedText: 'Old text',
        to: 9,
      },
      blockId: 'paragraph-1',
      createdAt: BASE_DOCUMENT.createdAt,
      documentId: BASE_DOCUMENT.id,
      groupId: 'group-1',
      id: 'suggestion-1',
      originalMarkdown: 'Old text',
      originalText: 'Old text',
      origin: { kind: 'global_run' },
      proposedText: 'New text',
      status: 'pending',
      updatedAt: BASE_DOCUMENT.updatedAt,
    }

    expect(buildCollaborationSuggestionTargetMarkdown('Old text', [suggestion]).trim()).toBe(
      'New text',
    )

    expect(buildCollaborationSuggestionTargetMarkdown(
      'Alpha old omega',
      [{
        ...suggestion,
        anchor: {
          from: 7,
          quoteAfter: ' omega',
          quoteBefore: 'Alpha ',
          selectedMarkdown: 'old',
          selectedText: 'old',
          to: 10,
        },
        originalMarkdown: 'old',
        originalText: 'old',
        proposedText: 'new',
      }],
    ).trim()).toBe('Alpha new omega')
  })

  it('rebases duplicate quote targets from their live relative positions after the barrier', () => {
    const editor = new HeadlessEditor({
      content: parseEditorMarkdown('old between old'),
      element: null,
      extensions: createEditorSchemaExtensions({ enableUndoRedo: false }),
      injectCSS: false,
    })
    try {
      const storedAnchor = serializePrivateSuggestionAnchor(anchor(1, 4), serializer).anchor
      const suggestion: EditorSuggestionRecord = {
        anchor: {
          ...storedAnchor,
          quoteAfter: '',
          quoteBefore: '',
          selectedMarkdown: 'old',
          selectedText: 'old',
        },
        blockId: 'paragraph-1',
        createdAt: BASE_DOCUMENT.createdAt,
        documentId: BASE_DOCUMENT.id,
        groupId: 'group-1',
        id: 'suggestion-1',
        originalMarkdown: 'old',
        originalText: 'old',
        origin: { kind: 'global_run' },
        proposedText: 'new',
        status: 'pending',
        updatedAt: BASE_DOCUMENT.updatedAt,
      }
      const prepared = prepareCollaborationSuggestionBatch(
        editor as unknown as Editor,
        [suggestion],
        {
          fromProseMirrorPosition: () => '',
          toProseMirrorPosition: (position) => ({
            'relative-1': 13,
            'relative-4': 16,
          })[position] ?? null,
        },
        'en',
      )

      expect(prepared.errors).toEqual({})
      expect(prepared.suggestions[0]?.anchor).toMatchObject({ from: 13, to: 16 })
      expect(buildCollaborationSuggestionTargetMarkdown(
        'old between old',
        prepared.suggestions,
      ).trim()).toBe('old between new')
    } finally {
      editor.destroy()
    }
  })

  it('links only a durable matching command result to the private suggestion', () => {
    expect(collaborationPublicationFromResponse(
      {
        command_id: 'command-1',
        patch_id: 'patch-1',
        sequence: 18,
        suggestion_ids: ['shared-suggestion-1'],
      },
      { commandId: 'command-1', expectedSequence: 17, patchId: 'patch-1' },
      'en',
    )).toEqual({
      commandId: 'command-1',
      patchId: 'patch-1',
      sequence: 18,
      suggestionIds: ['shared-suggestion-1'],
    })

    expect(() => collaborationPublicationFromResponse(
      {
        command_id: 'command-1',
        patch_id: 'patch-1',
        sequence: 17,
        suggestion_ids: ['shared-suggestion-1'],
      },
      { commandId: 'command-1', expectedSequence: 17, patchId: 'patch-1' },
      'en',
    )).toThrow('not confirmed durably')
  })
})
