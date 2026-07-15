import type { HocuspocusProvider } from '@hocuspocus/provider'
import type { Extensions } from '@tiptap/core'
import { describe, expect, it, vi } from 'vitest'
import * as Y from 'yjs'

vi.mock('../tiptap', () => ({
  collaborationCaretOptions: {
    render: vi.fn(),
    selectionRender: vi.fn(),
  },
  commentDecorationPluginKey: {},
  createEditorExtensions: () => [],
  normalizeEditorMarkdownForTiptap: (markdown: string) => markdown,
  serializeEditorFinalProjectionMarkdown: () => '',
  serializeEditorMarkdown: () => '',
  suggestionDecorationPluginKey: {},
}))

import {
  collaborationBindingForEditorDocument,
  configureEditorExtensionsForCollaboration,
  editorSurfaceInitialContent,
  editorSurfaceLifecyclePolicy,
  isCollaborationSurfaceSynced,
} from './MarkdownEditorSurface'

describe('MarkdownEditorSurface lifecycle policy', () => {
  it('preserves the complete legacy markdown lifecycle', () => {
    expect(editorSurfaceLifecyclePolicy({
      collaborationCanEdit: false,
      collaborationMode: false,
      collaborationReady: false,
      mode: 'source',
    })).toEqual({
      applyExternalContent: true,
      editable: false,
      emitFullBodyChanges: true,
      renderSourceEditor: true,
    })
  })

  it('never applies or emits full markdown bodies after collaboration join', () => {
    expect(editorSurfaceLifecyclePolicy({
      collaborationCanEdit: true,
      collaborationMode: true,
      collaborationReady: true,
      mode: 'live',
    })).toEqual({
      applyExternalContent: false,
      editable: true,
      emitFullBodyChanges: false,
      renderSourceEditor: false,
    })
  })

  it('keeps collaboration read-only until lifecycle sync is complete', () => {
    expect(editorSurfaceLifecyclePolicy({
      collaborationCanEdit: true,
      collaborationMode: true,
      collaborationReady: false,
      mode: 'live',
    }).editable).toBe(false)
  })

  it('does not treat provider presence as lifecycle readiness', () => {
    expect(isCollaborationSurfaceSynced({
      provider: {} as HocuspocusProvider,
      synced: false,
    } as never)).toBe(false)
    expect(isCollaborationSurfaceSynced({
      provider: {} as HocuspocusProvider,
      synced: true,
    } as never)).toBe(true)
  })

  it('does not bind document A collaboration under document B during a switch', () => {
    const yDocument = new Y.Doc()
    const documentB = {
      collaboration: {
        generation: 3,
        persistedSequence: 0,
        projectionSequence: 0,
        schemaVersion: 1,
      },
      contentMarkdown: '# B',
      contentMode: 'collaboration',
      createdAt: '2026-07-15T10:00:00.000Z',
      folderId: null,
      id: 'document-b',
      revision: 1,
      source: 'blank',
      title: 'B',
      updatedAt: '2026-07-15T10:00:00.000Z',
    } as const

    expect(collaborationBindingForEditorDocument(documentB, {
      canEdit: true,
      document: yDocument,
      documentId: 'document-a',
      generation: 2,
      lifecycleKey: 'document-a:g2',
      provider: {} as HocuspocusProvider,
      synced: true,
      user: { color: '#2563EB', id: 'user-1', name: 'Ada' },
    } as never)).toBeNull()
    yDocument.destroy()
  })

  it('shows the last projected markdown read-only while collaboration is unavailable', () => {
    const policy = editorSurfaceLifecyclePolicy({
      collaborationCanEdit: false,
      collaborationMode: true,
      collaborationReady: false,
      mode: 'live',
    })

    expect(editorSurfaceInitialContent({
      collaborationMode: true,
      collaborationReady: false,
      contentMarkdown: '# Last durable projection',
    })).toEqual({
      content: '# Last durable projection',
      contentType: 'markdown',
    })
    expect(policy).toMatchObject({
      applyExternalContent: true,
      editable: false,
      emitFullBodyChanges: false,
    })
  })

  it('leaves content initialization to Yjs once collaboration is ready', () => {
    expect(editorSurfaceInitialContent({
      collaborationMode: true,
      collaborationReady: true,
      contentMarkdown: '# Projection must not enter Yjs',
    })).toEqual({})
  })
})

describe('MarkdownEditorSurface collaboration extensions', () => {
  it('leaves legacy extensions untouched', () => {
    const configure = vi.fn()
    const extensions = [{ configure, name: 'starterKit' }] as unknown as Extensions

    expect(configureEditorExtensionsForCollaboration(extensions, null)).toBe(extensions)
    expect(configure).not.toHaveBeenCalled()
  })

  it('disables normal undo while a collaboration provider is still joining', () => {
    const configuredStarterKit = { name: 'starterKit' }
    const configure = vi.fn(() => configuredStarterKit)
    const extensions = [{ configure, name: 'starterKit' }] as unknown as Extensions

    const configured = configureEditorExtensionsForCollaboration(
      extensions,
      null,
      true,
    )

    expect(configure).toHaveBeenCalledWith({ undoRedo: false })
    expect(configured).toEqual([configuredStarterKit])
  })

  it('disables normal undo only when Yjs collaboration extensions are attached', () => {
    const configuredStarterKit = { name: 'starterKit' }
    const configure = vi.fn(() => configuredStarterKit)
    const extensions = [
      { configure, name: 'starterKit' },
      { name: 'paragraph' },
    ] as unknown as Extensions
    const document = new Y.Doc()

    const configured = configureEditorExtensionsForCollaboration(extensions, {
      canEdit: true,
      document,
      lifecycleKey: 'document-1:g1',
      provider: {} as HocuspocusProvider,
      synced: true,
      user: { color: '#2563EB', id: 'user-1', name: 'Ada' },
    })

    expect(configure).toHaveBeenCalledWith({ undoRedo: false })
    expect(configured.map((extension) => extension.name)).toEqual([
      'starterKit',
      'paragraph',
      'collaboration',
      'collaborationCaret',
    ])
    expect(configuredStarterKit).toBe(configured[0])
    document.destroy()
  })
})
