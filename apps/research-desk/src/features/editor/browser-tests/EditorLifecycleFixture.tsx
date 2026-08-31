import '@fontsource-variable/inter/index.css'
import 'katex/dist/katex.min.css'
import '@/styles/globals.css'

import { useCallback, useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import Collaboration from '@tiptap/extension-collaboration'
import { Extension, getSchema } from '@tiptap/core'
import { EditorContent, useEditor } from '@tiptap/react'
import { Plugin } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'
import { prosemirrorJSONToYDoc } from '@tiptap/y-tiptap'
import {
  EDITOR_SCHEMA_VERSION,
  EDITOR_YJS_FRAGMENT,
} from '@inqtrix/editor-schema'
import * as Y from 'yjs'

import { AppProviders } from '@/app/AppProviders'
import type {
  EditorCollaborationCommentThread,
} from '@/api/inqtrixClient'
import { Info } from '@/components/icons'
import { ExplorerHistoryRow } from '@/components/ui/explorer-list'
import { useModalFocusTrap } from '@/components/ui/use-modal-focus-trap'
import { createEmptyProjectState } from '@/features/project/seedProject'
import type { EditorDocumentRecord, EditorSuggestionRecord, ProjectState } from '@/features/project/types'
import { ProfileAvatar } from '@/features/researchDesk/components/ProfileAvatar'
import { researchDeskReducer } from '@/features/researchDesk/state'
import { ShareDialog } from '@/features/sharing/ShareDialog'
import type { MentionComposerHandle } from '@/features/composer/MentionComposer'
import { EditorAssistantComposer, EditorTopBar } from '../EditorWorkspace'
import { DocumentDiffView } from '../DocumentDiffView'
import { EditorDocumentChangesSection } from '../EditorDocumentChangesSection'
import { editorCopy } from '../editorCopy'
import {
  collaborationProjectionController,
} from '../collaborationProjection'
import {
  collaborationBindingForEditorDocument,
  MarkdownEditorSurface,
} from '../core/MarkdownEditorSurface'
import {
  editorForSurfaceIdentity,
  updateEditorSurfaceRegistration,
  type EditorSurfaceIdentity,
  type EditorSurfaceRegistration,
} from '../core/editorSurfaceRegistration'
import { EditorCollaborationStatus, EditorInspector } from '../inspector/EditorInspector'
import { TeamCommentsPanel } from '../inspector/TeamCommentsPanel'
import { EditorTopBarLayout } from '../inspector/EditorTopBarLayout'
import {
  buildEditorCollaborationStatusModel,
  type EditorWriteMode,
  type InspectorChange,
} from '../inspector/model'
import {
  applyCollaborationEditorPolicy,
  collaborationEditorPolicyUpdate,
  type CollaborationEditorPolicyInput,
} from '../inspector/editorPolicy'
import {
  createEditorExtensions,
  renderCollaborationCaret,
  serializeEditorFinalProjectionMarkdown,
} from '../tiptap'
import {
  containsSuggestionBoundary,
  containsStructureSuggestionAttribute,
  useCollaborationDocument,
} from '../useCollaborationDocument'
import type {
  CollaborationCommentsHandle,
} from '../useCollaborationComments'
import { planEditorDocumentAutosave } from '../useEditorHistoryApi'
import {
  editorAiReadOnlyReason,
  privateSuggestionPublishDisabledReason,
} from '../useEditorSuggestions'

type RenderSnapshot = {
  controllerRegistered: boolean
  handleDocumentId: string | null
  lifecycleStatus: string
  requestedDocumentId: string
}

declare global {
  interface Window {
    __collaborationRenderSnapshots?: RenderSnapshot[]
    __commentScaleFirstInteractiveMs?: number
    __commentScaleSelectionMs?: number
    __commentScaleSelectionStartedAt?: number
    __slashCollaborationUpdate?: number[]
    __slashCollaborationUpdates?: number[][]
    __slashCollaborationServerUpdate?: number[]
    __slashSuggestionBoundaryDetections?: boolean[]
    __slashStructureUpdateDetections?: boolean[]
  }
}

const noOp = () => undefined
const noOpAsync = async () => undefined
const textImprovement = { enabled: false, workspaceId: 'browser-fixture' }
const fixtureModuleStartedAt = performance.now()

function collaborationDocument(id: string, generation: number): EditorDocumentRecord {
  return {
    access: { mode: 'owner', permission: 'edit' },
    collaboration: {
      generation,
      persistedSequence: 0,
      projectionSequence: 0,
      schemaVersion: EDITOR_SCHEMA_VERSION,
    },
    contentMarkdown: `# Projection ${id.toUpperCase()}\n\nFallback body for ${id.toUpperCase()}.`,
    contentMode: 'collaboration',
    createdAt: '2026-07-15T08:00:00.000Z',
    folderId: null,
    id,
    metadataRevision: 2,
    revision: 1,
    source: 'blank',
    title: `${id.toUpperCase()}.md`,
    updatedAt: '2026-07-15T08:00:00.000Z',
  }
}

const switchDocuments = {
  'doc-a': collaborationDocument('doc-a', 1),
  'doc-b': collaborationDocument('doc-b', 2),
}

function Surface({
  collaboration,
  document,
  onChange = noOp,
  onEditorReady,
}: {
  collaboration?: ReturnType<typeof useCollaborationDocument> | null
  document: EditorDocumentRecord
  onChange?: (contentMarkdown: string) => void
  onEditorReady: Parameters<typeof MarkdownEditorSurface>[0]['onEditorReady']
}) {
  return (
    <MarkdownEditorSurface
      collaboration={collaboration}
      comments={[]}
      copy={editorCopy.en}
      diffAnchorMarkdown={null}
      document={document}
      embedded
      isDiffVisible={false}
      mode="live"
      onAcceptSuggestion={noOp}
      onChange={onChange}
      onCreateComment={noOp}
      onEditSuggestion={noOp}
      onEditorReady={onEditorReady}
      onMarkSuggestionStale={noOp}
      onRefineSuggestion={noOpAsync}
      onRejectSuggestion={noOp}
      onSelectComment={noOp}
      onStopSuggestion={noOp}
      runningSuggestionIds={[]}
      selectedCommentId={null}
      suggestionErrors={{}}
      suggestions={[]}
      textImprovement={textImprovement}
    />
  )
}

function DocumentSwitchFixture() {
  const [requestedDocumentId, setRequestedDocumentId] = useState<'doc-a' | 'doc-b'>('doc-a')
  const [registration, setRegistration] = useState<EditorSurfaceRegistration | null>(null)
  const document = switchDocuments[requestedDocumentId]
  const identity = useMemo<EditorSurfaceIdentity>(() => ({
    documentId: document.id,
    generation: document.collaboration?.generation ?? null,
  }), [document])
  const collaboration = useCollaborationDocument({
    active: true,
    apiKey: undefined,
    document,
    workspaceId: 'browser-fixture',
  })
  const controllerRegistered = collaborationProjectionController(
    collaboration,
    identity.documentId,
    identity.generation,
  ) !== null
  const binding = collaborationBindingForEditorDocument(document, collaboration)
  const onEditorReady = useCallback<Parameters<typeof Surface>[0]['onEditorReady']>(
    (editor) => setRegistration((current) => updateEditorSurfaceRegistration(
      current,
      identity,
      editor,
    )),
    [identity],
  )

  useLayoutEffect(() => {
    const snapshots = window.__collaborationRenderSnapshots ?? []
    snapshots.push({
      controllerRegistered,
      handleDocumentId: collaboration.documentId,
      lifecycleStatus: collaboration.lifecycleStatus,
      requestedDocumentId,
    })
    window.__collaborationRenderSnapshots = snapshots
  }, [collaboration.documentId, collaboration.lifecycleStatus, controllerRegistered, requestedDocumentId])

  return (
    <main className="h-screen bg-background p-4">
      <button data-testid="switch-document" onClick={() => setRequestedDocumentId('doc-b')}>
        Switch to B
      </button>
      <dl
        className="t-meta-sm mt-2"
        data-binding-document-id={binding ? document.id : 'none'}
        data-controller-registered={controllerRegistered ? 'true' : 'false'}
        data-handle-document-id={collaboration.documentId ?? 'none'}
        data-lifecycle-status={collaboration.lifecycleStatus}
        data-registered-editor-id={editorForSurfaceIdentity(registration, identity) ? document.id : 'none'}
        data-requested-document-id={requestedDocumentId}
        data-testid="switch-state"
      >
        <dt>Requested</dt>
        <dd>{requestedDocumentId}</dd>
      </dl>
      <section className="mt-3 h-80 overflow-hidden border border-border" data-testid="switch-surface">
        <Surface collaboration={collaboration} document={document} onEditorReady={onEditorReady} />
      </section>
    </main>
  )
}

function activationState(): ProjectState {
  const base = createEmptyProjectState()
  const document: EditorDocumentRecord = {
    access: { mode: 'owner', permission: 'edit' },
    contentMarkdown: '# Writable legacy body\n\nThis starts in Markdown mode.',
    createdAt: '2026-07-15T08:00:00.000Z',
    folderId: null,
    id: 'activation-document',
    metadataRevision: 4,
    revision: 7,
    source: 'blank',
    title: 'Clock skew document.md',
    updatedAt: '2099-01-01T00:00:00.000Z',
  }
  return {
    ...base,
    dirty: false,
    editorDocumentOrder: [document.id],
    editorDocuments: { [document.id]: document },
    editorUi: {
      ...base.editorUi,
      activeDocumentId: document.id,
      openDocumentIds: [document.id],
    },
  }
}

function ActivationFixture() {
  const [state, dispatch] = useReducer(researchDeskReducer, undefined, activationState)
  const [bodyChangeCount, setBodyChangeCount] = useState(0)
  const document = state.editorDocuments['activation-document']
  const autosavePlan = planEditorDocumentAutosave(document)

  const activate = () => dispatch({
    collaboration: {
      generation: 5,
      persistedSequence: 0,
      projectionSequence: 0,
      schemaVersion: EDITOR_SCHEMA_VERSION,
    },
    documentId: document.id,
    metadataRevision: 5,
    type: 'activateEditorDocumentCollaboration',
  })
  const resolveExactDetail = () => dispatch({
    document: {
      ...document,
      collaboration: {
        generation: 5,
        persistedSequence: 2,
        projectionSequence: 2,
        projectionUpdatedAt: '2026-07-15T08:05:00.000Z',
        schemaVersion: EDITOR_SCHEMA_VERSION,
      },
      contentMarkdown: '# Exact hydrated projection\n\nThe delayed detail response won.',
      contentMode: 'collaboration',
      metadataRevision: 5,
      updatedAt: '2026-07-15T08:05:00.000Z',
    },
    type: 'setServerEditorDocumentDetail',
  })

  return (
    <main className="h-screen bg-background p-4">
      <div className="flex gap-2">
        <button data-testid="activation-success" onClick={activate}>Resolve activation</button>
        <button data-testid="detail-success" onClick={resolveExactDetail}>Resolve exact detail</button>
      </div>
      <dl
        className="t-meta-sm mt-2"
        data-autosave-kind={autosavePlan.kind}
        data-body-change-count={bodyChangeCount}
        data-content-mode={document.contentMode ?? 'markdown'}
        data-document-updated-at={document.updatedAt}
        data-testid="activation-state"
      >
        <dt>Mode</dt>
        <dd>{document.contentMode ?? 'markdown'}</dd>
      </dl>
      <section className="mt-3 h-80 overflow-hidden border border-border" data-testid="activation-surface">
        <Surface
          collaboration={null}
          document={document}
          onChange={(contentMarkdown) => {
            setBodyChangeCount((count) => count + 1)
            dispatch({
              contentMarkdown,
              documentId: document.id,
              type: 'updateEditorDocumentMarkdown',
            })
          }}
          onEditorReady={noOp}
        />
      </section>
    </main>
  )
}

const participants = [
  { color: '#2563EB', id: 'ada', name: 'Ada Lovelace' },
  { color: '#047857', id: 'lin', name: 'Lin Chen' },
  { color: '#B45309', id: 'max', name: 'Max Weber' },
  { color: '#7C3AED', id: 'zoe', name: 'Zoe Smith' },
]

function TopBarFixture() {
  const model = buildEditorCollaborationStatusModel({
    access: 'edit',
    active: true,
    canEdit: true,
    connectionStatus: 'connected',
    durabilityStatus: 'saved',
    participants,
    projectionUpdatedAt: '2026-07-15T10:02:00.000Z',
    synced: true,
  })
  return (
    <main className="w-screen bg-background">
      <EditorTopBarLayout
        actions={<EditorCollaborationStatus collaborationExpected model={model} variant="topbar" />}
        leading={(
          <span className="t-list-regular min-w-0 truncate" data-testid="long-title">
            A deliberately long collaboration document title that must remain inside its narrow track.md
          </span>
        )}
        toolbar={<span className="t-meta-sm whitespace-nowrap">Review</span>}
      />
    </main>
  )
}

function StatusHysteresisFixture() {
  const [durabilityStatus, setDurabilityStatus] = useState<'pending' | 'saved'>('saved')
  const model = buildEditorCollaborationStatusModel({
    access: 'edit',
    active: true,
    canEdit: true,
    connectionStatus: 'connected',
    durabilityStatus,
    participants: [],
    synced: true,
  })
  const pulse = (durationMs: number) => {
    setDurabilityStatus('pending')
    window.setTimeout(() => setDurabilityStatus('saved'), durationMs)
  }
  return (
    <main className="space-y-3 p-4">
      <div className="w-64">
        <EditorCollaborationStatus collaborationExpected model={model} variant="topbar" />
      </div>
      <button data-testid="quick-save-pulse" onClick={() => pulse(100)} type="button">
        Quick pulse
      </button>
      <button data-testid="slow-save-pulse" onClick={() => pulse(900)} type="button">
        Slow pulse
      </button>
    </main>
  )
}

const compoundChange: InspectorChange = {
  author: { color: '#2563EB', id: 'test-user', name: 'Test User' },
  createdAt: 10,
  id: 'patch-compound',
  originalText: 'old',
  position: 1,
  proposedText: 'new',
  suggestionIds: ['delete-active', 'insert-active'],
  type: 'replacement',
}

const inactiveDeletionChange: InspectorChange = {
  ...compoundChange,
  id: 'patch-inactive',
  originalText: 'hidden',
  proposedText: '',
  suggestionIds: ['delete-inactive'],
  type: 'deletion',
}

function reviewPolicy(
  writeMode: EditorWriteMode,
  overrides: Partial<CollaborationEditorPolicyInput> = {},
): CollaborationEditorPolicyInput {
  return {
    changesView: 'open',
    collaboration: true,
    display: 'simple',
    documentId: 'policy-document',
    inspectorTab: 'changes',
    selectedChangeId: null,
    visibleChanges: [],
    writeAuthorId: 'test-user',
    writeMode,
    ...overrides,
  }
}

function insertAtDocumentEnd(editor: NonNullable<ReturnType<typeof useEditor>>, text: string) {
  editor.view.dispatch(editor.state.tr.insertText(text, editor.state.doc.content.size - 1))
}

function InitialSuggestPolicyFixture() {
  const initialSuggestEditor = useEditor({
    content: '<p>Initial</p>',
    extensions: createEditorExtensions({
      collaborationReview: collaborationEditorPolicyUpdate(reviewPolicy('suggest')),
    }),
    immediatelyRender: false,
    onCreate: ({ editor }) => {
      insertAtDocumentEnd(editor, ' immediate')
    },
  })

  return (
    <main className="h-screen bg-background p-4">
      <section data-testid="initial-suggest-editor">
        <EditorContent editor={initialSuggestEditor} />
      </section>
    </main>
  )
}

function ModeSwitchPolicyFixture() {
  const modeSwitchEditor = useEditor({
    content: '<p>Switch</p>',
    extensions: createEditorExtensions({
      collaborationReview: collaborationEditorPolicyUpdate(reviewPolicy('edit')),
    }),
    immediatelyRender: false,
  })

  const switchToSuggestAndType = () => {
    if (!modeSwitchEditor) return
    applyCollaborationEditorPolicy(modeSwitchEditor, reviewPolicy('suggest'))
    insertAtDocumentEnd(modeSwitchEditor, ' switched')
  }

  return (
    <main className="h-screen bg-background p-4">
      <section data-testid="mode-switch-editor">
        <button data-testid="switch-to-suggest" onClick={switchToSuggestAndType}>Suggest</button>
        <EditorContent editor={modeSwitchEditor} />
      </section>
    </main>
  )
}

function SuggestionUndoPolicyFixture() {
  const [undoAttempts, setUndoAttempts] = useState(0)
  const [undoPatchId, setUndoPatchId] = useState<string | null>(null)
  const [updateCount, setUpdateCount] = useState(0)
  const document = useMemo(() => prosemirrorJSONToYDoc(
    getSchema(createEditorExtensions()),
    {
      content: [{
        attrs: { textAlign: null },
        content: [{ text: 'Before', type: 'text' }],
        type: 'paragraph',
      }],
      type: 'doc',
    },
    EDITOR_YJS_FRAGMENT,
  ), [])
  const editor = useEditor({
    extensions: [
      ...createEditorExtensions({
        collaborationReview: collaborationEditorPolicyUpdate(reviewPolicy('suggest')),
        onCollaborationSuggestionUndo: async (patchId) => {
          setUndoPatchId(patchId)
          setUndoAttempts((count) => count + 1)
          throw new Error('Fixture rejects the durable Undo decision.')
        },
      }).map((extension) => (
        extension.name === 'starterKit'
          ? extension.configure({ undoRedo: false })
          : extension
      )),
      Collaboration.configure({
        document,
        field: EDITOR_YJS_FRAGMENT,
      }),
    ],
    immediatelyRender: false,
  })

  useEffect(() => {
    const handleUpdate = () => setUpdateCount((count) => count + 1)
    document.on('update', handleUpdate)
    return () => document.off('update', handleUpdate)
  }, [document])

  return (
    <main className="h-screen bg-background p-4">
      <dl
        data-testid="suggestion-undo-state"
        data-undo-attempts={undoAttempts}
        data-undo-patch-id={undoPatchId ?? 'none'}
        data-update-count={updateCount}
      >
        <dt>Undo patch</dt>
        <dd>{undoPatchId ?? 'none'}</dd>
      </dl>
      <section className="min-h-80 border border-border p-4" data-testid="suggestion-undo-editor">
        <EditorContent editor={editor} />
      </section>
    </main>
  )
}

function ViewPolicyFixture() {
  const viewEditor = useEditor({
    content: '<p>View</p>',
    extensions: createEditorExtensions({
      collaborationReview: collaborationEditorPolicyUpdate(reviewPolicy('view')),
    }),
    immediatelyRender: false,
    onCreate: ({ editor }) => insertAtDocumentEnd(editor, ' blocked'),
  })

  return (
    <main className="h-screen bg-background p-4">
      <section data-testid="initial-view-editor">
        <EditorContent editor={viewEditor} />
      </section>
    </main>
  )
}

function SimpleMarkupFixture() {
  const markupEditor = useEditor({
    content: {
      content: [{
        content: [
          {
            marks: [{
              attrs: {
                authorId: 'test-user',
                createdAt: 10,
                id: 'delete-active',
                kind: 'deletion',
                patchId: 'patch-compound',
                suggestionId: 'delete-active',
              },
              type: 'deletion',
            }],
            text: 'old',
            type: 'text',
          },
          {
            marks: [{
              attrs: {
                authorId: 'test-user',
                createdAt: 10,
                id: 'insert-active',
                kind: 'insertion',
                patchId: 'patch-compound',
                suggestionId: 'insert-active',
              },
              type: 'insertion',
            }],
            text: 'new',
            type: 'text',
          },
          {
            marks: [{
              attrs: {
                authorId: 'test-user',
                createdAt: 11,
                id: 'delete-inactive',
                kind: 'deletion',
                patchId: 'patch-inactive',
                suggestionId: 'delete-inactive',
              },
              type: 'deletion',
            }],
            text: 'hidden',
            type: 'text',
          },
        ],
        type: 'paragraph',
      }],
      type: 'doc',
    },
    editable: false,
    extensions: createEditorExtensions({
      collaborationReview: collaborationEditorPolicyUpdate(reviewPolicy('suggest', {
        selectedChangeId: compoundChange.id,
        visibleChanges: [compoundChange, inactiveDeletionChange],
      })),
    }),
    immediatelyRender: false,
  })
  const finalMarkdown = markupEditor
    ? serializeEditorFinalProjectionMarkdown(markupEditor)
    : ''

  return (
    <main className="h-screen bg-background p-4">
      <section data-testid="simple-markup-editor">
        <EditorContent editor={markupEditor} />
      </section>
      <pre aria-label="Canonical Source" data-testid="canonical-source">{finalMarkdown}</pre>
      <section data-testid="canonical-diff">
        <DocumentDiffView
          anchorMarkdown="old"
          copy={editorCopy.en}
          currentMarkdown={finalMarkdown}
        />
      </section>
    </main>
  )
}

function SlashCollaborationFixture() {
  const searchParams = new URLSearchParams(window.location.search)
  const slashWriteMode = searchParams.get('mode') === 'suggest'
    ? 'suggest'
    : 'edit'
  const slashSourceKind = searchParams.get('source') === 'heading'
    ? 'heading'
    : 'paragraph'
  const document = useMemo(() => {
    const initialDocument = prosemirrorJSONToYDoc(
      getSchema(createEditorExtensions()),
      {
        content: slashWriteMode === 'suggest'
          ? [
              {
                attrs: { level: 1, textAlign: null },
                content: [{ text: 'Slash Suggest fixture', type: 'text' }],
                type: 'heading',
              },
              {
                attrs: slashSourceKind === 'heading'
                  ? { level: 2, textAlign: null }
                  : { textAlign: null },
                content: [{ text: 'Suggest target', type: 'text' }],
                type: slashSourceKind,
              },
            ]
          : [{
              attrs: { textAlign: null },
              content: [{ text: 'Before', type: 'text' }],
              type: 'paragraph',
            }],
        type: 'doc',
      },
      EDITOR_YJS_FRAGMENT,
    )
    if (slashWriteMode !== 'suggest') return initialDocument
    const synchronizedDocument = new Y.Doc()
    Y.applyUpdate(
      synchronizedDocument,
      Y.encodeStateAsUpdate(initialDocument),
    )
    window.__slashCollaborationServerUpdate = Array.from(
      Y.encodeStateAsUpdate(synchronizedDocument),
    )
    initialDocument.destroy()
    return synchronizedDocument
  }, [slashSourceKind, slashWriteMode])
  const editor = useEditor({
    extensions: [
      ...createEditorExtensions({
        collaborationReview: collaborationEditorPolicyUpdate(
          reviewPolicy('edit'),
        ),
        slash: {
          labels: {
            closeHint: editorCopy.de.slashClose,
            empty: editorCopy.de.slashEmpty,
            groupInsert: editorCopy.de.slashGroupInsert,
            groupStyle: editorCopy.de.slashGroupStyle,
            heading1: editorCopy.de.slashHeading1,
            heading2: editorCopy.de.slashHeading2,
            heading3: editorCopy.de.slashHeading3,
            navHint: editorCopy.de.slashNav,
            selectHint: editorCopy.de.slashSelect,
            text: editorCopy.de.slashText,
            title: editorCopy.de.slashTitle,
            bulletList: editorCopy.de.slashBulletList,
            orderedList: editorCopy.de.slashOrderedList,
            taskList: editorCopy.de.slashTaskList,
            blockquote: editorCopy.de.slashBlockquote,
            codeBlock: editorCopy.de.slashCodeBlock,
            table: editorCopy.de.slashTable,
            divider: editorCopy.de.slashDivider,
            suggestUnavailable: editorCopy.de.slashSuggestUnavailable,
          },
        },
      }).map((extension) => (
        extension.name === 'starterKit'
          ? extension.configure({ undoRedo: false })
          : extension
      )),
      Collaboration.configure({
        document,
        field: EDITOR_YJS_FRAGMENT,
      }),
    ],
    immediatelyRender: false,
  })

  useLayoutEffect(() => {
    if (!editor || slashWriteMode !== 'suggest') return
    applyCollaborationEditorPolicy(editor, reviewPolicy('suggest', {
      writeAuthorId: '11111111-1111-4111-8111-111111111111',
    }))
  }, [editor, slashWriteMode])

  useEffect(() => {
    const publish = (update?: Uint8Array) => {
      window.__slashCollaborationUpdate = Array.from(Y.encodeStateAsUpdate(document))
      if (update) {
        const updates = window.__slashCollaborationUpdates ?? []
        updates.push(Array.from(update))
        window.__slashCollaborationUpdates = updates
        const detections = window.__slashStructureUpdateDetections ?? []
        detections.push(containsStructureSuggestionAttribute(update))
        window.__slashStructureUpdateDetections = detections
        const suggestionBoundaries = window.__slashSuggestionBoundaryDetections ?? []
        suggestionBoundaries.push(containsSuggestionBoundary(update))
        window.__slashSuggestionBoundaryDetections = suggestionBoundaries
      }
    }
    window.__slashCollaborationUpdates = []
    window.__slashSuggestionBoundaryDetections = []
    window.__slashStructureUpdateDetections = []
    publish()
    document.on('update', publish)
    return () => document.off('update', publish)
  }, [document])

  return (
    <main className="h-screen bg-background p-4">
      <section className="min-h-80 border border-border p-4" data-testid="slash-collaboration-editor">
        <EditorContent editor={editor} />
      </section>
    </main>
  )
}

const COMMENT_FIXTURE_THREADS: EditorCollaborationCommentThread[] = Array.from(
  { length: 60 },
  (_, index) => {
    const revision = index + 1
    const createdAt = 1_784_390_000 + index * 60
    const author = {
      id: `user-${index % 6}`,
      kind: 'user' as const,
      name: `Reviewer ${index % 6 + 1}`,
    }
    return {
      anchor: { quote: `Passage ${revision}` },
      author,
      can_resolve: true,
      created_at: createdAt,
      document_id: 'comment-scale-document',
      generation: 1,
      id: `thread-${String(revision).padStart(2, '0')}`,
      messages: Array.from({ length: 3 }, (_, messageIndex) => ({
        author: messageIndex === 0
          ? author
          : {
              id: `user-${(index + messageIndex) % 6}`,
              kind: 'user' as const,
              name: `Reviewer ${(index + messageIndex) % 6 + 1}`,
            },
        body_markdown: messageIndex === 1 && index % 17 === 0
          ? null
          : `Review note ${revision}.${messageIndex + 1} with a deliberately long explanation that verifies compact wrapping without horizontal overflow in an enterprise inspector.`,
        can_delete: messageIndex > 0,
        can_edit: messageIndex > 0,
        created_at: createdAt + messageIndex * 10,
        deleted_at: messageIndex === 1 && index % 17 === 0
          ? createdAt + 30
          : null,
        edited_at: messageIndex === 2 && index % 11 === 0
          ? createdAt + 40
          : null,
        id: `message-${revision}-${messageIndex + 1}`,
        mentions: messageIndex === 2 && index % 7 === 0
          ? [{ id: 'user-1', kind: 'user' as const, name: 'Reviewer 2' }]
          : [],
        revision,
      })),
      quote: `Selected source passage ${revision} with enough text to exercise the compact quote preview.`,
      resolved_at: index >= 55 ? createdAt + 120 : null,
      resolved_by: index >= 55 ? author : null,
      revision,
      status: index >= 55 ? 'resolved' : 'open',
      updated_at: createdAt + 20,
    }
  },
)

function CommentScaleFixture() {
  const [loadedCount, setLoadedCount] = useState(50)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const threads = COMMENT_FIXTURE_THREADS.slice(0, loadedCount)
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      window.__commentScaleFirstInteractiveMs = (
        performance.now() - fixtureModuleStartedAt
      )
    })
    return () => cancelAnimationFrame(frame)
  }, [])
  useEffect(() => {
    if (!selectedThreadId || window.__commentScaleSelectionStartedAt === undefined) return
    const frame = requestAnimationFrame(() => {
      window.__commentScaleSelectionMs = (
        performance.now() - window.__commentScaleSelectionStartedAt!
      )
      delete window.__commentScaleSelectionStartedAt
    })
    return () => cancelAnimationFrame(frame)
  }, [selectedThreadId])
  const comments = useMemo<CollaborationCommentsHandle>(() => ({
    createThread: async () => COMMENT_FIXTURE_THREADS[0]!,
    deleteMessage: noOpAsync,
    drafts,
    editMessage: noOpAsync,
    error: null,
    hasMore: loadedCount < COMMENT_FIXTURE_THREADS.length,
    isLoading: false,
    isLoadingMore,
    lastReadRevision: 42,
    loadMore: async () => {
      setIsLoadingMore(true)
      await Promise.resolve()
      setLoadedCount((current) => Math.min(current + 50, 60))
      setIsLoadingMore(false)
    },
    markRead: noOpAsync,
    mentionEventVersion: 0,
    participants: Array.from({ length: 6 }, (_, index) => ({
      id: `user-${index}`,
      kind: 'user' as const,
      name: `Reviewer ${index + 1}`,
    })),
    pendingIds: new Set<string>(),
    reply: noOpAsync,
    revision: 60,
    setDraft: (key, value) => setDrafts((current) => ({
      ...current,
      [key]: value,
    })),
    setStatus: noOpAsync,
    threads,
    unreadCount: threads.filter((thread) => thread.revision > 42).length,
  }), [drafts, isLoadingMore, loadedCount, threads])
  const orphanedThreadIds = useMemo(
    () => new Set(
      COMMENT_FIXTURE_THREADS
        .filter((_, index) => index % 13 === 0)
        .map((thread) => thread.id),
    ),
    [],
  )
  const positionByThreadId = useMemo(
    () => new Map(
      COMMENT_FIXTURE_THREADS.map((thread, index) => [thread.id, index * 10]),
    ),
    [],
  )

  return (
    <main className="h-screen min-w-0 bg-background">
      <aside
        className="h-full w-full min-w-0 overflow-hidden border-l border-border sm:ml-auto sm:w-[400px]"
        data-testid="comment-scale-panel"
      >
        <TeamCommentsPanel
          assistantAvailable={false}
          canComment
          comments={comments}
          currentUserId="user-1"
          onSelectThread={(threadId) => {
            window.__commentScaleSelectionStartedAt = performance.now()
            setSelectedThreadId(threadId)
          }}
          onUseWithAssistant={noOp}
          orphanedThreadIds={orphanedThreadIds}
          positionByThreadId={positionByThreadId}
          selectedThreadId={selectedThreadId}
        />
      </aside>
    </main>
  )
}

function ViewOnlyAiFixture() {
  const composerRef = useRef<MentionComposerHandle | null>(null)
  const [access, setAccess] = useState<'edit' | 'view'>('edit')
  const [anchorInvocations, setAnchorInvocations] = useState(0)
  const [decisionInvocations, setDecisionInvocations] = useState(0)
  const [invocations, setInvocations] = useState(0)
  const [publishInvocations, setPublishInvocations] = useState(0)
  const document: EditorDocumentRecord = {
    ...collaborationDocument('private-ai-document', 4),
    access: { mode: 'owner', permission: 'edit' },
  }
  const collaboration = {
    access,
    canEdit: access === 'edit',
    documentId: document.id,
    generation: document.collaboration?.generation ?? null,
    readAuthority: () => ({
      access,
      blockingFailure: null,
      canEdit: access === 'edit',
      connectionStatus: access === 'edit' ? 'connected' as const : 'read_only' as const,
      documentId: document.id,
      generation: document.collaboration?.generation ?? null,
      lifecycleStatus: access === 'edit' ? 'saved' as const : 'read_only' as const,
      revision: access === 'edit' ? 0 : 1,
      synced: true,
    }),
  } as ReturnType<typeof useCollaborationDocument>
  const reason = editorAiReadOnlyReason(document, collaboration.access, 'en')
  const publishDisabledReason = privateSuggestionPublishDisabledReason(
    document,
    collaboration,
    'en',
  )
  const collaborationStatus = buildEditorCollaborationStatusModel({
    access,
    active: true,
    canEdit: access === 'edit',
    connectionStatus: access === 'edit' ? 'connected' : 'read_only',
    durabilityStatus: 'saved',
    participants: [],
    synced: true,
  })
  const suggestion: EditorSuggestionRecord = {
    anchor: {
      from: 1,
      quoteAfter: '',
      quoteBefore: '',
      selectedText: 'old',
      to: 4,
    },
    blockId: 'paragraph-1',
    createdAt: document.createdAt,
    documentId: document.id,
    groupId: 'private-ai-group',
    id: 'private-ai-suggestion',
    originalText: 'old',
    origin: { kind: 'global_run' },
    proposedText: 'new',
    status: 'pending',
    updatedAt: document.updatedAt,
  }
  return (
    <main className="h-screen bg-background">
      <button data-testid="downgrade-ai-access" onClick={() => setAccess('view')}>Downgrade</button>
      <output data-testid="anchor-invocations">{anchorInvocations}</output>
      <output data-testid="decision-invocations">{decisionInvocations}</output>
      <output data-testid="ai-invocations">{invocations}</output>
      <output data-testid="publish-invocations">{publishInvocations}</output>
      <EditorTopBar
        canFlushForShare={false}
        collaborationAccess={access}
        collaborationActive
        collaborationCanEdit={access === 'edit'}
        collaborationStatus={collaborationStatus}
        copy={editorCopy.en}
        diffAnchorDisabledReason={publishDisabledReason}
        diffAnchorError={null}
        diffAnchorPending={false}
        dispatch={noOp}
        document={document}
        editor={null}
        isCommentPanelVisible={false}
        isDiffVisible={false}
        isDirty={false}
        isTreeVisible={false}
        onExportWordMarkdown={async () => document.contentMarkdown}
        onSetDiffAnchor={() => setAnchorInvocations((count) => count + 1)}
        onShareDocument={noOp}
        onWriteModeChange={noOp}
        sharingAvailable={false}
        viewMode="live"
        writeMode={access === 'edit' ? 'edit' : 'view'}
      />
      <section data-testid="private-suggestion-controls">
        <EditorDocumentChangesSection
          labels={{
            accept: 'Accept',
            acceptAll: 'Accept all',
            documentChanges: 'Document changes',
            proposedChange: 'Proposed change',
            reject: 'Reject',
            rejectAll: 'Reject all',
          }}
          onAcceptGroup={() => setPublishInvocations((count) => count + 1)}
          onAcceptSuggestion={() => setPublishInvocations((count) => count + 1)}
          onRejectGroup={noOp}
          onRejectSuggestion={noOp}
          onSelectSuggestion={noOp}
          publishDisabledReason={publishDisabledReason}
          suggestionErrors={{}}
          suggestions={[suggestion]}
        />
      </section>
      <section className="h-96" data-testid="batch-inspector">
        <EditorInspector
          activeTab="changes"
          assistant={<div />}
          canDecide={access === 'edit'}
          changes={[compoundChange]}
          changesError={null}
          changesView="open"
          collaborationActive
          collaborationStatus={collaborationStatus}
          commentCount={0}
          comments={<div />}
          commentUnreadCount={0}
          decisionError={null}
          display="simple"
          history={[]}
          historyError={null}
          historyFilters={{ actorId: null, type: null }}
          historyLoading={false}
          isDecisionPending={false}
          onActiveTabChange={noOp}
          onChangesViewChange={noOp}
          onClose={noOp}
          onDecision={() => setDecisionInvocations((count) => count + 1)}
          onDisplayChange={noOp}
          onHistoryFiltersChange={noOp}
          onOpenFiltersChange={noOp}
          onSelectedChangeIdChange={noOp}
          openFilters={{ authorId: null, type: null }}
          selectedChangeId={null}
        />
      </section>
      <EditorAssistantComposer
        aiReadOnlyReason={reason}
        attachedCommentIds={[]}
        attachmentChips={[]}
        chatModelOptions={[]}
        chatModelOptionsStatus="missing"
        comments={[]}
        composerRef={composerRef}
        copy={editorCopy.en}
        defaultChatModel={null}
        dispatch={noOp}
        draft="Rewrite this paragraph"
        editorContextBase={{ conversation: 0, documents: 0, reports: 0, rules: 0 }}
        editorContextCapacity={{ contextWindowTokens: null, reservedOutputTokens: 0 }}
        fileGroupOptions={[]}
        fileOptions={[]}
        instructionFeedback={null}
        isAttachActive={false}
        isRunning={false}
        isVisible
        isWideCanvas={false}
        onAttachFiles={noOp}
        onAttachRule={noOp}
        onDismissInstructionFeedback={noOp}
        onRefsChange={noOp}
        onRemoveAttachedComment={noOp}
        onRemoveChip={noOp}
        onReorderPending={noOp}
        onReorderPill={noOp}
        onSend={() => setInvocations((count) => count + 1)}
        onStop={noOp}
        onToggleAttach={noOp}
        pendingKeys={[]}
        pillKeys={[]}
        reportOptions={[]}
        ruleOptions={[]}
        selectedEffort={null}
        selectedModel={null}
        selectedModelTier={null}
        textImprovement={textImprovement}
      />
    </main>
  )
}

function ModalFocusRaceFixture() {
  const [open, setOpen] = useState(false)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const panelRef = useRef<HTMLElement | null>(null)
  const initialFocusRef = useRef<HTMLInputElement | null>(null)
  useModalFocusTrap({
    initialFocusRef,
    onClose: () => setOpen(false),
    open,
    panelRef,
  })
  useEffect(() => {
    if (open) triggerRef.current?.focus()
  }, [open])
  return (
    <main className="h-screen bg-background p-8">
      <button
        data-testid="modal-launcher"
        onClick={() => setOpen(true)}
        ref={triggerRef}
        type="button"
      >
        Open modal
      </button>
      {open ? (
        <section
          aria-label="Focus race dialog"
          aria-modal="true"
          ref={panelRef}
          role="dialog"
          tabIndex={-1}
        >
          <input aria-label="Preferred modal control" ref={initialFocusRef} />
          <button onClick={() => setOpen(false)} type="button">Close</button>
        </section>
      ) : null}
    </main>
  )
}

function ShareMenuFocusFixture() {
  const [open, setOpen] = useState(false)
  const [returnFocusTarget, setReturnFocusTarget] = useState<HTMLElement | null>(null)
  const document = useMemo(() => collaborationDocument('share-menu', 1), [])
  const collaborationStatus = buildEditorCollaborationStatusModel({
    access: 'edit',
    active: true,
    canEdit: true,
    connectionStatus: 'connected',
    durabilityStatus: 'saved',
    participants: [],
    synced: true,
  })
  return (
    <main className="h-screen bg-background">
      <EditorTopBar
        canFlushForShare
        collaborationAccess="edit"
        collaborationActive
        collaborationCanEdit
        collaborationStatus={collaborationStatus}
        copy={editorCopy.en}
        diffAnchorDisabledReason={null}
        diffAnchorError={null}
        diffAnchorPending={false}
        dispatch={noOp}
        document={document}
        editor={null}
        isCommentPanelVisible={false}
        isDiffVisible={false}
        isDirty={false}
        isTreeVisible={false}
        onExportWordMarkdown={async () => document.contentMarkdown}
        onSetDiffAnchor={noOp}
        onShareDocument={(_document, target) => {
          setReturnFocusTarget(target ?? null)
          setOpen(true)
        }}
        onWriteModeChange={noOp}
        sharingAvailable
        viewMode="live"
        writeMode="edit"
      />
      {open ? (
        <ShareDialog
          demo
          initialTab="access"
          onClose={() => setOpen(false)}
          ownerEmail="owner@example.de"
          ownerName="Owner"
          refreshToken={0}
          resourceId={document.id}
          resourceTitle={document.title}
          resourceType="editor_document"
          returnFocusTarget={returnFocusTarget}
        />
      ) : null}
    </main>
  )
}

function ExplorerActionModalFixture() {
  const [open, setOpen] = useState(false)
  const [pinCount, setPinCount] = useState(0)
  const [returnFocusTarget, setReturnFocusTarget] = useState<HTMLElement | null>(null)
  const title = 'Geteiltes Dokument.md'
  const detailsLabel = 'Dokumentdetails'

  return (
    <main className="h-screen bg-background p-8">
      <div className="w-72 rounded-lg border border-border bg-surface p-2">
        <ExplorerHistoryRow
          actions={[{
            ariaLabel: `${detailsLabel}: ${title}`,
            icon: <Info className="icon-sm" />,
            label: detailsLabel,
            onSelect: () => {
              const activeElement = window.document.activeElement
              setReturnFocusTarget(activeElement instanceof HTMLElement ? activeElement : null)
              setOpen(true)
            },
          }]}
          onSelect={noOp}
          timeLabel="gerade eben"
          title={title}
        />
        <ExplorerHistoryRow
          actions={[{
            ariaLabel: 'Anheften: Lokales Dokument.md',
            icon: <span aria-hidden="true">P</span>,
            label: 'Anheften',
            onSelect: () => setPinCount((count) => count + 1),
          }]}
          onSelect={noOp}
          timeLabel="1 Minute"
          title="Lokales Dokument.md"
        />
        <output data-testid="explorer-pin-count">{pinCount}</output>
      </div>
      {open ? (
        <ShareDialog
          demo
          documentDetails={{
            createdAt: '2026-08-03T17:30:14.000Z',
            openCommentCount: 1,
            openSuggestionCount: 1,
            participantCount: 2,
            updatedAt: '2026-08-03T17:30:14.000Z',
            wordCount: 42,
          }}
          initialTab="overview"
          onClose={() => setOpen(false)}
          ownerEmail="owner@example.invalid"
          ownerName="Fixture Owner"
          refreshToken={0}
          resourceId="shared-document"
          resourceTitle={title}
          resourceType="editor_document"
          returnFocusTarget={returnFocusTarget}
        />
      ) : null}
    </main>
  )
}

const LONG_COLLABORATOR_NAME = (
  'Dr. Alexandra-Maria von Beispielhausen 🧪 · International Collaboration Reviewer'
)

const presenceLabelSamples = [
  { color: '#2563eb', kind: 'current', side: 'left', top: 72 },
  { color: '#7c3aed', kind: 'current', side: 'right', top: 152 },
  { color: '#0f766e', kind: 'legacy', side: 'left', top: 232 },
  { color: '#be123c', kind: 'legacy', side: 'right', top: 312 },
] as const

function legacyCollaborationCaret(name: string, color: string): HTMLElement {
  const caret = document.createElement('span')
  caret.className = 'collaboration-carets__caret'
  caret.style.borderColor = color

  const label = document.createElement('span')
  label.className = 'collaboration-carets__label'
  label.style.backgroundColor = color
  label.textContent = name
  caret.append(label)
  return caret
}

const PresenceLabelFixtureExtension = Extension.create({
  name: 'presenceLabelFixture',
  addProseMirrorPlugins() {
    return [
      new Plugin({
        props: {
          decorations(state) {
            const position = Math.min(1, state.doc.content.size)
            const widgets = presenceLabelSamples.map((sample) => {
              const sampleId = `${sample.kind}-${sample.side}`
              return Decoration.widget(position, () => {
                const anchor = document.createElement('span')
                anchor.contentEditable = 'false'
                anchor.dataset.presenceAnchor = sampleId
                anchor.style.position = 'absolute'
                anchor.style.top = `${sample.top}px`
                if (sample.side === 'left') anchor.style.left = '4px'
                else anchor.style.right = '4px'

                const name = `${LONG_COLLABORATOR_NAME} · ${sample.kind} ${sample.side}`
                const caret = sample.kind === 'current'
                  ? renderCollaborationCaret({ color: sample.color, name })
                  : legacyCollaborationCaret(name, sample.color)
                caret.dataset.presenceSample = sampleId
                anchor.append(caret)
                return anchor
              }, { key: `presence-label-${sampleId}`, side: -1 })
            })
            return DecorationSet.create(state.doc, widgets)
          },
        },
      }),
    ]
  },
})

function PresenceLabelFixture() {
  const editor = useEditor({
    content: [
      '<p>Remote identities must remain readable without covering the active paragraph.</p>',
      '<p>The same presentation contract applies at either horizontal edge.</p>',
      '<p>Unicode names and emoji are ordinary collaboration identities.</p>',
    ].join(''),
    editorProps: {
      attributes: {
        class: 'relative min-h-[28rem] px-3 py-12 outline-none',
        'data-testid': 'presence-label-editor',
      },
    },
    extensions: [...createEditorExtensions(), PresenceLabelFixtureExtension],
    immediatelyRender: true,
  })

  return (
    <main className="min-h-screen overflow-x-hidden bg-background p-3 text-foreground">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h1 className="t-title">Collaboration presence</h1>
        <output
          className="max-w-full rounded-full border border-border bg-surface px-2 py-1 t-meta"
          data-testid="presence-demo-state"
        >
          Demo-Modus eingeschaltet
        </output>
      </header>
      <section
        className="overflow-visible rounded-lg border border-border bg-surface"
        data-labels-mounted={editor ? 'true' : 'false'}
        data-testid="presence-label-boundary"
      >
        <EditorContent editor={editor} />
      </section>
    </main>
  )
}

function ProfileMenuFixture() {
  const [logoutCount, setLogoutCount] = useState(0)
  return (
    <main className="relative h-screen bg-background">
      <output data-testid="profile-logout-count">{logoutCount}</output>
      <div className="absolute bottom-2 left-2">
        <ProfileAvatar
          authMode="local"
          onLogin={noOp}
          onLogout={() => setLogoutCount((count) => count + 1)}
          onOpenSecuritySettings={noOp}
          session={{
            projectNamespace: 'ws_browser_fixture',
            status: 'authenticated',
            user: {
              displayName: 'Fixture Owner',
              email: 'owner@example.invalid',
              id: 'fixture-owner',
              role: 'admin',
            },
          }}
        />
      </div>
    </main>
  )
}

const scenario = new URLSearchParams(window.location.search).get('scenario')
const fixture = scenario === 'switch'
  ? <DocumentSwitchFixture />
  : scenario === 'activation'
    ? <ActivationFixture />
    : scenario === 'initial-suggest-policy'
      ? <InitialSuggestPolicyFixture />
      : scenario === 'suggestion-undo-policy'
        ? <SuggestionUndoPolicyFixture />
      : scenario === 'mode-switch-policy'
        ? <ModeSwitchPolicyFixture />
        : scenario === 'view-policy'
          ? <ViewPolicyFixture />
          : scenario === 'simple-markup'
            ? <SimpleMarkupFixture />
            : scenario === 'slash-collaboration'
              ? <SlashCollaborationFixture />
              : scenario === 'comment-scale'
                ? <CommentScaleFixture />
                : scenario === 'status-hysteresis'
                  ? <StatusHysteresisFixture />
                  : scenario === 'modal-focus-race'
                    ? <ModalFocusRaceFixture />
                    : scenario === 'share-menu-focus'
                      ? <ShareMenuFocusFixture />
                      : scenario === 'explorer-action-modal'
                        ? <ExplorerActionModalFixture />
                      : scenario === 'presence-label'
                        ? <PresenceLabelFixture />
                      : scenario === 'profile-menu'
                        ? <ProfileMenuFixture />
      : scenario === 'view-ai'
        ? <ViewOnlyAiFixture />
        : <TopBarFixture />

createRoot(document.getElementById('root')!).render(<AppProviders>{fixture}</AppProviders>)
