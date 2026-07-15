import '@fontsource-variable/inter/index.css'
import 'katex/dist/katex.min.css'
import '@/styles/globals.css'

import { useCallback, useLayoutEffect, useMemo, useReducer, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { EditorContent, useEditor } from '@tiptap/react'

import { AppProviders } from '@/app/AppProviders'
import { createEmptyProjectState } from '@/features/project/seedProject'
import type { EditorDocumentRecord, EditorSuggestionRecord, ProjectState } from '@/features/project/types'
import { researchDeskReducer } from '@/features/researchDesk/state'
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
import { createEditorExtensions, serializeEditorFinalProjectionMarkdown } from '../tiptap'
import { useCollaborationDocument } from '../useCollaborationDocument'
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
  }
}

const noOp = () => undefined
const noOpAsync = async () => undefined
const textImprovement = { enabled: false, workspaceId: 'browser-fixture' }

function collaborationDocument(id: string, generation: number): EditorDocumentRecord {
  return {
    access: { mode: 'owner', permission: 'edit' },
    collaboration: {
      generation,
      persistedSequence: 0,
      projectionSequence: 0,
      schemaVersion: 1,
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
      schemaVersion: 1,
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
        schemaVersion: 1,
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
        actions={<EditorCollaborationStatus model={model} variant="topbar" />}
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

const compoundChange: InspectorChange = {
  author: { color: '#2563EB', id: 'test-user', name: 'Test User' },
  createdAt: 10,
  id: 'patch-compound',
  originalText: 'old',
  position: 1,
  proposedText: 'new',
  suggestionIds: ['delete-active', 'insert-active'],
  type: 'modification',
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
        commentCount={0}
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
          collaborationStatus={collaborationStatus}
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

const scenario = new URLSearchParams(window.location.search).get('scenario')
const fixture = scenario === 'switch'
  ? <DocumentSwitchFixture />
  : scenario === 'activation'
    ? <ActivationFixture />
    : scenario === 'initial-suggest-policy'
      ? <InitialSuggestPolicyFixture />
      : scenario === 'mode-switch-policy'
        ? <ModeSwitchPolicyFixture />
        : scenario === 'view-policy'
          ? <ViewPolicyFixture />
          : scenario === 'simple-markup'
            ? <SimpleMarkupFixture />
      : scenario === 'view-ai'
        ? <ViewOnlyAiFixture />
        : <TopBarFixture />

createRoot(document.getElementById('root')!).render(<AppProviders>{fixture}</AppProviders>)
