import { useEffect, useMemo, useRef, useState, type Dispatch } from 'react'
import {
  createEditorSchemaExtensions,
  createRelativePositionAdapter,
  createSecurePrefixedId,
  createSecureUuid,
  parseEditorMarkdown,
  serializeEditorJson,
  type EditorRelativePositionAdapter,
  type ProseMirrorMapping,
} from '@inqtrix/editor-schema'
import { Editor as HeadlessEditor, type JSONContent } from '@tiptap/core'
import { ySyncPluginKey } from '@tiptap/y-tiptap'
import type { Editor } from '@tiptap/react'
import type * as Y from 'yjs'
import {
  applyEditorPatch,
  deleteEditorComment,
  deleteEditorCommentSuggestionDraft,
  getEditorDocument,
  getEditorPatch,
  hasHttpStatus,
  listEditorDocumentPatches,
  publishEditorCollaborationSuggestion,
  rejectEditorPatch,
  saveEditorComments,
  saveEditorCommentSuggestionDraft,
  type ClientOptions,
  type EditorCollaborationSuggestionPublishResponse,
  type EditorSuggestionDraftCreateWire,
  type EditorSuggestionDraftRevisionRequestWire,
} from '@/api/inqtrixClient'
import type { AgentPatchWire } from '@/features/agent/types'
import type { ChatModelTier, InqtrixCapabilities } from '@/features/researchRuns/types'
import { deriveEditorAbortMs } from '@/features/researchRuns/clientTimeouts'
import type {
  ChatContextReferenceRecord,
  EditorCommentAnchorRecord,
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorSuggestionEditPosition,
  EditorSuggestionCollaborationPublication,
  EditorSuggestionGroupRecord,
  EditorSuggestionOrigin,
  EditorSuggestionRecord,
  ProjectState,
} from '@/features/project/types'
import {
  assetIdsFromChatRefs,
  attachmentContextReadiness,
  referenceDocsFromRefs,
} from '@/features/project/selectors'
import { renderChatRuleAttachmentContent } from '@/features/project/chatRuleRendering'
import type { ResearchDeskAction } from '@/features/researchDesk/state'
import {
  blockInsertionPositionForRange,
  clampAnchor,
  materializeAnchorForRange,
  materializeCommentThread,
  resolveAnchorRange,
  type EditorTextRange,
} from './anchoring'
import { normalizeEditorMarkdownForTiptap } from './tiptap'
import {
  hasCollaborationRelativeAnchor,
  resolveCollaborationAnchor,
  serializeCollaborationAnchor,
} from './inspector/relativeAnchors'
import {
  LlmSuggestionProducer,
  type InstructionProposal,
  type SuggestionProducer,
} from './suggestionProducer'
import {
  clearRunning,
  clearRuns,
  markError,
  markErrors,
  markManyRunning,
  markRunning,
  runErrors as runErrorsOf,
  runningIds as runningIdsOf,
  type EditorRunStateMap,
} from './editorRunState'
import {
  CollaborationProjectionBarrierError,
  collaborationProjectionController,
  flushCollaborationProjectionBarrier,
  setAuthoritativeCollaborationSequence,
  type CollaborationProjectionController,
  type ConfirmedCollaborationProjection,
} from './collaborationProjection'
import {
  beginCollaborationAuthorityGuard,
  collaborationAuthorityDisabledReason,
  CollaborationAuthorityError,
  type CollaborationAuthorityGuard,
  type CollaborationAuthorityRequirement,
} from './collaborationAuthority'
import type { CollaborationDocumentHandle } from './useCollaborationDocument'
import {
  privateSuggestionDraftRecordFromServer,
  serverCommentPayload,
} from './editorSync'

export type UseEditorSuggestionsArgs = {
  activeDocument: EditorDocumentRecord | null
  activeEditor: Editor | null
  apiKey?: string
  attachedCommentIds: string[]
  attachedRefs: ChatContextReferenceRecord[]
  /** Server capability manifest. Its `timeouts.editor_wait_seconds` sets the
   * client abort budget so a server-side EDITOR_ASSISTANT_TIMEOUT raise is not
   * silently capped by the browser. Null offline / pre-discovery -> a fixed
   * fallback (logged once). */
  capabilities: InqtrixCapabilities | null
  collaboration: CollaborationDocumentHandle
  comments: EditorCommentThreadRecord[]
  dispatch: Dispatch<ResearchDeskAction>
  /** Loads attached file-asset bodies on demand before an AI run reads them
   * (M6c load-on-use). Absent offline / when assets are not server-synced —
   * bodies are local then and resolveRunContext is a no-op pass-through. */
  ensureAssetBodiesLoaded?: (assetIds: readonly string[]) => Promise<Map<string, string>>
  locale: 'de' | 'en'
  onCollaborationPublication?: (
    documentId: string,
    publication: CollaborationSuggestionPublication,
  ) => void
  onGlobalSuccess: () => void
  selectedModelTier: ChatModelTier | null
  state: ProjectState
}

export type EditorSuggestionController = {
  aiReadOnlyReason: string | null
  clearInstructionFeedback: () => void
  documentSuggestions: EditorSuggestionRecord[]
  handleAcceptSuggestionGroup: (groupId: string) => Promise<void>
  handleAcceptSuggestion: (suggestion: EditorSuggestionRecord) => Promise<void>
  handleEditSuggestionProposal: (suggestionId: string, proposedText: string) => Promise<void>
  handleGlobalRun: (globalInstruction: string) => Promise<void>
  handleInstructionRun: (instruction: string) => Promise<void>
  handleMarkSuggestionStale: (suggestionId: string) => void
  handleRefineSuggestion: (suggestionId: string, instruction: string) => Promise<void>
  handleRejectSuggestion: (suggestionId: string) => Promise<void>
  handleRejectSuggestionGroup: (groupId: string) => Promise<void>
  handleRunComment: (comment: EditorCommentThreadRecord) => Promise<void>
  handleStopRun: () => void
  handleStopSuggestionRun: (suggestionId: string) => void
  instructionFeedback: EditorInstructionFeedback | null
  isGlobalRunning: boolean
  runErrors: Record<string, string>
  runningCommentIds: readonly string[]
  savingCommentDraftIds: readonly string[]
  runningSuggestionIds: readonly string[]
  suggestionPublishDisabledReason: string | null
  suggestionErrors: Record<string, string>
}

export type EditorInstructionFeedback = {
  editCount?: number
  message: string
  state: 'error' | 'result' | 'thinking'
  warnings?: string[]
}

export type CollaborationSuggestionPublication = EditorSuggestionCollaborationPublication

export type EditorAiDocumentContext = {
  markdown: string
  sequence: number | null
}

type PrivateSuggestionPublicationAttempt = {
  commandId: string
  expectedSequence: number
  patchId: string
  targetMarkdown: string
}

export function tryAcquireEditorRunLatch(
  latch: { current: boolean },
): boolean {
  if (latch.current) return false
  latch.current = true
  return true
}

export function editorAiReadOnlyReason(
  document: EditorDocumentRecord | null,
  collaborationAccess: CollaborationDocumentHandle['access'],
  locale: 'de' | 'en',
): string | null {
  if (document?.recovery) return editorRecoveryAiDisabledMessage(locale)
  if (document?.contentMode !== 'collaboration') return null
  const access = collaborationAccess ?? document.access?.permission ?? null
  if (access !== 'view') return null
  return locale === 'de'
    ? 'KI-Bearbeitung ist mit schreibgeschütztem Zugriff nicht verfügbar.'
    : 'AI editing is unavailable with view-only access.'
}

export async function invokeEditorAiProvider<T>(
  document: EditorDocumentRecord,
  collaborationAccess: CollaborationDocumentHandle['access'],
  locale: 'de' | 'en',
  invoke: () => Promise<T>,
  authorityGuard: CollaborationAuthorityGuard | null = null,
): Promise<T> {
  const reason = editorAiReadOnlyReason(document, collaborationAccess, locale)
  if (reason) throw new Error(reason)
  authorityGuard?.assertCurrent()
  const result = await invoke()
  authorityGuard?.assertCurrent()
  return result
}

export function editorCollaborationActionDisabledReason(
  document: EditorDocumentRecord | null,
  collaboration: CollaborationDocumentHandle,
  requirement: CollaborationAuthorityRequirement,
  locale: 'de' | 'en',
): string | null {
  if (document?.recovery) return editorRecoveryAiDisabledMessage(locale)
  if (document?.contentMode !== 'collaboration') return null
  const identity = collaborationIdentity(document)
  if (!identity) return collaborationPublishForbiddenMessage(locale)
  return collaborationAuthorityDisabledReason(collaboration, identity, requirement, locale)
}

export function privateSuggestionPublishDisabledReason(
  document: EditorDocumentRecord | null,
  collaboration: CollaborationDocumentHandle,
  locale: 'de' | 'en',
): string | null {
  // A recovery copy is a local legacy document. Its preserved local
  // suggestions may still be accepted or rejected, but there is no
  // collaboration publication surface until the copy is deliberately
  // promoted to a new document.
  if (document?.recovery) return null
  return editorCollaborationActionDisabledReason(document, collaboration, 'write', locale)
}

/** Gate the complete private-to-shared publication operation before it can
 * resolve a projection or invoke the publication endpoint. */
export async function invokePrivateSuggestionPublication<T>(
  document: EditorDocumentRecord,
  collaboration: CollaborationDocumentHandle,
  locale: 'de' | 'en',
  invoke: () => Promise<T>,
  authorityGuard: CollaborationAuthorityGuard | null = null,
): Promise<T> {
  const guard = authorityGuard ?? beginEditorCollaborationAuthorityGuard(
    document,
    collaboration,
    'write',
    locale,
  )
  guard?.assertCurrent()
  const result = await invoke()
  guard?.assertCurrent()
  return result
}

export type PrivateSuggestionAnchorStatus =
  | 'degraded'
  | 'failed'
  | 'legacy'
  | 'relative'

export type PrivateSuggestionAnchorResult = {
  anchor: EditorCommentAnchorRecord
  reason?: 'adapter_unavailable' | 'encoding_failed' | 'relative_missing' | 'relative_unresolved'
  status: PrivateSuggestionAnchorStatus
}

/** Return only a durable canonical projection for collaboration AI context.
 * A live/projection mismatch means local Yjs updates are still pending and is
 * an explicit failure; the local markdown snapshot is never a fallback. */
export async function resolveEditorAiDocumentContext(
  document: EditorDocumentRecord,
  editor: Pick<Editor, 'getJSON'> | null,
  controller: CollaborationProjectionController | null,
  options: ClientOptions,
  locale: 'de' | 'en',
  flushBarrier: typeof flushCollaborationProjectionBarrier = flushCollaborationProjectionBarrier,
  authorityGuard: CollaborationAuthorityGuard | null = null,
): Promise<EditorAiDocumentContext> {
  if (document.contentMode !== 'collaboration') {
    return { markdown: document.contentMarkdown, sequence: null }
  }
  if (!editor) throw new Error(collaborationEditorUnavailableMessage(locale))
  authorityGuard?.assertCurrent()

  let projection: ConfirmedCollaborationProjection
  try {
    projection = await flushBarrier({
      authorityGuard,
      clientOptions: options,
      controller,
      documentId: document.id,
      generation: document.collaboration?.generation,
    })
  } catch (error) {
    if (error instanceof CollaborationProjectionBarrierError) {
      throw new Error(collaborationProjectionPendingMessage(locale), { cause: error })
    }
    throw error
  }
  authorityGuard?.assertCurrent()
  const projectedMarkdown = serializeEditorJson(
    parseEditorMarkdown(projection.markdown),
    'final',
  )
  const liveMarkdown = serializeEditorJson(editor.getJSON(), 'final')
  if (projectedMarkdown !== liveMarkdown) {
    throw new Error(collaborationProjectionPendingMessage(locale))
  }
  return {
    markdown: projection.markdown,
    sequence: projection.sequence,
  }
}

export function beginEditorCollaborationAuthorityGuard(
  document: EditorDocumentRecord,
  collaboration: CollaborationDocumentHandle,
  requirement: CollaborationAuthorityRequirement,
  locale: 'de' | 'en',
): CollaborationAuthorityGuard | null {
  if (document.contentMode !== 'collaboration') return null
  const identity = collaborationIdentity(document)
  if (!identity) throw new Error(collaborationPublishForbiddenMessage(locale))
  return beginCollaborationAuthorityGuard(collaboration, identity, requirement, locale)
}

export function useEditorSuggestions({
  activeDocument,
  activeEditor,
  apiKey,
  attachedCommentIds,
  attachedRefs,
  capabilities,
  collaboration,
  comments,
  dispatch,
  ensureAssetBodiesLoaded,
  locale,
  onCollaborationPublication,
  onGlobalSuccess,
  selectedModelTier,
  state,
}: UseEditorSuggestionsArgs): EditorSuggestionController {
  const suggestionProducer = useMemo<SuggestionProducer>(
    () => new LlmSuggestionProducer({
      apiKey,
      locale,
      stack: state.ui.selectedStack,
      workspaceId: state.workspaceId,
    }),
    [apiKey, locale, state.ui.selectedStack, state.workspaceId],
  )
  // Read the latest derived abort via a ref so run-start handlers pick up the
  // current value without threading it through every useCallback dependency.
  // The fallback path's visibility (No Silent Fallbacks) is surfaced centrally
  // in ResearchDesk via isMissingServerTimeouts, not per hook.
  const editorAbortMsRef = useRef(deriveEditorAbortMs(capabilities))
  editorAbortMsRef.current = deriveEditorAbortMs(capabilities)
  // The reference docs + chat-rule snippet an AI run carries. Both read each
  // attached file's extractedText, which on a server-synced project may be
  // hydrated empty (M6c load-on-use). resolveRunContext fetches those bodies
  // ON USE (deduped, via ensureAssetBodiesLoaded) and rebuilds both with the
  // fetched bodies overriding the stale state snapshot, so an AI run never
  // sends an empty attachment. Called at the start of each run handler (inside
  // its try, so a load failure surfaces on that run's error channel). When
  // ensureAssetBodiesLoaded is absent (offline / not synced) it is a no-op
  // pass-through and the bodies come straight from state, as before.
  const buildRuleSnippet = (bodies?: ReadonlyMap<string, string>): string =>
    attachedRefs
      .filter((ref) => ref.kind === 'chat-rule')
      .map((ref) => {
        if (ref.kind !== 'chat-rule') return ''
        const rule = state.chatRules[ref.ruleId]
        return rule
          ? renderChatRuleAttachmentContent(state, rule, new Date().toISOString(), bodies)
          : ''
      })
      .filter(Boolean)
      .join('\n\n')

  const resolveRunContext = async (
    authorityGuard: CollaborationAuthorityGuard | null = null,
  ) => {
    authorityGuard?.assertCurrent()
    const metadataReadiness = attachmentContextReadiness(state, attachedRefs)
    if (metadataReadiness.status !== 'ready') {
      throw new Error(
        metadataReadiness.error
        ?? (
          metadataReadiness.status === 'pending'
            ? locale === 'de'
              ? 'Die angehängte Datei wird noch hochgeladen oder verarbeitet.'
              : 'The attached file is still uploading or being processed.'
            : locale === 'de'
              ? 'Die angehängte Datei ist nicht als verwendbare Serverquelle verfügbar.'
              : 'The attached file is not available as a usable server source.'
        ),
      )
    }
    const bodies = ensureAssetBodiesLoaded
      ? await ensureAssetBodiesLoaded(assetIdsFromChatRefs(state, attachedRefs))
      : undefined
    authorityGuard?.assertCurrent()
    const contentReadiness = attachmentContextReadiness(state, attachedRefs, {
      assetBodyOverride: bodies,
      requireContent: true,
    })
    if (contentReadiness.status !== 'ready') {
      throw new Error(
        contentReadiness.error
        ?? (
          locale === 'de'
            ? 'Die angehängte Datei enthält keinen verwendbaren extrahierten Inhalt.'
            : 'The attached file contains no usable extracted content.'
        ),
      )
    }
    return {
      attachments: referenceDocsFromRefs(state, attachedRefs, bodies),
      ruleSnippet: buildRuleSnippet(bodies),
    }
  }

  const resolveDocumentContext = async (
    signal?: AbortSignal,
    authorityGuard: CollaborationAuthorityGuard | null = null,
  ) => {
    if (!activeDocument) throw new Error(collaborationEditorUnavailableMessage(locale))
    authorityGuard?.assertCurrent()
    const context = await resolveEditorAiDocumentContext(
      activeDocument,
      activeEditor,
      collaborationProjectionController(collaboration),
      { apiKey, signal, workspaceId: state.workspaceId },
      locale,
      flushCollaborationProjectionBarrier,
      authorityGuard,
    )
    authorityGuard?.assertCurrent()
    return context
  }

  const [commentRuns, setCommentRuns] = useState<EditorRunStateMap>({})
  const [suggestionRuns, setSuggestionRuns] = useState<EditorRunStateMap>({})
  const [savingCommentDraftIds, setSavingCommentDraftIds] = useState<string[]>([])
  const runningCommentIds = useMemo(() => runningIdsOf(commentRuns), [commentRuns])
  const runningSuggestionIds = useMemo(() => runningIdsOf(suggestionRuns), [suggestionRuns])
  const runErrors = useMemo(() => runErrorsOf(commentRuns), [commentRuns])
  const suggestionErrors = useMemo(() => runErrorsOf(suggestionRuns), [suggestionRuns])
  const [isGlobalRunning, setIsGlobalRunning] = useState(false)
  const [instructionFeedback, setInstructionFeedback] = useState<EditorInstructionFeedback | null>(null)
  const runAbortRef = useRef<AbortController | null>(null)
  const editorRunInFlightRef = useRef(false)
  const publicationInFlightRef = useRef(new Set<string>())
  const publicationAttemptRef = useRef(
    new Map<string, PrivateSuggestionPublicationAttempt>(),
  )
  const mountedRef = useRef(true)
  const selectedModelTierRef = useRef(selectedModelTier)
  const selectedModelRef = useRef(state.ui.selectedChatModel)
  const selectedEffortRef = useRef(state.ui.selectedChatEffort)

  async function persistPrivateCommentSuggestion(
    comment: EditorCommentThreadRecord,
    suggestion: EditorSuggestionRecord,
    authorityGuard: CollaborationAuthorityGuard | null,
    revisionInstruction?: string,
    revisionSource: EditorSuggestionDraftRevisionRequestWire['revision_source'] = 'llm_refine',
  ) {
    if (!activeDocument || activeDocument.contentMode !== 'collaboration') {
      throw new Error(collaborationEditorUnavailableMessage(locale))
    }
    const expectedRevision = comment.suggestionDraft?.revision ?? 0
    setSavingCommentDraftIds((ids) => (
      ids.includes(comment.id) ? ids : [...ids, comment.id]
    ))
    try {
      authorityGuard?.assertCurrent()
      const response = await saveEditorCommentSuggestionDraft(
        activeDocument.id,
        comment.id,
        {
          draft: expectedRevision === 0
            ? privateSuggestionDraftCreatePayload(suggestion)
            : privateSuggestionDraftRevisionPayload(
                suggestion,
                revisionSource,
                revisionInstruction,
              ),
          expected_revision: expectedRevision,
        },
        { apiKey, workspaceId: state.workspaceId },
      )
      authorityGuard?.assertCurrent()
      return privateSuggestionDraftRecordFromServer(response.suggestion_draft)
    } finally {
      setSavingCommentDraftIds((ids) => ids.filter((id) => id !== comment.id))
    }
  }

  /** Die Traegerzeilen des VORIGEN Anweisungslaufs abraeumen.
   *
   * Der Reducer setzt die Vorschlaege eines abgeloesten Laufs nur lokal auf
   * verworfen -- er ist synchron und kann den Server nicht rufen. Ohne diesen
   * Schritt blieben Traegerzeile und Entwurf dauerhaft liegen: sichtbare
   * Kommentar-Markierungen im Fliesstext, die zu keiner Notiz gehoeren und
   * die niemand entfernen kann, und eine Veroeffentlichungs-Autorisierung,
   * die einen laengst verworfenen Vorschlag nach dem Neuladen zurueckholt.
   *
   * Der Entwurf zuerst, dann die Zeile: der Loeschpfad des Entwurfs prueft
   * `patch_id` und Revision und ist damit die eigentliche Ruecknahme der
   * Autorisierung. Ein Fehlschlag hier darf den neuen Lauf nicht verhindern
   * -- er wird gemeldet, nicht verschluckt. */
  async function retirePreviousInstructionCarriers(
    authorityGuard: CollaborationAuthorityGuard | null,
  ): Promise<void> {
    if (!activeDocument) return
    const veraltet = documentSuggestions.filter((suggestion) =>
      suggestion.origin.kind === 'assistant_edit'
      && (suggestion.status === 'pending' || suggestion.status === 'stale'))
    for (const suggestion of veraltet) {
      const commentId = suggestion.origin.commentId
      if (!commentId) continue
      try {
        authorityGuard?.assertCurrent()
        if (suggestion.privateDraft) {
          await deleteEditorCommentSuggestionDraft(
            activeDocument.id,
            commentId,
            {
              expected_revision: suggestion.privateDraft.revision,
              patch_id: suggestion.privateDraft.patchId,
            },
            { apiKey, workspaceId: state.workspaceId },
          )
        }
        authorityGuard?.assertCurrent()
        await deleteEditorComment(
          activeDocument.id,
          commentId,
          { apiKey, workspaceId: state.workspaceId },
        )
        dispatch({ commentId, type: 'deleteEditorComment' })
      } catch (error) {
        if (error instanceof CollaborationAuthorityError) throw error
        // Eine liegengebliebene Zeile ist ein Schoenheitsfehler, ein
        // abgebrochener Lauf waere ein Funktionsverlust. Sichtbar bleibt es
        // trotzdem: die Konsole nennt Dokument und Zeile.
        console.warn('assistant_edit carrier cleanup failed', {
          commentId,
          documentId: activeDocument.id,
          reason: messageFromError(error),
        })
      }
    }
  }

  /** Jedem Assistenten-Edit seine Vorautorisierung geben.
   *
   * Der Serverwaechter laesst eine Veroeffentlichung mit `actor_kind:
   * 'assistant'` nur zu, wenn ein creator-privater Entwurf genau dieses
   * `patch_id`/`command_id` vorher autorisiert hat, und ein Entwurf kann nur
   * an einer Kommentarzeile haengen. Ein Assistentenlauf hat keinen Kommentar
   * des Nutzers, also legt er je Edit eine eigene Traegerzeile an.
   *
   * Die Reihenfolge ist zwingend: erst die Zeile, dann der Entwurf -- der
   * Server heftet einen Entwurf nur an einen Kommentar, den der Aufrufer
   * selbst erstellt hat. Scheitert ein Edit, scheitert nur dieser: die
   * bereits autorisierten bleiben gueltig, statt den ganzen Lauf zu
   * verwerfen. Was nicht autorisiert werden konnte, wird gar nicht erst als
   * Vorschlag angezeigt -- ein Knopf, der sicher 409 liefert, ist schlimmer
   * als ein Vorschlag, den es nicht gibt. */
  async function persistInstructionCarriers(
    suggestions: readonly EditorSuggestionRecord[],
    instruction: string,
    now: string,
    authorityGuard: CollaborationAuthorityGuard | null,
  ): Promise<EditorSuggestionRecord[]> {
    if (!activeDocument) return []
    await retirePreviousInstructionCarriers(authorityGuard)
    const authorized: EditorSuggestionRecord[] = []
    for (const suggestion of suggestions) {
      const carrier: EditorCommentThreadRecord = {
        anchor: suggestion.anchor,
        commentMarkdown: instruction,
        createdAt: now,
        documentId: activeDocument.id,
        id: createLocalId('editor-comment'),
        kind: 'assistant_edit',
        status: 'open',
        updatedAt: now,
      }
      try {
        authorityGuard?.assertCurrent()
        await saveEditorComments(
          activeDocument.id,
          [serverCommentPayload(carrier)],
          { apiKey, workspaceId: state.workspaceId },
        )
        const privateSuggestion = preparePrivateSuggestionRecord(
          suggestion,
          undefined,
          locale,
        )
        const draft = await persistPrivateCommentSuggestion(
          carrier,
          privateSuggestion,
          authorityGuard,
        )
        // Der Traeger muss in den Zustand: "Verfeinern" und "Bearbeiten"
        // schlagen ihren Kommentar ueber origin.commentId nach und wuerfen
        // sonst eine irrefuehrende "Text hat sich geaendert"-Meldung.
        dispatch({ comment: carrier, type: 'adoptEditorCarrierComment' })
        authorized.push({
          ...privateSuggestion,
          origin: { commentId: carrier.id, kind: 'assistant_edit' },
          privateDraft: {
            patchId: draft.patchId,
            publicationCommandId: draft.publicationCommandId,
            revision: draft.revision,
          },
        })
      } catch (error) {
        // Ein Abbruch der Sitzung ist kein Teilfehler eines Edits.
        if (error instanceof CollaborationAuthorityError) throw error
        // Sonst gilt: dieser eine Edit konnte nicht autorisiert werden,
        // die uebrigen bleiben gueltig. Er wird gar nicht erst als
        // Vorschlag angezeigt -- ein Knopf, der sicher 409 liefert, ist
        // schlimmer als ein Vorschlag, den es nicht gibt. Wie viele
        // ausgefallen sind, sagt der Lauf dem Nutzer ausdruecklich --
        // gezaehlt wird das beim Aufrufer als Differenz zur Zahl der
        // vorgeschlagenen Edits.
      }
    }
    return authorized
  }

  useEffect(() => {
    selectedModelTierRef.current = selectedModelTier
    selectedModelRef.current = state.ui.selectedChatModel
    selectedEffortRef.current = state.ui.selectedChatEffort
  }, [selectedModelTier, state.ui.selectedChatModel, state.ui.selectedChatEffort])

  useEffect(() => {
    mountedRef.current = true
    // A view switch unmounts the editor mid-run. We deliberately do NOT abort the
    // in-flight request here: the run finishes and its result dispatches into the
    // project reducer (owned by the still-mounted parent). Legacy results may use
    // quote fallback after unmount; collaboration results fail visibly when a
    // required relative anchor can no longer be encoded. The client-side timeout
    // still bounds the request.
    return () => {
      mountedRef.current = false
    }
  }, [])

  const documentSuggestions = useMemo(
    () => Object.values(state.editorSuggestions).filter((suggestion) => suggestion.documentId === activeDocument?.id),
    [activeDocument?.id, state.editorSuggestions],
  )
  const documentSuggestionsRef = useRef(documentSuggestions)
  useEffect(() => {
    documentSuggestionsRef.current = documentSuggestions
  }, [documentSuggestions])

  // P7: pending AGENT patches of the open markdown document become
  // tracked-change suggestions on open/reload — the server rows were
  // previously invisible after the tool approval. Collaboration
  // documents keep their own shared-changes review surface. Fetched
  // once per document focus; already-mirrored patch ids (any status,
  // this session) are skipped so re-opening never duplicates groups.
  useEffect(() => {
    const document = activeDocument
    if (!document || document.contentMode === 'collaboration' || !activeEditor) {
      return
    }
    let cancelled = false
    const options = { apiKey, workspaceId: state.workspaceId }
    void (async () => {
      let pendingRows: Array<{ patch_id: string; source: string }>
      try {
        pendingRows = (
          await listEditorDocumentPatches(document.id, 'pending', options)
        ).data
      } catch (error) {
        // 404 = deployment without the editor-patch service (router is
        // conditional) — nothing to mirror. Anything else: warn loudly;
        // the rows stay pending server-side and reload on the next open,
        // so no decision is lost.
        if (!hasHttpStatus(error, 404)) {
          console.warn('Editor-Patch-Abruf fehlgeschlagen:', error)
        }
        return
      }
      const known = new Set(
        documentSuggestionsRef.current.flatMap((item) =>
          item.serverPatch ? [item.serverPatch.patchId] : [],
        ),
      )
      for (const row of pendingRows) {
        if (row.source !== 'agent' || known.has(row.patch_id)) continue
        let patch: AgentPatchWire
        try {
          patch = await getEditorPatch(row.patch_id, options)
        } catch (error) {
          console.warn('Editor-Patch-Detail fehlgeschlagen:', error)
          continue
        }
        if (cancelled || patch.status !== 'pending') continue
        const now = new Date().toISOString()
        const groupId = createLocalId('editor-suggestion-group')
        const records = createAgentPatchSuggestionRecords({
          activeEditor: mountedRef.current ? activeEditor : null,
          document,
          groupId,
          locale,
          now,
          patch,
        })
        if (cancelled || records.length === 0) continue
        dispatch({
          group: {
            assistantMessage: patch.summary,
            createdAt: now,
            documentId: document.id,
            id: groupId,
            origin: { kind: 'global_run' },
            warnings: patch.warnings.length ? patch.warnings : undefined,
          },
          suggestions: records,
          type: 'createEditorSuggestionGroup',
        })
      }
    })()
    return () => {
      cancelled = true
    }
    // Deliberately narrow deps: fetch once per document focus;
    // suggestion state is read through the ref above.
  }, [activeDocument?.id, activeDocument?.contentMode, activeEditor])
  const aiReadOnlyReason = editorCollaborationActionDisabledReason(
    activeDocument,
    collaboration,
    'write',
    locale,
  )
  const suggestionPublishDisabledReason = privateSuggestionPublishDisabledReason(
    activeDocument,
    collaboration,
    locale,
  )

  async function handleRunComment(comment: EditorCommentThreadRecord) {
    if (!activeDocument || !tryAcquireEditorRunLatch(editorRunInFlightRef)) return
    try {
      await executeCommentRun(comment)
    } finally {
      editorRunInFlightRef.current = false
    }
  }

  async function executeCommentRun(comment: EditorCommentThreadRecord) {
    if (!activeDocument) return
    let authorityGuard: CollaborationAuthorityGuard | null
    try {
      authorityGuard = beginEditorCollaborationAuthorityGuard(
        activeDocument,
        collaboration,
        'write',
        locale,
      )
    } catch (error) {
      setCommentRuns((map) => markError(map, comment.id, messageFromError(error)))
      return
    }
    const requireRelative = activeDocument.contentMode === 'collaboration'
    const origin: EditorSuggestionOrigin = comment.kind === 'evidence_review'
      ? { commentId: comment.id, kind: 'evidence_review', preset: comment.evidencePreset ?? 'add_sources' }
      : { commentId: comment.id, kind: 'inline_edit' }
    dispatch({ commentId: comment.id, type: 'selectEditorComment' })
    runAbortRef.current?.abort()
    const controller = new AbortController()
    runAbortRef.current = controller
    setCommentRuns((map) => markRunning(map, comment.id))
    const clearRunTimeout = startEditorRunTimeout(editorAbortMsRef.current, () => {
      controller.abort()
      setCommentRuns((map) => markError(map, comment.id, editorTimeoutMessage(locale)))
    })
    try {
      const { attachments } = await resolveRunContext(authorityGuard)
      const documentContext = await resolveDocumentContext(controller.signal, authorityGuard)
      const materialized = mountedRef.current && activeEditor
        ? materializePrivateSuggestionComment(activeEditor, comment, undefined, requireRelative)
        : {
            anchor: {
              anchor: comment.anchor,
              ...(requireRelative ? { reason: 'adapter_unavailable' as const } : {}),
              status: requireRelative ? 'failed' as const : 'legacy' as const,
            },
            comment: requireRelative ? null : comment,
          }
      if (requireRelative && materialized.anchor.status !== 'relative') {
        throw new Error(requiredRelativeAnchorMessage(locale))
      }
      const liveComment = materialized.comment
      if (!liveComment) {
        authorityGuard?.assertCurrent()
        dispatch({ commentId: comment.id, status: 'stale', type: 'setEditorCommentStatus' })
        throw new Error(staleAnchorMessage(locale))
      }
      if (requireRelative) {
        authorityGuard?.assertCurrent()
        await saveEditorComments(
          activeDocument.id,
          [serverCommentPayload(liveComment)],
          { apiKey, workspaceId: state.workspaceId },
        )
        authorityGuard?.assertCurrent()
      }
      const modelTier = selectedModelTierRef.current
      const model = selectedModelRef.current
      const effort = selectedEffortRef.current
      const proposal = await invokeEditorAiProvider(
        activeDocument,
        collaboration.access,
        locale,
        () => suggestionProducer.produce({
        anchor: liveComment.anchor,
        attachments,
        documentId: liveComment.documentId,
        documentMarkdown: documentContext.markdown,
        instruction: liveComment.commentMarkdown,
        modelTier,
        model,
        effort,
        origin,
        originalMarkdown: liveComment.anchor.selectedMarkdown,
        originalText: liveComment.anchor.selectedText,
        signal: controller.signal,
        }),
        authorityGuard,
      )
      const now = new Date().toISOString()
      const groupId = comment.suggestionDraft?.groupId
        ?? createLocalId('editor-suggestion-group')
      const group: EditorSuggestionGroupRecord = { createdAt: now, documentId: comment.documentId, id: groupId, origin }
      authorityGuard?.assertCurrent()
      const suggestion = createSuggestionRecord({
        comment: liveComment,
        groupId,
        now,
        origin,
        proposal,
      })
      if (requireRelative) {
        const privateSuggestion = preparePrivateSuggestionRecord(
          suggestion,
          comment.suggestionDraft,
          locale,
        )
        const draft = await persistPrivateCommentSuggestion(
          comment,
          privateSuggestion,
          authorityGuard,
          comment.commentMarkdown,
        )
        authorityGuard?.assertCurrent()
        dispatch({
          anchor: liveComment.anchor,
          commentId: comment.id,
          suggestionDraft: draft,
          type: 'adoptEditorCommentSuggestionDraft',
        })
      } else {
        dispatch({
          group,
          suggestions: [suggestion],
          type: 'createEditorSuggestionGroup',
        })
      }
    } catch (error) {
      if (controller.signal.aborted) return
      setCommentRuns((map) => markError(map, comment.id, messageFromError(error)))
    } finally {
      clearRunTimeout()
      setCommentRuns((map) => clearRunning(map, [comment.id]))
    }
  }

  async function handleAcceptSuggestion(suggestion: EditorSuggestionRecord) {
    if (suggestion.serverPatch) {
      // P7: a server-side agent patch is ONE unit — accepting any of its
      // edits applies the whole patch through the official endpoint
      // (audit-true status, CAS against the current revision), never the
      // client-side edit path.
      await decideServerPatch(suggestion, 'accept')
      return
    }
    if (activeDocument?.contentMode === 'collaboration') {
      await publishCollaborationSuggestionBatch([suggestion])
      return
    }
    acceptLegacySuggestion(suggestion)
  }

  async function decideServerPatch(
    suggestion: EditorSuggestionRecord,
    decision: 'accept' | 'reject',
  ): Promise<void> {
    const patchId = suggestion.serverPatch?.patchId
    const document = activeDocument
    if (!patchId || !document) return
    const siblings = documentSuggestions.filter(
      (item) =>
        item.serverPatch?.patchId === patchId && item.status === 'pending',
    )
    const siblingIds = siblings.map((item) => item.id)
    if (siblingIds.length === 0) return
    setSuggestionRuns((map) => {
      let next = map
      for (const id of siblingIds) next = markRunning(next, id)
      return next
    })
    try {
      if (decision === 'accept') {
        const contentBefore = document.contentMarkdown
        await applyEditorPatch(patchId, document.revision, {
          apiKey,
          workspaceId: state.workspaceId,
        })
        const server = await getEditorDocument(document.id, {
          apiKey,
          workspaceId: state.workspaceId,
        })
        // Adopt the server-applied body via the SAME rebase the autosave
        // 409 path uses: keystrokes typed during the window win locally
        // and re-push as base+1 — nothing is silently discarded.
        dispatch({
          contentMarkdown: server.content_markdown ?? '',
          documentId: document.id,
          pushedContentMarkdown: contentBefore,
          revision: server.revision,
          type: 'rebaseServerEditorDocument',
        })
        for (const id of siblingIds) {
          dispatch({ suggestionId: id, type: 'acceptEditorSuggestion' })
        }
      } else {
        await rejectEditorPatch(patchId, '', {
          apiKey,
          workspaceId: state.workspaceId,
        })
        for (const id of siblingIds) {
          dispatch({ suggestionId: id, type: 'rejectEditorSuggestion' })
        }
      }
    } catch (error) {
      // Visible on every card of the patch (revision conflict, network):
      // the rows stay pending server-side, the user can retry.
      const message = messageFromError(error)
      setSuggestionRuns((map) => {
        let next = map
        for (const id of siblingIds) next = markError(next, id, message)
        return next
      })
      return
    } finally {
      setSuggestionRuns((map) => clearRunning(map, siblingIds))
    }
  }

  async function handleAcceptSuggestionGroup(groupId: string) {
    const anchorAdapter = activeEditor
      ? privateSuggestionAnchorAdapter(activeEditor)
      : null
    const groupSuggestions = sortPrivateSuggestionGroup(
      documentSuggestions.filter(
        (suggestion) => suggestion.groupId === groupId && suggestion.status === 'pending',
      ),
      anchorAdapter,
    )
    if (activeDocument?.contentMode === 'collaboration') {
      for (const suggestion of groupSuggestions) {
        await publishCollaborationSuggestionBatch([suggestion])
      }
      return
    }
    for (const suggestion of groupSuggestions) {
      acceptLegacySuggestion(suggestion)
    }
  }

  function acceptLegacySuggestion(suggestion: EditorSuggestionRecord): void {
    if (!activeEditor || !applySuggestionToEditor(activeEditor, suggestion)) {
      dispatch({ suggestionId: suggestion.id, type: 'markEditorSuggestionStale' })
      return
    }
    dispatch({ suggestionId: suggestion.id, type: 'acceptEditorSuggestion' })
  }

  async function publishCollaborationSuggestionBatch(
    suggestions: readonly EditorSuggestionRecord[],
  ): Promise<void> {
    if (!activeDocument || activeDocument.contentMode !== 'collaboration' || suggestions.length === 0) {
      return
    }
    const suggestionIds = suggestions.map((suggestion) => suggestion.id)
    if (suggestions.some((suggestion) => (
      suggestion.documentId !== activeDocument.id || suggestion.status !== 'pending'
    ))) return
    if (!activeEditor) {
      setSuggestionRuns((map) => markErrors(
        map,
        Object.fromEntries(suggestionIds.map((id) => [id, collaborationEditorUnavailableMessage(locale)])),
      ))
      return
    }
    let authorityGuard: CollaborationAuthorityGuard | null
    try {
      authorityGuard = beginEditorCollaborationAuthorityGuard(
        activeDocument,
        collaboration,
        'write',
        locale,
      )
    } catch (error) {
      const publicationDisabledReason = messageFromError(error)
      setSuggestionRuns((map) => markErrors(
        map,
        Object.fromEntries(suggestionIds.map((id) => [id, publicationDisabledReason])),
      ))
      return
    }

    if (suggestionIds.some((id) => publicationInFlightRef.current.has(id))) return
    for (const id of suggestionIds) publicationInFlightRef.current.add(id)

    setSuggestionRuns((map) => markManyRunning(map, suggestionIds))
    try {
      const suggestion = suggestions[0]
      const publication = await invokePrivateSuggestionPublication(
        activeDocument,
        collaboration,
        locale,
        async () => {
          let attempt = publicationAttemptRef.current.get(suggestion.id)
          if (!attempt) {
            const documentContext = await resolveDocumentContext(undefined, authorityGuard)
            if (documentContext.sequence === null) {
              throw new Error(collaborationProjectionInvalidMessage(locale))
            }
            const prepared = prepareCollaborationSuggestionBatch(
              activeEditor,
              [suggestion],
              privateSuggestionAnchorAdapter(activeEditor),
              locale,
            )
            if (Object.keys(prepared.errors).length > 0) {
              authorityGuard?.assertCurrent()
              setSuggestionRuns((map) => markErrors(map, prepared.errors))
              return null
            }
            const identity = privateSuggestionPublicationIdentity(suggestion, locale)
            attempt = {
              ...identity,
              expectedSequence: documentContext.sequence,
              targetMarkdown: buildCollaborationSuggestionTargetMarkdown(
                documentContext.markdown,
                prepared.suggestions,
                locale,
              ),
            }
            publicationAttemptRef.current.set(suggestion.id, attempt)
          }
          authorityGuard?.assertCurrent()
          const response = await publishEditorCollaborationSuggestion(
            activeDocument.id,
            {
              actor_kind: 'assistant',
              command_id: attempt.commandId,
              expected_sequence: attempt.expectedSequence,
              patch_id: attempt.patchId,
              target_markdown: attempt.targetMarkdown,
            },
            { apiKey, workspaceId: state.workspaceId },
          )
          authorityGuard?.assertCurrent()
          return collaborationPublicationFromResponse(
            response,
            attempt,
            locale,
          )
        },
        authorityGuard,
      )
      if (!publication) return
      publicationAttemptRef.current.delete(suggestion.id)
      authorityGuard?.assertCurrent()
      setAuthoritativeCollaborationSequence(
        collaborationProjectionController(collaboration),
        publication.sequence,
      )
      authorityGuard?.assertCurrent()
      onCollaborationPublication?.(activeDocument.id, publication)
      for (const suggestion of suggestions) {
        authorityGuard?.assertCurrent()
        dispatch({
          collaborationPublication: publication,
          suggestionId: suggestion.id,
          type: 'acceptEditorSuggestion',
        })
      }
    } catch (error) {
      if ([400, 401, 403, 404, 409, 413].some((status) => hasHttpStatus(error, status))) {
        for (const id of suggestionIds) publicationAttemptRef.current.delete(id)
      }
      const message = messageFromError(error)
      setSuggestionRuns((map) => markErrors(
        map,
        Object.fromEntries(suggestionIds.map((id) => [id, message])),
      ))
    } finally {
      for (const id of suggestionIds) publicationInFlightRef.current.delete(id)
      setSuggestionRuns((map) => clearRunning(map, suggestionIds))
    }
  }

  async function handleRejectSuggestion(suggestionId: string) {
    const suggestion = documentSuggestions.find((item) => item.id === suggestionId)
    if (!suggestion || suggestion.status !== 'pending') return
    if (suggestion.serverPatch) {
      await decideServerPatch(suggestion, 'reject')
      return
    }
    if (activeDocument?.contentMode === 'collaboration') {
      let authorityGuard: CollaborationAuthorityGuard | null
      try {
        authorityGuard = beginEditorCollaborationAuthorityGuard(
          activeDocument,
          collaboration,
          'write',
          locale,
        )
      } catch (error) {
        setSuggestionRuns((map) => markError(map, suggestionId, messageFromError(error)))
        return
      }
      const commentId = suggestion.origin.commentId
      if (suggestion.privateDraft && commentId) {
        setSuggestionRuns((map) => markRunning(map, suggestionId))
        try {
          authorityGuard?.assertCurrent()
          await deleteEditorCommentSuggestionDraft(
            activeDocument.id,
            commentId,
            {
              expected_revision: suggestion.privateDraft.revision,
              patch_id: suggestion.privateDraft.patchId,
            },
            { apiKey, workspaceId: state.workspaceId },
          )
          authorityGuard?.assertCurrent()
        } catch (error) {
          setSuggestionRuns((map) => markError(map, suggestionId, messageFromError(error)))
          return
        } finally {
          setSuggestionRuns((map) => clearRunning(map, [suggestionId]))
        }
      }
    }
    dispatch({ suggestionId, type: 'rejectEditorSuggestion' })
  }

  async function handleRejectSuggestionGroup(groupId: string) {
    const groupSuggestions = documentSuggestions
      .filter((suggestion) => suggestion.groupId === groupId && suggestion.status === 'pending')
    if (activeDocument?.contentMode === 'collaboration') {
      for (const suggestion of groupSuggestions) {
        await handleRejectSuggestion(suggestion.id)
      }
      return
    }
    dispatch({ groupId, type: 'rejectEditorSuggestionGroup' })
  }

  async function handleEditSuggestionProposal(suggestionId: string, proposedText: string) {
    const suggestion = documentSuggestions.find((item) => item.id === suggestionId)
    if (!suggestion || suggestion.status !== 'pending' || !proposedText.trim()) return
    const updatedSuggestion = { ...suggestion, proposedText }
    const commentId = suggestion.origin.commentId
    if (
      activeDocument?.contentMode === 'collaboration'
      && suggestion.privateDraft
      && commentId
    ) {
      let authorityGuard: CollaborationAuthorityGuard | null
      try {
        authorityGuard = beginEditorCollaborationAuthorityGuard(
          activeDocument,
          collaboration,
          'write',
          locale,
        )
        const comment = state.editorComments[commentId]
        if (!comment) throw new Error(staleAnchorMessage(locale))
        setSuggestionRuns((map) => markRunning(map, suggestionId))
        const draft = await persistPrivateCommentSuggestion(
          comment,
          updatedSuggestion,
          authorityGuard,
          undefined,
          'manual_edit',
        )
        authorityGuard?.assertCurrent()
        dispatch({
          anchor: comment.anchor,
          commentId,
          suggestionDraft: draft,
          type: 'adoptEditorCommentSuggestionDraft',
        })
        setSuggestionRuns((map) => clearRuns(map, [suggestionId]))
        return
      } catch (error) {
        setSuggestionRuns((map) => markError(map, suggestionId, messageFromError(error)))
        return
      } finally {
        setSuggestionRuns((map) => clearRunning(map, [suggestionId]))
      }
    }
    dispatch({
      proposedText,
      source: 'manual_edit',
      suggestionId,
      type: 'updateEditorSuggestionProposal',
    })
    setSuggestionRuns((map) => clearRuns(map, [suggestionId]))
  }

  async function handleRefineSuggestion(suggestionId: string, instruction: string) {
    if (!activeDocument || isGlobalRunning || runningSuggestionIds.includes(suggestionId)) return
    const trimmedInstruction = instruction.trim()
    const suggestion = documentSuggestions.find((item) => item.id === suggestionId)
    if (!suggestion || suggestion.status !== 'pending' || !trimmedInstruction) return
    if (!tryAcquireEditorRunLatch(editorRunInFlightRef)) return
    try {
      await executeRefineSuggestion(suggestion, trimmedInstruction)
    } finally {
      editorRunInFlightRef.current = false
    }
  }

  async function executeRefineSuggestion(
    suggestion: EditorSuggestionRecord,
    trimmedInstruction: string,
  ) {
    if (!activeDocument) return
    const suggestionId = suggestion.id
    let authorityGuard: CollaborationAuthorityGuard | null
    try {
      authorityGuard = beginEditorCollaborationAuthorityGuard(
        activeDocument,
        collaboration,
        'write',
        locale,
      )
    } catch (error) {
      setSuggestionRuns((map) => markError(map, suggestionId, messageFromError(error)))
      return
    }
    runAbortRef.current?.abort()
    const controller = new AbortController()
    runAbortRef.current = controller
    setSuggestionRuns((map) => markRunning(map, suggestionId))
    const clearRunTimeout = startEditorRunTimeout(editorAbortMsRef.current, () => {
      controller.abort()
      setSuggestionRuns((map) => markError(map, suggestionId, editorTimeoutMessage(locale)))
    })
    try {
      const { attachments } = await resolveRunContext(authorityGuard)
      const documentContext = await resolveDocumentContext(controller.signal, authorityGuard)
      const modelTier = selectedModelTierRef.current
      const model = selectedModelRef.current
      const effort = selectedEffortRef.current
      const originalInstruction = suggestion.origin.commentId
        ? state.editorComments[suggestion.origin.commentId]?.commentMarkdown
        : undefined
      const proposal = await invokeEditorAiProvider(
        activeDocument,
        collaboration.access,
        locale,
        () => suggestionProducer.refine({
        attachments,
        documentMarkdown: documentContext.markdown,
        instruction: trimmedInstruction,
        modelTier,
        model,
        effort,
        originalInstruction,
        signal: controller.signal,
        suggestion,
        }),
        authorityGuard,
      )
      if (controller.signal.aborted) return
      authorityGuard?.assertCurrent()
      const revisedSuggestion: EditorSuggestionRecord = {
        ...suggestion,
        changeSummary: proposal.changeSummary,
        proposedText: proposal.proposedText,
        warnings: proposal.warnings,
      }
      const commentId = suggestion.origin.commentId
      if (
        activeDocument.contentMode === 'collaboration'
        && suggestion.privateDraft
        && commentId
      ) {
        const comment = state.editorComments[commentId]
        if (!comment) throw new Error(staleAnchorMessage(locale))
        const draft = await persistPrivateCommentSuggestion(
          comment,
          revisedSuggestion,
          authorityGuard,
          trimmedInstruction,
        )
        authorityGuard?.assertCurrent()
        dispatch({
          anchor: comment.anchor,
          commentId,
          suggestionDraft: draft,
          type: 'adoptEditorCommentSuggestionDraft',
        })
        return
      }
      dispatch({
        changeSummary: proposal.changeSummary,
        instruction: trimmedInstruction,
        proposedText: proposal.proposedText,
        source: 'llm_refine',
        suggestionId,
        type: 'updateEditorSuggestionProposal',
        warnings: proposal.warnings,
      })
    } catch (error) {
      if (controller.signal.aborted) return
      setSuggestionRuns((map) => markError(map, suggestionId, messageFromError(error)))
    } finally {
      clearRunTimeout()
      setSuggestionRuns((map) => clearRunning(map, [suggestionId]))
    }
  }

  function handleMarkSuggestionStale(suggestionId: string) {
    dispatch({ suggestionId, type: 'markEditorSuggestionStale' })
    setSuggestionRuns((map) => markError(map, suggestionId, staleAnchorMessage(locale)))
  }

  function handleStopRun() {
    runAbortRef.current?.abort()
    setIsGlobalRunning(false)
    setCommentRuns((map) => clearRunning(map, Object.keys(map)))
    setInstructionFeedback((prev) => (prev?.state === 'thinking' ? null : prev))
  }

  function handleStopSuggestionRun(suggestionId: string) {
    runAbortRef.current?.abort()
    setSuggestionRuns((map) => clearRuns(map, [suggestionId]))
  }

  function clearInstructionFeedback() {
    setInstructionFeedback(null)
  }


  async function handleGlobalRun(globalInstruction: string) {
    if (!activeDocument || isGlobalRunning) return
    const hasTargets = comments.some((comment) =>
      comment.status === 'open'
      && comment.kind === 'collect'
      && attachedCommentIds.includes(comment.id))
    if (!hasTargets || !tryAcquireEditorRunLatch(editorRunInFlightRef)) return
    try {
      await executeGlobalRun(globalInstruction)
    } finally {
      editorRunInFlightRef.current = false
    }
  }

  async function executeGlobalRun(globalInstruction: string) {
    if (!activeDocument || isGlobalRunning) return
    const targets = comments.filter((comment) =>
      comment.status === 'open' && comment.kind === 'collect' && attachedCommentIds.includes(comment.id))
    if (targets.length === 0) return
    let authorityGuard: CollaborationAuthorityGuard | null
    try {
      authorityGuard = beginEditorCollaborationAuthorityGuard(
        activeDocument,
        collaboration,
        'write',
        locale,
      )
    } catch (error) {
      const message = messageFromError(error)
      setCommentRuns((map) => markErrors(
        map,
        Object.fromEntries(targets.map((comment) => [comment.id, message])),
      ))
      setInstructionFeedback({ message, state: 'error' })
      return
    }
    const draftInstruction = globalInstruction.trim()
    runAbortRef.current?.abort()
    const controller = new AbortController()
    runAbortRef.current = controller
    setIsGlobalRunning(true)
    setCommentRuns((map) => markManyRunning(map, targets.map((comment) => comment.id)))

    const documentId = activeDocument.id
    const modelTier = selectedModelTierRef.current
    const model = selectedModelRef.current
    const effort = selectedEffortRef.current
    const now = new Date().toISOString()
    const groupId = createLocalId('editor-suggestion-group')
    const clearRunTimeout = startEditorRunTimeout(editorAbortMsRef.current, () => {
      controller.abort()
      const timeoutErrors = Object.fromEntries(targets.map((comment) => [comment.id, editorTimeoutMessage(locale)]))
      setCommentRuns((map) => markErrors(map, timeoutErrors))
      setIsGlobalRunning(false)
    })

    // Load attached file bodies once before the parallel run (M6c load-on-use);
    // on failure mark all targets errored and bail (handleGlobalRun has no
    // outer try). A no-op pass-through when assets are not server-synced.
    const runContext = await resolveRunContext(authorityGuard).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setCommentRuns((map) =>
          markErrors(map, Object.fromEntries(targets.map((c) => [c.id, messageFromError(error)]))),
        )
      }
      clearRunTimeout()
      setIsGlobalRunning(false)
      return null
    })
    if (!runContext) return
    const documentContext = await resolveDocumentContext(
      controller.signal,
      authorityGuard,
    ).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setCommentRuns((map) =>
          markErrors(map, Object.fromEntries(targets.map((c) => [c.id, messageFromError(error)]))),
        )
      }
      clearRunTimeout()
      setIsGlobalRunning(false)
      return null
    })
    if (!documentContext) return

    const anchorAdapter = mountedRef.current && activeEditor
      ? privateSuggestionAnchorAdapter(activeEditor)
      : null
    const preparedTargets = new Map(targets.map((comment) => [
      comment.id,
      mountedRef.current && activeEditor
        ? materializePrivateSuggestionComment(
            activeEditor,
            comment,
            anchorAdapter,
            activeDocument.contentMode === 'collaboration',
          )
        : {
            anchor: {
              anchor: comment.anchor,
              ...(activeDocument.contentMode === 'collaboration'
                ? { reason: 'adapter_unavailable' as const }
                : {}),
              status: activeDocument.contentMode === 'collaboration'
                ? 'failed' as const
                : 'legacy' as const,
            },
            comment: activeDocument.contentMode === 'collaboration' ? null : comment,
          },
    ]))

    const produceForComment = async (comment: EditorCommentThreadRecord) => {
      // Materialized only after the projection barrier so AI sees a target from
      // the same durable document state. A later unmount cannot invalidate the
      // encoded Yjs relative positions.
      const prepared = preparedTargets.get(comment.id)
      if (
        activeDocument.contentMode === 'collaboration'
        && prepared?.anchor.status !== 'relative'
      ) {
        throw new Error(requiredRelativeAnchorMessage(locale))
      }
      const liveComment = prepared?.comment
      if (!liveComment) throw new Error(staleAnchorMessage(locale))
      if (activeDocument.contentMode === 'collaboration') {
        authorityGuard?.assertCurrent()
        await saveEditorComments(
          activeDocument.id,
          [serverCommentPayload(liveComment)],
          { apiKey, workspaceId: state.workspaceId },
        )
        authorityGuard?.assertCurrent()
      }
      const origin: EditorSuggestionOrigin = { commentId: comment.id, kind: 'global_run' }
      const proposal = await invokeEditorAiProvider(
        activeDocument,
        collaboration.access,
        locale,
        () => suggestionProducer.produce({
        anchor: liveComment.anchor,
        attachments: runContext.attachments,
        documentId,
        documentMarkdown: documentContext.markdown,
        globalInstruction: draftInstruction || undefined,
        instruction: liveComment.commentMarkdown,
        modelTier,
        model,
        effort,
        origin,
        originalMarkdown: liveComment.anchor.selectedMarkdown,
        originalText: liveComment.anchor.selectedText,
        signal: controller.signal,
        snippet: runContext.ruleSnippet || undefined,
        }),
        authorityGuard,
      )
      const suggestion = createSuggestionRecord({
        comment: liveComment,
        documentId,
        groupId,
        now,
        origin,
        proposal,
      })
      if (activeDocument.contentMode !== 'collaboration') {
        return { comment: liveComment, origin, proposal, suggestion }
      }
      const privateSuggestion = preparePrivateSuggestionRecord(
        suggestion,
        comment.suggestionDraft,
        locale,
      )
      const suggestionDraft = await persistPrivateCommentSuggestion(
        comment,
        privateSuggestion,
        authorityGuard,
        draftInstruction || comment.commentMarkdown,
      )
      return {
        comment: liveComment,
        origin,
        proposal,
        suggestion: privateSuggestion,
        suggestionDraft,
      }
    }

    const suggestions: EditorSuggestionRecord[] = []
    const savedDrafts: Array<{
      anchor: EditorCommentThreadRecord['anchor']
      commentId: string
      suggestionDraft: NonNullable<EditorCommentThreadRecord['suggestionDraft']>
    }> = []
    const errors: Record<string, string> = {}
    const poolSize = 4
    let authorityFailure: string | null = null
    for (let index = 0; index < targets.length; index += poolSize) {
      const batch = targets.slice(index, index + poolSize)
      try {
        authorityGuard?.assertCurrent()
      } catch (error) {
        authorityFailure = messageFromError(error)
        break
      }
      const settled = await Promise.allSettled(batch.map(produceForComment))
      settled.forEach((outcome, offset) => {
        const comment = batch[offset]
        if (outcome.status === 'fulfilled') {
          suggestions.push(outcome.value.suggestion)
          if (outcome.value.suggestionDraft) {
            savedDrafts.push({
              anchor: outcome.value.comment.anchor,
              commentId: comment.id,
              suggestionDraft: outcome.value.suggestionDraft,
            })
          }
        } else if (!controller.signal.aborted) {
          errors[comment.id] = messageFromError(outcome.reason)
        }
      })
    }

    try {
      authorityGuard?.assertCurrent()
    } catch (error) {
      authorityFailure = messageFromError(error)
    }
    if (authorityFailure) {
      setCommentRuns((map) => markErrors(
        map,
        Object.fromEntries(targets.map((comment) => [comment.id, authorityFailure])),
      ))
      clearRunTimeout()
      setIsGlobalRunning(false)
      setCommentRuns((map) => clearRunning(map, targets.map((comment) => comment.id)))
      return
    }

    if (controller.signal.aborted) {
      clearRunTimeout()
      setIsGlobalRunning(false)
      setCommentRuns((map) => clearRunning(map, targets.map((comment) => comment.id)))
      return
    }
    if (suggestions.length > 0) {
      authorityGuard?.assertCurrent()
      if (activeDocument.contentMode === 'collaboration') {
        for (const saved of savedDrafts) {
          authorityGuard?.assertCurrent()
          dispatch({
            anchor: saved.anchor,
            commentId: saved.commentId,
            suggestionDraft: saved.suggestionDraft,
            type: 'adoptEditorCommentSuggestionDraft',
          })
        }
      } else {
        dispatch({
          group: {
            createdAt: now,
            documentId,
            id: groupId,
            origin: { kind: 'global_run' },
          },
          suggestions,
          type: 'createEditorSuggestionGroup',
        })
      }
      const firstCommentId = suggestions[0]?.origin.commentId
      if (firstCommentId) {
        authorityGuard?.assertCurrent()
        dispatch({ commentId: firstCommentId, type: 'selectEditorComment' })
      }
    }
    if (Object.keys(errors).length > 0) {
      setCommentRuns((map) => markErrors(map, errors))
    } else {
      authorityGuard?.assertCurrent()
      dispatch({ draft: '', type: 'setEditorAssistantDraft' })
      authorityGuard?.assertCurrent()
      onGlobalSuccess()
    }
    clearRunTimeout()
    setIsGlobalRunning(false)
    setCommentRuns((map) => clearRunning(map, targets.map((comment) => comment.id)))
  }

  async function handleInstructionRun(instruction: string) {
    if (!activeDocument || isGlobalRunning || !instruction.trim()) return
    if (!tryAcquireEditorRunLatch(editorRunInFlightRef)) return
    try {
      await executeInstructionRun(instruction)
    } finally {
      editorRunInFlightRef.current = false
    }
  }

  async function executeInstructionRun(instruction: string) {
    if (!activeDocument || isGlobalRunning) return
    const draftInstruction = instruction.trim()
    if (!draftInstruction) return
    let authorityGuard: CollaborationAuthorityGuard | null
    try {
      authorityGuard = beginEditorCollaborationAuthorityGuard(
        activeDocument,
        collaboration,
        'write',
        locale,
      )
    } catch (error) {
      setInstructionFeedback({ message: messageFromError(error), state: 'error' })
      return
    }
    runAbortRef.current?.abort()
    const controller = new AbortController()
    runAbortRef.current = controller
    setIsGlobalRunning(true)
    setInstructionFeedback({
      message: locale === 'de' ? 'Dokument-Anweisung wird verarbeitet …' : 'Processing document instruction …',
      state: 'thinking',
    })
    const clearRunTimeout = startEditorRunTimeout(editorAbortMsRef.current, () => {
      controller.abort()
      setInstructionFeedback({
        message: editorTimeoutMessage(locale),
        state: 'error',
      })
      setIsGlobalRunning(false)
    })

    try {
      const { attachments, ruleSnippet: snippet } = await resolveRunContext(authorityGuard)
      const documentContext = await resolveDocumentContext(controller.signal, authorityGuard)
      const modelTier = selectedModelTierRef.current
      const model = selectedModelRef.current
      const effort = selectedEffortRef.current
      const proposal = await invokeEditorAiProvider(
        activeDocument,
        collaboration.access,
        locale,
        () => suggestionProducer.produceInstruction({
        attachments,
        documentMarkdown: documentContext.markdown,
        instruction: draftInstruction,
        modelTier,
        model,
        effort,
        signal: controller.signal,
        snippet: snippet || undefined,
        }),
        authorityGuard,
      )
      if (controller.signal.aborted) return
      authorityGuard?.assertCurrent()
      const now = new Date().toISOString()
      const groupId = createLocalId('editor-suggestion-group')
      const suggestions = createInstructionSuggestionRecords({
        // Legacy results retain quote fallback after a view switch. A
        // collaboration result without a live relative-anchor adapter throws and
        // is surfaced by this run's error state.
        activeEditor: mountedRef.current ? activeEditor : null,
        document: activeDocument,
        groupId,
        locale,
        now,
        proposal,
      })
      // Im Kollaborationsmodus braucht JEDER Vorschlag eine vorherige
      // Autorisierung durch einen creator-privaten Entwurf -- sonst weist
      // der Serverwaechter die spaetere Veroeffentlichung mit
      // "patch_not_found" ab. Der Entwurf kann nur an einer Kommentarzeile
      // haengen, also legt der Lauf je Edit eine Traegerzeile an. Sie ist
      // vom Typ 'assistant_edit' und taucht in der Notizliste nicht auf.
      const authorized = activeDocument.contentMode === 'collaboration'
        ? await persistInstructionCarriers(
            suggestions,
            draftInstruction,
            now,
            authorityGuard,
          )
        : suggestions
      // EINE Zahl fuer beide Ursachen: ein Edit, dessen Anker nicht
      // aufloesbar war, und einer, dessen Autorisierung scheiterte, sind
      // fuer den Nutzer dasselbe -- er sieht ihn nicht.
      const unauthorized = proposal.edits.length - authorized.length
      if (authorized.length > 0) {
        authorityGuard?.assertCurrent()
        dispatch({
          group: {
            assistantMessage: proposal.assistantMessage,
            createdAt: now,
            documentId: activeDocument.id,
            id: groupId,
            origin: { kind: 'global_run' },
            warnings: proposal.warnings,
          },
          suggestions: authorized,
          type: 'createEditorSuggestionGroup',
        })
      }
      authorityGuard?.assertCurrent()
      dispatch({ draft: '', type: 'setEditorAssistantDraft' })
      authorityGuard?.assertCurrent()
      onGlobalSuccess()
      // Ein Edit, der nicht autorisiert werden konnte, wird nicht angezeigt.
      // Das darf der Nutzer nicht erst durch Nachzaehlen merken.
      const warnings = unauthorized > 0
        ? [...(proposal.warnings ?? []), unauthorizedEditsWarning(locale, unauthorized)]
        : proposal.warnings
      setInstructionFeedback({
        editCount: authorized.length,
        message: proposal.assistantMessage || defaultInstructionResultMessage(locale, authorized.length),
        state: 'result',
        warnings,
      })
    } catch (error) {
      if (controller.signal.aborted) return
      setInstructionFeedback({
        message: messageFromError(error),
        state: 'error',
      })
    } finally {
      clearRunTimeout()
      setIsGlobalRunning(false)
    }
  }

  return {
    aiReadOnlyReason,
    clearInstructionFeedback,
    documentSuggestions,
    handleAcceptSuggestionGroup,
    handleAcceptSuggestion,
    handleEditSuggestionProposal,
    handleGlobalRun,
    handleInstructionRun,
    handleMarkSuggestionStale,
    handleRefineSuggestion,
    handleRejectSuggestion,
    handleRejectSuggestionGroup,
    handleRunComment,
    handleStopRun,
    handleStopSuggestionRun,
    instructionFeedback,
    isGlobalRunning,
    runErrors,
    runningCommentIds,
    savingCommentDraftIds,
    runningSuggestionIds,
    suggestionPublishDisabledReason,
    suggestionErrors,
  }
}

type SuggestionRecordArgs = {
  comment: EditorCommentThreadRecord
  documentId?: string
  groupId: string
  now: string
  origin: EditorSuggestionOrigin
  proposal: {
    changeSummary?: string[]
    evidence?: EditorSuggestionRecord['evidence']
    proposedText: string
    warnings?: string[]
  }
}

export function privateSuggestionDraftCreatePayload(
  suggestion: EditorSuggestionRecord,
): EditorSuggestionDraftCreateWire {
  if (!suggestion.privateDraft) {
    throw new Error('Private suggestion draft identity is missing.')
  }
  // Ankerstelle und Einfuegeart reisen NUR bei der Erstanlage mit: sie
  // gehoeren zur Identitaet des Vorschlags. Ohne sie baut der
  // Uebernahmepfad nach einem Neuladen jede Einfuegung als Ersetzung neu
  // auf -- aus "nach dem Anker einfuegen" wird "den Anker ersetzen".
  const anchorText = suggestion.anchorText ?? suggestion.originalText
  return {
    ...(anchorText ? { anchor_text: anchorText } : {}),
    anchor_version: 1,
    change_summary: suggestion.changeSummary ?? [],
    ...(suggestion.editPosition ? { edit_position: suggestion.editPosition } : {}),
    evidence: suggestion.evidence ?? null,
    group_id: suggestion.groupId,
    patch_id: suggestion.privateDraft.patchId,
    proposed_text: suggestion.proposedText,
    publication_command_id: suggestion.privateDraft.publicationCommandId,
    suggestion_id: suggestion.id,
    warnings: suggestion.warnings ?? [],
  }
}

export function privateSuggestionDraftRevisionPayload(
  suggestion: EditorSuggestionRecord,
  source: EditorSuggestionDraftRevisionRequestWire['revision_source'],
  instruction?: string,
): EditorSuggestionDraftRevisionRequestWire {
  return {
    change_summary: suggestion.changeSummary ?? [],
    evidence: suggestion.evidence ?? null,
    ...(instruction ? { instruction } : {}),
    proposed_text: suggestion.proposedText,
    revision_source: source,
    warnings: suggestion.warnings ?? [],
  }
}

/** Die Kennungen, unter denen dieser Vorschlag veroeffentlicht wird.
 *
 * Sie stammen IMMER aus dem gespeicherten Entwurf. Frueher wurden sie hier
 * frisch gewuerfelt, wenn keiner vorlag -- eine erfundene Identitaet, zu der
 * es per Konstruktion keine Autorisierung geben konnte. Der Server wies sie
 * mit "patch_not_found" ab, und der Nutzer las eine Konfliktmeldung fuer
 * etwas, das nie eine Chance hatte.
 *
 * Fehlt der Entwurf, ist das ein Programmfehler weiter oben und wird als
 * solcher benannt, statt eine Veroeffentlichung zu versuchen, die sicher
 * scheitert. */
export function privateSuggestionPublicationIdentity(
  suggestion: EditorSuggestionRecord,
  locale: 'de' | 'en',
): { commandId: string; patchId: string } {
  if (!suggestion.privateDraft) {
    throw new Error(missingPrivateDraftMessage(locale))
  }
  return {
    commandId: suggestion.privateDraft.publicationCommandId,
    patchId: suggestion.privateDraft.patchId,
  }
}

type InstructionRecordArgs = {
  activeEditor: Editor | null
  /** Vorbelegt aus dem Editor, wie bei `materializePrivateSuggestionComment`.
   *  Der echte Adapter haengt an einer lebenden Yjs-Bindung. */
  anchorAdapter?: EditorRelativePositionAdapter | null
  document: EditorDocumentRecord
  groupId: string
  locale: 'de' | 'en'
  now: string
  proposal: InstructionProposal
}

type SuggestionApplyTarget =
  | { kind: 'insert'; at: number }
  | { kind: 'replace'; range: EditorTextRange }

type SuggestionTargetResolution = {
  anchor: PrivateSuggestionAnchorResult
  target: SuggestionApplyTarget | null
}

type MaterializedPrivateSuggestionComment = {
  anchor: PrivateSuggestionAnchorResult
  comment: EditorCommentThreadRecord | null
}

function createSuggestionRecord({
  comment,
  documentId = comment.documentId,
  groupId,
  now,
  origin,
  proposal,
}: SuggestionRecordArgs): EditorSuggestionRecord {
  return {
    anchor: comment.anchor,
    blockId: comment.anchor.blockId ?? '',
    createdAt: now,
    documentId,
    groupId,
    id: createLocalId('editor-suggestion'),
    originalMarkdown: comment.anchor.selectedMarkdown,
    originalText: comment.anchor.selectedText,
    origin,
    proposedText: proposal.proposedText,
    revision: 1,
    status: 'pending',
    updatedAt: now,
    ...(proposal.changeSummary?.length ? { changeSummary: proposal.changeSummary } : {}),
    ...(proposal.evidence ? { evidence: proposal.evidence } : {}),
    ...(proposal.warnings?.length ? { warnings: proposal.warnings } : {}),
  }
}

function preparePrivateSuggestionRecord(
  suggestion: EditorSuggestionRecord,
  existingDraft: EditorCommentThreadRecord['suggestionDraft'],
  locale: 'de' | 'en',
): EditorSuggestionRecord {
  const publicationCommandId = existingDraft?.publicationCommandId
    ?? requiredRandomUuid(locale)
  const patchId = existingDraft?.patchId ?? requiredRandomUuid(locale)
  return {
    ...suggestion,
    groupId: existingDraft?.groupId ?? suggestion.groupId,
    id: existingDraft?.suggestionId ?? suggestion.id,
    privateDraft: {
      patchId,
      publicationCommandId,
      revision: existingDraft?.revision ?? 0,
    },
  }
}

/** Mirror one server-side AGENT patch as suggestion records (P7).

    Reuses the instruction converter (the wire edit shape is field-
    identical) ONE EDIT AT A TIME, so the edit↔record pairing survives
    the converter's legitimate skips (an empty insert is skipped here
    AND by the server apply). Every record carries the server patch
    identity — the accept/reject handlers route those through the
    official patch endpoints instead of the client-side edit path. */
export function createAgentPatchSuggestionRecords({
  activeEditor,
  document,
  groupId,
  locale,
  now,
  patch,
}: {
  activeEditor: Editor | null
  document: EditorDocumentRecord
  groupId: string
  locale: 'de' | 'en'
  now: string
  patch: AgentPatchWire
}): EditorSuggestionRecord[] {
  const records: EditorSuggestionRecord[] = []
  for (const edit of patch.edits) {
    const converted = createInstructionSuggestionRecords({
      activeEditor,
      document,
      groupId,
      locale,
      now,
      proposal: {
        assistantMessage: patch.summary,
        edits: [edit],
        warnings: patch.warnings.length ? patch.warnings : undefined,
      },
    })
    for (const record of converted) {
      records.push({
        ...record,
        serverPatch: {
          editId: edit.id,
          patchId: patch.patch_id,
        },
      })
    }
  }
  return records
}


export function createInstructionSuggestionRecords({
  activeEditor,
  anchorAdapter = activeEditor ? privateSuggestionAnchorAdapter(activeEditor) : null,
  document,
  groupId,
  locale,
  now,
  proposal,
}: InstructionRecordArgs): EditorSuggestionRecord[] {
  const suggestions: EditorSuggestionRecord[] = []
  proposal.edits.forEach((edit, index) => {
    const position = edit.position
    const anchorText = edit.find.trim()
    const proposedText = edit.text.trim()
    if (!proposedText && position !== 'replace') return
    const serializedAnchor = serializePrivateSuggestionAnchor(
      createInstructionAnchor(activeEditor, document.id, edit, index),
      anchorAdapter,
      document.contentMode === 'collaboration' && position !== 'append',
    )
    if (serializedAnchor.status === 'failed') {
      // Ohne Adapter kann KEIN Edit dieses Laufs einen relativen Anker
      // bekommen -- eine Aussage ueber den Lauf, also weiterhin ein Abbruch.
      // Mit Adapter betrifft der Fehlschlag genau diesen Edit: ihn zu
      // ueberspringen kostet den Nutzer eine Aenderung, ihn werfen zu lassen
      // kostet ihn alle, samt bezahltem Modelllauf. Uebersprungene Edits
      // bleiben nicht still -- sie gehen in die Differenz ein, aus der
      // `unauthorizedEditsWarning` ihre Zahl zieht.
      if (!anchorAdapter) {
        throw new Error(requiredRelativeAnchorMessage(locale))
      }
      return
    }
    const anchor = serializedAnchor.anchor
    const originalText = position === 'replace' ? anchorText : ''
    suggestions.push({
      anchor,
      anchorText,
      blockId: anchor.blockId ?? `instruction-${index}`,
      changeSummary: edit.note ? [edit.note] : undefined,
      createdAt: now,
      documentId: document.id,
      editPosition: position,
      groupId,
      id: createLocalId('editor-suggestion'),
      originalMarkdown: position === 'replace' ? anchorText : '',
      originalText,
      origin: { kind: 'global_run' },
      proposedText,
      revision: 1,
      status: 'pending',
      updatedAt: now,
      warnings: proposal.warnings?.length ? proposal.warnings : undefined,
    })
  })
  return suggestions
}

function createInstructionAnchor(
  activeEditor: Editor | null,
  documentId: string,
  edit: { find: string; position: EditorSuggestionEditPosition; quote_after: string; quote_before: string },
  index: number,
): EditorCommentAnchorRecord {
  const anchorText = edit.find.trim()
  if (activeEditor && anchorText) {
    // P7-E2: model-authored edits resolve STRICTLY (server semantics:
    // hard quote disqualification, abstain on ties) — an ambiguous
    // anchor degrades visibly to the end-of-document record below
    // instead of guessing an occurrence the server would skip. The
    // hint is inert in strict mode.
    const range = resolveAnchorRange(activeEditor, {
      hint: 1,
      mode: 'strict',
      quoteAfter: edit.quote_after,
      quoteBefore: edit.quote_before,
      text: anchorText,
    })
    if (range) return materializeAnchorForRange(activeEditor, range)
  }
  const docSize = activeEditor?.state.doc.content.size ?? 1
  return {
    blockId: `${documentId}:instruction:${index}`,
    from: docSize,
    quoteAfter: edit.quote_after,
    quoteBefore: edit.quote_before,
    selectedMarkdown: anchorText,
    selectedText: anchorText,
    to: docSize,
  }
}

export function resolveSuggestionTarget(
  editor: Editor,
  suggestion: EditorSuggestionRecord,
  adapter = privateSuggestionAnchorAdapter(editor),
): SuggestionTargetResolution {
  const position = suggestion.editPosition ?? 'replace'
  if (position === 'append') {
    return {
      anchor: { anchor: suggestion.anchor, status: 'legacy' },
      target: { at: editor.state.doc.content.size, kind: 'insert' },
    }
  }
  const anchorText = (suggestion.anchorText ?? suggestion.originalText).trim()
  const anchor = resolvePrivateSuggestionAnchor(suggestion.anchor, adapter)
  if (!anchorText) return { anchor, target: null }
  const range = resolveAnchorRange(editor, {
    hint: clampAnchor(anchor.anchor, editor).from,
    quoteAfter: anchor.anchor.quoteAfter,
    quoteBefore: anchor.anchor.quoteBefore,
    text: anchorText,
  })
  if (!range) return { anchor, target: null }
  if (position === 'replace') return { anchor, target: { kind: 'replace', range } }
  return {
    anchor,
    target: {
      at: blockInsertionPositionForRange(editor, range, position),
      kind: 'insert',
    },
  }
}

export function prepareCollaborationSuggestionBatch(
  editor: Editor,
  suggestions: readonly EditorSuggestionRecord[],
  adapter: EditorRelativePositionAdapter | null,
  locale: 'de' | 'en',
): { errors: Record<string, string>; suggestions: EditorSuggestionRecord[] } {
  const errors: Record<string, string> = {}
  const prepared: Array<{ position: number; suggestion: EditorSuggestionRecord }> = []
  for (const suggestion of suggestions) {
    const resolution = resolveSuggestionTarget(editor, suggestion, adapter)
    const anchorRequired = (suggestion.editPosition ?? 'replace') !== 'append'
    if (anchorRequired && resolution.anchor.status !== 'relative') {
      errors[suggestion.id] = degradedRelativeAnchorMessage(locale)
      continue
    }
    if (!resolution.target) {
      errors[suggestion.id] = staleAnchorMessage(locale)
      continue
    }
    prepared.push({
      position: resolution.target.kind === 'replace'
        ? resolution.target.range.from
        : resolution.target.at,
      suggestion: {
        ...suggestion,
        anchor: resolution.anchor.anchor,
      },
    })
  }
  return {
    errors,
    suggestions: prepared
      .sort((left, right) => right.position - left.position)
      .map(({ suggestion }) => suggestion),
  }
}

type AnchoredSuggestion = Pick<
  EditorSuggestionRecord,
  'anchor' | 'createdAt'
>

/** Resolve live Yjs positions before ordering a compound private AI group. */
export function sortPrivateSuggestionGroup<T extends AnchoredSuggestion>(
  suggestions: readonly T[],
  adapter: EditorRelativePositionAdapter | null,
): T[] {
  return [...suggestions].sort((left, right) => {
    const leftAnchor = resolvePrivateSuggestionAnchor(left.anchor, adapter)
    const rightAnchor = resolvePrivateSuggestionAnchor(right.anchor, adapter)
    return leftAnchor.anchor.from - rightAnchor.anchor.from
      || left.createdAt.localeCompare(right.createdAt)
  })
}

/** Resolve relative positions with an explicit status for quote degradation. */
export function resolvePrivateSuggestionAnchor(
  anchor: EditorCommentAnchorRecord,
  adapter: EditorRelativePositionAdapter | null,
): PrivateSuggestionAnchorResult {
  if (!hasCollaborationRelativeAnchor(anchor)) {
    return { anchor, status: 'legacy' }
  }
  if (!adapter) {
    return { anchor, reason: 'adapter_unavailable', status: 'degraded' }
  }
  const resolved = resolveCollaborationAnchor(anchor, adapter)
  return resolved.source === 'relative'
    ? { anchor: resolved.anchor, status: 'relative' }
    : { anchor, reason: 'relative_unresolved', status: 'degraded' }
}

/** Add Yjs-relative boundaries without sacrificing the quote/absolute fields. */
export function serializePrivateSuggestionAnchor(
  anchor: EditorCommentAnchorRecord,
  adapter: EditorRelativePositionAdapter | null,
  required = false,
): PrivateSuggestionAnchorResult {
  if (required && anchor.to <= anchor.from) {
    return { anchor, reason: 'relative_missing', status: 'failed' }
  }
  if (!adapter) {
    return required
      ? { anchor, reason: 'adapter_unavailable', status: 'failed' }
      : { anchor, status: 'legacy' }
  }
  try {
    const serialized = serializeCollaborationAnchor(anchor, adapter)
    if (required && (!serialized.relativeFrom || !serialized.relativeTo)) {
      return { anchor, reason: 'encoding_failed', status: 'failed' }
    }
    return { anchor: serialized, status: 'relative' }
  } catch {
    return required
      ? { anchor, reason: 'encoding_failed', status: 'failed' }
      : { anchor, reason: 'encoding_failed', status: 'degraded' }
  }
}

function materializePrivateSuggestionComment(
  editor: Editor,
  comment: EditorCommentThreadRecord,
  adapter = privateSuggestionAnchorAdapter(editor),
  requireRelative = false,
): MaterializedPrivateSuggestionComment {
  const resolved = resolvePrivateSuggestionAnchor(comment.anchor, adapter)
  const materialized = materializeCommentThread(editor, {
    ...comment,
    anchor: resolved.anchor,
  })
  if (!materialized) return { anchor: resolved, comment: null }
  const serialized = serializePrivateSuggestionAnchor(
    materialized.anchor,
    adapter,
    requireRelative,
  )
  const anchor = resolved.status === 'degraded' ? resolved : serialized
  return {
    anchor,
    comment: {
      ...materialized,
      anchor: serialized.anchor,
    },
  }
}

function privateSuggestionAnchorAdapter(
  editor: Editor,
): EditorRelativePositionAdapter | null {
  try {
    const pluginState = ySyncPluginKey.getState(editor.state) as {
      binding?: { mapping?: ProseMirrorMapping }
      doc?: Y.Doc
      type?: Y.XmlFragment
    } | undefined
    const document = pluginState?.doc
    const fragment = pluginState?.type
    const mapping = pluginState?.binding?.mapping
    if (!document || !fragment || !(mapping instanceof Map)) return null
    return createRelativePositionAdapter(document, fragment, mapping)
  } catch {
    return null
  }
}

/** Apply private proposals to an isolated editor projection. The live Yjs
 * editor is never mutated; the collaboration endpoint owns publication. */
export function buildCollaborationSuggestionTargetMarkdown(
  currentMarkdown: string,
  suggestions: readonly EditorSuggestionRecord[],
  locale: 'de' | 'en' = 'en',
): string {
  const editor = new HeadlessEditor({
    content: parseEditorMarkdown(currentMarkdown),
    element: null,
    extensions: createEditorSchemaExtensions({ enableUndoRedo: false }),
    injectCSS: false,
  })
  try {
    for (const suggestion of suggestions) {
      const { target } = resolveSuggestionTarget(editor, suggestion, null)
      if (!target) throw new Error(staleAnchorMessage(locale))
      const content = normalizeEditorMarkdownForTiptap(suggestion.proposedText)
      const applied = target.kind === 'replace'
        ? editor.commands.insertContentAt(
            target.range,
            suggestionReplacementContent(editor, target.range, content),
            { contentType: 'markdown' },
          )
        : editor.commands.insertContentAt(
            target.at,
            content,
            { contentType: 'markdown' },
          )
      if (!applied) throw new Error(collaborationTargetProjectionMessage(locale))
    }
    return serializeEditorJson(editor.getJSON(), 'final')
  } finally {
    editor.destroy()
  }
}

/** Einen angenommenen Vorschlag ins Dokument einsetzen.
 *
 * Frei statt in der Hook-Closure, damit der lokale Pfad ueberhaupt
 * pruefbar ist: die Testbahn laeuft ohne DOM, ein Hook waere dort nicht
 * zu rendern. Beide Einsetzstellen — hier und der Kollaborationspfad —
 * benutzen DIESELBE Inline-Entscheidung; sie existierte vorher nur im
 * Kollaborationszweig, weshalb lokal aus einem Satz drei Absaetze wurden. */
export function applySuggestionToEditor(
  editor: Editor,
  suggestion: EditorSuggestionRecord,
): boolean {
  const { target } = resolveSuggestionTarget(editor, suggestion)
  if (!target) return false
  const content = normalizeEditorMarkdownForTiptap(suggestion.proposedText)
  if (target.kind === 'replace') {
    editor.chain().focus().insertContentAt(
      target.range,
      suggestionReplacementContent(editor, target.range, content),
      { contentType: 'markdown' },
    ).run()
    return true
  }
  editor.chain().focus().insertContentAt(
    target.at,
    content,
    { contentType: 'markdown' },
  ).run()
  return true
}

/** Ein Ein-Absatz-Vorschlag in einem Inline-Bereich wird als Inline-Inhalt
 * eingesetzt, alles andere unveraendert als Markdown. Reine Editor-Semantik,
 * kein Kollaborationsbelang — deshalb ohne den frueheren Praefix. */
function suggestionReplacementContent(
  editor: Editor,
  range: EditorTextRange,
  markdown: string,
): string | JSONContent[] {
  const from = editor.state.doc.resolve(range.from)
  const to = editor.state.doc.resolve(range.to)
  if (!from.sameParent(to) || !from.parent.isTextblock) return markdown
  const parsed = parseEditorMarkdown(markdown)
  const onlyBlock = parsed.content?.length === 1 ? parsed.content[0] : undefined
  return onlyBlock?.type === 'paragraph' ? onlyBlock.content ?? [] : markdown
}

export function collaborationPublicationFromResponse(
  response: EditorCollaborationSuggestionPublishResponse,
  expected: { commandId: string; expectedSequence: number; patchId: string },
  locale: 'de' | 'en',
): CollaborationSuggestionPublication {
  if (
    response.command_id !== expected.commandId
    || response.patch_id !== expected.patchId
    || !Number.isSafeInteger(response.sequence)
    || response.sequence <= expected.expectedSequence
    || !Array.isArray(response.suggestion_ids)
    || response.suggestion_ids.length === 0
    || response.suggestion_ids.some((id) => typeof id !== 'string' || id.length === 0)
  ) {
    throw new Error(collaborationPublishInvalidMessage(locale))
  }
  return {
    commandId: response.command_id,
    patchId: response.patch_id,
    sequence: response.sequence,
    suggestionIds: [...response.suggestion_ids],
  }
}

function requiredRandomUuid(locale: 'de' | 'en'): string {
  try {
    return createSecureUuid()
  } catch {
    throw new Error(locale === 'de'
      ? 'Der Collaboration-Auftrag konnte nicht sicher identifiziert werden.'
      : 'The collaboration command could not be identified safely.')
  }
}

function messageFromError(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}

function startEditorRunTimeout(timeoutMs: number, onTimeout: () => void): () => void {
  const timeoutId = globalThis.setTimeout(onTimeout, timeoutMs)
  return () => globalThis.clearTimeout(timeoutId)
}

function editorTimeoutMessage(locale: 'de' | 'en'): string {
  if (locale === 'de') {
    return 'Keine Modellantwort innerhalb der Zeitgrenze. Der Lauf wurde abgebrochen; bitte erneut versuchen oder ein schnelleres Modell wählen.'
  }
  return 'No model response within the time limit. The run was cancelled; retry or choose a faster model.'
}

function editorRecoveryAiDisabledMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Speichern Sie die lokale Recovery-Kopie zuerst als neues Dokument, bevor Sie KI-Funktionen verwenden.'
    : 'Save the local recovery copy as a new document before using AI features.'
}

function defaultInstructionResultMessage(locale: 'de' | 'en', editCount: number): string {
  if (locale === 'de') {
    return editCount > 0
      ? `${editCount} Dokument-Änderungen vorgeschlagen.`
      : 'Keine konkreten Dokument-Änderungen vorgeschlagen.'
  }
  return editCount > 0
    ? `${editCount} document changes proposed.`
    : 'No concrete document changes proposed.'
}

function staleAnchorMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Textstelle hat sich geändert. Bitte markieren Sie den Text neu.'
    : 'The referenced text changed. Please select the passage again.'
}

function unauthorizedEditsWarning(locale: 'de' | 'en', count: number): string {
  return locale === 'de'
    ? `${count} ${count === 1 ? 'Aenderung konnte' : 'Aenderungen konnten'} nicht vorbereitet werden und ${count === 1 ? 'wird' : 'werden'} nicht angezeigt.`
    : `${count} ${count === 1 ? 'change' : 'changes'} could not be prepared and ${count === 1 ? 'is' : 'are'} not shown.`
}

function missingPrivateDraftMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Dieser Vorschlag hat keine gespeicherte Autorisierung und kann nicht veroeffentlicht werden. Bitte erzeugen Sie ihn neu.'
    : 'This suggestion has no stored authorisation and cannot be published. Please generate it again.'
}

function requiredRelativeAnchorMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Der Vorschlag wurde nicht erstellt, weil die Yjs-Verankerung nicht sicher gespeichert werden konnte.'
    : 'The suggestion was not created because its Yjs anchor could not be stored safely.'
}

function degradedRelativeAnchorMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Die relative Yjs-Verankerung ist nicht mehr auflösbar. Bitte markieren Sie die Textstelle neu.'
    : 'The relative Yjs anchor can no longer be resolved. Please select the passage again.'
}

function collaborationEditorUnavailableMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Das Collaboration-Dokument ist noch nicht vollständig verbunden.'
    : 'The collaboration document is not fully connected yet.'
}

function collaborationProjectionPendingMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Die letzten Collaboration-Änderungen sind noch nicht dauerhaft gespeichert. Bitte warten Sie kurz und versuchen Sie es erneut.'
    : 'The latest collaboration changes are not durable yet. Wait a moment and try again.'
}

function collaborationProjectionInvalidMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Der Server lieferte keine gültige Collaboration-Projektion.'
    : 'The server did not return a valid collaboration projection.'
}

function collaborationPublishForbiddenMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Für dieses Dokument dürfen keine gemeinsamen Vorschläge veröffentlicht werden.'
    : 'You cannot publish shared suggestions for this document.'
}

function collaborationPublishInvalidMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Der Collaboration-Vorschlag wurde nicht dauerhaft bestätigt.'
    : 'The collaboration suggestion was not confirmed durably.'
}

function collaborationTargetProjectionMessage(locale: 'de' | 'en'): string {
  return locale === 'de'
    ? 'Der private Vorschlag konnte nicht sicher auf die Collaboration-Projektion angewendet werden.'
    : 'The private suggestion could not be applied safely to the collaboration projection.'
}

function collaborationIdentity(
  document: EditorDocumentRecord,
): { documentId: string; generation: number } | null {
  const generation = document.collaboration?.generation
  if (typeof generation !== 'number' || !Number.isSafeInteger(generation) || generation < 0) {
    return null
  }
  return { documentId: document.id, generation }
}

function createLocalId(prefix: string): string {
  return createSecurePrefixedId(prefix)
}
