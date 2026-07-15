/**
 * Editor server sync (M6b project-persistence tier).
 *
 * The editor counterpart of useChatHistoryApi, sharing the same shape:
 * hydrate document + folder METADATA on mount, load a document's heavy body
 * AND its comments lazily on open, and a debounced serialized autosave that
 * diffs three collections (documents, folders, comments) via the shared
 * syncCollection helper. Persistence only — it never calls a model; the
 * editor's AI/suggestion paths are untouched.
 *
 * It does NOT own the import button: the explicit opt-in push is the
 * project-level useProjectServerImport (which pushes chat AND editor in one
 * flow). This hook only hydrates + autosaves once the project is opted in
 * (syncActive), seeding its synced fingerprint to WHAT THE SERVER HOLDS so a
 * local-newer entity is pushed up rather than stranded (the M6a P1 lesson).
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch } from 'react'

import {
  deleteEditorComment,
  deleteEditorDocument,
  deleteEditorFolder,
  getEditorDocument,
  hasHttpStatus,
  listEditorComments,
  listEditorDocuments,
  listEditorFolders,
  patchEditorDocumentMetadata,
  saveEditorComments,
  saveEditorDocument,
  saveEditorFolder,
} from '@/api/inqtrixClient'
import {
  commentRecordFromServer,
  documentRecordFromServer,
  folderRecordFromServer,
  isCollaborationDocument,
  serverCommentPayload,
  serverDocumentPayload,
  serverFolderPayload,
} from '@/features/editor/editorSync'
import type {
  EditorCommentOutboxEntry,
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorFolderRecord,
} from '@/features/project/types'
import {
  deleteTolerant404,
  syncCollection,
} from '@/features/project/syncCollection'
import {
  useProjectSyncLifecycle,
  type SyncLifecycleToken,
} from '@/features/project/useProjectSyncLifecycle'
import type { ResearchDeskAction } from '@/features/researchDesk/state'

const AUTOSAVE_DEBOUNCE_MS = 1_500
const PAGE_LIMIT = 200

function messageFromError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

type CommentFingerprint = { documentId: string; updatedAt: string }

export type EditorCommentReconciliationPlan = {
  pendingDeletedCommentIds: Set<string>
  preserveCommentIds: Set<string>
  serverComments: EditorCommentThreadRecord[]
}

export type EditorOpenHydrationPlan = {
  loadComments: boolean
  loadDocumentDetail: boolean
}

export type EditorServerDocumentProvenance = 'exact_detail' | 'metadata'

export type EditorDocumentAutosavePlan =
  | { kind: 'legacy_body' }
  | {
      kind: 'collaboration_metadata'
      payload: {
        expected_metadata_revision: number
        folder_id: string | null
        title: string
      }
    }
  | { kind: 'none' }

/** Result of pushing one document: `saved` carries the server's adopted
 * revisions, `rebased` means a concurrent writer won the revision guard
 * (the local record was rebased onto the server state), `skipped` means
 * the autosave plan had nothing to push for this record. */
export type EditorDocumentPushOutcome =
  | { kind: 'saved'; metadataRevision: number | undefined; revision: number }
  | { kind: 'rebased' }
  | { kind: 'skipped' }

/** Keep the legacy body contract and collaboration metadata contract disjoint. */
export function planEditorDocumentAutosave(
  document: EditorDocumentRecord,
  expectedMetadataRevision = document.metadataRevision ?? 1,
  serverDocument?: EditorDocumentRecord,
): EditorDocumentAutosavePlan {
  const collaborationDocument = serverDocument
    && isCollaborationDocument(serverDocument)
    ? serverDocument
    : isCollaborationDocument(document)
      ? document
      : null
  if (!collaborationDocument) return { kind: 'legacy_body' }
  if (collaborationDocument.access?.mode !== 'owner') return { kind: 'none' }
  return {
    kind: 'collaboration_metadata',
    payload: {
      expected_metadata_revision: expectedMetadataRevision,
      folder_id: document.folderId,
      title: document.title,
    },
  }
}

export function shouldLoadLegacyEditorBody(
  document: EditorDocumentRecord | undefined,
): boolean {
  return document === undefined || !isCollaborationDocument(document)
}

/** Document detail and private comments have independent lazy-load lifecycles. */
export function planEditorOpenHydration({
  collaborationDocument,
  hasCommentSnapshot,
  hasExactDocumentDetail,
  hasLocalDocumentBody,
}: {
  collaborationDocument: boolean
  hasCommentSnapshot: boolean
  hasExactDocumentDetail: boolean
  hasLocalDocumentBody: boolean
}): EditorOpenHydrationPlan {
  return {
    loadComments: !hasCommentSnapshot,
    loadDocumentDetail: collaborationDocument
      ? !hasExactDocumentDetail
      : !hasLocalDocumentBody,
  }
}

/** Exact bodies are valid only for the collaboration projection lifecycle
 * that produced them. Local presence and metadata-only list records never
 * establish this provenance. */
export function editorDocumentDetailProvenanceKey(
  document: EditorDocumentRecord | undefined,
): string | null {
  if (!document) return null
  if (!isCollaborationDocument(document)) return `${document.id}:markdown`
  const generation = document.collaboration?.generation
  const projectionSequence = document.collaboration?.projectionSequence
  return `${document.id}:collaboration:g${generation ?? 'unknown'}:p${projectionSequence ?? 'unknown'}`
}

export function editorServerDocumentObservation(
  document: EditorDocumentRecord,
  provenance: EditorServerDocumentProvenance,
): {
  exactDetailProvenanceKey: string | null
  metadataRevision: number
  syncedFingerprint: string | null
} {
  return {
    exactDetailProvenanceKey: provenance === 'exact_detail'
      ? editorDocumentDetailProvenanceKey(document)
      : null,
    metadataRevision: document.metadataRevision ?? 1,
    syncedFingerprint: document.access?.mode === 'shared' ? null : document.updatedAt,
  }
}

/** Shared-in documents are live server views, never local autosave inputs. */
export function editorDocumentsForAutosave(
  documents: Record<string, EditorDocumentRecord>,
  serverDocuments?: ReadonlyMap<string, EditorDocumentRecord>,
): Record<string, EditorDocumentRecord> {
  return Object.fromEntries(Object.entries(documents).filter(([documentId, document]) => {
    const authoritativeDocument = serverDocuments?.get(documentId) ?? document
    return authoritativeDocument.access?.mode !== 'shared'
  }))
}

/** Private comments may persist on shared collaboration documents when the
 * caller has suggestion access; legacy shared documents remain read-only. */
export function canPersistEditorCommentsForDocument(
  document: EditorDocumentRecord | undefined,
): boolean {
  if (!document || document.access?.mode !== 'shared') return true
  return isCollaborationDocument(document) && document.access.permission !== 'view'
}

export function editorCommentsForAutosave(
  comments: Record<string, EditorCommentThreadRecord>,
  documents: Record<string, EditorDocumentRecord>,
  serverDocuments?: ReadonlyMap<string, EditorDocumentRecord>,
): Record<string, EditorCommentThreadRecord> {
  return Object.fromEntries(Object.entries(comments).filter(([, comment]) => {
    const authoritativeDocument = serverDocuments?.get(comment.documentId)
      ?? documents[comment.documentId]
    return canPersistEditorCommentsForDocument(authoritativeDocument)
  }))
}

/** Reconcile one exact server list while preserving only explicit local
 * outbox work. Pending deletes stay absent when a racing GET still returns
 * their older server record. */
export function planEditorCommentReconciliation(
  documentId: string,
  localComments: Readonly<Record<string, EditorCommentThreadRecord>>,
  serverComments: readonly EditorCommentThreadRecord[],
  commentOutbox: Readonly<Record<string, EditorCommentOutboxEntry>>,
): EditorCommentReconciliationPlan {
  const preserveCommentIds = new Set(
    Object.entries(commentOutbox)
      .filter(([commentId, entry]) => (
        entry.documentId === documentId
        && entry.operation === 'upsert'
        && localComments[commentId] !== undefined
      ))
      .map(([commentId]) => commentId),
  )
  const pendingDeletedCommentIds = new Set(
    Object.entries(commentOutbox)
      .filter(([, entry]) => entry.documentId === documentId && entry.operation === 'delete')
      .map(([commentId]) => commentId),
  )
  return {
    pendingDeletedCommentIds,
    preserveCommentIds,
    serverComments: serverComments.filter(
      (comment) => !pendingDeletedCommentIds.has(comment.id),
    ),
  }
}

type UseEditorHistoryApiOptions = {
  apiKey: string | undefined
  dispatch: Dispatch<ResearchDeskAction>
  editorCommentOutbox?: Record<string, EditorCommentOutboxEntry>
  editorComments: Record<string, EditorCommentThreadRecord>
  editorDocuments: Record<string, EditorDocumentRecord>
  editorFolders: Record<string, EditorFolderRecord>
  /** In-session load counter (bumped on every wholesale project replace). Part
   * of the lifecycle identity so a switch to another synced project re-hydrates
   * from its own server state instead of inheriting this one's synced map. */
  projectEpoch: number
  /** Bumped by the user-scoped invalidation channel (sharing) and local
   * share actions. A bump re-runs the authoritative document/folder
   * hydration WITHOUT a reset, so a document shared with this user appears
   * without a page reload (the run list follows the same pattern). */
  refreshToken?: number
  selectedDocumentId: string | null
  /** ``serverSyncEnabled`` AND the durable capability AND not demo. */
  syncActive: boolean
  workspaceId: string
}

export type EditorHistoryApiHandle = {
  error: string | null
  /** Push one document's current body to the server NOW (serialized against
   * the autosave loop) and return the fresh server revisions, or `null` when
   * the push could not produce an authoritative base (sync inactive, plan
   * skipped, or a concurrent writer won and the record was rebased). The
   * share/collaboration-enable flow uses this so converting a document never
   * depends on the global project dirty flag. */
  flushDocumentForShare: (
    document: EditorDocumentRecord,
  ) => Promise<{ metadataRevision: number; revision: number } | null>
  registerOpenedServerDocument: (
    document: EditorDocumentRecord,
    provenance: EditorServerDocumentProvenance,
  ) => void
}

export function useEditorHistoryApi({
  apiKey,
  dispatch,
  editorCommentOutbox,
  editorComments,
  editorDocuments,
  editorFolders,
  projectEpoch,
  refreshToken,
  selectedDocumentId,
  syncActive,
  workspaceId,
}: UseEditorHistoryApiOptions): EditorHistoryApiHandle {
  const [error, setError] = useState<string | null>(null)
  const [hydrated, setHydrated] = useState(false)

  const documentsRef = useRef(editorDocuments)
  documentsRef.current = editorDocuments
  const foldersRef = useRef(editorFolders)
  foldersRef.current = editorFolders
  const commentsRef = useRef(editorComments)
  commentsRef.current = editorComments
  const commentOutboxRef = useRef(editorCommentOutbox ?? {})
  commentOutboxRef.current = editorCommentOutbox ?? {}
  const optionsRef = useRef({ apiKey, workspaceId })
  optionsRef.current = { apiKey, workspaceId }

  const syncedDocsRef = useRef(new Map<string, string>())
  const syncedFoldersRef = useRef(new Map<string, string>())
  const syncedCommentsRef = useRef(new Map<string, CommentFingerprint>())
  const metadataRevisionsRef = useRef(new Map<string, number>())
  const serverDocumentsRef = useRef(new Map<string, EditorDocumentRecord>())
  const loadedDocsRef = useRef(new Set<string>())
  const exactDetailProvenanceRef = useRef(new Map<string, string>())
  const loadedCommentsRef = useRef(new Set<string>())
  const hydratedRef = useRef(false)
  const flushingRef = useRef(false)
  const flushPendingRef = useRef(false)
  const syncActiveRef = useRef(syncActive)
  syncActiveRef.current = syncActive
  const [serverObservationEpoch, setServerObservationEpoch] = useState(0)

  const registerOpenedServerDocument = useCallback((
    document: EditorDocumentRecord,
    provenance: EditorServerDocumentProvenance,
  ) => {
    const observation = editorServerDocumentObservation(document, provenance)
    serverDocumentsRef.current.set(document.id, document)
    metadataRevisionsRef.current.set(document.id, observation.metadataRevision)
    if (observation.exactDetailProvenanceKey) {
      exactDetailProvenanceRef.current.set(
        document.id,
        observation.exactDetailProvenanceKey,
      )
      loadedDocsRef.current.add(document.id)
    }
    if (observation.syncedFingerprint) {
      syncedDocsRef.current.set(document.id, observation.syncedFingerprint)
    }
    setServerObservationEpoch((current) => current + 1)
  }, [])

  // -- pushing one entity ----------------------------------------------- #

  const pushDocument = useCallback(async (
    document: EditorDocumentRecord,
  ): Promise<EditorDocumentPushOutcome> => {
    const plan = planEditorDocumentAutosave(
      document,
      metadataRevisionsRef.current.get(document.id),
      serverDocumentsRef.current.get(document.id),
    )
    if (plan.kind === 'none') return { kind: 'skipped' }
    if (plan.kind === 'collaboration_metadata') {
      const saved = await patchEditorDocumentMetadata(
        document.id,
        plan.payload,
        optionsRef.current,
      )
      const metadataRevision = requireEditorMetadataRevision(saved.metadata_revision)
      metadataRevisionsRef.current.set(document.id, metadataRevision)
      dispatch({
        documentId: document.id,
        metadataRevision,
        type: 'adoptEditorDocumentMetadataRevision',
      })
      const savedRecord = documentRecordFromServer(saved)
      serverDocumentsRef.current.set(document.id, savedRecord)
      return { kind: 'saved', metadataRevision, revision: saved.revision }
    }

    let record = document
    if (
      syncedDocsRef.current.has(document.id) &&
      !loadedDocsRef.current.has(document.id)
    ) {
      // A tree-level metadata edit (rename / drag-to-folder / folder delete)
      // bumped this server-held document's updatedAt while its body was never
      // loaded (it is still ""). The PUT is a full-record upsert, so sending
      // body="" would ERASE the server's real body. Fetch the current server
      // body first and push the merged record (new metadata + kept body).
      const detail = await getEditorDocument(document.id, optionsRef.current)
      const contentMarkdown = detail.content_markdown ?? ''
      dispatch({
        contentMarkdown,
        documentId: document.id,
        type: 'setServerEditorDocumentBody',
      })
      record = { ...document, contentMarkdown }
    }
    try {
      const saved = await saveEditorDocument(
        record.id,
        serverDocumentPayload(record),
        optionsRef.current,
      )
      // Adopt the server's new revision as our base (revision now tracks the
      // SERVER, not local edits). Revision-only, never touching content/
      // updatedAt/dirty: the flush fingerprint is updatedAt, so this cannot
      // wedge a re-flush, and a live keystroke during the save keeps its
      // newer body. Without this, the next save would re-send the same base+1
      // and 409 against the just-advanced server on every flush.
      dispatch({
        documentId: record.id,
        revision: saved.revision,
        type: 'adoptEditorDocumentRevision',
      })
      // The upsert response also carries the metadata revision. Adopting it
      // makes a document created THIS session shareable immediately: without
      // it, `metadataRevision` stayed undefined until a reload re-hydrated
      // the record, and the share/collaboration entry points treat that as
      // "not on the server yet" (disabled Share button).
      const savedMetadataRevision = saved.metadata_revision
      const adoptedMetadataRevision = (
        savedMetadataRevision !== undefined
        && Number.isSafeInteger(savedMetadataRevision)
        && savedMetadataRevision >= 1
      )
        ? savedMetadataRevision
        : undefined
      if (adoptedMetadataRevision !== undefined) {
        metadataRevisionsRef.current.set(record.id, adoptedMetadataRevision)
        dispatch({
          documentId: record.id,
          metadataRevision: adoptedMetadataRevision,
          type: 'adoptEditorDocumentMetadataRevision',
        })
      }
      // We hold this document's body now, so re-opening it must not trigger
      // a redundant body fetch.
      loadedDocsRef.current.add(record.id)
      return {
        kind: 'saved',
        metadataRevision: adoptedMetadataRevision,
        revision: saved.revision,
      }
    } catch (cause) {
      if (!hasHttpStatus(cause, 409)) throw cause
      // The server's revision guard refused this save: a concurrent
      // writer (typically an agent patch apply) advanced the document
      // past our base. Refetch and rebase onto the server revision.
      // pushedContentMarkdown lets the reducer tell whether the user
      // kept typing during the PUT->GET window: if not, it adopts the
      // server body; if so, it KEEPS the live keystrokes and re-pushes
      // them on the fresh base (never silently overwriting a live edit).
      console.warn(
        `[inqtrix] Editor-Dokument ${record.id}: Autosave verlor den `
        + 'Revision-Guard (409) — Server-Stand wird uebernommen.',
      )
      const detail = await getEditorDocument(record.id, optionsRef.current)
      dispatch({
        contentMarkdown: detail.content_markdown ?? '',
        documentId: record.id,
        pushedContentMarkdown: record.contentMarkdown,
        revision: detail.revision,
        type: 'rebaseServerEditorDocument',
      })
      loadedDocsRef.current.add(record.id)
      return { kind: 'rebased' }
    }
  }, [dispatch])

  const pushFolder = useCallback(async (folder: EditorFolderRecord) => {
    await saveEditorFolder(folder.id, serverFolderPayload(folder), optionsRef.current)
  }, [])

  const pushComment = useCallback(async (comment: EditorCommentThreadRecord) => {
    await saveEditorComments(
      comment.documentId,
      [serverCommentPayload(comment)],
      optionsRef.current,
    )
  }, [])

  // -- autosave flush (debounced, serialized) --------------------------- #

  const flush = useCallback(async () => {
    if (!syncActiveRef.current || !hydratedRef.current) return
    if (flushingRef.current) {
      flushPendingRef.current = true
      return
    }
    flushingRef.current = true
    try {
      // Documents first: a document delete cascades its comments server-side.
      await syncCollection<EditorDocumentRecord, string>({
        current: editorDocumentsForAutosave(
          documentsRef.current,
          serverDocumentsRef.current,
        ),
        synced: syncedDocsRef.current,
        fingerprintOf: (document) => document.updatedAt,
        changed: (previous, current) => previous !== current,
        pushOne: async (document) => {
          await pushDocument(document)
        },
        deleteOne: async (id) => {
          await deleteTolerant404(
            () => deleteEditorDocument(id, optionsRef.current),
          )
          loadedDocsRef.current.delete(id)
          exactDetailProvenanceRef.current.delete(id)
          loadedCommentsRef.current.delete(id)
          serverDocumentsRef.current.delete(id)
          // The server cascade-deletes this document's comments. Mirror that
          // in the client bookkeeping so the comment sync below does not try a
          // redundant delete that 404s (parent gone) and wedges the autosave
          // in a retry loop.
          for (const [commentId, fingerprint] of [
            ...syncedCommentsRef.current.entries(),
          ]) {
            if (fingerprint.documentId === id) {
              syncedCommentsRef.current.delete(commentId)
            }
          }
        },
      })
      await syncCollection<EditorFolderRecord, string>({
        current: foldersRef.current,
        synced: syncedFoldersRef.current,
        fingerprintOf: (folder) => folder.updatedAt,
        changed: (previous, current) => previous !== current,
        pushOne: pushFolder,
        deleteOne: (id) => deleteTolerant404(
          () => deleteEditorFolder(id, optionsRef.current),
        ),
      })
      const acknowledgements: Array<{
        commentId: string
        operation: 'delete' | 'upsert'
        updatedAt?: string
      }> = []
      for (const [commentId, pending] of Object.entries(commentOutboxRef.current)) {
        const authoritativeDocument = serverDocumentsRef.current.get(pending.documentId)
          ?? documentsRef.current[pending.documentId]
        if (!canPersistEditorCommentsForDocument(authoritativeDocument)) continue
        if (pending.operation === 'upsert') {
          const comment = commentsRef.current[commentId]
          if (!comment || comment.updatedAt !== pending.updatedAt) continue
          await pushComment(comment)
          syncedCommentsRef.current.set(commentId, {
            documentId: comment.documentId,
            updatedAt: comment.updatedAt,
          })
        } else {
          await deleteTolerant404(
            () => deleteEditorComment(
              pending.documentId,
              commentId,
              optionsRef.current,
            ),
          )
          syncedCommentsRef.current.delete(commentId)
        }
        acknowledgements.push({
          commentId,
          operation: pending.operation,
          ...(pending.updatedAt ? { updatedAt: pending.updatedAt } : {}),
        })
      }
      if (acknowledgements.length > 0) {
        const nextOutbox = { ...commentOutboxRef.current }
        for (const acknowledgement of acknowledgements) {
          const current = nextOutbox[acknowledgement.commentId]
          if (
            current?.operation === acknowledgement.operation
            && current.updatedAt === acknowledgement.updatedAt
          ) {
            delete nextOutbox[acknowledgement.commentId]
          }
        }
        commentOutboxRef.current = nextOutbox
        dispatch({ acknowledgements, type: 'acknowledgeEditorCommentOutbox' })
      }
      setError(null)
    } catch (caught) {
      setError(messageFromError(caught))
    } finally {
      flushingRef.current = false
      if (flushPendingRef.current) {
        flushPendingRef.current = false
        void flush()
      }
    }
  }, [pushDocument, pushFolder, pushComment])

  const flushDocumentForShare = useCallback(async (
    document: EditorDocumentRecord,
  ): Promise<{ metadataRevision: number; revision: number } | null> => {
    if (!syncActiveRef.current || !hydratedRef.current) return null
    // Serialize against the debounced autosave loop: wait for an in-flight
    // cycle, then hold the same lock while pushing this one document, so the
    // share flow can never race a concurrent PUT of the same record.
    while (flushingRef.current) {
      await new Promise<void>((resolve) => {
        window.setTimeout(resolve, 50)
      })
    }
    flushingRef.current = true
    try {
      const outcome = await pushDocument(document)
      if (outcome.kind !== 'saved' || outcome.metadataRevision === undefined) {
        return null
      }
      // Mark this fingerprint as synced so the next autosave cycle does not
      // re-push the identical body.
      syncedDocsRef.current.set(document.id, document.updatedAt)
      return {
        metadataRevision: outcome.metadataRevision,
        revision: outcome.revision,
      }
    } finally {
      flushingRef.current = false
      if (flushPendingRef.current) {
        flushPendingRef.current = false
        void flush()
      }
    }
  }, [flush, pushDocument])

  // -- reset + hydrate lifecycle (re-armed on project identity) ---------- #

  const reset = useCallback(() => {
    hydratedRef.current = false
    setHydrated(false)
    syncedDocsRef.current.clear()
    syncedFoldersRef.current.clear()
    syncedCommentsRef.current.clear()
    metadataRevisionsRef.current.clear()
    serverDocumentsRef.current.clear()
    loadedDocsRef.current.clear()
    exactDetailProvenanceRef.current.clear()
    loadedCommentsRef.current.clear()
  }, [])

  const hydrate = useCallback((token: SyncLifecycleToken) => {
    void (async () => {
      try {
        const options = optionsRef.current
        const serverFolders = await listEditorFolders(options)
        const folderRecords = serverFolders.map(folderRecordFromServer)
        const documentRecords: EditorDocumentRecord[] = []
        let cursor: string | undefined
        do {
          const page = await listEditorDocuments({
            ...options,
            cursor,
            limit: PAGE_LIMIT,
            scope: 'all',
          })
          for (const serverDocument of page.data) {
            documentRecords.push(documentRecordFromServer(serverDocument))
          }
          cursor = page.next_cursor ?? undefined
        } while (cursor)
        if (token.cancelled) return
        // Documents already in the local project carry an authoritative body
        // (loaded from the markdown). Captured BEFORE the merge dispatch so a
        // server-only document (added with body="") is distinguishable.
        const locallyPresentIds = new Set(Object.keys(documentsRef.current))
        if (folderRecords.length > 0) {
          dispatch({ folders: folderRecords, type: 'upsertServerEditorFolders' })
        }
        if (documentRecords.length > 0) {
          dispatch({ documents: documentRecords, type: 'upsertServerEditorDocuments' })
        }
        // Seed each fingerprint to WHAT THE SERVER HOLDS (its updated_at);
        // a local-newer entity then differs and the first autosave pushes it.
        for (const record of documentRecords) {
          serverDocumentsRef.current.set(record.id, record)
          if (record.access?.mode === 'shared') continue
          syncedDocsRef.current.set(record.id, record.updatedAt)
          metadataRevisionsRef.current.set(record.id, record.metadataRevision ?? 1)
          if (locallyPresentIds.has(record.id)) {
            // Its body is authoritative locally: never overwrite it on open
            // (a local-newer body must survive + be pushed up), and it may be
            // pushed directly without first fetching the server body.
            loadedDocsRef.current.add(record.id)
          }
        }
        for (const record of folderRecords) {
          syncedFoldersRef.current.set(record.id, record.updatedAt)
        }
        hydratedRef.current = true
        setHydrated(true)
        setError(null)
      } catch (caught) {
        if (!token.cancelled) setError(messageFromError(caught))
      }
    })()
  }, [dispatch])

  useProjectSyncLifecycle({
    active: syncActive,
    identity: `${workspaceId}:${projectEpoch}`,
    reset,
    hydrate,
  })

  // Authoritative re-hydration on user invalidations (sharing): a document
  // shared with this user must appear WITHOUT a page reload. No reset —
  // the hydrate upserts are updatedAt-guarded and re-seed fingerprints to
  // the server state, so local-newer edits survive and re-push.
  const lastRefreshTokenRef = useRef(refreshToken)
  useEffect(() => {
    if (
      refreshToken === undefined
      || refreshToken === lastRefreshTokenRef.current
    ) return
    lastRefreshTokenRef.current = refreshToken
    if (!syncActiveRef.current || !hydratedRef.current) return
    const token: SyncLifecycleToken = { cancelled: false }
    hydrate(token)
    return () => {
      token.cancelled = true
    }
  }, [hydrate, refreshToken])

  // -- load a document's body + comments on open ------------------------ #

  useEffect(() => {
    if (!syncActive || !hydrated || !selectedDocumentId) return
    // Only documents the server holds (hydrated) need a body/comment fetch;
    // a locally-authored document keeps its in-memory body authoritative.
    if (!serverDocumentsRef.current.has(selectedDocumentId)) return
    const documentId = selectedDocumentId
    const cachedDocument = serverDocumentsRef.current.get(documentId)
      ?? documentsRef.current[documentId]
    const hadLocalDocumentBody = loadedDocsRef.current.has(documentId)
    const hydrationPlan = planEditorOpenHydration({
      collaborationDocument: isCollaborationDocument(cachedDocument),
      hasCommentSnapshot: loadedCommentsRef.current.has(documentId),
      hasExactDocumentDetail: exactDetailProvenanceRef.current.get(documentId)
        === editorDocumentDetailProvenanceKey(cachedDocument),
      hasLocalDocumentBody: hadLocalDocumentBody,
    })
    if (!hydrationPlan.loadComments && !hydrationPlan.loadDocumentDetail) return
    if (hydrationPlan.loadDocumentDetail) loadedDocsRef.current.add(documentId)
    if (hydrationPlan.loadComments) loadedCommentsRef.current.add(documentId)
    let cancelled = false
    let applied = false
    void (async () => {
      try {
        let document = serverDocumentsRef.current.get(documentId)
          ?? documentsRef.current[documentId]
        if (hydrationPlan.loadDocumentDetail) {
          const detail = await getEditorDocument(documentId, optionsRef.current)
          if (cancelled) return
          const detailRecord = documentRecordFromServer(detail)
          serverDocumentsRef.current.set(documentId, detailRecord)
          const provenanceKey = editorDocumentDetailProvenanceKey(detailRecord)
          if (provenanceKey) {
            exactDetailProvenanceRef.current.set(documentId, provenanceKey)
          }
          loadedDocsRef.current.add(documentId)
          metadataRevisionsRef.current.set(
            documentId,
            detailRecord.metadataRevision ?? 1,
          )
          document = detailRecord
          if (isCollaborationDocument(detailRecord)) {
            dispatch({ document: detailRecord, type: 'setServerEditorDocumentDetail' })
          } else if (shouldLoadLegacyEditorBody(document)) {
            dispatch({
              contentMarkdown: detail.content_markdown ?? '',
              documentId,
              type: 'setServerEditorDocumentBody',
            })
          }
        }
        if (hydrationPlan.loadComments) {
          const comments: EditorCommentThreadRecord[] = []
          let cursor: string | undefined
          do {
            const page = await listEditorComments(documentId, {
              ...optionsRef.current,
              cursor,
              limit: PAGE_LIMIT,
            })
            for (const serverComment of page.data) {
              comments.push(commentRecordFromServer(serverComment))
            }
            cursor = page.next_cursor ?? undefined
          } while (cursor)
          if (cancelled) return
          const reconciliation = planEditorCommentReconciliation(
            documentId,
            commentsRef.current,
            comments,
            commentOutboxRef.current,
          )
          dispatch({
            comments: reconciliation.serverComments,
            documentId,
            preserveCommentIds: [...reconciliation.preserveCommentIds],
            type: 'reconcileServerEditorComments',
          })
          if (canPersistEditorCommentsForDocument(document)) {
            const serverCommentIds = new Set(comments.map((comment) => comment.id))
            for (const [commentId, fingerprint] of [...syncedCommentsRef.current]) {
              if (fingerprint.documentId !== documentId) continue
              if (
                reconciliation.pendingDeletedCommentIds.has(commentId)
                && serverCommentIds.has(commentId)
              ) continue
              syncedCommentsRef.current.delete(commentId)
            }
            for (const comment of reconciliation.serverComments) {
              syncedCommentsRef.current.set(comment.id, {
                documentId: comment.documentId,
                updatedAt: comment.updatedAt,
              })
            }
          }
        }
        applied = true
        setError(null)
      } catch (caught) {
        if (!cancelled) {
          if (hydrationPlan.loadDocumentDetail && !hadLocalDocumentBody) {
            loadedDocsRef.current.delete(documentId)
          }
          if (hydrationPlan.loadComments) loadedCommentsRef.current.delete(documentId)
          setError(messageFromError(caught))
        }
      }
    })()
    return () => {
      cancelled = true
      // If the load was interrupted before it applied (the user switched
      // documents mid-fetch), release the id so re-opening this document
      // re-fetches — otherwise its body would stay empty for the session.
      if (!applied) {
        if (hydrationPlan.loadDocumentDetail && !hadLocalDocumentBody) {
          loadedDocsRef.current.delete(documentId)
        }
        if (hydrationPlan.loadComments) loadedCommentsRef.current.delete(documentId)
      }
    }
  }, [syncActive, hydrated, selectedDocumentId, serverObservationEpoch, dispatch])

  // -- debounced autosave trigger --------------------------------------- #

  useEffect(() => {
    if (!syncActive || !hydrated) return
    const timer = setTimeout(() => {
      void flush()
    }, AUTOSAVE_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [
    editorCommentOutbox,
    editorComments,
    editorDocuments,
    editorFolders,
    flush,
    hydrated,
    syncActive,
  ])

  return { error, flushDocumentForShare, registerOpenedServerDocument }
}

function requireEditorMetadataRevision(revision: number | undefined): number {
  if (!Number.isSafeInteger(revision) || revision === undefined || revision < 1) {
    throw new Error('The server returned an invalid editor metadata revision.')
  }
  return revision
}
