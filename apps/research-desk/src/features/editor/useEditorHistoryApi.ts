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
  saveEditorComments,
  saveEditorDocument,
  saveEditorFolder,
} from '@/api/inqtrixClient'
import {
  commentRecordFromServer,
  documentRecordFromServer,
  folderRecordFromServer,
  serverCommentPayload,
  serverDocumentPayload,
  serverFolderPayload,
} from '@/features/editor/editorSync'
import type {
  EditorCommentThreadRecord,
  EditorDocumentRecord,
  EditorFolderRecord,
} from '@/features/project/types'
import { syncCollection } from '@/features/project/syncCollection'
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

type UseEditorHistoryApiOptions = {
  apiKey: string | undefined
  dispatch: Dispatch<ResearchDeskAction>
  editorComments: Record<string, EditorCommentThreadRecord>
  editorDocuments: Record<string, EditorDocumentRecord>
  editorFolders: Record<string, EditorFolderRecord>
  /** In-session load counter (bumped on every wholesale project replace). Part
   * of the lifecycle identity so a switch to another synced project re-hydrates
   * from its own server state instead of inheriting this one's synced map. */
  projectEpoch: number
  selectedDocumentId: string | null
  /** ``serverSyncEnabled`` AND the durable capability AND not demo. */
  syncActive: boolean
  workspaceId: string
}

export type EditorHistoryApiHandle = {
  error: string | null
}

export function useEditorHistoryApi({
  apiKey,
  dispatch,
  editorComments,
  editorDocuments,
  editorFolders,
  projectEpoch,
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
  const optionsRef = useRef({ apiKey, workspaceId })
  optionsRef.current = { apiKey, workspaceId }

  const syncedDocsRef = useRef(new Map<string, string>())
  const syncedFoldersRef = useRef(new Map<string, string>())
  const syncedCommentsRef = useRef(new Map<string, CommentFingerprint>())
  const loadedDocsRef = useRef(new Set<string>())
  const hydratedRef = useRef(false)
  const flushingRef = useRef(false)
  const flushPendingRef = useRef(false)
  const syncActiveRef = useRef(syncActive)
  syncActiveRef.current = syncActive

  // -- pushing one entity ----------------------------------------------- #

  const pushDocument = useCallback(async (document: EditorDocumentRecord) => {
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
      return
    }
    // We hold this document's body now, so re-opening it must not trigger a
    // redundant body fetch.
    loadedDocsRef.current.add(record.id)
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
        current: documentsRef.current,
        synced: syncedDocsRef.current,
        fingerprintOf: (document) => document.updatedAt,
        changed: (previous, current) => previous !== current,
        pushOne: pushDocument,
        deleteOne: async (id) => {
          await deleteEditorDocument(id, optionsRef.current)
          loadedDocsRef.current.delete(id)
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
        deleteOne: (id) => deleteEditorFolder(id, optionsRef.current),
      })
      await syncCollection<EditorCommentThreadRecord, CommentFingerprint>({
        current: commentsRef.current,
        synced: syncedCommentsRef.current,
        fingerprintOf: (comment) => ({
          documentId: comment.documentId,
          updatedAt: comment.updatedAt,
        }),
        changed: (previous, current) =>
          previous === undefined || previous.updatedAt !== current.updatedAt,
        pushOne: pushComment,
        deleteOne: async (commentId) => {
          // The comment is gone from `current`; its parent doc id is still
          // in the synced fingerprint (deleted by syncCollection after this).
          const documentId = syncedCommentsRef.current.get(commentId)?.documentId
          if (documentId) {
            await deleteEditorComment(documentId, commentId, optionsRef.current)
          }
        },
      })
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

  // -- reset + hydrate lifecycle (re-armed on project identity) ---------- #

  const reset = useCallback(() => {
    hydratedRef.current = false
    setHydrated(false)
    syncedDocsRef.current.clear()
    syncedFoldersRef.current.clear()
    syncedCommentsRef.current.clear()
    loadedDocsRef.current.clear()
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
          syncedDocsRef.current.set(record.id, record.updatedAt)
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

  // -- load a document's body + comments on open ------------------------ #

  useEffect(() => {
    if (!syncActive || !hydrated || !selectedDocumentId) return
    // Only documents the server holds (hydrated) need a body/comment fetch;
    // a locally-authored document keeps its in-memory body authoritative.
    if (!syncedDocsRef.current.has(selectedDocumentId)) return
    if (loadedDocsRef.current.has(selectedDocumentId)) return
    loadedDocsRef.current.add(selectedDocumentId)
    const documentId = selectedDocumentId
    let cancelled = false
    let applied = false
    void (async () => {
      try {
        const detail = await getEditorDocument(documentId, optionsRef.current)
        if (cancelled) return
        dispatch({
          contentMarkdown: detail.content_markdown ?? '',
          documentId,
          type: 'setServerEditorDocumentBody',
        })
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
        if (comments.length > 0) {
          dispatch({ comments, type: 'upsertServerEditorComments' })
        }
        for (const comment of comments) {
          syncedCommentsRef.current.set(comment.id, {
            documentId: comment.documentId,
            updatedAt: comment.updatedAt,
          })
        }
        applied = true
        setError(null)
      } catch (caught) {
        if (!cancelled) {
          loadedDocsRef.current.delete(documentId)
          setError(messageFromError(caught))
        }
      }
    })()
    return () => {
      cancelled = true
      // If the load was interrupted before it applied (the user switched
      // documents mid-fetch), release the id so re-opening this document
      // re-fetches — otherwise its body would stay empty for the session.
      if (!applied) loadedDocsRef.current.delete(documentId)
    }
  }, [syncActive, hydrated, selectedDocumentId, dispatch])

  // -- debounced autosave trigger --------------------------------------- #

  useEffect(() => {
    if (!syncActive || !hydrated) return
    const timer = setTimeout(() => {
      void flush()
    }, AUTOSAVE_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [editorDocuments, editorFolders, editorComments, syncActive, hydrated, flush])

  return { error }
}
