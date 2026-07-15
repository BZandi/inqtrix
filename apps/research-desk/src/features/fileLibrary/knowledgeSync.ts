import {
  addKnowledgeDocument,
  createKnowledgeCollection,
  deleteKnowledgeCollection,
  fetchKnowledgeDocumentText,
  ingestKnowledgeFile,
} from '@/api/inqtrixClient'
import type { FileAssetRecord, VectorIndexRecord } from '@/features/project/types'

/** Connection facts needed to talk to the knowledge backend. */
export type KnowledgeSyncOptions = {
  apiKey?: string
  /**
   * Prefer server-side file ingestion (`file_id`) over browser-extracted
   * text for assets that carry a `serverFileId`. Set when the backend
   * advertises `features.document_parser` — the server then parses the
   * ORIGINAL file (better PDF/DOCX fidelity than browser extraction).
   */
  useFileIngestion?: boolean
  workspaceId?: string
}

export type KnowledgeReindexResult = {
  collectionId: string
  /** Embedding model the collection was built/extended with — persisted on the
   * index so a later reindex can tell "model changed" from "docs added". */
  serverCollectionModel: string
  /** Asset ids actually ingested + embedded this run (the COMPLETE truth for a
   * rebuild; the newly-added subset for an incremental ingest). */
  embeddedFileIds: string[]
  /** Asset ids skipped because no extracted text was available (never silently
   * dropped — the caller leaves them pending so the index reads honestly). */
  skippedFileIds: string[]
  /** Labels of the skipped members (back-compat surface for messaging). */
  skippedFiles: string[]
  uploadedDocuments: number
  /**
   * Assets the server (MarkItDown) parsed during this run, with the
   * resulting text. The caller back-fills these into the library so the
   * stored `extractedText` + `parserId` upgrade from the fast client parse to
   * the higher-fidelity server parse (and a later re-index reuses it).
   */
  reparsed: { assetId: string; text: string }[]
  /** fileId -> backend knowledge-document id for each member ingested this run
   * (persisted on the member so a later removal can delete the exact doc). */
  serverDocumentIds: Record<string, string>
}

/** Whether an asset should be ingested by `file_id` (server re-parses the
 * original) rather than by its stored text. When the stored text is already
 * `markitdown`-grade the upload would just re-produce it, so embed it directly
 * and skip the redundant S3 fetch + parse. */
function canUseFile(asset: FileAssetRecord, options: KnowledgeSyncOptions): boolean {
  return Boolean(
    options.useFileIngestion && asset.serverFileId && asset.parserId !== 'markitdown',
  )
}

export type KnowledgeAssetIngestResult = {
  embeddedFileIds: string[]
  skippedFileIds: string[]
  skippedFiles: string[]
  reparsed: { assetId: string; text: string }[]
  /** fileId -> backend knowledge-document id for each ingested member, so the
   * caller can persist it on the member (enables single-doc removal later). */
  serverDocumentIds: Record<string, string>
}

/** Per-member progress callback fired AFTER each member's server-confirmed
 * ingest (or skip), so the caller can advance the progress bar + flip that
 * file row to its real outcome — genuine server feedback, not cosmetic. */
export type MemberProgress = (event: {
  fileId: string
  done: number
  total: number
  embedded: boolean
}) => void

/** Ingest a list of member assets into an EXISTING collection id. Defined once
 * (design principle 4) and shared by the rebuild-from-scratch path and the
 * incremental add path. Members with neither usable text nor file ingestion are
 * reported back (never silently dropped), including a rebuild where every
 * member is terminally skipped and the fresh server collection stays empty. */
async function ingestMembersIntoCollection(
  collectionId: string,
  memberAssets: FileAssetRecord[],
  options: KnowledgeSyncOptions,
  onMemberDone?: MemberProgress,
): Promise<KnowledgeAssetIngestResult> {
  const embeddedFileIds: string[] = []
  const skippedFileIds: string[] = []
  const skippedFiles: string[] = []
  const reparsed: { assetId: string; text: string }[] = []
  const serverDocumentIds: Record<string, string> = {}
  const total = memberAssets.length
  let done = 0
  for (const asset of memberAssets) {
    const useFile = canUseFile(asset, options)
    if (!useFile && asset.extractedText.trim().length === 0) {
      skippedFileIds.push(asset.id)
      skippedFiles.push(asset.label)
    } else if (useFile) {
      const document = await ingestKnowledgeFile(
        collectionId,
        {
          fileId: asset.serverFileId as string,
          // `fileId` = the local asset id (member mapping); `file_id` = the
          // SERVER file id (when uploaded) so the knowledge citation viewer can
          // load the original PDF for the page-jump. Omitted when there is no
          // server file (text-only docs) → the viewer shows no source PDF.
          metadata: {
            fileId: asset.id,
            fileName: asset.fileName,
            ...(asset.serverFileId ? { file_id: asset.serverFileId } : {}),
          },
          title: asset.title,
        },
        options,
      )
      embeddedFileIds.push(asset.id)
      serverDocumentIds[asset.id] = document.id
      // The server just parsed the original with MarkItDown — read that text
      // back so the caller can upgrade the library asset's text + provenance.
      // Best-effort: indexing already succeeded, so a failed read only forgoes
      // the label upgrade, it does not fail the run.
      try {
        const parsedText = await fetchKnowledgeDocumentText(document.id, options)
        reparsed.push({ assetId: asset.id, text: parsedText.text })
      } catch {
        // Leave provenance as 'client'; the embeddings are still server-grade.
      }
    } else {
      const document = await addKnowledgeDocument(
        collectionId,
        {
          // `fileId` = the local asset id (member mapping); `file_id` = the
          // SERVER file id (when uploaded) so the knowledge citation viewer can
          // load the original PDF for the page-jump. Omitted when there is no
          // server file (text-only docs) → the viewer shows no source PDF.
          metadata: {
            fileId: asset.id,
            fileName: asset.fileName,
            ...(asset.serverFileId ? { file_id: asset.serverFileId } : {}),
          },
          text: asset.extractedText,
          title: asset.title,
        },
        options,
      )
      embeddedFileIds.push(asset.id)
      serverDocumentIds[asset.id] = document.id
    }
    // Report this member's REAL, server-confirmed outcome so the caller can
    // advance the progress bar + flip the file row live (never cosmetic).
    done += 1
    onMemberDone?.({
      fileId: asset.id,
      done,
      total,
      embedded: !skippedFileIds.includes(asset.id),
    })
  }
  return { embeddedFileIds, skippedFileIds, skippedFiles, reparsed, serverDocumentIds }
}

/** Create the server collection for a not-yet-built local vector index.

Once a collection exists its identity and embedding model are immutable from
this local setup surface. Refreshes run in place through the indexing-job API;
this function must therefore never delete or replace an existing collection
(which would revoke shares and invalidate every server reference). */
export async function createVectorIndexCollectionOnServer(
  index: VectorIndexRecord,
  memberAssets: FileAssetRecord[],
  options: KnowledgeSyncOptions,
  onMemberDone?: MemberProgress,
): Promise<KnowledgeReindexResult> {
  if (index.serverCollectionId) {
    throw new Error(
      'createVectorIndexCollectionOnServer requires an index without a server collection.',
    )
  }

  const collection = await createKnowledgeCollection(
    { embeddingModel: index.model, name: index.title },
    options,
  )

  let ingest: KnowledgeAssetIngestResult
  try {
    ingest = await ingestMembersIntoCollection(
      collection.id, memberAssets, options, onMemberDone,
    )
  } catch (error) {
    // A half-filled collection must not pose as a complete index:
    // remove it so the visible outcome is "failed", not "ready-ish".
    try {
      await deleteKnowledgeCollection(collection.id, options)
    } catch {
      // Cleanup is best-effort; the original error is the one that matters.
    }
    throw error
  }

  return {
    collectionId: collection.id,
    serverCollectionModel: index.model,
    embeddedFileIds: ingest.embeddedFileIds,
    skippedFileIds: ingest.skippedFileIds,
    skippedFiles: ingest.skippedFiles,
    uploadedDocuments: ingest.embeddedFileIds.length,
    reparsed: ingest.reparsed,
    serverDocumentIds: ingest.serverDocumentIds,
  }
}

/** Add local assets to an existing server collection. Used by both an owner's
 * canonical collection view and an accepted editor share; no local VectorIndex
 * record is created for the recipient. */
export async function ingestAssetsIntoKnowledgeCollection(
  collectionId: string,
  assets: FileAssetRecord[],
  options: KnowledgeSyncOptions,
  onMemberDone?: MemberProgress,
): Promise<KnowledgeAssetIngestResult> {
  return ingestMembersIntoCollection(collectionId, assets, options, onMemberDone)
}

/** Incrementally ingest the index's newly-added (pending) members into its
EXISTING server collection — no delete/recreate, no re-embedding of documents
already present. Used when documents are added to an already-built index and the
embedding model is unchanged: only the new members are uploaded, so a small add
no longer triggers a full rebuild (and the prior bug where added documents were
never ingested at all). Members without text are reported, not dropped; partial
success is fine (the caller marks only the embedded ones). */
export async function ingestNewVectorIndexMembers(
  index: VectorIndexRecord,
  pendingMemberAssets: FileAssetRecord[],
  options: KnowledgeSyncOptions,
  onMemberDone?: MemberProgress,
): Promise<KnowledgeReindexResult> {
  if (!index.serverCollectionId) {
    throw new Error(
      'ingestNewVectorIndexMembers requires an existing server collection.',
    )
  }
  const ingest = await ingestMembersIntoCollection(
    index.serverCollectionId,
    pendingMemberAssets,
    options,
    onMemberDone,
  )
  return {
    collectionId: index.serverCollectionId,
    serverCollectionModel: index.serverCollectionModel ?? index.model,
    embeddedFileIds: ingest.embeddedFileIds,
    skippedFileIds: ingest.skippedFileIds,
    skippedFiles: ingest.skippedFiles,
    uploadedDocuments: ingest.embeddedFileIds.length,
    reparsed: ingest.reparsed,
    serverDocumentIds: ingest.serverDocumentIds,
  }
}
