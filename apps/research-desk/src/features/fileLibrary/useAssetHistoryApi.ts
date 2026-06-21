/**
 * File-library server sync (M6c project-persistence tier).
 *
 * The file-library counterpart of useEditorHistoryApi, sharing the same
 * shape: hydrate section + group + asset METADATA on mount and a debounced
 * serialized autosave that diffs three collections via the shared
 * syncCollection helper. Persistence only — it never calls a model.
 *
 * The one structural difference from the editor: an asset's heavy
 * ``extractedText`` has no per-asset "open" event to hang a load on (the
 * library has no document viewer). Instead the body loads ON USE via the
 * exposed ``ensureAssetBodiesLoaded`` primitive: the chat composer prefetches
 * it when a file chip is attached and awaits it before send, so a freshly
 * hydrated (body-less) asset never produces an empty attachment. The same
 * primitive backs the autosave's body-guard (never PUT a body="" over a real
 * server body) — exactly the editor's content_markdown lesson.
 *
 * It does NOT own the import button: the explicit opt-in push is the
 * project-level useProjectServerImport. This hook only hydrates + autosaves
 * once the project is opted in (syncActive), seeding its synced fingerprint
 * to WHAT THE SERVER HOLDS so a local-newer entity is pushed up rather than
 * stranded (the M6a P1 lesson).
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch } from 'react'

import {
  deleteAsset,
  deleteAssetGroup,
  deleteAssetSection,
  getAsset,
  hasHttpStatus,
  listAssetGroups,
  listAssetSections,
  listAssets,
  saveAsset,
  saveAssetGroup,
  saveAssetSection,
} from '@/api/inqtrixClient'
import {
  assetRecordFromServer,
  groupRecordFromServer,
  sectionRecordFromServer,
  serverAssetPayload,
  serverGroupPayload,
  serverSectionPayload,
} from '@/features/fileLibrary/assetSync'
import type {
  FileAssetRecord,
  FileGroupRecord,
  FileLibrarySectionRecord,
} from '@/features/project/types'
import { deleteTolerant404, syncCollection } from '@/features/project/syncCollection'
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

/** Swallow a 404 on delete (the section-cascade case): see deleteTolerant404. */
function deleteTolerant(run: () => Promise<void>): Promise<void> {
  return deleteTolerant404(run, (error) => hasHttpStatus(error, 404))
}

type UseAssetHistoryApiOptions = {
  apiKey: string | undefined
  dispatch: Dispatch<ResearchDeskAction>
  fileAssets: Record<string, FileAssetRecord>
  fileGroups: Record<string, FileGroupRecord>
  fileLibrarySections: Record<string, FileLibrarySectionRecord>
  /** In-session load counter (bumped on every wholesale project replace). Part
   * of the lifecycle identity so a switch to another synced project re-hydrates
   * from its own server state instead of inheriting this one's synced map. */
  projectEpoch: number
  /** ``serverSyncEnabled`` AND the durable capability AND not demo. */
  syncActive: boolean
  workspaceId: string
}

export type AssetHistoryApiHandle = {
  error: string | null
  /** Resolve the extractedText of the given assets, fetching server-only
   * bodies on demand (deduped), and return them keyed by id. Dispatches each
   * fetched body into state too, so later reads see it. Call it fire-and-
   * forget to prefetch (chip attach) and awaited to guarantee presence before
   * a synchronous body consumer (chat send, index build). */
  ensureAssetBodiesLoaded: (assetIds: readonly string[]) => Promise<Map<string, string>>
}

export function useAssetHistoryApi({
  apiKey,
  dispatch,
  fileAssets,
  fileGroups,
  fileLibrarySections,
  projectEpoch,
  syncActive,
  workspaceId,
}: UseAssetHistoryApiOptions): AssetHistoryApiHandle {
  const [error, setError] = useState<string | null>(null)
  const [hydrated, setHydrated] = useState(false)

  const assetsRef = useRef(fileAssets)
  assetsRef.current = fileAssets
  const groupsRef = useRef(fileGroups)
  groupsRef.current = fileGroups
  const sectionsRef = useRef(fileLibrarySections)
  sectionsRef.current = fileLibrarySections
  const optionsRef = useRef({ apiKey, workspaceId })
  optionsRef.current = { apiKey, workspaceId }

  const syncedSectionsRef = useRef(new Map<string, string>())
  const syncedGroupsRef = useRef(new Map<string, string>())
  const syncedAssetsRef = useRef(new Map<string, string>())
  // Assets whose extractedText is authoritative locally (loaded on use,
  // locally-present at hydrate, or just pushed) — never to be overwritten on
  // a later body fetch, and safe to PUT without first reading the server body.
  const loadedAssetsRef = useRef(new Set<string>())
  // De-dupe concurrent body fetches per asset id (prefetch + send-await +
  // autosave guard can all want the same body at once).
  const inFlightBodyRef = useRef(new Map<string, Promise<string>>())
  const hydratedRef = useRef(false)
  const flushingRef = useRef(false)
  const flushPendingRef = useRef(false)
  const syncActiveRef = useRef(syncActive)
  syncActiveRef.current = syncActive

  // -- body load-on-use (deduped) --------------------------------------- #

  const loadBody = useCallback((assetId: string): Promise<string> => {
    const existing = inFlightBodyRef.current.get(assetId)
    if (existing) return existing
    const promise = (async () => {
      const detail = await getAsset(assetId, optionsRef.current)
      const extractedText = detail.extracted_text ?? ''
      dispatch({ assetId, extractedText, type: 'setServerAssetBody' })
      loadedAssetsRef.current.add(assetId)
      return extractedText
    })()
    inFlightBodyRef.current.set(assetId, promise)
    void promise.finally(() => {
      if (inFlightBodyRef.current.get(assetId) === promise) {
        inFlightBodyRef.current.delete(assetId)
      }
    })
    return promise
  }, [dispatch])

  const ensureAssetBodiesLoaded = useCallback(
    async (assetIds: readonly string[]): Promise<Map<string, string>> => {
      const result = new Map<string, string>()
      await Promise.all(
        [...new Set(assetIds)].map(async (id) => {
          const asset = assetsRef.current[id]
          if (!asset) return
          // A body is authoritative locally when the server does not know the
          // asset (locally created), or it was already loaded — no fetch.
          if (!syncedAssetsRef.current.has(id) || loadedAssetsRef.current.has(id)) {
            result.set(id, asset.extractedText)
            return
          }
          result.set(id, await loadBody(id))
        }),
      )
      return result
    },
    [loadBody],
  )

  // -- pushing one entity ----------------------------------------------- #

  const pushAsset = useCallback(async (asset: FileAssetRecord) => {
    let record = asset
    if (
      syncedAssetsRef.current.has(asset.id) &&
      !loadedAssetsRef.current.has(asset.id)
    ) {
      // A metadata-only edit (group-orphan on group delete, section move)
      // bumped this server-held asset's updatedAt while its body was never
      // loaded (still ""). The PUT is a full-record upsert, so sending
      // extractedText="" would ERASE the server body. Fetch it first and push
      // the merged record (new metadata + kept body).
      const extractedText = await loadBody(asset.id)
      record = { ...asset, extractedText }
    }
    await saveAsset(record.id, serverAssetPayload(record), optionsRef.current)
    loadedAssetsRef.current.add(record.id)
  }, [loadBody])

  const pushSection = useCallback(async (section: FileLibrarySectionRecord) => {
    await saveAssetSection(section.id, serverSectionPayload(section), optionsRef.current)
  }, [])

  const pushGroup = useCallback(async (group: FileGroupRecord) => {
    await saveAssetGroup(group.id, serverGroupPayload(group), optionsRef.current)
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
      // Parent-first so a freshly created section exists before its groups and
      // a group before its assets (the FK order). On delete the order is
      // harmless: a section delete cascades its groups + assets server-side,
      // and the child passes below tolerate the resulting 404s.
      await syncCollection<FileLibrarySectionRecord, string>({
        current: sectionsRef.current,
        synced: syncedSectionsRef.current,
        fingerprintOf: (section) => section.updatedAt,
        changed: (previous, current) => previous !== current,
        pushOne: pushSection,
        deleteOne: (id) =>
          deleteTolerant(() => deleteAssetSection(id, optionsRef.current)),
      })
      await syncCollection<FileGroupRecord, string>({
        current: groupsRef.current,
        synced: syncedGroupsRef.current,
        fingerprintOf: (group) => group.updatedAt,
        changed: (previous, current) => previous !== current,
        pushOne: pushGroup,
        deleteOne: (id) =>
          deleteTolerant(() => deleteAssetGroup(id, optionsRef.current)),
      })
      await syncCollection<FileAssetRecord, string>({
        current: assetsRef.current,
        synced: syncedAssetsRef.current,
        fingerprintOf: (asset) => asset.updatedAt,
        changed: (previous, current) => previous !== current,
        pushOne: pushAsset,
        deleteOne: async (id) => {
          await deleteTolerant(() => deleteAsset(id, optionsRef.current))
          loadedAssetsRef.current.delete(id)
          inFlightBodyRef.current.delete(id)
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
  }, [pushSection, pushGroup, pushAsset])

  // -- reset + hydrate lifecycle (re-armed on project identity) ---------- #

  const reset = useCallback(() => {
    hydratedRef.current = false
    setHydrated(false)
    syncedSectionsRef.current.clear()
    syncedGroupsRef.current.clear()
    syncedAssetsRef.current.clear()
    loadedAssetsRef.current.clear()
    inFlightBodyRef.current.clear()
  }, [])

  const hydrate = useCallback((token: SyncLifecycleToken) => {
    void (async () => {
      try {
        const options = optionsRef.current
        const sectionRecords = (await listAssetSections(options)).map(sectionRecordFromServer)
        const groupRecords = (await listAssetGroups(options)).map(groupRecordFromServer)
        const assetRecords: FileAssetRecord[] = []
        let cursor: string | undefined
        do {
          const page = await listAssets({ ...options, cursor, limit: PAGE_LIMIT })
          for (const serverAsset of page.data) {
            assetRecords.push(assetRecordFromServer(serverAsset))
          }
          cursor = page.next_cursor ?? undefined
        } while (cursor)
        if (token.cancelled) return
        // Assets already in the local project carry an authoritative body
        // (loaded from the markdown). Captured BEFORE the merge dispatch so a
        // server-only asset (hydrated with extractedText="") is distinguishable.
        const locallyPresentIds = new Set(Object.keys(assetsRef.current))
        if (sectionRecords.length > 0) {
          dispatch({ sections: sectionRecords, type: 'upsertServerAssetSections' })
        }
        if (groupRecords.length > 0) {
          dispatch({ groups: groupRecords, type: 'upsertServerAssetGroups' })
        }
        if (assetRecords.length > 0) {
          dispatch({ assets: assetRecords, type: 'upsertServerAssetMetadata' })
        }
        // Seed each fingerprint to WHAT THE SERVER HOLDS (its updatedAt); a
        // local-newer entity then differs and the first autosave pushes it.
        for (const record of sectionRecords) {
          syncedSectionsRef.current.set(record.id, record.updatedAt)
        }
        for (const record of groupRecords) {
          syncedGroupsRef.current.set(record.id, record.updatedAt)
        }
        for (const record of assetRecords) {
          syncedAssetsRef.current.set(record.id, record.updatedAt)
          if (locallyPresentIds.has(record.id)) {
            // Its body is authoritative locally: never overwrite it on a later
            // fetch, and it may be pushed without first reading the server body.
            loadedAssetsRef.current.add(record.id)
          }
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

  // -- debounced autosave trigger --------------------------------------- #

  useEffect(() => {
    if (!syncActive || !hydrated) return
    const timer = setTimeout(() => {
      void flush()
    }, AUTOSAVE_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [fileLibrarySections, fileGroups, fileAssets, syncActive, hydrated, flush])

  return { error, ensureAssetBodiesLoaded }
}
