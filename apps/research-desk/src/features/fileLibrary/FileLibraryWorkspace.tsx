import { useEffect, useMemo, useRef, useState, type DragEvent, type Dispatch } from 'react'
import { ChevronRight, Folder, FolderOpen, Inbox, Plus, Upload } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import {
  fileAssetReferenceCount,
  projectFileAssets,
  projectFileGroups,
  projectFileLibrarySections,
  projectStorageTotalBytes,
  projectVectorIndexes,
  vectorIndexById,
  vectorIndexMembersResolved,
} from '@/features/project/selectors'
import type { EmbedModelDescriptor, EmbedModelId, FileAssetRecord, ProjectState } from '@/features/project/types'
import { createDefaultFileParser } from '@/features/files/parsing'
import { ingestFiles, type ServerFileUpload } from '@/features/files/ingest'
import { Dropzone } from '@/features/files/Dropzone'
import { FILE_SECTION_TEMP_ID } from '@/features/files/sections'
import type { ResearchDeskAction } from '../researchDesk/state'
import {
  ingestNewVectorIndexMembers,
  reindexVectorIndexOnServer,
  type KnowledgeReindexResult,
  type KnowledgeSyncOptions,
  type MemberProgress,
} from './knowledgeSync'
import { useIndexingJobApi } from './useIndexingJobApi'
import { Rail } from './Rail'
import { useEmbeddingQuota } from '@/features/quota/useEmbeddingQuota'
import { IndexBar } from './IndexBar'
import { AddDocsPanel } from './AddDocsPanel'
import { FileCard, FileRow } from './FileItem'
import { FilePreviewPanel } from './FilePreviewPanel'
import { deleteKnowledgeDocument, type ClientOptions } from '@/api/inqtrixClient'
import { ConfirmDelete, InlineText, SortSelect, ViewToggle, type MoveTarget } from './controls'
import { groupSlug, isMemberInRun } from './helpers'
import { isInternalFileDrag, type ActiveTarget, type SortMode, type ViewMode } from './constants'

const parser = createDefaultFileParser()

type UploadTarget = { groupId: string | null; indexId?: string; sectionId: string }

type Band = {
  count: number
  groupId: string | null
  isGroup: boolean
  sectionId: string
  title: string
}

type Block = {
  band: Band | null
  breadcrumb: boolean
  items: FileAssetRecord[]
  key: string
}

function BandHeader({
  band,
  dropOver,
  onDeleteGroup,
  onDrop,
  onDragLeave,
  onDragOver,
  onNavigate,
  onRenameGroup,
  onUpload,
}: {
  band: Band
  dropOver: boolean
  onDeleteGroup: (groupId: string) => void
  onDrop: (event: DragEvent) => void
  onDragLeave: () => void
  onDragOver: (event: DragEvent) => void
  onNavigate?: () => void
  onRenameGroup: (groupId: string, title: string) => void
  onUpload: () => void
}) {
  const { t } = useLocale()
  return (
    <div
      className={cn(
        'group/band mt-3 flex min-h-8 items-center gap-2 rounded-md border-b border-border/60 px-2 py-1 transition-colors',
        dropOver && 'bg-brand-subtle/60 ring-1 ring-brand/30',
      )}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      {/* 28px (w-7) slot so the folder glyph centers exactly over the rows' type tiles. */}
      <span className="grid w-7 shrink-0 place-items-center">
        {band.isGroup ? <FolderOpen className="size-3.5 text-file" /> : <Folder className="size-3.5 text-muted-foreground" />}
      </span>
      {band.isGroup ? (
        <InlineText
          ariaLabel={t.fileLibrary.renameGroup}
          className="t-list text-foreground"
          onCommit={(title) => onRenameGroup(band.groupId as string, title)}
          value={band.title}
        />
      ) : onNavigate ? (
        <button className="inline-flex items-center gap-1 rounded-sm px-1 t-list text-foreground hover:bg-accent/60" onClick={onNavigate} type="button">
          {band.title}
          <ChevronRight className="size-3 text-muted-foreground" />
        </button>
      ) : (
        <span className="t-list text-foreground">{band.title}</span>
      )}
      <span className="shrink-0 rounded-full border border-border bg-surface/45 px-1.5 t-hint font-semibold leading-4 tabular-nums text-muted-foreground">{band.count}</span>
      {band.isGroup ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="hidden shrink-0 cursor-help items-center gap-1 rounded border border-border bg-surface px-1.5 py-0.5 font-mono t-hint text-muted-foreground sm:inline-flex">
              @filegroups:{groupSlug(band.title)}
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-[240px]" side="top">{t.fileLibrary.groupBundleTooltip}</TooltipContent>
        </Tooltip>
      ) : null}
      <div className="ml-auto flex items-center gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover/band:opacity-100">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button aria-label={t.fileLibrary.upload} className="size-7 text-muted-foreground hover:text-foreground" onClick={onUpload} size="icon" type="button" variant="ghost">
              <Upload className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="top">{t.fileLibrary.upload}</TooltipContent>
        </Tooltip>
        {band.isGroup ? (
          <ConfirmDelete ariaLabel={t.fileLibrary.removeGroup} hint={t.fileLibrary.removeGroupHint} onConfirm={() => onDeleteGroup(band.groupId as string)} />
        ) : null}
      </div>
    </div>
  )
}

function EmptyState({ onUpload, searching }: { onUpload: () => void; searching: boolean }) {
  const { t } = useLocale()
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border px-8 py-16 text-center">
      <span className="grid size-14 place-items-center rounded-2xl border border-border bg-surface text-muted-foreground">
        <Inbox className="size-6" />
      </span>
      <h3 className="mt-4 text-sm font-semibold text-foreground">{searching ? t.fileLibrary.emptySearchTitle : t.fileLibrary.emptyLibraryTitle}</h3>
      <p className="mt-1 max-w-xs text-xs leading-5 text-muted-foreground">{searching ? t.fileLibrary.emptySearchHint : t.fileLibrary.emptyLibraryHint}</p>
      {searching ? null : (
        <Button className="mt-4 gap-1.5" onClick={onUpload} size="sm" type="button" variant="outline">
          <Upload className="size-4" />
          {t.fileLibrary.uploadDocuments}
        </Button>
      )}
    </div>
  )
}

function IndexEmpty({ onAdd }: { onAdd: () => void }) {
  const { t } = useLocale()
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border px-8 py-14 text-center">
      <span className="grid size-14 place-items-center rounded-2xl border border-file/25 bg-file-subtle text-file">
        <Inbox className="size-6" />
      </span>
      <h3 className="mt-4 text-sm font-semibold text-foreground">{t.vectorIndex.indexEmptyTitle}</h3>
      <p className="mt-1 max-w-sm text-xs leading-5 text-muted-foreground">{t.vectorIndex.indexEmptyHint}</p>
      <Button className="mt-4 gap-1.5 bg-brand text-brand-foreground hover:bg-brand/90" onClick={onAdd} size="sm" type="button">
        <Plus className="size-4" />
        {t.vectorIndex.addDocuments}
      </Button>
    </div>
  )
}

export function FileLibraryWorkspace({
  dispatch,
  embedModels,
  ensureAssetBodiesLoaded,
  fileApiOptions,
  knowledgeSync,
  onAssetsIngested,
  serverFeatureLabels = null,
  serverFileUpload,
  state,
}: {
  dispatch: Dispatch<ResearchDeskAction>
  /** Active embedding catalog: server-provided when the knowledge engine
   * is enabled, the EMBED_MODELS fallback in demo/offline modes. */
  embedModels: readonly EmbedModelDescriptor[]
  /** Loads asset bodies on demand before a first-build reindex reads them
   * (M6c load-on-use). Absent in demo/offline — bodies are always local then. */
  ensureAssetBodiesLoaded?: (assetIds: readonly string[]) => Promise<Map<string, string>>
  /** Server options for the file preview (asset body + original download),
   * gated on the FILES/persistence tier (not knowledge); `null` in demo/offline.
   * Whether the Original tab is usable is then `serverFileId && fileApiOptions`. */
  fileApiOptions: ClientOptions | null
  /** Connection facts for real server-side embedding runs; `null` keeps
   * the historical client-side simulation (demo/offline). */
  knowledgeSync: KnowledgeSyncOptions | null
  /** Kicks off the non-blocking background server (MarkItDown) parse for
   * just-ingested assets, upgrading the instant client parse. No-op without
   * a server parser. */
  onAssetsIngested?: (assets: FileAssetRecord[]) => void
  /** Labels of active server features for the visible mode indicator;
   * `null` hides the line (demo or no server connected). */
  serverFeatureLabels?: string[] | null
  /** Uploads the ORIGINAL file to the server file store when the
   * backend advertises `features.files`; absent = local-only mode. */
  serverFileUpload?: ServerFileUpload
  state: ProjectState
}) {
  const { locale, t } = useLocale()
  const embeddingQuota = useEmbeddingQuota()
  const [active, setActive] = useState<ActiveTarget>({ kind: 'all' })
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortMode>('recent')
  const [view, setView] = useState<ViewMode>('list')
  const [dropKey, setDropKey] = useState<string | null>(null)
  const [pickerIndexId, setPickerIndexId] = useState<string | null>(null)
  const [previewAssetId, setPreviewAssetId] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const targetRef = useRef<UploadTarget>({ groupId: null, sectionId: FILE_SECTION_TEMP_ID })
  const reindexTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  const selectNewestIndex = useRef(false)

  useEffect(() => {
    const timers = reindexTimers.current
    return () => {
      timers.forEach((timer) => clearInterval(timer))
      timers.clear()
    }
  }, [])

  const sections = projectFileLibrarySections(state)
  const groups = projectFileGroups(state)
  const assets = projectFileAssets(state)
  const indexes = projectVectorIndexes(state)

  const assetsInSection = (sectionId: string) => assets.filter((asset) => asset.sectionId === sectionId)
  const customCollections = sections.filter((section) => section.kind === 'custom')
  const railCollections = sections.filter((section) => section.kind === 'custom' || assetsInSection(section.id).length > 0)

  // Reset selection if the active collection/index was deleted.
  useEffect(() => {
    if (active.kind === 'collection' && !sections.some((section) => section.id === active.sectionId)) setActive({ kind: 'all' })
    if (active.kind === 'index' && !indexes.some((index) => index.id === active.indexId)) {
      setActive({ kind: 'all' })
      setPickerIndexId(null)
    }
  }, [active, indexes, sections])

  // After creating an index, select it and open its add-documents panel.
  useEffect(() => {
    if (selectNewestIndex.current && indexes.length > 0) {
      selectNewestIndex.current = false
      setActive({ indexId: indexes[0].id, kind: 'index' })
      setPickerIndexId(indexes[0].id)
    }
  }, [indexes.length])

  const sectionTitle = (sectionId: string) => sections.find((section) => section.id === sectionId)?.title ?? ''
  const groupTitle = (groupId: string | null) => (groupId ? groups.find((group) => group.id === groupId)?.title ?? null : null)

  const moveTargets: MoveTarget[] = customCollections.flatMap((collection) => [
    { groupId: null, key: `${collection.id}:root`, label: `${collection.title} · ${t.fileLibrary.ungrouped}`, sectionId: collection.id },
    ...groups
      .filter((group) => group.sectionId === collection.id)
      .map((group) => ({ groupId: group.id, key: `${collection.id}:${group.id}`, label: `${collection.title} · ${group.title}`, sectionId: collection.id })),
  ])

  const sortAssets = (list: FileAssetRecord[]): FileAssetRecord[] => {
    if (sort === 'recent') return list
    const sorted = [...list]
    if (sort === 'name') sorted.sort((a, b) => a.label.localeCompare(b.label, locale))
    else if (sort === 'size') sorted.sort((a, b) => b.sizeBytes - a.sizeBytes)
    else if (sort === 'pages') sorted.sort((a, b) => (b.pageCount ?? 0) - (a.pageCount ?? 0))
    return sorted
  }

  const q = query.trim().toLowerCase()
  const matchesQuery = (asset: FileAssetRecord) =>
    !q || `${asset.label} ${asset.fileName} ${sectionTitle(asset.sectionId)} ${groupTitle(asset.groupId) ?? ''}`.toLowerCase().includes(q)
  const breadcrumbFor = (asset: FileAssetRecord) => [sectionTitle(asset.sectionId), groupTitle(asset.groupId)].filter(Boolean).join(' / ')

  const blocks: Block[] = useMemo(() => {
    if (active.kind === 'index') return []
    const pool = assets.filter(matchesQuery)
    if (q) return [{ band: null, breadcrumb: true, items: sortAssets(pool), key: 'search' }]
    if (active.kind === 'all') {
      return railCollections
        .map((collection) => {
          const items = sortAssets(pool.filter((asset) => asset.sectionId === collection.id))
          return {
            band: { count: items.length, groupId: null, isGroup: false, sectionId: collection.id, title: collection.title },
            breadcrumb: false,
            items,
            key: collection.id,
          }
        })
        .filter((block) => block.items.length > 0)
    }
    const sectionId = active.sectionId
    const ungrouped = sortAssets(pool.filter((asset) => asset.sectionId === sectionId && asset.groupId === null))
    const out: Block[] = ungrouped.length > 0
      ? [{
          band: { count: ungrouped.length, groupId: null, isGroup: false, sectionId, title: t.fileLibrary.ungrouped },
          breadcrumb: false,
          items: ungrouped,
          key: `${sectionId}:ungrouped`,
        }]
      : []
    groups
      .filter((group) => group.sectionId === sectionId)
      .forEach((group) => {
        const items = sortAssets(pool.filter((asset) => asset.groupId === group.id))
        out.push({ band: { count: items.length, groupId: group.id, isGroup: true, sectionId, title: group.title }, breadcrumb: false, items, key: group.id })
      })
    return out
  }, [active, assets, groups, railCollections, q, sort, locale, t.fileLibrary.ungrouped])

  const activeIndex = active.kind === 'index' ? vectorIndexById(state, active.indexId) : null
  const allIndexMembers = activeIndex ? vectorIndexMembersResolved(state, activeIndex.id) : []
  const indexMembers = sortMembersForSort(
    q ? allIndexMembers.filter((entry) => matchesQuery(entry.asset)) : allIndexMembers,
  )
  const activeIndexJob = activeIndex ? state.indexingJobs[activeIndex.id] : null
  // The per-file "Index this file" action only makes sense when the click can
  // actually run the INCREMENTAL path (existing collection, same model). On a
  // first build or a model change a single file forces a full rebuild — so the
  // per-row action is hidden there and the top button (which honestly says it
  // touches the whole index) takes over.
  const activeIndexCanIncrementalIndex =
    !!activeIndex
    && Boolean(activeIndex.serverCollectionId)
    && activeIndex.serverCollectionModel === activeIndex.model
  // A member's server-confirmed live outcome during an active client-build run
  // (the live job carries the per-file sets; absent → not yet processed).
  const memberLiveProgress = (fileId: string): 'embedded' | 'skipped' | undefined =>
    activeIndexJob?.embeddedFileIds?.includes(fileId)
      ? 'embedded'
      : activeIndexJob?.skippedFileIds?.includes(fileId)
        ? 'skipped'
        : undefined
  // Whether THIS file is part of the *actively running* job's working set —
  // only then does its row read "läuft" (see isMemberInRun: a still-queued job
  // pulses nothing, a file outside the run keeps its real state).
  const memberInRun = (fileId: string): boolean => isMemberInRun(activeIndexJob, fileId)
  function sortMembersForSort(members: ReturnType<typeof vectorIndexMembersResolved>) {
    if (sort === 'recent') return members
    const ordered = [...members]
    if (sort === 'name') ordered.sort((a, b) => a.asset.label.localeCompare(b.asset.label, locale))
    else if (sort === 'size') ordered.sort((a, b) => b.asset.sizeBytes - a.asset.sizeBytes)
    else if (sort === 'pages') ordered.sort((a, b) => (b.asset.pageCount ?? 0) - (a.asset.pageCount ?? 0))
    return ordered
  }
  const memberIds = useMemo(() => new Set(activeIndex ? activeIndex.members.map((member) => member.fileId) : []), [activeIndex])
  const isLibraryEmpty = active.kind !== 'index' && blocks.every((block) => block.items.length === 0)

  // ---- mutations ----
  async function ingestInto(files: File[], target: UploadTarget) {
    if (files.length === 0) return
    const existingLabels = assets.map((asset) => asset.label)
    const created = await ingestFiles(
      files,
      { groupId: target.groupId, kind: 'library', sectionId: target.sectionId },
      parser,
      existingLabels,
      serverFileUpload,
    )
    if (created.length === 0) return
    dispatch({ assets: created, type: 'ingestFileAssets' })
    onAssetsIngested?.(created)
    if (target.indexId) dispatch({ fileIds: created.map((asset) => asset.id), indexId: target.indexId, type: 'addDocsToVectorIndex' })
  }
  const openUpload = (target: UploadTarget) => {
    targetRef.current = target
    fileInputRef.current?.click()
  }

  const moveFile = (fileId: string, sectionId: string, groupId: string | null) => dispatch({ fileId, groupId, sectionId, type: 'moveFileAsset' })
  const renameFile = (fileId: string, label: string) => dispatch({ fileId, label, type: 'renameFileAsset' })
  const deleteFile = (fileId: string) => dispatch({ fileId, type: 'deleteFileAsset' })

  const { cancelReindex, startReindex } = useIndexingJobApi({
    apiKey: knowledgeSync?.apiKey,
    enabled: knowledgeSync !== null,
    onCancelled: (indexId) => dispatch({ indexId, type: 'markVectorIndexCancelled' }),
    onComplete: (indexId) => dispatch({ indexId, type: 'completeVectorIndexReindex' }),
    onDocumentCompleted: (indexId, documentId) =>
      dispatch({ indexId, serverDocumentId: documentId, type: 'markVectorIndexDocumentEmbedded' }),
    onError: (indexId, message) => dispatch({ indexId, message, type: 'markVectorIndexError' }),
    onProgress: (indexId, completedDocuments, totalDocuments, currentDocumentTitle) =>
      dispatch({ completedDocuments, currentDocumentTitle, indexId, totalDocuments, type: 'markVectorIndexProgress' }),
    onQueued: (indexId, queuePosition) =>
      dispatch({ indexId, queuePosition, type: 'markVectorIndexQueued' }),
    onStart: (indexId, jobId, totalDocuments) =>
      dispatch({ indexId, jobId, source: 'server', totalDocuments, type: 'startVectorIndexReindex' }),
    workspaceId: knowledgeSync?.workspaceId ?? '',
  })

  // `onlyFileId` scopes the run to a SINGLE pending member (the per-row "Index
  // this file" action) — it filters the pending set to that one document so the
  // incremental path ingests just it. Omitted = the whole index (top button).
  // `forceRebuild` (the "Neu aufbauen" action) bypasses the cheap incremental /
  // re-embed paths and re-ingests every member from its original file, so an OLD
  // collection picks up ingest-time provenance (page numbers, file_id, parser
  // upgrades) it predates — at the cost of re-embedding all + a brief churn.
  const triggerReindex = (indexId: string, onlyFileId?: string, forceRebuild = false) => {
    const index = vectorIndexById(state, indexId)
    // One job per index at a time (the per-row action must not race the top run).
    if (!index || index.status === 'indexing') return
    const memberEntries = vectorIndexMembersResolved(state, indexId)
    const pendingEntries = memberEntries.filter(
      (entry) =>
        entry.member.state === 'pending' && (!onlyFileId || entry.asset.id === onlyFileId),
    )
    // Single-file action on a non-pending member: nothing to do (never fall
    // through to a full re-embed of the whole collection).
    if (onlyFileId && pendingEntries.length === 0) return
    if (!knowledgeSync) {
      // Demo: the local simulator (effect below) drives progress to done. Scope
      // mirrors the real paths — index only the new (pending) members for an
      // incremental add / per-row action, the whole index for a refresh/rebuild,
      // so the demo never makes already-embedded rows read "läuft".
      const demoWorkingSet =
        !forceRebuild && pendingEntries.length > 0
          ? pendingEntries.map((entry) => entry.asset.id)
          : memberEntries.map((entry) => entry.asset.id)
      dispatch({
        indexId,
        jobId: `demo-${indexId}-${Date.now()}`,
        runningFileIds: demoWorkingSet,
        source: 'demo',
        totalDocuments: demoWorkingSet.length,
        type: 'startVectorIndexReindex',
      })
      return
    }
    const sync = knowledgeSync
    // serverCollectionModel is absent on indexes built before the field existed
    // -> reads as a mismatch, so the next reindex heals via a full rebuild.
    const sameModel = index.serverCollectionModel === index.model

    // Resolve member bodies (a server-synced project hydrates them empty, M6c
    // load-on-use), run a client-driven ingest, back-fill re-parsed text, and
    // report the embedded set so the reducer marks ONLY what actually landed
    // (skipped/text-less members stay pending -> the index reads honestly
    // stale). Shared by the incremental-add and full-rebuild branches.
    const runClientIngest = (
      assets: FileAssetRecord[],
      ingest: (
        resolved: FileAssetRecord[],
        onMemberDone: MemberProgress,
      ) => Promise<KnowledgeReindexResult>,
    ) => {
      const titleByFileId = new Map(
        assets.map((asset) => [asset.id, asset.title || asset.label]),
      )
      // Server-confirmed per-member progress → advance the bar + flip each
      // file row live (no cosmetic guessing; fires only after the await).
      const onMemberDone: MemberProgress = ({ fileId, done, total, embedded }) => {
        dispatch({
          completedDocuments: done,
          currentDocumentTitle: titleByFileId.get(fileId),
          embedded,
          fileId,
          indexId,
          totalDocuments: total,
          type: 'markVectorIndexProgress',
        })
      }
      void (async () => {
        try {
          const bodies = ensureAssetBodiesLoaded
            ? await ensureAssetBodiesLoaded(assets.map((asset) => asset.id))
            : null
          const resolved = bodies
            ? assets.map((asset) => ({
                ...asset,
                extractedText: bodies.get(asset.id) ?? asset.extractedText,
              }))
            : assets
          const result = await ingest(resolved, onMemberDone)
          // Upgrade each member the server re-parsed (MarkItDown) from the fast
          // client parse to the higher-fidelity text + 'markitdown' provenance.
          for (const { assetId, text } of result.reparsed) {
            dispatch({ assetId, extractedText: text, type: 'upgradeFileAssetParse' })
          }
          dispatch({
            embeddedFileIds: result.embeddedFileIds,
            skippedFileIds: result.skippedFileIds,
            indexId,
            serverCollectionId: result.collectionId,
            serverCollectionModel: result.serverCollectionModel,
            serverDocumentIds: result.serverDocumentIds,
            type: 'completeVectorIndexReindex',
          })
        } catch (error: unknown) {
          dispatch({
            indexId,
            message: error instanceof Error ? error.message : String(error),
            type: 'markVectorIndexError',
          })
        }
      })()
    }

    // Incremental add: a built collection, SAME embedding model, and only new
    // (pending) members — ingest just those into the existing collection. No
    // full rebuild, no re-embedding of documents already present. (This closes
    // the bug where docs added after the first build were never ingested.)
    if (!forceRebuild && index.serverCollectionId && sameModel && pendingEntries.length > 0) {
      const pendingAssets = pendingEntries.map((entry) => entry.asset)
      // The complete embedded set after this run = members already embedded
      // plus whatever of the pending set actually ingests.
      const alreadyEmbedded = memberEntries
        .filter((entry) => entry.member.state === 'embedded')
        .map((entry) => entry.asset.id)
      dispatch({
        indexId,
        jobId: `ingest-${indexId}-${Date.now()}`,
        // Only the new (pending) members run — already-embedded members stay out
        // of the working set so their rows keep reading "Indexiert".
        runningFileIds: pendingAssets.map((asset) => asset.id),
        source: 'build',
        totalDocuments: pendingAssets.length,
        type: 'startVectorIndexReindex',
      })
      runClientIngest(pendingAssets, async (resolved, onMemberDone) => {
        const result = await ingestNewVectorIndexMembers(index, resolved, sync, onMemberDone)
        return {
          ...result,
          embeddedFileIds: [...alreadyEmbedded, ...result.embeddedFileIds],
        }
      })
      return
    }

    // Re-embed in place: a built collection, SAME model, nothing new pending —
    // the durable server job re-vectorizes the stored text in place (real
    // per-document progress, cancellable, the collection stays online and is
    // never deleted/recreated). It preserves per-chunk page numbers via the
    // document's stored `_chunk_pages` (reembed_document re-aligns by index), so
    // a refresh keeps the PDF page-jump. NEW documents get their pages captured
    // at ingest (the incremental-add + file paths), not here — so this stays the
    // cheap, churn-free refresh rather than a delete+recreate that risks
    // orphaning the prior collection.
    if (!forceRebuild && index.serverCollectionId && sameModel) {
      void startReindex(index).catch((error: unknown) => {
        dispatch({
          indexId,
          message: error instanceof Error ? error.message : String(error),
          type: 'markVectorIndexError',
        })
      })
      return
    }

    // Full rebuild: first build (no collection yet), the embedding model changed
    // (a new vector dimension needs a fresh collection), OR an explicit "Neu
    // aufbauen" (forceRebuild) to re-read the original files. Client-driven
    // re-ingest of all current members (captures page numbers + file_id via the
    // ingest paths); deletes any stale prior collection.
    const memberAssets = memberEntries.map((entry) => entry.asset)
    dispatch({
      indexId,
      jobId: `build-${indexId}-${Date.now()}`,
      // A rebuild re-ingests every member, so the whole set is in the run.
      runningFileIds: memberAssets.map((asset) => asset.id),
      source: 'build',
      totalDocuments: memberAssets.length,
      type: 'startVectorIndexReindex',
    })
    runClientIngest(memberAssets, (resolved, onMemberDone) =>
      reindexVectorIndexOnServer(index, resolved, sync, onMemberDone),
    )
  }

  // "X" on a member: delete the exact document from the searchable collection
  // first (so removal is immediately effective, no full rebuild), THEN drop it
  // locally. Requires a tracked serverDocumentId — every member gets one after
  // a (re)index with this build. A member from an OLDER index (no tracked id)
  // or an offline session falls back to local-only removal: its server document
  // stays searchable until the index is rebuilt (or an admin runs the manual
  // reconcile sweep). Re-indexing such an index gives every member an id and
  // makes "X" exact. The remove control is disabled while a run is in flight.
  const removeMember = (fileId: string) => {
    if (!activeIndex) return
    const indexId = activeIndex.id
    const serverDocumentId = activeIndex.members.find(
      (member) => member.fileId === fileId,
    )?.serverDocumentId
    if (knowledgeSync && serverDocumentId) {
      void deleteKnowledgeDocument(serverDocumentId, knowledgeSync)
        .then(() => dispatch({ fileId, indexId, type: 'removeDocFromVectorIndex' }))
        .catch((error: unknown) => {
          // On failure DON'T dispatch the local removal: the member stays in the
          // list, which is the visible, honest signal that it did not leave the
          // searchable index (the user can re-click to retry). The index-level
          // error channel (markVectorIndexError) is gated on a run in flight, so
          // it cannot carry a removal failure; the member-stays signal is the
          // right-sized one. Console for diagnostics.
          console.error('Knowledge-Dokument konnte nicht geloescht werden', serverDocumentId, error)
        })
      return
    }
    dispatch({ fileId, indexId, type: 'removeDocFromVectorIndex' })
  }

  const handleCancelReindex = (indexId: string) => {
    const job = state.indexingJobs[indexId]
    if (!job) return
    // Only a durable server job can be cancelled server-side; demo and
    // first-build runs cancel locally. The decision reads the authoritative
    // `source` fact, never the job-id format (No-Silent-Fallbacks).
    if (knowledgeSync && job.source === 'server') {
      void cancelReindex(job.jobId).catch((error: unknown) => {
        // Server cancel failed (network/5xx): fall back to a local cancel
        // but surface why, so a still-running server job is not silent.
        console.warn('Inqtrix reindex cancel failed; cancelling locally.', error)
        dispatch({ indexId, type: 'markVectorIndexCancelled' })
      })
      return
    }
    dispatch({ indexId, type: 'markVectorIndexCancelled' })
  }

  useEffect(() => {
    // Demo-only reindex simulator: step each demo job's progress to done
    // (mount picks up a seeded mid-reindex too — demo "resume"). Server
    // jobs are driven by the hook's SSE stream instead.
    if (knowledgeSync) return undefined
    const timers = reindexTimers.current
    for (const [indexId, job] of Object.entries(state.indexingJobs)) {
      if (job.source !== 'demo' || timers.has(indexId)) continue
      const total = Math.max(1, job.totalDocuments)
      // The working set in processing order — confirm one file per tick so the
      // demo shows the same per-file flip as the real server path.
      const running = job.runningFileIds ?? []
      let done = job.completedDocuments
      const interval = setInterval(() => {
        const fileId = running[done]
        done += 1
        dispatch({
          completedDocuments: done,
          embedded: true,
          fileId,
          indexId,
          totalDocuments: total,
          type: 'markVectorIndexProgress',
        })
        if (done >= total) {
          dispatch({ indexId, type: 'completeVectorIndexReindex' })
          const handle = timers.get(indexId)
          if (handle) clearInterval(handle)
          timers.delete(indexId)
        }
      }, 850)
      timers.set(indexId, interval)
    }
    for (const indexId of Array.from(timers.keys())) {
      if (!state.indexingJobs[indexId]) {
        const handle = timers.get(indexId)
        if (handle) clearInterval(handle)
        timers.delete(indexId)
      }
    }
    return undefined
  }, [dispatch, knowledgeSync, state.indexingJobs])

  const handleNewIndex = () => {
    selectNewestIndex.current = true
    const defaultModel = embedModels[0]
    dispatch({
      dims: defaultModel?.dims,
      fileIds: [],
      model: defaultModel?.id,
      title: t.vectorIndex.newIndexTitle,
      type: 'createVectorIndex',
    })
  }
  const handleNewCollection = () => dispatch({ sectionId: '', title: t.fileLibrary.newCollectionTitle, type: 'createFileLibrarySection' })

  // ---- drop helpers ----
  const dropProps = (key: string, sectionId: string, groupId: string | null) => ({
    dropOver: dropKey === key,
    onDragLeave: () => setDropKey((current) => (current === key ? null : current)),
    onDragOver: (event: DragEvent) => {
      if (!isInternalFileDrag(event)) return
      event.preventDefault()
      event.dataTransfer.dropEffect = 'move'
      setDropKey(key)
    },
    onDrop: (event: DragEvent) => {
      if (!isInternalFileDrag(event)) return
      event.preventDefault()
      event.stopPropagation()
      const fileId = event.dataTransfer.getData('application/x-inqtrix-file-id')
      if (fileId) moveFile(fileId, sectionId, groupId)
      setDropKey(null)
    },
  })

  const rowCallbacks = {
    moveTargets,
    onDelete: deleteFile,
    onMove: moveFile,
    onPreview: setPreviewAssetId,
    onRename: renameFile,
  }

  // File preview overlay: the local record always carries the markdown for a
  // local asset; the Original (PDF) tab needs a connected files server, so the
  // options are null in demo/offline and the tab disables itself. Gated on the
  // FILES/persistence tier (fileApiOptions) — NOT on knowledge, so an uploaded
  // original stays viewable even when vector/knowledge is off.
  const previewAsset = previewAssetId ? state.fileAssets[previewAssetId] ?? null : null
  const previewOptions = fileApiOptions

  // ---- header bits ----
  const activeCollection = active.kind === 'collection' ? sections.find((section) => section.id === active.sectionId) ?? null : null
  const headerTitle = q
    ? t.fileLibrary.searchPlaceholderDocs
    : active.kind === 'all'
      ? t.fileLibrary.allCollections
      : active.kind === 'collection'
        ? activeCollection?.title ?? ''
        : activeIndex?.title ?? ''
  const crumbRoot = active.kind === 'index' ? t.vectorIndex.title : t.fileLibrary.sectionCollections

  return (
    <div className="grid h-[calc(100svh-var(--header-h))] grid-cols-1 bg-background lg:grid-cols-[17rem_minmax(0,1fr)]">
      <input
        className="hidden"
        multiple
        onChange={(event) => {
          void ingestInto(Array.from(event.target.files ?? []), targetRef.current)
          event.target.value = ''
        }}
        ref={fileInputRef}
        type="file"
      />

      <Rail
        active={active}
        collections={railCollections.map((collection) => ({ count: assetsInSection(collection.id).length, id: collection.id, title: collection.title }))}
        embeddingQuota={embeddingQuota}
        indexes={indexes.map((index) => ({ count: index.members.length, id: index.id, status: index.status, title: index.title }))}
        onDropToCollection={(sectionId, fileId) => moveFile(fileId, sectionId, null)}
        onNewCollection={handleNewCollection}
        onNewIndex={handleNewIndex}
        onQueryChange={setQuery}
        onSelectAll={() => setActive({ kind: 'all' })}
        onSelectCollection={(sectionId) => setActive({ kind: 'collection', sectionId })}
        onSelectIndex={(indexId) => setActive({ indexId, kind: 'index' })}
        query={query}
        storage={{ collectionCount: railCollections.length, docCount: assets.length, indexCount: indexes.length, usedBytes: projectStorageTotalBytes(state) }}
        totalDocCount={assets.length}
      />

      <div className="flex min-h-0 min-w-0 flex-col">
        {/* No bottom divider on the workspace header (deliberate, departs from DESIGN
            §8): the only line per section is the hairline under its section title. */}
        <header className="flex shrink-0 flex-wrap items-center gap-3 px-4 py-3 md:px-6">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5 t-meta text-muted-foreground">
              <span>{crumbRoot}</span>
              <ChevronRight className="size-3" />
              <span className="truncate text-foreground">{headerTitle}</span>
            </div>
            {active.kind === 'collection' && activeCollection ? (
              <InlineText
                ariaLabel={t.fileLibrary.renameCollection}
                className="mt-0.5 t-section text-foreground"
                onCommit={(title) => dispatch({ sectionId: activeCollection.id, title, type: 'renameFileLibrarySection' })}
                value={activeCollection.title}
              />
            ) : active.kind === 'index' && activeIndex ? (
              <InlineText
                ariaLabel={t.vectorIndex.rename}
                className="mt-0.5 t-section text-foreground"
                onCommit={(title) => dispatch({ indexId: activeIndex.id, title, type: 'renameVectorIndex' })}
                value={activeIndex.title}
              />
            ) : (
              <h1 className="mt-0.5 truncate t-section text-foreground">{headerTitle}</h1>
            )}
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-2">
            <ViewToggle onChange={setView} value={view} />
            <SortSelect onChange={setSort} value={sort} />
            {active.kind === 'index' && activeIndex ? (
              <Button className="gap-1.5 bg-brand text-brand-foreground hover:bg-brand/90" onClick={() => setPickerIndexId(activeIndex.id)} size="sm" type="button">
                <Plus className="size-4" />
                {t.vectorIndex.addDocuments}
              </Button>
            ) : (
              <>
                {active.kind === 'collection' && activeCollection ? (
                  <>
                    <Button
                      className="gap-1.5"
                      onClick={() => dispatch({ sectionId: activeCollection.id, title: t.fileLibrary.newGroupTitle, type: 'createFileGroup' })}
                      size="sm"
                      type="button"
                      variant="outline"
                    >
                      <FolderOpen className="size-4" />
                      {t.fileLibrary.createGroup}
                    </Button>
                    {activeCollection.kind === 'custom' ? (
                      <ConfirmDelete
                        ariaLabel={t.fileLibrary.removeCollection}
                        hint={t.fileLibrary.removeCollectionHint}
                        label={t.fileLibrary.removeCollection}
                        onConfirm={() => dispatch({ sectionId: activeCollection.id, type: 'deleteFileLibrarySection' })}
                      />
                    ) : null}
                  </>
                ) : null}
                <Button
                  className="gap-1.5 bg-brand text-brand-foreground hover:bg-brand/90"
                  onClick={() =>
                    openUpload(
                      active.kind === 'collection'
                        ? { groupId: null, sectionId: active.sectionId }
                        : { groupId: null, sectionId: customCollections[0]?.id ?? FILE_SECTION_TEMP_ID },
                    )
                  }
                  size="sm"
                  type="button"
                >
                  <Upload className="size-4" />
                  {t.fileLibrary.upload}
                </Button>
              </>
            )}
          </div>
        </header>

        <ScrollArea className="min-h-0 flex-1">
          <div className="flex flex-col gap-4 p-4 md:p-6">
            {active.kind === 'index' && activeIndex ? (
              <>
                <IndexBar
                  embedModels={embedModels}
                  embeddingQuota={embeddingQuota}
                  index={activeIndex}
                  live={state.indexingJobs[activeIndex.id] ?? null}
                  members={allIndexMembers}
                  onCancel={handleCancelReindex}
                  onDelete={(indexId) => dispatch({ indexId, type: 'deleteVectorIndex' })}
                  onModel={(indexId, model: EmbedModelId) => {
                    const descriptor = embedModels.find((entry) => entry.id === model)
                    dispatch({ dims: descriptor?.dims, indexId, model, type: 'setVectorIndexModel' })
                  }}
                  onReindex={triggerReindex}
                  onRebuild={(indexId) => triggerReindex(indexId, undefined, true)}
                  serverBacked={knowledgeSync !== null}
                  serverFeatureLabels={serverFeatureLabels}
                />
                {pickerIndexId === activeIndex.id ? (
                  <AddDocsPanel
                    docs={assets}
                    groups={groups}
                    memberIds={memberIds}
                    onAdd={(fileIds) => {
                      dispatch({ fileIds, indexId: activeIndex.id, type: 'addDocsToVectorIndex' })
                      setPickerIndexId(null)
                    }}
                    onClose={() => setPickerIndexId(null)}
                    onUpload={() => openUpload({ groupId: null, indexId: activeIndex.id, sectionId: FILE_SECTION_TEMP_ID })}
                    sections={railCollections}
                  />
                ) : null}
                {allIndexMembers.length === 0 ? (
                  <IndexEmpty onAdd={() => setPickerIndexId(activeIndex.id)} />
                ) : indexMembers.length === 0 ? (
                  <EmptyState
                    onUpload={() => openUpload({ groupId: null, indexId: activeIndex.id, sectionId: FILE_SECTION_TEMP_ID })}
                    searching
                  />
                ) : view === 'list' ? (
                  <div className="min-w-0 overflow-x-auto">
                    <div className="min-w-[54rem]">
                      <div className="flex flex-col">
                        {indexMembers.map(({ asset, member }) => (
                          <FileRow
                            asset={asset}
                            inRun={memberInRun(asset.id)}
                            indexing={activeIndex.status === 'indexing'}
                            key={asset.id}
                            liveProgress={memberLiveProgress(asset.id)}
                            memberState={member.state}
                            mode="index"
                            onIndexMember={activeIndexCanIncrementalIndex ? (fileId) => triggerReindex(activeIndex.id, fileId) : undefined}
                            onRemoveFromIndex={removeMember}
                            source={sectionTitle(asset.sectionId)}
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-2.5">
                    {indexMembers.map(({ asset, member }) => (
                      <FileCard
                        asset={asset}
                        inRun={memberInRun(asset.id)}
                        indexing={activeIndex.status === 'indexing'}
                        key={asset.id}
                        liveProgress={memberLiveProgress(asset.id)}
                        memberState={member.state}
                        mode="index"
                        onIndexMember={activeIndexCanIncrementalIndex ? (fileId) => triggerReindex(activeIndex.id, fileId) : undefined}
                        onRemoveFromIndex={removeMember}
                        source={sectionTitle(asset.sectionId)}
                      />
                    ))}
                  </div>
                )}
              </>
            ) : (
              <Dropzone
                label={t.fileLibrary.dropFiles}
                onFiles={(files) =>
                  void ingestInto(files, active.kind === 'collection' ? { groupId: null, sectionId: active.sectionId } : { groupId: null, sectionId: customCollections[0]?.id ?? FILE_SECTION_TEMP_ID })
                }
              >
                {isLibraryEmpty ? (
                  <EmptyState
                    onUpload={() => openUpload({ groupId: null, sectionId: customCollections[0]?.id ?? FILE_SECTION_TEMP_ID })}
                    searching={Boolean(q)}
                  />
                ) : (
                  <div className="flex flex-col">
                    {view === 'list' ? (
                      <div className="min-w-0 overflow-x-auto">
                        <div className="min-w-[54rem]">
                          <div className="flex flex-col gap-1.5">
                            {blocks.map((block) => {
                              const key = block.band ? block.key : `${block.key}:list`
                              const drop = block.band
                                ? dropProps(block.key, block.band.sectionId, block.band.groupId)
                                : active.kind === 'collection' && !q
                                  ? dropProps(block.key, active.sectionId, null)
                                  : null
                              return (
                                <section key={key}>
                                  {block.band ? (
                                    <BandHeader
                                      band={block.band}
                                      dropOver={drop?.dropOver ?? false}
                                      onDeleteGroup={(groupId) => dispatch({ groupId, type: 'deleteFileGroup' })}
                                      onDragLeave={drop?.onDragLeave ?? (() => undefined)}
                                      onDragOver={drop?.onDragOver ?? (() => undefined)}
                                      onDrop={drop?.onDrop ?? (() => undefined)}
                                      onNavigate={active.kind === 'all' ? () => setActive({ kind: 'collection', sectionId: block.band!.sectionId }) : undefined}
                                      onRenameGroup={(groupId, title) => dispatch({ groupId, title, type: 'renameFileGroup' })}
                                      onUpload={() => openUpload({ groupId: block.band!.groupId, sectionId: block.band!.sectionId })}
                                    />
                                  ) : null}
                                  {block.items.length === 0 ? (
                                    <p className="px-2 py-3 t-meta-sm text-muted-foreground">{t.fileLibrary.emptyGroup}</p>
                                  ) : (
                                    <div
                                      className={cn('flex flex-col transition-colors', !block.band && drop?.dropOver && 'rounded-md bg-brand-subtle/40 ring-1 ring-brand/25')}
                                      onDragLeave={!block.band ? drop?.onDragLeave : undefined}
                                      onDragOver={!block.band ? drop?.onDragOver : undefined}
                                      onDrop={!block.band ? drop?.onDrop : undefined}
                                    >
                                      {block.items.map((asset) => (
                                        <FileRow
                                          asset={asset}
                                          breadcrumb={block.breadcrumb ? breadcrumbFor(asset) : null}
                                          key={asset.id}
                                          mode="library"
                                          referenceCount={fileAssetReferenceCount(state, asset.id)}
                                          source={breadcrumbFor(asset)}
                                          {...rowCallbacks}
                                        />
                                      ))}
                                    </div>
                                  )}
                                </section>
                              )
                            })}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col gap-5">
                        {blocks.map((block) => {
                          const key = block.band ? block.key : `${block.key}:list`
                          const drop = block.band
                            ? dropProps(block.key, block.band.sectionId, block.band.groupId)
                            : active.kind === 'collection' && !q
                              ? dropProps(block.key, active.sectionId, null)
                              : null
                          return (
                            <section key={key}>
                              {block.band ? (
                                <BandHeader
                                  band={block.band}
                                  dropOver={drop?.dropOver ?? false}
                                  onDeleteGroup={(groupId) => dispatch({ groupId, type: 'deleteFileGroup' })}
                                  onDragLeave={drop?.onDragLeave ?? (() => undefined)}
                                  onDragOver={drop?.onDragOver ?? (() => undefined)}
                                  onDrop={drop?.onDrop ?? (() => undefined)}
                                  onNavigate={active.kind === 'all' ? () => setActive({ kind: 'collection', sectionId: block.band!.sectionId }) : undefined}
                                  onRenameGroup={(groupId, title) => dispatch({ groupId, title, type: 'renameFileGroup' })}
                                  onUpload={() => openUpload({ groupId: block.band!.groupId, sectionId: block.band!.sectionId })}
                                />
                              ) : null}
                              {block.items.length === 0 ? (
                                <p className="px-2 py-3 t-meta-sm text-muted-foreground">{t.fileLibrary.emptyGroup}</p>
                              ) : (
                                <div
                                  className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-2.5"
                                  onDragLeave={!block.band ? drop?.onDragLeave : undefined}
                                  onDragOver={!block.band ? drop?.onDragOver : undefined}
                                  onDrop={!block.band ? drop?.onDrop : undefined}
                                >
                                  {block.items.map((asset) => (
                                    <FileCard
                                      asset={asset}
                                      breadcrumb={block.breadcrumb ? breadcrumbFor(asset) : null}
                                      key={asset.id}
                                      mode="library"
                                      referenceCount={fileAssetReferenceCount(state, asset.id)}
                                      {...rowCallbacks}
                                    />
                                  ))}
                                </div>
                              )}
                            </section>
                          )
                        })}
                      </div>
                    )}
                  </div>
                )}
              </Dropzone>
            )}
          </div>
        </ScrollArea>
      </div>
      {previewAsset ? (
        <FilePreviewPanel
          asset={previewAsset}
          onClose={() => setPreviewAssetId(null)}
          options={previewOptions}
        />
      ) : null}
    </div>
  )
}
