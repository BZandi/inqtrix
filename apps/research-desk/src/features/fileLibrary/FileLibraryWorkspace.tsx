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
import type { EmbedModelId, FileAssetRecord, ProjectState } from '@/features/project/types'
import { createDefaultFileParser } from '@/features/files/parsing'
import { ingestFiles } from '@/features/files/ingest'
import { Dropzone } from '@/features/files/Dropzone'
import { FILE_SECTION_TEMP_ID } from '@/features/files/sections'
import type { ResearchDeskAction } from '../researchDesk/state'
import { Rail } from './Rail'
import { IndexBar } from './IndexBar'
import { AddDocsPanel } from './AddDocsPanel'
import { FileCard, FileRow, ListHeader } from './FileItem'
import { ConfirmDelete, InlineText, SortSelect, ViewToggle, type MoveTarget } from './controls'
import { groupSlug } from './helpers'
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
        'group/band mb-1.5 flex items-center gap-2 rounded-md px-1.5 py-1 transition-colors',
        dropOver && 'bg-brand-subtle/60 ring-1 ring-brand/30',
      )}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      {band.isGroup ? <FolderOpen className="size-3.5 shrink-0 text-file" /> : <Folder className="size-3.5 shrink-0 text-muted-foreground" />}
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
      <span className="shrink-0 rounded-full border border-border px-1.5 t-hint font-semibold leading-4 tabular-nums text-muted-foreground">{band.count}</span>
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

export function FileLibraryWorkspace({ dispatch, state }: { dispatch: Dispatch<ResearchDeskAction>; state: ProjectState }) {
  const { locale, t } = useLocale()
  const [active, setActive] = useState<ActiveTarget>({ kind: 'all' })
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortMode>('recent')
  const [view, setView] = useState<ViewMode>('list')
  const [dropKey, setDropKey] = useState<string | null>(null)
  const [pickerIndexId, setPickerIndexId] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const targetRef = useRef<UploadTarget>({ groupId: null, sectionId: FILE_SECTION_TEMP_ID })
  const reindexTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  const selectNewestIndex = useRef(false)

  useEffect(() => {
    const timers = reindexTimers.current
    return () => {
      timers.forEach((timer) => clearTimeout(timer))
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
    const out: Block[] = ungrouped.length > 0 ? [{ band: null, breadcrumb: false, items: ungrouped, key: `${sectionId}:ungrouped` }] : []
    groups
      .filter((group) => group.sectionId === sectionId)
      .forEach((group) => {
        const items = sortAssets(pool.filter((asset) => asset.groupId === group.id))
        out.push({ band: { count: items.length, groupId: group.id, isGroup: true, sectionId, title: group.title }, breadcrumb: false, items, key: group.id })
      })
    return out
  }, [active, assets, groups, railCollections, q, sort, locale])

  const activeIndex = active.kind === 'index' ? vectorIndexById(state, active.indexId) : null
  const indexMembers = activeIndex ? sortMembersForSort(vectorIndexMembersResolved(state, activeIndex.id)) : []
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
    const created = await ingestFiles(files, { groupId: target.groupId, kind: 'library', sectionId: target.sectionId }, parser, existingLabels)
    if (created.length === 0) return
    dispatch({ assets: created, type: 'ingestFileAssets' })
    if (target.indexId) dispatch({ fileIds: created.map((asset) => asset.id), indexId: target.indexId, type: 'addDocsToVectorIndex' })
  }
  const openUpload = (target: UploadTarget) => {
    targetRef.current = target
    fileInputRef.current?.click()
  }

  const moveFile = (fileId: string, sectionId: string, groupId: string | null) => dispatch({ fileId, groupId, sectionId, type: 'moveFileAsset' })
  const renameFile = (fileId: string, label: string) => dispatch({ fileId, label, type: 'renameFileAsset' })
  const deleteFile = (fileId: string) => dispatch({ fileId, type: 'deleteFileAsset' })

  const triggerReindex = (indexId: string) => {
    dispatch({ indexId, type: 'markVectorIndexIndexing' })
    const previous = reindexTimers.current.get(indexId)
    if (previous) clearTimeout(previous)
    const memberCount = vectorIndexById(state, indexId)?.members.length ?? 0
    const timer = setTimeout(() => {
      dispatch({ indexId, type: 'completeVectorIndexReindex' })
      reindexTimers.current.delete(indexId)
    }, 1400 + memberCount * 120)
    reindexTimers.current.set(indexId, timer)
  }

  const handleNewIndex = () => {
    selectNewestIndex.current = true
    dispatch({ fileIds: [], title: t.vectorIndex.newIndexTitle, type: 'createVectorIndex' })
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
    onRename: renameFile,
  }

  // ---- header bits ----
  const activeCollection = active.kind === 'collection' ? sections.find((section) => section.id === active.sectionId) ?? null : null
  const headerTitle = q
    ? t.fileLibrary.searchPlaceholderDocs
    : active.kind === 'all'
      ? t.fileLibrary.allDocuments
      : active.kind === 'collection'
        ? activeCollection?.title ?? ''
        : activeIndex?.title ?? ''
  const crumbRoot = active.kind === 'index' ? t.vectorIndex.title : t.fileLibrary.sectionDocuments

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
        <header className="flex shrink-0 flex-wrap items-center gap-3 border-b border-border px-4 py-3 md:px-6">
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
          <div className="mx-auto flex max-w-[960px] flex-col gap-4 p-4 md:p-6">
            {active.kind === 'index' && activeIndex ? (
              <>
                <IndexBar
                  index={activeIndex}
                  members={indexMembers}
                  onDelete={(indexId) => dispatch({ indexId, type: 'deleteVectorIndex' })}
                  onModel={(indexId, model: EmbedModelId) => dispatch({ indexId, model, type: 'setVectorIndexModel' })}
                  onReindex={triggerReindex}
                  onRename={(indexId, title) => dispatch({ indexId, title, type: 'renameVectorIndex' })}
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
                {indexMembers.length === 0 ? (
                  <IndexEmpty onAdd={() => setPickerIndexId(activeIndex.id)} />
                ) : view === 'list' ? (
                  <div className="overflow-hidden rounded-lg border border-border bg-card">
                    <ListHeader mode="index" />
                    {indexMembers.map(({ asset, member }) => (
                      <FileRow
                        asset={asset}
                        key={asset.id}
                        memberState={member.state}
                        mode="index"
                        onRemoveFromIndex={(fileId) => dispatch({ fileId, indexId: activeIndex.id, type: 'removeDocFromVectorIndex' })}
                        source={sectionTitle(asset.sectionId)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-2.5">
                    {indexMembers.map(({ asset, member }) => (
                      <FileCard
                        asset={asset}
                        key={asset.id}
                        memberState={member.state}
                        mode="index"
                        onRemoveFromIndex={(fileId) => dispatch({ fileId, indexId: activeIndex.id, type: 'removeDocFromVectorIndex' })}
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
                            <p className="px-3 py-3 text-xs text-muted-foreground">{t.fileLibrary.emptyGroup}</p>
                          ) : view === 'list' ? (
                            <div
                              className={cn('overflow-hidden rounded-lg border border-border bg-card transition-colors', !block.band && drop?.dropOver && 'ring-1 ring-brand/30')}
                              onDragLeave={!block.band ? drop?.onDragLeave : undefined}
                              onDragOver={!block.band ? drop?.onDragOver : undefined}
                              onDrop={!block.band ? drop?.onDrop : undefined}
                            >
                              <ListHeader mode="library" />
                              {block.items.map((asset) => (
                                <FileRow
                                  asset={asset}
                                  breadcrumb={block.breadcrumb ? breadcrumbFor(asset) : null}
                                  key={asset.id}
                                  mode="library"
                                  referenceCount={fileAssetReferenceCount(state, asset.id)}
                                  {...rowCallbacks}
                                />
                              ))}
                            </div>
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
              </Dropzone>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}
