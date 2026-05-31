import { useRef, useState, type DragEvent, type Dispatch, type KeyboardEvent } from 'react'
import { AlertTriangle, Database, FolderOpen, Paperclip, Plus, Trash2, Upload } from '@/components/icons'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import {
  fileAssetsForGroup,
  fileAssetsForSection,
  fileGroupsForSection,
  projectFileAssets,
  projectFileLibrarySections,
} from '@/features/project/selectors'
import type { FileAssetRecord, FileLibrarySectionRecord, ProjectState } from '@/features/project/types'
import { createDefaultFileParser } from '@/features/files/parsing'
import { ingestFiles } from '@/features/files/ingest'
import { Dropzone } from '@/features/files/Dropzone'
import type { ResearchDeskAction } from '../researchDesk/state'

const parser = createDefaultFileParser()

/** Internal drag type for moving a file row into a group/section. Distinct from
 * external file drops (which carry the `Files` type and are handled by Dropzone). */
const FILE_DRAG_TYPE = 'application/x-inqtrix-file-id'

function isInternalFileDrag(event: DragEvent): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes(FILE_DRAG_TYPE)
}

type UploadTarget = {
  groupId: string | null
  sectionId: string
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unitIndex = 0
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  return `${value.toFixed(unitIndex === 0 || value >= 10 ? 0 : 1)} ${units[unitIndex]}`
}

function InlineText({
  ariaLabel,
  className,
  onCommit,
  value,
}: {
  ariaLabel: string
  className?: string
  onCommit: (next: string) => void
  value: string
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)

  if (!editing) {
    return (
      <button
        aria-label={ariaLabel}
        className={cn('truncate rounded-sm px-1 text-left hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring', className)}
        onClick={() => {
          setDraft(value)
          setEditing(true)
        }}
        type="button"
      >
        {value}
      </button>
    )
  }

  const commit = () => {
    setEditing(false)
    const next = draft.trim()
    if (next && next !== value) onCommit(next)
  }

  return (
    <input
      aria-label={ariaLabel}
      autoFocus
      className={cn('min-w-0 rounded-sm border border-input bg-background px-1 text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring', className)}
      onBlur={commit}
      onChange={(event) => setDraft(event.target.value)}
      onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') {
          event.preventDefault()
          commit()
        }
        if (event.key === 'Escape') {
          event.preventDefault()
          setEditing(false)
        }
      }}
      value={draft}
    />
  )
}

export function FileLibraryWorkspace({
  dispatch,
  state,
}: {
  dispatch: Dispatch<ResearchDeskAction>
  state: ProjectState
}) {
  const { t } = useLocale()
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const targetRef = useRef<UploadTarget>({ groupId: null, sectionId: '' })
  const sections = projectFileLibrarySections(state)

  async function ingestInto(files: File[], target: UploadTarget) {
    if (files.length === 0) return
    const existingLabels = projectFileAssets(state).map((asset) => asset.label)
    const assets = await ingestFiles(
      files,
      { groupId: target.groupId, kind: 'library', sectionId: target.sectionId },
      parser,
      existingLabels,
    )
    if (assets.length === 0) return
    dispatch({ assets, type: 'ingestFileAssets' })
  }

  function openUpload(target: UploadTarget) {
    targetRef.current = target
    fileInputRef.current?.click()
  }

  const moveTargets = sections.flatMap((section) => [
    { groupId: null, label: `${section.title} · ${t.fileLibrary.ungrouped}`, sectionId: section.id },
    ...fileGroupsForSection(state, section.id).map((group) => ({
      groupId: group.id,
      label: `${section.title} · ${group.title}`,
      sectionId: section.id,
    })),
  ])

  function FileRow({ asset }: { asset: FileAssetRecord }) {
    const hasWarning = asset.parseStatus !== 'parsed' || asset.textTruncated
    return (
      <div
        className="group flex min-w-0 cursor-grab items-center gap-2 rounded-md border border-border/70 bg-card/70 px-2 py-1.5 active:cursor-grabbing"
        draggable
        onDragStart={(event) => {
          event.dataTransfer.setData(FILE_DRAG_TYPE, asset.id)
          event.dataTransfer.effectAllowed = 'move'
        }}
      >
        <span className="grid size-7 shrink-0 place-items-center rounded-md border border-file/25 bg-file-subtle text-file">
          <Paperclip className="size-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <InlineText
              ariaLabel={t.fileLibrary.rename}
              className="max-w-full text-xs font-semibold text-foreground"
              onCommit={(label) => dispatch({ fileId: asset.id, label, type: 'renameFileAsset' })}
              value={asset.label}
            />
            {hasWarning && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="inline-flex shrink-0 items-center text-warning">
                    <AlertTriangle className="size-3.5" />
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top">{asset.parseWarning ?? t.fileLibrary.parseWarning}</TooltipContent>
              </Tooltip>
            )}
          </div>
          <div className="truncate text-[11px] text-muted-foreground" title={asset.fileName}>
            {asset.fileName}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5 text-[10px] text-muted-foreground">
          <Badge className="font-normal" variant="outline">{formatBytes(asset.sizeBytes)}</Badge>
          <Badge className="font-normal" variant="outline">
            {asset.pageCount != null ? `${asset.pageCount} ${t.fileLibrary.pagesUnit}` : t.fileLibrary.noPages}
          </Badge>
          <code className="rounded-sm bg-muted px-1 py-0.5 text-[10px] text-muted-foreground">@files:{asset.label}</code>
        </div>
        <div className="flex shrink-0 items-center opacity-60 transition group-hover:opacity-100">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button aria-label={t.fileLibrary.move} className="size-7" size="icon" type="button" variant="ghost">
                <FolderOpen className="size-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="max-h-72 overflow-y-auto">
              <DropdownMenuLabel>{t.fileLibrary.move}</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {moveTargets.map((target) => {
                const isCurrent = target.sectionId === asset.sectionId && target.groupId === asset.groupId
                return (
                  <DropdownMenuItem
                    disabled={isCurrent}
                    key={`${target.sectionId}:${target.groupId ?? 'root'}`}
                    onClick={() => dispatch({ fileId: asset.id, groupId: target.groupId, sectionId: target.sectionId, type: 'moveFileAsset' })}
                  >
                    {target.label}
                  </DropdownMenuItem>
                )
              })}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button
            aria-label={t.fileLibrary.remove}
            className="size-7 text-muted-foreground hover:text-destructive"
            onClick={() => dispatch({ fileId: asset.id, type: 'deleteFileAsset' })}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Trash2 className="size-3.5" />
          </Button>
        </div>
      </div>
    )
  }

  function SectionView({ section }: { section: FileLibrarySectionRecord }) {
    const ungrouped = fileAssetsForSection(state, section.id).filter((asset) => asset.groupId === null)
    const groups = fileGroupsForSection(state, section.id)
    const isEmpty = ungrouped.length === 0 && groups.length === 0

    return (
      <section
        className="rounded-lg border border-border bg-background/60 p-3"
        onDragOver={(event) => {
          if (!isInternalFileDrag(event)) return
          event.preventDefault()
          event.dataTransfer.dropEffect = 'move'
        }}
        onDrop={(event) => {
          if (!isInternalFileDrag(event)) return
          event.preventDefault()
          const fileId = event.dataTransfer.getData(FILE_DRAG_TYPE)
          if (fileId) dispatch({ fileId, groupId: null, sectionId: section.id, type: 'moveFileAsset' })
        }}
      >
        <div className="mb-2 flex items-center justify-between gap-2">
          <InlineText
            ariaLabel={t.fileLibrary.renameSection}
            className="text-sm font-semibold text-foreground"
            onCommit={(title) => dispatch({ sectionId: section.id, title, type: 'renameFileLibrarySection' })}
            value={section.title}
          />
          <div className="flex shrink-0 items-center gap-1">
            <Button
              className="h-7 gap-1.5 px-2 text-xs"
              onClick={() => dispatch({ sectionId: section.id, title: t.fileLibrary.newGroupTitle, type: 'createFileGroup' })}
              size="sm"
              type="button"
              variant="ghost"
            >
              <Plus className="size-3.5" />
              {t.fileLibrary.createGroup}
            </Button>
            <Button
              className="h-7 gap-1.5 px-2 text-xs"
              onClick={() => openUpload({ groupId: null, sectionId: section.id })}
              size="sm"
              type="button"
              variant="outline"
            >
              <Upload className="size-3.5" />
              {t.fileLibrary.upload}
            </Button>
          </div>
        </div>

        <Dropzone label={t.fileLibrary.dropFiles} onFiles={(files) => void ingestInto(files, { groupId: null, sectionId: section.id })}>
        <div className="flex flex-col gap-1.5">
          {ungrouped.map((asset) => <FileRow asset={asset} key={asset.id} />)}
          {groups.map((group) => {
            const groupAssets = fileAssetsForGroup(state, group.id)
            return (
              <div
                className="rounded-md border border-border/60 bg-muted/30 p-2"
                key={group.id}
                onDragOver={(event) => {
                  if (!isInternalFileDrag(event)) return
                  event.preventDefault()
                  event.dataTransfer.dropEffect = 'move'
                }}
                onDrop={(event) => {
                  if (!isInternalFileDrag(event)) return
                  event.preventDefault()
                  event.stopPropagation()
                  const fileId = event.dataTransfer.getData(FILE_DRAG_TYPE)
                  if (fileId) dispatch({ fileId, groupId: group.id, sectionId: section.id, type: 'moveFileAsset' })
                }}
              >
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <FolderOpen className="size-3.5 shrink-0 text-file" />
                    <InlineText
                      ariaLabel={t.fileLibrary.renameGroup}
                      className="text-xs font-semibold text-foreground"
                      onCommit={(title) => dispatch({ groupId: group.id, title, type: 'renameFileGroup' })}
                      value={group.title}
                    />
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Button
                      aria-label={t.fileLibrary.upload}
                      className="size-7"
                      onClick={() => openUpload({ groupId: group.id, sectionId: section.id })}
                      size="icon"
                      type="button"
                      variant="ghost"
                    >
                      <Upload className="size-3.5" />
                    </Button>
                    <Button
                      aria-label={t.fileLibrary.removeGroup}
                      className="size-7 text-muted-foreground hover:text-destructive"
                      onClick={() => dispatch({ groupId: group.id, type: 'deleteFileGroup' })}
                      size="icon"
                      type="button"
                      variant="ghost"
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </div>
                </div>
                <div className="flex flex-col gap-1.5">
                  {groupAssets.length === 0
                    ? <p className="px-1 py-2 text-[11px] text-muted-foreground">{t.fileLibrary.emptyGroup}</p>
                    : groupAssets.map((asset) => <FileRow asset={asset} key={asset.id} />)}
                </div>
              </div>
            )
          })}
          {isEmpty && <p className="px-1 py-3 text-xs text-muted-foreground">{t.fileLibrary.emptySection}</p>}
        </div>
        </Dropzone>
      </section>
    )
  }

  return (
    <ScrollArea className="h-[calc(100svh-var(--header-h))] w-full">
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
      <div className="mx-auto flex max-w-3xl flex-col gap-3 p-4 md:p-6">
        <header className="flex items-center gap-2.5">
          <span className="grid size-9 place-items-center rounded-lg border border-file/25 bg-file-subtle text-file">
            <Database className="size-4" />
          </span>
          <div>
            <h1 className="text-base font-semibold text-foreground">{t.fileLibrary.title}</h1>
            <p className="text-xs text-muted-foreground">{t.fileLibrary.subtitle}</p>
          </div>
        </header>
        {sections.map((section) => <SectionView key={section.id} section={section} />)}
      </div>
    </ScrollArea>
  )
}
