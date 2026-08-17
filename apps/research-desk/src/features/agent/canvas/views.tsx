import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type MutableRefObject,
  type MouseEvent,
} from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'

import { appMotion } from '@/motion/transitions'

import {
  getKnowledgeChunk,
  type KnowledgeChunkDetail,
} from '@/api/inqtrixClient'
import { canvasSurfaceClass } from '@/features/canvas/CanvasSurface'
import { subscribeRunEvents } from '@/features/researchRuns/runEventChannel'
import {
  AlertTriangle,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Copy,
  Download,
  ExternalLink,
  FileText,
  Globe2,
  PenLine,
  X,
} from '@/components/icons'
import { MarkdownRenderer } from '@/components/markdown/MarkdownRenderer'
import { MarkdownSelectionCopyMenu } from '@/components/markdown/MarkdownSelectionCopyMenu'
import { Button } from '@/components/ui/button'
import { PhaseSegments } from '@/components/ui/phase-segments'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  StatusBadge,
  type StatusTone,
} from '@/features/settings/parts'
import type { CanvasViewDescriptor } from '@/features/canvas/types'
import { DocumentDiffView } from '@/features/editor/DocumentDiffView'
import { editorCopy } from '@/features/editor/editorCopy'
import { MarkdownEditorSurface } from '@/features/editor/core/MarkdownEditorSurface'
import {
  FilePreviewBody,
  FilePreviewTabSwitch,
  useFilePreviewTabs,
} from '@/features/files/FilePreviewBody'
import { phaseLabel } from '@/features/researchDesk/components/runDisplay'
import { phaseOrder } from '@/features/researchDesk/types'
import type {
  ResearchRunEvent,
  ResearchRunSnapshot,
} from '@/features/researchRuns/types'
import type { FileAssetRecord } from '@/features/project/types'
import { useRunningDuration } from '@/features/researchRuns/useRunningDuration'
import {
  citationViews,
  firstOpenableCitation,
  groupCitationsByDocument,
} from '@/features/knowledge/citations'
import {
  CitationGroupList,
  CitationRow,
} from '@/features/knowledge/CitationRow'
import { useLocale } from '@/i18n/LocaleProvider'
import { scheduleIdle } from '@/lib/idle'
import { formatDuration } from '@/lib/time'
import { withAiDisclosure } from '@/lib/aiDisclosure'
import { cn } from '@/lib/utils'
import { suggestionDiffSegments } from '@/features/editor/suggestionDiff'
import { AgentActivityLine, AgentPulseTrack } from '../AgentPulseTrack'
import {
  canEditAgentRun,
  type AgentChildProgressRecord,
  type AgentPlanTaskRecord,
  isActiveAgentRun,
  isGateAgentRun,
  TERMINAL_AGENT_TASK_STATUSES,
  type AgentRunRecord,
} from '../model'
import { PlanReviewBody } from '../plan/PlanReviewBody'
import { usePlanApproval } from '../plan/usePlanApproval'
import { usePatchReview } from '../patch/usePatchReview'
import {
  activityDisplayText,
  terminalActivityErrorIndex,
} from '../activityPresentation'
import {
  agentArtifactReferences,
  agentCitationLabelFromHref,
  agentReferenceAsKnowledge,
  linkifyAgentArtifactCitations,
  type AgentArtifactReference,
} from '../artifactCitations'
import {
  EvidenceProvenancePanel,
} from '../EvidenceProvenancePanel'
import {
  evidenceLineageFromArtifactPayload,
  safeEvidenceHttpUrl,
} from '../evidenceProvenance'
import { WebEvidenceSourceRow } from '../WebEvidenceSourceRow'
import { activityText } from '../timeline/AgentTimeline'
import {
  agentTaskExecutionLabel,
  agentTaskElapsedSeconds,
  agentTaskExecutionSemantics,
  agentTaskGroup,
  agentTaskMetrics,
  agentTaskResultContent,
  agentTaskResultPreview,
  agentTaskStatusLabel,
  effectiveAgentTaskStatus,
  type AgentTaskEffectiveStatus,
  type AgentTaskGroup,
} from '../plan/taskPresentation'
import {
  childProgressMessage,
  completedResearchPhases,
  mergeResearchSnapshot,
  researchNodePhase,
  snapshotWithResearchMetrics,
  type ChildProgressMessage,
} from '../taskProgress'
import { useAgentCanvas } from './context'
import { prefetchableTaskResultIds } from '../taskResultPrefetch'
import type { AgentTaskResultWire } from '../types'
import {
  taskResultReferenceGroups,
} from '../taskResultReferences'
import { copyTextToClipboard } from '@/lib/clipboard'

const SAVE_DEBOUNCE_MS = 900

// --- document ----------------------------------------------------------------

/**
 * The memo canvas (plan §5.2 `document`): while the agent writes, a
 * streaming-friendly report renderer with the "agent writing" chip; once
 * ready, the ONE Tiptap surface with debounced optimistic-concurrency saves
 * (E13 — 409s reload instead of overwriting).
 */
export function DocumentCanvasView({
  descriptor,
}: {
  descriptor: Extract<CanvasViewDescriptor, { view: 'document' }>
}) {
  const context = useAgentCanvas()
  const { locale, t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  const run = context.runs[descriptor.runId]
  const artifact = run?.artifacts[descriptor.artifactId]
  const [notice, setNotice] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const saveTimerRef = useRef<number | null>(null)
  const pendingRef = useRef<string | null>(null)
  const baseRevisionRef = useRef(0)

  // The body loads on demand (the list rows never carry it).
  useEffect(() => {
    if (!artifact || artifact.contentMarkdown !== undefined) return
    void context.loadArtifact(descriptor.runId, descriptor.artifactId)
  }, [artifact, context, descriptor.artifactId, descriptor.runId])

  const writing = artifact?.status === 'writing'
  const canEdit = Boolean(
    artifact
    && !writing
    && artifact.contentMarkdown !== undefined
    && canEditAgentRun(run),
  )
  const references = useMemo(
    () => agentArtifactReferences(artifact?.refs),
    [artifact?.refs],
  )

  useEffect(() => setEditing(false), [descriptor.artifactId])

  useEffect(() => {
    if (canEdit) return
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
    pendingRef.current = null
    setEditing(false)
  }, [canEdit])

  useEffect(() => {
    if (artifact && artifact.contentMarkdown !== undefined && pendingRef.current === null) {
      baseRevisionRef.current = artifact.revision
    }
  }, [artifact])

  const flushSave = useCallback(async () => {
    const markdown = pendingRef.current
    if (markdown === null || !artifact) return
    pendingRef.current = null
    const result = await context.saveArtifact(
      descriptor.runId,
      descriptor.artifactId,
      markdown,
      baseRevisionRef.current,
    )
    if (result.kind === 'saved') {
      baseRevisionRef.current = result.revision
      setNotice(null)
    } else if (result.kind === 'locked') {
      setNotice(t.agent.canvas.lockedByAgent)
    } else {
      // Stale edit: the fresh body was reloaded by saveArtifact; say so.
      if (result.currentRevision !== null) {
        baseRevisionRef.current = result.currentRevision
      }
      setNotice(t.agent.canvas.editConflict)
    }
  }, [artifact, context, descriptor.artifactId, descriptor.runId, t])

  const handleChange = useCallback(
    (markdown: string) => {
      pendingRef.current = markdown
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current)
      }
      saveTimerRef.current = window.setTimeout(() => {
        saveTimerRef.current = null
        void flushSave()
      }, SAVE_DEBOUNCE_MS)
    },
    [flushSave],
  )

  // Follow-up submits await the pending edit (plan §5.4 flush rule).
  useEffect(() => {
    context.pendingSaveRef.current = flushSave
    return () => {
      context.pendingSaveRef.current = null
    }
  }, [context.pendingSaveRef, flushSave])

  useEffect(
    () => () => {
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current)
        void flushSave()
      }
    },
    [flushSave],
  )

  const editorDocument = useMemo(() => {
    if (!artifact || artifact.contentMarkdown === undefined) return null
    return {
      contentMarkdown: artifact.contentMarkdown,
      createdAt: new Date(artifact.createdAt * 1000).toISOString(),
      folderId: null,
      // Revision is deliberately NOT part of the identity: the surface
      // resets content on id change only, preserving the cursor.
      id: `agent-artifact-${artifact.artifactId}`,
      revision: artifact.revision,
      source: 'agent-artifact' as const,
      sourceRunId: descriptor.runId,
      title: artifact.title,
      updatedAt: new Date(artifact.updatedAt * 1000).toISOString(),
    }
  }, [artifact, descriptor.runId])

  if (!run || !artifact) {
    return <CanvasMissing label={t.agent.canvas.empty} />
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-1.5">
        {writing && (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-brand/20 bg-brand-subtle px-2 py-0.5 t-hint font-semibold text-brand">
            <span
              aria-hidden="true"
              className={cn(
                'size-1.5 rounded-full bg-brand',
                !reduceMotion && 'inqtrix-running-dot',
              )}
            />
            {t.agent.canvas.agentWriting}
          </span>
        )}
        {artifact.revisions && artifact.revisions.length > 1 ? (
          <select
            aria-label={t.agent.plan.versionHistory}
            className="h-5 shrink-0 rounded border border-border bg-card px-1 t-hint tabular-nums text-muted-foreground"
            onChange={(event) => {
              const from = Number(event.target.value)
              if (Number.isFinite(from) && from < artifact.revision) {
                context.openCanvasView({
                  view: 'diff',
                  runId: descriptor.runId,
                  artifactId: descriptor.artifactId,
                  fromRevision: from,
                  toRevision: artifact.revision,
                })
              }
              event.target.value = String(artifact.revision)
            }}
            value={artifact.revision}
          >
            {artifact.revisions.map((entry) => (
              <option key={entry.revision} value={entry.revision}>
                {t.agent.canvas.revision.replace(
                  '{revision}',
                  String(entry.revision),
                )}
              </option>
            ))}
          </select>
        ) : (
          <span className="t-hint tabular-nums text-muted-foreground">
            {t.agent.canvas.revision.replace('{revision}', String(artifact.revision))}
          </span>
        )}
        {notice && (
          <span className="truncate t-hint font-semibold text-warning">{notice}</span>
        )}
        <span className="min-w-0 flex-1" />
        {canEdit && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                aria-label={editing ? t.agent.canvas.readMemo : t.agent.canvas.editMemo}
                className="size-7 text-muted-foreground hover:text-foreground"
                onClick={() => {
                  if (!editing) {
                    setEditing(true)
                    return
                  }
                  void flushSave().finally(() => setEditing(false))
                }}
                size="icon"
                type="button"
                variant="ghost"
              >
                {editing ? (
                  <BookOpen className="icon-md" />
                ) : (
                  <PenLine className="icon-md" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              {editing ? t.agent.canvas.readMemo : t.agent.canvas.editMemo}
            </TooltipContent>
          </Tooltip>
        )}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              aria-label={t.agent.canvas.copyMarkdown}
              className="size-7 text-muted-foreground hover:text-foreground"
              onClick={() => {
                if (artifact.contentMarkdown !== undefined) {
                  void copyTextToClipboard(
                    withAiDisclosure(artifact.contentMarkdown, t.aiTransparency.exportNotice),
                  )
                }
              }}
              size="icon"
              type="button"
              variant="ghost"
            >
              <Copy className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">{t.agent.canvas.copyMarkdown}</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              aria-label={t.agent.canvas.exportToEditor}
              className="size-7 text-muted-foreground hover:text-foreground"
              onClick={() => {
                void context
                  .exportArtifact(
                    descriptor.runId,
                    descriptor.artifactId,
                    artifact.title,
                  )
                  .catch((error: unknown) => {
                    setNotice(
                      error instanceof Error ? error.message : String(error),
                    )
                  })
              }}
              size="icon"
              type="button"
              variant="ghost"
            >
              <Download className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">{t.agent.canvas.exportToEditor}</TooltipContent>
        </Tooltip>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className={canvasSurfaceClass}>
          {editing && canEdit && editorDocument ? (
            <MarkdownEditorSurface
              comments={[]}
              copy={editorCopy[locale]}
              diffAnchorMarkdown={null}
              document={editorDocument}
              embedded
              isDiffVisible={false}
              mode="live"
              onAcceptSuggestion={noop}
              onChange={handleChange}
              onCreateComment={noop}
              onEditSuggestion={noop}
              onEditorReady={noop}
              onMarkSuggestionStale={noop}
              onRefineSuggestion={asyncNoop}
              onRejectSuggestion={noop}
              onSelectComment={noop}
              onStopSuggestion={noop}
              runningSuggestionIds={[]}
              selectedCommentId={null}
              suggestionErrors={{}}
              suggestions={[]}
              textImprovement={{
                enabled: false,
                workspaceId: context.workspaceId,
              }}
            />
          ) : artifact.contentMarkdown !== undefined ? (
            <MarkdownSelectionCopyMenu
              aiGenerated
              className="report-markdown w-full min-w-0 max-w-full [overflow-wrap:anywhere]"
              markdown={artifact.contentMarkdown}
              onClickCapture={(event: MouseEvent<HTMLDivElement>) => {
                const anchor = (event.target as HTMLElement | null)?.closest('a')
                const label = agentCitationLabelFromHref(
                  anchor?.getAttribute('href'),
                )
                if (!label || !references.some((item) => item.label === label)) {
                  return
                }
                event.preventDefault()
                event.stopPropagation()
                context.openCanvasView({
                  artifactId: descriptor.artifactId,
                  label,
                  runId: descriptor.runId,
                  view: 'evidence',
                })
              }}
            >
              <MarkdownRenderer
                markdown={linkifyAgentArtifactCitations(
                  artifact.contentMarkdown,
                  references,
                )}
                variant="report"
              />
            </MarkdownSelectionCopyMenu>
          ) : (
            <p className="t-hint text-muted-foreground">…</p>
          )}
          {references.length > 0 && (
            <AgentArtifactSources
              onOpen={(label) => context.openCanvasView({
                artifactId: descriptor.artifactId,
                label,
                runId: descriptor.runId,
                view: 'evidence',
              })}
              references={references}
            />
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

/** Artifact evidence rendered with the established Knowledge source-row
 * hierarchy. Inline chips are intentionally absent; sources live where a
 * reader expects them: after the report body. */
function AgentArtifactSources({
  onOpen,
  references,
}: {
  onOpen: (label: string) => void
  references: AgentArtifactReference[]
}) {
  const { t } = useLocale()
  const views = citationViews(
    references.map(agentReferenceAsKnowledge),
    [],
    t.knowledge.viewerSection,
  ).map((view) => ({ ...view, canOpen: true }))
  const internalGroups = new Map(
    groupCitationsByDocument(
      views.filter((view) => view.documentId !== null),
    ).map((group) => [group.documentId, group]),
  )
  const groups = taskResultReferenceGroups(
    references.map((reference) => ({
      chunk_index: reference.chunkIndex,
      document_id: reference.documentId,
      excerpt: reference.excerpt,
      grounded_support: reference.groundedSupport,
      label: reference.label,
      page_number: reference.pageNumber,
      citation_id: reference.citationId,
      provider_snippet: reference.providerSnippet,
      query_id: reference.queryId,
      source_id: reference.sourceId,
      title: reference.title,
      url: reference.url,
    })),
  )
  return (
    <section className="mt-8 border-t border-border/70 pt-4">
      <h2 className="t-section text-foreground">{t.knowledge.sources}</h2>
      <ul className="mt-2 space-y-2">
        {groups.flatMap((sourceGroup) => {
          if (sourceGroup.kind === 'web') {
            return [
              <li key={sourceGroup.reference.key}>
                <WebEvidenceSourceRow
                  onInspect={() => {
                    if (sourceGroup.reference.label) {
                      onOpen(sourceGroup.reference.label)
                    }
                  }}
                  reference={sourceGroup.reference}
                />
              </li>,
            ]
          }
          const documentId = sourceGroup.references[0]?.documentId
          const group = documentId ? internalGroups.get(documentId) : undefined
          if (!group) return []
          if (group.citations.length === 1) {
            const view = group.citations[0]
            return view ? [
              <li key={`${documentId}:${view.label}`}>
                <CitationRow
                  onOpen={(selected) => onOpen(selected.label)}
                  view={view}
                />
              </li>,
            ] : []
          }
          return [
            <li key={documentId}>
              <CitationGroupList
                activeKey={null}
                groups={[group]}
                onOpen={(selected) => onOpen(selected.label)}
                onOpenDocument={(selectedGroup) => {
                  const selected = firstOpenableCitation(selectedGroup)
                  if (selected) onOpen(selected.label)
                }}
              />
            </li>,
          ]
        })}
      </ul>
    </section>
  )
}

// --- plan ---------------------------------------------------------------------

export function PlanCanvasView({
  descriptor,
}: {
  descriptor: Extract<CanvasViewDescriptor, { view: 'plan' }>
}) {
  const context = useAgentCanvas()
  const { t } = useLocale()
  const run = context.runs[descriptor.runId]
  const planApproval = usePlanApproval({
    decideApproval: context.decideApproval,
    draft: context.planDrafts[descriptor.runId] ?? null,
    onDraftChange: (draft) => context.setPlanDraft(descriptor.runId, draft),
    run: run ?? emptyRunPlaceholder,
  })
  // On-open refresh: the background loop only fetches on a stale flag, and
  // a pre-plan 404 clears that flag with no content — opening the tab in
  // either window would render empty forever. One re-flag per open (the
  // ref guard keeps the 404 -> clear -> re-flag cycle from looping); a
  // later `plan.proposed` event re-flags through the normal path.
  const planRequestedForRef = useRef<string | null>(null)
  const hasRun = Boolean(run)
  const hasPlan = Boolean(run?.plan)
  const planStale = Boolean(run?.planStale)
  useEffect(() => {
    if (!hasRun || hasPlan || planStale) return
    if (planRequestedForRef.current === descriptor.runId) return
    planRequestedForRef.current = descriptor.runId
    context.requestPlanRefresh(descriptor.runId)
  }, [context, descriptor.runId, hasPlan, hasRun, planStale])
  if (!run) return <CanvasMissing label={t.agent.plan.noPlan} />
  const pending = planApproval.pendingApproval
  const canEdit = canEditAgentRun(run)
  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className={cn(canvasSurfaceClass, 'space-y-3')}>
        <PlanReviewBody
          density="full"
          draft={planApproval.draft}
          editable={Boolean(pending) && canEdit}
          planSource={context.planSource}
          run={run}
          updateDraft={planApproval.updateDraft}
        />
        {run.plan && run.plan.versions.length > 1 && (
          <div>
            <p className="t-caption text-muted-foreground">
              {t.agent.plan.versionHistory}
            </p>
            <ul className="mt-1 space-y-0.5">
              {run.plan.versions.map((version) => (
                <li
                  className="t-meta tabular-nums text-muted-foreground"
                  key={version.planId}
                >
                  v{version.version} · {version.status} · {version.createdBy}
                  {version.reason ? ` · ${version.reason}` : ''}
                </li>
              ))}
            </ul>
          </div>
        )}
        {planApproval.error && (
          <p className="t-meta-sm text-destructive">{planApproval.error}</p>
        )}
        {pending && canEdit && (
          <div className="flex items-center justify-end gap-1.5">
            <Button
              className="h-7 px-2.5 text-xs"
              disabled={planApproval.submitting}
              onClick={() => void planApproval.decide('reject')}
              size="sm"
              type="button"
              variant="outline"
            >
              {t.agent.timeline.reject}
            </Button>
            <Button
              className="h-7 bg-brand px-2.5 text-xs text-brand-foreground hover:bg-brand/90"
              disabled={planApproval.submitting}
              onClick={() => void planApproval.decide('approve')}
              size="sm"
              type="button"
            >
              {t.agent.timeline.approve}
            </Button>
          </div>
        )}
      </div>
    </ScrollArea>
  )
}

// --- evidence -------------------------------------------------------------------

export function EvidenceCanvasView({
  descriptor,
}: {
  descriptor: Extract<CanvasViewDescriptor, { view: 'evidence' }>
}) {
  const context = useAgentCanvas()
  const { t } = useLocale()
  const run = context.runs[descriptor.runId]
  const artifact = run?.artifacts[descriptor.artifactId]
  const references = agentArtifactReferences(artifact?.refs)
  const reference = references.find((item) => item.label === descriptor.label)
  const evidenceArtifact = Object.values(run?.artifacts ?? {}).find(
    (item) => item.kind === 'evidence_bundle',
  )
  const lineage = reference
    ? evidenceLineageFromArtifactPayload(evidenceArtifact?.payload, reference)
    : null
  const safeReferenceUrl = safeEvidenceHttpUrl(reference?.url ?? null)
  const [chunk, setChunk] = useState<KnowledgeChunkDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lineageLoadFailed, setLineageLoadFailed] = useState(false)

  useEffect(() => {
    if (!artifact || artifact.contentMarkdown !== undefined) return
    void context.loadArtifact(descriptor.runId, descriptor.artifactId)
  }, [artifact, context, descriptor.artifactId, descriptor.runId])

  useEffect(() => {
    setLineageLoadFailed(false)
    if (
      reference?.documentId
      || (!reference?.queryId && !reference?.sourceId)
      || !evidenceArtifact
      || evidenceArtifact.payload !== undefined
    ) return
    void context
      .loadArtifact(descriptor.runId, evidenceArtifact.artifactId)
      .catch(() => setLineageLoadFailed(true))
  }, [
    context,
    descriptor.runId,
    evidenceArtifact,
    reference?.documentId,
    reference?.queryId,
    reference?.sourceId,
  ])

  useEffect(() => {
    setChunk(null)
    setError(null)
    if (!reference?.documentId || !context.clientOptions) return
    let cancelled = false
    getKnowledgeChunk(
      reference.documentId,
      reference.chunkIndex ?? 0,
      1,
      context.clientOptions,
    )
      .then((detail) => {
        if (!cancelled) setChunk(detail)
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught))
        }
      })
    return () => {
      cancelled = true
    }
  }, [context.clientOptions, reference?.chunkIndex, reference?.documentId])

  if (!artifact) {
    return <CanvasMissing label={t.agent.canvas.evidenceUnavailable} />
  }
  if (artifact.contentMarkdown === undefined) {
    return <CanvasMissing label="…" />
  }
  if (!reference) {
    return <CanvasMissing label={t.agent.canvas.evidenceUnavailable} />
  }
  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className={cn(canvasSurfaceClass, 'space-y-4')}>
        <div className="flex min-w-0 items-start gap-2">
          {reference.documentId ? (
            <BookOpen className="mt-0.5 icon-md shrink-0 text-brand" />
          ) : (
            <Globe2 className="mt-0.5 icon-md shrink-0 text-brand" />
          )}
          <div className="min-w-0 flex-1">
            <p className="break-words t-section text-foreground">
              {reference.title}
            </p>
            <p className="mt-0.5 t-meta-sm text-muted-foreground">
              {reference.label} · {reference.documentId
                ? t.agent.canvas.knowledgeEvidence
                : t.agent.canvas.webEvidence}
            </p>
          </div>
        </div>
        {safeReferenceUrl && (
          <a
            className="inline-flex max-w-full items-center gap-1.5 t-meta text-brand hover:underline"
            href={safeReferenceUrl}
            rel="noreferrer noopener"
            target="_blank"
          >
            <ExternalLink className="icon-sm shrink-0" />
            <span className="truncate">{reference.url}</span>
          </a>
        )}
        {reference.groundedSupport && !lineage && (
          <section className="rounded-lg border border-brand/20 bg-brand-subtle/25 px-3 py-3">
            <p className="t-caption uppercase tracking-wide text-muted-foreground">
              {t.agent.canvas.providerGroundedSupport}
            </p>
            <p className="mt-1 whitespace-pre-wrap break-words t-body text-foreground/90">
              {reference.groundedSupport}
            </p>
          </section>
        )}
        {!reference.documentId
          && !reference.excerpt
          && !reference.groundedSupport
          && !lineage && (
          <p className="t-meta text-muted-foreground">
            {t.agent.canvas.legacyEvidenceUnavailable}
          </p>
        )}
        {reference.documentId && (
          <p className="t-mono text-muted-foreground">
            doc:{reference.documentId}#{reference.chunkIndex ?? 0}
          </p>
        )}
        {error && <p className="t-meta text-destructive">{error}</p>}
        {reference.documentId && chunk && (
          <>
            <p className="t-caption uppercase tracking-wide text-muted-foreground">
              {t.agent.canvas.exactKnowledgeChunk}
            </p>
            <blockquote className="rounded-md border-l-2 border-brand bg-brand-subtle/30 px-3 py-2 t-body text-foreground">
              {chunk.excerpt}
            </blockquote>
            {chunk.neighbors?.map((neighbor) => (
              <p
                className="t-meta text-muted-foreground"
                key={neighbor.chunk_index}
              >
                {neighbor.excerpt}
              </p>
            ))}
            {chunk.page_number !== null && (
              <p className="t-hint text-muted-foreground">
                S. {chunk.page_number}
              </p>
            )}
          </>
        )}
        {reference.documentId && !chunk && !error && (
          <p className="t-hint text-muted-foreground">…</p>
        )}
        <EvidenceProvenancePanel
          lineage={lineage}
          lineageLoadFailed={lineageLoadFailed}
          reference={reference}
        />
      </div>
    </ScrollArea>
  )
}

// --- file -----------------------------------------------------------------------

export function FileCanvasView({
  descriptor,
}: {
  descriptor: Extract<CanvasViewDescriptor, { view: 'file' }>
}) {
  const context = useAgentCanvas()
  const { t } = useLocale()
  const { originalOpened, switchTab, tab } = useFilePreviewTabs()
  const asset: FileAssetRecord | undefined = descriptor.assetId
    ? context.fileAssets[descriptor.assetId]
    : undefined
  if (!asset) {
    return <CanvasMissing label={t.agent.canvas.views.file} />
  }
  const canShowOriginal = Boolean(asset.serverFileId && context.clientOptions)
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-1.5">
        <FileText className="icon-sm shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate t-meta text-muted-foreground">
          {asset.fileName}
        </span>
        <FilePreviewTabSwitch
          canShowOriginal={canShowOriginal}
          onSwitch={switchTab}
          tab={tab}
        />
      </div>
      <FilePreviewBody
        asset={asset}
        options={context.clientOptions}
        originalOpened={originalOpened}
        tab={tab}
      />
    </div>
  )
}

// --- diff -----------------------------------------------------------------------

export function DiffCanvasView({
  descriptor,
}: {
  descriptor: Extract<CanvasViewDescriptor, { view: 'diff' }>
}) {
  const context = useAgentCanvas()
  const { locale, t } = useLocale()
  const [fromMarkdown, setFromMarkdown] = useState<string | null>(null)
  const [toMarkdown, setToMarkdown] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setFromMarkdown(null)
    setToMarkdown(null)
    setError(null)
    void Promise.all([
      context.loadArtifact(
        descriptor.runId,
        descriptor.artifactId,
        descriptor.fromRevision,
      ),
      context.loadArtifact(
        descriptor.runId,
        descriptor.artifactId,
        descriptor.toRevision,
      ),
    ])
      .then(([from, to]) => {
        if (cancelled) return
        setFromMarkdown(from.content_markdown)
        setToMarkdown(to.content_markdown)
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught))
        }
      })
    return () => {
      cancelled = true
    }
  }, [context, descriptor])

  if (error) {
    return <CanvasMissing label={error} />
  }
  if (fromMarkdown === null || toMarkdown === null) {
    return <CanvasMissing label="…" />
  }
  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className={canvasSurfaceClass}>
        <p className="mb-3 t-hint tabular-nums text-muted-foreground">
          {t.agent.canvas.revision.replace('{revision}', String(descriptor.fromRevision))}
          {' → '}
          {t.agent.canvas.revision.replace('{revision}', String(descriptor.toRevision))}
        </p>
        <DocumentDiffView
          anchorMarkdown={fromMarkdown}
          copy={editorCopy[locale]}
          currentMarkdown={toMarkdown}
        />
      </div>
    </ScrollArea>
  )
}

// --- patch ----------------------------------------------------------------------

/**
 * Patch review (M7): the agent's proposed edits against the target editor
 * document, one diff block per edit (the suggestion-diff vocabulary), with
 * the always-gated approve&apply / reject actions shared with the timeline
 * card via `usePatchReview`.
 */
export function PatchCanvasView({
  descriptor,
}: {
  descriptor: Extract<CanvasViewDescriptor, { view: 'patch' }>
}) {
  const context = useAgentCanvas()
  const { t } = useLocale()
  const run = context.runs[descriptor.runId]
  const review = usePatchReview({
    actions: {
      applyPatch: context.applyPatch,
      decideApproval: context.decideApproval,
      rejectPatch: context.rejectPatch,
    },
    run: run ?? emptyRunPlaceholder,
  })
  if (!run) return <CanvasMissing label={t.agent.canvas.views.patch} />
  const patch = run.patch
  if (!patch) return <CanvasMissing label="…" />
  const gatesActive = isGateAgentRun(run.status)
  const decided = patch.status !== 'pending'
  const canEdit = canEditAgentRun(run)

  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className={cn(canvasSurfaceClass, 'space-y-3')}>
        {patch.summary && (
          <p className="t-body text-foreground/90">{patch.summary}</p>
        )}
        <p className="t-hint tabular-nums text-muted-foreground">
          {t.agent.patch.editCount.replace(
            '{count}',
            String(patch.edits.length),
          )}
          {decided && (
            <span className="ml-2 font-semibold">
              {patch.status === 'accepted'
                ? t.agent.patch.applied
                : t.agent.patch.rejected}
            </span>
          )}
        </p>
        {patch.warnings.map((warning, index) => (
          <p className="t-meta-sm text-warning" key={index}>
            {warning}
          </p>
        ))}
        <ul className="space-y-2">
          {patch.edits.map((edit) => (
            <PatchEditBlock
              edit={edit}
              key={edit.id}
              skipped={
                patch.appliedEditIds !== null
                && !patch.appliedEditIds.includes(edit.id)
              }
              t={t}
            />
          ))}
        </ul>
        {review.notice && (
          <p className="t-meta text-warning">{review.notice}</p>
        )}
        {review.pendingApproval && gatesActive && !decided && canEdit && (
          <div className="flex items-center justify-end gap-1.5">
            <Button
              className="h-7 px-2.5 text-xs"
              disabled={review.submitting}
              onClick={() => void review.reject()}
              size="sm"
              type="button"
              variant="outline"
            >
              {t.agent.timeline.reject}
            </Button>
            <Button
              className="h-7 bg-brand px-2.5 text-xs text-brand-foreground hover:bg-brand/90"
              disabled={review.submitting}
              onClick={() => void review.approveAndApply()}
              size="sm"
              type="button"
            >
              {t.agent.patch.applyAndApprove}
            </Button>
          </div>
        )}
      </div>
    </ScrollArea>
  )
}

function PatchEditBlock({
  edit,
  skipped,
  t,
}: {
  edit: { id: string; find: string; position: string; text: string; note: string }
  skipped: boolean
  t: ReturnType<typeof useLocale>['t']
}) {
  const isReplace = edit.position === 'replace' && edit.find
  const isDelete = isReplace && !edit.text.trim()
  const segments = isReplace && !isDelete
    ? suggestionDiffSegments(edit.find, edit.text)
    : null
  return (
    <li
      className={cn(
        'rounded-md border border-border bg-surface/50 px-3 py-2',
        skipped && 'opacity-60',
      )}
    >
      {edit.note && (
        <p className="mb-1.5 t-meta text-muted-foreground">{edit.note}</p>
      )}
      {segments ? (
        <p className="whitespace-pre-wrap t-body">
          {segments.map((segment, index) =>
            segment.type === 'equal' ? (
              <span key={index}>{segment.text}</span>
            ) : segment.type === 'delete' ? (
              <del className="rounded-sm bg-destructive/10 text-destructive/90 line-through decoration-destructive/50" key={index}>
                {segment.text}
              </del>
            ) : (
              <ins className="rounded-sm bg-success-subtle text-success no-underline" key={index}>
                {segment.text}
              </ins>
            ),
          )}
        </p>
      ) : isDelete ? (
        <p className="whitespace-pre-wrap t-body">
          <span className="t-caption text-destructive">{t.agent.patch.deletion}</span>
          <del className="mt-1 block rounded-sm bg-destructive/10 text-destructive/90 line-through decoration-destructive/50">
            {edit.find}
          </del>
        </p>
      ) : (
        <div className="t-body">
          <span className="t-caption text-success">{t.agent.patch.insertion}</span>
          {edit.find && (
            <p className="mt-1 whitespace-pre-wrap t-meta text-muted-foreground">
              {edit.find}
            </p>
          )}
          <ins className="mt-1 block whitespace-pre-wrap rounded-sm bg-success-subtle text-success no-underline">
            {edit.text}
          </ins>
        </div>
      )}
    </li>
  )
}

// --- shared -------------------------------------------------------------------

// --- run (Verlauf) -------------------------------------------------------------

/** The live run view is the Agent Desk control room. A selected task remains
 * inside this run tab, so users drill down and Back without creating a second
 * canvas document or losing the run overview. */
export function RunCanvasView({
  descriptor,
}: {
  descriptor: Extract<CanvasViewDescriptor, { view: 'run' }>
}) {
  const context = useAgentCanvas()
  const { t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  const restoredFocusRef = useRef(false)
  const run = context.runs[descriptor.runId]
  if (!run) return <CanvasMissing label={t.agent.canvas.empty} />
  const selectedTaskId = descriptor.taskId
  const detailOpen = Boolean(selectedTaskId)
  // Page-in-page push (DESIGN.md motion table): the detail layer slides
  // in full width while the list parallaxes to -30% and stays MOUNTED
  // (scroll position and focus survive; `inert` blocks the covered
  // layer). Percent-based transforms keep the push resize-proof inside
  // the resizable panel; reduced motion collapses to a 120ms fade.
  return (
    <div className="relative flex min-h-0 flex-1 overflow-hidden">
      <motion.div
        animate={
          reduceMotion
            ? { opacity: detailOpen ? 0 : 1, x: 0 }
            : { x: detailOpen ? '-30%' : '0%' }
        }
        className="absolute inset-0 flex min-h-0 flex-col"
        inert={detailOpen || undefined}
        transition={
          reduceMotion
            ? { duration: 0.12 }
            : detailOpen
              ? appMotion.push
              : appMotion.pushExit
        }
      >
        <RunOverview
          descriptor={descriptor}
          restoredFocusRef={restoredFocusRef}
          run={run}
        />
      </motion.div>
      <AnimatePresence initial={false}>
        {selectedTaskId && (
          <motion.div
            animate={reduceMotion ? { opacity: 1, x: 0 } : { x: '0%' }}
            className="absolute inset-0 z-10 flex min-h-0 flex-col bg-background shadow-[-12px_0_24px_-12px_var(--shadow-soft)]"
            exit={
              reduceMotion
                ? { opacity: 0, x: 0, transition: { duration: 0.12 } }
                : { x: '100%', transition: appMotion.pushExit }
            }
            initial={reduceMotion ? { opacity: 0, x: 0 } : { x: '100%' }}
            key={selectedTaskId}
            transition={reduceMotion ? { duration: 0.12 } : appMotion.push}
          >
            <TaskDetailView
              onBack={() => {
                context.openCanvasView({
                  focusTaskId: selectedTaskId,
                  runId: descriptor.runId,
                  view: 'run',
                })
              }}
              run={run}
              taskId={selectedTaskId}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function RunOverview({
  descriptor,
  restoredFocusRef,
  run,
}: {
  descriptor: Extract<CanvasViewDescriptor, { view: 'run' }>
  restoredFocusRef: MutableRefObject<boolean>
  run: AgentRunRecord
}) {
  const context = useAgentCanvas()
  const { t } = useLocale()
  const isLive = isActiveAgentRun(run.status)
  const gate = isGateAgentRun(run.status)
  const tasks = (run.plan?.tasks ?? []).filter(
    (task) => task.toolKind !== 'synthesis',
  )
  const done = tasks.filter(
    (task) => effectiveAgentTaskStatus(task, run.taskStates[task.taskId]) === 'completed',
  ).length
  const transportDegraded = context.pollingRunIds.includes(run.runId)
  const groups = taskGroups(tasks, run)
  // Idle prefetch: settled task results of the OPEN run view warm before
  // the click (hover covers pointer users; this covers touch/keyboard
  // and the first click). The effect keys on the CANDIDATE SET, not on
  // the per-event taskStates identity — a live run streams events every
  // second and must not re-arm the idle callback for an unchanged set.
  const prefetchIdsKey = prefetchableTaskResultIds(
    run.plan?.tasks ?? [],
    run.taskStates,
    6,
  ).join(',')
  const contextRef = useRef(context)
  contextRef.current = context
  useEffect(() => {
    if (!prefetchIdsKey) return undefined
    return scheduleIdle(() => {
      for (const taskId of prefetchIdsKey.split(',')) {
        contextRef.current.prefetchTaskResult(run.runId, taskId)
      }
    }, { timeout: 800 })
  }, [prefetchIdsKey, run.runId])
  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className={cn(canvasSurfaceClass, 'space-y-4')}>
        <p className="break-words t-body text-foreground/90">{run.question}</p>
        <div className="flex items-center gap-3">
          <AgentPulseTrack className="min-w-0 flex-1" run={run} withLabels />
          <StatusBadge
            className="self-start"
            density="table"
            label={t.status[run.status]}
            tone={runStatusTone(run.status)}
          />
        </div>
        {isLive && (
          <AgentActivityLine gate={gate} text={activityText(run, t)} />
        )}
        {isLive && transportDegraded && (
          <p className="flex items-center gap-1.5 t-hint text-muted-foreground" role="status">
            <AlertTriangle className="icon-xs shrink-0 text-warning" />
            {t.agent.timeline.transportDegraded}
          </p>
        )}
        {run.error && (
          <p className="break-words t-meta-sm text-destructive">{run.error}</p>
        )}
        {tasks.length > 0 && (
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <span className="t-card text-foreground">
                {t.agent.timeline.tasks}
              </span>
              <span className="t-hint tabular-nums text-muted-foreground">
                {done}/{tasks.length}
              </span>
            </div>
            <div className="space-y-3">
              {groups.map((group) => (
                <section key={group.kind}>
                  <div className="mb-1 flex items-center justify-between">
                    <p className="t-caption uppercase tracking-wide text-muted-foreground">
                      {taskGroupLabel(group.kind, t)}
                    </p>
                    <span className="t-hint tabular-nums text-muted-foreground">
                      {group.tasks.length}
                    </span>
                  </div>
                  <ul className="space-y-2">
                    {group.tasks.map((task) => {
                      const live = run.taskStates[task.taskId]
                      return (
                        <li key={task.taskId}>
                          <TaskWorkUnit
                            child={childForTask(run, task, live?.childRunId)}
                            id={taskButtonId(task.taskId)}
                            live={live}
                            onClick={() => {
                              restoredFocusRef.current = false
                              context.openCanvasView({
                                runId: run.runId,
                                taskId: task.taskId,
                                view: 'run',
                              })
                            }}
                            onPrefetch={
                              TERMINAL_AGENT_TASK_STATUSES.has(
                                effectiveAgentTaskStatus(task, live),
                              )
                                ? () => context.prefetchTaskResult(
                                  run.runId,
                                  task.taskId,
                                )
                                : undefined
                            }
                            onFocusRestored={() => {
                              restoredFocusRef.current = true
                            }}
                            restoreFocus={
                              !restoredFocusRef.current
                              && descriptor.focusTaskId === task.taskId
                            }
                            task={task}
                          />
                        </li>
                      )
                    })}
                  </ul>
                </section>
              ))}
            </div>
          </div>
        )}
      </div>
    </ScrollArea>
  )
}

function TaskWorkUnit({
  child,
  id,
  live,
  onClick,
  onFocusRestored,
  onPrefetch,
  restoreFocus,
  task,
}: {
  child: AgentChildProgressRecord | undefined
  id: string
  live: AgentRunRecord['taskStates'][string] | undefined
  onClick: () => void
  onFocusRestored: () => void
  /** Intent signal (hover/focus): warm the task-result cache so the
   * detail view opens without its fetch delay. Terminal tasks only. */
  onPrefetch?: () => void
  restoreFocus: boolean
  task: AgentPlanTaskRecord
}) {
  const { t } = useLocale()
  const buttonRef = useRef<HTMLButtonElement>(null)
  const status = effectiveAgentTaskStatus(task, live)
  const activeOperation = status === 'running' || status === 'cancel_requested'
  const startedAtIso = live?.startedAt === undefined
    ? undefined
    : new Date(live.startedAt * 1000).toISOString()
  const runningDuration = useRunningDuration(
    activeOperation ? 'running' : 'settled',
    startedAtIso,
  )
  const elapsedSeconds = agentTaskElapsedSeconds(live, Date.now() / 1000)
  const elapsedLabel = live?.startedAt === undefined
    ? undefined
    : activeOperation
      ? runningDuration
      : elapsedSeconds === undefined
        ? undefined
        : formatDuration(elapsedSeconds)

  useLayoutEffect(() => {
    if (!restoreFocus || !buttonRef.current) return
    buttonRef.current.focus({ preventScroll: true })
    onFocusRestored()
  }, [onFocusRestored, restoreFocus])
  const snapshot = snapshotWithResearchMetrics(
    projectedChildSnapshot(child),
    live?.metrics,
    localRequestCount(task),
  )
  const phase = researchNodePhase(child?.currentNode ?? snapshot?.current_node)
  const activity = live?.activity
  const phaseMessage = child?.message
    ?? snapshot?.last_message
    ?? (activity ? activityDisplayText(activity, t) : undefined)
  const message = phaseMessage
    ?? (
      status === 'running'
      && task.toolKind === 'web_research'
      && !phase
        ? t.agent.canvas.child.preparing
        : undefined
    )
  const liveError = live?.error ?? child?.error
  const updatedAt = child?.updatedAt ?? activity?.at
  const resultSummary = task.resultSummary || live?.resultSummary
  const resultPreview = resultSummary
    ? agentTaskResultPreview(task.title, resultSummary)
    : ''
  return (
    <button
      className="group relative w-full rounded-lg border border-border bg-card py-2.5 pl-3 pr-9 text-left shadow-[0_1px_2px_var(--shadow-hairline)] transition-colors hover:bg-accent/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      id={id}
      onClick={onClick}
      onFocus={onPrefetch}
      onPointerEnter={onPrefetch}
      ref={buttonRef}
      type="button"
    >
      <div className="flex min-w-0 items-start gap-2">
        <TaskStatusGlyph status={status} />
        <div className="min-w-0 flex-1">
          <p className="break-words t-list font-semibold text-foreground">
            {task.title}
          </p>
          <p className="mt-0.5 break-words t-meta-sm text-muted-foreground">
            {agentTaskExecutionLabel(task, t)}
            {(live?.attempt ?? child?.attempt ?? 1) > 1
              && ` · ${t.agent.canvas.child.attempt.replace('{count}', String(live?.attempt ?? child?.attempt))}`}
          </p>
          {/* Multi-query tasks report per-query progress on the card —
              running AND settled — so no search stays invisible behind
              the aggregate badge (Verlauf = Protokoll). */}
          {task.queries.length > 1 && (
            <p className="mt-0.5 t-hint tabular-nums text-muted-foreground">
              {t.agent.task.queryProgress
                .replace(
                  '{done}',
                  String(
                    (live?.activityHistory ?? []).filter(
                      (item) =>
                        item.status === 'completed'
                        || item.status === 'failed',
                    ).length,
                  ),
                )
                .replace('{total}', String(task.queries.length))}
            </p>
          )}
        </div>
        <span className="mr-1 shrink-0">
          <StatusBadge
            density="table"
            label={agentTaskStatusLabel(status, t)}
            tone={taskStatusTone(status)}
          />
        </span>
      </div>
      <ChevronRight
        aria-hidden="true"
        className="absolute right-3 top-1/2 icon-sm -translate-y-1/2 text-muted-foreground/55 transition-[color,transform] group-hover:translate-x-0.5 group-hover:text-foreground"
      />
      {phase && (
        <div className="mt-2.5">
          <PhaseSegments
            activePhase={phase}
            completedPhases={completedResearchPhases(
              child?.currentNode ?? snapshot?.current_node,
            )}
            phases={phaseOrder}
            thin
          />
        </div>
      )}
      {task.toolKind === 'web_instant' && activeOperation && (
        <IndeterminateTaskRail />
      )}
      <TaskMetrics
        includeZeroClaims={Boolean(child) || task.toolKind === 'web_research'}
        snapshot={snapshot}
      />
      {message && (
        <p className="mt-1.5 break-words t-meta text-muted-foreground" role="status">
          {message}
        </p>
      )}
      {resultPreview && !activeOperation && (
        <p className="mt-1.5 line-clamp-3 break-words t-meta text-foreground/85">
          {resultPreview}
        </p>
      )}
      {live?.fallback && (
        <p className="mt-1.5 flex items-start gap-1.5 break-words t-meta-sm text-warning">
          <AlertTriangle className="mt-0.5 icon-xs shrink-0" />
          {t.agent.canvas.child.fallbackUsed}
        </p>
      )}
      {liveError && (
        <p className="mt-1.5 break-words t-meta-sm text-destructive">
          {liveError}
        </p>
      )}
      {(updatedAt !== undefined || elapsedLabel) && (
        <div className="mt-1 flex items-center justify-between gap-3 t-hint tabular-nums text-muted-foreground/70">
          <span>
            {updatedAt === undefined
              ? ''
              : t.agent.canvas.child.lastUpdate.replace(
                '{time}',
                new Date(updatedAt * 1000).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                }),
              )}
          </span>
          {elapsedLabel && (
            <span className="inline-flex shrink-0 items-center gap-1">
              <Clock3 aria-hidden="true" className="icon-xs" />
              <span className="sr-only">{t.agent.canvas.child.elapsed}</span>
              {elapsedLabel}
            </span>
          )}
        </div>
      )}
    </button>
  )
}

function TaskDetailView({
  onBack,
  run,
  taskId,
}: {
  onBack: () => void
  run: AgentRunRecord
  taskId: string
}) {
  const context = useAgentCanvas()
  const { t } = useLocale()
  const task = run.plan?.tasks.find((candidate) => candidate.taskId === taskId)
  const live = run.taskStates[taskId]
  const childRunId = live?.childRunId ?? task?.childRunId ?? undefined
  const parentChild = childRunId ? run.children[childRunId] : undefined
  const parentChildRef = useRef(parentChild)
  parentChildRef.current = parentChild
  const [snapshot, setSnapshot] = useState<ResearchRunSnapshot | undefined>(
    projectedChildSnapshot(parentChild),
  )
  const [messages, setMessages] = useState<ChildProgressMessage[]>([])
  const [transport, setTransport] = useState<
    'connecting' | 'error' | 'idle' | 'polling' | 'settled' | 'sse'
  >(childRunId ? 'connecting' : 'idle')
  const [transportError, setTransportError] = useState<string | null>(null)
  const [taskResult, setTaskResult] = useState<AgentTaskResultWire | null>(null)
  const [resultLoading, setResultLoading] = useState(false)
  const [resultError, setResultError] = useState<string | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const [cancelError, setCancelError] = useState<string | null>(null)
  // The push marks the covered list `inert`, which silently evicts
  // keyboard focus to <body> — land it on the Back button instead
  // (back-nav focus restore is handled by the run overview).
  const backButtonRef = useRef<HTMLButtonElement | null>(null)
  useEffect(() => {
    backButtonRef.current?.focus({ preventScroll: true })
  }, [taskId])
  const status = task
    ? effectiveAgentTaskStatus(task, live)
    : 'pending'

  useEffect(() => {
    setSnapshot((current) => mergeResearchSnapshot(
      current,
      projectedChildSnapshot(parentChild),
    ))
  }, [parentChild])

  useEffect(() => {
    setSnapshot(projectedChildSnapshot(parentChildRef.current))
    setMessages([])
    setTransportError(null)
    if (!childRunId) {
      setTransport('idle')
      return undefined
    }
    if (!context.clientOptions) {
      setTransport('error')
      setTransportError(t.agent.canvas.child.connectionUnavailable)
      return undefined
    }
    setTransport('connecting')
    const controller = new AbortController()
    void subscribeRunEvents({
      eventsUrl: `/v1/runs/${childRunId}/events`,
      options: context.clientOptions,
      signal: controller.signal,
      onTransportChange: setTransport,
      onEvent: (event: ResearchRunEvent) => {
        setSnapshot((current) => mergeResearchSnapshot(
          current,
          childEventSnapshot(event),
        ))
        const message = childProgressMessage(event)
        if (message) {
          setMessages((current) => {
            if (current.some((item) => item.sequence === message.sequence)) {
              return current
            }
            return [...current.slice(-99), message]
          })
        }
      },
    })
      .then(() => {
        if (!controller.signal.aborted) setTransport('settled')
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return
        setTransport('error')
        setTransportError(error instanceof Error ? error.message : String(error))
      })
    return () => controller.abort()
  }, [childRunId, context.clientOptions, t.agent.canvas.child.connectionUnavailable])

  useEffect(() => {
    setTaskResult(null)
    setResultError(null)
    if (
      !task
      || !context.clientOptions
      || status === 'pending'
      || status === 'running'
      || status === 'cancel_requested'
    ) {
      setResultLoading(false)
      return undefined
    }
    let active = true
    setResultLoading(true)
    void context.loadTaskResult(run.runId, taskId)
      .then((result) => {
        if (active) setTaskResult(result)
      })
      .catch((loadError: unknown) => {
        if (active) {
          setResultError(
            loadError instanceof Error ? loadError.message : String(loadError),
          )
        }
      })
      .finally(() => {
        if (active) setResultLoading(false)
      })
    return () => {
      active = false
    }
  }, [context.clientOptions, context.loadTaskResult, run.runId, status, task, taskId])

  if (!task) return <CanvasMissing label={t.agent.canvas.views.run} />
  const child = childRunId ? run.children[childRunId] : undefined
  const phase = researchNodePhase(child?.currentNode ?? snapshot?.current_node)
  const liveError = live?.error ?? child?.error
  const currentMessage = child?.message ?? snapshot?.last_message
  const localActivity = live?.activity
  const localActivityHistory = live?.activityHistory ?? []
  const liveMetricsSnapshot = snapshotWithResearchMetrics(
    snapshot,
    live?.metrics,
    localRequestCount(task),
  )
  const resultSummary = task.resultSummary || live?.resultSummary
  const resultContent = agentTaskResultContent(
    taskResult,
    resultSummary ?? '',
    resultError,
  )
  const resultMarkdown = resultContent.markdown
  const resultIsPreviewOnly = resultContent.previewOnly
  const metricsSnapshot = taskResult
    ? snapshotWithResearchMetrics(
      liveMetricsSnapshot,
      taskResult.metrics,
      localRequestCount(task),
    )
    : liveMetricsSnapshot
  const displayedError = taskResult?.error?.message ?? liveError
  const duplicatedTerminalErrorIndex = terminalActivityErrorIndex(
    localActivityHistory,
    displayedError,
  )
  const canCancel = canEditAgentRun(run) && Boolean(context.clientOptions) && (
    status === 'pending' || status === 'running'
  )
  const cancellationPending = cancelling || status === 'cancel_requested'

  const requestTaskCancel = async () => {
    setCancelling(true)
    setCancelError(null)
    try {
      await context.cancelTask(run.runId, task.taskId)
    } catch (requestError) {
      setCancelError(
        requestError instanceof Error ? requestError.message : String(requestError),
      )
    } finally {
      setCancelling(false)
    }
  }

  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className={cn(canvasSurfaceClass, 'space-y-4')}>
        <Button
          className="gap-1.5 px-2 text-muted-foreground hover:text-foreground"
          onClick={onBack}
          ref={backButtonRef}
          size="sm"
          type="button"
          variant="ghost"
        >
          <ChevronLeft className="icon-sm" />
          {t.agent.canvas.child.back}
        </Button>
        <div className="flex min-w-0 items-start gap-2">
          <TaskStatusGlyph status={status} />
          <div className="min-w-0 flex-1">
            <p className="break-words t-card text-foreground">{task.title}</p>
            {task.objective && (
              <p className="mt-0.5 break-words t-meta text-muted-foreground">
                {task.objective}
              </p>
            )}
            <p className="mt-1 t-meta-sm text-muted-foreground">
              {agentTaskExecutionLabel(task, t)}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            {(canCancel || cancellationPending) && (
              <Button
                className="h-7 gap-1 px-2 text-xs text-muted-foreground hover:text-destructive"
                disabled={cancellationPending}
                onClick={() => void requestTaskCancel()}
                size="sm"
                type="button"
                variant="ghost"
              >
                <X className="icon-xs" />
                {cancellationPending
                  ? t.agent.canvas.child.cancellingTask
                  : t.agent.canvas.child.cancelTask}
              </Button>
            )}
            <StatusBadge
              density="table"
              label={agentTaskStatusLabel(status, t)}
              tone={taskStatusTone(status)}
            />
          </div>
        </div>

        {childRunId && (
          <div className="rounded-lg border border-border bg-card px-3 py-3">
            {phase ? (
              <PhaseSegments
                activePhase={phase}
                completedPhases={completedResearchPhases(
                  child?.currentNode ?? snapshot?.current_node,
                )}
                labelFor={(item) => phaseLabel(item, t)}
                phases={phaseOrder}
                withLabels
              />
            ) : (
              <p className="t-meta text-muted-foreground" role="status">
                {t.agent.canvas.child.preparing}
              </p>
            )}
            {currentMessage && (
              <p className="mt-2 break-words t-meta text-muted-foreground" role="status">
                {currentMessage}
              </p>
            )}
            <p
              className={cn(
                'mt-2 flex items-center gap-1.5 t-hint',
                transport === 'error' ? 'text-destructive' : 'text-muted-foreground',
              )}
              role="status"
            >
              {transport === 'error' && <AlertTriangle className="icon-xs shrink-0" />}
              {childTransportLabel(transport, t)}
            </p>
            {transportError && (
              <p className="mt-1 break-words t-meta-sm text-destructive">
                {transportError}
              </p>
            )}
          </div>
        )}

        {!childRunId && (localActivity || status === 'running') && (
          <div className="rounded-lg border border-border bg-card px-3 py-3">
            {localActivity ? (
              <p className="break-words t-meta text-muted-foreground" role="status">
                {activityDisplayText(localActivity, t)}
              </p>
            ) : (
              <p className="t-meta text-muted-foreground" role="status">
                {t.agent.canvas.child.preparing}
              </p>
            )}
            {task.toolKind === 'web_instant'
              && (status === 'running' || status === 'cancel_requested') && (
              <IndeterminateTaskRail />
            )}
          </div>
        )}

        <TaskMetrics
          includeZeroClaims={Boolean(childRunId) || task.toolKind === 'web_research'}
          snapshot={metricsSnapshot}
        />

        {!childRunId && localActivityHistory.length > 0 && (
          <div>
            <p className="t-caption uppercase tracking-wide text-muted-foreground">
              {t.agent.canvas.child.operations}
            </p>
            <ul className="mt-1.5 space-y-1.5">
              {localActivityHistory.map((activity, index) => {
                const activityError = index === duplicatedTerminalErrorIndex
                  ? undefined
                  : activity.error
                const attention = Boolean(
                  activityError
                  || activity.fallback
                  || activity.status === 'failed',
                )
                return (
                  <li
                    className={cn(
                      'flex items-start gap-1.5 break-words t-meta-sm',
                      attention ? 'text-warning' : 'text-muted-foreground',
                    )}
                    key={activity.activityId ?? `${activity.at}-${index}`}
                  >
                    {attention && (
                      <AlertTriangle className="mt-0.5 icon-xs shrink-0" />
                    )}
                    <span>
                      {activityDisplayText(activity, t)}
                      {activityError ? ` · ${activityError}` : ''}
                    </span>
                  </li>
                )
              })}
            </ul>
          </div>
        )}

        {messages.length > 0 && (
          <div>
            <p className="t-caption uppercase tracking-wide text-muted-foreground">
              {t.agent.canvas.child.messages}
            </p>
            <ul className="mt-1.5 space-y-1.5">
              {messages.map((message) => (
                <li
                  className={cn(
                    'flex items-start gap-1.5 break-words t-meta-sm',
                    message.severity === 'error'
                      ? 'text-destructive'
                      : message.severity === 'warning'
                        ? 'text-warning'
                        : 'text-muted-foreground',
                  )}
                  key={message.sequence}
                >
                  {message.severity !== 'info' && (
                    <AlertTriangle className="mt-0.5 icon-xs shrink-0" />
                  )}
                  <span>{message.text}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {resultLoading && (
          <div
            className="rounded-lg border border-border bg-surface/50 px-3 py-2.5"
            role="status"
          >
            <span className="sr-only">
              {t.agent.canvas.child.resultLoading}
            </span>
            <div aria-hidden="true" className="space-y-2">
              <Skeleton className="h-4 w-[86%]" />
              <Skeleton className="h-4 w-[72%]" />
              <Skeleton className="h-4 w-[64%]" />
            </div>
          </div>
        )}
        {resultMarkdown && (
          <div className="rounded-lg border border-border bg-surface/50 px-3 py-2.5">
            <p className="t-caption uppercase tracking-wide text-muted-foreground">
              {t.agent.canvas.child.result}
            </p>
            <MarkdownSelectionCopyMenu
              aiGenerated
              className="chat-markdown mt-1 min-w-0 text-sm leading-snug text-foreground/90"
              markdown={resultMarkdown}
            >
              <MarkdownRenderer
                markdown={resultMarkdown}
                variant="chat"
              />
            </MarkdownSelectionCopyMenu>
            {taskResult && taskResult.references.length > 0 && (
              <TaskResultSources references={taskResult.references} />
            )}
          </div>
        )}
        {resultIsPreviewOnly && (
          <p className="flex items-start gap-1.5 break-words t-meta text-warning">
            <AlertTriangle className="mt-0.5 icon-xs shrink-0" />
            {t.agent.canvas.child.resultPreviewOnly}
          </p>
        )}
        {resultError && (
          <p className="break-words t-meta text-warning">
            {t.agent.canvas.child.resultUnavailable} · {resultError}
          </p>
        )}
        {cancelError && (
          <p className="break-words t-meta text-destructive">{cancelError}</p>
        )}
        {live?.fallback && (
          <p className="flex items-start gap-1.5 break-words t-meta text-warning">
            <AlertTriangle className="mt-0.5 icon-xs shrink-0" />
            {t.agent.canvas.child.fallbackUsed}
          </p>
        )}
        {displayedError && (
          <p className="flex items-start gap-1.5 break-words t-meta text-destructive">
            <AlertTriangle className="mt-0.5 icon-xs shrink-0" />
            {displayedError}
          </p>
        )}
        <details className="rounded-md border border-border bg-surface/40 px-3 py-2">
          <summary className="cursor-pointer t-meta font-medium text-muted-foreground">
            {t.agent.canvas.child.technicalDetails}
          </summary>
          <dl className="mt-2 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-3 gap-y-1 t-hint text-muted-foreground">
            <dt>{t.agent.canvas.child.detailChildRun}</dt>
            <dd className="min-w-0 break-all t-mono text-foreground/80">
              {childRunId ?? '—'}
            </dd>
            <dt>{t.agent.canvas.child.detailOperation}</dt>
            <dd className="min-w-0 break-all t-mono text-foreground/80">
              {localActivity?.operationCode
                ?? localActivity?.operation
                ?? task.toolKind}
            </dd>
            <dt>{t.agent.canvas.child.detailAttempt}</dt>
            <dd className="tabular-nums text-foreground/80">
              {live?.attempt ?? child?.attempt ?? 1}
            </dd>
            <dt>{t.agent.canvas.child.detailTransport}</dt>
            <dd className="text-foreground/80">{childTransportLabel(transport, t)}</dd>
            <dt>{t.agent.canvas.child.detailFallback}</dt>
            <dd className="text-foreground/80">
              {live?.fallback
                ? t.agent.canvas.child.detailYes
                : t.agent.canvas.child.detailNo}
            </dd>
            {(taskResult?.error?.code ?? live?.errorCode ?? child?.errorCode) && (
              <>
                <dt>{t.agent.canvas.child.detailErrorCode}</dt>
                <dd className="min-w-0 break-all t-mono text-foreground/80">
                  {taskResult?.error?.code ?? live?.errorCode ?? child?.errorCode}
                </dd>
              </>
            )}
            {taskResult && (
              <>
                <dt>{t.agent.canvas.child.detailPromptTokens}</dt>
                <dd className="tabular-nums text-foreground/80">
                  {taskResult.metrics.prompt_tokens}
                </dd>
                <dt>{t.agent.canvas.child.detailCompletionTokens}</dt>
                <dd className="tabular-nums text-foreground/80">
                  {taskResult.metrics.completion_tokens}
                </dd>
              </>
            )}
          </dl>
        </details>
      </div>
    </ScrollArea>
  )
}

function TaskResultSources({
  references,
}: {
  references: readonly Record<string, unknown>[]
}) {
  const { t } = useLocale()
  const groups = taskResultReferenceGroups(references)
  if (groups.length === 0) return null
  return (
    <section className="mt-3 border-t border-border/70 pt-2.5">
      <p className="t-caption uppercase tracking-wide text-muted-foreground">
        {t.knowledge.sources}
      </p>
      <ul className="mt-1.5 space-y-2">
        {groups.map((group) => {
          if (group.kind === 'web') {
            return (
              <li key={group.reference.key}>
                <WebEvidenceSourceRow reference={group.reference} />
              </li>
            )
          }
          const several = group.references.length > 1
          return (
            <li key={`document:${group.references[0]?.documentId ?? group.title}`}>
              {several && (
                <div className="flex min-w-0 items-center gap-1.5">
                  <FileText className="icon-sm shrink-0 text-muted-foreground/70" />
                  <span className="min-w-0 flex-1 truncate t-list text-foreground">
                    {group.title}
                  </span>
                  <span className="t-hint tabular-nums text-muted-foreground">
                    {group.references.length}
                  </span>
                </div>
              )}
              <ul className={cn('space-y-1', several && 'mt-1 border-l border-border/60 pl-2')}>
                {group.references.map((reference) => (
                  <li className="min-w-0" key={reference.key}>
                    {!several && (
                      <p className="t-list text-foreground">{reference.title}</p>
                    )}
                    {reference.excerpt && (
                      <p className="line-clamp-3 break-words t-meta text-foreground/85">
                        {reference.excerpt}
                      </p>
                    )}
                    {(reference.pageNumber !== null || reference.chunkIndex !== null) && (
                      <p className="mt-0.5 t-meta-sm text-muted-foreground">
                        {[
                          reference.pageNumber === null
                            ? null
                            : t.knowledge.citationPage.replace(
                              '{n}',
                              String(reference.pageNumber),
                            ),
                          reference.chunkIndex === null
                            ? null
                            : t.knowledge.viewerSection.replace(
                              '{n}',
                              String(reference.chunkIndex + 1),
                            ),
                        ].filter(Boolean).join(' · ')}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

function TaskMetrics({
  includeZeroClaims,
  snapshot,
}: {
  includeZeroClaims: boolean
  snapshot: ResearchRunSnapshot | undefined
}) {
  const { t } = useLocale()
  const metrics = agentTaskMetrics(snapshot, includeZeroClaims)
  if (metrics.length === 0) return null
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 t-meta-sm tabular-nums text-muted-foreground">
      {metrics.map((metric) => (
        <span key={metric.kind}>
          <strong className="font-semibold text-foreground">{metric.value}</strong>{' '}
          {metric.kind === 'sources'
            ? t.runCard.sources
            : metric.kind === 'queries'
              ? t.runCard.queries
              : t.runCard.claims}
        </span>
      ))}
    </div>
  )
}

function TaskStatusGlyph({ status }: { status: AgentTaskEffectiveStatus }) {
  if (status === 'completed') {
    return <Check className="mt-0.5 icon-sm shrink-0 text-success" />
  }
  if (status === 'failed') {
    return <X className="mt-0.5 icon-sm shrink-0 text-destructive" />
  }
  if (status === 'insufficient_evidence') {
    return <AlertTriangle className="mt-0.5 icon-sm shrink-0 text-warning" />
  }
  if (status === 'cancelled') {
    return <X className="mt-0.5 icon-sm shrink-0 text-muted-foreground" />
  }
  return (
    <span
      aria-hidden="true"
      className={cn(
        'mt-1.5 size-1.5 shrink-0 rounded-full',
        status === 'running' || status === 'cancel_requested'
          ? 'bg-brand inqtrix-running-dot'
          : 'bg-muted-foreground/30',
      )}
    />
  )
}

function childForTask(
  run: AgentRunRecord,
  task: AgentPlanTaskRecord,
  liveChildRunId: string | undefined,
): AgentChildProgressRecord | undefined {
  const childRunId = liveChildRunId ?? task.childRunId ?? undefined
  if (childRunId) return run.children[childRunId]
  return Object.values(run.children).find((child) => child.taskId === task.taskId)
}

function projectedChildSnapshot(
  child: AgentChildProgressRecord | undefined,
): ResearchRunSnapshot | undefined {
  if (!child) return undefined
  const projected: ResearchRunSnapshot = {
    ...(child.snapshot ?? {}),
    ...(child.currentNode ? { current_node: child.currentNode } : {}),
    ...(child.message ? { last_message: child.message } : {}),
  }
  return snapshotWithResearchMetrics(
    Object.keys(projected).length > 0 ? projected : undefined,
    child.metrics,
  )
}

function localRequestCount(task: AgentPlanTaskRecord): number | undefined {
  if (task.toolKind !== 'web_instant' && task.toolKind !== 'rag_query') {
    return undefined
  }
  return agentTaskExecutionSemantics(task).requestCount
}

function childEventSnapshot(event: ResearchRunEvent): ResearchRunSnapshot | undefined {
  const snapshot = mergeResearchSnapshot(undefined, event.data.snapshot)
  const currentNode = typeof event.data.current_node === 'string'
    ? event.data.current_node
    : undefined
  const message = typeof event.data.message === 'string'
    ? event.data.message
    : undefined
  if (!snapshot && !currentNode && !message) return undefined
  return {
    ...(snapshot ?? {}),
    ...(currentNode ? { current_node: currentNode } : {}),
    ...(message ? { last_message: message } : {}),
  }
}

function childTransportLabel(
  transport: 'connecting' | 'error' | 'idle' | 'polling' | 'settled' | 'sse',
  t: ReturnType<typeof useLocale>['t'],
): string {
  if (transport === 'sse') return t.agent.canvas.child.connectionLive
  if (transport === 'polling') return t.agent.canvas.child.connectionPolling
  if (transport === 'error') return t.agent.canvas.child.connectionError
  if (transport === 'settled') return t.agent.canvas.child.connectionSettled
  if (transport === 'idle') return t.agent.canvas.child.connectionIdle
  return t.agent.canvas.child.connectionConnecting
}

function taskGroups(
  tasks: AgentPlanTaskRecord[],
  run: AgentRunRecord,
): Array<{ kind: AgentTaskGroup; tasks: AgentPlanTaskRecord[] }> {
  const groups: Array<{ kind: AgentTaskGroup; tasks: AgentPlanTaskRecord[] }> = [
    { kind: 'active', tasks: [] },
    { kind: 'attention', tasks: [] },
    { kind: 'completed', tasks: [] },
  ]
  for (const task of tasks) {
    const status = effectiveAgentTaskStatus(task, run.taskStates[task.taskId])
    const kind = agentTaskGroup(
      status,
      run.taskStates[task.taskId]?.fallback === true,
    )
    groups.find((group) => group.kind === kind)?.tasks.push(task)
  }
  return groups.filter((group) => group.tasks.length > 0)
}

function taskGroupLabel(
  kind: AgentTaskGroup,
  t: ReturnType<typeof useLocale>['t'],
): string {
  if (kind === 'attention') return t.agent.canvas.taskGroups.attention
  if (kind === 'completed') return t.agent.canvas.taskGroups.completed
  return t.agent.canvas.taskGroups.active
}

function taskStatusTone(status: AgentTaskEffectiveStatus): StatusTone {
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'destructive'
  if (status === 'insufficient_evidence') return 'warning'
  if (status === 'cancel_requested') return 'warning'
  if (status === 'running') return 'brand'
  return 'neutral'
}

function runStatusTone(status: AgentRunRecord['status']): StatusTone {
  if (status === 'completed') return 'success'
  if (status === 'failed' || status === 'cancelled') return 'destructive'
  if (status === 'waiting_for_approval' || status === 'waiting_for_input') {
    return 'warning'
  }
  if (isActiveAgentRun(status)) return 'brand'
  return 'neutral'
}

function taskButtonId(taskId: string): string {
  return `agent-task-${encodeURIComponent(taskId)}`
}

function IndeterminateTaskRail() {
  return (
    <div
      aria-hidden="true"
      className="mt-2 h-1 overflow-hidden rounded-full bg-brand/15"
    >
      <span className="inqtrix-segment-breathe block h-full rounded-full bg-brand" />
    </div>
  )
}

function CanvasMissing({ label }: { label: string }) {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center px-6">
      <p className="t-meta text-muted-foreground">{label}</p>
    </div>
  )
}

function noop() {}
async function asyncNoop() {}

const emptyRunPlaceholder = {
  runId: '',
  kind: 'agent',
  question: '',
  status: 'queued',
  phase: 'intake',
  station: 'intake',
  createdAt: '',
  lastSequence: 0,
  planStale: false,
  approvals: [],
  approvalsStale: false,
  clarifications: [],
  clarificationsStale: false,
  artifactOrder: [],
  artifacts: {},
  artifactsStale: false,
  taskStates: {},
  children: {},
  patchStale: false,
} as unknown as AgentRunRecord


/**
 * The Agent Desk view registry — a MODULE CONSTANT: renderer identities
 * never change across workspace renders, so the active view (editor
 * cursor, task SSE subscription) survives state updates. Data flows via
 * `AgentCanvasReactContext`.
 */
export const AGENT_CANVAS_REGISTRY = {
  diff: DiffCanvasView,
  document: DocumentCanvasView,
  evidence: EvidenceCanvasView,
  file: FileCanvasView,
  patch: PatchCanvasView,
  plan: PlanCanvasView,
  run: RunCanvasView,
} as const
