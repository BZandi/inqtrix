import type { JobFilter, JobPhase, JobStatus, AppView } from '@/features/researchDesk/types'
import type { Locale } from '@/i18n/translations'
import type { ContrastMode, ThemeMode, ThemePreset } from '@/theme/ThemeProvider'
import type {
  ResearchClaim,
  ResearchMetrics,
  ReportReference,
  ChatModelTier,
  ResearchRunEvent,
  ResearchRunMode,
  ResearchRunResult,
  ResearchRunSummary,
  ResearchRunSnapshot,
  ResearchSource,
} from '@/features/researchRuns/types'

export const PROJECT_SCHEMA_VERSION = 1

export type ProjectConnectionKind = 'demo' | 'directory' | 'download' | 'local'

export type ProjectConnection = {
  directoryHandle?: FileSystemDirectoryHandle
  directoryName?: string
  kind: ProjectConnectionKind
  writable: boolean
}

export type ProjectMetadata = {
  createdAt: string
  name: string
  schemaVersion: typeof PROJECT_SCHEMA_VERSION
  updatedAt: string
}

export type ProjectUiState = {
  activeFilter: JobFilter
  activeView: AppView
  chatChainingEnabled: boolean
  expandedJobId: string | null
  isChatHistoryVisible: boolean
  isComposerVisible: boolean
  isReportExpanded: boolean
  isReportVisible: boolean
  pendingChatAttachmentRefs: ChatContextReferenceRecord[]
  pendingChatReportRunId: string | null
  selectedChatModelTier: ChatModelTier | null
  selectedChatThreadId: string | null
  selectedJobId: string | null
  selectedStack: string
}

export type EditorDocumentSource = 'blank' | 'imported-research-report' | 'pasted'

export type EditorFolderRecord = {
  createdAt: string
  id: string
  title: string
  updatedAt: string
}

export type EditorCommentStatus = 'open' | 'resolved' | 'stale'

export type EditorCommentKind = 'collect' | 'inline_edit' | 'evidence_review'

export type EditorEvidencePreset = 'add_sources' | 'fact_check' | 'verify_citations'

export type EditorCommentAnchorRecord = {
  blockId?: string
  from: number
  quoteAfter: string
  quoteBefore: string
  selectedMarkdown?: string
  selectedText: string
  to: number
}

export type EditorCommentThreadRecord = {
  anchor: EditorCommentAnchorRecord
  commentMarkdown: string
  createdAt: string
  documentId: string
  evidencePreset?: EditorEvidencePreset
  id: string
  kind: EditorCommentKind
  status: EditorCommentStatus
  updatedAt: string
}

export type EditorSuggestionStatus = 'pending' | 'accepted' | 'rejected' | 'stale'
export type EditorSuggestionEditPosition = 'replace' | 'before' | 'after' | 'append'
export type EditorSuggestionRevisionSource = 'llm_refine' | 'manual_edit'

export type EditorSuggestionRevisionRecord = {
  changeSummary?: string[]
  createdAt: string
  instruction?: string
  proposedText: string
  source: EditorSuggestionRevisionSource
  warnings?: string[]
}

export type EditorSuggestionOrigin =
  | { commentId?: string; kind: 'global_run' }
  | { commentId: string; kind: 'inline_edit' }
  | { commentId: string; kind: 'evidence_review'; preset: EditorEvidencePreset }

export type EditorSuggestionEvidenceSource = {
  title: string
  url: string
}

export type EditorSuggestionEvidence = {
  mode: EditorEvidencePreset
  sources: EditorSuggestionEvidenceSource[]
}

export type EditorSuggestionRecord = {
  anchor: EditorCommentAnchorRecord
  anchorText?: string
  blockId: string
  changeSummary?: string[]
  createdAt: string
  documentId: string
  editPosition?: EditorSuggestionEditPosition
  evidence?: EditorSuggestionEvidence
  groupId: string
  id: string
  originalMarkdown?: string
  originalText: string
  origin: EditorSuggestionOrigin
  proposedText: string
  revision?: number
  revisionHistory?: EditorSuggestionRevisionRecord[]
  status: EditorSuggestionStatus
  updatedAt: string
  warnings?: string[]
}

export type EditorSuggestionGroupRecord = {
  assistantMessage?: string
  createdAt: string
  documentId: string
  id: string
  origin: EditorSuggestionOrigin
  warnings?: string[]
}

export type EditorDocumentRecord = {
  contentMarkdown: string
  createdAt: string
  diffAnchorMarkdown?: string
  diffAnchorUpdatedAt?: string
  folderId: string | null
  id: string
  revision: number
  source: EditorDocumentSource
  sourceRunId?: string
  title: string
  updatedAt: string
}

export type EditorPanelTab = 'comments' | 'assistant'

export type EditorViewMode = 'live' | 'source'

export type EditorUiState = {
  activeDocumentId: string | null
  assistantDraft: string
  isAssistantVisible: boolean
  isCommentPanelVisible: boolean
  isDiffVisible: boolean
  isTreeVisible: boolean
  openDocumentIds: string[]
  panelTab: EditorPanelTab
  selectedCommentId: string | null
  viewMode: EditorViewMode
}

export type ProjectPreferences = {
  contrastMode: ContrastMode
  locale: Locale
  theme: ThemeMode
  themePreset: ThemePreset
}

export type ResearchRunSource = 'api' | 'imported' | 'mock'

export type ResearchRunEventKind = 'progress' | 'system'

export type ResearchRunEventSeverity = 'error' | 'info' | 'success' | 'warning'

export type ResearchRunEventRecord = {
  active?: boolean
  createdAt: string
  id: string
  kind: ResearchRunEventKind
  phase?: JobPhase
  severity: ResearchRunEventSeverity
  title: string
}

export type ResearchRunPhaseState = {
  activePhase: JobPhase
  completedPhases: JobPhase[]
}

export type ResearchRunCardSummary = {
  queueNote?: string
  score?: string
  title: string
}

export type ResearchRunCardMetrics = {
  queries: number
  rounds: string
  sources: number
}

export type ResearchRunResultRecord = {
  markdown: string
  metrics?: Partial<ResearchMetrics>
  references: ReportReference[]
  topClaims: ResearchClaim[]
  topSources: ResearchSource[]
  usage?: ResearchRunResult['usage']
}

export type ResearchRunRecord = {
  agentOverrides: Record<string, unknown>
  createdAt: string
  durationSeconds?: number
  error?: string
  events: ResearchRunEventRecord[]
  finishedAt?: string
  metrics: ResearchRunCardMetrics
  mode?: ResearchRunMode
  phaseState: ResearchRunPhaseState
  queuePosition?: number | null
  result?: ResearchRunResultRecord
  runId: string
  snapshot?: ResearchRunSnapshot
  source: ResearchRunSource
  stack: string
  startedAt?: string
  status: JobStatus
  submittedAt: string
  summary: ResearchRunCardSummary
}

export type ChatRole = 'assistant' | 'user'

export type ChatRuleRecord = {
  contentMarkdown: string
  createdAt: string
  id: string
  label: string
  title: string
  updatedAt: string
}

export type ChatThreadGroupRecord = {
  createdAt: string
  id: string
  title: string
  updatedAt: string
}

export type ResearchReportAttachmentRecord = {
  attachedAt: string
  contentMarkdown: string
  kind: 'research-report'
  label?: string
  runId: string
  title: string
}

export type ChatRuleAttachmentRecord = {
  attachedAt: string
  contentMarkdown: string
  kind: 'chat-rule'
  label: string
  ruleId: string
  title: string
}

export type FileParseStatus = 'parsed' | 'partial' | 'unsupported' | 'error'

export type FileAssetOrigin = 'chat' | 'editor' | 'library'

export type FileSectionKind = 'temporary' | 'custom'

export type FileLibrarySectionRecord = {
  createdAt: string
  id: string
  kind: FileSectionKind
  title: string
  updatedAt: string
}

export type FileGroupRecord = {
  createdAt: string
  id: string
  sectionId: string
  title: string
  updatedAt: string
}

export type FileAssetRecord = {
  createdAt: string
  extractedText: string
  fileName: string
  groupId: string | null
  id: string
  label: string
  mimeType: string
  origin: FileAssetOrigin
  pageCount: number | null
  parseStatus: FileParseStatus
  parseWarning: string | null
  sectionId: string
  sizeBytes: number
  textTruncated: boolean
  title: string
  updatedAt: string
}

export type FileAssetAttachmentRecord = {
  attachedAt: string
  contentMarkdown: string
  fileId: string
  kind: 'file-asset'
  label: string
  pageCount: number | null
  sizeBytes: number
  title: string
}

export type FileGroupAttachmentRecord = {
  attachedAt: string
  contentMarkdown: string
  fileId: string
  groupId: string
  groupLabel: string
  kind: 'file-group'
  label: string
  pageCount: number | null
  sizeBytes: number
  title: string
}

export type ChatMessageAttachmentRecord =
  | ChatRuleAttachmentRecord
  | FileAssetAttachmentRecord
  | FileGroupAttachmentRecord
  | ResearchReportAttachmentRecord

export type ChatContextReferenceRecord =
  | { kind: 'chat-rule'; ruleId: string }
  | { kind: 'research-report'; runId: string }
  | { fileId: string; kind: 'file-asset' }
  | { groupId: string; kind: 'file-group' }

export type ChatMessageModelResolutionRecord = {
  effort: string
  effortSource: string
  model: string
  modelSource: string
  requestedTier: string
  tier: string
}

export type ChatChainStepStatus = 'ok' | 'error' | 'stopped'

export type ChatChainStepRecord = {
  label: string
  output: string
  status: ChatChainStepStatus
}

export type ChatMessageRecord = {
  attachments?: ChatMessageAttachmentRecord[]
  chainTrace?: ChatChainStepRecord[]
  contentMarkdown: string
  createdAt: string
  id: string
  modelResolution?: ChatMessageModelResolutionRecord
  role: ChatRole
}

export type ChatThreadRecord = {
  createdAt: string
  id: string
  messages: ChatMessageRecord[]
  preview: string
  source: 'api' | 'imported' | 'mock'
  title: string
  updatedAt: string
}

export type ProjectState = {
  chatRuleOrder: string[]
  chatRules: Record<string, ChatRuleRecord>
  chatThreadGroupMemberships: Record<string, string | null>
  chatThreadGroupOrder: string[]
  chatThreadGroups: Record<string, ChatThreadGroupRecord>
  chatThreadOrder: string[]
  chatThreads: Record<string, ChatThreadRecord>
  connection: ProjectConnection
  dirty: boolean
  editorComments: Record<string, EditorCommentThreadRecord>
  editorDocumentOrder: string[]
  editorDocuments: Record<string, EditorDocumentRecord>
  editorFolderOrder: string[]
  editorFolders: Record<string, EditorFolderRecord>
  editorSuggestionGroups: Record<string, EditorSuggestionGroupRecord>
  editorSuggestions: Record<string, EditorSuggestionRecord>
  editorUi: EditorUiState
  fileAssetOrder: string[]
  fileAssets: Record<string, FileAssetRecord>
  fileGroupOrder: string[]
  fileGroups: Record<string, FileGroupRecord>
  fileLibrarySectionOrder: string[]
  fileLibrarySections: Record<string, FileLibrarySectionRecord>
  localRunCounter: number
  preferences: ProjectPreferences
  project: ProjectMetadata
  researchRunOrder: string[]
  researchRuns: Record<string, ResearchRunRecord>
  ui: ProjectUiState
  workspaceId: string
}

export type ProjectWriteResult = {
  connection: ProjectConnection
  savedAt: string
}

export function fromRunSummary(
  summary: ResearchRunSummary,
  fallbackStack: string,
): ResearchRunRecord {
  const now = new Date().toISOString()
  const submittedAt = toIsoString(summary.created_at) ?? now
  const startedAt = toIsoString(summary.started_at)
  const finishedAt = toIsoString(summary.finished_at)
  const snapshot = summary.snapshot
  const maxRounds = snapshot.max_rounds ?? numberFromUnknown(summary.agent_overrides.max_rounds)
  const currentRounds = snapshot.active_round ?? snapshot.completed_rounds ?? 0
  const title = summary.question.trim() || summary.run_id

  return {
    agentOverrides: summary.agent_overrides,
    createdAt: submittedAt,
    durationSeconds: terminalStatus(summary.status)
      ? summary.elapsed_seconds ?? undefined
      : undefined,
    error: summary.error?.message,
    events: [],
    finishedAt: finishedAt ?? undefined,
    metrics: {
      queries: snapshot.total_queries ?? 0,
      rounds: maxRounds ? `${currentRounds} / ${maxRounds}` : String(currentRounds),
      sources: snapshot.total_sources ?? snapshot.total_citations ?? 0,
    },
    mode: summary.mode,
    phaseState: {
      activePhase: nodeToPhase(snapshot.current_node),
      completedPhases: completedPhasesFromNode(snapshot.current_node),
    },
    queuePosition: summary.queue_position,
    runId: summary.run_id,
    snapshot,
    source: 'api',
    stack: summary.stack || fallbackStack,
    startedAt: startedAt ?? undefined,
    status: normalizeRunStatus(summary.status),
    submittedAt,
    summary: {
      queueNote: queueNoteFromSummary(summary),
      score: confidenceScore(snapshot.confidence),
      title,
    },
  }
}

export function mergeRunSummary(
  current: ResearchRunRecord | undefined,
  summary: ResearchRunSummary,
  fallbackStack: string,
): ResearchRunRecord {
  const next = fromRunSummary(summary, fallbackStack)
  if (!current) return next

  return {
    ...next,
    events: current.events,
    result: current.result ?? next.result,
    summary: {
      ...next.summary,
      score: next.summary.score ?? current.summary.score,
    },
  }
}

export function applyRunEvent(
  record: ResearchRunRecord,
  event: ResearchRunEvent,
): ResearchRunRecord {
  const snapshot = snapshotFromEvent(event) ?? record.snapshot
  const nextStatus = statusFromEvent(event) ?? record.status
  const eventRecord = eventRecordFromRunEvent(event)
  const events = eventRecord
    ? [
      ...record.events.map((item) => ({ ...item, active: false })),
      eventRecord,
    ]
    : record.events
  const deduplicatedEvents = events.filter((item, index, allEvents) => (
    allEvents.findIndex((candidate) => candidate.id === item.id) === index
  ))
  const updated = applySnapshotToRecord(record, snapshot)

  return {
    ...updated,
    error: errorFromEvent(event) ?? updated.error,
    events: deduplicatedEvents,
    finishedAt: terminalStatus(nextStatus) && !updated.finishedAt
      ? toIsoString(event.created_at) ?? new Date().toISOString()
      : updated.finishedAt,
    status: nextStatus,
    summary: {
      ...updated.summary,
      queueNote: nextStatus === 'queued' ? updated.summary.queueNote : undefined,
      score: updated.summary.score,
    },
  }
}

export function attachRunResult(
  record: ResearchRunRecord,
  result: ResearchRunResult,
): ResearchRunRecord {
  const maxRounds = record.snapshot?.max_rounds
    ?? numberFromUnknown(record.agentOverrides.max_rounds)
    ?? result.metrics.rounds

  return {
    ...record,
    durationSeconds: result.metrics.elapsed_seconds,
    finishedAt: new Date().toISOString(),
    metrics: {
      queries: result.metrics.total_queries,
      rounds: `${result.metrics.rounds} / ${maxRounds}`,
      sources: result.metrics.total_citations,
    },
    phaseState: {
      activePhase: 'answer',
      completedPhases: ['analysis', 'planning', 'search', 'evaluation', 'answer'],
    },
    result: {
      markdown: result.answer,
      metrics: result.metrics,
      references: result.references ?? [],
      topClaims: result.top_claims,
      topSources: result.top_sources,
      usage: result.usage,
    },
    status: 'completed',
    summary: {
      ...record.summary,
      queueNote: undefined,
      score: `${result.metrics.confidence.toFixed(1)} / 10`,
    },
  }
}

function normalizeRunStatus(status: ResearchRunSummary['status']): JobStatus {
  return status
}

function toIsoString(timestamp: number | null) {
  if (timestamp === null) return null
  return new Date(timestamp * 1000).toISOString()
}

function nodeToPhase(node?: string): JobPhase {
  return phaseFromNode(node) ?? 'analysis'
}

function phaseFromNode(node?: string): JobPhase | undefined {
  if (node === 'classify') return 'analysis'
  if (node === 'plan') return 'planning'
  if (node === 'search') return 'search'
  if (node === 'evaluate') return 'evaluation'
  if (node === 'answer' || node === 'direct_llm') return 'answer'
  return undefined
}

function completedPhasesFromNode(node?: string): JobPhase[] {
  if (node === 'plan') return ['analysis']
  if (node === 'search') return ['analysis', 'planning']
  if (node === 'evaluate') return ['analysis', 'planning', 'search']
  if (node === 'answer') return ['analysis', 'planning', 'search', 'evaluation']
  if (node === 'direct_llm') return ['analysis']
  return []
}

function applySnapshotToRecord(
  record: ResearchRunRecord,
  snapshot: ResearchRunSnapshot | undefined,
): ResearchRunRecord {
  if (!snapshot) return record

  const maxRounds = snapshot.max_rounds
    ?? record.snapshot?.max_rounds
    ?? numberFromUnknown(record.agentOverrides.max_rounds)
  const currentRounds = snapshot.active_round
    ?? snapshot.completed_rounds
    ?? record.snapshot?.active_round
    ?? record.snapshot?.completed_rounds
    ?? 0
  const confidence = snapshot.confidence ?? record.snapshot?.confidence

  return {
    ...record,
    metrics: {
      queries: snapshot.total_queries ?? record.metrics.queries,
      rounds: maxRounds ? `${currentRounds} / ${maxRounds}` : String(currentRounds),
      sources: snapshot.total_sources ?? snapshot.total_citations ?? record.metrics.sources,
    },
    phaseState: {
      activePhase: nodeToPhase(snapshot.current_node),
      completedPhases: completedPhasesFromNode(snapshot.current_node),
    },
    snapshot,
    summary: {
      ...record.summary,
      queueNote: record.status === 'queued' ? record.summary.queueNote : undefined,
      score: confidenceScore(confidence) ?? record.summary.score,
    },
  }
}

function snapshotFromEvent(event: ResearchRunEvent): ResearchRunSnapshot | undefined {
  const snapshot = event.data.snapshot
  if (!snapshot || typeof snapshot !== 'object' || Array.isArray(snapshot)) return undefined
  return snapshot as ResearchRunSnapshot
}

function statusFromEvent(event: ResearchRunEvent): JobStatus | undefined {
  if (event.type === 'inqtrix.run.queued') return 'queued'
  if (event.type === 'inqtrix.run.started') return 'running'
  if (event.type === 'inqtrix.run.completed') return 'completed'
  if (event.type === 'inqtrix.run.failed') return 'failed'
  if (event.type === 'inqtrix.run.cancelled') return 'cancelled'

  const status = event.data.status
  if (
    status === 'queued'
    || status === 'running'
    || status === 'completed'
    || status === 'failed'
    || status === 'cancelled'
    || status === 'expired'
  ) {
    return status
  }
  return undefined
}

function terminalStatus(status: JobStatus) {
  return status === 'completed'
    || status === 'failed'
    || status === 'cancelled'
    || status === 'expired'
}

function eventRecordFromRunEvent(event: ResearchRunEvent): ResearchRunEventRecord | null {
  if (!isVisibleProtocolEvent(event)) return null
  const phase = phaseFromRunEvent(event)

  return {
    active: !terminalStatus(statusFromEvent(event) ?? 'running') || undefined,
    createdAt: toIsoString(event.created_at) ?? new Date().toISOString(),
    // Model-resolution events fire once per node per round but the model is
    // stable per node, so give them a node-stable id; the dedup below then
    // keeps one row per node instead of one per round (no timeline spam).
    id: event.type === 'inqtrix.node.model_resolution'
      ? `${event.run_id}-model-${stringFromUnknown(event.data.node) ?? event.sequence}`
      : `${event.run_id}-${event.sequence}`,
    kind: eventKindFromRunEvent(event),
    ...(phase ? { phase } : {}),
    severity: eventSeverityFromRunEvent(event),
    title: titleFromRunEvent(event),
  }
}

function phaseFromRunEvent(event: ResearchRunEvent): JobPhase | undefined {
  const explicitPhase = phaseFromUnknown(event.data.phase)
  if (explicitPhase) return explicitPhase

  const snapshotPhase = phaseFromNode(snapshotFromEvent(event)?.current_node)
  if (snapshotPhase) return snapshotPhase

  return phaseFromNode(stringFromUnknown(event.data.node))
}

function phaseFromUnknown(value: unknown): JobPhase | undefined {
  if (
    value === 'analysis'
    || value === 'planning'
    || value === 'search'
    || value === 'evaluation'
    || value === 'answer'
  ) {
    return value
  }
  return phaseFromNode(stringFromUnknown(value))
}

function isVisibleProtocolEvent(event: ResearchRunEvent) {
  if (event.type === 'inqtrix.progress.message') {
    const message = stringFromUnknown(event.data.message)
    return Boolean(message && message !== 'done' && !message.startsWith('chat_only.'))
  }
  if (
    event.type === 'inqtrix.run.snapshot'
    || event.type === 'inqtrix.run.started'
    || event.type === 'inqtrix.node.started'
    || event.type === 'inqtrix.node.finished'
    || event.type === 'inqtrix.output_text.delta'
  ) {
    return false
  }
  return event.type === 'inqtrix.run.completed'
    || event.type === 'inqtrix.run.cancel_requested'
    || event.type === 'inqtrix.run.cancelled'
    || event.type === 'inqtrix.run.failed'
    || event.type === 'inqtrix.node.failed'
    || event.type === 'inqtrix.node.model_resolution'
}

function eventKindFromRunEvent(event: ResearchRunEvent): ResearchRunEventKind {
  return event.type === 'inqtrix.progress.message' ? 'progress' : 'system'
}

function eventSeverityFromRunEvent(event: ResearchRunEvent): ResearchRunEventSeverity {
  const severity = event.data.severity
  if (severity === 'warning' || severity === 'error' || severity === 'success') {
    return severity
  }
  if (event.type === 'inqtrix.run.completed') return 'success'
  if (event.type === 'inqtrix.run.failed' || event.type === 'inqtrix.node.failed') return 'error'
  if (event.type === 'inqtrix.run.cancel_requested' || event.type === 'inqtrix.run.cancelled') return 'warning'

  const message = stringFromUnknown(event.data.message) ?? titleFromRunEvent(event)
  return warningLikeMessage(message) ? 'warning' : 'info'
}

function titleFromRunEvent(event: ResearchRunEvent) {
  const message = stringFromUnknown(event.data.message)
  if (message) return message

  const node = stringFromUnknown(event.data.node)
  if (event.type === 'inqtrix.node.model_resolution' && node) {
    const model = stringFromUnknown(event.data.model) || '(default)'
    const source = stringFromUnknown(event.data.model_source) ?? ''
    const effort = stringFromUnknown(event.data.effort) ?? ''
    const sourceLabel = source.startsWith('tier:')
      ? `${source.slice(5)} tier`
      : source === 'reasoning_model_default'
        ? 'reasoning_model default'
        : source === 'per_node_override'
          ? 'per-node override'
          : source
    const effortSuffix = effort && effort !== 'none' ? `, effort ${effort}` : ''
    return `${node}: ${model} (${sourceLabel}${effortSuffix})`
  }
  if (event.type === 'inqtrix.node.failed' && node) return `Failed ${node}`
  if (event.type === 'inqtrix.run.queued') return 'Queued'
  if (event.type === 'inqtrix.run.completed') return 'Run completed'
  if (event.type === 'inqtrix.run.cancel_requested') return 'Cancellation requested'
  if (event.type === 'inqtrix.run.cancelled') return 'Run cancelled'
  if (event.type === 'inqtrix.run.failed') return errorFromEvent(event) ?? 'Run failed'
  return event.type.replace(/^inqtrix\./, '').replace(/\./g, ' ')
}

function warningLikeMessage(message: string) {
  return /\b(ALGO-FAIL|Warnung|Warning|failed|fehlgeschlagen|violated|verletzt|unvollstaendig|unvollständig|Fallback|fallback|Kontextfenster|context window)\b/i.test(message)
}

function errorFromEvent(event: ResearchRunEvent) {
  const error = event.data.error
  if (!error || typeof error !== 'object' || Array.isArray(error)) return undefined
  return stringFromUnknown((error as { message?: unknown }).message)
}

function queueNoteFromSummary(summary: ResearchRunSummary) {
  if (summary.error?.message) return summary.error.message
  if (summary.queue_position) return `Queue position ${summary.queue_position}`
  return undefined
}

function confidenceScore(confidence: number | undefined) {
  return confidence ? `${confidence.toFixed(1)} / 10` : undefined
}

function numberFromUnknown(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function stringFromUnknown(value: unknown) {
  return typeof value === 'string' && value.trim() ? value : undefined
}
