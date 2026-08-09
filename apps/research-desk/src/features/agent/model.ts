/**
 * Agent Desk record model (plan §5.6) — separate from ResearchRunRecord.
 *
 * Rows are the truth (rule R1): the control surfaces (plan, approvals,
 * clarifications, artifacts) are FETCHED rows converted here; SSE events only
 * flip `*Stale` flags and carry live task/phase state. All converters are
 * pure so they stay unit-testable without React.
 */

import type {
  AgentApprovalWire,
  AgentArtifactDetailWire,
  AgentArtifactMetaWire,
  AgentClarificationWire,
  AgentPatchWire,
  AgentPlanTaskWire,
  AgentPlanWire,
} from './types'
import type {
  ResearchRunAccess,
  ResearchRunSnapshot,
  ResearchRunStatus,
  ResearchRunSummary,
} from '@/features/researchRuns/types'
import type {
  AgentSessionModelSelection,
  AgentSourcePolicy,
} from './executionPolicy'

// --- Pulse-track stations (the signature progress line) ---------------------

/**
 * The metro-line stations of the pulse track. Deliberately fewer than the
 * backend phases: `clarification` is a GATE overlay (not a station) and
 * `evidence` folds into synthesis — six stations keep the track compact.
 */
export const AGENT_PHASE_STATIONS = [
  'intake',
  'discovery',
  'planning',
  'execution',
  'synthesis',
  'critic',
] as const

export type AgentPhaseStation = (typeof AGENT_PHASE_STATIONS)[number]

/**
 * Backend phase -> station index; `null` keeps the previous station (gates
 * and unknown phases must never bounce the track backwards).
 */
export function agentPhaseStation(phase: string): AgentPhaseStation | null {
  switch (phase) {
    case 'intake':
      return 'intake'
    case 'discovery':
      return 'discovery'
    case 'planning':
      return 'planning'
    case 'execution':
      return 'execution'
    case 'evidence':
    case 'synthesis':
      return 'synthesis'
    case 'critic':
      return 'critic'
    default:
      return null
  }
}

export function isTerminalAgentPhase(phase: string): boolean {
  return phase === 'done' || phase === 'cancelled'
}

// --- Records -----------------------------------------------------------------

export type AgentPlanTaskRecord = {
  taskId: string
  ordinal: number
  title: string
  toolKind: AgentPlanTaskWire['tool_kind']
  objective: string
  queries: string[]
  gapIds: string[]
  dependsOn: string[]
  budget: Record<string, unknown>
  params: Record<string, unknown>
  expectedOutput: string
  isFalsification: boolean
  status: string
  childRunId: string | null
  resultSummary: string
}

export type AgentPlanVersionRecord = {
  planId: string
  version: number
  status: string
  createdBy: 'agent' | 'user'
  reason: string
  createdAt: number
}

export type AgentPlanRecord = AgentPlanVersionRecord & {
  summaryMarkdown: string
  assumptions: string[]
  successCriteria: string[]
  tasks: AgentPlanTaskRecord[]
  versions: AgentPlanVersionRecord[]
}

export type AgentApprovalRecord = {
  approvalId: string
  kind: AgentApprovalWire['kind']
  status: AgentApprovalWire['status']
  subjectType: string
  subjectId: string
  payload: Record<string, unknown>
  decision: string
  note: string
  createdAt: number
  decidedAt: number | null
}

export type AgentClarificationOption = {
  id: string
  label: string
  description: string
}

export type AgentClarificationQuestion = {
  id: string
  prompt: string
  options: AgentClarificationOption[]
  multiSelect: boolean
}

export type AgentClarificationRecord = {
  clarificationId: string
  question: string
  options: AgentClarificationOption[]
  /** Structured round questions; empty = legacy single-text round. */
  questions: AgentClarificationQuestion[]
  /** Structured answers by question id (empty until answered). */
  answers: Record<string, { optionIds: string[]; text: string }>
  defaultAssumption: string
  status: AgentClarificationWire['status']
  answer: string
  optionId: string
  createdAt: number
  answeredAt: number | null
}

export type AgentArtifactRecord = {
  artifactId: string
  kind: AgentArtifactMetaWire['kind']
  title: string
  status: AgentArtifactMetaWire['status'] | 'interrupted'
  revision: number
  updatedBy: 'agent' | 'user'
  refsCount: number
  createdAt: number
  updatedAt: number
  /** Body + refs, present only after a detail fetch (list stays lean). */
  contentMarkdown?: string
  /** Kind-specific detail payload. Undefined means the detail has not been
   * fetched; an empty object is a fetched legacy artifact without payload. */
  payload?: Record<string, unknown>
  refs?: Record<string, unknown>[]
  revisions?: { revision: number; createdBy: string; createdAt: number }[]
  /** Live answer publication identity/UTF-8 cursor. Wire artifact fetches do
   * not carry these; they exist only while SSE deltas are authoritative. */
  publicationId?: string
  publicationOffset?: number
  /** The streamed body remains visible until the authoritative artifact
   * detail (including references) replaces it after publication. */
  publicationNeedsReconcile?: boolean
}

/** One editor patch proposed by the run (M7); mirrors the wire detail. */
export type AgentPatchRecord = {
  patchId: string
  documentId: string
  source: AgentPatchWire['source']
  status: AgentPatchWire['status']
  summary: string
  edits: AgentPatchWire['edits']
  warnings: string[]
  revisionBefore: number
  appliedRevision: number | null
  appliedEditIds: string[] | null
  note: string
  documentRevision: number
}

export function agentPatchFromWire(wire: AgentPatchWire): AgentPatchRecord {
  return {
    patchId: wire.patch_id,
    documentId: wire.document_id,
    source: wire.source,
    status: wire.status,
    summary: wire.summary,
    edits: wire.edits.map((edit) => ({ ...edit })),
    warnings: [...wire.warnings],
    revisionBefore: wire.revision_before,
    appliedRevision: wire.applied_revision,
    appliedEditIds: wire.applied_edit_ids ? [...wire.applied_edit_ids] : null,
    note: wire.note,
    documentRevision: wire.document_revision,
  }
}

/** THE settled-task vocabulary: statuses whose result rows exist and no
 * longer change. One definition for event reduction, result caching and
 * prefetch eligibility (Prinzip 4). */
export const TERMINAL_AGENT_TASK_STATUSES: ReadonlySet<string> = new Set([
  'cancelled',
  'completed',
  'failed',
  'insufficient_evidence',
  'skipped',
])

/** Live per-task state derived from `inqtrix.agent.task.*` events. */
export type AgentTaskLiveState = {
  status:
    | 'running'
    | 'cancel_requested'
    | 'cancelled'
    | 'completed'
    | 'failed'
    | 'insufficient_evidence'
    | 'skipped'
  /** Terminal task outcome as reported (completed|failed|insufficient_evidence…). */
  outcome?: string
  attempt: number
  childRunId?: string
  error?: string
  /** Latest task-scoped operation for local Instant/Knowledge work. */
  activity?: AgentActivityRecord
  /** Bounded invocation-level history for the selected task detail. */
  activityHistory?: AgentActivityRecord[]
  /** Truthful terminal counters emitted by the task executor. */
  metrics?: Record<string, unknown>
  /** Immediate task outcome from the terminal event. The authoritative plan
   * row replaces it on refresh, but live cards must not wait for that fetch. */
  resultSummary?: string
  errorCode?: string
  fallback?: boolean
  /** First task/activity boundary (unix seconds). Retries retain the original
   * start so the card reports the complete work-unit duration. */
  startedAt?: number
  /** Earliest truthful terminal operation boundary. A later wave-level
   * task.finished event must not inflate parallel task durations. */
  finishedAt?: number
}

/** Latest child-run snapshot (throttled `child.progress` events). */
export type AgentChildProgressRecord = {
  childRunId: string
  taskId: string
  /** Identity-only child-start events carry no snapshot. The first real
   * research snapshot fills this field; absence means "preparing", never
   * "answer". */
  snapshot?: ResearchRunSnapshot
  /** Durable parent-stream projection. These fields keep the parent work-unit
   * useful without opening a second child subscription. */
  runStatus?: string
  currentNode?: string
  message?: string
  metrics?: Record<string, unknown>
  attempt?: number
  error?: string
  errorCode?: string
  updatedAt?: number
}

/** The one-line live activity readout (signature activity line). */
export type AgentActivityRecord = {
  activityId?: string
  kind: string
  detail: string
  label?: string
  status?: string
  operation?: import('./activityPresentation').AgentOperation
  operationCode?: string
  current?: number
  total?: number
  count?: number
  taskId?: string
  purpose?: string
  metrics?: Record<string, unknown>
  attempt?: number
  error?: string
  errorCode?: string
  fallback?: boolean
  at: number
}

/**
 * One transcript line of the main-window step stream, appended by
 * `applyAgentRunEvent` in event order (replay-safe via the record's
 * `lastSequence` guard) and bounded to {@link MAX_STEP_LOG}. Decision
 * markers (`clarification_answered`, `approval_decided`) carry only ids;
 * the stream joins them with the control ROWS for display (rule R1 —
 * rows are the truth for content, the log only orders the story).
 */
export type AgentStepEntry = {
  /** Event sequence — stable identity and ordering across replays. */
  seq: number
  /** Event timestamp (seconds). */
  at: number
  kind:
    | 'phase'
    | 'activity'
    | 'plan'
    | 'task_started'
    | 'task_finished'
    | 'task_failed'
    | 'clarification_answered'
    | 'approval_decided'
    | 'narration'
  phase?: string
  taskId?: string
  activityKind?: string
  activityOperation?: import('./activityPresentation').AgentOperation
  activityOperationCode?: string
  activityKey?: string
  activityCount?: number
  detail?: string
  label?: string
  status?: string
  fallback?: boolean
  current?: number
  total?: number
  metrics?: Record<string, unknown>
  purpose?: string
  attempt?: number
  version?: number
  autoApproved?: boolean
  clarificationId?: string
  approvalId?: string
  error?: string
  /** Narration prose (B-M3). */
  text?: string
  /** Stable narration slot id (e.g. `n-synthesis`, `n-section-0`). The
   * backend re-emits the SAME id on node re-execution (critic replan
   * loop) with a fresh sequence — the reducer upserts on this id so the
   * line updates in place instead of multiplying. */
  narrationId?: string
}

/** Step-log bound; the tail is the recent story, the plan tab the map. */
export const MAX_STEP_LOG = 400

/**
 * Non-terminal run statuses: the run still occupies its session — the
 * composer shows stop, pills/canvas stay live, the rail marks activity.
 * `waiting_for_children` is a NORMAL mid-execution state (the parent is
 * parked while its child research runs work), never idleness. THE one
 * predicate for every "is something going on?" check — surfaces must
 * not hand-roll their own status lists (they drifted apart once).
 */
export function isActiveAgentRun(status: ResearchRunStatus): boolean {
  return (
    status === 'running'
    || status === 'queued'
    || status === 'waiting_for_approval'
    || status === 'waiting_for_input'
    || status === 'waiting_for_children'
  )
}

/**
 * Parked on a HUMAN decision (approval or answer): surfaces switch from
 * the working pulse to the calm warning breathe, and the gate tray has
 * something to show. A children-wait is deliberately NOT a gate — the
 * children are doing the work, so "working" = active && !gate.
 */
export function isGateAgentRun(status: ResearchRunStatus): boolean {
  return status === 'waiting_for_approval' || status === 'waiting_for_input'
}

/** The single Agent-UI permission gate. A view share is strictly read-only;
 * edit shares and owned/unscoped/local runs keep the existing status gates. */
export function canEditAgentRun(
  run: Pick<AgentRunRecord, 'access'> | null | undefined,
): boolean {
  return Boolean(
    run
    && (run.access?.mode !== 'shared' || run.access.permission === 'edit'),
  )
}

export type AgentRunRecord = {
  runId: string
  sessionId?: string
  kind: 'agent' | 'agent_child' | 'standard'
  /** Durable run-tree links used to retain loaded children while their root
   * remains visible and prune the whole tree when the root disappears. */
  parentRunId?: string
  rootRunId?: string
  question: string
  autonomy?: string
  /** Thoroughness: 'deep' shows the badge; unset = normal. */
  depth?: string
  /** Selected Stufe: schnell|gruendlich|tief; unset = legacy depth
   * semantics. Drives the gate's Suchtiefe options and add-task
   * defaults (the server validator stays the enforcement). */
  agentTier?: string
  /** Latest ANSWER-node model resolution from the live event stream
   * (R5-light). Live-only by design: agent runs are not
   * project-persisted, so after a reload the chip is honestly absent
   * instead of guessed. */
  modelResolution?: {
    model: string
    effort: string
    tier: string
    modelSource: string
  }
  status: ResearchRunStatus
  /** Backend phase string (intake…critic|done|cancelled). */
  phase: string
  /** Last station the pulse track reached (never moves backwards). */
  station: AgentPhaseStation
  access?: ResearchRunAccess
  queuePosition?: number | null
  createdAt: string
  startedAt?: string
  finishedAt?: string
  elapsedSeconds?: number
  timing?: ResearchRunSummary['timing']
  error?: string
  snapshot?: ResearchRunSnapshot
  /** Highest event sequence applied (drives `?after=` replay filtering). */
  lastSequence: number
  // Control rows (fetched; `*Stale` flags are the SSE refetch signals).
  plan?: AgentPlanRecord
  planStale: boolean
  approvals: AgentApprovalRecord[]
  approvalsStale: boolean
  clarifications: AgentClarificationRecord[]
  clarificationsStale: boolean
  artifactOrder: string[]
  artifacts: Record<string, AgentArtifactRecord>
  artifactsStale: boolean
  // Live progress.
  taskStates: Record<string, AgentTaskLiveState>
  children: Record<string, AgentChildProgressRecord>
  activity?: AgentActivityRecord
  /** Ordered transcript lines for the step stream (bounded). */
  stepLog: AgentStepEntry[]
  /** Auto-approved replan note (plan.revised{auto_approved}). */
  lastAutoApprovedVersion?: number
  /** Proposed editor patch (M7): id from the patch.proposed event, the
   * detail fetched row-first like every other control surface. */
  patchId?: string
  patch?: AgentPatchRecord
  patchStale: boolean
}

export type AgentSessionRecord = {
  id: string
  title: string
  groupId: string | null
  createdAt: string
  updatedAt: string
  /** Run ids of this session's turns, oldest first. */
  runIds: string[]
  /** Persistent source availability for runs started in this session. */
  sourcePolicy: AgentSourcePolicy
  /** The model picked in THIS session, kept so a reload returns to it rather
   * than to the account preference. Rides `items_json` beside the source
   * policy — no schema change. Absent means nothing was picked here and the
   * account preference seeds the composer. */
  modelSelection?: AgentSessionModelSelection
  /** Client-only: true once a server response for this session carried
   * `items_json`. The LIST endpoint is deliberately metadata-only, so before
   * the detail fetch lands, "no pick stored" and "pick not loaded yet" are
   * indistinguishable — and preference seeding must wait. This flag is what
   * it waits for. Never serialized. */
  metadataHydrated?: boolean
  /** `false` only for recipient-side views derived from shared runs. Such
   * sessions exist solely to render the Agent Desk and must never cross the
   * agent-session persistence API. Absent means persistable for backwards
   * compatibility with existing local state. */
  persistable?: boolean
  /** Durable server-owned deletion lifecycle. The row remains visible until
   * the terminal receipt removes it. */
  deletion?: import('@/features/project/sessionDeletion').SessionDeletionState
}

export type AgentSessionGroupRecord = {
  id: string
  title: string
  createdAt: string
  updatedAt: string
}

/**
 * History timestamp of a session row: the latest turn's `createdAt`,
 * falling back to `updatedAt`. The session record only bumps on rename /
 * source-policy edits (run membership is server-derived and unstamped),
 * so `updatedAt` alone would show a stale age for active conversations.
 */
export function agentSessionHistoryTimeIso(
  session: AgentSessionRecord,
  runs: Record<string, AgentRunRecord>,
): string {
  let latest = ''
  for (const runId of session.runIds) {
    const createdAt = runs[runId]?.createdAt ?? ''
    if (createdAt > latest) latest = createdAt
  }
  return latest || session.updatedAt
}

/**
 * The session to re-open after server hydration: the persisted intent
 * when it still exists, else the most recent session, else none. Pure —
 * the workspace effect dispatches the actual selection exactly once.
 */
export function restoredAgentSessionId(
  persistedId: string | null,
  sessionOrder: readonly string[],
  sessions: Record<string, AgentSessionRecord>,
  runs: Record<string, AgentRunRecord>,
): string | null {
  if (persistedId && sessions[persistedId] && !sessions[persistedId].deletion) {
    return persistedId
  }
  // Fallback by CONVERSATION recency (latest turn, not `updatedAt` —
  // which only bumps on rename/source-policy edits) and never by list
  // position: hydration appends in server list order, which is not a
  // guaranteed newest-first contract.
  let latestId: string | null = null
  let latestTime = ''
  for (const id of sessionOrder) {
    const session = sessions[id]
    if (!session || session.deletion) continue
    const time = agentSessionHistoryTimeIso(session, runs)
    if (!latestId || time > latestTime) {
      latestId = session.id
      latestTime = time
    }
  }
  return latestId
}

/**
 * What the desk's center column shows. Pure so the anti-flash contract is
 * pinnable offline: once the sessions are KNOWN (first listing settled for
 * this hydration identity — a view switch does not reset that), an empty
 * desk is a real welcome state and never the loading skeleton. The skeleton
 * remains for the genuine unknowns: the first listing of a fresh identity,
 * and a selected session whose runs are still paging in.
 */
export function agentCenterScreen({
  hasRuns,
  hasSelectedSession,
  runsHydrated,
  serverEnabled,
  sessionsKnown,
}: {
  hasRuns: boolean
  hasSelectedSession: boolean
  runsHydrated: boolean
  serverEnabled: boolean
  sessionsKnown: boolean
}): 'skeleton' | 'transcript' | 'welcome' {
  if (hasRuns) return 'transcript'
  if (!serverEnabled) return 'welcome'
  if (!sessionsKnown || (hasSelectedSession && !runsHydrated)) return 'skeleton'
  return 'welcome'
}

// --- Wire -> record converters (pure) ---------------------------------------

export function agentPlanFromWire(wire: AgentPlanWire): AgentPlanRecord {
  return {
    planId: wire.plan_id,
    version: wire.version,
    status: wire.status,
    createdBy: wire.created_by,
    reason: wire.reason,
    createdAt: wire.created_at,
    summaryMarkdown: wire.summary_markdown,
    assumptions: [...wire.assumptions],
    successCriteria: [...wire.success_criteria],
    tasks: wire.tasks.map(agentPlanTaskFromWire),
    versions: wire.versions.map((version) => ({
      planId: version.plan_id,
      version: version.version,
      status: version.status,
      createdBy: version.created_by,
      reason: version.reason,
      createdAt: version.created_at,
    })),
  }
}

export function agentPlanTaskFromWire(
  wire: AgentPlanTaskWire,
): AgentPlanTaskRecord {
  return {
    taskId: wire.task_id,
    ordinal: wire.ordinal,
    title: wire.title,
    toolKind: wire.tool_kind,
    objective: wire.objective,
    queries: [...wire.queries],
    gapIds: [...wire.gap_ids],
    dependsOn: [...wire.depends_on],
    budget: { ...wire.budget },
    params: { ...wire.params },
    expectedOutput: wire.expected_output,
    isFalsification: wire.is_falsification,
    status: wire.status,
    childRunId: wire.child_run_id,
    resultSummary: wire.result_summary,
  }
}

export function agentApprovalFromWire(
  wire: AgentApprovalWire,
): AgentApprovalRecord {
  return {
    approvalId: wire.approval_id,
    kind: wire.kind,
    status: wire.status,
    subjectType: wire.subject_type,
    subjectId: wire.subject_id,
    payload: { ...wire.payload },
    decision: wire.decision,
    note: wire.note,
    createdAt: wire.created_at,
    decidedAt: wire.decided_at,
  }
}

/**
 * Reconcile an approvals write into the existing rows. Approval decisions
 * are immutable server-side (a new gate is a new approval_id), so a row
 * that is already decided locally must never regress to ``pending`` — the
 * decided-event refetch can race the decision commit and observe a stale
 * pending snapshot, which with a blind replace re-opened the gate tray.
 * Incoming rows otherwise win; unknown ids append in arrival order.
 */
export function mergeAgentRunApprovals(
  existing: AgentApprovalRecord[],
  incoming: AgentApprovalRecord[],
): AgentApprovalRecord[] {
  const byId = new Map(existing.map((row) => [row.approvalId, row]))
  for (const row of incoming) {
    const current = byId.get(row.approvalId)
    if (current && current.status !== 'pending' && row.status === 'pending') {
      continue
    }
    byId.set(row.approvalId, row)
  }
  return [...byId.values()]
}

/**
 * Reconcile a clarifications write into the existing rows — the exact
 * sibling of :func:`mergeAgentRunApprovals`: an answered round must
 * never regress to ``pending`` when the answered-event refetch races
 * the answer commit (a regressed row re-opens the composer gate tray).
 */
export function mergeAgentRunClarifications(
  existing: AgentClarificationRecord[],
  incoming: AgentClarificationRecord[],
): AgentClarificationRecord[] {
  const byId = new Map(existing.map((row) => [row.clarificationId, row]))
  for (const row of incoming) {
    const current = byId.get(row.clarificationId)
    if (current && current.status !== 'pending' && row.status === 'pending') {
      continue
    }
    byId.set(row.clarificationId, row)
  }
  return [...byId.values()]
}


export function agentClarificationFromWire(
  wire: AgentClarificationWire,
): AgentClarificationRecord {
  return {
    clarificationId: wire.clarification_id,
    question: wire.question,
    options: wire.options.map(clarificationOptionFromWire),
    questions: (wire.questions ?? []).map((question) => ({
      id: question.id,
      prompt: question.prompt,
      options: question.options.map(clarificationOptionFromWire),
      multiSelect: question.multi_select,
    })),
    answers: Object.fromEntries(
      Object.entries(wire.answers ?? {}).map(([questionId, entry]) => [
        questionId,
        { optionIds: [...entry.option_ids], text: entry.text },
      ]),
    ),
    defaultAssumption: wire.default_assumption,
    status: wire.status,
    answer: wire.answer,
    optionId: wire.option_id,
    createdAt: wire.created_at,
    answeredAt: wire.answered_at,
  }
}

function clarificationOptionFromWire(option: {
  id: string
  label: string
  description?: string
}): AgentClarificationOption {
  return {
    id: option.id,
    label: option.label,
    description: option.description ?? '',
  }
}

export function agentArtifactFromWire(
  wire: AgentArtifactMetaWire | AgentArtifactDetailWire,
): AgentArtifactRecord {
  const record: AgentArtifactRecord = {
    artifactId: wire.artifact_id,
    kind: wire.kind,
    title: wire.title,
    status: wire.status,
    revision: wire.revision,
    updatedBy: wire.updated_by,
    refsCount: wire.refs_count,
    createdAt: wire.created_at,
    updatedAt: wire.updated_at,
  }
  if ('content_markdown' in wire) {
    record.contentMarkdown = wire.content_markdown
    record.payload = wire.payload ? { ...wire.payload } : {}
    record.refs = wire.refs.map((ref) => ({ ...ref }))
    record.revisions = wire.revisions.map((revision) => ({
      revision: revision.revision,
      createdBy: revision.created_by,
      createdAt: revision.created_at,
    }))
  }
  return record
}

/** Whether a run summary belongs on the Agent Desk (root agent runs only).
 * Both engines count — the FE twin of the backend's AGENT_MODE_IDS. */
export function isAgentRunSummary(summary: ResearchRunSummary): boolean {
  return (
    (summary.mode === 'workspace_agent' || summary.mode === 'agent_kernel')
    && summary.kind !== 'agent_child'
  )
}

/**
 * Session identity for one agent summary. Shared runs must not materialize the
 * owner's `session_id` in the recipient's syncable session namespace. A view
 * session is therefore derived from the server-generated root run id; children
 * resolve to the same id and the whole tree prunes together.
 */
export function agentSessionIdFromSummary(
  summary: ResearchRunSummary,
): string | undefined {
  if (summary.access?.mode === 'shared') {
    return `shared-agent-view:${summary.root_run_id || summary.run_id}`
  }
  return summary.session_id || undefined
}

export function agentRunFromSummary(
  summary: ResearchRunSummary,
): AgentRunRecord {
  const phase = phaseFromSnapshot(summary.snapshot)
  return {
    runId: summary.run_id,
    // '' and null both mean "sessionless" on the wire — normalize to
    // undefined so merges can never stomp a known session with emptiness.
    sessionId: agentSessionIdFromSummary(summary),
    kind: summary.kind ?? 'standard',
    parentRunId: summary.parent_run_id,
    rootRunId: summary.root_run_id,
    question: summary.question,
    autonomy: autonomyFromOverrides(summary.agent_overrides),
    depth: depthFromOverrides(summary.agent_overrides),
    agentTier: tierFromOverrides(summary.agent_overrides),
    status: summary.status,
    phase,
    station: agentPhaseStation(phase) ?? 'intake',
    access: summary.access,
    queuePosition: summary.queue_position,
    createdAt: isoFromSeconds(summary.created_at) ?? new Date().toISOString(),
    startedAt: isoFromSeconds(summary.started_at),
    finishedAt: isoFromSeconds(summary.finished_at),
    elapsedSeconds: summary.elapsed_seconds ?? undefined,
    timing: summary.timing,
    error: summary.error?.message,
    snapshot: summary.snapshot,
    lastSequence: 0,
    planStale: true,
    approvals: [],
    approvalsStale: true,
    clarifications: [],
    clarificationsStale: true,
    artifactOrder: [],
    artifacts: {},
    artifactsStale: true,
    taskStates: {},
    children: {},
    stepLog: [],
    patchStale: false,
  }
}

/** Merge a fresh summary into an existing record, keeping fetched rows. */
export function mergeAgentRunSummary(
  current: AgentRunRecord | undefined,
  summary: ResearchRunSummary,
): AgentRunRecord {
  const next = agentRunFromSummary(summary)
  if (!current) return next
  const phase = next.phase || current.phase
  return {
    ...current,
    sessionId: next.sessionId || current.sessionId,
    parentRunId: next.parentRunId ?? current.parentRunId,
    rootRunId: next.rootRunId ?? current.rootRunId,
    status: next.status,
    phase,
    station: agentPhaseStation(phase) ?? current.station,
    queuePosition: next.queuePosition,
    startedAt: next.startedAt ?? current.startedAt,
    finishedAt: next.finishedAt ?? current.finishedAt,
    elapsedSeconds: next.elapsedSeconds ?? current.elapsedSeconds,
    timing: next.timing ?? current.timing,
    error: next.error ?? current.error,
    snapshot: next.snapshot ?? current.snapshot,
    access: next.access ?? current.access,
  }
}

function phaseFromSnapshot(snapshot: ResearchRunSnapshot | undefined): string {
  const raw = (snapshot as Record<string, unknown> | undefined)?.phase
  return typeof raw === 'string' && raw ? raw : 'intake'
}

function autonomyFromOverrides(
  overrides: Record<string, unknown>,
): string | undefined {
  const raw = overrides.autonomy
  return typeof raw === 'string' && raw ? raw : undefined
}

function depthFromOverrides(
  overrides: Record<string, unknown>,
): string | undefined {
  return overrides.depth === 'deep' ? 'deep' : undefined
}

function tierFromOverrides(
  overrides: Record<string, unknown>,
): string | undefined {
  const raw = overrides.agent_tier
  return typeof raw === 'string' && raw ? raw : undefined
}

function isoFromSeconds(value: number | null | undefined): string | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined
  return new Date(value * 1000).toISOString()
}
