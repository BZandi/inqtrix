/**
 * Pure SSE-event application for agent runs (the `applyKnowledgeRunEvent`
 * pattern): `(record, event) -> record`, no i18n, no clocks, no fetches —
 * fully unit-testable. Control-row payloads are NOT copied from events
 * (rows are the truth, rule R1); events only flip the `*Stale` flags the
 * run hook turns into refetches, and carry the live task/phase state.
 */

import type {
  AgentRunRecord,
  AgentStepEntry,
  AgentTaskLiveState,
} from './model'
import {
  agentPhaseStation,
  MAX_STEP_LOG,
  TERMINAL_AGENT_TASK_STATUSES,
} from './model'
import type {
  ResearchRunEvent,
  ResearchRunSnapshot,
} from '@/features/researchRuns/types'
import { asFiniteNumber, asString } from '@/lib/coerce'
import { isoFromUnixSeconds, unixSecondsFromIso } from '@/lib/time'
import { normalizeAgentOperation } from './activityPresentation'
import { mergeResearchSnapshot } from './taskProgress'

/** Apply the durable cancellation acknowledgement immediately.
 *
 * The HTTP response is authoritative even when SSE is degraded. Keeping this
 * projection beside the event reducer prevents a second, subtly different
 * task-state transition in the control hook.
 */
export function acknowledgeAgentTaskCancellation(
  record: AgentRunRecord,
  taskId: string,
  status: 'cancel_requested' | 'cancelled',
  childRunId: string | null,
): AgentRunRecord {
  const previous = record.taskStates[taskId]
  return {
    ...record,
    planStale: true,
    taskStates: {
      ...record.taskStates,
      [taskId]: taskCancellationState(previous, status, childRunId),
    },
  }
}


function taskCancellationState(
  previous: AgentTaskLiveState | undefined,
  status: 'cancel_requested' | 'cancelled',
  childRunId: string | null,
  finishedAt?: number,
): AgentTaskLiveState {
  const nextStatus = monotonicTaskStatus(previous?.status, status)
  if (previous && nextStatus === previous.status) return previous
  return {
    ...(previous ?? { attempt: 1, status: 'running' }),
    status: nextStatus,
    childRunId: childRunId ?? previous?.childRunId,
    ...(status === 'cancelled' && finishedAt !== undefined
      ? { finishedAt }
      : {}),
  }
}

function monotonicTaskStatus(
  current: AgentTaskLiveState['status'] | undefined,
  incoming: AgentTaskLiveState['status'],
  currentAttempt = 1,
  incomingAttempt = currentAttempt,
): AgentTaskLiveState['status'] {
  if (!current) return incoming
  if (
    current === 'failed'
    && incoming === 'running'
    && incomingAttempt > currentAttempt
  ) return incoming
  if (TERMINAL_AGENT_TASK_STATUSES.has(current)) return current
  if (current === 'cancel_requested' && incoming !== 'cancelled') {
    return current
  }
  return incoming
}

export function applyAgentRunEvent(
  record: AgentRunRecord,
  event: ResearchRunEvent,
): AgentRunRecord {
  // Replayed/duplicate frames must be no-ops (reconnects replay history).
  if (event.sequence <= record.lastSequence) return record
  const next: AgentRunRecord = {
    ...record,
    lastSequence: event.sequence,
  }
  const data = event.data ?? {}

  switch (event.type) {
    case 'inqtrix.run.queued':
      next.status = 'queued'
      return next
    case 'inqtrix.run.started':
      next.status = 'running'
      next.startedAt ??= isoFromUnixSeconds(event.created_at)
      return next
    case 'inqtrix.run.waiting': {
      const status = stringField(data, 'status')
      next.status =
        status === 'waiting_for_input'
          ? 'waiting_for_input'
          : status === 'waiting_for_children'
            ? 'waiting_for_children'
            : 'waiting_for_approval'
      return next
    }
    case 'inqtrix.run.completed':
      next.status = 'completed'
      next.artifactsStale = true
      applyTerminalRunTime(next, event)
      return next
    case 'inqtrix.run.failed':
      next.status = 'failed'
      next.error = stringField(data, 'message') || record.error
      applyTerminalRunTime(next, event)
      return next
    case 'inqtrix.run.cancelled':
      next.status = 'cancelled'
      applyTerminalRunTime(next, event)
      return next
    case 'inqtrix.run.snapshot': {
      const snapshot = data.snapshot as ResearchRunSnapshot | undefined
      if (snapshot) next.snapshot = snapshot
      return next
    }

    case 'inqtrix.agent.phase.changed': {
      const phase = stringField(data, 'phase')
      if (phase) {
        next.phase = phase
        const station = agentPhaseStation(phase)
        if (station) next.station = station
        appendStep(next, { kind: 'phase', phase })
      }
      return next
    }

    case 'inqtrix.agent.plan.proposed':
    case 'inqtrix.agent.plan.revised': {
      next.planStale = true
      const version = numberField(data, 'version')
      // A revised plan can arrive auto-approved (balanced replans) — the
      // timeline shows that as a note instead of an approval card.
      const autoApproved =
        event.type === 'inqtrix.agent.plan.revised'
        && Boolean(data.auto_approved)
      if (autoApproved && version !== undefined) {
        next.lastAutoApprovedVersion = version
      }
      appendStep(next, { autoApproved, kind: 'plan', version })
      return next
    }

    case 'inqtrix.agent.approval.requested':
      next.approvalsStale = true
      return next
    case 'inqtrix.agent.approval.decided': {
      next.approvalsStale = true
      const approvalId = stringField(data, 'approval_id')
      const approvalKind =
        stringField(data, 'kind')
        || stringField(data, 'approval_kind')
        || record.approvals.find(
          (approval) => approval.approvalId === approvalId,
        )?.kind
      if (approvalKind === 'plan' || approvalKind === 'replan') {
        next.planStale = true
      }
      // Decision marker for the transcript; the rows carry the content.
      appendStep(next, {
        approvalId: approvalId || undefined,
        kind: 'approval_decided',
        detail: stringField(data, 'status') || undefined,
      })
      return next
    }

    case 'inqtrix.agent.clarification.requested':
      next.clarificationsStale = true
      return next
    case 'inqtrix.node.model_resolution': {
      // R5-light: only the ANSWER-shaped nodes drive the chip — the
      // resolution of intake/critic/... is provenance noise here.
      const node = stringField(data, 'node')
      if (
        node !== 'agent_answer'
        && node !== 'agent_answer_light'
        && node !== 'agent_synthesis'
        && node !== 'agent_kernel'
      ) {
        return next
      }
      next.modelResolution = {
        model: stringField(data, 'model'),
        effort: stringField(data, 'effort'),
        tier: stringField(data, 'tier'),
        modelSource: stringField(data, 'model_source'),
      }
      return next
    }
    case 'inqtrix.agent.clarification.answered': {
      next.clarificationsStale = true
      appendStep(next, {
        clarificationId:
          stringField(data, 'clarification_id') || undefined,
        kind: 'clarification_answered',
      })
      return next
    }

    case 'inqtrix.agent.task.started': {
      const taskId = stringField(data, 'task_id')
      if (!taskId) return next
      const previous = record.taskStates[taskId]
      const incomingAttempt = numberField(data, 'attempt') ?? 1
      const status = monotonicTaskStatus(
        previous?.status,
        'running',
        previous?.attempt ?? 1,
        incomingAttempt,
      )
      const state: AgentTaskLiveState = {
        ...(previous ?? {}),
        status,
        attempt: Math.max(
          previous?.attempt ?? 1,
          incomingAttempt,
        ),
        childRunId:
          stringField(data, 'child_run_id')
          || previous?.childRunId,
        activity: previous?.activity,
        activityHistory: previous?.activityHistory,
        fallback: previous?.fallback,
        metrics: previous?.metrics,
        resultSummary: previous?.resultSummary,
        startedAt: previous?.startedAt ?? event.created_at,
        ...(status === 'running' ? { finishedAt: undefined } : {}),
      }
      next.taskStates = { ...record.taskStates, [taskId]: state }
      appendStep(next, { kind: 'task_started', taskId })
      return next
    }
    case 'inqtrix.agent.task.cancel_requested': {
      const taskId = stringField(data, 'task_id')
      if (!taskId) return next
      const status = stringField(data, 'status')
      const previous = record.taskStates[taskId]
      next.taskStates = {
        ...record.taskStates,
        [taskId]: taskCancellationState(
          previous,
          status === 'cancelled' ? 'cancelled' : 'cancel_requested',
          stringField(data, 'child_run_id') || null,
          event.created_at,
        ),
      }
      next.planStale = true
      return next
    }
    case 'inqtrix.agent.task.finished':
    case 'inqtrix.agent.task.failed': {
      const taskId = stringField(data, 'task_id')
      if (!taskId) return next
      const outcome = stringField(data, 'status')
      const parsedError = errorInfo(data.error)
      const failed =
        event.type === 'inqtrix.agent.task.failed' || outcome === 'failed'
      const insufficient = outcome === 'insufficient_evidence'
      const cancelled = outcome === 'cancelled'
      const previousTask = record.taskStates[taskId]
      const incomingStatus: AgentTaskLiveState['status'] = cancelled
          ? 'cancelled'
          : insufficient
          ? 'insufficient_evidence'
          : failed
            ? 'failed'
            : 'completed'
      const status = monotonicTaskStatus(
        previousTask?.status,
        incomingStatus,
      )
      const lifecycleAccepted = status === incomingStatus
      const state: AgentTaskLiveState = {
        status,
        outcome: lifecycleAccepted
          ? outcome || (failed ? 'failed' : 'completed')
          : previousTask?.outcome,
        attempt: previousTask?.attempt ?? 1,
        childRunId:
          stringField(data, 'child_run_id') ||
          previousTask?.childRunId,
        error: lifecycleAccepted ? parsedError.message : previousTask?.error,
        errorCode: lifecycleAccepted ? parsedError.code : previousTask?.errorCode,
        activity: previousTask?.activity,
        activityHistory: previousTask?.activityHistory,
        metrics: recordField(data, 'metrics') ?? previousTask?.metrics,
        resultSummary:
          lifecycleAccepted
            ? stringField(data, 'result_summary') || previousTask?.resultSummary
            : previousTask?.resultSummary,
        fallback: previousTask?.fallback,
        startedAt: previousTask?.startedAt,
        finishedAt: lifecycleAccepted
          ? previousTask?.finishedAt ?? event.created_at
          : previousTask?.finishedAt,
      }
      next.taskStates = { ...record.taskStates, [taskId]: state }
      if (lifecycleAccepted) {
        // Settle any still-open activity rows of the task: a retry
        // notice or hard-aborted invocation must not stay "running"
        // forever in the protocol once the task itself concluded.
        const settled = status === 'completed' ? 'completed' : 'failed'
        const openRows = record.stepLog.some(
          (item) =>
            item.kind === 'activity'
            && item.taskId === taskId
            && item.status !== 'completed'
            && item.status !== 'failed',
        )
        if (openRows) {
          next.stepLog = (
            next.stepLog === record.stepLog
              ? [...record.stepLog]
              : next.stepLog
          ).map((item) =>
            item.kind === 'activity'
            && item.taskId === taskId
            && item.status !== 'completed'
            && item.status !== 'failed'
              ? { ...item, status: settled }
              : item,
          )
        }
      }
      // Task outcomes update the plan rows (status/result_summary) too.
      next.planStale = true
      appendStep(next, {
        error: state.error,
        kind: failed || insufficient ? 'task_failed' : 'task_finished',
        taskId,
      })
      return next
    }

    case 'inqtrix.agent.child.progress': {
      const childRunId = stringField(data, 'child_run_id')
      const taskId = stringField(data, 'task_id')
      if (!childRunId) return next
      const previousChild = record.children[childRunId]
      const snapshot = mergeResearchSnapshot(
        previousChild?.snapshot,
        data.snapshot,
      )
      const runStatus = stringField(data, 'run_status') || previousChild?.runStatus
      const currentNode = stringField(data, 'current_node') || previousChild?.currentNode
      const message = stringField(data, 'message') || previousChild?.message
      const parsedError = errorInfo(data.error)
      const error = parsedError.message || previousChild?.error
      const errorCode = parsedError.code || previousChild?.errorCode
      const metrics = recordField(data, 'metrics') ?? previousChild?.metrics
      const attempt = numberField(data, 'attempt') ?? previousChild?.attempt
      const updatedAt = numberField(data, 'updated_at') ?? event.created_at
      next.children = {
        ...record.children,
        [childRunId]: {
          childRunId,
          taskId: taskId || previousChild?.taskId || '',
          ...(snapshot ? { snapshot } : {}),
          ...(runStatus ? { runStatus } : {}),
          ...(currentNode ? { currentNode } : {}),
          ...(message ? { message } : {}),
          ...(metrics ? { metrics } : {}),
          ...(attempt !== undefined ? { attempt } : {}),
          ...(error ? { error } : {}),
          ...(errorCode ? { errorCode } : {}),
          updatedAt,
        },
      }
      if (taskId) {
        const previousTask = record.taskStates[taskId]
        const projectedStatus = monotonicTaskStatus(
          previousTask?.status,
          childTaskStatus(runStatus, previousTask?.status),
          previousTask?.attempt ?? 1,
          attempt ?? previousTask?.attempt ?? 1,
        )
        const childTerminal =
          projectedStatus === 'completed'
          || projectedStatus === 'failed'
          || projectedStatus === 'insufficient_evidence'
        next.taskStates = {
          ...next.taskStates,
          [taskId]: {
            ...(previousTask ?? { status: 'running', attempt: 1 }),
            status: projectedStatus,
            childRunId,
            attempt:
              attempt
              ?? previousTask?.attempt
              ?? 1,
            ...(error ? { error } : {}),
            ...(errorCode ? { errorCode } : {}),
            startedAt: previousTask?.startedAt ?? event.created_at,
            ...(childTerminal
              ? { finishedAt: previousTask?.finishedAt ?? updatedAt }
              : { finishedAt: undefined }),
          },
        }
      }
      return next
    }

    case 'inqtrix.agent.artifact.created':
    case 'inqtrix.agent.artifact.updated':
    case 'inqtrix.agent.artifact.edit_conflict':
      // Signal only (rule R1): the cached row must NOT adopt the event's
      // revision — a bumped revision over the OLD body would defeat the
      // list-refetch staleness check and let a user edit of stale text
      // pass the server's expected_revision gate (the E13 overwrite).
      // edit_conflict rides the same refetch: the agent preserved the
      // user's text and appended its update, so the canvas reloads the
      // reconciled memo instead of the user's stale local copy.
      next.artifactsStale = true
      return next

    case 'inqtrix.agent.patch.proposed': {
      const patchId = stringField(data, 'patch_id')
      if (patchId) {
        next.patchId = patchId
        next.patchStale = true
      }
      next.artifactsStale = true
      return next
    }

    case 'inqtrix.agent.activity': {
      const rawProbe = stringField(data, 'probe')
      const operationCode = stringField(data, 'operation') || rawProbe
      const operation =
        normalizeAgentOperation(operationCode)
        ?? normalizeAgentOperation(rawProbe)
      const taskId = stringField(data, 'task_id') || undefined
      const status = stringField(data, 'status') || undefined
      const parsedError = errorInfo(data.error)
      const fallback = data.fallback === true || status === 'fallback'
      const metrics = recordField(data, 'metrics')
      const purpose = stringField(data, 'purpose') || undefined
      const attempt = numberField(data, 'attempt')
      const detail =
        stringField(data, 'detail')
        || stringField(data, 'query')
        || (operation ? '' : rawProbe)
      const previousTaskActivity = taskId
        ? record.taskStates[taskId]?.activity
        : undefined
      const effectiveDetail = detail || (
        status !== undefined
        && (
          (operation !== undefined
            && previousTaskActivity?.operation === operation)
          || previousTaskActivity?.operationCode === operationCode
        )
          ? previousTaskActivity.detail
          : ''
      )
      // The transcript is an append-only protocol. Contract per event
      // shape (P3):
      //   started (new invocation)   -> append one row
      //   started (same invocation)  -> upsert row (retry/progress text)
      //   completed/failed           -> settle THAT row in place
      //   task.finished/failed       -> settle any still-open rows of
      //                                 the task (see terminal branch)
      // Every INVOCATION (one query of a multi-query task) gets its own
      // row, so "search 6 of 6" never hides inside a counter on row 1.
      // Identity: activity id, else the bare query (stable across retry
      // notices whose detail text changes), else the detail.
      const invocationKey =
        stringField(data, 'activity_id')
        || stringField(data, 'operation_id')
        || stringField(data, 'query')
        || effectiveDetail
      const activityKey = operation
        ? `${record.phase}:${operation}:${taskId ?? ''}:${invocationKey}`
        : undefined
      const previousStep = activityKey
        ? record.stepLog.find((item) => item.activityKey === activityKey)
        : undefined
      const startsInvocation = status === undefined || status === 'started'
      const count = previousStep
        ? startsInvocation
          ? (previousStep.activityCount ?? 1) + 1
          : previousStep.activityCount ?? 1
        : 1
      next.activity = {
        activityId:
          stringField(data, 'activity_id')
          || stringField(data, 'operation_id')
          || undefined,
        kind: stringField(data, 'kind') || 'working',
        detail: effectiveDetail,
        label: stringField(data, 'label') || undefined,
        status,
        operation,
        operationCode: operationCode || undefined,
        current: numberField(data, 'current'),
        total: numberField(data, 'total'),
        count,
        taskId,
        purpose,
        metrics,
        attempt,
        error: parsedError.message,
        errorCode: parsedError.code,
        fallback,
        at: event.created_at,
      }
      if (taskId) {
        const previousTask = record.taskStates[taskId]
        const activityHistory = updateTaskActivityHistory(
          previousTask?.activityHistory,
          next.activity,
        )
        const activityTerminal = status === 'completed' || status === 'failed'
        const lifecycleLocked = previousTask !== undefined && (
          TERMINAL_AGENT_TASK_STATUSES.has(previousTask.status)
          || previousTask.status === 'cancel_requested'
        )
        next.taskStates = {
          ...next.taskStates,
          [taskId]: {
            ...(previousTask ?? { status: 'running', attempt: 1 }),
            activity: next.activity,
            activityHistory,
            ...(parsedError.message
              ? { error: parsedError.message }
              : {}),
            ...(parsedError.code ? { errorCode: parsedError.code } : {}),
            ...(fallback ? { fallback: true } : {}),
            startedAt: previousTask?.startedAt ?? event.created_at,
            ...(activityTerminal && !lifecycleLocked
              ? { finishedAt: event.created_at }
              : {}),
            ...(status === 'started' && !lifecycleLocked
              ? { finishedAt: undefined }
              : {}),
          },
        }
      }
      appendActivityStep(next, {
        activityKey,
        activityKind: next.activity.kind,
        activityOperation: operation,
        activityOperationCode: next.activity.operationCode,
        detail: next.activity.detail || undefined,
        kind: 'activity',
        label: next.activity.label,
        status: next.activity.status,
        fallback: next.activity.fallback,
        error: next.activity.error,
        current: next.activity.current,
        total: next.activity.total,
        metrics: next.activity.metrics,
        purpose: next.activity.purpose,
        attempt: next.activity.attempt,
        activityCount: count,
        taskId,
      })
      return next
    }

    case 'inqtrix.agent.narration': {
      const text = stringField(data, 'text')
      if (text) {
        appendStep(next, {
          detail: stringField(data, 'kind') || undefined,
          kind: 'narration',
          narrationId: stringField(data, 'narration_id') || undefined,
          phase: stringField(data, 'phase') || undefined,
          text,
        })
      }
      return next
    }

    default:
      return next
  }

  /** Bounded ordered append with narration UPSERT: a line carrying a
   * ``narrationId`` already in the log replaces that entry in place
   * (keeping its position and React key) rather than appending a
   * duplicate — the backend re-emits stable narration ids on node
   * re-execution (critic replan loop) with fresh sequences, which the
   * sequence guard does not suppress. seq/at ride the event envelope. */
  function appendStep(
    target: AgentRunRecord,
    entry: Omit<AgentStepEntry, 'at' | 'seq'>,
  ): void {
    const log =
      target.stepLog === record.stepLog
        ? [...record.stepLog]
        : target.stepLog
    if (entry.narrationId) {
      const existing = log.findIndex(
        (item) => item.narrationId === entry.narrationId,
      )
      if (existing !== -1) {
        // Update text/at in place; keep the original seq so the React
        // key is stable and the line does not jump to the bottom.
        log[existing] = {
          ...log[existing],
          ...entry,
          at: event.created_at,
          seq: log[existing].seq,
        }
        target.stepLog = log
        return
      }
    }
    log.push({ ...entry, at: event.created_at, seq: event.sequence })
    target.stepLog =
      log.length > MAX_STEP_LOG ? log.slice(log.length - MAX_STEP_LOG) : log
  }

  /** Consecutive probes/calls of the same semantic operation update one
   * transcript row. The plan retains every literal query; the timeline stays
   * a readable status story rather than four identical capability ids. */
  function appendActivityStep(
    target: AgentRunRecord,
    entry: Omit<AgentStepEntry, 'at' | 'seq'>,
  ): void {
    const operation = entry.activityOperation
    if (!operation) {
      appendStep(target, entry)
      return
    }
    // The caller passes the per-invocation key; the phase-scoped
    // fallback only serves legacy entries without one.
    const activityKey =
      entry.activityKey
      ?? `${record.phase}:${operation}:${entry.taskId ?? ''}`
    const log =
      target.stepLog === record.stepLog
        ? [...record.stepLog]
        : target.stepLog
    const existing = log.findIndex((item) => item.activityKey === activityKey)
    if (existing === -1) {
      appendStep(target, {
        ...entry,
        activityCount: entry.activityCount ?? 1,
        activityKey,
      })
      return
    }
    log[existing] = {
      ...log[existing],
      ...entry,
      detail: entry.detail ?? log[existing].detail,
      label: entry.label ?? log[existing].label,
      activityOperation:
        entry.activityOperation ?? log[existing].activityOperation,
      activityOperationCode:
        entry.activityOperationCode ?? log[existing].activityOperationCode,
      activityCount:
          entry.activityCount ?? log[existing].activityCount ?? 1,
      activityKey,
      at: event.created_at,
      seq: log[existing].seq,
    }
    target.stepLog = log
  }
}

/** Preserve the server summary when present; otherwise derive the terminal
 * stopwatch from the same event boundaries already used by live replay. */
function applyTerminalRunTime(
  record: AgentRunRecord,
  event: ResearchRunEvent,
): void {
  record.finishedAt ??= isoFromUnixSeconds(event.created_at)
  const wireElapsed = asFiniteNumber(event.data?.elapsed_seconds)
  if (wireElapsed !== undefined) {
    record.elapsedSeconds = Math.max(0, wireElapsed)
    return
  }
  if (record.elapsedSeconds !== undefined) return
  if (!record.startedAt) return
  record.elapsedSeconds = Math.max(
    0,
    event.created_at - unixSecondsFromIso(record.startedAt),
  )
}

/** ``(data, key)`` accessor with the '' fallback this module's call sites
 * expect; the coercion itself is the shared {@link asString}. */
function stringField(data: Record<string, unknown>, key: string): string {
  return asString(data[key]) ?? ''
}

/** ``(data, key)`` accessor over the shared {@link asFiniteNumber}. */
function numberField(
  data: Record<string, unknown>,
  key: string,
): number | undefined {
  return asFiniteNumber(data[key])
}

function recordField(
  data: Record<string, unknown>,
  key: string,
): Record<string, unknown> | undefined {
  const value = data[key]
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function errorInfo(value: unknown): { code?: string; message?: string } {
  if (typeof value === 'string' && value.trim()) {
    return { message: value.trim() }
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const error = value as Record<string, unknown>
  const message = asString(error.message)
  const code = asString(error.code) ?? asString(error.type)
  return {
    ...(message ? { message } : {}),
    ...(code ? { code } : {}),
  }
}

const MAX_TASK_ACTIVITY_HISTORY = 100

function updateTaskActivityHistory(
  current: AgentTaskLiveState['activityHistory'],
  activity: NonNullable<AgentTaskLiveState['activity']>,
): NonNullable<AgentTaskLiveState['activityHistory']> {
  const history = [...(current ?? [])]
  const updatesInvocation =
    activity.status !== undefined && activity.status !== 'started'
  let index = -1
  if (updatesInvocation) {
    if (activity.activityId) {
      index = history.findIndex(
        (item) => item.activityId === activity.activityId,
      )
    }
    if (index === -1) {
      for (let cursor = history.length - 1; cursor >= 0; cursor -= 1) {
        const item = history[cursor]
        if (
          item.operation === activity.operation
          && item.kind === activity.kind
          && item.status !== 'completed'
          && item.status !== 'failed'
          && item.status !== 'fallback'
        ) {
          index = cursor
          break
        }
      }
    }
  }
  if (index === -1) history.push(activity)
  else {
    history[index] = {
      ...history[index],
      ...activity,
      detail: activity.detail || history[index].detail,
      label: activity.label ?? history[index].label,
      operation: activity.operation ?? history[index].operation,
      operationCode: activity.operationCode ?? history[index].operationCode,
      purpose: activity.purpose ?? history[index].purpose,
      metrics: activity.metrics ?? history[index].metrics,
      attempt: activity.attempt ?? history[index].attempt,
    }
  }
  return history.length > MAX_TASK_ACTIVITY_HISTORY
    ? history.slice(history.length - MAX_TASK_ACTIVITY_HISTORY)
    : history
}

function childTaskStatus(
  runStatus: string | undefined,
  current: AgentTaskLiveState['status'] | undefined,
): AgentTaskLiveState['status'] {
  if (runStatus === 'cancelled' && current === 'cancel_requested') {
    return 'cancel_requested'
  }
  if (runStatus === 'failed' || runStatus === 'cancelled' || runStatus === 'expired') {
    return 'failed'
  }
  if (runStatus === 'completed') return 'completed'
  if (runStatus === 'insufficient_evidence') return 'insufficient_evidence'
  if (
    runStatus === 'queued'
    || runStatus === 'running'
    || runStatus === 'waiting_for_approval'
    || runStatus === 'waiting_for_children'
    || runStatus === 'waiting_for_input'
  ) return 'running'
  return current ?? 'running'
}
