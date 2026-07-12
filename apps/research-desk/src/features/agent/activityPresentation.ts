import type { TranslationDictionary } from '@/i18n/translations'
import type { AgentActivityRecord, AgentStepEntry } from './model'

export type AgentOperation =
  | 'knowledge_collections'
  | 'knowledge_search'
  | 'web_instant'
  | 'discovery_summary'

export type AgentActivityIconKind = 'generic' | 'knowledge' | 'web'

export function agentActivityIconKind(
  operation: AgentOperation | undefined,
): AgentActivityIconKind {
  if (operation === 'web_instant') return 'web'
  if (operation === 'knowledge_collections' || operation === 'knowledge_search') {
    return 'knowledge'
  }
  return 'generic'
}

/** Normalize the stable activity vocabulary and the capability ids emitted by
 * older servers into one presentation contract. Unknown values deliberately
 * remain undefined so their raw label stays visible instead of being guessed. */
export function normalizeAgentOperation(value: unknown): AgentOperation | undefined {
  if (typeof value !== 'string') return undefined
  switch (value) {
    case 'knowledge_collections':
    case 'knowledge.collections':
    case 'knowledge.collections.list':
      return 'knowledge_collections'
    case 'knowledge_search':
    case 'knowledge.search':
      return 'knowledge_search'
    case 'web_instant':
    case 'web.search.instant':
      return 'web_instant'
    case 'discovery_summary':
      return 'discovery_summary'
    default:
      return undefined
  }
}

export function activityDisplayText(
  activity: Pick<
    AgentActivityRecord,
    'count' | 'current' | 'detail' | 'kind' | 'label' | 'metrics' | 'operation' | 'operationCode' | 'purpose' | 'status' | 'total'
  >,
  t: TranslationDictionary,
): string {
  const operation = activity.operation
    ?? normalizeAgentOperation(activity.detail)
    ?? normalizeAgentOperation(activity.label)
  const humanDetail = cleanLegacyDetail(activity.detail || activity.purpose || '')
  const title = operation
    ? operationLabel(operation, t)
    : activity.label
      || humanDetail
      || activity.operationCode
      || fallbackKindLabel(activity.kind, t)
  const detail = operation
    ? cleanOperationDetail(activity.detail || activity.purpose || '', operation)
    : activity.label && humanDetail !== activity.label
      ? humanDetail
      : ''
  const progress = activityProgress(activity, t)
  const metrics = activityMetrics(activity.metrics, t)
  return [title, detail, progress, metrics].filter(Boolean).join(' · ')
}

export function activityStepDisplayText(
  entry: AgentStepEntry,
  t: TranslationDictionary,
): string {
  return activityDisplayText(
    {
      count: entry.activityCount,
      current: entry.current,
      detail: entry.detail ?? '',
      kind: entry.activityKind ?? 'working',
      label: entry.label,
      operation: entry.activityOperation,
      operationCode: entry.activityOperationCode,
      metrics: entry.metrics,
      purpose: entry.purpose,
      status: entry.status,
      total: entry.total,
    },
    t,
  )
}

/** Return the one terminal activity error already rendered by task detail.
 * Earlier matching failures remain in the operation history as retry evidence. */
export function terminalActivityErrorIndex(
  history: readonly Pick<AgentActivityRecord, 'error' | 'status'>[],
  terminalError: string | undefined,
): number {
  const normalized = terminalError?.trim()
  if (!normalized) return -1
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const activity = history[index]
    if (
      activity?.status === 'failed'
      && activity.error?.trim() === normalized
    ) return index
  }
  return -1
}

export function discoveryProbeDisplay(
  probe: Record<string, unknown>,
  t: TranslationDictionary,
): { detail: string; title: string } {
  const rawKind = typeof probe.kind === 'string' ? probe.kind : ''
  const operation = normalizeAgentOperation(probe.operation) ?? normalizeAgentOperation(rawKind)
  const detail = typeof probe.query === 'string' ? probe.query.trim() : ''
  return {
    detail,
    title: operation ? operationLabel(operation, t) : rawKind || t.agent.activity.execution,
  }
}

function operationLabel(
  operation: AgentOperation,
  t: TranslationDictionary,
): string {
  switch (operation) {
    case 'knowledge_collections':
      return t.agent.activityOperations.knowledgeCollections
    case 'knowledge_search':
      return t.agent.activityOperations.knowledgeSearch
    case 'web_instant':
      return t.agent.activityOperations.webInstant
    case 'discovery_summary':
      return t.agent.activityOperations.discoverySummary
  }
}

function cleanOperationDetail(
  detail: string,
  operation: AgentOperation | undefined,
): string {
  const cleaned = cleanLegacyDetail(detail)
  if (!cleaned) return ''
  return normalizeAgentOperation(cleaned) === operation ? '' : cleaned
}

function cleanLegacyDetail(detail: string): string {
  return normalizeAgentOperation(detail) ? '' : detail.trim()
}

function activityProgress(
  activity: Pick<AgentActivityRecord, 'count' | 'current' | 'status' | 'total'>,
  t: TranslationDictionary,
): string {
  if (
    typeof activity.current === 'number'
    && typeof activity.total === 'number'
    && activity.total > 0
  ) {
    return t.agent.activityOperations.progress
      .replace('{current}', String(activity.current))
      .replace('{total}', String(activity.total))
  }
  if ((activity.count ?? 0) > 1) {
    return t.agent.activityOperations.operations.replace(
      '{count}',
      String(activity.count),
    )
  }
  if (activity.status === 'failed') return t.status.failed
  return ''
}

function fallbackKindLabel(kind: string, t: TranslationDictionary): string {
  if (kind === 'searching') return t.agent.activity.searching
  return t.agent.activity.execution
}

function activityMetrics(
  metrics: Record<string, unknown> | undefined,
  t: TranslationDictionary,
): string {
  if (!metrics) return ''
  const counts: string[] = []
  const results = finiteMetric(metrics.result_count)
  const references = finiteMetric(metrics.reference_count)
  const claims = finiteMetric(metrics.claim_count)
  if (results !== undefined) {
    counts.push(t.agent.activityOperations.results.replace('{count}', String(results)))
  }
  if (references !== undefined) {
    counts.push(t.agent.activityOperations.references.replace('{count}', String(references)))
  }
  if (claims !== undefined) {
    counts.push(t.agent.activityOperations.claims.replace('{count}', String(claims)))
  }
  return counts.join(' · ')
}

function finiteMetric(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
