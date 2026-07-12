import type {
  ResearchRunEvent,
  ResearchRunSnapshot,
} from '@/features/researchRuns/types'
import type { JobPhase } from '@/features/researchDesk/types'

export type ChildProgressMessage = {
  sequence: number
  severity: 'error' | 'info' | 'warning'
  text: string
}

export function researchNodePhase(node: string | undefined): JobPhase | null {
  if (node === 'classify') return 'analysis'
  if (node === 'plan') return 'planning'
  if (node === 'search') return 'search'
  if (node === 'evaluate') return 'evaluation'
  if (node === 'answer' || node === 'direct_llm') return 'answer'
  return null
}

export function completedResearchPhases(node: string | undefined): JobPhase[] {
  const phase = researchNodePhase(node)
  if (!phase) return []
  const order: JobPhase[] = [
    'analysis',
    'planning',
    'search',
    'evaluation',
    'answer',
  ]
  return order.slice(0, order.indexOf(phase))
}

/** Empty child-start snapshots carry identity only. They must never be
 * interpreted as a completed/answering research run. */
export function meaningfulResearchSnapshot(
  value: unknown,
): ResearchRunSnapshot | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const snapshot = value as ResearchRunSnapshot
  return Object.keys(snapshot).length > 0 ? snapshot : undefined
}

export function mergeResearchSnapshot(
  current: ResearchRunSnapshot | undefined,
  incoming: unknown,
): ResearchRunSnapshot | undefined {
  const next = meaningfulResearchSnapshot(incoming)
  if (!next) return current
  return { ...(current ?? {}), ...next }
}

export function snapshotWithResearchMetrics(
  snapshot: ResearchRunSnapshot | undefined,
  metrics: Record<string, unknown> | undefined,
  fallbackQueryCount: number | undefined = undefined,
): ResearchRunSnapshot | undefined {
  const projected: ResearchRunSnapshot = { ...(snapshot ?? {}) }
  const totalSources = metricNumber(
    metrics,
    'total_sources',
    'sources',
    'reference_count',
  )
  const totalQueries = metricNumber(metrics, 'total_queries', 'queries')
    ?? fallbackQueryCount
  const claims = metricNumber(
    metrics,
    'consolidated_claim_count',
    'claims',
    'claim_count',
  )
  if (totalSources !== undefined) projected.total_sources = totalSources
  if (totalQueries !== undefined) projected.total_queries = totalQueries
  if (claims !== undefined) projected.consolidated_claim_count = claims
  return Object.keys(projected).length > 0 ? projected : undefined
}

export function childProgressMessage(
  event: ResearchRunEvent,
): ChildProgressMessage | null {
  const message = nonEmptyString(event.data.message)
  if (event.type === 'inqtrix.progress.message' && message && message !== 'done') {
    return {
      sequence: event.sequence,
      severity: event.data.severity === 'warning'
        ? 'warning'
        : event.data.severity === 'error'
          ? 'error'
          : warningLike(message)
            ? 'warning'
            : 'info',
      text: message,
    }
  }
  if (event.type === 'inqtrix.node.failed' || event.type === 'inqtrix.run.failed') {
    const error = event.data.error
    const errorMessage = error && typeof error === 'object' && !Array.isArray(error)
      ? nonEmptyString((error as { message?: unknown }).message)
      : undefined
    return {
      sequence: event.sequence,
      severity: 'error',
      text: errorMessage ?? message ?? 'Run failed',
    }
  }
  if (event.type === 'inqtrix.run.cancelled') {
    return {
      sequence: event.sequence,
      severity: 'warning',
      text: message ?? nonEmptyString(event.data.reason) ?? 'Run cancelled',
    }
  }
  return null
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function warningLike(message: string): boolean {
  return /\b(ALGO-FAIL|Warnung|Warning|failed|fehlgeschlagen|Fallback|fallback|Limit|budget)\b/i.test(
    message,
  )
}

function metricNumber(
  metrics: Record<string, unknown> | undefined,
  ...keys: string[]
): number | undefined {
  for (const key of keys) {
    const value = metrics?.[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
  }
  return undefined
}
