import { useMemo } from 'react'
import { seedQuotaUsage } from './demo'
import { useQuotaMeterGate } from './QuotaMeterContext'
import { useQuotaUsage } from './useQuotaUsage'

export type EmbeddingQuota = {
  used: number
  /** Effective limit, or ``null`` when unlimited. */
  limit: number | null
  /** Whether the monthly embedding budget is reached (limited and at/over). */
  exhausted: boolean
  /** Flow-window start (unix seconds) — drives the month label on the bar. */
  periodStart: number
  /** Flow-window reset (unix seconds), or ``0`` when not windowed. */
  resetAt: number
}

/** The caller's embedding-token usage for the Database subpage.

 * Returns ``null`` when quotas do not apply (so the stat and the
 * reindex gate vanish, keeping non-oidc deployments byte-identical).
 * Shares the meter's gate ({@link QuotaMeterProvider}) and poll.
 */
export function useEmbeddingQuota(): EmbeddingQuota | null {
  const { enabled, demo } = useQuotaMeterGate()
  const { state } = useQuotaUsage(enabled && !demo)
  const now = useMemo(() => Math.floor(Date.now() / 1000), [])

  if (!enabled) return null
  const rows = demo ? seedQuotaUsage(now) : state.rows
  const row = rows.find((entry) => entry.dimension === 'embedding_tokens')
  const used = row?.used ?? 0
  const limit = row?.limit ?? null
  return {
    exhausted: limit != null && limit > 0 && used >= limit,
    limit,
    periodStart: row?.period_start ?? now,
    resetAt: row?.reset_at ?? 0,
    used,
  }
}
