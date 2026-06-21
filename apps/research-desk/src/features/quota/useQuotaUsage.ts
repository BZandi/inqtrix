import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchQuotaUsage } from '@/api/inqtrixClient'
import type { QuotaDimensionUsage } from './model'

export type QuotaUsageState = {
  error: string | null
  rows: QuotaDimensionUsage[]
  status: 'error' | 'idle' | 'loading' | 'ready'
}

/** Server-truth quota usage for the signed-in caller (the composer meter).

 * Loads once when *enabled* and refreshes on a slow interval so the meter
 * tracks consumption without per-keystroke chatter. The whole meter is
 * gated upstream (oidc + ``capabilities.quota`` + authenticated), so a
 * disabled hook stays idle with no request (this is what keeps
 * none/apikey/demo deployments byte-identical). ``reload`` is exposed for
 * a caller that wants to force an immediate refresh.
 *
 * The hook's effectful behaviour (race guard, interval cleanup,
 * disabled = no fetch) is intentionally NOT unit-tested: this repo's
 * vitest runs in the node environment (pure logic only, see
 * vitest.config.ts); the pure parts live in model.ts and ARE covered.
 */
const REFRESH_MS = 30_000

export function useQuotaUsage(enabled: boolean) {
  const [state, setState] = useState<QuotaUsageState>({
    error: null,
    rows: [],
    status: 'idle',
  })
  const generationRef = useRef(0)

  const reload = useCallback(async () => {
    if (!enabled) return
    const generation = ++generationRef.current
    setState((current) => ({ ...current, status: 'loading' }))
    try {
      const rows = await fetchQuotaUsage()
      if (generationRef.current !== generation) return
      setState({ error: null, rows, status: 'ready' })
    } catch (error) {
      if (generationRef.current !== generation) return
      setState({
        error: error instanceof Error ? error.message : String(error),
        rows: [],
        status: 'error',
      })
    }
  }, [enabled])

  useEffect(() => {
    if (!enabled) {
      generationRef.current += 1
      setState({ error: null, rows: [], status: 'idle' })
      return
    }
    void reload()
    const timer = setInterval(() => {
      void reload()
    }, REFRESH_MS)
    return () => clearInterval(timer)
  }, [enabled, reload])

  return { reload, state }
}
