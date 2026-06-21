import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  fetchOutgoingShareCounts,
  listSharedWithMe,
} from '@/api/inqtrixClient'
import { demoOutgoingShareCounts, demoSharedWithMe } from './demoShares'
import { sharedWithMeByResourceId } from './shareModel'
import type { SharedWithMeEntry } from './types'

/**
 * Outgoing-share counts for the badge layer, keyed by resource id.
 * Errors degrade to "no badges" silently visible in the console — a
 * missing count must never block the run list itself.
 */
export function useOutgoingShareCounts(
  resourceType: string,
  resourceIds: readonly string[],
  enabled: boolean,
  demo = false,
) {
  const [counts, setCounts] = useState<Record<string, number>>({})
  const idsKey = useMemo(() => [...resourceIds].sort().join('|'), [resourceIds])
  const idsRef = useRef<readonly string[]>(resourceIds)
  idsRef.current = resourceIds

  const refresh = useCallback(async () => {
    if (!enabled || idsRef.current.length === 0) {
      setCounts({})
      return
    }
    if (demo) {
      setCounts(demoOutgoingShareCounts(resourceType, idsRef.current))
      return
    }
    try {
      setCounts(await fetchOutgoingShareCounts(resourceType, idsRef.current))
    } catch (error) {
      console.warn('Share counts unavailable:', error)
      setCounts({})
    }
  }, [demo, enabled, resourceType])

  useEffect(() => {
    void refresh()
    // idsKey re-triggers when the visible resource set changes.
  }, [idsKey, refresh])

  return { counts, refresh }
}

/**
 * The caller's shared-in grants of one kind, keyed by resource id —
 * source of the "Geteilt von <Name>" recipient badge.
 */
export function useSharedWithMe(
  resourceType: string,
  enabled: boolean,
  demo = false,
) {
  const [entries, setEntries] = useState<readonly SharedWithMeEntry[]>([])

  const refresh = useCallback(async () => {
    if (!enabled) {
      setEntries([])
      return
    }
    if (demo) {
      setEntries(
        demoSharedWithMe.filter((entry) => entry.resource_type === resourceType),
      )
      return
    }
    try {
      setEntries(await listSharedWithMe(resourceType))
    } catch (error) {
      console.warn('Shared-with-me unavailable:', error)
      setEntries([])
    }
  }, [demo, enabled, resourceType])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const byResourceId = useMemo(() => sharedWithMeByResourceId(entries), [entries])
  return { byResourceId, refresh }
}
