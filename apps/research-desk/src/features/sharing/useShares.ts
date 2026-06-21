import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createShares,
  listShares,
  revokeShare,
} from '@/api/inqtrixClient'
import {
  grantDemoShares,
  listDemoShares,
  revokeDemoShare,
} from './demoShares'
import type { ShareInvitee, ShareRecordInfo } from './types'

export type SharesState = {
  error: string | null
  records: ShareRecordInfo[]
  status: 'error' | 'idle' | 'loading' | 'ready'
}

/**
 * Server-truth share state for ONE resource (the open dialog). No
 * ProjectState involvement by design: shares live on the server, the
 * dialog is their only surface, and every mutation re-reads the
 * listing so re-grants and races resolve to what the server says.
 */
export function useShares(
  resourceType: string,
  resourceId: string | null,
  demo = false,
) {
  const [state, setState] = useState<SharesState>({
    error: null,
    records: [],
    status: 'idle',
  })
  const generationRef = useRef(0)

  const reload = useCallback(async () => {
    if (!resourceId) return
    const generation = ++generationRef.current
    setState((current) => ({ ...current, status: 'loading' }))
    try {
      const records = demo
        ? listDemoShares(resourceType, resourceId)
        : await listShares(resourceType, resourceId)
      if (generationRef.current !== generation) return
      setState({ error: null, records, status: 'ready' })
    } catch (error) {
      if (generationRef.current !== generation) return
      setState({
        error: error instanceof Error ? error.message : String(error),
        records: [],
        status: 'error',
      })
    }
  }, [demo, resourceId, resourceType])

  useEffect(() => {
    if (!resourceId) {
      generationRef.current += 1
      setState({ error: null, records: [], status: 'idle' })
      return
    }
    void reload()
  }, [reload, resourceId])

  const grant = useCallback(
    async (invitees: ShareInvitee[]) => {
      if (!resourceId || invitees.length === 0) return
      if (demo) grantDemoShares(resourceType, resourceId, invitees)
      else await createShares(resourceType, resourceId, invitees)
      await reload()
    },
    [demo, reload, resourceId, resourceType],
  )

  const revoke = useCallback(
    async (shareId: string) => {
      if (demo) revokeDemoShare(shareId)
      else await revokeShare(shareId)
      await reload()
    },
    [demo, reload],
  )

  return { grant, reload, revoke, state }
}
