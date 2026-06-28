import { useCallback, useEffect, useState } from 'react'
import {
  acceptShare,
  fetchOutgoingShares,
  fetchSharingInbox,
  revokeShare,
} from '@/api/inqtrixClient'
import {
  acceptDemoShare,
  demoOutgoingShares,
  demoSharingInbox,
  dropDemoInboxShare,
} from './demoShares'
import type { InboxShare, OutgoingShare } from './types'

type SharingStatus = 'idle' | 'loading' | 'ready' | 'error'

export type SharingInboxState = {
  accepted: InboxShare[]
  error: string | null
  mutationError: string | null
  outgoing: OutgoingShare[]
  pending: InboxShare[]
  status: SharingStatus
}

export type SharingInboxHandle = {
  accept: (shareId: string) => Promise<void>
  drop: (shareId: string) => Promise<void>
  pendingCount: number
  reload: () => Promise<void>
  state: SharingInboxState
}

const EMPTY: SharingInboxState = {
  accepted: [],
  error: null,
  mutationError: null,
  outgoing: [],
  pending: [],
  status: 'idle',
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Server-truth state for the sharing settings panel and the nav badge: the
 * recipient inbox (pending consent queue + accepted) and the caller's outgoing
 * shares, plus the recipient mutations — accept a pending invite, drop one's
 * own share (decline if pending, leave if accepted; same DELETE either way).
 * Every mutation re-reads, so the panel and the badge can never disagree with
 * the server. Disabled (gate off) resolves to the empty state without a fetch.
 */
export function useSharingInbox({
  demo,
  enabled,
}: {
  demo: boolean
  enabled: boolean
}): SharingInboxHandle {
  const [state, setState] = useState<SharingInboxState>(EMPTY)

  const reload = useCallback(async () => {
    if (!enabled) {
      setState(EMPTY)
      return
    }
    if (demo) {
      const inbox = demoSharingInbox()
      setState({
        accepted: inbox.accepted,
        error: null,
        mutationError: null,
        outgoing: demoOutgoingShares(),
        pending: inbox.pending,
        status: 'ready',
      })
      return
    }
    setState((prev) => ({
      ...prev,
      status: prev.status === 'ready' ? 'ready' : 'loading',
    }))
    try {
      const [inbox, outgoing] = await Promise.all([
        fetchSharingInbox(),
        fetchOutgoingShares(),
      ])
      setState({
        accepted: inbox.accepted,
        error: null,
        mutationError: null,
        outgoing,
        pending: inbox.pending,
        status: 'ready',
      })
    } catch (error) {
      setState((prev) => ({ ...prev, error: errorText(error), status: 'error' }))
    }
  }, [demo, enabled])

  useEffect(() => {
    void reload()
  }, [reload])

  const accept = useCallback(
    async (shareId: string) => {
      setState((prev) => ({ ...prev, mutationError: null }))
      try {
        if (demo) acceptDemoShare(shareId)
        else await acceptShare(shareId)
        await reload()
      } catch (error) {
        setState((prev) => ({ ...prev, mutationError: errorText(error) }))
      }
    },
    [demo, reload],
  )

  const drop = useCallback(
    async (shareId: string) => {
      setState((prev) => ({ ...prev, mutationError: null }))
      try {
        if (demo) dropDemoInboxShare(shareId)
        else await revokeShare(shareId)
        await reload()
      } catch (error) {
        setState((prev) => ({ ...prev, mutationError: errorText(error) }))
      }
    },
    [demo, reload],
  )

  return { accept, drop, pendingCount: state.pending.length, reload, state }
}
