import { useEffect, useState } from 'react'
import {
  fetchCapabilities,
  fetchHealth,
  fetchStacks,
  hasHttpStatus,
} from '@/api/inqtrixClient'
import type {
  InqtrixCapabilities,
  InqtrixHealth,
  InqtrixStack,
  InqtrixStackList,
  StackDiscoveryStatus,
} from './types'

type UseServerDiscoveryOptions = {
  enabled: boolean
}

type StackDiscoveryCacheEntry = {
  payload?: InqtrixStackList
  promise?: Promise<InqtrixStackList | null>
  status: StackDiscoveryStatus
}

const stackDiscoveryCache = new Map<string, StackDiscoveryCacheEntry>()
const STACK_DISCOVERY_CACHE_KEY = 'default'

/**
 * Workspace-INDEPENDENT server discovery: health, the capability manifest, and
 * the stack catalog. Split out of {@link useResearchRunApi} because these probes
 * gate the auth/namespace resolution (cookie mode is derived from `health`),
 * which in turn produces the per-user workspace namespace that run operations
 * scope to. Resolving discovery in its own hook keeps that ordering linear --
 * discovery -> auth -> namespace -> run ops -- so the run hook can receive the
 * already-resolved namespace instead of re-probing the session itself (which
 * left a window where a run could be scoped to the browser id before the
 * namespace resolved).
 *
 * None of these endpoints are workspace-scoped, so the hook takes no
 * `workspaceId` and re-runs only on the `enabled` toggle (demo on/off). Probe
 * failures surface via `lastError`; a 404 from the capability manifest is an
 * older backend and is swallowed (every capability-gated affordance stays
 * hidden), matching the prior behaviour exactly.
 */
export function useServerDiscovery({ enabled }: UseServerDiscoveryOptions) {
  const [health, setHealth] = useState<InqtrixHealth | null>(null)
  const [capabilities, setCapabilities] = useState<InqtrixCapabilities | null>(null)
  const [defaultStackName, setDefaultStackName] = useState<string | null>(null)
  const [stackNames, setStackNames] = useState<string[]>([])
  const [stackDiscoveryStatus, setStackDiscoveryStatus] = useState<StackDiscoveryStatus>('unknown')
  const [stacks, setStacks] = useState<InqtrixStack[]>([])
  const [lastError, setLastError] = useState<string | null>(null)
  // `true` once the health probe has settled (success OR failure), so the auth
  // mode is determinable. Consumers gate run-listing on this: until health
  // resolves, `health` is null and a naive auth-mode read would default to
  // `none` and list prematurely under the browser id.
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (!enabled) {
      setHealth(null)
      setCapabilities(null)
      setDefaultStackName(null)
      setStackNames([])
      setStackDiscoveryStatus('unknown')
      setStacks([])
      setLastError(null)
      setReady(false)
      return undefined
    }

    let ignore = false

    async function discover() {
      // Clear any error from a prior pass so `lastError` reflects only this
      // discovery attempt (a stale error must not stick once it is resolved).
      if (!ignore) setLastError(null)
      try {
        const healthPayload = await fetchHealth()
        if (!ignore) setHealth(healthPayload)
      } catch (error) {
        if (!ignore) setLastError(messageFromError(error))
      }
      // Health has now been probed (ok or failed): the auth mode is
      // determinable, so listing may proceed once the caller is admitted.
      if (!ignore) setReady(true)

      try {
        const capabilitiesPayload = await fetchCapabilities()
        if (!ignore) setCapabilities(capabilitiesPayload)
      } catch (error) {
        // 404 = older backend without the capability manifest; every
        // capability-gated affordance simply stays hidden. Anything
        // else is a real error and surfaces like the other probes.
        if (!ignore) {
          setCapabilities(null)
          if (!hasHttpStatus(error, 404)) setLastError(messageFromError(error))
        }
      }

      try {
        const stackPayload = await discoverStacks()
        if (!ignore) {
          if (stackPayload) {
            setDefaultStackName(stackPayload.default)
            setStackNames(stackPayload.stacks.map((stack) => stack.name))
            setStackDiscoveryStatus('available')
            setStacks(stackPayload.stacks)
          } else {
            setDefaultStackName(null)
            setStackNames([])
            setStackDiscoveryStatus('unsupported')
            setStacks([])
          }
        }
      } catch (error) {
        if (!ignore) {
          setDefaultStackName(null)
          setStackNames([])
          setStackDiscoveryStatus('unknown')
          setStacks([])
          setLastError(messageFromError(error))
        }
      }
    }

    void discover()

    return () => {
      ignore = true
    }
  }, [enabled])

  return {
    capabilities,
    defaultStackName,
    health,
    lastError,
    ready,
    stackDiscoveryStatus,
    stackNames,
    stacks,
  }
}

async function discoverStacks() {
  const cached = stackDiscoveryCache.get(STACK_DISCOVERY_CACHE_KEY)
  if (cached?.status === 'unsupported') return null
  if (cached?.payload) return cached.payload
  if (cached?.promise) return cached.promise

  const entry: StackDiscoveryCacheEntry = {
    status: 'unknown',
  }
  entry.promise = fetchStacks()
    .then((payload) => {
      entry.payload = payload
      entry.status = 'available'
      return payload
    })
    .catch((error) => {
      if (hasHttpStatus(error, 404)) {
        entry.status = 'unsupported'
        return null
      }
      stackDiscoveryCache.delete(STACK_DISCOVERY_CACHE_KEY)
      throw error
    })
    .finally(() => {
      entry.promise = undefined
    })
  stackDiscoveryCache.set(STACK_DISCOVERY_CACHE_KEY, entry)
  return entry.promise
}

function messageFromError(error: unknown) {
  return error instanceof Error ? error.message : 'Inqtrix request failed.'
}
