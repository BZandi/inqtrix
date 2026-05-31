import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import {
  cancelResearchRun,
  createResearchRun,
  fetchHealth,
  fetchResearchRunResult,
  fetchStacks,
  hasHttpStatus,
  listResearchRuns,
  streamResearchRunEvents,
} from '@/api/inqtrixClient'
import type {
  CreateResearchRunRequest,
  InqtrixHealth,
  InqtrixStack,
  InqtrixStackList,
  ResearchRunEvent,
  ResearchRunResult,
  ResearchRunSummary,
  StackDiscoveryStatus,
} from './types'

type LiveRunCallbacks = {
  onEvent: (event: ResearchRunEvent) => void
  onResult: (result: ResearchRunResult) => void
  onRunError: (runId: string, message: string) => void
  onSummary: (summary: ResearchRunSummary, options?: { select?: boolean }) => void
}

type UseResearchRunApiOptions = LiveRunCallbacks & {
  apiKey?: string
  enabled: boolean
  workspaceId: string
}

type StackDiscoveryCacheEntry = {
  payload?: InqtrixStackList
  promise?: Promise<InqtrixStackList | null>
  status: StackDiscoveryStatus
}

const stackDiscoveryCache = new Map<string, StackDiscoveryCacheEntry>()
const STACK_DISCOVERY_CACHE_KEY = 'default'

export function useResearchRunApi({
  apiKey,
  enabled,
  onEvent,
  onResult,
  onRunError,
  onSummary,
  workspaceId,
}: UseResearchRunApiOptions) {
  const [health, setHealth] = useState<InqtrixHealth | null>(null)
  const [defaultStackName, setDefaultStackName] = useState<string | null>(null)
  const [stackNames, setStackNames] = useState<string[]>([])
  const [stackDiscoveryStatus, setStackDiscoveryStatus] = useState<StackDiscoveryStatus>('unknown')
  const [stacks, setStacks] = useState<InqtrixStack[]>([])
  const [lastError, setLastError] = useState<string | null>(null)
  const streamsRef = useRef(new Map<string, AbortController>())
  const callbacksRef = useRef<LiveRunCallbacks>({
    onEvent,
    onResult,
    onRunError,
    onSummary,
  })

  useEffect(() => {
    callbacksRef.current = {
      onEvent,
      onResult,
      onRunError,
      onSummary,
    }
  }, [onEvent, onResult, onRunError, onSummary])

  const loadResult = useCallback(async (runId: string) => {
    try {
      const result = await fetchResearchRunResult(runId, { apiKey, workspaceId })
      callbacksRef.current.onResult(result)
    } catch (error) {
      callbacksRef.current.onRunError(runId, messageFromError(error))
    }
  }, [apiKey, workspaceId])

  const startStream = useCallback((summary: ResearchRunSummary) => {
    if (streamsRef.current.has(summary.run_id)) return
    if (terminalStatus(summary.status)) {
      if (summary.status === 'completed') {
        void loadResult(summary.run_id)
      }
      return
    }

    const controller = new AbortController()
    streamsRef.current.set(summary.run_id, controller)
    void streamResearchRunEvents(summary.events_url, {
      apiKey,
      signal: controller.signal,
      workspaceId,
      onEvent: (event) => {
        callbacksRef.current.onEvent(event)
        if (event.type === 'inqtrix.run.completed') {
          void loadResult(event.run_id)
        }
      },
    }).catch((error) => {
      if (controller.signal.aborted) return
      callbacksRef.current.onRunError(summary.run_id, messageFromError(error))
    }).finally(() => {
      streamsRef.current.delete(summary.run_id)
    })
  }, [apiKey, loadResult, workspaceId])

  const submitRun = useCallback(async (request: CreateResearchRunRequest) => {
    try {
      setLastError(null)
      const summary = await createResearchRun(request, { apiKey, workspaceId })
      callbacksRef.current.onSummary(summary, { select: true })
      startStream(summary)
    } catch (error) {
      const message = messageFromError(error)
      setLastError(message)
      console.warn('Inqtrix run creation failed.', error)
    }
  }, [apiKey, startStream, workspaceId])

  const cancelRun = useCallback(async (runId: string) => {
    try {
      setLastError(null)
      const summary = await cancelResearchRun(runId, { apiKey, workspaceId })
      callbacksRef.current.onSummary(summary)
      startStream(summary)
    } catch (error) {
      const message = messageFromError(error)
      setLastError(message)
      throw new Error(message, { cause: error })
    }
  }, [apiKey, startStream, workspaceId])

  useEffect(() => {
    if (!enabled) {
      setHealth(null)
      setDefaultStackName(null)
      setStackNames([])
      setStackDiscoveryStatus('unknown')
      setStacks([])
      for (const controller of streamsRef.current.values()) {
        controller.abort()
      }
      streamsRef.current.clear()
      return undefined
    }

    let ignore = false
    for (const controller of streamsRef.current.values()) {
      controller.abort()
    }
    streamsRef.current.clear()

    async function hydrate() {
      let healthPayload: InqtrixHealth | null = null
      try {
        healthPayload = await fetchHealth()
        if (!ignore) setHealth(healthPayload)
      } catch (error) {
        if (!ignore) setLastError(messageFromError(error))
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

      if (healthPayload?.auth_required && !apiKey) {
        return
      }

      try {
        const summaries = await listResearchRuns({ apiKey, workspaceId })
        if (ignore) return
        for (const summary of summaries) {
          callbacksRef.current.onSummary(summary)
          startStream(summary)
        }
      } catch (error) {
        if (!ignore) setLastError(messageFromError(error))
      }
    }

    void hydrate()

    return () => {
      ignore = true
    }
  }, [apiKey, enabled, startStream, workspaceId])

  useEffect(() => {
    return () => {
      for (const controller of streamsRef.current.values()) {
        controller.abort()
      }
      streamsRef.current.clear()
    }
  }, [])

  return {
    cancelRun,
    defaultStackName,
    health,
    lastError,
    stackDiscoveryStatus,
    stackNames,
    stacks,
    submitRun,
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

function terminalStatus(status: ResearchRunSummary['status']) {
  return status === 'completed'
    || status === 'failed'
    || status === 'cancelled'
    || status === 'expired'
}

function messageFromError(error: unknown) {
  return error instanceof Error ? error.message : 'Inqtrix request failed.'
}
