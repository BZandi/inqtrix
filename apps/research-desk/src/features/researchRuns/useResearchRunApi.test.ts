import { describe, expect, it } from 'vitest'

import {
  replayTerminalEventPages,
  shouldReplayTerminalAgentEvents,
  waitForRunTerminal,
} from './useResearchRunApi'
import type { ResearchRunStatus, ResearchRunSummary } from './types'

describe('shouldReplayTerminalAgentEvents', () => {
  it('replays terminal root agent runs for both engines', () => {
    expect(shouldReplayTerminalAgentEvents({
      kind: 'agent',
      mode: 'workspace_agent',
      status: 'completed',
    })).toBe(true)
    expect(shouldReplayTerminalAgentEvents({
      kind: 'agent',
      mode: 'agent_kernel',
      status: 'failed',
    })).toBe(true)
  })

  it('does not replay children, active agents, or standard runs', () => {
    expect(shouldReplayTerminalAgentEvents({
      kind: 'agent_child',
      mode: 'workspace_agent',
      status: 'completed',
    })).toBe(false)
    expect(shouldReplayTerminalAgentEvents({
      kind: 'agent',
      mode: 'workspace_agent',
      status: 'running',
    })).toBe(false)
    expect(shouldReplayTerminalAgentEvents({
      kind: 'standard',
      mode: 'research',
      status: 'completed',
    })).toBe(false)
  })
})

describe('replayTerminalEventPages', () => {
  it('replays every page in sequence until the durable terminal page', async () => {
    const afterValues: Array<number | null> = []
    const delivered: number[] = []
    await replayTerminalEventPages({
      fetchPage: async (after) => {
        afterValues.push(after)
        return after === null
          ? {
            data: [{
              created_at: 1,
              data: {},
              run_id: 'r1',
              sequence: 1,
              type: 'inqtrix.agent.phase.changed',
            }],
            terminal: false,
          }
          : {
            data: [{
              created_at: 2,
              data: {},
              run_id: 'r1',
              sequence: 2,
              type: 'inqtrix.run.completed',
            }],
            terminal: true,
          }
      },
      onEvent: (event) => delivered.push(event.sequence),
    })
    expect(afterValues).toEqual([null, 1])
    expect(delivered).toEqual([1, 2])
  })

  it('fails visibly when a non-terminal replay cannot advance', async () => {
    await expect(replayTerminalEventPages({
      fetchPage: async () => ({ data: [], terminal: false }),
      onEvent: () => undefined,
    })).rejects.toThrow('before a terminal page')
  })
})

function summaryWithStatus(status: ResearchRunStatus): ResearchRunSummary {
  return { run_id: 'r1', status } as ResearchRunSummary
}

describe('waitForRunTerminal', () => {
  it('resolves on the first summary when it is already terminal', async () => {
    let sleeps = 0
    const summary = await waitForRunTerminal({
      fetchSummary: async () => summaryWithStatus('cancelled'),
      sleep: async () => {
        sleeps += 1
      },
    })
    expect(summary?.status).toBe('cancelled')
    expect(sleeps).toBe(0)
  })

  it('polls with the injected sleep until the run terminalizes', async () => {
    const statuses: ResearchRunStatus[] = ['running', 'running', 'cancelled']
    let fetches = 0
    let sleptMs = 0
    const summary = await waitForRunTerminal({
      fetchSummary: async () => summaryWithStatus(statuses[fetches++]),
      pollMs: 100,
      maxWaitMs: 1000,
      sleep: async (ms) => {
        sleptMs += ms
      },
    })
    expect(summary?.status).toBe('cancelled')
    expect(fetches).toBe(3)
    expect(sleptMs).toBe(200)
  })

  it('returns null once the wait bound is exhausted', async () => {
    let fetches = 0
    const summary = await waitForRunTerminal({
      fetchSummary: async () => {
        fetches += 1
        return summaryWithStatus('running')
      },
      pollMs: 100,
      maxWaitMs: 250,
      sleep: async () => undefined,
    })
    expect(summary).toBeNull()
    // 0ms, 100ms, 200ms waited -> fourth fetch sees waited >= bound.
    expect(fetches).toBe(4)
  })

  it('propagates fetch errors to the caller', async () => {
    await expect(waitForRunTerminal({
      fetchSummary: async () => {
        throw new Error('summary fetch failed')
      },
      sleep: async () => undefined,
    })).rejects.toThrow('summary fetch failed')
  })
})
