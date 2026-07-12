import { describe, expect, it } from 'vitest'

import {
  replayTerminalEventPages,
  shouldReplayTerminalAgentEvents,
} from './useResearchRunApi'

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
