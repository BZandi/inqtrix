import { describe, expect, it } from 'vitest'

import type { ResearchRunEvent } from '@/features/researchRuns/types'
import {
  childProgressMessage,
  meaningfulResearchSnapshot,
  researchNodePhase,
  snapshotWithResearchMetrics,
} from './taskProgress'

describe('agent child progress', () => {
  it('never maps an empty or unknown child snapshot to answer', () => {
    expect(meaningfulResearchSnapshot({})).toBeUndefined()
    expect(researchNodePhase(undefined)).toBeNull()
    expect(researchNodePhase('unknown-node')).toBeNull()
    expect(researchNodePhase('search')).toBe('search')
  })

  it('surfaces progress warnings and terminal errors', () => {
    const progress: ResearchRunEvent = {
      created_at: 1,
      data: { message: 'Fallback auf reduzierte Suche' },
      run_id: 'child',
      sequence: 2,
      type: 'inqtrix.progress.message',
    }
    expect(childProgressMessage(progress)?.severity).toBe('warning')
    expect(childProgressMessage({
      ...progress,
      data: { error: { message: 'Token-Budget erreicht' } },
      sequence: 3,
      type: 'inqtrix.run.failed',
    })).toMatchObject({ severity: 'error', text: 'Token-Budget erreicht' })
  })

  it('maps durable local task metric keys without inventing counts', () => {
    expect(snapshotWithResearchMetrics(undefined, {
      reference_count: 7,
      claim_count: 4,
    }, 1)).toMatchObject({
      total_sources: 7,
      consolidated_claim_count: 4,
      total_queries: 1,
    })
    expect(snapshotWithResearchMetrics(undefined, undefined)).toBeUndefined()
  })
})
