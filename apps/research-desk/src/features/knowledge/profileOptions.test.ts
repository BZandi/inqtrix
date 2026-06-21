import { describe, expect, it } from 'vitest'
import type { KnowledgeProfileManifestEntry } from '@/features/researchRuns/types'
import { knowledgeProfileOptionsFromManifest } from './profileOptions'

const manifest: KnowledgeProfileManifestEntry[] = [
  {
    degraded: [],
    id: 'schnell',
    stages: {
      decompose: false,
      gate_rounds: 0,
      grounding: false,
      rerank: false,
      report: false,
      vocabulary_bridge: false,
    },
  },
  {
    degraded: ['rerank', 'vocabulary_bridge'],
    id: 'gruendlich',
    stages: {
      decompose: false,
      gate_rounds: 2,
      grounding: true,
      rerank: false,
      report: false,
      vocabulary_bridge: false,
    },
  },
  {
    delegates_to: ['schnell', 'gruendlich'],
    id: 'auto',
  },
]

describe('knowledgeProfileOptionsFromManifest', () => {
  it('renders exactly the manifest profiles, never a hardcoded list', () => {
    const options = knowledgeProfileOptionsFromManifest(manifest)
    expect(options.map((option) => option.id)).toEqual(['schnell', 'gruendlich', 'auto'])
  })

  it('returns no options without a manifest (picker stays hidden)', () => {
    expect(knowledgeProfileOptionsFromManifest(undefined)).toEqual([])
    expect(knowledgeProfileOptionsFromManifest([])).toEqual([])
  })

  it('passes degraded stages through as the muted hint source', () => {
    const options = knowledgeProfileOptionsFromManifest(manifest)
    expect(options[0].degraded).toEqual([])
    expect(options[1].degraded).toEqual(['rerank', 'vocabulary_bridge'])
  })

  it('maps stage facts and flags the delegating auto entry', () => {
    const options = knowledgeProfileOptionsFromManifest(manifest)
    expect(options[1].stages).toEqual({
      decompose: false,
      gateRounds: 2,
      grounding: true,
      rerank: false,
      report: false,
      vocabularyBridge: false,
    })
    expect(options[2]).toMatchObject({
      delegatesTo: ['schnell', 'gruendlich'],
      isAuto: true,
      stages: null,
    })
  })

  it('drops malformed entries without an id', () => {
    const options = knowledgeProfileOptionsFromManifest([
      ...manifest,
      { id: '' } as KnowledgeProfileManifestEntry,
    ])
    expect(options.map((option) => option.id)).toEqual(['schnell', 'gruendlich', 'auto'])
  })
})
