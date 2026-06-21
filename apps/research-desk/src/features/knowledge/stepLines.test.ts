import { describe, expect, it } from 'vitest'
import { translations } from '@/i18n/translations'
import type { KnowledgeRunStepRecord } from '@/features/project/types'
import { knowledgeStepLine } from './stepLines'

const t = translations.de.knowledge

function line(step: KnowledgeRunStepRecord, collectionCount = 1) {
  return knowledgeStepLine(step, { collectionCount, t })
}

describe('knowledgeStepLine', () => {
  it('renders the deep RAG live ledger lines used by the demo twin', () => {
    const steps: KnowledgeRunStepRecord[] = [
      {
        facts: { autoSelected: false, degradedStages: ['rerank'], profile: 'tief' },
        id: 'profile',
        kind: 'profile',
        status: 'done',
      },
      { facts: {}, id: 'vocabulary', kind: 'vocabulary', status: 'done' },
      {
        facts: { candidateCount: 8, topK: 8 },
        id: 'retrieval',
        kind: 'retrieval',
        status: 'done',
      },
      {
        facts: { subQueryCount: 0 },
        id: 'decompose',
        kind: 'decompose',
        status: 'done',
      },
      {
        facts: { rewritten: true, round: 4, roundsTotal: 4, sufficient: false },
        id: 'gate-3',
        kind: 'gate',
        status: 'done',
      },
      { facts: {}, id: 'answer', kind: 'answer', status: 'running' },
    ]

    expect(line(steps[0])).toMatchObject({
      primary: 'Profil: Tief',
      secondary: 'Reduzierte Stufen: rerank',
    })
    expect(line(steps[1]).primary).toBe('Formuliere fachsprachlich um')
    expect(line(steps[2]).primary).toBe('Durchsuche 1 Sammlung… (8 Treffer)')
    expect(line(steps[3]).primary).toBe('Zerlege Frage in 0 Teilfragen')
    expect(line(steps[4])).toMatchObject({
      primary: 'Bewerte Evidenz (Runde 4/4)',
      secondary: 'nicht ausreichend · Suchanfrage umformuliert',
    })
    expect(line(steps[5]).primary).toBe('Formuliere Antwort…')
  })

  it('surfaces the searched document count when known (coverage signal)', () => {
    const retrieval: KnowledgeRunStepRecord = {
      facts: { candidateCount: 16, collectionDocumentCount: 11, topK: 8 },
      id: 'retrieval',
      kind: 'retrieval',
      status: 'done',
    }
    expect(line(retrieval).primary).toBe('Durchsuche 11 Dokumente… (16 Treffer)')
  })

  it('falls back to the collection line when the document count is unknown', () => {
    const retrieval: KnowledgeRunStepRecord = {
      facts: { candidateCount: 16, topK: 8 },
      id: 'retrieval',
      kind: 'retrieval',
      status: 'done',
    }
    expect(line(retrieval).primary).toBe('Durchsuche 1 Sammlung… (16 Treffer)')
  })
})
