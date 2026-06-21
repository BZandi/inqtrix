import type { KnowledgeRunStepRecord } from '@/features/project/types'
import type { TranslationDictionary } from '@/i18n/translations'

export type KnowledgeStepLine = {
  id: string
  status: 'running' | 'done'
  primary: string
  secondary?: string
}

type KnowledgeStrings = TranslationDictionary['knowledge']

/** Render one captured step record as localized card lines. */
export function knowledgeStepLine(
  step: KnowledgeRunStepRecord,
  options: { collectionCount: number; t: KnowledgeStrings },
): KnowledgeStepLine {
  const { collectionCount, t } = options
  const facts = step.facts

  switch (step.kind) {
    case 'profile': {
      const template = facts.autoSelected ? t.stepProfileAuto : t.stepProfile
      return {
        id: step.id,
        primary: template.replace('{profile}', profileDisplayName(facts.profile ?? '', t)),
        secondary: facts.degradedStages && facts.degradedStages.length > 0
          ? t.stepDegraded.replace('{stages}', facts.degradedStages.join(', '))
          : undefined,
        status: step.status,
      }
    }
    case 'decompose':
      return {
        id: step.id,
        primary: t.stepDecompose.replace('{count}', String(facts.subQueryCount ?? 0)),
        status: step.status,
      }
    case 'vocabulary':
      return { id: step.id, primary: t.stepVocabulary, status: step.status }
    case 'retrieval': {
      // Done WITH a known document count → surface coverage ("N documents
      // searched"). Otherwise the collection-scoped line (running, or done when
      // the count couldn't be resolved).
      const template = facts.candidateCount === undefined
        ? collectionCount === 1 ? t.stepRetrievalRunningOne : t.stepRetrievalRunning
        : facts.collectionDocumentCount !== undefined
          ? t.stepRetrievalDoneDocs
          : collectionCount === 1 ? t.stepRetrievalDoneOne : t.stepRetrievalDone
      return {
        id: step.id,
        primary: template
          .replace('{count}', String(collectionCount))
          .replace('{docs}', String(facts.collectionDocumentCount ?? 0))
          .replace('{hits}', String(facts.candidateCount ?? 0)),
        status: step.status,
      }
    }
    case 'evidence': {
      const kept = facts.kept ?? 0
      return {
        id: step.id,
        primary: t.stepEvidence
          .replace('{kept}', String(kept))
          .replace('{total}', String(kept + (facts.dropped ?? 0))),
        status: step.status,
      }
    }
    case 'gate': {
      const verdict = facts.sufficient ? t.stepGateSufficient : t.stepGateInsufficient
      const rewrite = facts.rewritten ? ` · ${t.stepGateRewritten}` : ''
      return {
        id: step.id,
        primary: t.stepGate
          .replace('{round}', String(facts.round ?? 1))
          .replace('{total}', String(facts.roundsTotal ?? 1)),
        secondary: `${verdict}${rewrite}`,
        status: step.status,
      }
    }
    case 'gate-exhausted':
      return {
        id: step.id,
        primary: t.stepGateExhausted,
        status: step.status,
      }
    case 'answer':
      return {
        id: step.id,
        primary: step.status === 'running' ? t.stepAnswerRunning : t.stepAnswerDone,
        status: step.status,
      }
    case 'grounding':
      return {
        id: step.id,
        primary: t.stepGrounding
          .replace('{total}', String(facts.quotesTotal ?? 0))
          .replace('{verified}', String(facts.quotesVerified ?? 0)),
        status: step.status,
      }
  }
}

/** German display name for a known profile id; raw id otherwise — the
 * picker and step lines must never invent profiles the manifest does
 * not carry, but unknown ids still render honestly. */
export function profileDisplayName(profileId: string, t: KnowledgeStrings): string {
  switch (profileId) {
    case 'schnell':
      return t.profileSchnell
    case 'standard':
      return t.profileStandard
    case 'gruendlich':
      return t.profileGruendlich
    case 'tief':
      return t.profileTief
    case 'auto':
      return t.profileAuto
    default:
      return profileId
  }
}

/** One-line picker description (scope + expected latency) per known
 * profile id; empty for unknown ids. */
export function profileDescription(profileId: string, t: KnowledgeStrings): string {
  switch (profileId) {
    case 'schnell':
      return t.profileSchnellDescription
    case 'standard':
      return t.profileStandardDescription
    case 'gruendlich':
      return t.profileGruendlichDescription
    case 'tief':
      return t.profileTiefDescription
    case 'auto':
      return t.profileAutoDescription
    default:
      return ''
  }
}
