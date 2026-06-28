import type { ResearchRunEvent } from '@/features/researchRuns/types'
import type {
  KnowledgeRunPlanRecord,
  KnowledgeRunProgressRecord,
  KnowledgeRunStepRecord,
  KnowledgeStepFacts,
} from '@/features/project/types'

/**
 * Pure event-to-step mapping for the live knowledge run card.
 *
 * The reducer feeds every `appendApiRunEvent` for a knowledge run
 * through this function; the card then renders the accumulated steps
 * as localized lines. Keeping the mapping pure (progress in, progress
 * out, no i18n, no time) makes the SSE protocol → UI contract unit
 * testable without React.
 *
 * Step lifecycle: a step appended as `running` is marked `done` by the
 * event that logically ends it (e.g. `retrieval.completed` finishes
 * the retrieval and vocabulary steps). A synthetic `answer` step is
 * inserted once the pipeline can only be generating the answer — after
 * the gate judged the evidence sufficient, after the last possible
 * gate round, or directly after retrieval when the plan runs no gate.
 * Terminal run events mark every remaining step done.
 */
export function applyKnowledgeRunEvent(
  progress: KnowledgeRunProgressRecord,
  event: ResearchRunEvent,
): KnowledgeRunProgressRecord {
  switch (event.type) {
    case 'inqtrix.knowledge.contextualized':
      return appendStep(progress, {
        facts: {
          contextMarker: stringFact(event.data.marker),
          rewritten: event.data.rewritten === true,
        },
        id: 'context',
        kind: 'context',
        status: 'done',
      })
    case 'inqtrix.knowledge.profile.resolved':
      return applyProfileResolved(progress, event)
    case 'inqtrix.knowledge.decomposition.completed':
      return appendStep(progress, {
        facts: { subQueryCount: numberFact(event.data.sub_query_count) },
        id: 'decompose',
        kind: 'decompose',
        status: 'done',
      })
    case 'inqtrix.knowledge.retrieval.completed':
      return applyRetrievalCompleted(progress, event)
    case 'inqtrix.knowledge.evidence.truncated':
      return appendStep(progress, {
        facts: {
          dropped: numberFact(event.data.dropped),
          kept: numberFact(event.data.kept),
        },
        id: 'evidence',
        kind: 'evidence',
        status: 'done',
      })
    case 'inqtrix.knowledge.gate.evaluated':
      return applyGateEvaluated(progress, event)
    case 'inqtrix.knowledge.gate.exhausted':
      return applyGateExhausted(progress)
    case 'inqtrix.knowledge.grounding.checked':
      return applyGroundingChecked(progress, event)
    case 'inqtrix.run.completed':
    case 'inqtrix.run.failed':
    case 'inqtrix.run.cancelled':
      return finishAllSteps(progress)
    default:
      return progress
  }
}

function applyProfileResolved(
  progress: KnowledgeRunProgressRecord,
  event: ResearchRunEvent,
): KnowledgeRunProgressRecord {
  const data = event.data
  const plan: KnowledgeRunPlanRecord = {
    autoReason: stringFact(data.auto_reason) ?? null,
    autoSelected: data.auto_selected === true,
    decompose: data.decompose === true,
    degradedStages: stringArrayFact(data.degraded_stages),
    gateRounds: numberFact(data.gate_rounds) ?? 0,
    grounding: data.grounding === true,
    profile: stringFact(data.profile) ?? '',
    requestedProfile: stringFact(data.requested_profile) ?? null,
    vocabularyBridge: data.vocabulary_bridge === true,
  }

  let steps: KnowledgeRunStepRecord[] = [
    ...progress.steps,
    {
      facts: {
        autoSelected: plan.autoSelected,
        degradedStages: plan.degradedStages,
        profile: plan.profile,
      },
      id: 'profile',
      kind: 'profile',
      status: 'done',
    },
  ]
  if (plan.vocabularyBridge) {
    steps = [...steps, { facts: {}, id: 'vocabulary', kind: 'vocabulary', status: 'running' }]
  }
  steps = [...steps, { facts: {}, id: 'retrieval', kind: 'retrieval', status: 'running' }]

  return { plan, steps }
}

function applyRetrievalCompleted(
  progress: KnowledgeRunProgressRecord,
  event: ResearchRunEvent,
): KnowledgeRunProgressRecord {
  const facts: KnowledgeStepFacts = {
    candidateCount: numberFact(event.data.candidate_count),
    collectionDocumentCount: numberFact(event.data.collection_document_count),
    topK: numberFact(event.data.top_k),
    finalK: numberFact(event.data.final_k),
    finalKOverridden: event.data.final_k_overridden === true,
  }
  let next = upsertStep(progress, {
    facts,
    id: 'retrieval',
    kind: 'retrieval',
    status: 'done',
  })
  next = markStepDone(next, 'vocabulary')

  // No gate planned (or nothing retrieved to judge): the next call is
  // already the answer generation.
  const gateRounds = next.plan?.gateRounds ?? 0
  if (gateRounds === 0 || (facts.candidateCount ?? 0) === 0) {
    next = appendAnswerStep(next)
  }
  return next
}

function applyGateEvaluated(
  progress: KnowledgeRunProgressRecord,
  event: ResearchRunEvent,
): KnowledgeRunProgressRecord {
  const round = numberFact(event.data.round) ?? 0
  const sufficient = event.data.sufficient === true
  // round 0 = first judgement; the plan's gateRounds counts rewrites,
  // so the visible total is rewrites + the initial judgement.
  const roundsTotal = (progress.plan?.gateRounds ?? 0) + 1
  let next = appendStep(progress, {
    facts: {
      rewritten: event.data.rewritten === true,
      round: round + 1,
      roundsTotal,
      sufficient,
    },
    id: `gate-${round}`,
    kind: 'gate',
    status: 'done',
  })
  if (sufficient || round + 1 >= roundsTotal) {
    next = appendAnswerStep(next)
  }
  return next
}

// The gate stopped early because a rewrite round added no new evidence (R5).
// Surface that honestly and move straight to the answer — the backend ran no
// further rounds, so the step ledger must not imply rounds it never executed.
function applyGateExhausted(
  progress: KnowledgeRunProgressRecord,
): KnowledgeRunProgressRecord {
  let next = appendStep(progress, {
    facts: {},
    id: 'gate-exhausted',
    kind: 'gate-exhausted',
    status: 'done',
  })
  next = appendAnswerStep(next)
  return next
}

function applyGroundingChecked(
  progress: KnowledgeRunProgressRecord,
  event: ResearchRunEvent,
): KnowledgeRunProgressRecord {
  // grounding.checked implies the answer was generated. Guarantee the answer
  // step exists before resolving it, so every terminal path (gate exhausted,
  // budget reached, no-gate) ends with a complete ledger.
  let next = progress.steps.some((step) => step.id === 'answer')
    ? progress
    : appendAnswerStep(progress)
  next = markStepDone(next, 'answer')
  next = appendStep(next, {
    facts: {
      quotesTotal: numberFact(event.data.quotes_total),
      quotesVerified: numberFact(event.data.quotes_verified),
    },
    id: 'grounding',
    kind: 'grounding',
    status: 'done',
  })
  return next
}

function appendAnswerStep(progress: KnowledgeRunProgressRecord): KnowledgeRunProgressRecord {
  return appendStep(progress, { facts: {}, id: 'answer', kind: 'answer', status: 'running' })
}

function appendStep(
  progress: KnowledgeRunProgressRecord,
  step: KnowledgeRunStepRecord,
): KnowledgeRunProgressRecord {
  if (progress.steps.some((existing) => existing.id === step.id)) {
    return upsertStep(progress, step)
  }
  return { ...progress, steps: [...progress.steps, step] }
}

function upsertStep(
  progress: KnowledgeRunProgressRecord,
  step: KnowledgeRunStepRecord,
): KnowledgeRunProgressRecord {
  if (!progress.steps.some((existing) => existing.id === step.id)) {
    return { ...progress, steps: [...progress.steps, step] }
  }
  return {
    ...progress,
    steps: progress.steps.map((existing) => (
      existing.id === step.id
        ? { ...existing, facts: { ...existing.facts, ...step.facts }, status: step.status }
        : existing
    )),
  }
}

function markStepDone(
  progress: KnowledgeRunProgressRecord,
  stepId: string,
): KnowledgeRunProgressRecord {
  if (!progress.steps.some((step) => step.id === stepId && step.status === 'running')) {
    return progress
  }
  return {
    ...progress,
    steps: progress.steps.map((step) => (
      step.id === stepId ? { ...step, status: 'done' } : step
    )),
  }
}

function finishAllSteps(progress: KnowledgeRunProgressRecord): KnowledgeRunProgressRecord {
  if (progress.steps.every((step) => step.status === 'done')) return progress
  return {
    ...progress,
    steps: progress.steps.map((step) => (
      step.status === 'done' ? step : { ...step, status: 'done' }
    )),
  }
}

function numberFact(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function stringFact(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}

function stringArrayFact(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === 'string')
}
