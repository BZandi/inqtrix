import type { ResearchRunResult } from '@/features/researchRuns/types'
import type {
  KnowledgeAnswerRecord,
  KnowledgeReferenceRecord,
} from '@/features/project/types'

/**
 * Parse the document target out of a knowledge citation URL. The
 * algorithm emits either `{base}/v1/sources/{document_id}?chunk={n}`
 * (public base URL configured) or the `inqtrix://documents/{id}#chunk-{n}`
 * fallback. Unknown shapes yield null — the reference then renders
 * without a viewer link instead of opening the wrong document.
 */
export function parseKnowledgeReferenceUrl(
  url: string,
): { documentId: string; chunkIndex: number | null } | null {
  const http = url.match(/\/v1\/sources\/([^/?#]+)(?:\?chunk=(\d+))?/)
  if (http) {
    return { chunkIndex: http[2] ? Number(http[2]) : null, documentId: http[1] }
  }
  const internal = url.match(/^inqtrix:\/\/documents\/([^/#?]+)(?:#chunk-(\d+))?/)
  if (internal) {
    return { chunkIndex: internal[2] ? Number(internal[2]) : null, documentId: internal[1] }
  }
  return null
}

/** The honest no-evidence answer emitted by the knowledge algorithm. */
const REFUSAL_PATTERN = /keine relevanten\s+Inhalte/i

export function isKnowledgeRefusal(answer: string, referenceCount: number): boolean {
  return referenceCount === 0 && REFUSAL_PATTERN.test(answer)
}

/**
 * Project a completed native run result onto the knowledge answer
 * record. Reads the knowledge-specific result keys defensively: rich
 * deployments deliver `report_references` (with document titles),
 * grounding quotes and profile facts; older payloads degrade to the
 * plain `references` export without breaking the card.
 */
export function knowledgeAnswerFromRunResult(result: ResearchRunResult): KnowledgeAnswerRecord {
  const rawReferences = result.report_references ?? result.references ?? []
  const references: KnowledgeReferenceRecord[] = rawReferences.map((reference) => {
    const parsed = parseKnowledgeReferenceUrl(reference.url)
    const explicit = reference as {
      title?: string
      document_id?: string | null
      chunk_index?: number | null
      excerpt?: string | null
      source_text?: string | null
      page_number?: number | null
    }
    return {
      // Prefer the explicit backend fields (reliable open + the exact passage);
      // fall back to the URL-parsed id for older payloads.
      chunkIndex: explicit.chunk_index ?? parsed?.chunkIndex ?? null,
      documentId: explicit.document_id ?? parsed?.documentId ?? null,
      excerpt: explicit.excerpt ?? null,
      sourceText: explicit.source_text ?? null,
      pageNumber: explicit.page_number ?? null,
      label: reference.label,
      tier: String(reference.tier ?? 'unknown'),
      title: explicit.title,
      url: reference.url,
    }
  })

  const gate = result.knowledge_gate
  const grounding = result.knowledge_grounding
  const profile = result.knowledge_profile

  return {
    answerMarkdown: result.answer,
    autoSelected: profile?.auto_selected === true,
    candidateCount: result.knowledge_candidates ?? null,
    degradedStages: profile?.degraded_stages ?? [],
    evidenceUsed: result.knowledge_evidence_used ?? null,
    gate: gate?.enabled && gate.sufficient !== undefined
      ? {
        maxRounds: gate.max_rounds ?? 0,
        roundsUsed: gate.rounds_used ?? 0,
        sufficient: gate.sufficient,
      }
      : null,
    grounding: grounding?.enabled && grounding.quotes_total !== undefined
      ? {
        total: grounding.quotes_total,
        verified: grounding.quotes_verified ?? 0,
      }
      : null,
    profileId: profile?.id ?? null,
    quotes: grounding?.quotes ?? [],
    references,
    refusal: isKnowledgeRefusal(result.answer, references.length),
  }
}

const CITATION_HREF_PREFIX = '#kref-'

/**
 * Make citation tokens clickable WITHOUT touching the off-limits Markdown
 * renderer: rewrite them into `[K1](#kref-K1)` links and let a capture-phase
 * click handler on the surrounding container intercept `#kref-*` anchors.
 *
 * Two forms are handled:
 *  - Bracketed `[K1]` (not already a link) — always linkified.
 *  - BARE `K1` (the model sometimes drops the brackets) — linkified ONLY when
 *    the token matches a real reference label (`knownLabels`), so a stray
 *    `K2` in prose (e.g. potassium) never becomes a dead citation link.
 *
 * Bare tokens must be delimited (not part of a word and not inside an
 * already-rewritten link). Adjacent runs without a separator (`K3K2`) are
 * split only when every segment is a known label.
 */
export function linkifyCitationLabels(
  markdown: string,
  knownLabels?: ReadonlySet<string>,
): string {
  let out = markdown.replace(/\[(K\d+)\](?!\()/g, (_match, label: string) => (
    `[${label}](${CITATION_HREF_PREFIX}${label})`
  ))
  if (knownLabels && knownLabels.size > 0) {
    out = out.replace(
      /(^|[^\w[\]()/#-])((?:K\d+){2,})\b/g,
      (match, pre: string, run: string) => {
        const labels = splitKnownCitationRun(run, knownLabels)
        return labels ? `${pre}${labels.map((label) => citationLink(label)).join('')}` : match
      },
    )
    out = out.replace(
      /(^|[^\w[\]()/#-])(K\d+)\b/g,
      (match, pre: string, label: string) => (
        knownLabels.has(label) ? `${pre}${citationLink(label)}` : match
      ),
    )
  }
  return out
}

function citationLink(label: string): string {
  return `[${label}](${CITATION_HREF_PREFIX}${label})`
}

function splitKnownCitationRun(
  run: string,
  knownLabels: ReadonlySet<string>,
): string[] | null {
  const memo = new Map<number, string[] | null>()
  const splitFrom = (offset: number): string[] | null => {
    if (offset === run.length) return []
    const cached = memo.get(offset)
    if (cached !== undefined) return cached
    const candidates = [...knownLabels]
      .filter((label) => run.startsWith(label, offset))
      .sort((a, b) => b.length - a.length)
    for (const label of candidates) {
      const rest = splitFrom(offset + label.length)
      if (rest) {
        const result = [label, ...rest]
        memo.set(offset, result)
        return result
      }
    }
    memo.set(offset, null)
    return null
  }
  const labels = splitFrom(0)
  return labels && labels.length > 1 ? labels : null
}

/** Extract the citation label from an intercepted `#kref-*` href. */
export function citationLabelFromHref(href: string | null | undefined): string | null {
  if (!href) return null
  const index = href.indexOf(CITATION_HREF_PREFIX)
  if (index === -1) return null
  const label = href.slice(index + CITATION_HREF_PREFIX.length)
  return /^K\d+$/.test(label) ? label : null
}
