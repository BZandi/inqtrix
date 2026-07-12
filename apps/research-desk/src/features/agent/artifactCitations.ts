import {
  citationLabelFromHref,
  linkifyCitationLabels,
} from '@/components/markdown/citationLinks'
import type { KnowledgeReferenceRecord } from '@/features/project/types'

export type AgentArtifactReference = {
  chunkIndex: number | null
  documentId: string | null
  excerpt: string | null
  /** Provider-grounded statement used to support the synthesized claim. It is
   * not presented as a verbatim source excerpt. */
  groundedSupport: string | null
  label: string
  pageNumber: number | null
  title: string
  url: string | null
}

const isAgentCitationLabel = (label: string): boolean => /^[KW]\d+$/.test(label)

/** Normalize trusted artifact refs into one UI shape. Older rows without an
 * excerpt remain useful: title, URL and exact RAG identity still survive. */
export function agentArtifactReferences(
  refs: readonly Record<string, unknown>[] | undefined,
): AgentArtifactReference[] {
  if (!refs) return []
  return refs.flatMap((ref) => {
    const label = stringField(ref, 'label')
    if (!label || !isAgentCitationLabel(label)) return []
    const documentId = stringField(ref, 'document_id') || null
    const url = stringField(ref, 'url') || null
    if (!documentId && !url) return []
    const excerpt = firstString(ref, ['excerpt', 'source_text', 'snippet'])
    const groundedSupport = stringField(ref, 'grounded_support') || null
    const title = stringField(ref, 'title') || url || documentId || label
    return [{
      chunkIndex: numberField(ref, 'chunk_index'),
      documentId,
      excerpt: excerpt || null,
      groundedSupport,
      label,
      pageNumber: numberField(ref, 'page_number'),
      title,
      url,
    }]
  })
}

export function linkifyAgentArtifactCitations(
  markdown: string,
  references: readonly AgentArtifactReference[],
): string {
  return linkifyCitationLabels(
    markdown,
    isAgentCitationLabel,
    new Set(references.map((reference) => reference.label)),
    { requireKnownBracketed: true },
  )
}

export function agentCitationLabelFromHref(
  href: string | null | undefined,
): string | null {
  return citationLabelFromHref(href, isAgentCitationLabel)
}

/** Adapter for the established Knowledge source-row presentation. Web refs use
 * their real URL; internal refs keep a synthetic URL only as stable identity. */
export function agentReferenceAsKnowledge(
  reference: AgentArtifactReference,
): KnowledgeReferenceRecord {
  return {
    chunkIndex: reference.chunkIndex,
    documentId: reference.documentId,
    excerpt: reference.excerpt,
    label: reference.label,
    pageNumber: reference.pageNumber,
    sourceText: reference.excerpt,
    tier: 'unknown',
    title: reference.title,
    url: reference.url
      ?? `inqtrix://documents/${reference.documentId ?? reference.label}`,
  }
}

function firstString(
  record: Record<string, unknown>,
  keys: readonly string[],
): string {
  for (const key of keys) {
    const value = stringField(record, key)
    if (value) return value
  }
  return ''
}

function stringField(record: Record<string, unknown>, key: string): string {
  const value = record[key]
  return typeof value === 'string' ? value.trim() : ''
}

function numberField(
  record: Record<string, unknown>,
  key: string,
): number | null {
  const value = record[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}
