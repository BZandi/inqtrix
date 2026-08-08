import { describe, expect, it } from 'vitest'

import { normalizeReportReferences } from './reportReferences'

describe('report reference normalization', () => {
  it('preserves compact web and Knowledge provenance without exposing arbitrary fields', () => {
    const [reference] = normalizeReportReferences(
      [{
        citation_id: 'citation-1',
        citation_ids: ['citation-1'],
        chunk_id: 'chunk-1',
        chunk_index: 3,
        document_id: 'document-1',
        generation_id: 'generation-1',
        ignored_provider_payload: { secret: true },
        label: 'K1',
        provenance_status: 'verified_span',
        provider_snippet: 'Provider metadata',
        query_id: 'query-1',
        query_ids: ['query-1'],
        reference_id: 'reference-1',
        revision_id: 'revision-1',
        source_id: 'source-1',
        source_run_id: 'run-1',
        source_run_ids: ['run-1'],
        source_span: {
          document_content_hash: 'sha256:document',
          end: 42,
          offset_unit: 'utf8_byte',
          start: 12,
        },
        tier: 'primary',
        title: 'Official source',
        url: 'https://example.test/source',
      }],
      '',
      [],
    )

    expect(reference).toEqual(expect.objectContaining({
      citation_id: 'citation-1',
      citation_ids: ['citation-1'],
      chunk_id: 'chunk-1',
      chunk_index: 3,
      document_id: 'document-1',
      generation_id: 'generation-1',
      provenance_status: 'verified_span',
      provider_snippet: 'Provider metadata',
      query_id: 'query-1',
      query_ids: ['query-1'],
      reference_id: 'reference-1',
      revision_id: 'revision-1',
      source_id: 'source-1',
      source_run_id: 'run-1',
      source_run_ids: ['run-1'],
      source_span: {
        document_content_hash: 'sha256:document',
        end: 42,
        offset_unit: 'utf8_byte',
        start: 12,
      },
    }))
    expect(reference).not.toHaveProperty('ignored_provider_payload')
  })
})
