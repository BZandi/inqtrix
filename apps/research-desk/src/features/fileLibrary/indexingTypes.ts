/**
 * Wire types for the background reindex-job API
 * (``/v1/knowledge/...reindex`` + ``/v1/knowledge/indexing-jobs``).
 *
 * Mirrors the research-run envelope (summary + SSE event) so the
 * indexing hook can reuse the same streaming/resume machinery. The
 * shapes match the server's ``build_indexing_job_summary`` and the
 * ``inqtrix.index.*`` event payloads byte-for-byte.
 */

export type IndexingJobStatus =
  | 'queued'
  | 'running'
  | 'cancelling'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'expired'

/** Live progress carried on every event and the summary. */
export type IndexingJobSnapshot = {
  completed_documents?: number
  current_document_title?: string
  /** 0..1, mirrors the research-run snapshot's progress reader. */
  progress_estimate?: number
  total_documents?: number
}

export type IndexingJobSummary = {
  collection_id: string
  collection_name: string
  completed_documents: number
  created_at: number
  elapsed_seconds: number | null
  embedding_model: string
  error: { message: string; type: string } | null
  events_url: string
  finished_at: number | null
  /** Optional caller correlation only. Authorization and UI ownership are
   * always derived from `collection_id`. */
  index_id: string | null
  job_id: string
  percent: number
  /** 1-based FIFO slot while still queued; null once running/terminal. */
  queue_position: number | null
  snapshot: IndexingJobSnapshot
  started_at: number | null
  status: IndexingJobStatus
  total_documents: number
  workspace_id: string | null
}

/** One SSE frame; ``type`` is ``inqtrix.index.{started,progress,...}``. */
export type IndexingJobEvent = {
  created_at: number
  data: Record<string, unknown>
  job_id: string
  sequence: number
  type: string
}

export const TERMINAL_INDEXING_STATUSES: ReadonlySet<IndexingJobStatus> =
  new Set(['completed', 'failed', 'cancelled', 'expired'])

export function isTerminalIndexingStatus(status: IndexingJobStatus): boolean {
  return TERMINAL_INDEXING_STATUSES.has(status)
}
