import type { ReferenceDoc } from '@/features/files/referenceBlocks'
import type {
  IndexingJobEvent,
  IndexingJobSummary,
} from '@/features/fileLibrary/indexingTypes'
import type {
  PromptTemplateInfo,
  PromptTemplatePayload,
} from '@/features/promptLibrary/templateSync'
import type {
  QuotaAdminSnapshot,
  WorkspaceMembership,
} from '@/features/quota/admin'
import type { QuotaDimensionUsage } from '@/features/quota/model'
import type {
  ShareInvitee,
  ShareRecordInfo,
  SharedWithMeEntry,
  UserSearchResult,
} from '@/features/sharing/types'
import type {
  AgentOverrides,
  CreateResearchRunRequest,
  InqtrixCapabilities,
  InqtrixError,
  InqtrixHealth,
  InqtrixStackList,
  KnowledgeChatFilters,
  KnowledgeCollectionInfo,
  KnowledgeDocumentInfo,
  KnowledgeDocumentText,
  KnowledgeSearchHit,
  NodeModelResolution,
  ResearchRunMode,
  ResearchRunEvent,
  ResearchRunResult,
  ResearchRunSummary,
} from '@/features/researchRuns/types'

export type ClientOptions = {
  apiKey?: string
  baseUrl?: string
  signal?: AbortSignal
  workspaceId?: string
}

type StreamRunEventsOptions = ClientOptions & {
  onEvent: (event: ResearchRunEvent) => void
}

export type ChatCompletionMessage = {
  content: string
  role: 'assistant' | 'system' | 'user'
}

export type ChatCompletionRequest = {
  agentOverrides?: AgentOverrides
  includeProgress?: boolean
  /** Knowledge-retrieval scope; only meaningful with `mode: 'knowledge'`. */
  knowledgeFilters?: KnowledgeChatFilters
  messages: ChatCompletionMessage[]
  mode?: ResearchRunMode
  model?: string
  stack?: string
  stream?: boolean
}

export type ChatCompletionResponse = {
  choices: Array<{
    finish_reason?: string | null
    index: number
    message: {
      content: string
      role: 'assistant'
    }
  }>
  created: number
  id: string
  inqtrix?: {
    model_resolution?: NodeModelResolution
  }
  model: string
  object: 'chat.completion'
  usage?: {
    completion_tokens: number
    prompt_tokens: number
    total_tokens: number
  }
}

export type TextImprovementContext = 'chat_input' | 'prompt_template'

export type TextImprovementRequest = {
  context: TextImprovementContext
  guidance?: string
  locale: 'de' | 'en'
  stack?: string
  text: string
}

export type TextImprovementResponse = {
  change_summary: string[]
  clarification_questions: string[]
  improved_text: string
  needs_clarification: boolean
  warnings: string[]
}

type ChatCompletionChunk = {
  choices?: Array<{
    delta?: {
      content?: string
      role?: 'assistant'
    }
    finish_reason?: string | null
    index?: number
  }>
  inqtrix?: {
    model_resolution?: NodeModelResolution
  }
}

type StreamChatCompletionOptions = ClientOptions & {
  onDone?: () => void
  onDelta: (delta: string) => void
  onModelResolution?: (resolution: NodeModelResolution) => void
}

const DEFAULT_BASE_URL = import.meta.env.VITE_INQTRIX_API_BASE_URL ?? ''
const CHAT_MODEL_NAME = 'research-agent'

export type InqtrixRequestError = Error & {
  status?: number
}

export function hasHttpStatus(error: unknown, status: number) {
  return error instanceof Error && (error as InqtrixRequestError).status === status
}

export async function fetchHealth(options: ClientOptions = {}) {
  return requestJson<InqtrixHealth>('/health', options)
}

export async function fetchStacks(options: ClientOptions = {}) {
  return requestJson<InqtrixStackList>('/v1/stacks', options)
}

export async function fetchCapabilities(options: ClientOptions = {}) {
  return requestJson<InqtrixCapabilities>('/v1/capabilities', options)
}

export async function fetchQuotaUsage(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: QuotaDimensionUsage[] }>(
    '/v1/quota/usage',
    options,
  )
  return payload.data
}

export async function listWorkspaces(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: WorkspaceMembership[] }>(
    '/v1/workspaces',
    options,
  )
  return payload.data
}

// Quota administration is instance-admin power (tenant-wide), so it lives
// under /v1/admin/quota* and carries no workspace context — the server gates
// it on the session's instance_role.
export async function fetchQuotaAdmin(options: ClientOptions = {}) {
  return requestJson<QuotaAdminSnapshot>('/v1/admin/quota', options)
}

export async function setQuotaLimit(
  body: { subject_id: string; dimension: string; value: number },
  options: ClientOptions = {},
) {
  await request('/v1/admin/quota/limits', {
    ...options,
    body,
    method: 'PUT',
  })
}

export async function clearQuotaLimit(
  subjectId: string,
  dimension: string,
  options: ClientOptions = {},
) {
  const query = new URLSearchParams({ dimension, subject_id: subjectId })
  await request(`/v1/admin/quota/limits?${query}`, {
    ...options,
    method: 'DELETE',
  })
}

export async function resetQuota(
  body: { subject_id: string; dimension: string },
  options: ClientOptions = {},
) {
  await request('/v1/admin/quota/reset', {
    ...options,
    body,
    method: 'POST',
  })
}

export async function listKnowledgeCollections(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: KnowledgeCollectionInfo[] }>(
    '/v1/knowledge/collections',
    options,
  )
  return payload.data
}

export async function createKnowledgeCollection(
  request: { embeddingModel?: string; name: string },
  options: ClientOptions = {},
) {
  return requestJson<KnowledgeCollectionInfo>('/v1/knowledge/collections', {
    ...options,
    body: {
      embedding_model: request.embeddingModel,
      name: request.name,
    },
    method: 'POST',
  })
}

export async function deleteKnowledgeCollection(
  collectionId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/knowledge/collections/${collectionId}`, {
    ...options,
    method: 'DELETE',
  })
}

/** Delete a single knowledge document from its collection (Postgres + vectors).
 * Used when a member is removed from a vector index so it leaves the searchable
 * collection immediately, without a full rebuild. */
export async function deleteKnowledgeDocument(
  documentId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/knowledge/documents/${documentId}`, {
    ...options,
    method: 'DELETE',
  })
}

export type ServerFileInfo = {
  content_type: string
  created_at: number
  file_name: string
  id: string
  sha256: string
  size_bytes: number
  workspace_id: string | null
}

/**
 * Upload one original file to the server file store (`POST /v1/files`).
 * Multipart: the browser sets the boundary header itself, so this goes
 * through `fetch` directly instead of the JSON helper.
 */
export async function uploadServerFile(
  file: File,
  options: ClientOptions = {},
) {
  const headers = new Headers()
  if (options.apiKey) headers.set('Authorization', `Bearer ${options.apiKey}`)
  if (options.workspaceId) headers.set('X-Inqtrix-Workspace-Id', options.workspaceId)
  attachCsrfHeader(headers, 'POST')
  const body = new FormData()
  body.append('file', file, file.name)
  const response = await fetch(resolveUrl('/v1/files', options.baseUrl), {
    method: 'POST',
    headers,
    body,
    signal: options.signal,
    credentials: 'include',
  })
  if (!response.ok) throw await requestError(response)
  return (await response.json()) as ServerFileInfo
}

export type ServerFileText = {
  file_id: string
  parser_id: string | null
  text: string
}

/**
 * Server-side extracted text of an uploaded file (`GET /v1/files/{id}/text`),
 * via the backend parser ladder (MarkItDown by default). Called in the
 * BACKGROUND after upload so the stronger, browser-independent server parse
 * replaces the instant in-browser one (which can fail, e.g. pdf.js on Safari).
 * Throws on 501 (no server parser) or 422 (file not convertible).
 */
export async function fetchServerFileText(
  fileId: string,
  options: ClientOptions = {},
) {
  return requestJson<ServerFileText>(`/v1/files/${fileId}/text`, options)
}

/**
 * Ingest a registered server file into a knowledge collection — the
 * server fetches, parses (MarkItDown by default) and indexes it.
 */
export async function ingestKnowledgeFile(
  collectionId: string,
  payload: { fileId: string; metadata?: Record<string, unknown>; title?: string },
  options: ClientOptions = {},
) {
  return requestJson<KnowledgeDocumentInfo>(
    `/v1/knowledge/collections/${collectionId}/documents`,
    {
      ...options,
      body: {
        file_id: payload.fileId,
        metadata: payload.metadata,
        title: payload.title,
      },
      method: 'POST',
    },
  )
}

export async function addKnowledgeDocument(
  collectionId: string,
  document: { metadata?: Record<string, unknown>; text: string; title: string },
  options: ClientOptions = {},
) {
  return requestJson<KnowledgeDocumentInfo>(
    `/v1/knowledge/collections/${collectionId}/documents`,
    {
      ...options,
      body: document,
      method: 'POST',
    },
  )
}

/** Synchronous retrieval search (`POST /v1/knowledge/search`) — the
 * "Finden" mode of the knowledge workspace. */
export async function searchKnowledge(
  request: { query: string; collectionIds?: string[]; topK?: number },
  options: ClientOptions = {},
) {
  const payload = await requestJson<{ data: KnowledgeSearchHit[] }>(
    '/v1/knowledge/search',
    {
      ...options,
      body: {
        query: request.query,
        ...(request.collectionIds && request.collectionIds.length > 0
          ? { collection_ids: request.collectionIds }
          : {}),
        ...(request.topK ? { top_k: request.topK } : {}),
      },
      method: 'POST',
    },
  )
  return payload.data
}

/** Full extracted text of one knowledge document (document reader). */
export async function fetchKnowledgeDocumentText(
  documentId: string,
  options: ClientOptions = {},
) {
  return requestJson<KnowledgeDocumentText>(
    `/v1/knowledge/documents/${documentId}/text`,
    options,
  )
}

/**
 * Download one original server file as a Blob (`GET /v1/files/{id}/content`).
 * Returned as blob + content type so callers can build an object URL —
 * needed because an `<iframe src>` cannot carry Bearer auth headers.
 */
export async function fetchServerFileContent(
  fileId: string,
  options: ClientOptions = {},
) {
  const response = await request(`/v1/files/${fileId}/content`, options)
  const blob = await response.blob()
  return {
    blob,
    contentType: response.headers.get('content-type') ?? blob.type ?? '',
  }
}

/** All prompt templates visible to the caller, newest first. */
export async function listPromptTemplates(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: PromptTemplateInfo[] }>(
    '/v1/prompt-templates',
    options,
  )
  return payload.data
}

/** Create one prompt template on the server. */
export async function createPromptTemplate(
  payload: PromptTemplatePayload,
  options: ClientOptions = {},
) {
  return requestJson<PromptTemplateInfo>('/v1/prompt-templates', {
    ...options,
    body: payload,
    method: 'POST',
  })
}

/** Replace one template's writable fields (last write wins). */
export async function updatePromptTemplate(
  templateId: string,
  payload: PromptTemplatePayload,
  options: ClientOptions = {},
) {
  return requestJson<PromptTemplateInfo>(
    `/v1/prompt-templates/${templateId}`,
    { ...options, body: payload, method: 'PUT' },
  )
}

/** Delete one template (owner-only; revokes its shares server-side). */
export async function deletePromptTemplate(
  templateId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/prompt-templates/${templateId}`, {
    ...options,
    method: 'DELETE',
  })
}

// --- Chat-history persistence (M6a project tier) ---------------------------
//
// Thin wire wrappers over /v1/chat/threads(+/messages) and
// /v1/chat/thread-groups. They speak the SERVER shape verbatim: snake_case
// keys and unix-seconds FLOAT timestamps. The DTO<->ProjectState-record
// conversion (epoch<->ISO, message metadata pack/unpack, group membership)
// lives in features/chat/chatHistorySync.ts, the way fromRunSummary converts
// the run wire shape — the client stays a transport layer.

/** One chat thread as the server stores it (timestamps = unix seconds). */
export type ServerChatThread = {
  id: string
  title: string
  preview: string
  source: string
  group_id: string | null
  created_at: number
  updated_at: number
}

/** One chat message as the server stores it. ``metadata`` holds the verbatim
 * optional record fields (attachments / chainTrace / modelResolution). */
export type ServerChatMessage = {
  id: string
  thread_id: string
  role: string
  content_markdown: string
  metadata?: Record<string, unknown>
  created_at: number
}

/** One chat-thread group as the server stores it. */
export type ServerChatThreadGroup = {
  id: string
  title: string
  created_at: number
  updated_at: number
}

type PageOptions = ClientOptions & { cursor?: string; limit?: number }

function pageQuery(options: PageOptions): string {
  const params = new URLSearchParams()
  if (options.cursor) params.set('cursor', options.cursor)
  if (options.limit !== undefined) params.set('limit', String(options.limit))
  const query = params.toString()
  return query ? `?${query}` : ''
}

/** One keyset page of the caller's chat threads (newest first). */
export async function listChatThreads(options: PageOptions = {}) {
  return requestJson<{ data: ServerChatThread[]; next_cursor: string | null }>(
    `/v1/chat/threads${pageQuery(options)}`,
    options,
  )
}

/** Create or idempotently update one thread (the autosave upsert). */
export async function saveChatThread(
  threadId: string,
  payload: {
    title: string
    preview: string
    source: string
    group_id: string | null
    created_at: number
    updated_at: number
  },
  options: ClientOptions = {},
) {
  return requestJson<ServerChatThread>(`/v1/chat/threads/${threadId}`, {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

/** Delete one thread and its messages (owner-only; idempotent). */
export async function deleteChatThread(
  threadId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/chat/threads/${threadId}`, {
    ...options,
    method: 'DELETE',
  })
}

/** One keyset page of a thread's messages (newest first). */
export async function listChatMessages(
  threadId: string,
  options: PageOptions = {},
) {
  return requestJson<{ data: ServerChatMessage[]; next_cursor: string | null }>(
    `/v1/chat/threads/${threadId}/messages${pageQuery(options)}`,
    options,
  )
}

/** Append/idempotently upsert messages into a thread. */
export async function appendChatMessages(
  threadId: string,
  messages: Array<{
    id: string
    role: string
    content_markdown: string
    metadata?: Record<string, unknown>
    created_at: number
  }>,
  options: ClientOptions = {},
) {
  return requestJson<{ data: ServerChatMessage[] }>(
    `/v1/chat/threads/${threadId}/messages`,
    { ...options, body: { messages }, method: 'POST' },
  )
}

/** All of the caller's chat-thread groups (newest first). */
export async function listChatThreadGroups(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: ServerChatThreadGroup[] }>(
    '/v1/chat/thread-groups',
    options,
  )
  return payload.data
}

/** Create or idempotently update one thread group. */
export async function saveChatThreadGroup(
  groupId: string,
  payload: { title: string; created_at: number; updated_at: number },
  options: ClientOptions = {},
) {
  return requestJson<ServerChatThreadGroup>(
    `/v1/chat/thread-groups/${groupId}`,
    { ...options, body: payload, method: 'PUT' },
  )
}

/** Delete one thread group (its threads orphan to ungrouped). */
export async function deleteChatThreadGroup(
  groupId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/chat/thread-groups/${groupId}`, {
    ...options,
    method: 'DELETE',
  })
}

// --- Knowledge-session persistence ---------------------------------------

export type ServerKnowledgeSession = {
  id: string
  title: string
  group_id: string | null
  created_at: number
  updated_at: number
  items_json?: string | null
}

export type ServerKnowledgeSessionGroup = {
  id: string
  title: string
  created_at: number
  updated_at: number
}

export async function listKnowledgeSessions(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: ServerKnowledgeSession[] }>(
    '/v1/knowledge-sessions',
    options,
  )
  return payload.data
}

export async function getKnowledgeSession(
  sessionId: string,
  options: ClientOptions = {},
) {
  return requestJson<ServerKnowledgeSession>(
    `/v1/knowledge-sessions/${sessionId}`,
    options,
  )
}

export async function saveKnowledgeSession(
  sessionId: string,
  payload: {
    title: string
    items_json: string
    group_id: string | null
    created_at: number
    updated_at: number
  },
  options: ClientOptions = {},
) {
  return requestJson<ServerKnowledgeSession>(`/v1/knowledge-sessions/${sessionId}`, {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

export async function listKnowledgeSessionGroups(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: ServerKnowledgeSessionGroup[] }>(
    '/v1/knowledge-session-groups',
    options,
  )
  return payload.data
}

export async function saveKnowledgeSessionGroup(
  groupId: string,
  payload: {
    title: string
    created_at: number
    updated_at: number
  },
  options: ClientOptions = {},
) {
  return requestJson<ServerKnowledgeSessionGroup>(`/v1/knowledge-session-groups/${groupId}`, {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

export async function deleteKnowledgeSessionGroup(
  groupId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/knowledge-session-groups/${groupId}`, {
    ...options,
    method: 'DELETE',
  })
}

export async function deleteKnowledgeSession(
  sessionId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/knowledge-sessions/${sessionId}`, {
    ...options,
    method: 'DELETE',
  })
}

// --- Editor-history persistence (M6b project tier) -------------------------
//
// Thin wire wrappers over /v1/editor/documents(+/comments) and
// /v1/editor/folders. Same conventions as the chat client (snake_case,
// unix-seconds float timestamps); the DTO<->record conversion lives in
// features/editor/editorSync.ts. The document body is excluded from the
// list shape and present only on getEditorDocument (load-on-open).

/** One editor document as the server stores it. ``content_markdown`` is the
 * heavy body — present on getEditorDocument, ABSENT on the list. */
export type ServerEditorDocument = {
  id: string
  title: string
  content_markdown?: string
  folder_id: string | null
  source: string
  source_run_id: string | null
  revision: number
  diff_anchor_markdown: string | null
  diff_anchor_updated_at: number | null
  created_at: number
  updated_at: number
}

/** One editor folder as the server stores it. */
export type ServerEditorFolder = {
  id: string
  title: string
  created_at: number
  updated_at: number
}

/** One editor comment as the server stores it. */
export type ServerEditorComment = {
  id: string
  document_id: string
  comment_markdown: string
  anchor: Record<string, unknown>
  kind: string
  status: string
  evidence_preset: string | null
  created_at: number
  updated_at: number
}

/** One keyset page of the caller's documents (newest first, METADATA only). */
export async function listEditorDocuments(options: PageOptions = {}) {
  return requestJson<{ data: ServerEditorDocument[]; next_cursor: string | null }>(
    `/v1/editor/documents${pageQuery(options)}`,
    options,
  )
}

/** One document WITH its body (load-on-open). */
export async function getEditorDocument(
  documentId: string,
  options: ClientOptions = {},
) {
  return requestJson<ServerEditorDocument>(
    `/v1/editor/documents/${documentId}`,
    options,
  )
}

/** Create or idempotently update one document (autosave upsert, with body). */
export async function saveEditorDocument(
  documentId: string,
  payload: {
    title: string
    content_markdown: string
    folder_id: string | null
    source: string
    source_run_id: string | null
    revision: number
    diff_anchor_markdown: string | null
    diff_anchor_updated_at: number | null
    created_at: number
    updated_at: number
  },
  options: ClientOptions = {},
) {
  return requestJson<ServerEditorDocument>(`/v1/editor/documents/${documentId}`, {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

/** Delete one document and its comments (owner-only; idempotent). */
export async function deleteEditorDocument(
  documentId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/editor/documents/${documentId}`, {
    ...options,
    method: 'DELETE',
  })
}

/** One keyset page of a document's comments (newest first). */
export async function listEditorComments(
  documentId: string,
  options: PageOptions = {},
) {
  return requestJson<{ data: ServerEditorComment[]; next_cursor: string | null }>(
    `/v1/editor/documents/${documentId}/comments${pageQuery(options)}`,
    options,
  )
}

/** Upsert comments into a document. */
export async function saveEditorComments(
  documentId: string,
  comments: Array<{
    id: string
    comment_markdown: string
    anchor: Record<string, unknown>
    kind: string
    status: string
    evidence_preset: string | null
    created_at: number
    updated_at: number
  }>,
  options: ClientOptions = {},
) {
  return requestJson<{ data: ServerEditorComment[] }>(
    `/v1/editor/documents/${documentId}/comments`,
    { ...options, body: { comments }, method: 'POST' },
  )
}

/** Delete one comment from a document. */
export async function deleteEditorComment(
  documentId: string,
  commentId: string,
  options: ClientOptions = {},
) {
  await request(
    `/v1/editor/documents/${documentId}/comments/${commentId}`,
    { ...options, method: 'DELETE' },
  )
}

/** All of the caller's editor folders (newest first). */
export async function listEditorFolders(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: ServerEditorFolder[] }>(
    '/v1/editor/folders',
    options,
  )
  return payload.data
}

/** Create or idempotently update one editor folder. */
export async function saveEditorFolder(
  folderId: string,
  payload: { title: string; created_at: number; updated_at: number },
  options: ClientOptions = {},
) {
  return requestJson<ServerEditorFolder>(`/v1/editor/folders/${folderId}`, {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

/** Delete one editor folder (its documents orphan to ungrouped). */
export async function deleteEditorFolder(
  folderId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/editor/folders/${folderId}`, {
    ...options,
    method: 'DELETE',
  })
}

// -- file-asset records (M6c) ----------------------------------------------
//
// Thin wire wrappers over /v1/assets/sections, /v1/assets/groups and
// /v1/assets(+/{id}). Same conventions as the editor client (snake_case,
// unix-seconds float timestamps); the DTO<->record conversion lives in
// features/fileLibrary/assetSync.ts. The heavy extracted_text is excluded
// from the asset LIST and present only on getAsset (load-on-use).

/** One file-library section as the server stores it. */
export type ServerAssetSection = {
  id: string
  kind: string
  title: string
  created_at: number
  updated_at: number
}

/** One file-library group as the server stores it. */
export type ServerAssetGroup = {
  id: string
  section_id: string
  title: string
  created_at: number
  updated_at: number
}

/** One file-asset record as the server stores it. ``extracted_text`` is the
 * heavy body — present on getAsset, ABSENT on the list (metadata only). */
export type ServerAsset = {
  id: string
  section_id: string
  group_id: string | null
  title: string
  label: string
  file_name: string
  mime_type: string
  origin: string
  page_count: number | null
  parse_status: string
  parse_warning: string | null
  text_truncated: boolean
  size_bytes: number
  server_file_id: string | null
  // Optional on the wire: a server predating the provenance field omits it.
  parser_id?: string | null
  extracted_text?: string
  created_at: number
  updated_at: number
}

/** All of the caller's file-library sections. */
export async function listAssetSections(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: ServerAssetSection[] }>(
    '/v1/assets/sections',
    options,
  )
  return payload.data
}

/** Create or idempotently update one section. */
export async function saveAssetSection(
  sectionId: string,
  payload: { kind: string; title: string; created_at: number; updated_at: number },
  options: ClientOptions = {},
) {
  return requestJson<ServerAssetSection>(`/v1/assets/sections/${sectionId}`, {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

/** Delete one section (cascades its groups + assets server-side). */
export async function deleteAssetSection(sectionId: string, options: ClientOptions = {}) {
  await request(`/v1/assets/sections/${sectionId}`, { ...options, method: 'DELETE' })
}

/** All of the caller's file-library groups. */
export async function listAssetGroups(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: ServerAssetGroup[] }>(
    '/v1/assets/groups',
    options,
  )
  return payload.data
}

/** Create or idempotently update one group. */
export async function saveAssetGroup(
  groupId: string,
  payload: { section_id: string; title: string; created_at: number; updated_at: number },
  options: ClientOptions = {},
) {
  return requestJson<ServerAssetGroup>(`/v1/assets/groups/${groupId}`, {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

/** Delete one group (its assets orphan to ungrouped server-side). */
export async function deleteAssetGroup(groupId: string, options: ClientOptions = {}) {
  await request(`/v1/assets/groups/${groupId}`, { ...options, method: 'DELETE' })
}

/** One keyset page of the caller's assets (newest first, METADATA only). */
export async function listAssets(options: PageOptions = {}) {
  return requestJson<{ data: ServerAsset[]; next_cursor: string | null }>(
    `/v1/assets${pageQuery(options)}`,
    options,
  )
}

/** One asset WITH its extracted text (load-on-use). */
export async function getAsset(assetId: string, options: ClientOptions = {}) {
  return requestJson<ServerAsset>(`/v1/assets/${assetId}`, options)
}

/** Create or idempotently update one asset (with its extracted text). */
export async function saveAsset(
  assetId: string,
  payload: {
    section_id: string
    group_id: string | null
    title: string
    label: string
    file_name: string
    mime_type: string
    origin: string
    page_count: number | null
    parse_status: string
    parse_warning: string | null
    text_truncated: boolean
    size_bytes: number
    server_file_id: string | null
    parser_id: string | null
    extracted_text: string
    created_at: number
    updated_at: number
  },
  options: ClientOptions = {},
) {
  return requestJson<ServerAsset>(`/v1/assets/${assetId}`, {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

/** Delete one asset (owner-only; idempotent). */
export async function deleteAsset(assetId: string, options: ClientOptions = {}) {
  await request(`/v1/assets/${assetId}`, { ...options, method: 'DELETE' })
}

// -- vector-index records (M6c) ---------------------------------------------
//
// Thin wire wrappers over /v1/vector-indexes(+/{id}). Same conventions as the
// asset client (snake_case, unix-seconds float timestamps); the DTO<->record
// conversion lives in features/fileLibrary/vectorIndexSync.ts. A vector index
// has no heavy lazy body — the list returns FULL records (members + history).

/** One vector-index member (referenced document) as the server stores it. */
export type ServerVectorIndexMember = {
  file_id: string
  state: string
  /** Backend knowledge-document id this member was ingested as, once known;
   * absent/null for members ingested before this was tracked. Lets "remove
   * from index" delete the exact document without a full rebuild. */
  server_document_id?: string | null
}

/** One past reindex run as the server stores it (newest first). */
export type ServerVectorIndexHistoryEntry = {
  result: string
  documents: number
  duration_ms: number
  error: string | null
  started_at: number
  finished_at: number
}

/** One vector index as the server stores it (full record incl. children). */
export type ServerVectorIndex = {
  id: string
  title: string
  handle: string
  model: string
  dims: number
  status: string
  server_collection_id: string | null
  server_collection_model: string | null
  last_error: string | null
  members: ServerVectorIndexMember[]
  history: ServerVectorIndexHistoryEntry[]
  created_at: number
  updated_at: number
}

/** One keyset page of the caller's vector indexes (newest first, FULL records). */
export async function listVectorIndexes(options: PageOptions = {}) {
  return requestJson<{ data: ServerVectorIndex[]; next_cursor: string | null }>(
    `/v1/vector-indexes${pageQuery(options)}`,
    options,
  )
}

/** Create or idempotently update one vector index (replaces its members +
 * history wholesale server-side). */
export async function saveVectorIndex(
  indexId: string,
  payload: {
    title: string
    handle: string
    model: string
    dims: number
    status: string
    server_collection_id: string | null
    server_collection_model: string | null
    last_error: string | null
    members: ServerVectorIndexMember[]
    history: ServerVectorIndexHistoryEntry[]
    created_at: number
    updated_at: number
  },
  options: ClientOptions = {},
) {
  return requestJson<ServerVectorIndex>(`/v1/vector-indexes/${indexId}`, {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

/** Delete one vector index and its members + history (owner-only; idempotent). */
export async function deleteVectorIndex(indexId: string, options: ClientOptions = {}) {
  await request(`/v1/vector-indexes/${indexId}`, { ...options, method: 'DELETE' })
}

// -- account preferences (M6c) ----------------------------------------------
//
// A single per-user settings row (theme/locale/contrast). NOT project data and
// NOT part of the project import — it follows the user across devices. GET
// returns 404 when the user has never saved; getAccountPreferences maps that
// to null so the caller keeps its own default (the defaults are a frontend
// SSOT, never fabricated server-side).

/** Account preferences as the server stores them. */
export type ServerAccountPreferences = {
  contrast_mode: string
  locale: string
  theme: string
  theme_preset: string
  updated_at: number
}

/** The caller's account preferences, or null when never saved (HTTP 404). */
export async function getAccountPreferences(
  options: ClientOptions = {},
): Promise<ServerAccountPreferences | null> {
  try {
    return await requestJson<ServerAccountPreferences>('/v1/account/preferences', options)
  } catch (error) {
    if (hasHttpStatus(error, 404)) return null
    throw error
  }
}

/** Create or idempotently update the caller's account preferences. */
export async function saveAccountPreferences(
  payload: {
    contrast_mode: string
    locale: string
    theme: string
    theme_preset: string
    updated_at: number
  },
  options: ClientOptions = {},
) {
  return requestJson<ServerAccountPreferences>('/v1/account/preferences', {
    ...options,
    body: payload,
    method: 'PUT',
  })
}

/** Prefix typeahead over the tenant's user mirror (min 2 chars). */
export async function searchUsers(query: string, options: ClientOptions = {}) {
  const payload = await requestJson<{ data: UserSearchResult[] }>(
    `/v1/users/search?q=${encodeURIComponent(query)}`,
    options,
  )
  return payload.data
}

/** Active shares on one owned resource (profile-enriched rows). */
export async function listShares(
  resourceType: string,
  resourceId: string,
  options: ClientOptions = {},
) {
  const params = new URLSearchParams({
    resource_id: resourceId,
    resource_type: resourceType,
  })
  const payload = await requestJson<{ data: ShareRecordInfo[] }>(
    `/v1/shares?${params.toString()}`,
    options,
  )
  return payload.data
}

/** Grant shares to one or more users in a single batch. */
export async function createShares(
  resourceType: string,
  resourceId: string,
  invitees: ShareInvitee[],
  options: ClientOptions = {},
) {
  const payload = await requestJson<{ data: ShareRecordInfo[] }>('/v1/shares', {
    ...options,
    body: {
      invitees: invitees.map((invitee) => ({
        permission: invitee.permission,
        subject_id: invitee.subjectId,
      })),
      resource_id: resourceId,
      resource_type: resourceType,
    },
    method: 'POST',
  })
  return payload.data
}

/** Revoke one share by id (404s are indistinct from absence). */
export async function revokeShare(shareId: string, options: ClientOptions = {}) {
  await request(`/v1/shares/${shareId}`, { ...options, method: 'DELETE' })
}

/** Shared-in resources of one kind for the caller. */
export async function listSharedWithMe(
  resourceType: string,
  options: ClientOptions = {},
) {
  const payload = await requestJson<{ data: SharedWithMeEntry[] }>(
    `/v1/shares/shared-with-me?resource_type=${encodeURIComponent(resourceType)}`,
    options,
  )
  return payload.data
}

/** Active-share counts for the badge layer, keyed by resource id. */
export async function fetchOutgoingShareCounts(
  resourceType: string,
  resourceIds: readonly string[],
  options: ClientOptions = {},
) {
  if (resourceIds.length === 0) return {} as Record<string, number>
  const params = new URLSearchParams({ resource_type: resourceType })
  for (const id of resourceIds) params.append('resource_id', id)
  const payload = await requestJson<{ data: Record<string, number> }>(
    `/v1/shares/outgoing?${params.toString()}`,
    options,
  )
  return payload.data
}

export async function listResearchRuns(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: ResearchRunSummary[] }>(
    '/v1/runs',
    options,
  )
  return payload.data
}

export async function createResearchRun(
  request: CreateResearchRunRequest,
  options: ClientOptions = {},
) {
  return requestJson<ResearchRunSummary>('/v1/runs', {
    ...options,
    method: 'POST',
    body: {
      question: request.question,
      stack: request.stack,
      mode: request.mode,
      workspace_id: options.workspaceId,
      agent_overrides: serializeOverrides(request.agentOverrides),
      knowledge_filters: serializeKnowledgeFilters(request.knowledgeFilters),
    },
  })
}

export async function cancelResearchRun(
  runId: string,
  options: ClientOptions = {},
) {
  return requestJson<ResearchRunSummary>(`/v1/runs/${runId}/cancel`, {
    ...options,
    method: 'POST',
  })
}

/** Permanently delete one terminal run from the durable store (owner-only).
 * A run removed locally alone re-hydrates on the next list; this is what makes
 * deletion survive a reload. 409 when the run is still active (cancel first). */
export async function deleteResearchRun(
  runId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/runs/${runId}`, {
    ...options,
    method: 'DELETE',
  })
}

/** A completed report carried in from a loaded project file, pushed to the
 * durable run tier so it survives reload + follows the user. `result` is the
 * server result payload (answer/metrics/top_sources/top_claims/references/
 * usage) WITHOUT run_id/status (the result endpoint adds those on read). */
export type ImportResearchRunPayload = {
  run_id: string
  question: string
  stack: string
  mode?: string
  status?: string
  /** The report's ORIGINAL date (display/ordering); retention starts at import. */
  created_at?: number
  agent_overrides?: Record<string, unknown>
  snapshot?: Record<string, unknown>
  result?: Record<string, unknown>
}

export async function importResearchRun(
  payload: ImportResearchRunPayload,
  options: ClientOptions = {},
) {
  return requestJson<ResearchRunSummary>('/v1/runs/import', {
    ...options,
    method: 'POST',
    body: { ...payload, workspace_id: options.workspaceId },
  })
}

export async function fetchResearchRunResult(
  runId: string,
  options: ClientOptions = {},
) {
  return requestJson<ResearchRunResult>(`/v1/runs/${runId}/result`, options)
}

export async function createChatCompletion(
  chatRequest: ChatCompletionRequest,
  options: ClientOptions = {},
) {
  return requestJson<ChatCompletionResponse>('/v1/chat/completions', {
    ...options,
    method: 'POST',
    body: serializeChatCompletionRequest(chatRequest, false),
  })
}

export async function improveText(
  textRequest: TextImprovementRequest,
  options: ClientOptions = {},
) {
  return requestJson<TextImprovementResponse>('/v1/text/improvements', {
    ...options,
    method: 'POST',
    body: {
      context: textRequest.context,
      guidance: textRequest.guidance,
      locale: textRequest.locale,
      stack: textRequest.stack,
      text: textRequest.text,
      workspace_id: options.workspaceId,
    },
  })
}

/**
 * Map attached reference documents to the additive snake_case `attachments`
 * wire field shared by both editor endpoints. Returns `undefined` for no
 * attachments so the request body stays byte-identical to the pre-attachment
 * shape (the backend treats a missing field as "no attachments").
 */
function attachmentsPayload(docs: readonly ReferenceDoc[] | undefined) {
  if (!docs || docs.length === 0) return undefined
  return docs.map((doc) => ({
    content: doc.content,
    label: doc.label,
    page_count: doc.pageCount ?? null,
    size_bytes: doc.sizeBytes ?? null,
  }))
}

export type EditorSuggestRequest = {
  attachments?: ReferenceDoc[]
  background?: string
  blockMarkdown?: string
  blockText: string
  currentSuggestionMarkdown?: string
  refinementInstruction?: string
  globalInstruction?: string
  instruction?: string
  locale: 'de' | 'en'
  modelTier?: 'high' | 'mid' | 'fast' | null
  model?: string | null
  effort?: string | null
  snippet?: string
  stack?: string
}

export type EditorSuggestResponse = {
  change_summary: string[]
  improved_text: string
  warnings: string[]
}

export async function suggestEditorBlock(
  suggestRequest: EditorSuggestRequest,
  options: ClientOptions = {},
) {
  return requestJson<EditorSuggestResponse>('/v1/editor/suggest', {
    ...options,
    method: 'POST',
    body: {
      agent_overrides: editorAgentOverrides(suggestRequest),
      attachments: attachmentsPayload(suggestRequest.attachments),
      background: suggestRequest.background,
      block_markdown: suggestRequest.blockMarkdown,
      block_text: suggestRequest.blockText,
      current_suggestion_markdown: suggestRequest.currentSuggestionMarkdown,
      global_instruction: suggestRequest.globalInstruction,
      instruction: suggestRequest.instruction,
      locale: suggestRequest.locale,
      refinement_instruction: suggestRequest.refinementInstruction,
      snippet: suggestRequest.snippet,
      stack: suggestRequest.stack,
      workspace_id: options.workspaceId,
    },
  })
}

export type EditorInstructEdit = {
  find: string
  note: string
  position: 'replace' | 'before' | 'after' | 'append'
  quote_after: string
  quote_before: string
  text: string
}

export type EditorInstructRequest = {
  attachments?: ReferenceDoc[]
  documentMarkdown: string
  instruction: string
  locale: 'de' | 'en'
  modelTier?: 'high' | 'mid' | 'fast' | null
  model?: string | null
  effort?: string | null
  stack?: string
}

export type EditorInstructResponse = {
  assistant_message: string
  edits: EditorInstructEdit[]
  warnings: string[]
}

export async function instructEditorDocument(
  instructRequest: EditorInstructRequest,
  options: ClientOptions = {},
) {
  return requestJson<EditorInstructResponse>('/v1/editor/instruct', {
    ...options,
    method: 'POST',
    body: {
      agent_overrides: editorAgentOverrides(instructRequest),
      attachments: attachmentsPayload(instructRequest.attachments),
      document_markdown: instructRequest.documentMarkdown,
      instruction: instructRequest.instruction,
      locale: instructRequest.locale,
      stack: instructRequest.stack,
      workspace_id: options.workspaceId,
    },
  })
}

export async function streamChatCompletion(
  chatRequest: ChatCompletionRequest,
  options: StreamChatCompletionOptions,
) {
  const response = await request('/v1/chat/completions', {
    ...options,
    method: 'POST',
    body: serializeChatCompletionRequest(chatRequest, true),
  })
  if (!response.body) {
    throw new Error('Inqtrix chat stream did not return a response body.')
  }
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('text/event-stream')) {
    throw new Error('Inqtrix chat streaming is not available for this response.')
  }

  const reader = response.body
    .pipeThrough(new TextDecoderStream())
    .getReader()
  let buffer = ''
  let doneEmitted = false

  function emitDone() {
    if (doneEmitted) return
    doneEmitted = true
    options.onDone?.()
  }

  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const data = parseSseData(frame)
      if (!data) continue
      if (data === '[DONE]') {
        emitDone()
        return
      }

      const chunk = JSON.parse(data) as ChatCompletionChunk
      const modelResolution = chunk.inqtrix?.model_resolution
      if (modelResolution) options.onModelResolution?.(modelResolution)
      const choice = chunk.choices?.[0]
      const delta = choice?.delta?.content
      if (delta) options.onDelta(delta)
      if (choice?.finish_reason) emitDone()
    }
  }

  emitDone()
}

type StreamEventsOptions<T> = ClientOptions & {
  onEvent: (event: T) => void
}

/**
 * Generic Server-Sent-Events reader: open *eventsUrl*, split on the
 * SSE frame boundary, parse each frame's data line, and hand the parsed
 * object to ``onEvent``. Both the research-run and the reindex-job
 * streams are thin typed wrappers over this (Designprinzip 4).
 */
export async function streamServerSentEvents<T>(
  eventsUrl: string,
  options: StreamEventsOptions<T>,
) {
  const response = await request(eventsUrl, options)
  if (!response.body) {
    throw new Error('Inqtrix event stream did not return a response body.')
  }

  const reader = response.body
    .pipeThrough(new TextDecoderStream())
    .getReader()
  let buffer = ''

  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const data = parseSseData(frame)
      if (!data) continue
      options.onEvent(JSON.parse(data) as T)
    }
  }
}

export async function streamResearchRunEvents(
  eventsUrl: string,
  options: StreamRunEventsOptions,
) {
  return streamServerSentEvents<ResearchRunEvent>(eventsUrl, options)
}

export async function startIndexingJob(
  collectionId: string,
  request: { indexId?: string },
  options: ClientOptions = {},
) {
  return requestJson<IndexingJobSummary>(
    `/v1/knowledge/collections/${collectionId}/reindex`,
    {
      ...options,
      method: 'POST',
      body: { index_id: request.indexId, workspace_id: options.workspaceId },
    },
  )
}

export async function cancelIndexingJob(
  jobId: string,
  options: ClientOptions = {},
) {
  return requestJson<IndexingJobSummary>(
    `/v1/knowledge/indexing-jobs/${jobId}/cancel`,
    { ...options, method: 'POST' },
  )
}

export async function listIndexingJobs(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: IndexingJobSummary[] }>(
    '/v1/knowledge/indexing-jobs',
    options,
  )
  return payload.data
}

export async function streamIndexingJobEvents(
  eventsUrl: string,
  options: StreamEventsOptions<IndexingJobEvent>,
) {
  return streamServerSentEvents<IndexingJobEvent>(eventsUrl, options)
}

type RequestJsonOptions = ClientOptions & {
  body?: unknown
  method?: 'DELETE' | 'GET' | 'PATCH' | 'POST' | 'PUT'
}

async function requestJson<T>(path: string, options: RequestJsonOptions = {}) {
  const response = await request(path, options)
  return (await response.json()) as T
}

async function request(path: string, options: RequestJsonOptions = {}) {
  const method = options.method ?? 'GET'
  const headers = new Headers()
  if (options.body !== undefined) {
    headers.set('Content-Type', 'application/json')
  }
  if (options.apiKey) {
    headers.set('Authorization', `Bearer ${options.apiKey}`)
  }
  if (options.workspaceId) {
    headers.set('X-Inqtrix-Workspace-Id', options.workspaceId)
  }
  attachCsrfHeader(headers, method)

  const response = await fetch(resolveUrl(path, options.baseUrl), {
    method,
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
    credentials: 'include',
  })

  if (!response.ok) {
    throw await requestError(response)
  }

  return response
}

/**
 * Attach the OIDC double-submit CSRF token on unsafe methods. The
 * token lives in a non-HttpOnly cookie BY DESIGN (OWASP signed
 * double-submit): reading it here means no call site has to thread a
 * token around — sessions stay entirely cookie-driven. No cookie
 * means no OIDC session (apikey/none modes), and nothing is sent.
 */
function attachCsrfHeader(headers: Headers, method: string) {
  if (method === 'GET' || method === 'HEAD') return
  const token = readCsrfCookie()
  if (token) headers.set('X-CSRF-Token', token)
}

function readCsrfCookie(): string | null {
  if (typeof document === 'undefined') return null
  for (const name of ['__Host-inqtrix_csrf', 'inqtrix_csrf']) {
    const match = document.cookie
      .split('; ')
      .find((entry) => entry.startsWith(`${name}=`))
    if (match) return decodeURIComponent(match.slice(name.length + 1))
  }
  return null
}

/** Session facts the SPA bootstraps from `GET /api/auth/session`. */
export type AuthSessionInfo = {
  authenticated: boolean
  csrf_token?: string
  display_name?: string | null
  email?: string | null
  /** Instance role driving the admin-UI gate; absent reads as `user`. */
  role?: string
  sub?: string
  /** The user's canonical project namespace (a `ws_...` string), resolved
   * server-side: on the first authenticated boot the server ADOPTS the browser
   * namespace carried in the `X-Inqtrix-Workspace-Id` request header and returns
   * it here; thereafter it returns the same adopted value on every device, so the
   * project follows the user. `null`/absent until a namespace has been adopted
   * (or in non-cookie modes), in which case the desk keeps the browser-local id. */
  project_namespace?: string | null
}

/**
 * Fetch the current OIDC session state (cookie-driven, never 401s).
 *
 * Pass `options.workspaceId` (the browser's namespace) so the probe carries it as
 * the adoption candidate: a first authenticated boot adopts it as the user's
 * canonical `project_namespace`; later boots ignore it and return the already-
 * adopted value.
 */
export async function fetchAuthSession(options: ClientOptions = {}) {
  return requestJson<AuthSessionInfo>('/api/auth/session', options)
}

/** Destroy the server-side OIDC session (CSRF token read from cookie). */
export async function logoutSession(options: ClientOptions = {}) {
  return requestJson<{ logged_out: boolean }>('/api/auth/logout', {
    ...options,
    body: {},
    method: 'POST',
  })
}

/**
 * Full-page navigation target that starts the OIDC login redirect.
 * A plain link/location change on purpose — the authorization flow is
 * a top-level browser navigation, never an XHR.
 */
export function buildLoginUrl(baseUrl?: string, next: string = '/') {
  const query = next && next !== '/' ? `?next=${encodeURIComponent(next)}` : ''
  return resolveUrl(`/api/auth/login${query}`, baseUrl)
}

/** One login method the SPA can render (an ordered list per deployment). */
export type AuthLoginMethod = {
  kind: 'sso' | 'password' | 'apikey'
  label: string
  image_url?: string
  identifier?: 'email' | 'username'
}

/**
 * Pre-login auth capabilities from `GET /api/auth/config` — unauthenticated
 * and always mounted, so the SPA can label the SSO button with the real
 * provider name and learn the session contract before authenticating.
 */
export type AuthConfig = {
  auth_mode: 'none' | 'apikey' | 'oidc' | 'local' | 'ldap'
  auth_required: boolean
  login_methods: AuthLoginMethod[]
  provider_name: string | null
  registration: { self_service: boolean; needs_owner: boolean }
  pat_available: boolean
  supports_logout: boolean
  csrf_required: boolean
  csrf_header: string
}

/** Pre-login discovery (presentation hints only; degrade open on failure). */
export async function fetchAuthConfig(options: ClientOptions = {}) {
  return requestJson<AuthConfig>('/api/auth/config', options)
}

// --- Native local / LDAP auth + first-run owner setup --------------------
// All cookie-driven (the BFF session machinery, ADR-AUTH-3): no token is
// threaded by the caller and CSRF rides the cookie on unsafe methods.

/** Create the first owner exactly once; the server logs them straight in. */
export async function createOwner(
  input: { email: string; password: string; displayName?: string },
  options: ClientOptions = {},
) {
  return requestJson<{ authenticated: boolean }>('/api/setup/owner', {
    ...options,
    method: 'POST',
    body: {
      email: input.email,
      password: input.password,
      display_name: input.displayName,
    },
  })
}

/** Email/password login → the same session+CSRF cookies as OIDC. */
export async function loginLocal(
  input: { identifier: string; password: string },
  options: ClientOptions = {},
) {
  return requestJson<{ authenticated: boolean }>('/api/auth/login/local', {
    ...options,
    method: 'POST',
    body: { identifier: input.identifier, password: input.password },
  })
}

/** LDAP bind login → the same session+CSRF cookies as OIDC. */
export async function loginLdap(
  input: { identifier: string; password: string },
  options: ClientOptions = {},
) {
  return requestJson<{ authenticated: boolean }>('/api/auth/login/ldap', {
    ...options,
    method: 'POST',
    body: { identifier: input.identifier, password: input.password },
  })
}

/** Self-service password change (local accounts; re-verifies the current). */
export async function changePassword(
  input: { currentPassword: string; newPassword: string },
  options: ClientOptions = {},
) {
  return requestJson<{ changed: boolean }>('/api/auth/password', {
    ...options,
    method: 'POST',
    body: {
      current_password: input.currentPassword,
      new_password: input.newPassword,
    },
  })
}

// --- Instance administration (/v1/admin/*) -------------------------------
// Session-only + admin-gated server-side; denial is an indistinguishable
// 404. A PAT can never administer users.

export type AdminSystemRuntime = {
  api: {
    openapi: boolean
  }
  files: {
    blob_storage: string
    enabled: boolean
    max_file_bytes: number | null
    object_store: string
    object_store_available: boolean
  }
  knowledge: {
    contextual_retrieval: boolean
    default_top_k: number | null
    document_parser: string
    embedding_model: string | null
    embedding_provider: string | null
    enabled: boolean
    hybrid_retrieval: boolean
    reranker: string
    sparse: string | null
    vector_store: string
    vector_store_available: boolean
  }
  runs: {
    execution: string
    queue: string
    queue_available: boolean
    store: string
    worker_dispatch: boolean
  }
  storage: {
    backend: string
    durable: boolean
  }
}

/** Sanitized runtime categories for the instance-admin System panel. */
export async function fetchAdminSystemRuntime(options: ClientOptions = {}) {
  return requestJson<AdminSystemRuntime>('/v1/admin/system/runtime', options)
}

/** One row of the instance user list. */
export type AdminUser = {
  subject: string
  email: string | null
  display_name: string | null
  instance_role: 'admin' | 'user'
  disabled: boolean
  last_login_at: number | null
}

/** Every mirrored identity in the tenant (admin listing). */
export async function listAdminUsers(options: ClientOptions = {}) {
  return requestJson<{ users: AdminUser[] }>('/v1/admin/users', options)
}

/** Change a user's instance role (last-admin demotion is refused, 409). */
export async function setAdminUserRole(
  subject: string,
  instanceRole: 'admin' | 'user',
  options: ClientOptions = {},
) {
  return requestJson<AdminUser>(
    `/v1/admin/users/${encodeURIComponent(subject)}`,
    { ...options, method: 'PATCH', body: { instance_role: instanceRole } },
  )
}

/** Disable or enable a user (disable cascades sessions + tokens). */
export async function setAdminUserDisabled(
  subject: string,
  disabled: boolean,
  options: ClientOptions = {},
) {
  const action = disabled ? 'disable' : 'enable'
  return requestJson<AdminUser>(
    `/v1/admin/users/${encodeURIComponent(subject)}:${action}`,
    { ...options, method: 'POST', body: {} },
  )
}

/** Create a local account (local mode only); the admin sets the password. */
export async function createAdminUser(
  input: {
    email: string
    password: string
    instanceRole?: 'admin' | 'user'
    displayName?: string
  },
  options: ClientOptions = {},
) {
  return requestJson<AdminUser>('/v1/admin/users', {
    ...options,
    method: 'POST',
    body: {
      email: input.email,
      password: input.password,
      instance_role: input.instanceRole,
      display_name: input.displayName,
    },
  })
}

/** Admin password reset for a local account (forgotten-password recovery). */
export async function resetUserPassword(
  subject: string,
  password: string,
  options: ClientOptions = {},
) {
  return requestJson<{ reset: boolean }>(
    `/v1/admin/users/${encodeURIComponent(subject)}:reset-password`,
    { ...options, method: 'POST', body: { password } },
  )
}

// --- Admin workspace management (/v1/admin/workspaces) --------------------
// Instance-admin surface: create workspaces and position users into them.
// The collaboration role (viewer..owner) lives entirely inside a workspace.

export type WorkspaceRoleValue = 'viewer' | 'commenter' | 'editor' | 'owner'

/** One workspace row in the admin overview. */
export type AdminWorkspace = {
  workspace_id: string
  name: string
  created_by_sub: string
  member_count: number
}

/** One member row of a workspace (enriched with the mirror profile). */
export type WorkspaceMember = {
  sub: string
  role: WorkspaceRoleValue
  display_name: string | null
  email: string | null
}

/** Every workspace in the tenant with its member count. */
export async function listAdminWorkspaces(options: ClientOptions = {}) {
  const payload = await requestJson<{ data: AdminWorkspace[] }>(
    '/v1/admin/workspaces',
    options,
  )
  return payload.data
}

/** Create a workspace; the calling admin becomes its OWNER. */
export async function createAdminWorkspace(
  name: string,
  options: ClientOptions = {},
) {
  return requestJson<{ workspace_id: string; name: string }>(
    '/v1/admin/workspaces',
    { ...options, method: 'POST', body: { name } },
  )
}

/** Rename a workspace. */
export async function renameAdminWorkspace(
  workspaceId: string,
  name: string,
  options: ClientOptions = {},
) {
  return requestJson<{ workspace_id: string; name: string }>(
    `/v1/admin/workspaces/${encodeURIComponent(workspaceId)}`,
    { ...options, method: 'PATCH', body: { name } },
  )
}

/** Delete a workspace and cascade its memberships. */
export async function deleteAdminWorkspace(
  workspaceId: string,
  options: ClientOptions = {},
) {
  await request(`/v1/admin/workspaces/${encodeURIComponent(workspaceId)}`, {
    ...options,
    method: 'DELETE',
  })
}

/** Members of one workspace. */
export async function listWorkspaceMembers(
  workspaceId: string,
  options: ClientOptions = {},
) {
  const payload = await requestJson<{ data: WorkspaceMember[] }>(
    `/v1/admin/workspaces/${encodeURIComponent(workspaceId)}/members`,
    options,
  )
  return payload.data
}

/** Assign a user to a workspace at a role (adds them as a member). */
export async function addWorkspaceMember(
  workspaceId: string,
  sub: string,
  role: WorkspaceRoleValue,
  options: ClientOptions = {},
) {
  return requestJson<{ sub: string; role: WorkspaceRoleValue }>(
    `/v1/admin/workspaces/${encodeURIComponent(workspaceId)}/members`,
    { ...options, method: 'POST', body: { sub, role } },
  )
}

/** Change an existing member's role (last-owner demotion is refused, 409). */
export async function setWorkspaceMemberRole(
  workspaceId: string,
  sub: string,
  role: WorkspaceRoleValue,
  options: ClientOptions = {},
) {
  return requestJson<{ sub: string; role: WorkspaceRoleValue }>(
    `/v1/admin/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(sub)}`,
    { ...options, method: 'PATCH', body: { role } },
  )
}

/** Remove a member (last-owner removal is refused, 409). */
export async function removeWorkspaceMember(
  workspaceId: string,
  sub: string,
  options: ClientOptions = {},
) {
  await request(
    `/v1/admin/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(sub)}`,
    { ...options, method: 'DELETE' },
  )
}

// --- Personal access tokens (/api/auth/tokens) ---------------------------
// Session-only management; the plaintext secret is emitted exactly once at
// creation and never retrievable again.

/** One PAT row (never carries the hash or the plaintext). */
export type AccessToken = {
  token_id: string
  name: string
  created_at: number
  expires_at: number | null
  last_used_at: number | null
  scopes: string[]
}

/** The current session owner's tokens. */
export async function listAccessTokens(options: ClientOptions = {}) {
  return requestJson<{ tokens: AccessToken[] }>('/api/auth/tokens', options)
}

/** Mint a token; the response's one-time `token` field is the plaintext. */
export async function createAccessToken(
  input: { name: string; expiresInDays?: number },
  options: ClientOptions = {},
) {
  return requestJson<AccessToken & { token: string }>('/api/auth/tokens', {
    ...options,
    method: 'POST',
    body: { name: input.name, expires_in_days: input.expiresInDays },
  })
}

/** Revoke a token (foreign/unknown ids are an indistinguishable 404). */
export async function revokeAccessToken(
  tokenId: string,
  options: ClientOptions = {},
) {
  return requestJson<{ revoked: boolean }>(
    `/api/auth/tokens/${encodeURIComponent(tokenId)}`,
    { ...options, method: 'DELETE' },
  )
}

async function requestError(response: Response) {
  const fallbackMessage = `Inqtrix request failed with HTTP ${response.status}.`
  try {
    const payload = await response.json() as { error?: InqtrixError }
    const error = payload.error
    if (error?.message) {
      const enrichedError = new Error(error.message) as InqtrixRequestError
      enrichedError.name = error.type || 'InqtrixRequestError'
      enrichedError.status = response.status
      return enrichedError
    }
  } catch {
    // The server may return an empty body for infrastructure failures.
  }
  const fallbackError = new Error(fallbackMessage) as InqtrixRequestError
  fallbackError.status = response.status
  return fallbackError
}

function resolveUrl(path: string, baseUrl = DEFAULT_BASE_URL) {
  if (/^https?:\/\//.test(path)) return path
  const normalizedBase = baseUrl.replace(/\/$/, '')
  return `${normalizedBase}${path}`
}

function parseSseData(frame: string) {
  const dataLines = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
  return dataLines.length > 0 ? dataLines.join('\n') : null
}

function serializeOverrides(overrides?: AgentOverrides) {
  if (!overrides) return undefined
  return {
    max_rounds: overrides.maxRounds,
    min_rounds: overrides.minRounds,
    confidence_stop: overrides.confidenceStop,
    report_profile: overrides.reportProfile,
    max_total_seconds: overrides.maxTotalSeconds,
    first_round_queries: overrides.firstRoundQueries,
    skip_search: overrides.skipSearch,
    model_tier: overrides.modelTier,
    model: overrides.model,
    effort: overrides.effort,
  }
}

/** Build the editor ``agent_overrides`` slice from picker selections.
 *  An explicit model/effort (UI picker) takes precedence; modelTier remains
 *  the fallback. Returns undefined when nothing is selected. */
function editorAgentOverrides(req: {
  modelTier?: 'high' | 'mid' | 'fast' | null
  model?: string | null
  effort?: string | null
}) {
  const overrides: Record<string, string> = {}
  if (req.model) overrides.model = req.model
  if (req.effort) overrides.effort = req.effort
  if (req.modelTier) overrides.model_tier = req.modelTier
  return Object.keys(overrides).length > 0 ? overrides : undefined
}

function serializeKnowledgeFilters(filters?: KnowledgeChatFilters) {
  if (!filters || filters.collectionIds.length === 0) return undefined
  return {
    collection_ids: filters.collectionIds,
    ...(filters.topK ? { top_k: filters.topK } : {}),
    ...(filters.profile ? { profile: filters.profile } : {}),
  }
}

function serializeChatCompletionRequest(
  request: ChatCompletionRequest,
  stream: boolean,
) {
  return {
    include_progress: request.includeProgress ?? false,
    knowledge_filters: serializeKnowledgeFilters(request.knowledgeFilters),
    messages: request.messages,
    mode: request.mode ?? 'direct_llm',
    model: request.model ?? CHAT_MODEL_NAME,
    stack: request.stack,
    stream,
    agent_overrides: serializeOverrides(request.agentOverrides),
  }
}
