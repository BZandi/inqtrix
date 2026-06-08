import type { ReferenceDoc } from '@/features/files/referenceBlocks'
import type {
  AgentOverrides,
  CreateResearchRunRequest,
  InqtrixError,
  InqtrixHealth,
  InqtrixStackList,
  NodeModelResolution,
  ResearchRunMode,
  ResearchRunEvent,
  ResearchRunResult,
  ResearchRunSummary,
} from '@/features/researchRuns/types'

type ClientOptions = {
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

export async function streamResearchRunEvents(
  eventsUrl: string,
  options: StreamRunEventsOptions,
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
      options.onEvent(JSON.parse(data) as ResearchRunEvent)
    }
  }
}

type RequestJsonOptions = ClientOptions & {
  body?: unknown
  method?: 'GET' | 'POST'
}

async function requestJson<T>(path: string, options: RequestJsonOptions = {}) {
  const response = await request(path, options)
  return (await response.json()) as T
}

async function request(path: string, options: RequestJsonOptions = {}) {
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

  const response = await fetch(resolveUrl(path, options.baseUrl), {
    method: options.method ?? 'GET',
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  })

  if (!response.ok) {
    throw await requestError(response)
  }

  return response
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

function serializeChatCompletionRequest(
  request: ChatCompletionRequest,
  stream: boolean,
) {
  return {
    include_progress: request.includeProgress ?? false,
    messages: request.messages,
    mode: request.mode ?? 'direct_llm',
    model: request.model ?? CHAT_MODEL_NAME,
    stack: request.stack,
    stream,
    agent_overrides: serializeOverrides(request.agentOverrides),
  }
}
