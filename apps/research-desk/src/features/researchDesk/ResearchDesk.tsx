import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react'
import { useReducedMotion } from 'motion/react'
import {
  createChatCompletion,
  hasHttpStatus,
  listResearchRuns,
  streamChatCompletion,
  type ChatCompletionMessage,
} from '@/api/inqtrixClient'
import { AuthLockScreen } from './components/AuthLockScreen'
import ChatWorkspace from '@/features/chat/ChatWorkspace'
import EditorWorkspace from '@/features/editor/EditorWorkspace'
import { exportProject, loadProject, saveProject } from '@/features/project/fileSystem'
import {
  chatAttachmentsFromRefs,
  projectChatHistorySections,
  chatRuleOptionsFromRules,
  chatAttachmentChipsFromRefs,
  chatContextRefKey,
  completedReportOptions,
  dedupeChatContextRefs,
  fileGroupMentionOptions,
  fileMentionOptions,
  projectChatThreads,
  projectChatRules,
  projectFileAssets,
  projectResearchJobs,
  selectedResearchRun,
} from '@/features/project/selectors'
import { chatFunctionChainTemplatesFromRefs } from '@/features/project/chatRules'
import type {
  ChatChainStepRecord,
  ChatContextReferenceRecord,
  ChatMessageAttachmentRecord,
  ChatMessageModelResolutionRecord,
  ChatMessageRecord,
  ChatThreadRecord,
  ProjectPreferences,
} from '@/features/project/types'
import { isPillKind } from '@/features/composer/mentionDoc'
import { useResearchRunApi } from '@/features/researchRuns/useResearchRunApi'
import type {
  ChatModelOption,
  ChatModelTier,
  CreateResearchRunRequest,
  InqtrixHealth,
  InqtrixStack,
  ModelCatalogEntry,
  NodeModelResolution,
  ResearchRunEvent,
  ResearchRunResult,
  ResearchRunSummary,
} from '@/features/researchRuns/types'
import SettingsWorkspace from '@/features/settings/SettingsWorkspace'
import { FileLibraryWorkspace } from '@/features/fileLibrary/FileLibraryWorkspace'
import { PromptLibraryWorkspace } from '@/features/promptLibrary/PromptLibraryWorkspace'
import { ingestFiles } from '@/features/files/ingest'
import { FILE_SECTION_TEMP_ID } from '@/features/files/sections'
import {
  evaluateBudget,
  shouldShowAttachmentBudgetNotice,
} from '@/features/files/budget'
import { estimateTokensFromText } from '@/features/files/contextTokens'
import { useLocale } from '@/i18n/LocaleProvider'
import { useTheme } from '@/theme/ThemeProvider'
import { contentWithAttachmentContext } from '@/features/project/attachmentContext'
import { AppRail } from './components/AppRail'
import { ResearchWorkspace } from './components/ResearchWorkspace'
import { Topbar } from './components/Topbar'
import { useMediaQuery } from './hooks/useMediaQuery'
import {
  initializeResearchDeskState,
  researchDeskReducer,
  stackOptions,
  visibleResearchJobs,
} from './state'

type ActiveChatRequest = {
  assistantMessageId: string
  phase: 'streaming' | 'submitted'
  threadId: string
}

type ScheduledChatContent = {
  assistantMessageId: string
  contentMarkdown: string
  threadId: string
}

type ChatSendOptions = {
  modelTier?: ChatModelTier
  model?: string | null
  effort?: string | null
}

type ChatModelOptionsState = {
  options: ChatModelOption[]
  status: 'available' | 'missing' | 'unresolved'
}

const INCOGNITO_THREAD_ID = 'chat-incognito-session'

/** Per-step timeout for a prompt-chaining run; exceeding it aborts the chain. */
const CHAT_CHAIN_STEP_TIMEOUT_MS = 120_000
const MAX_PARALLEL_CHAT_REQUESTS = 3

function removeChatRequest(
  requests: Record<string, ActiveChatRequest>,
  threadId: string,
  assistantMessageId?: string,
) {
  const request = requests[threadId]
  if (!request) return requests
  if (assistantMessageId && request.assistantMessageId !== assistantMessageId) return requests

  const next = { ...requests }
  delete next[threadId]
  return next
}

export function ResearchDesk() {
  const { locale, setLocale, t } = useLocale()
  const {
    contrastMode,
    preset: themePreset,
    setContrastMode,
    setPreset: setThemePreset,
    setTheme,
    theme,
  } = useTheme()
  const reduceMotion = useReducedMotion()
  const isDesktop = useMediaQuery('(min-width: 1024px)')
  const [state, dispatch] = useReducer(
    researchDeskReducer,
    undefined,
    initializeResearchDeskState,
  )
  const [projectAction, setProjectAction] = useState<'export' | 'load' | 'save' | null>(null)
  const [projectActionError, setProjectActionError] = useState<string | null>(null)
  const [apiKey, setApiKey] = useState('')
  const [apiKeyDraft, setApiKeyDraft] = useState('')
  const [authLockError, setAuthLockError] = useState<string | null>(null)
  const [isAuthSubmitting, setIsAuthSubmitting] = useState(false)
  const [cancelSubmittingRunIds, setCancelSubmittingRunIds] = useState<ReadonlySet<string>>(() => new Set())
  const [cancelErrorByRunId, setCancelErrorByRunId] = useState<Record<string, string>>({})
  const [activeChatRequestsByThreadId, setActiveChatRequestsByThreadId] = useState<Record<string, ActiveChatRequest>>({})
  const [chatErrorByThreadId, setChatErrorByThreadId] = useState<Record<string, string>>({})
  const [chatNoticeByThreadId, setChatNoticeByThreadId] = useState<Record<string, string>>({})
  const [isIncognitoChat, setIsIncognitoChat] = useState(false)
  const [incognitoThread, setIncognitoThread] = useState<ChatThreadRecord>(() => createIncognitoThread(
    t.chat.incognitoTitle,
    t.chat.incognitoPreview,
  ))
  const [chatStreamingEnabled, setChatStreamingEnabled] = useState(true)
  const chatControllerByThreadIdRef = useRef<Map<string, AbortController>>(new Map())
  const chatFlushFrameByThreadIdRef = useRef<Map<string, number>>(new Map())
  const chatStreamContentByThreadIdRef = useRef<Map<string, string>>(new Map())
  const scheduledChatContentByThreadIdRef = useRef<Map<string, ScheduledChatContent>>(new Map())
  const allJobs = projectResearchJobs(state)
  const visibleJobs = visibleResearchJobs(allJobs, state.ui.activeFilter)
  const isDemoMode = state.connection.kind === 'demo'

  // Demo-only "live" simulator: while the seed run is running in demo mode, feed
  // synthetic snapshot events through the real appendApiRunEvent pipeline so the
  // running card visibly progresses (phases advance, metrics count up and flash,
  // new live-status rows rise). No-op outside demo and for real API runs.
  const researchRunsRef = useRef(state.researchRuns)
  useEffect(() => {
    researchRunsRef.current = state.researchRuns
  }, [state.researchRuns])
  useEffect(() => {
    if (!isDemoMode) return undefined
    const seedId = 'RO-0247'
    const seed = researchRunsRef.current[seedId]
    if (!seed || seed.status !== 'running') return undefined

    const maxRounds = 5
    const nodeOrder = ['classify', 'plan', 'search', 'evaluate', 'answer'] as const
    let nodeIndex = nodeOrder.length - 1
    let round = 2
    let queries = seed.metrics.queries
    let sources = seed.metrics.sources
    let claims = seed.metrics.claims
    let sequence = 1000

    const intervalId = window.setInterval(() => {
      if (researchRunsRef.current[seedId]?.status !== 'running') return
      nodeIndex += 1
      if (nodeIndex >= nodeOrder.length) {
        nodeIndex = 0
        round = round >= maxRounds ? 1 : round + 1
      }
      const node = nodeOrder[nodeIndex]
      queries += 3
      sources += 5
      claims += node === 'search' || node === 'evaluate' ? 4 : 1
      const message = node === 'plan'
        ? `Planning search queries (round ${round}/${maxRounds})...`
        : node === 'search'
          ? `Searching ${4 + round} queries (round ${round}/${maxRounds})...`
          : node === 'evaluate'
            ? `Evaluating information quality (after round ${round}/${maxRounds})...`
            : node === 'answer'
              ? 'Synthesizing the answer from verified evidence...'
              : 'Analyzing question and extracting required aspects...'

      sequence += 1
      dispatch({
        event: {
          created_at: Math.floor(Date.now() / 1000),
          data: {
            message,
            snapshot: {
              active_round: round,
              consolidated_claim_count: claims,
              current_node: node,
              max_rounds: maxRounds,
              total_queries: queries,
              total_sources: sources,
            },
          },
          run_id: seedId,
          sequence,
          type: 'inqtrix.progress.message',
        },
        type: 'appendApiRunEvent',
      })
    }, 3500)

    return () => window.clearInterval(intervalId)
  }, [isDemoMode, dispatch])

  const selectedRun = selectedResearchRun(state)
  const chatThreads = useMemo(
    () => projectChatThreads(state),
    [state.chatThreadOrder, state.chatThreads],
  )
  const chatHistorySections = useMemo(
    () => projectChatHistorySections(state),
    [
      state.chatThreadGroupMemberships,
      state.chatThreadGroupOrder,
      state.chatThreadGroups,
      state.chatThreadOrder,
      state.chatThreads,
    ],
  )
  const chatRules = useMemo(
    () => projectChatRules(state),
    [state.chatRuleOrder, state.chatRules],
  )
  const ruleOptions = useMemo(
    () => chatRuleOptionsFromRules(chatRules, 'chat'),
    [chatRules],
  )
  const reportOptions = completedReportOptions(state)
  const [chatPillRefs, setChatPillRefs] = useState<ChatContextReferenceRecord[]>([])
  const combinedChatRefs = dedupeChatContextRefs([...chatPillRefs, ...state.ui.pendingChatAttachmentRefs])
  const pendingChips = chatAttachmentChipsFromRefs(state, combinedChatRefs)
  const fileOptions = fileMentionOptions(state)
  const fileGroupOptions = fileGroupMentionOptions(state)

  const handleAttachChatFiles = async (files: File[]) => {
    const existingLabels = projectFileAssets(state).map((asset) => asset.label)
    const assets = await ingestFiles(files, { kind: 'chat', sectionId: FILE_SECTION_TEMP_ID }, undefined, existingLabels)
    if (assets.length === 0) return
    dispatch({ assets, type: 'ingestFileAssets' })
    for (const asset of assets) {
      dispatch({ ref: { fileId: asset.id, kind: 'file-asset' }, type: 'attachChatContextToDraft' })
    }
  }

  const pendingAttachmentBudget = evaluateBudget(
    chatAttachmentsFromRefs(state, combinedChatRefs).map((attachment) => ({
      content: attachment.contentMarkdown,
      label: attachment.label ?? attachment.title,
    })),
  )
  const attachmentBudgetNotice = shouldShowAttachmentBudgetNotice(pendingAttachmentBudget)
    ? t.chat.attachmentBudgetWarning
    : null
  const displayedChatThread = isIncognitoChat
    ? incognitoThread
    : state.ui.selectedChatThreadId
    ? state.chatThreads[state.ui.selectedChatThreadId]
    : chatThreads[0] ?? null
  const activeChatThreadId = isIncognitoChat ? INCOGNITO_THREAD_ID : displayedChatThread?.id ?? null
  const activeChatRequest = activeChatThreadId
    ? activeChatRequestsByThreadId[activeChatThreadId] ?? null
    : null
  const runningChatThreadIds = useMemo(
    () => new Set(Object.keys(activeChatRequestsByThreadId)),
    [activeChatRequestsByThreadId],
  )
  const currentPreferences: ProjectPreferences = {
    contrastMode,
    locale,
    theme,
    themePreset,
  }
  const handleApiSummary = useCallback((summary: ResearchRunSummary, options?: { select?: boolean }) => {
    dispatch({ select: options?.select, summary, type: 'upsertApiRunSummary' })
  }, [])
  const handleApiEvent = useCallback((event: ResearchRunEvent) => {
    dispatch({ event, type: 'appendApiRunEvent' })
  }, [])
  const handleApiResult = useCallback((result: ResearchRunResult) => {
    dispatch({ result, type: 'attachApiRunResult' })
  }, [])
  const handleApiRunError = useCallback((runId: string, message: string) => {
    dispatch({ message, runId, type: 'markApiRunError' })
  }, [])
  const {
    defaultStackName,
    health,
    lastError: apiError,
    stackDiscoveryStatus,
    stackNames: apiStackNames,
    stacks: apiStacks,
    cancelRun,
    submitRun,
  } = useResearchRunApi({
    apiKey: apiKey.trim() || undefined,
    enabled: !isDemoMode,
    onEvent: handleApiEvent,
    onResult: handleApiResult,
    onRunError: handleApiRunError,
    onSummary: handleApiSummary,
    workspaceId: state.workspaceId,
  })
  const singleStackLabel = useMemo(() => resolveSingleStackLabel(health), [health])
  const effectiveStackOptions = isDemoMode
    ? stackOptions
    : stackDiscoveryStatus === 'available'
      ? apiStackNames
      : []
  const displayedSelectedStack = !isDemoMode && stackDiscoveryStatus === 'unsupported'
    ? singleStackLabel
    : state.ui.selectedStack
  const textImprovementStack = !isDemoMode && stackDiscoveryStatus === 'available'
    ? state.ui.selectedStack
    : undefined
  const canImproveText = !isDemoMode && !(health?.auth_required && !apiKey.trim())
  const chatModelDiscoveryStack = stackDiscoveryStatus === 'available'
    ? state.ui.selectedStack
    : defaultStackName
  const chatModelOptionsState = useMemo(
    () => resolveChatModelOptions(health, chatModelDiscoveryStack, apiStacks),
    [apiStacks, chatModelDiscoveryStack, health],
  )
  const chatModelOptions = chatModelOptionsState.options
  const chatModelCatalog = useMemo(
    () => resolveModelCatalog(health, chatModelDiscoveryStack, apiStacks),
    [apiStacks, chatModelDiscoveryStack, health],
  )
  const defaultChatModel = useMemo(
    () => resolveDefaultChatModel(health, chatModelDiscoveryStack, apiStacks),
    [apiStacks, chatModelDiscoveryStack, health],
  )
  const selectedChatCard = chatModelCatalog.find(
    (entry) => entry.model_id === state.ui.selectedChatModel,
  )?.card ?? null
  // Per-category token estimate for the composer meter (the composer draft is
  // added live inside ChatWorkspace). The attachment content is the same the
  // request will carry; history mirrors buildChatMessages' last-20 window.
  const chatContextBase = useMemo(() => {
    const attachments = chatAttachmentsFromRefs(state, combinedChatRefs)
    let documents = 0
    let reports = 0
    let rules = 0
    for (const attachment of attachments) {
      const tokens = estimateTokensFromText(attachment.contentMarkdown ?? '')
      if (attachment.kind === 'research-report') reports += tokens
      else if (attachment.kind === 'chat-rule') rules += tokens
      else documents += tokens
    }
    const history = (displayedChatThread?.messages ?? []).slice(-20)
    const conversation = history.reduce(
      (sum, message) => sum + estimateTokensFromText(message.contentMarkdown ?? ''),
      0,
    )
    return { documents, reports, rules, conversation }
  }, [state, combinedChatRefs, displayedChatThread])
  const chatContextCapacity = {
    contextWindowTokens: selectedChatCard?.context_window_tokens ?? null,
    reservedOutputTokens: selectedChatCard?.max_output_tokens ?? 0,
  }
  const isAuthLocked = !isDemoMode && health?.auth_required === true && !apiKey.trim()

  const flushScheduledChatContent = useCallback((threadId: string) => {
    const scheduled = scheduledChatContentByThreadIdRef.current.get(threadId)
    if (!scheduled) return
    scheduledChatContentByThreadIdRef.current.delete(threadId)
    if (scheduled.threadId === INCOGNITO_THREAD_ID) {
      setIncognitoThread((current) => setThreadAssistantMessageContent(
        current,
        scheduled.assistantMessageId,
        scheduled.contentMarkdown,
      ))
      return
    }
    dispatch({
      assistantMessageId: scheduled.assistantMessageId,
      contentMarkdown: scheduled.contentMarkdown,
      threadId: scheduled.threadId,
      type: 'setChatAssistantMessageContent',
    })
  }, [])

  const flushPendingChatContent = useCallback((threadId: string) => {
    const frame = chatFlushFrameByThreadIdRef.current.get(threadId)
    if (frame !== undefined) {
      window.cancelAnimationFrame(frame)
      chatFlushFrameByThreadIdRef.current.delete(threadId)
    }
    flushScheduledChatContent(threadId)
  }, [flushScheduledChatContent])

  const scheduleChatContent = useCallback((
    threadId: string,
    assistantMessageId: string,
    contentMarkdown: string,
  ) => {
    scheduledChatContentByThreadIdRef.current.set(threadId, {
      assistantMessageId,
      contentMarkdown,
      threadId,
    })
    if (chatFlushFrameByThreadIdRef.current.has(threadId)) return
    const frame = window.requestAnimationFrame(() => {
      chatFlushFrameByThreadIdRef.current.delete(threadId)
      flushScheduledChatContent(threadId)
    })
    chatFlushFrameByThreadIdRef.current.set(threadId, frame)
  }, [flushScheduledChatContent])

  useEffect(() => {
    return () => {
      for (const controller of chatControllerByThreadIdRef.current.values()) {
        controller.abort()
      }
      for (const frame of chatFlushFrameByThreadIdRef.current.values()) {
        window.cancelAnimationFrame(frame)
      }
      chatControllerByThreadIdRef.current.clear()
      chatFlushFrameByThreadIdRef.current.clear()
      chatStreamContentByThreadIdRef.current.clear()
      scheduledChatContentByThreadIdRef.current.clear()
    }
  }, [])

  function reportProjectActionError(error: unknown) {
    logProjectActionError(error)
    if (error instanceof DOMException && error.name === 'AbortError') return
    setProjectActionError(`${t.topbar.projectActionFailed}: ${messageFromError(error)}`)
  }

  async function handleLoadProject() {
    setProjectActionError(null)
    try {
      const nextState = await loadProject({ onWorkStart: () => setProjectAction('load') })
      abortAllChatRequests()
      applyProjectPreferences(nextState.preferences)
      setIsIncognitoChat(false)
      setIncognitoThread(createIncognitoThread(t.chat.incognitoTitle, t.chat.incognitoPreview))
      setActiveChatRequestsByThreadId({})
      dispatch({ state: nextState, type: 'hydrateProject' })
    } catch (error) {
      reportProjectActionError(error)
    } finally {
      setProjectAction(null)
    }
  }

  async function handleExportProject() {
    setProjectActionError(null)
    try {
      const result = await exportProject(
        projectStateWithPreferences(state, currentPreferences),
        { onWorkStart: () => setProjectAction('export') },
      )
      dispatch({
        connection: result.connection,
        preferences: currentPreferences,
        savedAt: result.savedAt,
        type: 'markProjectSaved',
      })
    } catch (error) {
      reportProjectActionError(error)
    } finally {
      setProjectAction(null)
    }
  }

  async function handleSaveProject() {
    setProjectActionError(null)
    try {
      const result = await saveProject(
        projectStateWithPreferences(state, currentPreferences),
        { onWorkStart: () => setProjectAction('save') },
      )
      dispatch({
        connection: result.connection,
        preferences: currentPreferences,
        savedAt: result.savedAt,
        type: 'markProjectSaved',
      })
    } catch (error) {
      reportProjectActionError(error)
    } finally {
      setProjectAction(null)
    }
  }

  function handleComposerSubmit(request: CreateResearchRunRequest) {
    if (isDemoMode) {
      dispatch({ request, type: 'createLocalRun' })
      return
    }
    if (health?.auth_required && !apiKey.trim()) return

    void submitRun(stackDiscoveryStatus === 'unsupported'
      ? { ...request, stack: undefined }
      : request)
  }

  async function handleChatMessageSubmit(
    contentMarkdown: string,
    inlineAttachmentRefs: ChatContextReferenceRecord[] = [],
    options: ChatSendOptions = {},
  ) {
    const trimmedContent = contentMarkdown.trim()
    if (!trimmedContent) return
    if (health?.auth_required && !apiKey.trim()) return

    const selectedThread = isIncognitoChat
      ? incognitoThread
      : state.ui.selectedChatThreadId
      ? state.chatThreads[state.ui.selectedChatThreadId]
      : chatThreads[0] ?? undefined
    const threadId = isIncognitoChat
      ? INCOGNITO_THREAD_ID
      : selectedThread?.id ?? createClientId('chat')
    if (chatControllerByThreadIdRef.current.has(threadId)) return
    if (chatControllerByThreadIdRef.current.size >= MAX_PARALLEL_CHAT_REQUESTS) {
      setChatNoticeByThreadId((current) => ({
        ...current,
        [threadId]: t.chat.parallelLimitReached,
      }))
      return
    }

    const userMessageId = createClientId('msg')
    const assistantMessageId = createClientId('msg')
    const createdAt = new Date().toISOString()
    const useStreaming = chatStreamingEnabled
    const modelTier = options.modelTier ?? state.ui.selectedChatModelTier
    const explicitModel = options.model ?? state.ui.selectedChatModel
    const explicitEffort = options.effort ?? state.ui.selectedChatEffort
    const chatStack = stackDiscoveryStatus === 'available' ? state.ui.selectedStack : undefined
    const modelResolution = explicitModel
      ? explicitModelResolution(explicitModel, explicitEffort)
      : chatMessageModelResolutionForTier(
          chatModelOptionsState,
          defaultChatModel,
          modelTier,
        )
    const messageAttachmentRefs = dedupeChatContextRefs([
      ...inlineAttachmentRefs,
      ...state.ui.pendingChatAttachmentRefs,
    ])
    const messageAttachments = chatAttachmentsFromRefs(state, messageAttachmentRefs)
    const requestMessages = buildChatMessages(
      selectedThread?.messages ?? [],
      trimmedContent,
      messageAttachments,
    )

    if (isIncognitoChat) {
      setIncognitoThread((current) => appendChatExchangeToThread(
        current,
        {
          assistantMessageId,
          attachments: messageAttachments,
          contentMarkdown: trimmedContent,
          createdAt,
          modelResolution,
          userMessageId,
        },
      ))
      dispatch({ type: 'clearChatDraftAttachment' })
    } else {
      dispatch({
        assistantMessageId,
        contentMarkdown: trimmedContent,
        createdAt,
        attachmentRefs: messageAttachmentRefs,
        modelResolution,
        threadId,
        type: 'startChatExchange',
        userMessageId,
      })
    }
    setChatErrorByThreadId((current) => {
      const next = { ...current }
      delete next[threadId]
      return next
    })
    setChatNoticeByThreadId((current) => {
      const next = { ...current }
      delete next[threadId]
      return next
    })

    const controller = new AbortController()
    chatControllerByThreadIdRef.current.set(threadId, controller)
    chatStreamContentByThreadIdRef.current.set(threadId, '')
    setActiveChatRequestsByThreadId((current) => ({
      ...current,
      [threadId]: {
        assistantMessageId,
        phase: 'submitted',
        threadId,
      },
    }))

    const chainTemplates = state.ui.chatChainingEnabled
      ? chatFunctionChainTemplatesFromRefs(state.chatRules, messageAttachmentRefs)
      : []

    if (chainTemplates.length > 0) {
      await runChatChainRequest({
        assistantMessageId,
        controller,
        history: selectedThread?.messages ?? [],
        modelTier,
        model: explicitModel,
        effort: explicitEffort,
        sourceAttachments: chatAttachmentsFromRefs(
          state,
          messageAttachmentRefs.filter((ref) => isPillKind(ref.kind)),
        ),
        stack: chatStack,
        templates: chainTemplates,
        threadId,
        useStreaming,
        userText: trimmedContent,
      })
    } else {
      await runChatAssistantRequest({
        assistantMessageId,
        controller,
        requestMessages,
        modelTier,
        model: explicitModel,
        effort: explicitEffort,
        stack: chatStack,
        threadId,
        useStreaming,
      })
    }
  }

  async function runChatChainRequest({
    assistantMessageId,
    controller,
    history,
    modelTier,
    model,
    effort,
    sourceAttachments,
    stack,
    templates,
    threadId,
    useStreaming,
    userText,
  }: {
    assistantMessageId: string
    controller: AbortController
    history: ChatMessageRecord[]
    modelTier: ChatModelTier | null
    model: string | null
    effort: string | null
    sourceAttachments: ChatMessageAttachmentRecord[]
    stack?: string
    templates: { instruction: string; label: string }[]
    threadId: string
    useStreaming: boolean
    userText: string
  }) {
    const trace: ChatChainStepRecord[] = []
    let running = userText
    let modelResolution: ChatMessageModelResolutionRecord | undefined

    try {
      if (isDemoMode) throw new Error(t.chat.demoModeDisabled)

      for (let index = 0; index < templates.length; index += 1) {
        const template = templates[index]
        const isFinal = index === templates.length - 1
        const stepMessages = buildChainStepMessages(
          index === 0 ? history : [],
          template.instruction,
          running,
          index === 0 ? sourceAttachments : [],
        )
        const baseStepRequest = {
          agentOverrides: chatAgentOverrides(modelTier, model, effort),
          includeProgress: false,
          messages: stepMessages,
          mode: 'direct_llm' as const,
          stack,
        }
        const stepTimeout = globalThis.setTimeout(() => controller.abort(), CHAT_CHAIN_STEP_TIMEOUT_MS)
        try {
          if (isFinal && useStreaming) {
            chatStreamContentByThreadIdRef.current.set(threadId, '')
            await streamChatCompletion(
              { ...baseStepRequest, stream: true },
              {
                apiKey: apiKey.trim() || undefined,
                signal: controller.signal,
                workspaceId: state.workspaceId,
                onDelta: (delta) => {
                  const nextContent = `${chatStreamContentByThreadIdRef.current.get(threadId) ?? ''}${delta}`
                  chatStreamContentByThreadIdRef.current.set(threadId, nextContent)
                  setActiveChatRequestsByThreadId((current) => {
                    const request = current[threadId]
                    if (request?.assistantMessageId !== assistantMessageId) return current
                    return { ...current, [threadId]: { ...request, phase: 'streaming' } }
                  })
                  scheduleChatContent(threadId, assistantMessageId, nextContent)
                },
                onModelResolution: (resolution) => {
                  const normalized = normalizeChatMessageModelResolution(resolution)
                  if (normalized) modelResolution = normalized
                },
              },
            )
            flushPendingChatContent(threadId)
            running = (chatStreamContentByThreadIdRef.current.get(threadId) ?? '').trim() || t.chat.emptyResponse
          } else {
            const response = await createChatCompletion(
              { ...baseStepRequest, stream: false },
              {
                apiKey: apiKey.trim() || undefined,
                signal: controller.signal,
                workspaceId: state.workspaceId,
              },
            )
            running = response.choices[0]?.message.content.trim() || t.chat.emptyResponse
            if (isFinal) {
              modelResolution = normalizeChatMessageModelResolution(response.inqtrix?.model_resolution) ?? modelResolution
            }
          }
        } finally {
          globalThis.clearTimeout(stepTimeout)
        }

        if (controller.signal.aborted) {
          trace.push({ label: template.label, output: running, status: 'stopped' })
          setChatAssistantMessageContent(
            threadId,
            assistantMessageId,
            stoppedChatContent(isFinal ? running : '', t.chat.generationStopped),
            modelResolution,
            trace,
          )
          return
        }
        trace.push({ label: template.label, output: running, status: 'ok' })
        setChatAssistantMessageContent(
          threadId,
          assistantMessageId,
          isFinal ? running : (chatStreamContentByThreadIdRef.current.get(threadId) ?? ''),
          modelResolution,
          trace,
        )
      }
    } catch (error) {
      flushPendingChatContent(threadId)
      if (controller.signal.aborted) {
        setChatAssistantMessageContent(
          threadId,
          assistantMessageId,
          stoppedChatContent(running, t.chat.generationStopped),
          modelResolution,
          trace,
        )
      } else {
        const errorMessage = `${t.chat.requestFailed}: ${messageFromError(error)}`
        const failedTrace: ChatChainStepRecord[] = [
          ...trace,
          { label: templates[trace.length]?.label ?? t.chat.chainStepFailed, output: errorMessage, status: 'error' },
        ]
        setChatAssistantMessageContent(threadId, assistantMessageId, errorMessage, modelResolution, failedTrace)
        setChatErrorByThreadId((current) => ({ ...current, [threadId]: errorMessage }))
      }
    } finally {
      if (chatControllerByThreadIdRef.current.get(threadId) === controller) {
        chatControllerByThreadIdRef.current.delete(threadId)
      }
      chatStreamContentByThreadIdRef.current.delete(threadId)
      setActiveChatRequestsByThreadId((current) => removeChatRequest(current, threadId, assistantMessageId))
    }
  }

  async function runChatAssistantRequest({
    assistantMessageId,
    controller,
    modelTier,
    model,
    effort,
    requestMessages,
    stack,
    threadId,
    useStreaming,
  }: {
    assistantMessageId: string
    controller: AbortController
    modelTier: ChatModelTier | null
    model: string | null
    effort: string | null
    requestMessages: ChatCompletionMessage[]
    stack?: string
    threadId: string
    useStreaming: boolean
  }) {
    const baseChatRequest = {
      agentOverrides: chatAgentOverrides(modelTier, model, effort),
      includeProgress: false,
      messages: requestMessages,
      mode: 'direct_llm' as const,
      stack,
    }

    async function writeBlockingAnswer() {
      const response = await createChatCompletion(
        {
          ...baseChatRequest,
          stream: false,
        },
        {
          apiKey: apiKey.trim() || undefined,
          signal: controller.signal,
          workspaceId: state.workspaceId,
        },
      )
      const answer = response.choices[0]?.message.content.trim() || t.chat.emptyResponse
      setChatAssistantMessageContent(
        threadId,
        assistantMessageId,
        answer,
        normalizeChatMessageModelResolution(response.inqtrix?.model_resolution),
      )
    }

    try {
      if (isDemoMode) {
        throw new Error(t.chat.demoModeDisabled)
      }

      if (useStreaming) {
        await streamChatCompletion(
          {
            ...baseChatRequest,
            stream: true,
          },
          {
            apiKey: apiKey.trim() || undefined,
            signal: controller.signal,
            workspaceId: state.workspaceId,
            onDelta: (delta) => {
              const nextContent = `${chatStreamContentByThreadIdRef.current.get(threadId) ?? ''}${delta}`
              chatStreamContentByThreadIdRef.current.set(threadId, nextContent)
              setActiveChatRequestsByThreadId((current) => {
                const request = current[threadId]
                if (request?.assistantMessageId !== assistantMessageId) return current
                return {
                  ...current,
                  [threadId]: { ...request, phase: 'streaming' },
                }
              })
              scheduleChatContent(threadId, assistantMessageId, nextContent)
            },
            onModelResolution: (resolution) => {
              const modelResolution = normalizeChatMessageModelResolution(resolution)
              if (!modelResolution) return
              setChatAssistantMessageContent(
                threadId,
                assistantMessageId,
                chatStreamContentByThreadIdRef.current.get(threadId) ?? '',
                modelResolution,
              )
            },
          },
        )
        flushPendingChatContent(threadId)
        const streamedContent = chatStreamContentByThreadIdRef.current.get(threadId) ?? ''
        if (!streamedContent.trim()) {
          setChatStreamingEnabled(false)
          setChatNoticeByThreadId((current) => ({
            ...current,
            [threadId]: t.chat.streamingFallback,
          }))
          await writeBlockingAnswer()
        }
      } else {
        await writeBlockingAnswer()
      }
    } catch (error) {
      flushPendingChatContent(threadId)
      let chatFailure = error
      if (
        useStreaming
        && !controller.signal.aborted
        && !(chatStreamContentByThreadIdRef.current.get(threadId) ?? '').trim()
      ) {
        try {
          setChatStreamingEnabled(false)
          setChatNoticeByThreadId((current) => ({
            ...current,
            [threadId]: t.chat.streamingFallback,
          }))
          await writeBlockingAnswer()
          return
        } catch (fallbackError) {
          chatFailure = fallbackError
        }
      }

      const streamContent = chatStreamContentByThreadIdRef.current.get(threadId) ?? ''
      const partialContent = streamContent.trim()
      const errorMessage = `${t.chat.requestFailed}: ${messageFromError(chatFailure)}`
      const content = controller.signal.aborted
        ? stoppedChatContent(streamContent, t.chat.generationStopped)
        : partialContent
          ? `${streamContent.trimEnd()}\n\n_${errorMessage}_`
        : errorMessage

      setChatAssistantMessageContent(threadId, assistantMessageId, content)
      if (!controller.signal.aborted) {
        setChatErrorByThreadId((current) => ({
          ...current,
          [threadId]: errorMessage,
        }))
      }
    } finally {
      if (chatControllerByThreadIdRef.current.get(threadId) === controller) {
        chatControllerByThreadIdRef.current.delete(threadId)
      }
      chatStreamContentByThreadIdRef.current.delete(threadId)
      setActiveChatRequestsByThreadId((current) => removeChatRequest(current, threadId, assistantMessageId))
    }
  }

  async function handleAnswerLastUserMessage(threadId: string, messageId: string) {
    if (health?.auth_required && !apiKey.trim()) return

    const selectedThread = threadId === INCOGNITO_THREAD_ID
      ? incognitoThread
      : state.chatThreads[threadId]
    const lastMessage = selectedThread?.messages[selectedThread.messages.length - 1]
    if (
      !selectedThread
      || !lastMessage
      || lastMessage.id !== messageId
      || lastMessage.role !== 'user'
      || !lastMessage.contentMarkdown.trim()
      || chatControllerByThreadIdRef.current.has(threadId)
    ) {
      return
    }
    if (chatControllerByThreadIdRef.current.size >= MAX_PARALLEL_CHAT_REQUESTS) {
      setChatNoticeByThreadId((current) => ({
        ...current,
        [threadId]: t.chat.parallelLimitReached,
      }))
      return
    }

    const assistantMessageId = createClientId('msg')
    const createdAt = new Date().toISOString()
    const useStreaming = chatStreamingEnabled
    const modelTier = state.ui.selectedChatModelTier
    const explicitModel = state.ui.selectedChatModel
    const explicitEffort = state.ui.selectedChatEffort
    const modelResolution = explicitModel
      ? explicitModelResolution(explicitModel, explicitEffort)
      : chatMessageModelResolutionForTier(
          chatModelOptionsState,
          defaultChatModel,
          modelTier,
        )
    const requestMessages = buildChatMessages(
      selectedThread.messages.slice(0, -1),
      lastMessage.contentMarkdown,
      lastMessage.attachments ?? [],
    )

    if (threadId === INCOGNITO_THREAD_ID) {
      setIncognitoThread((current) => appendAssistantResponseToLastUserMessage(
        current,
        {
          assistantMessageId,
          createdAt,
          modelResolution,
          userMessageId: messageId,
        },
      ))
    } else {
      dispatch({
        assistantMessageId,
        createdAt,
        modelResolution,
        threadId,
        type: 'startChatAssistantResponse',
        userMessageId: messageId,
      })
    }
    setChatErrorByThreadId((current) => {
      const next = { ...current }
      delete next[threadId]
      return next
    })
    setChatNoticeByThreadId((current) => {
      const next = { ...current }
      delete next[threadId]
      return next
    })

    const controller = new AbortController()
    chatControllerByThreadIdRef.current.set(threadId, controller)
    chatStreamContentByThreadIdRef.current.set(threadId, '')
    setActiveChatRequestsByThreadId((current) => ({
      ...current,
      [threadId]: {
        assistantMessageId,
        phase: 'submitted',
        threadId,
      },
    }))

    await runChatAssistantRequest({
      assistantMessageId,
      controller,
      modelTier,
      model: explicitModel,
      effort: explicitEffort,
      requestMessages,
      stack: stackDiscoveryStatus === 'available' ? state.ui.selectedStack : undefined,
      threadId,
      useStreaming,
    })
  }

  function handleStopChatGeneration() {
    if (!activeChatThreadId) return
    chatControllerByThreadIdRef.current.get(activeChatThreadId)?.abort()
  }

  function setChatAssistantMessageContent(
    threadId: string,
    assistantMessageId: string,
    contentMarkdown: string,
    modelResolution?: ChatMessageModelResolutionRecord,
    chainTrace?: ChatChainStepRecord[],
  ) {
    if (threadId === INCOGNITO_THREAD_ID) {
      setIncognitoThread((current) => setThreadAssistantMessageContent(
        current,
        assistantMessageId,
        contentMarkdown,
        modelResolution,
        chainTrace,
      ))
      return
    }

    dispatch({
      assistantMessageId,
      chainTrace,
      contentMarkdown,
      modelResolution,
      threadId,
      type: 'setChatAssistantMessageContent',
    })
  }

  function discardChatRequestRuntime(threadId: string) {
    chatControllerByThreadIdRef.current.get(threadId)?.abort()
    chatControllerByThreadIdRef.current.delete(threadId)
    chatStreamContentByThreadIdRef.current.delete(threadId)
    scheduledChatContentByThreadIdRef.current.delete(threadId)
    const frame = chatFlushFrameByThreadIdRef.current.get(threadId)
    if (frame !== undefined) {
      window.cancelAnimationFrame(frame)
      chatFlushFrameByThreadIdRef.current.delete(threadId)
    }
  }

  function abortAllChatRequests() {
    for (const threadId of Array.from(chatControllerByThreadIdRef.current.keys())) {
      discardChatRequestRuntime(threadId)
    }
  }

  function handleCreateChatThread(groupId?: string | null) {
    if (groupId) {
      if (isIncognitoChat && chatControllerByThreadIdRef.current.has(INCOGNITO_THREAD_ID)) return
      if (isIncognitoChat) {
        setIsIncognitoChat(false)
        setIncognitoThread(createIncognitoThread(t.chat.incognitoTitle, t.chat.incognitoPreview))
      }
      dispatch({ groupId, type: 'createChatThread' })
      return
    }

    if (isIncognitoChat) {
      if (chatControllerByThreadIdRef.current.has(INCOGNITO_THREAD_ID)) return
      setChatErrorByThreadId((current) => {
        const next = { ...current }
        delete next[INCOGNITO_THREAD_ID]
        return next
      })
      setChatNoticeByThreadId((current) => {
        const next = { ...current }
        delete next[INCOGNITO_THREAD_ID]
        return next
      })
      setIncognitoThread(createIncognitoThread(t.chat.incognitoTitle, t.chat.incognitoPreview))
      dispatch({ type: 'clearChatDraftAttachment' })
      return
    }

    dispatch({ type: 'createChatThread' })
  }

  function handleClearChatThread() {
    const threadId = isIncognitoChat ? INCOGNITO_THREAD_ID : state.ui.selectedChatThreadId
    if (threadId && chatControllerByThreadIdRef.current.has(threadId)) return
    if (isIncognitoChat) {
      setIncognitoThread(createIncognitoThread(t.chat.incognitoTitle, t.chat.incognitoPreview))
      setChatErrorByThreadId((current) => {
        const next = { ...current }
        delete next[INCOGNITO_THREAD_ID]
        return next
      })
      setChatNoticeByThreadId((current) => {
        const next = { ...current }
        delete next[INCOGNITO_THREAD_ID]
        return next
      })
      dispatch({ type: 'clearChatDraftAttachment' })
      return
    }
    if (!threadId) return
    dispatch({ threadId, type: 'clearChatThread' })
    setChatErrorByThreadId((current) => {
      const next = { ...current }
      delete next[threadId]
      return next
    })
    setChatNoticeByThreadId((current) => {
      const next = { ...current }
      delete next[threadId]
      return next
    })
  }

  function handleIncognitoChange(enabled: boolean) {
    if (activeChatThreadId && chatControllerByThreadIdRef.current.has(activeChatThreadId)) return
    setIsIncognitoChat(enabled)
    setIncognitoThread(createIncognitoThread(t.chat.incognitoTitle, t.chat.incognitoPreview))
    dispatch({ type: 'clearChatDraftAttachment' })
  }

  function handleSelectChatThread(threadId: string) {
    if (isIncognitoChat && chatControllerByThreadIdRef.current.has(INCOGNITO_THREAD_ID)) return
    if (isIncognitoChat) {
      setIsIncognitoChat(false)
      setIncognitoThread(createIncognitoThread(t.chat.incognitoTitle, t.chat.incognitoPreview))
    }
    dispatch({ threadId, type: 'selectChatThread' })
  }

  function handleDeleteChatThread(threadId: string) {
    discardChatRequestRuntime(threadId)
    dispatch({ threadId, type: 'deleteChatThread' })
    setChatErrorByThreadId((current) => {
      const next = { ...current }
      delete next[threadId]
      return next
    })
    setChatNoticeByThreadId((current) => {
      const next = { ...current }
      delete next[threadId]
      return next
    })
    setActiveChatRequestsByThreadId((current) => removeChatRequest(current, threadId))
  }

  function handleDeleteChatMessages(threadId: string, messageIds: string[]) {
    if (messageIds.length === 0 || chatControllerByThreadIdRef.current.has(threadId)) return
    if (threadId === INCOGNITO_THREAD_ID) {
      setIncognitoThread((current) => removeMessagesFromThread(current, messageIds))
      return
    }

    dispatch({ messageIds, threadId, type: 'deleteChatMessages' })
  }

  function handleEditChatMessage(threadId: string, messageId: string, contentMarkdown: string) {
    if (chatControllerByThreadIdRef.current.has(threadId)) return
    if (threadId === INCOGNITO_THREAD_ID) {
      setIncognitoThread((current) => editUserMessageInThread(current, messageId, contentMarkdown))
      return
    }

    dispatch({ contentMarkdown, messageId, threadId, type: 'editChatUserMessage' })
  }

  function handleBranchChatThreadFromMessage(threadId: string, messageId: string) {
    if (threadId === INCOGNITO_THREAD_ID || chatControllerByThreadIdRef.current.has(threadId)) return
    dispatch({ messageId, threadId, type: 'branchChatThreadFromMessage' })
  }

  function handleCreateChatThreadGroup() {
    dispatch({ title: t.chat.newGroupTitle, type: 'createChatThreadGroup' })
  }

  async function handleAuthUnlock(token: string) {
    const trimmedToken = token.trim()
    if (!trimmedToken) {
      setAuthLockError(t.authLock.tokenRequired)
      return
    }

    setIsAuthSubmitting(true)
    setAuthLockError(null)
    try {
      await listResearchRuns({ apiKey: trimmedToken, workspaceId: state.workspaceId })
      setApiKey(trimmedToken)
      setApiKeyDraft(trimmedToken)
    } catch (error) {
      setAuthLockError(hasHttpStatus(error, 401) || hasHttpStatus(error, 403)
        ? t.authLock.tokenRejected
        : t.authLock.tokenCheckFailed)
    } finally {
      setIsAuthSubmitting(false)
    }
  }

  function handleSettingsApiKeyChange(nextApiKey: string) {
    setApiKeyDraft(nextApiKey)
    setApiKey(nextApiKey.trim())
    setAuthLockError(null)
  }

  function applyProjectPreferences(preferences: ProjectPreferences) {
    setContrastMode(preferences.contrastMode)
    setLocale(preferences.locale)
    setTheme(preferences.theme)
    setThemePreset(preferences.themePreset)
  }

  async function handleCancelJob(runId: string) {
    const run = state.researchRuns[runId]
    if (!run || (run.status !== 'running' && run.status !== 'queued')) return

    setCancelErrorByRunId((current) => {
      const next = { ...current }
      delete next[runId]
      return next
    })

    if (isDemoMode || run.source !== 'api') {
      dispatch({ runId, type: 'cancelLocalRun' })
      return
    }

    setCancelSubmittingRunIds((current) => new Set(current).add(runId))
    try {
      await cancelRun(runId)
    } catch (error) {
      setCancelErrorByRunId((current) => ({
        ...current,
        [runId]: messageFromError(error),
      }))
    } finally {
      setCancelSubmittingRunIds((current) => {
        const next = new Set(current)
        next.delete(runId)
        return next
      })
    }
  }

  return (
    <main className="min-h-svh bg-canvas text-foreground lg:flex lg:h-svh lg:flex-col lg:overflow-hidden">
      <Topbar
        activeView={state.ui.activeView}
        dirty={state.dirty}
        isProjectActionPending={projectAction !== null}
        onDismissProjectActionError={() => setProjectActionError(null)}
        onExportProject={() => void handleExportProject()}
        onLoadProject={() => void handleLoadProject()}
        onSaveProject={() => void handleSaveProject()}
        projectActionError={projectActionError}
        projectConnection={state.connection}
        projectName={state.project.name}
      />
      <div className="flex min-h-0 w-full flex-1">
        <AppRail
          activeView={state.ui.activeView}
          onViewChange={(view) => dispatch({ type: 'setActiveView', view })}
        />
        <div className="min-h-0 min-w-0 flex-1">
          {state.ui.activeView === 'research' ? (
            <ResearchWorkspace
              activeFilter={state.ui.activeFilter}
              allJobs={allJobs}
              expandedJobId={state.ui.expandedJobId}
              isComposerVisible={state.ui.isComposerVisible}
              isDesktop={isDesktop}
              isReportExpanded={state.ui.isReportExpanded}
              isReportVisible={state.ui.isReportVisible}
              jobs={visibleJobs}
              cancelErrorByRunId={cancelErrorByRunId}
              cancelSubmittingRunIds={cancelSubmittingRunIds}
              onActiveFilterChange={(filter) => dispatch({ filter, type: 'setActiveFilter' })}
              onComposerSubmit={handleComposerSubmit}
              onComposerVisibleChange={(isVisible) => dispatch({
                isVisible,
                type: 'setComposerVisible',
              })}
              onDeleteJob={(jobId) => dispatch({ jobId, type: 'deleteJob' })}
              onCancelJob={(jobId) => void handleCancelJob(jobId)}
              onReportExpandedChange={(isExpanded) => dispatch({
                isExpanded,
                type: 'setReportExpanded',
              })}
              onReportVisibleChange={(isVisible) => dispatch({
                isVisible,
                type: 'setReportVisible',
              })}
              onSelectJob={(jobId) => dispatch({ jobId, type: 'selectJob' })}
              onToggleJob={(jobId) => dispatch({ jobId, type: 'toggleJob' })}
              onUseReportInChat={(runId) => dispatch({ runId, type: 'attachReportToNewChat' })}
              reduceMotion={reduceMotion}
              selectedJobId={state.ui.selectedJobId}
              selectedRun={selectedRun}
              selectedStack={displayedSelectedStack}
            />
          ) : state.ui.activeView === 'chat' ? (
            <ChatWorkspace
              activeAssistantMessageId={activeChatRequest?.assistantMessageId ?? null}
              chatModelOptions={chatModelOptions}
              chatModelOptionsStatus={chatModelOptionsState.status}
              chatHistorySections={chatHistorySections}
              defaultChatModel={defaultChatModel}
              isDesktop={isDesktop}
              isHistoryVisible={state.ui.isChatHistoryVisible}
              isIncognito={isIncognitoChat}
              isSending={activeChatRequest !== null}
              onAttachContext={(ref) => dispatch({ ref, type: 'attachChatContextToDraft' })}
              onAttachFiles={(files) => void handleAttachChatFiles(files)}
              onPillRefsChange={setChatPillRefs}
              onAnswerLastUserMessage={(threadId, messageId) => void handleAnswerLastUserMessage(threadId, messageId)}
              onClearThread={handleClearChatThread}
              onCreateThread={handleCreateChatThread}
              onCreateThreadGroup={handleCreateChatThreadGroup}
              onBranchFromMessage={handleBranchChatThreadFromMessage}
              onDeleteMessages={handleDeleteChatMessages}
              onDeleteThreadGroup={(groupId) => dispatch({ groupId, type: 'deleteChatThreadGroup' })}
              onDeleteThread={handleDeleteChatThread}
              onEditMessage={handleEditChatMessage}
              chainingEnabled={state.ui.chatChainingEnabled}
              onChainingEnabledChange={(enabled) => dispatch({ enabled, type: 'setChatChainingEnabled' })}
              onIncognitoChange={handleIncognitoChange}
              onHistoryVisibleChange={(isVisible) => dispatch({
                isVisible,
                type: 'setChatHistoryVisible',
              })}
              onOpenPromptLibrary={() => dispatch({ type: 'setActiveView', view: 'prompt-library' })}
              onRenameThread={(threadId, title) => dispatch({ threadId, title, type: 'renameChatThread' })}
              onRenameThreadGroup={(groupId, title) => dispatch({ groupId, title, type: 'renameChatThreadGroup' })}
              onMoveThreadGroup={(groupId, targetIndex) => dispatch({
                groupId,
                targetIndex,
                type: 'moveChatThreadGroup',
              })}
              onMoveThreadToGroup={(threadId, groupId, targetIndex) => dispatch({
                groupId,
                targetIndex,
                threadId,
                type: 'moveChatThreadToGroup',
              })}
              onRemoveContext={(ref) => dispatch({ ref, type: 'removeChatContextFromDraft' })}
              onReorderContext={(fromIndex, toIndex) => dispatch({ fromIndex, toIndex, type: 'reorderChatContextInDraft' })}
              pendingReorderKeys={state.ui.pendingChatAttachmentRefs.map(chatContextRefKey)}
              pillKeys={chatPillRefs.map(chatContextRefKey)}
              onSendMessage={(contentMarkdown, refs, options) => void handleChatMessageSubmit(contentMarkdown, refs, options)}
              onSelectThread={handleSelectChatThread}
              onSelectedModelTierChange={(tier) => dispatch({ tier, type: 'setSelectedChatModelTier' })}
              chatModelCatalog={chatModelCatalog}
              selectedChatModel={state.ui.selectedChatModel}
              selectedChatEffort={state.ui.selectedChatEffort}
              onSelectedChatModelChange={(model) => dispatch({ model, type: 'setSelectedChatModel' })}
              onSelectedChatEffortChange={(effort) => dispatch({ effort, type: 'setSelectedChatEffort' })}
              chatContextBase={chatContextBase}
              chatContextCapacity={chatContextCapacity}
              onStopGenerating={handleStopChatGeneration}
              onStreamingEnabledChange={setChatStreamingEnabled}
              attachmentBudgetNotice={attachmentBudgetNotice}
              pendingChips={pendingChips}
              reduceMotion={reduceMotion}
              requestError={activeChatThreadId
                ? chatErrorByThreadId[activeChatThreadId] ?? null
                : null}
              requestNotice={activeChatThreadId
                ? chatNoticeByThreadId[activeChatThreadId] ?? null
                : null}
              fileGroupOptions={fileGroupOptions}
              fileOptions={fileOptions}
              reportOptions={reportOptions}
              runningThreadIds={runningChatThreadIds}
              ruleOptions={ruleOptions}
              selectedModelTier={state.ui.selectedChatModelTier}
              selectedThreadId={state.ui.selectedChatThreadId}
              streamingEnabled={chatStreamingEnabled}
              temporaryThread={incognitoThread}
              textImprovement={{
                apiKey: apiKey.trim() || undefined,
                enabled: canImproveText,
                selectedStack: textImprovementStack,
                workspaceId: state.workspaceId,
              }}
              threads={chatThreads}
            />
          ) : state.ui.activeView === 'editor' ? (
            <EditorWorkspace
              apiKey={apiKey.trim() || undefined}
              chatModelOptions={chatModelOptions}
              chatModelOptionsStatus={chatModelOptionsState.status}
              chatModelCatalog={chatModelCatalog}
              defaultChatModel={defaultChatModel}
              dispatch={dispatch}
              reportOptions={reportOptions}
              selectedModelTier={state.ui.selectedChatModelTier}
              state={state}
              textImprovement={{
                apiKey: apiKey.trim() || undefined,
                enabled: canImproveText,
                selectedStack: textImprovementStack,
                workspaceId: state.workspaceId,
              }}
            />
          ) : state.ui.activeView === 'prompt-library' ? (
            <PromptLibraryWorkspace
              dispatch={dispatch}
              state={state}
              textImprovement={{
                apiKey: apiKey.trim() || undefined,
                enabled: canImproveText,
                selectedStack: textImprovementStack,
                workspaceId: state.workspaceId,
              }}
            />
          ) : state.ui.activeView === 'database' ? (
            <FileLibraryWorkspace dispatch={dispatch} state={state} />
          ) : (
            <SettingsWorkspace
              apiError={apiError}
              apiHealth={health}
              apiKey={apiKeyDraft}
              isDemoMode={isDemoMode}
              onApiKeyChange={handleSettingsApiKeyChange}
              onDemoModeChange={(enabled) => dispatch({ enabled, type: 'setDemoMode' })}
              onStackChange={(stack) => dispatch({ stack, type: 'setSelectedStack' })}
              reduceMotion={reduceMotion}
              selectedStack={displayedSelectedStack}
              stackDiscoveryStatus={stackDiscoveryStatus}
              stackOptions={effectiveStackOptions}
            />
          )}
        </div>
      </div>
      {isAuthLocked && (
        <AuthLockScreen
          error={authLockError}
          isSubmitting={isAuthSubmitting}
          onSubmit={(token) => void handleAuthUnlock(token)}
          onTokenChange={(token) => {
            setApiKeyDraft(token)
            setAuthLockError(null)
          }}
          reduceMotion={reduceMotion}
          token={apiKeyDraft}
        />
      )}
    </main>
  )
}

function createIncognitoThread(title: string, preview: string): ChatThreadRecord {
  const now = new Date().toISOString()
  return {
    createdAt: now,
    id: INCOGNITO_THREAD_ID,
    messages: [],
    preview,
    source: 'api',
    title,
    updatedAt: now,
  }
}

function appendChatExchangeToThread(
  thread: ChatThreadRecord,
  options: {
    assistantMessageId: string
    attachments: ChatMessageAttachmentRecord[]
    contentMarkdown: string
    createdAt: string
    modelResolution?: ChatMessageModelResolutionRecord
    userMessageId: string
  },
): ChatThreadRecord {
  const userMessage: ChatMessageRecord = {
    attachments: options.attachments.length > 0 ? options.attachments : undefined,
    contentMarkdown: options.contentMarkdown,
    createdAt: options.createdAt,
    id: options.userMessageId,
    role: 'user',
  }
  const assistantMessage: ChatMessageRecord = {
    contentMarkdown: '',
    createdAt: options.createdAt,
    id: options.assistantMessageId,
    modelResolution: options.modelResolution,
    role: 'assistant',
  }

  return {
    ...thread,
    messages: [...thread.messages, userMessage, assistantMessage],
    preview: options.contentMarkdown,
    title: thread.messages.some((message) => message.role === 'user')
      ? thread.title
      : titleFromChatMessage(options.contentMarkdown),
    updatedAt: options.createdAt,
  }
}

function appendAssistantResponseToLastUserMessage(
  thread: ChatThreadRecord,
  options: {
    assistantMessageId: string
    createdAt: string
    modelResolution?: ChatMessageModelResolutionRecord
    userMessageId: string
  },
): ChatThreadRecord {
  const lastMessage = thread.messages[thread.messages.length - 1]
  if (!lastMessage || lastMessage.id !== options.userMessageId || lastMessage.role !== 'user') {
    return thread
  }

  return threadWithMessages(
    thread,
    [
      ...thread.messages,
      {
        contentMarkdown: '',
        createdAt: options.createdAt,
        id: options.assistantMessageId,
        modelResolution: options.modelResolution,
        role: 'assistant',
      },
    ],
  )
}

function setThreadAssistantMessageContent(
  thread: ChatThreadRecord,
  assistantMessageId: string,
  contentMarkdown: string,
  modelResolution?: ChatMessageModelResolutionRecord,
  chainTrace?: ChatChainStepRecord[],
): ChatThreadRecord {
  return {
    ...thread,
    messages: thread.messages.map((message) => (
      message.id === assistantMessageId
        ? {
          ...message,
          chainTrace: chainTrace ?? message.chainTrace,
          contentMarkdown,
          modelResolution: modelResolution ?? message.modelResolution,
        }
        : message
    )),
    updatedAt: new Date().toISOString(),
  }
}

function removeMessagesFromThread(
  thread: ChatThreadRecord,
  messageIds: readonly string[],
): ChatThreadRecord {
  const messageIdSet = new Set(messageIds)
  const messages = thread.messages.filter((message) => !messageIdSet.has(message.id))
  if (messages.length === thread.messages.length) return thread
  return threadWithMessages(thread, messages)
}

function editUserMessageInThread(
  thread: ChatThreadRecord,
  messageId: string,
  contentMarkdown: string,
): ChatThreadRecord {
  const nextContent = contentMarkdown.trim()
  const currentMessage = thread.messages.find((message) => message.id === messageId)
  if (!nextContent || !currentMessage || currentMessage.role !== 'user') return thread
  if (currentMessage.contentMarkdown === nextContent) return thread

  const messages = thread.messages.map((message) => (
    message.id === messageId
      ? { ...message, contentMarkdown: nextContent }
      : message
  ))
  const autoTitle = titleFromChatMessage(currentMessage.contentMarkdown)
  const nextTitle = thread.title === autoTitle
    ? titleFromChatMessage(nextContent)
    : thread.title

  return {
    ...threadWithMessages(thread, messages),
    title: nextTitle,
  }
}

function threadWithMessages(
  thread: ChatThreadRecord,
  messages: ChatMessageRecord[],
): ChatThreadRecord {
  return {
    ...thread,
    messages,
    preview: chatPreviewFromMessages(messages),
    updatedAt: new Date().toISOString(),
  }
}

function chatPreviewFromMessages(messages: readonly ChatMessageRecord[]) {
  return [...messages].reverse().find((message) => message.role === 'user')?.contentMarkdown ?? 'No user message yet'
}

function titleFromChatMessage(contentMarkdown: string) {
  return contentMarkdown
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 72) || 'Incognito chat'
}

function projectStateWithPreferences(
  state: ReturnType<typeof initializeResearchDeskState>,
  preferences: ProjectPreferences,
) {
  return {
    ...state,
    preferences,
  }
}

function buildChatMessages(
  history: ChatMessageRecord[],
  contentMarkdown: string,
  attachments: ChatMessageAttachmentRecord[],
): ChatCompletionMessage[] {
  const messages = history
    .filter((message) => message.contentMarkdown.trim().length > 0)
    .slice(-20)
    .map((message) => ({
      content: message.contentMarkdown,
      role: message.role,
    }))

  messages.push({
    content: attachments.length > 0
      ? contentWithAttachmentContext(contentMarkdown, attachments)
      : contentMarkdown,
    role: 'user',
  })

  return messages
}

/**
 * Build the request messages for one prompt-chaining step: apply a template's
 * instruction to the running text. Thread history and attached sources are only
 * supplied for the first step (the others transform the previous step's result).
 */
function buildChainStepMessages(
  history: ChatMessageRecord[],
  instruction: string,
  runningText: string,
  sources: ChatMessageAttachmentRecord[],
): ChatCompletionMessage[] {
  const messages = history
    .filter((message) => message.contentMarkdown.trim().length > 0)
    .slice(-20)
    .map((message) => ({ content: message.contentMarkdown, role: message.role }))
  const subject = sources.length > 0
    ? contentWithAttachmentContext(runningText, sources)
    : runningText
  messages.push({
    content: `${instruction}\n\n---\n${subject}`,
    role: 'user',
  })
  return messages
}

function directChatModelResolution(
  nodeModels: Record<string, NodeModelResolution> | undefined,
): NodeModelResolution | null {
  if (!nodeModels) return null
  return nodeModels.direct_chat ?? null
}

function resolveChatModelOptions(
  health: InqtrixHealth | null,
  selectedStackName: string | null,
  stacks: InqtrixStack[],
): ChatModelOptionsState {
  const selectedStackModels = selectedStackName
    ? stacks.find((stack) => stack.name === selectedStackName)?.models
    : undefined
  const options = selectedStackName
    ? selectedStackModels?.chat_model_options
    : health?.chat_model_options
  if (!options || options.length === 0) {
    return { options: [], status: 'missing' }
  }
  const hasEveryTier = chatModelTierOrder.every((tier) => (
    options.some((option) => option.tier === tier)
  ))
  if (!hasEveryTier) {
    return { options, status: 'missing' }
  }
  if (options.some((option) => !option.model?.trim())) {
    return { options, status: 'unresolved' }
  }
  return { options, status: 'available' }
}

function resolveModelCatalog(
  health: InqtrixHealth | null,
  selectedStackName: string | null,
  stacks: InqtrixStack[],
): ModelCatalogEntry[] {
  const selectedStackModels = selectedStackName
    ? stacks.find((stack) => stack.name === selectedStackName)?.models
    : undefined
  const catalog = selectedStackName
    ? selectedStackModels?.models_catalog
    : health?.models_catalog
  return catalog ?? []
}

function resolveDefaultChatModel(
  health: InqtrixHealth | null,
  selectedStackName: string | null,
  stacks: InqtrixStack[],
): NodeModelResolution | null {
  const selectedStackModels = selectedStackName
    ? stacks.find((stack) => stack.name === selectedStackName)?.models
    : undefined
  if (selectedStackName) {
    return directChatModelResolution(selectedStackModels?.node_models)
  }
  return directChatModelResolution(health?.node_models)
}

const chatModelTierOrder: ChatModelTier[] = ['high', 'mid', 'fast']

function chatAgentOverrides(
  modelTier: ChatModelTier | null,
  model: string | null,
  effort: string | null,
) {
  // An explicitly picked model wins over the tier (mirrors the backend's
  // explicit_request resolution); empty effort inherits the provider default.
  if (model) return effort ? { model, effort } : { model }
  if (modelTier) return { modelTier }
  return undefined
}

function explicitModelResolution(
  model: string,
  effort: string | null,
): ChatMessageModelResolutionRecord {
  return {
    effort: effort ?? '',
    effortSource: effort ? 'explicit_request' : '',
    model,
    modelSource: 'explicit_request',
    requestedTier: '',
    tier: '',
  }
}

function chatMessageModelResolutionForTier(
  optionsState: ChatModelOptionsState,
  defaultModel: NodeModelResolution | null,
  selectedTier: ChatModelTier | null,
): ChatMessageModelResolutionRecord | undefined {
  if (!selectedTier) {
    return normalizeChatMessageModelResolution(defaultModel)
  }
  if (optionsState.status !== 'available') return undefined
  return normalizeChatMessageModelResolution(
    optionsState.options.find((option) => option.tier === selectedTier),
  )
}

function normalizeChatMessageModelResolution(
  resolution: NodeModelResolution | null | undefined,
): ChatMessageModelResolutionRecord | undefined {
  if (!resolution) return undefined
  const model = resolution.model?.trim()
  if (!model) return undefined
  return {
    effort: resolution.effort?.trim() ?? '',
    effortSource: resolution.effort_source?.trim() ?? '',
    model,
    modelSource: resolution.model_source?.trim() ?? '',
    requestedTier: resolution.requested_tier?.trim() ?? '',
    tier: resolution.tier?.trim() ?? '',
  }
}

function resolveSingleStackLabel(health: InqtrixHealth | null) {
  if (!health) return 'Server default'
  const providerParts = [
    health.llm.provider,
    health.search.provider,
  ].filter(Boolean)
  const providerLabel = providerParts.length > 0
    ? providerParts.join(' + ')
    : 'Server default'
  return health.reasoning_model
    ? `${providerLabel} · ${health.reasoning_model}`
    : providerLabel
}

function stoppedChatContent(partialContent: string, stoppedLabel: string) {
  const trimmed = partialContent.trim()
  if (!trimmed) return stoppedLabel
  return `${partialContent.trimEnd()}\n\n_${stoppedLabel}_`
}

function createClientId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function logProjectActionError(error: unknown) {
  if (error instanceof DOMException && error.name === 'AbortError') return
  console.warn('Inqtrix project action failed.', error)
}

function messageFromError(error: unknown) {
  return error instanceof Error ? error.message : 'Inqtrix cancel request failed.'
}
