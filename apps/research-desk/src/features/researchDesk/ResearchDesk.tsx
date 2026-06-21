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
  uploadServerFile,
  createChatCompletion,
  fetchKnowledgeDocumentText,
  fetchServerFileContent,
  fetchServerFileText,
  hasHttpStatus,
  listResearchRuns,
  loginLdap,
  loginLocal,
  searchKnowledge,
  streamChatCompletion,
  type AuthConfig,
  type ChatCompletionMessage,
  type ClientOptions,
} from '@/api/inqtrixClient'
import { type AuthMode, isCookieSessionMode } from '@/features/auth/authMode'
import { AuthLockScreen } from './components/AuthLockScreen'
import ChatWorkspace from '@/features/chat/ChatWorkspace'
import type { KnowledgeIndexOption } from '@/features/chat/ChatWorkspace'
import { useChatHistoryApi } from '@/features/chat/useChatHistoryApi'
import { useEditorHistoryApi } from '@/features/editor/useEditorHistoryApi'
import { useAssetHistoryApi } from '@/features/fileLibrary/useAssetHistoryApi'
import { useVectorIndexHistoryApi } from '@/features/fileLibrary/useVectorIndexHistoryApi'
import { useAccountPreferences } from '@/features/account/useAccountPreferences'
import { useProjectServerImport } from '@/features/project/useProjectServerImport'
import EditorWorkspace from '@/features/editor/EditorWorkspace'
import { exportProject, loadProject, saveProject } from '@/features/project/fileSystem'
import {
  assetIdsFromChatRefs,
  chatAttachmentsFromRefs,
  projectChatHistorySections,
  chatRuleOptionsFromRules,
  chatAttachmentChipsFromRefs,
  chatContextRefKey,
  completedReportOptions,
  dedupeChatContextRefs,
  fileGroupMentionOptions,
  fileMentionOptions,
  projectAllKnowledgeItems,
  projectChatThreads,
  projectChatRules,
  projectFileAssets,
  projectKnowledgeItems,
  projectKnowledgeSessionSections,
  projectKnowledgeSessions,
  projectResearchJobs,
  projectVectorIndexes,
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
  EmbedModelDescriptor,
  KnowledgeSessionRecord,
  KnowledgeThreadItemRecord,
  ProjectPreferences,
} from '@/features/project/types'
import { EMBED_MODELS, type FileAssetRecord } from '@/features/project/types'
import { isPillKind } from '@/features/composer/mentionDoc'
import { useResearchRunApi } from '@/features/researchRuns/useResearchRunApi'
import { useResearchRunImport } from '@/features/researchRuns/useResearchRunImport'
import { useServerDiscovery } from '@/features/researchRuns/useServerDiscovery'
import { deriveChatStepTimeoutMs, isMissingServerTimeouts } from '@/features/researchRuns/clientTimeouts'
import type {
  ChatModelOption,
  ChatModelTier,
  CreateResearchRunRequest,
  InqtrixHealth,
  InqtrixStack,
  ModelCatalogEntry,
  NodeModelResolution,
  ResearchRunEvent,
  ResearchRunMode,
  ResearchRunResult,
  ResearchRunSummary,
} from '@/features/researchRuns/types'
import SettingsWorkspace from '@/features/settings/SettingsWorkspace'
import { KnowledgeWorkspace, type KnowledgeMode } from '@/features/knowledge/KnowledgeWorkspace'
import { useKnowledgeSessionsApi } from '@/features/knowledge/useKnowledgeSessionsApi'
import {
  buildDemoAskScript,
  createDemoKnowledgeDataSource,
  DEMO_KNOWLEDGE_DEFAULT_PROFILE,
  DEMO_KNOWLEDGE_DEFAULT_TOP_K,
  DEMO_KNOWLEDGE_PROFILE_MANIFEST,
} from '@/features/knowledge/demo'
import { knowledgeProfileOptionsFromManifest } from '@/features/knowledge/profileOptions'
import type { KnowledgeCollectionOption, KnowledgeDataSource } from '@/features/knowledge/types'
import { useAuthSession } from '@/features/auth/useAuthSession'
import { QuotaMeterProvider } from '@/features/quota/QuotaMeterContext'
import { ShareDialog } from '@/features/sharing/ShareDialog'
import { DEMO_OWNER } from '@/features/sharing/demoShares'
import {
  DEMO_RUNNING_MAX_ROUNDS,
  DEMO_RUNNING_RUN_ID,
} from '@/features/project/seedProject'
import { personLabel } from '@/features/sharing/shareModel'
import { useOutgoingShareCounts, useSharedWithMe } from '@/features/sharing/useShareSignals'
import { useTemplateSync } from '@/features/promptLibrary/useTemplateSync'
import { FileLibraryWorkspace } from '@/features/fileLibrary/FileLibraryWorkspace'
import { PromptLibraryWorkspace } from '@/features/promptLibrary/PromptLibraryWorkspace'
import { ingestFiles, scheduleServerParse, type ServerFileUpload } from '@/features/files/ingest'
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
import { ProfileAvatar } from './components/ProfileAvatar'
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
  knowledgeCollectionIds?: string[]
}

type ChatModelOptionsState = {
  options: ChatModelOption[]
  status: 'available' | 'missing' | 'unresolved'
}

const INCOGNITO_THREAD_ID = 'chat-incognito-session'

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

export function ResearchDesk({
  authConfig = null,
}: { authConfig?: AuthConfig | null } = {}) {
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
  // Credential-mode (local/ldap) login fields; the password never persists
  // beyond a successful submit.
  const [authIdentifier, setAuthIdentifier] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  // Mirrors the cookie session's authenticated state for the run-list hook
  // (declared before it; the session itself resolves later from health). An
  // anonymous->authenticated flip re-hydrates the run list after an in-app
  // local/ldap login, which mints no remount.
  const [cookieAuthed, setCookieAuthed] = useState(false)
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
    // The seed run id is shared with seedProject so the two never drift.
    const seedId = DEMO_RUNNING_RUN_ID
    const seed = researchRunsRef.current[seedId]
    if (!seed || seed.status !== 'running') return undefined

    const maxRounds = DEMO_RUNNING_MAX_ROUNDS
    const nodeOrder = ['classify', 'plan', 'search', 'evaluate', 'answer'] as const
    // Resume from the seed's active phase (evaluation, round 1) so the streamed
    // counter continues the card's static metrics rather than contradicting them.
    let nodeIndex = nodeOrder.indexOf('evaluate')
    let round = 1
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
        ? `Plane Suchanfragen (Runde ${round}/${maxRounds})...`
        : node === 'search'
          ? `Durchsuche ${4 + round} Suchanfragen (Runde ${round}/${maxRounds})...`
          : node === 'evaluate'
            ? `Bewerte Informationsqualitaet (nach Runde ${round}/${maxRounds})...`
            : node === 'answer'
              ? 'Formuliere Antwort aus verifizierter Evidenz...'
              : 'Analysiere Frage und extrahiere Pflichtaspekte...'

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
  // Session-scoped composer drafts held in the shell (which never unmounts on a view
  // switch) so unsent text survives navigation. Intentionally NOT in the reducer:
  // these must not outlive a full reload (per the "page-switch only" decision).
  const [chatDraft, setChatDraft] = useState('')
  const [researchQuestion, setResearchQuestion] = useState('')
  const combinedChatRefs = dedupeChatContextRefs([...chatPillRefs, ...state.ui.pendingChatAttachmentRefs])
  const pendingChips = chatAttachmentChipsFromRefs(state, combinedChatRefs)
  const fileOptions = fileMentionOptions(state)
  const fileGroupOptions = fileGroupMentionOptions(state)

  const handleAttachChatFiles = async (files: File[]) => {
    const existingLabels = projectFileAssets(state).map((asset) => asset.label)
    const assets = await ingestFiles(
      files,
      { kind: 'chat', sectionId: FILE_SECTION_TEMP_ID },
      undefined,
      existingLabels,
      serverFileUpload,
    )
    if (assets.length === 0) return
    dispatch({ assets, type: 'ingestFileAssets' })
    runServerParse(assets)
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
  // Memoized so the reference is stable across unrelated re-renders (chat
  // streaming re-renders ~every frame): the account-preferences autosave
  // effect then re-arms only on a genuine preference change, matching the
  // stable-reference contract the sibling sync hooks rely on (M6c).
  const currentPreferences = useMemo<ProjectPreferences>(
    () => ({ contrastMode, locale, theme, themePreset }),
    [contrastMode, locale, theme, themePreset],
  )
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
  // Server discovery (health/capabilities/stacks) is workspace-INDEPENDENT and
  // resolved first, because `health` gates cookie mode and `capabilities` gates
  // the persistence tier -- both feeding the auth + namespace resolution below.
  const {
    capabilities,
    defaultStackName,
    health,
    lastError: discoveryError,
    ready: discoveryReady,
    stackDiscoveryStatus,
    stackNames: apiStackNames,
    stacks: apiStacks,
  } = useServerDiscovery({ enabled: !isDemoMode })
  // Client AI-request aborts are derived from the server's published HTTP waits
  // (the /v1/capabilities timeouts block) so a raised server-side timeout is not
  // silently capped by the browser. A ref keeps run-start handlers on the
  // current value without re-threading it through their dependency lists.
  const chatStepTimeoutMsRef = useRef(deriveChatStepTimeoutMs(capabilities))
  chatStepTimeoutMsRef.current = deriveChatStepTimeoutMs(capabilities)
  // No Silent Fallbacks: warn once when a real backend exposes no timeouts block
  // (older server) so the editor/chat fixed-fallback aborts are never silent.
  const clientTimeoutsWarnedRef = useRef(false)
  useEffect(() => {
    if (isMissingServerTimeouts(capabilities) && !clientTimeoutsWarnedRef.current) {
      clientTimeoutsWarnedRef.current = true
      console.warn(
        '/v1/capabilities exposed no timeouts block; editor and chat client '
        + 'aborts are using fixed fallbacks. Update the backend so client '
        + 'timeouts track the server waits.',
      )
    }
  }, [capabilities])
  // Auth + the per-user project namespace are resolved here, after discovery and
  // before the run-API hook, so the run hook and every project-data call scope to
  // the SAME namespace from one resolved session (no in-hook re-probe).
  const authMode: AuthMode = isDemoMode
    ? 'none'
    : health?.auth_mode
      ?? (health?.auth_required ? 'apikey' : 'none')
  const isCookieMode = isCookieSessionMode(authMode)
  const {
    session: authSession,
    login: ssoLogin,
    logout: ssoLogout,
    refresh: refreshAuthSession,
  } = useAuthSession(isCookieMode, state.workspaceId)
  // Project-persistence tier (M6): durable server persistence is offered only
  // by a Postgres backend (not demo). For an authenticated cookie-session user
  // (local/oidc/ldap) it is AUTOMATIC -- serverSyncEnabled is derived from the
  // session below, so the project auto-hydrates on boot and auto-saves with no
  // opt-in. The apikey / local-first tiers keep the explicit import opt-in.
  const projectPersistenceAvailable =
    !isDemoMode && capabilities?.features.project_persistence === true
  // The single auth gate: admitted to list/run when anonymous (`none`), apikey
  // with a key, or a live cookie session. Drives the run list, the auth-lock
  // screen (below), and the namespace flip -- all from the one resolved session.
  const authUnlocked =
    authMode === 'none'
    || (authMode === 'apikey' && apiKey.trim() !== '')
    || (isCookieMode && authSession.status === 'authenticated')
  // The workspace namespace every project-data + run server call scopes to. For
  // an authenticated cookie-session user it is the per-user namespace resolved
  // from the session (adopted once on first boot, identical on every device, so
  // the project follows the user); otherwise (anonymous / apikey / local-first /
  // demo) it stays the browser-local id. Gated on the SAME signal as authUnlocked
  // (authSession.status), so the namespace is in hand the instant the desk is
  // usable: a run created the moment the composer unlocks already scopes to the
  // namespace (no browser-id window). serverSyncEnabled (which activates the
  // project-data sync hooks) lags one render behind via cookieAuthed, by which
  // point this is already the namespace -- so the sync lifecycle keys on the
  // per-user namespace from its first hydrate and never re-keys from the browser id.
  const effectiveWorkspaceId =
    projectPersistenceAvailable
    && authSession.status === 'authenticated'
    && authSession.projectNamespace
      ? authSession.projectNamespace
      : state.workspaceId
  const {
    cancelRun,
    deleteRun,
    lastError: runError,
    submitRun,
  } = useResearchRunApi({
    apiKey: apiKey.trim() || undefined,
    // Gate on discoveryReady so listing waits for the real auth mode: until
    // health settles, authMode reads as `none` (authUnlocked true) and would
    // list prematurely under the browser id before the namespace resolves.
    canList: discoveryReady && authUnlocked,
    enabled: !isDemoMode,
    onEvent: handleApiEvent,
    onResult: handleApiResult,
    onRunError: handleApiRunError,
    onSummary: handleApiSummary,
    workspaceId: effectiveWorkspaceId,
  })
  // One badge for any API error: discovery probes or run operations.
  const apiError = discoveryError ?? runError
  const singleStackLabel = useMemo(() => resolveSingleStackLabel(health), [health])
  const knowledgeAvailable = !isDemoMode && capabilities?.features.knowledge === true
  const embedCatalog = useMemo<readonly EmbedModelDescriptor[]>(() => {
    const serverCatalog = capabilities?.knowledge?.embedding_catalog
    if (!knowledgeAvailable || !serverCatalog || serverCatalog.length === 0) {
      return EMBED_MODELS
    }
    return serverCatalog.map((entry) => ({
      dims: entry.card?.dims ?? 0,
      id: entry.model_id,
      label: entry.card?.display_name ?? entry.model_id,
      provider: entry.card?.vendor ?? '',
    }))
  }, [capabilities, knowledgeAvailable])
  const serverFilesAvailable = !isDemoMode && capabilities?.features.files === true
  const serverParserAvailable =
    serverFilesAvailable && capabilities?.features.document_parser === true
  const knowledgeSyncOptions = useMemo(
    () => (knowledgeAvailable
      ? {
          apiKey: apiKey.trim() || undefined,
          useFileIngestion: serverParserAvailable,
          workspaceId: effectiveWorkspaceId,
        }
      : null),
    [apiKey, knowledgeAvailable, serverParserAvailable, effectiveWorkspaceId],
  )
  const serverFileUpload = useMemo<ServerFileUpload | undefined>(() => {
    if (!serverFilesAvailable) return undefined
    const options = { apiKey: apiKey.trim() || undefined, workspaceId: effectiveWorkspaceId }
    // Upload the ORIGINAL bytes only — the file appears instantly from the
    // client parse; the MarkItDown upgrade runs in the background (see
    // runServerParse) and again at index time as a fallback.
    return async (file: File) => (await uploadServerFile(file, options)).id
  }, [apiKey, serverFilesAvailable, effectiveWorkspaceId])
  // Client options for the file-library preview (getAsset body + original file
  // download). Gated on the FILES/persistence tier — NOT on knowledge: an
  // original exists whenever it was uploaded to the files server, independent of
  // whether vector/knowledge is enabled. `serverFileId` is the per-asset gate.
  const fileApiOptions = useMemo<ClientOptions | null>(
    () => (!isDemoMode && (serverFilesAvailable || projectPersistenceAvailable)
      ? { apiKey: apiKey.trim() || undefined, workspaceId: effectiveWorkspaceId }
      : null),
    [apiKey, isDemoMode, serverFilesAvailable, projectPersistenceAvailable, effectiveWorkspaceId],
  )
  // Kick off the non-blocking background server parse for freshly-uploaded
  // assets (upgrades the instant client parse to browser-independent MarkItDown
  // text). A no-op without a server parser — assets then stay client-parsed
  // until indexed. Bound once here so chat and library ingest share the wiring.
  const runServerParse = useCallback(
    (assets: FileAssetRecord[]) => {
      if (!serverParserAvailable) return
      const options = { apiKey: apiKey.trim() || undefined, workspaceId: effectiveWorkspaceId }
      scheduleServerParse(assets, {
        fetchText: (fileId) => fetchServerFileText(fileId, options).then((r) => r.text),
        onPending: (assetId) => dispatch({ assetId, pending: true, type: 'setFileAssetParsePending' }),
        onParsed: (assetId, text) => dispatch({ assetId, extractedText: text, type: 'upgradeFileAssetParse' }),
        onFailed: (assetId) => dispatch({ assetId, pending: false, type: 'setFileAssetParsePending' }),
      })
    },
    [apiKey, serverParserAvailable, effectiveWorkspaceId, dispatch],
  )
  const serverFeatureLabels = useMemo<string[] | null>(() => {
    if (isDemoMode || !capabilities) return null
    const features = capabilities.features
    const labels: string[] = []
    if (features.knowledge) labels.push(t.vectorIndex.featureKnowledge)
    if (features.hybrid_retrieval) labels.push(t.vectorIndex.featureHybrid)
    if (features.reranker) labels.push(t.vectorIndex.featureReranker)
    if (features.contextual_retrieval) labels.push(t.vectorIndex.featureContextual)
    if (features.document_parser) labels.push(t.vectorIndex.featureParser)
    if (features.files) labels.push(t.vectorIndex.featureFiles)
    return labels
  }, [capabilities, isDemoMode, t])
  const projectSyncActive = projectPersistenceAvailable && state.serverSyncEnabled
  const {
    error: chatSyncError,
    hasMoreThreads: chatHistoryHasMore,
    isLoadingMore: chatHistoryLoadingMore,
    loadMoreThreads: loadMoreChatHistory,
  } = useChatHistoryApi({
    apiKey: apiKey.trim() || undefined,
    chatThreadGroupMemberships: state.chatThreadGroupMemberships,
    chatThreadGroups: state.chatThreadGroups,
    chatThreads: state.chatThreads,
    dispatch,
    projectEpoch: state.projectEpoch,
    selectedThreadId: state.ui.selectedChatThreadId,
    syncActive: projectSyncActive,
    workspaceId: effectiveWorkspaceId,
  })
  const { error: editorSyncError } = useEditorHistoryApi({
    apiKey: apiKey.trim() || undefined,
    dispatch,
    editorComments: state.editorComments,
    editorDocuments: state.editorDocuments,
    editorFolders: state.editorFolders,
    projectEpoch: state.projectEpoch,
    selectedDocumentId: state.editorUi.activeDocumentId,
    syncActive: projectSyncActive,
    workspaceId: effectiveWorkspaceId,
  })
  const { error: assetSyncError, ensureAssetBodiesLoaded } = useAssetHistoryApi({
    apiKey: apiKey.trim() || undefined,
    dispatch,
    fileAssets: state.fileAssets,
    fileGroups: state.fileGroups,
    fileLibrarySections: state.fileLibrarySections,
    projectEpoch: state.projectEpoch,
    syncActive: projectSyncActive,
    workspaceId: effectiveWorkspaceId,
  })
  const { error: vectorIndexSyncError } = useVectorIndexHistoryApi({
    apiKey: apiKey.trim() || undefined,
    dispatch,
    vectorIndexes: state.vectorIndexes,
    projectEpoch: state.projectEpoch,
    syncActive: projectSyncActive,
    workspaceId: effectiveWorkspaceId,
  })
  const { error: knowledgeSessionSyncError } = useKnowledgeSessionsApi({
    apiKey: apiKey.trim() || undefined,
    dispatch,
    itemOrder: state.knowledgeItemOrder,
    items: state.knowledgeItems,
    projectEpoch: state.projectEpoch,
    selectedSessionId: state.selectedKnowledgeSessionId,
    sessionGroupMemberships: state.knowledgeSessionGroupMemberships,
    sessionGroups: state.knowledgeSessionGroups,
    sessionOrder: state.knowledgeSessionOrder,
    sessions: state.knowledgeSessions,
    syncActive: projectSyncActive,
    workspaceId: effectiveWorkspaceId,
  })
  // Account preferences (theme/locale/contrast) are an ACCOUNT tier, not
  // project data: they sync on a real per-user session (cookieAuthed) + the
  // durable capability, independent of the project's server-sync opt-in, and
  // are NOT part of the project import. "Account wins on login".
  const accountSyncActive = projectPersistenceAvailable && cookieAuthed
  const { error: accountPreferencesError } = useAccountPreferences({
    apiKey: apiKey.trim() || undefined,
    applyPreferences: applyProjectPreferences,
    preferences: currentPreferences,
    syncActive: accountSyncActive,
    workspaceId: effectiveWorkspaceId,
  })
  const { importPending: projectImportPending, importToServer } =
    useProjectServerImport({
      apiKey: apiKey.trim() || undefined,
      canPersist: projectPersistenceAvailable,
      dispatch,
      state,
      workspaceId: effectiveWorkspaceId,
    })
  // Reports loaded from a project file are pushed to the durable run tier so
  // they survive reload + follow the user (the runs analogue of the project
  // import-up), gated on the same server-sync activation as the project hooks.
  const { error: runImportError } = useResearchRunImport({
    apiKey: apiKey.trim() || undefined,
    enabled: projectSyncActive,
    researchRuns: state.researchRuns,
    researchRunOrder: state.researchRunOrder,
    workspaceId: effectiveWorkspaceId,
  })
  // Any entity's background autosave failure surfaces on the one badge.
  const serverSyncError =
    chatSyncError ?? editorSyncError ?? assetSyncError ?? vectorIndexSyncError
    ?? knowledgeSessionSyncError ?? accountPreferencesError ?? runImportError
  const [selectedKnowledgeIndexIds, setSelectedKnowledgeIndexIds] = useState<string[]>([])
  const knowledgeIndexOptions = useMemo<KnowledgeIndexOption[] | null>(() => {
    if (!knowledgeAvailable) return null
    return projectVectorIndexes(state)
      .filter((index) => index.status === 'ready' && index.serverCollectionId)
      .map((index) => ({
        collectionId: index.serverCollectionId as string,
        id: index.id,
        title: index.title,
      }))
  }, [knowledgeAvailable, state.vectorIndexOrder, state.vectorIndexes])
  useEffect(() => {
    if (!knowledgeIndexOptions) {
      if (selectedKnowledgeIndexIds.length > 0) setSelectedKnowledgeIndexIds([])
      return
    }
    const available = new Set(knowledgeIndexOptions.map((option) => option.id))
    if (selectedKnowledgeIndexIds.some((id) => !available.has(id))) {
      setSelectedKnowledgeIndexIds((current) => current.filter((id) => available.has(id)))
    }
  }, [knowledgeIndexOptions, selectedKnowledgeIndexIds])

  // --- "Wissen" workspace (knowledge Q&A + Finden + reader) ----------------
  // Run-tracking wiring choice: knowledge asks are native runs submitted
  // through the EXISTING useResearchRunApi machinery (createResearchRun +
  // SSE stream + result fetch). The reducer routes the same
  // appendApiRunEvent / attachApiRunResult / markApiRunError actions into
  // the knowledgeItems thread slice by run id, so no second event pipeline
  // exists; the only additions are `select: false` (a knowledge ask must
  // not hijack the research workspace selection) and the knowledge-mode
  // filter in projectResearchJobs (the thread is its surface, not a job
  // card). Demo asks reuse the identical event path with synthetic events.
  const knowledgeWorkspaceVisible = isDemoMode || capabilities?.features.knowledge === true
  const [knowledgeMode, setKnowledgeMode] = useState<KnowledgeMode>('ask')
  const [knowledgeQuestion, setKnowledgeQuestion] = useState('')
  const [knowledgeCollectionIds, setKnowledgeCollectionIds] = useState<string[]>([])
  const [knowledgeProfileId, setKnowledgeProfileId] = useState<string | null>(null)
  const [knowledgeTopK, setKnowledgeTopK] = useState<number | null>(null)
  const [knowledgeAskError, setKnowledgeAskError] = useState<string | null>(null)
  const knowledgeDemoTimeoutsRef = useRef<number[]>([])
  const knowledgeCollections = useMemo<KnowledgeCollectionOption[]>(() => {
    if (!knowledgeWorkspaceVisible) return []
    return projectVectorIndexes(state)
      .filter((index) => isDemoMode || index.status === 'ready')
      .filter((index) => isDemoMode || Boolean(index.serverCollectionId))
      .map((index) => ({
        collectionId: isDemoMode ? index.id : (index.serverCollectionId as string),
        id: index.id,
        title: index.title,
      }))
  }, [isDemoMode, knowledgeWorkspaceVisible, state])
  useEffect(() => {
    const available = new Set(knowledgeCollections.map((option) => option.id))
    setKnowledgeCollectionIds((current) => (
      current.some((id) => !available.has(id))
        ? current.filter((id) => available.has(id))
        : current
    ))
  }, [knowledgeCollections])
  const knowledgeProfileOptions = useMemo(
    () => knowledgeProfileOptionsFromManifest(
      isDemoMode ? DEMO_KNOWLEDGE_PROFILE_MANIFEST : capabilities?.knowledge?.profiles,
    ),
    [capabilities, isDemoMode],
  )
  const knowledgeDefaultProfileId = isDemoMode
    ? DEMO_KNOWLEDGE_DEFAULT_PROFILE
    : capabilities?.knowledge?.default_profile ?? null
  const knowledgeDefaultTopK = isDemoMode
    ? DEMO_KNOWLEDGE_DEFAULT_TOP_K
    : capabilities?.knowledge?.default_top_k ?? DEMO_KNOWLEDGE_DEFAULT_TOP_K
  const knowledgeItems = useMemo(
    () => projectKnowledgeItems(state),
    [state.knowledgeItemOrder, state.knowledgeItems, state.selectedKnowledgeSessionId],
  )
  const knowledgeAllItems = useMemo(
    () => projectAllKnowledgeItems(state),
    [state.knowledgeItemOrder, state.knowledgeItems],
  )
  const knowledgeSessions = useMemo(
    () => projectKnowledgeSessions(state),
    [state.knowledgeSessionOrder, state.knowledgeSessions],
  )
  const knowledgeSessionSections = useMemo(
    () => projectKnowledgeSessionSections(state),
    [
      state.knowledgeSessionGroupMemberships,
      state.knowledgeSessionGroupOrder,
      state.knowledgeSessionGroups,
      state.knowledgeSessionOrder,
      state.knowledgeSessions,
    ],
  )
  const isKnowledgeAskRunning = knowledgeAllItems.some((item) => item.status === 'running')
  const knowledgeDataSource = useMemo<KnowledgeDataSource>(() => {
    if (isDemoMode) return createDemoKnowledgeDataSource()
    const clientOptions = { apiKey: apiKey.trim() || undefined, workspaceId: effectiveWorkspaceId }
    return {
      loadDocumentText: (documentId) => fetchKnowledgeDocumentText(documentId, clientOptions),
      // An original file exists whenever it was uploaded to the files server —
      // independent of the knowledge `features.files` flag — so gate this the
      // SAME way as the file-library preview (which works on the persistence
      // tier). Gating on serverFilesAvailable alone hid the "Quelle" PDF tab on
      // deployments that have persistence but not the files capability.
      loadFileContent: serverFilesAvailable || projectPersistenceAvailable
        ? (fileId) => fetchServerFileContent(fileId, clientOptions)
        : null,
      search: (query, collectionIds, topK) =>
        searchKnowledge({ collectionIds, query, topK }, clientOptions),
    }
  }, [apiKey, isDemoMode, serverFilesAvailable, projectPersistenceAvailable, effectiveWorkspaceId])

  // Falls back to research when the knowledge capability disappears
  // (e.g. backend switch) while the view is active.
  useEffect(() => {
    if (state.ui.activeView === 'knowledge' && !knowledgeWorkspaceVisible) {
      dispatch({ type: 'setActiveView', view: 'research' })
    }
  }, [knowledgeWorkspaceVisible, state.ui.activeView])

  useEffect(() => {
    const timeouts = knowledgeDemoTimeoutsRef.current
    return () => {
      for (const timeoutId of timeouts) window.clearTimeout(timeoutId)
    }
  }, [])

  async function handleKnowledgeAsk(
    question: string,
    options: {
      collectionIds?: string[]
      profileId?: string | null
      topK?: number | null
    } = {},
  ) {
    if (isKnowledgeAskRunning) return
    if (!isDemoMode && !authUnlocked) return
    const selectedIds = options.collectionIds ?? knowledgeCollectionIds
    const selectedProfileId = options.profileId ?? knowledgeProfileId
    const selectedTopK = options.topK ?? knowledgeTopK
    const selected = knowledgeCollections.filter((collection) =>
      selectedIds.includes(collection.id))
    if (selected.length === 0) {
      setKnowledgeAskError(t.knowledge.collectionsRequired)
      return
    }
    setKnowledgeAskError(null)

    const collectionTitles = selected.map((collection) => collection.title)
    const backendCollectionIds = selected.map((collection) => collection.collectionId)
    const sessionId =
      state.selectedKnowledgeSessionId
      ?? state.knowledgeSessionOrder[0]
      ?? createClientId('ks')
    const buildItem = (runId: string): KnowledgeThreadItemRecord => ({
      collectionTitles,
      createdAt: new Date().toISOString(),
      id: createClientId('kn'),
      progress: { steps: [] },
      question,
      requestedProfile: selectedProfileId,
      runId,
      sessionId,
      status: 'running',
    })

    if (isDemoMode) {
      const runId = createClientId('kn-demo')
      dispatch({ item: buildItem(runId), type: 'startKnowledgeAsk' })
      const script = buildDemoAskScript(runId)
      let elapsed = 0
      for (const step of script.steps) {
        elapsed += step.delayMs
        knowledgeDemoTimeoutsRef.current.push(window.setTimeout(() => {
          dispatch({ event: step.event, type: 'appendApiRunEvent' })
        }, elapsed))
      }
      knowledgeDemoTimeoutsRef.current.push(window.setTimeout(() => {
        dispatch({ answer: script.answer, runId, type: 'completeKnowledgeItem' })
      }, elapsed + script.completeAfterMs))
      return
    }

    const summary = await submitRun(
      {
        knowledgeFilters: {
          collectionIds: backendCollectionIds,
          ...(selectedProfileId ? { profile: selectedProfileId } : {}),
          ...(selectedTopK ? { topK: selectedTopK } : {}),
        },
        mode: 'knowledge',
        question,
      },
      {
        onCreated: (created) => {
          dispatch({ item: buildItem(created.run_id), type: 'startKnowledgeAsk' })
        },
        select: false,
      },
    )
    if (!summary) {
      setKnowledgeAskError(t.knowledge.askFailed)
    }
  }

  function handleKnowledgeDemoAsk() {
    if (!isDemoMode || isKnowledgeAskRunning) return
    const demoCollection =
      knowledgeCollections.find((collection) => collection.id === 'vector-index-eu-recht')
      ?? knowledgeCollections[0]
      ?? null
    if (!demoCollection) {
      setKnowledgeAskError(t.knowledge.collectionsRequired)
      return
    }
    setKnowledgeCollectionIds([demoCollection.id])
    setKnowledgeProfileId('tief')
    setKnowledgeTopK(DEMO_KNOWLEDGE_DEFAULT_TOP_K)
    void handleKnowledgeAsk(t.knowledge.demoSearchQuestion, {
      collectionIds: [demoCollection.id],
      profileId: 'tief',
      topK: DEMO_KNOWLEDGE_DEFAULT_TOP_K,
    })
  }

  function createKnowledgeSession(title = t.knowledge.newSession): KnowledgeSessionRecord {
    const now = new Date().toISOString()
    return {
      createdAt: now,
      id: createClientId('ks'),
      title,
      updatedAt: now,
    }
  }
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
  const [settingsRequestedSection, setSettingsRequestedSection] =
    useState<'security' | null>(null)
  // Feed the cookie session's authenticated state back to the run-list hook
  // (declared above) so an in-app local/ldap login re-hydrates prior runs.
  useEffect(() => {
    setCookieAuthed(isCookieMode && authSession.status === 'authenticated')
  }, [isCookieMode, authSession.status])
  // Server-first persistence for an authenticated (cookie-session) user: derive
  // serverSyncEnabled from the live session + durable capability instead of the
  // manual import button, so the project auto-hydrates on every boot/reload and
  // auto-saves with no opt-in. Drives the flag both ways: a login enables sync
  // (the empty boot state then hydrates from the server), a logout disables it
  // (the hooks reset to empty so the prior user's data is not shown). persistLocal
  // is false: the flag is session-derived, never written to the local manifest or
  // marked dirty. ONLY the cookie tier (local/oidc/ldap = a real per-user
  // identity); the apikey / local-first / demo tiers keep the manual opt-in
  // (this effect no-ops outside cookie mode), so their behaviour is unchanged.
  useEffect(() => {
    if (!isCookieMode) return
    const shouldSync = projectPersistenceAvailable && cookieAuthed
    if (state.serverSyncEnabled !== shouldSync) {
      dispatch({ enabled: shouldSync, persistLocal: false, type: 'setServerSyncEnabled' })
    }
  }, [isCookieMode, projectPersistenceAvailable, cookieAuthed, state.serverSyncEnabled])
  // No lock flash while the very first session probe is in flight.
  const isAuthLocked =
    !isDemoMode
    && !authUnlocked
    && (!isCookieMode || authSession.status !== 'unknown')
  const canImproveText = !isDemoMode && authUnlocked
  // Sharing exists on the oidc surface with a live session; the capability
  // flag keeps none/apikey deployments byte-identical. Demo mode simulates it
  // from seeded data so the feature is visible offline (like the quota meter).
  const sharingEnabled =
    isDemoMode
    || (capabilities?.features.sharing === true
      && isCookieMode
      && authSession.status === 'authenticated')
  // The quota meter follows the same cookie-session + capability gate;
  // demo mode shows seeded figures so the feature is visible offline.
  const quotaMeterEnabled =
    isDemoMode
    || (capabilities?.features.quota === true
      && isCookieMode
      && authSession.status === 'authenticated')
  const [shareTarget, setShareTarget] = useState<
    { resourceId: string; resourceType: string; title: string } | null
  >(null)
  const ownApiRunIds = useMemo(
    () => (sharingEnabled
      ? allJobs.filter((job) => !job.access).map((job) => job.id)
      : []),
    [allJobs, sharingEnabled],
  )
  const { counts: shareCountByRunId, refresh: refreshShareCounts } =
    useOutgoingShareCounts('run', ownApiRunIds, sharingEnabled, isDemoMode)
  const { byResourceId: sharedWithMeByRunId } = useSharedWithMe(
    'run',
    sharingEnabled,
    isDemoMode,
  )
  // Prompt templates persist server-side whenever the capability is
  // live and the caller is unlocked (works in apikey/none too);
  // demo mode stays browser-local.
  const templatesEnabled =
    !isDemoMode
    && capabilities?.features.prompt_templates === true
    && authUnlocked
  const templateSync = useTemplateSync({
    dispatch,
    enabled: templatesEnabled,
    localRules: chatRules,
  })
  const sharedByLabelByRunId = useMemo(() => {
    const labels = new Map<string, string>()
    for (const [runId, entry] of sharedWithMeByRunId) {
      labels.set(
        runId,
        personLabel(entry.granted_by_display_name, null, entry.granted_by_sub),
      )
    }
    return labels
  }, [sharedWithMeByRunId])

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

  async function handleImportProjectToServer() {
    // The import is the one user-facing chat-sync action that runs BEFORE
    // the project is server-synced, so the "Synced" badge (which carries
    // sync errors) is not yet mounted to show a failure. Surface it through
    // the same red banner as the other project actions (No Silent Fallbacks).
    setProjectActionError(null)
    try {
      await importToServer()
    } catch (error) {
      reportProjectActionError(error)
    }
  }

  async function handleLoadProject() {
    setProjectActionError(null)
    try {
      const nextState = await loadProject({ onWorkStart: () => setProjectAction('load') })
      abortAllChatRequests()
      // Account preferences (theme/locale/contrast) are an account tier, not
      // project data: while a real per-user session drives them, a loaded
      // project file must NOT bleed its embedded prefs into the live theme
      // (which the account autosave would then PUT, clobbering the account
      // from an unrelated file). Project-embedded prefs apply only offline /
      // when there is no account sync (the local-first case). M6c.
      if (!accountSyncActive) applyProjectPreferences(nextState.preferences)
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
    if (!isDemoMode && !authUnlocked) return

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
    if (!isDemoMode && !authUnlocked) return

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
    // Guarantee any attached file-asset bodies are loaded before they are read
    // synchronously into the outgoing attachments (M6c load-on-use). The
    // prefetch on attach usually made this instant; the returned map overrides
    // the state snapshot, which the just-dispatched bodies have not reached. A
    // failed fetch must surface (No-Silent-Fallbacks) and abort the send rather
    // than ship an empty attachment — nothing has been mutated yet, so the
    // draft is preserved and the user can retry.
    let assetBodies: Map<string, string>
    try {
      assetBodies = await ensureAssetBodiesLoaded(
        assetIdsFromChatRefs(state, messageAttachmentRefs),
      )
    } catch (error) {
      setChatErrorByThreadId((current) => ({
        ...current,
        [threadId]: `${t.chat.requestFailed}: ${messageFromError(error)}`,
      }))
      return
    }
    const messageAttachments = chatAttachmentsFromRefs(
      state,
      messageAttachmentRefs,
      assetBodies,
    )
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

    const knowledgeCollectionIds = options.knowledgeCollectionIds ?? []
    const chainTemplates = state.ui.chatChainingEnabled && knowledgeCollectionIds.length === 0
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
          assetBodies,
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
        knowledgeCollectionIds,
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
        const stepTimeout = globalThis.setTimeout(() => controller.abort(), chatStepTimeoutMsRef.current)
        try {
          if (isFinal && useStreaming) {
            chatStreamContentByThreadIdRef.current.set(threadId, '')
            await streamChatCompletion(
              { ...baseStepRequest, stream: true },
              {
                apiKey: apiKey.trim() || undefined,
                signal: controller.signal,
                workspaceId: effectiveWorkspaceId,
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
                workspaceId: effectiveWorkspaceId,
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
    knowledgeCollectionIds = [],
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
    knowledgeCollectionIds?: string[]
    requestMessages: ChatCompletionMessage[]
    stack?: string
    threadId: string
    useStreaming: boolean
  }) {
    const knowledgeMode = knowledgeCollectionIds.length > 0
    const baseChatRequest = {
      agentOverrides: chatAgentOverrides(modelTier, model, effort),
      includeProgress: false,
      ...(knowledgeMode
        ? { knowledgeFilters: { collectionIds: knowledgeCollectionIds } }
        : {}),
      messages: requestMessages,
      mode: (knowledgeMode ? 'knowledge' : 'direct_llm') as ResearchRunMode,
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
          workspaceId: effectiveWorkspaceId,
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

      // Knowledge mode answers non-streaming: the backend rejects
      // stream=true for retrieval algorithms until streaming
      // dispatches through the registry.
      if (useStreaming && !knowledgeMode) {
        await streamChatCompletion(
          {
            ...baseChatRequest,
            stream: true,
          },
          {
            apiKey: apiKey.trim() || undefined,
            signal: controller.signal,
            workspaceId: effectiveWorkspaceId,
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
    if (!isDemoMode && !authUnlocked) return

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

  async function handleCredentialLogin() {
    const identifier = authIdentifier.trim()
    if (!identifier || !authPassword) {
      setAuthLockError(t.authLock.credentialRequired)
      return
    }
    setIsAuthSubmitting(true)
    setAuthLockError(null)
    try {
      const credentials = { identifier, password: authPassword }
      await (authMode === 'ldap'
        ? loginLdap(credentials)
        : loginLocal(credentials))
      setAuthPassword('')
      // The login set the session cookie; re-probe so the lock lifts.
      await refreshAuthSession()
    } catch (error) {
      setAuthLockError(
        hasHttpStatus(error, 429)
          ? t.authLock.credentialRateLimited
          : hasHttpStatus(error, 401) || hasHttpStatus(error, 403)
            ? t.authLock.credentialRejected
            : t.authLock.credentialCheckFailed,
      )
    } finally {
      setIsAuthSubmitting(false)
    }
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

  async function handleDeleteJob(runId: string) {
    const run = state.researchRuns[runId]
    if (!run) return
    // API and imported runs are durable on the server; deleting them locally
    // alone lets a reload re-hydrate them from the store (the "deleted report
    // comes back" bug). Mock/demo runs and the no-server case have no durable
    // record, so they delete locally only. The server delete must succeed
    // BEFORE the local removal — except a 404, which deleteRun treats as
    // already-gone, so it resolves and the local removal proceeds.
    const serverBacked = !isDemoMode
      && (run.source === 'api' || run.source === 'imported')
      && discoveryReady
      && authUnlocked
    if (!serverBacked) {
      dispatch({ jobId: runId, type: 'deleteJob' })
      return
    }
    try {
      await deleteRun(runId)
      dispatch({ jobId: runId, type: 'deleteJob' })
    } catch {
      // A real failure (e.g. 409 while still active) is surfaced via runError
      // -> the apiError banner; keep the run in the list so it stays visible.
    }
  }

  return (
    <QuotaMeterProvider demo={isDemoMode} enabled={Boolean(quotaMeterEnabled)}>
    <main className="min-h-svh bg-canvas text-foreground lg:flex lg:h-svh lg:flex-col lg:overflow-hidden">
      <Topbar
        activeView={state.ui.activeView}
        canPersistProject={projectPersistenceAvailable}
        dirty={state.dirty}
        importPending={projectImportPending}
        isProjectActionPending={projectAction !== null}
        onDismissProjectActionError={() => setProjectActionError(null)}
        onExportProject={() => void handleExportProject()}
        onImportProjectToServer={() => void handleImportProjectToServer()}
        onLoadProject={() => void handleLoadProject()}
        onSaveProject={() => void handleSaveProject()}
        projectActionError={projectActionError}
        projectConnection={state.connection}
        projectName={state.project.name}
        serverSyncEnabled={state.serverSyncEnabled}
        serverSyncError={serverSyncError}
      />
      <div className="flex min-h-0 w-full flex-1">
        <AppRail
          activeView={state.ui.activeView}
          onViewChange={(view) => {
            setSettingsRequestedSection(null)
            dispatch({ type: 'setActiveView', view })
          }}
          showKnowledge={knowledgeWorkspaceVisible}
          profileSlot={
            <ProfileAvatar
              authMode={authMode}
              isDemo={isDemoMode}
              session={authSession}
              onLogin={ssoLogin}
              onLogout={() => void ssoLogout()}
              onOpenSecuritySettings={() => {
                setSettingsRequestedSection('security')
                dispatch({ type: 'setActiveView', view: 'settings' })
              }}
            />
          }
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
              researchQuestion={researchQuestion}
              onResearchQuestionChange={setResearchQuestion}
              onComposerVisibleChange={(isVisible) => dispatch({
                isVisible,
                type: 'setComposerVisible',
              })}
              onDeleteJob={(jobId) => void handleDeleteJob(jobId)}
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
              onShareJob={sharingEnabled
                ? (jobId) => setShareTarget({
                  resourceId: jobId,
                  resourceType: 'run',
                  title: state.researchRuns[jobId]?.summary.title ?? '',
                })
                : undefined}
              reduceMotion={reduceMotion}
              selectedJobId={state.ui.selectedJobId}
              selectedRun={selectedRun}
              selectedStack={displayedSelectedStack}
              shareCountByRunId={shareCountByRunId}
              sharedByLabelByRunId={sharedByLabelByRunId}
            />
          ) : state.ui.activeView === 'chat' ? (
            <ChatWorkspace
              activeAssistantMessageId={activeChatRequest?.assistantMessageId ?? null}
              chatModelOptions={chatModelOptions}
              chatModelOptionsStatus={chatModelOptionsState.status}
              chatHistorySections={chatHistorySections}
              chatHistoryHasMore={chatHistoryHasMore}
              chatHistoryLoadingMore={chatHistoryLoadingMore}
              onLoadMoreChatHistory={loadMoreChatHistory}
              defaultChatModel={defaultChatModel}
              isDesktop={isDesktop}
              isHistoryVisible={state.ui.isChatHistoryVisible}
              isIncognito={isIncognitoChat}
              isSending={activeChatRequest !== null}
              onAttachContext={(ref) => {
                dispatch({ ref, type: 'attachChatContextToDraft' })
                // Prefetch the file body in the background so it is already in
                // hand when the message is sent (M6c load-on-use); the send
                // guard awaits it regardless if this has not finished, so a
                // prefetch failure here is intentionally best-effort.
                void ensureAssetBodiesLoaded(assetIdsFromChatRefs(state, [ref])).catch(() => {})
              }}
              onAttachFiles={(files) => void handleAttachChatFiles(files)}
              onPillRefsChange={setChatPillRefs}
              chatDraft={chatDraft}
              onChatDraftChange={setChatDraft}
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
              knowledgeIndexOptions={knowledgeIndexOptions}
              selectedKnowledgeIndexIds={selectedKnowledgeIndexIds}
              onSelectedKnowledgeIndexIdsChange={setSelectedKnowledgeIndexIds}
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
                workspaceId: effectiveWorkspaceId,
              }}
              threads={chatThreads}
            />
          ) : state.ui.activeView === 'editor' ? (
            <EditorWorkspace
              apiKey={apiKey.trim() || undefined}
              capabilities={capabilities}
              chatModelOptions={chatModelOptions}
              chatModelOptionsStatus={chatModelOptionsState.status}
              chatModelCatalog={chatModelCatalog}
              defaultChatModel={defaultChatModel}
              dispatch={dispatch}
              ensureAssetBodiesLoaded={ensureAssetBodiesLoaded}
              reportOptions={reportOptions}
              selectedModelTier={state.ui.selectedChatModelTier}
              state={state}
              textImprovement={{
                apiKey: apiKey.trim() || undefined,
                enabled: canImproveText,
                selectedStack: textImprovementStack,
                workspaceId: effectiveWorkspaceId,
              }}
            />
          ) : state.ui.activeView === 'knowledge' ? (
            <KnowledgeWorkspace
              collections={knowledgeCollections}
              composerNotice={knowledgeAskError}
              dataSource={knowledgeDataSource}
              defaultProfileId={knowledgeDefaultProfileId}
              defaultTopK={knowledgeDefaultTopK}
              historyItems={knowledgeAllItems}
              isAskDisabled={!isDemoMode && !authUnlocked}
              isAskRunning={isKnowledgeAskRunning}
              isHistoryVisible={state.ui.isKnowledgeHistoryVisible}
              items={knowledgeItems}
              sessionSections={knowledgeSessionSections}
              sessions={knowledgeSessions}
              mode={knowledgeMode}
              onCreateSession={(groupId) => {
                const session = createKnowledgeSession()
                dispatch({ session, type: 'createKnowledgeSession' })
                if (groupId) {
                  dispatch({ groupId, sessionId: session.id, targetIndex: 0, type: 'moveKnowledgeSessionToGroup' })
                }
              }}
              onCreateSessionGroup={() => dispatch({ title: t.knowledge.newFolder, type: 'createKnowledgeSessionGroup' })}
              onDeleteSessionGroup={(groupId) => dispatch({ groupId, type: 'deleteKnowledgeSessionGroup' })}
              onDeleteSession={(sessionId) => dispatch({ sessionId, type: 'deleteKnowledgeSession' })}
              onDemoAsk={isDemoMode ? handleKnowledgeDemoAsk : undefined}
              onHistoryVisibleChange={(isVisible) => dispatch({ isVisible, type: 'setKnowledgeHistoryVisible' })}
              onAsk={(question) => void handleKnowledgeAsk(question)}
              knowledgeQuestion={knowledgeQuestion}
              onKnowledgeQuestionChange={setKnowledgeQuestion}
              onModeChange={setKnowledgeMode}
              onMoveSessionGroup={(groupId, targetIndex) => dispatch({ groupId, targetIndex, type: 'moveKnowledgeSessionGroup' })}
              onMoveSessionToGroup={(sessionId, groupId, targetIndex) =>
                dispatch({ groupId, sessionId, targetIndex, type: 'moveKnowledgeSessionToGroup' })}
              onProfileChange={setKnowledgeProfileId}
              onRenameSessionGroup={(groupId, title) => dispatch({ groupId, title, type: 'renameKnowledgeSessionGroup' })}
              onRenameSession={(sessionId, title) => dispatch({ sessionId, title, type: 'renameKnowledgeSession' })}
              onSelectSession={(sessionId) => dispatch({ sessionId, type: 'selectKnowledgeSession' })}
              onSelectedCollectionIdsChange={setKnowledgeCollectionIds}
              onTopKChange={setKnowledgeTopK}
              profileId={knowledgeProfileId}
              profileOptions={knowledgeProfileOptions}
              selectedCollectionIds={knowledgeCollectionIds}
              selectedSessionId={state.selectedKnowledgeSessionId}
              topK={knowledgeTopK}
            />
          ) : state.ui.activeView === 'prompt-library' ? (
            <PromptLibraryWorkspace
              dispatch={dispatch}
              sharing={sharingEnabled
                ? {
                  onShareRule: (rule) => {
                    if (!rule.serverTemplateId) return
                    setShareTarget({
                      resourceId: rule.serverTemplateId,
                      resourceType: 'prompt_template',
                      title: rule.title,
                    })
                  },
                }
                : null}
              state={state}
              templateSync={templateSync}
              textImprovement={{
                apiKey: apiKey.trim() || undefined,
                enabled: canImproveText,
                selectedStack: textImprovementStack,
                workspaceId: effectiveWorkspaceId,
              }}
            />
          ) : state.ui.activeView === 'database' ? (
            <FileLibraryWorkspace
              dispatch={dispatch}
              embedModels={embedCatalog}
              ensureAssetBodiesLoaded={ensureAssetBodiesLoaded}
              fileApiOptions={fileApiOptions}
              knowledgeSync={knowledgeSyncOptions}
              onAssetsIngested={runServerParse}
              serverFeatureLabels={serverFeatureLabels}
              serverFileUpload={serverFileUpload}
              state={state}
            />
          ) : (
            <SettingsWorkspace
              apiCapabilities={capabilities}
              apiError={apiError}
              apiHealth={health}
              apiKey={apiKeyDraft}
              authMode={authMode}
              authSession={authSession}
              patAvailable={authConfig?.pat_available}
              onSsoLogin={ssoLogin}
              onSsoLogout={() => void ssoLogout()}
              isDemoMode={isDemoMode}
              onApiKeyChange={handleSettingsApiKeyChange}
              onDemoModeChange={(enabled) => dispatch({ enabled, type: 'setDemoMode' })}
              onStackChange={(stack) => dispatch({ stack, type: 'setSelectedStack' })}
              reduceMotion={reduceMotion}
              selectedStack={displayedSelectedStack}
              requestedSection={settingsRequestedSection}
              stackDiscoveryStatus={stackDiscoveryStatus}
              stackOptions={effectiveStackOptions}
            />
          )}
        </div>
      </div>
      {sharingEnabled && shareTarget && (
        <ShareDialog
          demo={isDemoMode}
          onChanged={() => void refreshShareCounts()}
          onClose={() => setShareTarget(null)}
          ownerEmail={isDemoMode ? DEMO_OWNER.email : authSession.email}
          ownerName={isDemoMode ? DEMO_OWNER.displayName : authSession.displayName}
          resourceId={shareTarget.resourceId}
          resourceTitle={shareTarget.title}
          resourceType={shareTarget.resourceType}
        />
      )}
      {isAuthLocked && (
        <AuthLockScreen
          error={authLockError}
          identifier={authIdentifier}
          isSubmitting={isAuthSubmitting}
          mode={
            authMode === 'oidc'
              ? 'sso'
              : authMode === 'local'
                ? 'local'
                : authMode === 'ldap'
                  ? 'ldap'
                  : 'apikey'
          }
          onCredentialSubmit={() => void handleCredentialLogin()}
          onIdentifierChange={(value) => {
            setAuthIdentifier(value)
            setAuthLockError(null)
          }}
          onPasswordChange={(value) => {
            setAuthPassword(value)
            setAuthLockError(null)
          }}
          onSsoLogin={ssoLogin}
          providerName={authConfig?.provider_name}
          onSubmit={(token) => void handleAuthUnlock(token)}
          onTokenChange={(token) => {
            setApiKeyDraft(token)
            setAuthLockError(null)
          }}
          password={authPassword}
          reduceMotion={reduceMotion}
          token={apiKeyDraft}
        />
      )}
    </main>
    </QuotaMeterProvider>
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
