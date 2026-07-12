import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import type { Dispatch } from 'react'
import { useReducedMotion } from 'motion/react'

import type { ClientOptions } from '@/api/inqtrixClient'
import {
  BookOpen,
  FileText,
  Folder,
  Globe2,
  ListChecks,
  PenLine,
  Plus,
  Repeat2,
  Waypoints,
} from '@/components/icons'
import type { LucideIcon } from '@/components/icons'
import {
  AnimatedPanelBody,
  AnimatedResizableHandle,
} from '@/components/ui/animated-panel'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  OptionMenuItem,
  optionMenuContentClassName,
} from '@/components/ui/option-menu'
import { useAnimatedResizablePanelCollapse } from '@/components/ui/animated-panel-motion'
import { PanelToggle } from '@/components/ui/panel-toggle'
import {
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable'
import { ResponsiveSidePanel } from '@/components/ui/responsive-side-panel'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useMarkdownCodePreload } from '@/components/markdown/useMarkdownCodePreload'
import { ConversationSkeleton } from '@/components/ui/conversation-skeleton'
import { WelcomeState } from '@/components/ui/welcome-state'
import { CanvasHost } from '@/features/canvas/CanvasHost'
import { useCanvasFollow } from '@/features/canvas/useCanvasFollow'
import {
  activeCanvasView,
  type CanvasViewDescriptor,
} from '@/features/canvas/types'
import type { ProjectState } from '@/features/project/types'
import type {
  CreateResearchRunRequest,
  InqtrixCapabilities,
} from '@/features/researchRuns/types'
import type { ResearchDeskAction } from '@/features/researchDesk/state'
import { useMediaQuery } from '@/features/researchDesk/hooks/useMediaQuery'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import {
  AgentComposer,
  type AgentCollectionOption,
  type AgentComposerSubmit,
  type AgentDocumentOption,
  type AgentResponseForm,
} from './AgentComposer'
import { AgentSessionRail } from './AgentSessionRail'
import { ComposerGateTray, pendingGate } from './ComposerGateTray'
import { AGENT_CANVAS_REGISTRY } from './canvas/views'
import {
  AgentCanvasReactContext,
  type AgentCanvasContextValue,
} from './canvas/context'
import { routeAgentRunToView } from './followTarget'
import { agentOverridesFromSelection } from '@/features/researchRuns/modelSelection'
import {
  isActiveAgentRun,
  restoredAgentSessionId,
  type AgentRunRecord,
} from './model'
import {
  vectorBackendDisplay,
  type PlanSourceInfo,
} from './plan/sourceLabel'
import {
  DEMO_AGENT_OVERVIEW_SOURCE,
  DEMO_AGENT_TIERS,
  type AgentDemoActions,
} from './demo'
import {
  buildAgentOverview,
  countAgentToolUse,
  defaultEngineMode,
  type AgentEngineMode,
} from './agentStatusOverview'
import {
  DEFAULT_AGENT_SOURCE_POLICY,
  normalizeAgentExecutionSnapshot,
  type AgentExecutionDirective,
  type AgentSourcePolicy,
} from './executionPolicy'
import { AgentRunTurn, type AgentTimelineActions } from './timeline/AgentTimeline'
import { useAgentControlApi } from './useAgentControlApi'
import { useAgentSessionsApi } from './useAgentSessionsApi'

const SESSIONS_PANEL_ID = 'agent-sessions-panel'
const CANVAS_PANEL_ID = 'agent-canvas-panel'
const AGENT_CENTER_PANEL_ID = 'agent-center-panel'
const AGENT_TIMELINE_PANEL_ID = 'agent-timeline-panel'

/**
 * The Agent Desk (plan §5): session rail | timeline (command spine with the
 * docked composer) | polymorphic canvas. Follow-the-agent drives the canvas
 * from run state; every manual navigation pins. Focus mode collapses the
 * timeline and gives the canvas the full center.
 */
export function AgentWorkspace({
  apiKey,
  cancelRun,
  deleteRun,
  canvasPanelSize,
  capabilities,
  collections,
  dispatch,
  pollingRunIds,
  runsHydrated = true,
  documents = [],
  draftQuestion,
  memoryEnabled = false,
  skillsApi = null,
  onCanvasPanelSizeChange,
  onDraftQuestionChange,
  onSelectedCollectionIdsChange,
  onSelectedDocumentIdChange,
  onSessionsPanelSizeChange,
  selectedAutonomy,
  onAutonomyChange,
  selectedDepth = 'normal',
  onDepthChange,
  selectedTier = null,
  onTierChange,
  modelCatalog = [],
  modelOptions = [],
  modelOptionsStatus = 'missing',
  defaultChatModel = null,
  selectedCollectionIds,
  selectedDocumentId,
  serverEnabled,
  demo = null,
  state,
  submitRun,
  workspaceId,
}: {
  apiKey: string | undefined
  cancelRun: (runId: string) => Promise<unknown> | void
  /** Runs on the polling fallback (plan M1 T1) — shown as a visible
   * degradation hint on their live status line. */
  pollingRunIds?: string[]
  /** Initial run-list hydration has settled (success or error). While
   * false, a selected session with no runs shows a loading skeleton
   * instead of the welcome state — the transcript may still be paging in. */
  runsHydrated?: boolean
  /** Deletes one durable run (cancel-then-delete on 409). Session delete
   * removes its runs server-side too — otherwise their `session_id`
   * resurrects the session on the next hydration. */
  deleteRun: (runId: string) => Promise<unknown> | void
  canvasPanelSize: number
  capabilities: InqtrixCapabilities | null
  collections: AgentCollectionOption[]
  dispatch: Dispatch<ResearchDeskAction>
  /** Patchable editor documents (server-synced or demo). */
  documents?: AgentDocumentOption[]
  draftQuestion: string
  /** Account preference `enable_agent_memory` — the run overview shows it. */
  memoryEnabled?: boolean
  /** Skill library handle (plan M3); null = feature off. */
  skillsApi?: import('@/features/skills/useSkillsApi').SkillsApiHandle | null
  onCanvasPanelSizeChange: (size: number) => void
  onDraftQuestionChange: (draft: string) => void
  onSelectedCollectionIdsChange: (ids: string[]) => void
  onSelectedDocumentIdChange: (id: string | null) => void
  onSessionsPanelSizeChange: (size: number) => void
  selectedAutonomy: string
  onAutonomyChange: (autonomy: string) => void
  /** User-picked Stufe; null = server default (capabilities). */
  selectedTier?: import('@/features/researchRuns/types').AgentTierId | null
  onTierChange?: (
    tier: import('@/features/researchRuns/types').AgentTierId,
  ) => void
  /** Thoroughness (plan M4), lifted like autonomy — genuinely sticky
   * across view switches. */
  selectedDepth?: 'normal' | 'deep'
  onDepthChange?: (mode: 'normal' | 'deep') => void
  /** Model catalog for the R3 override picker (health-derived, same
   * source as the chat picker; empty catalog -> tier fallback). */
  modelCatalog?: import('@/features/researchRuns/types').ModelCatalogEntry[]
  modelOptions?: import('@/features/researchRuns/types').ChatModelOption[]
  modelOptionsStatus?: 'available' | 'missing' | 'unresolved'
  defaultChatModel?:
    | import('@/features/researchRuns/types').NodeModelResolution
    | null
  selectedCollectionIds: string[]
  selectedDocumentId: string | null
  /** Live server present (false in demo / local-only). */
  serverEnabled: boolean
  /** Demo runtime: replaces the server actions in demo mode. */
  demo?: AgentDemoActions | null
  state: ProjectState
  submitRun: (request: CreateResearchRunRequest) => Promise<unknown> | void
  workspaceId: string
}) {
  const { t } = useLocale()
  const reduceMotion = Boolean(useReducedMotion())
  const isDesktop = useMediaQuery('(min-width: 1024px)')
  const [isMobileSessionsOpen, setIsMobileSessionsOpen] = useState(false)
  // Output-form override (plan M1): workspace-local, defaults to Auto —
  // the agent's intake decides unless the user forces a form.
  const [responseForm, setResponseForm] = useState<AgentResponseForm>('auto')
  // Engine selection (plan M2 FE wiring): user-selectable only when the
  // server registered the kernel; initialized from the published default
  // once capabilities arrive. The demo simulates a current server.
  const [engineMode, setEngineMode] = useState<AgentEngineMode | null>(null)
  // Attached skills and a direct one-message route are workspace-local.
  // The route clears only after the server admits the run.
  const [selectedSkillIds, setSelectedSkillIds] = useState<string[]>([])
  const [executionDirective, setExecutionDirective] =
    useState<AgentExecutionDirective | null>(null)
  const [draftSourcePolicy, setDraftSourcePolicy] =
    useState<AgentSourcePolicy>({ ...DEFAULT_AGENT_SOURCE_POLICY })
  const pendingSaveRef = useRef<(() => Promise<void>) | null>(null)
  const canvas = state.agentCanvas
  const agentBlock = capabilities?.agent ?? null
  const allAutonomyModes = agentBlock?.autonomy_modes ?? [
    'strict',
    'balanced',
    'autonomous',
  ]
  // Two-mode UI (plan M1 S7, Cowork pattern): servers publishing
  // mode_presets narrow the composer to Standard/Auto; the legacy
  // three-way control stays for older servers and when the operator
  // republishes it (advanced_autonomy). Wire vocabulary is unchanged.
  // The demo simulates a CURRENT server, so it presets too (the new UI
  // must be demo-visible).
  const modePresets =
    agentBlock?.mode_presets
    ?? (demo
      ? [
        { autonomy: 'balanced', id: 'standard' },
        { autonomy: 'autonomous', id: 'auto' },
      ]
      : undefined)
  const autonomyModes =
    modePresets && modePresets.length > 0 && !agentBlock?.advanced_autonomy
      ? modePresets
        .map((preset) => preset.autonomy)
        .filter((mode) => allAutonomyModes.includes(mode))
      : allAutonomyModes
  const agentAvailable =
    !serverEnabled || capabilities?.features.workspace_agent === true
  // Run overview (composer status menu): server facts, or the demo's
  // current-server mirror — new UI must be demo-visible. Recomputed on
  // mode switch so the approvals group always describes the SELECTED mode.
  const slashSkills = useMemo(
    () =>
      (skillsApi?.skills ?? [])
        .filter((skill) => skill.include_in_autocomplete)
        .map((skill) => ({
          id: skill.id,
          label: skill.label,
          description: skill.description,
          argument_hint: skill.argument_hint,
        })),
    [skillsApi],
  )
  const overviewSource = agentBlock ?? (demo ? DEMO_AGENT_OVERVIEW_SOURCE : null)
  const kernelSelectable =
    capabilities?.features.agent_kernel === true || Boolean(demo)
  // Deep toggle only when the server publishes it (feature detection);
  // the demo simulates a current server.
  const depthSelectable =
    Boolean(
      agentBlock?.depth_modes?.some((mode) => mode.id === 'deep'),
    ) || Boolean(demo)
  // Stufen (feature detection): published tiers replace the depth
  // toggle; the demo simulates a current server.
  const tiers = agentBlock?.tiers?.length
    ? agentBlock.tiers
    : demo
      ? DEMO_AGENT_TIERS
      : null
  const effectiveTier = tiers
    ? selectedTier ?? agentBlock?.default_tier ?? 'gruendlich'
    : null
  const sourceAvailability = useMemo(() => {
    const published = overviewSource?.source_controls
    if (published?.length) {
      return {
        web: published.find((entry) => entry.id === 'web')?.available === true,
        knowledge:
          published.find((entry) => entry.id === 'knowledge')?.available === true,
      }
    }
    const manifest = new Set(overviewSource?.tools.map((tool) => tool.id) ?? [])
    return {
      web: manifest.has('web.search.instant'),
      knowledge: manifest.has('knowledge.search'),
    }
  }, [overviewSource])
  const executionDirectiveAvailability = useMemo(() => {
    const published = overviewSource?.execution_directives
    if (published?.length) {
      return {
        quick_web:
          published.find((entry) => entry.id === 'quick_web')?.available === true,
        knowledge_only:
          published.find((entry) => entry.id === 'knowledge_only')?.available === true,
      }
    }
    return {
      quick_web: sourceAvailability.web,
      knowledge_only: sourceAvailability.knowledge,
    }
  }, [overviewSource, sourceAvailability])
  // R3 model override picker: agent-OWN selection (never the chat
  // fields — different cost profile), catalog from the same health
  // source the chat picker reads. Hidden without any catalog/options.
  const agentModelSelection = {
    tier: state.ui.selectedAgentModelTier,
    model: state.ui.selectedAgentModel,
    effort: state.ui.selectedAgentEffort,
  }
  const modelPicker =
    modelCatalog.length > 0 || modelOptions.length > 0
      ? {
        catalog: modelCatalog,
        options: modelOptions,
        optionsStatus: modelOptionsStatus,
        defaultModel: defaultChatModel,
        selectedTier: agentModelSelection.tier,
        selectedModel: agentModelSelection.model,
        selectedEffort: agentModelSelection.effort,
        onTierChange: (tier: import('@/features/researchRuns/types').ChatModelTier | null) =>
          dispatch({ tier, type: 'setSelectedAgentModelTier' }),
        onModelChange: (model: string | null) =>
          dispatch({ model, type: 'setSelectedAgentModel' }),
        onEffortChange: (effort: string | null) =>
          dispatch({ effort, type: 'setSelectedAgentEffort' }),
      }
      : null
  const effectiveEngineMode: AgentEngineMode =
    (kernelSelectable ? engineMode : null)
    ?? (demo
      ? 'agent_kernel'
      : defaultEngineMode(overviewSource, capabilities?.features ?? null))
  const overview = useMemo(
    () => buildAgentOverview({
      agent: overviewSource,
      autonomy: selectedAutonomy,
      kernelSelectable,
      mode: effectiveEngineMode,
    }),
    [effectiveEngineMode, kernelSelectable, overviewSource, selectedAutonomy],
  )

  const clientOptions = useMemo<ClientOptions | null>(
    () => (serverEnabled ? { apiKey, workspaceId } : null),
    [apiKey, serverEnabled, workspaceId],
  )

  const control = useAgentControlApi({
    apiKey,
    dispatch,
    enabled: serverEnabled,
    runs: state.agentRuns,
    workspaceId,
  })

  const { error: sessionSyncError, settled: sessionsSettled } =
    useAgentSessionsApi({
      apiKey,
      dispatch,
      projectEpoch: state.projectEpoch,
      sessionGroups: state.agentSessionGroups,
      sessionOrder: state.agentSessionOrder,
      selectedSessionId: state.selectedAgentSessionId,
      sessions: state.agentSessions,
      syncActive: serverEnabled,
      workspaceId,
    })

  // Reload restore: once the session list settles, re-open the persisted
  // selection (or the most recent session) — a reload must land back in
  // the conversation, not the empty state. Once per HYDRATION IDENTITY
  // (workspace + project epoch), so loading a project file or switching
  // workspaces restores again; layout effect, so the selection commits
  // before paint (no one-frame welcome flash).
  const restoredForRef = useRef<string | null>(null)
  const hydrationIdentity = `${workspaceId}:${state.projectEpoch}`
  useLayoutEffect(() => {
    if (!sessionsSettled) return
    if (restoredForRef.current === hydrationIdentity) return
    restoredForRef.current = hydrationIdentity
    if (state.selectedAgentSessionId) return
    const sessionId = restoredAgentSessionId(
      state.ui.selectedAgentSessionId,
      state.agentSessionOrder,
      state.agentSessions,
      state.agentRuns,
    )
    if (sessionId) dispatch({ sessionId, type: 'selectAgentSession' })
  }, [
    dispatch,
    hydrationIdentity,
    sessionsSettled,
    state.agentRuns,
    state.agentSessionOrder,
    state.agentSessions,
    state.selectedAgentSessionId,
    state.ui.selectedAgentSessionId,
  ])

  const selectedSession = state.selectedAgentSessionId
    ? state.agentSessions[state.selectedAgentSessionId]
    : undefined
  const sourcePolicy = selectedSession?.sourcePolicy ?? draftSourcePolicy
  const handleSourcePolicyChange = useCallback(
    (next: AgentSourcePolicy) => {
      setDraftSourcePolicy(next)
      if (!selectedSession) return
      dispatch({
        sessionId: selectedSession.id,
        sourcePolicy: next,
        type: 'setAgentSessionSourcePolicy',
      })
    },
    [dispatch, selectedSession],
  )
  const sessionRuns = useMemo(() => {
    if (!selectedSession) return []
    return selectedSession.runIds
      .map((runId) => state.agentRuns[runId])
      .filter((run): run is AgentRunRecord => Boolean(run))
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt))
  }, [selectedSession, state.agentRuns])
  // Warm the shiki token cache for every markdown body already in memory
  // (answers auto-fetch, memo/report bodies load on tab open) — parity
  // with Chat/Knowledge, so opening a task/answer never pays the
  // highlighter cold start. SETTLED runs only: a live run streams events
  // every second, and re-extracting the whole session's code fences per
  // event would be pure churn (the streaming surface renders its code
  // through the same token cache anyway).
  const agentHighlightFingerprint = useMemo(() => {
    const markdowns: string[] = []
    for (const run of sessionRuns) {
      if (isActiveAgentRun(run.status)) continue
      for (const artifactId of run.artifactOrder) {
        const body = run.artifacts[artifactId]?.contentMarkdown
        if (body && body.trim().length > 0) markdowns.push(body)
      }
    }
    return markdowns.join('\u0000')
  }, [sessionRuns])
  // Keyed on the joined CONTENT, not on sessionRuns identity — a live
  // run re-derives sessionRuns on every SSE event, and the preload
  // effect must only re-arm when a settled body actually changed.
  const agentMarkdownsForHighlight = useMemo(
    () =>
      agentHighlightFingerprint
        ? agentHighlightFingerprint.split('\u0000')
        : [],
    [agentHighlightFingerprint],
  )
  useMarkdownCodePreload(agentMarkdownsForHighlight)
  const latestRun = sessionRuns.at(-1)
  const runningRun = sessionRuns.find((run) => isActiveAgentRun(run.status))
  const statusExecution = useMemo(
    () => normalizeAgentExecutionSnapshot(latestRun?.snapshot),
    [latestRun?.snapshot],
  )
  const toolUseCounts = useMemo(
    () => statusExecution?.toolUseCounts
      ?? countAgentToolUse({
        tasks: latestRun?.plan?.tasks ?? [],
        taskStates: latestRun?.taskStates ?? {},
      }),
    [latestRun?.plan?.tasks, latestRun?.taskStates, statusExecution?.toolUseCounts],
  )

  const openCanvasView = useCallback(
    (descriptor: CanvasViewDescriptor) => {
      dispatch({ descriptor, source: 'user', type: 'openAgentCanvasView' })
    },
    [dispatch],
  )

  const setPlanDraft = useCallback(
    (runId: string, draft: Parameters<AgentCanvasContextValue['setPlanDraft']>[1]) =>
      dispatch({ draft, runId, type: 'setAgentPlanDraft' }),
    [dispatch],
  )

  const requestPlanRefresh = useCallback(
    (runId: string) => {
      dispatch({ runId, type: 'markAgentRunPlanStale' })
    },
    [dispatch],
  )

  // Display context for plan tasks (id -> title + backend label): the
  // "wo" of the approval transparency, shared by the timeline card and
  // the canvas plan view.
  const planSource = useMemo<PlanSourceInfo>(
    () => ({
      collections,
      vectorBackendLabel: vectorBackendDisplay(
        capabilities?.knowledge?.vector_backend,
      ),
    }),
    [capabilities, collections],
  )

  const canvasContext = useMemo<AgentCanvasContextValue>(
    () => ({
      applyPatch: demo ? demo.applyPatch : control.applyPatch,
      cancelTask: control.cancelTask,
      clientOptions,
      decideApproval: demo ? demo.decideApproval : control.decideApproval,
      rejectPatch: demo ? demo.rejectPatch : control.rejectPatch,
      exportArtifact: control.exportArtifact,
      fileAssets: state.fileAssets,
      loadArtifact: control.loadArtifact,
      loadTaskResult: control.loadTaskResult,
      openCanvasView,
      pendingSaveRef,
      planDrafts: state.agentPlanDrafts,
      planSource,
      pollingRunIds: pollingRunIds ?? [],
      prefetchTaskResult: control.prefetchTaskResult,
      requestPlanRefresh,
      runs: state.agentRuns,
      saveArtifact: control.saveArtifact,
      setPlanDraft,
      workspaceId,
    }),
    [
      clientOptions,
      control,
      demo,
      openCanvasView,
      planSource,
      pollingRunIds,
      requestPlanRefresh,
      setPlanDraft,
      state.agentPlanDrafts,
      state.agentRuns,
      state.fileAssets,
      workspaceId,
    ],
  )

  // Follow-the-agent: the LATEST run of the selected session steers.
  const followTarget = useMemo(
    () => (latestRun ? routeAgentRunToView(latestRun) : null),
    [latestRun],
  )
  useCanvasFollow({
    canvas,
    // Follow default on; mobile default off (plan §5.3) — the canvas is a
    // full occluding overlay there and must never take over on its own.
    enabled: isDesktop,
    onAutoPin: () => dispatch({ pinned: true, type: 'setAgentCanvasPinned' }),
    openView: (descriptor) =>
      dispatch({ descriptor, source: 'agent', type: 'openAgentCanvasView' }),
    target: followTarget,
  })

  // Esc leaves focus mode (tabs replaced the overlay stack — closing a
  // tab is an explicit affordance, never a global key).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return
      if (canvas.focus) {
        dispatch({ type: 'toggleAgentCanvasFocus' })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [canvas.focus, dispatch])

  const timelineActions = useMemo<AgentTimelineActions>(
    () => ({
      answerClarification: demo
        ? demo.answerClarification
        : control.answerClarification,
      applyPatch: demo ? demo.applyPatch : control.applyPatch,
      decideApproval: demo ? demo.decideApproval : control.decideApproval,
      rejectPatch: demo ? demo.rejectPatch : control.rejectPatch,
      onCancelRun: demo
        ? (runId) => demo.cancel(runId)
        : (runId) => void cancelRun(runId),
      onOpenCanvas: openCanvasView,
      planDrafts: state.agentPlanDrafts,
      planSource,
      setPlanDraft,
    }),
    [
      cancelRun,
      control,
      demo,
      openCanvasView,
      planSource,
      setPlanDraft,
      state.agentPlanDrafts,
    ],
  )

  const handleSubmit = useCallback(
    async ({
      autonomy,
      collectionIds,
      depth,
      tier,
      documentId,
      engineMode: submitEngineMode,
      executionDirective: submitDirective,
      question,
      responseForm,
      skillIds,
      sourcePolicy: submitSourcePolicy,
    }: AgentComposerSubmit) => {
      // A pending clarification absorbs the send (ONE input locus, plan
      // B3): free text answers the gate instead of racing a new run.
      const gate = runningRun ? pendingGate(runningRun) : null
      if (gate?.kind === 'clarification') {
        await timelineActions.answerClarification(
          runningRun!.runId,
          gate.clarification.clarificationId,
          { answer: question },
        )
        return true
      }
      // A follow-up turn must not race a pending canvas edit: the agent
      // reads the LATEST revision, so unsaved text would be lost (§5.4).
      if (pendingSaveRef.current) {
        await pendingSaveRef.current()
      }
      let sessionId = state.selectedAgentSessionId
      if (!sessionId || !state.agentSessions[sessionId]) {
        sessionId = `agent-session-${Date.now().toString(36)}`
        const now = new Date().toISOString()
        dispatch({
          session: {
            id: sessionId,
            title: question.trim().slice(0, 80),
            groupId: null,
            createdAt: now,
            updatedAt: now,
            runIds: [],
            sourcePolicy: submitSourcePolicy,
          },
          type: 'createAgentSession',
        })
      }
      if (demo) {
        demo.submit({
          agentTier: tier ?? '',
          autonomy,
          collectionIds,
          depth,
          documentId,
          engineMode: submitEngineMode,
          question,
          responseForm,
          sessionId,
          skillLabels: skillIds.map(
            (id) => slashSkills.find((skill) => skill.id === id)?.label ?? id,
          ),
          executionDirective: submitDirective,
          sourcePolicy: submitSourcePolicy,
          modelSelection: {
            model: state.ui.selectedAgentModel,
            effort: state.ui.selectedAgentEffort,
            tier: state.ui.selectedAgentModelTier,
          },
        })
        setSelectedSkillIds([])
        setExecutionDirective(null)
        return true
      }
      const overrides = agentOverridesFromSelection(
        depth,
        state.ui.selectedAgentModelTier,
        state.ui.selectedAgentModel,
        state.ui.selectedAgentEffort,
        tier,
      )
      const summary = await submitRun({
        agentOverrides: overrides,
        autonomy,
        documentId,
        knowledgeFilters:
          collectionIds.length > 0 ? { collectionIds } : undefined,
        mode: submitEngineMode,
        question,
        responseForm,
        sessionId,
        skillIds: skillIds.length > 0 ? skillIds : undefined,
        sourcePolicy: submitSourcePolicy,
        executionDirective: submitDirective,
      })
      if (!summary) return false
      setSelectedSkillIds([])
      setExecutionDirective(null)
      return true
    },
    [
      demo,
      dispatch,
      runningRun,
      slashSkills,
      state.agentSessions,
      state.selectedAgentSessionId,
      state.ui.selectedAgentEffort,
      state.ui.selectedAgentModel,
      state.ui.selectedAgentModelTier,
      submitRun,
      timelineActions,
    ],
  )

  const createSession = useCallback(() => {
    const now = new Date().toISOString()
    dispatch({
      session: {
        id: `agent-session-${Date.now().toString(36)}`,
        title: t.agent.sessions.create,
        groupId: null,
        createdAt: now,
        updatedAt: now,
        runIds: [],
        sourcePolicy: { ...DEFAULT_AGENT_SOURCE_POLICY },
      },
      type: 'createAgentSession',
    })
  }, [dispatch, t])

  const railVisible = state.ui.isAgentSessionsVisible
  const railMotion = useAnimatedResizablePanelCollapse({
    expanded: railVisible,
    expandedSize: state.ui.panelLayout.agentSessions,
    reduceMotion,
  })
  const canvasMotion = useAnimatedResizablePanelCollapse({
    expanded: canvas.open,
    expandedSize: canvasPanelSize,
    reduceMotion,
  })

  const canvasLabelFor = useCallback(
    (descriptor: CanvasViewDescriptor): string => {
      if (descriptor.view === 'document') {
        const run = state.agentRuns[descriptor.runId]
        return (
          run?.artifacts[descriptor.artifactId]?.title
          || t.agent.canvas.views.document
        )
      }
      if (descriptor.view === 'evidence') {
        return `${t.agent.canvas.views.evidence} ${descriptor.label}`
      }
      return t.agent.canvas.views[descriptor.view]
    },
    [state.agentRuns, t],
  )

  const rail = (
    <AgentSessionRail
      onCreateSession={createSession}
      onCreateSessionGroup={() =>
        dispatch({ title: t.agent.sessions.createGroup, type: 'createAgentSessionGroup' })}
      onDeleteSession={(sessionId) => {
        const session = state.agentSessions[sessionId]
        if (session && !demo) {
          for (const runId of session.runIds) void deleteRun(runId)
        }
        dispatch({ sessionId, type: 'deleteAgentSession' })
      }}
      onRenameSession={(sessionId, title) =>
        dispatch({ sessionId, title, type: 'renameAgentSession' })}
      onSelectSession={(sessionId) => {
        dispatch({ sessionId, type: 'selectAgentSession' })
        if (!isDesktop) setIsMobileSessionsOpen(false)
      }}
      onTogglePinnedSession={(sessionId) =>
        dispatch({ sessionId, type: 'togglePinnedAgentSession' })}
      pinnedSessionIds={state.ui.pinnedExplorer.agentSessionIds}
      syncError={sessionSyncError}
      runs={state.agentRuns}
      selectedSessionId={state.selectedAgentSessionId}
      sessionGroupOrder={state.agentSessionGroupOrder}
      sessionGroups={state.agentSessionGroups}
      sessionOrder={state.agentSessionOrder}
      sessions={state.agentSessions}
    />
  )

  const activeGate = runningRun ? pendingGate(runningRun) : null
  const memoWritingOrDone = Boolean(
    latestRun?.artifactOrder.some(
      (artifactId) => latestRun.artifacts[artifactId]?.kind === 'memo',
    ),
  )
  const earlyPhase =
    runningRun
    && ['intake', 'discovery', 'planning'].includes(runningRun.phase)
  // Context pills (plan B4): one tap from the composer to the live
  // canvas surface; the pulsing dot marks the live one.
  const pills = (runningRun || memoWritingOrDone) && (
    <div className="mx-auto mb-1.5 flex max-w-5xl flex-wrap items-center gap-1.5">
      {runningRun && (
        <button
          className="inline-flex h-6 items-center gap-1.5 rounded-full border border-brand/20 bg-brand-subtle px-2.5 t-hint font-semibold text-brand transition-colors hover:bg-brand-subtle/80"
          onClick={() =>
            openCanvasView({ runId: runningRun.runId, view: 'run' })}
          type="button"
        >
          <span
            aria-hidden="true"
            className={cn(
              'size-1.5 rounded-full bg-brand',
              !reduceMotion && 'inqtrix-running-dot',
            )}
          />
          {earlyPhase
            ? t.agent.pills.followPlanning
            : t.agent.pills.followExecution}
        </button>
      )}
      {memoWritingOrDone && latestRun && (
        <button
          className="inline-flex h-6 items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 t-hint font-semibold text-muted-foreground transition-colors hover:text-foreground"
          onClick={() => {
            const memoId = latestRun.artifactOrder.find(
              (artifactId) =>
                latestRun.artifacts[artifactId]?.kind === 'memo',
            )
            if (memoId) {
              openCanvasView({
                artifactId: memoId,
                runId: latestRun.runId,
                view: 'document',
              })
            }
          }}
          type="button"
        >
          <FileText aria-hidden="true" className="size-3" />
          {t.agent.pills.viewReport}
        </button>
      )}
    </div>
  )

  const composer = (
    <>
      {/* Pills ABOVE the gate tray: a pending gate docks directly onto
          the composer (one decision surface), the follow/report pills
          float above it. */}
      {pills}
      <ComposerGateTray actions={timelineActions} run={runningRun} />
      <AgentComposer
        answerMode={activeGate?.kind === 'clarification'}
        autonomy={selectedAutonomy}
        autonomyModes={autonomyModes}
        collections={collections}
        disabled={!agentAvailable}
        documents={documents}
        draftQuestion={draftQuestion}
        depthMode={selectedDepth}
        depthSelectable={depthSelectable}
        tierMode={effectiveTier}
        tiers={tiers}
        onTierModeChange={onTierChange}
        engineMode={effectiveEngineMode}
        kernelSelectable={kernelSelectable}
        memoryEnabled={memoryEnabled}
        modelPicker={modelPicker}
        notice={agentAvailable ? null : t.agent.composer.notAvailable}
        onAutonomyChange={onAutonomyChange}
        onDepthModeChange={onDepthChange}
        onDraftQuestionChange={onDraftQuestionChange}
        onEngineModeChange={setEngineMode}
        onSelectedCollectionIdsChange={onSelectedCollectionIdsChange}
        onSelectedDocumentIdChange={onSelectedDocumentIdChange}
        onStop={() => {
          if (!runningRun) return
          if (demo) demo.cancel(runningRun.runId)
          else void cancelRun(runningRun.runId)
        }}
        onResponseFormChange={setResponseForm}
        onSelectedSkillIdsChange={setSelectedSkillIds}
        onExecutionDirectiveChange={setExecutionDirective}
        onSourcePolicyChange={handleSourcePolicyChange}
        onSubmit={handleSubmit}
        overview={overview}
        responseForm={responseForm}
        running={Boolean(runningRun)}
        selectedCollectionIds={selectedCollectionIds}
        selectedDocumentId={selectedDocumentId}
        selectedSkillIds={selectedSkillIds}
        executionDirective={executionDirective}
        statusExecution={statusExecution}
        sourcePolicy={sourcePolicy}
        sourceAvailability={sourceAvailability}
        executionDirectiveAvailability={executionDirectiveAvailability}
        toolUseCounts={toolUseCounts}
        maxAttachedSkills={agentBlock?.skills?.max_attached ?? 3}
        slashSkills={slashSkills}
      />
    </>
  )

  const timeline = (
    <div className="inqtrix-contained-panel flex h-full min-h-0 min-w-0 flex-1 flex-col">
      <header className="z-10 flex inqtrix-panel-header shrink-0 items-center justify-between gap-2 border-b border-border bg-background px-3">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <PanelToggle
            collapseLabel={t.agent.sessions.title}
            controlsId={SESSIONS_PANEL_ID}
            expandLabel={t.agent.sessions.title}
            expanded={isDesktop ? railVisible : isMobileSessionsOpen}
            onToggle={(next) => {
              if (isDesktop) {
                if (next !== railVisible) dispatch({ type: 'toggleAgentSessionsVisible' })
                return
              }
              setIsMobileSessionsOpen(next)
            }}
            side="left"
          />
          <h1 className="truncate t-section text-foreground">
            {selectedSession?.title || t.navigation.agent}
          </h1>
        </div>
        <PanelToggle
          collapseLabel={t.agent.canvas.title}
          controlsId={CANVAS_PANEL_ID}
          expandLabel={t.agent.canvas.title}
          expanded={canvas.open}
          onToggle={(next) => {
            if (next) {
              dispatch({
                descriptor:
                  activeCanvasView(canvas)
                  ?? fallbackCanvasDescriptor(latestRun),
                source: 'user',
                type: 'openAgentCanvasView',
              })
            } else {
              dispatch({ type: 'closeAgentCanvas' })
            }
          }}
          side="right"
        />
      </header>
      <ScrollArea className="min-h-0 flex-1">
        {/* Same content width as the chat mode's transcript (max-w-5xl)
            so questions and answers read at the SAME measure across
            desks — and flush with the composer below. */}
        <div className="mx-auto w-full max-w-5xl px-4 py-4 md:px-8">
          {sessionRuns.length === 0
            && serverEnabled
            && (!sessionsSettled || (selectedSession && !runsHydrated)) ? (
            // Hydration window: sessions/runs are still paging in — a
            // skeleton, never a false "empty" welcome (same primitive as
            // Chat/Knowledge).
            <ConversationSkeleton reduceMotion={reduceMotion} />
          ) : sessionRuns.length === 0 ? (
            <div className="flex min-h-[40vh] items-center justify-center">
              <WelcomeState
                actions={
                  <div className="flex flex-wrap items-center justify-center gap-2">
                    {[
                      t.agent.empty.exampleWeb,
                      t.agent.empty.exampleKnowledge,
                      t.agent.empty.exampleMission,
                    ].map((example) => (
                      <button
                        className="inline-flex h-8 max-w-56 items-center rounded-md border border-border bg-background px-3 text-xs font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground"
                        key={example}
                        onClick={() => onDraftQuestionChange(example)}
                        type="button"
                      >
                        <span className="truncate">{example}</span>
                      </button>
                    ))}
                  </div>
                }
                subtitle={t.agent.empty.subtitle}
                title={t.agent.empty.title}
              />
            </div>
          ) : (
            <div className="space-y-6">
              {sessionRuns.map((run) => (
                <AgentRunTurn
                  actions={timelineActions}
                  key={run.runId}
                  run={run}
                  transportDegraded={pollingRunIds?.includes(run.runId)}
                />
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
      <div className="z-10 shrink-0 px-3 pb-4 pt-2 md:px-6">{composer}</div>
    </div>
  )

  const memoArtifactId = latestRun?.artifactOrder.find(
    (artifactId) => latestRun.artifacts[artifactId]?.kind === 'memo',
  )
  const addMenu = latestRun ? (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={t.agent.canvas.addTab}
          className="size-6 shrink-0 text-muted-foreground hover:text-foreground"
          size="icon"
          type="button"
          variant="ghost"
        >
          <Plus className="size-3.5" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className={optionMenuContentClassName}
        side="bottom"
        sideOffset={6}
      >
        <div className="py-1">
          <OptionMenuItem
            active={false}
            icon={ListChecks}
            label={t.agent.canvas.views.plan}
            onSelect={() =>
              openCanvasView({ runId: latestRun.runId, view: 'plan' })}
          />
          <OptionMenuItem
            active={false}
            icon={Waypoints}
            label={t.agent.canvas.views.run}
            onSelect={() =>
              openCanvasView({ runId: latestRun.runId, view: 'run' })}
          />
          {memoArtifactId && (
            <OptionMenuItem
              active={false}
              icon={FileText}
              label={t.agent.canvas.views.document}
              onSelect={() =>
                openCanvasView({
                  artifactId: memoArtifactId,
                  runId: latestRun.runId,
                  view: 'document',
                })}
            />
          )}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  ) : undefined

  const canvasHost = (
    <AgentCanvasReactContext.Provider value={canvasContext}>
      <CanvasHost
        addMenu={addMenu}
        canvas={canvas}
      emptyState={
        <div className="flex h-full items-center justify-center px-6">
          <div className="text-center">
            <p className="t-card text-foreground">{t.agent.canvas.empty}</p>
            <p className="mt-1 t-meta text-muted-foreground">
              {t.agent.canvas.emptyHint}
            </p>
          </div>
        </div>
      }
      iconFor={canvasIconFor}
      labelFor={canvasLabelFor}
      labels={{
        close: t.agent.canvas.close,
        closeTab: t.agent.canvas.closeTab,
        follow: t.agent.canvas.follow,
        pinTab: t.agent.canvas.pinTab,
        pinned: t.agent.canvas.pinned,
        previewTab: t.agent.canvas.previewTab,
        unpin: t.agent.canvas.unpin,
      }}
        onActivateTab={(key) =>
          dispatch({ key, type: 'activateAgentCanvasTab' })}
        onClose={() => dispatch({ type: 'closeAgentCanvas' })}
        onCloseTab={(key) => dispatch({ key, type: 'closeAgentCanvasTab' })}
        onPinTab={(key) => dispatch({ key, type: 'pinAgentCanvasTab' })}
        onSetPinned={(pinned) =>
          dispatch({ pinned, type: 'setAgentCanvasPinned' })}
        registry={AGENT_CANVAS_REGISTRY}
        working={Boolean(runningRun)}
      />
    </AgentCanvasReactContext.Provider>
  )

  const timelineAndCanvas = (
    <ResizablePanelGroup
      className="min-h-0 min-w-0 flex-1 overflow-hidden"
      elementRef={canvasMotion.groupRef}
      onLayoutChanged={(layout: Record<string, number>) => {
        const size = layout[CANVAS_PANEL_ID]
        if (
          canvas.open
          && !canvasMotion.isProgrammaticLayoutChange()
          && typeof size === 'number'
          && size > 0
        ) {
          onCanvasPanelSizeChange(size)
        }
      }}
      orientation="horizontal"
    >
      <ResizablePanel
        id={AGENT_TIMELINE_PANEL_ID}
        // Identity anchor — must stay unconditional so the timeline DOM
        // survives the desktop/mobile flip.
        key={AGENT_TIMELINE_PANEL_ID}
        minSize="32%"
      >
        {timeline}
      </ResizablePanel>
      {isDesktop && (
        <>
          <AnimatedResizableHandle expanded={canvas.open} />
          <ResizablePanel
            collapsible
            collapsedSize="0%"
            defaultSize={`${canvas.open ? canvasPanelSize : 0}%`}
            id={CANVAS_PANEL_ID}
            maxSize="62%"
            minSize={canvas.open ? '30%' : '0%'}
            panelRef={canvasMotion.panelRef}
          >
            <AnimatedPanelBody expanded={canvas.open} side="right">
              {canvasHost}
            </AnimatedPanelBody>
          </ResizablePanel>
        </>
      )}
    </ResizablePanelGroup>
  )

  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background">
      <div className="relative flex min-h-0 w-full flex-1 overflow-hidden bg-background">
        <ResizablePanelGroup
          className="min-h-0 min-w-0 flex-1 overflow-hidden"
          elementRef={railMotion.groupRef}
          onLayoutChanged={(layout: Record<string, number>) => {
            const size = layout[SESSIONS_PANEL_ID]
            if (
              railVisible
              && !railMotion.isProgrammaticLayoutChange()
              && typeof size === 'number'
              && size > 0
            ) {
              onSessionsPanelSizeChange(size)
            }
          }}
          orientation="horizontal"
        >
          {isDesktop && (
            <>
              <ResizablePanel
                collapsible
                collapsedSize="0%"
                defaultSize={`${railVisible ? state.ui.panelLayout.agentSessions : 0}%`}
                id={SESSIONS_PANEL_ID}
                maxSize="42%"
                minSize={railVisible ? '18%' : '0%'}
                panelRef={railMotion.panelRef}
              >
                <AnimatedPanelBody expanded={railVisible} side="left">
                  {rail}
                </AnimatedPanelBody>
              </ResizablePanel>
              <AnimatedResizableHandle expanded={railVisible} />
            </>
          )}
          <ResizablePanel
            id={AGENT_CENTER_PANEL_ID}
            // Identity anchor — must stay unconditional so the center content
            // survives the desktop/mobile flip.
            key={AGENT_CENTER_PANEL_ID}
            minSize="30%"
          >
            {timelineAndCanvas}
          </ResizablePanel>
        </ResizablePanelGroup>
        {!isDesktop && (
          <>
            <ResponsiveSidePanel
              closeLabel={t.agent.sessions.title}
              controlsId={SESSIONS_PANEL_ID}
              onOpenChange={setIsMobileSessionsOpen}
              open={isMobileSessionsOpen}
              showHeader={false}
              side="left"
              title={t.agent.sessions.title}
            >
              {rail}
            </ResponsiveSidePanel>
            <ResponsiveSidePanel
              closeLabel={t.agent.canvas.title}
              controlsId={CANVAS_PANEL_ID}
              onOpenChange={(open) => {
                if (open) {
                  dispatch({
                    descriptor:
                  activeCanvasView(canvas)
                  ?? fallbackCanvasDescriptor(latestRun),
                    source: 'user',
                    type: 'openAgentCanvasView',
                  })
                  return
                }
                dispatch({ type: 'closeAgentCanvas' })
              }}
              open={canvas.open}
              showHeader={false}
              side="right"
              title={t.agent.canvas.title}
            >
              {canvasHost}
            </ResponsiveSidePanel>
          </>
        )}
      </div>
    </section>
  )
}

/** Per-view tab glyphs for the canvas tab row (host stays agnostic). */
function canvasIconFor(descriptor: CanvasViewDescriptor): LucideIcon {
  switch (descriptor.view) {
    case 'plan':
      return ListChecks
    case 'run':
      return Waypoints
    case 'document':
      return FileText
    case 'evidence':
      return descriptor.label.startsWith('W') ? Globe2 : BookOpen
    case 'file':
      return Folder
    case 'diff':
      return Repeat2
    case 'patch':
      return PenLine
  }
}

function fallbackCanvasDescriptor(
  run: AgentRunRecord | undefined,
): CanvasViewDescriptor {
  if (run) {
    const memoId = run.artifactOrder.find(
      (artifactId) => run.artifacts[artifactId]?.kind === 'memo',
    )
    if (memoId) {
      return { view: 'document', runId: run.runId, artifactId: memoId }
    }
    // A live run without a memo: the Verlauf is what the user opens the
    // canvas for; settled runs land on the plan.
    if (run.status === 'running' || run.status === 'queued') {
      return { view: 'run', runId: run.runId }
    }
    return { view: 'plan', runId: run.runId }
  }
  return { view: 'plan', runId: '' }
}
