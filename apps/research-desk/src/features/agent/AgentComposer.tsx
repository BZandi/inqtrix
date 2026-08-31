import { useMemo, useRef, useState, useEffect, useLayoutEffect } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import type {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  SyntheticEvent,
} from 'react'

import {
  BookOpen,
  BookSearch,
  BrainCircuit,
  Database,
  FileText,
  Globe2,
  Layers,
  MessageSquareText,
  PenLine,
  Trash2,
  Plus,
  Gauge,
  Search,
  SendHorizontal,
  Shield,
  ShieldCheck,
  Sparkles,
  WandSparkles,
  Waypoints,
  Workflow,
  X,
  Zap,
} from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Chip } from '@/components/ui/chip'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { MentionMenu, type MentionMenuOption } from '@/components/ui/mention-menu'
import {
  OptionMenuHeader,
  OptionMenuItem,
  optionMenuContentClassName,
} from '@/components/ui/option-menu'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { composerIconButtonClassName } from '@/features/composer/ComposerIconButton'
import { ComposerStopButton } from '@/features/composer/ComposerStopButton'
import { resizeTextareaToRows } from '@/features/composer/textareaAutosize'
import {
  detectCollectionMention,
  type CollectionMentionState,
} from '@/features/composer/collectionMention'
import { QuotaMeter } from '@/features/quota/QuotaMeter'
import { ModelTierPicker } from '@/features/researchRuns/ModelTierPicker'
import type {
  AgentTierCapability,
  AgentTierId,
  ChatModelOption,
  ChatModelTier,
  ModelCatalogEntry,
  NodeModelResolution,
} from '@/features/researchRuns/types'
import { effortLevelLabel } from '@/lib/modelCard'
import {
  REPORT_GUIDANCE_MAX_CHARS_FALLBACK,
  REPORT_RULE_IDS_MAX_FALLBACK,
  hasReportRequirement,
  toggleReportRule,
} from '@/features/agent/reportRequirement'
import type { ReportRuleOption } from '@/features/agent/plan/PlanReviewBody'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { AgentStatusMenu } from './AgentStatusMenu'
import type {
  AgentEngineMode,
  AgentOverview,
  AgentToolUseCounts,
} from './agentStatusOverview'
import {
  EXECUTION_DIRECTIVE_OPTIONS,
  detectSkillSlash,
  type SkillSlashState,
} from './skillSlash'
import {
  extractTrailingExecutionDirective,
  type AgentExecutionDirective,
  type AgentExecutionSnapshot,
  type AgentSourcePolicy,
} from './executionPolicy'
import { ComposerDisclosureHint } from '@/features/composer/ComposerDisclosureHint'

export type AgentCollectionOption = { id: string; title: string }

export type AgentDocumentOption = { id: string; title: string }

/** A session canvas document offered for @-mention (P9, K5): pinned as
 * `canvas_context` (no comments) on the next submission. */
export type AgentCanvasDocumentOption = {
  artifactId: string
  revision: number
  /** Derived file name (`marktbericht.md`) — the display identity. */
  name: string
  title: string
}

export type AgentModelPickerProps = {
  /** Concrete-model catalog (empty -> tier fallback inside the picker). */
  catalog: ModelCatalogEntry[]
  options: ChatModelOption[]
  optionsStatus: 'available' | 'missing' | 'unresolved'
  defaultModel: NodeModelResolution | null
  selectedTier: ChatModelTier | null
  selectedModel: string | null
  selectedEffort: string | null
  onTierChange: (tier: ChatModelTier | null) => void
  onModelChange: (model: string | null) => void
  onEffortChange: (effort: string | null) => void
}

export type AgentComposerSubmit = {
  question: string
  autonomy: string
  collectionIds: string[]
  /** Target editor document for a patch assignment (M7); at most one. */
  documentId?: string
  /** The selected engine: the deterministic mission machine or
   * the conversational kernel. Callers gate availability server-side. */
  engineMode: AgentEngineMode
  /** Explicitly attached skill chips; server-admitted. */
  skillIds: string[]
  /** Source availability chosen for this Agent Desk session. */
  sourcePolicy: AgentSourcePolicy
  /** Optional direct route that applies to this message only. */
  executionDirective?: AgentExecutionDirective
  /** Output form: 'auto' lets the agent decide, 'chat' forces
   * the inline answer, 'canvas' the memo document. */
  responseForm: 'auto' | 'chat' | 'canvas'
  /** Thoroughness: 'deep' = budgets + verification pass. */
  depth: 'normal' | 'deep'
  /** Selected Stufe; null on servers without the tiers capability (the
   * legacy depth toggle applies then). */
  tier: AgentTierId | null
}

export type AgentResponseForm = AgentComposerSubmit['responseForm']

export const AGENT_RESPONSE_FORMS: readonly AgentResponseForm[] = [
  'auto',
  'chat',
  'canvas',
]

/**
 * The Agent Desk composer: context attachments above the textarea, a source
 * dock for persistent availability, one flat execution capsule, and the
 * independent run overview beside quota/send. The draft is caller-owned so it
 * survives view switches; `running` gates only the send, never drafting.
 */
export function AgentComposer({
  answerMode = false,
  autonomy,
  autonomyModes,
  canvasComments = [],
  canvasDocuments = [],
  pinnedCanvasDocumentId = null,
  onPinnedCanvasDocumentChange,
  collections,
  disabled = false,
  documents = [],
  draftQuestion,
  depthMode = 'normal',
  depthSelectable = false,
  tierMode = null,
  tiers = null,
  onTierModeChange,
  engineMode = 'workspace_agent',
  kernelSelectable = false,
  memoryEnabled = false,
  modelPicker = null,
  notice,
  onAutonomyChange,
  onDepthModeChange,
  onEngineModeChange,
  onDraftQuestionChange,
  onEditCanvasComment,
  onRemoveCanvasComment,
  onSelectedCollectionIdsChange,
  onSelectedDocumentIdChange,
  onResponseFormChange,
  onStop,
  onSubmit,
  onSelectedSkillIdsChange,
  onExecutionDirectiveChange,
  onSourcePolicyChange,
  overview = null,
  reportOptions = [],
  reportIds = [],
  reportIdsMax: reportLimit = 3,
  onReportIdsChange,
  reportGuidance = '',
  reportRuleIds = [],
  reportRuleOptions = [],
  reportGuidanceMaxChars: reportGuidanceLimit = REPORT_GUIDANCE_MAX_CHARS_FALLBACK,
  reportRuleIdsMax: reportRuleLimit = REPORT_RULE_IDS_MAX_FALLBACK,
  onReportGuidanceChange,
  onReportRuleIdsChange,
  responseForm = 'auto',
  running = false,
  selectedCollectionIds,
  selectedDocumentId = null,
  selectedSkillIds = [],
  executionDirective = null,
  statusExecution = null,
  sourcePolicy,
  sourceAvailability = { web: true, knowledge: true },
  executionDirectiveAvailability = {
    quick_web: true,
    knowledge_only: true,
  },
  toolUseCounts = { web: 0, knowledge: 0 },
  maxAttachedSkills = 3,
  slashSkills = [],
}: {
  /** A clarification is pending: the send answers it (ONE input locus)
   * instead of starting a new run, and the placeholder says so. */
  answerMode?: boolean
  autonomy: string
  /** From `capabilities.agent.autonomy_modes` — never hardcoded. */
  autonomyModes: string[]
  /** Queued canvas selection comments (P4/P9c) — stacked rows above
   * the input; they ride the next submission's canvas_context. */
  canvasComments?: import('./canvas/commentQueue').AgentCanvasCommentDraft[]
  /** Pencil on a stacked row (P9c): jump to the anchor + edit. */
  onEditCanvasComment?: (id: string) => void
  /** Session canvas documents for the @-mention group (P9, K5); empty
   * hides the group. */
  canvasDocuments?: AgentCanvasDocumentOption[]
  /** The mention-pinned canvas document (caller-owned, like the comment
   * queue); rides the next submission when no comments are queued. */
  pinnedCanvasDocumentId?: string | null
  onPinnedCanvasDocumentChange?: (artifactId: string | null) => void
  collections: AgentCollectionOption[]
  /** Finished Research-Desk reports that can be attached as INPUT
   * (P14). Empty hides the group entirely. */
  reportOptions?: import('@/features/project/selectors').CompletedReportOption[]
  /** Attached report ids for the next submission. Ids only — the server
   * resolves the name and the kernel fetches the body on demand. */
  reportIds?: string[]
  /** Server cap (published == enforced). */
  reportIdsMax?: number
  onReportIdsChange?: (reportIds: string[]) => void
  /** Result requirement for the NEXT submission (S6): how the result has
   * to look. Caller-owned like the comment queue, and cleared by the
   * caller on an accepted submission. Without it, a run that never
   * reaches a plan gate — autonomous, the speed tier, and every kernel
   * run — has no way to state one at all. */
  reportGuidance?: string
  /** Attached prompt-library rules, IDS ONLY; the server resolves their
   * text from the caller's own catalog. */
  reportRuleIds?: string[]
  /** Rules the user opted into for the agent surface (`visibility.agent`);
   * empty hides the section. */
  reportRuleOptions?: ReportRuleOption[]
  /** Server limits (published == enforced). */
  reportGuidanceMaxChars?: number
  reportRuleIdsMax?: number
  onReportGuidanceChange?: (guidance: string) => void
  onReportRuleIdsChange?: (ruleIds: string[]) => void
  disabled?: boolean
  /** Patchable editor documents (server-synced or demo); empty hides the
   * document scope entirely. */
  documents?: AgentDocumentOption[]
  draftQuestion: string
  /** Thoroughness; caller-owned like `autonomy`. */
  depthMode?: 'normal' | 'deep'
  /** True only when the server publishes `agent.depth_modes` with
   * 'deep' — the toggle hides otherwise (feature detection). */
  depthSelectable?: boolean
  /** Effective Stufe; only meaningful when `tiers` is published. */
  tierMode?: AgentTierId | null
  /** Published Stufen ladder (`capabilities.agent.tiers`); non-empty
   * replaces the legacy depth toggle with the Stufe control. */
  tiers?: AgentTierCapability[] | null
  onTierModeChange?: (tier: AgentTierId | null) => void
  /** The selected engine; caller-owned like `autonomy`. */
  engineMode?: AgentEngineMode
  /** True only when the server registered the kernel
   * (`features.agent_kernel`) — the picker hides otherwise. */
  kernelSelectable?: boolean
  /** Account preference `enable_agent_memory`, shown in the run overview. */
  memoryEnabled?: boolean
  /** Model/effort override picker (R3); null hides it (no catalog/health). */
  modelPicker?: AgentModelPickerProps | null
  /** Loud availability note (agent feature off / server missing). */
  notice?: string | null
  onAutonomyChange: (autonomy: string) => void
  onDepthModeChange?: (mode: 'normal' | 'deep') => void
  onDraftQuestionChange: (draft: string) => void
  onEngineModeChange?: (mode: AgentEngineMode) => void
  onRemoveCanvasComment?: (id: string) => void
  onSelectedCollectionIdsChange: (ids: string[]) => void
  onSelectedDocumentIdChange?: (id: string | null) => void
  onResponseFormChange?: (form: AgentResponseForm) => void
  onSelectedSkillIdsChange?: (ids: string[]) => void
  onExecutionDirectiveChange?: (directive: AgentExecutionDirective | null) => void
  onSourcePolicyChange: (policy: AgentSourcePolicy) => void
  onStop: () => void
  onSubmit: (submit: AgentComposerSubmit) => boolean | Promise<boolean>
  /** Server-fact summary for the run overview; null hides the menu
   * (older server without the agent capabilities block). */
  overview?: AgentOverview | null
  /** Output-form override; caller-owned like `autonomy`. */
  responseForm?: AgentResponseForm
  running?: boolean
  selectedCollectionIds: string[]
  selectedDocumentId?: string | null
  /** Attached skill ids (chips); caller-owned like `autonomy`. */
  selectedSkillIds?: string[]
  /** One-message route selected from the direct-command group. */
  executionDirective?: AgentExecutionDirective | null
  /** Accepted run facts for the read-only overview. The source dock still
   * reflects the next message's composer state. */
  statusExecution?: AgentExecutionSnapshot | null
  sourcePolicy: AgentSourcePolicy
  /** Deployment availability, separate from the session's choice. */
  sourceAvailability?: Record<'web' | 'knowledge', boolean>
  executionDirectiveAvailability?: Record<AgentExecutionDirective, boolean>
  toolUseCounts?: AgentToolUseCounts
  /** `capabilities.agent.skills.max_attached` (soft gate; the server
   * enforces the hard one). */
  maxAttachedSkills?: number
  /** Slash-menu skills (autocomplete-enabled, caller-filtered). */
  slashSkills?: { id: string; label: string; description: string; argument_hint: string }[]
}) {
  const { t } = useLocale()
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const [question, setQuestion] = useState(draftQuestion)
  const [mention, setMention] = useState<CollectionMentionState | null>(null)
  const [mentionIndex, setMentionIndex] = useState(0)
  const [slash, setSlash] = useState<SkillSlashState | null>(null)
  const [submitting, setSubmitting] = useState(false)

  // Echo guard: the parent mirrors every keystroke back through the
  // draftQuestion prop. Only an EXTERNAL change (view-switch restore,
  // programmatic clear) may replace the text and close the menus — the
  // echo of our own keystroke must not, or the @- and /-menus die
  // between events (visible as flicker, fatal for programmatic input).
  const questionRef = useRef(draftQuestion)
  useEffect(() => {
    if (draftQuestion === questionRef.current) return
    questionRef.current = draftQuestion
    setQuestion(draftQuestion)
    setMention(null)
    setSlash(null)
  }, [draftQuestion])
  useEffect(() => {
    questionRef.current = question
    onDraftQuestionChange(question)
  }, [question, onDraftQuestionChange])
  useLayoutEffect(() => {
    resizeTextareaToRows(textareaRef.current, 6)
  }, [question])

  const selectedCollections = collections.filter((collection) =>
    selectedCollectionIds.includes(collection.id))
  const selectedDocument =
    documents.find((document) => document.id === selectedDocumentId) ?? null
  const addableCollections = useMemo(
    () =>
      collections.filter(
        (collection) => !selectedCollectionIds.includes(collection.id),
      ),
    [collections, selectedCollectionIds],
  )
  const addableDocuments = useMemo(
    () => documents.filter((document) => document.id !== selectedDocumentId),
    [documents, selectedDocumentId],
  )
  const pinnedCanvasDocument =
    canvasDocuments.find(
      (document) => document.artifactId === pinnedCanvasDocumentId,
    ) ?? null
  // Single-document channel (P9, K5): queued comments already bind the
  // attachment to THEIR document, so other canvas documents stop being
  // offered while the queue is non-empty — the conflict cannot arise.
  const addableCanvasDocuments = useMemo(
    () =>
      canvasComments.length > 0
        ? []
        : canvasDocuments.filter(
          (document) => document.artifactId !== pinnedCanvasDocumentId,
        ),
    [canvasComments.length, canvasDocuments, pinnedCanvasDocumentId],
  )
  const mentionCandidates = useMemo(() => {
    if (!mention) return []
    const query = mention.query.toLowerCase()
    const matchingCollections = addableCollections
      .filter((collection) => collection.title.toLowerCase().includes(query))
      .map((collection) => ({ kind: 'collection' as const, option: collection }))
    const matchingDocuments = documents
      .filter(
        (document) =>
          document.id !== selectedDocumentId
          && document.title.toLowerCase().includes(query),
      )
      .map((document) => ({ kind: 'document' as const, option: document }))
    const matchingCanvasDocuments = addableCanvasDocuments
      .filter(
        (document) =>
          document.name.toLowerCase().includes(query)
          || document.title.toLowerCase().includes(query),
      )
      .map((document) => ({ kind: 'canvas' as const, option: document }))
    // Finished research reports as INPUT (P14). At the cap the group
    // disappears from the menu rather than offering a click that would
    // quietly do nothing.
    const matchingReports = reportIds.length >= reportLimit
      ? []
      : reportOptions
        .filter(
          (report) =>
            !reportIds.includes(report.runId)
            && (report.title.toLowerCase().includes(query)
              || report.label.toLowerCase().includes(query)),
        )
        .map((report) => ({ kind: 'report' as const, option: report }))
    return [
      ...matchingCollections,
      ...matchingDocuments,
      ...matchingCanvasDocuments,
      ...matchingReports,
    ]
  }, [
    addableCanvasDocuments,
    addableCollections,
    documents,
    mention,
    reportIds,
    reportLimit,
    reportOptions,
    selectedDocumentId,
  ])
  const mentionOptions: MentionMenuOption[] = mentionCandidates.map(
    (candidate) =>
      candidate.kind === 'collection'
        ? {
          group: t.knowledge.collectionGroup,
          icon: Database,
          isCategory: false,
          primary: candidate.option.title,
          secondary: t.knowledge.collectionMenuHandle,
          tone: 'brand',
        }
        : candidate.kind === 'document'
          ? {
            group: t.navigation.editor,
            icon: FileText,
            isCategory: false,
            primary: candidate.option.title,
            secondary: t.agent.patch.title,
            tone: 'file',
          }
          : candidate.kind === 'canvas'
            ? {
              group: t.agent.composer.canvasDocGroup,
              icon: FileText,
              isCategory: false,
              primary: candidate.option.name,
              secondary: t.agent.composer.canvasDocChip,
              tone: 'brand',
            }
            : {
              group: t.agent.composer.reportGroup,
              icon: BookOpen,
              isCategory: false,
              // A research report has no title of its own — the run's
              // question is what the Research Desk shows as its name.
              primary: candidate.option.title,
              secondary: t.agent.composer.reportChip,
              tone: 'brand',
            },
  )

  const slashCandidates = useMemo(() => {
    if (!slash) return []
    const canAttachMore = selectedSkillIds.length < maxAttachedSkills
    const skillMatches = canAttachMore
      ? slashSkills
        .filter(
          (skill) =>
            !selectedSkillIds.includes(skill.id)
            && skill.label.includes(slash.query),
        )
        .map((skill) => ({ kind: 'skill' as const, skill }))
      : []
    const directiveMatches = EXECUTION_DIRECTIVE_OPTIONS.filter(
      (directive) =>
        executionDirective !== directive.id
        && executionDirectiveAvailability[directive.id]
        && (directive.token.includes(slash.query)
          || directive.id.includes(slash.query)),
    ).map((directive) => ({ kind: 'directive' as const, directive }))
    return [...directiveMatches, ...skillMatches]
  }, [
    executionDirective,
    executionDirectiveAvailability,
    maxAttachedSkills,
    selectedSkillIds,
    slash,
    slashSkills,
  ])
  const slashOptions: MentionMenuOption[] = slashCandidates.map((candidate) =>
    candidate.kind === 'skill'
      ? {
        group: t.skills.title,
        icon: Sparkles,
        isCategory: false,
        primary: `/${candidate.skill.label}`,
        secondary:
          candidate.skill.argument_hint || candidate.skill.description,
        tone: 'success',
      }
      : {
        group: t.agent.composer.slashCommands,
        icon: candidate.directive.id === 'quick_web' ? Globe2 : Database,
        isCategory: false,
        primary:
          candidate.directive.id === 'quick_web'
            ? t.agent.composer.directiveWeb
            : t.agent.composer.directiveKnowledge,
        secondary: t.agent.composer.directiveHint,
        tone: 'brand',
      })

  // answerMode overrides the running gate: a parked run WAITS for this
  // input — the send answers the clarification instead of racing a run.
  // P9d: attached comments make the submission self-sufficient — the
  // backend instructs the model to address every comment, so an empty
  // input may send. The visible default question below keeps the
  // transcript honest (nothing travels silently).
  const canSubmit =
    !disabled
    && !submitting
    && (answerMode || !running)
    && (question.trim().length > 0 || canvasComments.length > 0)
  const placeholder = answerMode
    ? t.agent.timeline.answerPlaceholder
    : canvasComments.length > 0
      ? t.agent.composer.canvasCommentsPlaceholder
      : t.agent.composer.placeholder

  function updateMentionFromTextarea(textarea: HTMLTextAreaElement) {
    // A selection event may fire against the PRE-commit DOM value right
    // after a menu selection replaced the text (focus/setSelectionRange
    // in the rAF) — detecting against that stale value would reopen the
    // menu on the removed token. Only the current text counts.
    if (textarea.value !== questionRef.current) return
    const caret = textarea.selectionStart ?? textarea.value.length
    const nextMention = detectCollectionMention(textarea.value, caret)
    setMention(nextMention)
    setSlash(nextMention ? null : detectSkillSlash(textarea.value, caret))
    setMentionIndex(0)
  }

  function handleQuestionChange(event: ChangeEvent<HTMLTextAreaElement>) {
    // Sync the ref BEFORE detection: the stale-value guard in
    // updateMentionFromTextarea compares against it, and the passive
    // echo effect only catches up after this render.
    questionRef.current = event.currentTarget.value
    setQuestion(event.currentTarget.value)
    updateMentionFromTextarea(event.currentTarget)
  }

  function handleCaretChange(event: SyntheticEvent<HTMLTextAreaElement>) {
    updateMentionFromTextarea(event.currentTarget)
  }

  function selectMentionOption(index: number) {
    const candidate = mentionCandidates[index]
    if (!candidate || !mention) return
    const end = mention.start + 1 + mention.query.length
    const nextValue = `${question.slice(0, mention.start)}${question.slice(end)}`
    if (candidate.kind === 'collection') {
      onSelectedCollectionIdsChange([
        ...selectedCollectionIds,
        candidate.option.id,
      ])
    } else if (candidate.kind === 'document') {
      onSelectedDocumentIdChange?.(candidate.option.id)
    } else if (candidate.kind === 'report') {
      onReportIdsChange?.([...reportIds, candidate.option.runId])
    } else {
      onPinnedCanvasDocumentChange?.(candidate.option.artifactId)
    }
    setQuestion(nextValue)
    setMention(null)
    window.requestAnimationFrame(() => {
      const textarea = textareaRef.current
      if (!textarea) return
      textarea.focus()
      textarea.setSelectionRange(mention.start, mention.start)
    })
  }

  function selectSlashOption(index: number) {
    const candidate = slashCandidates[index]
    if (!candidate || !slash) return
    const end = slash.start + 1 + slash.query.length
    const nextValue = `${question.slice(0, slash.start)}${question.slice(end)}`
    if (candidate.kind === 'skill') {
      onSelectedSkillIdsChange?.([...selectedSkillIds, candidate.skill.id])
    } else {
      onExecutionDirectiveChange?.(candidate.directive.id)
    }
    setQuestion(nextValue)
    setSlash(null)
    window.requestAnimationFrame(() => {
      const textarea = textareaRef.current
      if (!textarea) return
      textarea.focus()
      textarea.setSelectionRange(slash.start, slash.start)
    })
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (slash && slashCandidates.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        setMentionIndex((current) => (current + 1) % slashCandidates.length)
        return
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        setMentionIndex(
          (current) =>
            (current - 1 + slashCandidates.length) % slashCandidates.length,
        )
        return
      }
      const exactDirectCommand = EXECUTION_DIRECTIVE_OPTIONS.some(
        (directive) => directive.token === slash.query,
      )
      if (event.key === 'Tab' || (event.key === 'Enter' && !exactDirectCommand)) {
        event.preventDefault()
        selectSlashOption(mentionIndex)
        return
      }
    }
    if (slash && event.key === 'Escape') {
      event.preventDefault()
      setSlash(null)
      return
    }
    if (mention && mentionOptions.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        setMentionIndex((current) => (current + 1) % mentionOptions.length)
        return
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        setMentionIndex(
          (current) =>
            (current - 1 + mentionOptions.length) % mentionOptions.length,
        )
        return
      }
      if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault()
        selectMentionOption(mentionIndex)
        return
      }
    }
    if (mention && event.key === 'Escape') {
      event.preventDefault()
      setMention(null)
      return
    }
    if (
      event.key === 'Enter'
      && !event.ctrlKey
      && !event.metaKey
      && !event.shiftKey
      && !event.nativeEvent.isComposing
    ) {
      event.preventDefault()
      void submit()
    }
  }

  async function submit() {
    if (!canSubmit) return
    const trailing = extractTrailingExecutionDirective(question)
    if (trailing && !executionDirectiveAvailability[trailing.directive]) return
    // P9d: with attached comments an empty input sends the VISIBLE
    // default question (it lands in the bubble like any typed text).
    const submittedQuestion = (trailing?.question ?? question.trim())
      || (canvasComments.length > 0
        ? t.agent.composer.canvasCommentsDefaultQuestion
        : '')
    if (!submittedQuestion) return
    setSubmitting(true)
    try {
      const accepted = await onSubmit({
        autonomy,
        collectionIds: selectedCollectionIds,
        depth: depthMode,
        tier: tierMode,
        // Resolved against the current options — a selection whose document
        // vanished (deleted / sync toggled off) must not submit a stale id.
        documentId: selectedDocument?.id,
        engineMode,
        executionDirective: trailing?.directive ?? executionDirective ?? undefined,
        question: submittedQuestion,
        responseForm,
        skillIds: selectedSkillIds,
        sourcePolicy,
      })
      if (!accepted) return
      setQuestion('')
      setMention(null)
      setSlash(null)
    } finally {
      setSubmitting(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void submit()
  }

  return (
    <form className="agent-composer-container mx-auto max-w-5xl" onSubmit={handleSubmit}>
      {/* Stacked pending canvas comments (P9c/P9d): an own card ABOVE
          the composer frame — the editor attach panel's design language
          (frame, header, entrance animation) with the agent's rows
          (quote + text + pencil + trash, the GitHub pending mechanic). */}
      <AnimatePresence initial={false}>
        {canvasComments.length > 0 ? (
          <motion.div
            animate={{ height: 'auto', opacity: 1 }}
            className="mb-2 overflow-hidden rounded-md border border-brand/30 bg-brand-subtle/30"
            exit={{ height: 0, opacity: 0 }}
            initial={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
          >
            <div className="flex items-center gap-1.5 px-3 pb-0.5 pt-2 t-meta-sm font-semibold text-brand">
              <MessageSquareText aria-hidden="true" className="size-3.5" />
              {t.agent.composer.canvasCommentHeader.replace(
                '{count}',
                String(canvasComments.length),
              )}
            </div>
            <div className="flex flex-col px-2 pb-2">
              {canvasComments.map((comment, index) => (
                <div
                  className="flex min-w-0 items-center gap-1.5 rounded px-1 py-0.5"
                  key={comment.id}
                >
                  <span className="grid size-4 shrink-0 place-items-center rounded-[4px] bg-brand-subtle t-hint font-semibold tabular-nums text-brand">
                    {index + 1}
                  </span>
                  <span
                    className="max-w-44 shrink-0 truncate t-meta-sm text-muted-foreground"
                    title={comment.plainText || comment.quote}
                  >
                    „{comment.plainText || comment.quote}“
                  </span>
                  {/* Full text stays reachable at the cut (9b). */}
                  <span
                    className="min-w-0 flex-1 truncate t-meta-sm text-foreground"
                    title={comment.comment}
                  >
                    {comment.comment}
                  </span>
                  <Button
                    aria-label={t.agent.composer.editCanvasComment}
                    className="size-5 shrink-0 text-muted-foreground hover:text-foreground"
                    onClick={() => onEditCanvasComment?.(comment.id)}
                    size="icon"
                    type="button"
                    variant="ghost"
                  >
                    <PenLine className="size-3" />
                  </Button>
                  <Button
                    aria-label={t.agent.composer.removeCanvasComment}
                    className="size-5 shrink-0 text-muted-foreground hover:text-destructive"
                    onClick={() => onRemoveCanvasComment?.(comment.id)}
                    size="icon"
                    type="button"
                    variant="ghost"
                  >
                    <Trash2 className="size-3" />
                  </Button>
                </div>
              ))}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
        <div className="relative overflow-visible rounded-xl border border-border bg-card px-2.5 py-2 shadow-[0_8px_28px_-12px_var(--shadow-soft)] transition-[border-color,box-shadow] duration-150 focus-within:border-brand/60 focus-within:ring-2 focus-within:ring-brand/15">
        {mention && (
          <MentionMenu
            activeIndex={mentionIndex}
            labels={{
              backHint: t.chat.mentionBackHint,
              closeHint: t.chat.mentionCloseHint,
              filterPlaceholder: t.chat.mentionFilterPlaceholder,
              navHint: t.chat.mentionNavHint,
              rootTitle: t.knowledge.collectionPickerTitle,
              selectHint: t.chat.mentionSelectHint,
            }}
            onHover={setMentionIndex}
            onSelect={selectMentionOption}
            options={mentionOptions.length > 0
              ? mentionOptions
              : [{
                group: undefined,
                icon: Database,
                isCategory: false,
                primary: t.knowledge.noCollectionMatches,
                secondary: t.knowledge.collectionPickerHint,
                tone: 'brand',
              }]}
            scope={{
              icon: Database,
              kind: t.knowledge.collections,
              query: mention.query,
              tone: 'brand',
            }}
          />
        )}

        {slash && (
          <MentionMenu
            activeIndex={mentionIndex}
            labels={{
              backHint: t.chat.mentionBackHint,
              closeHint: t.chat.mentionCloseHint,
              filterPlaceholder: t.chat.mentionFilterPlaceholder,
              navHint: t.chat.mentionNavHint,
              rootTitle: t.agent.composer.slashMenuTitle,
              selectHint: t.chat.mentionSelectHint,
            }}
            onHover={setMentionIndex}
            onSelect={selectSlashOption}
            options={slashOptions.length > 0
              ? slashOptions
              : [{
                group: undefined,
                icon: Sparkles,
                isCategory: false,
                primary: selectedSkillIds.length >= maxAttachedSkills
                  ? t.agent.composer.slashCapReached.replace(
                    '{count}',
                    String(maxAttachedSkills),
                  )
                  : t.agent.composer.slashNoMatches,
                secondary: t.agent.composer.slashHint,
                tone: 'success',
              }]}
            scope={{
              icon: Sparkles,
              kind: t.agent.composer.slashMenuTitle,
              query: slash.query,
              tone: 'success',
            }}
          />
        )}

        {notice && (
          <div className="mb-1.5 rounded-md border border-warning/25 bg-warning-subtle px-2 py-1 t-meta-sm font-semibold text-warning">
            {notice}
          </div>
        )}

        {(selectedCollections.length > 0
          || selectedDocument
          || selectedSkillIds.length > 0
          || hasReportRequirement(reportGuidance, reportRuleIds)
          || reportIds.length > 0
          || pinnedCanvasDocument) && (
          <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
            {/* Mention-pinned canvas document (P9, K5): hidden while
                comments are queued — they bind the channel themselves. */}
            {pinnedCanvasDocument && canvasComments.length === 0 && (
              <Chip
                active
                aria-label={`${t.agent.composer.canvasDocChip}: ${pinnedCanvasDocument.name}`}
                dot="bg-brand"
                onClick={() => onPinnedCanvasDocumentChange?.(null)}
                title={pinnedCanvasDocument.title}
              >
                <FileText aria-hidden="true" className="size-3 shrink-0" />
                <span className="max-w-48 truncate">
                  {pinnedCanvasDocument.name}
                </span>
                <X aria-hidden="true" className="size-3 shrink-0" />
              </Chip>
            )}
            {/* Attached research reports (P14): the agent reads them
                on demand, so the chip is the only place the user sees
                WHAT the run was given. */}
            {reportIds.map((runId) => {
              const report = reportOptions.find(
                (option) => option.runId === runId,
              )
              return (
                <Chip
                  active
                  aria-label={`${t.agent.composer.reportChip}: ${report?.title ?? runId}`}
                  dot="bg-brand"
                  key={runId}
                  onClick={() =>
                    onReportIdsChange?.(
                      reportIds.filter((id) => id !== runId),
                    )}
                  title={report?.title ?? runId}
                >
                  <BookOpen aria-hidden="true" className="size-3 shrink-0" />
                  <span className="max-w-48 truncate">
                    {report?.title ?? runId}
                  </span>
                  <X aria-hidden="true" className="size-3 shrink-0" />
                </Chip>
              )
            })}
            {/* The result requirement, visible BEFORE sending: it is
                the one attachment whose effect the user cannot read off
                the answer afterwards, so it must be legible here. */}
            {reportRuleIds.map((ruleId) => {
              const rule = reportRuleOptions.find(
                (option) => option.ruleId === ruleId,
              )
              return (
                <Chip
                  active
                  aria-label={`${t.agent.composer.reportRequirement}: ${rule?.label ?? ruleId}`}
                  dot="bg-brand"
                  key={ruleId}
                  onClick={() =>
                    onReportRuleIdsChange?.(
                      reportRuleIds.filter((id) => id !== ruleId),
                    )}
                  title={rule?.title ?? ruleId}
                >
                  <BookOpen aria-hidden="true" className="size-3 shrink-0" />
                  <span className="max-w-48 truncate">
                    {rule?.label ?? ruleId}
                  </span>
                  <X aria-hidden="true" className="size-3 shrink-0" />
                </Chip>
              )
            })}
            {reportGuidance.trim() && (
              <Chip
                active
                aria-label={`${t.agent.composer.reportRequirement}: ${reportGuidance}`}
                dot="bg-brand"
                onClick={() => onReportGuidanceChange?.('')}
                /* The cut is visual only — the full text stays reachable
                   here and in the menu it was typed in (9b). */
                title={reportGuidance}
              >
                <WandSparkles aria-hidden="true" className="size-3 shrink-0" />
                <span className="max-w-48 truncate">
                  {reportGuidance.trim()}
                </span>
                <X aria-hidden="true" className="size-3 shrink-0" />
              </Chip>
            )}
            {selectedSkillIds.map((skillId) => {
              const skill = slashSkills.find((item) => item.id === skillId)
              return (
                <Chip
                  active
                  aria-label={`${t.skills.title}: ${skill?.label ?? skillId}`}
                  dot="bg-success"
                  key={skillId}
                  onClick={() =>
                    onSelectedSkillIdsChange?.(
                      selectedSkillIds.filter((id) => id !== skillId),
                    )}
                >
                  <span className="max-w-48 truncate">
                    /{skill?.label ?? skillId}
                  </span>
                  <X aria-hidden="true" className="size-3 shrink-0" />
                </Chip>
              )
            })}
            {selectedDocument && (
              <Chip
                active
                aria-label={`${t.agent.patch.title}: ${selectedDocument.title}`}
                dot="bg-file"
                onClick={() => onSelectedDocumentIdChange?.(null)}
                title={t.agent.patch.title}
              >
                <FileText aria-hidden="true" className="size-3 shrink-0" />
                <span className="max-w-48 truncate">{selectedDocument.title}</span>
                <X aria-hidden="true" className="size-3 shrink-0" />
              </Chip>
            )}
            {selectedCollections.map((collection) => (
              <Chip
                active
                aria-label={`${t.knowledge.removeCollection}: ${collection.title}`}
                dot="bg-brand"
                key={collection.id}
                onClick={() =>
                  onSelectedCollectionIdsChange(
                    selectedCollectionIds.filter((id) => id !== collection.id),
                  )}
                title={t.knowledge.removeCollection}
              >
                <span className="max-w-48 truncate">{collection.title}</span>
                <X aria-hidden="true" className="size-3 shrink-0" />
              </Chip>
            ))}
          </div>
        )}

        <Textarea
          aria-label={placeholder}
          className="min-h-16 resize-none border-0 bg-transparent pb-2 pl-2 pr-2 pt-2 text-sm font-normal leading-6 shadow-none placeholder:text-muted-foreground/70 focus-visible:ring-0"
          data-testid="agent-composer-input"
          disabled={disabled}
          onBlur={() => {
            setMention(null)
            setSlash(null)
          }}
          onChange={handleQuestionChange}
          onClick={handleCaretChange}
          onFocus={handleCaretChange}
          onKeyDown={handleKeyDown}
          onSelect={handleCaretChange}
          placeholder={placeholder}
          ref={textareaRef}
          rows={1}
          value={question}
        />

        <div className="agent-composer-footer mt-1.5 flex items-center justify-between gap-2 border-t border-border/70 pt-1.5">
          <div className="agent-composer-primary flex min-w-0 flex-1 items-center gap-1.5">
            <DropdownMenu modal={false}>
              <DropdownMenuTrigger asChild>
                <Button
                  aria-label={t.agent.composer.addContext}
                  className={composerIconButtonClassName}
                  disabled={
                    disabled
                    || (addableCollections.length === 0
                      && addableDocuments.length === 0
                      && addableCanvasDocuments.length === 0)
                  }
                  type="button"
                  variant="ghost"
                >
                  <Plus />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="start"
                className={cn(optionMenuContentClassName, 'w-80')}
                side="top"
                sideOffset={8}
              >
                <OptionMenuHeader
                  count={addableCollections.length}
                  title={t.knowledge.collectionPickerTitle}
                />
                {addableCollections.length > 0 ? (
                  <div className="py-1">
                    {addableCollections.map((collection) => (
                      <OptionMenuItem
                        active={false}
                        description={t.knowledge.collectionMenuHandle}
                        icon={Database}
                        key={collection.id}
                        label={collection.title}
                        onSelect={() =>
                          onSelectedCollectionIdsChange([
                            ...selectedCollectionIds,
                            collection.id,
                          ])}
                      />
                    ))}
                  </div>
                ) : (
                  <p className="px-2.5 py-2 t-meta text-muted-foreground">
                    {t.knowledge.allCollectionsAdded}
                  </p>
                )}
                {documents.length > 0 ? (
                  <>
                    <DropdownMenuSeparator className="mx-0 my-1" />
                    <OptionMenuHeader
                      count={addableDocuments.length}
                      title={t.agent.composer.editTarget}
                    />
                    {addableDocuments.length > 0 ? (
                      <div className="py-1">
                        {addableDocuments.map((document) => (
                          <OptionMenuItem
                            active={false}
                            description={t.agent.patch.title}
                            icon={FileText}
                            key={document.id}
                            label={document.title}
                            onSelect={() =>
                              onSelectedDocumentIdChange?.(document.id)}
                          />
                        ))}
                      </div>
                    ) : null}
                  </>
                ) : null}
                {onReportIdsChange && reportOptions.length > 0 ? (
                  <>
                    <DropdownMenuSeparator className="mx-0 my-1" />
                    <OptionMenuHeader
                      count={reportOptions.length - reportIds.length}
                      title={t.agent.composer.reportGroup}
                    />
                    {reportIds.length >= reportLimit ? (
                      <p className="px-2.5 py-2 t-meta text-muted-foreground">
                        {t.agent.composer.reportCap.replace(
                          '{max}',
                          String(reportLimit),
                        )}
                      </p>
                    ) : (
                      <div className="py-1">
                        {reportOptions
                          .filter((report) => !reportIds.includes(report.runId))
                          .map((report) => (
                            <OptionMenuItem
                              active={false}
                              description={t.agent.composer.reportChip}
                              icon={BookOpen}
                              key={report.runId}
                              label={report.title}
                              onSelect={() =>
                                onReportIdsChange([
                                  ...reportIds,
                                  report.runId,
                                ])}
                            />
                          ))}
                      </div>
                    )}
                  </>
                ) : null}
                {addableCanvasDocuments.length > 0 ? (
                  <>
                    <DropdownMenuSeparator className="mx-0 my-1" />
                    <OptionMenuHeader
                      count={addableCanvasDocuments.length}
                      title={t.agent.composer.canvasDocGroup}
                    />
                    <div className="py-1">
                      {addableCanvasDocuments.map((document) => (
                        <OptionMenuItem
                          active={false}
                          description={t.agent.composer.canvasDocChip}
                          icon={FileText}
                          key={document.artifactId}
                          label={document.name}
                          onSelect={() =>
                            onPinnedCanvasDocumentChange?.(
                              document.artifactId,
                            )}
                        />
                      ))}
                    </div>
                  </>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>

            <div
              aria-label={t.agent.composer.sources}
              className="agent-source-dock flex h-7 shrink-0 items-center gap-1"
              role="group"
            >
              {(['web', 'knowledge'] as const).map((source) => {
                const isAvailable = sourceAvailability[source]
                const isEnabled = sourcePolicy[source] === 'available'
                const isForced =
                  (source === 'web' && executionDirective === 'quick_web')
                  || (source === 'knowledge'
                    && executionDirective === 'knowledge_only')
                const isSuppressed = Boolean(executionDirective) && !isForced
                const isEffectivelyEnabled = isForced
                  || (!executionDirective && isEnabled)
                const Icon = source === 'web' ? Globe2 : BookSearch
                const label = source === 'web'
                  ? t.agent.composer.sourceWeb
                  : t.agent.composer.sourceKnowledge
                const longLabel = source === 'web'
                  ? t.agent.composer.sourceWebLong
                  : t.agent.composer.sourceKnowledgeLong
                const stateHint = !isAvailable || disabled
                  ? t.agent.composer.sourceUnavailable
                  : isForced
                    ? t.agent.composer.sourceForced
                    : isSuppressed
                      ? t.agent.composer.sourceOverridden
                    : isEnabled
                      ? t.agent.composer.sourceAvailable
                      : t.agent.composer.sourceDisabled
                const revealLabel = (
                  !isAvailable || disabled
                    ? t.agent.composer.sourceUnavailableShort
                    : isForced
                      ? t.agent.composer.sourceForcedShort
                      : isSuppressed
                        ? t.agent.composer.sourceOverriddenShort
                        : isEnabled
                          ? t.agent.composer.sourceAvailableShort
                          : t.agent.composer.sourceDisabledShort
                ).replace('{source}', longLabel)
                return (
                  <button
                    aria-description={stateHint}
                    aria-disabled={!isAvailable || disabled}
                    aria-label={label}
                    aria-pressed={isEffectivelyEnabled}
                    className={cn(
                      'agent-source-toggle relative inline-flex h-6 min-w-6 items-center justify-center gap-1 overflow-hidden rounded px-1.5 text-xs font-medium',
                      isEffectivelyEnabled && !isForced
                        && 'agent-source-toggle-active',
                      isForced
                        ? 'agent-source-toggle-forced bg-brand-subtle text-brand shadow-[0_1px_2px_var(--shadow-hairline)]'
                        : isEffectivelyEnabled && isAvailable
                          ? 'bg-background text-foreground shadow-[0_1px_2px_var(--shadow-hairline)]'
                          : 'text-muted-foreground/60',
                      (!isAvailable || disabled)
                        && 'cursor-not-allowed opacity-50',
                    )}
                    key={source}
                    data-source={source}
                    data-source-state={
                      !isAvailable || disabled
                        ? 'unavailable'
                        : isForced
                          ? 'forced'
                          : isSuppressed
                            ? 'suppressed'
                            : isEnabled
                              ? 'available'
                              : 'disabled'
                    }
                    onClick={() => {
                      if (!isAvailable || disabled) return
                      if (executionDirective) {
                        onExecutionDirectiveChange?.(null)
                        return
                      }
                      onSourcePolicyChange({
                        ...sourcePolicy,
                        [source]: isEnabled ? 'disabled' : 'available',
                      })
                    }}
                    type="button"
                  >
                    <span
                      aria-hidden="true"
                      className={cn(
                        'agent-source-glyph relative grid size-4 shrink-0 place-items-center',
                        source === 'web'
                          ? 'agent-source-glyph-web'
                          : 'agent-source-glyph-knowledge',
                      )}
                    >
                      <Icon className="icon-sm" />
                      <span className="agent-source-glyph-tracer" />
                    </span>
                    <span className="agent-source-toggle-label grid">
                      <span className="overflow-hidden whitespace-nowrap">{revealLabel}</span>
                    </span>
                    {isForced ? (
                      <Zap
                        aria-hidden="true"
                        className="agent-source-force-mark absolute -right-0.5 -top-0.5 size-2.5 rounded-full bg-brand text-brand-foreground"
                      />
                    ) : null}
                  </button>
                )
              })}
            </div>

            <div className="agent-execution-capsule relative flex min-w-0 shrink items-center gap-0.5 overflow-visible">
              <DropdownMenu modal={false}>
                <DropdownMenuTrigger asChild>
                  <Button
                    aria-label={t.agent.composer.runSetup}
                    className="agent-run-setup-trigger h-7 min-w-0 shrink gap-1.5 rounded-md px-2 text-xs font-semibold text-muted-foreground [&_svg]:size-3.5 hover:bg-accent/70 hover:text-foreground data-[state=open]:bg-accent data-[state=open]:text-foreground"
                    disabled={disabled}
                    type="button"
                    variant="ghost"
                  >
                    <Workflow aria-hidden="true" className="agent-route-icon icon-sm shrink-0" />
                    <span className="agent-run-setup-label min-w-0 truncate">
                      {engineModeLabel(engineMode, t)} · {autonomyModeLabel(autonomy, t)}
                    </span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="start"
                  className={cn(
                    optionMenuContentClassName,
                    'w-96 max-w-[calc(100vw-2rem)]',
                  )}
                  side="top"
                  sideOffset={8}
                >
                  <OptionMenuHeader
                    count={0}
                    title={t.agent.composer.runSetup}
                    value={`${engineModeLabel(engineMode, t)} · ${autonomyModeLabel(autonomy, t)}`}
                  />
                  <div className="py-1">
                    <p className="px-2.5 pb-1 pt-1.5 t-caption text-muted-foreground/65">
                      {t.agent.composer.autonomy}
                    </p>
                    {autonomyModes.map((mode) => (
                      <OptionMenuItem
                        active={autonomy === mode}
                        description={autonomyModeHint(mode, t)}
                        descriptionLines={2}
                        icon={mode === 'autonomous' ? ShieldCheck : Shield}
                        keepOpen
                        key={mode}
                        label={autonomyModeLabel(mode, t)}
                        onSelect={() => onAutonomyChange(mode)}
                      />
                    ))}
                    <DropdownMenuSeparator className="mx-0 my-1" />
                    <p className="px-2.5 pb-1 pt-1.5 t-caption text-muted-foreground/65">
                      {t.agent.composer.engine}
                    </p>
                    {(kernelSelectable
                      ? (['agent_kernel', 'workspace_agent'] as const)
                      : (['workspace_agent'] as const)
                    ).map((mode) => (
                      <OptionMenuItem
                        active={engineMode === mode}
                        description={
                          mode === 'agent_kernel'
                            ? t.agent.composer.engineKernelHint
                            : t.agent.composer.engineMissionHint
                        }
                        descriptionLines={2}
                        icon={mode === 'agent_kernel' ? BrainCircuit : Waypoints}
                        keepOpen
                        key={mode}
                        label={engineModeLabel(mode, t)}
                        onSelect={() => onEngineModeChange?.(mode)}
                      />
                    ))}
                    {onReportGuidanceChange || onReportRuleIdsChange ? (
                      <>
                        <DropdownMenuSeparator className="mx-0 my-1" />
                        <p className="px-2.5 pb-1 pt-1.5 t-caption text-muted-foreground/65">
                          {t.agent.composer.reportRequirement}
                        </p>
                        {/* The result requirement, BEFORE the run (S6).
                            The plan gate carries the same pair, but a
                            run in Auto, on the speed tier, or in the
                            kernel never reaches one — so without this
                            the requirement had no entry point at all. */}
                        {reportRuleOptions.length > 0 && onReportRuleIdsChange ? (
                          <div className="flex flex-wrap items-center gap-1 px-2.5 pb-1.5">
                            {reportRuleOptions.map((option) => {
                              const active = reportRuleIds.includes(option.ruleId)
                              const blocked =
                                !active && reportRuleIds.length >= reportRuleLimit
                              return (
                                <button
                                  aria-pressed={active}
                                  className={cn(
                                    'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 t-hint transition-colors',
                                    active
                                      ? 'border-brand/50 bg-brand/10 text-brand'
                                      : 'border-border text-muted-foreground hover:text-foreground',
                                    blocked
                                      && 'cursor-not-allowed opacity-40 hover:text-muted-foreground',
                                  )}
                                  disabled={blocked || disabled}
                                  key={option.ruleId}
                                  onClick={() =>
                                    onReportRuleIdsChange(
                                      toggleReportRule(
                                        reportRuleIds,
                                        option.ruleId,
                                        reportRuleLimit,
                                      ),
                                    )}
                                  title={
                                    blocked
                                      ? t.agent.plan.reportRulesCap.replace(
                                        '{max}',
                                        String(reportRuleLimit),
                                      )
                                      : option.title
                                  }
                                  type="button"
                                >
                                  <BookOpen aria-hidden="true" className="icon-xs" />
                                  {option.label}
                                </button>
                              )
                            })}
                          </div>
                        ) : null}
                        {onReportGuidanceChange ? (
                          <div className="px-2.5 pb-1.5">
                            <textarea
                              aria-label={t.agent.composer.reportRequirement}
                              className="w-full resize-none rounded-md border border-border bg-card px-2.5 py-1.5 t-meta text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                              disabled={disabled}
                              maxLength={reportGuidanceLimit}
                              onChange={(event) =>
                                onReportGuidanceChange(event.target.value)}
                              // A menu closes on Enter and treats typed
                              // characters as type-ahead; the textarea
                              // keeps those to itself. Escape is NOT
                              // one of them — swallowing it left the
                              // menu open with no keyboard way out
                              // (observed live), so it travels on.
                              onKeyDown={(event) => {
                                if (event.key === 'Escape') return
                                event.stopPropagation()
                              }}
                              placeholder={t.agent.composer.reportRequirementPlaceholder}
                              rows={2}
                              value={reportGuidance}
                            />
                            <p className="mt-0.5 t-hint text-muted-foreground/70">
                              {t.agent.composer.reportRequirementHint}
                            </p>
                          </div>
                        ) : null}
                      </>
                    ) : null}
                    {onResponseFormChange ? (
                      <>
                        <DropdownMenuSeparator className="mx-0 my-1" />
                        <p className="px-2.5 pb-1 pt-1.5 t-caption text-muted-foreground/65">
                          {t.agent.composer.responseForm}
                        </p>
                        {AGENT_RESPONSE_FORMS.map((form) => (
                          <OptionMenuItem
                            active={responseForm === form}
                            description={responseFormHint(form, t)}
                            descriptionLines={2}
                            icon={responseFormIcon(form)}
                            keepOpen
                            key={form}
                            label={responseFormLabel(form, t)}
                            onSelect={() => onResponseFormChange(form)}
                          />
                        ))}
                      </>
                    ) : null}
                    {tiers && tiers.length > 0 && onTierModeChange ? (
                      <>
                        <DropdownMenuSeparator className="mx-0 my-1" />
                        <p className="px-2.5 pb-1 pt-1.5 t-caption text-muted-foreground/65">
                          {t.agent.composer.tierTitle}
                        </p>
                        {/* No extra "no tier" entry: the composer always
                            submits the effective Stufe (selected ?? the
                            published default), so the active mark on the
                            default IS the honest state (B6). */}
                        {tiers.map((tier) => (
                          <OptionMenuItem
                            active={tierMode === tier.id}
                            description={`${tierHint(tier.id, t)} · ${t.agent.composer.tierLatencyHint}`}
                            descriptionLines={2}
                            icon={tierIcon(tier.id)}
                            keepOpen
                            key={tier.id}
                            label={tierLabel(tier.id, t)}
                            onSelect={() => onTierModeChange(tier.id)}
                          />
                        ))}
                      </>
                    ) : depthSelectable && onDepthModeChange ? (
                      <>
                        <DropdownMenuSeparator className="mx-0 my-1" />
                        <p className="px-2.5 pb-1 pt-1.5 t-caption text-muted-foreground/65">
                          {t.agent.overview.rowDepth}
                        </p>
                        <OptionMenuItem
                          active={depthMode === 'normal'}
                          description={t.agent.composer.depthNormalHint}
                          descriptionLines={2}
                          icon={Gauge}
                          keepOpen
                          label={t.agent.overview.depthNormal}
                          onSelect={() => onDepthModeChange('normal')}
                        />
                        <OptionMenuItem
                          active={depthMode === 'deep'}
                          description={`${t.agent.composer.deepHint} ${t.agent.composer.deepTags}`}
                          descriptionLines={2}
                          icon={Layers}
                          keepOpen
                          label={t.agent.composer.deep}
                          onSelect={() => onDepthModeChange('deep')}
                        />
                      </>
                    ) : null}
                  </div>
                </DropdownMenuContent>
              </DropdownMenu>

              {modelPicker && (
                <ModelTierPicker
                  defaultModel={modelPicker.defaultModel}
                  disabled={disabled}
                  modelCatalog={modelPicker.catalog}
                  onChange={modelPicker.onTierChange}
                  onEffortChange={modelPicker.onEffortChange}
                  onModelChange={modelPicker.onModelChange}
                  options={modelPicker.options}
                  optionsStatus={modelPicker.optionsStatus}
                  selectedEffort={modelPicker.selectedEffort}
                  selectedModel={modelPicker.selectedModel}
                  selectedTier={modelPicker.selectedTier}
                  serverDefaultDescription={t.agent.composer.modelAutoDescription}
                  pickerTitle={t.agent.composer.modelPickerTitle}
                  serverDefaultLabel={t.agent.composer.modelAuto}
                  triggerPrefix={`${t.agent.composer.model}: `}
                  triggerVariant="agent-capsule"
                />
              )}
              <span aria-hidden="true" className="agent-execution-rail" />
            </div>
          </div>

          <div className="agent-composer-actions flex shrink-0 items-center gap-1.5">
            <span className="agent-composer-quota contents">
              <QuotaMeter disabled={disabled} />
            </span>
            {overview && (
              <AgentStatusMenu
                autonomyHint={autonomyModeHint(autonomy, t)}
                depthDeep={tierMode ? tierMode === 'tief' : depthMode === 'deep'}
                autonomyLabel={autonomyModeLabel(autonomy, t)}
                autonomyMode={autonomy}
                disabled={disabled}
                memoryEnabled={memoryEnabled}
                modelValue={agentModelOverviewValue(modelPicker, t)}
                overview={overview}
                responseFormValue={responseFormLabel(responseForm, t)}
                responseForm={responseForm}
                tier={tierMode}
                execution={statusExecution}
                executionDirective={executionDirective}
                sourcePolicy={sourcePolicy}
                toolUseCounts={toolUseCounts}
              />
            )}
            {running && !answerMode ? (
              <ComposerStopButton label={t.agent.composer.stop} onClick={onStop} />
            ) : (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    aria-label={t.agent.composer.submit}
                    className="size-7 shrink-0 rounded-md bg-brand text-brand-foreground hover:bg-brand/90"
                    data-testid="agent-submit"
                    disabled={!canSubmit}
                    size="icon"
                    type="submit"
                  >
                    <SendHorizontal className="size-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{t.agent.composer.submit}</TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>
      </div>
      <ComposerDisclosureHint />
    </form>
  )
}

function engineModeLabel(
  mode: AgentEngineMode,
  t: ReturnType<typeof useLocale>['t'],
): string {
  return mode === 'agent_kernel'
    ? t.agent.composer.engineAuto
    : t.agent.overview.brainMission
}

function responseFormIcon(form: AgentResponseForm) {
  if (form === 'chat') return MessageSquareText
  if (form === 'canvas') return FileText
  return WandSparkles
}

function responseFormLabel(
  form: AgentResponseForm,
  t: ReturnType<typeof useLocale>['t'],
): string {
  if (form === 'chat') return t.agent.composer.responseFormChat
  if (form === 'canvas') return t.agent.composer.responseFormCanvas
  return t.agent.composer.responseFormAuto
}

function responseFormHint(
  form: AgentResponseForm,
  t: ReturnType<typeof useLocale>['t'],
): string {
  if (form === 'chat') return t.agent.composer.responseFormChatHint
  if (form === 'canvas') return t.agent.composer.responseFormCanvasHint
  return t.agent.composer.responseFormAutoHint
}

function agentModelOverviewValue(
  picker: AgentModelPickerProps | null,
  t: ReturnType<typeof useLocale>['t'],
): string {
  if (!picker) return ''
  if (picker.selectedModel) {
    const card = picker.catalog.find(
      (entry) => entry.model_id === picker.selectedModel,
    )?.card
    const name = card?.display_name ?? picker.selectedModel
    const effort = picker.selectedEffort
      ? ` \u00b7 ${effortLevelLabel(picker.selectedEffort)}`
      : ''
    return t.agent.overview.modelOverrideValue.replace(
      '{name}',
      `${name}${effort}`,
    )
  }
  if (picker.selectedTier) {
    return t.agent.overview.modelOverrideValue.replace(
      '{name}',
      picker.selectedTier,
    )
  }
  return t.agent.overview.modelAutoValue
}

function autonomyModeLabel(
  mode: string,
  t: ReturnType<typeof useLocale>['t'],
): string {
  if (mode === 'strict') return t.agent.composer.autonomyStrict
  if (mode === 'autonomous') return t.agent.composer.autonomyAutonomous
  return t.agent.composer.autonomyBalanced
}

function autonomyModeHint(
  mode: string,
  t: ReturnType<typeof useLocale>['t'],
): string {
  if (mode === 'strict') return t.agent.composer.autonomyStrictHint
  if (mode === 'autonomous') return t.agent.composer.autonomyAutonomousHint
  return t.agent.composer.autonomyBalancedHint
}

/** Stufe display helpers — one label/hint vocabulary per tier id. */
function tierLabel(
  tier: AgentTierId,
  t: ReturnType<typeof useLocale>['t'],
): string {
  if (tier === 'schnell') return t.agent.composer.tierSchnell
  if (tier === 'tief') return t.agent.composer.tierTief
  return t.agent.composer.tierGruendlich
}

function tierHint(
  tier: AgentTierId,
  t: ReturnType<typeof useLocale>['t'],
): string {
  if (tier === 'schnell') return t.agent.composer.tierSchnellHint
  if (tier === 'tief') return t.agent.composer.tierTiefHint
  return t.agent.composer.tierGruendlichHint
}

function tierIcon(tier: AgentTierId) {
  if (tier === 'schnell') return Gauge
  if (tier === 'tief') return Layers
  return Search
}
