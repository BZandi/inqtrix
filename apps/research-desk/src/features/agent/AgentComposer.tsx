import { useMemo, useRef, useState, useEffect, useLayoutEffect } from 'react'
import type {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  SyntheticEvent,
} from 'react'

import {
  BookSearch,
  BrainCircuit,
  Database,
  FileText,
  Globe2,
  Layers,
  MessageSquareText,
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

export type AgentCollectionOption = { id: string; title: string }

export type AgentDocumentOption = { id: string; title: string }

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
  onSelectedCollectionIdsChange,
  onSelectedDocumentIdChange,
  onResponseFormChange,
  onStop,
  onSubmit,
  onSelectedSkillIdsChange,
  onExecutionDirectiveChange,
  onSourcePolicyChange,
  overview = null,
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
  collections: AgentCollectionOption[]
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
    return [...matchingCollections, ...matchingDocuments]
  }, [addableCollections, documents, mention, selectedDocumentId])
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
        : {
          group: t.navigation.editor,
          icon: FileText,
          isCategory: false,
          primary: candidate.option.title,
          secondary: t.agent.patch.title,
          tone: 'file',
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
  const canSubmit =
    !disabled
    && !submitting
    && (answerMode || !running)
    && question.trim().length > 0
  const placeholder = answerMode
    ? t.agent.timeline.answerPlaceholder
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
    } else {
      onSelectedDocumentIdChange?.(candidate.option.id)
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
    const submittedQuestion = trailing?.question ?? question.trim()
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
          || selectedSkillIds.length > 0) && (
          <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
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
                      && addableDocuments.length === 0)
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
                            description={`${tierHint(tier.id, t)} · ${tier.latency_hint}`}
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
