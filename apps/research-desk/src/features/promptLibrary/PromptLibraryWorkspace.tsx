import { useMemo, useState, type Dispatch } from 'react'
import { useReducedMotion } from 'motion/react'
import {
  BookOpen,
  Bot,
  Check,
  EyeOff,
  FolderOpen,
  Library,
  ListOrdered,
  Paperclip,
  Plus,
  Save,
  Search,
  Trash2,
  X,
  type LucideIcon,
} from '@/components/icons'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import {
  TextImproveButton,
  TextImproveFieldLayer,
  useTextImprovement,
  type TextImprovementApiOptions,
} from '@/features/textImprove'
import {
  chatAttachmentChipsFromRefs,
  chatContextRefKey,
  dedupeChatContextRefs,
  fileGroupMentionOptions,
  fileMentionOptions,
  projectChatRules,
} from '@/features/project/selectors'
import {
  chatRuleCategories,
  compareChatRulesByCategory,
  normalizeChatRule,
  normalizeLinkedContextRefs,
} from '@/features/project/chatRules'
import {
  contextPackPlaceholder,
  renderChatRuleAttachmentContent,
} from '@/features/project/chatRuleRendering'
import type {
  ChatContextReferenceRecord,
  ChatRuleCategory,
  ChatRuleRecord,
  ChatRuleVisibility,
  ProjectState,
} from '@/features/project/types'
import type { ResearchDeskAction } from '@/features/researchDesk/state'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { createRuleId, normalizeRuleLabel } from '@/features/chat/rules/ruleLabels'

type CategoryFilter = ChatRuleCategory | 'all'
type VisibilityFilter = 'all' | 'chat' | 'editor' | 'hidden'
type AutocompleteFilter = 'all' | 'hidden' | 'visible'

type PromptDraft = {
  category: ChatRuleCategory
  contentMarkdown: string
  error: string | null
  includeInAutocomplete: boolean
  isDirty: boolean
  label: string
  linkedContextRefs: ChatContextReferenceRecord[]
  selectedRuleId: string | null
  title: string
  visibility: ChatRuleVisibility
}

type ContextPickerOption = {
  icon: LucideIcon
  key: string
  label: string
  ref: ChatContextReferenceRecord
  title: string
  type: 'file' | 'group'
}

const contextPickerLimit = 8

const emptyDraft: PromptDraft = {
  category: 'instruction',
  contentMarkdown: '',
  error: null,
  includeInAutocomplete: true,
  isDirty: false,
  label: '',
  linkedContextRefs: [],
  selectedRuleId: null,
  title: '',
  visibility: { chat: true, editor: true },
}

export function PromptLibraryWorkspace({
  dispatch,
  state,
  textImprovement,
}: {
  dispatch: Dispatch<ResearchDeskAction>
  state: ProjectState
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}) {
  const { locale, t } = useLocale()
  const reduceMotion = useReducedMotion()
  const rules = useMemo(
    () => projectChatRules(state).toSorted(compareChatRulesByCategory),
    [state.chatRuleOrder, state.chatRules],
  )
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>('all')
  const [visibilityFilter, setVisibilityFilter] = useState<VisibilityFilter>('all')
  const [autocompleteFilter, setAutocompleteFilter] = useState<AutocompleteFilter>('all')
  const [query, setQuery] = useState('')
  const [contextQuery, setContextQuery] = useState('')
  const [isPreviewOpen, setPreviewOpen] = useState(false)
  const [promptImproveError, setPromptImproveError] = useState<string | null>(null)
  const [draft, setDraft] = useState<PromptDraft>(() => draftFromRule(rules[0] ?? null))
  const promptTextImprove = useTextImprovement({
    ...textImprovement,
    locale,
    messages: {
      requestFailed: (message) => `${t.textImprove.requestFailed}: ${message}`,
      sensitiveText: t.textImprove.sensitiveText,
      unavailable: t.textImprove.unavailable,
    },
  })
  const selectedRule = draft.selectedRuleId
    ? rules.find((rule) => rule.id === draft.selectedRuleId) ?? null
    : null
  const fileOptions = fileMentionOptions(state)
  const fileGroupOptions = fileGroupMentionOptions(state)
  const contextOptions = useMemo(
    () => [
      ...fileGroupOptions.map<ContextPickerOption>((group) => ({
        icon: FolderOpen,
        key: `file-group:${group.groupId}`,
        label: `@filegroups:${group.label}`,
        ref: { groupId: group.groupId, kind: 'file-group' as const },
        title: group.title,
        type: 'group' as const,
      })),
      ...fileOptions.map<ContextPickerOption>((file) => ({
        icon: Paperclip,
        key: `file-asset:${file.fileId}`,
        label: `@files:${file.label}`,
        ref: { fileId: file.fileId, kind: 'file-asset' as const },
        title: file.title,
        type: 'file' as const,
      })),
    ],
    [fileGroupOptions, fileOptions],
  )
  const contextOptionByKey = useMemo(
    () => new Map(contextOptions.map((option) => [option.key, option])),
    [contextOptions],
  )
  const contextChips = chatAttachmentChipsFromRefs(state, draft.linkedContextRefs)
  const filteredRules = rules.filter((rule) => {
    const normalized = normalizeChatRule(rule)
    const visibility = normalized.visibility ?? { chat: true, editor: true }
    const isAutocompleteVisible = normalized.includeInAutocomplete !== false
    const haystack = [
      normalized.label,
      normalized.title,
      normalized.contentMarkdown,
      categoryLabel(normalized.category ?? 'instruction', t),
    ].join(' ').toLowerCase()
    const trimmedQuery = query.trim().toLowerCase()
    const matchesQuery = !trimmedQuery || haystack.includes(trimmedQuery)
    const matchesCategory = categoryFilter === 'all' || normalized.category === categoryFilter
    const matchesVisibility = visibilityFilter === 'all'
      || (visibilityFilter === 'chat' && isAutocompleteVisible && visibility.chat)
      || (visibilityFilter === 'editor' && isAutocompleteVisible && visibility.editor)
      || (visibilityFilter === 'hidden' && (!isAutocompleteVisible || (!visibility.chat && !visibility.editor)))
    const matchesAutocomplete = autocompleteFilter === 'all'
      || (autocompleteFilter === 'visible' && isAutocompleteVisible)
      || (autocompleteFilter === 'hidden' && !isAutocompleteVisible)
    return matchesQuery && matchesCategory && matchesVisibility && matchesAutocomplete
  })
  const groupedRules = chatRuleCategories
    .map((category) => ({
      category,
      rules: filteredRules.filter((rule) => normalizeChatRule(rule).category === category),
    }))
    .filter((group) => group.rules.length > 0)
  const previewRule = draftRuleRecord(draft, selectedRule)
  const preview = renderChatRuleAttachmentContent(state, previewRule, new Date().toISOString())

  function loadRule(rule: ChatRuleRecord | null) {
    setDraft(draftFromRule(rule))
    setContextQuery('')
    setPromptImproveError(null)
    promptTextImprove.clearProposal()
    setPreviewOpen(false)
  }

  function updateDraft(patch: Partial<PromptDraft>) {
    setDraft((current) => ({
      ...current,
      ...patch,
      error: patch.error === undefined ? null : patch.error,
      isDirty: patch.isDirty ?? true,
    }))
  }

  function setVisibility(key: keyof ChatRuleVisibility, value: boolean) {
    if (!draft.includeInAutocomplete) return
    updateDraft({
      visibility: {
        ...draft.visibility,
        [key]: value,
      },
    })
  }

  function setAutocomplete(includeInAutocomplete: boolean) {
    updateDraft({
      includeInAutocomplete,
      visibility: includeInAutocomplete
        ? draft.visibility.chat || draft.visibility.editor
          ? draft.visibility
          : { chat: true, editor: true }
        : { chat: false, editor: false },
    })
  }

  function toggleContextRef(ref: ChatContextReferenceRecord) {
    const key = chatContextRefKey(ref)
    const exists = draft.linkedContextRefs.some((item) => chatContextRefKey(item) === key)
    updateDraft({
      linkedContextRefs: exists
        ? draft.linkedContextRefs.filter((item) => chatContextRefKey(item) !== key)
        : dedupeChatContextRefs([...draft.linkedContextRefs, ref]),
    })
  }

  function removeContextRef(ref: ChatContextReferenceRecord) {
    const key = chatContextRefKey(ref)
    updateDraft({
      linkedContextRefs: draft.linkedContextRefs.filter((item) => chatContextRefKey(item) !== key),
    })
  }

  function savePrompt() {
    const label = normalizeRuleLabel(draft.label)
    const title = draft.title.trim() || label
    const contentMarkdown = draft.contentMarkdown.trim()
    if (!label) {
      updateDraft({ error: t.promptLibrary.labelRequired, isDirty: draft.isDirty })
      return
    }
    if (rules.some((rule) => rule.label === label && rule.id !== draft.selectedRuleId)) {
      updateDraft({ error: t.promptLibrary.labelDuplicate, isDirty: draft.isDirty })
      return
    }
    if (!contentMarkdown) {
      updateDraft({ error: t.promptLibrary.promptRequired, isDirty: draft.isDirty })
      return
    }
    const now = new Date().toISOString()
    const rule = normalizeChatRule({
      category: draft.category,
      contentMarkdown,
      createdAt: selectedRule?.createdAt ?? now,
      id: draft.selectedRuleId ?? createRuleId(),
      includeInAutocomplete: draft.includeInAutocomplete,
      label,
      linkedContextRefs: draft.category === 'context'
        ? normalizeLinkedContextRefs(draft.linkedContextRefs)
        : [],
      title,
      updatedAt: now,
      visibility: draft.visibility,
    })
    dispatch({ rule, type: 'upsertChatRule' })
    setDraft(draftFromRule(rule))
  }

  function deletePrompt() {
    if (!selectedRule) return
    dispatch({ ruleId: selectedRule.id, type: 'deleteChatRule' })
    const next = rules.find((rule) => rule.id !== selectedRule.id) ?? null
    loadRule(next)
  }

  async function improvePrompt() {
    setPromptImproveError(null)
    try {
      await promptTextImprove.improve(
        'prompt_template',
        draft.contentMarkdown,
        promptImprovementGuidance(draft, t),
      )
    } catch (error) {
      setPromptImproveError(messageFromUnknown(error))
    }
  }

  function acceptPromptImprovement(contentMarkdown: string) {
    updateDraft({ contentMarkdown })
    promptTextImprove.clearProposal()
    setPromptImproveError(null)
  }

  return (
    <div className="grid h-[calc(100svh-var(--header-h))] min-h-0 bg-background lg:grid-cols-[320px_minmax(0,1fr)]">
      <aside className="flex min-h-0 min-w-0 flex-col border-b border-border bg-surface/50 lg:border-b-0 lg:border-r">
        <div className="border-b border-border p-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-2.5">
              <span className="grid size-9 place-items-center rounded-lg border border-brand/20 bg-brand-subtle text-brand">
                <Library className="size-4" />
              </span>
              <div className="min-w-0">
                <h1 className="truncate text-base font-semibold text-foreground">{t.promptLibrary.title}</h1>
                <p className="truncate text-xs text-muted-foreground">{t.promptLibrary.subtitle}</p>
              </div>
            </div>
            <Button aria-label={t.promptLibrary.newPrompt} className="size-8" onClick={() => loadRule(null)} size="icon" type="button" variant="outline">
              <Plus className="size-4" />
            </Button>
          </div>
          <label className="mt-3 flex items-center gap-2 rounded-md border border-border bg-background px-2 focus-within:ring-2 focus-within:ring-ring">
            <Search className="size-4 shrink-0 text-muted-foreground" />
            <input
              className="min-w-0 flex-1 border-0 bg-transparent py-2 text-sm text-foreground outline-none"
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t.promptLibrary.searchPlaceholder}
              value={query}
            />
          </label>
          <div className="mt-3 grid grid-cols-1 gap-2">
            <select
              className="h-8 rounded-md border border-border bg-background px-2 text-xs font-medium text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onChange={(event) => setCategoryFilter(event.target.value as CategoryFilter)}
              value={categoryFilter}
            >
              <option value="all">{t.promptLibrary.allCategories}</option>
              {chatRuleCategories.map((category) => (
                <option key={category} value={category}>{categoryLabel(category, t)}</option>
              ))}
            </select>
            <select
              className="h-8 rounded-md border border-border bg-background px-2 text-xs font-medium text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onChange={(event) => setVisibilityFilter(event.target.value as VisibilityFilter)}
              value={visibilityFilter}
            >
              <option value="all">{t.promptLibrary.allVisibility}</option>
              <option value="chat">{t.promptLibrary.chatVisible}</option>
              <option value="editor">{t.promptLibrary.editorVisible}</option>
              <option value="hidden">{t.promptLibrary.hiddenEverywhere}</option>
            </select>
            <select
              className="h-8 rounded-md border border-border bg-background px-2 text-xs font-medium text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onChange={(event) => setAutocompleteFilter(event.target.value as AutocompleteFilter)}
              value={autocompleteFilter}
            >
              <option value="all">{t.promptLibrary.allAutocomplete}</option>
              <option value="visible">{t.promptLibrary.autocompleteVisible}</option>
              <option value="hidden">{t.promptLibrary.autocompleteHidden}</option>
            </select>
          </div>
        </div>
        <ScrollArea className="min-h-0 flex-1">
          <div className="space-y-3 p-2">
            {groupedRules.length > 0 ? groupedRules.map((group) => (
              <div key={group.category}>
                <div className="mb-1.5 flex items-center gap-2 px-2 text-[10px] font-semibold uppercase text-muted-foreground">
                  <span className={cn('h-1.5 w-1.5 rounded-full', categoryTone(group.category).dot)} />
                  {categoryLabel(group.category, t)}
                </div>
                <div className="space-y-1">
                  {group.rules.map((rule) => (
                    <PromptListItem
                      isSelected={draft.selectedRuleId === rule.id}
                      key={rule.id}
                      onSelect={() => loadRule(rule)}
                      rule={rule}
                    />
                  ))}
                </div>
              </div>
            )) : (
              <div className="rounded-md border border-dashed border-border p-4 text-center text-xs text-muted-foreground">
                {t.promptLibrary.noPrompts}
              </div>
            )}
          </div>
        </ScrollArea>
      </aside>

      <ScrollArea className="min-h-0">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 p-4 md:p-6">
          <section className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-3">
              <div className="min-w-0">
                <h2 className="truncate text-sm font-semibold text-foreground">
                  {selectedRule ? selectedRule.title : t.promptLibrary.newPrompt}
                </h2>
                {draft.isDirty ? (
                  <p className="text-xs font-medium text-warning">{t.promptLibrary.unsaved}</p>
                ) : null}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Button
                  className="gap-1.5 text-destructive hover:text-destructive"
                  disabled={!selectedRule}
                  onClick={deletePrompt}
                  type="button"
                  variant="ghost"
                >
                  <Trash2 className="size-4" />
                  {t.promptLibrary.deletePrompt}
                </Button>
                <Button className="gap-1.5" onClick={savePrompt} type="button">
                  <Save className="size-4" />
                  {t.promptLibrary.savePrompt}
                </Button>
              </div>
            </div>

            <PromptUsageHint draft={draft} />

            <div className="grid gap-x-5 gap-y-4 lg:grid-cols-[minmax(0,1fr)_minmax(292px,340px)]">
              <div className="grid gap-3 sm:grid-cols-2 lg:col-start-1 lg:row-start-1">
                <label className="block space-y-1.5">
                  <span className="text-xs font-semibold text-muted-foreground">{t.promptLibrary.labelLabel}</span>
                  <div className="flex items-center rounded-md border border-border bg-background px-2 focus-within:ring-2 focus-within:ring-ring">
                    <span className="text-sm font-semibold text-muted-foreground">@rules:</span>
                    <input
                      className="min-w-0 flex-1 border-0 bg-transparent px-1 py-2 text-sm font-semibold text-foreground outline-none"
                      maxLength={48}
                      onChange={(event) => updateDraft({ label: normalizeRuleLabel(event.target.value) })}
                      placeholder={t.promptLibrary.labelPlaceholder}
                      value={draft.label}
                    />
                  </div>
                </label>
                <label className="block space-y-1.5">
                  <span className="text-xs font-semibold text-muted-foreground">{t.promptLibrary.titleLabel}</span>
                  <input
                    className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    onChange={(event) => updateDraft({ title: event.target.value })}
                    placeholder={t.promptLibrary.titlePlaceholder}
                    value={draft.title}
                  />
                </label>
              </div>

              <div className="space-y-1.5 lg:col-start-1 lg:row-start-2">
                <span className="text-xs font-semibold text-muted-foreground">{t.promptLibrary.categoryLabel}</span>
                <div className="grid gap-2 sm:grid-cols-3">
                  {chatRuleCategories.map((category) => {
                    const Icon = categoryIcon(category)
                    const active = draft.category === category
                    const tone = categoryTone(category)
                    return (
                      <button
                        aria-pressed={active}
                        className={cn(
                          'flex min-w-0 items-start gap-2 rounded-md border border-border bg-background px-3 py-2 text-left transition-colors hover:bg-accent',
                          active && tone.active,
                        )}
                        key={category}
                        onClick={() => updateDraft({ category })}
                        type="button"
                      >
                        <Icon className={cn('mt-0.5 size-4 shrink-0', active ? tone.icon : 'text-muted-foreground')} />
                        <span className="min-w-0">
                          <span className="block truncate text-sm font-semibold">{categoryLabel(category, t)}</span>
                          <span className="block truncate text-[11px] font-medium text-muted-foreground">
                            {categoryHint(category, t)}
                          </span>
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>

              <div className="lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:flex lg:items-end">
                <VisibilityPanel
                  draft={draft}
                  onAutocompleteChange={setAutocomplete}
                  onVisibilityChange={setVisibility}
                />
              </div>

              <div className={cn(
                'block space-y-1.5 lg:col-start-1 lg:row-start-3',
                draft.category !== 'context' && 'lg:col-span-2',
              )}>
                <div className="flex items-center justify-between gap-2">
                  <label className="text-xs font-semibold text-muted-foreground" htmlFor="prompt-library-prompt">
                    {t.promptLibrary.promptLabel}
                  </label>
                </div>
                <div className="relative">
                  <div className="absolute right-5 top-3 z-10 rounded-md bg-background/90 shadow-sm">
                    <TextImproveButton
                      disabled={!draft.contentMarkdown.trim() || !textImprovement.enabled}
                      isLoading={promptTextImprove.isImproving}
                      label={t.textImprove.improve}
                      loadingLabel={t.textImprove.improving}
                      onClick={() => void improvePrompt()}
                      reduceMotion={reduceMotion}
                    />
                  </div>
                  <Textarea
                    className="min-h-[23rem] pr-16 text-sm leading-6 [scrollbar-width:thin]"
                    id="prompt-library-prompt"
                    onChange={(event) => updateDraft({ contentMarkdown: event.target.value })}
                    placeholder={draft.category === 'context'
                      ? t.promptLibrary.contextPromptPlaceholder.replace('{placeholder}', contextPackPlaceholder)
                      : t.promptLibrary.promptPlaceholder}
                    value={draft.contentMarkdown}
                  />
                  <TextImproveFieldLayer
                    labels={{
                      accept: t.textImprove.accept,
                      changes: t.textImprove.changes,
                      noChanges: t.textImprove.noChanges,
                      reject: t.textImprove.reject,
                      title: t.textImprove.title,
                      warnings: t.textImprove.warnings,
                    }}
                    onAccept={acceptPromptImprovement}
                    onReject={promptTextImprove.clearProposal}
                    proposal={promptTextImprove.proposal}
                    reduceMotion={reduceMotion}
                  />
                </div>
                {promptImproveError ? (
                  <p className="text-xs font-medium text-warning">{promptImproveError}</p>
                ) : null}
              </div>

              {draft.category === 'context' ? (
                <div className="pt-[1.375rem] lg:col-start-2 lg:row-start-3">
                  <ContextPackPanel
                    contextChips={contextChips}
                    contextOptions={contextOptions}
                    contextOptionByKey={contextOptionByKey}
                    contextQuery={contextQuery}
                    draft={draft}
                    onOpenPreview={() => setPreviewOpen(true)}
                    onQueryChange={setContextQuery}
                    onRemove={removeContextRef}
                    onToggle={toggleContextRef}
                  />
                </div>
              ) : null}
            </div>
            {draft.error ? (
              <p className="pt-3 text-xs font-medium text-warning">{draft.error}</p>
            ) : null}
          </section>
        </div>
      </ScrollArea>
      {isPreviewOpen ? (
        <RenderedPreviewModal onClose={() => setPreviewOpen(false)} preview={preview} title={t.promptLibrary.previewLabel} />
      ) : null}
    </div>
  )
}

function PromptUsageHint({ draft }: { draft: PromptDraft }) {
  const { t } = useLocale()
  const tone = categoryTone(draft.category)
  const Icon = categoryIcon(draft.category)
  const shortcutLabel = normalizeRuleLabel(draft.label) || t.promptLibrary.labelPlaceholder
  const shortcut = `@rules:${shortcutLabel}`
  const usage = categoryUsage(draft.category, t)
    .replace('{mention}', shortcut)
    .replace('{placeholder}', contextPackPlaceholder)

  return (
    <section className={cn('flex items-center gap-3 rounded-md px-3 py-2.5', tone.callout)}>
      <span className={cn('grid size-7 shrink-0 place-items-center rounded-md', tone.calloutIcon)}>
        <Icon className="size-4" />
      </span>
      <div className="min-w-0 text-sm leading-6 text-foreground">
        <p>
          <span className="font-semibold">{categoryLabel(draft.category, t)}: </span>
          {usage}
        </p>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {t.promptLibrary.shortcutHint.replace('{mention}', shortcut)}
        </p>
      </div>
    </section>
  )
}

function VisibilityPanel({
  draft,
  onAutocompleteChange,
  onVisibilityChange,
}: {
  draft: PromptDraft
  onAutocompleteChange: (includeInAutocomplete: boolean) => void
  onVisibilityChange: (key: keyof ChatRuleVisibility, value: boolean) => void
}) {
  const { t } = useLocale()

  return (
    <section className={cn(
      'w-full rounded-md border border-border bg-background p-3 transition-colors',
      !draft.includeInAutocomplete && 'bg-muted/30',
    )}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-xs font-semibold text-muted-foreground">{t.promptLibrary.visibilityLabel}</h3>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            {draft.includeInAutocomplete
              ? t.promptLibrary.autocompleteEnabledHint
              : t.promptLibrary.autocompleteDisabledHint}
          </p>
        </div>
        <Switch
          aria-label={t.promptLibrary.autocompleteLabel}
          checked={draft.includeInAutocomplete}
          className="data-[state=checked]:bg-brand data-[state=unchecked]:bg-muted"
          onCheckedChange={onAutocompleteChange}
        />
      </div>
      <div className={cn('mt-3 grid grid-cols-2 gap-2', !draft.includeInAutocomplete && 'opacity-45')}>
        <VisibilityToggle
          active={draft.visibility.chat}
          disabled={!draft.includeInAutocomplete}
          label={t.promptLibrary.chatVisible}
          onClick={() => onVisibilityChange('chat', !draft.visibility.chat)}
        />
        <VisibilityToggle
          active={draft.visibility.editor}
          disabled={!draft.includeInAutocomplete}
          label={t.promptLibrary.editorVisible}
          onClick={() => onVisibilityChange('editor', !draft.visibility.editor)}
        />
      </div>
    </section>
  )
}

function VisibilityToggle({
  active,
  disabled,
  label,
  onClick,
}: {
  active: boolean
  disabled: boolean
  label: string
  onClick: () => void
}) {
  return (
    <button
      aria-pressed={active}
      className={cn(
        'flex h-9 items-center justify-center gap-2 rounded-md border border-border px-2 text-sm font-semibold text-foreground transition-colors',
        active && 'border-brand bg-brand text-brand-foreground shadow-[0_1px_2px_var(--brand-shadow)] hover:bg-brand',
        disabled && 'cursor-not-allowed',
      )}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      <span className={cn(
        'grid size-4 place-items-center rounded border border-current',
        active && 'bg-brand-foreground text-brand',
      )}>
        {active ? <Check className="size-3" /> : null}
      </span>
      {label}
    </button>
  )
}

function ContextPackPanel({
  contextChips,
  contextOptionByKey,
  contextOptions,
  contextQuery,
  draft,
  onOpenPreview,
  onQueryChange,
  onRemove,
  onToggle,
}: {
  contextChips: ReturnType<typeof chatAttachmentChipsFromRefs>
  contextOptionByKey: Map<string, ContextPickerOption>
  contextOptions: ContextPickerOption[]
  contextQuery: string
  draft: PromptDraft
  onOpenPreview: () => void
  onQueryChange: (value: string) => void
  onRemove: (ref: ChatContextReferenceRecord) => void
  onToggle: (ref: ChatContextReferenceRecord) => void
}) {
  const { t } = useLocale()
  const selectedKeys = new Set(draft.linkedContextRefs.map(chatContextRefKey))
  const selectedChips = contextChips.map((chip) => ({
    ...chip,
    option: contextOptionByKey.get(chatContextRefKey(chip.ref)),
  }))
  const trimmedQuery = contextQuery.trim().toLowerCase()
  const matchingOptions = contextOptions
    .filter((option) => !selectedKeys.has(chatContextRefKey(option.ref)))
    .filter((option) => {
      if (!trimmedQuery) return true
      return `${option.label} ${option.title}`.toLowerCase().includes(trimmedQuery)
    })
  const visibleOptions = matchingOptions.slice(0, contextPickerLimit)

  return (
    <section className="rounded-md border border-border bg-background p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold text-muted-foreground">{t.promptLibrary.contextLabel}</h3>
        <Badge className="h-5 rounded-full px-1.5 text-[10px]" variant="outline">
          {draft.linkedContextRefs.length}
        </Badge>
      </div>
      <p className="mb-3 text-xs leading-5 text-muted-foreground">
        {t.promptLibrary.contextPickerHint}
      </p>

      <div className="space-y-2">
        {selectedChips.length > 0 ? (
          <div className="space-y-1">
            {selectedChips.map((chip) => {
              const Icon = chip.option?.icon ?? Paperclip
              return (
                <div
                  className="flex min-w-0 items-center gap-2 rounded-md border border-border bg-muted/35 px-2 py-1.5"
                  key={chatContextRefKey(chip.ref)}
                >
                  <Icon className="size-4 shrink-0 text-muted-foreground" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-xs font-semibold text-foreground">{chip.label}</span>
                    <span className="block truncate text-[11px] text-muted-foreground">{chip.title}</span>
                  </span>
                  <button
                    aria-label={t.promptLibrary.removeContext}
                    className="grid size-6 shrink-0 place-items-center rounded-md text-muted-foreground hover:bg-background hover:text-foreground"
                    onClick={() => onRemove(chip.ref)}
                    type="button"
                  >
                    <X className="size-3.5" />
                  </button>
                </div>
              )
            })}
          </div>
        ) : (
          <p className="rounded-md border border-dashed border-border p-3 text-xs text-muted-foreground">
            {t.promptLibrary.noContextSelected}
          </p>
        )}

        <label className="flex items-center gap-2 rounded-md border border-border bg-background px-2 focus-within:ring-2 focus-within:ring-ring">
          <Search className="size-4 shrink-0 text-muted-foreground" />
          <input
            className="min-w-0 flex-1 border-0 bg-transparent py-2 text-sm text-foreground outline-none"
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder={t.promptLibrary.contextSearchPlaceholder}
            value={contextQuery}
          />
        </label>

        <div className="space-y-1">
          {visibleOptions.map((option) => (
            <ContextOptionButton
              isSelected={selectedKeys.has(chatContextRefKey(option.ref))}
              key={option.key}
              option={option}
              onClick={() => onToggle(option.ref)}
            />
          ))}
          {contextOptions.length === 0 ? (
            <p className="rounded-md border border-dashed border-border p-3 text-xs text-muted-foreground">
              {t.promptLibrary.noContextFiles}
            </p>
          ) : null}
          {contextOptions.length > 0 && visibleOptions.length === 0 ? (
            <p className="rounded-md border border-dashed border-border p-3 text-xs text-muted-foreground">
              {t.promptLibrary.noContextMatches}
            </p>
          ) : null}
          {matchingOptions.length > visibleOptions.length ? (
            <p className="px-1 text-[11px] text-muted-foreground">
              {t.promptLibrary.contextResultsLimited.replace('{count}', String(visibleOptions.length))}
            </p>
          ) : null}
        </div>

        <Button className="w-full gap-1.5" onClick={onOpenPreview} type="button" variant="outline">
          <BookOpen className="size-4" />
          {t.promptLibrary.openPreview}
        </Button>
      </div>
    </section>
  )
}

function ContextOptionButton({
  isSelected,
  onClick,
  option,
}: {
  isSelected: boolean
  onClick: () => void
  option: ContextPickerOption
}) {
  const Icon = option.icon
  return (
    <button
      aria-pressed={isSelected}
      className={cn(
        'flex w-full min-w-0 items-center gap-2 rounded-md border border-transparent px-2 py-1.5 text-left transition-colors hover:bg-accent',
        isSelected && 'border-brand/25 bg-brand-subtle text-brand',
      )}
      onClick={onClick}
      type="button"
    >
      <Icon className="size-4 shrink-0 text-muted-foreground" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs font-semibold">{option.label}</span>
        <span className="block truncate text-[11px] text-muted-foreground">{option.title}</span>
      </span>
      {isSelected ? <Check className="size-4 shrink-0" /> : null}
    </button>
  )
}

function PromptListItem({
  isSelected,
  onSelect,
  rule,
}: {
  isSelected: boolean
  onSelect: () => void
  rule: ChatRuleRecord
}) {
  const { t } = useLocale()
  const normalized = normalizeChatRule(rule)
  const category = normalized.category ?? 'instruction'
  const tone = categoryTone(category)
  const visibility = normalized.visibility ?? { chat: true, editor: true }
  const isAutocompleteVisible = normalized.includeInAutocomplete !== false
  const locations = isAutocompleteVisible
    ? [
      visibility.chat ? t.promptLibrary.chatVisible : null,
      visibility.editor ? t.promptLibrary.editorVisible : null,
    ].filter(Boolean)
    : []
  return (
    <button
      className={cn(
        'group relative w-full min-w-0 rounded-md border border-transparent bg-transparent py-2 pl-3.5 pr-2 text-left transition-colors hover:bg-background',
        !isAutocompleteVisible && 'bg-muted/35 opacity-75 hover:bg-muted/45',
        isSelected && 'border-border bg-background shadow-[0_1px_2px_var(--shadow-hairline)]',
        isSelected && !isAutocompleteVisible && 'bg-muted/45',
      )}
      onClick={onSelect}
      type="button"
    >
      <span className={cn('absolute bottom-2 left-1 top-2 w-1 rounded-full', tone.dot)} />
      <span className="flex min-w-0 items-center justify-between gap-2">
        <span className={cn(
          'min-w-0 truncate text-xs font-semibold text-muted-foreground',
          !isAutocompleteVisible && 'text-muted-foreground/75',
        )}>
          @rules:{normalized.label}
        </span>
        <Badge className={cn(
          'h-5 shrink-0 rounded-full px-1.5 text-[10px]',
          isAutocompleteVisible ? tone.badge : 'border-muted-foreground/20 bg-muted text-muted-foreground',
        )} variant="outline">
          {categoryLabel(category, t)}
        </Badge>
      </span>
      <span className={cn(
        'mt-0.5 block truncate text-sm font-semibold text-foreground',
        !isAutocompleteVisible && 'text-foreground/70',
      )}>
        {normalized.title}
      </span>
      <span className={cn(
        'mt-1 flex min-w-0 items-center gap-1.5 text-[11px] text-muted-foreground',
        !isAutocompleteVisible && 'text-destructive',
      )}>
        {locations.length > 0 ? (
          <span className="truncate">{locations.join(' · ')}</span>
        ) : (
          <span className="inline-flex min-w-0 items-center gap-1 rounded-full bg-destructive/10 px-1.5 py-0.5 font-semibold">
            <EyeOff className="size-3 shrink-0" />
            {t.promptLibrary.autocompleteHidden}
          </span>
        )}
      </span>
    </button>
  )
}

function RenderedPreviewModal({
  onClose,
  preview,
  title,
}: {
  onClose: () => void
  preview: string
  title: string
}) {
  const { t } = useLocale()
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-background/75 px-4 py-8 backdrop-blur"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <section
        aria-modal="true"
        className="w-full max-w-4xl overflow-hidden rounded-lg border border-border bg-background shadow-xl"
        role="dialog"
      >
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold text-foreground">{title}</h2>
            <p className="text-xs text-muted-foreground">{t.promptLibrary.previewHint}</p>
          </div>
          <button
            aria-label={t.common.close}
            className="grid size-8 shrink-0 place-items-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
            onClick={onClose}
            type="button"
          >
            <X className="size-4" />
          </button>
        </div>
        <div className="max-h-[72svh] overflow-auto p-4">
          <pre className="whitespace-pre-wrap rounded-md bg-muted/45 p-4 text-xs leading-5 text-foreground">
            {preview}
          </pre>
        </div>
      </section>
    </div>
  )
}

function draftFromRule(rule: ChatRuleRecord | null): PromptDraft {
  if (!rule) return emptyDraft
  const normalized = normalizeChatRule(rule)
  return {
    category: normalized.category ?? 'instruction',
    contentMarkdown: normalized.contentMarkdown,
    error: null,
    includeInAutocomplete: normalized.includeInAutocomplete ?? true,
    isDirty: false,
    label: normalized.label,
    linkedContextRefs: normalized.linkedContextRefs ?? [],
    selectedRuleId: normalized.id,
    title: normalized.title,
    visibility: normalized.visibility ?? { chat: true, editor: true },
  }
}

function draftRuleRecord(draft: PromptDraft, selectedRule: ChatRuleRecord | null): ChatRuleRecord {
  const now = new Date().toISOString()
  return normalizeChatRule({
    category: draft.category,
    contentMarkdown: draft.contentMarkdown,
    createdAt: selectedRule?.createdAt ?? now,
    id: draft.selectedRuleId ?? 'prompt-preview',
    includeInAutocomplete: draft.includeInAutocomplete,
    label: draft.label || 'preview',
    linkedContextRefs: draft.category === 'context' ? draft.linkedContextRefs : [],
    title: draft.title || draft.label || 'Preview',
    updatedAt: selectedRule?.updatedAt ?? now,
    visibility: draft.visibility,
  })
}

function categoryIcon(category: ChatRuleCategory) {
  if (category === 'function') return ListOrdered
  if (category === 'context') return BookOpen
  return Bot
}

function categoryLabel(category: ChatRuleCategory, t: ReturnType<typeof useLocale>['t']) {
  if (category === 'function') return t.promptLibrary.categoryFunction
  if (category === 'context') return t.promptLibrary.categoryContext
  return t.promptLibrary.categoryInstruction
}

function categoryHint(category: ChatRuleCategory, t: ReturnType<typeof useLocale>['t']) {
  if (category === 'function') return t.promptLibrary.categoryFunctionHint
  if (category === 'context') return t.promptLibrary.categoryContextHint
  return t.promptLibrary.categoryInstructionHint
}

function categoryUsage(category: ChatRuleCategory, t: ReturnType<typeof useLocale>['t']) {
  if (category === 'function') return t.promptLibrary.categoryFunctionUsage
  if (category === 'context') return t.promptLibrary.categoryContextUsage
  return t.promptLibrary.categoryInstructionUsage
}

function promptImprovementGuidance(draft: PromptDraft, t: ReturnType<typeof useLocale>['t']) {
  const label = normalizeRuleLabel(draft.label) || 'draft-label'
  const category = promptImprovementCategory(draft.category)
  const usage = categoryUsage(draft.category, t)
    .replace('{mention}', `@rules:${label}`)
    .replace('{placeholder}', contextPackPlaceholder)

  return [
    `Prompt Library category: ${category.name}.`,
    `Shortcut label: @rules:${label}.`,
    category.guidance,
    `Current UI usage guidance: ${usage}`,
    'Do not include this guidance in the improved prompt. Return only the improved prompt content.',
  ].join('\n')
}

function promptImprovementCategory(category: ChatRuleCategory) {
  if (category === 'function') {
    return {
      guidance: 'This prompt is a Function: it performs a reusable transformation such as translation, summarization, extraction, or rewriting. Optimize for clear inputs, task boundaries, chain-step reuse, and predictable output behavior.',
      name: 'Function',
    }
  }
  if (category === 'context') {
    return {
      guidance: `This prompt is a Context Pack: it provides long reusable context and may include rendered database file blocks. Preserve ${contextPackPlaceholder} exactly if it is present. If a file insertion point would improve the prompt, use ${contextPackPlaceholder} exactly once.`,
      name: 'Context Pack',
    }
  }
  return {
    guidance: 'This prompt is an Instruction: it changes the model role, behavior, style, working method, or skill posture. Optimize it as reusable model guidance without turning it into a one-off task.',
    name: 'Instruction',
  }
}

function messageFromUnknown(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

function categoryTone(category: ChatRuleCategory) {
  if (category === 'function') {
    return {
      active: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-800 dark:text-emerald-200',
      badge: 'border-emerald-500/25 bg-emerald-500/10 text-emerald-800 dark:text-emerald-200',
      callout: 'bg-emerald-500/10',
      calloutIcon: 'bg-emerald-500/15 text-emerald-800 dark:text-emerald-200',
      dot: 'bg-emerald-500',
      icon: 'text-emerald-700 dark:text-emerald-300',
    }
  }
  if (category === 'context') {
    return {
      active: 'border-amber-500/30 bg-amber-500/10 text-amber-900 dark:text-amber-200',
      badge: 'border-amber-500/25 bg-amber-500/10 text-amber-900 dark:text-amber-200',
      callout: 'bg-amber-500/10',
      calloutIcon: 'bg-amber-500/15 text-amber-900 dark:text-amber-200',
      dot: 'bg-amber-500',
      icon: 'text-amber-700 dark:text-amber-300',
    }
  }
  return {
    active: 'border-sky-500/30 bg-sky-500/10 text-sky-900 dark:text-sky-200',
    badge: 'border-sky-500/25 bg-sky-500/10 text-sky-900 dark:text-sky-200',
    callout: 'bg-sky-500/10',
    calloutIcon: 'bg-sky-500/15 text-sky-900 dark:text-sky-200',
    dot: 'bg-sky-500',
    icon: 'text-sky-700 dark:text-sky-300',
  }
}
