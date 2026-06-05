import { useMemo, useState, type Dispatch } from 'react'
import { useReducedMotion } from 'motion/react'
import {
  AlertTriangle,
  BookOpen,
  Bot,
  Check,
  EyeOff,
  FileText,
  FolderOpen,
  Library,
  ListOrdered,
  MessagesSquare,
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
import {
  toneAccentBorderLeft,
  toneActiveCard,
  toneBadge,
  toneBar,
  toneIconTile,
  toneText,
  type MentionTone,
} from '@/lib/tone'
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
  const [pendingNav, setPendingNav] = useState<(() => void) | null>(null)
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

  function guardedLoad(rule: ChatRuleRecord | null) {
    if (draft.isDirty) {
      setPendingNav(() => () => loadRule(rule))
      return
    }
    loadRule(rule)
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
    const visibility = { ...draft.visibility, [key]: value }
    updateDraft({
      includeInAutocomplete: visibility.chat || visibility.editor,
      visibility,
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

  function savePrompt(): boolean {
    const label = normalizeRuleLabel(draft.label)
    const title = draft.title.trim() || label
    const contentMarkdown = draft.contentMarkdown.trim()
    if (!label) {
      updateDraft({ error: t.promptLibrary.labelRequired, isDirty: draft.isDirty })
      return false
    }
    if (rules.some((rule) => rule.label === label && rule.id !== draft.selectedRuleId)) {
      updateDraft({ error: t.promptLibrary.labelDuplicate, isDirty: draft.isDirty })
      return false
    }
    if (!contentMarkdown) {
      updateDraft({ error: t.promptLibrary.promptRequired, isDirty: draft.isDirty })
      return false
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
    return true
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
            <Button aria-label={t.promptLibrary.newPrompt} className="size-8" onClick={() => guardedLoad(null)} size="icon" type="button" variant="outline">
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
          <div className="mt-3 space-y-2">
            <div className="flex items-center gap-0.5 rounded-md border border-border bg-background p-0.5">
              <button
                aria-pressed={categoryFilter === 'all'}
                className={cn(
                  'h-7 flex-1 rounded-[5px] px-2 text-xs font-semibold transition-colors',
                  categoryFilter === 'all' ? 'bg-accent text-foreground' : 'text-muted-foreground hover:text-foreground',
                )}
                onClick={() => setCategoryFilter('all')}
                type="button"
              >
                {t.promptLibrary.allFilter}
              </button>
              {chatRuleCategories.map((category) => {
                const Icon = categoryIcon(category)
                const active = categoryFilter === category
                return (
                  <button
                    aria-label={categoryLabel(category, t)}
                    aria-pressed={active}
                    className={cn(
                      'grid h-7 flex-1 place-items-center rounded-[5px] transition-colors',
                      active ? 'bg-accent' : 'hover:bg-accent/60',
                    )}
                    key={category}
                    onClick={() => setCategoryFilter(category)}
                    title={categoryLabel(category, t)}
                    type="button"
                  >
                    <Icon className={cn('size-3.5', active ? toneText[categoryToTone[category]] : 'text-muted-foreground')} />
                  </button>
                )
              })}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <select
                aria-label={t.promptLibrary.allVisibility}
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
                aria-label={t.promptLibrary.allAutocomplete}
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
        </div>
        <ScrollArea className="min-h-0 flex-1">
          <div className="space-y-3 p-2">
            {groupedRules.length > 0 ? groupedRules.map((group) => (
              <div key={group.category}>
                <div className="mb-1.5 flex items-center gap-2 px-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  <span className={cn('size-1.5 shrink-0 rounded-full', toneBar[categoryToTone[group.category]])} />
                  <span>{categoryLabel(group.category, t)}</span>
                  <span className="ml-auto shrink-0 tabular-nums text-muted-foreground/60">{group.rules.length}</span>
                </div>
                <div className="space-y-1">
                  {group.rules.map((rule) => (
                    <PromptListItem
                      isSelected={draft.selectedRuleId === rule.id}
                      key={rule.id}
                      onSelect={() => guardedLoad(rule)}
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

            <div className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-2">
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

              <div className="space-y-1.5">
                <span className="text-xs font-semibold text-muted-foreground">{t.promptLibrary.categoryLabel}</span>
                <div className="grid gap-2 sm:grid-cols-3">
                  {chatRuleCategories.map((category) => {
                    const Icon = categoryIcon(category)
                    const active = draft.category === category
                    const tone = categoryToTone[category]
                    return (
                      <button
                        aria-pressed={active}
                        className={cn(
                          'flex min-w-0 items-start gap-2 rounded-md border px-3 py-2 text-left transition-colors',
                          active ? toneActiveCard[tone] : 'border-border bg-background hover:bg-accent',
                        )}
                        key={category}
                        onClick={() => updateDraft({ category })}
                        type="button"
                      >
                        <Icon className={cn('mt-0.5 size-4 shrink-0', active ? toneText[tone] : 'text-muted-foreground')} />
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

              <VisibilityPanel draft={draft} onVisibilityChange={setVisibility} />

              <div className="space-y-1.5">
                <div className={cn(
                  'grid gap-4',
                  draft.category === 'context' && 'lg:grid-cols-[minmax(0,1fr)_320px]',
                )}>
                  <div className="overflow-hidden rounded-md border border-border bg-background shadow-[0_1px_2px_var(--shadow-hairline)] focus-within:border-brand/40 focus-within:ring-2 focus-within:ring-brand/15">
                    <div className="flex items-center gap-2 border-b border-border bg-surface/50 px-3 py-1.5">
                      <span className="text-xs font-semibold text-muted-foreground">{t.promptLibrary.promptLabel}</span>
                      {draft.category === 'context' ? (
                        <span className="inline-flex items-center gap-1 rounded border border-warning/25 bg-warning-subtle/60 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-warning">
                          {contextPackPlaceholder}
                        </span>
                      ) : null}
                      <span className="ml-auto shrink-0 tabular-nums text-[11px] text-muted-foreground/70">
                        {t.promptLibrary.charCount.replace('{count}', draft.contentMarkdown.length.toLocaleString(locale))}
                      </span>
                      <TextImproveButton
                        disabled={!draft.contentMarkdown.trim() || !textImprovement.enabled}
                        isLoading={promptTextImprove.isImproving}
                        label={t.textImprove.improve}
                        loadingLabel={t.textImprove.improving}
                        onClick={() => void improvePrompt()}
                        reduceMotion={reduceMotion}
                      />
                    </div>
                    <div className="relative min-w-0">
                      <Textarea
                        aria-label={t.promptLibrary.promptLabel}
                        className="min-h-[23rem] resize-y rounded-none border-0 bg-transparent text-sm leading-6 shadow-none focus-visible:ring-0 [scrollbar-width:thin]"
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
                  </div>

                  {draft.category === 'context' ? (
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
                  ) : null}
                </div>
                {promptImproveError ? (
                  <p className="text-xs font-medium text-warning">{promptImproveError}</p>
                ) : null}
              </div>
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
      {pendingNav ? (
        <UnsavedGuardModal
          onCancel={() => setPendingNav(null)}
          onDiscard={() => {
            const run = pendingNav
            setPendingNav(null)
            run?.()
          }}
          onSave={() => {
            const ok = savePrompt()
            const run = pendingNav
            setPendingNav(null)
            if (ok) run?.()
          }}
        />
      ) : null}
    </div>
  )
}

function PromptUsageHint({ draft }: { draft: PromptDraft }) {
  const { t } = useLocale()
  const tone = categoryToTone[draft.category]
  const Icon = categoryIcon(draft.category)
  const shortcutLabel = normalizeRuleLabel(draft.label) || t.promptLibrary.labelPlaceholder
  const shortcut = `@rules:${shortcutLabel}`
  const usage = categoryUsage(draft.category, t)
    .replace('{mention}', shortcut)
    .replace('{placeholder}', contextPackPlaceholder)

  return (
    <section className="flex items-start gap-2.5 rounded-md border border-border bg-surface/50 px-3 py-2.5">
      <span className={cn('mt-0.5 grid size-7 shrink-0 place-items-center rounded-md', toneIconTile[tone])}>
        <Icon className="size-4" />
      </span>
      <div className="min-w-0 text-xs leading-5 text-muted-foreground">
        <p>
          <span className="font-semibold text-foreground">{categoryLabel(draft.category, t)}: </span>
          {usage}
        </p>
        <p className="mt-0.5">
          {t.promptLibrary.shortcutHint.replace('{mention}', shortcut)}
        </p>
      </div>
    </section>
  )
}

function VisibilityPanel({
  draft,
  onVisibilityChange,
}: {
  draft: PromptDraft
  onVisibilityChange: (key: keyof ChatRuleVisibility, value: boolean) => void
}) {
  const { t } = useLocale()
  const hidden = !draft.visibility.chat && !draft.visibility.editor

  return (
    <section className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 rounded-md border border-border bg-surface/45 px-3 py-2">
      <div className="min-w-0">
        <span className="text-xs font-semibold text-foreground">{t.promptLibrary.visibilityLabel}</span>
        {hidden ? (
          <span className="ml-2 inline-flex items-center gap-1 align-middle text-[11px] font-medium text-destructive/80">
            <EyeOff className="size-3" />
            {t.promptLibrary.hiddenEverywhere}
          </span>
        ) : null}
      </div>
      <div className="flex shrink-0 items-center gap-1.5">
        <VisChip
          active={draft.visibility.chat}
          icon={MessagesSquare}
          label={t.promptLibrary.chatVisible}
          onClick={() => onVisibilityChange('chat', !draft.visibility.chat)}
        />
        <VisChip
          active={draft.visibility.editor}
          icon={FileText}
          label={t.promptLibrary.editorVisible}
          onClick={() => onVisibilityChange('editor', !draft.visibility.editor)}
        />
      </div>
    </section>
  )
}

function VisChip({
  active,
  icon: Icon,
  label,
  onClick,
}: {
  active: boolean
  icon: LucideIcon
  label: string
  onClick: () => void
}) {
  return (
    <button
      aria-pressed={active}
      className={cn(
        'inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs font-medium transition-colors',
        active
          ? 'border-brand/40 bg-brand-subtle text-brand'
          : 'border-border text-muted-foreground hover:bg-accent hover:text-foreground',
      )}
      onClick={onClick}
      type="button"
    >
      <Icon className="size-3.5" />
      {label}
      {active ? <Check className="size-3" /> : null}
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
    <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-md border border-border bg-surface/35">
      <div className="border-b border-border/60 px-3 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-xs font-semibold text-foreground">{t.promptLibrary.contextLabel}</h3>
          <Badge className="h-5 rounded-full border-border/70 bg-background px-1.5 text-[10px]" variant="outline">
            {draft.linkedContextRefs.length}
          </Badge>
        </div>
        <p className="mt-1 text-[11px] leading-4 text-muted-foreground">
          {t.promptLibrary.contextPickerHint}
        </p>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-2 p-3">
        {selectedChips.length > 0 ? (
          <div className="space-y-1">
            {selectedChips.map((chip) => {
              const Icon = chip.option?.icon ?? Paperclip
              return (
                <div
                  className="flex min-w-0 items-center gap-2 rounded-md border border-border/70 bg-background/70 px-2 py-1.5"
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
          <p className="rounded-md bg-background/55 px-2.5 py-2 text-xs text-muted-foreground">
            {t.promptLibrary.noContextSelected}
          </p>
        )}

        <label className="flex h-8 items-center gap-2 rounded-md border border-border bg-background px-2 focus-within:ring-2 focus-within:ring-ring">
          <Search className="size-4 shrink-0 text-muted-foreground" />
          <input
            className="min-w-0 flex-1 border-0 bg-transparent text-sm text-foreground outline-none"
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder={t.promptLibrary.contextSearchPlaceholder}
            value={contextQuery}
          />
        </label>

        <div className="min-h-0 space-y-1">
          {visibleOptions.map((option) => (
            <ContextOptionButton
              isSelected={selectedKeys.has(chatContextRefKey(option.ref))}
              key={option.key}
              option={option}
              onClick={() => onToggle(option.ref)}
            />
          ))}
          {contextOptions.length === 0 ? (
            <p className="rounded-md bg-background/45 px-2.5 py-2 text-xs text-muted-foreground">
              {t.promptLibrary.noContextFiles}
            </p>
          ) : null}
          {contextOptions.length > 0 && visibleOptions.length === 0 ? (
            <p className="rounded-md bg-background/45 px-2.5 py-2 text-xs text-muted-foreground">
              {t.promptLibrary.noContextMatches}
            </p>
          ) : null}
          {matchingOptions.length > visibleOptions.length ? (
            <p className="px-1 text-[11px] text-muted-foreground">
              {t.promptLibrary.contextResultsLimited.replace('{count}', String(visibleOptions.length))}
            </p>
          ) : null}
        </div>

        <Button className="mt-auto w-full gap-1.5" onClick={onOpenPreview} size="sm" type="button" variant="outline">
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
        'flex w-full min-w-0 items-center gap-2 rounded-md border border-transparent px-2 py-1.5 text-left transition-colors hover:bg-background/75',
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
  const tone = categoryToTone[category]
  const visibility = normalized.visibility ?? { chat: true, editor: true }
  const isAutocompleteVisible = normalized.includeInAutocomplete !== false
  const locations = isAutocompleteVisible
    ? [
      visibility.chat ? t.promptLibrary.chatVisible : null,
      visibility.editor ? t.promptLibrary.editorVisible : null,
    ].filter(Boolean)
    : []
  const hidden = locations.length === 0
  return (
    <button
      className={cn(
        'group flex w-full min-w-0 flex-col gap-0.5 border-l-2 py-1.5 pl-3 pr-2 text-left transition-colors',
        isSelected ? cn(toneAccentBorderLeft[tone], 'bg-card') : 'border-l-transparent hover:bg-card/60',
        hidden && 'opacity-60',
      )}
      onClick={onSelect}
      type="button"
    >
      <span className="flex min-w-0 items-center gap-2">
        <span className={cn(
          'min-w-0 flex-1 truncate text-[13px] font-medium',
          isSelected ? 'text-foreground' : 'text-foreground/90',
        )}>
          {normalized.title}
        </span>
        <Badge className={cn(
          'h-5 shrink-0 rounded px-1.5 text-[9px] font-semibold uppercase tracking-wide',
          hidden ? 'border-muted-foreground/20 bg-muted text-muted-foreground' : toneBadge[tone],
        )} variant="outline">
          {categoryLabelShort(category, t)}
        </Badge>
      </span>
      <span className="flex min-w-0 items-center gap-1.5 text-[11px] text-muted-foreground">
        {hidden ? (
          <span className="min-w-0 truncate">
            <span className="font-mono">@rules:{normalized.label}</span>{' '}
            <span className="inline-flex items-center gap-1 text-destructive/80">
              <EyeOff className="size-3 shrink-0" />
              {t.promptLibrary.hiddenEverywhere}
            </span>
          </span>
        ) : (
          <span className="min-w-0 truncate font-mono">
            @rules:{normalized.label}
            <span className="text-muted-foreground/70"> · {locations.join(' · ')}</span>
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

function UnsavedGuardModal({
  onCancel,
  onDiscard,
  onSave,
}: {
  onCancel: () => void
  onDiscard: () => void
  onSave: () => void
}) {
  const { t } = useLocale()
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/75 px-4 backdrop-blur"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onCancel()
      }}
    >
      <section
        aria-modal="true"
        className="w-full max-w-md overflow-hidden rounded-lg border border-border bg-background shadow-xl"
        role="dialog"
      >
        <div className="flex items-start gap-3 border-b border-border px-4 py-3">
          <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-md bg-warning-subtle text-warning">
            <AlertTriangle className="size-4" />
          </span>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-foreground">{t.promptLibrary.discardTitle}</h2>
            <p className="mt-0.5 text-xs leading-5 text-muted-foreground">{t.promptLibrary.discardBody}</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2 px-4 py-3">
          <Button autoFocus onClick={onCancel} type="button" variant="ghost">
            {t.promptLibrary.keepEditing}
          </Button>
          <Button
            className="text-destructive hover:text-destructive"
            onClick={onDiscard}
            type="button"
            variant="ghost"
          >
            {t.promptLibrary.discard}
          </Button>
          <Button className="gap-1.5" onClick={onSave} type="button">
            <Save className="size-4" />
            {t.promptLibrary.savePrompt}
          </Button>
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

function categoryLabelShort(category: ChatRuleCategory, t: ReturnType<typeof useLocale>['t']) {
  if (category === 'function') return t.promptLibrary.categoryFunctionShort
  if (category === 'context') return t.promptLibrary.categoryContextShort
  return t.promptLibrary.categoryInstructionShort
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

const categoryToTone: Record<ChatRuleCategory, MentionTone> = {
  context: 'warning',
  function: 'success',
  instruction: 'brand',
}
