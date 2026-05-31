import {
  Library,
  Plus,
  Save,
  Trash2,
  X,
} from '@/components/icons'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { ChatRuleRecord } from '@/features/project/types'
import {
  TextImproveButton,
  TextImproveFieldLayer,
  useTextImprovement,
  type TextImprovementApiOptions,
} from '@/features/textImprove'
import { useLocale } from '@/i18n/LocaleProvider'
import { appMotion } from '@/motion/transitions'
import { motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { createRuleId, normalizeRuleLabel } from './ruleLabels'
import { useRuleLibraryDraft } from './useRuleLibraryDraft'

export function RuleLibraryDialog({
  isOpen,
  onClose,
  onDeleteRule,
  onSaveRule,
  reduceMotion,
  rules,
  textImprovement,
}: {
  isOpen: boolean
  onClose: () => void
  onDeleteRule: (ruleId: string) => void
  onSaveRule: (rule: ChatRuleRecord) => void
  reduceMotion: boolean | null
  rules: ChatRuleRecord[]
  textImprovement: Omit<TextImprovementApiOptions, 'locale'>
}) {
  const { locale, t } = useLocale()
  const [promptCommitPulseKey, setPromptCommitPulseKey] = useState(0)
  const {
    draft,
    loadRule,
    markSaved,
    setContentDraft,
    setError,
    setLabelDraft,
    setTitleDraft,
    startNewRule,
  } = useRuleLibraryDraft()
  const wasOpenRef = useRef(false)
  const selectedRule = draft.selectedRuleId
    ? rules.find((rule) => rule.id === draft.selectedRuleId) ?? null
    : null
  const isExistingRule = Boolean(selectedRule)
  const promptTextImprove = useTextImprovement({
    ...textImprovement,
    locale,
    messages: {
      requestFailed: (message) => `${t.textImprove.requestFailed}: ${message}`,
      sensitiveText: t.textImprove.sensitiveText,
      unavailable: t.textImprove.unavailable,
    },
  })

  useEffect(() => {
    if (!isOpen) {
      wasOpenRef.current = false
      return
    }
    if (wasOpenRef.current) return

    wasOpenRef.current = true
    loadRule(rules[0] ?? null)
  }, [isOpen, loadRule, rules])

  useEffect(() => {
    if (!isOpen || !draft.selectedRuleId || draft.isDirty) return

    const currentRule = rules.find((rule) => rule.id === draft.selectedRuleId)
    if (!currentRule) {
      loadRule(rules[0] ?? null)
      return
    }

    const draftMatchesStoredRule = (
      draft.contentDraft === currentRule.contentMarkdown
      && draft.labelDraft === currentRule.label
      && draft.titleDraft === currentRule.title
    )
    if (!draftMatchesStoredRule) {
      loadRule(currentRule)
    }
  }, [
    draft.contentDraft,
    draft.isDirty,
    draft.labelDraft,
    draft.selectedRuleId,
    draft.titleDraft,
    isOpen,
    loadRule,
    rules,
  ])

  function selectRule(rule: ChatRuleRecord) {
    promptTextImprove.clearProposal()
    loadRule(rule)
  }

  function startNewRuleDraft() {
    promptTextImprove.clearProposal()
    startNewRule()
  }

  function saveRule() {
    const label = normalizeRuleLabel(draft.labelDraft)
    const title = draft.titleDraft.trim() || label
    const contentMarkdown = draft.contentDraft.trim()
    if (!label) {
      setError(t.chat.ruleLabelRequired)
      return
    }
    if (rules.some((rule) => rule.label === label && rule.id !== draft.selectedRuleId)) {
      setError(t.chat.ruleLabelDuplicate)
      return
    }
    if (!contentMarkdown) {
      setError(t.chat.rulePromptRequired)
      return
    }

    const now = new Date().toISOString()
    const rule: ChatRuleRecord = {
      contentMarkdown,
      createdAt: selectedRule?.createdAt ?? now,
      id: draft.selectedRuleId ?? createRuleId(),
      label,
      title,
      updatedAt: now,
    }
    onSaveRule(rule)
    markSaved(rule)
  }

  function deleteRule() {
    if (!selectedRule) return
    onDeleteRule(selectedRule.id)
    startNewRuleDraft()
  }

  async function improvePromptTemplate() {
    setError(null)
    try {
      await promptTextImprove.improve('prompt_template', draft.contentDraft)
    } catch (error) {
      setError(messageFromUnknown(error))
    }
  }

  function acceptPromptImprovement(text: string) {
    setContentDraft(text)
    promptTextImprove.clearProposal()
    setPromptCommitPulseKey((key) => key + 1)
  }

  if (!isOpen) return null

  return (
    <motion.div
      animate={{ opacity: 1 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/65 p-4 backdrop-blur-sm"
      initial={reduceMotion ? false : { opacity: 0 }}
      role="dialog"
      aria-modal="true"
      aria-label={t.chat.ruleLibrary}
    >
      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="grid max-h-[min(760px,calc(100svh-2rem))] w-full max-w-5xl overflow-hidden rounded-xl border border-border bg-card shadow-2xl md:grid-cols-[280px_minmax(0,1fr)]"
        initial={reduceMotion ? false : { opacity: 0, y: 10 }}
        transition={appMotion.panel}
      >
        <aside className="flex min-h-0 min-w-0 flex-col border-b border-border bg-surface/60 md:border-b-0 md:border-r">
          <div className="flex min-h-14 items-center justify-between gap-2 border-b border-border px-3">
            <div className="flex min-w-0 items-center gap-2">
              <Library className="size-4 shrink-0 text-muted-foreground" />
              <h2 className="truncate text-sm font-semibold text-foreground">
                {t.chat.rules}
              </h2>
            </div>
            <Button className="size-8" onClick={startNewRuleDraft} size="icon" type="button" variant="outline">
              <Plus className="size-4" />
            </Button>
          </div>
          <ScrollArea className="max-h-64 min-h-0 md:max-h-none md:flex-1">
            <div className="space-y-1 p-2">
              {rules.length > 0 ? rules.map((rule) => (
                <button
                  className={cnRuleButton(draft.selectedRuleId === rule.id)}
                  key={rule.id}
                  onClick={() => selectRule(rule)}
                  type="button"
                >
                  <span className="block truncate text-sm font-semibold text-foreground">
                    @rules:{rule.label}
                  </span>
                  <span className="mt-1 block truncate text-xs text-muted-foreground">
                    {rule.title}
                  </span>
                </button>
              )) : (
                <div className="rounded-md border border-dashed border-border p-4 text-center text-xs text-muted-foreground">
                  {t.chat.noRules}
                </div>
              )}
            </div>
          </ScrollArea>
        </aside>
        <section className="flex min-h-0 min-w-0 flex-col">
          <div className="flex min-h-14 items-center justify-between gap-2 border-b border-border px-4">
            <div className="min-w-0">
              <h3 className="truncate text-sm font-semibold text-foreground">
                {isExistingRule ? t.chat.editRule : t.chat.addRule}
              </h3>
              <p className="truncate text-xs text-muted-foreground">
                {t.chat.ruleLibraryDescription}
              </p>
            </div>
            <Button aria-label={t.common.close} className="size-8" onClick={onClose} size="icon" type="button" variant="ghost">
              <X className="size-4" />
            </Button>
          </div>
          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
            <label className="block space-y-1.5">
              <span className="text-xs font-semibold text-muted-foreground">{t.chat.ruleLabel}</span>
              <div className="flex items-center rounded-md border border-border bg-background px-2 focus-within:ring-2 focus-within:ring-ring">
                <span className="text-sm font-semibold text-muted-foreground">@rules:</span>
                <input
                  className="min-w-0 flex-1 border-0 bg-transparent px-1 py-2 text-sm font-semibold text-foreground outline-none"
                  maxLength={48}
                  onChange={(event) => {
                    setLabelDraft(normalizeRuleLabel(event.target.value))
                  }}
                  placeholder={t.chat.ruleLabelPlaceholder}
                  value={draft.labelDraft}
                />
              </div>
            </label>
            <label className="block space-y-1.5">
              <span className="text-xs font-semibold text-muted-foreground">{t.chat.ruleTitle}</span>
              <input
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                onChange={(event) => setTitleDraft(event.target.value)}
                placeholder={t.chat.ruleTitlePlaceholder}
                value={draft.titleDraft}
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-xs font-semibold text-muted-foreground">{t.chat.prompt}</span>
              <motion.div
                animate={
                  promptCommitPulseKey > 0 && !reduceMotion
                    ? {
                      boxShadow: [
                        '0 0 0 0 color-mix(in oklch, var(--brand) 0%, transparent)',
                        '0 0 0 3px color-mix(in oklch, var(--brand) 18%, transparent)',
                        '0 0 0 0 color-mix(in oklch, var(--brand) 0%, transparent)',
                      ],
                    }
                    : { boxShadow: '0 0 0 0 color-mix(in oklch, var(--brand) 0%, transparent)' }
                }
                className="relative rounded-md"
                transition={{ duration: 0.34, ease: appMotion.panel.ease }}
              >
                <textarea
                  className="block min-h-64 w-full resize-y rounded-md border border-border bg-background px-3 py-2 pr-11 text-sm leading-6 text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  onChange={(event) => {
                    promptTextImprove.clearProposal()
                    setContentDraft(event.target.value)
                  }}
                  placeholder={t.chat.ruleContentPlaceholder}
                  value={draft.contentDraft}
                />
                <TextImproveButton
                  className="absolute right-2 top-2 bg-background/80"
                  disabled={!draft.contentDraft.trim()}
                  isLoading={promptTextImprove.isImproving}
                  label={t.textImprove.improve}
                  loadingLabel={t.textImprove.improving}
                  onClick={() => void improvePromptTemplate()}
                  reduceMotion={reduceMotion}
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
              </motion.div>
            </label>
          </div>
          <div className="flex min-h-14 flex-wrap items-center justify-between gap-2 border-t border-border px-4 py-3">
            <div className="min-w-0 text-xs font-medium text-warning">
              {draft.error}
            </div>
            <div className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-2">
              <Button
                className="gap-1.5 text-destructive hover:text-destructive"
                disabled={!selectedRule}
                onClick={deleteRule}
                type="button"
                variant="ghost"
              >
                <Trash2 className="size-4" />
                {t.chat.deleteRule}
              </Button>
              <Button className="gap-1.5" onClick={saveRule} type="button">
                <Save className="size-4" />
                {t.chat.saveRule}
              </Button>
            </div>
          </div>
        </section>
      </motion.div>
    </motion.div>
  )
}

function cnRuleButton(isSelected: boolean) {
  return [
    'w-full min-w-0 rounded-md border border-transparent px-3 py-2 text-left transition-colors hover:bg-background',
    isSelected ? 'border-border bg-background shadow-[0_1px_2px_var(--shadow-hairline)]' : '',
  ].filter(Boolean).join(' ')
}

function messageFromUnknown(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}
