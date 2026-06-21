import {
  Check,
  FileText,
  Globe2,
  ListChecks,
  PanelBottomClose,
  Repeat2,
  Search,
  SendHorizontal,
  Shield,
  SlidersHorizontal,
  type LucideIcon,
} from '@/components/icons'
import {
  type Dispatch,
  forwardRef,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type SetStateAction,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import { motion } from 'motion/react'
import { resizeTextareaToRows } from '@/features/composer/textareaAutosize'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { OptionMenuHeader, OptionMenuItem, optionMenuContentClassName } from '@/components/ui/option-menu'
import { Separator } from '@/components/ui/separator'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { CreateResearchRunRequest } from '@/features/researchRuns/types'
import { ComposerIconButton, composerIconButtonClassName } from '@/features/composer/ComposerIconButton'
import { QuotaMeter } from '@/features/quota/QuotaMeter'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'

type ComposerProps = {
  form: ComposerFormState
  onHide: () => void
  onSubmit: (request: CreateResearchRunRequest) => void
  reduceMotion: boolean | null
  selectedStack: string
  setForm: Dispatch<SetStateAction<ComposerFormState>>
}

export type ComposerFormState = {
  confidenceStop: 7 | 8 | 9
  firstRoundQueries: 4 | 6 | 8
  maxRounds: 2 | 3 | 4 | 5
  minRounds: 1 | 2
  question: string
  reportProfile: 'compact' | 'deep'
  webSearch: boolean
}

export const defaultComposerFormState: ComposerFormState = {
  confidenceStop: 8,
  firstRoundQueries: 6,
  maxRounds: 4,
  minRounds: 2,
  question: '',
  reportProfile: 'deep',
  webSearch: true,
}

type ComposerMenuKey =
  | 'confidence'
  | 'maxRounds'
  | 'more'
  | 'queries'
  | 'report'
  | 'summary'

type ComposerOption = {
  description: string
  label: string
  value: string
}

function switchComposerMenu(
  currentMenu: ComposerMenuKey | null,
  nextMenu: ComposerMenuKey,
  onOpenMenuChange: (menu: ComposerMenuKey | null) => void,
) {
  if (currentMenu === nextMenu) return

  if (currentMenu === null) {
    onOpenMenuChange(nextMenu)
    return
  }

  window.setTimeout(() => onOpenMenuChange(nextMenu), 0)
}

export function buildComposerRequest(
  form: ComposerFormState,
  question: string,
  selectedStack: string,
): CreateResearchRunRequest {
  return {
    agentOverrides: {
      confidenceStop: form.confidenceStop,
      firstRoundQueries: form.firstRoundQueries,
      maxRounds: form.maxRounds,
      minRounds: Math.min(form.minRounds, form.maxRounds) as ComposerFormState['minRounds'],
      reportProfile: form.reportProfile,
    },
    mode: form.webSearch ? 'research' : 'direct_llm',
    question: question.trim(),
    stack: selectedStack,
  }
}

export const Composer = forwardRef<HTMLElement, ComposerProps>(function Composer(
  { form, onHide, onSubmit, reduceMotion, selectedStack, setForm },
  ref,
) {
  const { t } = useLocale()
  const questionTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const [openMenu, setOpenMenu] = useState<ComposerMenuKey | null>(null)
  const canSubmit = form.question.trim().length > 0
  const reportProfileOptions: ComposerOption[] = [
    {
      description: t.composer.optionCompactDescription,
      label: t.composer.compact,
      value: 'compact',
    },
    {
      description: t.composer.optionDeepDescription,
      label: t.composer.deep,
      value: 'deep',
    },
  ]
  const confidenceOptions: ComposerOption[] = [
    {
      description: t.composer.optionConfidence7Description,
      label: '7 / 10',
      value: '7',
    },
    {
      description: t.composer.optionConfidence8Description,
      label: '8 / 10',
      value: '8',
    },
    {
      description: t.composer.optionConfidence9Description,
      label: '9 / 10',
      value: '9',
    },
  ]
  const firstQueryOptions: ComposerOption[] = [
    {
      description: t.composer.optionQueries4Description,
      label: '4',
      value: '4',
    },
    {
      description: t.composer.optionQueries6Description,
      label: '6',
      value: '6',
    },
    {
      description: t.composer.optionQueries8Description,
      label: '8',
      value: '8',
    },
  ]
  const maxRoundOptions: ComposerOption[] = [
    {
      description: t.composer.optionRounds2Description,
      label: '2',
      value: '2',
    },
    {
      description: t.composer.optionRounds3Description,
      label: '3',
      value: '3',
    },
    {
      description: t.composer.optionRounds4Description,
      label: '4',
      value: '4',
    },
    {
      description: t.composer.optionRounds5Description,
      label: '5',
      value: '5',
    },
  ]

  useLayoutEffect(() => {
    resizeTextareaToRows(questionTextareaRef.current, 4)
  }, [form.question])

  function submitResearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    submitCurrentQuestion()
  }

  function handleQuestionKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key !== 'Enter'
      || event.ctrlKey
      || event.metaKey
      || event.shiftKey
      || event.nativeEvent.isComposing
    ) {
      return
    }

    event.preventDefault()
    submitCurrentQuestion()
  }

  function submitCurrentQuestion() {
    if (!canSubmit) return

    onSubmit(buildComposerRequest(form, form.question, selectedStack))
    setForm((currentForm) => ({ ...currentForm, question: '' }))
  }

  return (
    <motion.section
      id="research-composer"
      ref={ref}
      initial={reduceMotion ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={appMotion.composer}
      className="shrink-0 px-4 pb-4 pt-2"
    >
      <form className="mx-auto max-w-4xl" onSubmit={submitResearch}>
        <div className="relative rounded-xl border border-border bg-card px-3 py-2.5 shadow-[0_8px_28px_-12px_var(--shadow-soft)] transition-[border-color,box-shadow] duration-150 focus-within:border-brand/60 focus-within:ring-2 focus-within:ring-brand/15">
          <Textarea
            aria-label={t.composer.placeholder}
            className={cn(
              'min-h-16 resize-none border-0 bg-transparent pb-2 pl-2 pr-11 pt-2 text-sm font-normal leading-6 shadow-none placeholder:text-muted-foreground/70 focus-visible:ring-0',
              '[scrollbar-width:thin]',
              '[scrollbar-color:color-mix(in_oklch,var(--muted-foreground)_22%,transparent)_transparent]',
              '[&::-webkit-scrollbar]:w-1',
              '[&::-webkit-scrollbar-track]:bg-transparent',
              '[&::-webkit-scrollbar-thumb]:rounded-full',
              '[&::-webkit-scrollbar-thumb]:bg-border/70',
              'hover:[&::-webkit-scrollbar-thumb]:bg-muted-foreground/35',
            )}
            onChange={(event) => setForm((currentForm) => ({
              ...currentForm,
              question: event.target.value,
            }))}
            onKeyDown={handleQuestionKeyDown}
            placeholder={t.composer.placeholder}
            ref={questionTextareaRef}
            rows={1}
            value={form.question}
          />
          <div className="mt-1.5 flex items-center justify-between gap-2 border-t border-border/70 pt-1.5">
            <div className="flex min-w-0 items-center gap-1 overflow-hidden">
              <ComposerIconButton
                icon={PanelBottomClose}
                label={t.composer.hide}
                onClick={onHide}
              />
              <Separator className="mx-0.5 h-5" orientation="vertical" />
              <ComposerParameterMenu
                icon={FileText}
                label={t.composer.reportProfile}
                menuKey="report"
                onOpenMenuChange={setOpenMenu}
                onValueChange={(value) => setForm((currentForm) => ({
                  ...currentForm,
                  reportProfile: value as ComposerFormState['reportProfile'],
                }))}
                openMenu={openMenu}
                options={reportProfileOptions}
                value={form.reportProfile}
              />
              <ComposerParameterMenu
                icon={Shield}
                label={t.composer.confidenceTarget}
                menuKey="confidence"
                onOpenMenuChange={setOpenMenu}
                onValueChange={(value) => setForm((currentForm) => ({
                  ...currentForm,
                  confidenceStop: Number(value) as ComposerFormState['confidenceStop'],
                }))}
                openMenu={openMenu}
                options={confidenceOptions}
                value={String(form.confidenceStop)}
              />
              <ComposerParameterMenu
                icon={Search}
                label={t.composer.firstQueries}
                menuKey="queries"
                onOpenMenuChange={setOpenMenu}
                onValueChange={(value) => setForm((currentForm) => ({
                  ...currentForm,
                  firstRoundQueries: Number(value) as ComposerFormState['firstRoundQueries'],
                }))}
                openMenu={openMenu}
                options={firstQueryOptions}
                value={String(form.firstRoundQueries)}
              />
              <ComposerParameterMenu
                icon={Repeat2}
                label={t.composer.maxRounds}
                menuKey="maxRounds"
                onOpenMenuChange={setOpenMenu}
                onValueChange={(value) => setForm((currentForm) => {
                  const maxRounds = Number(value) as ComposerFormState['maxRounds']
                  return {
                    ...currentForm,
                    maxRounds,
                    minRounds: Math.min(currentForm.minRounds, maxRounds) as ComposerFormState['minRounds'],
                  }
                })}
                openMenu={openMenu}
                options={maxRoundOptions}
                value={String(form.maxRounds)}
              />
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <QuotaMeter />
              <ComposerStatusMenu
                confidenceStop={form.confidenceStop}
                firstRoundQueries={form.firstRoundQueries}
                maxRounds={form.maxRounds}
                minRounds={form.minRounds}
                onOpenMenuChange={setOpenMenu}
                openMenu={openMenu}
                reportProfile={form.reportProfile}
                selectedStack={selectedStack}
                webSearch={form.webSearch}
              />
              <DropdownMenu
                modal={false}
                onOpenChange={(isOpen) => setOpenMenu(isOpen ? 'more' : null)}
                open={openMenu === 'more'}
              >
                <Tooltip>
                  <DropdownMenuTrigger asChild>
                    <TooltipTrigger asChild>
                      <Button
                        aria-label={t.composer.moreSettings}
                        className={composerIconButtonClassName}
                        onPointerDown={(event) => {
                          if (openMenu === 'more') return

                          event.preventDefault()
                          event.currentTarget.focus()
                          switchComposerMenu(openMenu, 'more', setOpenMenu)
                        }}
                        type="button"
                        variant="ghost"
                      >
                        <SlidersHorizontal className="size-3.5" />
                      </Button>
                    </TooltipTrigger>
                  </DropdownMenuTrigger>
                  <TooltipContent>{t.composer.moreSettings}</TooltipContent>
                </Tooltip>
                <DropdownMenuContent align="end" className={optionMenuContentClassName} side="top" sideOffset={8}>
                  <OptionMenuHeader count={2} title={t.composer.moreSettings} />
                  <div className="py-1">
                    <ComposerMenuToggle
                      checked={form.webSearch}
                      description={t.composer.webSearchDescription}
                      icon={Globe2}
                      label={t.composer.webSearch}
                      onCheckedChange={(checked) => setForm((currentForm) => ({
                        ...currentForm,
                        webSearch: checked,
                      }))}
                    />
                    <DropdownMenuSeparator className="mx-0 my-1" />
                    <div className="px-2.5 pb-1 pt-1.5">
                      <div className="flex items-center gap-1.5">
                        <Repeat2 className="icon-sm shrink-0 text-muted-foreground/70" />
                        <span className="t-list text-foreground">{t.composer.minRounds}</span>
                        <span className="ml-auto rounded-md bg-brand-subtle px-1.5 py-0.5 t-hint font-medium tabular-nums text-brand">
                          {form.minRounds}
                        </span>
                      </div>
                      <p className="mt-0.5 t-meta-sm text-muted-foreground">{t.composer.minRoundsDescription}</p>
                      <div className="mt-2 grid h-7 grid-cols-2 rounded-md bg-surface p-0.5">
                        {[1, 2].map((rounds) => {
                          const disabled = rounds > form.maxRounds
                          const active = form.minRounds === rounds
                          return (
                            <button
                              aria-pressed={active}
                              className={cn(
                                'inline-flex items-center justify-center gap-1 rounded px-2 t-meta-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
                                active && 'bg-brand-subtle text-brand shadow-[0_1px_2px_var(--shadow-hairline)] ring-1 ring-brand/20',
                                !active && !disabled && 'text-muted-foreground hover:bg-background hover:text-foreground',
                              )}
                              disabled={disabled}
                              key={rounds}
                              onClick={() => setForm((currentForm) => ({
                                ...currentForm,
                                minRounds: rounds as ComposerFormState['minRounds'],
                              }))}
                              type="button"
                            >
                              <span className="tabular-nums">{rounds}</span>
                              <span className="flex icon-xs items-center justify-center">
                                {active ? <Check className="icon-xs" /> : null}
                              </span>
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  </div>
                </DropdownMenuContent>
              </DropdownMenu>
              <Button
                aria-label={t.composer.send}
                className={cn(
                  'size-7 shrink-0 rounded-md',
                  canSubmit
                    ? 'bg-brand text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground'
                    : 'text-muted-foreground/45',
                )}
                disabled={!canSubmit}
                size="icon"
                type="submit"
                variant={canSubmit ? 'default' : 'ghost'}
              >
                <SendHorizontal className="size-4" />
              </Button>
            </div>
          </div>
        </div>
      </form>
    </motion.section>
  )
})

function ComposerStatusMenu({
  confidenceStop,
  firstRoundQueries,
  maxRounds,
  minRounds,
  onOpenMenuChange,
  openMenu,
  reportProfile,
  selectedStack,
  webSearch,
}: {
  confidenceStop: ComposerFormState['confidenceStop']
  firstRoundQueries: ComposerFormState['firstRoundQueries']
  maxRounds: ComposerFormState['maxRounds']
  minRounds: ComposerFormState['minRounds']
  onOpenMenuChange: (menu: ComposerMenuKey | null) => void
  openMenu: ComposerMenuKey | null
  reportProfile: ComposerFormState['reportProfile']
  selectedStack: string
  webSearch: boolean
}) {
  const { t } = useLocale()
  const reportProfileLabel = reportProfile === 'compact' ? t.composer.compact : t.composer.deep

  return (
    <DropdownMenu
      modal={false}
      onOpenChange={(isOpen) => onOpenMenuChange(isOpen ? 'summary' : null)}
      open={openMenu === 'summary'}
    >
      <Tooltip>
        <DropdownMenuTrigger asChild>
          <TooltipTrigger asChild>
            <Button
              aria-label={t.composer.settingsSummary}
              className={composerIconButtonClassName}
              onPointerDown={(event) => {
                if (openMenu === 'summary') return

                event.preventDefault()
                event.currentTarget.focus()
                switchComposerMenu(openMenu, 'summary', onOpenMenuChange)
              }}
              type="button"
              variant="ghost"
            >
              <ListChecks className="size-3.5" />
            </Button>
          </TooltipTrigger>
        </DropdownMenuTrigger>
        <TooltipContent>{t.composer.settingsSummary}</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" className={optionMenuContentClassName} side="top" sideOffset={8}>
        <OptionMenuHeader count={7} title={t.composer.settingsSummary} />
        <div className="py-1">
          <SummaryGroup label={t.composer.summaryStrategy}>
            <StatusRow label={t.common.stack} value={selectedStack} />
            <StatusRow
              label={t.composer.webSearch}
              tone={webSearch ? 'success' : 'muted'}
              value={webSearch ? t.composer.enabled : t.composer.disabled}
            />
          </SummaryGroup>
          <DropdownMenuSeparator className="mx-0 my-1" />
          <SummaryGroup label={t.composer.summaryPlanning}>
            <StatusRow label={t.composer.firstQueries} value={String(firstRoundQueries)} />
            <StatusRow label={t.composer.minRounds} value={String(minRounds)} />
            <StatusRow label={t.composer.maxRounds} value={String(maxRounds)} />
          </SummaryGroup>
          <DropdownMenuSeparator className="mx-0 my-1" />
          <SummaryGroup label={t.composer.summaryStopAndOutput}>
            <StatusRow label={t.composer.confidenceTarget} value={`${confidenceStop} / 10`} />
            <StatusRow label={t.composer.reportProfile} value={reportProfileLabel} />
          </SummaryGroup>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function SummaryGroup({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div>
      <div className="px-2.5 pb-0.5 pt-1.5 t-caption text-muted-foreground/60">{label}</div>
      <div className="grid gap-0.5">{children}</div>
    </div>
  )
}

function StatusRow({
  label,
  tone = 'default',
  value,
}: {
  label: string
  tone?: 'default' | 'muted' | 'success'
  value: string
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-2.5 py-1">
      <span className="truncate t-meta-sm text-muted-foreground">{label}</span>
      <span
        className={cn(
          'max-w-36 truncate rounded-md px-1.5 py-0.5 text-right t-meta-sm font-medium',
          tone === 'success' && 'bg-success-subtle text-success',
          tone === 'muted' && 'bg-surface text-muted-foreground',
          tone === 'default' && 'bg-background text-foreground',
        )}
      >
        {value}
      </span>
    </div>
  )
}

function ComposerMenuToggle({
  checked,
  description,
  icon: Icon,
  label,
  onCheckedChange,
}: {
  checked: boolean
  description: string
  icon: LucideIcon
  label: string
  onCheckedChange: (checked: boolean) => void
}) {
  const { t } = useLocale()

  return (
    <DropdownMenuItem
      className="group relative items-center gap-2.5 rounded-none px-2.5 py-1.5 hover:bg-accent/50 focus:bg-accent/80 data-[highlighted]:bg-accent/80"
      onSelect={(event) => {
        event.preventDefault()
        onCheckedChange(!checked)
      }}
    >
      <span
        className={cn(
          'absolute inset-y-1 left-0 w-0.5 rounded-full opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100 group-data-[highlighted]:opacity-100',
          checked ? 'bg-success' : 'bg-muted-foreground/50',
        )}
      />
      <Icon
        className={cn(
          'icon-md shrink-0 transition-colors',
          checked
            ? 'text-success'
            : 'text-muted-foreground/70 group-hover:text-foreground group-focus:text-foreground group-data-[highlighted]:text-foreground',
        )}
      />
      <span className="min-w-0 flex-1 text-left">
        <span className="block truncate t-list text-foreground">{label}</span>
        <span className="block truncate t-meta-sm text-muted-foreground">{description}</span>
      </span>
      <button
        aria-label={`${label}: ${checked ? t.composer.enabled : t.composer.disabled}`}
        className="shrink-0"
        onClick={(event) => {
          event.stopPropagation()
          onCheckedChange(!checked)
        }}
        type="button"
      >
        <ToggleVisual checked={checked} />
      </button>
    </DropdownMenuItem>
  )
}

function ComposerParameterMenu({
  icon: Icon,
  label,
  menuKey,
  onOpenMenuChange,
  onValueChange,
  openMenu,
  options,
  value,
}: {
  icon: LucideIcon
  label: string
  menuKey: ComposerMenuKey
  onOpenMenuChange: (menu: ComposerMenuKey | null) => void
  onValueChange: (value: string) => void
  openMenu: ComposerMenuKey | null
  options: ComposerOption[]
  value: string
}) {
  const selectedOption = options.find((option) => option.value === value)
  const valueLabel = selectedOption?.label ?? value
  const triggerLabel = `${label}: ${valueLabel}`

  return (
    <DropdownMenu
      modal={false}
      onOpenChange={(isOpen) => onOpenMenuChange(isOpen ? menuKey : null)}
      open={openMenu === menuKey}
    >
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button
              aria-label={triggerLabel}
              className={cn(composerIconButtonClassName, 'w-10 gap-0.5 px-1')}
              onPointerDown={(event) => {
                if (openMenu === menuKey) return

                event.preventDefault()
                event.currentTarget.focus()
                switchComposerMenu(openMenu, menuKey, onOpenMenuChange)
              }}
              type="button"
              variant="ghost"
            >
              <Icon className="size-3.5 shrink-0" />
              <span className="sr-only">{valueLabel}</span>
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent>{triggerLabel}</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="start" className={optionMenuContentClassName} side="top" sideOffset={8}>
        <OptionMenuHeader count={options.length} title={label} value={valueLabel} />
        <div className="py-1">
          {options.map((option) => (
            <OptionMenuItem
              active={option.value === value}
              description={option.description}
              icon={Icon}
              key={option.value}
              label={option.label}
              onSelect={() => onValueChange(option.value)}
            />
          ))}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function ToggleVisual({ checked }: { checked: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        'inline-flex h-5 w-9 shrink-0 items-center rounded-full border-2 border-transparent shadow-sm transition-colors',
        checked ? 'bg-primary' : 'bg-input',
      )}
    >
      <span
        className={cn(
          'block h-4 w-4 rounded-full bg-background shadow-lg ring-0 transition-transform',
          checked ? 'translate-x-4' : 'translate-x-0',
        )}
      />
    </span>
  )
}
