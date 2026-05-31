import {
  FileText,
  Globe2,
  ListChecks,
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
  type SetStateAction,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import { motion } from 'motion/react'
import { resizeTextareaToRows } from '@/features/composer/textareaAutosize'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { CreateResearchRunRequest } from '@/features/researchRuns/types'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'

type ComposerProps = {
  form: ComposerFormState
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
  { form, onSubmit, reduceMotion, selectedStack, setForm },
  ref,
) {
  const { t } = useLocale()
  const questionTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const canSubmit = form.question.trim().length > 0

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
      className="sticky bottom-0 z-20 rounded-lg border border-border bg-card/98 p-3 shadow-[0_18px_50px_var(--shadow-soft)] backdrop-blur will-change-opacity"
    >
      <form onSubmit={submitResearch}>
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
          <Textarea
            aria-label={t.composer.placeholder}
            className={cn(
              'min-h-24 resize-none border-0 py-2 pl-3 pr-2 text-base leading-6 focus-visible:ring-0',
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
          <Button
            aria-label={t.composer.send}
            className="mb-2 h-8 w-8"
            disabled={!canSubmit}
            size="icon"
            type="submit"
            variant="ghost"
          >
            <SendHorizontal className="size-3.5" />
          </Button>
        </div>
        <Separator className="my-1" />
        <div className="flex flex-wrap items-center gap-1">
          <ComposerSelect
            icon={FileText}
            label={t.composer.reportProfile}
            onValueChange={(value) => setForm((currentForm) => ({
              ...currentForm,
              reportProfile: value as ComposerFormState['reportProfile'],
            }))}
            options={[
              { label: t.composer.compact, value: 'compact' },
              { label: t.composer.deep, value: 'deep' },
            ]}
            value={form.reportProfile}
          />
          <ComposerSelect
            icon={Shield}
            label={t.composer.confidenceTarget}
            onValueChange={(value) => setForm((currentForm) => ({
              ...currentForm,
              confidenceStop: Number(value) as ComposerFormState['confidenceStop'],
            }))}
            options={[
              { label: '7 / 10', value: '7' },
              { label: '8 / 10', value: '8' },
              { label: '9 / 10', value: '9' },
            ]}
            value={String(form.confidenceStop)}
          />
          <ComposerSelect
            icon={Search}
            label={t.composer.firstQueries}
            onValueChange={(value) => setForm((currentForm) => ({
              ...currentForm,
              firstRoundQueries: Number(value) as ComposerFormState['firstRoundQueries'],
            }))}
            options={[
              { label: '4', value: '4' },
              { label: '6', value: '6' },
              { label: '8', value: '8' },
            ]}
            value={String(form.firstRoundQueries)}
          />
          <ComposerSelect
            icon={Repeat2}
            label={t.composer.maxRounds}
            onValueChange={(value) => setForm((currentForm) => {
              const maxRounds = Number(value) as ComposerFormState['maxRounds']
              return {
                ...currentForm,
                maxRounds,
                minRounds: Math.min(currentForm.minRounds, maxRounds) as ComposerFormState['minRounds'],
              }
            })}
            options={[
              { label: '2', value: '2' },
              { label: '3', value: '3' },
              { label: '4', value: '4' },
              { label: '5', value: '5' },
            ]}
            value={String(form.maxRounds)}
          />
          <div className="ml-auto flex items-center gap-1">
            <ComposerStatusMenu
              confidenceStop={form.confidenceStop}
              firstRoundQueries={form.firstRoundQueries}
              maxRounds={form.maxRounds}
              minRounds={form.minRounds}
              reportProfile={form.reportProfile}
              selectedStack={selectedStack}
              webSearch={form.webSearch}
            />
            <DropdownMenu>
              <Tooltip>
                <DropdownMenuTrigger asChild>
                  <TooltipTrigger asChild>
                    <Button
                      aria-label={t.composer.moreSettings}
                      className={composerIconButtonClassName()}
                      type="button"
                      variant="ghost"
                    >
                      <SlidersHorizontal className="size-3.5" />
                    </Button>
                  </TooltipTrigger>
                </DropdownMenuTrigger>
                <TooltipContent>{t.composer.moreSettings}</TooltipContent>
              </Tooltip>
              <DropdownMenuContent align="end" className="w-64">
                <DropdownMenuLabel>{t.composer.moreSettings}</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <ComposerMenuToggle
                  checked={form.webSearch}
                  icon={Globe2}
                  label={t.composer.webSearch}
                  onCheckedChange={(checked) => setForm((currentForm) => ({
                    ...currentForm,
                    webSearch: checked,
                  }))}
                />
                <DropdownMenuSeparator />
                <DropdownMenuLabel className="text-xs text-muted-foreground">
                  {t.composer.minRounds}
                </DropdownMenuLabel>
                {[1, 2].map((rounds) => (
                  <DropdownMenuCheckboxItem
                    checked={form.minRounds === rounds}
                    disabled={rounds > form.maxRounds}
                    key={rounds}
                    onCheckedChange={() => setForm((currentForm) => ({
                      ...currentForm,
                      minRounds: rounds as ComposerFormState['minRounds'],
                    }))}
                  >
                    {rounds}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
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
  reportProfile,
  selectedStack,
  webSearch,
}: {
  confidenceStop: ComposerFormState['confidenceStop']
  firstRoundQueries: ComposerFormState['firstRoundQueries']
  maxRounds: ComposerFormState['maxRounds']
  minRounds: ComposerFormState['minRounds']
  reportProfile: ComposerFormState['reportProfile']
  selectedStack: string
  webSearch: boolean
}) {
  const { t } = useLocale()
  const reportProfileLabel = reportProfile === 'compact' ? t.composer.compact : t.composer.deep

  return (
    <DropdownMenu>
      <Tooltip>
        <DropdownMenuTrigger asChild>
          <TooltipTrigger asChild>
            <Button
              aria-label={t.composer.settingsSummary}
              className={composerIconButtonClassName()}
              type="button"
              variant="ghost"
            >
              <ListChecks className="size-3.5" />
            </Button>
          </TooltipTrigger>
        </DropdownMenuTrigger>
        <TooltipContent>{t.composer.settingsSummary}</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" className="w-72">
        <DropdownMenuLabel>{t.composer.settingsSummary}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <div className="grid gap-1 p-1">
          <StatusRow label={t.common.stack} value={selectedStack} />
          <StatusRow label={t.composer.reportProfile} value={reportProfileLabel} />
          <StatusRow label={t.composer.confidenceTarget} value={`${confidenceStop} / 10`} />
          <StatusRow label={t.composer.firstQueries} value={String(firstRoundQueries)} />
          <StatusRow label={t.composer.maxRounds} value={String(maxRounds)} />
          <StatusRow label={t.composer.minRounds} value={String(minRounds)} />
          <StatusRow
            label={t.composer.webSearch}
            value={webSearch ? t.composer.enabled : t.composer.disabled}
          />
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 rounded-md px-2 py-1.5">
      <span className="truncate text-xs font-semibold text-muted-foreground">{label}</span>
      <span className="max-w-40 truncate text-right text-xs font-semibold text-foreground">{value}</span>
    </div>
  )
}

function ComposerMenuToggle({
  checked,
  icon: Icon,
  label,
  onCheckedChange,
}: {
  checked: boolean
  icon: LucideIcon
  label: string
  onCheckedChange: (checked: boolean) => void
}) {
  const { t } = useLocale()

  return (
    <DropdownMenuCheckboxItem
      checked={checked}
      className="gap-3 py-2 pl-2 pr-2 [&>span:first-child]:hidden"
      onCheckedChange={onCheckedChange}
      onSelect={(event) => event.preventDefault()}
    >
      <Icon className="size-4 shrink-0 text-muted-foreground" />
      <span className="grid min-w-0 flex-1 text-left leading-tight">
        <span className="truncate text-sm font-medium">{label}</span>
        <span className="truncate text-xs text-muted-foreground">
          {checked ? t.composer.enabled : t.composer.disabled}
        </span>
      </span>
      <ToggleVisual checked={checked} />
    </DropdownMenuCheckboxItem>
  )
}

function ComposerSelect({
  icon: Icon,
  label,
  onValueChange,
  options,
  value,
}: {
  icon: LucideIcon
  label: string
  onValueChange: (value: string) => void
  options: Array<{ label: string; value: string }>
  value: string
}) {
  const selectedOption = options.find((option) => option.value === value)
  const valueLabel = selectedOption?.label ?? value
  const triggerLabel = `${label}: ${valueLabel}`

  return (
    <Select onValueChange={onValueChange} value={value}>
      <Tooltip>
        <TooltipTrigger asChild>
          <SelectTrigger
            aria-label={triggerLabel}
            className={cn(composerIconButtonClassName(), 'w-10 gap-0.5 px-1')}
          >
            <Icon className="size-3.5 shrink-0" />
            <span className="sr-only">
              <SelectValue />
            </span>
          </SelectTrigger>
        </TooltipTrigger>
        <TooltipContent>{triggerLabel}</TooltipContent>
      </Tooltip>
      <SelectContent>
        <SelectGroup>
          <SelectLabel className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
            {label}
          </SelectLabel>
          <SelectSeparator />
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  )
}

function composerIconButtonClassName() {
  return 'h-7 w-7 rounded-md border border-transparent bg-transparent p-0 text-muted-foreground shadow-none hover:bg-accent hover:text-foreground focus-visible:ring-1 data-[state=open]:bg-accent data-[state=open]:text-foreground'
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
