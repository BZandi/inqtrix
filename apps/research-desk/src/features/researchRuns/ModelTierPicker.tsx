import { Bot, Check, ChevronDown, Info, Server } from '@/components/icons'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type { ChatModelOption, ChatModelTier, NodeModelResolution } from './types'
import {
  modelDetailLabel,
  modelEffortLabel,
  modelNameLabel,
  modelTierDescription,
  modelTierLabel,
} from './modelLabels'

const modelTierOrder: ChatModelTier[] = ['high', 'mid', 'fast']

type ModelTierPickerProps = {
  defaultModel: NodeModelResolution | null
  disabled: boolean
  onChange: (tier: ChatModelTier | null) => void
  options: ChatModelOption[]
  optionsStatus: 'available' | 'missing' | 'unresolved'
  selectedTier: ChatModelTier | null
}

export function ModelTierPicker({
  defaultModel,
  disabled,
  onChange,
  options,
  optionsStatus,
  selectedTier,
}: ModelTierPickerProps) {
  const { t } = useLocale()
  const selectedOption = selectedTier ? modelOptionForTier(options, selectedTier) : null
  const activeModel = selectedOption ?? defaultModel ?? modelOptionForTier(options, 'mid') ?? null
  const unavailableLabel = optionsStatus === 'unresolved'
    ? t.chat.modelMetadataMissing
    : t.chat.modelDiscoveryMissing
  const activeLabel = selectedTier && optionsStatus !== 'available'
    ? `${modelTierLabel(selectedTier, t.chat)} · ${unavailableLabel}`
    : `${modelNameLabel(activeModel, t.chat.modelUnknown)} · ${modelEffortLabel(activeModel, t.chat)}`

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={t.chat.modelPicker}
          className={cn(
            'h-7 min-w-0 max-w-[min(48vw,17rem)] shrink rounded-md px-1.5 text-xs font-semibold text-muted-foreground hover:bg-accent/70 hover:text-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-0',
            'data-[state=open]:bg-accent data-[state=open]:text-foreground',
          )}
          disabled={disabled}
          type="button"
          variant="ghost"
        >
          <span className="min-w-0 truncate">{activeLabel}</span>
          <ChevronDown className="size-3 shrink-0 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="w-72 max-w-[calc(100vw-2rem)] overflow-x-hidden rounded-xl p-0 shadow-lg"
        side="top"
        sideOffset={8}
      >
        <div className="flex items-center gap-1.5 border-b border-border px-2.5 py-1.5">
          <span className="t-meta-sm font-medium text-muted-foreground">{t.chat.modelPicker}</span>
          <span className="ml-auto t-hint tabular-nums text-muted-foreground/50">
            {optionsStatus === 'available' ? options.length : 0}
          </span>
        </div>

        <div className="max-h-80 overflow-x-hidden overflow-y-auto py-1">
          <ModelMenuRow
            active={selectedTier == null}
            description={t.chat.modelServerDefaultDescription}
            detail={modelDetailLabel(defaultModel, t.chat)}
            icon="server"
            label={t.chat.modelServerDefault}
            onSelect={() => onChange(null)}
          />
          <DropdownMenuSeparator className="mx-0 my-1" />
          {optionsStatus === 'available' ? modelTierOrder.map((tier) => {
            const option = modelOptionForTier(options, tier)
            const description = modelTierDescription(tier, t.chat)
            return (
              <div key={tier}>
                <div className="flex items-center gap-1.5 px-2.5 pb-0.5 pt-1.5">
                  <span className={cn('size-1.5 rounded-full', tierVisual[tier].dot)} />
                  <span className="t-caption text-muted-foreground/60">{tier.toUpperCase()}</span>
                </div>
                <ModelMenuRow
                  active={selectedTier === tier}
                  description={description}
                  detail={`${modelTierLabel(tier, t.chat)} · ${description}`}
                  icon="bot"
                  label={modelNameLabel(option, t.chat.modelUnknown)}
                  onSelect={() => onChange(tier)}
                  tone={tier}
                />
              </div>
            )
          }) : (
            <DropdownMenuItem disabled className="w-full min-w-0 items-start rounded-none px-2.5 py-2">
              <span className="grid min-w-0 flex-1 text-left">
                <span className="truncate t-list text-foreground">{unavailableLabel}</span>
                <span className="truncate t-meta-sm text-muted-foreground">
                  {t.chat.modelServerDefault}
                </span>
              </span>
            </DropdownMenuItem>
          )}
        </div>

        <div className="flex items-center gap-2 border-t border-border bg-surface/40 px-2.5 py-1.5">
          <span className="t-caption text-muted-foreground/65">{t.chat.modelReasoningLabel}</span>
          <span className="ml-auto max-w-[12rem] truncate rounded-md border border-border bg-background px-2 py-0.5 t-meta-sm font-medium text-foreground">
            {modelEffortLabel(activeModel, t.chat)}
          </span>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function ModelMenuRow({
  active,
  description,
  detail,
  icon,
  label,
  onSelect,
  tone,
}: {
  active: boolean
  description: string
  detail: string
  icon: 'bot' | 'server'
  label: string
  onSelect: () => void
  tone?: ChatModelTier
}) {
  const Icon = icon === 'server' ? Server : Bot
  const visual = tone ? tierVisual[tone] : defaultVisual

  return (
    <DropdownMenuItem
      className={cn(
        'group relative flex w-full min-w-0 max-w-full items-center gap-2.5 rounded-none px-2.5 py-1.5 pr-1.5 text-left hover:bg-accent/50 focus:bg-accent/80 data-[highlighted]:bg-accent/80',
        active ? 'bg-accent' : 'hover:bg-accent/50',
      )}
      onSelect={onSelect}
    >
      <span
        className={cn(
          'absolute inset-y-1 left-0 w-0.5 rounded-full transition-opacity group-hover:opacity-100 group-focus:opacity-100 group-data-[highlighted]:opacity-100',
          visual.bar,
          active ? 'opacity-100' : 'opacity-0',
        )}
      />
      <Icon
        className={cn(
          'icon-md shrink-0 transition-colors',
          active ? visual.icon : ['text-muted-foreground/70', visual.iconHover],
        )}
      />
      <span className="min-w-0 flex-1 overflow-hidden">
        <span className="block truncate t-list text-foreground">{label}</span>
        <span className="block truncate t-meta-sm text-muted-foreground">{detail}</span>
      </span>
      <span
        aria-hidden="true"
        className="flex size-4 shrink-0 items-center justify-center rounded-full text-muted-foreground/45"
        title={description}
      >
        <Info className="size-3.5" />
      </span>
      <span className="flex size-4 shrink-0 items-center justify-center">
        {active ? <Check className={cn('size-3.5', visual.check)} /> : null}
      </span>
    </DropdownMenuItem>
  )
}

function modelOptionForTier(
  options: readonly ChatModelOption[],
  tier: ChatModelTier,
): ChatModelOption | null {
  return options.find((option) => option.tier === tier) ?? null
}

const defaultVisual = {
  bar: 'bg-foreground',
  check: 'text-foreground',
  icon: 'text-foreground',
  iconHover: 'group-hover:text-foreground group-focus:text-foreground group-data-[highlighted]:text-foreground',
}

const tierVisual: Record<ChatModelTier, typeof defaultVisual & { dot: string }> = {
  fast: {
    bar: 'bg-success',
    check: 'text-success',
    dot: 'bg-success',
    icon: 'text-success',
    iconHover: 'group-hover:text-success group-focus:text-success group-data-[highlighted]:text-success',
  },
  high: {
    bar: 'bg-brand',
    check: 'text-brand',
    dot: 'bg-brand',
    icon: 'text-brand',
    iconHover: 'group-hover:text-brand group-focus:text-brand group-data-[highlighted]:text-brand',
  },
  mid: {
    bar: 'bg-file',
    check: 'text-file',
    dot: 'bg-file',
    icon: 'text-file',
    iconHover: 'group-hover:text-file group-focus:text-file group-data-[highlighted]:text-file',
  },
}
