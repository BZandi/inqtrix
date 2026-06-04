import { Database, FileText, Globe2, MessagesSquare, Settings, type LucideIcon } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useLocale } from '@/i18n/LocaleProvider'
import { cn } from '@/lib/utils'
import type { AppView } from '../types'

type AppRailProps = {
  activeView: AppView
  onViewChange: (view: AppView) => void
}

export function AppRail({ activeView, onViewChange }: AppRailProps) {
  const { t } = useLocale()
  const primaryItems: Array<{
    icon: LucideIcon
    label: string
    value: AppView
  }> = [
    { icon: Globe2, label: t.navigation.research, value: 'research' },
    { icon: MessagesSquare, label: t.navigation.chat, value: 'chat' },
    { icon: FileText, label: t.navigation.editor, value: 'editor' },
  ]
  const settingsItem = {
    icon: Settings,
    label: t.navigation.settings,
    value: 'settings',
  } satisfies {
    icon: LucideIcon
    label: string
    value: AppView
  }
  const databaseItem = {
    icon: Database,
    label: t.navigation.database,
    value: 'database',
  } satisfies {
    icon: LucideIcon
    label: string
    value: AppView
  }

  return (
    <nav
      aria-label={t.navigation.label}
      className="sticky top-[var(--header-h)] z-20 flex h-[calc(100svh-var(--header-h))] w-12 shrink-0 flex-col items-center border-r border-border bg-background/90 px-1.5 py-3 backdrop-blur md:w-14 md:px-2"
    >
      <div className="flex flex-col gap-1">
        {primaryItems.map((item) => {
          const Icon = item.icon
          const isActive = activeView === item.value
          return (
            <Tooltip key={item.value}>
              <TooltipTrigger asChild>
                <Button
                  aria-label={item.label}
                  aria-pressed={isActive}
                  className={cn(
                    'size-9 rounded-md text-muted-foreground',
                    isActive && 'bg-brand-subtle text-brand shadow-none hover:bg-brand-subtle hover:text-brand',
                  )}
                  onClick={() => onViewChange(item.value)}
                  size="icon"
                  type="button"
                  variant={isActive ? 'default' : 'ghost'}
                >
                  <Icon className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">{item.label}</TooltipContent>
            </Tooltip>
          )
        })}
      </div>
      <div className="mt-auto flex flex-col items-center gap-1">
        <span aria-hidden className="mb-0.5 h-px w-5 rounded-full bg-border/70" />
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              aria-label={databaseItem.label}
              aria-pressed={activeView === databaseItem.value}
              className={cn(
                'size-9 rounded-md text-muted-foreground',
                activeView === databaseItem.value && 'bg-brand-subtle text-brand shadow-none hover:bg-brand-subtle hover:text-brand',
              )}
              onClick={() => onViewChange(databaseItem.value)}
              size="icon"
              type="button"
              variant={activeView === databaseItem.value ? 'default' : 'ghost'}
            >
              <Database className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="right">{databaseItem.label}</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              aria-label={settingsItem.label}
              aria-pressed={activeView === settingsItem.value}
              className={cn(
                'size-9 rounded-md text-muted-foreground',
                activeView === settingsItem.value && 'bg-brand-subtle text-brand shadow-none hover:bg-brand-subtle hover:text-brand',
              )}
              onClick={() => onViewChange(settingsItem.value)}
              size="icon"
              type="button"
              variant={activeView === settingsItem.value ? 'default' : 'ghost'}
            >
              <Settings className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="right">{settingsItem.label}</TooltipContent>
        </Tooltip>
      </div>
    </nav>
  )
}
