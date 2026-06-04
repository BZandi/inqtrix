import {
  Download,
  FolderOpen,
  Languages,
  LoaderCircle,
  Monitor,
  Moon,
  Save,
  Sun,
  X,
  type LucideIcon,
} from '@/components/icons'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { BrandMark } from '@/components/BrandMark'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { ProjectConnection } from '@/features/project/types'
import type { AppView } from '@/features/researchDesk/types'
import { useLocale } from '@/i18n/LocaleProvider'
import type { TranslationDictionary } from '@/i18n/translations'
import { useTheme, type ThemeMode } from '@/theme/ThemeProvider'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'

type TopbarProps = {
  activeView: AppView
  dirty: boolean
  isProjectActionPending: boolean
  onDismissProjectActionError: () => void
  onExportProject: () => void
  onLoadProject: () => void
  onSaveProject: () => void
  projectActionError: string | null
  projectConnection: ProjectConnection
  projectName: string
}

export function Topbar({
  activeView,
  dirty,
  isProjectActionPending,
  onDismissProjectActionError,
  onExportProject,
  onLoadProject,
  onSaveProject,
  projectActionError,
  projectConnection,
  projectName,
}: TopbarProps) {
  const { t } = useLocale()
  const reduceMotion = useReducedMotion()
  const status = projectStatus(projectConnection, dirty, t)
  const activeModeLabel = viewLabel(activeView, t)

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/95 backdrop-blur">
      <div className="flex min-h-[var(--header-h)] w-full flex-wrap items-center gap-3 py-1.5 pr-4 md:pr-5 xl:pr-8">
        <div className="flex min-w-0 items-center">
          <div className="flex w-12 shrink-0 items-center justify-center md:w-14">
            <BrandMark className="size-8 shrink-0" />
          </div>
          <span className="text-lg font-semibold tracking-normal text-brand">
            {t.appName}
          </span>
          <span aria-hidden className="mx-2 hidden h-4 w-px bg-border sm:block" />
          <motion.span
            aria-live="polite"
            className="hidden min-w-0 items-center overflow-hidden text-sm font-medium text-muted-foreground sm:inline-grid"
            layout={!reduceMotion}
            transition={appMotion.list}
          >
            <AnimatePresence initial={false} mode="sync">
              <motion.span
                animate={reduceMotion
                  ? { opacity: 1 }
                  : { opacity: 1, y: 0 }}
                className="col-start-1 row-start-1 inline-flex whitespace-nowrap"
                exit={reduceMotion
                  ? { opacity: 0 }
                  : { opacity: 0, y: -2 }}
                initial={reduceMotion
                  ? { opacity: 0 }
                  : { opacity: 0, y: 2 }}
                key={activeModeLabel}
                transition={{
                  duration: reduceMotion ? 0.01 : 0.065,
                  ease: [0.22, 1, 0.36, 1],
                }}
              >
                {activeModeLabel}
              </motion.span>
            </AnimatePresence>
          </motion.span>
        </div>

        <div className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-2">
          <div
            className={cn(
              'inline-flex h-8 max-w-64 items-center gap-1.5 rounded-md px-2 text-sm font-semibold text-foreground',
              status.tone === 'warning' && 'text-brand',
            )}
            title={projectName}
          >
            <span className={cn(
              'size-2 shrink-0 rounded-full bg-success shadow-[0_0_0_4px_var(--success-subtle)]',
              status.tone === 'warning' && 'bg-brand shadow-[0_0_0_4px_var(--brand-subtle)]',
            )} />
            <span className="truncate">{status.label}</span>
          </div>
          <div
            aria-label={`${t.topbar.loadProject}, ${t.topbar.exportProject}, ${t.topbar.saveProject}`}
            className="inline-flex h-9 items-center rounded-md border border-border bg-card p-0.5 shadow-[0_1px_2px_var(--shadow-hairline)]"
            role="group"
          >
            <ProjectActionButton
              disabled={isProjectActionPending}
              icon={FolderOpen}
              label={t.topbar.loadProject}
              onClick={onLoadProject}
            />
            <ProjectActionButton
              disabled={isProjectActionPending}
              icon={Download}
              label={t.topbar.exportProject}
              onClick={onExportProject}
            />
            <ProjectActionButton
              disabled={isProjectActionPending || (!dirty && projectConnection.kind !== 'directory')}
              icon={isProjectActionPending ? LoaderCircle : Save}
              label={t.topbar.saveProject}
              onClick={onSaveProject}
              spin={isProjectActionPending}
            />
          </div>
          <ThemeToggle />
          <LanguageToggle />
        </div>
      </div>
      {projectActionError ? (
        <div
          className="flex items-center gap-2 border-t border-border bg-destructive/10 px-4 py-2 text-sm text-destructive md:px-5 xl:px-8"
          role="alert"
        >
          <span className="min-w-0 flex-1 truncate" title={projectActionError}>
            {projectActionError}
          </span>
          <button
            aria-label={t.common.close}
            className="shrink-0 rounded-[6px] p-1 hover:bg-destructive/15"
            onClick={onDismissProjectActionError}
            type="button"
          >
            <X className="size-4" />
          </button>
        </div>
      ) : null}
    </header>
  )
}

function viewLabel(view: AppView, t: TranslationDictionary) {
  return t.navigation[view]
}

function ProjectActionButton({
  disabled = false,
  icon: Icon,
  label,
  onClick,
  spin = false,
}: {
  disabled?: boolean
  icon: LucideIcon
  label: string
  onClick: () => void
  spin?: boolean
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          aria-label={label}
          className="size-8 rounded-[6px] px-0 text-muted-foreground hover:bg-accent hover:text-accent-foreground"
          disabled={disabled}
          onClick={onClick}
          type="button"
          variant="ghost"
        >
          <Icon className={cn('size-4', spin && 'animate-spin')} />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function projectStatus(
  connection: ProjectConnection,
  dirty: boolean,
  t: TranslationDictionary,
) {
  if (dirty) return { label: t.topbar.unsavedProject, tone: 'warning' as const }
  if (connection.kind === 'directory') {
    return { label: t.topbar.connectedProject, tone: 'success' as const }
  }
  if (connection.kind === 'download') {
    return { label: t.topbar.downloadFallback, tone: 'warning' as const }
  }
  if (connection.kind === 'demo') {
    return { label: t.common.demoMode, tone: 'success' as const }
  }
  return { label: t.topbar.localProject, tone: 'success' as const }
}

function ThemeToggle() {
  const { setTheme, theme } = useTheme()
  const { t } = useLocale()

  const options: Array<{
    icon: LucideIcon
    label: string
    value: ThemeMode
  }> = [
    { icon: Sun, label: t.common.light, value: 'light' },
    { icon: Moon, label: t.common.dark, value: 'dark' },
    { icon: Monitor, label: t.common.system, value: 'system' },
  ]

  return (
    <div
      aria-label={t.common.theme}
      className="inline-flex h-9 items-center rounded-md border border-border bg-card p-0.5 shadow-[0_1px_2px_var(--shadow-hairline)]"
      role="group"
    >
      {options.map((option) => {
        const Icon = option.icon
        return (
          <button
            aria-label={option.label}
            className={cn(
              'inline-flex size-8 items-center justify-center rounded-[6px] text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground',
              theme === option.value && 'bg-brand-subtle text-brand hover:bg-brand-subtle hover:text-brand',
            )}
            key={option.value}
            onClick={() => setTheme(option.value)}
            type="button"
          >
            <Icon className="size-4" />
          </button>
        )
      })}
    </div>
  )
}

function LanguageToggle() {
  const { locale, setLocale, t } = useLocale()

  return (
    <div
      aria-label={t.common.language}
      className="inline-flex h-9 items-center rounded-md border border-border bg-card p-0.5 shadow-[0_1px_2px_var(--shadow-hairline)]"
      role="group"
    >
      {(['de', 'en'] as const).map((nextLocale) => (
        <button
          className={cn(
            'inline-flex h-8 min-w-9 items-center justify-center gap-1 rounded-[6px] px-2 text-xs font-semibold text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground',
            locale === nextLocale && 'bg-brand-subtle text-brand hover:bg-brand-subtle hover:text-brand',
          )}
          key={nextLocale}
          onClick={() => setLocale(nextLocale)}
          type="button"
        >
          <Languages className="size-3.5" />
          <span>{nextLocale.toUpperCase()}</span>
        </button>
      ))}
    </div>
  )
}
