import {
  Download,
  ExternalLink,
  FolderOpen,
  Github,
  LoaderCircle,
  Monitor,
  Moon,
  Save,
  Server,
  Sun,
  Upload,
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
import type { Locale, TranslationDictionary } from '@/i18n/translations'
import { useTheme, type ThemeMode } from '@/theme/ThemeProvider'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'

type TopbarProps = {
  activeView: AppView
  /** The server supports durable project persistence (M6): shows the
   * "move to server" import action. */
  canPersistProject: boolean
  dirty: boolean
  /** A server import is in flight (the import button shows a spinner). */
  importPending: boolean
  isProjectActionPending: boolean
  onDismissProjectActionError: () => void
  onExportProject: () => void
  /** Push this project's chat to the server and opt into ongoing sync. */
  onImportProjectToServer: () => void
  onLoadProject: () => void
  onSaveProject: () => void
  projectActionError: string | null
  projectConnection: ProjectConnection
  projectName: string
  /** This project is opted into the server-persistence tier. */
  serverSyncEnabled: boolean
  /** Last background chat-sync failure, surfaced as a status (never silent). */
  serverSyncError: string | null
}

export function Topbar({
  activeView,
  canPersistProject,
  dirty,
  importPending,
  isProjectActionPending,
  onDismissProjectActionError,
  onExportProject,
  onImportProjectToServer,
  onLoadProject,
  onSaveProject,
  projectActionError,
  projectConnection,
  projectName,
  serverSyncEnabled,
  serverSyncError,
}: TopbarProps) {
  const { t } = useLocale()
  const reduceMotion = useReducedMotion()
  const status = projectStatus(projectConnection, dirty, t)
  const activeModeLabel = viewLabel(activeView, t)

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/95 backdrop-blur">
      <div className="flex min-h-[var(--header-h)] w-full flex-wrap items-center gap-2 pr-3 md:pr-4 xl:pr-6">
        <div className="flex min-w-0 items-center">
          <div className="flex w-12 shrink-0 items-center justify-center md:w-14">
            <BrandMark className="size-7 shrink-0" />
          </div>
          <span className="text-base font-semibold tracking-normal text-brand">
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

        <div className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-1.5">
          {!serverSyncEnabled ? (
            <ProjectStatusBadge projectName={projectName} status={status} />
          ) : null}
          <div
            aria-label={t.topbar.projectActions}
            className="inline-flex h-8 items-center rounded-md border border-border bg-card p-0.5 shadow-[0_1px_2px_var(--shadow-hairline)]"
            role="group"
          >
            {serverSyncEnabled ? (
              // Server-synced: the server is the live source, so the local file
              // is only for import/export. Loading a file imports it UP (merges
              // additively), and one "export backup" download replaces the
              // redundant Export+Save pair (Save = write-to-local, meaningless
              // when the server auto-saves).
              <>
                <ProjectActionButton
                  disabled={isProjectActionPending}
                  icon={FolderOpen}
                  label={t.topbar.importFile}
                  onClick={onLoadProject}
                />
                <ProjectActionButton
                  disabled={isProjectActionPending}
                  icon={Download}
                  label={t.topbar.exportBackup}
                  onClick={onExportProject}
                />
              </>
            ) : (
              <>
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
                {canPersistProject ? (
                  <ProjectActionButton
                    disabled={isProjectActionPending || importPending}
                    icon={importPending ? LoaderCircle : Upload}
                    label={t.topbar.importToServer}
                    onClick={onImportProjectToServer}
                    spin={importPending}
                  />
                ) : null}
              </>
            )}
          </div>
          {serverSyncEnabled ? <ServerSyncBadge error={serverSyncError} projectName={projectName} t={t} /> : null}
          <ThemeToggle />
          <LanguageToggle />
          <RepoLink />
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
  if (view === 'prompt-library') return t.navigation.promptLibrary
  return t.navigation[view]
}

function ProjectStatusBadge({
  projectName,
  status,
}: {
  projectName: string
  status: { label: string; tone: 'success' | 'warning' }
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          aria-label={`${status.label}: ${projectName}`}
          className={cn(
            'inline-flex h-8 max-w-40 items-center gap-1.5 rounded-md px-2 t-meta-sm font-semibold text-foreground',
            status.tone === 'warning' && 'text-brand',
          )}
          role="status"
          tabIndex={0}
        >
          <span className={cn(
            'size-2 shrink-0 rounded-full bg-success shadow-[0_0_0_4px_var(--success-subtle)]',
            status.tone === 'warning' && 'bg-brand shadow-[0_0_0_4px_var(--brand-subtle)]',
          )} />
          <span className="truncate">{status.label}</span>
        </div>
      </TooltipTrigger>
      <TooltipContent className="max-w-72">
        <span className="block font-semibold">{projectName}</span>
        <span className="block opacity-80">{status.label}</span>
      </TooltipContent>
    </Tooltip>
  )
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
          className="size-7 rounded-[6px] px-0 text-muted-foreground hover:bg-accent hover:text-accent-foreground"
          disabled={disabled}
          onClick={onClick}
          type="button"
          variant="ghost"
        >
          <Icon className={cn('icon-sm', spin && 'animate-spin')} />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function ServerSyncBadge({
  error,
  projectName,
  t,
}: {
  error: string | null
  projectName: string
  t: TranslationDictionary
}) {
  const label = error ? t.topbar.serverSyncError : t.topbar.serverSyncedShort
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          aria-label={`${label}: ${projectName}`}
          className={cn(
            'inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-card px-2.5 t-meta-sm font-medium shadow-[0_1px_2px_var(--shadow-hairline)]',
            error ? 'text-destructive' : 'text-muted-foreground',
          )}
          role="status"
          tabIndex={0}
        >
          <Server
            className={cn('icon-sm shrink-0', error ? 'text-destructive' : 'text-success')}
          />
          <span className="hidden sm:inline">{label}</span>
        </div>
      </TooltipTrigger>
      <TooltipContent className="max-w-72">
        <span className="block font-semibold">{projectName}</span>
        <span className="block opacity-80">{error ?? t.topbar.serverSyncedHint}</span>
      </TooltipContent>
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
      className="inline-flex h-8 items-center rounded-md border border-border bg-card p-0.5 shadow-[0_1px_2px_var(--shadow-hairline)]"
      role="group"
    >
      {options.map((option) => {
        const Icon = option.icon
        return (
          <button
            aria-label={option.label}
            className={cn(
              'inline-flex size-7 items-center justify-center rounded-[6px] text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground',
              theme === option.value && 'bg-brand-subtle text-brand hover:bg-brand-subtle hover:text-brand',
            )}
            key={option.value}
            onClick={() => setTheme(option.value)}
            type="button"
          >
            <Icon className="icon-sm" />
          </button>
        )
      })}
    </div>
  )
}

function LanguageToggle() {
  const { locale, setLocale, t } = useLocale()
  const reduceMotion = useReducedMotion()

  const options: Array<{ label: string; value: Locale }> = [
    { label: 'DE', value: 'de' },
    { label: 'EN', value: 'en' },
  ]

  return (
    <div
      aria-label={t.common.language}
      className="inline-flex h-8 items-center rounded-md border border-border bg-card p-0.5 shadow-[0_1px_2px_var(--shadow-hairline)]"
      role="group"
    >
      {options.map((option) => {
        const isActive = locale === option.value

        return (
          <button
            aria-label={option.label}
            aria-pressed={isActive}
            className={cn(
              'relative inline-flex h-7 items-center justify-center rounded-[6px] px-2.5 text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              isActive ? 'text-brand' : 'text-muted-foreground hover:text-foreground',
            )}
            key={option.value}
            onClick={() => setLocale(option.value)}
            type="button"
          >
            {isActive ? (
              <motion.span
                aria-hidden
                className="absolute inset-0 rounded-[6px] bg-brand-subtle"
                layoutId="topbar-locale-active"
                transition={reduceMotion ? { duration: 0 } : appMotion.list}
              />
            ) : null}
            <span className="relative z-10">{option.label}</span>
          </button>
        )
      })}
    </div>
  )
}

function RepoLink() {
  const { t } = useLocale()

  return (
    <a
      className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-card px-2.5 text-xs font-medium text-muted-foreground shadow-[0_1px_2px_var(--shadow-hairline)] transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      href={t.authLock.repositoryUrl}
      rel="noreferrer"
      target="_blank"
      title={t.authLock.repositoryLabel}
    >
      <Github className="icon-sm shrink-0" />
      <span className="hidden sm:inline">{t.authLock.repositoryLabel}</span>
      <ExternalLink className="hidden size-3 shrink-0 sm:inline" />
    </a>
  )
}
