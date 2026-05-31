import {
  AlertTriangle,
  BookOpen,
  Database,
  ExternalLink,
  Github,
  Monitor,
  Moon,
  Palette,
  Scale,
  Server,
  Settings,
  Sun,
  type LucideIcon,
} from '@/components/icons'
import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { useLocale } from '@/i18n/LocaleProvider'
import type { InqtrixHealth, StackDiscoveryStatus } from '@/features/researchRuns/types'
import { cn } from '@/lib/utils'
import { appMotion } from '@/motion/transitions'
import { useTheme, type ThemeMode, type ThemePreset } from '@/theme/ThemeProvider'

type SettingsWorkspaceProps = {
  apiError: string | null
  apiHealth: InqtrixHealth | null
  apiKey: string
  isDemoMode: boolean
  onApiKeyChange: (apiKey: string) => void
  onDemoModeChange: (enabled: boolean) => void
  onStackChange: (stack: string) => void
  reduceMotion: boolean | null
  selectedStack: string
  stackDiscoveryStatus: StackDiscoveryStatus
  stackOptions: string[]
}

export default function SettingsWorkspace({
  apiError,
  apiHealth,
  apiKey,
  isDemoMode,
  onApiKeyChange,
  onDemoModeChange,
  onStackChange,
  reduceMotion,
  selectedStack,
  stackDiscoveryStatus,
  stackOptions,
}: SettingsWorkspaceProps) {
  const { contrastMode, preset, setContrastMode, setPreset, setTheme, theme } = useTheme()
  const { t } = useLocale()
  const modeOptions: Array<{
    icon: LucideIcon
    label: string
    value: ThemeMode
  }> = [
    { icon: Sun, label: t.common.light, value: 'light' },
    { icon: Moon, label: t.common.dark, value: 'dark' },
    { icon: Monitor, label: t.common.system, value: 'system' },
  ]
  const presetOptions: Array<{
    accent: string
    description: string
    label: string
    surface: string
    value: ThemePreset
  }> = [
    {
      accent: 'oklch(0.5 0.18 262)',
      description: t.settings.standardDescription,
      label: t.settings.standard,
      surface: 'oklch(0.971 0.006 255)',
      value: 'standard',
    },
    {
      accent: 'oklch(0.48 0.14 245)',
      description: t.settings.slateDescription,
      label: t.settings.slate,
      surface: 'oklch(0.972 0.008 245)',
      value: 'slate',
    },
    {
      accent: 'oklch(0.42 0.025 260)',
      description: t.settings.graphiteDescription,
      label: t.settings.graphite,
      surface: 'oklch(0.968 0.002 260)',
      value: 'graphite',
    },
    {
      accent: 'oklch(0.48 0.11 155)',
      description: t.settings.sageDescription,
      label: t.settings.sage,
      surface: 'oklch(0.973 0.008 150)',
      value: 'sage',
    },
  ]

  const apiBaseUrl = import.meta.env.VITE_INQTRIX_API_BASE_URL || t.settings.sameOriginApi
  const legal = apiHealth?.legal
  const hasMultiStackSelection = stackDiscoveryStatus === 'available' && stackOptions.length > 1
  const projectSourceUrl = legal?.source_url ?? t.authLock.repositoryUrl
  const stackModeLabel = stackDiscoveryStatus === 'available'
    ? t.settings.multiStackServer
    : stackDiscoveryStatus === 'unsupported'
      ? t.settings.singleStackServer
      : t.settings.stackDiscoveryPending

  return (
    <div className="flex min-h-0 w-full px-4 py-4 md:px-5 lg:h-full xl:px-8">
      <motion.section
        initial={reduceMotion ? false : { opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={appMotion.panel}
        className="mx-auto flex min-h-[calc(100svh-var(--header-h)-2rem)] w-full max-w-5xl flex-col overflow-hidden rounded-lg border border-border bg-card shadow-[0_1px_2px_var(--shadow-hairline)] lg:h-full lg:min-h-0"
      >
        <div className="flex shrink-0 items-center justify-between gap-4 border-b border-border px-5 py-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Settings className="size-4 text-muted-foreground" />
              <h1 className="text-sm font-semibold text-foreground">{t.settings.title}</h1>
            </div>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">
              {t.settings.workspaceDescription}
            </p>
          </div>
          <span
            className={cn(
              'inline-flex h-7 shrink-0 items-center rounded-md border px-2 text-xs font-semibold',
              isDemoMode
                ? 'border-brand/25 bg-brand-subtle text-brand'
                : 'border-border bg-background text-muted-foreground',
            )}
          >
            {isDemoMode ? t.common.demoMode : t.settings.localWorkspace}
          </span>
        </div>
        <div className="min-h-0 flex-1 divide-y divide-border overflow-y-auto overscroll-contain [scrollbar-gutter:stable] [scrollbar-width:thin]">
          <SettingsGroup
            description={t.settings.workspaceDescription}
            icon={Database}
            title={t.settings.workspace}
          >
            <div className="rounded-lg border border-border bg-background/70">
              <div className="flex items-start justify-between gap-4 p-4">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-foreground">
                    {t.settings.demoMode}
                  </p>
                  <p className="mt-1 max-w-xl text-xs leading-5 text-muted-foreground">
                    {t.settings.demoModeDescription}
                  </p>
                </div>
                <Switch
                  aria-describedby="demo-mode-warning"
                  aria-label={t.settings.demoMode}
                  checked={isDemoMode}
                  onCheckedChange={onDemoModeChange}
                />
              </div>
              <div
                className="flex gap-2 border-t border-border/70 px-4 py-3 text-xs leading-5 text-muted-foreground"
                id="demo-mode-warning"
              >
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-warning" />
                <p className="min-w-0">{t.settings.demoModeWarning}</p>
              </div>
            </div>
          </SettingsGroup>

          <SettingsGroup
            description={t.settings.stackDescription}
            icon={Server}
            title={t.common.stack}
          >
            <div className="grid gap-4">
              <SettingsRow label={t.settings.apiConnection}>
                <div className="rounded-lg border border-border bg-background p-3 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={cn(
                        'inline-flex h-6 items-center rounded-md border px-2 text-xs font-semibold',
                        apiHealth?.status === 'ok'
                          ? 'border-success/20 bg-success-subtle text-success'
                          : 'border-border bg-muted text-muted-foreground',
                      )}
                    >
                      {apiHealth?.status ?? t.settings.notConnected}
                    </span>
                    {apiHealth?.auth_required && (
                      <span className="inline-flex h-6 items-center rounded-md border border-warning/25 bg-warning/10 px-2 text-xs font-semibold text-warning">
                        {t.settings.authRequired}
                      </span>
                    )}
                    {apiHealth && (
                      <span className="inline-flex h-6 items-center rounded-md border border-border bg-muted px-2 text-xs font-semibold text-muted-foreground">
                        {stackModeLabel}
                      </span>
                    )}
                  </div>
                  <p className="mt-2 break-words text-xs leading-5 text-muted-foreground">
                    {apiBaseUrl}
                  </p>
                  {apiHealth && (
                    <div className="mt-2 grid gap-1 text-xs leading-5 text-muted-foreground sm:grid-cols-2">
                      <span className="min-w-0 truncate">
                        {t.settings.llmProvider}: <strong className="font-semibold text-foreground">{apiHealth.llm.provider}</strong>
                      </span>
                      <span className="min-w-0 truncate">
                        {t.settings.searchProvider}: <strong className="font-semibold text-foreground">{apiHealth.search.provider}</strong>
                      </span>
                      {apiHealth.reasoning_model && (
                        <span className="min-w-0 truncate sm:col-span-2">
                          {t.settings.reasoningModel}: <strong className="font-semibold text-foreground">{apiHealth.reasoning_model}</strong>
                        </span>
                      )}
                    </div>
                  )}
                  {apiError && (
                    <p className="mt-2 text-xs leading-5 text-destructive">
                      {apiError}
                    </p>
                  )}
                </div>
              </SettingsRow>
              {(apiHealth?.auth_required || apiKey) && (
                <SettingsRow label={t.settings.apiToken}>
                  <input
                    aria-label={t.settings.apiToken}
                    autoComplete="off"
                    className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm text-foreground shadow-[0_1px_2px_var(--shadow-hairline)] outline-none transition focus-visible:ring-2 focus-visible:ring-ring"
                    onChange={(event) => onApiKeyChange(event.target.value)}
                    placeholder={t.settings.apiTokenPlaceholder}
                    type="password"
                    value={apiKey}
                  />
                </SettingsRow>
              )}
              <SettingsRow label={t.settings.currentStack}>
                {hasMultiStackSelection ? (
                  <div className="flex flex-wrap gap-2">
                    {stackOptions.map((stack) => {
                      const isActive = selectedStack === stack
                      return (
                        <Button
                          aria-pressed={isActive}
                          className="h-9 max-w-full"
                          key={stack}
                          onClick={() => onStackChange(stack)}
                          type="button"
                          variant={isActive ? 'default' : 'outline'}
                        >
                          <span className="max-w-64 truncate">{stack}</span>
                        </Button>
                      )
                    })}
                  </div>
                ) : (
                  <div className="max-w-full rounded-lg border border-border bg-background px-3 py-2">
                    <div className="flex min-w-0 flex-wrap items-center gap-2">
                      <span className="inline-flex h-6 shrink-0 items-center rounded-md border border-border bg-muted px-2 text-xs font-semibold text-muted-foreground">
                        {stackModeLabel}
                      </span>
                      <span className="min-w-0 truncate text-sm font-semibold text-foreground">
                        {selectedStack}
                      </span>
                    </div>
                    {stackDiscoveryStatus === 'unsupported' && (
                      <p className="mt-1 text-xs leading-5 text-muted-foreground">
                        {t.settings.singleStackDescription}
                      </p>
                    )}
                  </div>
                )}
              </SettingsRow>
            </div>
          </SettingsGroup>

          <SettingsGroup
            description={t.settings.visualDesignDescription}
            icon={Palette}
            title={t.settings.visualDesign}
          >
            <div className="grid gap-5">
              <SettingsRow label={t.settings.mode}>
                <div className="flex flex-wrap gap-2">
                  {modeOptions.map((option) => {
                    const Icon = option.icon
                    const isActive = theme === option.value
                    return (
                      <Button
                        aria-pressed={isActive}
                        className="h-9 gap-2"
                        key={option.value}
                        onClick={() => setTheme(option.value)}
                        type="button"
                        variant={isActive ? 'default' : 'outline'}
                      >
                        <Icon className="size-4" />
                        <span>{option.label}</span>
                      </Button>
                    )
                  })}
                </div>
              </SettingsRow>

              <SettingsRow label={t.settings.theme}>
                <div className="grid gap-2 sm:grid-cols-2">
                  {presetOptions.map((option) => {
                    const isActive = preset === option.value
                    return (
                      <button
                        aria-pressed={isActive}
                        className={cn(
                          'rounded-lg border border-border bg-background p-3 text-left shadow-[0_1px_2px_var(--shadow-hairline)] transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                          isActive && 'border-brand bg-brand-subtle',
                        )}
                        key={option.value}
                        onClick={() => setPreset(option.value)}
                        type="button"
                      >
                        <span className="flex items-center gap-2">
                          <span
                            className="size-4 rounded-full border border-border"
                            style={{ background: option.accent }}
                          />
                          <span
                            className="size-4 rounded-full border border-border"
                            style={{ background: option.surface }}
                          />
                          <span className="ml-auto size-3 rounded-full border border-border bg-background" />
                        </span>
                        <span className="mt-3 block text-sm font-semibold text-foreground">
                          {option.label}
                        </span>
                        <span className="mt-1 block text-xs leading-5 text-muted-foreground">
                          {option.description}
                        </span>
                      </button>
                    )
                  })}
                </div>
              </SettingsRow>

              <SettingsRow label={t.settings.highContrast}>
                <div className="rounded-lg border border-border bg-background p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-foreground">
                        {t.settings.highContrast}
                      </p>
                      <p
                        className="mt-1 max-w-xl text-xs leading-5 text-muted-foreground"
                        id="high-contrast-description"
                      >
                        {t.settings.highContrastDescription}
                      </p>
                    </div>
                    <Switch
                      aria-describedby="high-contrast-description"
                      aria-label={t.settings.highContrast}
                      checked={contrastMode === 'high'}
                      onCheckedChange={(checked) =>
                        setContrastMode(checked ? 'high' : 'standard')}
                    />
                  </div>
                </div>
              </SettingsRow>
            </div>
          </SettingsGroup>

          <SettingsGroup
            description={t.settings.licensingDescription}
            icon={Scale}
            title={t.settings.licensing}
          >
            <div className="rounded-lg border border-border bg-background/70 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-foreground">
                    {legal?.project ?? t.appName}
                  </p>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">
                    {legal?.copyright ?? t.authLock.copyright}
                  </p>
                </div>
                <span className="inline-flex h-7 shrink-0 items-center rounded-md border border-brand/25 bg-brand-subtle px-2 text-xs font-semibold text-brand">
                  {legal?.license ?? t.authLock.licenseLabel}
                </span>
              </div>
              <div className="mt-4 grid gap-2 text-xs leading-5 text-muted-foreground">
                <SettingsInlineValue
                  label={t.settings.projectSource}
                  value={projectSourceUrl}
                />
                <SettingsInlineValue
                  label={t.settings.legalNotice}
                  value={legal?.notice ?? t.settings.attributionNotice}
                />
                <SettingsInlineValue
                  label={t.settings.warranty}
                  value={legal?.warranty_notice ?? t.authLock.warrantyNotice}
                />
              </div>
              <p className="mt-3 text-xs leading-5 text-muted-foreground">
                {t.settings.commercialLicensing}
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <SettingsLink
                  href={projectSourceUrl}
                  icon={Github}
                  label={t.authLock.repositoryLabel}
                />
                <SettingsLink
                  href={t.authLock.documentationUrl}
                  icon={BookOpen}
                  label={t.authLock.documentationLabel}
                />
                <SettingsLink
                  href={t.authLock.licenseUrl}
                  icon={Scale}
                  label={t.authLock.licenseLabel}
                />
              </div>
            </div>
          </SettingsGroup>
        </div>
      </motion.section>
    </div>
  )
}

function SettingsGroup({
  children,
  description,
  icon: Icon,
  title,
}: {
  children: ReactNode
  description: string
  icon: LucideIcon
  title: string
}) {
  return (
    <section className="grid gap-4 px-4 py-5 md:grid-cols-[220px_minmax(0,1fr)] md:px-5">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <Icon className="size-4 text-muted-foreground" />
          <h2 className="text-sm font-semibold text-foreground">{title}</h2>
        </div>
        <p className="mt-1 max-w-sm text-xs leading-5 text-muted-foreground">
          {description}
        </p>
      </div>
      <div className="min-w-0">{children}</div>
    </section>
  )
}

function SettingsRow({
  children,
  label,
}: {
  children: ReactNode
  label: string
}) {
  return (
    <div className="grid gap-2 md:grid-cols-[150px_minmax(0,1fr)]">
      <p className="pt-2 text-xs font-semibold text-muted-foreground">{label}</p>
      <div className="min-w-0">{children}</div>
    </div>
  )
}

function SettingsInlineValue({
  label,
  value,
}: {
  label: string
  value: string
}) {
  return (
    <div className="grid gap-1 sm:grid-cols-[120px_minmax(0,1fr)]">
      <span className="font-semibold text-muted-foreground">{label}</span>
      <span className="min-w-0 break-words text-foreground">{value}</span>
    </div>
  )
}

function SettingsLink({
  href,
  icon: Icon,
  label,
}: {
  href: string
  icon: LucideIcon
  label: string
}) {
  return (
    <a
      className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-surface px-2 text-xs font-medium text-muted-foreground transition hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      href={href}
      rel="noreferrer"
      target="_blank"
    >
      <Icon className="size-3.5" />
      <span>{label}</span>
      <ExternalLink className="size-3" />
    </a>
  )
}
