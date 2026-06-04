import {
  AlertTriangle,
  BookOpen,
  Check,
  CircleUserRound,
  ExternalLink,
  Github,
  Monitor,
  Moon,
  Palette,
  Scale,
  Server,
  Settings,
  Shield,
  SlidersHorizontal,
  Sun,
  type LucideIcon,
} from '@/components/icons'
import { motion } from 'motion/react'
import { useState, type ReactNode } from 'react'
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

type SettingsSectionId =
  | 'appearance'
  | 'connection'
  | 'licensing'
  | 'preferences'
  | 'security'

type SettingsNavItem = {
  description: string
  icon: LucideIcon
  id: SettingsSectionId
  label: string
}

type SettingsNavGroup = {
  icon: LucideIcon
  id: 'account' | 'application'
  items: SettingsNavItem[]
  label: string
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
  const [activeSection, setActiveSection] = useState<SettingsSectionId>('preferences')
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
  const navGroups: SettingsNavGroup[] = [
    {
      icon: CircleUserRound,
      id: 'account',
      label: t.settings.account,
      items: [
        {
          description: t.settings.preferencesDescription,
          icon: SlidersHorizontal,
          id: 'preferences',
          label: t.settings.preferences,
        },
        {
          description: t.settings.securityDescription,
          icon: Shield,
          id: 'security',
          label: t.settings.security,
        },
      ],
    },
    {
      icon: Settings,
      id: 'application',
      label: t.settings.application,
      items: [
        {
          description: t.settings.appearanceDescription,
          icon: Palette,
          id: 'appearance',
          label: t.settings.appearance,
        },
        {
          description: t.settings.connectionDescription,
          icon: Server,
          id: 'connection',
          label: t.settings.connection,
        },
      ],
    },
  ]
  const standaloneItems: SettingsNavItem[] = [
    {
      description: t.settings.licensingDescription,
      icon: Scale,
      id: 'licensing',
      label: t.settings.licensing,
    },
  ]
  const activeItem =
    [...navGroups.flatMap((group) => group.items), ...standaloneItems].find(
      (item) => item.id === activeSection,
    ) ?? navGroups[0].items[0]

  return (
    <SettingsShell reduceMotion={reduceMotion}>
      <SettingsSidebar
        activeSection={activeSection}
        groups={navGroups}
        isDemoMode={isDemoMode}
        onSectionChange={setActiveSection}
        standaloneItems={standaloneItems}
      />
      <SettingsPanel
        activeItem={activeItem}
        apiBaseUrl={apiBaseUrl}
        apiError={apiError}
        apiHealth={apiHealth}
        apiKey={apiKey}
        contrastMode={contrastMode}
        hasMultiStackSelection={hasMultiStackSelection}
        isDemoMode={isDemoMode}
        legal={legal}
        modeOptions={modeOptions}
        onApiKeyChange={onApiKeyChange}
        onDemoModeChange={onDemoModeChange}
        onStackChange={onStackChange}
        preset={preset}
        presetOptions={presetOptions}
        projectSourceUrl={projectSourceUrl}
        selectedStack={selectedStack}
        setContrastMode={setContrastMode}
        setPreset={setPreset}
        setTheme={setTheme}
        stackDiscoveryStatus={stackDiscoveryStatus}
        stackModeLabel={stackModeLabel}
        stackOptions={stackOptions}
        theme={theme}
      />
    </SettingsShell>
  )
}

function SettingsShell({
  children,
  reduceMotion,
}: {
  children: ReactNode
  reduceMotion: boolean | null
}) {
  return (
    <div className="flex min-h-0 w-full bg-canvas lg:h-full">
      <motion.section
        animate={{ opacity: 1, y: 0 }}
        className="grid min-h-[calc(100svh-var(--header-h))] w-full grid-rows-[auto_minmax(0,1fr)] lg:h-full lg:min-h-0 lg:grid-cols-[240px_minmax(0,1fr)] lg:grid-rows-1"
        initial={reduceMotion ? false : { opacity: 0, y: 8 }}
        transition={appMotion.panel}
      >
        {children}
      </motion.section>
    </div>
  )
}

function SettingsSidebar({
  activeSection,
  groups,
  isDemoMode,
  onSectionChange,
  standaloneItems,
}: {
  activeSection: SettingsSectionId
  groups: SettingsNavGroup[]
  standaloneItems: SettingsNavItem[]
  isDemoMode: boolean
  onSectionChange: (section: SettingsSectionId) => void
}) {
  const { t } = useLocale()

  return (
    <aside className="min-w-0 border-b border-border bg-background/95 backdrop-blur lg:border-b-0 lg:border-r">
      <div className="flex items-start justify-between gap-4 px-4 py-4 md:px-5 lg:block lg:px-5 lg:pb-4 lg:pt-5">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Settings className="size-4 text-muted-foreground" />
            <h1 className="text-base font-semibold leading-6 text-foreground">{t.settings.title}</h1>
          </div>
          <p className="mt-1 hidden max-w-48 text-xs leading-5 text-muted-foreground lg:block">
            {t.settings.settingsDescription}
          </p>
        </div>
        <StatusBadge
          className="lg:mt-4"
          label={isDemoMode ? t.common.demoMode : t.settings.localWorkspace}
          tone={isDemoMode ? 'brand' : 'neutral'}
        />
      </div>
      <nav
        aria-label={t.settings.sectionsLabel}
        className="flex gap-2 overflow-x-auto px-4 pb-3 [scrollbar-width:none] md:px-5 lg:block lg:space-y-4 lg:overflow-visible lg:px-3 lg:pb-5 [&::-webkit-scrollbar]:hidden"
      >
        {groups.map((group) => {
          const GroupIcon = group.icon

          return (
            <div className="flex shrink-0 items-center gap-1.5 lg:block" key={group.id}>
              <p className="hidden h-8 items-center gap-2 rounded-md px-2 text-sm font-semibold text-foreground lg:flex">
                <GroupIcon className="size-4 text-muted-foreground" />
                {group.label}
              </p>
              <div className="flex gap-1.5 lg:ml-6 lg:flex-col lg:gap-1">
                {group.items.map((item) => (
                  <SettingsNavButton
                    item={item}
                    isActive={activeSection === item.id}
                    key={item.id}
                    onClick={() => onSectionChange(item.id)}
                  />
                ))}
              </div>
            </div>
          )
        })}
        {standaloneItems.length > 0 ? (
          <div className="flex shrink-0 items-center gap-1.5 lg:block">
            <div className="flex gap-1.5 lg:flex-col lg:gap-1">
              {standaloneItems.map((item) => (
                <SettingsNavButton
                  item={item}
                  isActive={activeSection === item.id}
                  key={item.id}
                  onClick={() => onSectionChange(item.id)}
                />
              ))}
            </div>
          </div>
        ) : null}
      </nav>
    </aside>
  )
}

function SettingsNavButton({
  isActive,
  item,
  onClick,
}: {
  isActive: boolean
  item: SettingsNavItem
  onClick: () => void
}) {
  const Icon = item.icon

  return (
    <button
      aria-current={isActive ? 'page' : undefined}
      className={cn(
        'relative flex h-9 shrink-0 items-center gap-2 rounded-md border border-transparent px-3 text-sm font-medium text-muted-foreground transition hover:border-border hover:bg-card hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring lg:h-8 lg:w-full lg:justify-start lg:px-2',
        isActive && 'border-brand/20 bg-brand-subtle text-brand shadow-[0_1px_2px_var(--shadow-hairline)] hover:border-brand/20 hover:bg-brand-subtle hover:text-brand',
      )}
      onClick={onClick}
      title={item.description}
      type="button"
    >
      <span
        aria-hidden
        className={cn(
          'absolute left-0 top-1/2 hidden h-4 w-0.5 -translate-y-1/2 rounded-full bg-transparent lg:block',
          isActive && 'bg-brand',
        )}
      />
      <Icon className="size-4 shrink-0" />
      <span className="whitespace-nowrap">{item.label}</span>
    </button>
  )
}

function SettingsPanel({
  activeItem,
  apiBaseUrl,
  apiError,
  apiHealth,
  apiKey,
  contrastMode,
  hasMultiStackSelection,
  isDemoMode,
  legal,
  modeOptions,
  onApiKeyChange,
  onDemoModeChange,
  onStackChange,
  preset,
  presetOptions,
  projectSourceUrl,
  selectedStack,
  setContrastMode,
  setPreset,
  setTheme,
  stackDiscoveryStatus,
  stackModeLabel,
  stackOptions,
  theme,
}: {
  activeItem: SettingsNavItem
  apiBaseUrl: string
  apiError: string | null
  apiHealth: InqtrixHealth | null
  apiKey: string
  contrastMode: 'high' | 'standard'
  hasMultiStackSelection: boolean
  isDemoMode: boolean
  legal: InqtrixHealth['legal'] | undefined
  modeOptions: Array<{
    icon: LucideIcon
    label: string
    value: ThemeMode
  }>
  onApiKeyChange: (apiKey: string) => void
  onDemoModeChange: (enabled: boolean) => void
  onStackChange: (stack: string) => void
  preset: ThemePreset
  presetOptions: Array<{
    accent: string
    description: string
    label: string
    surface: string
    value: ThemePreset
  }>
  projectSourceUrl: string
  selectedStack: string
  setContrastMode: (mode: 'high' | 'standard') => void
  setPreset: (preset: ThemePreset) => void
  setTheme: (theme: ThemeMode) => void
  stackDiscoveryStatus: StackDiscoveryStatus
  stackModeLabel: string
  stackOptions: string[]
  theme: ThemeMode
}) {
  return (
    <main className="min-h-0 overflow-y-auto overscroll-contain px-4 py-5 [scrollbar-gutter:stable] [scrollbar-width:thin] md:px-6 lg:px-8 xl:px-10">
      <div className="flex max-w-[920px] flex-col gap-6 pb-8">
        <SettingsPanelHeader item={activeItem} />
        {activeItem.id === 'preferences' ? (
          <PreferencesPanel
            isDemoMode={isDemoMode}
            onDemoModeChange={onDemoModeChange}
          />
        ) : activeItem.id === 'security' ? (
          <SecurityPanel
            apiHealth={apiHealth}
            apiKey={apiKey}
            onApiKeyChange={onApiKeyChange}
          />
        ) : activeItem.id === 'appearance' ? (
          <AppearancePanel
            contrastMode={contrastMode}
            modeOptions={modeOptions}
            preset={preset}
            presetOptions={presetOptions}
            setContrastMode={setContrastMode}
            setPreset={setPreset}
            setTheme={setTheme}
            theme={theme}
          />
        ) : activeItem.id === 'connection' ? (
          <ConnectionPanel
            apiBaseUrl={apiBaseUrl}
            apiError={apiError}
            apiHealth={apiHealth}
            hasMultiStackSelection={hasMultiStackSelection}
            onStackChange={onStackChange}
            selectedStack={selectedStack}
            stackDiscoveryStatus={stackDiscoveryStatus}
            stackModeLabel={stackModeLabel}
            stackOptions={stackOptions}
          />
        ) : (
          <LicensingPanel
            legal={legal}
            projectSourceUrl={projectSourceUrl}
          />
        )}
      </div>
    </main>
  )
}

function SettingsPanelHeader({ item }: { item: SettingsNavItem }) {
  return (
    <header className="min-w-0 border-b border-border pb-4">
      <h2 className="text-xl font-semibold leading-8 text-foreground">{item.label}</h2>
      <p className="mt-1 max-w-2xl text-sm leading-6 text-muted-foreground">
        {item.description}
      </p>
    </header>
  )
}

function PreferencesPanel({
  isDemoMode,
  onDemoModeChange,
}: {
  isDemoMode: boolean
  onDemoModeChange: (enabled: boolean) => void
}) {
  const { locale, setLocale, t } = useLocale()

  return (
    <SettingsSection>
      <div className="flex flex-col gap-3 px-4 py-3.5 sm:flex-row sm:items-start sm:justify-between sm:gap-6">
        <div className="min-w-0 sm:flex-1">
          <h4 className="text-sm font-medium text-foreground">{t.settings.demoMode}</h4>
          <p className="mt-0.5 text-xs leading-5 text-muted-foreground">{t.settings.demoModeDescription}</p>
          <div
            className="mt-2 flex gap-2 rounded-md border border-warning/25 bg-warning-subtle/35 p-2.5 text-xs leading-5 text-foreground"
            id="demo-mode-warning"
          >
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-warning" />
            <p className="min-w-0">{t.settings.demoModeWarning}</p>
          </div>
        </div>
        <div className="shrink-0 sm:mt-0.5">
          <Switch
            aria-describedby="demo-mode-warning"
            aria-label={t.settings.demoMode}
            checked={isDemoMode}
            onCheckedChange={onDemoModeChange}
          />
        </div>
      </div>
      <SettingsRow
        description={t.settings.languageDescription}
        title={t.common.language}
      >
        <SettingsSegmented
          ariaLabel={t.common.language}
          onChange={setLocale}
          options={[
            { label: 'DE', value: 'de' },
            { label: 'EN', value: 'en' },
          ]}
          value={locale}
        />
      </SettingsRow>
      <SettingsRow
        description={t.settings.workspaceStateDescription}
        title={t.settings.workspaceState}
      >
        <StatusBadge
          label={isDemoMode ? t.common.demoMode : t.settings.localWorkspace}
          tone={isDemoMode ? 'brand' : 'neutral'}
        />
      </SettingsRow>
    </SettingsSection>
  )
}

function SecurityPanel({
  apiHealth,
  apiKey,
  onApiKeyChange,
}: {
  apiHealth: InqtrixHealth | null
  apiKey: string
  onApiKeyChange: (apiKey: string) => void
}) {
  const { t } = useLocale()
  const shouldShowTokenInput = apiHealth?.auth_required || apiKey

  return (
    <SettingsSection>
      {shouldShowTokenInput ? (
        <SettingsRow
          description={t.settings.runtimeTokenDescription}
          title={t.settings.apiToken}
        >
          <input
            aria-label={t.settings.apiToken}
            autoComplete="off"
            className="h-9 w-full rounded-md border border-border bg-background px-3 text-left text-sm text-foreground shadow-[0_1px_2px_var(--shadow-hairline)] outline-none transition focus-visible:ring-2 focus-visible:ring-ring sm:w-72"
            onChange={(event) => onApiKeyChange(event.target.value)}
            placeholder={t.settings.apiTokenPlaceholder}
            type="password"
            value={apiKey}
          />
        </SettingsRow>
      ) : (
        <SettingsRow
          description={t.settings.noBearerRequiredDescription}
          title={t.settings.noBearerRequired}
        >
          <StatusBadge label={t.settings.notConnected} tone="neutral" />
        </SettingsRow>
      )}
    </SettingsSection>
  )
}

function AppearancePanel({
  contrastMode,
  modeOptions,
  preset,
  presetOptions,
  setContrastMode,
  setPreset,
  setTheme,
  theme,
}: {
  contrastMode: 'high' | 'standard'
  modeOptions: Array<{
    icon: LucideIcon
    label: string
    value: ThemeMode
  }>
  preset: ThemePreset
  presetOptions: Array<{
    accent: string
    description: string
    label: string
    surface: string
    value: ThemePreset
  }>
  setContrastMode: (mode: 'high' | 'standard') => void
  setPreset: (preset: ThemePreset) => void
  setTheme: (theme: ThemeMode) => void
  theme: ThemeMode
}) {
  const { t } = useLocale()

  return (
    <SettingsSection>
      <SettingsRow description={t.settings.modeDescription} title={t.settings.mode}>
        <SettingsSegmented
          ariaLabel={t.settings.mode}
          onChange={setTheme}
          options={modeOptions}
          value={theme}
        />
      </SettingsRow>
      <SettingsRowBlock description={t.settings.themeDescription} title={t.settings.theme}>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {presetOptions.map((option) => {
            const isActive = preset === option.value

            return (
              <button
                aria-pressed={isActive}
                className={cn(
                  'group relative rounded-md border border-border bg-background p-2.5 text-left transition-colors hover:border-brand/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  isActive && 'border-brand ring-2 ring-brand',
                )}
                key={option.value}
                onClick={() => setPreset(option.value)}
                type="button"
              >
                {isActive ? (
                  <span className="absolute right-2 top-2 inline-flex size-4 items-center justify-center rounded-full bg-brand text-brand-foreground">
                    <Check className="size-3" />
                  </span>
                ) : null}
                <span
                  aria-hidden
                  className="flex h-9 items-center gap-1.5 overflow-hidden rounded-[5px] border border-border px-2"
                  style={{ background: option.surface }}
                >
                  <span className="h-2.5 w-10 rounded-full" style={{ background: option.accent }} />
                  <span className="h-2.5 w-5 rounded-full opacity-40" style={{ background: option.accent }} />
                </span>
                <span className="mt-2 block text-sm font-medium text-foreground">{option.label}</span>
                <span className="mt-0.5 block text-xs leading-5 text-muted-foreground">{option.description}</span>
              </button>
            )
          })}
        </div>
      </SettingsRowBlock>
      <SettingsRow
        description={t.settings.highContrastDescription}
        descriptionId="high-contrast-description"
        title={t.settings.highContrast}
      >
        <Switch
          aria-describedby="high-contrast-description"
          aria-label={t.settings.highContrast}
          checked={contrastMode === 'high'}
          onCheckedChange={(checked) => setContrastMode(checked ? 'high' : 'standard')}
        />
      </SettingsRow>
    </SettingsSection>
  )
}

function ConnectionPanel({
  apiBaseUrl,
  apiError,
  apiHealth,
  hasMultiStackSelection,
  onStackChange,
  selectedStack,
  stackDiscoveryStatus,
  stackModeLabel,
  stackOptions,
}: {
  apiBaseUrl: string
  apiError: string | null
  apiHealth: InqtrixHealth | null
  hasMultiStackSelection: boolean
  onStackChange: (stack: string) => void
  selectedStack: string
  stackDiscoveryStatus: StackDiscoveryStatus
  stackModeLabel: string
  stackOptions: string[]
}) {
  const { t } = useLocale()
  const modelRows = apiHealth
    ? [
        { label: t.settings.classifyModel, value: apiHealth.classify_model },
        { label: t.settings.evaluateModel, value: apiHealth.evaluate_model },
        { label: t.settings.searchModel, value: apiHealth.search_model },
        { label: t.settings.summarizeModel, value: apiHealth.summarize_model },
        { label: t.settings.reasoningModel, value: apiHealth.reasoning_model },
      ].filter((row) => Boolean(row.value))
    : []
  const hasModelInfo =
    modelRows.length > 0
    || Boolean(apiHealth?.report_profile)
    || Boolean(apiHealth?.model_tier)
    || Boolean(apiHealth?.testing_mode)

  return (
    <>
      <SettingsSection
        description={t.settings.apiStatusDescription}
        title={t.settings.apiConnection}
      >
        <div className="grid gap-2 px-4 py-3.5">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge
              label={apiHealth?.status ?? t.settings.notConnected}
              tone={apiHealth?.status === 'ok' ? 'success' : 'neutral'}
            />
            {apiHealth?.auth_required ? (
              <StatusBadge label={t.settings.authRequired} tone="warning" />
            ) : null}
            {apiHealth ? <StatusBadge label={stackModeLabel} tone="neutral" /> : null}
          </div>
          <p className="break-words text-xs leading-5 text-muted-foreground">
            <span className="font-medium text-foreground">{t.settings.baseUrl}:</span> {apiBaseUrl}
          </p>
          {apiError ? (
            <p className="text-xs leading-5 text-destructive">{apiError}</p>
          ) : null}
        </div>
      </SettingsSection>
      {apiHealth ? (
        <SettingsSection
          description={t.settings.providerMetadataDescription}
          title={t.settings.providerMetadata}
        >
          <div className="grid gap-1 px-4 py-3.5 text-xs leading-5 text-muted-foreground sm:grid-cols-2">
            <span className="min-w-0 truncate">
              {t.settings.llmProvider}: <strong className="font-semibold text-foreground">{apiHealth.llm.provider}</strong>
            </span>
            <span className="min-w-0 truncate">
              {t.settings.searchProvider}: <strong className="font-semibold text-foreground">{apiHealth.search.provider}</strong>
            </span>
          </div>
          <SettingsRow description={t.settings.stackDescription} title={t.settings.currentStack}>
            {hasMultiStackSelection ? (
              <SettingsSegmented
                ariaLabel={t.common.stack}
                onChange={onStackChange}
                options={stackOptions.map((stack) => ({ label: stack, value: stack }))}
                value={selectedStack}
                wrap
              />
            ) : (
              <div className="grid gap-1">
                <div className="flex min-w-0 flex-wrap items-center gap-2 sm:justify-end">
                  <StatusBadge label={stackModeLabel} tone="neutral" />
                  <span className="min-w-0 truncate text-sm font-semibold text-foreground">
                    {selectedStack}
                  </span>
                </div>
                {stackDiscoveryStatus === 'unsupported' ? (
                  <p className="text-xs leading-5 text-muted-foreground sm:text-right">
                    {t.settings.singleStackDescription}
                  </p>
                ) : null}
              </div>
            )}
          </SettingsRow>
        </SettingsSection>
      ) : null}
      {apiHealth && hasModelInfo ? (
        <SettingsSection
          description={t.settings.activeModelsDescription}
          title={t.settings.activeModels}
        >
          <div className="grid gap-x-6 gap-y-1.5 px-4 py-3.5 text-xs leading-5 sm:grid-cols-2">
            {modelRows.map((row) => (
              <span className="min-w-0 truncate text-muted-foreground" key={row.label}>
                {row.label}: <strong className="font-semibold text-foreground">{row.value}</strong>
              </span>
            ))}
            {apiHealth.report_profile || apiHealth.model_tier || apiHealth.testing_mode ? (
              <div className="flex flex-wrap items-center gap-2 pt-1 sm:col-span-2">
                {apiHealth.report_profile ? (
                  <span className="text-muted-foreground">
                    {t.settings.reportProfile}: <strong className="font-semibold text-foreground">{apiHealth.report_profile}</strong>
                  </span>
                ) : null}
                {apiHealth.model_tier ? (
                  <StatusBadge label={`${t.settings.modelTier}: ${apiHealth.model_tier}`} tone="neutral" />
                ) : null}
                {apiHealth.testing_mode ? (
                  <StatusBadge label={t.settings.testingMode} tone="warning" />
                ) : null}
              </div>
            ) : null}
          </div>
        </SettingsSection>
      ) : null}
    </>
  )
}

function LicensingPanel({
  legal,
  projectSourceUrl,
}: {
  legal: InqtrixHealth['legal'] | undefined
  projectSourceUrl: string
}) {
  const { t } = useLocale()

  return (
    <SettingsSection>
      <SettingsRow
        description={legal?.copyright ?? t.authLock.copyright}
        title={legal?.project ?? t.appName}
      >
        <StatusBadge
          label={legal?.license ?? t.authLock.licenseLabel}
          tone="brand"
        />
      </SettingsRow>
      <SettingsRow title={t.settings.projectSource}>
        <span className="block min-w-0 break-words text-xs leading-5 text-foreground sm:text-right">
          {projectSourceUrl}
        </span>
      </SettingsRow>
      <SettingsRow title={t.settings.legalNotice}>
        <span className="block min-w-0 break-words text-xs leading-5 text-foreground sm:text-right">
          {legal?.notice ?? t.settings.attributionNotice}
        </span>
      </SettingsRow>
      <SettingsRow title={t.settings.warranty}>
        <span className="block min-w-0 break-words text-xs leading-5 text-foreground sm:text-right">
          {legal?.warranty_notice ?? t.authLock.warrantyNotice}
        </span>
      </SettingsRow>
      <SettingsRow
        description={t.settings.commercialLicensing}
        title={t.settings.legalResources}
      >
        <div className="flex flex-wrap gap-2 sm:justify-end">
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
      </SettingsRow>
    </SettingsSection>
  )
}

function SettingsSection({
  children,
  description,
  title,
}: {
  children: ReactNode
  description?: string
  title?: string
}) {
  return (
    <section className="overflow-hidden rounded-lg border border-border bg-card shadow-[0_1px_2px_var(--shadow-hairline)]">
      {title ? (
        <div className="border-b border-border px-4 py-3">
          <h3 className="text-sm font-semibold text-foreground">{title}</h3>
          {description ? (
            <p className="mt-0.5 text-xs leading-5 text-muted-foreground">{description}</p>
          ) : null}
        </div>
      ) : null}
      <div className="divide-y divide-border">{children}</div>
    </section>
  )
}

function SettingsRow({
  children,
  description,
  descriptionId,
  title,
}: {
  children: ReactNode
  description?: string
  descriptionId?: string
  title: string
}) {
  return (
    <div className="flex flex-col gap-3 px-4 py-3.5 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="min-w-0 sm:flex-1">
        <h4 className="text-sm font-medium text-foreground">{title}</h4>
        {description ? (
          <p className="mt-0.5 text-xs leading-5 text-muted-foreground" id={descriptionId}>
            {description}
          </p>
        ) : null}
      </div>
      <div className="min-w-0 shrink-0 sm:max-w-[60%] sm:text-right">{children}</div>
    </div>
  )
}

function SettingsRowBlock({
  children,
  description,
  title,
}: {
  children: ReactNode
  description?: string
  title: string
}) {
  return (
    <div className="px-4 py-3.5">
      <h4 className="text-sm font-medium text-foreground">{title}</h4>
      {description ? (
        <p className="mt-0.5 text-xs leading-5 text-muted-foreground">{description}</p>
      ) : null}
      <div className="mt-3">{children}</div>
    </div>
  )
}

function SettingsSegmented<T extends string>({
  ariaLabel,
  onChange,
  options,
  value,
  wrap,
}: {
  ariaLabel: string
  onChange: (value: T) => void
  options: Array<{ icon?: LucideIcon; label: string; value: T }>
  value: T
  wrap?: boolean
}) {
  return (
    <div
      aria-label={ariaLabel}
      className={cn(
        'inline-flex h-9 items-center rounded-md border border-border bg-card p-0.5 shadow-[0_1px_2px_var(--shadow-hairline)]',
        wrap && 'h-auto flex-wrap gap-0.5',
      )}
      role="group"
    >
      {options.map((option) => {
        const Icon = option.icon
        const isActive = value === option.value

        return (
          <button
            aria-pressed={isActive}
            className={cn(
              'inline-flex h-8 items-center gap-1.5 rounded-[6px] px-2.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              isActive && 'bg-brand-subtle text-brand hover:bg-brand-subtle hover:text-brand',
            )}
            key={option.value}
            onClick={() => onChange(option.value)}
            type="button"
          >
            {Icon ? <Icon className="size-4 shrink-0" /> : null}
            <span className={cn(wrap && 'max-w-[12rem] truncate')}>{option.label}</span>
          </button>
        )
      })}
    </div>
  )
}

function StatusBadge({
  className,
  label,
  tone,
}: {
  className?: string
  label: string
  tone: 'brand' | 'neutral' | 'success' | 'warning'
}) {
  return (
    <span
      className={cn(
        'inline-flex h-7 shrink-0 items-center rounded-md border px-2 text-xs font-semibold',
        tone === 'brand' && 'border-brand/25 bg-brand-subtle text-brand',
        tone === 'neutral' && 'border-border bg-background text-muted-foreground',
        tone === 'success' && 'border-success/20 bg-success-subtle text-success',
        tone === 'warning' && 'border-warning/25 bg-warning/10 text-warning',
        className,
      )}
    >
      {label}
    </span>
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
      className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-background px-2 text-xs font-medium text-muted-foreground transition hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
