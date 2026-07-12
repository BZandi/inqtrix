import {
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

export type ThemeMode = 'light' | 'dark' | 'system'
export type ThemePreset = 'standard' | 'slate' | 'graphite' | 'sage'
export type ContrastMode = 'standard' | 'high'
export type UserBubbleTone = 'gray' | 'mint' | 'orange' | 'sky' | 'violet' | 'ink'

type ThemeContextValue = {
  // Long-term agent memory opt-in. NOT a visual preference, but it rides the
  // same synced account-preferences bag (ThemeProvider + locale) that
  // useAccountPreferences reads/writes; the server row is authoritative and
  // wins on login. localStorage is only the device cache/default (OFF).
  agentMemoryEnabled: boolean
  contrastMode: ContrastMode
  preset: ThemePreset
  resolvedTheme: 'light' | 'dark'
  setAgentMemoryEnabled: (enabled: boolean) => void
  setContrastMode: (mode: ContrastMode) => void
  setPreset: (preset: ThemePreset) => void
  setTheme: (theme: ThemeMode) => void
  setUserBubbleTone: (tone: UserBubbleTone) => void
  theme: ThemeMode
  userBubbleTone: UserBubbleTone
}

const THEME_STORAGE_KEY = 'inqtrix.researchDesk.theme'
const THEME_PRESET_STORAGE_KEY = 'inqtrix.researchDesk.themePreset'
const CONTRAST_MODE_STORAGE_KEY = 'inqtrix.researchDesk.contrastMode'
const USER_BUBBLE_TONE_STORAGE_KEY = 'inqtrix.researchDesk.userBubbleTone'
const AGENT_MEMORY_ENABLED_STORAGE_KEY = 'inqtrix.researchDesk.agentMemoryEnabled'
const THEME_TRANSITION_SUPPRESSION_ATTR = 'data-theme-transition-suppressed'
const ThemeContext = createContext<ThemeContextValue | null>(null)

type ThemeProviderProps = {
  children: ReactNode
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [theme, setThemeState] = useState<ThemeMode>(() => readStoredTheme())
  const [preset, setPresetState] = useState<ThemePreset>(() => readStoredPreset())
  const [contrastMode, setContrastModeState] = useState<ContrastMode>(() =>
    readStoredContrastMode(),
  )
  const [userBubbleTone, setUserBubbleToneState] = useState<UserBubbleTone>(() =>
    readStoredUserBubbleTone(),
  )
  const [agentMemoryEnabled, setAgentMemoryEnabledState] = useState<boolean>(() =>
    readStoredAgentMemoryEnabled(),
  )
  const [systemTheme, setSystemTheme] = useState<'light' | 'dark'>(() =>
    getSystemTheme(),
  )

  const resolvedTheme = theme === 'system' ? systemTheme : theme

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = () => {
      setSystemTheme(mediaQuery.matches ? 'dark' : 'light')
    }

    handleChange()
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  useLayoutEffect(() => {
    const root = document.documentElement
    root.setAttribute(THEME_TRANSITION_SUPPRESSION_ATTR, '')
    root.classList.toggle('dark', resolvedTheme === 'dark')
    root.dataset.contrastMode = contrastMode
    root.dataset.themePreset = preset
    root.dataset.userBubbleTone = userBubbleTone
    root.style.colorScheme = resolvedTheme

    // Commit the final token values while transitions are disabled.
    root.getBoundingClientRect()
    return scheduleThemeTransitionRestore(root)
  }, [contrastMode, preset, resolvedTheme, userBubbleTone])

  const value = useMemo<ThemeContextValue>(
    () => ({
      agentMemoryEnabled,
      contrastMode,
      preset,
      resolvedTheme,
      setAgentMemoryEnabled(nextEnabled) {
        localStorage.setItem(
          AGENT_MEMORY_ENABLED_STORAGE_KEY,
          nextEnabled ? 'true' : 'false',
        )
        setAgentMemoryEnabledState(nextEnabled)
      },
      setContrastMode(nextMode) {
        localStorage.setItem(CONTRAST_MODE_STORAGE_KEY, nextMode)
        setContrastModeState(nextMode)
      },
      setPreset(nextPreset) {
        localStorage.setItem(THEME_PRESET_STORAGE_KEY, nextPreset)
        setPresetState(nextPreset)
      },
      setTheme(nextTheme) {
        localStorage.setItem(THEME_STORAGE_KEY, nextTheme)
        setThemeState(nextTheme)
      },
      setUserBubbleTone(nextTone) {
        localStorage.setItem(USER_BUBBLE_TONE_STORAGE_KEY, nextTone)
        setUserBubbleToneState(nextTone)
      },
      theme,
      userBubbleTone,
    }),
    [agentMemoryEnabled, contrastMode, preset, resolvedTheme, theme, userBubbleTone],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider.')
  }
  return context
}

function readStoredTheme(): ThemeMode {
  if (typeof localStorage === 'undefined') return 'system'
  const stored = localStorage.getItem(THEME_STORAGE_KEY)
  if (stored === 'light' || stored === 'dark' || stored === 'system') {
    return stored
  }
  return 'system'
}

function readStoredPreset(): ThemePreset {
  if (typeof localStorage === 'undefined') return 'standard'
  const stored = localStorage.getItem(THEME_PRESET_STORAGE_KEY)
  if (
    stored === 'standard'
    || stored === 'slate'
    || stored === 'graphite'
    || stored === 'sage'
  ) {
    return stored
  }
  return 'standard'
}

function readStoredContrastMode(): ContrastMode {
  if (typeof localStorage === 'undefined') return 'standard'
  const stored = localStorage.getItem(CONTRAST_MODE_STORAGE_KEY)
  return stored === 'high' ? stored : 'standard'
}

function readStoredUserBubbleTone(): UserBubbleTone {
  if (typeof localStorage === 'undefined') return 'gray'
  const stored = localStorage.getItem(USER_BUBBLE_TONE_STORAGE_KEY)
  if (
    stored === 'gray'
    || stored === 'mint'
    || stored === 'orange'
    || stored === 'sky'
    || stored === 'violet'
    || stored === 'ink'
  ) {
    return stored
  }
  return 'gray'
}

function readStoredAgentMemoryEnabled(): boolean {
  // Privacy default OFF; the server row wins on login (account-preferences sync).
  if (typeof localStorage === 'undefined') return false
  return localStorage.getItem(AGENT_MEMORY_ENABLED_STORAGE_KEY) === 'true'
}

function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light'
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light'
}

function scheduleThemeTransitionRestore(root: HTMLElement): () => void {
  let restored = false
  let frameId: number | null = null
  let timeoutId: number | null = null

  const restore = () => {
    if (restored) return
    restored = true
    root.removeAttribute(THEME_TRANSITION_SUPPRESSION_ATTR)
  }

  if (typeof window.requestAnimationFrame === 'function') {
    frameId = window.requestAnimationFrame(() => {
      timeoutId = window.setTimeout(restore, 0)
    })
  } else {
    timeoutId = window.setTimeout(restore, 0)
  }

  return () => {
    if (frameId !== null) window.cancelAnimationFrame(frameId)
    if (timeoutId !== null) window.clearTimeout(timeoutId)
    restore()
  }
}
