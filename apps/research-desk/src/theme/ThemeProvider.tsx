import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

export type ThemeMode = 'light' | 'dark' | 'system'
export type ThemePreset = 'standard' | 'slate' | 'graphite' | 'sage'
export type ContrastMode = 'standard' | 'high'
export type UserBubbleTone = 'gray' | 'mint' | 'orange' | 'sky' | 'violet' | 'ink'

type ThemeContextValue = {
  contrastMode: ContrastMode
  preset: ThemePreset
  resolvedTheme: 'light' | 'dark'
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

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', resolvedTheme === 'dark')
    root.dataset.contrastMode = contrastMode
    root.dataset.themePreset = preset
    root.dataset.userBubbleTone = userBubbleTone
    root.style.colorScheme = resolvedTheme
  }, [contrastMode, preset, resolvedTheme, userBubbleTone])

  const value = useMemo<ThemeContextValue>(
    () => ({
      contrastMode,
      preset,
      resolvedTheme,
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
    [contrastMode, preset, resolvedTheme, theme, userBubbleTone],
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

function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light'
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light'
}
