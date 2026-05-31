import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { translations, type Locale, type TranslationDictionary } from './translations'

type LocaleContextValue = {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: TranslationDictionary
}

const LOCALE_STORAGE_KEY = 'inqtrix.researchDesk.locale'
const LocaleContext = createContext<LocaleContextValue | null>(null)

type LocaleProviderProps = {
  children: ReactNode
}

export function LocaleProvider({ children }: LocaleProviderProps) {
  const [locale, setLocaleState] = useState<Locale>(() => readStoredLocale())

  useEffect(() => {
    document.documentElement.lang = locale
  }, [locale])

  const value = useMemo<LocaleContextValue>(
    () => ({
      locale,
      setLocale(nextLocale) {
        localStorage.setItem(LOCALE_STORAGE_KEY, nextLocale)
        setLocaleState(nextLocale)
      },
      t: translations[locale],
    }),
    [locale],
  )

  return (
    <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>
  )
}

export function useLocale() {
  const context = useContext(LocaleContext)
  if (!context) {
    throw new Error('useLocale must be used within LocaleProvider.')
  }
  return context
}

function readStoredLocale(): Locale {
  if (typeof localStorage === 'undefined') return 'de'
  const stored = localStorage.getItem(LOCALE_STORAGE_KEY)
  return stored === 'en' || stored === 'de' ? stored : 'de'
}
