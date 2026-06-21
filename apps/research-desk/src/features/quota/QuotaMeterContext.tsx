import { createContext, useContext, useMemo, type ReactNode } from 'react'

type QuotaMeterGate = {
  /** Whether the quota meter applies (oidc + capability + session, or demo). */
  enabled: boolean
  /** Demo mode renders seeded figures without a backend. */
  demo: boolean
}

const QuotaMeterContext = createContext<QuotaMeterGate>({
  demo: false,
  enabled: false,
})

/** Set once at the app root so every composer's {@link QuotaMeter} shares
 * one gate decision (no prop-drilling of capability/session/demo through
 * the chat/editor/research trees). */
export function QuotaMeterProvider({
  enabled,
  demo,
  children,
}: QuotaMeterGate & { children: ReactNode }) {
  const value = useMemo(() => ({ demo, enabled }), [demo, enabled])
  return (
    <QuotaMeterContext.Provider value={value}>
      {children}
    </QuotaMeterContext.Provider>
  )
}

export function useQuotaMeterGate(): QuotaMeterGate {
  return useContext(QuotaMeterContext)
}
