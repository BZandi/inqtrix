import type { ReactNode } from 'react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { LocaleProvider } from '@/i18n/LocaleProvider'
import { ThemeProvider } from '@/theme/ThemeProvider'

type AppProvidersProps = {
  children: ReactNode
}

export function AppProviders({ children }: AppProvidersProps) {
  return (
    <ThemeProvider>
      <LocaleProvider>
        <TooltipProvider delayDuration={250}>{children}</TooltipProvider>
      </LocaleProvider>
    </ThemeProvider>
  )
}
