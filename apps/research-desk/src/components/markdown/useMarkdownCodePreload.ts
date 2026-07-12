import { useEffect } from 'react'

import { scheduleIdle } from '@/lib/idle'
import { useTheme } from '@/theme/ThemeProvider'
import { preloadMarkdownCodeHighlights } from './MarkdownRenderer'

export function useMarkdownCodePreload(markdowns: readonly string[]): void {
  const { resolvedTheme } = useTheme()

  useEffect(() => {
    if (markdowns.length === 0) return undefined
    return scheduleIdle(() => {
      void preloadMarkdownCodeHighlights(markdowns, resolvedTheme)
    }, { timeout: 600 })
  }, [markdowns, resolvedTheme])
}
