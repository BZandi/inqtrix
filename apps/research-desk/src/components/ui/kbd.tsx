import * as React from 'react'

import { cn } from '@/lib/utils'

/**
 * Keyboard key badge — the canonical key hint used in menus and legends (e.g.
 * the mention-menu footer "↑ ↓ Navigate · Esc Close"). Fixed at the
 * design-system micro-control size: `h-4`, `text-[10px]`, bordered. See
 * `DESIGN.md` §4.
 */
export function Kbd({ className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return (
    <kbd
      className={cn(
        'inline-flex h-4 min-w-4 items-center justify-center rounded border border-border bg-surface px-1 text-[10px] font-sans text-muted-foreground',
        className,
      )}
      {...props}
    />
  )
}
