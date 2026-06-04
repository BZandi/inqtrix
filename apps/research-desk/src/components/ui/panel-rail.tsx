import { PanelLeftOpen, PanelRightOpen } from '@/components/icons'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

type PanelRailProps = {
  label: string
  onExpand: () => void
  side: 'left' | 'right'
}

/**
 * Narrow restore rail left behind when a side panel is collapsed. Clicking it
 * re-expands the panel. The directional `Panel{Side}Open` icon keeps the
 * collapse/expand language identical across the editor and chat panels (see the
 * report panel's own rail in `ReportPanel.tsx`).
 */
export function PanelRail({ label, onExpand, side }: PanelRailProps) {
  const Icon = side === 'left' ? PanelLeftOpen : PanelRightOpen
  return (
    <aside
      className={cn(
        'flex h-full w-11 shrink-0 flex-col items-center bg-surface py-2',
        side === 'left' ? 'border-r border-border' : 'border-l border-border',
      )}
    >
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            aria-label={label}
            onClick={onExpand}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Icon className="size-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side={side === 'left' ? 'right' : 'left'}>{label}</TooltipContent>
      </Tooltip>
    </aside>
  )
}
