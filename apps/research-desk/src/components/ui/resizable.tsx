import * as ResizablePrimitive from "react-resizable-panels"

import { cn } from "@/lib/utils"

const ResizablePanelGroup = ({
  className,
  ...props
}: React.ComponentProps<typeof ResizablePrimitive.Group>) => (
  <ResizablePrimitive.Group
    className={cn(
      "flex h-full w-full data-[panel-group-direction=vertical]:flex-col",
      className
    )}
    {...props}
  />
)

const ResizablePanel = ResizablePrimitive.Panel

/**
 * Minimalist split divider: a 1px hairline seam (so an adjacent panel reads as
 * attached, not as a floating window) with a wide transparent hit area for a
 * comfortable drag target. The line stays quiet at rest and only brightens on
 * hover, then turns to the brand colour while dragging (`data-separator` is the
 * state hook react-resizable-panels sets: inactive | hover | drag). No grip
 * pill — the affordance is the cursor plus the subtle highlight. Shared by the
 * research-desk report split and the chat history split so both feel identical.
 */
const ResizableHandle = ({
  className,
  ...props
}: React.ComponentProps<typeof ResizablePrimitive.Separator>) => (
  <ResizablePrimitive.Separator
    className={cn(
      "relative w-px shrink-0 cursor-col-resize bg-border outline-none transition-colors duration-150",
      "after:absolute after:inset-y-0 after:left-1/2 after:w-4 after:-translate-x-1/2",
      "hover:bg-foreground/20 data-[separator=hover]:bg-foreground/20 data-[separator=drag]:bg-brand",
      "focus-visible:ring-1 focus-visible:ring-ring",
      className
    )}
    {...props}
  />
)

export { ResizablePanelGroup, ResizablePanel, ResizableHandle }
