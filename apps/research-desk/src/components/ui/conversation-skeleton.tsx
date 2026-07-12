import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

/**
 * Placeholder for a conversation whose messages/items are still lazy-loading, so
 * opening an existing chat or knowledge session renders its structure instantly
 * (a quick fade + the shared pulse) instead of flashing the empty-state hero.
 * Shared by Chat and Knowledge Desk — a left content cluster, a right bubble, and
 * a second left cluster read as "a conversation is loading" on both surfaces.
 */
export function ConversationSkeleton({ reduceMotion }: { reduceMotion?: boolean | null }) {
  return (
    <div
      aria-hidden
      className={cn('flex flex-col gap-5', !reduceMotion && 'animate-in fade-in-0 duration-150')}
    >
      <div className="flex flex-col gap-2">
        <Skeleton className="h-3.5 w-24" />
        <Skeleton className="h-4 w-[86%]" />
        <Skeleton className="h-4 w-[72%]" />
        <Skeleton className="h-4 w-[64%]" />
      </div>
      <div className="flex justify-end">
        <Skeleton className="h-14 w-[46%] rounded-lg" />
      </div>
      <div className="flex flex-col gap-2">
        <Skeleton className="h-3.5 w-24" />
        <Skeleton className="h-4 w-[80%]" />
        <Skeleton className="h-4 w-[58%]" />
      </div>
    </div>
  )
}
