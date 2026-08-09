import { motion, useReducedMotion } from 'motion/react'
import type { ReactNode } from 'react'

import { appMotion } from '@/motion/transitions'

/** The ONE entry treatment for a workspace mount.
 *
 * Every view switch re-keys this wrapper, so the incoming workspace fades in
 * with the same 4px rise the report panel made familiar — one vocabulary
 * (appMotion.view: the panel curve at desktop view-switch tempo) instead of
 * per-view improvisation. Deliberately no
 * AnimatePresence and no exit leg: the outgoing view swaps instantly, which
 * keeps switching snappy, and the entry also papers over the first bare
 * commit some workspaces paint while their children mount. Views with their
 * own inner entries (report, settings) stay as they are — the curves are
 * identical, so the layers read as one movement.
 */
export function ViewEntry({ children, viewKey }: {
  children: ReactNode
  viewKey: string
}) {
  const reduceMotion = useReducedMotion()
  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="flex h-full min-h-0 w-full min-w-0 flex-col"
      initial={reduceMotion ? false : { opacity: 0, y: 4 }}
      key={viewKey}
      transition={appMotion.view}
    >
      {children}
    </motion.div>
  )
}
