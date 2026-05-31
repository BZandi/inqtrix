import {
  CheckCircle2,
  Circle,
  Clock3,
  FileText,
  OctagonAlert,
  LoaderCircle,
  XCircle,
  Search,
  SlidersHorizontal,
  type LucideIcon,
} from '@/components/icons'
import type { TranslationDictionary } from '@/i18n/translations'
import type { JobPhase, JobStatus } from '../types'

export const statusIcon = {
  cancelled: XCircle,
  completed: CheckCircle2,
  expired: Clock3,
  failed: OctagonAlert,
  queued: Clock3,
  running: LoaderCircle,
} satisfies Record<JobStatus, LucideIcon>

export const statusBadgeClassName = {
  cancelled: 'border-destructive/20 bg-destructive/10 text-destructive/85 hover:bg-destructive/10',
  completed: 'border-success/20 bg-success-subtle text-success hover:bg-success-subtle',
  expired: 'border-muted-foreground/20 bg-muted text-muted-foreground hover:bg-muted',
  failed: 'border-destructive/25 bg-destructive/10 text-destructive hover:bg-destructive/10',
  queued: 'border-border bg-muted text-muted-foreground hover:bg-muted',
  running: 'border-brand/20 bg-brand-subtle text-brand hover:bg-brand-subtle',
} satisfies Record<JobStatus, string>

export const phaseIcon = {
  analysis: CheckCircle2,
  planning: FileText,
  search: Search,
  evaluation: SlidersHorizontal,
  answer: FileText,
} satisfies Record<JobPhase, LucideIcon>

export const queuedPhaseIcon = Circle

export function phaseLabel(phase: JobPhase, t: TranslationDictionary) {
  if (phase === 'analysis') return t.runCard.analysis
  if (phase === 'planning') return t.runCard.planning
  if (phase === 'search') return t.runCard.search
  if (phase === 'evaluation') return t.runCard.evaluation
  return t.runCard.answer
}
