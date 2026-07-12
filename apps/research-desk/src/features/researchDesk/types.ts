import type { Locale } from '@/i18n/translations'
import type { ResearchRunAccess, ResearchRunStatus } from '@/features/researchRuns/types'

export type LocalizedText = Record<Locale, string>

export type JobStatus = ResearchRunStatus

export const phaseOrder = [
  'analysis',
  'planning',
  'search',
  'evaluation',
  'answer',
] as const

export type JobPhase = (typeof phaseOrder)[number]

export type ResearchJob = {
  access?: ResearchRunAccess
  activePhase: JobPhase
  cancelRequested?: boolean
  completedPhases: readonly JobPhase[]
  confidence?: string
  duration?: string
  error?: string
  events: Array<{
    active?: boolean
    kind: 'progress' | 'system'
    phase?: JobPhase
    severity: 'error' | 'info' | 'success' | 'warning'
    time: string
    title: LocalizedText
  }>
  id: string
  metrics: {
    claims: number
    queries: number
    rounds: string
    sources: number
  }
  phaseVisitCounts: Record<JobPhase, number>
  queueNote?: LocalizedText
  score?: string
  startedAtIso?: string
  startedAt?: string
  status: JobStatus
  submittedAt: string
  title: LocalizedText
}

export type JobFilter = Extract<JobStatus, 'cancelled' | 'completed' | 'queued' | 'running'> | 'all'

export type AppView = 'research' | 'chat' | 'agent' | 'editor' | 'knowledge' | 'prompt-library' | 'database' | 'settings'

export function localizedText(value: LocalizedText, locale: Locale) {
  return value[locale]
}
