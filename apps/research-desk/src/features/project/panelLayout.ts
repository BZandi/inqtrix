import type { ProjectPanelLayoutState } from './types'

export type ProjectPanelLayoutKey = keyof ProjectPanelLayoutState

type PanelLayoutLimit = {
  defaultSize: number
  max: number
  min: number
}

export const PANEL_LAYOUT_LIMITS: Record<ProjectPanelLayoutKey, PanelLayoutLimit> = {
  chatHistory: { defaultSize: 26, max: 42, min: 18 },
  knowledgeHistory: { defaultSize: 26, max: 42, min: 18 },
  knowledgeSource: { defaultSize: 36, max: 48, min: 28 },
  researchReport: { defaultSize: 42, max: 58, min: 26 },
}

export const DEFAULT_PANEL_LAYOUT: ProjectPanelLayoutState = {
  chatHistory: PANEL_LAYOUT_LIMITS.chatHistory.defaultSize,
  knowledgeHistory: PANEL_LAYOUT_LIMITS.knowledgeHistory.defaultSize,
  knowledgeSource: PANEL_LAYOUT_LIMITS.knowledgeSource.defaultSize,
  researchReport: PANEL_LAYOUT_LIMITS.researchReport.defaultSize,
}

export function clampPanelLayoutSize(
  key: ProjectPanelLayoutKey,
  value: number,
): number {
  const limit = PANEL_LAYOUT_LIMITS[key]
  if (!Number.isFinite(value)) return limit.defaultSize
  return Math.min(limit.max, Math.max(limit.min, value))
}

export function normalizePanelLayout(value: unknown): ProjectPanelLayoutState {
  const record = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
  return {
    chatHistory: clampPanelLayoutSize('chatHistory', numberOrDefault(record.chatHistory, DEFAULT_PANEL_LAYOUT.chatHistory)),
    knowledgeHistory: clampPanelLayoutSize('knowledgeHistory', numberOrDefault(record.knowledgeHistory, DEFAULT_PANEL_LAYOUT.knowledgeHistory)),
    knowledgeSource: clampPanelLayoutSize('knowledgeSource', numberOrDefault(record.knowledgeSource, DEFAULT_PANEL_LAYOUT.knowledgeSource)),
    researchReport: clampPanelLayoutSize('researchReport', numberOrDefault(record.researchReport, DEFAULT_PANEL_LAYOUT.researchReport)),
  }
}

function numberOrDefault(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}
