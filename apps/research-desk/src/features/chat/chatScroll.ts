export const CHAT_AUTO_FOLLOW_THRESHOLD_PX = 96
export const CHAT_BOTTOM_EPSILON_PX = 1

export type ChatScrollMetrics = {
  clientHeight: number
  scrollHeight: number
  scrollTop: number
}

export type ChatScrollMode = 'auto' | 'none' | 'smooth'

export function chatDistanceFromBottom(metrics: ChatScrollMetrics): number {
  return metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight
}

export function isChatNearBottom(
  metrics: ChatScrollMetrics,
  thresholdPx = CHAT_AUTO_FOLLOW_THRESHOLD_PX,
): boolean {
  return chatDistanceFromBottom(metrics) <= thresholdPx
}

export function chatScrollModeForUpdate({
  hasActiveAssistantMessage,
  nearBottom,
  reduceMotion,
  threadChanged,
}: {
  hasActiveAssistantMessage: boolean
  nearBottom: boolean
  reduceMotion: boolean | null
  threadChanged: boolean
}): ChatScrollMode {
  if (threadChanged) return 'auto'
  if (!nearBottom) return 'none'
  return reduceMotion || hasActiveAssistantMessage ? 'auto' : 'smooth'
}
