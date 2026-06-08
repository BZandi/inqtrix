/**
 * Pure presentation helpers for model cards. The single home for cost-tier,
 * token-format, reasoning-preset and capability-label logic so feature code
 * never re-derives them. See `DESIGN.md` and the model-picker components.
 */
import type { ModelCard } from '@/features/researchRuns/types'

/** A reasoning option shown in the picker footer; `effort` is the wire token. */
export type ReasoningPreset = { label: string; effort: string }

/**
 * Map a model's accepted effort tokens to up to three picker buttons
 * (`No think` / `Think` / `Think hard`). `No think` (effort `none`) is offered
 * only when the model can disable reasoning. An empty input means the model has
 * no effort control, so the picker hides the reasoning selector entirely.
 */
export function reasoningPresets(levels: string[]): ReasoningPreset[] {
  if (levels.length === 0) return []
  const canDisable = levels.includes('none')
  const graded = levels.filter((level) => level !== 'none')
  if (graded.length === 0) {
    return canDisable ? [{ label: 'No think', effort: 'none' }] : []
  }
  const presets: ReasoningPreset[] = []
  if (canDisable) presets.push({ label: 'No think', effort: 'none' })
  presets.push({ label: 'Think', effort: graded[Math.floor((graded.length - 1) / 2)] })
  if (graded.length > 1) {
    presets.push({ label: 'Think hard', effort: graded[graded.length - 1] })
  }
  return presets
}

/** Coarse `$`..`$$$$` cost tier (by output price) plus the exact value string. */
export function costTier(pricing: ModelCard['pricing']): { signs: string; value: string } {
  const output = pricing.output_per_mtok
  const signs = output <= 5 ? '$' : output <= 15 ? '$$' : output <= 30 ? '$$$' : '$$$$'
  return { signs, value: `$${output}/1M` }
}

/** Compact token count for the KONTEXT tile, e.g. 1_050_000 -> "1.05M", 200_000 -> "200k". */
export function formatTokens(count: number): string {
  if (count >= 1_000_000) {
    const millions = count / 1_000_000
    return `${Number.isInteger(millions) ? millions : Number(millions.toFixed(2))}M`
  }
  return `${Math.round(count / 1000)}k`
}

const CAPABILITY_LABELS: Record<string, string> = {
  reasoning: 'Reasoning',
  code: 'Code',
  tool_use: 'Tool-Use',
  vision: 'Vision',
}

/** Display label for a capability tag (falls back to the raw tag). */
export function capabilityLabel(capability: string): string {
  return CAPABILITY_LABELS[capability] ?? capability
}

/** Capitalised TEMPO label for the hover-card (e.g. "mittel" -> "Mittel"). */
export function speedLabel(speed: string): string {
  return speed.charAt(0).toUpperCase() + speed.slice(1)
}
