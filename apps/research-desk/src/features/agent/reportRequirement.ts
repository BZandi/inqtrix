import type { InqtrixCapabilities } from '@/features/researchRuns/types'

/** The published agent block — the one place these limits come from. */
export type AgentCapabilityBlock = InqtrixCapabilities['agent']

/**
 * The result requirement a user attaches to a run: free text plus
 * prompt-library rules.
 *
 * It has two entry points — the composer, before the run starts, and the
 * plan gate — and both are bounded by the SAME server limits. The
 * numbers come from the capability manifest (published == enforced); the
 * fallbacks below apply only to a server too old to publish them, and
 * match that server's own hardcoded values.
 */
export const REPORT_GUIDANCE_MAX_CHARS_FALLBACK = 2000
export const REPORT_RULE_IDS_MAX_FALLBACK = 3

export function reportGuidanceMaxChars(
  capabilities: AgentCapabilityBlock | null | undefined,
): number {
  const published = capabilities?.report_requirement?.max_chars
  return typeof published === 'number' && published > 0
    ? published
    : REPORT_GUIDANCE_MAX_CHARS_FALLBACK
}

export function reportRuleIdsMax(
  capabilities: AgentCapabilityBlock | null | undefined,
): number {
  const published = capabilities?.report_requirement?.max_rules
  return typeof published === 'number' && published > 0
    ? published
    : REPORT_RULE_IDS_MAX_FALLBACK
}

/**
 * Toggle one rule, honoring the server's cap.
 *
 * At the cap, selecting a further rule is a NO-OP rather than a silent
 * eviction of an earlier one: dropping a rule the user attached — to
 * make room for one they just picked — is exactly the kind of quiet
 * substitution that makes a requirement untrustworthy. The surface
 * disables the remaining entries so the cap is visible before the click.
 */
export function toggleReportRule(
  ruleIds: readonly string[],
  ruleId: string,
  max: number,
): string[] {
  if (ruleIds.includes(ruleId)) {
    return ruleIds.filter((id) => id !== ruleId)
  }
  if (ruleIds.length >= max) return [...ruleIds]
  return [...ruleIds, ruleId]
}

/** Whether anything would actually travel with the next submission. */
export function hasReportRequirement(
  guidance: string,
  ruleIds: readonly string[],
): boolean {
  return guidance.trim().length > 0 || ruleIds.length > 0
}
