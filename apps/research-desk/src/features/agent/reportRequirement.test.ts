import { describe, expect, it } from 'vitest'

import {
  REPORT_GUIDANCE_MAX_CHARS_FALLBACK,
  REPORT_RULE_IDS_MAX_FALLBACK,
  hasReportRequirement,
  reportGuidanceMaxChars,
  reportRuleIdsMax,
  toggleReportRule,
} from './reportRequirement'
import type { AgentCapabilityBlock } from './reportRequirement'

function capabilities(
  report_requirement?: { max_chars: number; max_rules: number },
): AgentCapabilityBlock {
  return { report_requirement } as unknown as AgentCapabilityBlock
}

describe('report requirement limits', () => {
  it('renders the server’s own numbers', () => {
    // Published == enforced: a surface that accepts what the server
    // refuses teaches the user a limit that is not real.
    const caps = capabilities({ max_chars: 500, max_rules: 1 })
    expect(reportGuidanceMaxChars(caps)).toBe(500)
    expect(reportRuleIdsMax(caps)).toBe(1)
  })

  it('falls back only when the server publishes nothing', () => {
    expect(reportGuidanceMaxChars(capabilities())).toBe(
      REPORT_GUIDANCE_MAX_CHARS_FALLBACK,
    )
    expect(reportRuleIdsMax(null)).toBe(REPORT_RULE_IDS_MAX_FALLBACK)
  })

  it('ignores a nonsensical published limit', () => {
    const caps = capabilities({ max_chars: 0, max_rules: -3 })
    expect(reportGuidanceMaxChars(caps)).toBe(
      REPORT_GUIDANCE_MAX_CHARS_FALLBACK,
    )
    expect(reportRuleIdsMax(caps)).toBe(REPORT_RULE_IDS_MAX_FALLBACK)
  })
})

describe('attaching rules', () => {
  it('adds and removes in attachment order', () => {
    expect(toggleReportRule([], 'a', 3)).toEqual(['a'])
    expect(toggleReportRule(['a'], 'b', 3)).toEqual(['a', 'b'])
    expect(toggleReportRule(['a', 'b'], 'a', 3)).toEqual(['b'])
  })

  it('never evicts an attached rule to make room', () => {
    // Silently dropping the first rule when the third is picked would
    // run under requirements the user believes are in force.
    expect(toggleReportRule(['a', 'b', 'c'], 'd', 3)).toEqual(['a', 'b', 'c'])
  })

  it('still lets the user detach at the cap', () => {
    expect(toggleReportRule(['a', 'b', 'c'], 'b', 3)).toEqual(['a', 'c'])
  })
})

describe('whether a requirement travels', () => {
  it('is false for whitespace alone', () => {
    expect(hasReportRequirement('   \n', [])).toBe(false)
  })

  it('is true for rules without any typed text', () => {
    expect(hasReportRequirement('', ['rule-1'])).toBe(true)
  })
})
