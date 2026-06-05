import { describe, expect, it } from 'vitest'
import { documentDiffPlan, suggestionDiffPlan } from './suggestionDiff'

describe('documentDiffPlan', () => {
  it('groups adjacent deleted and inserted markdown lines as one replacement', () => {
    const plan = documentDiffPlan(
      'Intro\n\nThe claim is false.\n\nEnd',
      'Intro\n\nThe claim is partly false.\n\nEnd',
    )

    expect(plan.map((block) => block.kind)).toEqual(['equal', 'replace', 'equal'])
    const replacement = plan[1]
    expect(replacement.kind).toBe('replace')
    if (replacement.kind !== 'replace') return
    expect(replacement.beforeMarkdown).toBe('The claim is false.\n')
    expect(replacement.afterMarkdown).toBe('The claim is partly false.\n')
  })

  it('uses inline tokens for simple prose replacements', () => {
    const plan = documentDiffPlan(
      'Google Gemini remains active and available.\n',
      'Google Gemini remains active, expanded, and available.\n',
    )

    const replacement = plan[0]
    expect(replacement.kind).toBe('replace')
    if (replacement.kind !== 'replace') return
    expect(replacement.inlineSegments).not.toBeNull()
    expect(replacement.inlineSegments?.some((segment) => segment.type === 'insert' && segment.text.includes('expanded'))).toBe(true)
  })

  it('keeps structural markdown replacements in a compact rendered fallback', () => {
    const plan = documentDiffPlan(
      '- Old evidence item\n',
      '- New evidence item\n',
    )

    const replacement = plan[0]
    expect(replacement.kind).toBe('replace')
    if (replacement.kind !== 'replace') return
    expect(replacement.inlineSegments).toBeNull()
  })

  it('does not inline changed markdown link syntax', () => {
    const plan = documentDiffPlan(
      'Evidence points to [E1](https://example.com/old).\n',
      'Evidence points to [E2](https://example.com/new).\n',
    )

    const replacement = plan[0]
    expect(replacement.kind).toBe('replace')
    if (replacement.kind !== 'replace') return
    expect(replacement.inlineSegments).toBeNull()
  })
})

describe('suggestionDiffPlan', () => {
  it('keeps the existing model suggestion diff path inline for short prose', () => {
    const plan = suggestionDiffPlan('Use short text.', 'Use sharper short text.')

    expect(plan.display).toBe('inline')
    expect(plan.reviewSurface).toBe('panel')
  })
})
