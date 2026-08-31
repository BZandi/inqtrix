import { describe, expect, it } from 'vitest'

import { translations } from '@/i18n/translations'
import { agentRunFailureText } from './runFailure'

const de = translations.de

describe('agentRunFailureText', () => {
  it('turns the machine code into a sentence that names the consequence', () => {
    // A user was shown "all_tasks_failed" — a word from inside the
    // engine, with no hint of what it meant.
    const text = agentRunFailureText('all_tasks_failed', de)
    expect(text).not.toBe('all_tasks_failed')
    expect(text).toContain('kein Ergebnis')
  })

  it('keeps the detail of a coded failure', () => {
    // The planner reports invalid plans as `code: detail`; the detail
    // names the rule that was broken and must survive.
    expect(
      agentRunFailureText('plan_invalid: genau ein synthesis-Task', de),
    ).toBe('Der Plan war nicht gültig: genau ein synthesis-Task')
  })

  it('passes an unknown failure through unchanged', () => {
    // A failure the UI cannot name is still one the user must see —
    // and the backend often sends a finished German sentence already.
    const sentence = 'Die Antwort wurde nicht veröffentlicht, weil …'
    expect(agentRunFailureText(sentence, de)).toBe(sentence)
    expect(agentRunFailureText('brandneuer_code', de)).toBe('brandneuer_code')
  })

  it('leaves an empty failure empty', () => {
    expect(agentRunFailureText('   ', de)).toBe('')
  })

  it('is translated in both locales', () => {
    expect(agentRunFailureText('all_tasks_failed', translations.en))
      .toContain('no result')
  })
})
