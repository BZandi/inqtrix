import { describe, expect, it } from 'vitest'

import {
  extractMarkdownCodeBlocks,
  extractMarkdownCodeLanguages,
  plainCodeLanguageFromClassName,
} from './markdownLanguage'

describe('markdown language helpers', () => {
  it('extracts supported fenced code languages once in encounter order', () => {
    const markdown = [
      '```python',
      'print("hello")',
      '```',
      '',
      '```json title="payload"',
      '{"ok": true}',
      '```',
      '',
      '```py',
      'print("again")',
      '```',
    ].join('\n')

    expect(extractMarkdownCodeLanguages(markdown)).toEqual(['python', 'json'])
  })

  it('reads language-* classes from markdown fallback code blocks', () => {
    expect(plainCodeLanguageFromClassName('language-python')).toBe('python')
    expect(plainCodeLanguageFromClassName(['rounded', 'language-py'])).toBe('python')
    expect(plainCodeLanguageFromClassName('language-unknown')).toBeNull()
  })

  it('extracts fenced code blocks with normalized languages', () => {
    const markdown = [
      'Text',
      '',
      '```py',
      'print("hello")',
      '```',
      '',
      '~~~bash',
      'echo ok',
      '~~~',
    ].join('\n')

    expect(extractMarkdownCodeBlocks(markdown)).toEqual([
      { code: 'print("hello")', language: 'python' },
      { code: 'echo ok', language: 'bash' },
    ])
  })
})
