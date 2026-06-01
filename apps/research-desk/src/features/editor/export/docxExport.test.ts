import { Paragraph, Table } from 'docx'
import { describe, expect, it } from 'vitest'
import { degradeMathDelimiters, docxFileName, markdownToDocxBlocks } from './docxExport'

describe('degradeMathDelimiters', () => {
  it('unwraps block and bracketed math to source text', () => {
    expect(degradeMathDelimiters('$$E = mc^2$$')).toBe('E = mc^2')
    expect(degradeMathDelimiters('\\[a^2 + b^2\\]')).toBe('a^2 + b^2')
    expect(degradeMathDelimiters('\\(x_i\\)')).toBe('x_i')
  })

  it('unwraps inline math only when it looks like math', () => {
    expect(degradeMathDelimiters('Sei $x_i = 0$ gegeben.')).toBe('Sei x_i = 0 gegeben.')
    expect(degradeMathDelimiters('Das kostet $5 und $10.')).toBe('Das kostet $5 und $10.')
  })

  it('leaves plain text untouched', () => {
    expect(degradeMathDelimiters('Kein Mathe hier.')).toBe('Kein Mathe hier.')
  })
})

describe('docxFileName', () => {
  const at = new Date(2026, 5, 1, 9, 7, 5)

  it('slugifies a short title and appends a YYYYMMDD-HHMMSS stamp', () => {
    expect(docxFileName('Gemini als KI-Produktfamilie', at))
      .toBe('gemini-als-ki-produktfamilie-20260601-090705.docx')
  })

  it('caps the title slug at 40 characters before the stamp', () => {
    const longTitle = 'Ein sehr langer Dokumenttitel der deutlich ueber vierzig Zeichen hinausgeht'
    const name = docxFileName(longTitle, at)
    expect(name.endsWith('-20260601-090705.docx')).toBe(true)
    expect(name.replace('-20260601-090705.docx', '').length).toBeLessThanOrEqual(40)
  })

  it('folds diacritics and umlauts to ASCII', () => {
    expect(docxFileName('Übersicht über Ähnliches', at)).toBe('ubersicht-uber-ahnliches-20260601-090705.docx')
  })

  it('falls back when the title yields no usable characters', () => {
    expect(docxFileName('   ', at)).toBe('dokument-20260601-090705.docx')
    expect(docxFileName('***', at)).toBe('dokument-20260601-090705.docx')
  })
})

describe('markdownToDocxBlocks', () => {
  it('maps headings, paragraphs, lists, tables and code to docx blocks', () => {
    const markdown = [
      '# Titel',
      '',
      'Ein **fetter** Absatz mit [Link](https://example.com).',
      '',
      '- eins',
      '- zwei',
      '',
      '| A | B |',
      '| - | - |',
      '| 1 | 2 |',
      '',
      '```ts',
      'const x = 1',
      '```',
    ].join('\n')
    const blocks = markdownToDocxBlocks(markdown)
    expect(blocks.length).toBeGreaterThan(0)
    expect(blocks.some((block) => block instanceof Table)).toBe(true)
    expect(blocks.every((block) => block instanceof Paragraph || block instanceof Table)).toBe(true)
  })

  it('skips raw HTML nodes for safety', () => {
    const blocks = markdownToDocxBlocks('<script>alert(1)</script>\n\nText.')
    expect(blocks.every((block) => block instanceof Paragraph)).toBe(true)
    expect(blocks.length).toBe(1)
  })
})
