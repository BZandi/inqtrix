import { parseEditorMarkdown, type SuggestionDescriptor } from '@inqtrix/editor-schema'

import { summarizeEditorChange } from '../src/changeSummary'

const AUTHOR_ID = '11111111-1111-4111-8111-111111111111'
const PATCH_ID = '22222222-2222-4222-8222-222222222222'

describe('bounded collaboration change summaries', () => {
  // Frueher hiess dieser Fall "without HTML" und erwartete 'unsafe world' --
  // der Auszug loeschte, was wie ein Tag aussah. Der String kommt aber aus
  // textBetween und ist reiner Text: geloescht wurde damit Nutzerinhalt, und
  // die Aenderungsanzeige zeigte etwas anderes als das Dokument. Die
  // Zusammenfassung ist eine Wiedergabe, keine Reinigung.
  it('gibt getippten Text woertlich wieder, auch wenn er wie Auszeichnung aussieht', () => {
    const summary = summarizeEditorChange({
      before: parseEditorMarkdown('Hello'),
      after: parseEditorMarkdown('Hello <script>unsafe</script> world'),
      changeKind: 'direct',
      decision: null,
      suggestions: [],
    })

    expect(summary).toEqual({
      edits: [{
        after: '<script>unsafe</script> world',
        before: '',
        kind: 'direct',
        position: 5,
      }],
      omittedEditCount: 0,
    })
  })

  // Gegenprobe zum eigentlichen Ausloeser: eine EINZELNE Klammer, wie sie beim
  // Tippen von Code oder einem Vergleich entsteht. Genau diese Eingabe hat den
  // Schreibvorgang abgelehnt und das Dokument fuer alle Beteiligten gesperrt.
  it('gibt eine einzelne Winkelklammer unveraendert weiter', () => {
    const summary = summarizeEditorChange({
      before: parseEditorMarkdown('if (a'),
      after: parseEditorMarkdown('if (a < b) return x > y'),
      changeKind: 'direct',
      decision: null,
      suggestions: [],
    })

    expect(summary.edits[0]?.after).toBe('< b) return x > y')
  })

  // Ein Emoji genau auf der Kuerzungsgrenze: schneidet der Auszug an einer
  // UTF-16-Einheit, entsteht ein alleinstehendes Surrogat. Das ist kein
  // gueltiges JSON, und Postgres weist den jsonb-Schreibvorgang zurueck
  // ("Unicode low surrogate must follow a high surrogate") -- derselbe
  // Schaden wie bei der Winkelklammer, nur ueber ein anderes Zeichen.
  it('kuerzt an Codepoint-Grenzen und zerreisst kein Ersatzzeichenpaar', () => {
    const summary = summarizeEditorChange({
      before: parseEditorMarkdown('x'),
      after: parseEditorMarkdown(`x${'a'.repeat(158)}\u{1F600}bbb`),
      changeKind: 'direct',
      decision: null,
      suggestions: [],
    })

    const auszug = summary.edits[0]?.after ?? ''
    const zerrissen = Array.from(auszug).some((zeichen) => {
      const punkt = zeichen.codePointAt(0) ?? 0
      return punkt >= 0xd800 && punkt <= 0xdfff
    })
    expect(zerrissen).toBe(false)
    expect(JSON.parse(JSON.stringify({ auszug })).auszug).toBe(auszug)
  })

  it('limits excerpts to 160 characters', () => {
    const summary = summarizeEditorChange({
      before: parseEditorMarkdown('Before'),
      after: parseEditorMarkdown('x'.repeat(240)),
      changeKind: 'direct',
      decision: null,
      suggestions: [],
    })

    expect(summary.edits).toHaveLength(1)
    expect(summary.edits[0]?.after).toHaveLength(160)
    expect(summary.edits[0]?.after.endsWith('…')).toBe(true)
  })

  it('counts non-visible suggestions while emitting at most three edits', () => {
    const suggestions = Array.from({ length: 5 }, (_, index) => descriptor(index))
    const summary = summarizeEditorChange({
      before: parseEditorMarkdown('Before'),
      after: parseEditorMarkdown('After'),
      changeKind: 'suggestion',
      decision: null,
      suggestions,
    })

    expect(summary.edits.length).toBeLessThanOrEqual(3)
    expect(summary.omittedEditCount).toBe(4)
  })
})

function descriptor(index: number): SuggestionDescriptor {
  return {
    authorId: AUTHOR_ID,
    createdAt: 1_784_112_000 + index,
    kind: index % 2 === 0 ? 'replacement' : 'format',
    patchId: PATCH_ID,
    suggestionId: `33333333-3333-4333-8333-${String(index).padStart(12, '0')}`,
  }
}
