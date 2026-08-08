import { parseEditorMarkdown, type SuggestionDescriptor } from '@inqtrix/editor-schema'

import { summarizeEditorChange } from '../src/changeSummary'

const AUTHOR_ID = '11111111-1111-4111-8111-111111111111'
const PATCH_ID = '22222222-2222-4222-8222-222222222222'

describe('bounded collaboration change summaries', () => {
  it('reports a compact direct-text difference without HTML', () => {
    const summary = summarizeEditorChange({
      before: parseEditorMarkdown('Hello'),
      after: parseEditorMarkdown('Hello <script>unsafe</script> world'),
      changeKind: 'direct',
      decision: null,
      suggestions: [],
    })

    expect(summary).toEqual({
      edits: [{
        after: 'unsafe world',
        before: '',
        kind: 'direct',
        position: 5,
      }],
      omittedEditCount: 0,
    })
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
