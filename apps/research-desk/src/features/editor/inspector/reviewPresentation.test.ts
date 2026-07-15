import { describe, expect, it } from 'vitest'

import {
  caretOnlySelectionRender,
  collaborationCaretLabelColor,
  collaborationReviewMarkPresentation,
  createEditorExtensions,
} from '../tiptap'

describe('collaboration presence presentation', () => {
  it('keeps remote selection ranges transparent while retaining a selection decoration', () => {
    const attributes = caretOnlySelectionRender({ color: '#ff0000', name: 'Ada' })

    expect(attributes).toMatchObject({
      class: 'inqtrix-collaboration-selection',
      nodeName: 'span',
      style: 'background-color: transparent; box-shadow: none;',
      'data-collaboration-selection': 'transparent',
    })
    expect(attributes.style).not.toContain('#ff0000')
  })

  it('chooses legible caret-label text for verified light and dark colors', () => {
    expect(collaborationCaretLabelColor('#f8fafc')).toBe('#111827')
    expect(collaborationCaretLabelColor('#1d4ed8')).toBe('#ffffff')
  })
})

describe('collaboration review overlay', () => {
  it('registers exactly one review extension on the editor', () => {
    const reviewExtensions = createEditorExtensions().filter(
      (extension) => extension.name === 'collaborationReview',
    )

    expect(reviewExtensions).toHaveLength(1)
  })

  it('projects final and original displays without unrelated change overlays', () => {
    expect(collaborationReviewMarkPresentation('deletion', 'final', false).style)
      .toBe('display: none;')
    expect(collaborationReviewMarkPresentation('insertion', 'original', false).style)
      .toBe('display: none;')
    expect(collaborationReviewMarkPresentation('deletion', 'original', false).style)
      .toContain('text-decoration: none')
    expect(collaborationReviewMarkPresentation('insertion', 'all', false).style)
      .toContain('var(--success-subtle)')

    const filtered = collaborationReviewMarkPresentation('deletion', 'all', false, false)
    expect(filtered.display).toBe('final')
    expect(filtered.style).toBe('display: none;')
  })

  it('adds a stable active class without changing the requested display', () => {
    const presentation = collaborationReviewMarkPresentation('modification', 'simple', true)

    expect(presentation.className).toContain('inqtrix-review-overlay-active')
    expect(presentation.display).toBe('simple')
    expect(presentation.style).toContain('var(--brand)')
  })

  it('reveals only the active deletion in Simple Markup', () => {
    const active = collaborationReviewMarkPresentation('deletion', 'simple', true)
    const inactive = collaborationReviewMarkPresentation('deletion', 'simple', false)

    expect(active.display).toBe('simple')
    expect(active.style).toContain('text-decoration: line-through')
    expect(active.style).not.toBe('display: none;')
    expect(inactive.style).toBe('display: none;')
  })
})
