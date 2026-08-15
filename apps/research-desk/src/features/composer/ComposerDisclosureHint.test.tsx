import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { LocaleProvider } from '@/i18n/LocaleProvider'
import { ComposerDisclosureHint } from './ComposerDisclosureHint'

function markup() {
  return renderToStaticMarkup(
    <LocaleProvider>
      <ComposerDisclosureHint />
    </LocaleProvider>,
  )
}

describe('ComposerDisclosureHint', () => {
  it('states that answers come from an AI system', () => {
    expect(markup()).toContain('Antworten werden von einem KI-System erzeugt.')
  })

  it('uses the helper-text role, not the smaller micro role', () => {
    // t-hint sits one step below the context meter beside it, which would
    // read as fine print — the exact thing the disclosure must not be.
    const html = markup()

    expect(html).toContain('t-meta')
    expect(html).not.toContain('t-hint')
  })

  it('keeps the colour token at full opacity', () => {
    // DESIGN.md forbids dimming metadata; a disclosure must not be quieter
    // than the controls around it.
    expect(markup()).toContain('text-muted-foreground')
    expect(markup()).not.toMatch(/text-muted-foreground\/\d/)
  })
})
