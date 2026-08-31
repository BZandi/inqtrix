import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EditorSuggestionRecord } from '@/features/project/types'
import { EditorDocumentChangesSection } from './EditorDocumentChangesSection'

const labels = {
  accept: 'Übernehmen',
  acceptAll: 'Alle übernehmen',
  documentChanges: 'Dokument-Änderungen',
  proposedChange: 'Vorschlag',
  reject: 'Verwerfen',
  rejectAll: 'Alle verwerfen',
}

function suggestion(overrides: Partial<EditorSuggestionRecord> = {}): EditorSuggestionRecord {
  return {
    anchor: { from: 10, quoteAfter: ' fuer', quoteBefore: 'als ', selectedText: 'Ausgangstext', to: 22 },
    anchorText: 'Ausgangstext',
    blockId: 'block-1',
    createdAt: '2026-08-24T09:00:00.000Z',
    documentId: 'editor-doc-1',
    groupId: 'group-1',
    id: 'suggestion-1',
    originalText: 'Ausgangstext',
    origin: { kind: 'global_run' },
    proposedText: 'Basistext',
    status: 'pending',
    updatedAt: '2026-08-24T09:00:00.000Z',
    ...overrides,
  }
}

function render(suggestionErrors: Record<string, string>) {
  return renderToStaticMarkup(
    <EditorDocumentChangesSection
      labels={labels}
      onAcceptGroup={() => {}}
      onAcceptSuggestion={() => {}}
      onRejectGroup={() => {}}
      onRejectSuggestion={() => {}}
      onSelectSuggestion={() => {}}
      suggestionErrors={suggestionErrors}
      suggestions={[suggestion()]}
    />,
  )
}

describe('EditorDocumentChangesSection', () => {
  // Die Karte, die den Uebernehmen-Knopf traegt, traegt auch dessen
  // Fehlschlag. Vorher endete der Fehlerkanal (suggestionErrors) vor dem
  // Panel: ein 409 auf suggestions:publish verschwand spurlos, der Nutzer
  // klickte und sah NICHTS -- stiller Verlust.
  it('zeigt den Fehlschlag an der Karte, an der geklickt wurde', () => {
    const markup = render({
      'suggestion-1': 'Der Vorschlag konnte nicht uebernommen werden (Struktur nicht unterstuetzt).',
    })

    expect(markup).toContain('Der Vorschlag konnte nicht uebernommen werden (Struktur nicht unterstuetzt).')
    // Vorlesbar angekuendigt, nicht nur eingefaerbt.
    expect(markup).toContain('role="alert"')
    // Der Vorschlag bleibt sichtbar erhalten statt lautlos zu verschwinden:
    // der Knopf steht weiter da, der Nutzer kann es erneut versuchen.
    expect(markup).toContain(labels.accept)
  })

  it('zeigt ohne Fehlschlag keine Fehlerzeile', () => {
    const markup = render({})
    expect(markup).not.toContain('role="alert"')
  })

  // Quell-Pin fuer die Verdrahtung, die in dieser DOM-losen Lane kein
  // Render-Test erreicht: der Fehlerkanal endete frueher VOR dem Panel.
  // Ein Revert der Durchreichung blieb ohne diesen Pin gruen -- dieselbe
  // Falle wie bei der SSE-Routenverdrahtung in Phase 4b.
  it('reicht den Fehlerkanal bis zu beiden Panel-Flaechen durch', async () => {
    const { readFileSync } = await import('node:fs')
    const path = await import('node:path')
    const source = readFileSync(
      path.resolve(__dirname, './EditorWorkspace.tsx'),
      'utf-8',
    )
    // Hook -> Panel
    expect(source).toContain('suggestionErrors={suggestionErrors}')
    // Panel -> Dokument-Aenderungen (die Flaeche der Assistenten-Vorschlaege)
    expect(source).toMatch(/suggestionErrors=\{suggestionErrors\}\s*\n\s*suggestions=\{documentChangeSuggestions\}/)
    // Panel -> kommentarverankerte Pruefflaeche
    expect(source).toContain('suggestionError={commentSuggestion ? suggestionErrors[commentSuggestion.id] : undefined}')
    expect(source).toContain('error={suggestionError}')
  })

  it('ordnet den Fehler seiner eigenen Karte zu, nicht irgendeiner', () => {
    // Ein Fehler auf einem FREMDEN Vorschlag darf diese Karte nicht faerben.
    const markup = render({ 'suggestion-anders': 'Fremder Fehlschlag' })
    expect(markup).not.toContain('Fremder Fehlschlag')
    expect(markup).not.toContain('role="alert"')
  })
})
