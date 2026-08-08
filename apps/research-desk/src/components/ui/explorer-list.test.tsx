import { TooltipProvider } from '@radix-ui/react-tooltip'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { ExplorerHistoryRow } from './explorer-list'

function renderRow(labels: string[]): string {
  return renderToStaticMarkup(
    <TooltipProvider>
      <ExplorerHistoryRow
        actions={labels.map((label) => ({ icon: <span />, label, onSelect: () => undefined }))}
        onSelect={() => undefined}
        timeLabel="1 Tag"
        title={<span>Untitled.md</span>}
      />
    </TooltipProvider>,
  )
}

/** The offset a row's action button carries, keyed by its aria-label. */
function offsetsByLabel(markup: string): Record<string, string | null> {
  const found: Record<string, string | null> = {}
  for (const button of markup.match(/<button[^>]*>/g) ?? []) {
    const label = button.match(/aria-label="([^"]*)"/)?.[1]
    if (!label) continue
    found[label] = button.match(/right-(?:1|7|13)\b/)?.[0] ?? null
  }
  return found
}

describe('ExplorerHistoryRow trailing actions', () => {
  it('lays actions out from the right edge inward', () => {
    expect(offsetsByLabel(renderRow(['Löschen']))).toMatchObject({ 'Löschen': 'right-1' })
    expect(offsetsByLabel(renderRow(['Anheften', 'Löschen']))).toMatchObject({
      'Anheften': 'right-7',
      'Löschen': 'right-1',
    })
    expect(offsetsByLabel(renderRow(['Details', 'Anheften', 'Löschen']))).toMatchObject({
      'Details': 'right-13',
      'Anheften': 'right-7',
      'Löschen': 'right-1',
    })
  })

  it('never renders an action without an offset', () => {
    // An action beyond the available slots would lose its `right-*` class and
    // fall to the row's leading edge, on top of the type icon.
    for (const labels of [['a'], ['a', 'b'], ['a', 'b', 'c']]) {
      const offsets = Object.values(offsetsByLabel(renderRow(labels)))
      expect(offsets).toHaveLength(labels.length)
      expect(offsets.every((offset) => offset !== null)).toBe(true)
    }
  })
})
