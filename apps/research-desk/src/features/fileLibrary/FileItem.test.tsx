import { renderToStaticMarkup } from 'react-dom/server'
import type { ComponentProps } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { LocaleProvider } from '@/i18n/LocaleProvider'
import type { FileAssetRecord } from '@/features/project/types'
import { FileRow } from './FileItem'

const asset: FileAssetRecord = {
  createdAt: '2026-07-23T08:00:00.000Z',
  extractedText: 'Canonical text',
  fileName: 'legacy.pdf',
  groupId: null,
  id: 'file-legacy',
  label: 'legacy',
  mimeType: 'application/pdf',
  origin: 'library',
  pageCount: 1,
  parseStatus: 'parsed',
  parseWarning: null,
  sectionId: 'section-1',
  sizeBytes: 128,
  textTruncated: false,
  title: 'Legacy.pdf',
  updatedAt: '2026-07-23T08:00:00.000Z',
}

function markup(status: 'blocked' | 'reconciling' | 'deleting' | 'delete_failed') {
  return renderToStaticMarkup(
    <LocaleProvider>
      <TooltipProvider>
        <FileRow
          asset={asset}
          indexRemoval={{
            error: status === 'blocked' ? 'Index refresh required' : undefined,
            status,
          }}
          memberState="embedded"
          mode="index"
          onRemoveFromIndex={vi.fn()}
          onRetryIndexRemoval={vi.fn()}
        />
      </TooltipProvider>
    </LocaleProvider>,
  )
}

describe('FileItem index-removal truthfulness', () => {
  it('keeps an unresolved legacy member visible and labels it as blocked', () => {
    const html = markup('blocked')

    expect(html).toContain('Abgleich erforderlich')
    expect(html).toContain('Löschen erneut versuchen')
    expect(html).not.toContain('Aus Suche entfernt')
  })

  it('shows source reconciliation as non-terminal work', () => {
    const html = markup('reconciling')

    expect(html).toContain('Serverabgleich')
    expect(html).not.toContain('Löschen erneut versuchen')
    expect(html).not.toContain('Indexiert')
  })
})

describe('FileItem indexing progress', () => {
  function progressMarkup(
    jobProgress: NonNullable<ComponentProps<typeof FileRow>['jobProgress']>,
  ) {
    return renderToStaticMarkup(
      <LocaleProvider>
        <TooltipProvider>
          <FileRow
            asset={asset}
            jobProgress={jobProgress}
            memberState="pending"
            mode="index"
          />
        </TooltipProvider>
      </LocaleProvider>,
    )
  }

  it('shows an idle queue label without an activity pulse', () => {
    const html = progressMarkup({ status: 'queued' })

    expect(html).toContain('Warteschlange')
    expect(html).not.toContain('inqtrix-running-dot')
    expect(html).not.toContain('Nicht indexiert')
  })

  it('keeps the running badge stable and renders batch progress beside it', () => {
    const html = progressMarkup({
      currentBatch: 18,
      phase: 'contextualization',
      status: 'running',
      totalBatches: 54,
    })

    expect(html).toContain('>Läuft<')
    expect(html).toContain('Kontext 18/54')
    expect(html).toContain('whitespace-nowrap')
    expect(html).toContain('inqtrix-running-dot')
  })

  it('shows a paused document honestly without a running pulse', () => {
    const html = progressMarkup({
      phase: 'embedding',
      status: 'paused_dependency',
    })

    expect(html).toContain('>Pausiert<')
    expect(html).not.toContain('>Läuft<')
    expect(html).not.toContain('inqtrix-running-dot')
  })

  it('renders embedding slice progress once batch numbers are known', () => {
    const html = progressMarkup({
      currentBatch: 6,
      phase: 'embedding',
      status: 'running',
      totalBatches: 9,
    })

    expect(html).toContain('Einbettung 6/9')
    expect(html).toContain('inqtrix-running-dot')
  })

  it('names a provider wait instead of hiding it behind a generic label', () => {
    const html = progressMarkup({
      currentBatch: 7,
      phase: 'embedding_wait',
      status: 'running',
      totalBatches: 9,
    })

    expect(html).toContain('Wartet auf Anbieter')
    expect(html).not.toContain('Vorbereitung')
    expect(html).toContain('inqtrix-running-dot')
  })
})
