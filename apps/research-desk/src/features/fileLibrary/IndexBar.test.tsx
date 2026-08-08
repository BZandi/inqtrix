import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { LocaleProvider } from '@/i18n/LocaleProvider'
import type { IndexingJobLive, VectorIndexRecord } from '@/features/project/types'
import { IndexBar } from './IndexBar'

const index: VectorIndexRecord = {
  createdAt: '2026-07-22T08:00:00.000Z',
  dims: 3,
  handle: 'contracts',
  id: 'index-1',
  members: [],
  model: 'text-embedding-3-small',
  serverCollectionId: 'collection-1',
  status: 'indexing',
  title: 'Contracts',
  updatedAt: '2026-07-22T08:00:00.000Z',
}

const paused: IndexingJobLive = {
  completedDocuments: 2,
  currentBatch: 7,
  jobId: 'job-1',
  pauseMessage: 'Provider did not answer before the network safety limit.',
  percent: 40,
  phase: 'contextualize',
  runningFileIds: [],
  source: 'server',
  startedAt: '2026-07-22T08:00:00.000Z',
  status: 'paused_dependency',
  totalBatches: 18,
  totalDocuments: 5,
}

describe('IndexBar paused indexing contract', () => {
  it('labels the client-derived vector count as an estimate', () => {
    const markup = renderToStaticMarkup(
      <LocaleProvider>
        <TooltipProvider>
          <IndexBar
            embedModels={[{
              dims: 3,
              id: 'text-embedding-3-small',
              label: 'text-embedding-3-small',
              provider: 'OpenAI',
            }]}
            index={index}
            members={[]}
            onCancel={vi.fn()}
            onDelete={vi.fn()}
            onModel={vi.fn()}
            onReindex={vi.fn()}
            onResume={vi.fn()}
            onResumeRaw={vi.fn()}
            serverBacked
          />
        </TooltipProvider>
      </LocaleProvider>,
    )

    expect(markup).toContain('Vektoren (geschätzt)')
    expect(markup).toContain('Clientseitige Schätzung aus Seitenzahl oder Dateigröße')
    expect(markup).not.toMatch(/>Vektoren</)
  })

  it('keeps the running progress rail at a stable width independent of the document title', () => {
    const live: IndexingJobLive = {
      ...paused,
      currentDocumentTitle: 'A very long document title that must not resize the progress bar.pdf',
      phase: 'contextualization',
      source: 'server',
      status: 'running',
    }
    const markup = renderToStaticMarkup(
      <LocaleProvider>
        <TooltipProvider>
          <IndexBar
            embedModels={[{
              dims: 3,
              id: 'text-embedding-3-small',
              label: 'text-embedding-3-small',
              provider: 'OpenAI',
            }]}
            index={index}
            live={live}
            members={[]}
            onCancel={vi.fn()}
            onDelete={vi.fn()}
            onModel={vi.fn()}
            onReindex={vi.fn()}
            onResume={vi.fn()}
            onResumeRaw={vi.fn()}
            serverBacked
          />
        </TooltipProvider>
      </LocaleProvider>,
    )

    expect(markup).toContain('md:flex-nowrap')
    expect(markup).toContain('md:w-48')
    expect(markup).toContain('truncate text-right')
  })

  it('keeps the active generation explicit and exposes both deliberate recovery paths', () => {
    const markup = renderToStaticMarkup(
      <LocaleProvider>
        <TooltipProvider>
          <IndexBar
            embedModels={[{
              dims: 3,
              id: 'text-embedding-3-small',
              label: 'text-embedding-3-small',
              provider: 'OpenAI',
            }]}
            index={index}
            live={paused}
            members={[]}
            onCancel={vi.fn()}
            onDelete={vi.fn()}
            onModel={vi.fn()}
            onReindex={vi.fn()}
            onResume={vi.fn()}
            onResumeRaw={vi.fn()}
            serverBacked
          />
        </TooltipProvider>
      </LocaleProvider>,
    )

    expect(markup).toContain('Pausiert · Abhängigkeit')
    expect(markup).toContain('Provider did not answer before the network safety limit.')
    expect(markup).toContain('Phase contextualize · Batch 7/18')
    expect(markup).toContain('Aktiver Index unverändert')
    expect(markup).toContain('Fortsetzen')
    expect(markup).toContain('Ohne Kontext neu aufbauen')
    expect(markup).not.toContain('inqtrix-running-dot')
  })

  it.each([
    [true, 'zerlegt, kontextualisiert und eingebettet', 'Kontextanreicherung ist in dieser Umgebung deaktiviert'],
    [false, 'Kontextanreicherung ist in dieser Umgebung deaktiviert', 'zerlegt, kontextualisiert und eingebettet'],
  ] as const)(
    'describes the actual contextual-retrieval capability when it is %s',
    (contextualRetrievalEnabled, expected, rejected) => {
      const live: IndexingJobLive = {
        ...paused,
        phase: 'embedding',
        source: 'build',
        status: 'running',
      }
      const markup = renderToStaticMarkup(
        <LocaleProvider>
          <TooltipProvider>
            <IndexBar
              contextualRetrievalEnabled={contextualRetrievalEnabled}
              embedModels={[{
                dims: 3,
                id: 'text-embedding-3-small',
                label: 'text-embedding-3-small',
                provider: 'OpenAI',
              }]}
              index={index}
              live={live}
              members={[]}
              onCancel={vi.fn()}
              onDelete={vi.fn()}
              onModel={vi.fn()}
              onReindex={vi.fn()}
              onResume={vi.fn()}
              onResumeRaw={vi.fn()}
              serverBacked
            />
          </TooltipProvider>
        </LocaleProvider>,
      )

      expect(markup).toContain(expected)
      expect(markup).not.toContain(rejected)
    },
  )
})
