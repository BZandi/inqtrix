import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { AdminSystemRuntime } from '@/api/inqtrixClient'
import { TooltipProvider } from '@/components/ui/tooltip'
import { LocaleProvider } from '@/i18n/LocaleProvider'
import { SystemStatusPanel } from './SystemStatusPanel'

function runtimeFixture(
  overrides: {
    runs?: Partial<AdminSystemRuntime['runs']>
    observability?: Partial<AdminSystemRuntime['observability']>
  } = {},
): AdminSystemRuntime {
  return {
    api: { openapi: false },
    files: {
      blob_storage: 'postgres',
      enabled: true,
      max_file_bytes: null,
      object_store: 'local',
      object_store_available: true,
    },
    knowledge: {
      contextual_retrieval: false,
      default_top_k: null,
      document_parser: 'markitdown',
      embedding_model: null,
      embedding_provider: null,
      enabled: false,
      hybrid_retrieval: false,
      reranker: 'none',
      sparse: null,
      vector_store: 'memory',
      vector_store_available: true,
    },
    runs: {
      execution: 'in_process',
      queue: 'memory',
      queue_available: true,
      queue_consumers: null,
      queue_depth: null,
      store: 'postgres',
      worker_dispatch: false,
      ...overrides.runs,
    },
    storage: { backend: 'postgres', durable: true },
    observability: {
      tracing: 'off',
      tracing_active: false,
      content_capture: false,
      sample_rate: 1,
      spool: false,
      retention_enforced: false,
      retention_days: null,
      ui_link_configured: false,
      ...overrides.observability,
    },
  }
}

function renderPanel(runtime: AdminSystemRuntime): string {
  return renderToStaticMarkup(
    <LocaleProvider>
      <TooltipProvider>
        <SystemStatusPanel
          capabilities={null}
          health={null}
          runtime={runtime}
          runtimeError={null}
          runtimeStatus="ready"
        />
      </TooltipProvider>
    </LocaleProvider>,
  )
}

describe('SystemStatusPanel queue telemetry', () => {
  it('warns when the dispatch queue has no recently active consumers', () => {
    const markup = renderPanel(
      runtimeFixture({
        runs: {
          execution: 'worker_dispatch',
          queue: 'valkey',
          queue_consumers: 0,
          queue_depth: 3,
          worker_dispatch: true,
        },
        observability: { retention_enforced: true },
      }),
    )
    expect(markup).toContain('Keine Konsumenten')
    expect(markup).toContain('3 Nachrichten')
  })

  it('stays silent about consumers for in-process execution', () => {
    const markup = renderPanel(runtimeFixture())
    expect(markup).not.toContain('Konsumenten')
  })
})

describe('SystemStatusPanel retention enforcement', () => {
  it('flags configured retention that no worker enforces', () => {
    const markup = renderPanel(
      runtimeFixture({
        observability: {
          tracing: 'otlp',
          tracing_active: true,
          retention_enforced: false,
          retention_days: 30,
        },
      }),
    )
    expect(markup).toContain('Ohne Worker nicht durchgesetzt')
  })

  it('shows no retention warning when the worker enforces it', () => {
    const markup = renderPanel(
      runtimeFixture({
        runs: {
          execution: 'worker_dispatch',
          queue: 'valkey',
          queue_consumers: 2,
          queue_depth: 0,
          worker_dispatch: true,
        },
        observability: {
          tracing: 'otlp',
          tracing_active: true,
          retention_enforced: true,
          retention_days: 30,
        },
      }),
    )
    expect(markup).not.toContain('Ohne Worker nicht durchgesetzt')
  })
})
