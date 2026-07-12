import { describe, expect, it } from 'vitest'

import { translations } from '@/i18n/translations'
import {
  agentActivityIconKind,
  activityDisplayText,
  discoveryProbeDisplay,
  normalizeAgentOperation,
  terminalActivityErrorIndex,
} from './activityPresentation'

describe('agent activity presentation', () => {
  const de = translations.de
  it('normalizes legacy capability ids without exposing them as labels', () => {
    expect(normalizeAgentOperation('knowledge.search')).toBe('knowledge_search')
    expect(activityDisplayText({
      count: 4,
      detail: 'knowledge.search',
      kind: 'searching',
      operation: 'knowledge_search',
    }, de)).toBe('Durchsucht Projektwissen · 4 Vorgänge')
  })

  it('assigns source-semantic timeline icon kinds', () => {
    expect(agentActivityIconKind('web_instant')).toBe('web')
    expect(agentActivityIconKind('knowledge_search')).toBe('knowledge')
    expect(agentActivityIconKind('knowledge_collections')).toBe('knowledge')
    expect(agentActivityIconKind('discovery_summary')).toBe('generic')
  })

  it('keeps the concrete purpose and deterministic progress', () => {
    expect(activityDisplayText({
      current: 2,
      detail: 'Regulatorische Auswirkungen in Deutschland',
      kind: 'searching',
      operation: 'knowledge_search',
      total: 4,
    }, de)).toBe(
      'Durchsucht Projektwissen · Regulatorische Auswirkungen in Deutschland · 2/4',
    )
  })

  it('uses one semantic presenter for discovery approvals', () => {
    expect(discoveryProbeDisplay({
      kind: 'web.search.instant',
      query: 'Welche Marktprognosen widersprechen sich?',
    }, de)).toEqual({
      detail: 'Welche Marktprognosen widersprechen sich?',
      title: 'Führt eine Instant-Websuche aus',
    })
  })

  it('leaves an unknown operation visible', () => {
    expect(discoveryProbeDisplay({ kind: 'custom.read' }, de).title).toBe('custom.read')
    expect(activityDisplayText({
      detail: '',
      kind: 'working',
      operationCode: 'vendor.custom.lookup',
    }, de)).toBe('vendor.custom.lookup')
  })

  it('keeps technical operation ids behind a human fallback explanation', () => {
    expect(activityDisplayText({
      detail: 'Veraltetes Task-Budget wird ignoriert',
      kind: 'working',
      operationCode: 'task.legacy_budget_ignored',
    }, de)).toBe('Veraltetes Task-Budget wird ignoriert')
  })

  it('renders operation result metrics without relabelling them as sources', () => {
    expect(activityDisplayText({
      detail: 'Welche Marktprognosen widersprechen sich?',
      kind: 'searching',
      metrics: { result_count: 4 },
      operation: 'web_instant',
    }, de)).toContain('4 Ergebnisse')
  })

  it('suppresses only the latest activity copy of a terminal task error', () => {
    const history = [
      { error: 'Provider timeout', status: 'failed' },
      { error: 'Recovered', status: 'completed' },
      { error: 'Provider timeout', status: 'failed' },
    ]
    expect(terminalActivityErrorIndex(history, 'Provider timeout')).toBe(2)
    expect(terminalActivityErrorIndex(history, 'Different error')).toBe(-1)
  })
})
