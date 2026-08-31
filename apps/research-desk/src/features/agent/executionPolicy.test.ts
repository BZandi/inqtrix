import { describe, expect, it } from 'vitest'
import {
  extractTrailingExecutionDirective,
  executionDirectiveFromSnapshot,
  normalizeAgentSourcePolicy,
  normalizeAgentExecutionSnapshot,
  resolveAgentExecutionDisplay,
  toolUseCountsFromSnapshot,
} from './executionPolicy'

describe('extractTrailingExecutionDirective', () => {
  it('converts exact trailing direct commands and removes them from the question', () => {
    expect(extractTrailingExecutionDirective('Wer gewann heute? /web')).toEqual({
      directive: 'quick_web',
      question: 'Wer gewann heute?',
    })
    expect(extractTrailingExecutionDirective('Fasse den Bestand zusammen /WISSEN  ')).toEqual({
      directive: 'knowledge_only',
      question: 'Fasse den Bestand zusammen',
    })
  })

  it('does not reinterpret paths, inline tokens, or unknown commands', () => {
    expect(extractTrailingExecutionDirective('https://example.com/web')).toBeNull()
    expect(extractTrailingExecutionDirective('Nutze /web und antworte')).toBeNull()
    expect(extractTrailingExecutionDirective('Frage /recherche')).toBeNull()
  })
})

describe('normalizeAgentSourcePolicy', () => {
  it('defaults absent and malformed metadata to both sources available', () => {
    expect(normalizeAgentSourcePolicy(null)).toEqual({
      web: 'available',
      knowledge: 'available',
    })
    expect(normalizeAgentSourcePolicy({ web: 'mystery', knowledge: 1 })).toEqual({
      web: 'available',
      knowledge: 'available',
    })
  })

  it('preserves explicit disabled states independently', () => {
    expect(normalizeAgentSourcePolicy({ web: 'disabled' })).toEqual({
      web: 'disabled',
      knowledge: 'available',
    })
  })
})

describe('executionDirectiveFromSnapshot', () => {
  it('reads accepted route metadata without guessing unknown values', () => {
    expect(executionDirectiveFromSnapshot({
      execution: { execution_directive: 'quick_web' },
    })).toBe('quick_web')
    expect(executionDirectiveFromSnapshot({ execution: { directive: 'other' } })).toBeNull()
  })
})

describe('normalizeAgentExecutionSnapshot', () => {
  it('normalizes the canonical effective execution contract', () => {
    expect(normalizeAgentExecutionSnapshot({
      execution: {
        execution_directive: 'quick_web',
        effective_mode: 'agent_kernel',
        response_form: 'chat',
        depth: 'normal',
        model: 'model-x',
        reasoning_effort: 'high',
        source_policy: { web: 'available', knowledge: 'disabled' },
        consent_reason: 'explicit_directive',
        tool_use_counts: { web: 1, knowledge: 0 },
        limits: {
          tool_calls: {
            used: 12,
            limit: 30,
            ceiling: 60,
            recoverable: true,
            extendable: true,
          },
          tokens: {
            used: 900,
            limit: 1000,
            ceiling: 1000,
            recoverable: false,
            extendable: false,
            reason: 'operator_ceiling_exactly_once_required',
          },
        },
      },
    })).toEqual({
      executionDirective: 'quick_web',
      effectiveMode: 'agent_kernel',
      responseForm: 'chat',
      depth: 'normal',
      model: 'model-x',
      reasoningEffort: 'high',
      sourcePolicy: { web: 'available', knowledge: 'disabled' },
      consentReason: 'explicit_directive',
      toolUseCounts: { web: 1, knowledge: 0 },
      limits: {
        tool_calls: {
          used: 12,
          limit: 30,
          ceiling: 60,
          recoverable: true,
          extendable: true,
          reason: null,
        },
        tokens: {
          used: 900,
          limit: 1000,
          ceiling: 1000,
          recoverable: false,
          extendable: false,
          reason: 'operator_ceiling_exactly_once_required',
        },
      },
      toolGrants: [],
    })
  })

  it('drops malformed limits instead of inventing operator policy', () => {
    const normalized = normalizeAgentExecutionSnapshot({
      execution: {
        limits: {
          broken: { limit: '30' },
          fixed: { used: -1, limit: 8, ceiling: 4, extendable: true },
        },
      },
    })
    expect(normalized?.limits).toEqual({})
  })
})

describe('toolUseCountsFromSnapshot', () => {
  it('reads only finite non-negative source counts', () => {
    expect(toolUseCountsFromSnapshot({
      execution: { tool_use_counts: { web: 1, knowledge: 2 } },
    })).toEqual({ web: 1, knowledge: 2 })
    expect(toolUseCountsFromSnapshot({})).toBeNull()
  })
})

describe('resolveAgentExecutionDisplay', () => {
  it('previews quick web as Kernel, Chat, and Normal without using sticky Mission/Canvas/Deep', () => {
    expect(resolveAgentExecutionDisplay({
      execution: null,
      pendingDirective: 'quick_web',
      selectedMode: 'workspace_agent',
      selectedResponseForm: 'canvas',
      selectedDepth: 'deep',
    })).toEqual({
      executionDirective: 'quick_web',
      effectiveMode: 'agent_kernel',
      responseForm: 'chat',
      depth: 'normal',
    })
  })

  it('previews knowledge only with the same conversational execution contract', () => {
    expect(resolveAgentExecutionDisplay({
      execution: null,
      pendingDirective: 'knowledge_only',
      selectedMode: 'workspace_agent',
      selectedResponseForm: 'canvas',
      selectedDepth: 'deep',
    })).toMatchObject({
      executionDirective: 'knowledge_only',
      effectiveMode: 'agent_kernel',
      responseForm: 'chat',
      depth: 'normal',
    })
  })

  it('previews a new one-shot directive over an older completed-run snapshot', () => {
    const execution = normalizeAgentExecutionSnapshot({
      execution: {
        execution_directive: 'quick_web',
        effective_mode: 'agent_kernel',
        response_form: 'chat',
        depth: 'normal',
        source_policy: { web: 'available', knowledge: 'disabled' },
        tool_use_counts: { web: 1, knowledge: 0 },
      },
    })
    expect(resolveAgentExecutionDisplay({
      execution,
      pendingDirective: 'knowledge_only',
      selectedMode: 'workspace_agent',
      selectedResponseForm: 'canvas',
      selectedDepth: 'deep',
    })).toMatchObject({
      executionDirective: 'knowledge_only',
      effectiveMode: 'agent_kernel',
      responseForm: 'chat',
      depth: 'normal',
    })
  })

  it('uses the accepted canonical snapshot after the pending directive clears', () => {
    const execution = normalizeAgentExecutionSnapshot({
      execution: {
        execution_directive: 'quick_web',
        effective_mode: 'agent_kernel',
        response_form: 'chat',
        depth: 'normal',
        source_policy: { web: 'available', knowledge: 'disabled' },
        tool_use_counts: { web: 1, knowledge: 0 },
      },
    })
    expect(resolveAgentExecutionDisplay({
      execution,
      pendingDirective: null,
      selectedMode: 'workspace_agent',
      selectedResponseForm: 'canvas',
      selectedDepth: 'deep',
    }).executionDirective).toBe('quick_web')
  })
})

describe('toolGrants normalization (P6B)', () => {
  it('reads run-wide grants and keeps only non-empty strings', () => {
    const snapshot = normalizeAgentExecutionSnapshot({
      execution: {
        execution_directive: '',
        effective_mode: 'agent_kernel',
        response_form: 'chat',
        depth: 'normal',
        model: '',
        reasoning_effort: '',
        source_policy: { web: 'available', knowledge: 'available' },
        consent_reason: '',
        tool_use_counts: { web: 0, knowledge: 0 },
        limits: {},
        tool_grants: ['web_instant', '', 7, 'load_skill'],
      },
    })
    expect(snapshot?.toolGrants).toEqual(['web_instant', 'load_skill'])
  })

  it('defaults to no grants on older servers without the key', () => {
    const snapshot = normalizeAgentExecutionSnapshot({
      execution: {
        execution_directive: '',
        effective_mode: 'agent_kernel',
        response_form: 'chat',
        depth: 'normal',
        model: '',
        reasoning_effort: '',
        source_policy: { web: 'available', knowledge: 'available' },
        consent_reason: '',
        tool_use_counts: { web: 0, knowledge: 0 },
        limits: {},
      },
    })
    expect(snapshot?.toolGrants).toEqual([])
  })
})
