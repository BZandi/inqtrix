/** Persistent source availability for one Agent Desk session. */
export type AgentSourcePolicyState = 'available' | 'disabled'

export type AgentSourcePolicy = {
  web: AgentSourcePolicyState
  knowledge: AgentSourcePolicyState
}

/** One-message execution route selected from the slash command menu. */
export type AgentExecutionDirective = 'quick_web' | 'knowledge_only'

export type AgentExecutionSnapshot = {
  executionDirective: AgentExecutionDirective | null
  effectiveMode: string | null
  responseForm: string | null
  depth: string | null
  model: string | null
  reasoningEffort: string | null
  sourcePolicy: AgentSourcePolicy
  consentReason: string | null
  toolUseCounts: { web: number; knowledge: number }
}

export type AgentExecutionDisplay = {
  executionDirective: AgentExecutionDirective | null
  effectiveMode: string
  responseForm: string
  depth: string
}

export const DEFAULT_AGENT_SOURCE_POLICY: AgentSourcePolicy = {
  web: 'available',
  knowledge: 'available',
}

/**
 * Effective status values before and after admission. A pending one-message
 * directive describes the next submission and therefore temporarily wins
 * over an older completed-run snapshot. The directive clears on admission;
 * from that point the accepted canonical snapshot becomes authoritative.
 */
export function resolveAgentExecutionDisplay({
  execution,
  pendingDirective,
  selectedDepth,
  selectedMode,
  selectedResponseForm,
}: {
  execution: AgentExecutionSnapshot | null
  pendingDirective: AgentExecutionDirective | null
  selectedDepth: string
  selectedMode: string
  selectedResponseForm: string
}): AgentExecutionDisplay {
  if (pendingDirective) {
    return {
      executionDirective: pendingDirective,
      effectiveMode: 'agent_kernel',
      responseForm: 'chat',
      depth: 'normal',
    }
  }
  if (execution) {
    return {
      executionDirective: execution.executionDirective,
      effectiveMode: execution.effectiveMode ?? selectedMode,
      responseForm: execution.responseForm ?? selectedResponseForm,
      depth: execution.depth ?? selectedDepth,
    }
  }
  return {
    executionDirective: null,
    effectiveMode: selectedMode,
    responseForm: selectedResponseForm,
    depth: selectedDepth,
  }
}

export function normalizeAgentSourcePolicy(value: unknown): AgentSourcePolicy {
  if (!value || typeof value !== 'object') {
    return { ...DEFAULT_AGENT_SOURCE_POLICY }
  }
  const candidate = value as Record<string, unknown>
  return {
    web: candidate.web === 'disabled' ? 'disabled' : 'available',
    knowledge:
      candidate.knowledge === 'disabled' ? 'disabled' : 'available',
  }
}

const DIRECTIVE_BY_TOKEN: Record<string, AgentExecutionDirective> = {
  web: 'quick_web',
  wissen: 'knowledge_only',
}

/**
 * Converts an exact trailing direct command into a one-message directive.
 * Slashes inside URLs and prose are untouched; the command must be its own
 * final whitespace-delimited token.
 */
export function extractTrailingExecutionDirective(value: string): {
  directive: AgentExecutionDirective
  question: string
} | null {
  const match = /(?:^|\s)\/(web|wissen)\s*$/i.exec(value)
  if (!match) return null
  const directive = DIRECTIVE_BY_TOKEN[match[1].toLowerCase()]
  const question = value.slice(0, match.index).trim()
  return { directive, question }
}

export function executionDirectiveFromSnapshot(
  snapshot: unknown,
): AgentExecutionDirective | null {
  return normalizeAgentExecutionSnapshot(snapshot)?.executionDirective ?? null
}

export function toolUseCountsFromSnapshot(snapshot: unknown): {
  web: number
  knowledge: number
} | null {
  return normalizeAgentExecutionSnapshot(snapshot)?.toolUseCounts ?? null
}

/** Canonical adapter for the server-published execution block. All status
 * surfaces consume this one normalized value rather than reading snapshot
 * keys independently. */
export function normalizeAgentExecutionSnapshot(
  snapshot: unknown,
): AgentExecutionSnapshot | null {
  if (!snapshot || typeof snapshot !== 'object') return null
  const execution = (snapshot as Record<string, unknown>).execution
  if (!execution || typeof execution !== 'object') return null
  const record = execution as Record<string, unknown>
  const rawDirective = record.execution_directive ?? record.directive
  const executionDirective =
    rawDirective === 'quick_web' || rawDirective === 'knowledge_only'
      ? rawDirective
      : null
  const rawCounts = record.tool_use_counts
  const counts = rawCounts && typeof rawCounts === 'object'
    ? rawCounts as Record<string, unknown>
    : {}
  const stringOrNull = (value: unknown): string | null =>
    typeof value === 'string' && value.length > 0 ? value : null
  return {
    executionDirective,
    effectiveMode: stringOrNull(record.effective_mode ?? record.mode),
    responseForm: stringOrNull(record.response_form),
    depth: stringOrNull(record.depth),
    model: stringOrNull(record.model),
    reasoningEffort: stringOrNull(record.reasoning_effort),
    sourcePolicy: normalizeAgentSourcePolicy(record.source_policy),
    consentReason: stringOrNull(record.consent_reason ?? record.consent),
    toolUseCounts: {
      web:
        typeof counts.web === 'number' && counts.web >= 0
          ? counts.web
          : typeof record.web_searches === 'number' && record.web_searches >= 0
            ? record.web_searches
            : 0,
      knowledge:
        typeof counts.knowledge === 'number' && counts.knowledge >= 0
          ? counts.knowledge
          : 0,
    },
  }
}
