export type NodeModelResolution = {
  node?: string
  model?: string
  tier?: string
  effort?: string
  model_source?: string
  effort_source?: string
  requested_tier?: string
}

export type ChatModelTier = 'high' | 'mid' | 'fast'

export type ChatModelOption = NodeModelResolution & {
  tier: ChatModelTier | string
}

export type InqtrixHealth = {
  status: 'ok' | 'degraded'
  auth_required: boolean
  classify_model?: string
  chat_model_options?: ChatModelOption[]
  evaluate_model?: string
  model_tier?: ChatModelTier | ''
  report_profile?: string
  reasoning_model?: string
  search_model?: string
  summarize_model?: string
  node_models?: Record<string, NodeModelResolution>
  testing_mode?: boolean
  high_risk_score_threshold?: number
  legal?: {
    copyright: string
    license: string
    notice: string
    project: string
    source_url: string
    warranty_notice?: string
  }
  llm: {
    provider: string
    status: 'ready' | 'unavailable'
  }
  search: {
    provider: string
    status: 'ready' | 'unavailable'
  }
}

export type InqtrixStackList = {
  default: string
  stacks: InqtrixStack[]
}

export type StackDiscoveryStatus = 'available' | 'unknown' | 'unsupported'

export type InqtrixStack = {
  name: string
  description?: string
  llm?: string
  search?: string
  ready: boolean
  models?: {
    classify_model?: string
    evaluate_model?: string
    reasoning_model?: string
    search_model?: string
    summarize_model?: string
    node_models?: Record<string, NodeModelResolution>
    chat_model_options?: ChatModelOption[]
  }
  llm_provider?: string
  search_provider?: string
}

export type ResearchRunMode = 'direct_llm' | 'research'

export type AgentOverrides = {
  maxRounds?: number
  minRounds?: number
  confidenceStop?: number
  reportProfile?: 'compact' | 'deep'
  maxTotalSeconds?: number
  firstRoundQueries?: number
  skipSearch?: boolean
  modelTier?: ChatModelTier
}

export type CreateResearchRunRequest = {
  question: string
  stack?: string
  mode?: ResearchRunMode
  agentOverrides?: AgentOverrides
}

export type ResearchRunStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'expired'

export type ResearchRunSnapshot = {
  current_node?: string
  completed_rounds?: number
  active_round?: number
  max_rounds?: number
  total_queries?: number
  total_citations?: number
  total_sources?: number
  confidence?: number
  source_tier_counts?: Record<SourceTier | string, number>
  source_quality_score?: number
  claim_status_counts?: Record<string, number>
  claim_quality_score?: number
  evidence_record_count?: number
  consolidated_claim_count?: number
  aspect_coverage?: number
  evidence_consistency?: number
  evidence_sufficiency?: number
  done?: boolean
  progress_estimate?: number
  last_message?: string
}

export type ResearchRunSummary = {
  run_id: string
  status: ResearchRunStatus
  queue_position: number | null
  question: string
  stack: string
  workspace_id?: string | null
  mode: ResearchRunMode
  agent_overrides: Record<string, unknown>
  created_at: number
  started_at: number | null
  finished_at: number | null
  elapsed_seconds: number | null
  snapshot: ResearchRunSnapshot
  error: InqtrixError | null
  events_url: string
  result_url: string
}

export type ResearchRunEvent = {
  type: string
  run_id: string
  sequence: number
  created_at: number
  data: Record<string, unknown>
}

export type SourceTier = 'primary' | 'mainstream' | 'stakeholder' | 'unknown' | 'low'

export type ResearchSource = {
  url: string
  tier: SourceTier | string
}

export type ReportReference = {
  label: string
  url: string
  tier: SourceTier | string
}

export type ResearchClaim = {
  text: string
  status: 'verified' | 'contested' | 'unverified' | string
  claim_type: 'fact' | 'actor_claim' | 'forecast' | string
  needs_primary: boolean
  status_reason: string
  support_count: number
  contradict_count: number
  source_tier_counts: Record<SourceTier | string, number>
  sources: string[]
}

export type SourceMetrics = {
  tier_counts: Record<SourceTier | string, number>
  quality_score: number
}

export type ClaimMetrics = {
  status_counts: Record<string, number>
  quality_score: number
}

export type ResearchMetrics = {
  answer_bound_claims_count: number
  aspect_coverage: number
  claims: ClaimMetrics
  completion_tokens: number
  confidence: number
  elapsed_seconds: number
  evidence_consistency: number
  evidence_contract_status: string
  evidence_sufficiency: number
  prompt_tokens: number
  rounds: number
  sources: SourceMetrics
  total_citations: number
  total_queries: number
  unbound_answer_citations_count: number
  verified_claims_used_count: number
}

export type ResearchRunResult = {
  run_id: string
  status: 'completed'
  answer: string
  metrics: ResearchMetrics
  references?: ReportReference[]
  top_sources: ResearchSource[]
  top_claims: ResearchClaim[]
  session_id?: string | null
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

export type InqtrixError = {
  available_stacks?: string[]
  message: string
  status?: string
  type: string
}
