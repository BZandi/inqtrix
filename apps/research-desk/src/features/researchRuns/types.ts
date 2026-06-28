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

export type ModelCardCategory = 'high' | 'mid' | 'fast'

export type ModelCard = {
  id: string
  display_name: string
  vendor: string
  category: ModelCardCategory
  speed: 'langsam' | 'mittel' | 'schnell'
  description: string
  context_window_tokens: number
  max_output_tokens: number
  reasoning_levels: string[]
  capabilities: string[]
  input_modalities: string[]
  knowledge_cutoff?: string | null
  pricing: { input_per_mtok: number; output_per_mtok: number }
}

export type ModelCatalogEntry = {
  model_id: string
  // null when the backend has no curated card for this id (degrade gracefully)
  card: ModelCard | null
}

export type InqtrixHealth = {
  status: 'ok' | 'degraded'
  auth_required: boolean
  /** Active auth mode; absent on older servers (then `auth_required`
   * implies `apikey`). `oidc` switches the UI to the SSO login;
   * `local`/`ldap` switch it to the credential form + owner setup. */
  auth_mode?: 'none' | 'apikey' | 'oidc' | 'local' | 'ldap'
  classify_model?: string
  chat_model_options?: ChatModelOption[]
  models_catalog?: ModelCatalogEntry[]
  context_window_tokens?: number | null
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
    models_catalog?: ModelCatalogEntry[]
    context_window_tokens?: number | null
  }
  llm_provider?: string
  search_provider?: string
}

export type ResearchRunMode = 'direct_llm' | 'knowledge' | 'research'

export type ResearchRunMessage = {
  content: string
  role: 'assistant' | 'system' | 'user'
}

/** Scope filters for `mode: 'knowledge'` requests. Serialized to the
 * backend `knowledge_filters` body field. */
export type KnowledgeChatFilters = {
  collectionIds: string[]
  topK?: number
  /** Surfaced-evidence count override (`final_k`). Omitted = the profile's
   * factor applies. Serialized to `knowledge_filters.final_k`. */
  finalK?: number
  /** Retrieval profile id (`schnell` | `standard` | `gruendlich` | `tief`
   * | `auto`). Omitted = server default. Valid ids come from the
   * capability manifest (`knowledge.profiles`), never a hardcoded list. */
  profile?: string
}

/** One entry of `capabilities.knowledge.profiles`. Concrete profiles
 * carry `stages` + `degraded`; the `auto` entry only `delegates_to`. */
export type KnowledgeProfileManifestEntry = {
  id: string
  /** Multiplier on the request `top_k` for the FINAL surfaced-evidence count;
   * lets the client render the effective `final_k` per profile (only `tief`
   * raises it above 1.0). */
  final_k_factor?: number
  stages?: {
    rerank: boolean
    gate_rounds: number
    grounding: boolean
    vocabulary_bridge: boolean
    decompose: boolean
    report: boolean
  }
  degraded?: string[]
  delegates_to?: string[]
}

/** One retrieval hit from `POST /v1/knowledge/search`. */
export type KnowledgeSearchHit = {
  document_id: string
  collection_id: string
  document_title: string
  chunk_index: number
  text: string
  score: number
}

/** Payload of `GET /v1/knowledge/documents/{id}/text` (document reader). */
export type KnowledgeDocumentText = {
  id: string
  collection_id: string
  title: string
  /** May contain `file_id` (original binary) when server-file ingested. */
  metadata: Record<string, unknown>
  chunk_count: number
  created_at: number
  text: string
}

export type EmbeddingCardInfo = {
  dims: number
  display_name: string
  id: string
  last_verified: string
  max_input_tokens: number
  multilingual: boolean
  pricing_input_per_mtok: number | null
  source_url: string
  vendor: string
}

export type EmbeddingCatalogEntry = {
  card: EmbeddingCardInfo | null
  model_id: string
}

export type AlgorithmManifestEntry = {
  display_name: string
  id: string
  produces?: string[]
  requires?: string[]
  streams_events?: boolean
  supports_chat_completions?: boolean
  [key: string]: unknown
}

/** GET /v1/capabilities — feature discovery so the UI never hardcodes
 * which algorithms/backends a deployment offers. `null` from the hook
 * means the endpoint is absent (older backend) and every knowledge
 * affordance stays hidden. */
export type InqtrixCapabilities = {
  algorithms: AlgorithmManifestEntry[]
  features: {
    embedding_provider: boolean
    knowledge: boolean
    openapi: boolean
    [key: string]: boolean
  }
  files?: {
    max_file_bytes: number
  }
  knowledge?: {
    default_embedding_model: string
    default_top_k: number
    /** Hard ceiling on the FINAL evidence count (`final_k`); bounds the
     * client's final_k override field. */
    evidence_k_max?: number
    embedding_catalog: EmbeddingCatalogEntry[]
    /** Selectable retrieval profiles; absent on backends without the
     * profile engine — the picker then stays hidden. */
    profiles?: KnowledgeProfileManifestEntry[]
    default_profile?: string
    /** Configured reranker provider id, shown in the run overview when the
     * rerank stage is active. */
    reranker_provider?: string
  }
  /** Effective server-side HTTP wait deadlines (seconds). The client derives
   * its own AbortController timeouts from these (server wait + margin) instead
   * of hardcoding them, so a raised server-side timeout is not silently capped
   * by the browser. Optional: absent on older backends -> client falls back. */
  timeouts?: {
    editor_wait_seconds: number
    chat_wait_seconds: number
    text_wait_seconds: number
  }
}

export type KnowledgeCollectionInfo = {
  created_at: number
  document_count: number
  embedding_dim: number
  embedding_model: string
  id: string
  name: string
}

export type KnowledgeDocumentInfo = {
  chunk_count: number
  collection_id: string
  created_at: number
  id: string
  metadata: Record<string, unknown>
  title: string
}

export type AgentOverrides = {
  maxRounds?: number
  minRounds?: number
  confidenceStop?: number
  reportProfile?: 'compact' | 'deep'
  maxTotalSeconds?: number
  firstRoundQueries?: number
  skipSearch?: boolean
  modelTier?: ChatModelTier
  /** Explicit model id from the model picker (bypasses tier for direct chat). */
  model?: string
  /** Reasoning effort for the picked model (model-dependent). */
  effort?: string
}

export type CreateResearchRunRequest = {
  question: string
  messages?: ResearchRunMessage[]
  stack?: string
  mode?: ResearchRunMode
  agentOverrides?: AgentOverrides
  /** Retrieval scope + profile; only meaningful with `mode: 'knowledge'`. */
  knowledgeFilters?: KnowledgeChatFilters
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

/**
 * Additive shared-in annotation on a run summary. Owned runs omit the
 * key entirely (historical wire shape); shared-in runs carry the
 * grant level so the UI can hide cancel/delete for view-only access.
 */
export type ResearchRunAccess = {
  permission: 'edit' | 'view'
  via: 'share'
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
  access?: ResearchRunAccess
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
  title?: string | null
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
  // -- knowledge-mode extras (result_state projection) --
  // All optional and read defensively: the export payload guarantees
  // only the research-shaped keys above; knowledge deployments add the
  // block below. Absence degrades the UI (no quote highlighting, no
  // gate meta), it never breaks it.
  queries?: string[]
  knowledge_gate?: {
    enabled: boolean
    sufficient?: boolean
    reason?: string
    rounds_used?: number
    max_rounds?: number
    second_pass?: boolean
  }
  knowledge_grounding?: {
    enabled: boolean
    quotes_total?: number
    quotes_verified?: number
    quotes?: Array<{ label: string; text: string; verified: boolean }>
  }
  report_references?: Array<{
    label: string
    url: string
    tier: string
    title?: string
    document_id?: string | null
    chunk_index?: number | null
    excerpt?: string | null
    source_text?: string | null
    page_number?: number | null
  }>
  knowledge_profile?: {
    id: string
    requested?: string | null
    auto_selected?: boolean
    auto_reason?: string | null
    degraded_stages?: string[]
  }
  knowledge_candidates?: number
  knowledge_evidence_used?: number
  knowledge_collections?: string[]
}

export type InqtrixError = {
  available_stacks?: string[]
  message: string
  status?: string
  type: string
}
