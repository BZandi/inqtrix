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

/** A saved tier preference, where `''` means "no preference" and the
 * deployment default applies — the same thing the picker shows as its
 * server-default entry. Lives beside {@link ChatModelTier} rather than next to
 * its consumers because both the theme layer and the project layer need it,
 * and this module imports nothing. */
export type ModelTierPreference = '' | ChatModelTier

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
  /** App version of the server build; absent on older servers. */
  version?: string
  legal?: {
    copyright: string
    license: string
    notice: string
    project: string
    source_url: string
    warranty_notice?: string
  }
  /** AI-transparency metadata; absent on older servers. `marker` and
   * `producer` are locale-neutral machine tokens and are the values the
   * client writes into `data-ai-*` attributes and exported document
   * properties. The two disclosure sentences are English-only: user-facing
   * surfaces render the localized `t.aiTransparency.*` wording instead. */
  ai_disclosure?: {
    marker: string
    producer: string
    system_disclosure: string
    content_disclosure: string
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

export type ResearchRunMode =
  | 'agent_kernel'
  | 'direct_llm'
  | 'knowledge'
  | 'research'
  | 'workspace_agent'

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
  reference_id: string
  chunk_id: string
  document_id: string
  collection_id: string
  document_title: string
  chunk_index: number
  /** Original source evidence only; synthetic retrieval context is never sent. */
  excerpt: string
  page_number: number | null
  source_span: {
    start: number
    end: number
    offset_unit: 'utf8_byte'
    document_content_hash: string
  } | null
  revision_id: string | null
  generation_id: string | null
  provenance_status: 'verified_span' | 'legacy_unspanned'
  rank: number
  score: number
}

/** Bounded technical shortfall reported by the shared Knowledge retrieval
 * layer. It is evidence metadata, not a provider error: results may still be
 * useful, but the requested breadth was not fully filled. */
export type KnowledgeRetrievalDegradation = {
  reason: string
  retrieval_mode: string
  /** Retrieval stage at which the bounded shortfall occurred. */
  stage: string
  /** Candidate depth requested before final ranking/reranking. */
  requested_candidate_pool: number
  /** Active, verified candidates actually available to final ranking. */
  returned_candidate_pool: number
  /** Requested width of the final evidence set. */
  final_top_k: number
  /** True when the shallower candidate pool still filled final evidence. */
  final_evidence_complete: boolean
  /** Compatibility counters; these describe the final evidence outcome. */
  requested_top_k: number
  returned_hits: number
  candidate_cap: number | null
}

/** One warning from the synchronous Knowledge-search envelope. Known
 * retrieval degradations carry the bounded counters above; other warnings
 * (visibility filtering, legacy chunks) retain their server message and
 * optional count without being collapsed into a retrieval claim. */
export type KnowledgeSearchWarning = Partial<KnowledgeRetrievalDegradation> & {
  code: string
  /** Search envelopes include a safe fallback message; text-free native run
   * receipts intentionally omit it and localize by code. */
  message?: string
  count?: number
  filtered_ids?: string[]
  recommended_action?: string | null
}

export type KnowledgeSearchResponse = {
  data: KnowledgeSearchHit[]
  warnings: KnowledgeSearchWarning[]
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

/** One capability-registry manifest entry (`capabilities.agent.tools`). */
export type AgentToolManifestEntry = {
  id: string
  summary: string
  effect: 'read' | 'write' | 'destructive'
  read_only: boolean
  idempotent: boolean
}

/** What ONE permission mode gates (`capabilities.agent.permission_modes`).
 * Generated server-side from the ENFORCING policy code (published ==
 * enforced) — the composer's run overview renders this verbatim and
 * never re-derives gating semantics client-side. */
export type AgentPermissionModeEntry = {
  /** The initial plan parks for approval (E16). */
  plan_gate: boolean
  /** A replan that adds web queries parks again for renewed consent. */
  web_replan_regate: boolean
  /** Editor patches always park — invariant across modes (E14). */
  patch_gate: boolean
  /** Kernel tools that unconditionally gate in this mode. */
  kernel_gated_tools: string[]
  /** Kernel tools that gate conditionally (e.g. only UN-scoped search). */
  kernel_conditional_tools: string[]
  /** Kernel tools gated in EVERY mode (write effects). */
  kernel_always_gated: string[]
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
    /** Whether GET /v1/stacks exists; only the multi-stack factory mounts it. */
    multi_stack: boolean
    openapi: boolean
    [key: string]: boolean
  }
  /** Additive operator-facing module state. `features.*` remains the simple
   * runtime gate; this map distinguishes deliberately disabled modules from
   * configured infrastructure that is currently degraded. */
  feature_status?: Record<string, {
    available: boolean
    configured: boolean
    reason_code: string | null
    state: 'enabled' | 'disabled' | 'degraded'
  }>
  collaboration?: {
    configured: boolean
    mode: string
    protocol_version: number
    schema_version: number
    service_available: boolean
    transport_path: string
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
    /** Configured vector backend NAME (e.g. "qdrant") — a descriptive
     * label for retrieval-source displays, not a reachability claim. */
    vector_backend?: string
  }
  /** Effective server-side HTTP wait deadlines (seconds). The client derives
   * its own AbortController timeouts from these (server wait + margin) instead
   * of hardcoding them, so a raised server-side timeout is not silently capped
   * by the browser. Optional: absent on older backends -> client falls back. */
  timeouts?: {
    editor_wait_seconds: number
    chat_wait_seconds: number
    text_wait_seconds: number
    reasoning_operation_seconds?: number
    editor_operation_seconds?: number
    search_operation_seconds?: number
    claim_extract_operation_seconds?: number
    research_run_seconds?: number
    max_attempts?: number
  }
  /** Workspace-agent limits + vocabulary (M5). Absent on backends without
   * the agent platform -> the Agent Desk stays hidden. */
  agent?: {
    autonomy_modes: string[]
    default_autonomy: string
    /** The effective desk algorithm: `workspace_agent`
     * (phase machine) or `agent_kernel`; absent on older servers ->
     * workspace_agent. */
    default_mode?: string
    /** Two-mode UI presets: the composer
     * shows Standard/Auto mapped onto the unchanged wire vocabulary.
     * Absent on older servers -> legacy three-way control. */
    mode_presets?: { id: string; autonomy: string }[]
    depth_modes?: { id: string }[]
    default_depth?: string
    /** Stufen ladder (speed/depth tiers), published from the server's
     * tier policy table (published == enforced). A tiers-aware composer
     * renders the Stufe control instead of the depth toggle; absent on
     * older servers -> legacy depth toggle. */
    tiers?: AgentTierCapability[]
    default_tier?: AgentTierId
    /** True republishes the legacy three-way control (incl. strict). */
    advanced_autonomy?: boolean
    max_parallel_children: number
    discovery_max_tool_calls: number
    max_plan_tasks: number
    /** Server-enforced Agent Desk boundaries. Values are descriptive facts,
     * never client-authored budget inputs. */
    limits?: AgentLimitCapabilities
    durable: boolean
    tools: AgentToolManifestEntry[]
    /** Source controls and direct routes are optional for compatibility with
     * servers predating the Agent Desk source dock. */
    source_controls?: Array<{
      id: 'web' | 'knowledge'
      default: 'available' | 'disabled'
      available: boolean
    }>
    execution_directives?: Array<{
      id: 'quick_web' | 'knowledge_only'
      available: boolean
    }>
    /** Per-mode gating facts for the run overview; absent on older
     * servers -> the overview hides its approvals group (never guesses). */
    permission_modes?: Record<string, AgentPermissionModeEntry>
    /** Skill limits; the skill list itself comes from the
     * authenticated GET /v1/skills. */
    skills?: {
      max_attached: number
      disclosure_budget_chars: number
    }
  }
}

export type AgentLimitCapabilities = {
  tokens: {
    enabled: boolean
    limit: number
    ceiling: number
    recoverable: false
    extendable: false
    reason: string
  }
  kernel: Record<'schnell' | 'normal' | 'deep', {
    tool_calls: number
    tool_calls_ceiling: number
    steps: number
    steps_ceiling: number
  }>
  directives: {
    quick_web: {
      web_searches: number
    }
  }
  mission: {
    discovery_tool_calls: number
    plan_tasks: number
    replan_rounds: number
    clarification_rounds: number
    parallel_children: number
  }
  research: {
    rounds: number
  }
}

export type KnowledgeCollectionInfo = {
  /** Authoritative caller relationship for this server collection. */
  access: ResearchRunAccess
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

export const DEEP_RESEARCH_FIRST_ROUND_QUERIES = 8
export const DEEP_RESEARCH_MAX_ROUNDS = 4

export type AgentTierId = 'schnell' | 'gruendlich' | 'tief'

/** One published Stufe (wire projection of the server's tier policy). */
export type AgentTierCapability = {
  id: AgentTierId
  clarification_rounds: number
  plan_gate: 'per_autonomy' | 'skip_unless_strict'
  web_research: boolean
  web_child_profile: string | null
  web_child_ceiling: string | null
  rag_default_profile: string
  verify: string
  response_form: 'auto' | 'chat' | 'canvas'
  latency_hint: string
}

export type AgentOverrides = {
  maxRounds?: number
  minRounds?: number
  confidenceStop?: number
  reportProfile?: 'schnell' | 'compact' | 'deep'
  maxTotalSeconds?: number
  firstRoundQueries?: number
  skipSearch?: boolean
  modelTier?: ChatModelTier
  /** Explicit model id from the model picker (bypasses tier for direct chat). */
  model?: string
  /** Reasoning effort for the picked model (model-dependent). */
  effort?: string
  /** Thoroughness: 'deep' = high effort, raised budgets,
   * DEEP child research and one verification pass. Legacy knob — a
   * tiers-aware composer sends `agentTier` instead (never both). */
  depth?: 'normal' | 'deep'
  /** Agent-Desk Stufe (speed/depth ladder); the server bridges it into
   * depth and reads the budgets from its tier policy table. */
  agentTier?: 'schnell' | 'gruendlich' | 'tief'
}

export type CreateResearchRunRequest = {
  question: string
  messages?: ResearchRunMessage[]
  stack?: string
  mode?: ResearchRunMode
  agentOverrides?: AgentOverrides
  /** Retrieval scope + profile; only meaningful with `mode: 'knowledge'`. */
  knowledgeFilters?: KnowledgeChatFilters
  /** Approval policy for `mode: 'workspace_agent'` (server default when
   * omitted; vocabulary from `capabilities.agent.autonomy_modes`). */
  autonomy?: string
  /** Saved Agent or Knowledge session that owns this run. */
  sessionId?: string
  /** Target editor document for a patch assignment (M7); the agent
   * proposes always-gated edits against it instead of only a memo. */
  documentId?: string
  /** Output-form override for `mode: 'workspace_agent'`:
   * `chat` forces the inline answer, `canvas` the memo; `auto` (or
   * omitted) lets the agent's intake decide. */
  responseForm?: 'auto' | 'chat' | 'canvas'
  /** Explicitly attached skill chips; the server
   * admits them (visibility + count cap). Agent modes only. */
  skillIds?: string[]
  /** Whitelisted composer tool hints from the `/`-functions group. */
  toolDirectives?: string[]
  /** Per-session availability policy for optional agent sources. */
  sourcePolicy?: import('@/features/agent/executionPolicy').AgentSourcePolicy
  /** One-message route selected through a direct slash command. */
  executionDirective?: import('@/features/agent/executionPolicy').AgentExecutionDirective
}

export type ResearchRunStatus =
  | 'queued'
  | 'running'
  | 'waiting_for_approval'
  | 'waiting_for_input'
  | 'waiting_for_children'
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
  /** Effective Agent Desk route metadata. Older runs omit this block. */
  execution?: {
    execution_directive?: 'quick_web' | 'knowledge_only' | null
    directive?: 'quick_web' | 'knowledge_only' | null
    effective_mode?: string
    response_form?: string
    depth?: string
    model?: string | null
    reasoning_effort?: string | null
    source_policy?: {
      web: 'available' | 'disabled'
      knowledge: 'available' | 'disabled'
    }
    consent_reason?: string | null
    tool_use_counts?: { web?: number; knowledge?: number }
    limits?: Record<string, {
      used?: number | null
      limit: number
      ceiling: number
      recoverable: boolean
      extendable: boolean
      reason?: string
    }>
  }
}

/** Canonical caller relationship emitted by every resource list. */
export type ResearchRunAccess = ResourceAccess

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
  /** Present for agent runs whose execution can park and resume. Wall time
   * remains anchored at the immutable first `started_at`; these counters
   * separate actual execution, explicit waits and queueing. */
  timing?: {
    total_seconds: number | null
    active_seconds: number
    waiting_seconds: number
    queued_seconds: number
    segment_count: number
    resume_count: number
    current_segment_id: string | null
  }
  snapshot: ResearchRunSnapshot
  error: InqtrixError | null
  events_url: string
  result_url: string
  access: ResearchRunAccess
  /** Emitted only as `true` while a cancel of a still-running run is
   * pending (the status stays `running` until the worker stops). */
  cancel_requested?: boolean
  /** Run tree/session extras — emitted only when non-default (agent runs). */
  kind?: 'standard' | 'agent' | 'agent_child'
  parent_run_id?: string
  root_run_id?: string
  session_id?: string
  children_url?: string
  plan_url?: string
  artifacts_url?: string
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
  /** Stable evidence-ledger identity. Optional for imported and legacy runs. */
  reference_id?: string | null
  source_id?: string | null
  query_id?: string | null
  query_ids?: string[]
  citation_id?: string | null
  citation_ids?: string[]
  source_run_id?: string | null
  source_run_ids?: string[]
  provider_snippet?: string | null
  document_id?: string | null
  collection_id?: string | null
  chunk_id?: string | null
  chunk_index?: number | null
  excerpt?: string | null
  source_text?: string | null
  grounded_support?: string | null
  page_number?: number | null
  source_span?: {
    start: number
    end: number
    offset_unit: 'utf8_byte' | string
    document_content_hash?: string | null
  } | null
  revision_id?: string | null
  generation_id?: string | null
  provenance_status?: string | null
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
    marker?: string
    quotes_total?: number
    quotes_verified?: number
    quotes?: Array<{ label: string; text: string; verified: boolean }>
  }
  report_references?: ReportReference[]
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
  knowledge_retrieval?: {
    degradations?: KnowledgeRetrievalDegradation[]
    warnings?: KnowledgeSearchWarning[]
  }
}

export type InqtrixError = {
  available_stacks?: string[]
  message: string
  status?: string
  type: string
}
import type { ResourceAccess } from '@/features/sharing/types'
