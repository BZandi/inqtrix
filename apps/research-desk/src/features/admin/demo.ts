import type {
  AdminSystemRuntime,
  AccessToken,
  AdminUser,
  AdminWorkspace,
  WorkspaceMember,
  WorkspaceRoleValue,
} from '@/api/inqtrixClient'
import type {
  InqtrixCapabilities,
  InqtrixHealth,
} from '@/features/researchRuns/types'
import { DEMO_OWNER } from '@/features/sharing/demoShares'

// 2026-01-01T00:00:00Z — the neutral fixed instant used across the demo
// digital twin (anonymised demo data; never Date.now()).
const DEMO_TS = 1_767_225_600
const DAY = 86_400

/**
 * Seeded instance users for the demo admin surface (digital twin). The
 * owner row IS `DEMO_OWNER` so the "you" badge + self-disable lock line up
 * with the demo session; the mix (a second admin, a disabled account)
 * exercises every cell state and guard the Users panel renders.
 */
export function seedAdminUsers(): AdminUser[] {
  return [
    {
      subject: DEMO_OWNER.subject,
      email: DEMO_OWNER.email,
      display_name: DEMO_OWNER.displayName,
      instance_role: 'admin',
      disabled: false,
      last_login_at: DEMO_TS - 2 * DAY,
    },
    {
      subject: 'user-ines',
      email: 'ines.adesina@example.com',
      display_name: 'Ines Adesina',
      instance_role: 'admin',
      disabled: false,
      last_login_at: DEMO_TS - 1 * DAY,
    },
    {
      subject: 'user-mara',
      email: 'mara.lindqvist@example.com',
      display_name: 'Mara Lindqvist',
      instance_role: 'user',
      disabled: false,
      last_login_at: DEMO_TS - 5 * DAY,
    },
    {
      subject: 'user-tomas',
      email: 'tomas.berg@example.com',
      display_name: 'Tomas Berg',
      instance_role: 'user',
      disabled: false,
      last_login_at: DEMO_TS - 11 * DAY,
    },
    {
      subject: 'user-paul',
      email: 'paul.henning@example.com',
      display_name: 'Paul Henning',
      instance_role: 'user',
      disabled: true,
      last_login_at: DEMO_TS - 64 * DAY,
    },
  ]
}

/** One seeded workspace with its members (the demo workspace admin twin). */
export type DemoWorkspaceSeed = {
  workspace: AdminWorkspace
  members: WorkspaceMember[]
}

/**
 * Seeded workspaces for the demo workspace-admin surface. Built over the SAME
 * five seeded users (`seedAdminUsers`) so the member rows enrich with real
 * names/emails and the cross-references line up: every role cell is exercised,
 * an unassigned user exists (the add-member pool is non-empty), and two
 * single-owner workspaces show the last-owner guard.
 */
export function seedAdminWorkspaces(): DemoWorkspaceSeed[] {
  const users = seedAdminUsers()
  const member = (sub: string, role: WorkspaceRoleValue): WorkspaceMember => {
    const user = users.find((candidate) => candidate.subject === sub)
    return {
      display_name: user?.display_name ?? null,
      email: user?.email ?? null,
      role,
      sub,
    }
  }
  const make = (
    workspaceId: string,
    name: string,
    createdBy: string,
    roster: Array<[string, WorkspaceRoleValue]>,
  ): DemoWorkspaceSeed => ({
    members: roster.map(([sub, role]) => member(sub, role)),
    workspace: {
      created_by_sub: createdBy,
      member_count: roster.length,
      name,
      workspace_id: workspaceId,
    },
  })
  const owner = DEMO_OWNER.subject
  return [
    make('ws-default', 'Default workspace', owner, [
      [owner, 'owner'],
      ['user-ines', 'editor'],
    ]),
    make('ws-research', 'Research Team', owner, [
      [owner, 'owner'],
      ['user-mara', 'editor'],
      ['user-tomas', 'commenter'],
      ['user-paul', 'viewer'],
    ]),
    make('ws-legal', 'Legal', 'user-ines', [
      ['user-ines', 'owner'],
      ['user-mara', 'viewer'],
    ]),
  ]
}

/**
 * Seeded personal access tokens for the demo. One scoped + expiring, one
 * read-only + non-expiring — enough to show the list, scopes, and the
 * revoke affordance without a backend.
 */
export function seedAccessTokens(): AccessToken[] {
  return [
    {
      token_id: 'demo-pat-ci',
      name: 'CI pipeline',
      created_at: DEMO_TS - 30 * DAY,
      expires_at: DEMO_TS + 60 * DAY,
      last_used_at: DEMO_TS - 1 * DAY,
      scopes: ['runs:read', 'runs:write'],
    },
    {
      token_id: 'demo-pat-notebook',
      name: 'Analysis notebook',
      created_at: DEMO_TS - 7 * DAY,
      expires_at: null,
      last_used_at: DEMO_TS - 3 * DAY,
      scopes: ['runs:read'],
    },
  ]
}

/**
 * Seeded identity for the demo System panel: a plausible local-mode deployment
 * stack so the read-only panel shows real-looking provider/model identity
 * offline (the live panel reads /health). Only the fields the panel reads
 * are meaningful; the rest satisfy the type.
 */
export function seedSystemHealth(): InqtrixHealth {
  return {
    auth_mode: 'local',
    auth_required: true,
    llm: { provider: 'anthropic', status: 'ready' },
    search: { provider: 'perplexity', status: 'ready' },
    chat_model_options: [
      {
        effort: 'medium',
        effort_source: 'demo',
        model: 'claude-opus-4-8',
        model_source: 'demo',
        requested_tier: 'high',
        tier: 'high',
      },
      {
        effort: 'medium',
        effort_source: 'demo',
        model: 'claude-sonnet-4-6',
        model_source: 'demo',
        requested_tier: 'mid',
        tier: 'mid',
      },
      {
        effort: 'none',
        effort_source: 'demo',
        model: 'claude-haiku-4-5',
        model_source: 'demo',
        requested_tier: 'fast',
        tier: 'fast',
      },
    ],
    models_catalog: [
      {
        card: {
          capabilities: ['analysis', 'long-context', 'reasoning'],
          category: 'high',
          context_window_tokens: 200_000,
          description: 'Deep analysis model for demanding research answers.',
          display_name: 'Claude Opus 4.8',
          id: 'claude-opus-4-8',
          input_modalities: ['text', 'image'],
          knowledge_cutoff: '2026-01',
          max_output_tokens: 32_000,
          pricing: { input_per_mtok: 15, output_per_mtok: 75 },
          reasoning_levels: ['minimal', 'medium', 'high'],
          speed: 'langsam',
          vendor: 'Anthropic',
        },
        model_id: 'claude-opus-4-8',
      },
      {
        card: {
          capabilities: ['analysis', 'chat', 'coding'],
          category: 'mid',
          context_window_tokens: 200_000,
          description: 'Balanced default model for everyday chat and research.',
          display_name: 'Claude Sonnet 4.6',
          id: 'claude-sonnet-4-6',
          input_modalities: ['text', 'image'],
          knowledge_cutoff: '2026-01',
          max_output_tokens: 32_000,
          pricing: { input_per_mtok: 3, output_per_mtok: 15 },
          reasoning_levels: ['minimal', 'medium', 'high'],
          speed: 'mittel',
          vendor: 'Anthropic',
        },
        model_id: 'claude-sonnet-4-6',
      },
      {
        card: {
          capabilities: ['chat', 'summarization'],
          category: 'fast',
          context_window_tokens: 200_000,
          description: 'Fast, low-latency model for compact responses.',
          display_name: 'Claude Haiku 4.5',
          id: 'claude-haiku-4-5',
          input_modalities: ['text'],
          knowledge_cutoff: '2026-01',
          max_output_tokens: 16_000,
          pricing: { input_per_mtok: 0.8, output_per_mtok: 4 },
          reasoning_levels: [],
          speed: 'schnell',
          vendor: 'Anthropic',
        },
        model_id: 'claude-haiku-4-5',
      },
    ],
    node_models: {
      direct_chat: {
        effort: 'medium',
        effort_source: 'demo',
        model: 'claude-sonnet-4-6',
        model_source: 'demo',
        node: 'direct_chat',
        requested_tier: 'mid',
        tier: 'mid',
      },
    },
    status: 'ok',
  }
}

/** Seeded feature matrix for the demo System panel (open features map). */
export function seedSystemCapabilities(): InqtrixCapabilities {
  return {
    algorithms: [],
    features: {
      contextual_retrieval: false,
      document_parser: true,
      embedding_provider: true,
      files: true,
      hybrid_retrieval: true,
      knowledge: true,
      openapi: true,
      prompt_templates: true,
      quota: true,
      reranker: false,
      sharing: true,
    },
    // Default-derived waits so the demo System Status panel shows the
    // same timeouts shape the real /v1/capabilities publishes. The demo editor/chat
    // client-abort path still uses the fixed fallbacks: discovery is disabled in
    // demo mode, so the run paths receive capabilities = null, not this seed.
    timeouts: {
      editor_wait_seconds: 630,
      chat_wait_seconds: 3630,
      text_wait_seconds: 630,
      reasoning_operation_seconds: 600,
      editor_operation_seconds: 600,
      search_operation_seconds: 600,
      claim_extract_operation_seconds: 600,
      research_run_seconds: 3600,
      max_attempts: 3,
    },
  }
}

/** Seeded runtime manifest for the demo System panel. */
export function seedAdminSystemRuntime(): AdminSystemRuntime {
  return {
    api: {
      openapi: true,
    },
    files: {
      blob_storage: 'volume',
      enabled: true,
      max_file_bytes: 52_428_800,
      object_store: 'local',
      object_store_available: true,
    },
    knowledge: {
      contextual_retrieval: false,
      default_top_k: 8,
      document_parser: 'markitdown',
      embedding_model: 'text-embedding-3-large',
      embedding_provider: 'azure',
      enabled: true,
      hybrid_retrieval: true,
      reranker: 'none',
      sparse: 'bm25_german',
      vector_store: 'qdrant',
      vector_store_available: true,
    },
    runs: {
      execution: 'in_process',
      queue: 'memory',
      queue_available: true,
      store: 'postgres',
      worker_dispatch: false,
    },
    storage: {
      backend: 'postgres',
      durable: true,
    },
  }
}
