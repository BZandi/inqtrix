/**
 * Agent Desk demo simulator (the `buildDemoAskScript` pattern): plays the
 * complete assignment arc through the REAL reducer pipeline — synthetic
 * wire events via `appendApiRunEvent` for live state (pulse track, task
 * dots) plus control-row actions for plans/approvals/artifacts (rows are
 * the truth, rule R1). Interactive: phase A runs to the plan gate and
 * PARKS; the user's approval plays phase B (execution -> memo -> done).
 */

import type { Dispatch } from 'react'

import type { ResearchDeskAction } from '@/features/researchDesk/state'
import type {
  AgentTierCapability,
  ResearchRunEvent,
  ResearchRunSummary,
} from '@/features/researchRuns/types'
import type {
  AgentApprovalWire,
  AgentArtifactMetaWire,
  AgentClarificationAnswerRequest,
  AgentClarificationWire,
  AgentPatchWire,
  AgentPlanWire,
} from './types'
import type { AgentOverviewSource } from './agentStatusOverview'
import type {
  AgentExecutionDirective,
  AgentSourcePolicy,
} from './executionPolicy'

export const DEMO_AGENT_RUN_PREFIX = 'RA-demo-'

/**
 * The demo's mirror of `capabilities.agent` for the composer's run
 * overview — same values a CURRENT server publishes (the new UI must be
 * demo-visible). Kept literal on purpose: the demo simulates a server,
 * it never imports server code.
 */
/** Demo mirror of `capabilities.agent.tiers` (the Stufen ladder must
 * be visible in the demo — memory rule: new UI shows in the demo). */
export const DEMO_AGENT_TIERS: AgentTierCapability[] = [
  {
    id: 'schnell',
    clarification_rounds: 0,
    plan_gate: 'skip_unless_strict',
    web_research: false,
    web_child_profile: null,
    web_child_ceiling: null,
    rag_default_profile: 'schnell',
    verify: 'labels',
    response_form: 'chat',
  },
  {
    id: 'gruendlich',
    clarification_rounds: 1,
    plan_gate: 'per_autonomy',
    web_research: true,
    web_child_profile: 'schnell',
    web_child_ceiling: 'compact',
    rag_default_profile: 'standard',
    verify: 'standard',
    response_form: 'auto',
  },
  {
    id: 'tief',
    clarification_rounds: 2,
    plan_gate: 'per_autonomy',
    web_research: true,
    web_child_profile: 'compact',
    web_child_ceiling: 'deep',
    rag_default_profile: 'gruendlich',
    verify: 'escalating',
    response_form: 'canvas',
  },
]

export const DEMO_AGENT_OVERVIEW_SOURCE: AgentOverviewSource = {
  default_mode: 'agent_kernel',
  durable: true,
  limits: {
    tokens: {
      enabled: false,
      limit: 0,
      ceiling: 0,
      recoverable: false,
      extendable: false,
      reason: 'operator_ceiling_exactly_once_required',
    },
    kernel: {
      schnell: { tool_calls: 30, tool_calls_ceiling: 30, steps: 33, steps_ceiling: 33 },
      normal: { tool_calls: 30, tool_calls_ceiling: 60, steps: 73, steps_ceiling: 145 },
      deep: { tool_calls: 60, tool_calls_ceiling: 120, steps: 121, steps_ceiling: 241 },
    },
    directives: { quick_web: { web_searches: 1 } },
    mission: {
      discovery_tool_calls: 15,
      plan_tasks: 8,
      replan_rounds: 2,
      clarification_rounds: 2,
      parallel_children: 6,
    },
    research: { rounds: 2 },
  },
  tools: [
    {
      id: 'web.search.instant',
      summary: 'Run one grounded web search (no research graph).',
      effect: 'read',
      read_only: true,
      idempotent: true,
    },
    {
      id: 'knowledge.search',
      summary: 'Hybrid retrieval over the knowledge base.',
      effect: 'read',
      read_only: true,
      idempotent: true,
    },
    {
      id: 'editor.patch.propose',
      summary: 'Propose anchored edits to an editor document.',
      effect: 'write',
      read_only: false,
      idempotent: false,
    },
  ],
  source_controls: [
    { id: 'web', default: 'available', available: true },
    { id: 'knowledge', default: 'available', available: true },
  ],
  execution_directives: [
    { id: 'quick_web', available: true },
    { id: 'knowledge_only', available: true },
  ],
  permission_modes: {
    balanced: {
      plan_gate: true,
      web_replan_regate: true,
      patch_gate: true,
      kernel_gated_tools: [
        'delegate_batch',
        'load_skill',
        'run_deep_mission',
        'run_web_research',
        'web_instant',
      ],
      kernel_conditional_tools: ['search_project_knowledge'],
      kernel_always_gated: ['propose_editor_patch'],
    },
    autonomous: {
      plan_gate: false,
      web_replan_regate: false,
      patch_gate: true,
      kernel_gated_tools: [],
      kernel_conditional_tools: [],
      kernel_always_gated: ['propose_editor_patch'],
    },
  },
}

type DemoStep = { delayMs: number; actions: ResearchDeskAction[] }

type DemoRuntime = {
  dispatch: Dispatch<ResearchDeskAction>
  timeouts: Set<number>
}

let demoRunCounter = 0

function playSteps(runtime: DemoRuntime, steps: DemoStep[]) {
  let at = 0
  for (const step of steps) {
    at += step.delayMs
    const id = window.setTimeout(() => {
      runtime.timeouts.delete(id)
      for (const action of step.actions) runtime.dispatch(action)
    }, at)
    runtime.timeouts.add(id)
  }
}

/** Fabricated resolution mirroring the backend shape: an explicit pick
 * reports `explicit_request`, otherwise the tier-map default. */
function demoModelResolution(
  node: string,
  selection?: {
    model: string | null
    effort: string | null
    tier: string | null
  },
) {
  if (selection?.model) {
    return {
      node,
      model: selection.model,
      effort: selection.effort ?? '',
      tier: selection.tier ?? 'high',
      model_source: 'explicit_request',
      effort_source: selection.effort ? 'explicit_request' : 'provider_default',
      requested_tier: '',
    }
  }
  return {
    node,
    model: 'claude-sonnet-4-6',
    effort: '',
    tier: selection?.tier ?? 'high',
    model_source: selection?.tier ? `tier:${selection.tier}` : 'tier:high',
    effort_source: 'provider_default',
    requested_tier: selection?.tier ?? '',
  }
}

export function demoRunSummary(
  runId: string,
  sessionId: string,
  question: string,
  autonomy: string,
  status: ResearchRunSummary['status'],
  depth: string = 'normal',
  engineMode: string = 'agent_kernel',
  responseForm: string = 'auto',
  sourcePolicy: AgentSourcePolicy = {
    web: 'available',
    knowledge: 'available',
  },
  executionDirective: AgentExecutionDirective | null = null,
  modelSelection?: DemoModelSelection,
  agentTier: string = '',
): ResearchRunSummary {
  const effectiveSourcePolicy = demoEffectiveSourcePolicy(
    sourcePolicy,
    executionDirective,
  )
  const effectiveMode = executionDirective ? 'agent_kernel' : engineMode
  const effectiveResponseForm = executionDirective
    ? 'chat'
    : responseForm === 'chat' || responseForm === 'canvas'
      ? responseForm
      : effectiveMode === 'agent_kernel'
        ? 'chat'
        : 'canvas'
  return {
    access: { mode: 'owner' },
    run_id: runId,
    status,
    queue_position: null,
    question,
    stack: 'demo',
    mode: effectiveMode as ResearchRunSummary['mode'],
    kind: 'agent',
    session_id: sessionId,
    agent_overrides: {
      autonomy,
      ...(agentTier ? { agent_tier: agentTier } : {}),
      ...(depth === 'deep' ? { depth } : {}),
      ...(responseForm !== 'auto' ? { response_form: responseForm } : {}),
      source_policy: effectiveSourcePolicy,
      ...(executionDirective
        ? { execution_directive: executionDirective }
        : {}),
      ...(modelSelection?.model ? { model: modelSelection.model } : {}),
      ...(modelSelection?.effort ? { effort: modelSelection.effort } : {}),
      ...(!modelSelection?.model && modelSelection?.tier
        ? { model_tier: modelSelection.tier }
        : {}),
    },
    created_at: Date.now() / 1000,
    started_at: Date.now() / 1000,
    finished_at: null,
    elapsed_seconds: null,
    snapshot: {
      execution: demoExecutionBlock({
        autonomy,
        depth: executionDirective ? 'normal' : depth,
        effectiveMode,
        executionDirective,
        modelSelection,
        responseForm: effectiveResponseForm,
        sourcePolicy: effectiveSourcePolicy,
        toolUseCounts: { web: 0, knowledge: 0 },
      }),
    },
    error: null,
    events_url: `/v1/runs/${runId}/events`,
    result_url: `/v1/runs/${runId}/result`,
  }
}

type DemoModelSelection = {
  model: string | null
  effort: string | null
  tier: string | null
}

type DemoExecutionBlockInput = {
  autonomy: string
  depth: string
  effectiveMode: string
  executionDirective: AgentExecutionDirective | null
  modelSelection?: DemoModelSelection
  responseForm: string
  sourcePolicy: AgentSourcePolicy
  toolUseCounts: { web: number; knowledge: number }
}

/** Canonical server-shape projection used by every demo run state. */
export function demoExecutionBlock({
  autonomy,
  depth,
  effectiveMode,
  executionDirective,
  modelSelection,
  responseForm,
  sourcePolicy,
  toolUseCounts,
}: DemoExecutionBlockInput): Record<string, unknown> {
  return {
    execution_directive: executionDirective ?? '',
    effective_mode: effectiveMode,
    response_form: responseForm,
    depth,
    model: modelSelection?.model ?? 'claude-sonnet-4-6',
    reasoning_effort: modelSelection?.effort ?? '',
    source_policy: sourcePolicy,
    consent_reason: executionDirective === 'quick_web'
      ? autonomy === 'strict'
        ? 'strict_approval_required'
        : 'explicit_directive'
      : autonomy === 'autonomous'
        ? 'autonomous_policy'
        : 'permission_policy',
    tool_use_counts: toolUseCounts,
  }
}

function demoEffectiveSourcePolicy(
  policy: AgentSourcePolicy,
  directive: AgentExecutionDirective | null,
): AgentSourcePolicy {
  if (directive === 'quick_web') {
    return { web: 'available', knowledge: 'disabled' }
  }
  if (directive === 'knowledge_only') {
    return { web: 'disabled', knowledge: 'available' }
  }
  return { ...policy }
}

function makeEvent(
  runId: string,
  sequence: number,
  type: string,
  data: Record<string, unknown>,
): ResearchDeskAction {
  const event: ResearchRunEvent = {
    type,
    run_id: runId,
    sequence,
    created_at: Date.now() / 1000,
    data,
  }
  return { event, type: 'appendApiRunEvent' }
}

/** Demo collection the rag task narrows to when the user picked none —
 * matches the seeded vector index ("EU-Recht"), so the source line
 * demonstrates the id -> title mapping. */
const DEMO_RAG_COLLECTION = 'vector-index-eu-recht'

export function demoPlan(
  runId: string,
  collectionIds: string[] = [],
  sourcePolicy: AgentSourcePolicy = {
    web: 'available',
    knowledge: 'available',
  },
): AgentPlanWire {
  const ragScope =
    collectionIds.length > 0 ? collectionIds : [DEMO_RAG_COLLECTION]
  const sourceTasks = [
    {
      task_id: 't1',
      ordinal: 0,
      title: 'Interne Berichte zu KI-Regulierung auswerten',
      tool_kind: 'rag_query' as const,
      objective: 'Bestand zu EU-AI-Act-Anforderungen konsolidieren',
      queries: [
        'Welche Anforderungen stellt der EU AI Act an Hochrisiko-Systeme?',
        'Welche internen Analysen bewerten die Compliance-Last fuer Anbieter?',
      ],
      gap_ids: ['g1'],
      depends_on: [],
      budget: {},
      params: { profile: 'standard', collection_ids: ragScope },
      expected_output: 'Belegte Kernaussagen aus dem Bestand',
      is_falsification: false,
      status: 'pending',
      child_run_id: null,
      result_summary: '',
    },
    {
      task_id: 't2',
      ordinal: 1,
      title: 'Aktuelle Marktlage recherchieren',
      tool_kind: 'web_research' as const,
      objective: 'Aktuelle Entwicklungen seit Q1 2026',
      queries: [
        'EU AI Act Mittelstand Auswirkungen 2026',
        'EU AI Act Umsetzungsstand Leitlinien 2026 offiziell',
      ],
      gap_ids: ['g2'],
      depends_on: [],
      budget: {},
      params: { profile: 'schnell', recency: 'month' },
      expected_output: 'Aktuelle externe Quellenlage',
      is_falsification: false,
      status: 'pending',
      child_run_id: null,
      result_summary: '',
    },
    {
      task_id: 't3',
      ordinal: 2,
      title: 'Gegenposition prüfen',
      tool_kind: 'web_instant' as const,
      objective: 'Kritische Stimmen zur Compliance-Last finden',
      queries: ['AI Act Kritik Compliance Kosten Mittelstand'],
      gap_ids: ['g2'],
      depends_on: [],
      budget: {},
      params: {},
      expected_output: 'Gegenläufige Einschätzungen',
      is_falsification: true,
      status: 'pending',
      child_run_id: null,
      result_summary: '',
    },
  ].filter((task) => {
    if (task.tool_kind === 'rag_query') {
      return sourcePolicy.knowledge === 'available'
    }
    return sourcePolicy.web === 'available'
  })
  const taskIds = sourceTasks.map((task) => task.task_id)
  const tasks = [
    ...sourceTasks,
    {
      task_id: 's',
      ordinal: sourceTasks.length,
      title: 'Memo synthetisieren',
      tool_kind: 'synthesis' as const,
      objective: 'Belegtes Memo mit Handlungsempfehlungen',
      queries: [],
      gap_ids: [],
      depends_on: taskIds,
      budget: {},
      params: {},
      expected_output: 'Memo',
      is_falsification: false,
      status: 'pending',
      child_run_id: null,
      result_summary: '',
    },
  ].map((task, ordinal) => ({ ...task, ordinal }))
  const planParts = [
    ...(sourcePolicy.knowledge === 'available' ? ['Bestand auswerten'] : []),
    ...(sourcePolicy.web === 'available'
      ? ['aktuelle Lage extern verifizieren', 'Gegenposition pruefen']
      : []),
    'ein belegtes Memo erstellen',
  ]
  return {
    plan_id: `plan-${runId}`,
    run_id: runId,
    version: 1,
    status: 'proposed',
    created_by: 'agent',
    reason: '',
    created_at: Date.now() / 1000,
    summary_markdown: `${planParts.join(', ')}.`,
    assumptions: ['Fokus auf Anbieter mit Hochrisiko-Anwendungen.'],
    success_criteria: [
      'Jede Kernaussage mit Quelle belegt.',
      ...(sourcePolicy.web === 'available'
        && sourcePolicy.knowledge === 'available'
        ? ['Interne und externe Sicht abgeglichen.']
        : []),
      'Konkrete Handlungsempfehlungen enthalten.',
    ],
    tasks,
    versions: [
      {
        plan_id: `plan-${runId}`,
        version: 1,
        status: 'proposed',
        created_by: 'agent',
        reason: '',
        created_at: Date.now() / 1000,
      },
    ],
  }
}

const MEMO_SECTIONS = [
  '# Markteinschätzung: EU AI Act und der KI-Mittelstand\n\n## Kernaussagen\n\nDie Anforderungen des EU AI Act treffen mittelständische Anbieter von Hochrisiko-Systemen ab August 2026 mit voller Wirkung [K1]. Die interne Analyse zeigt: Konformitätsbewertung und technische Dokumentation binden im Schnitt 15-20% der Entwicklungskapazität [K2].',
  '\n\n## Aktuelle Entwicklung\n\nDie externe Quellenlage bestätigt eine Konsolidierungswelle: kleinere Anbieter suchen Partnerschaften mit etablierten Compliance-Plattformen [W1]. Gleichzeitig entstehen neue Beratungsangebote, die die Erstzertifizierung deutlich beschleunigen [W2].',
  '\n\n## Gegenposition\n\nKritische Stimmen halten die Compliance-Last für überzeichnet: harmonisierte Normen und Sandboxes senken die realen Kosten schneller als erwartet [W3]. Diese Einschätzung widerspricht teilweise der internen Projektion [K2] — beide Szenarien sind im Risikoboard zu führen.\n\n## Offene Punkte\n\nDie Kostenwirkung der harmonisierten Normen ist noch nicht belastbar quantifiziert.',
]

function memoMeta(
  runId: string,
  revision: number,
  status: 'writing' | 'ready',
): AgentArtifactMetaWire {
  return {
    artifact_id: `memo-${runId}`,
    run_id: runId,
    session_id: null,
    kind: 'memo',
    title: 'Markteinschätzung EU AI Act',
    status,
    revision,
    updated_by: 'agent',
    refs_count: 6,
    created_at: Date.now() / 1000,
    updated_at: Date.now() / 1000,
  }
}

/** Demo K-citation passages. They are VERBATIM lines of the demo
 * knowledge documents (`knowledge/demo.ts`) so the evidence reader
 * highlights a real match instead of opening on nothing — the demo
 * has to exercise the same path the live desk does. A parity test
 * pins both the document ids and these excerpts. */
const DEMO_K1_PASSAGE = 'Ein KI-System gilt als Hochrisiko-KI-System, wenn es als Sicherheitsbauteil eines unter die Harmonisierungsrechtsvorschriften fallenden Produkts verwendet wird oder selbst ein solches Produkt ist.'
const DEMO_K2_PASSAGE = 'Fuer Hochrisiko-Anwendungen empfiehlt das BSI eine dokumentierte Risikoanalyse je Lebenszyklusphase sowie kontinuierliches Monitoring im Betrieb.'

/** The demo's knowledge citations, in ONE place: three artifacts cited
 * the same passages as separate literals, so a retarget had to be made
 * three times (and was missed once). The parity test asserts these
 * against the knowledge demo corpus. */
export const agentDemoKnowledgeRefs = [
  {
    chunkIndex: 0,
    documentId: 'kdoc-ai-act-volltext',
    label: 'K1',
    sourceText: DEMO_K1_PASSAGE,
    title: 'EU-AI-Act-Volltext.pdf',
  },
  {
    chunkIndex: 0,
    documentId: 'kdoc-bsi-kriterien',
    label: 'K2',
    sourceText: DEMO_K2_PASSAGE,
    title: 'BSI-Kriterienkatalog-KI.pdf',
  },
] as const

const demoKnowledgeRefWire = (label: 'K1' | 'K2') => {
  const reference = agentDemoKnowledgeRefs.find(
    (candidate) => candidate.label === label,
  )
  if (!reference) throw new Error(`unknown demo reference ${label}`)
  return {
    chunk_index: reference.chunkIndex,
    document_id: reference.documentId,
    label: reference.label,
    source_text: reference.sourceText,
    title: reference.title,
  }
}

const ANSWER_MARKDOWN = `Kurzantwort: Der EU AI Act trifft den KI-Mittelstand ab August 2026 mit voller Wirkung [K1]. Die wichtigsten Punkte im Vergleich:

| Aspekt | Einschätzung | Beleg |
| --- | --- | --- |
| Pflichten | Konformitätsbewertung + technische Dokumentation | [K1] |
| Aufwand | 15–20 % der Entwicklungskapazität | [K2] |
| Markt | Konsolidierung, Compliance-Partnerschaften | [W1] |

Der Ablauf bis zur Konformität:

\`\`\`mermaid
flowchart LR
  A[Risikoklassifizierung] --> B[Gap-Analyse]
  B --> C[Technische Doku]
  C --> D[Konformitätsbewertung]
  D --> E[CE-Kennzeichnung]
\`\`\`

Offen bleibt die endgültige Auslegung der Normenreihe — hier lohnt ein Blick auf die laufende Harmonisierung [W2].`

function answerMeta(
  runId: string,
  revision: number,
  status: 'writing' | 'ready',
): AgentArtifactMetaWire {
  return {
    artifact_id: `answer-${runId}`,
    run_id: runId,
    session_id: null,
    kind: 'answer',
    title: 'Antwort',
    status,
    revision,
    updated_by: 'agent',
    refs_count: 4,
    created_at: Date.now() / 1000,
    updated_at: Date.now() / 1000,
  }
}

function answerDetailAction(
  runId: string,
  revision: number,
  status: 'writing' | 'ready',
): ResearchDeskAction {
  return {
    artifact: {
      ...answerMeta(runId, revision, status),
      content_markdown: ANSWER_MARKDOWN,
      refs: [
        demoKnowledgeRefWire('K1'),
        demoKnowledgeRefWire('K2'),
        { label: 'W1', url: 'https://example.com/markt-konsolidierung', title: 'Marktbericht Konsolidierung' },
        { label: 'W2', url: 'https://example.com/zertifizierung', title: 'Zertifizierungs-Angebote 2026' },
      ],
      revisions: Array.from({ length: revision }, (_, index) => ({
        revision: index + 1,
        created_by: 'agent',
        created_at: Date.now() / 1000,
      })),
    },
    runId,
    type: 'setAgentRunArtifactDetail',
  }
}

// A kernel `write_canvas` deliverable — distinct from the inline answer, it
// opens as a document tab (Phase 1 deliverable routing must be demo-visible).
const DELIVERABLE_MARKDOWN = `# Kurz-Memo: EU AI Act

Die zentrale Pflicht ist die **CE-Kennzeichnung** für Hochrisiko-Systeme [K1].
Marktseitig zeichnet sich eine Konsolidierung der Anbieter ab [W1].

- Geltungsbereich prüfen (Annex III)
- Konformitätsbewertung einplanen
- Dokumentationspflichten ab Q3`

function deliverableMeta(
  runId: string,
  revision: number,
  status: 'writing' | 'ready',
): AgentArtifactMetaWire {
  return {
    artifact_id: `deliverable-${runId}`,
    run_id: runId,
    session_id: null,
    kind: 'deliverable',
    title: 'Kurz-Memo: EU AI Act',
    status,
    revision,
    updated_by: 'agent',
    refs_count: 2,
    created_at: Date.now() / 1000,
    updated_at: Date.now() / 1000,
  }
}

function deliverableDetailAction(
  runId: string,
  revision: number,
  status: 'writing' | 'ready',
): ResearchDeskAction {
  return {
    artifact: {
      ...deliverableMeta(runId, revision, status),
      content_markdown: DELIVERABLE_MARKDOWN,
      refs: [
        demoKnowledgeRefWire('K1'),
        { label: 'W1', url: 'https://example.com/markt-konsolidierung', title: 'Marktbericht Konsolidierung' },
      ],
      revisions: Array.from({ length: revision }, (_, index) => ({
        revision: index + 1,
        created_by: 'agent',
        created_at: Date.now() / 1000,
      })),
    },
    runId,
    type: 'setAgentRunArtifactDetail',
  }
}

function memoDetailAction(
  runId: string,
  revision: number,
  status: 'writing' | 'ready',
  sections: number,
): ResearchDeskAction {
  return {
    artifact: {
      ...memoMeta(runId, revision, status),
      content_markdown: MEMO_SECTIONS.slice(0, sections).join(''),
      refs: [
        demoKnowledgeRefWire('K1'),
        demoKnowledgeRefWire('K2'),
        { label: 'W1', url: 'https://example.com/markt-konsolidierung', title: 'Marktbericht Konsolidierung' },
        { label: 'W2', url: 'https://example.com/zertifizierung', title: 'Zertifizierungs-Angebote 2026' },
        { label: 'W3', url: 'https://example.com/kritik', title: 'Kommentar Compliance-Kosten' },
      ],
      revisions: Array.from({ length: revision }, (_, index) => ({
        revision: index + 1,
        created_by: 'agent',
        created_at: Date.now() / 1000,
      })),
    },
    runId,
    type: 'setAgentRunArtifactDetail',
  }
}

export type AgentDemoActions = {
  applyPatch: (
    runId: string,
    patchId: string,
    expectedRevision: number,
  ) => Promise<
    | { kind: 'applied'; revision: number; appliedEditIds: string[] }
    | { kind: 'conflict'; currentRevision: number | null }
  >
  rejectPatch: (
    runId: string,
    patchId: string,
    note: string,
  ) => Promise<unknown>
  answerClarification: (
    runId: string,
    clarificationId: string,
    answer: AgentClarificationAnswerRequest,
  ) => Promise<unknown>
  decideApproval: (
    runId: string,
    approvalId: string,
    decision: {
      decision: string
      actions?: { tool: string; args: Record<string, unknown> }[]
    },
  ) => Promise<unknown>
  submit: (input: {
    autonomy: string
    question: string
    sessionId: string
    /** @-selected collection scope; flows into the demo plan's rag task
     * so the per-task source line shows the real selection. */
    collectionIds?: string[]
    documentId?: string
    /** Output form: 'chat' plays the inline-answer variant. */
    responseForm?: string
    /** Selected engine: 'agent_kernel' plays the conversational
     * inline-answer variant — the demo's honest approximation of the
     * kernel's direct-answer behavior. */
    engineMode?: string
    /** Attached skill labels: shown as an intake narration
     * line so an attached demo skill is visibly acknowledged. */
    skillLabels?: string[]
    /** One-message route from the direct-command group. */
    executionDirective?: AgentExecutionDirective | null
    sourcePolicy?: AgentSourcePolicy
    /** Thoroughness: 'deep' plays the verification narration
     * and stamps the Deep badge onto the run summary. */
    depth?: string
    /** Selected Stufe: echoed as agent_overrides.agent_tier so the
     * tier-aware gate UI (Suchtiefe ceiling, add-task defaults) is
     * demo-visible. */
    agentTier?: string
    /** R3 picker selection: feeds the fabricated model_resolution
     * event so the answer chip is demo-visible. */
    modelSelection?: {
      model: string | null
      effort: string | null
      tier: string | null
    }
  }) => void
  cancel: (runId: string) => void
  dispose: () => void
}

/** Build the interactive demo runtime bound to the reducer dispatch.

 * `getDocumentMarkdown` reads the CURRENT local editor document so the
 * demo patch proposes real, deterministic edits against it and the
 * local apply lands as a visible tracked change.
 */
export function createAgentDemo(
  dispatch: Dispatch<ResearchDeskAction>,
  getDocumentMarkdown: (documentId: string) => string | null = () => null,
): AgentDemoActions {
  const runtime: DemoRuntime = { dispatch, timeouts: new Set() }
  const patchTargets = new Map<string, string>()
  const responseForms = new Map<string, string>()
  const sourcePolicies = new Map<string, AgentSourcePolicy>()
  const acceptedSummaries = new Map<string, ResearchRunSummary>()
  // Thoroughness per run: read by the post-clarification
  // sequence, which runs outside the submit closure.
  const depths = new Map<string, string>()
  // R3 picker selection per run — same closure rule as `depths`.
  const modelSelections = new Map<
    string,
    { model: string | null; effort: string | null; tier: string | null }
  >()
  const lastDemoPatchWires = new Map<string, AgentPatchWire>()
  const lastDemoPatchEdits = new Map<string, AgentPatchWire['edits']>()
  // Strict quick-web tool gate (Phase 2): the search execution is HELD until
  // the tool approval is decided; the resume thunk (built in submit, capturing
  // the run's step sequence) is played by decideApproval's apr-tool branch.
  const heldQuickWeb = new Map<
    string,
    (decision: {
      decision: string
      actions?: { tool: string; args: Record<string, unknown> }[]
    }) => void
  >()

  const submit: AgentDemoActions['submit'] = ({
    autonomy,
    question,
    sessionId,
    collectionIds = [],
    documentId,
    responseForm,
    engineMode,
    skillLabels = [],
    executionDirective = null,
    sourcePolicy = { web: 'available', knowledge: 'available' },
    depth = 'normal',
    agentTier = '',
    modelSelection,
  }) => {
    demoRunCounter += 1
    const runId = `${DEMO_AGENT_RUN_PREFIX}${demoRunCounter}`
    const effectiveSourcePolicy = demoEffectiveSourcePolicy(
      sourcePolicy,
      executionDirective,
    )
    sourcePolicies.set(runId, effectiveSourcePolicy)
    if (documentId) patchTargets.set(runId, documentId)
    if (depth === 'deep') depths.set(runId, 'deep')
    if (modelSelection) modelSelections.set(runId, modelSelection)
    if ((responseForm === 'chat' || engineMode === 'agent_kernel' || executionDirective) && !documentId) {
      responseForms.set(runId, 'chat')
    }
    const initialSummary = demoRunSummary(
      runId,
      sessionId,
      question,
      autonomy,
      'running',
      depth,
      engineMode ?? 'agent_kernel',
      responseForm ?? 'auto',
      sourcePolicy,
      executionDirective,
      modelSelection,
      agentTier,
    )
    acceptedSummaries.set(runId, initialSummary)
    dispatch({
      select: true,
      summary: initialSummary,
      type: 'upsertAgentRunSummary',
    })
    let seq = 0
    const next = () => (seq += 1)
    if (executionDirective) {
      const toolKind = executionDirective === 'quick_web'
        ? 'web_instant'
        : 'rag_query'
      const usedSummary = {
        ...initialSummary,
        snapshot: {
          ...initialSummary.snapshot,
          execution: {
            ...initialSummary.snapshot.execution,
            tool_use_counts: executionDirective === 'quick_web'
              ? { web: 1, knowledge: 0 }
              : { web: 0, knowledge: 1 },
          },
        },
      }
      acceptedSummaries.set(runId, usedSummary)
      const preGate: DemoStep[] = [
        {
          delayMs: 300,
          actions: [
            makeEvent(runId, next(), 'inqtrix.run.started', {}),
            makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'intake' }),
          ],
        },
      ]
      // The search execution + answer/deliverable tail, parameterized on the
      // (possibly edited) query so a strict-mode edit flows into the demo.
      // Kernel lanes emit REAL tool events (bare-query args_preview) —
      // the demo mirrors that wire shape instead of fabricating a
      // generic searching activity (parity rule: new UI must be
      // visible in the demo).
      const demoTool = toolKind === 'web_instant'
        ? 'web_instant'
        : 'search_project_knowledge'
      const executionTail = (searchProbe: string): DemoStep[] => [
        {
          delayMs: 500,
          actions: [
            makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'execution', previous_phase: 'intake' }),
            makeEvent(runId, next(), 'inqtrix.agent.task.started', {
              task_id: 'direct-1', ordinal: 0, tool_kind: toolKind, attempt: 1,
            }),
            makeEvent(runId, next(), 'inqtrix.agent.tool.started', {
              tool: demoTool,
              tool_call_id: 'demo-direct-1',
              args_preview: searchProbe,
            }),
          ],
        },
        {
          delayMs: 1200,
          actions: [
            makeEvent(runId, next(), 'inqtrix.agent.tool.finished', {
              tool: demoTool,
              tool_call_id: 'demo-direct-1',
              status: 'success',
            }),
            makeEvent(runId, next(), 'inqtrix.agent.task.finished', {
              task_id: 'direct-1', ordinal: 0, tool_kind: toolKind, status: 'completed',
            }),
            { select: true, summary: usedSummary, type: 'upsertAgentRunSummary' },
            makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'synthesis', previous_phase: 'execution' }),
            makeEvent(runId, next(), 'inqtrix.node.model_resolution', demoModelResolution('agent_answer', modelSelection)),
            {
              artifacts: [answerMeta(runId, 2, 'ready'), deliverableMeta(runId, 1, 'ready')],
              runId,
              type: 'setAgentRunArtifacts',
            },
            answerDetailAction(runId, 2, 'ready'),
            deliverableDetailAction(runId, 1, 'ready'),
            makeEvent(runId, next(), 'inqtrix.agent.artifact.created', {
              artifact_id: `answer-${runId}`, kind: 'answer', revision: 2, updated_by: 'agent',
            }),
            makeEvent(runId, next(), 'inqtrix.agent.artifact.created', {
              artifact_id: `deliverable-${runId}`, kind: 'deliverable', revision: 1, updated_by: 'agent',
            }),
          ],
        },
        {
          delayMs: 500,
          actions: [
            makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'done', previous_phase: 'synthesis' }),
            makeEvent(runId, next(), 'inqtrix.run.completed', {}),
          ],
        },
      ]
      // Non-autonomous autonomy (Standard/balanced or strict) gates the direct
      // web search behind a tool approval (Phase 2, demo-visible): play up to
      // the gate, hold the search until decideApproval resolves it. (The demo
      // presets expose Standard/Auto; strict is not selectable here.)
      if (executionDirective === 'quick_web' && autonomy !== 'autonomous') {
        playSteps(runtime, [
          ...preGate,
          {
            delayMs: 500,
            actions: [
              { approvals: [demoToolApproval(runId, question)], runId, type: 'setAgentRunApprovals' },
              makeEvent(runId, next(), 'inqtrix.run.waiting', { status: 'waiting_for_approval' }),
            ],
          },
        ])
        heldQuickWeb.set(runId, (decision) => {
          const approved = decision.decision !== 'reject'
          const status = approved
            ? decision.decision === 'edit' ? 'edited' : 'approved'
            : 'rejected'
          dispatch({
            approvals: [{
              ...demoToolApproval(runId, question),
              status,
              decision: decision.decision,
              decided_at: Date.now() / 1000,
            }],
            runId,
            type: 'setAgentRunApprovals',
          })
          if (!approved) {
            // Mirrors the backend contract: a rejected gate COMPLETES the
            // run with a receipt as its answer (never a blank / failure).
            playSteps(runtime, [
              {
                delayMs: 300,
                actions: [
                  makeEvent(runId, next(), 'inqtrix.agent.approval.decided', { approval_id: `apr-tool-${runId}`, status: 'rejected' }),
                  makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'synthesis', previous_phase: 'execution' }),
                  {
                    artifacts: [{ ...answerMeta(runId, 1, 'ready'), title: 'Werkzeug abgelehnt', refs_count: 0 }],
                    runId,
                    type: 'setAgentRunArtifacts',
                  },
                  {
                    artifact: {
                      ...answerMeta(runId, 1, 'ready'),
                      title: 'Werkzeug abgelehnt',
                      refs_count: 0,
                      content_markdown: 'Die direkte Websuche wurde nicht freigegeben. Passe den Auftrag an oder gib die Suche frei.',
                      refs: [],
                      revisions: [{ revision: 1, created_by: 'agent', created_at: Date.now() / 1000 }],
                    },
                    runId,
                    type: 'setAgentRunArtifactDetail',
                  },
                  makeEvent(runId, next(), 'inqtrix.agent.artifact.created', { artifact_id: `answer-${runId}`, kind: 'answer', revision: 1, updated_by: 'agent' }),
                ],
              },
              {
                delayMs: 400,
                actions: [
                  makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'done', previous_phase: 'synthesis' }),
                  makeEvent(runId, next(), 'inqtrix.run.completed', {}),
                ],
              },
            ])
            return
          }
          const editedQuery = decision.actions?.[0]?.args?.query
          const searchProbe =
            typeof editedQuery === 'string' && editedQuery.trim()
              ? editedQuery
              : question
          playSteps(runtime, [
            {
              delayMs: 200,
              actions: [
                makeEvent(runId, next(), 'inqtrix.agent.approval.decided', {
                  approval_id: `apr-tool-${runId}`,
                  status: decision.decision === 'edit' ? 'edited' : 'approved',
                }),
              ],
            },
            ...executionTail(searchProbe),
          ])
        })
        return
      }
      playSteps(runtime, [...preGate, ...executionTail(question)])
      return
    }
    const gate = autonomy !== 'autonomous'
    const plan = demoPlan(runId, collectionIds, effectiveSourcePolicy)
    const taskCount = plan.tasks.length
    const steps: DemoStep[] = [
      {
        delayMs: 400,
        actions: [
          makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'intake' }),
          ...(skillLabels.length || executionDirective
            ? [makeEvent(runId, next(), 'inqtrix.agent.narration', {
              narration_id: 'n-skills',
              kind: 'discovery',
              phase: 'intake',
              text: [
                skillLabels.length
                  ? `Angehaengte Skills: ${skillLabels.map((label) => `/${label}`).join(', ')}.`
                  : '',
                executionDirective
                  ? `Direkter Ausfuehrungsweg: ${executionDirective}.`
                  : '',
              ].filter(Boolean).join(' '),
              final: true,
            })]
            : []),
        ],
      },
      { delayMs: 1400, actions: [makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'discovery', previous_phase: 'intake' })] },
      ...(effectiveSourcePolicy.web === 'available'
        ? [{ delayMs: 900, actions: [makeEvent(runId, next(), 'inqtrix.agent.activity', { kind: 'searching', probe: 'EU AI Act Anforderungen' })] }]
        : []),
      ...(effectiveSourcePolicy.knowledge === 'available'
        ? [{ delayMs: 1500, actions: [makeEvent(runId, next(), 'inqtrix.agent.activity', { kind: 'searching', probe: 'Bestand: Compliance-Analysen' })] }]
        : []),
      {
        delayMs: 1300,
        actions: [
          makeEvent(runId, next(), 'inqtrix.agent.narration', {
            narration_id: 'n-discovery',
            kind: 'discovery',
            phase: 'discovery',
            text: 'Ich habe 2 belegte Fakten und 2 offene Luecken identifiziert. Wichtigste Luecke: Die aktuelle Marktlage seit Q1 2026 fehlt im Bestand.',
            final: true,
          }),
        ],
      },
      {
        delayMs: 1400,
        actions: [
          makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'planning', previous_phase: 'discovery' }),
          { plan, runId, type: 'setAgentRunPlan' },
          makeEvent(runId, next(), 'inqtrix.agent.plan.proposed', { plan_id: `plan-${runId}`, version: 1, task_count: taskCount }),
          makeEvent(runId, next(), 'inqtrix.agent.narration', {
            narration_id: 'n-plan-1',
            kind: 'plan',
            phase: 'planning',
            text: `Mein Plan (${taskCount} Aufgaben): ${plan.summary_markdown}`,
            final: true,
          }),
        ],
      },
    ]
    if (gate) {
      steps.push({
        delayMs: 700,
        actions: [
          {
            approvals: [demoApproval(runId)],
            runId,
            type: 'setAgentRunApprovals',
          },
          makeEvent(runId, next(), 'inqtrix.run.waiting', { status: 'waiting_for_approval' }),
        ],
      })
    }
    playSteps(runtime, steps)
    if (!gate) {
      // Autonomous: no gates at all — chain both execution parts after
      // the planning steps (drop the clarification-gate tail of part 1
      // and part 2's synthetic resume events).
      const part1 = executionUntilClarification(runId, () => next()).slice(0, -1)
      const part2 = executionAfterClarification(runId, () => next()).map(
        (step, index) =>
          index === 0
            ? {
              ...step,
              actions: step.actions.slice(2),
            }
            : step,
      )
      playSteps(
        runtime,
        [...part1, ...part2].map((step, index) => ({
          ...step,
          delayMs: step.delayMs + (index === 0 ? 6600 : 0),
        })),
      )
    }
  }

  const decideApproval: AgentDemoActions['decideApproval'] = (
    runId,
    approvalId,
    decision,
  ) => {
    const approved = decision.decision !== 'reject'
    if (approvalId.startsWith('apr-tool-')) {
      // The held strict quick-web search resumes (approve/edit) or ends with
      // a reject receipt — all choreography lives in the submit-closure thunk.
      const resume = heldQuickWeb.get(runId)
      heldQuickWeb.delete(runId)
      resume?.(decision)
      return Promise.resolve()
    }
    if (approvalId.startsWith('apr-patch-')) {
      // The patch gate: either way the run finishes NORMALLY (the memo
      // stays the deliverable; the patch decision is its own record).
      dispatch({
        approvals: [
          {
            ...demoApproval(runId),
            approval_id: approvalId,
            kind: 'patch',
            subject_type: 'editor_patch',
            subject_id: `pch-${runId}`,
            status: approved ? 'approved' : 'rejected',
            decision: decision.decision,
            decided_at: Date.now() / 1000,
          },
        ],
        runId,
        type: 'setAgentRunApprovals',
      })
      playSteps(runtime, [
        {
          delayMs: 300,
          actions: [
            makeEvent(runId, 300, 'inqtrix.agent.approval.decided', {
              approval_id: approvalId,
              status: approved ? 'approved' : 'rejected',
            }),
            makeEvent(runId, 301, 'inqtrix.agent.phase.changed', { phase: 'done', previous_phase: 'patch' }),
            makeEvent(runId, 302, 'inqtrix.run.completed', {}),
          ],
        },
      ])
      return Promise.resolve()
    }
    dispatch({
      approvals: [
        {
          ...demoApproval(runId),
          approval_id: approvalId,
          status: approved ? 'approved' : 'rejected',
          decision: decision.decision,
          // The decided requirement rides the payload just as it does
          // live — without it the demo would show the gate but never
          // what the user asked the report to look like.
          decision_payload:
            'report_guidance' in decision
              ? { report_guidance: decision.report_guidance ?? '' }
              : {},
          decided_at: Date.now() / 1000,
        },
      ],
      runId,
      type: 'setAgentRunApprovals',
    })
    let seq = 100
    // The decided EVENT puts the decision into the transcript (the rows
    // above carry the content the stream joins with).
    const decided: DemoStep = {
      delayMs: 100,
      actions: [
        makeEvent(runId, (seq += 1), 'inqtrix.agent.approval.decided', {
          approval_id: approvalId,
          status: approved ? 'approved' : 'rejected',
        }),
      ],
    }
    if (!approved) {
      // Mirrors the backend contract: a rejected gate COMPLETES the run
      // with a deterministic receipt as its chat answer — no failure.
      const receipt =
        'Plan abgelehnt. Es wurde keine Recherche ausgefuehrt.'
        + '\n\nPasse den Auftrag an und sende ihn erneut, um einen neuen'
        + ' Plan zu erhalten.'
      playSteps(runtime, [
        decided,
        {
          delayMs: 300,
          actions: [
            makeEvent(runId, (seq += 1), 'inqtrix.agent.narration', {
              narration_id: 'n-plan-rejected-0',
              kind: 'plan',
              text: 'Plan abgelehnt; der Lauf endet ohne Recherche.',
              phase: 'planning',
              final: true,
            }),
            {
              artifacts: [
                {
                  ...answerMeta(runId, 1, 'ready'),
                  title: 'Plan abgelehnt',
                  refs_count: 0,
                },
              ],
              runId,
              type: 'setAgentRunArtifacts',
            },
            {
              artifact: {
                ...answerMeta(runId, 1, 'ready'),
                title: 'Plan abgelehnt',
                refs_count: 0,
                content_markdown: receipt,
                refs: [],
                revisions: [
                  {
                    revision: 1,
                    created_by: 'agent',
                    created_at: Date.now() / 1000,
                  },
                ],
              },
              runId,
              type: 'setAgentRunArtifactDetail',
            },
            makeEvent(runId, (seq += 1), 'inqtrix.agent.artifact.created', {
              artifact_id: `answer-${runId}`,
              kind: 'answer',
              revision: 1,
              updated_by: 'agent',
            }),
            makeEvent(runId, (seq += 1), 'inqtrix.run.completed', {}),
          ],
        },
      ])
      return Promise.resolve()
    }
    playSteps(runtime, [
      decided,
      ...executionUntilClarification(runId, () => (seq += 1)),
    ])
    return Promise.resolve()
  }

  const answerClarification: AgentDemoActions['answerClarification'] = (
    runId,
    clarificationId,
    answer,
  ) => {
    dispatch({
      clarifications: [
        {
          ...demoClarification(runId),
          clarification_id: clarificationId,
          status: 'answered',
          answer: answer.answer ?? '',
          option_id: answer.option_id ?? '',
          answers: answer.answers ?? {},
          answered_at: Date.now() / 1000,
        },
      ],
      runId,
      type: 'setAgentRunClarifications',
    })
    let seq = 200
    playSteps(runtime, [
      {
        delayMs: 100,
        actions: [
          makeEvent(runId, (seq += 1), 'inqtrix.agent.clarification.answered', {
            clarification_id: clarificationId,
          }),
        ],
      },
      ...executionAfterClarification(runId, () => (seq += 1)),
    ])
    return Promise.resolve()
  }

  const cancel: AgentDemoActions['cancel'] = (runId) => {
    playSteps(runtime, [
      {
        delayMs: 200,
        actions: [makeEvent(runId, 900, 'inqtrix.run.cancelled', {})],
      },
    ])
  }

  const applyPatch: AgentDemoActions['applyPatch'] = (runId, patchId) => {
    const documentId = patchTargets.get(runId)
    const markdown = documentId ? getDocumentMarkdown(documentId) : null
    if (!documentId || markdown === null) {
      return Promise.resolve({ kind: 'conflict', currentRevision: null })
    }
    // Local anchor application, mirroring the server semantics: verbatim
    // replace, append at the end; unresolvable anchors are skipped.
    let content = markdown
    const appliedEditIds: string[] = []
    for (const edit of lastDemoPatchEdits.get(patchId) ?? []) {
      if (edit.position === 'append') {
        content = `${content.trimEnd()}\n\n${edit.text}`
        appliedEditIds.push(edit.id)
      } else if (edit.find && content.includes(edit.find)) {
        content = content.replace(edit.find, edit.text)
        appliedEditIds.push(edit.id)
      }
    }
    dispatch({
      contentMarkdown: content,
      documentId,
      type: 'updateEditorDocumentMarkdown',
    })
    dispatch({
      patch: {
        ...(lastDemoPatchWires.get(patchId) as AgentPatchWire),
        status: 'accepted',
        applied_revision: 2,
        applied_edit_ids: appliedEditIds,
        decided_at: Date.now() / 1000,
        document_revision: 2,
      },
      runId,
      type: 'setAgentRunPatch',
    })
    return Promise.resolve({
      kind: 'applied',
      revision: 2,
      appliedEditIds,
    })
  }

  const rejectPatch: AgentDemoActions['rejectPatch'] = (runId, patchId, note) => {
    dispatch({
      patch: {
        ...(lastDemoPatchWires.get(patchId) as AgentPatchWire),
        status: 'rejected',
        note,
        decided_at: Date.now() / 1000,
      },
      runId,
      type: 'setAgentRunPatch',
    })
    return Promise.resolve()
  }

  return {
    answerClarification,
    applyPatch,
    cancel,
    decideApproval,
    dispose: () => {
      for (const id of runtime.timeouts) window.clearTimeout(id)
      runtime.timeouts.clear()
    },
    rejectPatch,
    submit,
  }

  function demoUsedSummaryAction(
    runId: string,
    policy: AgentSourcePolicy,
  ): ResearchDeskAction {
    const base = acceptedSummaries.get(runId)
    if (!base) {
      throw new Error(`Missing accepted demo summary for ${runId}`)
    }
    const snapshot = base.snapshot ?? {}
    const execution = snapshot.execution && typeof snapshot.execution === 'object'
      ? snapshot.execution as Record<string, unknown>
      : {}
    const updated: ResearchRunSummary = {
      ...base,
      snapshot: {
        ...snapshot,
        execution: {
          ...execution,
          tool_use_counts: {
            web: policy.web === 'available' ? 2 : 0,
            knowledge: policy.knowledge === 'available' ? 1 : 0,
          },
        },
      },
    }
    acceptedSummaries.set(runId, updated)
    return { select: true, summary: updated, type: 'upsertAgentRunSummary' }
  }

  function executionUntilClarification(
    runId: string,
    next: () => number,
  ): DemoStep[] {
    const policy = sourcePolicies.get(runId)
      ?? { web: 'available', knowledge: 'available' }
    return [
      {
        delayMs: 400,
        actions: [
          makeEvent(runId, next(), 'inqtrix.run.started', {}),
          makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'execution', previous_phase: 'planning' }),
          ...(policy.knowledge === 'available'
            ? [makeEvent(runId, next(), 'inqtrix.agent.task.started', { task_id: 't1', ordinal: 0, tool_kind: 'rag_query', attempt: 1 })]
            : []),
          ...(policy.web === 'available'
            ? [makeEvent(runId, next(), 'inqtrix.agent.task.started', { task_id: 't2', ordinal: 1, tool_kind: 'web_research', attempt: 1 })]
            : []),
        ],
      },
      {
        delayMs: 700,
        actions: [
          // Per-query protocol rows (Verlauf = Portal): each query of
          // the rag task is its own step, exactly like the backend
          // emits per invocation.
          ...(policy.knowledge === 'available'
            ? [makeEvent(runId, next(), 'inqtrix.agent.activity', {
              kind: 'searching',
              operation: 'knowledge.search',
              task_id: 't1',
              status: 'started',
              query: 'Welche Anforderungen stellt der EU AI Act an Hochrisiko-Systeme?',
              current: 1,
              total: 2,
            })]
            : []),
        ],
      },
      {
        delayMs: 900,
        actions: [
          ...(policy.knowledge === 'available'
            ? [
              makeEvent(runId, next(), 'inqtrix.agent.activity', {
                kind: 'searching',
                operation: 'knowledge.search',
                task_id: 't1',
                status: 'completed',
                query: 'Welche Anforderungen stellt der EU AI Act an Hochrisiko-Systeme?',
                current: 1,
                total: 2,
                metrics: { result_count: 5 },
              }),
              makeEvent(runId, next(), 'inqtrix.agent.activity', {
                kind: 'searching',
                operation: 'knowledge.search',
                task_id: 't1',
                status: 'started',
                query: 'Welche internen Analysen bewerten die Compliance-Last fuer Anbieter?',
                current: 2,
                total: 2,
              }),
            ]
            : []),
          ...(policy.web === 'available'
            ? [makeEvent(runId, next(), 'inqtrix.agent.child.progress', {
              task_id: 't2',
              child_run_id: `${runId}-child`,
              snapshot: { current_node: 'search', total_sources: 4, total_queries: 6, last_message: 'Durchsucht aktuelle Quellen …' },
            })]
            : []),
        ],
      },
      {
        delayMs: 1800,
        actions: [
          ...(policy.knowledge === 'available'
            ? [
              makeEvent(runId, next(), 'inqtrix.agent.activity', {
                kind: 'searching',
                operation: 'knowledge.search',
                task_id: 't1',
                status: 'completed',
                query: 'Welche internen Analysen bewerten die Compliance-Last fuer Anbieter?',
                current: 2,
                total: 2,
                metrics: { result_count: 4 },
              }),
              makeEvent(runId, next(), 'inqtrix.agent.task.finished', { task_id: 't1', ordinal: 0, tool_kind: 'rag_query', status: 'completed' }),
              makeEvent(runId, next(), 'inqtrix.agent.narration', {
                narration_id: 'n-task-t1',
                kind: 'task',
                phase: 'execution',
                text: 'Interne Berichte zu KI-Regulierung auswerten: Der Bestand belegt die Anforderungen fuer Hochrisiko-Systeme; Konformitaetsbewertung bindet 15-20% der Entwicklungskapazitaet.',
                final: true,
              }),
            ]
            : []),
        ],
      },
      // Mid-execution clarification gate (bundled at the wave end, §4
      // phase 2): the run PARKS until the user answers.
      {
        delayMs: 900,
        actions: [
          {
            clarifications: [demoClarification(runId)],
            runId,
            type: 'setAgentRunClarifications',
          },
          makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'clarification', previous_phase: 'execution' }),
          makeEvent(runId, next(), 'inqtrix.run.waiting', { status: 'waiting_for_input' }),
        ],
      },
    ]
  }

  function executionAfterClarification(
    runId: string,
    next: () => number,
  ): DemoStep[] {
    const policy = sourcePolicies.get(runId)
      ?? { web: 'available', knowledge: 'available' }
    return [
      {
        delayMs: 400,
        actions: [
          makeEvent(runId, next(), 'inqtrix.run.started', {}),
          makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'execution', previous_phase: 'clarification' }),
          ...(policy.web === 'available'
            ? [makeEvent(runId, next(), 'inqtrix.agent.task.started', { task_id: 't3', ordinal: 2, tool_kind: 'web_instant', attempt: 1 })]
            : []),
        ],
      },
      {
        delayMs: 1500,
        actions: [
          ...(policy.web === 'available'
            ? [
              makeEvent(runId, next(), 'inqtrix.agent.child.progress', {
                task_id: 't2',
                child_run_id: `${runId}-child`,
                snapshot: { current_node: 'evaluate', total_sources: 9, total_queries: 10, consolidated_claim_count: 12, last_message: 'Bewertet Belege …' },
              }),
              makeEvent(runId, next(), 'inqtrix.agent.task.finished', { task_id: 't3', ordinal: 2, tool_kind: 'web_instant', status: 'completed' }),
            ]
            : []),
        ],
      },
      {
        delayMs: 1900,
        actions: [
          ...(policy.web === 'available'
            ? [
              makeEvent(runId, next(), 'inqtrix.agent.task.finished', { task_id: 't2', ordinal: 1, tool_kind: 'web_research', status: 'completed', child_run_id: `${runId}-child` }),
              makeEvent(runId, next(), 'inqtrix.agent.narration', {
                narration_id: 'n-task-t2',
                kind: 'task',
                phase: 'execution',
                text: 'Aktuelle Marktlage recherchieren: 9 externe Quellen bestaetigen eine Konsolidierungswelle; kleinere Anbieter suchen Compliance-Partnerschaften.',
                final: true,
              }),
            ]
            : []),
          demoUsedSummaryAction(runId, policy),
          makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'evidence', previous_phase: 'execution' }),
        ],
      },
      ...(responseForms.get(runId) === 'chat'
        ? [
          {
            delayMs: 1400,
            actions: [
              makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'synthesis', previous_phase: 'evidence' }),
              makeEvent(runId, next(), 'inqtrix.node.model_resolution', demoModelResolution('agent_answer', modelSelections.get(runId))),
              { artifacts: [answerMeta(runId, 1, 'writing')], runId, type: 'setAgentRunArtifacts' as const },
              answerDetailAction(runId, 1, 'writing'),
              makeEvent(runId, next(), 'inqtrix.agent.artifact.created', { artifact_id: `answer-${runId}`, kind: 'answer', revision: 1, updated_by: 'agent' }),
              makeEvent(runId, next(), 'inqtrix.agent.narration', {
                narration_id: 'n-answer',
                kind: 'synthesis',
                phase: 'synthesis',
                text: 'Antwort verfasst.',
                final: true,
              }),
            ],
          },
          {
            delayMs: 1800,
            actions: [
              { artifacts: [answerMeta(runId, 2, 'ready')], runId, type: 'setAgentRunArtifacts' as const },
              answerDetailAction(runId, 2, 'ready'),
              makeEvent(runId, next(), 'inqtrix.agent.artifact.updated', { artifact_id: `answer-${runId}`, kind: 'answer', revision: 2, updated_by: 'agent' }),
            ],
          },
        ]
        : [
          {
            delayMs: 1400,
            actions: [
              makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'synthesis', previous_phase: 'evidence' }),
              makeEvent(runId, next(), 'inqtrix.node.model_resolution', demoModelResolution('agent_synthesis', modelSelections.get(runId))),
              { artifacts: [memoMeta(runId, 1, 'writing')], runId, type: 'setAgentRunArtifacts' as const },
              memoDetailAction(runId, 1, 'writing', 1),
              makeEvent(runId, next(), 'inqtrix.agent.artifact.created', { artifact_id: `memo-${runId}`, kind: 'memo', revision: 1, updated_by: 'agent' }),
              makeEvent(runId, next(), 'inqtrix.agent.narration', {
                narration_id: 'n-synthesis',
                kind: 'synthesis',
                phase: 'synthesis',
                text: "Ich schreibe jetzt das Memo 'Markteinschaetzung: EU AI Act und der KI-Mittelstand' mit 3 Abschnitten.",
                final: true,
              }),
            ],
          },
          {
            delayMs: 2600,
            actions: [
              memoDetailAction(runId, 2, 'writing', 2),
              makeEvent(runId, next(), 'inqtrix.agent.artifact.updated', { artifact_id: `memo-${runId}`, kind: 'memo', revision: 2, updated_by: 'agent' }),
              makeEvent(runId, next(), 'inqtrix.agent.narration', {
                narration_id: 'n-section-1',
                kind: 'synthesis',
                phase: 'synthesis',
                text: "Abschnitt 'Aktuelle Entwicklung' geschrieben.",
                final: true,
              }),
            ],
          },
          {
            delayMs: 2600,
            actions: [
              memoDetailAction(runId, 3, 'writing', 3),
              makeEvent(runId, next(), 'inqtrix.agent.artifact.updated', { artifact_id: `memo-${runId}`, kind: 'memo', revision: 3, updated_by: 'agent' }),
              makeEvent(runId, next(), 'inqtrix.agent.narration', {
                narration_id: 'n-section-2',
                kind: 'synthesis',
                phase: 'synthesis',
                text: "Abschnitt 'Gegenposition' geschrieben.",
                final: true,
              }),
            ],
          },
        ]),
      {
        delayMs: 1500,
        actions: [
          makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'critic', previous_phase: 'synthesis' }),
          // Deep: the verification pass is demo-visible as
          // its deterministic narration line.
          ...(depths.get(runId) === 'deep'
            ? [
              makeEvent(runId, next(), 'inqtrix.agent.narration', {
                narration_id: 'kernel_deep_review',
                kind: 'synthesis',
                phase: 'critic',
                text: 'Verifikations-Durchlauf: keine Befunde, Antwort besteht.',
                final: true,
              }),
            ]
            : []),
        ],
      },
      ...(patchTargets.has(runId)
        ? [
          {
            delayMs: 1600,
            actions: [
              memoDetailAction(runId, 3, 'ready', 3),
              makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'patch', previous_phase: 'critic' }),
              ...demoPatchActions(runId, patchTargets.get(runId) as string, next),
              makeEvent(runId, next(), 'inqtrix.run.waiting', { status: 'waiting_for_approval' }),
            ],
          },
        ]
        : [
          {
            delayMs: 1600,
            actions: [
              // Chat runs deliver the inline answer — a final memo write
              // here would wrongly re-attach the canvas deliverable.
              ...(responseForms.get(runId) === 'chat'
                ? []
                : [memoDetailAction(runId, 3, 'ready', 3)]),
              makeEvent(runId, next(), 'inqtrix.agent.phase.changed', { phase: 'done', previous_phase: 'critic' }),
              makeEvent(runId, next(), 'inqtrix.run.completed', {}),
            ],
          },
        ]),
    ]
  }

  /** Patch proposal built from the REAL document content (deterministic). */
  function demoPatchActions(
    runId: string,
    documentId: string,
    next: () => number,
  ): ResearchDeskAction[] {
    const markdown = getDocumentMarkdown(documentId) ?? ''
    const paragraph = markdown
      .split('\n\n')
      .map((block) => block.trim())
      .find((block) => block && !block.startsWith('#') && block.length > 60)
    const edits = [
      ...(paragraph
        ? [{
          id: 'ed_1',
          find: paragraph,
          quote_before: '',
          quote_after: '',
          position: 'replace' as const,
          text: `${paragraph}\n\nDie aktuelle Recherche bestaetigt diese Einschaetzung und ergaenzt sie um die Konsolidierungswelle bei Compliance-Plattformen [W1].`,
          note: 'Absatz um aktuelle Recherche-Erkenntnis ergaenzt.',
        }]
        : []),
      {
        id: paragraph ? 'ed_2' : 'ed_1',
        find: '',
        quote_before: '',
        quote_after: '',
        position: 'append' as const,
        text: '## Empfehlung\n\nDie Compliance-Roadmap sollte um ein Szenario fuer beschleunigte Erstzertifizierung ergaenzt werden [W2].',
        note: 'Neuer Empfehlungsabschnitt aus dem Memo.',
      },
    ]
    const wire: AgentPatchWire = {
      patch_id: `pch-${runId}`,
      document_id: documentId,
      run_id: runId,
      source: 'agent',
      status: 'pending',
      edit_count: edits.length,
      summary:
        'Zwei praezise Aenderungen: Kernabsatz aktualisiert, Empfehlungsabschnitt ergaenzt.',
      revision_before: 1,
      applied_revision: null,
      created_at: Date.now() / 1000,
      decided_at: null,
      edits,
      warnings: [],
      applied_edit_ids: null,
      note: '',
      document_revision: 1,
    }
    lastDemoPatchWires.set(wire.patch_id, wire)
    lastDemoPatchEdits.set(wire.patch_id, edits)
    return [
      {
        patch: {
          patch_id: `pch-${runId}`,
          document_id: documentId,
          run_id: runId,
          source: 'agent',
          status: 'pending',
          edit_count: edits.length,
          summary: 'Zwei praezise Aenderungen: Kernabsatz aktualisiert, Empfehlungsabschnitt ergaenzt.',
          revision_before: 1,
          applied_revision: null,
          created_at: Date.now() / 1000,
          decided_at: null,
          edits,
          warnings: [],
          applied_edit_ids: null,
          note: '',
          document_revision: 1,
        },
        runId,
        type: 'setAgentRunPatch',
      },
      makeEvent(runId, next(), 'inqtrix.agent.patch.proposed', {
        patch_id: `pch-${runId}`,
        document_id: documentId,
        edit_count: edits.length,
      }),
      {
        approvals: [
          {
            ...demoApproval(runId),
            approval_id: `apr-patch-${runId}`,
            kind: 'patch' as const,
            subject_type: 'editor_patch',
            subject_id: `pch-${runId}`,
          },
        ],
        runId,
        type: 'setAgentRunApprovals',
      },
    ]
  }
}

function demoApproval(runId: string): AgentApprovalWire {
  return {
    approval_id: `apr-${runId}`,
    run_id: runId,
    kind: 'plan',
    status: 'pending',
    subject_type: 'plan',
    subject_id: `plan-${runId}`,
    payload: {},
    decision: '',
    note: '',
    decided_by_user_id: null,
    created_at: Date.now() / 1000,
    decided_at: null,
  }
}

function demoToolApproval(runId: string, query: string): AgentApprovalWire {
  return {
    ...demoApproval(runId),
    approval_id: `apr-tool-${runId}`,
    kind: 'tool',
    subject_type: 'tool',
    subject_id: 'web_instant',
    payload: {
      // Production carries recency in the approval PAYLOAD, not the
      // tool args (web_instant(query) has no recency field) — the edit
      // form must show exactly the one editable arg it does in a real
      // run, so keep args to {query} here too.
      actions: [
        {
          tool: 'web_instant',
          args: { query },
          summary: 'Eine direkte Websuche ausfuehren.',
        },
      ],
      recency: 'month',
    },
  }
}

function demoClarification(runId: string): AgentClarificationWire {
  return {
    clarification_id: `clr-${runId}`,
    run_id: runId,
    question:
      'Sollen auch Nicht-EU-Märkte betrachtet werden? '
      + 'Welche Aspekte sind am wichtigsten?',
    options: [],
    questions: [
      {
        id: 'q1',
        prompt: 'Sollen auch Nicht-EU-Märkte betrachtet werden?',
        options: [
          { id: 'q1_o1', label: 'Nur EU', description: '' },
          {
            id: 'q1_o2',
            label: 'EU + wichtigste Drittmärkte',
            description: 'USA, UK und Schweiz zusätzlich',
          },
        ],
        multi_select: false,
      },
      {
        id: 'q2',
        prompt: 'Welche Aspekte sind am wichtigsten?',
        options: [
          { id: 'q2_o1', label: 'Pflichten', description: '' },
          { id: 'q2_o2', label: 'Fristen', description: '' },
          { id: 'q2_o3', label: 'Sanktionen', description: '' },
        ],
        multi_select: true,
      },
    ],
    answers: {},
    default_assumption: 'Nur EU',
    status: 'pending',
    answer: '',
    option_id: '',
    answered_by_user_id: null,
    created_at: Date.now() / 1000,
    answered_at: null,
  }
}
