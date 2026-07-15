# Agent platform (Agent Desk)

> Files: `src/inqtrix/agents/`, `src/inqtrix/capabilities/`,
> `src/inqtrix/core/results.py`,
> `src/inqtrix/services/agent_control_service.py`,
> `src/inqtrix/server/routers/agent_runs.py`, migrations 0029/0030

## Scope

`mode=workspace_agent` runs a staged, human-supervised research agent: it
explores read-only, proposes an editable plan, waits for approval,
executes tasks (child research runs, instant web lookups, in-process
knowledge retrievals, quarantined file analysis), merges evidence with
contradiction analysis, and synthesizes a cited memo artifact that a
critic reviews once. Everything user-negotiable is persisted in Inqtrix
control tables; run events are signals, rows are the truth.

## Agent tiers

| Tier | Mode(s) | Control flow | Tools | Interrupts |
|---|---|---|---|---|
| S0 | `direct_llm` | code, one call | none | no |
| S1 | `research`, `knowledge` | bounded loop | web/RAG read-only | no |
| S2 | `workspace_agent` | checkpointed phase machine | capabilities, children | yes |

The escalation doctrine is "smallest sufficient tier"; the UI sells
modes, never this taxonomy.

## Speed/depth Stufen (schnell | gruendlich | tief)

Orthogonal to the S0-S2 taxonomy above, the user-facing **Stufe** dial
selects how much work one Agent Desk run may do. All budgets live in ONE
frozen table (`agents/tier_policy.py`, `TIER_POLICIES`) that every
consumer reads — routing, clarification caps, the planner prompt, the
plan validator (`web_research_profile_ceiling`), child research
profiles, the plan-gate rule, synthesis form, and verification level.
Published equals enforced: `/v1/capabilities` exposes the same table as
`agent.tiers` + `agent.default_tier`.

| Stufe | Clarification | Plan gate | Web budget | Verification | Output |
|---|---|---|---|---|---|
| `schnell` | never (visible assumptions) | only under `strict` autonomy | exactly one `web_instant`; `web_research` rejected by the validator | citation labels only (critic skipped) | chat answer |
| `gruendlich` (default) | max. 1 round, blocking gaps only | per autonomy | children default to profile `schnell` (1 round), ceiling `compact` | critic (max. 1 revision) + advisory quote grounding | intake decides |
| `tief` | up to 2 rounds | per autonomy | children default `compact`, ceiling `deep` | additionally: unverified web-cited quotes flip a passing critic verdict to `revise` once | canvas report |

Admission is additive: `agent_overrides.agent_tier` (or the `AGENT_TIER`
env default) selects the Stufe; sending contradictory `depth` and
`agent_tier` in one request is a 400, while an explicit request `depth`
wins over a contradicting env-level tier (request beats env; the run
then keeps legacy depth semantics, logged visibly). `tief` bridges to
`depth=deep` internally so every existing depth consumer behaves
unchanged. The tier never influences model routing — model choice stays
exclusively with the model picker.

Engine scope: the per-task Suchtiefe ceiling applies to mission-engine
plans (the plan gate is where per-task profiles exist). Kernel-engine
(`agent_kernel`) child research always runs the tier's DEFAULT child
profile; the `schnell` tier additionally removes the kernel tools
`ask_user`, `run_web_research`, `run_deep_mission`, and `write_canvas`
at the dispatch chokepoint and clamps the iteration ceiling.

## Report quality pipeline

Deterministic quality building blocks live in `agents/report_quality.py`
and are shared by BOTH engines (mission machine and kernel):

* **Citation validation** — cited `[K#]`/`[W#]` labels must exist in the
  evidence ledger; the mission synthesis gets exactly one bounded LLM
  repair round, the kernel's `write_canvas` rejects unknown labels with
  a loud tool error (the kernel loop is its own repair round).
* **Quote grounding** — quoted passages are verified verbatim against
  the STORED evidence texts (internal chunks and web excerpts alike) via
  the knowledge normalizer (`knowledge/grounding.py::quote_is_verbatim`).
  Verification is excerpt-based by design: nothing is re-fetched and
  model memory is never consulted, so unverifiable quotes stay visible
  in the critic facts instead of being trusted.
* **Evidence ranking** — above `synthesis_evidence_budget` references,
  the outline/answer PROMPT digest is capped to the top-ranked entries
  (source tier from `domains.py`, cross-task corroboration, excerpt
  presence — no LLM call). The citation ledger itself is never
  truncated, so every cited label still resolves.
* **Report guidance** — the plan-approval decision body accepts an
  additive `report_guidance` string (max. 2000 chars, stored in the
  approval's `decision_payload`); it renders as a user-requirements
  block in the outline, section, and answer prompts of that run.

## Execution model

The algorithm (`agents/algorithm.py`) is a LangGraph state machine whose
nodes are the phases below — deterministic control flow, LLM calls only
INSIDE phases through the shared structured-call helper. It executes in
run-store worker threads like every other algorithm and is registered
conditionally (see the feature gate).

```
intake ─┬─ ask_user_first ─ clarify ─┐
        ├─ plan_now ─────────────────┤
        └─ discover ─ discovery ─ [clarify blocking gaps] ─┐
                                                           ▼
   plan (validate, 1 repair) ─ [plan approval] ─ execute waves
        ▲                                             │
        └── replan gate (≤2 additive rounds) ◄─ evidence merge
                                                      │
                     synthesize (memo sections) ─ critic (≤1 revision)
                                                      │
                       [patch ─ patch approval]* ─ finalize
```

\* Only when the assignment targets an editor document (`document_id`
on the run request): the agent proposes anchored edits through the ONE
editor-instruct pipeline and parks for the patch approval — ALWAYS, in
every autonomy mode (the E16 write invariant). The agent never applies;
apply happens solely through `POST /v1/editor/patches/{id}:apply`.

* **Interrupts**: approval/clarification nodes call LangGraph
  `interrupt()`; the algorithm then parks the run
  (`waiting_for_approval`/`waiting_for_input`, M3) and returns. The
  decision endpoints (M4) record the decision and resume the run
  atomically; on re-entry the algorithm reads the decision from the
  control store and fast-forwards via `Command(resume=...)`. Node writes
  that precede an interrupt are idempotent (deterministic ids per
  `(run_id, kind, round)`), because interrupted nodes re-execute from
  their start on resume.
* **Effective actor and resume authority**: a run keeps its original owner,
  but every executable segment has an explicit `execution_actor_user_id` and
  scope ceiling. Approve, edit, or resume atomically records the deciding
  editor as the effective actor together with the control decision, run
  revision, and next status. The worker resolves that canonical user live and
  uses the same actor for quotas, audit, child runs, skills, knowledge, and
  tools. It never reconstructs or falls back to the owner's authority. A
  shared editor can therefore continue the run only with resources that editor
  may currently use.
* **Pinned dependencies and revocation safepoints**: an unscoped "all visible
  collections" request is normalized to concrete collection ids before
  execution, and every attached skill is pinned by id plus integer revision.
  The worker re-checks the actor, run edit authority, collection visibility,
  and skill revision before and after provider/search calls and at tool,
  knowledge, skill, child-run, segment, and final-publication boundaries. If
  access disappears, the run ends with `authorization_revoked`; it does not
  continue with a silently smaller evidence set. A remote call already in
  flight cannot be recalled, but its return value is discarded after the
  post-call check and is not persisted or emitted.
* **Children wait (A1)**: `web_research` tasks are SUBMITTED as child
  runs and awaited through the same interrupt machinery — the execute
  node submits one batch per wave (width `max_parallel_children`), then
  the `children_wait` node interrupts and the run parks slot-free as
  `waiting_for_children`. The run store's terminal choke point re-queues
  the parent inside the LAST child's terminal transaction (never lost),
  and the park itself re-probes in the same transaction (lost-wakeup
  self-heal). On resume, `children_wait` folds each child's terminal
  outcome from the run store (rule R5); an explicitly classified transient
  provider/transport failure is resubmitted at most once with the unchanged
  operator-owned limits and the node parks again. Invalid input, policy,
  cancellation, authentication, and token-budget failures are never retried.
  A parent therefore never block-polls siblings out of
  the shared execution pool — an undersized pool serialises a wave
  instead of deadlocking it.
* **Checkpointing**: `PostgresSaver` (sync — worker threads) with
  `thread_id=run_id`; tables are library-owned (`saver.setup()`), NOT
  Alembic-managed. The checkpoint is ONLY a resumability cache (rule
  R5): wiping it fails in-flight runs LOUDLY ("Checkpoint ... verloren")
  while plans/artifacts stay intact and the assignment restarts as a new
  run. Terminal runs delete their checkpoint thread.
* **Autonomy (E16)**: `strict` approves the discovery probe list AND
  every plan delta; `balanced` (default) approves the initial plan,
  auto-applies replans of ≤2 new read-only tasks; `autonomous` skips
  plan interrupts (the plan is still persisted + visible). Write-effect
  capabilities are gated in every mode (arrive with the editor patch
  flow).
* **Gate rejection is a clean finish, not a failure.** Rejecting the
  plan, replan or discovery approval completes the run normally with a
  deterministic German receipt as its chat answer (the reject `note` is
  echoed as a blockquote); a rejected replan that already synthesized a
  draft keeps that draft as the deliverable. The approval/plan rows keep
  the durable `rejected` status, `result_state` carries
  `plan_decision: "rejected"`, the approval resume payload includes the
  decider's `note`, and finalize skips memory-candidate staging. (Same
  contract as patch rejection, which already ended the run normally.)
* **Discovery probes** (§4 Phase 1): deterministic, budget-capped
  (`discovery_max_tool_calls`). The wired set is P1 `knowledge.collections
  .list`, P2 `knowledge.search` per sub-goal, and the optional P6
  `web.search.instant` preview. The plan's P3-P5 are deliberately not
  probed: P3 (prior-run reuse) is a named follow-up needing a runs-read
  capability plus a similarity heuristic (a new catalog for one consumer;
  within a session the prior context already rides the memo lineage); P4
  (referenced files) has no input — a run carries collection ids and one
  patch-target document, not a list of @-mentioned files; P5
  (canvas/editor context) is already covered by the intake memo read (E15)
  and the patch-phase document read. See `agents/discovery.py`.
* **Tools per task** (E18/E19): normal Mission plans express each independent
  web evidence question as one `web_instant` task. That task contains exactly
  one self-contained natural-language question and performs exactly one
  grounded `SearchProvider.search()` through the capability registry; tasks
  in the same dependency wave may execute concurrently. `web_research` is
  reserved for Deep runs or an explicit user research directive/edit and
  spawns one CHILD research run (`kind=agent_child`): its listed strings are
  joint guidance questions for that child, not literal provider calls. The
  server selects `compact` for an explicit normal-depth child and `deep` for
  Deep; the public Research Desk's `schnell` report profile is not an Agent
  Desk child profile. Per-task model-generated token/time/child budgets are
  not execution authority; deployment quotas and report-profile timeouts are.
  `rag_query` runs the knowledge algorithm
  IN-PROCESS with its retrieval profile (`schnell`/`standard`/
  `gruendlich`/`tief`); `file_analysis` summarizes internal document
  content behind the deepagents quarantine seam (`agents/harness.py` —
  large documents go through a sub-agent with a file workspace, the
  caller only ever sees the compact summary).
* **Task-row truth.** Plan task rows transition idempotently from `pending` to
  `running` and then to `completed`, `failed`, `insufficient_evidence`,
  `skipped`, or `cancelled`; an in-flight synchronous provider call first
  records `cancel_requested`. The row stores the delegated child id and bounded
  result summary; `result_payload.answer_markdown` stores the complete task
  output without a character/word cutoff. The overview reads only the summary,
  while `GET /v1/runs/{run_id}/tasks/{task_id}/result` lazy-loads the full
  Markdown, references, metrics, and error. Events remain the chronological
  signal. A reconnect or terminal-run reload therefore reconciles status from
  control rows before replaying event detail. `POST
  /v1/runs/{run_id}/tasks/{task_id}/cancel` stops a pending task immediately,
  delegates to the existing child-run cancellation path when applicable, and
  suppresses retry/result commit for an in-flight instant operation without
  claiming that its synchronous network request was hard-killed.
* **Scoping**: every internal tool call is pinned to the current effective
  actor's live resource rights. Explicit collection ids are asserted; an
  omitted scope is normalized to that actor's visible collections rather than
  being rediscovered under the owner later. Limitation: with collections on
  DIFFERENT embedding models, an un-pinned retrieval fails loudly per
  task ("query one model scope at a time") — pin `collection_ids` in
  the plan for mixed-model deployments.
* **Evidence**: dedup by the platform citation identity
  (`doc:{id}#{chunk}` / canonical URL) and assign stable `K#`/`W#` labels in
  first-seen order. Exact retrieval fields (`source_text`/`excerpt`) remain
  source passages. If an instant provider exposes only URLs, the bounded
  `grounded_support` field preserves the statement around that citation from
  the provider-grounded answer; it is explicitly provenance context, never a
  verbatim source excerpt. Duplicate observations may fill missing metadata
  but never replace stronger source text. Each synthesis section receives only
  the labels selected for that outline section, avoiding whole-ledger citation
  flooding. Contradiction analysis runs one mid-tier call over lexically
  overlapping claim pairs only; sufficiency is deterministic criterion
  coverage plus one fast-tier three-way verdict. Workspace and kernel results
  export the same structured references; the kernel persists its trusted
  `reference_id` ledger in the existing `evidence_bundle` artifact so
  park/resume and `read_canvas`/`write_canvas` retain labels and evidence.
* **Synthesis**: outline first, then one call per section; every section
  flushes into the memo artifact (revision++, `artifact.updated`), so
  the canvas streams section-wise without a delta protocol (E12). Quotes
  are verbatim-verified through the knowledge grounding normalizer;
  unverified quotes stay visible, never block. A follow-up turn in the
  same session reads the ONE session memo (`(session_id,'memo')`) at
  intake — including any edit the user made after the previous turn — so
  the outline CONTINUES it (E15, "latest revision wins") and each flush
  CAS-guards against that revision. If the user edited it concurrently the
  guarded write raises the same conflict as a user PUT: the agent keeps
  the user's text, appends its update, and emits `artifact.edit_conflict`
  rather than overwriting (E13/R10 — the agent's symmetric half of the
  optimistic-concurrency contract). There is no `context_artifact_ids`
  wire field: the session key is the single source for the memo, so an
  explicit id list would be unconsumed surface.
  Agent-generated Markdown is normalized at generation/write boundaries so
  currency dollars cannot open accidental KaTeX spans. Fenced/inline code,
  URLs, block math, and recognized inline formulas remain byte-preserved;
  raw `grounded_support` provenance is deliberately not presentation-escaped.
  Synthesis accepts only citation labels that exist in the canonical evidence
  ledger. An unknown label triggers exactly one complete-text repair call;
  empty or still-invalid repaired output fails loudly and is never persisted.
* **Critic**: judges against precomputed facts (citation coverage, quote
  verification, contradiction mentions); `revise` buys exactly one
  revision. Long-term memory, when enabled, reaches the critic only as
  non-citable context; current evidence wins, and contradictions between
  memory and evidence are surfaced as `memory_conflict` activity. The
  report lands as a `critic_report` artifact and in `result_state`.
* **Long-term memory**: optional and personal. Run memory remains the
  control/artifact system and session memory remains the session memo.
  Accepted long-term memories live behind `AgentMemoryProvider` (first
  provider: self-hosted Mem0) and the Inqtrix `AgentMemoryService`.
  Anonymous/static principals cannot use it; namespaces are derived
  server-side from `(tenant_id, principal.user_id)`, and client owner fields
  are rejected. The default learning mode is candidate-only: finalize
  can stage `MemoryCandidate` rows, but users accept/edit/reject them in
  Settings. Memory is never evidence and must not be cited.
* **Patch (M7)**: document-targeted assignments validate the target at
  INTAKE (an invisible or mistyped document fails in seconds, before any
  research spend); the patch phase re-loads it with the effective actor's
  visibility, proposes edits via the shared instruct pipeline (memo as
  reference material, the editor-assistant timeout window), persists ONE
  `editor_patches` row (the artifact row carries only `{patch_id}`, rule
  R3), and interrupts with approval `kind='patch'`. Rejecting the gate
  also rejects the patch row and ends the run normally — the memo
  remains the deliverable; `result_state` records `patch_decision`.
  Proposal failures (timeout, parse failure, oversized document) are
  HARD run failures, never a silent "no changes needed".

## Feature gate (E8)

The algorithm registers only when checkpoints can survive:
`INQTRIX_STORAGE_BACKEND=postgres`, or the dev escape
`INQTRIX_AGENT_ALLOW_VOLATILE=true` (InMemorySaver, WARNING logged,
`workspace_agent_durable: false`). Otherwise `mode=workspace_agent` is
the loud 400 listing available modes and `/v1/capabilities` reports
`features.workspace_agent: false`. Queue-mode parking is fenced: the
worker's handle parks with the claim fence and ACKs the dispatch message;
the resume re-enqueues a fresh message that any worker continues from the
persisted payload + checkpoint.

## Settings (`INQTRIX_AGENT_*`)

`enabled`, `max_parallel_children` (6), `discovery_max_tool_calls` (15),
`max_plan_tasks` (8), `max_replan_rounds` (2),
`max_clarification_rounds` (2), `default_autonomy` (`balanced`),
`allow_web_discovery_preview` (on), `advanced_autonomy` (off),
`allow_volatile` (off). All enforced in code, never prompted. Model
tiers: `agent_intake`/`agent_sufficiency`/`agent_critic` fast,
`agent_discovery_analyst`/`agent_contradiction`/`agent_file_analysis`/
`agent_answer_light` mid, `agent_plan`/`agent_synthesis`/`agent_answer`
high — overridable through the existing
`agent_overrides.model/model_tier/effort` and per-task
`params.model_tier`.

## Interaction layer (plan M1)

* **Two permission modes (E16 amendment).** The UI presents
  Standard/Auto (`/v1/capabilities` `agent.mode_presets`, mapped onto
  the unchanged wire vocabulary `balanced`/`autonomous`;
  `advanced_autonomy` republishes the legacy three-way control incl.
  `strict`). In Standard the approved plan is the ONLY web-search
  consent surface — plan tasks carry their verbatim queries (shown
  inline in the approval tray), a replan that adds web tasks always
  re-gates (`INTERNAL_READ_ONLY_TOOLS` in `agents/replan.py`), and the
  discovery web preview runs only in `autonomous`.
* **Structured clarifications.** A gate round carries 1-3 questions
  with 2-4 pickable options each (`ClarificationRecord.questions`,
  sanitized deterministic ids); free text stays possible per question
  AND as a whole-round answer (`answer`), so older clients keep
  working. Answers land in `answers` (per-question map) and compose
  into the resume history through ONE shared function
  (`round_qa_lines`).
* **Chat-form deliverable.** `response_form` on the run request
  (`auto|chat|canvas`, agent-only) or the intake profile decides the
  turn's deliverable: `chat` writes a run-local `answer` artifact
  rendered inline in the transcript (one `write_chat_answer` call —
  no outline loop; R1 model routing uses the approved plan contract:
  any web task selects the high tier even when retrieval returned no
  references, while a plan without web tasks selects
  `agent_answer_light`/mid), `canvas` keeps the session-memo
  path (E15 untouched). A patch assignment always uses canvas.
* **Session context (K1-K4).** Every follow-up composes session metadata
  server-side from durable rows (`list_session_runs` + clarifications +
  approvals + result answers + `list_session_artifacts` registry). An
  explicit `request.history` replaces only the K1 conversational history;
  the artifact registry, last effective output form, and prior canonical
  evidence count are still reconstructed. The deterministic trim policy
  drops older turns first and visibly truncates an oversized newest turn.
* **Event transport fallback (T).** `GET /v1/runs/{id}/events?format=json
  [&after=N]` returns the SAME replay buffer as an immediate JSON page
  (`{data, terminal}`); the frontend channel degrades from SSE to this
  poller (and back) with a visible hint when proxies buffer SSE.

Memory knobs: `memory_provider` (`none`/`mem0`), `memory_mode`
(`off`/`candidate_only`/`auto_safe`), `mem0_base_url`, `mem0_api_key`.
`auto_safe` is accepted for forward compatibility but currently reports
`effective_mode: candidate_only` with a degraded reason; it does not
auto-retain memories yet.

## Cognitive kernel (`mode=agent_kernel`, plan M2, ADR-PLAT-2)

A SECOND registered algorithm next to the phase machine: an LLM
tool-calling loop on deepagents, assembled through the ONE harness seam
(`agents/harness.py::build_kernel_agent`, pin `deepagents>=0.6.12,<0.7`
with contract tests in `tests/agents/test_harness_kernel_contract.py`
as the upgrade gate). The phase machine stays registered — it is the
kernel's `run_deep_mission` child capability, and parked missions must
resume under their own algorithm.

* **Two-layer architecture.** The kernel (`agents/kernel/`) is the
  conversational brain: it answers directly, asks structured questions
  (`ask_user` -> the M1 clarification machinery, deterministic ids
  hashed from the checkpointed `tool_call_id`), searches, writes canvas
  deliverables, and DELEGATES multi-strand researched deliverables to
  the phase machine as child runs.
* **Same platform seams.** Registry algorithm behind the normal
  RunService -> queue -> worker path; sync `graph.stream` only; shared
  checkpointer with `thread_id=run_id`; park/resume against control
  rows (R5). Three compiled graph variants (one per autonomy policy —
  `interrupt_on` is compile-time); per-segment state travels through a
  ContextVar, so the compiled graphs are stateless and shared across
  users.
* **Policy gates.** `agents/kernel/policy.py` maps autonomy onto HITL
  gates: balanced gates `web_instant` (query verbatim in the approval
  payload), UN-scoped `search_project_knowledge` (a `when` predicate),
  and both child-run tools; strict gates every capability tool;
  `propose_editor_patch` gates in EVERY mode (E14). A gate parks the
  run as a `kind="tool"` approval whose id derives from the
  checkpoint-stable interrupt id; decisions map onto the HITL resume
  contract (approve/reject fan out over the batch, edit replaces the
  args of exactly one action, tool never swappable).
  Edited arguments are validated against the selected tool's real Pydantic
  input model; unknown fields and type errors are rejected. Tool name and
  stored action id are immutable. Identity, tenant, run, workspace, and
  authorization context are never editable model arguments — the server
  injects them and rechecks the target resource at execution.
* **Tools.** All wrappers over the wave-1 capability registry or
  platform services, identity injected via `CapabilityContext` (never a
  model-controlled argument, E5). Denials and outages return as VISIBLE
  tool results the prompt obliges the model to acknowledge.
  `write_canvas` writes session-scoped `deliverable` artifacts with
  optimistic concurrency (`expected_revision`); child-run tools park
  slot-free (`waiting_for_children`) and re-find their child via
  `origin_key=tool_call_id` on resume re-execution (persisted in the
  replay payload, no schema change).
* **Budgets.** `kernel_max_iterations` restores a hard
  `recursion_limit` (deepagents lifts the default); the model-call
  boundary checks the cancel event AND the run-cumulative token budget
  (prior segments recovered from checkpointed `usage_metadata`).
* **Continuity + events.** The per-run user message carries the K1
  session context (history block, artifact registry for CAS updates,
  response-form hints) — the system prompt stays compile-time static.
  The update stream emits `tool.started/finished` (redacted previews),
  `todo.updated`, deterministic `narration` (content-hash ids), and
  `phase.changed` for timeline compatibility.
  Run SSE independently re-resolves the credential and current run access
  immediately before each data frame. Revoking a viewer closes only that
  viewer's stream; revoking or downgrading the effective actor also terminates
  execution at the next safepoint. Reconnect after revoke cannot replay older
  frames to the former recipient.
* **Rollout.** `INQTRIX_AGENT_KERNEL_ENABLED` (default off) gates
  registration together with the checkpointer rule, deepagents
  availability, and `supports_tool_calls()` on the default provider
  (failed gates WARN). Capabilities publish `features.agent_kernel` and
  the effective `agent.default_mode` (`INQTRIX_AGENT_DEFAULT_MODE`
  falls back to `workspace_agent` while the kernel gate fails).

## Source policy and one-shot routes

Source availability is a server-enforced run policy, independent of whether
the model ultimately chooses a tool. Agent-native `POST /v1/runs` requests may
add:

```json
{
  "source_policy": {
    "web": "available",
    "knowledge": "available"
  },
  "execution_directive": "quick_web"
}
```

`source_policy.web` and `source_policy.knowledge` accept `available` or
`disabled`; omitting the block preserves the historical behaviour in which
both source families are available. The policy reaches both agent engines,
their planner/tool dispatch chokepoints, and delegated child runs. Disabling a
source removes its tasks from planning and rejects a violating tool/task
before it can contact the source. The Agent Desk persists this preference per
session in that session's existing `items_json`; it sends the current value on
each run, so no run-table or session-table schema change is required.

`execution_directive` is a one-message route, not a prompt hint:

* `quick_web` forces `agent_kernel`, chat output, normal depth, and an
  effective web-only source policy. It derives exactly ONE search query from
  the current question plus bounded history, invokes
  `web.search.instant` exactly once, then synthesizes a grounded chat answer
  from that result. It creates no plan, research child, knowledge retrieval,
  or canvas artifact. Query derivation and answer synthesis both honor the
  requested model and reasoning effort. The Standard (`balanced`) policy
  treats the explicit directive as consent for this one external request;
  `strict` persists the derived query in the normal tool-approval record,
  parks before search, and reuses the reviewed query on resume.
* `knowledge_only` forces `agent_kernel`, chat output, normal depth, and an
  effective knowledge-only source policy. The kernel may search/read project
  knowledge and ask a clarification, but web, mission, canvas, and editor
  tools are excluded for that message. Insufficient project evidence must be
  reported rather than silently filled with world knowledge.

The older `tool_directives` field remains accepted as a planner/kernel hint
for compatible clients. A request containing both
`execution_directive` and `tool_directives` is ambiguous and fails with HTTP
400; the Agent Desk no longer creates legacy directives.

Policy precedence is fixed and restrictive: deployment availability,
identity/visibility, strict approval, and write protection first; activated
skill policy second; the one-shot execution route third; the session source
policy fourth; automatic tool choice last. A lower layer can narrow what the
agent may do but cannot reopen a source or side effect denied above it.

`/v1/capabilities` publishes `agent.source_controls[]` (`id`, `default`,
`available`) and `agent.execution_directives[]` (`id`, `available`). Clients
must feature-detect these entries: a disabled or unwired backend is an
unavailable control with an explanation, never a silent fallback to another
route.

Both agent engines project the same `execution` object into state-bearing run
snapshots, the terminal result state, and the public completed-result export:

`execution_directive`, `effective_mode`, `response_form`, `depth`, `model`,
`reasoning_effort`, effective `source_policy`, `consent_reason`, and
`tool_use_counts.web|knowledge`. The key set is total (empty optional strings
and zero counts are explicit), so a client never has to infer whether a source
was merely available or actually invoked. Successful source-tool calls advance
the counters; parked kernel segments restore them from the checkpoint. See
[Run events](../observability/run-events.md#run-summary) for the wire example.

## Skills (plan M3)

Skills are SERVER-ENFORCED policy objects, not a prompt category: the
entity (`content/skills.py::SkillRecord`, table `skill_templates`,
migration 0041 with the usual RLS recipe) bundles instructions,
clarification points, a deliverable hint, a plan requirement, a tool
whitelist, and the invocation rule under one `/label`. CRUD lives at
`/v1/skills` (mandatory integer `expected_revision`, 409 on a stale write,
sharing through
`resource_type="skill_template"` with consent); SKILL.md round-trips
through `content/skill_markdown.py` (agentskills.io-compatible, policy
fields under the `x-inqtrix` frontmatter key), exposed as
`GET /v1/skills/{id}/markdown` and `POST /v1/skills/import` — the
import parser maps only the file shape, the service validator stays
the single policy gate.

* **Placeholder coupling.** Every `{{name}}` in the instructions must
  have a same-named clarification point (loud 400); points without a
  placeholder are allowed as context-only inputs. At intake a
  fast-tier check (`SkillPointCheck`) decides which points the
  question/history already answer: missing REQUIRED points join the
  M1 options-chip clarification round (options map 1:1), missing
  optional points run on their `default_assumption` — visibly named
  in the run.
* **Admission.** The composer submits `skill_ids` (slash menu) plus
  `tool_directives` (hard planner/kernel hints from the closed
  `AGENT_TOOL_DIRECTIVES` whitelist — unknown ids are a loud 400).
  The runs router admits skills through live owner-or-accepted-direct-share
  visibility (indistinct 404, `skills_max_attached` cap) and pins each skill's
  integer `revision` into the stored run request. Every worker segment resolves
  the effective actor and re-checks both current visibility and the exact
  pinned revision. Both engines fail closed with `authorization_revoked` if
  access, policy, or instructions changed before a later segment.
* **`requires_plan` matrix.** `always` forces the plan gate in every
  mode (including Auto — the only way a skill can force review);
  `never` frees only the mission-weight default; `auto` follows the
  mode. The STRICTEST activated skill wins; the E16 web-regate and
  the patch gate are untouchable by skills.
* **Model activation.** Attached skills inject as delimited
  USER-content blocks (marked as user content that cannot lift
  security or approval rules), but only after the kernel middleware has
  run the same `SkillPointCheck` and durably resolved every required input.
  Missing inputs park in deterministic options-chip batches before the first
  main-model call. Dynamic `load_skill` uses the same resolver and activates
  the skill/tool restrictions only after its required inputs are answered.
  Own `model_allowed` +
  autocomplete-eligible skills disclose as one-liners under a
  deterministic character budget (`skills_disclosure_budget_chars`,
  visible overflow line). The kernel's `load_skill` tool activates a
  disclosed skill mid-run (gated in balanced/strict, free in
  autonomous); trusted `[skill_geladen:<id>@<revision>]` transcript
  markers re-arm the exact admitted revision on every resumed segment,
  so a tool limit acquired before a park survives park/resume without
  policy drift. Shared-in skills are
  structurally never model-activatable for the recipient.
* **`allowed_tools` enforcement.** The union over activated skills
  narrows the run at the dispatch chokepoints: kernel tool bodies
  refuse blocked tools with a visible tool error, and the phase
  machine's planner receives blocked task kinds as repair errors
  (`KERNEL_TOOL_TO_TASK_KIND`).
* **Capabilities.** `features.skills` plus
  `agent.skills {max_attached, disclosure_budget_chars}`; the skill
  LIST itself comes only from the authenticated `GET /v1/skills`
  (capabilities are unauthenticated).

Settings: `INQTRIX_AGENT_SKILLS_MAX_ATTACHED` (3),
`INQTRIX_AGENT_SKILLS_DISCLOSURE_BUDGET_CHARS` (4000).

## Deep mode (plan M4)

Thoroughness is ORTHOGONAL to the permission mode: `agent_overrides.depth`
(`normal` | `deep`, default `normal`, published as `agent.depth_modes` +
`agent.default_depth`) selects deterministic budget-plus-verification —
never an unbounded loop:

* **Effort.** The kernel node runs on `high` reasoning effort unless the
  request or a skill pin already chose one (precedence: explicit request
  effort > skill pin > deep > tier map). The tier stays with the tier
  map (`agent_kernel` is high-tier already).
* **Budgets.** The kernel `recursion_limit` rises to
  `INQTRIX_AGENT_KERNEL_MAX_ITERATIONS_DEEP` (40) — a raised, still
  hard-and-loud ceiling. Independently, all model-emitted tool calls count
  against a checkpoint-derived cumulative limit (30 normal, 60 deep); an
  overflowing multi-call batch is rejected before any tool executes.
* **Child profiles.** Normal kernel `run_web_research` children are forced
  to COMPACT; Deep children and every deep phase-machine child research run
  are forced onto the DEEP report profile
  (deterministic, with no model-controlled profile argument);
  a deep kernel run delegates `run_deep_mission` children with
  `depth=deep`, so the mission's own children inherit it.
* **Verification pass.** Before finalizing, a mid-tier rubric check
  (`agent_deep_review` in the tier map) receives the complete effective
  assignment plus chat and every canvas created or updated by the run.
  Findings target chat or a known artifact id. Exactly one revision call
  returns a complete chat text and content-only canvas replacements with
  exact expected revisions; multiple canvases commit through one batch CAS,
  preserving payloads and references. Unknown targets, parse/provider errors,
  empty revisions, and CAS conflicts leave all outputs unchanged and end with
  visible `kernel_deep_review` narration.
* **Prompt bias.** The deep user-message section biases delegation
  toward `run_deep_mission` and thorough research over instant search,
  and demands explicit assumptions. The phase machine keeps its critic
  as the built-in verification pass — Deep there means DEEP research
  children, not a second reviewer.

## Dependencies

Optional extra `agent`: `deepagents` (isolated behind
`agents/harness.py`; pulls `langchain-anthropic` and
`langchain-google-genai` unconditionally — see THIRD_PARTY_NOTICES),
`langchain`, `langgraph-checkpoint-postgres`, `psycopg`. Deployments
should set `LANGGRAPH_STRICT_MSGPACK=true`. Sync with
`uv sync --extra agent --extra knowledge-qdrant --extra queue-valkey`.

## Related docs

- [Run events](../observability/run-events.md) — waiting statuses,
  agent event catalog, control endpoints
- [Knowledge retrieval](knowledge-retrieval.md) — the profiles rag tasks
  select
- [LLM calls](llm-calls.md) — per-node tiering
