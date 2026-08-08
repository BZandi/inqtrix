"""The workspace-agent algorithm (Stufe 3, ``mode=workspace_agent``).

A CHECKPOINTED LangGraph phase machine — deterministic control flow, the
LLM only fills phase-shaped structured calls (§4). Human-in-the-loop
pauses are LangGraph ``interrupt()`` calls surfaced outward as the run
waiting statuses (M3): the algorithm parks the run and returns; the
decision endpoints (M4) resume it, and this class fast-forwards from the
checkpoint with the recorded decision (rule R5: the control store is the
truth, the checkpoint only the resumability cache).

Node writes that precede an ``interrupt()`` re-execute on resume, so
every such write is idempotent, keyed on ``(run_id, kind, round)``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, TypedDict

from inqtrix.constants import REASONING_TIMEOUT
from inqtrix.agents import (
    critic_phase,
    evidence,
    memory_reflection,
    report_quality,
    synthesis,
)
from inqtrix.agents.checkpoint_guard import ensure_checkpoint_restart_safe
from inqtrix.agents.memory_ports import AgentMemoryUnavailable
from inqtrix.agents.markdown import normalize_agent_markdown
from inqtrix.agents.patterns._structured import observe_structured_retries
from inqtrix.agents.telemetry import model_retry_activity, provider_retry_activity
from inqtrix.evidence import (
    attach_web_search_lineage,
    build_instant_web_search_ledger,
    merge_web_search_ledgers,
)
from inqtrix.agents.clarification import (
    blocking_questions,
    build_clarification,
    filter_repeated_questions,
    intake_questions,
    round_qa_lines,
    sanitize_questions,
)
from inqtrix.agents.skills_runtime import (
    allowed_task_kinds,
    build_skills_block,
    build_tool_directives_line,
    check_skill_points,
    skill_model_pins,
    skill_point_key,
    strictest_requires_plan,
    unanswered_required_points,
)
from inqtrix.agents.control_ports import (
    ApprovalRecord,
    ArtifactNotFound,
    ArtifactRecord,
    ArtifactRevisionConflict,
    ClarificationNotFound,
    PlanRecord,
    TASK_TERMINAL_STATUSES,
    PlanTaskRecord,
    settle_cancelled_plan_tasks,
)
from inqtrix.agents.discovery import (
    build_probe_plan,
    execute_probes,
    run_discovery_analysis,
)
from inqtrix.agents.harness import run_quarantined_file_analysis
from inqtrix.agents.intake import route_readiness, run_intake
from inqtrix.agents.limit_contract import LIMIT_REACHED_EVENT
from inqtrix.agents.phase_models import (
    AgentCriticFinding,
    AgentCriticReport,
    AssignmentProfile,
    ClarificationOptionModel,
    ClarificationQuestionModel,
    ContradictionReport,
    DiscoveryResult,
    MemoryReflection,
    ReportOutline,
    SufficiencyJudgement,
)
from inqtrix.agents.narration import (
    NARRATION_EVENT,
    discovery_narration,
    plan_narration,
    section_narration,
    synthesis_narration,
    task_narration,
)
from inqtrix.agents.plan_collections import CollectionCatalogEntry
from inqtrix.agents.plan_models import WEB_RESEARCH_PROFILE_ORDER
from inqtrix.agents.planner import (
    PlanningFailed,
    plan_to_records,
    run_planner,
)
from inqtrix.agents.replan import autonomy_auto_approves, evaluate_replan
from inqtrix.agents.session_context import build_session_context
from inqtrix.agents.scheduler import (
    TaskOutcome,
    execute_wave,
    project_child_run_outcome,
    should_retry,
    task_result_payload,
    task_result_summary,
    topological_waves,
)
from inqtrix.agents.source_policy import (
    allowed_task_kinds_for_policy,
    coerce_source_policy,
    effective_source_policy,
    require_task_allowed,
)
from inqtrix.agents.tier_policy import resolve_tier_policy
from inqtrix.agents.web_execution_policy import (
    WebResearchPolicy,
    derive_web_research_policy,
)
from inqtrix.exceptions import (
    AgentCancelled,
    AgentPolicyDenied,
    AgentTokenBudgetExceeded,
)
from inqtrix.execution_authority import pinned_knowledge_collection_ids
from inqtrix.execution_failures import (
    RETRYABLE_AGENT_TASK_ORCHESTRATION_CODES,
    classify_execution_failure,
)
from inqtrix.model_routing import (
    describe_resolution,
    describe_unresolved_resolution,
)
from inqtrix.providers.base import observe_provider_retries
from inqtrix.urls import sanitize_error

if TYPE_CHECKING:
    from inqtrix.agents.checkpointing import CheckpointerHandle
    from inqtrix.agents.control_ports import AgentControlStore
    from inqtrix.capabilities import CapabilityRegistry
    from inqtrix.core.context import RunContext, RuntimeContext
    from inqtrix.core.results import AgentResult, RunRequest
    from inqtrix.services.agent_context import AgentContextResolver
    from inqtrix.services.run_service import RunService
    from inqtrix.settings import AgentPlatformSettings

log = logging.getLogger("inqtrix")

_AGENT_NODES = (
    "agent_intake",
    "agent_discovery_analyst",
    "agent_skill_point_check",
    "agent_plan",
    "agent_contradiction",
    "agent_sufficiency",
    "agent_synthesis",
    "agent_answer",
    "agent_answer_light",
    "agent_critic",
    "agent_memory_reflection",
    "agent_file_analysis",
    "agent_patch",
)

_OVERRIDE_SCOPED_NODES = frozenset(
    {"agent_plan", "agent_synthesis", "agent_answer", "agent_answer_light"}
)
"""The THINKING nodes a per-run model/tier/effort override (request or
skill pin, R3/R4) applies to. Assembly-line nodes (intake, critic,
sufficiency, point check, ...) stay on the tier map — an explicit
"Opus, effort high" is meant to strengthen the reasoning, not to pay
frontier prices for classification calls. The kernel resolves only its
own node, so it needs no scope set."""

PHASE_CHANGED_EVENT = "inqtrix.agent.phase.changed"
PATCH_PROPOSED_EVENT = "inqtrix.agent.patch.proposed"
PLAN_PROPOSED_EVENT = "inqtrix.agent.plan.proposed"
PLAN_REVISED_EVENT = "inqtrix.agent.plan.revised"
APPROVAL_REQUESTED_EVENT = "inqtrix.agent.approval.requested"
CLARIFICATION_REQUESTED_EVENT = "inqtrix.agent.clarification.requested"
TASK_STARTED_EVENT = "inqtrix.agent.task.started"
TASK_FINISHED_EVENT = "inqtrix.agent.task.finished"
TASK_FAILED_EVENT = "inqtrix.agent.task.failed"
ARTIFACT_CREATED_EVENT = "inqtrix.agent.artifact.created"
ARTIFACT_UPDATED_EVENT = "inqtrix.agent.artifact.updated"
ARTIFACT_EDIT_CONFLICT_EVENT = "inqtrix.agent.artifact.edit_conflict"
ACTIVITY_EVENT = "inqtrix.agent.activity"

_WAITING_STATUS_BY_KIND = {
    "clarification": "waiting_for_input",
    "approval": "waiting_for_approval",
    "children": "waiting_for_children",
}
"""Interrupt kind -> run waiting status. ``children`` parks the parent
WITHOUT holding an execution slot while its child research runs execute;
the store wakes it when the last child terminates (no human involved)."""

class AgentPhaseState(TypedDict, total=False):
    """Checkpointed graph state — JSON-serializable values only."""

    question: str
    history: str
    autonomy: str
    session_id: str
    skill_point_answers: dict[str, dict[str, str]]
    """Per attached skill: point-name -> answer (from context or the
    clarify round); feeds the {{name}} substitution."""
    skill_questions: list[dict[str, Any]]
    """Pending REQUIRED skill points for the clarify gate; emptied
    after the round (never re-asked)."""
    artifact_registry: list[dict[str, Any]]
    """Meta of ALL session deliverables (K2 multi-deliverable index),
    filled at intake from the session-context builder."""
    last_response_form: str
    """Output form of the latest completed session turn (K3 routing
    memory); '' when unknown."""
    deliverable: str
    """The turn's output form: ``chat`` writes the
    run-local inline answer artifact, ``canvas`` the session memo.
    Decided ONCE at intake (request override > intake profile > canvas;
    a patch assignment forces canvas)."""
    phase: str
    profile: dict[str, Any] | None
    route: str
    probe_plan: list[dict[str, Any]]
    probe_stats: dict[str, Any]
    discovery_tool_use_counts: dict[str, int]
    """Successful source probes run before the persisted mission plan."""
    memory_briefing: str
    prior_evidence_count: int
    """Canonical count of evidence already attached to session artifacts."""
    memory_status: str
    memory_candidates: list[dict[str, Any]]
    discovery: dict[str, Any] | None
    clarification_rounds: int
    assumptions_note: str
    plan_version: int
    plan_id: str
    planned_rounds: int
    needs_plan_interrupt: bool
    replan_noop: bool
    """The latest model delta deliberately added/skipped no work; after its
    auditable plan version is stored, route directly to synthesis."""
    web_research_consent: bool
    """Whether a user plan edit explicitly selected at least one
    ``web_research`` task. Sticky for the run and reconstructed from the
    persisted plan lineage when an older checkpoint lacks the field."""
    web_research_consent_checked_version: int
    """Newest persisted plan version inspected while reconstructing consent."""
    success_criteria: list[str]
    report_guidance: str
    """Decision-scoped user guidance for the report (structure, focus,
    audience) attached at the plan gate; rendered into the outline,
    section and chat-answer prompts."""
    clarified_context: str
    """Compact answered-clarification text for the deterministic
    discovery prober (P2). MUST be declared here: LangGraph drops
    undeclared keys between nodes."""
    plan_rejected: bool
    """A human gate (plan, replan or discovery approval) was rejected.
    NOT a failure: the run completes cleanly with a deterministic
    receipt as its answer (or keeps an already-synthesized draft), and
    the result payload derives ``plan_decision="rejected"`` from it."""
    outcomes: dict[str, dict[str, Any]]
    pending_children: dict[str, dict[str, Any]]
    """Submitted-but-unfolded child research runs, keyed by task id:
    ``{"child_run_id": str, "attempt": int}``. Non-empty exactly while
    the run is (heading towards) ``waiting_for_children`` — the
    children_wait node folds them into ``outcomes`` on resume."""
    legacy_budget_notice_task_ids: list[str]
    """Legacy-budget tasks already narrated in this checkpointed run."""
    references: list[dict[str, Any]]
    web_search_ledger: dict[str, Any]
    """Provider search answers and citation lineage retained for the Canvas."""
    claims: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]
    tool_use_counts: dict[str, int]
    """Cumulative successful source-tool calls for the execution overview."""
    sufficiency: dict[str, Any] | None
    replan_rounds: int
    memo_markdown: str
    memo_title: str
    artifact_id: str
    prior_memo: str
    memo_base_revision: int
    memo_user_prefix: str
    critic: dict[str, Any] | None
    critic_recheck_pending: bool
    revisions_used: int
    target_document_id: str
    patch_id: str
    patch_decision: str
    usage: dict[str, int]
    cancelled: bool
    failure: str


class WorkspaceAgentAlgorithm:
    """Workspace agent over the platform seams.

    Args:
        control_store: Plans, approvals, clarifications, and artifacts.
        run_service: Child research runs + the run store handle.
        resolver: Builds child ``ResolvedAgentContext``s (report
            profiles per task according to the task contract).
        capability_registry: Read-only tool surface; ``None``
            degrades discovery/instant tools loudly per probe.
        checkpointer: Postgres checkpointer or the volatile development
            escape.
        platform: Server-side agent limits (never prompted).
    """

    id = "workspace_agent"
    display_name = "Workspace-Agent"

    def __init__(
        self,
        *,
        control_store: "AgentControlStore",
        run_service: "RunService",
        resolver: "AgentContextResolver",
        capability_registry: "CapabilityRegistry | None",
        checkpointer: "CheckpointerHandle",
        platform: "AgentPlatformSettings",
        permission_service: Any = None,
        knowledge_service: Any = None,
        editor_patch_service: Any = None,
        editor_persistence_service: Any = None,
        agent_memory_service: Any = None,
        skill_service: Any = None,
    ) -> None:
        self._control = control_store
        self._run_service = run_service
        self._resolver = resolver
        self._capabilities = capability_registry
        self._checkpointer = checkpointer
        self._platform = platform
        self._permissions = permission_service
        self._knowledge = knowledge_service
        self._editor_patches = editor_patch_service
        self._editor_docs = editor_persistence_service
        self._memory = agent_memory_service
        self._skills = skill_service
        self._graph: Any = None
        self._graph_lock = threading.Lock()

    def capabilities(self) -> dict[str, Any]:
        """Registry manifest entry (no research-graph streaming)."""
        return {
            "requires": ["llm"],
            "streams_events": True,
            "supports_chat_completions": False,
            "terminal_node": "agent_answer",
            "produces": ["markdown", "artifacts", "plan"],
            "interrupts": ["approval", "clarification"],
        }

    # -- execution --------------------------------------------------------- #

    def run(
        self,
        request: "RunRequest",
        *,
        runtime: "RuntimeContext",
        context: "RunContext",
    ) -> "AgentResult":
        """Execute or RESUME one agent run (segment-aware, §4)."""
        from inqtrix.core.results import AgentResult

        if context.run_id is None or context.park is None:
            raise RuntimeError(
                "workspace_agent laeuft nur ueber /v1/runs (Park-Faehigkeit "
                "und run_id erforderlich)."
            )
        run_id = context.run_id
        graph = self._compiled_graph()
        config = {"configurable": {"thread_id": run_id}}
        deps = _RunDeps(self, request, runtime, context)

        existing = graph.get_state(config)
        # Usage at segment START: execute_run_request books raw.usage
        # after EVERY segment, so each return must carry only this
        # segment's DELTA — re-booking the checkpointed cumulative total
        # would double-count every earlier segment against the quota.
        start_usage = dict(
            (existing.values.get("usage") or {})
            if existing is not None and existing.values
            else {}
        )
        if existing is None or not existing.values:
            ensure_checkpoint_restart_safe(
                run_id,
                control=self._control,
                run_store=self._run_service.run_store,
                run_async=_run_async,
            )
            graph_input: Any = {
                "question": request.question,
                "history": request.history,
                "autonomy": request.autonomy
                or self._platform.default_autonomy,
                "session_id": request.session_id,
                "target_document_id": request.document_id,
                "phase": "intake",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "clarification_rounds": 0,
                "replan_rounds": 0,
                "revisions_used": 0,
                "outcomes": {},
                "references": [],
                "web_search_ledger": {
                    "schema_version": 1,
                    "kind": "web_search_ledger",
                    "searches": {},
                },
                "claims": [],
                "contradictions": [],
                "tool_use_counts": {"web": 0, "knowledge": 0},
                "discovery_tool_use_counts": {"web": 0, "knowledge": 0},
                "cancelled": False,
            }
        elif existing.tasks and any(
            task.interrupts for task in existing.tasks
        ):
            from langgraph.types import Command

            resume_value = self._decision_for_interrupt(run_id, existing)
            graph_input = Command(resume=resume_value)
        else:
            # Crash between segments without a pending interrupt: the
            # checkpoint fast-forwards past completed nodes on its own.
            graph_input = None

        _DEPS.set(deps)
        try:
            try:
                with observe_provider_retries(
                    deps.llm,
                    lambda notice: deps.emit_model_retry(
                        "agent_model", notice
                    ),
                ), observe_structured_retries(deps.emit_model_retry):
                    result_state = graph.invoke(graph_input, config)
            except AgentCancelled as exc:
                usage_total = dict(getattr(exc, "usage", {}) or start_usage)
                _run_async(settle_cancelled_plan_tasks(self._control, run_id))
                cancel_reason = (
                    "token_budget_exceeded"
                    if isinstance(exc, AgentTokenBudgetExceeded)
                    else "client_requested_cancel"
                )
                self._checkpointer.delete_thread(run_id)
                return AgentResult(
                    answer="",
                    result_type="workspace_agent_cancelled",
                    raw={
                        "answer": "",
                        "usage": _usage_delta(usage_total, start_usage),
                        "result_state": {
                            "cancelled": True,
                            "cancel_reason": cancel_reason,
                            "usage_total": usage_total,
                            "execution": _workspace_execution_payload(
                                deps, {"usage": usage_total}
                            ),
                        },
                    },
                )
        finally:
            _DEPS.set(None)

        interrupts = result_state.get("__interrupt__") or ()
        if interrupts:
            payload = interrupts[0].value if interrupts else {}
            waiting_status = _WAITING_STATUS_BY_KIND.get(
                payload.get("kind", ""), "waiting_for_approval"
            )
            context.park(waiting_status)
            return AgentResult(
                answer="",
                result_type="workspace_agent_parked",
                raw={
                    "answer": "",
                    "usage": _usage_delta(
                        result_state.get("usage", {}), start_usage
                    ),
                    "result_state": {
                        "cancelled": False,
                        "parked": True,
                        "phase": result_state.get("phase", ""),
                        "usage_total": dict(result_state.get("usage", {})),
                        "execution": _workspace_execution_payload(
                            deps, result_state
                        ),
                    },
                },
            )

        self._checkpointer.delete_thread(run_id)
        usage_total = dict(result_state.get("usage", {}))
        usage = _usage_delta(usage_total, start_usage)
        cancelled = bool(result_state.get("cancelled"))
        answer = result_state.get("memo_markdown", "") or ""
        failure = result_state.get("failure", "")
        # Patch-phase failures are HARD even though a memo exists: the
        # user explicitly targeted a document — completing quietly would
        # hide that the core of the assignment did not happen.
        hard_patch_failure = failure in (
            "patch_document_not_found",
            "editor_patches_unavailable",
            "patch_proposal_failed",
            "patch_document_too_large",
        )
        if failure and not cancelled and (not answer or hard_patch_failure):
            raise RuntimeError(failure)
        return AgentResult(
            answer=answer,
            result_type="workspace_agent_result",
            raw={
                "answer": answer,
                "usage": usage,
                "result_state": {
                    "cancelled": cancelled,
                    "answer": answer,
                    "usage_total": usage_total,
                    "phase": result_state.get("phase", "done"),
                    "plan_decision": (
                        "rejected"
                        if result_state.get("plan_rejected")
                        else ""
                    ),
                    "plan_version": result_state.get("plan_version", 0),
                    "artifact_id": result_state.get("artifact_id", ""),
                    "critic_report": result_state.get("critic"),
                    "references": result_state.get("references", []),
                    "web_search_ledger": result_state.get(
                        "web_search_ledger", {}
                    ),
                    "report_references": result_state.get(
                        "references", []
                    ),
                    "answer_claim_bindings": result_state.get(
                        "answer_claim_bindings", []
                    ),
                    "contradictions": result_state.get(
                        "contradictions", []
                    ),
                    "replan_rounds": result_state.get("replan_rounds", 0),
                    "revisions_used": result_state.get(
                        "revisions_used", 0
                    ),
                    "patch_id": result_state.get("patch_id", ""),
                    "patch_decision": result_state.get(
                        "patch_decision", ""
                    ),
                    "memory_status": result_state.get("memory_status", ""),
                    "memory_candidates": result_state.get(
                        "memory_candidates", []
                    ),
                    "execution": _workspace_execution_payload(
                        deps, result_state
                    ),
                },
            },
        )

    def _decision_for_interrupt(self, run_id: str, snapshot: Any) -> dict[str, Any]:
        """Resolve the resume payload from the control store (rule R5)."""
        payload: dict[str, Any] = {}
        for task in snapshot.tasks:
            for intr in task.interrupts:
                payload = dict(intr.value or {})
                break
        kind = payload.get("kind", "approval")
        if kind == "children":
            # No control-store row backs a children wait: the child RUN
            # rows are the truth (rule R5) — the children_wait node
            # reads their terminal outcomes from the run store itself.
            return {"kind": "children"}
        if kind == "clarification":
            record = _run_async(
                self._control.get_clarification(
                    run_id, payload.get("id", "")
                )
            )
            return {
                "kind": "clarification",
                "status": record.status,
                "answer": record.answer,
                "option_id": record.option_id,
                "answers": dict(record.answers),
            }
        record = _run_async(
            self._control.get_approval(run_id, payload.get("id", ""))
        )
        return {
            "kind": "approval",
            "status": record.status,
            "decision": record.decision,
            "note": record.note,
            "report_guidance": str(
                dict(record.decision_payload).get("report_guidance", "")
            ),
        }

    # -- graph ------------------------------------------------------------- #

    def _compiled_graph(self) -> Any:
        with self._graph_lock:
            if self._graph is None:
                self._graph = _build_graph(self._checkpointer.saver())
            return self._graph


# The per-run dependency bundle travels via a context variable: LangGraph
# node functions receive only the state, and the compiled graph is shared
# across runs (thread_id separates them) — a threading-local would break
# if langgraph ever re-schedules nodes, contextvars follow the execution.
import contextvars

_DEPS: contextvars.ContextVar["_RunDeps | None"] = contextvars.ContextVar(
    "inqtrix_agent_deps", default=None
)


def _deps() -> "_RunDeps":
    deps = _DEPS.get()
    if deps is None:  # pragma: no cover - graph never runs undepped
        raise RuntimeError("agent graph invoked without run dependencies")
    return deps


class _RunDeps:
    """Everything one segment execution needs (never checkpointed)."""

    def __init__(
        self,
        algorithm: WorkspaceAgentAlgorithm,
        request: "RunRequest",
        runtime: "RuntimeContext",
        context: "RunContext",
    ) -> None:
        self.algorithm = algorithm
        self.request = request
        self.runtime = runtime
        self.context = context
        self.control = algorithm._control
        self.run_service = algorithm._run_service
        self.resolver = algorithm._resolver
        self.capabilities = algorithm._capabilities
        self.source_policy = effective_source_policy(
            request.source_policy, request.execution_directive
        )
        self.platform = algorithm._platform
        self.llm = context.providers.llm
        self.knowledge = algorithm._knowledge
        self.editor_patches = algorithm._editor_patches
        self.editor_docs = algorithm._editor_docs
        self.memory = algorithm._memory
        # Per-user visibility, resolved ONCE per segment: the agent's
        # tool calls must see exactly what its owner sees. By contract,
        # visible_to=None is the
        # see-everything view of the anonymous/static modes only).
        self.visible_to = None
        if (
            algorithm._permissions is not None
            and context.principal is not None
        ):
            self.visible_to = _run_async(
                algorithm._permissions.resolve_user_context(
                    context.principal
                )
            )
        self.knowledge_collection_ids = pinned_knowledge_collection_ids(
            request.knowledge_filters,
            scoped_principal=bool(
                context.principal is not None
                and context.principal.user_id is not None
            ),
        )
        # Long-term agent memory is opt-in per user (privacy default OFF),
        # resolved ONCE per segment like visible_to. The read runs on the
        # account-preferences NullPool store (loop-agnostic), safe from this
        # sync worker thread; opt_in_enabled degrades to False visibly on any
        # read error, so a failure never silently enables memory.
        self.agent_memory_opt_in = False
        if algorithm._memory is not None and context.principal is not None:
            self.agent_memory_opt_in = _run_async(
                algorithm._memory.opt_in_enabled(context.principal)
            )
        self._collection_scope: list[str] | None = None
        settings = context.agent_settings
        self.timeout = float(
            getattr(settings, "reasoning_timeout", REASONING_TIMEOUT)
        )
        self.depth = settings.depth
        # The selected Stufe ('' = legacy depth semantics). All budget/
        # gate consequences resolve through tier_policy.TIER_POLICIES.
        self.tier: str = getattr(settings, "agent_tier", "") or ""
        self.tier_policy = (
            resolve_tier_policy(self.tier) if self.tier else None
        )
        # Router-admitted skills, loaded ONCE per segment from durable
        # rows (rows are truth) — BEFORE the node resolution so skill
        # model pins (R4) can inform it. A skill deleted between
        # admission and this segment fails the run LOUDLY — the user
        # attached it explicitly, running without it would be a silent
        # behavior change (Designprinzip 1).
        self.skills: list[Any] = []
        if request.skill_ids:
            if algorithm._skills is None:
                raise RuntimeError(
                    "Skills sind angehaengt, aber auf diesem Server "
                    "nicht eingerichtet."
                )
            for skill_id in request.skill_ids:
                try:
                    record, _access = _run_async(
                        algorithm._skills.get_visible(
                            skill_id,
                            tenant_id=(
                                context.principal.tenant_id
                                if context.principal is not None
                                else "default"
                            ),
                            visible_to=self.visible_to,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 — loud, never partial
                    raise RuntimeError(
                        f"Angehaengter Skill {skill_id} ist nicht mehr "
                        "verfuegbar — Lauf abgebrochen."
                    ) from exc
                expected_revision = request.skill_revisions.get(skill_id)
                if (
                    expected_revision is None
                    or record.revision != expected_revision
                ):
                    raise RuntimeError(
                        f"Angehaengter Skill {skill_id} hat sich seit der "
                        "Freigabe geaendert — Lauf abgebrochen."
                    )
                self.skills.append(record)
        self.resolutions = self._resolve_nodes(settings)

    def _resolve_nodes(self, settings: Any) -> dict[str, dict[str, str]]:
        provider_models = getattr(self.llm, "models", None)
        requested_tier = (getattr(settings, "model_tier", "") or "").strip()
        requested_model = (getattr(settings, "model", "") or "").strip()
        requested_effort = (getattr(settings, "effort", "") or "").strip()
        pin_tier, pin_effort = skill_model_pins(self.skills)
        if (pin_tier and not requested_tier) or (
            pin_effort and not requested_effort
        ):
            # R4 precedence: explicit user override > skill pin > tier
            # map. The pin only fills what the request left empty.
            log.info(
                "Skill-Pin aktiv: model_tier=%r effort=%r",
                pin_tier or "(kein)",
                pin_effort or "(kein)",
            )
        requested_tier = requested_tier or pin_tier
        requested_effort = requested_effort or pin_effort
        if requested_tier or requested_model or requested_effort:
            log.info(
                "Modell-Override aktiv (tier=%r model=%r effort=%r) — "
                "wirkt auf %s; Fliessband-Nodes bleiben auf der Tier-Map.",
                requested_tier or "(kein)",
                requested_model or "(kein)",
                requested_effort or "(kein)",
                ", ".join(sorted(_OVERRIDE_SCOPED_NODES)),
            )
        resolutions: dict[str, dict[str, str]] = {}
        for node in _AGENT_NODES:
            scoped = node in _OVERRIDE_SCOPED_NODES
            node_tier = requested_tier if scoped else ""
            node_model = requested_model if scoped else ""
            node_effort = requested_effort if scoped else ""
            if provider_models is None and not node_model:
                desc = describe_unresolved_resolution(node, node_tier)
            else:
                desc = describe_resolution(
                    node,
                    provider_models,
                    node_tier,
                    requested_model=node_model,
                    requested_effort=node_effort,
                )
            resolutions[node] = desc
            self.emit("inqtrix.node.model_resolution", desc)
        return resolutions

    def resolved(self, node: str) -> tuple[str | None, str | None]:
        desc = self.resolutions[node]
        return (desc.get("model") or None, desc.get("effort") or None)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.context.event_sink is not None:
            self.context.event_sink(event_type, payload)

    def emit_model_retry(
        self,
        node: str,
        notice: dict[str, Any],
    ) -> None:
        """Emit one provider retry through the existing Agent activity stream."""
        self.emit(ACTIVITY_EVENT, model_retry_activity(notice, node=node))

    def cancelled(self) -> bool:
        token = self.context.cancel_token
        return bool(token is not None and token.is_set())

    def collection_scope(self) -> list[str]:
        """The admitted collection ids this run may retrieve from.

        A concrete persisted list, including an empty list, is immutable for
        the run. Only deliberately unscoped anonymous/static execution keeps
        the historical live-all view. Current visibility is still checked so
        a revoke fails closed instead of silently shrinking the corpus.
        """
        if self._collection_scope is not None:
            return self._collection_scope
        if self.source_policy.knowledge != "available":
            self._collection_scope = []
            return self._collection_scope
        if self.knowledge_collection_ids is not None:
            requested = sorted(self.knowledge_collection_ids)
            self._collection_scope = (
                self.assert_collections(requested) if requested else []
            )
            return self._collection_scope
        if self.knowledge is None:
            self._collection_scope = []
            return self._collection_scope
        collections = _run_async(
            self.knowledge.list_collections(visible_to=self.visible_to)
        )
        self._collection_scope = [
            str(getattr(item, "id", None) or item.get("id"))
            if not hasattr(item, "id")
            else str(item.id)
            for item in collections
        ]
        return self._collection_scope

    def assert_collections(self, collection_ids: list[str]) -> list[str]:
        """Require both run admission and current actor visibility."""
        from inqtrix.knowledge.stores.ports import CollectionNotFound

        requested = set(collection_ids)
        if (
            self.knowledge_collection_ids is not None
            and not requested.issubset(self.knowledge_collection_ids)
        ):
            raise CollectionNotFound(
                "Collection is outside the admitted run scope."
            )
        if self.knowledge is None:
            raise RuntimeError(
                "Collection-Zugriff ohne Wissens-Dienst nicht pruefbar."
            )
        _run_async(
            self.knowledge.assert_collections_visible(
                collection_ids, visible_to=self.visible_to
            )
        )
        return list(collection_ids)


def _usage_delta(
    total: dict[str, Any], start: dict[str, Any]
) -> dict[str, int]:
    """This segment's token spend (cumulative total minus segment start)."""
    return {
        "prompt_tokens": max(
            0,
            int(total.get("prompt_tokens", 0) or 0)
            - int(start.get("prompt_tokens", 0) or 0),
        ),
        "completion_tokens": max(
            0,
            int(total.get("completion_tokens", 0) or 0)
            - int(start.get("completion_tokens", 0) or 0),
        ),
    }


def _workspace_execution_payload(
    deps: "_RunDeps", state: AgentPhaseState
) -> dict[str, object]:
    """Canonical Agent Desk execution block for the mission algorithm."""
    from inqtrix.agents.source_policy import execution_payload

    response_form = str(
        state.get("deliverable") or deps.request.response_form or "auto"
    )
    resolution_node = (
        "agent_answer" if response_form == "chat" else "agent_synthesis"
    )
    model, effort = deps.resolved(resolution_node)
    autonomy = str(
        state.get("autonomy")
        or deps.request.autonomy
        or deps.platform.default_autonomy
    )
    limits: dict[str, object] = {
        "discovery_tool_calls": {
            "used": sum(
                int(value or 0)
                for value in dict(
                    state.get("discovery_tool_use_counts") or {}
                ).values()
            ),
            "limit": deps.platform.discovery_max_tool_calls,
            "ceiling": deps.platform.discovery_max_tool_calls,
            "recoverable": False,
            "extendable": False,
            "reason": "deterministic_discovery_budget",
        },
        "plan_tasks": {
            "limit": deps.platform.max_plan_tasks,
            "ceiling": deps.platform.max_plan_tasks,
            "recoverable": False,
            "extendable": False,
            "reason": "validated_plan_shape",
        },
        "replan_rounds": {
            "used": int(state.get("replan_rounds", 0) or 0),
            "limit": deps.platform.max_replan_rounds,
            "ceiling": deps.platform.max_replan_rounds,
            "recoverable": False,
            "extendable": False,
            "reason": "deterministic_mission_budget",
        },
        "clarification_rounds": {
            "used": int(state.get("clarification_rounds", 0) or 0),
            "limit": deps.platform.max_clarification_rounds,
            "ceiling": deps.platform.max_clarification_rounds,
            "recoverable": False,
            "extendable": False,
            "reason": "deterministic_mission_budget",
        },
    }
    token_budget = int(deps.context.token_budget or 0)
    if token_budget > 0:
        usage = dict(state.get("usage") or {})
        limits["tokens"] = {
            "used": sum(int(value or 0) for value in usage.values()),
            "limit": token_budget,
            "ceiling": token_budget,
            "recoverable": False,
            "extendable": False,
            "reason": "operator_ceiling_exactly_once_required",
        }
    return execution_payload(
        execution_directive=deps.request.execution_directive,
        effective_mode="workspace_agent",
        response_form=response_form,
        depth=deps.depth,
        model=model,
        reasoning_effort=effort,
        source_policy=deps.source_policy,
        consent_reason=(
            "autonomous_policy"
            if autonomy == "autonomous"
            else "permission_policy"
        ),
        tool_use_counts=dict(state.get("tool_use_counts") or {}),
        limits=limits,
    )


def _run_async(coro: Any) -> Any:
    """Drive one control-store coroutine from the worker thread."""
    return asyncio.run(coro)


def _emit_narration(
    deps: "_RunDeps",
    *,
    narration_id: str,
    kind: str,
    text: str,
    phase: str,
) -> None:
    """One transcript prose line (plan B2), silently skipping empties.

    ``narration_id`` is DETERMINISTIC per emission site (plan version,
    task id, section index ...), never a running counter — checkpointed
    nodes may re-execute after an interrupt, and a stable id keeps the
    replayed line recognizable instead of multiplying.
    """
    if not text.strip():
        return
    deps.emit(
        NARRATION_EVENT,
        {
            "narration_id": narration_id,
            "kind": kind,
            "text": text,
            "phase": phase,
            "final": True,
        },
    )


def _add_usage(state: AgentPhaseState, delta: dict[str, int]) -> None:
    usage = dict(state.get("usage", {}))
    usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + int(
        delta.get("prompt_tokens", 0) or 0
    )
    usage["completion_tokens"] = usage.get("completion_tokens", 0) + int(
        delta.get("completion_tokens", 0) or 0
    )
    state["usage"] = usage
    deps = _DEPS.get()
    if deps is None:
        # Pure helper tests call phase-finalization functions without a
        # running graph. Production graph execution always installs deps.
        return
    budget = int(deps.context.token_budget or 0)
    used = usage["prompt_tokens"] + usage["completion_tokens"]
    if budget and used >= budget:
        log.warning(
            "Workspace-Agent wegen serverseitigem Tokenbudget gestoppt: "
            "%d/%d Tokens.",
            used,
            budget,
        )
        deps.emit(
            LIMIT_REACHED_EVENT,
            {
                "kind": "tokens",
                "used": used,
                "limit": budget,
                "ceiling": budget,
                "extendable": False,
                "recoverable": False,
                "reason": "operator_ceiling_exactly_once_required",
                "state": "failed_at_node_boundary",
            },
        )
        _emit_narration(
            deps,
            narration_id="n-limit-tokens",
            kind="limit",
            text=(
                "Das feste serverseitige Tokenlimit ist erreicht. Der Lauf "
                "wurde ohne automatische Teilantwort beendet; eine "
                "Erweiterung ist für diesen Lauf nicht möglich."
            ),
            phase=str(state.get("phase", "execution") or "execution"),
        )
        deps.emit(
            ACTIVITY_EVENT,
            {
                "scope": "run",
                "phase": str(state.get("phase", "")),
                "operation": "run.token_budget",
                "detail": "Serverseitiges Tokenbudget erreicht",
                "status": "failed",
                "error": {
                    "code": "token_budget_exceeded",
                    "message": "Serverseitiges Tokenbudget erreicht.",
                },
            },
        )
        raise AgentTokenBudgetExceeded(
            "Lauf wegen Token-Budget (max_tokens_per_run) gestoppt.",
            usage=usage,
        )


def _set_phase(state: AgentPhaseState, phase: str) -> None:
    """THE one phase transition (rule R6): state + event + snapshot."""
    previous = state.get("phase", "")
    state["phase"] = phase
    _deps().emit(
        PHASE_CHANGED_EVENT,
        {
            "phase": phase,
            "previous_phase": previous,
            "snapshot": {
                "current_node": phase,
                "phase": phase,
                "execution": _workspace_execution_payload(_deps(), state),
            },
        },
    )


def _load_memory_briefing(
    deps: "_RunDeps", state: AgentPhaseState
) -> None:
    """Load non-evidentiary long-term memory context for this run."""
    service = deps.memory
    principal = deps.context.principal
    if service is None or principal is None:
        state["memory_status"] = "disabled"
        return
    if not deps.agent_memory_opt_in:
        # Long-term memory is opt-in per user (privacy default OFF): with no
        # opt-in the agent recalls nothing. The choice stays visible via the
        # settings toggle and the "disabled" status (Designprinzip 1).
        state["memory_status"] = "disabled"
        return
    status = service.status(principal)
    if not status.get("available") or not status.get("principal_eligible"):
        state["memory_status"] = "disabled"
        return
    try:
        briefing, memory_status = _run_async(
            service.recall_briefing(
                principal=principal,
                query=state.get("question", ""),
                limit=5,
            )
        )
    except AgentMemoryUnavailable:
        briefing, memory_status = "", "unavailable"
    state["memory_briefing"] = briefing
    state["memory_status"] = memory_status
    label = {
        "used": "Memory verwendet",
        "empty": "Memory geprueft",
        "unavailable": "Memory unavailable",
    }.get(memory_status, "Memory geprueft")
    deps.emit(
        ACTIVITY_EVENT,
        {
            "kind": (
                "memory_unavailable"
                if memory_status == "unavailable"
                else "memory"
            ),
            "label": label,
            "status": memory_status,
            "detail": (
                "Langzeit-Memory liefert Kontext, aber keine Evidenz."
                if memory_status == "used"
                else "Langzeit-Memory ist fuer diesen Lauf nicht nutzbar."
                if memory_status == "unavailable"
                else ""
            ),
        },
    )


# -- nodes ------------------------------------------------------------------ #


def _node_intake(state: AgentPhaseState) -> AgentPhaseState:
    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    _set_phase(state, "intake")
    # A patch assignment validates its TARGET here, before any spend: a
    # typo'd or invisible document must fail in seconds, not after the
    # whole research pipeline ran (and the failure path books nothing).
    document_id = state.get("target_document_id", "")
    if document_id:
        if deps.editor_patches is None or deps.editor_docs is None:
            state["failure"] = "editor_patches_unavailable"
            return state
        from inqtrix.project.editor_ports import DocumentNotFound

        try:
            _run_async(
                deps.editor_docs.get_document(
                    document_id, visible_to=deps.visible_to
                )
            )
        except DocumentNotFound:
            state["failure"] = "patch_document_not_found"
            return state
    # Session-memo lineage (E15): a follow-up turn continues the ONE memo
    # canvas of its session. Read its LATEST revision now — including any
    # edit the user made after the previous turn — so synthesis extends it
    # and the write CAS-guards against that exact revision (never clobbers
    # the user's text, rule R10).
    session_id = state.get("session_id", "")
    if session_id:
        prior = _run_async(
            deps.control.get_session_artifact(session_id, "memo")
        )
        if prior is not None:
            state["artifact_id"] = prior.artifact_id
            state["memo_base_revision"] = prior.revision
            state["prior_memo"] = prior.content_markdown
            state["memo_title"] = prior.title
    # Session metadata is always reconstructed. Explicit request history may
    # replace K1, but it must not erase the K2-K4 continuity contract.
    if session_id:
        pack = build_session_context(
            session_id,
            run_store=deps.run_service.run_store,
            control=deps.control,
            run_async=_run_async,
            visible_to=deps.visible_to,
            current_run_id=deps.context.run_id or "",
        )
        if pack.history_block and not str(state.get("history", "")).strip():
            state["history"] = pack.history_block
        if pack.artifact_registry:
            state["artifact_registry"] = [
                dict(item) for item in pack.artifact_registry
            ]
        if pack.last_response_form:
            state["last_response_form"] = pack.last_response_form
        state["prior_evidence_count"] = pack.prior_evidence_count
    _load_memory_briefing(deps, state)
    model, effort = deps.resolved("agent_intake")
    outcome = run_intake(
        deps.llm,
        question=state["question"],
        history=state.get("history", ""),
        model=model,
        reasoning_effort=effort,
        timeout=deps.timeout,
        skills_block=_skills_prompt_block(deps, state),
        artifact_registry=list(state.get("artifact_registry") or []),
        last_response_form=str(state.get("last_response_form") or ""),
        prior_evidence_count=int(state.get("prior_evidence_count") or 0),
    )
    _add_usage(state, outcome.usage)
    profile = outcome.value
    state["profile"] = (
        profile.model_dump() if isinstance(profile, AssignmentProfile) else None
    )
    state["success_criteria"] = (
        list(profile.success_criteria)
        if isinstance(profile, AssignmentProfile)
        else []
    )
    state["route"] = route_readiness(
        profile if isinstance(profile, AssignmentProfile) else None
    )
    # Deliverable decision (M1 S3), ONCE per run: an explicit request
    # override wins, then a skill's pinned deliverable, then the intake
    # profile's inference; a failed intake or a patch assignment stays
    # on the established canvas path (safe default = today's behavior).
    skill_deliverable = next(
        (
            "chat" if skill.deliverable == "chat" else "canvas"
            for skill in deps.skills
            if skill.deliverable
        ),
        "",
    )
    requested_form = (deps.request.response_form or "").strip()
    tier_form = (
        deps.tier_policy.response_form
        if deps.tier_policy is not None
        and deps.tier_policy.response_form != "auto"
        else ""
    )
    if document_id:
        state["deliverable"] = "canvas"
    elif requested_form in ("chat", "canvas"):
        state["deliverable"] = requested_form
    elif skill_deliverable:
        state["deliverable"] = skill_deliverable
    elif tier_form:
        # Tier default (schnell -> chat, tief -> canvas): weaker than an
        # explicit request/skill pin, stronger than intake inference.
        state["deliverable"] = tier_form
    elif (
        isinstance(profile, AssignmentProfile)
        and profile.response_form == "chat"
    ):
        state["deliverable"] = "chat"
    else:
        state["deliverable"] = "canvas"
    # Skill point check, once per run: what
    # question+history already answer never gets asked again; missing
    # REQUIRED points queue structured questions for the clarify gate.
    if deps.skills and "skill_point_answers" not in state:
        point_answers: dict[str, dict[str, str]] = {}
        skill_questions: list[dict[str, Any]] = []
        check_model, check_effort = deps.resolved("agent_skill_point_check")
        for skill in deps.skills:
            answers, usage = check_skill_points(
                deps.llm,
                skill=skill,
                question=state["question"],
                history=state.get("history", ""),
                model=check_model,
                reasoning_effort=check_effort,
                timeout=deps.timeout,
            )
            _add_usage(state, usage)
            point_answers[skill.id] = answers
            for point in unanswered_required_points(skill, answers):
                skill_questions.append(
                    {
                        "skill_id": skill.id,
                        "key": skill_point_key(point),
                        "name": str(point.get("name", "") or ""),
                        "prompt": str(point.get("question", "")),
                        "options": [
                            {
                                "label": option.get("label", ""),
                                "description": option.get(
                                    "description", ""
                                ),
                            }
                            for option in point.get("options", [])
                        ],
                    }
                )
        state["skill_point_answers"] = point_answers
        state["skill_questions"] = skill_questions
        if skill_questions and state.get("route") != "ask_user_first":
            # Missing required inputs block exactly like an ambiguous
            # assignment: ask BEFORE any probe runs.
            state["route"] = "ask_user_first"
    return state


def _node_clarify(state: AgentPhaseState) -> AgentPhaseState:
    """Ask the user (intake route or blocking gaps) — ONE interrupt."""
    from langgraph.types import interrupt

    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    rounds = state.get("clarification_rounds", 0)
    max_rounds = deps.platform.max_clarification_rounds
    if deps.tier_policy is not None:
        # The tier caps rounds tighter, never wider (permission-style
        # min): schnell asks nothing, gruendlich at most once.
        max_rounds = min(max_rounds, deps.tier_policy.clarification_rounds)
    if rounds >= max_rounds:
        state["assumptions_note"] = (
            "Maximale Rueckfrage-Runden erreicht; beste Annahme verwendet."
        )
        deps.emit(
            LIMIT_REACHED_EVENT,
            {
                "kind": "clarification_rounds",
                "used": rounds,
                "limit": max_rounds,
                "ceiling": max_rounds,
                "extendable": False,
                "recoverable": False,
                "reason": "deterministic_mission_budget",
                "state": "continued_with_visible_assumption",
            },
        )
        _emit_narration(
            deps,
            narration_id="n-clarification-limit",
            kind="limit",
            text=(
                "Das Rückfrage-Limit ist erreicht. Der Lauf arbeitet mit "
                "der im Ergebnis ausdrücklich ausgewiesenen besten Annahme weiter."
            ),
            phase="clarification",
        )
        return state
    profile = _profile_of(state)
    discovery = _discovery_of(state)
    questions = (
        intake_questions(profile)
        if state.get("route") == "ask_user_first" and discovery is None
        else blocking_questions(discovery)
    )
    run_id = deps.context.run_id or ""
    clarification_id = f"clr_{run_id[-12:]}_{rounds}"
    if questions:
        # Deterministic backstop behind the analyst's never-re-ask rule:
        # a round-2 question that rephrases an already-answered round is
        # dropped LOUDLY. The current round's own (possibly persisted)
        # record is excluded, so an interrupt re-execution filters
        # against exactly the same prior rounds as the first pass.
        asked_prompts = _asked_clarification_prompts(
            deps, run_id, exclude_id=clarification_id
        )
        questions, dropped = filter_repeated_questions(
            questions, asked_prompts
        )
        if dropped:
            log.warning(
                "Rueckfragen-Dedup hat bereits beantwortete Eintraege "
                "uebersprungen (run=%s, count=%d).",
                run_id,
                len(dropped),
            )
            _emit_narration(
                deps,
                narration_id=f"n-clarify-skip-{rounds}",
                kind="discovery",
                text=(
                    f"{len(dropped)} bereits beantwortete "
                    "Rueckfrage(n) uebersprungen."
                ),
                phase="clarification",
            )
    # Skill point questions join the same round —
    # appended LAST so their positional q-ids stay recoverable for the
    # answer mapping below (sanitize assigns ids by position). The
    # round cap still applies: overflowing skill questions are cut
    # LOUDLY by the sanitizer and fall back to their visible default
    # assumptions.
    skill_questions = list(state.get("skill_questions") or [])
    if skill_questions:
        questions = sanitize_questions(
            [
                ClarificationQuestionModel(
                    prompt=str(item["prompt"]),
                    options=[
                        ClarificationOptionModel(
                            label=str(option.get("label", "")),
                            description=str(option.get("description", "")),
                        )
                        for option in item.get("options", [])
                    ],
                    multi_select=bool(item.get("multi_select")),
                )
                for item in [*questions, *skill_questions]
            ]
        )
    if not questions:
        return state
    _set_phase(state, "clarification")
    try:
        record = _run_async(
            deps.control.get_clarification(run_id, clarification_id)
        )
    except ClarificationNotFound:
        record = _run_async(
            deps.control.create_clarification(
                build_clarification(
                    run_id,
                    questions=questions,
                    clarification_id=clarification_id,
                )
            )
        )
        deps.emit(
            CLARIFICATION_REQUESTED_EVENT,
            {
                "clarification_id": record.clarification_id,
                "question": record.question,
                "options": [dict(option) for option in record.options],
                "question_count": len(record.questions),
            },
        )
    decision = interrupt({"kind": "clarification", "id": clarification_id})
    state["clarification_rounds"] = rounds + 1
    history_block = _clarification_history_block(record, decision)
    if history_block:
        state["history"] = state.get("history", "") + history_block
        # Compact answer text for the DETERMINISTIC discovery prober:
        # the probes run after the answers exist, so their queries must
        # honor what the user just pinned down (market, region, ...).
        # Per Q/A block (blocks are blank-line separated) with newlines
        # collapsed — a multi-line free-text answer keeps ALL its lines.
        answers_text = " ".join(
            " ".join(block.split("Antwort: ", 1)[1].split())
            for block in history_block.split("\n\n")
            if "Antwort: " in block
        ).strip()
        if answers_text:
            state["clarified_context"] = (
                f"{state.get('clarified_context', '')} {answers_text}"
            ).strip()
    if skill_questions:
        # Positional mapping back onto the skill points: the skill
        # questions are the LAST len(skill_questions) of the round, in
        # declaration order. A whole-round free-text answer cannot be
        # attributed per point — it rides the history instead and the
        # points fall back to their visible default assumptions.
        answers_map = decision.get("answers") or {}
        stored_questions = list(record.questions)
        point_answers = dict(state.get("skill_point_answers") or {})
        # Offset from the STORED record, not the recomputed list: an
        # interrupt re-execution may re-filter differently (e.g. the
        # duplicate listing failed on the first pass), and the persisted
        # row is the layout the answer ids actually refer to.
        stored_profile_count = max(
            0, len(stored_questions) - len(skill_questions)
        )
        for index, item in enumerate(skill_questions):
            question_index = stored_profile_count + index
            if question_index >= len(stored_questions):
                break
            stored = stored_questions[question_index]
            entry = answers_map.get(str(stored.get("id", ""))) or {}
            labels = {
                option["id"]: option["label"]
                for option in stored.get("options", [])
            }
            parts = [
                labels[oid]
                for oid in entry.get("option_ids", [])
                if oid in labels
            ]
            text = str(entry.get("text", "") or "").strip()
            if text:
                parts.append(text)
            if not parts:
                continue
            per_skill = dict(point_answers.get(item["skill_id"]) or {})
            # The canonical point key (skill_point_key) — the reader in
            # skill_input_lines resolves the SAME key, so answers to
            # free (nameless) points reach the prompt too.
            per_skill[item.get("key") or item["name"]] = ", ".join(parts)
            point_answers[item["skill_id"]] = per_skill
        state["skill_point_answers"] = point_answers
        # The round is answered — never re-ask the same points.
        state["skill_questions"] = []
    return state


def _asked_clarification_prompts(
    deps: "_RunDeps", run_id: str, *, exclude_id: str
) -> list[str]:
    """Every question prompt already asked in this run's earlier rounds.

    ``exclude_id`` keeps the CURRENT round out of its own duplicate
    check — on interrupt re-execution the round's record already exists
    and would otherwise filter itself empty. A failed listing degrades
    to no deduplication, visibly (the prompt-side rule still applies).
    """
    try:
        records = _run_async(deps.control.list_clarifications(run_id))
    except Exception as exc:  # noqa: BLE001 — dedup is an enhancement
        log.warning(
            "Rueckfragen-Dedup ohne fruehere Runden (Listing schlug "
            "fehl; run=%s, error_type=%s).",
            run_id,
            type(exc).__name__,
        )
        return []
    prompts: list[str] = []
    for record in records:
        if record.clarification_id == exclude_id:
            continue
        questions = list(record.questions)
        for question in questions:
            prompt = str(question.get("prompt", "")).strip()
            if prompt:
                prompts.append(prompt)
        if not questions and record.question.strip():
            prompts.append(record.question.strip())
    return prompts


def _clarification_history_block(
    record: Any, decision: dict[str, Any]
) -> str:
    """The deterministic Q/A transcript block of one answered round.

    Composition lives in :func:`round_qa_lines` (shared with the K1
    session-context builder). An unanswered round contributes nothing.
    """
    lines = round_qa_lines(
        questions=list(record.questions),
        question=record.question,
        options=list(record.options),
        answers=decision.get("answers") or {},
        answer=str(decision.get("answer", "")),
        option_id=str(decision.get("option_id", "")),
    )
    if not lines:
        return ""
    return "\n\n" + "\n\n".join(
        f"Rueckfrage: {prompt}\nAntwort: {answer}" for prompt, answer in lines
    )


def _node_discovery(state: AgentPhaseState) -> AgentPhaseState:
    from langgraph.types import interrupt

    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    _set_phase(state, "discovery")
    profile = _profile_of(state)
    try:
        collection_ids = deps.collection_scope()
    except Exception as exc:  # noqa: BLE001 — invisible ids fail the run loudly
        state["failure"] = f"Collection nicht sichtbar: {exc}"
        return state
    plan = build_probe_plan(
        profile,
        question=state["question"],
        clarified_context=state.get("clarified_context", ""),
        collection_ids=collection_ids,
        max_calls=deps.platform.discovery_max_tool_calls,
        # Standard mode (balanced) keeps ALL web contact behind the plan
        # gate (the tasks carry their verbatim queries — that is the
        # consent surface); only Auto (autonomous) may probe the web
        # during discovery in autonomous mode. Internal probes
        # stay available in every mode.
        web_preview_allowed=(
            deps.platform.allow_web_discovery_preview
            and deps.capabilities is not None
            and state.get("autonomy") == "autonomous"
            and deps.source_policy.web == "available"
        ),
        knowledge_allowed=(
            deps.source_policy.knowledge == "available"
        ),
    )
    state["probe_plan"] = plan.as_payload()
    if plan.omitted_count > 0:
        deps.emit(
            LIMIT_REACHED_EVENT,
            {
                "kind": "discovery_tool_calls",
                "used": len(plan.probes),
                "limit": plan.limit,
                "ceiling": plan.limit,
                "extendable": False,
                "recoverable": False,
                "omitted": plan.omitted_count,
                "reason": "deterministic_discovery_budget",
                "state": "continued_with_visible_omission",
            },
        )
        _emit_narration(
            deps,
            narration_id="n-discovery-limit",
            kind="limit",
            text=(
                f"Das Discovery-Limit von {plan.limit} Aufrufen ist "
                f"erreicht; {plan.omitted_count} geplante Sondierung(en) "
                "wurden sichtbar ausgelassen."
            ),
            phase="discovery",
        )

    if state.get("autonomy") == "strict":
        run_id = deps.context.run_id or ""
        approval_id = f"apr_{run_id[-12:]}_discovery_0"
        _ensure_approval(
            deps,
            run_id,
            approval_id,
            kind="discovery",
            payload={"probes": plan.as_payload()},
        )
        decision = interrupt({"kind": "approval", "id": approval_id})
        if decision.get("decision") == "reject":
            # Same contract as the plan gate: a rejected gate is a clean
            # user decision, not a run failure.
            _finish_with_gate_rejection(
                deps,
                state,
                decision=decision,
                gate_subject="Erkundung",
                narration_id="n-discovery-rejected",
                phase="discovery",
            )
            return state

    if deps.capabilities is None:
        digest, stats = "(keine Capabilities verfuegbar)", {
            "planned": len(plan.probes),
            "executed": 0,
            "failed": len(plan.probes),
            "source_tool_counts": {"web": 0, "knowledge": 0},
        }
    else:
        capability_context = _capability_context(
            deps,
            on_provider_retry=lambda notice: deps.emit(
                ACTIVITY_EVENT,
                provider_retry_activity(
                    notice,
                    purpose="Webvorschau in der Erkundung",
                    scope="discovery",
                    phase="discovery",
                ),
            ),
        )
        operation_totals: dict[str, int] = {}
        for probe in plan.probes:
            operation = str(probe.get("kind", ""))
            operation_totals[operation] = (
                operation_totals.get(operation, 0) + 1
            )
        operation_current: dict[str, int] = {}

        def _invoke_probe(
            capability_id: str, payload: dict[str, Any]
        ) -> Any:
            current = operation_current.get(capability_id, 0) + 1
            operation_current[capability_id] = current
            event = {
                "kind": "searching",
                "scope": "discovery",
                "phase": "discovery",
                "operation": capability_id,
                "detail": str(payload.get("query", "")),
                "status": "started",
                "current": current,
                "total": operation_totals.get(capability_id, 1),
                **(
                    {"query": str(payload.get("query", ""))}
                    if payload.get("query")
                    else {}
                ),
            }
            deps.emit(ACTIVITY_EVENT, event)
            try:
                output = _run_async(
                    deps.capabilities.invoke(
                        capability_id, payload, capability_context
                    )
                )
            except Exception as exc:
                deps.emit(
                    ACTIVITY_EVENT,
                    {
                        **event,
                        "status": "failed",
                        "error": {
                            "code": _task_failure_code(exc),
                            "message": sanitize_error(exc),
                        },
                    },
                )
                raise
            deps.emit(
                ACTIVITY_EVENT,
                {
                    **event,
                    "status": "completed",
                    "metrics": {"result_count": _result_count(output)},
                },
            )
            return output

        digest, stats = execute_probes(
            plan,
            registry=deps.capabilities,
            capability_context=capability_context,
            invoke=_invoke_probe,
        )
    state["probe_stats"] = stats
    state["discovery_tool_use_counts"] = dict(
        stats.get("source_tool_counts") or {"web": 0, "knowledge": 0}
    )
    state["tool_use_counts"] = dict(state["discovery_tool_use_counts"])
    model, effort = deps.resolved("agent_discovery_analyst")
    outcome = run_discovery_analysis(
        deps.llm,
        question=state["question"],
        probe_digest=digest,
        profile=profile,
        model=model,
        reasoning_effort=effort,
        timeout=deps.timeout,
        history=state.get("history", ""),
    )
    _add_usage(state, outcome.usage)
    discovery = outcome.value
    state["discovery"] = (
        discovery.model_dump()
        if isinstance(discovery, DiscoveryResult)
        else None
    )
    _emit_narration(
        deps,
        narration_id="n-discovery",
        kind="discovery",
        text=discovery_narration(
            discovery if isinstance(discovery, DiscoveryResult) else None
        ),
        phase="discovery",
    )
    return state


def _skills_prompt_block(deps: "_RunDeps", state: AgentPhaseState) -> str:
    """The combined skill + tool-directive prompt section of one run."""
    parts: list[str] = []
    directives = build_tool_directives_line(
        deps.request.tool_directives or ()
    )
    if directives:
        parts.append(directives)
    block = build_skills_block(
        deps.skills, dict(state.get("skill_point_answers") or {})
    )
    if block:
        parts.append(block)
    return "\n\n".join(parts)


def _node_plan(state: AgentPhaseState) -> AgentPhaseState:
    """Plan + validate + persist (NO interrupt here: the node's state
    writes must COMMIT — an interrupt in the same node would discard
    them, losing the planner's token usage; the approval lives in
    :func:`_node_plan_approval`)."""
    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    _set_phase(state, "planning")
    run_id = deps.context.run_id or ""
    profile = _profile_of(state)
    discovery = _discovery_of(state)
    replan_round = state.get("replan_rounds", 0)
    autonomy = state.get("autonomy", "balanced")

    # Round marker (NOT the plan version: user edits append versions of
    # their own, which must never make a later replan a silent no-op).
    if state.get("planned_rounds", 0) <= replan_round:
        previous_plan: PlanRecord | None = None
        previous_tasks: list[PlanTaskRecord] = []
        if replan_round > 0:
            previous_plan, previous_tasks = _run_async(
                deps.control.get_plan(run_id)
            )
        edit_consent = _plan_lineage_web_research_consent(
            deps,
            state,
            current_plan=previous_plan,
            current_tasks=previous_tasks,
        )
        model, effort = deps.resolved("agent_plan")
        try:
            plan, usage = run_planner(
                deps.llm,
                question=state["question"],
                discovery=discovery,
                profile=profile,
                max_tasks=deps.platform.max_plan_tasks,
                web_allowed=_web_allowed(deps),
                knowledge_allowed=(
                    deps.source_policy.knowledge == "available"
                ),
                model=model,
                reasoning_effort=effort,
                timeout=deps.timeout,
                replan_context=_build_replan_context(state),
                memory_briefing=state.get("memory_briefing", ""),
                collection_catalog=_collection_catalog(deps),
                skills_block=_skills_prompt_block(deps, state),
                allowed_task_kinds=allowed_task_kinds_for_policy(
                    allowed_task_kinds(deps.skills),
                    policy=deps.source_policy,
                ),
                depth=deps.depth,
                explicit_web_research=(
                    "web_research" in deps.request.tool_directives
                    or edit_consent
                ),
                previous_plan=previous_plan,
                previous_tasks=previous_tasks,
                history=state.get("history", ""),
                tier=deps.tier or None,
            )
        except PlanningFailed as exc:
            state["failure"] = f"plan_invalid: {'; '.join(exc.errors)}"
            return state
        _add_usage(state, usage)
        if plan.success_criteria:
            state["success_criteria"] = list(plan.success_criteria)
        reason = (
            ""
            if replan_round == 0
            else "critic_research"
            if state.get("route") == "critic_research"
            else "replan"
        )
        record, tasks = plan_to_records(
            plan,
            run_id=run_id,
            created_by="agent",
            reason=reason,
            previous_tasks=previous_tasks,
        )
        # requires_plan matrix (strictest wins): `always`
        # forces the gate in EVERY mode and round (the one way a skill
        # author reins in Auto), `never` frees only ROUND 0 — the E16
        # web-replan re-gate is a security consent no skill may lift.
        skill_plan_policy = strictest_requires_plan(deps.skills)
        needs_interrupt = autonomy != "autonomous" and replan_round == 0
        if (
            replan_round == 0
            and deps.tier_policy is not None
            and deps.tier_policy.plan_gate == "skip_unless_strict"
        ):
            # Speed tier: no gate unless the PERMISSION dimension
            # (strict) demands one — permission always beats speed.
            needs_interrupt = autonomy == "strict"
        if replan_round == 0:
            if skill_plan_policy == "always":
                needs_interrupt = True
            elif skill_plan_policy == "never":
                needs_interrupt = False
        if replan_round > 0:
            # The E16 replan policy judges the DELTA: only tasks that
            # were not executed before count as new (deltas are
            # additive, results never discarded).
            previously_executed = set(state.get("outcomes", {}))
            new_tasks = [
                task
                for task in tasks
                if task.task_id not in previously_executed
                and task.tool_kind != "synthesis"
            ]
            previous_source_ids = {
                task.task_id
                for task in previous_tasks
                if task.tool_kind != "synthesis"
            }
            current_source_ids = {
                task.task_id
                for task in tasks
                if task.tool_kind != "synthesis"
            }
            state["replan_noop"] = (
                not new_tasks
                and current_source_ids == previous_source_ids
            )
            needs_interrupt = not autonomy_auto_approves(
                autonomy=autonomy, new_tasks=new_tasks
            )
            if state["replan_noop"]:
                # An explicit no-op delta means "proceed with the known gap".
                # It neither creates new external contact nor benefits from a
                # human gate; route straight to synthesis after persisting the
                # auditable plan version.
                needs_interrupt = False
            if skill_plan_policy == "always":
                needs_interrupt = not state["replan_noop"]
        else:
            state["replan_noop"] = False
        record = replace(
            record, status="proposed" if needs_interrupt else "approved"
        )
        saved = _run_async(
            deps.control.save_plan(run_id=run_id, plan=record, tasks=tasks)
        )
        state["plan_id"] = saved.plan_id
        state["plan_version"] = saved.version
        state["planned_rounds"] = replan_round + 1
        state["needs_plan_interrupt"] = needs_interrupt
        deps.emit(
            PLAN_PROPOSED_EVENT if replan_round == 0 else PLAN_REVISED_EVENT,
            {
                "plan_id": saved.plan_id,
                "version": saved.version,
                "task_count": len(tasks),
                **(
                    {"reason": reason, "auto_approved": not needs_interrupt}
                    if replan_round
                    else {}
                ),
            },
        )
        _emit_narration(
            deps,
            narration_id=f"n-plan-{saved.version}",
            kind="plan",
            text=plan_narration(plan.summary_markdown, len(tasks)),
            phase="planning",
        )
    return state


def _finish_with_gate_rejection(
    deps: "_RunDeps",
    state: AgentPhaseState,
    *,
    decision: dict[str, Any],
    gate_subject: str,
    narration_id: str,
    phase: str,
) -> None:
    """Clean no-research finish for a rejected human gate.

    A gate rejection (plan, replan or discovery approval) is an ordered
    user decision, not a run failure: the run completes with a
    deterministic German receipt (no LLM call) as its chat answer, while
    the approval row keeps the durable ``rejected`` status. A rejected
    REPLAN that already synthesized a draft (critic verdict "research")
    keeps that draft as the deliverable instead — the pre-existing
    completed-with-draft behavior.

    Args:
        decision: Resume payload of the approval interrupt; ``note``
            carries the decider's free text and is echoed as a Markdown
            blockquote (newlines re-quoted so a multi-line note cannot
            escape the quote).
        gate_subject: German subject for the receipt ("Plan",
            "Erkundung") — kept nominative so the same word works in
            title, receipt and narration.
        narration_id: Deterministic id in the established ``n-*`` scheme.
        phase: Station vocabulary value ("planning", "discovery").
    """
    state["plan_rejected"] = True
    if state.get("memo_markdown"):
        _emit_narration(
            deps,
            narration_id=narration_id,
            kind="plan",
            text=(
                f"{gate_subject} abgelehnt; der bisherige Entwurf "
                "bleibt das Ergebnis."
            ),
            phase=phase,
        )
        return
    note = str(decision.get("note", "") or "").strip()
    receipt = (
        f"{gate_subject} abgelehnt. Es wurde keine Recherche "
        "ausgefuehrt."
    )
    if note:
        receipt += "\n\n> Notiz: " + note.replace("\n", "\n> ")
    receipt += (
        "\n\nPasse den Auftrag an und sende ihn erneut, um einen "
        "neuen Plan zu erhalten."
    )
    state["deliverable"] = "chat"
    state["memo_markdown"] = receipt
    state["memo_title"] = f"{gate_subject} abgelehnt"
    _emit_narration(
        deps,
        narration_id=narration_id,
        kind="plan",
        text=(
            f"{gate_subject} abgelehnt; der Lauf endet ohne Recherche."
        ),
        phase=phase,
    )


def _node_plan_approval(state: AgentPhaseState) -> AgentPhaseState:
    """The plan approval interrupt (idempotent under re-execution)."""
    from langgraph.types import interrupt

    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    if not state.get("needs_plan_interrupt"):
        return state
    run_id = deps.context.run_id or ""
    replan_round = state.get("replan_rounds", 0)
    plan_id = str(state.get("plan_id", ""))
    if not plan_id:
        latest_plan, _tasks = _run_async(deps.control.get_plan(run_id))
        plan_id = latest_plan.plan_id
        state["plan_id"] = plan_id
    approval_id = f"apr_{run_id[-12:]}_plan_{replan_round}"
    _ensure_approval(
        deps,
        run_id,
        approval_id,
        kind="plan" if replan_round == 0 else "replan",
        payload={
            "plan_id": plan_id,
            "plan_version": state.get("plan_version", 1),
        },
    )
    decision = interrupt({"kind": "approval", "id": approval_id})
    if decision.get("decision") == "reject":
        _finish_with_gate_rejection(
            deps,
            state,
            decision=decision,
            gate_subject="Plan",
            narration_id=f"n-plan-rejected-{replan_round}",
            phase="planning",
        )
        return state
    guidance = str(decision.get("report_guidance", "") or "").strip()
    if guidance:
        state["report_guidance"] = guidance
    # approve OR edit: the control store holds the authoritative latest
    # version (an edit appended one) — reload (rule R5).
    latest, latest_tasks = _run_async(deps.control.get_plan(run_id))
    _plan_lineage_web_research_consent(
        deps,
        state,
        current_plan=latest,
        current_tasks=latest_tasks,
    )
    state["plan_version"] = latest.version
    if latest.success_criteria:
        state["success_criteria"] = list(latest.success_criteria)
    return state


def _outcomes_from_state(state: AgentPhaseState) -> dict[str, TaskOutcome]:
    """Rehydrate the checkpointed outcomes map, tolerating schema drift.

    One place for the read half of the durable-checkpoint round-trip (was
    an identical ``{k: TaskOutcome(**v) ...}`` comprehension in four nodes);
    routes through :meth:`TaskOutcome.from_state` so a schema change does not
    break resume of an in-flight run.
    """
    return {
        key: TaskOutcome.from_state(value)
        for key, value in state.get("outcomes", {}).items()
    }


def _outcomes_to_state(
    outcomes: dict[str, TaskOutcome],
) -> dict[str, dict[str, Any]]:
    """The checkpoint-serializable outcomes map (write half of the round-trip)."""
    return {key: outcome.to_state() for key, outcome in outcomes.items()}


def _source_tool_counts(
    tasks: list[PlanTaskRecord],
    outcomes: dict[str, TaskOutcome],
    *,
    base_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    """Count completed source operations represented by task outcomes."""
    base = base_counts or {}
    counts = {
        "web": max(0, int(base.get("web", 0) or 0)),
        "knowledge": max(0, int(base.get("knowledge", 0) or 0)),
    }
    for task in tasks:
        outcome = outcomes.get(task.task_id)
        if outcome is None or outcome.status not in {
            "completed",
            "insufficient_evidence",
        }:
            continue
        width = max(1, len(_task_queries(task)))
        if task.tool_kind == "web_research":
            # One delegated web tool invocation; its internal query count is
            # intentionally owned by the child run's own execution block.
            counts["web"] += 1
        elif task.tool_kind == "web_instant":
            counts["web"] += width
        elif task.tool_kind in ("rag_query", "file_analysis"):
            counts["knowledge"] += width
    return counts


def _child_origin_key(task: PlanTaskRecord, attempt: int) -> str:
    """Stable child-submission identity across checkpoint re-execution."""
    return (
        f"mission-plan:{task.plan_id}:task:{task.task_id}:attempt:{attempt}"
    )


def _existing_child_for_attempt(
    deps: "_RunDeps", task: PlanTaskRecord, attempt: int
) -> dict[str, Any] | None:
    """Find the durable child for one logical task attempt, if submitted."""
    origin_key = _child_origin_key(task, attempt)
    return next(
        (
            row
            for row in deps.run_service.run_store.children(
                deps.context.run_id or ""
            )
            if row.get("origin_key") == origin_key
        ),
        None,
    )


def _attempt_for_child(
    task: PlanTaskRecord,
    child_run_id: str,
    pending_info: dict[str, Any] | None,
    children: list[dict[str, Any]],
) -> int:
    """Recover a child's logical attempt from checkpoint or origin key."""
    if pending_info and str(pending_info.get("child_run_id", "")) == child_run_id:
        return max(1, int(pending_info.get("attempt", 1) or 1))
    origins = {
        _child_origin_key(task, attempt): attempt for attempt in (1, 2)
    }
    for child in children:
        if str(child.get("run_id", "")) == child_run_id:
            return origins.get(str(child.get("origin_key", "")), 1)
    return 1


def _outcome_from_persisted_task(
    task: PlanTaskRecord,
    *,
    checkpointed: TaskOutcome | None = None,
    child_outcome: TaskOutcome | None = None,
) -> TaskOutcome:
    """Rebuild a complete outcome from row columns plus recovery payload."""
    checkpoint_matches = (
        checkpointed is not None and checkpointed.status == task.status
    )
    base = (
        checkpointed
        if checkpoint_matches and checkpointed is not None
        else TaskOutcome(status=task.status)
    )
    if (
        child_outcome is not None
        and task.status == child_outcome.status
        and task.status in {"completed", "insufficient_evidence"}
    ):
        base = child_outcome
    payload = task.result_payload or {}
    persisted_evidence = [
        dict(item)
        for item in payload.get("evidence", [])
        if isinstance(item, dict)
    ]
    persisted_claims = [
        dict(item)
        for item in payload.get("claims", [])
        if isinstance(item, dict)
    ]
    persisted_ledger = (
        dict(payload.get("web_search_ledger"))
        if isinstance(payload.get("web_search_ledger"), dict)
        else {}
    )
    persisted_usage_raw = payload.get("usage")
    persisted_usage = (
        {
            "prompt_tokens": int(
                persisted_usage_raw.get("prompt_tokens", 0) or 0
            ),
            "completion_tokens": int(
                persisted_usage_raw.get("completion_tokens", 0) or 0
            ),
        }
        if isinstance(persisted_usage_raw, dict)
        else {"prompt_tokens": 0, "completion_tokens": 0}
    )
    failure_reason = base.failure_reason
    failure_code = base.failure_code
    if task.status in {"failed", "cancelled"}:
        failure_reason = (
            failure_reason
            or str(payload.get("failure_reason") or "")
            or (task.result_summary if task.status == "failed" else "")
        )
        failure_code = (
            failure_code
            or str(payload.get("failure_code") or "")
            or (
                "persisted_task_failed"
                if task.status == "failed"
                else "task_cancelled"
            )
        )
    return TaskOutcome(
        status=task.status,
        summary=task.result_summary,
        answer_markdown=(
            base.answer_markdown
            or str(payload.get("answer_markdown") or "")
        ),
        evidence=list(base.evidence) or persisted_evidence,
        web_search_ledger=(
            dict(base.web_search_ledger) or persisted_ledger
        ),
        claims=list(base.claims) or persisted_claims,
        child_run_id=task.child_run_id or base.child_run_id,
        failure_reason=failure_reason,
        failure_code=failure_code,
        usage=(
            dict(base.usage)
            if checkpoint_matches or child_outcome is not None
            else persisted_usage
        ),
        transient=False,
    )


def _cancelled_task_outcome(
    spent: TaskOutcome | None = None,
) -> TaskOutcome:
    """Canonical cancel projection that retains only consumed resources.

    A provider result that arrives after the authoritative task row entered
    ``cancel_requested`` must not become synthesis input.  Usage remains
    accountable, while evidence and prose from the superseded attempt stay
    outside the task-result projection (child runs keep their own audit
    lineage).  Treating late evidence as an ordinary cancelled-task result
    would let a user cancellation influence the final answer on resume.
    """
    return TaskOutcome(
        status="cancelled",
        summary="Aufgabe auf Nutzerwunsch abgebrochen.",
        child_run_id=spent.child_run_id if spent is not None else None,
        failure_code="task_cancelled",
        usage=dict(spent.usage) if spent is not None else {},
    )


def _current_plan_task(
    deps: "_RunDeps", task: PlanTaskRecord
) -> PlanTaskRecord:
    """Reload one task from the authoritative latest plan version."""
    _plan, tasks = _run_async(deps.control.get_plan(task.run_id))
    return next(
        row
        for row in tasks
        if row.plan_id == task.plan_id and row.task_id == task.task_id
    )


def _reconcile_persisted_execution(
    deps: "_RunDeps",
    state: AgentPhaseState,
    tasks: list[PlanTaskRecord],
    outcomes: dict[str, TaskOutcome],
    pending: dict[str, dict[str, Any]],
    *,
    research_policy: WebResearchPolicy,
) -> tuple[dict[str, TaskOutcome], dict[str, dict[str, Any]]]:
    """Reconcile checkpoint caches with task rows without repeating work.

    A terminal row is never reopened. A running child reattaches to its
    durable run (including the submit-before-checkpoint crash window). A
    running local task has an unknowable external outcome and therefore fails
    closed instead of executing twice.
    """
    children = deps.run_service.run_store.children(deps.context.run_id or "")
    for original in tasks:
        if original.tool_kind == "synthesis":
            continue
        task = original
        checkpointed = outcomes.get(task.task_id)
        info = pending.get(task.task_id)
        if task.status in TASK_TERMINAL_STATUSES:
            pending.pop(task.task_id, None)
            child_outcome: TaskOutcome | None = None
            if task.child_run_id:
                attempt = _attempt_for_child(
                    task, task.child_run_id, info, children
                )
                child_outcome = _child_task_outcome(
                    deps, task.child_run_id, attempt
                )
            recovered = _outcome_from_persisted_task(
                task,
                checkpointed=checkpointed,
                child_outcome=child_outcome,
            )
            outcomes[task.task_id] = recovered
            if checkpointed is None and any(recovered.usage.values()):
                # The row won the terminal-write -> checkpoint crash race.
                # Its usage has not reached state yet. A checkpointed outcome
                # already contributed to cumulative usage and must never be
                # charged again on resume.
                _add_usage(state, recovered.usage)
            continue
        if task.tool_kind != "web_research":
            pending.pop(task.task_id, None)
            if task.status == "cancel_requested":
                outcome = _cancelled_task_outcome()
                outcomes[task.task_id] = outcome
                _emit_task_ended(deps, task, task.task_id, outcome)
                continue
            if task.status == "running":
                outcome = TaskOutcome(
                    status="failed",
                    summary=(
                        "Ausführung nach einer Unterbrechung nicht erneut "
                        "gestartet."
                    ),
                    failure_reason="local_execution_interrupted",
                    failure_code="local_execution_interrupted",
                )
                outcomes[task.task_id] = outcome
                _emit_task_ended(deps, task, task.task_id, outcome)
            continue
        if task.status == "cancel_requested" and not (
            task.child_run_id or info
        ):
            outcome = _cancelled_task_outcome()
            outcomes[task.task_id] = outcome
            _emit_task_ended(deps, task, task.task_id, outcome)
            continue
        if task.status not in {"running", "cancel_requested"} and info is None:
            continue

        child_id = str(task.child_run_id or (info or {}).get("child_run_id") or "")
        attempt = max(1, int((info or {}).get("attempt", 1) or 1))
        if child_id:
            attempt = _attempt_for_child(task, child_id, info, children)
        else:
            # Prefer a later retry if it exists; otherwise submit-or-find the
            # logical attempt checkpointed before the crash (normally one).
            existing = _existing_child_for_attempt(deps, task, 2)
            if existing is not None:
                attempt = 2
            else:
                existing = _existing_child_for_attempt(deps, task, attempt)
            if existing is not None:
                child_id = str(existing["run_id"])
            else:
                try:
                    child_id = _submit_child_run(
                        deps,
                        state,
                        task,
                        attempt,
                        research_policy=research_policy,
                    )
                except Exception as exc:  # noqa: BLE001
                    outcome = TaskOutcome(
                        status="failed",
                        failure_reason=sanitize_error(exc),
                        failure_code=_task_failure_code(exc),
                    )
                    outcomes[task.task_id] = outcome
                    pending.pop(task.task_id, None)
                    _emit_task_ended(deps, task, task.task_id, outcome)
                    continue
        if task.status == "pending" or task.child_run_id != child_id:
            task = _transition_task(
                deps,
                task,
                status="running",
                child_run_id=child_id,
            )
        pending[task.task_id] = {
            "child_run_id": child_id,
            "attempt": attempt,
        }
    return outcomes, pending


def _fold_pending_children(
    deps: "_RunDeps",
    state: AgentPhaseState,
    tasks: list[PlanTaskRecord],
    outcomes: dict[str, TaskOutcome],
    pending: dict[str, dict[str, Any]],
    *,
    research_policy: WebResearchPolicy,
) -> tuple[dict[str, TaskOutcome], dict[str, dict[str, Any]]]:
    """Fold terminal children and retain only genuinely active attempts."""
    tasks_by_id = {task.task_id: task for task in tasks}
    still_pending: dict[str, dict[str, Any]] = {}
    for task_id, info in pending.items():
        child_id = str(info.get("child_run_id", ""))
        attempt = int(info.get("attempt", 1) or 1)
        task = tasks_by_id.get(task_id)
        outcome = _child_task_outcome(deps, child_id, attempt)
        if outcome is None:
            still_pending[task_id] = info
            continue
        if task is not None and task.status == "cancel_requested":
            outcome = _cancelled_task_outcome(outcome)
        if task is not None and should_retry(
            outcome, attempt=attempt, cancelled=deps.cancelled()
        ):
            log.warning(
                "Task %s transient fehlgeschlagen (failure_code=%s) — ein Retry mit "
                "unveraenderten Operatorgrenzen.",
                task_id,
                outcome.failure_code or "unknown",
            )
            try:
                retry_id = _submit_child_run(
                    deps,
                    state,
                    task,
                    2,
                    research_policy=research_policy,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Agent-Task %s Retry konnte nicht eingereicht werden "
                    "(error_type=%s).",
                    task_id,
                    type(exc).__name__,
                )
                outcomes[task_id] = outcome
                _emit_task_ended(deps, task, task_id, outcome)
                continue
            still_pending[task_id] = {
                "child_run_id": retry_id,
                "attempt": 2,
            }
            _transition_task(
                deps,
                task,
                status="running",
                child_run_id=retry_id,
            )
            continue
        outcomes[task_id] = outcome
        _emit_task_ended(deps, task, task_id, outcome)
    return outcomes, still_pending


def _node_execute(state: AgentPhaseState) -> AgentPhaseState:
    """Run the plan's waves — in-process tools inline, children parked.

    ``web_research`` tasks are SUBMITTED as child runs and awaited via
    the children_wait interrupt (the parent parks slot-free instead of
    block-polling siblings out of the shared execution pool); all other
    tools run in-process through :func:`execute_wave` exactly as
    before. The node re-enters after every fold with the remaining
    todo, so waves progress across park/resume segments.
    """
    deps = _deps()
    if deps.cancelled():
        _run_async(
            settle_cancelled_plan_tasks(
                deps.control, deps.context.run_id or ""
            )
        )
        return {"cancelled": True}
    _set_phase(state, "execution")
    run_id = deps.context.run_id or ""
    plan_record, tasks = _run_async(deps.control.get_plan(run_id))
    _emit_legacy_budget_notices(deps, state, tasks)
    research_policy = _research_child_policy(
        deps,
        edit_consent=_plan_lineage_web_research_consent(
            deps,
            state,
            current_plan=plan_record,
            current_tasks=tasks,
        ),
    )
    outcomes = _outcomes_from_state(state)
    pending = dict(state.get("pending_children") or {})
    outcomes, pending = _reconcile_persisted_execution(
        deps,
        state,
        tasks,
        outcomes,
        pending,
        research_policy=research_policy,
    )
    outcomes, pending = _fold_pending_children(
        deps,
        state,
        tasks,
        outcomes,
        pending,
        research_policy=research_policy,
    )
    if pending:
        state["outcomes"] = _outcomes_to_state(outcomes)
        state["tool_use_counts"] = _source_tool_counts(
            tasks,
            outcomes,
            base_counts=state.get("discovery_tool_use_counts"),
        )
        state["pending_children"] = pending
        return state
    todo = [
        task
        for task in tasks
        if task.tool_kind != "synthesis"
        and task.task_id not in outcomes
        and task.task_id not in pending
    ]
    max_parallel = max(1, deps.platform.max_parallel_children)
    for wave in topological_waves(todo):
        if state.get("cancelled") or deps.cancelled():
            state["cancelled"] = True
            break
        local_tasks = [
            task for task in wave if task.tool_kind != "web_research"
        ]
        child_tasks = [
            task for task in wave if task.tool_kind == "web_research"
        ]
        if child_tasks:
            # Submit children first so they overlap the independent local work
            # below. When both kinds exist and capacity permits, reserve one
            # slot for local work; this is one shared wave limit, not a second
            # scheduler.
            child_limit = (
                max_parallel - 1
                if local_tasks and max_parallel > 1
                else max_parallel
            )
            for task in child_tasks:
                if len(pending) >= child_limit:
                    break
                try:
                    task = _transition_task(deps, task, status="running")
                except ValueError:
                    current = _current_plan_task(deps, task)
                    if current.status not in TASK_TERMINAL_STATUSES:
                        raise
                    outcome = _outcome_from_persisted_task(current)
                    outcomes[task.task_id] = outcome
                    _emit_task_ended(
                        deps, current, current.task_id, outcome
                    )
                    continue
                deps.emit(
                    TASK_STARTED_EVENT,
                    {
                        "task_id": task.task_id,
                        "ordinal": task.ordinal,
                        "tool_kind": task.tool_kind,
                        "attempt": 1,
                    },
                )
                try:
                    child_id = _submit_child_run(
                        deps,
                        state,
                        task,
                        1,
                        research_policy=research_policy,
                    )
                except Exception as exc:  # noqa: BLE001 — admission stays visible
                    log.warning(
                        "Agent-Task %s (web_research) konnte nicht "
                        "eingereicht werden (error_type=%s).",
                        task.task_id,
                        type(exc).__name__,
                    )
                    outcome = TaskOutcome(
                        status="failed",
                        failure_reason=sanitize_error(exc),
                        failure_code=_task_failure_code(exc),
                        transient=False,
                    )
                    outcomes[task.task_id] = outcome
                    _emit_task_ended(deps, task, task.task_id, outcome)
                    continue
                pending[task.task_id] = {
                    "child_run_id": child_id,
                    "attempt": 1,
                }
                try:
                    _transition_task(
                        deps,
                        task,
                        status="running",
                        child_run_id=child_id,
                    )
                except ValueError:
                    current = _current_plan_task(deps, task)
                    if current.status != "cancel_requested":
                        raise
                    _transition_task(
                        deps,
                        current,
                        status="cancel_requested",
                        child_run_id=child_id,
                    )
                    deps.run_service.run_store.cancel(
                        child_id,
                        workspace_id=deps.context.workspace_id,
                    )
        local_capacity = max_parallel - len(pending)
        if local_tasks and local_capacity > 0:
            # Control-store clients may own loop-affine async engines. Persist
            # task state serially on the graph thread; worker-pool functions
            # execute provider work only and never touch the store. Mark only
            # the next executable batch running: queued work beyond the shared
            # capacity remains honestly pending if a cancel arrives.
            for offset in range(0, len(local_tasks), local_capacity):
                if state.get("cancelled") or deps.cancelled():
                    state["cancelled"] = True
                    break
                candidate_batch = local_tasks[
                    offset : offset + local_capacity
                ]
                batch: list[PlanTaskRecord] = []
                for task in candidate_batch:
                    try:
                        batch.append(
                            _transition_task(deps, task, status="running")
                        )
                    except ValueError:
                        current = _current_plan_task(deps, task)
                        if current.status not in TASK_TERMINAL_STATUSES:
                            raise
                        outcome = _outcome_from_persisted_task(current)
                        outcomes[task.task_id] = outcome
                        _emit_task_ended(
                            deps, current, current.task_id, outcome
                        )
                if not batch:
                    continue

                def persist_local_outcome(
                    task: PlanTaskRecord, outcome: TaskOutcome
                ) -> None:
                    _emit_task_ended(
                        deps, task, task.task_id, outcome
                    )
                    current = _current_plan_task(deps, task)
                    outcomes[task.task_id] = _outcome_from_persisted_task(
                        current,
                        checkpointed=(
                            outcome
                            if current.status == outcome.status
                            else None
                        ),
                    )

                def admit_local_retry(task: PlanTaskRecord) -> bool:
                    current = _current_plan_task(deps, task)
                    admitted = (
                        current.status == "running"
                        and not deps.cancelled()
                    )
                    if not admitted:
                        log.info(
                            "Agent-Task %s: Retry wegen Status %s verworfen.",
                            task.task_id,
                            current.status,
                        )
                    return admitted

                batch_outcomes = execute_wave(
                    batch,
                    executor=lambda task, attempt: _execute_task(
                        deps, state, task, attempt
                    ),
                    max_parallel=local_capacity,
                    cancelled=deps.cancelled,
                    on_outcome=persist_local_outcome,
                    retry_allowed=admit_local_retry,
                )
                batch_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
                for outcome in batch_outcomes.values():
                    batch_usage["prompt_tokens"] += int(
                        outcome.usage.get("prompt_tokens", 0) or 0
                    )
                    batch_usage["completion_tokens"] += int(
                        outcome.usage.get("completion_tokens", 0) or 0
                    )
                # Persist every parallel outcome before enforcing the
                # cumulative run cap; a typed stop must never strand a
                # sibling that already finished in ``running``.
                _add_usage(state, batch_usage)
                if any(
                    outcomes[task_id].failure_code
                    == "token_budget_exceeded"
                    for task_id in batch_outcomes
                ):
                    raise AgentTokenBudgetExceeded(
                        "Lauf wegen Token-Budget gestoppt.",
                        usage=state.get("usage", {}),
                    )
                if any(
                    outcomes[task_id].failure_code
                    == "client_requested_cancel"
                    for task_id in batch_outcomes
                ):
                    state["cancelled"] = True
                    break
        if state.get("cancelled") or deps.cancelled():
            state["cancelled"] = True
            break
        if pending:
            # The parent parks after its overlapping local work finishes;
            # remaining tasks re-enter from authoritative rows after wake.
            break
    if deps.cancelled():
        # A cancel that arrived DURING the final wave leaves the loop
        # without the between-waves check firing; the flag must land so
        # the run ends cancelled, not "all_tasks_failed".
        state["cancelled"] = True
    state["outcomes"] = _outcomes_to_state(outcomes)
    state["tool_use_counts"] = _source_tool_counts(
        tasks,
        outcomes,
        base_counts=state.get("discovery_tool_use_counts"),
    )
    state["pending_children"] = pending
    if (
        outcomes
        and not pending
        and not state.get("cancelled")
        and all(
            outcome.status == "failed" for outcome in outcomes.values()
        )
    ):
        # §4 failure policy: synthesis always runs EXCEPT when every
        # task hard-failed — an evidence-free memo would be a lie.
        # (insufficient_evidence is NOT a hard failure: it feeds the
        # replan gate. Pending children defer the verdict: their
        # outcomes are still open.)
        synthesis_task = next(
            (task for task in tasks if task.tool_kind == "synthesis"),
            None,
        )
        if synthesis_task is not None and synthesis_task.status == "pending":
            _emit_task_ended(
                deps,
                synthesis_task,
                synthesis_task.task_id,
                TaskOutcome(
                    status="skipped",
                    summary=(
                        "Nicht ausgeführt, weil alle vorgelagerten Aufgaben "
                        "fehlgeschlagen sind."
                    ),
                ),
            )
        state["failure"] = "all_tasks_failed"
    return state


def _emit_task_ended(
    deps: "_RunDeps",
    task: "PlanTaskRecord | None",
    task_id: str,
    outcome: TaskOutcome,
) -> None:
    """Emit the shared TASK_FINISHED/TASK_FAILED projection."""
    persisted_summary = outcome.summary or outcome.failure_reason
    row_summary = task_result_summary(persisted_summary)
    if task is not None:
        try:
            task = _transition_task(
                deps,
                task,
                status=outcome.status,
                child_run_id=outcome.child_run_id,
                result_summary=row_summary,
                result_payload=task_result_payload(
                    outcome, persisted_summary=row_summary
                ),
            )
        except ValueError:
            current = _current_plan_task(deps, task)
            if current.status not in {"cancel_requested", "cancelled"}:
                raise
            outcome = _cancelled_task_outcome(outcome)
            persisted_summary = outcome.summary
            row_summary = task_result_summary(persisted_summary)
            task = (
                current
                if current.status == "cancelled"
                else _transition_task(
                    deps,
                    current,
                    status="cancelled",
                    child_run_id=current.child_run_id,
                    result_summary=row_summary,
                    result_payload=task_result_payload(
                        outcome, persisted_summary=row_summary
                    ),
                )
            )
    metrics = {
        "reference_count": len(outcome.evidence),
        "claim_count": len(outcome.claims),
    }
    deps.emit(
        TASK_FINISHED_EVENT
        if outcome.status != "failed"
        else TASK_FAILED_EVENT,
        {
            "task_id": task_id,
            "ordinal": task.ordinal if task else 0,
            "tool_kind": task.tool_kind if task else "",
            "status": outcome.status,
            "child_run_id": outcome.child_run_id,
            "result_summary": row_summary,
            "metrics": metrics,
            **(
                {
                    "error": outcome.failure_reason,
                    "failure": {
                        "code": outcome.failure_code or "task_failed",
                        "message": outcome.failure_reason,
                    }
                }
                if outcome.failure_reason
                else {}
            ),
        },
    )
    if outcome.status == "completed":
        _emit_narration(
            deps,
            narration_id=f"n-task-{task_id}",
            kind="task",
            text=task_narration(
                task.title if task else task_id, outcome.summary
            ),
            phase="execution",
        )


def _node_children_wait(state: AgentPhaseState) -> AgentPhaseState:
    """Park on the submitted children; fold their outcomes on resume.

    The run rows are checked before parking so a child that finished in the
    submit/checkpoint window is folded immediately. Only genuinely active
    children reach ``interrupt``; their terminal write re-queues the parent.
    A transient failure is resubmitted once with unchanged operator limits.
    """
    from langgraph.types import interrupt

    deps = _deps()
    if not state.get("pending_children"):
        if deps.cancelled():
            _run_async(
                settle_cancelled_plan_tasks(
                    deps.control, deps.context.run_id or ""
                )
            )
            return {"cancelled": True}
        return state
    run_id = deps.context.run_id or ""
    plan_record, tasks = _run_async(deps.control.get_plan(run_id))
    _emit_legacy_budget_notices(deps, state, tasks)
    research_policy = _research_child_policy(
        deps,
        edit_consent=_plan_lineage_web_research_consent(
            deps,
            state,
            current_plan=plan_record,
            current_tasks=tasks,
        ),
    )
    outcomes = _outcomes_from_state(state)
    outcomes, still_pending = _fold_pending_children(
        deps,
        state,
        tasks,
        outcomes,
        dict(state.get("pending_children") or {}),
        research_policy=research_policy,
    )
    state["outcomes"] = _outcomes_to_state(outcomes)
    state["tool_use_counts"] = _source_tool_counts(
        tasks,
        outcomes,
        base_counts=state.get("discovery_tool_use_counts"),
    )
    state["pending_children"] = still_pending
    if deps.cancelled():
        # A child may have committed its terminal run immediately before the
        # parent cancel arrived. Fold first so the authoritative completed
        # task cannot be downgraded to failed by cancellation settlement.
        _run_async(settle_cancelled_plan_tasks(deps.control, run_id))
        state["cancelled"] = True
        return state
    if still_pending:
        interrupt({"kind": "children"})
    return state


def _node_evidence(state: AgentPhaseState) -> AgentPhaseState:
    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    _set_phase(state, "evidence")
    outcomes = _outcomes_from_state(state)
    references, claims = evidence.merge_evidence(outcomes)
    web_search_ledger = merge_web_search_ledgers(
        [
            outcome.web_search_ledger
            for outcome in outcomes.values()
            if outcome.web_search_ledger
        ]
    )
    references = attach_web_search_lineage(
        [dict(reference) for reference in references],
        web_search_ledger,
    )
    state["references"] = references
    state["claims"] = claims
    state["web_search_ledger"] = web_search_ledger
    if web_search_ledger.get("searches"):
        _run_async(
            deps.control.upsert_artifact(
                run_id=deps.context.run_id or "",
                kind="evidence_bundle",
                session_id=None,
                title="Agent evidence",
                status="ready",
                content_markdown="",
                payload={
                    "schema_version": 1,
                    "web_search_ledger": web_search_ledger,
                },
                refs=references,
                updated_by="agent",
                artifact_id=f"art_{(deps.context.run_id or '')[-12:]}_evidence",
            )
        )

    consolidator = getattr(
        deps.context.strategies, "claim_consolidation", None
    )
    if claims and consolidator is not None:
        pairs = evidence.overlapping_claim_pairs(
            claims, signature=consolidator.claim_signature
        )
        model, effort = deps.resolved("agent_contradiction")
        outcome = evidence.run_contradiction_analysis(
            deps.llm,
            pairs=pairs,
            model=model,
            reasoning_effort=effort,
            timeout=deps.timeout,
        )
        if outcome is not None:
            _add_usage(state, outcome.usage)
            report = outcome.value
            state["contradictions"] = (
                [c.model_dump() for c in report.contradictions]
                if isinstance(report, ContradictionReport)
                else []
            )
    model, effort = deps.resolved("agent_sufficiency")
    outcome = evidence.run_sufficiency_judgement(
        deps.llm,
        success_criteria=state.get("success_criteria", []),
        evidence_digest=evidence.evidence_digest(references),
        model=model,
        reasoning_effort=effort,
        timeout=deps.timeout,
    )
    _add_usage(state, outcome.usage)
    judgement = outcome.value
    if isinstance(judgement, SufficiencyJudgement):
        state["sufficiency"] = judgement.model_dump()
    else:
        # A failed verdict must never silently mean "covered": degrade
        # to the conservative middle (partial) and say so.
        log.warning(
            "Sufficiency-Urteil nicht auswertbar — konservativ als "
            "'partial' behandelt."
        )
        state["sufficiency"] = {"coverage": "partial", "missing": []}
    return state


def _node_replan_gate(state: AgentPhaseState) -> AgentPhaseState:
    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    outcomes = _outcomes_from_state(state)
    discovery = _discovery_of(state)
    blocking_uncovered = False
    if discovery is not None:
        covered_gaps: set[str] = set()
        run_id = deps.context.run_id or ""
        _plan, tasks = _run_async(deps.control.get_plan(run_id))
        for task in tasks:
            if (
                task.task_id in outcomes
                and outcomes[task.task_id].status == "completed"
            ):
                covered_gaps.update(task.gap_ids)
        blocking_uncovered = any(
            gap.blocking and gap.gap_id not in covered_gaps
            for gap in discovery.gaps
        )
    sufficiency = state.get("sufficiency") or {}
    needs_replan = evaluate_replan(
        outcomes=outcomes,
        blocking_gap_uncovered=blocking_uncovered,
        sufficiency_coverage=str(sufficiency.get("coverage", "covered")),
        replan_rounds_used=state.get("replan_rounds", 0),
        max_replan_rounds=deps.platform.max_replan_rounds,
    )
    if needs_replan:
        # Make the gate's decision visible at the decision site (its sibling
        # decision nodes all log/emit) with the ACCURATE reason derived from
        # the same inputs — the old ReplanDecision.reason mislabelled the
        # blocking-gap branch as "task_failed" and is intentionally not
        # revived (Prinzip 5).
        reason = (
            "blocking_gap_uncovered"
            if blocking_uncovered
            else "sufficiency_uncovered"
            if str(sufficiency.get("coverage", "covered")) == "uncovered"
            else "task_failure_partial"
        )
        log.info(
            "Replan-Gate: Runde %d ausgeloest (%s).",
            state.get("replan_rounds", 0) + 1,
            reason,
        )
        state["replan_rounds"] = state.get("replan_rounds", 0) + 1
        state["route"] = "replan"
    else:
        state["route"] = "synthesize"
    return state


def _restore_completed_synthesis(
    deps: "_RunDeps", state: AgentPhaseState
) -> bool:
    """Restore a synthesis output committed before its node checkpoint."""
    run_id = deps.context.run_id or ""
    artifact_id = state.get("artifact_id") or (
        f"art_{run_id[-12:]}_answer"
        if state.get("deliverable") == "chat"
        else f"art_{(state.get('session_id') or run_id)[-12:]}_memo"
    )
    try:
        artifact, _revisions = _run_async(
            deps.control.get_artifact(
                run_id=run_id,
                artifact_id=artifact_id,
            )
        )
    except ArtifactNotFound:
        return False
    if not artifact.content_markdown:
        return False
    state["artifact_id"] = artifact.artifact_id
    state["memo_markdown"] = artifact.content_markdown
    state["memo_title"] = artifact.title
    state["memo_base_revision"] = artifact.revision
    return True


def _node_synthesize(state: AgentPhaseState) -> AgentPhaseState:
    """Execute synthesis exactly once and persist its plan-task lifecycle."""
    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    run_id = deps.context.run_id or ""
    _plan, tasks = _run_async(deps.control.get_plan(run_id))
    task = next((row for row in tasks if row.tool_kind == "synthesis"), None)
    if task is None:
        state["failure"] = "synthesis_task_missing"
        return state
    if task.status == "completed":
        if not _restore_completed_synthesis(deps, state):
            state["failure"] = "synthesis_output_missing"
        return state
    if task.status in {"failed", "skipped", "insufficient_evidence"}:
        state["failure"] = task.result_summary or "synthesis_failed"
        return state
    if task.status == "running":
        outcome = TaskOutcome(
            status="failed",
            summary="Synthese nach einer Unterbrechung nicht erneut gestartet.",
            failure_reason="synthesis_execution_interrupted",
            failure_code="synthesis_execution_interrupted",
        )
        _emit_task_ended(deps, task, task.task_id, outcome)
        state["failure"] = "synthesis_execution_interrupted"
        return state
    task = _transition_task(deps, task, status="running")
    deps.emit(
        TASK_STARTED_EVENT,
        {
            "task_id": task.task_id,
            "ordinal": task.ordinal,
            "tool_kind": task.tool_kind,
            "attempt": 1,
        },
    )
    try:
        result = _perform_synthesis(state)
    except synthesis.CitationValidationFailed as exc:
        _add_usage(state, exc.usage)
        deps.emit(
            ACTIVITY_EVENT,
            {
                "kind": "citation_validation_failed",
                "operation": "synthesis.citation_validation",
                "status": "failed",
                "error": {
                    "code": "citation_validation_failed",
                    "message": (
                        "Die Synthese enthielt nach einem Reparaturversuch "
                        "weiterhin unbekannte Belege-Labels."
                    ),
                },
            },
        )
        _emit_task_ended(
            deps,
            task,
            task.task_id,
            TaskOutcome(
                status="failed",
                summary="Synthese wegen ungültiger Belege-Labels verworfen.",
                failure_reason="citation_validation_failed",
                failure_code="citation_validation_failed",
            ),
        )
        raise
    except AgentCancelled:
        _emit_task_ended(
            deps,
            task,
            task.task_id,
            TaskOutcome(
                status="failed",
                summary=(
                    "Synthese abgebrochen; ein begonnenes Ergebnis wurde "
                    "nicht übernommen."
                ),
                failure_reason="client_requested_cancel",
                failure_code="client_requested_cancel",
            ),
        )
        raise
    except Exception:
        _emit_task_ended(
            deps,
            task,
            task.task_id,
            TaskOutcome(
                status="failed",
                summary="Synthese fehlgeschlagen.",
                failure_reason="synthesis_failed",
                failure_code="synthesis_failed",
            ),
        )
        raise
    if result.get("cancelled"):
        _emit_task_ended(
            deps,
            task,
            task.task_id,
            TaskOutcome(
                status="failed",
                summary=(
                    "Synthese abgebrochen; ein begonnenes Ergebnis wurde "
                    "nicht übernommen."
                ),
                failure_reason="client_requested_cancel",
                failure_code="client_requested_cancel",
            ),
        )
        return result
    outcome = TaskOutcome(
        status="completed",
        summary=task_result_summary(
            result.get("memo_markdown", "") or "Synthese abgeschlossen."
        ),
        answer_markdown=str(result.get("memo_markdown", "") or ""),
    )
    _emit_task_ended(deps, task, task.task_id, outcome)
    return result


def _ranked_digest_references(
    deps: "_RunDeps", references: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The PROMPT digest scope: rank-capped above the evidence budget.

    The citation ledger itself is never truncated — every label a
    section cites still resolves; only what the synthesis/revision
    prompt SEES is bounded.
    """
    tiering = getattr(deps.context.strategies, "source_tiering", None)
    return report_quality.rank_evidence(
        references,
        budget=deps.platform.synthesis_evidence_budget,
        tier_for_url=tiering.tier_for_url if tiering is not None else None,
    )


def _perform_synthesis(state: AgentPhaseState) -> AgentPhaseState:
    """Build the chat or canvas deliverable after lifecycle admission."""
    deps = _deps()
    _set_phase(state, "synthesis")
    references = state.get("references", [])
    digest_references = _ranked_digest_references(deps, references)
    digest = evidence.evidence_digest(digest_references)
    contradictions_digest = "\n".join(
        f"- {c.get('internal_position', '')} VS "
        f"{c.get('external_position', '')} ({c.get('severity', '')})"
        for c in state.get("contradictions", [])
    )
    if state.get("deliverable") == "chat":
        return _synthesize_chat_answer(
            deps, state, digest, contradictions_digest
        )
    model, effort = deps.resolved("agent_synthesis")
    skills_block = _skills_prompt_block(deps, state)
    outline_outcome = synthesis.run_outline(
        deps.llm,
        question=state["question"],
        success_criteria=state.get("success_criteria", []),
        evidence_digest=digest,
        prior_memo=state.get("prior_memo", ""),
        skills_block=skills_block,
        user_guidance=state.get("report_guidance", ""),
        model=model,
        reasoning_effort=effort,
        timeout=deps.timeout,
    )
    _add_usage(state, outline_outcome.usage)
    outline = outline_outcome.value
    if not isinstance(outline, ReportOutline):
        outline = ReportOutline(title="Memo", sections=[])
    if not outline.sections:
        from inqtrix.agents.phase_models import ReportSection

        outline = ReportOutline(
            title=outline.title or "Memo",
            sections=[
                ReportSection(
                    title="Ergebnis",
                    focus="Beantworte den Auftrag mit allen Belegen.",
                    criterion_ids=[],
                    evidence_labels=[
                        ref["label"] for ref in digest_references[:12]
                    ],
                ),
                ReportSection(
                    title="Offene Punkte",
                    focus="Unbelegtes und Luecken ehrlich benennen.",
                    criterion_ids=[],
                    evidence_labels=[],
                ),
            ],
        )
    _emit_narration(
        deps,
        narration_id="n-synthesis",
        kind="synthesis",
        text=synthesis_narration(outline.title, len(outline.sections)),
        phase="synthesis",
    )
    sections: list[tuple[str, str]] = []
    for index, section in enumerate(outline.sections):
        if deps.cancelled():
            state["cancelled"] = True
            break
        section_digest = evidence.evidence_digest(
            references, labels=section.evidence_labels
        )
        body, usage = synthesis.write_section(
            deps.llm,
            question=state["question"],
            section_title=section.title,
            section_focus=section.focus,
            evidence_digest=section_digest,
            contradictions_digest=contradictions_digest,
            skills_block=skills_block,
            user_guidance=state.get("report_guidance", ""),
            model=model,
            reasoning_effort=effort,
            timeout=deps.timeout,
            known_labels=[str(ref.get("label") or "") for ref in references],
        )
        _add_usage(state, usage)
        sections.append((section.title, body))
        memo = synthesis.assemble_memo(outline.title, sections)
        state["memo_markdown"] = memo
        state["memo_title"] = outline.title
        _flush_memo(deps, state, memo, status="writing")
        _emit_narration(
            deps,
            narration_id=f"n-section-{index}",
            kind="synthesis",
            text=section_narration(section.title),
            phase="synthesis",
        )
    return state


def answer_node_for(tasks: list[Any]) -> str:
    """The R1 auto-downgrade rule, deterministic and in ONE place.

    An approved web task keeps synthesis on the high tier even when that task
    returned no references. Plans without a web task use the mid tier. This
    ties model selection to the authorized work rather than to output luck.
    """
    has_web = any(
        str(
            task.get("tool_kind", "")
            if isinstance(task, dict)
            else getattr(task, "tool_kind", "")
        )
        in {"web_research", "web_instant"}
        for task in tasks
    )
    return "agent_answer" if has_web else "agent_answer_light"


def _synthesize_chat_answer(
    deps: "_RunDeps",
    state: AgentPhaseState,
    digest: str,
    contradictions_digest: str,
) -> AgentPhaseState:
    """The chat-form deliverable: ONE answer call, no outline loop.

    Model routing (R1) follows the approved plan: a web task uses the high
    tier even when retrieval yielded no references; all-internal work uses
    the mid tier.
    """
    _plan, tasks = _run_async(
        deps.control.get_plan(deps.context.run_id or "")
    )
    node = answer_node_for(tasks)
    model, effort = deps.resolved(node)
    body, usage = synthesis.write_chat_answer(
        deps.llm,
        question=state["question"],
        evidence_digest=digest,
        contradictions_digest=contradictions_digest,
        history=state.get("history", ""),
        prior_memo=state.get("prior_memo", ""),
        skills_block=_skills_prompt_block(deps, state),
        user_guidance=state.get("report_guidance", ""),
        model=model,
        reasoning_effort=effort,
        timeout=deps.timeout,
        known_labels=[
            str(ref.get("label") or "")
            for ref in state.get("references", [])
        ],
    )
    _add_usage(state, usage)
    state["memo_markdown"] = body
    state["memo_title"] = "Antwort"
    _flush_deliverable(deps, state, body, status="writing")
    _emit_narration(
        deps,
        narration_id="n-answer",
        kind="synthesis",
        text="Antwort verfasst.",
        phase="synthesis",
    )
    return state


def _node_critic(state: AgentPhaseState) -> AgentPhaseState:
    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    _set_phase(state, "critic")
    state["critic_recheck_pending"] = False
    memo = state.get("memo_markdown", "")
    references = state.get("references", [])
    coverage = synthesis.citation_coverage(memo)
    quote_checks = synthesis.verify_quotes(memo, references)
    facts = critic_phase.precomputed_facts(
        coverage=coverage,
        quote_checks=quote_checks,
        contradictions=state.get("contradictions", []),
        memo_markdown=memo,
    )
    memory_briefing = state.get("memory_briefing", "").strip()
    if memory_briefing:
        facts = (
            f"{facts}\n\n"
            "Nicht zitierfaehiges Memory-Briefing (nur Kontext; aktuelle "
            "Evidenz gewinnt bei Konflikten):\n"
            f"{memory_briefing}"
        )
    model, effort = deps.resolved("agent_critic")
    outcome = critic_phase.run_critic(
        deps.llm,
        memo_markdown=memo,
        success_criteria=state.get("success_criteria", []),
        facts=facts,
        user_guidance=state.get("report_guidance", ""),
        model=model,
        reasoning_effort=effort,
        timeout=deps.timeout,
    )
    _add_usage(state, outcome.usage)
    report = outcome.value
    if (
        isinstance(report, AgentCriticReport)
        and report.verdict == "pass"
        and deps.tier_policy is not None
        and deps.tier_policy.verify == "escalating"
        and state.get("revisions_used", 0) < 1
    ):
        # Tief tier: unverified web-cited quotes flip a passing verdict to
        # revise ONCE — consuming the existing revision budget, no new loop.
        flagged = report_quality.unverified_web_quotes(memo, quote_checks)
        if flagged:
            log.warning(
                "Tief-Stufe eskaliert: %d unverifizierte web-zitierte "
                "Zitate kippen den Critic auf revise.",
                len(flagged),
            )
            deps.emit(
                ACTIVITY_EVENT,
                {
                    "kind": "critic_web_quote_escalation",
                    "label": "Web-Zitate nicht verifiziert",
                    "detail": (
                        "Woertliche Zitate mit Web-Beleg wurden nicht in "
                        "den gespeicherten Auszuegen gefunden; eine "
                        "Revision behebt das."
                    ),
                },
            )
            report = report.model_copy(
                update={
                    "verdict": "revise",
                    "findings": [
                        *report.findings,
                        AgentCriticFinding(
                            kind="unverified_web_quote",
                            detail=(
                                "Woertliche Zitate mit Web-Beleg konnten "
                                "nicht gegen die gespeicherten "
                                "Quellenauszuege verifiziert werden: "
                                + "; ".join(
                                    f'"{quote[:80]}"'
                                    for quote in flagged[:3]
                                )
                            ),
                            suggested_fix=(
                                "Zitate durch belegte Formulierungen aus "
                                "den Auszuegen ersetzen oder als "
                                "Paraphrase ohne Anfuehrungszeichen "
                                "kennzeichnen."
                            ),
                        ),
                    ],
                }
            )
    state["critic"] = (
        report.model_dump()
        if isinstance(report, AgentCriticReport)
        else None
    )
    if isinstance(report, AgentCriticReport) and any(
        finding.kind == "memory_conflict" for finding in report.findings
    ):
        deps.emit(
            ACTIVITY_EVENT,
            {
                "kind": "memory_conflict",
                "label": "Memory-Konflikt erkannt",
                "detail": "Aktuelle Evidenz hat Vorrang vor Memory-Kontext.",
            },
        )
    if (
        isinstance(report, AgentCriticReport)
        and report.verdict == "research"
    ):
        if state.get("replan_rounds", 0) < deps.platform.max_replan_rounds:
            state["replan_rounds"] = state.get("replan_rounds", 0) + 1
            state["route"] = "critic_research"
            deps.emit(
                ACTIVITY_EVENT,
                {
                    "kind": "critic_research",
                    "label": "Critic fordert Recherche",
                    "detail": (
                        "Das Memo braucht weitere Evidenz; der Plan wird "
                        "additiv erweitert."
                    ),
                },
            )
        else:
            state["route"] = "critic_research_exhausted"
            deps.emit(
                LIMIT_REACHED_EVENT,
                {
                    "kind": "replan_rounds",
                    "used": int(state.get("replan_rounds", 0) or 0),
                    "limit": deps.platform.max_replan_rounds,
                    "ceiling": deps.platform.max_replan_rounds,
                    "extendable": False,
                    "recoverable": False,
                    "reason": "deterministic_mission_budget",
                    "state": "continued_with_visible_gap",
                },
            )
            deps.emit(
                ACTIVITY_EVENT,
                {
                    "kind": "critic_research_exhausted",
                    "label": "Recherche-Cap erreicht",
                    "detail": (
                        "Der Critic sieht weiteren Recherchebedarf, aber "
                        "das Replan-Limit ist erreicht."
                    ),
                },
            )
        return state
    if (
        isinstance(report, AgentCriticReport)
        and report.verdict == "revise"
        and state.get("revisions_used", 0) < 1
    ):
        state["revisions_used"] = state.get("revisions_used", 0) + 1
        fixes = "\n".join(
            f"- {finding.detail}: {finding.suggested_fix}"
            for finding in report.findings
        )
        if state.get("deliverable") == "chat":
            # Chat answers revise through the SAME answer call (no memo
            # section structure to demand) — the revised text replaces
            # the answer artifact via the deliverable dispatcher.
            body, usage = synthesis.write_chat_answer(
                deps.llm,
                question=(
                    f"{state['question']}\n\nUeberarbeite deine bisherige "
                    f"Antwort und behebe diese Kritikpunkte:\n{fixes}\n\n"
                    f"Bisherige Antwort:\n{state.get('memo_markdown', '')}"
                ),
                skills_block=_skills_prompt_block(deps, state),
                evidence_digest=evidence.evidence_digest(
                    _ranked_digest_references(
                        deps, state.get("references", [])
                    )
                ),
                contradictions_digest="",
                history=state.get("history", ""),
                prior_memo=state.get("prior_memo", ""),
                user_guidance=state.get("report_guidance", ""),
                model=deps.resolved("agent_answer")[0],
                reasoning_effort=deps.resolved("agent_answer")[1],
                timeout=deps.timeout,
                known_labels=[
                    str(ref.get("label") or "")
                    for ref in state.get("references", [])
                ],
            )
            _add_usage(state, usage)
            if body.strip():
                state["memo_markdown"] = body
                _flush_deliverable(deps, state, body, status="writing")
            else:
                log.warning(
                    "Kritik-Revision fuer Chat-Antwort ohne Inhalt "
                    "verworfen — die bisherige Antwort bleibt bestehen."
                )
            state["critic_recheck_pending"] = True
            return state
        synth_model, synth_effort = deps.resolved("agent_synthesis")
        body, usage = synthesis.write_section(
            deps.llm,
            question=state["question"],
            section_title="Ueberarbeitung",
            section_focus=(
                "Behebe die Kritikpunkte im gesamten Memo und gib das "
                f"VOLLSTAENDIGE ueberarbeitete Memo aus:\n{fixes}"
            ),
            evidence_digest=evidence.evidence_digest(
                _ranked_digest_references(deps, state.get("references", []))
            ),
            contradictions_digest="",
            skills_block=_skills_prompt_block(deps, state),
            user_guidance=state.get("report_guidance", ""),
            model=synth_model,
            reasoning_effort=synth_effort,
            timeout=deps.timeout,
            known_labels=[
                str(ref.get("label") or "")
                for ref in state.get("references", [])
            ],
        )
        _add_usage(state, usage)
        if body.strip() and "## " in body:
            state["memo_markdown"] = body
            _flush_deliverable(deps, state, body, status="writing")
        elif body.strip():
            log.warning(
                "Kritik-Revision ohne Abschnittsstruktur verworfen — "
                "das urspruengliche Memo bleibt bestehen."
            )
        state["critic_recheck_pending"] = True
    return state


def _node_patch(state: AgentPhaseState) -> AgentPhaseState:
    """Propose an editor patch against the assignment's target document.

    Runs only when the request carried a ``document_id``. NO interrupt
    here — the LLM call and the control-store writes must COMMIT before
    the approval node interrupts (the plan/plan_approval split). The
    agent NEVER applies the patch: apply happens solely
    through ``POST /v1/editor/patches/{id}:apply``.
    """
    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    _set_phase(state, "patch")
    run_id = deps.context.run_id or ""
    document_id = state.get("target_document_id", "")
    if deps.editor_patches is None or deps.editor_docs is None:
        # Loud, never silent: the assignment targeted a document but the
        # deployment has no editor persistence/patch surface.
        state["failure"] = "editor_patches_unavailable"
        return state
    from inqtrix.project.editor_ports import DocumentNotFound
    from inqtrix.services.editor_persistence_service import (
        CollaborationProjectionUnavailable,
    )

    try:
        document = _run_async(
            deps.editor_docs.get_document_for_ai(
                document_id, visible_to=deps.visible_to
            )
        )
    except DocumentNotFound:
        state["failure"] = "patch_document_not_found"
        return state
    except CollaborationProjectionUnavailable:
        state["failure"] = "collaboration_projection_unavailable"
        return state

    from inqtrix.agents.patch_phase import (
        PatchProposalFailed,
        propose_patch_edits,
    )

    profile = state.get("profile") or {}
    model, effort = deps.resolved("agent_patch")
    settings = deps.context.agent_settings
    try:
        result, tokens = propose_patch_edits(
            deps.llm,
            question=state.get("question", ""),
            memo_markdown=state.get("memo_markdown", ""),
            document_markdown=document.content_markdown,
            language=str(profile.get("language", "de") or "de"),
            model=model,
            reasoning_effort=effort,
            # The editor budget, not the reasoning one — the SAME window
            # /v1/editor/instruct enforces (decoupled by design).
            timeout=float(
                getattr(settings, "editor_assistant_timeout", deps.timeout)
            ),
        )
    except PatchProposalFailed as exc:
        # HARD failure (never "no changes needed"): the requested
        # document edit did not happen — visible, not smoothed over.
        state["failure"] = str(exc)
        return state
    # Combined prompt+completion count (the instruct pipeline reports one
    # sum); booked under prompt_tokens — one bucket, never dropped.
    _add_usage(
        state, {"prompt_tokens": tokens, "completion_tokens": 0}
    )
    if not result.edits:
        # A genuine zero-edit judgement is a legitimate outcome.
        state["patch_decision"] = "no_changes"
        return state
    principal = deps.context.principal
    patch = _run_async(
        deps.editor_patches.propose(
            document_id=document_id,
            run_id=run_id,
            source="agent",
            edits=[edit.to_payload() for edit in result.edits],
            summary=result.assistant_message,
            warnings=list(result.warnings),
            created_by_user_id=getattr(principal, "user_id", None),
            visible_to=deps.visible_to,
            # Attributes the proposal in the audit trail as an agent write;
            # actor_user_id remains the segment's effective actor.
            principal=principal,
        )
    )
    state["patch_id"] = patch.patch_id
    # E14/R3: the artifact row references the patch id — the edits live
    # exactly once, in the editor_patches row.
    artifact_id = f"art_{run_id[-12:]}_patch"
    _run_async(
        deps.control.upsert_artifact(
            run_id=run_id,
            kind="editor_patch",
            session_id=None,
            title=document.title,
            status="ready",
            content_markdown="",
            payload={"patch_id": patch.patch_id},
            refs=[],
            updated_by="agent",
            artifact_id=artifact_id,
        )
    )
    deps.emit(
        PATCH_PROPOSED_EVENT,
        {
            "patch_id": patch.patch_id,
            "document_id": document_id,
            "artifact_id": artifact_id,
            "edit_count": len(result.edits),
        },
    )
    return state


def _node_patch_approval(state: AgentPhaseState) -> AgentPhaseState:
    """The patch approval interrupt is ALWAYS gated.

    Deliberately no autonomy branch: a write effect interrupts in every
    mode, including ``autonomous``. Rejection ends the run NORMALLY (the
    memo stays the deliverable); only the patch decision is recorded.
    """
    from langgraph.types import interrupt

    deps = _deps()
    if deps.cancelled():
        return {"cancelled": True}
    run_id = deps.context.run_id or ""
    approval_id = f"apr_{run_id[-12:]}_patch_0"
    _ensure_approval(
        deps,
        run_id,
        approval_id,
        kind="patch",
        payload={
            "patch_id": state.get("patch_id", ""),
            "document_id": state.get("target_document_id", ""),
        },
    )
    decision = interrupt({"kind": "approval", "id": approval_id})
    rejected = decision.get("decision") == "reject"
    state["patch_decision"] = "rejected" if rejected else "approved"
    if rejected and deps.editor_patches is not None:
        # The generic M4 decisions endpoint decides only the approval row;
        # the patch row must follow, or the explicitly rejected edits
        # would stay pending and appliable forever. Idempotent — the FE
        # review flow rejects the row first, this replay returns stored.
        from inqtrix.project.editor_patch_ports import PatchAlreadyDecided

        try:
            _run_async(
                deps.editor_patches.reject(
                    state.get("patch_id", ""),
                    note="",
                    visible_to=deps.visible_to,
                )
            )
        except PatchAlreadyDecided:
            pass
    return state


def _node_finalize(state: AgentPhaseState) -> AgentPhaseState:
    deps = _deps()
    if state.get("cancelled"):
        _run_async(
            settle_cancelled_plan_tasks(
                deps.control, deps.context.run_id or ""
            )
        )
    memo = state.get("memo_markdown", "")
    if memo:
        _flush_deliverable(deps, state, memo, status="ready")
        # The evidence artifact remains the complete audit ledger.  Public
        # answer references are only those still cited after deterministic
        # claim repair, so an unknown/attributed-only source cannot re-enter
        # the answer through the final result projection.
        state["references"] = synthesis.cited_references(
            state.get("memo_markdown", ""),
            state.get("references", []),
        )
    critic = state.get("critic")
    if critic is not None:
        run_id = deps.context.run_id or ""
        _run_async(
            deps.control.upsert_artifact(
                run_id=run_id,
                kind="critic_report",
                session_id=None,
                title="Kritik-Bericht",
                status="ready",
                content_markdown="",
                payload=critic,
                refs=[],
                updated_by="agent",
                artifact_id=f"art_{run_id[-12:]}_critic",
            )
        )
    # A gate-rejected run produced no research worth remembering — an
    # LLM reflection over the deterministic receipt would be pure waste
    # (and the draft case adds nothing a completed run would not).
    if not state.get("plan_rejected"):
        _stage_memory_candidates(deps, state)
    _set_phase(state, "done" if not state.get("cancelled") else "cancelled")
    return state


_MEMO_CONFLICT_SEPARATOR = (
    "\n\n---\n\n> Hinweis: Das Memo wurde zwischenzeitlich manuell "
    "bearbeitet. Die folgende Aktualisierung stammt aus diesem Agent-Lauf.\n\n"
)
_MEMO_FLUSH_MAX_ATTEMPTS = 5


def _flush_deliverable(
    deps: "_RunDeps", state: AgentPhaseState, body: str, *, status: str
) -> None:
    """Route the deliverable write to its ONE artifact channel (M1 S3).

    Canvas deliverables stay on the session-memo path with their provenance
    reconcile (E15/R10).  Chat deliverables are deliberately not persisted
    inside graph nodes: the native RunService publisher creates the empty
    ``writing`` answer artifact only when the algorithm has returned its final
    Markdown, then finalizes it immediately before ``answer.ready``.
    """
    if state.get("deliverable") == "chat":
        _ground_mission_output(deps, state, body)
        return
    _flush_memo(deps, state, body, status=status)


def _ground_mission_output(
    deps: "_RunDeps", state: AgentPhaseState, body: str
) -> str:
    """Commit the already synthesized answer without a second content gate."""

    _ = deps
    answer = normalize_agent_markdown(body)
    state["memo_markdown"] = answer
    return answer


def _flush_memo(
    deps: "_RunDeps", state: AgentPhaseState, memo: str, *, status: str
) -> None:
    """Write the memo artifact, reconciling by PROVENANCE, never clobbering.

    The session memo is one row shared across a session's turns, and the
    graph checkpoints only at node boundaries — so on a crash-retry the
    checkpointed ``memo_base_revision`` reverts while the DB row is already
    ahead from this run's own partial section flushes. A blind revision-CAS
    cannot tell that self-authored advance from a genuine user edit. The
    bounded reconcile loop below distinguishes them via ``updated_by``:

    * ``updated_by == 'agent'`` with this ``run_id`` on a conflict = our OWN
      uncommitted partial (retry fast-forward) — adopt its revision and
      re-write, NO user flag;
    * an agent write from another run is a foreign concurrent edit and is
      preserved exactly like a user edit;
    * ``updated_by == 'user'`` = a real edit (E13/R10) — keep that text as
      a prefix, append this run's memo below it, surface the conflict ONCE;
    * ``expected_revision`` is passed verbatim (``0`` on a fresh session)
      so a concurrently-inserted row is caught as a conflict too.
    """
    memo = _ground_mission_output(deps, state, memo)
    run_id = deps.context.run_id or ""
    session_id = state.get("session_id") or None
    resolved_id = (
        state.get("artifact_id")
        or f"art_{(state.get('session_id') or run_id)[-12:]}_memo"
    )

    def _write(content: str, expected: int) -> ArtifactRecord:
        cited_refs = synthesis.cited_references(
            content, state.get("references", [])
        )
        return _run_async(
            deps.control.upsert_artifact(
                run_id=run_id,
                kind="memo",
                session_id=session_id,
                title=state.get("memo_title", "Memo"),
                status=status,
                content_markdown=content,
                payload={},
                refs=cited_refs,
                updated_by="agent",
                artifact_id=resolved_id,
                expected_revision=expected,
            )
        )

    base = state.get("memo_base_revision", 0)
    prefix = state.get("memo_user_prefix", "")
    record: ArtifactRecord | None = None
    for _ in range(_MEMO_FLUSH_MAX_ATTEMPTS):
        body = memo if not prefix else prefix + _MEMO_CONFLICT_SEPARATOR + memo
        try:
            record = _write(body, base)
            break
        except ArtifactRevisionConflict:
            # Re-read the conflicting row by the SAME key the write used:
            # the session-memo lineage key when this run belongs to a
            # session, else the run-scoped artifact_id (a session-less run
            # writes ``resolved_id`` with ``session_id=None``, which
            # get_session_artifact — filtering on session_id equality —
            # can never match, so a by-id read is the only witness).
            current: ArtifactRecord | None = None
            if session_id:
                current = _run_async(
                    deps.control.get_session_artifact(session_id, "memo")
                )
            if current is None:
                try:
                    current, _ = _run_async(
                        deps.control.get_artifact(run_id, resolved_id)
                    )
                except ArtifactNotFound:
                    current = None
            if current is None:
                base = 0  # vanished between attempts — recreate
                continue
            if current.updated_by == "agent" and current.run_id == run_id:
                # Our own partial from a crashed attempt — adopt, not a
                # user edit; the re-run's cumulative memo overwrites it.
                base = current.revision
                # If that partial already embedded a preserved user edit
                # (before the first separator), recover it so the retry
                # keeps appending BELOW the user's text — checkpoint state
                # reverted on the crash, the DB row is the only witness.
                if (
                    not prefix
                    and _MEMO_CONFLICT_SEPARATOR in current.content_markdown
                ):
                    prefix = current.content_markdown.split(
                        _MEMO_CONFLICT_SEPARATOR, 1
                    )[0]
                    state["memo_user_prefix"] = prefix
                continue
            # A genuine user edit: preserve it once, then keep appending
            # below it on every subsequent flush of this run.
            if not prefix:
                prefix = current.content_markdown
                state["memo_user_prefix"] = prefix
                deps.emit(
                    ARTIFACT_EDIT_CONFLICT_EVENT,
                    {"artifact_id": resolved_id, "kind": "memo"},
                )
            base = current.revision
            continue
    if record is None:
        # Pathological churn (a user editing on every retry) — fail loudly,
        # never a silent corrupt write (Prinzip 1).
        raise RuntimeError("memo_flush_conflict_unresolved")

    first = not state.get("artifact_id")
    state["artifact_id"] = record.artifact_id
    state["memo_base_revision"] = record.revision
    deps.emit(
        (
            ARTIFACT_CREATED_EVENT
            if first and record.revision == 1
            else ARTIFACT_UPDATED_EVENT
        ),
        {
            "artifact_id": record.artifact_id,
            "kind": "memo",
            "revision": record.revision,
            "updated_by": "agent",
        },
    )


def _stage_memory_candidates(
    deps: "_RunDeps", state: AgentPhaseState
) -> None:
    """Generate candidate-only long-term memories after a finished run."""
    service = deps.memory
    principal = deps.context.principal
    if service is None or principal is None or state.get("cancelled"):
        return
    if not deps.agent_memory_opt_in:
        # No opt-in -> stage nothing (same per-user privacy gate as the read
        # path in _load_memory_briefing).
        return
    status = service.status(principal)
    if (
        status.get("provider") == "none"
        or status.get("mode") == "off"
        or not status.get("principal_eligible")
    ):
        return
    memo = state.get("memo_markdown", "")
    if not memo.strip():
        return
    model, effort = deps.resolved("agent_memory_reflection")
    try:
        outcome = memory_reflection.run_memory_reflection(
            deps.llm,
            question=state.get("question", ""),
            memo_markdown=memo,
            critic_digest=_critic_digest(state),
            task_digest=_task_digest(state),
            model=model,
            reasoning_effort=effort,
            timeout=deps.timeout,
        )
    except Exception as exc:  # noqa: BLE001 - candidate generation is optional
        log.warning(
            "Agent memory reflection failed (error_type=%s).",
            type(exc).__name__,
        )
        state["memory_status"] = "unavailable"
        deps.emit(
            ACTIVITY_EVENT,
            {
                "kind": "memory_unavailable",
                "label": "Memory unavailable",
                "detail": "Memory-Kandidaten konnten nicht erzeugt werden.",
            },
        )
        return
    _add_usage(state, outcome.usage)
    reflection = outcome.value
    if not isinstance(reflection, MemoryReflection):
        return
    staged = _run_async(
        service.stage_candidates(
            principal=principal,
            candidates=[
                candidate.model_dump() for candidate in reflection.candidates
            ],
            source_run_id=deps.context.run_id or "",
        )
    )
    state["memory_candidates"] = [
        {
            "id": candidate.candidate_id,
            "scope": candidate.scope,
            "category": candidate.category,
            "content": candidate.content,
            "reason": candidate.reason,
            "confidence": candidate.confidence,
            "status": candidate.status,
        }
        for candidate in staged
    ]
    if staged:
        state["memory_status"] = "candidate_created"
        deps.emit(
            ACTIVITY_EVENT,
            {
                "kind": "memory_candidate",
                "label": "Memory-Kandidat erzeugt",
                "count": len(staged),
                "detail": "Der Kandidat wartet auf Nutzerfreigabe.",
            },
        )


def _critic_digest(state: AgentPhaseState) -> str:
    critic = state.get("critic") or {}
    if not critic:
        return "(kein Critic-Bericht)"
    findings = [
        f"- {finding.get('kind', '')}: {finding.get('detail', '')}"
        for finding in critic.get("findings", [])
    ]
    return "\n".join(
        [
            f"Verdict: {critic.get('verdict', '')}",
            "Uncovered: "
            + ", ".join(str(item) for item in critic.get("criteria_uncovered", [])),
            *findings[:8],
        ]
    )


def _task_digest(state: AgentPhaseState) -> str:
    outcomes = state.get("outcomes", {})
    lines: list[str] = []
    for task_id, outcome in sorted(outcomes.items()):
        lines.append(
            f"- {task_id}: {outcome.get('status', '')}; "
            f"{str(outcome.get('summary', ''))[:300]}"
        )
    return "\n".join(lines) or "(keine Task-Ergebnisse)"


# -- task execution ----------------------------------------------------------- #


def _task_failure_code(exc: Exception) -> str:
    """Map one task exception onto a stable failure contract."""
    return classify_execution_failure(exc, fallback="task_failed")


def _retryable_task_error(exc: Exception) -> bool:
    """Whether *exc* failed before a provider-owned operation ran."""
    return (
        _task_failure_code(exc)
        in RETRYABLE_AGENT_TASK_ORCHESTRATION_CODES
    )


def _emit_legacy_budget_notices(
    deps: "_RunDeps",
    state: AgentPhaseState,
    tasks: list[PlanTaskRecord],
) -> None:
    """Expose each ignored historic task budget once per checkpointed run."""
    emitted = set(state.get("legacy_budget_notice_task_ids", []))
    for task in tasks:
        if not task.budget or task.task_id in emitted:
            continue
        log.warning(
            "Agent-Task %s enthaelt ein veraltetes Task-Budget; die Werte "
            "werden ignoriert, Operatorgrenzen bleiben autoritativ.",
            task.task_id,
        )
        deps.emit(
            ACTIVITY_EVENT,
            {
                "activity_id": f"legacy-budget:{task.task_id}",
                "scope": "task",
                "phase": "execution",
                "operation": "task.legacy_budget_ignored",
                "detail": "Veraltetes Task-Budget wird ignoriert",
                "status": "completed",
                "task_id": task.task_id,
                "fallback": True,
            },
        )
        emitted.add(task.task_id)
    state["legacy_budget_notice_task_ids"] = sorted(emitted)


def _transition_task(
    deps: "_RunDeps",
    task: PlanTaskRecord,
    *,
    status: str,
    child_run_id: str | None = None,
    result_summary: str | None = None,
    result_payload: dict[str, Any] | None = None,
) -> PlanTaskRecord:
    """Persist one authoritative task-state transition before UI events."""
    return _run_async(
        deps.control.transition_plan_task(
            run_id=task.run_id,
            plan_id=task.plan_id,
            task_id=task.task_id,
            status=status,
            child_run_id=child_run_id,
            result_summary=result_summary,
            result_payload=result_payload,
        )
    )


def _execute_task(
    deps: "_RunDeps",
    state: AgentPhaseState,
    task: PlanTaskRecord,
    attempt: int,
) -> TaskOutcome:
    from inqtrix.knowledge.stores.ports import CollectionNotFound

    deps.emit(
        TASK_STARTED_EVENT,
        {
            "task_id": task.task_id,
            "ordinal": task.ordinal,
            "tool_kind": task.tool_kind,
            "attempt": attempt,
        },
    )
    try:
        require_task_allowed(
            task.tool_kind, policy=coerce_source_policy(deps)
        )
        # web_research never reaches this executor: _node_execute
        # partitions child tasks out and submits them as parked child
        # runs (children_wait folds their outcomes).
        if task.tool_kind == "web_instant":
            return _run_web_instant(deps, task)
        if task.tool_kind == "rag_query":
            return _run_rag_query(deps, state, task, attempt)
        if task.tool_kind == "file_analysis":
            return _run_file_analysis(deps, task)
        raise ValueError(f"unknown tool_kind {task.tool_kind!r}")
    except AgentCancelled:
        raise
    except CollectionNotFound as exc:
        # A plan reference the store cannot resolve is a CONTRACT error,
        # not a flaky call: retrying re-fails identically and burns more
        # operator quota. Non-transient, with a message instead of the
        # raw KeyError repr ("'EU-AI-Act'").
        reason = (
            "Sammlung nicht sichtbar oder unbekannt: "
            f"{exc.args[0] if exc.args else exc}"
        )
        log.warning(
            "Agent-Task %s (%s) fehlgeschlagen (failure_code=invalid_input).",
            task.task_id,
            task.tool_kind,
        )
        return TaskOutcome(
            status="failed",
            failure_reason=reason,
            failure_code="invalid_input",
            transient=False,
        )
    except Exception as exc:  # noqa: BLE001 — task failures stay visible
        failure_code = _task_failure_code(exc)
        log.warning(
            "Agent-Task %s (%s) fehlgeschlagen "
            "(failure_code=%s, error_type=%s).",
            task.task_id,
            task.tool_kind,
            failure_code,
            type(exc).__name__,
        )
        return TaskOutcome(
            status="failed",
            failure_reason=sanitize_error(exc),
            failure_code=failure_code,
            transient=attempt == 1 and _retryable_task_error(exc),
        )


def _submit_child_run(
    deps: "_RunDeps",
    state: AgentPhaseState,
    task: PlanTaskRecord,
    attempt: int,
    *,
    research_policy: WebResearchPolicy,
) -> str:
    """Submit one child research run; returns its run id.

    The child is a REAL run in the shared pool (kind ``agent_child``,
    parented to this run) — the parent does NOT wait here: it parks via
    the children_wait interrupt and the store wakes it when the last
    child terminates. A retry uses the same operator-owned limits. The
    child's own research-graph deadline and the parent's waiting TTL are
    the only time bounds.
    """
    require_task_allowed(
        task.tool_kind, policy=coerce_source_policy(deps)
    )
    if not research_policy.allowed:
        raise AgentPolicyDenied(
            "web_research ist in einem normalen Agent-Desk-Lauf nur nach "
            "einer ausdruecklichen Recherche-Anweisung oder Planbearbeitung "
            "erlaubt."
        )
    existing = _existing_child_for_attempt(deps, task, attempt)
    if existing is not None:
        return str(existing["run_id"])
    profile = research_policy.profile
    if profile is None:
        raise RuntimeError("permitted web research has no profile")
    requested_profile = str(task.params.get("profile") or "")
    ceiling = research_policy.max_profile
    if (
        ceiling is not None
        and requested_profile in WEB_RESEARCH_PROFILE_ORDER
        and WEB_RESEARCH_PROFILE_ORDER[requested_profile]
        <= WEB_RESEARCH_PROFILE_ORDER[ceiling]
    ):
        # Tier runs: the validator already admitted any profile up to the
        # tier ceiling — execution must honor that choice (published ==
        # enforced), not silently normalize it back to the tier default.
        profile = requested_profile
    if requested_profile and requested_profile != profile:
        log.warning(
            "Agent-Task %s enthaelt das veraltete Research-Profil %r; "
            "serverseitig wird %r verwendet.",
            task.task_id,
            requested_profile,
            profile,
        )
        deps.emit(
            ACTIVITY_EVENT,
            {
                "scope": "task",
                "phase": "execution",
                "operation": "task.research_profile_normalized",
                "detail": f"Research-Profil auf {profile} normalisiert",
                "status": "completed",
                "task_id": task.task_id,
                "fallback": True,
            },
        )
    overrides: dict[str, Any] = {"report_profile": profile}
    if task.params.get("model_tier"):
        overrides["model_tier"] = task.params["model_tier"]
    resolve_payload: dict[str, Any] = {
        "mode": "research",
        "agent_overrides": overrides,
    }
    parent_stack = getattr(deps.context, "stack_name", "") or ""
    if parent_stack:
        # Parity with the kernel (F7c): a mission admitted on stack X
        # fans its research children out on the SAME provider stack, not
        # the default stack's models/search.
        resolve_payload["stack"] = parent_stack
    resolved_child = deps.resolver.resolve(resolve_payload)
    question = _task_execution_question(task)
    run_id = deps.context.run_id or ""
    summary = deps.run_service.submit(
        question=question or task.title,
        history="",
        messages=[],
        resolved=resolved_child,
        workspace_id=deps.context.workspace_id,
        principal=deps.context.principal,
        kind="agent_child",
        parent_run_id=run_id,
        root_run_id=run_id,
        session_id=state.get("session_id") or None,
        parent_task_id=task.task_id,
        parent_task_attempt=attempt,
        origin_key=_child_origin_key(task, attempt),
        source_policy={
            "web": deps.source_policy.web,
            "knowledge": deps.source_policy.knowledge,
        },
        web_recency=task.params.get("recency") or None,
    )
    return str(summary["run_id"])


def _child_task_outcome(
    deps: "_RunDeps", child_id: str, attempt: int
) -> TaskOutcome | None:
    """Compatibility wrapper around the shared pure child projector."""
    return project_child_run_outcome(
        deps.run_service.run_store,
        child_id,
        attempt,
        visible_to=getattr(deps, "visible_to", None),
    )


def _run_web_instant(deps: "_RunDeps", task: PlanTaskRecord) -> TaskOutcome:
    if deps.capabilities is None:
        return TaskOutcome(
            status="failed",
            failure_reason="web.search.instant nicht verfuegbar",
            failure_code="capability_unavailable",
        )
    query = task.queries[0] if task.queries else task.objective or task.title
    payload: dict[str, Any] = {"query": query, "max_sources": 8}
    if task.params.get("recency"):
        payload["recency"] = task.params["recency"]
    activity = {
        "kind": "searching",
        "scope": "task",
        "phase": "execution",
        "operation": "web.search.instant",
        "detail": query,
        "status": "started",
        "current": 1,
        "total": 1,
        "task_id": task.task_id,
        "query": query,
        "purpose": task.objective or task.title,
    }
    deps.emit(ACTIVITY_EVENT, activity)
    try:
        output = _run_async(
            deps.capabilities.invoke(
                "web.search.instant",
                payload,
                _capability_context(
                    deps,
                    on_provider_retry=lambda notice: deps.emit(
                        ACTIVITY_EVENT,
                        provider_retry_activity(
                            notice,
                            task_id=task.task_id,
                            purpose=task.objective or task.title,
                        ),
                    ),
                ),
            )
        )
    except Exception as exc:
        deps.emit(
            ACTIVITY_EVENT,
            {
                **activity,
                "status": "failed",
                "error": {
                    "code": _task_failure_code(exc),
                    "message": sanitize_error(exc),
                },
            },
        )
        raise
    data = output.model_dump() if hasattr(output, "model_dump") else output
    provider_answer = str(data.get("answer", "")).strip()
    answer = normalize_agent_markdown(provider_answer)
    sources = list(data.get("sources", []))
    web_search_ledger = build_instant_web_search_ledger(
        run_id=str(deps.context.run_id or ""),
        query_id=str(data.get("query_id") or ""),
        query=str(data.get("query") or query),
        provider=str(data.get("provider") or ""),
        answer=provider_answer,
        sources=[
            dict(source) for source in sources if isinstance(source, dict)
        ],
        parameters=(
            dict(data.get("parameters"))
            if isinstance(data.get("parameters"), dict)
            else {}
        ),
        started_at=str(data.get("started_at") or ""),
        finished_at=str(data.get("finished_at") or ""),
        prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
        completion_tokens=int(data.get("completion_tokens", 0) or 0),
    )
    usage = {
        "prompt_tokens": int(data.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(data.get("completion_tokens", 0) or 0),
    }
    deps.emit(
        ACTIVITY_EVENT,
        {
            **activity,
            "status": "completed",
            "metrics": {
                "result_count": len(sources),
                **usage,
            },
        },
    )
    instant_evidence = evidence.enrich_instant_evidence(
        provider_answer,
        [
            {
                "url": source.get("url"),
                "title": source.get("title"),
                "provider_snippet": source.get("snippet"),
            }
            for source in sources
            if isinstance(source, dict)
        ],
    )
    instant_evidence = attach_web_search_lineage(
        instant_evidence, web_search_ledger
    )
    return TaskOutcome(
        status="completed" if answer else "insufficient_evidence",
        summary=task_result_summary(
            answer
            or "; ".join(
                str(s.get("snippet", "")) for s in sources[:3]
            )
        ),
        answer_markdown=answer,
        evidence=instant_evidence,
        web_search_ledger=web_search_ledger,
        usage=usage,
    )


def _run_rag_query(
    deps: "_RunDeps",
    state: AgentPhaseState,
    task: PlanTaskRecord,
    attempt: int,
) -> TaskOutcome:
    from inqtrix.core.context import RunContext
    from inqtrix.core.results import RunRequest

    try:
        knowledge = deps.runtime.registry.get("knowledge")
    except Exception as exc:  # noqa: BLE001 — mode not registered
        return TaskOutcome(
            status="failed",
            failure_reason=(
                "knowledge-Modus nicht verfuegbar: "
                f"{sanitize_error(exc)}"
            ),
            failure_code="capability_unavailable",
        )
    filters: dict[str, Any] = {}
    if task.params.get("profile"):
        filters["profile"] = task.params["profile"]
    filters["collection_ids"] = _task_collection_scope(deps, task)
    queries = _task_queries(task)
    answers: list[str] = []
    references_by_key: dict[str, dict[str, Any]] = {}
    retrieval_warning_messages: list[str] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    active_query_index = 0

    def _forward_knowledge_event(
        event: str, payload: dict[str, Any]
    ) -> None:
        """Preserve nested Knowledge verdicts in mission/parent audit."""

        if event not in {
            "inqtrix.knowledge.grounding.checked",
            "inqtrix.knowledge.retrieval.degraded",
            "inqtrix.knowledge.retrieval.warning",
        }:
            return
        deps.emit(
            event,
            {
                **dict(payload),
                "task_id": task.task_id,
                "attempt": attempt,
                "query_index": active_query_index,
            },
        )

    sub_context = RunContext(
        providers=deps.context.providers,
        strategies=deps.context.strategies,
        agent_settings=deps.context.agent_settings,
        principal=deps.context.principal,
        workspace_id=deps.context.workspace_id,
        run_id=None,
        cancel_token=deps.context.cancel_token,
        event_sink=_forward_knowledge_event,
        token_budget=int(deps.context.token_budget or 0),
    )
    for index, query in enumerate(queries, start=1):
        active_query_index = index
        activity = {
            "kind": "searching",
            "scope": "task",
            "phase": "execution",
            "operation": "knowledge.search",
            "detail": query,
            "status": "started",
            "current": index,
            "total": len(queries),
            "task_id": task.task_id,
            "query": query,
            "purpose": task.objective or task.title,
        }
        deps.emit(ACTIVITY_EVENT, activity)
        try:
            result = knowledge.run(
                RunRequest(
                    mode="knowledge",
                    question=_task_query_prompt(task, query),
                    knowledge_filters=filters,
                ),
                runtime=deps.runtime,
                context=sub_context,
            )
        except Exception as exc:
            deps.emit(
                ACTIVITY_EVENT,
                {
                    **activity,
                    "status": "failed",
                    "error": {
                        "code": _task_failure_code(exc),
                        "message": sanitize_error(exc),
                    },
                },
            )
            raise
        raw_state = result.raw.get("result_state", {}) or {}
        retrieval_state = raw_state.get("knowledge_retrieval", {}) or {}
        retrieval_degradations = list(
            retrieval_state.get("degradations", []) or []
        )
        retrieval_warnings = list(retrieval_state.get("warnings", []) or [])
        for degradation in retrieval_degradations:
            if not isinstance(degradation, dict):
                continue
            message = (
                "Die Knowledge-Suche erreichte eine technische "
                "Kandidatengrenze und lieferte "
                f"{int(degradation.get('returned_hits', 0) or 0)}/"
                f"{int(degradation.get('requested_top_k', 0) or 0)} "
                "angeforderte verifizierte Treffer; das Ergebnis ist kein "
                "Vollständigkeitsnachweis."
            )
            if message not in retrieval_warning_messages:
                retrieval_warning_messages.append(message)
        for warning in retrieval_warnings:
            if not isinstance(warning, dict):
                continue
            count = int(warning.get("count", 0) or 0)
            code = str(warning.get("code", "") or "").strip()
            rendered = (
                "Die Knowledge-Suche schloss "
                f"{count if count else 'einzelne'} Treffer durch eine "
                "Integritätsprüfung aus"
                f"{f' ({code})' if code else ''}."
            )
            if rendered not in retrieval_warning_messages:
                retrieval_warning_messages.append(rendered)
        usage = result.raw.get("usage", {}) or {}
        usage_total["prompt_tokens"] += int(
            usage.get("prompt_tokens", 0) or 0
        )
        usage_total["completion_tokens"] += int(
            usage.get("completion_tokens", 0) or 0
        )
        terminal_failure = result.terminal_failure
        if terminal_failure is not None:
            # A nested Knowledge run retains its token usage and safe failure
            # explanation, but neither its rejected completion nor a partial
            # multi-query result may enter mission synthesis.  Stop this task
            # at the first terminal verdict and persist the same stable code.
            deps.emit(
                ACTIVITY_EVENT,
                {
                    **activity,
                    "status": "failed",
                    "error": {
                        "code": terminal_failure.type,
                        "message": terminal_failure.message,
                    },
                },
            )
            return TaskOutcome(
                status="failed",
                summary=task_result_summary(terminal_failure.message),
                failure_reason=terminal_failure.message,
                failure_code=terminal_failure.type,
                usage=usage_total,
                transient=False,
            )
        if result.answer:
            answers.append(result.answer)
        for ref in raw_state.get("report_references", []):
            if not isinstance(ref, dict):
                continue
            normalized = dict(ref)
            if not normalized.get("excerpt") and normalized.get("source_text"):
                normalized["excerpt"] = normalized["source_text"]
            key = str(
                normalized.get("url")
                or (
                    normalized.get("document_id"),
                    normalized.get("chunk_index"),
                )
            )
            references_by_key.setdefault(key, normalized)
        deps.emit(
            ACTIVITY_EVENT,
            {
                **activity,
                "status": "completed",
                "metrics": {
                    "result_count": len(
                        raw_state.get("report_references", []) or []
                    )
                },
                "warnings": [*retrieval_degradations, *retrieval_warnings],
            },
        )
    references = list(references_by_key.values())
    answer = "\n\n".join(answers)
    if retrieval_warning_messages:
        answer = (
            answer
            + "\n\nRetrieval-Hinweis:\n"
            + "\n".join(f"- {message}" for message in retrieval_warning_messages)
        ).strip()
    return TaskOutcome(
        status=(
            "completed"
            if answer.strip() and references
            else "insufficient_evidence"
        ),
        summary=task_result_summary(answer),
        answer_markdown=normalize_agent_markdown(answer),
        evidence=references,
        usage=usage_total,
    )


def _run_file_analysis(deps: "_RunDeps", task: PlanTaskRecord) -> TaskOutcome:
    if deps.capabilities is None:
        return TaskOutcome(
            status="failed",
            failure_reason="Datei-Capabilities nicht verfuegbar",
            failure_code="capability_unavailable",
        )
    hits_by_key: dict[str, dict[str, Any]] = {}
    retrieval_warning_messages: list[str] = []
    queries = _task_queries(task)
    for index, query in enumerate(queries, start=1):
        activity = {
            "kind": "searching",
            "scope": "task",
            "phase": "execution",
            "operation": "knowledge.search",
            "detail": query,
            "status": "started",
            "current": index,
            "total": len(queries),
            "task_id": task.task_id,
            "query": query,
            "purpose": task.objective or task.title,
        }
        deps.emit(ACTIVITY_EVENT, activity)
        try:
            output = _run_async(
                deps.capabilities.invoke(
                    "knowledge.search",
                    {
                        "query": _task_query_prompt(task, query),
                        "top_k": 8,
                        "collection_ids": _task_collection_scope(deps, task),
                    },
                    _capability_context(deps),
                )
            )
        except Exception as exc:
            deps.emit(
                ACTIVITY_EVENT,
                {
                    **activity,
                    "status": "failed",
                    "error": {
                        "code": _task_failure_code(exc),
                        "message": sanitize_error(exc),
                    },
                },
            )
            raise
        data = output.model_dump() if hasattr(output, "model_dump") else output
        query_hits = list(data.get("hits") or data.get("results") or [])
        query_warnings = [
            warning
            for warning in list(data.get("warnings") or [])
            if isinstance(warning, dict)
        ]
        for warning in query_warnings:
            message = str(warning.get("message") or "").strip()
            if message and message not in retrieval_warning_messages:
                retrieval_warning_messages.append(message)
        for hit in query_hits:
            key = str(
                (
                    hit.get("chunk_id"),
                    hit.get("document_id"),
                    hit.get("chunk_index"),
                )
            )
            hits_by_key.setdefault(key, hit)
        deps.emit(
            ACTIVITY_EVENT,
            {
                **activity,
                "status": "completed",
                "metrics": {"result_count": len(query_hits)},
                "warnings": query_warnings,
            },
        )
    hits = list(hits_by_key.values())
    # Knowledge capabilities deliberately expose only canonical excerpts.
    # Never revive the former ``text`` fallback here: that field can be the
    # embedding-only text with a model-generated retrieval prefix.
    content = "\n\n".join(str(hit.get("excerpt", "")) for hit in hits)
    if not content.strip():
        return TaskOutcome(
            status="insufficient_evidence",
            summary="Keine Dokumentinhalte gefunden.",
        )
    model, effort = deps.resolved("agent_file_analysis")
    summary, usage = run_quarantined_file_analysis(
        deps.llm,
        objective=_task_execution_question(task),
        content=content,
        model=model,
        reasoning_effort=effort,
        timeout=deps.timeout,
    )
    if summary is None:
        return TaskOutcome(
            status="failed",
            failure_reason="Analyse-Antwort nicht auswertbar",
            failure_code="parse_error",
            usage=usage,
            transient=False,
        )
    warning_note = (
        "\n\nRetrieval-Hinweis: "
        + " ".join(retrieval_warning_messages)
        + " Die Trefferliste ist kein Vollständigkeitsnachweis."
        if retrieval_warning_messages
        else ""
    )
    answer_markdown = normalize_agent_markdown(summary.summary + warning_note)
    return TaskOutcome(
        status="completed",
        summary=task_result_summary(summary.summary + warning_note),
        answer_markdown=answer_markdown,
        evidence=[
            {
                **dict(hit),
                "excerpt": hit.get("excerpt"),
            }
            for hit in hits
        ],
        usage=usage,
    )


# -- helpers ------------------------------------------------------------------ #


def _task_queries(task: PlanTaskRecord) -> list[str]:
    """Queries a task should actually execute, never just the first one."""
    queries = [query.strip() for query in task.queries if query.strip()]
    return queries or [task.objective or task.title]


def _task_execution_question(task: PlanTaskRecord) -> str:
    """Full task contract passed to child agents and analytical tools."""
    lines = [
        f"Task: {task.title}",
        f"Objective: {task.objective or task.title}",
    ]
    queries = _task_queries(task)
    if queries:
        lines.append("Queries:")
        lines.extend(f"- {query}" for query in queries)
    if task.expected_output:
        lines.append(f"Expected output: {task.expected_output}")
    if task.is_falsification:
        lines.append(
            "This is a falsification task: actively seek counter-evidence."
        )
    return "\n".join(lines)


def _task_query_prompt(task: PlanTaskRecord, query: str) -> str:
    """One retrieval query framed by the full task contract."""
    prompt = _task_execution_question(task)
    return f"{prompt}\n\nCurrent query: {query}"


def _task_collection_scope(
    deps: "_RunDeps", task: PlanTaskRecord
) -> list[str]:
    """Visible collection ids for internal retrieval tasks."""
    explicit = [
        str(item).strip()
        for item in (task.params.get("collection_ids") or [])
        if str(item).strip()
    ]
    if explicit:
        return deps.assert_collections(explicit)
    return deps.collection_scope()


def _collection_catalog(
    deps: "_RunDeps",
) -> list[CollectionCatalogEntry] | None:
    """Currently visible metadata inside the immutable run boundary.

    ``None`` (no knowledge service wired) tells the planner path to skip
    catalog handling entirely; an EMPTY list is a real answer ("this
    run admitted no collections") and makes every explicit reference a
    validation error. Newly visible collections never enter a running plan.
    """
    if deps.source_policy.knowledge != "available":
        return []
    if deps.knowledge is None:
        return None
    admitted = set(deps.collection_scope())
    if not admitted:
        return []
    collections = _run_async(
        deps.knowledge.list_collections(visible_to=deps.visible_to)
    )
    return [
        CollectionCatalogEntry(
            collection_id=str(item.id),
            name=str(item.name),
            embedding_model=str(getattr(item, "embedding_model", "") or ""),
            document_count=int(getattr(item, "document_count", 0) or 0),
        )
        for item in collections
        if str(item.id) in admitted
    ]


def _ensure_approval(
    deps: "_RunDeps",
    run_id: str,
    approval_id: str,
    *,
    kind: str,
    payload: dict[str, Any],
) -> ApprovalRecord:
    """Idempotent approval creation (interrupt nodes re-execute)."""
    from inqtrix.agents.control_ports import ApprovalNotFound

    try:
        return _run_async(deps.control.get_approval(run_id, approval_id))
    except ApprovalNotFound:
        record = _run_async(
            deps.control.create_approval(
                ApprovalRecord(
                    approval_id=approval_id,
                    run_id=run_id,
                    kind=kind,
                    subject_type=(
                        "plan"
                        if kind in ("plan", "replan")
                        else "editor_patch" if kind == "patch" else ""
                    ),
                    subject_id=(
                        str(payload.get("patch_id", ""))
                        if kind == "patch"
                        else str(payload.get("plan_id", ""))
                        if kind in ("plan", "replan")
                        else ""
                    ),
                    payload=payload,
                )
            )
        )
        deps.emit(
            APPROVAL_REQUESTED_EVENT,
            {
                "approval_id": record.approval_id,
                "kind": kind,
                "subject_type": record.subject_type,
                "subject_id": record.subject_id,
            },
        )
        return record


def _capability_context(
    deps: "_RunDeps",
    *,
    on_provider_retry: Callable[[dict[str, object]], None] | None = None,
) -> Any:
    from inqtrix.capabilities import CapabilityContext

    return CapabilityContext(
        principal=deps.context.principal,
        visible_to=deps.visible_to,
        workspace_id=deps.context.workspace_id,
        run_id=deps.context.run_id,
        knowledge_collection_ids=deps.knowledge_collection_ids,
        search_provider=deps.context.providers.search,
        authority_check=getattr(deps.context, "authority_check", None),
        on_provider_retry=on_provider_retry,
    )


def _result_count(output: Any) -> int:
    """Count the primary result rows of a capability output."""
    data = output.model_dump() if hasattr(output, "model_dump") else output
    if not isinstance(data, dict):
        return 0
    for key in ("sources", "hits", "results", "collections"):
        rows = data.get(key)
        if isinstance(rows, list):
            return len(rows)
    return 0


def _profile_of(state: AgentPhaseState) -> AssignmentProfile | None:
    raw = state.get("profile")
    return AssignmentProfile.model_validate(raw) if raw else None


def _discovery_of(state: AgentPhaseState) -> DiscoveryResult | None:
    raw = state.get("discovery")
    return DiscoveryResult.model_validate(raw) if raw else None


def _web_allowed(deps: "_RunDeps") -> bool:
    return (
        deps.capabilities is not None
        and getattr(deps.context.providers, "search", None) is not None
        and deps.source_policy.web == "available"
    )


def _plan_lineage_web_research_consent(
    deps: "_RunDeps",
    state: AgentPhaseState,
    *,
    current_plan: PlanRecord | None = None,
    current_tasks: list[PlanTaskRecord] | None = None,
) -> bool:
    """Recover explicit user consent for compact research from plan history.

    A plan's ``created_by`` describes authorship, not an execution policy. The
    checkpoint keeps the already-established decision on the hot path, while
    persisted user-authored plan versions remain the recovery source after a
    deployment or an older checkpoint. Consent is granted only when the user
    version actually contains a ``web_research`` task.
    """
    run_id = deps.context.run_id or ""

    def _user_selected_research(
        plan: PlanRecord, tasks: list[PlanTaskRecord]
    ) -> bool:
        return plan.created_by == "user" and any(
            task.tool_kind == "web_research" for task in tasks
        )

    if current_plan is not None and _user_selected_research(
        current_plan, list(current_tasks or [])
    ):
        state["web_research_consent"] = True
        state["web_research_consent_checked_version"] = current_plan.version
        return True
    if state.get("web_research_consent", False):
        return True

    versions = _run_async(deps.control.list_plan_versions(run_id))
    newest_version = max((plan.version for plan in versions), default=0)
    if int(state.get("web_research_consent_checked_version", 0) or 0) >= newest_version:
        return False
    for plan in versions:
        if plan.created_by != "user":
            continue
        _stored, tasks = _run_async(
            deps.control.get_plan(run_id, version=plan.version)
        )
        if _user_selected_research(plan, tasks):
            state["web_research_consent"] = True
            state["web_research_consent_checked_version"] = newest_version
            return True
    state["web_research_consent"] = False
    state["web_research_consent_checked_version"] = newest_version
    return False


def _research_child_policy(
    deps: "_RunDeps", *, edit_consent: bool
) -> WebResearchPolicy:
    """Server authority for a multi-step mission research child."""
    return derive_web_research_policy(
        depth=deps.depth,
        admitted_directive=(
            "web_research" in deps.request.tool_directives or edit_consent
        ),
        tier=deps.tier,
    )


def _build_replan_context(state: AgentPhaseState) -> str:
    """Compact additive-replan context for the planner."""
    if state.get("replan_rounds", 0) <= 0:
        return ""
    lines: list[str] = []
    outcomes = state.get("outcomes", {})
    if outcomes:
        lines.append("Bisherige Task-Outcomes:")
        for task_id, outcome in sorted(outcomes.items()):
            status = outcome.get("status", "")
            failure = outcome.get("failure_reason", "")
            summary = str(outcome.get("summary", ""))[:400]
            detail = f" - {task_id}: {status}"
            if failure:
                detail += f"; failure={failure}"
            if summary:
                detail += f"; summary={summary}"
            lines.append(detail)
    sufficiency = state.get("sufficiency") or {}
    missing = sufficiency.get("missing") or []
    if missing:
        lines.append("Sufficiency-Luecken:")
        lines.extend(f" - {item}" for item in missing)
    contradictions = state.get("contradictions", [])
    if contradictions:
        lines.append("Bekannte Widersprueche:")
        for item in contradictions[:5]:
            lines.append(
                " - "
                + str(item.get("internal_position", ""))[:180]
                + " VS "
                + str(item.get("external_position", ""))[:180]
            )
    critic = state.get("critic") or {}
    uncovered = critic.get("criteria_uncovered") or []
    if uncovered:
        lines.append("Vom Critic offene Erfolgskriterien:")
        lines.extend(f" - {item}" for item in uncovered)
    if state.get("route") == "critic_research":
        lines.append(
            "Critic-Entscheid: neue Recherche erforderlich; additiv planen."
        )
    done = sorted(outcomes)
    if done:
        lines.append(
            "Bereits erledigte Task-IDs nicht erneut planen: "
            + ", ".join(done)
        )
    return "\n".join(lines)


# -- graph wiring --------------------------------------------------------------- #


def _build_graph(saver: Any) -> Any:
    from langgraph.graph import END, StateGraph

    graph = StateGraph(AgentPhaseState)
    graph.add_node("intake", _node_intake)
    graph.add_node("clarify", _node_clarify)
    graph.add_node("discovery", _node_discovery)
    graph.add_node("clarify_gaps", _node_clarify)
    graph.add_node("plan", _node_plan)
    graph.add_node("plan_approval", _node_plan_approval)
    graph.add_node("execute", _node_execute)
    graph.add_node("children_wait", _node_children_wait)
    graph.add_node("evidence", _node_evidence)
    graph.add_node("replan_gate", _node_replan_gate)
    graph.add_node("synthesize", _node_synthesize)
    graph.add_node("critic", _node_critic)
    graph.add_node("patch", _node_patch)
    graph.add_node("patch_approval", _node_patch_approval)
    graph.add_node("finalize", _node_finalize)

    graph.set_entry_point("intake")

    def _after_intake(state: AgentPhaseState) -> str:
        if state.get("cancelled") or state.get("failure"):
            return "finalize"
        deps = _DEPS.get()
        policy = deps.tier_policy if deps is not None else None
        if policy is not None and not policy.discovery:
            # Speed tier (schnell): no probes, no intake questions —
            # plan immediately. Open questions degrade to the plan's
            # visible assumptions instead of an interrupt.
            return "plan"
        route = state.get("route", "discover_first")
        if route == "ask_user_first":
            return "clarify"
        if route == "plan_now":
            return "plan"
        return "discovery"

    graph.add_conditional_edges("intake", _after_intake)

    def _after_clarify(state: AgentPhaseState) -> str:
        if state.get("cancelled"):
            return "finalize"
        return "discovery" if state.get("discovery") is None else "plan"

    graph.add_conditional_edges("clarify", _after_clarify)

    def _after_discovery(state: AgentPhaseState) -> str:
        if state.get("cancelled") or state.get("failure"):
            return "finalize"
        if state.get("plan_rejected"):
            return "finalize"
        discovery = state.get("discovery")
        if discovery and any(
            gap.get("blocking") for gap in discovery.get("gaps", [])
        ):
            return "clarify_gaps"
        return "plan"

    graph.add_conditional_edges("discovery", _after_discovery)

    def _after_clarify_gaps(state: AgentPhaseState) -> str:
        return "finalize" if state.get("cancelled") else "plan"

    graph.add_conditional_edges("clarify_gaps", _after_clarify_gaps)

    def _after_plan(state: AgentPhaseState) -> str:
        if state.get("cancelled") or state.get("failure"):
            return "finalize"
        return "plan_approval"

    graph.add_conditional_edges("plan", _after_plan)

    def _after_plan_approval(state: AgentPhaseState) -> str:
        if state.get("cancelled") or state.get("failure"):
            return "finalize"
        if state.get("plan_rejected"):
            return "finalize"
        if state.get("replan_noop"):
            return "synthesize"
        return "execute"

    graph.add_conditional_edges("plan_approval", _after_plan_approval)

    def _after_execute(state: AgentPhaseState) -> str:
        if state.get("cancelled") or state.get("failure"):
            return "finalize"
        if state.get("pending_children"):
            return "children_wait"
        return "evidence"

    graph.add_conditional_edges("execute", _after_execute)

    def _after_children_wait(state: AgentPhaseState) -> str:
        if state.get("cancelled") or state.get("failure"):
            return "finalize"
        if state.get("pending_children"):
            # A transient child was resubmitted (or an early wake left
            # children outstanding): park again for the next round.
            return "children_wait"
        # Fold done: execute re-enters with the remaining todo (next
        # wave) — an empty todo falls straight through to evidence.
        return "execute"

    graph.add_conditional_edges("children_wait", _after_children_wait)

    def _after_evidence(state: AgentPhaseState) -> str:
        return "finalize" if state.get("cancelled") else "replan_gate"

    graph.add_conditional_edges("evidence", _after_evidence)

    def _after_replan_gate(state: AgentPhaseState) -> str:
        if state.get("cancelled"):
            return "finalize"
        return "plan" if state.get("route") == "replan" else "synthesize"

    graph.add_conditional_edges("replan_gate", _after_replan_gate)

    def _after_synthesize(state: AgentPhaseState) -> str:
        if state.get("cancelled") or state.get("failure"):
            return "finalize"
        deps = _DEPS.get()
        if (
            deps is not None
            and deps.tier_policy is not None
            and deps.tier_policy.verify == "labels"
        ):
            # Speed tier: citation-label validation already ran inside
            # synthesis; the critic pass is deliberately skipped. A
            # document-targeted assignment still MUST propose its patch —
            # skipping the critic never skips the deliverable.
            return "patch" if state.get("target_document_id") else "finalize"
        return "critic"

    graph.add_conditional_edges("synthesize", _after_synthesize)

    def _after_critic(state: AgentPhaseState) -> str:
        if state.get("cancelled") or state.get("failure"):
            return "finalize"
        if state.get("critic_recheck_pending"):
            return "critic"
        if state.get("route") == "critic_research":
            return "plan"
        # The patch phase runs only for assignments that TARGET an editor
        # document; the target travels on the run request, mirrored into
        # state by intake so the routing stays checkpoint-replayable.
        return "patch" if state.get("target_document_id") else "finalize"

    graph.add_conditional_edges("critic", _after_critic)

    def _after_patch(state: AgentPhaseState) -> str:
        if state.get("cancelled") or state.get("failure"):
            return "finalize"
        # No edits proposed -> nothing to approve (honest no-op note).
        return "patch_approval" if state.get("patch_id") else "finalize"

    graph.add_conditional_edges("patch", _after_patch)

    graph.add_edge("patch_approval", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=saver)
