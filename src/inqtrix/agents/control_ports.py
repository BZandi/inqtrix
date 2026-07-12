"""Contracts of the agent control store (plans, approvals, clarifications,
artifacts).

The control tables are the SOURCE OF TRUTH for everything a workspace-agent
run negotiates with its user (plan rule R1): plan versions, human-in-the-loop
approvals and clarifications, and the artifact documents the canvas renders.
Run events are signals only — clients reconcile via the REST reads over this
store.

Two implementations behind one port:
:class:`~inqtrix.agents.control_memory.MemoryAgentControlStore` (offline/test)
and
:class:`~inqtrix.storage.agent_control_postgres.PostgresAgentControlStore`.
Scoping stays OUT of the store: every row hangs off a run, and the router
resolves the run with the caller's visibility first (denial == absence), so
the store never sees principals.

Retention: all control rows are children of their run row (``ON DELETE
CASCADE`` in Postgres, mirrored in memory) — the durable run retention window
is the single retention authority. The session memo artifact survives across
turns because every turn re-anchors it onto the newest run via
:meth:`AgentControlStore.upsert_artifact`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

PLAN_STATUSES = ("draft", "proposed", "approved", "rejected", "superseded")
"""Lifecycle of one plan VERSION. ``superseded`` marks a version replaced by
a newer one (user edit or replan); at most one version per run is not
superseded/rejected."""

APPROVAL_KINDS = ("discovery", "plan", "patch", "replan", "tool")
"""What the run is asking permission for (plan decision E16).

``tool`` is the kernel's per-call policy gate (M2): the payload carries
``actions: [{tool, args, summary}]`` with the web query VERBATIM in
``args`` — the args ARE the approval content the user reviews. Edit
decisions are allowed for exactly one action (the HITL
``edited_action`` contract)."""

APPROVAL_STATUSES = ("pending", "approved", "rejected", "edited")
"""``edited`` is the approve-with-changes outcome: the decision carried a
revised plan, which became the new approved plan version."""

APPROVAL_DECISIONS = ("approve", "reject", "edit")

APPROVAL_STATUS_BY_DECISION = {
    "approve": "approved",
    "reject": "rejected",
    "edit": "edited",
}
"""THE one decision-verb -> approval-status mapping; the service (event
payloads) and both stores (row writes) consume it, so the emitted status
can never desynchronize from the stored one."""

CLARIFICATION_STATUSES = ("pending", "answered")

ARTIFACT_KINDS = (
    "memo",
    "evidence_bundle",
    "critic_report",
    "editor_patch",
    "answer",
    "deliverable",
)
"""``answer`` is the run-local chat-form deliverable (M1): rendered
inline in the agent timeline instead of the session memo canvas. Added
to the DB CHECK by migration 0039 together with the structured
clarification columns (one migration, plan M1).

``deliverable`` is a kernel canvas document (M2, ``write_canvas``):
session-scoped and (session_id, artifact_id)-addressed like the memo,
but a session may hold SEVERAL — the artifact registry in the session
context keeps them distinguishable (K2). Its ``payload.deliverable_kind``
carries the format hint (memo|email|talking_points|generic). Added to
the DB CHECK by migration 0040."""

ARTIFACT_STATUSES = ("writing", "ready")
"""``writing`` locks the artifact against user edits (E13: the canvas is
read-only while the agent streams into it)."""

SESSION_SINGLETON_ARTIFACT_KINDS = frozenset({"memo"})
"""Session-scoped artifact kinds addressed by ``(session_id, kind)``.

Kernel ``deliverable`` documents are deliberately absent: one session may
hold several of them, so they are always addressed by ``artifact_id``.
"""

RUN_SINGLETON_ARTIFACT_KINDS = frozenset(
    {"evidence_bundle", "critic_report", "editor_patch", "answer"}
)
"""Session-less diagnostic kinds addressed by ``(run_id, kind)``.

``deliverable`` is deliberately absent for the same reason as in the
session-scoped contract: one run may create several canvases.
"""

TASK_TOOL_KINDS = (
    "web_research",
    "web_instant",
    "rag_query",
    "file_analysis",
    "synthesis",
)

TASK_STATUSES = (
    "pending",
    "running",
    "cancel_requested",
    "cancelled",
    "completed",
    "failed",
    "insufficient_evidence",
    "skipped",
)
TASK_TERMINAL_STATUSES = frozenset(
    {"cancelled", "completed", "failed", "insufficient_evidence", "skipped"}
)
TASK_RESULT_SUMMARY_MAX_WORDS = 300


class PlanNotFound(KeyError):
    """Raised when a run has no plan (yet) or the version is unknown."""


class PlanTaskNotFound(KeyError):
    """Raised when a task is not part of the addressed plan version."""


class PlanTaskCancellationConflict(Exception):
    """Raised when an immutable or synthesis task cannot be cancelled."""

    def __init__(self, task: "PlanTaskRecord") -> None:
        super().__init__(
            f"task {task.task_id} cannot be cancelled from {task.status}"
        )
        self.task = task


class ApprovalNotFound(KeyError):
    """Raised when an approval id is unknown for the run (HTTP 404)."""


class ApprovalAlreadyDecided(Exception):
    """Raised when a decision hits an approval that is no longer pending.

    Carries the stored record so the caller can distinguish a replay of the
    SAME decision (idempotent 200) from a conflicting one (409).
    """

    def __init__(self, approval: "ApprovalRecord") -> None:
        super().__init__(f"approval {approval.approval_id} already decided")
        self.approval = approval


class ClarificationNotFound(KeyError):
    """Raised when a clarification id is unknown for the run (HTTP 404)."""


class ClarificationAlreadyAnswered(Exception):
    """Raised when an answer hits a clarification that is not pending.

    Carries the stored record for the replay-idempotency check (same
    answer -> 200, different answer -> 409).
    """

    def __init__(self, clarification: "ClarificationRecord") -> None:
        super().__init__(
            f"clarification {clarification.clarification_id} already answered"
        )
        self.clarification = clarification


class ArtifactNotFound(KeyError):
    """Raised when an artifact id (or requested revision) is unknown."""


class ArtifactLocked(Exception):
    """User edit rejected because the agent is writing (E13, 409)."""


class ArtifactRevisionConflict(Exception):
    """Optimistic-concurrency miss on a user edit (E13, 409).

    Attributes:
        current_revision: The revision the row actually holds — returned to
            the client so it can rebase instead of blind-retrying.
    """

    def __init__(self, current_revision: int) -> None:
        super().__init__(f"artifact is at revision {current_revision}")
        self.current_revision = current_revision


@dataclass(frozen=True)
class PlanTaskRecord:
    """One task row of a plan version.

    Attributes:
        task_id: Stable id (``task-...``), unique within the plan version;
            ``depends_on`` references these ids.
        plan_id: Owning plan-version row.
        run_id: Denormalized run id (query convenience; always equals the
            plan's run).
        ordinal: Display/served order (0-based, stable).
        title: User-facing task label (German, like all agent surface text).
        tool_kind: One of :data:`TASK_TOOL_KINDS`.
        objective: What the task must produce, for the executing child.
        queries: Concrete search/ask strings the task starts from.
        gap_ids: Discovery gaps this task covers (empty for synthesis).
        depends_on: Task ids that must complete first (wave scheduling).
        budget: Deprecated read-only bridge for historic rows. New planner
            and edit writes store ``{}``; operator quotas and profile
            timeouts are the only execution authority.
        params: Tool tuning (``profile``/``model_tier``/``recency``/
            ``collection_ids``), validated by the plan schema.
        expected_output: Short description of the deliverable shape.
        is_falsification: Marks the deliberate counter-evidence task
            (contested topics).
        status: One of :data:`TASK_STATUSES`; execution state written by the
            scheduler. ``cancel_requested`` is non-terminal and means a
            synchronous provider call may still be unwinding.
        child_run_id: The spawned child run for ``web_research`` tasks.
        result_summary: Compact result the supervisor keeps (<= 300 words).
        result_payload: Internal checkpoint-recovery fields that are not
            represented by the authoritative scalar columns (evidence,
            claims, usage, complete answer Markdown and failure detail). This
            field is not exposed by the HTTP plan contract; the authenticated
            lazy task-result surface projects the complete value separately.
    """

    task_id: str
    plan_id: str
    run_id: str
    ordinal: int
    title: str
    tool_kind: str
    objective: str = ""
    queries: tuple[str, ...] = ()
    gap_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    is_falsification: bool = False
    status: str = "pending"
    child_run_id: str | None = None
    result_summary: str = ""
    result_payload: dict[str, Any] = field(default_factory=dict)


def transitioned_plan_task(
    current: PlanTaskRecord,
    *,
    status: str,
    child_run_id: str | None = None,
    result_summary: str | None = None,
    result_payload: dict[str, Any] | None = None,
) -> PlanTaskRecord:
    """Validate and materialize one idempotent task-state transition.

    A running retry may replace its child id. Once terminal, status, child id,
    summary, and recovery payload are frozen; an identical replay returns the
    existing record.
    """
    if status not in TASK_STATUSES:
        raise ValueError(f"unknown task status: {status!r}")
    normalized_summary: str | None = None
    if result_summary is not None:
        normalized_summary = " ".join(str(result_summary).split())
        if len(normalized_summary.split()) > TASK_RESULT_SUMMARY_MAX_WORDS:
            raise ValueError(
                f"task result_summary exceeds "
                f"{TASK_RESULT_SUMMARY_MAX_WORDS} words"
            )
        if status not in TASK_TERMINAL_STATUSES:
            raise ValueError(
                "task result_summary may only be written with a terminal status"
            )
    normalized_payload: dict[str, Any] | None = None
    if result_payload is not None:
        normalized_payload = dict(result_payload)
        if status not in TASK_TERMINAL_STATUSES:
            raise ValueError(
                "task result_payload may only be written with a terminal status"
            )
    allowed = {
        "pending": {"running", "skipped", "cancelled"},
        "running": {"running", "cancel_requested", *TASK_TERMINAL_STATUSES},
        "cancel_requested": {"cancel_requested", "cancelled"},
    }
    if current.status in TASK_TERMINAL_STATUSES:
        if status != current.status:
            raise ValueError(
                f"task {current.task_id} is terminal ({current.status})"
            )
        if child_run_id is not None and child_run_id != current.child_run_id:
            raise ValueError(
                f"task {current.task_id} already belongs to child "
                f"{current.child_run_id}"
            )
        if (
            normalized_summary is not None
            and normalized_summary != current.result_summary
        ):
            raise ValueError(
                f"task {current.task_id} already has a different result"
            )
        if (
            normalized_payload is not None
            and normalized_payload != current.result_payload
        ):
            raise ValueError(
                f"task {current.task_id} already has a different result payload"
            )
        return current
    elif status != current.status and status not in allowed.get(
        current.status, set()
    ):
        raise ValueError(
            f"task {current.task_id} cannot transition "
            f"{current.status} -> {status}"
        )
    if (
        child_run_id is not None
        and current.status == "pending"
        and status != "running"
    ):
        raise ValueError(
            f"task {current.task_id} cannot attach a child before running"
        )
    next_child = current.child_run_id
    if child_run_id is not None:
        if next_child is not None and next_child != child_run_id:
            if current.status != "running" or status != "running":
                raise ValueError(
                    f"task {current.task_id} already belongs to child "
                    f"{next_child}"
                )
        next_child = child_run_id
    next_summary = current.result_summary
    if normalized_summary is not None:
        if next_summary and next_summary != normalized_summary:
            raise ValueError(
                f"task {current.task_id} already has a different result"
            )
        next_summary = normalized_summary
    next_payload = current.result_payload
    if normalized_payload is not None:
        if next_payload and next_payload != normalized_payload:
            raise ValueError(
                f"task {current.task_id} already has a different result payload"
            )
        next_payload = normalized_payload
    return replace(
        current,
        status=status,
        child_run_id=next_child,
        result_summary=next_summary,
        result_payload=next_payload,
    )


def requested_task_cancellation(current: PlanTaskRecord) -> PlanTaskRecord:
    """Apply the atomic task-cancellation state machine.

    Pending source work ends immediately. Running source work records an
    honest request until its synchronous operation or child run unwinds.
    Already cancelled requests are idempotent; every other terminal result
    and synthesis are immutable conflicts.
    """
    if current.tool_kind == "synthesis":
        raise PlanTaskCancellationConflict(current)
    if current.status in {"cancel_requested", "cancelled"}:
        return current
    if current.status == "pending":
        return transitioned_plan_task(
            current,
            status="cancelled",
            result_summary="Aufgabe vor der Ausfuehrung abgebrochen.",
            result_payload={
                "failure_code": "task_cancelled",
                "failure_reason": "user_requested_task_cancel",
            },
        )
    if current.status == "running":
        return transitioned_plan_task(current, status="cancel_requested")
    raise PlanTaskCancellationConflict(current)


def additive_replan_errors(
    previous_tasks: list[PlanTaskRecord],
    candidate_tasks: list[PlanTaskRecord],
) -> list[str]:
    """Validate stable task identities across an additive replan.

    Only terminal source tasks are immutable execution results. They must
    remain present under the same id and with the same definition. A changed
    operation therefore needs a new task id; otherwise the checkpointed
    outcome map would make the changed operation look already completed.
    Synthesis is deliberately excluded because every plan version synthesizes
    the complete, newly available evidence again.
    """
    previous = {
        task.task_id: task
        for task in previous_tasks
        if task.tool_kind != "synthesis"
        and task.status in TASK_TERMINAL_STATUSES
    }
    candidate = {task.task_id: task for task in candidate_tasks}
    errors: list[str] = []
    missing = sorted(set(previous) - set(candidate))
    if missing:
        errors.append(
            "Ein additiver Replan darf erledigte Tasks nicht entfernen: "
            + ", ".join(missing)
            + "."
        )
    changed = sorted(
        task_id
        for task_id in set(previous) & set(candidate)
        if _task_definition(previous[task_id])
        != _task_definition(candidate[task_id])
    )
    if changed:
        errors.append(
            "Ein additiver Replan darf erledigte Task-IDs nicht fuer eine "
            "geaenderte Aufgabe wiederverwenden; behalte sie unveraendert "
            "und vergib neue IDs fuer: "
            + ", ".join(changed)
            + "."
        )
    return errors


def carry_forward_terminal_task_results(
    previous_tasks: list[PlanTaskRecord],
    candidate_tasks: list[PlanTaskRecord],
) -> list[PlanTaskRecord]:
    """Carry immutable source-task results into a validated plan version."""
    errors = additive_replan_errors(previous_tasks, candidate_tasks)
    if errors:
        raise ValueError("; ".join(errors))
    previous = {
        task.task_id: task
        for task in previous_tasks
        if task.tool_kind != "synthesis"
        and task.status in TASK_TERMINAL_STATUSES
    }
    return [
        replace(
            task,
            status=prior.status,
            child_run_id=prior.child_run_id,
            result_summary=prior.result_summary,
            result_payload=dict(prior.result_payload),
        )
        if (prior := previous.get(task.task_id)) is not None
        else task
        for task in candidate_tasks
    ]


def _task_definition(task: PlanTaskRecord) -> tuple[Any, ...]:
    """Execution definition protected by a stable task id."""
    return (
        task.tool_kind,
        task.objective,
        tuple(task.queries),
        tuple(task.gap_ids),
        tuple(task.depends_on),
        _freeze_json(task.params),
        task.expected_output,
        task.is_falsification,
    )


def _freeze_json(value: Any) -> Any:
    """Canonical immutable comparison form for JSON-compatible values."""
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_json(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


async def settle_cancelled_plan_tasks(
    store: "AgentControlStore", run_id: str
) -> None:
    """Close every unfinished task when its parent run is cancelled.

    Pending rows were never dispatched and become ``skipped``. Running rows
    may already have caused an external side effect, so they become ``failed``
    instead of claiming they were not executed. Terminal rows are immutable.
    """
    await settle_terminal_plan_tasks(store, run_id, status="cancelled")


async def settle_terminal_plan_tasks(
    store: "AgentControlStore", run_id: str, *, status: str
) -> None:
    """Reconcile unfinished plan rows after a terminal run transition.

    This is the idempotent recovery boundary for a run that can no longer
    re-enter its graph. Cancellation and infrastructure failure use distinct
    summaries, while the task-state semantics stay identical: undispatched
    work is skipped and begun work fails. Completed task results are never
    rewritten.

    Args:
        store: Authoritative agent control store.
        run_id: Terminal agent run whose latest plan is reconciled.
        status: ``cancelled`` or ``failed``.

    Raises:
        ValueError: If a non-terminal status reaches this boundary.
    """
    if status not in {"cancelled", "failed"}:
        raise ValueError(f"unsupported terminal plan status: {status}")
    try:
        _plan, tasks = await store.get_plan(run_id)
    except PlanNotFound:
        return
    pending_summary = (
        "Wegen Abbruch nicht ausgeführt."
        if status == "cancelled"
        else "Nicht ausgeführt, weil der Lauf fehlgeschlagen ist."
    )
    running_summary = (
        "Ausführung abgebrochen; ein begonnenes Ergebnis wurde nicht übernommen."
        if status == "cancelled"
        else "Ausführung durch einen Lauf-Fehler unvollständig beendet."
    )
    for task in tasks:
        if task.status == "pending":
            await store.transition_plan_task(
                run_id=task.run_id,
                plan_id=task.plan_id,
                task_id=task.task_id,
                status="skipped",
                result_summary=pending_summary,
            )
        elif task.status == "running":
            await store.transition_plan_task(
                run_id=task.run_id,
                plan_id=task.plan_id,
                task_id=task.task_id,
                status="failed",
                child_run_id=task.child_run_id,
                result_summary=running_summary,
            )
        elif task.status == "cancel_requested":
            await store.transition_plan_task(
                run_id=task.run_id,
                plan_id=task.plan_id,
                task_id=task.task_id,
                status="cancelled",
                child_run_id=task.child_run_id,
                result_summary=(
                    "Aufgabe auf Nutzerwunsch abgebrochen; ein begonnenes "
                    "Ergebnis wurde nicht übernommen."
                ),
                result_payload={
                    "failure_code": "task_cancelled",
                    "failure_reason": "user_requested_task_cancel",
                },
            )


@dataclass(frozen=True)
class PlanRecord:
    """One plan VERSION of a run (append-only versioning).

    Attributes:
        plan_id: Row id (``plan-...``).
        run_id: The run this plan belongs to.
        version: 1-based, unique per run; a user edit or replan appends
            version+1 and supersedes the previous one.
        status: One of :data:`PLAN_STATUSES`.
        created_by: ``agent`` (planner output) or ``user`` (edit decision).
        summary_markdown: The plan's one-paragraph intent, rendered above the
            task list.
        assumptions: Non-blocking open points the plan proceeds on.
        success_criteria: Measurable criteria (German) the critic later
            checks the memo against.
        reason: Why this version exists — empty for the initial plan,
            ``critic_research``/``replan`` for current agent-generated
            revisions, or ``user_edit`` afterwards. The unconstrained string
            deliberately preserves older persisted reason values, but no
            budget-driven reason is generated anymore.
        created_at: Unix seconds.
    """

    plan_id: str
    run_id: str
    version: int
    status: str
    created_by: str
    summary_markdown: str = ""
    assumptions: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    reason: str = ""
    created_at: float = 0.0


@dataclass(frozen=True)
class ApprovalRecord:
    """One human-in-the-loop approval request of a run.

    Attributes:
        approval_id: Row id (``apr_...``).
        run_id: The asking run.
        kind: One of :data:`APPROVAL_KINDS`.
        status: One of :data:`APPROVAL_STATUSES`.
        subject_type/subject_id: What is being approved (``plan`` +
            plan_id for plan/replan approvals; probe list payload for
            discovery).
        payload: Kind-specific request context shown to the user (for
            discovery: the planned probes; empty for plan kinds — the plan
            is fetched via its own endpoint).
        decision: The verb that resolved it (``approve``/``reject``/
            ``edit``), empty while pending.
        decision_payload: The edited plan for ``edit`` decisions; empty
            otherwise.
        note: Optional free-text the decider attached.
        decided_by_sub: Verified subject that decided.
        interrupt_key: Checkpoint correlation key for the M5 resume
            (opaque to this layer).
        created_at/decided_at: Unix seconds.
    """

    approval_id: str
    run_id: str
    kind: str
    status: str = "pending"
    subject_type: str = ""
    subject_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    decision: str = ""
    decision_payload: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    decided_by_sub: str | None = None
    interrupt_key: str = ""
    created_at: float = 0.0
    decided_at: float | None = None


@dataclass(frozen=True)
class ClarificationRecord:
    """One clarification GATE ROUND the run asked its user.

    A round carries 1-3 structured questions (decision #8 refinement);
    the legacy single-question columns stay valid so a whole-round
    free-text answer through the composer keeps working unchanged.

    Attributes:
        clarification_id: Row id (``clr_...``).
        run_id: The asking run.
        question: Joined prompt text of the round (German) — the legacy
            single-text reading and the free-text answer target.
        options: LEGACY single-question options, each ``{"id": ..,
            "label": ..}`` (mirrors ``questions[0]["options"]`` when the
            round has exactly one question, else empty). Kept so older
            readers and the plain ``option_id`` answer path survive.
        questions: Structured round payload, each ``{"id", "prompt",
            "options": [{"id", "label", "description"}], "multi_select"}``
            (sanitized, deterministic ids — see
            :func:`inqtrix.agents.clarification.sanitize_questions`).
            Empty for legacy rounds.
        answers: Structured answers by question id, each
            ``{"option_ids": [..], "text": ".."}``; empty when the round
            was answered with whole-round free text or a legacy option.
        default_assumption: What the agent assumes when the question is
            dismissed or times out — shown to the user up front.
        status: One of :data:`CLARIFICATION_STATUSES`.
        answer: Whole-round free-text answer (empty when structured
            ``answers`` or a legacy option was given).
        option_id: Picked legacy option id (empty otherwise).
        answered_by_sub: Verified subject that answered.
        created_at/answered_at: Unix seconds.
    """

    clarification_id: str
    run_id: str
    question: str
    options: tuple[dict[str, Any], ...] = ()
    questions: tuple[dict[str, Any], ...] = ()
    answers: dict[str, Any] = field(default_factory=dict)
    default_assumption: str = ""
    status: str = "pending"
    answer: str = ""
    option_id: str = ""
    answered_by_sub: str | None = None
    created_at: float = 0.0
    answered_at: float | None = None


@dataclass(frozen=True)
class ArtifactRecord:
    """One artifact document of a run (canvas content, rule R1 truth).

    Attributes:
        artifact_id: Row id (``art_...``).
        run_id: The run that currently anchors the artifact. The session
            memo is re-anchored to the newest run on every turn (upsert by
            ``(session_id, kind='memo')``), so its retention follows the
            LATEST run.
        session_id: Agent-desk session for cross-run artifacts (the memo);
            ``None`` for run-local diagnostics (critic report, evidence
            bundle).
        kind: One of :data:`ARTIFACT_KINDS`.
        title: Display title.
        status: ``writing`` (agent streams, user edits rejected with 409)
            or ``ready``.
        revision: 1-based optimistic-concurrency counter; every content
            write increments it and appends a revision row.
        updated_by: ``agent`` or ``user`` — who wrote the current revision.
        content_markdown: The document body (markdown with ``[K#]``/``[W#]``
            citation labels).
        payload: Kind-specific structured data (e.g. the critic report
            JSON); the memo keeps it empty.
        refs: Evidence references backing the content (citation label ->
            source descriptor).
        created_at/updated_at: Unix seconds.
    """

    artifact_id: str
    run_id: str
    kind: str
    session_id: str | None = None
    title: str = ""
    status: str = "ready"
    revision: int = 1
    updated_by: str = "agent"
    content_markdown: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    refs: tuple[dict[str, Any], ...] = ()
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True)
class ArtifactRevisionRecord:
    """One append-only content revision of an artifact (diff history).

    Attributes:
        artifact_id: Owning artifact.
        revision: The revision number this row snapshots.
        created_by: ``agent`` or ``user``.
        content_markdown: Full body at that revision (revision diffs are
            computed client-side from two full bodies).
        created_at: Unix seconds.
    """

    artifact_id: str
    revision: int
    created_by: str
    content_markdown: str = ""
    created_at: float = 0.0


@dataclass(frozen=True)
class ArtifactBatchRevision:
    """One content-only artifact update inside an atomic revision batch.

    Attributes:
        artifact_id: Existing session artifact to revise.
        expected_revision: Exact current revision required by CAS.
        content_markdown: Complete replacement body. Metadata and evidence
            references are preserved from the stored artifact.
    """

    artifact_id: str
    expected_revision: int
    content_markdown: str


@runtime_checkable
class AgentControlStore(Protocol):
    """Persistence port for agent run control data.

    Writes that resolve an interrupt (approval decision, clarification
    answer) do NOT resume the run here — the service composes them with the
    run store so the durable backend can put the decision write and the
    ``waiting -> queued`` flip into one transaction (rule R9).
    """

    # -- plans --------------------------------------------------------- #

    async def save_plan(
        self,
        *,
        run_id: str,
        plan: PlanRecord,
        tasks: list[PlanTaskRecord],
    ) -> PlanRecord:
        """Append one plan version with its tasks.

        The version must be exactly ``latest + 1`` (or 1 for the first);
        any non-rejected older versions flip to ``superseded`` in the same
        write. Raises ``ValueError`` on version gaps.
        """
        ...

    async def get_plan(
        self, run_id: str, *, version: int | None = None
    ) -> tuple[PlanRecord, list[PlanTaskRecord]]:
        """One plan version (latest when ``version`` is None) with its
        tasks ordered by ordinal; :class:`PlanNotFound` when absent."""
        ...

    async def list_plan_versions(self, run_id: str) -> list[PlanRecord]:
        """All versions newest-first, WITHOUT tasks (version picker)."""
        ...

    async def transition_plan_task(
        self,
        *,
        run_id: str,
        plan_id: str,
        task_id: str,
        status: str,
        child_run_id: str | None = None,
        result_summary: str | None = None,
        result_payload: dict[str, Any] | None = None,
    ) -> PlanTaskRecord:
        """Move one task through its idempotent execution lifecycle.

        Terminal writes persist the compact public summary and the internal
        checkpoint-recovery payload in the same store transition.
        """
        ...

    async def request_plan_task_cancel(
        self,
        *,
        run_id: str,
        plan_id: str,
        task_id: str,
    ) -> PlanTaskRecord:
        """Atomically request cancellation of one source task."""
        ...

    # -- approvals ------------------------------------------------------ #

    async def create_approval(self, approval: ApprovalRecord) -> ApprovalRecord: ...

    async def list_approvals(self, run_id: str) -> list[ApprovalRecord]:
        """All approvals of the run, newest-first."""
        ...

    async def get_approval(self, run_id: str, approval_id: str) -> ApprovalRecord:
        """One approval; :class:`ApprovalNotFound` when absent (the run id
        must match — approvals are not addressable across runs)."""
        ...

    async def decide_approval_and_resume(
        self,
        *,
        run_id: str,
        approval_id: str,
        decision: str,
        decision_payload: dict[str, Any],
        note: str,
        decided_by_sub: str | None,
        resume: Any,
        edited_plan: PlanRecord | None = None,
        edited_tasks: list[PlanTaskRecord] | None = None,
    ) -> tuple[ApprovalRecord, dict[str, Any]]:
        """Record the decision AND resume the waiting run — atomically per
        backend (rule R9).

        *resume* is an AWAITABLE callable ``await resume(control_write)
        -> run summary``
        supplied by the service (it wraps ``run_store.resume_run``): the
        memory store calls ``resume(None)`` after its in-lock decision CAS
        (reverting the CAS when the resume fails); the Postgres store hands
        ``resume`` a coroutine writer that performs the decision CAS — and
        the ``edit`` decision's new plan version — INSIDE the run store's
        ``waiting -> queued`` transaction, so a crash can never separate
        the decision from the resume.

        ``edit`` decisions carry *edited_plan* (``version`` assigned by the
        store as latest+1; older versions flip to ``superseded``) and
        *edited_tasks*.

        Raises:
            ApprovalNotFound: Unknown approval for the run.
            ApprovalAlreadyDecided: The approval is not pending (carries
                the stored record for the replay-idempotency check).
            Whatever *resume* raises when the run cannot resume (the
                decision is rolled back in that case).
        """
        ...

    # -- clarifications -------------------------------------------------- #

    async def create_clarification(
        self, clarification: ClarificationRecord
    ) -> ClarificationRecord: ...

    async def list_clarifications(self, run_id: str) -> list[ClarificationRecord]:
        """All clarifications of the run, newest-first."""
        ...

    async def get_clarification(
        self, run_id: str, clarification_id: str
    ) -> ClarificationRecord: ...

    async def answer_clarification_and_resume(
        self,
        *,
        run_id: str,
        clarification_id: str,
        answer: str,
        option_id: str,
        answers: dict[str, Any],
        answered_by_sub: str | None,
        resume: Any,
    ) -> tuple[ClarificationRecord, dict[str, Any]]:
        """Record the answer AND resume the waiting run (rule R9).

        Same composition contract as
        :meth:`decide_approval_and_resume` — exactly one of *answer* /
        *option_id* / *answers* is non-empty (the service validates
        that, including the per-question shape of *answers*).

        Raises:
            ClarificationNotFound: Unknown clarification for the run.
            ClarificationAlreadyAnswered: Not pending (carries the stored
                record for the replay check).
            Whatever *resume* raises (the answer is rolled back then).
        """
        ...

    # -- artifacts ------------------------------------------------------- #

    async def upsert_artifact(
        self,
        *,
        run_id: str,
        kind: str,
        session_id: str | None,
        title: str,
        status: str,
        content_markdown: str,
        payload: dict[str, Any],
        refs: list[dict[str, Any]],
        updated_by: str,
        artifact_id: str | None = None,
        expected_revision: int | None = None,
    ) -> ArtifactRecord:
        """Create or advance an artifact (agent write path, M5).

        Upsert key: explicit ``artifact_id`` when given, else
        ``(session_id, kind)`` for session-scoped kinds, else
        ``(run_id, kind)``. Every content change bumps ``revision`` and
        appends a revision row; the run anchor moves to *run_id*.

        Args:
            expected_revision: Optimistic-concurrency guard (decision E13,
                the agent's symmetric half). ``None`` (default) advances
                unconditionally — correct for write-once per-run kinds
                (evidence bundles, critic reports) the user never edits.
                An int makes the write conditional against an EXISTING row:
                if its revision differs (someone wrote since the caller
                read it), :class:`ArtifactRevisionConflict` is raised
                (carrying the current revision) instead of clobbering.
                The sentinel ``0`` means "I expect to CREATE" — a
                concurrently-inserted row is a conflict, not a silent
                advance (closes the fresh-session insert race). When no row
                exists the value is irrelevant: the insert proceeds.
        """
        ...

    async def get_session_artifact(
        self, session_id: str, kind: str
    ) -> ArtifactRecord | None:
        """The current session-scoped artifact of *kind*, or ``None``.

        The cross-run read behind session-memo lineage (decision E15):
        a follow-up turn is a NEW run, so :meth:`get_artifact` (run-scoped)
        cannot reach the prior turn's memo — this resolves it by the
        durable ``(session_id, kind)`` key and returns the LATEST revision
        with its body, so the agent continues from (and never clobbers)
        the user's most recent edit.
        """
        ...

    async def get_session_artifact_by_id(
        self, session_id: str, artifact_id: str
    ) -> ArtifactRecord:
        """Return one full session artifact addressed by id.

        This is the cross-run read used by the kernel's ``read_canvas``
        tool. Both keys are required so an id from another session cannot be
        used as a confused-deputy target.

        Raises:
            ArtifactNotFound: The id is unknown or belongs to another
                session.
        """
        ...

    async def revise_session_artifacts_atomically(
        self,
        *,
        run_id: str,
        session_id: str | None,
        revisions: list[ArtifactBatchRevision],
    ) -> list[ArtifactRecord]:
        """Apply a content-only multi-artifact CAS as one transaction.

        Every id must belong to ``session_id`` and every expected revision
        must match before any row changes. Payloads and evidence refs are
        immutable through this path.

        Raises:
            ArtifactNotFound: Any target is unknown or outside the session.
            ArtifactRevisionConflict: Any target revision has moved.
        """
        ...

    async def list_session_artifacts(
        self, session_id: str
    ) -> list[ArtifactRecord]:
        """ALL session-scoped artifacts of a session, oldest first,
        META-ONLY (``content_markdown`` empty).

        The session-context builder (plan K2) lists every deliverable of
        the session so a follow-up turn can name and target the right
        one; bodies stay out (the registry is a prompt-sized index, the
        body read stays :meth:`get_session_artifact`).
        """
        ...

    async def user_update_artifact(
        self,
        *,
        run_id: str,
        artifact_id: str,
        content_markdown: str,
        expected_revision: int,
    ) -> ArtifactRecord:
        """Optimistic user edit (E13).

        Raises :class:`ArtifactNotFound`, :class:`ArtifactLocked` (status
        ``writing``), or :class:`ArtifactRevisionConflict` (revision CAS
        miss, carries the current revision).
        """
        ...

    async def get_artifact(
        self, run_id: str, artifact_id: str, *, revision: int | None = None
    ) -> tuple[ArtifactRecord, list[ArtifactRevisionRecord]]:
        """One artifact plus its revision METADATA list (no bodies).

        With ``revision`` the returned record carries that revision's body
        (the row metadata stays current); :class:`ArtifactNotFound` for
        unknown ids or revisions.
        """
        ...

    async def list_artifacts(
        self,
        run_id: str,
        *,
        kind: str | None = None,
        limit: int = 50,
        after: tuple[float, str] | None = None,
    ) -> tuple[list[ArtifactRecord], str | None]:
        """Metadata page (``content_markdown=""``), newest-first keyset
        pagination; returns ``(rows, next_cursor)``."""
        ...

    async def aclose(self) -> None: ...
