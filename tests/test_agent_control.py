"""Agent control service + memory store: state machines, R9 composition,
artifact 409 matrix, events, audit (offline tier).

The Postgres lockstep counterpart lives in
``tests/storage/test_agent_control_postgres.py``.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from inqtrix.agents.control_memory import MemoryAgentControlStore
from inqtrix.agents.control_ports import (
    ApprovalAlreadyDecided,
    ApprovalRecord,
    ArtifactLocked,
    ArtifactNotFound,
    ArtifactRevisionConflict,
    ClarificationRecord,
    PlanNotFound,
    PlanRecord,
    PlanTaskCancellationConflict,
    PlanTaskRecord,
    settle_terminal_plan_tasks,
)
from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.principal import Principal
from inqtrix.server.runs import RunActive, RunHandle, RunStore
from inqtrix.services.agent_control_service import (
    AgentControlService,
    AgentControlUnavailable,
    AgentControlValidationError,
)

PRINCIPAL = Principal(
    sub="owner-1", kind="oidc_session", tenant_id="default", role="member"
)


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition was not reached before timeout")


@pytest.fixture()
def run_store() -> RunStore:
    return RunStore(
        max_concurrent=2,
        max_queue_size=8,
        completed_ttl_seconds=60,
        event_buffer_size=64,
    )


@pytest.fixture()
def identity() -> MemoryIdentityStore:
    return MemoryIdentityStore()


@pytest.fixture()
def service(run_store: RunStore, identity: MemoryIdentityStore) -> AgentControlService:
    return AgentControlService(
        store=MemoryAgentControlStore(),
        run_store=run_store,
        audit=identity,
        editor_persistence=None,
        durable=False,
    )


def _parked_agent_run(
    run_store: RunStore,
    *,
    resumed_release: threading.Event | None = None,
) -> str:
    """Submit an agent run that parks itself and completes on resume."""
    calls = {"count": 0}

    def segmented(handle: RunHandle) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            handle.wait("waiting_for_approval")
            return
        if resumed_release is not None:
            assert resumed_release.wait(timeout=10.0)
        handle.complete({"answer": "fortgesetzt"})

    summary = run_store.submit(
        question="Agentenauftrag",
        stack_name="default",
        work=segmented,
        kind="agent",
        session_id="sess-1",
    )
    run_id = summary["run_id"]
    _wait_until(
        lambda: run_store.get(run_id)["status"] == "waiting_for_approval"
    )
    return run_id


def _approval(run_id: str, *, kind: str = "plan") -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=f"apr_{uuid.uuid4().hex[:8]}",
        run_id=run_id,
        kind=kind,
    )


def _plan_body() -> dict[str, Any]:
    return {
        "summary_markdown": "Ueberarbeiteter Plan",
        "tasks": [
            {
                "id": "t1",
                "title": "Interne Suche",
                "tool_kind": "rag_query",
                "queries": ["Welche internen Berichte sind relevant?"],
                "params": {"profile": "standard"},
            },
            {
                "id": "s",
                "title": "Synthese",
                "tool_kind": "synthesis",
                "depends_on": ["t1"],
            },
        ],
        "success_criteria": ["Alle Kernfragen belegt."],
    }


# -- approval decision state machine ------------------------------------- #


@pytest.mark.asyncio
async def test_approve_decision_resumes_run_and_audits(
    service: AgentControlService,
    run_store: RunStore,
    identity: MemoryIdentityStore,
) -> None:
    resumed_release = threading.Event()
    run_id = _parked_agent_run(
        run_store, resumed_release=resumed_release
    )
    approval = await service.store.create_approval(_approval(run_id))

    try:
        decided, summary, replayed = await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="approve",
            plan_body=None,
            note="passt",
            principal=PRINCIPAL,
        )
    finally:
        resumed_release.set()

    assert not replayed
    assert decided.status == "approved"
    assert decided.decided_by_sub == "owner-1"
    assert decided.note == "passt"
    assert summary["status"] in {"queued", "running", "completed"}
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")

    subscription = run_store.subscribe(run_id)
    try:
        decision_event = next(
            event
            for event in subscription.replay
            if event["type"] == "inqtrix.agent.approval.decided"
        )
        resumed_event = next(
            event
            for event in subscription.replay
            if event["type"] == "inqtrix.run.queued"
            and event["data"].get("resumed")
        )
        # The decision event follows the resumed-queued event: rows are
        # truth, events are post-commit signals (rule R1).
        assert resumed_event["sequence"] < decision_event["sequence"]
    finally:
        subscription.close()
    actions = [entry.action for entry in identity.audit_entries]
    assert "agent.approval_decided" in actions
    entry = identity.audit_entries[-1]
    assert entry.actor_type == "user"
    assert entry.detail["decision"] == "approve"


@pytest.mark.asyncio
async def test_replay_of_same_decision_is_idempotent(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))
    await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="reject",
        plan_body=None,
        note="",
        principal=PRINCIPAL,
    )

    decided, _summary, replayed = await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="reject",
        plan_body=None,
        note="",
        principal=PRINCIPAL,
    )

    assert replayed
    assert decided.status == "rejected"

    with pytest.raises(ApprovalAlreadyDecided):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="approve",
            plan_body=None,
            note="",
            principal=PRINCIPAL,
        )


@pytest.mark.asyncio
async def test_decision_on_non_waiting_run_rolls_back(
    service: AgentControlService, run_store: RunStore
) -> None:
    summary = run_store.submit(
        question="normal",
        stack_name="default",
        work=lambda handle: handle.complete({}),
        kind="agent",
    )
    run_id = summary["run_id"]
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")
    approval = await service.store.create_approval(_approval(run_id))

    with pytest.raises(RunActive):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="approve",
            plan_body=None,
            note="",
            principal=PRINCIPAL,
        )

    # The decision must NOT stand: the approval stays pending (R9).
    stored = await service.store.get_approval(run_id, approval.approval_id)
    assert stored.status == "pending"
    subscription = run_store.subscribe(run_id)
    try:
        assert "inqtrix.agent.approval.decided" not in {
            event["type"] for event in subscription.replay
        }
    finally:
        subscription.close()


@pytest.mark.asyncio
async def test_edit_decision_appends_approved_plan_version(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    initial = PlanRecord(
        plan_id="plan_v1",
        run_id=run_id,
        version=1,
        status="proposed",
        created_by="agent",
    )
    await service.store.save_plan(
        run_id=run_id,
        plan=initial,
        tasks=[
            PlanTaskRecord(
                task_id="t1",
                plan_id="plan_v1",
                run_id=run_id,
                ordinal=0,
                title="Alt",
                tool_kind="rag_query",
            )
        ],
    )
    approval = await service.store.create_approval(_approval(run_id))

    decided, _summary, _replayed = await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="edit",
        plan_body=_plan_body(),
        note="",
        principal=PRINCIPAL,
    )

    assert decided.status == "edited"
    assert decided.decision_payload["plan"]["summary_markdown"] == (
        "Ueberarbeiteter Plan"
    )
    plan, tasks, versions = await service.get_plan(run_id)
    assert plan.version == 2
    assert plan.status == "approved"
    assert plan.created_by == "user"
    assert plan.reason == "user_edit"
    assert [task.task_id for task in tasks] == ["t1", "s"]
    assert [(v.version, v.status) for v in versions] == [
        (2, "approved"),
        (1, "superseded"),
    ]


@pytest.mark.asyncio
async def test_edit_decision_validates_through_the_plan_validator(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))
    broken = _plan_body()
    broken["tasks"][1]["depends_on"] = []  # synthesis no longer covers t1

    with pytest.raises(AgentControlValidationError) as exc_info:
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=broken,
            note="",
            principal=PRINCIPAL,
        )

    assert any("synthesis" in error for error in exc_info.value.errors)
    stored = await service.store.get_approval(run_id, approval.approval_id)
    assert stored.status == "pending"
    with pytest.raises(PlanNotFound):
        await service.store.get_plan(run_id)


@pytest.mark.asyncio
async def test_task_transitions_follow_retry_and_terminal_contract(
    service: AgentControlService,
) -> None:
    await service.store.save_plan(
        run_id="run-task",
        plan=PlanRecord(
            plan_id="plan-task",
            run_id="run-task",
            version=1,
            status="approved",
            created_by="agent",
        ),
        tasks=[
            PlanTaskRecord(
                task_id="t1",
                plan_id="plan-task",
                run_id="run-task",
                ordinal=0,
                title="Research",
                tool_kind="web_research",
            )
        ],
    )

    first = await service.store.transition_plan_task(
        run_id="run-task",
        plan_id="plan-task",
        task_id="t1",
        status="running",
        child_run_id="run-child-1",
    )
    assert first.child_run_id == "run-child-1"
    retry = await service.store.transition_plan_task(
        run_id="run-task",
        plan_id="plan-task",
        task_id="t1",
        status="running",
        child_run_id="run-child-2",
    )
    assert retry.child_run_id == "run-child-2"
    terminal = await service.store.transition_plan_task(
        run_id="run-task",
        plan_id="plan-task",
        task_id="t1",
        status="insufficient_evidence",
        child_run_id="run-child-2",
        result_summary="Only one independent source was available.",
        result_payload={
            "evidence": [{"url": "https://example.test/source"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )
    assert terminal.status == "insufficient_evidence"
    assert terminal.result_payload["usage"]["prompt_tokens"] == 3
    replay = await service.store.transition_plan_task(
        run_id="run-task",
        plan_id="plan-task",
        task_id="t1",
        status="insufficient_evidence",
        child_run_id="run-child-2",
        result_summary="Only one independent source was available.",
        result_payload=terminal.result_payload,
    )
    assert replay == terminal
    with pytest.raises(ValueError):
        await service.store.transition_plan_task(
            run_id="run-task",
            plan_id="plan-task",
            task_id="t1",
            status="completed",
        )
    with pytest.raises(ValueError):
        await service.store.transition_plan_task(
            run_id="run-task",
            plan_id="plan-task",
            task_id="t1",
            status="insufficient_evidence",
            child_run_id="run-child-3",
        )
    with pytest.raises(ValueError):
        await service.store.transition_plan_task(
            run_id="run-task",
            plan_id="plan-task",
            task_id="t1",
            status="insufficient_evidence",
            result_summary="A different terminal result.",
        )
    with pytest.raises(ValueError):
        await service.store.transition_plan_task(
            run_id="run-task",
            plan_id="plan-task",
            task_id="t1",
            status="insufficient_evidence",
            result_payload={"claims": [{"text": "different"}]},
        )


@pytest.mark.asyncio
async def test_task_transition_rejects_metadata_before_its_lifecycle_stage(
    service: AgentControlService,
) -> None:
    await service.store.save_plan(
        run_id="run-task-metadata",
        plan=PlanRecord(
            plan_id="plan-task-metadata",
            run_id="run-task-metadata",
            version=1,
            status="approved",
            created_by="agent",
        ),
        tasks=[
            PlanTaskRecord(
                task_id="t1",
                plan_id="plan-task-metadata",
                run_id="run-task-metadata",
                ordinal=0,
                title="Research",
                tool_kind="web_research",
            )
        ],
    )

    with pytest.raises(ValueError, match="before running"):
        await service.store.transition_plan_task(
            run_id="run-task-metadata",
            plan_id="plan-task-metadata",
            task_id="t1",
            status="skipped",
            child_run_id="run-child-early",
        )
    with pytest.raises(ValueError, match="terminal status"):
        await service.store.transition_plan_task(
            run_id="run-task-metadata",
            plan_id="plan-task-metadata",
            task_id="t1",
            status="running",
            result_summary="Too early.",
        )


@pytest.mark.asyncio
async def test_task_cancellation_is_atomic_idempotent_and_source_only(
    service: AgentControlService,
) -> None:
    plan = PlanRecord(
        plan_id="plan-task-cancel",
        run_id="run-task-cancel",
        version=1,
        status="approved",
        created_by="agent",
    )
    tasks = [
        PlanTaskRecord(
            task_id="pending",
            plan_id=plan.plan_id,
            run_id=plan.run_id,
            ordinal=0,
            title="Pending",
            tool_kind="web_instant",
        ),
        PlanTaskRecord(
            task_id="running",
            plan_id=plan.plan_id,
            run_id=plan.run_id,
            ordinal=1,
            title="Running",
            tool_kind="web_instant",
        ),
        PlanTaskRecord(
            task_id="s",
            plan_id=plan.plan_id,
            run_id=plan.run_id,
            ordinal=2,
            title="Synthesis",
            tool_kind="synthesis",
        ),
    ]
    await service.store.save_plan(run_id=plan.run_id, plan=plan, tasks=tasks)
    await service.store.transition_plan_task(
        run_id=plan.run_id,
        plan_id=plan.plan_id,
        task_id="running",
        status="running",
    )

    pending = await service.store.request_plan_task_cancel(
        run_id=plan.run_id,
        plan_id=plan.plan_id,
        task_id="pending",
    )
    assert pending.status == "cancelled"
    assert pending.result_payload["failure_code"] == "task_cancelled"
    assert (
        await service.store.request_plan_task_cancel(
            run_id=plan.run_id,
            plan_id=plan.plan_id,
            task_id="pending",
        )
    ) == pending

    running = await service.store.request_plan_task_cancel(
        run_id=plan.run_id,
        plan_id=plan.plan_id,
        task_id="running",
    )
    assert running.status == "cancel_requested"
    await settle_terminal_plan_tasks(
        service.store,
        plan.run_id,
        status="cancelled",
    )
    _settled_plan, settled_tasks = await service.store.get_plan(plan.run_id)
    cancelled = next(task for task in settled_tasks if task.task_id == "running")
    assert cancelled.status == "cancelled"
    assert cancelled.result_payload == {
        "failure_code": "task_cancelled",
        "failure_reason": "user_requested_task_cancel",
    }

    with pytest.raises(PlanTaskCancellationConflict):
        await service.store.request_plan_task_cancel(
            run_id=plan.run_id,
            plan_id=plan.plan_id,
            task_id="s",
        )


@pytest.mark.asyncio
async def test_cancel_settlement_folds_child_completed_before_parent_cancel(
    service: AgentControlService,
    run_store: RunStore,
) -> None:
    """The synchronous waiting-run cancel cannot downgrade a finished child."""
    run_id = _parked_agent_run(run_store)
    child = run_store.submit(
        question="Child research",
        stack_name="default",
        work=lambda handle: handle.complete(
            {
                "answer": "Belegte Antwort",
                "references": [{"url": "https://example.test/source"}],
                "top_claims": [],
            }
        ),
        kind="agent_child",
        parent_run_id=run_id,
        root_run_id=run_id,
        request_payload={
            "body": {"parent_task_id": "r", "parent_task_attempt": 1}
        },
    )
    child_id = child["run_id"]
    _wait_until(lambda: run_store.get(child_id)["status"] == "completed")
    await service.store.save_plan(
        run_id=run_id,
        plan=PlanRecord(
            plan_id="plan-cancel-child",
            run_id=run_id,
            version=1,
            status="approved",
            created_by="agent",
        ),
        tasks=[
            PlanTaskRecord(
                task_id="r",
                plan_id="plan-cancel-child",
                run_id=run_id,
                ordinal=0,
                title="Research",
                tool_kind="web_research",
                status="running",
                child_run_id=child_id,
            ),
            PlanTaskRecord(
                task_id="s",
                plan_id="plan-cancel-child",
                run_id=run_id,
                ordinal=1,
                title="Synthesis",
                tool_kind="synthesis",
                depends_on=("r",),
            ),
        ],
    )

    assert run_store.cancel(run_id)["status"] == "cancelled"
    await service.settle_cancelled_tasks(run_id)

    _plan, tasks, _versions = await service.get_plan(run_id)
    assert tasks[0].status == "completed"
    assert tasks[0].result_payload["evidence"] == [
        {"url": "https://example.test/source"}
    ]
    assert tasks[1].status == "skipped"


@pytest.mark.asyncio
async def test_tree_cancel_reconciles_nested_agent_plan_rows(
    service: AgentControlService,
    run_store: RunStore,
) -> None:
    """Immediate child cancellation cannot leave its own plan in progress."""
    root_id = _parked_agent_run(run_store)

    def parked_child(handle: RunHandle) -> None:
        handle.wait("waiting_for_input")

    child = run_store.submit(
        question="Nested mission",
        stack_name="default",
        work=parked_child,
        kind="agent_child",
        parent_run_id=root_id,
        root_run_id=root_id,
        request_payload={"body": {"mode": "workspace_agent"}},
    )
    child_id = child["run_id"]
    _wait_until(
        lambda: run_store.get(child_id)["status"] == "waiting_for_input"
    )
    await service.store.save_plan(
        run_id=child_id,
        plan=PlanRecord(
            plan_id="plan-nested-cancel",
            run_id=child_id,
            version=1,
            status="approved",
            created_by="agent",
        ),
        tasks=[
            PlanTaskRecord(
                task_id="running",
                plan_id="plan-nested-cancel",
                run_id=child_id,
                ordinal=0,
                title="Begonnen",
                tool_kind="web_instant",
                status="running",
            ),
            PlanTaskRecord(
                task_id="pending",
                plan_id="plan-nested-cancel",
                run_id=child_id,
                ordinal=1,
                title="Ausstehend",
                tool_kind="synthesis",
            ),
        ],
    )

    summary, affected = run_store.cancel_tree(root_id)
    assert summary["status"] == "cancelled"
    assert child_id in affected
    await service.reconcile_terminal_run_tree(affected)

    _plan, tasks = await service.store.get_plan(child_id)
    assert [task.status for task in tasks] == ["failed", "skipped"]


@pytest.mark.asyncio
async def test_plan_read_repairs_cancelled_run_after_post_commit_gap(
    service: AgentControlService,
    run_store: RunStore,
) -> None:
    """Reload is the durable recovery when cancel committed before settle."""
    run_id = _parked_agent_run(run_store)
    await service.store.save_plan(
        run_id=run_id,
        plan=PlanRecord(
            plan_id="plan-read-recovery",
            run_id=run_id,
            version=1,
            status="approved",
            created_by="agent",
        ),
        tasks=[
            PlanTaskRecord(
                task_id="pending",
                plan_id="plan-read-recovery",
                run_id=run_id,
                ordinal=0,
                title="Ausstehend",
                tool_kind="synthesis",
            )
        ],
    )
    assert run_store.cancel(run_id)["status"] == "cancelled"
    _raw_plan, raw_tasks = await service.store.get_plan(run_id)
    assert raw_tasks[0].status == "pending"

    _plan, repaired, _versions = await service.get_plan(run_id)
    assert repaired[0].status == "skipped"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [("approve", "approved"), ("reject", "rejected")],
)
async def test_plan_decision_updates_approval_and_plan_together(
    service: AgentControlService,
    run_store: RunStore,
    decision: str,
    expected_status: str,
) -> None:
    run_id = _parked_agent_run(run_store)
    await service.store.save_plan(
        run_id=run_id,
        plan=PlanRecord(
            plan_id="plan-v1",
            run_id=run_id,
            version=1,
            status="proposed",
            created_by="agent",
        ),
        tasks=[],
    )
    approval = await service.store.create_approval(
        ApprovalRecord(
            approval_id=f"apr-{decision}",
            run_id=run_id,
            kind="plan",
            subject_type="plan",
            payload={"plan_version": 1},
        )
    )
    assert approval.subject_id == "plan-v1"

    await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision=decision,
        plan_body=None,
        note="",
        principal=PRINCIPAL,
    )

    plan, _tasks = await service.store.get_plan(run_id)
    assert plan.status == expected_status


@pytest.mark.asyncio
async def test_plan_update_failure_leaves_memory_approval_pending(
    service: AgentControlService,
    run_store: RunStore,
) -> None:
    run_id = _parked_agent_run(run_store)
    await service.store.save_plan(
        run_id=run_id,
        plan=PlanRecord(
            plan_id="plan-missing",
            run_id=run_id,
            version=1,
            status="proposed",
            created_by="agent",
        ),
        tasks=[],
    )
    approval = await service.store.create_approval(
        ApprovalRecord(
            approval_id="apr-missing-plan",
            run_id=run_id,
            kind="plan",
            subject_type="plan",
            payload={"plan_version": 1},
        )
    )
    store = service.store
    assert isinstance(store, MemoryAgentControlStore)
    with store._lock:
        store._plans.pop("plan-missing")
        store._plan_tasks.pop("plan-missing", None)

    with pytest.raises(PlanNotFound):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="approve",
            plan_body=None,
            note="",
            principal=PRINCIPAL,
        )

    stored = await store.get_approval(run_id, approval.approval_id)
    assert stored.status == "pending"
    assert run_store.get(run_id)["status"] == "waiting_for_approval"


@pytest.mark.asyncio
async def test_edited_plan_rejects_client_managed_task_budget(
    service: AgentControlService,
    run_store: RunStore,
) -> None:
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))
    body = _plan_body()
    body["tasks"][0]["budget"] = {"max_tokens": 1800}

    with pytest.raises(AgentControlValidationError) as exc_info:
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=body,
            note="",
            principal=PRINCIPAL,
        )

    assert exc_info.value.error_type == "task_budget_server_managed"
    stored = await service.store.get_approval(run_id, approval.approval_id)
    assert stored.status == "pending"


def _research_edit_plan(profile: str) -> dict[str, Any]:
    return {
        "summary_markdown": "Explicit research",
        "tasks": [
            {
                "id": "r1",
                "title": "Research",
                "tool_kind": "web_research",
                "queries": ["Which evidence answers the question?"],
                "params": {"profile": profile},
            },
            {
                "id": "s",
                "title": "Synthesis",
                "tool_kind": "synthesis",
                "depends_on": ["r1"],
            },
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("depth", "profile", "accepted"),
    [
        ("normal", "compact", True),
        ("normal", "deep", False),
        ("deep", "deep", True),
        ("deep", "compact", False),
    ],
)
async def test_edited_research_plan_uses_server_selected_profile(
    service: AgentControlService,
    run_store: RunStore,
    depth: str,
    profile: str,
    accepted: bool,
) -> None:
    summary = run_store.submit(
        question="Agent assignment",
        stack_name="default",
        work=lambda handle: handle.complete({"answer": "done"}),
        kind="agent",
        agent_overrides={"depth": depth},
    )
    _wait_until(
        lambda: run_store.get(summary["run_id"])["status"] == "completed"
    )
    if accepted:
        _plan, tasks = await service._parse_edited_plan(
            summary["run_id"], _research_edit_plan(profile), None
        )
        assert tasks[0].params["profile"] == profile
        return
    with pytest.raises(AgentControlValidationError) as exc_info:
        await service._parse_edited_plan(
            summary["run_id"], _research_edit_plan(profile), None
        )
    assert any("profile=" in error for error in exc_info.value.errors)


@pytest.mark.asyncio
async def test_edited_plan_preflights_collection_visibility(
    run_store: RunStore, identity: MemoryIdentityStore
) -> None:
    """An edited rag task pointing at an invisible collection is a 400 at
    edit time (E5 semantics, plan §4) — never a silent approve that fails
    mid-run. The caller-visible catalog (THE shared resolver/validator
    rule the M5 planner also runs) is the gate."""

    class CatalogKnowledge:
        def __init__(self) -> None:
            self.calls = 0

        async def list_collections(self, *, visible_to=None):
            self.calls += 1
            return [
                SimpleNamespace(id="kc_visible", name="Marktdaten"),
            ]

    knowledge = CatalogKnowledge()
    service = AgentControlService(
        store=MemoryAgentControlStore(),
        run_store=run_store,
        audit=identity,
        knowledge=knowledge,
        durable=False,
    )
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))
    body = _plan_body()
    body["tasks"][0]["params"] = {
        "profile": "standard",
        "collection_ids": ["col-forbidden"],
    }

    with pytest.raises(AgentControlValidationError) as exc_info:
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=body,
            note="",
            principal=PRINCIPAL,
        )

    assert knowledge.calls == 1
    assert any(
        "col-forbidden" in error for error in exc_info.value.errors
    )
    # Nothing was committed: the approval stays pending, no plan version.
    stored = await service.store.get_approval(run_id, approval.approval_id)
    assert stored.status == "pending"
    with pytest.raises(PlanNotFound):
        await service.store.get_plan(run_id)


@pytest.mark.asyncio
async def test_edited_plan_canonicalizes_collection_names(
    run_store: RunStore, identity: MemoryIdentityStore
) -> None:
    """A user edit naming a collection ("EU-AI-Act") saves the plan with
    the canonical id — the exact reference shape retrieval expects, so
    the approved plan can never fail as a raw unknown-collection error."""

    class CatalogKnowledge:
        async def list_collections(self, *, visible_to=None):
            return [
                SimpleNamespace(id="kc_18d4", name="EU-AI-Act"),
            ]

    service = AgentControlService(
        store=MemoryAgentControlStore(),
        run_store=run_store,
        audit=identity,
        knowledge=CatalogKnowledge(),
        durable=False,
    )
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))
    body = _plan_body()
    body["tasks"][0]["params"] = {
        "profile": "standard",
        "collection_ids": ["EU-AI-Act"],
    }

    await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="edit",
        plan_body=body,
        note="",
        principal=PRINCIPAL,
    )

    _plan, tasks = await service.store.get_plan(run_id)
    assert tasks[0].params["collection_ids"] == ["kc_18d4"]


@pytest.mark.asyncio
async def test_edit_requires_plan_kind_and_plan_body(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    discovery = await service.store.create_approval(
        _approval(run_id, kind="discovery")
    )
    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=discovery.approval_id,
            decision="edit",
            plan_body=_plan_body(),
            note="",
            principal=PRINCIPAL,
        )
    plan_approval = await service.store.create_approval(_approval(run_id))
    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=plan_approval.approval_id,
            decision="edit",
            plan_body=None,
            note="",
            principal=PRINCIPAL,
        )
    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=plan_approval.approval_id,
            decision="approve",
            plan_body=_plan_body(),
            note="",
            principal=PRINCIPAL,
        )


# -- tool approvals (kernel policy gates, M2) ----------------------------- #


def _tool_approval(run_id: str) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=f"apr_{uuid.uuid4().hex[:8]}",
        run_id=run_id,
        kind="tool",
        payload={
            "actions": [
                {
                    "tool": "web_instant",
                    "args": {"query": "EU AI Act Fristen"},
                    "summary": "Websuche",
                }
            ]
        },
    )


@pytest.mark.asyncio
async def test_tool_edit_replaces_args_and_resumes(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_tool_approval(run_id))

    decided, summary, replayed = await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="edit",
        plan_body=None,
        note="",
        principal=PRINCIPAL,
        actions_body=[
            {"tool": "web_instant", "args": {"query": "EU AI Act 2027"}}
        ],
    )

    assert not replayed
    assert decided.status == "edited"
    assert decided.decision_payload == {
        "actions": [
            {"tool": "web_instant", "args": {"query": "EU AI Act 2027"}}
        ]
    }
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")


@pytest.mark.asyncio
async def test_tool_edit_validation_matrix(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_tool_approval(run_id))

    # Tool swap is not an edit — the gate approved THIS tool.
    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=None,
            note="",
            principal=PRINCIPAL,
            actions_body=[{"tool": "rag_query", "args": {"query": "x"}}],
        )
    # Exactly one action, args must be an object.
    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=None,
            note="",
            principal=PRINCIPAL,
            actions_body=[],
        )
    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=None,
            note="",
            principal=PRINCIPAL,
            actions_body=[{"tool": "web_instant", "args": "nope"}],
        )
    # A plan body belongs to plan approvals, actions to tool approvals.
    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=_plan_body(),
            note="",
            principal=PRINCIPAL,
        )
    plan_approval = await service.store.create_approval(_approval(run_id))
    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=plan_approval.approval_id,
            decision="edit",
            plan_body=_plan_body(),
            note="",
            principal=PRINCIPAL,
            actions_body=[{"tool": "web_instant", "args": {}}],
        )
    # actions without decision=edit is a 400, mirroring the plan rule.
    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="approve",
            plan_body=None,
            note="",
            principal=PRINCIPAL,
            actions_body=[{"tool": "web_instant", "args": {}}],
        )


@pytest.mark.asyncio
async def test_tool_edit_replay_same_actions_is_idempotent(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_tool_approval(run_id))
    edited = [{"tool": "web_instant", "args": {"query": "praeziser"}}]

    await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="edit",
        plan_body=None,
        note="",
        principal=PRINCIPAL,
        actions_body=edited,
    )
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")

    replay, _summary, replayed = await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="edit",
        plan_body=None,
        note="",
        principal=PRINCIPAL,
        actions_body=edited,
    )
    assert replayed and replay.status == "edited"

    with pytest.raises(ApprovalAlreadyDecided):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=None,
            note="",
            principal=PRINCIPAL,
            actions_body=[{"tool": "web_instant", "args": {"query": "anders"}}],
        )


# -- clarifications ------------------------------------------------------- #


@pytest.mark.asyncio
async def test_clarification_answer_resumes_and_validates_options(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    clarification = await service.store.create_clarification(
        ClarificationRecord(
            clarification_id=f"clr_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            question="Welcher Zeitraum?",
            options=(
                {"id": "q1", "label": "Q1"},
                {"id": "q2", "label": "Q2"},
            ),
            default_assumption="Q1",
        )
    )

    with pytest.raises(AgentControlValidationError):
        await service.answer_clarification(
            run_id=run_id,
            clarification_id=clarification.clarification_id,
            answer="Q1",
            option_id="q1",
            principal=PRINCIPAL,
        )
    with pytest.raises(AgentControlValidationError):
        await service.answer_clarification(
            run_id=run_id,
            clarification_id=clarification.clarification_id,
            answer=None,
            option_id="ghost",
            principal=PRINCIPAL,
        )

    answered, summary, replayed = await service.answer_clarification(
        run_id=run_id,
        clarification_id=clarification.clarification_id,
        answer=None,
        option_id="q2",
        principal=PRINCIPAL,
    )
    assert not replayed
    assert answered.status == "answered"
    assert answered.option_id == "q2"
    assert summary["status"] in {"queued", "running", "completed"}
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")

    # Replay of the same answer is idempotent.
    _answered, _summary, replayed = await service.answer_clarification(
        run_id=run_id,
        clarification_id=clarification.clarification_id,
        answer=None,
        option_id="q2",
        principal=PRINCIPAL,
    )
    assert replayed


# -- artifacts -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sessionless_deliverables_are_not_implicit_singletons(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    first = await service.store.upsert_artifact(
        run_id=run_id,
        kind="deliverable",
        session_id=None,
        title="One",
        status="ready",
        content_markdown="# One",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_sessionless_one",
    )
    second = await service.store.upsert_artifact(
        run_id=run_id,
        kind="deliverable",
        session_id=None,
        title="Two",
        status="ready",
        content_markdown="# Two",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_sessionless_two",
    )

    assert first.artifact_id != second.artifact_id
    rows, _cursor = await service.store.list_artifacts(
        run_id, kind="deliverable", limit=10
    )
    assert {row.artifact_id for row in rows} == {
        first.artifact_id,
        second.artifact_id,
    }


@pytest.mark.asyncio
async def test_artifact_upsert_versioning_and_user_edit_matrix(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    store = service.store
    first = await store.upsert_artifact(
        run_id=run_id,
        kind="memo",
        session_id="sess-1",
        title="Memo",
        status="ready",
        content_markdown="# V1",
        payload={},
        refs=[{"label": "K1"}],
        updated_by="agent",
        artifact_id="art_memo",
    )
    assert first.revision == 1
    # Session upsert re-anchors to the same artifact, bumping revision.
    second = await store.upsert_artifact(
        run_id=run_id,
        kind="memo",
        session_id="sess-1",
        title="Memo",
        status="ready",
        content_markdown="# V2",
        payload={},
        refs=[],
        updated_by="agent",
    )
    assert second.artifact_id == "art_memo"
    assert second.revision == 2

    with pytest.raises(ArtifactRevisionConflict) as conflict:
        await service.user_update_artifact(
            run_id=run_id,
            artifact_id="art_memo",
            content_markdown="# stale",
            expected_revision=1,
            principal=PRINCIPAL,
        )
    assert conflict.value.current_revision == 2

    edited = await service.user_update_artifact(
        run_id=run_id,
        artifact_id="art_memo",
        content_markdown="# V3 vom Nutzer",
        expected_revision=2,
        principal=PRINCIPAL,
    )
    assert edited.revision == 3
    assert edited.updated_by == "user"

    # Writing status locks user edits (E13).
    await store.upsert_artifact(
        run_id=run_id,
        kind="memo",
        session_id="sess-1",
        title="Memo",
        status="writing",
        content_markdown="# V4 agent schreibt",
        payload={},
        refs=[],
        updated_by="agent",
    )
    with pytest.raises(ArtifactLocked):
        await service.user_update_artifact(
            run_id=run_id,
            artifact_id="art_memo",
            content_markdown="# waehrend writing",
            expected_revision=4,
            principal=PRINCIPAL,
        )
    with pytest.raises(ArtifactNotFound):
        await service.user_update_artifact(
            run_id=run_id,
            artifact_id="art_unbekannt",
            content_markdown="x",
            expected_revision=1,
            principal=PRINCIPAL,
        )

    artifact, revisions = await store.get_artifact(run_id, "art_memo")
    assert artifact.revision == 4
    assert [row.revision for row in revisions] == [4, 3, 2, 1]
    assert all(row.content_markdown == "" for row in revisions)
    old, _ = await store.get_artifact(run_id, "art_memo", revision=2)
    assert old.content_markdown == "# V2"
    with pytest.raises(ArtifactNotFound):
        await store.get_artifact(run_id, "art_memo", revision=99)


@pytest.mark.asyncio
async def test_session_memo_lineage_read_and_agent_cas(
    service: AgentControlService, run_store: RunStore
) -> None:
    """E15 cross-run read + E13 agent-side CAS on the memo artifact.

    Turn 1 writes the session memo; the user then edits it. A follow-up
    turn (a DIFFERENT run) must READ that latest user revision by session
    (get_session_artifact, run-agnostic) and its guarded write must REFUSE
    to advance from the pre-edit revision (ArtifactRevisionConflict) while
    succeeding from the current one.
    """
    store = service.store
    run1 = _parked_agent_run(run_store)
    await store.upsert_artifact(
        run_id=run1,
        kind="memo",
        session_id="sess-lin",
        title="Memo",
        status="ready",
        content_markdown="# Turn 1",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_lin_memo",
    )
    edited = await service.user_update_artifact(
        run_id=run1,
        artifact_id="art_lin_memo",
        content_markdown="# Turn 1 mit Nutzer-Edit",
        expected_revision=1,
        principal=PRINCIPAL,
    )
    assert edited.revision == 2

    # The follow-up run cannot see the memo run-scoped (different run)...
    run_store.cancel(run1)
    assert run_store.get(run1)["status"] == "cancelled"
    run2 = _parked_agent_run(run_store)
    with pytest.raises(ArtifactNotFound):
        await store.get_artifact(run2, "art_lin_memo")
    # ...but the session read reaches the LATEST user revision (E15).
    prior = await store.get_session_artifact("sess-lin", "memo")
    assert prior is not None
    assert prior.revision == 2
    assert prior.content_markdown == "# Turn 1 mit Nutzer-Edit"

    # A guarded write from the STALE revision the agent might have cached
    # is refused instead of clobbering the user's edit (E13/R10)...
    with pytest.raises(ArtifactRevisionConflict) as conflict:
        await store.upsert_artifact(
            run_id=run2,
            kind="memo",
            session_id="sess-lin",
            title="Memo",
            status="writing",
            content_markdown="# Turn 2 (stale base)",
            payload={},
            refs=[],
            updated_by="agent",
            artifact_id="art_lin_memo",
            expected_revision=1,
        )
    assert conflict.value.current_revision == 2
    # ...and succeeds from the revision the agent actually read.
    advanced = await store.upsert_artifact(
        run_id=run2,
        kind="memo",
        session_id="sess-lin",
        title="Memo",
        status="writing",
        content_markdown="# Turn 2 (fortgesetzt)",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_lin_memo",
        expected_revision=2,
    )
    assert advanced.revision == 3
    # An unguarded write (expected_revision=None) stays a blind bump — the
    # write-once per-run kinds (evidence, critic) rely on that default.
    blind = await store.upsert_artifact(
        run_id=run2,
        kind="memo",
        session_id="sess-lin",
        title="Memo",
        status="ready",
        content_markdown="# Turn 2 final",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_lin_memo",
    )
    assert blind.revision == 4
    # expected_revision=0 means "expect to create": a row that already
    # exists is a conflict, never a silent clobber (fresh-session race).
    with pytest.raises(ArtifactRevisionConflict) as insert_conflict:
        await store.upsert_artifact(
            run_id=run2,
            kind="memo",
            session_id="sess-lin",
            title="Memo",
            status="ready",
            content_markdown="# would clobber",
            payload={},
            refs=[],
            updated_by="agent",
            artifact_id="art_lin_memo",
            expected_revision=0,
        )
    assert insert_conflict.value.current_revision == 4


@pytest.mark.asyncio
async def test_artifact_listing_paginates_and_filters(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    store = service.store
    for index in range(3):
        await store.upsert_artifact(
            run_id=run_id,
            kind="evidence_bundle" if index else "memo",
            session_id=None,
            title=f"A{index}",
            status="ready",
            content_markdown="body",
            payload={},
            refs=[],
            updated_by="agent",
            artifact_id=f"art_{index}",
        )
        await asyncio.sleep(0.01)

    page, cursor = await store.list_artifacts(run_id, limit=2)
    assert len(page) == 2
    assert cursor is not None
    assert all(row.content_markdown == "" for row in page)
    from inqtrix.pagination import decode_cursor

    rest, cursor2 = await store.list_artifacts(
        run_id, limit=2, after=decode_cursor(cursor)
    )
    assert len(rest) == 1
    assert cursor2 is None
    memos, _ = await store.list_artifacts(run_id, kind="memo", limit=10)
    assert [row.artifact_id for row in memos] == ["art_0"]


@pytest.mark.asyncio
async def test_export_without_editor_persistence_fails_loudly(
    service: AgentControlService, run_store: RunStore
) -> None:
    run_id = _parked_agent_run(run_store)
    await service.store.upsert_artifact(
        run_id=run_id,
        kind="memo",
        session_id=None,
        title="Memo",
        status="ready",
        content_markdown="# Inhalt",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_memo",
    )
    with pytest.raises(AgentControlUnavailable):
        await service.export_artifact(
            run_id=run_id,
            artifact_id="art_memo",
            title=None,
            folder_id=None,
            principal=PRINCIPAL,
            caller_sub="owner-1",
            workspace_id=None,
        )


@pytest.mark.asyncio
async def test_concurrent_decides_yield_exactly_one_resume(
    service: AgentControlService, run_store: RunStore
) -> None:
    """Two racing decisions: one wins, the other replays or conflicts."""
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))
    results: list[Any] = []
    errors: list[BaseException] = []

    async def _decide() -> None:
        try:
            results.append(
                await service.decide_approval(
                    run_id=run_id,
                    approval_id=approval.approval_id,
                    decision="approve",
                    plan_body=None,
                    note="",
                    principal=PRINCIPAL,
                )
            )
        except BaseException as exc:  # noqa: BLE001 — collected below
            errors.append(exc)

    def _run_decide() -> None:
        asyncio.run(_decide())

    threads = [threading.Thread(target=_run_decide) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors, errors
    assert len(results) == 2
    # Exactly one actually decided; the other took the replay path.
    assert sorted(result[2] for result in results) == [False, True]
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")


@pytest.mark.asyncio
async def test_edit_replay_with_different_plan_conflicts(
    service: AgentControlService, run_store: RunStore
) -> None:
    """The plan payload IS the edit decision: a retry with a DIFFERENT
    plan must 409, never be swallowed as an idempotent replay."""
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))
    await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="edit",
        plan_body=_plan_body(),
        note="",
        principal=PRINCIPAL,
    )

    # Same verb, same plan -> idempotent replay.
    _decided, _summary, replayed = await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="edit",
        plan_body=_plan_body(),
        note="",
        principal=PRINCIPAL,
    )
    assert replayed

    different = _plan_body()
    different["summary_markdown"] = "Ein ganz anderer Plan"
    with pytest.raises(ApprovalAlreadyDecided):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=different,
            note="",
            principal=PRINCIPAL,
        )
    # The differing plan was NOT applied: still exactly one version.
    _plan, _tasks, versions = await service.get_plan(run_id)
    assert len(versions) == 1


@pytest.mark.asyncio
async def test_failed_edit_resume_restores_prior_plan_status(
    service: AgentControlService, run_store: RunStore
) -> None:
    """Memory revert must undo the supersede flip too (Postgres gets it
    from the transaction rollback — lockstep on the failure path)."""
    summary = run_store.submit(
        question="normal",
        stack_name="default",
        work=lambda handle: handle.complete({}),
        kind="agent",
    )
    run_id = summary["run_id"]
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")
    await service.store.save_plan(
        run_id=run_id,
        plan=PlanRecord(
            plan_id="plan_v1",
            run_id=run_id,
            version=1,
            status="proposed",
            created_by="agent",
        ),
        tasks=[],
    )
    approval = await service.store.create_approval(_approval(run_id))

    with pytest.raises(RunActive):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body=_plan_body(),
            note="",
            principal=PRINCIPAL,
        )

    stored = await service.store.get_approval(run_id, approval.approval_id)
    assert stored.status == "pending"
    plan, _tasks, versions = await service.get_plan(run_id)
    assert plan.version == 1
    assert plan.status == "proposed"
    assert len(versions) == 1
