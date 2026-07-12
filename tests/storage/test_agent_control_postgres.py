"""Postgres integration tests for the agent control store (gated suite).

Lockstep with the memory tier in ``tests/test_agent_control.py``: same
state machines, same error types. The R9 one-transaction property (the
approval decision commits or rolls back TOGETHER with the run's
``waiting -> queued`` flip) can only be proven here.
"""

from __future__ import annotations

import asyncio
from importlib import import_module
import os
import threading
import time
import uuid
from typing import Callable

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.agents.control_ports import (
    ApprovalRecord,
    ArtifactBatchRevision,
    ArtifactLocked,
    ArtifactRevisionConflict,
    ClarificationRecord,
    PlanNotFound,
    PlanRecord,
    PlanTaskRecord,
    settle_terminal_plan_tasks,
)
from inqtrix.runs.postgres_store import PostgresRunStore
from inqtrix.server.runs import RunActive
from inqtrix.services.agent_control_service import (
    AgentControlService,
    AgentControlValidationError,
)
from inqtrix.storage.agent_control_postgres import PostgresAgentControlStore
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"

_MIGRATION_0043 = import_module(
    "inqtrix.storage.migrations.versions."
    "0043_agent_task_execution_contract"
)


@pytest.fixture(scope="session", autouse=True)
def schema_migrated():
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture()
async def wiped():
    engine = build_engine(TEST_DATABASE_URL)
    factory = build_session_factory(engine)
    async with factory() as session:
        async with session.begin():
            bypasses = (
                await session.execute(
                    text(
                        "SELECT rolsuper OR rolbypassrls FROM pg_roles "
                        "WHERE rolname = current_user"
                    )
                )
            ).scalar_one()
            if not bypasses:
                pytest.fail(
                    "INQTRIX_TEST_DATABASE_URL must connect as a "
                    "superuser/BYPASSRLS user for cross-tenant cleanup."
                )
            # Control rows cascade with their runs.
            await session.execute(text("DELETE FROM run_events"))
            await session.execute(text("DELETE FROM runs"))
            await session.execute(text("DELETE FROM agent_sessions"))
            await session.execute(text("DELETE FROM agent_session_groups"))
    await engine.dispose()
    yield


@pytest.fixture()
def run_store(wiped):
    store = PostgresRunStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=2,
        max_queue_size=10,
        completed_ttl_seconds=300,
        worker_id="pytest-agent-control",
    )
    yield store
    store.close()


@pytest_asyncio.fixture()
async def control_store(wiped):
    store = PostgresAgentControlStore(
        engine=build_engine(TEST_DATABASE_URL), app_role=APP_ROLE
    )
    yield store
    await store.aclose()


@pytest.fixture()
def service(control_store, run_store):
    return AgentControlService(
        store=control_store,
        run_store=run_store,
        audit=None,
        editor_persistence=None,
        durable=True,
    )


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail("condition was not reached before timeout")


def _parked_agent_run(
    run_store: PostgresRunStore,
    *,
    resumed_release: threading.Event | None = None,
    session_id: str = "sess-pg",
) -> str:
    calls = {"count": 0}

    def segmented(handle) -> None:
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
        request_payload={"question": "x", "body": {"mode": "research"}},
        kind="agent",
        session_id=session_id,
    )
    run_id = summary["run_id"]
    _wait_until(
        lambda: run_store.get(run_id)["status"] == "waiting_for_approval"
    )
    return run_id


def _approval(run_id: str, *, kind: str = "plan") -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=f"apr_{uuid.uuid4().hex[:8]}", run_id=run_id, kind=kind
    )


@pytest.mark.asyncio
async def test_decision_and_resume_commit_together(service, run_store):
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
            note="",
            principal=None,
        )
    finally:
        resumed_release.set()

    assert not replayed
    assert decided.status == "approved"
    assert summary["status"] in {"queued", "running", "completed"}
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")

    events = run_store.subscribe(run_id)
    try:
        decision_event = next(
            event
            for event in events.replay
            if event["type"] == "inqtrix.agent.approval.decided"
        )
        resumed = [
            event
            for event in events.replay
            if event["type"] == "inqtrix.run.queued"
            and event["data"].get("resumed")
        ]
        assert len(resumed) == 1
        assert resumed[0]["sequence"] < decision_event["sequence"]
    finally:
        events.close()


@pytest.mark.asyncio
async def test_plan_decision_updates_plan_status_in_resume_transaction(
    service, run_store
):
    run_id = _parked_agent_run(run_store)
    await service.store.save_plan(
        run_id=run_id,
        plan=PlanRecord(
            plan_id="plan-pg-v1",
            run_id=run_id,
            version=1,
            status="proposed",
            created_by="agent",
        ),
        tasks=[],
    )
    approval = await service.store.create_approval(
        ApprovalRecord(
            approval_id="apr-plan-pg",
            run_id=run_id,
            kind="plan",
            subject_type="plan",
            payload={"plan_version": 1},
        )
    )
    assert approval.subject_id == "plan-pg-v1"

    await service.decide_approval(
        run_id=run_id,
        approval_id=approval.approval_id,
        decision="approve",
        plan_body=None,
        note="",
        principal=None,
    )

    plan, _tasks = await service.store.get_plan(run_id)
    assert plan.status == "approved"


@pytest.mark.asyncio
async def test_migration_backfills_legacy_decided_plan_subject(
    service, run_store
):
    run_id = _parked_agent_run(run_store)
    await service.store.save_plan(
        run_id=run_id,
        plan=PlanRecord(
            plan_id="plan-legacy-subject",
            run_id=run_id,
            version=1,
            status="proposed",
            created_by="agent",
        ),
        tasks=[],
    )
    approval = await service.store.create_approval(
        ApprovalRecord(
            approval_id="apr-legacy-subject",
            run_id=run_id,
            kind="plan",
            subject_type="plan",
            payload={"plan_version": 1},
        )
    )
    engine = build_engine(TEST_DATABASE_URL)
    factory = build_session_factory(engine)
    try:
        async with factory() as session:
            async with session.begin():
                await session.execute(
                    text(
                        "UPDATE run_approvals SET subject_id = '', "
                        "status = 'approved', decision = 'approve' "
                        "WHERE approval_id = :approval_id"
                    ),
                    {"approval_id": approval.approval_id},
                )
                await session.execute(
                    text(_MIGRATION_0043._APPROVAL_SUBJECT_BACKFILL_SQL)
                )
                subject_id = (
                    await session.execute(
                        text(
                            "SELECT subject_id FROM run_approvals "
                            "WHERE approval_id = :approval_id"
                        ),
                        {"approval_id": approval.approval_id},
                    )
                ).scalar_one()
    finally:
        await engine.dispose()

    assert subject_id == "plan-legacy-subject"


@pytest.mark.asyncio
async def test_task_transition_retry_and_terminal_contract_postgres(
    service, run_store
):
    run_id = _parked_agent_run(run_store)
    await service.store.save_plan(
        run_id=run_id,
        plan=PlanRecord(
            plan_id="plan-task-pg",
            run_id=run_id,
            version=1,
            status="approved",
            created_by="agent",
        ),
        tasks=[
            PlanTaskRecord(
                task_id="task-pg",
                plan_id="plan-task-pg",
                run_id=run_id,
                ordinal=0,
                title="Research",
                tool_kind="web_research",
            )
        ],
    )
    await service.store.transition_plan_task(
        run_id=run_id,
        plan_id="plan-task-pg",
        task_id="task-pg",
        status="running",
        child_run_id="child-pg-1",
    )
    retry = await service.store.transition_plan_task(
        run_id=run_id,
        plan_id="plan-task-pg",
        task_id="task-pg",
        status="running",
        child_run_id="child-pg-2",
    )
    assert retry.child_run_id == "child-pg-2"
    terminal = await service.store.transition_plan_task(
        run_id=run_id,
        plan_id="plan-task-pg",
        task_id="task-pg",
        status="insufficient_evidence",
        child_run_id="child-pg-2",
        result_summary="One source remained after verification.",
        result_payload={
            "claims": [{"text": "One source remained."}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        },
    )
    assert terminal.status == "insufficient_evidence"
    assert terminal.result_payload["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 1,
    }
    with pytest.raises(ValueError):
        await service.store.transition_plan_task(
            run_id=run_id,
            plan_id="plan-task-pg",
            task_id="task-pg",
            status="completed",
        )


@pytest.mark.asyncio
async def test_task_cancellation_state_machine_postgres(service, run_store):
    run_id = _parked_agent_run(run_store)
    plan = PlanRecord(
        plan_id="plan-task-cancel-pg",
        run_id=run_id,
        version=1,
        status="approved",
        created_by="agent",
    )
    await service.store.save_plan(
        run_id=run_id,
        plan=plan,
        tasks=[
            PlanTaskRecord(
                task_id="pending-pg",
                plan_id=plan.plan_id,
                run_id=run_id,
                ordinal=0,
                title="Pending",
                tool_kind="web_instant",
            ),
            PlanTaskRecord(
                task_id="running-pg",
                plan_id=plan.plan_id,
                run_id=run_id,
                ordinal=1,
                title="Running",
                tool_kind="web_instant",
            ),
        ],
    )
    await service.store.transition_plan_task(
        run_id=run_id,
        plan_id=plan.plan_id,
        task_id="running-pg",
        status="running",
    )

    pending = await service.store.request_plan_task_cancel(
        run_id=run_id,
        plan_id=plan.plan_id,
        task_id="pending-pg",
    )
    running = await service.store.request_plan_task_cancel(
        run_id=run_id,
        plan_id=plan.plan_id,
        task_id="running-pg",
    )

    assert pending.status == "cancelled"
    assert pending.result_payload["failure_code"] == "task_cancelled"
    assert running.status == "cancel_requested"
    await settle_terminal_plan_tasks(service.store, run_id, status="cancelled")
    _settled_plan, settled_tasks = await service.store.get_plan(run_id)
    settled = next(task for task in settled_tasks if task.task_id == "running-pg")
    assert settled.status == "cancelled"
    assert settled.result_payload == {
        "failure_code": "task_cancelled",
        "failure_reason": "user_requested_task_cancel",
    }


@pytest.mark.asyncio
async def test_decision_rolls_back_when_run_is_not_waiting(service, run_store):
    """R9: the decision CAS lives inside the resume transaction — a run
    that is no longer waiting leaves the approval untouched."""
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))
    run_store.cancel(run_id)
    _wait_until(lambda: run_store.get(run_id)["status"] == "cancelled")

    with pytest.raises(RunActive):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="approve",
            plan_body=None,
            note="",
            principal=None,
        )

    stored = await service.store.get_approval(run_id, approval.approval_id)
    assert stored.status == "pending"
    events = run_store.subscribe(run_id)
    try:
        assert "inqtrix.agent.approval.decided" not in {
            event["type"] for event in events.replay
        }
    finally:
        events.close()


@pytest.mark.asyncio
async def test_edit_decision_writes_plan_in_the_same_transaction(
    service, run_store
):
    run_id = _parked_agent_run(run_store)
    await service.store.save_plan(
        run_id=run_id,
        plan=PlanRecord(
            plan_id="plan_pg_v1",
            run_id=run_id,
            version=1,
            status="proposed",
            created_by="agent",
        ),
        tasks=[
            PlanTaskRecord(
                task_id="t1",
                plan_id="plan_pg_v1",
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
        plan_body={
            "summary_markdown": "Neu",
            "tasks": [
                {
                    "id": "t1",
                    "title": "Suche",
                    "tool_kind": "rag_query",
                    "queries": ["Welche belastbare Evidenz liegt vor?"],
                },
                {
                    "id": "s",
                    "title": "Synthese",
                    "tool_kind": "synthesis",
                    "depends_on": ["t1"],
                },
            ],
        },
        note="",
        principal=None,
    )

    assert decided.status == "edited"
    plan, tasks, versions = await service.get_plan(run_id)
    assert plan.version == 2
    assert plan.created_by == "user"
    assert [task.task_id for task in tasks] == ["t1", "s"]
    assert [(v.version, v.status) for v in versions] == [
        (2, "approved"),
        (1, "superseded"),
    ]
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")


@pytest.mark.asyncio
async def test_invalid_edit_leaves_everything_pending(service, run_store):
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))

    with pytest.raises(AgentControlValidationError):
        await service.decide_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
            decision="edit",
            plan_body={"tasks": [{"id": "t1", "title": "x", "tool_kind": "rag_query"}]},
            note="",
            principal=None,
        )

    stored = await service.store.get_approval(run_id, approval.approval_id)
    assert stored.status == "pending"
    assert run_store.get(run_id)["status"] == "waiting_for_approval"
    with pytest.raises(PlanNotFound):
        await service.store.get_plan(run_id)


@pytest.mark.asyncio
async def test_clarification_answer_resumes(service, run_store):
    run_id = _parked_agent_run(run_store)
    clarification = await service.store.create_clarification(
        ClarificationRecord(
            clarification_id=f"clr_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            question="Zeitraum?",
            options=({"id": "q1", "label": "Q1"},),
        )
    )

    answered, _summary, replayed = await service.answer_clarification(
        run_id=run_id,
        clarification_id=clarification.clarification_id,
        answer="Das erste Quartal",
        option_id=None,
        principal=None,
    )

    assert not replayed
    assert answered.status == "answered"
    assert answered.answer == "Das erste Quartal"
    _wait_until(lambda: run_store.get(run_id)["status"] == "completed")

    # Replay of the same answer is idempotent, a different one conflicts.
    _answered, _summary2, replayed = await service.answer_clarification(
        run_id=run_id,
        clarification_id=clarification.clarification_id,
        answer="Das erste Quartal",
        option_id=None,
        principal=None,
    )
    assert replayed


@pytest.mark.asyncio
async def test_artifact_matrix_and_revisions(service, run_store, control_store):
    run_id = _parked_agent_run(run_store)
    await control_store.upsert_artifact(
        run_id=run_id,
        kind="memo",
        session_id="sess-pg",
        title="Memo",
        status="ready",
        content_markdown="# V1",
        payload={},
        refs=[{"label": "K1"}],
        updated_by="agent",
        artifact_id="art_pg",
    )
    # Session upsert without id finds the memo and bumps the revision.
    bumped = await control_store.upsert_artifact(
        run_id=run_id,
        kind="memo",
        session_id="sess-pg",
        title="Memo",
        status="ready",
        content_markdown="# V2",
        payload={},
        refs=[],
        updated_by="agent",
    )
    assert bumped.artifact_id == "art_pg"
    assert bumped.revision == 2

    with pytest.raises(ArtifactRevisionConflict) as conflict:
        await control_store.user_update_artifact(
            run_id=run_id,
            artifact_id="art_pg",
            content_markdown="# stale",
            expected_revision=1,
        )
    assert conflict.value.current_revision == 2

    edited = await control_store.user_update_artifact(
        run_id=run_id,
        artifact_id="art_pg",
        content_markdown="# V3",
        expected_revision=2,
    )
    assert edited.revision == 3
    assert edited.updated_by == "user"

    await control_store.upsert_artifact(
        run_id=run_id,
        kind="memo",
        session_id="sess-pg",
        title="Memo",
        status="writing",
        content_markdown="# V4",
        payload={},
        refs=[],
        updated_by="agent",
    )
    with pytest.raises(ArtifactLocked):
        await control_store.user_update_artifact(
            run_id=run_id,
            artifact_id="art_pg",
            content_markdown="# x",
            expected_revision=4,
        )

    artifact, revisions = await control_store.get_artifact(run_id, "art_pg")
    assert artifact.revision == 4
    assert [row.revision for row in revisions] == [4, 3, 2, 1]
    old, _ = await control_store.get_artifact(run_id, "art_pg", revision=1)
    assert old.content_markdown == "# V1"

    page, cursor = await control_store.list_artifacts(run_id, limit=10)
    assert [row.artifact_id for row in page] == ["art_pg"]
    assert cursor is None
    assert page[0].content_markdown == ""


@pytest.mark.asyncio
async def test_concurrent_first_memo_insert_surfaces_conflict_not_integrityerror(
    run_store, control_store
):
    """A3: two runs of the same session doing a first-ever memo write must not
    abort with a raw IntegrityError.

    ``expected_revision=0`` promises "I expect to CREATE; a concurrently
    inserted row is a conflict." The insert branch ignored that and raised a
    raw IntegrityError on the partial-unique/PK collision, which _flush_memo's
    reconcile loop (catches only ArtifactRevisionConflict) could not absorb.
    Now the loser gets a reconcilable ArtifactRevisionConflict whether it lost
    the insert race (ON CONFLICT DO NOTHING) or found the row already present
    (the CAS guard) — never a 500. A few rounds raise the odds of the true
    interleave; the invariant (exactly one success, the other a conflict, no
    raw error) must hold every round.
    """
    for round_index in range(6):
        run_id = _parked_agent_run(
            run_store,
            session_id=f"sess-race-run-{round_index}",
        )
        artifact_id = f"art_race_{round_index}_memo"
        session_id = f"sess-race-{round_index}"

        async def first_write(marker: str):
            return await control_store.upsert_artifact(
                run_id=run_id,
                kind="memo",
                session_id=session_id,
                title="Memo",
                status="writing",
                content_markdown=marker,
                payload={},
                refs=[],
                updated_by="agent",
                artifact_id=artifact_id,
                expected_revision=0,
            )

        results = await asyncio.gather(
            first_write("# A"), first_write("# B"), return_exceptions=True
        )
        oks = [r for r in results if not isinstance(r, Exception)]
        raw = [
            r
            for r in results
            if isinstance(r, Exception)
            and not isinstance(r, ArtifactRevisionConflict)
        ]
        assert raw == [], f"round {round_index}: raw error escaped: {raw}"
        assert len(oks) == 1, f"round {round_index}: {results}"

        artifact, _ = await control_store.get_artifact(run_id, artifact_id)
        assert artifact.revision == 1


@pytest.mark.asyncio
async def test_session_memo_lineage_and_agent_cas_on_postgres(
    service, run_store, control_store
):
    """Postgres lockstep for E15 cross-run read + E13 agent-side CAS."""
    run1 = _parked_agent_run(run_store, session_id="sess-pg-lin")
    await control_store.upsert_artifact(
        run_id=run1,
        kind="memo",
        session_id="sess-pg-lin",
        title="Memo",
        status="ready",
        content_markdown="# Turn 1",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_pg_lin",
    )
    edited = await control_store.user_update_artifact(
        run_id=run1,
        artifact_id="art_pg_lin",
        content_markdown="# Turn 1 + Nutzer",
        expected_revision=1,
    )
    assert edited.revision == 2

    prior = await control_store.get_session_artifact("sess-pg-lin", "memo")
    assert prior is not None
    assert prior.revision == 2
    assert prior.content_markdown == "# Turn 1 + Nutzer"

    run_store.cancel(run1)
    _wait_until(lambda: run_store.get(run1)["status"] == "cancelled")
    run2 = _parked_agent_run(run_store, session_id="sess-pg-lin")
    with pytest.raises(ArtifactRevisionConflict) as conflict:
        await control_store.upsert_artifact(
            run_id=run2,
            kind="memo",
            session_id="sess-pg-lin",
            title="Memo",
            status="writing",
            content_markdown="# stale base",
            payload={},
            refs=[],
            updated_by="agent",
            artifact_id="art_pg_lin",
            expected_revision=1,
        )
    assert conflict.value.current_revision == 2
    advanced = await control_store.upsert_artifact(
        run_id=run2,
        kind="memo",
        session_id="sess-pg-lin",
        title="Memo",
        status="writing",
        content_markdown="# Turn 2",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_pg_lin",
        expected_revision=2,
    )
    assert advanced.revision == 3
    # expected_revision=0 = "expect to create": an existing row conflicts.
    with pytest.raises(ArtifactRevisionConflict) as insert_conflict:
        await control_store.upsert_artifact(
            run_id=run2,
            kind="memo",
            session_id="sess-pg-lin",
            title="Memo",
            status="ready",
            content_markdown="# would clobber",
            payload={},
            refs=[],
            updated_by="agent",
            artifact_id="art_pg_lin",
            expected_revision=0,
        )
    assert insert_conflict.value.current_revision == 3


@pytest.mark.asyncio
async def test_kernel_kinds_pass_the_db_checks(service, run_store, control_store):
    """Migration 0040 gate: tool approvals + deliverable artifacts persist.

    The offline stores never see the CHECK constraints — only a live
    Postgres proves migration 0040 widened ``ck_run_approvals_kind`` and
    ``ck_run_artifacts_kind`` to match the constants.
    """
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(
        ApprovalRecord(
            approval_id="apr_tool_pg",
            run_id=run_id,
            kind="tool",
            payload={
                "actions": [
                    {"tool": "web_instant", "args": {"query": "EU AI Act"}}
                ]
            },
        )
    )
    stored = await control_store.get_approval(run_id, approval.approval_id)
    assert stored.kind == "tool"
    assert stored.payload["actions"][0]["args"] == {"query": "EU AI Act"}

    artifact = await control_store.upsert_artifact(
        run_id=run_id,
        kind="deliverable",
        session_id="sess-pg",
        title="E-Mail-Entwurf",
        status="ready",
        content_markdown="# Entwurf",
        payload={"deliverable_kind": "email"},
        refs=[],
        updated_by="agent",
        artifact_id="art_deliverable_pg",
    )
    assert artifact.kind == "deliverable"
    assert artifact.payload == {"deliverable_kind": "email"}


@pytest.mark.asyncio
async def test_multi_deliverable_and_atomic_batch_cas(run_store, control_store):
    run_id = _parked_agent_run(run_store)
    first = await control_store.upsert_artifact(
        run_id=run_id,
        kind="deliverable",
        session_id="sess-pg",
        title="One",
        status="ready",
        content_markdown="old one",
        payload={"stable": 1},
        refs=[{"url": "https://example.com/one"}],
        updated_by="agent",
        artifact_id="art_pg_one",
    )
    second = await control_store.upsert_artifact(
        run_id=run_id,
        kind="deliverable",
        session_id="sess-pg",
        title="Two",
        status="ready",
        content_markdown="old two",
        payload={"stable": 2},
        refs=[{"url": "https://example.com/two"}],
        updated_by="agent",
        artifact_id="art_pg_two",
    )
    rows = await control_store.list_session_artifacts("sess-pg")
    assert {row.artifact_id for row in rows if row.kind == "deliverable"} == {
        first.artifact_id,
        second.artifact_id,
    }

    revised = await control_store.revise_session_artifacts_atomically(
        run_id=run_id,
        session_id="sess-pg",
        revisions=[
            ArtifactBatchRevision(first.artifact_id, 1, "new one"),
            ArtifactBatchRevision(second.artifact_id, 1, "new two"),
        ],
    )
    assert [row.revision for row in revised] == [2, 2]
    assert revised[0].payload == {"stable": 1}
    assert tuple(revised[0].refs) == ({"url": "https://example.com/one"},)


@pytest.mark.asyncio
async def test_multiple_sessionless_deliverables(run_store, control_store):
    run_id = _parked_agent_run(run_store)
    first = await control_store.upsert_artifact(
        run_id=run_id,
        kind="deliverable",
        session_id=None,
        title="One",
        status="ready",
        content_markdown="one",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_sessionless_pg_one",
    )
    second = await control_store.upsert_artifact(
        run_id=run_id,
        kind="deliverable",
        session_id=None,
        title="Two",
        status="ready",
        content_markdown="two",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_sessionless_pg_two",
    )

    assert first.artifact_id != second.artifact_id
    rows, _cursor = await control_store.list_artifacts(
        run_id, kind="deliverable", limit=10
    )
    assert {row.artifact_id for row in rows} == {
        first.artifact_id,
        second.artifact_id,
    }


@pytest.mark.asyncio
async def test_control_rows_cascade_with_their_run(service, run_store, control_store):
    run_id = _parked_agent_run(run_store)
    approval = await service.store.create_approval(_approval(run_id))
    await control_store.upsert_artifact(
        run_id=run_id,
        kind="critic_report",
        session_id=None,
        title="Kritik",
        status="ready",
        content_markdown="ok",
        payload={},
        refs=[],
        updated_by="agent",
        artifact_id="art_critic",
    )
    run_store.cancel(run_id)
    _wait_until(lambda: run_store.get(run_id)["status"] == "cancelled")
    run_store.delete(run_id)

    engine = build_engine(TEST_DATABASE_URL)
    try:
        factory = build_session_factory(engine)
        async with factory() as session:
            for table in ("run_approvals", "run_artifacts", "run_artifact_revisions"):
                count = (
                    await session.execute(
                        text(f"SELECT count(*) FROM {table}")
                    )
                ).scalar_one()
                assert count == 0, table
    finally:
        await engine.dispose()
    assert approval.approval_id  # silence unused warning
