"""In-memory agent control store (offline/test tier).

Lockstep counterpart of
:class:`~inqtrix.storage.agent_control_postgres.PostgresAgentControlStore`:
same port, same error types, same ordering. Thread-safe via one lock (the
M5 runtime touches the store from worker threads); no awaits happen while
the lock is held.

Retention mirrors the Postgres ``ON DELETE CASCADE`` only logically: rows
whose run expired become unreachable (every route resolves the run first),
and process lifetime bounds growth — the same volatility contract as the
in-memory run store itself.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any

from inqtrix.agents.control_ports import (
    APPROVAL_STATUS_BY_DECISION,
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ApprovalRecord,
    ArtifactLocked,
    ArtifactBatchRevision,
    ArtifactNotFound,
    ArtifactRecord,
    ArtifactRevisionConflict,
    ArtifactRevisionRecord,
    ClarificationAlreadyAnswered,
    ClarificationNotFound,
    ClarificationRecord,
    PlanNotFound,
    PlanRecord,
    PlanTaskNotFound,
    PlanTaskRecord,
    SESSION_SINGLETON_ARTIFACT_KINDS,
    RUN_SINGLETON_ARTIFACT_KINDS,
    transitioned_plan_task,
    requested_task_cancellation,
)
from inqtrix.pagination import keyset_page


class MemoryAgentControlStore:
    """Dict-backed :class:`~inqtrix.agents.control_ports.AgentControlStore`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._plans: dict[str, PlanRecord] = {}
        self._plan_tasks: dict[str, list[PlanTaskRecord]] = {}
        self._approvals: dict[str, ApprovalRecord] = {}
        self._clarifications: dict[str, ClarificationRecord] = {}
        self._artifacts: dict[str, ArtifactRecord] = {}
        self._revisions: dict[str, list[ArtifactRevisionRecord]] = {}

    # -- plans ---------------------------------------------------------- #

    async def save_plan(
        self,
        *,
        run_id: str,
        plan: PlanRecord,
        tasks: list[PlanTaskRecord],
    ) -> PlanRecord:
        with self._lock:
            return self._save_plan_locked(run_id=run_id, plan=plan, tasks=tasks)

    def _save_plan_locked(
        self, *, run_id: str, plan: PlanRecord, tasks: list[PlanTaskRecord]
    ) -> PlanRecord:
        latest = max(
            (p.version for p in self._plans.values() if p.run_id == run_id),
            default=0,
        )
        version = plan.version if plan.version > 0 else latest + 1
        if version != latest + 1:
            raise ValueError(
                f"plan version must be {latest + 1}, got {version}"
            )
        for plan_id, existing in list(self._plans.items()):
            if existing.run_id == run_id and existing.status not in (
                "rejected",
                "superseded",
            ):
                self._plans[plan_id] = replace(existing, status="superseded")
        stored = replace(
            plan,
            run_id=run_id,
            version=version,
            created_at=plan.created_at or time.time(),
        )
        self._plans[stored.plan_id] = stored
        self._plan_tasks[stored.plan_id] = [
            replace(task, plan_id=stored.plan_id, run_id=run_id)
            for task in tasks
        ]
        return stored

    async def get_plan(
        self, run_id: str, *, version: int | None = None
    ) -> tuple[PlanRecord, list[PlanTaskRecord]]:
        with self._lock:
            candidates = [
                p for p in self._plans.values() if p.run_id == run_id
            ]
            if not candidates:
                raise PlanNotFound(run_id)
            if version is None:
                plan = max(candidates, key=lambda p: p.version)
            else:
                matching = [p for p in candidates if p.version == version]
                if not matching:
                    raise PlanNotFound(run_id)
                plan = matching[0]
            tasks = sorted(
                self._plan_tasks.get(plan.plan_id, []),
                key=lambda t: t.ordinal,
            )
            return plan, list(tasks)

    async def list_plan_versions(self, run_id: str) -> list[PlanRecord]:
        with self._lock:
            return sorted(
                (p for p in self._plans.values() if p.run_id == run_id),
                key=lambda p: p.version,
                reverse=True,
            )

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
        with self._lock:
            tasks = self._plan_tasks.get(plan_id)
            if tasks is None:
                raise PlanTaskNotFound(task_id)
            for index, current in enumerate(tasks):
                if current.task_id == task_id and current.run_id == run_id:
                    stored = transitioned_plan_task(
                        current,
                        status=status,
                        child_run_id=child_run_id,
                        result_summary=result_summary,
                        result_payload=result_payload,
                    )
                    tasks[index] = stored
                    return stored
            raise PlanTaskNotFound(task_id)

    async def request_plan_task_cancel(
        self,
        *,
        run_id: str,
        plan_id: str,
        task_id: str,
    ) -> PlanTaskRecord:
        with self._lock:
            tasks = self._plan_tasks.get(plan_id)
            if tasks is None:
                raise PlanTaskNotFound(task_id)
            for index, current in enumerate(tasks):
                if current.task_id == task_id and current.run_id == run_id:
                    stored = requested_task_cancellation(current)
                    tasks[index] = stored
                    return stored
            raise PlanTaskNotFound(task_id)

    # -- approvals ------------------------------------------------------- #

    async def create_approval(self, approval: ApprovalRecord) -> ApprovalRecord:
        with self._lock:
            stored = replace(
                approval, created_at=approval.created_at or time.time()
            )
            if stored.subject_type == "plan" and not stored.subject_id:
                version = int(stored.payload.get("plan_version", 0) or 0)
                matching = [
                    plan
                    for plan in self._plans.values()
                    if plan.run_id == stored.run_id
                    and (version <= 0 or plan.version == version)
                ]
                if not matching:
                    raise PlanNotFound(stored.run_id)
                stored = replace(
                    stored,
                    subject_id=max(
                        matching, key=lambda item: item.version
                    ).plan_id,
                )
            self._approvals[stored.approval_id] = stored
            return stored

    async def list_approvals(self, run_id: str) -> list[ApprovalRecord]:
        with self._lock:
            return sorted(
                (a for a in self._approvals.values() if a.run_id == run_id),
                key=lambda a: (a.created_at, a.approval_id),
                reverse=True,
            )

    async def get_approval(self, run_id: str, approval_id: str) -> ApprovalRecord:
        with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is None or approval.run_id != run_id:
                raise ApprovalNotFound(approval_id)
            return approval

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
        plans_before: dict[str, PlanRecord] = {}
        tasks_before: dict[str, list[PlanTaskRecord]] = {}
        with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is None or approval.run_id != run_id:
                raise ApprovalNotFound(approval_id)
            if approval.status != "pending":
                raise ApprovalAlreadyDecided(approval)
            plan_decision = approval.subject_type == "plan"
            mutates_plan = edited_plan is not None or plan_decision
            if mutates_plan:
                # Snapshot every plan row of the run BEFORE the save: the
                # supersede flip touches prior versions, and the revert
                # below must restore them too (Postgres gets this for
                # free from the transaction rollback — lockstep).
                plans_before = {
                    plan_id: plan
                    for plan_id, plan in self._plans.items()
                    if plan.run_id == run_id
                }
                tasks_before = {
                    plan_id: list(self._plan_tasks.get(plan_id, []))
                    for plan_id in plans_before
                }
            target_plan = (
                self._plans.get(approval.subject_id)
                if plan_decision and edited_plan is None
                else None
            )
            if plan_decision and edited_plan is None and (
                target_plan is None or target_plan.run_id != run_id
            ):
                raise PlanNotFound(run_id)
            decided = replace(
                approval,
                status=APPROVAL_STATUS_BY_DECISION[decision],
                decision=decision,
                decision_payload=dict(decision_payload),
                note=note,
                decided_by_sub=decided_by_sub,
                decided_at=time.time(),
            )
            try:
                self._approvals[approval_id] = decided
                if target_plan is not None:
                    self._plans[target_plan.plan_id] = replace(
                        target_plan,
                        status=(
                            "approved"
                            if decision == "approve"
                            else "rejected"
                        ),
                    )
                if edited_plan is not None:
                    self._save_plan_locked(
                        run_id=run_id,
                        plan=edited_plan,
                        tasks=list(edited_tasks or []),
                    )
            except BaseException:
                self._approvals[approval_id] = approval
                if mutates_plan:
                    self._restore_run_plans_locked(
                        run_id,
                        plans_before=plans_before,
                        tasks_before=tasks_before,
                    )
                raise
        try:
            summary = await resume(None)
        except BaseException:
            # Single-process rollback: the decision must not stand on a
            # run that could not resume (the caller maps the error).
            with self._lock:
                self._approvals[approval_id] = approval
                if mutates_plan:
                    self._restore_run_plans_locked(
                        run_id,
                        plans_before=plans_before,
                        tasks_before=tasks_before,
                    )
            raise
        with self._lock:
            return self._approvals[approval_id], summary

    def _restore_run_plans_locked(
        self,
        run_id: str,
        *,
        plans_before: dict[str, PlanRecord],
        tasks_before: dict[str, list[PlanTaskRecord]],
    ) -> None:
        """Restore a run's plan rows after an in-memory transaction abort."""
        for plan_id, plan in list(self._plans.items()):
            if plan.run_id == run_id and plan_id not in plans_before:
                self._plans.pop(plan_id, None)
                self._plan_tasks.pop(plan_id, None)
        self._plans.update(plans_before)
        for plan_id, tasks in tasks_before.items():
            self._plan_tasks[plan_id] = list(tasks)

    # -- clarifications --------------------------------------------------- #

    async def create_clarification(
        self, clarification: ClarificationRecord
    ) -> ClarificationRecord:
        with self._lock:
            stored = replace(
                clarification,
                created_at=clarification.created_at or time.time(),
            )
            self._clarifications[stored.clarification_id] = stored
            return stored

    async def list_clarifications(self, run_id: str) -> list[ClarificationRecord]:
        with self._lock:
            return sorted(
                (
                    c
                    for c in self._clarifications.values()
                    if c.run_id == run_id
                ),
                key=lambda c: (c.created_at, c.clarification_id),
                reverse=True,
            )

    async def get_clarification(
        self, run_id: str, clarification_id: str
    ) -> ClarificationRecord:
        with self._lock:
            clarification = self._clarifications.get(clarification_id)
            if clarification is None or clarification.run_id != run_id:
                raise ClarificationNotFound(clarification_id)
            return clarification

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
        with self._lock:
            clarification = self._clarifications.get(clarification_id)
            if clarification is None or clarification.run_id != run_id:
                raise ClarificationNotFound(clarification_id)
            if clarification.status != "pending":
                raise ClarificationAlreadyAnswered(clarification)
            answered = replace(
                clarification,
                status="answered",
                answer=answer,
                option_id=option_id,
                answers=dict(answers),
                answered_by_sub=answered_by_sub,
                answered_at=time.time(),
            )
            self._clarifications[clarification_id] = answered
        try:
            summary = await resume(None)
        except BaseException:
            with self._lock:
                self._clarifications[clarification_id] = clarification
            raise
        with self._lock:
            return self._clarifications[clarification_id], summary

    # -- artifacts --------------------------------------------------------- #

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
        with self._lock:
            existing = self._find_upsert_target_locked(
                artifact_id=artifact_id,
                session_id=session_id,
                run_id=run_id,
                kind=kind,
            )
            if existing is not None and (
                existing.kind != kind
                or existing.session_id != session_id
                or (session_id is None and existing.run_id != run_id)
            ):
                raise ArtifactNotFound(artifact_id or existing.artifact_id)
            if (
                existing is not None
                and expected_revision is not None
                and existing.revision != expected_revision
            ):
                # E13 symmetric guard: a user edited this row since the
                # agent read it — refuse to overwrite (same conflict the
                # user PUT path raises).
                raise ArtifactRevisionConflict(existing.revision)
            now = time.time()
            if existing is None:
                if artifact_id is None:
                    raise ValueError("new artifacts need an explicit id")
                stored = ArtifactRecord(
                    artifact_id=artifact_id,
                    run_id=run_id,
                    kind=kind,
                    session_id=session_id,
                    title=title,
                    status=status,
                    revision=1,
                    updated_by=updated_by,
                    content_markdown=content_markdown,
                    payload=dict(payload),
                    refs=tuple(dict(ref) for ref in refs),
                    created_at=now,
                    updated_at=now,
                )
            else:
                stored = replace(
                    existing,
                    run_id=run_id,
                    title=title,
                    status=status,
                    revision=existing.revision + 1,
                    updated_by=updated_by,
                    content_markdown=content_markdown,
                    payload=dict(payload),
                    refs=tuple(dict(ref) for ref in refs),
                    updated_at=now,
                )
            self._artifacts[stored.artifact_id] = stored
            self._revisions.setdefault(stored.artifact_id, []).append(
                ArtifactRevisionRecord(
                    artifact_id=stored.artifact_id,
                    revision=stored.revision,
                    created_by=updated_by,
                    content_markdown=content_markdown,
                    created_at=now,
                )
            )
            return stored

    def _find_upsert_target_locked(
        self,
        *,
        artifact_id: str | None,
        session_id: str | None,
        run_id: str,
        kind: str,
    ) -> ArtifactRecord | None:
        if artifact_id is not None:
            return self._artifacts.get(artifact_id)
        if (
            session_id is not None
            and kind not in SESSION_SINGLETON_ARTIFACT_KINDS
        ):
            return None
        if session_id is None and kind not in RUN_SINGLETON_ARTIFACT_KINDS:
            return None
        for artifact in self._artifacts.values():
            if artifact.kind != kind:
                continue
            if session_id is not None:
                if artifact.session_id == session_id:
                    return artifact
            elif artifact.session_id is None and artifact.run_id == run_id:
                return artifact
        return None

    async def user_update_artifact(
        self,
        *,
        run_id: str,
        artifact_id: str,
        content_markdown: str,
        expected_revision: int,
    ) -> ArtifactRecord:
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None or artifact.run_id != run_id:
                raise ArtifactNotFound(artifact_id)
            if artifact.status == "writing":
                raise ArtifactLocked(artifact_id)
            if artifact.revision != expected_revision:
                raise ArtifactRevisionConflict(artifact.revision)
            now = time.time()
            stored = replace(
                artifact,
                revision=artifact.revision + 1,
                updated_by="user",
                content_markdown=content_markdown,
                updated_at=now,
            )
            self._artifacts[artifact_id] = stored
            self._revisions.setdefault(artifact_id, []).append(
                ArtifactRevisionRecord(
                    artifact_id=artifact_id,
                    revision=stored.revision,
                    created_by="user",
                    content_markdown=content_markdown,
                    created_at=now,
                )
            )
            return stored

    async def get_artifact(
        self, run_id: str, artifact_id: str, *, revision: int | None = None
    ) -> tuple[ArtifactRecord, list[ArtifactRevisionRecord]]:
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None or artifact.run_id != run_id:
                raise ArtifactNotFound(artifact_id)
            revisions = list(self._revisions.get(artifact_id, ()))
            if revision is not None:
                matching = [r for r in revisions if r.revision == revision]
                if not matching:
                    raise ArtifactNotFound(artifact_id)
                artifact = replace(
                    artifact, content_markdown=matching[0].content_markdown
                )
            metadata = [
                replace(row, content_markdown="") for row in revisions
            ]
            metadata.sort(key=lambda row: row.revision, reverse=True)
            return artifact, metadata

    async def get_session_artifact(
        self, session_id: str, kind: str
    ) -> ArtifactRecord | None:
        with self._lock:
            for artifact in self._artifacts.values():
                if artifact.session_id == session_id and artifact.kind == kind:
                    return artifact
            return None

    async def get_session_artifact_by_id(
        self, session_id: str, artifact_id: str
    ) -> ArtifactRecord:
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None or artifact.session_id != session_id:
                raise ArtifactNotFound(artifact_id)
            return artifact

    async def revise_session_artifacts_atomically(
        self,
        *,
        run_id: str,
        session_id: str | None,
        revisions: list[ArtifactBatchRevision],
    ) -> list[ArtifactRecord]:
        with self._lock:
            ids = [item.artifact_id for item in revisions]
            if len(ids) != len(set(ids)):
                raise ValueError("artifact revision batch contains duplicate ids")
            current: list[ArtifactRecord] = []
            for item in revisions:
                artifact = self._artifacts.get(item.artifact_id)
                if artifact is None or artifact.session_id != session_id:
                    raise ArtifactNotFound(item.artifact_id)
                if artifact.revision != item.expected_revision:
                    raise ArtifactRevisionConflict(artifact.revision)
                current.append(artifact)
            now = time.time()
            stored_rows: list[ArtifactRecord] = []
            for item, artifact in zip(revisions, current, strict=True):
                stored = replace(
                    artifact,
                    run_id=run_id,
                    revision=artifact.revision + 1,
                    updated_by="agent",
                    content_markdown=item.content_markdown,
                    updated_at=now,
                )
                self._artifacts[stored.artifact_id] = stored
                self._revisions.setdefault(stored.artifact_id, []).append(
                    ArtifactRevisionRecord(
                        artifact_id=stored.artifact_id,
                        revision=stored.revision,
                        created_by="agent",
                        content_markdown=stored.content_markdown,
                        created_at=now,
                    )
                )
                stored_rows.append(stored)
            return stored_rows

    async def list_session_artifacts(
        self, session_id: str
    ) -> list[ArtifactRecord]:
        with self._lock:
            return sorted(
                (
                    replace(artifact, content_markdown="")
                    for artifact in self._artifacts.values()
                    if artifact.session_id == session_id
                ),
                key=lambda artifact: (
                    artifact.created_at,
                    artifact.artifact_id,
                ),
            )

    async def list_artifacts(
        self,
        run_id: str,
        *,
        kind: str | None = None,
        limit: int = 50,
        after: tuple[float, str] | None = None,
    ) -> tuple[list[ArtifactRecord], str | None]:
        with self._lock:
            rows = [
                replace(artifact, content_markdown="")
                for artifact in self._artifacts.values()
                if artifact.run_id == run_id
                and (kind is None or artifact.kind == kind)
            ]
        rows.sort(key=lambda a: (a.created_at, a.artifact_id), reverse=True)
        return keyset_page(
            rows,
            limit=limit,
            after=after,
            created_at_of=lambda a: a.created_at,
            id_of=lambda a: a.artifact_id,
        )

    async def aclose(self) -> None:
        """No-op; symmetric with the Postgres store."""
