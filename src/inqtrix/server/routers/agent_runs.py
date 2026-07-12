"""Agent run control endpoints (``/v1/runs/{run_id}/plan|approvals|...``).

Thin HTTP layer over
:class:`~inqtrix.services.agent_control_service.AgentControlService`.
Access rule for EVERY subresource here: the parent run is resolved first
with the caller's visibility (share grants included) — unknown, foreign,
or cross-workspace runs answer with the indistinct 404 (denial ==
absence). Reads work from a ``view`` share; mutations require the owner
or an ``edit`` share and deny with the same 404 (the M3 cancel
precedent).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from inqtrix.agents.control_ports import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ApprovalRecord,
    ArtifactLocked,
    ArtifactNotFound,
    ArtifactRecord,
    ArtifactRevisionConflict,
    ArtifactRevisionRecord,
    ClarificationAlreadyAnswered,
    ClarificationNotFound,
    ClarificationRecord,
    PlanNotFound,
    PlanRecord,
    PlanTaskCancellationConflict,
    PlanTaskNotFound,
    PlanTaskRecord,
)
from inqtrix.auth.principal import Principal, UserContext
from inqtrix.pagination import InvalidCursor, clamp_limit, decode_cursor, list_envelope
from inqtrix.runs.shared import access_permits_edit
from inqtrix.server.routers import build_shared_grants_dependency
from inqtrix.server.runs import RunActive, RunNotFound
from inqtrix.services.agent_control_service import (
    AgentControlUnavailable,
    AgentControlValidationError,
)
from inqtrix.services.request_parsing import (
    error_response,
    workspace_id_from_request,
)

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")

_RUN_NOT_FOUND = ("Run nicht gefunden", "not_found")


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the agent control routes against the container."""
    service = container.agent_control_service
    if service is None:
        raise RuntimeError(
            "agent_runs router requires container.agent_control_service"
        )
    router = APIRouter()
    run_store = container.run_store
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency
    shared_runs_dep = build_shared_grants_dependency(
        container.share_service, principal_dep, resource_type="run"
    )

    async def _resolve_run(
        req: Request,
        run_id: str,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, Any] | None",
        *,
        mutation: bool,
    ) -> dict[str, Any] | JSONResponse:
        """The parent-run gate every subresource route goes through.

        Returns the run summary, or the ready-made error response. A
        ``view`` share admits reads; mutations need the owner or an
        ``edit`` share — denied with the indistinct 404 (M3 cancel
        precedent: a viewer must not learn that mutating would have
        been possible).
        """
        try:
            workspace_id = workspace_id_from_request(req)
            summary = await asyncio.to_thread(
                run_store.get,
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except RunNotFound:
            return error_response(404, *_RUN_NOT_FOUND)
        if mutation and not access_permits_edit(summary.get("access")):
            return error_response(404, *_RUN_NOT_FOUND)
        return summary

    async def _json_object(req: Request) -> dict[str, Any] | None:
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001 — malformed body is a client error
            return None
        return body if isinstance(body, dict) else None

    # -- plan (#4) -------------------------------------------------------- #

    @router.get("/v1/runs/{run_id}/plan")
    async def get_run_plan(
        run_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Return one plan version (latest by default) with its tasks and
        the version history (``?version=`` selects an older one)."""
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=False
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        version_raw = req.query_params.get("version")
        version: int | None = None
        if version_raw is not None:
            try:
                version = int(version_raw)
            except ValueError:
                return error_response(
                    400,
                    "version muss eine Ganzzahl sein",
                    "invalid_request_error",
                )
        try:
            plan, tasks, versions = await service.get_plan(
                run_id, version=version
            )
        except PlanNotFound:
            return error_response(
                404, "Noch kein Plan vorhanden", "not_found"
            )
        return {
            **_plan_payload(plan),
            "tasks": [_task_payload(task) for task in tasks],
            "versions": [_plan_payload(row) for row in versions],
        }

    @router.get("/v1/runs/{run_id}/tasks/{task_id}/result")
    async def get_run_task_result(
        run_id: str,
        task_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Return complete Markdown and evidence for one plan task.

        ``?version=`` addresses an older plan version. The regular plan
        response remains compact so a run overview never transfers every
        provider answer eagerly.
        """
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=False
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        version_raw = req.query_params.get("version")
        version: int | None = None
        if version_raw is not None:
            try:
                version = int(version_raw)
            except ValueError:
                return error_response(
                    400,
                    "version muss eine Ganzzahl sein",
                    "invalid_request_error",
                )
        try:
            task = await service.get_task_result(
                run_id, task_id, version=version
            )
        except (PlanNotFound, PlanTaskNotFound):
            return error_response(404, "Aufgabe nicht gefunden", "not_found")
        payload = task.result_payload or {}
        references = [
            dict(item)
            for item in payload.get("evidence", [])
            if isinstance(item, dict)
        ]
        claims = [
            dict(item)
            for item in payload.get("claims", [])
            if isinstance(item, dict)
        ]
        usage = payload.get("usage")
        normalized_usage = (
            {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    usage.get("completion_tokens", 0) or 0
                ),
            }
            if isinstance(usage, dict)
            else {"prompt_tokens": 0, "completion_tokens": 0}
        )
        answer_markdown = str(payload.get("answer_markdown") or "")
        failure_message = str(payload.get("failure_reason") or "")
        failure_code = str(payload.get("failure_code") or "")
        return {
            "task_id": task.task_id,
            "status": task.status,
            "child_run_id": task.child_run_id,
            "result_summary": task.result_summary,
            "answer_markdown": answer_markdown,
            "references": references,
            "claims": claims,
            "metrics": {
                "reference_count": len(references),
                "claim_count": len(claims),
                **normalized_usage,
            },
            "error": (
                {"code": failure_code, "message": failure_message}
                if failure_code or failure_message
                else None
            ),
            "legacy_summary_only": bool(
                task.result_summary and not answer_markdown
            ),
        }

    @router.post("/v1/runs/{run_id}/tasks/{task_id}/cancel")
    async def cancel_run_task(
        run_id: str,
        task_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Request cancellation of one source task, preserving siblings."""
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=True
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        try:
            task = await service.request_task_cancel(
                run_id,
                task_id,
                workspace_id=resolved.get("workspace_id"),
            )
        except (PlanNotFound, PlanTaskNotFound):
            return error_response(404, "Aufgabe nicht gefunden", "not_found")
        except PlanTaskCancellationConflict as exc:
            return error_response(
                409,
                "Diese Aufgabe kann nicht mehr abgebrochen werden.",
                "task_cancel_conflict",
                status=exc.task.status,
            )
        return {
            "task_id": task.task_id,
            "status": task.status,
            "child_run_id": task.child_run_id,
        }

    # -- approvals (#5, #6) -------------------------------------------------- #

    @router.get("/v1/runs/{run_id}/approvals")
    async def list_run_approvals(
        run_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """List the run's approval requests, newest first."""
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=False
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        approvals = await service.list_approvals(run_id)
        return {
            "object": "list",
            "data": [_approval_payload(row) for row in approvals],
        }

    @router.post("/v1/runs/{run_id}/approvals/{approval_id}")
    async def decide_run_approval(
        run_id: str,
        approval_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Decide one approval (approve / reject / edit).

        An edit carries ``plan`` for plan/replan approvals or ``actions``
        (exactly one action, args-only change) for tool approvals.
        Replaying the SAME decision answers 200 with the stored state; a
        different decision on an already-decided approval answers 409.
        """
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=True
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        body = await _json_object(req)
        if body is None:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        decision = body.get("decision")
        if not isinstance(decision, str) or not decision:
            return error_response(
                400, "decision ist erforderlich.", "invalid_request_error"
            )
        note = body.get("note")
        if note is not None and not isinstance(note, str):
            return error_response(
                400, "note muss ein String sein.", "invalid_request_error"
            )
        report_guidance = body.get("report_guidance")
        if report_guidance is not None and (
            not isinstance(report_guidance, str)
            or len(report_guidance) > 2000
        ):
            return error_response(
                400,
                "report_guidance muss ein String mit maximal 2000 "
                "Zeichen sein.",
                "invalid_request_error",
            )
        plan_body = body.get("plan")
        if plan_body is not None and not isinstance(plan_body, dict):
            return error_response(
                400, "plan muss ein Objekt sein.", "invalid_request_error"
            )
        actions_body = body.get("actions")
        if actions_body is not None and not isinstance(actions_body, list):
            return error_response(
                400, "actions muss eine Liste sein.", "invalid_request_error"
            )
        try:
            approval, summary, _replayed = await service.decide_approval(
                run_id=run_id,
                approval_id=approval_id,
                decision=decision,
                plan_body=plan_body,
                note=note or "",
                report_guidance=(report_guidance or "").strip(),
                principal=principal,
                visible_to=visible_to,
                actions_body=actions_body,
            )
        except AgentControlValidationError as exc:
            return error_response(
                400, str(exc), exc.error_type, errors=exc.errors
            )
        except ApprovalNotFound:
            return error_response(
                404, "Genehmigung nicht gefunden", "not_found"
            )
        except ApprovalAlreadyDecided as exc:
            return error_response(
                409,
                "Die Genehmigung wurde bereits anders entschieden.",
                "conflict",
                status=exc.approval.status,
                decision=exc.approval.decision,
            )
        except RunNotFound:
            return error_response(404, *_RUN_NOT_FOUND)
        except RunActive:
            return error_response(
                409,
                "Der Run wartet nicht (mehr) auf diese Entscheidung.",
                "conflict",
            )
        return {**_approval_payload(approval), "run": summary}

    # -- clarifications (#7, #8) --------------------------------------------- #

    @router.get("/v1/runs/{run_id}/clarifications")
    async def list_run_clarifications(
        run_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """List the run's clarification questions, newest first."""
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=False
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        clarifications = await service.list_clarifications(run_id)
        return {
            "object": "list",
            "data": [_clarification_payload(row) for row in clarifications],
        }

    @router.post("/v1/runs/{run_id}/clarifications/{clarification_id}")
    async def answer_run_clarification(
        run_id: str,
        clarification_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Answer one clarification (free text OR an option id)."""
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=True
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        body = await _json_object(req)
        if body is None:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        answer = body.get("answer")
        option_id = body.get("option_id")
        answers = body.get("answers")
        if answer is not None and not isinstance(answer, str):
            return error_response(
                400, "answer muss ein String sein.", "invalid_request_error"
            )
        if option_id is not None and not isinstance(option_id, str):
            return error_response(
                400,
                "option_id muss ein String sein.",
                "invalid_request_error",
            )
        if answers is not None and not isinstance(answers, dict):
            return error_response(
                400,
                "answers muss ein Objekt {frage_id: {option_ids, text}} "
                "sein.",
                "invalid_request_error",
            )
        try:
            clarification, summary, _replayed = (
                await service.answer_clarification(
                    run_id=run_id,
                    clarification_id=clarification_id,
                    answer=answer,
                    option_id=option_id,
                    answers=answers,
                    principal=principal,
                )
            )
        except AgentControlValidationError as exc:
            return error_response(
                400, str(exc), exc.error_type, errors=exc.errors
            )
        except ClarificationNotFound:
            return error_response(
                404, "Rueckfrage nicht gefunden", "not_found"
            )
        except ClarificationAlreadyAnswered:
            return error_response(
                409,
                "Die Rueckfrage wurde bereits anders beantwortet.",
                "conflict",
            )
        except RunNotFound:
            return error_response(404, *_RUN_NOT_FOUND)
        except RunActive:
            return error_response(
                409,
                "Der Run wartet nicht (mehr) auf diese Antwort.",
                "conflict",
            )
        return {**_clarification_payload(clarification), "run": summary}

    # -- artifacts (#9 - #12) -------------------------------------------------- #

    @router.get("/v1/runs/{run_id}/artifacts")
    async def list_run_artifacts(
        run_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Artifact METADATA page (no bodies), ``?kind&limit&cursor``."""
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=False
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        try:
            after = decode_cursor(req.query_params.get("cursor"))
        except InvalidCursor:
            return error_response(400, "Ungueltiger Cursor", "invalid_cursor")
        limit = clamp_limit(req.query_params.get("limit"))
        artifacts, next_cursor = await service.list_artifacts(
            run_id,
            kind=req.query_params.get("kind"),
            limit=limit,
            after=after,
        )
        return list_envelope(
            [_artifact_meta_payload(row) for row in artifacts], next_cursor
        )

    @router.get("/v1/runs/{run_id}/artifacts/{artifact_id}")
    async def get_run_artifact(
        run_id: str,
        artifact_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """One artifact with body, refs, and revision history
        (``?revision=`` serves an older body)."""
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=False
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        revision_raw = req.query_params.get("revision")
        revision: int | None = None
        if revision_raw is not None:
            try:
                revision = int(revision_raw)
            except ValueError:
                return error_response(
                    400,
                    "revision muss eine Ganzzahl sein",
                    "invalid_request_error",
                )
        try:
            artifact, revisions = await service.get_artifact(
                run_id, artifact_id, revision=revision
            )
        except ArtifactNotFound:
            return error_response(404, "Artefakt nicht gefunden", "not_found")
        return {
            **_artifact_meta_payload(artifact),
            "refs": [dict(ref) for ref in artifact.refs],
            "content_markdown": artifact.content_markdown,
            # Kind-specific context (e.g. the deliverable_kind format
            # hint of kernel canvases, M2) — additive, "{}" historically.
            "payload": dict(artifact.payload),
            "revisions": [_revision_payload(row) for row in revisions],
        }

    @router.put("/v1/runs/{run_id}/artifacts/{artifact_id}")
    async def update_run_artifact(
        run_id: str,
        artifact_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Optimistic user edit of an artifact body (E13 409 matrix)."""
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=True
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        body = await _json_object(req)
        if body is None:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        content = body.get("content_markdown")
        if not isinstance(content, str):
            return error_response(
                400,
                "content_markdown ist erforderlich.",
                "invalid_request_error",
            )
        expected = body.get("expected_revision")
        # bool subclasses int — reject it explicitly (the prompt-template
        # precedent for numeric preconditions).
        if isinstance(expected, bool) or not isinstance(expected, int):
            return error_response(
                400,
                "expected_revision ist erforderlich (Ganzzahl).",
                "invalid_request_error",
            )
        try:
            artifact = await service.user_update_artifact(
                run_id=run_id,
                artifact_id=artifact_id,
                content_markdown=content,
                expected_revision=expected,
                principal=principal,
            )
        except ArtifactNotFound:
            return error_response(404, "Artefakt nicht gefunden", "not_found")
        except ArtifactLocked:
            return error_response(
                409,
                "Der Agent schreibt gerade in dieses Artefakt.",
                "conflict",
                locked_by="agent",
            )
        except ArtifactRevisionConflict as exc:
            return error_response(
                409,
                "Das Artefakt wurde zwischenzeitlich geaendert.",
                "conflict",
                current_revision=exc.current_revision,
            )
        return {
            "id": artifact.artifact_id,
            "revision": artifact.revision,
            "updated_by": artifact.updated_by,
        }

    @router.post(
        "/v1/runs/{run_id}/artifacts/{artifact_id}/export", status_code=201
    )
    async def export_run_artifact(
        run_id: str,
        artifact_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Copy the artifact into a NEW editor document (copy-out).

        Deliberately NOT idempotent: each call creates a fresh document
        owned by the caller. A view share suffices — exporting reads the
        run and writes only into the caller's own namespace.
        """
        resolved = await _resolve_run(
            req, run_id, visible_to, also_visible, mutation=False
        )
        if isinstance(resolved, JSONResponse):
            return resolved
        body = await _json_object(req) or {}
        target = body.get("target", "editor_document")
        if target != "editor_document":
            return error_response(
                400,
                "target muss 'editor_document' sein.",
                "invalid_request_error",
            )
        title = body.get("title")
        if title is not None and not isinstance(title, str):
            return error_response(
                400, "title muss ein String sein.", "invalid_request_error"
            )
        folder_id = body.get("folder_id")
        if folder_id is not None and not isinstance(folder_id, str):
            return error_response(
                400,
                "folder_id muss ein String sein.",
                "invalid_request_error",
            )
        try:
            workspace_id = workspace_id_from_request(req, body)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        caller_sub = (
            principal.sub
            if principal.kind in ("oidc_session", "pat")
            else None
        )
        try:
            document = await service.export_artifact(
                run_id=run_id,
                artifact_id=artifact_id,
                title=title,
                folder_id=folder_id,
                principal=principal,
                caller_sub=caller_sub,
                workspace_id=workspace_id,
            )
        except ArtifactNotFound:
            return error_response(404, "Artefakt nicht gefunden", "not_found")
        except AgentControlUnavailable as exc:
            return error_response(502, str(exc), "server_error")
        except IntegrityError:
            # The one client-attributable store failure: folder_id is a
            # real FK to editor_folders. Everything else (DB outage,
            # editor-store faults) propagates as a genuine 500 — a 400
            # would wrongly blame the caller and discourage retries.
            # Backend note: the MEMORY editor store validates no folder
            # ids (offline tier), so this 400 fires only on Postgres.
            log.warning(
                "Artefakt-Export fuer Run %s: Ordner %r existiert nicht.",
                run_id,
                folder_id,
            )
            return error_response(
                400, "Ordner nicht gefunden.", "invalid_request_error"
            )
        return JSONResponse(status_code=201, content=document)

    return router


# -- wire payloads ------------------------------------------------------------ #


def _plan_payload(plan: PlanRecord) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "run_id": plan.run_id,
        "version": plan.version,
        "status": plan.status,
        "created_by": plan.created_by,
        "summary_markdown": plan.summary_markdown,
        "assumptions": list(plan.assumptions),
        "success_criteria": list(plan.success_criteria),
        "reason": plan.reason,
        "created_at": plan.created_at,
    }


def _task_payload(task: PlanTaskRecord) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "ordinal": task.ordinal,
        "title": task.title,
        "tool_kind": task.tool_kind,
        "objective": task.objective,
        "queries": list(task.queries),
        "gap_ids": list(task.gap_ids),
        "depends_on": list(task.depends_on),
        "budget": dict(task.budget),
        "params": dict(task.params),
        "expected_output": task.expected_output,
        "is_falsification": task.is_falsification,
        "status": task.status,
        "child_run_id": task.child_run_id,
        "result_summary": task.result_summary,
    }


def _approval_payload(approval: ApprovalRecord) -> dict[str, Any]:
    return {
        "approval_id": approval.approval_id,
        "run_id": approval.run_id,
        "kind": approval.kind,
        "status": approval.status,
        "subject_type": approval.subject_type,
        "subject_id": approval.subject_id,
        "payload": dict(approval.payload),
        "decision": approval.decision,
        "note": approval.note,
        "decided_by_sub": approval.decided_by_sub,
        "created_at": approval.created_at,
        "decided_at": approval.decided_at,
    }


def _clarification_payload(clarification: ClarificationRecord) -> dict[str, Any]:
    return {
        "clarification_id": clarification.clarification_id,
        "run_id": clarification.run_id,
        "question": clarification.question,
        "options": [dict(option) for option in clarification.options],
        "questions": [
            dict(question) for question in clarification.questions
        ],
        "answers": dict(clarification.answers),
        "default_assumption": clarification.default_assumption,
        "status": clarification.status,
        "answer": clarification.answer,
        "option_id": clarification.option_id,
        "answered_by_sub": clarification.answered_by_sub,
        "created_at": clarification.created_at,
        "answered_at": clarification.answered_at,
    }


def _artifact_meta_payload(artifact: ArtifactRecord) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "run_id": artifact.run_id,
        "session_id": artifact.session_id,
        "kind": artifact.kind,
        "title": artifact.title,
        "status": artifact.status,
        "revision": artifact.revision,
        "updated_by": artifact.updated_by,
        "refs_count": len(artifact.refs),
        "created_at": artifact.created_at,
        "updated_at": artifact.updated_at,
    }


def _revision_payload(revision: ArtifactRevisionRecord) -> dict[str, Any]:
    return {
        "revision": revision.revision,
        "created_by": revision.created_by,
        "created_at": revision.created_at,
    }
