"""Native run lifecycle endpoints (``/v1/runs*``).

Submission dispatches through the
:class:`~inqtrix.services.run_service.RunService`; the read side
(list/get/result/cancel/events) consumes the in-memory run store
directly — it already is the repository surface.
"""

from __future__ import annotations

import asyncio
import logging
from queue import Empty
from typing import TYPE_CHECKING, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.knowledge.stores.ports import CollectionNotFound
from inqtrix.quota.models import QuotaDimension
from inqtrix.server.routers import (
    build_shared_grants_dependency,
    quota_admission,
    quota_record,
    stack_error_response,
)
from inqtrix.server.runs import (
    RunActive,
    RunNotFound,
    RunQueueFull,
    format_sse_event,
)
from inqtrix.services.agent_context import StackResolutionError
from inqtrix.services.request_parsing import (
    error_response,
    format_history,
    question_and_messages,
    workspace_id_from_request,
)

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")

TERMINAL_EVENTS = {
    "inqtrix.run.completed",
    "inqtrix.run.failed",
    "inqtrix.run.cancelled",
}


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the native run routes against the container."""
    router = APIRouter()
    settings = container.settings
    resolver = container.resolver
    run_service = container.run_service
    run_store = container.run_store
    quota_service = container.quota_service
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency
    share_service = container.share_service
    workspace_admin = container.workspace_admin
    shared_runs_dep = build_shared_grants_dependency(
        share_service, principal_dep, resource_type="run"
    )
    knowledge_service = container.knowledge_service
    shared_collections_dep = build_shared_grants_dependency(
        share_service, principal_dep, resource_type="knowledge_collection"
    )

    @router.post("/v1/runs")
    async def create_run(
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        collection_grants=Depends(shared_collections_dep),
    ):
        """Create a queued native research run for browser UI clients."""
        try:
            body = await req.json()
        except Exception:
            return error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")

        try:
            workspace_id = workspace_id_from_request(req, body)
            question, messages = question_and_messages(body, settings.server)
            resolved = resolver.resolve(body)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except StackResolutionError as exc:
            return stack_error_response(exc)

        # Admission gate for knowledge asks: strict like the chat
        # path — ONE invisible collection denies the whole submission
        # (an ask that silently searched fewer collections than picked
        # would change the answer's meaning without a trace), and the
        # worker only re-executes admitted requests.
        requested_collections = resolved.knowledge_filters.get(
            "collection_ids"
        )
        if (
            knowledge_service is not None
            and visible_to is not None
            and isinstance(requested_collections, list)
            and requested_collections
        ):
            try:
                await knowledge_service.assert_collections_visible(
                    [str(item) for item in requested_collections],
                    visible_to=visible_to,
                    also_visible=collection_grants,
                )
            except CollectionNotFound:
                return error_response(
                    404, "Collection nicht gefunden", "not_found"
                )

        if len(question) > resolved.agent_settings.max_question_length:
            return error_response(
                400,
                (
                    f"Frage zu lang ({len(question)} Zeichen, "
                    f"max. {resolved.agent_settings.max_question_length})"
                ),
                "invalid_request_error",
            )

        history = format_history(
            messages, max_messages=settings.server.max_messages_history
        )

        # Quota admission: one run counts as one run-unit, AND the run is
        # the highest LLM-token consumer — so a caller already over their
        # monthly token budget is blocked here too (the run's token spend
        # is recorded post-hoc at completion; admission is the only gate
        # that works regardless of which process executes the run).
        # Blocked callers never enter the queue (429 before any cost).
        for dimension in (QuotaDimension.RUNS, QuotaDimension.LLM_TOKENS):
            denied = await quota_admission(quota_service, principal, dimension)
            if denied is not None:
                return denied

        try:
            # to_thread: the durable store's submit blocks on database
            # round-trips; running it inline would stall the event loop
            # for every concurrent request.
            summary = await asyncio.to_thread(
                run_service.submit,
                question=question,
                history=history,
                messages=messages,
                resolved=resolved,
                workspace_id=workspace_id,
                principal=principal,
            )
        except RunQueueFull:
            return error_response(
                429,
                "Zu viele wartende Recherche-Auftraege. Bitte warten.",
                "rate_limit_error",
            )
        # Booked only once the run is accepted into the queue: the
        # run's LLM-token spend is booked separately at completion.
        await quota_record(quota_service, principal, QuotaDimension.RUNS, 1)
        return JSONResponse(status_code=202, content=summary)

    @router.post("/v1/runs/import")
    async def import_run(
        req: Request,
        principal: Principal = Depends(principal_dep),
    ):
        """Persist a COMPLETED report carried in from a loaded project file.

        A report in a project file is a terminal-run snapshot with no
        server-side execution; importing stores it durably under the caller so
        it survives reload + follows the user across devices (the runs analogue
        of the chat/editor/prompt import-up). Idempotent on the report's
        ``run_id`` for the caller; never executes the agent (no quota cost).
        """
        try:
            body = await req.json()
        except Exception:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        try:
            workspace_id = workspace_id_from_request(req, body)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)

        run_id = body.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return error_response(
                400, "run_id ist erforderlich.", "invalid_request_error"
            )

        def _epoch(value: object) -> float | None:
            return float(value) if isinstance(value, (int, float)) else None

        try:
            summary = await asyncio.to_thread(
                run_store.import_completed_run,
                run_id=run_id,
                question=str(body.get("question") or ""),
                stack_name=str(body.get("stack") or "default"),
                result=body.get("result") if isinstance(body.get("result"), dict) else {},
                status=str(body.get("status") or "completed"),
                mode=str(body.get("mode") or "research"),
                agent_overrides=body.get("agent_overrides")
                if isinstance(body.get("agent_overrides"), dict)
                else {},
                snapshot=body.get("snapshot")
                if isinstance(body.get("snapshot"), dict)
                else {},
                error=body.get("error")
                if isinstance(body.get("error"), dict)
                else None,
                created_at=_epoch(body.get("created_at")),
                workspace_id=workspace_id,
                created_by_sub=principal.sub,
                created_by_tenant_id=principal.tenant_id,
            )
        except ValueError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        return JSONResponse(status_code=200, content=summary)

    @router.get("/v1/runs")
    def list_runs(
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """List all queued, running, and short-lived terminal native runs."""
        try:
            workspace_id = workspace_id_from_request(req)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return {
            "object": "list",
            "data": run_store.list(
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            ),
        }

    @router.get("/v1/runs/{run_id}")
    def get_run(
        run_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Return the current public summary for one native run."""
        try:
            workspace_id = workspace_id_from_request(req)
            return run_store.get(
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except RunNotFound:
            return error_response(404, "Run nicht gefunden", "not_found")

    @router.get("/v1/runs/{run_id}/result")
    def get_run_result(
        run_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Return the final report payload for a completed native run."""
        try:
            workspace_id = workspace_id_from_request(req)
            summary = run_store.get(
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except RunNotFound:
            return error_response(404, "Run nicht gefunden", "not_found")
        if summary["status"] != "completed":
            return error_response(
                409,
                "Run ist noch nicht abgeschlossen",
                "run_not_completed",
                status=summary["status"],
            )
        try:
            return run_store.result(
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except RunNotFound:
            return error_response(404, "Run-Ergebnis nicht gefunden", "not_found")

    @router.post("/v1/runs/{run_id}/cancel")
    def cancel_run(
        run_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Request cancellation for a queued or running native run."""
        try:
            workspace_id = workspace_id_from_request(req)
            return run_store.cancel(
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except RunNotFound:
            return error_response(404, "Run nicht gefunden", "not_found")

    @router.delete("/v1/runs/{run_id}", status_code=204)
    async def delete_run(
        run_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
    ):
        """Permanently delete one terminal run (owner-only).

        Durable deletion is the only thing a reload respects: a run removed
        client-side alone re-hydrates from the store on the next list. The
        gate is creator identity (stronger than cancel, which a shared-in
        editor may call); only terminal runs delete (an active run is
        cancelled first). Deleting also revokes any shares so a recipient's
        shared-with-me stops naming a run that no longer exists.
        """
        try:
            workspace_id = workspace_id_from_request(req)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        try:
            await asyncio.to_thread(
                run_store.delete,
                run_id,
                workspace_id=workspace_id,
                requester_sub=principal.sub,
            )
        except RunNotFound:
            return error_response(404, "Run nicht gefunden", "not_found")
        except RunActive:
            return error_response(
                409,
                "Run ist noch aktiv; bitte zuerst abbrechen.",
                "run_active",
            )
        if workspace_admin is not None and share_service is not None:
            revoked = await workspace_admin.revoke_shares_for_resource(
                tenant_id=principal.tenant_id,
                resource_type="run",
                resource_id=run_id,
                revoked_by_sub=principal.sub,
            )
            if revoked:
                log.info(
                    "Run %s geloescht; %d Freigaben entzogen", run_id, revoked
                )

    @router.get("/v1/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_runs_dep),
    ):
        """Stream buffered and live native run events as SSE."""
        try:
            workspace_id = workspace_id_from_request(req)
            subscription = await asyncio.to_thread(
                run_store.subscribe,
                run_id,
                workspace_id=workspace_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except RunNotFound:
            return error_response(404, "Run nicht gefunden", "not_found")

        async def _event_generator():
            try:
                terminal_replayed = False
                for event in subscription.replay:
                    yield format_sse_event(event)
                    terminal_replayed = event.get("type") in TERMINAL_EVENTS
                if terminal_replayed:
                    return
                while True:
                    if await req.is_disconnected():
                        return
                    try:
                        event = await asyncio.to_thread(
                            subscription.queue.get,
                            True,
                            0.5,
                        )
                    except Empty:
                        continue
                    yield format_sse_event(event)
                    if event.get("type") in TERMINAL_EVENTS:
                        return
            finally:
                subscription.close()

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
