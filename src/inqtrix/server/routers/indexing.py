"""Background reindex-job endpoints (``/v1/knowledge/...reindex`` + jobs).

Registered alongside the knowledge surface (only when the knowledge
engine is enabled). Submission resolves and edit-checks the target
collection, gates the embedding-token quota, and queues a job on the
:class:`~inqtrix.server.indexing.IndexingJobStore`. Job authorization follows
the parent collection: viewers may inspect its jobs and editors may cancel
them, independent of which authorized user originally submitted the job.
"""

from __future__ import annotations

import asyncio
from queue import Empty
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inqtrix.auth.permissions import SharePermission
from inqtrix.auth.principal import (
    Principal,
    UserContext,
    resolve_live_principal,
)
from inqtrix.knowledge.stores.ports import CollectionNotFound
from inqtrix.quota.models import QuotaDimension
from inqtrix.server.indexing import (
    TERMINAL_INDEXING_EVENTS,
    IndexingJobConflict,
    IndexingJobNotFound,
    IndexingQueueFull,
    format_sse_event,
)
from inqtrix.server.routers import quota_admission
from inqtrix.services.indexing_service import ReindexUnsupported
from inqtrix.services.request_parsing import (
    error_response,
    workspace_id_from_request,
)

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the reindex-job routes against the container.

    Raises:
        RuntimeError: When called without a wired indexing service —
            registration is a composition decision, not a runtime
            fallback (mirrors the knowledge router).
    """
    indexing_service = container.indexing_service
    knowledge_service = container.knowledge_service
    if indexing_service is None or knowledge_service is None:
        raise RuntimeError(
            "build_router(indexing) requires a wired indexing service; "
            "register the router only when knowledge is enabled."
        )
    job_store = indexing_service.job_store
    router = APIRouter()
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency
    quota_service = container.quota_service

    async def _authorized_job(
        job_id: str,
        *,
        visible_to: UserContext | None,
        require_edit: bool = False,
    ) -> dict[str, Any]:
        """Resolve one job through its parent collection authority."""
        try:
            summary = await asyncio.to_thread(job_store.get, job_id)
            collection = await knowledge_service.knowledge.store.get_collection(
                str(summary["collection_id"])
            )
            await knowledge_service.collection_access(
                collection,
                visible_to,
                minimum=(
                    SharePermission.EDIT
                    if require_edit
                    else SharePermission.VIEW
                ),
            )
        except (CollectionNotFound, IndexingJobNotFound, KeyError) as exc:
            raise IndexingJobNotFound(job_id) from exc
        return summary

    @router.post("/v1/knowledge/collections/{collection_id}/reindex")
    async def start_reindex(
        collection_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Queue a background re-embed of one collection's documents."""
        try:
            body = await req.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        index_id = body.get("index_id")
        index_id = str(index_id) if index_id is not None else None
        try:
            workspace_id = workspace_id_from_request(req, body)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)

        # Embedding-token admission (block-next): a caller already over
        # their monthly embedding budget is denied before the job starts.
        # The job's real per-document spend is booked as it embeds.
        denied = await quota_admission(
            quota_service, principal, QuotaDimension.EMBEDDING_TOKENS
        )
        if denied is not None:
            return denied

        try:
            collection = await knowledge_service.knowledge.store.get_collection(
                collection_id
            )
            await knowledge_service.collection_access(
                collection,
                visible_to,
                minimum=SharePermission.EDIT,
            )
            summary = indexing_service.submit(
                collection=collection,
                index_id=index_id,
                workspace_id=workspace_id,
                principal=principal,
            )
        except CollectionNotFound:
            return error_response(404, "Collection nicht gefunden", "not_found")
        except ReindexUnsupported as exc:
            return error_response(501, str(exc), "not_implemented")
        except IndexingJobConflict:
            return error_response(
                409,
                "Fuer diese Sammlung laeuft bereits eine Indizierung.",
                "reindex_in_progress",
            )
        except IndexingQueueFull:
            return error_response(
                429,
                "Zu viele wartende Indizierungen. Bitte warten.",
                "rate_limit_error",
            )
        return JSONResponse(status_code=202, content=summary)

    @router.get("/v1/knowledge/indexing-jobs")
    async def list_jobs(
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """List visible reindex jobs (active and recent), newest first.

        Optional ``?collection_id=`` narrows to one collection's history
        (the inline "last N" view); the resume path lists all active
        jobs for the workspace.
        """
        try:
            workspace_id_from_request(req)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        collection_id = req.query_params.get("collection_id") or None
        visible_collections = await knowledge_service.list_collections(
            visible_to=visible_to,
        )
        visible_collection_ids = {
            collection.id for collection in visible_collections
        }
        jobs = await asyncio.to_thread(
            job_store.list,
            collection_id=collection_id,
        )
        return {
            "object": "list",
            "data": [
                job
                for job in jobs
                if job.get("collection_id") in visible_collection_ids
            ],
        }

    @router.get("/v1/knowledge/indexing-jobs/{job_id}")
    async def get_job(
        job_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Return the current public summary for one reindex job."""
        try:
            workspace_id_from_request(req)
            return await _authorized_job(
                job_id,
                visible_to=visible_to,
            )
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except IndexingJobNotFound:
            return error_response(404, "Indizierung nicht gefunden", "not_found")

    @router.post("/v1/knowledge/indexing-jobs/{job_id}/cancel")
    async def cancel_job(
        job_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Request cancellation for a queued or running reindex job."""
        try:
            workspace_id_from_request(req)
            await _authorized_job(
                job_id,
                visible_to=visible_to,
                require_edit=True,
            )
            return await asyncio.to_thread(
                job_store.cancel,
                job_id,
                actor_user_id=principal.user_id,
            )
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except IndexingJobNotFound:
            return error_response(404, "Indizierung nicht gefunden", "not_found")

    @router.get("/v1/knowledge/indexing-jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Stream buffered and live reindex-job events as SSE."""
        try:
            workspace_id_from_request(req)
            await _authorized_job(
                job_id,
                visible_to=visible_to,
            )
            subscription = await asyncio.to_thread(
                job_store.subscribe,
                job_id,
            )
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except IndexingJobNotFound:
            return error_response(404, "Indizierung nicht gefunden", "not_found")

        async def _event_generator():
            async def _authorized_frame() -> bool:
                try:
                    current = await resolve_live_principal(principal_dep, req)
                    if (
                        current.user_id != principal.user_id
                        or current.kind != principal.kind
                        or current.tenant_id != principal.tenant_id
                        or current.session_id != principal.session_id
                        or current.pat_id != principal.pat_id
                        or current.scopes != principal.scopes
                    ):
                        return False
                    live_visible_to = (
                        await container.permission_service.resolve_user_context(
                            current
                        )
                    )
                    await _authorized_job(
                        job_id,
                        visible_to=live_visible_to,
                    )
                except (HTTPException, IndexingJobNotFound):
                    return False
                return True

            try:
                terminal_replayed = False
                for event in subscription.replay:
                    if not await _authorized_frame():
                        return
                    yield format_sse_event(event)
                    terminal_replayed = event.get("type") in TERMINAL_INDEXING_EVENTS
                if terminal_replayed:
                    return
                while True:
                    if await req.is_disconnected():
                        return
                    try:
                        event = await asyncio.to_thread(
                            subscription.queue.get, True, 0.5
                        )
                    except Empty:
                        continue
                    if not await _authorized_frame():
                        return
                    yield format_sse_event(event)
                    if event.get("type") in TERMINAL_INDEXING_EVENTS:
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
