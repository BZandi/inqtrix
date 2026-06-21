"""OpenAI-compatible ``/v1/chat/completions`` endpoint.

Streaming responses go through :func:`guarded_stream` (module-global
lookup at call time — the monkeypatch seam tests rely on); the
non-streaming path delegates to the
:class:`~inqtrix.services.chat_service.ChatService`.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.knowledge.stores.ports import CollectionNotFound
from inqtrix.quota.models import QuotaDimension, consumed_tokens
from inqtrix.server.routers import (
    build_shared_grants_dependency,
    quota_admission,
    quota_record,
    stack_error_response,
)
from inqtrix.server.streaming import guarded_stream
from inqtrix.services.agent_context import StackResolutionError
from inqtrix.services.request_parsing import (
    format_history,
    question_and_messages,
    workspace_id_from_request,
)

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the chat-completions route against the container."""
    router = APIRouter()
    settings = container.settings
    resolver = container.resolver
    chat_service = container.chat_service
    knowledge_service = container.knowledge_service
    quota_service = container.quota_service
    user_context_dep = container.user_context_dependency
    shared_collections_dep = build_shared_grants_dependency(
        container.share_service,
        container.principal_dependency,
        resource_type="knowledge_collection",
    )

    async def _deny_invisible_collections(
        resolved_filters: dict,
        visible_to: "UserContext | None",
        collection_grants,
    ) -> JSONResponse | None:
        """Admission gate for asks against knowledge collections.

        Strict: ONE invisible collection denies the whole request —
        silently answering from fewer collections than the caller
        picked would change the answer's meaning without a trace. The
        worker only re-executes admitted requests, so this gate covers
        all three execution paths.
        """
        requested = resolved_filters.get("collection_ids")
        if (
            knowledge_service is None
            or visible_to is None
            or not isinstance(requested, list)
            or not requested
        ):
            return None
        try:
            await knowledge_service.assert_collections_visible(
                [str(item) for item in requested],
                visible_to=visible_to,
                also_visible=collection_grants,
            )
        except CollectionNotFound:
            return JSONResponse(
                status_code=404,
                content={"error": {
                    "message": "Collection nicht gefunden",
                    "type": "not_found",
                }},
            )
        return None

    @router.post("/v1/chat/completions")
    async def chat_completions(
        req: Request,
        principal: Principal = Depends(container.principal_dependency),
        visible_to: UserContext | None = Depends(user_context_dep),
        collection_grants=Depends(shared_collections_dep),
    ):
        try:
            body = await req.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": "Ungueltiger JSON-Body",
                    "type": "invalid_request_error",
                }},
            )

        try:
            workspace_id = workspace_id_from_request(req, body)
            question, messages = question_and_messages(body, settings.server)
            resolved = resolver.resolve(body)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        except StackResolutionError as exc:
            return stack_error_response(exc)

        denied = await _deny_invisible_collections(
            resolved.knowledge_filters, visible_to, collection_grants
        )
        if denied is not None:
            return denied

        if (
            not resolved.agent_settings.skip_search
            and len(question) > resolved.agent_settings.max_question_length
        ):
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": (
                        f"Frage zu lang ({len(question)} Zeichen, "
                        f"max. {resolved.agent_settings.max_question_length})"
                    ),
                    "type": "invalid_request_error",
                }},
            )
        chat_agent_settings = resolver.chat_settings_for_question(
            resolved.agent_settings,
            question,
        )

        # Quota admission covers BOTH the streamed and non-streamed
        # branch (it runs before either): a caller over their monthly
        # LLM-token budget is blocked here, the real spend is recorded
        # after the answer is produced.
        denied = await quota_admission(
            quota_service, principal, QuotaDimension.LLM_TOKENS
        )
        if denied is not None:
            return denied

        history = format_history(
            messages, max_messages=settings.server.max_messages_history
        )

        stream = body.get("stream", False)
        include_progress_raw = body.get("include_progress", True)
        include_progress = (
            include_progress_raw
            if isinstance(include_progress_raw, bool)
            else True
        )

        sem = container.semaphore_factory()
        if sem.locked():
            return JSONResponse(
                status_code=429,
                content={"error": {
                    "message": "Zu viele gleichzeitige Anfragen. Bitte warten.",
                    "type": "rate_limit_error",
                }},
            )

        if stream:
            algorithm = container.registry.get(resolved.mode)
            if not algorithm.capabilities().get("streams_via_research_graph"):
                # The streamed path still executes the research graph
                # directly (server/streaming.py); running a non-graph
                # algorithm through it would silently execute the wrong
                # engine. Reject loudly until streaming dispatches
                # through the registry.
                return JSONResponse(
                    status_code=400,
                    content={"error": {
                        "message": (
                            f"mode='{resolved.mode}' unterstuetzt "
                            "stream=true noch nicht"
                        ),
                        "type": "invalid_request_error",
                    }},
                )
            cancel_event = threading.Event()
            return StreamingResponse(
                guarded_stream(
                    question,
                    history,
                    sem,
                    providers=resolved.providers,
                    strategies=resolved.strategies,
                    settings=chat_agent_settings,
                    messages=messages,
                    include_progress=include_progress,
                    request=req,
                    cancel_event=cancel_event,
                    stack_name=resolved.stack_name,
                    workspace_id=workspace_id or "",
                    quota_service=quota_service,
                    principal=principal,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await chat_service.complete(
            question=question,
            history=history,
            messages=messages,
            resolved=resolved,
            chat_agent_settings=chat_agent_settings,
            semaphore=sem,
            principal=principal,
        )
        # Book the real token spend on success; an error envelope
        # (JSONResponse) carries no usage and is not metered.
        if isinstance(response, dict):
            await quota_record(
                quota_service,
                principal,
                QuotaDimension.LLM_TOKENS,
                consumed_tokens(response.get("usage")),
            )
        return response

    return router
