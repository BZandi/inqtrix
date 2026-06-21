"""Vector-index-record endpoints (M6c project tier).

Thin per-surface router mirroring asset_records: parse, delegate to
:class:`~inqtrix.services.vector_index_service.VectorIndexService`,
serialize. Records are private per owner within a workspace (no sharing in
M6c). The list returns FULL records (members + history) — there is no heavy
body to defer — so there is no single-record GET.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.pagination import (
    InvalidCursor,
    clamp_limit,
    decode_cursor,
    list_envelope,
)
from inqtrix.project.vector_index_ports import (
    VectorIndexHistoryEntry,
    VectorIndexMember,
    VectorIndexNotFound,
    VectorIndexRecord,
)
from inqtrix.services.request_parsing import (
    error_response,
    workspace_id_from_request,
)
from inqtrix.services.vector_index_service import VectorIndexValidationError

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer


def _caller_sub(principal: Principal) -> str | None:
    return principal.sub if principal.kind in ("oidc_session", "pat") else None


def _record_payload(r: VectorIndexRecord) -> dict[str, Any]:
    return {
        "id": r.id, "title": r.title, "handle": r.handle, "model": r.model,
        "dims": r.dims, "status": r.status,
        "server_collection_id": r.server_collection_id,
        "server_collection_model": r.server_collection_model,
        "last_error": r.last_error,
        "members": [
            {
                "file_id": m.file_id,
                "state": m.state,
                "server_document_id": m.server_document_id,
            }
            for m in r.members
        ],
        "history": [
            {
                "result": h.result, "documents": h.documents,
                "duration_ms": h.duration_ms, "error": h.error,
                "started_at": h.started_at, "finished_at": h.finished_at,
            }
            for h in r.history
        ],
        "created_at": r.created_at, "updated_at": r.updated_at,
    }


def _require_numbers(body: dict[str, Any]) -> bool:
    return isinstance(body.get("created_at"), (int, float)) and isinstance(
        body.get("updated_at"), (int, float)
    )


class _PayloadError(ValueError):
    """A malformed members/history array (maps to HTTP 400)."""


def _parse_members(raw: Any) -> tuple[VectorIndexMember, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _PayloadError("members muss eine Liste sein")
    members: list[VectorIndexMember] = []
    for item in raw:
        if not isinstance(item, dict):
            raise _PayloadError("member muss ein Objekt sein")
        file_id = item.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            raise _PayloadError("member.file_id ist erforderlich")
        raw_doc_id = item.get("server_document_id")
        server_document_id = raw_doc_id if isinstance(raw_doc_id, str) and raw_doc_id else None
        members.append(
            VectorIndexMember(
                file_id=file_id,
                state=str(item.get("state", "pending")),
                server_document_id=server_document_id,
            )
        )
    return tuple(members)


def _parse_history(raw: Any) -> tuple[VectorIndexHistoryEntry, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _PayloadError("history muss eine Liste sein")
    entries: list[VectorIndexHistoryEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            raise _PayloadError("history-Eintrag muss ein Objekt sein")
        if not isinstance(item.get("started_at"), (int, float)) or not isinstance(
            item.get("finished_at"), (int, float)
        ):
            raise _PayloadError("history-Eintrag braucht numerische Zeitstempel")
        documents = item.get("documents", 0)
        duration_ms = item.get("duration_ms", 0)
        if not isinstance(documents, int) or not isinstance(duration_ms, int):
            raise _PayloadError("history-Eintrag braucht ganzzahlige documents/duration_ms")
        error = item.get("error")
        entries.append(VectorIndexHistoryEntry(
            result=str(item.get("result", "ok")),
            documents=int(documents),
            duration_ms=int(duration_ms),
            error=str(error) if error is not None else None,
            started_at=float(item["started_at"]),
            finished_at=float(item["finished_at"]),
        ))
    return tuple(entries)


def build_router(container: "AppContainer") -> APIRouter:
    service = container.vector_index_service
    if service is None:
        raise RuntimeError("build_router(vector_indexes) requires a wired service.")
    router = APIRouter()
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency

    @router.get("/v1/vector-indexes")
    async def list_indexes(req: Request, principal: Principal = Depends(principal_dep)):
        try:
            after = decode_cursor(req.query_params.get("cursor"))
        except InvalidCursor:
            return error_response(400, "Ungueltiger Cursor", "invalid_cursor")
        limit = clamp_limit(req.query_params.get("limit"))
        indexes, next_cursor = await service.list_indexes(
            caller_sub=_caller_sub(principal),
            workspace_id=workspace_id_from_request(req), limit=limit, after=after,
        )
        return list_envelope([_record_payload(r) for r in indexes], next_cursor)

    @router.put("/v1/vector-indexes/{index_id}")
    async def save_index(
        index_id: str, req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        body = await _json_object(req)
        if not isinstance(body, dict) or not _require_numbers(body):
            return error_response(400, "Ungueltiger Body", "invalid_request_error")
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return error_response(400, "model ist erforderlich", "invalid_request_error")
        dims = body.get("dims")
        if not isinstance(dims, int):
            return error_response(400, "dims muss Ganzzahl sein", "invalid_request_error")
        try:
            members = _parse_members(body.get("members"))
            history = _parse_history(body.get("history"))
        except _PayloadError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        try:
            index = await service.save_index(
                id=index_id, title=str(body.get("title", "")),
                handle=str(body.get("handle", "")), model=model, dims=dims,
                status=str(body.get("status", "stale")),
                server_collection_id=(
                    str(body["server_collection_id"])
                    if body.get("server_collection_id") is not None else None
                ),
                server_collection_model=(
                    str(body["server_collection_model"])
                    if body.get("server_collection_model") is not None else None
                ),
                last_error=(
                    str(body["last_error"]) if body.get("last_error") is not None else None
                ),
                members=members, history=history,
                created_at=float(body["created_at"]), updated_at=float(body["updated_at"]),
                caller_sub=_caller_sub(principal),
                workspace_id=workspace_id_from_request(req), visible_to=visible_to,
            )
        except VectorIndexValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except VectorIndexNotFound:
            return error_response(404, "Index nicht gefunden", "not_found")
        return _record_payload(index)

    @router.delete("/v1/vector-indexes/{index_id}", status_code=204)
    async def delete_index(
        index_id: str, req: Request,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        try:
            await service.delete_index(
                index_id, visible_to=visible_to,
                request_workspace_id=workspace_id_from_request(req),
            )
        except VectorIndexNotFound:
            return error_response(404, "Index nicht gefunden", "not_found")

    return router


async def _json_object(req: Request) -> Any:
    try:
        return await req.json()
    except Exception:
        return None
