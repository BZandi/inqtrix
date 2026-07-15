"""File-asset-record endpoints (M6c project tier).

Thin per-surface router mirroring chat_history / editor_persistence: parse,
delegate to
:class:`~inqtrix.services.asset_records_service.AssetRecordsService`,
serialize. Records are private per owner within a workspace (no sharing in
M6c). The asset LIST returns metadata only; the single-asset GET returns the
extracted text (load-on-open).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.pagination import (
    InvalidCursor,
    clamp_limit,
    decode_cursor,
    list_envelope,
)
from inqtrix.project.asset_records_ports import (
    AssetGroup,
    AssetNotFound,
    AssetRecord,
    AssetSection,
    GroupNotFound,
    SectionNotFound,
)
from inqtrix.services.asset_records_service import AssetValidationError
from inqtrix.services.request_parsing import (
    error_response,
    workspace_id_from_request,
)

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer


def _caller_user_id(principal: Principal) -> uuid.UUID | None:
    return principal.user_id if principal.kind in ("oidc_session", "pat") else None


def _section_payload(s: AssetSection) -> dict[str, Any]:
    return {"id": s.id, "kind": s.kind, "title": s.title,
            "created_at": s.created_at, "updated_at": s.updated_at}


def _group_payload(g: AssetGroup) -> dict[str, Any]:
    return {"id": g.id, "section_id": g.section_id, "title": g.title,
            "created_at": g.created_at, "updated_at": g.updated_at}


def _asset_meta_payload(a: AssetRecord) -> dict[str, Any]:
    return {
        "id": a.id, "section_id": a.section_id, "group_id": a.group_id,
        "title": a.title, "label": a.label, "file_name": a.file_name,
        "mime_type": a.mime_type, "origin": a.origin, "page_count": a.page_count,
        "parse_status": a.parse_status, "parse_warning": a.parse_warning,
        "text_truncated": a.text_truncated, "size_bytes": a.size_bytes,
        "server_file_id": a.server_file_id, "parser_id": a.parser_id,
        "created_at": a.created_at, "updated_at": a.updated_at,
    }


def _asset_detail_payload(a: AssetRecord) -> dict[str, Any]:
    return {**_asset_meta_payload(a), "extracted_text": a.extracted_text}


def _require_numbers(body: dict[str, Any]) -> bool:
    return isinstance(body.get("created_at"), (int, float)) and isinstance(
        body.get("updated_at"), (int, float)
    )


def build_router(container: "AppContainer") -> APIRouter:
    service = container.asset_records_service
    if service is None:
        raise RuntimeError("build_router(asset_records) requires a wired service.")
    router = APIRouter()
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency

    # -- sections --------------------------------------------------------- #

    @router.get("/v1/assets/sections")
    async def list_sections(req: Request, principal: Principal = Depends(principal_dep)):
        sections = await service.list_sections(
            caller_user_id=_caller_user_id(principal), workspace_id=workspace_id_from_request(req)
        )
        return {"object": "list", "data": [_section_payload(s) for s in sections]}

    @router.put("/v1/assets/sections/{section_id}")
    async def save_section(
        section_id: str, req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        body = await _json_object(req)
        if not isinstance(body, dict) or not _require_numbers(body):
            return error_response(400, "Ungueltiger Body", "invalid_request_error")
        try:
            section = await service.save_section(
                id=section_id, kind=str(body.get("kind", "custom")),
                title=str(body.get("title", "")), created_at=float(body["created_at"]),
                updated_at=float(body["updated_at"]), caller_user_id=_caller_user_id(principal),
                workspace_id=workspace_id_from_request(req), visible_to=visible_to,
            )
        except AssetValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except SectionNotFound:
            return error_response(404, "Sektion nicht gefunden", "not_found")
        return _section_payload(section)

    @router.delete("/v1/assets/sections/{section_id}", status_code=204)
    async def delete_section(
        section_id: str, req: Request,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        try:
            await service.delete_section(
                section_id, visible_to=visible_to,
                request_workspace_id=workspace_id_from_request(req),
            )
        except SectionNotFound:
            return error_response(404, "Sektion nicht gefunden", "not_found")

    # -- groups ----------------------------------------------------------- #

    @router.get("/v1/assets/groups")
    async def list_groups(req: Request, principal: Principal = Depends(principal_dep)):
        groups = await service.list_groups(
            caller_user_id=_caller_user_id(principal), workspace_id=workspace_id_from_request(req)
        )
        return {"object": "list", "data": [_group_payload(g) for g in groups]}

    @router.put("/v1/assets/groups/{group_id}")
    async def save_group(
        group_id: str, req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        body = await _json_object(req)
        if not isinstance(body, dict) or not _require_numbers(body):
            return error_response(400, "Ungueltiger Body", "invalid_request_error")
        section_id = body.get("section_id")
        if not isinstance(section_id, str) or not section_id:
            return error_response(400, "section_id ist erforderlich", "invalid_request_error")
        try:
            group = await service.save_group(
                id=group_id, section_id=section_id, title=str(body.get("title", "")),
                created_at=float(body["created_at"]), updated_at=float(body["updated_at"]),
                caller_user_id=_caller_user_id(principal),
                workspace_id=workspace_id_from_request(req), visible_to=visible_to,
            )
        except GroupNotFound:
            return error_response(404, "Gruppe nicht gefunden", "not_found")
        return _group_payload(group)

    @router.delete("/v1/assets/groups/{group_id}", status_code=204)
    async def delete_group(
        group_id: str, req: Request,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        try:
            await service.delete_group(
                group_id, visible_to=visible_to,
                request_workspace_id=workspace_id_from_request(req),
            )
        except GroupNotFound:
            return error_response(404, "Gruppe nicht gefunden", "not_found")

    # -- assets ----------------------------------------------------------- #

    @router.get("/v1/assets")
    async def list_assets(req: Request, principal: Principal = Depends(principal_dep)):
        try:
            after = decode_cursor(req.query_params.get("cursor"))
        except InvalidCursor:
            return error_response(400, "Ungueltiger Cursor", "invalid_cursor")
        limit = clamp_limit(req.query_params.get("limit"))
        assets, next_cursor = await service.list_assets(
            caller_user_id=_caller_user_id(principal),
            workspace_id=workspace_id_from_request(req), limit=limit, after=after,
        )
        return list_envelope([_asset_meta_payload(a) for a in assets], next_cursor)

    @router.get("/v1/assets/{asset_id}")
    async def get_asset(asset_id: str, visible_to: UserContext | None = Depends(user_context_dep)):
        try:
            asset = await service.get_asset(asset_id, visible_to=visible_to)
        except AssetNotFound:
            return error_response(404, "Asset nicht gefunden", "not_found")
        return _asset_detail_payload(asset)

    @router.put("/v1/assets/{asset_id}")
    async def save_asset(
        asset_id: str, req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        body = await _json_object(req)
        if not isinstance(body, dict) or not _require_numbers(body):
            return error_response(400, "Ungueltiger Body", "invalid_request_error")
        section_id = body.get("section_id")
        if not isinstance(section_id, str) or not section_id:
            return error_response(400, "section_id ist erforderlich", "invalid_request_error")
        group_id = body.get("group_id")
        if group_id is not None and not isinstance(group_id, str):
            return error_response(400, "group_id muss String oder null sein", "invalid_request_error")
        page_count = body.get("page_count")
        if page_count is not None and not isinstance(page_count, int):
            return error_response(400, "page_count muss Ganzzahl oder null sein", "invalid_request_error")
        try:
            asset = await service.save_asset(
                id=asset_id, section_id=section_id, group_id=group_id,
                title=str(body.get("title", "")), label=str(body.get("label", "")),
                file_name=str(body.get("file_name", "")),
                mime_type=str(body.get("mime_type", "")),
                origin=str(body.get("origin", "library")), page_count=page_count,
                parse_status=str(body.get("parse_status", "parsed")),
                parse_warning=(str(body["parse_warning"]) if body.get("parse_warning") is not None else None),
                text_truncated=bool(body.get("text_truncated", False)),
                size_bytes=int(body.get("size_bytes", 0)),
                server_file_id=(str(body["server_file_id"]) if body.get("server_file_id") is not None else None),
                parser_id=(str(body["parser_id"]) if body.get("parser_id") is not None else None),
                extracted_text=str(body.get("extracted_text", "")),
                created_at=float(body["created_at"]), updated_at=float(body["updated_at"]),
                caller_user_id=_caller_user_id(principal),
                workspace_id=workspace_id_from_request(req), visible_to=visible_to,
            )
        except AssetValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except AssetNotFound:
            return error_response(404, "Asset nicht gefunden", "not_found")
        return _asset_detail_payload(asset)

    @router.delete("/v1/assets/{asset_id}", status_code=204)
    async def delete_asset(
        asset_id: str, req: Request,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        try:
            await service.delete_asset(
                asset_id, visible_to=visible_to,
                request_workspace_id=workspace_id_from_request(req),
            )
        except AssetNotFound:
            return error_response(404, "Asset nicht gefunden", "not_found")

    return router


async def _json_object(req: Request) -> Any:
    try:
        return await req.json()
    except Exception:
        return None
