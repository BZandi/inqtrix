"""Editor-persistence endpoints (M6b project tier).

Thin per-surface router mirroring ``chat_history.py``: parse, delegate to
:class:`~inqtrix.services.editor_persistence_service.EditorPersistenceService`,
serialize. Documents are private per-owner within a workspace (no sharing
surface in M6b). The list endpoint returns document METADATA only; the
single-document GET returns the body (load-on-open).
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
from inqtrix.project.editor_ports import (
    DocumentNotFound,
    DocumentRevisionConflict,
    EditorComment,
    EditorDocument,
    EditorFolder,
    FolderNotFound,
)
from inqtrix.services.editor_persistence_service import EditorValidationError
from inqtrix.services.request_parsing import (
    error_response,
    workspace_id_from_request,
)

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer


def _caller_sub(principal: Principal) -> str | None:
    return principal.sub if principal.kind in ("oidc_session", "pat") else None


def _document_meta_payload(document: EditorDocument) -> dict[str, Any]:
    """Document WITHOUT the body — the list shape."""
    return {
        "id": document.id,
        "title": document.title,
        "folder_id": document.folder_id,
        "source": document.source,
        "source_run_id": document.source_run_id,
        "revision": document.revision,
        "diff_anchor_markdown": document.diff_anchor_markdown,
        "diff_anchor_updated_at": document.diff_anchor_updated_at,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }


def _document_detail_payload(document: EditorDocument) -> dict[str, Any]:
    """Document WITH the body — the get-one / upsert-response shape."""
    return {**_document_meta_payload(document), "content_markdown": document.content_markdown}


def _folder_payload(folder: EditorFolder) -> dict[str, Any]:
    return {
        "id": folder.id,
        "title": folder.title,
        "created_at": folder.created_at,
        "updated_at": folder.updated_at,
    }


def _comment_payload(comment: EditorComment) -> dict[str, Any]:
    return {
        "id": comment.id,
        "document_id": comment.document_id,
        "comment_markdown": comment.comment_markdown,
        "anchor": dict(comment.anchor),
        "kind": comment.kind,
        "status": comment.status,
        "evidence_preset": comment.evidence_preset,
        "created_at": comment.created_at,
        "updated_at": comment.updated_at,
    }


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the editor-persistence routes against the container."""
    service = container.editor_persistence_service
    if service is None:
        raise RuntimeError(
            "build_router(editor_persistence) requires a wired editor service."
        )
    router = APIRouter()
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency

    # -- documents -------------------------------------------------------- #

    @router.get("/v1/editor/documents")
    async def list_documents(
        req: Request,
        principal: Principal = Depends(principal_dep),
    ):
        """One keyset page of the caller's documents (metadata only)."""
        try:
            after = decode_cursor(req.query_params.get("cursor"))
        except InvalidCursor:
            return error_response(400, "Ungueltiger Cursor", "invalid_cursor")
        limit = clamp_limit(req.query_params.get("limit"))
        documents, next_cursor = await service.list_documents(
            caller_sub=_caller_sub(principal),
            workspace_id=workspace_id_from_request(req),
            limit=limit,
            after=after,
        )
        return list_envelope(
            [_document_meta_payload(document) for document in documents],
            next_cursor,
        )

    @router.get("/v1/editor/documents/{document_id}")
    async def get_document(
        document_id: str,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """One document WITH its body (load-on-open)."""
        try:
            document = await service.get_document(
                document_id, visible_to=visible_to
            )
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")
        return _document_detail_payload(document)

    @router.put("/v1/editor/documents/{document_id}")
    async def save_document(
        document_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Create or idempotently update a document (autosave upsert)."""
        body = await _json_object(req)
        if not isinstance(body, dict):
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        created_at = body.get("created_at")
        updated_at = body.get("updated_at")
        if not isinstance(created_at, (int, float)) or not isinstance(
            updated_at, (int, float)
        ):
            return error_response(
                400, "created_at und updated_at muessen Zahlen sein",
                "invalid_request_error",
            )
        folder_id = body.get("folder_id")
        if folder_id is not None and not isinstance(folder_id, str):
            return error_response(
                400, "folder_id muss ein String oder null sein",
                "invalid_request_error",
            )
        diff_anchor_updated_at = body.get("diff_anchor_updated_at")
        if diff_anchor_updated_at is not None and not isinstance(
            diff_anchor_updated_at, (int, float)
        ):
            return error_response(
                400, "diff_anchor_updated_at muss eine Zahl oder null sein",
                "invalid_request_error",
            )
        revision = body.get("revision", 1)
        if not isinstance(revision, int):
            return error_response(
                400, "revision muss eine Ganzzahl sein", "invalid_request_error"
            )
        try:
            document = await service.save_document(
                id=document_id,
                title=str(body.get("title", "")),
                content_markdown=str(body.get("content_markdown", "")),
                folder_id=folder_id,
                source=str(body.get("source", "blank")),
                source_run_id=(
                    str(body["source_run_id"])
                    if body.get("source_run_id") is not None
                    else None
                ),
                revision=revision,
                diff_anchor_markdown=(
                    str(body["diff_anchor_markdown"])
                    if body.get("diff_anchor_markdown") is not None
                    else None
                ),
                diff_anchor_updated_at=(
                    float(diff_anchor_updated_at)
                    if diff_anchor_updated_at is not None
                    else None
                ),
                created_at=float(created_at),
                updated_at=float(updated_at),
                caller_sub=_caller_sub(principal),
                workspace_id=workspace_id_from_request(req),
                visible_to=visible_to,
            )
        except EditorValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")
        except DocumentRevisionConflict as exc:
            # A concurrent writer (agent patch apply vs. this autosave)
            # moved the document past the caller's base. 409 with the
            # live revision — the client refetches and rebases (same
            # envelope as the artifact/patch CAS conflicts).
            return error_response(
                409,
                "Das Dokument wurde zwischenzeitlich geaendert — bitte "
                "neu laden und erneut speichern.",
                "conflict",
                current_revision=exc.current_revision,
            )
        return _document_detail_payload(document)

    @router.delete("/v1/editor/documents/{document_id}", status_code=204)
    async def delete_document(
        document_id: str,
        req: Request,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Delete a document and its comments (owner-only)."""
        try:
            await service.delete_document(
                document_id,
                visible_to=visible_to,
                request_workspace_id=workspace_id_from_request(req),
            )
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")

    # -- comments --------------------------------------------------------- #

    @router.get("/v1/editor/documents/{document_id}/comments")
    async def list_comments(
        document_id: str,
        req: Request,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """One keyset page of a document's comments (newest first)."""
        try:
            after = decode_cursor(req.query_params.get("cursor"))
        except InvalidCursor:
            return error_response(400, "Ungueltiger Cursor", "invalid_cursor")
        limit = clamp_limit(req.query_params.get("limit"))
        try:
            comments, next_cursor = await service.list_comments(
                document_id, limit=limit, after=after, visible_to=visible_to
            )
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")
        return list_envelope(
            [_comment_payload(comment) for comment in comments], next_cursor
        )

    @router.post("/v1/editor/documents/{document_id}/comments", status_code=201)
    async def save_comments(
        document_id: str,
        req: Request,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Upsert comments into a document the caller may edit."""
        body = await _json_object(req)
        if not isinstance(body, dict):
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        raw_comments = body.get("comments")
        if not isinstance(raw_comments, list):
            return error_response(
                400, "Feld 'comments' muss eine Liste sein",
                "invalid_request_error",
            )
        try:
            stored = await service.save_comments(
                document_id, comments=raw_comments, visible_to=visible_to
            )
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")
        except EditorValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        return {
            "object": "list",
            "data": [_comment_payload(comment) for comment in stored],
        }

    @router.delete(
        "/v1/editor/documents/{document_id}/comments/{comment_id}",
        status_code=204,
    )
    async def delete_comment(
        document_id: str,
        comment_id: str,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Delete one comment from a document the caller may edit."""
        try:
            await service.delete_comment(
                document_id, comment_id, visible_to=visible_to
            )
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")

    # -- folders ---------------------------------------------------------- #

    @router.get("/v1/editor/folders")
    async def list_folders(
        req: Request,
        principal: Principal = Depends(principal_dep),
    ):
        """All of the caller's editor folders (newest first)."""
        folders = await service.list_folders(
            caller_sub=_caller_sub(principal),
            workspace_id=workspace_id_from_request(req),
        )
        return {
            "object": "list",
            "data": [_folder_payload(folder) for folder in folders],
        }

    @router.put("/v1/editor/folders/{folder_id}")
    async def save_folder(
        folder_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Create or idempotently update an editor folder."""
        body = await _json_object(req)
        if not isinstance(body, dict):
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        created_at = body.get("created_at")
        updated_at = body.get("updated_at")
        if not isinstance(created_at, (int, float)) or not isinstance(
            updated_at, (int, float)
        ):
            return error_response(
                400, "created_at und updated_at muessen Zahlen sein",
                "invalid_request_error",
            )
        try:
            folder = await service.save_folder(
                id=folder_id,
                title=str(body.get("title", "")),
                created_at=float(created_at),
                updated_at=float(updated_at),
                caller_sub=_caller_sub(principal),
                workspace_id=workspace_id_from_request(req),
                visible_to=visible_to,
            )
        except FolderNotFound:
            return error_response(404, "Ordner nicht gefunden", "not_found")
        return _folder_payload(folder)

    @router.delete("/v1/editor/folders/{folder_id}", status_code=204)
    async def delete_folder(
        folder_id: str,
        req: Request,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Delete a folder; its documents orphan to ungrouped."""
        try:
            await service.delete_folder(
                folder_id,
                visible_to=visible_to,
                request_workspace_id=workspace_id_from_request(req),
            )
        except FolderNotFound:
            return error_response(404, "Ordner nicht gefunden", "not_found")

    return router


async def _json_object(req: Request) -> Any:
    """Parse the request body as JSON, or ``None`` on a malformed body."""
    try:
        return await req.json()
    except Exception:
        return None
