"""File endpoints (``/v1/files*``): upload, list, metadata, download, delete.

Thin layer over the :class:`~inqtrix.services.file_service.FileService`.
Downloads stream through the API after the permission check — the
object store itself is never exposed to clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

import re

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.content.ports import FileNotFound, FileRecord
from inqtrix.quota.models import QuotaDimension, QuotaSubject
from inqtrix.server.routers import (
    quota_admission,
    quota_record_for_subject,
)
from inqtrix.services.file_service import (
    FileParserUnavailable,
    FileTextExtractionError,
    FileTooLarge,
)
from inqtrix.services.request_parsing import (
    error_response,
    workspace_id_from_request,
)
from inqtrix.storage.object_store import ObjectStoreError

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

_UPLOAD_CHUNK_BYTES = 1024 * 1024

_DISPOSITION_SAFE = re.compile(r"[^A-Za-z0-9._ ()\[\]-]")


def _content_disposition(file_name: str) -> str:
    """Build a header-injection-safe attachment disposition.

    The original filename is display data from the client; everything
    outside a conservative ASCII subset is replaced so the header can
    never carry CR/LF, quotes, or non-ASCII bytes.
    """
    safe = _DISPOSITION_SAFE.sub("_", file_name).strip() or "download"
    return f'attachment; filename="{safe}"'


def _file_payload(record: FileRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "file_name": record.file_name,
        "content_type": record.content_type,
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
        "workspace_id": record.workspace_id,
        "created_at": record.created_at,
    }


async def _upload_chunks(upload: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
        yield chunk


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the file routes against the container.

    Raises:
        RuntimeError: When called without a wired file service —
            registration is a composition decision, not a runtime
            fallback.
    """
    service = container.file_service
    if service is None:
        raise RuntimeError(
            "build_router(files) requires a wired file service."
        )
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency
    quota_service = container.quota_service

    def _owner_subject(record: FileRecord) -> QuotaSubject | None:
        """The metered owner of a file's stored bytes, or ``None``.

        Stock occupancy belongs to the uploader (``owner_sub``); legacy
        unscoped uploads carry no subject and are not metered.
        """
        if not record.owner_sub:
            return None
        return QuotaSubject(tenant_id=record.tenant_id, sub=record.owner_sub)

    router = APIRouter()

    max_file_bytes = container.settings.storage.max_file_bytes
    # Generous allowance for multipart boundaries/headers around the
    # actual file part.
    max_request_bytes = max_file_bytes + 64 * 1024

    @router.post("/v1/files", status_code=201)
    async def upload_file(
        req: Request,
        file: UploadFile,
        principal: Principal = Depends(principal_dep),
    ):
        """Accept one multipart file upload and register it.

        The Content-Length precheck below rejects oversized requests
        before the framework parses (and disk-spools) the body; the
        service's running-size check remains the authoritative limit
        for chunked or lying clients. Deployments exposed directly to
        untrusted networks should additionally cap request bodies at
        the reverse proxy.
        """
        declared = req.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_request_bytes:
            return error_response(
                413,
                f"Datei zu gross (Limit {max_file_bytes} Bytes)",
                "invalid_request_error",
            )
        try:
            workspace_id = workspace_id_from_request(req)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        # Stored-bytes admission (block-next): a caller already at their
        # occupancy limit is denied before the body is spooled. The
        # multipart Content-Length includes envelope overhead and is not
        # the file size, so it is NOT used as the amount; the exact size
        # is measured server-side and booked after. A single upload
        # cannot run away — it is bounded by ``max_file_bytes``.
        denied = await quota_admission(
            quota_service, principal, QuotaDimension.STORED_BYTES
        )
        if denied is not None:
            return denied
        try:
            record = await service.upload(
                chunks=_upload_chunks(file),
                file_name=file.filename or "upload",
                content_type=file.content_type or "application/octet-stream",
                principal=principal,
                workspace_id=workspace_id,
            )
        except FileTooLarge as exc:
            return error_response(
                413,
                (
                    "Datei zu gross (Limit "
                    f"{exc.limit_bytes} Bytes)"
                ),
                "invalid_request_error",
            )
        except ObjectStoreError:
            return error_response(
                502,
                "Datei konnte nicht gespeichert werden (Object Store)",
                "server_error",
            )
        # Book the exact stored size against the owner (the uploader).
        await quota_record_for_subject(
            quota_service,
            _owner_subject(record),
            QuotaDimension.STORED_BYTES,
            record.size_bytes,
        )
        return JSONResponse(status_code=201, content=_file_payload(record))

    @router.get("/v1/files")
    async def list_files(
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """List files visible to the principal, newest first."""
        try:
            workspace_id = workspace_id_from_request(req)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        records = await service.list(
            principal=principal,
            user_context_is_scoped=visible_to is not None,
            workspace_id=workspace_id,
        )
        return {
            "object": "list",
            "data": [_file_payload(record) for record in records],
        }

    @router.get("/v1/files/{file_id}")
    async def get_file(
        file_id: str,
        principal: Principal = Depends(principal_dep),
    ):
        """Return one file's metadata after the access check."""
        try:
            record = await service.get(file_id, principal=principal)
        except FileNotFound:
            return error_response(404, "Datei nicht gefunden", "not_found")
        return _file_payload(record)

    @router.get("/v1/files/{file_id}/content")
    async def download_file(
        file_id: str,
        principal: Principal = Depends(principal_dep),
    ):
        """Stream the file bytes after the access check."""
        try:
            record, chunks = await service.open_stream(
                file_id, principal=principal
            )
        except FileNotFound:
            return error_response(404, "Datei nicht gefunden", "not_found")
        except ObjectStoreError:
            return error_response(
                502,
                "Dateiinhalt nicht abrufbar (Object Store)",
                "server_error",
            )
        return StreamingResponse(
            chunks,
            media_type=record.content_type,
            headers={
                "Content-Length": str(record.size_bytes),
                "Content-Disposition": _content_disposition(record.file_name),
                # The content type is client-declared at upload; never
                # let browsers second-guess it into something active.
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.get("/v1/files/{file_id}/text")
    async def file_text(
        file_id: str,
        principal: Principal = Depends(principal_dep),
    ):
        """Server-side extracted text of one file via the parser ladder.

        The browser calls this in the BACKGROUND right after upload (it does
        not block the file from appearing): the server parser (MarkItDown by
        default) is the authoritative, browser-independent source for ingestion
        text, where the in-browser parse may fail (e.g. pdf.js on Safari).
        Returns ``501`` when no parser is configured (the client keeps its
        local parse) and ``422`` when the file cannot be converted -- a visible
        error, never a silent empty body (Designprinzip 1).
        """
        try:
            extracted = await service.extract_text(file_id, principal=principal)
        except FileParserUnavailable as exc:
            return error_response(501, str(exc), "not_implemented")
        except FileNotFound:
            return error_response(404, "Datei nicht gefunden", "not_found")
        except ObjectStoreError:
            return error_response(
                502,
                "Dateiinhalt nicht abrufbar (Object Store)",
                "server_error",
            )
        except FileTextExtractionError as exc:
            return error_response(422, str(exc), "unprocessable_entity")
        return {
            "file_id": extracted.file_id,
            "parser_id": extracted.parser_id,
            "text": extracted.text,
        }

    @router.delete("/v1/files/{file_id}", status_code=204)
    async def delete_file(
        file_id: str,
        principal: Principal = Depends(principal_dep),
    ):
        """Delete metadata and blob after the manage-access check.

        Returns the deleted record so the owner's stored-bytes stock is
        freed by exactly what was held — the owner, never the deleter,
        since the bytes belonged to the owner.
        """
        try:
            record = await service.delete(file_id, principal=principal)
        except FileNotFound:
            return error_response(404, "Datei nicht gefunden", "not_found")
        await quota_record_for_subject(
            quota_service,
            _owner_subject(record),
            QuotaDimension.STORED_BYTES,
            -record.size_bytes,
        )
        return Response(status_code=204)

    return router
