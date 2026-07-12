"""Editor-patch endpoints (M7 patch lifecycle).

Thin per-surface router over
:class:`~inqtrix.services.editor_patch_service.EditorPatchService`,
following the ``agent_runs.py`` conventions: manual body parsing, the
``error_response`` envelope, and 404 (never 403) for anything the caller
may not see. The parent-document visibility rule lives in the service —
a foreign document answers exactly like an absent one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.pagination import list_envelope
from inqtrix.project.editor_patch_ports import (
    EditorPatchRecord,
    PatchAlreadyDecided,
    PatchNotFound,
    PatchRevisionConflict,
)
from inqtrix.project.editor_ports import DocumentNotFound
from inqtrix.services.editor_patch_service import EditorPatchValidationError
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

_PATCH_NOT_FOUND = ("Patch nicht gefunden", "not_found")


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the editor-patch routes against the container."""
    service = container.editor_patch_service
    if service is None:
        raise RuntimeError(
            "editor_patches router requires container.editor_patch_service"
        )
    router = APIRouter()
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency
    # No shared-grants dependency here ON PURPOSE: editor documents have
    # no sharing surface today (the ShareService resolves only run /
    # knowledge_collection / prompt_template owners, and
    # editor_persistence.py wires no grants either), so also_visible
    # stays None end to end. When document sharing lands, wire
    # build_shared_grants_dependency for the editor-document resource
    # type here AND in editor_persistence.py together — the service
    # already threads also_visible through.

    @router.get("/v1/editor/documents/{document_id}/patches")
    async def list_document_patches(
        document_id: str,
        req: Request,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Patch METADATA of one document (no edit bodies), newest first.

        ``?status=`` filters to one lifecycle state.
        """
        try:
            patches = await service.list_for_document(
                document_id,
                status=req.query_params.get("status"),
                visible_to=visible_to,
            )
        except EditorPatchValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")
        return list_envelope(
            [_patch_meta_payload(patch) for patch in patches], None
        )

    @router.get("/v1/editor/patches/{patch_id}")
    async def get_patch(
        patch_id: str,
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """One patch with its full edits, plus the document's CURRENT
        revision (the frontend applies against fresh state)."""
        try:
            patch, document_revision = await service.get_patch(
                patch_id, visible_to=visible_to
            )
        except PatchNotFound:
            return error_response(404, *_PATCH_NOT_FOUND)
        return {
            **_patch_detail_payload(patch),
            "document_revision": document_revision,
        }

    @router.post("/v1/editor/patches/{patch_id}:apply")
    async def apply_patch(
        patch_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Apply a pending patch server-side (CAS on the document revision).

        Replaying the SAME apply answers 200 with the stored outcome; a
        stale ``expected_revision`` answers 409 with the current and
        proposed-against revisions.
        """
        body = await _json_object(req)
        if body is None:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        expected = body.get("expected_revision")
        # bool subclasses int — reject it explicitly (the artifact-PUT
        # precedent for numeric preconditions).
        if isinstance(expected, bool) or not isinstance(expected, int):
            return error_response(
                400,
                "expected_revision ist erforderlich (Ganzzahl).",
                "invalid_request_error",
            )
        try:
            patch = await service.apply(
                patch_id,
                expected_revision=expected,
                visible_to=visible_to,
                principal=principal,
            )
        except PatchNotFound:
            return error_response(404, *_PATCH_NOT_FOUND)
        except PatchRevisionConflict as exc:
            return error_response(
                409,
                "Das Dokument wurde zwischenzeitlich geaendert.",
                "conflict",
                current_revision=exc.current_revision,
                revision_before=exc.revision_before,
            )
        except PatchAlreadyDecided as exc:
            return error_response(
                409,
                "Der Patch wurde bereits anders entschieden.",
                "conflict",
                status=exc.patch.status,
            )
        return {
            "document_id": patch.document_id,
            "revision": patch.applied_revision,
            "applied_edit_ids": list(patch.applied_edit_ids or ()),
        }

    @router.post("/v1/editor/patches/{patch_id}:reject")
    async def reject_patch(
        patch_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
    ):
        """Reject a pending patch (optional ``note``); replay answers 200."""
        body = await _json_object(req)
        if body is None:
            body = {}
        note = body.get("note")
        if note is not None and not isinstance(note, str):
            return error_response(
                400, "note muss ein String sein.", "invalid_request_error"
            )
        try:
            patch = await service.reject(
                patch_id,
                note=note or "",
                visible_to=visible_to,
                principal=principal,
            )
        except PatchNotFound:
            return error_response(404, *_PATCH_NOT_FOUND)
        except PatchAlreadyDecided as exc:
            return error_response(
                409,
                "Der Patch wurde bereits anders entschieden.",
                "conflict",
                status=exc.patch.status,
            )
        return _patch_detail_payload(patch)

    return router


async def _json_object(req: Request) -> dict[str, Any] | None:
    """Parse the request body as a JSON object, or ``None`` when malformed."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001 — malformed body is a client error
        return None
    return body if isinstance(body, dict) else None


# -- wire payloads ------------------------------------------------------------ #


def _patch_meta_payload(patch: EditorPatchRecord) -> dict[str, Any]:
    """Patch WITHOUT the edit bodies — the list shape."""
    return {
        "patch_id": patch.patch_id,
        "document_id": patch.document_id,
        "run_id": patch.run_id,
        "source": patch.source,
        "status": patch.status,
        "edit_count": len(patch.edits),
        "summary": patch.summary,
        "revision_before": patch.revision_before,
        "applied_revision": patch.applied_revision,
        "created_at": patch.created_at,
        "decided_at": patch.decided_at,
    }


def _patch_detail_payload(patch: EditorPatchRecord) -> dict[str, Any]:
    """Full patch record including the anchored edits — the detail shape."""
    return {
        **_patch_meta_payload(patch),
        "edits": [_edit_payload(edit) for edit in patch.edits],
        "warnings": list(patch.warnings),
        "applied_edit_ids": (
            list(patch.applied_edit_ids)
            if patch.applied_edit_ids is not None
            else None
        ),
        "note": patch.note,
    }


def _edit_payload(edit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": edit.get("id", ""),
        "find": edit.get("find", ""),
        "quote_before": edit.get("quote_before", ""),
        "quote_after": edit.get("quote_after", ""),
        "position": edit.get("position", ""),
        "text": edit.get("text", ""),
        "note": edit.get("note", ""),
    }
