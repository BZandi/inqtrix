"""Editor capabilities: document context (read) and patch lifecycle (write).

The wave-1 read capability bundles a document with its comments; the M7
write pair (``editor.patch.propose`` / ``editor.patch.apply``) are THIN
wrappers over
:class:`~inqtrix.services.editor_patch_service.EditorPatchService` —
proposals created through the capability always carry ``source='agent'``
and the run attribution from the injected context.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from inqtrix.capabilities.contracts import (
    CapabilityContext,
    CapabilityDefinition,
    CapabilityError,
    Effect,
)
from inqtrix.project.editor_patch_ports import (
    PatchAlreadyDecided,
    PatchNotFound,
    PatchRevisionConflict,
)
from inqtrix.project.editor_ports import DocumentNotFound
from inqtrix.services.editor_patch_service import EditorPatchValidationError
from inqtrix.services.editor_persistence_service import (
    CollaborationProjectionUnavailable,
)

if TYPE_CHECKING:
    from inqtrix.services.editor_patch_service import EditorPatchService
    from inqtrix.services.editor_persistence_service import (
        EditorPersistenceService,
    )

class DocumentContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., min_length=1)


class EditorCommentView(BaseModel):
    id: str
    comment_markdown: str
    kind: str
    status: str


class DocumentContextOutput(BaseModel):
    id: str
    title: str
    content_markdown: str
    revision: int
    comments: list[EditorCommentView]


def build_editor_capabilities(
    service: "EditorPersistenceService",
) -> list[CapabilityDefinition]:
    """Build the wave-1 editor capabilities bound to *service*."""

    async def _read_context(
        payload: DocumentContextInput, context: CapabilityContext
    ) -> DocumentContextOutput:
        try:
            document, comments = await service.get_document_context(
                payload.document_id,
                visible_to=context.visible_to,
            )
        except DocumentNotFound as exc:
            raise CapabilityError(
                "editor.document_not_found",
                "Dokument nicht gefunden.",
                http_status=404,
            ) from exc
        except CollaborationProjectionUnavailable as exc:
            raise CapabilityError(
                "editor.collaboration_projection_unavailable",
                "Der aktuelle Kollaborationsstand konnte nicht gespeichert werden.",
                http_status=503,
            ) from exc
        return DocumentContextOutput(
            id=document.id,
            title=document.title,
            content_markdown=document.content_markdown,
            revision=document.revision,
            comments=[
                EditorCommentView(
                    id=comment.id,
                    comment_markdown=comment.comment_markdown,
                    kind=comment.kind,
                    status=comment.status,
                )
                for comment in comments
            ],
        )

    return [
        CapabilityDefinition(
            id="editor.document.read",
            summary="Read one editor document with its comments as context.",
            input_model=DocumentContextInput,
            output_model=DocumentContextOutput,
            effect=Effect.READ,
            idempotent=True,
            handler=_read_context,
        ),
    ]


# -- patch lifecycle (M7 write pair) ---------------------------------------- #


class PatchEditInput(BaseModel):
    """One anchored edit in the ``editor_instructions`` shape."""

    model_config = ConfigDict(extra="forbid")

    find: str = ""
    quote_before: str = ""
    quote_after: str = ""
    position: Literal["replace", "before", "after", "append"]
    text: str = ""
    note: str = ""


class PatchProposeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., min_length=1)
    edits: list[PatchEditInput] = Field(..., min_length=1)
    summary: str = ""
    expected_revision: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Document revision the caller read before proposing (P7-E1); "
            "a mismatch refuses the proposal with "
            "editor.patch_revision_conflict. None skips the pin."
        ),
    )


class PatchProposeOutput(BaseModel):
    patch_id: str
    document_id: str
    status: str
    edit_count: int
    revision_before: int


class PatchApplyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch_id: str = Field(..., min_length=1)
    expected_revision: int


class PatchApplyOutput(BaseModel):
    document_id: str
    revision: int
    applied_edit_ids: list[str]


def _capability_user_id(context: CapabilityContext) -> uuid.UUID | None:
    """Return the canonical caller ID used by scoped persistence surfaces."""
    principal = context.principal
    if principal is not None and principal.kind in ("oidc_session", "pat"):
        return principal.user_id
    return None


def build_editor_patch_capabilities(
    service: "EditorPatchService",
) -> list[CapabilityDefinition]:
    """Build the M7 patch write capabilities bound to *service*.

    Thin wrappers only: the visibility rule, the revision snapshot, and
    the CAS apply live in the service. Registered exclusively when the
    patch service is wired (conditional-registration pattern).
    """

    async def _propose(
        payload: PatchProposeInput, context: CapabilityContext
    ) -> PatchProposeOutput:
        try:
            patch = await service.propose(
                document_id=payload.document_id,
                run_id=context.run_id,
                source="agent",
                edits=[edit.model_dump() for edit in payload.edits],
                summary=payload.summary,
                warnings=[],
                created_by_user_id=_capability_user_id(context),
                visible_to=context.visible_to,
                principal=context.principal,
                expected_revision=payload.expected_revision,
            )
        except DocumentNotFound as exc:
            raise CapabilityError(
                "editor.document_not_found",
                "Dokument nicht gefunden.",
                http_status=404,
            ) from exc
        except EditorPatchValidationError as exc:
            raise CapabilityError(
                "editor.patch_invalid",
                str(exc),
                http_status=400,
            ) from exc
        except PatchRevisionConflict as exc:
            # Same conflict vocabulary as apply: a proposal pinned to a
            # stale read must fail loudly, never anchor against text the
            # proposer has not seen.
            raise CapabilityError(
                "editor.patch_revision_conflict",
                "Das Dokument wurde zwischenzeitlich geaendert "
                f"(aktuelle Revision {exc.current_revision}). Lies das "
                "Dokument erneut, bevor du Aenderungen vorschlaegst.",
                http_status=409,
            ) from exc
        return PatchProposeOutput(
            patch_id=patch.patch_id,
            document_id=patch.document_id,
            status=patch.status,
            edit_count=len(patch.edits),
            revision_before=patch.revision_before,
        )

    async def _apply(
        payload: PatchApplyInput, context: CapabilityContext
    ) -> PatchApplyOutput:
        try:
            patch = await service.apply(
                payload.patch_id,
                expected_revision=payload.expected_revision,
                visible_to=context.visible_to,
                principal=context.principal,
            )
        except PatchNotFound as exc:
            raise CapabilityError(
                "editor.patch_not_found",
                "Patch nicht gefunden.",
                http_status=404,
            ) from exc
        except PatchRevisionConflict as exc:
            raise CapabilityError(
                "editor.patch_revision_conflict",
                "Das Dokument wurde zwischenzeitlich geaendert "
                f"(aktuelle Revision {exc.current_revision}).",
                http_status=409,
            ) from exc
        except PatchAlreadyDecided as exc:
            raise CapabilityError(
                "editor.patch_already_decided",
                "Der Patch wurde bereits anders entschieden.",
                http_status=409,
            ) from exc
        return PatchApplyOutput(
            document_id=patch.document_id,
            revision=patch.applied_revision or 0,
            applied_edit_ids=list(patch.applied_edit_ids or ()),
        )

    return [
        CapabilityDefinition(
            id="editor.patch.propose",
            summary="Propose anchored edits against one editor document.",
            input_model=PatchProposeInput,
            output_model=PatchProposeOutput,
            effect=Effect.WRITE,
            idempotent=False,
            handler=_propose,
        ),
        CapabilityDefinition(
            id="editor.patch.apply",
            summary="Apply a pending editor patch at an expected document revision.",
            input_model=PatchApplyInput,
            output_model=PatchApplyOutput,
            effect=Effect.WRITE,
            idempotent=True,
            handler=_apply,
        ),
    ]
