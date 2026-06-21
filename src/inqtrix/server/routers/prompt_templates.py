"""Prompt-template CRUD (``/v1/prompt-templates*``).

The server half of the prompt library: list returns the caller's
visible set (owned plus shared-in plus ownerless) with the additive
``access`` annotation, writes follow the shared owned-resource rule
(edit grant for updates, owner-only deletion), and a deleted
template's shares are revoked in the same breath.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.content.prompt_templates import (
    PromptTemplateConflict,
    PromptTemplateNotFound,
    PromptTemplateRecord,
)
from inqtrix.runs.shared import access_annotation
from inqtrix.server.routers import build_shared_grants_dependency
from inqtrix.services.prompt_template_service import (
    PromptTemplateValidationError,
)
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")

_SCOPED_KINDS = frozenset({"oidc_session", "pat"})


def _template_payload(
    record: PromptTemplateRecord,
    *,
    access: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "label": record.label,
        "category": record.category,
        "content_markdown": record.content_markdown,
        "visibility": dict(record.visibility),
        "include_in_autocomplete": record.include_in_autocomplete,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        **({"access": access} if access is not None else {}),
    }


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the prompt-template routes against the container.

    Raises:
        RuntimeError: When called without a wired template service —
            registration is a composition decision, not a runtime
            fallback.
    """
    service = container.prompt_template_service
    if service is None:
        raise RuntimeError(
            "build_router(prompt_templates) requires a wired template "
            "service."
        )
    router = APIRouter()
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency
    workspace_admin = container.workspace_admin
    share_service = container.share_service
    shared_templates_dep = build_shared_grants_dependency(
        share_service, principal_dep, resource_type="prompt_template"
    )

    async def _parsed_body(req: Request):
        try:
            body = await req.json()
        except Exception:
            return None
        return body if isinstance(body, dict) else None

    @router.get("/v1/prompt-templates")
    async def list_templates(
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_templates_dep),
    ):
        """The caller's visible templates, newest first."""
        records = await service.list_visible(
            tenant_id=principal.tenant_id,
            visible_to=visible_to,
            also_visible=also_visible,
        )
        return {
            "object": "list",
            "data": [
                _template_payload(record, access=access_annotation(shared))
                for record, shared in records
            ],
        }

    @router.post("/v1/prompt-templates", status_code=201)
    async def create_template(
        req: Request,
        principal: Principal = Depends(principal_dep),
    ):
        """Create one template; scoped principals own what they create."""
        body = await _parsed_body(req)
        if body is None:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        try:
            record = await service.create(
                body,
                tenant_id=principal.tenant_id,
                owner_sub=(
                    principal.sub
                    if principal.kind in _SCOPED_KINDS
                    else None
                ),
            )
        except PromptTemplateValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        return _template_payload(record)

    @router.put("/v1/prompt-templates/{template_id}")
    async def update_template(
        template_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_templates_dep),
    ):
        """Replace the writable fields (edit grant required).

        ``expected_updated_at`` (unix seconds) is the optional
        optimistic-concurrency precondition; a stale value yields 409.
        """
        body = await _parsed_body(req)
        if body is None:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        raw_expected = body.get("expected_updated_at")
        # `bool` is an `int` subclass, so `True`/`False` would slip past
        # a plain isinstance(..., (int, float)) and coerce to 1.0/0.0 —
        # reject it explicitly so the contract ("a number") holds.
        if raw_expected is not None and (
            isinstance(raw_expected, bool)
            or not isinstance(raw_expected, (int, float))
        ):
            return error_response(
                400,
                "Feld 'expected_updated_at' muss eine Zahl sein",
                "invalid_request_error",
            )
        try:
            record = await service.update(
                template_id,
                body,
                tenant_id=principal.tenant_id,
                expected_updated_at=(
                    float(raw_expected) if raw_expected is not None else None
                ),
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except PromptTemplateValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except PromptTemplateConflict:
            return error_response(
                409,
                "Die Vorlage wurde zwischenzeitlich geaendert",
                "conflict",
            )
        except PromptTemplateNotFound:
            return error_response(
                404, "Vorlage nicht gefunden", "not_found"
            )
        return _template_payload(record)

    @router.delete("/v1/prompt-templates/{template_id}", status_code=204)
    async def delete_template(
        template_id: str,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_templates_dep),
    ):
        """Delete one template (owner-only) and revoke its shares."""
        try:
            await service.delete(
                template_id,
                tenant_id=principal.tenant_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except PromptTemplateNotFound:
            return error_response(
                404, "Vorlage nicht gefunden", "not_found"
            )
        if workspace_admin is not None and share_service is not None:
            revoked = await workspace_admin.revoke_shares_for_resource(
                tenant_id=principal.tenant_id,
                resource_type="prompt_template",
                resource_id=template_id,
                revoked_by_sub=principal.sub,
            )
            if revoked:
                log.info(
                    "Vorlage %s geloescht; %d Freigaben entzogen",
                    template_id,
                    revoked,
                )

    return router
