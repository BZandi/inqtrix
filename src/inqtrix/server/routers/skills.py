"""Skill CRUD (``/v1/skills*``, plan M3 `3.1`).

The server half of the skill library: list returns the caller's
visible set (owned plus shared-in plus ownerless) with the additive
``access`` annotation, writes follow the shared owned-resource rule
(edit grant for updates, owner-only deletion), and a deleted skill's
shares are revoked in the same breath — the prompt-template router
shape verbatim, over the skill service.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request, Response

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.content.skill_markdown import (
    SkillMarkdownError,
    skill_from_markdown,
    skill_to_markdown,
)
from inqtrix.content.skills import (
    SkillConflict,
    SkillNotFound,
    SkillRecord,
)
from inqtrix.runs.shared import access_annotation
from inqtrix.server.routers import build_shared_grants_dependency
from inqtrix.services.request_parsing import error_response
from inqtrix.services.skill_service import SkillValidationError

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")

_SCOPED_KINDS = frozenset({"oidc_session", "pat"})

_IMPORT_MAX_CHARS = 512_000
"""Hard size cap of one imported SKILL.md (belt to the no-alias YAML
loader — parse cost stays linear AND bounded)."""


def skill_payload(
    record: SkillRecord,
    *,
    access: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The wire shape of one skill (list + detail + write responses)."""
    return {
        "id": record.id,
        "label": record.label,
        "title": record.title,
        "description": record.description,
        "when_to_use": record.when_to_use,
        "instructions_markdown": record.instructions_markdown,
        "clarification_points": [
            dict(point) for point in record.clarification_points
        ],
        "deliverable": record.deliverable,
        "allowed_tools": list(record.allowed_tools),
        "requires_plan": record.requires_plan,
        "invocation": record.invocation,
        "argument_hint": record.argument_hint,
        "model_tier": record.model_tier,
        "effort": record.effort,
        "include_in_autocomplete": record.include_in_autocomplete,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        **({"access": access} if access is not None else {}),
    }


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the skill routes against the container.

    Raises:
        RuntimeError: When called without a wired skill service —
            registration is a composition decision, not a runtime
            fallback.
    """
    service = container.skill_service
    if service is None:
        raise RuntimeError(
            "build_router(skills) requires a wired skill service."
        )
    router = APIRouter()
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency
    workspace_admin = container.workspace_admin
    share_service = container.share_service
    shared_skills_dep = build_shared_grants_dependency(
        share_service, principal_dep, resource_type="skill_template"
    )

    async def _parsed_body(req: Request):
        try:
            body = await req.json()
        except Exception:
            return None
        return body if isinstance(body, dict) else None

    @router.get("/v1/skills")
    async def list_skills(
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_skills_dep),
    ):
        """The caller's visible skills, newest first."""
        records = await service.list_visible(
            tenant_id=principal.tenant_id,
            visible_to=visible_to,
            also_visible=also_visible,
        )
        return {
            "object": "list",
            "data": [
                skill_payload(record, access=access_annotation(shared))
                for record, shared in records
            ],
        }

    @router.post("/v1/skills", status_code=201)
    async def create_skill(
        req: Request,
        principal: Principal = Depends(principal_dep),
    ):
        """Create one skill; scoped principals own what they create."""
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
        except SkillValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        return skill_payload(record)

    @router.get("/v1/skills/{skill_id}/markdown")
    async def export_skill_markdown(
        skill_id: str,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_skills_dep),
    ):
        """One visible skill as its SKILL.md document (text/markdown)."""
        try:
            record, _ = await service.get_visible(
                skill_id,
                tenant_id=principal.tenant_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except SkillNotFound:
            return error_response(404, "Skill nicht gefunden", "not_found")
        return Response(
            content=skill_to_markdown(record),
            media_type="text/markdown; charset=utf-8",
        )

    @router.post("/v1/skills/import", status_code=201)
    async def import_skill_markdown(
        req: Request,
        principal: Principal = Depends(principal_dep),
    ):
        """Create one skill from a SKILL.md document.

        Body: ``{"markdown": "<SKILL.md text>"}``. The parser only maps
        the file shape; the service validator stays the single gate for
        every policy rule (placeholder coupling, enums, point limits).
        """
        body = await _parsed_body(req)
        if body is None or not isinstance(body.get("markdown"), str):
            return error_response(
                400,
                "Feld 'markdown' (String) fehlt",
                "invalid_request_error",
            )
        if len(body["markdown"]) > _IMPORT_MAX_CHARS:
            return error_response(
                400,
                "SKILL.md ist zu gross (max. "
                f"{_IMPORT_MAX_CHARS} Zeichen)",
                "invalid_request_error",
            )
        try:
            payload = skill_from_markdown(body["markdown"])
        except SkillMarkdownError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        try:
            record = await service.create(
                payload,
                tenant_id=principal.tenant_id,
                owner_sub=(
                    principal.sub
                    if principal.kind in _SCOPED_KINDS
                    else None
                ),
            )
        except SkillValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        return skill_payload(record)

    @router.put("/v1/skills/{skill_id}")
    async def update_skill(
        skill_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_skills_dep),
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
        # `bool` is an `int` subclass — reject it explicitly so the
        # contract ("a number") holds (the template router's rule).
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
                skill_id,
                body,
                tenant_id=principal.tenant_id,
                expected_updated_at=(
                    float(raw_expected) if raw_expected is not None else None
                ),
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except SkillValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except SkillConflict:
            return error_response(
                409,
                "Der Skill wurde zwischenzeitlich geaendert",
                "conflict",
            )
        except SkillNotFound:
            return error_response(404, "Skill nicht gefunden", "not_found")
        return skill_payload(record)

    @router.delete("/v1/skills/{skill_id}", status_code=204)
    async def delete_skill(
        skill_id: str,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_skills_dep),
    ):
        """Delete one skill (owner-only) and revoke its shares."""
        try:
            await service.delete(
                skill_id,
                tenant_id=principal.tenant_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except SkillNotFound:
            return error_response(404, "Skill nicht gefunden", "not_found")
        if workspace_admin is not None and share_service is not None:
            revoked = await workspace_admin.revoke_shares_for_resource(
                tenant_id=principal.tenant_id,
                resource_type="skill_template",
                resource_id=skill_id,
                revoked_by_sub=principal.sub,
            )
            if revoked:
                log.info(
                    "Skill %s geloescht; %d Freigaben entzogen",
                    skill_id,
                    revoked,
                )

    return router
