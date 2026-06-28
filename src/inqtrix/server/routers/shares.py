"""Resource-share routes (``/v1/shares*``) — mounted only in oidc mode.

Flat by design: ONE router plus the ShareService's owner-resolver
registry serve every shareable resource kind, matching the
polymorphic ``resource_shares`` table; nested per-resource routes
would touch every resource router for each new kind.

Conventions: denials hide behind 404 (membership and existence are
not disclosed), validation errors carry German messages, and the
listing payloads join display names from the users mirror so the
share dialog needs no second request.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from inqtrix.auth.permissions import SharePermission
from inqtrix.auth.shares import (
    ShareNotAllowed,
    ShareValidationError,
)
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")

_SCOPED_KINDS = frozenset({"oidc_session", "pat"})


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the share routes against the container."""
    router = APIRouter()
    principal_dep = container.principal_dependency
    service = container.share_service
    users = getattr(container.auth_provider, "users", None)

    async def _scoped_principal(request: Request):
        principal = await principal_dep(request)
        if principal.kind not in _SCOPED_KINDS:
            return None, error_response(404, "Nicht gefunden", "not_found")
        return principal, None

    async def _enrich(records, tenant_id: str) -> list[dict]:
        """Join display names from the users mirror in one batch."""
        profiles = {}
        subjects = tuple(
            record.subject_id
            for record in records
            if record.subject_type == "user"
        )
        if users is not None and subjects:
            profiles = await users.profiles_for_subjects(
                tenant_id=tenant_id, subs=subjects
            )
        payloads = []
        for record in records:
            profile = profiles.get(record.subject_id)
            payloads.append({
                "id": record.id,
                "subject_type": record.subject_type,
                "subject_id": record.subject_id,
                "resource_type": record.resource_type,
                "resource_id": record.resource_id,
                "permission": record.permission.value,
                "granted_by_sub": record.granted_by_sub,
                "created_at": record.created_at,
                "display_name": (
                    profile.display_name if profile is not None else None
                ),
                "email": profile.email if profile is not None else None,
            })
        return payloads

    @router.get("/v1/shares")
    async def list_shares(
        request: Request, resource_type: str = "", resource_id: str = ""
    ):
        """Active shares on one resource (requires view access)."""
        principal, error = await _scoped_principal(request)
        if error is not None:
            return error
        if not resource_type or not resource_id:
            return error_response(
                400,
                "Parameter 'resource_type' und 'resource_id' sind "
                "erforderlich",
                "invalid_request_error",
            )
        try:
            records = await service.list_for_resource(
                principal,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        except ShareValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except ShareNotAllowed:
            return error_response(
                404, "Ressource nicht gefunden", "not_found"
            )
        return {
            "object": "list",
            "data": await _enrich(records, principal.tenant_id),
        }

    @router.post("/v1/shares", status_code=201)
    async def create_shares(request: Request):
        """Grant shares to one or more users (owner/manage only)."""
        principal, error = await _scoped_principal(request)
        if error is not None:
            return error
        try:
            body = await request.json()
        except Exception:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        if not isinstance(body, dict):
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        resource_type = str(body.get("resource_type", "")).strip()
        resource_id = str(body.get("resource_id", "")).strip()
        raw_invitees = body.get("invitees")
        if not resource_type or not resource_id:
            return error_response(
                400,
                "Felder 'resource_type' und 'resource_id' sind "
                "erforderlich",
                "invalid_request_error",
            )
        if not isinstance(raw_invitees, list) or not raw_invitees:
            return error_response(
                400,
                "Feld 'invitees' muss eine nicht-leere Liste sein",
                "invalid_request_error",
            )
        invitees: list[tuple[str, SharePermission]] = []
        for item in raw_invitees:
            if not isinstance(item, dict):
                return error_response(
                    400,
                    "Jeder Eintrag in 'invitees' muss ein Objekt sein",
                    "invalid_request_error",
                )
            subject = str(item.get("subject_id", "")).strip()
            raw_permission = str(item.get("permission", "")).strip()
            try:
                permission = SharePermission(raw_permission)
            except ValueError:
                return error_response(
                    400,
                    "Feld 'permission' muss 'view' oder 'edit' sein",
                    "invalid_request_error",
                )
            if not subject:
                return error_response(
                    400,
                    "Feld 'subject_id' ist erforderlich",
                    "invalid_request_error",
                )
            invitees.append((subject, permission))
        try:
            created = await service.grant(
                principal,
                resource_type=resource_type,
                resource_id=resource_id,
                invitees=invitees,
            )
        except ShareValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except ShareNotAllowed:
            return error_response(
                404, "Ressource nicht gefunden", "not_found"
            )
        return {
            "object": "list",
            "data": await _enrich(created, principal.tenant_id),
        }

    @router.post("/v1/shares/{share_id}/accept")
    async def accept_share(share_id: str, request: Request):
        """Accept one pending share addressed to the caller.

        Consent is the recipient's alone: until this lands the share grants
        nothing. An unknown id, a foreign recipient, an already-accepted or a
        revoked share all return the same 404 (denial hidden behind absence).
        """
        principal, error = await _scoped_principal(request)
        if error is not None:
            return error
        if not await service.accept(principal, share_id=share_id):
            return error_response(
                404, "Freigabe nicht gefunden", "not_found"
            )
        return {"accepted": True}

    @router.delete("/v1/shares/{share_id}")
    async def remove_share(share_id: str, request: Request):
        """Remove one share — two callers, one verb.

        The owner/manager revokes it, OR the recipient drops their own (declines
        a pending invitation / leaves an accepted share). The paths are mutually
        exclusive: only the owner passes the manage gate, only the recipient
        matches ``subject_id``. Either way the response is ``{"revoked": true}``;
        a caller who is neither gets the surface's indistinct 404.
        """
        principal, error = await _scoped_principal(request)
        if error is not None:
            return error
        if await service.revoke(principal, share_id=share_id):
            return {"revoked": True}
        if await service.recipient_drop(principal, share_id=share_id):
            return {"revoked": True}
        return error_response(404, "Freigabe nicht gefunden", "not_found")

    @router.get("/v1/shares/shared-with-me")
    async def shared_with_me(request: Request, resource_type: str = ""):
        """Shared-in resources of one kind for the caller."""
        principal, error = await _scoped_principal(request)
        if error is not None:
            return error
        if not resource_type:
            return error_response(
                400,
                "Parameter 'resource_type' ist erforderlich",
                "invalid_request_error",
            )
        shared = await service.shared_with_me(
            principal, resource_type=resource_type
        )
        # One batch join for the "Geteilt von <Name>" badge — same
        # enrichment idiom as the share listings above.
        grantors = {}
        grantor_subs = tuple(
            {record.granted_by_sub for record in shared.values()}
        )
        if users is not None and grantor_subs:
            grantors = await users.profiles_for_subjects(
                tenant_id=principal.tenant_id, subs=grantor_subs
            )
        return {
            "object": "list",
            "data": [
                {
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "permission": record.permission.value,
                    "granted_by_sub": record.granted_by_sub,
                    "granted_by_display_name": (
                        grantors[record.granted_by_sub].display_name
                        if record.granted_by_sub in grantors
                        else None
                    ),
                    "created_at": record.created_at,
                }
                for resource_id, record in sorted(shared.items())
            ],
        }

    @router.get("/v1/shares/outgoing")
    async def outgoing(request: Request, resource_type: str = ""):
        """Active-share counts for the badge layer (owned ids only).

        The service drops every id the caller does not own before
        counting — counts would otherwise be an existence oracle for
        foreign resources.
        """
        principal, error = await _scoped_principal(request)
        if error is not None:
            return error
        if not resource_type:
            return error_response(
                400,
                "Parameter 'resource_type' ist erforderlich",
                "invalid_request_error",
            )
        ids = request.query_params.getlist("resource_id")
        try:
            counts = await service.outgoing_counts(
                principal, resource_type=resource_type, resource_ids=ids
            )
        except ShareValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        return {"object": "map", "data": dict(counts)}

    @router.get("/v1/shares/inbox")
    async def inbox(request: Request):
        """The caller's incoming shares, split into pending and accepted.

        One title-enriched listing across every shareable kind, powering the
        settings "Eingegangen" (pending consent) and "Mit mir geteilt"
        (accepted) sections. Grantor display names are joined here in one batch
        (the same idiom as shared-with-me); the service supplies the titles.
        """
        principal, error = await _scoped_principal(request)
        if error is not None:
            return error
        items = await service.inbox(principal)
        grantors = {}
        grantor_subs = tuple({item.granted_by_sub for item in items})
        if users is not None and grantor_subs:
            grantors = await users.profiles_for_subjects(
                tenant_id=principal.tenant_id, subs=grantor_subs
            )

        def _payload(item) -> dict:
            profile = grantors.get(item.granted_by_sub)
            return {
                "id": item.share_id,
                "resource_type": item.resource_type,
                "resource_id": item.resource_id,
                "resource_title": item.resource_title,
                "permission": item.permission.value,
                "granted_by_sub": item.granted_by_sub,
                "granted_by_display_name": (
                    profile.display_name if profile is not None else None
                ),
                "created_at": item.created_at,
                "accepted_at": item.accepted_at,
            }

        return {
            "object": "map",
            "data": {
                "pending": [
                    _payload(item)
                    for item in items
                    if item.accepted_at is None
                ],
                "accepted": [
                    _payload(item)
                    for item in items
                    if item.accepted_at is not None
                ],
            },
        }

    @router.get("/v1/shares/mine")
    async def mine(request: Request):
        """The caller's outgoing shares, grouped per resource (the settings
        "Von mir geteilt" section). Each row carries the title and the active /
        pending recipient counts; per-recipient management reuses the existing
        share dialog against this resource id."""
        principal, error = await _scoped_principal(request)
        if error is not None:
            return error
        items = await service.outgoing(principal)
        return {
            "object": "list",
            "data": [
                {
                    "resource_type": item.resource_type,
                    "resource_id": item.resource_id,
                    "resource_title": item.resource_title,
                    "share_count": item.share_count,
                    "pending_count": item.pending_count,
                }
                for item in items
            ],
        }

    return router
