"""Quota endpoints: the caller's own meter + the instance-admin surface.

Two audiences, one router (mounted only when ``quota_service`` is wired,
i.e. a cookie-session mode with quotas on):

* ``GET /v1/quota/usage`` — the authenticated caller's own usage per
  dimension. Feeds the composer meters; available to any scoped
  principal.
* ``/v1/admin/quota*`` — the editable admin surface, gated to the instance
  admin (``instance_role == "admin"``, 404-not-403 house convention): an
  overview of metered subjects with their usage and effective limits, plus
  setting the tenant default / per-user overrides and resetting flow usage.

Quota subjects are ``(tenant_id, sub)`` and quota administration is
tenant-wide, so it lives on the instance-admin axis — never on workspace
ownership (a workspace owner administers collaboration, not the
deployment's quotas). Every admin mutation is audited (``quota.override`` /
``quota.override_cleared`` / ``quota.reset``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from inqtrix.auth.permissions import AuditEntry
from inqtrix.quota.models import QuotaDimension, period_end
from inqtrix.server.routers._admin_guard import require_instance_admin
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")


def _parse_dimension(value: object) -> QuotaDimension | None:
    """Parse a dimension string, or ``None`` when it is not valid."""
    try:
        return QuotaDimension(str(value))
    except ValueError:
        return None


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the quota self + admin routes against the container.

    Raises:
        RuntimeError: When called without a wired quota service —
            registration is a composition decision, not a runtime
            fallback (mirrors the knowledge/files routers).
    """
    quota_service = container.quota_service
    if quota_service is None:
        raise RuntimeError(
            "build_router(quota) requires a wired quota service; register "
            "the router only when quotas are enabled."
        )
    router = APIRouter()
    principal_dep = container.principal_dependency
    provider = container.auth_provider
    workspace_admin = container.workspace_admin
    users = getattr(container.auth_provider, "users", None)

    async def _admin(request: Request):
        """Resolve the caller as an instance admin (the quota-admin axis).

        Quota administration is tenant-wide platform administration, so it
        is gated on ``instance_role == "admin"`` (the users-mirror single
        source of truth) via the shared :func:`require_instance_admin`
        guard — never on workspace ownership. Returns ``(principal, None)``
        for an admin or ``(None, error_response)`` otherwise.
        """
        resolved, error = await require_instance_admin(provider, request)
        if error is not None:
            return None, error
        principal, _session, _mirror = resolved
        return principal, None

    @router.get("/v1/quota/usage")
    async def my_usage(request: Request):
        """The caller's own usage per dimension (the composer meter).

        Empty for an unscoped principal (no metered subject) — the UI
        only shows the meter for authenticated users anyway.
        """
        principal = await principal_dep(request)
        subject = quota_service.subject_for(principal)
        if subject is None:
            return {"object": "list", "data": []}
        rows = await quota_service.usage_for(subject)
        return {
            "object": "list",
            "data": [
                {
                    "dimension": row.dimension.value,
                    "used": row.used,
                    "limit": row.limit,
                    "remaining": row.remaining,
                    "period_start": row.period_start,
                    "reset_at": period_end(row.period_start),
                }
                for row in rows
            ],
        }

    @router.get("/v1/admin/quota")
    async def admin_overview(request: Request):
        """Admin overview: metered subjects, usage, limits, ceilings."""
        principal, error = await _admin(request)
        if error is not None:
            return error
        snapshot = await quota_service.admin_snapshot(principal.tenant_id)
        if users is not None and snapshot["subjects"]:
            profiles = await users.profiles_for_subjects(
                tenant_id=principal.tenant_id,
                subs=tuple(row["sub"] for row in snapshot["subjects"]),
            )
            for row in snapshot["subjects"]:
                profile = profiles.get(row["sub"])
                row["display_name"] = (
                    profile.display_name if profile is not None else None
                )
                row["email"] = profile.email if profile is not None else None
        return snapshot

    @router.put("/v1/admin/quota/limits")
    async def set_limit(request: Request):
        """Set the tenant default or a per-user override (instance admin only).

        Body: ``{"subject_id": "<sub>"|"__quota_default__", "dimension":
        "<dim>", "value": <int>=0>}``. ``value`` ``0`` is explicit
        unlimited; the operator ceiling still clamps at read.
        """
        principal, error = await _admin(request)
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
        subject_id = str(body.get("subject_id", "") or "")
        if not subject_id:
            return error_response(
                400, "Feld 'subject_id' ist erforderlich", "invalid_request_error"
            )
        dimension = _parse_dimension(body.get("dimension"))
        if dimension is None:
            return error_response(
                400,
                "Feld 'dimension' muss eines von "
                + ", ".join(d.value for d in QuotaDimension)
                + " sein",
                "invalid_request_error",
            )
        raw_value = body.get("value")
        if not isinstance(raw_value, int) or isinstance(raw_value, bool) or raw_value < 0:
            return error_response(
                400,
                "Feld 'value' muss eine nicht-negative Ganzzahl sein (0 = unbegrenzt)",
                "invalid_request_error",
            )
        await quota_service.set_limit_for(
            tenant_id=principal.tenant_id,
            subject_sub=subject_id,
            dimension=dimension,
            value=raw_value,
            set_by_sub=principal.sub,
        )
        await _audit(
            principal,
            "quota.override",
            subject_id,
            dimension,
            {"value": str(raw_value)},
        )
        return {"subject_id": subject_id, "dimension": dimension.value, "value": raw_value}

    @router.delete("/v1/admin/quota/limits", status_code=204)
    async def clear_limit(
        request: Request,
        subject_id: str = "",
        dimension: str = "",
    ):
        """Drop a limit so it falls back to the next layer (instance admin only).

        ``subject_id`` and ``dimension`` are query parameters (not path
        segments): an OIDC ``sub`` may be a URI containing slashes, which
        a path segment cannot represent — and this keeps the identifying
        fields off the path, consistent with the body-carried subject on
        PUT/reset.
        """
        principal, error = await _admin(request)
        if error is not None:
            return error
        if not subject_id:
            return error_response(
                400,
                "Query-Parameter 'subject_id' ist erforderlich",
                "invalid_request_error",
            )
        parsed = _parse_dimension(dimension)
        if parsed is None:
            return error_response(
                400, "Ungueltige Dimension", "invalid_request_error"
            )
        await quota_service.clear_limit_for(
            tenant_id=principal.tenant_id,
            subject_sub=subject_id,
            dimension=parsed,
        )
        await _audit(principal, "quota.override_cleared", subject_id, parsed, {})

    @router.post("/v1/admin/quota/reset")
    async def reset_usage(request: Request):
        """Zero one subject's current-window flow usage (instance admin only).

        Stock dimensions cannot be reset (freed by deletion, never
        reset) — the attempt is a 400, not a silent no-op.
        """
        principal, error = await _admin(request)
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
        subject_id = str(body.get("subject_id", "") or "")
        if not subject_id:
            return error_response(
                400, "Feld 'subject_id' ist erforderlich", "invalid_request_error"
            )
        dimension = _parse_dimension(body.get("dimension"))
        if dimension is None:
            return error_response(
                400, "Ungueltige Dimension", "invalid_request_error"
            )
        try:
            await quota_service.reset_for(
                tenant_id=principal.tenant_id,
                subject_sub=subject_id,
                dimension=dimension,
            )
        except ValueError as exc:
            # Stock dimension reset rejected by the store contract.
            return error_response(400, str(exc), "invalid_request_error")
        await _audit(principal, "quota.reset", subject_id, dimension, {})
        return {"subject_id": subject_id, "dimension": dimension.value}

    async def _audit(
        principal,
        action: str,
        subject_id: str,
        dimension: QuotaDimension,
        detail: dict[str, str],
    ) -> None:
        if workspace_admin is None:
            return
        await workspace_admin.record(
            AuditEntry(
                tenant_id=principal.tenant_id,
                actor_sub=principal.sub,
                action=action,
                resource_type="quota",
                resource_id=f"{subject_id}:{dimension.value}",
                detail=detail,
            )
        )

    return router
