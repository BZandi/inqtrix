"""Admin workspace + membership management (``/v1/admin/workspaces*``).

The instance-admin surface for the platform's collaboration spaces: an
instance admin creates workspaces, assigns/positions users into them, and
manages member roles. This is the org-administration axis (Slack-Grid /
Notion-teamspace pattern) — gated on ``instance_role == "admin"`` via the
shared :func:`require_instance_admin` guard, never on workspace ownership.
``WorkspaceRole`` (viewer..owner) stays the in-workspace collaboration role.

Mounted only for a cookie-session provider that carries a user mirror AND a
workspace-admin (membership) store; in every other configuration the routes
are plain 404s. Denials hide behind 404 (the permission layer's not-403
convention); every mutation is audited.

One UX invariant: a workspace must not be left without an OWNER by demoting
or removing its last one (409 ``last_owner``) — a recoverable orphan an admin
could otherwise create by accident. The guard is read-then-write at the
router (not atomic): an orphaned workspace stays fully manageable by instance
admins, so the mild race is acceptable, unlike the deployment-locking
last-instance-admin guard which is atomic in the store.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from inqtrix.auth.permissions import AuditEntry, WorkspaceRole
from inqtrix.server.routers._admin_guard import require_instance_admin
from inqtrix.services.request_parsing import error_response
from inqtrix.services.workspace_administration import (
    WorkspaceNameError,
    create_workspace_for_admin,
    normalize_workspace_name,
)

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")


def _parse_role(value: object) -> WorkspaceRole | None:
    """Parse a workspace-role string, or ``None`` when it is not valid."""
    try:
        return WorkspaceRole(str(value))
    except ValueError:
        return None


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the admin workspace/membership routes against the container.

    Raises:
        RuntimeError: When called without a wired workspace-admin store —
            registration is a composition decision, not a runtime fallback
            (mirrors the knowledge/files/quota routers).
    """
    workspace_admin = container.workspace_admin
    if workspace_admin is None:
        raise RuntimeError(
            "build_router(admin_workspaces) requires a wired workspace_admin; "
            "register the router only for a membership-backed cookie-session mode."
        )
    router = APIRouter()
    provider = container.auth_provider
    users = getattr(provider, "users", None)

    async def _admin(request: Request):
        """Resolve the caller as an instance admin, or yield the error."""
        resolved, error = await require_instance_admin(provider, request)
        if error is not None:
            return None, error
        principal, _session, _mirror = resolved
        return principal, None

    async def _audit(principal, action: str, resource_id: str, detail: dict) -> None:
        await workspace_admin.record(
            AuditEntry(
                tenant_id=principal.tenant_id,
                actor_sub=principal.sub,
                action=action,
                resource_type="workspace",
                resource_id=resource_id,
                detail=detail,
            )
        )

    async def _members_or_404(tenant_id: str, workspace_id: str):
        """``list_members`` with the absent-workspace case mapped to a 404."""
        members = await workspace_admin.list_members(
            tenant_id=tenant_id, workspace_id=workspace_id
        )
        if members is None:
            return None, error_response(
                404, "Workspace nicht gefunden", "not_found"
            )
        return members, None

    def _last_owner_blocked(
        members: tuple[tuple[str, WorkspaceRole], ...],
        *,
        target_sub: str,
        keeps_owner: bool,
    ) -> bool:
        """Whether the op would strip a workspace's only OWNER.

        *keeps_owner* is ``True`` when the op leaves *target_sub* an owner
        (a no-op for the guard); ``False`` for a removal or a demotion.
        """
        if keeps_owner:
            return False
        owners = [sub for sub, role in members if role is WorkspaceRole.OWNER]
        return target_sub in owners and len(owners) == 1

    # ----------------------------------------------------------------- #
    # Workspaces
    # ----------------------------------------------------------------- #

    @router.get("/v1/admin/workspaces")
    async def list_workspaces(request: Request):
        """Every workspace in the tenant with its member count."""
        principal, error = await _admin(request)
        if error is not None:
            return error
        rows = await workspace_admin.list_all_workspaces(
            tenant_id=principal.tenant_id
        )
        return {
            "object": "list",
            "data": [
                {
                    "workspace_id": workspace_id,
                    "name": name,
                    "created_by_sub": created_by_sub,
                    "member_count": member_count,
                }
                for workspace_id, name, created_by_sub, member_count in rows
            ],
        }

    @router.post("/v1/admin/workspaces", status_code=201)
    async def create_workspace(request: Request):
        """Create one workspace; the creating admin becomes its OWNER."""
        principal, error = await _admin(request)
        if error is not None:
            return error
        try:
            body = await request.json()
        except Exception:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        try:
            workspace_id, created_name = await create_workspace_for_admin(
                workspace_admin,
                principal=principal,
                name=(body or {}).get("name", ""),
            )
        except WorkspaceNameError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        return {"workspace_id": workspace_id, "name": created_name}

    @router.patch("/v1/admin/workspaces/{workspace_id}")
    async def rename_workspace(workspace_id: str, request: Request):
        """Rename one workspace."""
        principal, error = await _admin(request)
        if error is not None:
            return error
        try:
            body = await request.json()
        except Exception:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        try:
            name = normalize_workspace_name((body or {}).get("name", ""))
        except WorkspaceNameError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        renamed = await workspace_admin.rename_workspace(
            tenant_id=principal.tenant_id, workspace_id=workspace_id, name=name
        )
        if not renamed:
            return error_response(
                404, "Workspace nicht gefunden", "not_found"
            )
        await _audit(
            principal, "workspace.renamed", workspace_id, {"name": name}
        )
        return {"workspace_id": workspace_id, "name": name}

    @router.delete("/v1/admin/workspaces/{workspace_id}", status_code=204)
    async def delete_workspace(workspace_id: str, request: Request):
        """Delete one workspace and cascade its memberships."""
        principal, error = await _admin(request)
        if error is not None:
            return error
        deleted = await workspace_admin.delete_workspace(
            tenant_id=principal.tenant_id, workspace_id=workspace_id
        )
        if not deleted:
            return error_response(
                404, "Workspace nicht gefunden", "not_found"
            )
        await _audit(principal, "workspace.deleted", workspace_id, {})

    # ----------------------------------------------------------------- #
    # Members
    # ----------------------------------------------------------------- #

    @router.get("/v1/admin/workspaces/{workspace_id}/members")
    async def list_members(workspace_id: str, request: Request):
        """Members of one workspace, enriched with display name + email."""
        principal, error = await _admin(request)
        if error is not None:
            return error
        members, error = await _members_or_404(principal.tenant_id, workspace_id)
        if error is not None:
            return error
        profiles = {}
        if users is not None and members:
            profiles = await users.profiles_for_subjects(
                tenant_id=principal.tenant_id,
                subs=tuple(sub for sub, _role in members),
            )
        data = []
        for sub, role in members:
            profile = profiles.get(sub)
            data.append(
                {
                    "sub": sub,
                    "role": role.value,
                    "display_name": (
                        profile.display_name if profile is not None else None
                    ),
                    "email": profile.email if profile is not None else None,
                }
            )
        return {"object": "list", "data": data}

    @router.post(
        "/v1/admin/workspaces/{workspace_id}/members", status_code=201
    )
    async def add_member(workspace_id: str, request: Request):
        """Assign a user to the workspace at a role (or update their role)."""
        principal, error = await _admin(request)
        if error is not None:
            return error
        try:
            body = await request.json()
        except Exception:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        sub = str((body or {}).get("sub", "")).strip()
        if not sub:
            return error_response(
                400, "Feld 'sub' ist erforderlich", "invalid_request_error"
            )
        role = _parse_role((body or {}).get("role"))
        if role is None:
            valid = ", ".join(member.value for member in WorkspaceRole)
            return error_response(
                400,
                f"Feld 'role' muss eines von {valid} sein",
                "invalid_request_error",
            )
        # The target must be a known user (no phantom memberships).
        if users is not None and not await users.has_subject(
            tenant_id=principal.tenant_id, sub=sub
        ):
            return error_response(404, "Benutzer nicht gefunden", "not_found")
        members, error = await _members_or_404(principal.tenant_id, workspace_id)
        if error is not None:
            return error
        if _last_owner_blocked(
            members, target_sub=sub, keeps_owner=role is WorkspaceRole.OWNER
        ):
            return error_response(
                409,
                "Der letzte OWNER kann nicht herabgestuft werden",
                "last_owner",
            )
        await workspace_admin.assign_member(
            tenant_id=principal.tenant_id,
            workspace_id=workspace_id,
            sub=sub,
            role=role,
        )
        await _audit(
            principal,
            "workspace.member_added",
            f"{workspace_id}:{sub}",
            {"role": role.value},
        )
        return {"sub": sub, "role": role.value}

    @router.patch("/v1/admin/workspaces/{workspace_id}/members/{sub}")
    async def set_member_role(workspace_id: str, sub: str, request: Request):
        """Change an existing member's role."""
        principal, error = await _admin(request)
        if error is not None:
            return error
        try:
            body = await request.json()
        except Exception:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        role = _parse_role((body or {}).get("role"))
        if role is None:
            valid = ", ".join(member.value for member in WorkspaceRole)
            return error_response(
                400,
                f"Feld 'role' muss eines von {valid} sein",
                "invalid_request_error",
            )
        members, error = await _members_or_404(principal.tenant_id, workspace_id)
        if error is not None:
            return error
        if sub not in {member_sub for member_sub, _role in members}:
            return error_response(
                404, "Mitglied nicht gefunden", "not_found"
            )
        if _last_owner_blocked(
            members, target_sub=sub, keeps_owner=role is WorkspaceRole.OWNER
        ):
            return error_response(
                409,
                "Der letzte OWNER kann nicht herabgestuft werden",
                "last_owner",
            )
        await workspace_admin.assign_member(
            tenant_id=principal.tenant_id,
            workspace_id=workspace_id,
            sub=sub,
            role=role,
        )
        await _audit(
            principal,
            "workspace.member_role_set",
            f"{workspace_id}:{sub}",
            {"role": role.value},
        )
        return {"sub": sub, "role": role.value}

    @router.delete(
        "/v1/admin/workspaces/{workspace_id}/members/{sub}", status_code=204
    )
    async def remove_member(workspace_id: str, sub: str, request: Request):
        """Remove a member from the workspace."""
        principal, error = await _admin(request)
        if error is not None:
            return error
        members, error = await _members_or_404(principal.tenant_id, workspace_id)
        if error is not None:
            return error
        if sub not in {member_sub for member_sub, _role in members}:
            return error_response(
                404, "Mitglied nicht gefunden", "not_found"
            )
        if _last_owner_blocked(members, target_sub=sub, keeps_owner=False):
            return error_response(
                409,
                "Der letzte OWNER kann nicht entfernt werden",
                "last_owner",
            )
        await workspace_admin.remove_member(
            tenant_id=principal.tenant_id, workspace_id=workspace_id, sub=sub
        )
        await _audit(
            principal, "workspace.member_removed", f"{workspace_id}:{sub}", {}
        )

    return router
