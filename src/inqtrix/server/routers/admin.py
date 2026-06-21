"""Instance administration routes (``/v1/admin/*``).

User management for the cookie-session modes (local/oidc/ldap): list
users, change the instance role, enable/disable, and (local mode) create
accounts. Authorization is SESSION-ONLY and admin-gated — a PAT can never
administer users (a leaked token must not be able to promote itself).
Denial is hidden behind a 404 (the permission layer's not-403 convention).

Two invariants keep the deployment from locking itself out:
* the LAST active admin can be neither demoted nor disabled (409);
* an admin cannot disable themselves.

Disable is a complete cut-off cascade (no half-disabled state): mirror
flag + session purge + PAT revoke, plus the local credential for local
accounts so the password login is refused too.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from inqtrix.server.routers._admin_guard import require_instance_admin
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.auth.directory import MirroredUser

TENANT = "default"
_MIN_PASSWORD_LEN = 12


def _user_payload(user: "MirroredUser") -> dict:
    return {
        "subject": user.subject,
        "email": user.email,
        "display_name": user.display_name,
        "instance_role": user.instance_role,
        "disabled": user.disabled_at is not None,
        "last_login_at": user.last_login_at,
    }


def build_admin_router(provider) -> APIRouter:
    """Bind the admin routes against one cookie-session provider."""
    router = APIRouter()
    users = provider.users

    async def _require_admin(request: Request):
        """Resolve a session principal and require instance admin.

        Delegates to the shared :func:`require_instance_admin` guard so the
        instance-admin check is defined exactly once (Designprinzip 4) and
        every administrative surface (users, quota, workspaces) authorizes
        byte-identically. The local name keeps the call sites below
        unchanged.
        """
        return await require_instance_admin(provider, request)

    async def _find_target(subject: str) -> "MirroredUser | None":
        """Resolve a target by its subject (carries the issuer for ops)."""
        for user in await users.list_users(tenant_id=TENANT):
            if user.subject == subject:
                return user
        return None

    @router.get("/v1/admin/users")
    async def list_users(request: Request):
        resolved, error = await _require_admin(request)
        if error is not None:
            return error
        rows = await users.list_users(tenant_id=TENANT)
        return {"users": [_user_payload(user) for user in rows]}

    @router.patch("/v1/admin/users/{subject}")
    async def set_role(subject: str, request: Request):
        resolved, error = await _require_admin(request)
        if error is not None:
            return error
        _principal, session, _mirror = resolved
        try:
            body = await request.json()
        except Exception:
            return error_response(
                400, "Ungueltiger JSON-Body", "invalid_request_error"
            )
        role = str((body or {}).get("instance_role", "")).strip()
        if role not in {"admin", "user"}:
            return error_response(
                400,
                "Feld 'instance_role' muss 'admin' oder 'user' sein",
                "invalid_request_error",
            )
        target = await _find_target(subject)
        if target is None:
            return error_response(404, "Benutzer nicht gefunden", "not_found")
        if role == "user" and target.subject == session.sub:
            # Self-demotion is blocked (mirrors the self-disable guard and the
            # UI): an admin must not silently drop their own access — another
            # admin demotes them. The last-admin guard below is the separate,
            # always-on invariant.
            return error_response(
                409, "Sie koennen sich nicht selbst herabstufen", "self_demote"
            )
        if role == "user":
            # Atomic last-admin guard: demoting the only active admin would
            # lock the instance out of its own administration. The check and
            # the write are ONE operation, so concurrent demotions cannot both
            # pass a stale count and strand the instance with zero admins.
            if not await users.demote_if_not_last_admin(
                tenant_id=TENANT, issuer=target.issuer, subject=target.subject
            ):
                return error_response(
                    409,
                    "Der letzte Admin kann nicht herabgestuft werden",
                    "last_admin",
                )
        else:
            await users.set_instance_role(
                tenant_id=TENANT,
                issuer=target.issuer,
                subject=target.subject,
                role="admin",
            )
        updated = await _find_target(subject)
        return _user_payload(updated)

    async def _set_user_disabled(subject: str, *, disabled: bool, request: Request):
        resolved, error = await _require_admin(request)
        if error is not None:
            return error
        _principal, session, _mirror = resolved
        target = await _find_target(subject)
        if target is None:
            return error_response(404, "Benutzer nicht gefunden", "not_found")
        if disabled and target.subject == session.sub:
            return error_response(
                409, "Sie koennen sich nicht selbst deaktivieren", "self_disable"
            )
        now = time.time()
        disabled_at = now if disabled else None
        if disabled:
            # Atomic last-admin guard (same race-free contract as demote):
            # the durable mirror flag is set only when this is not the last
            # active admin. Denies oidc/ldap re-admission once set.
            if not await users.disable_if_not_last_admin(
                tenant_id=TENANT,
                issuer=target.issuer,
                subject=target.subject,
                disabled_at=now,
            ):
                return error_response(
                    409,
                    "Der letzte Admin kann nicht deaktiviert werden",
                    "last_admin",
                )
        else:
            # Re-enable clears the flag; enabling never reduces admins.
            await users.set_disabled(
                tenant_id=TENANT,
                issuer=target.issuer,
                subject=target.subject,
                disabled_at=None,
            )
        # Local accounts: the credential is the source of truth for the
        # password login, so disable/enable it too.
        credentials = getattr(provider, "credentials", None)
        if credentials is not None:
            await credentials.set_disabled(
                tenant_id=TENANT, subject=target.subject, disabled_at=disabled_at
            )
        if disabled:
            # Cut off live access immediately: sessions + tokens.
            await provider.sessions.delete_for_owner(
                issuer=target.issuer, sub=target.subject
            )
            if provider.pat_service is not None:
                await provider.pat_service.revoke_all_for_owner(
                    tenant_id=TENANT,
                    owner_issuer=target.issuer,
                    owner_sub=target.subject,
                )
        updated = await _find_target(subject)
        return _user_payload(updated)

    @router.post("/v1/admin/users/{subject}:disable")
    async def disable_user(subject: str, request: Request):
        return await _set_user_disabled(subject, disabled=True, request=request)

    @router.post("/v1/admin/users/{subject}:enable")
    async def enable_user(subject: str, request: Request):
        return await _set_user_disabled(subject, disabled=False, request=request)

    if provider.mode == "local":
        from inqtrix.auth.credentials import (
            LOCAL_ISSUER,
            LocalCredential,
            hash_password,
            new_subject,
        )

        @router.post("/v1/admin/users", status_code=201)
        async def create_local_user(request: Request):
            resolved, error = await _require_admin(request)
            if error is not None:
                return error
            try:
                body = await request.json()
            except Exception:
                return error_response(
                    400, "Ungueltiger JSON-Body", "invalid_request_error"
                )
            email = str((body or {}).get("email", "")).strip()
            password = str((body or {}).get("password", ""))
            role = str((body or {}).get("instance_role", "user")).strip()
            if "@" not in email or len(email) > 320:
                return error_response(
                    400, "Ungueltige E-Mail-Adresse", "invalid_request_error"
                )
            if len(password) < _MIN_PASSWORD_LEN:
                return error_response(
                    400,
                    f"Passwort muss mindestens {_MIN_PASSWORD_LEN} Zeichen lang sein",
                    "invalid_request_error",
                )
            if role not in {"admin", "user"}:
                return error_response(
                    400,
                    "Feld 'instance_role' muss 'admin' oder 'user' sein",
                    "invalid_request_error",
                )
            display_name = str((body or {}).get("display_name", "")).strip() or email
            credential = LocalCredential(
                subject=new_subject(),
                email=email,
                password_hash=hash_password(password),
                display_name=display_name,
                created_at=time.time(),
            )
            created = await provider.credentials.create(
                credential, tenant_id=TENANT, allow_first_only=False
            )
            if not created:
                return error_response(
                    409, "E-Mail-Adresse ist bereits vergeben", "duplicate_email"
                )
            if users is not None:
                await users.record_login(
                    tenant_id=TENANT,
                    issuer=LOCAL_ISSUER,
                    subject=credential.subject,
                    email=email,
                    email_verified=True,
                    display_name=display_name,
                )
                if role == "admin":
                    await users.set_instance_role(
                        tenant_id=TENANT,
                        issuer=LOCAL_ISSUER,
                        subject=credential.subject,
                        role="admin",
                    )
            target = await _find_target(credential.subject)
            return _user_payload(target)

        @router.post("/v1/admin/users/{subject}:reset-password")
        async def reset_password(subject: str, request: Request):
            """Admin sets a new password for a local account.

            Forgotten-password recovery without email: the admin supplies the
            new password (the UI shows it once, like create). Live sessions are
            purged so the old password — and anything minted with it — stops
            working; PATs are intentionally left intact (a reset is not the
            full disable cut-off). Does not change the disabled state.
            """
            resolved, error = await _require_admin(request)
            if error is not None:
                return error
            try:
                body = await request.json()
            except Exception:
                return error_response(
                    400, "Ungueltiger JSON-Body", "invalid_request_error"
                )
            password = str((body or {}).get("password", ""))
            if len(password) < _MIN_PASSWORD_LEN:
                return error_response(
                    400,
                    f"Passwort muss mindestens {_MIN_PASSWORD_LEN} Zeichen lang sein",
                    "invalid_request_error",
                )
            credential = await provider.credentials.get_by_subject(
                tenant_id=TENANT, subject=subject
            )
            if credential is None:
                return error_response(
                    404, "Kein lokales Konto gefunden", "not_found"
                )
            await provider.credentials.set_password(
                tenant_id=TENANT,
                subject=subject,
                password_hash=hash_password(password),
            )
            await provider.sessions.delete_for_owner(
                issuer=LOCAL_ISSUER, sub=subject
            )
            return {"reset": True}

    return router
