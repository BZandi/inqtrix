"""Login-time provisioning shared by every cookie-session transport.

The single home for "turn a verified login into local instance facts" so
the OIDC, LDAP (and future) callbacks cannot drift (Designprinzip #4 keine
Redundanz). It is deliberately I/O-only over the
:class:`~inqtrix.auth.directory.UserDirectory`; the claim-to-signal logic
lives in the pure :mod:`inqtrix.auth.mapping` layer, which this module
does not import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inqtrix.auth.directory import UserDirectory


async def apply_admin_grant(
    users: "UserDirectory",
    *,
    tenant_id: str,
    issuer: str,
    subject: str,
    is_admin: bool,
    first_login_owner: bool,
) -> None:
    """Grant-only instance-admin provisioning from a verified login.

    The one path both the OIDC admin-claim mapping and the LDAP admin-group
    match flow through, so their semantics stay identical.

    Args:
        users: The local user mirror to mutate.
        tenant_id: Tenant anchor (currently always ``"default"``).
        issuer: Identity issuer — the real OIDC issuer or a synthetic
            transport issuer (``"ldap"``).
        subject: Stable subject id within *issuer*.
        is_admin: Whether the directory/IdP signalled admin for this user.
        first_login_owner: When no admin signal is present, whether the
            first ever login bootstraps the instance owner.

    Behaviour:
        Promotes to admin on a positive *is_admin* signal; otherwise, when
        *first_login_owner* is set, the first authenticated user on a fresh
        instance becomes the owner (a no-op once any admin exists). It
        NEVER demotes: role revocation is the admin surface's responsibility
        (it owns the last-admin guard), so a directory group or IdP claim
        only ever promotes — auto-demoting here would silently undo an
        admin-UI promotion and sidestep that guard.
    """
    if is_admin:
        await users.set_instance_role(
            tenant_id=tenant_id, issuer=issuer, subject=subject, role="admin"
        )
    elif first_login_owner:
        await users.promote_if_no_admin(
            tenant_id=tenant_id, issuer=issuer, subject=subject
        )
