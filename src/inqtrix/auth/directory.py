"""Local user mirror: JIT provisioning keyed on ``(issuer, subject)``.

The identity anchor is the pair — subjects are only unique per issuer
and email is mutable profile data, never identity (the ``users``
table's ``UNIQUE(issuer, subject)`` pins this). Every successful OIDC
login upserts the mirror row so collaboration features (share
typeaheads, audit display names) have a local source without ever
proxying the IdP's admin API.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MirroredUser:
    """One mirrored identity (memory backend's record shape).

    Attributes:
        disabled_at: Soft-disable timestamp; a disabled user is denied
            at login admission. ``None`` (the default, and the value
            for rows written before the field existed) means active.
    """

    issuer: str
    subject: str
    email: str
    email_verified: bool
    display_name: str | None
    disabled_at: float | None = None
    # Instance-wide role for the admin surface (admin|user). Additive
    # default so every existing record / serialized row reads as a regular
    # user (mirrors the disabled_at additive precedent).
    instance_role: str = "user"
    # Last successful login (unix seconds); audit column for the admin list.
    last_login_at: float | None = None
    # The user's canonical project namespace (a ``ws_...`` string), adopted on
    # first authenticated boot so the project follows the user across devices.
    # Additive default ``None`` = not yet adopted (mirrors the disabled_at /
    # instance_role additive precedent).
    default_workspace_id: str | None = None


class UserDirectory(Protocol):
    """Port for the local user mirror."""

    async def record_login(
        self,
        *,
        tenant_id: str,
        issuer: str,
        subject: str,
        email: str,
        email_verified: bool,
        display_name: str | None,
    ) -> None:
        """Upsert one identity on successful login (JIT provisioning)."""
        ...

    async def find_user(
        self, *, tenant_id: str, issuer: str, subject: str
    ) -> "MirroredUser | None":
        """The mirrored identity (incl. instance_role), or ``None``."""
        ...

    async def set_instance_role(
        self, *, tenant_id: str, issuer: str, subject: str, role: str
    ) -> bool:
        """Set the instance role (admin|user); ``True`` when a row changed."""
        ...

    async def resolve_default_workspace(
        self, *, tenant_id: str, issuer: str, subject: str, candidate: str
    ) -> str | None:
        """Return the user's canonical project namespace, ADOPTING ``candidate``
        on the first call (when none is set yet).

        Idempotent claim-on-first-use: if the user already has a
        ``default_workspace_id`` it is returned unchanged (``candidate`` is
        ignored), so every device converges on the namespace adopted by the
        first authenticated boot — the data follows the user. The write is
        atomic (set-only-when-NULL) so concurrent first-logins cannot split the
        namespace. Returns ``None`` only when the user row is absent (an
        un-mirrored identity), so the caller falls back to the browser-local id.
        """
        ...

    async def set_disabled(
        self,
        *,
        tenant_id: str,
        issuer: str,
        subject: str,
        disabled_at: float | None,
    ) -> bool:
        """Set/clear the mirror disable flag; ``True`` when a row changed."""
        ...

    async def list_users(
        self, *, tenant_id: str
    ) -> tuple["MirroredUser", ...]:
        """All mirrored identities in the tenant (admin listing)."""
        ...

    async def count_admins(self, *, tenant_id: str) -> int:
        """Active instance-admin count (the last-admin guard)."""
        ...

    async def promote_if_no_admin(
        self, *, tenant_id: str, issuer: str, subject: str
    ) -> bool:
        """Promote to admin iff no active admin exists (first-login owner)."""
        ...

    async def demote_if_not_last_admin(
        self, *, tenant_id: str, issuer: str, subject: str
    ) -> bool:
        """Demote to ``user`` atomically, guarding the last-admin invariant.

        The check ("is this the last active admin?") and the write are ONE
        operation so two concurrent demotions cannot both pass a stale count
        and leave the instance with zero admins. Returns ``True`` when the
        demotion was applied (including the no-op case where the target was
        not an active admin), ``False`` ONLY when the target is the last
        active admin (the caller maps that to 409) or the row is absent.
        """
        ...

    async def disable_if_not_last_admin(
        self, *, tenant_id: str, issuer: str, subject: str, disabled_at: float
    ) -> bool:
        """Set the disable flag atomically, guarding the last-admin invariant.

        Same contract as :meth:`demote_if_not_last_admin`: ``True`` when the
        flag was set (including disabling a non-admin), ``False`` only when
        the target is the last active admin or the row is absent. The rest of
        the cut-off cascade (sessions, PATs, the local credential) is left to
        the admin router and runs only after this returns ``True``.
        """
        ...


class UserLookup(Protocol):
    """Narrow read port for admission checks.

    A separate protocol on purpose: growing :class:`UserDirectory`
    would silently break implementations elsewhere; admission only
    needs this one lookup.
    """

    async def find_user(
        self, *, tenant_id: str, issuer: str, subject: str
    ) -> MirroredUser | None:
        """The mirrored identity, or ``None`` for first-time logins."""
        ...


class MemoryUserDirectory:
    """Process-local mirror (zero-infrastructure default)."""

    def __init__(self) -> None:
        self.users: dict[tuple[str, str], MirroredUser] = {}
        self._lock = threading.Lock()

    async def record_login(
        self,
        *,
        tenant_id: str,
        issuer: str,
        subject: str,
        email: str,
        email_verified: bool,
        display_name: str | None,
    ) -> None:
        import time

        with self._lock:
            existing = self.users.get((issuer, subject))
            self.users[(issuer, subject)] = MirroredUser(
                issuer=issuer,
                subject=subject,
                email=email,
                email_verified=email_verified,
                display_name=display_name,
                # Preserve admin-managed state (disable + role) across logins;
                # only profile fields and the login timestamp refresh.
                disabled_at=(
                    existing.disabled_at if existing is not None else None
                ),
                instance_role=(
                    existing.instance_role if existing is not None else "user"
                ),
                last_login_at=time.time(),
                # Preserve the adopted project namespace across logins (like
                # disabled_at / instance_role) so re-login does not orphan data.
                default_workspace_id=(
                    existing.default_workspace_id if existing is not None else None
                ),
            )

    async def resolve_default_workspace(
        self, *, tenant_id: str, issuer: str, subject: str, candidate: str
    ) -> str | None:
        """Return the adopted project namespace, claiming ``candidate`` if unset."""
        from dataclasses import replace

        with self._lock:
            existing = self.users.get((issuer, subject))
            if existing is None:
                return None
            if existing.default_workspace_id is not None:
                return existing.default_workspace_id
            self.users[(issuer, subject)] = replace(
                existing, default_workspace_id=candidate
            )
            return candidate

    async def set_instance_role(
        self, *, tenant_id: str, issuer: str, subject: str, role: str
    ) -> bool:
        """Set the instance role; ``True`` when a row changed."""
        from dataclasses import replace

        with self._lock:
            existing = self.users.get((issuer, subject))
            if existing is None:
                return False
            self.users[(issuer, subject)] = replace(
                existing, instance_role=role
            )
            return True

    async def set_disabled(
        self, *, tenant_id: str, issuer: str, subject: str, disabled_at: float | None
    ) -> bool:
        """Set/clear the soft-disable timestamp; ``True`` when a row changed."""
        from dataclasses import replace

        with self._lock:
            existing = self.users.get((issuer, subject))
            if existing is None:
                return False
            self.users[(issuer, subject)] = replace(
                existing, disabled_at=disabled_at
            )
            return True

    async def list_users(
        self, *, tenant_id: str
    ) -> tuple[MirroredUser, ...]:
        """All mirrored identities (admin listing)."""
        with self._lock:
            return tuple(sorted(self.users.values(), key=lambda u: u.email))

    async def count_admins(self, *, tenant_id: str) -> int:
        """Active instance-admins (the last-admin guard reads this)."""
        with self._lock:
            return sum(
                1
                for u in self.users.values()
                if u.instance_role == "admin" and u.disabled_at is None
            )

    async def promote_if_no_admin(
        self, *, tenant_id: str, issuer: str, subject: str
    ) -> bool:
        """Promote this user to admin IFF no active admin exists yet.

        The first-login-owner bootstrap, atomic under the lock: exactly
        the first authenticated user in a fresh deployment becomes admin.
        """
        from dataclasses import replace

        with self._lock:
            has_admin = any(
                u.instance_role == "admin" and u.disabled_at is None
                for u in self.users.values()
            )
            existing = self.users.get((issuer, subject))
            if has_admin or existing is None:
                return False
            self.users[(issuer, subject)] = replace(
                existing, instance_role="admin"
            )
            return True

    def _is_last_active_admin(
        self, existing: MirroredUser, issuer: str, subject: str
    ) -> bool:
        """Whether *existing* is an active admin with no other active admin.

        Caller must hold ``self._lock``. The basis of the atomic last-admin
        guards: if removing this user's active-admin status leaves none, the
        guarded op refuses.
        """
        if existing.instance_role != "admin" or existing.disabled_at is not None:
            return False
        return not any(
            u.instance_role == "admin"
            and u.disabled_at is None
            and (u.issuer, u.subject) != (issuer, subject)
            for u in self.users.values()
        )

    async def demote_if_not_last_admin(
        self, *, tenant_id: str, issuer: str, subject: str
    ) -> bool:
        """Demote to ``user`` unless this is the last active admin (atomic)."""
        from dataclasses import replace

        with self._lock:
            existing = self.users.get((issuer, subject))
            if existing is None:
                return False
            if self._is_last_active_admin(existing, issuer, subject):
                return False
            self.users[(issuer, subject)] = replace(
                existing, instance_role="user"
            )
            return True

    async def disable_if_not_last_admin(
        self, *, tenant_id: str, issuer: str, subject: str, disabled_at: float
    ) -> bool:
        """Set the disable flag unless this is the last active admin (atomic)."""
        from dataclasses import replace

        with self._lock:
            existing = self.users.get((issuer, subject))
            if existing is None:
                return False
            if self._is_last_active_admin(existing, issuer, subject):
                return False
            self.users[(issuer, subject)] = replace(
                existing, disabled_at=disabled_at
            )
            return True

    async def find_user(
        self, *, tenant_id: str, issuer: str, subject: str
    ) -> MirroredUser | None:
        with self._lock:
            return self.users.get((issuer, subject))

    async def profiles_for_subjects(
        self, *, tenant_id: str, subs: tuple[str, ...]
    ) -> dict[str, MirroredUser]:
        """``sub -> profile`` for share-listing enrichment (one pass)."""
        wanted = set(subs)
        with self._lock:
            return {
                user.subject: user
                for user in self.users.values()
                if user.subject in wanted
            }

    async def has_subject(self, *, tenant_id: str, sub: str) -> bool:
        """Whether any active mirrored identity carries this subject
        (the share-grant typo guard; issuer-agnostic by design — the
        share table keys on the bare sub)."""
        with self._lock:
            return any(
                user.subject == sub and user.disabled_at is None
                for user in self.users.values()
            )

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
        limit: int = 10,
        exclude_subject: str = "",
    ) -> tuple[MirroredUser, ...]:
        """Prefix search over email and display name (share typeahead).

        Disabled users never appear; the caller is excluded so the
        picker cannot offer self-shares.
        """
        needle = query.strip().lower()
        if not needle:
            return ()
        with self._lock:
            matches = [
                user
                for user in self.users.values()
                if user.disabled_at is None
                and user.subject != exclude_subject
                and (
                    user.email.lower().startswith(needle)
                    or (user.display_name or "").lower().startswith(needle)
                )
            ]
        return tuple(
            sorted(matches, key=lambda user: user.email)[:limit]
        )
