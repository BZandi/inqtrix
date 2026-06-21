"""Share administration: the CRUD side of ``resource_shares``.

The READ side (``ShareRepository.permission_for``) has powered the
permission layer since the identity schema landed; this module adds
the missing write/listing surface plus the service the ``/v1/shares``
routes call. Google-Drive model: a share grants access to THE ONE
server resource (no copies); recipients list shared-in resources via
``shares_for_subjects``.

v1 scope (binding): direct USER shares only (``subject_type='user'``)
and the permissions ``view``/``edit``. Group shares and ``manage``
grants stay schema-supported for later — the permission layer already
unions them — but no UI/endpoint mints them yet.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Mapping, Protocol, Sequence

from inqtrix.auth.permissions import (
    AuditEntry,
    SharePermission,
    SubjectRef,
)

if TYPE_CHECKING:
    from inqtrix.auth.permissions import AuditSink, PermissionService
    from inqtrix.auth.principal import Principal

log = logging.getLogger("inqtrix")

GRANTABLE_PERMISSIONS = (SharePermission.VIEW, SharePermission.EDIT)
"""Permissions the v1 surface may mint. ``manage`` stays reserved for
owners; ``comment`` waits for a commenting surface."""


@dataclass(frozen=True)
class ShareRecord:
    """One share grant (the full row, not just the permission).

    Attributes:
        id: Server-assigned share identifier (revocation handle).
        tenant_id: Tenant scope.
        subject_type: ``"user"`` in v1 (groups later).
        subject_id: The recipient's ``sub``.
        resource_type: Polymorphic resource kind (``"run"``,
            ``"knowledge_collection"``, ``"prompt_template"``).
        resource_id: Identifier within the resource kind.
        permission: Granted level.
        granted_by_sub: Granting subject (audit/display).
        created_at: Unix seconds.
    """

    id: str
    tenant_id: str
    subject_type: str
    subject_id: str
    resource_type: str
    resource_id: str
    permission: SharePermission
    granted_by_sub: str
    created_at: float


class ShareNotAllowed(Exception):
    """Raised when a share operation is not permitted.

    Two cases, both hidden behind the same 404 (denial indistinguishable
    from absence — the surface convention): the caller may not manage shares
    on the resource, OR — under workspace-scoped sharing — the invitee is not
    a co-member (and so must not be revealed as existing). Carries no detail
    on purpose; the operator-facing visibility happens via log + audit.
    """


class ShareValidationError(Exception):
    """Raised on invalid grant input (German message for the 400)."""


class ShareAdminRepository(Protocol):
    """Write/listing port over ``resource_shares``."""

    async def create_share(
        self,
        *,
        tenant_id: str,
        subject_type: str,
        subject_id: str,
        resource_type: str,
        resource_id: str,
        permission: SharePermission,
        granted_by_sub: str,
    ) -> ShareRecord:
        """Grant (or re-grant) one share.

        Re-granting an existing active tuple REPLACES its permission
        (soft-revoke + insert in the Postgres backend, honoring the
        partial unique index) — the caller's intent "share with X as
        editor" must win over a stale earlier grant.
        """
        ...

    async def get_share(
        self, *, tenant_id: str, share_id: str
    ) -> ShareRecord | None:
        """One ACTIVE share by id (authorization happens above)."""
        ...

    async def revoke_share_by_id(
        self, *, tenant_id: str, share_id: str, revoked_by_sub: str
    ) -> ShareRecord | None:
        """Soft-revoke one ACTIVE share; returns it, or ``None``.

        Named with the ``_by_id`` suffix because the memory identity
        store already carries a tuple-keyed ``revoke_share`` test
        seam — a same-name overload would shadow it.
        """
        ...

    async def revoke_shares_for_resource(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        revoked_by_sub: str,
    ) -> int:
        """Soft-revoke EVERY active share on one resource.

        The cleanup half of resource deletion (a deleted knowledge
        collection must not leave grants dangling); returns the number
        of revoked shares. Authorization is the deleting caller's —
        the deletion itself already passed the owner gate.
        """
        ...

    async def list_shares_for_resource(
        self, *, tenant_id: str, resource_type: str, resource_id: str
    ) -> tuple[ShareRecord, ...]:
        """Active shares on one resource, oldest first."""
        ...

    async def shares_for_subjects(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        subjects: Sequence[SubjectRef],
    ) -> Mapping[str, ShareRecord]:
        """``resource_id -> highest-permission share`` for the subjects.

        Powers shared-with-me listings and the run-visibility union;
        one indexed query per request.
        """
        ...

    async def share_counts_for_resources(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_ids: Sequence[str],
    ) -> Mapping[str, int]:
        """Active-share count per resource (badge numbers)."""
        ...


OwnerResolver = Callable[[str, str], Awaitable[str | None]]
"""``(tenant_id, resource_id) -> owner_sub`` or ``None`` when the
resource does not exist. One resolver per shareable resource type,
registered by the composition root."""


class ShareService:
    """Orchestrates grants behind the ``/v1/shares`` routes.

    Args:
        shares: The share write/listing repository.
        permissions: The permission chokepoint (manage checks and
            subject assembly).
        owner_resolvers: ``resource_type -> resolver``; an unknown
            type is a 400, a vanished resource a 404.
        user_lookup: Async ``(tenant_id, sub) -> bool`` existence
            check for invitees (typo guard against granting to
            nonexistent subjects). The users mirror backs it.
        audit: Audit sink for grant/revoke events.
        restrict_to_members: When true, a grant is permitted only between
            workspace co-members (``settings.sharing.restrict_to_workspace_members``).
            Default false keeps sharing tenant-wide, byte-identical. It is a
            grant-time (write) check only — the read/``can`` semantics are
            unchanged, so flipping it never revokes an existing grant.
    """

    def __init__(
        self,
        *,
        shares: ShareAdminRepository,
        permissions: "PermissionService",
        owner_resolvers: Mapping[str, OwnerResolver],
        user_lookup: Callable[[str, str], Awaitable[bool]],
        audit: "AuditSink | None" = None,
        restrict_to_members: bool = False,
    ) -> None:
        self._shares = shares
        self._permissions = permissions
        self._owner_resolvers = dict(owner_resolvers)
        self._user_lookup = user_lookup
        self._audit = audit
        self._restrict_to_members = restrict_to_members

    @property
    def resource_types(self) -> tuple[str, ...]:
        """The shareable resource kinds this deployment knows."""
        return tuple(sorted(self._owner_resolvers))

    async def _owner_or_manager(
        self, principal: "Principal", resource_type: str, resource_id: str
    ) -> str:
        """The resource owner's sub; raises when the caller may not
        manage shares (owner or an explicit ``manage`` grant)."""
        resolver = self._owner_resolvers.get(resource_type)
        if resolver is None:
            raise ShareValidationError(
                "Unbekannter Ressourcentyp: " + resource_type
            )
        owner_sub = await resolver(principal.tenant_id, resource_id)
        if owner_sub is None:
            raise ShareNotAllowed()
        if owner_sub == principal.sub:
            return owner_sub
        if await self._permissions.can(
            principal,
            SharePermission.MANAGE,
            resource_type=resource_type,
            resource_id=resource_id,
        ):
            return owner_sub
        raise ShareNotAllowed()

    async def grant(
        self,
        principal: "Principal",
        *,
        resource_type: str,
        resource_id: str,
        invitees: Sequence[tuple[str, SharePermission]],
    ) -> tuple[ShareRecord, ...]:
        """Grant shares to *invitees* (``(subject_sub, permission)``).

        Invitees are validated and written one at a time in order, so a later
        invitee that fails any check (including the workspace-member boundary)
        raises AFTER earlier valid ones were already persisted —
        first-failure-wins, no rollback. The ``/v1/shares`` UI sends one
        invitee per request, so this partial-write window is not user-visible;
        a multi-invitee caller must treat a raised error as "earlier grants in
        the list may have landed".

        Raises:
            ShareValidationError: Unknown resource type, ungrantable
                permission, self/owner grants, or (tenant-wide sharing) an
                unknown invitee.
            ShareNotAllowed: Caller may not manage the resource, or — when
                workspace-scoped sharing is on — the invitee is not a
                co-member. Both render 404 (denial hidden behind absence);
                the boundary denial is logged + audited.
        """
        owner_sub = await self._owner_or_manager(
            principal, resource_type, resource_id
        )
        created: list[ShareRecord] = []
        for subject_sub, permission in invitees:
            if permission not in GRANTABLE_PERMISSIONS:
                raise ShareValidationError(
                    "Berechtigung muss 'view' oder 'edit' sein"
                )
            if subject_sub == principal.sub:
                raise ShareValidationError(
                    "Eine Freigabe an sich selbst ist nicht moeglich"
                )
            if subject_sub == owner_sub:
                raise ShareValidationError(
                    "Die Eigentuemerin braucht keine Freigabe"
                )
            if self._restrict_to_members:
                # Workspace-scoped sharing: the invitee must be a co-member.
                # A non-co-member — or a non-existent sub, which belongs to no
                # workspace — is hidden behind the SAME 404 as a foreign
                # resource (indistinguishable, consistent with the scoped
                # typeahead that never offers them, so co-membership is not an
                # existence oracle). The denial is operator-visible via
                # log + audit, never silent (Designprinzip 1).
                if not await self._permissions.share_workspace(
                    tenant_id=principal.tenant_id,
                    sub_a=principal.sub,
                    sub_b=subject_sub,
                ):
                    await self._deny_share(
                        principal, resource_type, resource_id, subject_sub
                    )
                    raise ShareNotAllowed()
            elif not await self._user_lookup(
                principal.tenant_id, subject_sub
            ):
                raise ShareValidationError(
                    "Nutzer nicht gefunden: " + subject_sub
                )
            record = await self._shares.create_share(
                tenant_id=principal.tenant_id,
                subject_type="user",
                subject_id=subject_sub,
                resource_type=resource_type,
                resource_id=resource_id,
                permission=permission,
                granted_by_sub=principal.sub,
            )
            created.append(record)
            await self._audit_event(
                principal, "share.granted", resource_type, resource_id,
                {"subject": subject_sub, "permission": permission.value},
            )
        return tuple(created)

    async def revoke(
        self, principal: "Principal", *, share_id: str
    ) -> bool:
        """Revoke one share the caller may manage.

        Authorization runs BEFORE the write: the share is read, the
        manage check applies to ITS resource, and only then does the
        guarded revoke land. Unknown ids and foreign resources are
        indistinguishable ``False`` (the router's 404).
        """
        share = await self._shares.get_share(
            tenant_id=principal.tenant_id, share_id=share_id
        )
        if share is None:
            return False
        try:
            await self._owner_or_manager(
                principal, share.resource_type, share.resource_id
            )
        except (ShareNotAllowed, ShareValidationError):
            return False
        revoked = await self._shares.revoke_share_by_id(
            tenant_id=principal.tenant_id,
            share_id=share_id,
            revoked_by_sub=principal.sub,
        )
        if revoked is None:
            return False
        await self._audit_event(
            principal,
            "share.revoked",
            revoked.resource_type,
            revoked.resource_id,
            {"subject": revoked.subject_id},
        )
        return True

    async def list_for_resource(
        self, principal: "Principal", *, resource_type: str, resource_id: str
    ) -> tuple[ShareRecord, ...]:
        """Active shares on a resource the caller may at least view."""
        resolver = self._owner_resolvers.get(resource_type)
        if resolver is None:
            raise ShareValidationError(
                "Unbekannter Ressourcentyp: " + resource_type
            )
        owner_sub = await resolver(principal.tenant_id, resource_id)
        if owner_sub is None:
            raise ShareNotAllowed()
        if owner_sub != principal.sub and not await self._permissions.can(
            principal,
            SharePermission.VIEW,
            resource_type=resource_type,
            resource_id=resource_id,
        ):
            raise ShareNotAllowed()
        return await self._shares.list_shares_for_resource(
            tenant_id=principal.tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )

    async def shared_with_me(
        self, principal: "Principal", *, resource_type: str
    ) -> Mapping[str, ShareRecord]:
        """Shared-in resources of one kind for the caller."""
        subjects = await self._permissions.subjects_for(principal)
        return await self._shares.shares_for_subjects(
            tenant_id=principal.tenant_id,
            resource_type=resource_type,
            subjects=subjects,
        )

    async def outgoing_counts(
        self,
        principal: "Principal",
        *,
        resource_type: str,
        resource_ids: Sequence[str],
    ) -> Mapping[str, int]:
        """Active-share counts for badge rendering — owned ids only.

        Foreign ids are silently dropped BEFORE counting: counts are
        an existence/metadata oracle, and a caller probing resources
        they do not own must learn nothing (the same hiding rule as
        every other denial on this surface).

        Raises:
            ShareValidationError: Unknown resource type.
        """
        resolver = self._owner_resolvers.get(resource_type)
        if resolver is None:
            raise ShareValidationError(
                "Unbekannter Ressourcentyp: " + resource_type
            )
        owned_ids = []
        for resource_id in resource_ids:
            owner_sub = await resolver(principal.tenant_id, resource_id)
            if owner_sub == principal.sub:
                owned_ids.append(resource_id)
        if not owned_ids:
            return {}
        return await self._shares.share_counts_for_resources(
            tenant_id=principal.tenant_id,
            resource_type=resource_type,
            resource_ids=owned_ids,
        )

    async def _deny_share(
        self,
        principal: "Principal",
        resource_type: str,
        resource_id: str,
        subject_sub: str,
    ) -> None:
        """Make a workspace-boundary share denial operator-visible.

        Logged AND audited (mirroring :meth:`PermissionService._deny`) even
        though the caller only ever sees an indistinct 404 — an authorization
        denial must never be silent (Designprinzip 1).
        """
        log.warning(
            "share denied: actor=%s resource=%s/%s subject=%s "
            "reason=not_workspace_member",
            principal.sub,
            resource_type,
            resource_id,
            subject_sub,
        )
        await self._audit_event(
            principal,
            "share.denied",
            resource_type,
            resource_id,
            {"subject": subject_sub, "reason": "not_workspace_member"},
        )

    async def _audit_event(
        self,
        principal: "Principal",
        action: str,
        resource_type: str,
        resource_id: str,
        detail: dict[str, str],
    ) -> None:
        if self._audit is None:
            return
        await self._audit.record(
            AuditEntry(
                tenant_id=principal.tenant_id,
                actor_sub=principal.sub,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                detail=detail,
            )
        )
