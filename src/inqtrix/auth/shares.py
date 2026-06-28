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
        accepted_at: Unix seconds the recipient consented, or ``None`` while
            the share is still pending. ``None`` grants no access (the consent
            gate); a non-``None`` value is what every visibility query filters
            on. Defaults to ``None`` so a freshly minted grant starts pending.
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
    accepted_at: float | None = None


@dataclass(frozen=True)
class InboxItem:
    """One share addressed to the caller, title-enriched for the inbox.

    The recipient-facing view of a :class:`ShareRecord`: it adds the resolved
    resource title (a pending recipient has no access yet, so the title comes
    from an owner-bypassing read) and keeps ``accepted_at`` so the surface can
    split pending invitations from accepted (active) shares.

    Attributes:
        share_id: Revocation/accept handle.
        resource_type: Polymorphic kind.
        resource_id: Identifier within the kind.
        resource_title: Human-readable title, resolved server-side.
        permission: Granted level.
        granted_by_sub: The grantor (the router joins the display name).
        created_at: Unix seconds the grant was minted.
        accepted_at: Unix seconds the caller consented, or ``None`` (pending).
    """

    share_id: str
    resource_type: str
    resource_id: str
    resource_title: str
    permission: SharePermission
    granted_by_sub: str
    created_at: float
    accepted_at: float | None


@dataclass(frozen=True)
class OutgoingItem:
    """One resource the caller has shared out, grouped across its recipients.

    Attributes:
        resource_type: Polymorphic kind.
        resource_id: Identifier within the kind.
        resource_title: Human-readable title, resolved server-side.
        share_count: Active shares on the resource granted by the caller.
        pending_count: Of those, how many are still awaiting consent.
    """

    resource_type: str
    resource_id: str
    resource_title: str
    share_count: int
    pending_count: int


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

    async def accept_share_by_id(
        self, *, tenant_id: str, share_id: str, subject_sub: str
    ) -> ShareRecord | None:
        """Mark one pending share accepted; returns it, or ``None``.

        Only flips a share that is active, still pending
        (``accepted_at IS NULL``), and addressed to *subject_sub* (the
        recipient is the sole party who can consent). Returns ``None`` for an
        unknown id, a foreign recipient, an already-accepted share, or a
        revoked one — all indistinguishable (the surface's 404 rule).
        """
        ...

    async def inbox_for_subjects(
        self,
        *,
        tenant_id: str,
        subjects: Sequence[SubjectRef],
    ) -> tuple[ShareRecord, ...]:
        """Every active (pending OR accepted) share addressed to the
        subjects, across ALL resource kinds — the recipient inbox source.

        Distinct from :meth:`shares_for_subjects`, which is accepted-only
        (the access union) and single-kind: the inbox needs the pending rows
        too so the recipient can consent to them.
        """
        ...

    async def outgoing_shares_for_grantor(
        self, *, tenant_id: str, grantor_sub: str
    ) -> tuple[ShareRecord, ...]:
        """Every active share *grantor_sub* granted, across all resource
        kinds — the "shared by me" source (oldest first)."""
        ...


OwnerResolver = Callable[[str, str], Awaitable[str | None]]
"""``(tenant_id, resource_id) -> owner_sub`` or ``None`` when the
resource does not exist. One resolver per shareable resource type,
registered by the composition root."""

TitleResolver = Callable[[str, str], Awaitable[str | None]]
"""``(tenant_id, resource_id) -> human-readable title`` or ``None`` when
the resource is gone. The same shape as :data:`OwnerResolver` but a
distinct registry: it reads the resource's display title (owner-bypassing,
so a pending recipient can see what they were offered). A missing resolver
for a kind simply yields no title — the inbox/outgoing listing skips that
row rather than inventing one."""


class ShareService:
    """Orchestrates grants behind the ``/v1/shares`` routes.

    Args:
        shares: The share write/listing repository.
        permissions: The permission chokepoint (manage checks and
            subject assembly).
        owner_resolvers: ``resource_type -> resolver``; an unknown
            type is a 400, a vanished resource a 404.
        title_resolvers: ``resource_type -> title resolver`` for the
            recipient inbox and the "shared by me" listing. Optional and
            additive: a kind with no resolver yields untitled rows that the
            listings skip, so the access/grant surface is unaffected.
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
        title_resolvers: Mapping[str, TitleResolver] | None = None,
    ) -> None:
        self._shares = shares
        self._permissions = permissions
        self._owner_resolvers = dict(owner_resolvers)
        self._title_resolvers = dict(title_resolvers or {})
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

    async def accept(
        self, principal: "Principal", *, share_id: str
    ) -> bool:
        """Accept one pending share addressed to the caller.

        Consent is the recipient's alone: the repository flips the share only
        when it is active, still pending, and has ``subject_id ==
        principal.sub``. Until this lands the share grants nothing — the
        accepted timestamp is exactly what every visibility query filters on.

        A successful acceptance is audited as ``share.accepted`` so the consent
        is operator-visible. A denial (unknown id, foreign recipient,
        already-accepted, revoked) returns an indistinguishable ``False`` — the
        same 404-hiding convention as the owner-side :meth:`revoke`, and not an
        audited security event: the benign cases dominate and the share id is a
        UUID a foreign caller cannot guess, so loud auditing here would be noise
        without signal (Designprinzip 5).
        """
        accepted = await self._shares.accept_share_by_id(
            tenant_id=principal.tenant_id,
            share_id=share_id,
            subject_sub=principal.sub,
        )
        if accepted is None:
            return False
        await self._audit_event(
            principal,
            "share.accepted",
            accepted.resource_type,
            accepted.resource_id,
            {"granted_by": accepted.granted_by_sub},
        )
        return True

    async def recipient_drop(
        self, principal: "Principal", *, share_id: str
    ) -> bool:
        """Remove the caller's OWN share — decline a pending invitation or
        leave an accepted one.

        The recipient is the only party this empowers (the owner uses
        :meth:`revoke` instead): the share must be a user share addressed to
        ``principal.sub``. It soft-revokes, stamped with the recipient as
        ``revoked_by_sub``, and audits ``share.declined`` (was pending) or
        ``share.left`` (was accepted) so the action is operator-visible. A
        foreign or unknown share is an indistinguishable ``False`` (the
        router's 404).
        """
        share = await self._shares.get_share(
            tenant_id=principal.tenant_id, share_id=share_id
        )
        if (
            share is None
            or share.subject_type != "user"
            or share.subject_id != principal.sub
        ):
            return False
        dropped = await self._shares.revoke_share_by_id(
            tenant_id=principal.tenant_id,
            share_id=share_id,
            revoked_by_sub=principal.sub,
        )
        if dropped is None:
            return False
        action = (
            "share.declined" if dropped.accepted_at is None else "share.left"
        )
        await self._audit_event(
            principal,
            action,
            dropped.resource_type,
            dropped.resource_id,
            {"granted_by": dropped.granted_by_sub},
        )
        return True

    async def inbox(self, principal: "Principal") -> tuple[InboxItem, ...]:
        """The caller's incoming DIRECT shares (pending + accepted), all kinds,
        title-enriched and oldest first.

        Pending rows are the consent queue; accepted rows are the "shared with
        me" list — the surface splits them on ``accepted_at``. Rows whose
        resource has vanished (no title) are dropped: an orphaned grant is
        nothing the recipient can act on. The grantor display name is joined by
        the router, which owns the users mirror.

        Only the caller's OWN user shares are listed — NOT shares to groups the
        caller belongs to. The inbox is the per-user consent surface, so every
        row must be one the caller can accept or drop themselves; a group share
        is a group-admin concern (and is schema-supported but never minted in
        v1). This keeps the inbox symmetric with :meth:`recipient_drop` — no
        undeclinable rows — and isolated from the permission/visibility union,
        which keeps unioning groups via ``subjects_for``.
        """
        subject = SubjectRef(subject_type="user", subject_id=principal.sub)
        records = await self._shares.inbox_for_subjects(
            tenant_id=principal.tenant_id, subjects=(subject,)
        )
        items: list[InboxItem] = []
        for record in records:
            title = await self._resolve_title(
                principal.tenant_id, record.resource_type, record.resource_id
            )
            if title is None:
                continue
            items.append(
                InboxItem(
                    share_id=record.id,
                    resource_type=record.resource_type,
                    resource_id=record.resource_id,
                    resource_title=title,
                    permission=record.permission,
                    granted_by_sub=record.granted_by_sub,
                    created_at=record.created_at,
                    accepted_at=record.accepted_at,
                )
            )
        return tuple(items)

    async def outgoing(
        self, principal: "Principal"
    ) -> tuple[OutgoingItem, ...]:
        """The resources the caller has shared out, grouped per resource with
        active and pending counts, title-enriched (oldest resource first).

        v1 reads ``granted_by_sub == caller`` — only owners mint shares today,
        so this is exactly "what I shared". Resources without a title (deleted)
        are dropped, mirroring :meth:`inbox`. (v2 note: if ``manage`` grants
        become grantable, a co-manager's re-grant would list under the
        re-grantor here, not the owner — revisit whether outgoing should track
        the originating owner separately.)
        """
        records = await self._shares.outgoing_shares_for_grantor(
            tenant_id=principal.tenant_id, grantor_sub=principal.sub
        )
        grouped: dict[tuple[str, str], list[ShareRecord]] = {}
        for record in records:
            grouped.setdefault(
                (record.resource_type, record.resource_id), []
            ).append(record)
        items: list[OutgoingItem] = []
        for (resource_type, resource_id), group in grouped.items():
            title = await self._resolve_title(
                principal.tenant_id, resource_type, resource_id
            )
            if title is None:
                continue
            pending = sum(1 for record in group if record.accepted_at is None)
            items.append(
                OutgoingItem(
                    resource_type=resource_type,
                    resource_id=resource_id,
                    resource_title=title,
                    share_count=len(group),
                    pending_count=pending,
                )
            )
        return tuple(items)

    async def _resolve_title(
        self, tenant_id: str, resource_type: str, resource_id: str
    ) -> str | None:
        """Human-readable title for one resource, or ``None`` when the kind has
        no resolver or the resource is gone — the listings skip such rows
        rather than showing a bare id."""
        resolver = self._title_resolvers.get(resource_type)
        if resolver is None:
            return None
        return await resolver(tenant_id, resource_id)

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
