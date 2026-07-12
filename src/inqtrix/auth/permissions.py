"""Permission chokepoint: workspace membership, shares, and visibility.

This module is the single authority for authorization decisions
(ADR-style commitment from the platform rebuild plan): routers and
services ask :class:`PermissionService`; nothing else combines
membership, share, and group facts. Postgres row-level security is a
coarse tenant defense layer underneath, never the fine-grained truth.

Two deliberate behavioural anchors:

* **Legacy principals are unscoped.** The ``anonymous`` and ``static``
  principals (open server / static Bearer key) predate multi-user
  operation; for them every visibility question resolves to "no
  filtering" so existing single-tenant deployments behave
  bit-for-bit. Scoping activates only for ``oidc_session``/``pat``
  principals.
* **Denied is indistinguishable from absent.** Failed access checks
  raise not-found errors, never 403 — resource existence is itself
  information (OWASP guidance; GitHub/Slack API behaviour). Every
  denial is still loudly visible to operators via ``log.warning`` and
  the audit sink (Designprinzip 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Mapping, Protocol, Sequence

from inqtrix.auth.principal import Principal, UserContext

log = logging.getLogger("inqtrix")


class WorkspaceRole(StrEnum):
    """Ordered coarse role of a member within one workspace.

    The order is total and ascending — every later role implies every
    earlier one. Comparisons go through :meth:`at_least` instead of
    enum identity so adding a role between two existing ones stays a
    one-line change.
    """

    VIEWER = "viewer"
    COMMENTER = "commenter"
    EDITOR = "editor"
    OWNER = "owner"

    def at_least(self, minimum: "WorkspaceRole") -> bool:
        """Whether this role grants at least *minimum*'s privileges."""
        return _WORKSPACE_ROLE_RANK[self] >= _WORKSPACE_ROLE_RANK[minimum]


_WORKSPACE_ROLE_RANK: dict[WorkspaceRole, int] = {
    role: rank for rank, role in enumerate(WorkspaceRole)
}


class SharePermission(StrEnum):
    """Ordered permission granted by one share tuple.

    Mirrors the ``share_permission`` Postgres enum; the ascending
    order makes ``can(edit)`` true for a ``manage`` grant without
    enumerating implications at every call site.
    """

    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    MANAGE = "manage"

    def at_least(self, minimum: "SharePermission") -> bool:
        """Whether this grant satisfies a requirement of *minimum*."""
        return _SHARE_PERMISSION_RANK[self] >= _SHARE_PERMISSION_RANK[minimum]


_SHARE_PERMISSION_RANK: dict[SharePermission, int] = {
    permission: rank for rank, permission in enumerate(SharePermission)
}


def highest_grant(
    grants: Iterable[SharePermission | None],
) -> SharePermission | None:
    """Reduce a set of grants to the single highest-ranked one.

    The authorization-critical "highest grant wins" fold, defined
    exactly once (Designprinzip 4) and shared by the service-level
    union and every share-repository backend — the comparator must
    never diverge between memory and Postgres.
    """
    best: SharePermission | None = None
    for grant in grants:
        if grant is not None and (best is None or grant.at_least(best)):
            best = grant
    return best

def grant_for_owned_resource(
    *,
    owner_sub: str | None,
    resource_tenant_id: str,
    resource_id: str,
    visible_to: UserContext | None,
    also_visible: "Mapping[str, SharePermission] | None",
) -> tuple[bool, SharePermission | None]:
    """The unified owner/legacy/share visibility rule, defined once.

    Returns ``(visible, shared_grant)``: ``(True, None)`` for full
    access without a share (unscoped callers, ownerless legacy
    resources, the owner), ``(True, grant)`` for shared-in access,
    and ``(False, None)`` when the caller may not see the resource —
    each resource kind raises its OWN not-found on that, keeping
    denial and absence byte-identical per surface. Shared by
    knowledge collections and prompt templates (and every future
    owned resource kind) so the rule cannot drift.
    """
    if visible_to is None:
        return True, None
    if owner_sub is None:
        return True, None
    principal = visible_to.principal
    if owner_sub == principal.sub and resource_tenant_id == principal.tenant_id:
        return True, None
    shared = (
        also_visible.get(resource_id) if also_visible is not None else None
    )
    if shared is not None:
        return True, shared
    return False, None


def resolve_owned_access(
    *,
    owner_sub: str | None,
    resource_tenant_id: str,
    resource_id: str,
    visible_to: "UserContext | None",
    also_visible: "Mapping[str, SharePermission] | None",
    not_found: type[KeyError],
) -> SharePermission | None:
    """:func:`grant_for_owned_resource` + raise the resource's own not-found.

    Returns ``None`` for full access without a share (unscoped caller,
    ownerless legacy resource, the owner) and the grant level for shared-in
    access; raises *not_found* (the resource kind's own KeyError) when the
    caller may not see it, so denial and absence stay byte-identical. The
    one access wrapper every owned-resource service shares (chat threads,
    editor documents, and any future kind) so the rule cannot drift
    (Designprinzip 4).
    """
    visible, shared = grant_for_owned_resource(
        owner_sub=owner_sub,
        resource_tenant_id=resource_tenant_id,
        resource_id=resource_id,
        visible_to=visible_to,
        also_visible=also_visible,
    )
    if not visible:
        raise not_found(resource_id)
    return shared


_ROLE_IMPLIED_PERMISSION: dict[WorkspaceRole, SharePermission] = {
    WorkspaceRole.VIEWER: SharePermission.VIEW,
    WorkspaceRole.COMMENTER: SharePermission.COMMENT,
    WorkspaceRole.EDITOR: SharePermission.EDIT,
    WorkspaceRole.OWNER: SharePermission.MANAGE,
}
"""Highest share-equivalent permission each workspace role implies on
resources living in that workspace. Kept as data (not branching) so
the union rule in :meth:`PermissionService.can` stays a single max."""


class WorkspaceNotFound(KeyError):
    """Raised when a workspace does not exist *for this principal*.

    Deliberately identical for "no such workspace" and "exists but the
    principal is not a member" — callers map it to HTTP 404.
    """


class ResourceNotFound(KeyError):
    """Raised when a resource access check fails for this principal.

    Same hiding rule as :class:`WorkspaceNotFound`: denial and absence
    are indistinguishable to the caller.
    """


@dataclass(frozen=True)
class SubjectRef:
    """One share subject the principal acts as (itself or a group)."""

    subject_type: str
    """``"user"`` or ``"group"`` — mirrors ``resource_shares.subject_type``."""
    subject_id: str
    """``users.sub`` for users, the group id for groups."""


@dataclass(frozen=True)
class AuditEntry:
    """One append-only audit fact emitted by the permission layer.

    Attributes:
        tenant_id: Tenant the action happened in.
        actor_sub: Subject that attempted the action.
        action: Verb-like action label (``"authz.denied"`` etc.).
        resource_type: Kind of resource acted on.
        resource_id: Identifier of the resource acted on.
        detail: Small free-form context (requested permission, route).
            Must never contain secrets or full request bodies.
        actor_type: ``user`` (default, every direct principal action) or
            ``agent`` — actions a workspace-agent run performs on the
            owner's behalf (``agent.*`` actions from the agent runtime).
            ``actor_sub`` then still carries the OWNING user's sub: the
            column distinguishes WHO acted, not on whose authority.
    """

    tenant_id: str
    actor_sub: str
    action: str
    resource_type: str
    resource_id: str
    detail: dict[str, str] = field(default_factory=dict)
    actor_type: str = "user"


class MembershipRepository(Protocol):
    """Read port for workspace membership facts."""

    async def workspace_ids_for(
        self, *, tenant_id: str, sub: str
    ) -> tuple[str, ...]:
        """All workspace ids *sub* is an active member of."""
        ...

    async def role_in_workspace(
        self, *, tenant_id: str, sub: str, workspace_id: str
    ) -> WorkspaceRole | None:
        """The member's role, or ``None`` when not a member (or no
        such workspace — the two are indistinguishable by design)."""
        ...


class MembershipAdminRepository(Protocol):
    """Write/admin port for workspace and membership management.

    The instance-admin surface (create/list/rename/delete workspaces, assign
    and remove members) builds on this. It is deliberately separate from the
    read-only :class:`MembershipRepository` the permission chokepoint consumes:
    administration is the platform-admin axis, membership resolution is the
    per-request authorization path. Both the memory and Postgres identity
    backends implement it, so the admin router is wired identically in both
    modes (Baukasten).
    """

    async def create_workspace(
        self, *, tenant_id: str, name: str, created_by_sub: str
    ) -> tuple[str, str]:
        """Create one workspace with *created_by_sub* as its OWNER.

        Returns ``(workspace_id, name)``.
        """
        ...

    async def list_all_workspaces(
        self, *, tenant_id: str
    ) -> tuple[tuple[str, str, str, int], ...]:
        """Every workspace in the tenant as ``(id, name, created_by_sub,
        member_count)``, name-sorted (the admin overview)."""
        ...

    async def rename_workspace(
        self, *, tenant_id: str, workspace_id: str, name: str
    ) -> bool:
        """Rename one workspace; ``False`` when it does not exist."""
        ...

    async def delete_workspace(
        self, *, tenant_id: str, workspace_id: str
    ) -> bool:
        """Delete one workspace and its memberships (cascade); ``False`` when
        it does not exist."""
        ...

    async def list_members(
        self, *, tenant_id: str, workspace_id: str
    ) -> tuple[tuple[str, WorkspaceRole], ...] | None:
        """``(sub, role)`` per member, sub-sorted, or ``None`` when the
        workspace does not exist — distinct from an existing workspace whose
        membership is empty, so the router can answer 404 vs 200."""
        ...

    async def assign_member(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        sub: str,
        role: WorkspaceRole,
    ) -> bool:
        """Add a member or change an existing member's role (upsert);
        ``False`` when the workspace does not exist.

        Unlike invitation acceptance (never-downgrade), the admin sets the
        EXACT role — raising or lowering it — because positioning users is a
        deliberate administrative act.
        """
        ...

    async def remove_member(
        self, *, tenant_id: str, workspace_id: str, sub: str
    ) -> bool:
        """Remove one membership; ``False`` when *sub* is not a member."""
        ...


class GroupRepository(Protocol):
    """Read port for group membership facts."""

    async def group_ids_for(self, *, tenant_id: str, sub: str) -> tuple[str, ...]:
        """All group ids *sub* is an active member of."""
        ...


class ShareRepository(Protocol):
    """Read port for direct/group share grants on resources."""

    async def permission_for(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        subjects: Sequence[SubjectRef],
    ) -> SharePermission | None:
        """Highest active grant any of *subjects* holds on the resource,
        or ``None`` when no grant exists."""
        ...


class AuditSink(Protocol):
    """Write port for the append-only audit log."""

    async def record(self, entry: AuditEntry) -> None:
        """Persist one audit fact. Implementations must be append-only."""
        ...


_UNSCOPED_PRINCIPAL_KINDS = frozenset({"anonymous", "static"})


class PermissionService:
    """The single authorization authority (chokepoint pattern).

    Combines three grant sources with a max-rank union: the
    principal's workspace role (implies a permission on resources in
    that workspace), direct user shares, and group shares. There are
    deliberately no negative/deny rules — revocation removes the
    grant, it never out-ranks another one.

    Args:
        members: Membership read port.
        groups: Group membership read port.
        shares: Share grant read port.
        audit: Append-only audit sink; receives every denial.
    """

    def __init__(
        self,
        *,
        members: MembershipRepository,
        groups: GroupRepository,
        shares: ShareRepository,
        audit: AuditSink,
    ) -> None:
        self._members = members
        self._groups = groups
        self._shares = shares
        self._audit = audit

    async def resolve_user_context(
        self, principal: Principal
    ) -> UserContext | None:
        """Resolve server-side membership facts for a principal.

        Returns:
            ``None`` for the legacy ``anonymous``/``static`` principals
            — the explicit "no visibility filtering" marker that keeps
            single-tenant deployments bit-for-bit. For scoped
            principals, a :class:`UserContext` with memberships
            resolved from the repositories (never from client input).
        """
        if principal.kind in _UNSCOPED_PRINCIPAL_KINDS:
            return None
        workspace_ids = await self._members.workspace_ids_for(
            tenant_id=principal.tenant_id, sub=principal.sub
        )
        group_ids = await self._groups.group_ids_for(
            tenant_id=principal.tenant_id, sub=principal.sub
        )
        return UserContext(
            principal=principal,
            workspace_ids=workspace_ids,
            groups=group_ids,
        )

    async def resolve_workspace(
        self,
        principal: Principal,
        workspace_id: str,
        *,
        min_role: WorkspaceRole = WorkspaceRole.VIEWER,
    ) -> str:
        """Validate that the principal may act in *workspace_id*.

        Unscoped legacy principals pass unconditionally (single-tenant
        owner semantics).

        Returns:
            The validated workspace id.

        Raises:
            WorkspaceNotFound: When the workspace does not exist, the
                principal is not a member, or the member's role is
                below *min_role* — indistinguishably.
        """
        if principal.kind in _UNSCOPED_PRINCIPAL_KINDS:
            return workspace_id
        role = await self._members.role_in_workspace(
            tenant_id=principal.tenant_id,
            sub=principal.sub,
            workspace_id=workspace_id,
        )
        if role is None or not role.at_least(min_role):
            await self._deny(
                principal,
                resource_type="workspace",
                resource_id=workspace_id,
                detail={"min_role": min_role.value, "role": getattr(role, "value", "")},
            )
            raise WorkspaceNotFound(workspace_id)
        return workspace_id

    async def can(
        self,
        principal: Principal,
        permission: SharePermission,
        *,
        resource_type: str,
        resource_id: str,
        workspace_id: str | None = None,
    ) -> bool:
        """Whether the principal holds *permission* on the resource.

        Grant union (max rank wins): workspace-role-implied permission
        (when *workspace_id* locates the resource in a workspace),
        direct user share, group shares.
        """
        if principal.kind in _UNSCOPED_PRINCIPAL_KINDS:
            return True
        role_implied: SharePermission | None = None
        if workspace_id is not None:
            role = await self._members.role_in_workspace(
                tenant_id=principal.tenant_id,
                sub=principal.sub,
                workspace_id=workspace_id,
            )
            if role is not None:
                role_implied = _ROLE_IMPLIED_PERMISSION[role]
        subjects = await self.subjects_for(principal)
        shared = await self._shares.permission_for(
            tenant_id=principal.tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            subjects=subjects,
        )
        best = highest_grant((role_implied, shared))
        return best is not None and best.at_least(permission)

    async def subjects_for(
        self, principal: Principal
    ) -> tuple[SubjectRef, ...]:
        """The share subjects this principal acts as (self + groups).

        The ONE assembly site — share lookups in routers/services must
        call this instead of rebuilding the user+groups union, so a
        future subject kind (e.g. workspace) lands everywhere at once.
        """
        subjects = [SubjectRef(subject_type="user", subject_id=principal.sub)]
        subjects.extend(
            SubjectRef(subject_type="group", subject_id=group_id)
            for group_id in await self._groups.group_ids_for(
                tenant_id=principal.tenant_id, sub=principal.sub
            )
        )
        return tuple(subjects)

    async def share_workspace_filter(
        self,
        *,
        tenant_id: str,
        grantor_sub: str,
        candidate_subs: Sequence[str],
    ) -> set[str]:
        """The subset of *candidate_subs* that share a workspace with the
        grantor — the batch primitive behind workspace-scoped sharing.

        The grantor's workspace set is read ONCE (it is loop-invariant), then
        each candidate's memberships are intersected against it. A grantor
        with no workspace shares one with nobody, so the result is empty.
        Reads only the membership port — no change to the ``can``/share read
        semantics. :meth:`share_workspace` is the single-candidate convenience.
        """
        grantor_workspaces = set(
            await self._members.workspace_ids_for(
                tenant_id=tenant_id, sub=grantor_sub
            )
        )
        if not grantor_workspaces:
            return set()
        allowed: set[str] = set()
        for sub in candidate_subs:
            candidate_workspaces = await self._members.workspace_ids_for(
                tenant_id=tenant_id, sub=sub
            )
            if any(
                workspace_id in grantor_workspaces
                for workspace_id in candidate_workspaces
            ):
                allowed.add(sub)
        return allowed

    async def share_workspace(
        self, *, tenant_id: str, sub_a: str, sub_b: str
    ) -> bool:
        """Whether two subjects share at least one workspace.

        The single-candidate membership-boundary predicate for workspace-
        scoped sharing: a grant is permitted only between co-members.
        Delegates to :meth:`share_workspace_filter` so the rule lives once.
        """
        return bool(
            await self.share_workspace_filter(
                tenant_id=tenant_id,
                grantor_sub=sub_a,
                candidate_subs=(sub_b,),
            )
        )

    async def require(
        self,
        principal: Principal,
        permission: SharePermission,
        *,
        resource_type: str,
        resource_id: str,
        workspace_id: str | None = None,
    ) -> None:
        """Assert *permission* or raise :class:`ResourceNotFound`.

        Denials are audited and logged before raising so authorization
        failures are operator-visible even though the client sees an
        indistinct 404 (Designprinzip 1).
        """
        allowed = await self.can(
            principal,
            permission,
            resource_type=resource_type,
            resource_id=resource_id,
            workspace_id=workspace_id,
        )
        if not allowed:
            await self._deny(
                principal,
                resource_type=resource_type,
                resource_id=resource_id,
                detail={"permission": permission.value},
            )
            raise ResourceNotFound(resource_id)

    async def _deny(
        self,
        principal: Principal,
        *,
        resource_type: str,
        resource_id: str,
        detail: dict[str, str],
    ) -> None:
        """Make one denial visible in the log and the audit trail."""
        log.warning(
            "authz denied: sub=%s kind=%s resource=%s/%s detail=%s",
            principal.sub,
            principal.kind,
            resource_type,
            resource_id,
            detail,
        )
        await self._audit.record(
            AuditEntry(
                tenant_id=principal.tenant_id,
                actor_sub=principal.sub,
                action="authz.denied",
                resource_type=resource_type,
                resource_id=resource_id,
                detail=detail,
            )
        )
