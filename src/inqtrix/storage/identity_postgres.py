"""Postgres-backed identity repositories and audit sink.

One backend object implements every read port the
:class:`~inqtrix.auth.permissions.PermissionService` consumes plus the
audit sink, mirroring the shape of
:class:`~inqtrix.auth.identity_memory.MemoryIdentityStore` so the
service is wired identically in both modes.

Every operation runs inside :func:`~inqtrix.storage.db.tenant_session`
— one transaction, restricted application role, transaction-local
tenant GUC. The explicit ``tenant_id`` predicates in the queries are
layer 1; row-level security underneath is layer 2 catching the bugs.
"""

from __future__ import annotations

import uuid
from contextlib import AbstractAsyncContextManager
from typing import Sequence

from sqlalchemy import delete, func, insert, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from inqtrix.auth.permissions import (
    AuditEntry,
    SharePermission,
    SubjectRef,
    WorkspaceRole,
    highest_grant,
)
from inqtrix.storage.db import tenant_session
from inqtrix.storage.identity_orm import (
    audit_log,
    group_members,
    resource_shares,
    workspace_members,
    workspaces,
)


def _as_uuid(value: str) -> uuid.UUID | None:
    """Parse a client-supplied id against the UUID column type.

    A malformed id can never match a row, so it maps to ``None`` —
    the same answer a non-member gets. Letting the string reach
    asyncpg instead would raise a DataError, turning "hidden" into a
    distinguishable 500 (an existence oracle).
    """
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


class PostgresIdentityBackend:
    """Identity read ports and audit sink over the identity schema.

    Args:
        session_factory: Factory from
            :func:`inqtrix.storage.db.build_session_factory`.
        app_role: Restricted Postgres role for
            :func:`~inqtrix.storage.db.tenant_session` (see
            ``StorageSettings.app_role``).
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        app_role: str,
    ) -> None:
        self._session_factory = session_factory
        self._app_role = app_role

    def _session(
        self, tenant_id: str
    ) -> AbstractAsyncContextManager["AsyncSession"]:
        """One tenant transaction with this backend's app role bound."""
        return tenant_session(
            self._session_factory, tenant_id=tenant_id, app_role=self._app_role
        )

    # ------------------------------------------------------------- #
    # MembershipRepository
    # ------------------------------------------------------------- #

    async def workspace_ids_for(
        self, *, tenant_id: str, sub: str
    ) -> tuple[str, ...]:
        """All workspace ids *sub* is a member of within *tenant_id*."""
        async with self._session(tenant_id) as session:
            rows = await session.execute(
                select(workspace_members.c.workspace_id).where(
                    workspace_members.c.tenant_id == tenant_id,
                    workspace_members.c.sub == sub,
                )
            )
            return tuple(str(workspace_id) for (workspace_id,) in rows)

    async def role_in_workspace(
        self, *, tenant_id: str, sub: str, workspace_id: str
    ) -> WorkspaceRole | None:
        """The member's role, or ``None`` for non-members, unknown
        workspaces, and malformed workspace ids alike (existence and
        id validity stay hidden)."""
        workspace_uuid = _as_uuid(workspace_id)
        if workspace_uuid is None:
            return None
        async with self._session(tenant_id) as session:
            row = await session.execute(
                select(workspace_members.c.role).where(
                    workspace_members.c.tenant_id == tenant_id,
                    workspace_members.c.sub == sub,
                    workspace_members.c.workspace_id == workspace_uuid,
                )
            )
            value = row.scalar_one_or_none()
            return WorkspaceRole(value) if value is not None else None

    # ------------------------------------------------------------- #
    # ShareAdminRepository (write/listing surface)
    # ------------------------------------------------------------- #

    async def create_share(
        self,
        *,
        tenant_id: str,
        subject_type: str,
        subject_id: str,
        resource_type: str,
        resource_id: str,
        permission: "SharePermission",
        granted_by_sub: str,
    ) -> "ShareRecord":
        """Grant or re-grant in one transaction.

        The partial unique index allows one ACTIVE row per tuple, so a
        re-grant soft-revokes the existing row first — the caller's
        latest intent wins, the history stays auditable. A re-grant carries
        the prior row's ``accepted_at`` forward, so changing the permission on
        an already-accepted share keeps access live (a brand-new grant has no
        prior row and starts pending).
        """
        from inqtrix.auth.shares import ShareRecord

        share_id = uuid.uuid4()
        async with self._session(tenant_id) as session:
            prior = (
                await session.execute(
                    update(resource_shares)
                    .where(
                        resource_shares.c.tenant_id == tenant_id,
                        resource_shares.c.subject_type == subject_type,
                        resource_shares.c.subject_id == subject_id,
                        resource_shares.c.resource_type == resource_type,
                        resource_shares.c.resource_id == resource_id,
                        resource_shares.c.revoked_at.is_(None),
                    )
                    .values(revoked_at=func.now(), revoked_by_sub=granted_by_sub)
                    .returning(resource_shares.c.accepted_at)
                )
            ).first()
            carried_accepted_at = prior.accepted_at if prior is not None else None
            await session.execute(
                insert(resource_shares).values(
                    id=share_id,
                    tenant_id=tenant_id,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    permission=permission.value,
                    granted_by_sub=granted_by_sub,
                    accepted_at=carried_accepted_at,
                )
            )
        import time as _time

        return ShareRecord(
            id=str(share_id),
            tenant_id=tenant_id,
            subject_type=subject_type,
            subject_id=subject_id,
            resource_type=resource_type,
            resource_id=resource_id,
            permission=permission,
            granted_by_sub=granted_by_sub,
            created_at=_time.time(),
            accepted_at=(
                carried_accepted_at.timestamp()
                if carried_accepted_at is not None
                else None
            ),
        )

    async def get_share(
        self, *, tenant_id: str, share_id: str
    ) -> "ShareRecord | None":
        share_uuid = _as_uuid(share_id)
        if share_uuid is None:
            return None
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(resource_shares).where(
                        resource_shares.c.tenant_id == tenant_id,
                        resource_shares.c.id == share_uuid,
                        resource_shares.c.revoked_at.is_(None),
                    )
                )
            ).first()
        return self._share_record(row) if row is not None else None

    async def revoke_share_by_id(
        self, *, tenant_id: str, share_id: str, revoked_by_sub: str
    ) -> "ShareRecord | None":
        share_uuid = _as_uuid(share_id)
        if share_uuid is None:
            return None
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    update(resource_shares)
                    .where(
                        resource_shares.c.tenant_id == tenant_id,
                        resource_shares.c.id == share_uuid,
                        resource_shares.c.revoked_at.is_(None),
                    )
                    .values(
                        revoked_at=func.now(),
                        revoked_by_sub=revoked_by_sub,
                    )
                    .returning(resource_shares)
                )
            ).first()
        return self._share_record(row) if row is not None else None

    async def accept_share_by_id(
        self, *, tenant_id: str, share_id: str, subject_sub: str
    ) -> "ShareRecord | None":
        """Flip one pending share to accepted; returns it, or ``None``.

        Guarded in the predicate: the row must be active, still pending
        (``accepted_at IS NULL``), and addressed to *subject_sub* — so a
        foreign recipient, an already-accepted share, and a missing one all
        update zero rows and return ``None`` (the surface's 404 rule).

        No ``subject_type`` guard is needed (unlike :meth:`recipient_drop`'s
        explicit one): *subject_sub* is a user ``sub``, and a group share's
        ``subject_id`` is a group id, so the ``subject_id`` match already
        excludes group rows structurally.
        """
        share_uuid = _as_uuid(share_id)
        if share_uuid is None:
            return None
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    update(resource_shares)
                    .where(
                        resource_shares.c.tenant_id == tenant_id,
                        resource_shares.c.id == share_uuid,
                        resource_shares.c.subject_id == subject_sub,
                        resource_shares.c.revoked_at.is_(None),
                        resource_shares.c.accepted_at.is_(None),
                    )
                    .values(accepted_at=func.now())
                    .returning(resource_shares)
                )
            ).first()
        return self._share_record(row) if row is not None else None

    async def revoke_shares_for_resource(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        revoked_by_sub: str,
    ) -> int:
        """Soft-revoke every active share on one resource (cleanup)."""
        async with self._session(tenant_id) as session:
            result = await session.execute(
                update(resource_shares)
                .where(
                    resource_shares.c.tenant_id == tenant_id,
                    resource_shares.c.resource_type == resource_type,
                    resource_shares.c.resource_id == resource_id,
                    resource_shares.c.revoked_at.is_(None),
                )
                .values(
                    revoked_at=func.now(),
                    revoked_by_sub=revoked_by_sub,
                )
            )
        return int(result.rowcount or 0)

    async def list_shares_for_resource(
        self, *, tenant_id: str, resource_type: str, resource_id: str
    ) -> tuple["ShareRecord", ...]:
        async with self._session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(resource_shares)
                    .where(
                        resource_shares.c.tenant_id == tenant_id,
                        resource_shares.c.resource_type == resource_type,
                        resource_shares.c.resource_id == resource_id,
                        resource_shares.c.revoked_at.is_(None),
                    )
                    .order_by(resource_shares.c.created_at)
                )
            ).all()
        return tuple(self._share_record(row) for row in rows)

    async def inbox_for_subjects(
        self, *, tenant_id: str, subjects: Sequence[SubjectRef]
    ) -> tuple["ShareRecord", ...]:
        """Active (pending + accepted) shares to the subjects, all kinds.

        The recipient inbox source — unlike :meth:`shares_for_subjects` it
        keeps pending rows (so they can be consented to) and spans every
        resource kind in one query.
        """
        if not subjects:
            return ()
        pairs = [(s.subject_type, s.subject_id) for s in subjects]
        async with self._session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(resource_shares)
                    .where(
                        resource_shares.c.tenant_id == tenant_id,
                        resource_shares.c.revoked_at.is_(None),
                        tuple_(
                            resource_shares.c.subject_type,
                            resource_shares.c.subject_id,
                        ).in_(pairs),
                    )
                    .order_by(resource_shares.c.created_at)
                )
            ).all()
        return tuple(self._share_record(row) for row in rows)

    async def outgoing_shares_for_grantor(
        self, *, tenant_id: str, grantor_sub: str
    ) -> tuple["ShareRecord", ...]:
        """Active shares *grantor_sub* granted, all kinds, oldest first."""
        async with self._session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(resource_shares)
                    .where(
                        resource_shares.c.tenant_id == tenant_id,
                        resource_shares.c.granted_by_sub == grantor_sub,
                        resource_shares.c.revoked_at.is_(None),
                    )
                    .order_by(resource_shares.c.created_at)
                )
            ).all()
        return tuple(self._share_record(row) for row in rows)

    async def shares_for_subjects(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        subjects: Sequence[SubjectRef],
    ) -> dict[str, "ShareRecord"]:
        if not subjects:
            return {}
        pairs = [(s.subject_type, s.subject_id) for s in subjects]
        async with self._session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(resource_shares).where(
                        resource_shares.c.tenant_id == tenant_id,
                        resource_shares.c.resource_type == resource_type,
                        resource_shares.c.revoked_at.is_(None),
                        resource_shares.c.accepted_at.isnot(None),
                        tuple_(
                            resource_shares.c.subject_type,
                            resource_shares.c.subject_id,
                        ).in_(pairs),
                    )
                )
            ).all()
        best: dict[str, "ShareRecord"] = {}
        for row in rows:
            record = self._share_record(row)
            current = best.get(record.resource_id)
            if current is None or record.permission.at_least(
                current.permission
            ):
                best[record.resource_id] = record
        return best

    async def share_counts_for_resources(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_ids: Sequence[str],
    ) -> dict[str, int]:
        if not resource_ids:
            return {}
        async with self._session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(
                        resource_shares.c.resource_id,
                        func.count(resource_shares.c.id),
                    )
                    .where(
                        resource_shares.c.tenant_id == tenant_id,
                        resource_shares.c.resource_type == resource_type,
                        resource_shares.c.resource_id.in_(list(resource_ids)),
                        resource_shares.c.revoked_at.is_(None),
                    )
                    .group_by(resource_shares.c.resource_id)
                )
            ).all()
        return {resource_id: count for resource_id, count in rows}

    @staticmethod
    def _share_record(row) -> "ShareRecord":
        from inqtrix.auth.shares import ShareRecord

        return ShareRecord(
            id=str(row.id),
            tenant_id=row.tenant_id,
            subject_type=row.subject_type,
            subject_id=row.subject_id,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            permission=SharePermission(row.permission),
            granted_by_sub=row.granted_by_sub,
            created_at=(
                row.created_at.timestamp()
                if row.created_at is not None
                else 0.0
            ),
            accepted_at=(
                row.accepted_at.timestamp()
                if row.accepted_at is not None
                else None
            ),
        )

    # ------------------------------------------------------------- #
    # Workspace admin (creation + listing)
    # ------------------------------------------------------------- #

    async def create_workspace(
        self, *, tenant_id: str, name: str, created_by_sub: str
    ) -> tuple[str, str]:
        """Create one workspace with *created_by_sub* as its OWNER, atomically.

        The instance-admin workspace surface (``/v1/admin/workspaces``) and
        the self-serve ``POST /v1/workspaces`` both land here; the workspace
        row and the creator's OWNER membership are written in one transaction.
        """
        workspace_id = uuid.uuid4()
        async with self._session(tenant_id) as session:
            await session.execute(
                insert(workspaces).values(
                    id=workspace_id,
                    tenant_id=tenant_id,
                    name=name,
                    created_by_sub=created_by_sub,
                )
            )
            await session.execute(
                insert(workspace_members).values(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    sub=created_by_sub,
                    role=WorkspaceRole.OWNER.value,
                )
            )
        return str(workspace_id), name

    async def list_workspaces_for(
        self, *, tenant_id: str, sub: str
    ) -> tuple[tuple[str, str, WorkspaceRole], ...]:
        """``(id, name, role)`` per membership, name-sorted."""
        async with self._session(tenant_id) as session:
            rows = await session.execute(
                select(
                    workspaces.c.id,
                    workspaces.c.name,
                    workspace_members.c.role,
                )
                .join(
                    workspace_members,
                    workspace_members.c.workspace_id == workspaces.c.id,
                )
                .where(
                    workspaces.c.tenant_id == tenant_id,
                    workspace_members.c.sub == sub,
                )
                .order_by(workspaces.c.name)
            )
            return tuple(
                (str(workspace_id), name, WorkspaceRole(role))
                for workspace_id, name, role in rows
            )

    # ------------------------------------------------------------- #
    # MembershipAdminRepository (workspace + membership administration)
    # ------------------------------------------------------------- #

    async def list_all_workspaces(
        self, *, tenant_id: str
    ) -> tuple[tuple[str, str, str, int], ...]:
        """``(id, name, created_by_sub, member_count)`` per tenant workspace.

        Outer-join so a workspace with zero members (every member removed)
        still reports with ``member_count == 0`` rather than vanishing.
        """
        async with self._session(tenant_id) as session:
            rows = await session.execute(
                select(
                    workspaces.c.id,
                    workspaces.c.name,
                    workspaces.c.created_by_sub,
                    func.count(workspace_members.c.sub),
                )
                .select_from(
                    workspaces.outerjoin(
                        workspace_members,
                        workspace_members.c.workspace_id == workspaces.c.id,
                    )
                )
                .where(workspaces.c.tenant_id == tenant_id)
                .group_by(
                    workspaces.c.id,
                    workspaces.c.name,
                    workspaces.c.created_by_sub,
                )
                .order_by(workspaces.c.name)
            )
            return tuple(
                (str(workspace_id), name, created_by_sub, int(count))
                for workspace_id, name, created_by_sub, count in rows
            )

    async def rename_workspace(
        self, *, tenant_id: str, workspace_id: str, name: str
    ) -> bool:
        """Rename one workspace; ``False`` for unknown / malformed id."""
        workspace_uuid = _as_uuid(workspace_id)
        if workspace_uuid is None:
            return False
        async with self._session(tenant_id) as session:
            result = await session.execute(
                update(workspaces)
                .where(
                    workspaces.c.tenant_id == tenant_id,
                    workspaces.c.id == workspace_uuid,
                )
                .values(name=name)
            )
        return int(result.rowcount or 0) > 0

    async def delete_workspace(
        self, *, tenant_id: str, workspace_id: str
    ) -> bool:
        """Delete the workspace; memberships cascade (ORM ON DELETE CASCADE)."""
        workspace_uuid = _as_uuid(workspace_id)
        if workspace_uuid is None:
            return False
        async with self._session(tenant_id) as session:
            result = await session.execute(
                delete(workspaces).where(
                    workspaces.c.tenant_id == tenant_id,
                    workspaces.c.id == workspace_uuid,
                )
            )
        return int(result.rowcount or 0) > 0

    async def _workspace_exists(
        self, session: "AsyncSession", *, tenant_id: str, workspace_uuid: uuid.UUID
    ) -> bool:
        row = await session.execute(
            select(workspaces.c.id).where(
                workspaces.c.tenant_id == tenant_id,
                workspaces.c.id == workspace_uuid,
            )
        )
        return row.scalar_one_or_none() is not None

    async def list_members(
        self, *, tenant_id: str, workspace_id: str
    ) -> tuple[tuple[str, WorkspaceRole], ...] | None:
        """``(sub, role)`` per member, sub-sorted, or ``None`` when absent."""
        workspace_uuid = _as_uuid(workspace_id)
        if workspace_uuid is None:
            return None
        async with self._session(tenant_id) as session:
            if not await self._workspace_exists(
                session, tenant_id=tenant_id, workspace_uuid=workspace_uuid
            ):
                return None
            rows = await session.execute(
                select(
                    workspace_members.c.sub,
                    workspace_members.c.role,
                )
                .where(
                    workspace_members.c.tenant_id == tenant_id,
                    workspace_members.c.workspace_id == workspace_uuid,
                )
                .order_by(workspace_members.c.sub)
            )
            return tuple((sub, WorkspaceRole(role)) for sub, role in rows)

    async def assign_member(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        sub: str,
        role: WorkspaceRole,
    ) -> bool:
        """Upsert one membership at the exact role; ``False`` when absent.

        Existence is checked first so an unknown workspace returns ``False``
        rather than surfacing the foreign-key violation as a 500.
        """
        workspace_uuid = _as_uuid(workspace_id)
        if workspace_uuid is None:
            return False
        async with self._session(tenant_id) as session:
            if not await self._workspace_exists(
                session, tenant_id=tenant_id, workspace_uuid=workspace_uuid
            ):
                return False
            await session.execute(
                pg_insert(workspace_members)
                .values(
                    tenant_id=tenant_id,
                    workspace_id=workspace_uuid,
                    sub=sub,
                    role=role.value,
                )
                .on_conflict_do_update(
                    index_elements=[
                        workspace_members.c.workspace_id,
                        workspace_members.c.sub,
                    ],
                    set_={"role": role.value},
                )
            )
        return True

    async def remove_member(
        self, *, tenant_id: str, workspace_id: str, sub: str
    ) -> bool:
        """Remove one membership; ``False`` when not a member / unknown."""
        workspace_uuid = _as_uuid(workspace_id)
        if workspace_uuid is None:
            return False
        async with self._session(tenant_id) as session:
            result = await session.execute(
                delete(workspace_members).where(
                    workspace_members.c.tenant_id == tenant_id,
                    workspace_members.c.workspace_id == workspace_uuid,
                    workspace_members.c.sub == sub,
                )
            )
        return int(result.rowcount or 0) > 0

    # ------------------------------------------------------------- #
    # GroupRepository
    # ------------------------------------------------------------- #

    async def group_ids_for(self, *, tenant_id: str, sub: str) -> tuple[str, ...]:
        """All group ids *sub* belongs to within *tenant_id*."""
        async with self._session(tenant_id) as session:
            rows = await session.execute(
                select(group_members.c.group_id).where(
                    group_members.c.tenant_id == tenant_id,
                    group_members.c.sub == sub,
                )
            )
            return tuple(str(group_id) for (group_id,) in rows)

    # ------------------------------------------------------------- #
    # ShareRepository
    # ------------------------------------------------------------- #

    async def permission_for(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        subjects: Sequence[SubjectRef],
    ) -> SharePermission | None:
        """Highest active, ACCEPTED grant any of *subjects* holds.

        The max-rank reduction happens in Python against the
        application ordering — the database stores permissions as
        plain text precisely so it never holds a second ordering
        authority. Pending shares (``accepted_at IS NULL``) are excluded:
        consent is the single gate, enforced here so every ``can``/visibility
        path inherits it without its own branch.
        """
        if not subjects:
            return None
        subject_pairs = [(s.subject_type, s.subject_id) for s in subjects]
        async with self._session(tenant_id) as session:
            rows = await session.execute(
                select(
                    resource_shares.c.subject_type,
                    resource_shares.c.subject_id,
                    resource_shares.c.permission,
                ).where(
                    resource_shares.c.tenant_id == tenant_id,
                    resource_shares.c.resource_type == resource_type,
                    resource_shares.c.resource_id == resource_id,
                    resource_shares.c.revoked_at.is_(None),
                    resource_shares.c.accepted_at.isnot(None),
                )
            )
            return highest_grant(
                SharePermission(permission)
                for subject_type, subject_id, permission in rows
                if (subject_type, subject_id) in subject_pairs
            )

    # ------------------------------------------------------------- #
    # AuditSink
    # ------------------------------------------------------------- #

    async def record(self, entry: AuditEntry) -> None:
        """Append one audit fact (INSERT-only grants on the table)."""
        async with self._session(entry.tenant_id) as session:
            await session.execute(
                insert(audit_log).values(
                    tenant_id=entry.tenant_id,
                    actor_sub=entry.actor_sub,
                    action=entry.action,
                    resource_type=entry.resource_type,
                    resource_id=entry.resource_id,
                    detail=dict(entry.detail),
                )
            )
