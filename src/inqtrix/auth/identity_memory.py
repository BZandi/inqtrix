"""In-memory identity backend: the no-infrastructure default.

Implements every read port consumed by
:class:`~inqtrix.auth.permissions.PermissionService` plus the audit
sink against plain dictionaries. This is the deployment default
(``INQTRIX_STORAGE_BACKEND=memory``): an empty store means scoped
principals have no memberships — they see nothing until granted —
while the legacy unscoped principals never consult it at all.

The mutation helpers (``add_workspace`` etc.) are the seam tests and
future admin surfaces use to arrange facts; the Postgres backend gets
its facts from the identity schema instead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Sequence

from inqtrix.auth.permissions import (
    AuditEntry,
    SharePermission,
    SubjectRef,
    WorkspaceRole,
    highest_grant,
)


@dataclass(frozen=True)
class MemoryWorkspace:
    """One workspace fact in the in-memory identity store."""

    workspace_id: str
    tenant_id: str
    name: str
    created_by_sub: str = ""
    """The creator (admin) — surfaced by the admin overview; mirrors the
    Postgres ``workspaces.created_by_sub`` column. Empty for workspaces
    arranged through the bare ``add_workspace`` test seam."""


@dataclass(frozen=True)
class _ShareKey:
    """Identity of one active share grant."""

    tenant_id: str
    subject_type: str
    subject_id: str
    resource_type: str
    resource_id: str


@dataclass
class MemoryIdentityStore:
    """Dictionary-backed identity facts with a coarse lock.

    One instance implements all four ports
    (:class:`~inqtrix.auth.permissions.MembershipRepository`,
    ``GroupRepository``, ``ShareRepository``, ``AuditSink``) so tests
    wire a single object. Thread-safe because run workers and request
    handlers live on different threads.
    """

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _workspaces: dict[str, MemoryWorkspace] = field(default_factory=dict)
    _members: dict[tuple[str, str], WorkspaceRole] = field(default_factory=dict)
    """``(workspace_id, sub) -> role``"""
    _groups: dict[str, set[str]] = field(default_factory=dict)
    """``group_id -> {sub, ...}`` (tenant captured in group id scope)."""
    _group_tenants: dict[str, str] = field(default_factory=dict)
    _shares: dict[_ShareKey, SharePermission] = field(default_factory=dict)
    _share_records: dict[str, "ShareRecord"] = field(default_factory=dict)
    """Full active share rows keyed by share id (the admin surface);
    ``_shares`` stays the permission_for fast path — both maps are
    written together by every mutation."""
    audit_entries: list[AuditEntry] = field(default_factory=list)
    """Recorded audit facts, oldest first (assert target in tests)."""

    # ------------------------------------------------------------- #
    # Mutation seam (tests / future admin surface)
    # ------------------------------------------------------------- #

    def add_workspace(
        self, workspace_id: str, *, tenant_id: str = "default", name: str = ""
    ) -> None:
        """Create one workspace fact."""
        with self._lock:
            self._workspaces[workspace_id] = MemoryWorkspace(
                workspace_id=workspace_id,
                tenant_id=tenant_id,
                name=name or workspace_id,
            )

    async def create_workspace(
        self, *, tenant_id: str, name: str, created_by_sub: str
    ) -> tuple[str, str]:
        """Create one workspace with its creator as OWNER (atomic here
        by the lock; the Postgres backend uses one transaction)."""
        import uuid as _uuid

        workspace_id = str(_uuid.uuid4())
        with self._lock:
            self._workspaces[workspace_id] = MemoryWorkspace(
                workspace_id=workspace_id,
                tenant_id=tenant_id,
                name=name,
                created_by_sub=created_by_sub,
            )
            self._members[(workspace_id, created_by_sub)] = (
                WorkspaceRole.OWNER
            )
        return workspace_id, name

    async def list_workspaces_for(
        self, *, tenant_id: str, sub: str
    ) -> tuple[tuple[str, str, WorkspaceRole], ...]:
        """``(id, name, role)`` per membership, name-sorted."""
        with self._lock:
            rows = [
                (
                    workspace_id,
                    self._workspaces[workspace_id].name,
                    role,
                )
                for (workspace_id, member_sub), role in self._members.items()
                if member_sub == sub
                and self._workspaces.get(workspace_id) is not None
                and self._workspaces[workspace_id].tenant_id == tenant_id
            ]
        return tuple(sorted(rows, key=lambda row: row[1]))

    def add_member(
        self, workspace_id: str, sub: str, role: WorkspaceRole
    ) -> None:
        """Add or update one workspace membership.

        The synchronous arrange/invitation seam (used by
        :class:`~inqtrix.auth.invitations.MemoryInvitationStore`). The async
        :meth:`assign_member` is the admin-surface counterpart.

        Raises:
            KeyError: When the workspace was never created — silently
                materializing it would hide arrangement bugs in tests.
        """
        with self._lock:
            if workspace_id not in self._workspaces:
                raise KeyError(f"unknown workspace: {workspace_id}")
            self._members[(workspace_id, sub)] = role

    # ------------------------------------------------------------- #
    # MembershipAdminRepository (workspace + membership administration)
    # ------------------------------------------------------------- #

    async def list_all_workspaces(
        self, *, tenant_id: str
    ) -> tuple[tuple[str, str, str, int], ...]:
        """``(id, name, created_by_sub, member_count)`` per tenant workspace."""
        with self._lock:
            rows = [
                (
                    workspace.workspace_id,
                    workspace.name,
                    workspace.created_by_sub,
                    sum(
                        1
                        for (member_workspace_id, _sub) in self._members
                        if member_workspace_id == workspace.workspace_id
                    ),
                )
                for workspace in self._workspaces.values()
                if workspace.tenant_id == tenant_id
            ]
        return tuple(sorted(rows, key=lambda row: row[1]))

    async def rename_workspace(
        self, *, tenant_id: str, workspace_id: str, name: str
    ) -> bool:
        """Rename one workspace; ``False`` for unknown / foreign-tenant."""
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None or workspace.tenant_id != tenant_id:
                return False
            self._workspaces[workspace_id] = replace(workspace, name=name)
            return True

    async def delete_workspace(
        self, *, tenant_id: str, workspace_id: str
    ) -> bool:
        """Delete the workspace and cascade its memberships; ``False`` when
        absent (mirrors the Postgres ON DELETE CASCADE)."""
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None or workspace.tenant_id != tenant_id:
                return False
            del self._workspaces[workspace_id]
            for key in [
                member_key
                for member_key in self._members
                if member_key[0] == workspace_id
            ]:
                del self._members[key]
            return True

    async def list_members(
        self, *, tenant_id: str, workspace_id: str
    ) -> tuple[tuple[str, WorkspaceRole], ...] | None:
        """``(sub, role)`` per member, sub-sorted, or ``None`` when absent."""
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None or workspace.tenant_id != tenant_id:
                return None
            rows = [
                (sub, role)
                for (member_workspace_id, sub), role in self._members.items()
                if member_workspace_id == workspace_id
            ]
        return tuple(sorted(rows, key=lambda row: row[0]))

    async def assign_member(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        sub: str,
        role: WorkspaceRole,
    ) -> bool:
        """Upsert one membership at the exact role; ``False`` when absent."""
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None or workspace.tenant_id != tenant_id:
                return False
            self._members[(workspace_id, sub)] = role
            return True

    async def remove_member(
        self, *, tenant_id: str, workspace_id: str, sub: str
    ) -> bool:
        """Remove one membership; ``False`` when not a member / unknown."""
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None or workspace.tenant_id != tenant_id:
                return False
            if (workspace_id, sub) not in self._members:
                return False
            del self._members[(workspace_id, sub)]
            return True

    def add_group(
        self, group_id: str, members: Sequence[str], *, tenant_id: str = "default"
    ) -> None:
        """Create one group with its member subs."""
        with self._lock:
            self._groups[group_id] = set(members)
            self._group_tenants[group_id] = tenant_id

    def _write_share_locked(
        self,
        *,
        tenant_id: str,
        subject_type: str,
        subject_id: str,
        resource_type: str,
        resource_id: str,
        permission: SharePermission,
        granted_by_sub: str,
        accepted_at: float | None,
    ) -> "ShareRecord":
        """Replace the active row for a tuple and mirror the access fast path.

        The lock must be held. ``_shares`` (the ``permission_for`` fast path)
        holds ONLY accepted grants: a pending write drops any stale fast-path
        entry, an accepted write sets it — so consent is enforced without a
        second filter in ``permission_for`` (Wurzel, nicht Symptom). One
        active record per tuple is preserved (re-grant replaces).
        """
        import time as _time
        import uuid as _uuid

        from inqtrix.auth.shares import ShareRecord

        key = _ShareKey(
            tenant_id=tenant_id,
            subject_type=subject_type,
            subject_id=subject_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        if accepted_at is not None:
            self._shares[key] = permission
        else:
            self._shares.pop(key, None)
        self._share_records = {
            share_id: record
            for share_id, record in self._share_records.items()
            if not (
                record.tenant_id == tenant_id
                and record.subject_type == subject_type
                and record.subject_id == subject_id
                and record.resource_type == resource_type
                and record.resource_id == resource_id
            )
        }
        share_id = str(_uuid.uuid4())
        record = ShareRecord(
            id=share_id,
            tenant_id=tenant_id,
            subject_type=subject_type,
            subject_id=subject_id,
            resource_type=resource_type,
            resource_id=resource_id,
            permission=permission,
            granted_by_sub=granted_by_sub,
            created_at=_time.time(),
            accepted_at=accepted_at,
        )
        self._share_records[share_id] = record
        return record

    def add_share(
        self,
        *,
        subject_type: str,
        subject_id: str,
        resource_type: str,
        resource_id: str,
        permission: SharePermission,
        tenant_id: str = "default",
        granted_by_sub: str = "seed",
        accepted: bool = True,
    ) -> None:
        """Grant one share tuple (subject x resource -> permission).

        The synchronous seed/arrange seam. Defaults to ``accepted=True`` so
        arranged fixtures grant access immediately (mirroring the migration
        that backfills pre-existing rows as accepted); pass ``accepted=False``
        to seed a pending invitation awaiting consent.
        """
        import time as _time

        with self._lock:
            self._write_share_locked(
                tenant_id=tenant_id,
                subject_type=subject_type,
                subject_id=subject_id,
                resource_type=resource_type,
                resource_id=resource_id,
                permission=permission,
                granted_by_sub=granted_by_sub,
                accepted_at=_time.time() if accepted else None,
            )

    def revoke_share(
        self,
        *,
        subject_type: str,
        subject_id: str,
        resource_type: str,
        resource_id: str,
        tenant_id: str = "default",
    ) -> None:
        """Remove one share tuple; missing grants are a no-op."""
        key = _ShareKey(
            tenant_id=tenant_id,
            subject_type=subject_type,
            subject_id=subject_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        with self._lock:
            self._shares.pop(key, None)
            self._share_records = {
                share_id: record
                for share_id, record in self._share_records.items()
                if not (
                    record.tenant_id == tenant_id
                    and record.subject_type == subject_type
                    and record.subject_id == subject_id
                    and record.resource_type == resource_type
                    and record.resource_id == resource_id
                )
            }

    # ------------------------------------------------------------- #
    # MembershipRepository
    # ------------------------------------------------------------- #

    async def workspace_ids_for(
        self, *, tenant_id: str, sub: str
    ) -> tuple[str, ...]:
        """All workspace ids *sub* belongs to within *tenant_id*."""
        with self._lock:
            return tuple(
                workspace_id
                for (workspace_id, member_sub), _role in self._members.items()
                if member_sub == sub
                and (workspace := self._workspaces.get(workspace_id)) is not None
                and workspace.tenant_id == tenant_id
            )

    async def role_in_workspace(
        self, *, tenant_id: str, sub: str, workspace_id: str
    ) -> WorkspaceRole | None:
        """The member's role, or ``None`` for non-members and unknown
        workspaces alike."""
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None or workspace.tenant_id != tenant_id:
                return None
            return self._members.get((workspace_id, sub))

    # ------------------------------------------------------------- #
    # GroupRepository
    # ------------------------------------------------------------- #

    async def group_ids_for(self, *, tenant_id: str, sub: str) -> tuple[str, ...]:
        """All group ids *sub* belongs to within *tenant_id*."""
        with self._lock:
            return tuple(
                group_id
                for group_id, members in self._groups.items()
                if sub in members and self._group_tenants[group_id] == tenant_id
            )

    # ------------------------------------------------------------- #
    # ShareRepository
    # ------------------------------------------------------------- #

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
        permission: SharePermission,
        granted_by_sub: str,
    ) -> "ShareRecord":
        """Grant or re-grant; the latest intent replaces the tuple.

        A re-grant carries the prior active row's ``accepted_at`` forward (a
        permission change on an accepted share keeps access live); a brand-new
        grant has no prior row and starts pending — the consent gate.
        """
        with self._lock:
            prior_accepted_at: float | None = None
            for record in self._share_records.values():
                if (
                    record.tenant_id == tenant_id
                    and record.subject_type == subject_type
                    and record.subject_id == subject_id
                    and record.resource_type == resource_type
                    and record.resource_id == resource_id
                ):
                    prior_accepted_at = record.accepted_at
                    break
            return self._write_share_locked(
                tenant_id=tenant_id,
                subject_type=subject_type,
                subject_id=subject_id,
                resource_type=resource_type,
                resource_id=resource_id,
                permission=permission,
                granted_by_sub=granted_by_sub,
                accepted_at=prior_accepted_at,
            )

    async def accept_share_by_id(
        self, *, tenant_id: str, share_id: str, subject_sub: str
    ) -> "ShareRecord | None":
        """Flip one pending share to accepted; returns it, or ``None``.

        Mirrors the Postgres guard: active, still pending, addressed to
        *subject_sub*. Accepting moves the grant into the ``_shares`` access
        fast path so ``permission_for`` starts honouring it. No ``subject_type``
        guard is needed (unlike ``recipient_drop``): *subject_sub* is a user
        ``sub`` and a group share's ``subject_id`` is a group id, so the
        ``subject_id`` match already excludes group rows.
        """
        import time as _time
        from dataclasses import replace as _replace

        with self._lock:
            record = self._share_records.get(share_id)
            if (
                record is None
                or record.tenant_id != tenant_id
                or record.subject_id != subject_sub
                or record.accepted_at is not None
            ):
                return None
            accepted = _replace(record, accepted_at=_time.time())
            self._share_records[share_id] = accepted
            self._shares[
                _ShareKey(
                    tenant_id=record.tenant_id,
                    subject_type=record.subject_type,
                    subject_id=record.subject_id,
                    resource_type=record.resource_type,
                    resource_id=record.resource_id,
                )
            ] = record.permission
            return accepted

    async def get_share(
        self, *, tenant_id: str, share_id: str
    ) -> "ShareRecord | None":
        with self._lock:
            record = self._share_records.get(share_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return record

    async def revoke_share_by_id(
        self, *, tenant_id: str, share_id: str, revoked_by_sub: str
    ) -> "ShareRecord | None":
        # The memory store is the volatile no-infra default: it hard-deletes
        # the row where Postgres soft-revokes (sets revoked_at/revoked_by_sub
        # for the durable history). The observable contract is identical —
        # every listing/visibility method returns active-only on both — and
        # the revocation FACT is still captured by the audit sink in both
        # backends. Hard-delete is also why create_share's carry-forward scan
        # below can match by tuple without a revoked_at filter: a revoked row
        # is simply gone, never a stale tombstone to skip.
        with self._lock:
            record = self._share_records.get(share_id)
            if record is None or record.tenant_id != tenant_id:
                return None
            del self._share_records[share_id]
            self._shares.pop(
                _ShareKey(
                    tenant_id=record.tenant_id,
                    subject_type=record.subject_type,
                    subject_id=record.subject_id,
                    resource_type=record.resource_type,
                    resource_id=record.resource_id,
                ),
                None,
            )
            return record

    async def revoke_shares_for_resource(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        revoked_by_sub: str,
    ) -> int:
        """Drop every active share on one resource (deletion cleanup)."""
        with self._lock:
            doomed = [
                share_id
                for share_id, record in self._share_records.items()
                if record.tenant_id == tenant_id
                and record.resource_type == resource_type
                and record.resource_id == resource_id
            ]
            for share_id in doomed:
                record = self._share_records.pop(share_id)
                self._shares.pop(
                    _ShareKey(
                        tenant_id=record.tenant_id,
                        subject_type=record.subject_type,
                        subject_id=record.subject_id,
                        resource_type=record.resource_type,
                        resource_id=record.resource_id,
                    ),
                    None,
                )
            return len(doomed)

    async def list_shares_for_resource(
        self, *, tenant_id: str, resource_type: str, resource_id: str
    ) -> tuple["ShareRecord", ...]:
        with self._lock:
            rows = [
                record
                for record in self._share_records.values()
                if record.tenant_id == tenant_id
                and record.resource_type == resource_type
                and record.resource_id == resource_id
            ]
        return tuple(sorted(rows, key=lambda record: record.created_at))

    async def shares_for_subjects(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        subjects: Sequence[SubjectRef],
    ) -> dict[str, "ShareRecord"]:
        wanted = {(s.subject_type, s.subject_id) for s in subjects}
        best: dict[str, "ShareRecord"] = {}
        with self._lock:
            for record in self._share_records.values():
                if (
                    record.tenant_id != tenant_id
                    or record.resource_type != resource_type
                    or record.accepted_at is None
                    or (record.subject_type, record.subject_id) not in wanted
                ):
                    continue
                current = best.get(record.resource_id)
                if current is None or record.permission.at_least(
                    current.permission
                ):
                    best[record.resource_id] = record
        return best

    async def inbox_for_subjects(
        self, *, tenant_id: str, subjects: Sequence[SubjectRef]
    ) -> tuple["ShareRecord", ...]:
        """Active (pending + accepted) shares to the subjects, all kinds.

        Keeps pending rows (unlike :meth:`shares_for_subjects`) so the
        recipient inbox can offer them for consent; oldest first.
        """
        wanted = {(s.subject_type, s.subject_id) for s in subjects}
        with self._lock:
            rows = [
                record
                for record in self._share_records.values()
                if record.tenant_id == tenant_id
                and (record.subject_type, record.subject_id) in wanted
            ]
        return tuple(sorted(rows, key=lambda record: record.created_at))

    async def outgoing_shares_for_grantor(
        self, *, tenant_id: str, grantor_sub: str
    ) -> tuple["ShareRecord", ...]:
        """Active shares *grantor_sub* granted, all kinds, oldest first."""
        with self._lock:
            rows = [
                record
                for record in self._share_records.values()
                if record.tenant_id == tenant_id
                and record.granted_by_sub == grantor_sub
            ]
        return tuple(sorted(rows, key=lambda record: record.created_at))

    async def share_counts_for_resources(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_ids: Sequence[str],
    ) -> dict[str, int]:
        wanted = set(resource_ids)
        counts: dict[str, int] = {}
        with self._lock:
            for record in self._share_records.values():
                if (
                    record.tenant_id == tenant_id
                    and record.resource_type == resource_type
                    and record.resource_id in wanted
                ):
                    counts[record.resource_id] = (
                        counts.get(record.resource_id, 0) + 1
                    )
        return counts

    async def permission_for(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        subjects: Sequence[SubjectRef],
    ) -> SharePermission | None:
        """Highest grant any of *subjects* holds on the resource."""
        with self._lock:
            return highest_grant(
                self._shares.get(
                    _ShareKey(
                        tenant_id=tenant_id,
                        subject_type=subject.subject_type,
                        subject_id=subject.subject_id,
                        resource_type=resource_type,
                        resource_id=resource_id,
                    )
                )
                for subject in subjects
            )

    # ------------------------------------------------------------- #
    # AuditSink
    # ------------------------------------------------------------- #

    async def record(self, entry: AuditEntry) -> None:
        """Append one audit fact (in-memory list, oldest first)."""
        with self._lock:
            self.audit_entries.append(entry)
