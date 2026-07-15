"""PostgreSQL implementation of the editor-collaboration persistence port."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, insert, select, text, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from inqtrix.auth.permissions import SharePermission
from inqtrix.project.editor_collaboration_ports import (
    CollaborationActivity,
    CollaborationChangeKind,
    CollaborationConflict,
    CollaborationDocumentNotFound,
    CollaborationDocumentState,
    CollaborationInstanceFenced,
    CollaborationInstanceLease,
    CollaborationLease,
    CollaborationLeaseInvalid,
    CollaborationLoadedState,
    CollaborationOpenPatch,
    CollaborationOpenPatchPage,
    CollaborationPolicyEvent,
    CollaborationPolicyPage,
    CollaborationPersistedCommand,
    CollaborationRateLimited,
    CollaborationSnapshot,
    CollaborationSnapshotCandidate,
    CollaborationSuggestionKind,
    CollaborationUpdate,
    CollaborationUpdateLookup,
    PersistCollaborationUpdate,
    PersistedCollaborationUpdate,
)
from inqtrix.storage.auth_orm import auth_sessions
from inqtrix.storage.db import tenant_session
from inqtrix.storage.editor_collaboration_orm import (
    editor_collaboration_instances,
    editor_collaboration_leases,
    editor_collaboration_snapshots,
    editor_collaboration_updates,
)
from inqtrix.storage.editor_orm import editor_documents
from inqtrix.storage.editor_patch_orm import editor_patches
from inqtrix.storage.identity_orm import resource_shares, users
from inqtrix.storage.user_event_orm import user_events
from inqtrix.storage.resource_access import (
    append_resource_effects,
    lock_resource_access,
    revoke_resource_shares,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = logging.getLogger("inqtrix")


def _require_sha256(value: str, *, field: str) -> None:
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _suggestion_kind_for_edit(
    edit: dict[str, Any],
) -> CollaborationSuggestionKind:
    """Infer the visible suggestion kind from an exact anchored edit."""
    find = edit.get("find")
    replacement = edit.get("text")
    position = edit.get("position")
    if position in {"append", "prepend"} or (not find and replacement):
        return "insertion"
    if find and not replacement:
        return "deletion"
    return "modification"


def _document_state(row: Any) -> CollaborationDocumentState:
    return CollaborationDocumentState(
        document_id=str(row.id),
        tenant_id=str(row.tenant_id),
        generation=int(row.collaboration_generation),
        schema_version=int(row.collaboration_schema_version),
        schema_hash=str(row.collaboration_schema_hash),
        persisted_sequence=int(row.persisted_sequence),
        projection_sequence=int(row.projection_sequence),
        content_markdown=str(row.content_markdown),
        projection_updated_at=row.projection_updated_at,
        owner_user_id=row.created_by_user_id,
        deleted_at=row.deleted_at,
    )


def _snapshot(row: Any) -> CollaborationSnapshot:
    return CollaborationSnapshot(
        document_id=str(row.document_id),
        tenant_id=str(row.tenant_id),
        generation=int(row.generation),
        covered_sequence=int(row.covered_sequence),
        state_update=bytes(row.state_update),
        state_vector=bytes(row.state_vector),
        state_hash=str(row.state_hash),
        projection_hash=str(row.projection_hash),
        schema_version=int(row.schema_version),
        schema_hash=str(row.schema_hash),
        created_at=float(row.created_at),
    )


def _collaboration_update(row: Any) -> CollaborationUpdate:
    return CollaborationUpdate(
        document_id=str(row.document_id),
        tenant_id=str(row.tenant_id),
        generation=int(row.generation),
        sequence=int(row.sequence),
        update_hash=str(row.update_hash),
        update_bytes=(
            bytes(row.update_bytes) if row.update_bytes is not None else None
        ),
        actor_user_id=row.actor_user_id,
        actor_kind=row.actor_kind,
        change_kind=row.change_kind,
        suggestion_ids=tuple(str(item) for item in (row.suggestion_ids or [])),
        command_id=row.command_id,
        created_at=float(row.created_at),
        payload_pruned_at=row.payload_pruned_at,
    )


def _lease(row: Any) -> CollaborationLease:
    return CollaborationLease(
        lease_id=row.lease_id,
        token_hash=str(row.token_hash),
        tenant_id=str(row.tenant_id),
        document_id=str(row.document_id),
        generation=int(row.generation),
        user_id=row.user_id,
        permission=row.permission,
        session_id=str(row.session_id),
        issued_at=float(row.issued_at),
        expires_at=float(row.expires_at),
        last_validated_at=float(row.validated_at or row.issued_at),
        rotation_command_id=row.rotation_command_id,
        rotated_from_lease_id=row.rotated_from_lease_id,
        revoked_at=row.revoked_at,
    )


def _lease_values(lease: CollaborationLease) -> dict[str, Any]:
    return {
        "lease_id": lease.lease_id,
        "tenant_id": lease.tenant_id,
        "token_hash": lease.token_hash,
        "document_id": lease.document_id,
        "generation": lease.generation,
        "user_id": lease.user_id,
        "permission": lease.permission,
        "session_id": lease.session_id,
        "issued_at": lease.issued_at,
        "expires_at": lease.expires_at,
        "validated_at": lease.last_validated_at,
        "revoked_at": lease.revoked_at,
        "rotation_command_id": lease.rotation_command_id,
        "rotated_from_lease_id": lease.rotated_from_lease_id,
    }


def _persisted_command(
    row: Any,
    patch_rows: list[Any],
) -> CollaborationPersistedCommand:
    patch_ids = tuple(sorted(str(item.patch_id) for item in patch_rows))
    if not patch_ids:
        raise CollaborationConflict("command_metadata_missing")
    decision = None
    if row.change_kind == "decision":
        statuses = {str(item.status) for item in patch_rows}
        if statuses == {"accepted"}:
            decision = "accept"
        elif statuses == {"rejected"}:
            decision = "reject"
        else:
            raise CollaborationConflict("command_metadata_conflict")
    return CollaborationPersistedCommand(
        actor_kind=row.actor_kind,
        actor_user_id=row.actor_user_id,
        change_kind=row.change_kind,
        command_id=row.command_id,
        command_payload_hash=row.command_payload_hash,
        decision=decision,
        generation=int(row.generation),
        patch_ids=patch_ids,
        sequence=int(row.sequence),
        suggestion_ids=tuple(str(item) for item in (row.suggestion_ids or [])),
        update_hash=str(row.update_hash),
    )


async def _lock_instance_fence(
    session: "AsyncSession",
    *,
    tenant_id: str,
    instance_id: str,
    instance_epoch: int,
    now: float,
) -> Any:
    """Lock and validate the writer fence inside its mutation transaction."""
    row = (
        await session.execute(
            select(editor_collaboration_instances)
            .where(
                editor_collaboration_instances.c.slot == "primary",
                editor_collaboration_instances.c.tenant_id == tenant_id,
                editor_collaboration_instances.c.instance_id == instance_id,
                editor_collaboration_instances.c.epoch == instance_epoch,
                editor_collaboration_instances.c.lease_expires_at > now,
            )
            .with_for_update()
        )
    ).one_or_none()
    if row is None:
        raise CollaborationInstanceFenced("stale collaboration instance")
    return row


async def _lock_lease_rate_scope(
    session: "AsyncSession",
    *,
    tenant_id: str,
    user_id: uuid.UUID,
) -> None:
    """Serialize per-user issuance counts across document lock domains."""
    await session.execute(
        select(
            func.pg_advisory_xact_lock(
                func.hashtextextended(
                    f"editor-collaboration-lease:{tenant_id}:{user_id}",
                    0,
                )
            )
        )
    )


class PostgresEditorCollaborationStore:
    """Durable collaboration state on the shared platform session factory."""

    def __init__(
        self,
        *,
        session_factory: "async_sessionmaker[AsyncSession]",
        app_role: str,
        restrict_to_workspace_members: bool,
    ) -> None:
        self._session_factory = session_factory
        self._app_role = app_role
        self._restrict_to_workspace_members = restrict_to_workspace_members

    def _session(
        self, tenant_id: str
    ) -> "AbstractAsyncContextManager[AsyncSession]":
        return tenant_session(
            self._session_factory,
            tenant_id=tenant_id,
            app_role=self._app_role,
        )

    async def load_state(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int | None = None,
    ) -> CollaborationLoadedState:
        async with self._session(tenant_id) as session:
            document_row = (
                await session.execute(
                    select(editor_documents).where(
                        editor_documents.c.tenant_id == tenant_id,
                        editor_documents.c.id == document_id,
                        editor_documents.c.content_mode == "collaboration",
                        editor_documents.c.deleted_at.is_(None),
                    )
                )
            ).one_or_none()
            if document_row is None:
                raise CollaborationDocumentNotFound(document_id)
            document = _document_state(document_row)
            if generation is not None and generation != document.generation:
                raise CollaborationConflict(
                    "generation_conflict",
                    current_sequence=document.persisted_sequence,
                )
            snapshot_rows = (
                await session.execute(
                    select(editor_collaboration_snapshots)
                    .where(
                        editor_collaboration_snapshots.c.tenant_id == tenant_id,
                        editor_collaboration_snapshots.c.document_id == document_id,
                        editor_collaboration_snapshots.c.generation
                        == document.generation,
                        editor_collaboration_snapshots.c.covered_sequence
                        <= document.persisted_sequence,
                    )
                    .order_by(
                        editor_collaboration_snapshots.c.covered_sequence.desc()
                    )
                    .limit(2)
                )
            ).all()
            if not snapshot_rows:
                log.error(
                    "Collaboration document %s has no verified snapshot.",
                    document_id,
                )
                raise CollaborationConflict("snapshot_missing")
            oldest_covered_sequence = int(snapshot_rows[-1].covered_sequence)
            update_rows = (
                await session.execute(
                    select(editor_collaboration_updates)
                    .where(
                        editor_collaboration_updates.c.tenant_id == tenant_id,
                        editor_collaboration_updates.c.document_id == document_id,
                        editor_collaboration_updates.c.generation
                        == document.generation,
                        editor_collaboration_updates.c.sequence
                        > oldest_covered_sequence,
                        editor_collaboration_updates.c.sequence
                        <= document.persisted_sequence,
                    )
                    .order_by(editor_collaboration_updates.c.sequence)
                )
            ).all()
        all_updates = tuple(_collaboration_update(row) for row in update_rows)
        candidates: list[CollaborationSnapshotCandidate] = []
        for snapshot_row in snapshot_rows:
            snapshot = _snapshot(snapshot_row)
            updates = tuple(
                item
                for item in all_updates
                if item.sequence > snapshot.covered_sequence
            )
            if (
                len(updates)
                != document.persisted_sequence - snapshot.covered_sequence
                or any(
                    item.sequence != snapshot.covered_sequence + offset
                    for offset, item in enumerate(updates, start=1)
                )
                or any(item.update_bytes is None for item in updates)
            ):
                log.error(
                    "Collaboration document %s has an incomplete update tail after sequence %s.",
                    document_id,
                    snapshot.covered_sequence,
                )
                raise CollaborationConflict("update_tail_incomplete")
            candidates.append(
                CollaborationSnapshotCandidate(snapshot=snapshot, updates=updates)
            )
        primary = candidates[0]
        return CollaborationLoadedState(
            document=document,
            snapshot=primary.snapshot,
            updates=primary.updates,
            fallback_candidates=tuple(candidates[1:]),
        )

    async def enable_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        owner_user_id: uuid.UUID,
        expected_revision: int,
        expected_metadata_revision: int,
        schema_version: int,
        schema_hash: str,
        snapshot: CollaborationSnapshot,
        projection_markdown: str,
        now: float,
    ) -> CollaborationDocumentState:
        if (
            snapshot.document_id != document_id
            or snapshot.tenant_id != tenant_id
            or snapshot.generation != 1
            or snapshot.covered_sequence != 0
            or snapshot.schema_version != schema_version
            or snapshot.schema_hash != schema_hash
        ):
            raise ValueError("initial snapshot does not match the activation contract")
        _require_sha256(schema_hash, field="schema_hash")
        _require_sha256(snapshot.state_hash, field="state_hash")
        _require_sha256(snapshot.projection_hash, field="projection_hash")
        if not hmac.compare_digest(
            hashlib.sha256(snapshot.state_update).hexdigest(),
            snapshot.state_hash,
        ):
            raise ValueError("state_hash does not match state_update")
        if not hmac.compare_digest(
            hashlib.sha256(projection_markdown.encode("utf-8")).hexdigest(),
            snapshot.projection_hash,
        ):
            raise ValueError("projection_hash does not match projection_markdown")
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(editor_documents)
                    .where(
                        editor_documents.c.tenant_id == tenant_id,
                        editor_documents.c.id == document_id,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if row is None or row.deleted_at is not None:
                raise CollaborationDocumentNotFound(document_id)
            if row.created_by_user_id != owner_user_id:
                raise CollaborationDocumentNotFound(document_id)
            if row.content_mode != "markdown":
                raise CollaborationConflict("mode_conflict")
            if int(row.revision) != expected_revision:
                raise CollaborationConflict("revision_conflict")
            if int(row.metadata_revision) != expected_metadata_revision:
                raise CollaborationConflict("metadata_conflict")
            active_share = await session.scalar(
                select(resource_shares.c.id)
                .where(
                    resource_shares.c.tenant_id == tenant_id,
                    resource_shares.c.resource_type == "editor_document",
                    resource_shares.c.resource_id == document_id,
                    resource_shares.c.revoked_at.is_(None),
                )
                .limit(1)
            )
            if active_share is not None:
                raise CollaborationConflict("share_conflict")
            await session.execute(
                insert(editor_collaboration_snapshots).values(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    generation=1,
                    covered_sequence=0,
                    state_update=snapshot.state_update,
                    state_vector=snapshot.state_vector,
                    state_hash=snapshot.state_hash,
                    projection_hash=snapshot.projection_hash,
                    schema_version=schema_version,
                    schema_hash=schema_hash,
                    created_at=snapshot.created_at,
                )
            )
            stored_row = (
                await session.execute(
                    update(editor_documents)
                    .where(
                        editor_documents.c.tenant_id == tenant_id,
                        editor_documents.c.id == document_id,
                    )
                    .values(
                        content_mode="collaboration",
                        collaboration_generation=1,
                        collaboration_schema_version=schema_version,
                        collaboration_schema_hash=schema_hash,
                        persisted_sequence=0,
                        projection_sequence=0,
                        content_markdown=projection_markdown,
                        projection_updated_at=now,
                        metadata_revision=expected_metadata_revision + 1,
                        updated_at=now,
                    )
                    .returning(editor_documents)
                )
            ).one()
            await append_resource_effects(
                session,
                tenant_id=tenant_id,
                actor_user_id=owner_user_id,
                owner_user_id=owner_user_id,
                action="editor_document.collaboration_enabled",
                resource_type="editor_document",
                resource_id=document_id,
                scope="editor_documents",
            )
        return _document_state(stored_row)

    async def issue_lease(
        self,
        lease: CollaborationLease,
        *,
        max_active: int,
        max_issued_per_window: int,
        issued_since: float,
    ) -> CollaborationLease:
        if max_active < 1 or max_issued_per_window < 1:
            raise ValueError("lease limits must be positive")
        minimum = SharePermission(lease.permission)
        async with self._session(lease.tenant_id) as session:
            await _lock_lease_rate_scope(
                session,
                tenant_id=lease.tenant_id,
                user_id=lease.user_id,
            )
            access = await lock_resource_access(
                session,
                tenant_id=lease.tenant_id,
                actor_user_id=lease.user_id,
                resource_type="editor_document",
                resource_table=editor_documents,
                id_column=editor_documents.c.id,
                resource_id=lease.document_id,
                owner_column=editor_documents.c.created_by_user_id,
                minimum=minimum,
                restrict_to_workspace_members=self._restrict_to_workspace_members,
            )
            if access is None:
                raise CollaborationDocumentNotFound(lease.document_id)
            session_exists = await session.scalar(
                select(auth_sessions.c.id).where(
                    auth_sessions.c.tenant_id == lease.tenant_id,
                    auth_sessions.c.id == lease.session_id,
                    auth_sessions.c.user_id == lease.user_id,
                    auth_sessions.c.expires_at > lease.issued_at,
                )
            )
            document_row = (
                await session.execute(
                    select(editor_documents).where(
                        editor_documents.c.tenant_id == lease.tenant_id,
                        editor_documents.c.id == lease.document_id,
                    )
                )
            ).one()
            if session_exists is None:
                raise CollaborationLeaseInvalid("session_invalid")
            if (
                document_row.content_mode != "collaboration"
                or document_row.deleted_at is not None
                or int(document_row.collaboration_generation) != lease.generation
            ):
                raise CollaborationConflict("generation_conflict")
            active_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(editor_collaboration_leases)
                    .where(
                        editor_collaboration_leases.c.tenant_id
                        == lease.tenant_id,
                        editor_collaboration_leases.c.document_id
                        == lease.document_id,
                        editor_collaboration_leases.c.generation
                        == lease.generation,
                        editor_collaboration_leases.c.user_id == lease.user_id,
                        editor_collaboration_leases.c.revoked_at.is_(None),
                        editor_collaboration_leases.c.expires_at
                        > lease.issued_at,
                    )
                )
                or 0
            )
            if active_count >= max_active:
                raise CollaborationRateLimited("session_limit")
            issued_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(editor_collaboration_leases)
                    .where(
                        editor_collaboration_leases.c.tenant_id
                        == lease.tenant_id,
                        editor_collaboration_leases.c.user_id == lease.user_id,
                        editor_collaboration_leases.c.issued_at >= issued_since,
                    )
                )
                or 0
            )
            if issued_count >= max_issued_per_window:
                raise CollaborationRateLimited("session_rate_limited")
            await session.execute(
                insert(editor_collaboration_leases).values(**_lease_values(lease))
            )
        return lease

    async def rotate_lease(
        self,
        *,
        previous_lease_id: uuid.UUID,
        previous_token_hash: str,
        replacement: CollaborationLease,
        max_issued_per_window: int,
        issued_since: float,
    ) -> CollaborationLease:
        if max_issued_per_window < 1:
            raise ValueError("lease limits must be positive")
        if (
            replacement.rotation_command_id is None
            or replacement.rotated_from_lease_id != previous_lease_id
        ):
            raise ValueError("replacement must identify its rotation command")
        async with self._session(replacement.tenant_id) as session:
            await _lock_lease_rate_scope(
                session,
                tenant_id=replacement.tenant_id,
                user_id=replacement.user_id,
            )
            existing_replacement = (
                await session.execute(
                    select(editor_collaboration_leases)
                    .where(
                        editor_collaboration_leases.c.tenant_id
                        == replacement.tenant_id,
                        editor_collaboration_leases.c.rotation_command_id
                        == replacement.rotation_command_id,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            previous = (
                await session.execute(
                    select(editor_collaboration_leases)
                    .where(
                        editor_collaboration_leases.c.tenant_id
                        == replacement.tenant_id,
                        editor_collaboration_leases.c.lease_id
                        == previous_lease_id,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if previous is None or not hmac.compare_digest(
                str(previous.token_hash), previous_token_hash
            ):
                raise CollaborationLeaseInvalid("lease_invalid")
            if existing_replacement is not None:
                if (
                    existing_replacement.rotated_from_lease_id
                    != previous_lease_id
                    or existing_replacement.document_id
                    != replacement.document_id
                    or int(existing_replacement.generation)
                    != replacement.generation
                    or existing_replacement.user_id != replacement.user_id
                    or existing_replacement.session_id != replacement.session_id
                ):
                    raise CollaborationConflict("rotation_command_conflict")
                if existing_replacement.revoked_at is not None:
                    raise CollaborationLeaseInvalid("lease_revoked")
                if float(existing_replacement.expires_at) <= replacement.issued_at:
                    raise CollaborationLeaseInvalid("lease_expired")
                return _lease(existing_replacement)
            if previous.revoked_at is not None:
                raise CollaborationLeaseInvalid("lease_revoked")
            if float(previous.expires_at) <= replacement.issued_at:
                raise CollaborationLeaseInvalid("lease_expired")
            if (
                previous.document_id != replacement.document_id
                or int(previous.generation) != replacement.generation
                or previous.user_id != replacement.user_id
                or previous.session_id != replacement.session_id
            ):
                raise CollaborationLeaseInvalid("lease_invalid")
            access = await lock_resource_access(
                session,
                tenant_id=replacement.tenant_id,
                actor_user_id=replacement.user_id,
                resource_type="editor_document",
                resource_table=editor_documents,
                id_column=editor_documents.c.id,
                resource_id=replacement.document_id,
                owner_column=editor_documents.c.created_by_user_id,
                minimum=SharePermission(replacement.permission),
                restrict_to_workspace_members=self._restrict_to_workspace_members,
            )
            active_session = await session.scalar(
                select(auth_sessions.c.id).where(
                    auth_sessions.c.tenant_id == replacement.tenant_id,
                    auth_sessions.c.id == replacement.session_id,
                    auth_sessions.c.user_id == replacement.user_id,
                    auth_sessions.c.expires_at > replacement.issued_at,
                )
            )
            if access is None or active_session is None:
                raise CollaborationLeaseInvalid("access_revoked")
            issued_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(editor_collaboration_leases)
                    .where(
                        editor_collaboration_leases.c.tenant_id
                        == replacement.tenant_id,
                        editor_collaboration_leases.c.user_id
                        == replacement.user_id,
                        editor_collaboration_leases.c.issued_at >= issued_since,
                    )
                )
                or 0
            )
            if issued_count >= max_issued_per_window:
                raise CollaborationRateLimited("session_rate_limited")
            await session.execute(
                update(editor_collaboration_leases)
                .where(
                    editor_collaboration_leases.c.tenant_id
                    == replacement.tenant_id,
                    editor_collaboration_leases.c.lease_id
                    == previous_lease_id,
                )
                .values(revoked_at=replacement.issued_at)
            )
            await session.execute(
                insert(editor_collaboration_leases).values(
                    **_lease_values(replacement)
                )
            )
        return replacement

    async def introspect_lease(
        self,
        *,
        tenant_id: str,
        lease_id: uuid.UUID,
        token_hash: str,
        now: float,
    ) -> CollaborationLease:
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(editor_collaboration_leases)
                    .where(
                        editor_collaboration_leases.c.tenant_id == tenant_id,
                        editor_collaboration_leases.c.lease_id == lease_id,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if row is None or not hmac.compare_digest(str(row.token_hash), token_hash):
                raise CollaborationLeaseInvalid()
            if row.revoked_at is not None:
                raise CollaborationLeaseInvalid("access_revoked")
            if float(row.expires_at) <= now:
                raise CollaborationLeaseInvalid("lease_expired")
            access = await lock_resource_access(
                session,
                tenant_id=tenant_id,
                actor_user_id=row.user_id,
                resource_type="editor_document",
                resource_table=editor_documents,
                id_column=editor_documents.c.id,
                resource_id=row.document_id,
                owner_column=editor_documents.c.created_by_user_id,
                minimum=SharePermission(row.permission),
                restrict_to_workspace_members=self._restrict_to_workspace_members,
            )
            active_session = await session.scalar(
                select(auth_sessions.c.id).where(
                    auth_sessions.c.tenant_id == tenant_id,
                    auth_sessions.c.id == row.session_id,
                    auth_sessions.c.user_id == row.user_id,
                    auth_sessions.c.expires_at > now,
                )
            )
            if access is None or active_session is None:
                await session.execute(
                    update(editor_collaboration_leases)
                    .where(editor_collaboration_leases.c.lease_id == lease_id)
                    .values(revoked_at=now)
                )
                raise CollaborationLeaseInvalid("access_revoked")
            document_row = (
                await session.execute(
                    select(editor_documents).where(
                        editor_documents.c.tenant_id == tenant_id,
                        editor_documents.c.id == row.document_id,
                    )
                )
            ).one()
            if (
                document_row.content_mode != "collaboration"
                or document_row.deleted_at is not None
                or int(document_row.collaboration_generation) != int(row.generation)
            ):
                raise CollaborationLeaseInvalid("generation_conflict")
            stored = (
                await session.execute(
                    update(editor_collaboration_leases)
                    .where(editor_collaboration_leases.c.lease_id == lease_id)
                    .values(validated_at=now)
                    .returning(editor_collaboration_leases)
                )
            ).one()
        return _lease(stored)

    async def revoke_leases(
        self,
        *,
        tenant_id: str,
        document_id: str,
        user_id: uuid.UUID | None,
        now: float,
    ) -> int:
        predicates = [
            editor_collaboration_leases.c.tenant_id == tenant_id,
            editor_collaboration_leases.c.document_id == document_id,
            editor_collaboration_leases.c.revoked_at.is_(None),
        ]
        if user_id is not None:
            predicates.append(editor_collaboration_leases.c.user_id == user_id)
        async with self._session(tenant_id) as session:
            result = await session.execute(
                update(editor_collaboration_leases)
                .where(*predicates)
                .values(revoked_at=now)
            )
        return int(result.rowcount or 0)

    async def acquire_instance(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        now: float,
        lease_seconds: float,
    ) -> CollaborationInstanceLease:
        if not instance_id.strip() or lease_seconds <= 0:
            raise ValueError("instance_id and lease_seconds must be valid")
        async with self._session(tenant_id) as session:
            expires_at = now + lease_seconds
            await session.execute(
                pg_insert(editor_collaboration_instances)
                .values(
                    slot="primary",
                    tenant_id=tenant_id,
                    instance_id=instance_id,
                    epoch=1,
                    lease_expires_at=expires_at,
                    updated_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        editor_collaboration_instances.c.tenant_id,
                        editor_collaboration_instances.c.slot,
                    ]
                )
            )
            row = (
                await session.execute(
                    select(editor_collaboration_instances)
                    .where(
                        editor_collaboration_instances.c.slot == "primary",
                        editor_collaboration_instances.c.tenant_id == tenant_id,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if row is None:
                raise CollaborationInstanceFenced("primary instance slot unavailable")
            if row.instance_id == instance_id and float(row.lease_expires_at) > now:
                stored = (
                    await session.execute(
                        update(editor_collaboration_instances)
                        .where(
                            editor_collaboration_instances.c.slot == "primary",
                            editor_collaboration_instances.c.tenant_id == tenant_id,
                        )
                        .values(lease_expires_at=expires_at, updated_at=now)
                        .returning(editor_collaboration_instances)
                    )
                ).one()
            elif float(row.lease_expires_at) > now:
                raise CollaborationInstanceFenced("primary instance lease is active")
            else:
                stored = (
                    await session.execute(
                        update(editor_collaboration_instances)
                        .where(
                            editor_collaboration_instances.c.slot == "primary",
                            editor_collaboration_instances.c.tenant_id == tenant_id,
                        )
                        .values(
                            instance_id=instance_id,
                            epoch=int(row.epoch) + 1,
                            lease_expires_at=expires_at,
                            updated_at=now,
                        )
                        .returning(editor_collaboration_instances)
                    )
                ).one()
        return CollaborationInstanceLease(
            instance_id=str(stored.instance_id),
            epoch=int(stored.epoch),
            lease_expires_at=float(stored.lease_expires_at),
            updated_at=float(stored.updated_at),
        )

    async def renew_instance(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        epoch: int,
        now: float,
        lease_seconds: float,
    ) -> CollaborationInstanceLease:
        async with self._session(tenant_id) as session:
            stored = (
                await session.execute(
                    update(editor_collaboration_instances)
                    .where(
                        editor_collaboration_instances.c.slot == "primary",
                        editor_collaboration_instances.c.tenant_id == tenant_id,
                        editor_collaboration_instances.c.instance_id == instance_id,
                        editor_collaboration_instances.c.epoch == epoch,
                        editor_collaboration_instances.c.lease_expires_at > now,
                    )
                    .values(lease_expires_at=now + lease_seconds, updated_at=now)
                    .returning(editor_collaboration_instances)
                )
            ).one_or_none()
        if stored is None:
            raise CollaborationInstanceFenced("instance lease was lost")
        return CollaborationInstanceLease(
            instance_id=str(stored.instance_id),
            epoch=int(stored.epoch),
            lease_expires_at=float(stored.lease_expires_at),
            updated_at=float(stored.updated_at),
        )

    async def validate_instance(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        epoch: int,
        now: float,
    ) -> CollaborationInstanceLease:
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(editor_collaboration_instances).where(
                        editor_collaboration_instances.c.slot == "primary",
                        editor_collaboration_instances.c.tenant_id == tenant_id,
                        editor_collaboration_instances.c.instance_id == instance_id,
                        editor_collaboration_instances.c.epoch == epoch,
                        editor_collaboration_instances.c.lease_expires_at > now,
                    )
                )
            ).one_or_none()
        if row is None:
            raise CollaborationInstanceFenced("instance lease was lost")
        return CollaborationInstanceLease(
            instance_id=str(row.instance_id),
            epoch=int(row.epoch),
            lease_expires_at=float(row.lease_expires_at),
            updated_at=float(row.updated_at),
        )

    async def get_current_instance(
        self,
        *,
        tenant_id: str,
        now: float,
    ) -> CollaborationInstanceLease | None:
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(editor_collaboration_instances).where(
                        editor_collaboration_instances.c.slot == "primary",
                        editor_collaboration_instances.c.tenant_id == tenant_id,
                        editor_collaboration_instances.c.lease_expires_at > now,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        return CollaborationInstanceLease(
            instance_id=str(row.instance_id),
            epoch=int(row.epoch),
            lease_expires_at=float(row.lease_expires_at),
            updated_at=float(row.updated_at),
        )

    async def append_update(
        self, update_record: PersistCollaborationUpdate
    ) -> PersistedCollaborationUpdate:
        now = update_record.now or time.time()
        if not update_record.update_bytes:
            raise ValueError("update_bytes must be non-empty")
        _require_sha256(update_record.update_hash, field="update_hash")
        if not hmac.compare_digest(
            hashlib.sha256(update_record.update_bytes).hexdigest(),
            update_record.update_hash,
        ):
            raise ValueError("update_hash does not match update_bytes")
        suggestion_ids = {item.suggestion_id for item in update_record.suggestions}
        if suggestion_ids != set(update_record.suggestion_ids):
            raise ValueError("suggestion descriptors do not match suggestion_ids")
        patch_ids = {item.patch_id for item in update_record.patches}
        if any(
            item.patch_id not in patch_ids for item in update_record.suggestions
        ):
            raise ValueError("suggestion descriptor references an unknown patch")
        if update_record.decision is None:
            if update_record.change_kind == "decision":
                raise ValueError("decision is required for decision updates")
        elif (
            update_record.change_kind != "decision"
            or update_record.command_id is None
            or not update_record.patches
        ):
            raise ValueError("decision metadata requires a command and patches")
        if (
            update_record.change_kind not in {"suggestion", "decision"}
            and (update_record.suggestions or update_record.patches)
        ):
            raise ValueError("patch metadata is only valid for suggestion updates")
        command_id = update_record.command_id
        command_payload_hash = update_record.command_payload_hash
        if (command_id is None) != (command_payload_hash is None):
            raise ValueError("command_id and command_payload_hash must be paired")
        if command_payload_hash is not None:
            _require_sha256(command_payload_hash, field="command_payload_hash")
        minimum = (
            SharePermission.SUGGEST
            if update_record.change_kind == "suggestion"
            else SharePermission.EDIT
        )
        async with self._session(update_record.tenant_id) as session:
            await _lock_instance_fence(
                session,
                tenant_id=update_record.tenant_id,
                instance_id=update_record.instance_id,
                instance_epoch=update_record.instance_epoch,
                now=now,
            )
            lease_row = None
            if update_record.lease_id is not None:
                lease_row = (
                    await session.execute(
                        select(editor_collaboration_leases)
                        .where(
                            editor_collaboration_leases.c.tenant_id
                            == update_record.tenant_id,
                            editor_collaboration_leases.c.lease_id
                            == update_record.lease_id,
                            editor_collaboration_leases.c.document_id
                            == update_record.document_id,
                            editor_collaboration_leases.c.generation
                            == update_record.generation,
                            editor_collaboration_leases.c.revoked_at.is_(None),
                            editor_collaboration_leases.c.expires_at > now,
                        )
                        .with_for_update()
                    )
                ).one_or_none()
                if (
                    lease_row is None
                    or lease_row.user_id != update_record.actor_user_id
                ):
                    raise CollaborationLeaseInvalid()
                if (
                    update_record.change_kind == "suggestion"
                    and lease_row.permission not in {"suggest", "edit"}
                ) or (
                    update_record.change_kind != "suggestion"
                    and lease_row.permission != "edit"
                ):
                    raise CollaborationLeaseInvalid("permission_denied")
                active_session = await session.scalar(
                    select(auth_sessions.c.id).where(
                        auth_sessions.c.tenant_id == update_record.tenant_id,
                        auth_sessions.c.id == lease_row.session_id,
                        auth_sessions.c.user_id == lease_row.user_id,
                        auth_sessions.c.expires_at > now,
                    )
                )
                if active_session is None:
                    raise CollaborationLeaseInvalid("session_invalid")
            elif (
                command_id is None
                or update_record.change_kind not in {"decision", "suggestion"}
            ):
                raise CollaborationLeaseInvalid("lease_required")
            access = await lock_resource_access(
                session,
                tenant_id=update_record.tenant_id,
                actor_user_id=update_record.actor_user_id,
                resource_type="editor_document",
                resource_table=editor_documents,
                id_column=editor_documents.c.id,
                resource_id=update_record.document_id,
                owner_column=editor_documents.c.created_by_user_id,
                minimum=minimum,
                restrict_to_workspace_members=self._restrict_to_workspace_members,
            )
            if access is None:
                raise CollaborationLeaseInvalid("access_revoked")
            document = (
                await session.execute(
                    select(editor_documents)
                    .where(
                        editor_documents.c.tenant_id == update_record.tenant_id,
                        editor_documents.c.id == update_record.document_id,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if document is None or document.deleted_at is not None:
                raise CollaborationDocumentNotFound(update_record.document_id)
            if (
                document.content_mode != "collaboration"
                or int(document.collaboration_generation)
                != update_record.generation
            ):
                raise CollaborationConflict(
                    "generation_conflict",
                    current_sequence=int(document.persisted_sequence),
                )
            current_sequence = int(document.persisted_sequence)
            duplicate = (
                await session.execute(
                    select(
                        editor_collaboration_updates.c.sequence,
                        editor_collaboration_updates.c.command_id,
                        editor_collaboration_updates.c.command_payload_hash,
                    ).where(
                        editor_collaboration_updates.c.tenant_id
                        == update_record.tenant_id,
                        editor_collaboration_updates.c.document_id
                        == update_record.document_id,
                        editor_collaboration_updates.c.generation
                        == update_record.generation,
                        editor_collaboration_updates.c.update_hash
                        == update_record.update_hash,
                    )
                )
            ).one_or_none()
            if duplicate is not None:
                if (
                    duplicate.command_id != command_id
                    or duplicate.command_payload_hash != command_payload_hash
                ):
                    raise CollaborationConflict("command_conflict")
                return PersistedCollaborationUpdate(
                    sequence=int(duplicate.sequence),
                    persisted_sequence=current_sequence,
                    duplicate=True,
                    persisted_at=now,
                )
            if command_id is not None:
                command_row = (
                    await session.execute(
                        select(
                            editor_collaboration_updates.c.document_id,
                            editor_collaboration_updates.c.generation,
                            editor_collaboration_updates.c.sequence,
                            editor_collaboration_updates.c.update_hash,
                            editor_collaboration_updates.c.command_payload_hash,
                        ).where(
                            editor_collaboration_updates.c.command_id == command_id
                        )
                    )
                ).one_or_none()
                if command_row is not None:
                    if (
                        command_row.document_id == update_record.document_id
                        and int(command_row.generation) == update_record.generation
                        and command_row.update_hash == update_record.update_hash
                        and command_row.command_payload_hash
                        == command_payload_hash
                    ):
                        return PersistedCollaborationUpdate(
                            sequence=int(command_row.sequence),
                            persisted_sequence=current_sequence,
                            duplicate=True,
                            persisted_at=now,
                        )
                    raise CollaborationConflict("command_conflict")
            if (
                update_record.expected_sequence is not None
                and update_record.expected_sequence != current_sequence
            ):
                raise CollaborationConflict(
                    "sequence_conflict", current_sequence=current_sequence
                )
            next_sequence = current_sequence + 1
            await self._persist_patch_states(
                session,
                update_record=update_record,
                document=document,
                next_sequence=next_sequence,
                now=now,
            )
            await session.execute(
                insert(editor_collaboration_updates).values(
                    document_id=update_record.document_id,
                    tenant_id=update_record.tenant_id,
                    generation=update_record.generation,
                    sequence=next_sequence,
                    update_hash=update_record.update_hash,
                    update_bytes=update_record.update_bytes,
                    actor_user_id=update_record.actor_user_id,
                    actor_kind=update_record.actor_kind,
                    change_kind=update_record.change_kind,
                    suggestion_ids=list(update_record.suggestion_ids),
                    command_id=command_id,
                    command_payload_hash=command_payload_hash,
                    created_at=now,
                    payload_pruned_at=None,
                )
            )
            await session.execute(
                update(editor_documents)
                .where(
                    editor_documents.c.tenant_id == update_record.tenant_id,
                    editor_documents.c.id == update_record.document_id,
                )
                .values(persisted_sequence=next_sequence, updated_at=now)
            )
        return PersistedCollaborationUpdate(
            sequence=next_sequence,
            persisted_sequence=next_sequence,
            duplicate=False,
            persisted_at=now,
        )

    async def _persist_patch_states(
        self,
        session: "AsyncSession",
        *,
        update_record: PersistCollaborationUpdate,
        document: Any,
        next_sequence: int,
        now: float,
    ) -> None:
        """Persist patch membership and decisions in the update transaction."""
        if not update_record.patches:
            return
        suggestions_by_patch: dict[str, dict[str, dict[str, Any]]] = {}
        for suggestion in update_record.suggestions:
            suggestions_by_patch.setdefault(suggestion.patch_id, {})[
                suggestion.suggestion_id
            ] = {
                "suggestion_id": suggestion.suggestion_id,
                "patch_id": suggestion.patch_id,
                "author_id": str(suggestion.author_id),
                "created_at": suggestion.created_at,
                "kind": suggestion.kind,
            }
        for patch_state in update_record.patches:
            if (
                update_record.decision is None
                and not patch_state.active_suggestion_ids
            ):
                raise CollaborationConflict("patch_suggestions_empty")
            row = (
                await session.execute(
                    select(editor_patches)
                    .where(
                        editor_patches.c.tenant_id == update_record.tenant_id,
                        editor_patches.c.patch_id == patch_state.patch_id,
                        editor_patches.c.document_id == update_record.document_id,
                    )
                    .with_for_update()
                )
            ).mappings().one_or_none()
            if row is None:
                if (
                    update_record.actor_kind != "human"
                    or update_record.change_kind != "suggestion"
                    or patch_state.author_id != update_record.actor_user_id
                    or not patch_state.active_suggestion_ids
                ):
                    raise CollaborationConflict("patch_not_found")
                descriptors = suggestions_by_patch.get(patch_state.patch_id, {})
                if set(descriptors) != set(patch_state.active_suggestion_ids):
                    raise CollaborationConflict("patch_metadata_conflict")
                await session.execute(
                    insert(editor_patches).values(
                        patch_id=patch_state.patch_id,
                        tenant_id=update_record.tenant_id,
                        document_id=update_record.document_id,
                        run_id=None,
                        source="human",
                        status="pending",
                        edits=list(descriptors.values()),
                        summary="",
                        warnings=[],
                        revision_before=int(document.revision),
                        collaboration_generation=update_record.generation,
                        base_sequence=next_sequence - 1,
                        decision_sequence=None,
                        suggestion_ids=list(patch_state.active_suggestion_ids),
                        applied_revision=None,
                        applied_edit_ids=None,
                        note="",
                        created_by_user_id=patch_state.author_id,
                        decided_by_user_id=None,
                        command_id=None,
                        created_at=patch_state.created_at,
                        decided_at=None,
                    )
                )
                continue
            if (
                row["collaboration_generation"] != update_record.generation
                or row["created_by_user_id"] != patch_state.author_id
                or row["status"] != "pending"
            ):
                raise CollaborationConflict("patch_metadata_conflict")
            if update_record.decision is not None:
                if patch_state.active_suggestion_ids:
                    raise CollaborationConflict("patch_decision_incomplete")
                prior_ids = [str(value) for value in (row["suggestion_ids"] or [])]
                await session.execute(
                    update(editor_patches)
                    .where(
                        editor_patches.c.tenant_id == update_record.tenant_id,
                        editor_patches.c.patch_id == patch_state.patch_id,
                        editor_patches.c.status == "pending",
                    )
                    .values(
                        status=(
                            "accepted"
                            if update_record.decision == "accept"
                            else "rejected"
                        ),
                        decision_sequence=next_sequence,
                        applied_edit_ids=(
                            prior_ids
                            if update_record.decision == "accept"
                            else None
                        ),
                        decided_by_user_id=update_record.actor_user_id,
                        command_id=update_record.command_id,
                        decided_at=now,
                    )
                )
                continue
            active_ids = set(patch_state.active_suggestion_ids)
            prior_active_ids = {
                str(item) for item in (row["suggestion_ids"] or [])
            }
            if update_record.actor_user_id != row["created_by_user_id"]:
                raise CollaborationConflict("patch_author_conflict")
            if not prior_active_ids.issubset(active_ids):
                raise CollaborationConflict("patch_membership_shrink")
            new_descriptors = suggestions_by_patch.get(patch_state.patch_id, {})
            if any(
                descriptor["author_id"] != str(row["created_by_user_id"])
                for descriptor in new_descriptors.values()
            ):
                raise CollaborationConflict("patch_author_conflict")
            known_ids = prior_active_ids | set(new_descriptors)
            if not active_ids.issubset(known_ids):
                raise CollaborationConflict("patch_metadata_conflict")
            values: dict[str, Any] = {
                "suggestion_ids": list(patch_state.active_suggestion_ids),
                "command_id": (
                    update_record.command_id
                    if update_record.command_id is not None
                    else row["command_id"]
                ),
            }
            if row["source"] == "human":
                descriptors = {
                    str(item.get("suggestion_id")): dict(item)
                    for item in (row["edits"] or [])
                    if isinstance(item, dict) and item.get("suggestion_id")
                }
                descriptors.update(new_descriptors)
                if not active_ids.issubset(descriptors):
                    raise CollaborationConflict("patch_metadata_conflict")
                values["edits"] = [
                    descriptors[suggestion_id]
                    for suggestion_id in patch_state.active_suggestion_ids
                ]
            await session.execute(
                update(editor_patches)
                .where(
                    editor_patches.c.tenant_id == update_record.tenant_id,
                    editor_patches.c.patch_id == patch_state.patch_id,
                    editor_patches.c.status == "pending",
                )
                .values(**values)
            )

    async def lookup_command(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        command_id: uuid.UUID,
        command_payload_hash: str,
    ) -> CollaborationPersistedCommand | None:
        _require_sha256(command_payload_hash, field="command_payload_hash")
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(editor_collaboration_updates).where(
                        editor_collaboration_updates.c.command_id == command_id
                    )
                )
            ).one_or_none()
            if row is None:
                return None
            if (
                row.document_id != document_id
                or int(row.generation) != generation
                or row.command_payload_hash != command_payload_hash
                or row.change_kind not in {"decision", "suggestion"}
                or row.actor_user_id is None
            ):
                raise CollaborationConflict("command_conflict")
            patch_rows = (
                await session.execute(
                    select(
                        editor_patches.c.patch_id,
                        editor_patches.c.status,
                    ).where(
                        editor_patches.c.tenant_id == tenant_id,
                        editor_patches.c.document_id == document_id,
                        editor_patches.c.collaboration_generation == generation,
                        editor_patches.c.command_id == command_id,
                    )
                )
            ).all()
        return _persisted_command(row, patch_rows)

    async def lookup_decision_command_by_id(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        command_id: uuid.UUID,
    ) -> CollaborationPersistedCommand | None:
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(editor_collaboration_updates).where(
                        editor_collaboration_updates.c.tenant_id == tenant_id,
                        editor_collaboration_updates.c.command_id == command_id,
                    )
                )
            ).one_or_none()
            if row is None:
                return None
            if (
                row.document_id != document_id
                or int(row.generation) != generation
                or row.change_kind != "decision"
                or row.actor_user_id is None
            ):
                raise CollaborationConflict("command_conflict")
            patch_rows = (
                await session.execute(
                    select(
                        editor_patches.c.patch_id,
                        editor_patches.c.status,
                    ).where(
                        editor_patches.c.tenant_id == tenant_id,
                        editor_patches.c.document_id == document_id,
                        editor_patches.c.collaboration_generation == generation,
                        editor_patches.c.command_id == command_id,
                    )
                )
            ).all()
        return _persisted_command(row, patch_rows)

    async def lookup_updates_by_hashes(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        update_hashes: tuple[str, ...],
        instance_id: str,
        instance_epoch: int,
        now: float,
    ) -> tuple[CollaborationUpdateLookup, ...]:
        """Resolve update hashes after validating the scoped generation."""
        if (
            len(update_hashes) > 1000
            or len(set(update_hashes)) != len(update_hashes)
        ):
            raise ValueError("update_hashes must contain at most 1000 unique digests")
        for update_hash in update_hashes:
            _require_sha256(update_hash, field="update_hash")
        async with self._session(tenant_id) as session:
            await _lock_instance_fence(
                session,
                tenant_id=tenant_id,
                instance_id=instance_id,
                instance_epoch=instance_epoch,
                now=now,
            )
            document = (
                await session.execute(
                    select(editor_documents).where(
                        editor_documents.c.tenant_id == tenant_id,
                        editor_documents.c.id == document_id,
                        editor_documents.c.content_mode == "collaboration",
                        editor_documents.c.deleted_at.is_(None),
                    )
                )
            ).one_or_none()
            if document is None:
                raise CollaborationDocumentNotFound(document_id)
            if int(document.collaboration_generation) != generation:
                raise CollaborationConflict(
                    "generation_conflict",
                    current_sequence=int(document.persisted_sequence),
                )
            if not update_hashes:
                return ()
            rows = (
                await session.execute(
                    select(
                        editor_collaboration_updates.c.update_hash,
                        editor_collaboration_updates.c.sequence,
                    ).where(
                        editor_collaboration_updates.c.tenant_id == tenant_id,
                        editor_collaboration_updates.c.document_id == document_id,
                        editor_collaboration_updates.c.generation == generation,
                        editor_collaboration_updates.c.update_hash.in_(update_hashes),
                    )
                )
            ).all()
        sequence_by_hash = {
            str(row.update_hash): int(row.sequence) for row in rows
        }
        return tuple(
            CollaborationUpdateLookup(
                update_hash=update_hash,
                sequence=sequence_by_hash[update_hash],
            )
            for update_hash in update_hashes
            if update_hash in sequence_by_hash
        )

    async def store_snapshot(
        self,
        snapshot: CollaborationSnapshot,
        *,
        projection_markdown: str,
        instance_id: str,
        instance_epoch: int,
        now: float,
    ) -> None:
        _require_sha256(snapshot.state_hash, field="state_hash")
        _require_sha256(snapshot.projection_hash, field="projection_hash")
        _require_sha256(snapshot.schema_hash, field="schema_hash")
        if not hmac.compare_digest(
            hashlib.sha256(snapshot.state_update).hexdigest(),
            snapshot.state_hash,
        ):
            raise ValueError("state_hash does not match state_update")
        if not hmac.compare_digest(
            hashlib.sha256(projection_markdown.encode("utf-8")).hexdigest(),
            snapshot.projection_hash,
        ):
            raise ValueError("projection_hash does not match projection_markdown")
        async with self._session(snapshot.tenant_id) as session:
            await _lock_instance_fence(
                session,
                tenant_id=snapshot.tenant_id,
                instance_id=instance_id,
                instance_epoch=instance_epoch,
                now=now,
            )
            document = (
                await session.execute(
                    select(editor_documents)
                    .where(
                        editor_documents.c.tenant_id == snapshot.tenant_id,
                        editor_documents.c.id == snapshot.document_id,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if document is None or document.deleted_at is not None:
                raise CollaborationDocumentNotFound(snapshot.document_id)
            if (
                document.content_mode != "collaboration"
                or int(document.collaboration_generation) != snapshot.generation
                or int(document.collaboration_schema_version)
                != snapshot.schema_version
                or document.collaboration_schema_hash != snapshot.schema_hash
            ):
                raise CollaborationConflict("schema_conflict")
            if snapshot.covered_sequence > int(document.persisted_sequence):
                raise CollaborationConflict(
                    "snapshot_ahead",
                    current_sequence=int(document.persisted_sequence),
                )
            statement = pg_insert(editor_collaboration_snapshots).values(
                document_id=snapshot.document_id,
                tenant_id=snapshot.tenant_id,
                generation=snapshot.generation,
                covered_sequence=snapshot.covered_sequence,
                state_update=snapshot.state_update,
                state_vector=snapshot.state_vector,
                state_hash=snapshot.state_hash,
                projection_hash=snapshot.projection_hash,
                schema_version=snapshot.schema_version,
                schema_hash=snapshot.schema_hash,
                created_at=snapshot.created_at,
            )
            statement = statement.on_conflict_do_nothing(
                index_elements=[
                    editor_collaboration_snapshots.c.document_id,
                    editor_collaboration_snapshots.c.generation,
                    editor_collaboration_snapshots.c.covered_sequence,
                ]
            )
            result = await session.execute(statement)
            if not result.rowcount:
                existing = (
                    await session.execute(
                        select(editor_collaboration_snapshots).where(
                            editor_collaboration_snapshots.c.document_id
                            == snapshot.document_id,
                            editor_collaboration_snapshots.c.generation
                            == snapshot.generation,
                            editor_collaboration_snapshots.c.covered_sequence
                            == snapshot.covered_sequence,
                        )
                    )
                ).one()
                if (
                    existing.state_hash != snapshot.state_hash
                    or existing.projection_hash != snapshot.projection_hash
                    or existing.schema_hash != snapshot.schema_hash
                ):
                    raise CollaborationConflict("snapshot_conflict")
            if snapshot.covered_sequence >= int(document.projection_sequence):
                await session.execute(
                    update(editor_documents)
                    .where(
                        editor_documents.c.tenant_id == snapshot.tenant_id,
                        editor_documents.c.id == snapshot.document_id,
                        editor_documents.c.collaboration_generation
                        == snapshot.generation,
                        editor_documents.c.projection_sequence
                        <= snapshot.covered_sequence,
                    )
                    .values(
                        content_markdown=projection_markdown,
                        projection_sequence=snapshot.covered_sequence,
                        projection_updated_at=snapshot.created_at,
                        updated_at=snapshot.created_at,
                    )
                )

    async def update_projection(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        covered_sequence: int,
        content_markdown: str,
        projection_hash: str,
        now: float,
    ) -> CollaborationDocumentState:
        _require_sha256(projection_hash, field="projection_hash")
        if not hmac.compare_digest(
            hashlib.sha256(content_markdown.encode("utf-8")).hexdigest(),
            projection_hash,
        ):
            raise ValueError("projection_hash does not match content_markdown")
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(editor_documents)
                    .where(
                        editor_documents.c.tenant_id == tenant_id,
                        editor_documents.c.id == document_id,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if row is None or row.deleted_at is not None:
                raise CollaborationDocumentNotFound(document_id)
            if (
                row.content_mode != "collaboration"
                or int(row.collaboration_generation) != generation
            ):
                raise CollaborationConflict("generation_conflict")
            if covered_sequence > int(row.persisted_sequence):
                raise CollaborationConflict(
                    "projection_ahead",
                    current_sequence=int(row.persisted_sequence),
                )
            if covered_sequence < int(row.projection_sequence):
                return _document_state(row)
            stored = (
                await session.execute(
                    update(editor_documents)
                    .where(
                        editor_documents.c.tenant_id == tenant_id,
                        editor_documents.c.id == document_id,
                    )
                    .values(
                        content_markdown=content_markdown,
                        projection_sequence=covered_sequence,
                        projection_updated_at=now,
                        updated_at=now,
                    )
                    .returning(editor_documents)
                )
            ).one()
        return _document_state(stored)

    async def list_activity(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        before_sequence: int | None,
        author_user_id: uuid.UUID | None,
        change_kind: CollaborationChangeKind | None,
        limit: int,
    ) -> tuple[CollaborationActivity, ...]:
        statement = select(
            editor_collaboration_updates.c.sequence,
            editor_collaboration_updates.c.actor_user_id,
            editor_collaboration_updates.c.actor_kind,
            editor_collaboration_updates.c.change_kind,
            editor_collaboration_updates.c.suggestion_ids,
            editor_collaboration_updates.c.command_id,
            editor_collaboration_updates.c.created_at,
        ).where(
            editor_collaboration_updates.c.tenant_id == tenant_id,
            editor_collaboration_updates.c.document_id == document_id,
            editor_collaboration_updates.c.generation == generation,
        )
        if before_sequence is not None:
            statement = statement.where(
                editor_collaboration_updates.c.sequence < before_sequence
            )
        if author_user_id is not None:
            statement = statement.where(
                editor_collaboration_updates.c.actor_user_id == author_user_id
            )
        if change_kind is not None:
            statement = statement.where(
                editor_collaboration_updates.c.change_kind == change_kind
            )
        statement = statement.order_by(
            editor_collaboration_updates.c.sequence.desc()
        ).limit(max(1, min(limit, 201)))
        async with self._session(tenant_id) as session:
            rows = (await session.execute(statement)).all()
        return tuple(
            CollaborationActivity(
                sequence=int(row.sequence),
                actor_user_id=row.actor_user_id,
                actor_kind=row.actor_kind,
                change_kind=row.change_kind,
                suggestion_ids=tuple(
                    str(item) for item in (row.suggestion_ids or [])
                ),
                command_id=row.command_id,
                created_at=float(row.created_at),
            )
            for row in rows
        )

    async def list_open_patches(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        before: tuple[float, str] | None,
        author_user_id: uuid.UUID | None,
        suggestion_kind: CollaborationSuggestionKind | None,
        limit: int,
    ) -> CollaborationOpenPatchPage:
        bounded_limit = max(1, min(limit, 200))
        statement = select(editor_patches).where(
            editor_patches.c.tenant_id == tenant_id,
            editor_patches.c.document_id == document_id,
            editor_patches.c.collaboration_generation == generation,
            editor_patches.c.status == "pending",
            func.jsonb_array_length(editor_patches.c.suggestion_ids) > 0,
        )
        if before is not None:
            statement = statement.where(
                tuple_(editor_patches.c.created_at, editor_patches.c.patch_id)
                < tuple_(before[0], before[1])
            )
        if author_user_id is not None:
            statement = statement.where(
                editor_patches.c.created_by_user_id == author_user_id
            )
        if suggestion_kind is not None:
            statement = statement.where(
                text(
                    """
                    EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements(
                            CAST(editor_patches.edits AS jsonb)
                        ) AS items(edit)
                        WHERE CASE
                            WHEN editor_patches.source = 'human'
                                THEN items.edit ->> 'kind'
                            WHEN items.edit ->> 'position'
                                IN ('append', 'prepend')
                                OR (
                                    COALESCE(items.edit ->> 'find', '') = ''
                                    AND COALESCE(items.edit ->> 'text', '') <> ''
                                )
                                THEN 'insertion'
                            WHEN COALESCE(items.edit ->> 'find', '') <> ''
                                AND COALESCE(items.edit ->> 'text', '') = ''
                                THEN 'deletion'
                            ELSE 'modification'
                        END = :suggestion_kind
                    )
                    """
                ).bindparams(suggestion_kind=suggestion_kind)
            )
        statement = statement.order_by(
            editor_patches.c.created_at.desc(),
            editor_patches.c.patch_id.desc(),
        ).limit(bounded_limit + 1)
        async with self._session(tenant_id) as session:
            rows = (await session.execute(statement)).mappings().all()
        page_rows = rows[:bounded_limit]
        next_cursor = (
            (
                float(page_rows[-1]["created_at"]),
                str(page_rows[-1]["patch_id"]),
            )
            if len(rows) > bounded_limit and page_rows
            else None
        )
        result: list[CollaborationOpenPatch] = []
        for row in page_rows:
            if row["created_by_user_id"] is None:
                log.error(
                    "Collaboration patch %s has no author identity.",
                    row["patch_id"],
                )
                raise CollaborationConflict("patch_metadata_conflict")
            edits = row["edits"] or []
            exact_edits: tuple[dict[str, Any], ...] | None
            if row["source"] == "human":
                exact_edits = None
                kinds = tuple(
                    dict.fromkeys(
                        str(item.get("kind"))
                        for item in edits
                        if isinstance(item, dict)
                        and item.get("kind")
                        in {"insertion", "deletion", "modification"}
                    )
                )
            else:
                exact_edits = tuple(
                    dict(item) for item in edits if isinstance(item, dict)
                )
                kinds = tuple(
                    dict.fromkeys(_suggestion_kind_for_edit(item) for item in exact_edits)
                )
            if not kinds:
                log.error(
                    "Collaboration patch %s has no suggestion kind.",
                    row["patch_id"],
                )
                raise CollaborationConflict("patch_metadata_conflict")
            if suggestion_kind is not None and suggestion_kind not in kinds:
                log.error(
                    "Collaboration patch %s disagrees with its SQL suggestion kind.",
                    row["patch_id"],
                )
                raise CollaborationConflict("patch_metadata_conflict")
            result.append(
                CollaborationOpenPatch(
                    patch_id=str(row["patch_id"]),
                    author_user_id=row["created_by_user_id"],
                    created_at=float(row["created_at"]),
                    suggestion_ids=tuple(
                        str(item) for item in (row["suggestion_ids"] or [])
                    ),
                    kinds=kinds,
                    exact_edits=exact_edits,
                )
            )
        return CollaborationOpenPatchPage(
            patches=tuple(result),
            next_cursor=next_cursor,
        )

    async def list_open_patch_ids_at_sequence(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        expected_sequence: int,
        limit: int,
    ) -> tuple[str, ...]:
        """Read a bounded all-open selection under an authoritative CAS."""
        if expected_sequence < 0 or limit < 1:
            raise ValueError("expected_sequence and limit must be valid")
        async with self._session(tenant_id) as session:
            document = (
                await session.execute(
                    select(editor_documents)
                    .where(
                        editor_documents.c.tenant_id == tenant_id,
                        editor_documents.c.id == document_id,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if document is None or document.deleted_at is not None:
                raise CollaborationDocumentNotFound(document_id)
            if (
                document.content_mode != "collaboration"
                or int(document.collaboration_generation) != generation
            ):
                raise CollaborationConflict("generation_conflict")
            if int(document.persisted_sequence) != expected_sequence:
                raise CollaborationConflict(
                    "sequence_conflict",
                    current_sequence=int(document.persisted_sequence),
                )
            rows = (
                await session.execute(
                    select(editor_patches.c.patch_id)
                    .where(
                        editor_patches.c.tenant_id == tenant_id,
                        editor_patches.c.document_id == document_id,
                        editor_patches.c.collaboration_generation == generation,
                        editor_patches.c.status == "pending",
                        func.jsonb_array_length(
                            editor_patches.c.suggestion_ids
                        )
                        > 0,
                    )
                    .order_by(
                        editor_patches.c.created_at,
                        editor_patches.c.patch_id,
                    )
                    .limit(limit + 1)
                )
            ).scalars().all()
        if len(rows) > limit:
            raise CollaborationConflict("all_open_too_large")
        return tuple(str(patch_id) for patch_id in rows)

    async def policy_events_after(
        self,
        *,
        tenant_id: str,
        cursor: int,
        limit: int,
    ) -> CollaborationPolicyPage:
        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        bounded_limit = max(1, min(limit, 500))
        async with self._session(tenant_id) as session:
            bounds = (
                await session.execute(
                    select(
                        func.min(user_events.c.id),
                        func.max(user_events.c.id),
                    ).where(user_events.c.tenant_id == tenant_id)
                )
            ).one()
            oldest = int(bounds[0]) if bounds[0] is not None else 0
            current = int(bounds[1]) if bounds[1] is not None else 0
            if cursor > current and cursor != 0:
                return CollaborationPolicyPage(
                    (), current, reset_required=True
                )
            if cursor and oldest and cursor < oldest - 1:
                return CollaborationPolicyPage(
                    (), current, reset_required=True
                )
            rows = (
                await session.execute(
                    select(user_events)
                    .where(
                        user_events.c.tenant_id == tenant_id,
                        user_events.c.id > cursor,
                        user_events.c.resource_type.in_(
                            ("editor_document", "user")
                        ),
                    )
                    .order_by(user_events.c.id)
                    .limit(bounded_limit + 1)
                )
            ).all()
        if len(rows) > bounded_limit:
            return CollaborationPolicyPage((), current, reset_required=True)
        return CollaborationPolicyPage(
            events=tuple(
                CollaborationPolicyEvent(
                    id=int(row.id),
                    target_user_id=row.target_user_id,
                    scope=str(row.scope),
                    resource_type=str(row.resource_type),
                    resource_id=(
                        str(row.resource_id)
                        if row.resource_id is not None
                        else None
                    ),
                )
                for row in rows
            ),
            current_cursor=current,
        )

    async def compact(
        self,
        *,
        tenant_id: str,
        document_id: str,
        generation: int,
        instance_id: str,
        instance_epoch: int,
        now: float,
        payload_retention_seconds: float,
        metadata_retention_seconds: float,
    ) -> tuple[int, int]:
        if payload_retention_seconds < 0 or metadata_retention_seconds <= 0:
            raise ValueError("retention windows must be non-negative")
        async with self._session(tenant_id) as session:
            await _lock_instance_fence(
                session,
                tenant_id=tenant_id,
                instance_id=instance_id,
                instance_epoch=instance_epoch,
                now=now,
            )
            snapshot_sequences = list(
                (
                    await session.execute(
                        select(
                            editor_collaboration_snapshots.c.covered_sequence
                        )
                        .where(
                            editor_collaboration_snapshots.c.tenant_id
                            == tenant_id,
                            editor_collaboration_snapshots.c.document_id
                            == document_id,
                            editor_collaboration_snapshots.c.generation
                            == generation,
                        )
                        .order_by(
                            editor_collaboration_snapshots.c.covered_sequence.desc()
                        )
                    )
                ).scalars()
            )
            if not snapshot_sequences:
                raise CollaborationConflict("snapshot_missing")
            retained_snapshot_sequences = snapshot_sequences[:2]
            covered_sequence = int(retained_snapshot_sequences[-1])
            payload_result = await session.execute(
                update(editor_collaboration_updates)
                .where(
                    editor_collaboration_updates.c.tenant_id == tenant_id,
                    editor_collaboration_updates.c.document_id == document_id,
                    editor_collaboration_updates.c.generation == generation,
                    editor_collaboration_updates.c.sequence <= covered_sequence,
                    editor_collaboration_updates.c.update_bytes.isnot(None),
                    editor_collaboration_updates.c.created_at
                    <= now - payload_retention_seconds,
                )
                .values(update_bytes=None, payload_pruned_at=now)
            )
            metadata_result = await session.execute(
                delete(editor_collaboration_updates).where(
                    editor_collaboration_updates.c.tenant_id == tenant_id,
                    editor_collaboration_updates.c.document_id == document_id,
                    editor_collaboration_updates.c.generation == generation,
                    editor_collaboration_updates.c.sequence <= covered_sequence,
                    editor_collaboration_updates.c.update_bytes.is_(None),
                    editor_collaboration_updates.c.created_at
                    <= now - metadata_retention_seconds,
                )
            )
            if len(snapshot_sequences) > 2:
                await session.execute(
                    delete(editor_collaboration_snapshots).where(
                        editor_collaboration_snapshots.c.tenant_id == tenant_id,
                        editor_collaboration_snapshots.c.document_id == document_id,
                        editor_collaboration_snapshots.c.generation == generation,
                        editor_collaboration_snapshots.c.covered_sequence.in_(
                            snapshot_sequences[2:]
                        ),
                    )
                )
        return (
            int(payload_result.rowcount or 0),
            int(metadata_result.rowcount or 0),
        )

    async def tombstone_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        owner_user_id: uuid.UUID,
        now: float,
    ) -> int:
        async with self._session(tenant_id) as session:
            access = await lock_resource_access(
                session,
                tenant_id=tenant_id,
                actor_user_id=owner_user_id,
                resource_type="editor_document",
                resource_table=editor_documents,
                id_column=editor_documents.c.id,
                resource_id=document_id,
                owner_column=editor_documents.c.created_by_user_id,
                minimum=SharePermission.VIEW,
                restrict_to_workspace_members=self._restrict_to_workspace_members,
                owner_only=True,
            )
            if access is None:
                raise CollaborationDocumentNotFound(document_id)
            row = (
                await session.execute(
                    select(editor_documents)
                    .where(
                        editor_documents.c.tenant_id == tenant_id,
                        editor_documents.c.id == document_id,
                    )
                    .with_for_update()
                )
            ).one()
            if row.content_mode != "collaboration":
                raise CollaborationConflict("mode_conflict")
            if row.deleted_at is not None:
                return int(row.collaboration_generation)
            next_generation = int(row.collaboration_generation) + 1
            recipients = await revoke_resource_shares(
                session,
                tenant_id=tenant_id,
                resource_type="editor_document",
                resource_id=document_id,
                revoked_by_user_id=owner_user_id,
            )
            await session.execute(
                update(editor_collaboration_leases)
                .where(
                    editor_collaboration_leases.c.tenant_id == tenant_id,
                    editor_collaboration_leases.c.document_id == document_id,
                    editor_collaboration_leases.c.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )
            await session.execute(
                update(editor_documents)
                .where(
                    editor_documents.c.tenant_id == tenant_id,
                    editor_documents.c.id == document_id,
                )
                .values(
                    collaboration_generation=next_generation,
                    metadata_revision=int(row.metadata_revision) + 1,
                    deleted_at=now,
                    updated_at=now,
                )
            )
            await append_resource_effects(
                session,
                tenant_id=tenant_id,
                actor_user_id=owner_user_id,
                owner_user_id=owner_user_id,
                action="editor_document.deleted",
                resource_type="editor_document",
                resource_id=document_id,
                scope="editor_documents",
                additional_targets=recipients,
            )
        return next_generation

    async def purge_tombstones(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        instance_epoch: int,
        now: float,
        retention_seconds: float,
    ) -> int:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        async with self._session(tenant_id) as session:
            await _lock_instance_fence(
                session,
                tenant_id=tenant_id,
                instance_id=instance_id,
                instance_epoch=instance_epoch,
                now=now,
            )
            result = await session.execute(
                delete(editor_documents).where(
                    editor_documents.c.tenant_id == tenant_id,
                    editor_documents.c.content_mode == "collaboration",
                    editor_documents.c.deleted_at.isnot(None),
                    editor_documents.c.deleted_at <= now - retention_seconds,
                )
            )
        return int(result.rowcount or 0)
